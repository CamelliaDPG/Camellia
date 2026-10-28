// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  LinearTermSumFactorized.cpp
//  Camellia
//

#include "LinearTermSumFactorized.h"

#include "Basis.h"
#include "BasisCache.h"
#include "CellTopology.h"
#include "DofOrdering.h"
#include "Function.h"
#include "LinearTerm.h"
#include "Var.h"

#include "Intrepid2_PAMatrix.hpp"

// Camellia requires the PAMatrix API from Trilinos' intrepid2-pamatrix branch, in which assemble() is sum-factorized.  (In
// older PAMatrix versions, assemble() was an IntegrationTools-based placeholder, and the sum-factorized method was named
// assemble2(); building against those would silently fall back to the slow path.)
namespace
{
  template<class PAMatrixType>
  constexpr bool hasSumFactorizedPAMatrixAPI = requires(const PAMatrixType &m) { m.leftBasisValues(); };
}
static_assert(hasSumFactorizedPAMatrixAPI<Intrepid2::PAMatrix<Kokkos::DefaultExecutionSpace::device_type,double>>,
              "Camellia requires the PAMatrix API from Trilinos' intrepid2-pamatrix branch (PAMatrix::assemble() sum-factorized)");
#include "Intrepid2_SerendipityBasis.hpp"
#include "Intrepid2_TensorBasis.hpp"
#include "Intrepid2_TransformedBasisValues.hpp"

#include <cstdlib>
#include <map>
#include <vector>

using namespace Camellia;

namespace
{
  using Intrepid2::Data;
  using Intrepid2::TensorData;
  using Intrepid2::TransformedBasisValues;

  //! physical coordinate index for a single first-derivative operator, or -1 for OP_VALUE, or -2 if unsupported
  int derivativeCoordinate(Camellia::EOperator op)
  {
    switch (op)
    {
      case OP_VALUE: return -1;
      case OP_DX: return 0;
      case OP_DY: return 1;
      case OP_DZ: return 2;
      case OP_DI1: case OP_DI2: case OP_DI3: case OP_DI4: case OP_DI5: case OP_DI6: case OP_DI7:
        return int(op) - int(OP_DI1);
      default: return -2;
    }
  }

  //! summands of one LinearTerm that belong to one variable
  struct VarSide
  {
    int varID = -1;
    bool isDerivative = false;              // all summands OP_VALUE (false) or all single first derivatives (true)
    std::vector<FunctionPtr> weights;
    std::vector<int> derivativeCoordinates; // one per summand, when isDerivative
  };

  //! groups the summands of lt by variable; returns false if any variable's summands are unsupported
  bool groupSummandsByVar(LinearTermPtr lt, int spaceDim, std::map<int,VarSide> &sides)
  {
    for (const auto &summand : lt->summands())
    {
      const FunctionPtr &weight = summand.first;
      const VarPtr &var = summand.second;
      if ((weight == Teuchos::null) || (weight->rank() != 0)) return false;
      if (var->rank() != 0) return false;
      const int coordinate = derivativeCoordinate(var->op());
      if ((coordinate == -2) || (coordinate >= spaceDim)) return false;
      auto &side = sides[var->ID()];
      const bool isDerivative = (coordinate >= 0);
      if (side.weights.size() == 0)
      {
        side.varID = var->ID();
        side.isDerivative = isDerivative;
      }
      else if (side.isDerivative != isDerivative)
      {
        return false; // mixed value and derivative summands for one variable
      }
      side.weights.push_back(weight);
      if (isDerivative) side.derivativeCoordinates.push_back(coordinate);
    }
    return true;
  }

  //! true if basis is an Intrepid2 tensor basis whose tensor components are each one-dimensional, numComponents of them;
  //! this is the structure (matching the cubature's 1D point components) that PAMatrix sum factorization requires
  bool hasOneDimensionalTensorComponents(BasisPtr basis, const int numComponents)
  {
    auto intrepid2Basis = basis->intrepid2Basis();
    if (intrepid2Basis == Teuchos::null) return false;
    using IntrepidBasis = Intrepid2::Basis<DeviceType,double,double>;
    // a Serendipity basis is an ordinal-filtered tensor basis; PAMatrix applies the filter to its sum-factorized assembly
    auto serendipityBasis = dynamic_cast<const Intrepid2::SerendipityBasis<IntrepidBasis>*>(intrepid2Basis.get());
    if (serendipityBasis != NULL) intrepid2Basis = serendipityBasis->getUnderlyingBasis();
    auto tensorBasis = dynamic_cast<const Intrepid2::Basis_TensorBasis<IntrepidBasis>*>(intrepid2Basis.get());
    if (tensorBasis == NULL) return numComponents == 1 && intrepid2Basis->getDomainDimension() == 1;
    const auto components = tensorBasis->getTensorBasisComponents();
    if (int(components.size()) != numComponents) return false;
    for (const auto &component : components)
    {
      if (component->getDomainDimension() != 1) return false;
    }
    return true;
  }

  //! basis values must have one family whose tensor components match the points' component point counts
  bool basisValuesMatchPoints(const Intrepid2::BasisValues<double,DeviceType> &values,
                              const Intrepid2::TensorPoints<double,DeviceType> &points)
  {
    if (values.numFamilies() != 1) return false;
    const auto &tensorData = values.tensorData(0);
    if (tensorData.numTensorComponents() != points.numTensorComponents()) return false;
    for (int r=0; r<points.numTensorComponents(); r++)
    {
      if (tensorData.getTensorComponent(r).extent_int(1) != points.componentPointCount(r)) return false;
    }
    return true;
  }

  //! supported volume basis for the variable in this ordering, or null
  BasisPtr supportedVolumeBasis(DofOrderingPtr ordering, int varID)
  {
    if (!ordering->hasBasisEntry(varID, VOLUME_INTERIOR_SIDE_INDEX)) return Teuchos::null;
    if (!ordering->isVolumeVar(varID)) return Teuchos::null;
    BasisPtr basis = ordering->getBasis(varID, VOLUME_INTERIOR_SIDE_INDEX);
    if (dynamic_cast<Intrepid2BasisWrapper<double>*>(basis.get()) == NULL) return Teuchos::null;
    const auto fs = basis->functionSpace();
    if ((fs != Camellia::FUNCTION_SPACE_HGRAD) && (fs != Camellia::FUNCTION_SPACE_HVOL)) return Teuchos::null;
    if (basis->rangeRank() != 0) return Teuchos::null;
    return basis;
  }

  //! the transformed basis values representing sum_i weight_i * op_i(phi), at the cache's points
  TransformedBasisValues<double,DeviceType> transformedValues(const VarSide &side, BasisPtr basis, BasisCachePtr basisCache,
                                                              const Camellia::FieldContainer<double> &jacobianDet,
                                                              const Camellia::FieldContainer<double> &jacobianInv)
  {
    const int numCells  = jacobianDet.dimension(0);
    const int numPoints = jacobianDet.dimension(1);
    // reference (cell) dimension; for space-time caches this exceeds getSpaceDim(), which counts only spatial dimensions
    const int cellDim   = jacobianInv.dimension(2);
    const bool isHVOL = (basis->functionSpace() == Camellia::FUNCTION_SPACE_HVOL);

    // pointwise weights, evaluated once per summand
    std::vector<Camellia::FieldContainer<double>> weightValues(side.weights.size(), Camellia::FieldContainer<double>(numCells, numPoints));
    for (size_t i=0; i<side.weights.size(); i++) side.weights[i]->values(weightValues[i], basisCache);

    if (!side.isDerivative)
    {
      // (C,P) transform: sum_i c_i, divided by det J for HVOL (Intrepid2's HVOL value transform)
      Kokkos::View<double**,DeviceType> transformView("LinearTerm sum-factorized value transform", numCells, numPoints);
      auto transformHost = Kokkos::create_mirror_view(transformView);
      for (int c=0; c<numCells; c++)
      {
        for (int p=0; p<numPoints; p++)
        {
          double sum = 0;
          for (const auto &w : weightValues) sum += w(c,p);
          transformHost(c,p) = isHVOL ? sum / jacobianDet(c,p) : sum;
        }
      }
      Kokkos::deep_copy(transformView, transformHost);
      const auto &referenceValues = basisCache->getValues(basis, Intrepid2::OPERATOR_VALUE, false);
      return TransformedBasisValues<double,DeviceType>(Data<double,DeviceType>(transformView), referenceValues);
    }

    // (C,P,D) transform contracted against reference gradients: sum_i c_i d/dx_{k_i} = sum_d t_d d/dxi_d,
    // t_d = sum_i c_i (J^{-1})_{d,k_i}.  (Derivatives are only supported for HGRAD bases.)
    TEUCHOS_TEST_FOR_EXCEPTION(isHVOL, std::invalid_argument, "derivatives of HVOL bases are not supported");
    Kokkos::View<double***,DeviceType> transformView("LinearTerm sum-factorized gradient transform", numCells, numPoints, cellDim);
    auto transformHost = Kokkos::create_mirror_view(transformView);
    for (int c=0; c<numCells; c++)
    {
      for (int p=0; p<numPoints; p++)
      {
        for (int d=0; d<cellDim; d++)
        {
          double sum = 0;
          for (size_t i=0; i<weightValues.size(); i++) sum += weightValues[i](c,p) * jacobianInv(c,p,d,side.derivativeCoordinates[i]);
          transformHost(c,p,d) = sum;
        }
      }
    }
    Kokkos::deep_copy(transformView, transformHost);
    const auto &referenceGradients = basisCache->getValues(basis, Intrepid2::OPERATOR_GRAD, false);
    return TransformedBasisValues<double,DeviceType>(Data<double,DeviceType>(transformView), referenceGradients);
  }
}

namespace
{
  //! for a side cache on a tensor-product cell: the (d-1)-component side basis values, with a singleton (1 field,
  //! 1 point, value 1) component inserted at singletonComponent, so that they pair with volume-basis values at the side's
  //! points (which have a singleton point component in the side's normal direction)
  bool sideBasisValuesWithSingleton(Intrepid2::BasisValues<double,DeviceType> &result,
                                    const Intrepid2::BasisValues<double,DeviceType> &sideValues,
                                    const int singletonComponent, const int numVolumeComponents)
  {
    if (sideValues.numFamilies() != 1) return false;
    const auto &sideTensorData = sideValues.tensorData(0);
    if (sideTensorData.numTensorComponents() != numVolumeComponents - 1) return false;
    Kokkos::View<double**,DeviceType> singletonView("LinearTerm sum-factorized side singleton", 1, 1);
    Kokkos::deep_copy(singletonView, 1.0);
    std::vector<Data<double,DeviceType>> components;
    int sideComponent = 0;
    for (int r=0; r<numVolumeComponents; r++)
    {
      if (r == singletonComponent) components.push_back(Data<double,DeviceType>(singletonView));
      else components.push_back(sideTensorData.getTensorComponent(sideComponent++));
    }
    result = Intrepid2::BasisValues<double,DeviceType>(TensorData<double,DeviceType>(components));
    // the singleton component has one field, so full tensor-basis ordinals (and hence any Serendipity ordinal filter) are unchanged
    result.setOrdinalFilter(sideValues.ordinalFilter());
    return true;
  }
}

namespace
{
  bool &sumFactorizedLinearTermIntegrationFlag()
  {
    static bool enabled = (getenv("CAMELLIA_DISABLE_SUM_FACTORIZED_LINEAR_TERMS") == NULL);
    return enabled;
  }
}

namespace
{
  //! side (trace) term integration; see integrateLinearTermsSumFactorized
  bool integrateSideLinearTermsSumFactorized(Camellia::FieldContainer<double> &values,
                                             LinearTermPtr u, DofOrderingPtr uOrdering,
                                             LinearTermPtr v, DofOrderingPtr vOrdering,
                                             BasisCachePtr sideCache)
  {
    BasisCachePtr volumeCache = sideCache->getVolumeBasisCache();
    if (volumeCache == Teuchos::null) return false;
    if (!volumeCache->cellTopology()->isHypercube()) return false;
    const int sideIndex = sideCache->getSideIndex();
    // only true sides (codimension 1) of the volume cell; projections can also build caches on lower-dimensional subcells
    if ((sideIndex < 0) || (sideIndex >= int(volumeCache->cellTopology()->getSideCount()))) return false;
    if (sideCache->getRefCellPoints().numTensorComponents() == 0) return false;
    if (sideCache->getRefCellPoints().extent_int(1) != int(volumeCache->cellTopology()->getDimension()) - 1) return false;
    const int cellDim   = volumeCache->cellTopology()->getDimension();

    // side points in volume reference coordinates must keep the tensor structure, with one singleton component
    const auto &volumePoints = sideCache->getSideRefCellPointsInVolumeCoordinates();
    const int numVolumeComponents = volumePoints.numTensorComponents();
    int singletonComponent = -1;
    for (int r=0; r<numVolumeComponents; r++)
    {
      if (volumePoints.getTensorComponent(r).extent_int(0) == 1)
      {
        if (singletonComponent >= 0) return false;
        singletonComponent = r;
      }
    }
    if (singletonComponent < 0) return false;
    // The singleton component must be the coordinate that is fixed on this side, and the other components must vary in
    // the remaining coordinates in order, matching the side basis's components.  Check against the side points mapped to
    // the volume reference cell.
    {
      const auto &sideVolumePointsFC = sideCache->getSideRefCellPointsInVolumeCoordinatesFC(); // (P,D)
      const int numSidePoints = sideVolumePointsFC.dimension(0);
      int fixedCoordinate = -1;
      for (int d=0; d<sideVolumePointsFC.dimension(1); d++)
      {
        bool constant = true;
        for (int p=1; p<numSidePoints; p++) if (sideVolumePointsFC(p,d) != sideVolumePointsFC(0,d)) { constant = false; break; }
        if (constant)
        {
          if (fixedCoordinate >= 0) return false;
          fixedCoordinate = d;
        }
      }
      if (fixedCoordinate != singletonComponent) return false;
      // each tensor component is one-dimensional, so component r corresponds to coordinate r
      for (int r=0; r<numVolumeComponents; r++)
      {
        if (volumePoints.getTensorComponent(r).extent_int(1) != 1) return false;
      }
    }

    std::map<int,VarSide> uSides, vSides;
    if (!groupSummandsByVar(u, cellDim, uSides)) return false;
    if (!groupSummandsByVar(v, cellDim, vSides)) return false;
    for (const auto &entry : uSides) if (entry.second.isDerivative) return false;
    for (const auto &entry : vSides) if (entry.second.isDerivative) return false;

    // one entry per variable: the basis values to use, and the dof indices
    struct SideVar
    {
      BasisPtr basis;
      Intrepid2::BasisValues<double,DeviceType> values;
      std::vector<int> dofIndices;
      Camellia::FieldContainer<double> pointScaling; // (C,P) value transform for side HVOL bases; empty for the identity
    };
    auto setUpVar = [&](DofOrderingPtr ordering, int varID, SideVar &sideVar) -> bool
    {
      const bool volumeVar = ordering->isVolumeVar(varID);
      const int basisSideIndex = volumeVar ? VOLUME_INTERIOR_SIDE_INDEX : sideIndex;
      if (!ordering->hasBasisEntry(varID, basisSideIndex)) return false;
      BasisPtr basis = ordering->getBasis(varID, basisSideIndex);
      if (dynamic_cast<Intrepid2BasisWrapper<double>*>(basis.get()) == NULL) return false;
      if (basis->rangeRank() != 0) return false;
      // volume HVOL bases would need the volume Jacobian determinant, which is not supported here
      if (volumeVar && (basis->functionSpace() == Camellia::FUNCTION_SPACE_HVOL)) return false;
      if (volumeVar)
      {
        if (!hasOneDimensionalTensorComponents(basis, numVolumeComponents)) return false;
        const bool useCubPointsSideRefCell = true;
        sideVar.values = sideCache->getValues(basis, Intrepid2::OPERATOR_VALUE, useCubPointsSideRefCell);
        if (!basisValuesMatchPoints(sideVar.values, volumePoints)) return false;
      }
      else
      {
        if (basis->domainTopology()->getDimension() != volumeCache->cellTopology()->getDimension() - 1) return false;
        if (!hasOneDimensionalTensorComponents(basis, numVolumeComponents - 1)) return false;
        const auto &sideValues = sideCache->getValues(basis, Intrepid2::OPERATOR_VALUE, false);
        if (!basisValuesMatchPoints(sideValues, sideCache->getRefCellPoints())) return false;
        if (!sideBasisValuesWithSingleton(sideVar.values, sideValues, singletonComponent, numVolumeComponents)) return false;
        if (basis->functionSpace() == Camellia::FUNCTION_SPACE_HVOL)
        {
          // Side HVOL (skeleton) bases are scaled pointwise, by a geometric factor that BasisEvaluation determines (e.g.,
          // the normal scaling, or the side Jacobian determinant on some space-time sides).  Recover that factor from the
          // standard transformed values, so that this path matches the standard one exactly.
          auto transformed = sideCache->getTransformedValues(basis, Camellia::OP_VALUE); // (C,F,P)
          const auto &reference = sideValues; // (F,P)
          const int numCellsT = transformed->dimension(0), numFieldsT = transformed->dimension(1), numPointsT = transformed->dimension(2);
          sideVar.pointScaling.resize(numCellsT, numPointsT);
          for (int c=0; c<numCellsT; c++)
          {
            for (int p=0; p<numPointsT; p++)
            {
              int bestField = 0;
              double bestValue = 0;
              for (int f=0; f<numFieldsT; f++)
              {
                const double r = std::abs(reference(f,p));
                if (r > bestValue) { bestValue = r; bestField = f; }
              }
              if (bestValue == 0) return false;
              sideVar.pointScaling(c,p) = (*transformed)(c,bestField,p) / reference(bestField,p);
            }
          }
        }
      }
      sideVar.basis = basis;
      sideVar.dofIndices = ordering->getDofIndices(varID, basisSideIndex);
      return true;
    };

    std::map<int,SideVar> uVars, vVars;
    for (const auto &entry : uSides) if (!setUpVar(uOrdering, entry.first, uVars[entry.first])) return false;
    for (const auto &entry : vSides) if (!setUpVar(vOrdering, entry.first, vVars[entry.first])) return false;

    const auto &weightedMeasures = sideCache->getWeightedMeasures(); // (C,P): side measure * cubature weights
    const int numCells  = weightedMeasures.dimension(0);
    const int numPoints = weightedMeasures.dimension(1);
    Kokkos::View<double**,DeviceType> cellMeasureView("LinearTerm sum-factorized side measures", numCells, numPoints);
    {
      auto cellMeasureHost = Kokkos::create_mirror_view(cellMeasureView);
      for (int c=0; c<numCells; c++) for (int p=0; p<numPoints; p++) cellMeasureHost(c,p) = weightedMeasures(c,p);
      Kokkos::deep_copy(cellMeasureView, cellMeasureHost);
    }
    TensorData<double,DeviceType> cellMeasures(std::vector<Data<double,DeviceType>>{Data<double,DeviceType>(cellMeasureView)});

    // scalar (C,P) transform: sum of summand weights, times the flux parity for FLUX terms
    const auto &parities = volumeCache->getCellSideParities();
    auto transform = [&](const VarSide &side, bool isFlux, const Camellia::FieldContainer<double> &pointScaling) {
      const bool scaled = (pointScaling.size() > 0);
      std::vector<Camellia::FieldContainer<double>> weightValues(side.weights.size(), Camellia::FieldContainer<double>(numCells, numPoints));
      for (size_t i=0; i<side.weights.size(); i++) side.weights[i]->values(weightValues[i], sideCache);
      Kokkos::View<double**,DeviceType> transformView("LinearTerm sum-factorized side value transform", numCells, numPoints);
      auto transformHost = Kokkos::create_mirror_view(transformView);
      for (int c=0; c<numCells; c++)
      {
        const double parity = isFlux ? parities(c,sideIndex) : 1.0;
        for (int p=0; p<numPoints; p++)
        {
          double sum = 0;
          for (const auto &w : weightValues) sum += w(c,p);
          transformHost(c,p) = scaled ? parity * sum * pointScaling(c,p) : parity * sum;
        }
      }
      Kokkos::deep_copy(transformView, transformHost);
      return Data<double,DeviceType>(transformView);
    };

    for (const auto &uEntry : uSides)
    {
      const auto &uVar = uVars[uEntry.first];
      TransformedBasisValues<double,DeviceType> uValues(transform(uEntry.second, u->termType() == FLUX, uVar.pointScaling), uVar.values);
      for (const auto &vEntry : vSides)
      {
        const auto &vVar = vVars[vEntry.first];
        TransformedBasisValues<double,DeviceType> vValues(transform(vEntry.second, v->termType() == FLUX, vVar.pointScaling), vVar.values);
        Intrepid2::PAMatrix<DeviceType,double> paMatrix(uValues, cellMeasures, vValues);
        auto integrals = paMatrix.allocateMatrixStorage();
        paMatrix.assemble(integrals);
        const int uCardinality = uVar.dofIndices.size();
        const int vCardinality = vVar.dofIndices.size();
        Kokkos::View<double***,DeviceType> integralsView("LinearTerm sum-factorized side integrals", numCells, uCardinality, vCardinality);
        Kokkos::parallel_for(Kokkos::MDRangePolicy<typename DeviceType::execution_space,Kokkos::Rank<3>>({0,0,0},{numCells,uCardinality,vCardinality}),
                             KOKKOS_LAMBDA(const int c, const int i, const int j) {
          integralsView(c,i,j) = integrals(c,i,j);
        });
        auto integralsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), integralsView);
        for (int c=0; c<numCells; c++)
          for (int i=0; i<uCardinality; i++)
            for (int j=0; j<vCardinality; j++)
              values(c, uVar.dofIndices[i], vVar.dofIndices[j]) += integralsHost(c,i,j);
      }
    }
    return true;
  }
}

bool Camellia::sumFactorizedLinearTermIntegrationEnabled()
{
  return sumFactorizedLinearTermIntegrationFlag();
}

void Camellia::setSumFactorizedLinearTermIntegrationEnabled(bool enabled)
{
  sumFactorizedLinearTermIntegrationFlag() = enabled;
}

bool Camellia::integrateLinearTermsSumFactorized(Camellia::FieldContainer<double> &values,
                                                       LinearTermPtr u, DofOrderingPtr uOrdering,
                                                       LinearTermPtr v, DofOrderingPtr vOrdering,
                                                       BasisCachePtr basisCache)
{
  if (!sumFactorizedLinearTermIntegrationEnabled()) return false;
  if (basisCache->isSideCache()) return integrateSideLinearTermsSumFactorized(values, u, uOrdering, v, vOrdering, basisCache);
  if ((u->termType() == FLUX) || (v->termType() == FLUX)) return false;
  if (!basisCache->cellTopology()->isHypercube()) return false;
  // derivative operators OP_DX, ... index physical coordinates of the cell (for space-time, spatial ones)
  const int cellDim = basisCache->cellTopology()->getDimension();

  std::map<int,VarSide> uSides, vSides;
  if (!groupSummandsByVar(u, cellDim, uSides)) return false;
  if (!groupSummandsByVar(v, cellDim, vSides)) return false;

  const auto &refPoints = basisCache->getRefCellPoints();
  const int numPointComponents = refPoints.numTensorComponents();
  std::map<int,BasisPtr> uBases, vBases;
  for (const auto &entry : uSides)
  {
    BasisPtr basis = supportedVolumeBasis(uOrdering, entry.first);
    if (basis == Teuchos::null) return false;
    if (entry.second.isDerivative && (basis->functionSpace() != Camellia::FUNCTION_SPACE_HGRAD)) return false;
    if (!hasOneDimensionalTensorComponents(basis, numPointComponents)) return false;
    uBases[entry.first] = basis;
  }
  for (const auto &entry : vSides)
  {
    BasisPtr basis = supportedVolumeBasis(vOrdering, entry.first);
    if (basis == Teuchos::null) return false;
    if (entry.second.isDerivative && (basis->functionSpace() != Camellia::FUNCTION_SPACE_HGRAD)) return false;
    if (!hasOneDimensionalTensorComponents(basis, numPointComponents)) return false;
    vBases[entry.first] = basis;
  }
  for (const auto &entry : uBases) if (!basisValuesMatchPoints(basisCache->getValues(entry.second, Intrepid2::OPERATOR_VALUE, false), refPoints)) return false;
  for (const auto &entry : vBases) if (!basisValuesMatchPoints(basisCache->getValues(entry.second, Intrepid2::OPERATOR_VALUE, false), refPoints)) return false;

  const auto &jacobianDet = basisCache->getJacobianDet(); // (C,P)
  const auto &jacobianInv = basisCache->getJacobianInv(); // (C,P,D,D)
  const int numCells  = jacobianDet.dimension(0);
  const int numPoints = jacobianDet.dimension(1);

  // cell measures: det J * reference cubature weights, (C,P)
  const auto referenceWeights = basisCache->getCubatureWeights();
  Kokkos::View<double**,DeviceType> cellMeasureView("LinearTerm sum-factorized cell measures", numCells, numPoints);
  {
    auto cellMeasureHost = Kokkos::create_mirror_view(cellMeasureView);
    Kokkos::View<double*,DeviceType> referenceWeightView("reference cubature weights", numPoints);
    Kokkos::parallel_for(Kokkos::RangePolicy<typename DeviceType::execution_space>(0,numPoints), KOKKOS_LAMBDA(const int p) {
      referenceWeightView(p) = referenceWeights(p);
    });
    auto referenceWeightHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), referenceWeightView);
    for (int c=0; c<numCells; c++)
      for (int p=0; p<numPoints; p++)
        cellMeasureHost(c,p) = jacobianDet(c,p) * referenceWeightHost(p);
    Kokkos::deep_copy(cellMeasureView, cellMeasureHost);
  }
  TensorData<double,DeviceType> cellMeasures(std::vector<Data<double,DeviceType>>{Data<double,DeviceType>(cellMeasureView)});

  for (const auto &uEntry : uSides)
  {
    auto uValues = transformedValues(uEntry.second, uBases[uEntry.first], basisCache, jacobianDet, jacobianInv);
    const auto &uDofIndices = uOrdering->getDofIndices(uEntry.first, VOLUME_INTERIOR_SIDE_INDEX);
    for (const auto &vEntry : vSides)
    {
      auto vValues = transformedValues(vEntry.second, vBases[vEntry.first], basisCache, jacobianDet, jacobianInv);
      const auto &vDofIndices = vOrdering->getDofIndices(vEntry.first, VOLUME_INTERIOR_SIDE_INDEX);

      Intrepid2::PAMatrix<DeviceType,double> paMatrix(uValues, cellMeasures, vValues);
      auto integrals = paMatrix.allocateMatrixStorage();
      paMatrix.assemble(integrals);

      const int uCardinality = uDofIndices.size();
      const int vCardinality = vDofIndices.size();
      // integrals may be stored with a constant cell dimension (rank-2 underlying view); getEntry handles both
      Kokkos::View<double***,DeviceType> integralsView("LinearTerm sum-factorized integrals", numCells, uCardinality, vCardinality);
      Kokkos::parallel_for(Kokkos::MDRangePolicy<typename DeviceType::execution_space,Kokkos::Rank<3>>({0,0,0},{numCells,uCardinality,vCardinality}),
                           KOKKOS_LAMBDA(const int c, const int i, const int j) {
        integralsView(c,i,j) = integrals(c,i,j);
      });
      auto integralsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), integralsView);
      for (int c=0; c<numCells; c++)
        for (int i=0; i<uCardinality; i++)
          for (int j=0; j<vCardinality; j++)
            values(c, uDofIndices[i], vDofIndices[j]) += integralsHost(c,i,j);
    }
  }
  return true;
}
