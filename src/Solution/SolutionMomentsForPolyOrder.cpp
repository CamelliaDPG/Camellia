//
//  SolutionMomentsForPolyOrder.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 6/3/25.
//

#include "SolutionMomentsForPolyOrder.h"

#include "doubleBasisConstruction.h"
#include "TensorBasis.h"

#include "CamelliaDebugUtility.h"

#include <Intrepid2_SerendipityBasis.hpp>
#include <Intrepid2_TensorBasis.hpp>

using namespace Camellia;

namespace
{
  // private (anonymous-namespaced) class to define a simple basis integration kernel.  This might naturally go in a lambda defined in the SolutionMomentsForPolyOrder constructor, but for such constructs, nvcc has complained about not being able to take the address of the enclosing function.
  class BasisIntegral
  {
    const int numPoints_;
    
    using DeviceType = ::Camellia::DeviceType;
    
    Intrepid2::TensorPoints<double, DeviceType> refPoints_;
    Intrepid2::TensorData<double,DeviceType> cubatureWeights_;
    Camellia::DeviceView<double> basisValues_;
    Camellia::DeviceView<double> extrusionBasisIntegral_;
  public:
    BasisIntegral(Intrepid2::TensorPoints<double, DeviceType> refPoints,
                  Intrepid2::TensorData<double,DeviceType> cubatureWeights,
                  Camellia::DeviceView<double> basisValues,
                  Camellia::DeviceView<double> extrusionBasisIntegral)
    :
    numPoints_(refPoints.extent_int(0)),
    refPoints_(refPoints),
    cubatureWeights_(cubatureWeights),
    basisValues_(basisValues),
    extrusionBasisIntegral_(extrusionBasisIntegral)
    {
      
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int &momentNumber, const int &basisOrdinal) const
    {
      double integralValue = 0.0;
      for (int pointOrdinal=0; pointOrdinal<numPoints_; pointOrdinal++)
      {
        const double &x = refPoints_(pointOrdinal,0);
        double xWeight = 1.0;
        for (int i=0; i<momentNumber; i++)
        {
          xWeight *= x;
        }
        const double &quadratureWeight = cubatureWeights_(pointOrdinal);
        const double &basisValue = basisValues_(basisOrdinal,pointOrdinal);
        integralValue += quadratureWeight * xWeight * basisValue;
      }
      extrusionBasisIntegral_(momentNumber,basisOrdinal) = integralValue / 2.0;
    }
  };

  //! Compute (n choose k)
  KOKKOS_INLINE_FUNCTION
  int choose(const int &n, const int &k)
  {
    int result = 1;
    if (n - k < k)
    {
      return choose(n, n-k);
    }
    else
    {
      for (int i=1; i<=k; i++)
      {
        result *= (n - k + i);
      }
      for (int i=2; i<=k; i++)
      {
        result /= i;
      }
    }
    return result;
  }

  //! Compute (n choose k) a^(n-k) b^k
  KOKKOS_INLINE_FUNCTION
  double binomialCoefficient(const double &a, const double &b, const int &n, const int &k)
  {
    return pow(a,n-k) * pow(b,k) * choose(n,k);
  }
}

SolutionMomentsForPolyOrder::SolutionMomentsForPolyOrder(const int &deltaP,
                                                         const int &maxMomentNumber, SolutionPtr phaseSpaceSoln, VarPtr fieldVar, MeshPtr physicalMesh, int survivingExtrusionDimension)
:
_deltaP(deltaP),
_maxMomentNumber(maxMomentNumber),
_survivingExtrusionDimension(survivingExtrusionDimension),
_basisReconciliation(true) // true: cache results
{
  _soln = phaseSpaceSoln;
  _fieldVar = fieldVar;
  
  // check that physical mesh has the right number of dimensions:
  const int numDimensionsPhaseSpace = phaseSpaceSoln->mesh()->getDimension();
  
  int numSurvivingExtrusionDimensions = (_survivingExtrusionDimension == -1) ? 0 : 1;
  ExtrudedMeshTopology* phaseSpaceExtrudedMeshTopology = dynamic_cast<ExtrudedMeshTopology*>(phaseSpaceSoln->mesh()->getTopology().get());
  _numExtrusionDimensions = phaseSpaceExtrudedMeshTopology->numExtrusionDimensions();
  _numDimensionsPhysicalMesh = numDimensionsPhaseSpace - _numExtrusionDimensions + numSurvivingExtrusionDimensions;
  
  setMesh(physicalMesh);

  // we assume that all elements are isotropic in p
  // we also assume that all elements have the same topology
  // we also assume that dimensions in which integration is performed are orthogonal extrusions
  
  auto sampleElemType = _soln->mesh()->globalDofAssignment()->getSampleElementType();
  auto cellTopo = sampleElemType->cellTopoPtr;
  
  int tensorComponentCount = cellTopo->getTensorialDegree();
  
  _myElemType = _soln->mesh()->globalDofAssignment()->getElementTypeForKey(std::make_pair(cellTopo->getKey(), _deltaP));
  auto trialOrdering = _myElemType->trialOrderPtr;
  
  _phaseSpaceBasis = trialOrdering->getBasis(fieldVar->ID());
  
  _extrusionBases = std::vector<BasisPtr>(_numExtrusionDimensions);
  
  using Intrepid2Basis = Intrepid2::Basis<DeviceType,double,double>;
  
  using SerendipityBasis = Intrepid2::SerendipityBasis<Intrepid2Basis>;
  using TensorBasis = Intrepid2::Basis_TensorBasis<Intrepid2Basis>;
  
  SerendipityBasis* serendipityBasis = dynamic_cast<SerendipityBasis*>(_phaseSpaceBasis->intrepid2Basis().get());
  TensorBasis* tensorBasis;
  
  if (serendipityBasis != NULL)
  {
    // set tensorBasis
    auto underlyingBasis = serendipityBasis->getUnderlyingBasis();
    tensorBasis = dynamic_cast<TensorBasis*>(underlyingBasis.get());
    
    TEUCHOS_TEST_FOR_EXCEPTION(tensorBasis == NULL, std::invalid_argument, "Basis underlying SerendipityBasis is not a TensorBasis");
    
    // set up maps between serendipity ordinals and underlying tensor ordinals
    
    auto ordinalMap = serendipityBasis->ordinalMap();
    auto reverseOrdinalMap = SerendipityOrdinalMap("_tensorToSerendipityOrdinal",tensorBasis->getCardinality());
    
    Kokkos::deep_copy(reverseOrdinalMap, -1);
    
    Kokkos::parallel_for("reverse ordinal map", ordinalMap.extent(0),
    KOKKOS_LAMBDA (const int serendipityOrdinal) {
      auto tensorOrdinal = ordinalMap(serendipityOrdinal);
      reverseOrdinalMap(tensorOrdinal) = serendipityOrdinal;
    });
    
    _serendipityToTensorBasisOrdinal = ordinalMap;
    _tensorToSerendipityOrdinal = reverseOrdinalMap;
  }
  else
  {
    tensorBasis = dynamic_cast<TensorBasis*>(_phaseSpaceBasis->intrepid2Basis().get());
  }
  
  TEUCHOS_TEST_FOR_EXCEPTION(tensorBasis == NULL, std::invalid_argument, "Basis must be either a SerendipityBasis or a TensorBasis");
  
  if (tensorBasis != NULL)
  {
    auto basisComponents = tensorBasis->getTensorBasisComponents();
    
    int extrusionOffset = static_cast<int>(basisComponents.size()) - _numExtrusionDimensions;
    // set _spatialBasis
    {
      auto spatialBasisIntrepid2 = basisComponents[0];
      ::Camellia::EFunctionSpace fs = fromIntrepid2FunctionSpace(spatialBasisIntrepid2->getFunctionSpace());
      
      for (int i=1; i<extrusionOffset; i++)
      {
        auto extrusionIntrepid2 = basisComponents[i];
        spatialBasisIntrepid2 = Teuchos::rcp( new TensorBasis(spatialBasisIntrepid2, extrusionIntrepid2, spatialBasisIntrepid2->getFunctionSpace()) );
      }
      _spatialBasis = Teuchos::rcp(new Intrepid2BasisWrapper<>(spatialBasisIntrepid2, spatialBasisIntrepid2->getDomainDimension(), 0, fs) );
    }
    // set _extrusionBases
    for (int i=extrusionOffset; i<basisComponents.size(); i++)
    {
      auto intrepid2Basis = basisComponents[i];
      ::Camellia::EFunctionSpace fs = fromIntrepid2FunctionSpace(intrepid2Basis->getFunctionSpace());
      _extrusionBases[i-extrusionOffset] = Teuchos::rcp(new Intrepid2BasisWrapper<>(intrepid2Basis, 1, 0, fs) );
      _componentBasisCardinalities[i-extrusionOffset+1] = _extrusionBases[i-extrusionOffset]->getCardinality();
    }
  }
  _componentBasisCardinalities[0] = _spatialBasis->getCardinality();
  
  _extrudedSpatialBasis = _spatialBasis;
  if (_survivingExtrusionDimension != -1)
  {
    const int componentCardinality = _extrusionBases[_survivingExtrusionDimension]->getCardinality();
    _extrudedSpatialBasis = intrepid2TensorBasis(_extrudedSpatialBasis,_extrusionBases[_survivingExtrusionDimension]);
  }
  
  for (int extrusionOrdinal=0; extrusionOrdinal<_numExtrusionDimensions; extrusionOrdinal++)
  {
    const auto &extrusionBasis = _extrusionBases[extrusionOrdinal];
    
    int basisCardinality = extrusionBasis->getCardinality();
    int basisDegree = extrusionBasis->getDegree();
    
    vector<int> basisIntegralDimensions { _maxMomentNumber+1, basisCardinality };
    auto extrusionBasisIntegral = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage( basisIntegralDimensions, "SolutionMomentFunction extrusionBasisIntegral" );
    
    vector<int> quadratureDegree {basisDegree + _maxMomentNumber};
    
    CubatureFactory cubFactory;
    auto cubature = cubFactory.create(CellTopology::line(), quadratureDegree);
    auto refPoints       = cubature->allocateCubaturePoints();
    auto cubatureWeights = cubature->allocateCubatureWeights();
    cubature->getCubature(refPoints, cubatureWeights);
    
    const int numPoints = refPoints.extent_int(0);
    
    // TODO: since we have switched to using Intrepid2 basis, use the refPoints directly (not refPointsFC)
    auto refPointsFC = FieldContainerConversion<double>::fromTensorPoints(refPoints);
    
    FieldContainer<double> valuesFC(basisCardinality, numPoints);
    extrusionBasis->getValues(valuesFC, refPointsFC, OP_VALUE);
    
    auto values = valuesFC.checkoutDeviceView(true); // true: deep copy from host
    
    auto policy = Kokkos::MDRangePolicy<ExecSpaceType,Kokkos::Rank<2>>({0,0},{_maxMomentNumber+1,basisCardinality});
    
    BasisIntegral basisIntegralFunctor(refPoints, cubatureWeights, values, extrusionBasisIntegral);
    
    Kokkos::parallel_for("integrate", policy, basisIntegralFunctor);
    
    valuesFC.checkinDeviceView(false); // false: don't deep copy to host
    
    _extrusionBasisIntegrals[extrusionOrdinal] = extrusionBasisIntegral;
  }
}

const Camellia::DeviceView<double> SolutionMomentsForPolyOrder::basisCoefficients() const
{
  return _spatialBasisCoefficients;
}

BasisPtr SolutionMomentsForPolyOrder::extrudedSpatialBasis()
{
  return _extrudedSpatialBasis;
}

Camellia::DeviceView<double> SolutionMomentsForPolyOrder::getSpatialBasisCoefficients()
{
  return _spatialBasisCoefficients;
}

int SolutionMomentsForPolyOrder::localCellOrdinal(const IndexType &cellID) const
{
  auto matchingEntry = std::lower_bound(_physicalCellIDs.begin(), _physicalCellIDs.end(), cellID);
  if (matchingEntry == _physicalCellIDs.end())
  {
    // cellID not found
    return -1;
  }
  else
  {
    const int cellOrdinal = static_cast<int>(matchingEntry - _physicalCellIDs.begin());
    return cellOrdinal;
  }
}

int SolutionMomentsForPolyOrder::maxMomentNumber() const
{
  return _maxMomentNumber;
}

MeshPtr SolutionMomentsForPolyOrder::mesh()
{
  return _configurationSpaceMesh;
}

MeshPtr SolutionMomentsForPolyOrder::phaseSpaceMesh()
{
  return _soln->mesh();
}

int SolutionMomentsForPolyOrder::numMomentEntries() const
{
  const int numDims = numVelocityDims();
  return MomentIndices::getNumMomentEntries(_maxMomentNumber+1,numDims);
}

int SolutionMomentsForPolyOrder::numVelocityDims() const
{
  const int numSurvivingExtrusionDimensions = (_survivingExtrusionDimension >= 0) ? 1 : 0;
  const int numDims = _numExtrusionDimensions - numSurvivingExtrusionDimensions;
  
  return numDims;
}

const std::vector<IndexType> & SolutionMomentsForPolyOrder::physicalCellIDs() const
{
  return _physicalCellIDs;
}
                               
void SolutionMomentsForPolyOrder::setMesh(MeshPtr configurationSpaceMesh)
{
  _configurationSpaceMesh = configurationSpaceMesh;
  if (_configurationSpaceMesh != Teuchos::null)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_numDimensionsPhysicalMesh != configurationSpaceMesh->getDimension(), std::invalid_argument, "dimension does not match expected");
  }
}

//! recompute all moments.
void SolutionMomentsForPolyOrder::update()
{
  MeshPtr phaseSpaceMesh = _soln->mesh();
  const auto & myPhysicalCells = _configurationSpaceMesh->cellIDsInPartition(); // all physical cells
  const auto & myPhaseSpaceCells = phaseSpaceMesh->cellIDsOfType(_myElemType); // only the phase-space cells that match element type
  
  using Intrepid2Basis   = Intrepid2::Basis<DeviceType,double,double>;
  using SerendipityBasis = Intrepid2::SerendipityBasis<Intrepid2Basis>;
  using OrdinalTypeArray1D = typename SerendipityBasis::OrdinalTypeArray1D;
  
  // when bases are serendipity bases, we will also need to reason in terms of the underlying tensor basis:
  Teuchos::RCP<Intrepid2Basis> extrudedSpatialUnfilteredBasis, phaseSpaceUnfilteredBasis;
  
  // 1. Cast to Serendipity basis
  auto spatialSerendipityBasis    = dynamic_cast<SerendipityBasis*>(_extrudedSpatialBasis->intrepid2Basis().get());
  auto phaseSpaceSerendipityBasis = dynamic_cast<SerendipityBasis*>(_phaseSpaceBasis->intrepid2Basis().get());
  
  OrdinalTypeArray1D spatialSerendipityMap, phaseSpaceSerendipityMap;
  OrdinalTypeArray1D reverseSpatialMap, reversePhaseMap;
  
  // 2. If match, set "unfiltered" to underlying; otherwise set to original
  // 3. If match, construct a reverse map for the filter that indicates which of the tensor indices are retained in Serendipity
  if (spatialSerendipityBasis)
  {
    extrudedSpatialUnfilteredBasis = spatialSerendipityBasis->getUnderlyingBasis();
    spatialSerendipityMap          = spatialSerendipityBasis->ordinalMap();
    reverseSpatialMap = OrdinalTypeArray1D("SolutionMomentsForPolyOrder reverse spatial map", extrudedSpatialUnfilteredBasis->getCardinality());
    Kokkos::deep_copy(reverseSpatialMap, -1);
    Kokkos::parallel_for("SolutionMomentsForPolyOrder: fill reverse spatial map", spatialSerendipityMap.size(),
    KOKKOS_LAMBDA (const int serendipityBasisOrdinal) {
      auto tensorOrdinal = spatialSerendipityMap(serendipityBasisOrdinal);
      reverseSpatialMap(tensorOrdinal) = serendipityBasisOrdinal;
    });
  }
  else
  {
    extrudedSpatialUnfilteredBasis = _extrudedSpatialBasis->intrepid2Basis();
  }
  if (phaseSpaceSerendipityBasis)
  {
    phaseSpaceUnfilteredBasis = phaseSpaceSerendipityBasis->getUnderlyingBasis();
    phaseSpaceSerendipityMap  = phaseSpaceSerendipityBasis->ordinalMap();
    reversePhaseMap = OrdinalTypeArray1D("SolutionMomentsForPolyOrder reverse phase map", phaseSpaceUnfilteredBasis->getCardinality());
    Kokkos::deep_copy(reversePhaseMap, -1);
    Kokkos::parallel_for("SolutionMomentsForPolyOrder: fill reverse phase map", phaseSpaceSerendipityMap.size(),
    KOKKOS_LAMBDA (const int serendipityBasisOrdinal) {
      auto tensorOrdinal = phaseSpaceSerendipityMap(serendipityBasisOrdinal);
      reversePhaseMap(tensorOrdinal) = serendipityBasisOrdinal;
    });
  }
  else
  {
    phaseSpaceUnfilteredBasis = _phaseSpaceBasis->intrepid2Basis();
  }
  
  // 4. In cardinality definitions below, distinguish between full and filtered.
  
  const int numCells = static_cast<int>(myPhysicalCells.size());
  
  const int numCellsPrevious  = _spatialBasisCoefficients.extent_int(1);
  
  const int numSurvivingExtrusionDimensions = (_survivingExtrusionDimension >= 0) ? 1 : 0;
  const int numMomentEntries = MomentIndices::getNumMomentEntries(_maxMomentNumber+1, _numExtrusionDimensions - numSurvivingExtrusionDimensions);
  
  int extrudedSpatialBasisCardinality = _extrudedSpatialBasis->getCardinality();
  
  // the "extruded spatial" basis is the spatial basis, times any extrusion basis not integrated away
  int extrudedSpatialUnfilteredBasisCardinality = extrudedSpatialUnfilteredBasis->getCardinality();
  // copy containers to avoid implicit references to this in lambdas below:
  auto componentBasisCardinalities = _componentBasisCardinalities;
  int survivingExtrusionDimension  = _survivingExtrusionDimension;
  
  const int phaseSpaceUnfilteredBasisCardinality = phaseSpaceUnfilteredBasis->getCardinality();
  const int numUnfilteredPhaseBasisFunctionsPerSpatialBasisFunction = phaseSpaceUnfilteredBasisCardinality / extrudedSpatialUnfilteredBasisCardinality;
  
  // make sure _spatialBasisCoefficients is properly allocated
  if (numCells != numCellsPrevious)
  {
    // reallocate _spatialBasisCoefficients containers
    std::vector<int> spatialBasisCoefficientsDims {numMomentEntries, numCells, extrudedSpatialBasisCardinality};
    
    // check in previous, if any
    if (_spatialBasisCoefficientsHasBeenCheckedOut)
    {
      SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(_spatialBasisCoefficients);
    }
    // check out current
    _spatialBasisCoefficients = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage(spatialBasisCoefficientsDims, "SolutionMomentFunction spatial basis coefficients");
    _spatialBasisCoefficientsHasBeenCheckedOut = true;
  }
  
  // clear values before accumulating
  Kokkos::deep_copy(_spatialBasisCoefficients, 0.0);
  
  auto phaseSpaceMeshTopology = dynamic_cast<const ExtrudedMeshTopology*>(_soln->mesh()->getTopology()->baseMeshTopology());
  TEUCHOS_TEST_FOR_EXCEPTION(phaseSpaceMeshTopology == NULL, std::invalid_argument, "phase space MeshTopology must be an ExtrudedMeshTopology");
  
  Camellia::FieldContainer<double> phaseSpaceBasisCoefficientsFC(_phaseSpaceBasis->getCardinality());
  Camellia::FieldContainer<double> coarsePhysicalCellContributionsFC(extrudedSpatialBasisCardinality,numMomentEntries); // just for the current phase space cellID
  
  std::map<IndexType, FieldContainer<double> > activePhysicalCellContributions;
  
  const int maxMomentNumber   = _maxMomentNumber; // copy for use in lambda below
  const int physicalCellCount = phaseSpaceMeshTopology->physicalMeshTopology()->cellCount();
  auto possiblyExtrudedPhysicalMeshTopology = _configurationSpaceMesh->getTopology();
  for (const auto &phaseSpaceCellID : myPhaseSpaceCells)
  {
    auto coarsePhysicalCellContributions = coarsePhysicalCellContributionsFC.checkoutDeviceView(false); // false: don't deep copy from host
    const double extrusionMeasure = phaseSpaceMeshTopology->extrusionMeasureForPhaseSpaceCell(phaseSpaceCellID, _survivingExtrusionDimension);
    _soln->getBasisCoefficientsForCell(phaseSpaceBasisCoefficientsFC, _fieldVar->ID(), phaseSpaceCellID, STANDARD_SOLUTION, 0);
    auto phaseSpaceBasisCoefficients = phaseSpaceBasisCoefficientsFC.checkoutDeviceView(true); // true: deep copy from host
    
    int numExtrusionDimensions = _numExtrusionDimensions;
    Kokkos::Array<double,Intrepid2::Parameters::MaxTensorComponents> extrusion_v0, extrusion_v1;
    for (int d=0; d<numExtrusionDimensions; d++)
    {
      phaseSpaceMeshTopology->extrusionVerticesForPhaseSpaceCell(extrusion_v0[d], extrusion_v1[d], phaseSpaceCellID, d);
    }
    
    auto extrusionBasisIntegralsArray = _extrusionBasisIntegrals; // avoiding implicit reference to this in lambda below
    Kokkos::Array<bool,Intrepid2::Parameters::MaxTensorComponents> extrusionBasisIsL2; // implies we need to transform
    for (int i=0; i<int(_extrusionBases.size()); i++)
    {
      auto basis = _extrusionBases[i];
      extrusionBasisIsL2[i] = (basis->functionSpace() == Camellia::FUNCTION_SPACE_HVOL);
    }
    using namespace ::Camellia::TensorIndices;
    Kokkos::parallel_for("integrate over phase space", extrudedSpatialUnfilteredBasisCardinality,
    KOKKOS_LAMBDA (const int extrudedSpatialBasisOrdinal) {
      Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> phaseFieldOrdinalComponents;
      int remainingExtrudedSpatialBasisOrdinal = extrudedSpatialBasisOrdinal;
      extractNextComponentOrdinal(phaseFieldOrdinalComponents[0], remainingExtrudedSpatialBasisOrdinal, componentBasisCardinalities[0]);
      if (survivingExtrusionDimension != -1)
      {
        extractNextComponentOrdinal(phaseFieldOrdinalComponents[survivingExtrusionDimension+1],
                                    remainingExtrudedSpatialBasisOrdinal,
                                    componentBasisCardinalities[survivingExtrusionDimension+1]);
      }
      
      int iUpperBound = (survivingExtrusionDimension == 0) || (numExtrusionDimensions < 1) ? 1 : maxMomentNumber+1;
      
      Kokkos::Array<int,3> momentIndex; // 3-component index
      int momentIndex1D = 0; // 1D "tetrahedral" index
      for (int i=0; i<iUpperBound; i++)
      {
        momentIndex[0] = i;
        int jUpperBound = (survivingExtrusionDimension == 1) || (numExtrusionDimensions < 2) ? 1 : maxMomentNumber+1 - i;
        for (int j=0; j<jUpperBound; j++)
        {
          momentIndex[1] = j;
          int kUpperBound = (survivingExtrusionDimension == 1) || (numExtrusionDimensions < 3) ? 1 : maxMomentNumber+1 - i - j;
          for (int k=0; k<kUpperBound; k++)
          {
            momentIndex[2] = k;
            double extrusionIntegralValue = 0.0;
            for (int integratedExtrusionFieldOrdinal = 0; integratedExtrusionFieldOrdinal < numUnfilteredPhaseBasisFunctionsPerSpatialBasisFunction; integratedExtrusionFieldOrdinal++)
            {
              int remainingIntegratedExtrusionFieldOrdinal = integratedExtrusionFieldOrdinal;
              for (int d=0; d<numExtrusionDimensions; d++)
              {
                if (survivingExtrusionDimension != d)
                {
                  extractNextComponentOrdinal(phaseFieldOrdinalComponents[d+1], remainingIntegratedExtrusionFieldOrdinal, componentBasisCardinalities[d+1]);
                }
              }
              
              int phaseFieldTensorOrdinal;
              setTensorOrdinal(phaseFieldTensorOrdinal, phaseFieldOrdinalComponents, componentBasisCardinalities, numExtrusionDimensions+1);
              int phaseFieldOrdinal;
              if (reversePhaseMap.size() == 0)
              {
                phaseFieldOrdinal = phaseFieldTensorOrdinal;
              }
              else
              {
                phaseFieldOrdinal = reversePhaseMap(phaseFieldTensorOrdinal);
              }
              if (phaseFieldOrdinal < 0)
              {
                continue;
              }
              
              const double &phaseFieldCoefficient = phaseSpaceBasisCoefficients(phaseFieldOrdinal);
              double extrusionComponentProduct = phaseFieldCoefficient;
              for (int d=0; d<numExtrusionDimensions; d++)
              {
                if (survivingExtrusionDimension == d) continue; // this dimension remains un-integrated; skip
                int momentNumber = momentIndex[d];
                const auto & extrusionBasisIntegrals = extrusionBasisIntegralsArray[d];
                const int &extrusionComponentFieldOrdinal = phaseFieldOrdinalComponents[d+1];
                
                const double &v0 = extrusion_v0[d];
                const double &v1 = extrusion_v1[d];
                
                const double a = v0 + v1;
                const double b = v1 - v0;
                
                const double piolaWeight = extrusionBasisIsL2[d] ? 2.0 / b : 1.0;
                
                if (momentNumber == 0)
                {
                  const double & extrusionBasisIntegral = extrusionBasisIntegrals(momentNumber,extrusionComponentFieldOrdinal);
                  extrusionComponentProduct *= extrusionBasisIntegral * piolaWeight;
                }
                else
                {
                  double binomialSum = 0.0;
                  
                  for (int l=0; l<=momentNumber; l++)
                  {
                    const double & extrusionBasisIntegral = extrusionBasisIntegrals(l,extrusionComponentFieldOrdinal);
                    const double binomialWeight = binomialCoefficient(a, b, momentNumber, l);
                    binomialSum += binomialWeight * extrusionBasisIntegral;
                  }
                  extrusionComponentProduct *= pow(0.5,momentNumber) * binomialSum * piolaWeight;
                }
              }
              extrusionIntegralValue += extrusionComponentProduct;
            }
            extrusionIntegralValue *= extrusionMeasure;
            
            if (reverseSpatialMap.size() == 0)
            {
              coarsePhysicalCellContributions(extrudedSpatialBasisOrdinal,momentIndex1D) = extrusionIntegralValue;
            }
            else
            {
              auto filteredSpatialBasisOrdinal = reverseSpatialMap(extrudedSpatialBasisOrdinal);
              if (filteredSpatialBasisOrdinal >= 0)
              {
                coarsePhysicalCellContributions(filteredSpatialBasisOrdinal,momentIndex1D) = extrusionIntegralValue;
              }
            }
            momentIndex1D++;
          }
        }
      }
    });
    
    coarsePhysicalCellContributionsFC.checkinDeviceView(true); // true:  do deep copy back to host
    phaseSpaceBasisCoefficientsFC.checkinDeviceView(false);    // false: don't deep copy back to host
    
    IndexType possiblyExtrudedPhysicalCellID; // extruded, if there is an extrusion dimension in which we do not integrate
    
    {
      const IndexType physicalCellID = phaseSpaceMeshTopology->physicalCellIDForPhaseSpaceCellID(phaseSpaceCellID);
      if (survivingExtrusionDimension == -1)
      {
        possiblyExtrudedPhysicalCellID = physicalCellID;
      }
      else
      {
        IndexType extrusionCellID = phaseSpaceMeshTopology->extrusionCellIDForPhaseSpaceCellID(phaseSpaceCellID, survivingExtrusionDimension);
        possiblyExtrudedPhysicalCellID = SubExtrudedMeshTopology::getCompositeCellIndex(physicalCellID, extrusionCellID);
      }
    }
    
    CellPtr physicalCell = possiblyExtrudedPhysicalMeshTopology->getCell(possiblyExtrudedPhysicalCellID);
    
    // physical cell matches the geometry of phase space cell in physical dimension(s), but it may not be active in the physical mesh;
    // it may have been refined.
    const auto physicalCellActiveDescendants = physicalCell->getDescendants(possiblyExtrudedPhysicalMeshTopology, true); // true: leaf nodes only
    
    for (const auto & activePhysicalCellID : physicalCellActiveDescendants)
    {
      auto activePhysicalCellContributionEntry = activePhysicalCellContributions.find(activePhysicalCellID);
      if (activePhysicalCellContributionEntry == activePhysicalCellContributions.end())
      {
        // allocate (will be zero-filled)
        activePhysicalCellContributions[activePhysicalCellID] = FieldContainer<double>(extrudedSpatialBasisCardinality, numMomentEntries);
      }
      auto &activePhysicalCellContributionFC = activePhysicalCellContributions[activePhysicalCellID];
      CellPtr activePhysicalCell = possiblyExtrudedPhysicalMeshTopology->getCell(activePhysicalCellID);
      
      SubBasisReconciliationWeights trueSpatialBasisWeights; // weights to relate the phase space basis's physical component to the possibly refined physical cell's basis
      SubBasisReconciliationWeights extrusionBasisWeights;   // weights for the surviving extrusion basis, if any
      if (survivingExtrusionDimension == -1)
      {
        // physical mesh really is just a physical mesh
        RefinementBranch refBranch = activePhysicalCell->refinementBranch(possiblyExtrudedPhysicalCellID);
        trueSpatialBasisWeights    = _basisReconciliation.constrainedWeights(_extrudedSpatialBasis, refBranch, _extrudedSpatialBasis, 0);
        
        if (!trueSpatialBasisWeights.isIdentity)
        {
//              cout << "trueSpatialBasisWeights.weights:\n" << trueSpatialBasisWeights.weights;
          
          const auto & fineOrdinals   = trueSpatialBasisWeights.fineOrdinals;
          const auto & coarseOrdinals = trueSpatialBasisWeights.coarseOrdinals;
          auto fineIt = fineOrdinals.begin();
          for (int fineOrdinalOrdinal = 0; fineOrdinalOrdinal < fineOrdinals.size(); fineOrdinalOrdinal++)
          {
            auto coarseIt = coarseOrdinals.begin();
            for (int coarseOrdinalOrdinal = 0; coarseOrdinalOrdinal < coarseOrdinals.size(); coarseOrdinalOrdinal++)
            {
              const double & weight = trueSpatialBasisWeights.weights(fineOrdinalOrdinal,coarseOrdinalOrdinal);
              for (int momentEntryOrdinal=0; momentEntryOrdinal<numMomentEntries; momentEntryOrdinal++)
              {
                activePhysicalCellContributionFC(*fineIt,momentEntryOrdinal) += weight * coarsePhysicalCellContributionsFC(*coarseIt,momentEntryOrdinal);
              }
              coarseIt++;
            }
            fineIt++;
          }
        }
        else
        {
          for (int basisOrdinal = 0; basisOrdinal < extrudedSpatialBasisCardinality; basisOrdinal++)
          {
            for (int momentEntryOrdinal = 0; momentEntryOrdinal < numMomentEntries; momentEntryOrdinal++)
            {
              activePhysicalCellContributionFC(basisOrdinal,momentEntryOrdinal) += coarsePhysicalCellContributionsFC(basisOrdinal, momentEntryOrdinal);
            }
          }
        }
      }
      else
      {
        // "physical" mesh is a SubExtrudedMeshTopology
        IndexType truePhysicalCellID, extrusionCellID;
        SubExtrudedMeshTopology::componentCellIndices(truePhysicalCellID, extrusionCellID, activePhysicalCellID);
        
        IndexType truePhysicalAncestor, extrusionAncestor;
        SubExtrudedMeshTopology::componentCellIndices(truePhysicalAncestor, extrusionAncestor, possiblyExtrudedPhysicalCellID);
        
        CellPtr trueActivePhysicalCell = phaseSpaceMeshTopology->physicalMeshTopology()->getCell(truePhysicalCellID);
        RefinementBranch physicalRefBranch = trueActivePhysicalCell->refinementBranch(truePhysicalAncestor);
        
        CellPtr activeExtrusionCell = phaseSpaceMeshTopology->extrusionMeshTopology(survivingExtrusionDimension)->getCell(extrusionCellID);
        RefinementBranch extrusionRefBranch = activeExtrusionCell->refinementBranch(extrusionAncestor);
        
        trueSpatialBasisWeights = _basisReconciliation.constrainedWeights(_spatialBasis, physicalRefBranch, _spatialBasis, 0);
        extrusionBasisWeights   = _basisReconciliation.constrainedWeights(_extrusionBases[survivingExtrusionDimension], extrusionRefBranch, _extrusionBases[survivingExtrusionDimension], 0);

        const int spatialCardinality   = _spatialBasis->getCardinality();
        const int extrusionCardinality = _extrusionBases[survivingExtrusionDimension]->getCardinality();
        
        if (!trueSpatialBasisWeights.isIdentity)
        {
          TEUCHOS_TEST_FOR_EXCEPTION(trueSpatialBasisWeights.fineOrdinals.size()   != spatialCardinality, std::logic_error, "weight count should match basis cardinality");
          TEUCHOS_TEST_FOR_EXCEPTION(trueSpatialBasisWeights.coarseOrdinals.size() != spatialCardinality, std::logic_error, "weight count should match basis cardinality");
        }
        if (!extrusionBasisWeights.isIdentity)
        {
          TEUCHOS_TEST_FOR_EXCEPTION(extrusionBasisWeights.fineOrdinals.size()   != extrusionCardinality, std::logic_error, "weight count should match basis cardinality");
          TEUCHOS_TEST_FOR_EXCEPTION(extrusionBasisWeights.coarseOrdinals.size() != extrusionCardinality, std::logic_error, "weight count should match basis cardinality");
        }
        
        for (int spatialFineOrdinal=0; spatialFineOrdinal<spatialCardinality; spatialFineOrdinal++)
        {
          for (int spatialCoarseOrdinal=0; spatialCoarseOrdinal<spatialCardinality; spatialCoarseOrdinal++)
          {
            double spatialWeight;
            if (trueSpatialBasisWeights.isIdentity)
            {
              spatialWeight = (spatialFineOrdinal == spatialCoarseOrdinal) ? 1.0 : 0.0;
            }
            else
            {
              spatialWeight = trueSpatialBasisWeights.weights(spatialFineOrdinal,spatialCoarseOrdinal);
            }
            
            for (int extrusionFineOrdinal=0; extrusionFineOrdinal<extrusionCardinality; extrusionFineOrdinal++)
            {
              int fineTensorOrdinal;
              setTensorOrdinal(fineTensorOrdinal, spatialFineOrdinal, spatialCardinality, extrusionFineOrdinal);
              for (int extrusionCoarseOrdinal=0; extrusionCoarseOrdinal<extrusionCardinality; extrusionCoarseOrdinal++)
              {
                int coarseTensorOrdinal;
                setTensorOrdinal(coarseTensorOrdinal, spatialCoarseOrdinal, spatialCardinality, extrusionCoarseOrdinal);
                
                double extrusionWeight;
                if (extrusionBasisWeights.isIdentity)
                {
                  extrusionWeight = (extrusionFineOrdinal == extrusionCoarseOrdinal) ? 1.0 : 0.0;
                }
                else
                {
                  extrusionWeight = extrusionBasisWeights.weights(extrusionFineOrdinal,extrusionCoarseOrdinal);
                }
                if (extrusionWeight != 0.0)
                {
                  for (int momentEntryOrdinal=0; momentEntryOrdinal < numMomentEntries; momentEntryOrdinal++)
                  {
                    activePhysicalCellContributionFC(fineTensorOrdinal, momentEntryOrdinal) += spatialWeight * extrusionWeight * coarsePhysicalCellContributionsFC(coarseTensorOrdinal, momentEntryOrdinal);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
      
  // send the physical cell contributions to their owners, and store in _spatialBasisCoefficients container
  // we use a convenience method in MPIWrapper for this; note that this sends a bunch of extra data that a manual data-packing implementation would not need to.  (The convenience method requires that we have a vector of fixed-length entries; that's where the inefficiency comes from.)
  using DataType = std::tuple<IndexType,int,int,double>; // (possiblyExtrudedPhysicalCellID,momentEntryOrdinal,spatialBasisOrdinal,integralContribution)
  std::map<int,std::vector<DataType>> dataToSend;
  
  for (const auto & entry : activePhysicalCellContributions)
  {
    const IndexType & physicalCellID = entry.first;
    const FieldContainer<double> &physicalCellContribution = entry.second;
    
    const PartitionIndexType physicalCellOwner = _configurationSpaceMesh->partitionForCellID(physicalCellID);
    TEUCHOS_TEST_FOR_EXCEPTION(physicalCellOwner == -1, std::logic_error, "partition not assigned for physicalCellID -- inactive?");
    
    std::vector<DataType> & ownerDataVector = dataToSend[physicalCellOwner];
    
    for (int basisOrdinal=0; basisOrdinal<physicalCellContribution.dimension(0); basisOrdinal++)
    {
      for (int momentEntryOrdinal=0; momentEntryOrdinal<numMomentEntries; momentEntryOrdinal++)
      {
        const double & value = physicalCellContribution(basisOrdinal,momentEntryOrdinal);
        DataType dataEntry = make_tuple(physicalCellID,momentEntryOrdinal,basisOrdinal,value);
        ownerDataVector.push_back(dataEntry);
      }
    }
  }
  
  std::vector<DataType> dataReceived;
  MPIWrapper::sendDataVectors(_configurationSpaceMesh->Comm(), dataToSend, dataReceived);
  
    // note that the following map logic is not maximally efficient, but it's unlikely to be a bottleneck due to the relatively small size of the data
  using std::map;
  using std::pair;
  map<IndexType, FieldContainer<double> > contributionsToMyPhysicalCells;
  for (const DataType &dataEntry : dataReceived)
  {
    using std::get;
    const IndexType & physicalCellID = get<0>(dataEntry);
    const int &momentEntryOrdinal    = get<1>(dataEntry);
    const int &spatialBasisOrdinal   = get<2>(dataEntry);
    const double &value              = get<3>(dataEntry);
    if (contributionsToMyPhysicalCells.find(physicalCellID) == contributionsToMyPhysicalCells.end())
    {
      contributionsToMyPhysicalCells[physicalCellID] = FieldContainer<double>(numMomentEntries,extrudedSpatialBasisCardinality);
    }
    FieldContainer<double> &physicalCellIntegrals = contributionsToMyPhysicalCells[physicalCellID];
    
    physicalCellIntegrals(momentEntryOrdinal,spatialBasisOrdinal) += value;
  }
  // confirm that we have not received too many entries
  if (int(contributionsToMyPhysicalCells.size()) > numCells)
  {
    Camellia::print("locally owned physical cells", myPhysicalCells);
    std::cout << "contributionsToMyPhysicalCells.size() = " << contributionsToMyPhysicalCells.size() << std::endl;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(int(contributionsToMyPhysicalCells.size()) > numCells, std::invalid_argument, "Received cell count exceeds number of locally owned physical cells");
  
  _physicalCellIDs = std::vector<IndexType>(myPhysicalCells.begin(),myPhysicalCells.end());
  
  auto spatialBasisCoefficientsHost = Kokkos::create_mirror_view(_spatialBasisCoefficients);
  Kokkos::deep_copy(spatialBasisCoefficientsHost,0.0);
  
  int cellOrdinal = 0;
  for (const auto &cellEntry : contributionsToMyPhysicalCells)
  {
    const IndexType &cellID = cellEntry.first;
    const FieldContainer<double> & cellIntegrals = cellEntry.second;
    if (_physicalCellIDs[cellOrdinal] != cellID)
    {
      cellOrdinal = this->localCellOrdinal(cellID);
    }
    TEUCHOS_TEST_FOR_EXCEPTION(_physicalCellIDs[cellOrdinal] != cellID, std::invalid_argument, "mismatch in physical cell ID");
    for (int momentEntryOrdinal=0; momentEntryOrdinal<numMomentEntries; momentEntryOrdinal++)
    {
      for (int spatialBasisOrdinal=0; spatialBasisOrdinal < extrudedSpatialBasisCardinality; spatialBasisOrdinal++)
      {
        spatialBasisCoefficientsHost(momentEntryOrdinal,cellOrdinal,spatialBasisOrdinal) += cellIntegrals(momentEntryOrdinal,spatialBasisOrdinal);
      }
    }
    cellOrdinal++;
  }
  Kokkos::deep_copy(_spatialBasisCoefficients,spatialBasisCoefficientsHost);
}

void SolutionMomentsForPolyOrder::valuesForAllMoments(DeviceView<double> values, BasisCachePtr configurationSpaceBasisCache)
{
  /*
   TODO: During update(), ensure that the physicalCellIDs corresponding to owned phase-space cellIDs are present in _spatialBasisCoefficients.
   (update() is expected/allowed to do MPI communication.  valuesForAllMoments() is not.)
   */
  
  TEUCHOS_FUNC_TIME_MONITOR("SolutionMomentsForPolyOrder::valuesForAllMoments()");
 
  const int numMomentEntries = _spatialBasisCoefficients.extent_int(0);
  TEUCHOS_TEST_FOR_EXCEPTION(numMomentEntries != values.extent_int(2), std::invalid_argument, "values must have shape (C,P,M)");
  
  // this implementation adapted from BasisSumFunction
  
  BasisPtr extrudedSpatialBasis = this->extrudedSpatialBasis();
  
  const int numDofs = extrudedSpatialBasis->getCardinality();
  const int spaceDim = configurationSpaceBasisCache->cellTopology()->getDimension();

  bool basisIsVolumeBasis = extrudedSpatialBasis->domainTopology()->getDimension() == spaceDim;
  bool useCubPointsSideRefCell = basisIsVolumeBasis && configurationSpaceBasisCache->isSideCache();

  constFCPtr transformedValues = configurationSpaceBasisCache->getTransformedValues(extrudedSpatialBasis, OP_VALUE, useCubPointsSideRefCell);
  
  // transformedValues has dimensions (C,F,P)
  // therefore, the rank of the sum is transformedValues->rank() - 3
  TEUCHOS_TEST_FOR_EXCEPTION(3 != values.rank(), std::invalid_argument, "values rank must be 3 (C,P,M).");
  TEUCHOS_TEST_FOR_EXCEPTION(3 != transformedValues->rank(), std::invalid_argument, "transformedValues rank must be 3 (C,F,P).");

  auto dv_transformedValues = transformedValues->checkoutDeviceView();
  
  auto spatialBasisCoefficients  = this->getSpatialBasisCoefficients();
  
  int numCellIDs = configurationSpaceBasisCache->cellIDs().size();
  std::vector<int> physicalCellOrdinalLookupDims {numCellIDs};
  auto physicalCellOrdinalLookup = SharedScratchStorage<IndexType>::sharedScratchStorage().checkoutScratchStorage(physicalCellOrdinalLookupDims, "SolutionMomentFunction physical cell ordinal lookup");
  
  const auto & physicalCellIDs   = this->physicalCellIDs();
  
  auto cellOrdinalLookupHost = Kokkos::create_mirror_view(physicalCellOrdinalLookup);
  int basisCacheCellOrdinal = 0;
  for (const auto &cellID : configurationSpaceBasisCache->cellIDs())
  {
    auto matchingEntry = std::lower_bound(physicalCellIDs.begin(), physicalCellIDs.end(), cellID);
    TEUCHOS_TEST_FOR_EXCEPTION((matchingEntry == physicalCellIDs.end()) || *matchingEntry != cellID,std::invalid_argument,"cellID not found");
    
    int cellOrdinal = matchingEntry - physicalCellIDs.begin();
    
    cellOrdinalLookupHost(basisCacheCellOrdinal) = cellOrdinal;
    basisCacheCellOrdinal++;
  }
  Kokkos::deep_copy(physicalCellOrdinalLookup, cellOrdinalLookupHost);
  
  Kokkos::deep_copy(values, 0.0); // initialize
  using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<3> >::array_index_type;
  Kokkos::parallel_for("SolutionMomentFunction_values", FieldContainer<double>::getMDRangePolicy3(values),
  KOKKOS_LAMBDA (const array_index_type cellOrdinal, const array_index_type pointOrdinal, const array_index_type momentEntryOrdinal)
  {
    const int &cellOrdinalInSpatialBasisCoefficients = physicalCellOrdinalLookup(cellOrdinal);
    for (int dof = 0; dof < numDofs; dof++)
    {
      double weight = spatialBasisCoefficients(momentEntryOrdinal,cellOrdinalInSpatialBasisCoefficients,dof);
      values(cellOrdinal,pointOrdinal,momentEntryOrdinal) += dv_transformedValues.access(cellOrdinal,dof,pointOrdinal) * weight;
    }
  });
  transformedValues->checkinDeviceView(false); // false: don't copy back to host (has not changed)
  SharedScratchStorage<IndexType>::sharedScratchStorage().checkinScratchStorage(physicalCellOrdinalLookup);
}

SolutionMomentsForPolyOrder::~SolutionMomentsForPolyOrder()
{
  for (int d=0; d<_numExtrusionDimensions; d++)
  {
    auto extrusionBasisIntegral = _extrusionBasisIntegrals[d];
    SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(extrusionBasisIntegral);
  }
  if (_spatialBasisCoefficientsHasBeenCheckedOut)
  {
    SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(_spatialBasisCoefficients);
  }
}


// below, we set up unit tests against the private classes and methods above
#include "Teuchos_UnitTestHarness.hpp"

TEUCHOS_UNIT_TEST( SolutionMomentsForPolyOrder_cpp, BinomialSum )
{
  // check that the binomialCoefficient() method works to compute appropriate weights.
  const double a = 0.75;
  const double b = 0.25;

  /*
   Binomial theorem says
   
   (a + b)^n = sum_{k=0..n} ( (n choose k) a^{n-k} b^k )
   
   Since a + b = 1, (a+b)^n = 1 as well.
   
   */
  
  const double tol = 1e-16; // we should loosen this…
  const int maxPower = 10;
  for (int n=0; n<=maxPower; n++)
  {
    double binomialSum = 0.0;
    for (int k=0; k<=n; k++)
    {
      const double binomialTerm = binomialCoefficient(a, b, n, k);
      binomialSum += binomialTerm;
    }
    out << "\nTesting n = " << n << std::endl;
    TEST_FLOATING_EQUALITY(binomialSum, 1.0, tol);
  }
}

TEUCHOS_UNIT_TEST( SolutionMomentsForPolyOrder_cpp, Choose )
{
  // manually compute factorials defining (n choose k)
  // compare to what we compute in the (more efficient) choose() method
  const int maxPower = 10; // small enough that we can use integer arithmetic for everything
  
  int n_factorial = 1;
  for (int n=0; n<=maxPower; n++)
  {
    if (n != 0)  n_factorial *= n;
    
    int k_factorial = 1;
    int n_minus_k_factorial = n_factorial;
    for (int k=0; k<=n; k++)
    {
      if (k != 0)
      {
        k_factorial *= k;

        if (k != n)
        {
          n_minus_k_factorial /= (n - (k - 1));
        }
      }
      int expected = n_factorial / (k_factorial * n_minus_k_factorial);
      int actual = choose(n, k);
      out << "Testing (" << n << " choose " << k << ")\n";
      TEST_EQUALITY(expected, actual);
    }
  }
}
