#include "Teuchos_UnitTestHarness.hpp"

#include "TensorPAMatrixBridge.h"

#include <Intrepid2_CellGeometryTestUtils.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Intrepid2_HierarchicalBasisFamily.hpp>
#include <Intrepid2_NodalBasisFamily.hpp>

#include "PAMatrixAssembly.hpp"
#include "StandardAssembly.hpp"

#include "Kokkos_Core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Camellia;

namespace
{
  using Scalar = double;
  using DeviceType = Kokkos::DefaultExecutionSpace::device_type;
  using BasisFamily = Intrepid2::HierarchicalBasisFamily<DeviceType,Scalar,Scalar>;
  using NonHGRADBasisFamily = Intrepid2::DerivedNodalBasisFamily<DeviceType,Scalar,Scalar>;

  template<int spaceDim>
  Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType> makeUnitGeometry(const int cellsPerDim)
  {
    Kokkos::Array<Scalar,spaceDim> domainExtents;
    Kokkos::Array<int,spaceDim> gridCellCounts;
    for (int d=0; d<spaceDim; d++)
    {
      domainExtents[d] = 1.0;
      gridCellCounts[d] = cellsPerDim;
    }
    using CellGeometry = Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType>;
    if (spaceDim > 3)
    {
      Kokkos::Array<Scalar,spaceDim> origin;
      for (int d=0; d<spaceDim; d++) origin[d] = 0.0;
      return CellGeometry(origin, domainExtents, gridCellCounts,
                          CellGeometry::NO_SUBDIVISION,
                          CellGeometry::HYPERCUBE_NODE_ORDER_TENSOR);
    }

    auto uniformGeometry = Intrepid2::uniformCartesianMesh<Scalar,spaceDim,DeviceType>(domainExtents, gridCellCounts);
    const bool copyAffineness = false;
    return Intrepid2::getNodalCellGeometry(uniformGeometry, copyAffineness);
  }

  void compareDenseMatrices(const std::vector<Scalar> &actual,
                            const std::vector<Scalar> &expected,
                            const std::string &label,
                            Teuchos::FancyOStream &out, bool &success)
  {
    Scalar maxError = 0.0;
    Scalar maxExpected = 0.0;
    for (size_t i=0; i<actual.size(); i++)
    {
      maxError = std::max(maxError, std::abs(actual[i] - expected[i]));
      maxExpected = std::max(maxExpected, std::abs(expected[i]));
    }
    const Scalar relativeError = maxError / std::max<Scalar>(1.0, maxExpected);
    if ((maxError > 1e-10) && (relativeError > 1e-10))
    {
      out << label << " mismatch: maxAbsError=" << maxError
          << ", maxRelError=" << relativeError << std::endl;
      success = false;
    }
  }

  void addPAMatrixAssemblyToDense(Intrepid2::PAMatrix<DeviceType,Scalar> &paMatrix,
                                  const Scalar weight,
                                  std::vector<Scalar> &dense)
  {
    auto paMatrixData = paMatrix.allocateMatrixStorage();
    paMatrix.assemble(paMatrixData);
    const int cells = paMatrixData.extent_int(0);
    const int rows  = paMatrixData.extent_int(1);
    const int cols  = paMatrixData.extent_int(2);
    if (dense.empty()) dense.assign(static_cast<size_t>(cells * rows * cols), 0.0);

    if (paMatrixData.getUnderlyingViewRank() == 3)
    {
      auto paHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), paMatrixData.template getUnderlyingView<3>());
      for (int c=0; c<cells; c++)
        for (int i=0; i<rows; i++)
          for (int j=0; j<cols; j++)
            dense[static_cast<size_t>((c * rows + i) * cols + j)] += weight * paHost(c,i,j);
    }
    else if (paMatrixData.getUnderlyingViewRank() == 2)
    {
      auto paHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), paMatrixData.template getUnderlyingView<2>());
      for (int c=0; c<cells; c++)
        for (int i=0; i<rows; i++)
          for (int j=0; j<cols; j++)
            dense[static_cast<size_t>((c * rows + i) * cols + j)] += weight * paHost(i,j);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unexpected PAMatrix storage rank in bridge comparison");
    }
  }

  template<class ViewType>
  void addStandardAssemblyToDense(const ViewType &standardMatrix,
                                  const Scalar weight,
                                  std::vector<Scalar> &dense)
  {
    auto standardHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), standardMatrix);
    const int cells = standardHost.extent_int(0);
    const int rows  = standardHost.extent_int(1);
    const int cols  = standardHost.extent_int(2);
    if (dense.empty()) dense.assign(static_cast<size_t>(cells * rows * cols), 0.0);
    for (int c=0; c<cells; c++)
      for (int i=0; i<rows; i++)
        for (int j=0; j<cols; j++)
          dense[static_cast<size_t>((c * rows + i) * cols + j)] += weight * standardHost(c,i,j);
  }

  template<int spaceDim>
  void compareDirectPAMatrixToStandardAssembly(const Intrepid2::EFunctionSpace functionSpace,
                                               const Intrepid2::EOperator op,
                                               const std::string &label,
                                               Teuchos::FancyOStream &out, bool &success,
                                               const int p = 2)
  {
    auto geometry = makeUnitGeometry<spaceDim>(1);
    const int worksetSize = geometry.numCells();

    auto paMatrix = constructPAMatrix<Scalar,NonHGRADBasisFamily,Scalar,spaceDim,DeviceType>(
        geometry,
        p, functionSpace, op,
        p, functionSpace, op);

    double standardFlops = 0.0;
    double jacobianFlops = 0.0;
    auto standardMatrix = performStandardAssembly<Scalar,NonHGRADBasisFamily,Scalar,spaceDim,DeviceType>(
        geometry, worksetSize,
        p, functionSpace, op,
        p, functionSpace, op,
        standardFlops, jacobianFlops);

    std::vector<Scalar> actual;
    std::vector<Scalar> expected;
    addPAMatrixAssemblyToDense(paMatrix, Scalar(1.0), actual);
    addStandardAssemblyToDense(standardMatrix, Scalar(1.0), expected);
    compareDenseMatrices(actual, expected, label, out, success);
  }

  template<int spaceDim>
  Teuchos::RCP<std::vector<Scalar>> cubatureWeightMultipliersForCoefficient(Intrepid2::CellGeometry<Scalar,spaceDim,DeviceType> &geometry,
                                                                            const int p,
                                                                            const TensorFunctionPtr &coefficient)
  {
    if (coefficient == Teuchos::null) return Teuchos::null;

    auto cubature = Intrepid2::DefaultCubatureFactory::create<DeviceType>(geometry.cellTopology(), 2 * p);
    const int numPoints = cubature->getNumPoints();
    Intrepid2::ScalarView<Scalar,DeviceType> cubaturePoints("coefficient reference cubature points", numPoints, spaceDim);
    Intrepid2::ScalarView<double,DeviceType> cubatureWeights("coefficient reference cubature weights", numPoints);
    cubature->getCubature(cubaturePoints, cubatureWeights);

    auto pointsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cubaturePoints);
    auto multipliers = Teuchos::rcp(new std::vector<Scalar>(numPoints, Scalar(1.0)));
    std::vector<double> coord(spaceDim, 0.0);
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      for (int d=0; d<spaceDim; d++) coord[d] = pointsHost(pointOrdinal,d);
      (*multipliers)[pointOrdinal] = static_cast<Scalar>(coefficient->value(coord));
    }
    return multipliers;
  }

  template<int spaceDim>
  void compareBridgeTermToStandardAssembly(const TensorBilinearTerm &term,
                                           const std::string &label,
                                           Teuchos::FancyOStream &out, bool &success,
                                           const int p = 2)
  {
    auto geometry = makeUnitGeometry<spaceDim>(1);
    const int worksetSize = geometry.numCells();

    std::vector<TensorPAMatrixTerm<DeviceType,Scalar>> bridgeTerms;
    Camellia::appendTensorHGRADVolumeTermsToPAMatrix<spaceDim,DeviceType,Scalar>(bridgeTerms, geometry, p, term, label);

    const auto fs1 = Camellia::intrepidFunctionSpaceForTensorDescriptor(term.trialOperator());
    const auto fs2 = Camellia::intrepidFunctionSpaceForTensorDescriptor(term.testOperator());
    const auto op1 = Camellia::pamatrixOperatorForTensorDescriptor(term.trialOperator());
    const auto op2 = Camellia::pamatrixOperatorForTensorDescriptor(term.testOperator());
    auto vectorWeight1 = Camellia::vectorWeightForSingleDerivative<spaceDim,Scalar>(term.trialOperator());
    auto vectorWeight2 = Camellia::vectorWeightForSingleDerivative<spaceDim,Scalar>(term.testOperator());
    auto coefficientMultipliers = cubatureWeightMultipliersForCoefficient<spaceDim>(geometry, p, term.coefficient());

    double standardFlops = 0.0;
    double jacobianFlops = 0.0;
    auto standardMatrix = performStandardAssembly<Scalar,BasisFamily,Scalar,spaceDim,DeviceType>(
        geometry, worksetSize,
        p, fs1, op1, vectorWeight1,
        p, fs2, op2, vectorWeight2,
        standardFlops, jacobianFlops, coefficientMultipliers);

    std::vector<Scalar> actual;
    std::vector<Scalar> expected;
    for (auto &bridgeTerm : bridgeTerms)
    {
      addPAMatrixAssemblyToDense(bridgeTerm.matrix, bridgeTerm.weight, actual);
    }
    addStandardAssemblyToDense(standardMatrix, Scalar(1.0), expected);
    compareDenseMatrices(actual, expected, label, out, success);
  }

  TEUCHOS_UNIT_TEST( TensorPAMatrixBridge, DescriptorMappingAndVectorWeight )
  {
    auto valueDescriptor = TensorComponentOperatorDescriptor::valueOnly(3);
    TEST_EQUALITY(Camellia::pamatrixOperatorForTensorDescriptor(valueDescriptor), Intrepid2::OPERATOR_VALUE);

    auto derivativeDescriptor = TensorComponentOperatorDescriptor::singleDerivative(3, 2);
    TEST_EQUALITY(Camellia::pamatrixOperatorForTensorDescriptor(derivativeDescriptor), Intrepid2::OPERATOR_GRAD);

    auto l2ValueDescriptor = TensorComponentOperatorDescriptor::valueOnly(3, TENSOR_COMPONENT_FUNCTION_SPACE_HVOL);
    TEST_EQUALITY(Camellia::intrepidFunctionSpaceForTensorDescriptor(l2ValueDescriptor), Intrepid2::FUNCTION_SPACE_HVOL);
    TEST_EQUALITY(Camellia::pamatrixOperatorForTensorDescriptor(l2ValueDescriptor), Intrepid2::OPERATOR_VALUE);

    auto vectorWeight = Camellia::vectorWeightForSingleDerivative<3,double>(derivativeDescriptor);
    TEST_ASSERT(vectorWeight != Teuchos::null);
    TEST_FLOATING_EQUALITY((*vectorWeight)[0], 0.0, 1e-15);
    TEST_FLOATING_EQUALITY((*vectorWeight)[1], 0.0, 1e-15);
    TEST_FLOATING_EQUALITY((*vectorWeight)[2], 1.0, 1e-15);

    auto curlDescriptor = TensorComponentOperatorDescriptor::curl(2);
    TEST_EQUALITY(curlDescriptor.functionSpace(), TENSOR_COMPONENT_FUNCTION_SPACE_HCURL);
    TEST_EQUALITY(curlDescriptor.op(0), TENSOR_COMPONENT_OP_CURL);
    TEST_THROW(Camellia::pamatrixOperatorForTensorDescriptor(curlDescriptor), std::invalid_argument);

    auto divDescriptor = TensorComponentOperatorDescriptor::div(2);
    TEST_EQUALITY(divDescriptor.functionSpace(), TENSOR_COMPONENT_FUNCTION_SPACE_HDIV);
    TEST_EQUALITY(divDescriptor.op(0), TENSOR_COMPONENT_OP_DIV);
    TEST_THROW(Camellia::pamatrixOperatorForTensorDescriptor(divDescriptor), std::invalid_argument);
  }

  TEUCHOS_UNIT_TEST( TensorPAMatrixBridge, DecomposesAdditiveSeparableCoefficient )
  {
    auto coefficient = TensorFunction::constant(-0.5) *
                       (TensorFunction::componentCoordinate(0) + TensorFunction::componentCoordinate(1));

    auto addends = Camellia::decomposeTensorCoefficientForPAMatrixBridge<3>(coefficient);
    TEST_EQUALITY(addends.size(), 2);

    TEST_FLOATING_EQUALITY(addends[0]->value(std::vector<double>{2.0, 0.0, 0.0}), -1.0, 1e-15);
    TEST_FLOATING_EQUALITY(addends[1]->value(std::vector<double>{0.0, 4.0, 0.0}), -2.0, 1e-15);
  }

  TEUCHOS_UNIT_TEST( TensorPAMatrixBridge, RejectsCoupledCoefficientProduct )
  {
    auto coefficient = TensorFunction::componentCoordinate(0) * TensorFunction::componentCoordinate(1);
    TEST_THROW(Camellia::decomposeTensorCoefficientForPAMatrixBridge<3>(coefficient), std::invalid_argument);
  }

  TEUCHOS_UNIT_TEST( TensorPAMatrixBridge, RejectsSolutionDependentCoefficient )
  {
    auto coefficient = TensorFunction::withDependencySource(
        TensorFunction::componentCoordinate(0),
        TensorFunctionDependencySource(TENSOR_FUNCTION_SOURCE_NAMED_SOLUTION, "frozen-state"));
    TEST_THROW(Camellia::decomposeTensorCoefficientForPAMatrixBridge<2>(coefficient), std::invalid_argument);
  }

  TEUCHOS_UNIT_TEST( TensorPAMatrixBridge, RejectsBoundaryTerm )
  {
    const int spaceDim = 2;
    auto trialVar = Camellia::makeHGRADTensorFieldVar("f", spaceDim);
    auto testVar  = Camellia::makeHGRADTensorFieldVar("w", spaceDim);
    TensorBilinearTerm boundaryTerm(trialVar,
                                    testVar,
                                    TensorComponentOperatorDescriptor::valueOnly(spaceDim),
                                    TensorComponentOperatorDescriptor::valueOnly(spaceDim),
                                    TensorFunction::constant(1.0),
                                    TENSOR_BILINEAR_TERM_BOUNDARY,
                                    TensorOperatorCapabilitySet::matrixFreeAndPAMatrix());

    auto geometry = makeUnitGeometry<spaceDim>(1);
    TEST_THROW((Camellia::adaptTensorHGRADVolumeTermToPAMatrix<spaceDim,DeviceType,Scalar>(geometry, 1, boundaryTerm, "boundary guard")), std::invalid_argument);
  }

  TEUCHOS_UNIT_TEST( TensorPAMatrixBridge, DirectHCURLValuePAMatrixMatchesStandardAssembly )
  {
    compareDirectPAMatrixToStandardAssembly<2>(Intrepid2::FUNCTION_SPACE_HCURL,
                                               Intrepid2::OPERATOR_VALUE,
                                               "direct HCURL VALUE/VALUE PAMatrix probe",
                                               out, success);
  }

  TEUCHOS_UNIT_TEST( TensorPAMatrixBridge, DirectHDIVValuePAMatrixMatchesStandardAssembly )
  {
    compareDirectPAMatrixToStandardAssembly<2>(Intrepid2::FUNCTION_SPACE_HDIV,
                                               Intrepid2::OPERATOR_VALUE,
                                               "direct HDIV VALUE/VALUE PAMatrix probe",
                                               out, success);
  }

  TEUCHOS_UNIT_TEST( TensorPAMatrixBridge, DirectHVOLValuePAMatrixMatchesStandardAssembly )
  {
    compareDirectPAMatrixToStandardAssembly<2>(Intrepid2::FUNCTION_SPACE_HVOL,
                                               Intrepid2::OPERATOR_VALUE,
                                               "direct HVOL VALUE/VALUE PAMatrix probe",
                                               out, success);
  }

  TEUCHOS_UNIT_TEST( TensorPAMatrixBridge, ConstantVlasovTermMatchesStandardAssembly )
  {
    const int spaceDim = 2;
    auto trialVar = Camellia::makeHGRADTensorFieldVar("f", spaceDim);
    auto testVar  = Camellia::makeHGRADTensorFieldVar("w", spaceDim);
    TensorBilinearTerm vlasovDtTerm(trialVar,
                                     testVar,
                                     TensorComponentOperatorDescriptor::valueOnly(spaceDim),
                                     TensorComponentOperatorDescriptor::singleDerivative(spaceDim, 1),
                                     TensorFunction::constant(-1.0),
                                     TENSOR_BILINEAR_TERM_VOLUME,
                                     TensorOperatorCapabilitySet::matrixFreeAndPAMatrix());

    compareBridgeTermToStandardAssembly<spaceDim>(vlasovDtTerm, "constant Vlasov value/d1 bridge", out, success);
  }

  TEUCHOS_UNIT_TEST( TensorPAMatrixBridge, L2TrialHGRADTestVlasovDtTermMatchesStandardAssembly )
  {
    // Keep this correctness comparison within the existing StandardAssembly
    // reference helper's 1D-3D Jacobian flop-estimate support.  The production
    // Planar MITL target remains the 4-component (x,t,ux,uy) term documented in
    // FRESH-TENSOR-IMPLEMENTATION-PLAN-PART-2.md; this smoke covers the new
    // L2/HVOL-trial + HGRAD-test routing without depending on a 4D dense
    // reference helper.
    const int spaceDim = 3;
    const int timeComponentOrdinal = 1;
    auto trialVar = Camellia::makeHVOLTensorFieldVar("f", spaceDim);
    auto testVar  = Camellia::makeHGRADTensorFieldVar("w", spaceDim);
    TensorBilinearTerm vlasovDtTerm(trialVar,
                                     testVar,
                                     TensorComponentOperatorDescriptor::valueOnly(spaceDim, TENSOR_COMPONENT_FUNCTION_SPACE_HVOL),
                                     TensorComponentOperatorDescriptor::singleDerivative(spaceDim, timeComponentOrdinal),
                                     TensorFunction::constant(1.0),
                                     TENSOR_BILINEAR_TERM_VOLUME,
                                     TensorOperatorCapabilitySet::matrixFreeAndPAMatrix());

    compareBridgeTermToStandardAssembly<spaceDim>(vlasovDtTerm, "L2/HGRAD Vlasov value/dt bridge", out, success);
  }

  TEUCHOS_UNIT_TEST( TensorPAMatrixBridge, CoordinateCoefficientVlasovTermMatchesStandardAssembly )
  {
    const int spaceDim = 2;
    auto trialVar = Camellia::makeHGRADTensorFieldVar("f", spaceDim);
    auto testVar  = Camellia::makeHGRADTensorFieldVar("w", spaceDim);
    TensorBilinearTerm vlasovCoordinateTerm(trialVar,
                                            testVar,
                                            TensorComponentOperatorDescriptor::valueOnly(spaceDim),
                                            TensorComponentOperatorDescriptor::singleDerivative(spaceDim, 1),
                                            TensorFunction::componentCoordinate(0),
                                            TENSOR_BILINEAR_TERM_VOLUME,
                                            TensorOperatorCapabilitySet::matrixFreeAndPAMatrix());

    compareBridgeTermToStandardAssembly<spaceDim>(vlasovCoordinateTerm, "coordinate Vlasov x0 value/d1 bridge", out, success);
  }

  TEUCHOS_UNIT_TEST( TensorPAMatrixBridge, AdditiveSeparableVlasovTermMatchesStandardAssembly )
  {
    const int spaceDim = 3;
    auto trialVar = Camellia::makeHGRADTensorFieldVar("f", spaceDim);
    auto testVar  = Camellia::makeHGRADTensorFieldVar("w", spaceDim);
    auto coefficient = TensorFunction::constant(-0.5) *
                       (TensorFunction::componentCoordinate(0) + TensorFunction::componentCoordinate(1));
    TensorBilinearTerm vlasovLinearTerm(trialVar,
                                        testVar,
                                        TensorComponentOperatorDescriptor::valueOnly(spaceDim),
                                        TensorComponentOperatorDescriptor::singleDerivative(spaceDim, 2),
                                        coefficient,
                                        TENSOR_BILINEAR_TERM_VOLUME,
                                        TensorOperatorCapabilitySet::matrixFreeAndPAMatrix());

    compareBridgeTermToStandardAssembly<spaceDim>(vlasovLinearTerm, "additive separable Vlasov x0+x1 value/d2 bridge", out, success);
  }
}
