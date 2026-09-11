#include "TensorOperator.h"
#include "TensorOperatorFactory.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <stdexcept>

#include "Basis.h"
#include "BasisFamily.h"
#include "CamelliaIntrepidExtendedTypes.h"
#include "CellTopology.h"
#include "CubatureFactory.h"
#include "Intrepid2_CubatureDirectLineGauss.hpp"
#include "TensorFunction.h"
#include "TensorCoefficientExpansion.h"
#include "TensorCoefficientMap.h"
#include "TensorBasisCache.h"
#include "TensorLocalSchurComplement.h"
#include "TensorProductLocalOperator.h"
#include "Var.h"
#include "Teuchos_TestForException.hpp"

using namespace Camellia;

namespace
{
  struct LineQuadratureRule
  {
    std::vector<double> points;
    std::vector<double> weights;
  };

  struct BasisData1D
  {
    BasisPtr basis;
    std::vector<double> valuesAtQuadrature;
    std::vector<double> derivativesAtQuadrature;
    int fieldCount = 0;
    int pointCount = 0;
  };

  struct TraceEntityGeometryData
  {
    int orthogonalComponentOrdinal = -1;
    IndexType vertexOrdinal = -1;
    std::vector<int> retainedComponentOrdinals;
    std::vector<int> pointCounts;
    std::vector<std::vector<double>> refPointsByComponent;
    std::vector<std::vector<double>> refWeightsByComponent;
    std::vector<double> measureScales;
    std::vector<double> derivativeScales;
    std::vector<double> fixedCoordinates;
    double orthogonalReferenceCoordinate = 0.0;
    std::vector<double> retainedCellLeft;
    std::vector<double> retainedCellWidth;
    int tensorPointCount = 0;
    TensorFunction::PointView physicalPoints;
    std::vector<double> quadratureWeights;
  };

  void recordTensorOperatorStage(TensorOperatorInstrumentation *instrumentation,
                                 TensorOperatorInstrumentation::Stage stage,
                                 const std::chrono::steady_clock::time_point &start)
  {
    if (instrumentation == nullptr) return;

    const auto end = std::chrono::steady_clock::now();
    instrumentation->add(stage, std::chrono::duration<double>(end - start).count());
  }

  bool tensorVarIsFluxLike(const TensorVarPtr &tensorVar)
  {
    return (tensorVar != Teuchos::null) && tensorVar->isFluxLikeTrace();
  }

  bool tensorVarIsValueTrace(const TensorVarPtr &tensorVar)
  {
    return (tensorVar != Teuchos::null) && tensorVar->isValueTrace();
  }

  bool tensorVarIsTraceLike(const TensorVarPtr &tensorVar)
  {
    return (tensorVar != Teuchos::null) && tensorVar->isTrace();
  }

  std::string joinedVarNames(const std::vector<std::string> &varNames)
  {
    std::string joined;
    for (size_t varOrdinal=0; varOrdinal<varNames.size(); varOrdinal++)
    {
      if (varOrdinal > 0) joined += ",";
      joined += varNames[varOrdinal];
    }
    return joined;
  }

  bool vectorContainsName(const std::vector<std::string> &varNames,
                          const std::string &varName)
  {
    return varNames.end() != std::find(varNames.begin(), varNames.end(), varName);
  }

  bool hasMatchingVolumeTerms(const TensorBilinearFormPtr &bilinearForm,
                              const std::string &trialVarName,
                              const std::string &testVarName)
  {
    for (const auto &term : bilinearForm->terms())
    {
      if (term.trialVar()->name() != trialVarName) continue;
      if (term.testVar()->name()  != testVarName)  continue;
      if (term.isVolumeTerm()) return true;
    }
    return false;
  }

  bool hasMatchingBoundaryTerms(const TensorBilinearFormPtr &bilinearForm,
                                const std::string &trialVarName,
                                const std::string &testVarName)
  {
    for (const auto &term : bilinearForm->terms())
    {
      if (term.trialVar()->name() != trialVarName) continue;
      if (term.testVar()->name()  != testVarName)  continue;
      if (term.isBoundaryTerm()) return true;
    }
    return false;
  }

  LineQuadratureRule lineQuadratureFromCubatureFactory(int exactnessDegree)
  {
    CubatureFactory cubFactory;
    auto cubature = cubFactory.create(CellTopology::line(), exactnessDegree);

    const int pointCount = cubature->getNumPoints();
    auto points = cubature->allocateCubaturePoints();
    auto weights = cubature->allocateCubatureWeights();

    cubature->getCubature(points, weights);

    auto pointsView  = points.getTensorComponent(0);
    auto weightsView = weights.getTensorComponent(0).getUnderlyingView();

    auto pointsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pointsView);
    auto weightsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), weightsView);

    LineQuadratureRule rule;
    rule.points.resize(pointCount);
    rule.weights.resize(pointCount);
    for (int p=0; p<pointCount; p++)
    {
      rule.points[p] = pointsHost(p,0);
      rule.weights[p] = weightsHost(p);
    }
    return rule;
  }

  int quadratureDegreeForComponent(int trialPolyOrder,
                                   int testPolyOrder,
                                   bool trialDerivative,
                                   bool testDerivative)
  {
    int degree = trialPolyOrder + testPolyOrder;
    if (trialDerivative) degree -= 1;
    if (testDerivative) degree -= 1;
    return std::max(1, degree + 1);
  }

  BasisData1D buildBasisDataForComponent(const TensorMeshPtr &tensorMesh,
                                         const TensorVarDiscretizationPtr &discretization,
                                         int componentOrdinal,
                                         bool isTest,
                                         const ComponentCellID1D &cellID,
                                         const std::vector<double> &quadraturePoints)
  {
    BasisData1D basisData;
    auto basisFamily = tensorMesh->effectiveBasisFamily(discretization);
    VarPtr componentVar = discretization->tensorVar()->componentVar(componentOrdinal);
    const int polyOrder = tensorMesh->effectivePolyOrder(discretization, cellID, componentOrdinal, isTest);
    basisData.basis = basisFamily->getBasis(CellTopology::line(), efsForSpace(componentVar->space()), polyOrder);

    basisData.fieldCount = basisData.basis->getCardinality();
    basisData.pointCount = static_cast<int>(quadraturePoints.size());
    basisData.valuesAtQuadrature.assign(basisData.fieldCount * basisData.pointCount, 0.0);
    basisData.derivativesAtQuadrature.assign(basisData.fieldCount * basisData.pointCount, 0.0);

    Camellia::FieldContainer<double> refPointContainer(basisData.pointCount, 1);
    for (int p=0; p<basisData.pointCount; p++)
    {
      refPointContainer(p,0) = quadraturePoints[p];
    }

    Camellia::FieldContainer<double> basisValues(basisData.fieldCount, basisData.pointCount);
    basisData.basis->getValues(basisValues, refPointContainer, OP_VALUE);
    for (int f=0; f<basisData.fieldCount; f++)
    {
      for (int p=0; p<basisData.pointCount; p++)
      {
        basisData.valuesAtQuadrature[f * basisData.pointCount + p] = basisValues(f,p);
      }
    }

    Camellia::FieldContainer<double> basisDerivatives(basisData.fieldCount, basisData.pointCount, 1);
    basisData.basis->getValues(basisDerivatives, refPointContainer, OP_GRAD);
    for (int f=0; f<basisData.fieldCount; f++)
    {
      for (int p=0; p<basisData.pointCount; p++)
      {
        basisData.derivativesAtQuadrature[f * basisData.pointCount + p] = basisDerivatives(f,p,0);
      }
    }

    return basisData;
  }

  double basisEvaluationAtPoint(const std::vector<int> &componentBasisOrdinals,
                                const std::vector<BasisData1D> &basisData,
                                const TensorComponentOperatorDescriptor &descriptor,
                                const std::vector<int> &pointOrdinals,
                                const std::vector<double> &derivativeScales)
  {
    double value = 1.0;
    const int componentCount = descriptor.componentCount();
    for (int d=0; d<componentCount; d++)
    {
      const int basisOrdinal = componentBasisOrdinals[d];
      const int pointOrdinal = pointOrdinals[d];
      const int flatIndex = basisOrdinal * basisData[d].pointCount + pointOrdinal;
      if (descriptor.takesDerivativeInComponent(d))
      {
        value *= basisData[d].derivativesAtQuadrature[flatIndex] * derivativeScales[d];
      }
      else
      {
        value *= basisData[d].valuesAtQuadrature[flatIndex];
      }
    }
    return value;
  }

  double cachedBasisEvaluationAtPoint(const std::vector<int> &componentBasisOrdinals,
                                      const TensorBasisCacheND &basisCache,
                                      const TensorComponentOperatorDescriptor &descriptor,
                                      const std::vector<int> &pointOrdinals)
  {
    double value = 1.0;
    const int componentCount = descriptor.componentCount();
    for (int d=0; d<componentCount; d++)
    {
      const auto &componentCache = basisCache.componentCache(d);
      const int basisOrdinal = componentBasisOrdinals[d];
      const int pointOrdinal = pointOrdinals[d];
      TEUCHOS_TEST_FOR_EXCEPTION((basisOrdinal < 0) || (basisOrdinal >= componentCache.basisCardinality()), std::invalid_argument,
                                 "cachedBasisEvaluationAtPoint basis ordinal out of range.");
      TEUCHOS_TEST_FOR_EXCEPTION((pointOrdinal < 0) || (pointOrdinal >= componentCache.pointCount()), std::invalid_argument,
                                 "cachedBasisEvaluationAtPoint point ordinal out of range.");
      if (descriptor.takesDerivativeInComponent(d))
      {
        value *= componentCache.derivativesAtQuadrature().h_view(basisOrdinal, pointOrdinal) * componentCache.derivativeScale();
      }
      else
      {
        value *= componentCache.valuesAtQuadrature().h_view(basisOrdinal, pointOrdinal);
      }
    }
    return value;
  }

  using CoefficientFactorTuple = TensorCoefficientFactorTuple;
  using CoefficientFactorExpansion = TensorCoefficientFactorExpansion;

  bool termSupportsSumFactorizedVolumeContraction(const TensorBilinearTerm &term,
                                                  CoefficientFactorExpansion &coefficientFactorTerms)
  {
    if (!term.isVolumeTerm()) return false;
    if (tensorVarIsTraceLike(term.trialVar()) || tensorVarIsTraceLike(term.testVar())) return false;

    coefficientFactorTerms.clear();
    if (!appendTensorSeparableCoefficientFactorTerms(term.coefficient(), term.componentCount(), coefficientFactorTerms)) return false;
    return !coefficientFactorTerms.empty();
  }

  std::vector<double> coefficientValuesForComponent(const TensorFunctionPtr &coefficient,
                                                    const TensorBasisCache1D &componentCache,
                                                    int componentCount,
                                                    int componentOrdinal)
  {
    TensorFunction::PointView points("MatrixFreeTensorOperator::componentCoefficientPoints",
                                     componentCache.pointCount(),
                                     componentCount);
    auto pointsHost = Kokkos::create_mirror_view(points);
    for (int pointOrdinal=0; pointOrdinal<componentCache.pointCount(); pointOrdinal++)
    {
      for (int d=0; d<componentCount; d++)
      {
        pointsHost(pointOrdinal,d) = 0.0;
      }
      const double xi = componentCache.referencePoints().h_view(pointOrdinal);
      pointsHost(pointOrdinal,componentOrdinal) = componentCache.cellLeft() + 0.5 * (xi + 1.0) * componentCache.cellWidth();
    }
    Kokkos::deep_copy(points, pointsHost);

    TensorFunction::ValueView values("MatrixFreeTensorOperator::componentCoefficientValues",
                                     componentCache.pointCount());
    coefficient->values(points, values);
    auto valuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);

    std::vector<double> coefficientValues(componentCache.pointCount(), 0.0);
    for (int pointOrdinal=0; pointOrdinal<componentCache.pointCount(); pointOrdinal++)
    {
      coefficientValues[pointOrdinal] = valuesHost(pointOrdinal);
    }
    return coefficientValues;
  }

  double constantCoefficientValue(const TensorFunctionPtr &coefficient,
                                  int componentCount)
  {
    TensorFunction::PointView points("MatrixFreeTensorOperator::constantCoefficientPoint", 1, componentCount);
    auto pointsHost = Kokkos::create_mirror_view(points);
    for (int d=0; d<componentCount; d++) pointsHost(0,d) = 0.0;
    Kokkos::deep_copy(points, pointsHost);

    TensorFunction::ValueView values("MatrixFreeTensorOperator::constantCoefficientValue", 1);
    coefficient->values(points, values);
    auto valuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    return valuesHost(0);
  }

  void applySumFactorizedVolumeTermToCell(const TensorBilinearTerm &term,
                                          const TensorDofLayoutPtr &tensorDofLayout,
                                          const TensorDofLayout::VariableLayout &domainLayout,
                                          const TensorDofLayout::VariableLayout &rangeLayout,
                                          TensorLocalOrdering domainOrdering,
                                          TensorLocalOrdering rangeOrdering,
                                          const TensorBasisCacheND &trialBasisCache,
                                          const TensorBasisCacheND &testBasisCache,
                                          const TensorCoefficientTraversal &trialTraversal,
                                          const TensorCoefficientTraversal &testTraversal,
                                          const std::vector<double> &localInput,
                                          std::vector<double> &localOutput,
                                          TensorOperatorInstrumentation *instrumentation)
  {
    const int componentCount = term.componentCount();
    CoefficientFactorExpansion coefficientFactorTerms;
    TEUCHOS_TEST_FOR_EXCEPTION(!termSupportsSumFactorizedVolumeContraction(term, coefficientFactorTerms), std::invalid_argument,
                               "applySumFactorizedVolumeTermToCell requires a supported separable volume term.");
    if (instrumentation != nullptr)
    {
      instrumentation->recordSumFactorizedVolumeTerm();
    }

    const auto &trialShape = trialTraversal.shape();
    const auto &testShape  = testTraversal.shape();

    std::vector<double> cellInput(trialShape.totalDofs(), 0.0);
    for (IndexType localTrialOrdinal=0; localTrialOrdinal<trialShape.totalDofs(); localTrialOrdinal++)
    {
      cellInput[localTrialOrdinal] = localInput[trialTraversal.globalIndexForLocalOrdinal(localTrialOrdinal, tensorDofLayout)];
    }

    std::vector<double> cellOutput(testShape.totalDofs(), 0.0);
    for (const auto &coefficientFactors : coefficientFactorTerms)
    {
      const auto contractionConstructionStart = std::chrono::steady_clock::now();
      TensorProductLocalOperator localOperator = TensorLocalSchurComplementBuilder::buildVolumeBOperator(term,
                                                                                                        trialShape,
                                                                                                        testShape,
                                                                                                        trialBasisCache,
                                                                                                        testBasisCache,
                                                                                                        coefficientFactors);
      recordTensorOperatorStage(instrumentation,
                                TensorOperatorInstrumentation::CONTRACTION_1D_CONSTRUCTION,
                                contractionConstructionStart);

      const auto localActionStart = std::chrono::steady_clock::now();
      std::vector<double> termCellOutput;
      localOperator.apply(cellInput, termCellOutput);
      for (IndexType localTestOrdinal=0; localTestOrdinal<testShape.totalDofs(); localTestOrdinal++)
      {
        cellOutput[localTestOrdinal] += termCellOutput[localTestOrdinal];
      }
      recordTensorOperatorStage(instrumentation,
                                TensorOperatorInstrumentation::LOCAL_ACTION,
                                localActionStart);
    }

    const auto scatterStart = std::chrono::steady_clock::now();
    for (IndexType localTestOrdinal=0; localTestOrdinal<testShape.totalDofs(); localTestOrdinal++)
    {
      localOutput[testTraversal.globalIndexForLocalOrdinal(localTestOrdinal, tensorDofLayout)] += cellOutput[localTestOrdinal];
    }
    recordTensorOperatorStage(instrumentation,
                              TensorOperatorInstrumentation::SCATTER_LOCAL_OUTPUT_ACCUMULATION,
                              scatterStart);
  }

  bool termIsSupportedTraceAwareBoundaryTerm(const TensorBilinearTerm &term)
  {
    if (!term.isBoundaryTerm()) return false;

    const bool trialTraceLike = tensorVarIsTraceLike(term.trialVar());
    const bool testTraceLike  = tensorVarIsTraceLike(term.testVar());

    if (trialTraceLike == testTraceLike) return false;
    if (!term.trialOperator().isValueOnly()) return false;
    if (!term.testOperator().isValueOnly())  return false;

    return true;
  }

  int supportedBoundaryOrthogonalComponentOrdinal(const TensorBilinearTerm &term)
  {
    if (tensorVarIsTraceLike(term.trialVar())) return term.trialVar()->traceComponentOrdinal();
    return term.testVar()->traceComponentOrdinal();
  }

  void validateSupportedBoundaryTermAgainstCurrentSemantics(const TensorBilinearTerm &term)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(!termIsSupportedTraceAwareBoundaryTerm(term), std::invalid_argument,
                               "MatrixFreeTensorOperator encountered an unsupported trace-aware boundary term. The current implementation supports only a mixed volume/trace boundary subset with no derivative action on either side, using trace-support selection in the distinguished trace component and lower-dimensional basis action in the retained components.");

    const bool trialTraceLike = tensorVarIsTraceLike(term.trialVar());
    const bool testTraceLike  = tensorVarIsTraceLike(term.testVar());
    TensorVarPtr traceVar = trialTraceLike ? term.trialVar() : term.testVar();
    VarPtr traceComponentVar = traceVar->componentVar(traceVar->traceComponentOrdinal());

    if (tensorVarIsValueTrace(traceVar))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(traceComponentVar->space() != HGRAD, std::invalid_argument,
                                 "MatrixFreeTensorOperator value-trace realization currently expects an HGRAD trace component variable.");
    }
    else if (tensorVarIsFluxLike(traceVar))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(traceComponentVar->space() != L2, std::invalid_argument,
                                 "MatrixFreeTensorOperator flux-trace realization currently expects an L2 flux component variable.");
    }

    TEUCHOS_TEST_FOR_EXCEPTION(trialTraceLike && testTraceLike, std::invalid_argument,
                               "MatrixFreeTensorOperator does not yet support trace/trace boundary terms.");
  }

  class MatrixFreeTensorOperator : public TensorOperator
  {
    TensorMeshPtr _tensorMesh;
    TensorDofLayoutPtr _tensorDofLayout;
    TensorBilinearFormPtr _bilinearForm;
    TensorSpaceRole _domainRole;
    TensorSpaceRole _rangeRole;
    std::string _domainVarName;
    std::string _rangeVarName;
    TensorLocalOrdering _domainOrdering;
    TensorLocalOrdering _rangeOrdering;
    TensorGatherScatter _domainGatherScatter;
    TensorGatherScatter _rangeGatherScatter;
    TensorApplyPipelineDescription _pipeline;
    TensorDofLayout::VariableLayout _domainLayout;
    TensorDofLayout::VariableLayout _rangeLayout;
    TensorFactorSpaceShape _ambientDomainShape;
    TensorFactorSpaceShape _ambientRangeShape;
    mutable TensorOperatorInstrumentation _instrumentation;

    void validateInputVector(const std::vector<double> &input) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(input.size()) != domainSize(), std::invalid_argument,
                                 "MatrixFreeTensorOperator::apply input size does not match domain size.");
    }

    void decodeDomainOrdinal(IndexType ordinal,
                             std::vector<int> &componentBasisOrdinals) const
    {
      _tensorDofLayout->decodeLocalDofOrdinal(_domainLayout,
                                              _ambientDomainShape,
                                              _domainOrdering,
                                              ordinal,
                                              componentBasisOrdinals);
    }

    void decodeRangeOrdinal(IndexType ordinal,
                            std::vector<int> &componentBasisOrdinals) const
    {
      _tensorDofLayout->decodeLocalDofOrdinal(_rangeLayout,
                                              _ambientRangeShape,
                                              _rangeOrdering,
                                              ordinal,
                                              componentBasisOrdinals);
    }

    bool componentBasisOrdinalsAreAdmissibleForShape(const std::vector<int> &componentBasisOrdinals,
                                                     const TensorFactorSpaceShape &shape) const
    {
      TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(componentBasisOrdinals.size()) != shape.componentCount(), std::invalid_argument,
                                 "MatrixFreeTensorOperator::componentBasisOrdinalsAreAdmissibleForShape rank mismatch.");
      for (int d=0; d<shape.componentCount(); d++)
      {
        if ((componentBasisOrdinals[d] < 0) || (componentBasisOrdinals[d] >= shape.componentCardinality(d))) return false;
      }
      return true;
    }

    double sideParity(const ComponentMeshTopology1D &componentTopo,
                      const ComponentCellID1D &cellID,
                      IndexType vertexOrdinal) const
    {
      const IndexType leftVertex  = componentTopo.cellVertexIndex(cellID, 0);
      const IndexType rightVertex = componentTopo.cellVertexIndex(cellID, 1);
      if (vertexOrdinal == leftVertex)  return -1.0;
      if (vertexOrdinal == rightVertex) return  1.0;

      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                                 "MatrixFreeTensorOperator::sideParity requires a cell incident on the requested skeleton vertex.");
    }

    TraceEntityGeometryData buildTraceEntityGeometryData(const TensorMeshTopology::TensorSkeletonEntity &entity,
                                                         const TensorBilinearTerm &term,
                                                         const TensorVarDiscretizationPtr &trialDiscretization,
                                                         const TensorVarDiscretizationPtr &testDiscretization) const
    {
      TraceEntityGeometryData geometryData;
      geometryData.orthogonalComponentOrdinal = entity.orthogonalComponentOrdinal();
      geometryData.vertexOrdinal = entity.vertexOrdinal();
      geometryData.retainedComponentOrdinals = _tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(entity.orthogonalComponentOrdinal());

      const int componentCount = _tensorMesh->topology()->componentCount();
      geometryData.pointCounts.assign(componentCount, 1);
      geometryData.refPointsByComponent.resize(componentCount);
      geometryData.refWeightsByComponent.resize(componentCount);
      geometryData.measureScales.assign(componentCount, 1.0);
      geometryData.derivativeScales.assign(componentCount, 1.0);
      geometryData.fixedCoordinates.assign(componentCount, 0.0);
      geometryData.retainedCellLeft.assign(componentCount, 0.0);
      geometryData.retainedCellWidth.assign(componentCount, 1.0);

      auto orthogonalTopo = _tensorMesh->topology()->componentMeshTopology(entity.orthogonalComponentOrdinal());
      geometryData.fixedCoordinates[entity.orthogonalComponentOrdinal()] = orthogonalTopo->activeVertexCoordinates().h_view(entity.vertexOrdinal());
      const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
      const IndexType incidentEnd   = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal() + 1);
      TEUCHOS_TEST_FOR_EXCEPTION(incidentBegin >= incidentEnd, std::logic_error,
                                 "MatrixFreeTensorOperator expected at least one incident orthogonal-component cell for the requested skeleton vertex.");
      const ComponentCellID1D &incidentOrthogonalCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
      const IndexType leftVertex  = orthogonalTopo->cellVertexIndex(incidentOrthogonalCell, 0);
      const IndexType rightVertex = orthogonalTopo->cellVertexIndex(incidentOrthogonalCell, 1);
      geometryData.orthogonalReferenceCoordinate = (entity.vertexOrdinal() == leftVertex) ? -1.0 : 1.0;

      for (size_t retainedOrdinal=0; retainedOrdinal<geometryData.retainedComponentOrdinals.size(); retainedOrdinal++)
      {
        const int d = geometryData.retainedComponentOrdinals[retainedOrdinal];
        const ComponentCellID1D &cellID = entity.retainedCellTuple()[retainedOrdinal];
        const int trialPolyOrder = _tensorMesh->effectivePolyOrder(trialDiscretization, cellID, d, false);
        const int testPolyOrder  = _tensorMesh->effectivePolyOrder(testDiscretization,  cellID, d, true);
        const bool trialDerivative = term.trialOperator().takesDerivativeInComponent(d);
        const bool testDerivative  = term.testOperator().takesDerivativeInComponent(d);
        const int quadratureDegree = quadratureDegreeForComponent(trialPolyOrder, testPolyOrder,
                                                                  trialDerivative, testDerivative);

        LineQuadratureRule quadrature = lineQuadratureFromCubatureFactory(quadratureDegree);
        geometryData.refPointsByComponent[d] = quadrature.points;
        geometryData.refWeightsByComponent[d] = quadrature.weights;
        geometryData.pointCounts[d] = static_cast<int>(quadrature.points.size());

        auto componentTopo = _tensorMesh->topology()->componentMeshTopology(d);
        geometryData.retainedCellLeft[d] = componentTopo->cellLeft(cellID);
        geometryData.retainedCellWidth[d] = componentTopo->cellWidth(cellID);
        geometryData.measureScales[d] = 0.5 * geometryData.retainedCellWidth[d];
        geometryData.derivativeScales[d] = 2.0 / geometryData.retainedCellWidth[d];
      }

      geometryData.tensorPointCount = 1;
      for (int retainedComponentOrdinal : geometryData.retainedComponentOrdinals)
      {
        geometryData.tensorPointCount *= geometryData.pointCounts[retainedComponentOrdinal];
      }

      geometryData.physicalPoints = TensorFunction::PointView("MatrixFreeTensorOperator::tracePhysicalPoints",
                                                              geometryData.tensorPointCount,
                                                              componentCount);
      auto physicalPointsHost = Kokkos::create_mirror_view(geometryData.physicalPoints);
      geometryData.quadratureWeights.assign(geometryData.tensorPointCount, 1.0);

      std::vector<int> retainedPointOrdinals(componentCount, 0);
      for (int flatPointOrdinal=0; flatPointOrdinal<geometryData.tensorPointCount; flatPointOrdinal++)
      {
        int workingOrdinal = flatPointOrdinal;
        double weight = 1.0;

        for (int d=0; d<componentCount; d++)
        {
          if (d == geometryData.orthogonalComponentOrdinal)
          {
            physicalPointsHost(flatPointOrdinal, d) = geometryData.fixedCoordinates[d];
          }
        }

        for (size_t retainedOrdinal=0; retainedOrdinal<geometryData.retainedComponentOrdinals.size(); retainedOrdinal++)
        {
          const int d = geometryData.retainedComponentOrdinals[retainedOrdinal];
          retainedPointOrdinals[d] = workingOrdinal % geometryData.pointCounts[d];
          workingOrdinal /= geometryData.pointCounts[d];

          const double xi = geometryData.refPointsByComponent[d][retainedPointOrdinals[d]];
          const double xPhysical = geometryData.retainedCellLeft[d] + 0.5 * (xi + 1.0) * geometryData.retainedCellWidth[d];
          physicalPointsHost(flatPointOrdinal, d) = xPhysical;
          weight *= geometryData.refWeightsByComponent[d][retainedPointOrdinals[d]] * geometryData.measureScales[d];
        }
        geometryData.quadratureWeights[flatPointOrdinal] = weight;
      }

      Kokkos::deep_copy(geometryData.physicalPoints, physicalPointsHost);
      return geometryData;
    }

    std::vector<BasisData1D> buildBasisDataForBoundaryEntity(const TensorVarDiscretizationPtr &discretization,
                                                             bool isTest,
                                                             const TensorMeshTopology::TensorSkeletonEntity &entity,
                                                             const TraceEntityGeometryData &geometryData) const
    {
      const int componentCount = _tensorMesh->topology()->componentCount();
      std::vector<BasisData1D> basisData(componentCount);

      for (int d=0; d<componentCount; d++)
      {
        if (d == geometryData.orthogonalComponentOrdinal)
        {
          if (!tensorVarIsTraceLike(discretization->tensorVar()))
          {
            auto orthogonalTopo = _tensorMesh->topology()->componentMeshTopology(d);
            const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
            const IndexType incidentEnd   = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal() + 1);
            TEUCHOS_TEST_FOR_EXCEPTION(incidentBegin >= incidentEnd, std::logic_error,
                                       "MatrixFreeTensorOperator expected at least one incident orthogonal-component cell for orthogonal basis evaluation.");
            const ComponentCellID1D &incidentOrthogonalCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
            basisData[d] = buildBasisDataForComponent(_tensorMesh,
                                                      discretization,
                                                      d,
                                                      isTest,
                                                      incidentOrthogonalCell,
                                                      std::vector<double>{geometryData.orthogonalReferenceCoordinate});
          }
          continue;
        }

        IndexType retainedOrdinal = -1;
        for (IndexType r=0; r<static_cast<IndexType>(geometryData.retainedComponentOrdinals.size()); r++)
        {
          if (geometryData.retainedComponentOrdinals[r] == d)
          {
            retainedOrdinal = r;
            break;
          }
        }
        TEUCHOS_TEST_FOR_EXCEPTION(retainedOrdinal < 0, std::logic_error,
                                   "MatrixFreeTensorOperator internal error mapping retained component ordinal.");

        const ComponentCellID1D &cellID = entity.retainedCellTuple()[retainedOrdinal];
        basisData[d] = buildBasisDataForComponent(_tensorMesh,
                                                  discretization,
                                                  d,
                                                  isTest,
                                                  cellID,
                                                  geometryData.refPointsByComponent[d]);
      }
      return basisData;
    }

    double traceAwareBasisEvaluationAtPoint(const TensorVarPtr &tensorVar,
                                            const std::vector<int> &componentBasisOrdinals,
                                            const std::vector<BasisData1D> &basisData,
                                            const TensorComponentOperatorDescriptor &descriptor,
                                            const TraceEntityGeometryData &geometryData,
                                            const std::vector<int> &pointOrdinals,
                                            const TensorMeshTopology::TensorSkeletonEntity &entity) const
    {
      (void) entity;
      double value = 1.0;
      const int componentCount = descriptor.componentCount();

      for (int d=0; d<componentCount; d++)
      {
        if (tensorVarIsTraceLike(tensorVar) && (d == tensorVar->traceComponentOrdinal()))
        {
          value *= (componentBasisOrdinals[d] == 0) ? 1.0 : 0.0;
          continue;
        }

        const int basisOrdinal = componentBasisOrdinals[d];
        const int pointOrdinal = (d == geometryData.orthogonalComponentOrdinal) ? 0 : pointOrdinals[d];
        const int flatIndex = basisOrdinal * basisData[d].pointCount + pointOrdinal;

        if (descriptor.takesDerivativeInComponent(d))
        {
          value *= basisData[d].derivativesAtQuadrature[flatIndex] * geometryData.derivativeScales[d];
        }
        else
        {
          value *= basisData[d].valuesAtQuadrature[flatIndex];
        }
      }

      return value;
    }

    TensorCoefficientTraversal domainTraversalForCell(const TensorMeshTopology::TensorCellTuple &cellTuple) const
    {
      return _domainGatherScatter.coefficientMap().volumeTraversal(cellTuple);
    }

    TensorCoefficientTraversal rangeTraversalForCell(const TensorMeshTopology::TensorCellTuple &cellTuple) const
    {
      return _rangeGatherScatter.coefficientMap().volumeTraversal(cellTuple);
    }

    TensorCoefficientTraversal domainTraversalForSkeletonEntity(const TensorMeshTopology::TensorSkeletonEntity &entity) const
    {
      auto domainVar = _tensorMesh->tensorVarFactory()->tensorVar(_domainVarName);
      if (tensorVarIsTraceLike(domainVar))
      {
        return _domainGatherScatter.coefficientMap().skeletonTraversal(entity);
      }

      const int componentCount = _tensorMesh->topology()->componentCount();
      std::vector<ComponentCellID1D> componentCells(componentCount);
      auto orthogonalTopo = _tensorMesh->topology()->componentMeshTopology(entity.orthogonalComponentOrdinal());
      const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
      TEUCHOS_TEST_FOR_EXCEPTION(incidentBegin >= orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal() + 1), std::logic_error,
                                 "MatrixFreeTensorOperator expected at least one incident orthogonal-component cell for domain coefficient lookup.");
      const ComponentCellID1D &incidentCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
      for (int d=0; d<componentCount; d++) componentCells[d] = incidentCell;
      const auto retainedOrdinals = _tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(entity.orthogonalComponentOrdinal());
      for (size_t r=0; r<retainedOrdinals.size(); r++) componentCells[retainedOrdinals[r]] = entity.retainedCellTuple()[r];
      return _domainGatherScatter.coefficientMap().volumeTraversal(componentCells);
    }

    TensorCoefficientTraversal rangeTraversalForSkeletonEntity(const TensorMeshTopology::TensorSkeletonEntity &entity) const
    {
      auto rangeVar = _tensorMesh->tensorVarFactory()->tensorVar(_rangeVarName);
      if (tensorVarIsTraceLike(rangeVar))
      {
        return _rangeGatherScatter.coefficientMap().skeletonTraversal(entity);
      }

      const int componentCount = _tensorMesh->topology()->componentCount();
      std::vector<ComponentCellID1D> componentCells(componentCount);
      auto orthogonalTopo = _tensorMesh->topology()->componentMeshTopology(entity.orthogonalComponentOrdinal());
      const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
      TEUCHOS_TEST_FOR_EXCEPTION(incidentBegin >= orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal() + 1), std::logic_error,
                                 "MatrixFreeTensorOperator expected at least one incident orthogonal-component cell for range coefficient lookup.");
      const ComponentCellID1D &incidentCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
      for (int d=0; d<componentCount; d++) componentCells[d] = incidentCell;
      const auto retainedOrdinals = _tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(entity.orthogonalComponentOrdinal());
      for (size_t r=0; r<retainedOrdinals.size(); r++) componentCells[retainedOrdinals[r]] = entity.retainedCellTuple()[r];
      return _rangeGatherScatter.coefficientMap().volumeTraversal(componentCells);
    }

    void applyVolumeTerm(const TensorBilinearTerm &term,
                         const std::vector<double> &localInput,
                         std::vector<double> &localOutput) const
    {
      auto trialDiscretization = _tensorMesh->trialDiscretization(_domainVarName);
      auto testDiscretization  = _tensorMesh->testDiscretization(_rangeVarName);
      const auto &activeCellTuples = _tensorMesh->topology()->activeCellTuples();
      const int componentCount = term.componentCount();

      std::vector<int> trialBasisOrdinals(componentCount, 0);
      std::vector<int> testBasisOrdinals(componentCount, 0);
      std::vector<int> pointOrdinals(componentCount, 0);

      for (const auto &cellTuple : activeCellTuples)
      {
        auto localContext = _tensorDofLayout->volumeContext(_domainVarName, _rangeVarName, cellTuple);
        const auto &trialShape = localContext.trialShape();
        const auto &testShape  = localContext.testShape();
        const auto trialTraversal = domainTraversalForCell(cellTuple);
        const auto testTraversal  = rangeTraversalForCell(cellTuple);

        std::vector<int> quadratureDegrees(componentCount, 1);
        for (int d=0; d<componentCount; d++)
        {
          const int trialPolyOrder = _tensorMesh->effectivePolyOrder(trialDiscretization, cellTuple[d], d, false);
          const int testPolyOrder  = _tensorMesh->effectivePolyOrder(testDiscretization,  cellTuple[d], d, true);
          const bool trialDerivative = term.trialOperator().takesDerivativeInComponent(d);
          const bool testDerivative  = term.testOperator().takesDerivativeInComponent(d);
          quadratureDegrees[d] = quadratureDegreeForComponent(trialPolyOrder, testPolyOrder,
                                                             trialDerivative, testDerivative);
        }

        const auto basisCacheConstructionStart = std::chrono::steady_clock::now();
        TensorBasisCacheND trialBasisCache(_tensorMesh, trialDiscretization, false, cellTuple, quadratureDegrees);
        TensorBasisCacheND testBasisCache (_tensorMesh, testDiscretization,  true,  cellTuple, quadratureDegrees);
        recordTensorOperatorStage(&_instrumentation,
                                  TensorOperatorInstrumentation::BASIS_CACHE_CONSTRUCTION,
                                  basisCacheConstructionStart);

        std::vector<int> pointCounts(componentCount, 0);
        for (int d=0; d<componentCount; d++)
        {
          pointCounts[d] = trialBasisCache.componentCache(d).pointCount();
        }

        const IndexType tensorPointCount = trialBasisCache.tensorPointCount();
        const auto coefficientEvaluationStart = std::chrono::steady_clock::now();
        TensorFunction::ValueView coefficientValues("MatrixFreeTensorOperator::coefficientValues", tensorPointCount);
        term.coefficient()->values(trialBasisCache.physicalPoints().d_view, coefficientValues);
        auto coefficientValuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coefficientValues);
        recordTensorOperatorStage(&_instrumentation,
                                  TensorOperatorInstrumentation::COEFFICIENT_EVALUATION,
                                  coefficientEvaluationStart);
        auto quadratureWeights = trialBasisCache.quadratureWeights();
        quadratureWeights.sync_host();

        CoefficientFactorExpansion sumFactorizedCoefficientFactorTerms;
        if (termSupportsSumFactorizedVolumeContraction(term, sumFactorizedCoefficientFactorTerms))
        {
          applySumFactorizedVolumeTermToCell(term,
                                            _tensorDofLayout,
                                            _domainLayout,
                                            _rangeLayout,
                                            _domainOrdering,
                                            _rangeOrdering,
                                            trialBasisCache,
                                            testBasisCache,
                                            trialTraversal,
                                            testTraversal,
                                            localInput,
                                            localOutput,
                                            &_instrumentation);
          continue;
        }

        _instrumentation.recordFullTensorVolumeTerm();

        const auto localActionStart = std::chrono::steady_clock::now();
        std::vector<double> trialPointValues(tensorPointCount, 0.0);
        for (IndexType localTrialOrdinal=0; localTrialOrdinal<trialShape.totalDofs(); localTrialOrdinal++)
        {
          _tensorDofLayout->decodeLocalDofOrdinal(_domainLayout, trialShape, _domainOrdering, localTrialOrdinal, trialBasisOrdinals);

          const double trialCoefficient = localInput[trialTraversal.globalIndexForLocalOrdinal(localTrialOrdinal, _tensorDofLayout)];
          if (trialCoefficient == 0.0) continue;

          for (int flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
          {
            int workingOrdinal = flatPointOrdinal;
            for (int d=0; d<componentCount; d++)
            {
              pointOrdinals[d] = workingOrdinal % pointCounts[d];
              workingOrdinal /= pointCounts[d];
            }
            const double trialValue = cachedBasisEvaluationAtPoint(trialBasisOrdinals,
                                                                   trialBasisCache,
                                                                   term.trialOperator(),
                                                                   pointOrdinals);
            trialPointValues[flatPointOrdinal] += trialCoefficient * trialValue;
          }
        }

        std::vector<double> cellOutput(testShape.totalDofs(), 0.0);
        for (IndexType localTestOrdinal=0; localTestOrdinal<testShape.totalDofs(); localTestOrdinal++)
        {
          _tensorDofLayout->decodeLocalDofOrdinal(_rangeLayout, testShape, _rangeOrdering, localTestOrdinal, testBasisOrdinals);

          double contribution = 0.0;
          for (int flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
          {
            int workingOrdinal = flatPointOrdinal;
            for (int d=0; d<componentCount; d++)
            {
              pointOrdinals[d] = workingOrdinal % pointCounts[d];
              workingOrdinal /= pointCounts[d];
            }
            const double testValue = cachedBasisEvaluationAtPoint(testBasisOrdinals,
                                                                  testBasisCache,
                                                                  term.testOperator(),
                                                                  pointOrdinals);
            contribution += quadratureWeights.h_view(flatPointOrdinal)
                            * coefficientValuesHost(flatPointOrdinal)
                            * trialPointValues[flatPointOrdinal]
                            * testValue;
          }
          cellOutput[localTestOrdinal] += contribution;
        }
        recordTensorOperatorStage(&_instrumentation,
                                  TensorOperatorInstrumentation::LOCAL_ACTION,
                                  localActionStart);

        const auto scatterStart = std::chrono::steady_clock::now();
        for (IndexType localTestOrdinal=0; localTestOrdinal<testShape.totalDofs(); localTestOrdinal++)
        {
          localOutput[testTraversal.globalIndexForLocalOrdinal(localTestOrdinal, _tensorDofLayout)] += cellOutput[localTestOrdinal];
        }
        recordTensorOperatorStage(&_instrumentation,
                                  TensorOperatorInstrumentation::SCATTER_LOCAL_OUTPUT_ACCUMULATION,
                                  scatterStart);
      }
    }

    void applyBoundaryTerm(const TensorBilinearTerm &term,
                           const std::vector<double> &localInput,
                           std::vector<double> &localOutput) const
    {
      validateSupportedBoundaryTermAgainstCurrentSemantics(term);

      auto trialDiscretization = _tensorMesh->trialDiscretization(_domainVarName);
      auto testDiscretization  = _tensorMesh->testDiscretization(_rangeVarName);

      const int orthogonalComponentOrdinal = supportedBoundaryOrthogonalComponentOrdinal(term);
      auto skeleton = _tensorMesh->topology()->tensorSkeleton(orthogonalComponentOrdinal);

      std::vector<int> trialBasisOrdinals(term.componentCount(), 0);
      std::vector<int> testBasisOrdinals(term.componentCount(), 0);
      std::vector<int> pointOrdinals(term.componentCount(), 0);

      for (const auto &entity : skeleton.entities())
      {
        auto localContext = _tensorDofLayout->skeletonContext(_domainVarName, _rangeVarName, entity);
        const auto &trialShape = localContext.trialShape();
        const auto &testShape  = localContext.testShape();
        const auto trialTraversal = domainTraversalForSkeletonEntity(entity);
        const auto testTraversal  = rangeTraversalForSkeletonEntity(entity);

        const auto basisCacheConstructionStart = std::chrono::steady_clock::now();
        TraceEntityGeometryData geometryData = buildTraceEntityGeometryData(entity,
                                                                           term,
                                                                           trialDiscretization,
                                                                           testDiscretization);

        auto trialBasisData = buildBasisDataForBoundaryEntity(trialDiscretization, false, entity, geometryData);
        auto testBasisData  = buildBasisDataForBoundaryEntity(testDiscretization,  true,  entity, geometryData);
        recordTensorOperatorStage(&_instrumentation,
                                  TensorOperatorInstrumentation::BASIS_CACHE_CONSTRUCTION,
                                  basisCacheConstructionStart);

        const auto coefficientEvaluationStart = std::chrono::steady_clock::now();
        TensorFunction::ValueView coefficientValues("MatrixFreeTensorOperator::traceCoefficientValues",
                                                    geometryData.tensorPointCount);
        term.coefficient()->values(geometryData.physicalPoints, coefficientValues);
        auto coefficientValuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coefficientValues);
        recordTensorOperatorStage(&_instrumentation,
                                  TensorOperatorInstrumentation::COEFFICIENT_EVALUATION,
                                  coefficientEvaluationStart);

        double trialParity = 1.0;
        double testParity  = 1.0;

        auto orthogonalTopo = _tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);
        const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
        const IndexType incidentEnd   = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal() + 1);
        TEUCHOS_TEST_FOR_EXCEPTION(incidentBegin >= incidentEnd, std::logic_error,
                                   "MatrixFreeTensorOperator expected at least one incident orthogonal-component cell for the requested skeleton vertex.");
        const ComponentCellID1D &incidentCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];

        if (tensorVarIsFluxLike(term.trialVar()))
        {
          trialParity = sideParity(*orthogonalTopo, incidentCell, entity.vertexOrdinal());
        }
        if (tensorVarIsFluxLike(term.testVar()))
        {
          testParity = sideParity(*orthogonalTopo, incidentCell, entity.vertexOrdinal());
        }

        const auto localActionStart = std::chrono::steady_clock::now();
        std::vector<double> trialPointValues(geometryData.tensorPointCount, 0.0);
        for (IndexType localTrialOrdinal=0; localTrialOrdinal<trialShape.totalDofs(); localTrialOrdinal++)
        {
          _tensorDofLayout->decodeLocalDofOrdinal(_domainLayout, trialShape, _domainOrdering, localTrialOrdinal, trialBasisOrdinals);

          const double trialCoefficient = localInput[trialTraversal.globalIndexForLocalOrdinal(localTrialOrdinal, _tensorDofLayout)];
          if (trialCoefficient == 0.0) continue;

          for (int flatPointOrdinal=0; flatPointOrdinal<geometryData.tensorPointCount; flatPointOrdinal++)
          {
            int workingOrdinal = flatPointOrdinal;
            for (size_t retainedOrdinal=0; retainedOrdinal<geometryData.retainedComponentOrdinals.size(); retainedOrdinal++)
            {
              const int d = geometryData.retainedComponentOrdinals[retainedOrdinal];
              pointOrdinals[d] = workingOrdinal % geometryData.pointCounts[d];
              workingOrdinal /= geometryData.pointCounts[d];
            }
            pointOrdinals[geometryData.orthogonalComponentOrdinal] = 0;

            const double trialValue = traceAwareBasisEvaluationAtPoint(term.trialVar(),
                                                                       trialBasisOrdinals,
                                                                       trialBasisData,
                                                                       term.trialOperator(),
                                                                       geometryData,
                                                                       pointOrdinals,
                                                                       entity);
            trialPointValues[flatPointOrdinal] += trialCoefficient * trialParity * trialValue;
          }
        }

        std::vector<double> cellOutput(testShape.totalDofs(), 0.0);
        for (IndexType localTestOrdinal=0; localTestOrdinal<testShape.totalDofs(); localTestOrdinal++)
        {
          _tensorDofLayout->decodeLocalDofOrdinal(_rangeLayout, testShape, _rangeOrdering, localTestOrdinal, testBasisOrdinals);

          double contribution = 0.0;

          for (int flatPointOrdinal=0; flatPointOrdinal<geometryData.tensorPointCount; flatPointOrdinal++)
          {
            int workingOrdinal = flatPointOrdinal;
            for (size_t retainedOrdinal=0; retainedOrdinal<geometryData.retainedComponentOrdinals.size(); retainedOrdinal++)
            {
              const int d = geometryData.retainedComponentOrdinals[retainedOrdinal];
              pointOrdinals[d] = workingOrdinal % geometryData.pointCounts[d];
              workingOrdinal /= geometryData.pointCounts[d];
            }
            pointOrdinals[geometryData.orthogonalComponentOrdinal] = 0;

            const double testValue = traceAwareBasisEvaluationAtPoint(term.testVar(),
                                                                      testBasisOrdinals,
                                                                      testBasisData,
                                                                      term.testOperator(),
                                                                      geometryData,
                                                                      pointOrdinals,
                                                                      entity);

            contribution += geometryData.quadratureWeights[flatPointOrdinal]
                            * coefficientValuesHost(flatPointOrdinal)
                            * trialPointValues[flatPointOrdinal]
                            * (testParity * testValue);
          }
          cellOutput[localTestOrdinal] += contribution;
        }
        recordTensorOperatorStage(&_instrumentation,
                                  TensorOperatorInstrumentation::LOCAL_ACTION,
                                  localActionStart);

        const auto scatterStart = std::chrono::steady_clock::now();
        for (IndexType localTestOrdinal=0; localTestOrdinal<testShape.totalDofs(); localTestOrdinal++)
        {
          localOutput[testTraversal.globalIndexForLocalOrdinal(localTestOrdinal, _tensorDofLayout)] += cellOutput[localTestOrdinal];
        }
        recordTensorOperatorStage(&_instrumentation,
                                  TensorOperatorInstrumentation::SCATTER_LOCAL_OUTPUT_ACCUMULATION,
                                  scatterStart);
      }
    }

    void applyTerm(const TensorBilinearTerm &term,
                   const std::vector<double> &localInput,
                   std::vector<double> &localOutput) const
    {
      if (term.isVolumeTerm())
      {
        applyVolumeTerm(term, localInput, localOutput);
      }
      else if (term.isBoundaryTerm())
      {
        applyBoundaryTerm(term, localInput, localOutput);
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                                   "MatrixFreeTensorOperator encountered an unrecognized tensor bilinear term domain.");
      }
    }

  public:
    MatrixFreeTensorOperator(const TensorMeshPtr &tensorMesh,
                             const TensorDofLayoutPtr &tensorDofLayout,
                             const TensorBilinearFormPtr &bilinearForm,
                             TensorSpaceRole domainRole,
                             const std::string &domainVarName,
                             TensorSpaceRole rangeRole,
                             const std::string &rangeVarName,
                             TensorLocalOrdering domainOrdering = TENSOR_VARIABLE_ORDERING,
                             TensorLocalOrdering rangeOrdering = TENSOR_VARIABLE_ORDERING)
    : _tensorMesh(tensorMesh),
      _tensorDofLayout(tensorDofLayout),
      _bilinearForm(bilinearForm),
      _domainRole(domainRole),
      _rangeRole(rangeRole),
      _domainVarName(domainVarName),
      _rangeVarName(rangeVarName),
      _domainOrdering(domainOrdering),
      _rangeOrdering(rangeOrdering),
      _domainGatherScatter(tensorMesh, tensorDofLayout, domainRole, domainVarName, domainOrdering),
      _rangeGatherScatter(tensorMesh, tensorDofLayout, rangeRole, rangeVarName, rangeOrdering),
      _pipeline(TensorApplyPipelineDescription::standardMatrixFreePipeline()),
      _domainLayout(tensorDofLayout->layout(domainVarName, domainRole)),
      _rangeLayout(tensorDofLayout->layout(rangeVarName, rangeRole)),
      _ambientDomainShape(tensorDofLayout->ambientShape(domainVarName, domainRole)),
      _ambientRangeShape(tensorDofLayout->ambientShape(rangeVarName, rangeRole))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_tensorMesh == Teuchos::null, std::invalid_argument,
                                 "MatrixFreeTensorOperator requires a non-null TensorMesh.");
      TEUCHOS_TEST_FOR_EXCEPTION(_tensorDofLayout == Teuchos::null, std::invalid_argument,
                                 "MatrixFreeTensorOperator requires a non-null TensorDofLayout.");
      TEUCHOS_TEST_FOR_EXCEPTION(_bilinearForm == Teuchos::null, std::invalid_argument,
                                 "MatrixFreeTensorOperator requires a non-null TensorBilinearForm.");
      _bilinearForm->validateAgainst(_tensorMesh);
    }

    TensorOperatorBackend backend() const override
    {
      return TENSOR_OPERATOR_BACKEND_MATRIX_FREE;
    }

    TensorMeshPtr tensorMesh() const override
    {
      return _tensorMesh;
    }

    TensorDofLayoutPtr tensorDofLayout() const override
    {
      return _tensorDofLayout;
    }

    TensorBilinearFormPtr bilinearForm() const override
    {
      return _bilinearForm;
    }

    TensorSpaceRole domainRole() const override
    {
      return _domainRole;
    }

    TensorSpaceRole rangeRole() const override
    {
      return _rangeRole;
    }

    const std::string &domainVarName() const override
    {
      return _domainVarName;
    }

    const std::string &rangeVarName() const override
    {
      return _rangeVarName;
    }

    IndexType domainSize() const override
    {
      return _domainGatherScatter.coefficientCount();
    }

    IndexType rangeSize() const override
    {
      return _rangeGatherScatter.coefficientCount();
    }

    TensorApplyPipelineDescription pipelineDescription() const override
    {
      return _pipeline;
    }

    void resetInstrumentation() const override
    {
      _instrumentation.reset();
    }

    TensorOperatorInstrumentation instrumentation() const override
    {
      return _instrumentation;
    }

    void apply(const std::vector<double> &input,
               std::vector<double> &output) const override
    {
      validateInputVector(input);

      const std::vector<double> &localInput = input;

      std::vector<double> localOutput(rangeSize(), 0.0);
      for (const auto &term : _bilinearForm->terms())
      {
        const bool correctDomain = (term.trialVar()->name() == _domainVarName);
        const bool correctRange  = (term.testVar()->name()  == _rangeVarName);
        if (correctDomain && correctRange)
        {
          applyTerm(term, localInput, localOutput);
        }
      }

      output = localOutput;
    }
  };

  class MatrixFreeGraphNormSchurTensorOperator : public TensorOperator
  {
    TensorMeshPtr _tensorMesh;
    TensorDofLayoutPtr _tensorDofLayout;
    TensorBilinearFormPtr _bilinearForm;
    std::vector<std::string> _trialVarNames;
    std::vector<std::string> _testVarNames;
    std::string _domainVarName;
    std::string _rangeVarName;
    double _l2Weight;
    bool _includeBoundaryTerms;
    TensorApplyPipelineDescription _pipeline;
    IndexType _trialDofCount = 0;
    mutable TensorOperatorInstrumentation _instrumentation;

    struct CachedGraphSchurBlock
    {
      std::vector<IndexType> trialGids;
      std::vector<double> bMatrix;
      TensorLocalSchurComplementBuilder::LocalGraphNormAction graphAction;

      CachedGraphSchurBlock(const std::vector<IndexType> &trialGids,
                            const std::vector<double> &bMatrix,
                            const TensorLocalSchurComplementBuilder::LocalGraphNormAction &graphAction)
      : trialGids(trialGids), bMatrix(bMatrix), graphAction(graphAction)
      {}

      IndexType trialDofCount() const
      {
        return static_cast<IndexType>(trialGids.size());
      }
    };
    std::vector<CachedGraphSchurBlock> _cachedBlocks;

    IndexType compactColumnOrdinal(IndexType trialGid,
                                   IndexType testDofCount,
                                   std::vector<IndexType> &trialGids,
                                   std::map<IndexType,IndexType> &trialGidToColumn,
                                   std::vector<double> &bMatrix) const
    {
      auto found = trialGidToColumn.find(trialGid);
      if (found != trialGidToColumn.end()) return found->second;

      const IndexType oldColumnCount = static_cast<IndexType>(trialGids.size());
      const IndexType newColumnCount = oldColumnCount + 1;
      std::vector<double> expanded(static_cast<size_t>(testDofCount * newColumnCount), 0.0);
      for (IndexType row=0; row<testDofCount; row++)
      {
        for (IndexType col=0; col<oldColumnCount; col++)
        {
          expanded[static_cast<size_t>(row * newColumnCount + col)] = bMatrix[static_cast<size_t>(row * oldColumnCount + col)];
        }
      }
      bMatrix.swap(expanded);
      trialGids.push_back(trialGid);
      trialGidToColumn[trialGid] = oldColumnCount;
      return oldColumnCount;
    }

    void appendCachedVolumeBlocks(const TensorOperatorData &operatorData,
                                  const std::vector<IndexType> &trialOffsets,
                                  const std::string &testVarName)
    {
      const IndexType trialVarCount = static_cast<IndexType>(_trialVarNames.size());
      std::vector<std::vector<bool>> pairHasBlocks(1, std::vector<bool>(trialVarCount, false));
      std::vector<std::vector<TensorLocalSchurComplementBuilder::LocalVolumeSchurComplementBlock>> pairBlocks(trialVarCount);
      for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
      {
        const std::string &trialVarName = _trialVarNames[trialVarOrdinal];
        if (!hasMatchingVolumeTerms(_bilinearForm, trialVarName, testVarName)) continue;
        pairBlocks[trialVarOrdinal] = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(_tensorMesh,
                                                                                                            _tensorDofLayout,
                                                                                                            _bilinearForm,
                                                                                                            trialVarName,
                                                                                                            testVarName);
        TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(pairBlocks[trialVarOrdinal].size()) != _tensorMesh->topology()->activeCellCount(), std::logic_error,
                                   "MatrixFreeGraphNormSchurTensorOperator expected one volume B block per active tensor cell.");
        pairHasBlocks[0][trialVarOrdinal] = true;
      }

      const auto testDiscretization = _tensorMesh->testDiscretization(testVarName);
      for (IndexType tensorCellOrdinal=0; tensorCellOrdinal<_tensorMesh->topology()->activeCellCount(); tensorCellOrdinal++)
      {
        const auto cellTuple = _tensorMesh->topology()->tensorCellTuple(tensorCellOrdinal);
        auto testShape = _tensorDofLayout->factorSpaceShape(testVarName, TENSOR_TEST_SPACE, cellTuple);
        const IndexType localTestDofCount = operatorData.volumeLocalDofCount(TENSOR_TEST_SPACE, testVarName, tensorCellOrdinal);
        TEUCHOS_TEST_FOR_EXCEPTION(localTestDofCount != testShape.totalDofs(), std::logic_error,
                                   "MatrixFreeGraphNormSchurTensorOperator local test shape does not match TensorOperatorData.");
        IndexType compactTrialDofCount = 0;
        for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
        {
          if (!pairHasBlocks[0][trialVarOrdinal]) continue;
          const std::string &trialVarName = _trialVarNames[trialVarOrdinal];
          compactTrialDofCount += operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal);
        }
        std::vector<IndexType> localTrialGids;
        localTrialGids.reserve(static_cast<size_t>(compactTrialDofCount));
        std::vector<double> localB(static_cast<size_t>(localTestDofCount * compactTrialDofCount), 0.0);
        IndexType compactColumnStart = 0;

        for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
        {
          if (!pairHasBlocks[0][trialVarOrdinal]) continue;
          const std::string &trialVarName = _trialVarNames[trialVarOrdinal];
          const auto &block = pairBlocks[trialVarOrdinal][tensorCellOrdinal];
          const IndexType localTrialDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal);
          TEUCHOS_TEST_FOR_EXCEPTION((localTrialDofCount != block.trialShape.totalDofs()) || (localTestDofCount != block.testShape.totalDofs()), std::logic_error,
                                     "MatrixFreeGraphNormSchurTensorOperator local volume B block shape does not match TensorOperatorData.");
          for (IndexType localTrialOrdinal=0; localTrialOrdinal<localTrialDofCount; localTrialOrdinal++)
          {
            const IndexType trialGid = trialOffsets[trialVarOrdinal]
                + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal, localTrialOrdinal);
            const IndexType compactColumn = compactColumnStart + localTrialOrdinal;
            localTrialGids.push_back(trialGid);
            for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
            {
              localB[static_cast<size_t>(localTestOrdinal * compactTrialDofCount + compactColumn)] += block.bMatrix[static_cast<size_t>(localTestOrdinal * localTrialDofCount + localTrialOrdinal)];
            }
          }
          compactColumnStart += localTrialDofCount;
        }

        std::vector<int> gQuadratureDegrees(testShape.componentCount(), 1);
        for (int d=0; d<testShape.componentCount(); d++)
        {
          const int testPolyOrder = _tensorMesh->effectivePolyOrder(testDiscretization, cellTuple[d], d, true);
          gQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
        }
        TensorBasisCacheND gTestBasisCache(_tensorMesh, testDiscretization, true, cellTuple, gQuadratureDegrees);
        _cachedBlocks.push_back(CachedGraphSchurBlock(localTrialGids,
                                                      localB,
                                                      TensorLocalSchurComplementBuilder::localGraphNormAction(localTestDofCount,
                                                                                                             TensorLocalSchurComplementBuilder::graphTestNormMatrix(testShape, gTestBasisCache, _l2Weight))));
      }
    }

    void buildCachedBlocks()
    {
      TensorOperatorData operatorData(_tensorMesh, _tensorDofLayout);
      std::vector<IndexType> trialOffsets(_trialVarNames.size(), 0);
      IndexType totalTrialDofCount = 0;
      for (size_t trialVarOrdinal=0; trialVarOrdinal<_trialVarNames.size(); trialVarOrdinal++)
      {
        trialOffsets[trialVarOrdinal] = totalTrialDofCount;
        totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, _trialVarNames[trialVarOrdinal]);
      }
      TEUCHOS_TEST_FOR_EXCEPTION(totalTrialDofCount != _trialDofCount, std::logic_error,
                                 "MatrixFreeGraphNormSchurTensorOperator cached trial size does not match operator size.");

      _cachedBlocks.clear();
      for (const auto &testVarName : _testVarNames)
      {
        if (_includeBoundaryTerms)
        {
          appendCachedVolumeBoundaryBlocks(operatorData, trialOffsets, testVarName);
        }
        else
        {
          appendCachedVolumeBlocks(operatorData, trialOffsets, testVarName);
        }
      }
    }

    void appendCachedVolumeBoundaryBlocks(const TensorOperatorData &operatorData,
                                          const std::vector<IndexType> &trialOffsets,
                                          const std::string &testVarName)
    {
      const IndexType trialVarCount = static_cast<IndexType>(_trialVarNames.size());
      TensorCoefficientMap testMap(_tensorMesh, _tensorDofLayout, TENSOR_TEST_SPACE, testVarName);
      const auto testDiscretization = _tensorMesh->testDiscretization(testVarName);

      for (IndexType tensorCellOrdinal=0; tensorCellOrdinal<_tensorMesh->topology()->activeCellCount(); tensorCellOrdinal++)
      {
        const auto cellTuple = _tensorMesh->topology()->tensorCellTuple(tensorCellOrdinal);
        auto testShape = _tensorDofLayout->factorSpaceShape(testVarName, TENSOR_TEST_SPACE, cellTuple);
        const IndexType localTestDofCount = operatorData.volumeLocalDofCount(TENSOR_TEST_SPACE, testVarName, tensorCellOrdinal);
        std::vector<int> gQuadratureDegrees(testShape.componentCount(), 1);
        for (int d=0; d<testShape.componentCount(); d++)
        {
          const int testPolyOrder = _tensorMesh->effectivePolyOrder(testDiscretization, cellTuple[d], d, true);
          gQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
        }
        TensorBasisCacheND gTestBasisCache(_tensorMesh, testDiscretization, true, cellTuple, gQuadratureDegrees);
        std::vector<IndexType> localTrialGids;
        std::map<IndexType,IndexType> localTrialGidToColumn;
        std::vector<double> localB;

        for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
        {
          const std::string &trialVarName = _trialVarNames[trialVarOrdinal];
          if (hasMatchingVolumeTerms(_bilinearForm, trialVarName, testVarName))
          {
            auto volumeBlocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(_tensorMesh,
                                                                                                      _tensorDofLayout,
                                                                                                      _bilinearForm,
                                                                                                      trialVarName,
                                                                                                      testVarName);
            const auto &block = volumeBlocks[tensorCellOrdinal];
            const IndexType localTrialDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal);
            for (IndexType localTrialOrdinal=0; localTrialOrdinal<localTrialDofCount; localTrialOrdinal++)
            {
              const IndexType trialGid = trialOffsets[trialVarOrdinal]
                  + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal, localTrialOrdinal);
              const IndexType compactColumn = compactColumnOrdinal(trialGid, localTestDofCount, localTrialGids, localTrialGidToColumn, localB);
              for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
              {
                localB[static_cast<size_t>(localTestOrdinal * localTrialGids.size() + compactColumn)] += block.bMatrix[static_cast<size_t>(localTestOrdinal * localTrialDofCount + localTrialOrdinal)];
              }
            }
          }

          if (hasMatchingBoundaryTerms(_bilinearForm, trialVarName, testVarName))
          {
            auto boundaryBlocks = TensorLocalSchurComplementBuilder::affineBoundaryBBlocks(_tensorMesh,
                                                                                          _tensorDofLayout,
                                                                                          _bilinearForm,
                                                                                          trialVarName,
                                                                                          testVarName);
            const auto &trialVariableData = operatorData.variableData(TENSOR_TRIAL_SPACE, trialVarName);
            TensorCoefficientMap trialMap(_tensorMesh, _tensorDofLayout, TENSOR_TRIAL_SPACE, trialVarName);
            for (const auto &block : boundaryBlocks)
            {
              const IndexType entityOrdinal = block.entityOrdinal;
              const int orthogonalComponentOrdinal = block.entity.orthogonalComponentOrdinal();
              const auto retainedComponents = _tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(orthogonalComponentOrdinal);
              auto orthogonalTopo = _tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);
              const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(block.entity.vertexOrdinal());
              const ComponentCellID1D incidentCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
              std::vector<ComponentCellID1D> testCells(_tensorMesh->topology()->componentCount(), incidentCell);
              for (size_t retainedOrdinal=0; retainedOrdinal<retainedComponents.size(); retainedOrdinal++)
              {
                testCells[retainedComponents[retainedOrdinal]] = block.entity.retainedCellTuple()[retainedOrdinal];
              }
              if (_tensorMesh->topology()->tensorCellOrdinal(testCells) != tensorCellOrdinal) continue;

              auto testIndices = testMap.indicesForVolumeCell(testCells);
              for (IndexType localTrialOrdinal=0; localTrialOrdinal<block.trialShape.totalDofs(); localTrialOrdinal++)
              {
                IndexType trialGid = trialOffsets[trialVarOrdinal];
                trialGid += trialVariableData.isTraceVariable()
                    ? operatorData.skeletonLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, entityOrdinal, localTrialOrdinal)
                    : trialMap.globalIndexForVolumeLocalOrdinal(testCells, localTrialOrdinal);
                const IndexType compactColumn = compactColumnOrdinal(trialGid, localTestDofCount, localTrialGids, localTrialGidToColumn, localB);
                for (IndexType boundaryTestOrdinal=0; boundaryTestOrdinal<block.testShape.totalDofs(); boundaryTestOrdinal++)
                {
                  const IndexType testGid = testIndices[boundaryTestOrdinal];
                  IndexType cellLocalTestOrdinal = -1;
                  for (IndexType candidate=0; candidate<localTestDofCount; candidate++)
                  {
                    if (operatorData.volumeLocalToGlobalIndex(TENSOR_TEST_SPACE, testVarName, tensorCellOrdinal, candidate) == testGid)
                    {
                      cellLocalTestOrdinal = candidate;
                      break;
                    }
                  }
                  TEUCHOS_TEST_FOR_EXCEPTION(cellLocalTestOrdinal < 0, std::logic_error,
                                             "MatrixFreeGraphNormSchurTensorOperator could not map a boundary test index into the graph volume cell.");
                  localB[static_cast<size_t>(cellLocalTestOrdinal * localTrialGids.size() + compactColumn)] += block.bMatrix[static_cast<size_t>(boundaryTestOrdinal * block.trialShape.totalDofs() + localTrialOrdinal)];
                }
              }
            }
          }
        }

        _cachedBlocks.push_back(CachedGraphSchurBlock(localTrialGids,
                                                      localB,
                                                      TensorLocalSchurComplementBuilder::localGraphNormAction(localTestDofCount,
                                                                                                             TensorLocalSchurComplementBuilder::graphTestNormMatrix(testShape, gTestBasisCache, _l2Weight))));
      }
    }

  public:
    MatrixFreeGraphNormSchurTensorOperator(const TensorMeshPtr &tensorMesh,
                                           const TensorDofLayoutPtr &tensorDofLayout,
                                           const TensorBilinearFormPtr &bilinearForm,
                                           const std::vector<std::string> &trialVarNames,
                                           const std::vector<std::string> &testVarNames,
                                           bool includeBoundaryTerms,
                                           double l2Weight)
    : _tensorMesh(tensorMesh),
      _tensorDofLayout(tensorDofLayout),
      _bilinearForm(bilinearForm),
      _trialVarNames(trialVarNames),
      _testVarNames(testVarNames),
      _domainVarName(joinedVarNames(trialVarNames)),
      _rangeVarName(joinedVarNames(trialVarNames)),
      _l2Weight(l2Weight),
      _includeBoundaryTerms(includeBoundaryTerms),
      _pipeline(TensorApplyPipelineDescription::standardMatrixFreePipeline())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_tensorMesh == Teuchos::null, std::invalid_argument,
                                 "MatrixFreeGraphNormSchurTensorOperator requires a non-null TensorMesh.");
      TEUCHOS_TEST_FOR_EXCEPTION(_tensorDofLayout == Teuchos::null, std::invalid_argument,
                                 "MatrixFreeGraphNormSchurTensorOperator requires a non-null TensorDofLayout.");
      TEUCHOS_TEST_FOR_EXCEPTION(_bilinearForm == Teuchos::null, std::invalid_argument,
                                 "MatrixFreeGraphNormSchurTensorOperator requires a non-null TensorBilinearForm.");
      TEUCHOS_TEST_FOR_EXCEPTION(_trialVarNames.empty() || _testVarNames.empty(), std::invalid_argument,
                                 "MatrixFreeGraphNormSchurTensorOperator requires trial and test variables.");
      _bilinearForm->validateAgainst(_tensorMesh);
      TensorOperatorData operatorData(_tensorMesh, _tensorDofLayout);
      for (const auto &trialVarName : _trialVarNames)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(!_tensorDofLayout->hasLayout(trialVarName, TENSOR_TRIAL_SPACE), std::invalid_argument,
                                   "MatrixFreeGraphNormSchurTensorOperator requires trial-space layouts for all trial variables.");
        _trialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarName);
      }
      for (const auto &testVarName : _testVarNames)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(!_tensorDofLayout->hasLayout(testVarName, TENSOR_TEST_SPACE), std::invalid_argument,
                                   "MatrixFreeGraphNormSchurTensorOperator requires test-space layouts for all eliminated test variables.");
      }
      buildCachedBlocks();
    }

    TensorOperatorBackend backend() const override
    {
      return TENSOR_OPERATOR_BACKEND_MATRIX_FREE;
    }

    TensorMeshPtr tensorMesh() const override
    {
      return _tensorMesh;
    }

    TensorDofLayoutPtr tensorDofLayout() const override
    {
      return _tensorDofLayout;
    }

    TensorBilinearFormPtr bilinearForm() const override
    {
      return _bilinearForm;
    }

    TensorSpaceRole domainRole() const override
    {
      return TENSOR_TRIAL_SPACE;
    }

    TensorSpaceRole rangeRole() const override
    {
      return TENSOR_TRIAL_SPACE;
    }

    const std::string &domainVarName() const override
    {
      return _domainVarName;
    }

    const std::string &rangeVarName() const override
    {
      return _rangeVarName;
    }

    IndexType domainSize() const override
    {
      return _trialDofCount;
    }

    IndexType rangeSize() const override
    {
      return _trialDofCount;
    }

    TensorApplyPipelineDescription pipelineDescription() const override
    {
      return _pipeline;
    }

    void resetInstrumentation() const override
    {
      _instrumentation.reset();
    }

    TensorOperatorInstrumentation instrumentation() const override
    {
      return _instrumentation;
    }

    void apply(const std::vector<double> &input,
               std::vector<double> &output) const override
    {
      TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(input.size()) != _trialDofCount, std::invalid_argument,
                                 "MatrixFreeGraphNormSchurTensorOperator input size does not match trial coefficient count.");
      const auto schurActionStart = std::chrono::steady_clock::now();
      output.assign(_trialDofCount, 0.0);
      for (const auto &cachedBlock : _cachedBlocks)
      {
        std::vector<double> localTrialVector(cachedBlock.trialDofCount(), 0.0);
        for (IndexType localOrdinal=0; localOrdinal<cachedBlock.trialDofCount(); localOrdinal++)
        {
          localTrialVector[localOrdinal] = input[cachedBlock.trialGids[localOrdinal]];
        }
        _instrumentation.recordGraphNormSchurActiveTrialDofs(cachedBlock.trialDofCount());
        auto localOutput = cachedBlock.graphAction.applySchurComplement(cachedBlock.bMatrix,
                                                                        cachedBlock.trialDofCount(),
                                                                        localTrialVector);
        for (IndexType localOrdinal=0; localOrdinal<cachedBlock.trialDofCount(); localOrdinal++)
        {
          output[cachedBlock.trialGids[localOrdinal]] += localOutput[localOrdinal];
        }
      }
      recordTensorOperatorStage(&_instrumentation,
                                TensorOperatorInstrumentation::LOCAL_ACTION,
                                schurActionStart);
      _instrumentation.recordGraphNormSchurApply();
    }
  };
}

TensorApplyPipelineDescription::TensorApplyPipelineDescription()
{
}

TensorApplyPipelineDescription::TensorApplyPipelineDescription(const std::vector<TensorApplyStage> &stages)
: _stages(stages)
{
}

const std::vector<TensorApplyStage> &TensorApplyPipelineDescription::stages() const
{
  return _stages;
}

bool TensorApplyPipelineDescription::containsStage(TensorApplyStage stage) const
{
  return _stages.end() != std::find(_stages.begin(), _stages.end(), stage);
}

TensorApplyPipelineDescription TensorApplyPipelineDescription::standardMatrixFreePipeline()
{
  return TensorApplyPipelineDescription({TENSOR_APPLY_STAGE_GATHER,
                                         TENSOR_APPLY_STAGE_LOCAL_APPLY,
                                         TENSOR_APPLY_STAGE_SCATTER});
}

TensorOperatorBackend MatrixFreeTensorOperatorFactory::backend() const
{
  return TENSOR_OPERATOR_BACKEND_MATRIX_FREE;
}

bool MatrixFreeTensorOperatorFactory::canBuild(const TensorMeshPtr &tensorMesh,
                                               const TensorDofLayoutPtr &tensorDofLayout,
                                               const TensorBilinearFormPtr &bilinearForm,
                                               TensorSpaceRole domainRole,
                                               const std::string &domainVarName,
                                               TensorSpaceRole rangeRole,
                                               const std::string &rangeVarName) const
{
  if ((tensorMesh == Teuchos::null) || (tensorDofLayout == Teuchos::null) || (bilinearForm == Teuchos::null))
  {
    return false;
  }
  if (!tensorDofLayout->hasLayout(domainVarName, domainRole)) return false;
  if (!tensorDofLayout->hasLayout(rangeVarName, rangeRole)) return false;
  if (!bilinearForm->allTermsSupport(TENSOR_OPERATOR_BACKEND_MATRIX_FREE)) return false;

  try
  {
    bilinearForm->validateAgainst(tensorMesh);
    for (const auto &term : bilinearForm->terms())
    {
      if (term.isBoundaryTerm() && !termIsSupportedTraceAwareBoundaryTerm(term)) return false;
    }
  }
  catch (...)
  {
    return false;
  }

  return true;
}

TensorOperatorPtr MatrixFreeTensorOperatorFactory::build(const TensorMeshPtr &tensorMesh,
                                                         const TensorDofLayoutPtr &tensorDofLayout,
                                                         const TensorBilinearFormPtr &bilinearForm,
                                                         TensorSpaceRole domainRole,
                                                         const std::string &domainVarName,
                                                         TensorSpaceRole rangeRole,
                                                         const std::string &rangeVarName) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!canBuild(tensorMesh, tensorDofLayout, bilinearForm,
                                       domainRole, domainVarName,
                                       rangeRole, rangeVarName),
                             std::invalid_argument,
                             "MatrixFreeTensorOperatorFactory::build called with unsupported inputs.");

  return Teuchos::rcp(new MatrixFreeTensorOperator(tensorMesh,
                                                   tensorDofLayout,
                                                   bilinearForm,
                                                   domainRole,
                                                   domainVarName,
                                                   rangeRole,
                                                   rangeVarName));
}

MatrixFreeGraphNormSchurTensorOperatorFactory::MatrixFreeGraphNormSchurTensorOperatorFactory(double l2Weight)
: _l2Weight(l2Weight)
{
}

double MatrixFreeGraphNormSchurTensorOperatorFactory::l2Weight() const
{
  return _l2Weight;
}

std::string MatrixFreeGraphNormSchurTensorOperatorFactory::validationMessage(const TensorMeshPtr &tensorMesh,
                                                                             const TensorDofLayoutPtr &tensorDofLayout,
                                                                             const TensorBilinearFormPtr &bilinearForm,
                                                                             const std::vector<std::string> &trialVarNames,
                                                                             const std::vector<std::string> &testVarNames,
                                                                             bool includeBoundaryTerms) const
{
  if ((tensorMesh == Teuchos::null) || (tensorDofLayout == Teuchos::null) || (bilinearForm == Teuchos::null)) return "graph-norm Schur facade requires non-null tensor mesh, dof layout, and bilinear form.";
  if (_l2Weight < 0.0) return "graph-norm Schur facade requires a nonnegative L2 graph-norm weight.";
  if (trialVarNames.empty() || testVarNames.empty()) return "graph-norm Schur facade requires at least one trial variable and one eliminated test variable.";

  try
  {
    bilinearForm->validateAgainst(tensorMesh);
    TensorOperatorData operatorData(tensorMesh, tensorDofLayout);
    for (const auto &trialVarName : trialVarNames)
    {
      if (!tensorDofLayout->hasLayout(trialVarName, TENSOR_TRIAL_SPACE)) return "graph-norm Schur facade requested a trial variable without a trial-space layout.";
      const auto &trialData = operatorData.variableData(TENSOR_TRIAL_SPACE, trialVarName);
      if (!includeBoundaryTerms && trialData.isTraceVariable()) return "graph-norm Schur volume facade does not accept trace trial variables unless boundary terms are enabled.";
    }
    for (const auto &testVarName : testVarNames)
    {
      if (!tensorDofLayout->hasLayout(testVarName, TENSOR_TEST_SPACE)) return "graph-norm Schur facade requested an eliminated test variable without a test-space layout.";
      const auto &testData = operatorData.variableData(TENSOR_TEST_SPACE, testVarName);
      if (testData.isTraceVariable()) return "graph-norm Schur facade currently requires volume test variables.";
    }

    bool hasUsableTerm = false;
    for (const auto &term : bilinearForm->terms())
    {
      const bool selectedTrial = vectorContainsName(trialVarNames, term.trialVar()->name());
      const bool selectedTest  = vectorContainsName(testVarNames, term.testVar()->name());
      if (!selectedTrial && !selectedTest) continue;
      if (!selectedTrial || !selectedTest) return "graph-norm Schur facade found a term touching only one side of the requested trial/test block.";
      if (!term.isMatrixFreeCandidate()) return "graph-norm Schur facade requires matrix-free-capable terms.";
      if (term.isVolumeTerm())
      {
        if (tensorVarIsTraceLike(term.trialVar()) || tensorVarIsTraceLike(term.testVar())) return "graph-norm Schur volume terms must use volume trial and test variables.";
        hasUsableTerm = true;
        continue;
      }
      if (term.isBoundaryTerm())
      {
        if (!includeBoundaryTerms) return "graph-norm Schur facade found boundary terms but boundary handling was not enabled.";
        if (!termIsSupportedTraceAwareBoundaryTerm(term)) return "graph-norm Schur facade encountered an unsupported boundary/trace term combination.";
        hasUsableTerm = true;
        continue;
      }
      return "graph-norm Schur facade encountered an unsupported bilinear-term domain.";
    }
    if (!hasUsableTerm) return "graph-norm Schur facade found no terms for the requested trial/test block.";
  }
  catch (const std::exception &e)
  {
    return e.what();
  }
  catch (...)
  {
    return "graph-norm Schur facade validation failed with an unknown exception.";
  }

  return "";
}

std::string MatrixFreeGraphNormSchurTensorOperatorFactory::validationMessage(const TensorMeshPtr &tensorMesh,
                                                                             const TensorDofLayoutPtr &tensorDofLayout,
                                                                             const TensorBilinearFormPtr &bilinearForm,
                                                                             TensorSpaceRole domainRole,
                                                                             const std::string &domainVarName,
                                                                             TensorSpaceRole rangeRole,
                                                                             const std::string &rangeVarName) const
{
  if (domainRole != TENSOR_TRIAL_SPACE) return "graph-norm Schur facade requires a trial-space domain.";
  if (rangeRole != TENSOR_TRIAL_SPACE) return "graph-norm Schur facade is trial-to-trial and requires a trial-space range.";
  if (domainVarName != rangeVarName) return "graph-norm Schur facade requires matching domain and range trial variable names.";
  if ((tensorMesh == Teuchos::null) || (tensorDofLayout == Teuchos::null) || (bilinearForm == Teuchos::null)) return "graph-norm Schur facade requires non-null tensor mesh, dof layout, and bilinear form.";

  std::string testVarName;
  for (const auto &term : bilinearForm->terms())
  {
    if (term.trialVar()->name() != domainVarName) continue;
    if (testVarName.empty()) testVarName = term.testVar()->name();
    if (testVarName != term.testVar()->name()) return "graph-norm Schur scalar facade supports exactly one eliminated test variable.";
  }
  if (testVarName.empty()) return "graph-norm Schur facade found no terms for the requested trial variable.";
  return validationMessage(tensorMesh,
                           tensorDofLayout,
                           bilinearForm,
                           std::vector<std::string>{domainVarName},
                           std::vector<std::string>{testVarName},
                           false);
}

TensorOperatorBackend MatrixFreeGraphNormSchurTensorOperatorFactory::backend() const
{
  return TENSOR_OPERATOR_BACKEND_MATRIX_FREE;
}

bool MatrixFreeGraphNormSchurTensorOperatorFactory::canBuild(const TensorMeshPtr &tensorMesh,
                                                             const TensorDofLayoutPtr &tensorDofLayout,
                                                             const TensorBilinearFormPtr &bilinearForm,
                                                             TensorSpaceRole domainRole,
                                                             const std::string &domainVarName,
                                                             TensorSpaceRole rangeRole,
                                                             const std::string &rangeVarName) const
{
  return validationMessage(tensorMesh,
                           tensorDofLayout,
                           bilinearForm,
                           domainRole,
                           domainVarName,
                           rangeRole,
                           rangeVarName).empty();
}

TensorOperatorPtr MatrixFreeGraphNormSchurTensorOperatorFactory::build(const TensorMeshPtr &tensorMesh,
                                                                       const TensorDofLayoutPtr &tensorDofLayout,
                                                                       const TensorBilinearFormPtr &bilinearForm,
                                                                       TensorSpaceRole domainRole,
                                                                       const std::string &domainVarName,
                                                                       TensorSpaceRole rangeRole,
                                                                       const std::string &rangeVarName) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!canBuild(tensorMesh, tensorDofLayout, bilinearForm,
                                       domainRole, domainVarName,
                                       rangeRole, rangeVarName),
                             std::invalid_argument,
                             "MatrixFreeGraphNormSchurTensorOperatorFactory::build called with unsupported inputs.");

  std::string testVarName;
  for (const auto &term : bilinearForm->terms())
  {
    if (term.trialVar()->name() != domainVarName) continue;
    if (testVarName.empty()) testVarName = term.testVar()->name();
    TEUCHOS_TEST_FOR_EXCEPTION(testVarName != term.testVar()->name(), std::invalid_argument,
                               "MatrixFreeGraphNormSchurTensorOperatorFactory minimal facade supports one eliminated test variable.");
  }

  return build(tensorMesh,
               tensorDofLayout,
               bilinearForm,
               std::vector<std::string>{domainVarName},
               std::vector<std::string>{testVarName},
               false);
}

bool MatrixFreeGraphNormSchurTensorOperatorFactory::canBuild(const TensorMeshPtr &tensorMesh,
                                                             const TensorDofLayoutPtr &tensorDofLayout,
                                                             const TensorBilinearFormPtr &bilinearForm,
                                                             const std::vector<std::string> &trialVarNames,
                                                             const std::vector<std::string> &testVarNames,
                                                             bool includeBoundaryTerms) const
{
  return validationMessage(tensorMesh,
                           tensorDofLayout,
                           bilinearForm,
                           trialVarNames,
                           testVarNames,
                           includeBoundaryTerms).empty();
}

TensorOperatorPtr MatrixFreeGraphNormSchurTensorOperatorFactory::build(const TensorMeshPtr &tensorMesh,
                                                                       const TensorDofLayoutPtr &tensorDofLayout,
                                                                       const TensorBilinearFormPtr &bilinearForm,
                                                                       const std::vector<std::string> &trialVarNames,
                                                                       const std::vector<std::string> &testVarNames,
                                                                       bool includeBoundaryTerms) const
{
  const std::string message = validationMessage(tensorMesh,
                                                tensorDofLayout,
                                                bilinearForm,
                                                trialVarNames,
                                                testVarNames,
                                                includeBoundaryTerms);
  TEUCHOS_TEST_FOR_EXCEPTION(!message.empty(),
                             std::invalid_argument,
                             "MatrixFreeGraphNormSchurTensorOperatorFactory::build called with unsupported multi-variable inputs: " + message);
  return Teuchos::rcp(new MatrixFreeGraphNormSchurTensorOperator(tensorMesh,
                                                                 tensorDofLayout,
                                                                 bilinearForm,
                                                                 trialVarNames,
                                                                 testVarNames,
                                                                 includeBoundaryTerms,
                                                                 _l2Weight));
}
