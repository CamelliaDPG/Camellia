#include "TensorOperator.h"
#include "TensorOperatorFactory.h"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "Basis.h"
#include "BasisFamily.h"
#include "CamelliaIntrepidExtendedTypes.h"
#include "CellTopology.h"
#include "CubatureFactory.h"
#include "Intrepid2_CubatureDirectLineGauss.hpp"
#include "TensorFunction.h"
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

        std::vector<BasisData1D> trialBasisData(componentCount);
        std::vector<BasisData1D> testBasisData(componentCount);
        std::vector<int> pointCounts(componentCount, 0);
        std::vector<std::vector<double>> refPoints(componentCount);
        std::vector<std::vector<double>> refWeights(componentCount);
        std::vector<double> measureScales(componentCount, 1.0);
        std::vector<double> derivativeScales(componentCount, 1.0);
        std::vector<double> cellLeft(componentCount, 0.0);
        std::vector<double> cellWidth(componentCount, 1.0);

        for (int d=0; d<componentCount; d++)
        {
          const int trialPolyOrder = _tensorMesh->effectivePolyOrder(trialDiscretization, cellTuple[d], d, false);
          const int testPolyOrder  = _tensorMesh->effectivePolyOrder(testDiscretization,  cellTuple[d], d, true);
          const bool trialDerivative = term.trialOperator().takesDerivativeInComponent(d);
          const bool testDerivative  = term.testOperator().takesDerivativeInComponent(d);
          const int quadratureDegree = quadratureDegreeForComponent(trialPolyOrder, testPolyOrder,
                                                                    trialDerivative, testDerivative);

          LineQuadratureRule quadrature = lineQuadratureFromCubatureFactory(quadratureDegree);
          refPoints[d] = quadrature.points;
          refWeights[d] = quadrature.weights;
          pointCounts[d] = static_cast<int>(quadrature.points.size());

          trialBasisData[d] = buildBasisDataForComponent(_tensorMesh, trialDiscretization, d, false, cellTuple[d], refPoints[d]);
          testBasisData[d]  = buildBasisDataForComponent(_tensorMesh, testDiscretization,  d, true,  cellTuple[d], refPoints[d]);

          auto componentTopo = _tensorMesh->topology()->componentMeshTopology(d);
          cellLeft[d] = componentTopo->cellLeft(cellTuple[d]);
          cellWidth[d] = componentTopo->cellWidth(cellTuple[d]);
          measureScales[d] = 0.5 * cellWidth[d];
          derivativeScales[d] = 2.0 / cellWidth[d];
        }

        int tensorPointCount = 1;
        for (int d=0; d<componentCount; d++) tensorPointCount *= pointCounts[d];

        TensorFunction::PointView physicalPoints("MatrixFreeTensorOperator::physicalPoints", tensorPointCount, componentCount);
        auto physicalPointsHost = Kokkos::create_mirror_view(physicalPoints);
        std::vector<double> quadratureWeights(tensorPointCount, 1.0);

        for (int flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
        {
          int workingOrdinal = flatPointOrdinal;
          double weight = 1.0;
          for (int d=0; d<componentCount; d++)
          {
            pointOrdinals[d] = workingOrdinal % pointCounts[d];
            workingOrdinal /= pointCounts[d];
            const double xi = refPoints[d][pointOrdinals[d]];
            const double xPhysical = cellLeft[d] + 0.5 * (xi + 1.0) * cellWidth[d];
            physicalPointsHost(flatPointOrdinal, d) = xPhysical;
            weight *= refWeights[d][pointOrdinals[d]] * measureScales[d];
          }
          quadratureWeights[flatPointOrdinal] = weight;
        }
        Kokkos::deep_copy(physicalPoints, physicalPointsHost);

        TensorFunction::ValueView coefficientValues("MatrixFreeTensorOperator::coefficientValues", tensorPointCount);
        term.coefficient()->values(physicalPoints, coefficientValues);
        auto coefficientValuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coefficientValues);

        std::vector<double> trialPointValues(tensorPointCount, 0.0);
        for (IndexType localTrialOrdinal=0; localTrialOrdinal<_ambientDomainShape.totalDofs(); localTrialOrdinal++)
        {
          decodeDomainOrdinal(localTrialOrdinal, trialBasisOrdinals);
          if (!componentBasisOrdinalsAreAdmissibleForShape(trialBasisOrdinals, trialShape)) continue;

          const double trialCoefficient = localInput[localTrialOrdinal];
          if (trialCoefficient == 0.0) continue;

          for (int flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
          {
            int workingOrdinal = flatPointOrdinal;
            for (int d=0; d<componentCount; d++)
            {
              pointOrdinals[d] = workingOrdinal % pointCounts[d];
              workingOrdinal /= pointCounts[d];
            }
            const double trialValue = basisEvaluationAtPoint(trialBasisOrdinals,
                                                             trialBasisData,
                                                             term.trialOperator(),
                                                             pointOrdinals,
                                                             derivativeScales);
            trialPointValues[flatPointOrdinal] += trialCoefficient * trialValue;
          }
        }

        for (IndexType localTestOrdinal=0; localTestOrdinal<_ambientRangeShape.totalDofs(); localTestOrdinal++)
        {
          decodeRangeOrdinal(localTestOrdinal, testBasisOrdinals);
          if (!componentBasisOrdinalsAreAdmissibleForShape(testBasisOrdinals, testShape)) continue;

          double contribution = 0.0;
          for (int flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
          {
            int workingOrdinal = flatPointOrdinal;
            for (int d=0; d<componentCount; d++)
            {
              pointOrdinals[d] = workingOrdinal % pointCounts[d];
              workingOrdinal /= pointCounts[d];
            }
            const double testValue = basisEvaluationAtPoint(testBasisOrdinals,
                                                            testBasisData,
                                                            term.testOperator(),
                                                            pointOrdinals,
                                                            derivativeScales);
            contribution += quadratureWeights[flatPointOrdinal]
                            * coefficientValuesHost(flatPointOrdinal)
                            * trialPointValues[flatPointOrdinal]
                            * testValue;
          }
          localOutput[localTestOrdinal] += contribution;
        }
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

        TraceEntityGeometryData geometryData = buildTraceEntityGeometryData(entity,
                                                                           term,
                                                                           trialDiscretization,
                                                                           testDiscretization);

        auto trialBasisData = buildBasisDataForBoundaryEntity(trialDiscretization, false, entity, geometryData);
        auto testBasisData  = buildBasisDataForBoundaryEntity(testDiscretization,  true,  entity, geometryData);

        TensorFunction::ValueView coefficientValues("MatrixFreeTensorOperator::traceCoefficientValues",
                                                    geometryData.tensorPointCount);
        term.coefficient()->values(geometryData.physicalPoints, coefficientValues);
        auto coefficientValuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coefficientValues);

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

        std::vector<double> trialPointValues(geometryData.tensorPointCount, 0.0);
        for (IndexType localTrialOrdinal=0; localTrialOrdinal<_ambientDomainShape.totalDofs(); localTrialOrdinal++)
        {
          decodeDomainOrdinal(localTrialOrdinal, trialBasisOrdinals);
          if (!componentBasisOrdinalsAreAdmissibleForShape(trialBasisOrdinals, trialShape)) continue;

          const double trialCoefficient = localInput[localTrialOrdinal];
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

        for (IndexType localTestOrdinal=0; localTestOrdinal<_ambientRangeShape.totalDofs(); localTestOrdinal++)
        {
          decodeRangeOrdinal(localTestOrdinal, testBasisOrdinals);
          if (!componentBasisOrdinalsAreAdmissibleForShape(testBasisOrdinals, testShape)) continue;

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
          localOutput[localTestOrdinal] += contribution;
        }
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
      return _domainGatherScatter.localDofCount(_ambientDomainShape);
    }

    IndexType rangeSize() const override
    {
      return _rangeGatherScatter.localDofCount(_ambientRangeShape);
    }

    TensorApplyPipelineDescription pipelineDescription() const override
    {
      return _pipeline;
    }

    void apply(const std::vector<double> &input,
               std::vector<double> &output) const override
    {
      validateInputVector(input);

      std::vector<double> localInput;
      _domainGatherScatter.gatherGlobalToLocal(input, _ambientDomainShape, localInput, 0);

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

      output.assign(rangeSize(), 0.0);
      _rangeGatherScatter.scatterLocalToGlobal(localOutput, _ambientRangeShape, output, 0);
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
