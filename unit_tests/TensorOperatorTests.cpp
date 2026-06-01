#include "Teuchos_UnitTestHarness.hpp"

#include "TensorOperatorFactory.h"
#include "TensorBilinearForm.h"
#include "TensorBilinearFormBuilder.h"
#include "TensorDofLayout.h"
#include "TensorMesh.h"
#include "Basis.h"
#include "BasisFamily.h"
#include "CamelliaIntrepidExtendedTypes.h"
#include "CamelliaTestingHelpers.h"
#include "CubatureFactory.h"
#include "Intrepid2_CubatureDirectLineGauss.hpp"
#include "Var.h"

using namespace Camellia;

namespace
{
struct LineQuadratureRule
{
  std::vector<double> points;
  std::vector<double> weights;
};

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

TensorVarPtr makeFieldTensorVar(const std::string &name)
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, name + "_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, name + "_t", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  return Teuchos::rcp(new TensorVar(name, TENSOR_FIELD, std::vector<VarPtr>{vx, vt}));
}

TensorVarPtr makeTraceTensorVar(const std::string &name, int traceComponentOrdinal)
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, name + "_x", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, name + "_t", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  return Teuchos::rcp(new TensorVar(name, TENSOR_TRACE, std::vector<VarPtr>{vx, vt}, traceComponentOrdinal));
}

TensorVarPtr makeFluxTensorVar(const std::string &name, int traceComponentOrdinal)
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, name + "_x", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, name + "_t", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  return Teuchos::rcp(new TensorVar(name, TENSOR_TRACE, std::vector<VarPtr>{vx, vt}, traceComponentOrdinal));
}

TensorMeshPtr makeTensorMeshForOperatorTests(bool nonuniformFirstComponent = false)
{
  auto component0 = nonuniformFirstComponent
      ? Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2))
      : Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 3));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));

  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("u"), Teuchos::null, std::vector<int>{0,0})));
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("v"), Teuchos::null, std::vector<int>{0,0})));
  return tensorMesh;
}

TensorMeshPtr makeTensorMeshForTraceOperatorTests(bool nonuniformFirstComponent = false)
{
  auto tensorMesh = makeTensorMeshForOperatorTests(nonuniformFirstComponent);
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeTraceTensorVar("uhat", 0), Teuchos::null, std::vector<int>{0,0})));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFluxTensorVar("sigmahat", 0), Teuchos::null, std::vector<int>{0,0})));
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(makeTraceTensorVar("vhat", 0), Teuchos::null, std::vector<int>{0,0})));
  return tensorMesh;
}

TensorBilinearFormPtr makeSimpleBilinearFormForOperatorTests()
{
  auto form = Teuchos::rcp(new TensorBilinearForm("simple"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u"),
                                   makeFieldTensorVar("v"),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0)));
  return form;
}

struct BasisData1D
{
  BasisPtr basis;
  std::vector<double> refPoints;
  std::vector<double> refWeights;
  std::vector<double> values;
  std::vector<double> derivatives;
  int fieldCount = 0;
  int pointCount = 0;
};

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

  basisData.refPoints = quadraturePoints;
  basisData.fieldCount = basisData.basis->getCardinality();
  basisData.pointCount = static_cast<int>(quadraturePoints.size());
  basisData.values.assign(basisData.fieldCount * basisData.pointCount, 0.0);
  basisData.derivatives.assign(basisData.fieldCount * basisData.pointCount, 0.0);

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
      basisData.values[f * basisData.pointCount + p] = basisValues(f,p);
    }
  }

  Camellia::FieldContainer<double> basisDerivatives(basisData.fieldCount, basisData.pointCount, 1);
  basisData.basis->getValues(basisDerivatives, refPointContainer, OP_GRAD);
  for (int f=0; f<basisData.fieldCount; f++)
  {
    for (int p=0; p<basisData.pointCount; p++)
    {
      basisData.derivatives[f * basisData.pointCount + p] = basisDerivatives(f,p,0);
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
      value *= basisData[d].derivatives[flatIndex] * derivativeScales[d];
    }
    else
    {
      value *= basisData[d].values[flatIndex];
    }
  }
  return value;
}

double traceAwareBasisEvaluationAtPoint(const TensorVarPtr &tensorVar,
                                        const std::vector<int> &componentBasisOrdinals,
                                        const std::vector<BasisData1D> &basisData,
                                        const TensorComponentOperatorDescriptor &descriptor,
                                        const std::vector<double> &derivativeScales,
                                        int traceComponentOrdinal,
                                        int orthogonalComponentOrdinal,
                                        IndexType vertexOrdinal,
                                        const std::vector<int> &pointOrdinals)
{
  double value = 1.0;
  const int componentCount = descriptor.componentCount();
  for (int d=0; d<componentCount; d++)
  {
    if ((tensorVar != Teuchos::null) && tensorVar->isTrace() && (d == traceComponentOrdinal))
    {
      value *= (componentBasisOrdinals[d] == 0) ? 1.0 : 0.0;
      continue;
    }

    const int basisOrdinal = componentBasisOrdinals[d];
    const int pointOrdinal = (d == orthogonalComponentOrdinal) ? 0 : pointOrdinals[d];
    const int flatIndex = basisOrdinal * basisData[d].pointCount + pointOrdinal;
    if (descriptor.takesDerivativeInComponent(d))
    {
      value *= basisData[d].derivatives[flatIndex] * derivativeScales[d];
    }
    else
    {
      value *= basisData[d].values[flatIndex];
    }
  }
  return value;
}

double sideParity(const TensorMeshPtr &tensorMesh,
                  int orthogonalComponentOrdinal,
                  const ComponentCellID1D &cellID,
                  IndexType vertexOrdinal)
{
  auto componentTopo = tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);
  const IndexType leftVertex  = componentTopo->cellVertexIndex(cellID, 0);
  const IndexType rightVertex = componentTopo->cellVertexIndex(cellID, 1);
  if (vertexOrdinal == leftVertex) return -1.0;
  if (vertexOrdinal == rightVertex) return 1.0;
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                             "Reference parity computation expected a vertex incident on the selected cell.");
}

double orthogonalReferenceCoordinate(const TensorMeshPtr &tensorMesh,
                                     int orthogonalComponentOrdinal,
                                     const ComponentCellID1D &incidentCell,
                                     IndexType vertexOrdinal)
{
  auto componentTopo = tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);
  const IndexType leftVertex  = componentTopo->cellVertexIndex(incidentCell, 0);
  const IndexType rightVertex = componentTopo->cellVertexIndex(incidentCell, 1);
  if (vertexOrdinal == leftVertex) return -1.0;
  if (vertexOrdinal == rightVertex) return 1.0;
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                             "Reference orthogonal-coordinate computation expected a vertex incident on the selected cell.");
}

bool componentBasisOrdinalsAreAdmissibleForBoundaryEntity(const TensorVarPtr &tensorVar,
                                                          const std::vector<int> &componentBasisOrdinals,
                                                          const std::vector<BasisData1D> &basisData,
                                                          int orthogonalComponentOrdinal)
{
  const int componentCount = static_cast<int>(componentBasisOrdinals.size());
  for (int d=0; d<componentCount; d++)
  {
    if ((tensorVar != Teuchos::null) && tensorVar->isTrace() && (d == tensorVar->traceComponentOrdinal()))
    {
      continue;
    }
    if (d == orthogonalComponentOrdinal)
    {
      continue;
    }

    const int basisOrdinal = componentBasisOrdinals[d];
    if ((basisOrdinal < 0) || (basisOrdinal >= basisData[d].fieldCount)) return false;
  }
  return true;
}

std::vector<double> applyReferenceOperator(const TensorMeshPtr &tensorMesh,
                                           const TensorDofLayoutPtr &dofLayout,
                                           const TensorBilinearFormPtr &form,
                                           const std::string &trialVarName,
                                           const std::string &testVarName,
                                           const std::vector<double> &input)
{
  auto trialDiscretization = tensorMesh->trialDiscretization(trialVarName);
  auto testDiscretization  = tensorMesh->testDiscretization(testVarName);
  const auto &domainLayout = dofLayout->layout(trialVarName, TENSOR_TRIAL_SPACE);
  const auto &rangeLayout  = dofLayout->layout(testVarName,  TENSOR_TEST_SPACE);
  const auto &activeCellTuples = tensorMesh->topology()->activeCellTuples();

  std::vector<double> output(rangeLayout.totalDofs(), 0.0);

  for (const auto &term : form->terms())
  {
    if (term.trialVar()->name() != trialVarName) continue;
    if (term.testVar()->name()  != testVarName) continue;
    if (!term.isVolumeTerm()) continue;

    const int componentCount = term.componentCount();
    std::vector<int> trialBasisOrdinals(componentCount, 0);
    std::vector<int> testBasisOrdinals(componentCount, 0);
    std::vector<int> pointOrdinals(componentCount, 0);

    for (const auto &cellTuple : activeCellTuples)
    {
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
        const int trialPolyOrder = tensorMesh->effectivePolyOrder(trialDiscretization, cellTuple[d], d, false);
        const int testPolyOrder  = tensorMesh->effectivePolyOrder(testDiscretization,  cellTuple[d], d, true);
        const bool trialDerivative = term.trialOperator().takesDerivativeInComponent(d);
        const bool testDerivative  = term.testOperator().takesDerivativeInComponent(d);
        const int quadratureDegree = quadratureDegreeForComponent(trialPolyOrder, testPolyOrder,
                                                                  trialDerivative, testDerivative);
        LineQuadratureRule quadrature = lineQuadratureFromCubatureFactory(quadratureDegree);
        refPoints[d] = quadrature.points;
        refWeights[d] = quadrature.weights;
        pointCounts[d] = static_cast<int>(quadrature.points.size());

        trialBasisData[d] = buildBasisDataForComponent(tensorMesh, trialDiscretization, d, false, cellTuple[d], refPoints[d]);
        testBasisData[d]  = buildBasisDataForComponent(tensorMesh, testDiscretization,  d, true,  cellTuple[d], refPoints[d]);

        auto componentTopo = tensorMesh->topology()->componentMeshTopology(d);
        cellLeft[d] = componentTopo->cellLeft(cellTuple[d]);
        cellWidth[d] = componentTopo->cellWidth(cellTuple[d]);
        measureScales[d] = 0.5 * cellWidth[d];
        derivativeScales[d] = 2.0 / cellWidth[d];
      }

      int tensorPointCount = 1;
      for (int d=0; d<componentCount; d++) tensorPointCount *= pointCounts[d];

      TensorFunction::PointView physicalPoints("reference-physical-points", tensorPointCount, componentCount);
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

      TensorFunction::ValueView coefficientValues("reference-coefficient-values", tensorPointCount);
      term.coefficient()->values(physicalPoints, coefficientValues);
      auto coefficientValuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coefficientValues);

      std::vector<std::vector<double>> matrix(rangeLayout.totalDofs(), std::vector<double>(domainLayout.totalDofs(), 0.0));

      for (IndexType j=0; j<domainLayout.totalDofs(); j++)
      {
        dofLayout->decodeLocalDofOrdinal(trialVarName, TENSOR_TRIAL_SPACE, TENSOR_VARIABLE_ORDERING, j, trialBasisOrdinals);
        for (IndexType i=0; i<rangeLayout.totalDofs(); i++)
        {
          dofLayout->decodeLocalDofOrdinal(testVarName, TENSOR_TEST_SPACE, TENSOR_VARIABLE_ORDERING, i, testBasisOrdinals);
          double entry = 0.0;
          for (int flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
          {
            int workingOrdinal = flatPointOrdinal;
            for (int d=0; d<componentCount; d++)
            {
              pointOrdinals[d] = workingOrdinal % pointCounts[d];
              workingOrdinal /= pointCounts[d];
            }
            const double trialValue = basisEvaluationAtPoint(trialBasisOrdinals, trialBasisData, term.trialOperator(), pointOrdinals, derivativeScales);
            const double testValue  = basisEvaluationAtPoint(testBasisOrdinals,  testBasisData,  term.testOperator(),  pointOrdinals, derivativeScales);
            entry += quadratureWeights[flatPointOrdinal] * coefficientValuesHost(flatPointOrdinal) * trialValue * testValue;
          }
          matrix[i][j] += entry;
        }
      }

      for (IndexType i=0; i<rangeLayout.totalDofs(); i++)
      {
        for (IndexType j=0; j<domainLayout.totalDofs(); j++)
        {
          output[i] += matrix[i][j] * input[j];
        }
      }
    }
  }

  return output;
}

std::vector<double> applyReferenceBoundaryOperator(const TensorMeshPtr &tensorMesh,
                                                   const TensorDofLayoutPtr &dofLayout,
                                                   const TensorBilinearFormPtr &form,
                                                   const std::string &trialVarName,
                                                   const std::string &testVarName,
                                                   const std::vector<double> &input)
{
  auto trialDiscretization = tensorMesh->trialDiscretization(trialVarName);
  auto testDiscretization  = tensorMesh->testDiscretization(testVarName);
  const auto &domainLayout = dofLayout->layout(trialVarName, TENSOR_TRIAL_SPACE);
  const auto &rangeLayout  = dofLayout->layout(testVarName,  TENSOR_TEST_SPACE);

  std::vector<double> output(rangeLayout.totalDofs(), 0.0);

  for (const auto &term : form->terms())
  {
    if (term.trialVar()->name() != trialVarName) continue;
    if (term.testVar()->name()  != testVarName) continue;
    if (!term.isBoundaryTerm()) continue;

    const bool trialTraceLike = term.trialVar()->isTrace();
    const bool testTraceLike  = term.testVar()->isTrace();
    TEUCHOS_TEST_FOR_EXCEPTION(trialTraceLike == testTraceLike, std::invalid_argument,
                               "Reference boundary operator supports only mixed volume/trace cases.");

    const int orthogonalComponentOrdinal = trialTraceLike ? term.trialVar()->traceComponentOrdinal()
                                                          : term.testVar()->traceComponentOrdinal();
    auto skeleton = tensorMesh->topology()->tensorSkeleton(orthogonalComponentOrdinal);

    std::vector<int> trialBasisOrdinals(term.componentCount(), 0);
    std::vector<int> testBasisOrdinals(term.componentCount(), 0);
    std::vector<int> pointOrdinals(term.componentCount(), 0);

    for (const auto &entity : skeleton.entities())
    {
      const int componentCount = tensorMesh->topology()->componentCount();
      const auto retainedComponents = tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(orthogonalComponentOrdinal);

      std::vector<BasisData1D> trialBasisData(componentCount);
      std::vector<BasisData1D> testBasisData(componentCount);
      std::vector<int> pointCounts(componentCount, 1);
      std::vector<std::vector<double>> refPoints(componentCount);
      std::vector<std::vector<double>> refWeights(componentCount);
      std::vector<double> measureScales(componentCount, 1.0);
      std::vector<double> derivativeScales(componentCount, 1.0);
      std::vector<double> fixedCoordinates(componentCount, 0.0);
      std::vector<double> cellLeft(componentCount, 0.0);
      std::vector<double> cellWidth(componentCount, 1.0);

      auto orthogonalTopo = tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);
      fixedCoordinates[orthogonalComponentOrdinal] = orthogonalTopo->activeVertexCoordinates().h_view(entity.vertexOrdinal());
      const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
      const IndexType incidentEnd   = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal() + 1);
      TEUCHOS_TEST_FOR_EXCEPTION(incidentBegin >= incidentEnd, std::logic_error,
                                 "Reference boundary operator expected at least one incident orthogonal-component cell for the requested skeleton vertex.");
      const ComponentCellID1D &incidentOrthogonalCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
      const double orthRefCoord = orthogonalReferenceCoordinate(tensorMesh, orthogonalComponentOrdinal, incidentOrthogonalCell, entity.vertexOrdinal());

      if (!term.trialVar()->isTrace())
      {
        trialBasisData[orthogonalComponentOrdinal] = buildBasisDataForComponent(tensorMesh,
                                                                                trialDiscretization,
                                                                                orthogonalComponentOrdinal,
                                                                                false,
                                                                                incidentOrthogonalCell,
                                                                                std::vector<double>{orthRefCoord});
      }
      if (!term.testVar()->isTrace())
      {
        testBasisData[orthogonalComponentOrdinal] = buildBasisDataForComponent(tensorMesh,
                                                                               testDiscretization,
                                                                               orthogonalComponentOrdinal,
                                                                               true,
                                                                               incidentOrthogonalCell,
                                                                               std::vector<double>{orthRefCoord});
      }

      for (size_t retainedOrdinal=0; retainedOrdinal<retainedComponents.size(); retainedOrdinal++)
      {
        const int d = retainedComponents[retainedOrdinal];
        const ComponentCellID1D &cellID = entity.retainedCellTuple()[retainedOrdinal];
        const int trialPolyOrder = tensorMesh->effectivePolyOrder(trialDiscretization, cellID, d, false);
        const int testPolyOrder  = tensorMesh->effectivePolyOrder(testDiscretization,  cellID, d, true);
        const bool trialDerivative = term.trialOperator().takesDerivativeInComponent(d);
        const bool testDerivative  = term.testOperator().takesDerivativeInComponent(d);
        const int quadratureDegree = quadratureDegreeForComponent(trialPolyOrder, testPolyOrder,
                                                                  trialDerivative, testDerivative);
        LineQuadratureRule quadrature = lineQuadratureFromCubatureFactory(quadratureDegree);
        refPoints[d] = quadrature.points;
        refWeights[d] = quadrature.weights;
        pointCounts[d] = static_cast<int>(quadrature.points.size());

        trialBasisData[d] = buildBasisDataForComponent(tensorMesh, trialDiscretization, d, false, cellID, refPoints[d]);
        testBasisData[d]  = buildBasisDataForComponent(tensorMesh, testDiscretization,  d, true,  cellID, refPoints[d]);

        auto componentTopo = tensorMesh->topology()->componentMeshTopology(d);
        cellLeft[d] = componentTopo->cellLeft(cellID);
        cellWidth[d] = componentTopo->cellWidth(cellID);
        measureScales[d] = 0.5 * cellWidth[d];
        derivativeScales[d] = 2.0 / cellWidth[d];
      }

      int tensorPointCount = 1;
      for (int d : retainedComponents) tensorPointCount *= pointCounts[d];

      TensorFunction::PointView physicalPoints("reference-boundary-physical-points", tensorPointCount, componentCount);
      auto physicalPointsHost = Kokkos::create_mirror_view(physicalPoints);
      std::vector<double> quadratureWeights(tensorPointCount, 1.0);

      for (int flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
      {
        int workingOrdinal = flatPointOrdinal;
        double weight = 1.0;
        for (int d=0; d<componentCount; d++)
        {
          if (d == orthogonalComponentOrdinal)
          {
            physicalPointsHost(flatPointOrdinal, d) = fixedCoordinates[d];
          }
        }
        for (size_t retainedOrdinal=0; retainedOrdinal<retainedComponents.size(); retainedOrdinal++)
        {
          const int d = retainedComponents[retainedOrdinal];
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

      TensorFunction::ValueView coefficientValues("reference-boundary-coefficient-values", tensorPointCount);
      term.coefficient()->values(physicalPoints, coefficientValues);
      auto coefficientValuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coefficientValues);

      const bool trialFluxLike = term.trialVar()->isFluxLikeTrace();
      const bool testFluxLike  = term.testVar()->isFluxLikeTrace();
      double trialParity = 1.0;
      double testParity  = 1.0;
      if (trialFluxLike || testFluxLike)
      {
        const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
        const IndexType incidentEnd   = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal() + 1);
        TEUCHOS_TEST_FOR_EXCEPTION(incidentBegin >= incidentEnd, std::logic_error,
                                   "Reference boundary operator expected at least one incident orthogonal-component cell for the requested skeleton vertex.");
        const ComponentCellID1D &incidentCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
        if (trialFluxLike) trialParity = sideParity(tensorMesh, orthogonalComponentOrdinal, incidentCell, entity.vertexOrdinal());
        if (testFluxLike)  testParity  = sideParity(tensorMesh, orthogonalComponentOrdinal, incidentCell, entity.vertexOrdinal());
      }

      std::vector<double> trialPointValues(tensorPointCount, 0.0);
      for (IndexType j=0; j<domainLayout.totalDofs(); j++)
      {
        dofLayout->decodeLocalDofOrdinal(trialVarName, TENSOR_TRIAL_SPACE, TENSOR_VARIABLE_ORDERING, j, trialBasisOrdinals);
        if (!componentBasisOrdinalsAreAdmissibleForBoundaryEntity(term.trialVar(),
                                                                  trialBasisOrdinals,
                                                                  trialBasisData,
                                                                  orthogonalComponentOrdinal)) continue;

        const double coeff = input[j];
        if (coeff == 0.0) continue;

        for (int flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
        {
          int workingOrdinal = flatPointOrdinal;
          for (size_t retainedOrdinal=0; retainedOrdinal<retainedComponents.size(); retainedOrdinal++)
          {
            const int d = retainedComponents[retainedOrdinal];
            pointOrdinals[d] = workingOrdinal % pointCounts[d];
            workingOrdinal /= pointCounts[d];
          }
          pointOrdinals[orthogonalComponentOrdinal] = 0;

          const double trialValue = traceAwareBasisEvaluationAtPoint(term.trialVar(),
                                                                     trialBasisOrdinals,
                                                                     trialBasisData,
                                                                     term.trialOperator(),
                                                                     derivativeScales,
                                                                     orthogonalComponentOrdinal,
                                                                     orthogonalComponentOrdinal,
                                                                     entity.vertexOrdinal(),
                                                                     pointOrdinals);
          trialPointValues[flatPointOrdinal] += coeff * trialParity * trialValue;
        }
      }

      for (IndexType i=0; i<rangeLayout.totalDofs(); i++)
      {
        dofLayout->decodeLocalDofOrdinal(testVarName, TENSOR_TEST_SPACE, TENSOR_VARIABLE_ORDERING, i, testBasisOrdinals);
        if (!componentBasisOrdinalsAreAdmissibleForBoundaryEntity(term.testVar(),
                                                                  testBasisOrdinals,
                                                                  testBasisData,
                                                                  orthogonalComponentOrdinal)) continue;

        double contribution = 0.0;
        for (int flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
        {
          int workingOrdinal = flatPointOrdinal;
          for (size_t retainedOrdinal=0; retainedOrdinal<retainedComponents.size(); retainedOrdinal++)
          {
            const int d = retainedComponents[retainedOrdinal];
            pointOrdinals[d] = workingOrdinal % pointCounts[d];
            workingOrdinal /= pointCounts[d];
          }
          pointOrdinals[orthogonalComponentOrdinal] = 0;

          const double testValue = traceAwareBasisEvaluationAtPoint(term.testVar(),
                                                                    testBasisOrdinals,
                                                                    testBasisData,
                                                                    term.testOperator(),
                                                                    derivativeScales,
                                                                    orthogonalComponentOrdinal,
                                                                    orthogonalComponentOrdinal,
                                                                    entity.vertexOrdinal(),
                                                                    pointOrdinals);
          contribution += quadratureWeights[flatPointOrdinal]
                          * coefficientValuesHost(flatPointOrdinal)
                          * trialPointValues[flatPointOrdinal]
                          * (testParity * testValue);
        }
        output[i] += contribution;
      }
    }
  }

  return output;
}

void compareVectors(const std::vector<double> &actual,
                    const std::vector<double> &expected,
                    Teuchos::FancyOStream &out,
                    bool &success,
                    double relTol = 1e-12,
                    double absTol = 1e-15)
{
  TEST_EQUALITY(actual.size(), expected.size());
  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA_ABSTOLTOO( actual, expected, relTol, absTol );
}

TEUCHOS_UNIT_TEST( TensorOperator, StandardMatrixFreePipelineContainsExpectedStages )
{
  auto pipeline = TensorApplyPipelineDescription::standardMatrixFreePipeline();
  TEST_EQUALITY(pipeline.stages().size(), 3);
  TEST_ASSERT(pipeline.containsStage(TENSOR_APPLY_STAGE_GATHER));
  TEST_ASSERT(pipeline.containsStage(TENSOR_APPLY_STAGE_LOCAL_APPLY));
  TEST_ASSERT(pipeline.containsStage(TENSOR_APPLY_STAGE_SCATTER));
}

TEUCHOS_UNIT_TEST( TensorOperator, MatrixFreeFactoryBuildsOperatorForSupportedInputs )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = makeSimpleBilinearFormForOperatorTests();
  MatrixFreeTensorOperatorFactory factory;

  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TEST_SPACE,  "v");

  TEST_ASSERT(op != Teuchos::null);
  TEST_EQUALITY(op->backend(), TENSOR_OPERATOR_BACKEND_MATRIX_FREE);
  TEST_EQUALITY(op->domainVarName(), "u");
  TEST_EQUALITY(op->rangeVarName(),  "v");
}

TEUCHOS_UNIT_TEST( TensorOperator, DomainAndRangeSizesMatchVariableLocalTensorSpace )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = makeSimpleBilinearFormForOperatorTests();
  MatrixFreeTensorOperatorFactory factory;

  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TEST_SPACE,  "v");

  TEST_EQUALITY(op->domainSize(), dofLayout->layout("u", TENSOR_TRIAL_SPACE).totalDofs());
  TEST_EQUALITY(op->rangeSize(),  dofLayout->layout("v", TENSOR_TEST_SPACE).totalDofs());
}

TEUCHOS_UNIT_TEST( TensorOperator, ApplyProducesExpectedSizedOutput )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = makeSimpleBilinearFormForOperatorTests();
  MatrixFreeTensorOperatorFactory factory;

  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TEST_SPACE,  "v");

  std::vector<double> input(op->domainSize(), 1.0);
  std::vector<double> output;
  op->apply(input, output);

  TEST_EQUALITY(output.size(), op->rangeSize());
}

TEUCHOS_UNIT_TEST( TensorOperator, ApplyProducesNonzeroOutputForSimpleVolumeTerm )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = makeSimpleBilinearFormForOperatorTests();
  MatrixFreeTensorOperatorFactory factory;

  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TEST_SPACE,  "v");

  std::vector<double> input(op->domainSize(), 2.0);
  std::vector<double> output;
  op->apply(input, output);

  bool someNonzero = false;
  for (double value : output)
  {
    if (value != 0.0) someNonzero = true;
  }
  TEST_ASSERT(someNonzero);
}

TEUCHOS_UNIT_TEST( TensorOperator, ValueValueConstantApplyMatchesExplicitReference )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = makeSimpleBilinearFormForOperatorTests();
  MatrixFreeTensorOperatorFactory factory;
  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TEST_SPACE,  "v");

  std::vector<double> input(op->domainSize(), 0.0);
  for (IndexType i=0; i<op->domainSize(); i++) input[i] = 1.0 + 0.25 * i;

  std::vector<double> actual;
  op->apply(input, actual);
  auto expected = applyReferenceOperator(tensorMesh, dofLayout, form, "u", "v", input);

  compareVectors(actual, expected, out, success);
}

TEUCHOS_UNIT_TEST( TensorOperator, DerivativeValueConstantApplyMatchesExplicitReference )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();

  auto form = Teuchos::rcp(new TensorBilinearForm("derivative-value"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u"),
                                   makeFieldTensorVar("v"),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0)));

  MatrixFreeTensorOperatorFactory factory;
  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TEST_SPACE,  "v");

  std::vector<double> input(op->domainSize(), 0.0);
  for (IndexType i=0; i<op->domainSize(); i++) input[i] = 0.5 + 0.5 * i;

  std::vector<double> actual;
  op->apply(input, actual);
  auto expected = applyReferenceOperator(tensorMesh, dofLayout, form, "u", "v", input);

  compareVectors(actual, expected, out, success);
}

TEUCHOS_UNIT_TEST( TensorOperator, ValueValueCoordinateCoefficientApplyMatchesExplicitReference )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();

  auto form = Teuchos::rcp(new TensorBilinearForm("coordinate-mass"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u"),
                                   makeFieldTensorVar("v"),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::componentCoordinate(0)));

  MatrixFreeTensorOperatorFactory factory;
  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TEST_SPACE,  "v");

  std::vector<double> input(op->domainSize(), 0.0);
  for (IndexType i=0; i<op->domainSize(); i++) input[i] = (i % 2 == 0) ? 1.0 : -0.5;

  std::vector<double> actual;
  op->apply(input, actual);
  auto expected = applyReferenceOperator(tensorMesh, dofLayout, form, "u", "v", input);

  compareVectors(actual, expected, out, success);
}

TEUCHOS_UNIT_TEST( TensorOperator, DerivativeDerivativeCoordinateCoefficientApplyMatchesExplicitReference )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();

  TensorBilinearFormBuilder builder("builder-realization");
  auto form = builder.addDerivativeDerivativeTerm(makeFieldTensorVar("u"),
                                                  makeFieldTensorVar("v"),
                                                  0,
                                                  0,
                                                  TensorFunction::componentCoordinate(1))
                     .build();

  MatrixFreeTensorOperatorFactory factory;
  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TEST_SPACE,  "v");

  std::vector<double> input(op->domainSize(), 0.0);
  for (IndexType i=0; i<op->domainSize(); i++) input[i] = 0.25 * (i + 1);

  std::vector<double> actual;
  op->apply(input, actual);
  auto expected = applyReferenceOperator(tensorMesh, dofLayout, form, "u", "v", input);

  compareVectors(actual, expected, out, success);
}

TEUCHOS_UNIT_TEST( TensorOperator, DerivativeMetadataChangesApplyWeighting )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();

  auto formValue = Teuchos::rcp(new TensorBilinearForm("value"));
  formValue->addTerm(TensorBilinearTerm(makeFieldTensorVar("u"),
                                        makeFieldTensorVar("v"),
                                        TensorComponentOperatorDescriptor::valueOnly(2),
                                        TensorComponentOperatorDescriptor::valueOnly(2),
                                        TensorFunction::constant(1.0)));

  auto formDerivative = Teuchos::rcp(new TensorBilinearForm("derivative"));
  formDerivative->addTerm(TensorBilinearTerm(makeFieldTensorVar("u"),
                                             makeFieldTensorVar("v"),
                                             TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                             TensorComponentOperatorDescriptor::valueOnly(2),
                                             TensorFunction::constant(1.0)));

  MatrixFreeTensorOperatorFactory factory;
  auto opValue = factory.build(tensorMesh, dofLayout, formValue,
                               TENSOR_TRIAL_SPACE, "u",
                               TENSOR_TEST_SPACE,  "v");
  auto opDerivative = factory.build(tensorMesh, dofLayout, formDerivative,
                                    TENSOR_TRIAL_SPACE, "u",
                                    TENSOR_TEST_SPACE,  "v");

  std::vector<double> input(opValue->domainSize(), 1.0);
  std::vector<double> outputValue, outputDerivative;
  opValue->apply(input, outputValue);
  opDerivative->apply(input, outputDerivative);

  TEST_EQUALITY(outputValue.size(), outputDerivative.size());
  bool differs = false;
  for (size_t i=0; i<outputValue.size(); i++)
  {
    if (outputValue[i] != outputDerivative[i]) differs = true;
  }
  TEST_ASSERT(differs);
}

TEUCHOS_UNIT_TEST( TensorOperator, CoordinateCoefficientChangesApplyResult )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();

  auto formConstant = Teuchos::rcp(new TensorBilinearForm("constant"));
  formConstant->addTerm(TensorBilinearTerm(makeFieldTensorVar("u"),
                                           makeFieldTensorVar("v"),
                                           TensorComponentOperatorDescriptor::valueOnly(2),
                                           TensorComponentOperatorDescriptor::valueOnly(2),
                                           TensorFunction::constant(1.0)));

  auto formCoordinate = Teuchos::rcp(new TensorBilinearForm("coordinate"));
  formCoordinate->addTerm(TensorBilinearTerm(makeFieldTensorVar("u"),
                                             makeFieldTensorVar("v"),
                                             TensorComponentOperatorDescriptor::valueOnly(2),
                                             TensorComponentOperatorDescriptor::valueOnly(2),
                                             TensorFunction::componentCoordinate(0)));

  MatrixFreeTensorOperatorFactory factory;
  auto opConstant = factory.build(tensorMesh, dofLayout, formConstant,
                                  TENSOR_TRIAL_SPACE, "u",
                                  TENSOR_TEST_SPACE,  "v");
  auto opCoordinate = factory.build(tensorMesh, dofLayout, formCoordinate,
                                    TENSOR_TRIAL_SPACE, "u",
                                    TENSOR_TEST_SPACE,  "v");

  std::vector<double> input(opConstant->domainSize(), 1.0);
  std::vector<double> outputConstant, outputCoordinate;
  opConstant->apply(input, outputConstant);
  opCoordinate->apply(input, outputCoordinate);

  TEST_EQUALITY(outputConstant.size(), outputCoordinate.size());
  bool differs = false;
  for (size_t i=0; i<outputConstant.size(); i++)
  {
    if (outputConstant[i] != outputCoordinate[i]) differs = true;
  }
  TEST_ASSERT(differs);
}

TEUCHOS_UNIT_TEST( TensorOperator, DistinctTrialCoefficientPatternsProduceDistinctOutputs )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();

  auto derivativeForm = Teuchos::rcp(new TensorBilinearForm("derivative"));
  derivativeForm->addTerm(TensorBilinearTerm(makeFieldTensorVar("u"),
                                             makeFieldTensorVar("v"),
                                             TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                             TensorComponentOperatorDescriptor::valueOnly(2),
                                             TensorFunction::constant(1.0)));

  MatrixFreeTensorOperatorFactory factory;
  auto op = factory.build(tensorMesh, dofLayout, derivativeForm,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TEST_SPACE,  "v");

  std::vector<double> inputA(op->domainSize(), 0.0);
  std::vector<double> inputB(op->domainSize(), 0.0);
  TEST_ASSERT(op->domainSize() >= 2);
  inputA[0] = 1.0;
  inputB[1] = 1.0;

  std::vector<double> outputA, outputB;
  op->apply(inputA, outputA);
  op->apply(inputB, outputB);

  TEST_EQUALITY(outputA.size(), outputB.size());
  bool differs = false;
  for (size_t i=0; i<outputA.size(); i++)
  {
    if (outputA[i] != outputB[i]) differs = true;
  }
  TEST_ASSERT(differs);
}

TEUCHOS_UNIT_TEST( TensorOperator, RealizedOperatorReportsStandardPipelineMetadata )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = makeSimpleBilinearFormForOperatorTests();
  MatrixFreeTensorOperatorFactory factory;

  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TEST_SPACE,  "v");

  auto pipeline = op->pipelineDescription();
  TEST_ASSERT(pipeline.containsStage(TENSOR_APPLY_STAGE_GATHER));
  TEST_ASSERT(pipeline.containsStage(TENSOR_APPLY_STAGE_LOCAL_APPLY));
  TEST_ASSERT(pipeline.containsStage(TENSOR_APPLY_STAGE_SCATTER));
}

TEUCHOS_UNIT_TEST( TensorOperator, BuilderProducedFormCanBeRealized )
{
  auto tensorMesh = makeTensorMeshForOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();

  TensorBilinearFormBuilder builder("builder-realization");
  auto form = builder.addDerivativeDerivativeTerm(makeFieldTensorVar("u"),
                                                  makeFieldTensorVar("v"),
                                                  0,
                                                  0,
                                                  TensorFunction::componentCoordinate(1))
                     .build();

  MatrixFreeTensorOperatorFactory factory;
  TEST_ASSERT(factory.canBuild(tensorMesh, dofLayout, form,
                               TENSOR_TRIAL_SPACE, "u",
                               TENSOR_TEST_SPACE,  "v"));

  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TEST_SPACE,  "v");
  std::vector<double> input(op->domainSize(), 1.0);
  std::vector<double> output;
  op->apply(input, output);
  TEST_EQUALITY(output.size(), op->rangeSize());
}

TEUCHOS_UNIT_TEST( TensorOperator, MixedValueTraceBoundaryApplyMatchesExplicitReference )
{
  auto tensorMesh = makeTensorMeshForTraceOperatorTests(false);
  auto dofLayout = tensorMesh->createDofLayout();

  auto form = Teuchos::rcp(new TensorBilinearForm("trace-boundary-value"));
  form->addTerm(TensorBilinearTerm(makeTraceTensorVar("uhat", 0),
                                   makeFieldTensorVar("v"),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0),
                                   TENSOR_BILINEAR_TERM_BOUNDARY));

  MatrixFreeTensorOperatorFactory factory;
  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "uhat",
                          TENSOR_TEST_SPACE,  "v");

  std::vector<double> input(op->domainSize(), 0.0);
  for (IndexType i=0; i<op->domainSize(); i++) input[i] = 0.75 + 0.125 * i;

  std::vector<double> actual;
  op->apply(input, actual);
  auto expected = applyReferenceBoundaryOperator(tensorMesh, dofLayout, form, "uhat", "v", input);

  compareVectors(actual, expected, out, success);
}

TEUCHOS_UNIT_TEST( TensorOperator, MixedFluxTraceBoundaryParityChangesApplyResult )
{
  auto tensorMesh = makeTensorMeshForTraceOperatorTests(false);
  auto dofLayout = tensorMesh->createDofLayout();

  auto valueTraceForm = Teuchos::rcp(new TensorBilinearForm("trace-boundary-value"));
  valueTraceForm->addTerm(TensorBilinearTerm(makeTraceTensorVar("uhat", 0),
                                             makeFieldTensorVar("v"),
                                             TensorComponentOperatorDescriptor::valueOnly(2),
                                             TensorComponentOperatorDescriptor::valueOnly(2),
                                             TensorFunction::constant(1.0),
                                             TENSOR_BILINEAR_TERM_BOUNDARY));

  auto fluxTraceForm = Teuchos::rcp(new TensorBilinearForm("trace-boundary-flux"));
  fluxTraceForm->addTerm(TensorBilinearTerm(makeFluxTensorVar("sigmahat", 0),
                                            makeFieldTensorVar("v"),
                                            TensorComponentOperatorDescriptor::valueOnly(2),
                                            TensorComponentOperatorDescriptor::valueOnly(2),
                                            TensorFunction::constant(1.0),
                                            TENSOR_BILINEAR_TERM_BOUNDARY));

  MatrixFreeTensorOperatorFactory factory;
  auto opValue = factory.build(tensorMesh, dofLayout, valueTraceForm,
                               TENSOR_TRIAL_SPACE, "uhat",
                               TENSOR_TEST_SPACE,  "v");
  auto opFlux = factory.build(tensorMesh, dofLayout, fluxTraceForm,
                              TENSOR_TRIAL_SPACE, "sigmahat",
                              TENSOR_TEST_SPACE,  "v");

  std::vector<double> inputValue(opValue->domainSize(), 0.0);
  std::vector<double> inputFlux(opFlux->domainSize(), 0.0);
  for (IndexType i=0; i<opValue->domainSize(); i++)
  {
    inputValue[i] = 1.0 + 0.2 * i;
  }
  for (IndexType i=0; i<opFlux->domainSize(); i++)
  {
    inputFlux[i] = 1.0 + 0.2 * i;
  }

  std::vector<double> outputValue, outputFlux;
  opValue->apply(inputValue, outputValue);
  opFlux->apply(inputFlux, outputFlux);

  auto expectedFlux = applyReferenceBoundaryOperator(tensorMesh, dofLayout, fluxTraceForm, "sigmahat", "v", inputFlux);
  compareVectors(outputFlux, expectedFlux, out, success);

  TEST_EQUALITY(outputValue.size(), outputFlux.size());
  bool differs = false;
  for (size_t i=0; i<outputValue.size(); i++)
  {
    if (outputValue[i] != outputFlux[i]) differs = true;
  }
  TEST_ASSERT(differs);
}

TEUCHOS_UNIT_TEST( TensorOperator, UnsupportedBoundaryDerivativeTermIsRejectedByFactory )
{
  auto tensorMesh = makeTensorMeshForTraceOperatorTests();
  auto dofLayout = tensorMesh->createDofLayout();

  auto form = Teuchos::rcp(new TensorBilinearForm("unsupported-boundary-derivative"));
  form->addTerm(TensorBilinearTerm(makeTraceTensorVar("uhat", 0),
                                   makeFieldTensorVar("v"),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 1),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0),
                                   TENSOR_BILINEAR_TERM_BOUNDARY));

  MatrixFreeTensorOperatorFactory factory;
  TEST_ASSERT(!factory.canBuild(tensorMesh, dofLayout, form,
                                TENSOR_TRIAL_SPACE, "uhat",
                                TENSOR_TEST_SPACE,  "v"));
  TEST_THROW(factory.build(tensorMesh, dofLayout, form,
                           TENSOR_TRIAL_SPACE, "uhat",
                           TENSOR_TEST_SPACE,  "v"), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorOperator, TraceAwareBoundaryOperatorRejectsPNonuniformRetainedComponentTopology )
{
  auto tensorMesh = makeTensorMeshForTraceOperatorTests(false);
  auto retainedComponentTopo = tensorMesh->topology()->componentMeshTopology(1);
  retainedComponentTopo->setCellPolyOrder(ComponentCellID1D(0,1,0), retainedComponentTopo->cellPolyOrder(ComponentCellID1D(0,1,0)) + 1);

  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("trace-boundary-value"));
  form->addTerm(TensorBilinearTerm(makeTraceTensorVar("uhat", 0),
                                   makeFieldTensorVar("v"),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0),
                                   TENSOR_BILINEAR_TERM_BOUNDARY));

  MatrixFreeTensorOperatorFactory factory;
  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "uhat",
                          TENSOR_TEST_SPACE,  "v");

  std::vector<double> input(op->domainSize(), 1.0);
  std::vector<double> output;
  TEST_THROW(op->apply(input, output), std::invalid_argument);
}
}
