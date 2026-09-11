#include "Teuchos_UnitTestHarness.hpp"

#include "TensorBasisCache.h"
#include "TensorBilinearForm.h"
#include "TensorDofLayout.h"
#include "TensorLocalSchurComplement.h"
#include "TensorMesh.h"
#include "BasisFamily.h"
#include "Var.h"

#include <cmath>
#include <vector>

using namespace Camellia;

namespace
{
TensorFactorSpaceShape makeShape(const std::vector<int> &cardinalities)
{
  std::vector<IndexType> strides(cardinalities.size(), 1);
  IndexType totalDofs = 1;
  for (int d=0; d<static_cast<int>(cardinalities.size()); d++)
  {
    strides[d] = totalDofs;
    totalDofs *= cardinalities[d];
  }
  return TensorFactorSpaceShape(cardinalities, strides, totalDofs);
}

TensorVarPtr makeFieldTensorVar(const std::string &name,
                                int componentCount)
{
  std::vector<VarPtr> componentVars;
  for (int d=0; d<componentCount; d++)
  {
    componentVars.push_back(Teuchos::rcp(new Var(d, 0, name + "_" + std::to_string(d), Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS)));
  }
  return Teuchos::rcp(new TensorVar(name, TENSOR_FIELD, componentVars));
}

TensorMeshPtr makeSingleCellTensorMesh(const std::vector<std::pair<double,double>> &componentExtents)
{
  std::vector<Teuchos::RCP<ComponentMeshTopology1D>> components;
  for (int d=0; d<static_cast<int>(componentExtents.size()); d++)
  {
    components.push_back(Teuchos::rcp(new ComponentMeshTopology1D({componentExtents[d].first, componentExtents[d].second},
                                                                  { ActiveLeafSet1D({{0,0}}) },
                                                                  false,
                                                                  1)));
  }
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology(components));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));
  const int componentCount = static_cast<int>(componentExtents.size());
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("u", componentCount), Teuchos::null, std::vector<int>(componentCount, 0))));
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("v", componentCount), Teuchos::null, std::vector<int>(componentCount, 0))));
  return tensorMesh;
}

TensorMeshPtr makeTwoByTwoAffineTensorMesh()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 3.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("u", 2), Teuchos::null, std::vector<int>{0,0})));
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("v", 2), Teuchos::null, std::vector<int>{0,0})));
  return tensorMesh;
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

std::vector<int> quadratureDegreesForTerm(const TensorMeshPtr &tensorMesh,
                                          const TensorBilinearTerm &term,
                                          const TensorMeshTopology::TensorCellTuple &cellTuple)
{
  auto trialDiscretization = tensorMesh->trialDiscretization(term.trialVar()->name());
  auto testDiscretization  = tensorMesh->testDiscretization(term.testVar()->name());
  std::vector<int> quadratureDegrees(term.componentCount(), 1);
  for (int d=0; d<term.componentCount(); d++)
  {
    const int trialPolyOrder = tensorMesh->effectivePolyOrder(trialDiscretization, cellTuple[d], d, false);
    const int testPolyOrder  = tensorMesh->effectivePolyOrder(testDiscretization,  cellTuple[d], d, true);
    quadratureDegrees[d] = quadratureDegreeForComponent(trialPolyOrder,
                                                        testPolyOrder,
                                                        term.trialOperator().takesDerivativeInComponent(d),
                                                        term.testOperator().takesDerivativeInComponent(d));
  }
  return quadratureDegrees;
}

void decodeOrdinal(const TensorFactorSpaceShape &shape,
                   IndexType ordinal,
                   std::vector<int> &componentOrdinals)
{
  componentOrdinals.assign(shape.componentCount(), 0);
  for (int d=0; d<shape.componentCount(); d++)
  {
    componentOrdinals[d] = static_cast<int>((ordinal / shape.componentStride(d)) % shape.componentCardinality(d));
  }
}

double linearBasisValue(int basisOrdinal,
                        double xi)
{
  return (basisOrdinal == 0) ? 0.5 * (1.0 - xi) : 0.5 * (1.0 + xi);
}

std::vector<double> componentMassEntries(double cellWidth)
{
  const double a = 1.0 / std::sqrt(3.0);
  const std::vector<double> points{-a, a};
  const double jacobian = 0.5 * cellWidth;

  std::vector<double> entries(4, 0.0);
  for (int row=0; row<2; row++)
  {
    for (int col=0; col<2; col++)
    {
      double value = 0.0;
      for (double xi : points)
      {
        value += jacobian * linearBasisValue(row, xi) * linearBasisValue(col, xi);
      }
      entries[row * 2 + col] = value;
    }
  }
  return entries;
}

std::vector<double> componentMassDiagonal(double cellWidth)
{
  auto entries = componentMassEntries(cellWidth);
  return {entries[0], entries[3]};
}

double explicitTensorBEntry(const TensorFactorSpaceShape &testShape,
                            const TensorFactorSpaceShape &trialShape,
                            IndexType testOrdinal,
                            IndexType trialOrdinal,
                            const std::vector<double> &cellWidths)
{
  const double a = 1.0 / std::sqrt(3.0);
  const std::vector<double> points{-a, a};
  std::vector<int> testComponentOrdinals;
  std::vector<int> trialComponentOrdinals;
  decodeOrdinal(testShape, testOrdinal, testComponentOrdinals);
  decodeOrdinal(trialShape, trialOrdinal, trialComponentOrdinals);

  double entry = 0.0;
  for (double xi0 : points)
  {
    for (double xi1 : points)
    {
      const std::vector<double> xis{xi0, xi1};
      double value = 1.0;
      for (int d=0; d<testShape.componentCount(); d++)
      {
        value *= (0.5 * cellWidths[d])
                 * linearBasisValue(testComponentOrdinals[d], xis[d])
                 * linearBasisValue(trialComponentOrdinals[d], xis[d]);
      }
      entry += value;
    }
  }
  return entry;
}

double explicitTensorGDiagonalEntry(const TensorFactorSpaceShape &testShape,
                                    IndexType testOrdinal,
                                    const std::vector<double> &cellWidths)
{
  const double a = 1.0 / std::sqrt(3.0);
  const std::vector<double> points{-a, a};
  std::vector<int> testComponentOrdinals;
  decodeOrdinal(testShape, testOrdinal, testComponentOrdinals);

  double entry = 0.0;
  for (double xi0 : points)
  {
    for (double xi1 : points)
    {
      const std::vector<double> xis{xi0, xi1};
      double value = 1.0;
      for (int d=0; d<testShape.componentCount(); d++)
      {
        const double basisValue = linearBasisValue(testComponentOrdinals[d], xis[d]);
        value *= (0.5 * cellWidths[d]) * basisValue * basisValue;
      }
      entry += value;
    }
  }
  return entry;
}

double cachedBasisEvaluationAtPoint(const std::vector<int> &componentBasisOrdinals,
                                    const TensorBasisCacheND &basisCache,
                                    const TensorComponentOperatorDescriptor &descriptor,
                                    const std::vector<int> &pointOrdinals)
{
  double value = 1.0;
  for (int d=0; d<descriptor.componentCount(); d++)
  {
    const auto &componentCache = basisCache.componentCache(d);
    auto values = componentCache.valuesAtQuadrature();
    auto derivatives = componentCache.derivativesAtQuadrature();
    values.sync_host();
    derivatives.sync_host();
    const int basisOrdinal = componentBasisOrdinals[d];
    const int pointOrdinal = pointOrdinals[d];
    value *= descriptor.takesDerivativeInComponent(d)
        ? derivatives.h_view(basisOrdinal, pointOrdinal) * componentCache.derivativeScale()
        : values.h_view(basisOrdinal, pointOrdinal);
  }
  return value;
}

std::vector<double> coefficientValues(const TensorFunctionPtr &coefficient,
                                      const TensorBasisCacheND &basisCache)
{
  TensorFunction::ValueView values("local-schur-test-coefficient-values", basisCache.tensorPointCount());
  coefficient->values(basisCache.physicalPoints().d_view, values);
  auto valuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  std::vector<double> result(basisCache.tensorPointCount(), 0.0);
  for (IndexType pointOrdinal=0; pointOrdinal<basisCache.tensorPointCount(); pointOrdinal++)
  {
    result[pointOrdinal] = valuesHost(pointOrdinal);
  }
  return result;
}

bool closeEnough(double actual,
                 double expected,
                 double relTol,
                 double absTol)
{
  const double scale = std::max(std::abs(actual), std::abs(expected));
  if (scale <= absTol)
  {
    return std::abs(actual - expected) <= absTol;
  }
  return std::abs(actual - expected) <= relTol * scale;
}

double explicitCachedBEntry(const TensorBilinearTerm &term,
                            const TensorDofLayoutPtr &dofLayout,
                            const TensorDofLayout::VariableLayout &trialLayout,
                            const TensorDofLayout::VariableLayout &testLayout,
                            const TensorFactorSpaceShape &trialShape,
                            const TensorFactorSpaceShape &testShape,
                            const TensorBasisCacheND &trialCache,
                            const TensorBasisCacheND &testCache,
                            const std::vector<double> &coefficientValues,
                            IndexType testOrdinal,
                            IndexType trialOrdinal)
{
  std::vector<int> trialBasisOrdinals(term.componentCount(), 0);
  std::vector<int> testBasisOrdinals(term.componentCount(), 0);
  std::vector<int> pointOrdinals(term.componentCount(), 0);
  dofLayout->decodeLocalDofOrdinal(trialLayout, trialShape, TENSOR_VARIABLE_ORDERING, trialOrdinal, trialBasisOrdinals);
  dofLayout->decodeLocalDofOrdinal(testLayout,  testShape,  TENSOR_VARIABLE_ORDERING, testOrdinal,  testBasisOrdinals);

  auto weights = trialCache.quadratureWeights();
  weights.sync_host();

  double entry = 0.0;
  for (IndexType flatPointOrdinal=0; flatPointOrdinal<trialCache.tensorPointCount(); flatPointOrdinal++)
  {
    IndexType workingOrdinal = flatPointOrdinal;
    for (int d=0; d<term.componentCount(); d++)
    {
      pointOrdinals[d] = workingOrdinal % trialCache.componentCache(d).pointCount();
      workingOrdinal /= trialCache.componentCache(d).pointCount();
    }
    entry += weights.h_view(flatPointOrdinal)
             * coefficientValues[flatPointOrdinal]
             * cachedBasisEvaluationAtPoint(trialBasisOrdinals, trialCache, term.trialOperator(), pointOrdinals)
             * cachedBasisEvaluationAtPoint(testBasisOrdinals,  testCache,  term.testOperator(),  pointOrdinals);
  }
  return entry;
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, LocalBAndDiagonalGMatchExplicitTensorQuadratureReference )
{
  auto shape = makeShape({2,2});
  const std::vector<double> cellWidths{2.0, 3.0};
  TensorProductLocalOperator bOperator(shape,
                                       shape,
                                       {TensorComponentLocalMatrix(2, 2, componentMassEntries(cellWidths[0])),
                                        TensorComponentLocalMatrix(2, 2, componentMassEntries(cellWidths[1]))});

  auto bMatrix = TensorLocalSchurComplementBuilder::denseBMatrix(bOperator);
  TEST_EQUALITY(bMatrix.size(), static_cast<size_t>(shape.totalDofs() * shape.totalDofs()));
  for (IndexType row=0; row<shape.totalDofs(); row++)
  {
    for (IndexType col=0; col<shape.totalDofs(); col++)
    {
      const double expected = explicitTensorBEntry(shape, shape, row, col, cellWidths);
      TEST_FLOATING_EQUALITY(bMatrix[row * shape.totalDofs() + col], expected, 1e-14);
    }
  }

  auto gDiagonal = TensorLocalSchurComplementBuilder::tensorProductDiagonal(shape,
                                                                            {componentMassDiagonal(cellWidths[0]),
                                                                             componentMassDiagonal(cellWidths[1])});
  TEST_EQUALITY(gDiagonal.size(), static_cast<size_t>(shape.totalDofs()));
  for (IndexType row=0; row<shape.totalDofs(); row++)
  {
    const double expected = explicitTensorGDiagonalEntry(shape, row, cellWidths);
    TEST_FLOATING_EQUALITY(gDiagonal[row], expected, 1e-14);
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, LocalKMatchesExplicitDiagonalTestNormReference )
{
  auto shape = makeShape({2,2});
  const std::vector<double> cellWidths{2.0, 3.0};
  TensorProductLocalOperator bOperator(shape,
                                       shape,
                                       {TensorComponentLocalMatrix(2, 2, componentMassEntries(cellWidths[0])),
                                        TensorComponentLocalMatrix(2, 2, componentMassEntries(cellWidths[1]))});
  auto gDiagonal = TensorLocalSchurComplementBuilder::tensorProductDiagonal(shape,
                                                                            {componentMassDiagonal(cellWidths[0]),
                                                                             componentMassDiagonal(cellWidths[1])});

  auto kMatrix = TensorLocalSchurComplementBuilder::formDiagonalTestNormSchurComplement(bOperator, gDiagonal);
  TEST_EQUALITY(kMatrix.size(), static_cast<size_t>(shape.totalDofs() * shape.totalDofs()));

  for (IndexType row=0; row<shape.totalDofs(); row++)
  {
    for (IndexType col=0; col<shape.totalDofs(); col++)
    {
      double expected = 0.0;
      for (IndexType testOrdinal=0; testOrdinal<shape.totalDofs(); testOrdinal++)
      {
        const double bRow = explicitTensorBEntry(shape, shape, testOrdinal, row, cellWidths);
        const double bCol = explicitTensorBEntry(shape, shape, testOrdinal, col, cellWidths);
        const double gEntry = explicitTensorGDiagonalEntry(shape, testOrdinal, cellWidths);
        expected += bRow * bCol / gEntry;
      }
      TEST_FLOATING_EQUALITY(kMatrix[row * shape.totalDofs() + col], expected, 1e-14);
      TEST_FLOATING_EQUALITY(kMatrix[row * shape.totalDofs() + col], kMatrix[col * shape.totalDofs() + row], 1e-14);
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, CachedAffineVolumeBAndGMatchExplicitReference )
{
  auto tensorMesh = makeSingleCellTensorMesh({{0.0, 2.0}, {-1.0, 2.0}});
  auto dofLayout = tensorMesh->createDofLayout();
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorBilinearTerm term(makeFieldTensorVar("u", 2),
                          makeFieldTensorVar("v", 2),
                          TensorComponentOperatorDescriptor::valueOnly(2),
                          TensorComponentOperatorDescriptor::valueOnly(2),
                          TensorFunction::componentCoordinate(0) * TensorFunction::componentCoordinate(1));
  auto quadratureDegrees = quadratureDegreesForTerm(tensorMesh, term, cellTuple);
  TensorBasisCacheND trialCache(tensorMesh, tensorMesh->trialDiscretization("u"), false, cellTuple, quadratureDegrees);
  TensorBasisCacheND testCache (tensorMesh, tensorMesh->testDiscretization("v"),  true,  cellTuple, quadratureDegrees);
  auto localContext = dofLayout->volumeContext("u", "v", cellTuple);

  auto bOperator = TensorLocalSchurComplementBuilder::buildVolumeBOperator(term,
                                                                           localContext.trialShape(),
                                                                           localContext.testShape(),
                                                                           trialCache,
                                                                           testCache,
                                                                           {TensorFunction::componentCoordinate(0), TensorFunction::componentCoordinate(1)});
  auto bMatrix = TensorLocalSchurComplementBuilder::denseBMatrix(bOperator);
  auto fullCoefficientValues = coefficientValues(term.coefficient(), trialCache);
  const auto trialLayout = dofLayout->layout("u", TENSOR_TRIAL_SPACE);
  const auto testLayout  = dofLayout->layout("v", TENSOR_TEST_SPACE);
  for (IndexType row=0; row<localContext.testShape().totalDofs(); row++)
  {
    for (IndexType col=0; col<localContext.trialShape().totalDofs(); col++)
    {
      const double expected = explicitCachedBEntry(term,
                                                   dofLayout,
                                                   trialLayout,
                                                   testLayout,
                                                   localContext.trialShape(),
                                                   localContext.testShape(),
                                                   trialCache,
                                                   testCache,
                                                   fullCoefficientValues,
                                                   row,
                                                   col);
      TEST_ASSERT(closeEnough(bMatrix[row * localContext.trialShape().totalDofs() + col], expected, 1e-12, 1e-14));
    }
  }

  auto gDiagonal = TensorLocalSchurComplementBuilder::l2TestMassDiagonal(localContext.testShape(), testCache);
  TEST_EQUALITY(gDiagonal.size(), static_cast<size_t>(localContext.testShape().totalDofs()));
  for (double entry : gDiagonal)
  {
    TEST_ASSERT(entry > 0.0);
  }

  auto kMatrix = TensorLocalSchurComplementBuilder::formDiagonalTestNormSchurComplement(bOperator, gDiagonal);
  for (IndexType row=0; row<localContext.trialShape().totalDofs(); row++)
  {
    for (IndexType col=0; col<localContext.trialShape().totalDofs(); col++)
    {
      TEST_FLOATING_EQUALITY(kMatrix[row * localContext.trialShape().totalDofs() + col],
                             kMatrix[col * localContext.trialShape().totalDofs() + row],
                             1e-12);
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, CachedVlasovLikeVolumeBMatchesExplicitReference )
{
  auto tensorMesh = makeSingleCellTensorMesh({{0.0, 1.0}, {0.0, 1.0}, {-2.0, 2.0}});
  auto dofLayout = tensorMesh->createDofLayout();
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorBilinearTerm term(makeFieldTensorVar("u", 3),
                          makeFieldTensorVar("v", 3),
                          TensorComponentOperatorDescriptor::valueOnly(3),
                          TensorComponentOperatorDescriptor::singleDerivative(3, 0),
                          TensorFunction::constant(-1.0) * TensorFunction::componentCoordinate(2));
  auto quadratureDegrees = quadratureDegreesForTerm(tensorMesh, term, cellTuple);
  TensorBasisCacheND trialCache(tensorMesh, tensorMesh->trialDiscretization("u"), false, cellTuple, quadratureDegrees);
  TensorBasisCacheND testCache (tensorMesh, tensorMesh->testDiscretization("v"),  true,  cellTuple, quadratureDegrees);
  auto localContext = dofLayout->volumeContext("u", "v", cellTuple);

  auto bOperator = TensorLocalSchurComplementBuilder::buildVolumeBOperator(term,
                                                                           localContext.trialShape(),
                                                                           localContext.testShape(),
                                                                           trialCache,
                                                                           testCache,
                                                                           {TensorFunction::constant(-1.0), Teuchos::null, TensorFunction::componentCoordinate(2)});
  auto bMatrix = TensorLocalSchurComplementBuilder::denseBMatrix(bOperator);
  auto fullCoefficientValues = coefficientValues(term.coefficient(), trialCache);
  const auto trialLayout = dofLayout->layout("u", TENSOR_TRIAL_SPACE);
  const auto testLayout  = dofLayout->layout("v", TENSOR_TEST_SPACE);

  for (IndexType row=0; row<localContext.testShape().totalDofs(); row++)
  {
    for (IndexType col=0; col<localContext.trialShape().totalDofs(); col++)
    {
      const double expected = explicitCachedBEntry(term,
                                                   dofLayout,
                                                   trialLayout,
                                                   testLayout,
                                                   localContext.trialShape(),
                                                   localContext.testShape(),
                                                   trialCache,
                                                   testCache,
                                                   fullCoefficientValues,
                                                   row,
                                                   col);
      TEST_ASSERT(closeEnough(bMatrix[row * localContext.trialShape().totalDofs() + col], expected, 1e-11, 1e-13));
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, AffineL2VolumeCellSchurComplementsMatchDenseReference )
{
  auto tensorMesh = makeTwoByTwoAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("local-assembled-affine-l2"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(2.0)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorFunction::componentCoordinate(1)));

  auto blocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                      dofLayout,
                                                                                      form,
                                                                                      "u",
                                                                                      "v");
  TEST_EQUALITY(blocks.size(), static_cast<size_t>(tensorMesh->topology()->activeCellCount()));

  const auto trialLayout = dofLayout->layout("u", TENSOR_TRIAL_SPACE);
  const auto testLayout  = dofLayout->layout("v", TENSOR_TEST_SPACE);
  for (const auto &block : blocks)
  {
    auto localContext = dofLayout->volumeContext("u", "v", block.cellTuple);
    TEST_ASSERT(block.trialShape == localContext.trialShape());
    TEST_ASSERT(block.testShape  == localContext.testShape());
    TEST_EQUALITY(block.bMatrix.size(), static_cast<size_t>(block.testShape.totalDofs() * block.trialShape.totalDofs()));
    TEST_EQUALITY(block.gDiagonal.size(), static_cast<size_t>(block.testShape.totalDofs()));
    TEST_EQUALITY(block.kMatrix.size(), static_cast<size_t>(block.trialShape.totalDofs() * block.trialShape.totalDofs()));

    std::vector<double> referenceB(block.bMatrix.size(), 0.0);
    for (const auto &term : form->terms())
    {
      auto quadratureDegrees = quadratureDegreesForTerm(tensorMesh, term, block.cellTuple);
      TensorBasisCacheND trialCache(tensorMesh, tensorMesh->trialDiscretization("u"), false, block.cellTuple, quadratureDegrees);
      TensorBasisCacheND testCache (tensorMesh, tensorMesh->testDiscretization("v"), true,  block.cellTuple, quadratureDegrees);
      auto fullCoefficientValues = coefficientValues(term.coefficient(), trialCache);
      for (IndexType row=0; row<block.testShape.totalDofs(); row++)
      {
        for (IndexType col=0; col<block.trialShape.totalDofs(); col++)
        {
          referenceB[row * block.trialShape.totalDofs() + col] += explicitCachedBEntry(term,
                                                                                       dofLayout,
                                                                                       trialLayout,
                                                                                       testLayout,
                                                                                       block.trialShape,
                                                                                       block.testShape,
                                                                                       trialCache,
                                                                                       testCache,
                                                                                       fullCoefficientValues,
                                                                                       row,
                                                                                       col);
        }
      }
    }

    for (size_t entryOrdinal=0; entryOrdinal<referenceB.size(); entryOrdinal++)
    {
      TEST_ASSERT(closeEnough(block.bMatrix[entryOrdinal], referenceB[entryOrdinal], 1e-12, 1e-14));
    }

    std::vector<int> gReferenceQuadratureDegrees(block.testShape.componentCount(), 1);
    for (int d=0; d<block.testShape.componentCount(); d++)
    {
      const int testPolyOrder = tensorMesh->effectivePolyOrder(tensorMesh->testDiscretization("v"), block.cellTuple[d], d, true);
      gReferenceQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
    }
    TensorBasisCacheND gTestCache(tensorMesh, tensorMesh->testDiscretization("v"), true, block.cellTuple, gReferenceQuadratureDegrees);
    auto referenceG = TensorLocalSchurComplementBuilder::l2TestMassDiagonal(block.testShape, gTestCache);
    auto referenceK = TensorLocalSchurComplementBuilder::formDiagonalTestNormSchurComplement(referenceB,
                                                                                            block.testShape.totalDofs(),
                                                                                            block.trialShape.totalDofs(),
                                                                                            referenceG);
    for (IndexType testOrdinal=0; testOrdinal<block.testShape.totalDofs(); testOrdinal++)
    {
      TEST_ASSERT(block.gDiagonal[testOrdinal] > 0.0);
      TEST_FLOATING_EQUALITY(block.gDiagonal[testOrdinal], referenceG[testOrdinal], 1e-14);
    }
    for (size_t entryOrdinal=0; entryOrdinal<referenceK.size(); entryOrdinal++)
    {
      TEST_ASSERT(closeEnough(block.kMatrix[entryOrdinal], referenceK[entryOrdinal], 1e-12, 1e-14));
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, RejectsInvalidGDiagonal )
{
  auto shape = makeShape({2});
  TensorProductLocalOperator bOperator(shape,
                                       shape,
                                       {TensorComponentLocalMatrix(2, 2, {1.0, 0.0,
                                                                          0.0, 1.0})});
  TEST_THROW(TensorLocalSchurComplementBuilder::formDiagonalTestNormSchurComplement(bOperator, {1.0}), std::invalid_argument);
  TEST_THROW(TensorLocalSchurComplementBuilder::formDiagonalTestNormSchurComplement(bOperator, {1.0, 0.0}), std::invalid_argument);
}
}
