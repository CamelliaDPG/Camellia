#include "Teuchos_UnitTestHarness.hpp"

#include "TensorOperatorFactory.h"
#include "TensorBilinearForm.h"
#include "TensorBilinearFormBuilder.h"
#include "TensorDofLayout.h"
#include "TensorMesh.h"
#include "Var.h"

using namespace Camellia;

namespace
{
TensorVarPtr makeFieldTensorVar(const std::string &name)
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, name + "_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, name + "_t", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  return Teuchos::rcp(new TensorVar(name, TENSOR_FIELD, std::vector<VarPtr>{vx, vt}));
}

TensorMeshPtr makeTensorMeshForOperatorTests()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }, false, 3));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));

  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("u"), Teuchos::null, std::vector<int>{0,0})));
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("v"), Teuchos::null, std::vector<int>{0,0})));
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
}
