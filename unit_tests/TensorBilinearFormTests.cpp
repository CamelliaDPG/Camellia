#include "Teuchos_UnitTestHarness.hpp"

#include "TensorBilinearForm.h"
#include "TensorOperatorFactory.h"
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

TensorMeshPtr makeTensorMeshForBilinearFormTests()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }, false, 2));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));

  auto trialVar = makeFieldTensorVar("u");
  auto testVar  = makeFieldTensorVar("v");

  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(trialVar, Teuchos::null, std::vector<int>{0,0})));
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(testVar,  Teuchos::null, std::vector<int>{0,0})));
  return tensorMesh;
}

TEUCHOS_UNIT_TEST( TensorBilinearForm, ValueOnlyDescriptorHasNoDerivatives )
{
  auto descriptor = TensorComponentOperatorDescriptor::valueOnly(3);
  TEST_EQUALITY(descriptor.componentCount(), 3);
  TEST_ASSERT(descriptor.isValueOnly());
  TEST_ASSERT(!descriptor.hasDerivatives());
  TEST_ASSERT(descriptor.isIdentityValueAction());
}

TEUCHOS_UNIT_TEST( TensorBilinearForm, SingleDerivativeDescriptorMarksRequestedComponent )
{
  auto descriptor = TensorComponentOperatorDescriptor::singleDerivative(3, 1);
  TEST_ASSERT(descriptor.hasDerivatives());
  TEST_ASSERT(descriptor.takesDerivativeInComponent(1));
  TEST_ASSERT(!descriptor.takesDerivativeInComponent(0));
  TEST_EQUALITY(descriptor.op(1), TENSOR_COMPONENT_OP_DERIVATIVE);
}

TEUCHOS_UNIT_TEST( TensorBilinearForm, TermCarriesCoefficientDependencyAndBackendMetadata )
{
  auto trialVar = makeFieldTensorVar("u");
  auto testVar  = makeFieldTensorVar("v");
  auto coefficient = TensorFunction::componentCoordinate(1);

  TensorBilinearTerm term(trialVar,
                          testVar,
                          TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                          TensorComponentOperatorDescriptor::valueOnly(2),
                          coefficient,
                          TENSOR_BILINEAR_TERM_VOLUME,
                          TensorOperatorCapabilitySet::matrixFreeAndPAMatrix());

  TEST_ASSERT(term.isVolumeTerm());
  TEST_ASSERT(!term.isBoundaryTerm());
  TEST_ASSERT(term.isMatrixFreeCandidate());
  TEST_ASSERT(term.isPAMatrixCandidate());
  TEST_ASSERT(term.coefficient()->dependencyScope().dependsOnComponent(1));
}

TEUCHOS_UNIT_TEST( TensorBilinearForm, FormCollectsVariableNamesAndCoupledComponents )
{
  auto trialVar = makeFieldTensorVar("u");
  auto testVar  = makeFieldTensorVar("v");
  auto coefficient = TensorFunction::componentCoordinate(0) * TensorFunction::componentCoordinate(1);

  auto form = Teuchos::rcp(new TensorBilinearForm("advection"));
  form->addTerm(TensorBilinearTerm(trialVar,
                                   testVar,
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   coefficient));

  TEST_EQUALITY(form->termCount(), 1);
  TEST_ASSERT(form->trialVarNames().find("u") != form->trialVarNames().end());
  TEST_ASSERT(form->testVarNames().find("v") != form->testVarNames().end());
  auto coupled = form->coupledComponentOrdinals();
  TEST_ASSERT(coupled.find(0) != coupled.end());
  TEST_ASSERT(coupled.find(1) != coupled.end());
}

TEUCHOS_UNIT_TEST( TensorBilinearForm, ValidateAgainstTensorMeshRequiresRegisteredVars )
{
  auto tensorMesh = makeTensorMeshForBilinearFormTests();
  auto form = Teuchos::rcp(new TensorBilinearForm("mass"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u"),
                                   makeFieldTensorVar("v"),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0)));

  TEST_NOTHROW(form->validateAgainst(tensorMesh));
}

TEUCHOS_UNIT_TEST( TensorBilinearForm, MatrixFreeFactoryAcceptsSupportedDescriptorSubset )
{
  auto tensorMesh = makeTensorMeshForBilinearFormTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("stiffness"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u"),
                                   makeFieldTensorVar("v"),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorFunction::constant(1.0)));

  MatrixFreeTensorOperatorFactory factory;
  TEST_ASSERT(factory.canBuild(tensorMesh, dofLayout, form,
                               TENSOR_TRIAL_SPACE, "u",
                               TENSOR_TEST_SPACE, "v"));
}

TEUCHOS_UNIT_TEST( TensorBilinearForm, BuilderConstructsCommonDerivativeValuePattern )
{
  TensorBilinearFormBuilder builder("builder-test");
  auto form = builder.addDerivativeValueTerm(makeFieldTensorVar("u"),
                                             makeFieldTensorVar("v"),
                                             1,
                                             TensorFunction::componentCoordinate(0))
                     .build();

  TEST_EQUALITY(form->termCount(), 1);
  TEST_ASSERT(form->term(0).trialOperator().takesDerivativeInComponent(1));
  TEST_ASSERT(!form->term(0).testOperator().hasDerivatives());
}

TEUCHOS_UNIT_TEST( TensorBilinearForm, AggregateBackendSupportReflectsAllTerms )
{
  auto form = Teuchos::rcp(new TensorBilinearForm("backend-check"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u"),
                                   makeFieldTensorVar("v"),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0),
                                   TENSOR_BILINEAR_TERM_VOLUME,
                                   TensorOperatorCapabilitySet::matrixFreeAndPAMatrix()));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u"),
                                   makeFieldTensorVar("v"),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0),
                                   TENSOR_BILINEAR_TERM_VOLUME,
                                   TensorOperatorCapabilitySet::matrixFreeOnly()));

  TEST_ASSERT(form->allTermsSupport(TENSOR_OPERATOR_BACKEND_MATRIX_FREE));
  TEST_ASSERT(!form->allTermsSupport(TENSOR_OPERATOR_BACKEND_PAMATRIX));
}
}
