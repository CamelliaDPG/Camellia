#include "Teuchos_UnitTestHarness.hpp"

#include "TensorVarFactory.h"
#include "Var.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorVarFactory, RegisterAndLookupTensorVar )
{
  TensorVarFactory factory;
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "u_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  auto tensorVar = Teuchos::rcp(new TensorVar("u", TENSOR_FIELD, std::vector<VarPtr>{vx}));

  factory.addTensorVar(tensorVar);

  TEST_ASSERT(factory.hasTensorVar("u"));
  TEST_EQUALITY(factory.tensorVar("u"), tensorVar);
  TEST_EQUALITY(factory.tensorVars().size(), 1);
}

TEUCHOS_UNIT_TEST( TensorVarFactory, RejectDuplicateName )
{
  TensorVarFactory factory;
  VarPtr vx0 = Teuchos::rcp(new Var(0, 0, "u_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vx1 = Teuchos::rcp(new Var(1, 0, "u2_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  auto tensorVar0 = Teuchos::rcp(new TensorVar("u", TENSOR_FIELD, std::vector<VarPtr>{vx0}));
  auto tensorVar1 = Teuchos::rcp(new TensorVar("u", TENSOR_FIELD, std::vector<VarPtr>{vx1}));

  factory.addTensorVar(tensorVar0);
  TEST_THROW(factory.addTensorVar(tensorVar1), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorVarFactory, DistinctCopiedNamesCanCoexist )
{
  TensorVarFactory factory;
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "u_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  auto trialVar = Teuchos::rcp(new TensorVar("u", TENSOR_FIELD, std::vector<VarPtr>{vx}));
  auto testVar = trialVar->copyWithNewName("v");

  factory.addTensorVar(trialVar);
  factory.addTensorVar(testVar);

  TEST_ASSERT(factory.hasTensorVar("u"));
  TEST_ASSERT(factory.hasTensorVar("v"));
  TEST_EQUALITY(factory.tensorVars().size(), 2);
}

TEUCHOS_UNIT_TEST( TensorVarFactory, DirectionalTraceFamilyCreatesPerComponentFluxTraces )
{
  std::vector<VarPtr> componentVars;
  componentVars.push_back(Teuchos::rcp(new Var(10, 0, "tn_x_component", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS)));
  componentVars.push_back(Teuchos::rcp(new Var(11, 0, "tn_t_component", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS)));
  componentVars.push_back(Teuchos::rcp(new Var(12, 0, "tn_ux_component", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS)));
  componentVars.push_back(Teuchos::rcp(new Var(13, 0, "tn_uy_component", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS)));

  auto traces = TensorVarFactory::directionalTraceFamily("tn", {"x", "t", "ux", "uy"}, componentVars);

  TEST_EQUALITY(traces.size(), 4);
  for (int d=0; d<4; d++)
  {
    TEST_ASSERT(traces[d]->isTrace());
    TEST_ASSERT(traces[d]->isFluxLikeTrace());
    TEST_EQUALITY(traces[d]->traceComponentOrdinal(), d);
    TEST_EQUALITY(traces[d]->componentCount(), 4);
    TEST_EQUALITY(traces[d]->componentVar(d)->ID(), componentVars[d]->ID());
    TEST_ASSERT(traces[d]->componentVar(d) != componentVars[d]);
  }
  TEST_EQUALITY(traces[0]->name(), "tn_x");
  TEST_EQUALITY(traces[1]->name(), "tn_t");
  TEST_EQUALITY(traces[2]->name(), "tn_ux");
  TEST_EQUALITY(traces[3]->name(), "tn_uy");
}

TEUCHOS_UNIT_TEST( TensorVarFactory, AddDirectionalTraceFamilyRegistersVariables )
{
  TensorVarFactory factory;
  std::vector<VarPtr> componentVars;
  componentVars.push_back(Teuchos::rcp(new Var(20, 0, "tn_x_component", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS)));
  componentVars.push_back(Teuchos::rcp(new Var(21, 0, "tn_t_component", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS)));

  auto traces = factory.addDirectionalTraceFamily("tn", {"x", "t"}, componentVars);

  TEST_EQUALITY(traces.size(), 2);
  TEST_ASSERT(factory.hasTensorVar("tn_x"));
  TEST_ASSERT(factory.hasTensorVar("tn_t"));
  TEST_EQUALITY(factory.tensorVar("tn_x")->traceComponentOrdinal(), 0);
  TEST_EQUALITY(factory.tensorVar("tn_t")->traceComponentOrdinal(), 1);
  TEST_EQUALITY(factory.tensorVars().size(), 2);
  TEST_THROW(factory.addDirectionalTraceFamily("tn", {"x", "t"}, componentVars), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorVarFactory, DirectionalTraceFamilyRejectsInvalidInputs )
{
  VarPtr traceVar = Teuchos::rcp(new Var(30, 0, "tn_x_component", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  TEST_THROW(TensorVarFactory::directionalTraceFamily("", {"x"}, {traceVar}), std::invalid_argument);
  TEST_THROW(TensorVarFactory::directionalTraceFamily("tn", {}, {}), std::invalid_argument);
  TEST_THROW(TensorVarFactory::directionalTraceFamily("tn", {"x", "t"}, {traceVar}), std::invalid_argument);
  TEST_THROW(TensorVarFactory::directionalTraceFamily("tn", {""}, {traceVar}), std::invalid_argument);
  TEST_THROW(TensorVarFactory::directionalTraceFamily("tn", {"x"}, {Teuchos::null}), std::invalid_argument);
}
}
