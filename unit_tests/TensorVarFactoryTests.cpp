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
}
