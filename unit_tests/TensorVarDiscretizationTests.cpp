#include "Teuchos_UnitTestHarness.hpp"

#include "TensorVarDiscretization.h"
#include "Var.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorVarDiscretization, ConstructWithPerComponentPolyOrderDeltas )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "u_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, "u_t", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  auto tensorVar = Teuchos::rcp(new TensorVar("u", TENSOR_FIELD, std::vector<VarPtr>{vx, vt}));
  auto basisFamily = BasisFamily::hierarchicalBasisFamily();

  TensorVarDiscretization discretization(tensorVar, basisFamily, {0, 2});

  TEST_EQUALITY(discretization.tensorVar(), tensorVar);
  TEST_ASSERT(discretization.hasBasisFamilyOverride());
  TEST_EQUALITY(discretization.basisFamilyOverride(), basisFamily);
  TEST_EQUALITY(discretization.componentCount(), 2);
  TEST_EQUALITY(discretization.componentPolyOrderDelta(0), 0);
  TEST_EQUALITY(discretization.componentPolyOrderDelta(1), 2);
}

TEUCHOS_UNIT_TEST( TensorVarDiscretization, ConstructWithoutBasisFamilyOverride )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "u_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  auto tensorVar = Teuchos::rcp(new TensorVar("u", TENSOR_FIELD, std::vector<VarPtr>{vx}));

  TensorVarDiscretization discretization(tensorVar, Teuchos::null, {1});

  TEST_ASSERT(!discretization.hasBasisFamilyOverride());
  TEST_EQUALITY(discretization.componentPolyOrderDelta(0), 1);
}

TEUCHOS_UNIT_TEST( TensorVarDiscretization, RejectWrongNumberOfPolyOrderDeltas )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "u_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, "u_t", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  auto tensorVar = Teuchos::rcp(new TensorVar("u", TENSOR_FIELD, std::vector<VarPtr>{vx, vt}));

  TEST_THROW(TensorVarDiscretization(tensorVar, Teuchos::null, {0}), std::invalid_argument);
}
}
