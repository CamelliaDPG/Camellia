#include "Teuchos_UnitTestHarness.hpp"

#include "TensorVar.h"
#include "Var.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorVar, ConstructFieldTensorVar )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "f_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, "f_t", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));

  TensorVar tensorVar("f", TENSOR_FIELD, {vx, vt});

  TEST_EQUALITY(tensorVar.name(), "f");
  TEST_ASSERT(tensorVar.isField());
  TEST_ASSERT(!tensorVar.isTrace());
  TEST_EQUALITY(tensorVar.componentCount(), 2);
  TEST_EQUALITY(tensorVar.componentVar(0), vx);
  TEST_EQUALITY(tensorVar.componentVar(1), vt);
}

TEUCHOS_UNIT_TEST( TensorVar, ConstructTraceTensorVar )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "fhat_x", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, "fhat_t", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));

  TensorVar tensorVar("fhat", TENSOR_TRACE, {vx, vt});

  TEST_ASSERT(!tensorVar.isField());
  TEST_ASSERT(tensorVar.isTrace());
  TEST_EQUALITY(tensorVar.componentCount(), 2);
}

TEUCHOS_UNIT_TEST( TensorVar, TraceKindAllowsFluxComponentVar )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "sigmahat_x", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  TensorVar tensorVar("sigmahat", TENSOR_TRACE, {vx});
  TEST_ASSERT(tensorVar.isTrace());
}

TEUCHOS_UNIT_TEST( TensorVar, CopyWithNewName )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "u_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, "u_t", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));

  TensorVar tensorVar("u", TENSOR_FIELD, {vx, vt});
  auto copied = tensorVar.copyWithNewName("v");

  TEST_EQUALITY(copied->name(), "v");
  TEST_ASSERT(copied->isField());
  TEST_EQUALITY(copied->componentCount(), 2);
  TEST_ASSERT(copied->componentVar(0) != vx);
  TEST_ASSERT(copied->componentVar(1) != vt);
}

TEUCHOS_UNIT_TEST( TensorVar, RejectMismatchedFieldKind )
{
  VarPtr traceVar = Teuchos::rcp(new Var(0, 0, "uhat_x", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  TEST_THROW(TensorVar("u", TENSOR_FIELD, {traceVar}), std::invalid_argument);
}
}
