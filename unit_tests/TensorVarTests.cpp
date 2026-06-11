#include "Teuchos_UnitTestHarness.hpp"

#include "TensorVar.h"
#include "Var.h"

using namespace Camellia;

namespace
{
TensorVar makeField1X1T1V(const std::string &name)
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, name + "_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, name + "_t", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vv = Teuchos::rcp(new Var(2, 0, name + "_v", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  return TensorVar(name, TENSOR_FIELD, {vx, vt, vv});
}

TEUCHOS_UNIT_TEST( TensorVar, ConstructFieldTensorVar )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "f_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, "f_t", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));

  TensorVar tensorVar("f", TENSOR_FIELD, {vx, vt});

  TEST_EQUALITY(tensorVar.name(), "f");
  TEST_ASSERT(tensorVar.isField());
  TEST_ASSERT(!tensorVar.isTrace());
  TEST_ASSERT(!tensorVar.hasTraceComponentOrdinal());
  TEST_EQUALITY(tensorVar.componentCount(), 2);
  TEST_EQUALITY(tensorVar.componentVar(0), vx);
  TEST_EQUALITY(tensorVar.componentVar(1), vt);
}

TEUCHOS_UNIT_TEST( TensorVar, ConstructTraceTensorVar )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "fhat_x", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, "fhat_t", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));

  TensorVar tensorVar("fhat", TENSOR_TRACE, {vx, vt}, 0);

  TEST_ASSERT(!tensorVar.isField());
  TEST_ASSERT(tensorVar.isTrace());
  TEST_ASSERT(tensorVar.hasTraceComponentOrdinal());
  TEST_EQUALITY(tensorVar.traceComponentOrdinal(), 0);
  TEST_EQUALITY(tensorVar.componentCount(), 2);
}

TEUCHOS_UNIT_TEST( TensorVar, TraceKindAllowsFluxComponentVar )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "sigmahat_x", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  TensorVar tensorVar("sigmahat", TENSOR_TRACE, {vx}, 0);
  TEST_ASSERT(tensorVar.isTrace());
  TEST_EQUALITY(tensorVar.traceComponentOrdinal(), 0);
}

TEUCHOS_UNIT_TEST( TensorVar, CopyWithNewName )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "u_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, "u_t", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));

  TensorVar tensorVar("u", TENSOR_FIELD, {vx, vt});
  auto copied = tensorVar.copyWithNewName("v");

  TEST_EQUALITY(copied->name(), "v");
  TEST_ASSERT(copied->isField());
  TEST_ASSERT(!copied->hasTraceComponentOrdinal());
  TEST_EQUALITY(copied->componentCount(), 2);
  TEST_ASSERT(copied->componentVar(0) != vx);
  TEST_ASSERT(copied->componentVar(1) != vt);
}

TEUCHOS_UNIT_TEST( TensorVar, ValueTraceSemanticHelpersRecognizeTraceButNotFlux )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "fhat_x", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, "fhat_t", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));

  TensorVar tensorVar("fhat", TENSOR_TRACE, {vx, vt}, 0);

  TEST_ASSERT(tensorVar.isTrace());
  TEST_ASSERT(tensorVar.hasTraceLikeComponentVar());
  TEST_ASSERT(!tensorVar.hasFluxLikeComponentVar());
  TEST_ASSERT(tensorVar.isValueTrace());
  TEST_ASSERT(!tensorVar.isFluxLikeTrace());
}

TEUCHOS_UNIT_TEST( TensorVar, FluxTraceSemanticHelpersRecognizeFluxLikeTrace )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "sigmahat_x", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  TensorVar tensorVar("sigmahat", TENSOR_TRACE, {vx}, 0);

  TEST_ASSERT(tensorVar.isTrace());
  TEST_ASSERT(tensorVar.hasFluxLikeComponentVar());
  TEST_ASSERT(!tensorVar.hasTraceLikeComponentVar());
  TEST_ASSERT(tensorVar.isFluxLikeTrace());
  TEST_ASSERT(!tensorVar.isValueTrace());
}

TEUCHOS_UNIT_TEST( TensorVar, CopyWithNewNamePreservesTraceComponentOrdinal )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "fhat_x", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, "fhat_t", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));

  TensorVar tensorVar("fhat", TENSOR_TRACE, {vx, vt}, 1);
  auto copied = tensorVar.copyWithNewName("ghat");

  TEST_ASSERT(copied->isTrace());
  TEST_ASSERT(copied->hasTraceComponentOrdinal());
  TEST_EQUALITY(copied->traceComponentOrdinal(), 1);
}

TEUCHOS_UNIT_TEST( TensorVar, ConstructFieldTensorVarFor1X1T1V )
{
  auto tensorVar = makeField1X1T1V("f");
  TEST_ASSERT(tensorVar.isField());
  TEST_EQUALITY(tensorVar.componentCount(), 3);
  TEST_EQUALITY(tensorVar.componentVar(0)->space(), L2);
  TEST_EQUALITY(tensorVar.componentVar(1)->space(), L2);
  TEST_EQUALITY(tensorVar.componentVar(2)->space(), L2);
}

TEUCHOS_UNIT_TEST( TensorVar, ConstructTraceTensorVarFor1X1T1VTemporalTrace )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "that_x", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, "that_t", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vv = Teuchos::rcp(new Var(2, 0, "that_v", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));

  TensorVar tensorVar("that", TENSOR_TRACE, {vx, vt, vv}, 1);
  TEST_ASSERT(tensorVar.isTrace());
  TEST_EQUALITY(tensorVar.traceComponentOrdinal(), 1);
  TEST_ASSERT(tensorVar.isValueTrace());
}

TEUCHOS_UNIT_TEST( TensorVar, ConstructFluxTraceTensorVarFor1X1T1VSpatialTrace )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "tn_x", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, "tn_t", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vv = Teuchos::rcp(new Var(2, 0, "tn_v", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));

  TensorVar tensorVar("tn", TENSOR_TRACE, {vx, vt, vv}, 0);
  TEST_ASSERT(tensorVar.isTrace());
  TEST_EQUALITY(tensorVar.traceComponentOrdinal(), 0);
  TEST_ASSERT(tensorVar.isFluxLikeTrace());
}

TEUCHOS_UNIT_TEST( TensorVar, RejectMismatchedFieldKind )
{
  VarPtr traceVar = Teuchos::rcp(new Var(0, 0, "uhat_x", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  TEST_THROW(TensorVar("u", TENSOR_FIELD, {traceVar}), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorVar, RejectFieldWithTraceComponentOrdinal )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "u_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  TEST_THROW(TensorVar("u", TENSOR_FIELD, {vx}, 0), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorVar, RejectTraceWithoutTraceComponentOrdinal )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "fhat_x", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  TEST_THROW(TensorVar("fhat", TENSOR_TRACE, {vx}), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorVar, RejectTraceWithOutOfRangeTraceComponentOrdinal )
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, "fhat_x", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  TEST_THROW(TensorVar("fhat", TENSOR_TRACE, {vx}, 1), std::invalid_argument);
}
}
