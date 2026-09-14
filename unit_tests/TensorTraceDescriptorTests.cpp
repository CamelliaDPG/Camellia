#include "Teuchos_UnitTestHarness.hpp"

#include "ConfigurationSpaceFormulationMaxwell1D2V.h"
#include "TensorTraceDescriptor.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorTraceDescriptor, NonconformingL2TraceCarriesOutwardNormalIntent )
{
  auto tn = TensorTraceDescriptor::nonconformingL2SkeletonTrace("f", "t_n", 1);

  TEST_EQUALITY(tn.fieldName(), "f");
  TEST_EQUALITY(tn.traceName(), "t_n");
  TEST_ASSERT(tn.isNonconformingL2Trace());
  TEST_EQUALITY(tn.supportKind(), TENSOR_TRACE_SUPPORT_COMPONENT_SKELETON);
  TEST_EQUALITY(tn.orthogonalComponentOrdinal(), 1);
  TEST_ASSERT(tn.usesOutwardNormalSign());
}

TEUCHOS_UNIT_TEST( TensorTraceDescriptor, H1PointValueTraceDoesNotUseOutwardNormalSign )
{
  auto exHat = TensorTraceDescriptor::h1PointValueTrace1D(ConfigurationSpaceFormulationMaxwell1D2V::S_EX,
                                                         ConfigurationSpaceFormulationMaxwell1D2V::S_EX_HAT);

  TEST_EQUALITY(exHat.fieldName(), ConfigurationSpaceFormulationMaxwell1D2V::S_EX);
  TEST_EQUALITY(exHat.traceName(), ConfigurationSpaceFormulationMaxwell1D2V::S_EX_HAT);
  TEST_ASSERT(exHat.isH1PointValueTrace());
  TEST_EQUALITY(exHat.supportKind(), TENSOR_TRACE_SUPPORT_1D_INTERFACE_POINTS);
  TEST_EQUALITY(exHat.orthogonalComponentOrdinal(), 0);
  TEST_ASSERT(!exHat.usesOutwardNormalSign());
}

TEUCHOS_UNIT_TEST( TensorTraceDescriptor, Maxwell1D2VH1TraceTargetsMatchConfigurationFormulationNames )
{
  auto traces = TensorTraceDescriptor::maxwell1D2VH1PointValueTraces(ConfigurationSpaceFormulationMaxwell1D2V::S_EX,
                                                                    ConfigurationSpaceFormulationMaxwell1D2V::S_EX_HAT,
                                                                    ConfigurationSpaceFormulationMaxwell1D2V::S_BY,
                                                                    ConfigurationSpaceFormulationMaxwell1D2V::S_BY_HAT,
                                                                    ConfigurationSpaceFormulationMaxwell1D2V::S_PHI,
                                                                    ConfigurationSpaceFormulationMaxwell1D2V::S_PHI_HAT,
                                                                    ConfigurationSpaceFormulationMaxwell1D2V::S_A,
                                                                    ConfigurationSpaceFormulationMaxwell1D2V::S_A_HAT);

  TEST_EQUALITY(traces.size(), 4);
  TEST_EQUALITY(traces[0].fieldName(), ConfigurationSpaceFormulationMaxwell1D2V::S_EX);
  TEST_EQUALITY(traces[1].fieldName(), ConfigurationSpaceFormulationMaxwell1D2V::S_BY);
  TEST_EQUALITY(traces[2].fieldName(), ConfigurationSpaceFormulationMaxwell1D2V::S_PHI);
  TEST_EQUALITY(traces[3].fieldName(), ConfigurationSpaceFormulationMaxwell1D2V::S_A);

  for (const auto &trace : traces)
  {
    TEST_ASSERT(trace.isH1PointValueTrace());
    TEST_EQUALITY(trace.supportKind(), TENSOR_TRACE_SUPPORT_1D_INTERFACE_POINTS);
    TEST_ASSERT(!trace.usesOutwardNormalSign());
  }
}

TEUCHOS_UNIT_TEST( TensorTraceDescriptor, RejectsInvalidTraceMetadata )
{
  TEST_THROW(TensorTraceDescriptor::h1PointValueTrace1D("", "u_hat"), std::invalid_argument);
  TEST_THROW(TensorTraceDescriptor::h1PointValueTrace1D("u", ""), std::invalid_argument);
  TEST_THROW(TensorTraceDescriptor::h1PointValueTrace1D("u", "u_hat", -1), std::invalid_argument);
  TEST_THROW(TensorTraceDescriptor("u",
                                   "u_hat",
                                   TENSOR_TRACE_H1_POINT_VALUE,
                                   TENSOR_TRACE_SUPPORT_1D_INTERFACE_POINTS,
                                   0,
                                   true),
             std::invalid_argument);
}
}
