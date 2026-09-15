#include "Teuchos_UnitTestHarness.hpp"

#include "ConfigurationSpaceFormulationMaxwell1D2V.h"
#include "TensorDofLayout.h"
#include "TensorMesh.h"
#include "TensorTraceDescriptor.h"
#include "TensorTraceLayout.h"
#include "Var.h"

using namespace Camellia;

namespace
{
void addH1PointTraceTrialDiscretization(TensorMeshPtr tensorMesh,
                                        const std::string &traceName,
                                        int varID)
{
  VarPtr traceComponent = Teuchos::rcp(new Var(varID, 0, traceName + "_component", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  auto traceVar = Teuchos::rcp(new TensorVar(traceName, TENSOR_TRACE, std::vector<VarPtr>{traceComponent}, 0));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(traceVar, Teuchos::null, std::vector<int>{0})));
}

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

TEUCHOS_UNIT_TEST( TensorTraceDescriptor, MaxwellFormulationRegistersH1PointTracesOnTensorMesh )
{
  auto component = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 0.5, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 1));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 0));

  ConfigurationSpaceFormulationMaxwell1D2V formulation(1.0, 1.0);
  formulation.registerTensorH1PointTraceDescriptors(tensorMesh);

  for (const auto &descriptor : ConfigurationSpaceFormulationMaxwell1D2V::tensorH1PointTraceDescriptors1D())
  {
    TEST_ASSERT(tensorMesh->hasTraceDescriptor(descriptor.traceName()));
    const auto &registered = tensorMesh->traceDescriptor(descriptor.traceName());
    TEST_EQUALITY(registered.fieldName(), descriptor.fieldName());
    TEST_EQUALITY(registered.supportKind(), TENSOR_TRACE_SUPPORT_1D_INTERFACE_POINTS);
    TEST_ASSERT(!registered.usesOutwardNormalSign());
  }

  TEST_NOTHROW(formulation.registerTensorH1PointTraceDescriptors(tensorMesh));
}

TEUCHOS_UNIT_TEST( TensorTraceDescriptor, MaxwellFormulationRetrievesConcreteH1PointTraceLayouts )
{
  auto component = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 0.5, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 1));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 0));

  ConfigurationSpaceFormulationMaxwell1D2V formulation(1.0, 1.0);
  formulation.registerTensorH1PointTraceDescriptors(tensorMesh);

  int varID = 1000;
  for (const auto &descriptor : ConfigurationSpaceFormulationMaxwell1D2V::tensorH1PointTraceDescriptors1D())
  {
    addH1PointTraceTrialDiscretization(tensorMesh, descriptor.traceName(), varID++);
  }

  TensorDofLayoutPtr dofLayout = tensorMesh->createDofLayout();
  auto layouts = formulation.h1PointTraceLayouts1D(dofLayout);
  TEST_EQUALITY(layouts.size(), 4);
  for (const auto &layout : layouts)
  {
    TEST_ASSERT(layout.descriptor().isH1PointValueTrace());
    TEST_EQUALITY(layout.interfacePointCount(), component->vertexCount());
    TEST_EQUALITY(layout.valueDofOrdinal(0), 0);
    TEST_EQUALITY(layout.valueDofOrdinal(static_cast<int>(component->vertexCount()) - 1), static_cast<int>(component->vertexCount()) - 1);
    TEST_ASSERT(!layout.descriptor().usesOutwardNormalSign());
  }

  auto exLayout = formulation.h1PointTraceLayout1D(dofLayout, ConfigurationSpaceFormulationMaxwell1D2V::S_EX);
  TEST_EQUALITY(exLayout.descriptor().fieldName(), ConfigurationSpaceFormulationMaxwell1D2V::S_EX);
  TEST_EQUALITY(exLayout.descriptor().traceName(), ConfigurationSpaceFormulationMaxwell1D2V::S_EX_HAT);
  TEST_EQUALITY(exLayout.endpointValueDofOrdinal(0, 1, TENSOR_TRACE_INTERFACE_LOWER), 0);
  TEST_EQUALITY(exLayout.endpointValueDofOrdinal(0, 1, TENSOR_TRACE_INTERFACE_UPPER), 1);
  TEST_THROW(formulation.h1PointTraceLayout1D(dofLayout, "missing Maxwell field"), std::invalid_argument);
  TEST_THROW(formulation.h1PointTraceLayout1D(Teuchos::null, ConfigurationSpaceFormulationMaxwell1D2V::S_EX), std::invalid_argument);
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
