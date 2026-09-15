#include "Teuchos_UnitTestHarness.hpp"

#include "ConfigurationSpaceFormulationMaxwell1D2V.h"
#include "TensorDofLayout.h"
#include "TensorMesh.h"
#include "TensorTraceLayout.h"
#include "Var.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorTraceLayout, H1PointTraceMapsInterfaceDofsAndEndpointValuesWithoutNormalSign )
{
  auto descriptor = TensorTraceDescriptor::h1PointValueTrace1D("E_x", "E_x_hat");
  TensorH1PointTraceLayout1D layout(descriptor, 3);

  TEST_EQUALITY(layout.descriptor().fieldName(), "E_x");
  TEST_EQUALITY(layout.descriptor().traceName(), "E_x_hat");
  TEST_EQUALITY(layout.interfacePointCount(), 3);
  TEST_EQUALITY(layout.valueDofOrdinal(0), 0);
  TEST_EQUALITY(layout.valueDofOrdinal(2), 2);
  TEST_EQUALITY(layout.endpointValueDofOrdinal(0, 1, TENSOR_TRACE_INTERFACE_LOWER), 0);
  TEST_EQUALITY(layout.endpointValueDofOrdinal(0, 1, TENSOR_TRACE_INTERFACE_UPPER), 1);
  TEST_THROW(layout.endpointValueDofOrdinal(0, 3, TENSOR_TRACE_INTERFACE_UPPER), std::invalid_argument);
  TEST_FLOATING_EQUALITY(layout.referencePointForSide(TENSOR_TRACE_INTERFACE_LOWER), -1.0, 1e-14);
  TEST_FLOATING_EQUALITY(layout.referencePointForSide(TENSOR_TRACE_INTERFACE_UPPER),  1.0, 1e-14);
  TEST_FLOATING_EQUALITY(layout.adaptEndpointValue(4.0, 7.0, TENSOR_TRACE_INTERFACE_LOWER), 4.0, 1e-14);
  TEST_FLOATING_EQUALITY(layout.adaptEndpointValue(4.0, 7.0, TENSOR_TRACE_INTERFACE_UPPER), 7.0, 1e-14);
  TEST_THROW(layout.valueDofOrdinal(3), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorTraceLayout, RejectsNonH1TraceDescriptor )
{
  auto descriptor = TensorTraceDescriptor::nonconformingL2SkeletonTrace("f", "t_n", 0);
  TEST_THROW(TensorH1PointTraceLayout1D layout(descriptor, 1), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorTraceLayout, MaxwellH1PointTracesAreContinuousAcrossSharedInterfacePoint )
{
  auto component = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 0.5, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 1));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 0));

  ConfigurationSpaceFormulationMaxwell1D2V formulation(1.0, 1.0);
  formulation.registerTensorH1PointTraceDescriptors(tensorMesh);

  int varID = 100;
  for (const auto &descriptor : ConfigurationSpaceFormulationMaxwell1D2V::tensorH1PointTraceDescriptors1D())
  {
    VarPtr traceComponent = Teuchos::rcp(new Var(varID++, 0, descriptor.traceName() + "_component", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
    auto traceVar = Teuchos::rcp(new TensorVar(descriptor.traceName(), TENSOR_TRACE, std::vector<VarPtr>{traceComponent}, 0));
    tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(traceVar, Teuchos::null, std::vector<int>{0})));
  }

  auto dofLayout = tensorMesh->createDofLayout();
  const int sharedInterfacePointOrdinal = 1;
  std::vector<double> traceValues(component->vertexCount(), 0.0);
  traceValues[sharedInterfacePointOrdinal] = 7.25;

  for (const auto &descriptor : ConfigurationSpaceFormulationMaxwell1D2V::tensorH1PointTraceDescriptors1D())
  {
    auto layout = formulation.h1PointTraceLayout1D(dofLayout, descriptor.fieldName());
    TEST_ASSERT(layout.adjacentCellEndpointsShareDof(sharedInterfacePointOrdinal));

    const auto endpointDofs = layout.adjacentCellEndpointDofOrdinals(sharedInterfacePointOrdinal);
    TEST_EQUALITY(endpointDofs.first,  layout.endpointValueDofOrdinal(0, sharedInterfacePointOrdinal, TENSOR_TRACE_INTERFACE_UPPER));
    TEST_EQUALITY(endpointDofs.second, layout.endpointValueDofOrdinal(sharedInterfacePointOrdinal, 2, TENSOR_TRACE_INTERFACE_LOWER));
    TEST_EQUALITY(endpointDofs.first, sharedInterfacePointOrdinal);
    TEST_EQUALITY(endpointDofs.second, sharedInterfacePointOrdinal);
    TEST_FLOATING_EQUALITY(traceValues[endpointDofs.first], traceValues[endpointDofs.second], 1e-14);
  }
}

TEUCHOS_UNIT_TEST( TensorTraceLayout, DofLayoutOwnsRegisteredH1PointTraceLayoutFromTensorMesh )
{
  auto component = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 0.5, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 1));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 0));

  VarPtr traceComponent = Teuchos::rcp(new Var(0, 0, "E_x_hat_component", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  auto traceVar = Teuchos::rcp(new TensorVar("E_x_hat", TENSOR_TRACE, std::vector<VarPtr>{traceComponent}, 0));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(traceVar, Teuchos::null, std::vector<int>{0})));
  tensorMesh->registerTraceDescriptor(TensorTraceDescriptor::h1PointValueTrace1D("E_x", "E_x_hat", 0));

  auto dofLayout = tensorMesh->createDofLayout();
  auto layout = dofLayout->h1PointTraceLayout1D("E_x_hat", TENSOR_TRIAL_SPACE);
  TEST_EQUALITY(layout.descriptor().fieldName(), "E_x");
  TEST_EQUALITY(layout.descriptor().traceName(), "E_x_hat");
  TEST_EQUALITY(layout.interfacePointCount(), component->vertexCount());
  TEST_EQUALITY(layout.valueDofOrdinal(0), 0);
  TEST_EQUALITY(layout.valueDofOrdinal(static_cast<int>(component->vertexCount()) - 1), static_cast<int>(component->vertexCount()) - 1);

  TEST_THROW(dofLayout->h1PointTraceLayout1D("missing_trace", TENSOR_TRIAL_SPACE), std::invalid_argument);
}
}
