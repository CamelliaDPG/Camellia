//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  LinearTermSumFactorizedTests
//  Camellia
//
//  Sum-factorized LinearTerm integration (Intrepid2::PAMatrix) against quadrature, for the element stiffness and graph-norm
//  Gram matrices of a relativistic 1D1T2V Vlasov space-time formulation: 4D extruded hypercube cells with coefficients
//  that vary in phase space (v = u / gamma) and in configuration space (the fields E_x, B_y).
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include "CamelliaTestingHelpers.h"
#include "ConfigurationSpaceFormulationMaxwell1D2V.h"
#include "IP.h"
#include "LinearTermSumFactorized.h"
#include "MeshFactory.h"
#include "MeshTools.h"
#include "RelativisticVlasovSpaceTimeFormulation.h"

using namespace Camellia;

namespace
{
  const double TEST_Q = -1.0, TEST_M = 1.0, TEST_EPS0 = 1.0;

  Teuchos::RCP<RelativisticVlasovSpaceTimeFormulation> relativisticForm1D2V(const EBasisFamily basisFamily)
  {
    const double c = 2.0;
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(0.0, 0.7, 2, false);
    std::vector<double> temporalVertices = MeshTools::verticesFor1DIntervalMesh(0.0, 0.2, 1);
    std::vector<double> velocitySpaceVertices_x = MeshTools::verticesFor1DIntervalMesh(-1.5, 2.5, 2);
    std::vector<double> velocitySpaceVertices_y = MeshTools::verticesFor1DIntervalMesh(-0.5, 1.25, 1);
    std::vector<double> velocitySpaceVertices_z;
    auto configForm = Teuchos::rcp(new ConfigurationSpaceFormulationMaxwell1D2V(TEST_Q, TEST_EPS0));
    const int H1Order = 2, deltaP = 1, numUniformRefinements = 0;
    const double L2Weight = 0.3;
    auto form = Teuchos::rcp(new RelativisticVlasovSpaceTimeFormulation(c, TEST_Q, TEST_M, configForm, spatialMeshTopo, temporalVertices,
                                                                        velocitySpaceVertices_x, velocitySpaceVertices_y,
                                                                        velocitySpaceVertices_z, H1Order, deltaP, numUniformRefinements,
                                                                        RelativisticVlasovSpaceTimeFormulation::GraphTestSpaceInnerProduct,
                                                                        L2Weight, basisFamily));
    // nonconstant fields, so that the acceleration varies within each cell
    FunctionPtr x = Function::xn(1);
    FunctionPtr t = Function::yn(1);
    std::map<int,FunctionPtr> configProjection;
    configProjection[configForm->E_fieldVar(0)->ID()] = 1.0 + 0.5 * x * t + 0.25 * x * x;
    configProjection[configForm->B_fieldVar(1)->ID()] = 0.75 - 0.5 * x + t;
    form->configurationSpaceSolution()->projectOntoMesh(configProjection);
    form->configurationSpaceFormulation()->updateConfigurationSpaceFunctions();
    return form;
  }

#ifdef CAMELLIA_HAVE_PAMATRIX
  //! element stiffness (B) and graph-norm Gram (G) matrices, sum-factorized vs. quadrature, on every cell
  void testRelativisticVlasov1D2V(const EBasisFamily basisFamily, Teuchos::FancyOStream &out, bool &success)
  {
    auto form = relativisticForm1D2V(basisFamily);
    auto mesh = form->vlasovMesh();
    auto bf = form->vlasovBF();
    auto ip = form->constructVlasovInnerProduct(); // the graph norm, as a LinearTerm inner product
    const bool initiallyEnabled = sumFactorizedLinearTermIntegrationEnabled();
    for (auto cellID : mesh->cellIDsInPartition())
    {
      auto elemType = mesh->getElementType(cellID);
      const int numTrialDofs = elemType->trialOrderPtr->totalDofs();
      const int numTestDofs  = elemType->testOrderPtr->totalDofs();
      auto basisCache   = BasisCache::basisCacheForCell(mesh, cellID, false, 0);
      auto ipBasisCache = BasisCache::basisCacheForCell(mesh, cellID, true, 0);
      const auto &parities = basisCache->getCellSideParities();

      // the sum-factorized path must actually apply here (otherwise the comparison below would be vacuous): the graph
      // norm's transport term, with coefficients varying in phase space, is supported
      auto testVar = form->w(0);
      LinearTermPtr transport = testVar->dy() + form->v_i(0) * testVar->dx();
      FieldContainer<double> direct(1, numTestDofs, numTestDofs);
      setSumFactorizedLinearTermIntegrationEnabled(true);
      const bool applied = integrateLinearTermsSumFactorized(direct, transport, elemType->testOrderPtr, transport, elemType->testOrderPtr,
                                                             ipBasisCache);
      TEST_ASSERT(applied);

      FieldContainer<double> bStandard(1,numTrialDofs,numTestDofs), bSumFactorized(1,numTrialDofs,numTestDofs); // rowMajor: (C,FTrial,FTest)
      FieldContainer<double> gStandard(1,numTestDofs,numTestDofs),  gSumFactorized(1,numTestDofs,numTestDofs);
      setSumFactorizedLinearTermIntegrationEnabled(false);
      bf->stiffnessMatrix(bStandard, elemType, parities, basisCache, true, false);
      ip->computeInnerProductMatrix(gStandard, elemType->testOrderPtr, ipBasisCache);
      setSumFactorizedLinearTermIntegrationEnabled(true);
      bf->stiffnessMatrix(bSumFactorized, elemType, parities, basisCache, true, false);
      ip->computeInnerProductMatrix(gSumFactorized, elemType->testOrderPtr, ipBasisCache);

      const double relTol = 1e-11, absTol = 1e-11;
      bool bOK = compareFloatingArraysCamellia(bSumFactorized, "sum-factorized B", bStandard, "standard B", relTol, absTol, out);
      bool gOK = compareFloatingArraysCamellia(gSumFactorized, "sum-factorized G", gStandard, "standard G", relTol, absTol, out);
      if (!bOK || !gOK)
      {
        out << "comparison failed on cell " << cellID << " (B " << bOK << ", G " << gOK << ")\n";
        success = false;
      }
    }
    setSumFactorizedLinearTermIntegrationEnabled(initiallyEnabled);
  }

  TEUCHOS_UNIT_TEST( LinearTermSumFactorized, RelativisticVlasov1D2V_Hierarchical )
  {
    testRelativisticVlasov1D2V(BASIS_FAMILY_HIERARCHICAL, out, success);
  }

  TEUCHOS_UNIT_TEST( LinearTermSumFactorized, RelativisticVlasov1D2V_Nodal )
  {
    testRelativisticVlasov1D2V(BASIS_FAMILY_NODAL, out, success);
  }

  TEUCHOS_UNIT_TEST( LinearTermSumFactorized, RelativisticVlasov1D2V_Serendipity )
  {
    testRelativisticVlasov1D2V(BASIS_FAMILY_SERENDIPITY, out, success);
  }

  //! unsupported cells fall back: a triangle mesh is not a tensor-product cell
  TEUCHOS_UNIT_TEST( LinearTermSumFactorized, NonTensorCellFallsBack )
  {
    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr v = vf->testVar("v", HGRAD);
    VarPtr u = vf->fieldVar("u");
    BFPtr bf = BF::bf(vf);
    bf->addTerm(u, v);
    MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, 2, 1, 1.0, 1.0, 1, 1, true); // true: triangles
    const GlobalIndexType cellID = *mesh->cellIDsInPartition().begin();
    auto elemType = mesh->getElementType(cellID);
    auto basisCache = BasisCache::basisCacheForCell(mesh, cellID, true, 0);
    const int numTestDofs = elemType->testOrderPtr->totalDofs();
    FieldContainer<double> values(1, numTestDofs, numTestDofs);
    LinearTermPtr lt = 1.0 * v;
    const bool initiallyEnabled = sumFactorizedLinearTermIntegrationEnabled();
    setSumFactorizedLinearTermIntegrationEnabled(true);
    TEST_ASSERT(!integrateLinearTermsSumFactorized(values, lt, elemType->testOrderPtr, lt, elemType->testOrderPtr, basisCache));
    setSumFactorizedLinearTermIntegrationEnabled(initiallyEnabled);
    for (int i=0; i<values.size(); i++) TEST_EQUALITY(values[i], 0.0); // unmodified on fallback
  }
#else
  //! built without Intrepid2::PAMatrix: never applied, so all integration is by quadrature
  TEUCHOS_UNIT_TEST( LinearTermSumFactorized, DisabledWithoutPAMatrix )
  {
    TEST_ASSERT(!sumFactorizedLinearTermIntegrationEnabled());
    setSumFactorizedLinearTermIntegrationEnabled(true);
    TEST_ASSERT(!sumFactorizedLinearTermIntegrationEnabled());

    auto form = relativisticForm1D2V(BASIS_FAMILY_HIERARCHICAL);
    auto mesh = form->vlasovMesh();
    const GlobalIndexType cellID = *mesh->cellIDsInPartition().begin();
    auto elemType = mesh->getElementType(cellID);
    auto ipBasisCache = BasisCache::basisCacheForCell(mesh, cellID, true, 0);
    const int numTestDofs = elemType->testOrderPtr->totalDofs();
    FieldContainer<double> values(1, numTestDofs, numTestDofs);
    LinearTermPtr lt = 1.0 * form->w(0);
    TEST_ASSERT(!integrateLinearTermsSumFactorized(values, lt, elemType->testOrderPtr, lt, elemType->testOrderPtr, ipBasisCache));
  }
#endif
} // namespace
