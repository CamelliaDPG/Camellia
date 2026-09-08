//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ConfigurationSpaceFormulationPoissonTests.cpp
//  runTests
//
//  Created by Roberts, Nathan V on 3/20/26.
//
//


#include "Teuchos_UnitTestHarness.hpp"
#include "ConfigurationSpaceFormulationPoisson.h"
#include "MeshFactory.h"
#include "Function.h"
#include "Solution.h"
#include "RHS.h"
#include "BC.h"
#include "SpatialFilter.h"
#include "HDF5Exporter.h"

using namespace Camellia;

namespace {
  struct ManufacturedPoisson1D
  {
    FunctionPtr E_exact;   // exact E(x)
    FunctionPtr VE_exact;  // exact V_E(x)
    int p; // polynomial order of VE (H^1 order should be 1 higher than this)
  };

  ManufacturedPoisson1D manufacturedSolution_ZeroE()
  {
    FunctionPtr x = Function::xn(1);

    FunctionPtr VE_exact = Function::constant(-1.0);
    FunctionPtr E_exact  = - VE_exact->dx();

    ManufacturedPoisson1D ms;
    ms.E_exact = E_exact;
    ms.VE_exact = VE_exact;
    ms.p = 0;
    return ms;
  }

  ManufacturedPoisson1D manufacturedSolution_ConstantE()
  {
    FunctionPtr x = Function::xn(1);

    FunctionPtr VE_exact = - x;
    FunctionPtr E_exact  = - VE_exact->dx();

    ManufacturedPoisson1D ms;
    ms.E_exact = E_exact;
    ms.VE_exact = VE_exact;
    ms.p = 1;
    return ms;
  }

  ManufacturedPoisson1D manufacturedSolution_LinearE()
  {
    FunctionPtr x = Function::xn(1);

    FunctionPtr VE_exact = - x * x;
    FunctionPtr E_exact  = - VE_exact->dx();

    ManufacturedPoisson1D ms;
    ms.E_exact = E_exact;
    ms.VE_exact = VE_exact;
    ms.p = 2;
    return ms;
  }

  ManufacturedPoisson1D manufacturedSolution_QuadraticE()
  {
    FunctionPtr x = Function::xn(1);

    FunctionPtr VE_exact = - x * x * x + x * x;
    FunctionPtr E_exact  = - VE_exact->dx();

    ManufacturedPoisson1D ms;
    ms.E_exact = E_exact;
    ms.VE_exact = VE_exact;
    ms.p = 3;
    return ms;
  }

  // adapted from VlasovSpaceTimeFormulation
  void addConfigurationSpaceBC(const int tag, SolutionPtr configSoln,
                               VarPtr configVariable, FunctionPtr dirichletValue)
  {
    if (configSoln->bc() == Teuchos::null)
    {
      // create new BC object
      configSoln->setBC(BC::bc());
    }
    configSoln->bc()->addDirichlet(configVariable, tag, dirichletValue);
  }

  // adapted from VlasovSpaceTimeFormulation
  int createConfigurationMeshSideSetTag(int &nextConfigTag, MeshTopologyPtr configurationSpaceMeshTopology,
                                        SpatialFilterPtr regionToMatch, const bool restrictToBoundary)
  {
    const int thisTag = nextConfigTag++;
    
    const int sideDim = configurationSpaceMeshTopology->getDimension() - 1;
    auto matchingSideSet = configurationSpaceMeshTopology->createEntitySet();
    std::vector<IndexType> matchingSideIndices = restrictToBoundary ? configurationSpaceMeshTopology->getBoundarySidesThatMatch(regionToMatch)
                                                                    : configurationSpaceMeshTopology->getSidesThatMatch        (regionToMatch);
    for (const auto & matchingSideIndex : matchingSideIndices)
    {
      matchingSideSet->addEntity(sideDim, matchingSideIndex);
    }
    configurationSpaceMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, thisTag, matchingSideSet);
     
//    std::cout << matchingSideIndices.size() << " sides matched the specified region";
    
    return thisTag;
  }

  void testPoissonSolve(ManufacturedPoisson1D manufacturedSoln, const bool useNonUniformSpatialMesh, const bool spaceTime,
                        Teuchos::FancyOStream &out, bool &success)
  {
    const int spaceDim = 1;
    const int H1Order = manufacturedSoln.p + 1;
    const int delta_k = 2;
    const int xElements = 3;
    const int tElements = 1;
    const double x0 = 0.0;
    const double xWidth = 1.0;
    const double q = 1.0;
    const double eps0 = 0.5;
    
    const double t0     = 0.0;
    const double tWidth = 1.0;

    // Create the spatial mesh topology
    std::vector<double> points;
    MeshTopologyPtr spatialMeshTopo;
    if (useNonUniformSpatialMesh) {
      spatialMeshTopo = MeshFactory::geometricGradedIntervalMeshTopology(x0, x0 + xWidth, xElements);
    } else {
      spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements);
    }
    
    std::vector<double> temporalVertices(tElements + 1);
    for (int i=0; i<=tElements; i++)
    {
      const double t = double(i) / double(tElements) * tWidth + t0;
      temporalVertices[i] = t;
    }

    TimeDiscretizationType temporalType = spaceTime ? SPACE_TIME : TIME_MARCHING;

    
    auto configurationSpaceMeshTopology = spaceTime ? MeshFactory::extrudedMeshTopology(spatialMeshTopo, {temporalVertices})
                                                    : spatialMeshTopo;
  
//    configurationSpaceMeshTopology->printAllEntities();
    
    // Create the configuration space formulation for Poisson
    Teuchos::RCP<ConfigurationSpaceFormulationPoisson> configForm = Teuchos::rcp(new ConfigurationSpaceFormulationPoisson(q, eps0, temporalType));
    configForm->initializeMesh(configurationSpaceMeshTopology, H1Order, delta_k);

    // Define the exact solution and RHS
    FunctionPtr E_exact = manufacturedSoln.E_exact;
    FunctionPtr VE_exact = manufacturedSoln.VE_exact;
    FunctionPtr ddx_E = E_exact->dx();

    // Create the RHS
    RHSPtr rhs = RHS::rhs();
    rhs->addTerm(ddx_E * configForm->v());
    
    // Create the solution
    SolutionPtr soln = configForm->solution();

    // Set the RHS and boundary conditions
    soln->setRHS(rhs);
    
    SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(x0);
    SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(x0 + xWidth);
    
    const bool usePeriodicBCs = false;
    const bool restrictSideSetsToBoundary = !usePeriodicBCs; // for periodic BCs, the config mesh region will *not* be on a mesh boundary
    int nextTag = 1;
    auto xLeftRightTag = createConfigurationMeshSideSetTag(nextTag, configurationSpaceMeshTopology,
                                                           xLeftFilter | xRightFilter,
                                                           restrictSideSetsToBoundary);
    auto VE_traceVar = configForm->VE_hat();
    addConfigurationSpaceBC(xLeftRightTag, soln, VE_traceVar, VE_exact);
    
    // Solve the system
    soln->solve();
    
    auto mesh = soln->mesh();

    // Extract the solution
    FunctionPtr E_soln = Function::solution(configForm->E_fieldVar(0), soln);

    // Compute the error
    double E_soln_magnitude = E_soln->l2norm(mesh);
    double E_err_l2 = (E_soln - E_exact)->l2norm(mesh);
    out << "Poisson E magnitude: " << E_soln_magnitude << "; E error: " << E_err_l2 << std::endl;

    // Check the error
    const double tol = 1e-12;
    if (E_soln_magnitude > tol)
    {
      TEST_COMPARE(E_err_l2, <=, tol * E_soln_magnitude);
    }
    else
    {
      TEST_COMPARE(E_err_l2, <=, tol);
    }

    // Optionally, export the solution for visualization
    const bool outputViz = true;
    if (outputViz) {
      std::string vizPrefix = useNonUniformSpatialMesh ? "PoissonSolveNonUniformMesh" :  "PoissonSolve";
      auto exporter = Teuchos::rcp(new HDF5Exporter(mesh, vizPrefix, "/tmp"));
      exporter->exportSolution(soln, std::vector<FunctionPtr>{E_soln - E_exact, E_exact}, std::vector<std::string>{"E err", "E exact"});
    }
  }

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, ManufacturedFunctionDerivatives) {
  const double tol = 1e-14;

  auto linearE = manufacturedSolution_LinearE();
  TEST_FLOATING_EQUALITY(linearE.E_exact->evaluate(0.25), 0.5, tol);
  TEST_FLOATING_EQUALITY(linearE.E_exact->evaluate(0.75), 1.5, tol);
  TEST_FLOATING_EQUALITY(linearE.E_exact->dx()->evaluate(0.25), 2.0, tol);
  TEST_FLOATING_EQUALITY(linearE.E_exact->dx()->evaluate(0.75), 2.0, tol);

  auto quadraticE = manufacturedSolution_QuadraticE();
  TEST_FLOATING_EQUALITY(quadraticE.E_exact->evaluate(0.25), -0.3125, tol);
  TEST_FLOATING_EQUALITY(quadraticE.E_exact->evaluate(0.75), 0.1875, tol);
  TEST_FLOATING_EQUALITY(quadraticE.E_exact->dx()->evaluate(0.25), -0.5, tol);
  TEST_FLOATING_EQUALITY(quadraticE.E_exact->dx()->evaluate(0.75), 2.5, tol);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveZeroE) {
  const bool useNonUniformSpatialMesh = false;
  const bool spaceTime = true;
  auto manufacturedSoln = manufacturedSolution_ZeroE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveZeroESpaceOnly) {
  const bool useNonUniformSpatialMesh = false;
  const bool spaceTime = false;
  auto manufacturedSoln = manufacturedSolution_ZeroE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveZeroENonUniformMesh) {
  const bool useNonUniformSpatialMesh = true;
  const bool spaceTime = true;
  auto manufacturedSoln = manufacturedSolution_ZeroE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveZeroENonUniformMeshSpaceOnly) {
  const bool useNonUniformSpatialMesh = true;
  const bool spaceTime = false;
  auto manufacturedSoln = manufacturedSolution_ZeroE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveConstantE) {
  const bool useNonUniformSpatialMesh = false;
  const bool spaceTime = true;
  auto manufacturedSoln = manufacturedSolution_ConstantE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveConstantENonUniformMesh) {
  const bool useNonUniformSpatialMesh = true;
  const bool spaceTime = true;
  auto manufacturedSoln = manufacturedSolution_ConstantE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveConstantESpaceOnly) {
  const bool useNonUniformSpatialMesh = false;
  const bool spaceTime = false;
  auto manufacturedSoln = manufacturedSolution_ConstantE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveConstantENonUniformMeshSpaceOnly) {
  const bool useNonUniformSpatialMesh = true;
  const bool spaceTime = false;
  auto manufacturedSoln = manufacturedSolution_ConstantE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveLinearE) {
  const bool useNonUniformSpatialMesh = false;
  const bool spaceTime = true;
  auto manufacturedSoln = manufacturedSolution_LinearE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveLinearENonUniformMesh) {
  const bool useNonUniformSpatialMesh = true;
  const bool spaceTime = true;
  auto manufacturedSoln = manufacturedSolution_LinearE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveLinearESpaceOnly) {
  const bool useNonUniformSpatialMesh = false;
  const bool spaceTime = false;
  auto manufacturedSoln = manufacturedSolution_LinearE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveLinearENonUniformMeshSpaceOnly) {
  const bool useNonUniformSpatialMesh = true;
  const bool spaceTime = false;
  auto manufacturedSoln = manufacturedSolution_LinearE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveQuadraticE) {
  const bool useNonUniformSpatialMesh = false;
  const bool spaceTime = true;
  auto manufacturedSoln = manufacturedSolution_QuadraticE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveQuadraticENonUniformMesh) {
  const bool useNonUniformSpatialMesh = true;
  const bool spaceTime = true;
  auto manufacturedSoln = manufacturedSolution_QuadraticE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveQuadraticESpaceOnly) {
  const bool useNonUniformSpatialMesh = false;
  const bool spaceTime = false;
  auto manufacturedSoln = manufacturedSolution_QuadraticE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

TEUCHOS_UNIT_TEST(ConfigurationSpaceFormulationPoisson, SolveQuadraticENonUniformMeshSpaceOnly) {
  const bool useNonUniformSpatialMesh = true;
  const bool spaceTime = false;
  auto manufacturedSoln = manufacturedSolution_QuadraticE();
  testPoissonSolve(manufacturedSoln, useNonUniformSpatialMesh, spaceTime, out, success);
}

} // end anonymous namespace
