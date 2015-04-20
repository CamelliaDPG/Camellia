//
//  SpaceTimeHeatFormulation
//  Camellia
//
//  Created by Nate Roberts on 11/25/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "SpaceTimeHeatFormulation.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace {
  void projectExactSolution(SpaceTimeHeatFormulation &form, SolutionPtr<double> heatSolution, FunctionPtr u) {
    double epsilon = form.epsilon();

    FunctionPtr sigma1, sigma2, sigma3;
    int spaceTimeDim = heatSolution->mesh()->getDimension();
    int spaceDim = spaceTimeDim - 1;

    sigma1 = epsilon * u->dx();
    if (spaceDim > 1) sigma2 = epsilon * u->dy();
    if (spaceDim > 2) sigma3 = epsilon * u->dz();

    LinearTermPtr sigma_n_lt = form.sigma_n_hat()->termTraced();
    LinearTermPtr u_lt = form.u_hat()->termTraced();

    map<int, FunctionPtr> exactMap;
    // fields:
    exactMap[form.u()->ID()] = u;
    exactMap[form.sigma(1)->ID()] = sigma1;
    if (spaceDim > 1) exactMap[form.sigma(2)->ID()] = sigma2;
    if (spaceDim > 2) exactMap[form.sigma(3)->ID()] = sigma3;

    // flux:
    // use the exact field variable solution together with the termTraced to determine the flux traced
    FunctionPtr sigma_n = sigma_n_lt->evaluate(exactMap);
    exactMap[form.sigma_n_hat()->ID()] = sigma_n;

    // traces:
    FunctionPtr u_hat = u_lt->evaluate(exactMap);
    exactMap[form.u_hat()->ID()] = u_hat;

    heatSolution->projectOntoMesh(exactMap);
  }

  void setupExactSolution(SpaceTimeHeatFormulation &form, FunctionPtr u,
                          MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k)
  {
    double epsilon = form.epsilon();

    FunctionPtr sigma1, sigma2, sigma3;
    int spaceTimeDim = meshTopo->getSpaceDim();
    int spaceDim = spaceTimeDim - 1;

    FunctionPtr forcingFunction = SpaceTimeHeatFormulation::forcingFunction(spaceDim, epsilon, u);

    form.initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction);
  }

  void testForcingFunctionForConstantU(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    // Forcing function should be zero for constant u
    FunctionPtr f_expected = Function::zero();

    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // one-element mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

    double epsilon = .1;
    int fieldPolyOrder = 1, delta_k = 1;

    FunctionPtr u = Function::constant(0.5);

    bool useConformingTraces = true;
    SpaceTimeHeatFormulation form(spaceDim, useConformingTraces, epsilon);

    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);

    MeshPtr mesh = form.solution()->mesh();

    FunctionPtr f_actual = form.forcingFunction(spaceDim, epsilon, u);

    double l2_diff = (f_expected-f_actual)->l2norm(mesh);
    TEST_COMPARE(l2_diff, <, 1e-14);
  }

  void testSpaceTimeHeatConsistencyConstantSolution(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // one-element mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

    double epsilon = .1;
    int fieldPolyOrder = 1, delta_k = 1;

    FunctionPtr u = Function::constant(0.5);

    bool useConformingTraces = true;
    SpaceTimeHeatFormulation form(spaceDim, useConformingTraces, epsilon);

    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);
    projectExactSolution(form, form.solution(), u);

    form.solution()->clearComputedResiduals();

    double energyError = form.solution()->energyErrorTotal();

    if (spaceDim != 3)
    {
      MeshPtr mesh = form.solution()->mesh();
      string outputDir = "/tmp";
      string solnName = (spaceDim == 1) ? "spaceTimeHeatConstantSolution_1D" : "spaceTimeHeatConstantSolution_2D";
      cout << "\nDebugging: Outputting spaceTimeHeatSolution to " << outputDir << "/" << solnName << endl;
      HDF5Exporter exporter(mesh, solnName, outputDir);
      exporter.exportSolution(form.solution());
    }

    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }

  void testSpaceTimeHeatConsistency(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
    vector<int> elementCounts(spaceDim,1);   // 1^d mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

    double epsilon = 0.1;
    int fieldPolyOrder = 2, delta_k = 1;

    FunctionPtr u;
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    FunctionPtr t = Function::tn(1);

    if (spaceDim == 1)
    {
      u = x * t;
    }
    else if (spaceDim == 2)
    {
      u = x * t + y;
    }
    else if (spaceDim == 3)
    {
      u = x * t + y - z;
    }

    bool useConformingTraces = true;
    SpaceTimeHeatFormulation form(spaceDim, useConformingTraces, epsilon);

    setupExactSolution(form, u, spaceTimeMeshTopo, fieldPolyOrder, delta_k);
    projectExactSolution(form, form.solution(), u);

    form.solution()->clearComputedResiduals();

    double energyError = form.solution()->energyErrorTotal();

    if (spaceDim != 3)
    {
      MeshPtr mesh = form.solution()->mesh();
      string outputDir = "/tmp";
      string solnName = (spaceDim == 1) ? "spaceTimeHeatSolution_1D" : "spaceTimeHeatSolution_2D";
      cout << "\nDebugging: Outputting spaceTimeHeatSolution to " << outputDir << "/" << solnName << endl;
      HDF5Exporter exporter(mesh, solnName, outputDir);
      exporter.exportSolution(form.solution());
    }

    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ConsistencyConstantSolution_1D )
  {
    // consistency test for space-time formulation with 1D space
    testSpaceTimeHeatConsistencyConstantSolution(1, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ConsistencyConstantSolution_2D )
  {
    // consistency test for space-time formulation with 2D space
    testSpaceTimeHeatConsistencyConstantSolution(2, out, success);
  }

//  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ConsistencyConstantSolution_3D )
//  {
//    // consistency test for space-time formulation with 3D space
//    testSpaceTimeHeatConsistencyConstantSolution(3, out, success);
//  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, Consistency_1D )
  {
    // consistency test for space-time formulation with 1D space
    testSpaceTimeHeatConsistency(1, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, Consistency_2D )
  {
    // consistency test for space-time formulation with 2D space
    testSpaceTimeHeatConsistency(2, out, success);
  }

//  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, Consistency_3D )
//  {
//    // consistency test for space-time formulation with 3D space
//    testSpaceTimeHeatConsistency(3, out, success);
//  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ForcingFunctionForConstantU_1D )
  {
    // constant u should imply forcing function is zero
    testForcingFunctionForConstantU(1, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ForcingFunctionForConstantU_2D )
  {
    // constant u should imply forcing function is zero
    testForcingFunctionForConstantU(2, out, success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeHeatFormulation, ForcingFunctionForConstantU_3D )
  {
    // constant u should imply forcing function is zero
    testForcingFunctionForConstantU(3, out, success);
  }
} // namespace
