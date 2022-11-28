//
// © 2016-2022 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  SpatialIntegralOfSpaceTimeSolutionTests
//  Camellia
//
//  Created by Nate Roberts on 11/28/22.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "SpatialIntegralOfSpaceTimeSolution.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( SpatialIntegralOfSpaceTimeSolution, PatchTest )
  {
    FunctionPtr x       = Function::xn(1);
    FunctionPtr t       = Function::yn(1);
    FunctionPtr f_exact = x * t;
    
    // create a sample space-time Mesh, with just a single trial field variable, f.
    VarFactoryPtr vf = VarFactory::varFactory();
    auto f = vf->fieldVarSpaceTime("f");
    auto v = vf->testVarSpaceTime("v", L2);
    
    auto bf = BF::bf(vf);
    
    double xLeft  = 0;
    double xRight = 1;
    double xElements = 5;
    MeshTopologyPtr spatialMeshTopo = MeshFactory::intervalMeshTopology(xLeft, xRight, xElements);
    
    double t0 = 0;
    double t1 = 1;
    int numTemporalDivisions = 5;
    
    int spatialH1Order  = 2; // want L^2 variables to be at least linear in space
    int temporalH1Order = 2; // want L^2 variables to be at least linear in time
    int pToAdd = 0; // Bubnov-Galerkin
    auto spaceTimeMeshTopo = MeshFactory::extrudedMeshTopology(spatialMeshTopo, {t0}, {t1-t0}, {numTemporalDivisions});
    bool allowPruning      = false;
    auto spaceTimeMesh     = MeshFactory::minRuleMesh(spaceTimeMeshTopo, bf, {spatialH1Order,temporalH1Order}, pToAdd, Teuchos::null, allowPruning);
    
    auto spaceTimeSoln = Solution::solution(bf, spaceTimeMesh);
    
    // project f_exact onto space-time solution:
    std::map<int,FunctionPtr> f_map;
    f_map[f->ID()] = f_exact;
    spaceTimeSoln->projectOntoMesh(f_map);
    
    FunctionPtr integrationWeight = x;
    int quadratureEnrichment = 1; // matches the order of the integration weight
    SpatialIntegralOfSpaceTimeSolution spatialIntegral(spaceTimeSoln, f, integrationWeight, quadratureEnrichment);
    
    auto actualIntegral = spatialIntegral.temporalFunction();
    
    // integral of f_exact * integrationWeight:
    // x * t * x = x^2 * t, integrated in x --> x^3 * t / 3, from xLeft to xRight
    // note that the temporal mesh is 1D, so the right way to express the exact function is in x:
    auto exactIntegral = x * (xRight * xRight * xRight - xLeft * xLeft * xLeft) / 3.; // t * (xRight^3 - xLeft^3) / 3.
    
    double err = (actualIntegral - exactIntegral)->l2norm(spatialIntegral.temporalMesh());
    
    double tol = 1e-13;
    TEST_COMPARE(err, <, tol);
  }

  TEUCHOS_UNIT_TEST( SpatialIntegralOfSpaceTimeSolution, PatchTestRefinedMesh )
  {
    FunctionPtr x       = Function::xn(1);
    FunctionPtr t       = Function::yn(1);
    FunctionPtr f_exact = x * t;
    
    // create a sample space-time Mesh, with just a single trial field variable, f.
    VarFactoryPtr vf = VarFactory::varFactory();
    auto f = vf->fieldVarSpaceTime("f");
    auto v = vf->testVarSpaceTime("v", L2);
    
    auto bf = BF::bf(vf);
    
    double xLeft  = 0;
    double xRight = 1;
    double xElements = 5;
    MeshTopologyPtr spatialMeshTopo = MeshFactory::intervalMeshTopology(xLeft, xRight, xElements);
    
    double t0 = 0;
    double t1 = 1;
    int numTemporalDivisions = 5;
    
    int spatialH1Order  = 2; // want L^2 variables to be at least linear in space
    int temporalH1Order = 2; // want L^2 variables to be at least linear in time
    int pToAdd = 0; // Bubnov-Galerkin
    auto spaceTimeMeshTopo = MeshFactory::extrudedMeshTopology(spatialMeshTopo, {t0}, {t1-t0}, {numTemporalDivisions});
    bool allowPruning      = false;
    auto spaceTimeMesh     = MeshFactory::minRuleMesh(spaceTimeMeshTopo, bf, {spatialH1Order,temporalH1Order}, pToAdd, Teuchos::null, allowPruning);
    
    // pick some arbitrary cell to refine
    GlobalIndexType cellIDToRefine = xElements * numTemporalDivisions / 2;
    std::vector<GlobalIndexType> cellsToRefine {cellIDToRefine};
    spaceTimeMesh->hRefine(cellsToRefine);
    
    auto spaceTimeSoln = Solution::solution(bf, spaceTimeMesh);
    
    // project f_exact onto space-time solution:
    std::map<int,FunctionPtr> f_map;
    f_map[f->ID()] = f_exact;
    spaceTimeSoln->projectOntoMesh(f_map);
    
    FunctionPtr integrationWeight = x;
    int quadratureEnrichment = 1; // matches the order of the integration weight
    SpatialIntegralOfSpaceTimeSolution spatialIntegral(spaceTimeSoln, f, integrationWeight, quadratureEnrichment);
    
    auto actualIntegral = spatialIntegral.temporalFunction();
    
    // integral of f_exact * integrationWeight:
    // x * t * x = x^2 * t, integrated in x --> x^3 * t / 3, from xLeft to xRight
    // note that the temporal mesh is 1D, so the right way to express the exact function is in x:
    auto exactIntegral = x * (xRight * xRight * xRight - xLeft * xLeft * xLeft) / 3.; // t * (xRight^3 - xLeft^3) / 3.
    
    double err = (actualIntegral - exactIntegral)->l2norm(spatialIntegral.temporalMesh());
    
    double tol = 1e-13;
    TEST_COMPARE(err, <, tol);
  }
} // namespace
