//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  AdditiveSchwarzTests
//  Camellia
//
//  Created by Nate Roberts on August 2, 2016.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Ifpack_Amesos.h"

#include "AdditiveSchwarz.h"
#include "Mesh.h"
#include "MeshFactory.h"
#include "MPIWrapper.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace
{
  Teuchos::RCP<AdditiveSchwarz<Ifpack_Amesos>> getSchwarzOp(SolutionPtr &soln, MeshTopologyPtr meshTopo, int overlapLevel, bool conformingTraces)
  {
    int spaceDim = meshTopo->getDimension();
    PoissonFormulation form(spaceDim, conformingTraces);
    int delta_k = 1; // we don't care how well we do in the Gram inversion..
    Teuchos_CommPtr Comm = MPIWrapper::TeuchosCommWorld();
    int H1Order = 1;
    MeshPtr mesh = MeshFactory::minRuleMesh(meshTopo, form.bf(), H1Order, delta_k, Comm);
    
    BCPtr bc = BC::bc();
    RHSPtr rhs = RHS::rhs();
    
    soln = Solution::solution(form.bf(), mesh, bc, rhs, form.bf()->graphNorm());
    
    soln->initializeStiffnessAndLoad();
    soln->populateStiffnessAndLoad();
    
    auto stiffness = soln->getStiffnessMatrix();
    bool useHierarchicalNeighbors = false;
    int dimensionForSchwarzNeighbors = spaceDim - 1; // always face neighbors, for us
    
    auto schwarzOp = Teuchos::rcp( new AdditiveSchwarz<Ifpack_Amesos>(stiffness.get(), overlapLevel, mesh, mesh,
                                                                      useHierarchicalNeighbors, dimensionForSchwarzNeighbors));
    return schwarzOp;
  }
  
  void testIncidenceCounting(vector<int> meshWidths, bool conformingTraces, int overlapLevel, int expectedIncidenceCount,
                             Teuchos::FancyOStream &out, bool &success)
  {
    int spaceDim = meshWidths.size();
    vector<double> dimensions(spaceDim,1);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, meshWidths);
    
    SolutionPtr soln; // keep reference so that stiffness matrix doesn't get deleted
    Teuchos::RCP<AdditiveSchwarz<Ifpack_Amesos>> schwarzOp = getSchwarzOp(soln, meshTopo, overlapLevel, conformingTraces);
    
    schwarzOp->Compute();
    
//    cout << "Max incidence count: " << schwarzOp->MaxGlobalIncidenceCount() << endl;
//    cout << "Max neighbor count:  " << schwarzOp->MaxGlobalOverlapSideNeighborCount() << endl;
    
    TEST_EQUALITY(schwarzOp->MaxGlobalIncidenceCount(), expectedIncidenceCount);
  }
  
  void testOverlapNeighborCounting(vector<int> meshWidths, bool conformingTraces, int overlapLevel, int expectedNeighborCount,
                                   Teuchos::FancyOStream &out, bool &success)
  {
    int spaceDim = meshWidths.size();
    vector<double> dimensions(spaceDim,1);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, meshWidths);
    
    PoissonFormulation form(spaceDim, conformingTraces);
    int delta_k = 1; // we don't care how well we do in the Gram inversion..
    Teuchos_CommPtr Comm = MPIWrapper::TeuchosCommWorld();
    int H1Order = 1;
    MeshPtr mesh = MeshFactory::minRuleMesh(meshTopo, form.bf(), H1Order, delta_k, Comm);
    bool useHierarchicalNeighbors = false;
    int dimensionForSchwarzNeighbors = spaceDim - 1; // always face neighbors, for us
    
    int count = AdditiveSchwarz<Ifpack_Amesos>::MaxGlobalOverlapSideNeighborCount(mesh, overlapLevel, useHierarchicalNeighbors,
                                                                                  dimensionForSchwarzNeighbors);
    
    TEST_EQUALITY(count, expectedNeighborCount);
  }
  
  TEUCHOS_UNIT_TEST( AdditiveSchwarz, IncidenceCountingZeroOverlap )
  {
    vector<int> meshWidths = {2,2};
    bool conformingTraces = true;
    int overlapLevel = 0;
    int expectedIncidenceCount = 4; // center vertex sees 4 elements
    testIncidenceCounting(meshWidths, conformingTraces, overlapLevel, expectedIncidenceCount, out, success);
    // same, but with conforming false --> edges, most they see is 2 elements
    conformingTraces = false;
    expectedIncidenceCount = 2;
    testIncidenceCounting(meshWidths, conformingTraces, overlapLevel, expectedIncidenceCount, out, success);
    meshWidths = {2,2,2};
    conformingTraces = true;
    expectedIncidenceCount = 8; // center vertex, again
    testIncidenceCounting(meshWidths, conformingTraces, overlapLevel, expectedIncidenceCount, out, success);
    conformingTraces = false;
    expectedIncidenceCount = 2;
    testIncidenceCounting(meshWidths, conformingTraces, overlapLevel, expectedIncidenceCount, out, success);
  }
  
  TEUCHOS_UNIT_TEST( AdditiveSchwarz, IncidenceCountingOneOverlap )
  {
    vector<int> meshWidths = {4,4};
    bool conformingTraces = true;
    int overlapLevel = 1;
    int expectedIncidenceCount = 12; // center vertex seen by 12 Schwarz domains
    testIncidenceCounting(meshWidths, conformingTraces, overlapLevel, expectedIncidenceCount, out, success);
    // same, but with conforming false --> edges, most they see is 2 elements
    conformingTraces = false;
    expectedIncidenceCount = 8; // center edges seen by 8 Schwarz domains
    testIncidenceCounting(meshWidths, conformingTraces, overlapLevel, expectedIncidenceCount, out, success);
  }

  TEUCHOS_UNIT_TEST( AdditiveSchwarz, NeighborCountingZeroOverlap )
  {
    vector<int> meshWidths = {2,2};
    bool conformingTraces = true;
    int overlapLevel = 0;
    int expectedNeighborCount = 3; // each element sees 2 face neighbors, plus itself
    testOverlapNeighborCounting(meshWidths, conformingTraces, overlapLevel, expectedNeighborCount, out, success);
    // same, but with conforming false --> edges, most they see is 2 elements
    conformingTraces = false;
    expectedNeighborCount = 3; // conformity doesn't enter into it (notably)
    testOverlapNeighborCounting(meshWidths, conformingTraces, overlapLevel, expectedNeighborCount, out, success);
    meshWidths = {2,2,2};
    conformingTraces = true;
    expectedNeighborCount = 4; // 3 face neighbors, plus self
    testOverlapNeighborCounting(meshWidths, conformingTraces, overlapLevel, expectedNeighborCount, out, success);
    conformingTraces = false;
    expectedNeighborCount = 4; // conformity doesn't enter into it
    testOverlapNeighborCounting(meshWidths, conformingTraces, overlapLevel, expectedNeighborCount, out, success);
  }
  
  TEUCHOS_UNIT_TEST( AdditiveSchwarz, NeighborCountingOneOverlap )
  {
    vector<int> meshWidths = {5,5};
    bool conformingTraces = true;
    int overlapLevel = 1;
    int expectedNeighborCount = 13; // center element has five elements in its overlap domain, and this has another 8 neighbors
    testOverlapNeighborCounting(meshWidths, conformingTraces, overlapLevel, expectedNeighborCount, out, success);
    conformingTraces = false;
    expectedNeighborCount = 13; // conformity doesn't enter into it
    testOverlapNeighborCounting(meshWidths, conformingTraces, overlapLevel, expectedNeighborCount, out, success);
  }

  
} // namespace
