//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  TestTemplate
//  Camellia
//
//  Created by Nate Roberts on 3/31/16.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "CellDataMigration.h"
#include "MeshFactory.h"
#include "MPIWrapper.h"
#include "PoissonFormulation.h"
#include "TypeDefs.h"

using namespace Camellia;
using namespace std;

namespace
{
  typedef pair<RefinementBranch,vector<GlobalIndexType>> LabeledRefinementBranch; // first cellID indicates the root cell ID; after that, each cellID indicates first child of each refinement
  typedef pair<LabeledRefinementBranch, vector<vector<double>> > RootedLabeledRefinementBranch; // second contains vertex coordinates for root cell

  bool verticesEquals(vector<vector<double>> &vertices1, vector<vector<double>> &vertices2)
  {
    double tol = 1e-15;
    if (vertices1.size() != vertices2.size()) return false;
    for (int i=0; i<vertices1.size(); i++)
    {
      if (vertices1[i].size() != vertices2[i].size()) return false;
      for (int j=0; j<vertices1[i].size(); j++)
      {
        double diff = abs(vertices1[i][j]-vertices2[i][j]);
        if (diff > tol) return false;
      }
    }
    return true;
  }
  
  bool labelsEquals(vector<GlobalIndexType> &labels1, vector<GlobalIndexType> &labels2)
  {
    if (labels1.size() != labels2.size()) return false;
    for (int i=0; i<labels1.size(); i++)
    {
      if (labels1[i] != labels2[i]) return false;
    }
    return true;
  }
  
  bool refBranchEquals(RefinementBranch &refBranch1, RefinementBranch &refBranch2)
  {
    if (refBranch1.size() != refBranch2.size()) return false;
    for (int i=0; i<refBranch1.size(); i++)
    {
      RefinementPattern* refPattern1 = refBranch1[i].first;
      RefinementPattern* refPattern2 = refBranch2[i].first;
      if (refPattern1->getKey() != refPattern2->getKey()) return false;
      unsigned childOrdinal1 = refBranch1[i].second;
      unsigned childOrdinal2 = refBranch2[i].second;
      if (childOrdinal1 != childOrdinal2) return false;
    }
    return true;
  }
  
  bool labeledRefBranchEquals(LabeledRefinementBranch &refBranch1, LabeledRefinementBranch &refBranch2)
  {
    bool refBranchesMatch = refBranchEquals(refBranch1.first, refBranch2.first);
    {
      // DEBUGGING: repeat the call
      if (!refBranchesMatch)
      {
        refBranchEquals(refBranch1.first, refBranch2.first);
      }
    }
    bool labelsMatch = labelsEquals(refBranch1.second, refBranch2.second);
    return refBranchesMatch && labelsMatch;
  }
  
  bool rootedRefBranchEquals(RootedLabeledRefinementBranch &refBranch1, RootedLabeledRefinementBranch &refBranch2)
  {
    bool labeledRefBranchesMatch = labeledRefBranchEquals(refBranch1.first, refBranch2.first);
    {
      // DEBUGGING: repeat the call if it's false so we can see why it's false
      if (!labeledRefBranchesMatch)
      {
        labeledRefBranchEquals(refBranch1.first, refBranch2.first);
      }
    }
    bool verticesMatch = verticesEquals(refBranch1.second, refBranch2.second);
    return verticesMatch && labeledRefBranchesMatch;
  }
  
  bool rootedRefBranchVectorEquals(vector<RootedLabeledRefinementBranch> &refBranches1, vector<RootedLabeledRefinementBranch> &refBranches2)
  {
    if (refBranches1.size() != refBranches2.size()) return false;
    
    for (int i=0; i<refBranches1.size(); i++)
    {
      if ( !rootedRefBranchEquals(refBranches1[i], refBranches2[i]) )
      {
        {
          // DEBUGGING: repeat the call so we can see why it's false
          rootedRefBranchEquals(refBranches1[i], refBranches2[i]);
        }
        return false;
      }
    }
    
    return true;
  }
  
  void testPackAndUnpackGeometry(MeshPtr mesh, Teuchos::FancyOStream &out, bool &success)
  {
    /* For every active cell in mesh, do the following:
     1. Get the data size.
     2. Allocate a buffer that is 3x the data size, initialized with -1 values.
     3. Pack the buffer.
     4. Check that the written size is exactly the advertised size.
     5. Unpack from the buffer (returns a vector<RootedLabeledRefinementBranch>).
     6. Compare the return value with CellDataMigration::getCellHaloGeometry().
     */
    
    MPIWrapper::CommWorld()->Barrier();
    
    const set<GlobalIndexType>* myCellIndices = &mesh->cellIDsInPartition();
    
    for (GlobalIndexType cellID : *myCellIndices)
    {
      int size = CellDataMigration::geometryDataSize(mesh.get(), cellID);
      vector<char> dataBuffer(size*3,static_cast<char>(-1));
      char* dataLocation = &dataBuffer[0];
      CellDataMigration::packGeometryData(mesh.get(), cellID, dataLocation, size);
      int sizeWritten = dataLocation - &dataBuffer[0];
      TEST_EQUALITY(sizeWritten, size);
      const char *constDataLocation = &dataBuffer[0];
      vector<RootedLabeledRefinementBranch> unpackedBranch;
      CellDataMigration::unpackGeometryData(mesh.get(), cellID, constDataLocation, size, unpackedBranch);
      int sizeRead = constDataLocation - &dataBuffer[0];
      TEST_EQUALITY(sizeRead, size);
      vector<RootedLabeledRefinementBranch> expectedBranch;
      
      // new 6-6-16: we don't expect any geometry to be included for non-distributed MeshTopology:
      bool isDistributed = mesh->getTopology()->isDistributed();
      
      if (isDistributed) CellDataMigration::getCellHaloGeometry(mesh.get(), cellID, expectedBranch);
      
//      {
//        // DEBUGGING
//        if (!rootedRefBranchVectorEquals(unpackedBranch, expectedBranch))
//        {
//          // repeat the call so we can see why it's false...
//          rootedRefBranchVectorEquals(unpackedBranch, expectedBranch);
//        }
//      }
      
      TEST_ASSERT(rootedRefBranchVectorEquals(unpackedBranch, expectedBranch));
    }
  }

  // test commented out because it seems to me it should be revised to make it more thorough.
//  TEUCHOS_UNIT_TEST( CellDataMigration, CellGeometryAvoidsRedundancy_1D )
//  {
//    // a simple test to see that we don't have redundant refinement branches for
//    // a particular cell and its ancestors.  (This is not a very thorough test yet.)
//    MPIWrapper::CommWorld()->Barrier();
//    int spaceDim = 1;
//    bool useConformingTraces = true;
//    int H1Order = 2;
//    int meshWidth = 2;
//    PoissonFormulation form(spaceDim,useConformingTraces,PoissonFormulation::ULTRAWEAK);
//    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0}, {meshWidth}, H1Order);
//    
//    // do a few refinements:
//    // start with a non-uniform one, then refine uniformly from there
//    set<GlobalIndexType> cellsToRefine = {0};
//    mesh->hRefine(cellsToRefine);
//    
//    cellsToRefine = mesh->getActiveCellIDsGlobal(); // {1,2,3} -- 1 is on the right
//    mesh->hRefine(cellsToRefine);
//    
//    GlobalIndexType cellID = 6; // the leftmost cell
//    set<GlobalIndexType> myCells = mesh->cellIDsInPartition();
//    if (myCells.find(cellID) != myCells.end())
//    {
//      vector<RootedLabeledRefinementBranch> rootedBranches;
//      CellDataMigration::getCellHaloGeometry(mesh.get(), cellID, rootedBranches);
//      CellPtr cell = mesh->getTopology()->getCell(cellID);
//      set<IndexType> cellAncestors;
//      while (cell->getParent() != Teuchos::null)
//      {
//        cell = cell->getParent();
//        cellAncestors.insert(cell->cellIndex());
//      }
//      for (auto rootedBranch : rootedBranches)
//      {
//        LabeledRefinementBranch labeledBranch = rootedBranch.first;
//        auto labels = labeledBranch.second;
//        IndexType firstLeafIndex = labels[labels.size() - 1];
//        TEST_ASSERT(cellAncestors.find(firstLeafIndex) == cellAncestors.end());
//      }
//    }
//  }
  
  TEUCHOS_UNIT_TEST( CellDataMigration, PackAndUnpackPureGeometry_1D )
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 1;
    bool useConformingTraces = true;
    int H1Order = 2;
    int meshWidth = 2;
    PoissonFormulation form(spaceDim,useConformingTraces,PoissonFormulation::ULTRAWEAK);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0}, {meshWidth}, H1Order);
    
    // do a few refinements:
    // start with a non-uniform one, then refine uniformly from there
    set<GlobalIndexType> cellsToRefine = {0};
    mesh->hRefine(cellsToRefine);
    
    cellsToRefine = mesh->getActiveCellIDsGlobal();
    mesh->hRefine(cellsToRefine);
    
//    cellsToRefine = mesh->getActiveCellIDsGlobal();
//    mesh->hRefine(cellsToRefine);
    
    testPackAndUnpackGeometry(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( CellDataMigration, PackAndUnpackPureGeometry_2D )
  {
    MPIWrapper::CommWorld()->Barrier();
    int spaceDim = 2;
    bool useConformingTraces = true;
    int H1Order = 2;
    int meshWidth = 2;
    PoissonFormulation form(spaceDim,useConformingTraces,PoissonFormulation::ULTRAWEAK);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, {meshWidth,meshWidth}, H1Order);
    
    // do a few refinements:
    // start with a non-uniform one, then refine uniformly from there
    set<GlobalIndexType> cellsToRefine = {0};
    mesh->hRefine(cellsToRefine);

    cellsToRefine = mesh->getActiveCellIDsGlobal();
    mesh->hRefine(cellsToRefine);
    
    testPackAndUnpackGeometry(mesh, out, success);
  }
  
  TEUCHOS_UNIT_TEST( CellDataMigration, PackAndUnpackPureGeometry_3D )
  {
    int spaceDim = 3;
    bool useConformingTraces = true;
    int H1Order = 2;
    int meshWidth = 2;
    PoissonFormulation form(spaceDim,useConformingTraces,PoissonFormulation::ULTRAWEAK);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0,1.0}, {meshWidth,meshWidth,meshWidth}, H1Order);
    
    // do a few refinements:
    // start with a non-uniform one, then refine uniformly from there
    set<GlobalIndexType> cellsToRefine = {0};
    mesh->hRefine(cellsToRefine);
    
    cellsToRefine = mesh->getActiveCellIDsGlobal();
    mesh->hRefine(cellsToRefine);
    
    testPackAndUnpackGeometry(mesh, out, success);
  }
} // namespace
