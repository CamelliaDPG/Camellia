//
//  MeshTransferFunctionTests
//  Camellia
//
//  Created by Nate Roberts on 11/27/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "PoissonFormulation.h"
#include "MeshFactory.h"

#include "MeshTransferFunction.h"
#include "CamelliaDebugUtility.h"

namespace {
  class CellIDFunction : public Function {
  public:
    CellIDFunction() : Function(0) {}
    
    virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
      
      for (int cellOrdinal=0; cellOrdinal<cellIDs.size(); cellOrdinal++) {
        for (int pointOrdinal=0; pointOrdinal<values.dimension(1); pointOrdinal++) {
          values(cellOrdinal,pointOrdinal) = cellIDs[cellOrdinal];
        }
      }
    }
  };
  
  TEUCHOS_UNIT_TEST( MeshTransferFunction, CellMap)
  {
    // test to check that the cell mapping is correct
    
    // first, try with some MeshFactory-generated quad meshes
    MeshPtr bottomMesh, topMesh;
    
    double x0 = 0, y0 = 0;
    
    int horizontalCells = 2, verticalCells = 1;
    double width = 1.0, height = 1.0;
    
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation formulation(spaceDim, conformingTraces);
    
    BFPtr poissonBF = formulation.bf();
    
    VarPtr phi_hat = formulation.phi_hat();
    
    MeshPtr mesh;
    
    int k = 1;
    int H1Order = k + 1;
    int delta_k = spaceDim;
    
    Teuchos::ParameterList pl;
    
    map<int,int> trialOrderEnhancements;
    Teuchos::RCP<BilinearForm> poissonBilinearForm = poissonBF;
    
    pl.set("useMinRule", true);
    pl.set("bf",poissonBilinearForm);
    pl.set("H1Order", H1Order);
    pl.set("delta_k", delta_k);
    pl.set("horizontalElements", horizontalCells);
    pl.set("verticalElements", verticalCells);
    pl.set("divideIntoTriangles", false);
    pl.set("useConformingTraces", conformingTraces);
    pl.set("trialOrderEnhancements", &trialOrderEnhancements);
    pl.set("x0",(double)x0);
    pl.set("y0",(double)y0);
    pl.set("width", width);
    pl.set("height",height);
    
    bottomMesh = MeshFactory::quadMesh(pl);
    
    double y_interface = y0 + height;
    pl.set("x0",(double)x0);
    pl.set("y0",(double)y_interface);
    topMesh = MeshFactory::quadMesh(pl);
    
    double elemHeight = height / verticalCells;
    double elemWidth = width / horizontalCells;
    
    double yCenterTopRowOfBottomMesh = y_interface - elemHeight / 2.0;
    double yCenterBottomRowOfTopMesh = y_interface + elemHeight / 2.0;
    double xCenter = elemWidth / 2.0;
    FieldContainer<double> midPointsBottomMesh(horizontalCells, spaceDim);
    FieldContainer<double> midPointsTopMesh(horizontalCells, spaceDim);
    for (int i=0; i<horizontalCells; i++) {
      midPointsBottomMesh(i,0) = xCenter;
      midPointsBottomMesh(i,1) = yCenterTopRowOfBottomMesh;
      midPointsTopMesh(i,0) = xCenter;
      midPointsTopMesh(i,1) = yCenterBottomRowOfTopMesh;
      xCenter += elemWidth;
    }
    
    vector<GlobalIndexType> cellIDs_bottomMesh = bottomMesh->cellIDsForPoints(midPointsBottomMesh, false);
    vector<GlobalIndexType> cellIDs_topMesh = topMesh->cellIDsForPoints(midPointsTopMesh, false);
    
    set<GlobalIndexType> myCellIDs_topMesh = topMesh->cellIDsInPartition();
    
    // hard-coded notion of which sides meet the interface.  Would be a cleaner test if we determined these
    // programmatically...
    unsigned topCellsSideOrdinal = 0;
    unsigned bottomCellsSideOrdinal = 2;
    
    typedef pair<GlobalIndexType, unsigned> CellSide;
    map<CellSide, CellSide> expectedMappingToNew, expectedMappingToOriginal;
    for (int i=0; i<horizontalCells; i++) {
      GlobalIndexType cellID_bottom = cellIDs_bottomMesh[i];
      GlobalIndexType cellID_top = cellIDs_topMesh[i];
      CellSide topCellSide = make_pair(cellID_top,topCellsSideOrdinal);
      CellSide bottomCellSide = make_pair(cellID_bottom, bottomCellsSideOrdinal);
      if (myCellIDs_topMesh.find(cellID_top) != myCellIDs_topMesh.end()) {
        expectedMappingToOriginal[topCellSide] = bottomCellSide;
        expectedMappingToNew[bottomCellSide] = topCellSide;
      }
    }
    
    MeshTransferFunction transferFunction(Function::zero(), bottomMesh, topMesh, y_interface);
    
    const map< pair<GlobalIndexType, unsigned>, pair<GlobalIndexType, unsigned> >* actualMapToOriginal = &transferFunction.mapToOriginalMesh();
    
    const map< pair<GlobalIndexType, unsigned>, pair<GlobalIndexType, unsigned> >* actualMapToNew = &transferFunction.mapToNewMesh();
    
    TEST_EQUALITY(actualMapToNew->size(), expectedMappingToNew.size());
    TEST_EQUALITY(actualMapToOriginal->size(), expectedMappingToOriginal.size());

    if (actualMapToNew->size() == expectedMappingToNew.size()) {
      for (map<CellSide, CellSide>::iterator expectedMapIt = expectedMappingToNew.begin();
           expectedMapIt != expectedMappingToNew.end(); expectedMapIt++) {
        CellSide originalCellSide = expectedMapIt->first;
        CellSide newCellSide = expectedMapIt->second;
        if (actualMapToNew->find(originalCellSide) == actualMapToNew->end()) {
          bool entryFound = false;
          TEST_ASSERT(entryFound);
        } else {
          CellSide newCellSideActual = actualMapToNew->find(originalCellSide)->second;
          TEST_EQUALITY(newCellSide, newCellSideActual);
        }
      }
    }

    if (actualMapToOriginal->size() == expectedMappingToOriginal.size()) {
      for (map<CellSide, CellSide>::iterator expectedMapIt = expectedMappingToOriginal.begin();
           expectedMapIt != expectedMappingToOriginal.end(); expectedMapIt++) {
        CellSide newCellSide = expectedMapIt->first;
        CellSide originalCellSide = expectedMapIt->second;
        if (actualMapToOriginal->find(newCellSide) == actualMapToOriginal->end()) {
          bool entryFound = false;
          TEST_ASSERT(entryFound);
        } else {
          CellSide originalCellSideActual = actualMapToOriginal->find(newCellSide)->second;
          TEST_EQUALITY(originalCellSide, originalCellSideActual);
        }
      }
    }
    
    // Now, check that findAncestralPairForNewMeshCellSide agrees with the above, and check that permutations are correctly determined
    for (set<GlobalIndexType>::iterator myCellIDIt = myCellIDs_topMesh.begin(); myCellIDIt != myCellIDs_topMesh.end(); myCellIDIt++) {
      GlobalIndexType myCellID = *myCellIDIt;
      CellPtr myCell = topMesh->getTopology()->getCell(myCellID);
      
      int sideCount = myCell->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
        pair<GlobalIndexType, unsigned> cellSide = make_pair(myCellID, sideOrdinal);
        
        pair<GlobalIndexType, unsigned> originalCellSide, newCellSideAncestor;
        unsigned permutation;
        
        bool matchFound = transferFunction.findAncestralPairForNewMeshCellSide(cellSide, newCellSideAncestor, originalCellSide, permutation);
        if (matchFound) {
          // since this is a test without refinements, newCellSideAncestor should be the same as cellSide
          TEST_EQUALITY(cellSide, newCellSideAncestor);
          
          CellSide expectedOriginalSide = expectedMappingToOriginal[cellSide];
          TEST_EQUALITY(expectedOriginalSide, originalCellSide);
          
          unsigned expectedPermutation = 1; // a flip, since in 2D edges are oriented counterclockwise
          TEST_EQUALITY(expectedPermutation, permutation);
        }
      }
    }
    
    // then try with some arbitrarily permuted cell numberings
  }
  
  TEUCHOS_UNIT_TEST( MeshTransferFunction, CellMapUnderRefinement)
  {
#ifdef HAVE_MPI
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
    Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
#endif
    // test to check that the cell mapping is correctly updated when the newMesh is refined
    
    // (may be worth checking that things are updated correctly when originalMesh is refined,
    //  but the newMesh one is the one that corresponds to the typical use case.)

    // test to check that the cell mapping is correct
    
    // first, try with some MeshFactory-generated quad meshes
    MeshPtr bottomMesh, topMesh;
    
    double x0 = 0, y0 = 0;
    
    int horizontalCells = 2, verticalCells = 1;
    double width = 1.0, height = 1.0;
    
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation formulation(spaceDim, conformingTraces);
    
    BFPtr poissonBF = formulation.bf();
    
    VarPtr phi_hat = formulation.phi_hat();
    
    MeshPtr mesh;
    
    int k = 1;
    int H1Order = k + 1;
    int delta_k = spaceDim;
    
    Teuchos::ParameterList pl;
    
    map<int,int> trialOrderEnhancements;
    Teuchos::RCP<BilinearForm> poissonBilinearForm = poissonBF;
    
    pl.set("useMinRule", true);
    pl.set("bf",poissonBilinearForm);
    pl.set("H1Order", H1Order);
    pl.set("delta_k", delta_k);
    pl.set("horizontalElements", horizontalCells);
    pl.set("verticalElements", verticalCells);
    pl.set("divideIntoTriangles", false);
    pl.set("useConformingTraces", conformingTraces);
    pl.set("trialOrderEnhancements", &trialOrderEnhancements);
    pl.set("x0",(double)x0);
    pl.set("y0",(double)y0);
    pl.set("width", width);
    pl.set("height",height);
    
    bottomMesh = MeshFactory::quadMesh(pl);
    
    double y_interface = y0 + height;
    pl.set("x0",(double)x0);
    pl.set("y0",(double)y_interface);
    topMesh = MeshFactory::quadMesh(pl);
    
    MeshTransferFunction transferFunction(Function::zero(), bottomMesh, topMesh, y_interface);
    
    double elemHeight = height / verticalCells;
    double elemWidth = width / horizontalCells;
    
    double yCenterTopRowOfBottomMesh = y_interface - elemHeight / 2.0;
    double yCenterBottomRowOfTopMesh = y_interface + elemHeight / 2.0;
    double xCenter = elemWidth / 2.0;
    FieldContainer<double> midPointsBottomMesh(horizontalCells, spaceDim);
    FieldContainer<double> midPointsTopMesh(horizontalCells, spaceDim);
    for (int i=0; i<horizontalCells; i++) {
      midPointsBottomMesh(i,0) = xCenter;
      midPointsBottomMesh(i,1) = yCenterTopRowOfBottomMesh;
      midPointsTopMesh(i,0) = xCenter;
      midPointsTopMesh(i,1) = yCenterBottomRowOfTopMesh;
      xCenter += elemWidth;
    }
    
    vector<GlobalIndexType> cellIDs_bottomMesh = bottomMesh->cellIDsForPoints(midPointsBottomMesh, false);
    vector<GlobalIndexType> cellIDs_topMesh = topMesh->cellIDsForPoints(midPointsTopMesh, false);
    
//    { //DEBUGGING
//      int rank = Teuchos::GlobalMPISession::getRank();
//      if (rank==1) {
//        cout << "Rank 1.\n";
//      }
//    }
    
    // refine topMesh
    set<GlobalIndexType> cellIDs;
    cellIDs.insert(0);
    topMesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternQuad());
    
    set<GlobalIndexType> myCellIDs_topMesh = topMesh->cellIDsInPartition();
    
    // hard-coded notion of which sides meet the interface.  Would be a cleaner test if we determined these
    // programmatically...
    unsigned topCellsSideOrdinal = 0;
    unsigned bottomCellsSideOrdinal = 2;
    
    typedef pair<GlobalIndexType, unsigned> CellSide;
    map<CellSide, CellSide> expectedMappingToNew, expectedMappingToOriginal;
    for (int i=0; i<horizontalCells; i++) {
      GlobalIndexType cellID_bottom = cellIDs_bottomMesh[i];
      GlobalIndexType cellID_top = cellIDs_topMesh[i];
      
      CellPtr topCell = topMesh->getTopology()->getCell(cellID_top);
      vector< pair<GlobalIndexType, unsigned> > topCellSides;
      if (topCell->isParent()) {
        topCellSides = topCell->childrenForSide(topCellsSideOrdinal);
      } else {
        topCellSides.push_back(make_pair(cellID_top,topCellsSideOrdinal) );
      }
      
      for (vector< pair<GlobalIndexType, unsigned> >::iterator topCellIDIt = topCellSides.begin(); topCellIDIt != topCellSides.end(); topCellIDIt++) {
        CellSide topCellSide = *topCellIDIt;
        CellSide topCellSideAncestor = make_pair(cellID_top, topCellsSideOrdinal); // may be identical to topCellSide, or may be its parent
        CellSide bottomCellSide = make_pair(cellID_bottom, bottomCellsSideOrdinal);
        
        if (myCellIDs_topMesh.find(topCellSide.first) != myCellIDs_topMesh.end()) {
          expectedMappingToOriginal[topCellSideAncestor] = bottomCellSide;
          expectedMappingToNew[bottomCellSide] = topCellSideAncestor;
        }
      }
    }
    
    const map< pair<GlobalIndexType, unsigned>, pair<GlobalIndexType, unsigned> >* actualMapToOriginal = &transferFunction.mapToOriginalMesh();
    
    const map< pair<GlobalIndexType, unsigned>, pair<GlobalIndexType, unsigned> >* actualMapToNew = &transferFunction.mapToNewMesh();
    
    TEST_EQUALITY(actualMapToNew->size(), expectedMappingToNew.size());
    TEST_EQUALITY(actualMapToOriginal->size(), expectedMappingToOriginal.size());
    
    if (actualMapToNew->size() == expectedMappingToNew.size()) {
      for (map<CellSide, CellSide>::iterator expectedMapIt = expectedMappingToNew.begin();
           expectedMapIt != expectedMappingToNew.end(); expectedMapIt++) {
        CellSide originalCellSide = expectedMapIt->first;
        CellSide newCellSide = expectedMapIt->second;
        if (actualMapToNew->find(originalCellSide) == actualMapToNew->end()) {
          bool entryFound = false;
          TEST_ASSERT(entryFound);
        } else {
          CellSide newCellSideActual = actualMapToNew->find(originalCellSide)->second;
          TEST_EQUALITY(newCellSide, newCellSideActual);
        }
      }
    } else {
      // DEBUGGING output
      int rank = Teuchos::GlobalMPISession::getRank();
      for (map<CellSide, CellSide>::iterator expectedMapIt = expectedMappingToNew.begin();
           expectedMapIt != expectedMappingToNew.end(); expectedMapIt++) {
        cout << "On rank " << rank << ", expectedMappingToNew[ (" << expectedMapIt->first.first << ", " << expectedMapIt->first.second << ") ] = ";
        cout << "(" << expectedMapIt->second.first << ", " << expectedMapIt->second.second << ")\n";
      }
    }
    
    if (actualMapToOriginal->size() == expectedMappingToOriginal.size()) {
      for (map<CellSide, CellSide>::iterator expectedMapIt = expectedMappingToOriginal.begin();
           expectedMapIt != expectedMappingToOriginal.end(); expectedMapIt++) {
        CellSide newCellSide = expectedMapIt->first;
        CellSide originalCellSide = expectedMapIt->second;
        if (actualMapToOriginal->find(newCellSide) == actualMapToOriginal->end()) {
          bool entryFound = false;
          TEST_ASSERT(entryFound);
        } else {
          CellSide originalCellSideActual = actualMapToOriginal->find(newCellSide)->second;
          TEST_EQUALITY(originalCellSide, originalCellSideActual);
        }
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( MeshTransferFunction, FunctionValuesPiecewiseConstant)
  {
    // test to check that functions are correctly valued
    
    // try with some functions that simply return the cellID
    // and check that this matches the cell map.
    
    // important to try this test on multiple MPI ranks...
    
    // then try with some arbitrarily permuted cell numberings

#ifdef HAVE_MPI
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
    Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
#endif
    // test to check that the cell mapping is correctly updated when the newMesh is refined
    
    // (may be worth checking that things are updated correctly when originalMesh is refined,
    //  but the newMesh one is the one that corresponds to the typical use case.)
    
    // test to check that the cell mapping is correct
    
    // first, try with some MeshFactory-generated quad meshes
    MeshPtr bottomMesh, topMesh;
    
    double x0 = 0, y0 = 0;
    
    int horizontalCells = 2, verticalCells = 1;
    double width = 1.0, height = 1.0;
    
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation formulation(spaceDim, conformingTraces);
    
    BFPtr poissonBF = formulation.bf();
    
    VarPtr phi_hat = formulation.phi_hat();
    
    MeshPtr mesh;
    
    int k = 1;
    int H1Order = k + 1;
    int delta_k = spaceDim;
    
    Teuchos::ParameterList pl;
    
    map<int,int> trialOrderEnhancements;
    Teuchos::RCP<BilinearForm> poissonBilinearForm = poissonBF;
    
    pl.set("useMinRule", true);
    pl.set("bf",poissonBilinearForm);
    pl.set("H1Order", H1Order);
    pl.set("delta_k", delta_k);
    pl.set("horizontalElements", horizontalCells);
    pl.set("verticalElements", verticalCells);
    pl.set("divideIntoTriangles", false);
    pl.set("useConformingTraces", conformingTraces);
    pl.set("trialOrderEnhancements", &trialOrderEnhancements);
    pl.set("x0",(double)x0);
    pl.set("y0",(double)y0);
    pl.set("width", width);
    pl.set("height",height);
    
    bottomMesh = MeshFactory::quadMesh(pl);
    
    double y_interface = y0 + height;
    pl.set("x0",(double)x0);
    pl.set("y0",(double)y_interface);
    topMesh = MeshFactory::quadMesh(pl);
    
    FunctionPtr originalCellIDFunction = Teuchos::rcp( new CellIDFunction );
    
    MeshTransferFunction transferFunction(originalCellIDFunction, bottomMesh, topMesh, y_interface);
    
    double elemHeight = height / verticalCells;
    double elemWidth = width / horizontalCells;
    
    double yCenterTopRowOfBottomMesh = y_interface - elemHeight / 2.0;
    double yCenterBottomRowOfTopMesh = y_interface + elemHeight / 2.0;
    double xCenter = elemWidth / 2.0;
    FieldContainer<double> midPointsBottomMesh(horizontalCells, spaceDim);
    FieldContainer<double> midPointsTopMesh(horizontalCells, spaceDim);
    for (int i=0; i<horizontalCells; i++) {
      midPointsBottomMesh(i,0) = xCenter;
      midPointsBottomMesh(i,1) = yCenterTopRowOfBottomMesh;
      midPointsTopMesh(i,0) = xCenter;
      midPointsTopMesh(i,1) = yCenterBottomRowOfTopMesh;
      xCenter += elemWidth;
    }
    
    vector<GlobalIndexType> cellIDs_bottomMesh = bottomMesh->cellIDsForPoints(midPointsBottomMesh, false);
    vector<GlobalIndexType> cellIDs_topMesh = topMesh->cellIDsForPoints(midPointsTopMesh, false);
    
    // refine topMesh
    set<GlobalIndexType> cellIDs;
    cellIDs.insert(0);
    topMesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternQuad());
    
    set<GlobalIndexType> myCellIDs_topMesh = topMesh->cellIDsInPartition();
    
    for (set<GlobalIndexType>::iterator myCellIDIt = myCellIDs_topMesh.begin(); myCellIDIt != myCellIDs_topMesh.end(); myCellIDIt++) {
      GlobalIndexType myCellID = *myCellIDIt;
      CellPtr myCell = topMesh->getTopology()->getCell(myCellID);
      
      int sideCount = myCell->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
        pair<GlobalIndexType, unsigned> cellSide = make_pair(myCellID, sideOrdinal);
        
        pair<GlobalIndexType, unsigned> originalCellSide, newCellSideAncestor;
        unsigned permutation;
        
        bool matchFound = transferFunction.findAncestralPairForNewMeshCellSide(cellSide, newCellSideAncestor, originalCellSide, permutation);
        
        if (matchFound) {
          BasisCachePtr myCellBasisCache = BasisCache::basisCacheForCell(topMesh, myCellID);
          BasisCachePtr myCellSideBasisCache = myCellBasisCache->getSideBasisCache(sideOrdinal);

          double expectedValue = (double) originalCellSide.first;
          
          int numPoints = myCellSideBasisCache->getRefCellPoints().dimension(0);
          int oneCell = 1;
          FieldContainer<double> actualValues(oneCell,numPoints);
          
          transferFunction.values(actualValues, myCellSideBasisCache);
          
          int cellOrdinal = 0;
          for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++) {
            TEST_FLOATING_EQUALITY(expectedValue, actualValues(cellOrdinal,pointOrdinal), 1e-15);
          }
        }
      }
    }
    
  }
  
  TEUCHOS_UNIT_TEST( MeshTransferFunction, FunctionValuesWithoutHangingNodes)
  {
    // test to check that functions are correctly valued
    
    // try with some functions that vary on the interface
    // (thereby checking that any permutations of the reference
    //  values are correctly imposed)
    
    // important to try this test on multiple MPI ranks...
    
    // test to check that functions are correctly valued
    
    // try with some functions that simply return the cellID
    // and check that this matches the cell map.
    
    // important to try this test on multiple MPI ranks...
    
    // then try with some arbitrarily permuted cell numberings
    
#ifdef HAVE_MPI
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
    Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
#endif
    // test to check that the cell mapping is correctly updated when the newMesh is refined
    
    // (may be worth checking that things are updated correctly when originalMesh is refined,
    //  but the newMesh one is the one that corresponds to the typical use case.)
    
    // test to check that the cell mapping is correct
    
    // first, try with some MeshFactory-generated quad meshes
    MeshPtr bottomMesh, topMesh;
    
    double x0 = 0, y0 = 0;
    
    int horizontalCells = 2, verticalCells = 1;
    double width = 1.0, height = 1.0;
    
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation formulation(spaceDim, conformingTraces);
    
    BFPtr poissonBF = formulation.bf();
    
    VarPtr phi_hat = formulation.phi_hat();
    
    MeshPtr mesh;
    
    int k = 1;
    int H1Order = k + 1;
    int delta_k = spaceDim;
    
    Teuchos::ParameterList pl;
    
    map<int,int> trialOrderEnhancements;
    Teuchos::RCP<BilinearForm> poissonBilinearForm = poissonBF;
    
    pl.set("useMinRule", true);
    pl.set("bf",poissonBilinearForm);
    pl.set("H1Order", H1Order);
    pl.set("delta_k", delta_k);
    pl.set("horizontalElements", horizontalCells);
    pl.set("verticalElements", verticalCells);
    pl.set("divideIntoTriangles", false);
    pl.set("useConformingTraces", conformingTraces);
    pl.set("trialOrderEnhancements", &trialOrderEnhancements);
    pl.set("x0",(double)x0);
    pl.set("y0",(double)y0);
    pl.set("width", width);
    pl.set("height",height);
    
    bottomMesh = MeshFactory::quadMesh(pl);
    
    double y_interface = y0 + height;
    pl.set("x0",(double)x0);
    pl.set("y0",(double)y_interface);
    topMesh = MeshFactory::quadMesh(pl);
    
    FunctionPtr myFunction = Function::xn(1);
    MeshTransferFunction transferFunction(myFunction, bottomMesh, topMesh, y_interface);
    
    double elemHeight = height / verticalCells;
    double elemWidth = width / horizontalCells;
    
    double yCenterTopRowOfBottomMesh = y_interface - elemHeight / 2.0;
    double yCenterBottomRowOfTopMesh = y_interface + elemHeight / 2.0;
    double xCenter = elemWidth / 2.0;
    FieldContainer<double> midPointsBottomMesh(horizontalCells, spaceDim);
    FieldContainer<double> midPointsTopMesh(horizontalCells, spaceDim);
    for (int i=0; i<horizontalCells; i++) {
      midPointsBottomMesh(i,0) = xCenter;
      midPointsBottomMesh(i,1) = yCenterTopRowOfBottomMesh;
      midPointsTopMesh(i,0) = xCenter;
      midPointsTopMesh(i,1) = yCenterBottomRowOfTopMesh;
      xCenter += elemWidth;
    }
    
    vector<GlobalIndexType> cellIDs_bottomMesh = bottomMesh->cellIDsForPoints(midPointsBottomMesh, false);
    vector<GlobalIndexType> cellIDs_topMesh = topMesh->cellIDsForPoints(midPointsTopMesh, false);
    
    set<GlobalIndexType> myCellIDs_topMesh = topMesh->cellIDsInPartition();
    
    for (set<GlobalIndexType>::iterator myCellIDIt = myCellIDs_topMesh.begin(); myCellIDIt != myCellIDs_topMesh.end(); myCellIDIt++) {
      GlobalIndexType myCellID = *myCellIDIt;
      CellPtr myCell = topMesh->getTopology()->getCell(myCellID);
      
      int sideCount = myCell->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
        pair<GlobalIndexType, unsigned> cellSide = make_pair(myCellID, sideOrdinal);
        
        pair<GlobalIndexType, unsigned> originalCellSide, newCellSideAncestor;
        unsigned permutation;
        
        bool matchFound = transferFunction.findAncestralPairForNewMeshCellSide(cellSide, newCellSideAncestor, originalCellSide, permutation);
        
        if (matchFound) {
          BasisCachePtr myCellBasisCache = BasisCache::basisCacheForCell(topMesh, myCellID);
          BasisCachePtr myCellSideBasisCache = myCellBasisCache->getSideBasisCache(sideOrdinal);
          
          int numPoints = myCellSideBasisCache->getRefCellPoints().dimension(0);
          int oneCell = 1;
          FieldContainer<double> expectedValues(oneCell,numPoints);
          FieldContainer<double> actualValues(oneCell,numPoints);
          
          myFunction->values(expectedValues, myCellSideBasisCache);
          transferFunction.values(actualValues, myCellSideBasisCache);
          
//          { // DEBUGGING
//            cout << "Physical points:\n" << myCellSideBasisCache->getPhysicalCubaturePoints();
//            cout << "Expected values:\n" << expectedValues;
//            cout << "Actual values:\n" << actualValues;
//          }
          
          TEST_COMPARE_FLOATING_ARRAYS(actualValues, expectedValues, 1e-15);
        }
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( MeshTransferFunction, FunctionValuesWithHangingNodes)
  {
    // TODO: write this test
    // test to check that functions are correctly valued
    
    // try with some functions that vary on the interface
    // (thereby checking that any permutations of the reference
    //  values are correctly imposed)
    
    // important to try this test on multiple MPI ranks...
    
    // test to check that functions are correctly valued
    
    // try with some functions that simply return the cellID
    // and check that this matches the cell map.
    
    // important to try this test on multiple MPI ranks...
    
    // then try with some arbitrarily permuted cell numberings
    
#ifdef HAVE_MPI
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
    Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
#endif
    // test to check that the cell mapping is correctly updated when the newMesh is refined
    
    // (may be worth checking that things are updated correctly when originalMesh is refined,
    //  but the newMesh one is the one that corresponds to the typical use case.)
    
    // test to check that the cell mapping is correct
    
    // first, try with some MeshFactory-generated quad meshes
    MeshPtr bottomMesh, topMesh;
    
    double x0 = 0, y0 = 0;
    
    int horizontalCells = 2, verticalCells = 1;
    double width = 1.0, height = 1.0;
    
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation formulation(spaceDim, conformingTraces);
    
    BFPtr poissonBF = formulation.bf();
    
    VarPtr phi_hat = formulation.phi_hat();
    
    MeshPtr mesh;
    
    int k = 1;
    int H1Order = k + 1;
    int delta_k = spaceDim;
    
    Teuchos::ParameterList pl;
    
    map<int,int> trialOrderEnhancements;
    Teuchos::RCP<BilinearForm> poissonBilinearForm = poissonBF;
    
    pl.set("useMinRule", true);
    pl.set("bf",poissonBilinearForm);
    pl.set("H1Order", H1Order);
    pl.set("delta_k", delta_k);
    pl.set("horizontalElements", horizontalCells);
    pl.set("verticalElements", verticalCells);
    pl.set("divideIntoTriangles", false);
    pl.set("useConformingTraces", conformingTraces);
    pl.set("trialOrderEnhancements", &trialOrderEnhancements);
    pl.set("x0",(double)x0);
    pl.set("y0",(double)y0);
    pl.set("width", width);
    pl.set("height",height);
    
    bottomMesh = MeshFactory::quadMesh(pl);
    
    double y_interface = y0 + height;
    pl.set("x0",(double)x0);
    pl.set("y0",(double)y_interface);
    topMesh = MeshFactory::quadMesh(pl);
    
    FunctionPtr myFunction = Function::xn(1);
    MeshTransferFunction transferFunction(myFunction, bottomMesh, topMesh, y_interface);
    
    double elemHeight = height / verticalCells;
    double elemWidth = width / horizontalCells;
    
    double yCenterTopRowOfBottomMesh = y_interface - elemHeight / 2.0;
    double yCenterBottomRowOfTopMesh = y_interface + elemHeight / 2.0;
    double xCenter = elemWidth / 2.0;
    FieldContainer<double> midPointsBottomMesh(horizontalCells, spaceDim);
    FieldContainer<double> midPointsTopMesh(horizontalCells, spaceDim);
    for (int i=0; i<horizontalCells; i++) {
      midPointsBottomMesh(i,0) = xCenter;
      midPointsBottomMesh(i,1) = yCenterTopRowOfBottomMesh;
      midPointsTopMesh(i,0) = xCenter;
      midPointsTopMesh(i,1) = yCenterBottomRowOfTopMesh;
      xCenter += elemWidth;
    }
    
    vector<GlobalIndexType> cellIDs_bottomMesh = bottomMesh->cellIDsForPoints(midPointsBottomMesh, false);
    vector<GlobalIndexType> cellIDs_topMesh = topMesh->cellIDsForPoints(midPointsTopMesh, false);
    
    // refine topMesh
    set<GlobalIndexType> cellIDs;
    cellIDs.insert(0);
    topMesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternQuad());
    
    set<GlobalIndexType> myCellIDs_topMesh = topMesh->cellIDsInPartition();
    
    for (set<GlobalIndexType>::iterator myCellIDIt = myCellIDs_topMesh.begin(); myCellIDIt != myCellIDs_topMesh.end(); myCellIDIt++) {
      GlobalIndexType myCellID = *myCellIDIt;
      CellPtr myCell = topMesh->getTopology()->getCell(myCellID);
      
      int sideCount = myCell->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
        pair<GlobalIndexType, unsigned> cellSide = make_pair(myCellID, sideOrdinal);
        
        pair<GlobalIndexType, unsigned> originalCellSide, newCellSideAncestor;
        unsigned permutation;
        
        bool matchFound = transferFunction.findAncestralPairForNewMeshCellSide(cellSide, newCellSideAncestor, originalCellSide, permutation);
        
        if (matchFound) {
          BasisCachePtr myCellBasisCache = BasisCache::basisCacheForCell(topMesh, myCellID);
          BasisCachePtr myCellSideBasisCache = myCellBasisCache->getSideBasisCache(sideOrdinal);
        
          int numPoints = myCellSideBasisCache->getRefCellPoints().dimension(0);
          int oneCell = 1;
          FieldContainer<double> expectedValues(oneCell,numPoints);
          FieldContainer<double> actualValues(oneCell,numPoints);
          
          myFunction->values(expectedValues, myCellSideBasisCache);
          transferFunction.values(actualValues, myCellSideBasisCache);
          
//          { // DEBUGGING
//            cout << "Physical points:\n" << myCellSideBasisCache->getPhysicalCubaturePoints();
//            cout << "Expected values:\n" << expectedValues;
//            cout << "Actual values:\n" << actualValues;
//          }
          
          TEST_COMPARE_FLOATING_ARRAYS(actualValues, expectedValues, 1e-15);
        }
      }
    }
  }
} // namespace