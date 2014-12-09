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

namespace {
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
    
    // then try with some arbitrarily permuted cell numberings
  }
  
  TEUCHOS_UNIT_TEST( MeshTransferFunction, CellMapUnderRefinement)
  {
    // TODO: write this test
    // test to check that the cell mapping is correctly updated when the newMesh is refined
    
    // (may be worth checking that things are updated correctly when originalMesh is refined,
    //  but the newMesh one is the one that corresponds to the typical use case.)

  }
  
  TEUCHOS_UNIT_TEST( MeshTransferFunction, FunctionValuesPiecewiseConstant)
  {
    // TODO: write this test
    // test to check that functions are correctly valued
    
    // try with some functions that simply return the cellID
    // and check that this matches the cell map.
    
    // important to try this test on multiple MPI ranks...
  }
  
  TEUCHOS_UNIT_TEST( MeshTransferFunction, FunctionValues)
  {
    // TODO: write this test
    // test to check that functions are correctly valued
    
    // try with some functions that vary on the interface
    // (thereby checking that any permutations of the reference
    //  values are correctly imposed)
    
    // important to try this test on multiple MPI ranks...
  }
} // namespace