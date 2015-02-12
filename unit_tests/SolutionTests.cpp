//
//  CellTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/18/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Intrepid_FieldContainer.hpp"

#include "MeshFactory.h"
#include "Cell.h"

#include "Solution.h"

#include "PoissonFormulation.h"

#include "GlobalDofAssignment.h"

namespace {

  vector<double> makeVertex(double v0, double v1) {
    vector<double> v;
    v.push_back(v0);
    v.push_back(v1);
    return v;
  }

  TEUCHOS_UNIT_TEST( Solution, ImportOffRankCellData )
  {
    int numCells = 8;
    int spaceDim = 1;
    // just want any bilinear form; we'll use Poisson
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim, useConformingTraces);
    
    double xLeft = 0, xRight = 1;
    
    int H1Order = 1, delta_k = 1;
    MeshPtr mesh = MeshFactory::intervalMesh(form.bf(), xLeft, xRight, numCells, H1Order, delta_k);
    
    MeshTopologyPtr meshTopo = mesh->getTopology();
    
    SolutionPtr soln = Solution::solution(mesh);
    
    set<GlobalIndexType> myCells = mesh->cellIDsInPartition();
    
    int rank = Teuchos::GlobalMPISession::getRank();
    
    // set up some dummy data
    for (set<GlobalIndexType>::iterator cellIDIt = myCells.begin(); cellIDIt != myCells.end(); cellIDIt++) {
      GlobalIndexType cellID = *cellIDIt;
      FieldContainer<double> cellDofs(mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
      cellDofs.initialize((double)rank);
      soln->setLocalCoefficientsForCell(cellID, cellDofs);
    }
    
    int otherRank = Teuchos::GlobalMPISession::getNProc() - 1 - rank;
    set<GlobalIndexType> cellIDsToRequest;
    if (otherRank != rank) {
      cellIDsToRequest = mesh->globalDofAssignment()->cellsInPartition(otherRank);
    }
    
//    cout << "On rank " << rank << ", otherRank = " << otherRank << endl;
    
    soln->importSolutionForOffRankCells(cellIDsToRequest);
    
    for (set<GlobalIndexType>::iterator cellIDIt = cellIDsToRequest.begin(); cellIDIt != cellIDsToRequest.end(); cellIDIt++) {
      GlobalIndexType cellID = *cellIDIt;
      FieldContainer<double> cellDofs = soln->allCoefficientsForCellID(cellID, false); // false: don't warn about off-rank requests
      
      TEST_ASSERT(cellDofs.size() == mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
      
      for (int i=0; i<cellDofs.size(); i++) {
        TEST_ASSERT(otherRank == cellDofs[i]);
      }
    }
  }
  TEUCHOS_UNIT_TEST( Solution, ProjectOnTensorMesh1D )
  {
    int tensorialDegree = 1;
    CellTopoPtr line_x_time = CellTopology::cellTopology(shards::getCellTopologyData<shards::Line<2> >(), tensorialDegree);

    vector<double> v00 = makeVertex(0,0);
    vector<double> v10 = makeVertex(1,0);
    vector<double> v20 = makeVertex(2,0);
    vector<double> v01 = makeVertex(0,1);
    vector<double> v11 = makeVertex(1,1);
    vector<double> v21 = makeVertex(2,1);

    vector< vector<double> > spaceTimeVertices;
    spaceTimeVertices.push_back(v00);
    spaceTimeVertices.push_back(v10);
    spaceTimeVertices.push_back(v20);
    spaceTimeVertices.push_back(v01);
    spaceTimeVertices.push_back(v11);
    spaceTimeVertices.push_back(v21);

    vector<unsigned> spaceTimeLine1VertexList;
    vector<unsigned> spaceTimeLine2VertexList;
    spaceTimeLine1VertexList.push_back(0);
    spaceTimeLine1VertexList.push_back(1);
    spaceTimeLine1VertexList.push_back(3);
    spaceTimeLine1VertexList.push_back(4);
    spaceTimeLine2VertexList.push_back(1);
    spaceTimeLine2VertexList.push_back(2);
    spaceTimeLine2VertexList.push_back(4);
    spaceTimeLine2VertexList.push_back(5);

    vector< vector<unsigned> > spaceTimeElementVertices;
    spaceTimeElementVertices.push_back(spaceTimeLine1VertexList);
    spaceTimeElementVertices.push_back(spaceTimeLine2VertexList);

    vector< CellTopoPtr > spaceTimeCellTopos;
    spaceTimeCellTopos.push_back(line_x_time);
    spaceTimeCellTopos.push_back(line_x_time);

    MeshGeometryPtr spaceTimeMeshGeometry = Teuchos::rcp( new MeshGeometry(spaceTimeVertices, spaceTimeElementVertices, spaceTimeCellTopos) );
    MeshTopologyPtr spaceTimeMeshTopology = Teuchos::rcp( new MeshTopology(spaceTimeMeshGeometry) );

    ////////////////////   DECLARE VARIABLES   ///////////////////////
    // define test variables
    VarFactory varFactory;
    VarPtr tau = varFactory.testVar("tau", HGRAD);
    VarPtr v = varFactory.testVar("v", HGRAD);

    // define trial variables
    VarPtr uhat = varFactory.fluxVar("uhat");
    VarPtr fhat = varFactory.fluxVar("fhat");
    VarPtr u = varFactory.fieldVar("u");
    VarPtr sigma = varFactory.fieldVar("sigma");

    ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
    BFPtr bf = Teuchos::rcp( new BF(varFactory) );
    // tau terms:
    bf->addTerm(sigma, tau);
    bf->addTerm(u, tau->dx());
    bf->addTerm(-uhat, tau);

    // v terms:
    bf->addTerm( sigma, v->dx() );
    bf->addTerm( fhat, v);

    ////////////////////   BUILD MESH   ///////////////////////
    int H1Order = 4, pToAdd = 2;
    Teuchos::RCP<Mesh> spaceTimeMesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );

    Teuchos::RCP<Solution> spaceTimeSolution = Teuchos::rcp( new Solution(spaceTimeMesh) );

    map<int, Teuchos::RCP<Function> > functionMap;
    functionMap[0] = Function::xn(1);
    functionMap[1] = Function::xn(1);
    functionMap[2] = Function::xn(1);
    functionMap[3] = Function::xn(1);
    spaceTimeSolution->projectOntoMesh(functionMap);

    TEST_ASSERT(true);
  }
} // namespace