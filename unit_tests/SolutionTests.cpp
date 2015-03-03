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

#include "MeshTools.h"

#include "HDF5Exporter.h"

#include "CamelliaDebugUtility.h"

namespace {

  vector<double> makeVertex(double v0) {
    vector<double> v;
    v.push_back(v0);
    return v;
  }

  vector<double> makeVertex(double v0, double v1) {
    vector<double> v;
    v.push_back(v0);
    v.push_back(v1);
    return v;
  }

  vector<double> makeVertex(double v0, double v1, double v2) {
    vector<double> v;
    v.push_back(v0);
    v.push_back(v1);
    v.push_back(v2);
    return v;
  }

  vector<double> makeVertex(double v0, double v1, double v2, double v3) {
    vector<double> v;
    v.push_back(v0);
    v.push_back(v1);
    v.push_back(v2);
    v.push_back(v3);
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
  
  void testProjectOnTensorMesh(CellTopoPtr spaceTopo, int H1Order, Teuchos::FancyOStream &out, bool &success) {
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo->getShardsTopology(), spaceTopo->getTensorialDegree() + 1);
    
    // TODO: write a generalization of the 1D/2D/3D tests below here, and invoke this method from each of those methods
    // (reduce redundant code...)
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
    spaceTimeVertices.push_back(v00); // 0
    spaceTimeVertices.push_back(v10); // 1
    spaceTimeVertices.push_back(v20); // 2
    spaceTimeVertices.push_back(v01); // 3
    spaceTimeVertices.push_back(v11); // 4
    spaceTimeVertices.push_back(v21); // 5

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
    MeshPtr spaceTimeMesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );

    SolutionPtr spaceTimeSolution = Solution::solution(spaceTimeMesh);

    FunctionPtr n = Function::normal();
    FunctionPtr parity = Function::sideParity();
    
    map<int, FunctionPtr > functionMap;
    FunctionPtr x = Function::xn(1);
    FunctionPtr xx = Function::vectorize(x, x);
    
    functionMap[uhat->ID()] = xx * n * parity;
    functionMap[fhat->ID()] = xx * n * parity;
    functionMap[u->ID()] = x;
    functionMap[sigma->ID()] = x;
    spaceTimeSolution->projectOntoMesh(functionMap);

    for (GlobalIndexType cellID=0; cellID <= 1; cellID++) {
      cout << "CellID " << cellID << " info:\n";
      FieldContainer<double> localCoefficients = spaceTimeSolution->allCoefficientsForCellID(cellID);
      
      DofOrderingPtr trialOrder = spaceTimeMesh->getElementType(cellID)->trialOrderPtr;
      
      Camellia::printLabeledDofCoefficients(varFactory, trialOrder, localCoefficients);
    }

    double tol = 1e-14;
    for (map<int, FunctionPtr >::iterator entryIt = functionMap.begin(); entryIt != functionMap.end(); entryIt++) {
      int trialID = entryIt->first;
      VarPtr trialVar = varFactory.trial(trialID);
      FunctionPtr f_expected = entryIt->second;
      FunctionPtr f_actual = Function::solution(trialVar, spaceTimeSolution);
      
      int cubDegreeEnrichment = 0;
      bool spatialSidesOnly = true;
      
      double err_L2 = (f_actual - f_expected)->l2norm(spaceTimeMesh, cubDegreeEnrichment, spatialSidesOnly);
      TEST_COMPARE(err_L2, <, tol);
    }
    
//    map<GlobalIndexType,GlobalIndexType> cellMap_t0, cellMap_t1;
//    MeshPtr meshSlice_t0 = MeshTools::timeSliceMesh(spaceTimeMesh, 0, cellMap_t0, H1Order);
//    FunctionPtr sliceFunction_t0 = MeshTools::timeSliceFunction(spaceTimeMesh, cellMap_t0, Function::xn(1), 0);
//    HDF5Exporter exporter0(meshSlice_t0, "Function1D_t0");
//    exporter0.exportFunction(sliceFunction_t0, "x");
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnTensorMesh2D )
  {
    int tensorialDegree = 1;
    CellTopoPtr quad_x_time = CellTopology::cellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >(), tensorialDegree);
    CellTopoPtr tri_x_time = CellTopology::cellTopology(shards::getCellTopologyData<shards::Triangle<3> >(), tensorialDegree);

    // let's draw a little house
    vector<double> v00 = makeVertex(-1,0,0);
    vector<double> v10 = makeVertex(1,0,0);
    vector<double> v20 = makeVertex(1,2,0);
    vector<double> v30 = makeVertex(-1,2,0);
    vector<double> v40 = makeVertex(0.0,3,0);
    vector<double> v01 = makeVertex(-1,0,1);
    vector<double> v11 = makeVertex(1,0,1);
    vector<double> v21 = makeVertex(1,2,1);
    vector<double> v31 = makeVertex(-1,2,1);
    vector<double> v41 = makeVertex(0.0,3,1);

    vector< vector<double> > spaceTimeVertices;
    spaceTimeVertices.push_back(v00);
    spaceTimeVertices.push_back(v10);
    spaceTimeVertices.push_back(v20);
    spaceTimeVertices.push_back(v30);
    spaceTimeVertices.push_back(v40);
    spaceTimeVertices.push_back(v01);
    spaceTimeVertices.push_back(v11);
    spaceTimeVertices.push_back(v21);
    spaceTimeVertices.push_back(v31);
    spaceTimeVertices.push_back(v41);

    vector<unsigned> spaceTimeQuadVertexList;
    spaceTimeQuadVertexList.push_back(0);
    spaceTimeQuadVertexList.push_back(1);
    spaceTimeQuadVertexList.push_back(2);
    spaceTimeQuadVertexList.push_back(3);
    spaceTimeQuadVertexList.push_back(5);
    spaceTimeQuadVertexList.push_back(6);
    spaceTimeQuadVertexList.push_back(7);
    spaceTimeQuadVertexList.push_back(8);
    vector<unsigned> spaceTimeTriVertexList;
    spaceTimeTriVertexList.push_back(3);
    spaceTimeTriVertexList.push_back(2);
    spaceTimeTriVertexList.push_back(4);
    spaceTimeTriVertexList.push_back(8);
    spaceTimeTriVertexList.push_back(7);
    spaceTimeTriVertexList.push_back(9);

    vector< vector<unsigned> > spaceTimeElementVertices;
    spaceTimeElementVertices.push_back(spaceTimeQuadVertexList);
    spaceTimeElementVertices.push_back(spaceTimeTriVertexList);

    vector< CellTopoPtr > spaceTimeCellTopos;
    spaceTimeCellTopos.push_back(quad_x_time);
    spaceTimeCellTopos.push_back(tri_x_time);

    MeshGeometryPtr spaceTimeMeshGeometry = Teuchos::rcp( new MeshGeometry(spaceTimeVertices, spaceTimeElementVertices, spaceTimeCellTopos) );
    MeshTopologyPtr spaceTimeMeshTopology = Teuchos::rcp( new MeshTopology(spaceTimeMeshGeometry) );

    ////////////////////   DECLARE VARIABLES   ///////////////////////
    // define test variables
    VarFactory varFactory;
    VarPtr tau = varFactory.testVar("tau", HDIV);
    VarPtr v = varFactory.testVar("v", HGRAD);

    // define trial variables
    VarPtr uhat = varFactory.traceVar("uhat");
    VarPtr fhat = varFactory.fluxVar("fhat");
    VarPtr u = varFactory.fieldVar("u");
    VarPtr sigma = varFactory.fieldVar("sigma", VECTOR_L2);

    ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
    BFPtr bf = Teuchos::rcp( new BF(varFactory) );
    // tau terms:
    bf->addTerm(sigma, tau);
    bf->addTerm(u, tau->div());
    bf->addTerm(-uhat, tau->dot_normal());

    // v terms:
    bf->addTerm( sigma, v->grad() );
    bf->addTerm( fhat, v);

    ////////////////////   BUILD MESH   ///////////////////////
    int H1Order = 4, pToAdd = 2;
    Teuchos::RCP<Mesh> spaceTimeMesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );

    Teuchos::RCP<Solution> spaceTimeSolution = Teuchos::rcp( new Solution(spaceTimeMesh) );

    map<int, Teuchos::RCP<Function> > functionMap;
    functionMap[uhat->ID()] = Function::xn(1);
    functionMap[fhat->ID()] = Function::xn(1);
    functionMap[u->ID()] = Function::xn(1);
    functionMap[sigma->ID()] = Function::xn(1);
    spaceTimeSolution->projectOntoMesh(functionMap);
    
    double tol = 1e-14;
    for (map<int, Teuchos::RCP<Function> >::iterator entryIt = functionMap.begin(); entryIt != functionMap.end(); entryIt++) {
      int trialID = entryIt->first;
      VarPtr trialVar = varFactory.trial(trialID);
      FunctionPtr f_expected = entryIt->second;
      FunctionPtr f_actual = Function::solution(trialVar, spaceTimeSolution);
      
      double err_L2 = (f_actual - f_expected)->l2norm(spaceTimeMesh);
      TEST_COMPARE(err_L2, <, tol);
    }
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnTensorMesh3D )
  {
    int tensorialDegree = 1;
    CellTopoPtr hex_x_time = CellTopology::cellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >(), tensorialDegree);

    // let's draw a little box
    vector<double> v00 = makeVertex(0,0,0,0);
    vector<double> v10 = makeVertex(1,0,0,0);
    vector<double> v20 = makeVertex(1,1,0,0);
    vector<double> v30 = makeVertex(0,1,0,0);
    vector<double> v40 = makeVertex(0,0,1,0);
    vector<double> v50 = makeVertex(1,0,1,0);
    vector<double> v60 = makeVertex(1,1,1,0);
    vector<double> v70 = makeVertex(0,1,1,0);
    vector<double> v01 = makeVertex(0,0,0,1);
    vector<double> v11 = makeVertex(1,0,0,1);
    vector<double> v21 = makeVertex(1,1,0,1);
    vector<double> v31 = makeVertex(0,1,0,1);
    vector<double> v41 = makeVertex(0,0,1,1);
    vector<double> v51 = makeVertex(1,0,1,1);
    vector<double> v61 = makeVertex(1,1,1,1);
    vector<double> v71 = makeVertex(0,1,1,1);

    vector< vector<double> > spaceTimeVertices;
    spaceTimeVertices.push_back(v00);
    spaceTimeVertices.push_back(v10);
    spaceTimeVertices.push_back(v20);
    spaceTimeVertices.push_back(v30);
    spaceTimeVertices.push_back(v40);
    spaceTimeVertices.push_back(v50);
    spaceTimeVertices.push_back(v60);
    spaceTimeVertices.push_back(v70);
    spaceTimeVertices.push_back(v01);
    spaceTimeVertices.push_back(v11);
    spaceTimeVertices.push_back(v21);
    spaceTimeVertices.push_back(v31);
    spaceTimeVertices.push_back(v41);
    spaceTimeVertices.push_back(v51);
    spaceTimeVertices.push_back(v61);
    spaceTimeVertices.push_back(v71);

    vector<unsigned> spaceTimeHexVertexList;
    spaceTimeHexVertexList.push_back(0);
    spaceTimeHexVertexList.push_back(1);
    spaceTimeHexVertexList.push_back(2);
    spaceTimeHexVertexList.push_back(3);
    spaceTimeHexVertexList.push_back(4);
    spaceTimeHexVertexList.push_back(5);
    spaceTimeHexVertexList.push_back(6);
    spaceTimeHexVertexList.push_back(7);
    spaceTimeHexVertexList.push_back(8);
    spaceTimeHexVertexList.push_back(9);
    spaceTimeHexVertexList.push_back(10);
    spaceTimeHexVertexList.push_back(11);
    spaceTimeHexVertexList.push_back(12);
    spaceTimeHexVertexList.push_back(13);
    spaceTimeHexVertexList.push_back(14);
    spaceTimeHexVertexList.push_back(15);

    vector< vector<unsigned> > spaceTimeElementVertices;
    spaceTimeElementVertices.push_back(spaceTimeHexVertexList);

    vector< CellTopoPtr > spaceTimeCellTopos;
    spaceTimeCellTopos.push_back(hex_x_time);

    MeshGeometryPtr spaceTimeMeshGeometry = Teuchos::rcp( new MeshGeometry(spaceTimeVertices, spaceTimeElementVertices, spaceTimeCellTopos) );
    MeshTopologyPtr spaceTimeMeshTopology = Teuchos::rcp( new MeshTopology(spaceTimeMeshGeometry) );

    ////////////////////   DECLARE VARIABLES   ///////////////////////
    // define test variables
    VarFactory varFactory;
    VarPtr tau = varFactory.testVar("tau", HDIV);
    VarPtr v = varFactory.testVar("v", HGRAD);

    // define trial variables
    VarPtr uhat = varFactory.traceVar("uhat");
    VarPtr fhat = varFactory.fluxVar("fhat");
    VarPtr u = varFactory.fieldVar("u");
    VarPtr sigma = varFactory.fieldVar("sigma", VECTOR_L2);

    ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
    BFPtr bf = Teuchos::rcp( new BF(varFactory) );
    // tau terms:
    bf->addTerm(sigma, tau);
    bf->addTerm(u, tau->div());
    bf->addTerm(-uhat, tau->dot_normal());

    // v terms:
    bf->addTerm( sigma, v->grad() );
    bf->addTerm( fhat, v);

    ////////////////////   BUILD MESH   ///////////////////////
    int H1Order = 4, pToAdd = 2;
    Teuchos::RCP<Mesh> spaceTimeMesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );

    Teuchos::RCP<Solution> spaceTimeSolution = Teuchos::rcp( new Solution(spaceTimeMesh) );

    map<int, Teuchos::RCP<Function> > functionMap;
    functionMap[uhat->ID()] = Function::xn(1);
    functionMap[fhat->ID()] = Function::xn(1);
    functionMap[u->ID()] = Function::xn(1);
    functionMap[sigma->ID()] = Function::xn(1);
    spaceTimeSolution->projectOntoMesh(functionMap);
    
    double tol = 1e-14;
    for (map<int, Teuchos::RCP<Function> >::iterator entryIt = functionMap.begin(); entryIt != functionMap.end(); entryIt++) {
      int trialID = entryIt->first;
      VarPtr trialVar = varFactory.trial(trialID);
      FunctionPtr f_expected = entryIt->second;
      FunctionPtr f_actual = Function::solution(trialVar, spaceTimeSolution);
      
      double err_L2 = (f_actual - f_expected)->l2norm(spaceTimeMesh);
      TEST_COMPARE(err_L2, <, tol);
    }
  }
} // namespace