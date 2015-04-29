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

#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "Cell.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "MeshTools.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "StokesVGPFormulation.h"
#include "Projector.h"

using namespace Camellia;
using namespace Intrepid;

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

  void testProjectTraceOnTensorMesh(CellTopoPtr spaceTopo, int H1Order, FunctionPtr f, VarType traceOrFlux,
                                    Teuchos::FancyOStream &out, bool &success) {
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo->getShardsTopology(), spaceTopo->getTensorialDegree() + 1);

    // very simply, take a one-element, reference space mesh, project a polynomial onto a trace variable,
    // and check whether we correctly project a function onto it...

    // define a VarFactory with just a trace variable, and an HGRAD test
    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr v = vf->testVar("v", HGRAD);
    VarPtr uhat;
    if (traceOrFlux == TRACE)
      uhat = vf->traceVar("uhat");
    else if (traceOrFlux == FLUX)
      uhat = vf->fluxVar("u_n");

    BFPtr bf = BF::bf(vf);

    vector< vector<double> > refCellNodes;
    CamelliaCellTools::refCellNodesForTopology(refCellNodes,spaceTimeTopo);

    int spaceDim = spaceTimeTopo->getDimension();
    int pToAdd = 1; // for this test, doesn't really affect much

    MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(spaceDim) );
    meshTopo->addCell(spaceTimeTopo, refCellNodes);

    MeshPtr mesh = Teuchos::rcp( new Mesh (meshTopo, bf, H1Order, pToAdd) );

    SolutionPtr soln = Solution::solution(mesh);
    map<int, FunctionPtr > functionMap;
    functionMap[uhat->ID()] = f;

    soln->projectOntoMesh(functionMap);

    // Now, manually project onto the basis for the trace to compute some expected coefficients
    Intrepid::FieldContainer<double> basisCoefficientsExpected;

    double tol = 1e-15;

    set<GlobalIndexType> cellIDs = mesh->cellIDsInPartition();

    for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
      GlobalIndexType cellID = *cellIDIt;
      DofOrderingPtr trialOrder = mesh->getElementType(cellID)->trialOrderPtr;

      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);

      for (int sideOrdinal = 0; sideOrdinal < spaceTimeTopo->getSideCount(); sideOrdinal++) {
        CellTopoPtr sideTopo = spaceTimeTopo->getSide(sideOrdinal);
        BasisPtr sideBasis = trialOrder->getBasis(uhat->ID(), sideOrdinal);
        BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);

        int numCells = 1;
        basisCoefficientsExpected.resize(numCells,sideBasis->getCardinality());

        Projector<double>::projectFunctionOntoBasis(basisCoefficientsExpected, f, sideBasis, sideBasisCache);

        FieldContainer<double> basisCoefficientsActual(sideBasis->getCardinality());

        soln->solnCoeffsForCellID(basisCoefficientsActual,cellID,uhat->ID(),sideOrdinal);

        for (int basisOrdinal=0; basisOrdinal < sideBasis->getCardinality(); basisOrdinal++) {
          double diff = basisCoefficientsActual[basisOrdinal] - basisCoefficientsExpected[basisOrdinal];
          TEST_COMPARE(abs(diff),<,tol);
        }
      }
//      { // DEBUGGING:
//        cout << "CellID " << cellID << " info:\n";
//        FieldContainer<double> localCoefficients = soln->allCoefficientsForCellID(cellID);
//        Camellia::printLabeledDofCoefficients(vf, trialOrder, localCoefficients);
//      }
    }
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectTraceOnOneElementTensorMesh1D )
  {
    int H1Order = 2;
    FunctionPtr f = Function::xn(1);
    testProjectTraceOnTensorMesh(CellTopology::line(), H1Order, f, TRACE, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectFluxOnOneElementTensorMesh1D )
  {
    int H1Order = 3;
    FunctionPtr n = Function::normalSpaceTime();
    FunctionPtr parity = Function::sideParity();
    FunctionPtr f = Function::xn(2) * n->x() * parity + Function::yn(1) * n->y() * parity;
    testProjectTraceOnTensorMesh(CellTopology::line(), H1Order, f, FLUX, out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, ProjectOnTensorMesh1D )
  {
    int tensorialDegree = 1;
    CellTopoPtr line_x_time = CellTopology::cellTopology(CellTopology::line(), tensorialDegree);

    vector<double> v00 = makeVertex(-1,-1);
    vector<double> v10 = makeVertex(1,-1);
    vector<double> v20 = makeVertex(2,-1);
    vector<double> v01 = makeVertex(-1,1);
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
    VarFactoryPtr varFactory = VarFactory::varFactory();
    VarPtr v = varFactory->testVar("v", HGRAD);

    // define trial variables
    VarPtr uhat = varFactory->fluxVar("uhat");

    ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
    BFPtr bf = Teuchos::rcp( new BF(varFactory) );

    ////////////////////   BUILD MESH   ///////////////////////
    int H1Order = 3, pToAdd = 1;
    MeshPtr spaceTimeMesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );

    SolutionPtr spaceTimeSolution = Solution::solution(spaceTimeMesh);

    FunctionPtr n = Function::normalSpaceTime();
    FunctionPtr parity = Function::sideParity();
    FunctionPtr f = Function::xn(2) * n->x() * parity + Function::yn(1) * n->y() * parity;

    map<int, FunctionPtr > functionMap;

    functionMap[uhat->ID()] = f;
    spaceTimeSolution->projectOntoMesh(functionMap);

//    for (GlobalIndexType cellID=0; cellID <= 1; cellID++) {
//      cout << "CellID " << cellID << " info:\n";
//      FieldContainer<double> localCoefficients = spaceTimeSolution->allCoefficientsForCellID(cellID);
//
//      DofOrderingPtr trialOrder = spaceTimeMesh->getElementType(cellID)->trialOrderPtr;
//
//      Camellia::printLabeledDofCoefficients(varFactory, trialOrder, localCoefficients);
//    }

    double tol = 1e-14;
    for (map<int, FunctionPtr >::iterator entryIt = functionMap.begin(); entryIt != functionMap.end(); entryIt++) {
      int trialID = entryIt->first;
      VarPtr trialVar = varFactory->trial(trialID);
      FunctionPtr f_expected = entryIt->second;
      FunctionPtr f_actual = Function::solution(trialVar, spaceTimeSolution);
      if (trialVar->varType() == FLUX) {
        // then Function::solution() will have included a parity weight, basically on the idea that we're also multiplying by normals
        // in our usage of the solution data.  (It may be that this is not the best way to do this.)

        // For this test, though, we want to reverse that:
        f_actual = parity * f_actual;
      }

      int cubDegreeEnrichment = 0;
      bool spatialSidesOnly = false;

      double err_L2 = (f_actual - f_expected)->l2norm(spaceTimeMesh, cubDegreeEnrichment, spatialSidesOnly);
      TEST_COMPARE(err_L2, <, tol);

      // pointwise comparison
      set<GlobalIndexType> cellIDs = spaceTimeMesh->cellIDsInPartition();
      for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
        GlobalIndexType cellID = *cellIDIt;
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(spaceTimeMesh, cellID);
        if ((trialVar->varType() == FLUX) || (trialVar->varType() == TRACE)) {
          int sideCount = spaceTimeMesh->getElementType(cellID)->cellTopoPtr->getSideCount();
          for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
            BasisCachePtr sideCache = basisCache->getSideBasisCache(sideOrdinal);
            FieldContainer<double> physicalPoints = sideCache->getPhysicalCubaturePoints();
            int numPoints = physicalPoints.dimension(1); // (C,P,D)
            out << "physicalPoints for side " << sideOrdinal << ":\n" << physicalPoints;
            FieldContainer<double> actualValues(1,numPoints); // assumes scalar-valued
            FieldContainer<double> expectedValues(1,numPoints); // assumes scalar-valued
            f_actual->values(actualValues, sideCache);
            f_expected->values(expectedValues, sideCache);
            TEST_COMPARE_FLOATING_ARRAYS(expectedValues, actualValues, tol);
          }
        } else {
          FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
          int numPoints = physicalPoints.dimension(1); // (C,P,D)
          out << "physicalPoints:\n" << physicalPoints;
          FieldContainer<double> actualValues(1,numPoints); // assumes scalar-valued
          FieldContainer<double> expectedValues(1,numPoints); // assumes scalar-valued
          f_actual->values(actualValues, basisCache);
          f_expected->values(expectedValues, basisCache);
          TEST_COMPARE_FLOATING_ARRAYS(expectedValues, actualValues, tol);
        }
      }
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
    CellTopoPtr quad_x_time = CellTopology::cellTopology(CellTopology::quad(), tensorialDegree);
    CellTopoPtr tri_x_time = CellTopology::cellTopology(CellTopology::triangle(), tensorialDegree);

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
    VarFactoryPtr varFactory = VarFactory::varFactory();
    VarPtr tau = varFactory->testVar("tau", HDIV);
    VarPtr v = varFactory->testVar("v", HGRAD);

    // define trial variables
    VarPtr uhat = varFactory->traceVar("uhat");
    VarPtr fhat = varFactory->fluxVar("fhat");
    VarPtr u = varFactory->fieldVar("u");
    VarPtr sigma = varFactory->fieldVar("sigma", VECTOR_L2);

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
    int H1Order = 3, pToAdd = 2;
    Teuchos::RCP<Mesh> spaceTimeMesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );

    SolutionPtr spaceTimeSolution = Teuchos::rcp( new Solution(spaceTimeMesh) );

    FunctionPtr n = Function::normalSpaceTime();
    FunctionPtr parity = Function::sideParity();
    FunctionPtr f_flux = Function::xn(2) * n->x() * parity + Function::yn(1) * n->y() * parity + Function::zn(1) * n->z() * parity;

    map<int, FunctionPtr > functionMap;
    functionMap[uhat->ID()] = Function::xn(1);
    functionMap[fhat->ID()] = f_flux;
    functionMap[u->ID()] = Function::xn(1);
    functionMap[sigma->ID()] = Function::vectorize(Function::xn(1), Function::yn(1));
    spaceTimeSolution->projectOntoMesh(functionMap);

    double tol = 1e-14;
    for (map<int, FunctionPtr >::iterator entryIt = functionMap.begin(); entryIt != functionMap.end(); entryIt++) {
      int trialID = entryIt->first;
      VarPtr trialVar = varFactory->trial(trialID);
      FunctionPtr f_expected = entryIt->second;
      FunctionPtr f_actual = Function::solution(trialVar, spaceTimeSolution);

      if (trialVar->varType() == FLUX) {
        // then Function::solution() will have included a parity weight, basically on the idea that we're also multiplying by normals
        // in our usage of the solution data.  (It may be that this is not the best way to do this.)

        // For this test, though, we want to reverse that:
        f_actual = parity * f_actual;
      }

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
    VarFactoryPtr varFactory = VarFactory::varFactory();
    VarPtr tau = varFactory->testVar("tau", HDIV);
    VarPtr v = varFactory->testVar("v", HGRAD);

    // define trial variables
    VarPtr uhat = varFactory->traceVar("uhat");
    VarPtr fhat = varFactory->fluxVar("fhat");
    VarPtr u = varFactory->fieldVar("u");
    VarPtr sigma = varFactory->fieldVar("sigma", VECTOR_L2);

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

    SolutionPtr spaceTimeSolution = Teuchos::rcp( new Solution(spaceTimeMesh) );

    FunctionPtr n = Function::normalSpaceTime();
    FunctionPtr parity = Function::sideParity();
    FunctionPtr f_flux = Function::xn(2) * n->x() * parity + Function::yn(1) * n->y() * parity + Function::zn(1) * n->z() * parity;

    map<int, FunctionPtr > functionMap;
    functionMap[uhat->ID()] = Function::xn(1);
    functionMap[fhat->ID()] = f_flux;
    functionMap[u->ID()] = Function::xn(1);
    functionMap[sigma->ID()] = Function::vectorize(Function::xn(1), Function::yn(1), Function::zn(1));
    spaceTimeSolution->projectOntoMesh(functionMap);

    double tol = 1e-14;
    for (map<int, FunctionPtr >::iterator entryIt = functionMap.begin(); entryIt != functionMap.end(); entryIt++) {
      int trialID = entryIt->first;
      VarPtr trialVar = varFactory->trial(trialID);
      FunctionPtr f_expected = entryIt->second;
      FunctionPtr f_actual = Function::solution(trialVar, spaceTimeSolution);

      if (trialVar->varType() == FLUX) {
        // then Function::solution() will have included a parity weight, basically on the idea that we're also multiplying by normals
        // in our usage of the solution data.  (It may be that this is not the best way to do this.)

        // For this test, though, we want to reverse that:
        f_actual = parity * f_actual;
      }

      double err_L2 = (f_actual - f_expected)->l2norm(spaceTimeMesh);
      TEST_COMPARE(err_L2, <, tol);
    }
  }

  void testSaveAndLoad2D(BFPtr bf, Teuchos::FancyOStream &out, bool &success)
  {
    int H1Order = 2;
    vector<double> dimensions = {1.0, 2.0}; // 1 x 2 domain
    vector<int> elementCounts = {3, 2}; // 3 x 2 mesh

    MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order);

    BCPtr bc = BC::bc();
    RHSPtr rhs = RHS::rhs();
    IPPtr ip = bf->graphNorm();
    SolutionPtr soln = Solution::solution(mesh,bc,rhs,ip);

    string filePrefix = "SavedSolution";
    soln->save(filePrefix);

    soln->load(bf, filePrefix);
    MeshPtr loadedMesh = soln->mesh();
    TEST_EQUALITY(loadedMesh->globalDofCount(), mesh->globalDofCount());

    // delete the files we created
    remove((filePrefix+".soln").c_str());
    remove((filePrefix+".mesh").c_str());

    // just to confirm that we can manipulate the loaded mesh:
    set<GlobalIndexType> cellsToRefine;
    cellsToRefine.insert(0);
    loadedMesh->pRefine(cellsToRefine);
  }

  TEUCHOS_UNIT_TEST( Solution, SaveAndLoadPoissonConforming )
  {
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    testSaveAndLoad2D(form.bf(), out, success);
  }

  TEUCHOS_UNIT_TEST( Solution, SaveAndLoadStokesConforming )
  {
    int spaceDim = 2;
    bool conformingTraces = true;
    StokesVGPFormulation form(spaceDim,conformingTraces);
    testSaveAndLoad2D(form.bf(), out, success);
  }
} // namespace
