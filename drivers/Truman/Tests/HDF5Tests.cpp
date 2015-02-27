
#include "Teuchos_RCP.hpp"

#include "MeshFactory.h"
#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Solution.h"
#include "MeshTools.h"
#include "HDF5Exporter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#endif

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

class EntireBoundary1D : public SpatialFilter {
  public:
    bool matchesPoint(double x) {
        return true;
    }
};

class EntireBoundary2D : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
        return true;
    }
};

class EntireBoundary3D : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y, double z) {
        return true;
    }
};

class EntireBoundary4D : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y, double z, double t) {
        return true;
    }
};

int main(int argc, char *argv[])
{
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
#endif
  int commRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

  {
    // 1D tests
    CellTopoPtr line_2 = CellTopology::line();
    // Space-time
    int tensorialDegree = 1;
    CellTopoPtr line_x_time = CellTopology::cellTopology(shards::getCellTopologyData<shards::Line<2> >(), tensorialDegree);

    // let's draw a line
    vector<double> v0 = makeVertex(0);
    vector<double> v1 = makeVertex(1);
    vector<double> v2 = makeVertex(2);
    // Space-time
    vector<double> v00 = makeVertex(0,0);
    vector<double> v10 = makeVertex(1,0);
    vector<double> v20 = makeVertex(2,0);
    vector<double> v01 = makeVertex(0,1);
    vector<double> v11 = makeVertex(1,1);
    vector<double> v21 = makeVertex(2,1);

    vector< vector<double> > vertices;
    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
    // Space-time
    vector< vector<double> > spaceTimeVertices;
    spaceTimeVertices.push_back(v00);
    spaceTimeVertices.push_back(v10);
    spaceTimeVertices.push_back(v20);
    spaceTimeVertices.push_back(v01);
    spaceTimeVertices.push_back(v11);
    spaceTimeVertices.push_back(v21);

    vector<unsigned> line1VertexList;
    vector<unsigned> line2VertexList;
    line1VertexList.push_back(0);
    line1VertexList.push_back(1);
    line2VertexList.push_back(1);
    line2VertexList.push_back(2);
    // Space-time
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

    vector< vector<unsigned> > elementVertices;
    elementVertices.push_back(line1VertexList);
    elementVertices.push_back(line2VertexList);
    // Space-time
    vector< vector<unsigned> > spaceTimeElementVertices;
    spaceTimeElementVertices.push_back(spaceTimeLine1VertexList);
    spaceTimeElementVertices.push_back(spaceTimeLine2VertexList);

    vector< CellTopoPtr > cellTopos;
    cellTopos.push_back(line_2);
    cellTopos.push_back(line_2);
    // Space-time
    vector< CellTopoPtr > spaceTimeCellTopos;
    spaceTimeCellTopos.push_back(line_x_time);
    spaceTimeCellTopos.push_back(line_x_time);

    MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );
    // Space-time
    MeshGeometryPtr spaceTimeMeshGeometry = Teuchos::rcp( new MeshGeometry(spaceTimeVertices, spaceTimeElementVertices, spaceTimeCellTopos) );

    MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry) );
    // Space-time
    MeshTopologyPtr spaceTimeMeshTopology = Teuchos::rcp( new MeshTopology(spaceTimeMeshGeometry) );

    // FunctionPtr x = Function::xn(1);
    // FunctionPtr function = x;
    // FunctionPtr fbdr = Function::restrictToCellBoundary(function);
    // vector<FunctionPtr> functions;
    // functions.push_back(function);
    // functions.push_back(function);
    // vector<string> functionNames;
    // functionNames.push_back("function1");
    // functionNames.push_back("function2");

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
    Teuchos::RCP<Mesh> mesh = Teuchos::rcp( new Mesh (meshTopology, bf, H1Order, pToAdd) );
    // Space-time
    Teuchos::RCP<Mesh> spaceTimeMesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );

    ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
    IPPtr ip = bf->graphNorm();

    ////////////////////   SPECIFY RHS   ///////////////////////
    RHSPtr rhs = RHS::rhs();
    FunctionPtr one = Function::constant(1.0);
    rhs->addTerm( one * v );

    ////////////////////   CREATE BCs   ///////////////////////
    BCPtr bc = BC::bc();
    FunctionPtr zero = Function::zero();
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new EntireBoundary1D );
    bc->addDirichlet(uhat, entireBoundary, zero);

    ////////////////////   SOLVE & REFINE   ///////////////////////
    // Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
    // Teuchos::RCP<Solution> spaceTimeSolution = Teuchos::rcp( new Solution(spaceTimeMesh, bc, rhs, ip) );
    Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh) );
    Teuchos::RCP<Solution> spaceTimeSolution = Teuchos::rcp( new Solution(spaceTimeMesh) );
    // solution->solve(false);
    // RefinementStrategy refinementStrategy( solution, 0.2);

    map<int, Teuchos::RCP<Function> > functionMap;
    functionMap[0] = Function::xn(1);
    functionMap[1] = Function::xn(1);
    functionMap[2] = Function::xn(1);
    functionMap[3] = Function::xn(1);
    solution->projectOntoMesh(functionMap);
    // spaceTimeSolution->projectOntoMesh(functionMap);

    // HDF5Exporter exporter(mesh, "Poisson1D");
    // exporter.exportSolution(solution, varFactory, 0, 2);

    // {
    //   HDF5Exporter exporter(spaceTimeMesh, "SliceAnimation");
    //   vector<FunctionPtr> fcns;
    //   fcns.push_back(Function::xn(1));
    //   fcns.push_back(Function::xn(2));
    //   vector<string> fcnnames;
    //   fcnnames.push_back("x");
    //   fcnnames.push_back("x2");
    //   exporter.exportTimeSlab(fcns, fcnnames, 0, 1, 4);
    //   vector<FunctionPtr> bfcns;
    //   bfcns.push_back(Function::restrictToCellBoundary(Function::constant(0)));
    //   vector<string> bfcnnames;
    //   bfcnnames.push_back("mesh");
    //   exporter.exportTimeSlab(bfcns, bfcnnames, 0, 1, 4);
    // }
    {
      cout << "FIELD" << endl;
      HDF5Exporter exporter(spaceTimeMesh, "SpaceTime1D");
      vector<FunctionPtr> fcns;
      fcns.push_back(Function::xn(1));
      fcns.push_back(Function::xn(2));
      vector<string> fcnnames;
      fcnnames.push_back("x");
      fcnnames.push_back("x2");
      exporter.exportFunction(fcns, fcnnames, 0, 2);
      cout << "TRACE" << endl;
      vector<FunctionPtr> bfcns;
      bfcns.push_back(Function::restrictToCellBoundary(Function::constant(0)));
      vector<string> bfcnnames;
      bfcnnames.push_back("mesh");
      exporter.exportFunction(bfcns, bfcnnames, 0, 3);
    }

    // map<GlobalIndexType,GlobalIndexType> cellMap_t0, cellMap_t1;
    // MeshPtr meshSlice_t0 = MeshTools::timeSliceMesh(spaceTimeMesh, 0, cellMap_t0, H1Order);
    // MeshPtr meshSlice_t1 = MeshTools::timeSliceMesh(spaceTimeMesh, 1,  cellMap_t1, H1Order);
    // FunctionPtr sliceFunction_t0 = MeshTools::timeSliceFunction(spaceTimeMesh, cellMap_t0, Function::xn(1), 0);
    // FunctionPtr sliceFunction_t1 = MeshTools::timeSliceFunction(mesh, cellMap_t1, Function::xn(1), 1);
    // HDF5Exporter exporter(spaceTimeMesh, "Function1D");
    // HDF5Exporter exporter0(meshSlice_t0, "Function1D_t0");
    // HDF5Exporter exporter1(meshSlice_t1, "Function1D_t1");
    // exporter0.exportFunction(sliceFunction_t0, "x");
    // exporter0.exportFunction(Function::xn(1), "x");
    // exporter0.exportFunction(Function::restrictToCellBoundary(Function::constant(0)), "mesh");
    // exporter1.exportFunction(sliceFunction_t1, "x");
    // exporter1.exportFunction(Function::restrictToCellBoundary(Function::constant(0)), "mesh");

    // {
    //     HDF5Exporter exporter(mesh, "function1", false);
    //     exporter.exportFunction(function, "function1");
    // }
    // {
    //     HDF5Exporter exporter(mesh, "boundary1", false);
    //     exporter.exportFunction(fbdr, "boundary1");
    // }
    // {
    //     HDF5Exporter exporter(mesh, "functions1", false);
    //     exporter.exportFunction(functions, functionNames);
    // }
  }
  {
    // 2D tests
    CellTopoPtr quad_4 = CellTopology::quad();
    CellTopoPtr tri_3 = CellTopology::triangle();
    // Space-time
    int tensorialDegree = 1;
    CellTopoPtr quad_x_time = CellTopology::cellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >(), tensorialDegree);
    CellTopoPtr tri_x_time = CellTopology::cellTopology(shards::getCellTopologyData<shards::Triangle<3> >(), tensorialDegree);

    // let's draw a little house
    vector<double> v0 = makeVertex(-1,0);
    vector<double> v1 = makeVertex(1,0);
    vector<double> v2 = makeVertex(1,2);
    vector<double> v3 = makeVertex(-1,2);
    vector<double> v4 = makeVertex(0,3);
    // Space-time
    vector<double> v00 = makeVertex(-1,0,0);
    vector<double> v10 = makeVertex(1, 0,0);
    vector<double> v20 = makeVertex(1, 2,0);
    vector<double> v30 = makeVertex(-1,2,0);
    vector<double> v40 = makeVertex(0, 3,0);
    vector<double> v01 = makeVertex(-1,0,1);
    vector<double> v11 = makeVertex(1, 0,1);
    vector<double> v21 = makeVertex(1, 2,1);
    vector<double> v31 = makeVertex(-1,2,1);
    vector<double> v41 = makeVertex(0, 3,1);

    vector< vector<double> > vertices;
    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
    vertices.push_back(v3);
    vertices.push_back(v4);
    // Space-time
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

    vector<unsigned> quadVertexList;
    quadVertexList.push_back(0);
    quadVertexList.push_back(1);
    quadVertexList.push_back(2);
    quadVertexList.push_back(3);
    // Space-time
    vector<unsigned> spaceTimeQuadVertexList;
    spaceTimeQuadVertexList.push_back(0);
    spaceTimeQuadVertexList.push_back(1);
    spaceTimeQuadVertexList.push_back(2);
    spaceTimeQuadVertexList.push_back(3);
    spaceTimeQuadVertexList.push_back(5);
    spaceTimeQuadVertexList.push_back(6);
    spaceTimeQuadVertexList.push_back(7);
    spaceTimeQuadVertexList.push_back(8);

    vector<unsigned> triVertexList;
    triVertexList.push_back(3);
    triVertexList.push_back(2);
    triVertexList.push_back(4);
    // Space-time
    vector<unsigned> spaceTimeTriVertexList;
    spaceTimeTriVertexList.push_back(3);
    spaceTimeTriVertexList.push_back(2);
    spaceTimeTriVertexList.push_back(4);
    spaceTimeTriVertexList.push_back(8);
    spaceTimeTriVertexList.push_back(7);
    spaceTimeTriVertexList.push_back(9);

    vector< vector<unsigned> > elementVertices;
    elementVertices.push_back(quadVertexList);
    elementVertices.push_back(triVertexList);
    // Space-time
    vector< vector<unsigned> > spaceTimeElementVertices;
    spaceTimeElementVertices.push_back(spaceTimeQuadVertexList);
    spaceTimeElementVertices.push_back(spaceTimeTriVertexList);

    vector< CellTopoPtr > cellTopos;
    cellTopos.push_back(quad_4);
    cellTopos.push_back(tri_3);
    // Space-time
    vector< CellTopoPtr > spaceTimeCellTopos;
    spaceTimeCellTopos.push_back(quad_x_time);
    spaceTimeCellTopos.push_back(tri_x_time);

    MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );
    // Space-time
    MeshGeometryPtr spaceTimeMeshGeometry = Teuchos::rcp( new MeshGeometry(spaceTimeVertices, spaceTimeElementVertices, spaceTimeCellTopos) );

    MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry) );
    // Space-time
    MeshTopologyPtr spaceTimeMeshTopology = Teuchos::rcp( new MeshTopology(spaceTimeMeshGeometry) );

  //   FunctionPtr x2 = Function::xn(2);
  //   FunctionPtr y2 = Function::yn(2);
  //   FunctionPtr function = x2 + y2;
  //   FunctionPtr vect = Function::vectorize(x2, y2);
  //   FunctionPtr fbdr = Function::restrictToCellBoundary(function);
  //   vector<FunctionPtr> functions;
  //   functions.push_back(function);
  //   functions.push_back(vect);
  //   vector<string> functionNames;
  //   functionNames.push_back("function");
  //   functionNames.push_back("vect");
  //   vector<FunctionPtr> bdrfunctions;
  //   bdrfunctions.push_back(fbdr);
  //   bdrfunctions.push_back(fbdr);
  //   vector<string> bdrfunctionNames;
  //   bdrfunctionNames.push_back("bdr1");
  //   bdrfunctionNames.push_back("bdr2");

  //   map<int, int> cellIDToNum1DPts;
  //   cellIDToNum1DPts[1] = 4;

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
    Teuchos::RCP<Mesh> mesh = Teuchos::rcp( new Mesh (meshTopology, bf, H1Order, pToAdd) );
    // Space-time
    Teuchos::RCP<Mesh> spaceTimeMesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopology, bf, H1Order, pToAdd) );

  //   ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  //   IPPtr ip = bf->graphNorm();

  //   ////////////////////   SPECIFY RHS   ///////////////////////
  //   RHSPtr rhs = RHS::rhs();
  //   // Teuchos::RCP<RHS> rhs = Teuchos::rcp( new RHS );
  //   FunctionPtr one = Function::constant(1.0);
  //   rhs->addTerm( one * v );

  //   ////////////////////   CREATE BCs   ///////////////////////
  //   // Teuchos::RCP<BC> bc = Teuchos::rcp( new BCEasy );
  //   BCPtr bc = BC::bc();
  //   FunctionPtr zero = Function::zero();
  //   SpatialFilterPtr entireBoundary = Teuchos::rcp( new EntireBoundary );
  //   bc->addDirichlet(uhat, entireBoundary, zero);

  //   ////////////////////   SOLVE & REFINE   ///////////////////////
  //   Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  //   solution->solve(false);
  //   RefinementStrategy refinementStrategy( solution, 0.2);

  //   // Output solution
  //   FunctionPtr uSoln = Function::solution(u, solution);
  //   FunctionPtr sigmaSoln = Function::solution(sigma, solution);
  //   FunctionPtr uhatSoln = Function::solution(uhat, solution);
  //   FunctionPtr fhatSoln = Function::solution(fhat, solution);
  //   vector<FunctionPtr> solnFunctions;
  //   solnFunctions.push_back(uSoln);
  //   solnFunctions.push_back(sigmaSoln);
  //   vector<string> solnNames;
  //   solnNames.push_back("u");
  //   solnNames.push_back("sigma");
  //   // set<GlobalIndexType> cellIndices;
  //   // cellIndices.insert(0);
  //   {
  //       // HDF5Exporter exporter(mesh, "HDF5Poisson", false);
  //       // exporter.exportFunction(solnFunctions, solnNames, 0, 4);
  //       // exporter.exportFunction(uSoln, "u", 0, 4);
  //       // exporter.exportFunction(uSoln, "u", 1, 5);
  //       // exporter.exportFunction(uhatSoln, "uhat", 0, 4);
  //       // exporter.exportFunction(uhatSoln, "uhat", 1, 5);
  //       // exporter.exportFunction(fhatSoln, "fhat", 0, 4);
  //       // exporter.exportFunction(fhatSoln, "fhat", 1, 5);
  //   }
  //   {
  //       // HDF5Exporter exporter(mesh, "Poisson");
  //       // exporter.exportSolution(solution, varFactory, 0, 2, cellIDToSubdivision(mesh, 4));
  //       // int numRefs = 1;
  //       // for (int ref = 1; ref <= numRefs; ref++)
  //       // {
  //       //     refinementStrategy.refine(commRank==0);
  //       //     solution->solve(false);
  //       //     exporter.exportSolution(solution, varFactory, ref, 2, cellIDToSubdivision(mesh, 4));
  //       // }
  //   }
  //   // exporter.exportFunction(sigmaSoln, "Poisson-s", "sigma", 0, 5);
  //   // exporter.exportFunction(uhatSoln, "Poisson-uhat", "uhat", 1, 6);

  //   {
  //       HDF5Exporter exporter(mesh, "Grid2D");
  //       // exporter.exportFunction(function, "function2", 0, 10);
  //       exporter.exportFunction(vect, "vect2", 1, 10, cellIDToNum1DPts);
  //       exporter.exportFunction(fbdr, "boundary2", 0);
  //       exporter.exportFunction(functions, functionNames, 1, 10);
  //   }
  //   {
  //       HDF5Exporter exporter(mesh, "BdrGrid2D");
  //       exporter.exportFunction(function, "function2", 0, 10);
  //       exporter.exportFunction(vect, "vect2", 1, 10, cellIDToNum1DPts);
  //       exporter.exportFunction(fbdr, "boundary2", 0);
  //       exporter.exportFunction(bdrfunctions, bdrfunctionNames, 1, 10);
  //   }
    {
      cout << "FIELD" << endl;
      HDF5Exporter exporter(spaceTimeMesh, "SpaceTime2D");
      vector<FunctionPtr> fcns;
      fcns.push_back(Function::xn(1));
      fcns.push_back(Function::xn(2));
      vector<string> fcnnames;
      fcnnames.push_back("x");
      fcnnames.push_back("x2");
      exporter.exportFunction(fcns, fcnnames, 0, 6);
      cout << "TRACE" << endl;
      vector<FunctionPtr> bfcns;
      bfcns.push_back(Function::restrictToCellBoundary(Function::constant(0)));
      vector<string> bfcnnames;
      bfcnnames.push_back("mesh");
      exporter.exportFunction(bfcns, bfcnnames, 0, 6);
    }

  }

  // {
  // // 3D tests
  //   CellTopoPtr hex = CellTopology::hexahedron();

  // // let's draw a little box
  //   vector<double> v0 = makeVertex(0,0,0);
  //   vector<double> v1 = makeVertex(1,0,0);
  //   vector<double> v2 = makeVertex(1,1,0);
  //   vector<double> v3 = makeVertex(0,1,0);
  //   vector<double> v4 = makeVertex(0,0,1);
  //   vector<double> v5 = makeVertex(1,0,1);
  //   vector<double> v6 = makeVertex(1,1,1);
  //   vector<double> v7 = makeVertex(0,1,1);

  //   vector< vector<double> > vertices;
  //   vertices.push_back(v0);
  //   vertices.push_back(v1);
  //   vertices.push_back(v2);
  //   vertices.push_back(v3);
  //   vertices.push_back(v4);
  //   vertices.push_back(v5);
  //   vertices.push_back(v6);
  //   vertices.push_back(v7);

  //   vector<unsigned> hexVertexList;
  //   hexVertexList.push_back(0);
  //   hexVertexList.push_back(1);
  //   hexVertexList.push_back(2);
  //   hexVertexList.push_back(3);
  //   hexVertexList.push_back(4);
  //   hexVertexList.push_back(5);
  //   hexVertexList.push_back(6);
  //   hexVertexList.push_back(7);

  //   // vector<unsigned> triVertexList;
  //   // triVertexList.push_back(2);
  //   // triVertexList.push_back(3);
  //   // triVertexList.push_back(4);

  //   vector< vector<unsigned> > elementVertices;
  //   elementVertices.push_back(hexVertexList);
  //   // elementVertices.push_back(triVertexList);

  //   vector< CellTopoPtr > cellTopos;
  //   cellTopos.push_back(hex);
  //   // cellTopos.push_back(tri_3);
  //   MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );

  //   MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry) );

  //   ////////////////////   DECLARE VARIABLES   ///////////////////////
  //   // define test variables
  //   VarFactory varFactory;
  //   VarPtr tau = varFactory.testVar("tau", HDIV);
  //   VarPtr v = varFactory.testVar("v", HGRAD);

  //   // define trial variables
  //   VarPtr uhat = varFactory.traceVar("uhat");
  //   VarPtr fhat = varFactory.fluxVar("fhat");
  //   VarPtr u = varFactory.fieldVar("u");
  //   VarPtr sigma = varFactory.fieldVar("sigma", VECTOR_L2);

  //   ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  //   BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  //   // tau terms:
  //   bf->addTerm(sigma, tau);
  //   bf->addTerm(u, tau->div());
  //   bf->addTerm(-uhat, tau->dot_normal());

  //   // v terms:
  //   bf->addTerm( sigma, v->grad() );
  //   bf->addTerm( fhat, v);

  //   ////////////////////   BUILD MESH   ///////////////////////
  //   int H1Order = 2, pToAdd = 2;
  //   Teuchos::RCP<Mesh> mesh = Teuchos::rcp( new Mesh (meshTopology, bf, H1Order, pToAdd) );
  //   set<GlobalIndexType> cellIDs;
  //   cellIDs.insert(0);
  //   mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternHexahedron());

  //   ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  //   IPPtr ip = bf->graphNorm();

  //   ////////////////////   SPECIFY RHS   ///////////////////////
  //   RHSPtr rhs = RHS::rhs();
  //   // Teuchos::RCP<RHS> rhs = Teuchos::rcp( new RHS );
  //   FunctionPtr one = Function::constant(1.0);
  //   rhs->addTerm( one * v );

  //   ////////////////////   CREATE BCs   ///////////////////////
  //   // Teuchos::RCP<BC> bc = Teuchos::rcp( new BCEasy );
  //   BCPtr bc = BC::bc();
  //   FunctionPtr zero = Function::zero();
  //   SpatialFilterPtr entireBoundary = Teuchos::rcp( new EntireBoundary3D );
  //   bc->addDirichlet(uhat, entireBoundary, Function::xn(1));

  //   ////////////////////   SOLVE & REFINE   ///////////////////////
  //   Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  //   solution->solve(false);

  //   {
  //       HDF5Exporter exporter(mesh, "Poisson3D");
  //       exporter.exportSolution(solution, varFactory, 0, 4, cellIDToSubdivision(mesh, 4));
  //   }


  //   FunctionPtr x = Function::xn(1);
  //   FunctionPtr y = Function::yn(1);
  //   FunctionPtr z = Function::zn(1);
  //   FunctionPtr function = x + y + z;
  //   FunctionPtr fbdr = Function::restrictToCellBoundary(function);
  //   FunctionPtr vect = Function::vectorize(x, y, z);
  //   vector<FunctionPtr> functions;
  //   functions.push_back(function);
  //   functions.push_back(vect);
  //   vector<string> functionNames;
  //   functionNames.push_back("function");
  //   functionNames.push_back("vect");

  //   // {
  //   //     HDF5Exporter exporter(mesh, "function3", false);
  //   //     exporter.exportFunction(function, "function3");
  //   // }
  //   // {
  //   //     HDF5Exporter exporter(mesh, "boundary3", false);
  //   //     exporter.exportFunction(fbdr, "boundary3");
  //   // }
  //   // {
  //   //     HDF5Exporter exporter(mesh, "vect3", false);
  //   //     exporter.exportFunction(vect, "vect3");
  //   // }
  //   // {
  //   //     HDF5Exporter exporter(mesh, "functions3", false);
  //   //     exporter.exportFunction(functions, functionNames);
  //   // }
  // }
}
