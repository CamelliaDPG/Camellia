
#include "Teuchos_RCP.hpp"

#include "Mesh.h"
#include "MeshFactory.h"
#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Solution.h"
#include "HDF5Exporter.h"

#include "GlobalDofAssignment.h"

#include "Epetra_FEVector.h"

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

class EntireBoundary : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
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

  // {
  // // 1D tests
  //   CellTopoPtrLegacy line_2 = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Line<2> >() ) );

  // // let's draw a line
  //   vector<double> v0 = makeVertex(0);
  //   vector<double> v1 = makeVertex(1);
  //   vector<double> v2 = makeVertex(2);

  //   vector< vector<double> > vertices;
  //   vertices.push_back(v0);
  //   vertices.push_back(v1);
  //   vertices.push_back(v2);

  //   vector<unsigned> line1VertexList;
  //   vector<unsigned> line2VertexList;
  //   line1VertexList.push_back(0);
  //   line1VertexList.push_back(1);
  //   line2VertexList.push_back(1);
  //   line2VertexList.push_back(2);

  //   vector< vector<unsigned> > elementVertices;
  //   elementVertices.push_back(line1VertexList);
  //   elementVertices.push_back(line2VertexList);

  //   vector< CellTopoPtrLegacy > cellTopos;
  //   cellTopos.push_back(line_2);
  //   cellTopos.push_back(line_2);
  //   MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );

  //   MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry) );

  //   FunctionPtr x = Function::xn(1);
  //   FunctionPtr function = x;
  //   FunctionPtr fbdr = Function::restrictToCellBoundary(function);
  //   vector<FunctionPtr> functions;
  //   functions.push_back(function);
  //   functions.push_back(function);
  //   vector<string> functionNames;
  //   functionNames.push_back("function1");
  //   functionNames.push_back("function2");

  //   // {
  //   //     HDF5Exporter exporter(mesh, "function1", false);
  //   //     exporter.exportFunction(function, "function1");
  //   // }
  //   // {
  //   //     HDF5Exporter exporter(mesh, "boundary1", false);
  //   //     exporter.exportFunction(fbdr, "boundary1");
  //   // }
  //   // {
  //   //     HDF5Exporter exporter(mesh, "functions1", false);
  //   //     exporter.exportFunction(functions, functionNames);
  //   // }
  // }
  {
  // 2D tests
    CellTopoPtrLegacy quad_4 = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
    CellTopoPtrLegacy tri_3 = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() ) );

  // let's draw a little house
    vector<double> v0 = makeVertex(-1,0);
    vector<double> v1 = makeVertex(1,0);
    vector<double> v2 = makeVertex(1,2);
    vector<double> v3 = makeVertex(-1,2);
    vector<double> v4 = makeVertex(0.0,3);

    vector< vector<double> > vertices;
    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
    vertices.push_back(v3);
    vertices.push_back(v4);

    vector<unsigned> quadVertexList;
    quadVertexList.push_back(0);
    quadVertexList.push_back(1);
    quadVertexList.push_back(2);
    quadVertexList.push_back(3);

    vector<unsigned> triVertexList;
    triVertexList.push_back(3);
    triVertexList.push_back(2);
    triVertexList.push_back(4);

    vector< vector<unsigned> > elementVertices;
    elementVertices.push_back(quadVertexList);
    elementVertices.push_back(triVertexList);

    vector< CellTopoPtrLegacy > cellTopos;
    cellTopos.push_back(quad_4);
    cellTopos.push_back(tri_3);
    MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );

    MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry) );

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

    ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
    IPPtr ip = bf->graphNorm();

    ////////////////////   SPECIFY RHS   ///////////////////////
    RHSPtr rhs = RHS::rhs();
    FunctionPtr one = Function::constant(1.0);
    rhs->addTerm( one * v );

    ////////////////////   CREATE BCs   ///////////////////////
    BCPtr bc = BC::bc();
    FunctionPtr zero = Function::zero();
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new EntireBoundary );
    bc->addDirichlet(uhat, entireBoundary, zero);

    ////////////////////   SOLVE & REFINE   ///////////////////////

    // Output solution
    FieldContainer<GlobalIndexType> savedCellPartition;
    Teuchos::RCP<Epetra_FEVector> savedLHSVector;

    {
      ////////////////////   BUILD MESH   ///////////////////////
      int H1Order = 4, pToAdd = 2;
      Teuchos::RCP<Mesh> mesh = Teuchos::rcp( new Mesh (meshTopology, bf, H1Order, pToAdd) );

      Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
      solution->solve(false);
      RefinementStrategy refinementStrategy( solution, 0.2);
      HDF5Exporter exporter(mesh, "Poisson");
      exporter.exportSolution(solution, varFactory, 0, 2, cellIDToSubdivision(mesh, 4));
        // Teuchos::RCP< RefinementHistory > refHistory = Teuchos::rcp( new RefinementHistory );
        // mesh->_refinementHistory = Teuchos::rcp( new RefinementHistory );
        // mesh->registerObserver(refHistory);
        // refHistory->saveToFile("Test3.txt");
      int numRefs = 1;
      for (int ref = 1; ref <= numRefs; ref++)
      {
        refinementStrategy.refine(commRank==0);
        solution->solve(false);
        mesh->saveToHDF5("Test0.h5");
        solution->saveToHDF5("Test0_soln.h5");
        exporter.exportSolution(solution, varFactory, ref, 2, cellIDToSubdivision(mesh, 4));
      }
      mesh->globalDofAssignment()->getPartitions(savedCellPartition);
      savedLHSVector = solution->getLHSVector();
    }
    // {
    //   MeshPtr loadedMesh = MeshFactory::loadFromHDF5(bf, "Test0.h5");
    //   Teuchos::RCP<Solution> loadedSolution = Teuchos::rcp( new Solution(loadedMesh, bc, rhs, ip) );
    //   loadedSolution->solve(false);
    //   HDF5Exporter exporter(loadedMesh, "MeshLoaded");
    //   exporter.exportSolution(loadedSolution, varFactory, 0, 2, cellIDToSubdivision(loadedMesh, 4));
    // }
    {
      MeshPtr loadedMesh = MeshFactory::loadFromHDF5(bf, "Test0.h5");
      FieldContainer<GlobalIndexType> loadedCellPartition;
      loadedMesh->globalDofAssignment()->getPartitions(loadedCellPartition);
      if (loadedCellPartition.size() != savedCellPartition.size()) {
        cout << "Error: the loaded partition has different size/shape than the saved one.\n";
        cout << "loaded size: " << loadedCellPartition.size() << "; saved size: " << savedCellPartition.size() << endl;
      } else {
        bool partitionsMatch = true;
        for (int i=0; i<loadedCellPartition.size(); i++) {
          if (loadedCellPartition[i] != savedCellPartition[i]) {
            partitionsMatch = false;
            break;
          }
        }
        if (partitionsMatch) cout << "Saved and loaded cell partitions match!\n";
        else {
          cout << "Saved and loaded cell partitions differ.\n";
          cout << "saved:\n" << savedCellPartition;
          cout << "loaded:\n" << loadedCellPartition;
        }
      }
      Teuchos::RCP<Solution> loadedSolution = Teuchos::rcp( new Solution(loadedMesh, bc, rhs, ip) );
      // loadedSolution->solve(false);
      // loadedSolution->addSolution(loadedSolution, -1.0);
      loadedSolution->loadFromHDF5("Test0_soln.h5");
      
      Teuchos::RCP<Epetra_FEVector> loadedLHSVector = loadedSolution->getLHSVector();
      if (loadedLHSVector->Map().MinLID() != savedLHSVector->Map().MinLID()) {
        cout << "On rank " << commRank << ", loaded min LID = " << loadedLHSVector->Map().MinLID();
        cout << ", but saved min LID = " << savedLHSVector->Map().MinLID() << endl;
      } else if (loadedLHSVector->Map().MaxLID() != savedLHSVector->Map().MaxLID()) {
        cout << "On rank " << commRank << ", loaded max LID = " << loadedLHSVector->Map().MaxLID();
        cout << ", but saved max LID = " << savedLHSVector->Map().MaxLID() << endl;
      } else {
        bool globalIDsMatch = true;
        for (int lid = loadedLHSVector->Map().MinLID(); lid <= loadedLHSVector->Map().MaxLID(); lid++) {
          if (loadedLHSVector->Map().GID(lid) != savedLHSVector->Map().GID(lid)) {
            globalIDsMatch = false;
          }
        }
        if (! globalIDsMatch) {
          cout << "On rank " << commRank << ", loaded and saved solution vector maps differ in their global IDs.\n";
        } else {
          cout << "On rank " << commRank << ", loaded and saved solution vector maps match in their global IDs.\n";
        }
        
        bool entriesMatch = true;
        double tol = 1e-16;
        if (loadedLHSVector->Map().MinLID() != loadedLHSVector->Map().MaxLID())
        {
          for (int lid = loadedLHSVector->Map().MinLID(); lid <= loadedLHSVector->Map().MaxLID(); lid++) {
            double loadedValue = (*loadedLHSVector)[0][lid];
            double savedValue = (*savedLHSVector)[0][lid];
            double diff = abs( loadedValue - savedValue );
            if (diff > tol) {
              entriesMatch = false;
              cout << "On rank " << commRank << ", loaded and saved solution vectors differ in entry with lid " << lid;
              cout << "; loaded value = " << loadedValue << "; saved value = " << savedValue << ".\n";
            }
          }
          if (entriesMatch) {
            cout << "On rank " << commRank << ", loaded and saved solution vectors match!\n";
          } else {
            cout << "On rank " << commRank << ", loaded and saved solution vectors do not match.\n";
          }
        }
      }
      
      HDF5Exporter exporter(loadedMesh, "SolutionLoaded");
      exporter.exportSolution(loadedSolution, varFactory, 0, 2, cellIDToSubdivision(loadedMesh, 4));
    }
  }

  // {
  // // 3D tests
  //   CellTopoPtrLegacy hex = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ));

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

  //   vector< CellTopoPtrLegacy > cellTopos;
  //   cellTopos.push_back(hex);
  //   // cellTopos.push_back(tri_3);
  //   MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );

  //   MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry) );

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
