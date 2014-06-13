
#include "Teuchos_RCP.hpp"

#include "Mesh.h"
#include "SolutionExporter.h"

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

int main(int argc, char *argv[])
{
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
#endif

  {
  // 2D tests
    CellTopoPtr line_2 = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Line<2> >() ) );

  // let's draw a little house
    vector<double> v0 = makeVertex(0);
    vector<double> v1 = makeVertex(1);
    vector<double> v2 = makeVertex(2);

    vector< vector<double> > vertices;
    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);

    vector<unsigned> line1VertexList;
    vector<unsigned> line2VertexList;
    line1VertexList.push_back(0);
    line1VertexList.push_back(1);
    line2VertexList.push_back(1);
    line2VertexList.push_back(2);

    vector< vector<unsigned> > elementVertices;
    elementVertices.push_back(line1VertexList);
    elementVertices.push_back(line2VertexList);

    vector< CellTopoPtr > cellTopos;
    cellTopos.push_back(line_2);
    cellTopos.push_back(line_2);
    MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );

    MeshTopologyPtr mesh = Teuchos::rcp( new MeshTopology(meshGeometry) );

    FunctionPtr x = Function::xn(1);
    FunctionPtr function = x;
    // FunctionPtr fbdr = Function::restrictToCellBoundary(function);

    XDMFExporter exporter(mesh);
    exporter.exportFunction(function, "function1");
    // exporter.exportFunction(fbdr, "boundary1");
  }
  {
  // 2D tests
    CellTopoPtr quad_4 = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
    CellTopoPtr tri_3 = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() ) );

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
    triVertexList.push_back(2);
    triVertexList.push_back(3);
    triVertexList.push_back(4);

    vector< vector<unsigned> > elementVertices;
    elementVertices.push_back(quadVertexList);
    elementVertices.push_back(triVertexList);

    vector< CellTopoPtr > cellTopos;
    cellTopos.push_back(quad_4);
    cellTopos.push_back(tri_3);
    MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );

    MeshTopologyPtr mesh = Teuchos::rcp( new MeshTopology(meshGeometry) );

    FunctionPtr x2 = Function::xn(2);
    FunctionPtr y2 = Function::yn(2);
    FunctionPtr function = x2 + y2;
    FunctionPtr vect = Function::vectorize(x2, y2);
    FunctionPtr fbdr = Function::restrictToCellBoundary(function);

    XDMFExporter exporter(mesh);
    exporter.exportFunction(function, "function2");
    exporter.exportFunction(vect, "vect2");
    exporter.exportFunction(fbdr, "boundary2");
  }

  {
  // 3D tests
    CellTopoPtr hex = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ));

  // let's draw a little box
    vector<double> v0 = makeVertex(0,0,0);
    vector<double> v1 = makeVertex(1,0,0);
    vector<double> v2 = makeVertex(1,1,0);
    vector<double> v3 = makeVertex(0,1,0);
    vector<double> v4 = makeVertex(0,0,1);
    vector<double> v5 = makeVertex(1,0,1);
    vector<double> v6 = makeVertex(1,1,1);
    vector<double> v7 = makeVertex(0,1,1);

    vector< vector<double> > vertices;
    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
    vertices.push_back(v3);
    vertices.push_back(v4);
    vertices.push_back(v5);
    vertices.push_back(v6);
    vertices.push_back(v7);

    vector<unsigned> hexVertexList;
    hexVertexList.push_back(0);
    hexVertexList.push_back(1);
    hexVertexList.push_back(2);
    hexVertexList.push_back(3);
    hexVertexList.push_back(4);
    hexVertexList.push_back(5);
    hexVertexList.push_back(6);
    hexVertexList.push_back(7);

    // vector<unsigned> triVertexList;
    // triVertexList.push_back(2);
    // triVertexList.push_back(3);
    // triVertexList.push_back(4);

    vector< vector<unsigned> > elementVertices;
    elementVertices.push_back(hexVertexList);
    // elementVertices.push_back(triVertexList);

    vector< CellTopoPtr > cellTopos;
    cellTopos.push_back(hex);
    // cellTopos.push_back(tri_3);
    MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );

    MeshTopologyPtr mesh = Teuchos::rcp( new MeshTopology(meshGeometry) );

    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    FunctionPtr function = x + y + z;
    FunctionPtr fbdr = Function::restrictToCellBoundary(function);
    FunctionPtr vect = Function::vectorize(x, y, z);

    XDMFExporter exporter(mesh);
    exporter.exportFunction(function, "function3");
    exporter.exportFunction(fbdr, "boundary3");
    exporter.exportFunction(vect, "vect3");
  }
}
