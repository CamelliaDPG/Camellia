#include "MeshTopology.h"

using namespace std;

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

vector< vector<double> > quadPoints(double x0, double y0, double width, double height) {
  vector< vector<double> > v(4);
  v[0] = makeVertex(x0,y0);
  v[1] = makeVertex(x0 + width,y0);
  v[2] = makeVertex(x0 + width,y0 + height);
  v[3] = makeVertex(x0,y0 + height);
  return v;
}

vector< vector<double> > hexPoints(double x0, double y0, double z0, double width, double height, double depth) {
  vector< vector<double> > v(8);
  v[0] = makeVertex(x0,y0,z0);
  v[1] = makeVertex(x0 + width,y0,z0);
  v[2] = makeVertex(x0 + width,y0 + height,z0);
  v[3] = makeVertex(x0,y0 + height,z0);
  v[4] = makeVertex(x0,y0,z0+depth);
  v[5] = makeVertex(x0 + width,y0,z0 + depth);
  v[6] = makeVertex(x0 + width,y0 + height,z0 + depth);
  v[7] = makeVertex(x0,y0 + height,z0 + depth);
  return v;
}

MeshTopologyPtr makeQuadMesh(double x0, double y0, double width, double height,
                            unsigned horizontalCells, unsigned verticalCells) {
  unsigned spaceDim = 2;
  Teuchos::RCP<MeshTopology> mesh = Teuchos::rcp( new MeshTopology(spaceDim) );
  double dx = width / horizontalCells;
  double dy = height / verticalCells;
  CellTopoPtrLegacy quadTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
  for (unsigned i=0; i<horizontalCells; i++) {
    double x = x0 + dx * i;
    for (unsigned j=0; j<verticalCells; j++) {
      double y = y0 + dy * j;
        vector< vector<double> > vertices = quadPoints(x, y, dx, dy);
        mesh->addCell(quadTopo, vertices);
    }
  }
  return mesh;
}

MeshTopologyPtr makeHexMesh(double x0, double y0, double z0, double width, double height, double depth,
                            unsigned horizontalCells, unsigned verticalCells, unsigned depthCells) {
  unsigned spaceDim = 3;
  Teuchos::RCP<MeshTopology> mesh = Teuchos::rcp( new MeshTopology(spaceDim) );
  double dx = width / horizontalCells;
  double dy = height / verticalCells;
  double dz = depth / depthCells;
  CellTopoPtrLegacy hexTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ) );
  for (unsigned i=0; i<horizontalCells; i++) {
    double x = x0 + dx * i;
    for (unsigned j=0; j<verticalCells; j++) {
      double y = y0 + dy * j;
      for (unsigned k=0; k<depthCells; k++) {
        double z = z0 + dz * k;
        vector< vector<double> > vertices = hexPoints(x, y, z, dx, dy, dz);
        mesh->addCell(hexTopo, vertices);
      }
    }
  }
  return mesh;
}

void refineUniformly(MeshTopologyPtr mesh) {
  set<unsigned> cellIndices = mesh->getActiveCellIndices();
  for (set<unsigned>::iterator cellIt = cellIndices.begin(); cellIt != cellIndices.end(); cellIt++) {
    mesh->refineCell(*cellIt, RefinementPattern::regularRefinementPatternHexahedron());
  }
}

int main(int argc, char *argv[]) {
  int horizontalCells = 128;
  int verticalCells = 128;
  MeshTopologyPtr mesh = makeQuadMesh(0,0,1,1,horizontalCells,verticalCells);
  
  cout << "Approximate size of " << horizontalCells << " x " << verticalCells << " mesh is ";
  
  long long memoryFootprintInBytes = mesh->approximateMemoryFootprint();
  double memoryFootprintInMegabytes = (double)memoryFootprintInBytes / (1024 * 1024);
  cout << setprecision(4) << memoryFootprintInMegabytes << " MB.\n";
  
  mesh->printApproximateMemoryReport();
  
  CellPtr cell = mesh->getCell(0);
  cell->printApproximateMemoryReport();
  
  mesh = Teuchos::null;
  cell = Teuchos::null;
  
  cout << "Hello!\n";
}