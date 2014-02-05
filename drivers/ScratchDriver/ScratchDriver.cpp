#include "MeshTopology.h"

#include <iostream.h>

#include "Epetra_SerialComm.h"

#include "Epetra_Time.h"

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

MeshTopologyPtr makeHexMesh(double x0, double y0, double z0, double width, double height, double depth,
                                       unsigned horizontalCells, unsigned verticalCells, unsigned depthCells) {
  unsigned spaceDim = 3;
  Teuchos::RCP<MeshTopology> mesh = Teuchos::rcp( new MeshTopology(spaceDim) );
  double dx = width / horizontalCells;
  double dy = height / verticalCells;
  double dz = depth / depthCells;
  CellTopoPtr hexTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ) );
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
  Epetra_SerialComm Comm;
  
  int nx = 100, ny = 100, nz = 10;
  
  cout << "creating " << nx * ny * nz << "-element mesh...\n";
  
  Epetra_Time timer(Comm);
  
  MeshTopologyPtr meshTopo = makeHexMesh(0, 0, 0, 1, 1, 1, nx, ny, nz);
  
  double timeMeshCreation = timer.ElapsedTime();

  cout << "...created.  Elapsed time " << timeMeshCreation << " seconds; pausing now to allow memory usage examination.  Enter a number to continue.\n";
  int n;
  cin >> n;
  
  int numRefs = 6;
  cout << "Creating mesh for " << numRefs << " uniform refinements.\n";
  timer.ResetStartTime();
  meshTopo = makeHexMesh(0, 0, 0, 1, 1, 1, 1, 1, 1);
  
  for (int ref=0; ref<numRefs; ref++) {
    refineUniformly(meshTopo);
  }
  
  double timeMeshRefinements  = timer.ElapsedTime();
  
  cout << "Completed refinements in " << timeMeshRefinements << " seconds.  Final mesh has " << meshTopo->activeCellCount() << " active cells, and " << meshTopo->cellCount() << " cells total.\n";
  
  cout << "Paused to allow memory usage examination.  Enter a number to exit.\n";
  
  cin >> n;
  
  return 0;
}