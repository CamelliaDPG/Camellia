#include "NewMeshTests.h"

#include "CamelliaCellTools.h"

NewMeshTests::NewMeshTests() {

}

void NewMeshTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (test1DMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (test2DMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (test3DMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

}
void NewMeshTests::setup() {
  
}

void NewMeshTests::teardown() {
  
}

bool NewMeshTests::test1DMesh() {
  bool success = true;
  
  CellTopoPtr line_2 = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Line<2> >() ) );
  RefinementPatternPtr lineRefPattern = RefinementPattern::regularRefinementPatternLine();
  
  vector<double> v0(1,0.0), v1(1,1.0), v2(1,3.0);
  vector< vector<double> > vertices;
  vertices.push_back(v0);
  vertices.push_back(v1);
  vertices.push_back(v2);
  
  vector< unsigned > elemVertexList(2);
  elemVertexList[0] = 0;
  elemVertexList[1] = 1;
  vector< vector< unsigned > > elementVertices;
  elementVertices.push_back(elemVertexList);
  
  elemVertexList[0] = 1;
  elemVertexList[1] = 2;
  elementVertices.push_back(elemVertexList);
  
  vector< CellTopoPtr > cellTopos(2,line_2);
  
  NewMeshGeometryPtr meshGeometry = Teuchos::rcp( new NewMeshGeometry(vertices, elementVertices, cellTopos) );
  
  NewMesh mesh(meshGeometry);
  
  if (mesh.cellCount() != 2) {
    success = false;
    cout << "After initialization, mesh doesn't have the expected number of cells.\n";
  }
  
  if (mesh.activeCellCount() != 2) {
    success = false;
    cout << "After initialization, mesh doesn't have the expected number of active cells.\n";
  }
  
  mesh.refineCell(0, lineRefPattern);
  if (mesh.cellCount() != 4) {
    success = false;
    cout << "After refinement, mesh doesn't have the expected number of cells.\n";
  }
  
  if (mesh.activeCellCount() != 3) {
    success = false;
    cout << "After refinement, mesh doesn't have the expected number of active cells.\n";
  }
  
  mesh.refineCell(2, lineRefPattern);
  if (mesh.cellCount() != 6) {
    success = false;
    cout << "After 2nd refinement, mesh doesn't have the expected number of cells.\n";
  }
  
  if (mesh.activeCellCount() != 4) {
    success = false;
    cout << "After 2nd refinement, mesh doesn't have the expected number of active cells.\n";
  }
  
  return success;
}

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

bool NewMeshTests::test2DMesh() {
  bool success = true;

  CellTopoPtr quad_4 = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
  CellTopoPtr tri_3 = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() ) );
  RefinementPatternPtr quadRefPattern = RefinementPattern::regularRefinementPatternQuad();
  RefinementPatternPtr triangleRefPattern = RefinementPattern::regularRefinementPatternTriangle();
  
  // let's draw a little house
  vector<double> v0 = makeVertex(0,0);
  vector<double> v1 = makeVertex(1,0);
  vector<double> v2 = makeVertex(1,1);
  vector<double> v3 = makeVertex(0,1);
  vector<double> v4 = makeVertex(0.5,1.5);
  
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
  NewMeshGeometryPtr meshGeometry = Teuchos::rcp( new NewMeshGeometry(vertices, elementVertices, cellTopos) );

  NewMesh mesh(meshGeometry);
  
  if (mesh.cellCount() != 2) {
    success = false;
    cout << "After initialization, mesh doesn't have the expected number of cells.\n";
  }
  
  if (mesh.activeCellCount() != 2) {
    success = false;
    cout << "After initialization, mesh doesn't have the expected number of active cells.\n";
  }

  mesh.refineCell(0, quadRefPattern);
  
  if (mesh.cellCount() != 6) {
    success = false;
    cout << "After quad refinement, mesh doesn't have the expected number of cells.\n";
  }
  
  if (mesh.activeCellCount() != 5) {
    success = false;
    cout << "After quad refinement, mesh doesn't have the expected number of active cells.\n";
  }
  
  mesh.refineCell(1, triangleRefPattern);

  if (mesh.cellCount() != 10) {
    success = false;
    cout << "After triangle refinement, mesh doesn't have the expected number of cells.\n";
  }
  
  if (mesh.activeCellCount() != 8) {
    success = false;
    cout << "After triangle refinement, mesh doesn't have the expected number of active cells.\n";
  }
  
  // TODO: test more than just the count.  Test vertex locations, say.  Test constraint counts (should be 0 here after both initial cells refined, 1 constrained edge after the first refinement).
  
  return success;
}

bool NewMeshTests::test3DMesh() {
  bool success = true;
  
  unsigned spaceDim = 3;
  unsigned hexNodeCount = 1 << spaceDim;
  FieldContainer<double> refHexPoints(hexNodeCount,spaceDim);
  Teuchos::RCP< shards::CellTopology > hexTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ));
  CamelliaCellTools::refCellNodesForTopology(refHexPoints, *hexTopo);
 
  vector< vector<double> > vertices;
  vector< unsigned > hexVertexIndices(hexNodeCount);
  vector< vector<unsigned> > elementVertices;
  for (unsigned nodeIndex=0; nodeIndex<hexNodeCount; nodeIndex++) {
    vector<double> vertex(spaceDim);
    for (unsigned d=0; d<spaceDim; d++) {
      vertex[d] = refHexPoints(nodeIndex,d);
    }
    hexVertexIndices[nodeIndex] = vertices.size();
    vertices.push_back(vertex);
  }
  elementVertices.push_back(hexVertexIndices);
  
  
//  // note that the following will create some repeated vertices, and that's not OK -- we'll need to do something different
//  vector<double> eastOffset = makeVertex(2,0,0);
//  for (unsigned nodeIndex=0; nodeIndex<hexNodeCount; nodeIndex++) {
//    vector<double> vertex(spaceDim);
//    for (unsigned d=0; d<spaceDim; d++) {
//      vertex[d] = refHexPoints(nodeIndex,d) + eastOffset[d];
//    }
//    hexVertexIndices[nodeIndex] = vertices.size();
//    vertices.push_back(vertex);
//  }
//  elementVertices.push_back(hexVertexIndices);
 
  vector< CellTopoPtr > cellTopos(1,hexTopo);
  NewMeshGeometryPtr meshGeometry = Teuchos::rcp( new NewMeshGeometry(vertices, elementVertices, cellTopos) );

  NewMesh mesh(meshGeometry);
  
  if (mesh.cellCount() != 1) {
    success = false;
    cout << "After initialization, mesh doesn't have the expected number of cells.\n";
  }
  
  if (mesh.activeCellCount() != 1) {
    success = false;
    cout << "After initialization, mesh doesn't have the expected number of active cells.\n";
  }
  
  RefinementPatternPtr hexRefPattern = RefinementPattern::regularRefinementPatternHexahedron();

  mesh.refineCell(0, hexRefPattern);

  if (mesh.cellCount() != 9) {
    success = false;
    cout << "After refinement, mesh doesn't have the expected number of cells.\n";
  }
  
  if (mesh.activeCellCount() != 8) {
    success = false;
    cout << "After refinement, mesh doesn't have the expected number of active cells.\n";
  }
  
  mesh.refineCell(1, hexRefPattern);

  if (mesh.cellCount() != 17) {
    success = false;
    cout << "After second refinement, mesh doesn't have the expected number of cells.\n";
  }
  
  if (mesh.activeCellCount() != 15) {
    success = false;
    cout << "After second refinement, mesh doesn't have the expected number of active cells.\n";
  }
 
  // TODO: test more than just the count.  Test vertex locations, say.  Test constraint counts (should be 0 here after first refinement, and there should be some constraints after the second).

  return success;
}