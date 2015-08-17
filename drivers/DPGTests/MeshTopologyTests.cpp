#include "MeshTopologyTests.h"

#include "CamelliaCellTools.h"

MeshTopologyTests::MeshTopologyTests()
{

}

void MeshTopologyTests::runTests(int &numTestsRun, int &numTestsPassed)
{
  setup();
  if (testCellsForEntity())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testNeighborRelationships())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testEntityConstraints())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testConstraintRelaxation())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (test1DMesh())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (test2DMesh())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (test3DMesh())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();


}
void MeshTopologyTests::setup()
{

}

void MeshTopologyTests::teardown()
{

}

bool MeshTopologyTests::test1DMesh()
{
  bool success = true;

  CellTopoPtr line_2 = Camellia::CellTopology::line();
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

  MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );

  MeshTopology mesh(meshGeometry);

  if (mesh.cellCount() != 2)
  {
    success = false;
    cout << "After initialization, mesh doesn't have the expected number of cells.\n";
  }

  if (mesh.activeCellCount() != 2)
  {
    success = false;
    cout << "After initialization, mesh doesn't have the expected number of active cells.\n";
  }

  mesh.refineCell(0, lineRefPattern);
  if (mesh.cellCount() != 4)
  {
    success = false;
    cout << "After refinement, mesh doesn't have the expected number of cells.\n";
  }

  if (mesh.activeCellCount() != 3)
  {
    success = false;
    cout << "After refinement, mesh doesn't have the expected number of active cells.\n";
  }

  mesh.refineCell(2, lineRefPattern);
  if (mesh.cellCount() != 6)
  {
    success = false;
    cout << "After 2nd refinement, mesh doesn't have the expected number of cells.\n";
  }

  if (mesh.activeCellCount() != 4)
  {
    success = false;
    cout << "After 2nd refinement, mesh doesn't have the expected number of active cells.\n";
  }

  return success;
}

vector<double> MeshTopologyTests::makeVertex(double v0)
{
  vector<double> v;
  v.push_back(v0);
  return v;
}

vector<double> MeshTopologyTests::makeVertex(double v0, double v1)
{
  vector<double> v;
  v.push_back(v0);
  v.push_back(v1);
  return v;
}

vector<double> MeshTopologyTests::makeVertex(double v0, double v1, double v2)
{
  vector<double> v;
  v.push_back(v0);
  v.push_back(v1);
  v.push_back(v2);
  return v;
}

bool MeshTopologyTests::test2DMesh()
{
  bool success = true;

  CellTopoPtr quad_4 = Camellia::CellTopology::quad();
  CellTopoPtr tri_3 = Camellia::CellTopology::triangle();
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
  MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );

  MeshTopology mesh(meshGeometry);

  if (mesh.cellCount() != 2)
  {
    success = false;
    cout << "After initialization, mesh doesn't have the expected number of cells.\n";
  }

  if (mesh.activeCellCount() != 2)
  {
    success = false;
    cout << "After initialization, mesh doesn't have the expected number of active cells.\n";
  }

  mesh.refineCell(0, quadRefPattern);

  if (mesh.cellCount() != 6)
  {
    success = false;
    cout << "After quad refinement, mesh doesn't have the expected number of cells.\n";
  }

  if (mesh.activeCellCount() != 5)
  {
    success = false;
    cout << "After quad refinement, mesh doesn't have the expected number of active cells.\n";
  }

  mesh.refineCell(1, triangleRefPattern);

  if (mesh.cellCount() != 10)
  {
    success = false;
    cout << "After triangle refinement, mesh doesn't have the expected number of cells.\n";
  }

  if (mesh.activeCellCount() != 8)
  {
    success = false;
    cout << "After triangle refinement, mesh doesn't have the expected number of active cells.\n";
  }

  // TODO: test more than just the count.  Test vertex locations, say.  Test constraint counts (should be 0 here after both initial cells refined, 1 constrained edge after the first refinement).

  return success;
}

bool MeshTopologyTests::test3DMesh()
{
  bool success = true;

  unsigned spaceDim = 3;
  unsigned hexNodeCount = 1 << spaceDim;
  FieldContainer<double> refHexPoints(hexNodeCount,spaceDim);
  CellTopoPtr hexTopo = Camellia::CellTopology::hexahedron();
  CamelliaCellTools::refCellNodesForTopology(refHexPoints, hexTopo);

  vector< vector<double> > vertices;
  vector< unsigned > hexVertexIndices(hexNodeCount);
  vector< vector<unsigned> > elementVertices;
  for (unsigned nodeIndex=0; nodeIndex<hexNodeCount; nodeIndex++)
  {
    vector<double> vertex(spaceDim);
    for (unsigned d=0; d<spaceDim; d++)
    {
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
  MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );

  MeshTopology mesh(meshGeometry);

  if (mesh.cellCount() != 1)
  {
    success = false;
    cout << "After initialization, mesh doesn't have the expected number of cells.\n";
  }

  if (mesh.activeCellCount() != 1)
  {
    success = false;
    cout << "After initialization, mesh doesn't have the expected number of active cells.\n";
  }

  RefinementPatternPtr hexRefPattern = RefinementPattern::regularRefinementPatternHexahedron();

  mesh.refineCell(0, hexRefPattern);

  if (mesh.cellCount() != 9)
  {
    success = false;
    cout << "After refinement, mesh doesn't have the expected number of cells.\n";
  }

  if (mesh.activeCellCount() != 8)
  {
    success = false;
    cout << "After refinement, mesh doesn't have the expected number of active cells.\n";
  }

  mesh.refineCell(1, hexRefPattern);

  if (mesh.cellCount() != 17)
  {
    success = false;
    cout << "After second refinement, mesh doesn't have the expected number of cells.\n";
  }

  if (mesh.activeCellCount() != 15)
  {
    success = false;
    cout << "After second refinement, mesh doesn't have the expected number of active cells.\n";
  }

  // TODO: test more than just the count.  Test vertex locations, say.  Test constraint counts (should be 0 here after first refinement, and there should be some constraints after the second).

  return success;
}

vector< vector<double> > MeshTopologyTests::quadPoints(double x0, double y0, double width, double height)
{
  vector< vector<double> > v(4); // defined counterclockwise
  v[0] = makeVertex(x0,y0);
  v[1] = makeVertex(x0 + width,y0);
  v[2] = makeVertex(x0 + width,y0 + height);
  v[3] = makeVertex(x0,y0 + height);
  return v;
}

vector< vector<double> > MeshTopologyTests::hexPoints(double x0, double y0, double z0, double width, double height, double depth)
{
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

MeshTopologyPtr MeshTopologyTests::makeRectMesh(double x0, double y0, double width, double height,
    unsigned horizontalCells, unsigned verticalCells)
{
  unsigned spaceDim = 2;
  Teuchos::RCP<MeshTopology> mesh = Teuchos::rcp( new MeshTopology(spaceDim) );
  double dx = width / horizontalCells;
  double dy = height / verticalCells;
  CellTopoPtrLegacy quadTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
  for (unsigned i=0; i<horizontalCells; i++)
  {
    double x = x0 + dx * i;
    for (unsigned j=0; j<verticalCells; j++)
    {
      double y = y0 + dy * j;
      vector< vector<double> > vertices = quadPoints(x, y, dx, dy);
      mesh->addCell(quadTopo, vertices);
    }
  }
  return mesh;
}

MeshTopologyPtr MeshTopologyTests::makeHexMesh(double x0, double y0, double z0, double width, double height, double depth,
    unsigned horizontalCells, unsigned verticalCells, unsigned depthCells)
{
  unsigned spaceDim = 3;
  MeshTopologyPtr mesh = Teuchos::rcp( new MeshTopology(spaceDim) );
  double dx = width / horizontalCells;
  double dy = height / verticalCells;
  double dz = depth / depthCells;
  CellTopoPtrLegacy hexTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ) );
  for (unsigned i=0; i<horizontalCells; i++)
  {
    double x = x0 + dx * i;
    for (unsigned j=0; j<verticalCells; j++)
    {
      double y = y0 + dy * j;
      for (unsigned k=0; k<depthCells; k++)
      {
        double z = z0 + dz * k;
        vector< vector<double> > vertices = hexPoints(x, y, z, dx, dy, dz);
        mesh->addCell(hexTopo, vertices);
      }
    }
  }
  return mesh;
}

void printMeshInfo(MeshTopologyPtr mesh)
{
  unsigned spaceDim = mesh->getDimension();
  unsigned vertexCount = mesh->getEntityCount(0);
  unsigned edgeCount = (spaceDim > 1) ? mesh->getEntityCount(1) : 0;
  unsigned faceCount = (spaceDim > 2) ? mesh->getEntityCount(2) : 0;
  if (vertexCount > 0)
  {
    cout << "Vertices:\n";
    for (int vertexIndex=0; vertexIndex < vertexCount; vertexIndex++)
    {
      mesh->printEntityVertices(0, vertexIndex);
    }
  }
  if (edgeCount > 0)
  {
    cout << "Edges:\n";
    for (int edgeIndex=0; edgeIndex < edgeCount; edgeIndex++)
    {
      cout << "Edge " << edgeIndex << ":\n";
      mesh->printEntityVertices(1, edgeIndex);
    }
  }
  if (faceCount > 0)
  {
    cout << "Faces:\n";
    for (int faceIndex=0; faceIndex < faceCount; faceIndex++)
    {
      cout << "Face " << faceIndex << ":\n";
      mesh->printEntityVertices(2, faceIndex);
    }
  }
}

string typeString(unsigned d)
{
  switch (d)
  {
  case 0:
    return "Vertex";
    break;
  case 1:
    return "Edge";
    break;
  case 2:
    return "Face";
  case 3:
    return "Volume";

  default:
    return "Unknown Entity Type";
    break;
  }
}

bool checkConstraints( MeshTopologyPtr mesh, unsigned entityDim, map<unsigned,pair<IndexType,unsigned> > &expectedConstraints, string meshName = "mesh")
{
  bool success = true;

  // check constraints for entities belonging to active cells
  set<unsigned> activeCells = mesh->getActiveCellIndices();

  for (set<unsigned>::iterator cellIt = activeCells.begin(); cellIt != activeCells.end(); cellIt++)
  {
    unsigned cellIndex = *cellIt;
    CellPtr cell = mesh->getCell(cellIndex);
    vector<unsigned> entitiesForCell = cell->getEntityIndices(entityDim);
    for (vector<unsigned>::iterator entityIt = entitiesForCell.begin(); entityIt != entitiesForCell.end(); entityIt++)
    {
      unsigned entityIndex = *entityIt;

      pair<IndexType,unsigned> constrainingEntity = mesh->getConstrainingEntity(entityDim, entityIndex);

      unsigned constrainingEntityIndex = constrainingEntity.first;
      unsigned constrainingEntityDim = constrainingEntity.second;
      if ((constrainingEntityIndex==entityIndex) && (constrainingEntityDim == entityDim))
      {
        // then we should expect not to have an entry in expectedConstraints:
        if (expectedConstraints.find(entityIndex) != expectedConstraints.end())
        {
          cout << "Expected entity constraint is not imposed in " << meshName << ".\n";
          cout << "Expected " << typeString(entityDim) << " " << entityIndex << " to be constrained by ";
          cout << typeString(expectedConstraints[entityIndex].second) << " " << expectedConstraints[entityIndex].first << endl;
          cout << typeString(entityDim) << " " << entityIndex << " vertices:\n";
          mesh->printEntityVertices(entityDim, entityIndex);
          cout << typeString(expectedConstraints[entityIndex].second) << " " << expectedConstraints[entityIndex].first << " vertices:\n";
          mesh->printEntityVertices(entityDim, expectedConstraints[entityIndex].first);
          success = false;
        }
      }
      else
      {
        if (expectedConstraints.find(entityIndex) == expectedConstraints.end())
        {
          cout << "Unexpected entity constraint is imposed in " << meshName << ".\n";

          string entityType;
          if (entityDim==0)
          {
            entityType = "Vertex ";
          }
          else if (entityDim==1)
          {
            entityType = "Edge ";
          }
          else if (entityDim==2)
          {
            entityType = "Face ";
          }
          else if (entityDim==3)
          {
            entityType = "Volume ";
          }
          string constrainingEntityType;
          if (constrainingEntityDim==0)
          {
            constrainingEntityType = "Vertex ";
          }
          else if (constrainingEntityDim==1)
          {
            constrainingEntityType = "Edge ";
          }
          else if (constrainingEntityDim==2)
          {
            constrainingEntityType = "Face ";
          }
          else if (constrainingEntityDim==3)
          {
            constrainingEntityType = "Volume ";
          }

          cout << entityType << entityIndex << " unexpectedly constrained by " << constrainingEntityType << constrainingEntityIndex << endl;
          cout << entityType << entityIndex << " vertices:\n";
          mesh->printEntityVertices(entityDim, entityIndex);
          cout << constrainingEntityType << constrainingEntityIndex << " vertices:\n";
          mesh->printEntityVertices(constrainingEntityDim, constrainingEntityIndex);
          success = false;
        }
        else
        {
          unsigned expectedConstrainingEntity = expectedConstraints[entityIndex].first;
          if (expectedConstrainingEntity != constrainingEntityIndex)
          {
            cout << "The constraining entity is not the expected one in " << meshName << ".\n";
            cout << "Expected " << typeString(entityDim) << " " << entityIndex << " to be constrained by ";
            cout << typeString(expectedConstraints[entityIndex].second) << " " << expectedConstrainingEntity;
            cout << "; was constrained by " << constrainingEntityIndex << endl;
            cout << typeString(entityDim) << " " << entityIndex << " vertices:\n";
            mesh->printEntityVertices(entityDim, entityIndex);
            cout << typeString(expectedConstraints[entityIndex].second) << " " << expectedConstrainingEntity << " vertices:\n";
            mesh->printEntityVertices(entityDim, expectedConstrainingEntity);
            cout << typeString(constrainingEntityDim) << " " << constrainingEntityIndex << " vertices:\n";
            mesh->printEntityVertices(constrainingEntityDim, constrainingEntityIndex);
            success = false;
          }
        }
      }
    }
  }
  return success;
}

bool MeshTopologyTests::testCellsForEntity()
{
  bool success = true;

  // to begin, a simple test that cellsForEntity()'s side ordinals are consistent

  MeshTopologyPtr mesh2D = makeRectMesh(0.0, 0.0, 1.0, 1.0, 1, 1);
  IndexType cellID = 0;
  CellPtr cell = mesh2D->getCell(cellID); // the only cell in the mesh

  int spaceDim = mesh2D->getDimension();
  int sideDim = spaceDim - 1;

  for (int d=0; d<spaceDim; d++)
  {
    int scCount = cell->topology()->getSubcellCount(d);
    for (int scord=0; scord<scCount; scord++)
    {
      IndexType scEntityIndex = cell->entityIndex(d, scord);
      set< pair<IndexType, unsigned> > cellsContainingEntity = mesh2D->getCellsContainingEntity(d, scEntityIndex);
      vector<IndexType> sidesContainingEntity = mesh2D->getSidesContainingEntity(d, scEntityIndex);
      if (cellsContainingEntity.size() != 1)
      {
        cout << "cellsContainingEntity should have exactly one entry, but has " << cellsContainingEntity.size() << endl;
        success = false;
      }
      else
      {
        pair<IndexType, unsigned> cellEntry = *(cellsContainingEntity.begin());
//        cout << "for d " << d << ", scord " << scord << ", cell containing entity is (" << cellEntry.first << ", " << cellEntry.second << ")\n";
        if (cellEntry.first != cellID)
        {
          cout << "Expected cellEntry.first = " << cellID << ", but is " << cellEntry.first << endl;
          success = false;
        }
        else
        {
          unsigned sideOrdinal = cellEntry.second;
          IndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
          if (std::find(sidesContainingEntity.begin(), sidesContainingEntity.end(), sideEntityIndex) == sidesContainingEntity.end())
          {
            cout << "The side returned by getCellsContainingEntity() does not contain the specified entity.\n";
            success = false;
          }
        }
      }
    }
  }

  return success;
}

bool MeshTopologyTests::testEntityConstraints()
{
  bool success = true;

  // make two simple meshes
  MeshTopologyPtr mesh2D = makeRectMesh(0.0, 0.0, 2.0, 1.0,
                                        2, 1);
  MeshTopologyPtr mesh3D = makeHexMesh(0.0, 0.0, 0.0, 2.0, 4.0, 3.0,
                                       2, 2, 1);

  unsigned vertexDim = 0;
  unsigned edgeDim = 1;
  unsigned faceDim = 2;

  // first, check that unconstrained edges and faces are unconstrained

  set< unsigned > boundaryEdges;
  set< unsigned > internalEdges;

  for (unsigned cellIndex=0; cellIndex<mesh2D->cellCount(); cellIndex++)
  {
    CellPtr cell = mesh2D->getCell(cellIndex);
    unsigned sideCount = cell->getSideCount();

    for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      unsigned edgeIndex = cell->entityIndex(edgeDim, sideOrdinal);
      unsigned numCells = mesh2D->getActiveCellCount(edgeDim,edgeIndex);
      if (numCells == 1)   // boundary edge
      {
        boundaryEdges.insert(edgeIndex);
      }
      else if (numCells == 2)
      {
        internalEdges.insert(edgeIndex);
      }
      else
      {
        success = false;
        cout << "testEntityConstraints: In initial 2D mesh, edge " << edgeIndex << " has active cell count of " << numCells << ".\n";
      }
    }
  }
  if (internalEdges.size() != 1)
  {
    success = false;
    cout << "testEntityConstraints: In initial 2D mesh, there are " << internalEdges.size() << " internal edges (expected 1).\n";
  }
  for (set<unsigned>::iterator edgeIt=internalEdges.begin(); edgeIt != internalEdges.end(); edgeIt++)
  {
    unsigned edgeIndex = *edgeIt;
    unsigned constrainingEntityIndex = mesh2D->getConstrainingEntity(edgeDim,edgeIndex).first;
    if (constrainingEntityIndex != edgeIndex)
    {
      success = false;
      cout << "testEntityConstraints: In initial 2D mesh, internal edge is constrained by a different edge.\n";
    }
  }

  set<unsigned> boundaryFaces;
  set<unsigned> internalFaces;
  map<unsigned, vector<unsigned> > faceToEdges;
  for (unsigned cellIndex=0; cellIndex<mesh3D->cellCount(); cellIndex++)
  {
    CellPtr cell = mesh3D->getCell(cellIndex);
    unsigned sideCount = cell->getSideCount();

    for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      unsigned faceIndex = cell->entityIndex(faceDim, sideOrdinal);
      unsigned numCells = mesh3D->getActiveCellCount(faceDim,faceIndex);
      if (numCells == 1)   // boundary face
      {
        boundaryFaces.insert(faceIndex);
      }
      else if (numCells == 2)
      {
        internalFaces.insert(faceIndex);
      }
      else
      {
        success = false;
        cout << "testEntityConstraints: In initial 3D mesh, face " << faceIndex << " has active cell count of " << numCells << ".\n";
      }

      if (faceToEdges.find(faceIndex) == faceToEdges.end())
      {
        CellTopoPtr faceTopo = cell->topology()->getSubcell(faceDim, sideOrdinal);
        unsigned numEdges = faceTopo->getSubcellCount(edgeDim);
        vector<unsigned> edgeIndices(numEdges);
        for (unsigned edgeOrdinal=0; edgeOrdinal<numEdges; edgeOrdinal++)
        {
          edgeIndices[edgeOrdinal] = mesh3D->getFaceEdgeIndex(faceIndex, edgeOrdinal);
        }
      }
    }
  }

  if (internalFaces.size() != 4)
  {
    success = false;
    cout << "testEntityConstraints: In initial 3D mesh, there are " << internalFaces.size() << " internal faces (expected 4).\n";
  }
  for (set<unsigned>::iterator faceIt=internalFaces.begin(); faceIt != internalFaces.end(); faceIt++)
  {
    unsigned faceIndex = *faceIt;
    unsigned constrainingEntityIndex = mesh3D->getConstrainingEntity(faceDim,faceIndex).first;
    if (constrainingEntityIndex != faceIndex)
    {
      success = false;
      cout << "testEntityConstraints: In initial 3D mesh, internal face is constrained by a different face.\n";
    }
  }

  // now, make a single refinement in each mesh:
  unsigned cellToRefine2D = 0, cellToRefine3D = 3;
  mesh2D->refineCell(cellToRefine2D, RefinementPattern::regularRefinementPatternQuad());
  mesh3D->refineCell(cellToRefine3D, RefinementPattern::regularRefinementPatternHexahedron());

//  printMeshInfo(mesh2D);

  // figure out which faces/edges were refined and add the corresponding

  map<unsigned,pair<IndexType,unsigned> > expectedEdgeConstraints2D;
  set<unsigned> refinedEdges;
  for (set<unsigned>::iterator edgeIt=boundaryEdges.begin(); edgeIt != boundaryEdges.end(); edgeIt++)
  {
    set<unsigned> children = mesh2D->getChildEntitiesSet(edgeDim, *edgeIt);
    if (children.size() > 0)
    {
      refinedEdges.insert(*edgeIt);
      boundaryEdges.insert(children.begin(), children.end());
    }
  }
  for (set<unsigned>::iterator edgeIt=internalEdges.begin(); edgeIt != internalEdges.end(); edgeIt++)
  {
    set<unsigned> children = mesh2D->getChildEntitiesSet(edgeDim, *edgeIt);
    if (children.size() > 0)
    {
      refinedEdges.insert(*edgeIt);
      internalEdges.insert(children.begin(), children.end());
      for (set<unsigned>::iterator childIt = children.begin(); childIt != children.end(); childIt++)
      {
        unsigned childIndex = *childIt;
        expectedEdgeConstraints2D[childIndex] = make_pair(*edgeIt, edgeDim);
      }
    }
  }
  // 1 quad refined: expect 4 refined edges
  if (refinedEdges.size() != 4)
  {
    success = false;
    cout << "After initial refinement, 2D mesh has " << refinedEdges.size() << " refined edges (expected 4).\n";
  }
  checkConstraints(mesh2D, edgeDim, expectedEdgeConstraints2D);

  set<unsigned> refinedFaces;
  map<unsigned,pair<IndexType,unsigned> > expectedFaceConstraints3D;
  map<unsigned,pair<IndexType,unsigned> > expectedEdgeConstraints3D;

  for (set<unsigned>::iterator faceIt=boundaryFaces.begin(); faceIt != boundaryFaces.end(); faceIt++)
  {
    set<unsigned> children = mesh3D->getChildEntitiesSet(faceDim, *faceIt);
    if (children.size() > 0)
    {
      refinedFaces.insert(*faceIt);
      boundaryFaces.insert(children.begin(), children.end());
    }
  }

  for (set<unsigned>::iterator faceIt=internalFaces.begin(); faceIt != internalFaces.end(); faceIt++)
  {
    vector<unsigned> children = mesh3D->getChildEntities(faceDim, *faceIt);
    if (children.size() > 0)
    {
      refinedFaces.insert(*faceIt);
      internalFaces.insert(children.begin(), children.end());
      for (unsigned childOrdinal = 0; childOrdinal < children.size(); childOrdinal++)
      {
        unsigned childIndex = children[childOrdinal];
        expectedFaceConstraints3D[childIndex] = make_pair(*faceIt, faceDim);
        unsigned numEdges = 4;
        unsigned internalEdgeCount = 0; // for each child of a quad, we expect to have 2 internal edges
        for (unsigned edgeOrdinal=0; edgeOrdinal<numEdges; edgeOrdinal++)
        {
          unsigned edgeIndex = mesh3D->getFaceEdgeIndex(childIndex, edgeOrdinal);
          unsigned activeCellCount = mesh3D->getActiveCellCount(edgeDim, edgeIndex);
          if (activeCellCount==2)
          {
            internalEdgeCount++;
            expectedEdgeConstraints3D[edgeIndex] = make_pair(*faceIt, faceDim);
          }
          else if (activeCellCount==1)     // hanging edge
          {
            if (! mesh3D->entityHasParent(edgeDim, edgeIndex))
            {
              cout << "Hanging edge with edgeIndex " << edgeIndex << " (in face " << childIndex << ") does not have a parent edge.\n";
              cout << "Edge vertices:\n";
              mesh3D->printEntityVertices(edgeDim, edgeIndex);
              cout << "Face vertices:\n";
              mesh3D->printEntityVertices(faceDim, childIndex);
              success = false;
            }
            else
            {
              unsigned edgeParentIndex = mesh3D->getEntityParent(edgeDim, edgeIndex);
              expectedEdgeConstraints3D[edgeIndex] = make_pair(edgeParentIndex, edgeDim);
            }
          }
          else
          {
            cout << "Unexpected number of active cells: " << activeCellCount << endl;
          }
        }
        if (internalEdgeCount != 2)
        {
          cout << "Expected internalEdgeCount to be 2; was " << internalEdgeCount << endl;
          success = false;
        }
      }
    }
  }
  // 1 hex refined: expect 6 refined faces
  if (refinedFaces.size() != 6)
  {
    success = false;
    cout << "After initial refinement, 3D mesh has " << refinedFaces.size() << " refined faces (expected 6).\n";
  }
  if (! checkConstraints(mesh3D, faceDim, expectedFaceConstraints3D, "refined 3D mesh") )
  {
    cout << "Failed face constraint check for refined 3D mesh." << endl;
    success = false;
  }
  if (! checkConstraints(mesh3D, edgeDim, expectedEdgeConstraints3D, "refined 3D mesh") )
  {
    cout << "Failed edge constraint check for refined 3D mesh." << endl;
    success = false;
  }

  // now, we refine one of the children of the refined cells in each mesh, to produce a 2-level constraint
  set<unsigned> edgeChildren2D;
  set<unsigned> cellsForEdgeChildren2D;
  for (map<unsigned,pair<IndexType,unsigned> >::iterator edgeConstraint=expectedEdgeConstraints2D.begin();
       edgeConstraint != expectedEdgeConstraints2D.end(); edgeConstraint++)
  {
    edgeChildren2D.insert(edgeConstraint->first);
    unsigned cellIndex = mesh2D->getActiveCellIndices(edgeDim, edgeConstraint->first).begin()->first;
    cellsForEdgeChildren2D.insert(cellIndex);
//    cout << "cellsForEdgeChildren2D: " << cellIndex << endl;
  }

  // one of these has (1,0) as one of its vertices.  Let's figure out which one:
  unsigned vertexIndex;
  if (! mesh2D->getVertexIndex(makeVertex(1, 0), vertexIndex) )
  {
    cout << "Error: vertex not found.\n";
    success = false;
  }

  vector< pair<unsigned,unsigned> > cellsForVertex = mesh2D->getActiveCellIndices(vertexDim, vertexIndex);
  if (cellsForVertex.size() != 2)
  {
    cout << "cellsForVertex should have 2 entries; has " << cellsForVertex.size() << endl;
    success = false;
  }
  unsigned childCellForVertex, childCellConstrainedEdge;
  set<unsigned> childNewlyConstrainingEdges; // the two interior edges that we break
  for (vector< pair<unsigned,unsigned> >::iterator cellIt=cellsForVertex.begin(); cellIt != cellsForVertex.end(); cellIt++)
  {
//    cout << "cellsForVertex: " << cellIt->first << endl;
    if ( cellsForEdgeChildren2D.find( cellIt->first ) != cellsForEdgeChildren2D.end() )
    {
      // found match
      childCellForVertex = cellIt->first;
      // now, figure out which of the "edgeChildren2D" is shared by this cell:
      CellPtr cell = mesh2D->getCell(childCellForVertex);
      unsigned numEdges = cell->getSideCount();
      for (unsigned edgeOrdinal=0; edgeOrdinal<numEdges; edgeOrdinal++)
      {
        unsigned edgeIndex = cell->entityIndex(edgeDim, edgeOrdinal);
        if (edgeChildren2D.find(edgeIndex) != edgeChildren2D.end())
        {
          childCellConstrainedEdge = edgeIndex;
        }
        else if ( mesh2D->getActiveCellCount(edgeDim, edgeIndex) == 2 )
        {
          childNewlyConstrainingEdges.insert(edgeIndex);
        }
      }
    }
  }
  if (childNewlyConstrainingEdges.size() != 2)
  {
    cout << "Expected 2 newly constraining edges after 2nd refinement of 2D mesh, but found " << childNewlyConstrainingEdges.size() << endl;
    success = false;
  }

  // refine the cell that matches (1,0):
  mesh2D->refineCell(childCellForVertex, RefinementPattern::regularRefinementPatternQuad());

  // now, fix the expected edge constraints, then check them...
  set<unsigned> childEdges = mesh2D->getChildEntitiesSet(edgeDim, childCellConstrainedEdge);
  if (childEdges.size() != 2)
  {
    cout << "Expected 2 child edges, but found " << childEdges.size() << ".\n";
    success = false;
  }
  for (set<unsigned>::iterator edgeIt = childEdges.begin(); edgeIt != childEdges.end(); edgeIt++)
  {
    expectedEdgeConstraints2D[*edgeIt] = expectedEdgeConstraints2D[childCellConstrainedEdge];
  }
  expectedEdgeConstraints2D.erase(childCellConstrainedEdge);
  for (set<unsigned>::iterator edgeIt = childNewlyConstrainingEdges.begin(); edgeIt != childNewlyConstrainingEdges.end(); edgeIt++)
  {
    set<unsigned> newChildEdges = mesh2D->getChildEntitiesSet(edgeDim, *edgeIt);
    for (set<unsigned>::iterator newEdgeIt = newChildEdges.begin(); newEdgeIt != newChildEdges.end(); newEdgeIt++)
    {
      expectedEdgeConstraints2D[*newEdgeIt] = make_pair(*edgeIt,edgeDim);
    }
  }

  if (! checkConstraints(mesh2D, edgeDim, expectedEdgeConstraints2D, "twice-refined 2D mesh") )
  {
    cout << "Failed constraint check for twice-refined 2D mesh." << endl;
    success = false;
  }

  // now, do a second level of refinement for 3D mesh
  // one of these has (1,2,0) as one of its vertices.  Let's figure out which one:
  if (! mesh3D->getVertexIndex(makeVertex(1, 2, 0), vertexIndex) )
  {
    cout << "Error: vertex not found.\n";
    success = false;
  }

  cellsForVertex = mesh3D->getActiveCellIndices(vertexDim, vertexIndex);
  if (cellsForVertex.size() != 4)
  {
    cout << "cellsForVertex should have 4 entries; has " << cellsForVertex.size() << endl;
    success = false;
  }

  vector<unsigned> justCellsForVertex;
  for (vector< pair<unsigned,unsigned> >::iterator entryIt = cellsForVertex.begin(); entryIt != cellsForVertex.end(); entryIt++)
  {
    justCellsForVertex.push_back(entryIt->first);
  }
  vector<unsigned> childCellIndices = mesh3D->getCell(cellToRefine3D)->getChildIndices(mesh3D);
  std::sort(childCellIndices.begin(), childCellIndices.end());
  vector<unsigned> matches(childCellIndices.size() + cellsForVertex.size());
  vector<unsigned>::iterator matchEnd = std::set_intersection(justCellsForVertex.begin(), justCellsForVertex.end(), childCellIndices.begin(), childCellIndices.end(), matches.begin());
  matches.resize(matchEnd-matches.begin());

  if (matches.size() != 1)
  {
    cout << "matches should have exactly one entry, but has " << matches.size();
    success = false;
  }
  unsigned childCellIndex = matches[0];
  CellPtr childCell = mesh3D->getCell(childCellIndex);
  set<unsigned> childInteriorUnconstrainedFaces;
  set<unsigned> childInteriorConstrainedFaces;
  unsigned faceCount = childCell->getSideCount();
  for (unsigned faceOrdinal=0; faceOrdinal<faceCount; faceOrdinal++)
  {
    unsigned faceIndex = childCell->entityIndex(faceDim, faceOrdinal);
    if (mesh3D->getActiveCellCount(faceDim, faceIndex) == 1)
    {
      // that's an interior constrained face, or a boundary face
      if (expectedFaceConstraints3D.find(faceIndex) != expectedFaceConstraints3D.end())
      {
        // constrained face
        childInteriorConstrainedFaces.insert(faceIndex);
      }
    }
    else if (mesh3D->getActiveCellCount(faceDim, faceIndex) == 2)
    {
      // an interior unconstrained face
      childInteriorUnconstrainedFaces.insert(faceIndex);
    }
    else
    {
      cout << "Error: unexpected active cell count.  Expected 1 or 2, but was " << mesh3D->getActiveCellCount(faceDim, faceIndex) << endl;
      success = false;
    }
  }
//  Camellia::print("childInteriorUnconstrainedFaces", childInteriorUnconstrainedFaces);
//  Camellia::print("childInteriorConstrainedFaces", childInteriorConstrainedFaces);

  mesh3D->refineCell(childCellIndex, RefinementPattern::regularRefinementPatternHexahedron());

  // update expected face and edge constraints
//  set<unsigned> edgeConstraintsToDrop;
  for (set<unsigned>::iterator faceIt=childInteriorConstrainedFaces.begin(); faceIt != childInteriorConstrainedFaces.end(); faceIt++)
  {
    unsigned faceIndex = *faceIt;
    set<unsigned> newChildFaces = mesh3D->getChildEntitiesSet(faceDim, faceIndex);
    for (set<unsigned>::iterator newChildIt=newChildFaces.begin(); newChildIt != newChildFaces.end(); newChildIt++)
    {
      unsigned newChildIndex = *newChildIt;
      expectedFaceConstraints3D[newChildIndex] = expectedFaceConstraints3D[faceIndex];
//      cout << "Expecting two-level face constraint: face " << newChildIndex << " constrained by face " << expectedFaceConstraints3D[newChildIndex].first << endl;
    }
    unsigned numEdges = mesh3D->getSubEntityCount(faceDim, faceIndex, edgeDim);
    set<IndexType> childEdgesOnParentBoundary;
    for (unsigned edgeOrdinal=0; edgeOrdinal<numEdges; edgeOrdinal++)
    {
      unsigned edgeIndex = mesh3D->getSubEntityIndex(faceDim, faceIndex, edgeDim, edgeOrdinal);
      set<unsigned> newChildEdges = mesh3D->getChildEntitiesSet(edgeDim, edgeIndex);
      for (set<unsigned>::iterator newChildIt=newChildEdges.begin(); newChildIt != newChildEdges.end(); newChildIt++)
      {
        unsigned newChildIndex = *newChildIt;
        expectedEdgeConstraints3D[newChildIndex] = expectedEdgeConstraints3D[edgeIndex];
//        cout << "Expecting two-level edge constraint: edge " << newChildIndex << " constrained by ";
//        cout << typeString(expectedEdgeConstraints3D[newChildIndex].second) << " " << expectedEdgeConstraints3D[newChildIndex].first << endl;
        childEdgesOnParentBoundary.insert(newChildIndex);
//        edgeConstraintsToDrop.insert(edgeIndex);
      }
    }

    for (set<unsigned>::iterator newChildIt=newChildFaces.begin(); newChildIt != newChildFaces.end(); newChildIt++)
    {
      unsigned newChildFaceIndex = *newChildIt;
      int numEdges = mesh3D->getSubEntityCount(faceDim, newChildFaceIndex, edgeDim);
      for (unsigned edgeOrdinal=0; edgeOrdinal<numEdges; edgeOrdinal++)
      {
        unsigned newChildEdgeIndex = mesh3D->getSubEntityIndex(faceDim, newChildFaceIndex, edgeDim, edgeOrdinal);
        if (childEdgesOnParentBoundary.find(newChildEdgeIndex) == childEdgesOnParentBoundary.end())
        {
          expectedEdgeConstraints3D[newChildEdgeIndex] = expectedFaceConstraints3D[faceIndex];
        }
      }
    }

    expectedFaceConstraints3D.erase(faceIndex);
  }
//  for (set<unsigned>::iterator edgeToDropIt=edgeConstraintsToDrop.begin(); edgeToDropIt != edgeConstraintsToDrop.end(); edgeToDropIt++) {
//    expectedEdgeConstraints3D.erase(*edgeToDropIt);
//  }
  for (set<unsigned>::iterator faceIt=childInteriorUnconstrainedFaces.begin(); faceIt != childInteriorUnconstrainedFaces.end(); faceIt++)
  {
    unsigned faceIndex = *faceIt;
    set<unsigned> newChildFaces = mesh3D->getChildEntitiesSet(faceDim, faceIndex);
    for (set<unsigned>::iterator newChildIt=newChildFaces.begin(); newChildIt != newChildFaces.end(); newChildIt++)
    {
      unsigned newChildIndex = *newChildIt;
      expectedFaceConstraints3D[newChildIndex] = make_pair(faceIndex, faceDim);
    }
    expectedFaceConstraints3D.erase(faceIndex);
    unsigned numEdges = mesh3D->getSubEntityCount(faceDim, faceIndex, edgeDim);
    set<IndexType> childEdgesOnParentBoundary;
    for (unsigned edgeOrdinal=0; edgeOrdinal<numEdges; edgeOrdinal++)
    {
      unsigned edgeIndex = mesh3D->getSubEntityIndex(faceDim, faceIndex, edgeDim, edgeOrdinal);
      set<unsigned> newChildEdges = mesh3D->getChildEntitiesSet(edgeDim, edgeIndex);
      for (set<unsigned>::iterator newChildIt=newChildEdges.begin(); newChildIt != newChildEdges.end(); newChildIt++)
      {
        unsigned newChildIndex = *newChildIt;
        if (expectedEdgeConstraints3D.find(newChildIndex) == expectedEdgeConstraints3D.end())   // only impose edge constraint if there is not one already present
        {
          expectedEdgeConstraints3D[newChildIndex] = make_pair(edgeIndex,edgeDim);
        }
        childEdgesOnParentBoundary.insert(newChildIndex);
      }
    }
    for (set<unsigned>::iterator newChildIt=newChildFaces.begin(); newChildIt != newChildFaces.end(); newChildIt++)
    {
      unsigned newChildFaceIndex = *newChildIt;
      int numEdges = mesh3D->getSubEntityCount(faceDim, newChildFaceIndex, edgeDim);
      for (unsigned edgeOrdinal=0; edgeOrdinal<numEdges; edgeOrdinal++)
      {
        unsigned newChildEdgeIndex = mesh3D->getSubEntityIndex(faceDim, newChildFaceIndex, edgeDim, edgeOrdinal);
        if (childEdgesOnParentBoundary.find(newChildEdgeIndex) == childEdgesOnParentBoundary.end())
        {
          if (expectedEdgeConstraints3D.find(newChildEdgeIndex) == expectedEdgeConstraints3D.end())   // only impose edge constraint if there is not one already present
          {
            expectedEdgeConstraints3D[newChildEdgeIndex] = make_pair(faceIndex, faceDim);
          }
        }
      }
    }
  }

  if (! checkConstraints(mesh3D, edgeDim, expectedEdgeConstraints3D, "twice-refined 3D mesh") )
  {
    cout << "Failed edge constraint check for twice-refined 3D mesh." << endl;
    success = false;
  }

  if (! checkConstraints(mesh3D, faceDim, expectedFaceConstraints3D, "twice-refined 3D mesh") )
  {
    cout << "Failed face constraint check for twice-refined 3D mesh." << endl;
    success = false;
  }

  return success;
}

bool MeshTopologyTests::testConstraintRelaxation()
{
  bool success = true;

  // tests to confirm that constraints are appropriately relaxed when refinements render neighbors compatible.

  // make two simple meshes
  MeshTopologyPtr mesh2D = makeRectMesh(0.0, 0.0, 2.0, 1.0,
                                        2, 1); // 2 initial elements
  MeshTopologyPtr mesh3D = makeHexMesh(0.0, 0.0, 0.0, 2.0, 4.0, 3.0,
                                       2, 2, 1); // 4 initial elements

  mesh2D->refineCell(0, RefinementPattern::regularRefinementPatternQuad());
  mesh2D->refineCell(1, RefinementPattern::regularRefinementPatternQuad());

  mesh3D->refineCell(0, RefinementPattern::regularRefinementPatternHexahedron());
  mesh3D->refineCell(1, RefinementPattern::regularRefinementPatternHexahedron());
  mesh3D->refineCell(2, RefinementPattern::regularRefinementPatternHexahedron());
  mesh3D->refineCell(3, RefinementPattern::regularRefinementPatternHexahedron());

  // empty containers:
  map<unsigned,pair<IndexType,unsigned> > expectedEdgeConstraints2D;
  map<unsigned,pair<IndexType,unsigned> > expectedFaceConstraints3D;
  map<unsigned,pair<IndexType,unsigned> > expectedEdgeConstraints3D;

  int edgeDim = 1, faceDim = 2;

  if (! checkConstraints(mesh2D, edgeDim, expectedEdgeConstraints2D, "compatible 2D mesh") )
  {
    cout << "Failed edge constraint check for compatible 2D mesh." << endl;
    success = false;
  }

  if (! checkConstraints(mesh3D, edgeDim, expectedEdgeConstraints3D, "compatible 3D mesh") )
  {
    cout << "Failed edge constraint check for compatible 3D mesh." << endl;
    success = false;
  }

  if (! checkConstraints(mesh3D, faceDim, expectedFaceConstraints3D, "compatible 3D mesh") )
  {
    cout << "Failed face constraint check for compatible 3D mesh." << endl;
    success = false;
  }

  return success;
}

bool MeshTopologyTests::testNeighborRelationships()
{
  bool success = true;

  MeshTopologyPtr mesh2D = makeRectMesh(0.0, 0.0, 1.0, 2.0,
                                        1, 2); // 2 initial elements

  // the following refinement sequence is interesting, in that the final refinement of cell 0
  // reduces a 2-irregular mesh to a 1-irregular one.  It's a nice test of our refinement logic
  mesh2D->refineCell(1, RefinementPattern::regularRefinementPatternQuad());
  mesh2D->refineCell(2, RefinementPattern::regularRefinementPatternQuad());
  mesh2D->refineCell(3, RefinementPattern::regularRefinementPatternQuad());
  mesh2D->refineCell(0, RefinementPattern::regularRefinementPatternQuad());

  // not worth going into the details, but cell 6's side 0 should point to cell 17's side 2.  cell 17's side 2 should point to cell 2, side 0 (an inactive peer).
  CellPtr cell6 = mesh2D->getCell(6);
  if (cell6->getNeighborInfo(0,mesh2D).first != 17)
  {
    success = false;
    cout << "cell 6's neighbor on side 0 is not 17, as expected, but " << cell6->getNeighborInfo(0,mesh2D).first << ".\n";
  }

  CellPtr cell17 = mesh2D->getCell(17);
  if (cell17->getNeighborInfo(2,mesh2D).first != 2)
  {
    success = false;
    cout << "cell 17's neighbor on side 2 is not 2, as expected, but " << cell17->getNeighborInfo(2,mesh2D).first << ".\n";
  }

  return success;
}