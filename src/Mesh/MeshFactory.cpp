//
//  MeshFactory.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/21/13.
//
//

#include "MeshFactory.h"

#include "ParametricCurve.h"

#include "GnuPlotUtil.h"

map<int,int> MeshFactory::_emptyIntIntMap;

static ParametricCurvePtr parametricRect(double width, double height, double x0, double y0) {
  // starts at the positive x axis and proceeds counter-clockwise, just like our parametric circle
  vector< pair<double, double> > vertices;
  vertices.push_back(make_pair(x0 + width/2.0, y0 + 0));
  vertices.push_back(make_pair(x0 + width/2.0, y0 + height/2.0));
  vertices.push_back(make_pair(x0 - width/2.0, y0 + height/2.0));
  vertices.push_back(make_pair(x0 - width/2.0, y0 - height/2.0));
  vertices.push_back(make_pair(x0 + width/2.0, y0 - height/2.0));
  return ParametricCurve::polygon(vertices);
}

/*class ParametricRect : public ParametricCurve {
  double _width, _height, _x0, _y0;
  vector< ParametricCurvePtr > _edgeLines;
  vector< double > _switchValues;
public:
  ParametricRect(double width, double height, double x0, double y0) {
    // starts at the positive x axis and proceeds counter-clockwise, just like our parametric circle

    _width = width; _height = height; _x0 = x0; _y0 = y0;
    _edgeLines.push_back(ParametricCurve::line(x0 + width/2.0, y0 + 0, x0 + width/2.0, y0 + height/2.0));
    _edgeLines.push_back(ParametricCurve::line(x0 + width/2.0, y0 + height/2.0, x0 - width/2.0, y0 + height/2.0));
    _edgeLines.push_back(ParametricCurve::line(x0 - width/2.0, y0 + height/2.0, x0 - width/2.0, y0 - height/2.0));
    _edgeLines.push_back(ParametricCurve::line(x0 - width/2.0, y0 - height/2.0, x0 + width/2.0, y0 - height/2.0));
    _edgeLines.push_back(ParametricCurve::line(x0 + width/2.0, y0 - height/2.0, x0 + width/2.0, y0 + 0));

    // switchValues are the points in (0,1) where we switch from one edge line to the next
    _switchValues.push_back(0.0);
    _switchValues.push_back(0.125);
    _switchValues.push_back(0.375);
    _switchValues.push_back(0.625);
    _switchValues.push_back(0.875);
    _switchValues.push_back(1.0);
  }
  void value(double t, double &x, double &y) {
    for (int i=0; i<_edgeLines.size(); i++) {
      if ( (t >= _switchValues[i]) && (t <= _switchValues[i+1]) ) {
        double edge_t = (t - _switchValues[i]) / (_switchValues[i+1] - _switchValues[i]);
        _edgeLines[i]->value(edge_t, x, y);
        return;
      }
    }
  }
};*/

MeshPtr MeshFactory::quadMesh(BilinearFormPtr bf, int H1Order, FieldContainer<double> &quadNodes, int pToAddTest) {
  if (quadNodes.size() != 8) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "quadNodes must be 4 x 2");
  }
  int spaceDim = 2;
  vector< vector<double> > vertices;
  for (int i=0; i<4; i++) {
    vector<double> vertex(spaceDim);
    vertex[0] = quadNodes[2*i];
    vertex[1] = quadNodes[2*i+1];
    vertices.push_back(vertex);
  }
  vector< vector<unsigned> > elementVertices;
  vector<unsigned> cell0;
  cell0.push_back(0);
  cell0.push_back(1);
  cell0.push_back(2);
  cell0.push_back(3);
  elementVertices.push_back(cell0);

  MeshPtr mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, bf, H1Order, pToAddTest) );
  return mesh;
}

MeshPtr MeshFactory::quadMesh(BilinearFormPtr bf, int H1Order, int pToAddTest,
                              double width, double height, int horizontalElements, int verticalElements, bool divideIntoTriangles,
                              double x0, double y0) {
  FieldContainer<double> quadPoints(4,2);
  quadPoints(0,0) = x0;
  quadPoints(0,1) = y0;
  quadPoints(1,0) = x0 + width;
  quadPoints(1,1) = y0;
  quadPoints(2,0) = x0 + width;
  quadPoints(2,1) = y0 + height;
  quadPoints(3,0) = x0;
  quadPoints(3,1) = y0 + height;

  int testOrder = pToAddTest + H1Order; // buildQuadMesh's interface asks for the order of the test space, not the delta p.  Better to be consistent in using the delta p going forward...

  return MeshFactory::buildQuadMesh(quadPoints, horizontalElements, verticalElements, bf, H1Order, testOrder, divideIntoTriangles);
}

MeshPtr MeshFactory::quadMeshMinRule(BilinearFormPtr bf, int H1Order, int pToAddTest,
                                      double width, double height, int horizontalElements, int verticalElements,
                                     bool divideIntoTriangles, double x0, double y0) {
  int spaceDim = 2;
  
  vector<vector<double> > vertices;
  vector< vector<unsigned> > allElementVertices;
  
  int numElements = divideIntoTriangles ? horizontalElements * verticalElements * 2 : horizontalElements * verticalElements;
  
  CellTopoPtr topo;
  if (divideIntoTriangles) {
    topo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() ));
  } else {
    topo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  }
  vector< CellTopoPtr > cellTopos(numElements, topo);
  
  FieldContainer<double> quadBoundaryPoints(4,2);
  quadBoundaryPoints(0,0) = x0;
  quadBoundaryPoints(0,1) = y0;
  quadBoundaryPoints(1,0) = x0 + width;
  quadBoundaryPoints(1,1) = y0;
  quadBoundaryPoints(2,0) = x0 + width;
  quadBoundaryPoints(2,1) = y0 + height;
  quadBoundaryPoints(3,0) = x0;
  quadBoundaryPoints(3,1) = y0 + height;
  
  cout << "creating mesh with boundary points:\n" << quadBoundaryPoints;
  
  double southWest_x = quadBoundaryPoints(0,0),
  southWest_y = quadBoundaryPoints(0,1);
  
  double elemWidth = width / horizontalElements;
  double elemHeight = height / verticalElements;
  
  // set up vertices:
  // vertexIndices is for easy vertex lookup by (x,y) index for our Cartesian grid:
  vector< vector<int> > vertexIndices(horizontalElements+1, vector<int>(verticalElements+1));
  for (int i=0; i<=horizontalElements; i++) {
    for (int j=0; j<=verticalElements; j++) {
      vertexIndices[i][j] = vertices.size();
      vector<double> vertex(spaceDim);
      vertex[0] = southWest_x + elemWidth*i;
      vertex[1] = southWest_y + elemHeight*j;
      vertices.push_back(vertex);
    }
  }
  
  for (int i=0; i<horizontalElements; i++) {
    for (int j=0; j<verticalElements; j++) {
      if (!divideIntoTriangles) {
        vector<unsigned> elemVertices;
        elemVertices.push_back(vertexIndices[i][j]);
        elemVertices.push_back(vertexIndices[i+1][j]);
        elemVertices.push_back(vertexIndices[i+1][j+1]);
        elemVertices.push_back(vertexIndices[i][j+1]);
        allElementVertices.push_back(elemVertices);
      } else {
        vector<unsigned> elemVertices1, elemVertices2; // elem1 is SE of quad, elem2 is NW
        elemVertices1.push_back(vertexIndices[i][j]);     // SIDE1 is SOUTH side of quad
        elemVertices1.push_back(vertexIndices[i+1][j]);   // SIDE2 is EAST
        elemVertices1.push_back(vertexIndices[i+1][j+1]); // SIDE3 is diagonal
        elemVertices2.push_back(vertexIndices[i][j+1]);   // SIDE1 is WEST
        elemVertices2.push_back(vertexIndices[i][j]);     // SIDE2 is diagonal
        elemVertices2.push_back(vertexIndices[i+1][j+1]); // SIDE3 is NORTH
        
        allElementVertices.push_back(elemVertices1);
        allElementVertices.push_back(elemVertices2);
      }
    }
  }
  
  MeshGeometryPtr geometry = Teuchos::rcp( new MeshGeometry(vertices, allElementVertices, cellTopos));
  
  MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(geometry) );
  
  return Teuchos::rcp( new Mesh(meshTopology, bf, H1Order, pToAddTest) );
}

MeshPtr MeshFactory::rectilinearMesh(BilinearFormPtr bf, vector<double> dimensions, vector<int> elementCounts, int H1Order, int pToAddTest) {
  int spaceDim = dimensions.size();
  if (pToAddTest==-1) {
    pToAddTest = spaceDim;
  }
  
  if (elementCounts.size() != dimensions.size()) {
    cout << "Element count container must match dimensions container in length.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Element count container must match dimensions container in length.\n");
  }
  
  if (spaceDim == 2) {
    return MeshFactory::quadMeshMinRule(bf, H1Order, dimensions[0], dimensions[1], elementCounts[0], elementCounts[1], pToAddTest);
  }
  
  if (spaceDim != 3) {
    cout << "For now, only spaceDim == 3 is supported by this MeshFactory method.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "For now, only spaceDim == 3 is supported by this MeshFactory method.");
  }
  
  CellTopoPtr topo;
  if (spaceDim==1) {
    topo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Line<2> >() ));
  } else if (spaceDim==2) {
    topo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  } else if (spaceDim==3) {
    topo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ));
  } else {
    cout << "Unsupported spatial dimension.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported spatial dimension");
  }
  
  
  int numElements = 1;
  vector<double> elemLinearMeasures(spaceDim);
  vector<double> origin(spaceDim);
  for (int d=0; d<spaceDim; d++) {
    numElements *= elementCounts[d];
    elemLinearMeasures[d] = 1.0 / elementCounts[d];
    origin[d] = 0;
  }
  vector< CellTopoPtr > cellTopos(numElements, topo);
    
  map< vector<int>, unsigned> vertexLookup;
  vector< vector<double> > vertices;
  
  for (int i=0; i<elementCounts[0]+1; i++) {
    double x = origin[0] + elemLinearMeasures[0] * i;
    
    for (int j=0; j<elementCounts[1]+1; j++) {
      double y = origin[1] + elemLinearMeasures[1] * j;
      
      for (int k=0; k<elementCounts[2]+1; k++) {
        double z = origin[2] + elemLinearMeasures[2] * k;
        
        vector<int> vertexIndex;
        vertexIndex.push_back(i);
        vertexIndex.push_back(j);
        vertexIndex.push_back(k);
        
        vector<double> vertex;
        vertex.push_back(x);
        vertex.push_back(y);
        vertex.push_back(z);
        
        vertexLookup[vertexIndex] = vertices.size();
        vertices.push_back(vertex);
      }
    }
  }
  
  vector< vector<unsigned> > elementVertices;
  for (int i=0; i<elementCounts[0]; i++) {
    for (int j=0; j<elementCounts[1]; j++) {
      for (int k=0; k<elementCounts[2]; k++) {
        vector< vector<int> > vertexIntCoords(8, vector<int>(3));
        vertexIntCoords[0][0] = i;
        vertexIntCoords[0][1] = j;
        vertexIntCoords[0][2] = k;
        vertexIntCoords[1][0] = i+1;
        vertexIntCoords[1][1] = j;
        vertexIntCoords[1][2] = k;
        vertexIntCoords[2][0] = i+1;
        vertexIntCoords[2][1] = j+1;
        vertexIntCoords[2][2] = k;
        vertexIntCoords[3][0] = i;
        vertexIntCoords[3][1] = j+1;
        vertexIntCoords[3][2] = k;
        vertexIntCoords[4][0] = i;
        vertexIntCoords[4][1] = j;
        vertexIntCoords[4][2] = k+1;
        vertexIntCoords[5][0] = i+1;
        vertexIntCoords[5][1] = j;
        vertexIntCoords[5][2] = k+1;
        vertexIntCoords[6][0] = i+1;
        vertexIntCoords[6][1] = j+1;
        vertexIntCoords[6][2] = k+1;
        vertexIntCoords[7][0] = i;
        vertexIntCoords[7][1] = j+1;
        vertexIntCoords[7][2] = k+1;
        
        vector<unsigned> elementVertexOrdinals;
        for (int n=0; n<8; n++) {
          elementVertexOrdinals.push_back(vertexLookup[vertexIntCoords[n]]);
        }
        
        elementVertices.push_back(elementVertexOrdinals);
      }
    }
  }
  
  MeshGeometryPtr geometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos));

  MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(geometry) );
  
  return Teuchos::rcp( new Mesh(meshTopology, bf, H1Order, pToAddTest) );

  // earlier attempt to handle all dimensions is below.  Part of the problem is that the quad
  // node ordering isn't tensor-product (it's counterclockwise).  This makes dimension-independent code
  // extra difficult.
  
//  vector< vector<double> > lineDivisions;
//  vector< vector<double> > vertices;
//
//  vector< vector<int> > vertexIndices;
//  
//  // following for loop does a Cartesian product of the dimensions
//  for (int d=0; d<spaceDim; d++) {
//    vector<double> divisions;
//    vector< vector<double> > newVertices;
//    vector< vector<int> > newVertexIndices;
//    for (int i=0; i<elementCounts[d] + 1; i++) {
//      double xd_i = origin[d] + i * elemLinearMeasures[d];
//      divisions.push_back(xd_i);
//      if (d==0) {
//        vector<int> vertexOrdinal(1,i);
//        newVertexIndices.push_back(vertexOrdinal);
//        newVertices.push_back(vector<double>(1,xd_i));
//      } else {
//        for (int j=0; j<vertices.size(); j++) { // these don't have a d-dimensional entry
//          vector<double> vertex = vertices[j];
//          vertex.push_back(xd_i);
//          newVertices.push_back(vertex);
//          
//          vector<int> vertexIndex = vertexIndices[j];
//          vertexIndex.push_back(i);
//          newVertexIndices.push_back(vertexIndex);
//        }
//      }
//    }
//    vertices = newVertices;
//    vertexIndices = newVertexIndices;
//    lineDivisions.push_back(divisions);
//  }
//
//  map< vector<int>, int> vertexOrdinalLookup;
//  for (int i=0; i<vertexIndices.size(); i++) {
//    vertexOrdinalLookup[vertexIndices[i]] = i;
//  }
//
//  for (int i=0; i<vertexIndices.size(); i++) {
//    vector<int> vertexIndex = vertexIndices[i];
//    bool onMeshBackBoundary = false;
//    for (int d=0; d<spaceDim; d++) {
//      if (vertexIndex[d] == elementCounts[d]) {
//        onMeshBackBoundary = true;
//      }
//    }
//    if (onMeshBackBoundary) continue;
//    // otherwise, we can build an element with this vertex as its bottom/left/near corner
//    
//    vector< vector<int> > elementVertexIndices;
//    for (int d=0; d<spaceDim; d++) {
//      vector< vector<int> > newElementVertexIndices;
//      if (d==0) {
//        vector<int> vi_d0(1,vertexIndex[d]);
//        vector<int> vi_d1(1,vertexIndex[d]+1);
//        newElementVertexIndices.push_back(vi_d0);
//        newElementVertexIndices.push_back(vi_d1);
//      } else {
//        for (int j=0; j<elementVertexIndices.size(); j++) {
//          vector<int> vj = elementVertexIndices[j];
//          vector<int> vj_d0 = vj;
//          vj_d0.push_back(<#const value_type &__x#>)
//          vector<int> vj_d1 = vj;
//          
//        }
//      }
//      elementVertexIndices = newElementVertexIndices;
//    }
//    
//    vector< int > elementVertexOrdinals;
//    
//  }
}

MeshGeometryPtr MeshFactory::shiftedHemkerGeometry(double xLeft, double xRight, double meshHeight, double cylinderRadius) {
  return shiftedHemkerGeometry(xLeft, xRight, -meshHeight/2.0, meshHeight/2.0, cylinderRadius);
}

MeshPtr MeshFactory::readMesh(string filePath, Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pToAdd)
{
  ifstream mshFile;
  mshFile.open(filePath.c_str());
  TEUCHOS_TEST_FOR_EXCEPTION(mshFile == NULL, std::invalid_argument, "Could not open msh file");
  string line;
  getline(mshFile, line);
  while (line != "$Nodes")
  {
    getline(mshFile, line);
  }
  int numNodes;
  mshFile >> numNodes;
  vector<vector<double> > vertices;
  int dummy;
  for (int i=0; i < numNodes; i++)
  {
    vector<double> vertex(2);
    mshFile >> dummy;
    mshFile >> vertex[0] >> vertex[1] >> dummy;
    vertices.push_back(vertex);
  }
  while (line != "$Elements")
  {
    getline(mshFile, line);
  }
  int numElems;
  mshFile >> numElems;
  int elemType;
  int numTags;
  vector< vector<unsigned> > elementIndices;
  for (int i=0; i < numElems; i++)
  {
    mshFile >> dummy >> elemType >> numTags;
    for (int j=0; j < numTags; j++)
      mshFile >> dummy;
    if (elemType == 2)
    {
      vector<unsigned> elemIndices(3);
      mshFile >> elemIndices[0] >> elemIndices[1] >> elemIndices[2];
      elemIndices[0]--;
      elemIndices[1]--;
      elemIndices[2]--;
      elementIndices.push_back(elemIndices);
    }
    if (elemType == 4)
    {
      vector<unsigned> elemIndices(3);
      mshFile >> elemIndices[0] >> elemIndices[1] >> elemIndices[2];
      elemIndices[0]--;
      elemIndices[1]--;
      elemIndices[2]--;
      elementIndices.push_back(elemIndices);
    }
    else
    {
      getline(mshFile, line);
    }
  }
  mshFile.close();
  
  Teuchos::RCP<Mesh> mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, bilinearForm, H1Order, pToAdd) );
  return mesh;
}

MeshPtr MeshFactory::readTriangle(string filePath, Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pToAdd)
{
  ifstream nodeFile;
  ifstream eleFile;
  string nodeFileName = filePath+".node";
  string eleFileName = filePath+".ele";
  nodeFile.open(nodeFileName.c_str());
  eleFile.open(eleFileName.c_str());
  TEUCHOS_TEST_FOR_EXCEPTION(nodeFile == NULL, std::invalid_argument, "Could not open node file: "+nodeFileName);
  TEUCHOS_TEST_FOR_EXCEPTION(eleFile == NULL, std::invalid_argument, "Could not open ele file: "+eleFileName);
  // Read node file
  string line;
  int numNodes;
  nodeFile >> numNodes;
  getline(nodeFile, line);
  vector<vector<double> > vertices;
  int dummy;
  int spaceDim = 2;
  vector<double> pt(spaceDim);
  for (int i=0; i < numNodes; i++)
  {
    nodeFile >> dummy >> pt[0] >> pt[1];
    getline(nodeFile, line);
    vertices.push_back(pt);
  }
  nodeFile.close();
  // Read ele file
  int numElems;
  eleFile >> numElems;
  getline(eleFile, line);
  vector< vector<unsigned> > elementIndices;
  vector<unsigned> el(3);
  for (int i=0; i < numElems; i++)
  {
    eleFile >> dummy >> el[0] >> el[1] >> el[2];
    el[0]--;
    el[1]--;
    el[2]--;
    elementIndices.push_back(el);
  }
  eleFile.close();
  
  Teuchos::RCP<Mesh> mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, bilinearForm, H1Order, pToAdd) );
  return mesh;
}

MeshPtr MeshFactory::buildQuadMesh(const FieldContainer<double> &quadBoundaryPoints,
                                   int horizontalElements, int verticalElements,
                                   Teuchos::RCP< BilinearForm > bilinearForm,
                                   int H1Order, int pTest, bool triangulate, bool useConformingTraces,
                                   map<int,int> trialOrderEnhancements,
                                   map<int,int> testOrderEnhancements) {
  //  if (triangulate) cout << "Mesh: Triangulating\n" << endl;
  int pToAddToTest = pTest - H1Order;
  int spaceDim = 2;
  // rectBoundaryPoints dimensions: (4,2) -- and should be in counterclockwise order
  
  vector<vector<double> > vertices;
  vector< vector<unsigned> > allElementVertices;
  
  TEUCHOS_TEST_FOR_EXCEPTION( ( quadBoundaryPoints.dimension(0) != 4 ) || ( quadBoundaryPoints.dimension(1) != 2 ),
                             std::invalid_argument,
                             "quadBoundaryPoints should be dimensions (4,2), points in ccw order.");
  
  int numElements = horizontalElements * verticalElements;
  if (triangulate) numElements *= 2;
  int numDimensions = 2;
  
  double southWest_x = quadBoundaryPoints(0,0),
  southWest_y = quadBoundaryPoints(0,1),
  southEast_x = quadBoundaryPoints(1,0),
  southEast_y = quadBoundaryPoints(1,1),
  northEast_x = quadBoundaryPoints(2,0),
  northEast_y = quadBoundaryPoints(2,1),
  northWest_x = quadBoundaryPoints(3,0),
  northWest_y = quadBoundaryPoints(3,1);
  
  double elemWidth = (southEast_x - southWest_x) / horizontalElements;
  double elemHeight = (northWest_y - southWest_y) / verticalElements;
  
  // set up vertices:
  // vertexIndices is for easy vertex lookup by (x,y) index for our Cartesian grid:
  vector< vector<int> > vertexIndices(horizontalElements+1, vector<int>(verticalElements+1));
  for (int i=0; i<=horizontalElements; i++) {
    for (int j=0; j<=verticalElements; j++) {
      vertexIndices[i][j] = vertices.size();
      vector<double> vertex(spaceDim);
      vertex[0] = southWest_x + elemWidth*i;
      vertex[1] = southWest_y + elemHeight*j;
      vertices.push_back(vertex);
      //      cout << "Mesh: vertex " << vertices.size() - 1 << ": (" << vertex(0) << "," << vertex(1) << ")\n";
    }
  }
  
  if ( ! triangulate ) {
    int SOUTH = 0, EAST = 1, NORTH = 2, WEST = 3;
    for (int i=0; i<horizontalElements; i++) {
      for (int j=0; j<verticalElements; j++) {
        vector<unsigned> elemVertices;
        elemVertices.push_back(vertexIndices[i][j]);
        elemVertices.push_back(vertexIndices[i+1][j]);
        elemVertices.push_back(vertexIndices[i+1][j+1]);
        elemVertices.push_back(vertexIndices[i][j+1]);
        allElementVertices.push_back(elemVertices);
      }
    }
  } else {
    int SIDE1 = 0, SIDE2 = 1, SIDE3 = 2;
    for (int i=0; i<horizontalElements; i++) {
      for (int j=0; j<verticalElements; j++) {
        // TEST CODE FOR LESZEK: match Leszek's meshes by setting diagonalUp = true here...
        //        bool diagonalUp = true;
        bool diagonalUp = (i%2 == j%2); // criss-cross pattern
        
        vector<unsigned> elemVertices1, elemVertices2;
        if (diagonalUp) {
          // elem1 is SE of quad, elem2 is NW
          elemVertices1.push_back(vertexIndices[i][j]);     // SIDE1 is SOUTH side of quad
          elemVertices1.push_back(vertexIndices[i+1][j]);   // SIDE2 is EAST
          elemVertices1.push_back(vertexIndices[i+1][j+1]); // SIDE3 is diagonal
          elemVertices2.push_back(vertexIndices[i][j+1]);   // SIDE1 is WEST
          elemVertices2.push_back(vertexIndices[i][j]);     // SIDE2 is diagonal
          elemVertices2.push_back(vertexIndices[i+1][j+1]); // SIDE3 is NORTH
        } else {
          // elem1 is SW of quad, elem2 is NE
          elemVertices1.push_back(vertexIndices[i][j]);     // SIDE1 is SOUTH side of quad
          elemVertices1.push_back(vertexIndices[i+1][j]);   // SIDE2 is diagonal
          elemVertices1.push_back(vertexIndices[i][j+1]);   // SIDE3 is WEST
          elemVertices2.push_back(vertexIndices[i][j+1]);   // SIDE1 is diagonal
          elemVertices2.push_back(vertexIndices[i+1][j]);   // SIDE2 is EAST
          elemVertices2.push_back(vertexIndices[i+1][j+1]); // SIDE3 is NORTH
        }
        
        allElementVertices.push_back(elemVertices1);
        allElementVertices.push_back(elemVertices2);
      }
    }
  }
  return Teuchos::rcp( new Mesh(vertices,allElementVertices,bilinearForm,H1Order,pToAddToTest,useConformingTraces,
                                trialOrderEnhancements,testOrderEnhancements));
}

Teuchos::RCP<Mesh> MeshFactory::buildQuadMeshHybrid(const FieldContainer<double> &quadBoundaryPoints,
                                             int horizontalElements, int verticalElements,
                                             Teuchos::RCP< BilinearForm > bilinearForm,
                                             int H1Order, int pTest, bool useConformingTraces) {
  int pToAddToTest = pTest - H1Order;
  int spaceDim = 2;
  // rectBoundaryPoints dimensions: (4,2) -- and should be in counterclockwise order
  
  vector<vector<double> > vertices;
  vector< vector<unsigned> > allElementVertices;
  
  TEUCHOS_TEST_FOR_EXCEPTION( ( quadBoundaryPoints.dimension(0) != 4 ) || ( quadBoundaryPoints.dimension(1) != 2 ),
                             std::invalid_argument,
                             "quadBoundaryPoints should be dimensions (4,2), points in ccw order.");
  
  int numDimensions = 2;
  
  double southWest_x = quadBoundaryPoints(0,0),
  southWest_y = quadBoundaryPoints(0,1),
  southEast_x = quadBoundaryPoints(1,0),
  southEast_y = quadBoundaryPoints(1,1),
  northEast_x = quadBoundaryPoints(2,0),
  northEast_y = quadBoundaryPoints(2,1),
  northWest_x = quadBoundaryPoints(3,0),
  northWest_y = quadBoundaryPoints(3,1);
  
  double elemWidth = (southEast_x - southWest_x) / horizontalElements;
  double elemHeight = (northWest_y - southWest_y) / verticalElements;
  
  int cellID = 0;
  
  // set up vertices:
  // vertexIndices is for easy vertex lookup by (x,y) index for our Cartesian grid:
  vector< vector<int> > vertexIndices(horizontalElements+1, vector<int>(verticalElements+1));
  for (int i=0; i<=horizontalElements; i++) {
    for (int j=0; j<=verticalElements; j++) {
      vertexIndices[i][j] = vertices.size();
      vector<double> vertex(spaceDim);
      vertex[0] = southWest_x + elemWidth*i;
      vertex[1] = southWest_y + elemHeight*j;
      vertices.push_back(vertex);
    }
  }
  
  int SOUTH = 0, EAST = 1, NORTH = 2, WEST = 3;
  int SIDE1 = 0, SIDE2 = 1, SIDE3 = 2;
  for (int i=0; i<horizontalElements; i++) {
    for (int j=0; j<verticalElements; j++) {
      bool triangulate = (i >= horizontalElements / 2); // triangles on right half of mesh
      if ( ! triangulate ) {
        vector<unsigned> elemVertices;
        elemVertices.push_back(vertexIndices[i][j]);
        elemVertices.push_back(vertexIndices[i+1][j]);
        elemVertices.push_back(vertexIndices[i+1][j+1]);
        elemVertices.push_back(vertexIndices[i][j+1]);
        allElementVertices.push_back(elemVertices);
      } else {
        vector<unsigned> elemVertices1, elemVertices2; // elem1 is SE of quad, elem2 is NW
        elemVertices1.push_back(vertexIndices[i][j]);     // SIDE1 is SOUTH side of quad
        elemVertices1.push_back(vertexIndices[i+1][j]);   // SIDE2 is EAST
        elemVertices1.push_back(vertexIndices[i+1][j+1]); // SIDE3 is diagonal
        elemVertices2.push_back(vertexIndices[i][j+1]);   // SIDE1 is WEST
        elemVertices2.push_back(vertexIndices[i][j]);     // SIDE2 is diagonal
        elemVertices2.push_back(vertexIndices[i+1][j+1]); // SIDE3 is NORTH
        
        allElementVertices.push_back(elemVertices1);
        allElementVertices.push_back(elemVertices2);
      }
    }
  }
  return Teuchos::rcp( new Mesh(vertices,allElementVertices,bilinearForm,H1Order,pToAddToTest,useConformingTraces));
}

void MeshFactory::quadMeshCellIDs(FieldContainer<int> &cellIDs, int horizontalElements, int verticalElements, bool useTriangles) {
  // populates cellIDs with either (h,v) or (h,v,2)
  // where h: horizontalElements (indexed by i, below)
  //       v: verticalElements   (indexed by j)
  //       2: triangles per quad (indexed by k)
  
  TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.dimension(0)!=horizontalElements,
                             std::invalid_argument,
                             "cellIDs should have dimensions: (horizontalElements, verticalElements) or (horizontalElements, verticalElements,2)");
  TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.dimension(1)!=verticalElements,
                             std::invalid_argument,
                             "cellIDs should have dimensions: (horizontalElements, verticalElements) or (horizontalElements, verticalElements,2)");
  if (useTriangles) {
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.dimension(2)!=2,
                               std::invalid_argument,
                               "cellIDs should have dimensions: (horizontalElements, verticalElements,2)");
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.rank() != 3,
                               std::invalid_argument,
                               "cellIDs should have dimensions: (horizontalElements, verticalElements,2)");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.rank() != 2,
                               std::invalid_argument,
                               "cellIDs should have dimensions: (horizontalElements, verticalElements)");
  }
  
  int cellID = 0;
  for (int i=0; i<horizontalElements; i++) {
    for (int j=0; j<verticalElements; j++) {
      if (useTriangles) {
        cellIDs(i,j,0) = cellID++;
        cellIDs(i,j,1) = cellID++;
      } else {
        cellIDs(i,j) = cellID++;
      }
    }
  }
}

MeshGeometryPtr MeshFactory::shiftedHemkerGeometry(double xLeft, double xRight, double yBottom, double yTop, double cylinderRadius) {
  double meshHeight = yTop - yBottom;
  double embeddedSquareSideLength = cylinderRadius+meshHeight/2;
  return shiftedHemkerGeometry(xLeft, xRight, yBottom, yTop, cylinderRadius, embeddedSquareSideLength);
}

MeshGeometryPtr MeshFactory::shiftedHemkerGeometry(double xLeft, double xRight, double yBottom, double yTop, double cylinderRadius, double embeddedSquareSideLength) {
  // first, set up an 8-element mesh, centered at the origin
  ParametricCurvePtr circle = ParametricCurve::circle(cylinderRadius, 0, 0);
  double meshHeight = yTop - yBottom;
  ParametricCurvePtr rect = parametricRect(embeddedSquareSideLength, embeddedSquareSideLength, 0, 0);

  int numPoints = 8; // 8 points on rect, 8 on circle
  int spaceDim = 2;
  vector< vector<double> > vertices;
  vector<double> innerVertex(spaceDim), outerVertex(spaceDim);
  FieldContainer<double> innerVertices(numPoints,spaceDim), outerVertices(numPoints,spaceDim); // these are just for easy debugging output

  vector<IndexType> innerVertexIndices;
  vector<IndexType> outerVertexIndices;

  double t = 0;
  for (int i=0; i<numPoints; i++) {
    circle->value(t, innerVertices(i,0), innerVertices(i,1));
    rect  ->value(t, outerVertices(i,0), outerVertices(i,1));
    circle->value(t, innerVertex[0], innerVertex[1]);
    rect  ->value(t, outerVertex[0], outerVertex[1]);
    innerVertexIndices.push_back(vertices.size());
    vertices.push_back(innerVertex);
    outerVertexIndices.push_back(vertices.size());
    vertices.push_back(outerVertex);
    t += 1.0 / numPoints;
  }

  //  cout << "innerVertices:\n" << innerVertices;
  //  cout << "outerVertices:\n" << outerVertices;

//  GnuPlotUtil::writeXYPoints("/tmp/innerVertices.dat", innerVertices);
//  GnuPlotUtil::writeXYPoints("/tmp/outerVertices.dat", outerVertices);

  vector< vector<IndexType> > elementVertices;

  int totalVertices = vertices.size();

  t = 0;
  map< pair<IndexType, IndexType>, ParametricCurvePtr > edgeToCurveMap;
  for (int i=0; i<numPoints; i++) { // numPoints = numElements
    vector<IndexType> vertexIndices;
    int innerIndex0 = (i * 2) % totalVertices;
    int innerIndex1 = ((i+1) * 2) % totalVertices;
    int outerIndex0 = (i * 2 + 1) % totalVertices;
    int outerIndex1 = ((i+1) * 2 + 1) % totalVertices;
    vertexIndices.push_back(innerIndex0);
    vertexIndices.push_back(outerIndex0);
    vertexIndices.push_back(outerIndex1);
    vertexIndices.push_back(innerIndex1);
    elementVertices.push_back(vertexIndices);

    //    cout << "innerIndex0: " << innerIndex0 << endl;
    //    cout << "innerIndex1: " << innerIndex1 << endl;
    //    cout << "outerIndex0: " << outerIndex0 << endl;
    //    cout << "outerIndex1: " << outerIndex1 << endl;

    pair<int, int> innerEdge = make_pair(innerIndex1, innerIndex0); // order matters
    edgeToCurveMap[innerEdge] = ParametricCurve::subCurve(circle, t+1.0/numPoints, t);
    t += 1.0/numPoints;
  }

  int boundaryVertexOffset = vertices.size();
  // make some new vertices, going counter-clockwise:
  ParametricCurvePtr meshRect = parametricRect(xRight-xLeft, meshHeight, 0.5*(xLeft+xRight), 0.5*(yBottom + yTop));
  vector<double> boundaryVertex(spaceDim);
  boundaryVertex[0] = xRight;
  boundaryVertex[1] = 0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[1] = meshHeight / 2.0;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[0] = embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[0] = 0;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[0] = -embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[0] = xLeft;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[1] = embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[1] = 0;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[1] = -embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[1] = -meshHeight / 2.0;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[0] = -embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[0] = 0;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[0] = embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[0] = xRight;
  vertices.push_back(boundaryVertex);
  
  boundaryVertex[1] = -embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  vector<IndexType> vertexIndices(4);
  vertexIndices[0] = outerVertexIndices[0];
  vertexIndices[1] = boundaryVertexOffset;
  vertexIndices[2] = boundaryVertexOffset + 1;
  vertexIndices[3] = outerVertexIndices[1];
  elementVertices.push_back(vertexIndices);

  // mesh NE corner
  vertexIndices[0] = outerVertexIndices[1];
  vertexIndices[1] = boundaryVertexOffset + 1;
  vertexIndices[2] = boundaryVertexOffset + 2;
  vertexIndices[3] = boundaryVertexOffset + 3;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = outerVertexIndices[2];
  vertexIndices[1] = outerVertexIndices[1];
  vertexIndices[2] = boundaryVertexOffset + 3;
  vertexIndices[3] = boundaryVertexOffset + 4;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = outerVertexIndices[3];
  vertexIndices[1] = outerVertexIndices[2];
  vertexIndices[2] = boundaryVertexOffset + 4;
  vertexIndices[3] = boundaryVertexOffset + 5;
  elementVertices.push_back(vertexIndices);

  // NW corner
  vertexIndices[0] = boundaryVertexOffset + 7;
  vertexIndices[1] = outerVertexIndices[3];
  vertexIndices[2] = boundaryVertexOffset + 5;
  vertexIndices[3] = boundaryVertexOffset + 6;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 8;
  vertexIndices[1] = outerVertexIndices[4];
  vertexIndices[2] = outerVertexIndices[3];
  vertexIndices[3] = boundaryVertexOffset + 7;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 9;
  vertexIndices[1] = outerVertexIndices[5];
  vertexIndices[2] = outerVertexIndices[4];
  vertexIndices[3] = boundaryVertexOffset + 8;
  elementVertices.push_back(vertexIndices);

  // SW corner
  vertexIndices[0] = boundaryVertexOffset + 10;
  vertexIndices[1] = boundaryVertexOffset + 11;
  vertexIndices[2] = outerVertexIndices[5];
  vertexIndices[3] = boundaryVertexOffset + 9;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 11;
  vertexIndices[1] = boundaryVertexOffset + 12;
  vertexIndices[2] = outerVertexIndices[6];
  vertexIndices[3] = outerVertexIndices[5];
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 12;
  vertexIndices[1] = boundaryVertexOffset + 13;
  vertexIndices[2] = outerVertexIndices[7];
  vertexIndices[3] = outerVertexIndices[6];
  elementVertices.push_back(vertexIndices);

  // SE corner
  vertexIndices[0] = boundaryVertexOffset + 13;
  vertexIndices[1] = boundaryVertexOffset + 14;
  vertexIndices[2] = boundaryVertexOffset + 15;
  vertexIndices[3] = outerVertexIndices[7];
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = outerVertexIndices[7];
  vertexIndices[1] = boundaryVertexOffset + 15;
  vertexIndices[2] = boundaryVertexOffset;
  vertexIndices[3] = outerVertexIndices[0];
  elementVertices.push_back(vertexIndices);

  return Teuchos::rcp( new MeshGeometry(vertices, elementVertices, edgeToCurveMap) );
}

MeshPtr MeshFactory::shiftedHemkerMesh(double xLeft, double xRight, double meshHeight, double cylinderRadius, // cylinder is centered in quad mesh.
                                BilinearFormPtr bilinearForm, int H1Order, int pToAddTest) {
  MeshGeometryPtr geometry = MeshFactory::shiftedHemkerGeometry(xLeft, xRight, meshHeight, cylinderRadius);
  MeshPtr mesh = Teuchos::rcp( new Mesh(geometry->vertices(), geometry->elementVertices(),
                                        bilinearForm, H1Order, pToAddTest) );

  map< pair<IndexType, IndexType>, ParametricCurvePtr > localEdgeToCurveMap = geometry->edgeToCurveMap();
  map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > globalEdgeToCurveMap(localEdgeToCurveMap.begin(),localEdgeToCurveMap.end());
  mesh->setEdgeToCurveMap(globalEdgeToCurveMap);
  return mesh;
}
