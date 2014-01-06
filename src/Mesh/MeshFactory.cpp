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
                              double width, double height, int horizontalElements, int verticalElements) {
  FieldContainer<double> quadPoints(4,2);
  quadPoints(0,0) = 0.0;
  quadPoints(0,1) = 0.0;
  quadPoints(1,0) = width;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = width;
  quadPoints(2,1) = height;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = height;

  int testOrder = pToAddTest + H1Order; // buildQuadMesh's interface asks for the order of the test space, not the delta p.  Better to be consistent in using the delta p going forward...

  return Mesh::buildQuadMesh(quadPoints, horizontalElements, verticalElements, bf, H1Order, testOrder);
}

MeshGeometryPtr MeshFactory::shiftedHemkerGeometry(double xLeft, double xRight, double meshHeight, double cylinderRadius) {
  return shiftedHemkerGeometry(xLeft, xRight, -meshHeight/2.0, meshHeight/2.0, cylinderRadius);
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

  vector<unsigned> innerVertexIndices;
  vector<unsigned> outerVertexIndices;

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

  vector< vector<unsigned> > elementVertices;

  int totalVertices = vertices.size();

  t = 0;
  map< pair<int, int>, ParametricCurvePtr > edgeToCurveMap;
  for (int i=0; i<numPoints; i++) { // numPoints = numElements
    vector<unsigned> vertexIndices;
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

  vector<unsigned> vertexIndices(4);
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
  mesh->setEdgeToCurveMap(geometry->edgeToCurveMap());
  return mesh;
}
