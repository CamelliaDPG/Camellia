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

class ParametricRect : public ParametricCurve {
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
};

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

MeshGeometryPtr MeshFactory::hemkerGeometry(double meshWidth, double meshHeight, double cylinderRadius) {
  // later, we might want to do something more sophisticated, but for now, just an 8-element mesh, centered at origin
  ParametricCurvePtr circle = ParametricCurve::circle(cylinderRadius, 0, 0);
  ParametricCurvePtr rect = Teuchos::rcp( new ParametricRect(meshWidth, meshHeight, 0, 0) );
  
  int numPoints = 8; // 8 points on rect, 8 on circle
  int spaceDim = 2;
  vector< FieldContainer<double> > vertices;
  FieldContainer<double> innerVertex(spaceDim), outerVertex(spaceDim);
  FieldContainer<double> innerVertices(numPoints,spaceDim), outerVertices(numPoints,spaceDim); // these are just for easy debugging output
  double t = 0;
  for (int i=0; i<numPoints; i++) {
    circle->value(t, innerVertices(i,0), innerVertices(i,1));
    rect  ->value(t, outerVertices(i,0), outerVertices(i,1));
    circle->value(t, innerVertex(0), innerVertex(1));
    rect  ->value(t, outerVertex(0), outerVertex(1));
    vertices.push_back(innerVertex);
    //    cout << "vertex " << vertices.size() - 1 << ":\n" << vertices[vertices.size()-1];
    vertices.push_back(outerVertex);
    //    cout << "vertex " << vertices.size() - 1 << ":\n" << vertices[vertices.size()-1];
    t += 1.0 / numPoints;
  }
  
  //  cout << "innerVertices:\n" << innerVertices;
  //  cout << "outerVertices:\n" << outerVertices;
  
  GnuPlotUtil::writeXYPoints("/tmp/innerVertices.dat", innerVertices);
  GnuPlotUtil::writeXYPoints("/tmp/outerVertices.dat", outerVertices);
  
  vector< vector<int> > elementVertices;
  
  int totalVertices = vertices.size();
  
  t = 0;
  map< pair<int, int>, ParametricCurvePtr > edgeToCurveMap;
  for (int i=0; i<numPoints; i++) { // numPoints = numElements
    vector<int> vertexIndices;
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
  
  return Teuchos::rcp( new MeshGeometry(vertices, elementVertices, edgeToCurveMap) );
}

MeshPtr MeshFactory::hemkerMesh(double meshWidth, double meshHeight, double cylinderRadius, // cylinder is centered in quad mesh.
                                BilinearFormPtr bilinearForm, int H1Order, int pToAddTest) {
  MeshGeometryPtr geometry = MeshFactory::hemkerGeometry(meshWidth, meshHeight, cylinderRadius);
  MeshPtr mesh = Teuchos::rcp( new Mesh(geometry->vertices(), geometry->elementVertices(),
                                        bilinearForm, H1Order, pToAddTest) );
  mesh->setEdgeToCurveMap(geometry->edgeToCurveMap());
  return mesh;
}