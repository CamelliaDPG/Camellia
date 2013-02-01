//
//  ParametricSurface.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/28/13.
//
//

#include "ParametricSurface.h"
#include "IP.h"
#include "VarFactory.h"
#include "Projector.h"

class ParametricBubble : public ParametricCurve {
  ParametricCurvePtr _edgeCurve;
  double _x0, _y0, _x1, _y1;
public:
  ParametricBubble(ParametricCurvePtr edgeCurve) {
    _edgeCurve = edgeCurve;
    _edgeCurve->value(0, _x0,_y0);
    _edgeCurve->value(1, _x1,_y1);
  }
  void value(double t, double &x, double &y) {
    _edgeCurve->value(t, x,y);
    x -= _x0*(1.0-t) + _x1*t;
    y -= _y0*(1.0-t) + _y1*t;
  }
  static ParametricCurvePtr bubble(ParametricCurvePtr edgeCurve) {
    return Teuchos::rcp( new ParametricBubble(edgeCurve) );
  }
};

class LinearInterpolatingSurface : public ParametricSurface {
  vector< pair<double, double> > _vertices;
public:
  LinearInterpolatingSurface(const vector< ParametricCurvePtr > &curves) {
    for (int i=0; i<curves.size(); i++) {
      _vertices.push_back(make_pair(0,0));
      curves[i]->value(0, _vertices[i].first, _vertices[i].second);
    }
  }
  void value(double t1, double t2, double &x, double &y);
};

void LinearInterpolatingSurface::value(double t1, double t2, double &x, double &y) {
  if (_vertices.size() == 4) {
      x  = _vertices[0].first*(1-t1)*(1-t2) + _vertices[1].first*   t1 *(1-t2)
         + _vertices[2].first*   t1*    t2  + _vertices[3].first*(1-t1)*   t2;
      y  = _vertices[0].second*(1-t1)*(1-t2) + _vertices[1].second*   t1 *(1-t2)
         + _vertices[2].second*   t1*    t2  + _vertices[3].second*(1-t1)*   t2;
  } else if (_vertices.size() == 3) {
    // TODO: implement this
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads supported for now...");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads and (eventually) triangles supported...");
  }
}

class TransfiniteInterpolatingSurface : public ParametricSurface {
  vector< ParametricCurvePtr > _curves;
  vector< pair<double, double> > _vertices;
  bool _neglectVertices; // if true, then the value returned by value() is a "bubble" value...
public:
  TransfiniteInterpolatingSurface(const vector< ParametricCurvePtr > &curves) {
    _neglectVertices = false;
    _curves = curves;
    for (int i=0; i<curves.size(); i++) {
      _vertices.push_back(make_pair(0,0));
      _curves[i]->value(0, _vertices[i].first, _vertices[i].second);
    }
    if (_curves.size() != 4) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads supported for now...");
    }
    // we assume that the curves go CCW around the element; we flip the two opposite edges
    // so both sets of opposite edges run parallel to each other:
    _curves[2] = ParametricCurve::reverse(_curves[2]);
    _curves[3] = ParametricCurve::reverse(_curves[3]);
    
    // since we keep _vertices separately, can just store bubble functions in _curves
    _curves[0] = ParametricBubble::bubble(_curves[0]);
    _curves[1] = ParametricBubble::bubble(_curves[1]);
    _curves[2] = ParametricBubble::bubble(_curves[2]);
    _curves[3] = ParametricBubble::bubble(_curves[3]);
  }
  void setNeglectVertices(bool value) {
    _neglectVertices = value;
  }
  void value(double t1, double t2, double &x, double &y);
};

void ParametricSurface::basisWeightsForL2ProjectedInterpolant(FieldContainer<double> &basisCoefficients, BasisPtr basis,
                                                              MeshPtr mesh, int cellID) {
  vector< ParametricCurvePtr > curves = mesh->parametricEdgesForCell(cellID);
  Teuchos::RCP<TransfiniteInterpolatingSurface> exactSurface = Teuchos::rcp( new TransfiniteInterpolatingSurface(curves) );
  exactSurface->setNeglectVertices(true);
  IPPtr l2 = Teuchos::rcp( new IP );
  // we assume that basis is a vector HGRAD basis
  VarFactory vf;
  VarPtr v = vf.testVar("v", VECTOR_HGRAD);
  l2->addTerm(v);
  
  int maxTestDegree = mesh->getElement(cellID)->elementType()->testOrderPtr->maxBasisDegree();
  TEUCHOS_TEST_FOR_EXCEPTION(maxTestDegree < 1, std::invalid_argument, "Constant test spaces unsupported.");
  
  
  // since we'll be integrating pairs of geometry representation functions and these are of the same
  // order as test functions, we set up the basisCache just as we would for test space inner products...
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, true); // true: testVsTest
  
  // determine indices for the vertices (we want to project onto the space spanned by the basis \ {vertex nodal functions})
  set<int> vertexNodeFieldIndices;
  int vertexDim = 0;
  int numVertices = mesh->getElement(cellID)->elementType()->cellTopoPtr->getVertexCount();
  int spaceDim = basisCache->getSpaceDim();
  for (int vertexIndex=0; vertexIndex<numVertices; vertexIndex++) {
    for (int comp=0; comp<spaceDim; comp++) {
      int vertexNodeFieldIndex = basis->getDofOrdinal(vertexDim, vertexIndex, comp);
      vertexNodeFieldIndices.insert(vertexNodeFieldIndex);
    }
  }
  // project, skipping vertexNodeFieldIndices:
  Projector::projectFunctionOntoBasis(basisCoefficients, exactSurface, basis, basisCache, l2, v, vertexNodeFieldIndices);
  
  // determine basis weights for the linear interpolant:
  ParametricSurfacePtr linearInterpolant = ParametricSurface::linearInterpolant(curves);
  // this should live in the space spanned by basis.  It would be a bit cheaper to solve a system
  // to interpolate pointwise, but since it's easy to code, we'll just do a projection.  Since the
  // exact function we're after is in the space, mathematically it amounts to the same thing.
  FieldContainer<double> linearBasisCoefficients;
  Projector::projectFunctionOntoBasis(linearBasisCoefficients, linearInterpolant, basis, basisCache, l2, v);
  
//  cout << "basisCoefficients after projection:\n" << basisCoefficients;
//  cout << "linearBasisCoefficients:\n" << linearBasisCoefficients;
  
  // add the two sets of basis coefficients together
  for (int i=0; i<linearBasisCoefficients.size(); i++) {
    basisCoefficients[i] += linearBasisCoefficients[i];
  }
  basisCoefficients.resize(basis->getCardinality()); // get rid of dummy numCells dimension
}

void TransfiniteInterpolatingSurface::value(double t1, double t2, double &x, double &y) {
  if (_curves.size() == 4) {
    // t1 indexes curves 0 and 2, t2 1 and 3
    double x0, y0, x2, y2;
    _curves[0]->value(t1, x0,y0);
    _curves[2]->value(t1, x2,y2);
    double x1, y1, x3, y3;
    _curves[1]->value(t2, x1,y1);
    _curves[3]->value(t2, x3,y3);
    x = x0*(1-t2) + x1 * t1 + x2*t2 + x3*(1-t1);
    y = y0*(1-t2) + y1 * t1 + y2*t2 + y3*(1-t1);
    
    if (! _neglectVertices) {
      x += _vertices[0].first*(1-t1)*(1-t2) + _vertices[1].first*   t1 *(1-t2)
         + _vertices[2].first*   t1*    t2  + _vertices[3].first*(1-t1)*   t2;
      y += _vertices[0].second*(1-t1)*(1-t2) + _vertices[1].second*   t1 *(1-t2)
         + _vertices[2].second*   t1*    t2  + _vertices[3].second*(1-t1)*   t2;
    }
  } else if (_curves.size() == 3) {
    // TODO: implement this
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads supported for now...");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only quads and (eventually) triangles supported...");
  }
  
}

FieldContainer<double> & ParametricSurface::parametricQuadNodes() { // for CellTools cellWorkset argument
  static FieldContainer<double> quadNodes(1,4,2);
  static bool quadNodesSet = false;
  // there's probably a cleaner way to statically initialize this container,
  // but this setup should still do so exactly once
  if (!quadNodesSet) {
    quadNodes(0,0,0) = 0.0;
    quadNodes(0,0,1) = 0.0;
    quadNodes(0,1,0) = 1.0;
    quadNodes(0,0,1) = 0.0;
    quadNodes(0,2,0) = 1.0;
    quadNodes(0,2,1) = 1.0;
    quadNodes(0,3,0) = 0.0;
    quadNodes(0,3,1) = 1.0;
    quadNodesSet = true;
  }
  return quadNodes;
}

void ParametricSurface::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  const FieldContainer<double>* refPoints = &(basisCache->getRefCellPoints());
  int numPoints = refPoints->dimension(0);
  int spaceDim = refPoints->dimension(1);
  if (spaceDim != 2) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only 2D supported right now...");
  }
  FieldContainer<double> parametricPoints(numPoints,spaceDim); // map to (t1,t2) space
  int whichCell = 0;
  CellTools<double>::mapToPhysicalFrame(parametricPoints,*refPoints,
                                        ParametricSurface::parametricQuadNodes(),
                                        basisCache->cellTopology(),whichCell);
  // this is likely only to make sense, practically speaking, for a one-cell basisCache.
  // so we don't optimize the following to compute values on a single cell and copy to others,
  // although that would be relatively trivial
  int numCells = values.dimension(0);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    double x, y;
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double t1, t2;
      t1 = parametricPoints(ptIndex,0);
      t2 = parametricPoints(ptIndex,1);
      this->value(t1, t2, x, y);
      values(cellIndex,ptIndex,0) = x;
      values(cellIndex,ptIndex,1) = y;
    }
  }
}

ParametricSurfacePtr ParametricSurface::linearInterpolant(const vector< ParametricCurvePtr > &curves) {
  return Teuchos::rcp( new LinearInterpolatingSurface(curves) );
}

ParametricSurfacePtr ParametricSurface::transfiniteInterpolant(const vector< ParametricCurvePtr > &curves) {
  return Teuchos::rcp( new TransfiniteInterpolatingSurface(curves) );
}