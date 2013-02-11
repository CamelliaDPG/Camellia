//
//  ParametricCurve.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/15/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>
#include "ParametricCurve.h"

#include "Shards_CellTopology.hpp"
#include "Function.h"

#include "VarFactory.h"
#include "IP.h"

#include "Projector.h"

static const double PI  = 3.141592653589793238462;

//  void mapRefCellPointsToParameterSpace(FieldContainer<double> &refPoints);

static void CHECK_FUNCTION_ONLY_DEPENDS_ON_1D_SPACE(FunctionPtr fxn) {
  try {
    Function::evaluate(fxn, 0);
  } catch (...) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Function threw an exception when evaluation at x=0 was attempted.  This can happen if your function depends on things other than 1D point values, e.g. if your function depends on a basis that requires points in reference space.");
  }
}

double ParametricFunction::remapForSubCurve(double t) {
  // want to map (0,1) to (_t0,_t1)
  return _t0 + t * (_t1 - _t0);
}

void ParametricFunction::setArgumentMap() {
  FunctionPtr t = Teuchos::rcp( new Xn(1) );
  _argMap = _t0 + t * (_t1 - _t0);
}

ParametricFunction::ParametricFunction(FunctionPtr fxn, double t0, double t1, int derivativeOrder) : Function(fxn->rank()) {
  CHECK_FUNCTION_ONLY_DEPENDS_ON_1D_SPACE(fxn);
  _underlyingFxn = fxn;
  _t0 = t0;
  _t1 = t1;
  _derivativeOrder = derivativeOrder;
  setArgumentMap();
}
ParametricFunction::ParametricFunction(FunctionPtr fxn) : Function(fxn->rank()) {
  CHECK_FUNCTION_ONLY_DEPENDS_ON_1D_SPACE(fxn);
  _underlyingFxn = fxn;
  _t0 = 0;
  _t1 = 1;
  _derivativeOrder = 0;
  setArgumentMap();
}
void ParametricFunction::value(double t, double &x) {
  t = remapForSubCurve(t);
  static FieldContainer<double> onePoint(1,1,1);
  static Teuchos::RCP< PhysicalPointCache > onePointCache = Teuchos::rcp( new PhysicalPointCache(onePoint) );
  static FieldContainer<double> oneValue(1,1);
  onePointCache->writablePhysicalCubaturePoints()(0,0,0) = t;
  _underlyingFxn->values(oneValue, onePointCache);
  x = oneValue[0];
}
void ParametricFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  FieldContainer<double> parametricPoints = basisCache->computeParametricPoints();
  BasisCachePtr parametricCache = Teuchos::rcp( new PhysicalPointCache(parametricPoints) );
  int numParametricCells = 1;
  int numPoints = parametricPoints.dimension(1);
  FieldContainer<double> mappedPoints(numParametricCells,numPoints);
  _argMap->values(mappedPoints, parametricCache);
  mappedPoints.resize(numParametricCells,numPoints,1); // 1: spaceDim
  parametricCache = Teuchos::rcp(new PhysicalPointCache(mappedPoints));
  Teuchos::Array<int> dimensions;
  values.dimensions(dimensions);
  dimensions[0] = numParametricCells;
  FieldContainer<double> parametricValues(dimensions);
  _underlyingFxn->values(parametricValues, parametricCache);
  
  // parametricValues has dimensions (C,P) == (1,P)
  int numCells = values.dimension(0);
  typedef FunctionSpaceTools fst;
  if (_derivativeOrder > 0) {
    // HGRADtransformGRAD expects (F,P,D) for input, which here can be understood as (1,P,1)--one field
    parametricValues.resize(1,numPoints,1);
    
    // HGRADtransformGRAD outputs to (C,F,P,D), and values has shape (C,P,D), so we should reshape it
    values.resize(numCells,1,numPoints,1);
    
    FieldContainer<double> jacobianInverse = basisCache->getJacobianInv();
    
    // modify the jacobianInverse to account for the fact that we're on [0,1], not [-1,1]
    // basisCache's transformation goes from [-1,1] to [x0,x1]
    // F(xi)  = xi * (x1-x0) / 2 + (x1+x0) / 2
    // F'(xi) = (x1-x0) / 2 is the jacobian.
    // Our G(t) = t * (x1-x0) + x0
    // G'(t) = x1-x0
    // so the jacobian is doubled, and the inverse jacobian is halved.
    for (int i=0; i<jacobianInverse.size(); i++) {
      jacobianInverse[i] /= 2.0;
    }
    
    for (int i=0; i<_derivativeOrder; i++) {
      // apply "Piola" transform to values
      fst::HGRADtransformGRAD<double>(values, jacobianInverse, parametricValues);
    }
    values.resize(numCells,numPoints); // in 1D, Camellia Function "gradients" are scalar-valued (different from Intrepid's take on it).
  } else {
    // HGRADtransformVALUE outputs to (C,F,P), and values has shape (C,P), so we should reshape it
    values.resize(numCells,1,numPoints);
    fst::HGRADtransformVALUE<double>(values, parametricValues);
    values.resize(numCells,numPoints);
  }
}

FunctionPtr ParametricFunction::dx() { // really dt
  return dt();
}

ParametricFunctionPtr ParametricFunction::dt() {
  double tScale = _t1 - _t0;
  return Teuchos::rcp( new ParametricFunction(tScale * _underlyingFxn->dx(),_t0,_t1,_derivativeOrder+1) );
}

ParametricFunctionPtr ParametricFunction::parametricFunction(FunctionPtr fxn, double t0, double t1) {
  if (!fxn.get()) {
    return Teuchos::rcp((ParametricFunction*)NULL);
  }
  ParametricFunctionPtr wholeFunction = Teuchos::rcp( new ParametricFunction(fxn) );
  return wholeFunction->subFunction(t0, t1);
}

ParametricFunctionPtr ParametricFunction::subFunction(double t0, double t1) {
  double subcurve_t0 = this->remapForSubCurve(t0);
  double subcurve_t1 = this->remapForSubCurve(t1);
  return Teuchos::rcp( new ParametricFunction(_underlyingFxn,subcurve_t0,subcurve_t1) );
}

class ParametricBubble : public ParametricCurve {
  ParametricCurvePtr _edgeCurve;
  ParametricCurvePtr _edgeLine;
  double _x0, _y0, _x1, _y1;
  // support for dt():
  bool _isDerivative;
  double _xDiff, _yDiff;
  ParametricBubble(ParametricCurvePtr edgeCurve_dt, double xDiff, double yDiff) {
    _isDerivative = true;
    _edgeCurve = edgeCurve_dt;
    _xDiff = xDiff;
    _yDiff = yDiff;
  }
public:
  ParametricBubble(ParametricCurvePtr edgeCurve) {
    _edgeCurve = edgeCurve;
    _edgeCurve->value(0, _x0,_y0);
    _edgeCurve->value(1, _x1,_y1);
    _edgeLine = ParametricCurve::line(_x0, _y0, _x1, _y1);
    _isDerivative = false;
  }
  void value(double t, double &x, double &y) {
    _edgeCurve->value(t, x,y);
    if (! _isDerivative ) {
      x -= _x0*(1.0-t) + _x1*t;
      y -= _y0*(1.0-t) + _y1*t;
    } else {
      x -= _xDiff;
      y -= _yDiff;
    }
  }
  ParametricCurvePtr dt() {
    return Teuchos::rcp( new ParametricBubble(_edgeCurve->dt(),_x1-_x0, _y1-_y0) );
  }
  FunctionPtr x() {
    if (!_isDerivative) {
      return _edgeCurve->x() - _edgeLine->x();
    } else {
      return _edgeCurve->x() - Function::constant(_xDiff);
    }
  }
  FunctionPtr y() {
    if (!_isDerivative) {
      return _edgeCurve->y() - _edgeLine->y();
    } else {
      return _edgeCurve->y() - Function::constant(_yDiff);
    }
  }
};

// TODO: consider changing this so that it's a subclass of ParametricFunction instead (and 1D)
// TODO: come up with a way to compute derivatives of this...
class ParametricUnion : public ParametricCurve {
  vector< ParametricCurvePtr > _curves;
  vector<double> _cutPoints;
  
  int matchingCurve(double t) {
    for (int i=0; i<_curves.size(); i++) {
      if ((t >= _cutPoints[i]) && (t<=_cutPoints[i+1])) {
        return i;
      }
    }
    return -1;
  }
public:
  ParametricUnion(const vector< ParametricCurvePtr > &curves, const vector<double> &weights) {
    _curves = curves;
    int numCurves = _curves.size();
    
    TEUCHOS_TEST_FOR_EXCEPTION(numCurves != weights.size(), std::invalid_argument, "must have same number of curves and weights");
    
    // make the weights add to 1.0
    double weightSum = 0;
    for (int i=0; i<numCurves; i++) {
      weightSum += weights[i];
    }
    _cutPoints.push_back(0);
    //    cout << "_cutPoints: ";
    //    cout << _cutPoints[0] << " ";
    for (int i=0; i<numCurves; i++) {
      _cutPoints.push_back(_cutPoints[i] + weights[i] / weightSum);
      //      cout << _cutPoints[i+1] << " ";
    }
  }
  void value(double t, double &x, double &y) {
    int curveIndex = matchingCurve(t);
    // map t so that from curve's pov it spans (0,1)
    double curve_t0 = _cutPoints[curveIndex];
    double curve_t1 = _cutPoints[curveIndex+1];
    double curve_t = (t - curve_t0) / (curve_t1 - curve_t0);
    _curves[curveIndex]->value(curve_t, x,y);
  }
};

ParametricCurve::ParametricCurve(ParametricFunctionPtr xFxn_x_as_t, ParametricFunctionPtr yFxn_x_as_t, ParametricFunctionPtr zFxn_x_as_t) : Function(1) {
  _xFxn = xFxn_x_as_t;
  _yFxn = yFxn_x_as_t;
  _zFxn = zFxn_x_as_t;
}

ParametricCurve::ParametricCurve() : Function(1) {
//  cout << "ParametricCurve().\n";
}

//ParametricCurve::ParametricCurve(ParametricCurvePtr fxn, double t0, double t1) : Function(1) {
//  _xFxn = fxn->xPart();
//  _yFxn = fxn->yPart();
//  _zFxn = fxn->zPart();
//}

//FunctionPtr ParametricCurve::argumentMap() {
//  FunctionPtr t = Teuchos::rcp( new Xn(1) );
//  FunctionPtr argMap = (1-t) * _t0 + t * _t1;
//  return argMap;
//}

double ParametricCurve::linearLength() { // length of the interpolating line
  double x0,y0,x1,y1;
  this->value(0, x0,y0);
  this->value(1, x1,y1);
  return sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
}

ParametricCurvePtr ParametricCurve::interpolatingLine() {
  // bubble + interpolatingLine = edgeCurve
  double x0,y0,x1,y1;
  this->value(0, x0,y0);
  this->value(1, x1,y1);
  return line(x0, y0, x1, y1);
}

void ParametricCurve::projectionBasedInterpolant(FieldContainer<double> &basisCoefficients, BasisPtr basis1D, int component,
                                                 double lengthScale, bool useH1) {
  
  ParametricCurvePtr thisPtr = Teuchos::rcp(this,false);
  ParametricCurvePtr bubble = ParametricCurve::bubble(thisPtr);
  ParametricCurvePtr line = this->interpolatingLine();
  IPPtr ip_H1 = Teuchos::rcp( new IP );
  // we assume that basis is a vector HGRAD basis
  VarFactory vf;
  VarPtr v = vf.testVar("v", HGRAD);
  ip_H1->addTerm(v);
  if (useH1) { // otherwise, stick with L2
    ip_H1->addTerm(v->dx());
  }
  
  //  double x0,y0,x1,y1;
  //  line->value(0, x0,y0);
  //  line->value(1, x1,y1);
  
  int basisDegree = basis1D->getDegree();
  int cubatureDegree = max(basisDegree*2,15);
  BasisCachePtr basisCache = BasisCache::basisCache1D(0, lengthScale, cubatureDegree);
  
  // determine indices for the vertices (we want to project onto the space spanned by the basis \ {vertex nodal functions})
  set<int> vertexNodeFieldIndices;
  int vertexDim = 0;
  int numVertices = 2;
  int spaceDim = basisCache->getSpaceDim();
  for (int vertexIndex=0; vertexIndex<numVertices; vertexIndex++) {
    for (int comp=0; comp<spaceDim; comp++) {
      int vertexNodeFieldIndex = basis1D->getDofOrdinal(vertexDim, vertexIndex, comp);
      vertexNodeFieldIndices.insert(vertexNodeFieldIndex);
    }
  }
  // project, skipping vertexNodeFieldIndices:
  FunctionPtr bubbleComponent, lineComponent;
  if (component==0){
    bubbleComponent = bubble->x();
    lineComponent = line->x();
  } else if (component==1) {
    bubbleComponent = bubble->y();
    lineComponent = line->y();
  } else if (component==2) {
    bubbleComponent = bubble->z();
    lineComponent = line->z();
  }
  Projector::projectFunctionOntoBasis(basisCoefficients, bubbleComponent, basis1D, basisCache, ip_H1, v, vertexNodeFieldIndices);
  
  // the line should live in the space spanned by basis.  It would be a bit cheaper to solve a system
  // to interpolate pointwise, but since it's easy to code, we'll just do a projection.  Since the
  // exact function we're after is in the space, mathematically it amounts to the same thing.
  FieldContainer<double> linearBasisCoefficients;
  Projector::projectFunctionOntoBasis(linearBasisCoefficients, lineComponent, basis1D, basisCache, ip_H1, v);
  
  //  cout << "linearBasisCoefficients:\n" << linearBasisCoefficients;
  //  cout << "basisCoefficients, before sum:\n" << basisCoefficients;
  
  // add the two sets of basis coefficients together
  for (int i=0; i<linearBasisCoefficients.size(); i++) {
    basisCoefficients[i] += linearBasisCoefficients[i];
  }
  //  cout << "basisCoefficients, after sum:\n" << basisCoefficients;
  basisCoefficients.resize(basis1D->getCardinality()); // get rid of dummy numCells dimension
  //  cout << "basisCoefficients, after resize:\n" << basisCoefficients;
  
}

void ParametricCurve::value(double t, double &x) {
  _xFxn->value(t,x);
}

void ParametricCurve::value(double t, double &x, double &y) {
  _xFxn->value(t,x);
  _yFxn->value(t,y);
}

void ParametricCurve::value(double t, double &x, double &y, double &z) {
  _xFxn->value(t,x);
  _yFxn->value(t,y);
  _zFxn->value(t,z);
}

void ParametricCurve::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0); // likely to be 1--in any case, we're the same on each cell
  int numPoints = values.dimension(1);
  int spaceDim = values.dimension(2);
  if (_xFxn.get()) { // then this curve is defined by some Functions of x (not by overriding the value() methods)
    FieldContainer<double> xValues(1,numPoints);
    _xFxn->values(xValues, basisCache);
    vector< FieldContainer<double>* > valuesFCs;
    valuesFCs.push_back(&xValues);
    FieldContainer<double> yValues, zValues;
    if (_yFxn.get()) {
      yValues.resize(1,numPoints);
      _yFxn->values(yValues, basisCache);
      valuesFCs.push_back(&yValues);
    } else if (spaceDim >= 2) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim >= 2, but _yFxn undefined");
    }
    if (_zFxn.get()) {
      zValues.resize(1,numPoints);
      _zFxn->values(zValues, basisCache);
      valuesFCs.push_back(&zValues);
    } else if (spaceDim >= 3) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim >= 3, but _zFxn undefined");
    }
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        for (int d=0; d<spaceDim; d++) {
          values(cellIndex,ptIndex,d) = (*valuesFCs[d])(0,ptIndex);
        }
      }
    }
    return;
  }
  //  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
  //    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
  //      double t = parametricPoints(ptIndex,0);
  //      if (spaceDim==1) {
  //        value(t,values(cellIndex,ptIndex,0));
  //      } else if (spaceDim==2) {
  //        value(t,values(cellIndex,ptIndex,0),values(cellIndex,ptIndex,1));
  //      } else if (spaceDim==3) {
  //        value(t,values(cellIndex,ptIndex,0),values(cellIndex,ptIndex,1),values(cellIndex,ptIndex,2));
  //      } else {
  //        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported spaceDim");
  //      }
  //    }
  //  }
}

FunctionPtr ParametricCurve::x() {
  return _xFxn;
}

FunctionPtr ParametricCurve::y() {
  return _yFxn;
}

FunctionPtr ParametricCurve::z() {
  return _zFxn;
}

ParametricFunctionPtr ParametricCurve::xPart() {
  return _xFxn;
}

ParametricFunctionPtr ParametricCurve::yPart() {
  return _yFxn;
}

ParametricFunctionPtr ParametricCurve::zPart() {
  return _zFxn;
}

ParametricCurvePtr ParametricCurve::bubble(ParametricCurvePtr edgeCurve) {
  double x0,y0,x1,y1;
  edgeCurve->value(0, x0,y0);
  edgeCurve->value(1, x1,y1);
  ParametricCurvePtr edgeLine = ParametricCurve::line(x0, y0, x1, y1);

  return Teuchos::rcp( new ParametricBubble(edgeCurve) );
}

ParametricCurvePtr ParametricCurve::circle(double r, double x0, double y0) {
  FunctionPtr cos_2pi_t = Teuchos::rcp( new Cos_ax(2.0*PI) );
  FunctionPtr sin_2pi_t = Teuchos::rcp( new Sin_ax(2.0*PI) );
  FunctionPtr xFunction = r * cos_2pi_t + Function::constant(x0);
  FunctionPtr yFunction = r * sin_2pi_t + Function::constant(y0);
  
  return curve(xFunction,yFunction);
  
  //  return Teuchos::rcp( new ParametricCircle(r, x0, y0));
}

ParametricCurvePtr ParametricCurve::circularArc(double r, double x0, double y0, double theta0, double theta1) {
  ParametricCurvePtr circleFxn = circle(r, x0, y0);
  double t0 = theta0 / 2.0 * PI;
  double t1 = theta1 / 2.0 * PI;
  return subCurve(circleFxn, t0, t1);
}

ParametricCurvePtr ParametricCurve::curve(FunctionPtr xFxn_x_as_t, FunctionPtr yFxn_x_as_t, FunctionPtr zFxn_x_as_t) {
  ParametricFunctionPtr xParametric = ParametricFunction::parametricFunction(xFxn_x_as_t);
  ParametricFunctionPtr yParametric = ParametricFunction::parametricFunction(yFxn_x_as_t);
  ParametricFunctionPtr zParametric = ParametricFunction::parametricFunction(zFxn_x_as_t);
  
  return Teuchos::rcp( new ParametricCurve(xParametric,yParametric,zParametric) );
}

ParametricCurvePtr ParametricCurve::curveUnion(vector< ParametricCurvePtr > curves, vector<double> weights) {
  int numCurves = curves.size();
  if (weights.size()==0) {
    // default to even weighting
    for (int i=1; i<=numCurves; i++) {
      weights.push_back(1.0);
    }
  }
  return Teuchos::rcp( new ParametricUnion(curves, weights) );
}
//
//FunctionPtr ParametricCurve::dx() { // same as dt() (overrides Function::dx())
//  return dt();
//}

ParametricCurvePtr ParametricCurve::dt() { // the curve differentiated in t in each component.
  ParametricFunctionPtr dxdt, dydt, dzdt;
  
  if (_xFxn.get()) {
    dxdt = _xFxn->dt();
  }
  if (_yFxn.get()) {
    dydt = _yFxn->dt();
  }
  if (_zFxn.get()) {
    dzdt = _zFxn->dt();
  }
  return Teuchos::rcp( new ParametricCurve(dxdt,dydt,dzdt) );
}

ParametricCurvePtr ParametricCurve::polygon(vector< pair<double,double> > vertices, vector<double> weights) {
  int numVertices = vertices.size();
  if (weights.size()==0) {
    // default to weighting by length
    double x_prev = vertices[0].first;
    double y_prev = vertices[0].second;
    for (int i=1; i<=numVertices; i++) {
      double x = vertices[i%numVertices].first;  // modulus to sweep back to first vertex
      double y = vertices[i%numVertices].second;
      double d = sqrt( (x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev));
      weights.push_back(d);
      x_prev = x;
      y_prev = y;
    }
  }
  vector< ParametricCurvePtr > lines;
  for (int i=0; i<numVertices; i++) {
    double x0 = vertices[i].first;
    double y0 = vertices[i].second;
    double x1 = vertices[(i+1)%numVertices].first;
    double y1 = vertices[(i+1)%numVertices].second;
    lines.push_back(line(x0, y0, x1, y1));
  }
  return curveUnion(lines,weights);
}

ParametricCurvePtr ParametricCurve::line(double x0, double y0, double x1, double y1) {
  FunctionPtr t = Teuchos::rcp( new Xn(1) );
  FunctionPtr x0_f = Function::constant(x0);
  FunctionPtr y0_f = Function::constant(y0);
  FunctionPtr xFxn = (x1-x0) * t + x0_f;
  FunctionPtr yFxn = (y1-y0) * t + y0_f;
  
  return ParametricCurve::curve(xFxn,yFxn);
}
//
//void ParametricCurve::mapRefCellPointsToParameterSpace(FieldContainer<double> &refPoints) {
//  int numPoints = refPoints.dimension(0);
//  int spaceDim = refPoints.dimension(1);
//  if (spaceDim != 1) {
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refPoints must be of dimension 1");
//  }
//  for (int i=0; i<numPoints; i++) {
//    double x = refPoints(i,0);
//    double t = (x + 1) / 2;
//    refPoints(i,0) = t;
//  }
//}

std::vector< ParametricCurvePtr > ParametricCurve::referenceCellEdges(unsigned cellTopoKey) {
  if (cellTopoKey == shards::Quadrilateral<4>::key) {
    return referenceQuadEdges();
  } else if (cellTopoKey == shards::Triangle<3>::key) {
    return referenceTriangleEdges();
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled cellTopoKey");
  }
}

std::vector< ParametricCurvePtr > ParametricCurve::referenceQuadEdges() {
  std::vector< ParametricCurvePtr > edges;
  edges.push_back(line(-1,-1,1,-1));
  edges.push_back(line(1,-1,1,1));
  edges.push_back(line(1,1,-1,1));
  edges.push_back(line(-1,1,-1,-1));
  return edges;
}

std::vector< ParametricCurvePtr > ParametricCurve::referenceTriangleEdges() {
  std::vector< ParametricCurvePtr > edges;
  edges.push_back(line(0,0,1,0));
  edges.push_back(line(1,0,0,1));
  edges.push_back(line(0,1,0,0));
  return edges;
}

ParametricCurvePtr ParametricCurve::reverse(ParametricCurvePtr fxn) {
  return subCurve(fxn, 1, 0);
}

ParametricCurvePtr ParametricCurve::subCurve(ParametricCurvePtr fxn, double t0, double t1) {
  ParametricFunctionPtr x = fxn->xPart();
  ParametricFunctionPtr y = fxn->yPart();
  ParametricFunctionPtr z = fxn->zPart();
  
  if (x.get()) {
    x = x->subFunction(t0,t1);
  }
  if (y.get()) {
    y = y->subFunction(t0,t1);
  }
  if (z.get()) {
    z = z->subFunction(t0,t1);
  }
  return Teuchos::rcp( new ParametricCurve(x,y,z) );
}