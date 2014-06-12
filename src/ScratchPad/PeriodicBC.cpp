//
//  PeriodicBC.cpp
//  Camellia
//
//  Created by Nate Roberts on 6/12/14.
//
//

#include "PeriodicBC.h"

#include "SpatialFilter.h"
#include "Function.h"

#include "PhysicalPointCache.h"

class TransformXFunction : public SimpleVectorFunction {
  double _xFrom, _xTo;
public:
  TransformXFunction(double xFrom, double xTo) {
    _xFrom = xFrom;
    _xTo = xTo;
  }
  vector<double> value(double x, double y) {
    vector<double> value(2);
    double tol=1e-14;
    if (abs(x-_xFrom)>tol) {
      cout << "x must match xFrom!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "x must match xFrom!")
    }
    value[0] = _xTo;
    value[1] = y;
    return value;
  }
  vector<double> value(double x, double y, double z) {
    vector<double> value(3);
    double tol=1e-14;
    if (abs(x-_xFrom)>tol) {
      cout << "x must match xFrom!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "x must match xFrom!")
    }
    value[0] = _xTo;
    value[1] = y;
    value[2] = z;
    return value;
  }
};

class TransformYFunction : public SimpleVectorFunction {
  double _yFrom, _yTo;
public:
  TransformYFunction(double yFrom, double yTo) {
    _yFrom = yFrom;
    _yTo = yTo;
  }
  vector<double> value(double x, double y) {
    vector<double> value(2);
    double tol=1e-14;
    if (abs(y-_yFrom)>tol) {
      cout << "y must match yFrom!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "y must match yFrom!")
    }
    value[0] = x;
    value[1] = _yTo;
    return value;
  }
  vector<double> value(double x, double y, double z) {
    vector<double> value(3);
    double tol=1e-14;
    if (abs(y-_yFrom)>tol) {
      cout << "y must match yFrom!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "y must match yFrom!")
    }
    value[0] = x;
    value[1] = _yTo;
    value[2] = z;
    return value;
  }
};

class TransformZFunction : public SimpleVectorFunction {
  double _zFrom, _zTo;
public:
  TransformZFunction(double zFrom, double zTo) {
    _zFrom = zFrom;
    _zTo = zTo;
  }
  vector<double> value(double x, double y, double z) {
    vector<double> value(3);
    double tol=1e-14;
    if (abs(z-_zFrom)>tol) {
      cout << "z must match zFrom!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "z must match zFrom!")
    }
    value[0] = x;
    value[1] = y;
    value[2] = _zTo;
    return value;
  }
};

PeriodicBC::PeriodicBC(SpatialFilterPtr pointFilter0, SpatialFilterPtr pointFilter1,
                       FunctionPtr transform0to1, FunctionPtr transform1to0) {
  if ((transform0to1->rank() != 1) || (transform1to0->rank() != 1)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "transform function must be vector-valued (rank 1).");
  }

  _pointFilter0 = pointFilter0;
  _pointFilter1 = pointFilter1;
  _transform0to1 = transform0to1;
  _transform1to0 = transform1to0;
}

vector<double> PeriodicBC::getMatchingPoint(const vector<double> &point, int whichSide) {
  FunctionPtr f;
  if (whichSide==0) {
    f = _transform0to1;
  } else if (whichSide==1) {
    f = _transform1to0;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "whichSide must be 0 or 1!");
  }
  
  int spaceDim = point.size();
  FieldContainer<double> value(1,1,spaceDim); // (C,P,D)
  FieldContainer<double> physPoint(1,1,spaceDim);
  
  Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
  for (int d=0; d<spaceDim; d++) {
    dummyCache->writablePhysicalCubaturePoints()(0,0,d) = point[d];
  }
  f->values(value,dummyCache);
  
  vector<double> transformedPoint;
  for (int d=0; d<spaceDim; d++) {
    transformedPoint.push_back(value(0,0,d));
  }
  
  return transformedPoint;
}

int PeriodicBC::getMatchingSide(const vector<double> &point) {
  // returns 0 if the point matches pointFilter0, 1 if it matches pointFilter1, -1 otherwise.
  if (point.size() == 1) {
    if (_pointFilter0->matchesPoint(point[0])) {
      return 0;
    } else if (_pointFilter1->matchesPoint(point[0])) {
      return 1;
    }
  } else if (point.size() == 2) {
    if (_pointFilter0->matchesPoint(point[0],point[1])) {
      return 0;
    } else if (_pointFilter1->matchesPoint(point[0],point[1])) {
      return 1;
    }
  } else if (point.size() == 3) {
    if (_pointFilter0->matchesPoint(point[0],point[1],point[2])) {
      return 0;
    } else if (_pointFilter1->matchesPoint(point[0],point[1],point[2])) {
      return 1;
    }
  } else {
    cout << "Unsupported point size.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported point size");
  }
  return -1;
}

PeriodicBCPtr PeriodicBC::periodicBC(SpatialFilterPtr pointFilter1, SpatialFilterPtr pointFilter2, FunctionPtr transform1to2, FunctionPtr transform2to1) {
  return Teuchos::rcp( new PeriodicBC(pointFilter1,pointFilter2,transform1to2,transform2to1) );
}

Teuchos::RCP<PeriodicBC> PeriodicBC::xIdentification(double x1, double x2) {
  SpatialFilterPtr x1Filter = SpatialFilter::matchingX(x1);
  SpatialFilterPtr x2Filter = SpatialFilter::matchingX(x2);
  FunctionPtr x1_to_x2 = Teuchos::rcp( new TransformXFunction(x1,x2) );
  FunctionPtr x2_to_x1 = Teuchos::rcp( new TransformXFunction(x2,x1) );
  return periodicBC(x1Filter, x2Filter, x1_to_x2, x2_to_x1);
}

Teuchos::RCP<PeriodicBC> PeriodicBC::yIdentification(double y1, double y2) {
  SpatialFilterPtr y1Filter = SpatialFilter::matchingY(y1);
  SpatialFilterPtr y2Filter = SpatialFilter::matchingY(y2);
  FunctionPtr y1_to_y2 = Teuchos::rcp( new TransformYFunction(y1,y2) );
  FunctionPtr y2_to_y1 = Teuchos::rcp( new TransformYFunction(y2,y1) );
  return periodicBC(y1Filter, y2Filter, y1_to_y2, y2_to_y1);
}

Teuchos::RCP<PeriodicBC> PeriodicBC::zIdentification(double z1, double z2) {
  SpatialFilterPtr z1Filter = SpatialFilter::matchingZ(z1);
  SpatialFilterPtr z2Filter = SpatialFilter::matchingZ(z2);
  FunctionPtr z1_to_z2 = Teuchos::rcp( new TransformZFunction(z1,z2) );
  FunctionPtr z2_to_z1 = Teuchos::rcp( new TransformZFunction(z2,z1) );
  return periodicBC(z1Filter, z2Filter, z1_to_z2, z2_to_z1);
}
