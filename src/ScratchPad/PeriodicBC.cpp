//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  PeriodicBC.cpp
//  Camellia
//
//  Created by Nate Roberts on 6/12/14.
//
//

#include "PeriodicBC.h"

#include "Function.h"
#include "PhysicalPointCache.h"
#include "SimpleVectorFunction.h"
#include "SpatialFilter.h"

using namespace Camellia;
 
class TransformCoordFunction : public SimpleVectorFunction<double>
{
  double _pointFrom, _pointTo;
  int _comp; // which component is being transformed/identified
public:
  TransformCoordFunction(double pointFrom, double pointTo, int comp)
  {
    _pointFrom = pointFrom;
    _pointTo = pointTo;
    _comp = comp;
  }
  
  vector<double> value(double x) override
  {
    vector<double> value(1);
    vector<double> point {x};
    
    this->getValue(value, point);
    return value;
  }
  
  vector<double> value(double x, double y) override
  {
    vector<double> value(2);
    vector<double> point {x,y};
    
    this->getValue(value, point);
    return value;
  }
  
  vector<double> value(double x, double y, double z) override
  {
    vector<double> value(3);
    vector<double> point {x,y,z};
    
    this->getValue(value, point);
    return value;
  }
  
  void getValue(vector<double> &vectorOut, const vector<double> &vectorIn) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(vectorOut.size() != vectorIn.size(), std::invalid_argument, "vectorOut does not match vectorIn in length");
    
    const double tol=1e-14;
    if (abs(vectorIn[_comp]-_pointFrom)>tol)
    {
      cout << "point[comp] must match pointFrom!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "point[comp] must match pointFrom!")
    }
    
    for (int d=0; d<vectorOut.size(); d++)
    {
      if (d == _comp)
      {
        vectorOut[d] = _pointTo;
      }
      else
      {
        vectorOut[d] = vectorIn[d];
      }
    }
  }
};

class TransformXFunction : public TransformCoordFunction
{
  double _pointFrom, _pointTo;
  int _comp; // which component is being transformed/identified
public:
  TransformXFunction(double xFrom, double xTo)
  :
  TransformCoordFunction(xFrom,xTo,0)
  {}
};

class TransformYFunction : public TransformCoordFunction
{
  double _pointFrom, _pointTo;
  int _comp; // which component is being transformed/identified
public:
  TransformYFunction(double yFrom, double yTo)
  :
  TransformCoordFunction(yFrom,yTo,1)
  {}
};

class TransformZFunction : public TransformCoordFunction
{
  double _pointFrom, _pointTo;
  int _comp; // which component is being transformed/identified
public:
  TransformZFunction(double zFrom, double zTo)
  :
  TransformCoordFunction(zFrom,zTo,2)
  {}
};

PeriodicBC::PeriodicBC(SpatialFilterPtr pointFilter0, SpatialFilterPtr pointFilter1,
                       TFunctionPtr<double> transform0to1, TFunctionPtr<double> transform1to0)
{
  if ((transform0to1->rank() != 1) || (transform1to0->rank() != 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "transform function must be vector-valued (rank 1).");
  }

  _pointFilter0 = pointFilter0;
  _pointFilter1 = pointFilter1;
  _transform0to1 = transform0to1;
  _transform1to0 = transform1to0;
}

vector<double> PeriodicBC::getMatchingPoint(const std::vector<double> &point, int whichSide)
{
  TFunctionPtr<double> f;
  if (whichSide==0)
  {
    f = _transform0to1;
  }
  else if (whichSide==1)
  {
    f = _transform1to0;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "whichSide must be 0 or 1!");
  }

  int spaceDim = point.size();
  Camellia::FieldContainer<double> value(1,1,spaceDim); // (C,P,D)
  Camellia::FieldContainer<double> physPoint(1,1,spaceDim);

  Teuchos::RCP<PhysicalPointCache> dummyCache = Teuchos::rcp( new PhysicalPointCache(physPoint) );
  for (int d=0; d<spaceDim; d++)
  {
    dummyCache->writablePhysicalCubaturePoints()(0,0,d) = point[d];
  }
  f->values(value,dummyCache);

  vector<double> transformedPoint;
  for (int d=0; d<spaceDim; d++)
  {
    transformedPoint.push_back(value(0,0,d));
  }

  return transformedPoint;
}

std::vector<int> PeriodicBC::getMatchingSides(const std::vector<double> &point)
{
  vector<int> matches; // contains 0 if point filter 0 matches, 1 if point filter 1 matches.
  if (_pointFilter0->matchesPoint(point))
  {
    matches.push_back(0);
  }
  if (_pointFilter1->matchesPoint(point))
  {
    matches.push_back(1);
  }
  return matches;
}

PeriodicBCPtr PeriodicBC::periodicBC(SpatialFilterPtr pointFilter1, SpatialFilterPtr pointFilter2, TFunctionPtr<double> transform1to2, TFunctionPtr<double> transform2to1)
{
  return Teuchos::rcp( new PeriodicBC(pointFilter1,pointFilter2,transform1to2,transform2to1) );
}

Teuchos::RCP<PeriodicBC> PeriodicBC::xIdentification(double x1, double x2)
{
  SpatialFilterPtr x1Filter = SpatialFilter::matchingX(x1);
  SpatialFilterPtr x2Filter = SpatialFilter::matchingX(x2);
  TFunctionPtr<double> x1_to_x2 = Teuchos::rcp( new TransformXFunction(x1,x2) );
  TFunctionPtr<double> x2_to_x1 = Teuchos::rcp( new TransformXFunction(x2,x1) );
  return periodicBC(x1Filter, x2Filter, x1_to_x2, x2_to_x1);
}

Teuchos::RCP<PeriodicBC> PeriodicBC::yIdentification(double y1, double y2)
{
  SpatialFilterPtr y1Filter = SpatialFilter::matchingY(y1);
  SpatialFilterPtr y2Filter = SpatialFilter::matchingY(y2);
  TFunctionPtr<double> y1_to_y2 = Teuchos::rcp( new TransformYFunction(y1,y2) );
  TFunctionPtr<double> y2_to_y1 = Teuchos::rcp( new TransformYFunction(y2,y1) );
  return periodicBC(y1Filter, y2Filter, y1_to_y2, y2_to_y1);
}

Teuchos::RCP<PeriodicBC> PeriodicBC::zIdentification(double z1, double z2)
{
  SpatialFilterPtr z1Filter = SpatialFilter::matchingZ(z1);
  SpatialFilterPtr z2Filter = SpatialFilter::matchingZ(z2);
  TFunctionPtr<double> z1_to_z2 = Teuchos::rcp( new TransformZFunction(z1,z2) );
  TFunctionPtr<double> z2_to_z1 = Teuchos::rcp( new TransformZFunction(z2,z1) );
  return periodicBC(z1Filter, z2Filter, z1_to_z2, z2_to_z1);
}
