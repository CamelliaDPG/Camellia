// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  SpatialFilter.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_SpatialFilter_h
#define Camellia_SpatialFilter_h

#include "TypeDefs.h"

#include "BasisCache.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"
#include "Camellia_FieldContainer.hpp"

namespace Camellia
{
class SpatialFilter
{
public:
  virtual bool matchesPoint(double x);
  virtual bool matchesPoint(double x, double y);
  virtual bool matchesPoint(double x, double y, double z);
  virtual bool matchesPoint(double x, double y, double z, double t);
  virtual bool matchesPoint(double x0, double x1, double x2, double x3, double x4);
  virtual bool matchesPoint(double x0, double x1, double x2, double x3, double x4, double x5);
  virtual bool matchesPoint(double x0, double x1, double x2, double x3, double x4, double x5, double x6);
  virtual bool matchesPoint(const vector<double>&point);
  virtual bool matchesPoints(Camellia::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache);
  
  virtual void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double>) = 0;
  
  virtual bool matchesSpatialSides(); // default: true
  virtual bool matchesTemporalSides(); // default: false

  static SpatialFilterPtr allSpace();
  static SpatialFilterPtr unionFilter(SpatialFilterPtr a, SpatialFilterPtr b);
  static SpatialFilterPtr intersectionFilter(SpatialFilterPtr a, SpatialFilterPtr b);
  
  //! returns a SpatialFilter that matches wherever the convective direction c (indicated by the input function) dotted with the outward normal is negative.
  static SpatialFilterPtr inflowFilter(std::vector<FunctionPtr> convectionComponents);

  static SpatialFilterPtr negatedFilter(SpatialFilterPtr filterToNegate);

  static SpatialFilterPtr matchingX(double x);
  static SpatialFilterPtr matchingY(double y);
  static SpatialFilterPtr matchingZ(double z);
  
  static SpatialFilterPtr matchingT(double t, bool matchSpatialSides = false); // matchesSpatialSides() is false; matchesTemporalSides() is true

  static SpatialFilterPtr matchingComponent(int comp, double value);
  
  static SpatialFilterPtr lessThanX(double x);
  static SpatialFilterPtr lessThanY(double y);
  static SpatialFilterPtr lessThanZ(double z);

  static SpatialFilterPtr greaterThanX(double x);
  static SpatialFilterPtr greaterThanY(double y);
  static SpatialFilterPtr greaterThanZ(double z);
  
  static SpatialFilterPtr lessThanOrEqualToX(double x);
  static SpatialFilterPtr lessThanOrEqualToY(double y);
  static SpatialFilterPtr lessThanOrEqualToZ(double z);
  
  static SpatialFilterPtr greaterThanOrEqualToX(double x);
  static SpatialFilterPtr greaterThanOrEqualToY(double y);
  static SpatialFilterPtr greaterThanOrEqualToZ(double z);

  virtual ~SpatialFilter() {}
};

class SpatialFilterUnfiltered : public SpatialFilter
{
public:
  using SpatialFilter::matchesPoints;
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double>) override;
  
  using SpatialFilter::matchesPoint;
  virtual bool matchesPoint(const vector<double> &point) override;

  virtual bool matchesPoint(double x) override;
  virtual bool matchesPoint(double x, double y) override;
  virtual bool matchesPoint(double x, double y, double z) override;
  virtual bool matchesPoint(double x, double y, double z, double t) override;
};

class SpatialFilterLogicalOr : public SpatialFilter
{
  SpatialFilterPtr _sf1, _sf2;
public:
  SpatialFilterLogicalOr(SpatialFilterPtr sf1, SpatialFilterPtr sf2);
  
  using SpatialFilter::matchesPoints;
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double>) override;
  bool matchesPoints(Camellia::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache)  override;

  //  bool matchesPoint( double x, double y ) {
  //    return _sf1->matchesPoint(x,y) || _sf2->matchesPoint(x,y);
  //  }
  using SpatialFilter::matchesPoint;
  virtual bool matchesPoint(double x) override;
  virtual bool matchesPoint(double x, double y) override;
  virtual bool matchesPoint(double x, double y, double z) override;
  virtual bool matchesPoint(double x, double y, double z, double t) override;
  
  virtual bool matchesSpatialSides() override;
  virtual bool matchesTemporalSides() override;
};

class SpatialFilterLogicalAnd : public SpatialFilter
{
  SpatialFilterPtr _sf1, _sf2;
public:
  SpatialFilterLogicalAnd(SpatialFilterPtr sf1, SpatialFilterPtr sf2);
  bool matchesPoints(Camellia::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) override;
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double>) override;

  //  bool matchesPoint( double x, double y ) {
  //    return _sf1->matchesPoint(x,y) || _sf2->matchesPoint(x,y);
  //  }
  using SpatialFilter::matchesPoint;
  virtual bool matchesPoint(double x) override;
  virtual bool matchesPoint(double x, double y) override;
  virtual bool matchesPoint(double x, double y, double z) override;
  virtual bool matchesPoint(double x, double y, double z, double t) override;
  
  virtual bool matchesSpatialSides() override;
  virtual bool matchesTemporalSides() override;
};

class NegatedSpatialFilter : public SpatialFilter
{
  SpatialFilterPtr _filterToNegate;
public:
  NegatedSpatialFilter(SpatialFilterPtr FilterToNegate);
  using SpatialFilter::matchesPoints;
  bool matchesPoints(Camellia::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) override;
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double>) override;

  using SpatialFilter::matchesPoint;
  virtual bool matchesPoint(double x) override;
  virtual bool matchesPoint(double x, double y) override;
  virtual bool matchesPoint(double x, double y, double z) override;
  virtual bool matchesPoint(double x, double y, double z, double t) override;
  
  // ! Returns the same result as _filterToNegate; the negation is understood to happen in space or time, not both
  virtual bool matchesSpatialSides() override;
  
  // ! Returns the same result as _filterToNegate; the negation is understood to happen in space or time, not both
  virtual bool matchesTemporalSides() override;
};

SpatialFilterPtr operator!(SpatialFilterPtr sf);
SpatialFilterPtr operator|(SpatialFilterPtr sf1, SpatialFilterPtr sf2);
SpatialFilterPtr operator&(SpatialFilterPtr sf1, SpatialFilterPtr sf2);
}

#endif
