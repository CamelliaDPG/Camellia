//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  TimeRampFunction.cpp
//  Camellia
//
//  Created by Nate Roberts on 8/26/15.
//
//

#include "TimeRampFunction.h"
#include "SimpleFunction.h"

using namespace Camellia;

class TimeSteppingTimeRampFunction : public SimpleFunction<double>
{
  ParameterFunctionPtr _t;
  double _timeScale;
  double _startTime;
public:
  TimeSteppingTimeRampFunction(ParameterFunctionPtr t, double timeScale, double startTime=0)
  :
  SimpleFunction<double>("TimeSteppingTimeRampFunction", VARIES_IN_NO_DIMENSION)
  {
    _t = t;
    _timeScale = timeScale;
    _startTime = startTime;
  }
  
  virtual bool isZero() override
  {
    return false; // because the ParameterFunction can change, do not claim to be identically zero even if t == 0 right now.
  }
  
  virtual double value() override
  {
    const double t = _t->getConstantValue();
    if (t >= _timeScale)
    {
      return 1.0;
    }
    else
    {
      const double t_rel = (t > _startTime) ? t-_startTime : 0.0;
      return t_rel / (_timeScale-_startTime);
    }
  }
};

TimeRampFunction::TimeRampFunction(double timeScale)
: SimpleFunction<double>("TimeRampFunction", 0xff)
{
  _timeScale = timeScale;
}

double TimeRampFunction::value(double x, double t)
{
  if (t >= _timeScale)
  {
    return 1.0;
  }
  else
  {
    return t / _timeScale;
  }
}

double TimeRampFunction::value(double x, double y, double t)
{
  if (t >= _timeScale)
  {
    return 1.0;
  }
  else
  {
    return t / _timeScale;
  }
}

double TimeRampFunction::value(double x, double y, double z, double t)
{
  if (t >= _timeScale)
  {
    return 1.0;
  }
  else
  {
    return t / _timeScale;
  }
}

FunctionPtr TimeRampFunction::timeRamp(double timeScale)
{
  return Teuchos::rcp( new TimeRampFunction(timeScale) );
}

Teuchos::RCP< SimpleFunction<double> > TimeRampFunction::timeRamp(ParameterFunctionPtr t, double fullyOnTime, double startTime)
{
  return Teuchos::rcp( new TimeSteppingTimeRampFunction(t, fullyOnTime, startTime) );
}
