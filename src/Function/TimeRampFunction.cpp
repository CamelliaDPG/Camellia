//
//  TimeRampFunction.cpp
//  Camellia
//
//  Created by Nate Roberts on 8/26/15.
//
//

#include "TimeRampFunction.h"

using namespace Camellia;

TimeRampFunction::TimeRampFunction(double timeScale)
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