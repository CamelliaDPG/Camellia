//
//  TimeRampFunction.h
//  Camellia
//
//  Created by Nate Roberts on 8/26/15.
//
//

#ifndef Camellia_TimeRampFunction_h
#define Camellia_TimeRampFunction_h

#include "SimpleFunction.h"

namespace Camellia {
  class TimeRampFunction : public SimpleFunction<double>
  {
    double _timeScale;
  public:
    TimeRampFunction(double timeScale);
    double value(double x, double t);
    double value(double x, double y, double t);
    double value(double x, double y, double z, double t);
    
    static FunctionPtr timeRamp(double timeScale=1.0);
  };
}

#endif
