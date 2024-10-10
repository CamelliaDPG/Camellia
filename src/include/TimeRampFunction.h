// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  TimeRampFunction.h
//  Camellia
//
//  Created by Nate Roberts on 8/26/15.
//
//

#ifndef Camellia_TimeRampFunction_h
#define Camellia_TimeRampFunction_h

#include "ParameterFunction.h"
#include "SimpleFunction.h"

namespace Camellia {
  class TimeRampFunction : public SimpleFunction<double>
  {
    double _timeScale;
  public:
    TimeRampFunction(double timeScale);
    
    using SimpleFunction<double>::value;
    double value(double x, double t);
    double value(double x, double y, double t);
    double value(double x, double y, double z, double t);
    
    // use this time ramp for space-time formulations
    static FunctionPtr timeRamp(double timeScale=1.0);
    
    // use this one for time-stepping formulations; formulation is responsible keeping ParameterFunction up to date.
    static Teuchos::RCP< SimpleFunction<double> > timeRamp(ParameterFunctionPtr t, double fullyOnTime=1.0, double startTime=0.0);
  };
}

#endif
