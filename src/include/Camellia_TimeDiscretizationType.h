// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  Camellia_TimeDiscretizationType.h
//  Camellia
//
//  Created by Nate Roberts on 5/16/22.
//
//

#ifndef Camellia_TimeDiscretizationType_h
#define Camellia_TimeDiscretizationType_h

namespace Camellia
{
  enum TimeDiscretizationType
  {
    TIME_MARCHING,
    SPACE_TIME,
    EXPONENTIAL_INTEGRATOR
  };
}


#endif
