//
//  RHSEasy.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_RHSEasy_h
#define Camellia_RHSEasy_h

#include "RHS.h"
#include "LinearTerm.h"

class RHSEasy : public RHS {
public:
  RHSEasy() : RHS(false) { // false: not a legacy subclass
    cout << "WARNING: invoking RHSEasy, which is now deprecated.  (All its functionality has been moved into the RHS superclass.)\n";
  }
};

#endif
