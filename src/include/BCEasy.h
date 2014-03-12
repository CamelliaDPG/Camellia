//
//  BCEasy.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_BCEasy_h
#define Camellia_BCEasy_h

#include "BC.h"

class BCEasy : public BC {
public:
  BCEasy() : BC(false) {
    cout << "WARNING: using deprecated BCEasy class.  Use BC instead--all features of BCEasy now are supported in BC.\n";
  }
};

#endif
