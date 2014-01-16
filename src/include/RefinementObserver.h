//
//  RefinementObserver.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/2/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_RefinementObserver_h
#define Camellia_debug_RefinementObserver_h

#include "Teuchos_RCP.hpp"
#include "RefinementPattern.h"

using namespace std;

class RefinementObserver {
public:
  virtual ~RefinementObserver() {}
  virtual void hRefine(const set<int> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern) = 0;
  virtual void pRefine(const set<int> &cellIDs) = 0;
  virtual void hUnrefine(const set<int> &cellIDs) = 0;
};

#endif
