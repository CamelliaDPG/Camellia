//
//  RefinementHistory.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/2/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_RefinementHistory_h
#define Camellia_debug_RefinementHistory_h

#include "RefinementObserver.h"

using namespace std;

enum RefinementType {
  H_REFINEMENT, P_REFINEMENT, H_X_REFINEMENT, H_Y_REFINEMENT, H_UNREFINEMENT // X: cut vertically, Y: cut horizontally
};

class RefinementHistory : public RefinementObserver {
  typedef pair< RefinementType, set<int> > Refinement;
  vector< Refinement > _refinements;
public:
  void hRefine(const set<int> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern);
  void pRefine(const set<int> &cellIDs);
  void hUnrefine(const set<int> &cellIDs);
};


#endif
