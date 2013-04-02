//
//  RefinementHistory.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/2/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "RefinementHistory.h"

using namespace std;

void RefinementHistory::hRefine(const set<int> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
  // figure out what type of refinement we have:
  int numChildren = refPattern->numChildren();
  RefinementType refType;
  if (numChildren == 4) {
    refType = H_REFINEMENT;
  } else if (numChildren == 2) {
    if (refPattern->refinedNodes()(0,3,1) == 0.0) {
      // yAnisotropic: horizontal cut
      refType = H_Y_REFINEMENT;
    } else if (refPattern->refinedNodes()(1,0,0)==0.0) {
      refType = H_X_REFINEMENT;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported refinement pattern");
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported refinement pattern");
  }
  Refinement ref = make_pair(refType, cellIDs);
  _refinements.push_back(ref);
}
void RefinementHistory::pRefine(const set<int> &cellIDs) {
  Refinement ref = make_pair(P_REFINEMENT, cellIDs);
  _refinements.push_back(ref);
}
void RefinementHistory::hUnrefine(const set<int> &cellIDs) {
  Refinement ref = make_pair(H_UNREFINEMENT, cellIDs);
  _refinements.push_back(ref);
}