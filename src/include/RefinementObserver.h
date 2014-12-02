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

class MeshTopology;
typedef Teuchos::RCP<MeshTopology> MeshTopologyPtr;

#include "IndexType.h"

using namespace std;

class RefinementObserver {
public:
  virtual ~RefinementObserver() {}
  virtual void hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {}
  virtual void hRefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
    hRefine(cellIDs, refPattern);
  }
  virtual void didHRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {}
  virtual void didHRefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
    didHRefine(cellIDs, refPattern);
  }
  virtual void pRefine(const set<GlobalIndexType> &cellIDs) {}
  virtual void hUnrefine(const set<GlobalIndexType> &cellIDs) {}
  
  virtual void didHUnrefine(const set<GlobalIndexType> &cellIDs) {}
  virtual void didHUnrefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs) {
    didHUnrefine(cellIDs);
  }
};

#endif
