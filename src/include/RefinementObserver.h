// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  RefinementObserver.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/2/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_RefinementObserver_h
#define Camellia_debug_RefinementObserver_h

#include "TypeDefs.h"

#include "Teuchos_RCP.hpp"
#include "RefinementPattern.h"

using namespace std;

namespace Camellia
{
class RefinementObserver
{
public:
  virtual ~RefinementObserver() {}
  virtual void hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool repartitionAndRebuild) {}
  virtual void hRefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool repartitionAndRebuild)
  {
    hRefine(cellIDs, refPattern, repartitionAndRebuild);
  }
  virtual void didHRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool didRepartitionAndRebuild) {}
  virtual void didHRefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool didRepartitionAndRebuild)
  {
    didHRefine(cellIDs, refPattern, didRepartitionAndRebuild);
  }
  virtual void pRefine(const set<GlobalIndexType> &cellIDs, int pToAdd, bool repartitionAndRebuild) {}
  virtual void hUnrefine(const set<GlobalIndexType> &cellIDs, bool repartitionAndRebuild) {}

  virtual void willHUnrefine(const set<GlobalIndexType> &cellIDs, bool willRepartitionAndRebuild) {}
  virtual void willHUnrefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, bool willRepartitionAndRebuild)
  {
    willHUnrefine(cellIDs, willRepartitionAndRebuild);
  }
  
  virtual void didHUnrefine(const set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild) {}
  virtual void didHUnrefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild)
  {
    didHUnrefine(cellIDs, didRepartitionAndRebuild);
  }
  
  virtual void didRepartition(MeshTopologyPtr meshTopo) {}
};
}

#endif
