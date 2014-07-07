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

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include <EpetraExt_HDF5.h>
#endif

class Mesh;
typedef Teuchos::RCP<Mesh> MeshPtr;

using namespace std;

enum RefinementType {
  H_REFINEMENT, P_REFINEMENT, H_X_REFINEMENT, H_Y_REFINEMENT, H_Z_REFINEMENT, H_UNREFINEMENT, NULL_REFINEMENT, UNKNOWN_REFINEMENT // X: cut vertically, Y: cut horizontally
};

class RefinementHistory : public RefinementObserver {
  typedef pair< RefinementType, set<GlobalIndexType> > Refinement;
  vector< Refinement > _refinements;
public:
  void hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern);
  void hUnrefine(const set<GlobalIndexType> &cellIDs);
  
  void pRefine(const set<GlobalIndexType> &cellIDs);
  
  void playback(MeshPtr mesh);
  
  // file I/O
  void saveToFile(string fileName);
  void loadFromFile(string fileName);
#ifdef HAVE_EPETRAEXT_HDF5
  void saveToHDF5(EpetraExt::HDF5 &hdf5);
#endif
};

#endif
