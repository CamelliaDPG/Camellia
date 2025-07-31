// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  RefinementHistory.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/2/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_RefinementHistory_h
#define Camellia_debug_RefinementHistory_h

#include "TypeDefs.h"

#include "RefinementObserver.h"

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include <EpetraExt_HDF5.h>
#endif

using namespace std;

namespace Camellia
{
enum RefinementType
{
  H_REFINEMENT, P_REFINEMENT, P_UNREFINEMENT, H_X_REFINEMENT, H_Y_REFINEMENT, H_Z_REFINEMENT, H_UNREFINEMENT, NULL_REFINEMENT, UNKNOWN_REFINEMENT, REPARTITION_AND_REBUILD // X: cut vertically, Y: cut horizontally
};

class RefinementHistory : public RefinementObserver
{
  typedef pair< RefinementType, set<GlobalIndexType> > Refinement;
  vector< Refinement > _refinements;
  std::string _saveFile;
public:
  using RefinementObserver::hRefine;
  void hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool repartitionAndRebuild) override;
  void hUnrefine(const set<GlobalIndexType> &cellIDs, bool repartitionAndRebuild) override;

  void pRefine(const set<GlobalIndexType> &cellIDs, int pToAdd, bool repartitionAndRebuild) override;

  void didRepartition(MeshTopologyPtr meshTopo) override;
  
  void playback(MeshPtr mesh);

  //! if saveFile is not "", then will save on each change to refinements
  void setSaveFile(std::string &saveFile);
  
  // file I/O
  void saveToFile(string fileName);
  void loadFromFile(string fileName);
#ifdef HAVE_EPETRAEXT_HDF5
  void saveToHDF5(EpetraExt::HDF5 &hdf5);
#endif

  static RefinementPatternPtr refPatternForRefType(RefinementType refType, CellTopoPtr cellTopo);
};
}


#endif
