//
//  Cell.h
//  Camellia-debug
//
//  Created by Nate Roberts on 1/22/14.
//
//

#ifndef __Camellia_debug__Cell__
#define __Camellia_debug__Cell__

#include <iostream>
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "RefinementPattern.h"

typedef Teuchos::RCP<shards::CellTopology> CellTopoPtr;

using namespace std;

// "cells" are geometric entities -- they do not define any kind of basis
// "elements" are cells endowed with a (local) functional discretization
class Cell {
  unsigned _cellIndex;
  CellTopoPtr _cellTopo;
  vector< unsigned > _vertices;
  vector< vector<unsigned> > _entityIndices;  // indices: [subcdim][subcord]
  vector< map< unsigned, unsigned > > _subcellPermutations; // permutation to get from local ordering to the canonical one
  
  // for parents:
  vector< Teuchos::RCP< Cell > > _children;
  RefinementPatternPtr _refPattern;
  
  // for children:
  Teuchos::RCP<Cell> _parent; // doesn't own memory (avoid circular reference issues)
  
  //neighbors:
  vector< pair<unsigned, unsigned> > _neighbors; // cellIndex, neighborSideIndex (which may not refer to the same side)
  /* rules for neighbors:
     - hanging node sides point to the constraining neighbor (which may not be active)
     - cells with broken neighbors point to their peer, the ancestor of the active neighbors
   */
public:
  Cell(CellTopoPtr cellTopo, const vector<unsigned> &vertices, const vector< map< unsigned, unsigned > > &subcellPermutations,
       unsigned cellIndex, const vector< vector<unsigned> > &entityIndices);
  unsigned cellIndex();
  const vector< Teuchos::RCP< Cell > > &children();
  void setChildren(vector< Teuchos::RCP< Cell > > children);
  vector<unsigned> getChildIndices();
  vector< pair<unsigned, unsigned> > childrenForSide(unsigned sideIndex);
  vector< pair< unsigned, unsigned> > getDescendantsForSide(int sideIndex, bool leafNodesOnly = true);
  unsigned entityIndex(unsigned subcdim, unsigned subcord);
  const vector<unsigned>& getEntityIndices(unsigned subcdim);
  Teuchos::RCP<Cell> getParent();
  void setParent(Teuchos::RCP<Cell> parent);
  bool isParent();
  
  RefinementPatternPtr refinementPattern();
  void setRefinementPattern(RefinementPatternPtr refPattern);
  
  CellTopoPtr topology();
  
  pair<unsigned, unsigned> getNeighbor(unsigned sideIndex);
  void setNeighbor(unsigned sideIndex, unsigned neighborCellIndex, unsigned neighborSideIndex);
  
  const vector< unsigned > &vertices();
};

typedef Teuchos::RCP<Cell> CellPtr;

#endif /* defined(__Camellia_debug__Cell__) */
