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
public:
  Cell(CellTopoPtr cellTopo, const vector<unsigned> &vertices, const vector< map< unsigned, unsigned > > &subcellPermutations,
       unsigned cellIndex, const vector< vector<unsigned> > &entityIndices) {
    _cellTopo = cellTopo;
    _vertices = vertices;
    _subcellPermutations = subcellPermutations;
    _cellIndex = cellIndex;
    _entityIndices = entityIndices;
  }
  unsigned cellIndex() {
    return _cellIndex;
  }
  
  const vector< Teuchos::RCP< Cell > > &children() {
    return _children;
  }
  void setChildren(vector< Teuchos::RCP< Cell > > children) {
    _children = children;
    Teuchos::RCP< Cell > thisPtr = Teuchos::rcp( this, false ); // doesn't own memory
    for (vector< Teuchos::RCP< Cell > >::iterator childIt = children.begin(); childIt != children.end(); childIt++) {
      (*childIt)->setParent(thisPtr);
    }
  }
  vector<unsigned> getChildIndices() {
    vector<unsigned> indices(_children.size());
    for (unsigned childOrdinal=0; childOrdinal<_children.size(); childOrdinal++) {
      indices[childOrdinal] = _children[childOrdinal]->cellIndex();
    }
    return indices;
  }
  
  vector< pair<unsigned, unsigned> > childrenForSide(unsigned sideIndex);
  
  vector< pair< unsigned, unsigned> > getDescendantsForSide(int sideIndex, bool leafNodesOnly = true);
  
  unsigned entityIndex(unsigned subcdim, unsigned subcord) {
    return _entityIndices[subcdim][subcord];
  }
  
  Teuchos::RCP<Cell> getParent() {
    return _parent;
  }
  
  void setParent(Teuchos::RCP<Cell> parent) {
    _parent = parent;
  }
  
  bool isParent() { return _children.size() > 0; }
  
  RefinementPatternPtr refinementPattern() {
    return _refPattern;
  }
  void setRefinementPattern(RefinementPatternPtr refPattern) {
    _refPattern = refPattern;
  }
  
  CellTopoPtr topology() {
    return _cellTopo;
  }
  
  const vector< unsigned > &vertices() {return _vertices;}
};

typedef Teuchos::RCP<Cell> CellPtr;

#endif /* defined(__Camellia_debug__Cell__) */
