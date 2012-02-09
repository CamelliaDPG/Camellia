// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER 


/*
 *  DPGElement.cpp
 *
 */

#include "Element.h"

// constructor:
Element::Element(int cellID, Teuchos::RCP< ElementType > elemTypePtr, int cellIndex, int globalCellIndex) {
  _cellID = cellID;
  _numSides = elemTypePtr->cellTopoPtr->getSideCount();
  _cellIndex = cellIndex;
  _globalCellIndex = globalCellIndex;
  _neighbors = new pair< Element* , int >[_numSides];
  for (int i=0; i<_numSides; i++) {
    Element* nullNeighbor = NULL;
    _neighbors[i] = make_pair( nullNeighbor, -1 ); // default value
  }
//  _subSideIndicesInNeighbors = new int[_numSides];
//  for (int i=0; i<_numSides; i++) {
//    _subSideIndicesInNeighbors[i] = -1; // default value
//  }
  _elemTypePtr = elemTypePtr;
  _parent = NULL;
}

void Element::addChild(Teuchos::RCP< Element > childPtr) {
  _children.push_back(childPtr);
  int childIndex = _children.size() - 1;
  map<int,int> parentSideLookupTable = _refPattern->parentSideLookupForChild(childIndex);
  childPtr->setParent(this, parentSideLookupTable);
}

Teuchos::RCP< Element > Element::getChild(int childIndex) {
  return _children[childIndex];
}

Element* Element::getParent() {
  return _parent;
}

bool Element::isNeighbor(Teuchos::RCP<Element> putativeNeighbor, int &sideIndexForNeighbor) {
  int neighborCellID = putativeNeighbor->cellID();
  for (int sideIndex=0; sideIndex<_numSides; sideIndex++) {
    if (_neighbors[sideIndex].first->cellID() == neighborCellID) {
      // match
      sideIndexForNeighbor = sideIndex;
      return true;
    }
  }
  sideIndexForNeighbor = -1;
  return false;
}

int Element::numChildren() {
  return _children.size();
}

void Element::setNeighbor(int neighborsSideIndexInMe, Teuchos::RCP< Element > elemPtr, int mySideIndexInNeighbor) {
  TEST_FOR_EXCEPTION( ( neighborsSideIndexInMe >= _numSides ) || (neighborsSideIndexInMe < 0),
                     std::invalid_argument,
                     "neighbor's side index in me is out of bounds.");
  // get the raw pointer, to avoid circular references.  The Mesh owns a reference to each Element, which is good
  // enough.  We don't need Elements to outlast the Mesh!
  _neighbors[neighborsSideIndexInMe] = make_pair(elemPtr.get(), mySideIndexInNeighbor);
  // for hanging nodes:
  // _subSideIndicesInNeighbors[neighborsSideIndexInMe] = subSideIndex;
//  cout << "set cellID " << _cellID << "'s neighbor for side ";
//  cout << neighborsSideIndexInMe << " to cellID " << elemPtr->cellID();
//  cout << " (neighbor's sideIndex: " << mySideIndexInNeighbor << ")" << endl;
}

void Element::setParent(Element* parent, map<int,int> parentSideLookupTable) {
  _parent = parent;
  _parentSideLookupTable = parentSideLookupTable;
}

int Element::parentSideForSideIndex(int mySideIndex) {
  // returns the sideIndex in parent of the given side
  // returns -1 if the given side isn't shared with parent
  if (_parentSideLookupTable.find(mySideIndex) != _parentSideLookupTable.end() ) {
    return _parentSideLookupTable[mySideIndex];
  }
  return -1;
}

void Element::setRefinementPattern(Teuchos::RCP<RefinementPattern> &refPattern) {
  _refPattern = refPattern;
}

vector< pair<int,int> > & Element::childIndicesForSide(int sideIndex) {
  return _refPattern->childrenForSides()[sideIndex];
}

vector< pair< int, int> > Element::getDescendantsForSide(int sideIndex, bool leafNodesOnly) {
  // if leafNodesOnly == true,  returns a flat list of leaf nodes (descendants that are not themselves parents)
  // if leafNodesOnly == false, returns a list in descending order: immediate children, then their children, and so on.
  
  // guarantee is that if a child and its parent are both in the list, the parent will come first
  
  // pair (descendantCellID, descendantSideIndex)
  vector< pair< int, int> > descendantsForSide;
  if ( ! isParent() ) {
    descendantsForSide.push_back( make_pair( _cellID, sideIndex) );
    return descendantsForSide;
  }
  
  vector< pair<int,int> > childIndices = childIndicesForSide(sideIndex);
  vector< pair<int,int> >::iterator entryIt;
  
  for (entryIt=childIndices.begin(); entryIt != childIndices.end(); entryIt++) {
    int childIndex = (*entryIt).first;
    int childSideIndex = (*entryIt).second;
    if ( (! _children[childIndex]->isParent()) || (! leafNodesOnly ) ) {
      // (            leaf node              ) || ...
      descendantsForSide.push_back( make_pair( _children[childIndex]->cellID(), childSideIndex) );
    }
    if ( _children[childIndex]->isParent() ) {
      vector< pair<int,int> > childDescendants = _children[childIndex]->getDescendantsForSide(childSideIndex,leafNodesOnly);
      vector< pair<int,int> >::iterator childEntryIt;
      for (childEntryIt=childDescendants.begin(); childEntryIt != childDescendants.end(); childEntryIt++) {
        descendantsForSide.push_back(*childEntryIt);
      }
    }
  }
  return descendantsForSide;
}

void Element::getNeighbor(Element* &elemPtr, int & mySideIndexInNeighbor, int neighborsSideIndexInMe) {
  TEST_FOR_EXCEPTION( ( neighborsSideIndexInMe >= _numSides ) || (neighborsSideIndexInMe < 0),
                     std::invalid_argument,
                     "neighbor's side index in me is out of bounds.");
  elemPtr = _neighbors[neighborsSideIndexInMe].first;
  mySideIndexInNeighbor = _neighbors[neighborsSideIndexInMe].second;
}

int Element::getNeighborCellID(int sideIndex) {
  // returns -1 if neighbor isn't set (or is boundary)
  if (_neighbors[sideIndex].first == NULL) {
    return -1;
  } else {
    return _neighbors[sideIndex].first->cellID();
  }
}

int Element::getSideIndexInNeighbor(int sideIndex) {
  return _neighbors[sideIndex].second;
}

int Element::indexInParentSide(int parentSide) {
  Element* parent = getParent();
  int numChildrenForSide = parent->childIndicesForSide(parentSide).size();
  for (int childIndexInSide = 0; childIndexInSide<numChildrenForSide; childIndexInSide++) {
    int childIndex = parent->childIndicesForSide(parentSide)[childIndexInSide].first;
    int childCellID = parent->getChild(childIndex)->cellID();
    if (_cellID == childCellID) {
      return childIndexInSide;
    }
  }
  return -1; // not found
}

//int Element::subSideIndexInNeighbor(int neighborsSideIndexInMe) {
//  return _subSideIndicesInNeighbors[neighborsSideIndexInMe];
//}

bool Element::isChild() {
  return _parent != NULL;
}

bool Element::isParent() {
  return _children.size() > 0;
}

//destructor:
Element::~Element() {
  delete[] _neighbors;
}