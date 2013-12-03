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

#include "Intrepid_CellTools.hpp"


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
  _deleted = false;
}

void Element::addChild(Teuchos::RCP< Element > childPtr) {
  _children.push_back(childPtr);
  unsigned childIndex = _children.size() - 1;
  map<unsigned,unsigned> parentSideLookupTable = _refPattern->parentSideLookupForChild(childIndex);
  childPtr->setParent(this, parentSideLookupTable);
}

pair<int,int> Element::ancestralNeighborCellIDForSide(int sideIndex) {
  Element* elem = this;
  while (elem->getNeighborCellID(sideIndex) == -1) {
    if ( ! elem->isChild() ) {
      return make_pair(-1,-1);
    }
    sideIndex = elem->parentSideForSideIndex(sideIndex);
    if (sideIndex == -1) return make_pair(-1,-1);
    elem = elem->getParent();
  }
  // once we get here, we have the appropriate ancestor:
  int elemSideIndexInNeighbor = elem->getSideIndexInNeighbor(sideIndex);
  if (elemSideIndexInNeighbor >= 4) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "elemSideIndex >= 4");
  }
  return make_pair(elem->cellID(),elemSideIndexInNeighbor);
}

Teuchos::RCP< Element > Element::getChild(int childIndex) {
  return _children[childIndex];
}

void Element::getSidePointsInNeighborRefCoords(FieldContainer<double> &neighborRefPoints, int sideIndex,
                                               const FieldContainer<double> &refPoints) {
  // assumes neighbor on the side is a peer.  For what to do if not, see MeshTestSuite::neighborBasesAgreeOnSides().
  // TODO: consider incorporating similar logic here, and reduce the amount of logic in neighborBasesAgreeOnSides().
  
  if ((sideIndex >= _numSides) || (_neighbors[sideIndex].first == NULL) || (_neighbors[sideIndex].first->cellID() == -1) ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "neighbor is NULL.");
  }
  // 2D: assume that neighbor and element both have similarly ordered (CW or CCW) vertices, so that
  //     the transformation is just a flip from (-1,1) to (1,-1)
  FieldContainer<double> nodesInNeighborRef(1,2,1);
  nodesInNeighborRef(0,0,0) = 1.0;
  nodesInNeighborRef(0,1,0) = -1.0;
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  CellTools<double>::mapToPhysicalFrame(neighborRefPoints,refPoints,nodesInNeighborRef,line_2,0);
}

void Element::getSidePointsInParentRefCoords(FieldContainer<double> &parentRefPoints, int sideIndex, 
                                             const FieldContainer<double> &childRefPoints) {
  /*
   
   This is very much a 2D implementation of this method.
   Because our edge divisions can be assumed to be in exactly two equal pieces wherever they are divided, 
   we know that child's side indices in parent cell are either (-1,0) or (0,1).
   
   As in the Mesh class, we assume that there is directional agreement between parent and child: if child's
   vertices are ordered counter-clockwise, then so are parent's.
   */
   
  if (_parent == NULL) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"parent is null");
  }
  
  if (childRefPoints.size() != parentRefPoints.size()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"childRefPoints.size() != parentRefPoints.size()");
  }
  
  int parentSideIndex = parentSideForSideIndex(sideIndex);
  int numChildrenForSide = _parent->childIndicesForSide(parentSideIndex).size();
  if (numChildrenForSide==1) {
    // that's OK; we can just copy the childRefPoints into parentRefPoints
    int numPoints = childRefPoints.size();
    for (int pointIndex = 0; pointIndex < numPoints; pointIndex++) {
      parentRefPoints[pointIndex] = childRefPoints[pointIndex];
    }
    return;
  }
  
  FieldContainer<double> childEdgeNodesInParentRef(1,2,1);
  int childIndexInParentSide = indexInParentSide(parentSideIndex);
  if (childIndexInParentSide == 0) {
    childEdgeNodesInParentRef(0,0,0) = -1;
    childEdgeNodesInParentRef(0,1,0) = 0;
  } else if (childIndexInParentSide == 1) {
    childEdgeNodesInParentRef(0,0,0) = 0;
    childEdgeNodesInParentRef(0,1,0) = 1;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION( true, std::invalid_argument, "indexInParentSide isn't 0 or 1" );
  }
  
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  CellTools<double>::mapToPhysicalFrame(parentRefPoints,childRefPoints,childEdgeNodesInParentRef,line_2,0);
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
  TEUCHOS_TEST_FOR_EXCEPTION( ( neighborsSideIndexInMe >= _numSides ) || (neighborsSideIndexInMe < 0),
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

void Element::setParent(Element* parent, map<unsigned,unsigned> parentSideLookupTable) {
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

vector< pair<unsigned,unsigned> > & Element::childIndicesForSide(unsigned sideIndex) {
  return _refPattern->childrenForSides()[sideIndex];
}

set<int> Element::getDescendants(bool leafNodesOnly) {
  set<int> descendants;
  if (( numChildren() == 0) || !leafNodesOnly) {
    descendants.insert(_cellID);
  }
  for (int childIndex=0; childIndex<numChildren(); childIndex++) {
    set<int> childDescendants = this->getChild(childIndex)->getDescendants(leafNodesOnly);
    descendants.insert(childDescendants.begin(), childDescendants.end());
  }
  return descendants;
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
  
  vector< pair<unsigned,unsigned> > childIndices = childIndicesForSide(sideIndex);
  vector< pair<unsigned,unsigned> >::iterator entryIt;
  
  for (entryIt=childIndices.begin(); entryIt != childIndices.end(); entryIt++) {
    unsigned childIndex = (*entryIt).first;
    unsigned childSideIndex = (*entryIt).second;
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
  TEUCHOS_TEST_FOR_EXCEPTION( ( neighborsSideIndexInMe >= _numSides ) || (neighborsSideIndexInMe < 0),
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

void Element::deleteChildrenFromMesh(set< pair<int,int> > &affectedNeighborSides, set<int> &deletedElements) {
  for (int i=0; i<_children.size(); i++) {
    _children[i]->deleteFromMesh(affectedNeighborSides, deletedElements);
  }
  _children.clear();
}

void Element::deleteFromMesh(set< pair<int,int> > &affectedNeighborSides, set<int> &deletedElements) {
  deleteChildrenFromMesh(affectedNeighborSides, deletedElements);
  
  ElementPtr nullPtr = Teuchos::rcp( new Element(-1,elementType(),-1) );

  for (int sideIndex=0; sideIndex<_numSides; sideIndex++) {
    if (_neighbors[sideIndex].first->cellID() != -1) {
      affectedNeighborSides.insert(make_pair(_neighbors[sideIndex].first->cellID(),
                                             _neighbors[sideIndex].second));
      Element* neighbor = _neighbors[sideIndex].first;
      int mySideIndexInNeighbor = _neighbors[sideIndex].second;
      neighbor->setNeighbor(sideIndex, nullPtr, mySideIndexInNeighbor);
    }
  }
  deletedElements.insert(this->cellID());
  _deleted = true;
}

//int Element::subSideIndexInNeighbor(int neighborsSideIndexInMe) {
//  return _subSideIndicesInNeighbors[neighborsSideIndexInMe];
//}

bool Element::isChild() {
  return _parent != NULL;
}

bool Element::isActive() {
  return !isParent() && !_deleted;
}

bool Element::isParent() {
  return _children.size() > 0;
}

//destructor:
Element::~Element() {
  delete[] _neighbors;
}