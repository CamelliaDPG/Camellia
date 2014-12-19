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
#include "CamelliaCellTools.h"
#include "Mesh.h"

// constructor:
Element::Element(Mesh* mesh, GlobalIndexType cellID, Teuchos::RCP< ElementType > elemTypePtr, IndexType cellIndex, GlobalIndexType globalCellIndex) {
  _mesh = mesh;
  _cell = _mesh->getTopology()->getCell(cellID);
  _cellIndex = cellIndex;
  _globalCellIndex = globalCellIndex;
  _elemTypePtr = elemTypePtr;

  _deleted = false;
}

Teuchos::RCP< Element > Element::getChild(int childOrdinal) {
  return _mesh->getElement( _cell->getChildIndices()[childOrdinal] );
}

void Element::getSidePointsInNeighborRefCoords(FieldContainer<double> &neighborRefPoints, int sideIndex,
                                               const FieldContainer<double> &refPoints) {
  // assumes neighbor on the side is a peer.  For what to do if not, see MeshTestSuite::neighborBasesAgreeOnSides().
  // TODO: consider incorporating similar logic here, and reduce the amount of logic in neighborBasesAgreeOnSides().
  
  int mySideIndexInNeighbor;
  ElementPtr neighbor = this->getNeighbor(mySideIndexInNeighbor, sideIndex);
  
  if ((sideIndex >= this->numSides()) || (neighbor.get() == NULL) || (neighbor->cellID() == -1) ) {
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
   
  if (_cell->getParent().get() == NULL) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"parent is null");
  }
  
  if (childRefPoints.size() != parentRefPoints.size()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"childRefPoints.size() != parentRefPoints.size()");
  }
  
  int parentSideIndex = parentSideForSideIndex(sideIndex);
  int numChildrenForSide = _cell->getParent()->childrenForSide(sideIndex).size();
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

ElementPtr Element::getParent() {
  if (_cell->getParent().get() == NULL) return Teuchos::rcp((Element*) NULL);
  
  CellPtr parentCell = _mesh->getTopology()->getCell(_cell->getParent()->cellIndex());
  return _mesh->getElement(parentCell->cellIndex());
}

int Element::numChildren() {
  return _cell->children().size();
}

int Element::numSides() {
  return _cell->getSideCount();
}

int Element::parentSideForSideIndex(int mySideOrdinal) {
  // returns the sideIndex in parent of the given side
  // returns -1 if the given side isn't shared with parent
  CellPtr parentCell = _cell->getParent();
  if (parentCell.get()==NULL) return -1;
  
  CellTopoPtr cellTopo = parentCell->topology();

  int numSides = cellTopo->getSideCount();
  
  for (int parentSideOrdinal=0; parentSideOrdinal<numSides; parentSideOrdinal++) {
    vector< pair<GlobalIndexType, unsigned> > childrenForSide = parentCell->childrenForSide(parentSideOrdinal);
    for (vector< pair<GlobalIndexType, unsigned> >::iterator childIt = childrenForSide.begin();
         childIt != childrenForSide.end(); childIt++) {
      GlobalIndexType childCellIndex = childIt->first;
      if (childCellIndex == _cell->cellIndex()) {
        unsigned childSideOrdinal = childIt->second;
        if (childSideOrdinal == mySideOrdinal) {
          return parentSideOrdinal;
        }
      }
    }
  }
  return -1;
}

vector< pair<unsigned,unsigned> > & Element::childIndicesForSide(unsigned sideIndex) {
  return _cell->refinementPattern()->childrenForSides()[sideIndex];
}

set<int> Element::getDescendants(bool leafNodesOnly) {
  set<int> descendants;
  if (( numChildren() == 0) || !leafNodesOnly) {
    descendants.insert(_cell->cellIndex());
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
    descendantsForSide.push_back( make_pair( _cell->cellIndex(), sideIndex) );
    return descendantsForSide;
  }
  
  vector< pair<unsigned,unsigned> > childIndices = childIndicesForSide(sideIndex);
  vector< pair<unsigned,unsigned> >::iterator entryIt;
  
  for (entryIt=childIndices.begin(); entryIt != childIndices.end(); entryIt++) {
    unsigned childOrdinal = (*entryIt).first;
    unsigned childSideOrdinal = (*entryIt).second;
    CellPtr childCell = _cell->children()[childOrdinal];
    if ( (! childCell->isParent()) || (! leafNodesOnly ) ) {
      // (            leaf node              ) || ...
      descendantsForSide.push_back( make_pair( childCell->cellIndex(), childSideOrdinal) );
    }
    if ( childCell->isParent() ) {
      ElementPtr childElement = _mesh->getElement(childCell->cellIndex());
      vector< pair<int,int> > childDescendants = childElement->getDescendantsForSide(childSideOrdinal,leafNodesOnly);
      vector< pair<int,int> >::iterator childEntryIt;
      for (childEntryIt=childDescendants.begin(); childEntryIt != childDescendants.end(); childEntryIt++) {
        descendantsForSide.push_back(*childEntryIt);
      }
    }
  }
  return descendantsForSide;
}

ElementPtr Element::getNeighbor( int & mySideIndexInNeighbor, int neighborsSideIndexInMe) {
  TEUCHOS_TEST_FOR_EXCEPTION( ( neighborsSideIndexInMe >= numSides() ) || (neighborsSideIndexInMe < 0),
                     std::invalid_argument,
                     "neighbor's side index in me is out of bounds.");
  ElementPtr elemPtr;
  pair<IndexType, unsigned> neighborInfo = _cell->getNeighborInfo(neighborsSideIndexInMe);
  
  IndexType neighborCellID = neighborInfo.first;
  if (neighborCellID == -1) {
    mySideIndexInNeighbor = -1;
    return elemPtr; // NULL
  }
  mySideIndexInNeighbor = neighborInfo.second;
  return _mesh->getElement(neighborCellID);
}

int Element::getNeighborCellID(int sideIndex) {
  // returns -1 if neighbor isn't a peer (or is boundary)
  pair<IndexType, unsigned> neighborInfo = _cell->getNeighborInfo(sideIndex);
  if (neighborInfo.first == -1) return -1;
  CellPtr neighborCell = _mesh->getTopology()->getCell(neighborInfo.first);
  int sideOrdinalInNeighbor = neighborInfo.second;
  pair<IndexType, unsigned> neighborNeighborInfo = neighborCell->getNeighborInfo(sideOrdinalInNeighbor);
  if (neighborNeighborInfo.first != _cell->cellIndex()) { // they are not peers
    return -1;
  }
  return neighborInfo.first;
}

int Element::getSideIndexInNeighbor(int sideIndex) {
  pair<IndexType, unsigned> neighborInfo = _cell->getNeighborInfo(sideIndex);
  return neighborInfo.second;
}

int Element::indexInParentSide(int parentSide) {
  ElementPtr parent = getParent();
  int numChildrenForSide = parent->childIndicesForSide(parentSide).size();
  for (int childIndexInSide = 0; childIndexInSide<numChildrenForSide; childIndexInSide++) {
    int childIndex = parent->childIndicesForSide(parentSide)[childIndexInSide].first;
    int childCellID = parent->getChild(childIndex)->cellID();
    if (_cell->cellIndex() == childCellID) {
      return childIndexInSide;
    }
  }
  return -1; // not found
}

//int Element::subSideIndexInNeighbor(int neighborsSideIndexInMe) {
//  return _subSideIndicesInNeighbors[neighborsSideIndexInMe];
//}

bool Element::isChild() {
  return _cell->getParent().get() != NULL;
}

bool Element::isActive() {
  return !isParent() && !_deleted;
}

bool Element::isParent() {
  return _cell->isParent();
}

//destructor:
Element::~Element() {
}