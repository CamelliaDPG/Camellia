#ifndef DPG_ELEMENT
#define DPG_ELEMENT

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
 *  Element.h
 *
 */

// Teuchos includes
#include "Teuchos_RCP.hpp"
#include "ElementType.h"
#include "RefinementPattern.h"

using namespace std;

class Element {
private:
  int _cellID; // unique ID, also the index for the Mesh into the Elements vector
  Teuchos::RCP< ElementType > _elemTypePtr;
  int _cellIndex; // index into the Mesh's Elements vector for ElementType for a given partition
  int _globalCellIndex; // index into a vector of *all* elements of a given type across partitions...
  int _numSides;
  pair< Element* , int > * _neighbors; // the int is a sideIndex in neighbor (i.e. which of neighbor's sides are we?)
  //int * _subSideIndicesInNeighbors;    // for neighbors that have hanging nodes that we are part of...
  Teuchos::RCP<RefinementPattern> _refPattern; // the refinement pattern that gives rise to _children
  vector< Teuchos::RCP<Element> > _children;
  Element* _parent;
  map<int, int> _parentSideLookupTable; // childSideIndex --> shared parentSideIndex
public:
//constructor:
  Element(int cellID, Teuchos::RCP< ElementType > elemType, int cellIndex, int globalCellIndex=-1);
  Teuchos::RCP< ElementType > elementType() { return _elemTypePtr; }
  void setElementType( Teuchos::RCP< ElementType > newElementType) { _elemTypePtr = newElementType; }
  int cellID() { return _cellID; }
  int cellIndex() { return _cellIndex; }
  void setCellIndex(int newCellIndex) { _cellIndex = newCellIndex; }
  int globalCellIndex() { return _globalCellIndex; }
  void setGlobalCellIndex(int globalCellIndex) { _globalCellIndex = globalCellIndex; }
  int numSides() { return _numSides; }
  void getNeighbor(Element* &elemPtr, int & mySideIndexInNeighbor, int neighborsSideIndexInMe);
  int getNeighborCellID(int sideIndex);
  int getSideIndexInNeighbor(int sideIndex);
  void setNeighbor(int neighborsSideIndexInMe, Teuchos::RCP< Element > elemPtr, int mySideIndexInNeighbor);
  //int subSideIndexInNeighbor(int neighborsSideIndexInMe);
  
  void setRefinementPattern(Teuchos::RCP<RefinementPattern> &refPattern);
  vector< pair<int, int> > & childIndicesForSide(int sideIndex); // pair( child index, sideIndex in child of the side shared with parent)
  
  void addChild(Teuchos::RCP< Element > childPtr);
  Element* getParent();
  void setParent(Element* parentPtr, map<int,int> parentSideLookupTable);
  int parentSideForSideIndex(int mySideIndex);
  int numChildren();
  Teuchos::RCP< Element > getChild(int childIndex);
  bool isNeighbor(Teuchos::RCP<Element> putativeNeighbor, int &sideIndexForNeighbor);
  bool isParent();
  bool isChild();
  vector< pair<int,int> > getDescendentsForSide(int sideIndex);
//destructor:
  ~Element();
};

#endif