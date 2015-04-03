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

#include "TypeDefs.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"
#include "ElementType.h"
#include "RefinementPattern.h"

#include "Cell.h"

using namespace std;

namespace Camellia {
  class Element {
  private:
    CellPtr _cell;
    Mesh* _mesh;
    
    Teuchos::RCP< ElementType > _elemTypePtr;
    IndexType _cellIndex; // index into the Mesh's Elements vector for ElementType for a given partition
    GlobalIndexType _globalCellIndex; // index into a vector of *all* elements of a given type across partitions...

    bool _deleted;
  public:
  //constructor:
    Element(Mesh* mesh, GlobalIndexType cellID, Teuchos::RCP< ElementType > elemType, IndexType cellIndex, GlobalIndexType globalCellIndex=-1);
    
    Teuchos::RCP< ElementType > elementType() { return _elemTypePtr; }
    void setElementType( Teuchos::RCP< ElementType > newElementType) { _elemTypePtr = newElementType; }
    
    int cellID() { return _cell->cellIndex(); }
    int cellIndex() { return _cellIndex; }
    void setCellIndex(int newCellIndex) { _cellIndex = newCellIndex; }
    int globalCellIndex() { return _globalCellIndex; }
    void setGlobalCellIndex(int globalCellIndex) { _globalCellIndex = globalCellIndex; }
    int numSides();
    void getSidePointsInNeighborRefCoords(Intrepid::FieldContainer<double> &neighborRefPoints, int sideIndex,
                                          const Intrepid::FieldContainer<double> &refPoints);
    void getSidePointsInParentRefCoords(Intrepid::FieldContainer<double> &parentRefPoints, int sideIndex, 
                                        const Intrepid::FieldContainer<double> &childRefPoints);
    ElementPtr getNeighbor(int & mySideIndexInNeighbor, int neighborsSideIndexInMe);
    int getNeighborCellID(int sideIndex);
    int getSideIndexInNeighbor(int sideIndex);
    
    int indexInParentSide(int parentSide);
    
    void setNeighbor(int neighborsSideIndexInMe, Teuchos::RCP< Element > elemPtr, int mySideIndexInNeighbor);
    //int subSideIndexInNeighbor(int neighborsSideIndexInMe);
    
    void setRefinementPattern(Teuchos::RCP<RefinementPattern> &refPattern);
    vector< pair<unsigned, unsigned> > & childIndicesForSide(unsigned sideIndex); // pair( child index, sideIndex in child of the side shared with parent)
    
    ElementPtr getParent();

    int parentSideForSideIndex(int mySideIndex);
    int numChildren();
    ElementPtr getChild(int childIndex);
    bool isActive();
    bool isParent();
    bool isChild();
    vector< pair<int,int> > getDescendantsForSide(int sideIndex, bool leafNodesOnly = true);
    
    set<int> getDescendants(bool leafNodesOnly = true);
    
    void deleteChildrenFromMesh(set< pair<int,int> > &affectedNeighborSides, set<int> &deletedElements);
    void deleteFromMesh(set< pair<int,int> > &affectedNeighborSides, set<int> &deletedElements);
  //destructor:
    ~Element();
  };
}

#endif
