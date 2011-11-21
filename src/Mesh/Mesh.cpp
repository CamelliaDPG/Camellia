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
 *  Mesh.cpp
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */

#include "Mesh.h"
#include "BilinearFormUtility.h"
#include "ElementType.h"
#include "DofOrderingFactory.h"
#include "BasisFactory.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

using namespace Intrepid;

typedef Teuchos::RCP< ElementType > ElementTypePtr;
typedef Teuchos::RCP< Element > ElementPtr;
typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
typedef Teuchos::RCP< DofOrdering > DofOrderingPtr;

Mesh::Mesh(const vector<FieldContainer<double> > &vertices, vector< vector<int> > &elementVertices,
           Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pToAddTest) : _dofOrderingFactory(bilinearForm) {
  _vertices = vertices;
  _usePatchBasis = false;
  _partitionPolicy = Teuchos::rcp( new MeshPartitionPolicy() );
  _numPartitions = 1;

  int spaceDim = 2;
  vector<float> vertexCoords(spaceDim);
  int numVertices = vertices.size();
  for (int vertexIndex=0; vertexIndex<numVertices; vertexIndex++ ) {
    for (int j=0; j<spaceDim; j++) {
      vertexCoords[j] = vertices[vertexIndex](j);
    }
    _verticesMap[vertexCoords].push_back(vertexIndex);
  }
                                      
  _bilinearForm = bilinearForm;
  _boundary.setMesh(this);
  
  _pToAddToTest = pToAddTest;
  int pTest = H1Order + pToAddTest;
  
  Teuchos::RCP<shards::CellTopology> triTopoPtr, quadTopoPtr;
  quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  triTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() ));
  
  Teuchos::RCP<DofOrdering> quadTestOrderPtr = _dofOrderingFactory.testOrdering(pTest, *(quadTopoPtr.get()));
  Teuchos::RCP<DofOrdering> quadTrialOrderPtr = _dofOrderingFactory.trialOrdering(H1Order, *(quadTopoPtr.get()), true);
  ElementTypePtr quadElemTypePtr = _elementTypeFactory.getElementType(quadTrialOrderPtr, quadTestOrderPtr, quadTopoPtr );
  _nullPtr = Teuchos::rcp( new Element(-1,quadElemTypePtr,-1) );
  
  vector< vector<int> >::iterator elemIt;
  for (elemIt = elementVertices.begin(); elemIt != elementVertices.end(); elemIt++) {
    vector<int> thisElementVertices = *elemIt;
    if (thisElementVertices.size() == 3) {
      Teuchos::RCP<DofOrdering> triTestOrderPtr = _dofOrderingFactory.testOrdering(pTest, *(triTopoPtr.get()));
      Teuchos::RCP<DofOrdering> triTrialOrderPtr = _dofOrderingFactory.trialOrdering(H1Order, *(triTopoPtr.get()), true);
      ElementTypePtr triElemTypePtr = _elementTypeFactory.getElementType(triTrialOrderPtr, triTestOrderPtr, triTopoPtr );
      addElement(thisElementVertices,triElemTypePtr);
    } else if (thisElementVertices.size() == 4) {
      addElement(thisElementVertices,quadElemTypePtr);
    } else {
      TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only elements with 3 or 4 vertices are supported.");
    }
  }
  rebuildLookups();
}

vector< Teuchos::RCP< Element > > & Mesh::activeElements() {
  return _activeElements;
}

Teuchos::RCP<Mesh> Mesh::buildQuadMesh(const FieldContainer<double> &quadBoundaryPoints, 
                                       int horizontalElements, int verticalElements,
                                       Teuchos::RCP< BilinearForm > bilinearForm, 
                                       int H1Order, int pTest, bool triangulate) {
  if (triangulate) cout << "Mesh: Triangulating\n" << endl;
  int pToAddToTest = pTest - H1Order;
  int spaceDim = 2;
  // rectBoundaryPoints dimensions: (4,2) -- and should be in counterclockwise order
  
  vector<FieldContainer<double> > vertices;
  vector< vector<int> > allElementVertices;
  
  TEST_FOR_EXCEPTION( ( quadBoundaryPoints.dimension(0) != 4 ) || ( quadBoundaryPoints.dimension(1) != 2 ),
                     std::invalid_argument,
                     "quadBoundaryPoints should be dimensions (4,2), points in ccw order.");
  
  int numElements = horizontalElements * verticalElements;
  if (triangulate) numElements *= 2;
  int numDimensions = 2;
  
  double southWest_x = quadBoundaryPoints(0,0),
  southWest_y = quadBoundaryPoints(0,1),
  southEast_x = quadBoundaryPoints(1,0), 
  southEast_y = quadBoundaryPoints(1,1),
  northEast_x = quadBoundaryPoints(2,0),
  northEast_y = quadBoundaryPoints(2,1),
  northWest_x = quadBoundaryPoints(3,0),
  northWest_y = quadBoundaryPoints(3,1);
  
  double elemWidth = (southEast_x - southWest_x) / horizontalElements;
  double elemHeight = (northWest_y - southWest_y) / verticalElements;
  
  int cellID = 0;
  
  // set up vertices:
  // vertexIndices is for easy vertex lookup by (x,y) index for our Cartesian grid:
  vector< vector<int> > vertexIndices(horizontalElements+1, vector<int>(verticalElements+1));
  for (int i=0; i<=horizontalElements; i++) {
    for (int j=0; j<=verticalElements; j++) {
      vertexIndices[i][j] = vertices.size();
      FieldContainer<double> vertex(spaceDim);
      vertex(0) = southWest_x + elemWidth*i;
      vertex(1) = southWest_y + elemHeight*j;
      vertices.push_back(vertex);
      //cout << "Mesh: vertex " << _vertices.size() - 1 << ": (" << vertex(0) << "," << vertex(1) << ")\n";
    }
  }
  
  if ( ! triangulate ) {
    int SOUTH = 0, EAST = 1, NORTH = 2, WEST = 3;
    for (int i=0; i<horizontalElements; i++) {
      for (int j=0; j<verticalElements; j++) {
        vector<int> elemVertices;
        elemVertices.push_back(vertexIndices[i][j]);
        elemVertices.push_back(vertexIndices[i+1][j]);
        elemVertices.push_back(vertexIndices[i+1][j+1]);
        elemVertices.push_back(vertexIndices[i][j+1]);
        allElementVertices.push_back(elemVertices);
      }
    }
  } else {
    int SIDE1 = 0, SIDE2 = 1, SIDE3 = 2;
    for (int i=0; i<horizontalElements; i++) {
      for (int j=0; j<verticalElements; j++) {
        vector<int> elemVertices1, elemVertices2; // elem1 is SE of quad, elem2 is NW
        elemVertices1.push_back(vertexIndices[i][j]);     // SIDE1 is SOUTH side of quad
        elemVertices1.push_back(vertexIndices[i+1][j]);   // SIDE2 is EAST
        elemVertices1.push_back(vertexIndices[i+1][j+1]); // SIDE3 is diagonal
        elemVertices2.push_back(vertexIndices[i][j+1]);   // SIDE1 is WEST
        elemVertices2.push_back(vertexIndices[i][j]);     // SIDE2 is diagonal
        elemVertices2.push_back(vertexIndices[i+1][j+1]); // SIDE3 is NORTH
        
        allElementVertices.push_back(elemVertices1);
        allElementVertices.push_back(elemVertices2);
      }
    }
  }
  return Teuchos::rcp( new Mesh(vertices,allElementVertices,bilinearForm,H1Order,pToAddToTest));
}

Teuchos::RCP<Mesh> Mesh::buildQuadMeshHybrid(const FieldContainer<double> &quadBoundaryPoints, 
                                       int horizontalElements, int verticalElements,
                                       Teuchos::RCP< BilinearForm > bilinearForm, 
                                       int H1Order, int pTest) {
  int pToAddToTest = pTest - H1Order;
  int spaceDim = 2;
  // rectBoundaryPoints dimensions: (4,2) -- and should be in counterclockwise order
  
  vector<FieldContainer<double> > vertices;
  vector< vector<int> > allElementVertices;
  
  TEST_FOR_EXCEPTION( ( quadBoundaryPoints.dimension(0) != 4 ) || ( quadBoundaryPoints.dimension(1) != 2 ),
                     std::invalid_argument,
                     "quadBoundaryPoints should be dimensions (4,2), points in ccw order.");
  
  int numDimensions = 2;

  double southWest_x = quadBoundaryPoints(0,0),
  southWest_y = quadBoundaryPoints(0,1),
  southEast_x = quadBoundaryPoints(1,0), 
  southEast_y = quadBoundaryPoints(1,1),
  northEast_x = quadBoundaryPoints(2,0),
  northEast_y = quadBoundaryPoints(2,1),
  northWest_x = quadBoundaryPoints(3,0),
  northWest_y = quadBoundaryPoints(3,1);
  
  double elemWidth = (southEast_x - southWest_x) / horizontalElements;
  double elemHeight = (northWest_y - southWest_y) / verticalElements;
  
  int cellID = 0;
  
  // set up vertices:
  // vertexIndices is for easy vertex lookup by (x,y) index for our Cartesian grid:
  vector< vector<int> > vertexIndices(horizontalElements+1, vector<int>(verticalElements+1));
  for (int i=0; i<=horizontalElements; i++) {
    for (int j=0; j<=verticalElements; j++) {
      vertexIndices[i][j] = vertices.size();
      FieldContainer<double> vertex(spaceDim);
      vertex(0) = southWest_x + elemWidth*i;
      vertex(1) = southWest_y + elemHeight*j;
      vertices.push_back(vertex);
    }
  }
  
  int SOUTH = 0, EAST = 1, NORTH = 2, WEST = 3;
  int SIDE1 = 0, SIDE2 = 1, SIDE3 = 2;
  for (int i=0; i<horizontalElements; i++) {
    for (int j=0; j<verticalElements; j++) {
      bool triangulate = (i >= horizontalElements / 2); // triangles on right half of mesh
      if ( ! triangulate ) {
        vector<int> elemVertices;
        elemVertices.push_back(vertexIndices[i][j]);
        elemVertices.push_back(vertexIndices[i+1][j]);
        elemVertices.push_back(vertexIndices[i+1][j+1]);
        elemVertices.push_back(vertexIndices[i][j+1]);
        allElementVertices.push_back(elemVertices);
      } else {
        vector<int> elemVertices1, elemVertices2; // elem1 is SE of quad, elem2 is NW
        elemVertices1.push_back(vertexIndices[i][j]);     // SIDE1 is SOUTH side of quad
        elemVertices1.push_back(vertexIndices[i+1][j]);   // SIDE2 is EAST
        elemVertices1.push_back(vertexIndices[i+1][j+1]); // SIDE3 is diagonal
        elemVertices2.push_back(vertexIndices[i][j+1]);   // SIDE1 is WEST
        elemVertices2.push_back(vertexIndices[i][j]);     // SIDE2 is diagonal
        elemVertices2.push_back(vertexIndices[i+1][j+1]); // SIDE3 is NORTH
        
        allElementVertices.push_back(elemVertices1);
        allElementVertices.push_back(elemVertices2);          
      }
    }
  }
  return Teuchos::rcp( new Mesh(vertices,allElementVertices,bilinearForm,H1Order,pToAddToTest));
}

void Mesh::quadMeshCellIDs(FieldContainer<int> &cellIDs, 
                           int horizontalElements, int verticalElements,
                           bool useTriangles) {
  // populates cellIDs with either (h,v) or (h,v,2)
  // where h: horizontalElements (indexed by i, below)
  //       v: verticalElements   (indexed by j)
  //       2: triangles per quad (indexed by k)
  
  TEST_FOR_EXCEPTION(cellIDs.dimension(0)!=horizontalElements,
                     std::invalid_argument,
                     "cellIDs should have dimensions: (horizontalElements, verticalElements) or (horizontalElements, verticalElements,2)");
  TEST_FOR_EXCEPTION(cellIDs.dimension(1)!=verticalElements,
                     std::invalid_argument,
                     "cellIDs should have dimensions: (horizontalElements, verticalElements) or (horizontalElements, verticalElements,2)");
  if (useTriangles) {
    TEST_FOR_EXCEPTION(cellIDs.dimension(2)!=2,
                       std::invalid_argument,
                       "cellIDs should have dimensions: (horizontalElements, verticalElements,2)");
    TEST_FOR_EXCEPTION(cellIDs.rank() != 3,
                       std::invalid_argument,
                       "cellIDs should have dimensions: (horizontalElements, verticalElements,2)");
  } else {
    TEST_FOR_EXCEPTION(cellIDs.rank() != 2,
                       std::invalid_argument,
                       "cellIDs should have dimensions: (horizontalElements, verticalElements)");
  }
  
  int cellID = 0;
  for (int i=0; i<horizontalElements; i++) {
    for (int j=0; j<verticalElements; j++) {
      if (useTriangles) {
        cellIDs(i,j,0) = cellID++;
        cellIDs(i,j,1) = cellID++;
      } else {
        cellIDs(i,j) = cellID++;
      }
    }
  }
}

void Mesh::addDofPairing(int cellID1, int dofIndex1, int cellID2, int dofIndex2) {
  int firstCellID, firstDofIndex;
  int secondCellID, secondDofIndex;
  if (cellID1 < cellID2) {
    firstCellID = cellID1;
    firstDofIndex = dofIndex1;
    secondCellID = cellID2;
    secondDofIndex = dofIndex2;
  } else if (cellID1 > cellID2) {
    firstCellID = cellID2;
    firstDofIndex = dofIndex2;
    secondCellID = cellID1;
    secondDofIndex = dofIndex1;
  } else { // cellID1 == cellID2
    firstCellID = cellID1;
    secondCellID = cellID1;
    if (dofIndex1 < dofIndex2) {
      firstDofIndex = dofIndex1;
      secondDofIndex = dofIndex2;
    } else if (dofIndex1 > dofIndex2) {
      firstDofIndex = dofIndex2;
      secondDofIndex = dofIndex1;
    } else {
      TEST_FOR_EXCEPTION( ( dofIndex1 == dofIndex2 ) && ( cellID1 == cellID2 ),
                         std::invalid_argument,
                         "attempt to identify (cellID1, dofIndex1) with itself.");
      
    }
  }
  pair<int,int> key = make_pair(secondCellID,secondDofIndex);
  pair<int,int> value = make_pair(firstCellID,firstDofIndex);
  if ( _dofPairingIndex.find(key) != _dofPairingIndex.end() ) {
    // we already have an entry for this key: need to fix the linked list so it goes from greatest to least
    pair<int,int> existing = _dofPairingIndex[key];
    pair<int,int> entry1, entry2, entry3;
    entry3 = key; // know this is the greatest of the three
    int existingCellID = existing.first;
    int newCellID = value.first;
    int existingDofIndex = existing.second;
    int newDofIndex = value.second;
    
    if (existingCellID < newCellID) {
      entry1 = existing;
      entry2 = value;
    } else if (existingCellID > newCellID) {
      entry1 = value;
      entry2 = existing;
    } else { // cellID1 == cellID2
      if (existingDofIndex < newDofIndex) {
        entry1 = existing;
        entry2 = value;
      } else if (existingDofIndex > newDofIndex) {
        entry1 = value;
        entry2 = existing;
      } else {
        // existing == value --> we're trying to create an entry that already exists
        return;
      }
    }
    // go entry3 -> entry2 -> entry1 -> whatever entry1 pointed at before
    _dofPairingIndex[entry3] = entry2;
//    cout << "Added DofPairing (cellID,dofIndex) --> (earlierCellID,dofIndex) : (" << entry3.first; 
//    cout << "," << entry3.second << ") --> (";
//    cout << entry2.first << "," << entry2.second << ")." << endl;
    // now, there may already be an entry for entry2, so best to use recursion:
    addDofPairing(entry2.first,entry2.second,entry1.first,entry1.second);
  } else {
    _dofPairingIndex[key] = value;
//    cout << "Added DofPairing (cellID,dofIndex) --> (earlierCellID,dofIndex) : (";
//    cout << secondCellID << "," << secondDofIndex << ") --> (";
//    cout << firstCellID << "," << firstDofIndex << ")." << endl;
  }
}

void Mesh::addChildren(ElementPtr parent, vector< vector<int> > &children, vector< vector< pair< int, int> > > &childrenForSides) {
  // probably the first step is to remove the parent's edges.  We will add them back in through the children...
  // second step is to iterate through the children, calling addElement for each once we figure out the right type
  // third step: for neighbors adjacent to more than one child, either assign a multi-basis, or make those neighbor's children our neighbors...
  // probably we want to return the cellIDs for the children, by assigning them in the appropriate vector argument.  But maybe this is not necessary, if we do the job right...
  
  // this assumes 2D...
  // remove parent edges:
  vector<int> parentVertices = _verticesForCellID[parent->cellID()];
  int numVertices = parentVertices.size();
  for (int i=0; i<numVertices; i++) {
    // in 2D, numVertices==numSides, so we can use the vertex index to iterate over sides...
    // check whether parent has just one child on this side: then we need to delete the edge
    // because the child will take over the neighbor relationship
    if ( childrenForSides[i].size() == 1 ) {
      pair<int,int> edge = make_pair(parentVertices[i], parentVertices[ (i+1) % numVertices]);
      // remove this entry from _edgeToCellIDs:
      _edgeToCellIDs.erase(edge);
    }
    // also delete this edge from boundary if it happens to be there
    _boundary.deleteElement(parent->cellID(),i);
  }
  
  // work out the correct ElementTypePtr for each child.
  Teuchos::RCP< DofOrdering > parentTrialOrdering = parent->elementType()->trialOrderPtr;
  int pTrial = _dofOrderingFactory.polyOrder(parentTrialOrdering);
  shards::CellTopology parentTopo = * ( parent->elementType()->cellTopoPtr );
  Teuchos::RCP<shards::CellTopology> triTopoPtr, quadTopoPtr;
  quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  triTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() ));
  
  // we know that all the children have the same poly. orders on the inside, so
  // the edges shared by the children need not change.  Instead, we change according to 
  // what the parent's neighbor has along the side the child shares with the parent...
  
  int numChildren = children.size();
  Teuchos::RCP<DofOrdering> *childTrialOrders = new Teuchos::RCP<DofOrdering>[numChildren];
  Teuchos::RCP<shards::CellTopology> *childTopos = new Teuchos::RCP<shards::CellTopology>[numChildren];
  Teuchos::RCP<DofOrdering> *childTestOrders = new Teuchos::RCP<DofOrdering>[numChildren];
  vector< vector<int> >::iterator childIt;
  for (int childIndex=0; childIndex<numChildren; childIndex++) {
    if (children[childIndex].size() == 3) {
      childTopos[childIndex] = triTopoPtr;
    } else if (children[childIndex].size() == 4) {
      childTopos[childIndex] = quadTopoPtr;
    } else {
      TEST_FOR_EXCEPTION(true,std::invalid_argument,"child with unhandled # of vertices (not 3 or 4)");
    }
    Teuchos::RCP<DofOrdering> basicTrialOrdering = _dofOrderingFactory.trialOrdering(pTrial, *(childTopos[childIndex]), true);
    childTrialOrders[childIndex] = basicTrialOrdering; // we'll upgrade as needed below
  }
  
  int numSides = childrenForSides.size();
  
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    vector< pair< int, int> > childrenForSide = childrenForSides[sideIndex];
    
    vector< pair< int, int> >::iterator entryIt;
    int childIndexInParentSide = 0;
    for ( entryIt=childrenForSide.begin(); entryIt != childrenForSide.end(); entryIt++) {
      int childIndex = (*entryIt).first;
      int childSideIndex = (*entryIt).second;
      _dofOrderingFactory.childMatchParent(childTrialOrders[childIndex],childSideIndex,*(childTopos[childIndex]),
                                           childIndexInParentSide,
                                           parentTrialOrdering,sideIndex,parentTopo);
      childIndexInParentSide++;
    }
  }
  
  // determine test ordering for each child
  for (int childIndex=0; childIndex<numChildren; childIndex++) {
    int maxDegree = childTrialOrders[childIndex]->maxBasisDegree();
    childTestOrders[childIndex] = _dofOrderingFactory.testOrdering(pTrial + _pToAddToTest, *(childTopos[childIndex]));
  }

  vector<int> childCellIDs;
  // create ElementTypePtr for each child, and add child to mesh...
  for (int childIndex=0; childIndex<numChildren; childIndex++) {
    int maxDegree = childTrialOrders[childIndex]->maxBasisDegree();
    childTestOrders[childIndex] = _dofOrderingFactory.testOrdering(maxDegree + _pToAddToTest, *(childTopos[childIndex]));

    ElementTypePtr childType = _elementTypeFactory.getElementType(childTrialOrders[childIndex], 
                                                                  childTestOrders[childIndex], 
                                                                  childTopos[childIndex] );
    ElementPtr child = addElement(children[childIndex], childType);
    childCellIDs.push_back( child->cellID() );
    parent->addChild(child);
  }
  
  // check parent's neighbors along each side: if they are unbroken, then we need to assign an appropriate MultiBasis
  // (this is a job for DofOrderingFactory)
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    int childIndexInParentSide = 0;
    vector< pair< int, int> > childrenForSide = childrenForSides[sideIndex];
    vector< pair< int, int> >::iterator entryIt;
    for ( entryIt=childrenForSide.begin(); entryIt != childrenForSide.end(); entryIt++) {
      int childIndex = (*entryIt).first;
      int childSideIndex = (*entryIt).second;
      int childCellID = parent->getChild(childIndex)->cellID();
      if (parent->getChild(childIndex)->getNeighborCellID(childSideIndex) == -1) {
        // child does not yet have a neighbor
        // inherit the parity of parent: (this can be reversed child gets a neighbor)
        _cellSideParitiesForCellID[childCellID][childSideIndex] = _cellSideParitiesForCellID[parent->cellID()][sideIndex];
//        cout << "addChildren: set cellSideParity for cell " << childCellID << ", sideIndex " << childSideIndex << ": ";
//        cout << _cellSideParitiesForCellID[parent->cellID()][sideIndex] << endl;
      }
    }
    
    int neighborSideIndex;
    ElementPtr neighborToMatch = ancestralNeighborForSide(parent,sideIndex,neighborSideIndex);
    
    if (neighborToMatch->cellID() != -1) { // then we have a neighbor to match along that side...
      matchNeighbor(neighborToMatch,neighborSideIndex);
    }
  }
  delete[] childTrialOrders;
  delete[] childTopos;
  delete[] childTestOrders;
}

ElementPtr Mesh::addElement(const vector<int> & vertexIndices, ElementTypePtr elemType) {
  int numDimensions = elemType->cellTopoPtr->getDimension();
  int numVertices = vertexIndices.size();
  if ( numVertices != elemType->cellTopoPtr->getVertexCount(numDimensions,0) ) {
    TEST_FOR_EXCEPTION(true,
                       std::invalid_argument,
                       "incompatible number of vertices for cell topology");
  }
  if (numDimensions != 2) {
    TEST_FOR_EXCEPTION(true,std::invalid_argument,
                       "mesh only supports 2D right now...");
  }
  int cellID = _elements.size();
  _elements.push_back(Teuchos::rcp( new Element(cellID, elemType, -1) ) ); // cellIndex undefined for now...
  _cellSideParitiesForCellID.push_back(vector<int>(numVertices)); // placeholder parities
  _verticesForCellID.push_back( vertexIndices );
  for (int i=0; i<numVertices; i++ ) {
    // sideIndex is i...
    pair<int,int> edgeReversed = make_pair(vertexIndices[ (i+1) % numVertices ], vertexIndices[i]);
    pair<int,int> edge = make_pair( vertexIndices[i], vertexIndices[ (i+1) % numVertices ]);
    pair<int,int> myEntry = make_pair(cellID, i);
    if ( _edgeToCellIDs.find(edgeReversed) != _edgeToCellIDs.end() ) {
      // there's already an entry for this edge
      if ( _edgeToCellIDs[edgeReversed].size() > 1 ) {
        TEST_FOR_EXCEPTION(true,std::invalid_argument,
                           "In 2D mesh, shouldn't have more than 2 elements per edge...");
      }
      pair<int,int> entry = _edgeToCellIDs[edgeReversed][0];
      int neighborID = entry.first;
      int neighborSideIndex = entry.second;
      setNeighbor(_elements[cellID], i, _elements[neighborID], neighborSideIndex);
      // if there's just one, say howdy to our neighbor (and remove its edge from the boundary)
      // TODO: check to make sure this change is correct (was (cellID, i), but I don't see how that could be right--still, we passed tests....)
      _boundary.deleteElement(neighborID, neighborSideIndex);
    } else {
      setNeighbor(_elements[cellID], i, _nullPtr, i);
      _boundary.addElement(cellID, i);
    }
    if ( _edgeToCellIDs.find(edge) != _edgeToCellIDs.end() ) {
      TEST_FOR_EXCEPTION(true,std::invalid_argument,
                         "Either a duplicate edge (3 elements with a single edge), or element vertices not setup in CCW order.");
    }
    _edgeToCellIDs[edge].push_back(myEntry);
  }
  return _elements[cellID];
}

ElementPtr Mesh::ancestralNeighborForSide(ElementPtr elem, int sideIndex, int &elemSideIndexInNeighbor) {
  // returns neighbor for side, or the neighbor of the ancestor who has a neighbor along shared side
  while (elem->getNeighborCellID(sideIndex) == -1) {
    if ( ! elem->isChild() ) {
      //TEST_FOR_EXCEPTION(true, std::invalid_argument, "No ancestor has an active neighbor along shared side");
      elemSideIndexInNeighbor = -1;
      return _nullPtr;
    }
    sideIndex = elem->parentSideForSideIndex(sideIndex);
    if (sideIndex == -1) return _nullPtr;
    elem = _elements[elem->getParent()->cellID()];
  }
  // once we get here, we have the appropriate ancestor:
  elemSideIndexInNeighbor = elem->getSideIndexInNeighbor(sideIndex);
  if (elemSideIndexInNeighbor >= 4) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "elemSideIndex >= 4");
  }
  return _elements[elem->getNeighborCellID(sideIndex)];
}

void Mesh::buildLocalToGlobalMap() {
  _localToGlobalMap.clear();
  vector<ElementPtr>::iterator elemIterator;
  
  // debug code:
//  for (elemIterator = _activeElements.begin(); elemIterator != _activeElements.end(); elemIterator++) {
//    ElementPtr elemPtr = *(elemIterator);
//    cout << "cellID " << elemPtr->cellID() << "'s trialOrdering:\n";
//    cout << *(elemPtr->elementType()->trialOrderPtr);
//  }
  
  determineDofPairings();
  
  int globalIndex = 0;
  vector< int > trialIDs = _bilinearForm->trialIDs();
  
  for (elemIterator = _activeElements.begin(); elemIterator != _activeElements.end(); elemIterator++) {
    ElementPtr elemPtr = *(elemIterator);
    int cellID = elemPtr->cellID();
    ElementTypePtr elemTypePtr = elemPtr->elementType();
    for (vector<int>::iterator trialIt=trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *(trialIt);
      
      if (! _bilinearForm->isFluxOrTrace(trialID) ) {
        // then all these dofs are interior, so there's no overlap with other elements...
        int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,0);
        for (int dofOrdinal=0; dofOrdinal<numDofs; dofOrdinal++) {
          int localDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,0);
          _localToGlobalMap[make_pair(cellID,localDofIndex)] = globalIndex;
//          cout << "added localToGlobalMap(cellID=" << cellID << ", localDofIndex=" << localDofIndex;
//          cout << ") = " << globalIndex << "\n";
          globalIndex++;
        }
      } else {
        int numSides = elemPtr->numSides();
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
          for (int dofOrdinal=0; dofOrdinal<numDofs; dofOrdinal++) {
            int myLocalDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex);
            pair<int, int> myKey = make_pair(cellID,myLocalDofIndex);
            pair<int, int> myValue;
            if ( _dofPairingIndex.find(myKey) != _dofPairingIndex.end() ) {
              int earlierCellID = _dofPairingIndex[myKey].first;
              int earlierLocalDofIndex = _dofPairingIndex[myKey].second;
              if (_localToGlobalMap.find(_dofPairingIndex[myKey]) == _localToGlobalMap.end() ) {
                // error: we haven't processed the earlier key yet...
                TEST_FOR_EXCEPTION( true,
                                   std::invalid_argument,
                                   "global indices are being processed out of order (should be by cellID, then localDofIndex).");
              } else {
                _localToGlobalMap[myKey] = _localToGlobalMap[_dofPairingIndex[myKey]];
//                cout << "added flux/trace localToGlobalMap(cellID,localDofIndex)=(" << cellID << ", " << myLocalDofIndex;
//                cout << ") = " << _localToGlobalMap[myKey] << "\n";
              }
            } else {
              // this test is necessary because for conforming traces, getDofIndex() may not be 1-1
              // and therefore we might have already assigned a globalDofIndex to myKey....
              if ( _localToGlobalMap.find(myKey) == _localToGlobalMap.end() ) {
                _localToGlobalMap[myKey] = globalIndex;
//                cout << "added flux/trace localToGlobalMap(cellID,localDofIndex)=(" << cellID << ", " << myLocalDofIndex;
//                cout << ") = " << globalIndex << "\n";
                globalIndex++;
              }
            }
          }
        }
      }
    }
  }
  _numGlobalDofs = globalIndex;
}

void Mesh::buildTypeLookups() {
  _elementTypes.clear();
  _cellIDsForElementType.clear();
  
  vector< ElementPtr >::iterator elemIterator;
  for (elemIterator=_activeElements.begin(); elemIterator != _activeElements.end(); elemIterator++) {
    ElementPtr elem = *elemIterator;
    ElementTypePtr elemTypePtr = elem->elementType();
    if ( _cellIDsForElementType.find( elemTypePtr.get() ) == _cellIDsForElementType.end() ) {
      _elementTypes.push_back(elemTypePtr);
    }
    _cellIDsForElementType[elemTypePtr.get()].push_back(elem->cellID());
  }
  
  // now, build cellSideParities and physicalCellNodes lookups
  vector< ElementTypePtr >::iterator elemTypeIt;
  for (elemTypeIt=_elementTypes.begin(); elemTypeIt != _elementTypes.end(); elemTypeIt++) {
    //ElementTypePtr elemType = _elementTypeFactory.getElementType((*elemTypeIt)->trialOrderPtr,
//                                                                 (*elemTypeIt)->testOrderPtr,
//                                                                 (*elemTypeIt)->cellTopoPtr);
    ElementTypePtr elemType = *elemTypeIt; // don't enforce uniquing here (if we wanted to, we
                                           //   would also need to call elem.setElementType for 
                                           //   affected elements...)
    int spaceDim = elemType->cellTopoPtr->getDimension();
    int numSides = elemType->cellTopoPtr->getSideCount();
    vector<int> cellIDs = _cellIDsForElementType[elemType.get()];
    int numCells = cellIDs.size();
    FieldContainer<double> physicalCellNodes( numCells, numSides, spaceDim ) ;
    FieldContainer<double> cellSideParities( numCells, numSides );
    vector<int>::iterator cellIt;
    int cellIndex = 0;
    for (cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
      int cellID = *cellIt;
      ElementPtr elem = _elements[cellID];
      ElementTypePtr oldElemType;
      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
        for (int i=0; i<spaceDim; i++) {
          physicalCellNodes(cellIndex,sideIndex,i) = _vertices[_verticesForCellID[cellID][sideIndex]](i);
        }
        cellSideParities(cellIndex,sideIndex) = _cellSideParitiesForCellID[cellID][sideIndex];
      }
      elem->setCellIndex(cellIndex);
      cellIndex++;
    }
    _physicalCellNodesForElementType[elemType.get()] = physicalCellNodes;
    _cellSideParitiesForElementType[elemType.get()] = cellSideParities;
  }  
}

void Mesh::determineDofPairings() {
  _dofPairingIndex.clear();
  vector<ElementPtr>::iterator elemIterator;
  
  int globalIndex = 0;
  vector< int > trialIDs = _bilinearForm->trialIDs();
  
  for (elemIterator = _activeElements.begin(); elemIterator != _activeElements.end(); elemIterator++) {
    ElementPtr elemPtr = *(elemIterator);
    ElementTypePtr elemTypePtr = elemPtr->elementType();
    int cellID = elemPtr->cellID();
    if ( elemPtr->isParent() ) {
      TEST_FOR_EXCEPTION(true,std::invalid_argument,"elemPtr is in _activeElements, but is a parent...");
    }
    for (vector<int>::iterator trialIt=trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *(trialIt);
      
      if (_bilinearForm->isFluxOrTrace(trialID) ) {
        int numSides = elemPtr->numSides();
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
          Element* neighborPtr;
          int mySideIndexInNeighbor;
          elemPtr->getNeighbor(neighborPtr, mySideIndexInNeighbor, sideIndex);
          
          int neighborCellID = neighborPtr->cellID(); // may be -1 if it's the boundary
          if (neighborCellID != -1) {
            Teuchos::RCP<Element> neighbor = _elements[neighborCellID];
            // check that the bases agree in #dofs:
            int neighborNumDofs = neighbor->elementType()->trialOrderPtr->getBasisCardinality(trialID,mySideIndexInNeighbor);
            
            if ( !neighbor->isParent() && (numDofs != neighborNumDofs) ) { // neither a multi-basis, and we differ: a problem
              TEST_FOR_EXCEPTION(numDofs != neighborNumDofs,
                                 std::invalid_argument,
                                 "Element and neighbor don't agree on basis along shared side.");              
            }
            numDofs = min(neighborNumDofs,numDofs); // if there IS a multi-basis, we match the smaller basis with it...
            
            // Here, we need to deal with the possibility that neighbor is a parent, broken along the shared side
            //  -- if so, we have a MultiBasis, and we need to match with each of neighbor's descendents along that side...
            vector< pair<int,int> > descendentsForSide = neighbor->getDescendentsForSide(mySideIndexInNeighbor);
            vector< pair<int,int> >:: iterator entryIt;
            int descendentIndex = -1;
            for (entryIt = descendentsForSide.begin(); entryIt != descendentsForSide.end(); entryIt++) {
              descendentIndex++;
              int descendentSubSideIndexInMe = neighborChildPermutation(descendentIndex, descendentsForSide.size());
              neighborCellID = (*entryIt).first;
              mySideIndexInNeighbor = (*entryIt).second;
              neighbor = _elements[neighborCellID];
              for (int dofOrdinal=0; dofOrdinal<numDofs; dofOrdinal++) {
                int myLocalDofIndex;
                if (descendentsForSide.size() > 1) {
                  // multi-basis
                  myLocalDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex,descendentSubSideIndexInMe);
                } else {
                  myLocalDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex);
                }
                
                // neighbor's dofs are in reverse order from mine along each side
                // TODO: generalize this to some sort of permutation for 3D meshes...
                int permutedDofOrdinal = neighborDofPermutation(dofOrdinal,numDofs);

                int neighborLocalDofIndex = neighbor->elementType()->trialOrderPtr->getDofIndex(trialID,permutedDofOrdinal,mySideIndexInNeighbor);
                addDofPairing(cellID, myLocalDofIndex, neighborCellID, neighborLocalDofIndex);
//                cout << "added DofPairing (cellID, localDofIndex): (" << cellID << ", " << myLocalDofIndex << ") = ";
//                cout << "(" << neighborCellID << ", " << neighborLocalDofIndex << ")\n";
              }
            }
          }
        }
      }
    }
  }
  // now that we have all the dofPairings, reduce the map so that all the pairings point at the earliest paired index
  for (elemIterator = _activeElements.begin(); elemIterator != _activeElements.end(); elemIterator++) {
    ElementPtr elemPtr = *(elemIterator);
    int cellID = elemPtr->cellID();
    ElementTypePtr elemTypePtr = elemPtr->elementType();
    for (vector<int>::iterator trialIt=trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *(trialIt);
      if (_bilinearForm->isFluxOrTrace(trialID) ) {
        int numSides = elemPtr->numSides();
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
          for (int dofOrdinal=0; dofOrdinal<numDofs; dofOrdinal++) {
            int myLocalDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex);
            pair<int, int> myKey = make_pair(cellID,myLocalDofIndex);
            if (_dofPairingIndex.find(myKey) != _dofPairingIndex.end()) {
              pair<int, int> myValue = _dofPairingIndex[myKey];
              while (_dofPairingIndex.find(myValue) != _dofPairingIndex.end()) {
                myValue = _dofPairingIndex[myValue];
              }
              _dofPairingIndex[myKey] = myValue;
            }
          }
        }
      }
    }
  }  
}

BilinearForm & Mesh::bilinearForm() { 
  return *(_bilinearForm.get()); 
}

Boundary & Mesh::boundary() { 
  return _boundary; 
}

int Mesh::cellID(Teuchos::RCP< ElementType > elemTypePtr, int cellIndex) {
  return _cellIDsForElementType[elemTypePtr.get()][cellIndex];
}

FieldContainer<double> & Mesh::cellSideParities( ElementTypePtr elemTypePtr ) {
  return _cellSideParitiesForElementType[ elemTypePtr.get() ];  
}

void Mesh::determineActiveElements() {
  _activeElements.clear();
  vector<ElementPtr>::iterator elemIterator;
  
  for (elemIterator = _elements.begin(); elemIterator != _elements.end(); elemIterator++) {
    ElementPtr elemPtr = *(elemIterator);
    if ( ! elemPtr->isParent() ) {
      _activeElements.push_back(elemPtr);
    }
  }
  _partitions.clear();
  FieldContainer<int> partitionedMesh(_numPartitions,_activeElements.size());
  _partitionPolicy->partitionMesh(this,_numPartitions,partitionedMesh);
  for (int i=0; i<_numPartitions; i++) {
    vector<ElementPtr> partition;
    for (int j=0; j<_activeElements.size(); j++) {
      if (partitionedMesh(i,j) < 0) break; // no more elements in this partition
      partition.push_back( _elements[partitionedMesh(i,j)] );
    }
    _partitions.push_back( partition );
  }
}

vector< Teuchos::RCP< Element > > & Mesh::elements() { 
  return _elements; 
}

vector< Teuchos::RCP< ElementType > > Mesh::elementTypes() {
  return _elementTypes;
}

DofOrderingFactory & Mesh::getDofOrderingFactory() {
  return _dofOrderingFactory;
}

ElementTypeFactory & Mesh::getElementTypeFactory() {
  return _elementTypeFactory;
}

int Mesh::globalDofIndex(int cellID, int localDofIndex) {
  pair<int,int> key = make_pair(cellID, localDofIndex);
  map< pair<int,int>, int >::iterator mapEntryIt = _localToGlobalMap.find(key);
  if ( mapEntryIt == _localToGlobalMap.end() ) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "entry not found.");
  }
  return (*mapEntryIt).second;
}

void Mesh::hRefine(vector<int> cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
  vector<int>::iterator cellIt;
  for (cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    int cellID = *cellIt;
    ElementPtr elem = _elements[cellID];
    ElementTypePtr elemType = elem->elementType();
    
    int spaceDim = elemType->cellTopoPtr->getDimension();
    int numSides = elemType->cellTopoPtr->getSideCount();
    
    FieldContainer<double> cellNodes(numSides,spaceDim);
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      for (int i=0; i<spaceDim; i++) {
        cellNodes(sideIndex,i) = _vertices[_verticesForCellID[cellID][sideIndex]](i);
      }
    }
    
    FieldContainer<double> vertices = refPattern->verticesForRefinement(cellNodes);
    
    map<int, int> localToGlobalVertexIndex; // key: index in vertices; value: index in _vertices
    double tol = 1e-14; // tolerance for vertex equality
    
    int numVertices = vertices.dimension(0);
    for (int i=0; i<numVertices; i++) {
      int globalVertexIndex = -1;
      // TODO: we should probably split this out into a separate method for adding vertices...
      //  the below allows approximate matches to vertex
      //       ( _verticesMap keys are floats and values are a vector of ints -- _vertices indices.
      //        We iterate through these, looking for close-enough matches.  If none are close enough,
      //        add a vertex...  We could make an optional argument with a tolerance...)
      vector<float> vertexFloat(spaceDim);
      for (int dim=0; dim<spaceDim; dim++) {
        vertexFloat[dim] = (float) vertices(i,dim);
      }
      
      if ( _verticesMap.find(vertexFloat) != _verticesMap.end() ) {
        vector<int> vertexIndices = _verticesMap[vertexFloat];
        vector<int>::iterator vertexIndexIt;
        double minDistance = tol;
        int bestIndex = -1;
        for (vertexIndexIt=vertexIndices.begin(); vertexIndexIt != vertexIndices.end(); vertexIndexIt++) {
          double distance = 0.0;
          int vertexIndex = *vertexIndexIt;
          for (int dim=0; dim<spaceDim; dim++) {
            distance += (_vertices[vertexIndex](dim)-vertices(i,dim))*(_vertices[vertexIndex](dim)-vertices(i,dim));
          }
          distance = sqrt(distance);
          if (distance < minDistance) {
            bestIndex = vertexIndex;
            minDistance = distance;
          }
        }
        globalVertexIndex = bestIndex;
      }
      if (globalVertexIndex == -1) {
        // add the vertex
        FieldContainer<double> vertexFC(spaceDim);
        vector<double>::iterator coordIt;
        for (int j=0; j<spaceDim; j++) {
          vertexFC[j] = vertices(i,j);
        }
        globalVertexIndex = _vertices.size();
//        cout << "Adding vertex: (" << vertexFC(0) << ", " << vertexFC(1) << ")\n";
        _vertices.push_back(vertexFC);
        _verticesMap[vertexFloat].push_back(globalVertexIndex);
      }
      localToGlobalVertexIndex[i] = globalVertexIndex;
    }
    
    // get the children, as vectors of vertex indices:
    vector< vector<int> > children = refPattern->children(localToGlobalVertexIndex);
    
    // get the (child, child side) pairs for each side of the parent
    vector< vector< pair< int, int> > > childrenForSides = refPattern->childrenForSides(); // outer vector: indexed by parent's sides; inner vector: (child index in children, index of child's side shared with parent)
    
    _elements[cellID]->setRefinementPattern(refPattern);
    addChildren(_elements[cellID],children,childrenForSides);
  }
  rebuildLookups();
}

int Mesh::neighborChildPermutation(int childIndex, int numChildrenInSide) {
  // we'll need to be more sophisticated in 3D, but for now we just reverse the order
  return numChildrenInSide - childIndex - 1;
}

int Mesh::neighborDofPermutation(int dofIndex, int numDofsForSide) {
  // we'll need to be more sophisticated in 3D, but for now we just reverse the order
  return numDofsForSide - dofIndex - 1;
}

map< int, BasisPtr > Mesh::multiBasisUpgradeMap(ElementPtr parent, int sideIndex) {
  vector< pair< int, int> > childrenForSide = parent->childIndicesForSide(sideIndex);
  map< int, BasisPtr > varIDsToUpgrade;
  vector< map< int, BasisPtr > > childVarIDsToUpgrade;
  vector< pair< int, int> >::iterator entryIt;
  for ( entryIt=childrenForSide.begin(); entryIt != childrenForSide.end(); entryIt++) {
    int childCellIndex = (*entryIt).first;
    int childSideIndex = (*entryIt).second;
    ElementPtr childCell = parent->getChild(childCellIndex);
    DofOrderingPtr childTrialOrder = childCell->elementType()->trialOrderPtr;
    
    if ( childCell->isParent() && (childCell->childIndicesForSide(childSideIndex).size() > 1)) {
      childVarIDsToUpgrade.push_back( multiBasisUpgradeMap(childCell,childSideIndex) );
    } else {
      pair< DofOrderingPtr,int > entry = make_pair(childTrialOrder,childSideIndex);
      vector< pair< DofOrderingPtr,int > > childTrialOrdersForSide;
      childTrialOrdersForSide.push_back(entry);
      childVarIDsToUpgrade.push_back( _dofOrderingFactory.getMultiBasisUpgradeMap(childTrialOrdersForSide) );
    }
  }
  map< int, BasisPtr >::iterator varMapIt;
  for (varMapIt=childVarIDsToUpgrade[0].begin(); varMapIt != childVarIDsToUpgrade[0].end(); varMapIt++) {
    int varID = (*varMapIt).first;
    vector< BasisPtr > bases;
    int numChildrenInSide = childVarIDsToUpgrade.size();
    for (int childIndex = 0; childIndex<numChildrenInSide; childIndex++) {
      //permute child index (this is from neighbor's point of view)
      int permutedChildIndex = neighborChildPermutation(childIndex,numChildrenInSide);
      if (! childVarIDsToUpgrade[permutedChildIndex][varID].get()) {
        TEST_FOR_EXCEPTION(true, std::invalid_argument, "null basis");
      }
      bases.push_back(childVarIDsToUpgrade[permutedChildIndex][varID]);
    }
    BasisPtr multiBasis = BasisFactory::getMultiBasis(bases);
    varIDsToUpgrade[varID] = multiBasis;
  }
  return varIDsToUpgrade;
}

void Mesh::getMultiBasisOrdering(DofOrderingPtr &originalNonParentOrdering,
                                 ElementPtr parent, int sideIndex, int parentSideIndexInNeighbor,
                                 ElementPtr nonParent) {
  map< int, BasisPtr > varIDsToUpgrade = multiBasisUpgradeMap(parent,sideIndex);
  originalNonParentOrdering = _dofOrderingFactory.upgradeSide(originalNonParentOrdering,
                                                              *(nonParent->elementType()->cellTopoPtr),
                                                              varIDsToUpgrade,parentSideIndexInNeighbor);
}

void Mesh::matchNeighbor(const ElementPtr &elem, int sideIndex) {
  // sets new ElementType to match elem to neighbor on side sideIndex
  
  const shards::CellTopology cellTopo = *(elem->elementType()->cellTopoPtr.get());
  Element* neighbor;
  int mySideIndexInNeighbor;
  elem->getNeighbor(neighbor, mySideIndexInNeighbor, sideIndex);
  int neighborCellID = neighbor->cellID(); // may be -1 if it's the boundary
  if (neighborCellID < 0) {
    return; // no change
  }
  ElementPtr neighborRCP = _elements[neighborCellID];
  bool matchPOrder = true;
  // h-refinement handling:
  bool neighborIsBroken = (neighbor->isParent() && (neighbor->childIndicesForSide(mySideIndexInNeighbor).size() > 1));
  bool elementIsBroken  = (elem->isParent() && (elem->childIndicesForSide(sideIndex).size() > 1));
  if ( neighborIsBroken || elementIsBroken ) {
    bool bothBroken = ( neighborIsBroken && elementIsBroken );
    ElementPtr nonParent, parent; // for the case that one is a parent and the other isn't
    int parentSideIndexInNeighbor, neighborSideIndexInParent;
    if ( !bothBroken ) {
      if (! elementIsBroken ) {
        nonParent = elem;
        parent = neighborRCP;
        parentSideIndexInNeighbor = sideIndex;
        neighborSideIndexInParent = mySideIndexInNeighbor;
      } else {
        nonParent = neighborRCP;
        parent = elem;
        parentSideIndexInNeighbor = mySideIndexInNeighbor;
        neighborSideIndexInParent = sideIndex;
      }
    }
    
    if (bothBroken) {
      // match all the children -- we assume RefinementPatterns are compatible (e.g. divisions always by 1/2s)
      vector< pair<int,int> > childrenForSide = elem->childIndicesForSide(sideIndex);
      for (int childIndexInSide=0; childIndexInSide < childrenForSide.size(); childIndexInSide++) {
        int childIndex = childrenForSide[childIndexInSide].first;
        int childSideIndex = childrenForSide[childIndexInSide].second;
        matchNeighbor(elem->getChild(childIndex),childSideIndex);
      }
      // all our children matched => we're done:
      return;
    } else {
      vector< pair< int, int> > childrenForSide = parent->childIndicesForSide(neighborSideIndexInParent);
      
      Teuchos::RCP<DofOrdering> nonParentTrialOrdering = nonParent->elementType()->trialOrderPtr;
      
      if ( childrenForSide.size() > 1 ) { // then parent is broken along side, and neighbor isn't...
        ElementTypePtr nonParentType;
        if ( !_usePatchBasis ) {
          getMultiBasisOrdering( nonParentTrialOrdering, parent, neighborSideIndexInParent,
                                parentSideIndexInNeighbor, nonParent );
          nonParentType = _elementTypeFactory.getElementType(nonParentTrialOrdering, 
                                                                            nonParent->elementType()->testOrderPtr, 
                                                                            nonParent->elementType()->cellTopoPtr );
          // debug code:
          if ( ! _dofOrderingFactory.sideHasMultiBasis(nonParentTrialOrdering, parentSideIndexInNeighbor) ) {
            TEST_FOR_EXCEPTION(true, std::invalid_argument, "failed to add multi-basis to neighbor");
          }
        } else {
          TEST_FOR_EXCEPTION(true, std::invalid_argument, "Need to add PatchBasis creation to Mesh.");
        }
        nonParent->setElementType(nonParentType);
        
        vector< pair< int, int> > descendentsForSide = parent->getDescendentsForSide(neighborSideIndexInParent);
        
        vector< pair< int, int> >::iterator entryIt;
        for ( entryIt=descendentsForSide.begin(); entryIt != descendentsForSide.end(); entryIt++) {
          int childCellID = (*entryIt).first;
          int childSideIndex = (*entryIt).second;
          _boundary.deleteElement(childCellID, childSideIndex);
        }
        // by virtue of having assigned the multi-basis, we've already matched p-order ==> we're done
        return;
      }
    }
  }
  // p-refinement handling:
  const shards::CellTopology neighborTopo = *(neighbor->elementType()->cellTopoPtr.get());
  Teuchos::RCP<DofOrdering> elemTrialOrdering = elem->elementType()->trialOrderPtr;
  Teuchos::RCP<DofOrdering> elemTestOrdering = elem->elementType()->testOrderPtr;
  
  Teuchos::RCP<DofOrdering> neighborTrialOrdering = neighbor->elementType()->trialOrderPtr;
  Teuchos::RCP<DofOrdering> neighborTestOrdering = neighbor->elementType()->testOrderPtr;
  
  int changed = _dofOrderingFactory.matchSides(elemTrialOrdering, sideIndex, cellTopo,
                                               neighborTrialOrdering, mySideIndexInNeighbor, neighborTopo);
  // changed == 1 for me, 2 for neighbor, 0 for neither
  if (changed==1) {
    TEST_FOR_EXCEPTION(_bilinearForm->trialBoundaryIDs().size() == 0,
                       std::invalid_argument,
                       "BilinearForm has no traces or fluxes, but somehow element was upgraded...");
    int boundaryVarID = _bilinearForm->trialBoundaryIDs()[0];
    int neighborSidePolyOrder = BasisFactory::basisPolyOrder(neighborTrialOrdering->getBasis(boundaryVarID,mySideIndexInNeighbor));
    int mySidePolyOrder = BasisFactory::basisPolyOrder(elemTrialOrdering->getBasis(boundaryVarID,sideIndex));
    TEST_FOR_EXCEPTION(mySidePolyOrder != neighborSidePolyOrder,
                       std::invalid_argument,
                       "After matchSides(), the appropriate sides don't have the same order.");
    int testPolyOrder = _dofOrderingFactory.polyOrder(elemTestOrdering);
    if (testPolyOrder < mySidePolyOrder + _pToAddToTest) {
      elemTestOrdering = _dofOrderingFactory.testOrdering( mySidePolyOrder + _pToAddToTest, cellTopo);
    }
    elem->setElementType( _elementTypeFactory.getElementType(elemTrialOrdering, elemTestOrdering, 
                                                             elem->elementType()->cellTopoPtr ) );
    //return ELEMENT_NEEDED_NEW;
  } else if (changed==2) {
    // if need be, upgrade neighborTestOrdering as well.
    TEST_FOR_EXCEPTION(_bilinearForm->trialBoundaryIDs().size() == 0,
                       std::invalid_argument,
                       "BilinearForm has no traces or fluxes, but somehow neighbor was upgraded...");
    TEST_FOR_EXCEPTION(neighborTrialOrdering.get() == neighbor->elementType()->trialOrderPtr.get(),
                       std::invalid_argument,
                       "neighborTrialOrdering was supposed to be upgraded, but remains unchanged...");
    int boundaryVarID = _bilinearForm->trialBoundaryIDs()[0];
    int sidePolyOrder = BasisFactory::basisPolyOrder(neighborTrialOrdering->getBasis(boundaryVarID,mySideIndexInNeighbor));
    int mySidePolyOrder = BasisFactory::basisPolyOrder(elemTrialOrdering->getBasis(boundaryVarID,sideIndex));
    TEST_FOR_EXCEPTION(mySidePolyOrder != sidePolyOrder,
                       std::invalid_argument,
                       "After matchSides(), the appropriate sides don't have the same order.");
    int testPolyOrder = _dofOrderingFactory.polyOrder(neighborTestOrdering);
    if (testPolyOrder < sidePolyOrder + _pToAddToTest) {
      neighborTestOrdering = _dofOrderingFactory.testOrdering( sidePolyOrder + _pToAddToTest, neighborTopo);
    }
    neighbor->setElementType( _elementTypeFactory.getElementType(neighborTrialOrdering, neighborTestOrdering, 
                                                                 neighbor->elementType()->cellTopoPtr ) );
    //return NEIGHBOR_NEEDED_NEW;
  } else {
    //return NEITHER_NEEDED_NEW;
  }
}

int Mesh::numElements() {
  return _elements.size();
}

int Mesh::numElementsOfType( Teuchos::RCP< ElementType > elemTypePtr ) {
  return _physicalCellNodesForElementType[ elemTypePtr.get() ].dimension(0);
}

int Mesh::numGlobalDofs() {
  return _numGlobalDofs;
}

int Mesh::parityForSide(int cellID, int sideIndex) {
  ElementPtr elem = _elements[cellID];
  if ( elem->isParent() ) {
    // then it's not an active element, so we should use _cellSideParitiesForCellID
    // (we could always use this, but for tests I'd rather expose what's actually used in computation)
    return _cellSideParitiesForCellID[cellID][sideIndex];
  }
  // if we get here, then we have an active element...
  ElementTypePtr elemType = elem->elementType();
  int cellIndex = elem->cellIndex();
  int parity = _cellSideParitiesForElementType[elemType.get()](cellIndex,sideIndex);
  if (_cellSideParitiesForCellID[cellID][sideIndex] != parity ) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "parity lookups don't match");
  }
  return parity;
}

FieldContainer<double> & Mesh::physicalCellNodes( Teuchos::RCP< ElementType > elemTypePtr ) {
  return _physicalCellNodesForElementType[ elemTypePtr.get() ];
}

void Mesh::rebuildLookups() {
  determineActiveElements();
  buildTypeLookups(); // build data structures for efficient lookup by element type
  buildLocalToGlobalMap();
  _boundary.buildLookupTables();
  cout << "Mesh.numGlobalDofs: " << numGlobalDofs() << endl;
}

// the following is not at all meant to be efficient; we do a lot of rebuilding of
// data structures...
void Mesh::refine(vector<int> cellIDsForPRefinements, vector<int> cellIDsForHRefinements) {
  TEST_FOR_EXCEPTION( ( cellIDsForHRefinements.size() > 0 ),
                     std::invalid_argument,
                     "h-refinements not yet supported.");
  // TODO: Implement h-refinements...
  
  // p-refinements:
  // 1. Loop through cellIDsForPRefinements:
  //   a. create new DofOrderings for trial and test
  //   b. create new element type, and store for later
  //   c. Loop through neighbors, calling dofOrderingFactory.matchSides to sync edges and creating/recording new types
  
  // 1. Loop through cellIDsForPRefinements:
  vector<int>::iterator cellIt;
  for (cellIt=cellIDsForPRefinements.begin(); cellIt != cellIDsForPRefinements.end(); cellIt++) {
    int cellID = *cellIt;
    ElementPtr elem = _elements[cellID];
    const shards::CellTopology cellTopo = *(elem->elementType()->cellTopoPtr.get());
    //   a. create new DofOrderings for trial and test
    Teuchos::RCP<DofOrdering> currentTrialOrdering, currentTestOrdering;
    currentTrialOrdering = elem->elementType()->trialOrderPtr;
    currentTestOrdering  = elem->elementType()->testOrderPtr;
    Teuchos::RCP<DofOrdering> newTrialOrdering = _dofOrderingFactory.pRefine(currentTrialOrdering,
                                                                             cellTopo);
    Teuchos::RCP<DofOrdering> newTestOrdering;
    // determine what newTestOrdering should be:
    int trialPolyOrder = _dofOrderingFactory.polyOrder(newTrialOrdering);
    int testPolyOrder = _dofOrderingFactory.polyOrder(currentTestOrdering);
    if (testPolyOrder < trialPolyOrder + _pToAddToTest) {
      newTestOrdering = _dofOrderingFactory.testOrdering( trialPolyOrder + _pToAddToTest, cellTopo);
    } else {
      newTestOrdering = currentTestOrdering;
    }
    
    //   c. Loop through neighbors, calling dofOrderingFactory.matchSides to sync edges and creating/recording new types
    int numSides = elem->numSides();
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      // TODO: should be able to replace this whole loop with a single call to matchNeighbor
      // (WILL FIRST NEED TO SET elem->elementType so that it includes newTest/TrialOrdering...)
      Element* neighbor;
      int mySideIndexInNeighbor;
      elem->getNeighbor(neighbor, mySideIndexInNeighbor, sideIndex);
      int neighborCellID = neighbor->cellID(); // may be -1 if it's the boundary
      const shards::CellTopology neighborTopo = *(neighbor->elementType()->cellTopoPtr.get());
      if (neighborCellID >= 0) {
        Teuchos::RCP<DofOrdering> neighborTrialOrdering;
        Teuchos::RCP<DofOrdering> neighborTestOrdering; 
        neighborTrialOrdering = neighbor->elementType()->trialOrderPtr;
        neighborTestOrdering  = neighbor->elementType()->testOrderPtr;
        int changed = _dofOrderingFactory.matchSides(newTrialOrdering, sideIndex, cellTopo,
                                                     neighborTrialOrdering, mySideIndexInNeighbor, neighborTopo);
        // changed == 1 for me, 2 for neighbor, 0 for neither
        if (changed==1) {
          TEST_FOR_EXCEPTION(_bilinearForm->trialBoundaryIDs().size() == 0,
                             std::invalid_argument,
                             "BilinearForm has no traces or fluxes, but somehow element was upgraded...");
          int boundaryVarID = _bilinearForm->trialBoundaryIDs()[0];
          int neighborSidePolyOrder = BasisFactory::basisPolyOrder(neighborTrialOrdering->getBasis(boundaryVarID,mySideIndexInNeighbor));
          int mySidePolyOrder = BasisFactory::basisPolyOrder(newTrialOrdering->getBasis(boundaryVarID,sideIndex));
          TEST_FOR_EXCEPTION(mySidePolyOrder != neighborSidePolyOrder,
                             std::invalid_argument,
                             "After matchSides(), the appropriate sides don't have the same order.");
          int testPolyOrder = _dofOrderingFactory.polyOrder(newTestOrdering);
          if (testPolyOrder < mySidePolyOrder + _pToAddToTest) {
            newTestOrdering = _dofOrderingFactory.testOrdering( mySidePolyOrder + _pToAddToTest, cellTopo);
          }
        }
        if (changed==2) {
          // if need be, upgrade neighborTestOrdering as well.
          TEST_FOR_EXCEPTION(_bilinearForm->trialBoundaryIDs().size() == 0,
                             std::invalid_argument,
                             "BilinearForm has no traces or fluxes, but somehow neighbor was upgraded...");
          TEST_FOR_EXCEPTION(neighborTrialOrdering.get() == neighbor->elementType()->trialOrderPtr.get(),
                             std::invalid_argument,
                             "neighborTrialOrdering was supposed to be upgraded, but remains unchanged...");
          int boundaryVarID = _bilinearForm->trialBoundaryIDs()[0];
          int sidePolyOrder = BasisFactory::basisPolyOrder(neighborTrialOrdering->getBasis(boundaryVarID,mySideIndexInNeighbor));
          int mySidePolyOrder = BasisFactory::basisPolyOrder(newTrialOrdering->getBasis(boundaryVarID,sideIndex));
          TEST_FOR_EXCEPTION(mySidePolyOrder != sidePolyOrder,
                             std::invalid_argument,
                             "After matchSides(), the appropriate sides don't have the same order.");
          int testPolyOrder = _dofOrderingFactory.polyOrder(neighborTestOrdering);
          if (testPolyOrder < sidePolyOrder + _pToAddToTest) {
            neighborTestOrdering = _dofOrderingFactory.testOrdering( sidePolyOrder + _pToAddToTest, neighborTopo);
          }
          neighbor->setElementType( _elementTypeFactory.getElementType(neighborTrialOrdering, neighborTestOrdering, 
                                                                       neighbor->elementType()->cellTopoPtr ) );
        }
      }
    }
    //   b. create new element type
    elem->setElementType( _elementTypeFactory.getElementType(newTrialOrdering, newTestOrdering, 
                                                             elem->elementType()->cellTopoPtr ) );
  }
  rebuildLookups();
}

void Mesh::repartition() {
  rebuildLookups();
}

int Mesh::rowSizeUpperBound() {
  // includes multiplicity
  vector< Teuchos::RCP< ElementType > >::iterator elemTypeIt;
  int maxRowSize = 0;
  for (elemTypeIt = _elementTypes.begin(); elemTypeIt != _elementTypes.end();
       elemTypeIt++) {
    ElementTypePtr elemTypePtr = *elemTypeIt;
    int numSides = elemTypePtr->cellTopoPtr->getSideCount();
    vector< int > fluxIDs = _bilinearForm->trialBoundaryIDs();
    vector< int >::iterator fluxIDIt;
    int numFluxDofs = 0;
    for (fluxIDIt = fluxIDs.begin(); fluxIDIt != fluxIDs.end(); fluxIDIt++) {
      int fluxID = *fluxIDIt;
      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
        int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(fluxID,sideIndex);
        numFluxDofs += numDofs;
      }
    }
    int numFieldDofs = elemTypePtr->trialOrderPtr->totalDofs() - numFluxDofs;
    int maxPossible = numFluxDofs * 2 + numSides*fluxIDs.size() + numFieldDofs;  // a side can be shared by 2 elements, and vertices can be shared
    maxRowSize = max(maxPossible, maxRowSize);
  }
  return maxRowSize;
}

void Mesh::setNeighbor(ElementPtr elemPtr, int elemSide, ElementPtr neighborPtr, int neighborSide) {
  elemPtr->setNeighbor(elemSide,neighborPtr,neighborSide);
  neighborPtr->setNeighbor(neighborSide,elemPtr,elemSide);
  double parity;
  if (neighborPtr->cellID() < 0) {
    // boundary
    parity = 1.0;
  } else if (elemPtr->cellID() < neighborPtr->cellID()) {
    parity = 1.0;
  } else {
    parity = -1.0;
  }
  _cellSideParitiesForCellID[elemPtr->cellID()][elemSide] = parity;
//  cout << "setNeighbor: set cellSideParity for cell " << elemPtr->cellID() << ", sideIndex " << elemSide << ": ";
//  cout << _cellSideParitiesForCellID[elemPtr->cellID()][elemSide] << endl;
  
  if (neighborPtr->cellID() > -1) {
    _cellSideParitiesForCellID[neighborPtr->cellID()][neighborSide] = -parity;
//    cout << "setNeighbor: set cellSideParity for cell " << neighborPtr->cellID() << ", sideIndex " << neighborSide << ": ";
//    cout << _cellSideParitiesForCellID[neighborPtr->cellID()][neighborSide] << endl;
  }
//  cout << "set cellID " << elemPtr->cellID() << "'s neighbor for side ";
//  cout << elemSide << " to cellID " << neighborPtr->cellID();
//  cout << " (neighbor's sideIndex: " << neighborSide << ")" << endl;
}

void Mesh::setNumPartitions(int numPartitions) {
  _numPartitions = numPartitions;
}

void Mesh::setPartitionPolicy(  Teuchos::RCP< MeshPartitionPolicy > partitionPolicy ) {
  _partitionPolicy = partitionPolicy;
}

void Mesh::verticesForCell(FieldContainer<double>& vertices, int cellID) {
  vector<int> vertexIndices = _verticesForCellID[cellID];
  ElementTypePtr elemType = _elements[cellID]->elementType();
  int dimension = elemType->cellTopoPtr->getDimension();
  int numVertices = elemType->cellTopoPtr->getVertexCount(dimension,0);
  vertices.resize(numVertices,dimension);
  for (int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
    for (int i=0; i<dimension; i++) {
      vertices(vertexIndex,i) = _vertices[vertexIndices[vertexIndex]](i);
    }
  }
}

void Mesh::verticesForElementType(FieldContainer<double>& vertices, ElementTypePtr elemTypePtr) {
  int dimension = elemTypePtr->cellTopoPtr->getDimension();
  int numVertices = elemTypePtr->cellTopoPtr->getVertexCount(dimension,0);
  int numCells = numElementsOfType(elemTypePtr);
  vertices.resize(numCells,numVertices,dimension);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    int cellID = this->cellID(elemTypePtr,cellIndex);
    for (int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
      for (int i=0; i<dimension; i++) {
        vertices(cellIndex,vertexIndex,i) = _vertices[_verticesForCellID[cellID][vertexIndex]](i);
      }
    }
  }
}

void Mesh::verticesForSide(FieldContainer<double>& vertices, int cellID, int sideIndex) {
  ElementPtr elem = _elements[cellID];
  ElementTypePtr elemType = elem->elementType();
  int dimension = elemType->cellTopoPtr->getDimension();
  int numVertices = elemType->cellTopoPtr->getVertexCount(dimension-1,sideIndex);
  int numSides = elem->numSides();
  vertices.resize(numVertices,dimension);
  for (int vertexIndex = 0; vertexIndex < numVertices; vertexIndex++) {
    for (int i=0; i<dimension; i++) {
      vertices(vertexIndex,i) = _vertices[_verticesForCellID[cellID][(sideIndex+vertexIndex)%numSides]](i);
    }
  }
}