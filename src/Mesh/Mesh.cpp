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

#include "Solution.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"
#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#endif

using namespace Intrepid;

typedef Teuchos::RCP< ElementType > ElementTypePtr;
typedef Teuchos::RCP< Element > ElementPtr;
typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
typedef Teuchos::RCP< DofOrdering > DofOrderingPtr;

Mesh::Mesh(const vector<FieldContainer<double> > &vertices, vector< vector<int> > &elementVertices,
           Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pToAddTest) : _dofOrderingFactory(bilinearForm) {
  _vertices = vertices;
  _usePatchBasis = false;
  _enforceMBFluxContinuity = false;  
  _partitionPolicy = Teuchos::rcp( new MeshPartitionPolicy() );

#ifdef HAVE_MPI
  _numPartitions = Teuchos::GlobalMPISession::getNProc();
#else
  _numPartitions = 1;
#endif

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
  _numInitialElements = (int)numElements();
}

int Mesh::numInitialElements(){
  return _numInitialElements;
}

int Mesh::activeCellOffset() {
  return _activeCellOffset;
}

vector< Teuchos::RCP< Element > > & Mesh::activeElements() {
  return _activeElements;
}

Teuchos::RCP<Mesh> Mesh::readMsh(string filePath, Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pToAdd)
{
  ifstream mshFile;
  mshFile.open(filePath.c_str());
  TEST_FOR_EXCEPTION(mshFile == NULL, std::invalid_argument, "Could not open msh file");
  string line;
  getline(mshFile, line);
  while (line != "$Nodes")
  {
    getline(mshFile, line);
  }
  int numNodes;
  mshFile >> numNodes;
  vector<FieldContainer<double> > vertices;
  int dummy;
  for (int i=0; i < numNodes; i++)
  {
    FieldContainer<double> vertex(2);
    mshFile >> dummy;
    mshFile >> vertex(0) >> vertex(1) >> dummy;
    vertices.push_back(vertex);
  }
  while (line != "$Elements")
  {
    getline(mshFile, line);
  }
  int numElems;
  mshFile >> numElems;
  int elemType;
  int numTags;
  vector< vector<int> > elementIndices;
  for (int i=0; i < numElems; i++)
  {
    mshFile >> dummy >> elemType >> numTags;
    for (int j=0; j < numTags; j++)
      mshFile >> dummy;
    if (elemType == 2)
    {
      vector<int> elemIndices(3);
      mshFile >> elemIndices[0] >> elemIndices[1] >> elemIndices[2];
      elemIndices[0]--;
      elemIndices[1]--;
      elemIndices[2]--;
      elementIndices.push_back(elemIndices);
    }
    if (elemType == 4)
    {
      vector<int> elemIndices(3);
      mshFile >> elemIndices[0] >> elemIndices[1] >> elemIndices[2];
      elemIndices[0]--;
      elemIndices[1]--;
      elemIndices[2]--;
      elementIndices.push_back(elemIndices);
    }
    else
    {
      getline(mshFile, line);
    }
  }
  mshFile.close();

  Teuchos::RCP<Mesh> mesh;
  // // L-shaped domain for double ramp problem
  // FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2);
  // A(0) = 0.0; A(1) = 0.5;
  // B(0) = 0.0; B(1) = 1.0;
  // C(0) = 0.5; C(1) = 1.0;
  // D(0) = 1.0; D(1) = 1.0;
  // E(0) = 1.0; E(1) = 0.5;
  // F(0) = 1.0; F(1) = 0.0;
  // G(0) = 0.5; G(1) = 0.0;
  // H(0) = 0.5; H(1) = 0.5;
  // vector<FieldContainer<double> > fake_vertices;
  // fake_vertices.push_back(A); int A_index = 0;
  // fake_vertices.push_back(B); int B_index = 1;
  // fake_vertices.push_back(C); int C_index = 2;
  // fake_vertices.push_back(D); int D_index = 3;
  // fake_vertices.push_back(E); int E_index = 4;
  // fake_vertices.push_back(F); int F_index = 5;
  // fake_vertices.push_back(G); int G_index = 6;
  // fake_vertices.push_back(H); int H_index = 7;
  // vector< vector<int> > elementVertices;
  // vector<int> el1, el2, el3, el4, el5;
  // // left patch:
  // el1.push_back(A_index); el1.push_back(H_index); el1.push_back(C_index); el1.push_back(B_index);
  // // top right:
  // el2.push_back(H_index); el2.push_back(E_index); el2.push_back(D_index); el2.push_back(C_index);
  // // bottom right:
  // el3.push_back(G_index); el3.push_back(F_index); el3.push_back(E_index); el3.push_back(H_index);

  // elementVertices.push_back(el1);
  // elementVertices.push_back(el2);
  // elementVertices.push_back(el3);
  
  mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, bilinearForm, H1Order, pToAdd) );  
  return mesh;
}

Teuchos::RCP<Mesh> Mesh::buildQuadMesh(const FieldContainer<double> &quadBoundaryPoints, 
                                       int horizontalElements, int verticalElements,
                                       Teuchos::RCP< BilinearForm > bilinearForm, 
                                       int H1Order, int pTest, bool triangulate) {
//  if (triangulate) cout << "Mesh: Triangulating\n" << endl;
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
//      cout << "Mesh: vertex " << vertices.size() - 1 << ": (" << vertex(0) << "," << vertex(1) << ")\n";
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
    int testOrder = max(pTrial,maxDegree) + _pToAddToTest;
    
    childTestOrders[childIndex] = _dofOrderingFactory.testOrdering(testOrder, *(childTopos[childIndex]));
  }

  vector<int> childCellIDs;
  // create ElementTypePtr for each child, and add child to mesh...
  for (int childIndex=0; childIndex<numChildren; childIndex++) {
    int maxDegree = childTrialOrders[childIndex]->maxBasisDegree();
    int testOrder = max(pTrial,maxDegree) + _pToAddToTest;
    
    childTestOrders[childIndex] = _dofOrderingFactory.testOrdering(testOrder, *(childTopos[childIndex]));

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
  // the commented-out code here is an effort to move the logic for this into element, where it belongs
  // (but there's a bug either here or in ancestralNeighborCellIDForSide(sideIndex), which causes tests to
  //   fail with exceptions indicating that multi-basis isn't applied where it should be...)
//  pair<int,int> neighborCellAndSide = elem->ancestralNeighborCellIDForSide(sideIndex);
//  elemSideIndexInNeighbor = neighborCellAndSide.second;
//  int neighborCellID = neighborCellAndSide.first;
//  if (neighborCellID == -1) {
//    return _nullPtr;
//  } else {
//    return _elements[neighborCellID];
//  }
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
  _elementTypesForPartition.clear();
  _cellIDsForElementType.clear();
  _globalCellIndexToCellID.clear();
  _partitionedCellSideParitiesForElementType.clear();
  _partitionedPhysicalCellNodesForElementType.clear();
  set< ElementType* > elementTypeSet; // keep track of which ones we've seen globally (avoid duplicates in _elementTypes)
  map< ElementType*, int > globalCellIndices;
  
  for (int partitionNumber=0; partitionNumber < _numPartitions; partitionNumber++) {
    _cellIDsForElementType.push_back( map< ElementType*, vector<int> >() );
    _elementTypesForPartition.push_back( vector< ElementTypePtr >() );
    _partitionedPhysicalCellNodesForElementType.push_back( map< ElementType*, FieldContainer<double> >() );
    _partitionedCellSideParitiesForElementType.push_back( map< ElementType*, FieldContainer<double> >() );
    vector< ElementPtr >::iterator elemIterator;

    // this should loop over the elements in the partition instead
    for (elemIterator=_partitions[partitionNumber].begin(); 
         elemIterator != _partitions[partitionNumber].end(); elemIterator++) {
      ElementPtr elem = *elemIterator;
      ElementTypePtr elemTypePtr = elem->elementType();
      if ( _cellIDsForElementType[partitionNumber].find( elemTypePtr.get() ) == _cellIDsForElementType[partitionNumber].end() ) {
        _elementTypesForPartition[partitionNumber].push_back(elemTypePtr);
      }
      if (elementTypeSet.find( elemTypePtr.get() ) == elementTypeSet.end() ) {
        elementTypeSet.insert( elemTypePtr.get() );
        _elementTypes.push_back( elemTypePtr );
      }
      _cellIDsForElementType[partitionNumber][elemTypePtr.get()].push_back(elem->cellID());
    }
    
    // now, build cellSideParities and physicalCellNodes lookups
    vector< ElementTypePtr >::iterator elemTypeIt;
    for (elemTypeIt=_elementTypesForPartition[partitionNumber].begin(); elemTypeIt != _elementTypesForPartition[partitionNumber].end(); elemTypeIt++) {
      //ElementTypePtr elemType = _elementTypeFactory.getElementType((*elemTypeIt)->trialOrderPtr,
  //                                                                 (*elemTypeIt)->testOrderPtr,
  //                                                                 (*elemTypeIt)->cellTopoPtr);
      ElementTypePtr elemType = *elemTypeIt; // don't enforce uniquing here (if we wanted to, we
                                             //   would also need to call elem.setElementType for 
                                             //   affected elements...)
      int spaceDim = elemType->cellTopoPtr->getDimension();
      int numSides = elemType->cellTopoPtr->getSideCount();
      vector<int> cellIDs = _cellIDsForElementType[partitionNumber][elemType.get()];
      int numCells = cellIDs.size();
      FieldContainer<double> physicalCellNodes( numCells, numSides, spaceDim ) ;
      FieldContainer<double> cellSideParities( numCells, numSides );
      vector<int>::iterator cellIt;
      int cellIndex = 0;
      for (cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
        int cellID = *cellIt;
        ElementPtr elem = _elements[cellID];
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          for (int i=0; i<spaceDim; i++) {
            physicalCellNodes(cellIndex,sideIndex,i) = _vertices[_verticesForCellID[cellID][sideIndex]](i);
          }
          cellSideParities(cellIndex,sideIndex) = _cellSideParitiesForCellID[cellID][sideIndex];
        }
        elem->setCellIndex(cellIndex++);
        elem->setGlobalCellIndex(globalCellIndices[elemType.get()]++);
        _globalCellIndexToCellID[elemType.get()][elem->globalCellIndex()] = cellID;
        TEST_FOR_EXCEPTION( elem->cellID() != _globalCellIndexToCellID[elemType.get()][elem->globalCellIndex()],
                           std::invalid_argument, "globalCellIndex -> cellID inconsistency detected" );
      }
      _partitionedPhysicalCellNodesForElementType[partitionNumber][elemType.get()] = physicalCellNodes;
      _partitionedCellSideParitiesForElementType[partitionNumber][elemType.get()] = cellSideParities;
    }
  }
  // finally, build _physicalCellNodesForElementType and _cellSideParitiesForElementType:
  _physicalCellNodesForElementType.clear();
  for (vector< ElementTypePtr >::iterator elemTypeIt = _elementTypes.begin();
       elemTypeIt != _elementTypes.end(); elemTypeIt++) {
    ElementType* elemType = elemTypeIt->get();
    int numCells = globalCellIndices[elemType];
    int spaceDim = elemType->cellTopoPtr->getDimension();
    int numSides = elemType->cellTopoPtr->getSideCount();
    _physicalCellNodesForElementType[elemType] = FieldContainer<double>(numCells,numSides,spaceDim);
  }
  // copy from the local (per-partition) FieldContainers to the global ones
  for (int partitionNumber=0; partitionNumber < _numPartitions; partitionNumber++) {
    vector< ElementTypePtr >::iterator elemTypeIt;
    for (elemTypeIt  = _elementTypesForPartition[partitionNumber].begin(); 
         elemTypeIt != _elementTypesForPartition[partitionNumber].end(); elemTypeIt++) {
      ElementType* elemType = elemTypeIt->get();
      FieldContainer<double> partitionedPhysicalCellNodes = _partitionedPhysicalCellNodesForElementType[partitionNumber][elemType];
      FieldContainer<double> partitionedCellSideParities = _partitionedCellSideParitiesForElementType[partitionNumber][elemType];
      
      int numCells = partitionedPhysicalCellNodes.dimension(0);
      int numSides = partitionedPhysicalCellNodes.dimension(1);
      int spaceDim = partitionedPhysicalCellNodes.dimension(2);
      
      // this copying can be made more efficient by copying a whole FieldContainer at a time
      // (but it's probably not worth it, for now)
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        int cellID = _cellIDsForElementType[partitionNumber][elemType][cellIndex];
        int globalCellIndex = _elements[cellID]->globalCellIndex();
        TEST_FOR_EXCEPTION( cellID != _globalCellIndexToCellID[elemType][globalCellIndex],
                           std::invalid_argument, "globalCellIndex -> cellID inconsistency detected" );
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          for (int dim=0; dim<spaceDim; dim++) {
            _physicalCellNodesForElementType[elemType](globalCellIndex,sideIndex,dim) 
              = partitionedPhysicalCellNodes(cellIndex,sideIndex,dim);
          }
        }
      }
    }
  }
}

Teuchos::RCP<BilinearForm> Mesh::bilinearForm() { 
  return _bilinearForm; 
}

Boundary & Mesh::boundary() { 
  return _boundary; 
}

int Mesh::cellID(Teuchos::RCP< ElementType > elemTypePtr, int cellIndex, int partitionNumber) {
  if (partitionNumber == -1){ 
    if (( _globalCellIndexToCellID.find( elemTypePtr.get() ) != _globalCellIndexToCellID.end() )
        && 
        ( _globalCellIndexToCellID[elemTypePtr.get()].find( cellIndex )
         !=
          _globalCellIndexToCellID[elemTypePtr.get()].end()
         )
        )
      return _globalCellIndexToCellID[elemTypePtr.get()][ cellIndex ];
    else
      return -1;
  } else {
    if ( ( _cellIDsForElementType[partitionNumber].find( elemTypePtr.get() ) != _cellIDsForElementType[partitionNumber].end() )
        &&
         (_cellIDsForElementType[partitionNumber][elemTypePtr.get()].size() > cellIndex ) ) {
           return _cellIDsForElementType[partitionNumber][elemTypePtr.get()][cellIndex];
    } else return -1;
  }
}

int Mesh::cellPolyOrder(int cellID) {
  return _dofOrderingFactory.polyOrder(_elements[cellID]->elementType()->trialOrderPtr);
}

bool Mesh::colinear(double x0, double y0, double x1, double y1, double x2, double y2) {
  double tol = 1e-14;
  double d1 = distance(x0,y0,x1,y1);
  double d2 = distance(x1,y1,x2,y2);
  double d3 = distance(x2,y2,x0,y0);
  
  return (abs(d1 + d2 - d3) < tol) || (abs(d1 + d3 - d2) < tol) || (abs(d2 + d3 - d1) < tol);
}

void Mesh::determineDofPairings() {
  _dofPairingIndex.clear();
  vector<ElementPtr>::iterator elemIterator;
  
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
          int myNumDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
          Element* neighborPtr;
          int mySideIndexInNeighbor;
          elemPtr->getNeighbor(neighborPtr, mySideIndexInNeighbor, sideIndex);
          
          int neighborCellID = neighborPtr->cellID(); // may be -1 if it's the boundary
          if (neighborCellID != -1) {
            Teuchos::RCP<Element> neighbor = _elements[neighborCellID];
            // check that the bases agree in #dofs:
            
            bool hasMultiBasis = neighbor->isParent() && !_usePatchBasis;
            
            if ( ! neighbor->isParent() ) {
              int neighborNumDofs = neighbor->elementType()->trialOrderPtr->getBasisCardinality(trialID,mySideIndexInNeighbor);
              if ( !hasMultiBasis && (myNumDofs != neighborNumDofs) ) { // neither a multi-basis, and we differ: a problem
                TEST_FOR_EXCEPTION(myNumDofs != neighborNumDofs,
                                   std::invalid_argument,
                                   "Element and neighbor don't agree on basis along shared side.");              
              }
            }
            
            // Here, we need to deal with the possibility that neighbor is a parent, broken along the shared side
            //  -- if so, we have a MultiBasis, and we need to match with each of neighbor's descendants along that side...
            vector< pair<int,int> > descendantsForSide = neighbor->getDescendantsForSide(mySideIndexInNeighbor);
            vector< pair<int,int> >:: iterator entryIt;
            int descendantIndex = -1;
            for (entryIt = descendantsForSide.begin(); entryIt != descendantsForSide.end(); entryIt++) {
              descendantIndex++;
              int descendantSubSideIndexInMe = neighborChildPermutation(descendantIndex, descendantsForSide.size());
              neighborCellID = (*entryIt).first;
              mySideIndexInNeighbor = (*entryIt).second;
              neighbor = _elements[neighborCellID];
              int neighborNumDofs = neighbor->elementType()->trialOrderPtr->getBasisCardinality(trialID,mySideIndexInNeighbor);
              
              for (int dofOrdinal=0; dofOrdinal<neighborNumDofs; dofOrdinal++) {
                int myLocalDofIndex;
                if ( (descendantsForSide.size() > 1) && ( !_usePatchBasis ) ) {
                  // multi-basis
                  myLocalDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex,descendantSubSideIndexInMe);
                } else {
                  myLocalDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,dofOrdinal,sideIndex);
                }
                
                // neighbor's dofs are in reverse order from mine along each side
                // TODO: generalize this to some sort of permutation for 3D meshes...
                int permutedDofOrdinal = neighborDofPermutation(dofOrdinal,neighborNumDofs);
                
                int neighborLocalDofIndex = neighbor->elementType()->trialOrderPtr->getDofIndex(trialID,permutedDofOrdinal,mySideIndexInNeighbor);
                addDofPairing(cellID, myLocalDofIndex, neighborCellID, neighborLocalDofIndex);
              }
            }
            
            if ( hasMultiBasis && _enforceMBFluxContinuity ) {
              // marry the last node from one MB leaf to first node from the next
              // note that we're doing this for both traces and fluxes, but with traces this is redundant
              BasisPtr basis = elemTypePtr->trialOrderPtr->getBasis(trialID,sideIndex);
              MultiBasis* multiBasis = (MultiBasis *) basis.get();
              vector< pair<int,int> > adjacentDofOrdinals = multiBasis->adjacentVertexOrdinals();
              for (vector< pair<int,int> >::iterator dofPairIt = adjacentDofOrdinals.begin();
                   dofPairIt != adjacentDofOrdinals.end(); dofPairIt++) {
                int firstOrdinal  = dofPairIt->first;
                int secondOrdinal = dofPairIt->second;
                int firstDofIndex  = elemTypePtr->trialOrderPtr->getDofIndex(trialID,firstOrdinal, sideIndex);
                int secondDofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID,secondOrdinal,sideIndex);
                addDofPairing(cellID,firstDofIndex, cellID, secondDofIndex);
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

double Mesh::distance(double x0, double y0, double x1, double y1) {
  return sqrt( (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
}

vector<ElementPtr> Mesh::elementsForPoints(const FieldContainer<double> &physicalPoints) {
  // returns a vector of an active element per point, or null if there is no element including that point
  vector<ElementPtr> elemsForPoints;
//  cout << "entered elementsForPoints: \n" << physicalPoints;
  int numPoints = physicalPoints.dimension(0);
  // TODO: work out what to do here for 3D
  // figure out the last element of the original mesh:
  int lastCellID = 0;
  while ((_elements.size() > lastCellID) && ! _elements[lastCellID]->isChild()) {
    lastCellID++;
  }
  // NOTE: the above does depend on the domain of the mesh remaining fixed after refinements begin.
  
  for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
    double x = physicalPoints(pointIndex,0);
    double y = physicalPoints(pointIndex,1);
    // find the element from the original mesh that contains this point
    ElementPtr elem;
    for (int cellID = 0; cellID<lastCellID; cellID++) {
      if (elementContainsPoint(_elements[cellID],x,y)) {
        elem = _elements[cellID];
        break;
      }
    }
    if (elem.get() != NULL) {
      while ( elem->isParent() ) {
        int numChildren = elem->numChildren();
        bool foundMatchingChild = false;
        for (int childIndex = 0; childIndex < numChildren; childIndex++) {
          ElementPtr child = elem->getChild(childIndex);
          if ( elementContainsPoint(child,x,y) ) {
            elem = child;
            foundMatchingChild = true;
            break;
          }
        }
        if (!foundMatchingChild) {
          cout << "parent matches (" << x << ", " << y << "), but none of its children do...\n";
          TEST_FOR_EXCEPTION(true, std::invalid_argument, "parent matches, but none of its children do...");
        }
      }
    }
    elemsForPoints.push_back(elem);
  }
//  cout << "Returning from elementsForPoints\n";
  return elemsForPoints;
}

bool Mesh::elementContainsPoint(ElementPtr elem, double x, double y) {  
  // the following commented-out bit is the start of a more efficient version of this method:
//  vector<int> vertexIndices = _verticesForCellID[elem->cellID()];
//  int numVertices = vertexIndices.size();
//  
//  double maxX = _vertices[vertexIndices[0]](0);
//  double minX = maxX;
//  double maxY = _vertices[vertexIndices[0]](1);
//  double minY = maxY;
//  for (int vertexIndex=1; vertexIndex<numVertices; vertexIndex++) {
//    minX = min(minX,_vertices[vertexIndices[vertexIndex]](0));
//    maxX = max(maxX,_vertices[vertexIndices[vertexIndex]](0));
//    minY = min(minY,_vertices[vertexIndices[vertexIndex]](1));
//    maxY = max(maxY,_vertices[vertexIndices[vertexIndex]](1));
//  }
  
  // and here's an initial effort at a pure Intrepid solution to this problem
  // (nice thing is that it will work with little modification in 3D; bad thing is that
  //  it's inefficient in this context, and also so far it crashes on me...)
//  FieldContainer<double> nodes = physicalCellNodesForCell(elem->cellID());
//  FieldContainer<double> point(1,2);
//  point(0,0) = x; point(0,1) = y;
//  FieldContainer<int> inCell(1);
//  int cellIndex = 0;
//  CellTools<double>::checkPointwiseInclusion(inCell, point, nodes, *(elem->elementType()->cellTopoPtr),cellIndex);
//  return inCell(0) == 1;
  
  // first, check whether x or y is outside the axis-aligned bounding box for the element
  int numVertices = elem->numSides();
  int spaceDim = 2;
  FieldContainer<double> vertices(numVertices,spaceDim);
  verticesForCell(vertices, elem->cellID());
  TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument, "elementContainsPoint only supports 2D.");
  double maxX = vertices(0,0), minX = vertices(0,0);
  double maxY = vertices(0,1), minY = vertices(0,1);
  for (int vertexIndex=1; vertexIndex<numVertices; vertexIndex++) {
    minX = min(minX,vertices(vertexIndex,0));
    maxX = max(maxX,vertices(vertexIndex,0));
    minY = min(minY,vertices(vertexIndex,1));
    maxY = max(maxY,vertices(vertexIndex,1));
  }
  if ( (x < minX) || (x > maxX) ) return false;
  if ( (y < minY) || (y > maxY) ) return false;
  
  // now, use code derived from
  // http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
  // to check whether the point lies within the element.
  // (This can return false for points on the element edges.
  //  It's guaranteed to return true for exactly one of the two elements sharing the edge.
  //  We don't mind which element it returns true for, but when there's no other element sharing
  //  the edge--when we're on the boundary--then we care.  So we handle this case separately, below.)
  int i, j, result = false;
  for (i = 0, j = numVertices-1; i < numVertices; j = i++) {
    if ( ((vertices(i,1)>y) != (vertices(j,1)>y)) &&
        (x < (vertices(j,0)-vertices(i,0)) * (y-vertices(i,1)) / (vertices(j,1)-vertices(i,1)) + vertices(i,0)) )
      result = !result;
  }
  
  if ( !result ) {
    for (int sideIndex=0; sideIndex<elem->numSides(); sideIndex++) {
      vector< pair<int,int> > descendantsForSide = elem->getDescendantsForSide(sideIndex);
      for (vector< pair<int,int> >::iterator descendantIt = descendantsForSide.begin();
           descendantIt != descendantsForSide.end(); descendantIt++) {
        int descCellID = descendantIt->first;
        int descSide = descendantIt->second;
        if ( _boundary.boundaryElement(descCellID,descSide) ) {
          // then check whether the point lies along this side
          double x0 = vertices(sideIndex,0), y0 = vertices(sideIndex,1);
          double x1 = vertices((sideIndex+1)%numVertices,0), y1 = vertices((sideIndex+1)%numVertices,1);
          if (colinear(x0,y0,x1,y1,x,y)) result = true;
          // TODO: if we set result=true here, we could actually modify the element to be the descCellID that matched, if we passed elem by reference...  (It would save a little searching)
        }        
      }
    }
  }
  
  return result;  
}

//void Mesh::enforceOneIrregularity() {
//  enforceOneIrregularity(vector< Teuchos::RCP<Solution> >()); 
//}

void Mesh::enforceOneIrregularity() {
  int rank = 0;
  int numProcs = 1;
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  
  bool meshIsNotRegular = true; // assume it's not regular and check elements
  while (meshIsNotRegular) {
    vector <int> irregularTriangleCells;
    vector <int> irregularQuadCells;
    vector< Teuchos::RCP< Element > > newActiveElements = activeElements();
    vector< Teuchos::RCP< Element > >::iterator newElemIt;
    
    for (newElemIt = newActiveElements.begin(); newElemIt != newActiveElements.end(); newElemIt++) {
      Teuchos::RCP< Element > current_element = *(newElemIt);
      bool isIrregular = false;
      for (int sideIndex=0; sideIndex < current_element->numSides(); sideIndex++) {
        int mySideIndexInNeighbor;
        Element* neighbor; // may be a parent
        current_element->getNeighbor(neighbor, mySideIndexInNeighbor, sideIndex);
        int numNeighborsOnSide = neighbor->getDescendantsForSide(mySideIndexInNeighbor).size();
        if (numNeighborsOnSide > 2) isIrregular=true;
      }
      
      if (isIrregular){
        if ( 3 == current_element->numSides() ) {
          irregularTriangleCells.push_back(current_element->cellID());
        }
        else if (4 == current_element->numSides() ) {
          irregularQuadCells.push_back(current_element->cellID());
        }
	/*
	  if (rank==0){
          cout << "cell " << current_element->cellID() << " refined to maintain regularity" << endl;
        }
	*/
      }
    }
    if ((irregularQuadCells.size()>0) || (irregularTriangleCells.size()>0)) {
      hRefine(irregularTriangleCells,RefinementPattern::regularRefinementPatternTriangle());
      hRefine(irregularQuadCells,RefinementPattern::regularRefinementPatternQuad());
      irregularTriangleCells.clear();
      irregularQuadCells.clear();
    } else {
      meshIsNotRegular=false;
    }
  }
}

// commented this out because it appears to be unused
//Epetra_Map Mesh::getCellIDPartitionMap(int rank, Epetra_Comm* Comm){
//  int indexBase = 0; // 0 for cpp, 1 for fortran
//  int numActiveElements = activeElements().size();
//  
//  // determine cell IDs for this partition
//  vector< ElementPtr > elemsInPartition = elementsInPartition(rank);
//  int numElemsInPartition = elemsInPartition.size();
//  int *partitionLocalElems;
//  if (numElemsInPartition!=0){
//    partitionLocalElems = new int[numElemsInPartition];
//  }else{
//    partitionLocalElems = NULL;
//  }  
//  
//  // set partition-local cell IDs
//  for (int activeCellIndex=0; activeCellIndex<numElemsInPartition; activeCellIndex++) {
//    ElementPtr elemPtr = elemsInPartition[activeCellIndex];
//    int cellIndex = elemPtr->globalCellIndex();
//    partitionLocalElems[activeCellIndex] = cellIndex;
//  }
//  Epetra_Map partMap(numActiveElements, numElemsInPartition, partitionLocalElems, indexBase, *Comm);
//}

FieldContainer<double> & Mesh::cellSideParities( ElementTypePtr elemTypePtr ) {
#ifdef HAVE_MPI
  int partitionNumber     = Teuchos::GlobalMPISession::getRank();
#else
  int partitionNumber     = 0;
#endif
  return _partitionedCellSideParitiesForElementType[ partitionNumber ][ elemTypePtr.get() ];
}

FieldContainer<double> Mesh::cellSideParitiesForCell( int cellID ) {
  vector<int> parities = _cellSideParitiesForCellID[cellID];
  int numSides = parities.size();
  FieldContainer<double> cellSideParities(1,numSides);
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    cellSideParities(0,sideIndex) = parities[sideIndex];
  }
  return cellSideParities;
}

vector<double> Mesh::getCellCentroid(int cellID){
  int numVertices = _elements[cellID]->numSides();
  int spaceDim = 2;
  FieldContainer<double> vertices(numVertices,spaceDim);
  verticesForCell(vertices, cellID);    
  //average vertex positions together to get a centroid (avoids using vertex in case local enumeration overlaps)
  vector<double> coords(spaceDim,0.0);
  for (int k=0;k<spaceDim;k++){
    for (int j=0;j<numVertices;j++){
      coords[k] += vertices(j,k);
    }
    coords[k] = coords[k]/((double)(numVertices));
  }
  return coords;
}

void Mesh::determineActiveElements() {
#ifdef HAVE_MPI
  int partitionNumber     = Teuchos::GlobalMPISession::getRank();
#else
  int partitionNumber     = 0;
#endif
  
  _activeElements.clear();
  vector<ElementPtr>::iterator elemIterator;
  
  for (elemIterator = _elements.begin(); elemIterator != _elements.end(); elemIterator++) {
    ElementPtr elemPtr = *(elemIterator);
    if ( ! elemPtr->isParent() ) {
      _activeElements.push_back(elemPtr);
    }
  }
  _partitions.clear();
  _partitionForCellID.clear();
  FieldContainer<int> partitionedMesh(_numPartitions,_activeElements.size());
  _partitionPolicy->partitionMesh(this,_numPartitions,partitionedMesh);
  _activeCellOffset = 0;
  for (int i=0; i<_numPartitions; i++) {
    vector<ElementPtr> partition;
    for (int j=0; j<_activeElements.size(); j++) {
      if (partitionedMesh(i,j) < 0) break; // no more elements in this partition
      int cellID = partitionedMesh(i,j);
      partition.push_back( _elements[cellID] );
      _partitionForCellID[cellID] = i;
    }
    _partitions.push_back( partition );
    if (partitionNumber > i) {
      _activeCellOffset += partition.size();
    }
  }
}

void Mesh::determinePartitionDofIndices() {
  _partitionedGlobalDofIndices.clear();
  _partitionForGlobalDofIndex.clear();
  _partitionLocalIndexForGlobalDofIndex.clear();
  set<int> dofIndices;
  set<int> previouslyClaimedDofIndices;
  for (int i=0; i<_numPartitions; i++) {
    dofIndices.clear();
    vector< ElementPtr >::iterator elemIterator;
    for (elemIterator =  _partitions[i].begin(); elemIterator != _partitions[i].end(); elemIterator++) {
      ElementPtr elem = *elemIterator;
      ElementTypePtr elemTypePtr = elem->elementType();
      int numLocalDofs = elemTypePtr->trialOrderPtr->totalDofs();
      int cellID = elem->cellID();
      for (int localDofIndex=0; localDofIndex < numLocalDofs; localDofIndex++) {
        pair<int,int> key = make_pair(cellID, localDofIndex);
        map< pair<int,int>, int >::iterator mapEntryIt = _localToGlobalMap.find(key);
        if ( mapEntryIt == _localToGlobalMap.end() ) {
          TEST_FOR_EXCEPTION(true, std::invalid_argument, "entry not found.");
        }
        int dofIndex = (*mapEntryIt).second;
        if ( previouslyClaimedDofIndices.find( dofIndex ) == previouslyClaimedDofIndices.end() ) {
          dofIndices.insert( dofIndex );
          _partitionForGlobalDofIndex[ dofIndex ] = i;
          previouslyClaimedDofIndices.insert(dofIndex);
        }
      }
    }
    _partitionedGlobalDofIndices.push_back( dofIndices );
    int partitionDofIndex = 0;
    for (set<int>::iterator dofIndexIt = dofIndices.begin();
         dofIndexIt != dofIndices.end(); dofIndexIt++) {
      int globalDofIndex = *dofIndexIt;
      _partitionLocalIndexForGlobalDofIndex[globalDofIndex] = partitionDofIndex++;
    }
  }
}

vector< Teuchos::RCP< Element > > & Mesh::elements() { 
  return _elements; 
}

vector< ElementPtr > Mesh::elementsInPartition(int partitionNumber){
  return _partitions[partitionNumber];
}

vector< ElementPtr > Mesh::elementsOfType(int partitionNumber, ElementTypePtr elemTypePtr) {
  // returns the elements for a given partition and element type
  vector< ElementPtr > elementsOfType;
  vector<int> cellIDs = _cellIDsForElementType[partitionNumber][elemTypePtr.get()];
  int numCells = cellIDs.size();
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    elementsOfType.push_back(_elements[cellIDs[cellIndex]]);
  }
  return elementsOfType;
}

vector< ElementPtr > Mesh::elementsOfTypeGlobal(ElementTypePtr elemTypePtr) {
  vector< ElementPtr > elementsOfTypeVector;
  for (int partitionNumber=0; partitionNumber<_numPartitions; partitionNumber++) {
    vector< ElementPtr > elementsOfTypeForPartition = elementsOfType(partitionNumber,elemTypePtr);
    elementsOfTypeVector.insert(elementsOfTypeVector.end(),elementsOfTypeForPartition.begin(),elementsOfTypeForPartition.end());
  }
  return elementsOfTypeVector;
}

vector< Teuchos::RCP< ElementType > > Mesh::elementTypes(int partitionNumber) {
  if ((partitionNumber >= 0) && (partitionNumber < _numPartitions)) {
    return _elementTypesForPartition[partitionNumber];
  } else if (partitionNumber < 0) {
    return _elementTypes;
  } else {
    vector< Teuchos::RCP< ElementType > > noElementTypes;
    return noElementTypes;
  }
}

ElementPtr Mesh::getActiveElement(int index) {
  return _activeElements[index];
}

DofOrderingFactory & Mesh::getDofOrderingFactory() {
  return _dofOrderingFactory;
}

// added by Jesse - accumulates flux/field local dof indices into user-provided maps
void Mesh::getFieldFluxDofInds(map<int,set<int> > &localFluxInds, map<int,set<int> > &localFieldInds){
  
  // determine trialIDs
  vector< int > trialIDs = bilinearForm()->trialIDs();
  vector< int > fieldIDs;
  vector< int > fluxIDs;
  vector< int >::iterator idIt;

  for (idIt = trialIDs.begin();idIt!=trialIDs.end();idIt++){
    int trialID = *(idIt);
    if (!bilinearForm()->isFluxOrTrace(trialID)){ // if field
      fieldIDs.push_back(trialID);
    } else {
      fluxIDs.push_back(trialID);
    }
  } 

  // get all elems in mesh (more than just local info)
  vector< ElementPtr > activeElems = activeElements();
  vector< ElementPtr >::iterator elemIt;

  // gets dof indices
  for (elemIt=activeElems.begin();elemIt!=activeElems.end();elemIt++){
    int cellID = (*elemIt)->cellID();
    int globalCellIndex = (*elemIt)->globalCellIndex();
    int cellIndex = (*elemIt)->cellIndex();
    int numSides = (*elemIt)->numSides();
    ElementTypePtr elemType = (*elemIt)->elementType();
    
    // get local indices (for cell)
    vector<int> inds;
    for (idIt = fieldIDs.begin(); idIt != fieldIDs.end(); idIt++){
      int trialID = (*idIt);
      inds = elemType->trialOrderPtr->getDofIndices(trialID, 0);
      for (int i = 0;i<inds.size();++i){
	localFieldInds[cellID].insert(inds[i]);
      }
    }
    inds.clear();
    for (idIt = fluxIDs.begin(); idIt != fluxIDs.end(); idIt++){
      int trialID = (*idIt);
      for (int sideIndex = 0;sideIndex<numSides;sideIndex++){	
	inds = elemType->trialOrderPtr->getDofIndices(trialID, sideIndex);
	for (int i = 0;i<inds.size();++i){
	  localFluxInds[cellID].insert(inds[i]);
	}	
      }
    }
  }  
}

// Cruft code - remove soon.
void Mesh::getDofIndices(set<int> &allFluxInds, map<int,vector<int> > &globalFluxInds, map<int, vector<int> > &globalFieldInds, map<int,vector<int> > &localFluxInds, map<int,vector<int> > &localFieldInds){
  
 
  // determine trialIDs
  vector< int > trialIDs = bilinearForm()->trialIDs();
  vector< int > fieldIDs;
  vector< int > fluxIDs;
  vector< int >::iterator idIt;

  for (idIt = trialIDs.begin();idIt!=trialIDs.end();idIt++){
    int trialID = *(idIt);
    if (!bilinearForm()->isFluxOrTrace(trialID)){ // if field
      fieldIDs.push_back(trialID);
    } else {
      fluxIDs.push_back(trialID);
    }
  } 

  // get all elems in mesh (more than just local info)
  vector< ElementPtr > activeElems = activeElements();
  vector< ElementPtr >::iterator elemIt;

  // gets dof indices
  for (elemIt=activeElems.begin();elemIt!=activeElems.end();elemIt++){
    int cellID = (*elemIt)->cellID();
    int globalCellIndex = (*elemIt)->globalCellIndex();
    int cellIndex = (*elemIt)->cellIndex();
    int numSides = (*elemIt)->numSides();
    ElementTypePtr elemType = (*elemIt)->elementType();
    
    // get local indices (for cell)
    vector<int> inds;
    for (idIt = fieldIDs.begin(); idIt != fieldIDs.end(); idIt++){
      int trialID = (*idIt);
      inds = elemType->trialOrderPtr->getDofIndices(trialID, 0);
      localFieldInds[cellID].insert(localFieldInds[cellID].end(), inds.begin(), inds.end()); 
    }
    inds.clear();
    for (idIt = fluxIDs.begin(); idIt != fluxIDs.end(); idIt++){
      int trialID = (*idIt);
      for (int sideIndex = 0;sideIndex<numSides;sideIndex++){	
	inds = elemType->trialOrderPtr->getDofIndices(trialID, sideIndex);
	localFluxInds[cellID].insert(localFluxInds[cellID].end(), inds.begin(), inds.end()); 
      }
    }

    // gets global indices (across all cells/all procs)
    for (int i = 0;i<localFieldInds[cellID].size();i++){
      int dofIndex = globalDofIndex(cellID,localFieldInds[cellID][i]);
      globalFieldInds[cellID].push_back(dofIndex);
    }
    for (int i = 0;i<localFluxInds[cellID].size();i++){
      int dofIndex = globalDofIndex(cellID,localFluxInds[cellID][i]);
      globalFluxInds[cellID].push_back(dofIndex);
      allFluxInds.insert(dofIndex); // all flux indices      
    }    
  }  
}

ElementPtr Mesh::getElement(int cellID) {
  return _elements[cellID];
}

ElementTypeFactory & Mesh::getElementTypeFactory() {
  return _elementTypeFactory;
}

void Mesh::getMultiBasisOrdering(DofOrderingPtr &originalNonParentOrdering,
                                 ElementPtr parent, int sideIndex, int parentSideIndexInNeighbor,
                                 ElementPtr nonParent) {
  map< int, BasisPtr > varIDsToUpgrade = multiBasisUpgradeMap(parent,sideIndex);
  originalNonParentOrdering = _dofOrderingFactory.upgradeSide(originalNonParentOrdering,
                                                              *(nonParent->elementType()->cellTopoPtr),
                                                              varIDsToUpgrade,parentSideIndexInNeighbor);
}

Epetra_Map Mesh::getPartitionMap() {
  int rank = 0;
  int numProcs = 1;
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  
  // returns map for current processor's local-to-global dof indices
  // determine the local dofs we have, and what their global indices are:
  int localDofsSize;

  set<int> myGlobalIndicesSet = globalDofIndicesForPartition(rank);
  
  localDofsSize = myGlobalIndicesSet.size();
  
  int *myGlobalIndices;
  if (localDofsSize!=0){
    myGlobalIndices = new int[ localDofsSize ];      
  }else{
    myGlobalIndices = NULL;
  }
  
  // copy from set object into the allocated array
  int offset = 0;
  for ( set<int>::iterator indexIt = myGlobalIndicesSet.begin();
       indexIt != myGlobalIndicesSet.end();
       indexIt++ ){
    myGlobalIndices[offset++] = *indexIt;
  }
  
  int numGlobalDofs = this->numGlobalDofs();
  int indexBase = 0;
  //cout << "process " << rank << " about to construct partMap.\n";
  //Epetra_Map partMap(-1, localDofsSize, myGlobalIndices, indexBase, Comm);
  Epetra_Map partMap(numGlobalDofs, localDofsSize, myGlobalIndices, indexBase, Comm);
  
  if (localDofsSize!=0){
    delete myGlobalIndices;
  }
  return partMap;
}

void Mesh::getPatchBasisOrdering(DofOrderingPtr &originalChildOrdering, ElementPtr child, int sideIndex) {
  DofOrderingPtr parentTrialOrdering = child->getParent()->elementType()->trialOrderPtr;
//  cout << "Adding PatchBasis for element " << child->cellID() << " along side " << sideIndex << "\n";
//  
//  cout << "parent is cellID " << child->getParent()->cellID() << "; parent trialOrdering:\n";
//  cout << *parentTrialOrdering;
//  
//  cout << "original childTrialOrdering:\n" << *originalChildOrdering;
  
  //cout << "Adding PatchBasis for element " << child->cellID() << ":\n" << physicalCellNodesForCell(child->cellID());
  int parentSideIndex = child->parentSideForSideIndex(sideIndex);
  int childIndexInParentSide = child->indexInParentSide(parentSideIndex);
  map< int, BasisPtr > varIDsToUpgrade = _dofOrderingFactory.getPatchBasisUpgradeMap(originalChildOrdering, sideIndex,
                                                                                     parentTrialOrdering, parentSideIndex,
                                                                                     childIndexInParentSide);
  originalChildOrdering = _dofOrderingFactory.upgradeSide(originalChildOrdering,
                                                          *(child->elementType()->cellTopoPtr),
                                                          varIDsToUpgrade,parentSideIndex);
  
//  cout << "childTrialOrdering after upgrading side:\n" << *originalChildOrdering;
}

int Mesh::globalDofIndex(int cellID, int localDofIndex) {
  pair<int,int> key = make_pair(cellID, localDofIndex);
  map< pair<int,int>, int >::iterator mapEntryIt = _localToGlobalMap.find(key);
  if ( mapEntryIt == _localToGlobalMap.end() ) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "entry not found.");
  }
  return (*mapEntryIt).second;
}

set<int> Mesh::globalDofIndicesForPartition(int partitionNumber) {
  return _partitionedGlobalDofIndices[partitionNumber];
}

//void Mesh::hRefine(vector<int> cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
//  hRefine(cellIDs,refPattern,vector< Teuchos::RCP<Solution> >()); 
//}

void Mesh::hRefine(vector<int> cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
  // refine any registered meshes
  for (vector< Teuchos::RCP<Mesh> >::iterator meshIt = _registeredMeshes.begin();
       meshIt != _registeredMeshes.end(); meshIt++) {
    (*meshIt)->hRefine(cellIDs,refPattern);
  }
  
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
    for (vector< Teuchos::RCP<Solution> >::iterator solutionIt = _registeredSolutions.begin();
         solutionIt != _registeredSolutions.end(); solutionIt++) {
       // do projection
      int numChildren = _elements[cellID]->numChildren();
      vector<int> childIDs;
      for (int i=0; i<numChildren; i++) {
        childIDs.push_back(_elements[cellID]->getChild(i)->cellID());
      }
      (*solutionIt)->processSideUpgrades(_cellSideUpgrades);
      (*solutionIt)->projectOldCellOntoNewCells(cellID,elemType,childIDs);
    }
    _cellSideUpgrades.clear(); // these have been processed by all solutions that will ever have a chance to process them.
  }
  rebuildLookups();
  // now discard any old coefficients
  for (vector< Teuchos::RCP<Solution> >::iterator solutionIt = _registeredSolutions.begin();
       solutionIt != _registeredSolutions.end(); solutionIt++) {
    (*solutionIt)->discardInactiveCellCoefficients();
  }
}

int Mesh::neighborChildPermutation(int childIndex, int numChildrenInSide) {
  // we'll need to be more sophisticated in 3D, but for now we just reverse the order
  return numChildrenInSide - childIndex - 1;
}

int Mesh::neighborDofPermutation(int dofIndex, int numDofsForSide) {
  // we'll need to be more sophisticated in 3D, but for now we just reverse the order
  return numDofsForSide - dofIndex - 1;
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
    } else { // one broken
      vector< pair< int, int> > childrenForSide = parent->childIndicesForSide(neighborSideIndexInParent);
      
      if ( childrenForSide.size() > 1 ) { // then parent is broken along side, and neighbor isn't...
        vector< pair< int, int> > descendantsForSide;
        if ( !_usePatchBasis ) { // MultiBasis
          descendantsForSide = parent->getDescendantsForSide(neighborSideIndexInParent);
          
          Teuchos::RCP<DofOrdering> nonParentTrialOrdering = nonParent->elementType()->trialOrderPtr;

          getMultiBasisOrdering( nonParentTrialOrdering, parent, neighborSideIndexInParent,
                                parentSideIndexInNeighbor, nonParent );
          ElementTypePtr nonParentType = _elementTypeFactory.getElementType(nonParentTrialOrdering, 
                                                                            nonParent->elementType()->testOrderPtr, 
                                                                            nonParent->elementType()->cellTopoPtr );
          setElementType(nonParent->cellID(), nonParentType, true); // true: only a side upgrade
          //nonParent->setElementType(nonParentType);
          // debug code:
          if ( ! _dofOrderingFactory.sideHasMultiBasis(nonParentTrialOrdering, parentSideIndexInNeighbor) ) {
            TEST_FOR_EXCEPTION(true, std::invalid_argument, "failed to add multi-basis to neighbor");
          }
        } else { // PatchBasis
          // check to see if non-parent needs a p-upgrade
          int maxPolyOrder, minPolyOrder; 
          this->maxMinPolyOrder(maxPolyOrder, minPolyOrder, nonParent,parentSideIndexInNeighbor);
          DofOrderingPtr nonParentTrialOrdering = nonParent->elementType()->trialOrderPtr;
          DofOrderingPtr    parentTrialOrdering =    parent->elementType()->trialOrderPtr;
          
          int    parentPolyOrder = _dofOrderingFactory.polyOrder(   parentTrialOrdering);
          int nonParentPolyOrder = _dofOrderingFactory.polyOrder(nonParentTrialOrdering);
          
          if (maxPolyOrder > nonParentPolyOrder) {
            // upgrade p along the side in non-parent
            nonParentTrialOrdering = _dofOrderingFactory.setSidePolyOrder(nonParentTrialOrdering, parentSideIndexInNeighbor, maxPolyOrder, false);
            ElementTypePtr nonParentType = _elementTypeFactory.getElementType(nonParentTrialOrdering, 
                                                                              nonParent->elementType()->testOrderPtr, 
                                                                              nonParent->elementType()->cellTopoPtr );
            setElementType(nonParent->cellID(), nonParentType, true); // true: only a side upgrade

          }
          // now, importantly, do the same thing in the parent:
          if (maxPolyOrder > parentPolyOrder) {
            parentTrialOrdering = _dofOrderingFactory.setSidePolyOrder(parentTrialOrdering, neighborSideIndexInParent, maxPolyOrder, false);
            ElementTypePtr parentType = _elementTypeFactory.getElementType(parentTrialOrdering,
                                                                           parent->elementType()->testOrderPtr,
                                                                           parent->elementType()->cellTopoPtr);
            setElementType(parent->cellID(), parentType, true); // true: only a side upgrade            
          }

          // get all descendants, not just leaf nodes, for the PatchBasis upgrade
          // could make more efficient by checking whether we really need to upgrade the whole hierarchy, but 
          // this is probably more trouble than it's worth, unless we end up with some *highly* irregular meshes.
          descendantsForSide = parent->getDescendantsForSide(neighborSideIndexInParent, false);
          vector< pair< int, int> >::iterator entryIt;
          for ( entryIt=descendantsForSide.begin(); entryIt != descendantsForSide.end(); entryIt++) {
            int childCellID = (*entryIt).first;
            ElementPtr child = _elements[childCellID];
            int childSideIndex = (*entryIt).second;
            DofOrderingPtr childTrialOrdering = child->elementType()->trialOrderPtr;
            getPatchBasisOrdering(childTrialOrdering,child,childSideIndex);
            ElementTypePtr childType = _elementTypeFactory.getElementType(childTrialOrdering, 
                                                                          child->elementType()->testOrderPtr, 
                                                                          child->elementType()->cellTopoPtr );
            setElementType(childCellID,childType,true); //true: sideUpgradeOnly
            //child->setElementType(childType);
          }
        }
        
        vector< pair< int, int> >::iterator entryIt;
        for ( entryIt=descendantsForSide.begin(); entryIt != descendantsForSide.end(); entryIt++) {
          int childCellID = (*entryIt).first;
          int childSideIndex = (*entryIt).second;
          _boundary.deleteElement(childCellID, childSideIndex);
        }
        // by virtue of having assigned the patch- or multi-basis, we've already matched p-order ==> we're done
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
  // changed == 1 for me, 2 for neighbor, 0 for neither, -1 for PatchBasis
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
    ElementTypePtr newType = _elementTypeFactory.getElementType(elemTrialOrdering, elemTestOrdering, 
                                                                elem->elementType()->cellTopoPtr );
    setElementType(elem->cellID(), newType, true); // true: 
//    elem->setElementType( _elementTypeFactory.getElementType(elemTrialOrdering, elemTestOrdering, 
//                                                             elem->elementType()->cellTopoPtr ) );
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
    ElementTypePtr newType = _elementTypeFactory.getElementType(neighborTrialOrdering, neighborTestOrdering, 
                                                                neighbor->elementType()->cellTopoPtr );
    setElementType( neighbor->cellID(), newType, true); // true: sideUpgradeOnly
    //return NEIGHBOR_NEEDED_NEW;
  } else if (changed == -1) { // PatchBasis
    // if we get here, these are the facts:
    // 1. both element and neighbor are unbroken--leaf nodes.
    // 2. one of element or neighbor has a PatchBasis
    
    TEST_FOR_EXCEPTION(_bilinearForm->trialBoundaryIDs().size() == 0,
                       std::invalid_argument,
                       "BilinearForm has no traces or fluxes, but somehow neighbor was upgraded...");
    int boundaryVarID = _bilinearForm->trialBoundaryIDs()[0];
    
    // So what we need to do is figure out the right p-order for the side and set both bases accordingly.
    // determine polyOrder for each side--take the maximum
    int neighborPolyOrder = _dofOrderingFactory.polyOrder(neighborTrialOrdering);
    int myPolyOrder = _dofOrderingFactory.polyOrder(elemTrialOrdering);
    
    int polyOrder = max(neighborPolyOrder,myPolyOrder); // "maximum" rule
    
    // upgrade element
    elemTrialOrdering = _dofOrderingFactory.setSidePolyOrder(elemTrialOrdering,sideIndex,polyOrder,true);
    int sidePolyOrder = BasisFactory::basisPolyOrder(elemTrialOrdering->getBasis(boundaryVarID,mySideIndexInNeighbor));
    int testPolyOrder = _dofOrderingFactory.polyOrder(elemTestOrdering);
    if (testPolyOrder < sidePolyOrder + _pToAddToTest) {
      elemTestOrdering = _dofOrderingFactory.testOrdering( sidePolyOrder + _pToAddToTest, cellTopo );
    }
    ElementTypePtr newElemType = _elementTypeFactory.getElementType(elemTrialOrdering, elemTestOrdering, 
                                                                    elem->elementType()->cellTopoPtr );
    setElementType( elem->cellID(), newElemType, true); // true: sideUpgradeOnly
    
    // upgrade neighbor
    neighborTrialOrdering = _dofOrderingFactory.setSidePolyOrder(neighborTrialOrdering,mySideIndexInNeighbor,polyOrder,true);
    testPolyOrder = _dofOrderingFactory.polyOrder(neighborTestOrdering);
    if (testPolyOrder < sidePolyOrder + _pToAddToTest) {
      neighborTestOrdering = _dofOrderingFactory.testOrdering( sidePolyOrder + _pToAddToTest, neighborTopo);
    }
    ElementTypePtr newNeighborType = _elementTypeFactory.getElementType(neighborTrialOrdering, neighborTestOrdering, 
                                                                        neighbor->elementType()->cellTopoPtr );
    setElementType( neighbor->cellID(), newNeighborType, true); // true: sideUpgradeOnly
    
    // TEST_FOR_EXCEPTION(true, std::invalid_argument, "PatchBasis support still a work in progress!");
  } else {
    //return NEITHER_NEEDED_NEW;
  }
}

void Mesh::maxMinPolyOrder(int &maxPolyOrder, int &minPolyOrder, ElementPtr elem, int sideIndex) {
  // returns maximum polyOrder (on interior/field variables) of this element and any that border it on the given side
  int mySideIndexInNeighbor;
  ElementPtr neighbor = ancestralNeighborForSide(elem, sideIndex, mySideIndexInNeighbor);
  if ((neighbor.get() == NULL) || (neighbor->cellID() < 0)) { // as presently implemented, neighbor won't be NULL, but an "empty" elementPtr, _nullPtr, with cellID -1.  I'm a bit inclined to think a NULL would be better.  The _nullPtr conceit comes from the early days of the Mesh class, and seems a bit weird now.
    minPolyOrder = _dofOrderingFactory.polyOrder(elem->elementType()->trialOrderPtr);
    maxPolyOrder = minPolyOrder;
    return;
  }
  maxPolyOrder = max(_dofOrderingFactory.polyOrder(neighbor->elementType()->trialOrderPtr),
                     _dofOrderingFactory.polyOrder(elem->elementType()->trialOrderPtr));
  minPolyOrder = min(_dofOrderingFactory.polyOrder(neighbor->elementType()->trialOrderPtr),
                     _dofOrderingFactory.polyOrder(elem->elementType()->trialOrderPtr));
  int ancestorSideIndex;
  Element* ancestor;
  neighbor->getNeighbor(ancestor,ancestorSideIndex,mySideIndexInNeighbor);
  Element* parent = NULL;
  int parentSideIndex;
  if (ancestor->isParent()) {
    parent = ancestor;
    parentSideIndex = ancestorSideIndex;
  } else if (neighbor->isParent()) {
    parent = neighbor.get();
    parentSideIndex = mySideIndexInNeighbor;
  }
  if (parent != NULL) {
    vector< pair< int, int> > descendantSides = parent->getDescendantsForSide(parentSideIndex);
    vector< pair< int, int> >::iterator sideIt;
    for (sideIt = descendantSides.begin(); sideIt != descendantSides.end(); sideIt++) {
      int descendantID = sideIt->first;
      int descOrder = _dofOrderingFactory.polyOrder(_elements[descendantID]->elementType()->trialOrderPtr);
      maxPolyOrder = max(maxPolyOrder,descOrder);
      minPolyOrder = min(minPolyOrder,descOrder);
    }
  }
}

map< int, BasisPtr > Mesh::multiBasisUpgradeMap(ElementPtr parent, int sideIndex, int bigNeighborPolyOrder) {
  if (bigNeighborPolyOrder==-1) {
    // assumption is that we're at the top level: so parent's neighbor on sideIndex exists/is a peer
    int bigNeighborCellID = parent->getNeighborCellID(sideIndex);
    ElementPtr bigNeighbor = getElement(bigNeighborCellID);
    bigNeighborPolyOrder = _dofOrderingFactory.polyOrder( bigNeighbor->elementType()->trialOrderPtr );
  }
  vector< pair< int, int> > childrenForSide = parent->childIndicesForSide(sideIndex);
  map< int, BasisPtr > varIDsToUpgrade;
  vector< map< int, BasisPtr > > childVarIDsToUpgrade;
  vector< pair< int, int> >::iterator entryIt;
  for ( entryIt=childrenForSide.begin(); entryIt != childrenForSide.end(); entryIt++) {
    int childCellIndex = (*entryIt).first;
    int childSideIndex = (*entryIt).second;
    ElementPtr childCell = parent->getChild(childCellIndex);
    
    if ( childCell->isParent() && (childCell->childIndicesForSide(childSideIndex).size() > 1)) {
      childVarIDsToUpgrade.push_back( multiBasisUpgradeMap(childCell,childSideIndex,bigNeighborPolyOrder) );
    } else {
      DofOrderingPtr childTrialOrder = childCell->elementType()->trialOrderPtr;
      
      int childPolyOrder = _dofOrderingFactory.polyOrder( childTrialOrder );
      
      if (bigNeighborPolyOrder > childPolyOrder) {
        // upgrade child p along side
        childTrialOrder = _dofOrderingFactory.setSidePolyOrder(childTrialOrder, childSideIndex, bigNeighborPolyOrder, false);
        ElementTypePtr newChildType = _elementTypeFactory.getElementType(childTrialOrder, 
                                                                         childCell->elementType()->testOrderPtr, 
                                                                         childCell->elementType()->cellTopoPtr );
        setElementType(childCell->cellID(), newChildType, true); // true: only a side upgrade
      }
      
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
    // debugging:
//    ((MultiBasis*)multiBasis.get())->printInfo();
    varIDsToUpgrade[varID] = multiBasis;
  }
  return varIDsToUpgrade;
}

int Mesh::numActiveElements() {
  return _activeElements.size();
}

int Mesh::numElements() {
  return _elements.size();
}

int Mesh::numElementsOfType( Teuchos::RCP< ElementType > elemTypePtr ) {
  // returns the global total (across all MPI nodes)
  int numElements = 0;
  for (int partitionNumber=0; partitionNumber<_numPartitions; partitionNumber++) {
    if (   _partitionedPhysicalCellNodesForElementType[partitionNumber].find( elemTypePtr.get() )
        != _partitionedPhysicalCellNodesForElementType[partitionNumber].end() ) {
      numElements += _partitionedPhysicalCellNodesForElementType[partitionNumber][ elemTypePtr.get() ].dimension(0);
    }
  }
  return numElements;
}

int Mesh::numFluxDofs(){
  return numGlobalDofs()-numFieldDofs();
}

int Mesh::numFieldDofs(){
  int numFieldDofs = 0;  
  int numActiveElems = numActiveElements();
  for (int cellIndex = 0; cellIndex < numActiveElems; cellIndex++){
    ElementPtr elemPtr = getActiveElement(cellIndex);
    int cellID = elemPtr->cellID();
    int numSides = elemPtr->numSides();
    ElementTypePtr elemTypePtr = elemPtr->elementType();
    vector< int > fieldIDs = _bilinearForm->trialVolumeIDs();
    vector< int >::iterator fieldIDit;
    for (fieldIDit = fieldIDs.begin(); fieldIDit != fieldIDs.end() ; fieldIDit++){    
      int numDofs = elemTypePtr->trialOrderPtr->getBasisCardinality(*fieldIDit,0);
      numFieldDofs += numDofs;
    }
  }
  return numFieldDofs; // don't count double - each side is shared by two elems, we count it twice
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
  int partitionNumber = partitionForCellID(cellID);
  int parity = _partitionedCellSideParitiesForElementType[partitionNumber][elemType.get()](cellIndex,sideIndex);
  
  if (_cellSideParitiesForCellID[cellID][sideIndex] != parity ) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "parity lookups don't match");
  }
  return parity;
}

int Mesh::partitionForCellID( int cellID ) {
  return _partitionForCellID[ cellID ];
}

int Mesh::partitionForGlobalDofIndex( int globalDofIndex ) {
  if ( _partitionForGlobalDofIndex.find( globalDofIndex ) == _partitionForGlobalDofIndex.end() ) {
    return -1;
  }
  return _partitionForGlobalDofIndex[ globalDofIndex ];
}

int Mesh::partitionLocalIndexForGlobalDofIndex( int globalDofIndex ) {
  return _partitionLocalIndexForGlobalDofIndex[ globalDofIndex ];
}

FieldContainer<double> & Mesh::physicalCellNodes( Teuchos::RCP< ElementType > elemTypePtr) {
#ifdef HAVE_MPI
  int partitionNumber     = Teuchos::GlobalMPISession::getRank();
#else
  int partitionNumber     = 0;
#endif
  return _partitionedPhysicalCellNodesForElementType[ partitionNumber ][ elemTypePtr.get() ];
}

FieldContainer<double> Mesh::physicalCellNodesForCell( int cellID ) {
  ElementPtr elem = _elements[cellID];
  int numSides = elem->numSides();
  int spaceDim = elem->elementType()->cellTopoPtr->getDimension();
  int numCells = 1;
  FieldContainer<double> physicalCellNodes(numCells,numSides,spaceDim);
  
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    for (int i=0; i<spaceDim; i++) {
      physicalCellNodes(0,sideIndex,i) = _vertices[_verticesForCellID[cellID][sideIndex]](i);
    }
  }
  return physicalCellNodes;
}

FieldContainer<double> & Mesh::physicalCellNodesGlobal( Teuchos::RCP< ElementType > elemTypePtr ) {
  return _physicalCellNodesForElementType[ elemTypePtr.get() ];
}

void Mesh::printLocalToGlobalMap() {
  for (map< pair<int,int>, int>::iterator entryIt = _localToGlobalMap.begin();
       entryIt != _localToGlobalMap.end(); entryIt++) {
    int cellID = entryIt->first.first;
    int localDofIndex = entryIt->first.second;
    int globalDofIndex = entryIt->second;
    cout << "(" << cellID << "," << localDofIndex << ") --> " << globalDofIndex << endl;
  }
}

void Mesh::rebuildLookups() {
  _cellSideUpgrades.clear();
  determineActiveElements();
  buildTypeLookups(); // build data structures for efficient lookup by element type
  buildLocalToGlobalMap();
  determinePartitionDofIndices();
  _boundary.buildLookupTables();
  //cout << "Mesh.numGlobalDofs: " << numGlobalDofs() << endl;
}

void Mesh::registerMesh(Teuchos::RCP<Mesh> mesh) {
  _registeredMeshes.push_back(mesh);
}

void Mesh::registerSolution(Teuchos::RCP<Solution> solution) {
  _registeredSolutions.push_back( solution );
}

void Mesh::unregisterMesh(Teuchos::RCP<Mesh> mesh) {
  for (vector< Teuchos::RCP<Mesh> >::iterator meshIt = _registeredMeshes.begin();
       meshIt != _registeredMeshes.end(); meshIt++) {
    if ( (*meshIt).get() == mesh.get() ) {
      _registeredMeshes.erase(meshIt);
      return;
    }
  }
  cout << "Mesh::unregisterMesh: Mesh not found.\n";
}

void Mesh::unregisterSolution(Teuchos::RCP<Solution> solution) {
  for (vector< Teuchos::RCP<Solution> >::iterator solnIt = _registeredSolutions.begin();
       solnIt != _registeredSolutions.end(); solnIt++) {
    if ( (*solnIt).get() == solution.get() ) {
      _registeredSolutions.erase(solnIt);
      return;
    }
  }
  cout << "Mesh::unregisterSolution: Solution not found.\n";
}

//void Mesh::pRefine(vector<int> cellIDsForPRefinements) {
//  pRefine(cellIDsForPRefinements, vector< Teuchos::RCP<Solution> >());
//}

void Mesh::pRefine(vector<int> cellIDsForPRefinements) {
  // refine any registered meshes
  for (vector< Teuchos::RCP<Mesh> >::iterator meshIt = _registeredMeshes.begin();
       meshIt != _registeredMeshes.end(); meshIt++) {
    (*meshIt)->pRefine(cellIDsForPRefinements);
  }
  
  // p-refinements:
  // 1. Loop through cellIDsForPRefinements:
  //   a. create new DofOrderings for trial and test
  //   b. create and set new element type
  //   c. Loop through sides, calling matchNeighbor for each
  
  // 1. Loop through cellIDsForPRefinements:
  vector<int>::iterator cellIt;
  for (cellIt=cellIDsForPRefinements.begin(); cellIt != cellIDsForPRefinements.end(); cellIt++) {
    int cellID = *cellIt;
    ElementPtr elem = _elements[cellID];
    ElementTypePtr oldElemType = elem->elementType();
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

    ElementTypePtr newType = _elementTypeFactory.getElementType(newTrialOrdering, newTestOrdering, 
                                                                elem->elementType()->cellTopoPtr );
    setElementType(cellID,newType,false); // false: *not* sideUpgradeOnly
    
//    elem->setElementType( _elementTypeFactory.getElementType(newTrialOrdering, newTestOrdering, 
//                                                             elem->elementType()->cellTopoPtr ) );
    
    int numSides = elem->numSides();
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      // get and match the big neighbor along the side, if we're a small element…
      int neighborSideIndex;
      ElementPtr neighborToMatch = ancestralNeighborForSide(elem,sideIndex,neighborSideIndex);
      
      if (neighborToMatch->cellID() != -1) { // then we have a neighbor to match along that side...
        matchNeighbor(neighborToMatch,neighborSideIndex);
      }
    }
    
    for (vector< Teuchos::RCP<Solution> >::iterator solutionIt = _registeredSolutions.begin();
         solutionIt != _registeredSolutions.end(); solutionIt++) {
      // do projection
      vector<int> childIDs;
      childIDs.push_back(cellID);
      (*solutionIt)->processSideUpgrades(_cellSideUpgrades);
      (*solutionIt)->projectOldCellOntoNewCells(cellID,oldElemType,childIDs);
    }
    _cellSideUpgrades.clear(); // these have been processed by all solutions that will ever have a chance to process them.
  }
  rebuildLookups();
  
  // now discard any old coefficients
  for (vector< Teuchos::RCP<Solution> >::iterator solutionIt = _registeredSolutions.begin();
       solutionIt != _registeredSolutions.end(); solutionIt++) {
    (*solutionIt)->discardInactiveCellCoefficients();
  }
}

int Mesh::rowSizeUpperBound() {
  // includes multiplicity
  vector< Teuchos::RCP< ElementType > >::iterator elemTypeIt;
  int maxRowSize = 0;
  for (int partitionNumber=0; partitionNumber < _numPartitions; partitionNumber++) {
    for (elemTypeIt = _elementTypesForPartition[partitionNumber].begin(); 
         elemTypeIt != _elementTypesForPartition[partitionNumber].end();
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
  }
  return maxRowSize;
}

void Mesh::setElementType(int cellID, ElementTypePtr newType, bool sideUpgradeOnly) {
  ElementPtr elem = _elements[cellID];
  if (sideUpgradeOnly) { // need to track in _cellSideUpgrades
    ElementTypePtr oldType;
    map<int, pair<ElementTypePtr, ElementTypePtr> >::iterator existingEntryIt = _cellSideUpgrades.find(cellID);
    if (existingEntryIt != _cellSideUpgrades.end() ) {
      oldType = (existingEntryIt->second).first;
    } else {
      oldType = elem->elementType();
      if (oldType.get() == newType.get()) {
        // no change is actually happening
        return;
      }
    }
    _cellSideUpgrades[cellID] = make_pair(oldType,newType);
  }
  elem->setElementType(newType);
}

void Mesh::setEnforceMultiBasisFluxContinuity( bool value ) {
  _enforceMBFluxContinuity = value;
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
    if (neighborPtr->isParent()) { // then we need to set its children accordingly, too
      vector< pair< int, int> > descendantSides = neighborPtr->getDescendantsForSide(neighborSide);
      vector< pair< int, int> >::iterator sideIt;
      for (sideIt = descendantSides.begin(); sideIt != descendantSides.end(); sideIt++) {
        int descendantID = sideIt->first;
        int descendantSide = sideIt->second;
        _cellSideParitiesForCellID[descendantID][descendantSide] = -parity;
      }
    }
//    cout << "setNeighbor: set cellSideParity for cell " << neighborPtr->cellID() << ", sideIndex " << neighborSide << ": ";
//    cout << _cellSideParitiesForCellID[neighborPtr->cellID()][neighborSide] << endl;
  }
//  cout << "set cellID " << elemPtr->cellID() << "'s neighbor for side ";
//  cout << elemSide << " to cellID " << neighborPtr->cellID();
//  cout << " (neighbor's sideIndex: " << neighborSide << ")" << endl;
}

void Mesh::setPartitionPolicy(  Teuchos::RCP< MeshPartitionPolicy > partitionPolicy ) {
  _partitionPolicy = partitionPolicy;
  rebuildLookups();
}

void Mesh::setUsePatchBasis( bool value ) {
  // TODO: throw an exception if we've already been refined??
  _usePatchBasis = value;
}

bool Mesh::usePatchBasis() {
  return _usePatchBasis;
}

vector<int> Mesh::vertexIndicesForCell(int cellID) {
  return _verticesForCellID[cellID];
}

FieldContainer<double> Mesh::vertexCoordinates(int vertexIndex) {
  return _vertices[vertexIndex];
}

void Mesh::verticesForCell(FieldContainer<double>& vertices, int cellID) {
  vector<int> vertexIndices = _verticesForCellID[cellID];
  ElementTypePtr elemType = _elements[cellID]->elementType();
  int dimension = elemType->cellTopoPtr->getDimension();
  int numVertices = elemType->cellTopoPtr->getVertexCount(dimension,0);
  //vertices.resize(numVertices,dimension);
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

void Mesh::writeMeshPartitionsToFile(const string & fileName){
  ofstream myFile;
  myFile.open(fileName.c_str());
  myFile << "numPartitions="<<_numPartitions<<";"<<endl;

  int maxNumVertices=0;
  int maxNumElems=0;
  int spaceDim = 3;

  //initialize verts
  for (int i=0;i<_numPartitions;i++){
    vector< ElementPtr > elemsInPartition = _partitions[i];
    for (int l=0;l<spaceDim;l++){
      myFile << "verts{"<< i+1 <<","<< l+1 << "} = zeros(" << maxNumVertices << ","<< maxNumElems << ");"<< endl;
      for (int j=0;j<elemsInPartition.size();j++){
	FieldContainer<double> verts; // gets resized inside verticesForCell
	verticesForCell(verts, elemsInPartition[j]->cellID());  //verts(numVertsForCell,dim)
	maxNumVertices = max(maxNumVertices,verts.dimension(0));
	maxNumElems = max(maxNumElems,(int)elemsInPartition.size());
	spaceDim = verts.dimension(1);
      }
    }
  }  
  cout << "max number of elems = " << maxNumElems << endl;

  for (int i=0;i<_numPartitions;i++){
    vector< ElementPtr > elemsInPartition = _partitions[i];
    for (int l=0;l<spaceDim;l++){

      for (int j=0;j<elemsInPartition.size();j++){
	FieldContainer<double> vertices; // gets resized inside verticesForCell
	verticesForCell(vertices, elemsInPartition[j]->cellID());  //vertices(numVertsForCell,dim)
	int numVertices = vertices.dimension(0);
	
	// write vertex coordinates to file	
	for (int k=0;k<numVertices;k++){	  
	  myFile << "verts{"<< i+1 <<","<< l+1 <<"}("<< k+1 <<","<< j+1 <<") = "<< vertices(k,l) << ";"<<endl; // verts{numPartitions,spaceDim}
	}
      }
      
    }
  }
  myFile.close();
}
