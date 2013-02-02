// @HEADER
//
// Copyright © 2011 Nathan V. Roberts. All Rights Reserved.
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
// THIS SOFTWARE IS PROVIDED BY NATHAN V. ROBERTS "AS IS" AND ANY 
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

#include "RefinementPattern.h"

#include "Intrepid_CellTools.hpp"

using namespace Intrepid;

RefinementPattern::RefinementPattern(Teuchos::RCP< shards::CellTopology > cellTopoPtr, FieldContainer<double> refinedNodes) {
  _cellTopoPtr = cellTopoPtr;
  _nodes = refinedNodes;
  int numSubCells = refinedNodes.dimension(0);
  int numNodesPerCell = refinedNodes.dimension(1);
  int spaceDim = refinedNodes.dimension(2);
  vector<double> vertex(spaceDim);
  
  unsigned cellKey = cellTopoPtr->getKey();
  
  map< vector<double>, int> vertexLookup;
  vector< vector<double> > vertices;
  for (int cellIndex=0; cellIndex<numSubCells; cellIndex++) {
    vector<int> subCellNodes;
    for (int nodeIndex=0; nodeIndex<numNodesPerCell; nodeIndex++) {
      for (int dim=0; dim<spaceDim; dim++) {
        vertex[dim] = refinedNodes(cellIndex,nodeIndex,dim);
      }
      int vertexIndex;
      if ( vertexLookup.find(vertex) == vertexLookup.end() ) {
        vertexIndex = vertices.size();
        vertices.push_back(vertex);
        vertexLookup[vertex] = vertexIndex;
      } else {
        vertexIndex = vertexLookup[vertex];
      }
      subCellNodes.push_back(vertexIndex);
    }
    _subCells.push_back(subCellNodes);
  }
  // copy these to FieldContainer
  int numVertices = vertices.size();
  _vertices.resize(numVertices, spaceDim);
  for (int vertexIndex=0; vertexIndex<numVertices; vertexIndex++) {
    for (int dim=0; dim<spaceDim; dim++) {
      _vertices(vertexIndex,dim) = vertices[vertexIndex][dim];
    }
  }
  
  // determine which subcells are along each of parent's sides...
  int numSides;
  vector< vector< int > > refSides;
  if (cellKey == shards::Quadrilateral<4>::key ) {
    numSides = 4;
    vector<double> v0, v1, v2, v3;
    v0.push_back(-1.0);
    v0.push_back(-1.0);
    v1.push_back(1.0);
    v1.push_back(-1.0);
    v2.push_back(1.0);
    v2.push_back(1.0);
    v3.push_back(-1.0);
    v3.push_back(1.0);
    
    if ( vertexLookup.find(v0) == vertexLookup.end() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"v0 not found!");
    }
    if ( vertexLookup.find(v1) == vertexLookup.end() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"v1 not found!");
    }
    if ( vertexLookup.find(v2) == vertexLookup.end() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"v2 not found!");
    }
    if ( vertexLookup.find(v3) == vertexLookup.end() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"v3 not found!");
    }
    
    int v0_index = vertexLookup[v0];
    int v1_index = vertexLookup[v1];
    int v2_index = vertexLookup[v2];
    int v3_index = vertexLookup[v3];
    
    // first side: v0 to v1
    vector<int> side0, side1, side2, side3;
    side0.push_back(v0_index);
    side0.push_back(v1_index);
    side1.push_back(v1_index);
    side1.push_back(v2_index);
    side2.push_back(v2_index);
    side2.push_back(v3_index);
    side3.push_back(v3_index);
    side3.push_back(v0_index);
    refSides.push_back(side0);
    refSides.push_back(side1);
    refSides.push_back(side2);
    refSides.push_back(side3);
  } else if (cellKey == shards::Triangle<3>::key) {
    numSides = 3;
    vector<double> v0, v1, v2, v3;
    v0.push_back(0.0);
    v0.push_back(0.0);
    v1.push_back(1.0);
    v1.push_back(0.0);
    v2.push_back(0.0);
    v2.push_back(1.0);
    
    if ( vertexLookup.find(v0) == vertexLookup.end() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"v0 not found!");
    }
    if ( vertexLookup.find(v1) == vertexLookup.end() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"v1 not found!");
    }
    if ( vertexLookup.find(v2) == vertexLookup.end() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"v2 not found!");
    }
    
    int v0_index = vertexLookup[v0];
    int v1_index = vertexLookup[v1];
    int v2_index = vertexLookup[v2];
    
    // first side: v0 to v1
    vector<int> side0, side1, side2, side3;
    side0.push_back(v0_index);
    side0.push_back(v1_index);
    side1.push_back(v1_index);
    side1.push_back(v2_index);
    side2.push_back(v2_index);
    side2.push_back(v0_index);
    refSides.push_back(side0);
    refSides.push_back(side1);
    refSides.push_back(side2);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "RefinementPattern only supports quads and triangles right now.");
  }
  _childrenForSides = vector< vector< pair< int, int> > >(numSides);

  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    // find all the children that lie along this side....
    vector<double> v0 = vertices[refSides[sideIndex][0]];
    vector<double> v1 = vertices[refSides[sideIndex][1]];
    for (int childIndex=0; childIndex<numSubCells; childIndex++) {
      int numChildVertices = _subCells[childIndex].size();
      for (int childSideIndex=0; childSideIndex<numChildVertices; childSideIndex++) {
        vector<double> child_v0 = vertices[_subCells[childIndex][childSideIndex]];
        vector<double> child_v1 = vertices[_subCells[childIndex][(childSideIndex+1)%numChildVertices]];
        if ( colinear(v0,v1,child_v0) && colinear(v0,v1,child_v1) ) {
          // add this side...
          pair< int, int > entry = make_pair(childIndex,childSideIndex);
          _childrenForSides[sideIndex].push_back(entry);
          _parentSideForChildSide[entry] = sideIndex;
        }
      }
    }
    
    // sort _childrenForSides[sideIndex] according to child_v0's proximity to v0
    // bubble sort (we're likely to have at most a handful)
    int numEntriesForSideIndex = _childrenForSides[sideIndex].size();
    for (int entryIndex = 0; entryIndex < numEntriesForSideIndex; entryIndex++) {
      pair< int, int > entry = _childrenForSides[sideIndex][entryIndex];
      int childIndex = entry.first;
      int childSideIndex = entry.second;
      vector<double> child_v0 = vertices[_subCells[childIndex][childSideIndex]];
      double dist = distance(child_v0,v0);
      for (int secondEntryIndex = entryIndex+1; secondEntryIndex < numEntriesForSideIndex; secondEntryIndex++) {
        pair< int, int > secondEntry = _childrenForSides[sideIndex][secondEntryIndex];
        int secondChildIndex = secondEntry.first;
        int secondChildSideIndex = secondEntry.second;
        vector<double> secondChild_v0 = vertices[_subCells[secondChildIndex][secondChildSideIndex]];
        double secondDist = distance(secondChild_v0,v0);
        if ( secondDist < dist) {
          // swap secondEntry with entry
          _childrenForSides[sideIndex][entryIndex] = secondEntry;
          _childrenForSides[sideIndex][secondEntryIndex] = entry;
          // new distance to beat...
          dist = secondDist;
        }
      }
    }
  }
}

map< int, int > RefinementPattern::parentSideLookupForChild(int childIndex) {
  // returns a map for the child: childSideIndex --> parentSideIndex
  // (only populated for childSideIndices that are shared with the parent)
  map<int, int> lookupTable;
  int numSides = _cellTopoPtr->getSideCount();
  for (int childSideIndex = 0; childSideIndex<numSides; childSideIndex++) {
    pair< int, int > entry = make_pair(childIndex,childSideIndex);
    if ( _parentSideForChildSide.find(entry) != _parentSideForChildSide.end() ) {
      lookupTable[childSideIndex] = _parentSideForChildSide[entry];
    }
  }
  return lookupTable;
}

bool RefinementPattern::colinear(const vector<double> &v1_outside, const vector<double> &v2_outside, const vector<double> &v3_maybe_inside) {
  double tol = 1e-14;
  double d1 = distance(v1_outside,v3_maybe_inside);
  double d2 = distance(v3_maybe_inside,v2_outside);
  double d3 = distance(v1_outside,v2_outside);
  
  return abs(d1 + d2 - d3) < tol;
}

double RefinementPattern::distance(const vector<double> &v1, const vector<double> &v2) {
  int spaceDim = v1.size();
  double distance = 0.0;
  for (int i=0; i< spaceDim; i++) {
    double sqrt_dist = v1[i] - v2[i];
    distance += sqrt_dist*sqrt_dist;
  }
  return sqrt(distance);
}

FieldContainer<double> RefinementPattern::verticesForRefinement(FieldContainer<double> &cellNodes) {
  // compute the post-refinement physical vertices for the parent cell(s) given in cellNodes
  
  FieldContainer<double> verticesFC;
  bool singleCellNode = false;
  if ( cellNodes.rank() == 2) {
    // single cell
    singleCellNode = true;
    verticesFC.resize(1,_vertices.dimension(0),_vertices.dimension(1));
    cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
  } else if (cellNodes.rank() == 3) {
    int numCells = cellNodes.dimension(0);
    verticesFC.resize(numCells, _vertices.dimension(0), _vertices.dimension(1));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"cellNodes should be rank 2 or 3");
  }
  
  CellTools<double>::mapToPhysicalFrame(verticesFC,_vertices,cellNodes,*_cellTopoPtr);

  if (singleCellNode) {
    // caller will expect rank-2 return
    verticesFC.resize(_vertices.dimension(0),_vertices.dimension(1));
  }
  return verticesFC;
}

const FieldContainer<double> & RefinementPattern::verticesOnReferenceCell() {
  return _vertices;
}

vector< vector<int> > RefinementPattern::children(map<int, int> &localToGlobalVertexIndex) { 
  // localToGlobalVertexIndex key: index in vertices; value: index in _vertices
  // children returns a vector of global vertex indices for each child
  
  int numChildren = _subCells.size();
  vector< vector<int> > children(numChildren);
  vector< vector<int> >::iterator subCellIt;
  vector< vector<int> >::iterator childIt = children.begin();
  for (subCellIt=_subCells.begin(); subCellIt != _subCells.end(); subCellIt++) {
//    cout << "child global vertex indices: ";
    int numVertices = (*subCellIt).size();
    *childIt = vector<int>(numVertices);
    vector<int>::iterator vertexIt;
    vector<int>::iterator childVertexIt = (*childIt).begin();
    for (vertexIt = (*subCellIt).begin(); vertexIt != (*subCellIt).end(); vertexIt++) {
      int localIndex = *vertexIt;
      int globalIndex = localToGlobalVertexIndex[localIndex];
      
      *childVertexIt = globalIndex;
//      cout << globalIndex << " ";
      childVertexIt++;
    }
//    cout << endl;
    childIt++;
  }
  return children;
}

vector< vector< pair< int, int> > > & RefinementPattern::childrenForSides() {
  // outer vector: indexed by parent's sides; inner vector: (child index in children, index of child's side shared with parent)
  
  return _childrenForSides;
}


Teuchos::RCP<RefinementPattern> RefinementPattern::noRefinementPatternTriangle() {
  // TODO: implement this
  return Teuchos::rcp( (RefinementPattern*) NULL );
}

Teuchos::RCP<RefinementPattern> RefinementPattern::noRefinementPatternQuad() {
  FieldContainer<double> quadPoints(1,4,2);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 1.0;  
  Teuchos::RCP< shards::CellTopology > quad_4_ptr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  
  return Teuchos::rcp( new RefinementPattern(quad_4_ptr,quadPoints) );
}

Teuchos::RCP<RefinementPattern> RefinementPattern::regularRefinementPatternTriangle() {
  FieldContainer<double> triPoints(4,3,2);
  triPoints(0,0,0) = 0.0; // x1
  triPoints(0,0,1) = 0.0; // y1
  triPoints(0,1,0) = 0.5;
  triPoints(0,1,1) = 0.0;
  triPoints(0,2,0) = 0.0;
  triPoints(0,2,1) = 0.5;  
  triPoints(1,0,0) = 0.5; // x1
  triPoints(1,0,1) = 0.0; // y1
  triPoints(1,1,0) = 0.5;
  triPoints(1,1,1) = 0.5;
  triPoints(1,2,0) = 0.0;
  triPoints(1,2,1) = 0.5;  
  triPoints(2,0,0) = 0.5; // x1
  triPoints(2,0,1) = 0.0; // y1
  triPoints(2,1,0) = 1.0;
  triPoints(2,1,1) = 0.0;
  triPoints(2,2,0) = 0.5;
  triPoints(2,2,1) = 0.5;  
  triPoints(3,0,0) = 0.0; // x1
  triPoints(3,0,1) = 0.5; // y1
  triPoints(3,1,0) = 0.5;
  triPoints(3,1,1) = 0.5;
  triPoints(3,2,0) = 0.0;
  triPoints(3,2,1) = 1.0; 
  Teuchos::RCP< shards::CellTopology > tri_3_ptr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() ));
  
  return Teuchos::rcp( new RefinementPattern(tri_3_ptr,triPoints) );
}

Teuchos::RCP<RefinementPattern> RefinementPattern::regularRefinementPatternQuad() {
  // order of the sub-elements is CCW starting at bottom left
  FieldContainer<double> quadPoints(4,4,2);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 0.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 0.0;
  quadPoints(0,2,1) = 0.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 0.0;  
  quadPoints(1,0,0) = 0.0; // x1
  quadPoints(1,0,1) = -1.0; // y1
  quadPoints(1,1,0) = 1.0;
  quadPoints(1,1,1) = -1.0;
  quadPoints(1,2,0) = 1.0;
  quadPoints(1,2,1) = 0.0;
  quadPoints(1,3,0) = 0.0;
  quadPoints(1,3,1) = 0.0;  
  quadPoints(2,0,0) = 0.0; // x1
  quadPoints(2,0,1) = 0.0; // y1
  quadPoints(2,1,0) = 1.0;
  quadPoints(2,1,1) = 0.0;
  quadPoints(2,2,0) = 1.0;
  quadPoints(2,2,1) = 1.0;
  quadPoints(2,3,0) = 0.0;
  quadPoints(2,3,1) = 1.0;  
  quadPoints(3,0,0) = -1.0; // x1
  quadPoints(3,0,1) = 0.0; // y1
  quadPoints(3,1,0) = 0.0;
  quadPoints(3,1,1) = 0.0;
  quadPoints(3,2,0) = 0.0;
  quadPoints(3,2,1) = 1.0;
  quadPoints(3,3,0) = -1.0;
  quadPoints(3,3,1) = 1.0; 
  Teuchos::RCP< shards::CellTopology > quad_4_ptr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  
  return Teuchos::rcp( new RefinementPattern(quad_4_ptr,quadPoints) );
}

Teuchos::RCP<RefinementPattern> RefinementPattern::xAnisotropicRefinementPatternQuad() {
  // order of the sub-elements is CCW starting at bottom left
  FieldContainer<double> quadPoints(2,4,2);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 0.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 0.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 1.0;  
  quadPoints(1,0,0) = 0.0; // x1
  quadPoints(1,0,1) = -1.0; // y1
  quadPoints(1,1,0) = 1.0;
  quadPoints(1,1,1) = -1.0;
  quadPoints(1,2,0) = 1.0;
  quadPoints(1,2,1) = 1.0;
  quadPoints(1,3,0) = 0.0;
  quadPoints(1,3,1) = 1.0;   
  Teuchos::RCP< shards::CellTopology > quad_4_ptr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  
  return Teuchos::rcp( new RefinementPattern(quad_4_ptr,quadPoints) );
}

Teuchos::RCP<RefinementPattern> RefinementPattern::yAnisotropicRefinementPatternQuad() {
  // order of the sub-elements is CCW starting at bottom left
  FieldContainer<double> quadPoints(2,4,2);
  quadPoints(0,0,0) = -1.0; // x1
  quadPoints(0,0,1) = -1.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = -1.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 0.0;
  quadPoints(0,3,0) = -1.0;
  quadPoints(0,3,1) = 0.0;  
  quadPoints(1,0,0) = -1.0; // x1
  quadPoints(1,0,1) = 0.0; // y1
  quadPoints(1,1,0) = 1.0;
  quadPoints(1,1,1) = 0.0;
  quadPoints(1,2,0) = 1.0;
  quadPoints(1,2,1) = 1.0;
  quadPoints(1,3,0) = -1.0;
  quadPoints(1,3,1) = 1.0;   
  Teuchos::RCP< shards::CellTopology > quad_4_ptr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  
  return Teuchos::rcp( new RefinementPattern(quad_4_ptr,quadPoints) );
}
