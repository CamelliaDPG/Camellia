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

#include "MeshTopology.h"

#include "Intrepid_CellTools.hpp"
#include "CamelliaCellTools.h"

//#include "CamelliaDebugUtility.h" // includes print() methods.

using namespace Intrepid;

RefinementPattern::RefinementPattern(Teuchos::RCP< shards::CellTopology > cellTopoPtr, FieldContainer<double> refinedNodes, vector< RefinementPatternPtr > sideRefinementPatterns) { 
  _cellTopoPtr = cellTopoPtr;
  _nodes = refinedNodes;
  _sideRefinementPatterns = sideRefinementPatterns;
  
  int numSubCells = refinedNodes.dimension(0);
  int numNodesPerCell = refinedNodes.dimension(1);
  unsigned spaceDim = refinedNodes.dimension(2);
  unsigned sideCount = (_cellTopoPtr->getDimension() > 1) ? _cellTopoPtr->getSideCount() : _cellTopoPtr->getNodeCount();
  _childrenForSides = vector< vector< pair< unsigned, unsigned> > >(sideCount); // will populate below..
  
  if (_cellTopoPtr->getNodeCount() == numNodesPerCell) {
    _childTopos = vector< CellTopoPtr >(numSubCells, _cellTopoPtr);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "RefinementPattern: Still need to implement support for child topos that have different topology than parent...");
  }
  
  if (spaceDim > 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(sideRefinementPatterns.size() != sideCount, std::invalid_argument, "sideRefinementPatterns length != cellTopo.getSideCount()");
  }
  
  unsigned sideDim = spaceDim - 1;

  if (sideDim > 0) {
    // we will fill some entries in _patternForSubcell repeatedly/redundantly
    _patternForSubcell = vector< vector< RefinementPatternPtr > >(spaceDim);
    for (unsigned d=1; d<spaceDim; d++) {
      _patternForSubcell[d] = vector< RefinementPatternPtr >(_cellTopoPtr->getSubcellCount(d));
    }
    _patternForSubcell[sideDim] = vector< RefinementPatternPtr >(sideCount);
    for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
      _patternForSubcell[sideDim][sideOrdinal] = _sideRefinementPatterns[sideOrdinal];
      for (unsigned d=1; d<sideDim; d++) {
        shards::CellTopology sideTopo = _cellTopoPtr->getCellTopologyData(sideDim, sideOrdinal);
        unsigned sideSubcellCount = sideTopo.getSubcellCount(d);
        for (unsigned sideSubcellOrdinal=0; sideSubcellOrdinal<sideSubcellCount; sideSubcellOrdinal++) {
          unsigned subcord = CamelliaCellTools::subcellOrdinalMap(*cellTopoPtr, sideDim, sideOrdinal, d, sideSubcellOrdinal);
          _patternForSubcell[d][subcord] = _sideRefinementPatterns[sideOrdinal]->patternForSubcell(d, sideSubcellOrdinal);
        }
      }
    }
  }
  
  vector<double> vertex(spaceDim);
  
  unsigned cellKey = cellTopoPtr->getKey();
  
  map< vector<double>, unsigned> vertexLookup;
  vector< vector<double> > vertices;
  for (int cellIndex=0; cellIndex<numSubCells; cellIndex++) {
    vector<unsigned> subCellNodes;
    for (int nodeIndex=0; nodeIndex<numNodesPerCell; nodeIndex++) {
      for (int dim=0; dim<spaceDim; dim++) {
        vertex[dim] = refinedNodes(cellIndex,nodeIndex,dim);
      }
      unsigned vertexIndex;
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
  
  // create MeshTopology
  _refinementTopology = Teuchos::rcp( new MeshTopology(spaceDim) );
  RefinementPatternPtr thisPtr = Teuchos::rcp( this, false );
  
  FieldContainer<double> refCellNodes(cellTopoPtr->getNodeCount(),spaceDim);
  CamelliaCellTools::refCellNodesForTopology(refCellNodes, *cellTopoPtr);
  vector< vector<double> > refCellNodesVector;
  for (int nodeIndex=0; nodeIndex<refCellNodes.dimension(0); nodeIndex++) {
    vector<double> node;
    for (int d=0; d<spaceDim; d++) {
      node.push_back(refCellNodes(nodeIndex,d));
    }
    refCellNodesVector.push_back(node);
  }
  
  _refinementTopology->addCell(cellTopoPtr, refCellNodesVector);
  _refinementTopology->refineCell(0, thisPtr);
  
  CellPtr parentCell = _refinementTopology->getCell(0);
  
  // populate _childrenForSides and _parentSideForChildSide
  for (int sideIndex = 0; sideIndex < sideCount; sideIndex++) { // sideIndices in parent
//    cout << "sideIndex " << sideIndex << endl;
    int sideEntityIndex = parentCell->entityIndex(sideDim, sideIndex);
//    cout << "sideEntityIndex " << sideEntityIndex << endl;
    // determine which sides are children (refinements) of the side:
    set<unsigned> sideChildEntities = _refinementTopology->getChildEntitiesSet(sideDim, sideEntityIndex);
    if (sideChildEntities.size() == 0) { // unrefined side; for our purposes it is its own parent
      sideChildEntities.insert(sideEntityIndex);
    }
//    print("sideChildEntities", sideChildEntities);
    // search for these entities within the (volume) children
    vector<unsigned> childCellIndices = parentCell->getChildIndices();
//    print("childCellIndices", childCellIndices);
    for (int childIndexInParent = 0; childIndexInParent<childCellIndices.size(); childIndexInParent++) {
      unsigned childCellIndex = childCellIndices[childIndexInParent];
      CellPtr childCell = _refinementTopology->getCell(childCellIndex);
      int childSideCount = (childCell->topology()->getDimension() > 1) ? childCell->topology()->getSideCount() : childCell->topology()->getNodeCount();
      for (int childSideIndex=0; childSideIndex<childSideCount; childSideIndex++) {
        unsigned childSideEntityIndex = childCell->entityIndex(sideDim, childSideIndex);
        if (sideChildEntities.find(childSideEntityIndex) != sideChildEntities.end()) {
          // this child shares side with parent
          pair<int,int> entry = make_pair(childIndexInParent, childSideIndex);
          _childrenForSides[sideIndex].push_back(entry);
          _parentSideForChildSide[entry] = sideIndex;
        }
      }
    }
  }
  
  _sideRefinementChildIndices = vector< vector<unsigned> >(sideCount); // maps from index of child in side refinement to the index in volume refinement pattern
  for (int sideIndex = 0; sideIndex < sideCount; sideIndex++) { // sideIndices in parent
    int sideEntityIndex = parentCell->entityIndex(sideDim, sideIndex);
    vector<unsigned> sideChildEntities = _refinementTopology->getChildEntities(sideDim, sideEntityIndex); // these are in the order of the side refinement pattern
    if (sideChildEntities.size() == 0) { // unrefined side; for our purposes it is its own parent
      sideChildEntities.push_back(sideEntityIndex);
    }
    for (vector<unsigned>::iterator sideEntityIndexIt = sideChildEntities.begin(); sideEntityIndexIt != sideChildEntities.end(); sideEntityIndexIt++) {
      unsigned sideChildEntityIndex = *sideEntityIndexIt;
      // need to find the volume child that has this side
      vector<unsigned> childCellIndices = parentCell->getChildIndices();
      for (int childIndexInParent = 0; childIndexInParent<childCellIndices.size(); childIndexInParent++) {
        unsigned childCellIndex = childCellIndices[childIndexInParent];
        CellPtr childCell = _refinementTopology->getCell(childCellIndex);
        int childSideCount = childCell->topology()->getSideCount();
        for (int childSideIndex=0; childSideIndex<childSideCount; childSideIndex++) {
          if ( sideChildEntityIndex == childCell->entityIndex(sideDim, childSideIndex) ) {
            _sideRefinementChildIndices[sideIndex].push_back(childIndexInParent);
          }
        }
      }
    }
//    print("_sideRefinementChildIndices[sideIndex]",_sideRefinementChildIndices[sideIndex]);
  }
  
//  cout << "Before sorting (2D), _childrenForSides:\n";
//  for (int sideIndex=0; sideIndex<sideCount; sideIndex++) {
//    cout << "sideIndex " << sideIndex << ":\n";
//    for (int entryIndex=0; entryIndex<_childrenForSides[sideIndex].size(); entryIndex++) {
//      int childIndex = _childrenForSides[sideIndex][entryIndex].first;
//      int childSideIndex = _childrenForSides[sideIndex][entryIndex].second;
//      cout << "childIndex " << childIndex << ", childSideIndex " << childSideIndex << endl;
//    }
//  }
  
  for (int sideIndex = 0; sideIndex < sideCount; sideIndex++) { // sideIndices in parent
    // the following code is replicated from the old code populating _childrenForSides
    // the upshot is that it sorts _childrenForSides in a particular way.  It seems likely this is
    // important to some legacy code, so I'm leaving it in place.  But probably we should remove it
    // at some point...
    if (spaceDim==2) {
      vector< vector< int > > refSides;
      if (cellKey == shards::Quadrilateral<4>::key ) {
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
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "In 2D, RefinementPattern only supports quads and triangles.  This is for legacy reasons to do with childrenForSides, and probably this restriction can soon be eliminated.");
      }

      vector<double> v0 = vertices[refSides[sideIndex][0]];
      vector<double> v1 = vertices[refSides[sideIndex][1]];
    
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
  
//  cout << "At end of RefinementPattern construction, _childrenForSides:\n";
//  for (int sideIndex=0; sideIndex<sideCount; sideIndex++) {
//    cout << "sideIndex " << sideIndex << ":\n";
//    for (int entryIndex=0; entryIndex<_childrenForSides[sideIndex].size(); entryIndex++) {
//      int childIndex = _childrenForSides[sideIndex][entryIndex].first;
//      int childSideIndex = _childrenForSides[sideIndex][entryIndex].second;
//      cout << "childIndex " << childIndex << ", childSideIndex " << childSideIndex << endl;
//    }
//  }
}

map< unsigned, unsigned > RefinementPattern::parentSideLookupForChild(unsigned childIndex) {
  // returns a map for the child: childSideIndex --> parentSideIndex
  // (only populated for childSideIndices that are shared with the parent)
  map<unsigned, unsigned> lookupTable;
  int numSides = _cellTopoPtr->getSideCount();
  for (unsigned childSideIndex = 0; childSideIndex<numSides; childSideIndex++) {
    pair< unsigned, unsigned > entry = make_pair(childIndex,childSideIndex);
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

vector< vector<GlobalIndexType> > RefinementPattern::children(const map<unsigned, GlobalIndexType> &localToGlobalVertexIndex) {
  // localToGlobalVertexIndex key: index in vertices; value: index in _vertices
  // children returns a vector of global vertex indices for each child
  
  int numChildren = _subCells.size();
  vector< vector<GlobalIndexType> > children(numChildren);
  vector< vector<unsigned> >::iterator subCellIt;
  vector< vector<GlobalIndexType> >::iterator childIt = children.begin();
  for (subCellIt=_subCells.begin(); subCellIt != _subCells.end(); subCellIt++) {
//    cout << "child global vertex indices: ";
    int numVertices = (*subCellIt).size();
    *childIt = vector<GlobalIndexType>(numVertices);
    vector<unsigned>::iterator vertexIt;
    vector<GlobalIndexType>::iterator childVertexIt = (*childIt).begin();
    for (vertexIt = (*subCellIt).begin(); vertexIt != (*subCellIt).end(); vertexIt++) {
      unsigned localIndex = *vertexIt;
      GlobalIndexType globalIndex = localToGlobalVertexIndex.find(localIndex)->second;
      
      *childVertexIt = globalIndex;
//      cout << globalIndex << " ";
      childVertexIt++;
    }
//    cout << endl;
    childIt++;
  }
  return children;
}

vector< vector< pair< unsigned, unsigned> > > & RefinementPattern::childrenForSides() {
  // outer vector: indexed by parent's sides; inner vector: (child ordinal in children, ordinal of child's side shared with parent)
  
  return _childrenForSides;
}

Teuchos::RCP< shards::CellTopology > RefinementPattern::childTopology(unsigned childIndex) {
  return _childTopos[childIndex];
}

void RefinementPattern::initializeAnisotropicRelationships() {
  static bool initialized = false;
  
  // guard to avoid nested calls to this method...
  if (!initialized) {
    initialized = true;
    // quad and hex refinements
    RefinementPatternPtr verticalCutQuad = xAnisotropicRefinementPatternQuad();
    RefinementPatternPtr horizontalCutQuad = yAnisotropicRefinementPatternQuad();
    RefinementPatternPtr isotropicRefinementQuad = regularRefinementPatternQuad();
    
    vector< RefinementPatternRecipe > quadRecipes;
    RefinementPatternRecipe recipe;
    vector< unsigned > initialCell; // empty to specify initial cell
    vector< unsigned > firstCutChild0(1,0);  // 0 to pick first child
    vector< unsigned > firstCutChild1(1,1);  // 1 to pick second child
    
    recipe.push_back(make_pair(verticalCutQuad.get(),initialCell));
    recipe.push_back(make_pair(horizontalCutQuad.get(),firstCutChild0));
    recipe.push_back(make_pair(horizontalCutQuad.get(),firstCutChild1));
    quadRecipes.push_back(recipe);
    
    recipe.clear();
    recipe.push_back(make_pair(horizontalCutQuad.get(),initialCell));
    recipe.push_back(make_pair(verticalCutQuad.get(),firstCutChild0));
    recipe.push_back(make_pair(verticalCutQuad.get(),firstCutChild1));
    quadRecipes.push_back(recipe);
    
    recipe.clear();
    recipe.push_back(make_pair(isotropicRefinementQuad.get(),initialCell));
    quadRecipes.push_back(recipe);
    
    verticalCutQuad->setRelatedRecipes(quadRecipes);
    horizontalCutQuad->setRelatedRecipes(quadRecipes);
    isotropicRefinementQuad->setRelatedRecipes(quadRecipes);
    
    // TODO: once we add anisotropic refinements for the hexahedron, marry them here -- uncomment the following:
  /*  RefinementPatternPtr isotropicRefinementHex = regularRefinementPatternHexahedron();
    RefinementPatternPtr verticalCutHex = xAnisotropicRefinementPatternHexahedron();
    RefinementPatternPtr horizontalCutHex = yAnisotropicRefinementPatternHexahedron();
    RefinementPatternPtr depthCutHex = zAnisotropicRefinementPatternHexahedron();
   
   vector< unsigned > secondCutChild00(2,0);
   vector< unsigned > secondCutChild01(2);
   secondCutChild01[0] = 0;
   secondCutChild01[1] = 1;
   vector< unsigned > secondCutChild10(2);
   secondCutChild10[0] = 1;
   secondCutChild10[1] = 0;
   vector< unsigned > secondCutChild11(2);
   secondCutChild11[0] = 1;
   secondCutChild11[1] = 1;

   
    vector< RefinementPatternPtr > hexRefs;
    hexRefs.push_back(verticalCutHex);
    hexRefs.push_back(horizontalCutHex);
    hexRefs.push_back(depthCutHex);
    hexRefs.push_back(isotropicRefinementHex);
    
    verticalCutHex->setRelatedRefinementPatterns(hexRefs);
    horizontalCutHex->setRelatedRefinementPatterns(hexRefs);
    depthCutHex->setRelatedRefinementPatterns(hexRefs);
    isotropicRefinementHex->setRelatedRefinementPatterns(hexRefs);*/
  }
}

map<unsigned, set<unsigned> > RefinementPattern::getInternalSubcellOrdinals(RefinementBranch &refinements) {
  CellTopoPtr ancestralTopo = refinements[0].first->parentTopology();
  CellTopoPtr childTopo;
  set<unsigned> parentSidesToIntersect;
  int numSides = (ancestralTopo->getDimension() > 1) ? ancestralTopo->getSideCount() : ancestralTopo->getNodeCount();
  for (unsigned sideOrdinal=0; sideOrdinal < numSides; sideOrdinal++) {
    parentSidesToIntersect.insert(sideOrdinal);
  }
  for (int refIndex=0; refIndex<refinements.size(); refIndex++) {
    RefinementPattern* refPattern = refinements[refIndex].first;
    unsigned childOrdinal = refinements[refIndex].second;
    childTopo = refPattern->childTopology(childOrdinal);
    set<unsigned> childSidesThatMatch;
    for (set<unsigned>::iterator parentSideIt = parentSidesToIntersect.begin(); parentSideIt != parentSidesToIntersect.end(); parentSideIt++) {
      unsigned parentSide = *parentSideIt;
      vector< pair<unsigned, unsigned> > childrenForSide = refPattern->childrenForSides()[parentSide];       // vector: (child ordinal in children, ordinal of child's side shared with parent)
      for(vector< pair<unsigned, unsigned> >::iterator childForSideEntryIt = childrenForSide.begin();
          childForSideEntryIt != childrenForSide.end(); childForSideEntryIt++) {
        if (childForSideEntryIt->first == childOrdinal) {
          childSidesThatMatch.insert(childForSideEntryIt->second);
        }
      }
    }
    parentSidesToIntersect = childSidesThatMatch;
  }
  // once we get here, the entries in parentSidesToIntersect are exactly those descendant sides that have non-empty intersection with the ancestor's boundary.
  map<unsigned, set<unsigned> > externalSubcellOrdinals;
  unsigned spaceDim = ancestralTopo->getDimension();
  
  for (set<unsigned>::iterator externalSideIt = parentSidesToIntersect.begin(); externalSideIt != parentSidesToIntersect.end(); externalSideIt++) {
    unsigned externalSideOrdinal = *externalSideIt;
    shards::CellTopology sideTopo = childTopo->getCellTopologyData(spaceDim-1, externalSideOrdinal);
    for (unsigned d=0; d<spaceDim; d++) {
      set<unsigned> subcellsForSide; // of dimension d...
      unsigned scCount = sideTopo.getSubcellCount(d);
      for (unsigned scOrdinal=0; scOrdinal<scCount; scOrdinal++) {
        unsigned scOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(*childTopo, spaceDim-1, externalSideOrdinal, d, scOrdinal); // this is not a particularly efficient method...
        externalSubcellOrdinals[d].insert(scOrdinalInCell);
      }
    }
  }
  
  map<unsigned, set<unsigned> > internalSubcellOrdinals; // complement of externalSubcellOrdinals
  for (unsigned d=0; d<spaceDim; d++) {
    unsigned scCount = childTopo->getSubcellCount(d);
    for (unsigned scOrd=0; scOrd<scCount; scOrd++) {
      if (externalSubcellOrdinals[d].find(scOrd) == externalSubcellOrdinals[d].end()) {
        internalSubcellOrdinals[d].insert(scOrd);
      }
    }
  }
  return internalSubcellOrdinals;
}

unsigned RefinementPattern::mapSideChildIndex(unsigned sideIndex, unsigned sideRefinementChildIndex) {
//  print("sideRefinementChildIndices[sideIndex]",_sideRefinementChildIndices[sideIndex]);
  return _sideRefinementChildIndices[sideIndex][sideRefinementChildIndex];
}

unsigned RefinementPattern::mapSideOrdinalFromLeafToAncestor(unsigned descendantSideOrdinal, RefinementBranch &refinements) {
  // given a side ordinal in the leaf node of a branch, returns the corresponding side ordinal in the earliest ancestor in the branch.
  // returns (unsigned)-1 if the given sideOrdinal does not have a corresponding side in the earliest ancestor, as can happen if the descendant is interior to the ancestor.
  int numRefs = refinements.size();
  for (int i=numRefs-1; i>=0; i--) {
    unsigned childOrdinal = refinements[i].second;
    RefinementPattern* parentRefPattern = refinements[i].first;
    map<unsigned,unsigned> sideLookup = parentRefPattern->parentSideLookupForChild(childOrdinal);
    if (sideLookup.find(descendantSideOrdinal) == sideLookup.end()) {
      return -1;
    }
    descendantSideOrdinal = sideLookup[descendantSideOrdinal];
  }
  return descendantSideOrdinal;
}

RefinementPatternPtr RefinementPattern::noRefinementPattern(Teuchos::RCP< shards::CellTopology > cellTopoPtr) {
  static map< unsigned, RefinementPatternPtr > knownRefinementPatterns;
  unsigned key = cellTopoPtr->getKey();
  if (knownRefinementPatterns.find(key) == knownRefinementPatterns.end()) {
    unsigned numSides = cellTopoPtr->getSideCount();
    vector< RefinementPatternPtr > sideRefPatterns;
    unsigned d = cellTopoPtr->getDimension();
    if (d > 1) {
      for (unsigned sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
        Teuchos::RCP< shards::CellTopology > sideTopo = Teuchos::rcp( new shards::CellTopology(cellTopoPtr->getCellTopologyData(d-1, sideOrdinal)) );
        sideRefPatterns.push_back( noRefinementPattern(sideTopo) );
      }
    }
    FieldContainer<double> cellPoints(cellTopoPtr->getVertexCount(),d);
    CamelliaCellTools::refCellNodesForTopology(cellPoints, *cellTopoPtr);
    cellPoints.resize(1,cellPoints.dimension(0),cellPoints.dimension(1));
    
    knownRefinementPatterns[key] = Teuchos::rcp( new RefinementPattern(cellTopoPtr, cellPoints, sideRefPatterns) );
  }
  return knownRefinementPatterns[key];
}

Teuchos::RCP<RefinementPattern> RefinementPattern::noRefinementPatternLine() {
  static RefinementPatternPtr refPattern;
  
  if (refPattern.get() == NULL) {
    Teuchos::RCP< shards::CellTopology > line_2_ptr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<2> >() ));
    refPattern = noRefinementPattern(line_2_ptr);
  }
  
  return refPattern;
}

Teuchos::RCP<RefinementPattern> RefinementPattern::noRefinementPatternTriangle() {
  static RefinementPatternPtr refPattern;
  
  if (refPattern.get() == NULL) {
    Teuchos::RCP< shards::CellTopology > tri_3_ptr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() ));
    refPattern = noRefinementPattern(tri_3_ptr);
  }
  
  return refPattern;
}

Teuchos::RCP<RefinementPattern> RefinementPattern::noRefinementPatternQuad() {
  static RefinementPatternPtr refPattern;
  
  if (refPattern.get() == NULL) {
    Teuchos::RCP< shards::CellTopology > quad_4_ptr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
    refPattern = noRefinementPattern(quad_4_ptr);
  }
  return refPattern;
}

unsigned RefinementPattern::numChildren() {
  return _nodes.dimension(0);
}

RefinementPatternPtr RefinementPattern::regularRefinementPatternLine() {
  static RefinementPatternPtr refPattern;
  
  if (refPattern.get() == NULL) {
    Teuchos::RCP< shards::CellTopology > line_2_ptr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<2> >() ));

    FieldContainer<double> linePoints(2,2,1);
    linePoints(0,0,0) = -1.0;
    linePoints(0,1,0) =  0.0;
    linePoints(1,0,0) =  0.0;
    linePoints(1,1,0) =  1.0;
    refPattern = Teuchos::rcp( new RefinementPattern(line_2_ptr,linePoints,vector< RefinementPatternPtr >()) );
  }
  return refPattern;
}

Teuchos::RCP<RefinementPattern> RefinementPattern::regularRefinementPatternTriangle() {
  static RefinementPatternPtr refPattern;
  
  if (refPattern.get() == NULL) {

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
    
    refPattern = Teuchos::rcp( new RefinementPattern(tri_3_ptr,triPoints,vector<RefinementPatternPtr>(3,regularRefinementPatternLine())) );
  }
  return refPattern;
}

Teuchos::RCP<RefinementPattern> RefinementPattern::regularRefinementPatternQuad() {
  static RefinementPatternPtr refPattern;
  
  if (refPattern.get() == NULL) {
    
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
    
    refPattern = Teuchos::rcp( new RefinementPattern(quad_4_ptr,quadPoints,vector<RefinementPatternPtr>(4,regularRefinementPatternLine())) );
  }
  return refPattern;
}

Teuchos::RCP<RefinementPattern> RefinementPattern::regularRefinementPatternHexahedron() {
  static RefinementPatternPtr refPattern;
  
  if (refPattern.get() == NULL) {
    
    // order of the sub-elements is CCW starting at bottom left
    unsigned spaceDim = 3;
    
    // most of what follows should work for arbitrary spatial dimension -- could reimplement regular quad and line refinements in terms of this
    // (exceptions: CellTopology object, and the CamelliaCellTools::refCellNodesForTopology() call.)
    unsigned numChildren = 1 << spaceDim; // 2^3
    unsigned numNodesPerChild = 1 << spaceDim; // 2^3
    FieldContainer<double> hexPoints(numChildren,numNodesPerChild,spaceDim);
    FieldContainer<double> refHexPoints(numNodesPerChild,spaceDim);
    Teuchos::RCP< shards::CellTopology > hexTopo = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ));
    CamelliaCellTools::refCellNodesForTopology(refHexPoints, *hexTopo);
    
    // scale and shift ref points to be in the bottommost (in each spatial direction) corner.  xi --> (xi - 1) / 2
    for (int nodeIndex=0; nodeIndex<numNodesPerChild; nodeIndex++) {
      for (int d=0; d<spaceDim; d++) {
        refHexPoints(nodeIndex,d) = (refHexPoints(nodeIndex,d) - 1) / 2;
      }
    }
    
    for (int childIndex=0; childIndex<numChildren; childIndex++) {
      vector<double> offsets(spaceDim);
      int childIndexShifted = childIndex;
      for (int d=0; d<spaceDim; d++) {
        offsets[d] = (childIndexShifted%2 == 0) ? 0.0 : 1.0;
        childIndexShifted >>= 1;
      }
      for (int nodeIndex=0; nodeIndex<numNodesPerChild; nodeIndex++) {
        for (int d=0; d<spaceDim; d++) {
          hexPoints(childIndex,nodeIndex,d) = refHexPoints(nodeIndex,d) + offsets[d];
        }
      }
    }
    
    refPattern = Teuchos::rcp( new RefinementPattern(hexTopo,hexPoints,vector<RefinementPatternPtr>(6,regularRefinementPatternQuad())) );
  }
  return refPattern;
}

Teuchos::RCP<RefinementPattern> RefinementPattern::regularRefinementPattern(unsigned cellTopoKey) {
  switch (cellTopoKey) {
    case shards::Line<2>::key :
      return regularRefinementPatternLine();
    case shards::Quadrilateral<4>::key :
      return regularRefinementPatternQuad();
    case shards::Hexahedron<8>::key :
      return regularRefinementPatternHexahedron();
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported cellTopology");
  }
  return Teuchos::rcp( (RefinementPattern*) NULL );
}

// cuts a quad vertically (x-refines the element)
Teuchos::RCP<RefinementPattern> RefinementPattern::xAnisotropicRefinementPatternQuad() {
  static RefinementPatternPtr refPattern;
  
  if (refPattern.get() == NULL) {

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
    
    vector< RefinementPatternPtr > sideRefinements;
    sideRefinements.push_back(regularRefinementPatternLine());
    sideRefinements.push_back(     noRefinementPatternLine());
    sideRefinements.push_back(regularRefinementPatternLine());
    sideRefinements.push_back(     noRefinementPatternLine());
    
    refPattern = Teuchos::rcp( new RefinementPattern(quad_4_ptr,quadPoints,sideRefinements) );
  }
  return refPattern;
}

// cuts a quad horizontally (y-refines the element)
Teuchos::RCP<RefinementPattern> RefinementPattern::yAnisotropicRefinementPatternQuad() {
  static RefinementPatternPtr refPattern;
  
  if (refPattern.get() == NULL) {

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

    vector< RefinementPatternPtr > sideRefinements;
    sideRefinements.push_back(     noRefinementPatternLine());
    sideRefinements.push_back(regularRefinementPatternLine());
    sideRefinements.push_back(     noRefinementPatternLine());
    sideRefinements.push_back(regularRefinementPatternLine());
    
    refPattern = Teuchos::rcp( new RefinementPattern(quad_4_ptr,quadPoints, sideRefinements) );
  }
  return refPattern;
}

CellTopoPtr RefinementPattern::parentTopology() {
  return _cellTopoPtr;
}

RefinementPatternPtr RefinementPattern::patternForSubcell(unsigned subcdim, unsigned subcord) {
  return _patternForSubcell[subcdim][subcord];
}

const FieldContainer<double> & RefinementPattern::refinedNodes() {
  return _nodes;
}

MeshTopologyPtr RefinementPattern::refinementMeshTopology() {
  return _refinementTopology;
}

vector< RefinementPatternRecipe > & RefinementPattern::relatedRecipes() {
  // e.g. the anisotropic + isotropic refinements of the quad.  This should be an exhaustive list, and should be in order of increasing fineness--i.e. the isotropic refinement should come at the end of the list.  The current refinement pattern is required to be part of the list.  (A refinement pattern is related to itself.)
  return _relatedRecipes;
}
void RefinementPattern::setRelatedRecipes(vector<RefinementPatternRecipe> &recipes) {
  _relatedRecipes = recipes;
}

const vector< Teuchos::RCP<RefinementPattern> > & RefinementPattern::sideRefinementPatterns() {
  return _sideRefinementPatterns;
}

FieldContainer<double> RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(RefinementBranch refinementBranch) {
  CellTopoPtr parentTopo = refinementBranch[0].first->parentTopology();
  FieldContainer<double> ancestorNodes(parentTopo->getNodeCount(), parentTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(ancestorNodes, *parentTopo);
  
  return descendantNodes(refinementBranch, ancestorNodes);
}

FieldContainer<double> RefinementPattern::descendantNodes(RefinementBranch refinementBranch, const FieldContainer<double> &ancestorNodes) {
  vector< vector<double> > vertices;
  vector< vector<unsigned> > elementVertices;
  int numNodes = ancestorNodes.dimension(0);
  int spaceDim = ancestorNodes.dimension(1);
  
  vector<unsigned> ancestorVertexIndices;
  for (int nodeIndex = 0; nodeIndex < numNodes; nodeIndex++) {
    vector<double> node;
    for (int d=0; d<spaceDim; d++) {
      node.push_back(ancestorNodes(nodeIndex,d));
    }
    vertices.push_back(node);
    ancestorVertexIndices.push_back(nodeIndex);
  }
  elementVertices.push_back(ancestorVertexIndices);
  
  CellTopoPtr ancestorTopo = refinementBranch[0].first->parentTopology();
  vector< CellTopoPtr > cellTopos(1, ancestorTopo);
  
  MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );
  
  MeshTopologyPtr mesh = Teuchos::rcp( new MeshTopology(meshGeometry) );
  
  unsigned cellIndex = 0; // cellIndex of the current parent in the RefinementBranch
  for (int refIndex=0; refIndex<refinementBranch.size(); refIndex++) {
    RefinementPatternPtr tempRefPatternPtr = Teuchos::rcp(refinementBranch[refIndex].first, false);
    mesh->refineCell(cellIndex, tempRefPatternPtr);
    cellIndex = mesh->getCell(cellIndex)->getChildIndices()[refinementBranch[refIndex].second];
  }

  return mesh->physicalCellNodesForCell(cellIndex);
}

RefinementBranch RefinementPattern::sideRefinementBranch(RefinementBranch &volumeRefinementBranch, unsigned sideIndex) {
  RefinementBranch sideRefinements;
  if (volumeRefinementBranch.size()==0) return sideRefinements; // side refinement branch empty, too
  CellTopoPtr volumeTopo = volumeRefinementBranch[0].first->parentTopology();
  unsigned sideDim = volumeTopo->getDimension() - 1;
  if (sideDim == 0) {
    // then the empty refinement branch will suffice (since the "side" is actually a vertex)
    return sideRefinements;
  }
  
  for (int refIndex=0; refIndex<volumeRefinementBranch.size(); refIndex++) {
    RefinementPattern* refPattern = volumeRefinementBranch[refIndex].first;
    unsigned volumeBranchChild = volumeRefinementBranch[refIndex].second;
    RefinementPattern* sideRefPattern = refPattern->patternForSubcell(sideDim, sideIndex).get();
    
    int sideBranchChild = -1;
    for (int sideChildIndex = 0; sideChildIndex < sideRefPattern->numChildren(); sideChildIndex++) {
      if (refPattern->mapSideChildIndex(sideIndex, sideChildIndex) == volumeBranchChild) {
        sideBranchChild = sideChildIndex;
      }
    }
    if (sideBranchChild == -1) {
      cout << "RefinementPattern::sideRefinementBranch: Did not find child.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Did not find child");
    }
    
    sideRefinements.push_back(make_pair(sideRefPattern,sideBranchChild));
  }
  return sideRefinements;
}