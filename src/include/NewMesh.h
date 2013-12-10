//
//  NewMesh.h
//  Camellia-debug
//
//  Created by Nate Roberts on 12/2/13.
//
//

#ifndef Camellia_debug_NewMesh_h
#define Camellia_debug_NewMesh_h

#include "Shards_CellTopology.hpp"
#include "Intrepid_FieldContainer.hpp"

#include "ParametricCurve.h"
#include "RefinementPattern.h"

using namespace std;

typedef pair<int, int> Edge;
typedef Teuchos::RCP< shards::CellTopology > CellTopoPtr;

class NewMeshGeometry {
  vector< vector<double> > _vertices;
  vector< vector<unsigned> > _elementVertices;
  map< Edge, ParametricCurvePtr > _edgeToCurveMap;
  vector< CellTopoPtr > _cellTopos;
public:
  NewMeshGeometry(const vector< vector<double> > &vertices,
                  const vector< vector<unsigned> > &elementVertices,
                  const vector< CellTopoPtr > &cellTopos,
                  const map< Edge, ParametricCurvePtr > &edgeToCurveMap) {
    _vertices = vertices;
    _elementVertices = elementVertices;
    _cellTopos = cellTopos;
    _edgeToCurveMap = edgeToCurveMap;
  }
  
  NewMeshGeometry(const vector< vector<double> > &vertices,
                  const vector< vector<unsigned> > &elementVertices,
                  const vector< CellTopoPtr > &cellTopos) {
    _vertices = vertices;
    _elementVertices = elementVertices;
    _cellTopos = cellTopos;
  }
  
  map< Edge, ParametricCurvePtr > &edgeToCurveMap() {
    return _edgeToCurveMap;
  }
  
  vector< vector<unsigned> > &elementVertices() {
    return _elementVertices;
  }
  
  vector< vector<double> > &vertices() {
    return _vertices;
  }
  
  const vector< CellTopoPtr > &cellTopos() {
    return _cellTopos;
  }
};

typedef Teuchos::RCP<NewMeshGeometry> NewMeshGeometryPtr;

// "cells" are geometric entities -- they do not define any kind of basis
// "elements" are cells endowed with a (local) functional discretization
class NewMeshCell {
  unsigned _cellIndex;
  CellTopoPtr _cellTopo;
  vector< unsigned > _vertices;
  vector< vector<unsigned> > _entityIndices;  // indices: [subcdim][subcord]
  vector< map< unsigned, unsigned > > _subcellPermutations; // permutation to get from local ordering to the canonical one
  
  // for parents:
  vector< Teuchos::RCP< NewMeshCell > > _children;
  RefinementPatternPtr _refPattern;
public:
  NewMeshCell(CellTopoPtr cellTopo, const vector<unsigned> &vertices, const vector< map< unsigned, unsigned > > &subcellPermutations,
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

  const vector< Teuchos::RCP< NewMeshCell > > &children() {
    return _children;
  }
  void setChildren(vector< Teuchos::RCP< NewMeshCell > > children) {
    _children = children;
  }
  vector<unsigned> getChildIndices() {
    vector<unsigned> indices(_children.size());
    for (unsigned childOrdinal=0; childOrdinal<_children.size(); childOrdinal++) {
      indices[childOrdinal] = _children[childOrdinal]->cellIndex();
    }
    return indices;
  }
  
  unsigned entityIndex(unsigned subcdim, unsigned subcord) {
    return _entityIndices[subcdim][subcord];
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

typedef Teuchos::RCP<NewMeshCell> NewMeshCellPtr;

class NewMesh {
  unsigned _spaceDim; // dimension of the mesh
  
  map< vector<double>, unsigned > _vertexMap; // maps into indices in the vertices list -- here just for vertex identification (i.e. so we don't add the same vertex twice)
  
  vector< vector<double> > _vertices; // vertex locations
  
  // the following entity vectors are indexed on dimension of the entities
  vector< vector< set<unsigned> > > _entities; // vertices, edges, faces, solids, etc., up to dimension (_spaceDim - 1)
  vector< map< set<unsigned>, unsigned > > _knownEntities; // map keys are sets of vertices, values are entity indices in _entities[d]
  vector< map< unsigned, vector<unsigned> > > _canonicalEntityOrdering; // since we'll have one of these for each entity, could replace map with a vector
  vector< map< unsigned, set< pair<unsigned, unsigned> > > > _activeCellsForEntities; // set entries are (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)--I'm vascillating on whether this should contain entries for active ancestral cells.  Today, I think it should not.  I think we should have another set of activeEntities.  Things in that list either themselves have active cells or an ancestor that has an active cell.  So if your parent is inactive and you don't have any active cells of your own, then you know you can deactivate.
  vector< set< unsigned > > _activeEntities; // see note above
  vector< map< unsigned, unsigned > > _constrainingEntities; // map from broken entity to the whole (constraining) one.  May be "virtual" in the sense that there are no active cells that have the constraining entity as a subcell topology.
  vector< map< unsigned, set< unsigned > > > _constrainedEntities; // map from constraining entity to all broken ones constrained by it.
  vector< map< unsigned, vector< pair<unsigned, unsigned> > > > _parentEntities; // map from entity to its possible parents.  Not every entity has a parent.  We support entities having multiple parents.  Such things will be useful in the context of anisotropic refinements.  The pair entries here are (parentEntityIndex, refinementIndex), where the refinementIndex is the index into the _childEntities[d][parentEntityIndex] vector.
  vector< map< unsigned, vector< pair< RefinementPatternPtr, vector<unsigned> > > > > _childEntities; // map from parent to child entities, together with the RefinementPattern to get from one to the other.
  vector< map< unsigned, unsigned > > _entityCellTopologyKeys;
  
  vector< NewMeshCellPtr > _cells;
  set< unsigned > _activeCells;
  
  map< unsigned, shards::CellTopology > _knownTopologies; // key -> topo.  Might want to move this to a CellTopoFactory, but it is fairly simple

  set<unsigned> activeDescendants(unsigned d, unsigned entityIndex);
  set<unsigned> activeDescendantsNotInSet(unsigned d, unsigned entityIndex, const set<unsigned> &excludedSet);
  unsigned eldestActiveAncestor(unsigned d, unsigned entityIndex);
  unsigned addCell(CellTopoPtr cellTopo, const vector<unsigned> &cellVertices);
  unsigned addEntity(const shards::CellTopology &entityTopo, const vector<unsigned> &entityVertices, unsigned &entityPermutation); // returns the entityIndex
  void deactivateCell(NewMeshCellPtr cell);
  set<unsigned> descendants(unsigned d, unsigned entityIndex);
  pair< unsigned, set<unsigned> > determineEntityConstraints(unsigned d, unsigned entityIndex);
  void addChildren(NewMeshCellPtr cell, const vector< CellTopoPtr > &childTopos, const vector< vector<unsigned> > &childVertices);
  unsigned getVertexIndexAdding(const vector<double> &vertex, double tol);
  vector<unsigned> getVertexIndices(const FieldContainer<double> &vertices);
  vector<unsigned> getVertexIndices(const vector< vector<double> > &vertices);
  map<unsigned, unsigned> getVertexIndicesMap(const FieldContainer<double> &vertices);
  void init(unsigned spaceDim);
  void printVertex(unsigned vertexIndex);
  void printVertices(set<unsigned> vertexIndices);
  void refineCellEntities(NewMeshCellPtr cell, RefinementPatternPtr refPattern); // ensures that the appropriate child entities exist, and parental relationships are recorded in _parentEntities
  void updateConstraintsForCells(const set<unsigned> &cellIndices);
public:
  NewMesh(unsigned spaceDim);
  NewMesh(NewMeshGeometryPtr meshGeometry);
  NewMeshCellPtr addCell(CellTopoPtr cellTopo, const vector< vector<double> > &cellVertices);
  bool entityHasParent(unsigned d, unsigned entityIndex);
  unsigned getActiveCellCount(unsigned d, unsigned entityIndex);
  NewMeshCellPtr getCell(unsigned cellIndex);
  set<unsigned> getChildEntities(unsigned d, unsigned entityIndex);
  unsigned getConstrainingEntityIndex(unsigned d, unsigned entityIndex);
  unsigned getEntityCount(unsigned d);
  unsigned getEntityParent(unsigned d, unsigned entityIndex, unsigned parentOrdinal=0);
  unsigned getFaceEdgeIndex(unsigned faceIndex, unsigned edgeOrdinalInFace);
  unsigned getSpaceDim();
  unsigned getSubEntityIndex(unsigned d, unsigned entityIndex, unsigned subEntityDim, unsigned subEntityOrdinal);
  void refineCell(unsigned cellIndex, RefinementPatternPtr refPattern);
  unsigned cellCount();
  unsigned activeCellCount();
  
  void printEntityVertices(unsigned d, unsigned entityIndex);
};

typedef Teuchos::RCP<NewMesh> NewMeshPtr;

#endif
