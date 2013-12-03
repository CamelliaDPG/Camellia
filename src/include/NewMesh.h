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

struct ActiveEntityListEntry { // linked list to connect the active cells that share topological entities (vertices, edges, etc.)
  unsigned cellIndex;
  unsigned subcellOrdinal;
  ActiveEntityListEntry* nextEntry;
};

// "cells" are geometric entities -- they do not define any kind of basis
// "elements" are cells endowed with a (local) functional discretization
class NewMeshCell {
  CellTopoPtr _cellTopo;
  vector< unsigned > _vertices;
  vector< map< unsigned, unsigned > > _subcellPermutations; // permutation to get from local ordering to the canonical one
  vector< vector< ActiveEntityListEntry* > > _activeEntityEntries; // outer vector indexed by dim, inner by subcellOrdinal
  
  // for parents:
  vector< Teuchos::RCP< NewMeshCell > > _children;
  RefinementPatternPtr _refPattern;
public:
  NewMeshCell(CellTopoPtr cellTopo, const vector<unsigned> &vertices, const vector< map< unsigned, unsigned > > &subcellPermutations) {
    _cellTopo = cellTopo;
    _vertices = vertices;
    _subcellPermutations = subcellPermutations;
  }
  const vector< unsigned > &vertices() {return _vertices;}
  bool isParent() { return _children.size() > 0; }
  
  void setActiveEntityEntries(vector< vector< ActiveEntityListEntry* > > entries) {
    _activeEntityEntries = entries;
  }
  void setChildren(vector< Teuchos::RCP< NewMeshCell > > children) {
    _children = children;
  }
  void setRefinementPattern(RefinementPatternPtr refPattern) {
    _refPattern = refPattern;
  }
  RefinementPatternPtr refinementPattern() {
    return _refPattern;
  }
  CellTopoPtr topology() {
    return _cellTopo;
  }
};

typedef Teuchos::RCP<NewMeshCell> NewMeshCellPtr;

class NewMesh {
  unsigned _spaceDim; // dimension of the mesh
  
  map< vector<double>, unsigned > _vertexMap; // maps into indices in the vertices list -- here just for vertex identification (i.e. so we don't add the same vertex twice)
  
  vector< vector<double> > _vertices; // vertex locations
  
  // the following entity vectors are indexed on dimension of the entities
  vector< vector< set<unsigned> > > _entities; // vertices, edges, faces, solids, etc., up to dimension (_spaceDim - 1)
  vector< map< set<unsigned>, unsigned > > _knownEntities; // map keys are sets of vertices, values are entity indices in _entities[d]
  vector< map< unsigned, vector<unsigned> > > _canonicalEntityOrdering;
  vector< map< unsigned, ActiveEntityListEntry* > > _activeEntities;
//  vector< map< unsigned, set< pair<unsigned, unsigned> > > > _cellsForEntities; // set entries are (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)
  vector< map< unsigned, unsigned > > _constrainingEntities; // map from broken entity to the whole (constraining) one.  May be "virtual" in the sense that there are no active cells that have the constraining entity as a subcell topology.
  
  vector< NewMeshCellPtr > _cells;
  
  unsigned addCell(CellTopoPtr cellTopo, const vector<unsigned> &cellVertices);
  void addChildren(NewMeshCellPtr cell, const vector< CellTopoPtr > &childTopos, const vector< vector<unsigned> > &childVertices);
  unsigned getVertexIndexAdding(const vector<double> &vertex, double tol);
  map<unsigned, unsigned> getVertexIndices(const FieldContainer<double> &vertices);
public:
  NewMesh(NewMeshGeometryPtr meshGeometry);
  void refineCell(unsigned cellIndex, RefinementPatternPtr refPattern);
};

#endif
