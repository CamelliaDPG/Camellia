//
//  MeshTopology.h
//  Camellia-debug
//
//  Created by Nate Roberts on 12/2/13.
//
//

#ifndef Camellia_debug_MeshTopology_h
#define Camellia_debug_MeshTopology_h

#include "Shards_CellTopology.hpp"
#include "Intrepid_FieldContainer.hpp"

#include "MeshGeometry.h"

#include "RefinementPattern.h"

#include "Cell.h"

using namespace std;

class MeshTransformationFunction;

class Mesh;
typedef Teuchos::RCP<Mesh> MeshPtr;

class Cell;
typedef Teuchos::RCP<Cell> CellPtr;

class MeshTopology {
  unsigned _spaceDim; // dimension of the mesh
  
  map< vector<double>, unsigned > _vertexMap; // maps into indices in the vertices list -- here just for vertex identification (i.e. so we don't add the same vertex twice)
  
  vector< vector<double> > _vertices; // vertex locations
  
  // the following entity vectors are indexed on dimension of the entities
  vector< vector< set<unsigned> > > _entities; // vertices, edges, faces, solids, etc., up to dimension (_spaceDim - 1)
  vector< map< set<unsigned>, unsigned > > _knownEntities; // map keys are sets of vertices, values are entity indices in _entities[d]
  vector< map< unsigned, vector<unsigned> > > _canonicalEntityOrdering; // since we'll have one of these for each entity, could replace map with a vector
  vector< map< unsigned, set< pair<unsigned, unsigned> > > > _activeCellsForEntities; // set entries are (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)--I'm vascillating on whether this should contain entries for active ancestral cells.  Today, I think it should not.  I think we should have another set of activeEntities.  Things in that list either themselves have active cells or an ancestor that has an active cell.  So if your parent is inactive and you don't have any active cells of your own, then you know you can deactivate.
  vector< map<unsigned, set<unsigned> > > _activeSidesForEntities; // map keys are entity indices of dimension d (the outer vector index); map values are entities of dimension _spaceDim-1 belonging to active cells that contain the entity indicated by the map key.
  map< unsigned, pair< pair<unsigned, unsigned>, pair<unsigned, unsigned> > > _cellsForSideEntities; // key: sideEntityIndex.  value.first is (cellIndex1, sideOrdinal1), value.second is (cellIndex2, sideOrdinal2).  On initialization, (cellIndex2, sideOrdinal2) == ((unsigned)-1,(unsigned)-1).
  set<unsigned> _boundarySides; // entities of dimension _spaceDim-1 on the mesh boundary
  vector< map< unsigned, vector< pair<unsigned, unsigned> > > > _parentEntities; // map from entity to its possible parents.  Not every entity has a parent.  We support entities having multiple parents.  Such things will be useful in the context of anisotropic refinements.  The pair entries here are (parentEntityIndex, refinementIndex), where the refinementIndex is the index into the _childEntities[d][parentEntityIndex] vector.
  vector< map< unsigned, vector< pair< RefinementPatternPtr, vector<unsigned> > > > > _childEntities; // map from parent to child entities, together with the RefinementPattern to get from one to the other.
  vector< map< unsigned, unsigned > > _entityCellTopologyKeys;
  
  vector< CellPtr > _cells;
  set< unsigned > _activeCells;
  set< unsigned > _rootCells; // cells without parents

  // these guys presently only support 2D:
  set< int > _cellIDsWithCurves;
  map< pair<unsigned, unsigned>, ParametricCurvePtr > _edgeToCurveMap;
  Teuchos::RCP<MeshTransformationFunction> _transformationFunction; // for dealing with those curves
  
  map< unsigned, shards::CellTopology > _knownTopologies; // key -> topo.  Might want to move this to a CellTopoFactory, but it is fairly simple
  
//  set<unsigned> activeDescendants(unsigned d, unsigned entityIndex);
//  set<unsigned> activeDescendantsNotInSet(unsigned d, unsigned entityIndex, const set<unsigned> &excludedSet);
  unsigned addCell(CellTopoPtr cellTopo, const vector<unsigned> &cellVertices, unsigned parentCellIndex = -1);
  void addCellForSide(unsigned cellIndex, unsigned sideOrdinal, unsigned sideEntityIndex);
  void addEdgeCurve(pair<unsigned,unsigned> edge, ParametricCurvePtr curve);
  unsigned addEntity(const shards::CellTopology &entityTopo, const vector<unsigned> &entityVertices, unsigned &entityPermutation); // returns the entityIndex
  void deactivateCell(CellPtr cell);
  set<unsigned> descendants(unsigned d, unsigned entityIndex);

//  pair< unsigned, set<unsigned> > determineEntityConstraints(unsigned d, unsigned entityIndex);
  void addChildren(CellPtr cell, const vector< CellTopoPtr > &childTopos, const vector< vector<unsigned> > &childVertices);
  unsigned getCellCountForSide(unsigned sideEntityIndex);
  vector< pair<unsigned,unsigned> > getConstrainingSideAncestry(unsigned int sideEntityIndex);   // pair: first is the sideEntityIndex of the ancestor; second is the refinementIndex of the refinement to get from parent to child (see _parentEntities and _childEntities)
  unsigned getVertexIndexAdding(const vector<double> &vertex, double tol);
  vector<unsigned> getVertexIndices(const FieldContainer<double> &vertices);
  vector<unsigned> getVertexIndices(const vector< vector<double> > &vertices);
  map<unsigned, unsigned> getVertexIndicesMap(const FieldContainer<double> &vertices);
  set<unsigned> getEntitiesForSide(unsigned sideEntityIndex, unsigned d);
  bool entityIsAncestor(unsigned d, unsigned ancestor, unsigned descendent);
  void init(unsigned spaceDim);
  unsigned maxConstraint(unsigned d, unsigned entityIndex1, unsigned entityIndex2);
  void printVertex(unsigned vertexIndex);
  void printVertices(set<unsigned> vertexIndices);
  void refineCellEntities(CellPtr cell, RefinementPatternPtr refPattern); // ensures that the appropriate child entities exist, and parental relationships are recorded in _parentEntities
public:
  MeshTopology(unsigned spaceDim);
  MeshTopology(MeshGeometryPtr meshGeometry);
  CellPtr addCell(CellTopoPtr cellTopo, const vector< vector<double> > &cellVertices);
  bool entityHasParent(unsigned d, unsigned entityIndex);
  unsigned getActiveCellCount(unsigned d, unsigned entityIndex);
  const set< pair<unsigned,unsigned> > &getActiveCellIndices(unsigned d, unsigned entityIndex); // first entry in pair is the cellIndex, the second is the index of the entity in that cell (the subcord).
  CellPtr getCell(unsigned cellIndex);
//  vector< pair< unsigned, unsigned > > getCellNeighbors(unsigned cellIndex, unsigned sideIndex); // second entry in return is the sideIndex in neighbor (note that in context of h-refinements, one or both of the sides may be broken)
//  pair< CellPtr, unsigned > getCellAncestralNeighbor(unsigned cellIndex, unsigned sideIndex);
  bool cellHasCurvedEdges(unsigned cellIndex);
  vector<unsigned> getChildEntities(unsigned d, unsigned entityIndex);
  set<unsigned> getChildEntitiesSet(unsigned d, unsigned entityIndex);
  unsigned getConstrainingEntityIndex(unsigned d, unsigned entityIndex);
  unsigned getEntityIndex(unsigned d, const set<unsigned> &nodeSet);
  unsigned getEntityCount(unsigned d);
  unsigned getEntityParent(unsigned d, unsigned entityIndex, unsigned parentOrdinal=0);
  unsigned getEntityParentForSide(unsigned d, unsigned entityIndex, unsigned parentSideEntityIndex);   // returns the entity index for the parent (which might be the entity itself) of entity (d,entityIndex) that is a subcell of side parentSideEntityIndex
  const vector<unsigned> &getEntityVertexIndices(unsigned d, unsigned entityIndex);
  unsigned getFaceEdgeIndex(unsigned faceIndex, unsigned edgeOrdinalInFace);
  unsigned getSpaceDim();
  unsigned getSubEntityCount(unsigned int d, unsigned int entityIndex, unsigned int subEntityDim);
  unsigned getSubEntityIndex(unsigned d, unsigned entityIndex, unsigned subEntityDim, unsigned subEntityOrdinal);
  bool getVertexIndex(const vector<double> &vertex, unsigned &vertexIndex, double tol=1e-14);
  const vector<double>& getVertex(unsigned vertexIndex);
  FieldContainer<double> physicalCellNodesForCell(unsigned cellIndex);
  void refineCell(unsigned cellIndex, RefinementPatternPtr refPattern);
  unsigned cellCount();
  unsigned activeCellCount();
  
  const set<unsigned> &getActiveCellIndices();
  vector<double> getCellCentroid(unsigned cellIndex);
  
  const set<unsigned> &getRootCellIndices();
  
  // 2D only:
  vector< ParametricCurvePtr > parametricEdgesForCell(unsigned cellID, bool neglectCurves);
  void setEdgeToCurveMap(const map< pair<int, int>, ParametricCurvePtr > &edgeToCurveMap, MeshPtr mesh);
  
  void printEntityVertices(unsigned d, unsigned entityIndex);
  
  // not sure this should ultimately be exposed -- using it now to allow correctly timed call to updateCells()
  // (will be transitioning from having MeshTransformationFunction talk to Mesh to having it talk to MeshTopology)
  Teuchos::RCP<MeshTransformationFunction> transformationFunction();
  
};

typedef Teuchos::RCP<MeshTopology> MeshTopologyPtr;

#endif
