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

#include "IndexType.h"

#include "PeriodicBC.h"

using namespace std;

class MeshTransformationFunction;
class GlobalDofAssignment;

class Mesh;
typedef Teuchos::RCP<Mesh> MeshPtr;

class Cell;
typedef Teuchos::RCP<Cell> CellPtr;

class MeshTopology {
  unsigned _spaceDim; // dimension of the mesh
  
  map< vector<double>, IndexType > _vertexMap; // maps into indices in the vertices list -- here just for vertex identification (i.e. so we don't add the same vertex twice)
  vector< vector<double> > _vertices; // vertex locations
  
  vector< PeriodicBCPtr > _periodicBCs;
  map<IndexType, set< pair<int, int> > > _periodicBCIndicesMatchingNode; // pair: first = index in _periodicBCs; second: 0 or 1, indicating first or second part of the identification matches.  IndexType is the vertex index.
  map< pair<IndexType, pair<int,int> >, IndexType > _equivalentNodeViaPeriodicBC;
  
  // the following entity vectors are indexed on dimension of the entities
  vector< vector< vector<IndexType> > > _entities; // vertices, edges, faces, solids, etc., up to dimension (_spaceDim - 1).  Innermost container is sorted by value of IndexType (nodes). The outer two indices are entityDim, entityIndex.
  vector< map< vector<IndexType>, IndexType > > _knownEntities; // map keys are vectors of sorted vertex indices, values are entity indices in _entities[d]
  vector< vector< vector<IndexType> > > _canonicalEntityOrdering;
  vector< vector< vector< pair<IndexType, unsigned> > > > _activeCellsForEntities; // inner vector entries are sorted (cellIndex, entityIndexInCell) (entityIndexInCell aka subcord)--I'm vascillating on whether this should contain entries for active ancestral cells.  Today, I think it should not.  I think we should have another set of activeEntities.  Things in that list either themselves have active cells or an ancestor that has an active cell.  So if your parent is inactive and you don't have any active cells of your own, then you know you can deactivate.
  vector< vector< vector<IndexType> > > _sidesForEntities; // vector indices: dimension d, entity index; innermost container stores entity indices of dimension _spaceDim-1 belonging to cells that contain the indicated entity, sorted by index.
  map< IndexType, pair< pair<IndexType, unsigned>, pair<IndexType, unsigned> > > _cellsForSideEntities; // key: sideEntityIndex.  value.first is (cellIndex1, sideOrdinal1), value.second is (cellIndex2, sideOrdinal2).  On initialization, (cellIndex2, sideOrdinal2) == ((IndexType)-1,(IndexType)-1).
  set<IndexType> _boundarySides; // entities of dimension _spaceDim-1 on the mesh boundary
  vector< map< IndexType, vector< pair<IndexType, unsigned> > > > _parentEntities; // map from entity to its possible parents.  Not every entity has a parent.  We support entities having multiple parents.  Such things will be useful in the context of anisotropic refinements.  The pair entries here are (parentEntityIndex, refinementOrdinal), where the refinementOrdinal is the index into the _childEntities[d][parentEntityIndex] vector.
  
  vector< map< IndexType, pair<IndexType, unsigned> > > _generalizedParentEntities; // map from entity to its nearest generalized parent.  map entries are (parentEntityIndex, parentEntityDimension).  Generalized parents may be higher-dimensional or equal-dimensional to the child entity.
  vector< map< IndexType, vector< pair< RefinementPatternPtr, vector<IndexType> > > > > _childEntities; // map from parent to child entities, together with the RefinementPattern to get from one to the other.
  vector< vector< Camellia::CellTopologyKey > > _entityCellTopologyKeys;
  
  vector< CellPtr > _cells;
  set< IndexType > _activeCells;
  set< IndexType > _rootCells; // cells without parents
  
  // these guys presently only support 2D:
  set< IndexType > _cellIDsWithCurves;
  map< pair<IndexType, IndexType>, ParametricCurvePtr > _edgeToCurveMap;
  Teuchos::RCP<MeshTransformationFunction> _transformationFunction; // for dealing with those curves
  
  map< pair<unsigned,unsigned>, CellTopoPtr > _knownTopologies; // (shards key, tensorial degree) -> topo.  Might want to move this to a CellTopoFactory, but it is fairly simple
  
  //  set<IndexType> activeDescendants(IndexType d, IndexType entityIndex);
  //  set<IndexType> activeDescendantsNotInSet(IndexType d, IndexType entityIndex, const set<IndexType> &excludedSet);
  IndexType addCell(CellTopoPtrLegacy cellTopo, const vector<IndexType> &cellVertices, IndexType parentCellIndex = -1);
  IndexType addCell(CellTopoPtr cellTopo, const vector<IndexType> &cellVertices, IndexType parentCellIndex = -1);
  void addCellForSide(IndexType cellIndex, unsigned sideOrdinal, IndexType sideEntityIndex);
  void addEdgeCurve(pair<IndexType,IndexType> edge, ParametricCurvePtr curve);
//  IndexType addEntity(const shards::CellTopology &entityTopo, const vector<IndexType> &entityVertices, unsigned &entityPermutation); // returns the entityIndex
  IndexType addEntity(CellTopoPtr entityTopo, const vector<IndexType> &entityVertices, unsigned &entityPermutation); // returns the entityIndex

  void deactivateCell(CellPtr cell);
  set<IndexType> descendants(unsigned d, IndexType entityIndex);
  
  //  pair< IndexType, set<IndexType> > determineEntityConstraints(unsigned d, IndexType entityIndex);
  void addChildren(CellPtr cell, const vector< CellTopoPtr > &childTopos, const vector< vector<IndexType> > &childVertices);
  
  void determineGeneralizedParentsForRefinement(CellPtr cell, RefinementPatternPtr refPattern);
  
  IndexType getVertexIndexAdding(const vector<double> &vertex, double tol);
  vector<IndexType> getVertexIndices(const FieldContainer<double> &vertices);
  vector<IndexType> getVertexIndices(const vector< vector<double> > &vertices);
  map<unsigned, IndexType> getVertexIndicesMap(const FieldContainer<double> &vertices);
  set<IndexType> getEntitiesForSide(IndexType sideEntityIndex, unsigned d);
  void init(unsigned spaceDim);
  unsigned maxConstraint(unsigned d, IndexType entityIndex1, IndexType entityIndex2);
  void printVertex(IndexType vertexIndex);
  void printVertices(set<IndexType> vertexIndices);
  void refineCellEntities(CellPtr cell, RefinementPatternPtr refPattern); // ensures that the appropriate child entities exist, and parental relationships are recorded in _parentEntities
  void setEntityGeneralizedParent(unsigned entityDim, IndexType entityIndex, unsigned parentDim, IndexType parentEntityIndex);
  
  GlobalDofAssignment* _gda; // for cubature degree lookups
  
  map<string, long long> approximateMemoryCosts(); // for each private variable
  
  void addSideForEntity(unsigned entityDim, IndexType entityIndex, IndexType sideEntityIndex); // maintains _sidesForEntities container
public:
  MeshTopology(unsigned spaceDim, vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>());
  MeshTopology(MeshGeometryPtr meshGeometry, vector<PeriodicBCPtr> periodicBCs=vector<PeriodicBCPtr>());
  CellPtr addCell(CellTopoPtr cellTopo, const vector< vector<double> > &cellVertices);
  CellPtr addCell(CellTopoPtrLegacy cellTopo, const vector< vector<double> > &cellVertices);
  
  // ! This method only gets within a factor of 2 or so, but can give a rough estimate
  long long approximateMemoryFootprint();
  
  // ! This method only gets within a factor of 2 or so, but can give rough estimates
  void printApproximateMemoryReport();
  
  bool entityHasParent(unsigned d, IndexType entityIndex);
  bool entityHasChildren(unsigned d, IndexType entityIndex);
  IndexType getActiveCellCount(unsigned d, IndexType entityIndex);
  
  const vector< pair<IndexType,IndexType> > &getActiveCellIndices(unsigned d, IndexType entityIndex); // first entry in pair is the cellIndex, the second is the index of the entity in that cell (the subcord).
  CellPtr getCell(IndexType cellIndex);
  //  vector< pair< unsigned, unsigned > > getCellNeighbors(unsigned cellIndex, unsigned sideIndex); // second entry in return is the sideIndex in neighbor (note that in context of h-refinements, one or both of the sides may be broken)
  //  pair< CellPtr, unsigned > getCellAncestralNeighbor(unsigned cellIndex, unsigned sideIndex);
  bool cellHasCurvedEdges(IndexType cellIndex);
  
  bool cellContainsPoint(GlobalIndexType cellID, const std::vector<double> &point, int cubatureDegree);
  std::vector<IndexType> cellIDsForPoints(const FieldContainer<double> &physicalPoints);
  
  bool entityIsAncestor(unsigned d, IndexType ancestor, IndexType descendent);
  
  CellPtr findCellWithVertices(const vector< vector<double> > &cellVertices);
  
  vector<IndexType> getChildEntities(unsigned d, IndexType entityIndex);
  set<IndexType> getChildEntitiesSet(unsigned d, IndexType entityIndex);
  IndexType getConstrainingEntityIndexOfLikeDimension(unsigned d, IndexType entityIndex);
  pair<IndexType, unsigned> getConstrainingEntity(unsigned d, IndexType entityIndex);
  IndexType getEntityIndex(unsigned d, const set<IndexType> &nodeSet);
  IndexType getEntityCount(unsigned d);
  
  pair<IndexType,unsigned> getEntityGeneralizedParent(unsigned d, IndexType entityIndex); // returns (parentEntityIndex, parentDimension)
  
  IndexType getEntityParent(unsigned d, IndexType entityIndex, unsigned parentOrdinal=0);
  IndexType getEntityParentForSide(unsigned d, IndexType entityIndex, IndexType parentSideEntityIndex);   // returns the entity index for the parent (which might be the entity itself) of entity (d,entityIndex) that is a subcell of side parentSideEntityIndex
  vector<IndexType> getEntityVertexIndices(unsigned d, IndexType entityIndex);
  CellTopoPtr getEntityTopology(unsigned d, IndexType entityIndex);
  IndexType getFaceEdgeIndex(unsigned faceIndex, unsigned edgeOrdinalInFace);
  
  unsigned getCellCountForSide(IndexType sideEntityIndex); // 1 or 2
  pair<IndexType, unsigned> getFirstCellForSide(IndexType sideEntityIndex);
  pair<IndexType, unsigned> getSecondCellForSide(IndexType sideEntityIndex);
  
  vector<IndexType> getCanonicalEntityNodesViaPeriodicBCs(unsigned d, const vector<IndexType> &myEntityNodes); // if there are periodic BCs for this entity, this converts the provided nodes to the ones listed in the canonical ordering (allows permutation determination) -- this method is meant to be called internally, and from Cell.
  
  set< pair<IndexType, unsigned> > getCellsContainingEntity(unsigned d, IndexType entityIndex);
  vector< IndexType > getSidesContainingEntity(unsigned d, IndexType entityIndex);
  
  RefinementBranch getSideConstraintRefinementBranch(IndexType sideEntityIndex); // Returns a RefinementBranch that goes from the constraining side to the side indicated.
  
  unsigned getSpaceDim();
  unsigned getSubEntityCount(unsigned int d, IndexType entityIndex, unsigned subEntityDim);
  IndexType getSubEntityIndex(unsigned d, IndexType entityIndex, unsigned subEntityDim, unsigned subEntityOrdinal);
  unsigned getSubEntityPermutation(unsigned d, IndexType entityIndex, unsigned subEntityDim, unsigned subEntityOrdinal);
  bool getVertexIndex(const vector<double> &vertex, IndexType &vertexIndex, double tol=1e-14);
  const vector<double>& getVertex(IndexType vertexIndex);
  FieldContainer<double> physicalCellNodesForCell(unsigned cellIndex, bool includeCellDimension = false);
  void refineCell(IndexType cellIndex, RefinementPatternPtr refPattern);
  IndexType cellCount();
  IndexType activeCellCount();
  
  //  pair<IndexType,IndexType> leastActiveCellIndexContainingEntityConstrainedByConstrainingEntity(unsigned d, unsigned constrainingEntityIndex);
  
  void setGlobalDofAssignment(GlobalDofAssignment* gda); // for cubature degree lookups
  
  const set<IndexType> &getActiveCellIndices();
  set< pair<IndexType, unsigned> > getActiveBoundaryCells(); // (cellIndex, sideOrdinal)
  vector<double> getCellCentroid(IndexType cellIndex);
  
  const set<IndexType> &getRootCellIndices();
  
  pair<IndexType,IndexType> owningCellIndexForConstrainingEntity(unsigned d, unsigned constrainingEntityIndex);
  
  // 2D only:
  vector< ParametricCurvePtr > parametricEdgesForCell(IndexType cellID, bool neglectCurves);
  void setEdgeToCurveMap(const map< pair<IndexType, IndexType>, ParametricCurvePtr > &edgeToCurveMap, MeshPtr mesh);
  
  void printConstraintReport(unsigned d);
  void printEntityVertices(unsigned d, IndexType entityIndex);
  
  void printAllEntities();
  
  // not sure this should ultimately be exposed -- using it now to allow correctly timed call to updateCells()
  // (will be transitioning from having MeshTransformationFunction talk to Mesh to having it talk to MeshTopology)
  Teuchos::RCP<MeshTransformationFunction> transformationFunction();
  
  
  // ! This method exposed for the sake of tests
  vector< pair<IndexType,unsigned> > getConstrainingSideAncestry(IndexType sideEntityIndex);   // pair: first is the sideEntityIndex of the ancestor; second is the refinementIndex of the refinement to get from parent to child (see _parentEntities and _childEntities)
  
};

typedef Teuchos::RCP<MeshTopology> MeshTopologyPtr;

#endif