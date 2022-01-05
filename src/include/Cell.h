// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  Cell.h
//  Camellia-debug
//
//  Created by Nate Roberts on 1/22/14.
//
//

#ifndef __Camellia_debug__Cell__
#define __Camellia_debug__Cell__

#include "TypeDefs.h"

#include <iostream>
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "RefinementPattern.h"

using namespace std;

namespace Camellia
{
typedef Teuchos::RCP<shards::CellTopology> CellTopoPtrLegacy;

// "cells" are geometric entities -- they do not define any kind of basis
// "elements" are cells endowed with a (local) functional discretization
class Cell
{
  GlobalIndexType _cellIndex;
  int _level;
  CellTopoPtr _cellTopo;
  vector<IndexType> _vertices;
  vector< vector< unsigned > > _subcellPermutations; // permutation to get from local ordering to the canonical one
  vector<vector<IndexType>> _entityIndices;

  MeshTopology* _meshTopo;

  // for parents:
  vector<GlobalIndexType> _childIndices;
  RefinementPatternPtr _refPattern;
  vector<CellPtr> _children;

  // for children:
  CellPtr _parent; // doesn't own memory (avoid circular reference issues)

  //neighbors:
  vector< pair<GlobalIndexType, IndexType> > _neighbors; // cellIndex, neighborSideIndex (which may not refer to the same side)
  /* rules for neighbors:
     - hanging node sides point to the constraining neighbor (which may not be active)
     - cells with broken neighbors point to their peer, the ancestor of the active neighbors
   */
  
  // ! subcells that lie on the mesh boundary as a vector of (scdim, scord) pairs
  // ! this is a somewhat expensive operation the first time; the result is cached for reuse
  std::vector< std::pair<unsigned, unsigned> > _boundarySubcells;
  bool _haveComputedBoundarySubcells;

  map<string, long long> approximateMemoryCosts(); // for each private variable
public:
  Cell(CellTopoPtr cellTopo, const vector<IndexType> &vertices, const vector< vector< unsigned > > &subcellPermutations,
       IndexType cellIndex, MeshTopology* meshTopo);

  Teuchos::RCP<Cell> ancestralCellForSubcell(unsigned subcdim, unsigned subcord, ConstMeshTopologyViewPtr meshTopoViewForCellValidity);

  unsigned ancestralPermutationForSubcell(unsigned subcdim, unsigned subcord, ConstMeshTopologyViewPtr meshTopoViewForCellValidity);
  //  unsigned ancestralPermutationForSideSubcell(unsigned sideOrdinal, unsigned subcdim, unsigned subcord);

  pair<unsigned, unsigned> ancestralSubcellOrdinalAndDimension(unsigned subcdim, unsigned subcord, ConstMeshTopologyViewPtr meshTopoViewForCellValidity); // (subcord, subcdim) into the cell returned by ancestralCellForSubcell

  long long approximateMemoryFootprint(); // in bytes

  vector<unsigned> boundarySides();

  GlobalIndexType cellIndex();
  virtual const vector< Teuchos::RCP< Cell > > &children();
  void setChildren(const vector<GlobalIndexType> &childIndices);
  virtual vector<IndexType> getChildIndices(ConstMeshTopologyViewPtr meshTopoViewForCellValidity);
  vector< pair<IndexType, unsigned> > childrenForSide(unsigned sideOrdinal);
  // !! returns the ordinal of the specified child in the parent's children container (which matches the order in RefinementPattern).  Returns -1 if the cell has no child with the specified cellIndex.
  int findChildOrdinal(IndexType childCellIndex);
  int numChildren();

  set<IndexType> getDescendants(ConstMeshTopologyViewPtr meshTopoViewForCellValidity, bool leafNodesOnly = true);
  vector< pair< IndexType, unsigned> > getDescendantsForSide(int sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity, bool leafNodesOnly = true);
  virtual IndexType entityIndex(unsigned subcdim, unsigned subcord);
  vector<IndexType> getEntityIndices(unsigned subcdim);

  vector<IndexType> getEntityVertexIndices(unsigned subcdim, unsigned subcord);

  Teuchos::RCP<Cell> getParent();
  void setParent(Teuchos::RCP<Cell> parent);

  virtual bool isBoundary(unsigned sideOrdinal);
  bool isDescendant(IndexType ancestralCellIndex); // true if the provided cell index is one of this cell's ancestors
  bool isInteriorChild();
  bool isParent(ConstMeshTopologyViewPtr meshTopoViewForCellValidity);

  int level() const;
  
  unsigned childOrdinal(IndexType childIndex);
  unsigned findSubcellOrdinal(unsigned subcdim, IndexType subcEntityIndex); // this is pretty brute force right now
  unsigned findSubcellOrdinalInSide(unsigned subcdim, IndexType subcEntityIndex, unsigned sideOrdinal); // this is pretty brute force right now

  MeshTopology* meshTopology();

  bool ownsSide(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity);

  RefinementPatternPtr refinementPattern();
  void setRefinementPattern(RefinementPatternPtr refPattern);

  //! Returns the RefinementBranch relative to the specified ancestral cell ID; if -1 (the default) is specified, returns refinement branch up to the root level.
  virtual RefinementBranch refinementBranch(IndexType ancestralCellID = -1);
  GlobalIndexType rootCellIndex();
  
  RefinementBranch refinementBranchForSide(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity);

  RefinementBranch refinementBranchForSubcell(unsigned subcdim, unsigned subcord, ConstMeshTopologyViewPtr meshTopoViewForCellValidity);

  //! Returns the number of sides of the cell; that is, subcells of dimension 1 lower than the cell.  In 1D, returns the number of vertices.
  /*!

   \return the number of sides of the cell.
   */
  unsigned getSideCount();

  //! permutation that maps from MeshTopology's canonical ordering to the ordering seen by this cell
  unsigned subcellPermutation(unsigned d, unsigned scord);

  //! permutation that maps from MeshTopology's canonical ordering to the ordering seen by this cell's side
  unsigned sideSubcellPermutation(unsigned sideOrdinal, unsigned sideSubcdim, unsigned sideSubcord);

  CellTopoPtr topology();

  virtual Teuchos::RCP<Cell> getNeighbor(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity);
  virtual pair<GlobalIndexType, unsigned> getNeighborInfo(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity); // (neighborCellIndex, neighborSideOrdinal)
  virtual void setNeighbor(unsigned sideOrdinal, GlobalIndexType neighborCellIndex, unsigned neighborSideOrdinal, bool allowSameCellIndex = false);
  std::vector< Teuchos::RCP<Cell> > getNeighbors(ConstMeshTopologyViewPtr meshTopoViewForCellValidity);
  
  std::set<GlobalIndexType> getActiveNeighborIndices(ConstMeshTopologyViewPtr meshTopoViewForCellValidity);
  std::set<GlobalIndexType> getActiveNeighborIndices(unsigned dimensionForNeighborRelation, ConstMeshTopologyViewPtr meshTopoViewForCellValidity);
  std::set<GlobalIndexType> getPeerNeighborIndices(unsigned dimensionForNeighborRelation, ConstMeshTopologyViewPtr meshTopoViewForCellValidity);
  virtual std::vector<IndexType> entitiesOnNeighborInterfaces(unsigned dimensionOfInterest, bool peersOnly,
                                                              ConstMeshTopologyViewPtr meshTopoViewForCellValidity);
  
  // ! returns subcells that lie on the mesh boundary as a vector of (scdim, scord) pairs
  // ! this is a somewhat expensive operation the first time; the result is cached for reuse
  std::vector< std::pair<unsigned, unsigned> > boundarySubcells();
  
  void printApproximateMemoryReport(); // in bytes
  void printNeighborInfo(ConstMeshTopologyViewPtr meshTopoViewForCellValidity);

  const vector< vector< unsigned > > &subcellPermutations();

  const vector<IndexType> &vertices();
  
  void setVertices(const vector<IndexType> &vertexIndices);
};
}

#endif /* defined(__Camellia_debug__Cell__) */
