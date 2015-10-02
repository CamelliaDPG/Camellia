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
  unsigned _cellIndex;
  CellTopoPtr _cellTopo;
  vector< unsigned > _vertices;
  vector< vector< unsigned > > _subcellPermutations; // permutation to get from local ordering to the canonical one
  vector<vector<IndexType>> _entityIndices;

  MeshTopology* _meshTopo;

  // for parents:
  vector< Teuchos::RCP< Cell > > _children;
  RefinementPatternPtr _refPattern;

  // for children:
  Teuchos::RCP<Cell> _parent; // doesn't own memory (avoid circular reference issues)

  //neighbors:
  vector< pair<GlobalIndexType, unsigned> > _neighbors; // cellIndex, neighborSideIndex (which may not refer to the same side)
  /* rules for neighbors:
     - hanging node sides point to the constraining neighbor (which may not be active)
     - cells with broken neighbors point to their peer, the ancestor of the active neighbors
   */

  map<string, long long> approximateMemoryCosts(); // for each private variable
public:
  Cell(CellTopoPtr cellTopo, const vector<unsigned> &vertices, const vector< vector< unsigned > > &subcellPermutations,
       IndexType cellIndex, MeshTopology* meshTopo);

  Teuchos::RCP<Cell> ancestralCellForSubcell(unsigned subcdim, unsigned subcord, MeshTopologyViewPtr meshTopoViewForCellValidity);

  unsigned ancestralPermutationForSubcell(unsigned subcdim, unsigned subcord, MeshTopologyViewPtr meshTopoViewForCellValidity);
  //  unsigned ancestralPermutationForSideSubcell(unsigned sideOrdinal, unsigned subcdim, unsigned subcord);

  pair<unsigned, unsigned> ancestralSubcellOrdinalAndDimension(unsigned subcdim, unsigned subcord, MeshTopologyViewPtr meshTopoViewForCellValidity); // (subcord, subcdim) into the cell returned by ancestralCellForSubcell

  long long approximateMemoryFootprint(); // in bytes

  vector<unsigned> boundarySides();

  IndexType cellIndex();
  const vector< Teuchos::RCP< Cell > > &children();
  void setChildren(vector< Teuchos::RCP< Cell > > children);
  vector<IndexType> getChildIndices(MeshTopologyViewPtr meshTopoViewForCellValidity);
  vector< pair<IndexType, unsigned> > childrenForSide(unsigned sideOrdinal);
  // !! returns the ordinal of the specified child in the parent's children container (which matches the order in RefinementPattern).  Returns -1 if the cell has no child with the specified cellIndex.
  int findChildOrdinal(IndexType childCellIndex);
  int numChildren();

  set<IndexType> getDescendants(MeshTopologyViewPtr meshTopoViewForCellValidity, bool leafNodesOnly = true);
  vector< pair< IndexType, unsigned> > getDescendantsForSide(int sideOrdinal, MeshTopologyViewPtr meshTopoViewForCellValidity, bool leafNodesOnly = true);
  unsigned entityIndex(unsigned subcdim, unsigned subcord);
  vector<unsigned> getEntityIndices(unsigned subcdim);

  vector<unsigned> getEntityVertexIndices(unsigned subcdim, unsigned subcord);

  Teuchos::RCP<Cell> getParent();
  void setParent(Teuchos::RCP<Cell> parent);

  bool isBoundary(unsigned sideOrdinal);
  bool isParent(MeshTopologyViewPtr meshTopoViewForCellValidity);

  unsigned childOrdinal(IndexType childIndex);
  unsigned findSubcellOrdinal(unsigned subcdim, IndexType subcEntityIndex); // this is pretty brute force right now
  unsigned findSubcellOrdinalInSide(unsigned subcdim, IndexType subcEntityIndex, unsigned sideOrdinal); // this is pretty brute force right now

  MeshTopology* meshTopology();

  bool ownsSide(unsigned sideOrdinal, MeshTopologyViewPtr meshTopoViewForCellValidity);

  RefinementPatternPtr refinementPattern();
  void setRefinementPattern(RefinementPatternPtr refPattern);

  RefinementBranch refinementBranchForSide(unsigned sideOrdinal, MeshTopologyViewPtr meshTopoViewForCellValidity);

  RefinementBranch refinementBranchForSubcell(unsigned subcdim, unsigned subcord, MeshTopologyViewPtr meshTopoViewForCellValidity);

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

  Teuchos::RCP<Cell> getNeighbor(unsigned sideOrdinal, MeshTopologyViewPtr meshTopoViewForCellValidity);
  pair<GlobalIndexType, unsigned> getNeighborInfo(unsigned sideOrdinal, MeshTopologyViewPtr meshTopoViewForCellValidity); // (neighborCellIndex, neighborSideOrdinal)
  void setNeighbor(unsigned sideOrdinal, GlobalIndexType neighborCellIndex, unsigned neighborSideOrdinal);
  std::vector< Teuchos::RCP<Cell> > getNeighbors(MeshTopologyViewPtr meshTopoViewForCellValidity);

  void printApproximateMemoryReport(); // in bytes

  const vector< vector< unsigned > > &subcellPermutations();

  const vector< unsigned > &vertices();
};
}

#endif /* defined(__Camellia_debug__Cell__) */
