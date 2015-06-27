#ifndef DPG_REFINEMENT_PATTERN
#define DPG_REFINEMENT_PATTERN

#include "TypeDefs.h"

// Intrepid includes
#include "Intrepid_FieldContainer.hpp"

#include "CellTopology.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

// #include "MeshTopology.h"

using namespace std;

namespace Camellia {
  // class RefinementPattern;
  // typedef Teuchos::RCP<RefinementPattern> RefinementPatternPtr;
  typedef vector< pair<RefinementPattern*, unsigned> > RefinementBranch; //unsigned: the child ordinal; order is from coarse to fine
  typedef vector< pair<RefinementPattern*, vector<unsigned> > > RefinementPatternRecipe;

  struct RefinementBranchTier
  {
    CellTopoPtr previousTierTopo;            // topology of the previous tier
    unsigned rootDimension;                  // dimension of the root of this tier
    unsigned previousTierSubcellOrdinal;     // ordinal of the root of this tier in the leaf of the previous tier
    unsigned previousTierSubcellPermutation; // permutation of the root of this tier in the leaf of the previous tier (relative to the previousTierTopo)
    RefinementBranch refBranch;              // refinement branch for the tier
    unsigned leafSubcellDimension;           // leaf subcell dimension
    unsigned leafSubcellOrdinal;             // subcell ordinal in the refBranch's leaf
    unsigned leafSubcellPermutation;         // ordinal of the permutation that takes the leaf nodes as seen by topmost (volume) topology to their order in refBranch
  };
  
  typedef std::vector<RefinementBranchTier> GeneralizedRefinementBranch; // allows mapping child points that may fall inside a volume, e.g.
  
  class RefinementPattern {
    MeshTopologyPtr _refinementTopology;

    CellTopoPtr _cellTopoPtr;
    Intrepid::FieldContainer<double> _nodes;
    vector< vector< unsigned > > _subCells;
    Intrepid::FieldContainer<double> _vertices;

    vector< CellTopoPtr > _childTopos;

    vector< vector< Teuchos::RCP<RefinementPattern> > > _patternForSubcell;
    vector< RefinementPatternRecipe > _relatedRecipes;
    vector< Teuchos::RCP<RefinementPattern> > _sideRefinementPatterns;
    vector< vector< pair< unsigned, unsigned> > > _childrenForSides; // parentSide --> vector< pair(childIndex, childSideIndex) >

    vector< vector<unsigned> > _sideRefinementChildIndices; // maps from index of child in side refinement to the index in volume refinement pattern

    // map goes from (childIndex,childSideIndex) --> parentSide (essentially the inverse of the _childrenForSides)
    map< pair<unsigned,unsigned>, unsigned> _parentSideForChildSide;
    bool colinear(const vector<double> &v1_outside, const vector<double> &v2_outside, const vector<double> &v3_maybe_inside);

    double distance(const vector<double> &v1, const vector<double> &v2);

    static map< pair<unsigned,unsigned>, Teuchos::RCP<RefinementPattern> > _refPatternForKeyTensorialDegree;
    
    void determineChildSubcellInfoInSubcellRefinement(unsigned &childSubcellDimension, unsigned &childSubcellOrdinal,
                                                      unsigned &childSubcellPermutation, unsigned &subcellRefChild,
                                                      unsigned subcdim, unsigned subcord, unsigned childOrdinal,
                                                      bool preferSubcellsBelongingToVolumeChild);
    
    static RefinementPatternPtr refPatternExtrudedInTime(RefinementPatternPtr spaceRefPattern);
  public:
    RefinementPattern(CellTopoPtr cellTopoPtr, Intrepid::FieldContainer<double> refinedNodes,
                      vector< Teuchos::RCP<RefinementPattern> > sideRefinementPatterns);
  //  RefinementPattern(Teuchos::RCP< shards::CellTopology > shardsTopoPtr, Intrepid::FieldContainer<double> refinedNodes,
  //                    vector< Teuchos::RCP<RefinementPattern> > sideRefinementPatterns);

  static Teuchos::RCP<RefinementPattern> noRefinementPattern(CellTopoPtr cellTopoPtr);
  static Teuchos::RCP<RefinementPattern> noRefinementPattern(CellTopoPtrLegacy shardsTopoPtr);
  static Teuchos::RCP<RefinementPattern> noRefinementPatternLine();
  static Teuchos::RCP<RefinementPattern> noRefinementPatternTriangle();
  static Teuchos::RCP<RefinementPattern> noRefinementPatternQuad();

  //! A refinement pattern for the point (node) topology.  Provided mainly to allow the logic of tensor-product topologies to work for Node x Line topologies.
  static Teuchos::RCP<RefinementPattern> regularRefinementPatternPoint();
  //! Standard refinement pattern for the line topology; splits the line into two of equal size.
  static Teuchos::RCP<RefinementPattern> regularRefinementPatternLine();
  //! Standard refinement pattern for the triangle topology; splits the triangle into four of equal size (by bisecting edges).
  static Teuchos::RCP<RefinementPattern> regularRefinementPatternTriangle();
  //! Standard refinement pattern for the quadrilateral topology; splits the quad into four of equal size (by bisecting edges).
  static Teuchos::RCP<RefinementPattern> regularRefinementPatternQuad();
  //! Standard refinement pattern for the hexahedral topology; splits the quad into eight of equal size (by bisecting edges).
  static Teuchos::RCP<RefinementPattern> regularRefinementPatternHexahedron();
  static Teuchos::RCP<RefinementPattern> regularRefinementPattern(unsigned cellTopoKey);
  static Teuchos::RCP<RefinementPattern> regularRefinementPattern(CellTopoPtr cellTopo);
  static Teuchos::RCP<RefinementPattern> regularRefinementPattern(Camellia::CellTopologyKey cellTopoKey);
  static Teuchos::RCP<RefinementPattern> xAnisotropicRefinementPatternQuad(); // vertical cut
  static Teuchos::RCP<RefinementPattern> yAnisotropicRefinementPatternQuad(); // horizontal cut
    
  static RefinementPatternPtr xAnisotropicRefinementPatternQuadTimeExtruded();
  static RefinementPatternPtr yAnisotropicRefinementPatternQuadTimeExtruded();

  unsigned childOrdinalForPoint(const std::vector<double> &pointParentCoords); // returns -1 if no hit

  static void initializeAnisotropicRelationships();

  const Intrepid::FieldContainer<double> & verticesOnReferenceCell();
  Intrepid::FieldContainer<double> verticesForRefinement(Intrepid::FieldContainer<double> &cellNodes);

  vector< vector<GlobalIndexType> > children(const map<unsigned, GlobalIndexType> &localToGlobalVertexIndex); // localToGlobalVertexIndex key: index in vertices; value: index in _vertices
  // children returns a vector of global vertex indices for each child

  vector< vector< pair< unsigned, unsigned > > > & childrenForSides(); // outer vector: indexed by parent's sides; inner vector: (child index in children, index of child's side shared with parent)
  map< unsigned, unsigned > parentSideLookupForChild(unsigned childIndex); // inverse of childrenForSides

  CellTopoPtr childTopology(unsigned childIndex);
  CellTopoPtr parentTopology();
  MeshTopologyPtr refinementMeshTopology();

  unsigned numChildren();
  const Intrepid::FieldContainer<double> & refinedNodes();

  const vector< Teuchos::RCP<RefinementPattern> > &sideRefinementPatterns();
  Teuchos::RCP<RefinementPattern> patternForSubcell(unsigned subcdim, unsigned subcord);

  unsigned mapSideChildIndex(unsigned sideIndex, unsigned sideRefinementChildIndex); // map from index of child in side refinement to the index in volume refinement pattern

  pair<unsigned, unsigned> mapSubcellFromParentToChild(unsigned childOrdinal, unsigned subcdim, unsigned parentSubcord); // pair is (subcdim, subcord)
  pair<unsigned, unsigned> mapSubcellFromChildToParent(unsigned childOrdinal, unsigned subcdim, unsigned childSubcord);  // pair is (subcdim, subcord)

  unsigned mapSubcellOrdinalFromParentToChild(unsigned childOrdinal, unsigned subcdim, unsigned parentSubcord);
  unsigned mapSubcellOrdinalFromChildToParent(unsigned childOrdinal, unsigned subcdim, unsigned childSubcord);

  unsigned mapSubcellChildOrdinalToVolumeChildOrdinal(unsigned subcdim, unsigned subcord, unsigned subcellChildOrdinal);
  unsigned mapVolumeChildOrdinalToSubcellChildOrdinal(unsigned subcdim, unsigned subcord, unsigned volumeChildOrdinal);

  static unsigned mapSideOrdinalFromLeafToAncestor(unsigned descendantSideOrdinal, RefinementBranch &refinements); // given a side ordinal in the leaf node of a branch, returns the corresponding side ordinal in the earliest ancestor in the branch.

  void mapPointsToChildRefCoordinates(const Intrepid::FieldContainer<double> &pointsParentCoords, unsigned childOrdinal, Intrepid::FieldContainer<double> &pointsChildCoords);

  vector< RefinementPatternRecipe > &relatedRecipes(); // e.g. the anisotropic + isotropic refinements of the quad.  This should be an exhaustive list, and should be in order of increasing fineness--i.e. the isotropic refinement should come at the end of the list.  Unless the list is empty, the current refinement pattern is required to be part of the list.  (A refinement pattern is related to itself.)  It's the job of initializeAnisotropicRelationships to initialize this list for the default refinement patterns that support it.
  void setRelatedRecipes(vector< RefinementPatternRecipe > &recipes);

  static unsigned ancestralSubcellOrdinal(RefinementBranch &refBranch, unsigned subcdim, unsigned descendantSubcord);

  static unsigned descendantSubcellOrdinal(RefinementBranch &refBranch, unsigned subcdim, unsigned ancestralSubcord);

  static Intrepid::FieldContainer<double> descendantNodesRelativeToAncestorReferenceCell(RefinementBranch refinementBranch, unsigned ancestorReferenceCellPermutation=0);

  static Intrepid::FieldContainer<double> descendantNodes(RefinementBranch refinementBranch, const Intrepid::FieldContainer<double> &ancestorNodes);

  static CellTopoPtr descendantTopology(RefinementBranch &refinements);

  static map<unsigned, set<unsigned> > getInternalSubcellOrdinals(RefinementBranch &refinements);

  static RefinementBranch sideRefinementBranch(RefinementBranch &volumeRefinementBranch, unsigned sideIndex);

  // ! returns a refinement branch rooted in subcell ordinal of subcord and dimension of subcdim in the root of volumeRefinementBranch.
  static RefinementBranch subcellRefinementBranch(RefinementBranch &volumeRefinementBranch, unsigned subcdim, unsigned subcord,
      bool tolerateSubcellsWithoutDescendants=false);

  // ! returns a generalized refinement branch that has as its leaf the subcell (subcdim, subcord) in the leaf of the volumeRefinementBranch
  static GeneralizedRefinementBranch generalizedRefinementBranchForLeafSubcell(RefinementBranch &volumeRefinementBranch, unsigned subcdim, unsigned subcord);

  static void mapRefCellPointsToAncestor(GeneralizedRefinementBranch &generalizedRefBranch, const Intrepid::FieldContainer<double> &leafRefCellPoints,
                                         Intrepid::FieldContainer<double> &rootRefCellPoints);

  static void mapRefCellPointsToAncestor(RefinementBranch &refinementBranch, const Intrepid::FieldContainer<double> &leafRefCellPoints,
                                         Intrepid::FieldContainer<double> &rootRefCellPoints);
};

typedef Teuchos::RCP<RefinementPattern> RefinementPatternPtr;
}


#endif
