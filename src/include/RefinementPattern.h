// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#ifndef DPG_REFINEMENT_PATTERN
#define DPG_REFINEMENT_PATTERN

#include "TypeDefs.h"

// Intrepid includes
#include "Camellia_FieldContainer.hpp"

#include "CellTopology.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

// #include "MeshTopology.h"

#include <map>
#include <set>

using namespace std;

namespace Camellia {
  // class RefinementPattern;
  // typedef Teuchos::RCP<RefinementPattern> RefinementPatternPtr;
  typedef vector< pair<RefinementPattern*, unsigned> > RefinementBranch; //unsigned: the child ordinal; order is from coarse to fine
  typedef vector< pair<RefinementPattern*, vector<unsigned> > > RefinementPatternRecipe;

  // ! RefinementPatternKey: first is the CellTopologyKey for the CellTopology being refined; second is an enumerating identifier.
  // ! Convention is that (2^D)-1 is a null refinement, and 0 a regular isotropic refinement when there is such a thing (tetrahedra are an exception).
  typedef pair<CellTopologyKey,int> RefinementPatternKey;

  // ! The following define the refinement pattern enumeration for any topologies where we can think of dividing along the x/y/z/t axes.
  // ! Pass in 1 for each axis that is being refined.  (Refining in all directions gives 0.)
#define REFINEMENT_PATTERN_ORDINAL_1D(xRef) ((1-xRef) << 0)
#define REFINEMENT_PATTERN_ORDINAL_2D(xRef,yRef) ((1-xRef) << 1) | ((1-yRef) << 0)
#define REFINEMENT_PATTERN_ORDINAL_3D(xRef,yRef,zRef) ((1-xRef) << 2) | ((1-yRef) << 1) | ((1-zRef) << 0)
#define REFINEMENT_PATTERN_ORDINAL_4D(xRef,yRef,zRef,tRef) ((1-xRef) << 3) | ((1-yRef) << 2) | ((1-zRef) << 1) | ((1-tRef) << 0)

  // ! Time extruded means that we don't refine in the temporal direction:
#define REFINEMENT_PATTERN_ORDINAL_TIME_EXTRUSION(spacePatternOrdinal) (spacePatternOrdinal << 1) | ((1-0) << 0)
  
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

    int _refinementOrdinal; // the ordinal of this pattern in the enumeration of known patterns for the topology
    CellTopoPtr _cellTopoPtr;
    Camellia::FieldContainer<double> _nodes;
    vector< vector< unsigned > > _subCells;
    Camellia::FieldContainer<double> _vertices;

    vector< CellTopoPtr > _childTopos;
    vector<unsigned> _interiorChildOrdinals; // e.g. 1 for regular triangle refinement pattern

    vector< vector< Teuchos::RCP<RefinementPattern> > > _patternForSubcell;
    vector< RefinementPatternRecipe > _relatedRecipes;
    vector< Teuchos::RCP<RefinementPattern> > _sideRefinementPatterns;
    vector< vector< pair< unsigned, unsigned> > > _childrenForSides; // parentSide --> vector< pair(childIndex, childSideIndex) >

    vector< vector<unsigned> > _sideRefinementChildIndices; // maps from index of child in side refinement to the index in volume refinement pattern

    // map goes from (childIndex,childSideIndex) --> parentSide (essentially the inverse of the _childrenForSides)
    map< pair<unsigned,unsigned>, unsigned> _parentSideForChildSide;
    bool colinear(const vector<double> &v1_outside, const vector<double> &v2_outside, const vector<double> &v3_maybe_inside);

    double distance(const vector<double> &v1, const vector<double> &v2);

    static map<RefinementPatternKey, Teuchos::RCP<RefinementPattern> > _refPatterns;
    
    static map< pair< RefinementBranch, unsigned> , Camellia::FieldContainer<double> > _descendantNodesRelativeToPermutedReferenceCell;
    
    void determineChildSubcellInfoInSubcellRefinement(unsigned &childSubcellDimension, unsigned &childSubcellOrdinal,
                                                      unsigned &childSubcellPermutation, unsigned &subcellRefChild,
                                                      unsigned subcdim, unsigned subcord, unsigned childOrdinal,
                                                      bool preferSubcellsBelongingToVolumeChild);
    
    static RefinementPatternPtr refPatternExtrudedInTime(RefinementPatternPtr spaceRefPattern);
  public:
    RefinementPattern(CellTopoPtr cellTopoPtr, Camellia::FieldContainer<double> refinedNodes,
                      vector< Teuchos::RCP<RefinementPattern> > sideRefinementPatterns, int refinementOrdinal);
  //  RefinementPattern(Teuchos::RCP< shards::CellTopology > shardsTopoPtr, Camellia::FieldContainer<double> refinedNodes,
  //                    vector< Teuchos::RCP<RefinementPattern> > sideRefinementPatterns);

    //! dimensionRefinementFlags should be 0 for dimensions that are not refined, 1 for dimensions that are refined
    //! NOTE THAT THIS IS STILL UNDER DEVELOPMENT, AND SHOULD NOT YET BE RELIED ON IN PRODUCTION CODE.
    static RefinementPatternPtr anisotropicRefinementPattern(CellTopoPtr cellTopo, std::vector<int> dimensionRefinementFlags);
    
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
  static RefinementPatternPtr timeExtrudedRegularRefinementPattern(CellTopoPtr cellTopo);
  static Teuchos::RCP<RefinementPattern> xAnisotropicRefinementPatternQuad(); // vertical cut
  static Teuchos::RCP<RefinementPattern> yAnisotropicRefinementPatternQuad(); // horizontal cut
  
  static Teuchos::RCP<RefinementPattern> xAnisotropicRefinementPatternExtrudedLine();
  static Teuchos::RCP<RefinementPattern> yAnisotropicRefinementPatternExtrudedLine();
    
  static RefinementPatternPtr xAnisotropicRefinementPatternQuadTimeExtruded();
  static RefinementPatternPtr yAnisotropicRefinementPatternQuadTimeExtruded();


    // ! returns true if the child does not share any side with parent (as in triangular refinements)
    bool childIsInterior(unsigned childOrdinal) const;
  unsigned childOrdinalForPoint(const std::vector<double> &pointParentCoords); // returns -1 if no hit

  static void initializeAnisotropicRelationships();

  const Camellia::FieldContainer<double> & verticesOnReferenceCell();
  Camellia::FieldContainer<double> verticesForRefinement(Camellia::FieldContainer<double> &cellNodes);

  vector< vector<GlobalIndexType> > children(const map<unsigned, GlobalIndexType> &localToGlobalVertexIndex); // localToGlobalVertexIndex key: index in vertices; value: index in _vertices
  // children returns a vector of global vertex indices for each child

  vector< vector< pair< unsigned, unsigned > > > & childrenForSides(); // outer vector: indexed by parent's sides; inner vector: (child index in children, index of child's side shared with parent)
  map< unsigned, unsigned > parentSideLookupForChild(unsigned childIndex); // inverse of childrenForSides

  CellTopoPtr childTopology(unsigned childIndex);
  CellTopoPtr parentTopology();
  MeshTopologyPtr refinementMeshTopology();

  unsigned numChildren();
  const Camellia::FieldContainer<double> & refinedNodes();

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

  void mapPointsToChildRefCoordinates(const Camellia::FieldContainer<double> &pointsParentCoords, unsigned childOrdinal, Camellia::FieldContainer<double> &pointsChildCoords);

  vector< RefinementPatternRecipe > &relatedRecipes(); // e.g. the anisotropic + isotropic refinements of the quad.  This should be an exhaustive list, and should be in order of increasing fineness--i.e. the isotropic refinement should come at the end of the list.  Unless the list is empty, the current refinement pattern is required to be part of the list.  (A refinement pattern is related to itself.)  It's the job of initializeAnisotropicRelationships to initialize this list for the default refinement patterns that support it.
  void setRelatedRecipes(vector< RefinementPatternRecipe > &recipes);

  static unsigned ancestralSubcellOrdinal(RefinementBranch &refBranch, unsigned subcdim, unsigned descendantSubcord);

  static unsigned descendantSubcellOrdinal(RefinementBranch &refBranch, unsigned subcdim, unsigned ancestralSubcord);

  static Camellia::FieldContainer<double> descendantNodesRelativeToAncestorReferenceCell(RefinementBranch refinementBranch, unsigned ancestorReferenceCellPermutation=0,
                                                                                         bool cacheResults = true);

  static Camellia::FieldContainer<double> descendantNodes(RefinementBranch refinementBranch, const Camellia::FieldContainer<double> &ancestorNodes);

  static CellTopoPtr descendantTopology(RefinementBranch &refinements);

  static map<unsigned, set<unsigned> > getInternalSubcellOrdinals(RefinementBranch &refinements);

    RefinementPatternKey getKey() const;
    
    static RefinementPatternPtr refinementPattern(RefinementPatternKey key);
    
  static RefinementBranch sideRefinementBranch(RefinementBranch &volumeRefinementBranch, unsigned sideIndex);

  // ! returns a refinement branch rooted in subcell ordinal of subcord and dimension of subcdim in the root of volumeRefinementBranch.
  static RefinementBranch subcellRefinementBranch(RefinementBranch &volumeRefinementBranch, unsigned subcdim, unsigned subcord,
      bool tolerateSubcellsWithoutDescendants=false);

  // ! returns a generalized refinement branch that has as its leaf the subcell (subcdim, subcord) in the leaf of the volumeRefinementBranch
  static GeneralizedRefinementBranch generalizedRefinementBranchForLeafSubcell(RefinementBranch &volumeRefinementBranch, unsigned subcdim, unsigned subcord);

  static void mapRefCellPointsToAncestor(GeneralizedRefinementBranch &generalizedRefBranch, const Camellia::FieldContainer<double> &leafRefCellPoints,
                                         Camellia::FieldContainer<double> &rootRefCellPoints);

  static void mapRefCellPointsToAncestor(RefinementBranch &refinementBranch, const Camellia::FieldContainer<double> &leafRefCellPoints,
                                         Camellia::FieldContainer<double> &rootRefCellPoints);
    
  static pair<unsigned, unsigned> mapSubcellFromDescendantToAncestor(RefinementBranch &refBranch,
                                                                     unsigned subcdim, unsigned childSubcord);
  
  // clears static maps that are lazily populated in response to calls to RefinementPattern
  static void clearStaticMaps();
};

typedef Teuchos::RCP<RefinementPattern> RefinementPatternPtr;
}


#endif
