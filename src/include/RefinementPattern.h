#ifndef DPG_REFINEMENT_PATTERN
#define DPG_REFINEMENT_PATTERN

// Intrepid includes
#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

using namespace std;
using namespace Intrepid;

class RefinementPattern;
typedef Teuchos::RCP<RefinementPattern> RefinementPatternPtr;
typedef vector< pair<RefinementPattern*, unsigned> > RefinementBranch; //unsigned: the child ordinal
typedef vector< pair<RefinementPattern*, vector<unsigned> > > RefinementPatternRecipe;

class MeshTopology;
typedef Teuchos::RCP<MeshTopology> MeshTopologyPtr;

#include "MeshTopology.h"

class RefinementPattern {
  MeshTopologyPtr _refinementTopology; // ultimately, this may be able to supplant a number of structures here...
  
  Teuchos::RCP< shards::CellTopology > _cellTopoPtr;
  FieldContainer<double> _nodes;
  vector< vector< unsigned > > _subCells;
  FieldContainer<double> _vertices;
  
  vector< Teuchos::RCP< shards::CellTopology > > _childTopos;
  
  vector< vector< Teuchos::RCP<RefinementPattern> > > _patternForSubcell;
  vector< RefinementPatternRecipe > _relatedRecipes;
  vector< Teuchos::RCP<RefinementPattern> > _sideRefinementPatterns;
  vector< vector< pair< unsigned, unsigned> > > _childrenForSides; // parentSide --> vector< pair(childIndex, childSideIndex) >
  
  vector< vector<unsigned> > _sideRefinementChildIndices; // maps from index of child in side refinement to the index in volume refinement pattern
  
  // map goes from (childIndex,childSideIndex) --> parentSide (essentially the inverse of the _childrenForSides)
  map< pair<unsigned,unsigned>, unsigned> _parentSideForChildSide;
  bool colinear(const vector<double> &v1_outside, const vector<double> &v2_outside, const vector<double> &v3_maybe_inside);
  
  double distance(const vector<double> &v1, const vector<double> &v2);
  
public:
  RefinementPattern(Teuchos::RCP< shards::CellTopology > cellTopoPtr, FieldContainer<double> refinedNodes,
                    vector< Teuchos::RCP<RefinementPattern> > sideRefinementPatterns);

  static Teuchos::RCP<RefinementPattern> noRefinementPattern(Teuchos::RCP< shards::CellTopology > cellTopoPtr);
  static Teuchos::RCP<RefinementPattern> noRefinementPatternLine();
  static Teuchos::RCP<RefinementPattern> noRefinementPatternTriangle();
  static Teuchos::RCP<RefinementPattern> noRefinementPatternQuad();
  static Teuchos::RCP<RefinementPattern> regularRefinementPatternLine();
  static Teuchos::RCP<RefinementPattern> regularRefinementPatternTriangle();
  static Teuchos::RCP<RefinementPattern> regularRefinementPatternQuad();
  static Teuchos::RCP<RefinementPattern> regularRefinementPatternHexahedron();
  static Teuchos::RCP<RefinementPattern> regularRefinementPattern(unsigned cellTopoKey);
  static Teuchos::RCP<RefinementPattern> xAnisotropicRefinementPatternQuad(); // vertical cut
  static Teuchos::RCP<RefinementPattern> yAnisotropicRefinementPatternQuad(); // horizontal cut
  
  static void initializeAnisotropicRelationships();

  const FieldContainer<double> & verticesOnReferenceCell();
  FieldContainer<double> verticesForRefinement(FieldContainer<double> &cellNodes);
  
  vector< vector<GlobalIndexType> > children(const map<unsigned, GlobalIndexType> &localToGlobalVertexIndex); // localToGlobalVertexIndex key: index in vertices; value: index in _vertices
  // children returns a vector of global vertex indices for each child
  
  vector< vector< pair< unsigned, unsigned > > > & childrenForSides(); // outer vector: indexed by parent's sides; inner vector: (child index in children, index of child's side shared with parent)
  map< unsigned, unsigned > parentSideLookupForChild(unsigned childIndex); // inverse of childrenForSides

  Teuchos::RCP< shards::CellTopology > childTopology(unsigned childIndex);
  Teuchos::RCP< shards::CellTopology > parentTopology();
  MeshTopologyPtr refinementMeshTopology();
  
  unsigned numChildren();
  const FieldContainer<double> & refinedNodes();
  
  const vector< Teuchos::RCP<RefinementPattern> > &sideRefinementPatterns();
  Teuchos::RCP<RefinementPattern> patternForSubcell(unsigned subcdim, unsigned subcord);
  
  unsigned mapSideChildIndex(unsigned sideIndex, unsigned sideRefinementChildIndex); // map from index of child in side refinement to the index in volume refinement pattern
  
  static unsigned mapSideOrdinalFromLeafToAncestor(unsigned descendantSideOrdinal, RefinementBranch &refinements); // given a side ordinal in the leaf node of a branch, returns the corresponding side ordinal in the earliest ancestor in the branch.
  
  vector< RefinementPatternRecipe > &relatedRecipes(); // e.g. the anisotropic + isotropic refinements of the quad.  This should be an exhaustive list, and should be in order of increasing fineness--i.e. the isotropic refinement should come at the end of the list.  Unless the list is empty, the current refinement pattern is required to be part of the list.  (A refinement pattern is related to itself.)  It's the job of initializeAnisotropicRelationships to initialize this list for the default refinement patterns that support it.
  void setRelatedRecipes(vector< RefinementPatternRecipe > &recipes);
  
  static FieldContainer<double> descendantNodesRelativeToAncestorReferenceCell(RefinementBranch refinementBranch);
  
  static FieldContainer<double> descendantNodes(RefinementBranch refinementBranch, const FieldContainer<double> &ancestorNodes);
  
  static map<unsigned, set<unsigned> > getInternalSubcellOrdinals(RefinementBranch &refinements);
  
  static RefinementBranch sideRefinementBranch(RefinementBranch &volumeRefinementBranch, unsigned sideIndex);
};

typedef Teuchos::RCP<RefinementPattern> RefinementPatternPtr;

#endif
