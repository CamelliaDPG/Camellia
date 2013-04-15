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

class RefinementPattern {
  Teuchos::RCP< shards::CellTopology > _cellTopoPtr;
  FieldContainer<double> _nodes;
  vector< vector< int > > _subCells;
  FieldContainer<double> _vertices;
  vector< vector< pair< int, int> > > _childrenForSides; // parentSide --> vector< pair(childIndex, childSideIndex) >
  
  // map goes from (childIndex,childSideIndex) --> parentSide (essentially the inverse of the above)
  map< pair<int,int>, int> _parentSideForChildSide;
  bool colinear(const vector<double> &v1_outside, const vector<double> &v2_outside, const vector<double> &v3_maybe_inside);
  
  double distance(const vector<double> &v1, const vector<double> &v2);
public:
  RefinementPattern(Teuchos::RCP< shards::CellTopology > cellTopoPtr, FieldContainer<double> refinedNodes);
  
  static Teuchos::RCP<RefinementPattern> noRefinementPatternTriangle();
  static Teuchos::RCP<RefinementPattern> noRefinementPatternQuad();
  static Teuchos::RCP<RefinementPattern> regularRefinementPatternTriangle();
  static Teuchos::RCP<RefinementPattern> regularRefinementPatternQuad();
  static Teuchos::RCP<RefinementPattern> xAnisotropicRefinementPatternQuad();
  static Teuchos::RCP<RefinementPattern> yAnisotropicRefinementPatternQuad();

  const FieldContainer<double> & verticesOnReferenceCell();
  FieldContainer<double> verticesForRefinement(FieldContainer<double> &cellNodes);
  
  vector< vector<int> > children(map<int, int> &localToGlobalVertexIndex); // localToGlobalVertexIndex key: index in vertices; value: index in _vertices
  // children returns a vector of global vertex indices for each child
  
  vector< vector< pair< int, int> > > & childrenForSides(); // outer vector: indexed by parent's sides; inner vector: (child index in children, index of child's side shared with parent)
  map< int, int > parentSideLookupForChild(int childIndex); // inverse of childrenForSides
  
  int numChildren();
  const FieldContainer<double> & refinedNodes();
};

#endif
