//
//  BasisReconciliation.h
//  Camellia-debug
//
//  Created by Nate Roberts on 11/19/13.
//
//

#ifndef Camellia_debug_BasisReconciliation_h
#define Camellia_debug_BasisReconciliation_h

#include "Intrepid_FieldContainer.hpp"

#include "RefinementPattern.h"

#include "Basis.h"

using namespace Intrepid;
using namespace std;

struct SubBasisReconciliationWeights {
  FieldContainer<double> weights; // indices are (fine, coarse)
  set<int> fineOrdinals;
  set<int> coarseOrdinals;
};

class BasisReconciliation {
  bool _cacheResults;
  
  typedef pair< Camellia::Basis<>*, int > SideBasisRestriction;
  // cached values:
  typedef unsigned Permutation;
  typedef pair< Camellia::Basis<>*, Camellia::Basis<>*> BasisPair; // fineBasis first.
  map< pair<BasisPair, Permutation>, FieldContainer<double> > _simpleReconciliationWeights; // simple: no sides involved
  map< pair< pair< SideBasisRestriction, SideBasisRestriction >, Permutation >, SubBasisReconciliationWeights > _sideReconciliationWeights;
  
  // for the caching interface, may want to use vector< pair< RefinementPattern*, int childIndex > > to define the coarse element's neighbor refinement Branch.  Probably still need a vertexPermutation
  // as well to fully define the relationship.  Something like:
private:
  typedef pair< BasisPair, RefinementBranch > RefinedBasisPair; // fineBasis (the one on the refined element) is first in the BasisPair
  typedef pair< pair< SideBasisRestriction, SideBasisRestriction >, RefinementBranch > SideRefinedBasisPair;
  map< pair<RefinedBasisPair, Permutation>, FieldContainer<double> > _simpleReconcilationWeights_h;
  map< pair< SideRefinedBasisPair, Permutation> , SubBasisReconciliationWeights > _sideReconcilationWeights_h;
  
  static FieldContainer<double> filterBasisValues(const FieldContainer<double> &basisValues, set<int> &filter);
  
  static FieldContainer<double> permutedCubaturePoints(BasisCachePtr basisCache, Permutation cellTopoNodePermutation);
public:
  BasisReconciliation(bool cacheResults = true) { _cacheResults = cacheResults; }

  // p
  const FieldContainer<double> &constrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis, unsigned vertexNodePermutation); // requires these to be defined on the same topology
  const SubBasisReconciliationWeights &constrainedWeights(BasisPtr finerBasis, int finerBasisSideIndex, BasisPtr coarserBasis, int coarserBasisSideIndex, unsigned vertexNodePermutation); // requires the sides to have the same topology

  // h
  const FieldContainer<double> &constrainedWeights(BasisPtr finerBasis, RefinementBranch refinements, BasisPtr coarserBasis, unsigned vertexNodePermutation); // requires these to be defined on the same topology
  const SubBasisReconciliationWeights &constrainedWeights(BasisPtr finerBasis, int finerBasisSideIndex, RefinementBranch &volumeRefinements, BasisPtr coarserBasis, int coarserBasisSideIndex, unsigned vertexNodePermutation); // vertexPermutation is for the fine basis's ancestral orientation (how to permute side as seen by fine's ancestor to produce side as seen by coarse)...
  
  // static workhorse methods:
  
  /* Unbroken elements: */
  static FieldContainer<double> computeConstrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis, unsigned vertexNodePermutation); // requires these to be defined on the same topology
  static SubBasisReconciliationWeights computeConstrainedWeights(BasisPtr finerBasis, int finerBasisSideIndex, BasisPtr coarserBasis, int coarserBasisSideIndex, unsigned vertexNodePermutation); // requires the sides to have the same topology
  // vertexNodePermutation: how to permute side as seen by finerBasis to produce side seen by coarserBasis.  Specifically, if iota_1 maps finerBasis vertices \hat{v}_i^1 to v_i in physical space, and iota_2 does the same for vertices of coarserBasis's topology, then the permutation is the one corresponding to iota_2^(-1) \ocirc iota_1.
  // vertexNodePermutation is an index into a structure defined by CellTopologyTraits.  See CellTopology::getNodePermutation() and CellTopology::getNodePermutationInverse().
  
  /* Broken elements: */
  // matching the internal degrees of freedom:
  static FieldContainer<double> computeConstrainedWeights(BasisPtr finerBasis, RefinementBranch refinements, BasisPtr coarserBasis, unsigned vertexNodePermutation);
  static set<unsigned> internalDofIndicesForFinerBasis(BasisPtr finerBasis, RefinementBranch refinements); // which degrees of freedom in the finer basis have empty support on the boundary of the coarser basis's reference element? -- these are the ones for which the constrained weights are determined in computeConstrainedWeights.
  // matching along sides:
  static SubBasisReconciliationWeights computeConstrainedWeights(BasisPtr finerBasis, int finerBasisSideIndex, RefinementBranch &volumeRefinements,
                                                                 RefinementBranch &sideRefinements,
                                                                 BasisPtr coarserBasis, int coarserBasisSideIndex, unsigned vertexNodePermutation);
  // it's worth noting that these FieldContainer arguments are not especially susceptible to caching
  

  static FieldContainer<double> subBasisReconciliationWeightsForSubcell(SubBasisReconciliationWeights &subBasisWeights, unsigned subcdim,
                                                                        BasisPtr fineBasis, unsigned fineSubcord,
                                                                        BasisPtr coarseBasis, unsigned coarseSubcord,
                                                                        set<unsigned> &fineBasisDofOrdinals);
  
  static SubBasisReconciliationWeights composedSubBasisReconciliationWeights(SubBasisReconciliationWeights aWeights, SubBasisReconciliationWeights bWeights);
  
  static set<int> interiorDofOrdinalsForBasis(BasisPtr basis);
};

/* a few ideas come up here:
 
 1. Where should information about what continuities are enforced (e.g. vertices, edges, faces) reside?  Does it belong to the basis?  At a certain level that makes a good deal of sense, although it would break our current approach of using one lower order for L^2 functions, but otherwise using the H^1 basis.  Basis *does* offer the IntrepidExtendedTypes::EFunctionSpaceExtended functionSpace() method, which could be used for this purpose...  In fact, I believe we do get this right for our L^2 functions.
 2. Where should information about what pullbacks to use reside?  It seems to me this should also belong to the basis.  (And if we wanted to use functionSpace() to make this decision, this is what our L^2 basis implementation would break.)
 3. In particular, the answer to #1 affects our interface in BasisReconciliation.  The question is whether the side variants of computeConstrainedWeights should get subcell bases for vertices, edges, etc. or just the d-1 dimensional side.
 
 */


#endif
