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
  
  // TODO: simplify this: eliminate simple reconciliation weights, and the h/p distinction.  Everything can happen in terms of subcell reconciliation.  (Simple is just subcdim = domain dimension, subcord = 0.  The non-h variant is just an empty RefinementBranch.)
  
  typedef pair< Camellia::Basis<>*, pair<unsigned, unsigned> > SubcellBasisRestriction;  // second pair is (subcdim, subcord)
  typedef pair< Camellia::Basis<>*, int > SideBasisRestriction;
  // cached values:
  typedef unsigned Permutation;
  typedef pair< Camellia::Basis<>*, Camellia::Basis<>*> BasisPair; // fineBasis first.
  map< pair<BasisPair, Permutation>, FieldContainer<double> > _simpleReconciliationWeights; // simple: no sides involved
  map< pair< pair< SideBasisRestriction, SideBasisRestriction >, Permutation >, SubBasisReconciliationWeights > _sideReconciliationWeights;
private:
  typedef pair< BasisPair, RefinementBranch > RefinedBasisPair; // fineBasis (the one on the refined element) is first in the BasisPair
  typedef pair< pair< SideBasisRestriction, SideBasisRestriction >, RefinementBranch > SideRefinedBasisPair;
  typedef pair< pair< SubcellBasisRestriction, SubcellBasisRestriction >, RefinementBranch > SubcellRefinedBasisPair;
  map< pair<RefinedBasisPair, Permutation>, FieldContainer<double> > _simpleReconcilationWeights_h;
  map< pair< SideRefinedBasisPair, Permutation> , SubBasisReconciliationWeights > _sideReconcilationWeights_h;
  
  // this is the only map that actually needs to remain, after the code simplification described above...
  map< pair< SubcellRefinedBasisPair, Permutation> , SubBasisReconciliationWeights > _subcellReconcilationWeights;
  
  static FieldContainer<double> filterBasisValues(const FieldContainer<double> &basisValues, set<int> &filter);
  
  static FieldContainer<double> permutedCubaturePoints(BasisCachePtr basisCache, Permutation cellTopoNodePermutation);
  
  static SubBasisReconciliationWeights filterOutZeroRowsAndColumns(SubBasisReconciliationWeights &weights);
  static SubBasisReconciliationWeights filterToInclude(set<int> &rowOrdinals, set<int> &colOrdinals, SubBasisReconciliationWeights &weights);
public:
  BasisReconciliation(bool cacheResults = true) { _cacheResults = cacheResults; }

  // p
  const SubBasisReconciliationWeights &constrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis, unsigned vertexNodePermutation); // requires these to be defined on the same topology
  const SubBasisReconciliationWeights &constrainedWeights(BasisPtr finerBasis, int finerBasisSideIndex, BasisPtr coarserBasis, int coarserBasisSideIndex, unsigned vertexNodePermutation); // requires the sides to have the same topology

  // h
  const SubBasisReconciliationWeights &constrainedWeights(BasisPtr finerBasis, RefinementBranch refinements, BasisPtr coarserBasis, unsigned vertexNodePermutation); // requires these to be defined on the same topology
  const SubBasisReconciliationWeights &constrainedWeights(BasisPtr finerBasis, int finerBasisSideIndex, RefinementBranch &volumeRefinements, BasisPtr coarserBasis, int coarserBasisSideIndex, unsigned vertexNodePermutation); // vertexPermutation is for the fine basis's ancestral orientation (how to permute side as seen by fine's ancestor to produce side as seen by coarse)...
  
  // the new bottleneck method (the others can be reimplemented to call this one, or simply eliminated)
  const SubBasisReconciliationWeights &constrainedWeights(unsigned subcellDimension,
                                                          BasisPtr finerBasis, unsigned finerBasisSubcellOrdinal,
                                                          RefinementBranch &refinements,
                                                          BasisPtr coarserBasis, unsigned coarserBasisSubcellOrdinal,
                                                          unsigned vertexNodePermutation);  // vertexNodePermutation: how to permute the subcell vertices as seen by finerBasis to get the one seen by coarserBasis.
  
  // static workhorse methods:
  static SubBasisReconciliationWeights computeConstrainedWeights(unsigned subcellDimension,
                                                                 BasisPtr finerBasis, unsigned finerBasisSubcellOrdinal,
                                                                 RefinementBranch &refinements,
                                                                 BasisPtr coarserBasis, unsigned coarserBasisSubcellOrdinal,
                                                                 unsigned vertexNodePermutation);  // vertexNodePermutation: how to permute the subcell vertices as seen by finerBasis to get the one seen by coarserBasis.

  static SubBasisReconciliationWeights computeConstrainedWeights(unsigned fineSubcellDimension,
                                                                 BasisPtr finerBasis, unsigned finerBasisSubcellOrdinalInFineDomain,
                                                                 RefinementBranch &cellRefinementBranch, // i.e. ref. branch is in volume, even for skeleton domains
                                                                 unsigned fineDomainOrdinalInRefinementLeaf,
                                                                 unsigned coarseSubcellDimension,
                                                                 BasisPtr coarserBasis, unsigned coarserBasisSubcellOrdinalInCoarseDomain,
                                                                 unsigned coarseDomainOrdinalInRefinementRoot, // we use the coarserBasis's domain topology to determine the domain's space dimension
                                                                 unsigned coarseDomainPermutation);  // coarseDomainPermutation: how to permute the nodes of the refinement root seen by the fine basis to get the domain as seen by the coarse basis.  (This is analogous to the one in the other computeConstrainedWeights, though here we have the whole *domain's* permutation, where there we have the subcell permutation)

  
  static SubBasisReconciliationWeights weightsForCoarseSubcell(SubBasisReconciliationWeights &weights, BasisPtr constrainingBasis, unsigned subcdim, unsigned subcord, bool includeSubsubcells);
  
  static SubBasisReconciliationWeights composedSubBasisReconciliationWeights(SubBasisReconciliationWeights aWeights, SubBasisReconciliationWeights bWeights);
  
  static set<int> interiorDofOrdinalsForBasis(BasisPtr basis);
  
  static set<unsigned> internalDofOrdinalsForFinerBasis(BasisPtr finerBasis, RefinementBranch refinements); // which degrees of freedom in the finer basis have empty support on the boundary of the coarser basis's reference element? -- these are the ones for which the constrained weights are determined in computeConstrainedWeights.
  static set<unsigned> internalDofOrdinalsForFinerBasis(BasisPtr finerBasis, RefinementBranch refinements, unsigned subcdim, unsigned subcord);
  
  static unsigned minimumSubcellDimension(BasisPtr basis); // for continuity enforcement
};

#endif
