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

#include "Basis.h"

using namespace Intrepid;
using namespace std;

class BasisReconciliation {
  bool _cacheResults;
  
  typedef pair< Camellia::Basis<>*, int > SideBasisRestriction;
  // cached values:
  map< pair< Camellia::Basis<>*, Camellia::Basis<>*>, FieldContainer<double> > _simpleReconciliationWeights; // simple: no sides involved
  map< pair< pair< SideBasisRestriction, SideBasisRestriction >, unsigned >, FieldContainer<double> > _sideReconciliationWeights;
public:
  BasisReconciliation(bool cacheResults = true) { _cacheResults = cacheResults; }

  const FieldContainer<double> &constrainedWeights(BasisPtr largerBasis, BasisPtr smallerBasis); // requires these to be defined on the same topology
  const FieldContainer<double> &constrainedWeights(BasisPtr largerBasis, BasisPtr smallerBasis, int largerBasisSideIndex, int smallerBasisSideIndex, unsigned vertexNodePermutation); // requires the sides to have the same topology
  
  // static workhorse methods:
  static FieldContainer<double> computeConstrainedWeights(BasisPtr largerBasis, BasisPtr smallerBasis); // requires these to be defined on the same topology
  static FieldContainer<double> computeConstrainedWeights(BasisPtr largerBasis, BasisPtr smallerBasis, int largerBasisSideIndex, int smallerBasisSideIndex, unsigned vertexNodePermutation); // requires the sides to have the same topology
  // vertexNodePermutation: how to permute side as seen by largerBasis to produce side seen by smallerBasis.  Specifically, if iota_1 maps largerBasis vertices \hat{v}_i^1 to v_i in physical space, and iota_2 does the same for vertices of smallerBasis's topology, then the permutation is the one corresponding to iota_2^(-1) \ocirc iota_1.
  // vertexNodePermutation is an index into a structure defined by CellTopologyTraits.  See CellTopology::getNodePermutation() and CellTopology::getNodePermutationInverse().
};

/* a few ideas come up here:
 
 1. Where should information about what continuities are enforced (e.g. vertices, edges, faces) reside?  Does it belong to the basis?  At a certain level that makes a good deal of sense, although it would break our current approach of using one lower order for L^2 functions, but otherwise using the H^1 basis.  Basis *does* offer the IntrepidExtendedTypes::EFunctionSpaceExtended functionSpace() method, which could be used for this purpose...  In fact, I believe we do get this right for our L^2 functions.
 2. Where should information about what pullbacks to use reside?  It seems to me this should also belong to the basis.  (And if we wanted to use functionSpace() to make this decision, this is what our L^2 basis implementation would break.)
 3. In particular, the answer to #1 affects our interface in BasisReconciliation.  The question is whether the side variants of computeConstrainedWeights should get subcell bases for vertices, edges, etc. or just the d-1 dimensional side.
 
 */


#endif
