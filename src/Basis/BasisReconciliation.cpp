//
//  BasisReconciliation.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 11/19/13.
//
//

#include "BasisReconciliation.h"

#include "BasisCache.h"

#include "Intrepid_DefaultCubatureFactory.hpp"

#include "SerialDenseMatrixUtility.h"

#include "Intrepid_FunctionSpaceTools.hpp"

#include "CamelliaCellTools.h"

#include "SerialDenseWrapper.h"

#include "CamelliaDebugUtility.h" // includes print() methods

void sizeFCForBasisValues(FieldContainer<double> &fc, BasisPtr basis, int numPoints, bool includeCellDimension = false, int numBasisFieldsToInclude = -1) {
  // values should have shape: (F,P[,D,D,...]) where the # of D's = rank of the basis's range
  Teuchos::Array<int> dim;
  if (includeCellDimension) {
    dim.push_back(1);
  }
  if (numBasisFieldsToInclude == -1) {
    dim.push_back(basis->getCardinality()); // F
  } else {
    dim.push_back(numBasisFieldsToInclude);
  }
  dim.push_back(numPoints); // P
  for (int d=0; d<basis->rangeRank(); d++) {
    dim.push_back(basis->rangeDimension()); // D
  }
  fc.resize(dim);
}

void filterFCValues(FieldContainer<double> &filteredFC, const FieldContainer<double> &fc, set<int> ordinals, int basisCardinality) {
  // we use pointer arithmetic here, which doesn't allow Intrepid's safety checks, for two reasons:
  // 1. It's easier to manage in a way that's independent of the rank of the basis
  // 2. It's faster.
  int numEntriesPerBasisField = fc.size() / basisCardinality;
  int filteredFCIndex = 0;
  for (set<int>::iterator ordinalIt = ordinals.begin(); ordinalIt != ordinals.end(); ordinalIt++) {
    int ordinal = *ordinalIt;
    const double *fcEntry = &fc[ordinal * numEntriesPerBasisField];
    double *filteredEntry = &filteredFC[ filteredFCIndex * numEntriesPerBasisField ];
    for (int i=0; i<numEntriesPerBasisField; i++) {
      *filteredEntry = *fcEntry;
      filteredEntry++;
      fcEntry++;
    }
    
    filteredFCIndex++;
  }
}

unsigned vertexPermutation(shards::CellTopology &fineTopo, unsigned fineSideIndex, FieldContainer<double> &fineCellNodes,
                           shards::CellTopology &coarseTopo, unsigned coarseSideIndex, FieldContainer<double> &coarseCellNodes) {
  int d = fineTopo.getDimension();
  shards::CellTopology sideTopo = fineTopo.getBaseCellTopologyData(d-1, fineSideIndex);
  
  // a brute force search for a matching permutation going from fineTopo's view of the side to coarseTopo's
  int permutationCount = sideTopo.getNodePermutationCount();
  double tol = 1e-14;
  for (unsigned permutation=0; permutation<permutationCount; permutation++) {
    bool matches = true;
    for (unsigned sideNode=0; sideNode<sideTopo.getNodeCount(); sideNode++) {
      unsigned fineNode = fineTopo.getNodeMap(d-1, fineSideIndex, sideNode);
      unsigned permutedSideNode = sideTopo.getNodePermutation(permutation, sideNode);
      unsigned putativeCoarseNode = coarseTopo.getNodeMap(d-1, coarseSideIndex, permutedSideNode);
      for (int dim=0; dim<d; dim++) {
        if (abs(fineCellNodes(0,fineNode,dim) - coarseCellNodes(0,putativeCoarseNode,dim)) > tol) {
          // not a match
          matches = false;
          break;
        }
      }
      if (matches == false) break;
    }
    if (matches) {
      return permutation;
    }
  }
  cout << "Matching permutation not found.\n";
  cout << "fine side index: " << fineSideIndex << endl;
  cout << "fine nodes:\n" << fineCellNodes;
  cout << "coarse side index: " << coarseSideIndex << endl;
  cout << "coarse nodes:\n" << coarseCellNodes;
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "matching permutation not found");
  return -1; // just for compilers that would otherwise warn that we're missing a return value...
}

FieldContainer<double> BasisReconciliation::permutedCubaturePoints(BasisCachePtr basisCache, Permutation cellTopoNodePermutation) { // permutation is for side nodes if the basisCache is a side cache
  shards::CellTopology volumeTopo = basisCache->cellTopology();
  int spaceDim = volumeTopo.getDimension();
  shards::CellTopology cellTopo = basisCache->isSideCache() ? volumeTopo.getCellTopologyData(spaceDim-1, basisCache->getSideIndex()) : volumeTopo;
  int cubDegree = basisCache->  cubatureDegree(); // not really an argument that matters; we'll overwrite the cubature points in basisCacheForPermutation anyway
  BasisCachePtr basisCacheForPermutation = Teuchos::rcp( new BasisCache( cellTopo, cubDegree, false ) );
  FieldContainer<double> permutedCellTopoNodes(cellTopo.getNodeCount(), cellTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(permutedCellTopoNodes, cellTopo, cellTopoNodePermutation);
  basisCacheForPermutation->setRefCellPoints(basisCache->getRefCellPoints());
  const int oneCell = 1;
  permutedCellTopoNodes.resize(oneCell,permutedCellTopoNodes.dimension(0),permutedCellTopoNodes.dimension(1));
  basisCacheForPermutation->setPhysicalCellNodes(permutedCellTopoNodes, vector<GlobalIndexType>(), false);
  FieldContainer<double> permutedCubaturePoints = basisCacheForPermutation->getPhysicalCubaturePoints();
  // resize for reference space (no cellIndex dimension):
  permutedCubaturePoints.resize(permutedCubaturePoints.dimension(1), permutedCubaturePoints.dimension(2));

//  if (cellTopoNodePermutation != 0) {
//    cout << "For non-identity permutation, unpermuted cubature points:\n" << basisCache->getRefCellPoints();
//    cout << "Permuted points:\n" << permutedCubaturePoints;
//  }

  return permutedCubaturePoints;
}

SubBasisReconciliationWeights BasisReconciliation::composedSubBasisReconciliationWeights(SubBasisReconciliationWeights aWeights, SubBasisReconciliationWeights bWeights) {
  if (aWeights.coarseOrdinals.size() != bWeights.fineOrdinals.size()) {
    cout << "aWeights and bWeights are incompatible...\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "aWeights and bWeights are incompatible...");
  }
  SubBasisReconciliationWeights cWeights;
  cWeights.weights = FieldContainer<double>(aWeights.fineOrdinals.size(), bWeights.coarseOrdinals.size());
  if (cWeights.weights.size() != 0) {
    FieldContainer<double> aMatrix = aWeights.weights;
    FieldContainer<double> bMatrix = bWeights.weights;
    SerialDenseWrapper::multiply(cWeights.weights, aWeights.weights, bWeights.weights);
  }
  cWeights.fineOrdinals = aWeights.fineOrdinals;
  cWeights.coarseOrdinals = bWeights.coarseOrdinals;
  return cWeights;
}

//FieldContainer<double> BasisReconciliation::computeConstrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis, Permutation vertexPermutation) {
//  // we could define things in terms of Functions, and then use Projector class.  But this is simple enough that it's probably worth it to do it more manually.
//  // (also, I'm a bit concerned about the expense here, and the present implementation hopefully will be a bit lighter weight.)
//  
//  // DEBUGGING: do we ever use a non-identity permutation?
////  if (vertexPermutation != 0) {
//////    cout << "non-identity vertexPermutation.\n";
////  } else {
//////    cout << "identity vertexPermutation.\n";
////  }
//  
//  shards::CellTopology cellTopo = finerBasis->domainTopology();
//  TEUCHOS_TEST_FOR_EXCEPTION(cellTopo.getBaseKey() != coarserBasis->domainTopology().getBaseKey(), std::invalid_argument, "Bases must agree on domain topology.");
//  
//  int cubDegree = finerBasis->getDegree() * 2;
//  BasisCachePtr fineBasisCache = Teuchos::rcp( new BasisCache(cellTopo, cubDegree, false) );
//  BasisCachePtr coarseBasisCache = Teuchos::rcp( new BasisCache(cellTopo, cubDegree, false) );
//  
//  coarseBasisCache->setRefCellPoints(permutedCubaturePoints(fineBasisCache, vertexPermutation));
//  
//  FieldContainer<double> finerBasisValues = *fineBasisCache->getValues(finerBasis, OP_VALUE);
//  FieldContainer<double> coarserBasisValues = *coarseBasisCache->getValues(coarserBasis, OP_VALUE);
//  
//  set<int> filteredDofOrdinalsForFinerBasis = interiorDofOrdinalsForBasis(finerBasis);
//  int interiorCardinalityFine = filteredDofOrdinalsForFinerBasis.size();
//  
//  set<int> filteredDofOrdinalsForCoarserBasis = interiorDofOrdinalsForBasis(coarserBasis);
//  int interiorCardinalityCoarse = filteredDofOrdinalsForCoarserBasis.size();
//  
//  FieldContainer<double> constrainedWeights(interiorCardinalityFine,interiorCardinalityCoarse);
//  if (interiorCardinalityFine==0) { // empty constraint
//    cout << "WARNING: encountered empty constraint.\n";
//    return constrainedWeights;
//  }
//  if (interiorCardinalityCoarse==0) {
//    cout << "WARNING: encountered empty coarse basis.\n";
//    return constrainedWeights;
//  }
//  
//  FieldContainer<double> finerBasisValuesFiltered = filterBasisValues(finerBasisValues, filteredDofOrdinalsForFinerBasis);
//  FieldContainer<double> coarserBasisValuesFiltered = filterBasisValues(coarserBasisValues, filteredDofOrdinalsForCoarserBasis);
//  
//  FieldContainer<double> finerBasisValuesFilteredWeighted;
//  
//  // resize things with dummy cell dimension:
//  int numCubPoints = fineBasisCache->getRefCellPoints().dimension(0); // (P,D)
//  sizeFCForBasisValues(coarserBasisValuesFiltered, coarserBasis, numCubPoints, true, interiorCardinalityCoarse);
//  sizeFCForBasisValues(finerBasisValuesFiltered, finerBasis, numCubPoints, true, interiorCardinalityFine);
//  sizeFCForBasisValues(finerBasisValuesFilteredWeighted, finerBasis, numCubPoints, true, interiorCardinalityFine);
//  
//  FieldContainer<double> refCellNodes(cellTopo.getNodeCount(),cellTopo.getDimension());
//  CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);
//  refCellNodes.resize(1,refCellNodes.dimension(0),refCellNodes.dimension(1));
//  fineBasisCache->setPhysicalCellNodes(refCellNodes, vector<GlobalIndexType>(1,0), false);
//  FieldContainer<double> cubWeights = fineBasisCache->getWeightedMeasures();
//  FunctionSpaceTools::multiplyMeasure<double>(finerBasisValuesFilteredWeighted, cubWeights, finerBasisValuesFiltered);
//  
//  FieldContainer<double> lhsValues(1,interiorCardinalityFine,interiorCardinalityFine);
//  FieldContainer<double> rhsValues(1,interiorCardinalityFine,interiorCardinalityCoarse);
//  
//  FunctionSpaceTools::integrate<double>(lhsValues,finerBasisValuesFiltered,finerBasisValuesFilteredWeighted,COMP_CPP);
//  FunctionSpaceTools::integrate<double>(rhsValues,finerBasisValuesFilteredWeighted,coarserBasisValuesFiltered,COMP_CPP);
//  
//  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
//  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));
//  try {
//    SerialDenseMatrixUtility::solveSystemMultipleRHS(constrainedWeights, lhsValues, rhsValues);
//  } catch (...) {
//    cout << "BasisReconciliation: SerialDenseMatrixUtility::solveSystemMultipleRHS: failed with the following inputs:\n";
//    cout << "lhsValues:\n" << lhsValues;
//    cout << "rhsValues:\n" << rhsValues;
//  }
//  
//  return constrainedWeights;
//}
//
//SubBasisReconciliationWeights BasisReconciliation::computeConstrainedWeights(BasisPtr finerBasis, int finerBasisSideIndex,
//                                                                             BasisPtr coarserBasis, int coarserBasisSideIndex,
//                                                                             unsigned vertexNodePermutation) {
//  SubBasisReconciliationWeights weights;
//  
//  // use the functionSpace to determine what continuities should be enforced:
//  IntrepidExtendedTypes::EFunctionSpaceExtended fs = finerBasis->functionSpace();
//  TEUCHOS_TEST_FOR_EXCEPTION(fs != coarserBasis->functionSpace(), std::invalid_argument, "Bases must agree on functionSpace().");
//  
//  int d = finerBasis->domainTopology().getDimension();
//  int minSubcellDimension = minimumSubcellDimension(finerBasis);
//  int sideDimension = d-1;
//  
//  weights.fineOrdinals = finerBasis->dofOrdinalsForSubcell(sideDimension, finerBasisSideIndex, minSubcellDimension);
//  weights.coarseOrdinals = coarserBasis->dofOrdinalsForSubcell(sideDimension, coarserBasisSideIndex, minSubcellDimension);
//  weights.weights.resize(weights.fineOrdinals.size(), weights.coarseOrdinals.size());
//  
//  if (weights.fineOrdinals.size() == 0) {
//    if (weights.coarseOrdinals.size() != 0) {
//      cout << "ERROR in BasisReconciliation: empty fine basis (when restricted to the indicated side), but non-empty coarse basis.\n";
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "In BasisReconciliation, empty fine basis (when restricted to the indicated side), but non-empty coarse basis.");
//    } else {
//      // both empty: return empty weights container
//      return weights;
//    }
//  }
//
//  int cubDegree = finerBasis->getDegree() * 2; // on LHS, will integrate finerBasis against itself
//  shards::CellTopology coarseTopo = coarserBasis->domainTopology();
//  BasisCachePtr coarserBasisVolumeCache = Teuchos::rcp( new BasisCache(coarseTopo, cubDegree, false) ); // false: don't create all the side caches, since we just want one side
//  BasisCachePtr coarseSideBasisCache = BasisCache::sideBasisCache(coarserBasisVolumeCache, coarserBasisSideIndex); // this is a leaner way to construct a side cache, but we will need to inform about physicalCellNodes manually
//  
//  shards::CellTopology sideTopo = coarseTopo.getCellTopologyData(sideDimension, coarserBasisSideIndex);
//  
//  shards::CellTopology fineTopo = finerBasis->domainTopology();
//  BasisCachePtr finerBasisVolumeCache = Teuchos::rcp( new BasisCache(fineTopo, cubDegree, false) );
//  BasisCachePtr fineSideBasisCache = BasisCache::sideBasisCache(finerBasisVolumeCache, finerBasisSideIndex);
//  
//  // We compute on the fine basis reference cell, and on a permuted coarse basis reference cell
//  
//  // how do we get the nodes on the reference cell?  It appears Trilinos does not tell us, so we implement our own in CamelliaCellTools
//  FieldContainer<double> fineVolumeRefCellNodes(fineTopo.getNodeCount(), d);
//  CamelliaCellTools::refCellNodesForTopology(fineVolumeRefCellNodes, coarseTopo);
//  // resize for "physical" nodes:
//  unsigned oneCell = 1;
//  fineVolumeRefCellNodes.resize(oneCell,fineVolumeRefCellNodes.dimension(0), fineVolumeRefCellNodes.dimension(1));
//  fineSideBasisCache->setPhysicalCellNodes(fineVolumeRefCellNodes, vector<GlobalIndexType>(), false);
//
//  FieldContainer<double> permutedSideCubaturePoints;
//  if (d-1 > 0) {
//    permutedSideCubaturePoints = permutedCubaturePoints(fineSideBasisCache, vertexNodePermutation);
//    // create a ("volume") basis cache for the side topology--this allows us to determine the cubature points corresponding to the
//    // permuted nodes.
////    BasisCachePtr sideCacheForPermutation = Teuchos::rcp( new BasisCache( sideTopo, cubDegree, false ) );
////    FieldContainer<double> permutedSideTopoNodes(sideTopo.getNodeCount(), sideTopo.getDimension());
////    CamelliaCellTools::refCellNodesForTopology(permutedSideTopoNodes, sideTopo, vertexNodePermutation);
////    sideCacheForPermutation->setRefCellPoints(fineSideBasisCache->getRefCellPoints()); // these should be the same, but doesn't hurt to be sure
////    permutedSideTopoNodes.resize(oneCell,permutedSideTopoNodes.dimension(0),permutedSideTopoNodes.dimension(1));
////    sideCacheForPermutation->setPhysicalCellNodes(permutedSideTopoNodes, vector<GlobalIndexType>(), false);
////    permutedSideCubaturePoints = sideCacheForPermutation->getPhysicalCubaturePoints();
////    // resize for reference space (no cellIndex dimension):
////    permutedSideCubaturePoints.resize(permutedSideCubaturePoints.dimension(1), permutedSideCubaturePoints.dimension(2));
//  } else { // dim-0 topologies don't change under permutation...
//    permutedSideCubaturePoints = fineSideBasisCache->getRefCellPoints();
//  }
//
//  FieldContainer<double> coarseVolumeRefCellNodes(coarseTopo.getNodeCount(), d);
//  CamelliaCellTools::refCellNodesForTopology(coarseVolumeRefCellNodes, coarseTopo);
//  coarseVolumeRefCellNodes.resize(oneCell,coarseVolumeRefCellNodes.dimension(0), coarseVolumeRefCellNodes.dimension(1));
//  coarseSideBasisCache->setPhysicalCellNodes(coarseVolumeRefCellNodes, vector<GlobalIndexType>(), false);
//  coarseSideBasisCache->setRefCellPoints(permutedSideCubaturePoints); // this makes computing weighted values illegal (the cubature weights are no longer valid, so they're cleared automatically)
//  
//  Teuchos::RCP< const FieldContainer<double> > finerBasisValues = fineSideBasisCache->getTransformedValues(finerBasis, OP_VALUE, true);
//  Teuchos::RCP< const FieldContainer<double> > finerBasisValuesWeighted = fineSideBasisCache->getTransformedWeightedValues(finerBasis, OP_VALUE, true);
//  Teuchos::RCP< const FieldContainer<double> > coarserBasisValues = coarseSideBasisCache->getTransformedValues(coarserBasis, OP_VALUE, true);
//  
//  FieldContainer<double> fineBasisValuesFiltered, fineBasisValuesFilteredWeighted, coarserBasisValuesFiltered;
//  unsigned numPoints = fineSideBasisCache->getRefCellPoints().dimension(0);
//  sizeFCForBasisValues(fineBasisValuesFiltered, finerBasis, numPoints, true, weights.fineOrdinals.size());
//  sizeFCForBasisValues(fineBasisValuesFilteredWeighted, finerBasis, numPoints, true, weights.fineOrdinals.size());
//  sizeFCForBasisValues(coarserBasisValuesFiltered, coarserBasis, numPoints, true, weights.coarseOrdinals.size());
//
//  filterFCValues(fineBasisValuesFiltered, *(finerBasisValues.get()), weights.fineOrdinals, finerBasis->getCardinality());
//  filterFCValues(fineBasisValuesFilteredWeighted, *(finerBasisValuesWeighted.get()), weights.fineOrdinals, finerBasis->getCardinality());
//  filterFCValues(coarserBasisValuesFiltered, *(coarserBasisValues.get()), weights.coarseOrdinals, coarserBasis->getCardinality());
//  
//  FieldContainer<double> lhsValues(1,weights.fineOrdinals.size(),weights.fineOrdinals.size());
//  FieldContainer<double> rhsValues(1,weights.fineOrdinals.size(),weights.coarseOrdinals.size());
//  
//  FunctionSpaceTools::integrate<double>(lhsValues,fineBasisValuesFiltered,fineBasisValuesFilteredWeighted,COMP_CPP);
//  FunctionSpaceTools::integrate<double>(rhsValues,fineBasisValuesFilteredWeighted,coarserBasisValuesFiltered,COMP_CPP);
//  
//  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
//  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));
//  
//  weights.weights.resize(weights.fineOrdinals.size(), weights.coarseOrdinals.size());
//  
//  SerialDenseMatrixUtility::solveSystemMultipleRHS(weights.weights, lhsValues, rhsValues);
//  
//  return weights;
//}

//FieldContainer<double> BasisReconciliation::computeConstrainedWeights(BasisPtr finerBasis, RefinementBranch refinements, BasisPtr coarserBasis, Permutation vertexPermutation) {
//  
//  if (refinements.size() == 0) {
//    return computeConstrainedWeights(finerBasis, coarserBasis, vertexPermutation);
//  }
//  
//  // DEBUGGING: do we ever use a non-identity permutation?
//  if (vertexPermutation != 0) {
////    cout << "non-identity vertexPermutation.\n";
//  } else {
////    cout << "identity vertexPermutation.\n";
//  }
//  
//  // there's a LOT of code duplication between this and the p-oriented computeConstrainedWeights(finerBasis,coarserBasis)
//  // would be pretty easy to refactor to make them share the common code -- this one just needs an additional distinction between the cubature points for coarserBasis and those for finerBasis...
//  
//  // we could define things in terms of Functions, and then use Projector class.  But this is simple enough that it's probably worth it to do it more manually.
//  // (also, I'm a bit concerned about the expense here, and the present implementation hopefully will be a bit lighter weight.)
//  
//  shards::CellTopology cellTopo = finerBasis->domainTopology();
//  int spaceDim = cellTopo.getDimension();
//  
//  int cubDegree = finerBasis->getDegree() * 2;
//  
//  BasisCachePtr fineBasisCache = Teuchos::rcp( new BasisCache(cellTopo, cubDegree, false) );
//  BasisCachePtr coarseBasisCache = Teuchos::rcp( new BasisCache(cellTopo, cubDegree, false) );
//  
//  FieldContainer<double> refCellNodes(cellTopo.getNodeCount(),cellTopo.getDimension());
//  CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);
//  refCellNodes.resize(1,refCellNodes.dimension(0),refCellNodes.dimension(1));
//  fineBasisCache->setPhysicalCellNodes(refCellNodes, vector<GlobalIndexType>(1,0), false);
//  FieldContainer<double> cubWeights = fineBasisCache->getWeightedMeasures();
//
//  int numCubPoints = fineBasisCache->getRefCellPoints().dimension(0); // (P,D)
//  FieldContainer<double> fineCellNodesInCoarseRefCell = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(refinements);
//  fineCellNodesInCoarseRefCell.resize(1,fineCellNodesInCoarseRefCell.dimension(0),fineCellNodesInCoarseRefCell.dimension(1));
//  FieldContainer<double> cubPointsCoarseBasis(1,numCubPoints,spaceDim);
//  FieldContainer<double> permutedCubPointsFineBasis = permutedCubaturePoints(fineBasisCache, vertexPermutation);
//  CellTools<double>::mapToPhysicalFrame(cubPointsCoarseBasis,permutedCubPointsFineBasis,fineCellNodesInCoarseRefCell,cellTopo);
//  cubPointsCoarseBasis.resize(numCubPoints,spaceDim);
//  
//  coarseBasisCache->setRefCellPoints(cubPointsCoarseBasis);
//  
//  FieldContainer<double> finerBasisValuesFilteredWeighted;
//  
//  FieldContainer<double> finerBasisValues = *fineBasisCache->getValues(finerBasis, OP_VALUE);
//  FieldContainer<double> coarserBasisValues = *coarseBasisCache->getValues(coarserBasis, OP_VALUE);
//  
//  set<unsigned> filteredDofOrdinalsForFinerBasis = internalDofOrdinalsForFinerBasis(finerBasis, refinements);
//  set<int> filteredDofOrdinalsForFinerBasisInt(filteredDofOrdinalsForFinerBasis.begin(),filteredDofOrdinalsForFinerBasis.end());
//  FieldContainer<double> finerBasisValuesFiltered = filterBasisValues(finerBasisValues, filteredDofOrdinalsForFinerBasisInt);
//
//  // 3-24-14: getting rid of the filter on the coarse basis.  Pretty sure that having it was incorrect...
////  set<int> filteredDofOrdinalsForCoarserBasis = interiorDofOrdinalsForBasis(coarserBasis);
////  if (filteredDofOrdinalsForCoarserBasis.size() == 0) {
////    cout << "error: coarse basis dof ordinals size == 0.\n";
////    Camellia::print("filteredDofOrdinalsForFinerBasis", filteredDofOrdinalsForFinerBasis);
////    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarse basis dof ordinals size == 0");
////  }
////  
////  FieldContainer<double> coarserBasisValuesFiltered = filterBasisValues(coarserBasisValues, filteredDofOrdinalsForCoarserBasis);
//  
//  // resize things with dummy cell dimension:
//  bool includeCellDimension = true;
//  sizeFCForBasisValues(coarserBasisValues, coarserBasis, numCubPoints, includeCellDimension, coarserBasis->getCardinality());
//  sizeFCForBasisValues(finerBasisValuesFiltered, finerBasis, numCubPoints, includeCellDimension, filteredDofOrdinalsForFinerBasis.size());
//  sizeFCForBasisValues(finerBasisValuesFilteredWeighted, finerBasis, numCubPoints, includeCellDimension, filteredDofOrdinalsForFinerBasis.size());
//  cubWeights.resize(1,numCubPoints); // dummy cell dimension
//  FunctionSpaceTools::multiplyMeasure<double>(finerBasisValuesFilteredWeighted, cubWeights, finerBasisValuesFiltered);
//  
//  FieldContainer<double> constrainedWeights(filteredDofOrdinalsForFinerBasis.size(),coarserBasis->getCardinality());
//  
//  FieldContainer<double> lhsValues(1,filteredDofOrdinalsForFinerBasis.size(),filteredDofOrdinalsForFinerBasis.size());
//  FieldContainer<double> rhsValues(1,filteredDofOrdinalsForFinerBasis.size(),coarserBasis->getCardinality());
//  
//  FunctionSpaceTools::integrate<double>(lhsValues,finerBasisValuesFiltered,finerBasisValuesFilteredWeighted,COMP_CPP);
//  FunctionSpaceTools::integrate<double>(rhsValues,finerBasisValuesFilteredWeighted,coarserBasisValues,COMP_CPP);
//  
//  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
//  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));
//  
//  try {
//    SerialDenseMatrixUtility::solveSystemMultipleRHS(constrainedWeights, lhsValues, rhsValues);
//  } catch (...) {
//    cout << "BasisReconciliation: SerialDenseMatrixUtility::solveSystemMultipleRHS: failed with the following inputs:\n";
//    cout << "lhsValues:\n" << lhsValues;
//    cout << "rhsValues:\n" << rhsValues;
//  }
//  
//  return constrainedWeights;
//}
//
//// matching along sides:
//SubBasisReconciliationWeights BasisReconciliation::computeConstrainedWeights(BasisPtr finerBasis, int fineAncestralSideIndex, RefinementBranch &volumeRefinements,
//                                                                             RefinementBranch &sideRefinements, BasisPtr coarserBasis, int coarserBasisSideIndex,
//                                                                             unsigned vertexNodePermutation) {
//  
//  SubBasisReconciliationWeights weights;
//  
//  // use the functionSpace to determine what continuities should be enforced:
//  IntrepidExtendedTypes::EFunctionSpaceExtended fs = finerBasis->functionSpace();
//  TEUCHOS_TEST_FOR_EXCEPTION(fs != coarserBasis->functionSpace(), std::invalid_argument, "Bases must agree on functionSpace().");
//  
//  int spaceDim = finerBasis->domainTopology().getDimension();
//  int sideDimension = spaceDim-1;
//  int minSubcellDimension = minimumSubcellDimension(finerBasis);
//  
//  // figure out fineSideIndex
//  unsigned fineSideIndex = fineAncestralSideIndex;
//  if (sideDimension > 0) { // if sideDimension == 0, then "side" is a vertex, and the ancestral side index is exactly the side index
//    for (int refIndex=0; refIndex<volumeRefinements.size(); refIndex++) {
//      RefinementPattern* refPattern = volumeRefinements[refIndex].first;
//      unsigned childIndex = volumeRefinements[refIndex].second;
//      vector< pair<unsigned, unsigned> > childrenForSide = refPattern->childrenForSides()[fineSideIndex];
//      for (vector< pair<unsigned, unsigned> >::iterator entryIt=childrenForSide.begin(); entryIt != childrenForSide.end(); entryIt++) {
//        if (entryIt->first == childIndex) {
//          fineSideIndex = entryIt->second;
//        }
//      }
//    }
//  }
//  
//  weights.fineOrdinals = finerBasis->dofOrdinalsForSubcell(sideDimension, fineSideIndex, minSubcellDimension);
//  weights.coarseOrdinals = coarserBasis->dofOrdinalsForSubcell(sideDimension, coarserBasisSideIndex, minSubcellDimension);
//  
////  print("fineOrdinals", weights.fineOrdinals);
//  
//  int cubDegree = finerBasis->getDegree() * 2; // on LHS, will integrate finerBasis against itself
//  unsigned oneCell = 1;
//  
//  // determine cubature points as seen by the fine basis
//  shards::CellTopology fineTopo = finerBasis->domainTopology();
//  BasisCachePtr fineVolumeCache = Teuchos::rcp( new BasisCache(fineTopo, cubDegree, false) );
//  BasisCachePtr fineSideBasisCache = BasisCache::sideBasisCache(fineVolumeCache, fineSideIndex);
//  FieldContainer<double> fineVolumeRefNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(volumeRefinements);
////  FieldContainer<double> fineVolumeRefNodes(fineTopo.getNodeCount(),spaceDim);
////  CamelliaCellTools::refCellNodesForTopology(fineVolumeRefNodes, fineTopo);
//  fineVolumeRefNodes.resize(oneCell,fineVolumeRefNodes.dimension(0),fineVolumeRefNodes.dimension(1));
//  fineSideBasisCache->setPhysicalCellNodes(fineVolumeRefNodes, vector<GlobalIndexType>(), false);
////  FieldContainer<double> fineVolumeCubaturePoints = fineSideBasisCache->getPhysicalCubaturePoints(); // these are in volume coordinates
////  fineVolumeCubaturePoints.resize(fineVolumeCubaturePoints.dimension(1),fineVolumeCubaturePoints.dimension(2));
//  FieldContainer<double> fineSideCubaturePoints = fineSideBasisCache->getRefCellPoints();
//  
////  cout << "fineSideCubaturePoints:\n" << fineSideCubaturePoints;
//  
//  // now, we need to map those points into coarseSide/coarseVolume
//  // coarseSide
//  shards::CellTopology ancestralTopo = *volumeRefinements[0].first->parentTopology();
//  shards::CellTopology ancestralSideTopo = ancestralTopo.getCellTopologyData(sideDimension, fineAncestralSideIndex);
//  
//  FieldContainer<double> coarseSideNodes(ancestralSideTopo.getNodeCount(),sideDimension);
//  CamelliaCellTools::refCellNodesForTopology(coarseSideNodes, ancestralSideTopo, vertexNodePermutation);
//  
//  FieldContainer<double> fineSideNodesInCoarseSideTopology;
//  FieldContainer<double> coarseSideCubaturePoints;
//  shards::CellTopology fineSideTopo = fineTopo.getCellTopologyData(sideDimension, fineSideIndex);
//  BasisCachePtr sideBasisCacheAsVolume = Teuchos::rcp( new BasisCache(fineSideTopo, cubDegree, false) );
//
//  // coarseVolume
//  shards::CellTopology coarseTopo = coarserBasis->domainTopology();
//  FieldContainer<double> coarseVolumeRefNodes(coarseTopo.getNodeCount(),spaceDim);
//  CamelliaCellTools::refCellNodesForTopology(coarseVolumeRefNodes, coarseTopo);
//  coarseVolumeRefNodes.resize(oneCell,coarseVolumeRefNodes.dimension(0),coarseVolumeRefNodes.dimension(1));
//  BasisCachePtr coarseBasisVolumeCache = Teuchos::rcp( new BasisCache(coarseTopo, cubDegree, false) ); // false: don't create all the side caches, since we just want one side
//  BasisCachePtr coarseSideBasisCache = BasisCache::sideBasisCache(coarseBasisVolumeCache, coarserBasisSideIndex); // this is a leaner way to construct a side cache, but we will need to inform about physicalCellNodes manually
//  coarseSideBasisCache->setPhysicalCellNodes(coarseVolumeRefNodes, vector<GlobalIndexType>(), false);
//  
//  if (coarseSideNodes.size() > 0) { // can be empty if "side" is a vertex...
//    fineSideNodesInCoarseSideTopology = RefinementPattern::descendantNodes(sideRefinements, coarseSideNodes);
//    sideBasisCacheAsVolume->setRefCellPoints(fineSideCubaturePoints); // should be the same, but to guard against changes in BasisCache, set these.
//    fineSideNodesInCoarseSideTopology.resize(oneCell, fineSideNodesInCoarseSideTopology.dimension(0), fineSideNodesInCoarseSideTopology.dimension(1));
//    sideBasisCacheAsVolume->setPhysicalCellNodes(fineSideNodesInCoarseSideTopology, vector<GlobalIndexType>(), false);
//    coarseSideCubaturePoints = sideBasisCacheAsVolume->getPhysicalCubaturePoints();
//    coarseSideCubaturePoints.resize(coarseSideCubaturePoints.dimension(1), coarseSideCubaturePoints.dimension(2));
//    coarseSideBasisCache->setRefCellPoints(coarseSideCubaturePoints);
//  } else {
//    // nothing to do, I think: the ref cell points will be degenerate...
//  }
////  FieldContainer<double> coarseVolumeCubaturePoints = coarseSideBasisCache->getPhysicalCubaturePoints();
////  coarseVolumeCubaturePoints.resize(coarseVolumeCubaturePoints.dimension(1),coarseVolumeCubaturePoints.dimension(2));
//  
////  cout << "coarseSideBasisCache->getPhysicalCubaturePoints():\n" << coarseSideBasisCache->getPhysicalCubaturePoints();
//  
//  Teuchos::RCP< const FieldContainer<double> > finerBasisValues = fineSideBasisCache->getTransformedValues(finerBasis, OP_VALUE, true);
//  Teuchos::RCP< const FieldContainer<double> > finerBasisValuesWeighted = fineSideBasisCache->getTransformedWeightedValues(finerBasis, OP_VALUE, true);
//  Teuchos::RCP< const FieldContainer<double> > coarserBasisValues = coarseSideBasisCache->getTransformedValues(coarserBasis, OP_VALUE, true);
//
//  FieldContainer<double> fineBasisValuesFiltered, fineBasisValuesFilteredWeighted, coarserBasisValuesFiltered;
//  unsigned numPoints = fineSideBasisCache->getRefCellPoints().dimension(0);
//  sizeFCForBasisValues(fineBasisValuesFiltered, finerBasis, numPoints, true, weights.fineOrdinals.size());
//  sizeFCForBasisValues(fineBasisValuesFilteredWeighted, finerBasis, numPoints, true, weights.fineOrdinals.size());
//  sizeFCForBasisValues(coarserBasisValuesFiltered, coarserBasis, numPoints, true, weights.coarseOrdinals.size());
//  
//  filterFCValues(fineBasisValuesFiltered, *(finerBasisValues.get()), weights.fineOrdinals, finerBasis->getCardinality());
//  filterFCValues(fineBasisValuesFilteredWeighted, *(finerBasisValuesWeighted.get()), weights.fineOrdinals, finerBasis->getCardinality());
//  filterFCValues(coarserBasisValuesFiltered, *(coarserBasisValues.get()), weights.coarseOrdinals, coarserBasis->getCardinality());
//  
//  FieldContainer<double> lhsValues(1,weights.fineOrdinals.size(),weights.fineOrdinals.size());
//  FieldContainer<double> rhsValues(1,weights.fineOrdinals.size(),weights.coarseOrdinals.size());
//  
//  FunctionSpaceTools::integrate<double>(lhsValues,fineBasisValuesFiltered,fineBasisValuesFilteredWeighted,COMP_CPP);
//  FunctionSpaceTools::integrate<double>(rhsValues,fineBasisValuesFilteredWeighted,coarserBasisValuesFiltered,COMP_CPP);
//  
//  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
//  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));
//  
//  weights.weights.resize(weights.fineOrdinals.size(), weights.coarseOrdinals.size());
//  
//  SerialDenseMatrixUtility::solveSystemMultipleRHS(weights.weights, lhsValues, rhsValues);
//  
//  return weights;
//}

SubBasisReconciliationWeights BasisReconciliation::computeConstrainedWeights(unsigned int subcellDimension,
                                                                             BasisPtr finerBasis, unsigned int finerBasisSubcellOrdinal,
                                                                             RefinementBranch &refinements,
                                                                             BasisPtr coarserBasis, unsigned int coarserBasisSubcellOrdinal,
                                                                             unsigned int vertexNodePermutation) {
  SubBasisReconciliationWeights weights;
  
  // use the functionSpace to determine what continuities should be enforced:
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = finerBasis->functionSpace();
  TEUCHOS_TEST_FOR_EXCEPTION(fs != coarserBasis->functionSpace(), std::invalid_argument, "Bases must agree on functionSpace().");
  
  int spaceDim = finerBasis->domainTopology().getDimension();
  
  // figure out ancestralSubcellOrdinal
  unsigned ancestralSubcellOrdinal = RefinementPattern::ancestralSubcellOrdinal(refinements, subcellDimension, finerBasisSubcellOrdinal);
  
  //  set<unsigned> fineOrdinalsUnsigned = internalDofOrdinalsForFinerBasis(finerBasis, refinements, subcellDimension, finerBasisSubcellOrdinal);
  //  weights.fineOrdinals.insert(fineOrdinalsUnsigned.begin(),fineOrdinalsUnsigned.end());
  
  weights.fineOrdinals = finerBasis->dofOrdinalsForSubcell(subcellDimension, finerBasisSubcellOrdinal, 0);
  weights.coarseOrdinals = coarserBasis->dofOrdinalsForSubcell(subcellDimension,coarserBasisSubcellOrdinal,0);
  
  if (weights.fineOrdinals.size() == 0) {
    if (weights.coarseOrdinals.size() != 0) {
      cout << "fineOrdinals determined to be empty when coarseOrdinals is not!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fineOrdinals determined to be empty when coarseOrdinals is not!");
    }
    weights.weights.resize(0, 0);
    return weights;
  }
  
  int cubDegree = finerBasis->getDegree() * 2; // on LHS, will integrate finerBasis against itself (this is overkill; we could likely get away with less by reasoning about the cardinality of fineOrdinals)
  
  // determine cubature points as seen by the fine basis
  shards::CellTopology fineTopo = finerBasis->domainTopology();
  shards::CellTopology fineSubcellTopo = fineTopo.getCellTopologyData(subcellDimension, finerBasisSubcellOrdinal);
  
  shards::CellTopology coarseTopo = coarserBasis->domainTopology();
  
  FieldContainer<double> fineTopoRefNodes(fineTopo.getVertexCount(), fineTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(fineTopoRefNodes, fineTopo);
  FieldContainer<double> coarseTopoRefNodes(coarseTopo.getVertexCount(), coarseTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(coarseTopoRefNodes, coarseTopo);

  FieldContainer<double> coarseVolumeCubaturePoints;
  FieldContainer<double> fineVolumeCubaturePoints; // points at which the fine basis will be evaluated
  FieldContainer<double> cubatureWeights;
  FieldContainer<double> weightedMeasure;
  
  BasisCachePtr fineVolumeCache = Teuchos::rcp( new BasisCache(fineTopo, cubDegree, false) );
  BasisCachePtr coarseVolumeCache = Teuchos::rcp( new BasisCache(coarseTopo, cubDegree, false));
  
  if (subcellDimension > 0) {
    BasisCachePtr fineSubcellCache = Teuchos::rcp( new BasisCache(fineSubcellTopo, cubDegree, false) );
    int numPoints = fineSubcellCache->getRefCellPoints().dimension(0);
    fineVolumeCubaturePoints.resize(numPoints,spaceDim);
    FieldContainer<double> fineSubcellCubaturePoints = fineSubcellCache->getRefCellPoints();
    cubatureWeights = fineSubcellCache->getCubatureWeights();
    weightedMeasure = fineSubcellCache->getWeightedMeasures();
    if (subcellDimension == spaceDim) {
      fineVolumeCubaturePoints = fineSubcellCubaturePoints;
    } else {
      CamelliaCellTools::mapToReferenceSubcell(fineVolumeCubaturePoints, fineSubcellCubaturePoints, subcellDimension, finerBasisSubcellOrdinal, fineTopo);
    }
    
    RefinementBranch subcellRefinements = RefinementPattern::subcellRefinementBranch(refinements, subcellDimension, ancestralSubcellOrdinal);
    FieldContainer<double> fineSubcellRefNodes;
    if (subcellRefinements.size() > 0) {
      fineSubcellRefNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(subcellRefinements);
    } else {
      fineSubcellRefNodes.resize(fineSubcellTopo.getVertexCount(),fineSubcellTopo.getDimension());
      CamelliaCellTools::refCellNodesForTopology(fineSubcellRefNodes, fineSubcellTopo);
    }
    fineSubcellRefNodes.resize(1,fineSubcellRefNodes.dimension(0),fineSubcellRefNodes.dimension(1));
    fineSubcellCache->setPhysicalCellNodes(fineSubcellRefNodes, vector<GlobalIndexType>(), false);
    
    // now, fineSubcellCache's physicalCubaturePoints are exactly the ones in the ancestral subcell
    FieldContainer<double> ancestralSubcellCubaturePoints = fineSubcellCache->getPhysicalCubaturePoints();
    ancestralSubcellCubaturePoints.resize(ancestralSubcellCubaturePoints.dimension(1), ancestralSubcellCubaturePoints.dimension(2)); // get rid of cellIndex dimension
    
    // now, we need first to map the ancestralSubcellCubaturePoints to the subcell as seen by the coarse neigbor, then to map those points into the coarse volume topology
    
    shards::CellTopology ancestralSubcellTopo = (subcellRefinements.size() > 0) ? *subcellRefinements[0].first->parentTopology() : fineSubcellTopo;
    BasisCachePtr ancestralSubcellCache = Teuchos::rcp( new BasisCache(ancestralSubcellTopo, cubDegree, false) );
    ancestralSubcellCache->setRefCellPoints(ancestralSubcellCubaturePoints);
    
    FieldContainer<double> coarseSubcellNodes(ancestralSubcellTopo.getNodeCount(), subcellDimension);
    CamelliaCellTools::refCellNodesForTopology(coarseSubcellNodes, ancestralSubcellTopo, vertexNodePermutation);
    coarseSubcellNodes.resize(1,coarseSubcellNodes.dimension(0),coarseSubcellNodes.dimension(1));
    
    ancestralSubcellCache->setPhysicalCellNodes(coarseSubcellNodes, vector<GlobalIndexType>(), false);
    FieldContainer<double> coarseSubcellCubaturePoints = ancestralSubcellCache->getPhysicalCubaturePoints();
    coarseSubcellCubaturePoints.resize(coarseSubcellCubaturePoints.dimension(1),coarseSubcellCubaturePoints.dimension(2));

    coarseVolumeCubaturePoints.resize(numPoints,spaceDim); // points at which the coarse basis will be evaluated
    if (subcellDimension == spaceDim) {
      coarseVolumeCubaturePoints = coarseSubcellCubaturePoints;
    } else {
      CamelliaCellTools::mapToReferenceSubcell(coarseVolumeCubaturePoints, coarseSubcellCubaturePoints, subcellDimension, coarserBasisSubcellOrdinal, coarseTopo);
    }

  } else { // subcellDimension == 0 --> vertex
    fineVolumeCubaturePoints.resize(1,spaceDim);
    for (int d=0; d<spaceDim; d++) {
      fineVolumeCubaturePoints(0,d) = fineTopoRefNodes(finerBasisSubcellOrdinal,d);
    }
    cubatureWeights.resize(1);
    cubatureWeights(0) = 1.0;
    
    weightedMeasure.resize(1, 1);
    weightedMeasure(0,0) = 1.0;
    
    coarseVolumeCubaturePoints.resize(1,spaceDim);
    for (int d=0; d<spaceDim; d++) {
      coarseVolumeCubaturePoints(0,d) = coarseTopoRefNodes(coarserBasisSubcellOrdinal,d);
    }
  }
  fineVolumeCache->setRefCellPoints(fineVolumeCubaturePoints, cubatureWeights, weightedMeasure);

  if (refinements.size() > 0) {
    FieldContainer<double> fineVolumeNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(refinements);
    fineVolumeNodes.resize(1, fineVolumeNodes.dimension(0), fineVolumeNodes.dimension(1));
    fineVolumeCache->setPhysicalCellNodes(fineVolumeNodes, vector<GlobalIndexType>(), false);
  } else {
    fineTopoRefNodes.resize(1,fineTopoRefNodes.dimension(0),fineTopoRefNodes.dimension(1));
    fineVolumeCache->setPhysicalCellNodes(fineTopoRefNodes, vector<GlobalIndexType>(), false);
    fineTopoRefNodes.resize(fineTopoRefNodes.dimension(1),fineTopoRefNodes.dimension(2));
  }
  coarseVolumeCache->setRefCellPoints(coarseVolumeCubaturePoints);
  coarseTopoRefNodes.resize(1,coarseTopoRefNodes.dimension(0),coarseTopoRefNodes.dimension(1));
  coarseVolumeCache->setPhysicalCellNodes(coarseTopoRefNodes, vector<GlobalIndexType>(), false);
  coarseTopoRefNodes.resize(coarseTopoRefNodes.dimension(1),coarseTopoRefNodes.dimension(2));
  
  int numPoints = cubatureWeights.size();
  
  Teuchos::RCP< const FieldContainer<double> > finerBasisValues = fineVolumeCache->getTransformedValues(finerBasis, OP_VALUE, false);
  Teuchos::RCP< const FieldContainer<double> > finerBasisValuesWeighted = fineVolumeCache->getTransformedWeightedValues(finerBasis, OP_VALUE, false);
  Teuchos::RCP< const FieldContainer<double> > coarserBasisValues = coarseVolumeCache->getTransformedValues(coarserBasis, OP_VALUE, false);
  
  FieldContainer<double> fineBasisValuesFiltered, fineBasisValuesFilteredWeighted, coarserBasisValuesFiltered;
  sizeFCForBasisValues(fineBasisValuesFiltered, finerBasis, numPoints, true, weights.fineOrdinals.size());
  sizeFCForBasisValues(fineBasisValuesFilteredWeighted, finerBasis, numPoints, true, weights.fineOrdinals.size());
  sizeFCForBasisValues(coarserBasisValuesFiltered, coarserBasis, numPoints, true, weights.coarseOrdinals.size());
  
  filterFCValues(fineBasisValuesFiltered, *(finerBasisValues.get()), weights.fineOrdinals, finerBasis->getCardinality());
  filterFCValues(fineBasisValuesFilteredWeighted, *(finerBasisValuesWeighted.get()), weights.fineOrdinals, finerBasis->getCardinality());
  filterFCValues(coarserBasisValuesFiltered, *(coarserBasisValues.get()), weights.coarseOrdinals, coarserBasis->getCardinality());
  
  //  cout << "fineBasisValues:\n" << *finerBasisValues;
  //  cout << "fineBasisValuesFiltered:\n" << fineBasisValuesFiltered;
  
  FieldContainer<double> lhsValues(1,weights.fineOrdinals.size(),weights.fineOrdinals.size());
  FieldContainer<double> rhsValues(1,weights.fineOrdinals.size(),weights.coarseOrdinals.size());
  
  FunctionSpaceTools::integrate<double>(lhsValues,fineBasisValuesFiltered,fineBasisValuesFilteredWeighted,COMP_CPP);
  FunctionSpaceTools::integrate<double>(rhsValues,fineBasisValuesFilteredWeighted,coarserBasisValuesFiltered,COMP_CPP);
  
  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));
  
  weights.weights.resize(weights.fineOrdinals.size(), weights.coarseOrdinals.size());
  
  SerialDenseMatrixUtility::solveSystemMultipleRHS(weights.weights, lhsValues, rhsValues);
  
  return weights;
}

SubBasisReconciliationWeights BasisReconciliation::computeConstrainedWeights(unsigned fineSubcellDimension,
                                                                             BasisPtr finerBasis, unsigned finerBasisSubcellOrdinalInFineDomain,
                                                                             RefinementBranch &cellRefinementBranch, // i.e. ref. branch is in volume, even for skeleton domains
                                                                             unsigned fineDomainOrdinalInRefinementLeaf,
                                                                             unsigned coarseSubcellDimension,
                                                                             BasisPtr coarserBasis, unsigned coarserBasisSubcellOrdinalInCoarseDomain,
                                                                             unsigned coarseDomainOrdinalInRefinementRoot, // we use the coarserBasis's domain topology to determine the domain's space dimension
                                                                             unsigned coarseDomainPermutation) {
  SubBasisReconciliationWeights weights;
  
  // use the functionSpace to determine what continuities should be enforced:
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = finerBasis->functionSpace();
  TEUCHOS_TEST_FOR_EXCEPTION(fs != coarserBasis->functionSpace(), std::invalid_argument, "Bases must agree on functionSpace().");
  
  int domainDim = finerBasis->domainTopology().getDimension();
  if (domainDim != coarserBasis->domainTopology().getDimension()) {
    cout << "dimensions of finerBasis and coarserBasis domains do not match!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "dimensions of finerBasis and coarserBasis domains do not match!");
  }
  
  weights.fineOrdinals = finerBasis->dofOrdinalsForSubcell(fineSubcellDimension, finerBasisSubcellOrdinalInFineDomain, 0);
  weights.coarseOrdinals = coarserBasis->dofOrdinalsForSubcell(coarseSubcellDimension, coarserBasisSubcellOrdinalInCoarseDomain, 0);
  
  if (weights.fineOrdinals.size() == 0) {
    if (weights.coarseOrdinals.size() != 0) {
      cout << "fineOrdinals determined to be empty when coarseOrdinals is not!\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fineOrdinals determined to be empty when coarseOrdinals is not!");
    }
    weights.weights.resize(0, 0);
    return weights;
  }
  
  int cubDegree = finerBasis->getDegree() * 2; // on LHS, will integrate finerBasis against itself (this is overkill; we could likely get away with less by reasoning about the cardinality of fineOrdinals)
  
  // determine cubature points as seen by the fine basis
  shards::CellTopology fineTopo = finerBasis->domainTopology();
  shards::CellTopology fineSubcellTopo = fineTopo.getCellTopologyData(fineSubcellDimension, finerBasisSubcellOrdinalInFineDomain);
  
  shards::CellTopology coarseTopo = coarserBasis->domainTopology();
  
  FieldContainer<double> fineTopoRefNodes(fineTopo.getVertexCount(), fineTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(fineTopoRefNodes, fineTopo);
  FieldContainer<double> coarseTopoRefNodes(coarseTopo.getVertexCount(), coarseTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(coarseTopoRefNodes, coarseTopo);
  
  FieldContainer<double> coarseDomainPoints; // these are as seen from the coarse cell's neighbor -- we use the coarseDomainPermutation together with the physicalCellNodes on the coarse domain to convert...
  FieldContainer<double> fineDomainPoints; // points at which the fine basis will be evaluated
  FieldContainer<double> cubatureWeightsFineSubcell; // allows us to integrate over the fine subcell even when domain is higher-dimensioned
  FieldContainer<double> weightedMeasureFineSubcell;
  
//  BasisCachePtr fineDomainCache = Teuchos::rcp( new BasisCache(fineTopo, cubDegree, false) );
  BasisCachePtr coarseDomainCache = Teuchos::rcp( new BasisCache(coarseTopo, cubDegree, false));
  
  if (cellRefinementBranch.size() == 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellRefinementBranch must have at least one refinement!");
  }
  
  FieldContainer<double> leafCellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(cellRefinementBranch);
  shards::CellTopology leafCellTopo = *RefinementPattern::descendantTopology(cellRefinementBranch);
  
  RefinementBranch coarseDomainRefinements = RefinementPattern::subcellRefinementBranch(cellRefinementBranch, domainDim, coarseDomainOrdinalInRefinementRoot);
  unsigned coarseDomainOrdinalInLeafCell = RefinementPattern::descendantSubcellOrdinal(cellRefinementBranch, domainDim, coarseDomainOrdinalInRefinementRoot);

  // work out what the subcell ordinal of the fine subcell is in the leaf of coarseDomainRefinements...
  unsigned fineSubcellOrdinalInLeafCell = CamelliaCellTools::subcellOrdinalMap(leafCellTopo,
                                                                               domainDim, fineDomainOrdinalInRefinementLeaf,
                                                                               fineSubcellDimension, finerBasisSubcellOrdinalInFineDomain);
  unsigned finerBasisSubcellOrdinalInCoarseRefinementsLeaf = CamelliaCellTools::subcellReverseOrdinalMap(leafCellTopo,
                                                                                                         domainDim, coarseDomainOrdinalInLeafCell,
                                                                                                         fineSubcellDimension, fineSubcellOrdinalInLeafCell);
  
  shards::CellTopology leafCoarseDomainTopo = (coarseDomainRefinements.size() > 0) ? *RefinementPattern::descendantTopology(coarseDomainRefinements) : coarseTopo;
  FieldContainer<double> fineSubcellNodes; // nodes for the leaf of refinements
  
  BasisCachePtr fineCellCache = Teuchos::rcp( new BasisCache(leafCellTopo, cubDegree, true) ); // true: do create side cache
  leafCellNodes.resize(1,leafCellNodes.dimension(0),leafCellNodes.dimension(1));
  fineCellCache->setPhysicalCellNodes(leafCellNodes, vector<GlobalIndexType>(), true);
  BasisCachePtr fineDomainCache;
  if (leafCellTopo.getDimension() > domainDim) {
    if (leafCellTopo.getDimension() != domainDim + 1) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "basis domains that are not either on cells or cell sides are unsupported.");
    }
    fineDomainCache = fineCellCache->getSideBasisCache(fineDomainOrdinalInRefinementLeaf);
  } else {
    fineDomainCache = fineCellCache;
  }

  if (fineSubcellDimension > 0) {
    BasisCachePtr fineSubcellCache = Teuchos::rcp( new BasisCache(fineSubcellTopo, cubDegree, false) );
    int numPoints = fineSubcellCache->getRefCellPoints().dimension(0);
    fineDomainPoints.resize(numPoints,domainDim);
    FieldContainer<double> fineSubcellCubaturePoints = fineSubcellCache->getRefCellPoints();
    cubatureWeightsFineSubcell = fineSubcellCache->getCubatureWeights();
    weightedMeasureFineSubcell = fineSubcellCache->getWeightedMeasures();
    if (fineSubcellDimension == domainDim) {
      fineDomainPoints = fineSubcellCubaturePoints;
    } else {
      CamelliaCellTools::mapToReferenceSubcell(fineDomainPoints, fineSubcellCubaturePoints, fineSubcellDimension,
                                               finerBasisSubcellOrdinalInFineDomain, fineTopo);
    }

    coarseDomainPoints.resize(numPoints,domainDim);
    CamelliaCellTools::mapToReferenceSubcell(coarseDomainPoints, fineSubcellCubaturePoints, fineSubcellDimension,
                                             finerBasisSubcellOrdinalInCoarseRefinementsLeaf, leafCoarseDomainTopo);
  } else { // subcellDimension == 0 --> vertex
    fineDomainPoints.resize(1,domainDim);
    for (int d=0; d<domainDim; d++) {
      fineDomainPoints(0,d) = fineTopoRefNodes(finerBasisSubcellOrdinalInFineDomain,d);
    }
    cubatureWeightsFineSubcell.resize(1);
    cubatureWeightsFineSubcell(0) = 1.0;
    
    weightedMeasureFineSubcell.resize(1,1);
    weightedMeasureFineSubcell.initialize(1.0);
    
    coarseDomainPoints.resize(1,domainDim);
    FieldContainer<double> coarseReferenceLeafNodes(leafCoarseDomainTopo.getVertexCount(),domainDim);
    for (int d=0; d<domainDim; d++) {
      coarseDomainPoints(0,d) = coarseReferenceLeafNodes(finerBasisSubcellOrdinalInCoarseRefinementsLeaf,d);
    }
  }
  
  fineDomainCache->setRefCellPoints(fineDomainPoints, cubatureWeightsFineSubcell, weightedMeasureFineSubcell);

//  cout << "fineDomainPoints:\n" << fineDomainPoints;
//  cout << "coarseDomainPoints:\n" << coarseDomainPoints;
  
  coarseDomainCache->setRefCellPoints(coarseDomainPoints);
  FieldContainer<double> coarseTopoRefNodesPermuted(coarseTopo.getVertexCount(), coarseTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(coarseTopoRefNodesPermuted, coarseTopo, coarseDomainPermutation);
  coarseTopoRefNodesPermuted.resize(1,coarseTopoRefNodesPermuted.dimension(0),coarseTopoRefNodesPermuted.dimension(1));
  coarseDomainCache->setPhysicalCellNodes(coarseTopoRefNodesPermuted, vector<GlobalIndexType>(), false);
  
  int numPoints = cubatureWeightsFineSubcell.size();
  
  Teuchos::RCP< const FieldContainer<double> > finerBasisValues = fineDomainCache->getTransformedValues(finerBasis, OP_VALUE, false);
  Teuchos::RCP< const FieldContainer<double> > finerBasisValuesWeighted = fineDomainCache->getTransformedWeightedValues(finerBasis, OP_VALUE, false);
  Teuchos::RCP< const FieldContainer<double> > coarserBasisValues = coarseDomainCache->getTransformedValues(coarserBasis, OP_VALUE, false);
  
  FieldContainer<double> fineBasisValuesFiltered, fineBasisValuesFilteredWeighted, coarserBasisValuesFiltered;
  sizeFCForBasisValues(fineBasisValuesFiltered, finerBasis, numPoints, true, weights.fineOrdinals.size());
  sizeFCForBasisValues(fineBasisValuesFilteredWeighted, finerBasis, numPoints, true, weights.fineOrdinals.size());
  sizeFCForBasisValues(coarserBasisValuesFiltered, coarserBasis, numPoints, true, weights.coarseOrdinals.size());
  
  filterFCValues(fineBasisValuesFiltered, *(finerBasisValues.get()), weights.fineOrdinals, finerBasis->getCardinality());
  filterFCValues(fineBasisValuesFilteredWeighted, *(finerBasisValuesWeighted.get()), weights.fineOrdinals, finerBasis->getCardinality());
  filterFCValues(coarserBasisValuesFiltered, *(coarserBasisValues.get()), weights.coarseOrdinals, coarserBasis->getCardinality());
  
  //  cout << "fineBasisValues:\n" << *finerBasisValues;
  //  cout << "fineBasisValuesFiltered:\n" << fineBasisValuesFiltered;
  
  FieldContainer<double> lhsValues(1,weights.fineOrdinals.size(),weights.fineOrdinals.size());
  FieldContainer<double> rhsValues(1,weights.fineOrdinals.size(),weights.coarseOrdinals.size());
  
  FunctionSpaceTools::integrate<double>(lhsValues,fineBasisValuesFiltered,fineBasisValuesFilteredWeighted,COMP_CPP);
  FunctionSpaceTools::integrate<double>(rhsValues,fineBasisValuesFilteredWeighted,coarserBasisValuesFiltered,COMP_CPP);
  
  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));
  
  weights.weights.resize(weights.fineOrdinals.size(), weights.coarseOrdinals.size());
  
  SerialDenseMatrixUtility::solveSystemMultipleRHS(weights.weights, lhsValues, rhsValues);
  
  return weights;
}

const SubBasisReconciliationWeights& BasisReconciliation::constrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis, Permutation vertexNodePermutation) {
  unsigned domainDimension = finerBasis->domainTopology().getDimension();
  RefinementBranch noRefinements;
  
  return constrainedWeights(domainDimension, finerBasis, 0, noRefinements, coarserBasis, 0, vertexNodePermutation);
}

const SubBasisReconciliationWeights & BasisReconciliation::constrainedWeights(BasisPtr finerBasis, int finerBasisSideOrdinal, BasisPtr coarserBasis, int coarserBasisSideOrdinal,
                                                                              Permutation vertexNodePermutation) {
  unsigned domainDimension = finerBasis->domainTopology().getDimension();
  unsigned sideDimension = domainDimension - 1;
  RefinementBranch noRefinements;
  
  return constrainedWeights(sideDimension, finerBasis, finerBasisSideOrdinal, noRefinements, coarserBasis, coarserBasisSideOrdinal, vertexNodePermutation);
}

const SubBasisReconciliationWeights & BasisReconciliation::constrainedWeights(BasisPtr finerBasis, RefinementBranch refinements, BasisPtr coarserBasis, Permutation vertexNodePermutation) {
  unsigned domainDimension = finerBasis->domainTopology().getDimension();
  
  return constrainedWeights(domainDimension, finerBasis, 0, refinements, coarserBasis, 0, vertexNodePermutation);
}

const SubBasisReconciliationWeights & BasisReconciliation::constrainedWeights(BasisPtr finerBasis, int finerBasisSideOrdinal, RefinementBranch &volumeRefinements,
                                                                              BasisPtr coarserBasis, int coarserBasisSideOrdinal, unsigned vertexNodePermutation) { // vertexPermutation is for the fine basis's ancestral orientation (how to permute side as seen by fine's ancestor to produce side as seen by coarse)...
  
  unsigned domainDimension = finerBasis->domainTopology().getDimension();
  unsigned sideDimension = domainDimension - 1;
  
  return constrainedWeights(sideDimension, finerBasis, finerBasisSideOrdinal, volumeRefinements, coarserBasis, finerBasisSideOrdinal, vertexNodePermutation);
}

const SubBasisReconciliationWeights & BasisReconciliation::constrainedWeights(unsigned subcellDimension,
                                                                              BasisPtr finerBasis, unsigned finerBasisSubcellOrdinal,
                                                                              RefinementBranch &refinements,
                                                                              BasisPtr coarserBasis, unsigned coarserBasisSubcellOrdinal,
                                                                              unsigned vertexNodePermutation) {
  
  SubcellBasisRestriction fineBasisRestriction = make_pair(finerBasis.get(), make_pair(subcellDimension, finerBasisSubcellOrdinal) );
  SubcellBasisRestriction coarseBasisRestriction = make_pair(coarserBasis.get(), make_pair(subcellDimension, coarserBasisSubcellOrdinal) );
  SubcellRefinedBasisPair refinedBasisPair = make_pair(make_pair(fineBasisRestriction, coarseBasisRestriction), refinements);
  
  pair< SubcellRefinedBasisPair, Permutation> cacheKey = make_pair(refinedBasisPair, vertexNodePermutation);
  
  if (_subcellReconcilationWeights.find(cacheKey) == _subcellReconcilationWeights.end()) {
    _subcellReconcilationWeights[cacheKey] = computeConstrainedWeights(subcellDimension, finerBasis, finerBasisSubcellOrdinal, refinements,
                                                                       coarserBasis, coarserBasisSubcellOrdinal, vertexNodePermutation);
  }
  
  return _subcellReconcilationWeights[cacheKey];
}

FieldContainer<double> BasisReconciliation::filterBasisValues(const FieldContainer<double> &basisValues, set<int> &filter) {
  Teuchos::Array<int> dim;
  basisValues.dimensions(dim);
  int basisCardinality = dim[0];
  dim[0] = filter.size();
  FieldContainer<double> filteredValues(dim);
  
  if (filter.size() == 0) { // empty container
    return filteredValues;
  }
  
  // apply filter:
  double* filteredValue = &filteredValues[0];
  unsigned valuesPerDof = basisValues.size() / basisCardinality;
  for (set<int>::iterator filteredDofOrdinalIt = filter.begin();
       filteredDofOrdinalIt != filter.end(); filteredDofOrdinalIt++) {
    unsigned filteredDofOrdinal = *filteredDofOrdinalIt;
    for (int i=0; i<valuesPerDof; i++) {
      *filteredValue = basisValues[filteredDofOrdinal*valuesPerDof + i];
      filteredValue++;
    }
  }
  return filteredValues;
}

set<int> BasisReconciliation::interiorDofOrdinalsForBasis(BasisPtr basis) {
  // if L2, we include all dofs, not just the interior ones
  bool isL2 = (basis->functionSpace() == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) || (basis->functionSpace() == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL);
  int spaceDim = basis->domainTopology().getDimension();
  set<int> interiorDofOrdinals = isL2 ? basis->dofOrdinalsForSubcells(spaceDim, true) : basis->dofOrdinalsForInterior();
  if (interiorDofOrdinals.size() == 0) {
    cout << "Empty interiorDofOrdinals ";
    if (isL2)
      cout << "(L^2 basis).\n";
    else
      cout << "(non-L^2 basis).\n";
  }
  return interiorDofOrdinals;
}

set<unsigned> BasisReconciliation::internalDofOrdinalsForFinerBasis(BasisPtr finerBasis, RefinementBranch refinements) {
  // which degrees of freedom in the finer basis have empty support on the boundary of the coarser basis's reference element? -- these are the ones for which the constrained weights are determined in computeConstrainedWeights.
  set<unsigned> internalDofOrdinals;
  unsigned spaceDim = finerBasis->domainTopology().getDimension();
  
  if (refinements.size() == 0) {
    set<int> internalDofOrdinalsInt = BasisReconciliation::interiorDofOrdinalsForBasis(finerBasis);
    internalDofOrdinals.insert(internalDofOrdinalsInt.begin(),internalDofOrdinalsInt.end());
    return internalDofOrdinals;
  }
  
  bool isL2 = (finerBasis->functionSpace() == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) || (finerBasis->functionSpace() == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL);
  
  if (isL2) {
    // L2 basis, so everything is interior:
    set<int> dofOrdinalsInt = finerBasis->dofOrdinalsForSubcells(spaceDim,true);
    internalDofOrdinals.insert(dofOrdinalsInt.begin(),dofOrdinalsInt.end());
    return internalDofOrdinals;
  }
  
  map<unsigned, set<unsigned> > internalSubcellOrdinals = RefinementPattern::getInternalSubcellOrdinals(refinements); // might like to cache this result (there is a repeated, inefficient call inside this method)
  
  for (int scDim = 0; scDim < spaceDim; scDim++) {
    set<unsigned> internalOrdinals = internalSubcellOrdinals[scDim];
    
    for (set<unsigned>::iterator scOrdinalIt = internalOrdinals.begin(); scOrdinalIt != internalOrdinals.end(); scOrdinalIt++) {
      unsigned scOrdinal = *scOrdinalIt;
      set<int> scInternalDofs = finerBasis->dofOrdinalsForSubcell(scDim, scOrdinal, scDim);
      internalDofOrdinals.insert(scInternalDofs.begin(),scInternalDofs.end());
    }
  }
  set<int> cellInternalDofs = finerBasis->dofOrdinalsForSubcell(spaceDim, 0, spaceDim);
  internalDofOrdinals.insert(cellInternalDofs.begin(),cellInternalDofs.end());
  
  return internalDofOrdinals;
}

set<unsigned> BasisReconciliation::internalDofOrdinalsForFinerBasis(BasisPtr finerBasis, RefinementBranch refinements, unsigned subcdim, unsigned subcord) {
  // subcord is in the fine cell (as opposed to the ancestor)
  unsigned spaceDim = finerBasis->domainTopology().getDimension();
  if (subcdim == spaceDim) {
    if (subcord != 0) {
      cout << "ERROR: subcord out of range.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "subcord out of range");
    }
    return internalDofOrdinalsForFinerBasis(finerBasis, refinements);
  }
  
  // determine ancestral subcord:
  unsigned ancestralSubcord = RefinementPattern::ancestralSubcellOrdinal(refinements, subcdim, subcord);
//  unsigned ancestralSubcord = subcord;
//  for (int i=refinements.size()-1; i>=0; i--) {
//    RefinementPattern* refPattern = refinements[i].first;
//    unsigned childOrdinal = refinements[i].second;
//    ancestralSubcord = refPattern->mapSubcellOrdinalFromChildToParent(childOrdinal, subcdim, ancestralSubcord);
//  }
  
  RefinementBranch subcellRefinementBranch = RefinementPattern::subcellRefinementBranch(refinements, subcdim, ancestralSubcord);

  map< unsigned, set<unsigned> > internalSubSubcellOrdinals;
  if (subcellRefinementBranch.size() > 0) {
    internalSubSubcellOrdinals = RefinementPattern::getInternalSubcellOrdinals(subcellRefinementBranch); // might like to cache this result (there is a repeated, inefficient call inside this method)
  } else {
    // if there aren't any refinements, then there aren't any internal sub-subcells...
  }
  
  // these ordinals are relative to the fine subcell; we need to map them to the fine cell
  
  map< unsigned, set<unsigned> > internalSubcellOrdinals;
//  for (int d=0; d<subcdim; d++) {
  for (map< unsigned, set<unsigned> >::iterator entryIt = internalSubSubcellOrdinals.begin(); entryIt != internalSubSubcellOrdinals.end(); entryIt++) {
    unsigned d = entryIt->first;
    set<unsigned> subSubcellOrdinals = entryIt->second;
    set<unsigned> subcellOrdinals;
    for (set<unsigned>::iterator ordIt=subSubcellOrdinals.begin(); ordIt != subSubcellOrdinals.end(); ordIt++) {
      unsigned subcellOrdinal = CamelliaCellTools::subcellOrdinalMap(finerBasis->domainTopology(), subcdim, subcord, d, *ordIt);
      subcellOrdinals.insert(subcellOrdinal);
    }
    internalSubcellOrdinals[d] = subcellOrdinals;
  }
  
  set<unsigned> internalDofOrdinals;
  // first, add all dofs internal to the fine subcell:
  set<int> fineSubcellInternalDofs = finerBasis->dofOrdinalsForSubcell(subcdim, subcord);
  internalDofOrdinals.insert(fineSubcellInternalDofs.begin(),fineSubcellInternalDofs.end());
  for (int d=0; d<subcdim; d++) {
    set<unsigned> subcellOrdinals = internalSubcellOrdinals[d];
    for (set<unsigned>::iterator ordIt=subcellOrdinals.begin(); ordIt != subcellOrdinals.end(); ordIt++) {
      unsigned subcellOrdinal = *ordIt;
      set<int> subcellDofs = finerBasis->dofOrdinalsForSubcell(d, subcellOrdinal);
      internalDofOrdinals.insert(subcellDofs.begin(),subcellDofs.end());
    }
  }
  return internalDofOrdinals;
}

unsigned BasisReconciliation::minimumSubcellDimension(BasisPtr basis) {
  // use the functionSpace to determine what continuities should be enforced:
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = basis->functionSpace();
  
  int d = basis->domainTopology().getDimension();
  int minSubcellDimension = d-1;
  switch (fs) {
    case IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD:
    case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD:
    case IntrepidExtendedTypes::FUNCTION_SPACE_TENSOR_HGRAD:
      minSubcellDimension = 0; // vertices
      break;
    case IntrepidExtendedTypes::FUNCTION_SPACE_HCURL:
      minSubcellDimension = 1; // edges
      break;
    case IntrepidExtendedTypes::FUNCTION_SPACE_HDIV:
    case IntrepidExtendedTypes::FUNCTION_SPACE_HDIV_FREE:
      minSubcellDimension = d-1; // faces in 3D, edges in 2D.  (Unsure if this is right in 4D)
      break;
    case IntrepidExtendedTypes::FUNCTION_SPACE_HVOL:
    case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL:
      minSubcellDimension = d; // i.e. no continuities enforced
      break;
    default:
      cout << "ERROR: Unhandled functionSpace in BasisReconciliation.";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled functionSpace()");
      break;
  }
  return minSubcellDimension;
}

SubBasisReconciliationWeights BasisReconciliation::weightsForCoarseSubcell(SubBasisReconciliationWeights &weights, BasisPtr constrainingBasis, unsigned subcdim, unsigned subcord, bool includeSubsubcells) {
  int minSubcellDimension = includeSubsubcells ? 0 : subcdim;
  set<int> coarseDofsToInclude = constrainingBasis->dofOrdinalsForSubcell(subcdim, subcord, minSubcellDimension);
  vector<unsigned> columnOrdinalsToInclude;
  int columnOrdinal = 0;
  for (set<int>::iterator coarseOrdinalIt = weights.coarseOrdinals.begin(); coarseOrdinalIt != weights.coarseOrdinals.end(); coarseOrdinalIt++) {
    int coarseOrdinal = *coarseOrdinalIt;
    if (coarseDofsToInclude.find(coarseOrdinal) != coarseDofsToInclude.end()) {
      columnOrdinalsToInclude.push_back(columnOrdinal);
    }
    columnOrdinal++;
  }
  
  SubBasisReconciliationWeights filteredWeights;
  
  filteredWeights.coarseOrdinals = coarseDofsToInclude;
  filteredWeights.fineOrdinals = weights.fineOrdinals; // TODO: consider filtering out zero rows, too.
  
  filteredWeights.weights = FieldContainer<double>(weights.fineOrdinals.size(), columnOrdinalsToInclude.size());
  
  for (int i=0; i<filteredWeights.weights.dimension(0); i++) {
    for (int j=0; j<filteredWeights.weights.dimension(1); j++) {
      filteredWeights.weights(i,j) = weights.weights(i,columnOrdinalsToInclude[j]);
    }
  }
  return filteredWeights;
}

/*FieldContainer<double> BasisReconciliation::subBasisReconciliationWeightsForSubcell(SubBasisReconciliationWeights &subBasisWeights, unsigned subcdim,
                                                                                    BasisPtr fineBasis, unsigned fineSubcord,
                                                                                    BasisPtr coarseBasis, unsigned coarseSubcord,
                                                                                    set<unsigned> &fineBasisDofOrdinals) {
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = fineBasis->functionSpace();
  bool isL2 = (fs == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) || (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL);
  
  set<int> fineBasisSubcellOrdinals, coarseBasisSubcellOrdinals;
  if (!isL2) {
    fineBasisSubcellOrdinals = fineBasis->dofOrdinalsForSubcell(subcdim, fineSubcord, subcdim);
    coarseBasisSubcellOrdinals = coarseBasis->dofOrdinalsForSubcell(subcdim, coarseSubcord, subcdim);
//    Camellia::print("fineBasisSubcellOrdinals", fineBasisSubcellOrdinals);
//    Camellia::print("coarseBasisSubcellOrdinals", coarseBasisSubcellOrdinals);
  } else {
    int spaceDim = fineBasis->domainTopology().getDimension();
    if (subcdim==spaceDim) {
      // L2 basis, so everything is interior:
      fineBasisSubcellOrdinals = fineBasis->dofOrdinalsForSubcells(spaceDim,true);
      coarseBasisSubcellOrdinals = coarseBasis->dofOrdinalsForSubcells(spaceDim,true);
    }
    // if L2 and subcdim != spaceDim, there are no corresponding dofs for the subcell.
  }
  
  set<unsigned> rowFilter, colFilter;
  
  vector<unsigned> subBasisFineOrdinals(subBasisWeights.fineOrdinals.begin(),subBasisWeights.fineOrdinals.end());
  vector<unsigned> subBasisCoarseOrdinals(subBasisWeights.coarseOrdinals.begin(),subBasisWeights.coarseOrdinals.end());
  
  fineBasisDofOrdinals.clear();
  
  for (int i=0; i<subBasisFineOrdinals.size(); i++) {
    if (fineBasisSubcellOrdinals.find( subBasisFineOrdinals[i] ) != fineBasisSubcellOrdinals.end() ) {
      rowFilter.insert(i);
      fineBasisDofOrdinals.insert(subBasisFineOrdinals[i]);
    }
  }
  
  for (int j=0; j<subBasisCoarseOrdinals.size(); j++) {
    if (coarseBasisSubcellOrdinals.find( subBasisCoarseOrdinals[j] ) != coarseBasisSubcellOrdinals.end() ) {
      colFilter.insert(j);
    }
  }
  
  if (rowFilter.size() != subBasisFineOrdinals.size()) {
    cout << "Error: some required rows aren't present in subBasisWeights.\n";
    Camellia::print("subBasisFineOrdinals", subBasisFineOrdinals);
    Camellia::print("rowFilter", rowFilter);
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: some required rows aren't present in subBasisWeights.");
  }
  
  if (colFilter.size() != subBasisCoarseOrdinals.size()) {
    cout << "Error: some required columns aren't present in subBasisWeights.\n";
    Camellia::print("subBasisCoarseOrdinals", subBasisCoarseOrdinals);
    Camellia::print("coarseBasisSubcellOrdinals", coarseBasisSubcellOrdinals);
    Camellia::print("colFilter", colFilter);
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: some required columns aren't present in subBasisWeights.");
  }
  
  FieldContainer<double> constraintMatrixSubcell = SerialDenseWrapper::getSubMatrix(subBasisWeights.weights, rowFilter, colFilter);
  
  return constraintMatrixSubcell;
}*/