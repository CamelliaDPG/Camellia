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
  return permutedCubaturePoints;
}

SubBasisReconciliationWeights BasisReconciliation::composedSubBasisReconciliationWeights(SubBasisReconciliationWeights aWeights, SubBasisReconciliationWeights bWeights) {
  if (aWeights.coarseOrdinals.size() != bWeights.fineOrdinals.size()) {
    cout << "aWeights and bWeights are incompatible...\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "aWeights and bWeights are incompatible...");
  }
  SubBasisReconciliationWeights cWeights;
  cWeights.weights = FieldContainer<double>(aWeights.fineOrdinals.size(), bWeights.coarseOrdinals.size());
  FieldContainer<double> aMatrix = aWeights.weights;
  FieldContainer<double> bMatrix = bWeights.weights;
  SerialDenseWrapper::multiply(cWeights.weights, aWeights.weights, bWeights.weights);
  cWeights.fineOrdinals = aWeights.fineOrdinals;
  cWeights.coarseOrdinals = bWeights.coarseOrdinals;
  return cWeights;
}

FieldContainer<double> BasisReconciliation::computeConstrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis, Permutation vertexPermutation) {
  // we could define things in terms of Functions, and then use Projector class.  But this is simple enough that it's probably worth it to do it more manually.
  // (also, I'm a bit concerned about the expense here, and the present implementation hopefully will be a bit lighter weight.)
  
  shards::CellTopology cellTopo = finerBasis->domainTopology();
  TEUCHOS_TEST_FOR_EXCEPTION(cellTopo.getBaseKey() != coarserBasis->domainTopology().getBaseKey(), std::invalid_argument, "Bases must agree on domain topology.");
  
  int cubDegree = finerBasis->getDegree() * 2;
  BasisCachePtr fineBasisCache = Teuchos::rcp( new BasisCache(cellTopo, cubDegree, false) );
  BasisCachePtr coarseBasisCache = Teuchos::rcp( new BasisCache(cellTopo, cubDegree, false) );
  
  coarseBasisCache->setRefCellPoints(permutedCubaturePoints(fineBasisCache, vertexPermutation));
  
  FieldContainer<double> finerBasisValues = *fineBasisCache->getValues(finerBasis, OP_VALUE);
  FieldContainer<double> coarserBasisValues = *coarseBasisCache->getValues(coarserBasis, OP_VALUE);
  
  set<int> filteredDofOrdinalsForFinerBasis = interiorDofOrdinalsForBasis(finerBasis);
  int interiorCardinalityFine = filteredDofOrdinalsForFinerBasis.size();
  
  set<int> filteredDofOrdinalsForCoarserBasis = interiorDofOrdinalsForBasis(coarserBasis);
  int interiorCardinalityCoarse = filteredDofOrdinalsForCoarserBasis.size();
  
  FieldContainer<double> constrainedWeights(interiorCardinalityFine,interiorCardinalityCoarse);
  if (interiorCardinalityFine==0) { // empty constraint
    cout << "WARNING: encountered empty constraint.\n";
    return constrainedWeights;
  }
  
  FieldContainer<double> finerBasisValuesFiltered = filterBasisValues(finerBasisValues, filteredDofOrdinalsForFinerBasis);
  FieldContainer<double> coarserBasisValuesFiltered = filterBasisValues(coarserBasisValues, filteredDofOrdinalsForCoarserBasis);
  
  FieldContainer<double> finerBasisValuesFilteredWeighted;
  
  // resize things with dummy cell dimension:
  int numCubPoints = fineBasisCache->getRefCellPoints().dimension(0); // (P,D)
  sizeFCForBasisValues(coarserBasisValuesFiltered, coarserBasis, numCubPoints, true, interiorCardinalityCoarse);
  sizeFCForBasisValues(finerBasisValuesFiltered, finerBasis, numCubPoints, true, interiorCardinalityFine);
  sizeFCForBasisValues(finerBasisValuesFilteredWeighted, finerBasis, numCubPoints, true, interiorCardinalityFine);
  
  FieldContainer<double> refCellNodes(cellTopo.getNodeCount(),cellTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);
  refCellNodes.resize(1,refCellNodes.dimension(0),refCellNodes.dimension(1));
  fineBasisCache->setPhysicalCellNodes(refCellNodes, vector<GlobalIndexType>(1,0), false);
  FieldContainer<double> cubWeights = fineBasisCache->getWeightedMeasures();
  FunctionSpaceTools::multiplyMeasure<double>(finerBasisValuesFilteredWeighted, cubWeights, finerBasisValuesFiltered);
  
  FieldContainer<double> lhsValues(1,interiorCardinalityFine,interiorCardinalityFine);
  FieldContainer<double> rhsValues(1,interiorCardinalityFine,interiorCardinalityCoarse);
  
  FunctionSpaceTools::integrate<double>(lhsValues,finerBasisValuesFiltered,finerBasisValuesFilteredWeighted,COMP_CPP);
  FunctionSpaceTools::integrate<double>(rhsValues,finerBasisValuesFilteredWeighted,coarserBasisValuesFiltered,COMP_CPP);
  
  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));
  SerialDenseMatrixUtility::solveSystemMultipleRHS(constrainedWeights, lhsValues, rhsValues);
  
  return constrainedWeights;
}

SubBasisReconciliationWeights BasisReconciliation::computeConstrainedWeights(BasisPtr finerBasis, int finerBasisSideIndex, BasisPtr coarserBasis, int coarserBasisSideIndex,
                                                                             unsigned vertexNodePermutation) {
  SubBasisReconciliationWeights weights;
  
  // use the functionSpace to determine what continuities should be enforced:
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = finerBasis->functionSpace();
  TEUCHOS_TEST_FOR_EXCEPTION(fs != coarserBasis->functionSpace(), std::invalid_argument, "Bases must agree on functionSpace().");
  
  int d = finerBasis->domainTopology().getDimension();
  int minSubcellDimension = d-1;
  int sideDimension = d-1;
  switch (fs) {
    case IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD:
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
      minSubcellDimension = d; // i.e. no continuities enforced
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled functionSpace()");
      break;
  }
  
  weights.fineOrdinals = finerBasis->dofOrdinalsForSubcell(sideDimension, finerBasisSideIndex, minSubcellDimension);
  weights.coarseOrdinals = coarserBasis->dofOrdinalsForSubcell(sideDimension, coarserBasisSideIndex, minSubcellDimension);
  weights.weights.resize(weights.fineOrdinals.size(), weights.coarseOrdinals.size());
  
  if (weights.fineOrdinals.size() == 0) {
    if (weights.coarseOrdinals.size() != 0) {
      cout << "ERROR in BasisReconciliation: empty fine basis (when restricted to the indicated side), but non-empty coarse basis.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "In BasisReconciliation, empty fine basis (when restricted to the indicated side), but non-empty coarse basis.");
    } else {
      // both empty: return empty weights container
      return weights;
    }
  }

  int cubDegree = finerBasis->getDegree() * 2; // on LHS, will integrate finerBasis against itself
  shards::CellTopology coarseTopo = coarserBasis->domainTopology();
  BasisCachePtr coarserBasisVolumeCache = Teuchos::rcp( new BasisCache(coarseTopo, cubDegree, false) ); // false: don't create all the side caches, since we just want one side
  BasisCachePtr coarseSideBasisCache = BasisCache::sideBasisCache(coarserBasisVolumeCache, coarserBasisSideIndex); // this is a leaner way to construct a side cache, but we will need to inform about physicalCellNodes manually
  
  shards::CellTopology sideTopo = coarseTopo.getCellTopologyData(sideDimension, coarserBasisSideIndex);
  
  shards::CellTopology fineTopo = finerBasis->domainTopology();
  BasisCachePtr finerBasisVolumeCache = Teuchos::rcp( new BasisCache(fineTopo, cubDegree, false) );
  BasisCachePtr fineSideBasisCache = BasisCache::sideBasisCache(finerBasisVolumeCache, finerBasisSideIndex);
  
  // We compute on the fine basis reference cell, and on a permuted coarse basis reference cell
  
  // how do we get the nodes on the reference cell?  It appears Trilinos does not tell us, so we implement our own in CamelliaCellTools
  FieldContainer<double> fineVolumeRefCellNodes(fineTopo.getNodeCount(), d);
  CamelliaCellTools::refCellNodesForTopology(fineVolumeRefCellNodes, coarseTopo);
  // resize for "physical" nodes:
  unsigned oneCell = 1;
  fineVolumeRefCellNodes.resize(oneCell,fineVolumeRefCellNodes.dimension(0), fineVolumeRefCellNodes.dimension(1));
  fineSideBasisCache->setPhysicalCellNodes(fineVolumeRefCellNodes, vector<GlobalIndexType>(), false);

  FieldContainer<double> permutedSideCubaturePoints;
  if (d-1 > 0) {
    permutedSideCubaturePoints = permutedCubaturePoints(fineSideBasisCache, vertexNodePermutation);
    // create a ("volume") basis cache for the side topology--this allows us to determine the cubature points corresponding to the
    // permuted nodes.
//    BasisCachePtr sideCacheForPermutation = Teuchos::rcp( new BasisCache( sideTopo, cubDegree, false ) );
//    FieldContainer<double> permutedSideTopoNodes(sideTopo.getNodeCount(), sideTopo.getDimension());
//    CamelliaCellTools::refCellNodesForTopology(permutedSideTopoNodes, sideTopo, vertexNodePermutation);
//    sideCacheForPermutation->setRefCellPoints(fineSideBasisCache->getRefCellPoints()); // these should be the same, but doesn't hurt to be sure
//    permutedSideTopoNodes.resize(oneCell,permutedSideTopoNodes.dimension(0),permutedSideTopoNodes.dimension(1));
//    sideCacheForPermutation->setPhysicalCellNodes(permutedSideTopoNodes, vector<GlobalIndexType>(), false);
//    permutedSideCubaturePoints = sideCacheForPermutation->getPhysicalCubaturePoints();
//    // resize for reference space (no cellIndex dimension):
//    permutedSideCubaturePoints.resize(permutedSideCubaturePoints.dimension(1), permutedSideCubaturePoints.dimension(2));
  } else { // dim-0 topologies don't change under permutation...
    permutedSideCubaturePoints = fineSideBasisCache->getRefCellPoints();
  }

  FieldContainer<double> coarseVolumeRefCellNodes(coarseTopo.getNodeCount(), d);
  CamelliaCellTools::refCellNodesForTopology(coarseVolumeRefCellNodes, coarseTopo);
  coarseVolumeRefCellNodes.resize(oneCell,coarseVolumeRefCellNodes.dimension(0), coarseVolumeRefCellNodes.dimension(1));
  coarseSideBasisCache->setPhysicalCellNodes(coarseVolumeRefCellNodes, vector<GlobalIndexType>(), false);
  coarseSideBasisCache->setRefCellPoints(permutedSideCubaturePoints); // this makes computing weighted values illegal (the cubature weights are no longer valid, so they're cleared automatically)
  
  Teuchos::RCP< const FieldContainer<double> > finerBasisValues = fineSideBasisCache->getTransformedValues(finerBasis, OP_VALUE, true);
  Teuchos::RCP< const FieldContainer<double> > finerBasisValuesWeighted = fineSideBasisCache->getTransformedWeightedValues(finerBasis, OP_VALUE, true);
  Teuchos::RCP< const FieldContainer<double> > coarserBasisValues = coarseSideBasisCache->getTransformedValues(coarserBasis, OP_VALUE, true);
  
  FieldContainer<double> fineBasisValuesFiltered, fineBasisValuesFilteredWeighted, coarserBasisValuesFiltered;
  unsigned numPoints = fineSideBasisCache->getRefCellPoints().dimension(0);
  sizeFCForBasisValues(fineBasisValuesFiltered, finerBasis, numPoints, true, weights.fineOrdinals.size());
  sizeFCForBasisValues(fineBasisValuesFilteredWeighted, finerBasis, numPoints, true, weights.fineOrdinals.size());
  sizeFCForBasisValues(coarserBasisValuesFiltered, coarserBasis, numPoints, true, weights.coarseOrdinals.size());

  filterFCValues(fineBasisValuesFiltered, *(finerBasisValues.get()), weights.fineOrdinals, finerBasis->getCardinality());
  filterFCValues(fineBasisValuesFilteredWeighted, *(finerBasisValuesWeighted.get()), weights.fineOrdinals, finerBasis->getCardinality());
  filterFCValues(coarserBasisValuesFiltered, *(coarserBasisValues.get()), weights.coarseOrdinals, coarserBasis->getCardinality());
  
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


FieldContainer<double> BasisReconciliation::computeConstrainedWeights(BasisPtr finerBasis, RefinementBranch refinements, BasisPtr coarserBasis, Permutation vertexPermutation) {
  
  if (refinements.size() == 0) {
    return computeConstrainedWeights(finerBasis, coarserBasis, vertexPermutation);
  }
  
  // there's a LOT of code duplication between this and the p-oriented computeConstrainedWeights(finerBasis,coarserBasis)
  // would be pretty easy to refactor to make them share the common code -- this one just needs an additional distinction between the cubature points for coarserBasis and those for finerBasis...
  
  // we could define things in terms of Functions, and then use Projector class.  But this is simple enough that it's probably worth it to do it more manually.
  // (also, I'm a bit concerned about the expense here, and the present implementation hopefully will be a bit lighter weight.)
  
  shards::CellTopology cellTopo = finerBasis->domainTopology();
  int spaceDim = cellTopo.getDimension();
  
  int cubDegree = finerBasis->getDegree() * 2;
  
  BasisCachePtr fineBasisCache = Teuchos::rcp( new BasisCache(cellTopo, cubDegree, false) );
  BasisCachePtr coarseBasisCache = Teuchos::rcp( new BasisCache(cellTopo, cubDegree, false) );
  
  FieldContainer<double> refCellNodes(cellTopo.getNodeCount(),cellTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);
  refCellNodes.resize(1,refCellNodes.dimension(0),refCellNodes.dimension(1));
  fineBasisCache->setPhysicalCellNodes(refCellNodes, vector<GlobalIndexType>(1,0), false);
  FieldContainer<double> cubWeights = fineBasisCache->getWeightedMeasures();

  int numCubPoints = fineBasisCache->getRefCellPoints().dimension(0); // (P,D)
  FieldContainer<double> fineCellNodesInCoarseRefCell = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(refinements);
  fineCellNodesInCoarseRefCell.resize(1,fineCellNodesInCoarseRefCell.dimension(0),fineCellNodesInCoarseRefCell.dimension(1));
  FieldContainer<double> cubPointsCoarseBasis(1,numCubPoints,spaceDim);
  FieldContainer<double> permutedCubPointsFineBasis = permutedCubaturePoints(fineBasisCache, vertexPermutation);
  CellTools<double>::mapToPhysicalFrame(cubPointsCoarseBasis,permutedCubPointsFineBasis,fineCellNodesInCoarseRefCell,cellTopo);
  cubPointsCoarseBasis.resize(numCubPoints,spaceDim);
  
  coarseBasisCache->setRefCellPoints(cubPointsCoarseBasis);
  
  FieldContainer<double> finerBasisValuesFilteredWeighted;
  
  FieldContainer<double> finerBasisValues = *fineBasisCache->getValues(finerBasis, OP_VALUE);
  FieldContainer<double> coarserBasisValues = *coarseBasisCache->getValues(coarserBasis, OP_VALUE);
  
  set<unsigned> filteredDofOrdinalsForFinerBasis = internalDofIndicesForFinerBasis(finerBasis, refinements);
  set<int> filteredDofOrdinalsForFinerBasisInt(filteredDofOrdinalsForFinerBasis.begin(),filteredDofOrdinalsForFinerBasis.end());
  FieldContainer<double> finerBasisValuesFiltered = filterBasisValues(finerBasisValues, filteredDofOrdinalsForFinerBasisInt);
  
  set<int> filteredDofOrdinalsForCoarserBasis = interiorDofOrdinalsForBasis(coarserBasis);
  FieldContainer<double> coarserBasisValuesFiltered = filterBasisValues(coarserBasisValues, filteredDofOrdinalsForCoarserBasis);
  
  // resize things with dummy cell dimension:
  bool includeCellDimension = true;
  sizeFCForBasisValues(coarserBasisValuesFiltered, coarserBasis, numCubPoints, includeCellDimension, filteredDofOrdinalsForCoarserBasis.size());
  sizeFCForBasisValues(finerBasisValuesFiltered, finerBasis, numCubPoints, includeCellDimension, filteredDofOrdinalsForFinerBasis.size());
  sizeFCForBasisValues(finerBasisValuesFilteredWeighted, finerBasis, numCubPoints, includeCellDimension, filteredDofOrdinalsForFinerBasis.size());
  cubWeights.resize(1,numCubPoints); // dummy cell dimension
  FunctionSpaceTools::multiplyMeasure<double>(finerBasisValuesFilteredWeighted, cubWeights, finerBasisValuesFiltered);
  
  FieldContainer<double> constrainedWeights(filteredDofOrdinalsForFinerBasis.size(),filteredDofOrdinalsForCoarserBasis.size());
  
  FieldContainer<double> lhsValues(1,filteredDofOrdinalsForFinerBasis.size(),filteredDofOrdinalsForFinerBasis.size());
  FieldContainer<double> rhsValues(1,filteredDofOrdinalsForFinerBasis.size(),filteredDofOrdinalsForCoarserBasis.size());
  
  FunctionSpaceTools::integrate<double>(lhsValues,finerBasisValuesFiltered,finerBasisValuesFilteredWeighted,COMP_CPP);
  FunctionSpaceTools::integrate<double>(rhsValues,finerBasisValuesFilteredWeighted,coarserBasisValuesFiltered,COMP_CPP);
  
  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));
  SerialDenseMatrixUtility::solveSystemMultipleRHS(constrainedWeights, lhsValues, rhsValues);
  
  return constrainedWeights;
}

// matching along sides:
SubBasisReconciliationWeights BasisReconciliation::computeConstrainedWeights(BasisPtr finerBasis, int fineAncestralSideIndex, RefinementBranch &volumeRefinements,
                                                                             RefinementBranch &sideRefinements, BasisPtr coarserBasis, int coarserBasisSideIndex,
                                                                             unsigned vertexNodePermutation) {
  
  SubBasisReconciliationWeights weights;
  
  // use the functionSpace to determine what continuities should be enforced:
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = finerBasis->functionSpace();
  TEUCHOS_TEST_FOR_EXCEPTION(fs != coarserBasis->functionSpace(), std::invalid_argument, "Bases must agree on functionSpace().");
  
  int spaceDim = finerBasis->domainTopology().getDimension();
  int minSubcellDimension = spaceDim-1;
  int sideDimension = spaceDim-1;
  switch (fs) {
    case IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD:
    case IntrepidExtendedTypes::FUNCTION_SPACE_TENSOR_HGRAD:
      minSubcellDimension = 0; // vertices
      break;
    case IntrepidExtendedTypes::FUNCTION_SPACE_HCURL:
      minSubcellDimension = 1; // edges
      break;
    case IntrepidExtendedTypes::FUNCTION_SPACE_HDIV:
    case IntrepidExtendedTypes::FUNCTION_SPACE_HDIV_FREE:
      minSubcellDimension = spaceDim-1; // faces in 3D, edges in 2D.  (Unsure if this is right in 4D)
      break;
    case IntrepidExtendedTypes::FUNCTION_SPACE_HVOL:
      minSubcellDimension = spaceDim; // i.e. no continuities enforced
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled functionSpace()");
      break;
  }
  
  // figure out fineSideIndex
  unsigned fineSideIndex = fineAncestralSideIndex;
  for (int refIndex=0; refIndex<volumeRefinements.size(); refIndex++) {
    RefinementPattern* refPattern = volumeRefinements[refIndex].first;
    unsigned childIndex = volumeRefinements[refIndex].second;
    vector< pair<unsigned, unsigned> > childrenForSide = refPattern->childrenForSides()[fineSideIndex];
    for (vector< pair<unsigned, unsigned> >::iterator entryIt=childrenForSide.begin(); entryIt != childrenForSide.end(); entryIt++) {
      if (entryIt->first == childIndex) {
        fineSideIndex = entryIt->second;
      }
    }
  }
  
  weights.fineOrdinals = finerBasis->dofOrdinalsForSubcell(sideDimension, fineSideIndex, minSubcellDimension);
  weights.coarseOrdinals = coarserBasis->dofOrdinalsForSubcell(sideDimension, coarserBasisSideIndex, minSubcellDimension);
  
//  print("fineOrdinals", weights.fineOrdinals);
  
  int cubDegree = finerBasis->getDegree() * 2; // on LHS, will integrate finerBasis against itself
  unsigned oneCell = 1;
  
  // determine cubature points as seen by the fine basis
  shards::CellTopology fineTopo = finerBasis->domainTopology();
  BasisCachePtr fineVolumeCache = Teuchos::rcp( new BasisCache(fineTopo, cubDegree, false) );
  BasisCachePtr fineSideBasisCache = BasisCache::sideBasisCache(fineVolumeCache, fineSideIndex);
  FieldContainer<double> fineVolumeRefNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(volumeRefinements);
//  FieldContainer<double> fineVolumeRefNodes(fineTopo.getNodeCount(),spaceDim);
//  CamelliaCellTools::refCellNodesForTopology(fineVolumeRefNodes, fineTopo);
  fineVolumeRefNodes.resize(oneCell,fineVolumeRefNodes.dimension(0),fineVolumeRefNodes.dimension(1));
  fineSideBasisCache->setPhysicalCellNodes(fineVolumeRefNodes, vector<GlobalIndexType>(), false);
//  FieldContainer<double> fineVolumeCubaturePoints = fineSideBasisCache->getPhysicalCubaturePoints(); // these are in volume coordinates
//  fineVolumeCubaturePoints.resize(fineVolumeCubaturePoints.dimension(1),fineVolumeCubaturePoints.dimension(2));
  FieldContainer<double> fineSideCubaturePoints = fineSideBasisCache->getRefCellPoints();
  
  // now, we need to map those points into coarseSide/coarseVolume
  // coarseSide
  shards::CellTopology ancestralTopo = *volumeRefinements[0].first->parentTopology();
  shards::CellTopology ancestralSideTopo = ancestralTopo.getCellTopologyData(sideDimension, fineAncestralSideIndex);
  
  FieldContainer<double> coarseSideNodes(ancestralSideTopo.getNodeCount(),sideDimension);
  CamelliaCellTools::refCellNodesForTopology(coarseSideNodes, ancestralSideTopo, vertexNodePermutation);
  
  FieldContainer<double> fineSideNodesInCoarseSideTopology = RefinementPattern::descendantNodes(sideRefinements, coarseSideNodes);
  
  shards::CellTopology fineSideTopo = fineTopo.getCellTopologyData(sideDimension, fineSideIndex);
  BasisCachePtr sideBasisCacheAsVolume = Teuchos::rcp( new BasisCache(fineSideTopo, cubDegree, false) );
  sideBasisCacheAsVolume->setRefCellPoints(fineSideCubaturePoints); // should be the same, but to guard against changes in BasisCache, set these.
  fineSideNodesInCoarseSideTopology.resize(oneCell, fineSideNodesInCoarseSideTopology.dimension(0), fineSideNodesInCoarseSideTopology.dimension(1));
  sideBasisCacheAsVolume->setPhysicalCellNodes(fineSideNodesInCoarseSideTopology, vector<GlobalIndexType>(), false);
  FieldContainer<double> coarseSideCubaturePoints = sideBasisCacheAsVolume->getPhysicalCubaturePoints();
  coarseSideCubaturePoints.resize(coarseSideCubaturePoints.dimension(1), coarseSideCubaturePoints.dimension(2));
  
  // coarseVolume
  shards::CellTopology coarseTopo = coarserBasis->domainTopology();
  BasisCachePtr coarseBasisVolumeCache = Teuchos::rcp( new BasisCache(coarseTopo, cubDegree, false) ); // false: don't create all the side caches, since we just want one side
  BasisCachePtr coarseSideBasisCache = BasisCache::sideBasisCache(coarseBasisVolumeCache, coarserBasisSideIndex); // this is a leaner way to construct a side cache, but we will need to inform about physicalCellNodes manually
  coarseSideBasisCache->setRefCellPoints(coarseSideCubaturePoints);
  FieldContainer<double> coarseVolumeRefNodes(coarseTopo.getNodeCount(),spaceDim);
  CamelliaCellTools::refCellNodesForTopology(coarseVolumeRefNodes, coarseTopo);
  coarseVolumeRefNodes.resize(oneCell,coarseVolumeRefNodes.dimension(0),coarseVolumeRefNodes.dimension(1));
  coarseSideBasisCache->setPhysicalCellNodes(coarseVolumeRefNodes, vector<GlobalIndexType>(), false);
//  FieldContainer<double> coarseVolumeCubaturePoints = coarseSideBasisCache->getPhysicalCubaturePoints();
//  coarseVolumeCubaturePoints.resize(coarseVolumeCubaturePoints.dimension(1),coarseVolumeCubaturePoints.dimension(2));
  
//  cout << "coarseSideBasisCache->getPhysicalCubaturePoints():\n" << coarseSideBasisCache->getPhysicalCubaturePoints();
  
  Teuchos::RCP< const FieldContainer<double> > finerBasisValues = fineSideBasisCache->getTransformedValues(finerBasis, OP_VALUE, true);
  Teuchos::RCP< const FieldContainer<double> > finerBasisValuesWeighted = fineSideBasisCache->getTransformedWeightedValues(finerBasis, OP_VALUE, true);
  Teuchos::RCP< const FieldContainer<double> > coarserBasisValues = coarseSideBasisCache->getTransformedValues(coarserBasis, OP_VALUE, true);

  FieldContainer<double> fineBasisValuesFiltered, fineBasisValuesFilteredWeighted, coarserBasisValuesFiltered;
  unsigned numPoints = fineSideBasisCache->getRefCellPoints().dimension(0);
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

const FieldContainer<double>& BasisReconciliation::constrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis, Permutation vertexPermutation) {
  FieldContainer<double> weights;
  
  BasisPair basisPair = make_pair(finerBasis.get(), coarserBasis.get());
  pair< BasisPair, Permutation > cacheKey = make_pair(basisPair, vertexPermutation);
  
  if (_simpleReconciliationWeights.find(cacheKey) != _simpleReconciliationWeights.end()) {
    return _simpleReconciliationWeights.find(cacheKey)->second;
  }

  // compute weights
  _simpleReconciliationWeights[cacheKey] = computeConstrainedWeights(finerBasis, coarserBasis, vertexPermutation);
  
  return _simpleReconciliationWeights[cacheKey];
}

const SubBasisReconciliationWeights & BasisReconciliation::constrainedWeights(BasisPtr finerBasis, int finerBasisSideIndex, BasisPtr coarserBasis, int coarserBasisSideIndex,
                                                                              Permutation vertexNodePermutation) {
  SideBasisRestriction fineSideRestriction = make_pair(finerBasis.get(), finerBasisSideIndex);
  SideBasisRestriction coarseSideRestriction = make_pair(coarserBasis.get(), coarserBasisSideIndex);
  
  pair< pair <SideBasisRestriction, SideBasisRestriction>, unsigned > cacheKey = make_pair( make_pair(fineSideRestriction, coarseSideRestriction), vertexNodePermutation );
  
  if (_sideReconciliationWeights.find(cacheKey) != _sideReconciliationWeights.end()) {
    return _sideReconciliationWeights.find(cacheKey)->second;
  }
  
  _sideReconciliationWeights[cacheKey] = computeConstrainedWeights(finerBasis, finerBasisSideIndex, coarserBasis, coarserBasisSideIndex, vertexNodePermutation);
  
  return _sideReconciliationWeights[cacheKey];
}

const FieldContainer<double> & BasisReconciliation::constrainedWeights(BasisPtr finerBasis, RefinementBranch refinements, BasisPtr coarserBasis, Permutation vertexNodePermutation) {
  BasisPair basisPair = make_pair(finerBasis.get(), coarserBasis.get());
  RefinedBasisPair refinedBasisPair = make_pair(basisPair, refinements);
  pair<RefinedBasisPair, Permutation> cacheKey = make_pair(refinedBasisPair,vertexNodePermutation);

  if (_simpleReconcilationWeights_h.find(cacheKey) == _simpleReconcilationWeights_h.end()) {
    FieldContainer<double> weights = computeConstrainedWeights(finerBasis,refinements,coarserBasis,vertexNodePermutation);
    _simpleReconcilationWeights_h[cacheKey] = weights;
  }

  return _simpleReconcilationWeights_h[cacheKey];
}

const SubBasisReconciliationWeights & BasisReconciliation::constrainedWeights(BasisPtr finerBasis, int finerBasisSideIndex, RefinementBranch &volumeRefinements,
                                                                              BasisPtr coarserBasis, int coarserBasisSideIndex, unsigned vertexNodePermutation) { // vertexPermutation is for the fine basis's ancestral orientation (how to permute side as seen by fine's ancestor to produce side as seen by coarse)...
  
//  if (finerBasis->domainTopology().getDimension()==1) { // then side topology will be a vertex
//    // we may want to handle vertex bases specially
//  }
  
  if (volumeRefinements.size()==0) {
    return constrainedWeights(finerBasis, finerBasisSideIndex, coarserBasis, coarserBasisSideIndex, vertexNodePermutation);
  }
  
  SideBasisRestriction fineRestriction = make_pair(finerBasis.get(), finerBasisSideIndex);
  SideBasisRestriction coarseRestriction = make_pair(coarserBasis.get(), coarserBasisSideIndex);
  
  SideRefinedBasisPair refinedBasisPair = make_pair(make_pair(fineRestriction, coarseRestriction), volumeRefinements);

  pair< SideRefinedBasisPair, unsigned > cacheKey = make_pair(refinedBasisPair, vertexNodePermutation);
  
  if (_sideReconcilationWeights_h.find(cacheKey) == _sideReconcilationWeights_h.end()) {
    RefinementBranch sideRefinements = RefinementPattern::sideRefinementBranch(volumeRefinements, finerBasisSideIndex);
    SubBasisReconciliationWeights weights = computeConstrainedWeights(finerBasis, finerBasisSideIndex, volumeRefinements, sideRefinements,
                                                                      coarserBasis, coarserBasisSideIndex, vertexNodePermutation);
    _sideReconcilationWeights_h[cacheKey] = weights;
  }
  
  return _sideReconcilationWeights_h[cacheKey];
}

FieldContainer<double> BasisReconciliation::filterBasisValues(const FieldContainer<double> &basisValues, set<int> &filter) {
  Teuchos::Array<int> dim;
  basisValues.dimensions(dim);
  int basisCardinality = dim[0];
  dim[0] = filter.size();
  FieldContainer<double> filteredValues(dim);
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
  bool isL2 = (basis->functionSpace() == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL); // if L2, we include all dofs, not just the interior ones
  int spaceDim = basis->domainTopology().getDimension();
  set<int> filteredDofOrdinalsForFinerBasis = isL2 ? basis->dofOrdinalsForSubcells(spaceDim, true) : basis->dofOrdinalsForInterior();
  return filteredDofOrdinalsForFinerBasis;
}

set<unsigned> BasisReconciliation::internalDofIndicesForFinerBasis(BasisPtr finerBasis, RefinementBranch refinements) {
  // which degrees of freedom in the finer basis have empty support on the boundary of the coarser basis's reference element? -- these are the ones for which the constrained weights are determined in computeConstrainedWeights.
  set<unsigned> internalDofOrdinals;
  unsigned spaceDim = finerBasis->domainTopology().getDimension();
  
  if (finerBasis->functionSpace() == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) {
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

FieldContainer<double> BasisReconciliation::subBasisReconciliationWeightsForSubcell(SubBasisReconciliationWeights &subBasisWeights, unsigned subcdim,
                                                                                    BasisPtr fineBasis, unsigned fineSubcord,
                                                                                    BasisPtr coarseBasis, unsigned coarseSubcord,
                                                                                    set<unsigned> &fineBasisDofOrdinals) {
  set<int> fineBasisSubcellOrdinals = fineBasis->dofOrdinalsForSubcell(subcdim, fineSubcord, subcdim);
  set<int> coarseBasisSubcellOrdinals = coarseBasis->dofOrdinalsForSubcell(subcdim, coarseSubcord, subcdim);
  
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
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: some required rows aren't present in subBasisWeights.");
  }
  
  if (colFilter.size() != subBasisCoarseOrdinals.size()) {
    cout << "Error: some required columns aren't present in subBasisWeights.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: some required columns aren't present in subBasisWeights.");
  }
  
  FieldContainer<double> constraintMatrixSubcell = SerialDenseWrapper::getSubMatrix(subBasisWeights.weights, rowFilter, colFilter);
  
  return constraintMatrixSubcell;
}