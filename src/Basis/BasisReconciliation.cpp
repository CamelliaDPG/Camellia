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

FieldContainer<double> BasisReconciliation::computeConstrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis) {
  // we could define things in terms of Functions, and then use Projector class.  But this is simple enough that it's probably worth it to do it more manually.
  // (also, I'm a bit concerned about the expense here, and the present implementation hopefully will be a bit lighter weight.)

  shards::CellTopology cellTopo = finerBasis->domainTopology();
  TEUCHOS_TEST_FOR_EXCEPTION(cellTopo.getBaseKey() != coarserBasis->domainTopology().getBaseKey(), std::invalid_argument, "Bases must agree on domain topology.");
  
  int cubDegree = finerBasis->getDegree() + coarserBasis->getDegree();

  DefaultCubatureFactory<double> cubFactory;
  Teuchos::RCP<Cubature<double> > cellTopoCub = cubFactory.create(cellTopo, cubDegree);
  
  int cubDim       = cellTopoCub->getDimension();
  int numCubPoints = cellTopoCub->getNumPoints();
  
  FieldContainer<double> cubPoints(numCubPoints, cubDim);
  FieldContainer<double> cubWeights(numCubPoints);
  
  cellTopoCub->getCubature(cubPoints, cubWeights);
  
  FieldContainer<double> finerBasisValues;
  FieldContainer<double> finerBasisValuesWeighted;
  sizeFCForBasisValues(finerBasisValues, finerBasis, numCubPoints);
  
  FieldContainer<double> coarserBasisValues;
  sizeFCForBasisValues(coarserBasisValues, coarserBasis, numCubPoints);
  
  finerBasis->getValues(finerBasisValues, cubPoints, OPERATOR_VALUE);
  coarserBasis->getValues(coarserBasisValues, cubPoints, OPERATOR_VALUE);

  // resize things with dummy cell dimension:
  sizeFCForBasisValues(coarserBasisValues, coarserBasis, numCubPoints, true);
  sizeFCForBasisValues(finerBasisValues, finerBasis, numCubPoints, true);
  sizeFCForBasisValues(finerBasisValuesWeighted, finerBasis, numCubPoints, true);
  cubWeights.resize(1,numCubPoints); // dummy cell dimension
  FunctionSpaceTools::multiplyMeasure<double>(finerBasisValuesWeighted, cubWeights, finerBasisValues);
  
  FieldContainer<double> constrainedWeights(finerBasis->getCardinality(),coarserBasis->getCardinality());
  
  FieldContainer<double> lhsValues(1,finerBasis->getCardinality(),finerBasis->getCardinality());
  FieldContainer<double> rhsValues(1,finerBasis->getCardinality(),coarserBasis->getCardinality());
  
  FunctionSpaceTools::integrate<double>(lhsValues,finerBasisValues,finerBasisValuesWeighted,COMP_CPP);
  FunctionSpaceTools::integrate<double>(rhsValues,finerBasisValuesWeighted,coarserBasisValues,COMP_CPP);
  
  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));
  SerialDenseMatrixUtility::solveSystemMultipleRHS(constrainedWeights, lhsValues, rhsValues);
  
  return constrainedWeights;
}

SubBasisReconciliationWeights BasisReconciliation::computeConstrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis, int finerBasisSideIndex, int coarserBasisSideIndex,
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
      minSubcellDimension = 2; // faces
      break;
    case IntrepidExtendedTypes::FUNCTION_SPACE_HVOL:
      minSubcellDimension = d; // i.e. no continuities enforced
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled functionSpace()");
      break;
  }
  
  weights.fineOrdinals = finerBasis->dofOrdinalsForSubcell(sideDimension, finerBasisSideIndex, minSubcellDimension);
  weights.coarseOrdinals = coarserBasis->dofOrdinalsForSubcell(sideDimension, coarserBasisSideIndex, minSubcellDimension);

  int cubDegree = finerBasis->getDegree() * 2; // on LHS, will integrate finerBasis against itself
  shards::CellTopology coarseTopo = coarserBasis->domainTopology();
  BasisCachePtr coarserBasisVolumeCache = Teuchos::rcp( new BasisCache(coarseTopo, cubDegree, false) ); // false: don't create all the side caches, since we just want one side
  BasisCachePtr coarseSideBasisCache = BasisCache::sideBasisCache(coarserBasisVolumeCache, coarserBasisSideIndex); // this is a leaner way to construct a side cache, but we will need to inform about physicalCellNodes manually
  
  shards::CellTopology sideTopo = coarseTopo.getCellTopologyData(sideDimension, coarserBasisSideIndex);
  
  shards::CellTopology fineTopo = finerBasis->domainTopology();
  BasisCachePtr finerBasisVolumeCache = Teuchos::rcp( new BasisCache(fineTopo, cubDegree, false) );
  BasisCachePtr fineSideBasisCache = BasisCache::sideBasisCache(finerBasisVolumeCache, finerBasisSideIndex);
  
  // We compute on the coarse basis reference cell, and on a permuted fine basis reference cell
  
  // how do we get the nodes on the reference cell?  It appears Trilinos does not tell us, so we implement our own in CamelliaCellTools
  FieldContainer<double> coarseVolumeRefCellNodes(coarseTopo.getNodeCount(), d);
  CamelliaCellTools::refCellNodesForTopology(coarseVolumeRefCellNodes, coarseTopo);
  // resize for "physical" nodes:
  coarseVolumeRefCellNodes.resize(1,coarseVolumeRefCellNodes.dimension(0), coarseVolumeRefCellNodes.dimension(1));
  coarseSideBasisCache->setPhysicalCellNodes(coarseVolumeRefCellNodes, vector<int>(), false);
  
  unsigned finerVolumePermutation = CamelliaCellTools::matchingVolumePermutationForSidePermutation(fineTopo, finerBasisSideIndex, vertexNodePermutation);
  FieldContainer<double> fineVolumeRefCellNodes(fineTopo.getNodeCount(), d);
  CamelliaCellTools::refCellNodesForTopology(fineVolumeRefCellNodes, fineTopo, finerVolumePermutation);
  fineVolumeRefCellNodes.resize(1,fineVolumeRefCellNodes.dimension(0), fineVolumeRefCellNodes.dimension(1));
  fineSideBasisCache->setPhysicalCellNodes(fineVolumeRefCellNodes, vector<int>(), false);
  
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

const FieldContainer<double>& BasisReconciliation::constrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis) {
  FieldContainer<double> weights;
  
  pair< Camellia::Basis<>*, Camellia::Basis<>* > cacheKey = make_pair(finerBasis.get(), coarserBasis.get());
  
  if (_simpleReconciliationWeights.find(cacheKey) != _simpleReconciliationWeights.end()) {
    return _simpleReconciliationWeights.find(cacheKey)->second;
  }

  // compute weights
  _simpleReconciliationWeights[cacheKey] = computeConstrainedWeights(finerBasis, coarserBasis);
  
  return _simpleReconciliationWeights[cacheKey];
}

const SubBasisReconciliationWeights & BasisReconciliation::constrainedWeights(BasisPtr finerBasis, BasisPtr coarserBasis, int finerBasisSideIndex, int coarserBasisSideIndex,
                                                                       unsigned vertexNodePermutation) {
  SideBasisRestriction fineSideRestriction = make_pair(finerBasis.get(), finerBasisSideIndex);
  SideBasisRestriction coarseSideRestriction = make_pair(coarserBasis.get(), coarserBasisSideIndex);
  
  pair< pair <SideBasisRestriction, SideBasisRestriction>, unsigned > cacheKey = make_pair( make_pair(fineSideRestriction, coarseSideRestriction), vertexNodePermutation );
  
  if (_sideReconciliationWeights.find(cacheKey) != _sideReconciliationWeights.end()) {
    return _sideReconciliationWeights.find(cacheKey)->second;
  }
  
  _sideReconciliationWeights[cacheKey] = computeConstrainedWeights(finerBasis, coarserBasis, finerBasisSideIndex, coarserBasisSideIndex, vertexNodePermutation);
  
  return _sideReconciliationWeights[cacheKey];
}
