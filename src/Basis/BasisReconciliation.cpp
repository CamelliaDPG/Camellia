//
//  BasisReconciliation.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 11/19/13.
//
//

#include "BasisReconciliation.h"

#include "BasisEvaluation.h"

#include "Intrepid_DefaultCubatureFactory.hpp"

#include "SerialDenseMatrixUtility.h"

#include "Intrepid_FunctionSpaceTools.hpp"

void sizeFCForBasisValues(FieldContainer<double> &fc, BasisPtr basis, int numPoints, bool includeCellDimension = false) {
  // values should have shape: (F,P[,D,D,...]) where the # of D's = rank of the basis's range
  Teuchos::Array<int> dim;
  if (includeCellDimension) {
    dim.push_back(1);
  }
  dim.push_back(basis->getCardinality()); // F
  dim.push_back(numPoints); // P
  for (int d=0; d<basis->rangeRank(); d++) {
    dim.push_back(basis->rangeDimension()); // D
  }
  fc.resize(dim);
}

FieldContainer<double> BasisReconciliation::computeConstrainedWeights(BasisPtr largerBasis, BasisPtr smallerBasis) {
  // we could define things in terms of Functions, and then use Projector class.  But this is simple enough that it's probably worth it to do it more manually.
  // (also, I'm a bit concerned about the expense here, and the present implementation hopefully will be a bit lighter weight.)

  shards::CellTopology cellTopo = largerBasis->domainTopology();
  TEUCHOS_TEST_FOR_EXCEPTION(cellTopo.getBaseKey() != smallerBasis->domainTopology().getBaseKey(), std::invalid_argument, "Bases must agree on domain topology.");
  
  int cubDegree = largerBasis->getDegree() + smallerBasis->getDegree();

  DefaultCubatureFactory<double> cubFactory;
  Teuchos::RCP<Cubature<double> > cellTopoCub = cubFactory.create(cellTopo, cubDegree);
  
  int cubDim       = cellTopoCub->getDimension();
  int numCubPoints = cellTopoCub->getNumPoints();
  
  FieldContainer<double> cubPoints(numCubPoints, cubDim);
  FieldContainer<double> cubWeights(numCubPoints);
  
  cellTopoCub->getCubature(cubPoints, cubWeights);
  
  FieldContainer<double> largerBasisValues;
  FieldContainer<double> largerBasisValuesWeighted;
  sizeFCForBasisValues(largerBasisValues, largerBasis, numCubPoints);
  
  FieldContainer<double> smallerBasisValues;
  sizeFCForBasisValues(smallerBasisValues, smallerBasis, numCubPoints);
  
  largerBasis->getValues(largerBasisValues, cubPoints, OPERATOR_VALUE);
  smallerBasis->getValues(smallerBasisValues, cubPoints, OPERATOR_VALUE);

  // resize things with dummy cell dimension:
  sizeFCForBasisValues(smallerBasisValues, smallerBasis, numCubPoints, true);
  sizeFCForBasisValues(largerBasisValues, largerBasis, numCubPoints, true);
  sizeFCForBasisValues(largerBasisValuesWeighted, largerBasis, numCubPoints, true);
  cubWeights.resize(1,numCubPoints); // dummy cell dimension
  FunctionSpaceTools::multiplyMeasure<double>(largerBasisValuesWeighted, cubWeights, largerBasisValues);
  
  FieldContainer<double> constrainedWeights(largerBasis->getCardinality(),smallerBasis->getCardinality());
  
  FieldContainer<double> lhsValues(1,largerBasis->getCardinality(),largerBasis->getCardinality());
  FieldContainer<double> rhsValues(1,largerBasis->getCardinality(),smallerBasis->getCardinality());
  
  FunctionSpaceTools::integrate<double>(lhsValues,largerBasisValues,largerBasisValuesWeighted,COMP_CPP);
  FunctionSpaceTools::integrate<double>(rhsValues,largerBasisValuesWeighted,smallerBasisValues,COMP_CPP);
  
  lhsValues.resize(lhsValues.dimension(1),lhsValues.dimension(2));
  rhsValues.resize(rhsValues.dimension(1),rhsValues.dimension(2));
  SerialDenseMatrixUtility::solveSystemMultipleRHS(constrainedWeights, lhsValues, rhsValues);
  
  return constrainedWeights;
}

FieldContainer<double> BasisReconciliation::computeConstrainedWeights(BasisPtr largerBasis, BasisPtr smallerBasis, int largerBasisSideIndex, int smallerBasisSideIndex,
                                                                      unsigned vertexNodePermutation) {
  FieldContainer<double> weights;
  
  // use the functionSpace to determine what continuities should be enforced:
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = largerBasis->functionSpace();
  TEUCHOS_TEST_FOR_EXCEPTION(fs != smallerBasis->functionSpace(), std::invalid_argument, "Bases must agree on functionSpace().");
  
  int d = largerBasis->domainTopology().getDimension();
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
  
  set<int> largeBasisDofOrdinals = largerBasis->dofOrdinalsForSubcell(sideDimension, largerBasisSideIndex, minSubcellDimension);
  set<int> smallBasisDofOrdinals = smallerBasis->dofOrdinalsForSubcell(sideDimension, smallerBasisSideIndex, minSubcellDimension);
  
  // we probably want to cache these dofOrdinal sets, and include them in the return (alternative would be to call the dofOrdinalsForSubcell in caller, but
  // that's a bit annoying, and dofOrdinalsForSubcell involves a bit of construction)
  
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Method unfinished");
  
  return weights;
}

const FieldContainer<double>& BasisReconciliation::constrainedWeights(BasisPtr largerBasis, BasisPtr smallerBasis) {
  FieldContainer<double> weights;
  
  pair< Camellia::Basis<>*, Camellia::Basis<>* > cacheKey = make_pair(largerBasis.get(), smallerBasis.get());
  
  if (_simpleReconciliationWeights.find(cacheKey) != _simpleReconciliationWeights.end()) {
    return _simpleReconciliationWeights.find(cacheKey)->second;
  }

  // compute weights
  _simpleReconciliationWeights[cacheKey] = computeConstrainedWeights(largerBasis, smallerBasis);
  
  return _simpleReconciliationWeights[cacheKey];
}

const FieldContainer<double> & BasisReconciliation::constrainedWeights(BasisPtr largerBasis, BasisPtr smallerBasis, int largerBasisSideIndex, int smallerBasisSideIndex,
                                                                       unsigned vertexNodePermutation) {
  SideBasisRestriction largeSideRestriction = make_pair(largerBasis.get(), largerBasisSideIndex);
  SideBasisRestriction smallSideRestriction = make_pair(smallerBasis.get(), smallerBasisSideIndex);
  
  pair< pair <SideBasisRestriction, SideBasisRestriction>, unsigned > cacheKey = make_pair( make_pair(largeSideRestriction, smallSideRestriction), vertexNodePermutation );
  
  if (_sideReconciliationWeights.find(cacheKey) != _sideReconciliationWeights.end()) {
    return _sideReconciliationWeights.find(cacheKey)->second;
  }
  
  _sideReconciliationWeights[cacheKey] = computeConstrainedWeights(largerBasis, smallerBasis, largerBasisSideIndex, smallerBasisSideIndex, vertexNodePermutation);
  
  return _sideReconciliationWeights[cacheKey];
}
