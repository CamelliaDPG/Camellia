#include "BC.h"
#include "BCFunction.h"

void BC::imposeBC(FieldContainer<double> &dirichletValues, FieldContainer<bool> &imposeHere, 
                      int varID, FieldContainer<double> &unitNormals, BasisCachePtr basisCache) {
  // by default, call legacy version:
  // (basisCache->getPhysicalCubaturePoints() doesn't really return *cubature* points, but the boundary points
  //  that we're interested in)
  FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
  imposeBC(varID,physicalPoints,unitNormals,dirichletValues,imposeHere);
}

void BC::imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                      FieldContainer<double> &unitNormals,
                      FieldContainer<double> &dirichletValues,
                      FieldContainer<bool> &imposeHere) {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BC::imposeBC unimplemented.");
}

bool BC::singlePointBC(int varID) {
  return false; 
} 

bool BC::imposeZeroMeanConstraint(int varID) {
  return false;
}
// override if you want to implement a BC at a single, arbitrary point (and nowhere else).

// basisCoefficients has dimensions (C,F)
void BC::coefficientsForBC(FieldContainer<double> &basisCoefficients, Teuchos::RCP<BCFunction> bcFxn, 
                           BasisPtr basis, BasisCachePtr sideBasisCache) {
  int numFields = basis->getCardinality();
  TEUCHOS_TEST_FOR_EXCEPTION( basisCoefficients.dimension(1) != numFields, std::invalid_argument, "inconsistent basisCoefficients dimensions");
  
  Projector::projectFunctionOntoBasis(basisCoefficients, bcFxn, basis, sideBasisCache);
}