#include "BasisSumFunction.h"

void BasisSumFunction::getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints){
  
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  TEST_FOR_EXCEPTION(numCells!=1,std::invalid_argument,"BasisSumFunction only projects on cell at a time (allowing for differing bases per cell)");

  FieldContainer<double> refElemPoints(numCells, numPoints, spaceDim);
  CellTools::mapToReferenceFrame(refElemPoints,physicalPoints,_physicalCellNodes,_basis->getBaseCellTopology());
  refElemPoints.resize(numPoints,spaceDim); // to reduce down - numCells = 1 anyways
  
  int numDofs = basis->getCardinality();

  FieldContainer<double> basisValues(numDofs,numPoints);
  basis->getValues(basisValues, refElemPoints, IntrepidExtendedTypes::OPERATOR_VALUE);
  
  functionValues.resize(numPoints);
  functionValues.initialize(0.0);
  for (int ptIndex=0;ptIndex<numPoints;ptIndex++){
    for (int i=0;i<numDofs;i++){
      functionValues(ptIndex) += basisValues(i,ptIndex)*_coefficients(i);
    }
  }  
}


/*
class BasisSumFunction : public AbstractFunction {
 private:  
  BasisPtr _basis;
  FieldContainer<double> _coefficients;
 public:
  BasisSumFunction(BasisPtr basis, FieldContainer<double> coefficients,cell){
    _coefficients = coefficients;
    _basis = basis;
  }
  virtual void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints);
};
*/
