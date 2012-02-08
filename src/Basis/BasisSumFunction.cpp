#include "BasisSumFunction.h"
#include "Intrepid_CellTools.hpp"

void BasisSumFunction::getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints){
  
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  TEST_FOR_EXCEPTION(numCells!=1,std::invalid_argument,"BasisSumFunction only projects on cell at a time (allowing for differing bases per cell)");

  FieldContainer<double> refElemPoints(numCells, numPoints, spaceDim);
  typedef CellTools<double>  CellTools;
  CellTools::mapToReferenceFrame(refElemPoints,physicalPoints,_physicalCellNodes,_basis->getBaseCellTopology());
  refElemPoints.resize(numPoints,spaceDim); // to reduce down - numCells = 1 anyways
  
  int numDofs = _basis->getCardinality();

  FieldContainer<double> basisValues(numDofs,numPoints);
  _basis->getValues(basisValues, refElemPoints, Intrepid::OPERATOR_VALUE);
  
  functionValues.resize(numCells,numPoints);
  functionValues.initialize(0.0);
  int cellIndex = 0;
  for (int ptIndex=0;ptIndex<numPoints;ptIndex++){
    for (int i=0;i<numDofs;i++){
      functionValues(cellIndex,ptIndex) += basisValues(i,ptIndex)*_coefficients(i);
    }
  }  
}
