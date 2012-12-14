#include "BasisSumFunction.h"
#include "Intrepid_CellTools.hpp"

typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;

void BasisSumFunction::getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints){
  
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  TEUCHOS_TEST_FOR_EXCEPTION(numCells!=1,std::invalid_argument,"BasisSumFunction only projects on cell at a time (allowing for differing bases per cell)");

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

void NewBasisSumFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  int numDofs = _basis->getCardinality();
  
  int spaceDim = basisCache->getSpaceDim();
  
  bool basisIsVolumeBasis = true;
  if (spaceDim==2) {
    basisIsVolumeBasis = (_basis->getBaseCellTopology().getBaseKey() != shards::Line<2>::key);
  } else if (spaceDim==3) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim==3 not yet supported in basisIsVolumeBasis determination.");
  }
  
  bool useCubPointsSideRefCell = basisIsVolumeBasis && basisCache->isSideCache();
  
  constFCPtr transformedValues = basisCache->getTransformedValues(_basis, _op, useCubPointsSideRefCell);
  
  // transformedValues has dimensions (C,F,P,[D,D])
  // therefore, the rank of the sum is transformedValues->rank() - 3
  int rank = transformedValues->rank() - 3;
  TEUCHOS_TEST_FOR_EXCEPTION(rank != values.rank()-2, std::invalid_argument, "values rank is incorrect.");
  
  values.initialize(0.0);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int entriesPerPoint = values.size() / (numCells * numPoints);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int i=0;i<numDofs;i++){
      double weight = _coefficients(i);
      for (int ptIndex=0;ptIndex<numPoints;ptIndex++){
        int valueIndex = (cellIndex*numPoints + ptIndex)*entriesPerPoint;
        int basisValueIndex = (cellIndex*numPoints*numDofs + i*numPoints + ptIndex) * entriesPerPoint;
        double *value = &values[valueIndex];
        const double *basisValue = &((*transformedValues)[basisValueIndex]);
        for (int j=0; j<entriesPerPoint; j++) {
          *value++ += *basisValue++ * weight;
        }
      }
    }    
  }
}

FunctionPtr NewBasisSumFunction::x() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component evaluation only supported for NewBasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new NewBasisSumFunction(_basis, _coefficients, OP_X));
}
FunctionPtr NewBasisSumFunction::y() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component evaluation only supported for NewBasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new NewBasisSumFunction(_basis, _coefficients, OP_Y));
}
FunctionPtr NewBasisSumFunction::z() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component evaluation only supported for NewBasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new NewBasisSumFunction(_basis, _coefficients, OP_Z));
}

FunctionPtr NewBasisSumFunction::dx() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "derivatives only supported for NewBasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new NewBasisSumFunction(_basis, _coefficients, OP_DX));
}

FunctionPtr NewBasisSumFunction::dy() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "derivatives only supported for NewBasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new NewBasisSumFunction(_basis, _coefficients, OP_DY));
}

FunctionPtr NewBasisSumFunction::dz() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "derivatives only supported for NewBasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new NewBasisSumFunction(_basis, _coefficients, OP_DZ));
}

