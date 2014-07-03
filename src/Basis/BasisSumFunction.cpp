#include "BasisSumFunction.h"
#include "Intrepid_CellTools.hpp"

typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;

BasisSumFunction::BasisSumFunction(BasisPtr basis, const FieldContainer<double> &basisCoefficients, const FieldContainer<double> &physicalCellNodes){
  _coefficients = basisCoefficients;
  _basis = basis; // note - _basis->getBaseCellTopology
  _physicalCellNodes = physicalCellNodes; // note - rank 3, but dim(0) = 1
  TEUCHOS_TEST_FOR_EXCEPTION(_coefficients.dimension(0)!=basis->getCardinality(),std::invalid_argument,"BasisSumFunction: coefficients passed in do not match cardinality of basis.");
}

void BasisSumFunction::getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  TEUCHOS_TEST_FOR_EXCEPTION(numCells != 1,std::invalid_argument,"BasisSumFunction only projects one cell at a time (allowing for differing bases per cell)");

  FieldContainer<double> refElemPoints(numCells, numPoints, spaceDim);
  typedef CellTools<double>  CellTools;
  CellTools::mapToReferenceFrame(refElemPoints,physicalPoints,_physicalCellNodes,_basis->domainTopology());
  refElemPoints.resize(numPoints,spaceDim); // reshape for the single set of ref cell points
  
  int numDofs = _basis->getCardinality();

  FieldContainer<double> basisValues;
  if (_basis->rangeRank()==0) {
    basisValues = FieldContainer<double>(numDofs,numPoints);
  } else if (_basis->rangeRank()==1) {
    basisValues = FieldContainer<double>(numDofs,numPoints,_basis->rangeDimension());
  } else if (_basis->rangeRank()==2) {
    basisValues = FieldContainer<double>(numDofs,numPoints,_basis->rangeDimension(),_basis->rangeDimension());
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"BasisSumFunction only supports bases with ranks <= 2.");
  }
  _basis->getValues(basisValues, refElemPoints, Intrepid::OPERATOR_VALUE);
  
  Teuchos::Array<int> dim;
  basisValues.dimensions(dim);
  dim[0] = 1; // replace numDofs with numCells (which == 1).
  
  functionValues.resize(dim);
  functionValues.initialize(0.0);
  int cellIndex = 0;
  for (int ptIndex=0;ptIndex<numPoints;ptIndex++) {
    for (int i=0;i<numDofs;i++){
      if (_basis->rangeRank()==0) {
        functionValues(cellIndex,ptIndex) += basisValues(i,ptIndex)*_coefficients(i);
      } else if (_basis->rangeRank()==1) {
        for (int d=0; d<_basis->rangeDimension(); d++) {
          functionValues(cellIndex,ptIndex,d) += basisValues(i,ptIndex,d)*_coefficients(i);
        }
      } else if (_basis->rangeRank()==2) {
        for (int d1=0; d1<_basis->rangeDimension(); d1++) {
          for (int d2=0; d2<_basis->rangeDimension(); d2++) {
            functionValues(cellIndex,ptIndex,d1,d2) += basisValues(i,ptIndex,d1,d2)*_coefficients(i);
          }
        }
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"BasisSumFunction only supports bases with ranks <= 2.");
      }
    }
  }
}

NewBasisSumFunction::NewBasisSumFunction(BasisPtr basis, const FieldContainer<double> &basisCoefficients,
                                         BasisCachePtr overridingBasisCache, EOperatorExtended op, bool boundaryValueOnly) : Function( BasisFactory::getBasisRank(basis) ) {
  // TODO: fix the rank setter here to take into account rank-changing ops (e.g. DIV, GRAD)
  _coefficients = basisCoefficients;
  _overridingBasisCache = overridingBasisCache;
  if (_coefficients.rank()==1) {
    _coefficients.resize(1,_coefficients.dimension(0));
  } else if (_coefficients.rank() != 2) {
    cout << "basisCoefficients must be rank 1 or 2!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "basisCoefficients must be rank 1 or 2");
  }
  _boundaryValueOnly = boundaryValueOnly;
  _basis = basis; // note - _basis->getBaseCellTopology
  _op = op;
  int cardinality = basis->getCardinality();
  TEUCHOS_TEST_FOR_EXCEPTION( _coefficients.dimension(1) != cardinality,
                             std::invalid_argument,
                             "BasisSumFunction: coefficients passed in do not match cardinality of basis.");
}

void NewBasisSumFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  if (_overridingBasisCache.get() != NULL) {
    // we want to transform the physical "cubature" points given by basisCache into reference points on the _overridingBasisCache,
    // set the points there, and then replace basisCache with _overridingBasisCache
    // we implicitly assume that the points given lie inside the physical cell nodes for _overridingBasisCache
    // Note that this transformation does not take curvilinearity into account.
    
    const FieldContainer<double>* physicalCellNodes = &basisCache->getPhysicalCellNodes();
    int numCells = physicalCellNodes->dimension(0);
    int numNodes = physicalCellNodes->dimension(1);
    int spaceDim = physicalCellNodes->dimension(2);
    FieldContainer<double> relativeReferenceCellNodes(numCells, numNodes, spaceDim);
    typedef CellTools<double>  CellTools;
    CellTools::mapToReferenceFrame(relativeReferenceCellNodes,*physicalCellNodes,_overridingBasisCache->getPhysicalCellNodes(),_basis->domainTopology());
    
    FieldContainer<double> oneCellRelativeReferenceNodes(1,numNodes,spaceDim);
    for (int n=0; n<numNodes; n++) {
      for (int d=0; d<spaceDim; d++) {
        oneCellRelativeReferenceNodes(0,n,d) = relativeReferenceCellNodes(0,n,d);
      }
    }
    bool cachesAgreeOnSideness = basisCache->isSideCache() == _overridingBasisCache->isSideCache();
    FieldContainer<double> relativeReferencePoints = cachesAgreeOnSideness ? basisCache->getRefCellPoints() : basisCache->getSideRefCellPointsInVolumeCoordinates();
    FieldContainer<double> refPoints(1,relativeReferencePoints.dimension(0),relativeReferencePoints.dimension(1));
    CellTools::mapToPhysicalFrame(refPoints, relativeReferencePoints, oneCellRelativeReferenceNodes, basisCache->cellTopology());
    refPoints.resize(refPoints.dimension(1),refPoints.dimension(2)); // strip cell dimension
    _overridingBasisCache->setRefCellPoints(refPoints, basisCache->getCubatureWeights());
    basisCache = _overridingBasisCache;
  }
  
  int numDofs = _basis->getCardinality();
  
  int spaceDim = basisCache->getSpaceDim();
  
  bool basisIsVolumeBasis = _basis->domainTopology().getDimension() == spaceDim;
  
  bool useCubPointsSideRefCell = basisIsVolumeBasis && basisCache->isSideCache();
  
  constFCPtr transformedValues = basisCache->getTransformedValues(_basis, _op, useCubPointsSideRefCell);
  
//  cout << "BasisSumFunction: transformedValues:\n" << *transformedValues;
//  cout << "BasisSumFunction: coefficients:\n" << _coefficients;
  
  // transformedValues has dimensions (C,F,P,[D,D])
  // therefore, the rank of the sum is transformedValues->rank() - 3
  int rank = transformedValues->rank() - 3;
  TEUCHOS_TEST_FOR_EXCEPTION(rank != values.rank()-2, std::invalid_argument, "values rank is incorrect.");
  
  values.initialize(0.0);
  bool singleCoefficientVector = _coefficients.dimension(0) == 1;
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int entriesPerPoint = values.size() / (numCells * numPoints);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int i=0;i<numDofs;i++){
      double weight = singleCoefficientVector ? _coefficients(0,i) : _coefficients(cellIndex,i);
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
  return Teuchos::rcp( new NewBasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_X));
}
FunctionPtr NewBasisSumFunction::y() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component evaluation only supported for NewBasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new NewBasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_Y));
}
FunctionPtr NewBasisSumFunction::z() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component evaluation only supported for NewBasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new NewBasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_Z));
}

FunctionPtr NewBasisSumFunction::dx() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "derivatives only supported for NewBasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new NewBasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_DX));
}

FunctionPtr NewBasisSumFunction::dy() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "derivatives only supported for NewBasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new NewBasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_DY));
}

FunctionPtr NewBasisSumFunction::dz() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "derivatives only supported for NewBasisSumFunction with op = OP_VALUE");
  }
  // a bit of a hack: if the topology defined in 3D, then we'll define a derivative there...
  if (_basis->domainTopology().getDimension() > 2) {
    return Teuchos::rcp( new NewBasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_DZ));
  } else {
    return Function::null();
  }
}

bool NewBasisSumFunction::boundaryValueOnly() {
  return _boundaryValueOnly;
}

FunctionPtr NewBasisSumFunction::basisSumFunction(BasisPtr basis, const FieldContainer<double> &basisCoefficients) {
  return Teuchos::rcp( new NewBasisSumFunction(basis,basisCoefficients) );
}
