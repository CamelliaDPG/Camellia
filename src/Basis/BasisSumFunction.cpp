#include "BasisSumFunction.h"
#include "Intrepid_CellTools.hpp"

#include "BasisFactory.h"

typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;

BasisSumFunction::BasisSumFunction(BasisPtr basis, const FieldContainer<double> &basisCoefficients,
                                         BasisCachePtr overridingBasisCache, Camellia::EOperator op, bool boundaryValueOnly) : Function( BasisFactory::basisFactory()->getBasisRank(basis) ) {
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

void BasisSumFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
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
    CellTopoPtr domainTopo = _basis->domainTopology();
    if (domainTopo->getTensorialDegree() == 0) {
      CellTools::mapToReferenceFrame(relativeReferenceCellNodes,*physicalCellNodes,_overridingBasisCache->getPhysicalCellNodes(),_basis->domainTopology()->getShardsTopology());
    } else {
      cout << "ERROR: BasisSumFunction's mapToReferenceFrame doesn't yet support tensorial degree > 0.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisSumFunction's mapToReferenceFrame doesn't yet support tensorial degree > 0.");
    }

    FieldContainer<double> oneCellRelativeReferenceNodes(1,numNodes,spaceDim);
    for (int n=0; n<numNodes; n++) {
      for (int d=0; d<spaceDim; d++) {
        oneCellRelativeReferenceNodes(0,n,d) = relativeReferenceCellNodes(0,n,d);
      }
    }
    bool cachesAgreeOnSideness = basisCache->isSideCache() == _overridingBasisCache->isSideCache();
    FieldContainer<double> relativeReferencePoints = cachesAgreeOnSideness ? basisCache->getRefCellPoints() : basisCache->getSideRefCellPointsInVolumeCoordinates();
    FieldContainer<double> refPoints(1,relativeReferencePoints.dimension(0),relativeReferencePoints.dimension(1));
    if (domainTopo->getTensorialDegree() == 0) {
      CellTools::mapToPhysicalFrame(refPoints, relativeReferencePoints, oneCellRelativeReferenceNodes, basisCache->cellTopology()->getShardsTopology());
    } else {
      cout << "ERROR: BasisSumFunction's mapToReferenceFrame doesn't yet support tensorial degree > 0.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisSumFunction's mapToReferenceFrame doesn't yet support tensorial degree > 0.");
    }
    refPoints.resize(refPoints.dimension(1),refPoints.dimension(2)); // strip cell dimension
    _overridingBasisCache->setRefCellPoints(refPoints, basisCache->getCubatureWeights());
    basisCache = _overridingBasisCache;
  }
  
  int numDofs = _basis->getCardinality();
  
  int spaceDim = basisCache->getSpaceDim();
  
  bool basisIsVolumeBasis = _basis->domainTopology()->getDimension() == spaceDim;
  
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

FunctionPtr BasisSumFunction::x() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component evaluation only supported for BasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new BasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_X));
}
FunctionPtr BasisSumFunction::y() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component evaluation only supported for BasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new BasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_Y));
}
FunctionPtr BasisSumFunction::z() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component evaluation only supported for BasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new BasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_Z));
}

FunctionPtr BasisSumFunction::dx() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "derivatives only supported for BasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new BasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_DX));
}

FunctionPtr BasisSumFunction::dy() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "derivatives only supported for BasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new BasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_DY));
}

FunctionPtr BasisSumFunction::dz() {
  if (_op != OP_VALUE) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "derivatives only supported for BasisSumFunction with op = OP_VALUE");
  }
  // a bit of a hack: if the topology defined in 3D, then we'll define a derivative there...
  if (_basis->domainTopology()->getDimension() > 2) {
    return Teuchos::rcp( new BasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_DZ));
  } else {
    return Function::null();
  }
}

bool BasisSumFunction::boundaryValueOnly() {
  return _boundaryValueOnly;
}

FunctionPtr BasisSumFunction::basisSumFunction(BasisPtr basis, const FieldContainer<double> &basisCoefficients) {
  return Teuchos::rcp( new BasisSumFunction(basis,basisCoefficients) );
}
