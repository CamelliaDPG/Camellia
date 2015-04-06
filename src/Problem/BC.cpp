#include "BC.h"
#include "BCFunction.h"

#include "BC.h"
#include "Var.h"
#include "Function.h"

#include "SpatiallyFilteredFunction.h"

#include "PhysicalPointCache.h"

using namespace Intrepid;
using namespace Camellia;

typedef pair< SpatialFilterPtr, FunctionPtr > DirichletBC;

class BCLogicalOrFunction : public Function {
  FunctionPtr _f1, _f2;
  SpatialFilterPtr _sf1, _sf2;
  
public:
  BCLogicalOrFunction(FunctionPtr f1, SpatialFilterPtr sf1, FunctionPtr f2, SpatialFilterPtr sf2) : Function(f1->rank()) {
    _f1 = f1;
    _sf1 = sf1;
    _f2 = f2;
    _sf2 = sf2;
  }
  void setTime(double time)
  {
    _time = time;
    _f1->setTime(time);
    _f2->setTime(time);
  }
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    values.initialize(0.0);
    
    Teuchos::Array<int> dim;
    values.dimensions(dim);
    Teuchos::Array<int> valuesDim = dim;
    FieldContainer<double> f1Values;
    FieldContainer<double> f2Values;
    
    int entriesPerPoint = 1;
    for (int d=2; d<values.rank(); d++) {
      entriesPerPoint *= dim[d];
      dim[d] = 0; // clear so that these indices point to the start of storage for (cellIndex,ptIndex)
    }
    //    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    FieldContainer<bool> pointsMatch1(numCells,numPoints);
    FieldContainer<bool> pointsMatch2(numCells,numPoints);
    
    bool somePointMatches1 = _sf1->matchesPoints(pointsMatch1,basisCache);
    bool somePointMatches2 = _sf2->matchesPoints(pointsMatch2,basisCache);
    
    if ( somePointMatches1 ) {
      f1Values.resize(valuesDim);
      _f1->values(f1Values,basisCache);
    }
    if ( somePointMatches2) {
      f2Values.resize(valuesDim);
      _f2->values(f2Values,basisCache);
    }
    if (somePointMatches1 || somePointMatches2) {
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        dim[0] = cellIndex;
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          dim[1] = ptIndex;
          if ( pointsMatch1(cellIndex,ptIndex) ) {
            if (f1Values.size() == 0) {
              // resize, and compute f1
              f1Values.resize(valuesDim);
              _f1->values(f1Values,basisCache);
            }
            double* value = &values[values.getEnumeration(dim)];
            double* f1Value = &f1Values[f1Values.getEnumeration(dim)];
            for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
              *value++ = *f1Value++;
            }
          } else if ( pointsMatch2(cellIndex,ptIndex) ) {
            if (f2Values.size() == 0) {
              // resize, and compute f2
              f2Values.resize(valuesDim);
              _f2->values(f2Values,basisCache);
            }
            double* value = &values[values.getEnumeration(dim)];
            double* f2Value = &f2Values[f2Values.getEnumeration(dim)];
            for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
              *value++ = *f2Value++;
            }
          }
        }
      }
    }
  }
};

void BC::addDirichlet( VarPtr traceOrFlux, SpatialFilterPtr spatialPoints, FunctionPtr valueFunction ) {
  if ((traceOrFlux->varType() != TRACE) && (traceOrFlux->varType() != FLUX)) {
    cout << "WARNING: adding Dirichlet BC for variable that is neither a trace nor a flux.\n";
  }
  
  if (_dirichletBCs.find( traceOrFlux->ID() ) != _dirichletBCs.end() ) {
    // "or" the existing condition with the new one:
    SpatialFilterPtr existingFilter = _dirichletBCs[ traceOrFlux->ID() ].first;
    FunctionPtr existingFunction = _dirichletBCs[ traceOrFlux->ID() ].second;
    valueFunction = Teuchos::rcp( new BCLogicalOrFunction(existingFunction, existingFilter,
                                                          valueFunction, spatialPoints) );
    spatialPoints = Teuchos::rcp( new SpatialFilterLogicalOr( existingFilter, spatialPoints ) );
    //    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only one Dirichlet condition is allowed per variable.");
  }
  _dirichletBCs[ traceOrFlux->ID() ] = make_pair( spatialPoints, valueFunction );
}

void BC::addZeroMeanConstraint( VarPtr field ) {
  if ( field->varType() != FIELD ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Zero-mean constraints only supported for field vars");
  }
  _zeroMeanConstraints.insert( field->ID() );
}

void BC::removeZeroMeanConstraint( int fieldID ) {
  if (_zeroMeanConstraints.find(fieldID) != _zeroMeanConstraints.end()) {
    _zeroMeanConstraints.erase( _zeroMeanConstraints.find(fieldID) );
  }
}

void BC::addSinglePointBC( int fieldID, double value, GlobalIndexType vertexNumber ) {
  _singlePointBCs[ fieldID ] = make_pair(vertexNumber, value);
}

map< int, DirichletBC > & BC::dirichletBCs() {
  return _dirichletBCs;
}

BCPtr BC::copyImposingZero() {
  //returns a copy of this BC object, except with all zero Functions
  BCPtr zeroBC = Teuchos::rcp( new BC(*this) );
  map< int, DirichletBC >* dirichletBCs = &(zeroBC->dirichletBCs());
  for (map< int, DirichletBC >::iterator bcIt = dirichletBCs->begin();
       bcIt != dirichletBCs->end(); ++bcIt) {
    bcIt->second.second = Function::zero();
  }
  
  for (map< int, pair<GlobalIndexType,double> >::iterator singlePointIt = _singlePointBCs.begin(); singlePointIt != _singlePointBCs.end(); singlePointIt++) {
    int trialID = singlePointIt->first;
    GlobalIndexType vertexNumber = singlePointIt->second.first;
    double zero = 0.0;
    zeroBC->addSinglePointBC(trialID, zero, vertexNumber);
  }
  
  return zeroBC;
}

pair< SpatialFilterPtr, FunctionPtr > BC::getDirichletBC(int varID) {
  if (_dirichletBCs.find(varID) == _dirichletBCs.end()) {
    cout << "No Dirichlet BC for the indicated variable...\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No Dirichlet BC for the indicated variable...");
  }
  return _dirichletBCs[varID];
}

FunctionPtr BC::getSpatiallyFilteredFunctionForDirichletBC(int varID) {
  pair< SpatialFilterPtr, FunctionPtr > dirichletBC = getDirichletBC(varID);
  
  return Teuchos::rcp( new SpatiallyFilteredFunction(dirichletBC.second, dirichletBC.first) );
}

bool BC::isLegacySubclass() {
  return _legacyBCSubclass;
}

void BC::setTime(double time)
{
  _time = time;
  for (map< int, DirichletBC >::iterator bcIt = dirichletBCs().begin();
       bcIt != dirichletBCs().end(); ++bcIt)
  {
    bcIt->second.second->setTime(time);
  }
}

bool BC::bcsImposed(int varID) {
  // returns true if there are any BCs anywhere imposed on varID
  if (_legacyBCSubclass)
  {
    cout << "legacy BC subclasses must override bcsImposed().\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "legacy BC subclasses must override bcsImposed().");
    return false; // unreachable
  }
  else
  {
    return _dirichletBCs.find(varID) != _dirichletBCs.end();
  }
}

void BC::imposeBC(FieldContainer<double> &dirichletValues, FieldContainer<bool> &imposeHere, 
                      int varID, FieldContainer<double> &unitNormals, BasisCachePtr basisCache) {
  if (_legacyBCSubclass)
  {
    // by default, call legacy version:
    // (basisCache->getPhysicalCubaturePoints() doesn't really return *cubature* points, but the boundary points
    //  that we're interested in)
    FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
    imposeBC(varID,physicalPoints,unitNormals,dirichletValues,imposeHere);
  }
  else
  {
    FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
    
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    
    TEUCHOS_TEST_FOR_EXCEPTION( ( dirichletValues.dimension(0) != numCells )
                               || ( dirichletValues.dimension(1) != numPoints )
                               || ( dirichletValues.rank() != 2  ),
                               std::invalid_argument,
                               "dirichletValues dimensions should be (numCells,numPoints).");
    TEUCHOS_TEST_FOR_EXCEPTION( ( imposeHere.dimension(0) != numCells )
                               || ( imposeHere.dimension(1) != numPoints )
                               || ( imposeHere.rank() != 2  ),
                               std::invalid_argument,
                               "imposeHere dimensions should be (numCells,numPoints).");
    
    TEUCHOS_TEST_FOR_EXCEPTION( spaceDim > 3, std::invalid_argument,
                               "spaceDim > 3 not yet supported by imposeBC." );
    
    imposeHere.initialize(false);
    if ( _dirichletBCs.find(varID) == _dirichletBCs.end() ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Attempt to impose BC on varID without BCs.");
    }
    
    DirichletBC bc = _dirichletBCs[varID];
    SpatialFilterPtr filter = bc.first;
    FunctionPtr f = bc.second;
    
    filter->matchesPoints(imposeHere,basisCache);
    
    f->values(dirichletValues,basisCache);
  }
}

void BC::imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                      FieldContainer<double> &unitNormals,
                      FieldContainer<double> &dirichletValues,
                      FieldContainer<bool> &imposeHere) {
  if (_legacyBCSubclass)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BC::imposeBC unimplemented.");
  }
  else
  {
    cout << "ERROR: this version of imposeBC (the singleton version) is only supported by legacy subclasses.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "this version of imposeBC (the singleton version) is only supported by legacy subclasses.");
//    if (_singlePointBCs.find(varID) == _singlePointBCs.end()) {
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "this version of BC::imposeBC only supports singleton points.");
//    }
//    DirichletBC bc = _singlePointBCs[varID];
//    SpatialFilterPtr filter = bc.first;
//    FunctionPtr f = bc.second;
//    
//    
//    
//    BasisCachePtr basisCache = Teuchos::rcp( new PhysicalPointCache(physicalPoints) );
//    
//    filter->matchesPoints(imposeHere,basisCache);
//    f->values(dirichletValues,basisCache);
//    
////    cout << "BC::imposeBC (singleton BC implementation) called for varID " << varID << endl;
//    
//    bool pointMatched = false; // make sure we just impose this once
//    for (int i=0; i<imposeHere.size(); i++) {
//      if (imposeHere[i]) {
//        if (pointMatched) {
//          // then don't impose here
//          imposeHere[i] = false;
//        } else {
//          pointMatched = true;
//        }
//      }
//    }
  }
}

bool BC::singlePointBC(int varID) { // override if you want to implement a BC at a single, arbitrary point (and nowhere else).
  if (_legacyBCSubclass)
  {
    return false;
  }
  else
  {
    return _singlePointBCs.find(varID) != _singlePointBCs.end();
  }
}

bool BC::imposeZeroMeanConstraint(int varID) {
  if (_legacyBCSubclass)
  {
    return false;
  }
  else
  {
    return _zeroMeanConstraints.find(varID) != _zeroMeanConstraints.end();
  }
}

// basisCoefficients has dimensions (C,F)
void BC::coefficientsForBC(FieldContainer<double> &basisCoefficients, Teuchos::RCP<BCFunction> bcFxn, 
                           BasisPtr basis, BasisCachePtr sideBasisCache) {
  int numFields = basis->getCardinality();
  TEUCHOS_TEST_FOR_EXCEPTION( basisCoefficients.dimension(1) != numFields, std::invalid_argument, "inconsistent basisCoefficients dimensions");

  Projector::projectFunctionOntoBasisInterpolating(basisCoefficients, bcFxn, basis, sideBasisCache);
  
//  if (!bcFxn->isTrace()) {
//    // L^2 projection
//    Projector::projectFunctionOntoBasis(basisCoefficients, bcFxn, basis, sideBasisCache);
//  } else {
//    // TODO: projection-based interpolation
//    // (start with L^2-projection-based interpolation; proceed to H^1 once we have a clear story on
//    //  how to take derivatives of BCFunction)
//    Projector::projectFunctionOntoBasis(basisCoefficients, bcFxn, basis, sideBasisCache);
//  }
}

void BC::removeSinglePointBC(int fieldID) {
  if (_singlePointBCs.find(fieldID) != _singlePointBCs.end()) {
    _singlePointBCs.erase(fieldID);
  }
}

double BC::valueForSinglePointBC(int varID) {
  if (_singlePointBCs.find(varID) != _singlePointBCs.end())
    return _singlePointBCs[varID].second;
  else
    return -1;
}


GlobalIndexType BC::vertexForSinglePointBC(int varID) {
  if (_singlePointBCs.find(varID) != _singlePointBCs.end())
    return _singlePointBCs[varID].first;
  else
    return -1;
}

set<int> BC::getZeroMeanConstraints() {
  return _zeroMeanConstraints;
}

BCPtr BC::bc() {
  return Teuchos::rcp(new BC(false)); // false: not legacy subclass
}