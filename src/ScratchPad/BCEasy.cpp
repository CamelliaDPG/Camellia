//
//  BCEasy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "BCEasy.h"
#include "Var.h"
#include "Function.h"

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

void BCEasy::addDirichlet( VarPtr traceOrFlux, SpatialFilterPtr spatialPoints, FunctionPtr valueFunction ) {
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

void BCEasy::addZeroMeanConstraint( VarPtr field ) {
  if ( field->varType() != FIELD ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Zero-mean constraints only supported for field vars");
  }
  _zeroMeanConstraints.insert( field->ID() );
}

void BCEasy::removeZeroMeanConstraint( int fieldID ) {
  if (_zeroMeanConstraints.find(fieldID) != _zeroMeanConstraints.end()) {
    _zeroMeanConstraints.erase( _zeroMeanConstraints.find(fieldID) );
  }
}
  
void BCEasy::addSinglePointBC( int fieldID, FunctionPtr valueFunction, SpatialFilterPtr spatialPoints ) {
  DirichletBC bc = make_pair(spatialPoints, valueFunction);
  _singlePointBCs[ fieldID ] = bc;
}

bool BCEasy::bcsImposed(int varID) { // returns true if there are any BCs anywhere imposed on varID
  return _dirichletBCs.find(varID) != _dirichletBCs.end();
}

map< int, DirichletBC > & BCEasy::dirichletBCs() {
  return _dirichletBCs;
}

void BCEasy::imposeBC(FieldContainer<double> &dirichletValues, FieldContainer<bool> &imposeHere, 
                      int varID, FieldContainer<double> &unitNormals, BasisCachePtr basisCache) {
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
  
  TEUCHOS_TEST_FOR_EXCEPTION( spaceDim != 2, std::invalid_argument,
                     "spaceDim != 2 not yet supported by imposeBC." );
  
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

void BCEasy::imposeBC(int varID, FieldContainer<double> &physicalPoints,
                      FieldContainer<double> &unitNormals,
                      FieldContainer<double> &dirichletValues,
                      FieldContainer<bool> &imposeHere) {
  if (_singlePointBCs.find(varID) == _singlePointBCs.end()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BCEasy::imposeBC only supports singleton points.");
  }
  DirichletBC bc = _singlePointBCs[varID];
  SpatialFilterPtr filter = bc.first;
  FunctionPtr f = bc.second;
  
  BasisCachePtr basisCache = Teuchos::rcp( new PhysicalPointCache(physicalPoints) );
  
  filter->matchesPoints(imposeHere,basisCache);
  f->values(dirichletValues,basisCache);
  
  cout << "BCEasy::imposeBC (singleton BC implementation) called for varID " << varID << endl;
  
  bool pointMatched = false; // make sure we just impose this once
  for (int i=0; i<imposeHere.size(); i++) {
    if (imposeHere[i]) {
      if (pointMatched) {
        // then don't impose here
        imposeHere[i] = false;
      } else {
        pointMatched = true;
      }
    }
  }
}

bool BCEasy::singlePointBC(int varID) {
  // for now, these are unsupported
//  return false;
  return _singlePointBCs.find(varID) != _singlePointBCs.end();
} 

bool BCEasy::imposeZeroMeanConstraint(int varID) {
  return _zeroMeanConstraints.find(varID) != _zeroMeanConstraints.end();
}

Teuchos::RCP<BCEasy> BCEasy::copyImposingZero() {
  //returns a copy of this BC object, except with all zero Functions
  Teuchos::RCP<BCEasy> zeroBC = Teuchos::rcp( new BCEasy(*this) );
  map< int, DirichletBC >* dirichletBCs = &(zeroBC->dirichletBCs());
  for (map< int, DirichletBC >::iterator bcIt = dirichletBCs->begin();
       bcIt != dirichletBCs->end(); bcIt++) {
    bcIt->second.second = Function::zero();
  }
  
  return zeroBC;
}