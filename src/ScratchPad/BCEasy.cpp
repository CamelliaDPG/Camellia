//
//  BCEasy.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "BCEasy.h"
#include "Var.h"

void BCEasy::addDirichlet( VarPtr traceOrFlux, SpatialFilterPtr spatialPoints, FunctionPtr valueFunction ) {
  _dirichletBCs[ traceOrFlux->ID() ] = make_pair( spatialPoints, valueFunction );
}

void BCEasy::addZeroMeanConstraint( VarPtr field ) {
  _zeroMeanConstraints.insert( field->ID() );
}

//  addSinglePointBC( VarPtr field ) {
//    _singlePointBCs.insert( field->ID() );
//  }

bool BCEasy::bcsImposed(int varID) { // returns true if there are any BCs anywhere imposed on varID
  return _dirichletBCs.find(varID) != _dirichletBCs.end();
}

void BCEasy::imposeBC(FieldContainer<double> &dirichletValues, FieldContainer<bool> &imposeHere, 
              int varID, FieldContainer<double> &unitNormals, BasisCachePtr basisCache) {
  FieldContainer<double> physicalPoints = basisCache->getPhysicalCubaturePoints();
  
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  
  TEST_FOR_EXCEPTION( ( dirichletValues.dimension(0) != numCells ) 
                     || ( dirichletValues.dimension(1) != numPoints ) 
                     || ( dirichletValues.rank() != 2  ),
                     std::invalid_argument,
                     "dirichletValues dimensions should be (numCells,numPoints).");
  TEST_FOR_EXCEPTION( ( imposeHere.dimension(0) != numCells ) 
                     || ( imposeHere.dimension(1) != numPoints ) 
                     || ( imposeHere.rank() != 2  ),
                     std::invalid_argument,
                     "imposeHere dimensions should be (numCells,numPoints).");
  
  TEST_FOR_EXCEPTION( spaceDim != 2, std::invalid_argument,
                     "spaceDim != 2 not yet supported by imposeBC." );
  
  imposeHere.initialize(false);
  // TODO: add exceptions for varIDs that aren't supposed to have BCs imposed...
  
  SpatialFilterPtr filter = _dirichletBCs[varID].first;
  FunctionPtr f = _dirichletBCs[varID].second;
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double x = physicalPoints(cellIndex, ptIndex, 0);
      double y = physicalPoints(cellIndex, ptIndex, 1);
      
      imposeHere(cellIndex,ptIndex) = filter->matchesPoint(x,y);
    }
  }
  
  f->values(dirichletValues,basisCache);
}

bool BCEasy::singlePointBC(int varID) {
  // for now, these are unsupported
  return false;
  //    return _singlePointBCs.find(varID) != _singlePointBCs.end();
} 

bool BCEasy::imposeZeroMeanConstraint(int varID) {
  return _zeroMeanConstraints.find(varID) != _zeroMeanConstraints.end();
}