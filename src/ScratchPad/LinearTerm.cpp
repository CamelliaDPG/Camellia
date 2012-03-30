//
//  LinearTerm.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "LinearTerm.h"

typedef pair< FunctionPtr, VarPtr > LinearSummand;

const vector< LinearSummand > & LinearTerm::summands() const { return _summands; }
LinearTerm::LinearTerm(FunctionPtr weight, VarPtr var) {
  _rank = -1;
  _termType = UNKNOWN_TYPE;
  addVar(weight,var);
}
LinearTerm::LinearTerm(double weight, VarPtr var) {
  _rank = -1;
  _termType = UNKNOWN_TYPE;
  addVar(weight,var);
}
LinearTerm::LinearTerm(vector<double> weight, VarPtr var) {
  _rank = -1;
  _termType = UNKNOWN_TYPE;
  addVar(weight,var);
}
LinearTerm::LinearTerm( VarPtr v ) {
  _rank = -1;
  _termType = UNKNOWN_TYPE;
  addVar( 1.0, v);
}
// copy constructor:
LinearTerm::LinearTerm( const LinearTerm &a ) {
  _rank = a.rank();
  _termType = a.termType();
  _summands = a.summands();
}

void LinearTerm::addVar(FunctionPtr weight, VarPtr var) {
  // check ranks:
  int rank; // rank of weight * var
  if (weight->rank() == var->rank() ) { // then we dot like terms together, getting a scalar
    rank = 0;
  } else if ( weight->rank() == 0 || var->rank() == 0) { // then we multiply each term by scalar
    rank = (weight->rank() == 0) ? var->rank() : weight->rank(); // rank is the non-zero one
  } else {
    TEST_FOR_EXCEPTION( true, std::invalid_argument, "Unhandled rank combination.");
  }
  if (_rank == -1) { // LinearTerm's rank is unassigned
    _rank = rank;
  }
  if (_rank != rank) {
    TEST_FOR_EXCEPTION( true, std::invalid_argument, "Attempting to add terms of unlike rank." );
  }
  // check type:
  if (_termType == UNKNOWN_TYPE) {
    _termType = var->varType();
  }
  if (_termType != var->varType() ) {
    TEST_FOR_EXCEPTION( true, std::invalid_argument, "Attempting to add terms of differing type." );
  }
  _summands.push_back( make_pair( weight, var ) );
  _varIDs.insert(var->ID());
}

void LinearTerm::addVar(double weight, VarPtr var) {
  FunctionPtr weightFn = Teuchos::rcp( new ConstantScalarFunction(weight) );
  addVar( weightFn, var );
}
void LinearTerm::addVar(vector<double> vector_weight, VarPtr var) { // dots weight vector with vector var, makes a vector out of a scalar var
  FunctionPtr weightFn = Teuchos::rcp( new ConstantVectorFunction(vector_weight) );
  addVar( weightFn, var );
}

const set<int> & LinearTerm::varIDs() const {
  return _varIDs;
}

VarType LinearTerm::termType() const { return _termType; }
//  vector< EOperatorExtended > varOps(int varID);

// compute the value of linearTerm for non-zero varID at the cubature points, for each basis function in basis
// values shape: (C,F,P), (C,F,P,D), or (C,F,P,D,D)
void LinearTerm::values(FieldContainer<double> &values, int varID, BasisPtr basis, BasisCachePtr basisCache, 
            bool applyCubatureWeights, int sideIndex) {
  // can speed things up a lot by handling specially constant weights and 1.0 weights
  // (would need to move this logic into the Function class, and then ConstantFunction can
  //  override to provide the speedup)
  bool boundaryTerm = (sideIndex != -1);
  
  int valuesRankExpected = _rank + 3; // 3 for scalar, 4 for vector, etc.
  TEST_FOR_EXCEPTION( valuesRankExpected != values.rank(), std::invalid_argument,
                     "values FC does not have the expected rank" );
  int numCells = values.dimension(0);
  int numFields = values.dimension(1);
  int numPoints = values.dimension(2);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  values.initialize(0.0);
  Teuchos::Array<int> scalarFunctionValueDim;
  scalarFunctionValueDim.append(numCells);
  scalarFunctionValueDim.append(numPoints);
  
  // (could tune things by pre-allocating this storage)
  FieldContainer<double> fValues;
  
  Teuchos::Array<int> vectorFunctionValueDim = scalarFunctionValueDim;
  vectorFunctionValueDim.append(spaceDim);
  Teuchos::Array<int> tensorFunctionValueDim = vectorFunctionValueDim;
  tensorFunctionValueDim.append(spaceDim);
  
  TEST_FOR_EXCEPTION( numCells != basisCache->getPhysicalCubaturePoints().dimension(0),
                     std::invalid_argument, "values FC numCells disagrees with cubature points container");
  TEST_FOR_EXCEPTION( numFields != basis->getCardinality(),
                     std::invalid_argument, "values FC numFields disagrees with basis cardinality");
  if (! boundaryTerm) {
    TEST_FOR_EXCEPTION( numPoints != basisCache->getPhysicalCubaturePoints().dimension(1),
                       std::invalid_argument, "values FC numPoints disagrees with cubature points container");
  } else {
    TEST_FOR_EXCEPTION( numPoints != basisCache->getPhysicalCubaturePointsForSide(sideIndex).dimension(1),
                       std::invalid_argument, "values FC numPoints disagrees with cubature points container");      
  }
  for (vector< LinearSummand >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
    LinearSummand ls = *lsIt;
    if (ls.second->ID() == varID) {        
      if (ls.first->rank() == 0) {
        fValues.resize(scalarFunctionValueDim);
      } else if (ls.first->rank() == 1) {
        fValues.resize(vectorFunctionValueDim);
      } else if (ls.first->rank() == 2) {
        fValues.resize(tensorFunctionValueDim);
      } else {
        Teuchos::Array<int> fDim = tensorFunctionValueDim;
        for (int d=3; d < ls.first->rank(); d++) {
          fDim.append(spaceDim);
        }
        fValues.resize(fDim);
      }
      
      ls.first->values(fValues,basisCache);
      constFCPtr basisValues;
      if (applyCubatureWeights) {
        if (sideIndex == -1) {
          basisValues = basisCache->getTransformedWeightedValues(basis, ls.second->op());
        } else {
          // on sides, we use volume coords for test values
          bool useVolumeCoords = ls.second->varType() == TEST;
          basisValues = basisCache->getTransformedWeightedValues(basis, ls.second->op(), sideIndex, useVolumeCoords);
        }
      } else {
        if (sideIndex == -1) {
          basisValues = basisCache->getTransformedValues(basis, ls.second->op());
        } else {
          // on sides, we use volume coords for test values
          bool useVolumeCoords = ls.second->varType() == TEST;
          basisValues = basisCache->getTransformedValues(basis, ls.second->op(), sideIndex, useVolumeCoords);
        }
      }
      int numFields = basis->getCardinality();
      
      Teuchos::Array<int> fDim(fValues.rank());
      Teuchos::Array<int> bDim(basisValues->rank());
      
      // compute f * basisValues
      if ( ls.first->rank() == ls.second->rank() ) { // scalar result
        int entriesPerPoint = 1;
        int fRank = ls.first->rank();
        for (int d=0; d<fRank; d++) {
          entriesPerPoint *= spaceDim;
        }
        for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
          fDim[0] = cellIndex; bDim[0] = cellIndex;
          for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
            fDim[1] = ptIndex; bDim[2] = ptIndex;
            for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
              const double *fValue = &fValues[fValues.getEnumeration(fDim)];
              bDim[1] = fieldIndex;
              const double *bValue = &((*basisValues)[basisValues->getEnumeration(bDim)]);
              for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
                values(cellIndex,fieldIndex,ptIndex) += *fValue * *bValue;
                
                fValue++;
                bValue++;
              }
            }
          }
        }
      } else { // vector/tensor result
        // could pretty easily fold the scalar case above into the code below
        // (just change the logic in the pointer increments)
        int entriesPerPoint = 1;
        bool scalarF = ls.first->rank() == 0;
        int resultRank = scalarF ? ls.second->rank() : ls.first->rank();
        for (int d=0; d<resultRank; d++) {
          entriesPerPoint *= spaceDim;
        }
        Teuchos::Array<int> vDim( values.rank() );
        for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
          fDim[0] = cellIndex; bDim[0] = cellIndex; vDim[0] = cellIndex;
          for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
            fDim[1] = ptIndex; bDim[2] = ptIndex; vDim[2] = ptIndex;
            for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
              const double *fValue = &fValues[fValues.getEnumeration(fDim)];
              bDim[1] = fieldIndex; vDim[1] = fieldIndex;
              const double *bValue = &(*basisValues)[basisValues->getEnumeration(bDim)];
              
              double *value = &values[values.getEnumeration(vDim)];
              
              for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
                *value += *fValue * *bValue;
                value++; 
                if (scalarF) {
                  bValue++;
                } else {
                  fValue++;
                }
              }
            }
          }
        }
      }
    }
  }
}

int LinearTerm::rank() const { return _rank; }  // 0 for scalar, 1 for vector, etc.

// operator overloading niceties:

LinearTerm& LinearTerm::operator=(const LinearTerm &rhs) {
  if ( this == &rhs ) {
    return *this;
  }
  _rank = rhs.rank();
  _summands = rhs.summands();
  _varIDs = rhs.varIDs();
  return *this;
}    
LinearTerm& LinearTerm::operator+=(const LinearTerm &rhs) {
  if (_rank == -1) { // we're empty -- adopt rhs's rank
    _rank = rhs.rank();
  }
  if (_rank != rhs.rank()) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "attempting to add terms of unlike rank.");
  }
  if (_termType == UNKNOWN_TYPE) {
    _termType = rhs.termType();
  }
  if (_termType != rhs.termType()) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "attempting to add terms of unlike type.");
  }
  _summands.insert(_summands.end(), rhs.summands().begin(), rhs.summands().end());
  _varIDs.insert( rhs.varIDs().begin(), rhs.varIDs().end() );
  return *this;
}

LinearTerm& LinearTerm::operator+=(VarPtr v) {
  this->addVar(1.0, v);
  return *this;
}

// operator overloading for syntax sugar:
LinearTermPtr operator+(LinearTermPtr a1, LinearTermPtr a2) {
  LinearTermPtr sum = Teuchos::rcp( new LinearTerm(*a1) );
  //  cout << "sum->rank(): " << sum->rank() << endl;
  //  cout << "sum->summands.size() before adding a2: " << sum->summands().size() << endl;
  *sum += *a2;
  //  cout << "sum->summands.size() after adding a2: " << sum->summands().size() << endl;
  return sum;
}

LinearTermPtr operator+(VarPtr v, LinearTermPtr a) {
  LinearTermPtr sum = Teuchos::rcp( new LinearTerm(*a) );
  *sum += v;
  return sum;
}

LinearTermPtr operator+(LinearTermPtr a, VarPtr v) {
  return v + a;
}

LinearTermPtr operator*(FunctionPtr f, VarPtr v) {
  return Teuchos::rcp( new LinearTerm(f, v) );
}

LinearTermPtr operator*(double weight, VarPtr v) {
  return Teuchos::rcp( new LinearTerm(weight, v) );
}

LinearTermPtr operator*(VarPtr v, double weight) {
  return weight * v;
}

LinearTermPtr operator*(vector<double> weight, VarPtr v) {
  return Teuchos::rcp( new LinearTerm(weight, v) );
}

LinearTermPtr operator*(VarPtr v, vector<double> weight) {
  return weight * v;
}

LinearTermPtr operator+(VarPtr v1, VarPtr v2) {
  return 1.0 * v1 + 1.0 * v2;
}

LinearTermPtr operator/(VarPtr v, double weight) {
  return (1.0 / weight) * v;
}

LinearTermPtr operator-(VarPtr v1, VarPtr v2) {
  return v1 + (-1.0) * v2;
}

LinearTermPtr operator-(VarPtr v) {
  return (-1.0) * v;
}

LinearTermPtr operator-(LinearTermPtr a, VarPtr v) {
  return a + (-1.0) * v;
}
