//
//  Function.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/5/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "Function.h"
#include "BasisCache.h"

class Function;
typedef Teuchos::RCP<Function> FunctionPtr;

Function::Function() {
  _rank = 0;
}
Function::Function(int rank) { 
  _rank = rank; 
}

int Function::rank() { 
  return _rank; 
}

void Function::CHECK_VALUES_RANK(FieldContainer<double> &values) { // throws exception on bad values rank
  // values should have shape (C,P,D,D,D,...) where the # of D's = _rank
  TEST_FOR_EXCEPTION( values.rank() != _rank + 2, std::invalid_argument, "values has incorrect rank." );
}


void Function::addToValues(FieldContainer<double> &valuesToAddTo, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(valuesToAddTo);
  Teuchos::Array<int> dim;
  valuesToAddTo.dimensions(dim);
  FieldContainer<double> myValues(dim);
  this->values(myValues,basisCache);
  for (int i=0; i<myValues.size(); i++) {
    valuesToAddTo[i] += myValues[i];
  }
}

// divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
void Function::scalarMultiplyFunctionValues(FieldContainer<double> &functionValues, BasisCachePtr basisCache) {
  // functionValues has dimensions (C,P,...)
  scalarModifyFunctionValues(functionValues,basisCache,MULTIPLY);
}

// divide values by this function (supported only when this is a scalar)
void Function::scalarDivideFunctionValues(FieldContainer<double> &functionValues, BasisCachePtr basisCache) {
  // functionValues has dimensions (C,P,...)
  scalarModifyFunctionValues(functionValues,basisCache,DIVIDE);
}

// divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
void Function::scalarMultiplyBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
  // basisValues has dimensions (C,F,P,...)
//  cout << "scalarMultiplyBasisValues: basisValues:\n" << basisValues;
  scalarModifyBasisValues(basisValues,basisCache,MULTIPLY);
//  cout << "scalarMultiplyBasisValues: modified basisValues:\n" << basisValues;
}

// divide values by this function (supported only when this is a scalar)
void Function::scalarDivideBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
  // basisValues has dimensions (C,F,P,...)
  scalarModifyBasisValues(basisValues,basisCache,DIVIDE);
}

// note that valuesDottedWithTensor isn't called by anything right now
// (it's totally untried!! -- trying for first time with NewBurgersDriver, in RHS)
void Function::valuesDottedWithTensor(FieldContainer<double> &values, 
                                      FunctionPtr tensorFunctionOfLikeRank, 
                                      BasisCachePtr basisCache) {
  TEST_FOR_EXCEPTION( _rank != tensorFunctionOfLikeRank->rank(),std::invalid_argument,
                     "Can't dot functions of unlike rank");
  TEST_FOR_EXCEPTION( values.rank() != 2, std::invalid_argument,
                     "values container should have size (numCells, numPoints" );
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  values.initialize(0.0);
  
  Teuchos::Array<int> tensorValueIndex(_rank+2); // +2 for numCells, numPoints indices
  tensorValueIndex[0] = numCells;
  tensorValueIndex[1] = numPoints;
  for (int d=0; d<_rank; d++) {
    tensorValueIndex[d+2] = spaceDim;
  }
  
  FieldContainer<double> myTensorValues(tensorValueIndex);
  this->values(myTensorValues,basisCache);
  FieldContainer<double> otherTensorValues(tensorValueIndex);
  tensorFunctionOfLikeRank->values(otherTensorValues,basisCache);
  
  // clear out the spatial indices of tensorValueIndex so we can use it as index
  for (int d=0; d<_rank; d++) {
    tensorValueIndex[d+2] = 0;
  }
  
  int entriesPerPoint = 1;
  for (int d=0; d<_rank; d++) {
    entriesPerPoint *= spaceDim;
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    tensorValueIndex[0] = cellIndex;
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      tensorValueIndex[1] = ptIndex;
      double *myValue = &myTensorValues[ myTensorValues.getEnumeration(tensorValueIndex) ];
      double *otherValue = &otherTensorValues[ otherTensorValues.getEnumeration(tensorValueIndex) ];
      double *value = &values(cellIndex,ptIndex);
      
      for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
        *value += *myValue * *otherValue;
        myValue++; 
        otherValue++;
      }
    }
  }
}

void Function::scalarModifyFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache,
                                          FunctionModificationType modType) {
  TEST_FOR_EXCEPTION( rank() != 0, std::invalid_argument, "scalarModifyFunctionValues only supported for scalar functions" );
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  FieldContainer<double> scalarValues(numCells,numPoints);
  this->values(scalarValues,basisCache);
  
  Teuchos::Array<int> valueIndex(values.rank());
  
  int entriesPerPoint = 1;
  for (int d=0; d < values.rank()-2; d++) {  // -2 for numCells, numPoints indices
    entriesPerPoint *= spaceDim;
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    valueIndex[0] = cellIndex;
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      valueIndex[1] = ptIndex;
      double *value = &values[ values.getEnumeration(valueIndex) ];
      double scalarValue = scalarValues(cellIndex,ptIndex);
      for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
        if (modType == MULTIPLY) {
          *value++ *= scalarValue;
        } else if (modType == DIVIDE) {
          *value++ /= scalarValue;
        }
      }
    }
  }
}

void Function::scalarModifyBasisValues(FieldContainer<double> &values, BasisCachePtr basisCache,
                                       FunctionModificationType modType) {
  TEST_FOR_EXCEPTION( rank() != 0, std::invalid_argument, "scalarModifyBasisValues only supported for scalar functions" );
  int numCells = values.dimension(0);
  int numFields = values.dimension(1);
  int numPoints = values.dimension(2);
  
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  FieldContainer<double> scalarValues(numCells,numPoints);
  this->values(scalarValues,basisCache);
  
//  cout << "scalarModifyBasisValues: scalarValues:\n" << scalarValues;
  
  Teuchos::Array<int> valueIndex(values.rank());
  
  int entriesPerPoint = 1;
  for (int d=0; d<values.rank()-3; d++) {  // -3 for numCells, numFields, numPoints indices
    entriesPerPoint *= spaceDim;
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    valueIndex[0] = cellIndex;
    for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
      valueIndex[1] = fieldIndex;
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        valueIndex[2] = ptIndex;
        double scalarValue = scalarValues(cellIndex,ptIndex);
        double *value = &values[ values.getEnumeration(valueIndex) ];
        for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
          if (modType == MULTIPLY) {
            *value++ *= scalarValue;
          } else if (modType == DIVIDE) {
            *value++ /= scalarValue;
          }
        }
      }
    }
  }
//  cout << "scalarModifyBasisValues: values:\n" << values;
}

ConstantScalarFunction::ConstantScalarFunction(double value) : Function(0) { 
  _value = value; 
}

void ConstantScalarFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {\
  CHECK_VALUES_RANK(values);
  for (int i=0; i < values.size(); i++) {
    values[i] = _value;
  }
}
void ConstantScalarFunction::scalarMultiplyFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache) {
  if (_value != 1.0) {
    for (int i=0; i < values.size(); i++) {
      values[i] *= _value;
    }
  }
}
void ConstantScalarFunction::scalarDivideFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache) {
  if (_value != 1.0) {
    for (int i=0; i < values.size(); i++) {
      values[i] /= _value;
    }
  }
}
void ConstantScalarFunction::scalarMultiplyBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
  // we don't actually care about the shape of basisValues--just use the FunctionValues versions:
  scalarMultiplyFunctionValues(basisValues,basisCache);
}
void ConstantScalarFunction::scalarDivideBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache) {
  scalarDivideFunctionValues(basisValues,basisCache);
}

double ConstantScalarFunction::value() {
  return _value;
}

ConstantVectorFunction::ConstantVectorFunction(vector<double> value) : Function(1) { 
  _value = value; 
}

vector<double> ConstantVectorFunction::value() {
  return _value;
}

void ConstantVectorFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  // values are stored in (C,P,D) order, the important thing here being that we can do this:
  for (int i=0; i < values.size(); ) {
    for (int d=0; d < _value.size(); d++) {
      values[i++] = _value[d];
    }
  }
}


int ProductFunction::productRank(FunctionPtr f1, FunctionPtr f2) {
  if (f1->rank() == f2->rank()) return 0;
  if (f1->rank() == 0) return f2->rank();
  if (f2->rank() == 0) return f1->rank();
  TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank pairing for function product.");
}

ProductFunction::ProductFunction(FunctionPtr f1, FunctionPtr f2) : Function( productRank(f1,f2) ) {
  // for simplicity of values() code, ensure that rank of f1 ≤ rank of f2:
  if ( f1->rank() <= f2->rank() ) {
    _f1 = f1;
    _f2 = f2;
  } else {
    _f1 = f2;
    _f2 = f1;
  }
}
void ProductFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  if (( _f2->rank() > 0) && (this->rank() == 0)) { // tensor product resulting in scalar value
    _f2->valuesDottedWithTensor(values, _f1, basisCache);
  } else { // scalar multiplication by f1, then
    _f2->values(values,basisCache);
    _f1->scalarMultiplyFunctionValues(values, basisCache);
  }
}

QuotientFunction::QuotientFunction(FunctionPtr f, FunctionPtr scalarDivisor) : Function( f->rank() ) {
  if ( scalarDivisor->rank() != 0 ) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank combination.");
  }
  _f = f;
  _scalarDivisor = scalarDivisor;
}
void QuotientFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  _f->values(values,basisCache);
  _scalarDivisor->scalarDivideFunctionValues(values, basisCache);
}

SumFunction::SumFunction(FunctionPtr f1, FunctionPtr f2) : Function(f1->rank()) {
  TEST_FOR_EXCEPTION( f1->rank() != f2->rank(), std::invalid_argument, "summands must be of like rank.");
  _f1 = f1;
  _f2 = f2;
}
void SumFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  _f1->values(values,basisCache);
  _f2->addToValues(values,basisCache);
}

double hFunction::value(double x, double y, double h) {
    return h;
}
void hFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  
  FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
  const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    double h = sqrt(cellMeasures(cellIndex));
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double x = (*points)(cellIndex,ptIndex,0);
      double y = (*points)(cellIndex,ptIndex,1);
      values(cellIndex,ptIndex) = value(x,y,h);
    }
  }
}

void SimpleFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  
  const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double x = (*points)(cellIndex,ptIndex,0);
      double y = (*points)(cellIndex,ptIndex,1);
      values(cellIndex,ptIndex) = value(x,y);
    }
  }
}

UnitNormalFunction::UnitNormalFunction() : Function(1) {}

void UnitNormalFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  const FieldContainer<double> *sideNormals = &(basisCache->getSideNormals());
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double n1 = (*sideNormals)(cellIndex,ptIndex,0);
      double n2 = (*sideNormals)(cellIndex,ptIndex,1);
      values(cellIndex,ptIndex,0) = n1;
      values(cellIndex,ptIndex,1) = n2;
    }
  }
}

Teuchos::RCP<ProductFunction> operator*(FunctionPtr f1, FunctionPtr f2) {
  return Teuchos::rcp( new ProductFunction(f1,f2) );
}

Teuchos::RCP<QuotientFunction> operator/(FunctionPtr f1, FunctionPtr scalarDivisor) {
  return Teuchos::rcp( new QuotientFunction(f1,scalarDivisor) );
}

Teuchos::RCP<QuotientFunction> operator/(FunctionPtr f1, double divisor) {
  return f1 / Teuchos::rcp( new ConstantScalarFunction(divisor) );
}

//ConstantScalarFunctionPtr operator*(ConstantScalarFunctionPtr f1, ConstantScalarFunctionPtr f2) {
//  return Teuchos::rcp( new ConstantScalarFunction(f1->value() * f2->value()) );
//}
//
//ConstantScalarFunctionPtr operator/(ConstantScalarFunctionPtr f1, ConstantScalarFunctionPtr f2) {
//  return Teuchos::rcp( new ConstantScalarFunction(f1->value() / f2->value()) );  
//}

//ConstantVectorFunctionPtr operator*(ConstantVectorFunctionPtr f1, ConstantScalarFunctionPtr f2) {
//  vector<double> value = f1->value();
//  for (int d=0; d<value.size(); d++) {
//    value[d] *= f2->value();
//  }
//  return Teuchos::rcp( new ConstantVectorFunction(value) );  
//}
//
//ConstantVectorFunctionPtr operator*(ConstantScalarFunctionPtr f1, ConstantVectorFunctionPtr f2) {
//  return f2 * f1;
//}
//
//ConstantVectorFunctionPtr operator/(ConstantVectorFunctionPtr f1, ConstantScalarFunctionPtr f2) {
//  vector<double> value = f1->value();
//  for (int d=0; d<value.size(); d++) {
//    value[d] /= f2->value();
//  }
//  return Teuchos::rcp( new ConstantVectorFunction(value) );  
//}

FunctionPtr operator*(double weight, FunctionPtr f) {
  return Teuchos::rcp( new ConstantScalarFunction(weight) ) * f;
}

FunctionPtr operator*(FunctionPtr f, double weight) {
  return weight * f;
}

FunctionPtr operator*(vector<double> weight, FunctionPtr f) {
  return Teuchos::rcp( new ConstantVectorFunction(weight) ) * f;
}

FunctionPtr operator*(FunctionPtr f, vector<double> weight) {
  return weight * f;
}

SumFunctionPtr operator+(FunctionPtr f1, FunctionPtr f2) {
  return Teuchos::rcp( new SumFunction(f1, f2) );
}