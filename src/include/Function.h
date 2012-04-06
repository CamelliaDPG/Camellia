//
//  Function.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Function_h
#define Camellia_Function_h

#include "BasisCache.h"

class Function;
typedef Teuchos::RCP<Function> FunctionPtr;

class Function {
private:
  enum FunctionModificationType{ MULTIPLY, DIVIDE }; // private, used 
protected:
  int _rank;
public:
  Function();
  Function(int rank);
  
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) = 0;
  int rank();
  
  // divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
  virtual void scalarMultiplyFunctionValues(FieldContainer<double> &functionValues, BasisCachePtr basisCache);
  
  // divide values by this function (supported only when this is a scalar)
  virtual void scalarDivideFunctionValues(FieldContainer<double> &functionValues, BasisCachePtr basisCache);
  
  // divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
  virtual void scalarMultiplyBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache);
  
  // divide values by this function (supported only when this is a scalar)
  virtual void scalarDivideBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache);
  
  virtual void valuesDottedWithTensor(FieldContainer<double> &values, 
                                      FunctionPtr tensorFunctionOfLikeRank, 
                                      BasisCachePtr basisCache);
private:
  void scalarModifyFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache,
                                  FunctionModificationType modType);
  
  void scalarModifyBasisValues(FieldContainer<double> &values, BasisCachePtr basisCache,
                               FunctionModificationType modType);

};

class ConstantScalarFunction : public Function {
  double _value;
public:
  ConstantScalarFunction(double value);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  void scalarMultiplyFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache);
  void scalarDivideFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache);
  void scalarMultiplyBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache);
  void scalarDivideBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache);
};

class ConstantVectorFunction : public Function {
  vector<double> _value;
public:
  ConstantVectorFunction(vector<double> value);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

class ProductFunction : public Function {
private:
  int productRank(FunctionPtr f1, FunctionPtr f2);
  FunctionPtr _f1, _f2;
public:
  ProductFunction(FunctionPtr f1, FunctionPtr f2);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

class QuotientFunction : public Function {
  FunctionPtr _f, _scalarDivisor;
public:
  QuotientFunction(FunctionPtr f, FunctionPtr scalarDivisor);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

Teuchos::RCP<ProductFunction> operator*(FunctionPtr f1, FunctionPtr f2);
Teuchos::RCP<QuotientFunction> operator/(FunctionPtr f1, FunctionPtr scalarDivisor);

#endif
