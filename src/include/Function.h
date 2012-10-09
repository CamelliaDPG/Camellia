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
#include "BilinearForm.h"

class Mesh;
class ExactSolution;

class Function;
typedef Teuchos::RCP<Function> FunctionPtr;

class Function {
private:
  enum FunctionModificationType{ MULTIPLY, DIVIDE }; // private, used by scalarModify[.*]Values
protected:
  int _rank;
  void CHECK_VALUES_RANK(FieldContainer<double> &values); // throws exception on bad values rank
public:
  Function();
  Function(int rank);
  
  virtual bool isZero() { return false; } // if true, the function is identically zero
  
  virtual bool boundaryValueOnly() { return false; } // if true, indicates a function defined only on element boundaries (mesh skeleton)
  
  virtual void values(FieldContainer<double> &values, EOperatorExtended op, BasisCachePtr basisCache);
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) = 0;
  
  virtual FunctionPtr x();
  virtual FunctionPtr y();
  virtual FunctionPtr z();
  
  virtual FunctionPtr dx();
  virtual FunctionPtr dy();
  virtual FunctionPtr dz();
  virtual FunctionPtr div();
  virtual FunctionPtr grad();
  
  int rank();
  
  virtual void addToValues(FieldContainer<double> &valuesToAddTo, BasisCachePtr basisCache);
  
  void integrate(FieldContainer<double> &cellIntegrals, BasisCachePtr basisCache, bool sumInto=false);
  
  double integrate(Teuchos::RCP<Mesh> mesh);
  
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
  
  virtual string displayString() { return "f"; }
  
  void writeBoundaryValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath);
  void writeValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath);
  
  static FunctionPtr zero();
private:
  void scalarModifyFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache,
                                  FunctionModificationType modType);
  
  void scalarModifyBasisValues(FieldContainer<double> &values, BasisCachePtr basisCache,
                               FunctionModificationType modType);

};

class SimpleFunction : public Function {
public:
  virtual double value(double x, double y) = 0;
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};
typedef Teuchos::RCP<SimpleFunction> SimpleFunctionPtr;

class ConstantScalarFunction : public SimpleFunction {
  double _value;
  string _stringDisplay;
public:
  ConstantScalarFunction(double value);
  ConstantScalarFunction(double value, string stringDisplay);
  string displayString();
  bool isZero();
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  void scalarMultiplyFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache);
  void scalarDivideFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache);
  void scalarMultiplyBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache);
  void scalarDivideBasisValues(FieldContainer<double> &basisValues, BasisCachePtr basisCache);
  double value();
  double value(double x, double y);
  
  FunctionPtr dx();
  FunctionPtr dy();
  // FunctionPtr dz();  // Hmm... a design issue: if we implement dz() then grad() will return a 3D function, not what we want...  It may be that grad() should require a spaceDim argument.  I'm not sure.
};

class ConstantVectorFunction : public Function {
  vector<double> _value;
public:
  ConstantVectorFunction(vector<double> value);
  bool isZero();
  
  FunctionPtr x();
  FunctionPtr y();
//  FunctionPtr z();
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  vector<double> value();
};

class ExactSolutionFunction : public Function { // for scalars, for now
  Teuchos::RCP<ExactSolution> _exactSolution;
  int _trialID;
public:
  ExactSolutionFunction(Teuchos::RCP<ExactSolution> exactSolution, int trialID);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

class ProductFunction : public Function {
private:
  int productRank(FunctionPtr f1, FunctionPtr f2);
  FunctionPtr _f1, _f2;
public:
  ProductFunction(FunctionPtr f1, FunctionPtr f2);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  virtual bool boundaryValueOnly();
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
};

class QuotientFunction : public Function {
  FunctionPtr _f, _scalarDivisor;
public:
  QuotientFunction(FunctionPtr f, FunctionPtr scalarDivisor);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  virtual bool boundaryValueOnly();
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
};

class SumFunction : public Function {
  FunctionPtr _f1, _f2;
public:
  SumFunction(FunctionPtr f1, FunctionPtr f2);
  
  FunctionPtr x();
  FunctionPtr y();
  FunctionPtr z();
  
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

class hFunction : public Function {
public:
  virtual double value(double x, double y, double h);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

class UnitNormalFunction : public Function {
  int _comp;
public:
  UnitNormalFunction(int comp=-1); // -1: the vector normal.  Otherwise, picks out the comp component
  
  FunctionPtr x();
  FunctionPtr y();
  
  bool boundaryValueOnly();
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

class ScalarFunctionOfNormal : public Function { // 2D for now
public:
  bool boundaryValueOnly();
  virtual double value(double x, double y, double n1, double n2) = 0;
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

class SideParityFunction : public Function {
public:
  bool boundaryValueOnly();
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

class VectorizedFunction : public Function {
  vector< FunctionPtr > _fxns;
public:
  virtual FunctionPtr x();
  virtual FunctionPtr y();
//  virtual FunctionPtr z();
  
  VectorizedFunction(FunctionPtr f1, FunctionPtr f2);
  VectorizedFunction(FunctionPtr f1, FunctionPtr f2, FunctionPtr f3);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

//ConstantScalarFunctionPtr operator*(ConstantScalarFunctionPtr f1, ConstantScalarFunctionPtr f2);
//ConstantScalarFunctionPtr operator/(ConstantScalarFunctionPtr f1, ConstantScalarFunctionPtr f2);

FunctionPtr operator*(FunctionPtr f1, FunctionPtr f2);
FunctionPtr operator/(FunctionPtr f1, FunctionPtr scalarDivisor);
FunctionPtr operator/(FunctionPtr f1, double divisor);

//ConstantVectorFunctionPtr operator*(ConstantVectorFunctionPtr f1, ConstantScalarFunctionPtr f2);
//ConstantVectorFunctionPtr operator*(ConstantScalarFunctionPtr f1, ConstantVectorFunctionPtr f2);
//ConstantVectorFunctionPtr operator/(ConstantVectorFunctionPtr f1, ConstantScalarFunctionPtr f2);

FunctionPtr operator*(double weight, FunctionPtr f);
FunctionPtr operator*(FunctionPtr f, double weight);
FunctionPtr operator*(vector<double> weight, FunctionPtr f);
FunctionPtr operator*(FunctionPtr f, vector<double> weight);

FunctionPtr operator+(FunctionPtr f1, FunctionPtr f2);
FunctionPtr operator-(FunctionPtr f1, FunctionPtr f2);
FunctionPtr operator-(FunctionPtr f);

#endif