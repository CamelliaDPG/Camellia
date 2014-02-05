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

using namespace IntrepidExtendedTypes;

class Mesh;
class ExactSolution;
class Solution;
class Var;
class Function;
class BasisCache; // BasisCache.h and Function.h #include each other...

typedef Teuchos::RCP<Function> FunctionPtr;
typedef Teuchos::RCP<Var> VarPtr;
typedef Teuchos::RCP<Solution> SolutionPtr;

class Function {
private:
  enum FunctionModificationType{ MULTIPLY, DIVIDE }; // private, used by scalarModify[.*]Values
protected:
  int _rank;
  string _displayString; // this is here mostly for identifying functions in the debugger
  void CHECK_VALUES_RANK(FieldContainer<double> &values); // throws exception on bad values rank
  double _time;
public:
  Function();
  Function(int rank);
  virtual ~Function() {}

  virtual void setTime(double time);
  virtual double getTime();

  bool equals(FunctionPtr f, BasisCachePtr basisCacheForCellsToCompare, double tol = 1e-14);

  virtual bool isZero() { return false; } // if true, the function is identically zero

  virtual bool boundaryValueOnly() { return false; } // if true, indicates a function defined only on element boundaries (mesh skeleton)

  virtual void values(FieldContainer<double> &values, EOperatorExtended op, BasisCachePtr basisCache);
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) = 0;

  static FunctionPtr op(FunctionPtr f, EOperatorExtended op);

  virtual FunctionPtr x();
  virtual FunctionPtr y();
  virtual FunctionPtr z();

  virtual FunctionPtr dx();
  virtual FunctionPtr dy();
  virtual FunctionPtr dz();
  virtual FunctionPtr div();
  virtual FunctionPtr grad(int numComponents=-1);

// inverse() presently unused: and unclear how useful...
//  virtual FunctionPtr inverse();

  int rank();

  virtual void addToValues(FieldContainer<double> &valuesToAddTo, BasisCachePtr basisCache);

  double integralOfJump(Teuchos::RCP<Mesh> mesh, GlobalIndexType cellID, int sideIndex, int cubatureDegreeEnrichment);

  double integralOfJump(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment);

  double integrate(BasisCachePtr basisCache);
  void integrate(FieldContainer<double> &cellIntegrals, BasisCachePtr basisCache, bool sumInto=false);

  // integrate over only one cell
  //  double integrate(int cellID, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0);
  double integrate(GlobalIndexType cellID, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false);

  // return all cell integrals
  map<int,double> cellIntegrals( Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false);
  // return cell integrals specified in input argument cellIDs
  map<int,double> cellIntegrals(vector<GlobalIndexType> cellIDs, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false);

  double integrate( Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false, bool requireSideCaches = false);

  // adaptive quadrature
  double integrate(Teuchos::RCP<Mesh> mesh, double tol, bool testVsTest = false);

  bool isPositive(BasisCachePtr basisCache);
  bool isPositive(Teuchos::RCP<Mesh> mesh, int cubEnrich = 0, bool testVsTest = false);

  double l2norm(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0);

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

  virtual string displayString();

  void writeBoundaryValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath);
  void writeValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath);

  static double evaluate(FunctionPtr f, double x); // for testing
  static double evaluate(FunctionPtr f, double x, double y); // for testing
  static double evaluate(FunctionPtr f, double x, double y, double z); // for testing

  static bool isNull(FunctionPtr f);

  // static Function construction methods:
  static FunctionPtr composedFunction( FunctionPtr f, FunctionPtr arg_g); // note: SLOW! avoid when possible...
  static FunctionPtr constant(double value);
  static FunctionPtr constant(vector<double> &value);

  static FunctionPtr h();
  static FunctionPtr meshBoundaryCharacteristic(); // 1 on mesh boundary, 0 elsewhere
  static FunctionPtr meshSkeletonCharacteristic(); // 1 on mesh skeleton, 0 elsewhere
  static FunctionPtr polarize(FunctionPtr f);
  static FunctionPtr vectorize(FunctionPtr f1, FunctionPtr f2);
  static FunctionPtr normal(); // unit outward-facing normal on each element boundary
  static FunctionPtr null();
  static FunctionPtr sideParity();
  static FunctionPtr solution(VarPtr var, SolutionPtr soln);
  static FunctionPtr zero(int rank=0);
  static FunctionPtr restrictToCellBoundary(FunctionPtr f);

  static FunctionPtr xn(int n=1);
  static FunctionPtr yn(int n=1);
  static FunctionPtr zn(int n=1);
//  static FunctionPtr jump(FunctionPtr f);

  static FunctionPtr cellCharacteristic(GlobalIndexType cellID);
  static FunctionPtr cellCharacteristic(set<GlobalIndexType> cellIDs);

  static FunctionPtr xPart(FunctionPtr vectorFunction);
  static FunctionPtr yPart(FunctionPtr vectorFunction);
  static FunctionPtr zPart(FunctionPtr vectorFunction);
private:
  void scalarModifyFunctionValues(FieldContainer<double> &values, BasisCachePtr basisCache,
                                  FunctionModificationType modType);

  void scalarModifyBasisValues(FieldContainer<double> &values, BasisCachePtr basisCache,
                               FunctionModificationType modType);


};

// restricts a given function to just the mesh skeleton
class BoundaryFunction : public Function{
 private:
  FunctionPtr _f;
 public:
  BoundaryFunction(FunctionPtr f){
    _f = f;
  }
  bool boundaryValueOnly(){
    return true;
  }
  FunctionPtr getFunction(){
    return _f;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    _f->values(values,basisCache);
  }
};

// restricts a given function to just the mesh skeleton
class InternalBoundaryFunction : public BoundaryFunction{
 public:
 InternalBoundaryFunction(FunctionPtr f): BoundaryFunction(f){};
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    this->getFunction()->values(values,basisCache);

    
    // TODO: work out what was meant to happen here.  Should the following code be completed or excised?
    int sideIndex = basisCache->getSideIndex();
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0;i<cellIDs.size();i++){
      //      for (int sideIndex = 0
      //      }
    }

  }
};

class SimpleFunction : public Function {
public:
  virtual ~SimpleFunction() {}
  virtual double value(double x);
  virtual double value(double x, double y);
  virtual double value(double x, double y, double z);
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};
typedef Teuchos::RCP<SimpleFunction> SimpleFunctionPtr;

class PolarizedFunction : public Function { // takes a 2D Function of x and y, interpreting it as function of r and theta
  // i.e. to implement f(r,theta) = r sin theta
  // pass in a Function f(x,y) = x sin y.
  // Given the implementation, it is important that f depend *only* on x and y, and not on the mesh, etc.
  // (the only method in BasisCache that f may call is getPhysicalCubaturePoints())
  FunctionPtr _f;
public:
  PolarizedFunction( FunctionPtr f_of_xAsR_yAsTheta );
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);

  FunctionPtr dx();
  FunctionPtr dy();

  Teuchos::RCP<PolarizedFunction> dtheta();
  Teuchos::RCP<PolarizedFunction> dr();

  virtual string displayString(); // for PolarizedFunction, this should be _f->displayString() + "(r,theta)";

  bool isZero();

  static Teuchos::RCP<PolarizedFunction> r();
  static Teuchos::RCP<PolarizedFunction> sin_theta();
  static Teuchos::RCP<PolarizedFunction> cos_theta();
};

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
  FunctionPtr dz();  // Hmm... a design issue: if we implement dz() then grad() will return a 3D function, not what we want...  It may be that grad() should require a spaceDim argument.  I'm not sure.
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

  FunctionPtr x();
  FunctionPtr y();
  FunctionPtr z();

  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();

  string displayString(); // _f1->displayString() << " " << _f2->displayString();
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
  string displayString();
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

  FunctionPtr grad(int numComponents=-1); // gradient of sum is the sum of gradients
  FunctionPtr div();  // divergence of sum is sum of divergences

  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  bool boundaryValueOnly();

  string displayString();
};

class hFunction : public Function {
public:
  virtual double value(double x, double y, double h);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  string displayString();
};

class UnitNormalFunction : public Function {
  int _comp;
public:
  UnitNormalFunction(int comp=-1); // -1: the vector normal.  Otherwise, picks out the comp component

  FunctionPtr x();
  FunctionPtr y();

  bool boundaryValueOnly();
  string displayString();
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
  SideParityFunction();
  bool boundaryValueOnly();
  string displayString();
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
};

class VectorizedFunction : public Function {
private:
  vector< FunctionPtr > _fxns;
  FunctionPtr di(int i); // derivative in the ith coordinate direction
public:
  virtual FunctionPtr x();
  virtual FunctionPtr y();
  virtual FunctionPtr z();

  virtual FunctionPtr dx();
  virtual FunctionPtr dy();
  virtual FunctionPtr dz();

  VectorizedFunction(const vector< FunctionPtr > &fxns);
  VectorizedFunction(FunctionPtr f1, FunctionPtr f2);
  VectorizedFunction(FunctionPtr f1, FunctionPtr f2, FunctionPtr f3);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);

  virtual string displayString();
  int dim();
  
  bool isZero();
  
  virtual ~VectorizedFunction() {
//    cout << "VectorizedFunction destructor.\n";
  }
};

//ConstantScalarFunctionPtr operator*(ConstantScalarFunctionPtr f1, ConstantScalarFunctionPtr f2);
//ConstantScalarFunctionPtr operator/(ConstantScalarFunctionPtr f1, ConstantScalarFunctionPtr f2);

FunctionPtr operator*(FunctionPtr f1, FunctionPtr f2);
FunctionPtr operator/(FunctionPtr f1, FunctionPtr scalarDivisor);
FunctionPtr operator/(FunctionPtr f1, double divisor);
FunctionPtr operator/(double value, FunctionPtr scalarDivisor);

//ConstantVectorFunctionPtr operator*(ConstantVectorFunctionPtr f1, ConstantScalarFunctionPtr f2);
//ConstantVectorFunctionPtr operator*(ConstantScalarFunctionPtr f1, ConstantVectorFunctionPtr f2);
//ConstantVectorFunctionPtr operator/(ConstantVectorFunctionPtr f1, ConstantScalarFunctionPtr f2);

FunctionPtr operator*(double weight, FunctionPtr f);
FunctionPtr operator*(FunctionPtr f, double weight);
FunctionPtr operator*(vector<double> weight, FunctionPtr f);
FunctionPtr operator*(FunctionPtr f, vector<double> weight);

FunctionPtr operator+(FunctionPtr f1, FunctionPtr f2);
FunctionPtr operator+(FunctionPtr f1, double value);
FunctionPtr operator+(double value, FunctionPtr f1);

FunctionPtr operator-(FunctionPtr f1, FunctionPtr f2);
FunctionPtr operator-(FunctionPtr f1, double value);
FunctionPtr operator-(double value, FunctionPtr f1);

FunctionPtr operator-(FunctionPtr f);

// here, some particular functions
// TODO: hide the classes here, and instead implement as static FunctionPtr Function::cos_y(), e.g.
class Cos_y : public SimpleFunction {
  double value(double x, double y);
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
  string displayString();
};

class Sin_y : public SimpleFunction {
  double value(double x, double y);
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
  string displayString();
};

class Cos_x : public SimpleFunction {
  double value(double x, double y);
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
  string displayString();
};

class Sin_x : public SimpleFunction {
  double value(double x, double y);
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
  string displayString();
};

class Exp_x : public SimpleFunction {
public:
  double value(double x, double y);
  FunctionPtr dx();
  FunctionPtr dy();
  string displayString();
};

class Exp_y : public SimpleFunction {
public:
  double value(double x, double y);
  FunctionPtr dx();
  FunctionPtr dy();
  string displayString();
};

class Xn : public SimpleFunction {
  int _n;
public:
  Xn(int n);
  double value(double x);
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
  string displayString();
};

class Yn : public SimpleFunction {
  int _n;
public:
  Yn(int n);
  double value(double x, double y);
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
  string displayString();
};

class Zn : public SimpleFunction {
  int _n;
public:
  Zn(int n);
  double value(double x, double y, double z);
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
  string displayString();
};

class Cos_ax : public SimpleFunction {
  double _a,_b;
public:
  Cos_ax(double a, double b=0);
  double value(double x);
  FunctionPtr dx();
  FunctionPtr dy();

  string displayString();
};

class Sin_ax : public SimpleFunction {
  double _a, _b;
public:
  Sin_ax(double a, double b=0) {
    _a = a;
    _b = b;
  }
  double value(double x) {
    return sin( _a * x + _b);
  }
  FunctionPtr dx() {
    return _a * (FunctionPtr) Teuchos::rcp(new Cos_ax(_a,_b));
  }
  FunctionPtr dy() {
    return Function::zero();
  }
  string displayString() {
    ostringstream ss;
    ss << "\\sin( " << _a << " x )";
    return ss.str();
  }
};

class Cos_ay : public SimpleFunction {
  double _a;
public:
  Cos_ay(double a);
  double value(double x, double y);
  FunctionPtr dx();
  FunctionPtr dy();

  string displayString();
};

class Sin_ay : public SimpleFunction {
  double _a;
public:
  Sin_ay(double a) {
    _a = a;
  }
  double value(double x, double y) {
    return sin( _a * y);
  }
  FunctionPtr dx() {
    return Function::zero();
  }
  FunctionPtr dy() {
    return _a * (FunctionPtr) Teuchos::rcp(new Cos_ay(_a));
  }
  string displayString() {
    ostringstream ss;
    ss << "\\sin( " << _a << " y )";
    return ss.str();
  }
};


class Exp_ax : public SimpleFunction {
  double _a;
public:
  Exp_ax(double a) {
    _a = a;
  }
  double value(double x, double y) {
    return exp( _a * x);
  }
  FunctionPtr dx() {
    return _a * (FunctionPtr) Teuchos::rcp(new Exp_ax(_a));
  }
  FunctionPtr dy() {
    return Function::zero();
  }
  string displayString() {
    ostringstream ss;
    ss << "\\exp( " << _a << " x )";
    return ss.str();
  }
};

class PhysicalPointCache : public BasisCache {
  FieldContainer<double> _physCubPoints;
public:
  PhysicalPointCache(const FieldContainer<double> &physCubPoints);
  const FieldContainer<double> & getPhysicalCubaturePoints();
  FieldContainer<double> & writablePhysicalCubaturePoints();
};

#endif
