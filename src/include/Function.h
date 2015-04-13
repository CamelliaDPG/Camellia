//
//  Function.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Function_h
#define Camellia_Function_h

#include "TypeDefs.h"

#include "CamelliaIntrepidExtendedTypes.h"
#include "Intrepid_FieldContainer.hpp"

#include <string>

// using namespace Camellia;
using namespace std;

namespace Camellia {
  class ExactSolution;

  template <typename Scalar>
  class Function {
  private:
    enum FunctionModificationType{ MULTIPLY, DIVIDE }; // private, used by scalarModify[.*]Values
  protected:
    int _rank;
    string _displayString; // this is here mostly for identifying functions in the debugger
    void CHECK_VALUES_RANK(Intrepid::FieldContainer<Scalar> &values); // throws exception on bad values rank
    double _time;
  public:
    Function();
    Function(int rank);
    virtual ~Function() {}

    virtual void setTime(double time);
    virtual double getTime();

    bool equals(Teuchos::RCP<Function<Scalar> > f, BasisCachePtr basisCacheForCellsToCompare, double tol = 1e-14);

    virtual bool isZero() { return false; } // if true, the function is identically zero

    virtual bool boundaryValueOnly() { return false; } // if true, indicates a function defined only on element boundaries (mesh skeleton)

    virtual void values(Intrepid::FieldContainer<Scalar> &values, Camellia::EOperator op, BasisCachePtr basisCache);
    virtual void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) = 0;

    static Teuchos::RCP<Function<Scalar> > op(Teuchos::RCP<Function<Scalar> > f, Camellia::EOperator op);

    virtual Teuchos::RCP<Function<Scalar> > x();
    virtual Teuchos::RCP<Function<Scalar> > y();
    virtual Teuchos::RCP<Function<Scalar> > z();

    virtual Teuchos::RCP<Function<Scalar> > dx();
    virtual Teuchos::RCP<Function<Scalar> > dy();
    virtual Teuchos::RCP<Function<Scalar> > dz();
  //  virtual Teuchos::RCP<Function<Scalar> > dt(); // TODO: rework ParametricCurve (Function subclass) so that we can define dt() thus.

    virtual Teuchos::RCP<Function<Scalar> > div();
    virtual Teuchos::RCP<Function<Scalar> > curl();
    virtual Teuchos::RCP<Function<Scalar> > grad(int numComponents=-1);

    virtual void importCellData(std::vector<GlobalIndexType> cellIDs) {}

  // inverse() presently unused: and unclear how useful...
  //  virtual Teuchos::RCP<Function<Scalar> > inverse();

    int rank();

    virtual void addToValues(Intrepid::FieldContainer<Scalar> &valuesToAddTo, BasisCachePtr basisCache);

    Scalar integralOfJump(Teuchos::RCP<Mesh> mesh, GlobalIndexType cellID, int sideIndex, int cubatureDegreeEnrichment);

    Scalar integralOfJump(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment);

    Scalar integrate(BasisCachePtr basisCache);
    void integrate(Intrepid::FieldContainer<Scalar> &cellIntegrals, BasisCachePtr basisCache, bool sumInto=false);

    // integrate over only one cell
    //  double integrate(int cellID, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0);
    Scalar integrate(GlobalIndexType cellID, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false);

    // return all cell integrals
    map<int,Scalar> cellIntegrals( Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false);
    // return cell integrals specified in input argument cellIDs
    map<int,Scalar> cellIntegrals(vector<GlobalIndexType> cellIDs, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false);

    Scalar integrate( Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false, bool requireSideCaches = false,
                      bool spatialSidesOnly = false);

    // adaptive quadrature
    Scalar integrate(Teuchos::RCP<Mesh> mesh, double tol, bool testVsTest = false);

    bool isPositive(BasisCachePtr basisCache);
    bool isPositive(Teuchos::RCP<Mesh> mesh, int cubEnrich = 0, bool testVsTest = false);

    double l2norm(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool spatialSidesOnly = false);

    // divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
    virtual void scalarMultiplyFunctionValues(Intrepid::FieldContainer<Scalar> &functionValues, BasisCachePtr basisCache);

    // divide values by this function (supported only when this is a scalar)
    virtual void scalarDivideFunctionValues(Intrepid::FieldContainer<Scalar> &functionValues, BasisCachePtr basisCache);

    // divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
    virtual void scalarMultiplyBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache);

    // divide values by this function (supported only when this is a scalar)
    virtual void scalarDivideBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache);

    virtual void valuesDottedWithTensor(Intrepid::FieldContainer<Scalar> &values,
                                        Teuchos::RCP<Function<Scalar> > tensorFunctionOfLikeRank,
                                        BasisCachePtr basisCache);

    virtual string displayString();

    void writeBoundaryValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath);
    void writeValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath);

    // Note that in general, repeated calls to Function::evaluate() would be significantly more expensive than a call with many points to Function::values().
    // Also, evaluate() may fail for certain Function subclasses, including any that depend on the Mesh.
    virtual Scalar evaluate(double x);
    virtual Scalar evaluate(double x, double y);
    virtual Scalar evaluate(double x, double y, double z);

    virtual Scalar evaluate(Teuchos::RCP<Mesh> mesh, double x);
    virtual Scalar evaluate(Teuchos::RCP<Mesh> mesh, double x, double y);
    virtual Scalar evaluate(Teuchos::RCP<Mesh> mesh, double x, double y, double z);

    static Scalar evaluate(Teuchos::RCP<Function<Scalar> > f, double x); // for testing
    static Scalar evaluate(Teuchos::RCP<Function<Scalar> > f, double x, double y); // for testing
    static Scalar evaluate(Teuchos::RCP<Function<Scalar> > f, double x, double y, double z); // for testing

    static bool isNull(Teuchos::RCP<Function<Scalar> > f);

    // static Function construction methods:
    static Teuchos::RCP<Function<double> > composedFunction( Teuchos::RCP<Function<double> > f, Teuchos::RCP<Function<double> > arg_g); // note: SLOW! avoid when possible...
    static Teuchos::RCP<Function<Scalar> > constant(Scalar value);
    static Teuchos::RCP<Function<Scalar> > constant(vector<Scalar> &value);

    static Teuchos::RCP<Function<double> > min(Teuchos::RCP<Function<double> > f1, Teuchos::RCP<Function<double> > f2);
    static Teuchos::RCP<Function<double> > min(Teuchos::RCP<Function<double> > f1, double value);
    static Teuchos::RCP<Function<double> > min(double value, Teuchos::RCP<Function<double> > f2);
    static Teuchos::RCP<Function<double> > max(Teuchos::RCP<Function<double> > f1, Teuchos::RCP<Function<double> > f2);
    static Teuchos::RCP<Function<double> > max(Teuchos::RCP<Function<double> > f1, double value);
    static Teuchos::RCP<Function<double> > max(double value, Teuchos::RCP<Function<double> > f2);

    static Teuchos::RCP<Function<double> > h();
    // ! implements Heaviside step function, shifted right by xValue
    static Teuchos::RCP<Function<double> > heaviside(double xValue);

    static Teuchos::RCP<Function<double> > meshBoundaryCharacteristic(); // 1 on mesh boundary, 0 elsewhere
    static Teuchos::RCP<Function<double> > meshSkeletonCharacteristic(); // 1 on mesh skeleton, 0 elsewhere
    static Teuchos::RCP<Function<Scalar> > polarize(Teuchos::RCP<Function<Scalar> > f);
    static Teuchos::RCP<Function<Scalar> > vectorize(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2);
    static Teuchos::RCP<Function<Scalar> > vectorize(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2, Teuchos::RCP<Function<Scalar> > f3);
    static Teuchos::RCP<Function<double> > normal();    // unit outward-facing normal on each element boundary
    static Teuchos::RCP<Function<double> > normal_1D(); // -1 at left side of element, +1 at right
    static Teuchos::RCP<Function<double> > normalSpaceTime();
    static Teuchos::RCP<Function<Scalar> > null();
    static Teuchos::RCP<Function<double> > sideParity();
    static Teuchos::RCP<Function<Scalar> > solution(VarPtr var, SolutionPtr soln);
    static Teuchos::RCP<Function<double> > zero(int rank=0);
    static Teuchos::RCP<Function<Scalar> > restrictToCellBoundary(Teuchos::RCP<Function<Scalar> > f);

    static Teuchos::RCP<Function<double> > xn(int n=1);
    static Teuchos::RCP<Function<double> > yn(int n=1);
    static Teuchos::RCP<Function<double> > zn(int n=1);
    static Teuchos::RCP<Function<double> > tn(int n=1);
  //  static Teuchos::RCP<Function<Scalar> > jump(Teuchos::RCP<Function<Scalar> > f);

    static Teuchos::RCP<Function<double> > cellCharacteristic(GlobalIndexType cellID);
    static Teuchos::RCP<Function<double> > cellCharacteristic(set<GlobalIndexType> cellIDs);

    static Teuchos::RCP<Function<Scalar> > xPart(Teuchos::RCP<Function<Scalar> > vectorFunction);
    static Teuchos::RCP<Function<Scalar> > yPart(Teuchos::RCP<Function<Scalar> > vectorFunction);
    static Teuchos::RCP<Function<Scalar> > zPart(Teuchos::RCP<Function<Scalar> > vectorFunction);
  private:
    void scalarModifyFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache,
                                    FunctionModificationType modType);

    void scalarModifyBasisValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache,
                                 FunctionModificationType modType);
  };

  // restricts a given function to just the mesh skeleton
  template <typename Scalar>
  class BoundaryFunction : public Function<Scalar>{
   private:
    Teuchos::RCP<Function<Scalar> > _f;
   public:
    BoundaryFunction(Teuchos::RCP<Function<Scalar> > f){
      _f = f;
    }
    bool boundaryValueOnly(){
      return true;
    }
    Teuchos::RCP<Function<Scalar> > getFunction(){
      return _f;
    }
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache){
      _f->values(values,basisCache);
    }
  };

  // restricts a given function to just the mesh skeleton
  template <typename Scalar>
  class InternalBoundaryFunction : public BoundaryFunction<Scalar>{
   public:
   InternalBoundaryFunction(Teuchos::RCP<Function<Scalar> > f): BoundaryFunction<Scalar>(f){};
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache){
      this->getFunction()->values(values,basisCache);


      // TODO: work out what was meant to happen here.  Should the following code be completed or excised?
  //    int sideIndex = basisCache->getSideIndex();
  //    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
  //    int numPoints = values.dimension(1);
  //    Intrepid::FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
  //    for (int i = 0;i<cellIDs.size();i++){
  //      //      for (int sideIndex = 0
  //      //      }
  //    }

    }
  };

  template <typename Scalar>
  class SimpleFunction : public Function<Scalar> {
  public:
    virtual ~SimpleFunction() {}
    virtual Scalar value(double x);
    virtual Scalar value(double x, double y);
    virtual Scalar value(double x, double y, double z);
    virtual Scalar value(double x, double y, double z, double t);
    virtual void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  };

  template <typename Scalar>
  class SimpleVectorFunction : public Function<Scalar> {
  public:
    SimpleVectorFunction();
    virtual ~SimpleVectorFunction() {}
    virtual vector<Scalar> value(double x);
    virtual vector<Scalar> value(double x, double y);
    virtual vector<Scalar> value(double x, double y, double z);
    virtual void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  };

  template <typename Scalar>
  class PolarizedFunction : public Function<Scalar> { // takes a 2D Function of x and y, interpreting it as function of r and theta
    // i.e. to implement f(r,theta) = r sin theta
    // pass in a Function f(x,y) = x sin y.
    // Given the implementation, it is important that f depend *only* on x and y, and not on the mesh, etc.
    // (the only method in BasisCache that f may call is getPhysicalCubaturePoints())
    Teuchos::RCP<Function<Scalar> > _f;
  public:
    PolarizedFunction( Teuchos::RCP<Function<Scalar> > f_of_xAsR_yAsTheta );
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);

    Teuchos::RCP<Function<Scalar> > dx();
    Teuchos::RCP<Function<Scalar> > dy();

    Teuchos::RCP<PolarizedFunction<Scalar> > dtheta();
    Teuchos::RCP<PolarizedFunction<Scalar> > dr();

    virtual string displayString(); // for PolarizedFunction, this should be _f->displayString() + "(r,theta)";

    bool isZero();

    static Teuchos::RCP<PolarizedFunction<double> > r();
    static Teuchos::RCP<PolarizedFunction<double> > sin_theta();
    static Teuchos::RCP<PolarizedFunction<double> > cos_theta();
  };

  template <typename Scalar>
  class ConstantScalarFunction : public SimpleFunction<Scalar> {
    Scalar _value;
    string _stringDisplay;
  public:
    ConstantScalarFunction(Scalar value);
    ConstantScalarFunction(Scalar value, string stringDisplay);
    string displayString();
    bool isZero();
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    void scalarMultiplyFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    void scalarDivideFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    void scalarMultiplyBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache);
    void scalarDivideBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache);

    virtual Scalar value(double x);
    virtual Scalar value(double x, double y);
    virtual Scalar value(double x, double y, double z);

    using SimpleFunction<Scalar>::value; // avoid compiler warnings about the value() method below.
    Scalar value();

    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    Teuchos::RCP<Function<double> > dz();  // Hmm... a design issue: if we implement dz() then grad() will return a 3D function, not what we want...  It may be that grad() should require a spaceDim argument.  I'm not sure.
  };

  template <typename Scalar>
  class ConstantVectorFunction : public Function<Scalar> {
    vector<Scalar> _value;
  public:
    ConstantVectorFunction(vector<Scalar> value);
    bool isZero();

    Teuchos::RCP<Function<Scalar> > x();
    Teuchos::RCP<Function<Scalar> > y();
    Teuchos::RCP<Function<Scalar> > z();

    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    vector<Scalar> value();
  };

  class ExactSolutionFunction : public Function<double> { // for scalars, for now
    Teuchos::RCP<ExactSolution> _exactSolution;
    int _trialID;
  public:
    ExactSolutionFunction(Teuchos::RCP<ExactSolution> exactSolution, int trialID);
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
  };

  template <typename Scalar>
  class ProductFunction : public Function<Scalar> {
  private:
    int productRank(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2);
    Teuchos::RCP<Function<Scalar> > _f1, _f2;
  public:
    ProductFunction(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    virtual bool boundaryValueOnly();

    Teuchos::RCP<Function<Scalar> > x();
    Teuchos::RCP<Function<Scalar> > y();
    Teuchos::RCP<Function<Scalar> > z();

    Teuchos::RCP<Function<Scalar> > dx();
    Teuchos::RCP<Function<Scalar> > dy();
    Teuchos::RCP<Function<Scalar> > dz();

    string displayString(); // _f1->displayString() << " " << _f2->displayString();
  };

  template <typename Scalar>
  class QuotientFunction : public Function<Scalar> {
    Teuchos::RCP<Function<Scalar> > _f, _scalarDivisor;
  public:
    QuotientFunction(Teuchos::RCP<Function<Scalar> > f, Teuchos::RCP<Function<Scalar> > scalarDivisor);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    virtual bool boundaryValueOnly();
    Teuchos::RCP<Function<Scalar> > dx();
    Teuchos::RCP<Function<Scalar> > dy();
    Teuchos::RCP<Function<Scalar> > dz();
    string displayString();
  };

  template <typename Scalar>
  class SumFunction : public Function<Scalar> {
    Teuchos::RCP<Function<Scalar> > _f1, _f2;
  public:
    SumFunction(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2);

    Teuchos::RCP<Function<Scalar> > x();
    Teuchos::RCP<Function<Scalar> > y();
    Teuchos::RCP<Function<Scalar> > z();

    Teuchos::RCP<Function<Scalar> > dx();
    Teuchos::RCP<Function<Scalar> > dy();
    Teuchos::RCP<Function<Scalar> > dz();

    Teuchos::RCP<Function<Scalar> > grad(int numComponents=-1); // gradient of sum is the sum of gradients
    Teuchos::RCP<Function<Scalar> > div();  // divergence of sum is sum of divergences

    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
    bool boundaryValueOnly();

    string displayString();
  };

  class MinFunction : public Function<double> {
    Teuchos::RCP<Function<double> > _f1, _f2;
  public:
    MinFunction(Teuchos::RCP<Function<double> > f1, Teuchos::RCP<Function<double> > f2);

    Teuchos::RCP<Function<double> > x();
    Teuchos::RCP<Function<double> > y();
    Teuchos::RCP<Function<double> > z();

    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    bool boundaryValueOnly();

    string displayString();
  };

  class MaxFunction : public Function<double> {
    Teuchos::RCP<Function<double> > _f1, _f2;
  public:
    MaxFunction(Teuchos::RCP<Function<double> > f1, Teuchos::RCP<Function<double> > f2);

    Teuchos::RCP<Function<double> > x();
    Teuchos::RCP<Function<double> > y();
    Teuchos::RCP<Function<double> > z();

    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    bool boundaryValueOnly();

    string displayString();
  };

  class hFunction : public Function<double> {
  public:
    virtual double value(double x, double y, double h);
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    string displayString();
  };

  class UnitNormalFunction : public Function<double> {
    int _comp;
    bool _spaceTime;
  public:
    UnitNormalFunction(int comp=-1, bool spaceTime = false); // -1: the vector normal.  Otherwise, picks out the comp component

    Teuchos::RCP<Function<double> > x();
    Teuchos::RCP<Function<double> > y();
    Teuchos::RCP<Function<double> > z();

    bool boundaryValueOnly();
    string displayString();
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
  };

  template <typename Scalar>
  class ScalarFunctionOfNormal : public Function<Scalar> { // 2D for now
  public:
    bool boundaryValueOnly();
    virtual Scalar value(double x, double y, double n1, double n2) = 0;
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  };

  class SideParityFunction : public Function<double> {
  public:
    SideParityFunction();
    bool boundaryValueOnly();
    string displayString();
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
  };

  template <typename Scalar>
  class VectorizedFunction : public Function<Scalar> {
  private:
    vector< Teuchos::RCP<Function<Scalar> > > _fxns;
    Teuchos::RCP<Function<Scalar> > di(int i); // derivative in the ith coordinate direction
  public:
    virtual Teuchos::RCP<Function<Scalar> > x();
    virtual Teuchos::RCP<Function<Scalar> > y();
    virtual Teuchos::RCP<Function<Scalar> > z();

    virtual Teuchos::RCP<Function<Scalar> > dx();
    virtual Teuchos::RCP<Function<Scalar> > dy();
    virtual Teuchos::RCP<Function<Scalar> > dz();

    VectorizedFunction(const vector< Teuchos::RCP<Function<Scalar> > > &fxns);
    VectorizedFunction(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2);
    VectorizedFunction(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2, Teuchos::RCP<Function<Scalar> > f3);
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache);

    virtual string displayString();
    int dim();

    bool isZero();

    virtual ~VectorizedFunction() {
  //    cout << "VectorizedFunction destructor.\n";
    }
  };

  //ConstantScalarTeuchos::RCP<Function<Scalar> > operator*(ConstantScalarTeuchos::RCP<Function<Scalar> > f1, ConstantScalarTeuchos::RCP<Function<Scalar> > f2);
  //ConstantScalarTeuchos::RCP<Function<Scalar> > operator/(ConstantScalarTeuchos::RCP<Function<Scalar> > f1, ConstantScalarTeuchos::RCP<Function<Scalar> > f2);

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator*(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2);
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator/(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > scalarDivisor);
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator/(Teuchos::RCP<Function<Scalar> > f1, Scalar divisor);
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator/(Scalar value, Teuchos::RCP<Function<Scalar> > scalarDivisor);

  //ConstantVectorTeuchos::RCP<Function<Scalar> > operator*(ConstantVectorTeuchos::RCP<Function<Scalar> > f1, ConstantScalarTeuchos::RCP<Function<Scalar> > f2);
  //ConstantVectorTeuchos::RCP<Function<Scalar> > operator*(ConstantScalarTeuchos::RCP<Function<Scalar> > f1, ConstantVectorTeuchos::RCP<Function<Scalar> > f2);
  //ConstantVectorTeuchos::RCP<Function<Scalar> > operator/(ConstantVectorTeuchos::RCP<Function<Scalar> > f1, ConstantScalarTeuchos::RCP<Function<Scalar> > f2);

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator*(Scalar weight, Teuchos::RCP<Function<Scalar> > f);
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator*(Teuchos::RCP<Function<Scalar> > f, Scalar weight);
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator*(vector<Scalar> weight, Teuchos::RCP<Function<Scalar> > f);
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator*(Teuchos::RCP<Function<Scalar> > f, vector<Scalar> weight);

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator+(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2);
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator+(Teuchos::RCP<Function<Scalar> > f1, Scalar value);
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator+(Scalar value, Teuchos::RCP<Function<Scalar> > f1);

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator-(Teuchos::RCP<Function<Scalar> > f1, Teuchos::RCP<Function<Scalar> > f2);
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator-(Teuchos::RCP<Function<Scalar> > f1, Scalar value);
  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator-(Scalar value, Teuchos::RCP<Function<Scalar> > f1);

  template <typename Scalar>
  Teuchos::RCP<Function<Scalar> > operator-(Teuchos::RCP<Function<Scalar> > f);

  // here, some particular functions
  // TODO: hide the classes here, and instead implement as static Teuchos::RCP<Function<Scalar> > Function::cos_y(), e.g.
  class Cos_y : public SimpleFunction<double> {
    double value(double x, double y);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    Teuchos::RCP<Function<double> > dz();
    string displayString();
  };

  class Sin_y : public SimpleFunction<double> {
    double value(double x, double y);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    Teuchos::RCP<Function<double> > dz();
    string displayString();
  };

  class Cos_x : public SimpleFunction<double> {
    double value(double x, double y);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    Teuchos::RCP<Function<double> > dz();
    string displayString();
  };

  class Sin_x : public SimpleFunction<double> {
    double value(double x, double y);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    Teuchos::RCP<Function<double> > dz();
    string displayString();
  };

  class Exp_x : public SimpleFunction<double> {
  public:
    double value(double x, double y);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    Teuchos::RCP<Function<double> > dz();
    string displayString();
  };

  class Exp_y : public SimpleFunction<double> {
  public:
    double value(double x, double y);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    Teuchos::RCP<Function<double> > dz();
    string displayString();
  };

  class Exp_z : public SimpleFunction<double> {
  public:
    double value(double x, double y, double z);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    Teuchos::RCP<Function<double> > dz();
    string displayString();
  };

  class Xn : public SimpleFunction<double> {
    int _n;
  public:
    Xn(int n);
    double value(double x);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    Teuchos::RCP<Function<double> > dz();
    string displayString();
  };

  class Yn : public SimpleFunction<double> {
    int _n;
  public:
    Yn(int n);
    double value(double x, double y);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    Teuchos::RCP<Function<double> > dz();
    string displayString();
  };

  class Zn : public SimpleFunction<double> {
    int _n;
  public:
    Zn(int n);
    double value(double x, double y, double z);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    Teuchos::RCP<Function<double> > dz();
    string displayString();
  };

  class Tn : public SimpleFunction<double> {
    int _n;
  public:
    Tn(int n);
    double value(double x, double t);
    double value(double x, double y, double t);
    double value(double x, double y, double z, double t);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    Teuchos::RCP<Function<double> > dz();
    Teuchos::RCP<Function<double> > dt();
    string displayString();
  };

  class Cos_ax : public SimpleFunction<double> {
    double _a,_b;
  public:
    Cos_ax(double a, double b=0);
    double value(double x);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();

    string displayString();
  };

  class Sin_ax : public SimpleFunction<double> {
    double _a, _b;
  public:
    Sin_ax(double a, double b=0);
    double value(double x);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    string displayString();
  };

  class Cos_ay : public SimpleFunction<double> {
    double _a;
  public:
    Cos_ay(double a);
    double value(double x, double y);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();

    string displayString();
  };

  class Sin_ay : public SimpleFunction<double> {
    double _a;
  public:
    Sin_ay(double a);
    double value(double x, double y);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    string displayString();
  };


  class Exp_ax : public SimpleFunction<double> {
    double _a;
  public:
    Exp_ax(double a);
    double value(double x, double y);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    string displayString();
  };

  class Exp_ay : public SimpleFunction<double> {
    double _a;
  public:
    Exp_ay(double a);
    double value(double x, double y);
    Teuchos::RCP<Function<double> > dx();
    Teuchos::RCP<Function<double> > dy();
    string displayString();
  };
}

#endif
