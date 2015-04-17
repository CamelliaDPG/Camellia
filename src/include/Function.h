//
//  Function.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Function_h
#define Camellia_Function_h

#include "CamelliaIntrepidExtendedTypes.h"
#include "TypeDefs.h"

#include "Intrepid_FieldContainer.hpp"

#include <string>

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

    bool equals(FunctionPtr<Scalar> f, BasisCachePtr basisCacheForCellsToCompare, double tol = 1e-14);

    virtual bool isZero() { return false; } // if true, the function is identically zero

    virtual bool boundaryValueOnly() { return false; } // if true, indicates a function defined only on element boundaries (mesh skeleton)

    virtual void values(Intrepid::FieldContainer<Scalar> &values, Camellia::EOperator op, BasisCachePtr basisCache);
    virtual void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) = 0;

    static FunctionPtr<Scalar> op(FunctionPtr<Scalar> f, Camellia::EOperator op);

    virtual FunctionPtr<Scalar> x();
    virtual FunctionPtr<Scalar> y();
    virtual FunctionPtr<Scalar> z();
    virtual FunctionPtr<Scalar> t(); // defined to be the last dimension in a space-time context

    virtual FunctionPtr<Scalar> dx();
    virtual FunctionPtr<Scalar> dy();
    virtual FunctionPtr<Scalar> dz();
    virtual FunctionPtr<Scalar> dt();

    virtual FunctionPtr<Scalar> div();
    virtual FunctionPtr<Scalar> curl();
    virtual FunctionPtr<Scalar> grad(int numComponents=-1);

    virtual void importCellData(std::vector<GlobalIndexType> cellIDs) {}

  // inverse() presently unused: and unclear how useful...
  //  virtual FunctionPtr<Scalar> inverse();

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
                                        FunctionPtr<Scalar> tensorFunctionOfLikeRank,
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

    static Scalar evaluate(FunctionPtr<Scalar> f, double x); // for testing
    static Scalar evaluate(FunctionPtr<Scalar> f, double x, double y); // for testing
    static Scalar evaluate(FunctionPtr<Scalar> f, double x, double y, double z); // for testing

    static bool isNull(FunctionPtr<Scalar> f);

    // static Function construction methods:
    static FunctionPtr<double> composedFunction( FunctionPtr<double> f, FunctionPtr<double> arg_g); // note: SLOW! avoid when possible...
    static FunctionPtr<Scalar> constant(Scalar value);
    static FunctionPtr<Scalar> constant(vector<Scalar> &value);

    static FunctionPtr<double> min(FunctionPtr<double> f1, FunctionPtr<double> f2);
    static FunctionPtr<double> min(FunctionPtr<double> f1, double value);
    static FunctionPtr<double> min(double value, FunctionPtr<double> f2);
    static FunctionPtr<double> max(FunctionPtr<double> f1, FunctionPtr<double> f2);
    static FunctionPtr<double> max(FunctionPtr<double> f1, double value);
    static FunctionPtr<double> max(double value, FunctionPtr<double> f2);

    static FunctionPtr<double> h();
    // ! implements Heaviside step function, shifted right by xValue
    static FunctionPtr<double> heaviside(double xValue);

    static FunctionPtr<double> meshBoundaryCharacteristic(); // 1 on mesh boundary, 0 elsewhere
    static FunctionPtr<double> meshSkeletonCharacteristic(); // 1 on mesh skeleton, 0 elsewhere
    static FunctionPtr<Scalar> polarize(FunctionPtr<Scalar> f);
    static FunctionPtr<Scalar> vectorize(FunctionPtr<Scalar> f1, FunctionPtr<Scalar> f2);
    static FunctionPtr<Scalar> vectorize(FunctionPtr<Scalar> f1, FunctionPtr<Scalar> f2, FunctionPtr<Scalar> f3);
    static FunctionPtr<double> normal();    // unit outward-facing normal on each element boundary
    static FunctionPtr<double> normal_1D(); // -1 at left side of element, +1 at right
    static FunctionPtr<double> normalSpaceTime();
    static FunctionPtr<Scalar> null();
    static FunctionPtr<double> sideParity();
    static FunctionPtr<Scalar> solution(VarPtr var, SolutionPtr<Scalar> soln);
    static FunctionPtr<double> zero(int rank=0);
    static FunctionPtr<Scalar> restrictToCellBoundary(FunctionPtr<Scalar> f);

    static FunctionPtr<double> xn(int n=1);
    static FunctionPtr<double> yn(int n=1);
    static FunctionPtr<double> zn(int n=1);
    static FunctionPtr<double> tn(int n=1);
  //  static FunctionPtr<Scalar> jump(FunctionPtr<Scalar> f);

    static FunctionPtr<double> cellCharacteristic(GlobalIndexType cellID);
    static FunctionPtr<double> cellCharacteristic(set<GlobalIndexType> cellIDs);

    static FunctionPtr<Scalar> xPart(FunctionPtr<Scalar> vectorFunction);
    static FunctionPtr<Scalar> yPart(FunctionPtr<Scalar> vectorFunction);
    static FunctionPtr<Scalar> zPart(FunctionPtr<Scalar> vectorFunction);
  private:
    void scalarModifyFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache,
                                    FunctionModificationType modType);

    void scalarModifyBasisValues(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache,
                                 FunctionModificationType modType);
  };

  template <typename Scalar>
  FunctionPtr<Scalar> operator*(FunctionPtr<Scalar> f1, FunctionPtr<Scalar> f2);
  template <typename Scalar>
  FunctionPtr<Scalar> operator/(FunctionPtr<Scalar> f1, FunctionPtr<Scalar> scalarDivisor);
  template <typename Scalar>
  FunctionPtr<Scalar> operator/(FunctionPtr<Scalar> f1, Scalar divisor);
  template <typename Scalar>
  FunctionPtr<Scalar> operator/(Scalar value, FunctionPtr<Scalar> scalarDivisor);

  template <typename Scalar>
  FunctionPtr<Scalar> operator*(Scalar weight, FunctionPtr<Scalar> f);
  template <typename Scalar>
  FunctionPtr<Scalar> operator*(FunctionPtr<Scalar> f, Scalar weight);
  template <typename Scalar>
  FunctionPtr<Scalar> operator*(vector<Scalar> weight, FunctionPtr<Scalar> f);
  template <typename Scalar>
  FunctionPtr<Scalar> operator*(FunctionPtr<Scalar> f, vector<Scalar> weight);

  template <typename Scalar>
  FunctionPtr<Scalar> operator+(FunctionPtr<Scalar> f1, FunctionPtr<Scalar> f2);
  template <typename Scalar>
  FunctionPtr<Scalar> operator+(FunctionPtr<Scalar> f1, Scalar value);
  template <typename Scalar>
  FunctionPtr<Scalar> operator+(Scalar value, FunctionPtr<Scalar> f1);

  template <typename Scalar>
  FunctionPtr<Scalar> operator-(FunctionPtr<Scalar> f1, FunctionPtr<Scalar> f2);
  template <typename Scalar>
  FunctionPtr<Scalar> operator-(FunctionPtr<Scalar> f1, Scalar value);
  template <typename Scalar>
  FunctionPtr<Scalar> operator-(Scalar value, FunctionPtr<Scalar> f1);

  template <typename Scalar>
  FunctionPtr<Scalar> operator-(FunctionPtr<Scalar> f);
}
#endif
