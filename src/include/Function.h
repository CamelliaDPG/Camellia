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

  class Function {
  private:
    enum FunctionModificationType{ MULTIPLY, DIVIDE }; // private, used by scalarModify[.*]Values
  protected:
    int _rank;
    string _displayString; // this is here mostly for identifying functions in the debugger
    void CHECK_VALUES_RANK(Intrepid::FieldContainer<double> &values); // throws exception on bad values rank
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

    virtual void values(Intrepid::FieldContainer<double> &values, Camellia::EOperator op, BasisCachePtr basisCache);
    virtual void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) = 0;

    static FunctionPtr op(FunctionPtr f, Camellia::EOperator op);

    virtual FunctionPtr x();
    virtual FunctionPtr y();
    virtual FunctionPtr z();
    virtual FunctionPtr t(); // defined to be the last dimension in a space-time context

    virtual FunctionPtr dx();
    virtual FunctionPtr dy();
    virtual FunctionPtr dz();
    virtual FunctionPtr dt();

    virtual FunctionPtr div();
    virtual FunctionPtr curl();
    virtual FunctionPtr grad(int numComponents=-1);

    virtual void importCellData(std::vector<GlobalIndexType> cellIDs) {}

  // inverse() presently unused: and unclear how useful...
  //  virtual FunctionPtr inverse();

    int rank();

    virtual void addToValues(Intrepid::FieldContainer<double> &valuesToAddTo, BasisCachePtr basisCache);

    double integralOfJump(Teuchos::RCP<Mesh> mesh, GlobalIndexType cellID, int sideIndex, int cubatureDegreeEnrichment);

    double integralOfJump(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment);

    double integrate(BasisCachePtr basisCache);
    void integrate(Intrepid::FieldContainer<double> &cellIntegrals, BasisCachePtr basisCache, bool sumInto=false);

    // integrate over only one cell
    //  double integrate(int cellID, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0);
    double integrate(GlobalIndexType cellID, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false);

    // return all cell integrals
    map<int,double> cellIntegrals( Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false);
    // return cell integrals specified in input argument cellIDs
    map<int,double> cellIntegrals(vector<GlobalIndexType> cellIDs, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false);

    double integrate( Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false, bool requireSideCaches = false,
                      bool spatialSidesOnly = false);

    // adaptive quadrature
    double integrate(Teuchos::RCP<Mesh> mesh, double tol, bool testVsTest = false);

    bool isPositive(BasisCachePtr basisCache);
    bool isPositive(Teuchos::RCP<Mesh> mesh, int cubEnrich = 0, bool testVsTest = false);

    double l2norm(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool spatialSidesOnly = false);

    // divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
    virtual void scalarMultiplyFunctionValues(Intrepid::FieldContainer<double> &functionValues, BasisCachePtr basisCache);

    // divide values by this function (supported only when this is a scalar)
    virtual void scalarDivideFunctionValues(Intrepid::FieldContainer<double> &functionValues, BasisCachePtr basisCache);

    // divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
    virtual void scalarMultiplyBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache);

    // divide values by this function (supported only when this is a scalar)
    virtual void scalarDivideBasisValues(Intrepid::FieldContainer<double> &basisValues, BasisCachePtr basisCache);

    virtual void valuesDottedWithTensor(Intrepid::FieldContainer<double> &values,
                                        FunctionPtr tensorFunctionOfLikeRank,
                                        BasisCachePtr basisCache);

    virtual string displayString();

    void writeBoundaryValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath);
    void writeValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath);

    // Note that in general, repeated calls to Function::evaluate() would be significantly more expensive than a call with many points to Function::values().
    // Also, evaluate() may fail for certain Function subclasses, including any that depend on the Mesh.
    virtual double evaluate(double x);
    virtual double evaluate(double x, double y);
    virtual double evaluate(double x, double y, double z);

    virtual double evaluate(Teuchos::RCP<Mesh> mesh, double x);
    virtual double evaluate(Teuchos::RCP<Mesh> mesh, double x, double y);
    virtual double evaluate(Teuchos::RCP<Mesh> mesh, double x, double y, double z);

    static double evaluate(FunctionPtr f, double x); // for testing
    static double evaluate(FunctionPtr f, double x, double y); // for testing
    static double evaluate(FunctionPtr f, double x, double y, double z); // for testing

    static bool isNull(FunctionPtr f);

    // static Function construction methods:
    static FunctionPtr composedFunction( FunctionPtr f, FunctionPtr arg_g); // note: SLOW! avoid when possible...
    static FunctionPtr constant(double value);
    static FunctionPtr constant(vector<double> &value);

    static FunctionPtr min(FunctionPtr f1, FunctionPtr f2);
    static FunctionPtr min(FunctionPtr f1, double value);
    static FunctionPtr min(double value, FunctionPtr f2);
    static FunctionPtr max(FunctionPtr f1, FunctionPtr f2);
    static FunctionPtr max(FunctionPtr f1, double value);
    static FunctionPtr max(double value, FunctionPtr f2);

    static FunctionPtr h();
    // ! implements Heaviside step function, shifted right by xValue
    static FunctionPtr heaviside(double xValue);

    static FunctionPtr meshBoundaryCharacteristic(); // 1 on mesh boundary, 0 elsewhere
    static FunctionPtr meshSkeletonCharacteristic(); // 1 on mesh skeleton, 0 elsewhere
    static FunctionPtr polarize(FunctionPtr f);
    static FunctionPtr vectorize(FunctionPtr f1, FunctionPtr f2);
    static FunctionPtr vectorize(FunctionPtr f1, FunctionPtr f2, FunctionPtr f3);
    static FunctionPtr normal();    // unit outward-facing normal on each element boundary
    static FunctionPtr normal_1D(); // -1 at left side of element, +1 at right
    static FunctionPtr normalSpaceTime();
    static FunctionPtr null();
    static FunctionPtr sideParity();
    static FunctionPtr solution(VarPtr var, SolutionPtr soln);
    static FunctionPtr zero(int rank=0);
    static FunctionPtr restrictToCellBoundary(FunctionPtr f);

    static FunctionPtr xn(int n=1);
    static FunctionPtr yn(int n=1);
    static FunctionPtr zn(int n=1);
    static FunctionPtr tn(int n=1);
  //  static FunctionPtr jump(FunctionPtr f);

    static FunctionPtr cellCharacteristic(GlobalIndexType cellID);
    static FunctionPtr cellCharacteristic(set<GlobalIndexType> cellIDs);

    static FunctionPtr xPart(FunctionPtr vectorFunction);
    static FunctionPtr yPart(FunctionPtr vectorFunction);
    static FunctionPtr zPart(FunctionPtr vectorFunction);
  private:
    void scalarModifyFunctionValues(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache,
                                    FunctionModificationType modType);

    void scalarModifyBasisValues(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache,
                                 FunctionModificationType modType);
  };

  FunctionPtr operator*(FunctionPtr f1, FunctionPtr f2);
  FunctionPtr operator/(FunctionPtr f1, FunctionPtr scalarDivisor);
  FunctionPtr operator/(FunctionPtr f1, double divisor);
  FunctionPtr operator/(double value, FunctionPtr scalarDivisor);

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
}
#endif
