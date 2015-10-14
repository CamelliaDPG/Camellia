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

namespace Camellia
{
template <typename Scalar>
class TFunction
{
private:
  enum FunctionModificationType { MULTIPLY, DIVIDE }; // private, used by scalarModify[.*]Values
  
  Scalar evaluateAtMeshPoint(MeshPtr mesh, GlobalIndexType cellID, Intrepid::FieldContainer<double> &physicalPoint);
protected:
  int _rank;
  string _displayString; // this is here mostly for identifying functions in the debugger
  void CHECK_VALUES_RANK(Intrepid::FieldContainer<Scalar> &values); // throws exception on bad values rank
  double _time;
public:
  TFunction();
  TFunction(int rank);
  virtual ~TFunction() {}

  virtual void setTime(double time);
  virtual double getTime();

  bool equals(TFunctionPtr<Scalar> f, BasisCachePtr basisCacheForCellsToCompare, double tol = 1e-14);

  virtual bool isZero()
  {
    return false;  // if true, the function is identically zero
  }

  virtual bool boundaryValueOnly()
  {
    return false;  // if true, indicates a function defined only on element boundaries (mesh skeleton)
  }

  virtual void values(Intrepid::FieldContainer<Scalar> &values, Camellia::EOperator op, BasisCachePtr basisCache);
  virtual void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) = 0;

  static TFunctionPtr<Scalar> op(TFunctionPtr<Scalar> f, Camellia::EOperator op);

  virtual TFunctionPtr<Scalar> x();
  virtual TFunctionPtr<Scalar> y();
  virtual TFunctionPtr<Scalar> z();
  virtual TFunctionPtr<Scalar> t(); // defined to be the last dimension in a space-time context
  virtual TFunctionPtr<Scalar> spatialComponent(int d); // 1 for x(), 2 for y(), 3 for z().

  virtual TFunctionPtr<Scalar> dx();
  virtual TFunctionPtr<Scalar> dy();
  virtual TFunctionPtr<Scalar> dz();
  virtual TFunctionPtr<Scalar> dt();

  virtual TFunctionPtr<Scalar> div();
  virtual TFunctionPtr<Scalar> curl();
  virtual TFunctionPtr<Scalar> grad(int numComponents=-1);

  virtual void importCellData(std::vector<GlobalIndexType> cellIDs) {}

  // inverse() presently unused: and unclear how useful...
  //  virtual TFunctionPtr<Scalar> inverse();

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
                                      TFunctionPtr<Scalar> tensorFunctionOfLikeRank,
                                      BasisCachePtr basisCache);

  virtual string displayString();

  void writeBoundaryValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath);
  void writeValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath);

  // Note that in general, repeated calls to Function::evaluate() would be significantly more expensive than a call with many points to Function::values().
  // Also, evaluate() may fail for certain Function subclasses, including any that depend on the Mesh.
  virtual Scalar evaluate(double x);
  virtual Scalar evaluate(double x, double y);
  virtual Scalar evaluate(double x, double y, double z);

  // ! MPI-collective method. Evaluates this on the rank that owns the cell that matches the specified point.
  virtual Scalar evaluate(Teuchos::RCP<Mesh> mesh, double x);
  // ! MPI-collective method. Evaluates this on the rank that owns the cell that matches the specified point.
  virtual Scalar evaluate(Teuchos::RCP<Mesh> mesh, double x, double y);
  // ! MPI-collective method. Evaluates this on the rank that owns the cell that matches the specified point.
  virtual Scalar evaluate(Teuchos::RCP<Mesh> mesh, double x, double y, double z);

  static Scalar evaluate(TFunctionPtr<Scalar> f, double x); // for testing
  static Scalar evaluate(TFunctionPtr<Scalar> f, double x, double y); // for testing
  static Scalar evaluate(TFunctionPtr<Scalar> f, double x, double y, double z); // for testing

  static bool isNull(TFunctionPtr<Scalar> f);

  // static Function construction methods:
  static TFunctionPtr<double> composedFunction( TFunctionPtr<double> f, TFunctionPtr<double> arg_g); // note: SLOW! avoid when possible...
  static TFunctionPtr<Scalar> constant(Scalar value);
  static TFunctionPtr<Scalar> constant(vector<Scalar> value);

  static TFunctionPtr<double> min(TFunctionPtr<double> f1, TFunctionPtr<double> f2);
  static TFunctionPtr<double> min(TFunctionPtr<double> f1, double value);
  static TFunctionPtr<double> min(double value, TFunctionPtr<double> f2);
  static TFunctionPtr<double> max(TFunctionPtr<double> f1, TFunctionPtr<double> f2);
  static TFunctionPtr<double> max(TFunctionPtr<double> f1, double value);
  static TFunctionPtr<double> max(double value, TFunctionPtr<double> f2);

  static TFunctionPtr<double> h();
  // ! implements Heaviside step function, shifted right by xValue
  static TFunctionPtr<double> heaviside(double xValue);
  static TFunctionPtr<double> heavisideY(double yValue);

  static TFunctionPtr<double> meshBoundaryCharacteristic(); // 1 on mesh boundary, 0 elsewhere
  static TFunctionPtr<double> meshSkeletonCharacteristic(); // 1 on mesh skeleton, 0 elsewhere
  static TFunctionPtr<Scalar> polarize(TFunctionPtr<Scalar> f);
  static TFunctionPtr<Scalar> vectorize(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
  static TFunctionPtr<Scalar> vectorize(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2, TFunctionPtr<Scalar> f3);
  static TFunctionPtr<Scalar> vectorize(std::vector<TFunctionPtr<Scalar>> components);
  static TFunctionPtr<double> normal();    // unit outward-facing normal on each element boundary
  static TFunctionPtr<double> normal_1D(); // -1 at left side of element, +1 at right
  static TFunctionPtr<double> normalSpaceTime();
  static TFunctionPtr<Scalar> null();
  static TFunctionPtr<double> sideParity();

  // ! Will throw an exception if var is a flux variable (should call the one with the boolean weightFluxesBySideParity argument in this case)
  static TFunctionPtr<Scalar> solution(VarPtr var, TSolutionPtr<Scalar> soln);
  // ! When weightFluxesBySideParity = true, the solution function will be non-uniquely-valued
  static TFunctionPtr<Scalar> solution(VarPtr var, TSolutionPtr<Scalar> soln, bool weightFluxesBySideParity);
  static TFunctionPtr<double> zero(int rank=0);
  static TFunctionPtr<Scalar> restrictToCellBoundary(TFunctionPtr<Scalar> f);

  static TFunctionPtr<double> xn(int n=1);
  static TFunctionPtr<double> yn(int n=1);
  static TFunctionPtr<double> zn(int n=1);
  static TFunctionPtr<double> tn(int n=1);
  //  static TFunctionPtr<Scalar> jump(TFunctionPtr<Scalar> f);

  static TFunctionPtr<double> cellCharacteristic(GlobalIndexType cellID);
  static TFunctionPtr<double> cellCharacteristic(set<GlobalIndexType> cellIDs);

  static TFunctionPtr<Scalar> xPart(TFunctionPtr<Scalar> vectorFunction);
  static TFunctionPtr<Scalar> yPart(TFunctionPtr<Scalar> vectorFunction);
  static TFunctionPtr<Scalar> zPart(TFunctionPtr<Scalar> vectorFunction);
private:
  void scalarModifyFunctionValues(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache,
                                  FunctionModificationType modType);

  void scalarModifyBasisValues(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache,
                               FunctionModificationType modType);
};

template <typename Scalar>
TFunctionPtr<Scalar> operator*(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
template <typename Scalar>
TFunctionPtr<Scalar> operator/(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> scalarDivisor);
template <typename Scalar>
TFunctionPtr<Scalar> operator/(TFunctionPtr<Scalar> f1, Scalar divisor);
template <typename Scalar>
TFunctionPtr<Scalar> operator/(TFunctionPtr<Scalar> f1, int divisor);
template <typename Scalar>
TFunctionPtr<Scalar> operator/(Scalar value, TFunctionPtr<Scalar> scalarDivisor);
template <typename Scalar>
TFunctionPtr<Scalar> operator/(int value, TFunctionPtr<Scalar> scalarDivisor);

template <typename Scalar>
TFunctionPtr<Scalar> operator*(int weight, TFunctionPtr<Scalar> f);
template <typename Scalar>
TFunctionPtr<Scalar> operator*(Scalar weight, TFunctionPtr<Scalar> f);
template <typename Scalar>
TFunctionPtr<Scalar> operator*(TFunctionPtr<Scalar> f, Scalar weight);
template <typename Scalar>
TFunctionPtr<Scalar> operator*(TFunctionPtr<Scalar> f, int weight);
template <typename Scalar>
TFunctionPtr<Scalar> operator*(vector<Scalar> weight, TFunctionPtr<Scalar> f);
template <typename Scalar>
TFunctionPtr<Scalar> operator*(TFunctionPtr<Scalar> f, vector<Scalar> weight);

template <typename Scalar>
TFunctionPtr<Scalar> operator+(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
template <typename Scalar>
TFunctionPtr<Scalar> operator+(TFunctionPtr<Scalar> f1, Scalar value);
template <typename Scalar>
TFunctionPtr<Scalar> operator+(TFunctionPtr<Scalar> f1, int value);
template <typename Scalar>
TFunctionPtr<Scalar> operator+(Scalar value, TFunctionPtr<Scalar> f1);
template <typename Scalar>
TFunctionPtr<Scalar> operator+(int value, TFunctionPtr<Scalar> f1);

template <typename Scalar>
TFunctionPtr<Scalar> operator-(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2);
template <typename Scalar>
TFunctionPtr<Scalar> operator-(TFunctionPtr<Scalar> f1, Scalar value);
template <typename Scalar>
TFunctionPtr<Scalar> operator-(Scalar value, TFunctionPtr<Scalar> f1);
template <typename Scalar>
TFunctionPtr<Scalar> operator-(int value, TFunctionPtr<Scalar> f1);
template <typename Scalar>
TFunctionPtr<Scalar> operator-(TFunctionPtr<Scalar> f1, int value);

template <typename Scalar>
TFunctionPtr<Scalar> operator-(TFunctionPtr<Scalar> f);

extern template class TFunction<double>;
}
#endif
