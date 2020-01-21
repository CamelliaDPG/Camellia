// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

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

#include "Camellia_FieldContainer.hpp"

#include <string>

using namespace std;

namespace Camellia
{
template <typename Scalar>
class TFunction
{
private:
  enum FunctionModificationType { MULTIPLY, DIVIDE }; // private, used by scalarModify[.*]Values
  
  Scalar evaluateAtMeshPoint(MeshPtr mesh, GlobalIndexType cellID, Camellia::FieldContainer<double> &physicalPoint);
protected:
  int _rank;
  string _displayString; // this is here mostly for identifying functions in the debugger
  void CHECK_VALUES_RANK(Camellia::FieldContainer<Scalar> &values); // throws exception on bad values rank
  double _time;
public:
  enum JumpCombinationType
  {
    DIFFERENCE,
    SUM
  };
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

  virtual void values(Camellia::FieldContainer<Scalar> &values, Camellia::EOperator op, BasisCachePtr basisCache);
  virtual void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache) = 0;

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
  virtual TFunctionPtr<Scalar> di(int i); // 1 for dx(), 2 for dy(), 3 for dz()

  virtual TFunctionPtr<Scalar> div();
  virtual TFunctionPtr<Scalar> curl();
  virtual TFunctionPtr<Scalar> grad(int numComponents=-1);
  virtual TFunctionPtr<Scalar> hessian(int spaceDim); // grad grad (increases rank by 2)
  
  // ! For Functions that include VarFunctions among their members, takes Jacobian relative to corresponding variables,
  // ! using the solution provided to determine where the Jacobian is evaluated.
  // ! (For Functions that do not include VarFunctions among their members, returns zero.)
  virtual TLinearTermPtr<Scalar> jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap);

  virtual void importCellData(std::vector<GlobalIndexType> cellIDs) {}

  // inverse() presently unused: and unclear how useful...
  //  virtual TFunctionPtr<Scalar> inverse();

  int rank();

  virtual void addToValues(Camellia::FieldContainer<Scalar> &valuesToAddTo, BasisCachePtr basisCache);

  Scalar integralOfJump(Teuchos::RCP<Mesh> mesh, GlobalIndexType cellID, int sideIndex, int cubatureDegreeEnrichment);

  Scalar integralOfJump(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment);

  Scalar integrate(BasisCachePtr basisCache);
  void integrate(Camellia::FieldContainer<Scalar> &cellIntegrals, BasisCachePtr basisCache, bool sumInto=false);

  // integrate over only one cell
  //  double integrate(int cellID, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0);
  Scalar integrate(GlobalIndexType cellID, Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false);

  // ! Returns true if the Function guarantees that the value is zero at each point in the BasisCache
  // ! A false value does not necessarily mean that the Function is anywhere nonzero.
  virtual bool isZero(BasisCachePtr basisCache);
  
  Scalar integrate( Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool testVsTest = false, bool requireSideCaches = false,
                    bool spatialSidesOnly = false);

  // adaptive quadrature
  Scalar integrate(Teuchos::RCP<Mesh> mesh, double tol, bool testVsTest = false);

  bool isPositive(BasisCachePtr basisCache);
  bool isPositive(Teuchos::RCP<Mesh> mesh, int cubEnrich = 0, bool testVsTest = false);
  
  double l1norm(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool spatialSidesOnly = false);
  double l2norm(Teuchos::RCP<Mesh> mesh, int cubatureDegreeEnrichment = 0, bool spatialSidesOnly = false);
  double linfinitynorm(MeshPtr mesh, int cubatureDegreeEnrichment = 0);
  
  // ! returns the maximum value at quadrature points.  Performs MPI communication.
  double maximumValue(MeshPtr mesh, int cubatureDegreeEnrichment = 0);
  
  // ! returns the minimum value at quadrature points.  Performs MPI communication.
  double minimumValue(MeshPtr mesh, int cubatureDegreeEnrichment = 0);
  
  // ! returns locally-owned cells which take a negative value at quadrature points.  Does not perform MPI communication.
  std::vector<GlobalIndexType> localCellsWithNegativeValues(MeshPtr mesh, int cubatureDegreeEnrichment = 0);
  
  // ! computes the squared L^2 norm of the jumps of this Function along the mesh skeleton (external boundary values are treated as zero, so the jump
  // ! along the boundary is the value of the Function itself)
  // ! if weightBySideMeasure is true, then the edge lengths (2D) or face area will be used to weight the integral of the squared jump.
  // ! cubatureDegreeEnrichment is relative to the H^1 order of the mesh's trial space.
  // ! Returned map will have as keys the cell IDs of MPI-local cells.
  std::map<GlobalIndexType, double> squaredL2NormOfJumps(MeshPtr mesh, bool weightBySideMeasure, int cubatureDegreeEnrichment, JumpCombinationType jumpCombination=DIFFERENCE);
  
  // divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
  virtual void scalarMultiplyFunctionValues(Camellia::FieldContainer<Scalar> &functionValues, BasisCachePtr basisCache);

  // divide values by this function (supported only when this is a scalar)
  virtual void scalarDivideFunctionValues(Camellia::FieldContainer<Scalar> &functionValues, BasisCachePtr basisCache);

  // divide values by this function (supported only when this is a scalar--otherwise values would change rank...)
  virtual void scalarMultiplyBasisValues(Camellia::FieldContainer<double> &basisValues, BasisCachePtr basisCache);

  // divide values by this function (supported only when this is a scalar)
  virtual void scalarDivideBasisValues(Camellia::FieldContainer<double> &basisValues, BasisCachePtr basisCache);

  virtual void valuesDottedWithTensor(Camellia::FieldContainer<Scalar> &values,
                                      TFunctionPtr<Scalar> tensorFunctionOfLikeRank,
                                      BasisCachePtr basisCache);

  virtual string displayString();

  void writeBoundaryValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath);
  void writeValuesToMATLABFile(Teuchos::RCP<Mesh> mesh, const string &filePath);

  // ! returns true if this Function involves Vars.
  virtual bool isAbstract();
  
//  // ! For abstract function provided, returns the concrete function by evaluating Vars using Solution provided
//  // ! If the function provided is not abstract, just returns the function.
//  static TFunctionPtr<Scalar> evaluateAt(TFunctionPtr<Scalar> abstractFunction, SolutionPtr soln); // calls the version that takes a map argument
  
  // ! For abstract function provided, returns the concrete function by evaluating Vars using Solution provided
  // ! If the function provided is not abstract, just returns the function.
  // ! valueMap has keys corresponding to varID, values the function for that var.  Missing entries are by convention taken to be zero functions.
  static TFunctionPtr<Scalar> evaluateFunctionAt(TFunctionPtr<Scalar> abstractFunction, const map<int, TFunctionPtr<Scalar> > &valueMap);
  
  // ! the static evaluateFunctionAt is generally preferred, as it can avoid copying the Function when it is not abstract...
  // ! valueMap has keys corresponding to varID, values the function for that var.  Missing entries are by convention taken to be zero functions.
  virtual TFunctionPtr<Scalar> evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap); // calls the version that takes a map argument
  
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

  // ! Data packing mechanism for mesh-dependent Function subclasses.  Default implementations provide no data; mesh-dependent subclasses should override.
  virtual size_t getCellDataSize(GlobalIndexType cellID); // size in bytes
  virtual void   packCellData(GlobalIndexType cellID, char* cellData, size_t bufferLength);
  virtual size_t unpackCellData(GlobalIndexType cellID, const char* cellData, size_t bufferLength);
  
  // ! Data packing helper methods for Function subclasses that compose other Function subclasses, which may in turn be mesh-dependent.
  static size_t getCellDataSize(const std::vector<FunctionPtr> &functions, GlobalIndexType cellID);
  static void   packCellData(const std::vector<FunctionPtr> &functions, GlobalIndexType cellID, char* cellData, size_t bufferLength);
  static size_t unpackCellData(const std::vector<FunctionPtr> &functions,GlobalIndexType cellID, const char* cellData, size_t bufferLength);
  
  // ! Functions that are compositions of other Functions should implement this...
  // ! (Used initially in data migration logic for mesh-dependent Functions.)
  virtual std::vector< TFunctionPtr<Scalar> > memberFunctions();
  
  void importDataForOffRankCells(MeshPtr mesh, const std::set<GlobalIndexType> &offRankCells);
  
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
  
  //! neighborFunction is defined on the mesh skeleton; evaluates f from the point of view of the neighbor on each face.
  //! on the mesh boundary, simply computes f on the (interior) element.
  //! Note: for data-dependent f, importCellData() should be called prior to evaluation, passing in the cells whose neighbor values will
  //!       be computed.  This allows appropriate MPI communication to occur.
  //! Typically, this means doing something like: neighborFunction->importCellData(mesh->cellIDsInPartition()) after the data has changed.
  static TFunctionPtr<Scalar> neighborFunction(TFunctionPtr<Scalar> f, MeshPtr mesh);
  
  // ! Square root of the specified function.
  static TFunctionPtr<Scalar> sqrtFunction(TFunctionPtr<Scalar> f);

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
  static TFunctionPtr<Scalar> solution(VarPtr var, TSolutionPtr<Scalar> soln, const std::string &solutionIdentifierExponent="");
  // ! When weightFluxesBySideParity = true, the solution function will be non-uniquely-valued
  static TFunctionPtr<Scalar> solution(VarPtr var, TSolutionPtr<Scalar> soln, bool weightFluxesBySideParity, const std::string &solutionIdentifierExponent="");
  // ! Use this for solutions with multiple RHSes (including influence computations)
  static TFunctionPtr<Scalar> solution(VarPtr var, TSolutionPtr<Scalar> soln, bool weightFluxesBySideParity, int solutionOrdinal, const std::string &solutionIdentifierExponent="");
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
  void scalarModifyFunctionValues(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache,
                                  FunctionModificationType modType);

  void scalarModifyBasisValues(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache,
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
