// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  SimpleSolutionFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_SimpleSolutionFunction_h
#define Camellia_SimpleSolutionFunction_h

#include "Function.h"
#include "Solution.h" // required for SolutionComponent enumeration

namespace Camellia
{
template <typename Scalar>
class SimpleSolutionFunction : public TFunction<Scalar>
{
  TSolutionPtr<Scalar> _soln;
  VarPtr _var;
  bool _weightFluxesBySideParity;
  SolutionComponent _solutionComponent;
  int _solutionStage;
  std::string _solutionIdentifierExponent; // used in displayString (allows disambiguation of solutions; if non-empty means we'll drop the \overline business)
public:
  SimpleSolutionFunction(VarPtr var, TSolutionPtr<Scalar> soln, bool weightFluxesBySideParity, SolutionComponent solutionComponent=STANDARD_SOLUTION, int solutionStage=0, const std::string &solutionIdentifierExponent="");
  
  using TFunction<Scalar>::values;
  void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache) override;
  
  // device path:
  void values(DeviceView<Scalar> &values, ValueModificationType modType, BasisCachePtr basisCache) override;
  
  //! Returns true if the variable is defined on the given portion of the reference cell.  (False for L^2 volumetric variables on sides and trace variables on volumes, and for some trace variables that are only defined on certain sides.)
  bool isDefined(BasisCachePtr basisCache) const override;
  
  TFunctionPtr<Scalar> x() override;
  TFunctionPtr<Scalar> y() override;
  TFunctionPtr<Scalar> z() override;
  int numVectorComponents() const override;

  TFunctionPtr<Scalar> dx() override;
  TFunctionPtr<Scalar> dy() override;
  TFunctionPtr<Scalar> dz() override;
  
  TFunctionPtr<Scalar> di(int i) override;
  
  TFunctionPtr<Scalar> div() override;
  TFunctionPtr<Scalar> grad(int spaceDim=-1) override;
  TFunctionPtr<Scalar> curl() override;

  void importCellData(std::vector<GlobalIndexType> cellIDs) override;

  size_t getCellDataSize(GlobalIndexType cellID) override; // size in bytes
  void packCellData(GlobalIndexType cellID, char* cellData, size_t bufferLength) override;
  size_t unpackCellData(GlobalIndexType cellID, const char* cellData, size_t bufferLength) override; // returns bytes consumed
  
  std::string displayString() override;
  bool boundaryValueOnly() override;
};
}

#endif
