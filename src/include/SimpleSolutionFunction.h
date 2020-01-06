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

namespace Camellia
{
template <typename Scalar>
class SimpleSolutionFunction : public TFunction<Scalar>
{
  TSolutionPtr<Scalar> _soln;
  VarPtr _var;
  bool _weightFluxesBySideParity;
  int _solutionOrdinal;
  std::string _solutionIdentifierExponent; // used in displayString (allows disambiguation of solutions; if non-empty means we'll drop the \overline business)
public:
  SimpleSolutionFunction(VarPtr var, TSolutionPtr<Scalar> soln, bool weightFluxesBySideParity, int solutionOrdinal, const std::string &solutionIdentifierExponent="");
  void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  TFunctionPtr<Scalar> x();
  TFunctionPtr<Scalar> y();
  TFunctionPtr<Scalar> z();

  TFunctionPtr<Scalar> dx();
  TFunctionPtr<Scalar> dy();
  TFunctionPtr<Scalar> dz();
  // for reasons of efficiency, may want to implement div() and grad() as well

  void importCellData(std::vector<GlobalIndexType> cellIDs);

  size_t getCellDataSize(GlobalIndexType cellID); // size in bytes
  void packCellData(GlobalIndexType cellID, char* cellData, size_t bufferLength);
  size_t unpackCellData(GlobalIndexType cellID, const char* cellData, size_t bufferLength); // returns bytes consumed
  
  std::string displayString();
  bool boundaryValueOnly();
};
}

#endif
