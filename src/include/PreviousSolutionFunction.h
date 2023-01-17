// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  PreviousSolutionFunction.h
//  Camellia
//
//  Created by Nathan Roberts on 4/5/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_PreviousSolutionFunction_h
#define Camellia_PreviousSolutionFunction_h

#include "TypeDefs.h"

#include "Function.h"
#include "Element.h"
#include "Solution.h"
#include "InnerProductScratchPad.h"

namespace Camellia
{
template <typename Scalar>
class PreviousSolutionFunction : public TFunction<Scalar>
{
  bool _overrideMeshCheck;
  TSolutionPtr<Scalar> _soln;
  LinearTermPtr _solnExpression;
  SolutionComponent _solnComponent;
  int _stageNumber;
public:
  PreviousSolutionFunction(TSolutionPtr<Scalar> soln, LinearTermPtr solnExpression, bool multiplyFluxesByCellParity = true,
                           SolutionComponent solutionComponent = STANDARD_SOLUTION, int stageNumber = 0);
  PreviousSolutionFunction(TSolutionPtr<Scalar> soln, VarPtr var, bool multiplyFluxesByCellParity = true,
                           SolutionComponent solutionComponent = STANDARD_SOLUTION, int stageNumber = 0);
  bool boundaryValueOnly();
  void setOverrideMeshCheck(bool value, bool dontWarn=false);
  
  using TFunction<Scalar>::values;
  void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  static map<int, TFunctionPtr<Scalar> > functionMap( vector< VarPtr > varPtrs, TSolutionPtr<Scalar> soln, SolutionComponent solutionComponent, int stageNumber, bool weightFluxesByParity = true);
  string displayString();
  
  void importCellData(std::vector<GlobalIndexType> cellIDs);
  
  size_t getCellDataSize(GlobalIndexType cellID); // size in bytes
  void packCellData(GlobalIndexType cellID, char* cellData, size_t bufferLength);
  size_t unpackCellData(GlobalIndexType cellID, const char* cellData, size_t bufferLength); // returns bytes consumed
};
}


#endif
