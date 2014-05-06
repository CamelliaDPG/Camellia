#ifndef CAMELLIA_BC_FUNCTION
#define CAMELLIA_BC_FUNCTION

#include "Intrepid_FieldContainer.hpp"
#include "BasisCache.h"
#include "Projector.h"

#include "Function.h"

using namespace Intrepid;

class BC;
typedef Teuchos::RCP<BC> BCPtr;

class BCFunction : public Function {
  FieldContainer<bool> _imposeHere;
  int _varID;
  BCPtr _bc;
  bool _isTrace; // if false, it's a flux...
  FunctionPtr _spatiallyFilteredFunction;
public:
  BCFunction(BCPtr bc, int varID, bool isTrace, FunctionPtr spatiallyFilteredFunction, int rank);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  bool imposeOnCell(int cellIndex);
  int varID();
  bool isTrace();
  
  FunctionPtr curl();
  FunctionPtr div();
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
  
  static Teuchos::RCP<BCFunction> bcFunction(BCPtr bc, int varID, bool isTrace);
};

#endif
