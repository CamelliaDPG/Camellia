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
public:
  BCFunction(BCPtr bc, int varID);
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  bool imposeOnCell(int cellIndex);
  int varID();
};

#endif