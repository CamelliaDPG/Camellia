#ifndef BASIS_FUNCTION_INTERFACE
#define BASIS_FUNCTION_INTERFACE

#include "AbstractFunction.h"

using namespace Intrepid;
using namespace std;

class BasisFunction : public AbstractFunction {
  int _basisOrdinalToReturn;
  Teuchos::RCP< Basis<double,FieldContainer<double> > > _basisPtr;
 public:
 BasisFunction(Teuchos::RCP< Basis<double,FieldContainer<double> > > basis) : AbstractFunction() {   
    _basisPtr = basis;
    _basisOrdinalToReturn = 1; // default to lowest order degree for now
  }
  
  void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints){
    FieldContainer<double> allFunctionValues;
    _basisPtr->getValues(allFunctionValues, physicalPoints, IntrepidExtendedTypes::OPERATOR_VALUE);   
    
  }
  
  void set(int ordinal){
    _basisOrdinalToReturn = ordinal;
  }
};

#endif
