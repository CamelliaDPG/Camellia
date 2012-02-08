#ifndef BASIS_SUM_FUNCTION
#define BASIS_SUM_FUNCTION

// Teuchos includes
#include "Teuchos_RCP.hpp"

// Intrepid includes
#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"

using namespace Intrepid;
using namespace std;

typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;

class BasisSumFunction : public AbstractFunction {
 private:  
  BasisPtr _basis;
  FieldContainer<double> _coefficients;
  FieldContainer<double> _physicalCellNodes;    
 public:
  BasisSumFunction(BasisPtr basis, const FieldContainer<double> &basisCoefficients, const FieldContainer<double> &physicalCellNodes){
    _coefficients = basisCoefficients;
    _basis = basis; // note - _basis->getBaseCellTopology
    _physicalCellNodes = physicalCellNodes; // note - rank 3, but dim(0) = 1
  }
  virtual void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints);
};

#endif
