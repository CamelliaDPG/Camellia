#ifndef BASIS_SUM_FUNCTION
#define BASIS_SUM_FUNCTION

// Teuchos includes
#include "Teuchos_RCP.hpp"

// Intrepid includes
#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"

#include "BasisFactory.h"
#include "Function.h"
#include "BasisCache.h"

#include "AbstractFunction.h"

using namespace Intrepid;
using namespace std;

// NewBasisSumFunction is meant to replace the old, but it's not working yet.

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
    TEUCHOS_TEST_FOR_EXCEPTION(_coefficients.dimension(0)!=basis->getCardinality(),std::invalid_argument,"BasisSumFunction: coefficients passed in do not match cardinality of basis.");
  }
  virtual void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints);
};

class NewBasisSumFunction : public Function {
 private:  
  BasisPtr _basis;
  FieldContainer<double> _coefficients;
  EOperatorExtended _op;
  bool _boundaryValueOnly;
 public:
  NewBasisSumFunction(BasisPtr basis, const FieldContainer<double> &basisCoefficients,
                      EOperatorExtended op = OP_VALUE, bool boundaryValueOnly = false) : Function( BasisFactory::getBasisRank(basis) ) {
    _coefficients = basisCoefficients;
    _boundaryValueOnly = boundaryValueOnly;
    _basis = basis; // note - _basis->getBaseCellTopology
    _op = op;
    int cardinality = basis->getCardinality();
    TEUCHOS_TEST_FOR_EXCEPTION( _coefficients.dimension(0) != cardinality,
                               std::invalid_argument,
                               "BasisSumFunction: coefficients passed in do not match cardinality of basis.");
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache);
  
  FunctionPtr x();
  FunctionPtr y();
  FunctionPtr z();
  
  FunctionPtr dx();
  FunctionPtr dy();
  FunctionPtr dz();
  
  bool boundaryValueOnly();
  
  static FunctionPtr basisSumFunction(BasisPtr basis, const FieldContainer<double> &basisCoefficients);
};

#endif
