#ifndef PROJECTOR
#define PROJECTOR

#include "AbstractFunction.h"
#include "IP.h"

#include "Basis.h"

class Function;

typedef Teuchos::RCP<Function> FunctionPtr;

class Projector {
 public:

  // newest version:
  static void projectFunctionOntoBasis(Intrepid::FieldContainer<double> &basisCoefficients,
                                       FunctionPtr fxn, BasisPtr basis, BasisCachePtr basisCache,
                                       IPPtr ip, VarPtr v,
                                       std::set<int>fieldIndicesToSkip = std::set<int>());
  
  // new version:
  static void projectFunctionOntoBasis(Intrepid::FieldContainer<double> &basisCoefficients,
                                       FunctionPtr fxn, BasisPtr basis, BasisCachePtr basisCache);
  
  // old version:
  static void projectFunctionOntoBasis(Intrepid::FieldContainer<double> &basisCoefficients, Teuchos::RCP<AbstractFunction> fxn,
                                       BasisPtr basis, const Intrepid::FieldContainer<double> &physicalCellNodes);
  
  
  static void projectFunctionOntoBasisInterpolating(Intrepid::FieldContainer<double> &basisCoefficients,
                                                    FunctionPtr fxn, BasisPtr basis, BasisCachePtr domainBasisCache);
};
#endif