#ifndef PROJECTOR
#define PROJECTOR

//#include "DPGInnerProduct.h"
#include "AbstractFunction.h"
#include "IP.h"

#include "Basis.h"

class Function;

using namespace Intrepid;
using namespace std;

class Projector{
 public:

  // newest version:
  static void projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, Teuchos::RCP<Function> fxn, 
                                       BasisPtr basis, BasisCachePtr basisCache, IPPtr ip, VarPtr v,
                                       set<int>fieldIndicesToSkip = set<int>());
  
  // new version:
  static void projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, Teuchos::RCP<Function> fxn, 
                                       BasisPtr basis, BasisCachePtr basisCache);
  
  // old version:
  static void projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, Teuchos::RCP<AbstractFunction> fxn,
                                       BasisPtr basis, const FieldContainer<double> &physicalCellNodes);
  
  
  static void projectFunctionOntoBasisInterpolating(FieldContainer<double> &basisCoefficients, Teuchos::RCP<Function> fxn,
                                                    BasisPtr basis, BasisCachePtr domainBasisCache);
  
};
#endif
