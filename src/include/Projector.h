#ifndef PROJECTOR
#define PROJECTOR

//#include "DPGInnerProduct.h"
#include "AbstractFunction.h"
#include "Function.h"
#include "IP.h"

using namespace Intrepid;
using namespace std;

class Projector{
 public:

  // newest version:
  static void projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, Teuchos::RCP<Function> fxn, 
                                       Teuchos::RCP< Basis<double,FieldContainer<double> > > basis, 
                                       BasisCachePtr basisCache, IPPtr ip, VarPtr v);
  
  // new version:
  static void projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, Teuchos::RCP<Function> fxn, 
                                       Teuchos::RCP< Basis<double,FieldContainer<double> > > basis, 
                                       BasisCachePtr basisCache);
  
  // old version:
  static void projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, Teuchos::RCP<AbstractFunction> fxn, 
                                       Teuchos::RCP< Basis<double,FieldContainer<double> > > basis, 
                                       const FieldContainer<double> &physicalCellNodes);
};
#endif
