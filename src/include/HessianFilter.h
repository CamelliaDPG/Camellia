#ifndef HESSIAN_FILTER
#define HESSIAN_FILTER

#include "TypeDefs.h"

#include "LocalStiffnessMatrixFilter.h"
#include "BF.h" // has linearTerm,varfactory, and used to define the Hessian bilinear form

class HessianFilter : public LocalStiffnessMatrixFilter {
 private:
  BFPtr _hessianBF;
  VarFactory hessianVarFactory;
 public: 
  HessianFilter(BFPtr hessianBF ) {
    _hessianBF = hessianBF;
  };
  
  virtual void filter(FieldContainer<double> &localStiffnessMatrix, FieldContainer<double> &localRHSVector,
                      BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc) ;
};

#endif
