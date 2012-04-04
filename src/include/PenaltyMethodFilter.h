#ifndef DPG_PENALTY_METHOD_FILTER
#define DPG_PENALTY_METHOD_FILTER

#include "Constraints.h"
#include "LocalStiffnessMatrixFilter.h"

class PenaltyMethodFilter : public LocalStiffnessMatrixFilter {
 private:
  Teuchos::RCP<Constraints> _constraints;
 public: 
  PenaltyMethodFilter(Teuchos::RCP<Constraints> constraints);
  virtual void filter(FieldContainer<double> &localStiffnessMatrix, FieldContainer<double> &localRHSVector, 
                      BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc) ;
};

#endif
