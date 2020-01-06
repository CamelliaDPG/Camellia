// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#ifndef DPG_PENALTY_METHOD_FILTER
#define DPG_PENALTY_METHOD_FILTER

#include "Constraints.h"
#include "LocalStiffnessMatrixFilter.h"

namespace Camellia
{
class PenaltyMethodFilter : public LocalStiffnessMatrixFilter
{
private:
  Teuchos::RCP<Constraints> _constraints;
public:
  PenaltyMethodFilter(Teuchos::RCP<Constraints> constraints);
  virtual void filter(Camellia::FieldContainer<double> &localStiffnessMatrix, Camellia::FieldContainer<double> &localRHSVector,
                      BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc) ;
};
}

#endif
