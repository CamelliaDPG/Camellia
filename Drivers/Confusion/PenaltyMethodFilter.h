#ifndef DPG_PENALTY_METHOD_FILTER
#define DPG_PENALTY_METHOD_FILTER

#include "Constraints.h"
#include "LocalStiffnessMatrixFilter.h"


class PenaltyMethodFilter : public LocalStiffnessMatrixFilter {
 private:
  Constraints _constraints;
  /*  
  Teuchos::RCP<Mesh> _meshPtr;
  Teuchos::RCP<BC> _bcPtr;
  */
 public: 
  PenaltyMethodFilter(Constraints constraints);
  void filter(FieldContainer<double> &localStiffnessMatrix, 
	      const FieldContainer<double> &physicalCellNodes,
	      vector<int> &cellIDs, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc);
};

#endif
