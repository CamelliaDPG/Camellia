#include "PenaltyMethodFilter.h"
#include "Mesh.h"

// Intrepid includes - for numerical integration
#include "Intrepid_CellTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_PointTools.hpp"

PenaltyMethodFilter::PenaltyMethodFilter(Constraints constraints){
  _constraints = constraints;
}

void PenaltyMethodFilter::filter(FieldContainer<double> &localStiffnessMatrix, const FieldContainer<double> &physicalCellNodes,
				 vector<int> &cellIDs, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc){

  // will only enforce constraints on fluxes at the moment
  vector<int> trialIDs = mesh->bilinearForm().trialIDs(); 
  vector<int> fluxTraceIDs;
  /*
	FieldContainer<double> unitNormals; // NEED TO SET
	FieldContainer<double> cubaturePoints; // NEED TO SET
	// compute constraint matrix
	vector< map<int,FieldContainer<double> > > constraintCoeffs;
	vector< FieldContainer<double> > constraintValues;
	FieldContainer<bool> imposeHere;
	_constraints.imposeConstraints(trialID, cubaturePoints, unitNormals, 
				       constraintCoeffs, constraintValues,
				       imposeHere);
  */

}
