#include "PenaltyMethodFilter.h"
#include "Mesh.h"

// Intrepid includes - for numerical integration
//#include "Intrepid_CellTools.hpp"
//#include "Intrepid_DefaultCubatureFactory.hpp"
//#include "Intrepid_FunctionSpaceTools.hpp"
//#include "Intrepid_PointTools.hpp"

#include "BasisValueCache.h"
#include "DofOrdering.h"

PenaltyMethodFilter::PenaltyMethodFilter(Teuchos::RCP<Constraints> constraints){
  _constraints = constraints;
}

void PenaltyMethodFilter::filter(FieldContainer<double> &localStiffnessMatrix, const FieldContainer<double> &physicalCellNodes,
				 vector<int> &cellIDs, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc){

  typedef Teuchos::RCP< ElementType > ElementTypePtr;
  typedef Teuchos::RCP< Element > ElementPtr;
  typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;

  // will only enforce constraints on fluxes at the moment
  vector<int> trialIDs = mesh->bilinearForm().trialIDs(); 
  vector<int> fluxTraceIDs = trialIDs; // POPULATE with relevant trialIDs

  // assumption: filter gets elements of all the same type  
  ElementTypePtr elemTypePtr = mesh->elements()[cellIDs[0]]->elementType(); 
  int numSides = mesh->elements()[cellIDs[0]]->numSides();
  
  DofOrderingPtr trialOrderPtr = elemTypePtr->trialOrderPtr;
  int maxTrialDegree = trialOrderPtr->maxBasisDegree();
  BasisValueCache basisCache = BasisValueCache(physicalCellNodes, *(elemTypePtr->cellTopoPtr), *(trialOrderPtr), maxTrialDegree, true);

  for (vector<int>::iterator fluxTraceIt1 = fluxTraceIDs.begin(); fluxTraceIt1 != fluxTraceIDs.end(); fluxTraceIt1++) {
    for (vector<int>::iterator fluxTraceIt2 = fluxTraceIDs.begin(); fluxTraceIt2 != fluxTraceIDs.end(); fluxTraceIt2++) {

      int trialID1 = *fluxTraceIt1;
      int trialID2 = *fluxTraceIt2;	      
      
      for (int sideIndex = 0; sideIndex<numSides; sideIndex++){
	
	Teuchos::RCP < Basis<double,FieldContainer<double> > > basis1 = elemTypePtr->trialOrderPtr->getBasis(trialID1,sideIndex);
	Teuchos::RCP < Basis<double,FieldContainer<double> > > basis2 = elemTypePtr->trialOrderPtr->getBasis(trialID2,sideIndex); 
	
	//	const FieldContainer<double> & getPhysicalCubaturePoints();
	//	const FieldContainer<double> & getPhysicalCubaturePointsForSide(int sideOrdinal);
	Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > trialBasis1;
	Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > trialBasis2;
	trialBasis1 = trialOrderPtr->getBasis(trialID1,sideIndex);
	trialBasis2 = trialOrderPtr->getBasis(trialID2,sideIndex);

	EOperatorExtended trialOperator = IntrepidExtendedTypes::OPERATOR_VALUE;
	FieldContainer<double> trialValues1Transformed;
	FieldContainer<double> trialValues2TransformedWeighted;
	// for trial: the value lives on the side, so we don't use the volume coords either:
	trialValues1Transformed = *(basisCache.getTransformedValues(trialBasis1,trialOperator,sideIndex,false));
	trialValues2TransformedWeighted = *(basisCache.getTransformedWeightedValues(trialBasis2,trialOperator,sideIndex,false));

	// get cubature points and side normals to send to Constraints (Cell,Point, spaceDim)
	FieldContainer<double> sideCubPoints = basisCache.getPhysicalCubaturePointsForSide(sideIndex);
	FieldContainer<double> sideNormals = basisCache.getSideUnitNormals(sideIndex);

	// make copies b/c we can't fudge with return values from basisCache (const)
	// (Cell,Field - basis ordinal, Point)
	FieldContainer<double> trialValues1Copy = trialValues1Transformed;
	FieldContainer<double> trialValues2CopyWeighted = trialValues2TransformedWeighted;

	map<int,FieldContainer<double> > constraintCoeffs1;
	map<int,FieldContainer<double> > constraintCoeffs2;
	FieldContainer<double>  constraintValues;
	FieldContainer<bool> imposeHere1;	
	FieldContainer<bool> imposeHere2;	
	_constraints->imposeConstraints(trialID1,sideCubPoints,sideNormals,constraintCoeffs1,constraintValues1,imposeHere1);
	_constraints->imposeConstraints(trialID2,sideCubPoints,sideNormals,constraintCoeffs2,constraintValues2,imposeHere2);

	int numCells = sideCubPoints.dimension(0);
	int numPts = sideCubPoints.dimension(1);
	int spaceDim = sideCubPoints.dimension(2);
	for (int cellIndex=0;cellIndex<numCells;cellIndex++){
	  for (int dofIndex1=0;dofIndex1<numDofs1;dofIndex1++){
	    for (int dofIndex2=0;dofIndex2<numDofs2;dofIndex2++){
	      for (int ptIndex=0;ptIndex<numPts;ptIndex++){	    
		if (imposeHere1(cellIndex,ptIndex) && imposeHere2(cellIndex,ptIndex)){
		  trialValues1Copy(cellIndex,dofIndex1,ptIndex);
		  trialValues2CopyWeighted(cellIndex,dofIndex2,ptIndex);
		}
	      }
	    }
	  }
	}
	
	
	//	FunctionSpaceTools::integrate<double>(,trialValues1Transformed,trialValues2TransformedWeighted,COMP_CPP);
      
      }      
    }    
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
}
