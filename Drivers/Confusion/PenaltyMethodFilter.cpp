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
void PenaltyMethodFilter::filter(FieldContainer<double> &localStiffnessMatrix, const FieldContainer<double> &physicalCellNodes,vector<int> &cellIDs, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc){
  
  cout << "Applying Penalty method filter " << endl;

  typedef Teuchos::RCP< ElementType > ElementTypePtr;
  typedef Teuchos::RCP< Element > ElementPtr;
  typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;
    
  // assumption: filter gets elements of all the same type  
  TEST_FOR_EXCEPTION(cellIDs.size()==0,std::invalid_argument,"no cell IDs given to filter");

  ElementTypePtr elemTypePtr = mesh->elements()[cellIDs[0]]->elementType(); 
  int numCells = physicalCellNodes.dimension(0);
  
  DofOrderingPtr trialOrderPtr = elemTypePtr->trialOrderPtr;
  int maxTrialDegree = trialOrderPtr->maxBasisDegree();
  BasisValueCache basisCache(physicalCellNodes, *(elemTypePtr->cellTopoPtr), *(trialOrderPtr), maxTrialDegree, true);
  cout << "maxTrialDegree = " << maxTrialDegree << endl;
  unsigned numSides = elemTypePtr->cellTopoPtr->getSideCount();
  // only allows for L2 inner products at the moment. 
  EOperatorExtended trialOperator = IntrepidExtendedTypes::OPERATOR_VALUE;
	
  // loop over sides first 
  for (unsigned int sideIndex = 0; sideIndex<numSides; sideIndex++){
   
    // GET INTEGRATION INFO - get cubature points and side normals to send to Constraints (Cell,Point, spaceDim)
    FieldContainer<double> sideCubPoints = basisCache.getPhysicalCubaturePointsForSide(sideIndex);
    FieldContainer<double> sideNormals = basisCache.getSideUnitNormals(sideIndex);        
    cout << "got cub info " << endl;
    vector<int> ptDims;
    sideNormals.dimensions(ptDims);
    cout << "sideNormals has dimensions " << sideNormals.dimension(0) << ", " << sideNormals.dimension(1) << ", " << sideNormals.dimension(3) << endl;

    int numPts = sideCubPoints.dimension(1);

    cout << "getting constraints" << endl;

    // GET CONSTRAINT INFO
    vector<map<int, FieldContainer<double> > > constrCoeffsVector;
    vector<FieldContainer<double> > constraintValuesVector;
    vector<FieldContainer<bool> > imposeHereVector;
    _constraints->getConstraints(sideCubPoints,sideNormals,constrCoeffsVector,constraintValuesVector);
    cout << "got constraints" << endl;

    //loop thru constraints
    for (vector<map<int,FieldContainer<double> > >::iterator constrIt = constrCoeffsVector.begin(); constrIt !=constrCoeffsVector.end(); constrIt++){
      map<int,FieldContainer<double> > constrCoeffs = *constrIt;

      // loop thru pairs of trialIDs and constr coeffs
      for (map<int,FieldContainer<double> >::iterator constrIDIt = constrCoeffs.begin(); constrIDIt !=constrCoeffs.end(); constrIDIt++){
	pair<int,FieldContainer<double> > constrPair = *constrIDIt;
	int trialID = constrPair.first;

	// get basis to integrate
	Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > trialBasis1 = trialOrderPtr->getBasis(trialID,sideIndex);
	// for trial: the value lives on the side, so we don't use the volume coords either:
	FieldContainer<double> trialValuesTransformed = *(basisCache.getTransformedValues(trialBasis1,trialOperator,sideIndex,false));
        // make copies b/c we can't fudge with return values from basisCache (const) - dimensions (Cell,Field - basis ordinal, Point)
        FieldContainer<double> trialValuesCopy = trialValuesTransformed;

	cout << "transforming trial values " << endl;
	// transform trial values
	int numDofs1 = trialOrderPtr->getBasisCardinality(trialID,sideIndex); 
	for (int dofIndex=0; dofIndex<numDofs1; dofIndex++){
	  for (int cellIndex=0; cellIndex<numCells; cellIndex++){
	    for (int ptIndex=0; ptIndex<numPts; ptIndex++){
	      trialValuesCopy(cellIndex, dofIndex, ptIndex) *= constrPair.second(cellIndex, ptIndex); // scale by constraint coeff
	    }	   
	  }
	}
	
	/////////////////////////////////////////////////////////////////////////////////////

	for (map<int,FieldContainer<double> >::iterator constrTestIDIt = constrCoeffs.begin(); constrTestIDIt !=constrCoeffs.end(); constrTestIDIt++){
	  pair<int,FieldContainer<double> > constrTestPair = *constrTestIDIt;
	  int testTrialID = constrTestPair.first;

	  // get basis to integrate for testing fxns
	  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > testTrialBasis = trialOrderPtr->getBasis(testTrialID,sideIndex);
	  FieldContainer<double> testTrialValuesTransformedWeighted = *(basisCache.getTransformedWeightedValues(testTrialBasis,trialOperator,sideIndex,false));
	  // make copies b/c we can't fudge with return values from basisCache (const) - dimensions (Cell,Field - basis ordinal, Point)
	  FieldContainer<double> testTrialValuesWeightedCopy = testTrialValuesTransformedWeighted;
	  
	  cout << "transforming test values " << endl;
	  int numDofs2 = trialOrderPtr->getBasisCardinality(testTrialID,sideIndex); 
	  for (int cellIndex=0; cellIndex<numCells; cellIndex++){
	    for (int dofIndex=0; dofIndex<numDofs2; dofIndex++){
	      for (int ptIndex=0; ptIndex<numPts; ptIndex++){
		testTrialValuesWeightedCopy(cellIndex, dofIndex, ptIndex) *= constrTestPair.second(cellIndex, ptIndex); // scale by constraint coeff
	      }	   
	    }
	  }

	  double penaltyParameter = 1e7; // (single_precision)^(-1) - perhaps have this computed relative to terms in the matrix?

	  // integrate the transformed values, add them to the relevant trial/testTrialID dof combos
	  FieldContainer<double> unweightedPenaltyMatrix(numCells,numDofs1,numDofs2);
	  FunctionSpaceTools::integrate<double>(unweightedPenaltyMatrix,trialValuesCopy,testTrialValuesWeightedCopy,COMP_CPP);
	  for (int cellIndex=0; cellIndex<numCells; cellIndex++){
	    for (int trialDofIndex=0; trialDofIndex<numDofs1; trialDofIndex++){
	      for (int testDofIndex=0; testDofIndex<numDofs2; testDofIndex++){		
		int localTrialDof = trialOrderPtr->getDofIndex(trialID, trialDofIndex, sideIndex);
		int localTestDof = trialOrderPtr->getDofIndex(testTrialID, testDofIndex, sideIndex);
		localStiffnessMatrix(cellIndex,localTrialDof,localTestDof) += penaltyParameter*unweightedPenaltyMatrix(cellIndex,trialDofIndex,testDofIndex);
	      }
	    }
	  }	  	  	  
	}
	
	/////////////////////////////////////////////////////////////////////////////////////
	
      }
    }
  }
}
