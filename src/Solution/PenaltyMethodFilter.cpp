//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "PenaltyMethodFilter.h"
#include "Mesh.h"

#include "BasisCache.h"
#include "DofOrdering.h"

#include "Intrepid_FunctionSpaceTools.hpp"

#include "CamelliaCellTools.h"

using namespace Camellia;

PenaltyMethodFilter::PenaltyMethodFilter(Teuchos::RCP<Constraints> constraints)
{
  _constraints = constraints;
}
void PenaltyMethodFilter::filter(Camellia::FieldContainer<double> &localStiffnessMatrix, Camellia::FieldContainer<double> &localRHSVector,
                                 BasisCachePtr basisCache, Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc)
{

  // TODO: rewrite this to support imposing penalty conditions on field variables.  It at least looks like this is
  //       not presently supported.
  
  // assumption: filter gets elements of all the same type
  TEUCHOS_TEST_FOR_EXCEPTION(basisCache->cellIDs().size()==0,std::invalid_argument,"no cell IDs given to filter");

  ElementTypePtr elemTypePtr = mesh->getElementType(basisCache->cellIDs()[0]);
  int numCells = localStiffnessMatrix.dimension(0);

  DofOrderingPtr trialOrderPtr = elemTypePtr->trialOrderPtr;

  unsigned numSides = elemTypePtr->cellTopoPtr->getSideCount();
  // only allows for L2 inner products at the moment.
  Camellia::EOperator trialOperator =  Camellia::OP_VALUE;

  // loop over sides first
  for (unsigned int sideIndex = 0; sideIndex<numSides; sideIndex++)
  {

    // GET INTEGRATION INFO - get cubature points and side normals to send to Constraints (Cell,Point, spaceDim)
    Camellia::FieldContainer<double> sideCubPoints = basisCache->getPhysicalCubaturePointsForSide(sideIndex);
    Camellia::FieldContainer<double> sideNormals = basisCache->getSideUnitNormals(sideIndex);

    int numPts = sideCubPoints.dimension(1);

    // GET CONSTRAINT INFO
    vector<map<int, Camellia::FieldContainer<double> > > constrCoeffsVector;
    vector<Camellia::FieldContainer<double> > constraintValuesVector;
    vector<Camellia::FieldContainer<bool> > imposeHereVector;
    _constraints->getConstraints(sideCubPoints,sideNormals,constrCoeffsVector,constraintValuesVector);

    //loop thru constraints
    int i = 0;
    for (vector<map<int,Camellia::FieldContainer<double> > >::iterator constrIt = constrCoeffsVector.begin();
         constrIt !=constrCoeffsVector.end(); constrIt++)
    {
      map<int,Camellia::FieldContainer<double> > constrCoeffs = *constrIt;
      Camellia::FieldContainer<double> constrValues = constraintValuesVector[i];
      i++;

      double penaltyParameter = 1e7; // (single_precision)^(-1) - perhaps have this computed relative to terms in the matrix?

      for (map<int,Camellia::FieldContainer<double> >::iterator constrTestIDIt = constrCoeffs.begin();
           constrTestIDIt !=constrCoeffs.end(); constrTestIDIt++)
      {
        pair<int,Camellia::FieldContainer<double> > constrTestPair = *constrTestIDIt;
        int testTrialID = constrTestPair.first;

        if (! trialOrderPtr->hasBasisEntry(testTrialID, sideIndex)) continue;
        // get basis to integrate for testing fxns
        BasisPtr testTrialBasis = trialOrderPtr->getBasis(testTrialID,sideIndex);
        Camellia::FieldContainer<double> testTrialValuesTransformedWeighted = *(basisCache->getTransformedWeightedValues(testTrialBasis,trialOperator,
            sideIndex,false));
        // make copies b/c we can't fudge with return values from basisCache (const) - dimensions (Cell,Field - basis ordinal, Point)
        Camellia::FieldContainer<double> testTrialValuesWeightedCopy = testTrialValuesTransformedWeighted;

        int numDofs2 = trialOrderPtr->getBasisCardinality(testTrialID,sideIndex);
        for (int cellIndex=0; cellIndex<numCells; cellIndex++)
        {
          for (int dofIndex=0; dofIndex<numDofs2; dofIndex++)
          {
            for (int ptIndex=0; ptIndex<numPts; ptIndex++)
            {
              testTrialValuesWeightedCopy(cellIndex, dofIndex, ptIndex) *= constrTestPair.second(cellIndex, ptIndex); // scale by constraint coeff
            }
          }
        }

        // loop thru pairs of trialIDs and constr coeffs
        for (map<int,Camellia::FieldContainer<double> >::iterator constrIDIt = constrCoeffs.begin();
             constrIDIt !=constrCoeffs.end(); constrIDIt++)
        {
          pair<int,Camellia::FieldContainer<double> > constrPair = *constrIDIt;
          int trialID = constrPair.first;

          if (! trialOrderPtr->hasBasisEntry(testTrialID, sideIndex)) continue;

          // get basis to integrate
          BasisPtr trialBasis1 = trialOrderPtr->getBasis(trialID,sideIndex);
          // for trial: the value lives on the side, so we don't use the volume coords either:
          Camellia::FieldContainer<double> trialValuesTransformed = *(basisCache->getTransformedValues(trialBasis1,trialOperator,sideIndex,false));
          // make copies b/c we can't fudge with return values from basisCache (const) - dimensions (Cell,Field - basis ordinal, Point)
          Camellia::FieldContainer<double> trialValuesCopy = trialValuesTransformed;

          // transform trial values
          int numDofs1 = trialOrderPtr->getBasisCardinality(trialID,sideIndex);
          for (int dofIndex=0; dofIndex<numDofs1; dofIndex++)
          {
            for (int cellIndex=0; cellIndex<numCells; cellIndex++)
            {
              for (int ptIndex=0; ptIndex<numPts; ptIndex++)
              {
                trialValuesCopy(cellIndex, dofIndex, ptIndex) *= constrPair.second(cellIndex, ptIndex); // scale by constraint coeff
              }
            }
          }

          /////////////////////////////////////////////////////////////////////////////////////


          // integrate the transformed values, add them to the relevant trial/testTrialID dof combos
          Camellia::FieldContainer<double> unweightedPenaltyMatrix(numCells,numDofs1,numDofs2);
          
          Intrepid::FieldContainer<double> &unweightedPenaltyMatrixIntrepid     = unweightedPenaltyMatrix;
          Intrepid::FieldContainer<double> &trialValuesCopyIntrepid             = trialValuesCopy;
          Intrepid::FieldContainer<double> &testTrialValuesWeightedCopyIntrepid = testTrialValuesWeightedCopy;
          
          // the casts above are important to let FunctionSpaceTools invoke BLAS
          Intrepid::FunctionSpaceTools::integrate<double>(unweightedPenaltyMatrixIntrepid,
                                                          trialValuesCopyIntrepid,
                                                          testTrialValuesWeightedCopyIntrepid,
                                                          Intrepid::COMP_BLAS);

          for (int cellIndex=0; cellIndex<numCells; cellIndex++)
          {
            for (int testDofIndex=0; testDofIndex<numDofs2; testDofIndex++)
            {
              int localTestDof = trialOrderPtr->getDofIndex(testTrialID, testDofIndex, sideIndex);
              for (int trialDofIndex=0; trialDofIndex<numDofs1; trialDofIndex++)
              {
                int localTrialDof = trialOrderPtr->getDofIndex(trialID, trialDofIndex, sideIndex);
                localStiffnessMatrix(cellIndex,localTrialDof,localTestDof)
                += penaltyParameter*unweightedPenaltyMatrix(cellIndex,trialDofIndex,testDofIndex);
              }
            }
          }
        }

        /////////////////////////////////////////////////////////////////////////////////////
        // set penalty load
        Camellia::FieldContainer<double> unweightedRHSVector(numCells,numDofs2);
        
        Intrepid::FieldContainer<double> &unweightedRHSVectorIntrepid         = unweightedRHSVector;
        Intrepid::FieldContainer<double> &constrValuesIntrepid                = constrValues;
        Intrepid::FieldContainer<double> &testTrialValuesWeightedCopyIntrepid = testTrialValuesWeightedCopy;
        
        // the casts above are important to let FunctionSpaceTools invoke BLAS
        Intrepid::FunctionSpaceTools::integrate<double>(unweightedRHSVectorIntrepid,
                                                        constrValuesIntrepid,
                                                        testTrialValuesWeightedCopyIntrepid,
                                                        Intrepid::COMP_BLAS);
        
        for (int cellIndex=0; cellIndex<numCells; cellIndex++)
        {
          for (int testDofIndex=0; testDofIndex<numDofs2; testDofIndex++)
          {
            int localTestDof = trialOrderPtr->getDofIndex(testTrialID, testDofIndex, sideIndex);
            localRHSVector(cellIndex,localTestDof) += penaltyParameter*unweightedRHSVector(cellIndex,testDofIndex);
          }
        }

      }
    }
  }
}
