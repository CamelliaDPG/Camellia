// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

#include "BilinearForm.h"
#include "BasisCache.h"
#include "ElementType.h"
#include "BilinearFormUtility.h"
#include "VarFactory.h"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"

#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_SerialSpdDenseSolver.h"

#include "Epetra_DataAccess.h"

#include "Epetra_Time.h"

#include "Intrepid_FunctionSpaceTools.hpp"

#include "SerialDenseWrapper.h"

#include "RHS.h"
#include "IP.h"

#include "CamelliaCellTools.h"

static const string & S_OP_VALUE = "";
static const string & S_OP_GRAD = "\\nabla ";
static const string & S_OP_CURL = "\\nabla \\times ";
static const string & S_OP_DIV = "\\nabla \\cdot ";
static const string & S_OP_D1 = "D1 ";
static const string & S_OP_D2 = "D2 ";
static const string & S_OP_D3 = "D3 ";
static const string & S_OP_D4 = "D4 ";
static const string & S_OP_D5 = "D5 ";
static const string & S_OP_D6 = "D6 ";
static const string & S_OP_D7 = "D7 ";
static const string & S_OP_D8 = "D8 ";
static const string & S_OP_D9 = "D9 ";
static const string & S_OP_D10 = "D10 ";
static const string & S_OP_X = "{1 \\choose 0} \\cdot ";
static const string & S_OP_Y = "{0 \\choose 1} \\cdot ";
static const string & S_OP_Z = "\\bf{k} \\cdot ";
static const string & S_OP_DX = "\\frac{\\partial}{\\partial x} ";
static const string & S_OP_DY = "\\frac{\\partial}{\\partial y} ";
static const string & S_OP_DZ = "\\frac{\\partial}{\\partial z} ";
static const string & S_OP_CROSS_NORMAL = "\\times \\widehat{n} ";
static const string & S_OP_DOT_NORMAL = "\\cdot \\widehat{n} ";
static const string & S_OP_TIMES_NORMAL = " \\widehat{n} \\cdot ";
static const string & S_OP_TIMES_NORMAL_X = " \\widehat{n}_x ";
static const string & S_OP_TIMES_NORMAL_Y = " \\widehat{n}_y ";
static const string & S_OP_TIMES_NORMAL_Z = " \\widehat{n}_z ";
static const string & S_OP_VECTORIZE_VALUE = ""; // handle this one separately...
static const string & S_OP_UNKNOWN = "[UNKNOWN OPERATOR] ";

set<int> BilinearForm::_normalOperators;

BilinearForm::BilinearForm() {
  _useQRSolveForOptimalTestFunctions = true;
  _useSPDSolveForOptimalTestFunctions = false;
  _useIterativeRefinementsWithSPDSolve = false;
  _warnAboutZeroRowsAndColumns = true;
}

const vector< int > & BilinearForm::trialIDs() {
  return _trialIDs;
}

const vector< int > & BilinearForm::testIDs() {
  return _testIDs;
}

void BilinearForm::applyBilinearFormData(FieldContainer<double> &trialValues, FieldContainer<double> &testValues, 
                                         int trialID, int testID, int operatorIndex,
                                         const FieldContainer<double> &points) {
  applyBilinearFormData(trialID,testID,trialValues,testValues,points);
}

void BilinearForm::applyBilinearFormData(FieldContainer<double> &trialValues, FieldContainer<double> &testValues, 
                                         int trialID, int testID, int operatorIndex,
                                         Teuchos::RCP<BasisCache> basisCache) {
  applyBilinearFormData(trialValues, testValues, trialID, testID, operatorIndex, basisCache->getPhysicalCubaturePoints());
}

void BilinearForm::trialTestOperators(int testID1, int testID2, 
                                      vector<EOperatorExtended> &testOps1,
                                      vector<EOperatorExtended> &testOps2) {
  IntrepidExtendedTypes::EOperatorExtended testOp1, testOp2;
  testOps1.clear();
  testOps2.clear();
  if (trialTestOperator(testID1,testID2,testOp1,testOp2)) {
    testOps1.push_back(testOp1);
    testOps2.push_back(testOp2);
  }
}

void BilinearForm::localStiffnessMatrixAndRHS(FieldContainer<double> &localStiffness, FieldContainer<double> &rhsVector,
                                              Teuchos::RCP< DPGInnerProduct > ip, BasisCachePtr ipBasisCache, RHSPtr rhs, BasisCachePtr basisCache) {
  double testMatrixAssemblyTime = 0, testMatrixInversionTime = 0, localStiffnessDeterminationFromTestsTime = 0;
  double rhsIntegrationAgainstOptimalTestsTime = 0;
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  
  Epetra_Time timer(Comm);
  
  // localStiffness should have dim. (numCells, numTrialFields, numTrialFields)
  MeshPtr mesh = basisCache->mesh();
  if (mesh.get() == NULL) {
    cout << "localStiffnessMatrix requires BasisCache to have mesh set.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffnessMatrix requires BasisCache to have mesh set.");
  }
  const vector<GlobalIndexType>* cellIDs = &basisCache->cellIDs();
  int numCells = cellIDs->size();
  if (numCells != localStiffness.dimension(0)) {
    cout << "localStiffnessMatrix requires basisCache->cellIDs() to have the same # of cells as the first dimension of localStiffness\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffnessMatrix requires basisCache->cellIDs() to have the same # of cells as the first dimension of localStiffness");
  }
  
  ElementTypePtr elemType = mesh->getElementType((*cellIDs)[0]); // we assume all cells provided are of the same type
  DofOrderingPtr trialOrder = elemType->trialOrderPtr;
  DofOrderingPtr testOrder = elemType->testOrderPtr;
  int numTestDofs = testOrder->totalDofs();
  int numTrialDofs = trialOrder->totalDofs();
  if ((numTrialDofs != localStiffness.dimension(1)) || (numTrialDofs != localStiffness.dimension(2))) {
    cout << "localStiffness should have dimensions (C,numTrialFields,numTrialFields).\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffness should have dimensions (C,numTrialFields,numTrialFields).");
  }
  
  timer.ResetStartTime();
  
  bool printTimings = false;

  if (printTimings) {
    cout << "numCells: " << numCells << endl;
    cout << "numTestDofs: " << numTestDofs << endl;
    cout << "numTrialDofs: " << numTrialDofs << endl;
  }
  
  FieldContainer<double> ipMatrix(numCells,numTestDofs,numTestDofs);
  ip->computeInnerProductMatrix(ipMatrix, testOrder, ipBasisCache);
  
  testMatrixAssemblyTime += timer.ElapsedTime();
  
  //      cout << "ipMatrix:\n" << ipMatrix;
  
  timer.ResetStartTime();
  FieldContainer<double> optTestCoeffs(numCells,numTrialDofs,numTestDofs);
  FieldContainer<double> cellSideParities = basisCache->getCellSideParities();
  
  int optSuccess = this->optimalTestWeights(optTestCoeffs, ipMatrix, elemType,
                                            cellSideParities, basisCache);
  testMatrixInversionTime += timer.ElapsedTime();
//      cout << "optTestCoeffs:\n" << optTestCoeffs;
  
  if ( optSuccess != 0 ) {
    cout << "**** WARNING: in BilinearForm::localStiffnessMatrixAndRHS(), optimal test function computation failed with error code " << optSuccess << ". ****\n";
  }
  
  //cout << "optTestCoeffs\n" << optTestCoeffs;
  
  timer.ResetStartTime();
  
  BilinearFormUtility::computeStiffnessMatrix(localStiffness,ipMatrix,optTestCoeffs);
  localStiffnessDeterminationFromTestsTime += timer.ElapsedTime();
  //      cout << "finalStiffness:\n" << finalStiffness;
  
  timer.ResetStartTime();
  rhs->integrateAgainstOptimalTests(rhsVector, optTestCoeffs, testOrder, basisCache);
  rhsIntegrationAgainstOptimalTestsTime += timer.ElapsedTime();
  
  if (printTimings) {
    cout << "testMatrixAssemblyTime: " << testMatrixAssemblyTime << " seconds.\n";
    cout << "testMatrixInversionTime: " << testMatrixInversionTime << " seconds.\n";
    cout << "localStiffnessDeterminationFromTestsTime: " << localStiffnessDeterminationFromTestsTime << " seconds.\n";
    cout << "rhsIntegrationAgainstOptimalTestsTime: " << rhsIntegrationAgainstOptimalTestsTime << " seconds.\n";
  }
}

void BilinearForm::multiplyFCByWeight(FieldContainer<double> & fc, double weight) {
  int size = fc.size();
  double *valuePtr = &fc[0]; // to make this as fast as possible, do some pointer arithmetic...
  for (int i=0; i<size; i++) {
    *valuePtr *= weight;
    valuePtr++;
  }
}

bool checkSymmetry(FieldContainer<double> &innerProductMatrix) {
  double tol = 1e-10;
  int numCells = innerProductMatrix.dimension(0);
  int numRows = innerProductMatrix.dimension(1);
  if (numRows != innerProductMatrix.dimension(2)) {
    // non-square: obviously not symmetric!
    return false;
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int i=0; i<numRows; i++) {
      for (int j=0; j<i; j++) {
        double diff = abs( innerProductMatrix(cellIndex,i,j) - innerProductMatrix(cellIndex,j,i) );
        if (diff > tol) {
          return false;
        }
      }
    }
  }
  return true;
}

int BilinearForm::optimalTestWeights(FieldContainer<double> &optimalTestWeights,
                                     FieldContainer<double> &innerProductMatrix,
                                     ElementTypePtr elemType,
                                     FieldContainer<double> &cellSideParities,
                                     Teuchos::RCP<BasisCache> stiffnessBasisCache) {
  Teuchos::RCP<DofOrdering> trialOrdering = elemType->trialOrderPtr;
  Teuchos::RCP<DofOrdering> testOrdering = elemType->testOrderPtr;
  
  // all arguments are as in computeStiffnessMatrix, except:
  // optimalTestWeights, which has dimensions (numCells, numTrialDofs, numTestDofs)
  // innerProduct: the inner product which defines the sense in which these test functions are optimal
  int numCells = stiffnessBasisCache->getPhysicalCubaturePoints().dimension(0);
  int numTestDofs = testOrdering->totalDofs();
  int numTrialDofs = trialOrdering->totalDofs();
  
  // check that optimalTestWeights is properly dimensioned....
  TEUCHOS_TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(0) != numCells ),
                     std::invalid_argument,
                     "physicalCellNodes.dimension(0) and optimalTestWeights.dimension(0) (numCells) do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(1) != numTrialDofs ),
                     std::invalid_argument,
                     "trialOrdering->totalDofs() and optimalTestWeights.dimension(1) do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(2) != numTestDofs ),
                     std::invalid_argument,
                     "testOrdering->totalDofs() and optimalTestWeights.dimension(2) do not match.");
  
  FieldContainer<double> stiffnessMatrix(numCells,numTestDofs,numTrialDofs);
//  FieldContainer<double> stiffnessMatrixT(numCells,numTrialDofs,numTestDofs);
  
  // RHS:
  this->stiffnessMatrix(stiffnessMatrix, elemType, cellSideParities, stiffnessBasisCache);
//  cout << "trialOrdering:\n" << *trialOrdering;
  
//  BilinearFormUtility::transposeFCMatrices(stiffnessMatrixT, stiffnessMatrix);
  
  //cout << "stiffnessMatrixT: " << stiffnessMatrixT << endl;
  //cout << "stiffnessMatrix:" << stiffnessMatrix << endl;
  
  int solvedAll = 0;
  
  FieldContainer<double> optimalWeightsT(numTestDofs, numTrialDofs);
  Teuchos::Array<int> localIPDim(2);
  localIPDim[0] = numTestDofs;
  localIPDim[1] = numTestDofs;
  Teuchos::Array<int> localStiffnessDim(2);
  localStiffnessDim[0] = stiffnessMatrix.dimension(1);
  localStiffnessDim[1] = stiffnessMatrix.dimension(2);
  
  for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
    int result = 0;
    FieldContainer<double> cellIPMatrix(localIPDim, &innerProductMatrix(cellIndex,0,0));
    FieldContainer<double> cellStiffness(localStiffnessDim, &stiffnessMatrix(cellIndex,0,0));
    if (_useQRSolveForOptimalTestFunctions) {
      result = SerialDenseWrapper::solveSystemUsingQR(optimalWeightsT, cellIPMatrix, cellStiffness);
    } else if (_useSPDSolveForOptimalTestFunctions) {
      result = SerialDenseWrapper::solveSPDSystemMultipleRHS(optimalWeightsT, cellIPMatrix, cellStiffness);
      if (result != 0) {
        // may be that we're not SPD numerically
        cout << "During optimal test weight solution, encountered IP matrix that's not numerically SPD.  Solving with LU factorization instead of Cholesky.\n";
        result = SerialDenseWrapper::solveSystemMultipleRHS(optimalWeightsT, cellIPMatrix, cellStiffness);
      }
    } else {
      SerialDenseWrapper::solveSystemMultipleRHS(optimalWeightsT, cellIPMatrix, cellStiffness);
    }
    // copy/transpose the optimal test weights
    for (int i=0; i<optimalTestWeights.dimension(1); i++) {
      for (int j=0; j<optimalTestWeights.dimension(2); j++) {
        optimalTestWeights(cellIndex,i,j) = optimalWeightsT(j,i);
      }
    }
    if (result != 0) {
      solvedAll = result;
    }
  }

  return solvedAll;
}

void BilinearForm::stiffnessMatrix(FieldContainer<double> &stiffness, ElementTypePtr elemType,
                                   FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache) {

  DofOrderingPtr testOrdering  = elemType->testOrderPtr;
  DofOrderingPtr trialOrdering = elemType->trialOrderPtr;
  stiffnessMatrix(stiffness,trialOrdering,testOrdering,cellSideParities,basisCache);

}

void BilinearForm::stiffnessMatrix(FieldContainer<double> &stiffness, Teuchos::RCP<DofOrdering> trialOrdering, 
                                     Teuchos::RCP<DofOrdering> testOrdering,
                                     FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache) {
    
  // stiffness dimensions are: (numCells, # testOrdering Dofs, # trialOrdering Dofs)
  // (while (cell,trial,test) is more natural conceptually, I believe the above ordering makes
  //  more sense given the inversion that we must do to compute the optimal test functions...)
  
  // steps:
  // 0. Set up Cubature
  // 1. Determine Jacobians
  // 2. Determine quadrature points on interior and boundary
  // 3. For each (test, trial) combination:
  //   a. Apply the specified operators to the basis in the DofOrdering, at the cubature points
  //   b. Multiply the two bases together, weighted with Jacobian/Piola transform and cubature weights
  //   c. Pass the result to bilinearForm's applyBilinearFormData method
  //   d. Sum up (integrate) and place in stiffness matrix according to DofOrdering indices
  
  // check inputs
  int numTestDofs = testOrdering->totalDofs();
  int numTrialDofs = trialOrdering->totalDofs();
  
  shards::CellTopology cellTopo = basisCache->cellTopology();
  unsigned numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  unsigned spaceDim = cellTopo.getDimension();
  
  //cout << "trialOrdering: " << *trialOrdering;
  //cout << "testOrdering: " << *testOrdering;
  
  // check stiffness dimensions:
  TEUCHOS_TEST_FOR_EXCEPTION( ( numCells != stiffness.dimension(0) ),
                     std::invalid_argument,
                     "numCells and stiffness.dimension(0) do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION( ( numTestDofs != stiffness.dimension(1) ),
                     std::invalid_argument,
                     "numTestDofs and stiffness.dimension(1) do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION( ( numTrialDofs != stiffness.dimension(2) ),
                     std::invalid_argument,
                     "numTrialDofs and stiffness.dimension(2) do not match.");
  
  // 0. Set up BasisCache
  int cubDegreeTrial = trialOrdering->maxBasisDegree();
  int cubDegreeTest = testOrdering->maxBasisDegree();
  int cubDegree = cubDegreeTrial + cubDegreeTest;
  
  unsigned numSides = CamelliaCellTools::getSideCount(cellTopo);
  
  // 3. For each (test, trial) combination:
  vector<int> testIDs = this->testIDs();
  vector<int>::iterator testIterator;
  
  vector<int> trialIDs = this->trialIDs();
  vector<int>::iterator trialIterator;
  
  BasisPtr trialBasis, testBasis;
  
  stiffness.initialize(0.0);
  
  for (testIterator = testIDs.begin(); testIterator != testIDs.end(); testIterator++) {
    int testID = *testIterator;
    
    for (trialIterator = trialIDs.begin(); trialIterator != trialIDs.end(); trialIterator++) {
      int trialID = *trialIterator;
      
      vector<EOperatorExtended> trialOperators, testOperators;
      this->trialTestOperators(trialID, testID, trialOperators, testOperators);
      vector<EOperatorExtended>::iterator trialOpIt, testOpIt;
      testOpIt = testOperators.begin();
      TEUCHOS_TEST_FOR_EXCEPTION(trialOperators.size() != testOperators.size(), std::invalid_argument,
                         "trialOperators and testOperators must be the same length");
      int operatorIndex = -1;
      for (trialOpIt = trialOperators.begin(); trialOpIt != trialOperators.end(); trialOpIt++) {
        operatorIndex++;
        IntrepidExtendedTypes::EOperatorExtended trialOperator = *trialOpIt;
        IntrepidExtendedTypes::EOperatorExtended testOperator = *testOpIt;
        
        if (testOperator==OP_TIMES_NORMAL) {
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"OP_TIMES_NORMAL not supported for tests.  Use for trial only");
        }
        
        Teuchos::RCP < const FieldContainer<double> > testValuesTransformed;
        Teuchos::RCP < const FieldContainer<double> > trialValuesTransformed;
        Teuchos::RCP < const FieldContainer<double> > testValuesTransformedWeighted;
        
        //cout << "trial is " <<  this->trialName(trialID) << "; test is " << this->testName(testID) << endl;
        
        if (! this->isFluxOrTrace(trialID)) {
          trialBasis = trialOrdering->getBasis(trialID);
          testBasis = testOrdering->getBasis(testID);
          
          FieldContainer<double> miniStiffness( numCells, testBasis->getCardinality(), trialBasis->getCardinality() );
          
          trialValuesTransformed = basisCache->getTransformedValues(trialBasis,trialOperator);
          testValuesTransformedWeighted = basisCache->getTransformedWeightedValues(testBasis,testOperator);
          
          FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePoints();
          FieldContainer<double> materialDataAppliedToTrialValues = *trialValuesTransformed; // copy first
          FieldContainer<double> materialDataAppliedToTestValues = *testValuesTransformedWeighted; // copy first
          this->applyBilinearFormData(materialDataAppliedToTrialValues, materialDataAppliedToTestValues,
                                              trialID,testID,operatorIndex,basisCache);
          
          //integrate:
          FunctionSpaceTools::integrate<double>(miniStiffness,materialDataAppliedToTestValues,materialDataAppliedToTrialValues,COMP_BLAS);
          // place in the appropriate spot in the element-stiffness matrix
          // copy goes from (cell,trial_basis_dof,test_basis_dof) to (cell,element_trial_dof,element_test_dof)
          
          //cout << "miniStiffness for volume:\n" << miniStiffness;
          
          //checkForZeroRowsAndColumns("miniStiffness for pre-stiffness", miniStiffness);
          
          //cout << "trialValuesTransformed for trial " << this->trialName(trialID) << endl << trialValuesTransformed
          //cout << "testValuesTransformed for test " << this->testName(testID) << ": \n" << testValuesTransformed;
          //cout << "weightedMeasure:\n" << weightedMeasure;
          
          // there may be a more efficient way to do this copying:
          // (one strategy would be to reimplement fst::integrate to support offsets, so that no copying needs to be done...)
          for (int i=0; i < testBasis->getCardinality(); i++) {
            int testDofIndex = testOrdering->getDofIndex(testID,i);
            for (int j=0; j < trialBasis->getCardinality(); j++) {
              int trialDofIndex = trialOrdering->getDofIndex(trialID,j);
              for (unsigned k=0; k < numCells; k++) {
                stiffness(k,testDofIndex,trialDofIndex) += miniStiffness(k,i,j);
              }
            }
          }          
        } else {  // boundary integral
          int trialBasisRank = trialOrdering->getBasisRank(trialID);
          int testBasisRank = testOrdering->getBasisRank(testID);
          
          TEUCHOS_TEST_FOR_EXCEPTION( ( trialBasisRank != 0 ),
                             std::invalid_argument,
                             "Boundary trial variable (flux or trace) given with non-scalar basis.  Unsupported.");
          
          for (unsigned sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
            trialBasis = trialOrdering->getBasis(trialID,sideOrdinal);
            testBasis = testOrdering->getBasis(testID);
            
            bool isFlux = false; // i.e. the normal is "folded into" the variable definition, so that we must take parity into account
            const set<int> normalOperators = BilinearForm::normalOperators();
            if (   (normalOperators.find(testOperator)  == normalOperators.end() ) 
                && (normalOperators.find(trialOperator) == normalOperators.end() ) ) {
              // normal not yet taken into account -- so it must be "hidden" in the trial variable
              isFlux = true;
            }
            
            FieldContainer<double> miniStiffness( numCells, testBasis->getCardinality(), trialBasis->getCardinality() );    
            
            // for trial: the value lives on the side, so we don't use the volume coords either:
            trialValuesTransformed = basisCache->getTransformedValues(trialBasis,trialOperator,sideOrdinal,false);
            // for test: do use the volume coords:
            testValuesTransformed = basisCache->getTransformedValues(testBasis,testOperator,sideOrdinal,true);
            // 
            testValuesTransformedWeighted = basisCache->getTransformedWeightedValues(testBasis,testOperator,sideOrdinal,true);
            
            // copy before manipulating trialValues--these are the ones stored in the cache, so we're not allowed to change them!!
            FieldContainer<double> materialDataAppliedToTrialValues = *trialValuesTransformed;
            
            if (isFlux) {
              // we need to multiply the trialValues by the parity of the normal, since
              // the trial implicitly contains an outward normal, and we need to adjust for the fact
              // that the neighboring cells have opposite normal
              // trialValues should have dimensions (numCells,numFields,numCubPointsSide)
              int numFields = trialValuesTransformed->dimension(1);
              int numPoints = trialValuesTransformed->dimension(2);
              for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
                double parity = cellSideParities(cellIndex,sideOrdinal);
                if (parity != 1.0) {  // otherwise, we can just leave things be...
                  for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
                    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
                      materialDataAppliedToTrialValues(cellIndex,fieldIndex,ptIndex) *= parity;
                    }
                  }
                }
              }
            }
            
            FieldContainer<double> cubPointsSidePhysical = basisCache->getPhysicalCubaturePointsForSide(sideOrdinal);
            FieldContainer<double> materialDataAppliedToTestValues = *testValuesTransformedWeighted; // copy first
            this->applyBilinearFormData(materialDataAppliedToTrialValues,materialDataAppliedToTestValues,
                                                trialID,testID,operatorIndex,basisCache);
            
            
            //cout << "sideOrdinal: " << sideOrdinal << "; cubPointsSidePhysical" << endl << cubPointsSidePhysical;
            
            //   d. Sum up (integrate) and place in stiffness matrix according to DofOrdering indices
            FunctionSpaceTools::integrate<double>(miniStiffness,materialDataAppliedToTestValues,materialDataAppliedToTrialValues,COMP_BLAS);
            
            //checkForZeroRowsAndColumns("side miniStiffness for pre-stiffness", miniStiffness);
            
            //cout << "miniStiffness for side " << sideOrdinal << "\n:" << miniStiffness;
            // place in the appropriate spot in the element-stiffness matrix
            // copy goes from (cell,trial_basis_dof,test_basis_dof) to (cell,element_trial_dof,element_test_dof)
            for (int i=0; i < testBasis->getCardinality(); i++) {
              int testDofIndex = testOrdering->getDofIndex(testID,i,0);
              for (int j=0; j < trialBasis->getCardinality(); j++) {
                int trialDofIndex = trialOrdering->getDofIndex(trialID,j,sideOrdinal);
                for (unsigned k=0; k < numCells; k++) {
                  stiffness(k,testDofIndex,trialDofIndex) += miniStiffness(k,i,j);
                }
              }
            }
          }
        }
        testOpIt++;
      }
    }
  }
  if (_warnAboutZeroRowsAndColumns) {
    bool checkRows = false; // zero rows just mean a test basis function won't get used, which is fine
    bool checkCols = true; // zero columns mean that a trial basis function doesn't enter the computation, which is bad
    if (! BilinearFormUtility::checkForZeroRowsAndColumns("pre-stiffness", stiffness, checkRows, checkCols) ) {
      cout << "pre-stiffness matrix in which zero columns were found:\n";
      cout << stiffness;
      cout << "trialOrdering: \n" << *trialOrdering;
    }
  }
}

vector<int> BilinearForm::trialVolumeIDs() {
  vector<int> ids;
  for (vector<int>::iterator trialIt = _trialIDs.begin(); trialIt != _trialIDs.end(); trialIt++) {
    int trialID = *(trialIt);
    if ( ! isFluxOrTrace(trialID) ) {
      ids.push_back(trialID);
    }
  }
  return ids;
}

vector<int> BilinearForm::trialBoundaryIDs() {
  vector<int> ids;
  for (vector<int>::iterator trialIt = _trialIDs.begin(); trialIt != _trialIDs.end(); trialIt++) {
    int trialID = *(trialIt);
    if ( isFluxOrTrace(trialID) ) {
      ids.push_back(trialID);
    }
  }
  return ids;  
}

int BilinearForm::operatorRank(EOperatorExtended op, IntrepidExtendedTypes::EFunctionSpaceExtended fs) {
  // returns the rank of basis functions in the function space fs when op is applied
  // 0 scalar, 1 vector
  int SCALAR = 0, VECTOR = 1;
  switch (op) {
    case  IntrepidExtendedTypes::OP_VALUE:
      if (   (fs == IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) 
          || (fs == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL)
          || (fs == IntrepidExtendedTypes::FUNCTION_SPACE_REAL_SCALAR) )
        return SCALAR; 
      else
        return VECTOR;
    case IntrepidExtendedTypes::OP_GRAD:
    case IntrepidExtendedTypes::OP_CURL:
      return VECTOR;
    case IntrepidExtendedTypes::OP_DIV:
    case IntrepidExtendedTypes::OP_X:
    case IntrepidExtendedTypes::OP_Y:
    case IntrepidExtendedTypes::OP_Z:
    case IntrepidExtendedTypes::OP_DX:
    case IntrepidExtendedTypes::OP_DY:
    case IntrepidExtendedTypes::OP_DZ:
      return SCALAR; 
    case IntrepidExtendedTypes::OP_CROSS_NORMAL:
      return VECTOR; 
    case IntrepidExtendedTypes::OP_DOT_NORMAL:
      return SCALAR; 
    case IntrepidExtendedTypes::OP_TIMES_NORMAL:
      return VECTOR; 
    case IntrepidExtendedTypes::OP_TIMES_NORMAL_X:
      return SCALAR; 
    case IntrepidExtendedTypes::OP_TIMES_NORMAL_Y:
      return SCALAR; 
    case IntrepidExtendedTypes::OP_TIMES_NORMAL_Z:
      return SCALAR; 
    case IntrepidExtendedTypes::OP_VECTORIZE_VALUE:
      return VECTOR;
    default:
      return -1;
  }
}

const string & BilinearForm::operatorName(EOperatorExtended op) {
  switch (op) {
    case  IntrepidExtendedTypes::OP_VALUE:
      return S_OP_VALUE; 
      break;
    case IntrepidExtendedTypes::OP_GRAD:
      return S_OP_GRAD; 
      break;
    case IntrepidExtendedTypes::OP_CURL:
      return S_OP_CURL; 
      break;
    case IntrepidExtendedTypes::OP_DIV:
      return S_OP_DIV; 
      break;
    case IntrepidExtendedTypes::OP_D1:
      return S_OP_D1; 
      break;
    case IntrepidExtendedTypes::OP_D2:
      return S_OP_D2; 
      break;
    case IntrepidExtendedTypes::OP_D3:
      return S_OP_D3; 
      break;
    case IntrepidExtendedTypes::OP_D4:
      return S_OP_D4; 
      break;
    case IntrepidExtendedTypes::OP_D5:
      return S_OP_D5; 
      break;
    case IntrepidExtendedTypes::OP_D6:
      return S_OP_D6; 
      break;
    case IntrepidExtendedTypes::OP_D7:
      return S_OP_D7; 
      break;
    case IntrepidExtendedTypes::OP_D8:
      return S_OP_D8; 
      break;
    case IntrepidExtendedTypes::OP_D9:
      return S_OP_D9; 
      break;
    case IntrepidExtendedTypes::OP_D10:
      return S_OP_D10; 
      break;
    case IntrepidExtendedTypes::OP_X:
      return S_OP_X; 
      break;
    case IntrepidExtendedTypes::OP_Y:
      return S_OP_Y; 
      break;
    case IntrepidExtendedTypes::OP_Z:
      return S_OP_Z; 
      break;
    case IntrepidExtendedTypes::OP_DX:
      return S_OP_DX; 
      break;
    case IntrepidExtendedTypes::OP_DY:
      return S_OP_DY; 
      break;
    case IntrepidExtendedTypes::OP_DZ:
      return S_OP_DZ; 
      break;
    case IntrepidExtendedTypes::OP_CROSS_NORMAL:
      return S_OP_CROSS_NORMAL; 
      break;
    case IntrepidExtendedTypes::OP_DOT_NORMAL:
      return S_OP_DOT_NORMAL; 
      break;
    case IntrepidExtendedTypes::OP_TIMES_NORMAL:
      return S_OP_TIMES_NORMAL; 
      break;
    case IntrepidExtendedTypes::OP_TIMES_NORMAL_X:
      return S_OP_TIMES_NORMAL_X; 
      break;
    case IntrepidExtendedTypes::OP_TIMES_NORMAL_Y:
      return S_OP_TIMES_NORMAL_Y; 
      break;
    case IntrepidExtendedTypes::OP_TIMES_NORMAL_Z:
      return S_OP_TIMES_NORMAL_Z; 
      break;
    case IntrepidExtendedTypes::OP_VECTORIZE_VALUE:
      return S_OP_VECTORIZE_VALUE; 
      break;
    default:
      return S_OP_UNKNOWN;
      break;
  }
}

void BilinearForm::printTrialTestInteractions() {
  for (vector<int>::iterator testIt = _testIDs.begin(); testIt != _testIDs.end(); testIt++) {
    int testID = *testIt;
    cout << endl << "b(U," << testName(testID) << ") &= " << endl;
    bool first = true;
    int spaceDim = 2;
    FieldContainer<double> point(1,2); // (0,0)
    FieldContainer<double> testValueScalar(1,1,1); // 1 cell, 1 basis function, 1 point...
    FieldContainer<double> testValueVector(1,1,1,spaceDim); // 1 cell, 1 basis function, 1 point, spaceDim dimensions...
    FieldContainer<double> trialValueScalar(1,1,1); // 1 cell, 1 basis function, 1 point...
    FieldContainer<double> trialValueVector(1,1,1,spaceDim); // 1 cell, 1 basis function, 1 point, spaceDim dimensions...
    FieldContainer<double> testValue, trialValue;
    for (vector<int>::iterator trialIt = _trialIDs.begin(); trialIt != _trialIDs.end(); trialIt++) {
      int trialID = *trialIt;
      vector<EOperatorExtended> trialOperators, testOperators;
      trialTestOperators(trialID, testID, trialOperators, testOperators);
      vector<EOperatorExtended>::iterator trialOpIt, testOpIt;
      testOpIt = testOperators.begin();
      int operatorIndex = 0;
      for (trialOpIt = trialOperators.begin(); trialOpIt != trialOperators.end(); trialOpIt++) {
        IntrepidExtendedTypes::EOperatorExtended opTrial = *trialOpIt;
        IntrepidExtendedTypes::EOperatorExtended opTest = *testOpIt;
        int trialRank = operatorRank(opTrial, functionSpaceForTrial(trialID));
        int testRank = operatorRank(opTest, functionSpaceForTest(testID));
        trialValue = ( trialRank == 0 ) ? trialValueScalar : trialValueVector;
        testValue = (testRank == 0) ? testValueScalar : testValueVector;
        
        trialValue[0] = 1.0; testValue[0] = 1.0;
        FieldContainer<double> testWeight(1), trialWeight(1); // for storing values that come back from applyBilinearForm
        applyBilinearFormData(trialValue,testValue,trialID,testID,operatorIndex,point);
        if ((trialRank==1) && (trialValue.rank() == 3)) { // vector that became a scalar (a dot product)
          trialWeight.resize(spaceDim);
          trialWeight[0] = trialValue[0];
          for (int dim=1; dim<spaceDim; dim++) {
            trialValue = trialValueVector;
            trialValue.initialize(0.0);
            testValue = (testRank == 0) ? testValueScalar : testValueVector;
            trialValue[dim] = 1.0;
            applyBilinearFormData(trialValue,testValue,trialID,testID,operatorIndex,point);
            trialWeight[dim] = trialValue[0];
          }
        } else {
          trialWeight[0] = trialValue[0];
        }
        // same thing, but now for testWeight
        if ((testRank==1) && (testValue.rank() == 3)) { // vector that became a scalar (a dot product)
          testWeight.resize(spaceDim);
          testWeight[0] = trialValue[0];
          for (int dim=1; dim<spaceDim; dim++) {
            testValue = testValueVector;
            testValue.initialize(0.0);
            trialValue = (trialRank == 0) ? trialValueScalar : trialValueVector;
            testValue[dim] = 1.0;
            applyBilinearFormData(trialValue,testValue,trialID,testID,operatorIndex,point);
            testWeight[dim] = testValue[0];
          }
        } else {
          testWeight[0] = testValue[0];
        }
        if ((testWeight.size() == 2) && (trialWeight.size() == 2)) { // both vector values (unsupported)
          TEUCHOS_TEST_FOR_EXCEPTION( true, std::invalid_argument, "unsupported form." );
        } else {
          // scalar & vector: combine into one, in testWeight
          if ( (trialWeight.size() + testWeight.size()) == 3) {
            FieldContainer<double> smaller = (trialWeight.size()==1) ? trialWeight : testWeight;
            FieldContainer<double> bigger =  (trialWeight.size()==2) ? trialWeight : testWeight;
            testWeight.resize(spaceDim);
            for (int dim=0; dim<spaceDim; dim++) {
              testWeight[dim] = smaller[0] * bigger[dim];
            }
          } else { // both scalars: combine into one, in testWeight
            testWeight[0] *= trialWeight[0];
          }
        }
        if (testWeight.size() == 1) { // scalar weight
          if ( testWeight[0] == -1.0 ) {
            cout << " - ";
          } else {
            if (testWeight[0] == 1.0) {
              if (! first) cout << " + ";
            } else {
              if (testWeight[0] < 0.0) {
                cout << testWeight[0] << " ";
              } else {
                cout << " + " << testWeight[0] << " ";
              }
            }
          }
          if (! isFluxOrTrace(trialID) ) {
            cout << "\\int_{K} " ;
          } else {
            cout << "\\int_{\\partial K} " ;            
          }
          cout << operatorName(opTrial) << trialName(trialID) << " ";
        } else { // 
          if (! first) cout << " + ";
          if (! isFluxOrTrace(trialID) ) {
            cout << "\\int_{K} " ;
          } else {
            cout << "\\int_{\\partial K} " ;
          }
          if (opTrial != OP_TIMES_NORMAL) {
            cout << " \\begin{bmatrix}";
            for (int dim=0; dim<spaceDim; dim++) {
              if (testWeight[dim] != 1.0) {
                cout << testWeight[0];
              }
              if (dim != spaceDim-1) {
                cout << " \\\\ ";
              }
            }
            cout << "\\end{bmatrix} ";
            cout << trialName(trialID);
            cout << " \\cdot ";
          } else if (opTrial == OP_TIMES_NORMAL) {
            if (testWeight.size() == 2) {
              cout << " {";
              if (testWeight[0] != 1.0) {
                cout << testWeight[0];
              }
              cout << " n_1 " << " \\choose ";
              if (testWeight[1] != 1.0) {
                cout << testWeight[1];
              }
              cout << " n_2 " << "} " << trialName(trialID) << " \\cdot ";
            } else {
              if (testWeight[0] != 1.0) {
                cout << testWeight[0] << " " << trialName(trialID) << operatorName(opTrial);
              } else {
                cout << trialName(trialID) << operatorName(opTrial);
              }
            }
          }
        }
        if ((opTest == OP_CROSS_NORMAL) || (opTest == OP_DOT_NORMAL)) {
          // reverse the order:
          cout << testName(testID) << operatorName(opTest);
        } else {
          cout << operatorName(opTest) << testName(testID);
        }
        first = false;
        testOpIt++;
        operatorIndex++;
      }
    }
    cout << endl << "\\\\";
  }
}

const set<int> & BilinearForm::normalOperators() {
  if (_normalOperators.size() == 0) {
    _normalOperators.insert(OP_CROSS_NORMAL);
    _normalOperators.insert(OP_DOT_NORMAL);
    _normalOperators.insert(OP_TIMES_NORMAL);
    _normalOperators.insert(OP_TIMES_NORMAL_X);
    _normalOperators.insert(OP_TIMES_NORMAL_Y);
    _normalOperators.insert(OP_TIMES_NORMAL_Z);
  }
  return _normalOperators;
}

void BilinearForm::setUseSPDSolveForOptimalTestFunctions(bool value) {
  _useSPDSolveForOptimalTestFunctions = value;
}

void BilinearForm::setUseIterativeRefinementsWithSPDSolve(bool value) {
  _useIterativeRefinementsWithSPDSolve = value;
}

void BilinearForm::setUseExtendedPrecisionSolveForOptimalTestFunctions(bool value) {
  cout << "WARNING: BilinearForm no longer supports extended precision solve for optimal test functions.  Ignoring argument to setUseExtendedPrecisionSolveForOptimalTestFunctions().\n";
}

void BilinearForm::setWarnAboutZeroRowsAndColumns(bool value) {
  _warnAboutZeroRowsAndColumns = value;
}

VarFactory BilinearForm::varFactory() {
  // this is not meant to cover every possible subclass, but the known legacy subclasses.
  // (just here to allow compatibility with subclasses in DPGTests, e.g.; new implementations should use BF)
  VarFactory vf;
  vector<int> trialIDs = this->trialIDs();
  for (int trialIndex=0; trialIndex<trialIDs.size(); trialIndex++) {
    int trialID = trialIDs[trialIndex];
    string name = this->trialName(trialID);
    VarPtr trialVar;
    if (isFluxOrTrace(trialID)) {
      bool isFlux = this->functionSpaceForTrial(trialID) == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL;
      if (isFlux) {
        trialVar = vf.fluxVar(name);
      } else {
        trialVar = vf.traceVar(name);
      }
    } else {
      trialVar = vf.fieldVar(name);
    }
  }
  
  vector<int> testIDs = this->testIDs();
  for (int testIndex=0; testIndex<testIDs.size(); testIndex++) {
    int testID = testIDs[testIndex];
    string name = this->testName(testID);
    VarPtr testVar;
    IntrepidExtendedTypes::EFunctionSpaceExtended fs = this->functionSpaceForTest(testID);
    Space space;
    switch (fs) {
      case IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD:
        space = HGRAD;
        break;
      case IntrepidExtendedTypes::FUNCTION_SPACE_HCURL:
        space = HCURL;
        break;
      case IntrepidExtendedTypes::FUNCTION_SPACE_HDIV:
        space = HDIV;
        break;
      case IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD_DISC:
        space = HGRAD_DISC;
        break;
      case IntrepidExtendedTypes::FUNCTION_SPACE_HCURL_DISC:
        space = HCURL_DISC;
        break;
      case IntrepidExtendedTypes::FUNCTION_SPACE_HDIV_DISC:
        space = HDIV_DISC;
        break;
      case IntrepidExtendedTypes::FUNCTION_SPACE_HVOL:
        space = L2;
        break;
        
      default:
        cout << "BilinearForm::varFactory(): unhandled function space.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BilinearForm::varFactory(): unhandled function space.");
        break;
    }
    testVar = vf.testVar(name, space);
  }
  return vf;
}