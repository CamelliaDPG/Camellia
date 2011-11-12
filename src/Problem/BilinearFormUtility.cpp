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

#include "BasisFactory.h"
#include "DPGInnerProduct.h"

// Intrepid includes
#include "Intrepid_CellTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "Teuchos_LAPACK.hpp"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_DataAccess.h"

#include "BilinearFormUtility.h"
#include "BasisValueCache.h"

bool BilinearFormUtility::checkForZeroRowsAndColumns(string name, FieldContainer<double> &array) {
  // for now, only support rank 3 FCs 
  double tol = 1e-15;
  if ( array.rank() != 3) {
    TEST_FOR_EXCEPTION( array.rank() != 3, std::invalid_argument, "checkForZeroRowsAndColumns only supports rank-3 FieldContainers.");
  }
  int numCells = array.dimension(0);
  int numRows = array.dimension(1);
  int numCols = array.dimension(2);
  bool zeroRowOrColFound = false;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int i=0; i<numRows; i++) {
      bool nonZeroFound = false;
      int j=0;
      while ((!nonZeroFound) && (j<numCols)) {
        if (abs(array(cellIndex,i,j)) > tol) nonZeroFound = true;
        j++;
      }
      if ( ! nonZeroFound ) {
        cout << "warning: in matrix " << name << " for cell " << cellIndex << ", row " << i << " is all zeros." << endl;
        zeroRowOrColFound = true;
      }
    }
    for (int j=0; j<numCols; j++) {
      bool nonZeroFound = false;
      int i=0;
      while ((!nonZeroFound) && (i<numRows)) {
        if (abs(array(cellIndex,i,j)) > tol) nonZeroFound = true;
        i++;
      }
      if ( ! nonZeroFound ) {
        cout << "warning: in matrix " << name << " for cell " << cellIndex << ", column " << j << " is all zeros." << endl;
        zeroRowOrColFound = true;
      }
    }
  }
  return !zeroRowOrColFound; // return TRUE if no zero row or col found
}

void BilinearFormUtility::transposeFCMatrices(FieldContainer<double> &fcTranspose,
                                              const FieldContainer<double> &fc) {
  // check dimensions
  TEST_FOR_EXCEPTION( ( fc.dimension(0) != fcTranspose.dimension(0) ),
                     std::invalid_argument,
                     "fc.dimension(0) and fcTranspose.dimension(0) (numCells) do not match.");
  TEST_FOR_EXCEPTION( ( fc.dimension(1) != fcTranspose.dimension(2) ),
                     std::invalid_argument,
                     "fc.dimension(1) and fcTranspose.dimension(2) (numRows) do not match.");
  TEST_FOR_EXCEPTION( ( fc.dimension(2) != fcTranspose.dimension(1) ),
                     std::invalid_argument,
                     "fc.dimension(2) and fcTranspose.dimension(1) (numCols) do not match.");
  // transposes (C,i,j) --> (C,j,i)
  int numCells = fc.dimension(0);
  int numRows = fc.dimension(1);
  int numCols = fc.dimension(2);
  for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
    for (int i=0; i < numRows; i++) {
      for (int j=0; j < numCols; j++) {
        fcTranspose(cellIndex,j,i) = fc(cellIndex,i,j);
      }
    }
  }
}

int BilinearFormUtility::computeOptimalTest(FieldContainer<double> &optimalTestWeights,
                                            FieldContainer<double> &innerProductMatrix,
                                            BilinearForm &bilinearForm,
                                            Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                            shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes,
                                            FieldContainer<double> &cellSideParities) {
  // all arguments are as in computeStiffnessMatrix, except:
  // optimalTestWeights, which has dimensions (numCells, numTrialDofs, numTestDofs)
  // innerProduct: the inner product which defines the sense in which these test functions are optimal
  int numCells = physicalCellNodes.dimension(0);
  int numTestDofs = testOrdering->totalDofs();
  int numTrialDofs = trialOrdering->totalDofs();
  
  // check that optimalTestWeights is properly dimensioned....
  TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(0) != numCells ),
                     std::invalid_argument,
                     "physicalCellNodes.dimension(0) and optimalTestWeights.dimension(0) (numCells) do not match.");
  TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(1) != numTrialDofs ),
                     std::invalid_argument,
                     "trialOrdering->totalDofs() and optimalTestWeights.dimension(1) do not match.");
  TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(2) != numTestDofs ),
                     std::invalid_argument,
                     "testOrdering->totalDofs() and optimalTestWeights.dimension(2) do not match.");
  
  FieldContainer<double> stiffnessMatrix(numCells,numTestDofs,numTrialDofs);
  FieldContainer<double> stiffnessMatrixT(numCells,numTrialDofs,numTestDofs);
  
  // RHS:
  computeStiffnessMatrix(stiffnessMatrix, bilinearForm, trialOrdering, testOrdering,
                         cellTopo, physicalCellNodes, cellSideParities);
  
  transposeFCMatrices(stiffnessMatrixT, stiffnessMatrix);
  
  //cout << "stiffnessMatrixT: " << stiffnessMatrixT << endl;
  //cout << "stiffnessMatrix:" << stiffnessMatrix << endl;
  
  Epetra_SerialDenseSolver solver;
  
  int solvedAll = 0;
  
  for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
    // changed to Copy from View for debugging...
    Epetra_SerialDenseMatrix ipMatrixT(Copy,
                                       &innerProductMatrix(cellIndex,0,0),
                                       innerProductMatrix.dimension(2), // stride -- fc stores in row-major order (a.o.t. SDM)
                                       innerProductMatrix.dimension(2),innerProductMatrix.dimension(1));
    
    Epetra_SerialDenseMatrix stiffness(Copy,
                                       &stiffnessMatrixT(cellIndex,0,0),
                                       stiffnessMatrixT.dimension(2), // stride
                                       stiffnessMatrixT.dimension(2),stiffnessMatrixT.dimension(1));
    
    Epetra_SerialDenseMatrix optimalWeightsT(numTestDofs, numTrialDofs);
    
    
    solver.SetMatrix(ipMatrixT);
    //    solver.SolveWithTranspose(true); // not that it should matter -- ipMatrix should be symmetric
    int successLocal = solver.SetVectors(optimalWeightsT, stiffness);
    
    if (successLocal != 0) {
      cout << "computeOptimalTest: failed to SetVectors with error " << successLocal << endl;
    }
    
    bool equilibrated = false;
    if ( solver.ShouldEquilibrate() ) {
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();
      equilibrated = true;
    }
    
    successLocal = solver.Solve();
    
    if (successLocal != 0) {
      cout << "computeOptimalTest: Solve FAILED with error: " << successLocal << endl;
      solvedAll = successLocal;
    }
    
    if (equilibrated) {
      successLocal = solver.UnequilibrateLHS();
      if (successLocal != 0) {
        cout << "computeOptimalTest: unequilibration FAILED with error: " << successLocal << endl;
        solvedAll = successLocal;
      }
    }
    
    for (int i=0; i<optimalTestWeights.dimension(1); i++) {
      for (int j=0; j<optimalTestWeights.dimension(2); j++) {
        optimalTestWeights(cellIndex,i,j) = optimalWeightsT(j,i);
      }
    }
    
    // double oneNorm = ipMatrixT.OneNorm();
    
    //cout << "computeOptimalTest: ipMatrix.oneNorm = " << oneNorm << endl;
    
  }
  return solvedAll;
  
}

int BilinearFormUtility::computeOptimalTest(FieldContainer<double> &optimalTestWeights,
                                             DPGInnerProduct &innerProduct,
                                             BilinearForm &bilinearForm,
                                             Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                             shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes,
                                             FieldContainer<double> &cellSideParities) {
  // all arguments are as in computeStiffnessMatrix, except:
  // optimalTestWeights, which has dimensions (numCells, numTrialDofs, numTestDofs)
  // innerProduct: the inner product which defines the sense in which these test functions are optimal
  int numCells = physicalCellNodes.dimension(0);
  int numTestDofs = testOrdering->totalDofs();
  
  FieldContainer<double> innerProductMatrix(numCells,numTestDofs,numTestDofs);
  
  // LHS:
  innerProduct.computeInnerProductMatrix(innerProductMatrix, testOrdering,
                                          cellTopo, physicalCellNodes);
  
  return computeOptimalTest(optimalTestWeights,innerProductMatrix,bilinearForm,trialOrdering,
                            testOrdering,cellTopo,physicalCellNodes, cellSideParities);
}

void BilinearFormUtility::computeStiffnessMatrix(FieldContainer<double> &stiffness, 
                                                 FieldContainer<double> &innerProductMatrix,
                                                 FieldContainer<double> &optimalTestWeights) {
  // stiffness has dimensions (numCells, numTrialDofs, numTrialDofs)
  // innerProductMatrix has dim. (numCells, numTestDofs, numTestDofs)
  // optimalTestWeights has dim. (numCells, numTrialDofs, numTestDofs)
  // all this does is computes stiffness = weights^T * innerProductMatrix * weights
  int numCells = stiffness.dimension(0);
  int numTrialDofs = stiffness.dimension(1);
  int numTestDofs = innerProductMatrix.dimension(1);
  
  // check that all the dimensions are compatible:
  TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(0) != numCells ),
                     std::invalid_argument,
                     "stiffness.dimension(0) and optimalTestWeights.dimension(0) (numCells) do not match.");
  TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(1) != numTrialDofs ),
                     std::invalid_argument,
                     "numTrialDofs and optimalTestWeights.dimension(1) do not match.");
  TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(2) != numTestDofs ),
                     std::invalid_argument,
                     "numTestDofs and optimalTestWeights.dimension(2) do not match.");
  TEST_FOR_EXCEPTION( ( innerProductMatrix.dimension(2) != innerProductMatrix.dimension(1) ),
                     std::invalid_argument,
                     "innerProductMatrix.dimension(1) and innerProductMatrix.dimension(2) do not match.");
  
  TEST_FOR_EXCEPTION( ( stiffness.dimension(1) != stiffness.dimension(2) ),
                     std::invalid_argument,
                     "stiffness.dimension(1) and stiffness.dimension(2) do not match.");
  
  stiffness.initialize(0);
  
  for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
    Epetra_SerialDenseMatrix weightsT(Copy,
                                     &optimalTestWeights(cellIndex,0,0),
                                     optimalTestWeights.dimension(2), // stride
                                     optimalTestWeights.dimension(2),optimalTestWeights.dimension(1));
    
    Epetra_SerialDenseMatrix ipMatrixT(Copy,
                                      &innerProductMatrix(cellIndex,0,0),
                                      innerProductMatrix.dimension(2), // stride
                                      innerProductMatrix.dimension(2),innerProductMatrix.dimension(1));
    
    Epetra_SerialDenseMatrix   stiffT (View,
                                      &stiffness(cellIndex,0,0),
                                      stiffness.dimension(2), // stride
                                      stiffness.dimension(2),stiffness.dimension(1));
    
    Epetra_SerialDenseMatrix intermediate( numTrialDofs, numTestDofs );
    
    // account for the fact that SDM is column-major and FC is row-major: 
    //   (weightsT) * (ipMatrixT)^T * (weightsT)^T
    int success = intermediate.Multiply('T','T',1.0,weightsT,ipMatrixT,0.0);
    
    if (success != 0) {
      cout << "computeStiffnessMatrix: intermediate.Multiply() failed with error code " << success << endl;
    }
    
    success = stiffT.Multiply('N','N',1.0,intermediate,weightsT,0.0);
    // stiffT is technically the transpose of stiffness, but the construction A^T * B * A is symmetric even in general...
    
    if (success != 0) {
      cout << "computeStiffnessMatrix: stiffT.Multiply() failed with error code " << success << endl;
    }
  }
  
  if ( ! checkForZeroRowsAndColumns("stiffness",stiffness) ) {
    //cout << "stiffness: " << stiffness;
  }
}

void BilinearFormUtility::computeStiffnessMatrix(FieldContainer<double> &stiffness, BilinearForm &bilinearForm,
                                                 Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering, 
                                                 shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes,
                                                 FieldContainer<double> &cellSideParities) {
  
  // physicalCellNodes: the nodal points for the element(s) with topology cellTopo
  //                 The dimensions are (numCells, numNodesPerElement, spaceDimension)
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

  unsigned numCells = physicalCellNodes.dimension(0);
  unsigned numNodesPerElem = physicalCellNodes.dimension(1);
  unsigned spaceDim = physicalCellNodes.dimension(2);
  
  //cout << "trialOrdering: " << *trialOrdering;
  //cout << "testOrdering: " << *testOrdering;
  
  // Check that cellTopo and physicalCellNodes agree
  TEST_FOR_EXCEPTION( ( numNodesPerElem != cellTopo.getNodeCount() ),
                     std::invalid_argument,
                     "Second dimension of physicalCellNodes and cellTopo.getNodeCount() do not match.");
  TEST_FOR_EXCEPTION( ( spaceDim != cellTopo.getDimension() ),
                     std::invalid_argument,
                     "Third dimension of physicalCellNodes and cellTopo.getDimension() do not match.");
  // check stiffness dimensions:
  TEST_FOR_EXCEPTION( ( numCells != stiffness.dimension(0) ),
                     std::invalid_argument,
                     "numCells and stiffness.dimension(0) do not match.");
  TEST_FOR_EXCEPTION( ( numTestDofs != stiffness.dimension(1) ),
                     std::invalid_argument,
                     "numTestDofs and stiffness.dimension(1) do not match.");
  TEST_FOR_EXCEPTION( ( numTrialDofs != stiffness.dimension(2) ),
                     std::invalid_argument,
                     "numTrialDofs and stiffness.dimension(2) do not match.");
  
  // 0. Set up BasisValueCache
  int cubDegreeTrial = trialOrdering->maxBasisDegree();
  int cubDegreeTest = testOrdering->maxBasisDegree();
  int cubDegree = cubDegreeTrial + cubDegreeTest;

  BasisValueCache basisCache(physicalCellNodes, cellTopo, *trialOrdering, cubDegreeTest, true); // DO create side caches, too
  
  unsigned numSides = cellTopo.getSideCount();
  
  // 3. For each (test, trial) combination:
  vector<int> testIDs = bilinearForm.testIDs();
  vector<int>::iterator testIterator;
  
  vector<int> trialIDs = bilinearForm.trialIDs();
  vector<int>::iterator trialIterator;
  
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > trialBasis;
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > testBasis;
  
  stiffness.initialize(0.0);
  
  for (testIterator = testIDs.begin(); testIterator != testIDs.end(); testIterator++) {
    int testID = *testIterator;
  
    for (trialIterator = trialIDs.begin(); trialIterator != trialIDs.end(); trialIterator++) {
      int trialID = *trialIterator;
      
      vector<EOperatorExtended> trialOperators, testOperators;
      bilinearForm.trialTestOperators(trialID, testID, trialOperators, testOperators);
      vector<EOperatorExtended>::iterator trialOpIt, testOpIt;
      testOpIt = testOperators.begin();
      TEST_FOR_EXCEPTION(trialOperators.size() != testOperators.size(), std::invalid_argument,
                         "trialOperators and testOperators must be the same length");
      int operatorIndex = -1;
      for (trialOpIt = trialOperators.begin(); trialOpIt != trialOperators.end(); trialOpIt++) {
        operatorIndex++;
        EOperatorExtended trialOperator = *trialOpIt;
        EOperatorExtended testOperator = *testOpIt;
        
        if (testOperator==OPERATOR_TIMES_NORMAL) {
          TEST_FOR_EXCEPTION(true,std::invalid_argument,"OPERATOR_TIMES_NORMAL not supported for tests.  Use for trial only");
        }
        
        Teuchos::RCP < const FieldContainer<double> > testValuesTransformed;
        Teuchos::RCP < const FieldContainer<double> > trialValuesTransformed;
        Teuchos::RCP < const FieldContainer<double> > testValuesTransformedWeighted;
        
        //cout << "trial is " <<  bilinearForm.trialName(trialID) << "; test is " << bilinearForm.testName(testID) << endl;
        
        if (! bilinearForm.isFluxOrTrace(trialID)) {
          trialBasis = trialOrdering->getBasis(trialID);
          testBasis = testOrdering->getBasis(testID);
          
          FieldContainer<double> miniStiffness( numCells, testBasis->getCardinality(), trialBasis->getCardinality() );

          trialValuesTransformed = basisCache.getTransformedValues(trialBasis,trialOperator);
          testValuesTransformedWeighted = basisCache.getTransformedWeightedValues(testBasis,testOperator);
          
          FieldContainer<double> physicalCubaturePoints = basisCache.getPhysicalCubaturePoints();
          FieldContainer<double> materialDataAppliedToTrialValues = *trialValuesTransformed; // copy first
          FieldContainer<double> materialDataAppliedToTestValues = *testValuesTransformedWeighted; // copy first
          bilinearForm.applyBilinearFormData(materialDataAppliedToTrialValues, materialDataAppliedToTestValues,
                                             trialID,testID,operatorIndex,physicalCubaturePoints);
          
          //integrate:
          FunctionSpaceTools::integrate<double>(miniStiffness,materialDataAppliedToTestValues,materialDataAppliedToTrialValues,COMP_CPP);
          // place in the appropriate spot in the element-stiffness matrix
          // copy goes from (cell,trial_basis_dof,test_basis_dof) to (cell,element_trial_dof,element_test_dof)
          
          //cout << "miniStiffness for volume:\n" << miniStiffness;
          
          //checkForZeroRowsAndColumns("miniStiffness for pre-stiffness", miniStiffness);
          
          //cout << "trialValuesTransformed for trial " << bilinearForm.trialName(trialID) << endl << trialValuesTransformed
          //cout << "testValuesTransformed for test " << bilinearForm.testName(testID) << ": \n" << testValuesTransformed;
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
          
          TEST_FOR_EXCEPTION( ( trialBasisRank != 0 ),
                             std::invalid_argument,
                             "Boundary trial variable (flux or trace) given with non-scalar basis.  Unsupported.");
          
          for (unsigned sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
            trialBasis = trialOrdering->getBasis(trialID,sideOrdinal);
            testBasis = testOrdering->getBasis(testID);
            
            FieldContainer<double> miniStiffness( numCells, testBasis->getCardinality(), trialBasis->getCardinality() );    

            // for trial: the value lives on the side, so we don't use the volume coords either:
            trialValuesTransformed = basisCache.getTransformedValues(trialBasis,trialOperator,sideOrdinal,false);
            // for test: do use the volume coords:
            testValuesTransformed = basisCache.getTransformedValues(testBasis,testOperator,sideOrdinal,true);
            // 
            testValuesTransformedWeighted = basisCache.getTransformedWeightedValues(testBasis,testOperator,sideOrdinal,true);
            
            // copy before manipulating trialValues--these are the ones stored in the cache, so we're not allowed to change them!!
            FieldContainer<double> materialDataAppliedToTrialValues = *trialValuesTransformed;
            if ((testOperator != OPERATOR_CROSS_NORMAL) && (testOperator != OPERATOR_DOT_NORMAL) 
                && (trialOperator != OPERATOR_TIMES_NORMAL)) {
              // we take this to be a flux: since the normal hasn't entered a boundary integral, we assume it's part of the trial variable definition
              // this is a flux ==> take cell parity into account (because then there must be a normal folded into the flux definition)
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
           
            FieldContainer<double> cubPointsSidePhysical = basisCache.getPhysicalCubaturePointsForSide(sideOrdinal);
            FieldContainer<double> materialDataAppliedToTestValues = *testValuesTransformedWeighted; // copy first
            bilinearForm.applyBilinearFormData(materialDataAppliedToTrialValues,materialDataAppliedToTestValues,
                                               trialID,testID,operatorIndex,cubPointsSidePhysical);
            
            
            //cout << "sideOrdinal: " << sideOrdinal << "; cubPointsSidePhysical" << endl << cubPointsSidePhysical;
            
            //   d. Sum up (integrate) and place in stiffness matrix according to DofOrdering indices
            FunctionSpaceTools::integrate<double>(miniStiffness,materialDataAppliedToTestValues,materialDataAppliedToTrialValues,COMP_CPP);
            
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
  //cout << "trialOrdering: \n" << *trialOrdering;
  //cout << "testOrdering: \n" << *testOrdering;
  checkForZeroRowsAndColumns("pre-stiffness", stiffness);
}

void BilinearFormUtility::computeOptimalStiffnessMatrix(FieldContainer<double> &stiffness, 
                                                        FieldContainer<double> &optimalTestWeights,
                                                        BilinearForm &bilinearForm,
                                                        Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                                        shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes,
                                                        FieldContainer<double> &cellSideParities) {
  // lots of code copied and pasted from the very similar computeStiffnessMatrix.  The difference here is that for each optimal test function,
  // we need to ask the bilinear form about each of its components (it's a vector whereas the other guy had just a single basis function
  // for each...), and then apply the appropriate weights....
  // physicalCellNodes: the nodal points for the element(s) with topology cellTopo
  //                 The dimensions are (numCells, numNodesPerElement, spaceDimension)
  // optimalTestWeights dimensions are: (numCells, numTrial, numTest) -- numTrial is the optTest index
  // stiffness dimensions are: (numCells, # trialOrdering Dofs, # trialOrdering Dofs)
  // (while (cell,trial,test) is more natural conceptually, I believe the above ordering makes
  //  more sense given the inversion that we must do to compute the optimal test functions...)
  
  // steps:
  // 0. Set up BasisValueCache
  // 3. For each (test, trial) combination:
  //   a. Apply the specified operators to the basis in the DofOrdering, at the cubature points
  //   b. Multiply the two bases together, weighted with Jacobian/Piola transform and cubature weights
  //   c. Pass the result to bilinearForm's applyBilinearFormData method
  //   d. Sum up (integrate) and place in stiffness matrix according to DofOrdering indices
  
  // 0. Set up Cubature
  
  unsigned numCells = physicalCellNodes.dimension(0);
  unsigned numNodesPerElem = physicalCellNodes.dimension(1);
  unsigned spaceDim = physicalCellNodes.dimension(2);
  
  // Check that cellTopo and physicalCellNodes agree
  TEST_FOR_EXCEPTION( ( numNodesPerElem != cellTopo.getNodeCount() ),
                     std::invalid_argument,
                     "Second dimension of physicalCellNodes and cellTopo.getNodeCount() do not match.");
  TEST_FOR_EXCEPTION( ( spaceDim != cellTopo.getDimension() ),
                     std::invalid_argument,
                     "Third dimension of physicalCellNodes and cellTopo.getDimension() do not match.");
  
  int numOptTestFunctions = optimalTestWeights.dimension(1); // should also == numTrialDofs
  
  TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(1) != stiffness.dimension(2) ),
                     std::invalid_argument,
                     "optimalTestWeights.dimension(1) (=" << optimalTestWeights.dimension(1) << ") and stiffness.dimension(2) (=" << stiffness.dimension(2) << ") do not match.");
  TEST_FOR_EXCEPTION( ( stiffness.dimension(1) != stiffness.dimension(2) ),
                     std::invalid_argument,
                     "stiffness.dimension(1) (=" << stiffness.dimension(1) << ") and stiffness.dimension(2) (=" << stiffness.dimension(2) << ") do not match.");
  
  // Set up BasisValueCache
  int cubDegreeTrial = trialOrdering->maxBasisDegree();
  int cubDegreeTest = testOrdering->maxBasisDegree();
  int cubDegree = cubDegreeTrial + cubDegreeTest;
  
  BasisValueCache basisCache(physicalCellNodes, cellTopo, *trialOrdering, cubDegreeTest, true); // DO create side caches, too
  
  unsigned numSides = cellTopo.getSideCount();

  vector<int> testIDs = bilinearForm.testIDs();
  vector<int>::iterator testIterator;
  
  vector<int> trialIDs = bilinearForm.trialIDs();
  vector<int>::iterator trialIterator;
  
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > trialBasis;
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > testBasis;
  
  stiffness.initialize(0.0);
  
  for (trialIterator = trialIDs.begin(); trialIterator != trialIDs.end(); trialIterator++) {
    int trialID = *trialIterator;
    
    for (int optTestIndex=0; optTestIndex < numOptTestFunctions; optTestIndex++) {
      FieldContainer<double> weights(numCells,testOrdering->totalDofs());
      for (unsigned i=0; i<numCells; i++) {
        for (int j=0; j<testOrdering->totalDofs(); j++) {
          weights(i,j) = optimalTestWeights(i,optTestIndex,j);
        }
      }
      for (testIterator = testIDs.begin(); testIterator != testIDs.end(); testIterator++) {
        int testID = *testIterator;
        
        vector<EOperatorExtended> trialOperators, testOperators;
        bilinearForm.trialTestOperators(trialID, testID, trialOperators, testOperators);
        vector<EOperatorExtended>::iterator trialOpIt, testOpIt;
        testOpIt = testOperators.begin();
        TEST_FOR_EXCEPTION(trialOperators.size() != testOperators.size(), std::invalid_argument,
                           "trialOperators and testOperators must be the same length");
        int operatorIndex = -1;
        for (trialOpIt = trialOperators.begin(); trialOpIt != trialOperators.end(); trialOpIt++) {
          EOperatorExtended trialOperator = *trialOpIt;
          EOperatorExtended testOperator = *testOpIt;
          operatorIndex++;
          
          if (testOperator==OPERATOR_TIMES_NORMAL) {
            TEST_FOR_EXCEPTION(true,std::invalid_argument,"OPERATOR_TIMES_NORMAL not supported for tests.  Use for trial only");
          }
          
          Teuchos::RCP < const FieldContainer<double> > testValuesTransformed;
          Teuchos::RCP < const FieldContainer<double> > trialValuesTransformed;
          Teuchos::RCP < const FieldContainer<double> > testValuesTransformedWeighted;

          if (! bilinearForm.isFluxOrTrace(trialID)) {
            trialBasis = trialOrdering->getBasis(trialID);
            testBasis = testOrdering->getBasis(testID);
            FieldContainer<double> miniStiffness( numCells, testBasis->getCardinality(), trialBasis->getCardinality() );
            
            trialValuesTransformed = basisCache.getTransformedValues(trialBasis,trialOperator);
            testValuesTransformedWeighted = basisCache.getTransformedWeightedValues(testBasis,testOperator);
            
            FieldContainer<double> physicalCubaturePoints = basisCache.getPhysicalCubaturePoints();
            FieldContainer<double> materialDataAppliedToTrialValues = *trialValuesTransformed; // copy first
            FieldContainer<double> materialDataAppliedToTestValues = *testValuesTransformedWeighted; // copy first
            bilinearForm.applyBilinearFormData(materialDataAppliedToTrialValues,materialDataAppliedToTestValues, 
                                               trialID,testID,operatorIndex,physicalCubaturePoints);
              
            int testDofOffset = testOrdering->getDofIndex(testID,0);
            // note that weightCellBasisValues does depend on contiguous test basis dofs...
            // (this is the plan, since there shouldn't be any kind of identification between different test dofs,
            //  especially since test functions live only inside the cell)
            weightCellBasisValues(materialDataAppliedToTestValues, weights, testDofOffset);
              
            FunctionSpaceTools::integrate<double>(miniStiffness,materialDataAppliedToTestValues,materialDataAppliedToTrialValues,COMP_CPP);
            // place in the appropriate spot in the element-stiffness matrix
            // copy goes from (cell,trial_basis_dof,test_basis_dof) to (cell,element_trial_dof,element_test_dof)
            
            // there may be a more efficient way to do this copying:
            // (one strategy would be to reimplement fst::integrate to support offsets, so that no copying needs to be done...)
            for (int i=0; i < testBasis->getCardinality(); i++) {
              for (int j=0; j < trialBasis->getCardinality(); j++) {
                int trialDofIndex = trialOrdering->getDofIndex(trialID,j);
                for (unsigned k=0; k < numCells; k++) {
                  stiffness(k,optTestIndex,trialDofIndex) += miniStiffness(k,i,j);
                }
              }
            }          
          } else {  // boundary integral
            int trialBasisRank = trialOrdering->getBasisRank(trialID);
            int testBasisRank = testOrdering->getBasisRank(testID);
            
            TEST_FOR_EXCEPTION( ( trialBasisRank != 0 ),
                               std::invalid_argument,
                               "Boundary trial variable (flux or trace) given with non-scalar basis.  Unsupported.");
            
            for (unsigned sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
              trialBasis = trialOrdering->getBasis(trialID,sideOrdinal);
              testBasis = testOrdering->getBasis(testID);
              
              FieldContainer<double> miniStiffness( numCells, testBasis->getCardinality(), trialBasis->getCardinality() );
              
              // for trial: we never dot with normal, and the value lives on the side, so we don't use the volume coords either:
              trialValuesTransformed = basisCache.getTransformedValues(trialBasis,trialOperator,sideOrdinal,false);
              // for test: first, don't dot with normal, but do use the volume coords:
              //testValuesTransformed = basisCache.getTransformedValues(testBasis,testOperator,sideOrdinal,true);
              testValuesTransformedWeighted = basisCache.getTransformedWeightedValues(testBasis,testOperator,sideOrdinal,true);
              
              // copy before manipulating trialValues--these are the ones stored in the cache, so we're not allowed to change them!!
              FieldContainer<double> materialDataAppliedToTrialValues = *trialValuesTransformed;
              if ((testOperator != OPERATOR_CROSS_NORMAL) && (testOperator != OPERATOR_DOT_NORMAL)
                  && (testOperator != OPERATOR_TIMES_NORMAL)) {
                // we take this to be a flux: since the normal hasn't entered a boundary integral, we assume it's part of the trial variable definition
                // this being a flux ==> take cell parity into account (because then there must be a normal folded into the flux definition)
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
              
              FieldContainer<double> cubPointsSidePhysical = basisCache.getPhysicalCubaturePointsForSide(sideOrdinal);
              FieldContainer<double> materialDataAppliedToTestValues = *testValuesTransformedWeighted; // copy first
              bilinearForm.applyBilinearFormData(materialDataAppliedToTrialValues,materialDataAppliedToTestValues,
                                                 trialID,testID,operatorIndex,cubPointsSidePhysical);              
              
              int testDofOffset = testOrdering->getDofIndex(testID,0,0);
              weightCellBasisValues(materialDataAppliedToTestValues, weights, testDofOffset);
              
              //   d. Sum up (integrate) and place in stiffness matrix according to DofOrdering indices
              FunctionSpaceTools::integrate<double>(miniStiffness,materialDataAppliedToTestValues,materialDataAppliedToTrialValues,COMP_CPP);
              // place in the appropriate spot in the element-stiffness matrix
              // copy goes from (cell,trial_basis_dof,test_basis_dof) to (cell,element_trial_dof,element_test_dof)
                            
              for (int i=0; i < testBasis->getCardinality(); i++) {
                for (int j=0; j < trialBasis->getCardinality(); j++) {
                  int trialDofIndex = trialOrdering->getDofIndex(trialID,j,sideOrdinal);
                  for (unsigned k=0; k < numCells; k++) {
                    stiffness(k,optTestIndex,trialDofIndex) += miniStiffness(k,i,j);
                  }
                }
              }
            }
          }
          testOpIt++;
        }
      }
    }
  }
}

void BilinearFormUtility::computeRHS(FieldContainer<double> &rhsVector, 
                                     BilinearForm &bilinearForm, RHS &rhs, 
                                     FieldContainer<double> &optimalTestWeights,
                                     Teuchos::RCP<DofOrdering> testOrdering,
                                     shards::CellTopology &cellTopo, 
                                     FieldContainer<double> &physicalCellNodes) {
  // physicalCellNodes: the nodal points for the element(s) with topology cellTopo
  //                 The dimensions are (numCells, numNodesPerElement, spaceDimension)
  // optimalTestWeights dimensions are: (numCells, numTrial, numTest) -- numTrial is the optTest index
  // rhsVector dimensions are: (numCells, # trialOrdering Dofs)

  // steps:
  // 0. Set up Cubature
  // 3. For each optimalTestFunction
  //   a. Apply the value operators to the basis in the DofOrdering, at the cubature points
  //   b. weight with Jacobian/Piola transform and cubature weights
  //   c. Pass the result to RHS to get resultant values at each point
  //   d. Sum up (integrate) and place in rhsVector according to DofOrdering indices

  // 0. Set up Cubature

  unsigned numCells = physicalCellNodes.dimension(0);
  unsigned numNodesPerElem = physicalCellNodes.dimension(1);
  unsigned spaceDim = physicalCellNodes.dimension(2);

  // Check that cellTopo and physicalCellNodes agree
  TEST_FOR_EXCEPTION( ( numNodesPerElem != cellTopo.getNodeCount() ),
                     std::invalid_argument,
                     "Second dimension of physicalCellNodes and cellTopo.getNodeCount() do not match.");
  TEST_FOR_EXCEPTION( ( spaceDim != cellTopo.getDimension() ),
                     std::invalid_argument,
                     "Third dimension of physicalCellNodes and cellTopo.getDimension() do not match.");

  int numOptTestFunctions = optimalTestWeights.dimension(1); // should also == numTrialDofs

  TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(1) != rhsVector.dimension(1) ),
                     std::invalid_argument,
                     "optimalTestWeights.dimension(1) (=" << optimalTestWeights.dimension(1) << ") and rhsVector.dimension(1) (=" << rhsVector.dimension(1) << ") do not match.");

  // Get numerical integration points and weights
  DefaultCubatureFactory<double>  cubFactory;
  int cubDegreeTest = testOrdering->maxBasisDegree();
  int cubDegree = 2*cubDegreeTest;
  BasisValueCache basisCache(physicalCellNodes, cellTopo, cubDegree); // DON'T create side caches, too

  vector<int> testIDs = bilinearForm.testIDs();
  vector<int>::iterator testIterator;

  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > testBasis;

  rhsVector.initialize(0.0);
    
  for (int optTestIndex=0; optTestIndex < numOptTestFunctions; optTestIndex++) {
    FieldContainer<double> weights(numCells,testOrdering->totalDofs());
    for (unsigned i=0; i<numCells; i++) {
      for (int j=0; j<testOrdering->totalDofs(); j++) {
        weights(i,j) = optimalTestWeights(i,optTestIndex,j);
      }
    }
  //cout << "for optTestIndex " << optTestIndex << ", weights: " << endl << weights;
    for (testIterator = testIDs.begin(); testIterator != testIDs.end(); testIterator++) {
      int testID = *testIterator;
      
      EOperatorExtended testOperator = IntrepidExtendedTypes::OPERATOR_VALUE;
      bool notZero = rhs.nonZeroRHS(testID);
      if (notZero) { // compute the integral(s)
        
        testBasis = testOrdering->getBasis(testID);
        
        Teuchos::RCP< const FieldContainer<double> > testValuesTransformedWeighted;
        
        testValuesTransformedWeighted = basisCache.getTransformedWeightedValues(testBasis,testOperator);
        FieldContainer<double> physCubPoints = basisCache.getPhysicalCubaturePoints();
        
        int testDofOffset = testOrdering->getDofIndex(testID,0);
        // note that weightCellBasisValues does depend on contiguous test basis dofs...
        // (this is the plan, since there shouldn't be any kind of identification between different test dofs,
        //  especially since test functions live only inside the cell)
        FieldContainer<double> testValuesTransformedWeightedWeighted = *testValuesTransformedWeighted;
        weightCellBasisValues(testValuesTransformedWeightedWeighted, weights, testDofOffset);

        FieldContainer<double> rhsPointValues; // the rhs method will resize....
        rhs.rhs(testID,physCubPoints,rhsPointValues);
        
        //cout << "rhsPointValues for testID " << testID << ":" << endl << rhsPointValues;
        
        //cout << "d." << endl;
        //   d. Sum up (integrate)
        // to integrate, first multiply the testValues (C,F,P) or (C,F,P,D)
        //               by the rhsPointValues (C,P) or (C,P,D), respectively, and then sum.
        int numPoints = rhsPointValues.dimension(1);
        for (unsigned k=0; k < numCells; k++) {
          for (int i=0; i < testBasis->getCardinality(); i++) {
            for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
              if (rhsPointValues.rank() == 2) {
                rhsVector(k,optTestIndex) += testValuesTransformedWeightedWeighted(k,i,ptIndex) * rhsPointValues(k,ptIndex);
              } else {
                for (int d=0; d<spaceDim; d++) {
                  rhsVector(k,optTestIndex) += testValuesTransformedWeightedWeighted(k,i,ptIndex,d) * rhsPointValues(k,ptIndex,d);
                }
              }
            }
          }
        }
      }
    }
  }
  //cout << "rhsVector: " << endl << rhsVector;
}

void BilinearFormUtility::weightCellBasisValues(FieldContainer<double> &basisValues, const FieldContainer<double> &weights, int offset) {
  // weights are (numCells, offset+numFields)
  // basisValues are (numCells, numFields, ...)
  int numCells = basisValues.dimension(0);
  int numFields = basisValues.dimension(1);
  
  Teuchos::Array<int> dimensions;
  basisValues.dimensions(dimensions);

  int numAffectedValues = 1;
  for (int dimIndex=2; dimIndex<dimensions.size(); dimIndex++) {
    numAffectedValues *= dimensions[dimIndex];
  }
  
  Teuchos::Array<int> index(dimensions.size(),0);
  
  for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
    index[0] = cellIndex;
    for (int fieldIndex=0; fieldIndex < numFields; fieldIndex++) {
      index[1] = fieldIndex;
      int enumIndex = basisValues.getEnumeration(index);
      for (int valIndex=enumIndex; valIndex < numAffectedValues + enumIndex; valIndex++) {
        basisValues[valIndex] *= weights(cellIndex,fieldIndex+offset);
      }
    }
  }
  
}
