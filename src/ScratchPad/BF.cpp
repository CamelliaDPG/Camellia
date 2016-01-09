//
//  TBF.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "BF.h"
#include "VarFactory.h"
#include "BilinearFormUtility.h"
#include "Function.h"
#include "PreviousSolutionFunction.h"
#include "LinearTerm.h"

#include "Intrepid_FunctionSpaceTools.hpp"

#include "SerialDenseWrapper.h"

#include <iostream>
#include <sstream>

using namespace Intrepid;
using namespace std;

namespace Camellia
{
  template <typename Scalar>
  TBFPtr<Scalar> TBF<Scalar>::bf(VarFactoryPtr &vf)
  {
    return Teuchos::rcp( new TBF<Scalar>(vf) );
  }
  
  template <typename Scalar>
  TBF<Scalar>::TBF(bool isLegacySubclass)
  {
    if (!isLegacySubclass)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This constructor is for legacy subclasses only!  Call a VarFactory version instead");
    }
//    _useQRSolveForOptimalTestFunctions = false;
//    _useSPDSolveForOptimalTestFunctions = true;
//    _useIterativeRefinementsWithSPDSolve = false;
//    _warnAboutZeroRowsAndColumns = true;
    
    _isLegacySubclass = true;
  }
  
  template <typename Scalar>
  TBF<Scalar>::TBF( VarFactoryPtr varFactory )   // copies (note that external changes in VarFactory won't be registered by TBF)
  {
    _varFactory = varFactory;
    // set super's ID containers:
    _trialIDs = _varFactory->trialIDs();
    _testIDs = _varFactory->testIDs();
    _isLegacySubclass = false;
    
//    _useQRSolveForOptimalTestFunctions = true;
//    _useSPDSolveForOptimalTestFunctions = false;
//    _useIterativeRefinementsWithSPDSolve = false;
//    _warnAboutZeroRowsAndColumns = true;
  }
  
  template <typename Scalar>
  TBF<Scalar>::TBF( VarFactoryPtr varFactory, VarFactory::BubnovChoice choice )
  {
    _varFactory = varFactory->getBubnovFactory(choice);
    _trialIDs = _varFactory->trialIDs();
    _testIDs = _varFactory->testIDs();
    _isLegacySubclass = false;
    
//    _useQRSolveForOptimalTestFunctions = true;
//    _useSPDSolveForOptimalTestFunctions = false;
//    _useIterativeRefinementsWithSPDSolve = false;
//    _warnAboutZeroRowsAndColumns = true;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::addTerm( TLinearTermPtr<Scalar> trialTerm, TLinearTermPtr<Scalar> testTerm )
  {
    _terms.push_back( make_pair( trialTerm, testTerm ) );
  }
  
  template <typename Scalar>
  void TBF<Scalar>::addTerm( VarPtr trialVar, TLinearTermPtr<Scalar> testTerm )
  {
    addTerm( Teuchos::rcp( new LinearTerm(trialVar) ), testTerm );
  }
  
  template <typename Scalar>
  void TBF<Scalar>::addTerm( VarPtr trialVar, VarPtr testVar )
  {
    addTerm( Teuchos::rcp( new LinearTerm(trialVar) ), Teuchos::rcp( new LinearTerm(testVar) ) );
  }
  
  template <typename Scalar>
  void TBF<Scalar>::addTerm( TLinearTermPtr<Scalar> trialTerm, VarPtr testVar)
  {
    addTerm( trialTerm, Teuchos::rcp( new LinearTerm(testVar) ) );
  }
  
  template <typename Scalar>
  void TBF<Scalar>::applyBilinearFormData(FieldContainer<Scalar> &trialValues, FieldContainer<Scalar> &testValues,
                                          int trialID, int testID, int operatorIndex,
                                          const FieldContainer<double> &points)
  {
    applyBilinearFormData(trialID,testID,trialValues,testValues,points);
  }
  
  template <typename Scalar>
  void TBF<Scalar>::applyBilinearFormData(FieldContainer<Scalar> &trialValues, FieldContainer<Scalar> &testValues,
                                          int trialID, int testID, int operatorIndex,
                                          Teuchos::RCP<BasisCache> basisCache)
  {
    applyBilinearFormData(trialValues, testValues, trialID, testID, operatorIndex, basisCache->getPhysicalCubaturePoints());
  }
  
  template <typename Scalar>
  bool TBF<Scalar>::checkSymmetry(FieldContainer<Scalar> &innerProductMatrix)
  {
    double tol = 1e-10;
    int numCells = innerProductMatrix.dimension(0);
    int numRows = innerProductMatrix.dimension(1);
    if (numRows != innerProductMatrix.dimension(2))
    {
      // non-square: obviously not symmetric!
      return false;
    }
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int i=0; i<numRows; i++)
      {
        for (int j=0; j<i; j++)
        {
          double diff = abs( innerProductMatrix(cellIndex,i,j) - innerProductMatrix(cellIndex,j,i) );
          if (diff > tol)
          {
            return false;
          }
        }
      }
    }
    return true;
  }
  
  // BilinearForm implementation:
  template <typename Scalar>
  const string & TBF<Scalar>::testName(int testID)
  {
    return _varFactory->test(testID)->name();
  }
  template <typename Scalar>
  const string & TBF<Scalar>::trialName(int trialID)
  {
    return _varFactory->trial(trialID)->name();
  }
  
  template <typename Scalar>
  Camellia::EFunctionSpace TBF<Scalar>::functionSpaceForTest(int testID)
  {
    return efsForSpace(_varFactory->test(testID)->space());
  }
  
  template <typename Scalar>
  Camellia::EFunctionSpace TBF<Scalar>::functionSpaceForTrial(int trialID)
  {
    return efsForSpace(_varFactory->trial(trialID)->space());
  }
  
  template <typename Scalar>
  bool TBF<Scalar>::isFluxOrTrace(int trialID)
  {
    VarPtr trialVar = _varFactory->trial(trialID);
    if (trialVar.get() == NULL)   // if unknown trial ID, then it's not a flux or a trace!
    {
      return false;
    }
    VarType varType = trialVar->varType();
    return (varType == FLUX) || (varType == TRACE);
  }
  
  template <typename Scalar>
  string TBF<Scalar>::displayString()
  {
    ostringstream bfStream;
    bool first = true;
    for (typename vector< TBilinearTerm<Scalar> >:: iterator btIt = _terms.begin();
         btIt != _terms.end(); btIt++)
    {
      if (! first )
      {
        bfStream << " + ";
      }
      TBilinearTerm<Scalar> bt = *btIt;
      TLinearTermPtr<Scalar> trialTerm = btIt->first;
      TLinearTermPtr<Scalar> testTerm = btIt->second;
      bfStream << "( " << trialTerm->displayString() << ", " << testTerm->displayString() << ")";
      first = false;
    }
    return bfStream.str();
  }
  
  template <typename Scalar>
  vector<VarPtr> TBF<Scalar>::missingTestVars()
  {
    vector<VarPtr> missingTestVars;
    
    set<int> thisTestIDs;
    for (auto term : _terms)
    {
      LinearTermPtr testTerm = term.second;
      set<int> termIDs = testTerm->varIDs();
      thisTestIDs.insert(termIDs.begin(),termIDs.end());
    }
    
    map< int, VarPtr > testVars = _varFactory->testVars();
    for (auto testVarEntry : testVars)
    {
      if (thisTestIDs.find(testVarEntry.first) == thisTestIDs.end())
      {
        missingTestVars.push_back(testVarEntry.second);
      }
    }
    
    return missingTestVars;
  }
  
  template <typename Scalar>
  vector<VarPtr> TBF<Scalar>::missingTrialVars()
  {
    vector<VarPtr> missingTrialVars;
    
    set<int> thisTrialIDs;
    for (auto term : _terms)
    {
      LinearTermPtr trialTerm = term.first;
      set<int> termIDs = trialTerm->varIDs();
      thisTrialIDs.insert(termIDs.begin(),termIDs.end());
    }
    
    map< int, VarPtr > trialVars = _varFactory->trialVars();
    for (auto trialVarEntry : trialVars)
    {
      if (thisTrialIDs.find(trialVarEntry.first) == thisTrialIDs.end())
      {
        missingTrialVars.push_back(trialVarEntry.second);
      }
    }
    
    return missingTrialVars;
  }
  
  // ! returns the number of potential nonzeros for the given trial ordering and test ordering
  template <typename Scalar>
  int TBF<Scalar>::nonZeroEntryCount(DofOrderingPtr trialOrdering, DofOrderingPtr testOrdering)
  {
    int nonZeros = 0;
    
    set<pair<int,int>> trialTestInteractions;
    
    for (TBilinearTerm<Scalar> bt : _terms)
    {
      TLinearTermPtr<Scalar> trialTerm = bt.first;
      TLinearTermPtr<Scalar> testTerm = bt.second;
      
      set<int> trialIDs = trialTerm->varIDs();
      set<int> testIDs = testTerm->varIDs();
      for (int trialID : trialIDs)
      {
        for (int testID : testIDs)
        {
          trialTestInteractions.insert({trialID,testID});
        }
      }
    }
    
    for (pair<int,int> trialTestPair : trialTestInteractions)
    {
      int trialID = trialTestPair.first, testID = trialTestPair.second;
      int testCardinality = testOrdering->getBasis(testID)->getCardinality();
      vector<int> sidesForTrial = trialOrdering->getSidesForVarID(trialID);
      
      for (int trialSide : sidesForTrial)
      {
        int trialCardinality = trialOrdering->getBasisCardinality(trialID, trialSide);
        // if we get here, there is some (potential) interaction between test and trial on this side
        // at most, this will mean trialCardinality * testCardinality nonzeros
        nonZeros += trialCardinality * testCardinality;
      }
    }
    return nonZeros;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::printTrialTestInteractions()
  {
    if (!_isLegacySubclass)
    {
      cout << displayString() << endl;
    }
    else
    {
      for (vector<int>::iterator testIt = _testIDs.begin(); testIt != _testIDs.end(); testIt++)
      {
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
        for (vector<int>::iterator trialIt = _trialIDs.begin(); trialIt != _trialIDs.end(); trialIt++)
        {
          int trialID = *trialIt;
          vector<Camellia::EOperator> trialOperators, testOperators;
          trialTestOperators(trialID, testID, trialOperators, testOperators);
          vector<Camellia::EOperator>::iterator trialOpIt, testOpIt;
          testOpIt = testOperators.begin();
          int operatorIndex = 0;
          for (trialOpIt = trialOperators.begin(); trialOpIt != trialOperators.end(); trialOpIt++)
          {
            Camellia::EOperator opTrial = *trialOpIt;
            Camellia::EOperator opTest = *testOpIt;
            int trialRank = operatorRank(opTrial, functionSpaceForTrial(trialID));
            int testRank = operatorRank(opTest, functionSpaceForTest(testID));
            trialValue = ( trialRank == 0 ) ? trialValueScalar : trialValueVector;
            testValue = (testRank == 0) ? testValueScalar : testValueVector;
            
            trialValue[0] = 1.0;
            testValue[0] = 1.0;
            FieldContainer<double> testWeight(1), trialWeight(1); // for storing values that come back from applyBilinearForm
            applyBilinearFormData(trialValue,testValue,trialID,testID,operatorIndex,point);
            if ((trialRank==1) && (trialValue.rank() == 3))   // vector that became a scalar (a dot product)
            {
              trialWeight.resize(spaceDim);
              trialWeight[0] = trialValue[0];
              for (int dim=1; dim<spaceDim; dim++)
              {
                trialValue = trialValueVector;
                trialValue.initialize(0.0);
                testValue = (testRank == 0) ? testValueScalar : testValueVector;
                trialValue[dim] = 1.0;
                applyBilinearFormData(trialValue,testValue,trialID,testID,operatorIndex,point);
                trialWeight[dim] = trialValue[0];
              }
            }
            else
            {
              trialWeight[0] = trialValue[0];
            }
            // same thing, but now for testWeight
            if ((testRank==1) && (testValue.rank() == 3))   // vector that became a scalar (a dot product)
            {
              testWeight.resize(spaceDim);
              testWeight[0] = trialValue[0];
              for (int dim=1; dim<spaceDim; dim++)
              {
                testValue = testValueVector;
                testValue.initialize(0.0);
                trialValue = (trialRank == 0) ? trialValueScalar : trialValueVector;
                testValue[dim] = 1.0;
                applyBilinearFormData(trialValue,testValue,trialID,testID,operatorIndex,point);
                testWeight[dim] = testValue[0];
              }
            }
            else
            {
              testWeight[0] = testValue[0];
            }
            if ((testWeight.size() == 2) && (trialWeight.size() == 2))   // both vector values (unsupported)
            {
              TEUCHOS_TEST_FOR_EXCEPTION( true, std::invalid_argument, "unsupported form." );
            }
            else
            {
              // scalar & vector: combine into one, in testWeight
              if ( (trialWeight.size() + testWeight.size()) == 3)
              {
                FieldContainer<double> smaller = (trialWeight.size()==1) ? trialWeight : testWeight;
                FieldContainer<double> bigger =  (trialWeight.size()==2) ? trialWeight : testWeight;
                testWeight.resize(spaceDim);
                for (int dim=0; dim<spaceDim; dim++)
                {
                  testWeight[dim] = smaller[0] * bigger[dim];
                }
              }
              else     // both scalars: combine into one, in testWeight
              {
                testWeight[0] *= trialWeight[0];
              }
            }
            if (testWeight.size() == 1)   // scalar weight
            {
              if ( testWeight[0] == -1.0 )
              {
                cout << " - ";
              }
              else
              {
                if (testWeight[0] == 1.0)
                {
                  if (! first) cout << " + ";
                }
                else
                {
                  if (testWeight[0] < 0.0)
                  {
                    cout << testWeight[0] << " ";
                  }
                  else
                  {
                    cout << " + " << testWeight[0] << " ";
                  }
                }
              }
              if (! isFluxOrTrace(trialID) )
              {
                cout << "\\int_{K} " ;
              }
              else
              {
                cout << "\\int_{\\partial K} " ;
              }
              cout << operatorName(opTrial) << trialName(trialID) << " ";
            }
            else     //
            {
              if (! first) cout << " + ";
              if (! isFluxOrTrace(trialID) )
              {
                cout << "\\int_{K} " ;
              }
              else
              {
                cout << "\\int_{\\partial K} " ;
              }
              if (opTrial != OP_TIMES_NORMAL)
              {
                cout << " \\begin{bmatrix}";
                for (int dim=0; dim<spaceDim; dim++)
                {
                  if (testWeight[dim] != 1.0)
                  {
                    cout << testWeight[0];
                  }
                  if (dim != spaceDim-1)
                  {
                    cout << " \\\\ ";
                  }
                }
                cout << "\\end{bmatrix} ";
                cout << trialName(trialID);
                cout << " \\cdot ";
              }
              else if (opTrial == OP_TIMES_NORMAL)
              {
                if (testWeight.size() == 2)
                {
                  cout << " {";
                  if (testWeight[0] != 1.0)
                  {
                    cout << testWeight[0];
                  }
                  cout << " n_1 " << " \\choose ";
                  if (testWeight[1] != 1.0)
                  {
                    cout << testWeight[1];
                  }
                  cout << " n_2 " << "} " << trialName(trialID) << " \\cdot ";
                }
                else
                {
                  if (testWeight[0] != 1.0)
                  {
                    cout << testWeight[0] << " " << trialName(trialID) << operatorName(opTrial);
                  }
                  else
                  {
                    cout << trialName(trialID) << operatorName(opTrial);
                  }
                }
              }
            }
            if ((opTest == OP_CROSS_NORMAL) || (opTest == OP_DOT_NORMAL))
            {
              // reverse the order:
              cout << testName(testID) << operatorName(opTest);
            }
            else
            {
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
  }
  
  template <typename Scalar>
  void TBF<Scalar>::stiffnessMatrix(FieldContainer<Scalar> &stiffness, Teuchos::RCP<ElementType> elemType,
                                    FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache)
  {
    if (!_isLegacySubclass)
    {
      stiffnessMatrix(stiffness, elemType, cellSideParities, basisCache, true); // default to checking
    }
    else
    {
      // call legacy version:
      DofOrderingPtr testOrdering  = elemType->testOrderPtr;
      DofOrderingPtr trialOrdering = elemType->trialOrderPtr;
      stiffnessMatrix(stiffness,trialOrdering,testOrdering,cellSideParities,basisCache);
    }
  }
  
  // can override check for zero cols (i.e. in hessian matrix)
  template <typename Scalar>
  void TBF<Scalar>::stiffnessMatrix(FieldContainer<Scalar> &stiffness, Teuchos::RCP<ElementType> elemType,
                                    FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache,
                                    bool checkForZeroCols)
  {
    // stiffness is sized as (C, FTest, FTrial)
    stiffness.initialize(0.0);
    basisCache->setCellSideParities(cellSideParities);
    
    for (typename vector< TBilinearTerm<Scalar> >:: iterator btIt = _terms.begin();
         btIt != _terms.end(); btIt++)
    {
      TBilinearTerm<Scalar> bt = *btIt;
      TLinearTermPtr<Scalar> trialTerm = btIt->first;
      TLinearTermPtr<Scalar> testTerm = btIt->second;
      trialTerm->integrate(stiffness, elemType->trialOrderPtr,
                           testTerm,  elemType->testOrderPtr, basisCache);
    }
    if (checkForZeroCols)
    {
      bool checkRows = false; // zero rows just mean a test basis function won't get used, which is fine
      bool checkCols = true; // zero columns mean that a trial basis function doesn't enter the computation, which is bad
      if (! BilinearFormUtility<Scalar>::checkForZeroRowsAndColumns("TBF stiffness", stiffness, checkRows, checkCols) )
      {
        cout << "trial ordering:\n" << *(elemType->trialOrderPtr);
        //    cout << "test ordering:\n" << *(elemType->testOrderPtr);
        //    cout << "stiffness:\n" << stiffness;
      }
    }
  }
  
  // Legacy stiffnessMatrix() method:
  template <typename Scalar>
  void TBF<Scalar>::stiffnessMatrix(FieldContainer<Scalar> &stiffness, Teuchos::RCP<DofOrdering> trialOrdering,
                                    Teuchos::RCP<DofOrdering> testOrdering,
                                    FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache)
  {
    
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
    
    CellTopoPtr cellTopo = basisCache->cellTopology();
    unsigned numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    unsigned spaceDim = cellTopo->getDimension();
    
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
    
    unsigned numSides = cellTopo->getSideCount();
    
    // 3. For each (test, trial) combination:
    vector<int> testIDs = this->testIDs();
    vector<int>::iterator testIterator;
    
    vector<int> trialIDs = this->trialIDs();
    vector<int>::iterator trialIterator;
    
    BasisPtr trialBasis, testBasis;
    
    stiffness.initialize(0.0);
    
    for (testIterator = testIDs.begin(); testIterator != testIDs.end(); testIterator++)
    {
      int testID = *testIterator;
      
      for (trialIterator = trialIDs.begin(); trialIterator != trialIDs.end(); trialIterator++)
      {
        int trialID = *trialIterator;
        
        vector<Camellia::EOperator> trialOperators, testOperators;
        this->trialTestOperators(trialID, testID, trialOperators, testOperators);
        vector<Camellia::EOperator>::iterator trialOpIt, testOpIt;
        testOpIt = testOperators.begin();
        TEUCHOS_TEST_FOR_EXCEPTION(trialOperators.size() != testOperators.size(), std::invalid_argument,
                                   "trialOperators and testOperators must be the same length");
        int operatorIndex = -1;
        for (trialOpIt = trialOperators.begin(); trialOpIt != trialOperators.end(); trialOpIt++)
        {
          operatorIndex++;
          Camellia::EOperator trialOperator = *trialOpIt;
          Camellia::EOperator testOperator = *testOpIt;
          
          if (testOperator==OP_TIMES_NORMAL)
          {
            TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"OP_TIMES_NORMAL not supported for tests.  Use for trial only");
          }
          
          Teuchos::RCP < const FieldContainer<Scalar> > testValuesTransformed;
          Teuchos::RCP < const FieldContainer<Scalar> > trialValuesTransformed;
          Teuchos::RCP < const FieldContainer<Scalar> > testValuesTransformedWeighted;
          
          //cout << "trial is " <<  this->trialName(trialID) << "; test is " << this->testName(testID) << endl;
          
          if (! this->isFluxOrTrace(trialID))
          {
            trialBasis = trialOrdering->getBasis(trialID);
            testBasis = testOrdering->getBasis(testID);
            
            FieldContainer<Scalar> miniStiffness( numCells, testBasis->getCardinality(), trialBasis->getCardinality() );
            
            trialValuesTransformed = basisCache->getTransformedValues(trialBasis,trialOperator);
            testValuesTransformedWeighted = basisCache->getTransformedWeightedValues(testBasis,testOperator);
            
            FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePoints();
            FieldContainer<Scalar> materialDataAppliedToTrialValues = *trialValuesTransformed; // copy first
            FieldContainer<Scalar> materialDataAppliedToTestValues = *testValuesTransformedWeighted; // copy first
            this->applyBilinearFormData(materialDataAppliedToTrialValues, materialDataAppliedToTestValues,
                                        trialID,testID,operatorIndex,basisCache);
            
            //integrate:
            FunctionSpaceTools::integrate<Scalar>(miniStiffness,materialDataAppliedToTestValues,materialDataAppliedToTrialValues,COMP_BLAS);
            // place in the appropriate spot in the element-stiffness matrix
            // copy goes from (cell,trial_basis_dof,test_basis_dof) to (cell,element_trial_dof,element_test_dof)
            
            //cout << "miniStiffness for volume:\n" << miniStiffness;
            
            //checkForZeroRowsAndColumns("miniStiffness for pre-stiffness", miniStiffness);
            
            //cout << "trialValuesTransformed for trial " << this->trialName(trialID) << endl << trialValuesTransformed
            //cout << "testValuesTransformed for test " << this->testName(testID) << ": \n" << testValuesTransformed;
            //cout << "weightedMeasure:\n" << weightedMeasure;
            
            // there may be a more efficient way to do this copying:
            // (one strategy would be to reimplement fst::integrate to support offsets, so that no copying needs to be done...)
            for (int i=0; i < testBasis->getCardinality(); i++)
            {
              int testDofIndex = testOrdering->getDofIndex(testID,i);
              for (int j=0; j < trialBasis->getCardinality(); j++)
              {
                int trialDofIndex = trialOrdering->getDofIndex(trialID,j);
                for (unsigned k=0; k < numCells; k++)
                {
                  stiffness(k,testDofIndex,trialDofIndex) += miniStiffness(k,i,j);
                }
              }
            }
          }
          else      // boundary integral
          {
            int trialBasisRank = trialOrdering->getBasisRank(trialID);
            int testBasisRank = testOrdering->getBasisRank(testID);
            
            TEUCHOS_TEST_FOR_EXCEPTION( ( trialBasisRank != 0 ),
                                       std::invalid_argument,
                                       "Boundary trial variable (flux or trace) given with non-scalar basis.  Unsupported.");
            const vector<int>* sidesForTrial = &trialOrdering->getSidesForVarID(trialID);
            
            for (int sideOrdinal : *sidesForTrial)
            {
              trialBasis = trialOrdering->getBasis(trialID,sideOrdinal);
              testBasis = testOrdering->getBasis(testID);
              
              bool isFlux = false; // i.e. the normal is "folded into" the variable definition, so that we must take parity into account
              const set<Camellia::EOperator> normalOperators = Camellia::normalOperators();
              if (   (normalOperators.find(testOperator)  == normalOperators.end() )
                  && (normalOperators.find(trialOperator) == normalOperators.end() ) )
              {
                // normal not yet taken into account -- so it must be "hidden" in the trial variable
                isFlux = true;
              }
              
              FieldContainer<Scalar> miniStiffness( numCells, testBasis->getCardinality(), trialBasis->getCardinality() );
              
              // for trial: the value lives on the side, so we don't use the volume coords either:
              trialValuesTransformed = basisCache->getTransformedValues(trialBasis,trialOperator,sideOrdinal,false);
              // for test: do use the volume coords:
              testValuesTransformed = basisCache->getTransformedValues(testBasis,testOperator,sideOrdinal,true);
              //
              testValuesTransformedWeighted = basisCache->getTransformedWeightedValues(testBasis,testOperator,sideOrdinal,true);
              
              // copy before manipulating trialValues--these are the ones stored in the cache, so we're not allowed to change them!!
              FieldContainer<Scalar> materialDataAppliedToTrialValues = *trialValuesTransformed;
              
              if (isFlux)
              {
                // we need to multiply the trialValues by the parity of the normal, since
                // the trial implicitly contains an outward normal, and we need to adjust for the fact
                // that the neighboring cells have opposite normal
                // trialValues should have dimensions (numCells,numFields,numCubPointsSide)
                int numFields = trialValuesTransformed->dimension(1);
                int numPoints = trialValuesTransformed->dimension(2);
                for (int cellIndex=0; cellIndex<numCells; cellIndex++)
                {
                  double parity = cellSideParities(cellIndex,sideOrdinal);
                  if (parity != 1.0)    // otherwise, we can just leave things be...
                  {
                    for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++)
                    {
                      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
                      {
                        materialDataAppliedToTrialValues(cellIndex,fieldIndex,ptIndex) *= parity;
                      }
                    }
                  }
                }
              }
              
              FieldContainer<double> cubPointsSidePhysical = basisCache->getPhysicalCubaturePointsForSide(sideOrdinal);
              FieldContainer<Scalar> materialDataAppliedToTestValues = *testValuesTransformedWeighted; // copy first
              this->applyBilinearFormData(materialDataAppliedToTrialValues,materialDataAppliedToTestValues,
                                          trialID,testID,operatorIndex,basisCache);
              
              
              //cout << "sideOrdinal: " << sideOrdinal << "; cubPointsSidePhysical" << endl << cubPointsSidePhysical;
              
              //   d. Sum up (integrate) and place in stiffness matrix according to DofOrdering indices
              FunctionSpaceTools::integrate<Scalar>(miniStiffness,materialDataAppliedToTestValues,materialDataAppliedToTrialValues,COMP_BLAS);
              
              //checkForZeroRowsAndColumns("side miniStiffness for pre-stiffness", miniStiffness);
              
              //cout << "miniStiffness for side " << sideOrdinal << "\n:" << miniStiffness;
              // place in the appropriate spot in the element-stiffness matrix
              // copy goes from (cell,trial_basis_dof,test_basis_dof) to (cell,element_trial_dof,element_test_dof)
              for (int i=0; i < testBasis->getCardinality(); i++)
              {
                int testDofIndex = testOrdering->getDofIndex(testID,i);
                for (int j=0; j < trialBasis->getCardinality(); j++)
                {
                  int trialDofIndex = trialOrdering->getDofIndex(trialID,j,sideOrdinal);
                  for (unsigned k=0; k < numCells; k++)
                  {
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
    if (_warnAboutZeroRowsAndColumns)
    {
      bool checkRows = false; // zero rows just mean a test basis function won't get used, which is fine
      bool checkCols = true; // zero columns mean that a trial basis function doesn't enter the computation, which is bad
      if (! BilinearFormUtility<Scalar>::checkForZeroRowsAndColumns("pre-stiffness", stiffness, checkRows, checkCols) )
      {
        cout << "pre-stiffness matrix in which zero columns were found:\n";
        cout << stiffness;
        cout << "trialOrdering: \n" << *trialOrdering;
      }
    }
  }
  
  // No cellSideParities required, no checking of columns, integrates in a bubnov fashion
  template <typename Scalar>
  void TBF<Scalar>::bubnovStiffness(FieldContainer<Scalar> &stiffness, Teuchos::RCP<ElementType> elemType,
                                    FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache)
  {
    // stiffness is sized as (C, FTrial, FTrial)
    stiffness.initialize(0.0);
    basisCache->setCellSideParities(cellSideParities);
    
    for (typename vector< TBilinearTerm<Scalar> >:: iterator btIt = _terms.begin();
         btIt != _terms.end(); btIt++)
    {
      TBilinearTerm<Scalar> bt = *btIt;
      TLinearTermPtr<Scalar> trialTerm = btIt->first;
      TLinearTermPtr<Scalar> testTerm = btIt->second;
      trialTerm->integrate(stiffness, elemType->trialOrderPtr,
                           testTerm,  elemType->trialOrderPtr, basisCache);
    }
    
  }
  
  template <typename Scalar>
  TIPPtr<Scalar> TBF<Scalar>::graphNorm(double weightForL2TestTerms)
  {
    map<int, double> varWeights;
    return graphNorm(varWeights, weightForL2TestTerms);
  }
  
  template <typename Scalar>
  TIPPtr<Scalar> TBF<Scalar>::graphNorm(const map<int, double> &trialVarWeights, double weightForL2TestTerms)
  {
    vector<int> testVarIDs = _varFactory->testIDs();
    map<int,double> testL2Weights;
    for (int testVarID : testVarIDs)
    {
      testL2Weights[testVarID] = weightForL2TestTerms;
    }
    return this->graphNorm(trialVarWeights,testL2Weights);
  }
  
  template <typename Scalar>
  TIPPtr<Scalar> TBF<Scalar>::graphNorm(const map<int, double> &trialVarWeights, const map<int, double> &testVarL2TermWeights)
  {
    map<int, TLinearTermPtr<Scalar>> testTermsForVarID;
    vector<double> e1(3), e2(3), e3(3); // unit vectors
    e1[0] = 1.0;
    e2[1] = 1.0;
    e3[2] = 1.0;
    TFunctionPtr<double> e1Fxn = TFunction<double>::constant(e1);
    TFunctionPtr<double> e2Fxn = TFunction<double>::constant(e2);
    TFunctionPtr<double> e3Fxn = TFunction<double>::constant(e3);
    for (typename vector< TBilinearTerm<Scalar> >:: iterator btIt = _terms.begin();
         btIt != _terms.end(); btIt++)
    {
      TBilinearTerm<Scalar> bt = *btIt;
      TLinearTermPtr<Scalar> trialTerm = btIt->first;
      TLinearTermPtr<Scalar> testTerm = btIt->second;
      vector< TLinearSummand<Scalar> > summands = trialTerm->summands();
      for (typename vector< TLinearSummand<Scalar> >::iterator lsIt = summands.begin(); lsIt != summands.end(); lsIt++)
      {
        VarPtr trialVar = lsIt->second;
        if (trialVar->varType() == FIELD)
        {
          TFunctionPtr<Scalar> f = lsIt->first;
          if (trialVar->op() == OP_X)
          {
            f = e1Fxn * f;
          }
          else if (trialVar->op() == OP_Y)
          {
            f = e2Fxn * f;
          }
          else if (trialVar->op() == OP_Z)
          {
            f = e3Fxn * f;
          }
          else if (trialVar->op() != OP_VALUE)
          {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TBF<Scalar>::graphNorm() doesn't support non-value ops on field variables");
          }
          if (testTermsForVarID.find(trialVar->ID()) == testTermsForVarID.end())
          {
            testTermsForVarID[trialVar->ID()] = Teuchos::rcp( new LinearTerm );
          }
          testTermsForVarID[trialVar->ID()]->addTerm( f * testTerm );
        }
      }
    }
    TIPPtr<Scalar> ip = Teuchos::rcp( new IP );
    for (typename map<int, TLinearTermPtr<Scalar>>::iterator testTermIt = testTermsForVarID.begin();
         testTermIt != testTermsForVarID.end(); testTermIt++ )
    {
      double weight = 1.0;
      int varID = testTermIt->first;
      if (trialVarWeights.find(varID) != trialVarWeights.end())
      {
        double trialWeight = trialVarWeights.find(varID)->second;
        if (trialWeight <= 0)
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "variable weights must be positive.");
        }
        weight = 1.0 / sqrt(trialWeight);
      }
      ip->addTerm( TFunction<double>::constant(weight) * testTermIt->second );
    }
    // L^2 terms:
    map< int, VarPtr > testVars = _varFactory->testVars();
    for ( map< int, VarPtr >::iterator testVarIt = testVars.begin(); testVarIt != testVars.end(); testVarIt++)
    {
      double testL2Weight = testVarL2TermWeights.find(testVarIt->first)->second;
      ip->addTerm( sqrt(testL2Weight) * testVarIt->second );
    }
    
    return ip;
  }
  
  template <typename Scalar>
  TIPPtr<Scalar> TBF<Scalar>::l2Norm()
  {
    // L2 norm on test space:
    TIPPtr<Scalar> ip = Teuchos::rcp( new IP );
    map< int, VarPtr > testVars = _varFactory->testVars();
    for ( map< int, VarPtr >::iterator testVarIt = testVars.begin(); testVarIt != testVars.end(); testVarIt++)
    {
      ip->addTerm( testVarIt->second );
    }
    return ip;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::localStiffnessMatrixAndRHS(FieldContainer<Scalar> &localStiffness, FieldContainer<Scalar> &rhsVector,
                                               TIPPtr<Scalar> ip, BasisCachePtr ipBasisCache, TRHSPtr<Scalar> rhs, BasisCachePtr basisCache)
  {
    double testMatrixAssemblyTime = 0, localStiffnessDeterminationTime = 0;
    double rhsDeterminationTime = 0;
    
#ifdef HAVE_MPI
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
    //cout << "rank: " << rank << " of " << numProcs << endl;
#else
    Epetra_SerialComm Comm;
#endif
    
    Epetra_Time timer(Comm);
    
    // localStiffness should have dim. (numCells, numTrialFields, numTrialFields)
    MeshPtr mesh = basisCache->mesh();
    if (mesh.get() == NULL)
    {
      cout << "localStiffnessMatrix requires BasisCache to have mesh set.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffnessMatrix requires BasisCache to have mesh set.");
    }
    const vector<GlobalIndexType>* cellIDs = &basisCache->cellIDs();
    int numCells = cellIDs->size();
    if (numCells != localStiffness.dimension(0))
    {
      cout << "localStiffnessMatrix requires basisCache->cellIDs() to have the same # of cells as the first dimension of localStiffness\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffnessMatrix requires basisCache->cellIDs() to have the same # of cells as the first dimension of localStiffness");
    }
    
    ElementTypePtr elemType = mesh->getElementType((*cellIDs)[0]); // we assume all cells provided are of the same type
    DofOrderingPtr trialOrder = elemType->trialOrderPtr;
    DofOrderingPtr testOrder = elemType->testOrderPtr;
    int numTestDofs = testOrder->totalDofs();
    int numTrialDofs = trialOrder->totalDofs();
    if ((numTrialDofs != localStiffness.dimension(1)) || (numTrialDofs != localStiffness.dimension(2)))
    {
      cout << "localStiffness should have dimensions (C,numTrialFields,numTrialFields).\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffness should have dimensions (C,numTrialFields,numTrialFields).");
    }
    
    bool printTimings = false;
    
    if (printTimings)
    {
      cout << "numCells: " << numCells << endl;
      cout << "numTestDofs: " << numTestDofs << endl;
      cout << "numTrialDofs: " << numTrialDofs << endl;
    }
    
    timer.ResetStartTime();
    FieldContainer<double> cellSideParities = basisCache->getCellSideParities();

    if (ip == Teuchos::null)
    {
      // can we interpret as a Bubnov-Galerkin setting?
      TEUCHOS_TEST_FOR_EXCEPTION(numTestDofs != numTrialDofs, std::invalid_argument, "BF: ip is null, but the number of test dofs is different from the number of trial dofs (can't do Bubnov-Galerkin).");
      this->stiffnessMatrix(localStiffness, elemType, cellSideParities, basisCache);
      localStiffnessDeterminationTime += timer.ElapsedTime();

      timer.ResetStartTime();
      rhs->integrateAgainstStandardBasis(rhsVector, testOrder, basisCache);
      rhsDeterminationTime += timer.ElapsedTime();
    }
    else
    {
      //      cout << "ipMatrix:\n" << ipMatrix;
      
      timer.ResetStartTime();
      FieldContainer<Scalar> optTestCoeffs(numCells,numTrialDofs,numTestDofs);
      
      int optSuccess = this->optimalTestWeightsAndStiffness(optTestCoeffs, localStiffness, elemType,
                                                            cellSideParities, basisCache, ip, ipBasisCache);

      localStiffnessDeterminationTime += timer.ElapsedTime();
      //      cout << "optTestCoeffs:\n" << optTestCoeffs;
      
      if ( optSuccess != 0 )
      {
        cout << "**** WARNING: in BilinearForm::localStiffnessMatrixAndRHS(), optimal test function computation failed with error code " << optSuccess << ". ****\n";
      }
      
      timer.ResetStartTime();
      rhs->integrateAgainstOptimalTests(rhsVector, optTestCoeffs, testOrder, basisCache);
      rhsDeterminationTime += timer.ElapsedTime();
    }
    
    if (printTimings)
    {
      cout << "testMatrixAssemblyTime: " << testMatrixAssemblyTime << " seconds.\n";
      cout << "localStiffnessDeterminationTime: " << localStiffnessDeterminationTime << " seconds.\n";
      cout << "rhsDeterminationTime: " << rhsDeterminationTime << " seconds.\n";
    }
  }
  
  template <typename Scalar>
  TIPPtr<Scalar> TBF<Scalar>::naiveNorm(int spaceDim)
  {
    TIPPtr<Scalar> ip = Teuchos::rcp( new IP );
    map< int, VarPtr > testVars = _varFactory->testVars();
    for ( map< int, VarPtr >::iterator testVarIt = testVars.begin(); testVarIt != testVars.end(); testVarIt++)
    {
      VarPtr var = testVarIt->second;
      ip->addTerm( var );
      // HGRAD, HCURL, HDIV, L2, CONSTANT_SCALAR, VECTOR_HGRAD, VECTOR_L2
      if ( (var->space() == HGRAD) || (var->space() == VECTOR_HGRAD) )
      {
        ip->addTerm( var->grad() );
      }
      else if ( (var->space() == L2) || (var->space() == VECTOR_L2) )
      {
        // do nothing (we already added the L2 term
      }
      else if (var->space() == HCURL)
      {
        ip->addTerm( var->curl(spaceDim) );
      }
      else if (var->space() == HDIV)
      {
        ip->addTerm( var->div() );
      }
    }
    return ip;
  }
  
  template <typename Scalar>
  int TBF<Scalar>::optimalTestWeightsAndStiffness(FieldContainer<Scalar> &optimalTestWeights,
                                                  FieldContainer<Scalar> &stiffnessMatrix,
                                                  ElementTypePtr elemType,
                                                  FieldContainer<double> &cellSideParities,
                                                  BasisCachePtr stiffnessBasisCache,
                                                  IPPtr ip, BasisCachePtr ipBasisCache)
  {
    DofOrderingPtr trialOrdering = elemType->trialOrderPtr;
    DofOrderingPtr testOrdering = elemType->testOrderPtr;
    
    // all arguments are as in computeStiffnessMatrix, except:
    // optimalTestWeights, which has dimensions (numCells, numTrialDofs, numTestDofs)
    // innerProduct: the inner product which defines the sense in which these test functions are optimal
    int numCells = stiffnessBasisCache->getPhysicalCubaturePoints().dimension(0);
    int numTestDofs = testOrdering->totalDofs();
    int numTrialDofs = trialOrdering->totalDofs();
    
#ifdef HAVE_MPI
    Epetra_MpiComm Comm(MPI_COMM_WORLD);
    //cout << "rank: " << rank << " of " << numProcs << endl;
#else
    Epetra_SerialComm Comm;
#endif
    
    Epetra_Time timer(Comm);
    
    double timeG, timeB, timeT; // time to compute Gram matrix, the right-hand side B, and time to solve GT = B
    
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
    
    // to be memory-efficient, we'll compute directly into optimalTestWeights, but the most natural way to do this
    // is to compute the transpose.
    
    FieldContainer<Scalar> rectangularStiffnessMatrix(numCells,numTestDofs,numTrialDofs);
    //  FieldContainer<double> stiffnessMatrixT(numCells,numTrialDofs,numTestDofs);
    
    timer.ResetStartTime();
    // RHS:
    this->stiffnessMatrix(rectangularStiffnessMatrix, elemType, cellSideParities, stiffnessBasisCache);
    timeB = timer.ElapsedTime();
    
    int solvedAll = 0;
    
    Teuchos::Array<int> cellOptimalWeightsTDim(2); // data stored in transposed order relative to what we'll eventually want
    cellOptimalWeightsTDim[0] = numTestDofs;
    cellOptimalWeightsTDim[1] = numTrialDofs;
    
    Teuchos::Array<int> localIPDim(2);
    localIPDim[0] = numTestDofs;
    localIPDim[1] = numTestDofs;
    Teuchos::Array<int> localRectangularStiffnessDim(2);
    localRectangularStiffnessDim[0] = rectangularStiffnessMatrix.dimension(1);
    localRectangularStiffnessDim[1] = rectangularStiffnessMatrix.dimension(2);
    Teuchos::Array<int> localStiffnessDim(2);
    localStiffnessDim[0] = stiffnessMatrix.dimension(1);
    localStiffnessDim[1] = stiffnessMatrix.dimension(2);
    
    FieldContainer<Scalar> ipMatrix(numCells,numTestDofs,numTestDofs);
    DofOrderingPtr testOrder = elemType->testOrderPtr;
    timer.ResetStartTime();
    ip->computeInnerProductMatrix(ipMatrix, testOrder, ipBasisCache);
    timeG = timer.ElapsedTime();
    
    timer.ResetStartTime();
    for (int cellIndex=0; cellIndex < numCells; cellIndex++)
    {
      int result = 0;
      FieldContainer<Scalar> cellIPMatrix(localIPDim, &ipMatrix(cellIndex,0,0));
      FieldContainer<Scalar> cellRectangularStiffness(localRectangularStiffnessDim, &rectangularStiffnessMatrix(cellIndex,0,0));
      FieldContainer<Scalar> cellStiffness(localStiffnessDim, &stiffnessMatrix(cellIndex,0,0));
      FieldContainer<Scalar> cellOptimalWeightsT(cellOptimalWeightsTDim, &optimalTestWeights(cellIndex,0,0));
      if (_useQRSolveForOptimalTestFunctions)
      {
        bool useIPTranspose = true; // true value may allow less memory to be used during solveSystemUsingQR() (maybe only if we can get overwriting to work, below)
        bool allowIPOverwrite = true; // assert that we won't be using cellIPMatrix again
        result = SerialDenseWrapper::solveSystemUsingQR(cellOptimalWeightsT, cellIPMatrix, cellRectangularStiffness, useIPTranspose, allowIPOverwrite);
        //        result = SerialDenseWrapper::solveSystemUsingQR(optimalWeightsT, cellIPMatrix, cellRectangularStiffness);
      }
      else if (_useSPDSolveForOptimalTestFunctions)
      {
        bool allowIPOverwrite = false; // assert that we won't be using cellIPMatrix again
        result = SerialDenseWrapper::solveSPDSystemMultipleRHS(cellOptimalWeightsT, cellIPMatrix, cellRectangularStiffness, allowIPOverwrite);
        if (result != 0)
        {
          // may be that we're not SPD numerically
          cout << "During optimal test weight solution, SPD solve returned error " << result << ".  Solving with LU factorization instead of SPD solve.\n";
          result = SerialDenseWrapper::solveSystemMultipleRHS(cellOptimalWeightsT, cellIPMatrix, cellRectangularStiffness);
        }
      }
      else
      {
        SerialDenseWrapper::solveSystemMultipleRHS(cellOptimalWeightsT, cellIPMatrix, cellRectangularStiffness);
      }
      
      // multiply to determine stiffness matrix.
      SerialDenseWrapper::multiply(cellStiffness, cellOptimalWeightsT, cellRectangularStiffness, 'T', 'N'); // transpose A; don't transpose B
      
      // transpose the optimal test weights -- since this is a view into optimalTestWeights, this reorders (part of) that matrix according to contract with caller
      SerialDenseWrapper::transposeMatrix(cellOptimalWeightsT);
      
      if (result != 0)
      {
        solvedAll = result;
      }
    }
    timeT = timer.ElapsedTime();
   
    bool printTimings = false;
    if (printTimings)
    {
      cout << "BF timings: computed G in " << timeG << " seconds, B in " << timeB << "; solve for T in " << timeT << endl;
    }
    return solvedAll;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::setUseSPDSolveForOptimalTestFunctions(bool value)
  {
    _useSPDSolveForOptimalTestFunctions = value;
    if (_useSPDSolveForOptimalTestFunctions)
      _useQRSolveForOptimalTestFunctions = false;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::setUseIterativeRefinementsWithSPDSolve(bool value)
  {
    _useIterativeRefinementsWithSPDSolve = value;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::setUseExtendedPrecisionSolveForOptimalTestFunctions(bool value)
  {
    cout << "WARNING: BilinearForm no longer supports extended precision solve for optimal test functions.  Ignoring argument to setUseExtendedPrecisionSolveForOptimalTestFunctions().\n";
  }
  
  template <typename Scalar>
  void TBF<Scalar>::setWarnAboutZeroRowsAndColumns(bool value)
  {
    _warnAboutZeroRowsAndColumns = value;
  }
  
  template <typename Scalar>
  TLinearTermPtr<Scalar> TBF<Scalar>::testFunctional(TSolutionPtr<Scalar> trialSolution, bool excludeBoundaryTerms, bool overrideMeshCheck)
  {
    TLinearTermPtr<Scalar> functional = Teuchos::rcp(new LinearTerm());
    for (typename vector< TBilinearTerm<Scalar> >:: iterator btIt = _terms.begin();
         btIt != _terms.end(); btIt++)
    {
      TBilinearTerm<Scalar> bt = *btIt;
      TLinearTermPtr<Scalar> trialTerm = btIt->first;
      TLinearTermPtr<Scalar> testTerm = btIt->second;
      TFunctionPtr<Scalar> trialValue = Teuchos::rcp( new PreviousSolutionFunction<Scalar>(trialSolution, trialTerm) );
      static_cast< PreviousSolutionFunction<Scalar>* >(trialValue.get())->setOverrideMeshCheck(overrideMeshCheck);
      if ( (! excludeBoundaryTerms) || (! trialValue->boundaryValueOnly()) )
      {
        functional = functional + trialValue * testTerm;
      }
    }
    return functional;
  }
  
  template <typename Scalar>
  const vector< int > & TBF<Scalar>::trialIDs()
  {
    return _trialIDs;
  }
  
  template <typename Scalar>
  const vector< int > & TBF<Scalar>::testIDs()
  {
    return _testIDs;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::trialTestOperators(int testID1, int testID2,
                                       vector<Camellia::EOperator> &testOps1,
                                       vector<Camellia::EOperator> &testOps2)
  {
    Camellia::EOperator testOp1, testOp2;
    testOps1.clear();
    testOps2.clear();
    if (trialTestOperator(testID1,testID2,testOp1,testOp2))
    {
      testOps1.push_back(testOp1);
      testOps2.push_back(testOp2);
    }
  }
  
  template <typename Scalar>
  vector<int> TBF<Scalar>::trialVolumeIDs()
  {
    vector<int> ids;
    for (vector<int>::iterator trialIt = _trialIDs.begin(); trialIt != _trialIDs.end(); trialIt++)
    {
      int trialID = *(trialIt);
      if ( ! isFluxOrTrace(trialID) )
      {
        ids.push_back(trialID);
      }
    }
    return ids;
  }
  
  template <typename Scalar>
  vector<int> TBF<Scalar>::trialBoundaryIDs()
  {
    vector<int> ids;
    for (vector<int>::iterator trialIt = _trialIDs.begin(); trialIt != _trialIDs.end(); trialIt++)
    {
      int trialID = *(trialIt);
      if ( isFluxOrTrace(trialID) )
      {
        ids.push_back(trialID);
      }
    }
    return ids;
  }
  
  
  template <typename Scalar>
  VarFactoryPtr TBF<Scalar>::varFactory()
  {
    if (! _isLegacySubclass)
    {
      return _varFactory;
    }
    else
    {
      // this is not meant to cover every possible subclass, but the known legacy subclasses.
      // (just here to allow compatibility with subclasses in DPGTests, e.g.; new implementations should use TBF)
      VarFactoryPtr vf = VarFactory::varFactory();
      vector<int> trialIDs = this->trialIDs();
      for (int trialIndex=0; trialIndex<trialIDs.size(); trialIndex++)
      {
        int trialID = trialIDs[trialIndex];
        string name = this->trialName(trialID);
        VarPtr trialVar;
        if (isFluxOrTrace(trialID))
        {
          bool isFlux = this->functionSpaceForTrial(trialID) == Camellia::FUNCTION_SPACE_HVOL;
          if (isFlux)
          {
            trialVar = vf->fluxVar(name);
          }
          else
          {
            trialVar = vf->traceVar(name);
          }
        }
        else
        {
          trialVar = vf->fieldVar(name);
        }
      }
      
      vector<int> testIDs = this->testIDs();
      for (int testIndex=0; testIndex<testIDs.size(); testIndex++)
      {
        int testID = testIDs[testIndex];
        string name = this->testName(testID);
        VarPtr testVar;
        Camellia::EFunctionSpace fs = this->functionSpaceForTest(testID);
        Space space;
        switch (fs)
        {
          case Camellia::FUNCTION_SPACE_HGRAD:
            space = HGRAD;
            break;
          case Camellia::FUNCTION_SPACE_HCURL:
            space = HCURL;
            break;
          case Camellia::FUNCTION_SPACE_HDIV:
            space = HDIV;
            break;
          case Camellia::FUNCTION_SPACE_HGRAD_DISC:
            space = HGRAD_DISC;
            break;
          case Camellia::FUNCTION_SPACE_HCURL_DISC:
            space = HCURL_DISC;
            break;
          case Camellia::FUNCTION_SPACE_HDIV_DISC:
            space = HDIV_DISC;
            break;
          case Camellia::FUNCTION_SPACE_HVOL:
            space = L2;
            break;
            
          default:
            cout << "BilinearForm::varFactory(): unhandled function space.\n";
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BilinearForm::varFactory(): unhandled function space.");
            break;
        }
        testVar = vf->testVar(name, space);
      }
      return vf;
    }
  }
  template class TBF<double>;
}
