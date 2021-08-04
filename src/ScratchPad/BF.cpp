//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "BF.h"
#include "RieszRep.h"

#include "BilinearFormUtility.h"
#include "Function.h"
#include "PreviousSolutionFunction.h"
#include "LinearTerm.h"
#include "SerialDenseWrapper.h"
#include "TimeLogger.h"
#include "VarFactory.h"

#include "Intrepid_FunctionSpaceTools.hpp"

#include "Teuchos_BLAS.hpp"
#include "Teuchos_LAPACK.hpp"

#include <iostream>
#include <sstream>

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

    TimeLogger::sharedInstance()->createTimeEntry(BF_LOCAL_STIFFNESS_AND_RHS_TIMER_STRING);
    
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
    
    TimeLogger::sharedInstance()->createTimeEntry(BF_LOCAL_STIFFNESS_AND_RHS_TIMER_STRING);
    
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
  }
  
  template <typename Scalar>
  void TBF<Scalar>::addJumpTerm( TLinearTermPtr<Scalar> trialTerm, TLinearTermPtr<Scalar> testTerm )
  {
    _jumpTerms.push_back( make_pair( trialTerm, testTerm ) );
  }
  
  template <typename Scalar>
  void VALIDATE_TRIAL_TEST_PAIR(TLinearTermPtr<Scalar> trialTerm, TLinearTermPtr<Scalar> testTerm)
  {
    if (trialTerm == Teuchos::null)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "trialTerm may not be null!");
    }
    else if (testTerm == Teuchos::null)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "testTerm may not be null!");
    }
    else if (trialTerm->rank() == -1)
    {
      // debugging:
      cout << "trialTerm: " << trialTerm->displayString() << endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "trialTerm may not be empty!");
    }
    else if (testTerm->rank() == -1)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "testTerm may not be empty!");
    }
    if (trialTerm->rank() != testTerm->rank())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "trialTerm->rank() != testTerm->rank()");
    }
  }
  
  template <typename Scalar>
  void TBF<Scalar>::addTerm( TLinearTermPtr<Scalar> trialTerm, TLinearTermPtr<Scalar> testTerm )
  {
    VALIDATE_TRIAL_TEST_PAIR(trialTerm, testTerm);
    _terms.push_back( make_pair( trialTerm, testTerm ) );
  }
  
  template <typename Scalar>
  void TBF<Scalar>::addTerm( VarPtr trialVar, TLinearTermPtr<Scalar> testTerm )
  {
    auto trialTerm = Teuchos::rcp( new LinearTerm(trialVar) );
    VALIDATE_TRIAL_TEST_PAIR(trialTerm, testTerm);
    addTerm( trialTerm, testTerm );
  }
  
  template <typename Scalar>
  void TBF<Scalar>::addTerm( VarPtr trialVar, VarPtr testVar )
  {
    auto trialTerm = Teuchos::rcp( new LinearTerm(trialVar) );
    auto testTerm  = Teuchos::rcp( new LinearTerm(testVar) );
    VALIDATE_TRIAL_TEST_PAIR(trialTerm, testTerm);
    addTerm( trialTerm, testTerm );
  }
  
  template <typename Scalar>
  void TBF<Scalar>::addTerm( TLinearTermPtr<Scalar> trialTerm, VarPtr testVar)
  {
    auto testTerm  = Teuchos::rcp( new LinearTerm(testVar) );
    VALIDATE_TRIAL_TEST_PAIR(trialTerm, testTerm);
    addTerm( trialTerm, testTerm );
  }
  
  template <typename Scalar>
  void TBF<Scalar>::applyBilinearFormData(Camellia::FieldContainer<Scalar> &trialValues, Camellia::FieldContainer<Scalar> &testValues,
                                          int trialID, int testID, int operatorIndex,
                                          const Camellia::FieldContainer<double> &points)
  {
    applyBilinearFormData(trialID,testID,trialValues,testValues,points);
  }
  
  template <typename Scalar>
  void TBF<Scalar>::applyBilinearFormData(Camellia::FieldContainer<Scalar> &trialValues, Camellia::FieldContainer<Scalar> &testValues,
                                          int trialID, int testID, int operatorIndex,
                                          Teuchos::RCP<BasisCache> basisCache)
  {
    applyBilinearFormData(trialValues, testValues, trialID, testID, operatorIndex, basisCache->getPhysicalCubaturePointsFC());
  }
  
  template <typename Scalar>
  bool TBF<Scalar>::checkSymmetry(Camellia::FieldContainer<Scalar> &innerProductMatrix)
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
  
  template <typename Scalar>
  TBFPtr<Scalar> TBF<Scalar>::copy() const
  {
    return Teuchos::rcp( new BF(*this) );
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
    return _varFactory->isFluxOrTrace(trialID);
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
        Camellia::FieldContainer<double> point(1,2); // (0,0)
        Camellia::FieldContainer<double> testValueScalar(1,1,1); // 1 cell, 1 basis function, 1 point...
        Camellia::FieldContainer<double> testValueVector(1,1,1,spaceDim); // 1 cell, 1 basis function, 1 point, spaceDim dimensions...
        Camellia::FieldContainer<double> trialValueScalar(1,1,1); // 1 cell, 1 basis function, 1 point...
        Camellia::FieldContainer<double> trialValueVector(1,1,1,spaceDim); // 1 cell, 1 basis function, 1 point, spaceDim dimensions...
        Camellia::FieldContainer<double> testValue, trialValue;
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
            Camellia::FieldContainer<double> testWeight(1), trialWeight(1); // for storing values that come back from applyBilinearForm
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
                Camellia::FieldContainer<double> smaller = (trialWeight.size()==1) ? trialWeight : testWeight;
                Camellia::FieldContainer<double> bigger =  (trialWeight.size()==2) ? trialWeight : testWeight;
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
  void TBF<Scalar>::stiffnessMatrix(Camellia::FieldContainer<Scalar> &stiffness, Teuchos::RCP<ElementType> elemType,
                                    const Camellia::FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache)
  {
    if (!_isLegacySubclass)
    {
      stiffnessMatrix(stiffness, elemType, cellSideParities, basisCache, false, true); // default to column-major, checking for zeros
    }
    else
    {
      // call legacy version:
      DofOrderingPtr testOrdering  = elemType->testOrderPtr;
      DofOrderingPtr trialOrdering = elemType->trialOrderPtr;
      Camellia::FieldContainer<double> cellSideParitiesNonConst = cellSideParities; // copy for sake of legacy, non-const argument.
      stiffnessMatrix(stiffness,trialOrdering,testOrdering,cellSideParitiesNonConst,basisCache);
    }
  }
  
  // can override check for zero cols (i.e. in hessian matrix)
  template <typename Scalar>
  void TBF<Scalar>::stiffnessMatrix(Camellia::FieldContainer<Scalar> &stiffness, Teuchos::RCP<ElementType> elemType,
                                    const Camellia::FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache,
                                    bool rowMajor, bool checkForZeroCols)
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
      
      Camellia::FieldContainer<double> stiffnessCopyForTermwiseOutput;
      if (_printTermWiseIntegrationOutput)
      {
        // DEBUGGING
        stiffnessCopyForTermwiseOutput = stiffness;
        cout << "Integrating " << trialTerm->displayString() << " against " << testTerm->displayString() << endl;
      }
      if (rowMajor)
      {
        testTerm->integrate(stiffness, elemType->testOrderPtr,
                            trialTerm,  elemType->trialOrderPtr, basisCache);
      }
      else
      {
        trialTerm->integrate(stiffness, elemType->trialOrderPtr,
                             testTerm,  elemType->testOrderPtr, basisCache);
      }
      if (_printTermWiseIntegrationOutput)
      {
        // DEBUGGING
        cout << "This integration added the following to the stiffness matrix:\n";
        for (int cellOrdinal=0; cellOrdinal<stiffness.dimension(0); cellOrdinal++) // cell dimension
        {
          cout << "Cell ordinal " << cellOrdinal << endl;
          for (int i=0; i<stiffness.dimension(1); i++)
          {
            for (int j=0; j<stiffness.dimension(2); j++)
            {
              double diff = stiffness(cellOrdinal,i,j) - stiffnessCopyForTermwiseOutput(cellOrdinal,i,j);
              if (diff > 1e-15)
              {
                cout << i << "\t" << j << "\t" << diff << endl;
              }
              else if (std::isnan(stiffness(cellOrdinal,i,j)) && !std::isnan(stiffnessCopyForTermwiseOutput(cellOrdinal,i,j)))
              {
                cout << i << "\t" << j << "\t" << stiffness(cellOrdinal,i,j) << endl;
              }
            }
          }
        }
      }
    }
    if (checkForZeroCols)
    {
      bool checkRows, checkCols;
      if (rowMajor)
      {
        checkRows = true;  // zero columns mean that a trial basis function doesn't enter the computation, which is bad
        checkCols = false; // zero rows just mean a test basis function won't get used, which is fine
      }
      else
      {
        checkRows = false; // zero rows just mean a test basis function won't get used, which is fine
        checkCols = true; // zero columns mean that a trial basis function doesn't enter the computation, which is bad
      }
      if (! BilinearFormUtility<Scalar>::checkForZeroRowsAndColumns("TBF stiffness", stiffness, checkRows, checkCols) )
      {
        // tell trialOrderPtr about its VarFactory, for richer output
        elemType->trialOrderPtr->setVarFactory(_varFactory, true); // true: is trial ordering
        cout << "trial ordering:\n" << *(elemType->trialOrderPtr);
        elemType->testOrderPtr->setVarFactory(_varFactory, false); // false: is test ordering
        cout << "test ordering:\n" << *(elemType->testOrderPtr);
        //    cout << "test ordering:\n" << *(elemType->testOrderPtr);
        //    cout << "stiffness:\n" << stiffness;
      }
    }
  }
  
  // Legacy stiffnessMatrix() method:
  template <typename Scalar>
  void TBF<Scalar>::stiffnessMatrix(Camellia::FieldContainer<Scalar> &stiffness, Teuchos::RCP<DofOrdering> trialOrdering,
                                    Teuchos::RCP<DofOrdering> testOrdering,
                                    Camellia::FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache)
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
    unsigned numCells = basisCache->getPhysicalCubaturePointsFC().dimension(0);
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
          
          Teuchos::RCP < const Camellia::FieldContainer<Scalar> > testValuesTransformed;
          Teuchos::RCP < const Camellia::FieldContainer<Scalar> > trialValuesTransformed;
          Teuchos::RCP < const Camellia::FieldContainer<Scalar> > testValuesTransformedWeighted;
          
          //cout << "trial is " <<  this->trialName(trialID) << "; test is " << this->testName(testID) << endl;
          
          if (! this->isFluxOrTrace(trialID))
          {
            trialBasis = trialOrdering->getBasis(trialID);
            testBasis = testOrdering->getBasis(testID);
            
            Camellia::FieldContainer<Scalar> miniStiffness( numCells, testBasis->getCardinality(), trialBasis->getCardinality() );
            
            trialValuesTransformed = basisCache->getTransformedValues(trialBasis,trialOperator);
            testValuesTransformedWeighted = basisCache->getTransformedWeightedValues(testBasis,testOperator);
            
            Camellia::FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePointsFC();
            Camellia::FieldContainer<Scalar> materialDataAppliedToTrialValues = *trialValuesTransformed; // copy first
            Camellia::FieldContainer<Scalar> materialDataAppliedToTestValues = *testValuesTransformedWeighted; // copy first
            this->applyBilinearFormData(materialDataAppliedToTrialValues, materialDataAppliedToTestValues,
                                        trialID,testID,operatorIndex,basisCache);
            
            Intrepid::FieldContainer<double> &miniStiffnessIntrepid                    = miniStiffness;
            Intrepid::FieldContainer<double> &materialDataAppliedToTestValuesIntrepid  = materialDataAppliedToTestValues;
            Intrepid::FieldContainer<double> &materialDataAppliedToTrialValuesIntrepid = materialDataAppliedToTrialValues;
  
            // the casts above are important to let FunctionSpaceTools invoke BLAS
            //integrate:
            Intrepid::FunctionSpaceTools::integrate<Scalar>(miniStiffnessIntrepid,
                                                            materialDataAppliedToTestValuesIntrepid,
                                                            materialDataAppliedToTrialValuesIntrepid,
                                                            Intrepid::COMP_BLAS);
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
              
              Camellia::FieldContainer<Scalar> miniStiffness( numCells, testBasis->getCardinality(), trialBasis->getCardinality() );
              
              // for trial: the value lives on the side, so we don't use the volume coords either:
              trialValuesTransformed = basisCache->getTransformedValues(trialBasis,trialOperator,sideOrdinal,false);
              // for test: do use the volume coords:
              testValuesTransformed = basisCache->getTransformedValues(testBasis,testOperator,sideOrdinal,true);
              //
              testValuesTransformedWeighted = basisCache->getTransformedWeightedValues(testBasis,testOperator,sideOrdinal,true);
              
              // copy before manipulating trialValues--these are the ones stored in the cache, so we're not allowed to change them!!
              Camellia::FieldContainer<Scalar> materialDataAppliedToTrialValues = *trialValuesTransformed;
              
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
              
              Camellia::FieldContainer<double> cubPointsSidePhysical = basisCache->getPhysicalCubaturePointsForSide(sideOrdinal);
              Camellia::FieldContainer<Scalar> materialDataAppliedToTestValues = *testValuesTransformedWeighted; // copy first
              this->applyBilinearFormData(materialDataAppliedToTrialValues,materialDataAppliedToTestValues,
                                          trialID,testID,operatorIndex,basisCache);
              
              Intrepid::FieldContainer<double> &miniStiffnessIntrepid                    = miniStiffness;
              Intrepid::FieldContainer<double> &materialDataAppliedToTestValuesIntrepid  = materialDataAppliedToTestValues;
              Intrepid::FieldContainer<double> &materialDataAppliedToTrialValuesIntrepid = materialDataAppliedToTrialValues;
              
              // the casts above are important to let FunctionSpaceTools invoke BLAS
              //   d. Sum up (integrate) and place in stiffness matrix according to DofOrdering indices
              Intrepid::FunctionSpaceTools::integrate<Scalar>(miniStiffnessIntrepid,
                                                              materialDataAppliedToTestValuesIntrepid,
                                                              materialDataAppliedToTrialValuesIntrepid,
                                                              Intrepid::COMP_BLAS);

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
  void TBF<Scalar>::bubnovStiffness(Camellia::FieldContainer<Scalar> &stiffness, Teuchos::RCP<ElementType> elemType,
                                    const Camellia::FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache)
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
  int TBF<Scalar>::factoredCholeskySolve(Camellia::FieldContainer<Scalar> &ipMatrix, Camellia::FieldContainer<Scalar> &stiffnessEnriched,
                                         Camellia::FieldContainer<Scalar> &rhsEnriched, Camellia::FieldContainer<Scalar> &stiffness,
                                         Camellia::FieldContainer<Scalar> &rhs)
  {
    TEUCHOS_FUNC_TIME_MONITOR("BF::factoredCholeskySolve");
    int timerHandle = TimeLogger::sharedInstance()->startTimer(BF_FACTORED_CHOLESKY_SOLVE_TIMER_STRING);
    int N = ipMatrix.dimension(0);
    TEUCHOS_TEST_FOR_EXCEPTION(N != ipMatrix.dimension(1), std::invalid_argument, "ipMatrix must be square");
    int M = stiffnessEnriched.dimension(0);
    TEUCHOS_TEST_FOR_EXCEPTION(N != stiffnessEnriched.dimension(1), std::invalid_argument, "stiffnessEnriched must be have one dimension equal to test ipMatrix dimension");
    
    char UPLO = 'L'; // lower-triangular
    
    int result = 0;
    int INFO;
    
    Teuchos::LAPACK<int, double> lapack;
    Teuchos::BLAS<int, double> blas;
    
    // DPOEQU( N, A, LDA, S, SCOND, AMAX, INFO )
    Camellia::FieldContainer<double> scaleFactors(N);
    double scond, amax;
    lapack.POEQU(N, &ipMatrix[0], N, &scaleFactors[0], &scond, &amax, &INFO);
    
//    cout << "scaleFactors:\n" << scaleFactors;
    
    // do we need to equilibriate?
    // for now, we don't check, but just do the scaling...
    for (int i=0; i<N; i++)
    {
      double scale_i = scaleFactors[i];
      for (int j=0; j<N; j++)
      {
        ipMatrix(i,j) *= scale_i * scaleFactors[j];
      }
    }
    
//    cout << "ipMatrix equilibriated:\n" << ipMatrix;

    bool equilibriated = true;
    
    lapack.POTRF(UPLO, N, &ipMatrix[0], N, &INFO);
    
    if (INFO != 0)
    {
      result = INFO;
      cout << "dpotrf_ result: " << INFO << endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Cholesky factorization failed");
    }
    
    if (equilibriated)
    {
      // unequilibriate in the L factors:
      for (int j=0; j<N; j++)
      {
        for (int i=j; i<N; i++)
        {
          // lower-triangle is stored in (i,j) where i >= j
          ipMatrix(j,i) /= scaleFactors[i]; // Camellia::FieldContainer transposes, effectively
        }
      }
    }
    
//    cout << "ipMatrix, factored (lower-tri):\n" << ipMatrix;
    
    double ALPHA = 1.0;
    blas.TRSM(Teuchos::LEFT_SIDE, Teuchos::LOWER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, N, M, ALPHA, &ipMatrix[0], N,
              &stiffnessEnriched[0], N);
    
//    cout << "stiffnessEnriched, back-subbed:\n" << stiffnessEnriched;
    
    double BETA = 0.0;
    blas.SYRK(Teuchos::LOWER_TRI, Teuchos::TRANS, M, N, ALPHA, &stiffnessEnriched[0], N, BETA, &stiffness[0], M);
    
    // copy lower-triangular part of stiffness to the upper-triangular part (in column-major/Fortran order)
    for (int i=0; i<M; i++)
    {
      for (int j=i+1; j<M; j++)
      {
        // lower-triangle is stored in (i,j) where j >= i
        // column-major: store (i,j) in i*M+j
        // set (j,i) := (i,j)
        stiffness[j*M+i] = stiffness[i*M+j];
      }
    }
    
//    cout << "stiffness matrix:\n" << stiffness;
    
    // need also to take cellRectangularStiffness and do back-substitution with L^T, so that what's left in there is the
    // cellOptimalWeights
    int oneColumn = 1;

    blas.TRSM(Teuchos::LEFT_SIDE, Teuchos::LOWER_TRI, Teuchos::NO_TRANS, Teuchos::NON_UNIT_DIAG, N, oneColumn, ALPHA, &ipMatrix[0], N,
              &rhsEnriched[0], N);
    
    SerialDenseWrapper::multiply(rhs, stiffnessEnriched, rhsEnriched, 'N', 'N');
    TimeLogger::sharedInstance()->stopTimer(timerHandle);
    return result;
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
      TLinearTermPtr<Scalar> testTerm;
      // filter out any boundary-only parts:
      auto testSummands = btIt->second->summands();
      for (auto testSummand : testSummands)
      {
        if (testSummand.first->boundaryValueOnly())
        {
          continue;
        }
        else
        {
          testTerm = testTerm + testSummand.first * testSummand.second;
        }
      }
      if (testTerm == Teuchos::null) continue; // test is boundary-value only: skip
        
      vector< TLinearSummand<Scalar> > summands = trialTerm->summands();
      for (typename vector< TLinearSummand<Scalar> >::iterator lsIt = summands.begin(); lsIt != summands.end(); lsIt++)
      {
        VarPtr trialVar = lsIt->second;
        if ((trialVar->varType() == FIELD) && (lsIt->first->boundaryValueOnly()))
        {
          // for now anyway, we neglect field terms on the boundary.  (These are arising with CDPG.)
          // for now, we print a warning.
          static bool HAVE_WARNED = false;
          if (!HAVE_WARNED)
          {
            std::cout << "NOTE: in automatic determination of the graph norm, skipping field terms that have function weights defined only on the mesh skeleton (e.g. normals).  The usual thing is to rely on the L^2 terms that are always added to handle these, so this choice is probably fine.\n";
            HAVE_WARNED = true;
          }
          continue; // skip this term
        }
            
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
  const vector< TBilinearTerm<Scalar> > & TBF<Scalar>::getJumpTerms() const
  {
    return _jumpTerms;
  }
  
  template <typename Scalar>
  const vector< TBilinearTerm<Scalar> > & TBF<Scalar>::getTerms() const
  {
    return _terms;
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
  void TBF<Scalar>::localStiffnessMatrixAndRHS_DLS(Camellia::FieldContainer<Scalar> &stiffnessEnriched, Camellia::FieldContainer<Scalar> &rhsEnriched,
                                                   TIPPtr<Scalar> ip, BasisCachePtr ipBasisCache, TRHSPtr<Scalar> rhs,
                                                   BasisCachePtr basisCache)
  {
    TEUCHOS_FUNC_TIME_MONITOR("BF::localStiffnessMatrixAndRHS_DLS");
    int timerHandle = TimeLogger::sharedInstance()->startTimer(BF_LOCAL_STIFFNESS_AND_RHS_TIMER_STRING);
    double testMatrixAssemblyTime = 0, localStiffnessDeterminationTime = 0;
    double rhsDeterminationTime = 0;
    
    Epetra_Time timer(*MPIWrapper::CommSerial());
    bool printTimings = false;
    
    if (! _useSubgridMeshForOptimalTestSolve)
    {
      // localStiffness should have dim. (numCells, numTrialFields, numTestFields)
      MeshPtr mesh = basisCache->mesh();
      if (mesh.get() == NULL)
      {
        cout << "localStiffnessMatrix requires BasisCache to have mesh set.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffnessMatrix requires BasisCache to have mesh set.");
      }
      const vector<GlobalIndexType>* cellIDs = &basisCache->cellIDs();
      int numCells = cellIDs->size();
      if (numCells != stiffnessEnriched.dimension(0))
      {
        cout << "localStiffnessMatrix requires basisCache->cellIDs() to have the same # of cells as the first dimension of localStiffness\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffnessMatrix requires basisCache->cellIDs() to have the same # of cells as the first dimension of localStiffness");
      }
      
      ElementTypePtr elemType = mesh->getElementType((*cellIDs)[0]); // we assume all cells provided are of the same type
      DofOrderingPtr trialOrder = elemType->trialOrderPtr;
      DofOrderingPtr testOrder = elemType->testOrderPtr;
      int numTestDofs = testOrder->totalDofs();
      int numTrialDofs = trialOrder->totalDofs();
      if ((numTrialDofs != stiffnessEnriched.dimension(1)) || (numTestDofs != stiffnessEnriched.dimension(2)))
      {
        cout << "localStiffness should have dimensions (C,numTestFields,numTrialFields).\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffness should have dimensions (C,numTrialFields,numTestFields).");
      }
      
      if (printTimings)
      {
        cout << "numCells: " << numCells << endl;
        cout << "numTestDofs: " << numTestDofs << endl;
        cout << "numTrialDofs: " << numTrialDofs << endl;
      }
      
      timer.ResetStartTime();
      Camellia::FieldContainer<double> cellSideParities = basisCache->getCellSideParities();
      
      if (ip == Teuchos::null)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BF: ip is null in localStiffnessMatrixAndRHS_DLS (for which we can't do Bubnov-Galerkin).");
      }
      else
      {
        int numCells = basisCache->getPhysicalCubaturePointsFC().dimension(0);
        int numTestDofs = testOrder->totalDofs();
        int numTrialDofs = trialOrder->totalDofs();
        
        Epetra_Time timer(*MPIWrapper::CommSerial());
        
        double timeG, timeB, timeT, timeK; // time to compute Gram matrix, the right-hand side B, time to solve GT = B, and time to compute K = B^T T.
        
        timer.ResetStartTime();
        // RHS:
        this->stiffnessMatrix(stiffnessEnriched, elemType, cellSideParities, basisCache, true, true);
        timeB = timer.ElapsedTime();
        
        Teuchos::Array<int> localIPDim(2);
        localIPDim[0] = numTestDofs;
        localIPDim[1] = numTestDofs;
        Teuchos::Array<int> localStiffnessEnrichedDim(2);
        localStiffnessEnrichedDim[0] = stiffnessEnriched.dimension(1);
        localStiffnessEnrichedDim[1] = stiffnessEnriched.dimension(2);
        
        Camellia::FieldContainer<Scalar> ipMatrix(numCells,numTestDofs,numTestDofs);
        DofOrderingPtr testOrder = elemType->testOrderPtr;
        timer.ResetStartTime();
        ip->computeInnerProductMatrix(ipMatrix, testOrder, ipBasisCache);
        timeG = timer.ElapsedTime();
        
        rhs->integrateAgainstStandardBasis(rhsEnriched,testOrder,basisCache);
        
        Teuchos::Array<int> localRHSEnrichedDim(2);
        localRHSEnrichedDim[0] = rhsEnriched.dimension(1);
        localRHSEnrichedDim[1] = 1;
        
        Teuchos::Array<int> localRHSDim(2);
        localRHSDim[0] = numTrialDofs;
        localRHSDim[1] = 1;
        
        timeT = 0;
        timeK = 0;
        timer.ResetStartTime();
        
        Camellia::FieldContainer<Scalar> dummyStiffness(numTrialDofs,numTrialDofs); // computed in factoredCholeskySolve, but ignored
        Camellia::FieldContainer<Scalar> dummyRHS(numTrialDofs,1); // computed in factoredCholeskySolve, but ignored
        for (int cellIndex=0; cellIndex < numCells; cellIndex++)
        {
          Camellia::FieldContainer<Scalar> cellIPMatrix(localIPDim, &ipMatrix(cellIndex,0,0));
          Camellia::FieldContainer<Scalar> cellStiffnessEnriched(localStiffnessEnrichedDim, &stiffnessEnriched(cellIndex,0,0));
          Camellia::FieldContainer<Scalar> cellRHSEnriched(localRHSEnrichedDim, &rhsEnriched(cellIndex,0));
          
          // impose any BCs from the test space:
          if (ip->hasBCsImposed())
          {
            auto testDofOrdinalsToZero = ip->homogeneousBCDofOrdinalsForCell(ipBasisCache->mesh(),ipBasisCache->cellIDs()[cellIndex]);
            for (auto testDofOrdinal : testDofOrdinalsToZero)
            {
              for (int trialOrdinal=0; trialOrdinal<numTrialDofs; trialOrdinal++)
              {
                cellStiffnessEnriched(trialOrdinal,testDofOrdinal) = 0.0;
              }
            }
          }
          
          int result = factoredCholeskySolve(cellIPMatrix, cellStiffnessEnriched, cellRHSEnriched, dummyStiffness, dummyRHS);
          TEUCHOS_TEST_FOR_EXCEPTION(result != 0, std::invalid_argument, "Cholesky solve returned non-zero result");
        }
        timeK = timer.ElapsedTime();
        
        if (_optimalTestTimingCallback)
        {
          _optimalTestTimingCallback(numCells,timeG,timeB,timeT,timeK,elemType);
        }
      }
      
      if (_rhsTimingCallback)
      {
        _rhsTimingCallback(numCells,rhsDeterminationTime,elemType);
      }
    }
    else
    {
      // 1. set up a (serial) Mesh for IP.  Ultimately, can/should cache these (one per ElementType), and simply relabel the vertices
      // 2. set up a SerialComm-based dof Epetra_Map (Tpetra_Map?), and create FECrsMatrix and RHSVector
      // 3. Fill both LHS and RHS using appropriate LinearTerms.
      // 4. Solve.  (Direct solver to start, but ultimately, I want to try GMG.  Can reuse the prolongation operator, too.)
      // 5. Compute the local stiffness matrix and RHS.
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Subgrid mesh optimal test solve implementation not yet complete!");
    }
    
    TimeLogger::sharedInstance()->stopTimer(timerHandle);
    
    if (printTimings)
    {
      cout << "testMatrixAssemblyTime: " << testMatrixAssemblyTime << " seconds.\n";
      cout << "localStiffnessDeterminationTime: " << localStiffnessDeterminationTime << " seconds.\n";
      cout << "rhsDeterminationTime: " << rhsDeterminationTime << " seconds.\n";
    }
  }
  
  template <typename Scalar>
  void TBF<Scalar>::localStiffnessMatrixAndRHS(Camellia::FieldContainer<Scalar> &localStiffness, Camellia::FieldContainer<Scalar> &rhsVector,
                                               TIPPtr<Scalar> ip, BasisCachePtr ipBasisCache, TRHSPtr<Scalar> rhs, BasisCachePtr basisCache)
  {
    TEUCHOS_FUNC_TIME_MONITOR("BF::localStiffnessMatrixAndRHS");
    int timerHandle = TimeLogger::sharedInstance()->startTimer(BF_LOCAL_STIFFNESS_AND_RHS_TIMER_STRING);
    double testMatrixAssemblyTime = 0, localStiffnessDeterminationTime = 0;
    double rhsDeterminationTime = 0;
    
    Epetra_Time timer(*MPIWrapper::CommSerial());
    bool printTimings = false;

    if (! _useSubgridMeshForOptimalTestSolve)
    {
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
      
      if (printTimings)
      {
        cout << "numCells: " << numCells << endl;
        cout << "numTestDofs: " << numTestDofs << endl;
        cout << "numTrialDofs: " << numTrialDofs << endl;
      }
      
      timer.ResetStartTime();
      const Camellia::FieldContainer<double> & cellSideParities = basisCache->getCellSideParities();

      if (ip == Teuchos::null)
      {
        // can we interpret as a Bubnov-Galerkin setting?
        TEUCHOS_TEST_FOR_EXCEPTION(numTestDofs != numTrialDofs, std::invalid_argument, "BF: ip is null, but the number of test dofs is different from the number of trial dofs (can't do Bubnov-Galerkin).");
        this->stiffnessMatrix(localStiffness, elemType, cellSideParities, basisCache);
        
        // the above stores in (trial, test) order; we want (test, trial) -- so we transpose cell-wise:
        Teuchos::Array<int> dim;
        dim.push_back(numTrialDofs);
        dim.push_back(numTrialDofs);
        
        for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
        {
          Camellia::FieldContainer<double> cellLocalStiffness(dim, &localStiffness(cellOrdinal,0,0));
          SerialDenseWrapper::transposeSquareMatrix(cellLocalStiffness); // transposes data in place
        }
        
        localStiffnessDeterminationTime += timer.ElapsedTime();
        // "timeB" is basically the localStiffnessDeterminationTime
        if (_optimalTestTimingCallback)
        {
          _optimalTestTimingCallback(numCells,0,localStiffnessDeterminationTime,0,0,elemType);
        }
        
        timer.ResetStartTime();
        rhs->integrateAgainstStandardBasis(rhsVector, testOrder, basisCache);
        rhsDeterminationTime += timer.ElapsedTime();
      }
      else if (_optimalTestSolver == FACTORED_CHOLESKY)
      {
        int numCells = basisCache->getPhysicalCubaturePointsFC().dimension(0);
        int numTestDofs = testOrder->totalDofs();
        int numTrialDofs = trialOrder->totalDofs();
        
        Epetra_Time timer(*MPIWrapper::CommSerial());
        
        double timeG, timeB, timeT, timeK; // time to compute Gram matrix, the right-hand side B, time to solve GT = B, and time to compute K = B^T T.
        
        Camellia::FieldContainer<Scalar> stiffnessEnriched(numCells,numTrialDofs,numTestDofs);
        
        timer.ResetStartTime();
        // RHS:
        this->stiffnessMatrix(stiffnessEnriched, elemType, cellSideParities, basisCache, true, _warnAboutZeroRowsAndColumns);
        timeB = timer.ElapsedTime();
        
        Teuchos::Array<int> localIPDim(2);
        localIPDim[0] = numTestDofs;
        localIPDim[1] = numTestDofs;
        Teuchos::Array<int> localStiffnessEnrichedDim(2);
        localStiffnessEnrichedDim[0] = stiffnessEnriched.dimension(1);
        localStiffnessEnrichedDim[1] = stiffnessEnriched.dimension(2);
        Teuchos::Array<int> localStiffnessDim(2);
        localStiffnessDim[0] = localStiffness.dimension(1);
        localStiffnessDim[1] = localStiffness.dimension(2);
        
        Camellia::FieldContainer<Scalar> ipMatrix(numCells,numTestDofs,numTestDofs);
        DofOrderingPtr testOrder = elemType->testOrderPtr;
        timer.ResetStartTime();
        ip->computeInnerProductMatrix(ipMatrix, testOrder, ipBasisCache);
        timeG = timer.ElapsedTime();
        
        Camellia::FieldContainer<Scalar> rhsEnriched(numCells,numTestDofs);
        rhs->integrateAgainstStandardBasis(rhsEnriched,testOrder,basisCache);
        
        Teuchos::Array<int> localRHSEnrichedDim(2);
        localRHSEnrichedDim[0] = rhsEnriched.dimension(1);
        localRHSEnrichedDim[1] = 1;
        
        Teuchos::Array<int> localRHSDim(2);
        localRHSDim[0] = numTrialDofs;
        localRHSDim[1] = 1;
        
        timeT = 0;
        timeK = 0;
        timer.ResetStartTime();
        for (int cellIndex=0; cellIndex < numCells; cellIndex++)
        {
          Camellia::FieldContainer<Scalar> cellIPMatrix(localIPDim, &ipMatrix(cellIndex,0,0));
          Camellia::FieldContainer<Scalar> cellStiffnessEnriched(localStiffnessEnrichedDim, &stiffnessEnriched(cellIndex,0,0));
          Camellia::FieldContainer<Scalar> cellStiffness(localStiffnessDim, &localStiffness(cellIndex,0,0));
          Camellia::FieldContainer<Scalar> cellRHSEnriched(localRHSEnrichedDim, &rhsEnriched(cellIndex,0));
          Camellia::FieldContainer<Scalar> cellRHS(localRHSDim, &rhsVector(cellIndex,0));

          // impose any BCs from the test space:
          if (ip->hasBCsImposed())
          {
            auto testDofOrdinalsToZero = ip->homogeneousBCDofOrdinalsForCell(ipBasisCache->mesh(),ipBasisCache->cellIDs()[cellIndex]);
            for (auto testDofOrdinal : testDofOrdinalsToZero)
            {
              for (int trialOrdinal=0; trialOrdinal<numTrialDofs; trialOrdinal++)
              {
                cellStiffnessEnriched(trialOrdinal,testDofOrdinal) = 0.0;
              }
            }
          }
          
          int result = factoredCholeskySolve(cellIPMatrix, cellStiffnessEnriched, cellRHSEnriched, cellStiffness, cellRHS);
          TEUCHOS_TEST_FOR_EXCEPTION(result != 0, std::invalid_argument, "Cholesky solve returned non-zero result");
        }
        timeK = timer.ElapsedTime();
        
        if (_optimalTestTimingCallback)
        {
          _optimalTestTimingCallback(numCells,timeG,timeB,timeT,timeK,elemType);
        }
      }
      else
      {
        //      cout << "ipMatrix:\n" << ipMatrix;
        
        timer.ResetStartTime();
        Camellia::FieldContainer<Scalar> optTestCoeffs(numCells,numTrialDofs,numTestDofs);
        
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
      
      if (_rhsTimingCallback)
      {
        _rhsTimingCallback(numCells,rhsDeterminationTime,elemType);
      }
    }
    else
    {
      // 1. set up a (serial) Mesh for IP.  Ultimately, can/should cache these (one per ElementType), and simply relabel the vertices
      // 2. set up a SerialComm-based dof Epetra_Map (Tpetra_Map?), and create FECrsMatrix and RHSVector
      // 3. Fill both LHS and RHS using appropriate LinearTerms.
      // 4. Solve.  (Direct solver to start, but ultimately, I want to try GMG.  Can reuse the prolongation operator, too.)
      // 5. Compute the local stiffness matrix and RHS.
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Subgrid mesh optimal test solve implementation not yet complete!");
    }
    
    TimeLogger::sharedInstance()->stopTimer(timerHandle);
    
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
  int TBF<Scalar>::optimalTestWeightsAndStiffness(Camellia::FieldContainer<Scalar> &optimalTestWeights,
                                                  Camellia::FieldContainer<Scalar> &stiffnessMatrix,
                                                  ElementTypePtr elemType,
                                                  const Camellia::FieldContainer<double> &cellSideParities,
                                                  BasisCachePtr stiffnessBasisCache,
                                                  IPPtr ip, BasisCachePtr ipBasisCache)
  {
    TEUCHOS_FUNC_TIME_MONITOR("BF::optimalTestWeightsAndStiffness");
    DofOrderingPtr trialOrdering = elemType->trialOrderPtr;
    DofOrderingPtr testOrdering = elemType->testOrderPtr;
    
    // all arguments are as in computeStiffnessMatrix, except:
    // optimalTestWeights, which has dimensions (numCells, numTrialDofs, numTestDofs)
    // innerProduct: the inner product which defines the sense in which these test functions are optimal
    int numCells = stiffnessBasisCache->getPhysicalCubaturePointsFC().dimension(0);
    int numTestDofs = testOrdering->totalDofs();
    int numTrialDofs = trialOrdering->totalDofs();
    
    Epetra_Time timer(*MPIWrapper::CommSerial());
    
    double timeG, timeB, timeT, timeK; // time to compute Gram matrix, the right-hand side B, time to solve GT = B, and time to compute K = B^T T.
    
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
    
    Camellia::FieldContainer<Scalar> rectangularStiffnessMatrix(numCells,numTestDofs,numTrialDofs);
    Teuchos::RCP<Camellia::FieldContainer<Scalar>> optimalTestRHS;
    
    int solvedAll = 0;
    
    // do we have the same BF for optimal test determination as we integrate against?
    // (this is the typical case, and lets us avoid integrating twice, as well as saving some memory)
    bool sameBF = (_bfForOptimalTestSolve == Teuchos::null);
    
    TBFPtr<Scalar> bfForOptimalTestSolve;
    if ( !sameBF )
    {
      bfForOptimalTestSolve = _bfForOptimalTestSolve;
      // in this case, we need separate allocation for rectangular stiffness and the "load" for the optimal test solve
      optimalTestRHS = Teuchos::rcp(new Camellia::FieldContainer<Scalar>(numCells,numTestDofs,numTrialDofs));
    }
    else
    {
      bfForOptimalTestSolve = Teuchos::rcp(this, false); // false: does not own memory
      optimalTestRHS = Teuchos::rcp(&rectangularStiffnessMatrix, false); // same memory will do
    }
    
    timer.ResetStartTime();
    // RHS:
    if (_optimalTestSolver != FACTORED_CHOLESKY)
    {
      bfForOptimalTestSolve->stiffnessMatrix(*optimalTestRHS, elemType, cellSideParities, stiffnessBasisCache);
      if (!sameBF)
      {
        this->stiffnessMatrix(rectangularStiffnessMatrix, elemType, cellSideParities, stiffnessBasisCache);
      }
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_optimalTestSolver == FACTORED_CHOLESKY, std::invalid_argument, "optimalTestWeightsAndStiffness() should not be called for FACTORED_CHOLESKY");
      // row-major order
//      rectangularStiffnessMatrix.resize(numCells,numTrialDofs,numTestDofs);
//      this->stiffnessMatrix(rectangularStiffnessMatrix, elemType, cellSideParities, stiffnessBasisCache, true, true);
    }
    timeB = timer.ElapsedTime();
    
    Teuchos::Array<int> cellOptimalWeightsTDim(2); // data stored in transposed order relative to what we'll eventually want (except in case of FACTORED_CHOLESKY)
    cellOptimalWeightsTDim[0] = rectangularStiffnessMatrix.dimension(1);
    cellOptimalWeightsTDim[1] = rectangularStiffnessMatrix.dimension(2);
    
    Teuchos::Array<int> localIPDim(2);
    localIPDim[0] = numTestDofs;
    localIPDim[1] = numTestDofs;
    Teuchos::Array<int> localRectangularStiffnessDim(2);
    localRectangularStiffnessDim[0] = rectangularStiffnessMatrix.dimension(1);
    localRectangularStiffnessDim[1] = rectangularStiffnessMatrix.dimension(2);
    Teuchos::Array<int> localStiffnessDim(2);
    localStiffnessDim[0] = stiffnessMatrix.dimension(1);
    localStiffnessDim[1] = stiffnessMatrix.dimension(2);
    
    Camellia::FieldContainer<Scalar> ipMatrix(numCells,numTestDofs,numTestDofs);
    DofOrderingPtr testOrder = elemType->testOrderPtr;
    timer.ResetStartTime();
    ip->computeInnerProductMatrix(ipMatrix, testOrder, ipBasisCache);
    timeG = timer.ElapsedTime();
    
    timeT = 0;
    timeK = 0;
    for (int cellIndex=0; cellIndex < numCells; cellIndex++)
    {
      int result;
      timer.ResetStartTime();
      Camellia::FieldContainer<Scalar> cellIPMatrix(localIPDim, &ipMatrix(cellIndex,0,0));
      Camellia::FieldContainer<Scalar> cellOptTestRHS(localRectangularStiffnessDim, &(*optimalTestRHS)(cellIndex,0,0));
      Camellia::FieldContainer<Scalar> cellRectangularStiffness(localRectangularStiffnessDim, &rectangularStiffnessMatrix(cellIndex,0,0));
      Camellia::FieldContainer<Scalar> cellStiffness(localStiffnessDim, &stiffnessMatrix(cellIndex,0,0));
      Camellia::FieldContainer<Scalar> cellOptimalWeightsT(cellOptimalWeightsTDim, &optimalTestWeights(cellIndex,0,0));
      
      // impose any BCs from the test space:
      if (ip->hasBCsImposed())
      {
        auto testDofOrdinalsToZero = ip->homogeneousBCDofOrdinalsForCell(ipBasisCache->mesh(),ipBasisCache->cellIDs()[cellIndex]);
        for (auto testDofOrdinal : testDofOrdinalsToZero)
        {
          for (int trialOrdinal=0; trialOrdinal<numTrialDofs; trialOrdinal++)
          {
            cellRectangularStiffness(trialOrdinal,testDofOrdinal) = 0.0;
          }
        }
      }
      
      switch(_optimalTestSolver)
      {
        case CHOLESKY:
        {
          bool allowIPOverwrite = false; // assert that we won't be using cellIPMatrix again
          result = SerialDenseWrapper::solveSPDSystemMultipleRHS(cellOptimalWeightsT, cellIPMatrix, cellOptTestRHS, allowIPOverwrite);
          if (result != 0)
          {
            // may be that we're not SPD numerically
            cout << "During optimal test weight solution, SPD solve returned error " << result << ".  Solving with LU factorization instead of SPD solve.\n";
            result = SerialDenseWrapper::solveSystemMultipleRHS(cellOptimalWeightsT, cellIPMatrix, cellOptTestRHS);
          }
        }
          break;
        case QR:
        {
          bool useIPTranspose = true; // true value may allow less memory to be used during solveSystemUsingQR() (maybe only if we can get overwriting to work, below)
          bool allowIPOverwrite = true; // assert that we won't be using cellIPMatrix again
          result = SerialDenseWrapper::solveSystemUsingQR(cellOptimalWeightsT, cellIPMatrix, cellOptTestRHS, useIPTranspose, allowIPOverwrite);
        }
          break;
        case LU:
          result = SerialDenseWrapper::solveSystemMultipleRHS(cellOptimalWeightsT, cellIPMatrix, cellOptTestRHS);
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported case");
      }
      timeT += timer.ElapsedTime();
      
      timer.ResetStartTime();
      // multiply to determine stiffness matrix.
      SerialDenseWrapper::multiply(cellStiffness, cellOptimalWeightsT, cellRectangularStiffness, 'T', 'N'); // transpose A; don't transpose B
      // transpose the optimal test weights -- since this is a view into optimalTestWeights, this reorders (part of) that matrix according to contract with caller
      SerialDenseWrapper::transposeMatrix(cellOptimalWeightsT);
      timeK += timer.ElapsedTime();
      
      if (result != 0)
      {
        solvedAll = result;
      }
    }
   
    if (_optimalTestTimingCallback)
    {
      _optimalTestTimingCallback(numCells,timeG,timeB,timeT,timeK,elemType);
    }
    bool printTimings = false;
    if (printTimings)
    {
      cout << "BF timings: on " << numCells << " elements, computed G in " << timeG << " seconds, B in " << timeB << " seconds; solve for T in " << timeT;
      cout << " seconds; compute K=B^T T in " << timeK << " seconds." << endl;
    }
    return solvedAll;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::setBFForOptimalTestSolve(TBFPtr<Scalar> bf)
  {
    _bfForOptimalTestSolve = bf;
    if ((bf != Teuchos::null) && (_optimalTestSolver == FACTORED_CHOLESKY))
    {
      // can't use FACTORED_CHOLESKY if we have a different BF for optimal test solve
      _optimalTestSolver = QR;
    }
  }
  
  template <typename Scalar>
  void TBF<Scalar>::setOptimalTestTimingCallback(std::function<void(int numElements, double timeG, double timeB, double timeT, double timeK, ElementTypePtr elemType)> &optimalTestTimingCallback)
  {
    _optimalTestTimingCallback = optimalTestTimingCallback;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::setRHSTimingCallback(std::function<void(int numElements, double timeRHS, ElementTypePtr elemType)> &rhsTimingCallback)
  {
    _rhsTimingCallback = rhsTimingCallback;
  }
  
  template <typename Scalar>
  typename TBF<Scalar>::OptimalTestSolver TBF<Scalar>::optimalTestSolver() const
  {
    return _optimalTestSolver;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::setOptimalTestSolver(OptimalTestSolver choice)
  {
    _optimalTestSolver = choice;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::setPrintTermWiseIntegrationOutput(bool value)
  {
    _printTermWiseIntegrationOutput = value;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::setUseIterativeRefinementsWithSPDSolve(bool value)
  {
    _useIterativeRefinementsWithSPDSolve = value;
  }
  
  template <typename Scalar>
  void TBF<Scalar>::setUseSubgridMeshForOptimalTestFunctions(bool value)
  {
    _useSubgridMeshForOptimalTestSolve = value;
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
  TLinearTermPtr<Scalar> TBF<Scalar>::testFunctional(TSolutionPtr<Scalar> trialSolution, bool excludeBoundaryTerms,
                                                     SolutionComponent solutionComponent, int solutionStage)
  {
    TLinearTermPtr<Scalar> functional = Teuchos::rcp(new LinearTerm());
    auto trialVarMap = _varFactory->trialVars();
    std::vector<VarPtr> trialVars;
    for (auto trialEntry : trialVarMap)
    {
      trialVars.push_back(trialEntry.second);
    }
    auto solnMap = PreviousSolutionFunction<Scalar>::functionMap(trialVars, trialSolution, solutionComponent, solutionStage);
    for (TBilinearTerm<Scalar> & bt : _terms)
    {
      TLinearTermPtr<Scalar> trialTerm = bt.first;
      TLinearTermPtr<Scalar> testTerm  = bt.second;
      TFunctionPtr<Scalar> trialValue = trialTerm->evaluate(solnMap);
      if ( (! excludeBoundaryTerms) || (! trialValue->boundaryValueOnly()) )
      {
        functional = functional + trialValue * testTerm;
      }
    }
    return functional;
  }
  
  template <typename Scalar>
  TLinearTermPtr<Scalar> TBF<Scalar>::testFunctional(const std::map<int,FunctionPtr> &solnMap)
  {
    TLinearTermPtr<Scalar> functional = Teuchos::rcp(new LinearTerm());
    for (auto bilinearTerm : _terms)
    {
      TLinearTermPtr<Scalar> trialTerm = bilinearTerm.first;
      TLinearTermPtr<Scalar> testTerm = bilinearTerm.second;
      TFunctionPtr<Scalar> trialValue = trialTerm->evaluate(solnMap);
      functional = functional + trialValue * testTerm;
    }
    return functional;
  }
  
  template <typename Scalar>
  TLinearTermPtr<Scalar> TBF<Scalar>::trialFunctional(const std::map<int,FunctionPtr> &testMap)
  {
    TLinearTermPtr<Scalar> functional = Teuchos::rcp(new LinearTerm());
    bool weightFluxesByParity = true; // trying something...
    FunctionPtr parity = Function::sideParity();
    for (auto bilinearTerm : _terms)
    {
      TLinearTermPtr<Scalar> trialTerm = bilinearTerm.first;
      if ((trialTerm->termType() == FLUX) && weightFluxesByParity)
      {
        trialTerm = parity * trialTerm;
      }
      TLinearTermPtr<Scalar> testTerm = bilinearTerm.second;
      TFunctionPtr<Scalar> testValue = testTerm->evaluate(testMap);
      bool overrideTypeCheck = true; // we can have fluxes, traces, and fields all in this trial term
      functional->addTerm(testValue * trialTerm, overrideTypeCheck);
    }
    return functional;
  }

  // BK: function returns the adjoint operator in an uweak BF (i.e. A* in (u,A*v) = (f,v) for all v) applied to the DPG* solution (i.e. A*v, when v is a fixed DPG* soln)
  template <typename Scalar>
  map<int, TFunctionPtr<Scalar> > TBF<Scalar>::applyAdjointOperatorDPGstar(TRieszRepPtr<double> dualSolution)
  {
    TEUCHOS_FUNC_TIME_MONITOR("BF::applyAdjointOperatorDPGstar");
    // group test functional contributions based on trial variables (field only)
    map<int, TLinearTermPtr<Scalar>> testTermsForVarID;
    for (TBilinearTerm<Scalar> bt : _terms)
    {
      TLinearTermPtr<Scalar> trialTerm = bt.first;
      TLinearTermPtr<Scalar> testTerm = bt.second;

      vector< TLinearSummand<Scalar> > summands = trialTerm->summands();
      for (TLinearSummand<Scalar> summand : trialTerm->summands())
      {
        VarPtr trialVar = summand.second;
        if (trialVar->varType() == FIELD)
        {
          TFunctionPtr<Scalar> f = summand.first;
          f = Function::op( f, trialVar->op() );
          if (testTermsForVarID.find(trialVar->ID()) == testTermsForVarID.end())
          {
            testTermsForVarID[trialVar->ID()] = Teuchos::rcp( new LinearTerm );
          }
          testTermsForVarID[trialVar->ID()]->addTerm( f * testTerm );
        }
      }
    }

    // apply grouped test functional contributions to DPG* solution
    map<int, TFunctionPtr<Scalar>> opDualSolutionForVarID;
    for ( auto varIDTestTermEntry : testTermsForVarID )
    {
      // double weight = 1.0;
      // int varID = testTermIt->first;
      // if (trialVarWeights.find(varID) != trialVarWeights.end())
      // {
      //   double trialWeight = trialVarWeights.find(varID)->second;
      //   if (trialWeight <= 0)
      //   {
      //     TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "variable weights must be positive.");
      //   }
      //   weight = 1.0 / sqrt(trialWeight);
      // }

      TFunctionPtr<Scalar> fxn_sum = Function::zero();
      TLinearTermPtr<Scalar> testFunctional = varIDTestTermEntry.second;
      
      // std::cout << testFunctional->displayString() << std::endl;
      
      for ( TLinearSummand<Scalar> testSummand : testFunctional->summands() )
      {
        TFunctionPtr<Scalar> f = testSummand.first;
        VarPtr testVar = testSummand.second;

        TFunctionPtr<Scalar> fxn = Teuchos::rcp(new RepFunction<Scalar>(testVar, dualSolution));
        fxn = Function::op( fxn, testVar->op() );

        // std::cout << fxn->displayString() << std::endl;

        fxn_sum = fxn_sum + f * fxn;
      }

      int varID = varIDTestTermEntry.first;
      opDualSolutionForVarID[varID] = fxn_sum;

      // std::cout << fxn_sum->displayString() << std::endl;

      
    }

    return opDualSolutionForVarID;
  }
  
  template <typename Scalar>
  std::map<int,FunctionPtr> TBF<Scalar>::solutionMap(SolutionPtr soln, const std::string &expString)
  {
    std::map<int,FunctionPtr> solnMap;
    for (int ID : _trialIDs)
    {
      auto var = _varFactory->trial(ID);
      bool weightFluxesBySideParity = true;
      auto fxn = Function::solution(var, soln, weightFluxesBySideParity, expString);
      solnMap[ID] = fxn;
    }
    return solnMap;
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
        }
        testVar = vf->testVar(name, space);
      }
      return vf;
    }
  }
  template class TBF<double>;
}
