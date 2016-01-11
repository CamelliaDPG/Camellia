//
//  RHS.cpp
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "RHS.h"
#include "BasisCache.h"

using namespace Intrepid;
using namespace Camellia;

template <typename Scalar>
void TRHS<Scalar>::addTerm( TLinearTermPtr<Scalar> rhsTerm )
{
  TEUCHOS_TEST_FOR_EXCEPTION( rhsTerm->termType() != TEST, std::invalid_argument, "RHS should only involve test functions (no trials)");
  TEUCHOS_TEST_FOR_EXCEPTION( rhsTerm->rank() != 0, std::invalid_argument, "RHS only handles scalar terms.");
  if (_lt.get())
  {
    _lt = _lt + rhsTerm;
  }
  else
  {
    _lt = rhsTerm;
  }
  //  _terms.push_back( rhsTerm );

  set<int> varIDs = rhsTerm->varIDs();
  _varIDs.insert(varIDs.begin(),varIDs.end());
}

template <typename Scalar>
void TRHS<Scalar>::addTerm( VarPtr v )
{
  addTerm( Teuchos::rcp( new TLinearTerm<Scalar>( v ) ) );
}

// at a conceptual/design level, this method isn't necessary
template <typename Scalar>
bool TRHS<Scalar>::nonZeroRHS(int testVarID)
{
  if (!_legacySubclass)
  {
    return (_varIDs.find(testVarID) != _varIDs.end());
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Legacy subclasses must implement nonZeroRHS!");
  }
}

template <typename Scalar>
TLinearTermPtr<Scalar> TRHS<Scalar>::linearTerm()
{
  if (_lt == Teuchos::null)
  {
    _lt = Teuchos::rcp( new TLinearTerm<Scalar> );
  }
  return _lt;
}

template <typename Scalar>
TLinearTermPtr<Scalar> TRHS<Scalar>::linearTermCopy()
{
  return Teuchos::rcp( new TLinearTerm<Scalar> (*_lt) );
}

template <typename Scalar>
void TRHS<Scalar>::integrateAgainstStandardBasis(FieldContainer<Scalar> &rhsVector,
    Teuchos::RCP<DofOrdering> testOrdering,
    BasisCachePtr basisCache)
{
  // rhsVector dimensions are: (numCells, # testOrdering Dofs)

  if (!_legacySubclass)
  {
    rhsVector.initialize(0.0);

    if ( _lt.get() )
    {
      _lt->integrate(rhsVector, testOrdering, basisCache);
    }
  }
  else     // legacy subclass support
  {
    // steps:
    // 0. Set up Cubature
    // 3. For each optimalTestFunction
    //   a. Apply the value operators to the basis in the DofOrdering, at the cubature points
    //   b. weight with Jacobian/Piola transform and cubature weights
    //   c. Pass the result to RHS to get resultant values at each point
    //   d. Sum up (integrate) and place in rhsVector according to DofOrdering indices

    // 0. Set up Cubature

    CellTopoPtr cellTopo = basisCache->cellTopology();

    unsigned numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    unsigned spaceDim = cellTopo->getDimension();

    TEUCHOS_TEST_FOR_EXCEPTION( ( testOrdering->totalDofs() != rhsVector.dimension(1) ),
                                std::invalid_argument,
                                "testOrdering->totalDofs() (=" << testOrdering->totalDofs() << ") and rhsVector.dimension(1) (=" << rhsVector.dimension(1) << ") do not match.");

    set<int> testIDs = testOrdering->getVarIDs();
    set<int>::iterator testIterator;

    BasisPtr testBasis;

    FieldContainer<Scalar> rhsPointValues; // the rhs method will resize...

    rhsVector.initialize(0.0);

    for (testIterator = testIDs.begin(); testIterator != testIDs.end(); testIterator++)
    {
      int testID = *testIterator;

      vector<Camellia::EOperator> testOperators = this->operatorsForTestID(testID);
      int operatorIndex = -1;
      for (vector<Camellia::EOperator>::iterator testOpIt=testOperators.begin();
           testOpIt != testOperators.end(); testOpIt++)
      {
        operatorIndex++;
        Camellia::EOperator testOperator = *testOpIt;
        bool notZero = this->nonZeroRHS(testID);
        if (notZero)   // compute the integral(s)
        {

          testBasis = testOrdering->getBasis(testID);

          Teuchos::RCP< const FieldContainer<Scalar> > testValuesTransformedWeighted;

          testValuesTransformedWeighted = basisCache->getTransformedWeightedValues(testBasis,testOperator);
          FieldContainer<double> physCubPoints = basisCache->getPhysicalCubaturePoints();

          vector<int> testDofIndices = testOrdering->getDofIndices(testID,VOLUME_INTERIOR_SIDE_ORDINAL);

          this->rhs(testID,operatorIndex,basisCache,rhsPointValues);

          //   d. Sum up (integrate)
          // to integrate, first multiply the testValues (C,F,P) or (C,F,P,D)
          //               by the rhsPointValues (C,P) or (C,P,D), respectively, and then sum.
          int numPoints = rhsPointValues.dimension(1);
          for (unsigned k=0; k < numCells; k++)
          {
            for (int i=0; i < testBasis->getCardinality(); i++)
            {
              int testDofIndex = testDofIndices[i];
              for (int ptIndex=0; ptIndex < numPoints; ptIndex++)
              {
                if (rhsPointValues.rank() == 2)
                {
                  rhsVector(k,testDofIndex) += (*testValuesTransformedWeighted)(k,i,ptIndex) * rhsPointValues(k,ptIndex);
                }
                else
                {
                  for (int d=0; d<spaceDim; d++)
                  {
                    rhsVector(k,testDofIndex) += (*testValuesTransformedWeighted)(k,i,ptIndex,d) * rhsPointValues(k,ptIndex,d);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // cout << "rhsVector: " << endl << rhsVector;
}

template <typename Scalar>
void TRHS<Scalar>::integrateAgainstOptimalTests(FieldContainer<Scalar> &rhsVector,
    const FieldContainer<Scalar> &optimalTestWeights,
    Teuchos::RCP<DofOrdering> testOrdering,
    BasisCachePtr basisCache)
{
  // rhsVector dimensions are: (numCells, # trialOrdering Dofs) == (numCells, # optimal test functions)
  // optimalTestWeights dimensions are: (numCells, numTrial, numTest) -- numTrial is the optTest index

  TEUCHOS_TEST_FOR_EXCEPTION( ( optimalTestWeights.dimension(1) != rhsVector.dimension(1) ),
                              std::invalid_argument,
                              "optimalTestWeights.dimension(1) (=" << optimalTestWeights.dimension(1) << ") and rhsVector.dimension(1) (=" << rhsVector.dimension(1) << ") do not match.");

  // can represent this as the multiplication of optTestWeights matrix against the standard-basis RHS vector
  unsigned numCells = basisCache->getPhysicalCubaturePoints().dimension(0);

  unsigned numTrialDofs = optimalTestWeights.dimension(1);
  unsigned numTestDofs = testOrdering->totalDofs();

  FieldContainer<Scalar> rhsVectorStandardBasis(numCells,numTestDofs);

  integrateAgainstStandardBasis(rhsVectorStandardBasis,testOrdering,basisCache);

  rhsVector.initialize(0.0);

  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int trialDofIndex=0; trialDofIndex<numTrialDofs; trialDofIndex++)
    {
      for (int testDofIndex=0; testDofIndex<numTestDofs; testDofIndex++)
      {
        rhsVector(cellIndex,trialDofIndex) += optimalTestWeights(cellIndex,trialDofIndex,testDofIndex)
                                              * rhsVectorStandardBasis(cellIndex,testDofIndex);
      }
    }
  }

//  cout << "RHS vector:\n" << rhsVector;
}

template <typename Scalar>
vector<Camellia::EOperator> TRHS<Scalar>::operatorsForTestID(int testID)
{
  vector<Camellia::EOperator> ops;
  ops.push_back( Camellia::OP_VALUE);
  return ops;
}

template <typename Scalar>
void TRHS<Scalar>::rhs(int testVarID, int operatorIndex, Teuchos::RCP<BasisCache> basisCache, FieldContainer<Scalar> &values)
{
  rhs(testVarID, operatorIndex, basisCache->getPhysicalCubaturePoints(), values);
}

template <typename Scalar>
void TRHS<Scalar>::rhs(int testVarID, int operatorIndex, const FieldContainer<double> &physicalPoints, FieldContainer<Scalar> &values)
{
  TEUCHOS_TEST_FOR_EXCEPTION(operatorIndex != 0, std::invalid_argument, "base rhs() method called for operatorIndex != 0");
  rhs(testVarID,physicalPoints,values);
}

template <typename Scalar>
void TRHS<Scalar>::rhs(int testVarID, const FieldContainer<double> &physicalPoints, FieldContainer<Scalar> &values)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "no rhs() implemented within RHS");
}

namespace Camellia
{
template class TRHS<double>;
}

