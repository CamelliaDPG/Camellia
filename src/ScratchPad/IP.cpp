//
//  IP.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "Intrepid_FunctionSpaceTools.hpp"

#include "IP.h"
#include "SerialDenseMatrixUtility.h"
#include "SerialDenseWrapper.h"
#include "VarFactory.h"
#include "BasisCache.h"
#include "CellTopology.h"

using namespace Intrepid;
using namespace Camellia;

IP::IP() {
  _isLegacySubclass = false;
}
// if the terms are a1, a2, ..., then the inner product is (a1,a1) + (a2,a2) + ... 

IP::IP(BFPtr bfs) {
  _bilinearForm = bfs;
  _isLegacySubclass = true;
}

// added by Nate
LinearTermPtr IP::evaluate(const map< int, FunctionPtr> &varFunctions) {
  // include both the boundary and non-boundary parts
  return evaluate(varFunctions,true) + evaluate(varFunctions,false);
}

// added by Jesse - evaluate inner product at given varFunctions
LinearTermPtr IP::evaluate(const map< int, FunctionPtr> &varFunctions, bool boundaryPart) {
  LinearTermPtr ltEval = Teuchos::rcp(new LinearTerm);
  for ( vector< LinearTermPtr >:: const_iterator ltIt = _linearTerms.begin(); ltIt != _linearTerms.end(); ltIt++) {
    LinearTermPtr lt = *ltIt;
    FunctionPtr weight = lt->evaluate(varFunctions,boundaryPart);
    ltEval->addTerm(weight*lt);
  }
  return ltEval;
}

void IP::addTerm( LinearTermPtr a ) {
  _linearTerms.push_back(a);
}

void IP::addTerm( VarPtr v ) {
  _linearTerms.push_back( Teuchos::rcp( new LinearTerm(v) ) );
}

void IP::addZeroMeanTerm( LinearTermPtr a) {
  _zeroMeanTerms.push_back(a);
}

void IP::addZeroMeanTerm( VarPtr v ) {
  _zeroMeanTerms.push_back( Teuchos::rcp( new LinearTerm(v) ) );
}

void IP::addBoundaryTerm( LinearTermPtr a ) {
  _boundaryTerms.push_back(a);
}

void IP::addBoundaryTerm( VarPtr v ) {
  _boundaryTerms.push_back( Teuchos::rcp( new LinearTerm(v) ) );
}

void IP::applyInnerProductData(FieldContainer<double> &testValues1,
                               FieldContainer<double> &testValues2,
                               int testID1, int testID2, int operatorIndex,
                               Teuchos::RCP<BasisCache> basisCache) {
  applyInnerProductData(testValues1, testValues2, testID1, testID2, operatorIndex, basisCache->getPhysicalCubaturePoints());
}

void IP::applyInnerProductData(FieldContainer<double> &testValues1,
                                   FieldContainer<double> &testValues2,
                                   int testID1, int testID2, int operatorIndex,
                                   const FieldContainer<double>& physicalPoints) {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "You must override some version of applyInnerProductData!");
}

void IP::computeInnerProductMatrix(FieldContainer<double> &innerProduct,
                                   Teuchos::RCP<DofOrdering> dofOrdering, shards::CellTopology &cellTopo,
                                   FieldContainer<double>& physicalCellNodes) {
  if (_isLegacySubclass) {
    CellTopoPtr cellTopoPtr = CellTopology::cellTopology(cellTopo);
    Teuchos::RCP<ElementType> elemTypePtr = Teuchos::rcp( new ElementType(dofOrdering,dofOrdering, cellTopoPtr) );
    Teuchos::RCP<Mesh> nullMeshPtr = Teuchos::rcp( (Mesh*) NULL );
    BasisCachePtr ipBasisCache = Teuchos::rcp(new BasisCache(elemTypePtr, nullMeshPtr,true));
    ipBasisCache->setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(), false);
    computeInnerProductMatrix(innerProduct,dofOrdering,ipBasisCache);
  }
}

void IP::computeInnerProductMatrix(FieldContainer<double> &innerProduct,
                                   Teuchos::RCP<DofOrdering> dofOrdering,
                                   Teuchos::RCP<BasisCache> basisCache) {
  if (_isLegacySubclass) {
    // much of this code is the same as what's in the volume integration in computeStiffness...
    FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePoints();
    
    unsigned numCells = physicalCubaturePoints.dimension(0);
    unsigned spaceDim = physicalCubaturePoints.dimension(2);
    
    vector<int> testIDs = _bilinearForm->testIDs();
    vector<int>::iterator testIterator1;
    vector<int>::iterator testIterator2;
    
    BasisPtr test1Basis, test2Basis;
    
    innerProduct.initialize(0.0);
    
    for (testIterator1= testIDs.begin(); testIterator1 != testIDs.end(); testIterator1++) {
      int testID1 = *testIterator1;
      for (testIterator2= testIDs.begin(); testIterator2 != testIDs.end(); testIterator2++) {
        int testID2 = *testIterator2;
        
        vector<Camellia::EOperator> test1Operators;
        vector<Camellia::EOperator> test2Operators;
        
        operators(testID1,testID2,test1Operators,test2Operators);
        
        // check dimensions
        TEUCHOS_TEST_FOR_EXCEPTION( ( test1Operators.size() != test2Operators.size() ),
                                   std::invalid_argument,
                                   "test1Operators.size() and test2Operators.size() do not match.");
        
        vector<Camellia::EOperator>::iterator op1It;
        vector<Camellia::EOperator>::iterator op2It = test2Operators.begin();
        int operatorIndex = 0;
        for (op1It=test1Operators.begin(); op1It != test1Operators.end(); op1It++) {
          Camellia::EOperator op1 = *(op1It);
          Camellia::EOperator op2 = *(op2It);
          FieldContainer<double> test1Values; // these will be resized inside applyOperator..
          FieldContainer<double> test2Values; // derivative values
          
          test1Basis = dofOrdering->getBasis(testID1);
          test2Basis = dofOrdering->getBasis(testID2);
          
          int numDofs1 = test1Basis->getCardinality();
          int numDofs2 = test2Basis->getCardinality();
          
          FieldContainer<double> miniMatrix( numCells, numDofs1, numDofs2 );
          
          Teuchos::RCP< const FieldContainer<double> > test1ValuesTransformedWeighted, test2ValuesTransformed;
          
          test1ValuesTransformedWeighted = basisCache->getTransformedWeightedValues(test1Basis,op1);
          test2ValuesTransformed = basisCache->getTransformedValues(test2Basis,op2);
          
          FieldContainer<double> innerProductDataAppliedToTest2 = *test2ValuesTransformed; // copy first
          FieldContainer<double> innerProductDataAppliedToTest1 = *test1ValuesTransformedWeighted; // copy first
          
          //cout << "rank of test2ValuesTransformed: " << test2ValuesTransformed->rank() << endl;
          applyInnerProductData(innerProductDataAppliedToTest1, innerProductDataAppliedToTest2,
                                testID1, testID2, operatorIndex, basisCache);
          
          Intrepid::FunctionSpaceTools::integrate<double>(miniMatrix,innerProductDataAppliedToTest1,
                                                          innerProductDataAppliedToTest2,COMP_BLAS);
          
          int test1DofOffset = dofOrdering->getDofIndex(testID1,0);
          int test2DofOffset = dofOrdering->getDofIndex(testID2,0);
          
          // there may be a more efficient way to do this copying:
          for (int i=0; i < numDofs1; i++) {
            for (int j=0; j < numDofs2; j++) {
              for (unsigned k=0; k < numCells; k++) {
                innerProduct(k,i+test1DofOffset,j+test2DofOffset) += miniMatrix(k,i,j);
              }
            }
          }
          
          op2It++;
          operatorIndex++;
        }
        
      }
    }
  } else { // _isLegacySubclass is false
    
    // innerProduct FC is sized as (C,F,F)
    FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePoints();
    
    unsigned numCells = physicalCubaturePoints.dimension(0);
    unsigned numPoints = physicalCubaturePoints.dimension(1);
  //  unsigned spaceDim = physicalCubaturePoints.dimension(2);
    unsigned numDofs = dofOrdering->totalDofs();
    
    Teuchos::Array<int> ltValueDim;
    ltValueDim.push_back(numCells);
    ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
    ltValueDim.push_back(numPoints);
    
    innerProduct.initialize(0.0);
    
    int totalBasisCardinality = dofOrdering->getTotalBasisCardinality();
    // want to fit all basis values in 256K with a little room to spare -- 32768 doubles would be no room to spare
    // 256K is the size of Sandy Bridge's L2 cache.  Blue Gene Q has 32 MB shared L2 cache, so maybe there we could go bigger
    // (fitting within L1 is likely a non-starter for 3D DPG)
    int maxValuesAllowed = 30000;
    int maxPointsPerPhase = max(2, maxValuesAllowed / totalBasisCardinality); // minimally, compute 2 points at once
    
  //  basisCache->setMaxPointsPerCubaturePhase(maxPointsPerPhase);
    basisCache->setMaxPointsPerCubaturePhase(-1); // old behavior

    for (int phase=0; phase < basisCache->getCubaturePhaseCount(); phase++) {
      basisCache->setCubaturePhase(phase);
      for ( vector< LinearTermPtr >:: iterator ltIt = _linearTerms.begin();
           ltIt != _linearTerms.end(); ltIt++) {
        LinearTermPtr lt = *ltIt;
        // integrate lt against itself
        lt->integrate(innerProduct,dofOrdering,lt,dofOrdering,basisCache,basisCache->isSideCache());
      }
    }
    
    basisCache->setMaxPointsPerCubaturePhase(-1); // infinite (BasisCache doesn't yet properly support phased *side* caches)
    
    bool enforceNumericalSymmetry = false;
    if (enforceNumericalSymmetry) {
      for (unsigned int c=0; c < numCells; c++)
        for (unsigned int i=0; i < numDofs; i++)
          for (unsigned int j=i+1; j < numDofs; j++)
          {
            innerProduct(c,i,j) = (innerProduct(c,i,j) + innerProduct(c,j,i)) / 2.0;
            innerProduct(c,j,i) = innerProduct(c,i,j);
          }
    }
    
    // boundary terms:
    for ( vector< LinearTermPtr >:: iterator btIt = _boundaryTerms.begin();
         btIt != _boundaryTerms.end(); btIt++) {
      LinearTermPtr bt = *btIt;
      bool forceBoundary = true; // force interpretation of this as a term on the element boundary
      bt->integrate(innerProduct,dofOrdering,bt,dofOrdering,basisCache,forceBoundary);
    }
    
    // zero mean terms:
    for ( vector< LinearTermPtr >:: iterator ztIt = _zeroMeanTerms.begin();
         ztIt != _zeroMeanTerms.end(); ztIt++) {
      LinearTermPtr zt = *ztIt;
      FieldContainer<double> avgVector(numCells, numDofs);
      // Integrate against 1
      zt->integrate(avgVector, dofOrdering, basisCache);


      // cout << numDofs << avgVector << endl;

      // Sum into innerProduct
      for (unsigned int c=0; c < numCells; c++)
        for (unsigned int i=0; i < numDofs; i++)
          for (unsigned int j=0; j < numDofs; j++)
          {
            double valAdd = avgVector(c, i) * avgVector(c, j);
            // cout << "(" << innerProduct(c, i, j) << ", " << valAdd << ") ";
            innerProduct(c, i, j) += valAdd;
          }
    }
  }
}

double IP::computeMaxConditionNumber(DofOrderingPtr testSpace, BasisCachePtr basisCache) {
  int testDofs = testSpace->totalDofs();
  int numCells = basisCache->cellIDs().size();
  FieldContainer<double> innerProduct(numCells,testDofs,testDofs);
  this->computeInnerProductMatrix(innerProduct, testSpace, basisCache);
  double maxConditionNumber = -1;
  Teuchos::Array<int> cellIP_dim;
  cellIP_dim.push_back(testDofs);
  cellIP_dim.push_back(testDofs);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    FieldContainer<double> cellIP = FieldContainer<double>(cellIP_dim,&innerProduct(cellIndex,0,0) );
    double conditionNumber = SerialDenseMatrixUtility::estimate2NormConditionNumber(cellIP);
    maxConditionNumber = max(maxConditionNumber,conditionNumber);
  }
  return maxConditionNumber;
}

// compute IP vector when var==fxn
void IP::computeInnerProductVector(FieldContainer<double> &ipVector, 
                                   VarPtr var, FunctionPtr fxn,
                                   Teuchos::RCP<DofOrdering> dofOrdering, 
                                   Teuchos::RCP<BasisCache> basisCache) {
  // ipVector FC is sized as (C,F)
  FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePoints();
  
  if (!fxn.get()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fxn cannot be null!");
  }
  
  unsigned numCells = physicalCubaturePoints.dimension(0);
  unsigned numPoints = physicalCubaturePoints.dimension(1);
//  unsigned spaceDim = physicalCubaturePoints.dimension(2);
//  unsigned numDofs = dofOrdering->totalDofs();
    
  Teuchos::Array<int> ltValueDim;
  ltValueDim.push_back(numCells);
  ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
  ltValueDim.push_back(numPoints);
  
  ipVector.initialize(0.0);
  
  for ( vector< LinearTermPtr >:: iterator ltIt = _linearTerms.begin();
       ltIt != _linearTerms.end(); ltIt++) {
    LinearTermPtr lt = *ltIt;
    // integrate lt against itself, evaluated at var = fxn
    lt->integrate(ipVector,dofOrdering,lt,var,fxn,basisCache);
  }
  
  // boundary terms:
  for ( vector< LinearTermPtr >:: iterator btIt = _boundaryTerms.begin();
       btIt != _boundaryTerms.end(); btIt++) {
    LinearTermPtr bt = *btIt;
    bool forceBoundary = true; // force interpretation of this as a term on the element boundary
    bt->integrate(ipVector,dofOrdering,bt,var,fxn,basisCache,forceBoundary);
  }
  
  // zero mean terms:
  for ( vector< LinearTermPtr >:: iterator ztIt = _zeroMeanTerms.begin();
       ztIt != _zeroMeanTerms.end(); ztIt++) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "zero mean terms not yet supported in IP vector computation");
  }  
}

bool IP::hasBoundaryTerms() {
  if (_isLegacySubclass) return false;
  else return _boundaryTerms.size() > 0;
}

void IP::operators(int testID1, int testID2, 
                   vector<Camellia::EOperator> &testOp1,
                   vector<Camellia::EOperator> &testOp2) {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "IP::operators() not implemented.");
}

void IP::printInteractions() {
  if (_isLegacySubclass) {
    cout << "Inner product: test interactions\n";
    vector<int> testIDs = _bilinearForm->testIDs();
    for (vector<int>::iterator testIt = testIDs.begin(); testIt != testIDs.end(); testIt++) {
      int testID = *testIt;
      cout << endl << "****** Interactions with test variable " << _bilinearForm->testName(testID) << " ******* " << endl;
      bool first = true;
      for (vector<int>::iterator testIt2 = testIDs.begin(); testIt2 != testIDs.end(); testIt2++) {
        int testID2 = *testIt2;
        vector<Camellia::EOperator> ops1, ops2;
        operators(testID, testID2, ops1, ops2);
        int numOps = ops1.size();
        for (int i=0; i<numOps; i++) {
          if ( ! first) cout << " + ";
          cout << Camellia::operatorName(ops1[i]) << " " << _bilinearForm->testName(testID) << " ";
          cout << Camellia::operatorName(ops2[i]) << " " << _bilinearForm->testName(testID2);
          first = false;
        }
      }
      cout << endl;
    }
    return;
  } else {
    cout << "_linearTerms:\n";
    for (vector< LinearTermPtr >::iterator ltIt = _linearTerms.begin();
         ltIt != _linearTerms.end(); ltIt++) {
      cout << (*ltIt)->displayString() << endl;
    }
    cout << "_boundaryTerms:\n";
    for (vector< LinearTermPtr >::iterator ltIt = _boundaryTerms.begin();
         ltIt != _boundaryTerms.end(); ltIt++) {
      cout << (*ltIt)->displayString() << endl;
    }
    cout << "_zeroMeanTerms:\n";
    for (vector< LinearTermPtr >::iterator ltIt = _zeroMeanTerms.begin();
         ltIt != _zeroMeanTerms.end(); ltIt++) {
      cout << (*ltIt)->displayString() << endl;
    }
  }
}

pair<IPPtr, VarPtr> IP::standardInnerProductForFunctionSpace(Camellia::EFunctionSpace fs, bool useTraceVar, int spaceDim) {
  IPPtr ip = Teuchos::rcp( new IP );
  VarFactory vf;
  Camellia::Space space = Camellia::spaceForEFS(fs);
  VarPtr var = useTraceVar ? vf.traceVar("v",space) : vf.testVar("v", space);
  
  ip->addTerm(var);
  
  switch (fs) {
    case Camellia::FUNCTION_SPACE_HVOL:
    case Camellia::FUNCTION_SPACE_VECTOR_HVOL:
      break;
    case Camellia::FUNCTION_SPACE_HGRAD:
    case Camellia::FUNCTION_SPACE_VECTOR_HGRAD:
      ip->addTerm(var->grad());
      break;
    case Camellia::FUNCTION_SPACE_HCURL:
      ip->addTerm(var->curl(spaceDim));
      break;
    case Camellia::FUNCTION_SPACE_HDIV:
      ip->addTerm(var->div());
      break;
    case Camellia::FUNCTION_SPACE_REAL_SCALAR:
      break;
      
    default:
      cout << "Error: unhandled function space.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled function space");
      break;
  }
  return make_pair(ip,var);
}

IPPtr IP::ip() {
  return Teuchos::rcp( new IP );
}