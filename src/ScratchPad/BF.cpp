//
//  BF.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "BF.h"
#include "VarFactory.h"

BF::BF( VarFactory varFactory ) { // copies (note that external changes in VarFactory won't be registered by BF)
  _varFactory = varFactory;
  // set super's ID containers:
  _trialIDs = _varFactory.trialIDs();
  _testIDs = _varFactory.testIDs();
}

void BF::addTerm( LinearTermPtr trialTerm, LinearTermPtr testTerm ) {
  _terms.push_back( make_pair( trialTerm, testTerm ) );
}

void BF::addTerm( VarPtr trialVar, LinearTermPtr testTerm ) {
  addTerm( Teuchos::rcp( new LinearTerm(trialVar) ), testTerm );
}

void BF::addTerm( VarPtr trialVar, VarPtr testVar ) {
  addTerm( Teuchos::rcp( new LinearTerm(trialVar) ), Teuchos::rcp( new LinearTerm(testVar) ) );
}

void BF::addTerm( LinearTermPtr trialTerm, VarPtr testVar) {
  addTerm( trialTerm, Teuchos::rcp( new LinearTerm(testVar) ) );
}

// BilinearForm implementation:
const string & BF::testName(int testID) {
  return _varFactory.test(testID)->name();
}
const string & BF::trialName(int trialID) {
  return _varFactory.trial(trialID)->name();
}

EFunctionSpaceExtended BF::functionSpaceForTest(int testID) {
  return efsForSpace(_varFactory.test(testID)->space());
}

EFunctionSpaceExtended BF::functionSpaceForTrial(int trialID) {
  return efsForSpace(_varFactory.trial(trialID)->space());
}

bool BF::isFluxOrTrace(int trialID) {
  VarType varType = _varFactory.trial(trialID)->varType();
  return (varType == FLUX) || (varType == TRACE);
}

void BF::printTrialTestInteractions() {
  cout << "BF::printTrialTestInteractions() not yet implemented.\n";
}

// new, EXPERIMENTAL stiffnessMatrix implementation
// using the new, EXPERIMENTAL LinearTerm::integrate() 
void BF::stiffnessMatrix(FieldContainer<double> &stiffness, Teuchos::RCP<ElementType> elemType,
                         FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache) {
  // stiffness is sized as (C, FTest, FTrial)
  stiffness.initialize(0.0);
  basisCache->setCellSideParities(cellSideParities);
  
  for ( vector< BilinearTerm >:: iterator btIt = _terms.begin();
       btIt != _terms.end(); btIt++) {
    BilinearTerm bt = *btIt;
    LinearTermPtr trialTerm = btIt->first;
    LinearTermPtr testTerm = btIt->second;
    trialTerm->integrate(stiffness, elemType->trialOrderPtr,
                         testTerm, elemType->testOrderPtr, basisCache);
  }
}


// original stiffnessMatrix implementation below:
//void BF::stiffnessMatrix(FieldContainer<double> &stiffness, Teuchos::RCP<ElementType> elemType,
//                     FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache) {
//  // stiffness is sized as (C, FTest, FTrial)
//  DofOrderingPtr trialOrdering = elemType->trialOrderPtr;
//  DofOrderingPtr testOrdering = elemType->testOrderPtr;
//  FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePoints();
//  
//  unsigned numCells = physicalCubaturePoints.dimension(0);
//  unsigned numPoints = physicalCubaturePoints.dimension(1);
//  unsigned spaceDim = physicalCubaturePoints.dimension(2);
//  
//  shards::CellTopology cellTopo = basisCache->cellTopology();
//  
//  Teuchos::Array<int> ltValueDim;
//  ltValueDim.push_back(numCells);
//  ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
//  ltValueDim.push_back(numPoints);
//  
//  stiffness.initialize(0.0);
//  
//  for ( vector< BilinearTerm >:: iterator btIt = _terms.begin();
//       btIt != _terms.end(); btIt++) {
//    BilinearTerm bt = *btIt;
//    LinearTermPtr trialTerm = btIt->first;
//    LinearTermPtr testTerm = btIt->second;
//    set<int> trialIDs = trialTerm->varIDs();
//    set<int> testIDs = testTerm->varIDs();
//    
//    int rank = trialTerm->rank();
//    TEST_FOR_EXCEPTION( rank != testTerm->rank(), std::invalid_argument, "test and trial ranks disagree." );
//    
//    set<int>::iterator trialIt;
//    set<int>::iterator testIt;
//    
//    Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > trialBasis, testBasis;
//    
//    bool boundaryTerm = (trialTerm->termType() == FLUX) || (trialTerm->termType() == TRACE);
//    // num "sides" for volume integral: 1...
//    int numSides = boundaryTerm ? cellTopo.getSideCount() : 1;
//    for (int sideIndex = 0; sideIndex < numSides; sideIndex++) {
//      int numPointsSide = basisCache->getPhysicalCubaturePointsForSide(sideIndex).dimension(1);
//      
//      for (testIt= testIDs.begin(); testIt != testIDs.end(); testIt++) {
//        int testID = *testIt;
//        testBasis = testOrdering->getBasis(testID);
//        int numDofsTest = testBasis->getCardinality();
//        
//        // set up values container for test
//        Teuchos::Array<int> ltValueDim1 = ltValueDim;
//        ltValueDim1[1] = numDofsTest;
//        for (int d=0; d<rank; d++) {
//          ltValueDim1.push_back(spaceDim);
//        }
//        ltValueDim1[2] = boundaryTerm ? numPointsSide : numPoints;
//        FieldContainer<double> testValues(ltValueDim1);
//        bool applyCubatureWeights = true;
//        if (! boundaryTerm) {
//          testTerm->values(testValues,testID,testBasis,basisCache,applyCubatureWeights);
//        } else {
//          testTerm->values(testValues,testID,testBasis,basisCache,applyCubatureWeights,sideIndex);
//        }
//        
//        for (trialIt= trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
//          int trialID = *trialIt;
//          trialBasis = trialOrdering->getBasis(trialID,sideIndex);
//          int numDofsTrial = trialBasis->getCardinality();
//          
//          // set up values container for test2:
//          Teuchos::Array<int> ltValueDim2 = ltValueDim1;
//          ltValueDim2[1] = numDofsTrial;
//          
//          FieldContainer<double> trialValues(ltValueDim2);
//          
//          if (! boundaryTerm ) {
//            trialTerm->values(trialValues,trialID,trialBasis,basisCache);
//          } else {
//            trialTerm->values(trialValues,trialID,trialBasis,basisCache,false,sideIndex);
//            if ( trialTerm->termType() == FLUX ) {
//              // we need to multiply the trialValues by the parity of the normal, since
//              // the trial implicitly contains an outward normal, and we need to adjust for the fact
//              // that the neighboring cells have opposite normal
//              // trialValues should have dimensions (numCells,numFields,numCubPointsSide)
//              int numFields = trialValues.dimension(1);
//              int numPoints = trialValues.dimension(2);
//              for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
//                double parity = cellSideParities(cellIndex,sideIndex);
//                if (parity != 1.0) {  // otherwise, we can just leave things be...
//                  for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
//                    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
//                      trialValues(cellIndex,fieldIndex,ptIndex) *= parity;
//                    }
//                  }
//                }
//              }
//            }
//          }
//          
//          FieldContainer<double> miniMatrix( numCells, numDofsTest, numDofsTrial );
//          
//          FunctionSpaceTools::integrate<double>(miniMatrix,testValues,trialValues,COMP_CPP);
//          
//          // there may be a more efficient way to do this copying:
//          for (int i=0; i < numDofsTest; i++) {
//            int testDofIndex = testOrdering->getDofIndex(testID,i);
//            for (int j=0; j < numDofsTrial; j++) {
//              int trialDofIndex = boundaryTerm ? trialOrdering->getDofIndex(trialID,j,sideIndex)
//              : trialOrdering->getDofIndex(trialID,j);
//              for (unsigned k=0; k < numCells; k++) {
//                stiffness(k,testDofIndex,trialDofIndex) += miniMatrix(k,i,j);
//              }
//            }
//          }
//        }
//      }
//    }
//  }
//}