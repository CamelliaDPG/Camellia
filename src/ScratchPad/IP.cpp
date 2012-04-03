//
//  IP.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "IP.h"

// to satisfy the compiler, call the DPGInnerProduct constructor with a null argument:
IP::IP() : DPGInnerProduct( Teuchos::rcp( (BilinearForm*) NULL ) ) {}
// if the terms are a1, a2, ..., then the inner product is (a1,a1) + (a2,a2) + ... 

void IP::addTerm( LinearTermPtr a) {
  _linearTerms.push_back(a);
}

void IP::addTerm( VarPtr v ) {
  _linearTerms.push_back( Teuchos::rcp( new LinearTerm(v) ) );
}

void IP::addBoundaryTerm( LinearTermPtr a ) {
  _boundaryTerms.push_back(a);
}

void IP::addBoundaryTerm( VarPtr v ) {
  _boundaryTerms.push_back( Teuchos::rcp( new LinearTerm(v) ) );
}

void IP::computeInnerProductMatrix(FieldContainer<double> &innerProduct, 
                                   Teuchos::RCP<DofOrdering> dofOrdering,
                                   Teuchos::RCP<BasisCache> basisCache) {
  // innerProduct FC is sized as (C,F,F)
  FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePoints();
  
  unsigned numCells = physicalCubaturePoints.dimension(0);
  unsigned numPoints = physicalCubaturePoints.dimension(1);
  unsigned spaceDim = physicalCubaturePoints.dimension(2);
  
  shards::CellTopology cellTopo = basisCache->cellTopology();
  
  Teuchos::Array<int> ltValueDim;
  ltValueDim.push_back(numCells);
  ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
  ltValueDim.push_back(numPoints);
  
  innerProduct.initialize(0.0);
  
  for ( vector< LinearTermPtr >:: iterator ltIt = _linearTerms.begin();
       ltIt != _linearTerms.end(); ltIt++) {
    LinearTermPtr lt = *ltIt;
    // integrate lt against itself
    lt->integrate(innerProduct,dofOrdering,lt,dofOrdering,basisCache);
  }
  
  // boundary terms:
  for ( vector< LinearTermPtr >:: iterator btIt = _boundaryTerms.begin();
       btIt != _boundaryTerms.end(); btIt++) {
    LinearTermPtr bt = *btIt;
    bool forceBoundary = true; // force interpretation of this as a term on the element boundary
    bt->integrate(innerProduct,dofOrdering,bt,dofOrdering,basisCache,forceBoundary);
  }
}

bool IP::hasBoundaryTerms() {
  return _boundaryTerms.size() > 0;
}

// old inner product computation below
//void IP::computeInnerProductMatrix(FieldContainer<double> &innerProduct, 
//                                   Teuchos::RCP<DofOrdering> dofOrdering,
//                                   Teuchos::RCP<BasisCache> basisCache) {
//  // innerProduct FC is sized as (C,F,F)
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
//  innerProduct.initialize(0.0);
//  
//  for ( vector< LinearTermPtr >:: iterator ltIt = _linearTerms.begin();
//       ltIt != _linearTerms.end(); ltIt++) {
//    LinearTermPtr lt = *ltIt;
//    set<int> testIDs = lt->varIDs();
//    int rank = lt->rank();
//    
//    set<int>::iterator testIt1;
//    set<int>::iterator testIt2;
//    
//    Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > test1Basis, test2Basis;
//    
//    for (testIt1= testIDs.begin(); testIt1 != testIDs.end(); testIt1++) {
//      int testID1 = *testIt1;
//      test1Basis = dofOrdering->getBasis(testID1);
//      int numDofs1 = test1Basis->getCardinality();
//      
//      // set up values container for test1
//      Teuchos::Array<int> ltValueDim1 = ltValueDim;
//      ltValueDim1[1] = numDofs1;
//      for (int d=0; d<rank; d++) {
//        ltValueDim1.push_back(spaceDim);
//      }
//      FieldContainer<double> test1Values(ltValueDim1);
//      lt->values(test1Values,testID1,test1Basis,basisCache);
//      
//      for (testIt2= testIDs.begin(); testIt2 != testIDs.end(); testIt2++) {
//        int testID2 = *testIt2;
//        test2Basis = dofOrdering->getBasis(testID2);
//        int numDofs2 = test2Basis->getCardinality();
//        
//        // set up values container for test2:
//        Teuchos::Array<int> ltValueDim2 = ltValueDim1;
//        ltValueDim2[1] = numDofs2;
//        
//        FieldContainer<double> test2ValuesWeighted(ltValueDim2);
//        
//        lt->values(test2ValuesWeighted,testID2,test2Basis,basisCache,true);
//        
//        FieldContainer<double> miniMatrix( numCells, numDofs1, numDofs2 );
//        
//        FunctionSpaceTools::integrate<double>(miniMatrix,test1Values,test2ValuesWeighted,COMP_CPP);
//        
//        int test1DofOffset = dofOrdering->getDofIndex(testID1,0);
//        int test2DofOffset = dofOrdering->getDofIndex(testID2,0);
//        
//        // there may be a more efficient way to do this copying:
//        for (unsigned k=0; k < numCells; k++) {
//          for (int i=0; i < numDofs1; i++) {
//            for (int j=0; j < numDofs2; j++) {
//              innerProduct(k,i+test1DofOffset,j+test2DofOffset) += miniMatrix(k,i,j);
//            }
//          }
//        }
//      }
//    }
//  }
//}

void IP::operators(int testID1, int testID2, 
                   vector<IntrepidExtendedTypes::EOperatorExtended> &testOp1,
                   vector<IntrepidExtendedTypes::EOperatorExtended> &testOp2) {
  TEST_FOR_EXCEPTION(true, std::invalid_argument, "IP::operators() not implemented.");
}

void IP::printInteractions() {
  cout << "IP::printInteractions() not yet implemented.\n";
}
