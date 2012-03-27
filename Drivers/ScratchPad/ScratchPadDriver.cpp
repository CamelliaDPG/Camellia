//
//  ScratchPadDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/26/12.
//  Copyright (c) 2012. All rights reserved.
//

#include <iostream>

#include "InnerProductScratchPad.h"
#include "MathInnerProduct.h"
#include "StokesBilinearForm.h"
#include "ElementType.h"
#include "TestSuite.h"
#include "BasisFactory.h"

typedef Teuchos::RCP<IP> IPPtr;
typedef Teuchos::RCP<DPGInnerProduct> DPGInnerProductPtr;
typedef Teuchos::RCP<shards::CellTopology> CellTopoPtr;
typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;
typedef Teuchos::RCP<ElementType> ElementTypePtr;

int main(int argc, char *argv[]) {
  // define nodes for test
  FieldContainer<double> quadPoints(1,4,2);
  
  quadPoints(0,0,0) = 0.0; // x1
  quadPoints(0,0,1) = 0.0; // y1
  quadPoints(0,1,0) = 1.0;
  quadPoints(0,1,1) = 0.0;
  quadPoints(0,2,0) = 1.0;
  quadPoints(0,2,1) = 1.0;
  quadPoints(0,3,0) = 0.0;
  quadPoints(0,3,1) = 1.0;
  
  // 1. Implement the math norm for Stokes
  VarFactory varFactory; // provides unique IDs for test/trial functions, etc.
  VarPtr q1 = varFactory.testVar("q_1", HDIV);
  VarPtr q2 = varFactory.testVar("q_2", HDIV);
  VarPtr v1 = varFactory.testVar("v_1", HGRAD);
  VarPtr v2 = varFactory.testVar("v_2", HGRAD);
  VarPtr v3 = varFactory.testVar("v_3", HGRAD);
  
  double mu = 1.0;
  
  // the following should be replaced by some sort of DofOrderingFactory call or something...
  int polyOrder = 1;
  CellTopoPtr quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  BasisPtr divBasis = BasisFactory::getBasis(polyOrder, quadTopoPtr->getKey(), 
                                             IntrepidExtendedTypes::FUNCTION_SPACE_HDIV);
  BasisPtr gradBasis = BasisFactory::getBasis(polyOrder, quadTopoPtr->getKey(), 
                                             IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  DofOrderingPtr testOrdering = Teuchos::rcp(new DofOrdering() );
  testOrdering->addEntry(q1->ID(), divBasis, 1);
  testOrdering->addEntry(q2->ID(), divBasis, 1);
  testOrdering->addEntry(v1->ID(), gradBasis, 0);
  testOrdering->addEntry(v2->ID(), gradBasis, 0);
  testOrdering->addEntry(v3->ID(), gradBasis, 0);
  
  // just use testOrdering for both trial and test spaces (we only use to define BasisCache)
  ElementTypePtr elemType  = Teuchos::rcp( new ElementType(testOrdering, testOrdering, quadTopoPtr) );
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType, true) ); // true: test vs. test
  basisCache->setPhysicalCellNodes(quadPoints,vector<int>(1),false); // false: don't create side cache
  
  IPPtr mathIP = Teuchos::rcp(new IP());
  mathIP->addTerm(v1);
  mathIP->addTerm(v1->grad());
  mathIP->addTerm(v2);
  mathIP->addTerm(v2->grad());
  mathIP->addTerm(v3);
  mathIP->addTerm(v3->grad());
  mathIP->addTerm(q1);
  mathIP->addTerm(q1->div());
  mathIP->addTerm(q2);
  mathIP->addTerm(q2->div());
  
  Teuchos::RCP<BilinearForm> stokesBF = Teuchos::rcp(new StokesBilinearForm(mu));
  DPGInnerProductPtr autoMathIP = Teuchos::rcp( new MathInnerProduct(stokesBF) );
  
  int numCells = quadPoints.dimension(0);
  FieldContainer<double> expectedValues(numCells, testOrdering->totalDofs(), testOrdering->totalDofs() );
  FieldContainer<double> actualValues  (numCells, testOrdering->totalDofs(), testOrdering->totalDofs() );
  
  autoMathIP->computeInnerProductMatrix(expectedValues,testOrdering,basisCache);
  mathIP->computeInnerProductMatrix(actualValues,testOrdering,basisCache);
  
  double tol = 1e-14;
  double maxDiff = 0.0;
  if ( ! TestSuite::fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
    cout << "Test failed: automatic mathematician's inner product differs from new IP; maxDiff " << maxDiff << ".\n";
    cout << "Automatic: \n" << expectedValues;
    cout << "New IP: \n" << actualValues;
  } else {
    cout << "Automatic mathematician's inner product and new IP agree!!\n";
  }
  
  cout << "*** Math IP: ***\n";
  mathIP->printInteractions();
  
  IPPtr qoptIP = Teuchos::rcp(new IP());
                                               
  double beta = 1e-1;
  qoptIP->addTerm( q1->x() / mu + v1->dx() );
  qoptIP->addTerm( q1->x() / (2.0 * mu) + q2->y() / (2.0 * mu) );
  qoptIP->addTerm( q1->y() / (2.0 * mu) + q2->x() / (2.0 * mu) + v1->dy() + v2->dx() );
  qoptIP->addTerm( q2->y() / (2.0 * mu) + v2->dy() );
  qoptIP->addTerm( q1->y() - q2->x() );
  qoptIP->addTerm( q1->div() - v3->dx() );
  qoptIP->addTerm( sqrt(beta) * q1 );
  qoptIP->addTerm( sqrt(beta) * q2 );
  qoptIP->addTerm( sqrt(beta) * v1 );
  qoptIP->addTerm( sqrt(beta) * v2 );
  qoptIP->addTerm( sqrt(beta) * v3 );
  
  cout << "*** Quasi-Optimal IP: ***\n";
  qoptIP->printInteractions();
  
  return 0;
}