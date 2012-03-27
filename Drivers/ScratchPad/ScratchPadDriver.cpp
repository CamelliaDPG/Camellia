//
//  ScratchPadDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/26/12.
//  Copyright (c) 2012. All rights reserved.
//

#include <iostream>

#include "InnerProductScratchPad.h"

typedef Teuchos::RCP<IP> IPPtr;

int main(int argc, char *argv[]) {
  // 1. Implement the math norm for Stokes
  VarFactory varFactory; // provides unique IDs for test/trial functions, etc.
  VarPtr v1 = varFactory.testVar("v_1", HGRAD);
  VarPtr v2 = varFactory.testVar("v_2", HGRAD);
  VarPtr v3 = varFactory.testVar("v_3", HGRAD);
  VarPtr q1 = varFactory.testVar("q_1", HDIV);
  VarPtr q2 = varFactory.testVar("q_2", HDIV);
  
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
  
  cout << "*** Math IP: ***\n";
  mathIP->printInteractions();
  
  double mu = 1.0;
  double beta = 1e-1;
  IPPtr qoptIP = Teuchos::rcp(new IP());
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