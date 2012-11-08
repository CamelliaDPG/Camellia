//
//  IncompressibleFormulationsTests.h
//  Camellia
//
//  Created by Nathan Roberts on 4/9/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_IncompressibleFormulationsTests_h
#define Camellia_IncompressibleFormulationsTests_h

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Mesh.h"
#include "InnerProductScratchPad.h"
#include "BasisCache.h"
#include "ExactSolution.h"

#include "NavierStokesFormulation.h"
#include "StokesFormulation.h"

typedef vector< pair< FunctionPtr, int > > PolyExactFunctions; // (u1, u2, p) -> poly degree

class IncompressibleFormulationsTests : public TestSuite {
  FunctionPtr x, x2, x3, y, y2, y3, zero;
  
  VarPtr u1_vgp, u2_vgp, sigma11_vgp, sigma12_vgp, sigma21_vgp, sigma22_vgp, p_vgp;
  VarPtr v1_vgp, v2_vgp, tau1_vgp, tau2_vgp, q_vgp;
  vector< VarPtr > vgpFields;
  vector< VarPtr > vgpTests;
  
  vector< PolyExactFunctions > polyExactFunctions;
  vector< pair< int, int > > meshDimensions; // horizontal x vertical cells
  vector< int > pToAddValues;
  vector< double > muValues;
  
  SpatialFilterPtr entireBoundary;
  
  FieldContainer<double> quadPoints;
  
  void setup();
  void teardown() {}
  bool functionsAgree(FunctionPtr f1, FunctionPtr f2, Teuchos::RCP<Mesh> mesh);
  
  bool ltsAgree(LinearTermPtr lt1, LinearTermPtr lt2, Teuchos::RCP<Mesh> mesh);
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  
  bool testVGPStokesFormulationConsistency();
  bool testVGPStokesFormulationCorrectness();
  
  bool testVGPNavierStokesFormulationConsistency();
  bool testVGPNavierStokesFormulationCorrectness();
  
  std::string testSuiteName();
};


#endif
