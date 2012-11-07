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
  vector< VarPtr > vgpFields;
  
  vector< PolyExactFunctions > polyExactFunctions;
  vector< pair< int, int > > meshDimensions; // horizontal x vertical cells
  vector< int > pToAddValues;
  vector< double > muValues;
  
  FieldContainer<double> quadPoints;
  
  Teuchos::RCP<Mesh> _vgpStokesMesh; // used for both Stokes and Navier-Stokes
  Teuchos::RCP< VGPStokesFormulation > _vgpStokesFormulation;
  Teuchos::RCP< VGPNavierStokesFormulation > _vgpNavierStokesFormulation;
  Teuchos::RCP< Solution > _vgpStokesSolution, _vgpNavierStokesSolution;
  
  FieldContainer<double> _testPoints;
  
  Teuchos::RCP<ExactSolution> _vgpStokesExactSolution;
  Teuchos::RCP<ExactSolution> _vgpNavierStokesExactSolution;
  
  void setup();
  void teardown() {}
  bool functionsAgree(FunctionPtr f1, FunctionPtr f2, BasisCachePtr basisCache);
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  
  bool testVGPStokesFormulationConsistency();
  bool testVGPStokesFormulationCorrectness();
  
  bool testVGPNavierStokesFormulationConsistency();
  bool testVGPNavierStokesFormulationCorrectness();
  
  std::string testSuiteName();
};


#endif
