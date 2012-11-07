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

typedef Basis<double, FieldContainer<double> > DoubleBasis;
typedef Teuchos::RCP< DoubleBasis > BasisPtr;

class IncompressibleFormulationsTests : public TestSuite {
  Teuchos::RCP<Mesh> _vgpStokesMesh; // used for both Stokes and Navier-Stokes
  Teuchos::RCP< VGPStokesFormulation > _vgpStokesFormulation;
  Teuchos::RCP< VGPNavierStokesFormulation > _vgpNavierStokesFormulation;
  Teuchos::RCP< Solution > _vgpStokesSolution, _vgpNavierStokesSolution;
  
  FieldContainer<double> _testPoints;
  ElementTypePtr _elemType;
  BasisCachePtr _basisCache;
  
  Teuchos::RCP<ExactSolution> _vgpStokesExactSolution;
  Teuchos::RCP<ExactSolution> _vgpNavierStokesExactSolution;
  
  void setup();
  void teardown() {}
  bool functionsAgree(FunctionPtr f1, FunctionPtr f2, BasisCachePtr basisCache);
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  
  bool testVGPStokesFormulation();
  
  bool testVGPNavierStokesFormulation();
  
  std::string testSuiteName();
};


#endif
