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

class IncompressibleFormulationsTests : public TestSuite
{
  bool _thoroughMode; // if true, tests take considerably longer...

  FunctionPtr x, x2, x3, y, y2, y3, zero;

  VarPtr u1_vgp, u2_vgp, sigma11_vgp, sigma12_vgp, sigma21_vgp, sigma22_vgp, p_vgp;
  VarPtr u1hat_vgp, u2hat_vgp, t1n_vgp, t2n_vgp;

  VarPtr v1_vgp, v2_vgp, tau1_vgp, tau2_vgp, q_vgp;
  vector< VarPtr > vgpFields;
  vector< VarPtr > vgpTests;
  VarFactoryPtr vgpVarFactory;

  vector< PolyExactFunctions > polyExactFunctions;
  vector< pair< int, int > > meshDimensions; // horizontal x vertical cells
  vector< int > pToAddValues;
  vector< double > muValues;

  SpatialFilterPtr entireBoundary;

  FieldContainer<double> quadPoints, quadPointsKovasznay;



  void setup();
  void teardown();
  bool functionsAgree(FunctionPtr f1, FunctionPtr f2,
                      Teuchos::RCP<Mesh> mesh, double tol = 1e-14);

  bool ltsAgree(LinearTermPtr lt1, LinearTermPtr lt2,
                Teuchos::RCP<Mesh> mesh, VarFactoryPtr varFactory, double tol = 1e-14);
  bool ltsAgree(LinearTermPtr lt1, LinearTermPtr lt2,
                Teuchos::RCP<Mesh> mesh, IPPtr ip, double tol = 1e-14);

  map<int, FunctionPtr > vgpSolutionMap(FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact, double Re);

  vector< VarPtr > nonZeroComponents( LinearTermPtr lt, vector< VarPtr > &varsToTry, Teuchos::RCP<Mesh> mesh, IPPtr ip );
public:
  void runTests(int &numTestsRun, int &numTestsPassed);

  bool testVGPNavierStokesLocalConservation();

  bool testVGPStokesFormulationGraphNorm();
  bool testVVPStokesFormulationGraphNorm();

  bool testVGPStokesFormulationConsistency();
  bool testVGPStokesFormulationCorrectness();

  bool testVGPNavierStokesFormulationConsistency();
  bool testVGPNavierStokesFormulationCorrectness();
  bool testVGPNavierStokesFormulationKovasnayConvergence();
  bool testVGPNavierStokesFormulationLocalConservation();

public:
  IncompressibleFormulationsTests(bool thorough = true);
  std::string testSuiteName();
};


#endif
