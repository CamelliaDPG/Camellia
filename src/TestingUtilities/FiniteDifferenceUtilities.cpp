#include "FiniteDifferenceUtilities.h"
#include "TestingUtilities.h"

#include "RHS.h"

using namespace Camellia;

double FiniteDifferenceUtilities::finiteDifferenceGradient(MeshPtr mesh, RieszRepPtr residual, SolutionPtr backgroundSoln, int dofIndex){
  residual->computeRieszRep();
  double fx =  residual->getNorm();
  
  SolutionPtr solnPerturbation = TestingUtilities::makeNullSolution(mesh);

  // create perturbation in direction du
  solnPerturbation->clear(); // clear all solns
  TestingUtilities::initializeSolnCoeffs(solnPerturbation);
  TestingUtilities::setSolnCoeffForGlobalDofIndex(solnPerturbation,1.0,dofIndex);
  double h = 1e-7;
  backgroundSoln->addSolution(solnPerturbation,h);
      
  residual->computeRieszRep();
  double fxh = residual->getNorm();
  // get 1/2 squared norm (cost function) for each quantity
  double f = fx*fx*.5; 
  double fh = fxh*fxh*.5;
  double fd_gradient = (fh-f)/h;
      
  // remove contribution
  backgroundSoln->addSolution(solnPerturbation,-h);      

  return fd_gradient;
}

