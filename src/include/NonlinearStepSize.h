//
//  NonlinearStepSize.h
//  Camellia
//
//  Created by Nathan Roberts on 2/24/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_NonlinearStepSize_h
#define Camellia_NonlinearStepSize_h

#include "Solution.h"
#include "RieszRep.h"

namespace Camellia {
  class NonlinearStepSize {
    double _fixedStepSize;
  public:
    NonlinearStepSize(double fixedStepSize = 0.5) {
      _fixedStepSize = fixedStepSize;
    }
    virtual double stepSize(TSolutionPtr<double> u, TSolutionPtr<double> du) {
      return _fixedStepSize;
    }
  };

  class LineSearchStep {
    double _nlRes;
    Teuchos::RCP<RieszRep> _residual;    // _residual is a nonlinear residual WHICH MUST DEPEND ON THE BACKGROUND FLOW
   public:
   LineSearchStep(Teuchos::RCP<RieszRep> residual){ // don't need fixed step size, set to 0
      _residual = residual;
      _nlRes = 1e9; // initialize it to something big
    }
    double getNLResidual(){
      return _nlRes;
    }
    double stepSize(TSolutionPtr<double> u, TSolutionPtr<double> du){

      int maxIter = 25;
      double stepLength = 1.0;
      double c_decrease = .5; // decrease by this factor each time
      double tol = 1e-6;
      double minStepSize = 1e-8;

      double newNLErr, prevNLErr;
      _residual->computeRieszRep(); prevNLErr = _residual->getNorm();

      bool NLErrorDecreased = false;
      int iter = 0;
      while ((!NLErrorDecreased) && (iter<maxIter)){
        u->addSolution(du,stepLength); // add contribution to compute new NL residual
        _residual->computeRieszRep(); newNLErr = _residual->getNorm();
        u->addSolution(du,-stepLength); // remove contribution
        if (newNLErr > prevNLErr + tol){
    stepLength *= c_decrease;
    stepLength = max(stepLength,minStepSize);
        }else{
    NLErrorDecreased = true;
    _nlRes = newNLErr;
        }
        iter++;
      }
      return stepLength;
    }

    // Modification of routine: in the event that your nonlinear error changes under solution->solve() (i.e. when your fluxes are not linearized)
    double stepSize(TSolutionPtr<double> u, TSolutionPtr<double> du, double prevNLErr){

      int maxIter = 25;
      double stepLength = 1.0;
      double c_decrease = .5; // decrease by this factor each time
      double tol = 1e-6;
      double minStepSize = 1e-8;

      double newNLErr;
      bool NLErrorDecreased = false;
      int iter = 0;
      while ((!NLErrorDecreased) && (iter<maxIter)){
        u->addSolution(du,stepLength); // add contribution to compute new NL residual
        _residual->computeRieszRep(); newNLErr = _residual->getNorm();
        u->addSolution(du,-stepLength); // remove contribution
        if (newNLErr > prevNLErr + tol){
    stepLength *= c_decrease;
    stepLength = max(stepLength,minStepSize);
        }else{
    NLErrorDecreased = true;
        }
        iter++;
      }
      //    cout << "returning stepLength " << stepLength << ", with prev NL err = " << prevNLErr << " and new NL err = " << newNLErr << endl;
      _nlRes = newNLErr;
      return stepLength;
    }
  };
}

#endif
