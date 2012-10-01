/*
 *  StokesStudy.cpp
 *
 *  Created by Nathan Roberts on 7/21/11.
 *
 */

// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

#include "StokesStudy.h"
#include "StokesManufacturedSolution.h"
#include "StokesBilinearForm.h"

#include "HConvergenceStudy.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "StokesManufacturedSolution.h"
#include "StokesVVPBilinearForm.h"
#include "StokesMathBilinearForm.h"

#include "InnerProductScratchPad.h"

#include "../MultiOrderStudy/MultiOrderStudy.h"

#include "PreviousSolutionFunction.h"

#include "CGSolver.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

using namespace std;

// StokesFormulation class: a prototype which I hope to reuse elsewhere once it's ready
class StokesFormulation {
public:
  virtual BFPtr bf() = 0;
  virtual RHSPtr rhs(FunctionPtr f1, FunctionPtr f2) = 0;
  // so far, only have support for BCs defined on the entire boundary (i.e. no outflow type conditions)
  virtual BCPtr bc(FunctionPtr u1, FunctionPtr u2, SpatialFilterPtr entireBoundary) = 0;
  virtual IPPtr graphNorm() = 0;
  virtual Teuchos::RCP<ExactSolution> exactSolution(FunctionPtr u1, FunctionPtr u2, FunctionPtr p, 
                                                    SpatialFilterPtr entireBoundary) = 0;
};

class VGPStokesFormulation : public StokesFormulation {
  VarFactory varFactory;
  VarPtr tau1, tau2, q, p, sigma21, u1hat, u2hat; // for VGP
  VarPtr v1, v2, sigma1n, sigma2n, u1, u2, sigma11, sigma12, sigma22; // for both, but with different meanings  
  
  BFPtr _bf;
  IPPtr _graphNorm;
  double _mu;
  
public:
  VGPStokesFormulation(double mu) {
    _mu = mu;
    
    v1 = varFactory.testVar("v_1", HGRAD, StokesMathBilinearForm::V_1);
    v2 = varFactory.testVar("v_2", HGRAD, StokesMathBilinearForm::V_2);
    tau1 = varFactory.testVar("\\tau_1", HDIV, StokesMathBilinearForm::Q_1);
    tau2 = varFactory.testVar("\\tau_2", HDIV, StokesMathBilinearForm::Q_2);
    q = varFactory.testVar("q", HGRAD, StokesMathBilinearForm::V_3);
    
    u1hat = varFactory.traceVar("\\widehat{u}_1");
    u2hat = varFactory.traceVar("\\widehat{u}_2");
    sigma1n = varFactory.fluxVar("\\widehat{P - \\mu \\sigma_{1n}}");
    sigma2n = varFactory.fluxVar("\\widehat{P - \\mu \\sigma_{2n}}");
    u1 = varFactory.fieldVar("u_1");
    u2 = varFactory.fieldVar("u_2");
    sigma11 = varFactory.fieldVar("\\sigma_{11}");
    sigma12 = varFactory.fieldVar("\\sigma_{12}");
    sigma21 = varFactory.fieldVar("\\sigma_{21}");
    sigma22 = varFactory.fieldVar("\\sigma_{22}");
    p = varFactory.fieldVar("p");
    
    // construct bilinear form:
    _bf = Teuchos::rcp( new BF(varFactory) );
    // tau1 terms:
    _bf->addTerm(u1,tau1->div());
    _bf->addTerm(sigma11,tau1->x()); // (sigma1, tau1)
    _bf->addTerm(sigma12,tau1->y());
    _bf->addTerm(-u1hat, tau1->dot_normal());
    
    // tau2 terms:
    _bf->addTerm(u2, tau2->div());
    _bf->addTerm(sigma21,tau2->x()); // (sigma2, tau2)
    _bf->addTerm(sigma22,tau2->y());
    _bf->addTerm(-u2hat, tau2->dot_normal());
    
    // v1:
    _bf->addTerm(mu * sigma11,v1->dx()); // (mu sigma1, grad v1) 
    _bf->addTerm(mu * sigma12,v1->dy());
    _bf->addTerm( - p, v1->dx() );
    _bf->addTerm( sigma1n, v1);
    
    // v2:
    _bf->addTerm(mu * sigma21,v2->dx()); // (mu sigma2, grad v2)
    _bf->addTerm(mu * sigma22,v2->dy());
    _bf->addTerm( -p, v2->dy());
    _bf->addTerm( sigma2n, v2);
    
    // q:
    _bf->addTerm(-u1,q->dx()); // (-u, grad q)
    _bf->addTerm(-u2,q->dy());
    _bf->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), q);

    _graphNorm = Teuchos::rcp( new IP );
    _graphNorm->addTerm( mu * v1->dx() + tau1->x() ); // sigma11
    _graphNorm->addTerm( mu * v1->dy() + tau1->y() ); // sigma12
    _graphNorm->addTerm( mu * v2->dx() + tau2->x() ); // sigma21
    _graphNorm->addTerm( mu * v2->dy() + tau2->y() ); // sigma22
    _graphNorm->addTerm( v1->dx() + v2->dy() );     // pressure
    _graphNorm->addTerm( tau1->div() - q->dx() );    // u1
    _graphNorm->addTerm( tau2->div() - q->dy() );    // u2
    
    // L^2 terms:
    _graphNorm->addTerm( v1 );
    _graphNorm->addTerm( v2 );
    _graphNorm->addTerm( q );
    _graphNorm->addTerm( tau1 );
    _graphNorm->addTerm( tau2 );
  }
  BFPtr bf() {
    return _bf;
  }
  IPPtr graphNorm() {
    return _graphNorm;
  }
  RHSPtr rhs(FunctionPtr f1, FunctionPtr f2) {
    Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
    rhs->addTerm( f1 * v1 + f2 * v2 );
    return rhs;
  }
  BCPtr bc(FunctionPtr u1_fxn, FunctionPtr u2_fxn, SpatialFilterPtr entireBoundary) {
    Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
    bc->addDirichlet(u1hat, entireBoundary, u1_fxn);
    bc->addDirichlet(u2hat, entireBoundary, u2_fxn);
    bc->addZeroMeanConstraint(p);
    return bc;
  }
  Teuchos::RCP<ExactSolution> exactSolution(FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact,
                                            SpatialFilterPtr entireBoundary) {
    FunctionPtr f1 = p_exact->dx() - _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
    FunctionPtr f2 = p_exact->dy() - _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
    
    BCPtr bc = this->bc(u1_exact, u2_exact, entireBoundary);
    
    Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
    rhs->addTerm( f1 * v1 + f2 * v2 );
    Teuchos::RCP<ExactSolution> mySolution = Teuchos::rcp( new ExactSolution(_bf, bc, rhs) );
    mySolution->setSolutionFunction(u1, u1_exact);
    mySolution->setSolutionFunction(u2, u2_exact);
    
    mySolution->setSolutionFunction(p, p_exact);
    
    mySolution->setSolutionFunction(sigma11, u1_exact->dx());
    mySolution->setSolutionFunction(sigma12, u1_exact->dy());
    mySolution->setSolutionFunction(sigma21, u2_exact->dx());
    mySolution->setSolutionFunction(sigma22, u2_exact->dy());
    
    return mySolution;
  }
};

class EntireBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x+1.0) < tol) || (abs(x-1.0) < tol);
    bool yMatch = (abs(y+1.0) < tol) || (abs(y-1.0) < tol);
    return xMatch || yMatch;
  }
};

class Exp_x : public SimpleFunction {
public:
  double value(double x, double y) {
    return exp(x);
  }
  FunctionPtr dx() {
    return Teuchos::rcp( new Exp_x );
  }
  FunctionPtr dy() {
    return Function::zero();
  }
};

class Y : public SimpleFunction {
public:
  double value(double x, double y) {
    return y;
  }
  FunctionPtr dx() {
    return Function::zero();
  }
  FunctionPtr dy() {
    return Teuchos::rcp( new ConstantScalarFunction(1.0) );
  }
};

class Cos_y : public SimpleFunction {
  double value(double x, double y);
  FunctionPtr dx();
  FunctionPtr dy();
};

class Sin_y : public SimpleFunction {
  double value(double x, double y) {
    return sin(y);
  }
  FunctionPtr dx() {
    return Function::zero();
  }
  FunctionPtr dy() {
    return Teuchos::rcp( new Cos_y );
  }
};

double Cos_y::value(double x, double y) {
  return cos(y);
}
FunctionPtr Cos_y::dx() {
  return Function::zero();
}
FunctionPtr Cos_y::dy() {
  FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
  return - sin_y;
}

void parseArgs(int argc, char *argv[], int &polyOrder, int &minLogElements, int &maxLogElements,
               StokesManufacturedSolution::StokesFormulationType &formulationType,
               bool &useTriangles, bool &useMultiOrder, bool &useOptimalNorm, string &formulationTypeStr) {
  polyOrder = 2; minLogElements = 0; maxLogElements = 4;
  
  // set up defaults:
  useTriangles = false;
  useOptimalNorm = true; 
  formulationType=StokesManufacturedSolution::ORIGINAL_CONFORMING;
  formulationTypeStr = "original conforming";
  string multiOrderStudyType = "multiOrderQuad";
  
  useMultiOrder = false;
  
  string normChoice = "opt";
  
  /* Usage:
   Multi-Order, naive norm:
     StokesStudy "multiOrder{Tri|Quad}" formulationTypeStr
   Single Order, original conforming, optimal norm, quad:
     StokesStudy polyOrder minLogElements maxLogElements
   Single Order, original conforming, quad:
   StokesStudy normChoice polyOrder minLogElements maxLogElements
   Single Order, quad:
     StokesStudy normChoice formulationTypeStr polyOrder minLogElements maxLogElements
   Single Order, quad:
     StokesStudy normChoice formulationTypeStr polyOrder minLogElements maxLogElements {"quad"|"tri"}
   
   where:
   formulationTypeStr = {"original conforming"|"nonConforming"|"vvp"|"math"|"dev"}
   normChoice = {"opt"|"naive"}
   
   */
  
  
  if (argc == 3) {
    string multiOrderStudyType = argv[1];
    formulationTypeStr = argv[2];
    if (multiOrderStudyType == "multiOrderTri") {
      useTriangles = true;
    } else {
      useTriangles = false;
    }
    useMultiOrder = true;
    useOptimalNorm = false; // using naive norm for paper
    polyOrder = 1;
  } else if (argc == 4) {
    polyOrder = atoi(argv[1]);
    minLogElements = atoi(argv[2]);
    maxLogElements = atoi(argv[3]);
  } else if (argc == 5) {
    normChoice = argv[1];
    polyOrder = atoi(argv[2]);
    minLogElements = atoi(argv[3]);
    maxLogElements = atoi(argv[4]);
  } else if (argc == 6) {
    normChoice = argv[1];

    formulationTypeStr = argv[2];
    polyOrder = atoi(argv[3]);
    minLogElements = atoi(argv[4]);
    maxLogElements = atoi(argv[5]);
  } else if (argc == 7) {
    normChoice = argv[1];
    formulationTypeStr = argv[2];
    polyOrder = atoi(argv[3]);
    minLogElements = atoi(argv[4]);
    maxLogElements = atoi(argv[5]);
    string elementTypeStr = argv[6];
    if (elementTypeStr == "tri") {
      useTriangles = true;
    } else if (elementTypeStr == "quad") {
      useTriangles = false;
    } // otherwise, just use whatever was defined above
  }
  if (normChoice == "naive") {
    useOptimalNorm = false; // otherwise, use naive
  }
  if (formulationTypeStr == "math") {
    formulationType = StokesManufacturedSolution::VGP_CONFORMING;
  } else if (formulationTypeStr == "vvp") {
    formulationType = StokesManufacturedSolution::VVP_CONFORMING;
  } else if (formulationTypeStr == "dev") {
    formulationType = StokesManufacturedSolution::DDS_CONFORMING;
  } else if (formulationTypeStr == "nonConforming") {
    formulationType = StokesManufacturedSolution::ORIGINAL_NON_CONFORMING;
  } else {
    formulationType = StokesManufacturedSolution::ORIGINAL_CONFORMING;
    formulationTypeStr = "original conforming";
  }

}

int main(int argc, char *argv[]) {
  int rank = 0, numProcs = 1;
#ifdef HAVE_MPI
  // TODO: figure out the right thing to do here...
  // may want to modify argc and argv before we make the following call:
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  rank=mpiSession.getRank();
  numProcs=mpiSession.getNProc();
#else
#endif
  int pToAdd = 1; // for optimal test function approximation
  bool useCG = false;
  double cgTol = 1e-8;
  int cgMaxIt = 400000;
  Teuchos::RCP<Solver> cgSolver = Teuchos::rcp( new CGSolver(cgMaxIt, cgTol) );

  // parse args:
  int polyOrder, minLogElements, maxLogElements;
  bool useTriangles, useOptimalNorm, useMultiOrder;
  
  // NOTE: "normalize h factors" is probably a kooky idea, and I haven't even managed yet to get it to improve the conditioning
  //       (i.e. I'm probably not introducing the right factors to normalize given the various pullbacks involved)
  bool normalizeHFactors = false; // divide by h wherever h factors enter: a test to see if we can improve conditioning...
  
  
  StokesManufacturedSolution::StokesFormulationType formulationType;
  string formulationTypeStr;
  parseArgs(argc, argv, polyOrder, minLogElements, maxLogElements, formulationType, useTriangles,
            useMultiOrder, useOptimalNorm, formulationTypeStr);

  /////////////////////////// "VGP_CONFORMING" VERSION ///////////////////////
  VarFactory varFactory; 
  VarPtr tau1, tau2, q, p, sigma21, u1hat, u2hat; // for VGP
  VarPtr tau11, tau12, tau22, u1n1hat, utnhat, u2n2hat; // for DDS
  VarPtr v1, v2, sigma1n, sigma2n, u1, u2, sigma11, sigma12, sigma22; // for both, but with different meanings  
  
  if (formulationType == StokesManufacturedSolution::VGP_CONFORMING ) {
    v1 = varFactory.testVar("v_1", HGRAD, StokesMathBilinearForm::V_1);
    v2 = varFactory.testVar("v_2", HGRAD, StokesMathBilinearForm::V_2);
    tau1 = varFactory.testVar("\\tau_1", HDIV, StokesMathBilinearForm::Q_1);
    tau2 = varFactory.testVar("\\tau_2", HDIV, StokesMathBilinearForm::Q_2);
    q = varFactory.testVar("q", HGRAD, StokesMathBilinearForm::V_3);

    u1hat = varFactory.traceVar("\\widehat{u}_1");
    u2hat = varFactory.traceVar("\\widehat{u}_2");
    sigma1n = varFactory.fluxVar("\\widehat{P - \\mu \\sigma_{1n}}");
    sigma2n = varFactory.fluxVar("\\widehat{P - \\mu \\sigma_{2n}}");
    u1 = varFactory.fieldVar("u_1");
    u2 = varFactory.fieldVar("u_2");
    sigma11 = varFactory.fieldVar("\\sigma_{11}");
    sigma12 = varFactory.fieldVar("\\sigma_{12}");
    sigma21 = varFactory.fieldVar("\\sigma_{21}");
    sigma22 = varFactory.fieldVar("\\sigma_{22}");
    p = varFactory.fieldVar("p");
  } else {
    v1 = varFactory.testVar("v_1", HGRAD);
    v2 = varFactory.testVar("v_2", HGRAD);
    
    tau11 = varFactory.testVar("\\tau_{11}", HGRAD);
    tau12 = varFactory.testVar("\\tau_{12}", HGRAD);
    tau22 = varFactory.testVar("\\tau_{22}", HGRAD);
    
    sigma1n = varFactory.fluxVar("\\widehat{\\sigma}_{1n}");
    sigma2n = varFactory.fluxVar("\\widehat{\\sigma}_{2n}");
    u1n1hat = varFactory.fluxVar("\\widehat{2 \\mu u_1 n_1}");
    utnhat  = varFactory.fluxVar("\\widehat{\\mu {u_2 \\choose u_1} \\cdot n}");
    u2n2hat = varFactory.fluxVar("\\widehat{2 \\mu u_2 n_2}");

    u1 = varFactory.fieldVar("u_1");
    u2 = varFactory.fieldVar("u_2");
    sigma11 = varFactory.fieldVar("\\sigma_{11}");
    sigma12 = varFactory.fieldVar("\\sigma_{12}");
    sigma22 = varFactory.fieldVar("\\sigma_{22}");
  }

  
  ///////////////////////////////////////////////////////////////////////////
  
  if (rank == 0) {
    cout << "pToAdd = " << pToAdd << endl;
    cout << "formulationType = " << formulationTypeStr                  << "\n";
    cout << "useTriangles = "    << (useTriangles   ? "true" : "false") << "\n";
    cout << "useOptimalNorm = "  << (useOptimalNorm ? "true" : "false") << "\n";
  }
  
  double mu = 1.0;
  VGPStokesFormulation vgpForm(mu);
  
  Teuchos::RCP<ExactSolution> mySolution;
  if ((formulationType != StokesManufacturedSolution::VGP_CONFORMING) 
      && (formulationType != StokesManufacturedSolution::DDS_CONFORMING) ) {
    Teuchos::RCP<StokesManufacturedSolution> stokesExact = Teuchos::rcp(
    new StokesManufacturedSolution(StokesManufacturedSolution::EXPONENTIAL, 
                                   polyOrder, formulationType));
    mySolution = stokesExact;
    
    int pressureID = stokesExact->pressureID();
    bool singlePointBCs = ! stokesExact->bc()->imposeZeroMeanConstraint(pressureID);
    
    if (rank == 0) {
      cout << "singlePointBCs = "  << (singlePointBCs ? "true" : "false") << "\n";
    }
    
    stokesExact->setUseSinglePointBCForP(singlePointBCs);
  } else {
    // trying out the new ExactSolution features:
    FunctionPtr cos_y = Teuchos::rcp( new Cos_y );
    FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
    FunctionPtr exp_x = Teuchos::rcp( new Exp_x );
    FunctionPtr y = Teuchos::rcp( new Y );
    FunctionPtr u1_exact = - exp_x * ( y * cos_y + sin_y );
    FunctionPtr u2_exact = exp_x * y * sin_y;
    FunctionPtr p_exact = 2.0 * exp_x * sin_y;
    
    BFPtr stokesBF = Teuchos::rcp( new BF(varFactory) );
    BCPtr bc;
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new EntireBoundary );

    
    switch (formulationType) {
      case StokesManufacturedSolution::DDS_CONFORMING:
        // v1 terms:
        stokesBF->addTerm(-sigma11, v1->dx()); // (sigma1, grad v1)
        stokesBF->addTerm(-sigma12, v1->dy());
        stokesBF->addTerm(sigma1n, v1);
        // v2 terms:
        stokesBF->addTerm(-sigma12, v2->dx()); // (sigma2, grad v2)
        stokesBF->addTerm(-sigma22, v2->dy());
        stokesBF->addTerm(sigma2n, v2);
        // tau11 terms:
        stokesBF->addTerm(sigma11,tau11);
        stokesBF->addTerm(2 * mu * u1,tau11->dx());
        stokesBF->addTerm(-u1n1hat,tau11);
        // tau12 terms:
        stokesBF->addTerm(sigma12,tau12);
        stokesBF->addTerm(mu * u2,tau12->dx());
        stokesBF->addTerm(mu * u1,tau12->dy());
        stokesBF->addTerm(-utnhat,tau12);
        // tau22 terms:
        stokesBF->addTerm(sigma22,tau22);
        stokesBF->addTerm(2 * mu * u2,tau22->dy());
        stokesBF->addTerm(-u2n2hat,tau22);
        
        {
          FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
          Teuchos::RCP<BCEasy> bcEasy = Teuchos::rcp( new BCEasy );;
          bcEasy->addDirichlet(u1n1hat, entireBoundary, u1_exact * n->x());
          bcEasy->addDirichlet(utnhat,  entireBoundary, u2_exact * n->x() + u1_exact * n->y());
          bcEasy->addDirichlet(u2n2hat, entireBoundary, u2_exact * n->y());
          bc = bcEasy;
        }
        stokesBF->printTrialTestInteractions();
        break;
        
      case StokesManufacturedSolution::VGP_CONFORMING:
        stokesBF = vgpForm.bf();
        bc = vgpForm.bc(u1_exact,u2_exact,entireBoundary);
        
        break;
        
      default:
        break;
    }
    
    FunctionPtr f1 = p_exact->dx() - mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
    FunctionPtr f2 = p_exact->dy() - mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
    
    Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
    rhs->addTerm( f1 * v1 + f2 * v2 );
    mySolution = Teuchos::rcp( new ExactSolution(stokesBF, bc, rhs));
    mySolution->setSolutionFunction(u1, u1_exact);
    mySolution->setSolutionFunction(u2, u2_exact);
    
    if (formulationType != StokesManufacturedSolution::DDS_CONFORMING) // then p is defined...
      mySolution->setSolutionFunction(p, p_exact);
    
    if (formulationType == StokesManufacturedSolution::VGP_CONFORMING) {
      mySolution = vgpForm.exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
    } else if (formulationType == StokesManufacturedSolution::DDS_CONFORMING) {
//      FunctionPtr pTerm = 0.5 * sigma11 + 0.5 * sigma12; // p = tr(sigma) / n
      mySolution->setSolutionFunction(sigma11, p_exact + 2 * mu * u1_exact->dx());
      mySolution->setSolutionFunction(sigma12, mu * u1_exact->dy() + mu * u2_exact->dx());
      mySolution->setSolutionFunction(sigma22, p_exact + 2 * mu * u2_exact->dy());
    }
  }
  
  Teuchos::RCP<DPGInnerProduct> ip;
  if (useOptimalNorm) {
    if (formulationType == StokesManufacturedSolution::VGP_CONFORMING ) {
      ip = vgpForm.graphNorm();
      /*
      double mu = 1.0;
      
      IPPtr qoptIP = Teuchos::rcp(new IP());
      FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
      FunctionPtr h = Teuchos::rcp( new hFunction() );
      FunctionPtr h_inv = one / h;
      
      double beta = 1.0;
      if (normalizeHFactors) {
        cout << "WARNING: normalizeHFactors needs fixing: need to consider carefully the effects of Piola transform, etc.\n";
//        qoptIP->addTerm( mu * h_inv * v1->dx() + h_inv * q1->x() ); // sigma11
//        qoptIP->addTerm( mu * h_inv * v1->dy() + h_inv * q1->y() ); // sigma12
//        qoptIP->addTerm( mu * h_inv * v2->dx() + h_inv * q2->x() ); // sigma21
//        qoptIP->addTerm( mu * h_inv * v2->dy() + h_inv * q2->y() ); // sigma22
//        qoptIP->addTerm( h_inv * v1->dx() + h_inv * v2->dy() );     // pressure
//        qoptIP->addTerm( 1.0 * q1->div() - h_inv * v3->dx() );    // u1
//        qoptIP->addTerm( 1.0 * q2->div() - h_inv * v3->dy() );    // u2
        qoptIP->addTerm( mu * v1->dx() + tau1->x() ); // sigma11
        qoptIP->addTerm( mu * v1->dy() + tau1->y() ); // sigma12
        qoptIP->addTerm( mu * v2->dx() + tau2->x() ); // sigma21
        qoptIP->addTerm( mu * v2->dy() + tau2->y() ); // sigma22
        qoptIP->addTerm( v1->dx() + v2->dy() );     // pressure
        qoptIP->addTerm( h * tau1->div() - q->dx() );    // u1
        qoptIP->addTerm( h * tau2->div() - q->dy() );    // u2
      } else {
        qoptIP->addTerm( mu * v1->dx() + tau1->x() ); // sigma11
        qoptIP->addTerm( mu * v1->dy() + tau1->y() ); // sigma12
        qoptIP->addTerm( mu * v2->dx() + tau2->x() ); // sigma21
        qoptIP->addTerm( mu * v2->dy() + tau2->y() ); // sigma22
        qoptIP->addTerm( v1->dx() + v2->dy() );     // pressure
        qoptIP->addTerm( tau1->div() - q->dx() );    // u1
        qoptIP->addTerm( tau2->div() - q->dy() );    // u2
      }
      qoptIP->addTerm( sqrt(beta) * v1 );
      qoptIP->addTerm( sqrt(beta) * v2 );
      qoptIP->addTerm( sqrt(beta) * q );
      qoptIP->addTerm( sqrt(beta) * tau1 );
      qoptIP->addTerm( sqrt(beta) * tau2 );
      
      ip = qoptIP;
      
      if (rank==0)
        cout << "Stokes VGP formulation: using quasi-optimal IP with beta=" << beta << endl;*/
    } else if (formulationType==StokesManufacturedSolution::DDS_CONFORMING) {
      IPPtr qoptIP = Teuchos::rcp(new IP());
      FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
      FunctionPtr h = Teuchos::rcp( new hFunction() );
      FunctionPtr h_inv = one / h;
      
      double mu = 1.0;
            
      qoptIP->addTerm( tau11 - v1->dx() );            // sigma11
      qoptIP->addTerm( tau12 - v2->dx() - v1->dy() ); // sigma12
      qoptIP->addTerm( tau22 - v2->dy() );            // sigma22
      qoptIP->addTerm( 2 * mu * tau11->dx() + mu * tau12->dy() );    // u1
      qoptIP->addTerm( 2 * mu * tau22->dy() + mu * tau12->dx() );    // u2
      qoptIP->addTerm( v1 );
      qoptIP->addTerm( v2 );
      qoptIP->addTerm( tau11 );
      qoptIP->addTerm( tau12 );
      qoptIP->addTerm( tau22 );
      
      ip = qoptIP;

    } else {
      ip = Teuchos::rcp( new OptimalInnerProduct( mySolution->bilinearForm() ) );
    }
    
    
  } else {
    ip = Teuchos::rcp( new   MathInnerProduct( mySolution->bilinearForm() ) );
  }
  
  if (rank==0) 
    ip->printInteractions();
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;
  
  int u1_trialID, u2_trialID, p_trialID;
  int u1_traceID, u2_traceID;
  if (formulationType == StokesManufacturedSolution::VGP_CONFORMING) {
    u1_trialID = StokesMathBilinearForm::U1;
    u2_trialID = StokesMathBilinearForm::U2;
    p_trialID = StokesMathBilinearForm::P;
    u1_traceID = StokesMathBilinearForm::U1_HAT;
    u2_traceID = StokesMathBilinearForm::U2_HAT;
  } else if (formulationType == StokesManufacturedSolution::DDS_CONFORMING) {
    u1_trialID = u1->ID();
    u2_trialID = u2->ID();
    u1_traceID = -1; // no velocity traces available in DDS formulation
    u2_traceID = -1;
  } else if (formulationType == StokesManufacturedSolution::VVP_CONFORMING) {
    u1_trialID = StokesVVPBilinearForm::U1;
    u2_trialID = StokesVVPBilinearForm::U2;
    p_trialID = StokesVVPBilinearForm::P;
    u1_traceID = -1; // no velocity traces available in VVP formulation
    u2_traceID = -1;
  } else {
    u1_trialID = StokesBilinearForm::U1;
    u2_trialID = StokesBilinearForm::U2;
    p_trialID =  StokesBilinearForm::P;    
    u1_traceID = StokesBilinearForm::U1_HAT;
    u2_traceID = StokesBilinearForm::U2_HAT;
  }
  
  if ( !useMultiOrder ) {
    HConvergenceStudy study(mySolution,
                            mySolution->bilinearForm(),
                            mySolution->ExactSolution::rhs(),
                            mySolution->bc(), ip, 
                            minLogElements, maxLogElements, 
                            polyOrder+1, pToAdd, false, useTriangles, false);
    study.setReportRelativeErrors(true);
    if (useCG) study.setSolver(cgSolver);
    
    study.solve(quadPoints);
    
    if (rank == 0) {
      if (( formulationType != StokesManufacturedSolution::VGP_CONFORMING) 
      &&  ( formulationType != StokesManufacturedSolution::DDS_CONFORMING) )
      {
        ostringstream filePathPrefix;
        filePathPrefix << "stokes/u1_p" << polyOrder;
        
        study.writeToFiles(filePathPrefix.str(),u1_trialID,u1_traceID);
        filePathPrefix.str("");
        filePathPrefix << "stokes/u2_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),u2_trialID,u2_traceID);
        
        filePathPrefix.str("");
        filePathPrefix << "stokes/pressure_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),p_trialID);
      } else {
        cout << study.TeXErrorRateTable();
        vector<int> primaryVariables;
        primaryVariables.push_back(u1->ID());
        primaryVariables.push_back(u2->ID());
        if ( formulationType !=  StokesManufacturedSolution::DDS_CONFORMING) // no pressure for DDS
          primaryVariables.push_back(p->ID());
        cout << "******** Best Approximation comparison: ********\n";
        cout << study.TeXBestApproximationComparisonTable(primaryVariables);
        
        ostringstream filePathPrefix;
        filePathPrefix << "stokes/u1_p" << polyOrder;
        
        study.writeToFiles(filePathPrefix.str(),u1_trialID,u1_traceID);
        filePathPrefix.str("");
        filePathPrefix << "stokes/u2_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),u2_trialID,u2_traceID);
        
        if ( formulationType !=  StokesManufacturedSolution::DDS_CONFORMING) { // no pressure for DDS
          filePathPrefix.str("");
          filePathPrefix << "stokes/pressure_p" << polyOrder;
          study.writeToFiles(filePathPrefix.str(),p_trialID);
        }
        
        filePathPrefix.str("");
        filePathPrefix << "stokes/sigma11_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),sigma11->ID());
        filePathPrefix.str("");
        filePathPrefix << "stokes/sigma12_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),sigma12->ID());
        if ( formulationType !=  StokesManufacturedSolution::DDS_CONFORMING) { // no sigma21 for DDS
          filePathPrefix.str("");
          filePathPrefix << "stokes/sigma21_p" << polyOrder;
          study.writeToFiles(filePathPrefix.str(),sigma21->ID());
        }
        filePathPrefix.str("");
        filePathPrefix << "stokes/sigma22_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),sigma22->ID());
        
        cout << study.convergenceDataMATLAB(u1->ID());
        
        cout << study.convergenceDataMATLAB(u2->ID());
        
        if ( formulationType !=  StokesManufacturedSolution::DDS_CONFORMING) { // no pressure for DDS
          cout << study.convergenceDataMATLAB(p->ID());
        }
      }
    }
  } else {
    cout << "Generating mixed-order 16x16 mesh" << endl;
    Teuchos::RCP<Mesh> mesh = MultiOrderStudy::makeMultiOrderMesh16x16(quadPoints,
                                                                       mySolution->bilinearForm(),
                                                                       polyOrder+1, pToAdd,
                                                                       useTriangles);
    
    Solution solution(mesh, mySolution->bc(), mySolution->ExactSolution::rhs(), ip);
    solution.solve();
    int cubDegree = 15; // for error computations
    double  pError = mySolution->L2NormOfError(solution, p_trialID,  cubDegree);
    double u1Error = mySolution->L2NormOfError(solution, u1_trialID, cubDegree);
    double u2Error = mySolution->L2NormOfError(solution, u2_trialID, cubDegree);
    
    string meshType = (useTriangles) ? "triangular" : "quad";
    
    cout << "Multi-order, 16x16 " << meshType << " mesh, pressure error: "  <<  pError << endl;
    cout << "Multi-order, 16x16 " << meshType << " mesh, u1 error: "        << u1Error << endl;
    cout << "Multi-order, 16x16 " << meshType << " mesh, u2 error: "        << u2Error << endl;
  }
}