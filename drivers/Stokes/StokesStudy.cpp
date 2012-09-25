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

#include "CGSolver.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

using namespace std;

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
   formulationTypeStr = {"original conforming"|"nonConforming"|"vvp"|"math"}
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
    formulationType = StokesManufacturedSolution::MATH_CONFORMING;
  } else if (formulationTypeStr == "vvp") {
    formulationType = StokesManufacturedSolution::VVP_CONFORMING;
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

  /////////////////////////// "MATH_CONFORMING" VERSION ///////////////////////
  VarFactory varFactory; 
  VarPtr tau1 = varFactory.testVar("\\tau_1", HDIV, StokesMathBilinearForm::Q_1);
  VarPtr tau2 = varFactory.testVar("\\tau_2", HDIV, StokesMathBilinearForm::Q_2);
  VarPtr v1 = varFactory.testVar("v_1", HGRAD, StokesMathBilinearForm::V_1);
  VarPtr v2 = varFactory.testVar("v_2", HGRAD, StokesMathBilinearForm::V_2);
  VarPtr q = varFactory.testVar("q", HGRAD, StokesMathBilinearForm::V_3);
  
  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1");
  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2");
  //  VarPtr uhatn = varFactory.fluxVar("\\widehat{u}_n");
  VarPtr sigma1n = varFactory.fluxVar("\\widehat{P - \\mu \\sigma_{1n}}");
  VarPtr sigma2n = varFactory.fluxVar("\\widehat{P - \\mu \\sigma_{2n}}");
  VarPtr u1 = varFactory.fieldVar("u_1");
  VarPtr u2 = varFactory.fieldVar("u_2");
  VarPtr sigma11 = varFactory.fieldVar("\\sigma_11");
  VarPtr sigma12 = varFactory.fieldVar("\\sigma_12");
  VarPtr sigma21 = varFactory.fieldVar("\\sigma_21");
  VarPtr sigma22 = varFactory.fieldVar("\\sigma_22");
  VarPtr p = varFactory.fieldVar("p");
  
  ///////////////////////////////////////////////////////////////////////////
  
  if (rank == 0) {
    cout << "pToAdd = " << pToAdd << endl;
    cout << "formulationType = " << formulationTypeStr                  << "\n";
    cout << "useTriangles = "    << (useTriangles   ? "true" : "false") << "\n";
    cout << "useOptimalNorm = "  << (useOptimalNorm ? "true" : "false") << "\n";
  }
  
  Teuchos::RCP<ExactSolution> mySolution;
  if (formulationType != StokesManufacturedSolution::MATH_CONFORMING) {
    Teuchos::RCP<StokesManufacturedSolution> stokesExact = Teuchos::rcp(
    new StokesManufacturedSolution(StokesManufacturedSolution::EXPONENTIAL, 
                                   polyOrder, formulationType));
    mySolution = stokesExact;
    
    int pressureID = stokesExact->pressureID();
    bool singlePointBCs = ! mySolution->bc()->imposeZeroMeanConstraint(pressureID);
    
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
    double mu = 1.0;
    
    BFPtr stokesBFMath = Teuchos::rcp( new BF(varFactory) );
    stokesBFMath->addTerm(u1,tau1->div());
    stokesBFMath->addTerm(sigma11,tau1->x()); // (sigma1, tau1)
    stokesBFMath->addTerm(sigma12,tau1->y());
    stokesBFMath->addTerm(-u1hat, tau1->dot_normal());
    
    // tau2 terms:
    stokesBFMath->addTerm(u2, tau2->div());
    stokesBFMath->addTerm(sigma21,tau2->x()); // (sigma2, tau2)
    stokesBFMath->addTerm(sigma22,tau2->y());
    stokesBFMath->addTerm(-u2hat, tau2->dot_normal());
    
    // v1:
    stokesBFMath->addTerm(mu * sigma11,v1->dx()); // (mu sigma1, grad v1) 
    stokesBFMath->addTerm(mu * sigma12,v1->dy());
    stokesBFMath->addTerm( - p, v1->dx() );
    stokesBFMath->addTerm( sigma1n, v1);
    
    // v2:
    stokesBFMath->addTerm(mu * sigma21,v2->dx()); // (mu sigma2, grad v2)
    stokesBFMath->addTerm(mu * sigma22,v2->dy());
    stokesBFMath->addTerm( -p, v2->dy());
    stokesBFMath->addTerm( sigma2n, v2);
    
    // q:
    stokesBFMath->addTerm(-u1,q->dx()); // (-u, grad q)
    stokesBFMath->addTerm(-u2,q->dy());
    stokesBFMath->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), q);
    
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new EntireBoundary );

    Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
    bc->addDirichlet(u1, entireBoundary, u1_exact);
    bc->addDirichlet(u2, entireBoundary, u2_exact);
    
    FunctionPtr dpdx = p_exact->dx();
    FunctionPtr du1dx = u1_exact->dx();
    FunctionPtr du1dy = u1_exact->dy();
    FunctionPtr du2dx = u2_exact->dx();
    FunctionPtr du2dy = u2_exact->dy();
    
    FunctionPtr f1 = p_exact->dx() - mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
    FunctionPtr f2 = p_exact->dy() - mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
    
    Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
    rhs->addTerm( f1 * v1 + f2 * v2 );
    mySolution = Teuchos::rcp( new ExactSolution(stokesBFMath, bc, rhs));
    mySolution->setSolutionFunction(u1, u1_exact);
    mySolution->setSolutionFunction(u1hat, u1_exact);
    mySolution->setSolutionFunction(u2, u2_exact);
    mySolution->setSolutionFunction(u2hat, u2_exact);
    mySolution->setSolutionFunction(p, p_exact);
    
    FunctionPtr sigma11_exact = u1_exact->dx();
    FunctionPtr sigma12_exact = u1_exact->dy();
    FunctionPtr sigma21_exact = u2_exact->dx();
    FunctionPtr sigma22_exact = u2_exact->dy();
    
    FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
//    FunctionPtr sigma1n_exact = sigma11_exact * n->x() + sigma12_exact * n->y();
//    FunctionPtr sigma2n_exact = sigma21_exact * n->x() + sigma22_exact * n->y();
    
    FunctionPtr sigma1n_exact = u1_exact->grad() * n;
    FunctionPtr sigma2n_exact = u2_exact->grad() * n;
    
    mySolution->setSolutionFunction(sigma11, sigma11_exact);
    mySolution->setSolutionFunction(sigma12, sigma12_exact);
    mySolution->setSolutionFunction(sigma21, sigma21_exact);
    mySolution->setSolutionFunction(sigma22, sigma22_exact);
    mySolution->setSolutionFunction(sigma1n, sigma1n_exact);
    mySolution->setSolutionFunction(sigma2n, sigma2n_exact);
  }
  
  Teuchos::RCP<DPGInnerProduct> ip;
  if (useOptimalNorm) {
    if (formulationType == StokesManufacturedSolution::MATH_CONFORMING ) {
      
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
        cout << "Stokes Math formulation: using quasi-optimal IP with beta=" << beta << endl;
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
  if (formulationType == StokesManufacturedSolution::MATH_CONFORMING) {
    u1_trialID = StokesMathBilinearForm::U1;
    u2_trialID = StokesMathBilinearForm::U2;
    p_trialID = StokesMathBilinearForm::P;
    u1_traceID = StokesMathBilinearForm::U1_HAT;
    u2_traceID = StokesMathBilinearForm::U2_HAT;
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
      if ( formulationType != StokesManufacturedSolution::MATH_CONFORMING) {
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
        primaryVariables.push_back(p->ID());
        cout << "******** Best Approximation comparison: ********\n";
        cout << study.TeXBestApproximationComparisonTable(primaryVariables);
        
        ostringstream filePathPrefix;
        filePathPrefix << "stokes/u1_p" << polyOrder;
        
        study.writeToFiles(filePathPrefix.str(),u1_trialID,u1_traceID);
        filePathPrefix.str("");
        filePathPrefix << "stokes/u2_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),u2_trialID,u2_traceID);
        
        filePathPrefix.str("");
        filePathPrefix << "stokes/pressure_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),p_trialID);
        
        filePathPrefix.str("");
        filePathPrefix << "stokes/sigma11_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),sigma11->ID());
        filePathPrefix.str("");
        filePathPrefix << "stokes/sigma12_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),sigma12->ID());
        filePathPrefix.str("");
        filePathPrefix << "stokes/sigma21_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),sigma21->ID());
        filePathPrefix.str("");
        filePathPrefix << "stokes/sigma22_p" << polyOrder;
        study.writeToFiles(filePathPrefix.str(),sigma22->ID());
        
        cout << study.convergenceDataMATLAB(u1->ID());
        
        cout << study.convergenceDataMATLAB(u2->ID());
        
        cout << study.convergenceDataMATLAB(p->ID());
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