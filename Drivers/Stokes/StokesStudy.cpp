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

#include "MultiOrderStudy.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

using namespace std;

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
   Multi-Order, math norm:
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
   normChoice = {"opt"|"math"}
   
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
    useOptimalNorm = false; // using math norm for paper
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
  if (normChoice == "math") {
    useOptimalNorm = false; // otherwise, use math
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

  // parse args:
  int polyOrder, minLogElements, maxLogElements;
  bool useTriangles, useOptimalNorm, useMultiOrder;
  StokesManufacturedSolution::StokesFormulationType formulationType;
  string formulationTypeStr;
  parseArgs(argc, argv, polyOrder, minLogElements, maxLogElements, formulationType, useTriangles,
            useMultiOrder, useOptimalNorm, formulationTypeStr);

  /////////////////////////// "MATH_CONFORMING" VERSION ///////////////////////
  VarFactory varFactory; 
  VarPtr q1 = varFactory.testVar("q_1", HDIV, StokesMathBilinearForm::Q_1);
  VarPtr q2 = varFactory.testVar("q_2", HDIV, StokesMathBilinearForm::Q_2);
  VarPtr v1 = varFactory.testVar("v_1", HGRAD, StokesMathBilinearForm::V_1);
  VarPtr v2 = varFactory.testVar("v_2", HGRAD, StokesMathBilinearForm::V_2);
  VarPtr v3 = varFactory.testVar("v_3", HGRAD, StokesMathBilinearForm::V_3);
  
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
  
  Teuchos::RCP<StokesManufacturedSolution> mySolution = 
  Teuchos::rcp( new StokesManufacturedSolution(StokesManufacturedSolution::EXPONENTIAL, 
                                               polyOrder, formulationType) );
  
  int pressureID = mySolution->pressureID();
  bool singlePointBCs = ! mySolution->bc()->imposeZeroMeanConstraint(pressureID);
  
  if (rank == 0) {
    cout << "pToAdd = " << pToAdd << endl;
    cout << "formulationType = " << formulationTypeStr                  << "\n";
    cout << "useTriangles = "    << (useTriangles   ? "true" : "false") << "\n";
    cout << "useOptimalNorm = "  << (useOptimalNorm ? "true" : "false") << "\n";
    cout << "singlePointBCs = "  << (singlePointBCs ? "true" : "false") << "\n";
  }
  
  mySolution->setUseSinglePointBCForP(singlePointBCs);
  Teuchos::RCP<DPGInnerProduct> ip;
  if (useOptimalNorm) {
    if (formulationType == StokesManufacturedSolution::MATH_CONFORMING ) {
      
      double mu = 1.0;
      
      IPPtr qoptIP = Teuchos::rcp(new IP());
      
      double beta = 1.0;
      qoptIP->addTerm( mu * v1->dx() + q1->x() ); // sigma11
      qoptIP->addTerm( mu * v1->dy() + q1->y() ); // sigma12
      qoptIP->addTerm( mu * v2->dx() + q2->x() ); // sigma21
      qoptIP->addTerm( mu * v2->dy() + q2->y() ); // sigma22
      qoptIP->addTerm( v1->dx() + v2->dy() );     // pressure
      qoptIP->addTerm( q1->div() - v3->dx() );    // u1
      qoptIP->addTerm( q2->div() - v3->dy() );    // u2
      qoptIP->addTerm( sqrt(beta) * v1 );
      qoptIP->addTerm( sqrt(beta) * v2 );
      qoptIP->addTerm( sqrt(beta) * v3 );
      qoptIP->addTerm( sqrt(beta) * q1 );
      qoptIP->addTerm( sqrt(beta) * q2 );
      
      ip = qoptIP;
      
      if (rank==0)
        cout << "Stokes Math formulation: using quasi-optimal IP with beta=" << beta << endl;
    } else {
      ip = Teuchos::rcp( new OptimalInnerProduct( mySolution->bilinearForm() ) );
    }
    
    
  } else {
    ip = Teuchos::rcp( new    MathInnerProduct( mySolution->bilinearForm() ) );
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