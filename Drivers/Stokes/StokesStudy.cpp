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

#include "MultiOrderStudy.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

using namespace std;

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  // TODO: figure out the right thing to do here...
  // may want to modify argc and argv before we make the following call:
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  //rank=mpiSession.getRank();
  //numProcs=mpiSession.getNProc();
#else
#endif
  int polyOrder = 2, minLogElements = 0, maxLogElements = 4;
  
  int pToAdd = 2; // for optimal test function approximation
  bool useTriangles = false;
  bool useOptimalNorm = true; 
  StokesManufacturedSolution::StokesFormulationType formulationType=StokesManufacturedSolution::ORIGINAL_CONFORMING;
  string formulationTypeStr = "original conforming";
  
  bool useMultiOrder = false;
  
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
    if (formulationTypeStr == "vvp") {
      formulationType = StokesManufacturedSolution::VVP_CONFORMING;
    } else if (formulationTypeStr == "nonConforming") {
      formulationType = StokesManufacturedSolution::ORIGINAL_NON_CONFORMING;
    } else {
      formulationType = StokesManufacturedSolution::ORIGINAL_CONFORMING;
      formulationTypeStr = "original conforming";
    }
    
  } else if (argc == 4) {
    polyOrder = atoi(argv[1]);
    minLogElements = atoi(argv[2]);
    maxLogElements = atoi(argv[3]);
  } else if (argc == 5) {
    string normChoice = argv[1];
    if (normChoice == "math") {
      useOptimalNorm = false; // otherwise, use math
    }
    polyOrder = atoi(argv[2]);
    minLogElements = atoi(argv[3]);
    maxLogElements = atoi(argv[4]);
  } else if (argc == 6) {
    string normChoice = argv[1];
    if (normChoice == "math") {
      useOptimalNorm = false; // otherwise, use math
    }
    formulationTypeStr = argv[2];
    if (formulationTypeStr == "vvp") {
      formulationType = StokesManufacturedSolution::VVP_CONFORMING;
    } else if (formulationTypeStr == "nonConforming") {
      formulationType = StokesManufacturedSolution::ORIGINAL_NON_CONFORMING;
    }
    polyOrder = atoi(argv[3]);
    minLogElements = atoi(argv[4]);
    maxLogElements = atoi(argv[5]);
  } else if (argc == 7) {
    string normChoice = argv[1];
    if (normChoice == "math") {
      useOptimalNorm = false;
    }
    formulationTypeStr = argv[2];
    if (formulationTypeStr == "vvp") {
      formulationType = StokesManufacturedSolution::VVP_CONFORMING;
    } else if (formulationTypeStr == "nonConforming") {
      formulationType = StokesManufacturedSolution::ORIGINAL_NON_CONFORMING;
    }
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
  
  Teuchos::RCP<StokesManufacturedSolution> mySolution = 
  Teuchos::rcp( new StokesManufacturedSolution(StokesManufacturedSolution::EXPONENTIAL, 
                                               polyOrder, formulationType) );
  
  int pressureID = ( formulationType == StokesManufacturedSolution::VVP_CONFORMING ) ? StokesVVPBilinearForm::P : StokesBilinearForm::P;
  bool singlePointBCs = ! mySolution->bc()->imposeZeroMeanConstraint(pressureID);
  
  cout << "formulationType = " << formulationTypeStr                  << "\n";
  cout << "useTriangles = "    << (useTriangles   ? "true" : "false") << "\n";
  cout << "useOptimalNorm = "  << (useOptimalNorm ? "true" : "false") << "\n";
  cout << "singlePointBCs = "  << (singlePointBCs ? "true" : "false") << "\n";
  
  mySolution->setUseSinglePointBCForP(singlePointBCs);
  Teuchos::RCP<DPGInnerProduct> ip;
  if (useOptimalNorm) {
    ip = Teuchos::rcp( new OptimalInnerProduct( mySolution->bilinearForm() ) );
  } else {
    ip = Teuchos::rcp( new    MathInnerProduct( mySolution->bilinearForm() ) );
  }
  
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
  if (formulationType == StokesManufacturedSolution::VVP_CONFORMING) {
    u1_trialID = StokesVVPBilinearForm::U1;
    u2_trialID = StokesVVPBilinearForm::U2;
    p_trialID = StokesVVPBilinearForm::P;
  } else {
    u1_trialID = StokesBilinearForm::U1;
    u2_trialID = StokesBilinearForm::U2;
    p_trialID =  StokesBilinearForm::P;    
  }
  
  if ( !useMultiOrder ) {
    HConvergenceStudy study(mySolution,
                            mySolution->bilinearForm(),
                            mySolution->ExactSolution::rhs(),
                            mySolution->bc(), ip, 
                            minLogElements, maxLogElements, 
                            polyOrder+1, pToAdd, false, useTriangles, false);
    
    study.solve(quadPoints);
    
    ostringstream filePathPrefix;
    filePathPrefix << "stokes/u1_p" << polyOrder;
    
    study.writeToFiles(filePathPrefix.str(),u1_trialID);
    filePathPrefix.str("");
    filePathPrefix << "stokes/u2_p" << polyOrder;
    study.writeToFiles(filePathPrefix.str(),u2_trialID);
    
    filePathPrefix.str("");
    filePathPrefix << "stokes/pressure_p" << polyOrder;
    study.writeToFiles(filePathPrefix.str(),p_trialID);
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