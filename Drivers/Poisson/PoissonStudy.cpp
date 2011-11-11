
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

/*
 *  PoissonStudy.cpp
 *
 *  Created by Nathan Roberts on 7/21/11.
 *
 */

#include "PoissonStudy.h"
#include "PoissonExactSolution.h"
#include "PoissonBilinearForm.h"

#include "HConvergenceStudy.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"

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
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
  cout << "rank: " << rank << endl;
  cout << "numProcs: " << numProcs << endl;
#else
#endif
  
  int polyOrder = 2, minLogElements = 0, maxLogElements = 4;
  bool useTriangles = false;
  bool useHybridMesh = false;
  bool useMultiOrder = false;
  if (argc == 2) {
    string multiOrderStudyType = argv[1];
    if (multiOrderStudyType == "multiOrderTri") {
      useTriangles = true;
    } else {
      useTriangles = false;
    }
    useMultiOrder = true;
    polyOrder = 1;
  } else if (argc == 4) {
    polyOrder = atoi(argv[1]);
    minLogElements = atoi(argv[2]);
    maxLogElements = atoi(argv[3]);
  } else if (argc == 5) {
    polyOrder = atoi(argv[1]);
    minLogElements = atoi(argv[2]);
    maxLogElements = atoi(argv[3]);
    string elementTypeStr = argv[4];
    if (elementTypeStr == "tri") {
      useTriangles = true;
    } else if (elementTypeStr == "quad") {
      useTriangles = false;
    } else if (elementTypeStr == "hybrid") {
      useTriangles = false;
      useHybridMesh = true;
    }
    // otherwise, just use whatever was defined above
  }
  
  int pToAdd = 2; // for tests
  bool useConformingTraces = true;
  Teuchos::RCP<PoissonExactSolution> mySolution = 
  Teuchos::rcp( new PoissonExactSolution(PoissonExactSolution::EXPONENTIAL, 
                                        polyOrder, useConformingTraces) );
  mySolution->setUseSinglePointBCForPHI(false); // impose zero-mean constraint
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct( mySolution->bilinearForm() ) );

  if (useTriangles) {
    cout << "using triangles\n";
  } else {
    cout << "using quads\n";
  }

  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;
  
  if ( !useMultiOrder ) {
    HConvergenceStudy study(mySolution,
                            mySolution->bilinearForm(),
                            mySolution->ExactSolution::rhs(),
                            mySolution->bc(), ip, 
                            minLogElements, maxLogElements, 
                            polyOrder+1, pToAdd, false, useTriangles, useHybridMesh); // use triangles...

    study.solve(quadPoints);
    
    ostringstream filePathPrefix;
    filePathPrefix << "poisson/phi_p" << polyOrder;
    
    study.writeToFiles(filePathPrefix.str(),PoissonBilinearForm::PHI);
    filePathPrefix.str("");
    filePathPrefix << "poisson/psi1_p" << polyOrder;
    study.writeToFiles(filePathPrefix.str(),PoissonBilinearForm::PSI_1);
    
    filePathPrefix.str("");
    filePathPrefix << "poisson/psi2_p" << polyOrder;
    study.writeToFiles(filePathPrefix.str(),PoissonBilinearForm::PSI_2);
  } else {
    cout << "Generating mixed-order 16x16 mesh" << endl;
    Teuchos::RCP<Mesh> mesh = MultiOrderStudy::makeMultiOrderMesh16x16(quadPoints,
                                                                       mySolution->bilinearForm(),
                                                                       polyOrder+1, pToAdd,
                                                                       useTriangles);
    
    Solution solution(mesh, mySolution->bc(), mySolution->ExactSolution::rhs(), ip);
    solution.solve();
    int cubDegree = 15; // for error computations
    double phiError =  mySolution->L2NormOfError(solution, PoissonBilinearForm::PHI,   cubDegree);
    double psi1Error = mySolution->L2NormOfError(solution, PoissonBilinearForm::PSI_1, cubDegree);
    double psi2Error = mySolution->L2NormOfError(solution, PoissonBilinearForm::PSI_2, cubDegree);
    
    string meshType = (useTriangles) ? "triangular" : "quad";
    
    cout << "Multi-order, 16x16 " << meshType << " mesh, phi error: " << phiError << endl;
    cout << "Multi-order, 16x16 " << meshType << " mesh, psi1 error: " << psi1Error << endl;
    cout << "Multi-order, 16x16 " << meshType << " mesh, psi2 error: " << psi2Error << endl;
  }
}