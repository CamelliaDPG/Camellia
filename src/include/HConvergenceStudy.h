#ifndef DPG_H_CONVERGENCE_STUDY
#define DPG_H_CONVERGENCE_STUDY

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
 *  HConvergenceStudy.h
 *
 *  Created by Nathan Roberts on 7/21/11.
 *
 */

#include "Mesh.h"
#include "ExactSolution.h"

using namespace std;

class HConvergenceStudy {
  Teuchos::RCP<ExactSolution> _exactSolution;
  Teuchos::RCP<BilinearForm> _bilinearForm;
  Teuchos::RCP<RHS> _rhs;  
  Teuchos::RCP<BC> _bc;
  Teuchos::RCP<DPGInnerProduct> _ip;
  int _H1Order, _minLogElements, _maxLogElements, _pToAdd;
  vector< Teuchos::RCP<Solution> > _solutions;
  Teuchos::RCP<Solution> _fineZeroSolution;
  bool _randomRefinements;
  bool _useTriangles;
  bool _useHybrid;
  void randomlyRefine(Teuchos::RCP<Mesh> mesh);
  bool _reportRelativeErrors;
public:
  HConvergenceStudy(Teuchos::RCP<ExactSolution> exactSolution,
                    Teuchos::RCP<BilinearForm> bilinearForm,
                    Teuchos::RCP<RHS> rhs,
                    Teuchos::RCP<BC> bc,
                    Teuchos::RCP<DPGInnerProduct> ip,
                    int minLogElements, int maxLogElements, int H1Order, int pToAdd,
                    bool randomRefinements=false, bool useTriangles=false, bool useHybrid=false);
  void setReportRelativeErrors(bool reportRelativeErrors);
  void solve(const FieldContainer<double> &quadPoints);
  Teuchos::RCP<Solution> getSolution(int logElements); // logElements: a number between minLogElements and maxLogElements
  void writeToFiles(const string & filePathPrefix, int trialID, int traceID = -1);
};

#endif