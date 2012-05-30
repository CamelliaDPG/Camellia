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
 *  PoissonExactSolutionLinear.cpp
 *
 *  Created by Nathan Roberts on 7/5/11.
 *
 */

#include "PoissonBCLinear.h"
#include "PoissonRHSLinear.h"
#include "PoissonBilinearForm.h"

#include "PoissonExactSolutionLinear.h"

PoissonExactSolutionLinear::PoissonExactSolutionLinear() {
  _bc = Teuchos::rcp(new PoissonBCLinear());
  _bilinearForm = Teuchos::rcp(new PoissonBilinearForm());
  _rhs = Teuchos::rcp(new PoissonRHSLinear());
  
}

double PoissonExactSolutionLinear::solutionValue(int trialID,
                                                FieldContainer<double> &physicalPoint) {
  double x = physicalPoint(0);
  double y = physicalPoint(1);
  switch(trialID) {
      // (psi, grad v1)_K + (psi_hat_n, v1)_dK
    case PoissonBilinearForm::PHI:
      return x + y;
      break;
    case PoissonBilinearForm::PSI_1:
      return 0.0;
      break;
    case PoissonBilinearForm::PSI_2:
      return 0.0;
      break;
    case PoissonBilinearForm::PSI_HAT_N:
      // TODO: throw exception: other solutionValue (with normal) should be called here
      break;
  }
  // TODO: throw exception
  return 0.0;
}

double PoissonExactSolutionLinear::solutionValue(int trialID,
                                                 FieldContainer<double> &physicalPoint,
                                                 FieldContainer<double> &unitNormal) {
  // TODO: implement this
  return 0.0;
}

int PoissonExactSolutionLinear::H1Order() {
  return 2; // order is the H1 order--since x + y is linear in HVOL, we're quadratic.
}