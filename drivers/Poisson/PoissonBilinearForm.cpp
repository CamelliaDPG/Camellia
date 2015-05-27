// @HEADER
//
// Copyright © 2014 Nathan V. Roberts. All Rights Reserved.
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

#include "PoissonBilinearForm.h"

// trial variable names:
const string PoissonBilinearForm::S_PHI = "\\phi";
const string PoissonBilinearForm::S_PSI_1 = "\\psi_1";
const string PoissonBilinearForm::S_PSI_2 = "\\psi_2";
const string PoissonBilinearForm::S_PHI_HAT = "\\hat{\\phi}";
const string PoissonBilinearForm::S_PSI_HAT_N ="\\hat{\\psi}_n";

// test variable names:
const string PoissonBilinearForm::S_Q = "q";
const string PoissonBilinearForm::S_TAU = "\\tau";

BFPtr PoissonBilinearForm::poissonBilinearForm(bool useConformingTraces)
{
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr tau = varFactory->testVar(S_TAU, HDIV);
  VarPtr q = varFactory->testVar(S_Q, HGRAD);

  Space phiHatSpace = useConformingTraces ? HGRAD : L2;
  VarPtr phi_hat = varFactory->traceVar(S_PHI_HAT, phiHatSpace);
  //  VarPtr phi_hat = varFactory->traceVar(S_GDAMinimumRuleTests_PHI_HAT, L2);
  //  cout << "WARNING: temporarily using L^2 discretization for \\widehat{\\phi}.\n";
  VarPtr psi_n = varFactory->fluxVar(S_PSI_HAT_N);

  VarPtr phi = varFactory->fieldVar(S_PHI, L2);
  VarPtr psi1 = varFactory->fieldVar(S_PSI_1, L2);
  VarPtr psi2 = varFactory->fieldVar(S_PSI_2, L2);

  BFPtr bf = Teuchos::rcp( new BF(varFactory) );

  bf->addTerm(phi, tau->div());
  bf->addTerm(psi1, tau->x());
  bf->addTerm(psi2, tau->y());
  bf->addTerm(-phi_hat, tau->dot_normal());

  bf->addTerm(-psi1, q->dx());
  bf->addTerm(-psi2, q->dy());
  bf->addTerm(psi_n, q);

  return bf;
}
