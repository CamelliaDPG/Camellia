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

/*
 *  PoissonExactSolution.cpp
 *
 *  Created by Nathan Roberts on 7/7/11.
 *
 */

#include "ExpFunction.h"
#include "PoissonBilinearForm.h"
#include "PoissonExactSolution.h"
#include "SpatialFilter.h"
#include "TrigFunctions.h"

PoissonExactSolution::PoissonExactSolution(PoissonExactSolutionType type, int polyOrder, bool useConformingTraces) {
  // poly order here means that of phi
  _polyOrder = polyOrder;
  _type = type;
  _bf = PoissonBilinearForm::poissonBilinearForm(useConformingTraces);
  this->_bilinearForm = _bf;

  FunctionPtr phi_exact = phi();

  VarFactoryPtr vf = _bf->varFactory();
  VarPtr psi_hat_n = vf->fluxVar(PoissonBilinearForm::S_PSI_HAT_N);
  VarPtr phi_hat = vf->traceVar(PoissonBilinearForm::S_PHI_HAT);
  VarPtr phi = vf->fieldVar(PoissonBilinearForm::S_PHI);
  VarPtr psi_1 = vf->fieldVar(PoissonBilinearForm::S_PSI_1);
  VarPtr psi_2 = vf->fieldVar(PoissonBilinearForm::S_PSI_2);

  VarPtr q = vf->testVar(PoissonBilinearForm::S_Q, HGRAD);

  FunctionPtr psi_exact = phi_exact->grad();
  FunctionPtr n = Function::normal();

  this->setSolutionFunction(phi, phi_exact);
  this->setSolutionFunction(psi_1, psi_exact->x());
  this->setSolutionFunction(psi_2, psi_exact->y());
  this->setSolutionFunction(phi_hat, phi_exact);
  this->setSolutionFunction(psi_hat_n, psi_exact * n);

  SpatialFilterPtr wholeBoundary = SpatialFilter::allSpace();

  _rhs = RHS::rhs();
  FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy();
  _rhs->addTerm(f * q);

  setUseSinglePointBCForPHI(false, -1); // sets _bc
}

int PoissonExactSolution::H1Order() {
  return _polyOrder + 1;
}

FunctionPtr PoissonExactSolution::phi() {
  // simple solution choice: let phi = (x + 2y)^_polyOrder
  FunctionPtr f;
  double integral;
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr sin_x = Teuchos::rcp( new Sin_x );
  FunctionPtr cos_y = Teuchos::rcp( new Cos_y );
  FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
  switch (_type) {
    case POLYNOMIAL:
    {
      f = Function::constant(1);
      if (_polyOrder == 0) {
        f = Function::constant(0);
        integral = 0;
        break;
      }
      for (int i=0; i<_polyOrder; i++) {
        f = f * (x + 2 * y);
      }
      double two_to_power = 1; // power = _polyOrder + 2
      double three_to_power = 1;
      for (int i=0; i<_polyOrder+2; i++) {
        two_to_power *= 2.0;
        three_to_power *= 3.0;
      }
      integral = (three_to_power - two_to_power - 1) / (2 * (_polyOrder + 2) * (_polyOrder + 1) );
    }
      break;
    case TRIGONOMETRIC:
      f = sin_x * y + 3.0 * cos_y * x * x;
      integral = 0;
      break;
    case EXPONENTIAL:
    {
      integral = exp(1.0);
      FunctionPtr exp_x = Teuchos::rcp( new Exp_x );
      f = exp_x;
    }
  }
  f = f - integral;
  return f;
}

std::vector<double> PoissonExactSolution::getPointForBCImposition() {
  std::vector<double> point(2);
  point[0] = 1.0;
  point[1] = 1.0;
  return point;
}

void PoissonExactSolution::setUseSinglePointBCForPHI(bool useSinglePointBCForPhi, IndexType vertexIndexForZeroValue) {
  FunctionPtr phi_exact = phi();

  VarFactoryPtr vf = _bf->varFactory();
  VarPtr psi_hat_n = vf->fluxVar(PoissonBilinearForm::S_PSI_HAT_N);
  VarPtr q = vf->testVar(PoissonBilinearForm::S_Q, HGRAD);

  VarPtr phi = vf->fieldVar(PoissonBilinearForm::S_PHI);

  SpatialFilterPtr wholeBoundary = SpatialFilter::allSpace();

  FunctionPtr n = Function::normal();
  FunctionPtr psi_n_exact = phi_exact->grad() * n;

  _bc = BC::bc();
  _bc->addDirichlet(psi_hat_n, wholeBoundary, psi_n_exact);
  if (!useSinglePointBCForPhi) {
    _bc->addZeroMeanConstraint(phi);

  } else {
    std::vector<double> point = getPointForBCImposition();
    double value = Function::evaluate(phi_exact, point[0], point[1]);
//    cout << "PoissonExactSolution: imposing phi = " << value << " at (" << point[0] << ", " << point[1] << ")\n";
    _bc->addSinglePointBC(phi->ID(), value, vertexIndexForZeroValue);
  }
}

Teuchos::RCP<ExactSolution<double>> PoissonExactSolution::poissonExactPolynomialSolution(int polyOrder, bool useConformingTraces) {
  return Teuchos::rcp( new PoissonExactSolution(POLYNOMIAL,polyOrder,useConformingTraces));
}
