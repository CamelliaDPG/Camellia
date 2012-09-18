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
 *  PoissonExactSolution.cpp
 *
 *  Created by Nathan Roberts on 7/7/11.
 *
 */

#include "PoissonBilinearForm.h"
#include "PoissonExactSolution.h"
#include "PoissonBilinearFormConformingTraces.h"

typedef Sacado::Fad::SFad<double,2> F2; // FAD with # of ind. vars fixed at 2
typedef Sacado::Fad::SFad< Sacado::Fad::SFad<double,2>, 2> F2_2; // same thing, but nested so we can take 2 derivatives
//typedef Sacado::Fad::DFad< Sacado::Fad::DFad<double> > F2_2;

PoissonExactSolution::PoissonExactSolution(PoissonExactSolutionType type, int polyOrder, bool useConformingTraces) {
  // poly order here means that of phi
  _polyOrder = polyOrder;
  _type = type;
  _bc = Teuchos::rcp(this,false); // don't let the RCP own the memory
  _rhs = Teuchos::rcp(this,false);
  if ( ! useConformingTraces ) {
    _bilinearForm = Teuchos::rcp(new PoissonBilinearForm());
  } else {
    _bilinearForm = Teuchos::rcp(new PoissonBilinearFormConformingTraces());
  }
  _useSinglePointBCForPHI = false;
}

int PoissonExactSolution::H1Order() {
  // polynomialOrder here means the H1 order (i.e. polyOrder+1)
  return _polyOrder + 1;
}

template <typename T> const T PoissonExactSolution::phi(T &x, T &y) {
  // simple solution choice: let phi = (x + 2y)^_polyOrder
  T t;
  T integral; // over (0,1)^2 -- want to subtract this to make the average 0
  switch (_type) {
    case POLYNOMIAL:
    {
      t = 1;
      if (_polyOrder == 0) {
        t = 0;
        integral = 0;
        break;
      }
      for (int i=0; i<_polyOrder; i++) {
        t *= x + 2 * y;
      }
      T two_to_power = 1; // power = _polyOrder + 2
      T three_to_power = 1;
      for (int i=0; i<_polyOrder+2; i++) {
        two_to_power *= 2.0;
        three_to_power *= 3.0;
      }
      integral = (three_to_power - two_to_power - 1) / (2 * (_polyOrder + 2) * (_polyOrder + 1) );
    }
      break;
    case TRIGONOMETRIC:
      t = sin(x) * y + 3.0 * cos(y) * x * x;
      integral = 0;
      break;
    case EXPONENTIAL:
    {
      integral = 4.18597023381589 / 4.0; // solution average as reported by Mathematica
      t = exp(x * sin(y) );
    }
  }
  t -= integral;
  return t;
}

void PoissonExactSolution::setUseSinglePointBCForPHI(bool value) { 
  _useSinglePointBCForPHI = value; 
}

double PoissonExactSolution::solutionValue(int trialID,
                                           FieldContainer<double> &physicalPoint) {
  
  double x = physicalPoint(0);
  double y = physicalPoint(1);
  F2 sx(2,0,x), sy(2,1,y), sphi; // s for Sacado 
  //PoissonExactSolutionPolynomial<F2> myExact(_polyOrder);
	sphi = phi(sx,sy);
  
  switch(trialID) {
    case PoissonBilinearForm::PHI:
    case PoissonBilinearForm::PHI_HAT:
      return sphi.val();
    case PoissonBilinearForm::PSI_1:
      return sphi.dx(0); // PSI_1 == d/dx (phi)
    case PoissonBilinearForm::PSI_2:
      return sphi.dx(1); // PSI_2 == d/dy (phi)
    case PoissonBilinearForm::PSI_HAT_N:
      TEUCHOS_TEST_FOR_EXCEPTION( trialID == PoissonBilinearForm::PSI_HAT_N,
                         std::invalid_argument,
                         "for fluxes, you must call solutionValue with unitNormal argument.");
  }
  TEUCHOS_TEST_FOR_EXCEPTION( true,
                     std::invalid_argument,
                     "solutionValues called with unknown trialID.");
  return 0.0;
}

double PoissonExactSolution::solutionValue(int trialID,
                                           FieldContainer<double> &physicalPoint,
                                           FieldContainer<double> &unitNormal) {
  if ( trialID != PoissonBilinearForm::PSI_HAT_N ) {
    return solutionValue(trialID,physicalPoint);
  }
  // otherwise, get PSI_1 and PSI_2, and the unit normal
  double psi1 = solutionValue(PoissonBilinearForm::PSI_1,physicalPoint);
  double psi2 = solutionValue(PoissonBilinearForm::PSI_2,physicalPoint);
  double n1 = unitNormal(0);
  double n2 = unitNormal(1);
  return psi1*n1 + psi2*n2;
}

/********** RHS implementation **********/
bool PoissonExactSolution::nonZeroRHS(int testVarID) {
  if (testVarID == PoissonBilinearForm::Q_1) { // the vector test function, zero RHS
    return false;
  } else if (testVarID == PoissonBilinearForm::V_1) {
    return true;
  } else {
    return false; // could throw an exception here
  }
}

void PoissonExactSolution::rhs(int testVarID, const FieldContainer<double> &physicalPoints, FieldContainer<double> &values) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  if (testVarID == PoissonBilinearForm::V_1) {
    // then the value is (d^2/dx^2 + d^2/dy^2) ( phi )
    values.resize(numCells,numPoints);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = physicalPoints(cellIndex,ptIndex,0);
        double y = physicalPoints(cellIndex,ptIndex,1);
        F2_2 sx(2,0,x), sy(2,1,y), sphi; // s for Sacado 
        sx.val() = F2(2,0,x);
        sy.val() = F2(2,1,y);
        sphi = phi(sx,sy);/*
        double val = sphi.val().val();
        double dx = sphi.dx(0).val();
        double dy = sphi.dx(1).val();
        double dx2 = sphi.dx(0).dx(0);
        double dy2 = sphi.dx(1).dx(1);*/
        values(cellIndex,ptIndex) = sphi.dx(0).dx(0) + sphi.dx(1).dx(1);
      }
    }
  }
}

/***************** BC Implementation *****************/
bool PoissonExactSolution::bcsImposed(int varID){
  // returns true if there are any BCs anywhere imposed on varID
  return (varID == PoissonBilinearForm::PSI_HAT_N);
  
//  if ( ! _useSinglePointBCForPHI ) {
//    // then we impose BCs everywhere for both trace and flux:
//    return (varID == PoissonBilinearForm::PHI_HAT) || (varID == PoissonBilinearForm::PSI_HAT_N);
//  } else {
//    // otherwise, we just impose on PSI_HAT_N here, and then return true for PHI in singletonBC
//    return (varID == PoissonBilinearForm::PSI_HAT_N);
//  }  
}

bool PoissonExactSolution::singlePointBC(int varID) {
  if ( ! _useSinglePointBCForPHI ) {
    return false;
  } else {
    return (varID==PoissonBilinearForm::PHI);
  }
}

bool PoissonExactSolution::imposeZeroMeanConstraint(int trialID) {
  if ( _useSinglePointBCForPHI ) {
    return false;
  } else {
    return (trialID==PoissonBilinearForm::PHI);
  }
}


void PoissonExactSolution::imposeBC(int varID, FieldContainer<double> &physicalPoints,
                               FieldContainer<double> &unitNormals,
                               FieldContainer<double> &dirichletValues,
                               FieldContainer<bool> &imposeHere) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  
  TEUCHOS_TEST_FOR_EXCEPTION( ( spaceDim != 2  ),
                     std::invalid_argument,
                     "PoissonBCLinear expects spaceDim==2.");  
  
  TEUCHOS_TEST_FOR_EXCEPTION( ( dirichletValues.dimension(0) != numCells ) 
                     || ( dirichletValues.dimension(1) != numPoints ) 
                     || ( dirichletValues.rank() != 2  ),
                     std::invalid_argument,
                     "dirichletValues dimensions should be (numCells,numPoints).");
  TEUCHOS_TEST_FOR_EXCEPTION( ( imposeHere.dimension(0) != numCells ) 
                     || ( imposeHere.dimension(1) != numPoints ) 
                     || ( imposeHere.rank() != 2  ),
                     std::invalid_argument,
                     "imposeHere dimensions should be (numCells,numPoints).");
  Teuchos::Array<int> pointDimensions;
  pointDimensions.push_back(2);
  if ((varID == PoissonBilinearForm::PHI_HAT) || (varID == PoissonBilinearForm::PHI)) {
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        FieldContainer<double> physicalPoint(pointDimensions,
                                             &physicalPoints(cellIndex,ptIndex,0));
        dirichletValues(cellIndex,ptIndex) = solutionValue(PoissonBilinearForm::PHI_HAT,
                                                           physicalPoint);
        imposeHere(cellIndex,ptIndex) = true; // for now, just impose everywhere...
      }
    }
  } else if (varID == PoissonBilinearForm::PSI_HAT_N) {
    // value will be (1 1) \cdot n, the outward normal
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        FieldContainer<double> physicalPoint(pointDimensions,
                                             &physicalPoints(cellIndex,ptIndex,0));
        FieldContainer<double> unitNormal(pointDimensions,
                                          &unitNormals(cellIndex,ptIndex,0));
        dirichletValues(cellIndex,ptIndex) = solutionValue(PoissonBilinearForm::PSI_HAT_N,
                                                           physicalPoint, unitNormal);
        imposeHere(cellIndex,ptIndex) = true; // for now, just impose everywhere...
      }
    }
  }
}