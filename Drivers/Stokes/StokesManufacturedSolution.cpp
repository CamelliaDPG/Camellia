
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
 *  StokesManufacturedSolution.cpp
 *
 *  Created by Nathan Roberts on 7/21/11.
 *
 */

#include "StokesManufacturedSolution.h"

#include "StokesBilinearForm.h"
#include "StokesBilinearFormConforming.h"
#include "StokesVVPBilinearForm.h"
#include "StokesManufacturedSolution.h"
#include "StokesMathBilinearForm.h"

typedef Sacado::Fad::SFad<double,2> F2; // FAD with # of ind. vars fixed at 2
typedef Sacado::Fad::SFad< Sacado::Fad::SFad<double,2>, 2> F2_2; // same thing, but nested so we can take 2 derivatives
//typedef Sacado::Fad::DFad< Sacado::Fad::DFad<double> > F2_2;

int StokesManufacturedSolution::pressureID() {
//  return ( _formulationType == VVP_CONFORMING) ? StokesVVPBilinearForm::P : StokesBilinearForm::P;
  if ( _formulationType == StokesManufacturedSolution::MATH_CONFORMING ) {
    return StokesMathBilinearForm::P;
  } else if ( _formulationType == StokesManufacturedSolution::VVP_CONFORMING ) {
    return StokesVVPBilinearForm::P;
  } else {
    return StokesBilinearForm::P;
  } 
}

StokesManufacturedSolution::StokesManufacturedSolution(StokesManufacturedSolutionType type, 
                                                       int polyOrder, StokesFormulationType formulationType) {
  // poly order here means that of u1, u2
  _polyOrder = polyOrder;
  _type = type;
  ExactSolution::_bc = Teuchos::rcp(this,false); // don't let the RCP own the memory
  _rhs = Teuchos::rcp(this,false);
  _mu = 1.0;
  _formulationType = formulationType;
  if ( _formulationType ==  ORIGINAL_NON_CONFORMING) {
    _bilinearForm = Teuchos::rcp(new StokesBilinearForm(_mu));
  } else if ( _formulationType == ORIGINAL_CONFORMING ) {
    _bilinearForm = Teuchos::rcp(new StokesBilinearFormConforming(_mu));
  } else if ( _formulationType == VVP_CONFORMING ) {
    _bilinearForm = Teuchos::rcp(new StokesVVPBilinearForm(_mu));
  } else if ( _formulationType == MATH_CONFORMING ) {
    _bilinearForm = Teuchos::rcp(new StokesMathBilinearForm(_mu));
  }
  if ( _formulationType == ORIGINAL_NON_CONFORMING ) {
    _useSinglePointBCForP = true;
  } else {
    _useSinglePointBCForP = false;
  }
}

void StokesManufacturedSolution::setUseSinglePointBCForP(bool value) {
  _useSinglePointBCForP = value;

  if (value && (imposeZeroMeanConstraint(pressureID()) ) ) {
    cout << "warning: imposing zero mean constraint as well as single-point BC for Stokes pressure.\n";
  }
}

int StokesManufacturedSolution::H1Order() {
  // polynomialOrder here means the H1 order (i.e. polyOrder+1)
  return _polyOrder + 1;
}

template <typename T> const T StokesManufacturedSolution::u1(T &x, T &y) {
  // simple solution choice: let phi = (x + 2y)^_polyOrder
  T t;
  switch (_type) {
    case POLYNOMIAL:
      t = -2;
      for (int i=0; i<_polyOrder; i++) {
        t *= x + 2*y; //x + y;
      }
      break;
    case EXPONENTIAL:
      t = -exp(x) * ( y * cos(y) + sin(y) );
      break;
    default:
      break;
  } 
  return t;
}

template <typename T> const T StokesManufacturedSolution::u2(T &x, T &y) {
  // simple solution choice: let phi = (x + 2y)^_polyOrder
  T t;
  switch (_type) {
    case POLYNOMIAL:
      t = 1;
      for (int i=0; i<_polyOrder; i++) {
        t *= x + 2*y;
      }
      break;
    case EXPONENTIAL:
      t = exp(x) * y * sin(y);
      break;
    default:
      break;
  } 
  return t;
}


template <typename T> const T StokesManufacturedSolution::p(T &x, T &y) {
  T t;
  switch (_type) {
    case POLYNOMIAL:
      t = 1;
      for (int i=0; i<_polyOrder; i++) {
        t *= x + y;
      }
      break;
    case EXPONENTIAL:
      t = 2 * _mu * exp(x) * sin(y);
      break;
    default:
      break;
  } 
  return t;
}

double StokesManufacturedSolution::solutionValue(int trialID,
                                                 FieldContainer<double> &physicalPoint) {
  double x = physicalPoint(0);
  double y = physicalPoint(1);
  F2 sx(2,0,x), sy(2,1,y), sp, su1, su2; // s for Sacado 
  //StokesManufacturedSolutionPolynomial<F2> myExact(_polyOrder);
	sp = p(sx,sy);
  su1 = u1(sx,sy);
  su2 = u2(sx,sy);
  
    if (_formulationType == VVP_CONFORMING) {
      switch(trialID) {
        case StokesVVPBilinearForm::U1:
          return su1.val();
          break;
        case StokesVVPBilinearForm::U2:
          return su2.val();
          break;
        case StokesVVPBilinearForm::P_HAT:
        case StokesVVPBilinearForm::P:
          return sp.val();
          break;
        case StokesVVPBilinearForm::OMEGA:
          return ( su2.dx(0) - su1.dx(1) ); // OMEGA =  ( d/dx (u2) - d/dy (u1) )  [NOTE different omega definition for VVP formulation, a bit awkward--but I'm following James Lai's paper here, and my own in the other...]
          break;
        case StokesVVPBilinearForm::U_N_HAT:
        case StokesVVPBilinearForm::U_CROSS_N_HAT:
          TEST_FOR_EXCEPTION( true,
                             std::invalid_argument,
                             "for fluxes, you must call solutionValue with unitNormal argument.");
          break;
      }
      TEST_FOR_EXCEPTION( true,
                         std::invalid_argument,
                         "solutionValues called with unknown trialID.");
    } else if ( (_formulationType == ORIGINAL_CONFORMING) || (_formulationType == ORIGINAL_NON_CONFORMING) ) {
      switch(trialID) {
        case StokesBilinearForm::U1:
        case StokesBilinearForm::U1_HAT:
          return su1.val();
          break;
        case StokesBilinearForm::U2:
        case StokesBilinearForm::U2_HAT:
          return su2.val();
          break;
        case StokesBilinearForm::P:
          return sp.val();
          break;
        case StokesBilinearForm::SIGMA_11:
          return 2.0 * _mu * su1.dx(0) -sp.val(); // SIGMA_11 == 2 mu d/dx (u1) - P
          break;
        case StokesBilinearForm::SIGMA_21:
          return _mu * ( su1.dx(1) + su2.dx(0) ); // SIGMA_21 == mu (d/dy (u1) + d/dx (u2) ) 
          break;
        case StokesBilinearForm::SIGMA_22:
          return 2.0 * _mu * su2.dx(1) - sp.val(); // SIGMA_22 == 2 mu d/dy (u2) - P
          break;
        case StokesBilinearForm::OMEGA:
          return 0.5 * ( su1.dx(1) - su2.dx(0) ); // OMEGA =  1/2 (d/dy (u1) - d/dx (u2) ) 
          break;
        case StokesBilinearForm::U_N_HAT:
        case StokesBilinearForm::SIGMA1_N_HAT:
        case StokesBilinearForm::SIGMA2_N_HAT:
          TEST_FOR_EXCEPTION( true,
                             std::invalid_argument,
                             "for fluxes, you must call solutionValue with unitNormal argument.");
          break;
      }
      TEST_FOR_EXCEPTION( true,
                     std::invalid_argument,
                     "solutionValues called with unknown trialID.");
    } else if (_formulationType == MATH_CONFORMING) {
      switch(trialID) {
        case StokesMathBilinearForm::U1:
        case StokesMathBilinearForm::U1_HAT:
          return su1.val();
          break;
        case StokesMathBilinearForm::U2:
        case StokesMathBilinearForm::U2_HAT:
          return su2.val();
          break;
        case StokesMathBilinearForm::P:
          return sp.val();
          break;
        case StokesMathBilinearForm::SIGMA_11:
          return su1.dx(0); // SIGMA_11 == d/dx (u1)
          break;
        case StokesMathBilinearForm::SIGMA_12:
          return su1.dx(1); // SIGMA_12 == d/dy (u1)
          break;
        case StokesMathBilinearForm::SIGMA_21:
          return su2.dx(0); // SIGMA_21 == d/dx (u2)
          break;
        case StokesMathBilinearForm::SIGMA_22:
          return su2.dx(1); // SIGMA_22 == d/dy (u2)
          break;
        case StokesMathBilinearForm::SIGMA1_N_HAT:
        case StokesMathBilinearForm::SIGMA2_N_HAT:
          TEST_FOR_EXCEPTION( true,
                             std::invalid_argument,
                             "for fluxes, you must call solutionValue with unitNormal argument.");
          break;
      }
    }
  return 0.0;
}

double StokesManufacturedSolution::solutionValue(int trialID,
                                                 FieldContainer<double> &physicalPoint,
                                                 FieldContainer<double> &unitNormal) {
  if (_formulationType == MATH_CONFORMING) {
    if (   ( trialID != StokesMathBilinearForm::SIGMA1_N_HAT )
        && ( trialID != StokesMathBilinearForm::SIGMA2_N_HAT ) )
    {
      return solutionValue(trialID,physicalPoint);
    }
    
    double n1 = unitNormal(0);
    double n2 = unitNormal(1);
    
    if ( trialID == StokesMathBilinearForm::SIGMA1_N_HAT ) {
      double sigma11 = solutionValue(StokesMathBilinearForm::SIGMA_11,physicalPoint);
      double sigma12 = solutionValue(StokesMathBilinearForm::SIGMA_12,physicalPoint);
      double p = solutionValue(StokesMathBilinearForm::P,physicalPoint);
      return p - (sigma11*n1 + sigma12*n2); 
    } else if ( trialID == StokesMathBilinearForm::SIGMA2_N_HAT ) {
      double sigma21 = solutionValue(StokesMathBilinearForm::SIGMA_21,physicalPoint);
      double sigma22 = solutionValue(StokesMathBilinearForm::SIGMA_22,physicalPoint);
      double p = solutionValue(StokesMathBilinearForm::P,physicalPoint);
      return p - (sigma21*n1 + sigma22*n2);
    }
  } else if (_formulationType == VVP_CONFORMING) {
    if (   ( trialID != StokesVVPBilinearForm::U_N_HAT )
        && ( trialID != StokesVVPBilinearForm::U_CROSS_N_HAT ) )
    {
      return solutionValue(trialID,physicalPoint);
    }
    
    double n1 = unitNormal(0);
    double n2 = unitNormal(1);
    
    if ( trialID == StokesVVPBilinearForm::U_N_HAT ) {
      double u1 = solutionValue(StokesVVPBilinearForm::U1,physicalPoint);
      double u2 = solutionValue(StokesVVPBilinearForm::U2,physicalPoint);
      return u1*n1 + u2*n2;
    } else if ( trialID == StokesVVPBilinearForm::U_CROSS_N_HAT ) {
      double u1 = solutionValue(StokesVVPBilinearForm::U1,physicalPoint);
      double u2 = solutionValue(StokesVVPBilinearForm::U2,physicalPoint);
      return u1*n2 - u2*n1;
    }
  } else {
    if (   ( trialID != StokesBilinearForm::U_N_HAT )
        && ( trialID != StokesBilinearForm::SIGMA1_N_HAT )
        && ( trialID != StokesBilinearForm::SIGMA2_N_HAT ) )
    {
      return solutionValue(trialID,physicalPoint);
    }
    
    double n1 = unitNormal(0);
    double n2 = unitNormal(1);
    
    if ( trialID == StokesBilinearForm::U_N_HAT ) {
      double u1 = solutionValue(StokesBilinearForm::U1,physicalPoint);
      double u2 = solutionValue(StokesBilinearForm::U2,physicalPoint);
      return u1*n1 + u2*n2;
    } else if ( trialID == StokesBilinearForm::SIGMA1_N_HAT ) {
      double sigma11 = solutionValue(StokesBilinearForm::SIGMA_11,physicalPoint);
      double sigma21 = solutionValue(StokesBilinearForm::SIGMA_21,physicalPoint);
      return sigma11*n1 + sigma21*n2;
    } else if ( trialID == StokesBilinearForm::SIGMA2_N_HAT ) {
      double sigma21 = solutionValue(StokesBilinearForm::SIGMA_21,physicalPoint);
      double sigma22 = solutionValue(StokesBilinearForm::SIGMA_22,physicalPoint);
      return sigma21*n1 + sigma22*n2;
    }
  }
  return 0; // unreachable statement.
  
}

/********** RHS implementation **********/
bool StokesManufacturedSolution::nonZeroRHS(int testVarID) {
  if (_formulationType == VVP_CONFORMING) {
    return (testVarID == StokesVVPBilinearForm::Q_1) ;
  } else if ( (_formulationType == ORIGINAL_NON_CONFORMING) || (_formulationType == ORIGINAL_CONFORMING) ) {
    return (testVarID == StokesBilinearForm::V_1) || (testVarID == StokesBilinearForm::V_2);
  } else if ( _formulationType == MATH_CONFORMING ) {
    return (testVarID == StokesMathBilinearForm::V_1) || (testVarID == StokesMathBilinearForm::V_2);
  }
  TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled formulation type.");
}

void StokesManufacturedSolution::f_rhs(const FieldContainer<double> &physicalPoints, FieldContainer<double> &values, int vectorComponent) { // -1 for both
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  if (vectorComponent==-1) {
    values.resize(numCells,numPoints,spaceDim);
  } else {
    values.resize(numCells,numPoints);
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double x = physicalPoints(cellIndex,ptIndex,0);
      double y = physicalPoints(cellIndex,ptIndex,1);
      F2_2 sx(2,0,x), sy(2,1,y), sp, su1, su2; // s for Sacado 
      sx.val() = F2(2,0,x);
      sy.val() = F2(2,1,y);
      sp = p(sx,sy);
      su1 = u1(sx,sy);
      su2 = u2(sx,sy);
      
      // (f1) is - div (sigma11,sigma21)
      F2 sigma11 = 2.0 * _mu * su1.dx(0) - sp.val();
      F2 sigma21 = _mu * ( su1.dx(1) + su2.dx(0) );
      double f1 = - sigma11.dx(0) - sigma21.dx(1);

      // (f2) is - div (sigma21,sigma22)
      F2 sigma22 = 2.0 * _mu * su2.dx(1) - sp.val();
      double f2 =  - sigma21.dx(0) - sigma22.dx(1);
      double val21 = sigma21.val(), val22 = sigma22.val();
      double du1dy = su1.dx(1).val(), du2dx = su2.dx(0).val();
      double pval = sp.val().val();
      if ( vectorComponent == -1 ) {
        values(cellIndex,ptIndex,0) = f1;
        values(cellIndex,ptIndex,1) = f2;
      } else if (vectorComponent == 0) {
        values(cellIndex,ptIndex) = f1;
      } else {
        values(cellIndex,ptIndex) = f2;
      }
    }
  }
}

void StokesManufacturedSolution::rhs(int testVarID, const FieldContainer<double> &physicalPoints, FieldContainer<double> &values) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  
  if (_formulationType == VVP_CONFORMING) {
    TEST_FOR_EXCEPTION( testVarID != StokesVVPBilinearForm::Q_1, std::invalid_argument,
                       "for Stokes VVP, rhs called for testVarID other than Q_1" );
    f_rhs(physicalPoints,values,-1); // -1 for both components
  } else {
    int v1, v2;
    if ( (_formulationType == ORIGINAL_NON_CONFORMING) || (_formulationType == ORIGINAL_CONFORMING) ) {
      v1 = StokesBilinearForm::V_1;
      v2 = StokesBilinearForm::V_2;
    } else {
      v1 = StokesMathBilinearForm::V_1;
      v2 = StokesMathBilinearForm::V_2;
    }
    TEST_FOR_EXCEPTION( (testVarID != v1) && (testVarID != v2), std::invalid_argument,
                       "for Stokes (non-VVP), rhs called for testVarID other than V_1 and V_2" );
    if ( testVarID == v1 ) {
      f_rhs(physicalPoints,values,0);
    } else {
      f_rhs(physicalPoints,values,1);
    }
  }
}

/***************** BC Implementation *****************/
bool StokesManufacturedSolution::bcsImposed(int varID){
  // returns true if there are any BCs anywhere imposed on varID
  int pressureID = this->pressureID();
  if ( (! _useSinglePointBCForP ) && (! imposeZeroMeanConstraint(pressureID) ) )  {
    // THIS IS A BIT WEIRD.  DO WE EVER ACTUALLY HIT THIS BLOCK?
    // (IT LOOKS LIKE THIS IS IN CASE WE AREN'T IMPOSING ANY CONSTRAINT ON THE PRESSURE,
    //  PROBABLY JUST SOMETHING I DID DURING DEBUGGING...)
    cout << "WARNING: StokesManufacturedSolution: no BC set for pressure, so imposing (over-determined) BCs on other variables.\n";
    if ( _formulationType == MATH_CONFORMING ) {
      // then we impose BCs everywhere for velocity, plus SIGMA1_N_HAT and SIGMA2_N_HAT:
      return (varID == StokesMathBilinearForm::U1_HAT)   || (varID == StokesMathBilinearForm::U2_HAT)
      || (varID == StokesMathBilinearForm::SIGMA1_N_HAT) || (varID == StokesMathBilinearForm::SIGMA2_N_HAT);
    } else if ( _formulationType == VVP_CONFORMING) {
      return (varID == StokesVVPBilinearForm::U_CROSS_N_HAT)
      || (varID == StokesVVPBilinearForm::U_N_HAT) || (varID == StokesVVPBilinearForm::P_HAT);
    } else { // original
      // then we impose BCs everywhere for velocity, plus SIGMA1_N_HAT and SIGMA2_N_HAT:
      return (varID == StokesBilinearForm::U1_HAT) || (varID == StokesBilinearForm::U2_HAT)
      || (varID == StokesBilinearForm::U_N_HAT) || (varID == StokesBilinearForm::SIGMA1_N_HAT)
      || (varID == StokesBilinearForm::SIGMA2_N_HAT);
    }
  } else {
    if ( _formulationType == MATH_CONFORMING ) {
      return (varID == StokesMathBilinearForm::U1_HAT) || (varID == StokesMathBilinearForm::U2_HAT);
    } else if ( _formulationType == VVP_CONFORMING) {
      return (varID == StokesVVPBilinearForm::U_CROSS_N_HAT) || (varID == StokesVVPBilinearForm::U_N_HAT);
    } else { // original
      return (varID == StokesBilinearForm::U1_HAT) || (varID == StokesBilinearForm::U2_HAT) || (varID == StokesBilinearForm::U_N_HAT);
    }
  }
}

bool StokesManufacturedSolution::singlePointBC(int varID) {
  if ( ! _useSinglePointBCForP )  {
    return false;
  } else {
    return (varID == pressureID() );
  }
}

bool StokesManufacturedSolution::imposeZeroMeanConstraint(int trialID) {
  if ( _useSinglePointBCForP ) {
    return false;
  } else {
    return (trialID==pressureID());
  }
}

void StokesManufacturedSolution::imposeBC(int varID, FieldContainer<double> &physicalPoints,
                                    FieldContainer<double> &unitNormals,
                                    FieldContainer<double> &dirichletValues,
                                    FieldContainer<bool> &imposeHere) {
  int numCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  
  TEST_FOR_EXCEPTION( ( spaceDim != 2  ),
                     std::invalid_argument,
                     "PoissonBCLinear expects spaceDim==2.");  
  
  TEST_FOR_EXCEPTION( ( dirichletValues.dimension(0) != numCells ) 
                     || ( dirichletValues.dimension(1) != numPoints ) 
                     || ( dirichletValues.rank() != 2  ),
                     std::invalid_argument,
                     "dirichletValues dimensions should be (numCells,numPoints).");
  TEST_FOR_EXCEPTION( ( imposeHere.dimension(0) != numCells ) 
                     || ( imposeHere.dimension(1) != numPoints ) 
                     || ( imposeHere.rank() != 2  ),
                     std::invalid_argument,
                     "imposeHere dimensions should be (numCells,numPoints).");
  
  imposeHere.initialize(false);
  Teuchos::Array<int> pointDimensions;
  pointDimensions.push_back(2);
  // TODO: add exceptions for varIDs that aren't supposed to have BCs imposed...
  
  // for now, we impose everywhere, and always pass in the unit normal
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      FieldContainer<double> physicalPoint(pointDimensions,
                                           &physicalPoints(cellIndex,ptIndex,0));
      if (unitNormals.rank() == 3) {
        FieldContainer<double> unitNormal(pointDimensions,
                                          &unitNormals(cellIndex,ptIndex,0));
        dirichletValues(cellIndex,ptIndex) = solutionValue(varID, physicalPoint, unitNormal);
      } else {
        // we'll assume we don't need a unit normal, then!!
        dirichletValues(cellIndex,ptIndex) = solutionValue(varID, physicalPoint);
      }
      imposeHere(cellIndex,ptIndex) = true;
    }
  }

  /*bool imposeWithNormal = false;
  if ( _formulationType == VVP_CONFORMING ) {
    imposeWithNormal = ((varID == StokesVVPBilinearForm::U1_HAT) || (varID == StokesBilinearForm::U2_HAT)
                        || (varID == StokesBilinearForm::P) )
  } else {
    imposeWithNormal = ((varID == StokesBilinearForm::U1_HAT) || (varID == StokesBilinearForm::U2_HAT)
                        || (varID == StokesBilinearForm::P) );
  }*/

  
  /*
  if  {
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        FieldContainer<double> physicalPoint(pointDimensions,
                                             &physicalPoints(cellIndex,ptIndex,0));
        dirichletValues(cellIndex,ptIndex) = solutionValue(varID, physicalPoint);
        if (varID == StokesBilinearForm::P) { // our singleton BC
          imposeHere(cellIndex,ptIndex) = true; // let caller decide which point to use
          // this is a bit awkward -- we need to impose at (0,0), for maximum fidelity to previous experiments
          // (danger is that the BC won't be imposed...)
          if (((abs(physicalPoint(0)) < 1e-12) && (abs(physicalPoint(1)) < 1e-12)) 
              || (numCells<=2) ) {// this is REALLY cheating: using knowledge that we only fail to have node at origin if we have 1 cell (or 2, for triangles)
            // stupid test: don't impose any pressure BC--how badly/how soon does this affect u solution?
            imposeHere(cellIndex,ptIndex) = true;
          } else {
            //imposeHere(cellIndex,ptIndex) = true;
            //cout << "not imposing pressure BC at (" << physicalPoint(0) << "," << physicalPoint(1) << ")" << endl;
          }
        } else {
          imposeHere(cellIndex,ptIndex) = true; // for now, just impose everywhere...
        }
      }
    }
  } else if ((varID == StokesBilinearForm::U_N_HAT) || (varID == StokesBilinearForm::SIGMA1_N_HAT)
             || (varID == StokesBilinearForm::SIGMA2_N_HAT) ) {
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        FieldContainer<double> physicalPoint(pointDimensions,
                                             &physicalPoints(cellIndex,ptIndex,0));
        FieldContainer<double> unitNormal(pointDimensions,
                                          &unitNormals(cellIndex,ptIndex,0));
        dirichletValues(cellIndex,ptIndex) = solutionValue(varID, physicalPoint, unitNormal);
        imposeHere(cellIndex,ptIndex) = true; // for now, just impose everywhere...
      }
    }
  }*/
}