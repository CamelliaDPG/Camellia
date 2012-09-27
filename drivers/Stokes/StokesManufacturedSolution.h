#ifndef STOKES_MANUFACTURED_SOLUTION
#define STOKES_MANUFACTURED_SOLUTION

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
 *  StokesManufacturedSolution.h
 *
 *  Created by Nathan Roberts on 7/21/11.
 *
 */

#include <Sacado.hpp>		// for FAD and RAD
#include "ExactSolution.h"
#include "BC.h"
#include "RHS.h"

class StokesManufacturedSolution : public ExactSolution, public RHS, public BC {
public:
  enum StokesFormulationType {
    ORIGINAL_NON_CONFORMING,
    ORIGINAL_CONFORMING,
    VVP_CONFORMING,
    VGP_CONFORMING,
    DDS_CONFORMING
  };
  enum StokesManufacturedSolutionType {
    POLYNOMIAL = 0,
    EXPONENTIAL,
    TRIGONOMETRIC
  };
protected:
  int _polyOrder;
  StokesManufacturedSolutionType _type;
  StokesFormulationType _formulationType;
  double _mu;
  bool _useSinglePointBCForP;
  
  void f_rhs(const FieldContainer<double> &physicalPoints, FieldContainer<double> &values, int vectorComponent);
public:
  StokesManufacturedSolution(StokesManufacturedSolutionType type, 
                             int polyOrder=-2, StokesFormulationType formulationType=ORIGINAL_NON_CONFORMING); // poly order here means that of phi -- -2 for non-polynomial types
  
  template <typename T> const T u1(T &x, T &y);  // in 2 dimensions; div(u) = d/dx (u1) + d/dy (u2) == 0 
  template <typename T> const T u2(T &x, T &y);  // in 2 dimensions
  template <typename T> const T  p(T &x, T &y);  // in 2 dimensions
  
  // ExactSolution
  virtual int H1Order(); // here it means the H1 order (i.e. polyOrder+1)
  virtual double solutionValue(int trialID,
                               FieldContainer<double> &physicalPoint);
  virtual double solutionValue(int trialID,
                               FieldContainer<double> &physicalPoint,
                               FieldContainer<double> &unitNormal);
  // RHS 
  virtual bool nonZeroRHS(int testVarID);
  virtual void rhs(int testVarID, const FieldContainer<double> &physicalPoints, FieldContainer<double> &values);
  // BC
  virtual bool bcsImposed(int varID); // returns true if there are any BCs anywhere imposed on varID
  virtual void imposeBC(int varID, FieldContainer<double> &physicalPoints, 
                        FieldContainer<double> &unitNormals,
                        FieldContainer<double> &dirichletValues,
                        FieldContainer<bool> &imposeHere);
  bool singlePointBC(int varID);
  bool imposeZeroMeanConstraint(int trialID);
  void setUseSinglePointBCForP(bool value);
  
  int pressureID();
};

#endif