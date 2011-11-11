#ifndef DPG_EXACT_SOLUTION
#define DPG_EXACT_SOLUTION

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
 *  ExactSolution.h
 *
 *  Created by Nathan Roberts on 7/5/11.
 */

// Intrepid includes
#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "RHS.h"
#include "BC.h"

#include "Solution.h"

#include "BilinearForm.h"

class ExactSolution {
protected:
  Teuchos::RCP<BilinearForm> _bilinearForm;
  Teuchos::RCP<BC> _bc;
  Teuchos::RCP<RHS> _rhs;
  typedef Teuchos::RCP< ElementType > ElementTypePtr;
  typedef Teuchos::RCP< Element > ElementPtr;
  void squaredDifference(FieldContainer<double> &diffSquared, FieldContainer<double> &values1, FieldContainer<double> &values2);
public:
  Teuchos::RCP<BilinearForm> bilinearForm();
  Teuchos::RCP<BC> bc();
  Teuchos::RCP<RHS> rhs();
  void solutionValues(FieldContainer<double> &values, 
                      int trialID,
                      FieldContainer<double> &physicalPoints);
  virtual double solutionValue(int trialID,
                              FieldContainer<double> &physicalPoint) = 0;
  virtual double solutionValue(int trialID,
                              FieldContainer<double> &physicalPoint,
                              FieldContainer<double> &unitNormal) = 0;
  virtual int H1Order() = 0; // return -1 for non-polynomial solutions
  double L2NormOfError(Solution &solution, int trialID, int cubDegree=-1);
  void L2NormOfError(FieldContainer<double> &errorSquaredPerCell, Solution &solution, ElementTypePtr elemTypePtr, int trialID, int sideIndex=0, int cubDegree=-1, double solutionLift=0.0);
};

#endif