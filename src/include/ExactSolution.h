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

#include "TypeDefs.h"

// Intrepid includes
#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "RHS.h"
#include "BC.h"

#include "Solution.h"

#include "BF.h"
#include "BasisCache.h"

namespace Camellia
{
template <typename Scalar>
class ExactSolution
{
protected:
  TBFPtr<Scalar> _bilinearForm;
  TBCPtr<Scalar> _bc;
  TRHSPtr<Scalar> _rhs;
  // TODO: Fix this for complex (use norm)
  void squaredDifference(Intrepid::FieldContainer<double> &diffSquared, Intrepid::FieldContainer<Scalar> &values1, Intrepid::FieldContainer<Scalar> &values2);

  int _H1Order;
  map< int, TFunctionPtr<Scalar> > _exactFunctions; // var ID --> function
public:
  ExactSolution();
  ExactSolution(TBFPtr<Scalar> bf, TBCPtr<Scalar> bc, TRHSPtr<Scalar> rhs, int H1Order = -1);
  TBFPtr<Scalar> bilinearForm();
  TBCPtr<Scalar> bc();
  TRHSPtr<Scalar> rhs();
  const map< int, TFunctionPtr<Scalar> > exactFunctions(); // not supported by legacy subclasses
  virtual bool functionDefined(int trialID); // not supported by legacy subclasses
  void setSolutionFunction( VarPtr var, TFunctionPtr<Scalar> varFunction );
  void solutionValues(Intrepid::FieldContainer<Scalar> &values, int trialID, BasisCachePtr basisCache);
  void solutionValues(Intrepid::FieldContainer<Scalar> &values,
                      int trialID,
                      Intrepid::FieldContainer<double> &physicalPoints);
  void solutionValues(Intrepid::FieldContainer<Scalar> &values,
                      int trialID,
                      Intrepid::FieldContainer<double> &physicalPoints,
                      Intrepid::FieldContainer<double> &unitNormals);
  virtual Scalar solutionValue(int trialID,
                               Intrepid::FieldContainer<double> &physicalPoint);
  virtual Scalar solutionValue(int trialID,
                               Intrepid::FieldContainer<double> &physicalPoint,
                               Intrepid::FieldContainer<double> &unitNormal);
  virtual int H1Order(); // return -1 for non-polynomial solutions
  // TODO: Fix this for complex
  double L2NormOfError(TSolutionPtr<Scalar> solution, int trialID, int cubDegree=-1);
  void L2NormOfError(Intrepid::FieldContainer<double> &errorSquaredPerCell, TSolutionPtr<Scalar> solution, ElementTypePtr elemTypePtr, int trialID, int sideIndex=VOLUME_INTERIOR_SIDE_ORDINAL, int cubDegree=-1, double solutionLift=0.0);

  virtual ~ExactSolution() {}
};

extern template class ExactSolution<double>;
}

#endif
