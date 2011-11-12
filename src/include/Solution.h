#ifndef DPG_SOLUTION
#define DPG_SOLUTION

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
 *  Solution.h
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */

#include "Intrepid_FieldContainer.hpp"

#include "Mesh.h"
#include "ElementType.h"
#include "DPGInnerProduct.h"
#include "RHS.h"
#include "BC.h"

using namespace Intrepid;

class Solution {
private:
  typedef Teuchos::RCP< ElementType > ElementTypePtr;
  typedef Teuchos::RCP< Element > ElementPtr;
  
  map< ElementType*, FieldContainer<double> > _solutionForElementType; // for uniform mesh, just a single entry.
  map< ElementType*, FieldContainer<double> > _residualForElementType; // for uniform mesh, just a single entry.
  map< ElementType*, FieldContainer<double> > _errorRepresentationForElementType; // for uniform mesh, just a single entry.

  Teuchos::RCP<Mesh> _mesh;
  Teuchos::RCP<BC> _bc;
  Teuchos::RCP<RHS> _rhs;
  Teuchos::RCP<DPGInnerProduct> _ip;
  bool _residualsComputed;
  // the  values of this map have dimensions (numCells, numTrialDofs)
  void initialize();
  void integrateBasisFunctions(FieldContainer<int> &globalIndices, FieldContainer<double> &values, int trialID);
  void integrateBasisFunctions(FieldContainer<double> &values, ElementTypePtr elemTypePtr, int trialID);
protected:
  map< ElementType*, FieldContainer<double> > solutionForElementTypeMap() const;
  ElementTypePtr getEquivalentElementType(Teuchos::RCP<Mesh> otherMesh, ElementTypePtr elemType);
public:
  Solution(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc, Teuchos::RCP<RHS> rhs, Teuchos::RCP<DPGInnerProduct> ip);
  Solution(const Solution &soln);
  bool equals(Solution& otherSolution, double tol=0.0);
  void solve(bool useMumps=false); // could add arguments to allow different solution algorithms to be selected...
  virtual void solutionValues(FieldContainer<double> &values, 
                              ElementTypePtr elemTypePtr, 
                              int trialID,
                              FieldContainer<double> &physicalPoints);
  void solutionValues(FieldContainer<double> &values, 
                      ElementTypePtr elemTypePtr, 
                      int trialID,
                      FieldContainer<double> &physicalPoints,
                      FieldContainer<double> &sideRefCellPoints,
                      int sideIndex);
  void solnCoeffsForCellID(FieldContainer<double> &solnCoeffs, int cellID, int trialID, int sideIndex=0);
  void setSolnCoeffsForCellID(FieldContainer<double> &solnCoeffsToSet, int cellID, int trialID, int sideIndex);
  
  double integrateSolution(int trialID);
  void integrateSolution(FieldContainer<double> &values, ElementTypePtr elemTypePtr, int trialID);
  
  void integrateFlux(FieldContainer<double> &values, int trialID);
  void integrateFlux(FieldContainer<double> &values, ElementTypePtr elemTypePtr, int trialID);
  
  double meanValue(int trialID);
  double meshMeasure();
  
  void computeResiduals();
  void computeErrorRepresentation();
  void energyError(FieldContainer<double> &energyError);
  
  void writeToFile(int trialID, const string &filePath);
  void writeQuadSolutionToFile(int trialID, const string &filePath);
  
  Teuchos::RCP<Mesh> mesh() const;
  Teuchos::RCP<BC> bc() const;
  Teuchos::RCP<RHS> rhs() const;
  Teuchos::RCP<DPGInnerProduct> ip() const;
  
  // Jesse's additions:
  void writeFieldsToFile(int trialID, const string &filePath);
  void writeFluxesToFile(int trialID, const string &filePath);
};

#endif
