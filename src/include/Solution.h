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

// Epetra includes
#include <Epetra_Map.h>
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "Mesh.h"
#include "ElementType.h"
#include "DPGInnerProduct.h"
#include "RHS.h"
#include "BC.h"
#include "AbstractFunction.h"

#include "LocalStiffnessMatrixFilter.h"

using namespace Intrepid;

class Solution {
private:
  typedef Teuchos::RCP< ElementType > ElementTypePtr;
  typedef Teuchos::RCP< Element > ElementPtr;
  
  map< int, FieldContainer<double> > _solutionForCellIDGlobal; // eventually, replace this with a distributed _solutionForCellID
  
//  map< ElementType*, FieldContainer<double> > _solutionForElementType; // for uniform mesh, just a single entry.
  map< ElementType*, FieldContainer<double> > _residualForElementType; // for uniform mesh, just a single entry.
  map< ElementType*, FieldContainer<double> > _errorRepresentationForElementType; // for uniform mesh, just a single entry.

  Teuchos::RCP<Mesh> _mesh;
  Teuchos::RCP<BC> _bc;
  Teuchos::RCP<RHS> _rhs;
  Teuchos::RCP<DPGInnerProduct> _ip;
  Teuchos::RCP<LocalStiffnessMatrixFilter> _filter;

  bool _residualsComputed;
  // the  values of this map have dimensions (numCells, numTrialDofs)
  
  Epetra_Map getPartitionMap(int rank, set<int> & myGlobalIndicesSet, int numGlobalDofs, int zeroMeanConstraintsSize, Epetra_Comm* Comm );
  
  void initialize();
  void integrateBasisFunctions(FieldContainer<int> &globalIndices, FieldContainer<double> &values, int trialID);
  void integrateBasisFunctions(FieldContainer<double> &values, ElementTypePtr elemTypePtr, int trialID);
  
  // statistics for the last solve:
  double _totalTimeLocalStiffness, _totalTimeGlobalAssembly, _totalTimeBCImposition, _totalTimeSolve, _totalTimeDistributeSolution;
  double _meanTimeLocalStiffness, _meanTimeGlobalAssembly, _meanTimeBCImposition, _meanTimeSolve, _meanTimeDistributeSolution;
  double _maxTimeLocalStiffness, _maxTimeGlobalAssembly, _maxTimeBCImposition, _maxTimeSolve, _maxTimeDistributeSolution;
  double _minTimeLocalStiffness, _minTimeGlobalAssembly, _minTimeBCImposition, _minTimeSolve, _minTimeDistributeSolution;
  
protected:
  FieldContainer<double> solutionForElementTypeGlobal(ElementTypePtr elemType); // probably should be deprecated…
  ElementTypePtr getEquivalentElementType(Teuchos::RCP<Mesh> otherMesh, ElementTypePtr elemType);
public:
  Solution(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc, Teuchos::RCP<RHS> rhs, Teuchos::RCP<DPGInnerProduct> ip);
  Solution(const Solution &soln);
//  bool equals(Solution& otherSolution, double tol=0.0);
#ifdef HAVE_MPI
  void solve(bool useMumps=true); // could add arguments to allow different solution algorithms to be selected...
#else
  void solve(bool useMumps=false); // could add arguments to allow different solution algorithms to be selected...
#endif
  void addSolution(Teuchos::RCP<Solution> soln, double weight); // thisSoln += weight * soln
  
  virtual void solutionValues(FieldContainer<double> &values, 
                              ElementTypePtr elemTypePtr, 
                              int trialID,
                              const FieldContainer<double> &physicalPoints);
  void solutionValues(FieldContainer<double> &values, 
                      ElementTypePtr elemTypePtr, 
                      int trialID,
                      const FieldContainer<double> &physicalPoints,
                      const FieldContainer<double> &sideRefCellPoints,
                      int sideIndex);
  void solutionValues(FieldContainer<double> &values, int trialID, const FieldContainer<double> &physicalPoints);
  void solnCoeffsForCellID(FieldContainer<double> &solnCoeffs, int cellID, int trialID, int sideIndex=0);
  void setSolnCoeffsForCellID(FieldContainer<double> &solnCoeffsToSet, int cellID, int trialID, int sideIndex=0);
  
  const map< int, FieldContainer<double> > & solutionForCellIDGlobal() const;
  
  double integrateSolution(int trialID);
  void integrateSolution(FieldContainer<double> &values, ElementTypePtr elemTypePtr, int trialID);
  
  void integrateFlux(FieldContainer<double> &values, int trialID);
  void integrateFlux(FieldContainer<double> &values, ElementTypePtr elemTypePtr, int trialID);
  
  double meanValue(int trialID);
  double meshMeasure();

  void processSideUpgrades( const map<int, pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades );
  void projectOntoMesh(const map<int, Teuchos::RCP<AbstractFunction> > &functionMap);
  void projectOntoCell(const map<int, Teuchos::RCP<AbstractFunction> > &functionMap, int cellID);
  void projectOldCellOntoNewCells(int cellID, ElementTypePtr oldElemType, const vector<int> &childIDs);

  void setFilter(Teuchos::RCP<LocalStiffnessMatrixFilter> newFilter){
    _filter = newFilter;
  }
  
  void computeResiduals();
  void computeErrorRepresentation();
  
  void discardInactiveCellCoefficients();
  //  void energyError(FieldContainer<double> &energyError);
  void energyError(map<int, double> &energyError);
  
  void writeToFile(int trialID, const string &filePath);
  void writeQuadSolutionToFile(int trialID, const string &filePath);
  
  Teuchos::RCP<Mesh> mesh() const;
  Teuchos::RCP<BC> bc() const;
  Teuchos::RCP<RHS> rhs() const;
  Teuchos::RCP<DPGInnerProduct> ip() const;
  Teuchos::RCP<LocalStiffnessMatrixFilter> filter() const;
  
  // Jesse's additions:
  void writeFieldsToFile(int trialID, const string &filePath);
  void writeFluxesToFile(int trialID, const string &filePath);
  
  // statistics accessors:
  double totalTimeLocalStiffness();
  double totalTimeGlobalAssembly();
  double totalTimeBCImposition();
  double totalTimeSolve();
  double totalTimeDistributeSolution();
  
  double meanTimeLocalStiffness();
  double meanTimeGlobalAssembly();
  double meanTimeBCImposition();
  double meanTimeSolve();
  double meanTimeDistributeSolution();
  
  double maxTimeLocalStiffness();
  double maxTimeGlobalAssembly();
  double maxTimeBCImposition();
  double maxTimeSolve();
  double maxTimeDistributeSolution();
  
  double minTimeLocalStiffness();
  double minTimeGlobalAssembly();
  double minTimeBCImposition();
  double minTimeSolve();
  double minTimeDistributeSolution();
  
  void writeStatsToFile(const string &filePath, int precision=4);
};

#endif
