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
#include "BasisCache.h"
#include "AbstractFunction.h"
#include "LocalStiffnessMatrixFilter.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Solver.h"

class Function;
typedef Teuchos::RCP<Function> FunctionPtr;

class LagrangeConstraints;
class Epetra_LinearProblem;
class Solution;

typedef Teuchos::RCP<Solution> SolutionPtr;

using namespace Intrepid;

class Solution {
private:
  int _cubatureEnrichmentDegree;
  map< int, FieldContainer<double> > _solutionForCellIDGlobal; // eventually, replace this with a distributed _solutionForCellID
  map< int, double > _energyErrorForCellIDGlobal;

//  map< ElementType*, FieldContainer<double> > _solutionForElementType; // for uniform mesh, just a single entry.
  map< ElementType*, FieldContainer<double> > _residualForElementType; // for uniform mesh, just a single entry.
  map< ElementType*, FieldContainer<double> > _errorRepresentationForElementType; // for uniform mesh, just a single entry.

  // evaluates the inversion of the RHS
  map< ElementType*,FieldContainer<double> > _rhsForElementType;
  map< ElementType*,FieldContainer<double> > _rhsRepresentationForElementType;

  Teuchos::RCP<Mesh> _mesh;
  Teuchos::RCP<BC> _bc;
  Teuchos::RCP<RHS> _rhs;
  Teuchos::RCP<DPGInnerProduct> _ip;
  Teuchos::RCP<LocalStiffnessMatrixFilter> _filter;
  Teuchos::RCP<LagrangeConstraints> _lagrangeConstraints;

  bool _residualsComputed;
  bool _energyErrorComputed;
  // the  values of this map have dimensions (numCells, numTrialDofs)

  void initialize();
  void integrateBasisFunctions(FieldContainer<int> &globalIndices, FieldContainer<double> &values, int trialID);
  void integrateBasisFunctions(FieldContainer<double> &values, ElementTypePtr elemTypePtr, int trialID);

  // statistics for the last solve:
  double _totalTimeLocalStiffness, _totalTimeGlobalAssembly, _totalTimeBCImposition, _totalTimeSolve, _totalTimeDistributeSolution;
  double _meanTimeLocalStiffness, _meanTimeGlobalAssembly, _meanTimeBCImposition, _meanTimeSolve, _meanTimeDistributeSolution;
  double _maxTimeLocalStiffness, _maxTimeGlobalAssembly, _maxTimeBCImposition, _maxTimeSolve, _maxTimeDistributeSolution;
  double _minTimeLocalStiffness, _minTimeGlobalAssembly, _minTimeBCImposition, _minTimeSolve, _minTimeDistributeSolution;

  bool _reportConditionNumber, _reportTimingResults;
  bool _writeMatrixToMatlabFile;
  bool _writeMatrixToMatrixMarketFile;
  bool _writeRHSToMatrixMarketFile;
  string _matrixFilePath;
  string _rhsFilePath;

  double _globalSystemConditionEstimate;

  double _zmcRho;

  void clearComputedResiduals();
  static double conditionNumberEstimate( Epetra_LinearProblem & problem );

  void gatherSolutionData(); // get all solution data onto every node (not what we should do in the end)

protected:
  FieldContainer<double> solutionForElementTypeGlobal(ElementTypePtr elemType); // probably should be deprecated…
  ElementTypePtr getEquivalentElementType(Teuchos::RCP<Mesh> otherMesh, ElementTypePtr elemType);
public:
  Solution(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc = Teuchos::rcp( (BC*) NULL),
           Teuchos::RCP<RHS> rhs = Teuchos::rcp( (RHS*) NULL),
           Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( (DPGInnerProduct*)NULL) );
  Solution(const Solution &soln);
//  bool equals(Solution& otherSolution, double tol=0.0);

  const FieldContainer<double>& allCoefficientsForCellID(int cellID); // coefficients for all solution variables

  Epetra_Map getPartitionMap(int rank, set<int> & myGlobalIndicesSet, int numGlobalDofs, int zeroMeanConstraintsSize, Epetra_Comm* Comm );

  void solve(); // could add arguments to allow different solution algorithms to be selected...

  void solve(bool useMumps);

  void solve( Teuchos::RCP<Solver> solver );

  void addSolution(Teuchos::RCP<Solution> soln, double weight, bool allowEmptyCells = false, bool replaceFluxes=false); // thisSoln += weight * soln
  // static method interprets a set of trial ordering coefficients in terms of a specified DofOrdering
  // and returns a set of weights for the appropriate basis
  static void basisCoeffsForTrialOrder(FieldContainer<double> &basisCoeffs, DofOrderingPtr trialOrder,
                                       const FieldContainer<double> &allCoeffs, int trialID, int sideIndex);


  void clear();

  int cubatureEnrichmentDegree() const;
  void setCubatureEnrichmentDegree(int value);

  void setSolution(Teuchos::RCP<Solution> soln); // thisSoln = soln

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
  void solutionValues(FieldContainer<double> &values, int trialID, BasisCachePtr basisCache,
                      bool weightForCubature = false, EOperatorExtended op = OP_VALUE);
  void solutionValuesOverCells(FieldContainer<double> &values, int trialID, const FieldContainer<double> &physicalPoints);

  void solnCoeffsForCellID(FieldContainer<double> &solnCoeffs, int cellID, int trialID, int sideIndex=0);
  void setSolnCoeffsForCellID(FieldContainer<double> &solnCoeffsToSet, int cellID, int trialID, int sideIndex=0);
  void setSolnCoeffsForCellID(FieldContainer<double> &solnCoeffsToSet, int cellID);

  const map< int, FieldContainer<double> > & solutionForCellIDGlobal() const;

  double integrateSolution(int trialID);
  void integrateSolution(FieldContainer<double> &values, ElementTypePtr elemTypePtr, int trialID);

  void integrateFlux(FieldContainer<double> &values, int trialID);
  void integrateFlux(FieldContainer<double> &values, ElementTypePtr elemTypePtr, int trialID);

  double meanValue(int trialID);
  double meshMeasure();

  double InfNormOfSolution(int trialID);
  double InfNormOfSolutionGlobal(int trialID);

  double L2NormOfSolution(int trialID);
  double L2NormOfSolutionGlobal(int trialID);
  double L2NormOfSolutionInCell(int trialID, int cellID);

  Teuchos::RCP<LagrangeConstraints> lagrangeConstraints() const;

  void processSideUpgrades( const map<int, pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades);
  void processSideUpgrades( const map<int, pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades, const set<int> &cellIDsToSkip );

  // new projectOnto* methods:
  void projectOntoMesh(const map<int, Teuchos::RCP<Function> > &functionMap);
  void projectOntoCell(const map<int, Teuchos::RCP<Function> > &functionMap, int cellID, int sideIndex=-1);
  void projectFieldVariablesOntoOtherSolution(SolutionPtr otherSoln);

  // old projectOnto* methods:
  void projectOntoMesh(const map<int, Teuchos::RCP<AbstractFunction> > &functionMap);
  void projectOntoCell(const map<int, Teuchos::RCP<AbstractFunction> > &functionMap, int cellID);
  void projectOldCellOntoNewCells(int cellID, ElementTypePtr oldElemType, const vector<int> &childIDs);

  void setLagrangeConstraints( Teuchos::RCP<LagrangeConstraints> lagrangeConstraints);
  void setFilter(Teuchos::RCP<LocalStiffnessMatrixFilter> newFilter);
  void setReportConditionNumber(bool value);
  void setReportTimingResults(bool value);

  void computeResiduals();
  void computeErrorRepresentation();

  double globalCondEstLastSolve(); // the condition # estimate for the last system matrix used in a solve, if _reportConditionNumber is true.

  void discardInactiveCellCoefficients();
  double energyErrorTotal();
  const map<int,double> & energyError();

  void writeToFile(int trialID, const string &filePath);
  void writeQuadSolutionToFile(int trialID, const string &filePath);

  void setWriteMatrixToFile(bool value,const string &filePath);
  void setWriteMatrixToMatrixMarketFile(bool value,const string &filePath);
  void setWriteRHSToMatrixMarketFile(bool value, const string &filePath);

  Teuchos::RCP<Mesh> mesh() const;
  Teuchos::RCP<BC> bc() const;
  Teuchos::RCP<RHS> rhs() const;
  Teuchos::RCP<DPGInnerProduct> ip() const;
  Teuchos::RCP<LocalStiffnessMatrixFilter> filter() const;

  void setBC( Teuchos::RCP<BC> );
  void setRHS( Teuchos::RCP<RHS> );
  void setIP( Teuchos::RCP<DPGInnerProduct>);

  // Jesse's additions:
  void condensedSolve(Teuchos::RCP<Solver> globalSolver = Teuchos::rcp(new KluSolver()), bool saveMemory = false);
  void getElemData(ElementPtr elem, FieldContainer<double> &finalStiffness, FieldContainer<double> &localRHSVector);
  void getSubmatrices(set<int> fieldInds, set<int> fluxInds, const FieldContainer<double> K,Epetra_SerialDenseMatrix &K_field, Epetra_SerialDenseMatrix &K_coupl, Epetra_SerialDenseMatrix &K_flux);
  void getSubvectors(set<int> fieldInds, set<int> fluxInds, const FieldContainer<double> b, Epetra_SerialDenseVector &b_field, Epetra_SerialDenseVector &b_flux);

  void readFromFile(const string &filePath);
  void writeToFile(const string &filePath);

  // MATLAB output (belongs elsewhere)
  void writeFieldsToFile(int trialID, const string &filePath);
  void writeFluxesToFile(int trialID, const string &filePath);

  // Default of 0 adapts the number of points based on poly order
  void writeToVTK(const string& filePath, unsigned int num1DPts=0);
  void writeFieldsToVTK(const string& filePath, unsigned int num1DPts=0);
  void writeTracesToVTK(const string& filePath);

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

  void setZeroMeanConstraintRho(double value);
  double zeroMeanConstraintRho();
};

#endif
