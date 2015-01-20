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

#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"

#include "Mesh.h"
#include "ElementType.h"
#include "IP.h"
#include "RHS.h"
#include "BC.h"
#include "BasisCache.h"
#include "AbstractFunction.h"
#include "LocalStiffnessMatrixFilter.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Solver.h"

#include "DofInterpreter.h"

class Element;
typedef Teuchos::RCP< Element > ElementPtr;

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
  map< GlobalIndexType, FieldContainer<double> > _solutionForCellIDGlobal; // eventually, replace this with a distributed _solutionForCellID
  map< GlobalIndexType, double > _energyErrorForCell; // now rank local
  map< GlobalIndexType, double > _energyErrorForCellGlobal;

  map< GlobalIndexType, FieldContainer<double> > _residualForCell;
  map< GlobalIndexType, FieldContainer<double> > _errorRepresentationForCell;

  // OLD:
//  map< ElementType*, FieldContainer<double> > _residualForElementType; // for uniform mesh, just a single entry.
//  map< ElementType*, FieldContainer<double> > _errorRepresentationForElementType; // for uniform mesh, just a single entry.

  // evaluates the inversion of the RHS
//  map< GlobalIndexType,FieldContainer<double> > _rhsForCell;
  map< GlobalIndexType,FieldContainer<double> > _rhsRepresentationForCell;

  // OLD
//  map< ElementType*,FieldContainer<double> > _rhsForElementType;
//  map< ElementType*,FieldContainer<double> > _rhsRepresentationForElementType;

  Teuchos::RCP<Mesh> _mesh;
  Teuchos::RCP<BC> _bc;
  Teuchos::RCP<DofInterpreter> _dofInterpreter; // defaults to Mesh
  Teuchos::RCP<DofInterpreter> _oldDofInterpreter; // the one saved when we turn on condensed solve
  Teuchos::RCP<RHS> _rhs;
  IPPtr _ip;
  Teuchos::RCP<LocalStiffnessMatrixFilter> _filter;
  Teuchos::RCP<LagrangeConstraints> _lagrangeConstraints;

  Teuchos::RCP<Epetra_CrsMatrix> _globalStiffMatrix;
  Teuchos::RCP<Epetra_FEVector> _rhsVector;
  Teuchos::RCP<Epetra_FEVector> _lhsVector;
  
  bool _residualsComputed;
  bool _energyErrorComputed;
  bool _rankLocalEnergyErrorComputed;
  // the  values of this map have dimensions (numCells, numTrialDofs)

  void initialize();
  void integrateBasisFunctions(FieldContainer<GlobalIndexTypeToCast> &globalIndices, FieldContainer<double> &values, int trialID);
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
  bool _zmcsAsRankOneUpdate;
  
  string _matrixFilePath;
  string _rhsFilePath;

  double _globalSystemConditionEstimate;

  double _zmcRho;

  void clearComputedResiduals();
  static double conditionNumberEstimate( Epetra_LinearProblem & problem );

  void setGlobalSolutionFromCellLocalCoefficients();
  
  void gatherSolutionData(); // get all solution data onto every node (not what we should do in the end)
protected:
  FieldContainer<double> solutionForElementTypeGlobal(ElementTypePtr elemType); // probably should be deprecated…
  ElementTypePtr getEquivalentElementType(Teuchos::RCP<Mesh> otherMesh, ElementTypePtr elemType);
public:
  Solution(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc = Teuchos::null,
           Teuchos::RCP<RHS> rhs = Teuchos::null, IPPtr ip = Teuchos::null);
  Solution(const Solution &soln);
  virtual ~Solution() {}
//  bool equals(Solution& otherSolution, double tol=0.0);

  const FieldContainer<double>& allCoefficientsForCellID(GlobalIndexType cellID, bool warnAboutOffRankImports=true); // coefficients for all solution variables
  void setLocalCoefficientsForCell(GlobalIndexType cellID, const FieldContainer<double> &coefficients);

  Teuchos::RCP<DofInterpreter> getDofInterpreter() const;
  void setDofInterpreter(Teuchos::RCP<DofInterpreter> dofInterpreter);
  
  Epetra_Map getPartitionMap();
  Epetra_Map getPartitionMap(PartitionIndexType rank, set<GlobalIndexType> & myGlobalIndicesSet, GlobalIndexType numGlobalDofs, int zeroMeanConstraintsSize, Epetra_Comm* Comm );
  
  Epetra_MultiVector* getGlobalCoefficients();

  bool cellHasCoefficientsAssigned(GlobalIndexType cellID);
  
  // solve steps:
  void initializeLHSVector();
  void initializeStiffnessAndLoad();
  void populateStiffnessAndLoad();
  void imposeBCs();
  void setProblem(Teuchos::RCP<Solver> solver);
  int solveWithPrepopulatedStiffnessAndLoad(Teuchos::RCP<Solver> solver, bool callResolveInstead = false);
  void importSolution(); // imports for all rank-local cellIDs
  void importSolutionForOffRankCells(std::set<GlobalIndexType> cellIDs);
  void importGlobalSolution(); // imports (and interprets!) global solution.  NOT scalable.
  
  int solve();

  int solve(bool useMumps);

  int solve( Teuchos::RCP<Solver> solver );

  void addSolution(Teuchos::RCP<Solution> soln, double weight, bool allowEmptyCells = false, bool replaceBoundaryTerms=false); // thisSoln += weight * soln
  // will add terms in varsToAdd, but will replace all other variables
  void addSolution(Teuchos::RCP<Solution> soln, double weight, set<int> varsToAdd, bool allowEmptyCells = false); // thisSoln += weight * soln
  // static method interprets a set of trial ordering coefficients in terms of a specified DofOrdering
  // and returns a set of weights for the appropriate basis
  static void basisCoeffsForTrialOrder(FieldContainer<double> &basisCoeffs, DofOrderingPtr trialOrder,
                                       const FieldContainer<double> &allCoeffs, int trialID, int sideIndex);

  void clear();

  int cubatureEnrichmentDegree() const;
  void setCubatureEnrichmentDegree(int value);

  void setSolution(Teuchos::RCP<Solution> soln); // thisSoln = soln

//  void solutionValues(FieldContainer<double> &values,
//                      ElementTypePtr elemTypePtr,
//                      int trialID,
//                      const FieldContainer<double> &physicalPoints);
  void solutionValues(FieldContainer<double> &values,
                      ElementTypePtr elemTypePtr,
                      int trialID,
                      const FieldContainer<double> &physicalPoints,
                      const FieldContainer<double> &sideRefCellPoints,
                      int sideIndex);
  void solutionValues(FieldContainer<double> &values, int trialID, const FieldContainer<double> &physicalPoints); // searches for the elements that match the points provided
  void solutionValues(FieldContainer<double> &values, int trialID, BasisCachePtr basisCache,
                      bool weightForCubature = false, Camellia::EOperator op = OP_VALUE);

  void solnCoeffsForCellID(FieldContainer<double> &solnCoeffs, GlobalIndexType cellID, int trialID, int sideIndex=0);
  void setSolnCoeffsForCellID(FieldContainer<double> &solnCoeffsToSet, GlobalIndexType cellID, int trialID, int sideIndex=0);
  void setSolnCoeffsForCellID(FieldContainer<double> &solnCoeffsToSet, GlobalIndexType cellID);

  const map< GlobalIndexType, FieldContainer<double> > & solutionForCellIDGlobal() const;

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
  double L2NormOfSolutionInCell(int trialID, GlobalIndexType cellID);

  Teuchos::RCP<LagrangeConstraints> lagrangeConstraints() const;
  
  void processSideUpgrades( const map<GlobalIndexType, pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades);
  void processSideUpgrades( const map<GlobalIndexType, pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades, const set<GlobalIndexType> &cellIDsToSkip );

  // new projectOnto* methods:
  void projectOntoMesh(const map<int, Teuchos::RCP<Function> > &functionMap);
  void projectOntoCell(const map<int, Teuchos::RCP<Function> > &functionMap, GlobalIndexType cellID, int sideIndex=-1);
  void projectFieldVariablesOntoOtherSolution(SolutionPtr otherSoln);

  // old projectOnto* methods:
  void projectOntoMesh(const map<int, Teuchos::RCP<AbstractFunction> > &functionMap);
  void projectOntoCell(const map<int, Teuchos::RCP<AbstractFunction> > &functionMap, GlobalIndexType cellID);
  
  void projectOldCellOntoNewCells(GlobalIndexType cellID,
                                  ElementTypePtr oldElemType,
                                  const vector<GlobalIndexType> &childIDs);
  void projectOldCellOntoNewCells(GlobalIndexType cellID,
                                  ElementTypePtr oldElemType,
                                  const FieldContainer<double> &oldData,
                                  const vector<GlobalIndexType> &childIDs);
  
  void setLagrangeConstraints( Teuchos::RCP<LagrangeConstraints> lagrangeConstraints);
  void setFilter(Teuchos::RCP<LocalStiffnessMatrixFilter> newFilter);
  void setReportConditionNumber(bool value);
  void setReportTimingResults(bool value);

  void computeResiduals();
  void computeErrorRepresentation();

  double globalCondEstLastSolve(); // the condition # estimate for the last system matrix used in a solve, if _reportConditionNumber is true.

  void discardInactiveCellCoefficients();
  double energyErrorTotal();
  const map<GlobalIndexType,double> & globalEnergyError();
  const map<GlobalIndexType,double> & rankLocalEnergyError();

  void writeToFile(int trialID, const string &filePath);
  void writeQuadSolutionToFile(int trialID, const string &filePath);

  void setWriteMatrixToFile(bool value,const string &filePath);
  void setWriteMatrixToMatrixMarketFile(bool value,const string &filePath);
  void setWriteRHSToMatrixMarketFile(bool value, const string &filePath);

  Teuchos::RCP<Mesh> mesh() const;
  Teuchos::RCP<BC> bc() const;
  Teuchos::RCP<RHS> rhs() const;
  IPPtr ip() const;
  Teuchos::RCP<LocalStiffnessMatrixFilter> filter() const;

  void setBC( Teuchos::RCP<BC> );
  void setRHS( Teuchos::RCP<RHS> );
  
  Teuchos::RCP<Epetra_CrsMatrix> getStiffnessMatrix();
  void setStiffnessMatrix(Teuchos::RCP<Epetra_CrsMatrix> stiffness);

  Teuchos::RCP<Epetra_FEVector> getRHSVector();
  Teuchos::RCP<Epetra_FEVector> getLHSVector();
  
  void setIP( IPPtr);

#if defined(HAVE_MPI) && defined(HAVE_AMESOS_MUMPS)
  void condensedSolve(Teuchos::RCP<Solver> globalSolver = Teuchos::rcp(new MumpsSolver()), bool reduceMemoryFootprint = false); // when reduceMemoryFootprint is true, local stiffness matrices will be computed twice, rather than stored for reuse
#else
  void condensedSolve(Teuchos::RCP<Solver> globalSolver = Teuchos::rcp(new KluSolver()), bool reduceMemoryFootprint = false); // when reduceMemoryFootprint is true, local stiffness matrices will be computed twice, rather than stored for reuse
#endif
  void readFromFile(const string &filePath);
  void writeToFile(const string &filePath);

#ifdef HAVE_EPETRAEXT_HDF5
  void save(string meshAndSolutionPrefix);
  static SolutionPtr load(BilinearFormPtr bf, string meshAndSolutionPrefix);
  void saveToHDF5(string filename);
  void loadFromHDF5(string filename);
#endif

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

  void reportTimings();

  void setUseCondensedSolve(bool value);
  
  void writeStatsToFile(const string &filePath, int precision=4);

  vector<int> getZeroMeanConstraints();
  void setZeroMeanConstraintRho(double value);
  double zeroMeanConstraintRho();
  
  static SolutionPtr solution(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc = Teuchos::null,
                              Teuchos::RCP<RHS> rhs = Teuchos::null,
                              IPPtr ip = Teuchos::null);
};

#endif
