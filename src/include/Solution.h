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

#include "TypeDefs.h"

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
#include "Epetra_Operator.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"

#include "BasisCache.h"
#include "DofInterpreter.h"
#include "ElementType.h"
#include "LocalStiffnessMatrixFilter.h"
#include "Solver.h"

namespace Camellia {
  template <typename Scalar>
  class TSolution {
  private:
    int _cubatureEnrichmentDegree;
    std::map< GlobalIndexType, Intrepid::FieldContainer<Scalar> > _solutionForCellIDGlobal; // eventually, replace this with a distributed _solutionForCellID
    std::map< GlobalIndexType, double > _energyErrorForCell; // now rank local
    std::map< GlobalIndexType, double > _energyErrorForCellGlobal;

    map< GlobalIndexType, Intrepid::FieldContainer<double> > _residualForCell;
    std::map< GlobalIndexType, Intrepid::FieldContainer<double> > _errorRepresentationForCell;

    // evaluates the inversion of the RHS
    std::map< GlobalIndexType,Intrepid::FieldContainer<Scalar> > _rhsRepresentationForCell;

    MeshPtr _mesh;
    TBCPtr<Scalar> _bc;
    Teuchos::RCP<DofInterpreter> _dofInterpreter; // defaults to Mesh
    Teuchos::RCP<DofInterpreter> _oldDofInterpreter; // the one saved when we turn on condensed solve
    TBFPtr<Scalar> _bf;
    TRHSPtr<Scalar> _rhs;
    TIPPtr<Scalar> _ip;
    Teuchos::RCP<LocalStiffnessMatrixFilter> _filter;
    Teuchos::RCP<LagrangeConstraints> _lagrangeConstraints;

    Teuchos::RCP<Epetra_CrsMatrix> _globalStiffMatrix;
    Teuchos::RCP<Epetra_FEVector> _rhsVector;
    Teuchos::RCP<Epetra_FEVector> _lhsVector;

    TMatrixPtr<Scalar> _globalStiffMatrix2;
    TVectorPtr<Scalar> _rhsVector2;
    TVectorPtr<Scalar> _lhsVector2;

    bool _residualsComputed;
    bool _energyErrorComputed;
    bool _rankLocalEnergyErrorComputed;
    // the  values of this map have dimensions (numCells, numTrialDofs)

    void initialize();
    void integrateBasisFunctions(Intrepid::FieldContainer<GlobalIndexTypeToCast> &globalIndices,
                                 Intrepid::FieldContainer<Scalar> &values, int trialID);
    void integrateBasisFunctions(Intrepid::FieldContainer<Scalar> &values, ElementTypePtr elemTypePtr, int trialID);

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
    bool _zmcsAsLagrangeMultipliers;

    std::string _matrixFilePath;
    std::string _rhsFilePath;

    double _globalSystemConditionEstimate;

    double _zmcRho;

    static double conditionNumberEstimate( Epetra_LinearProblem & problem );

    void setGlobalSolutionFromCellLocalCoefficients();

    void gatherSolutionData(); // get all solution data onto every node (not what we should do in the end)
  protected:
    Intrepid::FieldContainer<Scalar> solutionForElementTypeGlobal(ElementTypePtr elemType); // probably should be deprecated…
    ElementTypePtr getEquivalentElementType(MeshPtr otherMesh, ElementTypePtr elemType);
  public:
    TSolution(TBFPtr<Scalar> bf, MeshPtr mesh, TBCPtr<Scalar> bc = Teuchos::null,
             TRHSPtr<Scalar> rhs = Teuchos::null, TIPPtr<Scalar> ip = Teuchos::null);
    // Deprecated constructor, use the one which explicitly passes in BF
    // Will eventually be removing BF reference from Mesh
    TSolution(MeshPtr mesh, TBCPtr<Scalar> bc = Teuchos::null,
             TRHSPtr<Scalar> rhs = Teuchos::null, TIPPtr<Scalar> ip = Teuchos::null);
    TSolution(const TSolution &soln);
    virtual ~TSolution() {}

    const Intrepid::FieldContainer<Scalar>& allCoefficientsForCellID(GlobalIndexType cellID, bool warnAboutOffRankImports=true); // coefficients for all solution variables
    void setLocalCoefficientsForCell(GlobalIndexType cellID, const Intrepid::FieldContainer<Scalar> &coefficients);

    Teuchos::RCP<DofInterpreter> getDofInterpreter() const;
    void setDofInterpreter(Teuchos::RCP<DofInterpreter> dofInterpreter);

    Epetra_Map getPartitionMap();
    Epetra_Map getPartitionMapSolutionDofsOnly(); // omits lagrange constraints, zmcs, etc.
    Epetra_Map getPartitionMap(PartitionIndexType rank, std::set<GlobalIndexType> &myGlobalIndicesSet,
                               GlobalIndexType numGlobalDofs, int zeroMeanConstraintsSize, Epetra_Comm* Comm );

    MapPtr getPartitionMap2();
    // Not implemented for now
    // MapPtr getPartitionMapSolutionDofsOnly2(); // omits lagrange constraints, zmcs, etc.
    MapPtr getPartitionMap2(PartitionIndexType rank, std::set<GlobalIndexType> &myGlobalIndicesSet,
                               GlobalIndexType numGlobalDofs, int zeroMeanConstraintsSize, Teuchos::RCP<const Teuchos::Comm<int> > Comm );

    Epetra_MultiVector* getGlobalCoefficients();
    TVectorPtr<Scalar> getGlobalCoefficients2();

    bool cellHasCoefficientsAssigned(GlobalIndexType cellID);
    void clearComputedResiduals();

    bool getZMCsAsGlobalLagrange() const;
    void setZMCsAsGlobalLagrange(bool value); // should be set before call to initializeLHSVector(), initializeStiffnessAndLoad()

    // solve steps:
    void initializeLHSVector();
    void initializeStiffnessAndLoad();
    void populateStiffnessAndLoad();
    void imposeBCs();
    void imposeZMCsUsingLagrange(); // if not using Lagrange for ZMCs, puts 1's in the diagonal for these rows
    void setProblem(TSolverPtr<Scalar> solver);
    int solveWithPrepopulatedStiffnessAndLoad(TSolverPtr<Scalar> solver, bool callResolveInstead = false);
    void importSolution(); // imports for all rank-local cellIDs
    void importSolutionForOffRankCells(std::set<GlobalIndexType> cellIDs);
    void importGlobalSolution(); // imports (and interprets!) global solution.  NOT scalable.

    int solve();

    int solve(bool useMumps);

    int solve( TSolverPtr<Scalar> solver );

    void addSolution(TSolutionPtr<Scalar> soln, double weight, bool allowEmptyCells = false, bool replaceBoundaryTerms=false); // thisSoln += weight * soln

    // will add terms in varsToAdd, but will replace all other variables
    void addSolution(TSolutionPtr<Scalar> soln, double weight, set<int> varsToAdd, bool allowEmptyCells = false); // thisSoln += weight * soln

    // static method interprets a set of trial ordering coefficients in terms of a specified DofOrdering
    // and returns a set of weights for the appropriate basis
    static void basisCoeffsForTrialOrder(Intrepid::FieldContainer<Scalar> &basisCoeffs, DofOrderingPtr trialOrder,
                                         const Intrepid::FieldContainer<Scalar> &allCoeffs, int trialID, int sideIndex);

    void clear();

    int cubatureEnrichmentDegree() const;
    void setCubatureEnrichmentDegree(int value);

    void setSolution(TSolutionPtr<Scalar> soln); // thisSoln = soln

    void solutionValues(Intrepid::FieldContainer<Scalar> &values, ElementTypePtr elemTypePtr, int trialID,
                        const Intrepid::FieldContainer<double> &physicalPoints,
                        const Intrepid::FieldContainer<double> &sideRefCellPoints,
                        int sideIndex);
    void solutionValues(Intrepid::FieldContainer<Scalar> &values, int trialID,
                        const Intrepid::FieldContainer<double> &physicalPoints); // searches for the elements that match the points provided
    void solutionValues(Intrepid::FieldContainer<Scalar> &values, int trialID, BasisCachePtr basisCache,
                        bool weightForCubature = false, Camellia::EOperator op = OP_VALUE);

    void solnCoeffsForCellID(Intrepid::FieldContainer<Scalar> &solnCoeffs, GlobalIndexType cellID, int trialID, int sideIndex=0);
    void setSolnCoeffsForCellID(Intrepid::FieldContainer<Scalar> &solnCoeffsToSet, GlobalIndexType cellID, int trialID, int sideIndex=0);
    void setSolnCoeffsForCellID(Intrepid::FieldContainer<Scalar> &solnCoeffsToSet, GlobalIndexType cellID);

    const std::map< GlobalIndexType, Intrepid::FieldContainer<Scalar> > & solutionForCellIDGlobal() const;

    Scalar integrateSolution(int trialID);
    void integrateSolution(Intrepid::FieldContainer<Scalar> &values, ElementTypePtr elemTypePtr, int trialID);

    void integrateFlux(Intrepid::FieldContainer<Scalar> &values, int trialID);
    void integrateFlux(Intrepid::FieldContainer<Scalar> &values, ElementTypePtr elemTypePtr, int trialID);

    Scalar meanValue(int trialID);
    double meshMeasure();

    double InfNormOfSolution(int trialID);
    double InfNormOfSolutionGlobal(int trialID);

    double L2NormOfSolution(int trialID);
    double L2NormOfSolutionGlobal(int trialID);
    double L2NormOfSolutionInCell(int trialID, GlobalIndexType cellID);

    Teuchos::RCP<LagrangeConstraints> lagrangeConstraints() const;

    void processSideUpgrades( const std::map<GlobalIndexType, std::pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades);
    void processSideUpgrades( const std::map<GlobalIndexType, std::pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades, const std::set<GlobalIndexType> &cellIDsToSkip );

    void projectOntoMesh(const std::map<int, TFunctionPtr<Scalar> > &functionMap);
    void projectOntoCell(const std::map<int, TFunctionPtr<Scalar> > &functionMap, GlobalIndexType cellID, int sideIndex=-1);
    void projectFieldVariablesOntoOtherSolution(TSolutionPtr<Scalar> otherSoln);

    void projectOldCellOntoNewCells(GlobalIndexType cellID,
                                    ElementTypePtr oldElemType,
                                    const vector<GlobalIndexType> &childIDs);
    void projectOldCellOntoNewCells(GlobalIndexType cellID,
                                    ElementTypePtr oldElemType,
                                    const Intrepid::FieldContainer<Scalar> &oldData,
                                    const std::vector<GlobalIndexType> &childIDs);

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

    void writeToFile(int trialID, const std::string &filePath);
    void writeQuadSolutionToFile(int trialID, const std::string &filePath);

    void setWriteMatrixToFile(bool value,const std::string &filePath);
    void setWriteMatrixToMatrixMarketFile(bool value,const std::string &filePath);
    void setWriteRHSToMatrixMarketFile(bool value, const std::string &filePath);

    MeshPtr mesh() const;
    TBFPtr<Scalar> bf() const;
    TBCPtr<Scalar> bc() const;
    TRHSPtr<Scalar> rhs() const;
    TIPPtr<Scalar> ip() const;
    Teuchos::RCP<LocalStiffnessMatrixFilter> filter() const;

    void setBC( TBCPtr<Scalar> );
    void setRHS( TRHSPtr<Scalar> );

    Teuchos::RCP<Epetra_CrsMatrix> getStiffnessMatrix();
    TMatrixPtr<Scalar> getStiffnessMatrix2();
    void setStiffnessMatrix(Teuchos::RCP<Epetra_CrsMatrix> stiffness);
    void setStiffnessMatrix2(TMatrixPtr<Scalar> stiffness);

    Teuchos::RCP<Epetra_FEVector> getRHSVector();
    Teuchos::RCP<Epetra_FEVector> getLHSVector();

    TVectorPtr<Scalar> getRHSVector2();
    TVectorPtr<Scalar> getLHSVector2();

    void setIP( TIPPtr<Scalar>);

  #if defined(HAVE_MPI) && defined(HAVE_AMESOS_MUMPS)
    void condensedSolve(TSolverPtr<Scalar> globalSolver = Teuchos::rcp(new MumpsSolver()), bool reduceMemoryFootprint = false); // when reduceMemoryFootprint is true, local stiffness matrices will be computed twice, rather than stored for reuse
  #else
    void condensedSolve(TSolverPtr<Scalar> globalSolver = Teuchos::rcp(new TAmesos2Solver<Scalar>()), bool reduceMemoryFootprint = false); // when reduceMemoryFootprint is true, local stiffness matrices will be computed twice, rather than stored for reuse
  #endif
    void readFromFile(const std::string &filePath);
    void writeToFile(const std::string &filePath);

  #ifdef HAVE_EPETRAEXT_HDF5
    void save(std::string meshAndSolutionPrefix);
    static TSolutionPtr<Scalar> load(TBFPtr<Scalar> bf, std::string meshAndSolutionPrefix);
    void saveToHDF5(std::string filename);
    void loadFromHDF5(std::string filename);
  #endif

    // MATLAB output (belongs elsewhere)
    void writeFieldsToFile(int trialID, const std::string &filePath);
    void writeFluxesToFile(int trialID, const std::string &filePath);

    // Default of 0 adapts the number of points based on poly order
    void writeToVTK(const std::string& filePath, unsigned int num1DPts=0);
    void writeFieldsToVTK(const std::string& filePath, unsigned int num1DPts=0);
    void writeTracesToVTK(const std::string& filePath);

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

    void writeStatsToFile(const std::string &filePath, int precision=4);

    std::vector<int> getZeroMeanConstraints();
    void setZeroMeanConstraintRho(double value);
    double zeroMeanConstraintRho();

    static TSolutionPtr<Scalar> solution(TBFPtr<Scalar> bf, MeshPtr mesh, TBCPtr<Scalar> bc = Teuchos::null,
                                TRHSPtr<Scalar> rhs = Teuchos::null,
                                TIPPtr<Scalar> ip = Teuchos::null);
    // Deprecated method, use the above one
    static TSolutionPtr<Scalar> solution(MeshPtr mesh, TBCPtr<Scalar> bc = Teuchos::null,
                                TRHSPtr<Scalar> rhs = Teuchos::null,
                                TIPPtr<Scalar> ip = Teuchos::null);
  };

  extern template class TSolution<double>;
}


#endif
