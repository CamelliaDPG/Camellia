#ifndef DPG_SOLUTION
#define DPG_SOLUTION

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

/*
 *  Solution.h
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */

#include "TypeDefs.h"

#include "Camellia_FieldContainer.hpp"

#include "BasisCache.h"
#include "DofInterpreter.h"
#include "ElementType.h"
#include "LocalStiffnessMatrixFilter.h"
#include "Narrator.h"
#include "Solver.h"

namespace Camellia
{
  const static string SOLUTION_LOCAL_STIFFNESS_LOAD_TIMER_STRING = "Solution: local stiffness/load";
  const static string SOLUTION_INTERPRET_LOCAL_DATA_TIMER_STRING = "Solution: interpretLocalData";
  
template <typename Scalar>
  class TSolution : public Narrator
{
private:
  int _cubatureEnrichmentDegree;
  Intrepid2::EPolyType _polyTypeForQuadrature = Intrepid2::POLYTYPE_MAX;
  
  template<class MapType, class KeyType, class ValueType>
  inline
  void replaceMapEntry(MapType &map, KeyType &key, ValueType &value)
  {
    // for const values, maps don't let you do map[key] = value
    // counter-intuitively, they also don't replace the value of an existing key when you do .insert()
    // this is a one-line invocation that does map[key] = value for this case
    map.erase(key);
    map.insert(std::make_pair(key,value));
    invalidateCachedBasisCoefficientsForFields();
  }
  
  // first index into _solutionForCellID corresponds to the index of the RHS (0 for primary, 1 for influence, at present)
  std::vector<std::map< GlobalIndexType, Camellia::FieldContainer<Scalar> > > _solutionForCellID;
  std::map< GlobalIndexType, double > _energyErrorForCell; // this is *just* for the primary solution.  Use RieszRep if dealing with the influence function...

  map< GlobalIndexType, const Camellia::FieldContainer<double> > _residualForCell;
  std::map< GlobalIndexType, const Camellia::FieldContainer<double> > _errorRepresentationForCell;

  // evaluates the inversion of the RHS
  std::map< GlobalIndexType, const Camellia::FieldContainer<Scalar> > _rhsRepresentationForCell;

  MeshPtr _mesh;
  TBCPtr<Scalar> _bc;
  Teuchos::RCP<DofInterpreter> _dofInterpreter; // defaults to Mesh
  Teuchos::RCP<DofInterpreter> _oldDofInterpreter; // the one saved when we turn on condensed solve
  TBFPtr<Scalar> _bf;
  TRHSPtr<Scalar> _rhs, _rhsWithoutTableauModification;
  TIPPtr<Scalar> _ip;
  
  int _cellBatchSize = -1; // -1 ==> automatically determined
  
  SolutionStagePtr _solutionStage;
  
  LinearTermPtr _goalOrientedRHS;  // trial space functional.
  
  Teuchos::RCP<LocalStiffnessMatrixFilter> _filter;
  Teuchos::RCP<LagrangeConstraints> _lagrangeConstraints;

  TMatrixPtr<Scalar> _globalStiffMatrix;
  TVectorPtr<Scalar> _rhsVector;
  TVectorPtr<Scalar> _lhsVector; // 1 or 2 wide for steady; for RK schemes, this is numStages + 1 wide (assuming no goal-oriented RHS)
  TVectorPtr<Scalar> _lhsVectorCurrentStage; // for RK schemes, this is numRHSes wide for the active stage

  bool _residualsComputed;
  bool _energyErrorComputed;
  bool _rankLocalEnergyErrorComputed;
  int _warnAboutDiscontinuousBCs = 1;
  // the  values of this map have dimensions (numCells, numTrialDofs)

  void initialize();
  void integrateBasisFunctions(Camellia::FieldContainer<GlobalIndexTypeToCast> &globalIndices,
                               Camellia::FieldContainer<Scalar> &values, int trialID);
  void integrateBasisFunctions(Camellia::FieldContainer<Scalar> &values, ElementTypePtr elemTypePtr, int trialID);

  // statistics for the last solve:
  double _totalTimeLocalStiffness, _totalTimeGlobalAssembly, _totalTimeBCImposition, _totalTimeSolve, _totalTimeDistributeSolution;
  double _meanTimeLocalStiffness, _meanTimeGlobalAssembly, _meanTimeBCImposition, _meanTimeSolve, _meanTimeDistributeSolution;
  double _maxTimeLocalStiffness, _maxTimeGlobalAssembly, _maxTimeBCImposition, _maxTimeSolve, _maxTimeDistributeSolution;
  double _minTimeLocalStiffness, _minTimeGlobalAssembly, _minTimeBCImposition, _minTimeSolve, _minTimeDistributeSolution;
  double _totalTimeApplyJumpTerms, _meanTimeApplyJumpTerms, _maxTimeApplyJumpTerms, _minTimeApplyJumpTerms;

  bool _reportConditionNumber, _reportTimingResults;
  bool _saveMeshOnSolveError = true; // if there is a solve error, save the mesh to disk for potential analysis
  bool _writeMatrixToMatrixMarketFile;
  bool _writeRHSToMatrixMarketFile;
  bool _zmcsAsRankOneUpdate;
  bool _zmcsAsLagrangeMultipliers;

  std::string _matrixFilePath;
  std::string _rhsFilePath;
  
  std::string _solutionIdentifier = ""; // used to allow Solutions to be presented distinctly to the user (e.g., current and previous time step)

  GlobalIndexType _bcDofCount; // global count of the number of BCs imposed during last call to imposeBCs().
  
  double _zmcRho;

  void setGlobalSolutionFromCellLocalCoefficients();

  void gatherSolutionData(); // get all solution data onto every node (not what we should do in the end)
  
  /*
   The following containers store cached FieldContainers to allow reuse of data copied to device in solutionValues().
   It may be that once Solution itself stores device views these will become unnecessary.
   
   Right now, these only store coefficients for a single batch of cellIDs; this reflects what we expect to be
   standard usage in the context of e.g. nonlinear solves: you have a BasisCache for a batch of cellIDs that you use
   to compute a bunch of (background flow) data based on solution values, but once you are done with that BasisCache,
   you move on to the next batch of cell IDs, and don't need the solution values again (until the next update, at which
   point the solution will have changed, and therefore the cached data will have been invalidated).
   
   For other use cases, we could increase caching (and therefore reuse) by having a lookup based on the batch of cellIDs,
   but this would introduce some additional complexity and memory costs.
   */
  std::vector< Teuchos::RCP<const Camellia::FieldContainer<Scalar> > > _cachedCellBasisCoefficientsForFields; // these are in field ID order
  std::vector<int> _cachedCellBasisCoefficientsForFieldsCellIDs; // which cells are represented above?
  int _cachedCellBasisCoefficientsForFieldsSolutionOrdinal = -2; // which solution ordinal do they correspond to?
  void invalidateCachedBasisCoefficientsForFields();
  
protected:
  RHSPtr rhsWithoutTableauModification() const;
public:
  TSolution(TBFPtr<Scalar> bf, MeshPtr mesh, TBCPtr<Scalar> bc = Teuchos::null,
            TRHSPtr<Scalar> rhs = Teuchos::null, TIPPtr<Scalar> ip = Teuchos::null);
  // Deprecated constructor, use the one which explicitly passes in BF
  // Will eventually be removing BF reference from Mesh
  TSolution(MeshPtr mesh, TBCPtr<Scalar> bc = Teuchos::null,
            TRHSPtr<Scalar> rhs = Teuchos::null, TIPPtr<Scalar> ip = Teuchos::null);
  TSolution(const TSolution &soln);
  virtual ~TSolution() {}

  const Camellia::FieldContainer<Scalar>& allCoefficientsForCellID(GlobalIndexType cellID, bool warnAboutOffRankImports=true,
                                                                   int solutionNumber=0); // coefficients for all solution variables
  void setLocalCoefficientsForCell(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &coefficients, int solutionNumber=0);

  void getBasisCoefficientsForCell(Camellia::FieldContainer<Scalar> &basisCoefficients, int trialID, GlobalIndexType cellID,
                                   SolutionComponent standardOrInfluence, int stageOrdinal, int sideIndex = -1);
  
  Teuchos::RCP<DofInterpreter> getDofInterpreter() const;
  void setDofInterpreter(Teuchos::RCP<DofInterpreter> dofInterpreter);

  TMapPtr getPartitionMap();
  TMapPtr getPartitionMap(PartitionIndexType rank, std::set<GlobalIndexType> &myGlobalIndicesSet,
                          GlobalIndexType numGlobalDofs, int zeroMeanConstraintsSize, Teuchos_CommPtr Comm );

  TVectorPtr<Scalar> getGlobalCoefficients();

  bool cellHasCoefficientsAssigned(GlobalIndexType cellID, int solutionOrdinal=0);
  void clearComputedResiduals();

  bool getZMCsAsGlobalLagrange() const;
  void setZMCsAsGlobalLagrange(bool value); // should be set before call to initializeLHSVector(), initializeStiffnessAndLoad()

  // solve steps:
  void initializeLHSVector();
  void initializeLoad(); // to be used if setStiffnessMatrix() will be called...
  void initializeStiffnessAndLoad();
  void applyDGJumpTerms();
  void populateStiffnessAndLoad();
  void imposeBCs();
  void imposeZMCsUsingLagrange(); // if not using Lagrange for ZMCs, puts 1's in the diagonal for these rows
  void setProblem(TSolverPtr<Scalar> solver);
  int solveWithPrepopulatedStiffnessAndLoad(TSolverPtr<Scalar> solver, bool callResolveInstead = false);
  void importSolution(); // imports for all rank-local cellIDs
  void importSolutionForOffRankCells(const std::set<GlobalIndexType> &cellIDs);
  void importGlobalSolution(); // imports (and interprets!) global solution.  NOT scalable.

  int solve();

  int solve(bool useMumps);

  int solve( TSolverPtr<Scalar> solver );

  void addSolution(TSolutionPtr<Scalar> soln, double weight, bool allowEmptyCells = false, bool replaceBoundaryTerms=false); // thisSoln += weight * soln

  void addSolution(TSolutionPtr<Scalar> soln, double weight, set<int> varsToAdd, bool allowEmptyCells = false); // thisSoln += weight * soln
  // will add terms in varsToAdd, but will replace all other variables
  void addReplaceSolution(TSolutionPtr<Scalar> soln, double weight, set<int> varsToAdd, set<int> varsToReplace, bool allowEmptyCells = false); // thisSoln += weight * soln
  
  // adds terms in varsToAdd using newValue = myWeight * oldValue + addedVarWeight * solnValue; replaces varsToReplace as newValue = solnValue
  void addReplaceSolution(TSolutionPtr<Scalar> soln, double addedVarWeight, double myWeight, set<int> varsToAdd, set<int> varsToReplace, bool allowEmptyCells = false); // thisSoln += weight * soln

  // static method interprets a set of trial ordering coefficients in terms of a specified DofOrdering
  // and returns a set of weights for the appropriate basis
  static void basisCoeffsForTrialOrder(Camellia::FieldContainer<Scalar> &basisCoeffs, DofOrderingPtr trialOrder,
                                       const Camellia::FieldContainer<Scalar> &allCoeffs, int trialID, int sideIndex);

  int cellBatchSize() const;
  void setCellBatchSize(const int newBatchSize);
  
  void clear();
  
  int cubatureEnrichmentDegree() const;
  void setCubatureEnrichmentDegree(int value);
  
  void setPolyTypeForQuadrature(Intrepid2::EPolyType polyTypeForQuadrature);
  
  const std::string & getIdentifier() const;
  void setIdentifier(const std::string &solutionIdentifier);
  // ! returns the global count of the number of degrees of freedom on which boundary conditions were imposed in
  // ! the last call to imposeBCs().  (imposeBCs() is called in solve().)
  GlobalIndexType getBCDofCount() const;

  void setSolution(TSolutionPtr<Scalar> soln); // thisSoln = soln

  void solutionValues(Camellia::FieldContainer<Scalar> &values, int trialID,
                      const Camellia::FieldContainer<double> &physicalPoints, SolutionComponent standardOrInfluence=STANDARD_SOLUTION,
                      int stageOrdinal=0); // searches for the elements that match the points provided
  void solutionValues(Camellia::FieldContainer<Scalar> &values, int trialID, BasisCachePtr basisCache,
                      SolutionComponent standardOrInfluence=STANDARD_SOLUTION, int stageOrdinal=0,
                      bool weightForCubature = false, Camellia::EOperator op = OP_VALUE);
  void solutionValues(typename Camellia::FieldContainer<Scalar>::DeviceView &values, int trialID, BasisCachePtr basisCache,
                      SolutionComponent standardOrInfluence=STANDARD_SOLUTION, int stageOrdinal=0,
                      bool weightForCubature = false, Camellia::EOperator op = OP_VALUE,
                      std::vector<GlobalIndexType> cellIDsOverride = std::vector<GlobalIndexType>(),
                      const int sideOrdinalOverride = -2);

  void solnCoeffsForCellID(Camellia::FieldContainer<Scalar> &solnCoeffs, GlobalIndexType cellID, int trialID, int sideIndex=VOLUME_INTERIOR_SIDE_INDEX,
                           int solutionOrdinal=0);
  void setSolnCoeffsForCellID(const Camellia::FieldContainer<Scalar> &solnCoeffsToSet, GlobalIndexType cellID, int trialID,
                              int sideIndex, int solutionOrdinal=0);
  void setSolnCoeffsForCellID(const Camellia::FieldContainer<Scalar> &solnCoeffsToSet, GlobalIndexType cellID, int solutionOrdinal=0);

  const std::vector< std::map< GlobalIndexType, Camellia::FieldContainer<Scalar> > > & solutionForCellID() const;

  double meshMeasure();

  double InfNormOfSolution(int trialID);
  double InfNormOfSolutionGlobal(int trialID);

  double L2NormOfSolution(int trialID);
  double L2NormOfSolutionGlobal(int trialID);

  Teuchos::RCP<LagrangeConstraints> lagrangeConstraints() const;

  // ! provides the GID (in stiffness/load) corresponding to an element lagrange constraint
  GlobalIndexType elementLagrangeIndex(GlobalIndexType cellID, int lagrangeOrdinal) const;
  
  //! returns the number of RHSes we use in the solve for; right now, this is 1 or 2 (2 in the case that goal-oriented RHS is set).
  int numRHSes() const;
  
  //! total number of solutions in lhsVector() -- (numStages+1) * numRHSes.
  int numSolutions() const;
  
  //! number of stages in the ERKTableau (0 if no ERKTableau is set).
  int numStages() const;
  
  //! get the index in lhsVector (and _solutionForCellID) for specified solutionComponent, solutionStage
  //! Note that solutionStage == -1 uses the current stage, as defined by _solutionStage.
  int solutionOrdinal(SolutionComponent solutionComponent, int solutionStage) const;
  
  void processSideUpgrades( const std::map<GlobalIndexType, std::pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades);
  void processSideUpgrades( const std::map<GlobalIndexType, std::pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades, const std::set<GlobalIndexType> &cellIDsToSkip );

  void projectOntoMesh(const std::map<int, TFunctionPtr<Scalar> > &functionMap, int solutionOrdinal=0);
  void projectOntoCell(const std::map<int, TFunctionPtr<Scalar> > &functionMap, GlobalIndexType cellID, int sideIndex, int solutionOrdinal=0);
  void projectFieldVariablesOntoOtherSolution(TSolutionPtr<Scalar> otherSoln);

  // ! project child coefficients onto parent; useful for unrefinement.  Caller is responsible for ensuring
  // ! that all child coefficient data is rank-local.  (See implementation of GlobalDofAssignment::projectChildCoefficientsOntoParents() for an example of how to do this.)
  void projectChildrenOntoParent(GlobalIndexType parentCellID, int solutionOrdinal=0);
  
  void projectOldCellOntoNewCells(GlobalIndexType cellID, ElementTypePtr oldElemType,
                                  const vector<GlobalIndexType> &childIDs, int solutionOrdinal=0);
  void projectOldCellOntoNewCells(GlobalIndexType cellID, ElementTypePtr oldElemType,
                                  const Camellia::FieldContainer<Scalar> &oldData,
                                  const std::vector<GlobalIndexType> &childIDs, int solutonOrdinal);
  void reverseParitiesForLocalCoefficients(GlobalIndexType cellID, const vector<int> &sidesWithChangedParities, int solutionOrdinal=0);

  void setLagrangeConstraints( Teuchos::RCP<LagrangeConstraints> lagrangeConstraints);
  void setFilter(Teuchos::RCP<LocalStiffnessMatrixFilter> newFilter);
  void setReportConditionNumber(bool value);
  void setReportTimingResults(bool value);

  void computeResiduals();
  void computeErrorRepresentation();

  void discardInactiveCellCoefficients();
  double energyErrorTotal();
  const map<GlobalIndexType,double> & rankLocalEnergyError();

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
  SolutionStagePtr getSolutionStage() const;
  void setSolutionStage( SolutionStagePtr solutionStage );
  
  //! returns the term that will be added to RHS if a SolutionStage is set
  //! (this is the bf evaluated at the previous solution)
  LinearTermPtr rhsTableauModificationTerm();
  
  // ! set an RHS involving trial-space variables.  Principal application at the moment
  // ! is for goal-oriented adaptivity (hence the name), but more generally this belongs to the DPG* method.
  void setGoalOrientedRHS( LinearTermPtr ltTrial );

  TMatrixPtr<Scalar> getStiffnessMatrix();
  void setStiffnessMatrix(TMatrixPtr<Scalar> stiffness);

  TVectorPtr<Scalar> getRHSVector();
  TVectorPtr<Scalar> getLHSVector();

  void setIP( TIPPtr<Scalar>);

#if defined(HAVE_MPI) && defined(HAVE_AMESOS_MUMPS)
  void condensedSolve(TSolverPtr<Scalar> globalSolver = Teuchos::rcp(new MumpsSolver()), bool reduceMemoryFootprint = false,
                      std::set<GlobalIndexType> offRankCellsToInclude = std::set<GlobalIndexType>()); // when reduceMemoryFootprint is true, local stiffness matrices will be computed twice, rather than stored for reuse
#else
  void condensedSolve(TSolverPtr<Scalar> globalSolver = Teuchos::rcp(new TAmesos2Solver<Scalar>()), bool reduceMemoryFootprint = false,
                      std::set<GlobalIndexType> offRankCellsToInclude = std::set<GlobalIndexType>()); // when reduceMemoryFootprint is true, local stiffness matrices will be computed twice, rather than stored for reuse
#endif
//  void readFromFile(const std::string &filePath, int solutionOrdinal=0);
//  void writeToFile(const std::string &filePath, int solutionOrdinal=0);

  void save(std::string meshAndSolutionPrefix);
  static TSolutionPtr<Scalar> load(TBFPtr<Scalar> bf, std::string meshAndSolutionPrefix);
  void saveToHDF5(std::string filename);
  void loadFromHDF5(std::string filename);

  // Default of 0 adapts the number of points based on poly order
  void writeToVTK(const std::string& filePath, unsigned int num1DPts=0);
  void writeFieldsToVTK(const std::string& filePath, unsigned int num1DPts=0);
  void writeTracesToVTK(const std::string& filePath);

  // statistics accessors:
  double totalTimeApplyJumpTerms();
  double totalTimeLocalStiffness();
  double totalTimeGlobalAssembly();
  double totalTimeBCImposition();
  double totalTimeSolve();
  double totalTimeDistributeSolution();

  double meanTimeApplyJumpTerms();
  double meanTimeLocalStiffness();
  double meanTimeGlobalAssembly();
  double meanTimeBCImposition();
  double meanTimeSolve();
  double meanTimeDistributeSolution();

  double maxTimeApplyJumpTerms();
  double maxTimeLocalStiffness();
  double maxTimeGlobalAssembly();
  double maxTimeBCImposition();
  double maxTimeSolve();
  double maxTimeDistributeSolution();

  double minTimeApplyJumpTerms();
  double minTimeLocalStiffness();
  double minTimeGlobalAssembly();
  double minTimeBCImposition();
  double minTimeSolve();
  double minTimeDistributeSolution();

  void reportTimings();

  // ! offRankCellsToInclude: the ones that the condensed dof interpreter needs to know how to interpret.  (Comes up in context of h-multigrid in particular.)
  void setUseCondensedSolve(bool value, std::set<GlobalIndexType> offRankCellsToInclude = std::set<GlobalIndexType>());

  bool usesCondensedSolve() const;
  
  void writeStatsToFile(const std::string &filePath, int precision=4);

  std::vector<int> getZeroMeanConstraints();
  void setZeroMeanConstraintRho(double value);
  double zeroMeanConstraintRho();

  /*
   0: don't warn
   1: warn, but don't print values (default)
   2: print values
   */
  int warnAboutDiscontinuousBCs() const;
  void setWarnAboutDiscontinuousBCs(int outputLevel);
  
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
