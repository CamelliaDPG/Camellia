// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  Boltzmann1D1VFormulation.h
//  Camellia
//
//  Created by Nate Roberts on March 3, 2020.
//
//

#ifndef Camellia_Boltzmann1D1VFormulation_h
#define Camellia_Boltzmann1D1VFormulation_h

#include "TypeDefs.h"

#include "BF.h"
#include "GMGSolver.h"
#include "PhasePhysicalMeshMap.h"
#include "PoissonFormulation.h"
#include "QuadraturePointFunction.h"
#include "SolutionMomentFunction.h"
#include "SolutionStage.h"
#include "TensorUtilities.h"
#include "VarFactory.h"

namespace Camellia
{

  class Boltzmann1D1VFormulation : public SolutionStageObserver
{
public:
  enum Boltzmann1D1VFormulationChoice
  {
    ULTRAWEAK_EXPLICIT_L2_TRACE,
    ULTRAWEAK_EXPLICIT_H1_TRACE,
    ULTRAWEAK_IMPLICIT_L2_TRACE,
    ULTRAWEAK_IMPLICIT_H1_TRACE
  };
private:
  Boltzmann1D1VFormulationChoice _formulationChoice;
  bool _usePeriodicBCs;
  bool _enforceQuasiNeutrality;
  
  Teuchos::RCP<PoissonFormulation> _poissonForm; // used by Gauss
  
  BFPtr _boltzmannBF;
  BFPtr _gaussBF;
  int _delta_k;
  int _tn_enrichment = 0;
  
  bool _useGeometricMultigrid = false;
  
  SolutionPtr _boltzmannSoln;
  SolutionPtr _boltzmannSolnPrevSubstep; // for fixed-point iteration, the current-time solution from a previous substep
  SolutionPtr _boltzmannSolnPrev;
  
  double _gaussMeshMeasure;
  double _boltzmannMeshMeasure;
  
  bool _haveTableau = false;
  
  SolutionPtr _gaussSoln;
  SolutionPtr _gaussSolnPrevSubstep; // for fixed-point iteration, the current-time solution from a previous substep
  double _t = 0; // current time
  
  //! on each refinement, refine any element with error greater than threshold times the maximum element error.  Values > 1.0 mean no refinement
  double _refinementThreshold   = 1.1;
  //! unrefinements will be performed to ensure that the mesh has no more than _maxElements active elements
  int _maxElements = std::numeric_limits<int>::max();
  
  //! on the last refinement, what was the maximum elementwise error? (-1 means have not done a refinement)
  double _lastRefinementMaxElementError = -1.0;
  
  int _maxRefinementLevel = std::numeric_limits<int>::max();
  
  RefinementStrategyPtr _refStrategy;
  
  //! If true, only refine when max element error is greater than the least error that would have induced a refinement in the last round of refinements
  bool _gateRefinements = false;
  
  bool _performRefinementsDuringTimeSteps = true;
  
  // set by user:
  ParameterFunctionPtr _dt;
  ParameterFunctionPtr _q;
  ParameterFunctionPtr _eps0;
  ParameterFunctionPtr _m;
  
  // set during solve process:
  ParameterFunctionPtr _n0; // integral of f over the whole 1D1V space
  
  FunctionPtr _fSoln; // Camellia Function defining the solution of distribution f (refers to _vlasovSoln)
  FunctionPtr _ESoln; // Camellia Function defining the solution of electric field E (refers to _poissonSoln)
  
  FunctionPtr _fDiff; // fixed-point step in Vlasov field f
  FunctionPtr _EDiff; // fixed-point step in electric field E
  
  int _minSubstepsForImplicit = 3;  // minimum number of fixed-point iterations to run (alternating vlasov, poisson solves)
  int _maxSubstepsForImplicit = 15; // maximum number of fixed-point iterations to run
  double _fixedPointTolerance = 1e-5; // relative difference in f between fixed-point substeps
  
  Teuchos::RCP<SolutionMoments> _solutionMoments;
  FunctionPtr _f_velocityIntegral; // integral of f over all velocity space
  FunctionPtr _E_extruded;  // E solution extended to the Boltzmann mesh
  
  // TODO: consider whether these are still useful; PhasePhysicalMeshMap1D1V should provide everything we need (and if it doesn't, make it so)
  std::map<GlobalIndexType, std::vector<GlobalIndexType> > _spatialCellIDToPhaseSpaceCellIDs;
  std::map<GlobalIndexType, std::set<GlobalIndexType> > _phaseSpaceCellIDToSpatialCellID;
  
  Teuchos::RCP<PhasePhysicalMeshMap> _phaseSpacePhysicalMeshMap;
  
  double _cgTol = 1e-6;
  std::vector<double> _cgTols; // one per substep; allows looser CG tolerances for early fixed-point steps
  int _maxIters = 10000;
  
  int _aztecOutputLevel = 0; // AZ_none -- no output.
  Teuchos::RCP<GMGSolver> _gmgSolver;
  
  // Boltzmann itself (discretization is 2D because 1D1V):
  static const string S_F;       // trial variable - distribution function f
  static const string S_F_HAT;   // trace variable
  static const string S_T_N_HAT; // trace variable
  static const string S_W;       // test variable
  
  // Gauss's law:
  static const string S_E;     // trial variable - electric field E
  static const string S_E_HAT; // trace variable - trace of E in H^1/2
  static const string S_TAU;   // test variable
  
  // tags for BCs:
  static const int GAUSS_TAG_X0 = 0; // left  (x=0)
  static const int GAUSS_TAG_X1 = 1; // right (x=x_max)
  
  static const int BOLTZMANN_TAG_INFLOW_LEFT  = 0; // left  (x=0,     v > 0)
  static const int BOLTZMANN_TAG_INFLOW_RIGHT = 1; // right (x=x_max, v < 0)
  static const int BOLTZMANN_TAG_VMIN         = 2; // bottom
  static const int BOLTZMANN_TAG_VMAX         = 3; // top
  
  std::vector< std::vector<int> > _iterationCounts; // outer vector index: solve number; inner: substep number.
  
  std::vector< std::vector<double> > _fixedPointL2StepSizesVlasov; // outer vector index: solve number; inner: substep number.
  std::vector< std::vector<double> > _fixedPointL2StepSizesPoisson; // outer vector index: solve number; inner: substep number.
  
  void updateElectricField();
public:
  //! Traces in Vlasov by default match the H^1 order in polynomial order (i.e., they are one order higher than volumetric variable).  To use a higher order, set vlasovTraceEnrichment to 1 or more.
  Boltzmann1D1VFormulation(Boltzmann1D1VFormulationChoice formulationChoice, double q, double eps0, double m, double dt,
                           const double x0, const double xWidth, const int xElements,
                           const double v0, const double vWidth, const int vElements,
                           const int H1Order, const int delta_k, const bool usePeriodicBCs,
                           const bool enforceQuasiNeutrality, ERKTableauPtr erkTableau,
                           const int vlasovTraceEnrichment,
                           const int numUniformRefinements);

  //! set initial conditions.  Solves for initial E, E_hat, and sets t=0;
  void setInitialConditions(FunctionPtr f0);
  
  //! take time step.
  void takeTimeStep();
  
  void setTimeStep(double dt);
  
  //! current time
  double t() const;
  
  // field variables:
  VarPtr f();  // Boltzmann
  VarPtr E();  // Gauss
  VarPtr VE();  // Gauss

  // traces:
  VarPtr f_hat(); // Boltzmann
  VarPtr t_n_hat(); // Boltzmann
  VarPtr E_hat();   // Gauss electric field trace
  VarPtr VE_hat();  // Gauss electric potential trace

  // test variables:
  VarPtr w();   // Boltzmann
  VarPtr tau(); // Gauss
  
  MeshPtr boltzmannMesh();
  MeshPtr gaussMesh();
  
  Teuchos::RCP<PhasePhysicalMeshMap> phaseSpacePhysicalMeshMap();
  
  SolutionPtr boltzmannSolution();
  SolutionPtr boltzmannPreviousSolution();
  SolutionPtr gaussSolution();
  
  FunctionPtr E_extruded();
  
  bool usesImplicitSolve() const;
  bool hasL2Trace() const;
  
  const std::vector< std::vector<int> > & iterationCounts() const;
  
  //! Set the boundary conditions for Boltzmann (relevant for the non-periodic case).
  void setBoltzmannInflow(FunctionPtr f_left, FunctionPtr f_right);
  
  //! Set the boundary conditions for Gauss (relevant for the non-periodic case).
  void setGaussBCs(FunctionPtr VE_left, FunctionPtr VE_right);
  
  /** \brief  Sets thresholds for adaptivity
      \param  refinementThreshold   [in]  - on each refinement, refine any element with error greater than refinementThreshold times the maximum element error.  Values > 1.0 mean no refinement.
      \param  maxElements [in]  - maximum number of active elements allowed in the mesh (unrefinements will be performed to maintain this rule)
      \param  maxRefinementLevel [in]  - maximum levels of refinement
      \param  gateRefinements [in] - if true, only refine when max element error is greater than the least error that would have induced a refinement in the last round of refinements (otherwise, refine once per time step)
  */
  void setRefinementThresholds(double refinementThreshold, int maxElements, int maxRefinementLevel, ErrorIndicatorPtr unrefinementErrorIndicator = Teuchos::null,
                               bool gateRefinements = false);
  
  /** \brief  Sets a fixed-count refinement strategy; will aim to refine the specified number of elements on each time step.
  */
  void setFixedCountRefinementStrategy(int count);
  
  //! perform refinements according to settings given during call to setRefinementThreshold().  (If gateRefinements was true, then refinements might not actually take place.)
  void performRefinements();
  
  //! Defaults to true; set to false if you want to manually call performRefinements after the time step is taken.  Useful if you want to examine the solution before a refinement takes place.
  void setPerformRefinementsDuringTimeSteps(bool value);
  
  RefinementStrategyPtr refinementStrategy();
  
  /** \brief  Set the tolerance for the fixed point iteration; relevant for implicit solves.
      \param  fTol   [in]  - the target relative difference in f between one substep and the next, as measured in the L^2 norm.
  */
  void setFixedPointTolerance(const double fTol);
  
  /** \brief  Set the minimum number of substeps to take for the fixed point iteration; relevant for implicit solves.
      \param  minSubsteps   [in]  - the minimum number of substeps to take for the fixed point solve.
  */
  void setFixedPointMinSubstepsForImplicit(const int minSubsteps);
  
  /** \brief  Set the maximum number of substeps to take for the fixed point iteration; relevant for implicit solves.
      \param  maxSubsteps   [in]  - the maximum number of substeps to take for the fixed point solve.
  */
  void setFixedPointMaxSubstepsForImplicit(const int maxSubsteps);
  
  /** \brief  Set the tolerance for the iterative solve (relevant for geometric multigrid).
      \param  cgTols   [in]  - the tolerance for the conjugate gradient solve in each fixed-point step; used in the stopping criterion for the CG iteration
  */
  
  /** \brief  Set the level of console output.
      \param  azOutputLevel   [in]  - Corresponds to the Aztec output level (though Belos is doing the output); here, 0 means no output (the default); -1 means final residual + warnings; -2 means summary, final residual, and warnings; and -3 means only print warnings.  Positive number n means print out residual every n steps.
  */
  void setIterativeSolveOutputLevel(int azOutputLevel);
  
  /** \brief  Set the tolerance for the iterative solve (relevant for geometric multigrid).
      \param  cgTol   [in]  - the tolerance for the conjugate gradient solve; used in the stopping criterion for the CG iteration
  */
  void setIterativeSolveTolerance(const double cgTol);
  
  /** \brief  Set the tolerance for the iterative solve (relevant for geometric multigrid).
      \param  cgTol   [in]  - the tolerance for the conjugate gradient solve; used in the stopping criterion for the CG iteration
  */
  void setIterativeSolveTolerances(const std::vector<double> cgTols);
  
  /** \brief  Set the maximum number of iterations for the iterative solve (relevant for geometric multigrid).
      \param  maxIters   [in]  - the maximum number of conjugate gradient iterations to take; will stop if either maxIters are reached, or the relative residual falls below the CG tolerance.
  */
  void setIterativeSolveMaxIterations(const int maxIters);
  
  //! notification that the boltzmann solve has set the RK stage:
  void setStage(const int stageNumber);
  
  Teuchos::RCP<SolutionMoments> solutionMoments();
  
  //! map from gauss mesh cellIDs to corresponding cellIDs in boltzmann mesh
  const std::map<GlobalIndexType, std::vector<GlobalIndexType> > &spatialCellIDToPhaseSpaceCellIDsMap();
  
  //! map that is reverse of the above; there is exactly one spatial cell for each phase space cell
  const std::map<GlobalIndexType, std::set<GlobalIndexType> > &phaseSpaceCellIDToSpatialCellIDs();
  
  int tagBoltzmannLeftInflow() const;
  int tagBoltzmannRightInflow() const;
  int tagBoltzmannMaxVelocity() const;
  int tagBoltzmannMinVelocity() const;
  
  //! x-velocity function
  FunctionPtr vx() const;
  
  void setUseGeometricMultigrid(bool useGeometricMultigrid);
  
  Teuchos::RCP<GMGSolver> constructGMGSolver();
};

}

#endif
