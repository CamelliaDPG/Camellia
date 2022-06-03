// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  VlasovPoissonSpaceTimeFormulation.h
//  Camellia
//
//  Created by Nate Roberts on March 3, 2020.
//
//

#ifndef Camellia_VlasovPoissonSpaceTimeFormulation_h
#define Camellia_VlasovPoissonSpaceTimeFormulation_h

#include "TypeDefs.h"

#include "BF.h"
#include "GMGSolver.h"
#include "PoissonFormulation.h"
#include "SolutionMomentFunction.h"
#include "TensorUtilities.h"
#include "VarFactory.h"

namespace Camellia
{

class VlasovPoissonSpaceTimeFormulation
{
public:
  enum VlasovPoissonSpaceTimeFormulationChoice
  {
    ULTRAWEAK
  };
private:
  VlasovPoissonSpaceTimeFormulationChoice _formulationChoice;
  bool _usePeriodicBCs;
  bool _enforceQuasiNeutrality;
  
  BFPtr _vlasovBF;
  BFPtr _poissonBF;
  int _delta_k;
  int _tn_enrichment;
  
  bool _useGeometricMultigrid = false;
  
  SolutionPtr _vlasovSoln;
  SolutionPtr _vlasovSolnPrev; // solution at the previous fixed point iteration; can be used to check convergence of fixed point iteration
  
  double _poissonMeshMeasure;
  double _vlasovMeshMeasure;
    
  SolutionPtr _poissonSoln;
  SolutionPtr _poissonSolnPrev; // solution at the previous fixed point iteration; can be used to check convergence of fixed point iteration
  
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
  
  // set by user:
  ParameterFunctionPtr _q;
  ParameterFunctionPtr _eps0;
  ParameterFunctionPtr _m;
  
  // set during solve process:
  ParameterFunctionPtr _n0; // integral of f over the whole 1D1V space
  
  Teuchos::RCP<SolutionMoments> _solutionMoments;
  FunctionPtr _f_velocityIntegral; // integral of f over all velocity space
  FunctionPtr _E_extruded;  // E solution extended to the Vlasov mesh
  
  FunctionPtr _fSoln; // Camellia Function defining the solution of distribution f (refers to _vlasovSoln)
  FunctionPtr _ESoln; // Camellia Function defining the solution of electric field E (refers to _poissonSoln)
  
  FunctionPtr _fDiff; // fixed-point step in Vlasov field f
  FunctionPtr _EDiff; // fixed-point step in electric field E
    
  int _minSubsteps = 3;  // minimum number of fixed-point iterations to run (alternating vlasov, poisson solves)
  int _maxSubsteps = 15; // maximum number of fixed-point iterations to run
  double _fixedPointTolerance = 1e-5; // relative difference in f between fixed-point substeps
  
  int _aztecOutputLevel = 0; // AZ_none -- no output.
  double _cgTol = 1e-6;
  std::vector<double> _cgTols; // one per substep; allows looser CG tolerances for early fixed-point steps
  int _maxIters = 10000;
  
  // Vlasov itself (discretization is 2D because 1D1V):
  static const string S_F;       // trial variable - distribution function f
  static const string S_F_HAT;   // trace variable
  static const string S_T_N_HAT; // trace variable
  static const string S_W;       // test variable
  
  // Gauss's law:
  static const string S_E;      // trial variable - electric field E
  static const string S_VE;     // trial variable - potential VE
  static const string S_E_HAT;  // trace variable - trace of E in H^1/2
  static const string S_VE_HAT; // trace variable - trace of E in H^1/2
  static const string S_V_POISSON;   // scalar test variable
  static const string S_TAU_POISSON; // vector test variable
  
  // tags for BCs:
  static const int POISSON_TAG_X0 = 0; // left  (x=0)
  static const int POISSON_TAG_X1 = 1; // right (x=x_max)
  
  static const int VLASOV_TAG_INFLOW_LEFT  = 0; // left  (x=0,     v > 0)
  static const int VLASOV_TAG_INFLOW_RIGHT = 1; // right (x=x_max, v < 0)
  static const int VLASOV_TAG_VMIN         = 2; // bottom
  static const int VLASOV_TAG_VMAX         = 3; // top
  static const int VLASOV_TAG_TMIN         = 4; // initial time
  
  std::vector< std::vector<int> > _iterationCounts; // outer vector index: solve number; inner: substep number.
  
  std::vector< std::vector<double> > _fixedPointL2StepSizesVlasov; // outer vector index: solve number; inner: substep number.
  std::vector< std::vector<double> > _fixedPointL2StepSizesPoisson; // outer vector index: solve number; inner: substep number.
  
  void updateElectricField();
public:
  VlasovPoissonSpaceTimeFormulation(VlasovPoissonSpaceTimeFormulationChoice formulationChoice,
                                    double q, double eps0, double m,
                                    const double x0, const double xWidth, const int xElementsCoarse,
                                    const double t0, const double tFinal, const int tElementsCoarse,
                                    const double v0, const double vWidth, const int vElementsCoarse,
                                    const int H1Order, const int delta_k,
                                    const bool usePeriodicBCs, const bool enforceQuasiNeutrality,
                                    const int vlasovTraceEnrichment,
                                    const double firstTimeDivisionSize = 0.0); // firstTimeDivisionSize: set this to a positive value to introduce an extra time division from t=0 to firstTimeDivisionSize.  tFinal will still be the final time.

  //! take time step.
  void solve();
  
  //! clear previously computed solution
  void clearSolution();
  
  // field variables:
  VarPtr f();  // Vlasov
  VarPtr E();  // Gauss (Poisson)
  VarPtr VE();  // Poisson electric potential

  // traces:
  VarPtr f_hat(); // Vlasov
  VarPtr t_n_hat(); // Vlasov
  VarPtr E_hat();   // Gauss electric field trace
  VarPtr VE_hat();  // Gauss electric potential trace

  // test variables:
  VarPtr w();   // Vlasov
  VarPtr tau(); // Gauss
  
  // x-velocity function:
  FunctionPtr vx() const;
  
  MeshPtr vlasovMesh();
  MeshPtr poissonMesh();
  
  Teuchos::RCP<SolutionMoments> solutionMoments();
  
  SolutionPtr vlasovSolution();
  SolutionPtr poissonSolution();
  
  FunctionPtr E_extruded();
  
  bool hasL2Trace() const;
  
  //! Set the initial conditions for Vlasov (relevant for the periodic case).
  void setVlasovInitialConditions(FunctionPtr f0);
  
  //! Set the initial, boundary conditions for Vlasov (relevant for the non-periodic case).
  void setVlasovInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_left, FunctionPtr f_right);
  
  //! Set the potential boundary conditions on the physical mesh (relevant for the non-periodic case).
  void setPotentialBCs(FunctionPtr VE_left, FunctionPtr VE_right);
  
  /** \brief  Set the tolerance for the fixed point iteration.
      \param  fTol   [in]  - the target relative difference in f between one substep and the next, as measured in the L^2 norm.
  */
  void setFixedPointTolerance(const double fTol);
  
  /** \brief  Set the minimum number of substeps to take for the fixed point iteration.
      \param  minSubsteps   [in]  - the minimum number of substeps to take for the fixed point solve.
  */
  void setFixedPointMinSubsteps(const int minSubsteps);
  
  /** \brief  Set the maximum number of substeps to take for the fixed point iteration.
      \param  maxSubsteps   [in]  - the maximum number of substeps to take for the fixed point solve.
  */
  void setFixedPointMaxSubsteps(const int maxSubsteps);
  
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
  
  void setIterativeSolveTolerances(const std::vector<double> cgTols);
  
  /** \brief  Set the maximum number of iterations for the iterative solve (relevant for geometric multigrid).
      \param  maxIters   [in]  - the maximum number of conjugate gradient iterations to take; will stop if either maxIters are reached, or the relative residual falls below the CG tolerance.
  */
  void setIterativeSolveMaxIterations(const int maxIters);
  
  /** \brief  Sets thresholds for adaptivity
      \param  refinementThreshold   [in]  - on each refinement, refine any element with error greater than refinementThreshold times the maximum element error.  Values > 1.0 mean no refinement.
      \param  maxElements [in]  - maximum number of active elements allowed in the mesh (unrefinements will be performed to maintain this rule)
      \param  maxRefinementLevel [in]  - maximum levels of refinement
      \param  gateRefinements [in] - if true, only refine when max element error is greater than the least error that would have induced a refinement in the last round of refinements (otherwise, refine once per time step)
  */
  void setRefinementThresholds(double refinementThreshold, int maxElements, int maxRefinementLevel, ErrorIndicatorPtr unrefinementErrorIndicator = Teuchos::null,
                               bool gateRefinements = false);
  
  void setUseGeometricMultigrid(bool useGeometricMultigrid);
  
  Teuchos::RCP<GMGSolver> constructGMGSolver();
  
  int tagVlasovInitialTime() const;
  int tagVlasovLeftInflow() const;
  int tagVlasovRightInflow() const;
  int tagVlasovMaxVelocity() const;
  int tagVlasovMinVelocity() const;
};

}

#endif
