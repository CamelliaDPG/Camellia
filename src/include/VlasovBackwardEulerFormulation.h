// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  VlasovBackwardEulerFormulation.h
//  Camellia
//
//  Created by Nate Roberts on May 9, 2022.
//
//

#ifndef Camellia_VlasovBackwardEulerFormulation_h
#define Camellia_VlasovBackwardEulerFormulation_h

#include "TypeDefs.h"

#include "BF.h"
#include "ConfigurationSpaceFormulation.h"
#include "GMGSolver.h"
#include "PoissonFormulation.h"
#include "RefinementObserver.h"
#include "SolutionMomentFunction.h"
#include "TensorUtilities.h"
#include "VarFactory.h"

namespace Camellia
{

class VlasovBackwardEulerFormulation
:
public RefinementObserver
{
public:
  enum TestIPChoice
  {
    VLASOV_TIME_MARCHING_GRAPH_INNER_PRODUCT,
    VLASOV_TIME_MARCHING_CONSERVATIVE_INNER_PRODUCT,
    VLASOV_TIME_MARCHING_L2_INNER_PRODUCT
  };
private:
  int _spaceDim;
  int _velocityDim;
  const bool _useSerendipityBasis; // const because we construct the mesh based on this in the constructor; we'd have to recreate the mesh when this changed.
  const bool _useConformingTraces;
  
  BFPtr _vlasovBF;
  int _delta_k;
  
  bool _useGeometricMultigrid = false;
  bool _enforceQuasiNeutrality = false;
  bool _crankNicolson = false;
  bool _enforcePositivity = false; // if true, use proximal Galerkin to enforce positivity
  
  double _alpha0 = 1.0;
  double _alphaMultiplier = 1.0;
  std::map<GlobalIndexType,double> _fluxLimitingWeightForCell; // last used weight for MPI-local cells
  std::map<GlobalIndexType,double> _expPsiForCell; // exponential of latent variable, available for MPI-local cells
  
  SolutionPtr _vlasovSoln;
  SolutionPtr _vlasovSolnPrev;      // solution at the previous fixed point iteration; can be used to check convergence of fixed point iteration
  SolutionPtr _vlasovSolnPriorTime; // solution at the prior time step; enters the RHS according to backward Euler
  
  Teuchos::RCP<ConfigurationSpaceFormulation> _configurationSpaceFormulation; // could be Poisson or Maxwell (among others)
  MeshTopologyPtr _configurationSpaceMeshTopology; // spatial mesh, created during VlasovBackwardEulerFormulation construction
  
  ParameterFunctionPtr _t = ParameterFunction::parameterFunction(0.0,"t"); // current time
  
  int _numSpecies;
  std::vector<double> _q; // index is species ordinal
  std::vector<double> _m;
  std::vector<std::string> _speciesLabels;
  std::vector<std::string> _f_names, _w_names, _f_hat_names, _tn_hat_names, _psi_names, _phi_names; // psi: latent variable for Proximal Galerkin (phi: test function for the new Proximal Galerkin equation)
  
  vector<FunctionPtr> _a; // acceleration term. Set during construction as a function of configuration-space solution.
  
  ParameterFunctionPtr _n0; // if quasi-neutrality is enforced, this is set when initial conditions are set, as the integral of f0 over the whole of phase space, divided by the measure of the configuration-space mesh.
  
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
  
  // weight for proximal-Galerkin latent variable (1/alpha weights the additional terms in the Vlasov equation)
  ParameterFunctionPtr _alphaPG;
  
  // set when inflow BCs are specified:
  ParameterFunctionPtr _strongInflowBCFlag; // 1 if strong BCs are used at inflow, 0 otherwise
  
  std::vector<Teuchos::RCP<SolutionMoments>> _solutionMoments; // one moments object per species
  std::vector<Teuchos::RCP<SolutionMoments>> _solutionMomentsPriorTime;
  
  std::vector<FunctionPtr> _fSoln; // Camellia Function defining the solution of distribution f_i (refers to _vlasovSoln)
  std::vector<FunctionPtr> _expPsiSoln; // the exponential of latent psi_i (refers to _vlasovSoln); a uniformly positive representation of f_i
  std::vector<FunctionPtr> _fDiff; // fixed-point step in Vlasov field f
  std::vector<FunctionPtr> _fSolnPriorTime; // distribution f at previous time step
  std::vector<FunctionPtr> _traceSolnPriorTime; // trace solution (tn_hat or f_hat) at previous time step

  int _minSubsteps = 3;  // minimum number of fixed-point iterations to run (alternating vlasov, configuration-space solves)
  int _maxSubsteps = 15; // maximum number of fixed-point iterations to run
  double _fixedPointTolerance = 1e-5; // relative difference in f between fixed-point substeps
  
  int _aztecOutputLevel = 0; // AZ_none -- no output.
  double _cgTol = 1e-6;
  std::vector<double> _cgTols; // one per substep; allows looser CG tolerances for early fixed-point steps
  int _maxIters = 10000;
  
  bool _runQuietly = false;
  
  Teuchos::RCP<GMGSolver> _gmgSolver;
  
  static const string S_F;       // trial variable - distribution function f
  static const string S_F_HAT;   // trace variable
  static const string S_T_N_HAT; // trace variable
  static const string S_W;       // test variable
  static const string S_PSI;     // latent variable psi
  static const string S_PHI;     // latent variable test function phi
  
  // tags for BCs:
  static const int VLASOV_TAG_SPATIAL_INFLOW = 0; // region where (v \cdot n_x) < 0
  static const int VLASOV_TAG_VXMIN          = 2; // vx min
  static const int VLASOV_TAG_VXMAX          = 3; // vx max
  static const int VLASOV_TAG_VYMIN          = 4; // vy min
  static const int VLASOV_TAG_VYMAX          = 5; // vy max
  static const int VLASOV_TAG_VZMIN          = 6; // vz min
  static const int VLASOV_TAG_VZMAX          = 7; // vz max
  
  std::vector< std::vector<int> > _iterationCounts; // outer vector index: solve number; inner: substep number.
  
  std::vector< int > _substepCounts;   // number of substeps for each call to takeTimeStep()
  
  std::vector< std::vector<double> > _fixedPointL2StepSizesVlasov; // outer vector index: solve number; inner: substep number.
  
  int _nextConfigurationSpaceTag = 0;
public:
  VlasovBackwardEulerFormulation(double q, double m, bool useSerendipityBasis,
                                 Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                 MeshTopologyPtr spatialMeshTopology,
                                 double dt,
                                 std::vector<double> velocitySpaceVertices_x,
                                 std::vector<double> velocitySpaceVertices_y,
                                 std::vector<double> velocitySpaceVertices_z,
                                 const int H1Order, const int delta_k,
                                 const int maxMomentNumber = 0,
                                 const bool useCrankNicolson = false,
                                 const bool enforcePositivity = false,
                                 const bool useConformingTraces = false);
  
  VlasovBackwardEulerFormulation(double q, double m, bool useSerendipityBasis,
                                 Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                 MeshTopologyPtr spatialMeshTopology,
                                 double dt,
                                 std::vector<double> velocitySpaceVertices_x,
                                 std::vector<double> velocitySpaceVertices_y,
                                 const int H1Order, const int delta_k,
                                 const int maxMomentNumber = 0,
                                 const bool useCrankNicolson = false,
                                 const bool enforcePositivity = false,
                                 const bool useConformingTraces = false);
  
  VlasovBackwardEulerFormulation(double q, double m, bool useSerendipityBasis,
                                 Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                 MeshTopologyPtr spatialMeshTopology,
                                 double dt,
                                 std::vector<double> velocitySpaceVertices_x,
                                 const int H1Order, const int delta_k,
                                 const int maxMomentNumber = 0,
                                 const bool useCrankNicolson = false,
                                 const bool enforcePositivity = false,
                                 const bool useConformingTraces = false);
  
  VlasovBackwardEulerFormulation(std::vector<double> q, std::vector<double> m, std::vector<std::string> speciesLabels,
                                 bool useSerendipityBasis,
                                 Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                 MeshTopologyPtr spatialMeshTopology,
                                 double dt,
                                 std::vector<double> velocitySpaceVertices_x,
                                 std::vector<double> velocitySpaceVertices_y,
                                 std::vector<double> velocitySpaceVertices_z,
                                 const int H1Order, const int delta_k,
                                 const int maxMomentNumber = 0,
                                 const bool useCrankNicolson = false,
                                 const bool enforcePositivity = false,
                                 const bool useConformingTraces = false);
  
  VlasovBackwardEulerFormulation(std::vector<double> q, std::vector<double> m, std::vector<std::string> speciesLabels,
                                 bool useSerendipityBasis,
                                 Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                 MeshTopologyPtr spatialMeshTopology,
                                 double dt,
                                 std::vector<double> velocitySpaceVertices_x,
                                 std::vector<double> velocitySpaceVertices_y,
                                 const int H1Order, const int delta_k,
                                 const int maxMomentNumber = 0,
                                 const bool useCrankNicolson = false,
                                 const bool enforcePositivity = false,
                                 const bool useConformingTraces = false);
  
  VlasovBackwardEulerFormulation(std::vector<double> q, std::vector<double> m, std::vector<std::string> speciesLabels,
                                 bool useSerendipityBasis,
                                 Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                 MeshTopologyPtr spatialMeshTopology,
                                 double dt,
                                 std::vector<double> velocitySpaceVertices_x,
                                 const int H1Order, const int delta_k,
                                 const int maxMomentNumber = 0,
                                 const bool useCrankNicolson = false,
                                 const bool enforcePositivity = false,
                                 const bool useConformingTraces = false);

  //! Returns the acceleration term, defined on the configuration space mesh, as an array of length velocityDim().
    vector<FunctionPtr> a();

  //! load from checkpoint
  void loadFromCheckpoint(const std::string &restartDir);
  
  //! save checkpoint
  void saveCheckpoint(const std::string &checkpointDir);
  
  //! take time step.  For externally-driven adaptive solves, one can instead call startTimeStep(); perform refinements on the initial-substep solver, and then call finishTimeStep() to do the fixed-point iteration.  (This is just one of many possible approaches.)
  void takeTimeStep();
  
  //! copy the current-time solution(s) to previous-time solution, starting a new time step.  Called from takeTimeStep().
  void startTimeStep();
  
  //! Perform a fixed-point iteration according to fixed-point parameters (tolerance, max and min substep counts).  Called from takeTimeStep().
  void finishTimeStep();

  //! current time.
  ParameterFunctionPtr t() const;
  
  //! solve for the configuration-space solution, based on current Vlasov solution.
  void configurationSpaceSolve();
  
  //! get the size of the (next) time step.
  double timeStep() const;
  
  //! set the size of the (next) time step.
  void setTimeStep(double dt);
  
  //! clear previously computed solution
  void clearSolution();
  
  // field variables:
  VarPtr f(int speciesOrdinal=0);
  VarPtr psi(int speciesOrdinal=0); // latent variable (for proximal Galerkin, when enforcePositivity is set to true)
  
  // traces:
  VarPtr f_hat(int speciesOrdinal=0);
  VarPtr t_n_hat(int speciesOrdinal=0);
  
  // test variables:
  VarPtr w(int speciesOrdinal=0);
  
  VarPtr phi(int speciesOrdinal=0); // latent test variable (for proximal Galerkin, when enforcePositivity is set to true)
  
  MeshPtr vlasovMesh();
  MeshPtr configurationSpaceMesh();
  
  Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation();
  
  Teuchos::RCP<SolutionMoments> solutionMoments(int i=0);
  
  //! SolutionMoments object for the prior-time solution.  Note that this does not get updated automatically in the course of a solve (it is not used internally).
  Teuchos::RCP<SolutionMoments> solutionMomentsPriorTime(int i=0);
  
  //! The Vlasov solution at the current time step.
  SolutionPtr vlasovSolution();
  
  //! The Vlasov solution from the previous substep.
  SolutionPtr vlasovSolutionPrevious();
  
  //! The Vlasov solution at the previous time step.
  SolutionPtr vlasovSolutionPriorTime();
  
  SolutionPtr configurationSpaceSolution();
    
  bool hasL2Trace() const;
  
  int maxSubstepsTaken() const;
  int minSubstepsTaken() const;
  double averageSubstepsTaken() const;
  
  //! Set the initial alpha value and multiplier for Proximal Galerkin.  Only applies when enforcePositivity was set to true at construction.  alpha is set to alpha0 at the start of each time step, and multiplied by alphaMultiplier during each substep.
  void setAlpha(const double alpha0, const double alphaMultiplier);
  
  //! Returns a (possibly dynamic) function for the alpha value in Proximal Galerkin.
  FunctionPtr alpha();
  
  //! Returns the exponential of the latent variable psi for the specified cell (which must be active and owned by this MPI rank).  Only valid if positivity enforcement is turned on.
  double expPsiForCell(GlobalIndexType cellID);
  
  //! Returns the flux-limiting weight for the specified cell (which must be active and owned by this MPI rank).  The weight is used to weight f and the exponential of the latent variable, psi.  A 1.0 weight means that f is given full weight (no limiting required); a 0 weight means f does not contribute at all.
  double fluxLimitingWeightForCell(GlobalIndexType cellID);
  
  //! Set the initial conditions without setting inflow conditions (relevant for the periodic case).  Note: configuration-space BCs should be set before this is called; this will perform a configuration-space solve.
  void setInitialConditions(FunctionPtr f0, bool imposeInflowWeakly);
  
  //! Set the initial, boundary conditions for Vlasov (relevant for the non-periodic case).  Note: configuration-space BCs should be set before this is called; this will perform a configuration-space solve.  Calls the four-argument setInitialAndInflowConditions() with imposeInflowWeakly = _enforcePositivity.
  void setInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_in,
                                     double timeRampForInflowConditions = 0);
  
  //! Set the initial, boundary conditions for Vlasov (relevant for the non-periodic case).  Note: configuration-space BCs should be set before this is called; this will perform a configuration-space solve.
  void setInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_in,
                                     double timeRampForInflowConditions,
                                     bool imposeInflowWeakly);
  
  //! Set the initial conditions without setting inflow conditions (relevant for the periodic case): multi-species version.  Note: configuration-space BCs should be set before this is called; this will perform a configuration-space solve.
  void setInitialConditions(std::vector<FunctionPtr> f0, bool imposeInflowWeakly);
  
  //! Set the initial, boundary conditions for Vlasov (relevant for the non-periodic case): multi-species version.  Note: configuration-space BCs should be set before this is called; this will perform a configuration-space solve.  Calls the four-argument setInitialAndInflowConditions() with imposeInflowWeakly = _enforcePositivity.
  void setInitialAndInflowConditions(std::vector<FunctionPtr> f0, std::vector<FunctionPtr> f_in,
                                     double timeRampForInflowConditions = 0);
  
  //! Set the initial, boundary conditions for Vlasov (relevant for the non-periodic case): multi-species version.  Note: configuration-space BCs should be set before this is called; this will perform a configuration-space solve.
  void setInitialAndInflowConditions(std::vector<FunctionPtr> f0, std::vector<FunctionPtr> f_in,
                                     double timeRampForInflowConditions,
                                     bool imposeInflowWeakly);
  
  void setTestIP(VlasovBackwardEulerFormulation::TestIPChoice ipChoice);
  
  void setTestIP(std::string ipChoiceString);
  
  void setEnforceQuasiNeutrality(bool enforceQuasiNeutrality);
  
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
  
  const std::vector< std::vector<int> > & iterationCounts() const;
    
  void setUseGeometricMultigrid(bool useGeometricMultigrid);
  
  //! Constructs a GMG solver for the current Vlasov mesh.  Does not update the internal _gmgSolver member; for that, see reconstructGMGSolver().
  Teuchos::RCP<GMGSolver> constructGMGSolver();
  
  //! If useGeometricMultigrid is true, sets internal _gmgSolver to the result of constructGMGSolver(), overwriting any previously set _gmgSolver.  Appropriate to call after performing adaptive refinements on the Vlasov mesh.
  void reconstructGMGSolver();
  
  //! Returns the member GMG solver; may be null if it has not yet been constructed, or if useGeometricMultigrid is false.
  Teuchos::RCP<GMGSolver> gmgSolver();
  
  /** \brief  Create a tag for an indicated region of configuration space.
      \param  regionToMatch   [in]  - A SpatialFilterPtr that matches the region where BCs should be enforced.
      \return tag for the configuration-space mesh; can be used to set BCs.
   
   Note that the matching is performed immediately; this is not an appropriate method for setting a dynamic region of the mesh (as when the inflow region depends on the solution).  For dynamic mesh regions, SpatialFilter-based BCs are likely better suited than tags.
  */
  int createConfigurationMeshSideSetTag(SpatialFilterPtr regionToMatch);
  
  MeshTopologyPtr configurationSpaceMeshTopology();
  
  /** \brief  Create a tag for an indicated region of configuration space.
   \param  tag   [in]  - A tag representing the region in configuration space where the BC should be enforced.
   \param  configVariable   [in]  - the variable on which the BC should be enforced.
   \param  dirichletValue   [in]  - the value of the variable within the matched region.
   
   \see B_configurationSpaceTrace(), E_configurationSpaceTrace(). VE_configurationSpaceTrace().
  */
  void addConfigurationSpaceBC(const int tag, VarPtr configVariable, FunctionPtr dirichletValue);
  
  /** \brief Reset the BCs for the configuration space mesh; can be used prior to fresh calls to addConfigurationSpaceBC().
  */
  void clearConfigurationSpaceBCs();
  
  /** \brief  A VarPtr for the B trace variable component specified, valid on the configuration-space mesh.
   \param  zeroBasedComp   [in]  - The component of B that is being requested; use 0 for B_x, 1 for B_y, and 2 for B_z.
   \return VarPtr for the specified B component; Teuchos::null if the ConfigurationSpaceFormulation does not include the specified component of B as a variable.
  */
  VarPtr B_configurationSpaceTrace(const int zeroBasedComp);
  
  /** \brief  A VarPtr for the E trace variable component specified, valid on the configuration-space mesh.
   \param  zeroBasedComp   [in]  - The component of E that is being requested; use 0 for E_x, 1 for E_y, and 2 for E_z.
   \return VarPtr for the specified B component; Teuchos::null if the ConfigurationSpaceFormulation does not include the specified component of B as a variable.
  */
  VarPtr E_configurationSpaceTrace(const int zeroBasedComp);
  
  /** \brief  A VarPtr for the VE trace variable (the potential), valid on the configuration-space mesh.
    \return VarPtr for the VE; Teuchos::null if the ConfigurationSpaceFormulation does not include a VE trace as a variable.
  */
  VarPtr VE_configurationSpaceTrace();
  
  int tagVlasovSpatialInflow() const;
  int tagVlasovMaxVelocity(int zeroBasedComp) const;
  int tagVlasovMinVelocity(int zeroBasedComp) const;
  
  double q(const int speciesOrdinal) const;
  
  double m(const int speciesOrdinal) const;
  
  /** \brief  Returns a Function representing the specified component of configuration space. */
  FunctionPtr x_i(const int zeroBasedComp) const;
  
  /** \brief  Returns a Function representing the specified velocity-space coordinate. */
  FunctionPtr v_i(const int zeroBasedComp) const;
  
  /** \brief Refines both phase-space mesh and configuration-space meshes uniformly the specified number of times. */
  void refineUniformly(const int numRefinements);
  
  //! RefinementObserver method.  Will be called when the Vlasov mesh is refined.
  void didHRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool didRepartitionAndRebuild) override;
  
  //! RefinementObserver method.  Will be called when the Vlasov mesh is p-refined.
  void didPRefine(const set<GlobalIndexType> &cellIDs, int pToAdd) override;
  
  //! RefinementObserver method.  Will be called when the Vlasov mesh is repartitioned.
  void didRepartition(MeshTopologyPtr meshTopo) override;
  
  //! RefinementObserver method.  Will be called when the Vlasov mesh is unrefined.
  void didHUnrefine(const set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild) override;
  
  //! If true, will not output details of fixed point iteration to console
  void setRunQuietly(bool runQuietly);
};

}

#endif
