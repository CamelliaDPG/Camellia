// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  VlasovSpaceTimeFormulation.h
//  Camellia
//
//  Created by Nate Roberts on January 11, 2022.
//
//

#ifndef Camellia_VlasovSpaceTimeFormulation_h
#define Camellia_VlasovSpaceTimeFormulation_h

#include "TypeDefs.h"

#include "BF.h"
#include "ConfigurationSpaceFormulation.h"
#include "GMGSolver.h"
#include "PoissonFormulation.h"
#include "SolutionMomentFunction.h"
#include "TensorUtilities.h"
#include "VarFactory.h"

namespace Camellia
{

  class VlasovSpaceTimeFormulation
{
private:
  int _spaceDim;
  int _velocityDim;
  bool _useSerendipityBasis;
  
  BFPtr _vlasovBF;
  int _H1Order;
  int _delta_k;
  
  bool _useGeometricMultigrid = false;
  bool _enforceQuasiNeutrality = false;
  
  SolutionPtr _vlasovSoln;
  SolutionPtr _vlasovSolnPrev; // solution at the previous fixed point iteration; can be used to check convergence of fixed point iteration
  
  Teuchos::RCP<ConfigurationSpaceFormulation> _configurationSpaceFormulation; // could be Poisson or Maxwell (among others)
  MeshTopologyPtr _configurationSpaceMeshTopology; // space-time mesh, created during VlasovSpaceTimeFormulation construction
  
  std::vector<SpatialFilterPtr> _configurationSpaceSideSetFilters; // we append one each time that createConfigurationMeshSideSetTag() is called.
  std::vector< std::tuple<int,VarPtr,FunctionPtr> > _configurationSpaceBCs; // (tag, var, value): arguments for each call to addConfigurationSpaceBC().
  
  double _q;
  double _m;
    
  Teuchos::RCP<SolutionMoments> _solutionMoments;
  ParameterFunctionPtr _n0; // if quasi-neutrality is enforced, this is set when initial conditions are set, as the integral of f0 over the whole of phase space, divided by the measure of the configuration-space mesh.
  
  FunctionPtr _fSoln; // Camellia Function defining the solution of distribution f (refers to _vlasovSoln)
  FunctionPtr _fDiff; // fixed-point step in Vlasov field f

  int _minSubsteps = 3;  // minimum number of fixed-point iterations to run (alternating vlasov, configuration-space solves)
  int _maxSubsteps = 15; // maximum number of fixed-point iterations to run
  double _fixedPointTolerance = 1e-5; // relative difference in f between fixed-point substeps
  
  int _aztecOutputLevel = 0; // AZ_none -- no output.
  double _cgTol = 1e-6;
  std::vector<double> _cgTols; // one per substep; allows looser CG tolerances for early fixed-point steps
  int _maxIters = 10000;
  
  static const string S_F;       // trial variable - distribution function f
  static const string S_F_HAT;   // trace variable
  static const string S_T_N_HAT; // trace variable
  static const string S_W;       // test variable
  
  // tags for BCs:
  static const int VLASOV_TAG_SPATIAL_INFLOW = 0; // region where (v \cdot n_x) < 0
  static const int VLASOV_TAG_TMIN           = 1; // initial time
  static const int VLASOV_TAG_VXMIN          = 2; // vx min
  static const int VLASOV_TAG_VXMAX          = 3; // vx max
  static const int VLASOV_TAG_VYMIN          = 4; // vy min
  static const int VLASOV_TAG_VYMAX          = 5; // vy max
  static const int VLASOV_TAG_VZMIN          = 6; // vz min
  static const int VLASOV_TAG_VZMAX          = 7; // vz max
  
  std::vector< std::vector<int> > _iterationCounts; // outer vector index: solve number; inner: substep number.
  std::vector< std::vector<double> > _fixedPointL2StepSizesVlasov; // outer vector index: solve number; inner: substep number.
  
  static const int CONFIGURATION_TAG_TMIN    = 0; // initial time
  int _nextConfigurationSpaceTag = CONFIGURATION_TAG_TMIN + 1;
  
  bool _exportViz;
  std::string _vizOutputDir = "/tmp";
  
  double _t0; // initial time for this slab
  double _t1; // final time for this slab
  
  // velocity-space vertices passed in at construction:
  std::vector<double> _velocitySpaceVertices_x;
  std::vector<double> _velocitySpaceVertices_y;
  std::vector<double> _velocitySpaceVertices_z;
  
  MeshTopologyPtr _spatialMeshTopology; // passed in at construction, potentially refined thereafter
  
  FunctionPtr _f0;       // initial condition set by setInflowConditions() or setInitialAndInflowConditions()
  FunctionPtr _f_inflow; // inflow condition set by setInitialAndInflowConditions()
public:
  VlasovSpaceTimeFormulation(double q, double m, bool useSerendipityBasis,
                             Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                             MeshTopologyPtr spatialMeshTopology,
                             std::vector<double> temporalVertices,
                             std::vector<double> velocitySpaceVertices_x,
                             std::vector<double> velocitySpaceVertices_y,
                             std::vector<double> velocitySpaceVertices_z,
                             const int H1Order, const int delta_k);

  //! take time step.
  void solve();
  
  //! clear previously computed solution
  void clearSolution();
  
  // field variables:
  VarPtr f();  // Vlasov
  
  // traces:
  VarPtr f_hat(); // Vlasov
  VarPtr t_n_hat(); // Vlasov
  
  // test variables:
  VarPtr w();   // Vlasov
  
  MeshPtr vlasovMesh();
  MeshPtr configurationSpaceMesh();
  
  Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation();
  
  Teuchos::RCP<SolutionMoments> solutionMoments();
      
  SolutionPtr vlasovSolution();
  SolutionPtr configurationSpaceSolution();
    
  bool hasL2Trace() const;
  
  //! Set the initial conditions without setting inflow conditions (relevant for the periodic case).
  void setInitialConditions(FunctionPtr f0);
  
  //! Set the initial, boundary conditions for Vlasov (relevant for the non-periodic case).
  void setInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_in);
  
  void setEnforceQuasiNeutrality(bool enforceQuasiNeutrality);
  bool enforceQuasiNeutrality() const;
  
  void setExportViz(bool exportViz, const std::string outputDir = "/tmp");
  
  /** \brief  Set the tolerance for the fixed point iteration.
      \param  fTol   [in]  - the target relative difference in f between one substep and the next, as measured in the L^2 norm.
  */
  void setFixedPointTolerance(const double fTol);
  double fixedPointTolerance() const;
  
  /** \brief  Set the minimum number of substeps to take for the fixed point iteration.
      \param  minSubsteps   [in]  - the minimum number of substeps to take for the fixed point solve.
  */
  void setFixedPointMinSubsteps(const int minSubsteps);
  int fixedPointMinSubsteps() const;
  
  /** \brief  Set the maximum number of substeps to take for the fixed point iteration.
      \param  maxSubsteps   [in]  - the maximum number of substeps to take for the fixed point solve.
  */
  void setFixedPointMaxSubsteps(const int maxSubsteps);
  int fixedPointMaxSubsteps() const;
  
  /** \brief  Set the tolerance for the iterative solve (relevant for geometric multigrid).
      \param  cgTols   [in]  - the tolerance for the conjugate gradient solve in each fixed-point step; used in the stopping criterion for the CG iteration
  */
  
  /** \brief  Set the level of console output.
      \param  azOutputLevel   [in]  - Corresponds to the Aztec output level (though Belos is doing the output); here, 0 means no output (the default); -1 means final residual + warnings; -2 means summary, final residual, and warnings; and -3 means only print warnings.  Positive number n means print out residual every n steps.
  */
  void setIterativeSolveOutputLevel(int azOutputLevel);
  int iterativeSolveOutputLevel() const;
  
  /** \brief  Set the tolerance for the iterative solve (relevant for geometric multigrid).
      \param  cgTol   [in]  - the tolerance for the conjugate gradient solve; used in the stopping criterion for the CG iteration
  */
  void setIterativeSolveTolerance(const double cgTol);
  double iterativeSolveTolerance() const;
  
  void setIterativeSolveTolerances(const std::vector<double> cgTols);
  std::vector<double> iterativeSolveTolerances() const;
  
  /** \brief  Set the maximum number of iterations for the iterative solve (relevant for geometric multigrid).
      \param  maxIters   [in]  - the maximum number of conjugate gradient iterations to take; will stop if either maxIters are reached, or the relative residual falls below the CG tolerance.
  */
  void setIterativeSolveMaxIterations(const int maxIters);
  int iterativeSolveMaxIterations() const;
  
  void setUseGeometricMultigrid(bool useGeometricMultigrid);
  bool useGeometricMultigrid() const;
  
  Teuchos::RCP<GMGSolver> constructGMGSolver();
  
  /** \brief  Create a tag for an indicated region of configuration space.
      \param  regionToMatch   [in]  - A SpatialFilterPtr that matches the region where BCs should be enforced.
      \return tag for the configuration-space mesh; can be used to set BCs.
   
   Note that the matching is performed immediately; this is not an appropriate method for setting a dynamic region of the mesh (as when the inflow region depends on the solution).  For dynamic mesh regions, SpatialFilter-based BCs are likely better suited than tags.
  */
  int createConfigurationMeshSideSetTag(SpatialFilterPtr regionToMatch);
  
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
  
  /** \brief  Returns a tag matching the initial time side set on the configuration-space mesh.
        The initial-time side set is created during construction.  See createConfigurationMeshSideSetTag() for specification of arbitrary regions of the configuration-space mesh boundary.
  */
  int tagConfigurationInitialTime() const;
  
  int tagVlasovInitialTime() const;
  int tagVlasovSpatialInflow() const;
  int tagVlasovMaxVelocity(int zeroBasedComp) const;
  int tagVlasovMinVelocity(int zeroBasedComp) const;
  
  //! Constructs and returns a VlasovSpaceTimeFormulation for the time slab following the present.  Should be called after all refinements and solves have been completed on the present time slab; modifications to the present time slab after the next slab is created may result in an invalid/incorrect application of boundary conditions in the next slab.
  VlasovSpaceTimeFormulation nextTimeSlab(double tWidth, int numTemporalElements);
  
  /** \brief  Returns a Function representing the specified component of configuration space. */
  FunctionPtr x_i(const int zeroBasedComp) const;
  
  /** \brief  Returns a Function representing the time coordinate. */
  FunctionPtr t() const;
  
  /** \brief  Returns a Function representing the specified velocity-space coordinate. */
  FunctionPtr v_i(const int zeroBasedComp) const;
};

}

#endif
