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
#include "PhasePhysicalMeshMap.h"
#include "PoissonFormulation.h"
#include "QuadraturePointFunction.h"
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
  
  SolutionPtr _vlasovSoln;
  
  double _poissonMeshMeasure;
  double _vlasovMeshMeasure;
    
  SolutionPtr _poissonSoln;
  
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
  
  Teuchos::RCP<SolutionMomentFunction> _f_velocityIntegral; // integral of f over all velocity space
  FunctionPtr _E_extruded;  // E solution extended to the Vlasov mesh
  
  Teuchos::RCP<PhasePhysicalMeshMap> _phaseSpacePhysicalMeshMap;
  
  int _numSubsteps = 3; // number of fixed-point iterations to take (alternating vlasov, poisson solves)
  
  // first
  int _aztecOutputLevel = 0; // AZ_none -- no output.
  Teuchos::RCP<GMGSolver> _gmgSolver;
  
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
  
  std::vector< std::vector<int> > _iterationCounts; // outer vector index: time step number; inner: substep number.
  
  void updateElectricField();
public:
  VlasovPoissonSpaceTimeFormulation(VlasovPoissonSpaceTimeFormulationChoice formulationChoice,
                                    double q, double eps0, double m,
                                    const double x0, const double xWidth, const int xElementsCoarse,
                                    const double t0, const double tFinal, const int tElementsCoarse,
                                    const double v0, const double vWidth, const int vElementsCoarse,
                                    const int H1Order, const int delta_k,
                                    const bool usePeriodicBCs, const bool enforceQuasiNeutrality);

  //! take time step.
  void solve();
  
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
  
  //! Returns a SolutionMomentFunction corresponding to the zero moment, that is, the integral of f over velocity space.  Call updateValues() to ensure that the integral has been performed after the last update to f.
  Teuchos::RCP<SolutionMomentFunction> zerothMoment();
  
  Teuchos::RCP<PhasePhysicalMeshMap> phaseSpacePhysicalMeshMap();
  
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
  
  /** \brief  Set the level of console output.
      \param  azOutputLevel   [in]  - Corresponds to the Aztec output level (though Belos is doing the output); here, 0 means no output (the default); -1 means final residual + warnings; -2 means summary, final residual, and warnings; and -3 means only print warnings.  Positive number n means print out residual every n steps.
  */
  void setIterativeSolveOutputLevel(int azOutputLevel);
  
  /** \brief  Sets thresholds for adaptivity
      \param  refinementThreshold   [in]  - on each refinement, refine any element with error greater than refinementThreshold times the maximum element error.  Values > 1.0 mean no refinement.
      \param  maxElements [in]  - maximum number of active elements allowed in the mesh (unrefinements will be performed to maintain this rule)
      \param  maxRefinementLevel [in]  - maximum levels of refinement
      \param  gateRefinements [in] - if true, only refine when max element error is greater than the least error that would have induced a refinement in the last round of refinements (otherwise, refine once per time step)
  */
  void setRefinementThresholds(double refinementThreshold, int maxElements, int maxRefinementLevel, ErrorIndicatorPtr unrefinementErrorIndicator = Teuchos::null,
                               bool gateRefinements = false);
  
  void setSubstepsForFixedPoint(int numSubsteps);
  
  int tagVlasovInitialTime() const;
  int tagVlasovLeftInflow() const;
  int tagVlasovRightInflow() const;
  int tagVlasovMaxVelocity() const;
  int tagVlasovMinVelocity() const;
};

}

#endif
