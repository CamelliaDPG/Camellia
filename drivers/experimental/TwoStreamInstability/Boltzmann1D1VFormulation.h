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
  
  SolutionPtr _boltzmannSoln;
  SolutionPtr _boltzmannSolnPrev;
  
  double _gaussMeshMeasure;
  double _boltzmannMeshMeasure;
  
  bool _haveTableau = false;
  
  SolutionPtr _gaussSoln;
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
  
  // set by user:
  ParameterFunctionPtr _dt;
  ParameterFunctionPtr _q;
  ParameterFunctionPtr _eps0;
  ParameterFunctionPtr _m;
  
  // set during solve process:
  ParameterFunctionPtr _n0; // integral of f over the whole 1D1V space
  
  Teuchos::RCP<SolutionMoments> _solutionMoments;
  FunctionPtr _f_velocityIntegral; // integral of f over all velocity space
  FunctionPtr _E_extruded;  // E solution extended to the Boltzmann mesh
  
  // TODO: consider whether these are still useful; PhasePhysicalMeshMap1D1V should provide everything we need (and if it doesn't, make it so)
  std::map<GlobalIndexType, std::vector<GlobalIndexType> > _spatialCellIDToPhaseSpaceCellIDs;
  std::map<GlobalIndexType, std::set<GlobalIndexType> > _phaseSpaceCellIDToSpatialCellID;
  
  Teuchos::RCP<PhasePhysicalMeshMap> _phaseSpacePhysicalMeshMap;
  
  int _numSubstepsForImplicit = 3; // usually the first solution is close enough to meet CG tolerance even after the electric field has been updated, so 3 is overkill
  
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
  
  std::vector< std::vector<int> > _iterationCounts; // outer vector index: time step number; inner: substep number.
  
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
  
  void setSubstepsForImplicit(int numSubsteps);
  
  //! notification that the boltzmann solve has set the RK stage:
  void setStage(const int stageNumber);
  
  //! map from gauss mesh cellIDs to corresponding cellIDs in boltzmann mesh
  const std::map<GlobalIndexType, std::vector<GlobalIndexType> > &spatialCellIDToPhaseSpaceCellIDsMap();
  
  //! map that is reverse of the above; there is exactly one spatial cell for each phase space cell
  const std::map<GlobalIndexType, std::set<GlobalIndexType> > &phaseSpaceCellIDToSpatialCellIDs();
};

}

#endif
