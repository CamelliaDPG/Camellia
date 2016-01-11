//
//  CompressibleNavierStokesFormulation.h
//  Camellia
//
//  Created by Truman Ellis on 12/4/15.
//
//

#ifndef Camellia_CompressibleNavierStokesFormulation_h
#define Camellia_CompressibleNavierStokesFormulation_h

#include <Teuchos_ParameterList.hpp>

#include "BF.h"
#include "ExactSolution.h"
#include "MeshTopology.h"
#include "GMGSolver.h"
#include "ParameterFunction.h"
#include "PoissonFormulation.h"
#include "RefinementStrategy.h"
#include "Solution.h"
#include "Solver.h"
#include "SpatialFilter.h"
#include "TimeSteppingConstants.h"
#include "TypeDefs.h"
#include "VarFactory.h"

namespace Camellia
{
class CompressibleNavierStokesFormulation
{
  BFPtr _bf;

  int _spaceDim;
  bool _useConformingTraces;
  double _mu;
  double _gamma;
  double _Pr;
  double _Cv;
  FunctionPtr _beta;
  int _spatialPolyOrder;
  int _temporalPolyOrder;
  int _delta_k;
  string _filePrefix;
  double _time;
  bool _timeStepping;
  bool _spaceTime;
  double _t0; // used in space-time
  bool _neglectFluxesOnRHS;

  int _nonlinearIterationCount; // starts at 0, increases for each iterate

  bool _haveOutflowConditionsImposed; // used to track whether we should impose point/zero mean conditions on pressure

  Teuchos::RCP<ParameterFunction> _dt; // use a ParameterFunction so that we can set value later and references (in BF, e.g.) automatically pick this up
  Teuchos::RCP<ParameterFunction> _t;  // use a ParameterFunction so that user can easily "ramp up" BCs in time...

  Teuchos::RCP<ParameterFunction> _theta; // selector for time step method; 0.5 is Crank-Nicolson

  LinearTermPtr _t1, _t2, _t3; // tractions

  BCPtr _bc, _zeroBC; // _zeroBC used when _neglectFluxesOnRHS = false, as it needs to be for GMG solves
  RHSPtr _rhsForSolve, _rhsForResidual;

  SolverPtr _solver;

  map<string, IPPtr> _ips;

  FunctionPtr _L2IncrementFunction, _L2SolutionFunction;

  SolutionPtr _backgroundFlow, _solnIncrement;

  SolutionPtr _solution, _previousSolution; // solution at current and previous time steps

  RefinementStrategyPtr _refinementStrategy, _hRefinementStrategy, _pRefinementStrategy;

  Teuchos::ParameterList _ctorParameters;

  std::map<int,int> _trialVariablePolyOrderAdjustments;

  VarFactoryPtr _vf;

  static const string S_rho;
  static const string S_u1, S_u2, S_u3;
  static const string S_T;
  static const string S_D11, S_D12, S_D13, S_D21, S_D22, S_D23, S_D31, S_D32, S_D33;
  static const string S_q1, S_q2, S_q3;

  static const string S_tc;
  static const string S_tm1, S_tm2, S_tm3;
  static const string S_te;
  static const string S_u1_hat, S_u2_hat, S_u3_hat;
  static const string S_T_hat;

  static const string S_vc;
  static const string S_vm1, S_vm2, S_vm3;
  static const string S_ve;
  static const string S_S1, S_S2, S_S3;
  static const string S_tau;

  void CHECK_VALID_COMPONENT(int i); // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)

  // // ! initialize the Solution object(s) using the provided MeshTopology
  // void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
  //                         FunctionPtr forcingFunction, std::string fileToLoadPrefix,
  //                         int temporalPolyOrder);

  void turnOffSuperLUDistOutput(Teuchos::RCP<GMGSolver> gmgSolver);
public:
  CompressibleNavierStokesFormulation(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters);

  // ! the Oldroyd-B VGP formulation bilinear form
  BFPtr bf();

  void addVelocityTraceCondition(SpatialFilterPtr region, FunctionPtr u_exact);

  void addTemperatureTraceCondition(SpatialFilterPtr region, FunctionPtr T_exact);

  void addMassFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact);

  void addMomentumFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact);

  void addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact);

  // ! sets a wall boundary condition
  // void addWallCondition(SpatialFilterPtr wall);

  // ! sets an inflow velocity boundary condition; u should be a vector-valued function.
  // void addInflowCondition(SpatialFilterPtr inflowRegion, FunctionPtr u);

  // ! Sets an initial condition for space-time.  u0 should have a number of components equal to the spatial dimension.
  // ! If a null pressure is provided (the default), no initial condition will be imposed on the pressure.
  // void addInitialCondition(double t0, vector<FunctionPtr> u0, FunctionPtr p0 = Teuchos::null);

  // ! sets a zero initial condition for space-time.
  // void addZeroInitialCondition(double t0);

  // ! sets an outflow velocity boundary condition.  If usePhysicalTractions is true, imposes zero-traction outflow conditions using penalty constraints.  Otherwise, imposes zero values on the "traction" arising from integration by parts on the (pI - L) term.
  // void addOutflowCondition(SpatialFilterPtr outflowRegion, bool usePhysicalTractions);

  // ! initialize the Solution object(s) using the provided MeshTopology
  // void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k = 1,
  //                         FunctionPtr forcingFunction = Teuchos::null,
  //                         int temporalPolyOrder = 1);

  // ! initialize the Solution object(s) using the provided MeshTopology
  // void initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k = 1,
  //                         FunctionPtr forcingFunction = Teuchos::null,
  //                         int temporalPolyOrder = 1);

  // ! returns true if this is a space-time formulation; false otherwise.
  bool isSpaceTime() const;

  // ! returns true if this is a steady formulation; false otherwise.
  bool isSteady() const;

  // ! returns true if this is a time-stepping formulation; false otherwise.
  bool isTimeStepping() const;

  // ! declare inner product
  void setIP(IPPtr ip);

  void setIP( string normName );

  // ! L^2 norm of the difference in u1, u2, and p from previous time step, normalized
  // double relativeL2NormOfTimeStep();

  // ! returns the L^2 norm of the incremental solution
  double L2NormSolutionIncrement();

  // ! returns the L^2 norm of the background flow
  double L2NormSolution();

  // ! returns the nonlinear iteration count (since last refinement)
  int nonlinearIterationCount();

  // ! Loads the mesh and solution from disk, if they were previously saved using save().  In the present
  // ! implementation, assumes that the constructor arguments provided to CompressibleNavierStokesFormulation were the same
  // ! on the CompressibleNavierStokesFormulation on which save() was invoked as they were for this CompressibleNavierStokesFormulation.
  void load(std::string prefixString);

  // ! Returns viscosity mu.
  double mu();

  // ! Returns gamma
  double gamma();

  // ! Returns Pr
  double Pr();

  // ! Returns Cv
  double Cv();

  // ! Returns Cp
  double Cp();

  // ! Returns R
  double R();

  // ! refine according to energy error in the solution
  void refine();

  // ! h-refine according to energy error in the solution
  void hRefine();

  // ! p-refine according to energy error in the solution
  void pRefine();

  // ! returns the RefinementStrategy object being used to drive refinements
  RefinementStrategyPtr getRefinementStrategy();

  // ! Returns an RHSPtr corresponding to the vector forcing function f and the formulation.
  RHSPtr rhs(FunctionPtr f, bool excludeFluxesAndTraces);

  // ! Saves the solution(s) and mesh to an HDF5 format.
  void save(std::string prefixString);

  // ! set the RefinementStrategy to use for driving refinements
  void setRefinementStrategy(RefinementStrategyPtr refStrategy);

  // ! get the Solver used for the linear updates
  SolverPtr getSolver();

  // ! set the Solver for the linear updates
  void setSolver(SolverPtr solver);

  // ! set current time step used for transient solve
  void setTimeStep(double dt);

  // ! Returns the specified component of the traction, expressed as a LinearTerm involving field variables.
  // LinearTermPtr getTraction(int i);

  // ! Returns the solution (at current time)
  SolutionPtr solution();

  // ! Returns the latest solution increment (at current time)
  SolutionPtr solutionIncrement();

  // ! The first time this is called, calls solution()->solve(), and the weight argument is ignored.  After the first call, solves for the next iterate, and adds to background flow with the specified weight.
  void solveAndAccumulate(double weight=1.0);

  // ! Returns the solution (at previous time)
  SolutionPtr solutionPreviousTimeStep();

  // ! Solves
  void solve();

  // ! Solves iteratively
  void solveIteratively(int maxIters, double cgTol, int azOutputLevel = 0, bool suppressSuperLUOutput = true);

  // ! Returns the spatial dimension.
  int spaceDim();

  // ! Takes a time step
  void takeTimeStep();

  // ! Returns the sum of the time steps taken thus far.
  double getTime();

  // ! Returns a FunctionPtr which gets updated with the current time.  Useful for setting BCs that vary in time.
  FunctionPtr getTimeFunction();

  // field variables:
  VarPtr rho();
  VarPtr u(int i);
  VarPtr T();
  VarPtr D(int i, int j); // D_ij is the Reynolds-weighted derivative of u_i in the j dimension
  VarPtr q(int i);

  // traces:
  VarPtr tc();
  VarPtr tm(int i);
  VarPtr te();
  VarPtr u_hat(int i);
  VarPtr T_hat();

  // test variables:
  VarPtr vc();
  VarPtr vm(int i);
  VarPtr ve();
  VarPtr S(int i);
  VarPtr tau();

  // ! returns a map indicating any trial variables that have adjusted polynomial orders relative to the standard poly order for the element.  Keys are variable IDs, values the difference between the indicated variable and the standard polynomial order.
  const std::map<int,int> &getTrialVariablePolyOrderAdjustments();

  // ! zeros out the solution increment
  void clearSolutionIncrement();

  Teuchos::ParameterList getConstructorParameters() const;

  // ! returns the forcing function for this formulation if u and p are the exact solutions.
  // FunctionPtr forcingFunction(FunctionPtr u, FunctionPtr p);

  // ! Set the forcing function for problem.  Should be a vector-valued function, with number of components equal to the spatial dimension.
  void setForcingFunction(FunctionPtr f);

  // static utility functions:
  static CompressibleNavierStokesFormulation steadyFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                      MeshTopologyPtr meshTopo, int polyOrder, int delta_k);
  static CompressibleNavierStokesFormulation spaceTimeFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                         MeshTopologyPtr meshTopo, int spatialPolyOrder, int temporalPolyOrder, int delta_k);
};
}


#endif
