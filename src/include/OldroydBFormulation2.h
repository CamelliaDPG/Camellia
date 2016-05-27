//
//  OldroydBFormulation2.h
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#ifndef Camellia_OldroydBFormulation2_h
#define Camellia_OldroydBFormulation2_h

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
class OldroydBFormulation2
{
  BFPtr _oldroydBBF;
  BFPtr _steadyStokesBF;

  int _spaceDim;
  bool _useConformingTraces;
  double _muS;
  double _muP;
  double _alpha;
  int _spatialPolyOrder;
  int _temporalPolyOrder;
  int _delta_k;
  string _filePrefix;
  double _time;
  bool _timeStepping;
  bool _spaceTime;
  bool _includeVelocityTracesInFluxTerm; // distinguishes between two space-time formulation options
  double _t0; // used in space-time
  bool _conservationFormulation;
  bool _neglectFluxesOnRHS;

  int _nonlinearIterationCount; // starts at 0, increases for each iterate

  bool _haveOutflowConditionsImposed; // used to track whether we should impose point/zero mean conditions on pressure

  Teuchos::RCP<ParameterFunction> _lambda; // use a ParameterFunction so that we can set value later and references (in BF, e.g.) automatically pick this up

  Teuchos::RCP<ParameterFunction> _dt; // use a ParameterFunction so that we can set value later and references (in BF, e.g.) automatically pick this up
  Teuchos::RCP<ParameterFunction> _t;  // use a ParameterFunction so that user can easily "ramp up" BCs in time...

  Teuchos::RCP<ParameterFunction> _theta; // selector for time step method; 0.5 is Crank-Nicolson

  LinearTermPtr _t1, _t2, _t3; // tractions

  BCPtr _bc, _zeroBC; // _zeroBC used when _neglectFluxesOnRHS = false, as it needs to be for GMG solves
  RHSPtr _rhsForSolve, _rhsForResidual;

  SolverPtr _solver;

  TFunctionPtr<double> _L2IncrementFunction, _L2SolutionFunction;

  TSolutionPtr<double> _backgroundFlow, _solnIncrement;

  TSolutionPtr<double> _solution, _previousSolution; // solution at current and previous time steps
  TSolutionPtr<double> _streamSolution;

  Teuchos::RCP<PoissonFormulation> _streamFormulation;

  RefinementStrategyPtr _refinementStrategy, _hRefinementStrategy, _pRefinementStrategy;

  std::map<int,int> _trialVariablePolyOrderAdjustments;

  VarFactoryPtr _vf;

  static const string S_U1, S_U2, S_U3;
  static const string S_L11, S_L12, S_L13, S_L21, S_L22, S_L23, S_L31, S_L32, S_L33;
  static const string S_T11, S_T12, S_T13, S_T22, S_T23, S_T33;
  static const string S_P;

  static const string S_U1_HAT, S_U2_HAT, S_U3_HAT;
  static const string S_SIGMAN1_HAT, S_SIGMAN2_HAT, S_SIGMAN3_HAT;
  static const string S_TUN11_HAT, S_TUN12_HAT, S_TUN13_HAT, S_TUN22_HAT, S_TUN23_HAT, S_TUN33_HAT;

  static const string S_V1, S_V2, S_V3;
  static const string S_M1, S_M2, S_M3;
  static const string S_S11, S_S12, S_S13, S_S22, S_S23, S_S33;
  static const string S_Q;

  void CHECK_VALID_COMPONENT(int i); // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)

  // ! initialize the Solution object(s) using the provided MeshTopology
  void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
                          TFunctionPtr<double> forcingFunction, std::string fileToLoadPrefix,
                          int temporalPolyOrder);

  void turnOffSuperLUDistOutput(Teuchos::RCP<GMGSolver> gmgSolver);
public:
  OldroydBFormulation2(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters);
//  OldroydBFormulation2(int spaceDim, bool useConformingTraces, double mu = 1.0,
//                       bool transient = false, double dt = 1.0);

  // ! the Oldroyd-B VGP formulation bilinear form
  BFPtr bf();

  // ! sets a wall boundary condition
  void addWallCondition(SpatialFilterPtr wall);

  // ! sets an inflow velocity boundary condition; u should be a vector-valued function.
  void addInflowCondition(SpatialFilterPtr inflowRegion, TFunctionPtr<double> u);

  // ! sets an inflow values of the viscoelastic stress Tun should be a (symmetric) 2-tensor.
  void addInflowViscoelasticStress(SpatialFilterPtr inflowRegion, TFunctionPtr<double> T11un, TFunctionPtr<double> T12un, TFunctionPtr<double> T22un);  

  // ! Sets an initial condition for space-time.  u0 should have a number of components equal to the spatial dimension.
  // ! If a null pressure is provided (the default), no initial condition will be imposed on the pressure.
  void addInitialCondition(double t0, vector<FunctionPtr> u0, FunctionPtr p0 = Teuchos::null);

  // ! sets a zero initial condition for space-time.
  void addZeroInitialCondition(double t0);

  // ! sets an outflow velocity boundary condition.  If usePhysicalTractions is true, imposes zero-traction outflow conditions using penalty constraints.  Otherwise, imposes zero values on the "traction" arising from integration by parts on the (pI - L) term.
  void addOutflowCondition(SpatialFilterPtr outflowRegion, bool usePhysicalTractions);

  // ! set a pressure condition at a point
  void addPointPressureCondition(vector<double> point = vector<double>());

  // ! set a pressure condition at a point
  void addZeroMeanPressureCondition();

  // ! initialize the Solution object(s) using the provided MeshTopology
  void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k = 1,
                          TFunctionPtr<double> forcingFunction = Teuchos::null,
                          int temporalPolyOrder = 1);

  // ! initialize the Solution object(s) using the provided MeshTopology
  void initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k = 1,
                          TFunctionPtr<double> forcingFunction = Teuchos::null,
                          int temporalPolyOrder = 1);

  // ! returns true if this is a space-time formulation; false otherwise.
  bool isSpaceTime() const;

  // ! returns true if this is a steady formulation; false otherwise.
  bool isSteady() const;

  // ! returns true if this is a time-stepping formulation; false otherwise.
  bool isTimeStepping() const;

  // ! declare inner product
  void setIP(IPPtr ip);

  // ! L^2 norm of the difference in u1, u2, and p from previous time step, normalized
  double relativeL2NormOfTimeStep();

  // ! returns the L^2 norm of the incremental solution
  double L2NormSolutionIncrement();

  // ! returns the L^2 norm of the background flow
  double L2NormSolution();

  // ! returns the nonlinear iteration count (since last refinement)
  int nonlinearIterationCount();

  // ! Loads the mesh and solution from disk, if they were previously saved using save().  In the present
  // ! implementation, assumes that the constructor arguments provided to OldroydBFormulation2 were the same
  // ! on the OldroydBFormulation2 on which save() was invoked as they were for this OldroydBFormulation2.
  void load(std::string prefixString);

  // ! Returns solvent viscosity, muS.
  double muS();

  // ! Returns polymeric viscosity, muP.
  double muP();

  // ! Returns alpha.
  double alpha();

  // ! Returns lambda.
  Teuchos::RCP<ParameterFunction> lambda();

  // ! Sets lambda.
  void setLambda(double lambda);

  // ! refine according to energy error in the solution
  void refine();

  // ! h-refine according to energy error in the solution
  void hRefine();

  // ! p-refine according to energy error in the solution
  void pRefine();

  // ! returns the RefinementStrategy object being used to drive refinements
  RefinementStrategyPtr getRefinementStrategy();

  // ! Returns an RHSPtr corresponding to the vector forcing function f and the formulation.
  RHSPtr rhs(TFunctionPtr<double> f, bool excludeFluxesAndTraces);

  // ! Saves the solution(s) and mesh to an HDF5 format.
  void save(std::string prefixString);

  // ! set the RefinementStrategy to use for driving refinements
  void setRefinementStrategy(RefinementStrategyPtr refStrategy);

  // ! set current time step used for transient solve
  void setTimeStep(double dt);

  // ! Returns the specified component of the traction, expressed as a LinearTerm involving field variables.
  LinearTermPtr getTraction(int i);

  // ! Returns the solution (at current time)
  TSolutionPtr<double> solution();

  // ! Returns the latest solution increment (at current time)
  TSolutionPtr<double> solutionIncrement();

  // ! Solve without accumulating solution
  void solveForIncrement();

  // ! Accumalate background flow with present value of solution increment
  void accumulate(double weight=1.0);

  // ! The first time this is called, calls solution()->solve(), and the weight argument is ignored.  After the first call, solves for the next iterate, and adds to background flow with the specified weight.
  void solveAndAccumulate(double weight=1.0);

  // ! Returns the solution (at previous time)
  TSolutionPtr<double> solutionPreviousTimeStep();

  // ! Solves
  void solve();

  // ! Solves iteratively
  void solveIteratively(int maxIters, double cgTol, int azOutputLevel = 0, bool suppressSuperLUOutput = true);
  
  // computes Res(u + s*delta_u) . delta_u
  double computeG(double s);

  // ! Returns the spatial dimension.
  int spaceDim();

  // ! Returns a reference to the Poisson formulation used for the stream solution.
  PoissonFormulation &streamFormulation();

  // ! Returns the variable in the stream solution that represents the stream function.
  VarPtr streamPhi();

  // ! Returns the stream solution (at current time).  (Stream solution is created during initializeSolution, but
  // ! streamSolution->solve() must be called manually.)  Use streamPhi() to get a VarPtr for the streamfunction.
  TSolutionPtr<double> streamSolution();

  // ! Takes a time step
  void takeTimeStep();

  // ! Returns the sum of the time steps taken thus far.
  double getTime();

  // ! Returns a TFunctionPtr<double> which gets updated with the current time.  Useful for setting BCs that vary in time.
  TFunctionPtr<double> getTimeFunction();

  // field variables:
  VarPtr T(int i, int j);
  VarPtr L(int i, int j); // L_ij is the Reynolds-weighted derivative of u_i in the j dimension
  VarPtr u(int i);
  VarPtr p();

  // traces:
  VarPtr Tun_hat(int i, int j);
  VarPtr sigman_hat(int i);
  VarPtr u_hat(int i);

  // test variables:
  VarPtr S(int i, int j);
  VarPtr M(int i);
  VarPtr v(int i);
  VarPtr q();

  // error representation function
  TRieszRepPtr<double> rieszResidual(FunctionPtr forcingFunction);


  // ! returns the pressure (which depends on the solution)
  TFunctionPtr<double> getPressureSolution();

  // ! returns a map indicating any trial variables that have adjusted polynomial orders relative to the standard poly order for the element.  Keys are variable IDs, values the difference between the indicated variable and the standard polynomial order.
  const std::map<int,int> &getTrialVariablePolyOrderAdjustments();

  // ! returns the velocity (which depends on the solution)
  TFunctionPtr<double> getVelocitySolution();

  // ! returns the vorticity (which depends on the solution)
  TFunctionPtr<double> getVorticity();

  // ! returns the forcing function for this formulation if u and p are the exact solutions.
  TFunctionPtr<double> forcingFunction(TFunctionPtr<double> u, TFunctionPtr<double> p);

  // // ! returns the friction on the mesh skeleton (sigma_n) x n
  // TFunctionPtr<double> friction(SolutionPtr soln);

  // ! Set the forcing function for problem.  Should be a vector-valued function, with number of components equal to the spatial dimension.
  void setForcingFunction(TFunctionPtr<double> f);

  // ! returns the convective term (u dot grad u) corresponding to the provided velocity function
  static FunctionPtr convectiveTerm(int spaceDim, FunctionPtr u_exact);

  // static OldroydBFormulation2 steadyFormulation(int spaceDim, double mu, bool useConformingTraces);

  // // ! when includeVelocityTracesInFluxTerm is true, u1_hat, etc. only defined on spatial interfaces; the temporal velocities are the spatially-normal components of tn_hat.
  // static OldroydBFormulation2 spaceTimeFormulation(int spaceDim, double mu, bool useConformingTraces, bool includeVelocityTracesInFluxTerm = true);
  // static OldroydBFormulation2 timeSteppingFormulation(int spaceDim, double mu, double dt, bool useConformingTraces, TimeStepType timeStepType = BACKWARD_EULER);
};
}


#endif
