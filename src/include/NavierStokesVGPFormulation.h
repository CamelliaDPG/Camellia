//
//  NavierStokesVGPFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 2/5/15.
//
//

#ifndef Camellia_NavierStokesVGPFormulation_h
#define Camellia_NavierStokesVGPFormulation_h

#include "VarFactory.h"
#include "BF.h"

#include "MeshTopology.h"
#include "Solution.h"

#include "Solver.h"

#include "ParameterFunction.h"
#include "RefinementStrategy.h"

#include "ExactSolution.h"

class NavierStokesVGPFormulation {
  BFPtr _navierStokesBF, _stokesBF;
  bool _useConformingTraces;
  double _mu;
  Teuchos::RCP<ParameterFunction> _dt; // use a ParameterFunction so that we can set value later and references (in BF, e.g.) automatically pick this up

  LinearTermPtr _t1, _t2, _t3; // tractions
  
  SolverPtr _solver;
  
  FunctionPtr _L2IncrementFunction, _L2SolutionFunction;
  
  SolutionPtr _backgroundFlow, _solnIncrement;
  
  RefinementStrategyPtr _refinementStrategy;
  
  bool _neglectFluxesOnRHS;
  
  int _nonlinearIterationCount; // starts at 0, increases for each iterate
  
  static const string S_U1, S_U2, S_U3;
  static const string S_P;
  static const string S_SIGMA1, S_SIGMA2, S_SIGMA3;
  
  static const string S_U1_HAT, S_U2_HAT, S_U3_HAT;
  static const string S_TN1_HAT, S_TN2_HAT, S_TN3_HAT;
  
  static const string S_V1, S_V2, S_V3;
  static const string S_Q;
  static const string S_TAU1, S_TAU2, S_TAU3;
public:
  NavierStokesVGPFormulation(MeshTopologyPtr meshTopology, double Re,
                             int fieldPolyOrder,
                             int delta_k = 1,
                             FunctionPtr forcingFunction = Teuchos::null,
                             bool transientFormulation = false,
                             bool useConformingTraces = false);
  
  // ! sets a wall boundary condition
  void addWallCondition(SpatialFilterPtr wall);
  
  // ! sets an inflow velocity boundary condition; in 2D and 3D, u should be a vector-valued function.
  void addInflowCondition(SpatialFilterPtr inflowRegion, FunctionPtr u);
  
  // ! sets an outflow velocity boundary condition
  void addOutflowCondition(SpatialFilterPtr outflowRegion);
  
  // ! set a pressure condition at a point
  void addPointPressureCondition();
  
  // ! set a pressure condition at a point
  void addZeroMeanPressureCondition();
  
  // ! return an ExactSolutionPtr corresponding to specified velocity (a rank 1 Function) and pressure.
  Teuchos::RCP<ExactSolution> exactSolution(FunctionPtr u, FunctionPtr p);
  
  // ! returns the L^2 norm of the incremental solution
  double L2NormSolutionIncrement();
  
  // ! returns the L^2 norm of the background flow
  double L2NormSolution();
  
  // ! returns the nonlinear iteration count (since last refinement)
  int nonlinearIterationCount();
  
  // ! set the inner product to use during solve and during energy error determination
  void setIP( IPPtr ip );
  
  // ! refine according to energy error in the accumulated solution
  void refine();
  
  // ! returns the RefinementStrategy object being used to drive refinements
  RefinementStrategyPtr getRefinementStrategy();
  
  // ! returns an RHSPtr corresponding to the vector forcing function f and the accumulated solution
  RHSPtr rhs(FunctionPtr f, bool excludeFluxesAndTraces);
  
  // ! set the RefinementStrategy to use for driving refinements
  void setRefinementStrategy(RefinementStrategyPtr refStrategy);
  
  // ! the Stokes bilinear form
  BFPtr stokesBF();
  
  // ! set current time step used for transient solve
  void setTimeStep(double dt);
  
  // ! Returns the background flow, i.e. the accumulated solution thus far
  SolutionPtr solution();
  
  // ! Returns the latest solution increment
  SolutionPtr solutionIncrement();
  
  // ! The first time this is called, calls solution()->solve(), and the weight argument is ignored.  After the first call, solves for the next iterate, and adds to background flow with the specified weight.
  void solveAndAccumulate(double weight=1.0);
  
  BFPtr bf();
  
  // field variables:
  VarPtr sigma(int i);
  VarPtr u(int i);
  VarPtr p();
  
  // traces:
  VarPtr tn_hat(int i);
  VarPtr u_hat(int i);
  
  // test variables:
  VarPtr tau(int i);
  VarPtr v(int i);
  
  static FunctionPtr forcingFunction(int spaceDim, double Re, FunctionPtr u, FunctionPtr p);
};

#endif
