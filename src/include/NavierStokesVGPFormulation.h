//
//  NavierStokesVGPFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 2/5/15.
//
//

#ifndef Camellia_NavierStokesVGPFormulation_h
#define Camellia_NavierStokesVGPFormulation_h

#include "BF.h"
#include "ExactSolution.h"
#include "MeshTopology.h"
#include "ParameterFunction.h"
#include "PoissonFormulation.h"
#include "RefinementStrategy.h"
#include "Solution.h"
#include "Solver.h"
#include "VarFactory.h"

namespace Camellia
{
class NavierStokesVGPFormulation
{
  BFPtr _navierStokesBF, _stokesBF;
  bool _useConformingTraces;

  int _spaceDim;

  double _mu;
  Teuchos::RCP<ParameterFunction> _dt; // use a ParameterFunction so that we can set value later and references (in BF, e.g.) automatically pick this up
  Teuchos::RCP<ParameterFunction> _theta; // selector for time step method; 0.5 is Crank-Nicolson

  RHSPtr _rhsForSolve, _rhsForResidual;

  LinearTermPtr _t1, _t2, _t3; // tractions

  SolverPtr _solver;

  TFunctionPtr<double> _L2IncrementFunction, _L2SolutionFunction;

  TSolutionPtr<double> _backgroundFlow, _solnIncrement;

  TSolutionPtr<double> _streamSolution;
  Teuchos::RCP<PoissonFormulation> _streamFormulation;

  RefinementStrategyPtr _refinementStrategy, _hRefinementStrategy, _pRefinementStrategy;;

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

  void initialize(MeshTopologyPtr meshTopology, std::string filePrefix,
                  int spaceDim, double Re, int fieldPolyOrder, int delta_k,
                  TFunctionPtr<double> forcingFunction, bool transientFormulation, bool useConformingTraces);

  void refine(RefinementStrategyPtr refStrategy);
public:
  NavierStokesVGPFormulation(MeshTopologyPtr meshTopology, double Re,
                             int fieldPolyOrder,
                             int delta_k = 1,
                             TFunctionPtr<double> forcingFunction = Teuchos::null,
                             bool transientFormulation = false,
                             bool useConformingTraces = false);

  NavierStokesVGPFormulation(std::string filePrefix, int spaceDim, double Re,
                             int fieldPolyOrder, int delta_k = 1,
                             TFunctionPtr<double> forcingFunction = Teuchos::null,
                             bool transientFormulation = false,
                             bool useConformingTraces = false);

  // ! sets a wall boundary condition
  void addWallCondition(SpatialFilterPtr wall);

  // ! sets an inflow velocity boundary condition; in 2D and 3D, u should be a vector-valued function.
  void addInflowCondition(SpatialFilterPtr inflowRegion, TFunctionPtr<double> u);

  // ! sets an outflow velocity boundary condition
  void addOutflowCondition(SpatialFilterPtr outflowRegion);

  // ! set a pressure condition at a point
  void addPointPressureCondition();

  // ! set a pressure condition at a point
  void addZeroMeanPressureCondition();

  // ! return an ExactTSolutionPtr<double> corresponding to specified velocity (a rank 1 Function) and pressure.
  Teuchos::RCP<ExactSolution<double>> exactSolution(TFunctionPtr<double> u, TFunctionPtr<double> p);

  // ! returns the L^2 norm of the incremental solution
  double L2NormSolutionIncrement();

  // ! returns the L^2 norm of the background flow
  double L2NormSolution();

  // ! returns the nonlinear iteration count (since last refinement)
  int nonlinearIterationCount();

  // ! Saves the solution(s) and mesh to an HDF5 format.
  void save(std::string prefixString);

  // ! set the inner product to use during solve and during energy error determination
  void setIP( IPPtr ip );

  // ! refine according to energy error in the accumulated solution
  void refine();

  // ! h-refine according to energy error in the solution
  void hRefine();

  // ! p-refine according to energy error in the solution
  void pRefine();

  // ! returns the RefinementStrategy object being used to drive refinements
  RefinementStrategyPtr getRefinementStrategy();

  // ! returns an RHSPtr corresponding to the vector forcing function f and the accumulated solution
  RHSPtr rhs(TFunctionPtr<double> f, bool excludeFluxesAndTraces);

  // ! set the RefinementStrategy to use for driving refinements
  void setRefinementStrategy(RefinementStrategyPtr refStrategy);

  // ! the Stokes bilinear form
  BFPtr stokesBF();

  // ! set current time step used for transient solve
  void setTimeStep(double dt);

  // ! Returns the background flow, i.e. the accumulated solution thus far
  TSolutionPtr<double> solution();

  // ! Returns the latest solution increment
  TSolutionPtr<double> solutionIncrement();

  // ! The first time this is called, calls solution()->solve(), and the weight argument is ignored.  After the first call, solves for the next iterate, and adds to background flow with the specified weight.
  void solveAndAccumulate(double weight=1.0);

  // ! Returns the variable in the stream solution that represents the stream function.
  VarPtr streamPhi();

  // ! Returns the stream solution (at current time).  (Stream solution is created during initializeSolution, but
  // ! streamSolution->solve() must be called manually.)  Use streamPhi() to get a VarPtr for the streamfunction.
  TSolutionPtr<double> streamSolution();

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

  static TFunctionPtr<double> forcingFunction(int spaceDim, double Re, TFunctionPtr<double> u, TFunctionPtr<double> p);
};
}


#endif
