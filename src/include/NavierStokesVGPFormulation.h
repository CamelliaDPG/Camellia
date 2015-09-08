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
#include "StokesVGPFormulation.h"
#include "VarFactory.h"

namespace Camellia
{
class NavierStokesVGPFormulation
{
  Teuchos::RCP<StokesVGPFormulation> _stokesForm;
  BFPtr _navierStokesBF;
  bool _useConformingTraces;

  int _spaceDim;

  BCPtr _bc, _zeroBC; // _zeroBC used when _neglectFluxesOnRHS = false, as it needs to be for GMG solves
  RHSPtr _rhsForSolve, _rhsForResidual;

  SolverPtr _solver;

  TFunctionPtr<double> _L2IncrementFunction, _L2SolutionFunction;

  TSolutionPtr<double> _backgroundFlow, _solnIncrement;

  TSolutionPtr<double> _streamSolution;
  Teuchos::RCP<PoissonFormulation> _streamFormulation;

  RefinementStrategyPtr _refinementStrategy, _hRefinementStrategy, _pRefinementStrategy;

  bool _neglectFluxesOnRHS;

  int _nonlinearIterationCount; // starts at 0, increases for each iterate

//  void initialize(MeshTopologyPtr meshTopology, std::string filePrefix,
//                  int spaceDim, double Re, int fieldPolyOrder, int delta_k,
//                  TFunctionPtr<double> forcingFunction, bool transientFormulation, bool useConformingTraces);

  void refine(RefinementStrategyPtr refStrategy);
public:
//  NavierStokesVGPFormulation(MeshTopologyPtr meshTopology, double Re,
//                             int fieldPolyOrder,
//                             int delta_k = 1,
//                             TFunctionPtr<double> forcingFunction = Teuchos::null,
//                             bool transientFormulation = false,
//                             bool useConformingTraces = false);
//
//  NavierStokesVGPFormulation(std::string filePrefix, int spaceDim, double Re,
//                             int fieldPolyOrder, int delta_k = 1,
//                             TFunctionPtr<double> forcingFunction = Teuchos::null,
//                             bool transientFormulation = false,
//                             bool useConformingTraces = false);

  NavierStokesVGPFormulation(MeshTopologyPtr meshTopology, Teuchos::ParameterList &parameters);
  
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

  // ! returns the forcing function corresponding to the indicated exact solution
  TFunctionPtr<double> forcingFunction(int spaceDim, double Re, TFunctionPtr<double> u, TFunctionPtr<double> p);
  
  // ! Set the forcing function for Navier-Stokes.  Should be a vector-valued function, with number of components equal to the spatial dimension.
  void setForcingFunction(TFunctionPtr<double> f);
  
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

  // ! set the Solver for the linear updates
  void setSolver(SolverPtr solver);
  
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
  
  // static utility functions:
  
  static NavierStokesVGPFormulation steadyFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                      MeshTopologyPtr meshTopo, int polyOrder, int delta_k);
  static NavierStokesVGPFormulation spaceTimeFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                         MeshTopologyPtr meshTopo, int spatialPolyOrder, int temporalPolyOrder, int delta_k);
  static NavierStokesVGPFormulation timeSteppingFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                            MeshTopologyPtr meshTopo, int polyOrder, int delta_k,
                                                            double dt, TimeStepType timeStepType = BACKWARD_EULER);

  // ! returns the convective term (u dot grad u) corresponding to the provided velocity function
  static FunctionPtr convectiveTerm(int spcaeDim, FunctionPtr u_exact);
  
  // ! returns the forcing function for steady-state Navier-Stokes corresponding to the indicated exact solution
  static TFunctionPtr<double> forcingFunctionSteady(int spaceDim, double Re, TFunctionPtr<double> u, TFunctionPtr<double> p);
  
  // ! returns the forcing function for space-time Navier-Stokes corresponding to the indicated exact solution
  static TFunctionPtr<double> forcingFunctionSpaceTime(int spaceDim, double Re, TFunctionPtr<double> u, TFunctionPtr<double> p);
};
}


#endif
