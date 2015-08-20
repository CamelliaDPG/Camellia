//
//  StokesVGPFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#ifndef Camellia_StokesVGPFormulation_h
#define Camellia_StokesVGPFormulation_h

#include <Teuchos_ParameterList.hpp>

#include "TypeDefs.h"

#include "VarFactory.h"
#include "BF.h"
#include "SpatialFilter.h"
#include "Solution.h"
#include "RefinementStrategy.h"
#include "ParameterFunction.h"
#include "PoissonFormulation.h"
#include "TimeSteppingConstants.h"

namespace Camellia
{
class StokesVGPFormulation
{
  BFPtr _stokesBF;
  BFPtr _steadyStokesBF;

  int _spaceDim;
  bool _useConformingTraces;
  double _mu;
  double _time;
  bool _timeStepping;
  bool _spaceTime;

  bool _haveOutflowConditionsImposed; // used to track whether we should impose point/zero mean conditions on pressure

  Teuchos::RCP<ParameterFunction> _dt; // use a ParameterFunction so that we can set value later and references (in BF, e.g.) automatically pick this up
  Teuchos::RCP<ParameterFunction> _t;  // use a ParameterFunction so that user can easily "ramp up" BCs in time...

  Teuchos::RCP<ParameterFunction> _theta; // selector for time step method; 0.5 is Crank-Nicolson

  LinearTermPtr _t1, _t2, _t3; // tractions

  SolverPtr _solver;

  TSolutionPtr<double> _solution, _previousSolution; // solution at current and previous time steps
  TSolutionPtr<double> _streamSolution;

  Teuchos::RCP<PoissonFormulation> _streamFormulation;

  RefinementStrategyPtr _refinementStrategy, _hRefinementStrategy, _pRefinementStrategy;

  VarFactoryPtr _vf;

  static const string S_U1, S_U2, S_U3;
  static const string S_P;
  static const string S_SIGMA1, S_SIGMA2, S_SIGMA3;

  static const string S_U1_HAT, S_U2_HAT, S_U3_HAT;
  static const string S_TN1_HAT, S_TN2_HAT, S_TN3_HAT;

  static const string S_V1, S_V2, S_V3;
  static const string S_Q;
  static const string S_TAU1, S_TAU2, S_TAU3;

  // ! initialize the Solution object(s) using the provided MeshTopology
  void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
                          TFunctionPtr<double> forcingFunction, std::string fileToLoadPrefix);

public:
  StokesVGPFormulation(Teuchos::ParameterList &parameters);
//  StokesVGPFormulation(int spaceDim, bool useConformingTraces, double mu = 1.0,
//                       bool transient = false, double dt = 1.0);

  // ! the Stokes VGP formulation bilinear form
  BFPtr bf();

  // ! sets a wall boundary condition
  void addWallCondition(SpatialFilterPtr wall);

  // ! sets an inflow velocity boundary condition; u should be a vector-valued function.
  void addInflowCondition(SpatialFilterPtr inflowRegion, TFunctionPtr<double> u);
  
  // ! sets a zero initial condition for space-time.
  void addZeroInitialCondition(double t0);

  // ! sets an outflow velocity boundary condition
  void addOutflowCondition(SpatialFilterPtr outflowRegion);

  // ! set a pressure condition at a point
  void addPointPressureCondition(vector<double> point = vector<double>());

  // ! set a pressure condition at a point
  void addZeroMeanPressureCondition();

  // ! initialize the Solution object(s) using the provided MeshTopology
  void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k = 1,
                          TFunctionPtr<double> forcingFunction = Teuchos::null);

  // ! initialize the Solution object(s) using the provided MeshTopology
  void initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k = 1,
                          TFunctionPtr<double> forcingFunction = Teuchos::null);

  // ! returns true if this is a space-time formulation; false otherwise.
  bool isSpaceTime() const;
  
  // ! returns true if this is a steady formulation; false otherwise.
  bool isSteady() const;
  
  // ! returns true if this is a time-stepping formulation; false otherwise.
  bool isTimeStepping() const;
  
  // ! L^2 norm of the difference in u1, u2, and p from previous time step, normalized
  double relativeL2NormOfTimeStep();

  // ! Loads the mesh and solution from disk, if they were previously saved using save().  In the present
  // ! implementation, assumes that the constructor arguments provided to StokesVGPFormulation were the same
  // ! on the StokesVGPFormulation on which save() was invoked as they were for this StokesVGPFormulation.
  void load(std::string prefixString);

  // ! Returns viscosity mu.
  double mu();

  // ! refine according to energy error in the solution
  void refine();

  // ! h-refine according to energy error in the solution
  void hRefine();

  // ! p-refine according to energy error in the solution
  void pRefine();

  // ! returns the RefinementStrategy object being used to drive refinements
  RefinementStrategyPtr getRefinementStrategy();

  // ! Returns an RHSPtr corresponding to the vector forcing function f and the formulation.
  RHSPtr rhs(TFunctionPtr<double> f);

  // ! Saves the solution(s) and mesh to an HDF5 format.
  void save(std::string prefixString);

  // ! set the RefinementStrategy to use for driving refinements
  void setRefinementStrategy(RefinementStrategyPtr refStrategy);

  // ! set current time step used for transient solve
  void setTimeStep(double dt);

  // ! Returns the solution (at current time)
  TSolutionPtr<double> solution();

  // ! Returns the solution (at previous time)
  TSolutionPtr<double> solutionPreviousTimeStep();

  // ! Solves
  void solve();

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
  VarPtr sigma(int i);
  VarPtr u(int i);
  VarPtr p();

  // traces:
  VarPtr tn_hat(int i);
  VarPtr u_hat(int i);

  // test variables:
  VarPtr tau(int i);
  VarPtr v(int i);

  // ! returns the forcing function for this formulation if u and p are the exact solutions.
  TFunctionPtr<double> forcingFunction(TFunctionPtr<double> u, TFunctionPtr<double> p);
  
  static StokesVGPFormulation steadyFormulation(int spaceDim, double mu, bool useConformingTraces);
  static StokesVGPFormulation spaceTimeFormulation(int spaceDim, double mu, bool useConformingTraces);
  
  static StokesVGPFormulation timeSteppingFormulation(int spaceDim, double mu, double dt, bool useConformingTraces, TimeStepType timeStepType);
};
}


#endif
