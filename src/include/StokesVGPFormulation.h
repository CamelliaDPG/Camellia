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

#include "BF.h"
#include "GMGSolver.h"
#include "ParameterFunction.h"
#include "PoissonFormulation.h"
#include "RefinementStrategy.h"
#include "Solution.h"
#include "SpatialFilter.h"
#include "TimeSteppingConstants.h"
#include "TypeDefs.h"
#include "VarFactory.h"

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
  bool _includeVelocityTracesInFluxTerm; // distinguishes between two space-time formulation options
  double _t0; // used in space-time

  bool _useStrongConservation; // experimental option to eliminate one scalar from the tensor velocity gradient using the mass conservation law

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

  std::map<int,int> _trialVariablePolyOrderAdjustments;

  VarFactoryPtr _vf;

  static const string S_U1, S_U2, S_U3;
  static const string S_P;
  static const string S_SIGMA11, S_SIGMA12, S_SIGMA13, S_SIGMA21, S_SIGMA22, S_SIGMA23, S_SIGMA31, S_SIGMA32, S_SIGMA33;

  static const string S_U1_HAT, S_U2_HAT, S_U3_HAT;
  static const string S_TN1_HAT, S_TN2_HAT, S_TN3_HAT;

  static const string S_V1, S_V2, S_V3;
  static const string S_Q;
  static const string S_TAU1, S_TAU2, S_TAU3;

  double computeMassFlux(bool takeAbsoluteValuesOnEachCell) const;

  void CHECK_VALID_COMPONENT(int i) const; // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)

  // ! initialize the Solution object(s) using the provided MeshTopology
  void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
                          TFunctionPtr<double> forcingFunction, std::string fileToLoadPrefix,
                          int temporalPolyOrder);

  void turnOffSuperLUDistOutput(Teuchos::RCP<GMGSolver> gmgSolver);
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

  // ! Sets an initial condition for space-time.  u0 should have a number of components equal to the spatial dimension.
  // ! If a null pressure is provided (the default), no initial condition will be imposed on the pressure.
  void addInitialCondition(double t0, vector<FunctionPtr> u0, FunctionPtr p0 = Teuchos::null);

  // ! sets a zero initial condition for space-time.
  void addZeroInitialCondition(double t0);

  // ! sets an outflow velocity boundary condition.  If usePhysicalTractions is true, imposes zero-traction outflow conditions using penalty constraints.  Otherwise, imposes zero values on the "traction" arising from integration by parts on the (pI - sigma) term.
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

  // ! Returns the specified component of the traction, expressed as a LinearTerm involving field variables.
  LinearTermPtr getTraction(int i);

  // ! Returns the solution (at current time)
  TSolutionPtr<double> solution() const;

  // ! Returns the solution (at previous time)
  TSolutionPtr<double> solutionPreviousTimeStep() const;

  // ! Solves
  void solve();

  // ! Solves iteratively
  void solveIteratively(int maxIters, double cgTol, int azOutputLevel = 0, bool suppressSuperLUOutput = true);

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
  VarPtr sigma(int i, int j) const; // sigma_ij is the Reynolds-weighted derivative of u_i in the j dimension
  VarPtr u(int i) const;
  VarPtr p() const;

  // traces:
  VarPtr tn_hat(int i) const;
  VarPtr u_hat(int i) const;

  // test variables:
  VarPtr tau(int i) const;
  VarPtr v(int i) const;
  VarPtr q() const;

  // ! returns the forcing function for this formulation if u and p are the exact solutions.
  TFunctionPtr<double> forcingFunction(TFunctionPtr<double> u, TFunctionPtr<double> p);

  // ! returns the pressure (which depends on the solution)
  TFunctionPtr<double> getPressureSolution();

  // ! returns a map indicating any trial variables that have adjusted polynomial orders relative to the standard poly order for the element.  Keys are variable IDs, values the difference between the indicated variable and the standard polynomial order.
  const std::map<int,int> &getTrialVariablePolyOrderAdjustments();

  // ! returns the velocity (which depends on the solution)
  TFunctionPtr<double> getVelocitySolution();

  // ! returns the vorticity (which depends on the solution)
  TFunctionPtr<double> getVorticity();

  // ! returns the sum of the absolute value of the mass flux through each element
  double absoluteMassFlux() const;

  // ! returns the sum of the signed mass flux through each element
  double netMassFlux() const;

  static StokesVGPFormulation steadyFormulation(int spaceDim, double mu, bool useConformingTraces);

  // ! Returns a steady StokesVGPFormulation that enforces conservation (in the sense of the diagonal of the gradient
  // ! sigma summing to zero) strongly.  Note that this is as of this writing (08-Nov-15) still experimental.
  static StokesVGPFormulation steadyFormulationStrongConservation(int spaceDim, double mu, bool useConformingTraces);

  // ! when includeVelocityTracesInFluxTerm is true, u1_hat, etc. only defined on spatial interfaces; the temporal velocities are the spatially-normal components of tn_hat.
  static StokesVGPFormulation spaceTimeFormulation(int spaceDim, double mu, bool useConformingTraces, bool includeVelocityTracesInFluxTerm = true);

  // ! Returns a steady StokesVGPFormulation that enforces conservation (in the sense of the diagonal of the gradient
  // ! sigma summing to zero) strongly.  Note that this is as of this writing (08-Nov-15) still experimental.
  // ! when includeVelocityTracesInFluxTerm is true, u1_hat, etc. only defined on spatial interfaces; the temporal velocities are the spatially-normal components of tn_hat.
  static StokesVGPFormulation spaceTimeFormulationStrongConservation(int spaceDim, double mu, bool useConformingTraces,
                                                                     bool includeVelocityTracesInFluxTerm = true);


  static StokesVGPFormulation timeSteppingFormulation(int spaceDim, double mu, double dt, bool useConformingTraces,
                                                      TimeStepType timeStepType = BACKWARD_EULER);

  // ! Returns a steady StokesVGPFormulation that enforces conservation (in the sense of the diagonal of the gradient
  // ! sigma summing to zero) strongly.  Note that this is as of this writing (08-Nov-15) still experimental.
  static StokesVGPFormulation timeSteppingFormulationStrongConservation(int spaceDim, double mu, double dt, bool useConformingTraces,
                                                                        TimeStepType timeStepType = BACKWARD_EULER);
};
}


#endif
