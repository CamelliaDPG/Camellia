//
//  CompressibleNavierStokesFormulation.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 3/15/18.
//

#ifndef CompressibleNavierStokesFormulation_hpp
#define CompressibleNavierStokesFormulation_hpp

// refactoring/reimplementing what has been in CompressibleNavierStokesFormulation, but does not have
// tests against it, and in its present form takes an impressively long time to compile.

#include "TypeDefs.h"

namespace Camellia
{
  class CompressibleNavierStokesFormulation
  {
    BFPtr _bf;
    RHSPtr _rhs;
    BFPtr _steadyBF;
    
    int _spaceDim;
    bool _useConformingTraces;
    bool _pureEulerMode; // turns off viscous terms, including tau and S equations (heat flux, velocity gradients)
    
    Teuchos::RCP<ParameterFunction> _fc, _fe; // forcing functions for continuity, energy equations
    std::vector<Teuchos::RCP<ParameterFunction> > _fm; // forcing for momentum equation(s)
    double _mu;
    FunctionPtr _muFunc;
    FunctionPtr _muSqrtFunc;
    Teuchos::RCP<ParameterFunction> _muParamFunc;
    Teuchos::RCP<ParameterFunction> _muSqrtParamFunc;
    double _gamma;
    double _Pr;
    double _Cv;
    FunctionPtr _beta;
    int _spatialPolyOrder;
    int _temporalPolyOrder;
    int _delta_k;
    std::string _filePrefix;
    double _time;
    bool _timeStepping;
    bool _spaceTime;
    double _t0; // used in space-time
    bool _neglectFluxesOnRHS;
    
    int _nonlinearIterationCount; // starts at 0, increases for each iterate
    
    Teuchos::RCP<ParameterFunction> _dt;
    Teuchos::RCP<ParameterFunction> _t;  // use a ParameterFunction so that user can easily "ramp up" BCs in time...
    
    Teuchos::RCP<ParameterFunction> _theta; // selector for time step method; 0.5 is Crank-Nicolson
    
    SolverPtr _solver;
    
    int _solveCode;
    
    std::map<std::string, IPPtr> _ips;
    
    FunctionPtr _L2IncrementFunction, _L2SolutionFunction, _L2ConservationIncrement;
    
    SolutionPtr _backgroundFlow, _solnIncrement, _solnPrevTime;
    
    // SolutionPtr _solution, _previousSolution; // solution at current and previous time steps
    
    RefinementStrategyPtr _refinementStrategy, _hRefinementStrategy, _pRefinementStrategy;
    
    Teuchos::ParameterList _ctorParameters;
    
    std::map<int,int> _trialVariablePolyOrderAdjustments;
    
    VarFactoryPtr _vf;
    
    static const std::string S_rho;
    static const std::string S_u1, S_u2, S_u3;
    static const std::string S_T;
    static const std::string S_D11, S_D12, S_D13, S_D21, S_D22, S_D23, S_D31, S_D32, S_D33;
    static const std::string S_q1, S_q2, S_q3;
    
    static const std::string S_tc;
    static const std::string S_tm1, S_tm2, S_tm3;
    static const std::string S_te;
    static const std::string S_u1_hat, S_u2_hat, S_u3_hat;
    static const std::string S_T_hat;
    
    static const std::string S_vc;
    static const std::string S_vm1, S_vm2, S_vm3;
    static const std::string S_ve;
    static const std::string S_S1, S_S2, S_S3;
    static const std::string S_tau;
    
    static const std::string S_u[3];
    static const std::string S_q[3];
    static const std::string S_D[3][3];
    
    static const std::string S_tm[3];
    static const std::string S_u_hat[3];
    
    static const std::string S_vm[3];
    static const std::string S_S[3];
    
    void CHECK_VALID_COMPONENT(int i); // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)
  public:
    CompressibleNavierStokesFormulation(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters);
    
    // ! the compressible Navier-Stokes formulation bilinear form
    BFPtr bf();
    
    // ! the compressible Navier-Stokes formulation rhs
    RHSPtr rhs();
    
    // ! the compressible Navier-Stokes formulation bilinear form, with any time terms dropped
    BFPtr steadyBF();
    
    void addVelocityTraceComponentCondition(SpatialFilterPtr region, FunctionPtr ui_exact, int i); // i is 1-based (1,2, or 3)
    void addVelocityTraceCondition(SpatialFilterPtr region, FunctionPtr u_exact); // vector u_exact
    
    void addTemperatureTraceCondition(SpatialFilterPtr region, FunctionPtr T_exact);
    
    void addMassFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact); // vector u_exact
    void addMassFluxCondition(SpatialFilterPtr region, FunctionPtr value);
    
    void addMomentumComponentFluxCondition(SpatialFilterPtr region, FunctionPtr tm_i_exact, int i);
    void addMomentumComponentFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact, int i);
    void addMomentumFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact);
    
    void addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr value);
    void addEnergyFluxCondition(SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact);
    
    // ! returns true if this is a space-time formulation; false otherwise.
    bool isSpaceTime() const;
    
    // ! returns true if this is a steady formulation; false otherwise.
    bool isSteady() const;
    
    // ! returns true if this is a time-stepping formulation; false otherwise.
    bool isTimeStepping() const;
    
    // ! declare inner product
    void setIP(IPPtr ip);
    
    void setIP( std::string normName );
    
    // ! L^2 norm of the difference in u1, u2, and p from previous time step, normalized
    // double relativeL2NormOfTimeStep();
    
    // ! returns the L^2 norm of the incremental solution
    double L2NormSolutionIncrement();
    
    // ! returns the L^2 norm of conservation variables in the incremental solution
    double L2NormSolutionIncrementConservation();
    
    // ! returns the L^2 norm of the background flow
    double L2NormSolution();
    
    // ! returns the nonlinear iteration count (since last refinement)
    int nonlinearIterationCount();
    
    // ! Loads the mesh and solution from disk, if they were previously saved using save().  In the present
    // ! implementation, assumes that the constructor arguments provided to CompressibleNavierStokesFormulation were the same
    // ! on the CompressibleNavierStokesFormulation on which save() was invoked as they were for this CompressibleNavierStokesFormulation instance.
    void load(std::string prefixString);
    
    // ! Returns viscosity mu.
    double mu();
    
    // ! Set viscosity
    void setMu(double value);
    
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
    // RHSPtr rhs(FunctionPtr f, bool excludeFluxesAndTraces);
    // RHSPtr rhs(bool excludeFluxesAndTraces);
    
    // ! Saves the solution(s) and mesh to an HDF5 format.
    void save(std::string prefixString);
    
    // ! set the RefinementStrategy to use for driving refinements
    void setRefinementStrategy(RefinementStrategyPtr refStrategy);
    
    // ! get the Solver used for the linear updates
    SolverPtr getSolver();
    
    // ! get the status of the last solve
    int getSolveCode();
    
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
    
    // ! Returns the solution (at previous time)
    SolutionPtr solutionPreviousTimeStep();
    
    // ! The first time this is called, calls solution()->solve(), and the weight argument is ignored.  After the first call, solves for the next iterate, and adds to background flow with the specified weight.
    double solveAndAccumulate();
    
    // ! Returns the L2 norm of the time residual
    double timeResidual();
    
    // ! Solves
    void solve();
    
    // ! Solves iteratively
    void solveIteratively(int maxIters, double cgTol, int azOutputLevel = 0, bool suppressSuperLUOutput = true);
    
    // ! Returns the spatial dimension.
    int spaceDim();
    
    // ! Takes a time step
    // void takeTimeStep();
    
    // ! Returns the sum of the time steps taken thus far.
    // double getTime();
    
    // ! Returns a FunctionPtr which gets updated with the current time.  Useful for setting BCs that vary in time.
    // FunctionPtr getTimeFunction();
    
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
    
    // ! For an exact solution (u, rho, T), produces a map that includes these as well as solution variables that depend on them (fields, traces, fluxes).
    // ! If includeFluxParity is true, then fluxes includes the side parity weight which gives a uniquely defined value everywhere, suitable for projection onto a Solution object
    // ! If includeFluxParity is false, fluxes do not include the side parity weight (suitable for substitution into a bilinear form, or linear form, as in BF::testFunctional()).
    std::map<int, FunctionPtr> exactSolutionMap(FunctionPtr u, FunctionPtr rho, FunctionPtr T, bool includeFluxParity);
    
    // ! For an exact solution (u, rho, T), returns the corresponding tc flux
    // ! If includeParity is true, then includes the side parity weight which gives a uniquely defined value everywhere, suitable for projection onto a Solution object
    // ! If includeParity is false, does not include the side parity weight (suitable for substitution into a bilinear form, or linear form, as in BF::testFunctional()).
    FunctionPtr exactSolution_tc(FunctionPtr u, FunctionPtr rho, FunctionPtr T, bool includeParity);
    
    // ! For an exact solution (u, rho, T), returns the corresponding tm flux.  (Since this is in general vector-valued, returns components in a std::vector.)
    // ! If includeParity is true, then includes the side parity weight which gives a uniquely defined value everywhere, suitable for projection onto a Solution object
    // ! If includeParity is false, does not include the side parity weight (suitable for substitution into a bilinear form, or linear form, as in BF::testFunctional()).
    std::vector<FunctionPtr> exactSolution_tm(FunctionPtr u, FunctionPtr rho, FunctionPtr T, bool includeParity);
    
    // ! For an exact solution (u, rho, T), returns the corresponding te flux
    // ! If includeParity is true, then includes the side parity weight which gives a uniquely defined value everywhere, suitable for projection onto a Solution object
    // ! If includeParity is false, does not include the side parity weight (suitable for substitution into a bilinear form, or linear form, as in BF::testFunctional()).
    FunctionPtr exactSolution_te(FunctionPtr u, FunctionPtr rho, FunctionPtr T, bool includeParity);
    
    // ! For an exact solution (u, rho, T), returns the corresponding forcing in the continuity equation
    FunctionPtr exactSolution_fc(FunctionPtr u, FunctionPtr rho, FunctionPtr T);
    
    // ! For an exact solution (u, rho, T), returns the corresponding forcing in the momentum equation
    std::vector<FunctionPtr> exactSolution_fm(FunctionPtr u, FunctionPtr rho, FunctionPtr T);
    
    // ! For an exact solution (u, rho, T), returns the corresponding forcing in the energy equation
    FunctionPtr exactSolution_fe(FunctionPtr u, FunctionPtr rho, FunctionPtr T);
    
    // ! returns a std::map indicating any trial variables that have adjusted polynomial orders relative to the standard poly order for the element.  Keys are variable IDs, values the difference between the indicated variable and the standard polynomial order.
    const std::map<int,int> &getTrialVariablePolyOrderAdjustments();
    
    // ! zeros out the solution increment
    void clearSolutionIncrement();
    
    Teuchos::ParameterList getConstructorParameters() const;
    
    FunctionPtr getTimeStep();
    
    // ! Set the forcing functions for problem.  f_momentum should have components equal to the number of spatial dimensions
    void setForcing(FunctionPtr f_continuity, std::vector<FunctionPtr> f_momentum, FunctionPtr f_energy);
    
    // static utility functions:
    static Teuchos::RCP<CompressibleNavierStokesFormulation> steadyFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                                                       MeshTopologyPtr meshTopo, int polyOrder, int delta_k);
    
    static Teuchos::RCP<CompressibleNavierStokesFormulation> timeSteppingFormulation(int spaceDim, double Re, bool useConformingTraces,
                                                                                             MeshTopologyPtr meshTopo, int spatialPolyOrder, int delta_k);
    
    static Teuchos::RCP<CompressibleNavierStokesFormulation> steadyEulerFormulation(int spaceDim, bool useConformingTraces, MeshTopologyPtr meshTopo,
                                                                                            int spatialPolyOrder, int delta_k);
    
    static Teuchos::RCP<CompressibleNavierStokesFormulation> timeSteppingEulerFormulation(int spaceDim, bool useConformingTraces,
                                                                                                  MeshTopologyPtr meshTopo,
                                                                                                  int spatialPolyOrder, int delta_k);
  };
}


#endif /* CompressibleNavierStokesFormulation_h */
