// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  Boltzmann1D3VFormulation.h
//  Camellia
//
//  Created by Nate Roberts on March 3, 2020.
//
//

#ifndef Camellia_Boltzmann1D3VFormulation_h
#define Camellia_Boltzmann1D3VFormulation_h

#include "TypeDefs.h"

#include "BF.h"
#include "GMGSolver.h"
#include "PhasePhysicalMeshMap.h"
#include "QuadraturePointFunction.h"
#include "SolutionMomentFunction.h"
#include "SolutionStage.h"
#include "TensorUtilities.h"
#include "VarFactory.h"

namespace Camellia
{

  class Boltzmann1D3VFormulation : public SolutionStageObserver
{
public:
  enum Boltzmann1D3VFormulationChoice
  {
    ULTRAWEAK_EXPLICIT_L2_TRACE,
    ULTRAWEAK_EXPLICIT_H1_TRACE,
    ULTRAWEAK_IMPLICIT_L2_TRACE,
    ULTRAWEAK_IMPLICIT_H1_TRACE
  };
private:
  Boltzmann1D3VFormulationChoice _formulationChoice;
  
  BFPtr _boltzmannBF;
  BFPtr _gaussBF;
  int _delta_k;
  
  SolutionPtr _boltzmannSoln;
  SolutionPtr _boltzmannSolnPrev;
  
  double _gaussMeshMeasure;
  double _boltzmannMeshMeasure;
  
  bool _haveTableau = false;
  
  SolutionPtr _gaussSoln;
  double _t = 0; // current time
  
  // set by user:
  ParameterFunctionPtr _dt;
  ParameterFunctionPtr _q;
  ParameterFunctionPtr _eps0;
  ParameterFunctionPtr _m;
  
  // set during solve process:
  ParameterFunctionPtr _n0; // integral of f over the whole 1D1V space
  
  Teuchos::RCP<SolutionMomentFunction> _f_velocityIntegral; // integral of f over all velocity space
  FunctionPtr _Ex_extruded;  // E_x solution extended to the Boltzmann mesh
    
  Teuchos::RCP<PhasePhysicalMeshMap> _phaseSpacePhysicalMeshMap;
  
  Teuchos::RCP<GMGSolver> _gmgSolver;
  
  // Boltzmann itself (discretization is 4D because 1D3V):
  static const string S_F;       // trial variable - distribution function f
  static const string S_F_HAT;   // trace variable
  static const string S_T_N_HAT; // trace variable
  static const string S_W;       // test variable
  
  // Gauss's law:
  static const string S_E;     // trial variable - electric field E
  static const string S_E_HAT; // trace variable - trace of E in H^1/2
  static const string S_TAU;   // test variable
  
  void updateElectricField();
public:
  Boltzmann1D3VFormulation(Boltzmann1D3VFormulationChoice formulationChoice, double q, double eps0, double m, double dt,
                           const double x0, const double xWidth, const int xElements,
                           const Kokkos::Array<double,3> v0, const Kokkos::Array<double,3> vDomainWidths, const Kokkos::Array<int,3> vMeshWidths,
                           const int H1Order, const int delta_k, const bool usePeriodicBCs, ERKTableauPtr erkTableau);

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
  VarPtr E_hat();   // Gauss

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
  
  //! Set the boundary conditions for Boltzmann (relevant for the non-periodic case).
  void setBoltzmannBCs(BCPtr bc);
  
  //! Set the boundary conditions for Gauss (relevant for the non-periodic case).
  void setGaussBCs(BCPtr bc);
  
  //! notification that the boltzmann solve has set the RK stage:
  void setStage(const int stageNumber);
};

}

#endif
