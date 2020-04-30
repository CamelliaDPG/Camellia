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
#include "QuadraturePointFunction.h"
#include "SolutionStage.h"
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
  
  Teuchos::RCP<QuadraturePointFunction> _f_velocityIntegral; // integral of f over all velocity space
  FunctionPtr _E_extruded;  // E solution extended to the Boltzmann mesh
  
  std::map<GlobalIndexType, std::vector<GlobalIndexType> > _spatialCellIDToPhaseSpaceCellIDs;
  std::map<GlobalIndexType, GlobalIndexType> _phaseSpaceCellIDToSpatialCellID;
  
  // Boltzmann itself (discretization is 2D because 1D1V):
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
  Boltzmann1D1VFormulation(Boltzmann1D1VFormulationChoice formulationChoice, double q, double eps0, double m, double dt,
                           const double x0, const double xWidth, const int xElements,
                           const double v0, const double vWidth, const int vElements,
                           const int H1Order, const int delta_k, ERKTableauPtr erkTableau);

  //! set initial conditions.  Solves for initial E, E_hat, and sets t=0;
  void setInitialConditions(FunctionPtr f0);
  
  //! take time step.
  void takeTimeStep();
  
  void setTimeStep(double dt);
  
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
  
  SolutionPtr boltzmannSolution();
  SolutionPtr boltzmannPreviousSolution();
  SolutionPtr gaussSolution();
  
  FunctionPtr E_extruded();
  
  bool usesImplicitSolve() const;
  bool hasL2Trace() const;
  
  //! notification that the boltzmann solve has set the RK stage:
  void setStage(const int stageNumber);
  
  //! map from gauss mesh cellIDs to corresponding cellIDs in boltzmann mesh
  const std::map<GlobalIndexType, std::vector<GlobalIndexType> > &spatialCellIDToPhaseSpaceCellIDsMap();
  
  //! map that is reverse of the above; there is exactly one spatial cell for each phase space cell
  const std::map<GlobalIndexType, GlobalIndexType> &phaseSpaceCellIDToSpatialCellID();
};
}

#endif
