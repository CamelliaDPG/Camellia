//
//  SpaceTimeHeatDivFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#ifndef Camellia_SpaceTimeHeatDivFormulation_h
#define Camellia_SpaceTimeHeatDivFormulation_h

#include "TypeDefs.h"

#include "VarFactory.h"
#include "BF.h"
#include "SpatialFilter.h"
#include "Solution.h"
#include "RefinementStrategy.h"
#include "ParameterFunction.h"
#include "PoissonFormulation.h"

namespace Camellia
{
class SpaceTimeHeatDivFormulation
{

  int _spaceDim;
  bool _useConformingTraces;
  double _epsilon;

  VarFactoryPtr _vf;
  BFPtr _bf;
  map<string, IPPtr> _ips;
  RHSPtr _rhs;

  TSolutionPtr<double> _solution;
  SolverPtr _solver;
  RefinementStrategyPtr _refinementStrategy;

  static const string s_u;
  static const string s_sigma1, s_sigma2, s_sigma3;

  static const string s_uhat;
  static const string s_tc;

  static const string s_v;
  static const string s_tau;

  // ! initialize the Solution object(s) using the provided MeshTopology
  void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k, string norm,
                          TLinearTermPtr<double> forcingTerm, std::string fileToLoadPrefix);
public:
  SpaceTimeHeatDivFormulation(int spaceDim, double epsilon, bool useConformingTraces = false);

  // ! the formulation's variable factory
  VarFactoryPtr vf();

  // ! the formulation's bilinear form
  BFPtr bf();

  // ! the formulation's bilinear form
  IPPtr ip(string normName);

  // ! initialize the Solution object(s) using the provided MeshTopology
  void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k = 1, string norm = "Graph",
                          TLinearTermPtr<double> forcingTerm = Teuchos::null);

  // ! initialize the Solution object(s) from file
  void initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k = 1, string norm = "Graph",
                          TLinearTermPtr<double> forcingTerm = Teuchos::null);

  // ! Loads the mesh and solution from disk, if they were previously saved using save().  In the present
  // ! implementation, assumes that the constructor arguments provided to SpaceTimeHeatDivFormulation were the same
  // ! on the SpaceTimeHeatDivFormulation on which save() was invoked as they were for this SpaceTimeHeatDivFormulation.
  void load(std::string prefixString);

  // ! Returns epsilon.
  double epsilon();

  // ! refine according to energy error in the solution
  void refine();

  // ! returns the RefinementStrategy object being used to drive refinements
  RefinementStrategyPtr getRefinementStrategy();

  // ! Returns an RHSPtr corresponding to the scalar forcing function f and the formulation.
  // RHSPtr rhs(TFunctionPtr<double> f);

  // ! Saves the solution(s) and mesh to an HDF5 format.
  void save(std::string prefixString);

  // ! set the RefinementStrategy to use for driving refinements
  void setRefinementStrategy(RefinementStrategyPtr refStrategy);

  // ! Returns the solution (at current time)
  TSolutionPtr<double> solution();

  // ! Solves
  void solve();

  // field variables:
  VarPtr sigma(int i);
  VarPtr u();

  // traces:
  VarPtr uhat();
  VarPtr tc();

  // test variables:
  VarPtr v();
  VarPtr tau();

  static TFunctionPtr<double> forcingFunction(int spaceDim, double epsilon, TFunctionPtr<double> u);
};
}


#endif
