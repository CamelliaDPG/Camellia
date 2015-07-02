//
//  SpaceTimeCompressibleFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#ifndef Camellia_SpaceTimeCompressibleFormulation_h
#define Camellia_SpaceTimeCompressibleFormulation_h

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
class CompressibleProblem;

class SpaceTimeCompressibleFormulation
{

  int _spaceDim;
  bool _steady;
  bool _useConformingTraces;
  double _mu;

  VarFactoryPtr _vf;
  BFPtr _bf;
  map<string, IPPtr> _ips;
  RHSPtr _rhs;

  MeshPtr _mesh;
  SolutionPtr _solutionUpdate;
  SolutionPtr _solutionBackground;
  SolverPtr _solver;
  RefinementStrategyPtr _refinementStrategy;

  static const string s_rho;
  static const string s_u1, s_u2, s_u3;
  static const string s_D11, s_D12, s_D13;
  static const string s_D21, s_D22, s_D23;
  static const string s_D31, s_D32, s_D33;
  static const string s_T;
  static const string s_q1, s_q2, s_q3;

  static const string s_u1hat, s_u2hat, s_u3hat;
  static const string s_That;
  static const string s_tc;
  static const string s_tm1, s_tm2hat, s_tm3hat;
  static const string s_te;

  static const string s_vc;
  static const string s_vm1, s_vm2, s_vm3;
  static const string s_ve;
  static const string s_S11, s_S12, s_S13;
  static const string s_S21, s_S22, s_S23;
  static const string s_S31, s_S32, s_S33;
  static const string s_tau;

  // // ! initialize the Solution object(s) using the provided MeshTopology
  // void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k, string norm,
  //                         LinearTermPtr forcingTerm, std::string fileToLoadPrefix);
public:
  SpaceTimeCompressibleFormulation(Teuchos::RCP<CompressibleProblem> problem, Teuchos::ParameterList &parameters);

  int spaceDim();

  // ! the formulation's variable factory
  VarFactoryPtr vf();

  // ! the formulation's bilinear form
  BFPtr bf();

  // ! the formulation's bilinear form
  IPPtr ip(string normName);

  // // ! initialize the Solution object(s) using the provided MeshTopology
  // void initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k = 1, string norm = "Graph",
  //                         LinearTermPtr forcingTerm = Teuchos::null);

  // // ! initialize the Solution object(s) from file
  // void initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k = 1, string norm = "Graph",
  //                         LinearTermPtr forcingTerm = Teuchos::null);

  // ! Loads the mesh and solution from disk, if they were previously saved using save().  In the present
  // ! implementation, assumes that the constructor arguments provided to SpaceTimeCompressibleFormulation were the same
  // ! on the SpaceTimeCompressibleFormulation on which save() was invoked as they were for this SpaceTimeCompressibleFormulation.
  void load(std::string prefixString);

  // ! Returns mu.
  double mu();

  // ! refine according to energy error in the solution
  void refine();

  // ! returns the RefinementStrategy object being used to drive refinements
  RefinementStrategyPtr getRefinementStrategy();

  // ! Saves the solution(s) and mesh to an HDF5 format.
  void save(std::string prefixString);

  // ! set the RefinementStrategy to use for driving refinements
  void setRefinementStrategy(RefinementStrategyPtr refStrategy);

  // ! Returns the solution (at current time)
  SolutionPtr solutionUpdate();
  SolutionPtr solutionBackground();
  void updateSolution();

  // ! Solves
  void solve();

  // field variables:
  VarPtr rho();
  VarPtr u(int i);
  VarPtr T();
  VarPtr D(int i, int j);
  VarPtr q(int i);

  // traces:
  VarPtr uhat(int i);
  VarPtr That();
  VarPtr tc();
  VarPtr tm(int i);
  VarPtr te();

  // test variables:
  VarPtr vc();
  VarPtr vm(int i);
  VarPtr ve();
  VarPtr S(int i, int j);
  VarPtr tau();

  set<int> nonlinearVars();
};
typedef Teuchos::RCP<SpaceTimeCompressibleFormulation> SpaceTimeCompressibleFormulationPtr;
}


#endif
