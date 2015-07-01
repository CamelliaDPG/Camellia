//
//  SpaceTimeIncompressibleFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#ifndef Camellia_SpaceTimeIncompressibleFormulation_h
#define Camellia_SpaceTimeIncompressibleFormulation_h

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
class IncompressibleProblem;

class SpaceTimeIncompressibleFormulation
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

  static const string s_u1, s_u2, s_u3;
  static const string s_sigma11, s_sigma12, s_sigma13;
  static const string s_sigma21, s_sigma22, s_sigma23;
  static const string s_sigma31, s_sigma32, s_sigma33;
  static const string s_p;

  static const string s_u1hat, s_u2hat, s_u3hat;
  static const string s_tm1hat, s_tm2hat, s_tm3hat;

  static const string s_v1, s_v2, s_v3;
  static const string s_tau1;
  static const string s_tau2;
  static const string s_tau3;
  static const string s_q;

public:
  SpaceTimeIncompressibleFormulation(Teuchos::RCP<IncompressibleProblem> problem, Teuchos::ParameterList &parameters);

  // ! the formulation's variable factory
  VarFactoryPtr vf();

  // ! the formulation's bilinear form
  BFPtr bf();

  // ! the formulation's bilinear form
  IPPtr ip(string normName);

  // ! Loads the mesh and solution from disk, if they were previously saved using save().  In the present
  // ! implementation, assumes that the constructor arguments provided to SpaceTimeIncompressibleFormulation were the same
  // ! on the SpaceTimeIncompressibleFormulation on which save() was invoked as they were for this SpaceTimeIncompressibleFormulation.
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
  VarPtr sigma(int i, int j);
  VarPtr u(int i);
  VarPtr p();

  // traces:
  VarPtr uhat(int i);
  VarPtr tmhat(int i);

  // test variables:
  VarPtr v(int i);
  VarPtr tau(int i);
  VarPtr q();

  set<int> nonlinearVars();
};
typedef Teuchos::RCP<SpaceTimeIncompressibleFormulation> SpaceTimeIncompressibleFormulationPtr;
}


#endif
