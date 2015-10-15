//
//  OldroydBFormulation.cpp
//  Camellia
//
//  Created by Truman Ellis on 9/28/15.
//
//

#include "OldroydBFormulation.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PreviousSolutionFunction.h"

using namespace Camellia;

const string OldroydBFormulation::s_u = "u";
const string OldroydBFormulation::s_tc = "tc";
const string OldroydBFormulation::s_uhat = "uhat";
const string OldroydBFormulation::s_v = "v";

// OldroydBFormulation::OldroydBFormulation(int spaceDim, double epsilon, TFunctionPtr<double> beta, bool useConformingTraces)
OldroydBFormulation::OldroydBFormulation(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
{
  int spaceDim = parameters.get<int>("spaceDim", 1);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces", false);
  int fieldPolyOrder = parameters.get<int>("fieldPolyOrder", 2);
  int delta_p = parameters.get<int>("delta_p", 2);
  int numTElems = parameters.get<int>("numTElems", 1);
  string norm = parameters.get<string>("norm", "Graph");
  string savedSolutionAndMeshPrefix = parameters.get<string>("savedSolutionAndMeshPrefix", "");

  _spaceDim = spaceDim;
  _useConformingTraces = useConformingTraces;

  FunctionPtr zero = Function::constant(1);
  FunctionPtr one = Function::constant(1);

  if ((spaceDim != 1) && (spaceDim != 2) && (spaceDim != 3))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim must be 1, 2, or 3");
  }

  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr u;

  // traces
  VarPtr tc;
  VarPtr uhat;

  // tests
  VarPtr v;

  _vf = VarFactory::varFactory();
  u = _vf->fieldVar(s_u);

  TFunctionPtr<double> n_x = TFunction<double>::normal(); // spatial normal
  TFunctionPtr<double> n_x_parity = n_x * TFunction<double>::sideParity();
  TFunctionPtr<double> n_xt = TFunction<double>::normalSpaceTime();
  TFunctionPtr<double> n_xt_parity = n_xt * TFunction<double>::sideParity();

  tc = _vf->fluxVar(s_tc);

  v = _vf->testVar(s_v, HGRAD);

  _bf = Teuchos::rcp( new BF(_vf) );


  // Define mesh
  BCPtr bc = BC::bc();

  vector<int> H1Order(2);
  H1Order[0] = fieldPolyOrder + 1;
  H1Order[1] = fieldPolyOrder + 1; // for now, use same poly. degree for temporal bases...
  if (savedSolutionAndMeshPrefix == "")
  {
    map<int,int> trialOrderEnhancements;
    _mesh = Teuchos::rcp( new Mesh(meshTopo, _bf, H1Order, delta_p, trialOrderEnhancements) ) ;

    _solutionUpdate = Solution::solution(_bf, _mesh, bc);
    _solutionBackground = Solution::solution(_bf, _mesh, bc);
    map<int, FunctionPtr> initialGuess;
    initialGuess[u()->ID()] = Function::constant(1);

    _solutionBackground->projectOntoMesh(initialGuess);
  }
  else
  {
    // // BFPTR version should be deprecated
    _mesh = MeshFactory::loadFromHDF5(_bf, savedSolutionAndMeshPrefix+".mesh");
    _solutionBackground = Solution::solution(_bf, _mesh, bc);
    _solutionBackground->loadFromHDF5(savedSolutionAndMeshPrefix+".soln");
    _solutionUpdate = Solution::solution(_bf, _mesh, bc);
    // _solutionUpdate->loadFromHDF5(savedSolutionAndMeshPrefix+"_update.soln");
  }

  FunctionPtr u_prev = Function::solution(u, _solutionBackground);
  FunctionPtr uhat_prev;




  // v terms
  _bf->addTerm(-u, v->dt());
  _bf->addTerm(-u_prev*u, v->dx());
  _bf->addTerm(tc, v);

  // Add residual to RHS
  _rhs = RHS::rhs();
  // v terms
  _rhs->addTerm( u_prev * v->dt());
  _rhs->addTerm( 0.5*u_prev*u_prev * v->dx());

  vector<VarPtr> missingTestVars = _bf->missingTestVars();
  vector<VarPtr> missingTrialVars = _bf->missingTrialVars();
  for (int i=0; i < missingTestVars.size(); i++)
  {
    VarPtr var = missingTestVars[i];
    cout << var->displayString() << endl;
  }
  for (int i=0; i < missingTrialVars.size(); i++)
  {
    VarPtr var = missingTrialVars[i];
    cout << var->displayString() << endl;
  }

  _ips["Graph"] = _bf->graphNorm();

  IPPtr ip = _ips.at(norm);

  _solutionUpdate->setRHS(_rhs);
  _solutionUpdate->setIP(ip);

  _mesh->registerSolution(_solutionBackground);
  _mesh->registerSolution(_solutionUpdate);

  LinearTermPtr residual = _rhs->linearTerm() - _bf->testFunctional(_solutionUpdate, false); // false: don't exclude boundary terms

  double energyThreshold = 0.2;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy( _mesh, residual, ip, energyThreshold ) );
}

VarFactoryPtr OldroydBFormulation::vf()
{
  return _vf;
}

BFPtr OldroydBFormulation::bf()
{
  return _bf;
}

IPPtr OldroydBFormulation::ip(string normName)
{
  return _ips.at(normName);
}

RefinementStrategyPtr OldroydBFormulation::getRefinementStrategy()
{
  return _refinementStrategy;
}

void OldroydBFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void OldroydBFormulation::refine()
{
  _refinementStrategy->refine();
}

// RHSPtr OldroydBFormulation::rhs(TFunctionPtr<double> f)
// {
//   RHSPtr rhs = RHS::rhs();
//
//   VarPtr v = this->v();
//
//   if (f != Teuchos::null)
//   {
//     rhs->addTerm( f * v );
//   }
//
//   return rhs;
// }

VarPtr OldroydBFormulation::u()
{
  return _vf->fieldVar(s_u);
}

// traces:
VarPtr OldroydBFormulation::tc()
{
  return _vf->fluxVar(s_tc);
}

VarPtr OldroydBFormulation::uhat()
{
  // return _vf->fluxVar(s_uhat);
  return _vf->traceVar(s_uhat);
}

// ! Saves the solution(s) and mesh to an HDF5 format.
void OldroydBFormulation::save(std::string prefixString)
{
  _solutionUpdate->mesh()->saveToHDF5(prefixString+".mesh");
  _solutionUpdate->saveToHDF5(prefixString+".soln");
}

// ! Solves
void OldroydBFormulation::solve()
{
  _solutionUpdate->solve();
}

VarPtr OldroydBFormulation::v()
{
  return _vf->testVar(s_v, HGRAD);
}

set<int> OldroydBFormulation::nonlinearVars()
{
  set<int> nonlinearVars;
  nonlinearVars.insert(u()->ID());
  return nonlinearVars;
}

// ! Returns the solution
SolutionPtr OldroydBFormulation::solutionUpdate()
{
  return _solutionUpdate;
}

// ! Returns the solution
SolutionPtr OldroydBFormulation::solutionBackground()
{
  return _solutionBackground;
}

void OldroydBFormulation::updateSolution()
{
  double alpha = 1;
  vector<int> trialIDs = _vf->trialIDs();
  set<int> trialIDSet(trialIDs.begin(), trialIDs.end());
  set<int> nlVars = nonlinearVars();
  set<int> lVars;
  set_difference(trialIDSet.begin(), trialIDSet.end(), nlVars.begin(), nlVars.end(),
      std::inserter(lVars, lVars.end()));
  _solutionBackground->addReplaceSolution(_solutionUpdate, alpha, nlVars, lVars);
}
