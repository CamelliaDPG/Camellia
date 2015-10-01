//
//  InviscidBurgersFormulation.cpp
//  Camellia
//
//  Created by Truman Ellis on 9/28/15.
//
//

#include "InviscidBurgersFormulation.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PreviousSolutionFunction.h"

using namespace Camellia;

const string InviscidBurgersFormulation::s_u = "u";
const string InviscidBurgersFormulation::s_tc = "tc";
const string InviscidBurgersFormulation::s_uhat = "uhat";
const string InviscidBurgersFormulation::s_v = "v";

// InviscidBurgersFormulation::InviscidBurgersFormulation(int spaceDim, double epsilon, TFunctionPtr<double> beta, bool useConformingTraces)
InviscidBurgersFormulation::InviscidBurgersFormulation(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
{
  int spaceDim = parameters.get<int>("spaceDim", 1);
  bool linearTrace = parameters.get<bool>("linearTrace", false);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces", false);
  int fieldPolyOrder = parameters.get<int>("fieldPolyOrder", 2);
  int delta_p = parameters.get<int>("delta_p", 2);
  int numTElems = parameters.get<int>("numTElems", 1);
  string norm = parameters.get<string>("norm", "Graph");
  string savedSolutionAndMeshPrefix = parameters.get<string>("savedSolutionAndMeshPrefix", "");

  _spaceDim = spaceDim;
  _linearTrace = linearTrace;
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

  // LinearTermPtr tc_lt;
  // if (spaceDim == 1)
  // {
  //   tc_lt = beta->x()*n_x_parity->x()*u
  //     -sigma1 * n_x_parity->x()
  //     + u*n_xt_parity->t();
  // }
  // else if (spaceDim == 2)
  // {
  //   tc_lt = beta->x()*n_x_parity->x()*u
  //     + beta->y()*n_x_parity->y()*u
  //     - sigma1 * n_x_parity->x()
  //     - sigma2 * n_x_parity->y()
  //     + u*n_xt_parity->t();
  // }
  // else if (spaceDim == 3)
  // {
  //   tc_lt = beta->x()*n_x_parity->x()*u
  //     + beta->y()*n_x_parity->y()*u
  //     + beta->z()*n_x_parity->z()*u
  //     - sigma1 * n_x_parity->x()
  //     - sigma2 * n_x_parity->y()
  //     - sigma3 * n_x_parity->z()
  //     + u*n_xt_parity->t();
  // }
  // tc = _vf->fluxVar(s_tc, tc_lt);
  if (!linearTrace)
    tc = _vf->fluxVar(s_tc);
  else
    // uhat = _vf->fluxVar(s_uhat);
    uhat = _vf->traceVar(s_uhat, 1.0 * u, L2);

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
    // if (linearTrace)
    //   trialOrderEnhancements[uhat->ID()] = 3;
    _mesh = Teuchos::rcp( new Mesh(meshTopo, _bf, H1Order, delta_p, trialOrderEnhancements) ) ;

    _solutionUpdate = Solution::solution(_bf, _mesh, bc);
    _solutionBackground = Solution::solution(_bf, _mesh, bc);
    map<int, FunctionPtr> initialGuess;
    initialGuess[u()->ID()] = Function::constant(1);
    if (linearTrace)
      initialGuess[uhat()->ID()] = Function::constant(1);

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
  if (linearTrace)
    uhat_prev = Function::solution(uhat, _solutionBackground);




  // v terms
  _bf->addTerm(-u, v->dt());
  _bf->addTerm(-u_prev*u, v->dx());
  if (!linearTrace)
    _bf->addTerm(tc, v);
  else
  {
    // _bf->addTerm(uhat, v);
    _bf->addTerm(uhat_prev*uhat*n_xt->x(), v);
    _bf->addTerm(uhat*n_xt->t(), v);
    // _bf->addTerm(uhat_prev*uhat*n_x_parity->x(), v);
    // _bf->addTerm(uhat*n_xt_parity->t(), v);
    // _bf->addTerm(uhat_prev*uhat*n_x->x(), v);
    // _bf->addTerm(uhat*n_xt->t(), v);
    // _bf->addTerm(uhat_prev*uhat, v);
    // _bf->addTerm(uhat, v);
  }

  // Add residual to RHS
  _rhs = RHS::rhs();
  // v terms
  _rhs->addTerm( u_prev * v->dt());
  _rhs->addTerm( 0.5*u_prev*u_prev * v->dx());
  if (linearTrace)
  {
    _rhs->addTerm( -0.5*uhat_prev*uhat_prev*n_xt->x() * v);
    _rhs->addTerm( -uhat_prev*n_xt->t() * v);
  }

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

  // _ips["CoupledRobust"] = Teuchos::rcp(new IP);
  // _ips["CoupledRobust"]->addTerm(_beta*v->grad());
  // _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_epsilon)))*tau);
  // _ips["CoupledRobust"]->addTerm(sqrt(_epsilon)*v->grad());
  // _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_epsilon)*one/Function::h(),one)*v);
  // if (spaceDim > 1)
  //   _ips["CoupledRobust"]->addTerm(tau->div() - v->dt() - beta*v->grad());
  // else
  //   _ips["CoupledRobust"]->addTerm(tau->dx() - v->dt() - beta*v->grad());

  // _ips["NSDecoupledH1"] = Teuchos::rcp(new IP);
  // _ips["NSDecoupledH1"]->addTerm(one/Function::h()*tau);
  // _ips["NSDecoupledH1"]->addTerm(v->grad());
  // _ips["NSDecoupledH1"]->addTerm(_beta*v->grad()+v->dt());
  // if (spaceDim > 1)
  //   _ips["NSDecoupledH1"]->addTerm(tau->div());
  // else
  //   _ips["NSDecoupledH1"]->addTerm(tau->dx());
  // _ips["NSDecoupledH1"]->addTerm(v);

  // _ips["NSDecoupledMin"] = Teuchos::rcp(new IP);
  // _ips["NSDecoupledMin"]->addTerm(Function::min(one/Function::h(), one/Function::constant(sqrt(_epsilon)))*tau);
  // _ips["NSDecoupledMin"]->addTerm(v->grad());
  // _ips["NSDecoupledMin"]->addTerm(_beta*v->grad()+v->dt());
  // if (spaceDim > 1)
  //   _ips["NSDecoupledMin"]->addTerm(tau->div());
  // else
  //   _ips["NSDecoupledMin"]->addTerm(tau->dx());
  // _ips["NSDecoupledMin"]->addTerm(v);

  IPPtr ip = _ips.at(norm);
  // if (problem->forcingFunction != Teuchos::null)
  // {
  //   _rhs->addTerm(problem->forcingFunction->x() * v1);
  //   _rhs->addTerm(problem->forcingFunction->y() * v2);
  // }

  _solutionUpdate->setRHS(_rhs);
  _solutionUpdate->setIP(ip);

  _mesh->registerSolution(_solutionBackground);
  _mesh->registerSolution(_solutionUpdate);

  LinearTermPtr residual = _rhs->linearTerm() - _bf->testFunctional(_solutionUpdate, false); // false: don't exclude boundary terms

  double energyThreshold = 0.2;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy( _mesh, residual, ip, energyThreshold ) );
}

VarFactoryPtr InviscidBurgersFormulation::vf()
{
  return _vf;
}

BFPtr InviscidBurgersFormulation::bf()
{
  return _bf;
}

IPPtr InviscidBurgersFormulation::ip(string normName)
{
  return _ips.at(normName);
}

// void InviscidBurgersFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k, string norm,
//     TLinearTermPtr<double> forcingTerm)
// {
//   this->initializeSolution(meshTopo, fieldPolyOrder, delta_k, norm, forcingTerm, "");
// }
//
// void InviscidBurgersFormulation::initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k, string norm,
//     TLinearTermPtr<double> forcingTerm)
// {
//   this->initializeSolution(Teuchos::null, fieldPolyOrder, delta_k, norm, forcingTerm, filePrefix);
// }
//
// void InviscidBurgersFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k, string norm,
//     TLinearTermPtr<double> forcingTerm, string savedSolutionAndMeshPrefix)
// {
//   TEUCHOS_TEST_FOR_EXCEPTION(meshTopo->getDimension() != _spaceDim + 1, std::invalid_argument, "MeshTopo must be space-time mesh");
//
//   BCPtr bc = BC::bc();
//
//   vector<int> H1Order(2);
//   H1Order[0] = fieldPolyOrder + 1;
//   H1Order[1] = fieldPolyOrder + 1; // for now, use same poly. degree for temporal bases...
//   MeshPtr mesh;
//   if (savedSolutionAndMeshPrefix == "")
//   {
//     mesh = Teuchos::rcp( new Mesh(meshTopo, _bf, H1Order, delta_k) ) ;
//     _solution = TSolution<double>::solution(mesh,bc);
//   }
//   else
//   {
//     mesh = MeshFactory::loadFromHDF5(_bf, savedSolutionAndMeshPrefix+".mesh");
//     _solution = TSolution<double>::solution(mesh, bc);
//     _solution->loadFromHDF5(savedSolutionAndMeshPrefix+".soln");
//   }
//
//   IPPtr ip = _ips.at(norm);
//   // RHSPtr rhs = this->rhs(forcingFunction); // in transient case, this will refer to _previousSolution
//   if (forcingTerm != Teuchos::null)
//     _rhs->addTerm(forcingTerm);
//
//   _solution->setRHS(_rhs);
//   _solution->setIP(ip);
//
//   mesh->registerSolution(_solution); // will project both time steps during refinements...
//
//   LinearTermPtr residual = _rhs->linearTerm() - _bf->testFunctional(_solution,false); // false: don't exclude boundary terms
//
//   double energyThreshold = 0.2;
//   _refinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold ) );
// }

RefinementStrategyPtr InviscidBurgersFormulation::getRefinementStrategy()
{
  return _refinementStrategy;
}

void InviscidBurgersFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void InviscidBurgersFormulation::refine()
{
  _refinementStrategy->refine();
}

// RHSPtr InviscidBurgersFormulation::rhs(TFunctionPtr<double> f)
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

VarPtr InviscidBurgersFormulation::u()
{
  return _vf->fieldVar(s_u);
}

// traces:
VarPtr InviscidBurgersFormulation::tc()
{
  return _vf->fluxVar(s_tc);
}

VarPtr InviscidBurgersFormulation::uhat()
{
  // return _vf->fluxVar(s_uhat);
  return _vf->traceVar(s_uhat);
}

// ! Saves the solution(s) and mesh to an HDF5 format.
void InviscidBurgersFormulation::save(std::string prefixString)
{
  _solutionUpdate->mesh()->saveToHDF5(prefixString+".mesh");
  _solutionUpdate->saveToHDF5(prefixString+".soln");
}

// ! Solves
void InviscidBurgersFormulation::solve()
{
  _solutionUpdate->solve();
}

VarPtr InviscidBurgersFormulation::v()
{
  return _vf->testVar(s_v, HGRAD);
}

set<int> InviscidBurgersFormulation::nonlinearVars()
{
  set<int> nonlinearVars;
  nonlinearVars.insert(u()->ID());
  if (_linearTrace)
    nonlinearVars.insert(uhat()->ID());
  return nonlinearVars;
}

// ! Returns the solution
SolutionPtr InviscidBurgersFormulation::solutionUpdate()
{
  return _solutionUpdate;
}

// ! Returns the solution
SolutionPtr InviscidBurgersFormulation::solutionBackground()
{
  return _solutionBackground;
}

void InviscidBurgersFormulation::updateSolution()
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
