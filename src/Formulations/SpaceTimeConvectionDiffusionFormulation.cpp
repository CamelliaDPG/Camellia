//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  SpaceTimeConvectionDiffusionFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#include "SpaceTimeConvectionDiffusionFormulation.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PoissonFormulation.h"
#include "RHS.h"

using namespace Camellia;

const string SpaceTimeConvectionDiffusionFormulation::s_u = "u";
const string SpaceTimeConvectionDiffusionFormulation::s_sigma1 = "sigma1";
const string SpaceTimeConvectionDiffusionFormulation::s_sigma2 = "sigma2";
const string SpaceTimeConvectionDiffusionFormulation::s_sigma3 = "sigma3";

const string SpaceTimeConvectionDiffusionFormulation::s_uhat = "uhat";
const string SpaceTimeConvectionDiffusionFormulation::s_tc = "tc";

const string SpaceTimeConvectionDiffusionFormulation::s_v = "v";
const string SpaceTimeConvectionDiffusionFormulation::s_tau = "tau";

// SpaceTimeConvectionDiffusionFormulation::SpaceTimeConvectionDiffusionFormulation(int spaceDim, double epsilon, TFunctionPtr<double> beta, bool useConformingTraces)
SpaceTimeConvectionDiffusionFormulation::SpaceTimeConvectionDiffusionFormulation(Teuchos::ParameterList &parameters, TFunctionPtr<double> beta)
{
  int spaceDim = parameters.get<int>("spaceDim", 1);
  bool steady = parameters.get<bool>("steady", true);
  double epsilon = parameters.get<double>("epsilon", 1e-2);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces", false);
  bool fluxLinearTerm = parameters.get<bool>("fluxLinearTerm", true);
  int fieldPolyOrder = parameters.get<int>("fieldPolyOrder", 2);
  int delta_p = parameters.get<int>("delta_p", 2);
  int numTElems = parameters.get<int>("numTElems", 1);
  string norm = parameters.get<string>("norm", "Graph");
  string savedSolutionAndMeshPrefix = parameters.get<string>("savedSolutionAndMeshPrefix", "");

  _spaceDim = spaceDim;
  _steady = steady;
  _epsilon = epsilon;
  _beta = beta;
  _useConformingTraces = useConformingTraces;
  TEUCHOS_TEST_FOR_EXCEPTION(epsilon==0, std::invalid_argument, "epsilon may not be 0!");

  FunctionPtr zero = Function::constant(1);
  FunctionPtr one = Function::constant(1);

  if ((spaceDim != 1) && (spaceDim != 2) && (spaceDim != 3))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim must be 1, 2, or 3");
  }

  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr u;
  VarPtr sigma1, sigma2, sigma3;

  // traces
  VarPtr uhat;
  VarPtr tc;

  // tests
  VarPtr v;
  VarPtr tau;

  _vf = VarFactory::varFactory();
  u = _vf->fieldVar(s_u);

  sigma1 = _vf->fieldVar(s_sigma1);
  if (spaceDim > 1) sigma2 = _vf->fieldVar(s_sigma2);
  if (spaceDim==3)
  {
    sigma3 = _vf->fieldVar(s_sigma3);
  }

  Space uHatSpace = useConformingTraces ? HGRAD : L2;

  uhat = _vf->traceVarSpaceOnly(s_uhat, 1.0 * u, uHatSpace);

  TFunctionPtr<double> n_x = TFunction<double>::normal(); // spatial normal
  TFunctionPtr<double> n_x_parity = n_x * TFunction<double>::sideParity();
  TFunctionPtr<double> n_xt = TFunction<double>::normalSpaceTime();
  TFunctionPtr<double> n_xt_parity = n_xt * TFunction<double>::sideParity();

  LinearTermPtr tc_lt;
  if (spaceDim == 1)
  {
    tc_lt = beta->x()*n_x_parity->x()*u
      -sigma1 * n_x_parity->x()
      + u*n_xt_parity->t();
  }
  else if (spaceDim == 2)
  {
    tc_lt = beta->x()*n_x_parity->x()*u
      + beta->y()*n_x_parity->y()*u
      - sigma1 * n_x_parity->x()
      - sigma2 * n_x_parity->y()
      + u*n_xt_parity->t();
  }
  else if (spaceDim == 3)
  {
    tc_lt = beta->x()*n_x_parity->x()*u
      + beta->y()*n_x_parity->y()*u
      + beta->z()*n_x_parity->z()*u
      - sigma1 * n_x_parity->x()
      - sigma2 * n_x_parity->y()
      - sigma3 * n_x_parity->z()
      + u*n_xt_parity->t();
  }
  if (fluxLinearTerm)
    tc = _vf->fluxVar(s_tc, tc_lt);
  else
    tc = _vf->fluxVar(s_tc);

  v = _vf->testVar(s_v, HGRAD);

  if (_spaceDim > 1)
    tau = _vf->testVar(s_tau, HDIV); // vector
  else
    tau = _vf->testVar(s_tau, HGRAD); // scalar

  _bf = Teuchos::rcp( new BF(_vf) );
  // v terms
  _bf->addTerm(-u, v->dt());
  _bf->addTerm(-beta->x()*u + sigma1, v->dx());
  if (_spaceDim > 1) _bf->addTerm(-beta->y()*u + sigma2, v->dy());
  if (_spaceDim > 2) _bf->addTerm(-beta->z()*u + sigma3, v->dz());
  _bf->addTerm(tc, v);

  // tau terms
  if (_spaceDim > 1)
  {
    _bf->addTerm((1.0 / _epsilon) * sigma1, tau->x());
    _bf->addTerm((1.0 / _epsilon) * sigma2, tau->y());
    if (_spaceDim > 2) _bf->addTerm((1.0 / _epsilon) * sigma3, tau->z());
    _bf->addTerm(u, tau->div());
    _bf->addTerm(-uhat, tau * n_x);
  }
  else
  {
    _bf->addTerm((1.0 / _epsilon) * sigma1, tau);
    _bf->addTerm(u, tau->dx());
    _bf->addTerm(-uhat, tau * n_x->x());
  }

  _ips["Graph"] = _bf->graphNorm();

  _ips["Math"] = Teuchos::rcp(new IP);
  _ips["Math"]->addTerm(v);
  _ips["Math"]->addTerm(v->grad());
  _ips["Math"]->addTerm(v->dt());
  _ips["Math"]->addTerm(tau);
  if (spaceDim > 1)
    _ips["Math"]->addTerm(tau->div());
  else
    _ips["Math"]->addTerm(tau->dx());

  _ips["Robust"] = Teuchos::rcp(new IP);
  _ips["Robust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_epsilon)))*tau);
  _ips["Robust"]->addTerm(sqrt(_epsilon)*v->grad());
  _ips["Robust"]->addTerm(Function::min(sqrt(_epsilon)*one/Function::h(),one)*v);
  _ips["Robust"]->addTerm(v->dt() + beta*v->grad());
  if (spaceDim > 1)
    _ips["Robust"]->addTerm(tau->div());
  else
    _ips["Robust"]->addTerm(tau->dx());

  _ips["FixedRobust"] = Teuchos::rcp(new IP);
  _ips["FixedRobust"]->addTerm(tau);
  _ips["FixedRobust"]->addTerm(sqrt(_epsilon)*v->grad());
  _ips["FixedRobust"]->addTerm(sqrt(_epsilon)*v);
  _ips["FixedRobust"]->addTerm(v->dt() + beta*v->grad());
  if (spaceDim > 1)
    _ips["FixedRobust"]->addTerm(tau->div());
  else
    _ips["FixedRobust"]->addTerm(tau->dx());

  _ips["CoupledRobust"] = Teuchos::rcp(new IP);
  _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_epsilon)))*tau);
  _ips["CoupledRobust"]->addTerm(sqrt(_epsilon)*v->grad());
  _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_epsilon)*one/Function::h(),one)*v);
  _ips["CoupledRobust"]->addTerm(v->dt() + beta*v->grad());
  if (spaceDim > 1)
    _ips["CoupledRobust"]->addTerm(tau->div() - v->dt() - beta*v->grad());
  else
    _ips["CoupledRobust"]->addTerm(tau->dx() - v->dt() - beta*v->grad());

  _ips["FixedCoupledRobust"] = Teuchos::rcp(new IP);
  _ips["FixedCoupledRobust"]->addTerm(tau);
  _ips["FixedCoupledRobust"]->addTerm(sqrt(_epsilon)*v->grad());
  _ips["FixedCoupledRobust"]->addTerm(sqrt(_epsilon)*v);
  _ips["FixedCoupledRobust"]->addTerm(v->dt() + beta*v->grad());
  if (spaceDim > 1)
    _ips["FixedCoupledRobust"]->addTerm(tau->div() - v->dt() - beta*v->grad());
  else
    _ips["FixedCoupledRobust"]->addTerm(tau->dx() - v->dt() - beta*v->grad());

  _ips["NSDecoupledH1"] = Teuchos::rcp(new IP);
  _ips["NSDecoupledH1"]->addTerm(one/Function::h()*tau);
  _ips["NSDecoupledH1"]->addTerm(v->grad());
  _ips["NSDecoupledH1"]->addTerm(beta*v->grad()+v->dt());
  if (spaceDim > 1)
    _ips["NSDecoupledH1"]->addTerm(tau->div());
  else
    _ips["NSDecoupledH1"]->addTerm(tau->dx());
  _ips["NSDecoupledH1"]->addTerm(v);

  _ips["NSDecoupledMin"] = Teuchos::rcp(new IP);
  _ips["NSDecoupledMin"]->addTerm(Function::min(one/Function::h(), one/Function::constant(sqrt(_epsilon)))*tau);
  _ips["NSDecoupledMin"]->addTerm(v->grad());
  _ips["NSDecoupledMin"]->addTerm(beta*v->grad()+v->dt());
  if (spaceDim > 1)
    _ips["NSDecoupledMin"]->addTerm(tau->div());
  else
    _ips["NSDecoupledMin"]->addTerm(tau->dx());
  _ips["NSDecoupledMin"]->addTerm(v);

  _rhs = RHS::rhs();
}

VarFactoryPtr SpaceTimeConvectionDiffusionFormulation::vf()
{
  return _vf;
}

BFPtr SpaceTimeConvectionDiffusionFormulation::bf()
{
  return _bf;
}

IPPtr SpaceTimeConvectionDiffusionFormulation::ip(string normName)
{
  return _ips.at(normName);
}

void SpaceTimeConvectionDiffusionFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k, string norm,
    FunctionPtr forcingFunction)
{
  this->initializeSolution(meshTopo, fieldPolyOrder, delta_k, norm, forcingFunction, "");
}

void SpaceTimeConvectionDiffusionFormulation::initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k, string norm,
    FunctionPtr forcingFunction)
{
  this->initializeSolution(Teuchos::null, fieldPolyOrder, delta_k, norm, forcingFunction, filePrefix);
}

void SpaceTimeConvectionDiffusionFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k, string norm,
    FunctionPtr forcingFunction, string savedSolutionAndMeshPrefix)
{
  TEUCHOS_TEST_FOR_EXCEPTION(meshTopo->getDimension() != _spaceDim + 1, std::invalid_argument, "MeshTopo must be space-time mesh");

  BCPtr bc = BC::bc();

  vector<int> H1Order(2);
  H1Order[0] = fieldPolyOrder + 1;
  H1Order[1] = fieldPolyOrder + 1; // for now, use same poly. degree for temporal bases...
  MeshPtr mesh;
  if (savedSolutionAndMeshPrefix == "")
  {
    mesh = Teuchos::rcp( new Mesh(meshTopo, _bf, H1Order, delta_k) ) ;
    _solution = TSolution<double>::solution(mesh,bc);
  }
  else
  {
    mesh = MeshFactory::loadFromHDF5(_bf, savedSolutionAndMeshPrefix+".mesh");
    _solution = TSolution<double>::solution(mesh, bc);
    _solution->loadFromHDF5(savedSolutionAndMeshPrefix+".soln");
  }

  IPPtr ip = _ips.at(norm);
  // RHSPtr rhs = this->rhs(forcingFunction); // in transient case, this will refer to _previousSolution
  if (forcingFunction != Teuchos::null)
    _rhs->addTerm(forcingFunction*this->v());

  _solution->setRHS(_rhs);
  _solution->setIP(ip);

  mesh->registerSolution(_solution); // will project both time steps during refinements...

  LinearTermPtr residual = _rhs->linearTerm() - _bf->testFunctional(_solution,false); // false: don't exclude boundary terms

  double energyThreshold = 0.2;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold ) );
}

double SpaceTimeConvectionDiffusionFormulation::epsilon()
{
  return _epsilon;
}

TFunctionPtr<double> SpaceTimeConvectionDiffusionFormulation::beta()
{
  return _beta;
}

RefinementStrategyPtr SpaceTimeConvectionDiffusionFormulation::getRefinementStrategy()
{
  return _refinementStrategy;
}

void SpaceTimeConvectionDiffusionFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void SpaceTimeConvectionDiffusionFormulation::refine()
{
  _refinementStrategy->refine();
}

// RHSPtr SpaceTimeConvectionDiffusionFormulation::rhs(TFunctionPtr<double> f)
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

VarPtr SpaceTimeConvectionDiffusionFormulation::sigma(int i)
{
  if (i > _spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i)
  {
  case 1:
    return _vf->fieldVar(s_sigma1);
  case 2:
    return _vf->fieldVar(s_sigma2);
  case 3:
    return _vf->fieldVar(s_sigma3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr SpaceTimeConvectionDiffusionFormulation::u()
{
  return _vf->fieldVar(s_u);
}

// traces:
VarPtr SpaceTimeConvectionDiffusionFormulation::tc()
{
  return _vf->fluxVar(s_tc);
}

VarPtr SpaceTimeConvectionDiffusionFormulation::uhat()
{
  return _vf->traceVarSpaceOnly(s_uhat);
}

// test variables:
VarPtr SpaceTimeConvectionDiffusionFormulation::tau()
{
  return _vf->testVar(s_tau, HDIV);
}

// ! Saves the solution(s) and mesh to an HDF5 format.
void SpaceTimeConvectionDiffusionFormulation::save(std::string prefixString)
{
  _solution->mesh()->saveToHDF5(prefixString+".mesh");
  _solution->saveToHDF5(prefixString+".soln");
}

// ! Returns the solution
TSolutionPtr<double> SpaceTimeConvectionDiffusionFormulation::solution()
{
  return _solution;
}

// ! Solves
void SpaceTimeConvectionDiffusionFormulation::solve()
{
  _solution->solve();
}

VarPtr SpaceTimeConvectionDiffusionFormulation::v()
{
  return _vf->testVar(s_v, HGRAD);
}

FunctionPtr SpaceTimeConvectionDiffusionFormulation::forcingFunction(FunctionPtr u_exact)
{
  FunctionPtr f = u_exact->dt() + _beta*u_exact->grad() - _epsilon*u_exact->grad()->div();
  return f;
}
