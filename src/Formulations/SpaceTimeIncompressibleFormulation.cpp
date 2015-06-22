//
//  SpaceTimeIncompressibleFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#include "SpaceTimeIncompressibleFormulation.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PoissonFormulation.h"
#include "PreviousSolutionFunction.h"
#include "IncompressibleProblems.h"
#include <algorithm>

using namespace Camellia;

const string SpaceTimeIncompressibleFormulation::s_u1 = "u1";
const string SpaceTimeIncompressibleFormulation::s_u2 = "u2";
const string SpaceTimeIncompressibleFormulation::s_u3 = "u3";
const string SpaceTimeIncompressibleFormulation::s_sigma11 = "sigma11";
const string SpaceTimeIncompressibleFormulation::s_sigma12 = "sigma12";
const string SpaceTimeIncompressibleFormulation::s_sigma13 = "sigma13";
const string SpaceTimeIncompressibleFormulation::s_sigma21 = "sigma21";
const string SpaceTimeIncompressibleFormulation::s_sigma22 = "sigma22";
const string SpaceTimeIncompressibleFormulation::s_sigma23 = "sigma23";
const string SpaceTimeIncompressibleFormulation::s_sigma31 = "sigma31";
const string SpaceTimeIncompressibleFormulation::s_sigma32 = "sigma32";
const string SpaceTimeIncompressibleFormulation::s_sigma33 = "sigma33";
const string SpaceTimeIncompressibleFormulation::s_p = "p";

const string SpaceTimeIncompressibleFormulation::s_u1hat = "u1hat";
const string SpaceTimeIncompressibleFormulation::s_u2hat = "u2hat";
const string SpaceTimeIncompressibleFormulation::s_u3hat = "u3hat";
const string SpaceTimeIncompressibleFormulation::s_tm1hat = "tm1hat";
const string SpaceTimeIncompressibleFormulation::s_tm2hat = "tm2hat";
const string SpaceTimeIncompressibleFormulation::s_tm3hat = "tm3hat";

const string SpaceTimeIncompressibleFormulation::s_v1 = "v1";
const string SpaceTimeIncompressibleFormulation::s_v2 = "v2";
const string SpaceTimeIncompressibleFormulation::s_v3 = "v3";
const string SpaceTimeIncompressibleFormulation::s_tau1 = "tau1";
const string SpaceTimeIncompressibleFormulation::s_tau2 = "tau2";
const string SpaceTimeIncompressibleFormulation::s_tau3 = "tau3";
const string SpaceTimeIncompressibleFormulation::s_q = "q";

SpaceTimeIncompressibleFormulation::SpaceTimeIncompressibleFormulation(int spaceDim, bool steady, double mu, bool useConformingTraces,
    Teuchos::RCP<IncompressibleProblem> problem, int fieldPolyOrder, int delta_k, int numTElems, string norm,
    string savedSolutionAndMeshPrefix)
{
  _spaceDim = spaceDim;
  _steady = steady;
  _mu = mu;
  _useConformingTraces = useConformingTraces;
  MeshTopologyPtr meshTopo = problem->meshTopology(numTElems);
  MeshGeometryPtr meshGeometry = problem->meshGeometry();

  if (!steady)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(meshTopo->getDimension() != _spaceDim + 1, std::invalid_argument, "MeshTopo must be space-time mesh for transient");
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(meshTopo->getDimension() != _spaceDim, std::invalid_argument, "MeshTopo must be spatial mesh for steady");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(mu==0, std::invalid_argument, "mu may not be 0!");
  TEUCHOS_TEST_FOR_EXCEPTION(spaceDim==1, std::invalid_argument, "Incompressible Navier-Stokes is trivial for spaceDim=1");
  TEUCHOS_TEST_FOR_EXCEPTION((spaceDim != 2) && (spaceDim != 3), std::invalid_argument, "spaceDim must be 2 or 3");


  Space uHatSpace = useConformingTraces ? HGRAD : L2;

  FunctionPtr zero = Function::constant(1);
  FunctionPtr one = Function::constant(1);
  FunctionPtr n_x = TFunction<double>::normal(); // spatial normal
  // FunctionPtr n_x_parity = n_x * TFunction<double>::sideParity();
  FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
  // FunctionPtr n_xt_parity = n_xt * TFunction<double>::sideParity();

  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr u1, u2, u3;
  VarPtr sigma11, sigma12, sigma13;
  VarPtr sigma21, sigma22, sigma23;
  VarPtr sigma31, sigma32, sigma33;
  VarPtr p;

  // traces
  VarPtr u1hat, u2hat, u3hat;
  VarPtr tm1hat, tm2hat, tm3hat;

  // tests
  VarPtr v1, v2, v3;
  VarPtr tau1, tau2, tau3;
  VarPtr q;

  _vf = VarFactory::varFactory();

  if (spaceDim == 1)
  {
    u1 = _vf->fieldVar(s_u1);
    sigma11 = _vf->fieldVar(s_sigma11);
    p = _vf->fieldVar(s_p);
    u1hat = _vf->traceVarSpaceOnly(s_u1hat, 1.0 * u1, uHatSpace);
    tm1hat = _vf->fluxVar(s_tm1hat);
    v1 = _vf->testVar(s_v1, HGRAD);
    tau1 = _vf->testVar(s_tau1, HGRAD); // scalar
    q = _vf->testVar(s_q, HGRAD);
  }
  if (spaceDim == 2)
  {
    u1 = _vf->fieldVar(s_u1);
    u2 = _vf->fieldVar(s_u2);
    sigma11 = _vf->fieldVar(s_sigma11);
    sigma12 = _vf->fieldVar(s_sigma12);
    sigma21 = _vf->fieldVar(s_sigma21);
    sigma22 = _vf->fieldVar(s_sigma22);
    p = _vf->fieldVar(s_p);
    u1hat = _vf->traceVarSpaceOnly(s_u1hat, 1.0 * u1, uHatSpace);
    u2hat = _vf->traceVarSpaceOnly(s_u2hat, 1.0 * u2, uHatSpace);
    tm1hat = _vf->fluxVar(s_tm1hat);
    tm2hat = _vf->fluxVar(s_tm2hat);
    v1 = _vf->testVar(s_v1, HGRAD);
    v2 = _vf->testVar(s_v2, HGRAD);
    tau1 = _vf->testVar(s_tau1, HDIV); // vector
    tau2 = _vf->testVar(s_tau2, HDIV); // vector
    q = _vf->testVar(s_q, HGRAD);
  }
  if (spaceDim == 3)
  {
    u1 = _vf->fieldVar(s_u1);
    u2 = _vf->fieldVar(s_u2);
    u3 = _vf->fieldVar(s_u3);
    sigma11 = _vf->fieldVar(s_sigma11);
    sigma12 = _vf->fieldVar(s_sigma12);
    sigma13 = _vf->fieldVar(s_sigma13);
    sigma21 = _vf->fieldVar(s_sigma21);
    sigma22 = _vf->fieldVar(s_sigma22);
    sigma23 = _vf->fieldVar(s_sigma23);
    sigma31 = _vf->fieldVar(s_sigma31);
    sigma32 = _vf->fieldVar(s_sigma32);
    sigma33 = _vf->fieldVar(s_sigma33);
    p = _vf->fieldVar(s_p);
    u1hat = _vf->traceVarSpaceOnly(s_u1hat, 1.0 * u1, uHatSpace);
    u2hat = _vf->traceVarSpaceOnly(s_u2hat, 1.0 * u2, uHatSpace);
    u3hat = _vf->traceVarSpaceOnly(s_u3hat, 1.0 * u3, uHatSpace);
    tm1hat = _vf->fluxVar(s_tm1hat);
    tm2hat = _vf->fluxVar(s_tm2hat);
    tm3hat = _vf->fluxVar(s_tm3hat);
    v1 = _vf->testVar(s_v1, HGRAD);
    v2 = _vf->testVar(s_v2, HGRAD);
    v3 = _vf->testVar(s_v3, HGRAD);
    tau1 = _vf->testVar(s_tau1, HDIV); // vector
    tau2 = _vf->testVar(s_tau2, HDIV); // vector
    tau3 = _vf->testVar(s_tau3, HDIV); // vector
    q = _vf->testVar(s_q, HGRAD);
  }

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

  _bf = Teuchos::rcp( new BF(_vf) );




  // Define mesh
  BCPtr bc = BC::bc();

  vector<int> H1Order(2);
  H1Order[0] = fieldPolyOrder + 1;
  H1Order[1] = fieldPolyOrder + 1; // for now, use same poly. degree for temporal bases...
  if (savedSolutionAndMeshPrefix == "")
  {
    MeshPtr proxyMesh = Teuchos::rcp( new Mesh(meshTopo->deepCopy(), _bf, H1Order, delta_k) ) ;
    _mesh = Teuchos::rcp( new Mesh(meshTopo, _bf, H1Order, delta_k) ) ;
    if (meshGeometry != Teuchos::null)
      _mesh->setEdgeToCurveMap(meshGeometry->edgeToCurveMap());
    proxyMesh->registerObserver(_mesh);
    problem->preprocessMesh(proxyMesh);

    _solutionUpdate = Solution::solution(_bf, _mesh, bc);
    _solutionBackground = Solution::solution(_bf, _mesh, bc);
    map<int, FunctionPtr> initialGuess;
    initialGuess[u(1)->ID()] = Function::zero();
    initialGuess[u(2)->ID()] = Function::zero();
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

  FunctionPtr u1_prev = Function::solution(u1, _solutionBackground);
  FunctionPtr u2_prev = Function::solution(u2, _solutionBackground);
  FunctionPtr u_prev = Function::vectorize(u1_prev, u2_prev);


  if (spaceDim == 2)
  {
    // stress equation
    _bf->addTerm((1.0 / _mu) * sigma11, tau1->x());
    _bf->addTerm((1.0 / _mu) * sigma12, tau1->y());
    _bf->addTerm((1.0 / _mu) * sigma21, tau2->x());
    _bf->addTerm((1.0 / _mu) * sigma22, tau2->y());
    _bf->addTerm(u1, tau1->div());
    _bf->addTerm(u2, tau2->div());
    _bf->addTerm(-u1hat, tau1 * n_x);
    _bf->addTerm(-u2hat, tau2 * n_x);

    // momentum equation
    if (!steady)
    {
      _bf->addTerm(-u1, v1->dt());
      _bf->addTerm(-u2, v2->dt());
    }
    _bf->addTerm(-u1_prev*u1, v1->dx());
    _bf->addTerm(-u1_prev*u1, v1->dx());
    _bf->addTerm(-u2_prev*u1, v1->dy());
    _bf->addTerm(-u1_prev*u2, v1->dy());
    _bf->addTerm(-u2_prev*u1, v2->dx());
    _bf->addTerm(-u1_prev*u2, v2->dx());
    _bf->addTerm(-u2_prev*u2, v2->dy());
    _bf->addTerm(-u2_prev*u2, v2->dy());

    _bf->addTerm(sigma11, v1->dx());
    _bf->addTerm(sigma12, v1->dy());
    _bf->addTerm(sigma21, v2->dx());
    _bf->addTerm(sigma22, v2->dy());

    _bf->addTerm(-p, v1->dx());
    _bf->addTerm(-p, v2->dy());

    _bf->addTerm(tm1hat, v1);
    _bf->addTerm(tm2hat, v2);

    // continuity equation
    _bf->addTerm(-u1, q->dx());
    _bf->addTerm(-u2, q->dy());

    // _bf->addTerm(u1hat, q->times_normal_x());
    // _bf->addTerm(u2hat, q->times_normal_y());
    _bf->addTerm(u1hat*n_x->x(), q);
    _bf->addTerm(u2hat*n_x->y(), q);
  }

  // Add residual to RHS
  _rhs = RHS::rhs();
  // stress equation
  _rhs->addTerm( -u1_prev * tau1->div() );
  _rhs->addTerm( -u2_prev * tau2->div() );

  // momentum equation
  if (!steady)
  {
    _rhs->addTerm( u1_prev * v1->dt());
    _rhs->addTerm( u2_prev * v2->dt());
  }
  _rhs->addTerm( u1_prev * u1_prev*v1->dx() );
  _rhs->addTerm( u1_prev * u2_prev*v1->dy() );
  _rhs->addTerm( u2_prev * u1_prev*v2->dx() );
  _rhs->addTerm( u2_prev * u2_prev*v2->dy() );

  // continuity equation
  _rhs->addTerm( u1_prev*q->dx());
  _rhs->addTerm( u2_prev*q->dy());

  _ips["Graph"] = _bf->graphNorm();

  _ips["CoupledRobust"] = Teuchos::rcp(new IP);
  // _ips["CoupledRobust"]->addTerm(_beta*v->grad());
  _ips["CoupledRobust"]->addTerm(u_prev*v1->grad());
  _ips["CoupledRobust"]->addTerm(u_prev*v2->grad());
  _ips["CoupledRobust"]->addTerm(u1_prev*v1->dx() + u2_prev*v2->dx());
  _ips["CoupledRobust"]->addTerm(u1_prev*v1->dy() + u2_prev*v2->dy());
  // _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau);
  _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau1);
  _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau2);
  // _ips["CoupledRobust"]->addTerm(sqrt(_mu)*v->grad());
  _ips["CoupledRobust"]->addTerm(sqrt(_mu)*v1->grad());
  _ips["CoupledRobust"]->addTerm(sqrt(_mu)*v2->grad());
  // _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v);
  _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v1);
  _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v2);
  // _ips["CoupledRobust"]->addTerm(tau->div() - v->dt() - beta*v->grad());
  if (!steady)
  {
    _ips["CoupledRobust"]->addTerm(tau1->div() -v1->dt() - u_prev*v1->grad() - u1_prev*v1->dx() - u2_prev*v2->dx());
    _ips["CoupledRobust"]->addTerm(tau2->div() -v2->dt() - u_prev*v2->grad() - u1_prev*v1->dy() - u2_prev*v2->dy());
  }
  else
  {
    _ips["CoupledRobust"]->addTerm(tau1->div() - u_prev*v1->grad() - u1_prev*v1->dx() - u2_prev*v2->dx());
    _ips["CoupledRobust"]->addTerm(tau2->div() - u_prev*v2->grad() - u1_prev*v1->dy() - u2_prev*v2->dy());
  }
  // _ips["CoupledRobust"]->addTerm(v1->dx() + v2->dy());
  _ips["CoupledRobust"]->addTerm(q->grad());
  _ips["CoupledRobust"]->addTerm(q);


  _ips["NSDecoupledH1"] = Teuchos::rcp(new IP);
  // _ips["NSDecoupledH1"]->addTerm(one/Function::h()*tau);
  _ips["NSDecoupledH1"]->addTerm(one/Function::h()*tau1);
  _ips["NSDecoupledH1"]->addTerm(one/Function::h()*tau2);
  // _ips["NSDecoupledH1"]->addTerm(v->grad());
  _ips["NSDecoupledH1"]->addTerm(v1->grad());
  _ips["NSDecoupledH1"]->addTerm(v2->grad());
  // _ips["NSDecoupledH1"]->addTerm(_beta*v->grad()+v->dt());
  if (!steady)
  {
    _ips["NSDecoupledH1"]->addTerm(v1->dt() + u_prev*v1->grad() + u1_prev*v1->dx() + u2_prev*v2->dx());
    _ips["NSDecoupledH1"]->addTerm(v2->dt() + u_prev*v2->grad() + u1_prev*v1->dy() + u2_prev*v2->dy());
  }
  else
  {
    _ips["NSDecoupledH1"]->addTerm(u_prev*v1->grad() + u1_prev*v1->dx() + u2_prev*v2->dx());
    _ips["NSDecoupledH1"]->addTerm(u_prev*v2->grad() + u1_prev*v1->dy() + u2_prev*v2->dy());
  }
  // _ips["NSDecoupledH1"]->addTerm(tau->div());
  _ips["NSDecoupledH1"]->addTerm(tau1->div());
  _ips["NSDecoupledH1"]->addTerm(tau2->div());
  // _ips["NSDecoupledH1"]->addTerm(v);
  _ips["NSDecoupledH1"]->addTerm(v1);
  _ips["NSDecoupledH1"]->addTerm(v2);
  // _ips["CoupledRobust"]->addTerm(v1->dx() + v2->dy());
  _ips["NSDecoupledH1"]->addTerm(q->grad());
  _ips["NSDecoupledH1"]->addTerm(q);

  IPPtr ip = _ips.at(norm);
  if (problem->forcingTerm != Teuchos::null)
    _rhs->addTerm(problem->forcingTerm);

  _solutionUpdate->setRHS(_rhs);
  _solutionUpdate->setIP(ip);

  // impose zero mean constraint
  _solutionUpdate->bc()->imposeZeroMeanConstraint(p->ID());
  // _solutionUpdate->bc()->singlePointBC(p->ID());

  _mesh->registerSolution(_solutionBackground);
  _mesh->registerSolution(_solutionUpdate);

  LinearTermPtr residual = _rhs->linearTerm() - _bf->testFunctional(_solutionUpdate,false); // false: don't exclude boundary terms

  double energyThreshold = 0.2;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy( _mesh, residual, ip, energyThreshold ) );
}

VarFactoryPtr SpaceTimeIncompressibleFormulation::vf()
{
  return _vf;
}

BFPtr SpaceTimeIncompressibleFormulation::bf()
{
  return _bf;
}

IPPtr SpaceTimeIncompressibleFormulation::ip(string normName)
{
  return _ips.at(normName);
}

double SpaceTimeIncompressibleFormulation::mu()
{
  return _mu;
}

RefinementStrategyPtr SpaceTimeIncompressibleFormulation::getRefinementStrategy()
{
  return _refinementStrategy;
}

void SpaceTimeIncompressibleFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void SpaceTimeIncompressibleFormulation::refine()
{
  _refinementStrategy->refine();
}

VarPtr SpaceTimeIncompressibleFormulation::u(int i)
{
  switch (i)
  {
    case 1:
      return _vf->fieldVar(s_u1);
    case 2:
      return _vf->fieldVar(s_u2);
    case 3:
      return _vf->fieldVar(s_u3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr SpaceTimeIncompressibleFormulation::sigma(int i, int j)
{
  if (i > _spaceDim || j > _spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i)
  {
  case 1:
    switch (j)
    {
      case 1:
        return _vf->fieldVar(s_sigma11);
      case 2:
        return _vf->fieldVar(s_sigma12);
      case 3:
        return _vf->fieldVar(s_sigma13);
    }
  case 2:
    switch (j)
    {
      case 1:
        return _vf->fieldVar(s_sigma21);
      case 2:
        return _vf->fieldVar(s_sigma22);
      case 3:
        return _vf->fieldVar(s_sigma23);
    }
  case 3:
    switch (j)
    {
      case 1:
        return _vf->fieldVar(s_sigma31);
      case 2:
        return _vf->fieldVar(s_sigma32);
      case 3:
        return _vf->fieldVar(s_sigma33);
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i, j values");
}

VarPtr SpaceTimeIncompressibleFormulation::p()
{
  return _vf->fieldVar(s_p);
}

// traces:
VarPtr SpaceTimeIncompressibleFormulation::tmhat(int i)
{
  switch (i)
  {
    case 1:
      return _vf->fluxVar(s_tm1hat);
    case 2:
      return _vf->fluxVar(s_tm2hat);
    case 3:
      return _vf->fluxVar(s_tm3hat);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr SpaceTimeIncompressibleFormulation::uhat(int i)
{
  switch (i)
  {
    case 1:
      return _vf->traceVarSpaceOnly(s_u1hat);
    case 2:
      return _vf->traceVarSpaceOnly(s_u2hat);
    case 3:
      return _vf->traceVarSpaceOnly(s_u3hat);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr SpaceTimeIncompressibleFormulation::v(int i)
{
  switch (i)
  {
    case 1:
      return _vf->testVar(s_v1, HGRAD);
    case 2:
      return _vf->testVar(s_v2, HGRAD);
    case 3:
      return _vf->testVar(s_v3, HGRAD);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

// test variables:
VarPtr SpaceTimeIncompressibleFormulation::tau(int i)
{
  switch (i)
  {
    case 1:
      return _vf->testVar(s_tau1, HDIV);
    case 2:
      return _vf->testVar(s_tau2, HDIV);
    case 3:
      return _vf->testVar(s_tau3, HDIV);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr SpaceTimeIncompressibleFormulation::q()
{
  return _vf->testVar(s_q, HGRAD);
}

set<int> SpaceTimeIncompressibleFormulation::nonlinearVars()
{
  set<int> nonlinearVars;//{u(1)->ID(),u(2)->ID()};
  nonlinearVars.insert(u(1)->ID());
  nonlinearVars.insert(u(2)->ID());
  return nonlinearVars;
}

// ! Saves the solution(s) and mesh to an HDF5 format.
void SpaceTimeIncompressibleFormulation::save(std::string prefixString)
{
  _solutionUpdate->mesh()->saveToHDF5(prefixString+".mesh");
  // _solutionUpdate->saveToHDF5(prefixString+"_update.soln");
  _solutionBackground->saveToHDF5(prefixString+".soln");
}

// ! Returns the solution
SolutionPtr SpaceTimeIncompressibleFormulation::solutionUpdate()
{
  return _solutionUpdate;
}

// ! Returns the solution
SolutionPtr SpaceTimeIncompressibleFormulation::solutionBackground()
{
  return _solutionBackground;
}

void SpaceTimeIncompressibleFormulation::updateSolution()
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

// ! Solves
void SpaceTimeIncompressibleFormulation::solve()
{
  _solutionUpdate->solve();
}
