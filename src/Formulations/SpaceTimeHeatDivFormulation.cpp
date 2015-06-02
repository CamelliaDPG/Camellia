//
//  SpaceTimeHeatDivFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#include "SpaceTimeHeatDivFormulation.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PoissonFormulation.h"
#include "PreviousSolutionFunction.h"

using namespace Camellia;

const string SpaceTimeHeatDivFormulation::s_u = "u";
const string SpaceTimeHeatDivFormulation::s_sigma = "sigma";

const string SpaceTimeHeatDivFormulation::s_uhat = "uhat";
const string SpaceTimeHeatDivFormulation::s_tc = "tc";

const string SpaceTimeHeatDivFormulation::s_v = "v";
const string SpaceTimeHeatDivFormulation::s_tau = "tau";

SpaceTimeHeatDivFormulation::SpaceTimeHeatDivFormulation(int spaceDim, double epsilon, bool useConformingTraces)
{
  _spaceDim = spaceDim;
  _epsilon = epsilon;
  _useConformingTraces = useConformingTraces;

  if ((spaceDim != 1) && (spaceDim != 2) && (spaceDim != 3))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim must be 1, 2, or 3");
  }

  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr u;
  VarPtr sigma;

  // traces
  VarPtr uhat;
  VarPtr tc;

  // tests
  VarPtr v;
  VarPtr tau;

  _vf = VarFactory::varFactory();
  u = _vf->fieldVar(s_u);

  if (spaceDim > 1)
    sigma = _vf->fieldVar(s_sigma, VECTOR_L2);
  else
    sigma = _vf->fieldVar(s_sigma, L2);

  Space uHatSpace = useConformingTraces ? HGRAD : L2;

  uhat = _vf->traceVarSpaceOnly(s_uhat, 1.0 * u, uHatSpace);

  TFunctionPtr<double> n_x = TFunction<double>::normal(); // spatial normal
  TFunctionPtr<double> n_x_parity = n_x * TFunction<double>::sideParity();
  TFunctionPtr<double> n_xt = TFunction<double>::normalSpaceTime();
  TFunctionPtr<double> n_xt_parity = n_xt * TFunction<double>::sideParity();

  // LinearTermPtr sigma_n_lt;
  // if (spaceDim == 1)
  // {
  //   sigma_n_lt = sigma1 * n_x_parity->x();
  // }
  // else if (spaceDim == 2)
  // {
  //   sigma_n_lt = sigma1 * n_x_parity->x() + sigma2 * n_x_parity->y();
  // }
  // else if (spaceDim == 3)
  // {
  //   sigma_n_lt = sigma1 * n_x_parity->x() + sigma2 * n_x_parity->y() + sigma3 * n_x_parity->z();
  // }
  // sigma_n_hat = _vf->fluxVarSpaceOnly(s_tc, sigma_n_lt);
  LinearTermPtr tc_lt;
  if (spaceDim > 1)
    tc_lt = -sigma*n_x_parity + u*n_xt->t();
  else
    tc_lt = -sigma*n_x_parity->x() + u*n_xt->t();
  tc = _vf->fluxVar(s_tc, tc_lt);

  v = _vf->testVar(s_v, HGRAD);

  if (_spaceDim > 1)
    tau = _vf->testVar(s_tau, HDIV); // vector
  else
    tau = _vf->testVar(s_tau, HGRAD); // scalar

  _bf = Teuchos::rcp( new BF(_vf) );
  // v terms
  _bf->addTerm(-u, v->dt());
  if (spaceDim > 1)
    _bf->addTerm(sigma, v->grad());
  else
    _bf->addTerm(sigma, v->dx());
  // if (_spaceDim > 1) _bf->addTerm(sigma2, v->dy());
  // if (_spaceDim > 2) _bf->addTerm(sigma3, v->dz());
  _bf->addTerm(tc, v);

  // tau terms
  // if (_spaceDim > 1)
  // {
    _bf->addTerm((1.0 / _epsilon) * sigma, tau);
    if (spaceDim > 1)
    {
      _bf->addTerm(u, tau->div());
      _bf->addTerm(-uhat, tau * n_x);
    }
    else
    {
      _bf->addTerm(u, tau->dx());
      _bf->addTerm(-uhat, tau * n_x->x());
    }
  // }
  // else
  // {
  //   _bf->addTerm((1.0 / _epsilon) * sigma, tau);
  //   _bf->addTerm(u, tau->dx());
  //   _bf->addTerm(-uhat, tau * n_x->x());
  // }

  _ips["Graph"] = _bf->graphNorm();
}

VarFactoryPtr SpaceTimeHeatDivFormulation::vf()
{
  return _vf;
}

BFPtr SpaceTimeHeatDivFormulation::bf()
{
  return _bf;
}

IPPtr SpaceTimeHeatDivFormulation::ip(string normName)
{
  return _ips.at(normName);
}

// TFunctionPtr<double> SpaceTimeHeatDivFormulation::forcingFunction(int spaceDim, double epsilon, TFunctionPtr<double> u_exact)
// {
//   TFunctionPtr<double> f = u_exact->dt() - epsilon * u_exact->dx()->dx();
//   if (spaceDim > 1) f = f - epsilon * u_exact->dy()->dy();
//   if (spaceDim > 2) f = f - epsilon * u_exact->dz()->dz();
//
//   return f;
// }
//
// void SpaceTimeHeatDivFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
//     TFunctionPtr<double> forcingFunction)
// {
//   this->initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction, "");
// }
//
// void SpaceTimeHeatDivFormulation::initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k,
//     TFunctionPtr<double> forcingFunction)
// {
//   this->initializeSolution(Teuchos::null, fieldPolyOrder, delta_k, forcingFunction, filePrefix);
// }

// void SpaceTimeHeatDivFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
//     TFunctionPtr<double> forcingFunction, string savedSolutionAndMeshPrefix)
// {
//   TEUCHOS_TEST_FOR_EXCEPTION(meshTopo->getSpaceDim() != _spaceDim + 1, std::invalid_argument, "MeshTopo must be space-time mesh");
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
//   RHSPtr rhs = this->rhs(forcingFunction); // in transient case, this will refer to _previousSolution
//   IPPtr ip = _bf->graphNorm();
//
//   _solution->setRHS(rhs);
//   _solution->setIP(ip);
//
//   mesh->registerSolution(_solution); // will project both time steps during refinements...
//
//   LinearTermPtr residual = rhs->linearTerm() - _bf->testFunctional(_solution,false); // false: don't exclude boundary terms
//
//   double energyThreshold = 0.2;
//   _refinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold ) );
// }

double SpaceTimeHeatDivFormulation::epsilon()
{
  return _epsilon;
}

RefinementStrategyPtr SpaceTimeHeatDivFormulation::getRefinementStrategy()
{
  return _refinementStrategy;
}

void SpaceTimeHeatDivFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void SpaceTimeHeatDivFormulation::refine()
{
  _refinementStrategy->refine();
}

RHSPtr SpaceTimeHeatDivFormulation::rhs(TFunctionPtr<double> f)
{
  RHSPtr rhs = RHS::rhs();

  VarPtr v = this->v();

  if (f != Teuchos::null)
  {
    rhs->addTerm( f * v );
  }

  return rhs;
}

VarPtr SpaceTimeHeatDivFormulation::sigma()
{
  return _vf->fieldVar(s_sigma);
}

VarPtr SpaceTimeHeatDivFormulation::u()
{
  return _vf->fieldVar(s_u);
}

// traces:
VarPtr SpaceTimeHeatDivFormulation::tc()
{
  return _vf->fluxVarSpaceOnly(s_tc);
}

VarPtr SpaceTimeHeatDivFormulation::uhat()
{
  return _vf->traceVar(s_uhat);
}

// test variables:
VarPtr SpaceTimeHeatDivFormulation::tau()
{
  return _vf->testVar(s_tau, HDIV);
}

// ! Saves the solution(s) and mesh to an HDF5 format.
void SpaceTimeHeatDivFormulation::save(std::string prefixString)
{
  _solution->mesh()->saveToHDF5(prefixString+".mesh");
  _solution->saveToHDF5(prefixString+".soln");
}

// ! Returns the solution
TSolutionPtr<double> SpaceTimeHeatDivFormulation::solution()
{
  return _solution;
}

// ! Solves
void SpaceTimeHeatDivFormulation::solve()
{
  _solution->solve();
}

VarPtr SpaceTimeHeatDivFormulation::v()
{
  return _vf->testVar(s_v, HGRAD);
}
