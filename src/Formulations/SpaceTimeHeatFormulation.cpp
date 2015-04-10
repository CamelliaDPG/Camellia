//
//  SpaceTimeHeatFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#include "SpaceTimeHeatFormulation.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PoissonFormulation.h"
#include "PreviousSolutionFunction.h"

using namespace Camellia;

const string SpaceTimeHeatFormulation::S_U = "u";
const string SpaceTimeHeatFormulation::S_SIGMA1 = "\\sigma_{1}";
const string SpaceTimeHeatFormulation::S_SIGMA2 = "\\sigma_{2}";
const string SpaceTimeHeatFormulation::S_SIGMA3 = "\\sigma_{3}";

const string SpaceTimeHeatFormulation::S_U_HAT = "\\widehat{u}";
const string SpaceTimeHeatFormulation::S_SIGMA_N_HAT = "\\widehat{\\sigma}_{n_x}";

const string SpaceTimeHeatFormulation::S_V = "v";
const string SpaceTimeHeatFormulation::S_TAU = "\\tau";

SpaceTimeHeatFormulation::SpaceTimeHeatFormulation(int spaceDim, double epsilon, bool useConformingTraces) {
  _spaceDim = spaceDim;
  _epsilon = epsilon;
  _useConformingTraces = useConformingTraces;

  if ((spaceDim != 1) && (spaceDim != 2) && (spaceDim != 3)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim must be 1, 2, or 3");
  }
  
  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr u;
  VarPtr sigma1, sigma2, sigma3;
  
  // traces
  VarPtr u_hat;
  VarPtr sigma_n_hat;
  
  // tests
  VarPtr v;
  VarPtr tau;
  VarPtr q;
  
  u = _vf.fieldVar(S_U);
  
  sigma1 = _vf.fieldVar(S_SIGMA1);
  if (spaceDim > 1) sigma2 = _vf.fieldVar(S_SIGMA2);
  if (spaceDim==3) {
    sigma3 = _vf.fieldVar(S_SIGMA3);
  }
  
  Space uHatSpace = useConformingTraces ? HGRAD : L2;
  
  u_hat = _vf.traceVar(S_U_HAT, 1.0 * u, uHatSpace);
  
  FunctionPtr n_x = Function::normal(); // spatial normal
  FunctionPtr n_x_parity = n_x * Function::sideParity();
  FunctionPtr n_xt = Function::normalSpaceTime();
  
  LinearTermPtr sigma_n_lt;
  if (spaceDim == 1)
  {
    sigma_n_lt = sigma1 * n_x_parity->x();
  }
  else if (spaceDim == 2)
  {
    sigma_n_lt = sigma1 * n_x_parity->x() + sigma2 * n_x_parity->y();
  }
  else if (spaceDim == 3)
  {
    sigma_n_lt = sigma1 * n_x_parity->x() + sigma2 * n_x_parity->y() + sigma3 * n_x_parity->z();
  }
  sigma_n_hat = _vf.fluxVarSpaceOnly(S_SIGMA_N_HAT, sigma_n_lt);
  
  v = _vf.testVar(S_V, HGRAD);

  if (_spaceDim > 1)
    tau = _vf.testVar(S_TAU, HDIV); // vector
  else
    tau = _vf.testVar(S_TAU, HGRAD); // scalar
  
  _bf = Teuchos::rcp( new BF(_vf) );
  // v terms
  _bf->addTerm(-u, v->dt());
  _bf->addTerm(u_hat, v * n_xt->t());
  _bf->addTerm(sigma1, v->dx());
  if (_spaceDim > 1) _bf->addTerm(sigma2, v->dy());
  if (_spaceDim > 2) _bf->addTerm(sigma3, v->dz());
  _bf->addTerm(-sigma_n_hat, v);
  
  // tau terms
  if (_spaceDim > 1) {
    _bf->addTerm((1.0 / _epsilon) * sigma1, tau->x());
    _bf->addTerm((1.0 / _epsilon) * sigma2, tau->y());
    if (_spaceDim > 2) _bf->addTerm((1.0 / _epsilon) * sigma3, tau->z());
    _bf->addTerm(u, tau->div());
    _bf->addTerm(-u_hat, tau * n_x);
  } else {
    _bf->addTerm((1.0 / _epsilon) * sigma1, tau);
    _bf->addTerm(u, tau->dx());
    _bf->addTerm(-u_hat, tau * n_x->x());
  }
}

BFPtr SpaceTimeHeatFormulation::bf() {
  return _bf;
}

FunctionPtr SpaceTimeHeatFormulation::forcingFunction(int spaceDim, double epsilon, FunctionPtr u_exact) {
  FunctionPtr f = u_exact->dt() - epsilon * u_exact->dx()->dx();
  if (spaceDim > 1) f = f - epsilon * u_exact->dy()->dy();
  if (spaceDim > 2) f = f - epsilon * u_exact->dz()->dz();
  
  return f;
}

void SpaceTimeHeatFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
                                                  FunctionPtr forcingFunction) {
  this->initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction, "");
}

void SpaceTimeHeatFormulation::initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k,
                                              FunctionPtr forcingFunction) {
  this->initializeSolution(Teuchos::null, fieldPolyOrder, delta_k, forcingFunction, filePrefix);
}

void SpaceTimeHeatFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
                                                  FunctionPtr forcingFunction, string savedSolutionAndMeshPrefix) {
  BCPtr bc = BC::bc();
  
  vector<int> H1Order(2);
  H1Order[0] = fieldPolyOrder + 1;
  H1Order[1] = fieldPolyOrder + 1; // for now, use same poly. degree for temporal bases...
  MeshPtr mesh;
  if (savedSolutionAndMeshPrefix == "") {
    mesh = Teuchos::rcp( new Mesh(meshTopo, _bf, H1Order, delta_k) ) ;
    _solution = Solution::solution(mesh,bc);
  } else {
    mesh = MeshFactory::loadFromHDF5(_bf, savedSolutionAndMeshPrefix+".mesh");
    _solution = Solution::solution(mesh, bc);
    _solution->loadFromHDF5(savedSolutionAndMeshPrefix+".soln");
  }
  
  RHSPtr rhs = this->rhs(forcingFunction); // in transient case, this will refer to _previousSolution
  IPPtr ip = _bf->graphNorm();
  
  _solution->setRHS(rhs);
  _solution->setIP(ip);
  
  mesh->registerSolution(_solution); // will project both time steps during refinements...
  
  LinearTermPtr residual = rhs->linearTerm() - _bf->testFunctional(_solution,false); // false: don't exclude boundary terms
  
  double energyThreshold = 0.2;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold ) );
}

double SpaceTimeHeatFormulation::epsilon() {
  return _epsilon;
}

RefinementStrategyPtr SpaceTimeHeatFormulation::getRefinementStrategy() {
  return _refinementStrategy;
}

void SpaceTimeHeatFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy) {
  _refinementStrategy = refStrategy;
}

void SpaceTimeHeatFormulation::refine() {
  _refinementStrategy->refine();
}

RHSPtr SpaceTimeHeatFormulation::rhs(FunctionPtr f) {
  RHSPtr rhs = RHS::rhs();
  
  VarPtr v = this->v();

  if (f != Teuchos::null) {
    rhs->addTerm( f * v );
  }
  
  return rhs;
}

VarPtr SpaceTimeHeatFormulation::sigma(int i) {
  if (i > _spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (i) {
    case 1:
      return _vf.fieldVar(S_SIGMA1);
    case 2:
      return _vf.fieldVar(S_SIGMA2);
    case 3:
      return _vf.fieldVar(S_SIGMA3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr SpaceTimeHeatFormulation::u() {
  return _vf.fieldVar(S_U);
}

// traces:
VarPtr SpaceTimeHeatFormulation::sigma_n_hat() {
  return _vf.fluxVarSpaceOnly(S_SIGMA_N_HAT);
}

VarPtr SpaceTimeHeatFormulation::u_hat() {
  return _vf.traceVar(S_U_HAT);
}

// test variables:
VarPtr SpaceTimeHeatFormulation::tau() {
  return _vf.testVar(S_TAU, HDIV);
}

// ! Saves the solution(s) and mesh to an HDF5 format.
void SpaceTimeHeatFormulation::save(std::string prefixString) {
  _solution->mesh()->saveToHDF5(prefixString+".mesh");
  _solution->saveToHDF5(prefixString+".soln");
}

// ! Returns the solution
SolutionPtr SpaceTimeHeatFormulation::solution() {
  return _solution;
}

// ! Solves
void SpaceTimeHeatFormulation::solve() {
  _solution->solve();
}

VarPtr SpaceTimeHeatFormulation::v() {
  return _vf.testVar(S_V, HGRAD);
}