//
//  NavierStokesVGPFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/5/15.
//
//

#include "NavierStokesVGPFormulation.h"
#include "PenaltyConstraints.h"
#include "PreviousSolutionFunction.h"

const string NavierStokesVGPFormulation::S_U1 = "u_1";
const string NavierStokesVGPFormulation::S_U2 = "u_2";
const string NavierStokesVGPFormulation::S_U3 = "u_3";
const string NavierStokesVGPFormulation::S_P = "p";
const string NavierStokesVGPFormulation::S_SIGMA1 = "\\sigma_{1}";
const string NavierStokesVGPFormulation::S_SIGMA2 = "\\sigma_{2}";
const string NavierStokesVGPFormulation::S_SIGMA3 = "\\sigma_{3}";

const string NavierStokesVGPFormulation::S_U1_HAT = "\\widehat{u}_1";
const string NavierStokesVGPFormulation::S_U2_HAT = "\\widehat{u}_2";
const string NavierStokesVGPFormulation::S_U3_HAT = "\\widehat{u}_3";
const string NavierStokesVGPFormulation::S_TN1_HAT = "\\widehat{t}_{1n}";
const string NavierStokesVGPFormulation::S_TN2_HAT = "\\widehat{t}_{2n}";
const string NavierStokesVGPFormulation::S_TN3_HAT = "\\widehat{t}_{3n}";

const string NavierStokesVGPFormulation::S_V1 = "v_1";
const string NavierStokesVGPFormulation::S_V2 = "v_2";
const string NavierStokesVGPFormulation::S_V3 = "v_3";
const string NavierStokesVGPFormulation::S_TAU1 = "\\tau_{1}";
const string NavierStokesVGPFormulation::S_TAU2 = "\\tau_{2}";
const string NavierStokesVGPFormulation::S_TAU3 = "\\tau_{3}";
const string NavierStokesVGPFormulation::S_Q = "q";

NavierStokesVGPFormulation::NavierStokesVGPFormulation(MeshTopologyPtr meshTopology, double Re,
                                                       int fieldPolyOrder, int delta_k,
                                                       FunctionPtr forcingFunction,
                                                       bool transientFormulation, bool useConformingTraces) {
  if (transientFormulation) {
    cout << "WARNING: transientFormulation is not yet implemented.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "transientFormulation not yet supported in NavierStokesVGPFormulation");
  }
  _spaceDim = meshTopology->getSpaceDim();
  _useConformingTraces = useConformingTraces;
  _mu = 1.0 / Re;
  
  _neglectFluxesOnRHS = true;
  
  if ((_spaceDim != 2) && (_spaceDim != 3)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim must be 2 or 3");
  }
  
  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr u1, u2, u3;
  VarPtr p;
  VarPtr sigma1, sigma2, sigma3;
  
  // traces
  VarPtr u1_hat, u2_hat, u3_hat;
  VarPtr t1n, t2n, t3n;
  
  // tests
  VarPtr v1, v2, v3;
  VarPtr tau1, tau2, tau3;
  VarPtr q;
  
  VarFactory vf;
  u1 = vf.fieldVar(S_U1);
  u2 = vf.fieldVar(S_U2);
  if (_spaceDim==3) u3 = vf.fieldVar(S_U3);
  
  p = vf.fieldVar(S_P);
  
  sigma1 = vf.fieldVar(S_SIGMA1, VECTOR_L2);
  sigma2 = vf.fieldVar(S_SIGMA2, VECTOR_L2);
  if (_spaceDim==3) {
    sigma3 = vf.fieldVar(S_SIGMA3, VECTOR_L2);
  }
  
  Space uHatSpace = useConformingTraces ? HGRAD : L2;
  
  u1_hat = vf.traceVar(S_U1_HAT, 1.0 * u1, uHatSpace);
  u2_hat = vf.traceVar(S_U2_HAT, 1.0 * u2, uHatSpace);
  if (_spaceDim==3) u3_hat = vf.traceVar(S_U3_HAT, 1.0 * u3, uHatSpace);
  
  FunctionPtr n = Function::normal();
  LinearTermPtr t1n_lt, t2n_lt, t3n_lt;
  t1n_lt = p * n->x() - sigma1 * n;
  t2n_lt = p * n->y() - sigma2 * n;
  if (_spaceDim==3) {
    t3n_lt = p * n->z() - sigma3 * n;
  }
  t1n = vf.fluxVar(S_TN1_HAT, t1n_lt);
  t2n = vf.fluxVar(S_TN2_HAT, t2n_lt);
  if (_spaceDim==3) t3n = vf.fluxVar(S_TN3_HAT, t3n_lt);
  
  v1 = vf.testVar(S_V1, HGRAD);
  v2 = vf.testVar(S_V2, HGRAD);
  if (_spaceDim==3) v3 = vf.testVar(S_V3, HGRAD);
  
  tau1 = vf.testVar(S_TAU1, HDIV);
  tau2 = vf.testVar(S_TAU2, HDIV);
  if (_spaceDim==3) {
    tau3 = vf.testVar(S_TAU3, HDIV);
  }
  
  q = vf.testVar(S_Q, HGRAD);
  
  _navierStokesBF = Teuchos::rcp( new BF(vf) );
  int H1Order = fieldPolyOrder + 1;
  MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopology, _navierStokesBF, H1Order, delta_k) ) ;
  
  _backgroundFlow = Solution::solution(mesh);
  _solnIncrement = Solution::solution(mesh);
  
  mesh->registerSolution(_backgroundFlow); // will project background flow during refinements...
  mesh->registerSolution(_solnIncrement);
  
  // ******* First, set Stokes part of BF ******
  // v1
  // tau1 terms:
  _navierStokesBF->addTerm(u1, tau1->div());
  _navierStokesBF->addTerm((1.0/_mu) * sigma1, tau1); // (Re * sigma1 , tau1)
  _navierStokesBF->addTerm(-u1_hat, tau1->dot_normal());
  
  // tau2 terms:
  _navierStokesBF->addTerm(u2, tau2->div());
  _navierStokesBF->addTerm((1.0/_mu) * sigma2, tau2);
  _navierStokesBF->addTerm(-u2_hat, tau2->dot_normal());
  
  // tau3:
  if (_spaceDim==3) {
    _navierStokesBF->addTerm(u3, tau3->div());
    _navierStokesBF->addTerm((1.0/_mu) * sigma3, tau3);
    _navierStokesBF->addTerm(-u3_hat, tau3->dot_normal());
  }
  
  // v1:
  _navierStokesBF->addTerm(sigma1, v1->grad()); // ( sigma1, grad v1)
  _navierStokesBF->addTerm( - p, v1->dx() );
  _navierStokesBF->addTerm( t1n, v1);
  
  // v2:
  _navierStokesBF->addTerm(sigma2, v2->grad()); // ( sigma2, grad v2)
  _navierStokesBF->addTerm( - p, v2->dy());
  _navierStokesBF->addTerm( t2n, v2);
  
  // v3:
  if (_spaceDim==3) {
    _navierStokesBF->addTerm(sigma3, v3->grad()); // ( sigma3, grad v3)
    _navierStokesBF->addTerm( - p, v3->dz());
    _navierStokesBF->addTerm( t3n, v3);
  }
  
  // q:
  _navierStokesBF->addTerm(-u1,q->dx()); // (-u, grad q)
  _navierStokesBF->addTerm(-u2,q->dy());
  if (_spaceDim==3) _navierStokesBF->addTerm(-u3, q->dz());
  
  if (_spaceDim==2) {
    _navierStokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y(), q);
  } else if (_spaceDim==3) {
    _navierStokesBF->addTerm(u1_hat * n->x() + u2_hat * n->y() + u3_hat * n->z(), q);
  }
  
  // copy the BF thus far into a new BF for Stokes
  _stokesBF = Teuchos::rcp( new BF(*_navierStokesBF) );
  
  // to avoid circular references, all previous solution references in BF won't own the memory:
  SolutionPtr backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false );
  
  // convective terms:
  FunctionPtr sigma1_prev = Function::solution(sigma1, backgroundFlowWeakReference);
  FunctionPtr sigma2_prev = Function::solution(sigma2, backgroundFlowWeakReference);
  FunctionPtr u1_prev = Function::solution(u1,backgroundFlowWeakReference);
  FunctionPtr u2_prev = Function::solution(u2,backgroundFlowWeakReference);
  FunctionPtr p_prev = Function::solution(p, backgroundFlowWeakReference);
  FunctionPtr u3_prev, sigma3_prev;
  if (_spaceDim == 3) {
    u3_prev = Function::solution(u3,backgroundFlowWeakReference);
    sigma3_prev = Function::solution(sigma3,backgroundFlowWeakReference);
  }
  
  if (_spaceDim == 2) {
    _navierStokesBF->addTerm( Re * sigma1_prev->x() * u1 + Re * sigma1_prev->y() * u2, v1);
    _navierStokesBF->addTerm( Re * sigma2_prev->x() * u1 + Re * sigma2_prev->y() * u2, v2);
    
    _navierStokesBF->addTerm( Re * u1_prev * sigma1->x() + Re * u2_prev * sigma1->y(), v1);
    _navierStokesBF->addTerm( Re * u1_prev * sigma2->x() + Re * u2_prev * sigma2->y(), v2);
  } else {
    
    _navierStokesBF->addTerm( Re * sigma1_prev->x() * u1 + Re * sigma1_prev->y() * u2 + Re * sigma1_prev->z() * u3, v1);
    _navierStokesBF->addTerm( Re * sigma2_prev->x() * u1 + Re * sigma2_prev->y() * u2 + Re * sigma2_prev->z() * u3, v2);
    _navierStokesBF->addTerm( Re * sigma3_prev->x() * u1 + Re * sigma3_prev->y() * u2 + Re * sigma3_prev->z() * u3, v3);
    
    _navierStokesBF->addTerm( Re * u1_prev * sigma1->x() + Re * u2_prev * sigma1->y() + Re * u3_prev * sigma1->z(), v1);
    _navierStokesBF->addTerm( Re * u1_prev * sigma2->x() + Re * u2_prev * sigma2->y() + Re * u3_prev * sigma2->z(), v2);
    _navierStokesBF->addTerm( Re * u1_prev * sigma3->x() + Re * u2_prev * sigma3->y() + Re * u3_prev * sigma3->z(), v3);
  }
  
  // set the inner product to the graph norm:
  setIP( _navierStokesBF->graphNorm() );
  
  // set the RHS:
  if (forcingFunction == Teuchos::null) {
    int vectorRank = 1;
    forcingFunction = Function::zero(vectorRank);
  }
  
  _rhsForSolve = this->rhs(forcingFunction, _neglectFluxesOnRHS);
  _rhsForResidual = this->rhs(forcingFunction, false);
  _solnIncrement->setRHS(_rhsForSolve);
  
  BCPtr bcSolnIncrement = BC::bc();
  
  _solnIncrement->setBC(bcSolnIncrement);
  
  // define tractions (used in outflow conditions)
  // definition of traction: _mu * ( (\nabla u) + (\nabla u)^T ) n - p n
  //                      = (sigma + sigma^T) n - p n
  if (_spaceDim == 2) {
    _t1 = n->x() * (2 * sigma1->x() - p)       + n->y() * (sigma1->x() + sigma2->x());
    _t2 = n->x() * (sigma1->y() + sigma2->x()) + n->y() * (2 * sigma2->y() - p);
  } else {
    _t1 = n->x() * (2 * sigma1->x() - p)       + n->y() * (sigma1->x() + sigma2->x()) + n->z() * (sigma1->z() + sigma3->x());
    _t2 = n->x() * (sigma1->y() + sigma2->x()) + n->y() * (2 * sigma2->y() - p)       + n->z() * (sigma2->z() + sigma3->y());
    _t3 = n->x() * (sigma1->z() + sigma3->x()) + n->y() * (sigma2->z() + sigma3->y()) + n->z() * (2 * sigma3->z() - p);
  }
  
  double energyThreshold = 0.20;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy(_solnIncrement, energyThreshold) );
  
  double maxDouble = std::numeric_limits<double>::max();
  double maxP = 20;
  _hRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, 0, 0, false ) );
  _pRefinementStrategy = Teuchos::rcp( new RefinementStrategy( _solnIncrement, energyThreshold, maxDouble, maxP, true ) );
  
  // Set up Functions for L^2 norm computations
  FunctionPtr sigma1_incr = Function::solution(sigma1, _solnIncrement);
  FunctionPtr sigma2_incr = Function::solution(sigma2, _solnIncrement);
  FunctionPtr u1_incr = Function::solution(u1,_solnIncrement);
  FunctionPtr u2_incr = Function::solution(u2,_solnIncrement);
  FunctionPtr u3_incr, sigma3_incr;
  FunctionPtr p_incr = Function::solution(p, _solnIncrement);
  if (_spaceDim == 3) {
    u3_incr = Function::solution(u3,_solnIncrement);
    sigma3_incr = Function::solution(sigma3,_solnIncrement);
  }
  
  if (_spaceDim == 2) {
    _L2IncrementFunction = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr
    + sigma1_incr * sigma1_incr + sigma2_incr * sigma2_incr;
    _L2SolutionFunction = u1_prev * u1_prev + u2_prev * u2_prev + p_prev * p_prev
    + sigma1_prev * sigma1_prev + sigma2_prev * sigma2_prev;
  } else {
    _L2IncrementFunction = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr
    + sigma1_incr * sigma1_incr + sigma2_incr * sigma2_incr + sigma3_incr * sigma3_incr;
    _L2SolutionFunction  = u1_prev * u1_prev + u2_prev * u2_prev + p_prev * p_prev
    + sigma1_prev * sigma1_prev + sigma2_prev * sigma2_prev + sigma3_prev * sigma3_prev;
  }
  
  _solver = Solver::getDirectSolver();
  
  _nonlinearIterationCount = 0;
  
  if (_spaceDim==2) {
    // finally, set up a stream function solve for 2D
    _streamFormulation = Teuchos::rcp( new PoissonFormulation(_spaceDim,_useConformingTraces) );
    
    MeshTopologyPtr streamMeshTopo = meshTopology->deepCopy();
    
    MeshPtr streamMesh = Teuchos::rcp( new Mesh(streamMeshTopo, _streamFormulation->bf(), H1Order, delta_k) ) ;
    mesh->registerObserver(streamMesh); // refine streamMesh whenever mesh is refined
    
    LinearTermPtr u1_dy = (1.0 / _mu) * this->sigma(1)->y();
    LinearTermPtr u2_dx = (1.0 / _mu) * this->sigma(2)->x();
    
    FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(_backgroundFlow, u2_dx - u1_dy) );
    RHSPtr streamRHS = RHS::rhs();
    VarPtr q_stream = _streamFormulation->q();
    streamRHS->addTerm( -vorticity * q_stream );
    bool dontWarnAboutOverriding = true;
    ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);
    
    /* Stream function phi is such that
     *    d/dx phi = -u2
     *    d/dy phi =  u1
     * Therefore, psi = grad phi = (-u2, u1), and psi * n = u1 n2 - u2 n1
     */
    
    FunctionPtr u1_soln = Teuchos::rcp( new PreviousSolutionFunction(_backgroundFlow, this->u(1) ) );
    FunctionPtr u2_soln = Teuchos::rcp( new PreviousSolutionFunction(_backgroundFlow, this->u(2) ) );
    ((PreviousSolutionFunction*) u1_soln.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);
    ((PreviousSolutionFunction*) u2_soln.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);
    
    FunctionPtr n = Function::normal();
    
    BCPtr streamBC = BC::bc();
    VarPtr phi = _streamFormulation->phi();
    streamBC->addZeroMeanConstraint(phi);
    
    VarPtr psi_n = _streamFormulation->psi_n_hat();
    streamBC->addDirichlet(psi_n, SpatialFilter::allSpace(), u1_soln * n->y() - u2_soln * n->x());
    
    IPPtr streamIP = _streamFormulation->bf()->graphNorm();
    _streamSolution = Solution::solution(streamMesh,streamBC,streamRHS,streamIP);
  }
}

void NavierStokesVGPFormulation::addInflowCondition(SpatialFilterPtr inflowRegion, FunctionPtr u) {
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getSpaceDim();

  VarPtr u1_hat = this->u_hat(1), u2_hat = this->u_hat(2);
  VarPtr u3_hat;
  if (spaceDim==3) u3_hat = this->u_hat(3);
  
  FunctionPtr u_incr;
  if (_neglectFluxesOnRHS) {
    // this also governs how we accumulate in the fluxes and traces, and hence whether we should use zero BCs or the true BCs for solution increment
    u_incr = u;
  } else {
    // we assume that _neglectFluxesOnRHS = true, in that we always use the full BCs, not their zero-imposing counterparts, when solving for solution increment
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "_neglectFluxesOnRHS = true assumed various places");
  }
  
  _solnIncrement->bc()->addDirichlet(u1_hat, inflowRegion, u_incr->x());
  _solnIncrement->bc()->addDirichlet(u2_hat, inflowRegion, u_incr->y());
  if (spaceDim==3) _solnIncrement->bc()->addDirichlet(u3_hat, inflowRegion, u_incr->z());
}

void NavierStokesVGPFormulation::addOutflowCondition(SpatialFilterPtr outflowRegion) {
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getSpaceDim();

  // my favorite way to do outflow conditions is via penalty constraints imposing a zero traction
  Teuchos::RCP<LocalStiffnessMatrixFilter> filter_incr = _solnIncrement->filter();
  
  Teuchos::RCP< PenaltyConstraints > pcRCP;
  PenaltyConstraints* pc;
  
  if (filter_incr.get() != NULL) {
    pc = dynamic_cast<PenaltyConstraints*>(filter_incr.get());
    if (pc == NULL) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't add PenaltyConstraints when a non-PenaltyConstraints LocalStiffnessMatrixFilter already in place");
    }
  } else {
    pcRCP = Teuchos::rcp( new PenaltyConstraints );
    pc = pcRCP.get();
  }
  FunctionPtr zero = Function::zero();
  pc->addConstraint(_t1==zero, outflowRegion);
  pc->addConstraint(_t2==zero, outflowRegion);
  if (spaceDim==3) pc->addConstraint(_t3==zero, outflowRegion);
  
  if (pcRCP != Teuchos::null) { // i.e., we're not just adding to a prior PenaltyConstraints object
    _solnIncrement->setFilter(pcRCP);
  }
}

void NavierStokesVGPFormulation::addPointPressureCondition() {
  VarPtr p = this->p();

  _solnIncrement->bc()->addSinglePointBC(p->ID(), 0.0);

  if (_solnIncrement->bc()->imposeZeroMeanConstraint(p->ID())) {
    _solnIncrement->bc()->removeZeroMeanConstraint(p->ID());
  }
}

void NavierStokesVGPFormulation::addWallCondition(SpatialFilterPtr wall) {
  int spaceDim = _solnIncrement->mesh()->getTopology()->getSpaceDim();
  vector<double> zero(spaceDim, 0.0);
  addInflowCondition(wall, Function::constant(zero));
}

void NavierStokesVGPFormulation::addZeroMeanPressureCondition() {
  VarPtr p = this->p();
  
  _solnIncrement->bc()->addZeroMeanConstraint(p);
  
  if (_solnIncrement->bc()->singlePointBC(p->ID())) {
    _solnIncrement->bc()->removeSinglePointBC(p->ID());
  }
}

BFPtr NavierStokesVGPFormulation::bf() {
  return _navierStokesBF;
}

Teuchos::RCP<ExactSolution> NavierStokesVGPFormulation::exactSolution(FunctionPtr u_exact, FunctionPtr p_exact) {
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getSpaceDim();
  
  // f1 and f2 are those for Stokes, but minus u \cdot \grad u
  FunctionPtr mu = Function::constant(_mu);
  FunctionPtr u1_exact = u_exact->x();
  FunctionPtr u2_exact = u_exact->y();
  FunctionPtr u3_exact = u_exact->z();
  
  FunctionPtr f = NavierStokesVGPFormulation::forcingFunction(spaceDim, 1.0 / _mu, u_exact, p_exact);
  
  VarPtr p = this->p();
  VarPtr u1_hat = this->u_hat(1);
  VarPtr u2_hat = this->u_hat(2);
  VarPtr u3_hat;
  if (spaceDim==3) u3_hat = this->u_hat(3);
  
  VarPtr u1 = this->u(1);
  VarPtr u2 = this->u(2);
  VarPtr u3;
  if (spaceDim==3) u3 = this->u(3);
  
  VarPtr sigma1 = this->sigma(1);
  VarPtr sigma2 = this->sigma(2);
  VarPtr sigma3;
  if (spaceDim==3) sigma3 = this->sigma(3);
  
  BCPtr bc = BC::bc();
  bc->addSinglePointBC(p->ID(), 0.0);
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  bc->addDirichlet(u1_hat, boundary, u1_exact);
  bc->addDirichlet(u2_hat, boundary, u2_exact);
  if (spaceDim == 3) bc->addDirichlet(u3_hat, boundary, u3_exact);
  
  RHSPtr rhs = this->rhs(f,false);
  Teuchos::RCP<ExactSolution> mySolution = Teuchos::rcp( new ExactSolution(_navierStokesBF, bc, rhs) );
  mySolution->setSolutionFunction(u1, u1_exact);
  mySolution->setSolutionFunction(u2, u2_exact);
  
  mySolution->setSolutionFunction(p, p_exact);
  
  double sigma_weight = _mu;
  
  FunctionPtr sigma1_exact = sigma_weight * u1_exact->grad(spaceDim);
  FunctionPtr sigma2_exact = sigma_weight * u2_exact->grad(spaceDim);
  FunctionPtr sigma3_exact;
  if (spaceDim==3) sigma3_exact = sigma_weight * u3_exact->grad(spaceDim);
  
  mySolution->setSolutionFunction(sigma1, sigma1_exact);
  mySolution->setSolutionFunction(sigma2, sigma2_exact);
  if (spaceDim==3)   mySolution->setSolutionFunction(sigma3, sigma3_exact);
  FunctionPtr sideParity = Function::sideParity();
  FunctionPtr n = Function::normal();
  FunctionPtr t1n_exact, t2n_exact, t3n_exact;
  
  t1n_exact = p_exact * n->x() - sigma1_exact * n;
  t2n_exact = p_exact * n->y() - sigma2_exact * n;
  if (spaceDim==3) {
    t3n_exact = p_exact * n->z() - sigma3_exact * n;
  }
  
  VarPtr t1n = this->tn_hat(1);
  VarPtr t2n = this->tn_hat(2);
  VarPtr t3n;
  if (spaceDim==3) t3n = this->tn_hat(3);
  
  mySolution->setSolutionFunction(u1_hat, u1_exact);
  mySolution->setSolutionFunction(u2_hat, u2_exact);
  mySolution->setSolutionFunction(t1n, t1n_exact * sideParity);
  mySolution->setSolutionFunction(t2n, t2n_exact * sideParity);
  
  if (spaceDim==3) {
    mySolution->setSolutionFunction(u3, u3_exact);
    mySolution->setSolutionFunction(t3n, t3n_exact);
  }
  
  return mySolution;
}

FunctionPtr NavierStokesVGPFormulation::forcingFunction(int spaceDim, double Re, FunctionPtr u_exact, FunctionPtr p_exact) {
  // f1 and f2 are those for Stokes, but minus u \cdot \grad u
  double mu = 1.0 / Re;
  FunctionPtr u1_exact = u_exact->x();
  FunctionPtr u2_exact = u_exact->y();
  FunctionPtr u3_exact = u_exact->z();

  FunctionPtr f;
  
  if (spaceDim == 2) {
    FunctionPtr f1, f2;
    f1 = p_exact->dx() - mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy()) + u1_exact * u1_exact->dx() + u2_exact * u1_exact->dy();
    f2 = p_exact->dy() - mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy()) + u1_exact * u2_exact->dx() + u2_exact * u2_exact->dy();
    f = Function::vectorize(f1, f2);
  } else {
    FunctionPtr f1, f2, f3;
    f1 = p_exact->dx() - mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy() + u1_exact->dz()->dz()) + u1_exact * u1_exact->dx() + u2_exact * u1_exact->dy() + u3_exact * u1_exact->dz();
    f2 = p_exact->dy() - mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy() + u2_exact->dz()->dz()) + u1_exact * u2_exact->dx() + u2_exact * u2_exact->dy() + u3_exact * u2_exact->dz();
    f3 = p_exact->dz() - mu * (u3_exact->dx()->dx() + u3_exact->dy()->dy() + u3_exact->dz()->dz()) + u1_exact * u3_exact->dx() + u2_exact * u3_exact->dy() + u3_exact * u3_exact->dz();
    f = Function::vectorize(f1, f2, f3);
  }
  return f;
}

double NavierStokesVGPFormulation::L2NormSolution() {
  double l2_squared = _L2SolutionFunction->integrate(_backgroundFlow->mesh());
  return sqrt(l2_squared);
}

double NavierStokesVGPFormulation::L2NormSolutionIncrement() {
  double l2_squared = _L2IncrementFunction->integrate(_solnIncrement->mesh());
  return sqrt(l2_squared);
}

int NavierStokesVGPFormulation::nonlinearIterationCount() {
  return _nonlinearIterationCount;
}

VarPtr NavierStokesVGPFormulation::p() {
  VarFactory vf = _navierStokesBF->varFactory();
  return vf.fieldVar(S_P);
}

void NavierStokesVGPFormulation::setIP(IPPtr ip) {
  _solnIncrement->setIP(ip);
}

RefinementStrategyPtr NavierStokesVGPFormulation::getRefinementStrategy() {
  return _refinementStrategy;
}

void NavierStokesVGPFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy) {
  _refinementStrategy = refStrategy;
}

void NavierStokesVGPFormulation::refine(RefinementStrategyPtr refStrategy) {
  _nonlinearIterationCount = 0;
  _solnIncrement->setRHS(_rhsForResidual);
  refStrategy->refine();
  _solnIncrement->setRHS(_rhsForSolve);
}

void NavierStokesVGPFormulation::refine() {
  _nonlinearIterationCount = 0;
  _solnIncrement->setRHS(_rhsForResidual);
  _refinementStrategy->refine();
  _solnIncrement->setRHS(_rhsForSolve);
}

void NavierStokesVGPFormulation::hRefine() {
  _hRefinementStrategy->refine();
}

void NavierStokesVGPFormulation::pRefine() {
  _pRefinementStrategy->refine();
}

RHSPtr NavierStokesVGPFormulation::rhs(FunctionPtr f, bool excludeFluxesAndTraces) {
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getSpaceDim();

  // set the RHS:
  RHSPtr rhs = RHS::rhs();
  
  VarPtr v1 = this->v(1);
  VarPtr v2 = this->v(2);
  VarPtr v3;
  if (spaceDim==3) v3 = this->v(3);
  
  if (f != Teuchos::null) {
    rhs->addTerm( f->x() * v1 );
    rhs->addTerm( f->y() * v2 );
    if (spaceDim == 3) rhs->addTerm( f->z() * v3 );
  }
  
  SolutionPtr backgroundFlowWeakReference = Teuchos::rcp(_backgroundFlow.get(), false);
  // subtract the stokesBF from the RHS:
  rhs->addTerm( -_stokesBF->testFunctional(backgroundFlowWeakReference, excludeFluxesAndTraces) );
  
  VarPtr u1 = this->u(1);
  VarPtr u2 = this->u(2);
  VarPtr u3;
  if (spaceDim==3) u3 = this->u(3);
  
  VarPtr sigma1 = this->sigma(1);
  VarPtr sigma2 = this->sigma(2);
  VarPtr sigma3;
  if (spaceDim==3) sigma3 = this->sigma(3);
  FunctionPtr sigma1_prev = Function::solution(sigma1, backgroundFlowWeakReference);
  FunctionPtr sigma2_prev = Function::solution(sigma2, backgroundFlowWeakReference);
  FunctionPtr u1_prev = Function::solution(u1,backgroundFlowWeakReference);
  FunctionPtr u2_prev = Function::solution(u2,backgroundFlowWeakReference);
  FunctionPtr u3_prev, sigma3_prev;
  if (spaceDim == 3) {
    u3_prev = Function::solution(u3,backgroundFlowWeakReference);
    sigma3_prev = Function::solution(sigma3,backgroundFlowWeakReference);
  }

  // finally, add the u sigma term:
  if (spaceDim == 2) {
    rhs->addTerm( -((u1_prev / _mu) * sigma1_prev->x() + (u2_prev / _mu) * sigma1_prev->y()) * v1 );
    rhs->addTerm( -((u1_prev / _mu) * sigma2_prev->x() + (u2_prev / _mu) * sigma2_prev->y()) * v2 );
  } else {
    rhs->addTerm( -((u1_prev / _mu) * sigma1_prev->x() + (u2_prev / _mu) * sigma1_prev->y() + (u3_prev / _mu) * sigma1_prev->z()) * v1 );
    rhs->addTerm( -((u1_prev / _mu) * sigma2_prev->x() + (u2_prev / _mu) * sigma2_prev->y() + (u3_prev / _mu) * sigma2_prev->z()) * v2 );
    rhs->addTerm( -((u1_prev / _mu) * sigma3_prev->x() + (u2_prev / _mu) * sigma3_prev->y() + (u3_prev / _mu) * sigma3_prev->z()) * v3 );
  }
  return rhs;
}

SolutionPtr NavierStokesVGPFormulation::solution() {
  return _backgroundFlow;
}

SolutionPtr NavierStokesVGPFormulation::solutionIncrement() {
  return _solnIncrement;
}

void NavierStokesVGPFormulation::solveAndAccumulate(double weight) {
  _solnIncrement->solve(_solver);
  bool allowEmptyCells = false;
  _backgroundFlow->addSolution(_solnIncrement, weight, allowEmptyCells, _neglectFluxesOnRHS);
  _nonlinearIterationCount++;
}

VarPtr NavierStokesVGPFormulation::streamPhi() {
  if (_spaceDim == 2) {
    if (_streamFormulation == Teuchos::null) {
      cout << "ERROR: streamPhi() called before initializeSolution called.  Returning null.\n";
      return Teuchos::null;
    }
    return _streamFormulation->phi();
  } else {
    cout << "ERROR: stream function is only supported on 2D solutions.  Returning null.\n";
    return Teuchos::null;
  }
}

SolutionPtr NavierStokesVGPFormulation::streamSolution() {
  if (_spaceDim == 2) {
    if (_streamFormulation == Teuchos::null) {
      cout << "ERROR: streamPhi() called before initializeSolution called.  Returning null.\n";
      return Teuchos::null;
    }
    return _streamSolution;
  } else {
    cout << "ERROR: stream function is only supported on 2D solutions.  Returning null.\n";
    return Teuchos::null;
  }
}

void NavierStokesVGPFormulation::setTimeStep(double dt) {
  _dt->setValue(dt);
}

VarPtr NavierStokesVGPFormulation::sigma(int i) {
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getSpaceDim();
  if (i > spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to spaceDim");
  }
  VarFactory vf = _navierStokesBF->varFactory();
  switch (i) {
    case 1:
      return vf.fieldVar(S_SIGMA1);
    case 2:
      return vf.fieldVar(S_SIGMA2);
    case 3:
      return vf.fieldVar(S_SIGMA3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

BFPtr NavierStokesVGPFormulation::stokesBF() {
  return _stokesBF;
}

VarPtr NavierStokesVGPFormulation::u(int i) {
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getSpaceDim();
  if (i > spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to spaceDim");
  }
  VarFactory vf = _navierStokesBF->varFactory();
  switch (i) {
    case 1:
      return vf.fieldVar(S_U1);
    case 2:
      return vf.fieldVar(S_U2);
    case 3:
      return vf.fieldVar(S_U3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

// traces:
VarPtr NavierStokesVGPFormulation::tn_hat(int i) {
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getSpaceDim();
  if (i > spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to spaceDim");
  }
  VarFactory vf = _navierStokesBF->varFactory();
  switch (i) {
    case 1:
      return vf.fluxVar(S_TN1_HAT);
    case 2:
      return vf.fluxVar(S_TN2_HAT);
    case 3:
      return vf.fluxVar(S_TN3_HAT);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr NavierStokesVGPFormulation::u_hat(int i) {
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getSpaceDim();
  if (i > spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to spaceDim");
  }
  VarFactory vf = _navierStokesBF->varFactory();
  switch (i) {
    case 1:
      return vf.traceVar(S_U1_HAT);
    case 2:
      return vf.traceVar(S_U2_HAT);
    case 3:
      return vf.traceVar(S_U3_HAT);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

// test variables:
VarPtr NavierStokesVGPFormulation::tau(int i) {
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getSpaceDim();
  if (i > spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to spaceDim");
  }
  VarFactory vf = _navierStokesBF->varFactory();
  switch (i) {
    case 1:
      return vf.testVar(S_TAU1, HDIV);
    case 2:
      return vf.testVar(S_TAU2, HDIV);
    case 3:
      return vf.testVar(S_TAU3, HDIV);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr NavierStokesVGPFormulation::v(int i) {
  int spaceDim = _backgroundFlow->mesh()->getTopology()->getSpaceDim();
  if (i > spaceDim) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to spaceDim");
  }
  VarFactory vf = _navierStokesBF->varFactory();
  switch (i) {
    case 1:
      return vf.testVar(S_V1, HGRAD);
    case 2:
      return vf.testVar(S_V2, HGRAD);
    case 3:
      return vf.testVar(S_V3, HGRAD);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}