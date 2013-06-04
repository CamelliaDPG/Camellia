//
//  NavierStokesFormulation.h
//  Camellia
//
//  Created by Nathan Roberts on 4/2/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_NavierStokesFormulation_h
#define Camellia_NavierStokesFormulation_h

#include "RieszRep.h"

#include "StokesFormulation.h"

// implementation of some standard Navier-Stokes Formulations.
class NavierStokesFormulation {
protected:
  FunctionPtr _Re;
  SolutionPtr _soln;
public:
  NavierStokesFormulation(double Reynolds, SolutionPtr soln) {
    _Re = Function::constant(Reynolds);
    _soln = soln;
  }
  NavierStokesFormulation(FunctionPtr Reynolds, SolutionPtr soln) {
    _Re = Reynolds;
    _soln = soln;
  }
  
  virtual BFPtr bf() = 0;
  virtual RHSPtr rhs(FunctionPtr f1, FunctionPtr f2, bool excludeFluxesAndTraces) = 0;
  // so far, only have support for BCs defined on the entire boundary (i.e. no outflow type conditions)
  virtual BCPtr bc(FunctionPtr u1, FunctionPtr u2, SpatialFilterPtr entireBoundary) = 0;
  virtual IPPtr graphNorm() = 0;
  virtual void primaryTrialIDs(vector<int> &fieldIDs) = 0; // used for best approximation error TeX output (u1,u2) or (u1,u2,p)
  virtual void trialIDs(vector<int> &fieldIDs, vector<int> &correspondingTraceIDs, vector<string> &fileFriendlyNames) = 0; // corr. ID == -1 if there isn't one
  virtual Teuchos::RCP<ExactSolution> exactSolution(FunctionPtr u1, FunctionPtr u2, FunctionPtr p, 
                                                    SpatialFilterPtr entireBoundary) = 0;
  
  FunctionPtr Re() {
    return _Re;
  }
  
  // the classical Kovasznay solution
  static void setKovasznay(double Re, Teuchos::RCP<Mesh> mesh,
                           FunctionPtr &u1_exact, FunctionPtr &u2_exact, FunctionPtr &p_exact) {
    const double PI  = 3.141592653589793238462;
    double lambda = Re / 2 - sqrt ( (Re / 2) * (Re / 2) + (2 * PI) * (2 * PI) );
    
    FunctionPtr exp_lambda_x = Teuchos::rcp( new Exp_ax( lambda ) );
    FunctionPtr exp_2lambda_x = Teuchos::rcp( new Exp_ax( 2 * lambda ) );
    FunctionPtr sin_2pi_y = Teuchos::rcp( new Sin_ay( 2 * PI ) );
    FunctionPtr cos_2pi_y = Teuchos::rcp( new Cos_ay( 2 * PI ) );
    
    u1_exact = Function::constant(1.0) - exp_lambda_x * cos_2pi_y;
    u2_exact = (lambda / (2 * PI)) * exp_lambda_x * sin_2pi_y;
    
    FunctionPtr one = Function::constant(1.0);
    double meshMeasure = one->integrate(mesh);
    
    p_exact = 0.5 * exp_2lambda_x;
    // adjust p to have zero average:
    int cubatureEnrichment = 10;
    double pMeasure = p_exact->integrate(mesh, cubatureEnrichment);
    p_exact = p_exact - Function::constant(pMeasure / meshMeasure);
  }
};

class VGPNavierStokesFormulation : public NavierStokesFormulation {
  VarFactory varFactory;
  // fields:
  VarPtr u1, u2, p, sigma11, sigma12, sigma21, sigma22;
  // fluxes & traces:
  VarPtr u1hat, u2hat, t1n, t2n;
  // tests:
  VarPtr tau1, tau2, q, v1, v2;
  BFPtr _bf, _stokesBF;
  IPPtr _graphNorm;
  FunctionPtr _mu;
  
  // previous solution Functions:
  FunctionPtr sigma11_prev;
  FunctionPtr sigma12_prev;
  FunctionPtr sigma21_prev;
  FunctionPtr sigma22_prev;
  FunctionPtr u1_prev;
  FunctionPtr u2_prev;
  
  void initVars() {
    varFactory = VGPStokesFormulation::vgpVarFactory();
    v1 = varFactory.testVar(VGP_V1_S, HGRAD);
    v2 = varFactory.testVar(VGP_V2_S, HGRAD);
    tau1 = varFactory.testVar(VGP_TAU1_S, HDIV);
    tau2 = varFactory.testVar(VGP_TAU2_S, HDIV);
    q = varFactory.testVar(VGP_Q_S, HGRAD);
    
    u1hat = varFactory.traceVar(VGP_U1HAT_S);
    u2hat = varFactory.traceVar(VGP_U2HAT_S);
    
    t1n = varFactory.fluxVar(VGP_T1HAT_S);
    t2n = varFactory.fluxVar(VGP_T2HAT_S);
    u1 = varFactory.fieldVar(VGP_U1_S);
    u2 = varFactory.fieldVar(VGP_U2_S);
    sigma11 = varFactory.fieldVar(VGP_SIGMA11_S);
    sigma12 = varFactory.fieldVar(VGP_SIGMA12_S);
    sigma21 = varFactory.fieldVar(VGP_SIGMA21_S);
    sigma22 = varFactory.fieldVar(VGP_SIGMA22_S);
    p = varFactory.fieldVar(VGP_P_S);
    
    sigma11_prev = Function::solution(sigma11, _soln);
    sigma12_prev = Function::solution(sigma12, _soln);
    sigma21_prev = Function::solution(sigma21, _soln);
    sigma22_prev = Function::solution(sigma22, _soln);
    u1_prev = Function::solution(u1,_soln);
    u2_prev = Function::solution(u2,_soln);
  }
  
  void init(FunctionPtr Re, SolutionPtr soln) {
    _mu = 1.0 / Re;
    
    initVars();
    
    _stokesBF = stokesBF(_mu);
    
    bool dontEnrichVelocity = false;
    _bf = stokesBF(_mu);

//    FunctionPtr u_prev = Function::vectorize(u1_prev, u2_prev);
//    
//    _bf->addTerm( u1, u_prev * v1->grad() );
//    _bf->addTerm( u1_prev * u1, v1->dx());
//    _bf->addTerm( u1_prev * u2, v1->dy());
//    
//    _bf->addTerm( u2, u_prev * v2->grad() );
//    _bf->addTerm( u2_prev * u1, v2->dx());
//    _bf->addTerm( u2_prev * u2, v2->dy());
    
    _bf->addTerm(- Re * sigma11_prev * u1 - Re * sigma12_prev * u2 - Re * u1_prev * sigma11 - Re * u2_prev * sigma12, v1);
    _bf->addTerm(- Re * sigma21_prev * u1 - Re * sigma22_prev * u2 - Re * u1_prev * sigma21 - Re * u2_prev * sigma22, v2);
//    _bf->addTerm( - ( u1_prev->dx() + u2_prev->dy() ) * u1, v1); // (div u) delta u v
//    _bf->addTerm( - ( u1_prev->dx() + u2_prev->dy() ) * u2, v2); // (div u) delta u v
    
    _graphNorm = _bf->graphNorm(); // just use the automatic for now
    
    // EXPERIMENT! :
    // when _mu is small, we lose control of the gradient of v, which we need control of for the
    // equivalence to the optimal test norm.  So here we add it back in:
//    _graphNorm->addTerm(v1->grad());
//    _graphNorm->addTerm(v2->grad());
  }
public:
  static BFPtr stokesBF(FunctionPtr mu) {
    VGPStokesFormulation stokesFormulation(mu);
    return stokesFormulation.bf();
  }
  
  VGPNavierStokesFormulation(double Re, SolutionPtr soln) : NavierStokesFormulation(Re, soln) {
    init(Function::constant(Re), soln);
  }
  VGPNavierStokesFormulation(FunctionPtr Re, SolutionPtr soln) : NavierStokesFormulation(Re, soln) {
    init(Re,soln);
  }
  
  BFPtr bf() {
    return _bf;
  }
  IPPtr graphNorm() {
    return _graphNorm;
  }
  RHSPtr rhs(FunctionPtr f1, FunctionPtr f2, bool excludeFluxesAndTraces) {
    Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
    rhs->addTerm( f1 * v1 + f2 * v2 );
    // add the subtraction of the stokes BF here:
    rhs->addTerm( -_stokesBF->testFunctional(_soln, excludeFluxesAndTraces) );
    
//    // finally, add convective term:
//    FunctionPtr u_prev = Function::vectorize(u1_prev,u2_prev);
//    rhs->addTerm( - u1_prev * u_prev * v1->grad() );
//    rhs->addTerm( - u2_prev * u_prev * v2->grad() );
    
    // finally, add the u sigma term:
    rhs->addTerm( ((u1_prev / _mu) * sigma11_prev + (u2_prev / _mu) * sigma12_prev) * v1 );
    rhs->addTerm( ((u1_prev / _mu) * sigma21_prev + (u2_prev / _mu) * sigma22_prev) * v2 );
//    rhs->addTerm( (u1_prev->dx() + u2_prev->dy()) * (u1_prev * v1 + u2_prev * v2) ); // (div u) u * v
    
    return rhs;
  }
  IPPtr scaleCompliantGraphNorm(FunctionPtr dt_inv = Function::zero()) {
    // corresponds to ||u||^2 + ||grad u||^2 + ||p||^2
    FunctionPtr h = Teuchos::rcp( new hFunction() );
    IPPtr compliantGraphNorm = Teuchos::rcp( new IP );
    
    compliantGraphNorm->addTerm( _mu * v1->dx() + tau1->x() - (u1_prev / _mu) * v1 ); // sigma11
    compliantGraphNorm->addTerm( _mu * v1->dy() + tau1->y() - (u2_prev / _mu) * v1 ); // sigma12
    compliantGraphNorm->addTerm( _mu * v2->dx() + tau2->x() - (u1_prev / _mu) * v2 ); // sigma21
    compliantGraphNorm->addTerm( _mu * v2->dy() + tau2->y() - (u2_prev / _mu) * v2); // sigma22
    
    compliantGraphNorm->addTerm(   h * tau1->div() - h * q->dx() - h * dt_inv * v1
                                 - (sigma11_prev / _mu) * v1 - (sigma21_prev / _mu) * v2 );  // u1
    compliantGraphNorm->addTerm(   h * tau2->div() - h * q->dy() - h * dt_inv * v2
                                 - (sigma12_prev / _mu) * v1 - (sigma22_prev / _mu) * v2 );  // u2
    
    compliantGraphNorm->addTerm( v1->dx() + v2->dy() );          // pressure
    
    compliantGraphNorm->addTerm( (1 / h) * v1 );
    compliantGraphNorm->addTerm( (1 / h) * v2 );
    compliantGraphNorm->addTerm( q );
    compliantGraphNorm->addTerm( tau1 );
    compliantGraphNorm->addTerm( tau2 );
    return compliantGraphNorm;
  }
  
  BCPtr bc(FunctionPtr u1_fxn, FunctionPtr u2_fxn, SpatialFilterPtr entireBoundary) {
    Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
    bc->addDirichlet(u1hat, entireBoundary, u1_fxn);
    bc->addDirichlet(u2hat, entireBoundary, u2_fxn);
    bc->addZeroMeanConstraint(p);
    return bc;
  }
  Teuchos::RCP<ExactSolution> exactSolution(FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact,
                                            SpatialFilterPtr entireBoundary) {
    // f1 and f2 are those for Stokes, but minus u \cdot \grad u
    FunctionPtr mu = 1.0 / _Re;
    FunctionPtr f1 = -p_exact->dx() + mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy())
                     - u1_exact * u1_exact->dx() - u2_exact * u1_exact->dy();
    FunctionPtr f2 = -p_exact->dy() + mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy())
                     - u1_exact * u2_exact->dx() - u2_exact * u2_exact->dy();
    
    BCPtr bc = this->bc(u1_exact, u2_exact, entireBoundary);
    
    RHSPtr rhs = this->rhs(f1,f2,false);
    Teuchos::RCP<ExactSolution> mySolution = Teuchos::rcp( new ExactSolution(_bf, bc, rhs) );
    mySolution->setSolutionFunction(u1, u1_exact);
    mySolution->setSolutionFunction(u2, u2_exact);
    
    mySolution->setSolutionFunction(p, p_exact);
    
    FunctionPtr sigma_weight = _mu;
    
    FunctionPtr sigma11_exact = sigma_weight * u1_exact->dx();
    FunctionPtr sigma12_exact = sigma_weight * u1_exact->dy();
    FunctionPtr sigma21_exact = sigma_weight * u2_exact->dx();
    FunctionPtr sigma22_exact = sigma_weight * u2_exact->dy();
    
    mySolution->setSolutionFunction(sigma11, sigma11_exact);
    mySolution->setSolutionFunction(sigma12, sigma12_exact);
    mySolution->setSolutionFunction(sigma21, sigma21_exact);
    mySolution->setSolutionFunction(sigma22, sigma22_exact);
    
    // tn = (mu sigma - pI)n
    FunctionPtr sideParity = Function::sideParity();
    FunctionPtr n = Function::normal();
    FunctionPtr t1n_exact = (sigma11_exact - p_exact) * n->x() + sigma12_exact * n->y();
    FunctionPtr t2n_exact = sigma21_exact * n->x() + (sigma22_exact - p_exact) * n->y();
    
    mySolution->setSolutionFunction(u1hat, u1_exact);
    mySolution->setSolutionFunction(u2hat, u2_exact);
    mySolution->setSolutionFunction(t1n, t1n_exact * sideParity);
    mySolution->setSolutionFunction(t2n, t2n_exact * sideParity);
    
    return mySolution;
  }
  
  void primaryTrialIDs(vector<int> &fieldIDs) {
    // (u1,u2,p) 
    FunctionPtr mu = 1.0 / _Re;
    VGPStokesFormulation stokesFormulation(mu);
    stokesFormulation.primaryTrialIDs(fieldIDs);
  }
  void trialIDs(vector<int> &fieldIDs, vector<int> &correspondingTraceIDs, vector<string> &fileFriendlyNames) {
    FunctionPtr mu = 1.0 / _Re;
    VGPStokesFormulation stokesFormulation(mu);
    stokesFormulation.trialIDs(fieldIDs,correspondingTraceIDs,fileFriendlyNames);
  }
  VarPtr u1var() {
    return u1;
  }
  VarPtr u2var() {
    return u2;
  }
  VarPtr pvar() {
    return p;
  }
};

class VGPNavierStokesProblem {
  SolutionPtr _backgroundFlow, _solnIncrement;
  Teuchos::RCP<Mesh> _mesh;
  Teuchos::RCP<BC> _bc, _bcForIncrement;
  Teuchos::RCP< ExactSolution > _exactSolution;
  Teuchos::RCP<BF> _bf;
  
  Teuchos::RCP< VGPNavierStokesFormulation > _vgpNavierStokesFormulation;
  int _iterations;
  double _iterationWeight;
  
  bool _neglectFluxesOnRHS;
  
  void init(FunctionPtr Re, MeshGeometryPtr geometry, int H1Order, int pToAdd,
            FunctionPtr f1, FunctionPtr f2, bool enrichVelocity) {
    _neglectFluxesOnRHS = true;
    FunctionPtr mu = 1.0 / Re;
    _iterations = 0;
    _iterationWeight = 1.0;
    
    Teuchos::RCP< VGPStokesFormulation > vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
    
    // create a new mesh:
    bool useConformingTraces = true;
    map<int, int> trialOrderEnhancements;
    VarPtr u1 = vgpStokesFormulation->u1var();
    VarPtr u2 = vgpStokesFormulation->u2var();
    if (enrichVelocity) {
      trialOrderEnhancements[u1->ID()] = 1;
      trialOrderEnhancements[u2->ID()] = 1;
    }
    _mesh = Teuchos::rcp( new Mesh(geometry->vertices(), geometry->elementVertices(),
                                   vgpStokesFormulation->bf(), H1Order, pToAdd,
                                   useConformingTraces, trialOrderEnhancements) );
    _mesh->setEdgeToCurveMap(geometry->edgeToCurveMap());
    
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered ); // SpatialFilterUnfiltered returns true everywhere
    
    _backgroundFlow = Teuchos::rcp( new Solution(_mesh) );
    
    _solnIncrement = Teuchos::rcp( new Solution(_mesh) );
    _solnIncrement->setCubatureEnrichmentDegree( H1Order-1 ); // can have weights with poly degree = trial degree
    
    _vgpNavierStokesFormulation = Teuchos::rcp( new VGPNavierStokesFormulation(Re, _backgroundFlow));
    
    _backgroundFlow->setRHS( _vgpNavierStokesFormulation->rhs(f1, f2,_neglectFluxesOnRHS) );
    _backgroundFlow->setIP( _vgpNavierStokesFormulation->graphNorm() );
    
    _mesh->setBilinearForm(_vgpNavierStokesFormulation->bf());
    
    _solnIncrement->setRHS( _vgpNavierStokesFormulation->rhs(f1,f2,_neglectFluxesOnRHS) );
    _solnIncrement->setIP( _vgpNavierStokesFormulation->graphNorm() );
  }
  
  void init(FunctionPtr Re, FieldContainer<double> &quadPoints, int horizontalCells,
            int verticalCells, int H1Order, int pToAdd,
            FunctionPtr u1_0, FunctionPtr u2_0, FunctionPtr f1, FunctionPtr f2, bool useEnrichedVelocity) {
    _neglectFluxesOnRHS = true;
    FunctionPtr mu = 1.0/Re;
    _iterations = 0;
    _iterationWeight = 1.0;
    
    Teuchos::RCP< VGPStokesFormulation > vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
    
    // create a new mesh:
    bool useConformingTraces = true;
    bool triangulate = false;
    map<int, int> trialOrderEnhancements;
    VarPtr u1 = vgpStokesFormulation->u1var();
    VarPtr u2 = vgpStokesFormulation->u2var();
    if (useEnrichedVelocity) {
      trialOrderEnhancements[u1->ID()] = 1;
      trialOrderEnhancements[u2->ID()] = 1;
    }
    _mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd,
                                triangulate, useConformingTraces, trialOrderEnhancements);
    
    SpatialFilterPtr entireBoundary = SpatialFilter::allSpace(); // allSpace() returns true everywhere
    
    BCPtr vgpBC = vgpStokesFormulation->bc(u1_0, u2_0, entireBoundary);
    
    _backgroundFlow = Teuchos::rcp( new Solution(_mesh, vgpBC) );
    
    // since we're disregarding accumulated fluxes, the incremental solutions have the usual BCs enforced:
    _solnIncrement = Teuchos::rcp( new Solution(_mesh, vgpBC) );
    _solnIncrement->setCubatureEnrichmentDegree( H1Order-1 ); // can have weights with poly degree = trial degree
    
    _vgpNavierStokesFormulation = Teuchos::rcp( new VGPNavierStokesFormulation(Re, _backgroundFlow));
    
    _backgroundFlow->setRHS( _vgpNavierStokesFormulation->rhs(f1, f2, _neglectFluxesOnRHS) );
    _backgroundFlow->setIP( _vgpNavierStokesFormulation->graphNorm() );
    
    _bf = _vgpNavierStokesFormulation->bf();
    _mesh->setBilinearForm(_bf);
    
    _solnIncrement->setRHS( _vgpNavierStokesFormulation->rhs(f1,f2, _neglectFluxesOnRHS) );
    _solnIncrement->setIP( _vgpNavierStokesFormulation->graphNorm() );
  }
  void init(FunctionPtr Re, FieldContainer<double> &quadPoints, int horizontalCells,
             int verticalCells, int H1Order, int pToAdd,
             FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact, bool enrichVelocity) {
    _neglectFluxesOnRHS = false; // main reason we don't neglect fluxes is because exact solution isn't yet set up to handle that
    FunctionPtr mu = 1/Re;
    _iterations = 0;
    _iterationWeight = 1.0;
    
    Teuchos::RCP< VGPStokesFormulation > vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
    
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered ); // SpatialFilterUnfiltered returns true everywhere
    
    BCPtr vgpBC = vgpStokesFormulation->bc(u1_exact, u2_exact, entireBoundary);
    
    // create a new mesh:
    bool useConformingTraces = true;
    bool triangulate = false;
    map<int, int> trialOrderEnhancements;
    VarPtr u1 = vgpStokesFormulation->u1var();
    VarPtr u2 = vgpStokesFormulation->u2var();
    if (enrichVelocity) {
      trialOrderEnhancements[u1->ID()] = 1;
      trialOrderEnhancements[u2->ID()] = 1;
    }
    _mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd,
                                triangulate, useConformingTraces, trialOrderEnhancements);
    
    _backgroundFlow = Teuchos::rcp( new Solution(_mesh, vgpBC) );
    
    // the incremental solutions have zero BCs enforced:
    FunctionPtr zero = Function::zero();
    BCPtr zeroBC = vgpStokesFormulation->bc(zero, zero, entireBoundary);
    _solnIncrement = Teuchos::rcp( new Solution(_mesh, zeroBC) );
    _solnIncrement->setCubatureEnrichmentDegree( H1Order-1 ); // can have weights with poly degree = trial degree
    
    _vgpNavierStokesFormulation = Teuchos::rcp( new VGPNavierStokesFormulation(Re, _backgroundFlow));
    
    _exactSolution = _vgpNavierStokesFormulation->exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
    _backgroundFlow->setRHS( _exactSolution->rhs() );
    _backgroundFlow->setIP( _vgpNavierStokesFormulation->graphNorm() );
    
    _mesh->setBilinearForm(_vgpNavierStokesFormulation->bf());
    
    _solnIncrement->setRHS( _exactSolution->rhs() );
    _solnIncrement->setIP( _vgpNavierStokesFormulation->graphNorm() );
  }
public:
  VGPNavierStokesProblem(FunctionPtr Re, MeshGeometryPtr geometry, int H1Order, int pToAdd,
                         FunctionPtr f1 = Function::zero(), FunctionPtr f2=Function::zero(),
                         bool enrichVelocity = false) {
    init(Re,geometry,H1Order,pToAdd, f1,f2, enrichVelocity);
    // note that this constructor leaves BC enforcement up to the user
  }
  
  VGPNavierStokesProblem(FunctionPtr Re, FieldContainer<double> &quadPoints, int horizontalCells,
                         int verticalCells, int H1Order, int pToAdd,
                         FunctionPtr u1_0, FunctionPtr u2_0, FunctionPtr f1, FunctionPtr f2,
                         bool enrichVelocity = false) {
    init(Re,quadPoints,horizontalCells,verticalCells,H1Order,pToAdd,u1_0,u2_0,f1,f2,enrichVelocity);
    // this constructor enforces Dirichlet BCs on the velocity at each iterate, and disregards accumulated trace and flux data
  }
  VGPNavierStokesProblem(double Re, FieldContainer<double> &quadPoints, int horizontalCells,
                         int verticalCells, int H1Order, int pToAdd,
                         FunctionPtr u1_0, FunctionPtr u2_0, FunctionPtr f1, FunctionPtr f2,
                         bool enrichVelocity = false) {
    init(Function::constant(Re),quadPoints,horizontalCells,verticalCells,H1Order,pToAdd,u1_0,u2_0,f1,f2, enrichVelocity);
    // this constructor enforces Dirichlet BCs on the velocity at each iterate, and disregards accumulated trace and flux data
  }
  VGPNavierStokesProblem(FunctionPtr Re, FieldContainer<double> &quadPoints, int horizontalCells,
                         int verticalCells, int H1Order, int pToAdd,
                         FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact, bool enrichVelocity) {
    init(Re,quadPoints,horizontalCells,verticalCells,H1Order,pToAdd,u1_exact,u2_exact,p_exact, enrichVelocity);
    // this constructor enforces Dirichlet BCs on the velocity on first iterate, and zero BCs on later (does *not* disregard accumulated trace and flux data)
  }
            
  VGPNavierStokesProblem(double Re, FieldContainer<double> &quadPoints, int horizontalCells,
                         int verticalCells, int H1Order, int pToAdd,
                         FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact, bool enrichVelocity) {
    init(Function::constant(Re),quadPoints,horizontalCells,verticalCells,H1Order,pToAdd,u1_exact,u2_exact,p_exact,enrichVelocity);
    // this constructor enforces Dirichlet BCs on the velocity on first iterate, and zero BCs on later (does *not* disregard accumulated trace and flux data)
  }
            
  SolutionPtr backgroundFlow() {
    return _backgroundFlow;
  }
  BFPtr bf() {
    return _vgpNavierStokesFormulation->bf();
  }
  Teuchos::RCP<ExactSolution> exactSolution() {
    return _exactSolution;
  }
  SolutionPtr solutionIncrement() {
    return _solnIncrement;
  }
  double lineSearchWeight() {
    double alpha = 2.0;
    double alphaMin = 1e-10;
    LinearTermPtr rhsLT = ((RHSEasy*) _backgroundFlow->rhs().get())->linearTerm();
    RieszRep rieszRep(_backgroundFlow->mesh(), _backgroundFlow->ip(), rhsLT);
    rieszRep.computeRieszRep();
    double costPrevious = rieszRep.getNorm();
    double costNew;
    do {
      alpha /= 2;
      _backgroundFlow->addSolution(_solnIncrement, alpha);
      rieszRep.computeRieszRep();
      costNew = rieszRep.getNorm();
      _backgroundFlow->addSolution(_solnIncrement, -alpha);
    } while ((costNew > costPrevious) && (alpha > alphaMin));
    if (costNew > costPrevious) {
      return 0;
    } else {
      return alpha;
    }
  }
  double iterate(bool useLineSearch) { // returns the weight used...    
    double weight;
    if (_iterations==0) {
      _solnIncrement->clear(); // zero out so we start afresh if the _iterations have been manually set...
      _backgroundFlow->solve();
      // want _solnIncrement to store the initial solution as the first increment:
      weight = 1.0;
      _solnIncrement->addSolution(_backgroundFlow, weight, true); // true: allow adds of empty cells
    } else {
      _solnIncrement->solve();
      if (!useLineSearch) {
        weight = _iterationWeight;
      } else {
        weight = lineSearchWeight();
      }
      if (_neglectFluxesOnRHS) {
        // then let's zero out the fluxes in background flow before we add in _solnIncrement
        // note: this is not the most efficient way to do this (would be faster if we set basisCoefficients
        //       for all fluxes at once, and faster still if we did managed this within the addSolution() call below)
        vector<int> fluxIDs = _bf->trialBoundaryIDs();
        set<int> cellIDs = _mesh->getActiveCellIDs();
        for (set<int>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
          int cellID = *cellIDIt;
          int numSides = _mesh->getElement(cellID)->numSides();
          FieldContainer<double> solnCoeffs;
          for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
            for (int i=0; i<fluxIDs.size(); i++) {
              int fluxID = fluxIDs[i];
              _backgroundFlow->solnCoeffsForCellID(solnCoeffs, cellID, fluxID, sideIndex); // just sizes solnCoeffs (again, not most efficient)
              solnCoeffs.initialize(0);
              _backgroundFlow->setSolnCoeffsForCellID(solnCoeffs, cellID, fluxID, sideIndex);
            }
          }
        }
      }
      _backgroundFlow->addSolution(_solnIncrement, weight);
    }
    _iterations++;
    return weight;
  }
  int iterationCount() {
    return _iterations;
  }
  void setIterationCount(int value) {
    _iterations = value;
  }
  Teuchos::RCP<Mesh> mesh() {
    return _mesh;
  }
  void setBC( Teuchos::RCP<BCEasy> bc ) {
    _backgroundFlow->setBC(bc);
    _solnIncrement->setBC(bc->copyImposingZero());
  }
  void setIP( IPPtr ip ) {
    _backgroundFlow->setIP( ip );
    _solnIncrement->setIP( ip );
  }
  BFPtr stokesBF() {
    FunctionPtr mu =  1.0 / _vgpNavierStokesFormulation->Re();
    return VGPNavierStokesFormulation::stokesBF( mu );
  }
  Teuchos::RCP< VGPNavierStokesFormulation > vgpNavierStokesFormulation() {
    return _vgpNavierStokesFormulation;
  }
};

#endif
