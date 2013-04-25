//
//  OseenFormulations.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/23/13.
//
//

#ifndef Camellia_debug_OseenFormulations_h
#define Camellia_debug_OseenFormulations_h

#include "StokesFormulation.h"

class OseenFormulation {
protected:
  FunctionPtr _Re;
public:
  OseenFormulation(double Reynolds) {
    _Re = Function::constant(Reynolds);
  }
  OseenFormulation(FunctionPtr Reynolds) {
    _Re = Reynolds;
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
};

class VGPOseenFormulation : public OseenFormulation {
  VarFactory varFactory;
  // fields:
  VarPtr u1, u2, p, sigma11, sigma12, sigma21, sigma22;
  // fluxes & traces:
  VarPtr u1hat, u2hat, t1n, t2n;
  // tests:
  VarPtr tau1, tau2, q, v1, v2;
  BFPtr _bf, _stokesBF;
  IPPtr _graphNorm;
  FunctionPtr _mu, _sqrt_mu;
  
  // background flow Functions:
  FunctionPtr _U1;
  FunctionPtr _U2;
  
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
  }
  
  void init(FunctionPtr Re, FunctionPtr sqrtRe, FunctionPtr U1, FunctionPtr U2) {
    _mu = 1.0 / Re;
    _sqrt_mu = 1.0 / sqrtRe;
    
    _U1 = U1;
    _U2 = U2;
    
    initVars();
    
    _stokesBF = stokesBF(_mu);
    
    // construct bilinear form:
    _bf = stokesBF(_mu);
    
    _bf->addTerm(- _U1 * sigma11 - _U2 * sigma12, v1);
    _bf->addTerm(- _U1 * sigma21 - _U2 * sigma22, v2);
    
    _graphNorm = _bf->graphNorm(); // just use the automatic for now
  }
public:
  static BFPtr stokesBF(FunctionPtr mu) {
    VGPStokesFormulation stokesFormulation(mu);
    return stokesFormulation.bf();
  }
  
  VGPOseenFormulation(double Re, FunctionPtr U1, FunctionPtr U2) : OseenFormulation(Re) {
    init(Function::constant(Re), Function::constant(sqrt(Re)),U1,U2);
  }
  VGPOseenFormulation(FunctionPtr Re, FunctionPtr sqrtRe, FunctionPtr U1, FunctionPtr U2) : OseenFormulation(Re) {
    init(Re,sqrtRe,U1,U2);
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
    
    return rhs;
  }
  IPPtr scaleCompliantGraphNorm() {
    // messing around: at any given time this may or may not correspond to "scale compliance", whatever that means
    FunctionPtr h = Teuchos::rcp( new hFunction() );
    IPPtr compliantGraphNorm = Teuchos::rcp( new IP );
    
    FunctionPtr sqrt_mu_inv = 1.0 / _sqrt_mu;
    
    compliantGraphNorm->addTerm( _sqrt_mu * v1->dx() + sqrt_mu_inv * ( tau1->x() - _U1 * v1 ) ); // sigma11
    compliantGraphNorm->addTerm( _sqrt_mu * v1->dy() + sqrt_mu_inv * ( tau1->y() - _U1 * v2 ) ); // sigma12
    compliantGraphNorm->addTerm( _sqrt_mu * v2->dx() + sqrt_mu_inv * ( tau2->x() - _U2 * v1 ) ); // sigma21
    compliantGraphNorm->addTerm( _sqrt_mu * v2->dy() + sqrt_mu_inv * ( tau2->y() - _U2 * v2 ) ); // sigma22
    compliantGraphNorm->addTerm( _sqrt_mu * v1->dx() + _sqrt_mu * v2->dy() );          // pressure
    compliantGraphNorm->addTerm( h * sqrt_mu_inv * ( tau1->div() - q->dx()) );  // u1
    compliantGraphNorm->addTerm( h * sqrt_mu_inv * ( tau2->div() - q->dy()) );  // u2
    
    compliantGraphNorm->addTerm( (_sqrt_mu / h) * v1 );
    compliantGraphNorm->addTerm( (_sqrt_mu / h) * v2 );
    compliantGraphNorm->addTerm( sqrt_mu_inv * q );
    compliantGraphNorm->addTerm( sqrt_mu_inv * tau1 );
    compliantGraphNorm->addTerm( sqrt_mu_inv * tau2 );
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
    
    FunctionPtr sigma11_exact = u1_exact->dx();
    FunctionPtr sigma12_exact = u1_exact->dy();
    FunctionPtr sigma21_exact = u2_exact->dx();
    FunctionPtr sigma22_exact = u2_exact->dy();
    
    mySolution->setSolutionFunction(sigma11, sigma11_exact);
    mySolution->setSolutionFunction(sigma12, sigma12_exact);
    mySolution->setSolutionFunction(sigma21, sigma21_exact);
    mySolution->setSolutionFunction(sigma22, sigma22_exact);
    
    // tn = (mu sigma - pI)n
    FunctionPtr sideParity = Function::sideParity();
    FunctionPtr n = Function::normal();
    FunctionPtr t1n_exact = (mu * sigma11_exact - p_exact) * n->x() + mu * sigma12_exact * n->y();
    FunctionPtr t2n_exact = mu * sigma21_exact * n->x() + (mu * sigma22_exact - p_exact) * n->y();
    
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
};

class VGPOseenProblem {
  SolutionPtr _soln;
  Teuchos::RCP<Mesh> _mesh;
  Teuchos::RCP< ExactSolution > _exactSolution;
  Teuchos::RCP<BF> _bf;
  
  Teuchos::RCP< VGPOseenFormulation > _vgpOseenFormulation;
  
  void init(FunctionPtr sqrtRe, FieldContainer<double> &quadPoints, int horizontalCells,
            int verticalCells, int H1Order, int pToAdd,
            FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact, bool enrichVelocity) {
    FunctionPtr Re = sqrtRe * sqrtRe;
    FunctionPtr mu = 1/Re;
    
    Teuchos::RCP< VGPStokesFormulation > vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu, enrichVelocity) );
    
    // create a new mesh:
    _mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd);
    
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered ); // SpatialFilterUnfiltered returns true everywhere
    
    BCPtr vgpBC = vgpStokesFormulation->bc(u1_exact, u2_exact, entireBoundary);
    
    _mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd);
        
    _soln = Teuchos::rcp( new Solution(_mesh, vgpBC) );
    _soln->setCubatureEnrichmentDegree( H1Order-1 ); // can have weights with poly degree = trial degree
    
    _vgpOseenFormulation = Teuchos::rcp( new VGPOseenFormulation(Re, sqrtRe, u1_exact, u2_exact) );
    
    _exactSolution = _vgpOseenFormulation->exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
    
    _mesh->setBilinearForm(_vgpOseenFormulation->bf());
    _soln->setRHS( _exactSolution->rhs() );
    _soln->setIP( _vgpOseenFormulation->graphNorm() );
  }
public:
  VGPOseenProblem(FunctionPtr Re, FunctionPtr sqrtRe, FieldContainer<double> &quadPoints, int horizontalCells,
                         int verticalCells, int H1Order, int pToAdd,
                         FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact, bool enrichVelocity) {
    init(sqrtRe,quadPoints,horizontalCells,verticalCells,H1Order,pToAdd,u1_exact,u2_exact,p_exact, enrichVelocity);
    // this constructor enforces Dirichlet BCs on the velocity on first iterate, and zero BCs on later (does *not* disregard accumulated trace and flux data)
  }
  VGPOseenProblem(double Re, FieldContainer<double> &quadPoints, int horizontalCells,
                         int verticalCells, int H1Order, int pToAdd,
                         FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact, bool enrichVelocity) {
    init(Function::constant(sqrt(Re)),quadPoints,horizontalCells,verticalCells,H1Order,pToAdd,u1_exact,u2_exact,p_exact,enrichVelocity);
    // this constructor enforces Dirichlet BCs on the velocity on first iterate, and zero BCs on later (does *not* disregard accumulated trace and flux data)
  }
  
  BFPtr bf() {
    return _vgpOseenFormulation->bf();
  }
  Teuchos::RCP<ExactSolution> exactSolution() {
    return _exactSolution;
  }
  SolutionPtr solution() {
    return _soln;
  }
  Teuchos::RCP<Mesh> mesh() {
    return _mesh;
  }
  void setBC( Teuchos::RCP<BCEasy> bc ) {
    _soln->setBC(bc);
  }
  void setIP( IPPtr ip ) {
    _soln->setIP( ip );
  }
  BFPtr stokesBF() {
    FunctionPtr mu =  1.0 / _vgpOseenFormulation->Re();
    return VGPOseenFormulation::stokesBF( mu );
  }
  Teuchos::RCP< VGPOseenFormulation > vgpOseenFormulation() {
    return _vgpOseenFormulation;
  }
};


#endif
