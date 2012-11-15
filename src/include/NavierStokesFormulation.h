//
//  NavierStokesFormulation.h
//  Camellia
//
//  Created by Nathan Roberts on 4/2/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_NavierStokesFormulation_h
#define Camellia_NavierStokesFormulation_h

#include "StokesFormulation.h"

// implementation of some standard Navier-Stokes Formulations.
class NavierStokesFormulation {
protected:
  double _Re;
  SolutionPtr _soln;
public:
  NavierStokesFormulation(double Reynolds, SolutionPtr soln) {
    _Re = Reynolds;
    _soln = soln;
  }
  
  virtual BFPtr bf() = 0;
  virtual RHSPtr rhs(FunctionPtr f1, FunctionPtr f2) = 0;
  // so far, only have support for BCs defined on the entire boundary (i.e. no outflow type conditions)
  virtual BCPtr bc(FunctionPtr u1, FunctionPtr u2, SpatialFilterPtr entireBoundary) = 0;
  virtual IPPtr graphNorm() = 0;
  virtual void primaryTrialIDs(vector<int> &fieldIDs) = 0; // used for best approximation error TeX output (u1,u2) or (u1,u2,p)
  virtual void trialIDs(vector<int> &fieldIDs, vector<int> &correspondingTraceIDs, vector<string> &fileFriendlyNames) = 0; // corr. ID == -1 if there isn't one
  virtual Teuchos::RCP<ExactSolution> exactSolution(FunctionPtr u1, FunctionPtr u2, FunctionPtr p, 
                                                    SpatialFilterPtr entireBoundary) = 0;
  
  double Re() {
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
    double pMeasure = p_exact->integrate(mesh);
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
  
public:
  static BFPtr stokesBF(double mu) {
    VGPStokesFormulation stokesFormulation(mu);
    return stokesFormulation.bf();
  }
  
  VGPNavierStokesFormulation(double Re, SolutionPtr soln) : NavierStokesFormulation(Re, soln) {
    double mu = 1.0 / Re;
    
    initVars();
    
    _stokesBF = stokesBF(mu);
    
    // construct bilinear form:
    _bf = stokesBF(mu);

    _bf->addTerm(- sigma11_prev * u1 - sigma12_prev * u2 - u1_prev * sigma11 - u2_prev * sigma12, v1);
    _bf->addTerm(- sigma21_prev * u1 - sigma22_prev * u2 - u1_prev * sigma21 - u2_prev * sigma22, v2);
    
    _graphNorm = _bf->graphNorm(); // just use the automatic for now
  }
  BFPtr bf() {
    return _bf;
  }
  IPPtr graphNorm() {
    return _graphNorm;
  }
  RHSPtr rhs(FunctionPtr f1, FunctionPtr f2) {
    Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
    rhs->addTerm( f1 * v1 + f2 * v2 );
    // add the subtraction of the stokes BF here!
    rhs->addTerm( -_stokesBF->testFunctional(_soln) );
    // finally, add the u sigma term:
    rhs->addTerm( (u1_prev * sigma11_prev + u2_prev * sigma12_prev) * v1 );
    rhs->addTerm( (u1_prev * sigma21_prev + u2_prev * sigma22_prev) * v2 );
    
    return rhs;
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
    double mu = 1.0 / _Re;
    FunctionPtr f1 = -p_exact->dx() + mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy())
                     - u1_exact * u1_exact->dx() - u2_exact * u1_exact->dy();
    FunctionPtr f2 = -p_exact->dy() + mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy())
                     - u1_exact * u2_exact->dx() - u2_exact * u2_exact->dy();
    
    BCPtr bc = this->bc(u1_exact, u2_exact, entireBoundary);
    
    RHSPtr rhs = this->rhs(f1,f2);
    Teuchos::RCP<ExactSolution> mySolution = Teuchos::rcp( new ExactSolution(_bf, bc, rhs) );
    mySolution->setSolutionFunction(u1, u1_exact);
    mySolution->setSolutionFunction(u2, u2_exact);
    
    mySolution->setSolutionFunction(p, p_exact);
    
    mySolution->setSolutionFunction(sigma11, u1_exact->dx());
    mySolution->setSolutionFunction(sigma12, u1_exact->dy());
    mySolution->setSolutionFunction(sigma21, u2_exact->dx());
    mySolution->setSolutionFunction(sigma22, u2_exact->dy());
    
    return mySolution;
  }
  
  void primaryTrialIDs(vector<int> &fieldIDs) {
    // (u1,u2,p) 
    double mu = 1.0 / _Re;
    VGPStokesFormulation stokesFormulation(mu);
    stokesFormulation.primaryTrialIDs(fieldIDs);
  }
  void trialIDs(vector<int> &fieldIDs, vector<int> &correspondingTraceIDs, vector<string> &fileFriendlyNames) {
    double mu = 1.0 / _Re;
    VGPStokesFormulation stokesFormulation(mu);
    stokesFormulation.trialIDs(fieldIDs,correspondingTraceIDs,fileFriendlyNames);
  }
};

class VGPNavierStokesProblem {
  SolutionPtr _backgroundFlow, _solnIncrement;
  Teuchos::RCP<Mesh> _mesh;
  Teuchos::RCP<BC> _bc, _bcForIncrement;
  Teuchos::RCP< ExactSolution > _exactSolution;
  Teuchos::RCP<BF> _bf;
  
  Teuchos::RCP< NavierStokesFormulation > _vgpNavierStokesFormulation;
  int _iterations;
  double _iterationWeight;
public:
  VGPNavierStokesProblem(double Re, FieldContainer<double> &quadPoints, int horizontalCells,
                         int verticalCells, int H1Order, int pToAdd,
                         FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact) {
    double mu = 1/Re;
    _iterations = 0;
    _iterationWeight = 1.0;
    
    Teuchos::RCP< VGPStokesFormulation > vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
    
    // create a new mesh:
    _mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd);
    
    
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered ); // SpatialFilterUnfiltered returns true everywhere
    
    Teuchos::RCP<ExactSolution> vgpStokesExactSolution = vgpStokesFormulation->exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
    
    BCPtr vgpBC = vgpStokesFormulation->bc(u1_exact, u2_exact, entireBoundary);
    
    _mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd);
    
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
  void iterate() {
    if (_iterations==0) {
      _backgroundFlow->solve();
    } else {
      _solnIncrement->solve();
      _backgroundFlow->addSolution(_solnIncrement, _iterationWeight);
    }
  }
  int iterationCount() {
    return _iterations;
  }
  Teuchos::RCP<Mesh> mesh() {
    return _mesh;
  }
  void setIP( IPPtr ip ) {
    _backgroundFlow->setIP( ip );
    _solnIncrement->setIP( ip );
  }
  BFPtr stokesBF() {
    double mu =  1.0 / _vgpNavierStokesFormulation->Re();
    return VGPNavierStokesFormulation::stokesBF( mu );
  }
};

#endif
