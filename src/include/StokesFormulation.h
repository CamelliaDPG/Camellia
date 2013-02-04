//
//  StokesFormulation.h
//  Camellia
//
//  Created by Nathan Roberts on 4/2/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_StokesFormulation_h
#define Camellia_StokesFormulation_h

#include "BF.h"
#include "BCEasy.h"
#include "RHSEasy.h"
#include "VarFactory.h"
#include "Var.h"
#include "ExactSolution.h"

// implementation of some standard Stokes Formulations.

enum StokesFormulationChoice {
  VSP, VVP, VGP, DDS, DDSP
};

// StokesFormulation class: a prototype which I hope to reuse elsewhere once it's ready
class StokesFormulation {
public:
  virtual BFPtr bf() = 0;
  virtual RHSPtr rhs(FunctionPtr f1, FunctionPtr f2) = 0;
  // so far, only have support for BCs defined on the entire boundary (i.e. no outflow type conditions)
  virtual BCPtr bc(FunctionPtr u1, FunctionPtr u2, SpatialFilterPtr entireBoundary) = 0;
  virtual IPPtr graphNorm() = 0;
  virtual void primaryTrialIDs(vector<int> &fieldIDs) = 0; // used for best approximation error TeX output (u1,u2) or (u1,u2,p)
  virtual void trialIDs(vector<int> &fieldIDs, vector<int> &correspondingTraceIDs, vector<string> &fileFriendlyNames) = 0; // corr. ID == -1 if there isn't one
  virtual Teuchos::RCP<ExactSolution> exactSolution(FunctionPtr u1, FunctionPtr u2, FunctionPtr p, 
                                                    SpatialFilterPtr entireBoundary) = 0;
};

class VSPStokesFormulation : public StokesFormulation {
  VarFactory varFactory;
  // fields:
  VarPtr u1, u2, p, sigma11, sigma12, sigma22, omega;
  // fluxes & traces:
  VarPtr u1hat, u2hat, sigma1n, sigma2n;
  // tests:
  VarPtr tau1, tau2, v1, v2, v3;
  BFPtr _bf;
  IPPtr _graphNorm;
  double _mu;
  
public:
  VSPStokesFormulation(double mu) {
    _mu = mu;
    
    tau1 = varFactory.testVar("\\tau_1", HDIV);
    tau2 = varFactory.testVar("\\tau_2", HDIV);
    v1 = varFactory.testVar("v_1", HGRAD);
    v2 = varFactory.testVar("v_2", HGRAD);
    v3 = varFactory.testVar("v_3", HGRAD);
    
    u1hat = varFactory.traceVar("\\widehat{u}_1");
    u2hat = varFactory.traceVar("\\widehat{u}_2");
    sigma1n = varFactory.fluxVar("\\widehat{\\sigma}_{1n}");
    sigma2n = varFactory.fluxVar("\\widehat{\\sigma}_{2n}");
    //    unhat = varFactory.fluxVar("\\widehat{u}_n"); // TEMPORARY: added back in for testing
    
    u1 = varFactory.fieldVar("u_1");
    u2 = varFactory.fieldVar("u_2");
    sigma11 = varFactory.fieldVar("\\sigma_{11}");
    sigma12 = varFactory.fieldVar("\\sigma_{12}");
    sigma22 = varFactory.fieldVar("\\sigma_{22}");
    omega = varFactory.fieldVar("\\omega");
    p = varFactory.fieldVar("p");
    
    // construct bilinear form:
    _bf = Teuchos::rcp( new BF(varFactory) );
    
    double mu_weight = 1.0 / (2.0 * _mu);
    // tau1 terms:
    _bf->addTerm( mu_weight * sigma11 + mu_weight * p, tau1->x() );
    _bf->addTerm( mu_weight * sigma12 + omega,         tau1->y() );
    _bf->addTerm( u1,                                  tau1->div());
    _bf->addTerm(-u1hat,                               tau1->dot_normal());
    
    // tau2 terms:
    _bf->addTerm( mu_weight * sigma12 - omega,         tau2->x() );
    _bf->addTerm( mu_weight * sigma22 + mu_weight * p, tau2->y() );
    _bf->addTerm( u2,                                  tau2->div());
    _bf->addTerm( - u2hat,                             tau2->dot_normal());
    
    // v1:
    _bf->addTerm( sigma11,    v1->dx()); // ( sigma1, grad v1) 
    _bf->addTerm( sigma12,    v1->dy());
    _bf->addTerm( - sigma1n,  v1);
    
    // v2:
    _bf->addTerm( sigma12,    v2->dx()); // ( sigma2, grad v2)
    _bf->addTerm( sigma22,    v2->dy());
    _bf->addTerm( - sigma2n,  v2);
    
    // v3:
    _bf->addTerm(-u1,v3->dx()); // (-u, grad q)
    _bf->addTerm(-u2,v3->dy());
    _bf->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), v3);
    //    _bf->addTerm(unhat,v3);
    
    _graphNorm = Teuchos::rcp( new IP );
    _graphNorm->addTerm( mu_weight * tau1->x() + v1->dx() ); // sigma11
    _graphNorm->addTerm( mu_weight * tau1->y() + mu_weight * tau2->x() + v1->dy() + v2->dx() ); // sigma21
    _graphNorm->addTerm( mu_weight * tau2->y() + v2->dy() ); // sigma22
    _graphNorm->addTerm( mu_weight * tau1->x() + mu_weight * tau2->y() );     // pressure
    _graphNorm->addTerm( tau1->y() - tau2->x() );     // omega
    _graphNorm->addTerm( tau1->div() - v3->dx() );    // u1
    _graphNorm->addTerm( tau2->div() - v3->dy() );    // u2
    
    // L^2 terms:
    _graphNorm->addTerm( v1 );
    _graphNorm->addTerm( v2 );
    _graphNorm->addTerm( v3 );
    _graphNorm->addTerm( tau1 );
    _graphNorm->addTerm( tau2 );
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
    return rhs;
  }
  BCPtr bc(FunctionPtr u1_fxn, FunctionPtr u2_fxn, SpatialFilterPtr entireBoundary) {
    Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
    bc->addDirichlet(u1hat, entireBoundary, u1_fxn);
    bc->addDirichlet(u2hat, entireBoundary, u2_fxn);
    FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
    //    bc->addDirichlet(unhat, entireBoundary, u1_fxn * n->x() + u2_fxn * n->y());
    bc->addZeroMeanConstraint(p);
    return bc;
  }
  Teuchos::RCP<ExactSolution> exactSolution(FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact,
                                            SpatialFilterPtr entireBoundary) {
    FunctionPtr f1 = p_exact->dx() - _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
    FunctionPtr f2 = p_exact->dy() - _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
    
    BCPtr bc = this->bc(u1_exact, u2_exact, entireBoundary);
    
    RHSPtr rhs = this->rhs(f1,f2);
    Teuchos::RCP<ExactSolution> mySolution = Teuchos::rcp( new ExactSolution(_bf, bc, rhs) );
    mySolution->setSolutionFunction(u1, u1_exact);
    mySolution->setSolutionFunction(u2, u2_exact);
    
    mySolution->setSolutionFunction(p, p_exact);
    
    mySolution->setSolutionFunction(sigma11, (2.0 * _mu) * u1_exact->dx() - p_exact);
    mySolution->setSolutionFunction(sigma12, _mu * u1_exact->dy() + _mu * u2_exact->dx());
    mySolution->setSolutionFunction(sigma22, (2.0 * _mu) * u2_exact->dy() - p_exact);
    mySolution->setSolutionFunction(omega, 0.5 * u1_exact->dy() - 0.5 * u2_exact->dx() );
    
    return mySolution;
  }
  
  void primaryTrialIDs(vector<int> &fieldIDs) {
    // (u1,u2,p) 
    fieldIDs.clear();
    fieldIDs.push_back(u1->ID());
    fieldIDs.push_back(u2->ID());
    fieldIDs.push_back(p->ID());
  }
  void trialIDs(vector<int> &fieldIDs, vector<int> &correspondingTraceIDs, vector<string> &fileFriendlyNames) {
    // corr. ID == -1 if there isn't one
    int NONE = -1;
    fieldIDs.clear();  correspondingTraceIDs.clear();  fileFriendlyNames.clear();
    fieldIDs.push_back(u1->ID());
    fileFriendlyNames.push_back("u1");
    correspondingTraceIDs.push_back(u1hat->ID());
    fieldIDs.push_back(u2->ID());
    fileFriendlyNames.push_back("u2");
    correspondingTraceIDs.push_back(u2hat->ID());
    fieldIDs.push_back(p->ID());
    fileFriendlyNames.push_back("pressure");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(sigma11->ID());
    fileFriendlyNames.push_back("sigma11");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(sigma12->ID());
    fileFriendlyNames.push_back("sigma12");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(sigma22->ID());
    fileFriendlyNames.push_back("sigma22");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(omega->ID());
    fileFriendlyNames.push_back("omega");
    correspondingTraceIDs.push_back(NONE);
  }
};

const static string VVP_V_S = "\\boldsymbol{v}";
const static string VVP_Q1_S = "q_1";
const static string VVP_Q2_S = "q_2";

const static string VVP_U1HAT_S = "\\widehat{u}_1";
const static string VVP_U2HAT_S = "\\widehat{u}_2";
const static string VVP_U_CROSS_HAT_S = "\\widehat{u}_{\\times n}";
const static string VVP_U_DOT_HAT_S = "\\widehat{u}_n";
const static string VVP_OMEGA_HAT_S = "\\widehat{\\omega}";
const static string VVP_P_HAT_S = "\\widehat{p}";

const static string VVP_U1_S = "u_1";
const static string VVP_U2_S = "u_2";
const static string VVP_OMEGA_S = "\\omega";
const static string VVP_P_S = "p";


class VVPStokesFormulation : public StokesFormulation {
  VarFactory varFactory;
  // fields:
  VarPtr u1, u2, p, omega;
  // fluxes & traces:
  VarPtr u1hat, u2hat, omega_hat, p_hat;
  // these are the true traces for u:
  VarPtr u_n, u_xn; // dot with n and cross with n.  Both should be fluxes.
  // tests:
  VarPtr v, q1, q2;
  BFPtr _bf;
  IPPtr _graphNorm;
  double _mu;
  bool _trueTraces;
  
public:
  static VarFactory vvpVarFactory(bool trueTraces = false) {
    // sets the order of the variables in a canonical way
    // uses the publicly accessible strings from above, so that VarPtrs
    // can be looked up...

    VarFactory varFactory;
    VarPtr myVar;
    myVar = varFactory.testVar(VVP_V_S, VECTOR_HGRAD);
    myVar = varFactory.testVar(VVP_Q1_S, HGRAD);
    myVar = varFactory.testVar(VVP_Q2_S, HGRAD);
    
    if (!trueTraces) {
      myVar = varFactory.traceVar(VVP_U1HAT_S);
      myVar = varFactory.traceVar(VVP_U2HAT_S);
    } else {
      myVar = varFactory.fluxVar(VVP_U_DOT_HAT_S);
      myVar = varFactory.fluxVar(VVP_U_CROSS_HAT_S);
    }
    myVar = varFactory.traceVar(VVP_OMEGA_HAT_S);
    myVar = varFactory.traceVar(VVP_P_HAT_S);
    
    myVar = varFactory.fieldVar(VVP_U1_S);
    myVar = varFactory.fieldVar(VVP_U2_S);
    myVar = varFactory.fieldVar(VVP_OMEGA_S);
    myVar = varFactory.fieldVar(VVP_P_S);

    return varFactory;
  }
  
  void initVars(bool trueTraces) {
    // create the VarPtrs:
    varFactory = vvpVarFactory(trueTraces);
    
    // look up the created VarPtrs:
    v = varFactory.testVar(VVP_V_S, VECTOR_HGRAD);
    q1 = varFactory.testVar(VVP_Q1_S, HGRAD);
    q2 = varFactory.testVar(VVP_Q2_S, HGRAD);
    
    if (!trueTraces) {
      u1hat = varFactory.traceVar(VVP_U1HAT_S);
      u2hat = varFactory.traceVar(VVP_U2HAT_S);
    } else {
      u_n = varFactory.fluxVar(VVP_U_DOT_HAT_S);
      u_xn = varFactory.fluxVar(VVP_U_CROSS_HAT_S);
    }
    omega_hat = varFactory.traceVar(VVP_OMEGA_HAT_S);
    p_hat = varFactory.traceVar(VVP_P_HAT_S);
    
    u1 = varFactory.fieldVar(VVP_U1_S);
    u2 = varFactory.fieldVar(VVP_U2_S);
    omega = varFactory.fieldVar(VVP_OMEGA_S);
    p = varFactory.fieldVar(VVP_P_S);
  }
  
  VVPStokesFormulation(double mu, bool trueTraces = false) {
    _mu = mu;
    _trueTraces = trueTraces;
    
    initVars(trueTraces);
    
    // construct bilinear form:
    _bf = Teuchos::rcp( new BF(varFactory) );
    // v terms:
    _bf->addTerm(mu * omega_hat, v->cross_normal());
    _bf->addTerm(- p_hat,v->dot_normal()); // (sigma1, tau1)
    _bf->addTerm(mu * omega,v->curl());
    _bf->addTerm(p, v->div());
    
    // q1 terms:
    if (!trueTraces) {
      _bf->addTerm( u1hat->times_normal_y() - u2hat->times_normal_x(), q1); // <u x n, q1>
    } else {
      _bf->addTerm( u_xn, q1); // <u x n, q1>
    }
    _bf->addTerm(-u1, q1->dy()); // ( -u, curl q1)
    _bf->addTerm( u2, q1->dx());
    _bf->addTerm(omega, q1);
    
    // q2 terms:
    if (!trueTraces) {
      _bf->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), q2); // <u * n, q2>
    } else {
      _bf->addTerm(u_n, q2); // <u * n, q2>
    }
    _bf->addTerm(-u1, q2->dx()); // (-u, grad q2)
    _bf->addTerm(-u2, q2->dy());
    
    _graphNorm = _bf->graphNorm();
//    _graphNorm = Teuchos::rcp( new IP );
//    _graphNorm->addTerm( q1 + mu * v->curl() ); // omega
//    _graphNorm->addTerm( v->div() ); // p
//    _graphNorm->addTerm( q2->grad() + q1->curl() ); // (u1,u2)
//    
//    // L^2 terms:
//    //    _graphNorm->addTerm( v );
//    // experiment on suspicion of the above line:
//    _graphNorm->addTerm( v->x() );
//    _graphNorm->addTerm( v->y() );
//    _graphNorm->addTerm( q1 );
//    _graphNorm->addTerm( q2 );
  }
  BFPtr bf() {
    return _bf;
  }
  IPPtr graphNorm() {
    return _graphNorm;
  }
  RHSPtr rhs(FunctionPtr f1, FunctionPtr f2) {
    Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
    rhs->addTerm( f1 * v->x() + f2 * v->y() );
    return rhs;
  }
  BCPtr bc(FunctionPtr u1_fxn, FunctionPtr u2_fxn, SpatialFilterPtr entireBoundary) {
    Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
    if (!_trueTraces) {
      bc->addDirichlet(u1hat, entireBoundary, u1_fxn);
      bc->addDirichlet(u2hat, entireBoundary, u2_fxn);
    } else {
      FunctionPtr n = Function::normal();
      bc->addDirichlet(u_n,  entireBoundary, u1_fxn * n->x() + u2_fxn * n->y());
      bc->addDirichlet(u_xn, entireBoundary, u1_fxn * n->y() - u2_fxn * n->x());
    }
    bc->addZeroMeanConstraint(p);
    return bc;
  }
  Teuchos::RCP<ExactSolution> exactSolution(FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact,
                                            SpatialFilterPtr entireBoundary) {
    FunctionPtr f1 = - p_exact->dx() + _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
    FunctionPtr f2 = - p_exact->dy() + _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
    
    //    cout << "VVP rhs: f_1 = " << f1->displayString() << "; f_2 = " << f2->displayString() << endl;
    
    BCPtr bc = this->bc(u1_exact, u2_exact, entireBoundary);
    
    RHSPtr rhs = this->rhs(f1,f2);
    Teuchos::RCP<ExactSolution> mySolution = Teuchos::rcp( new ExactSolution(_bf, bc, rhs) );
    mySolution->setSolutionFunction(u1, u1_exact);
    mySolution->setSolutionFunction(u2, u2_exact);
    
    mySolution->setSolutionFunction(p, p_exact);    
    mySolution->setSolutionFunction(omega, u2_exact->dx() - u1_exact->dy()); // curl u
    
    return mySolution;
  }
  
  void primaryTrialIDs(vector<int> &fieldIDs) {
    // (u1,u2,p) 
    fieldIDs.clear();
    fieldIDs.push_back(u1->ID());
    fieldIDs.push_back(u2->ID());
    fieldIDs.push_back(p->ID());
  }
  void trialIDs(vector<int> &fieldIDs, vector<int> &correspondingTraceIDs, vector<string> &fileFriendlyNames) {
    // corr. ID == -1 if there isn't one
    int NONE = -1;
    fieldIDs.clear();  correspondingTraceIDs.clear();  fileFriendlyNames.clear();
    fieldIDs.push_back(u1->ID());
    fileFriendlyNames.push_back("u1");
    if (!_trueTraces)
      correspondingTraceIDs.push_back(u1hat->ID());
    else
      correspondingTraceIDs.push_back(-1);
    fieldIDs.push_back(u2->ID());
    fileFriendlyNames.push_back("u2");
    if (!_trueTraces)
      correspondingTraceIDs.push_back(u2hat->ID());
    else
      correspondingTraceIDs.push_back(-1);
    fieldIDs.push_back(p->ID());
    fileFriendlyNames.push_back("pressure");
    correspondingTraceIDs.push_back(p_hat->ID());
    fieldIDs.push_back(omega->ID());
    fileFriendlyNames.push_back("omega");
    correspondingTraceIDs.push_back(omega_hat->ID());
  }
};

const static string VGP_V1_S = "v_1";
const static string VGP_V2_S = "v_2";
const static string VGP_TAU1_S = "\\tau_1";
const static string VGP_TAU2_S = "\\tau_2";
const static string VGP_Q_S = "q";

const static string VGP_U1HAT_S = "\\widehat{u}_1";
const static string VGP_U2HAT_S = "\\widehat{u}_2";
const static string VGP_T1HAT_S = "\\boldsymbol{t}_{1n}";
const static string VGP_T2HAT_S = "\\boldsymbol{t}_{2n}";

const static string VGP_U1_S = "u_1";
const static string VGP_U2_S = "u_2";
const static string VGP_SIGMA11_S = "\\sigma_{11}";
const static string VGP_SIGMA12_S = "\\sigma_{12}";
const static string VGP_SIGMA21_S = "\\sigma_{21}";
const static string VGP_SIGMA22_S = "\\sigma_{22}";
const static string VGP_P_S = "p";

class VGPStokesFormulation : public StokesFormulation {
  VarFactory varFactory;
  // fields:
  VarPtr u1, u2, p, sigma11, sigma12, sigma21, sigma22;
  // fluxes & traces:
  VarPtr u1hat, u2hat, t1n, t2n;
  // tests:
  VarPtr tau1, tau2, q, v1, v2;
  BFPtr _bf;
  IPPtr _graphNorm;
  double _mu;
  
public:
  static VarFactory vgpVarFactory() {
    // sets the order of the variables in a canonical way
    // uses the publicly accessible strings from above, so that VarPtrs
    // can be looked up...
    
    VarFactory varFactory;
    VarPtr myVar;
    myVar = varFactory.testVar(VGP_V1_S, HGRAD);
    myVar = varFactory.testVar(VGP_V2_S, HGRAD);
    myVar = varFactory.testVar(VGP_TAU1_S, HDIV);
    myVar = varFactory.testVar(VGP_TAU2_S, HDIV);
    myVar = varFactory.testVar(VGP_Q_S, HGRAD);
    
    myVar = varFactory.traceVar(VGP_U1HAT_S);
    myVar = varFactory.traceVar(VGP_U2HAT_S);
    
    myVar = varFactory.fluxVar(VGP_T1HAT_S);
    myVar = varFactory.fluxVar(VGP_T2HAT_S);
    myVar = varFactory.fieldVar(VGP_U1_S);
    myVar = varFactory.fieldVar(VGP_U2_S);
    myVar = varFactory.fieldVar(VGP_SIGMA11_S);
    myVar = varFactory.fieldVar(VGP_SIGMA12_S);
    myVar = varFactory.fieldVar(VGP_SIGMA21_S);
    myVar = varFactory.fieldVar(VGP_SIGMA22_S);
    myVar = varFactory.fieldVar(VGP_P_S);
    return varFactory;
  }
  
  void initVars() {
    // create the VarPtrs:
    varFactory = vgpVarFactory();
    
    // look up the created VarPtrs:
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
  
  VGPStokesFormulation(double mu) {
    _mu = mu;
    
    initVars();
    
    // tau1 is the first *row* of tau
    // (i.e. it's the components of tau that interact with u1
    //  in the tensor product (grad u, tau))
    
    // construct bilinear form:
    _bf = Teuchos::rcp( new BF(varFactory) );
    // tau1 terms:
    _bf->addTerm(u1,tau1->div());
    _bf->addTerm(sigma11,tau1->x()); // (sigma1, tau1)
    _bf->addTerm(sigma12,tau1->y());
    _bf->addTerm(-u1hat, tau1->dot_normal());
    
    // tau2 terms:
    _bf->addTerm(u2, tau2->div());
    _bf->addTerm(sigma21,tau2->x()); // (sigma2, tau2)
    _bf->addTerm(sigma22,tau2->y());
    _bf->addTerm(-u2hat, tau2->dot_normal());
    
    // v1:
    _bf->addTerm(- mu * sigma11,v1->dx()); // (mu sigma1, grad v1) 
    _bf->addTerm(- mu * sigma12,v1->dy());
    _bf->addTerm( p, v1->dx() );
    _bf->addTerm( t1n, v1);
    
    // v2:
    _bf->addTerm(- mu * sigma21,v2->dx()); // (mu sigma2, grad v2)
    _bf->addTerm(- mu * sigma22,v2->dy());
    _bf->addTerm(p, v2->dy());
    _bf->addTerm( t2n, v2);
    
    // q:
    _bf->addTerm(-u1,q->dx()); // (-u, grad q)
    _bf->addTerm(-u2,q->dy());
    _bf->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), q);

    _graphNorm = _bf->graphNorm();
//    _graphNorm = Teuchos::rcp( new IP );
//    _graphNorm->addTerm( mu * v1->dx() + tau1->x() ); // sigma11
//    _graphNorm->addTerm( mu * v1->dy() + tau1->y() ); // sigma12
//    _graphNorm->addTerm( mu * v2->dx() + tau2->x() ); // sigma21
//    _graphNorm->addTerm( mu * v2->dy() + tau2->y() ); // sigma22
//    _graphNorm->addTerm( v1->dx() + v2->dy() );     // pressure
//    _graphNorm->addTerm( tau1->div() - q->dx() );    // u1
//    _graphNorm->addTerm( tau2->div() - q->dy() );    // u2
//    
//    // L^2 terms:
//    _graphNorm->addTerm( v1 );
//    _graphNorm->addTerm( v2 );
//    _graphNorm->addTerm( q );
//    _graphNorm->addTerm( tau1 );
//    _graphNorm->addTerm( tau2 );
  }
  BFPtr bf() {
    return _bf;
  }
  IPPtr graphNorm() {
    return _graphNorm;
  }
  VarPtr ui(int i) {
    if (i==0) return u1;
    if (i==1) return u2;
    return Teuchos::rcp( (Var*) NULL );
  }
  VarPtr vi(int i) {
    if (i==0) return v1;
    if (i==1) return v2;
    return Teuchos::rcp( (Var*) NULL );
  }
  LinearTermPtr dui_dj(int i, int j) {
    if (i==0 && j==0) {
      return 1.0*sigma11;
    }
    if (i==0 && j==1) {
      return 1.0*sigma12;
    }
    if (i==1 && j==0) {
      return 1.0*sigma21;
    }
    if (i==1 && j==1) {
      return 1.0*sigma22;
    }
    return Teuchos::rcp( (LinearTerm*) NULL );
  }
  RHSPtr rhs(FunctionPtr f1, FunctionPtr f2) {
    Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
    rhs->addTerm( f1 * v1 + f2 * v2 );
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
    FunctionPtr f1 = -p_exact->dx() + _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
    FunctionPtr f2 = -p_exact->dy() + _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
    
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
    fieldIDs.clear();
    fieldIDs.push_back(u1->ID());
    fieldIDs.push_back(u2->ID());
    fieldIDs.push_back(p->ID());
  }
  void trialIDs(vector<int> &fieldIDs, vector<int> &correspondingTraceIDs, vector<string> &fileFriendlyNames) {
    // corr. ID == -1 if there isn't one
    int NONE = -1;
    fieldIDs.clear();  correspondingTraceIDs.clear();  fileFriendlyNames.clear();
    fieldIDs.push_back(u1->ID());
    fileFriendlyNames.push_back("u1");
    correspondingTraceIDs.push_back(u1hat->ID());
    fieldIDs.push_back(u2->ID());
    fileFriendlyNames.push_back("u2");
    correspondingTraceIDs.push_back(u2hat->ID());
    fieldIDs.push_back(p->ID());
    fileFriendlyNames.push_back("pressure");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(sigma11->ID());
    fileFriendlyNames.push_back("sigma11");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(sigma12->ID());
    fileFriendlyNames.push_back("sigma12");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(sigma21->ID());
    fileFriendlyNames.push_back("sigma21");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(sigma22->ID());
    fileFriendlyNames.push_back("sigma22");
    correspondingTraceIDs.push_back(NONE);
  }
};

class DDSStokesFormulation : public StokesFormulation {
  VarFactory varFactory;
  // fields:
  VarPtr u1, u2, sigma11, sigma12, sigma22;
  // fluxes & traces:
  VarPtr u1hat, u2hat, sigma1n, sigma2n;
  // tests:
  VarPtr tau11, tau12, tau22, v1, v2;
  BFPtr _bf;
  IPPtr _graphNorm;
  double _mu;
  
public:
  DDSStokesFormulation(double mu) {
    _mu = mu;
    
    v1 = varFactory.testVar("v_1", HGRAD);
    v2 = varFactory.testVar("v_2", HGRAD);
    
    tau11 = varFactory.testVar("\\tau_{11}", HGRAD);
    tau12 = varFactory.testVar("\\tau_{12}", HGRAD);
    tau22 = varFactory.testVar("\\tau_{22}", HGRAD);
    
    sigma1n = varFactory.fluxVar("\\widehat{\\sigma}_{1n}");
    sigma2n = varFactory.fluxVar("\\widehat{\\sigma}_{2n}");
    u1hat = varFactory.traceVar("\\widehat{u}_1");
    u2hat = varFactory.traceVar("\\widehat{u}_2");
    
    u1 = varFactory.fieldVar("u_1");
    u2 = varFactory.fieldVar("u_2");
    sigma11 = varFactory.fieldVar("\\sigma_{11}");
    sigma12 = varFactory.fieldVar("\\sigma_{12}");
    sigma22 = varFactory.fieldVar("\\sigma_{22}");
    
    // construct bilinear form:
    _bf = Teuchos::rcp( new BF(varFactory) );
    
    // v1 terms:
    _bf->addTerm(-sigma11, v1->dx()); // (sigma1, grad v1)
    _bf->addTerm(-sigma12, v1->dy());
    _bf->addTerm(sigma1n, v1);
    
    // v2 terms:
    _bf->addTerm(-sigma12, v2->dx()); // (sigma2, grad v2)
    _bf->addTerm(-sigma22, v2->dy());
    _bf->addTerm(sigma2n, v2);
    
    // tau11 terms:
    _bf->addTerm(sigma11 - sigma22,tau11);
    _bf->addTerm(4 * mu * u1, tau11->dx());
    _bf->addTerm(-4 * mu * u1hat, tau11->times_normal_x());
    
    // tau12 terms:
    _bf->addTerm(sigma12,tau12);
    _bf->addTerm(mu * u2,tau12->dx());
    _bf->addTerm(mu * u1,tau12->dy());
    _bf->addTerm(-mu * u1hat, tau12->times_normal_y());
    _bf->addTerm(-mu * u2hat, tau12->times_normal_x());
    
    // tau22 terms:
    _bf->addTerm(sigma22 - sigma11,tau22);
    _bf->addTerm(4 * mu * u2,tau22->dy());
    _bf->addTerm(- 4 * mu * u2hat, tau22->times_normal_y());
    
    _graphNorm = Teuchos::rcp( new IP );
    _graphNorm->addTerm( tau11 - tau22 - v1->dx() );            // sigma11
    _graphNorm->addTerm( tau12 - v2->dx() - v1->dy() ); // sigma12
    _graphNorm->addTerm( tau22 - tau11 - v2->dy() );            // sigma22
    _graphNorm->addTerm( 4 * mu * tau11->dx() + mu * tau12->dy() );    // u1
    _graphNorm->addTerm( 4 * mu * tau22->dy() + mu * tau12->dx() );    // u2
    _graphNorm->addTerm( v1 );
    _graphNorm->addTerm( v2 );
    _graphNorm->addTerm( tau11 );
    _graphNorm->addTerm( tau12 );
    _graphNorm->addTerm( tau22 );
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
    return rhs;
  }
  BCPtr bc(FunctionPtr u1_fxn, FunctionPtr u2_fxn, SpatialFilterPtr entireBoundary) {
    Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
    bc->addDirichlet(u1hat, entireBoundary, u1_fxn);
    bc->addDirichlet(u2hat, entireBoundary, u2_fxn);
    //    bc->addZeroMeanConstraint(p);
    return bc;
  }
  Teuchos::RCP<ExactSolution> exactSolution(FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact,
                                            SpatialFilterPtr entireBoundary) {
    FunctionPtr f1 = -p_exact->dx() + _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
    FunctionPtr f2 = -p_exact->dy() + _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
    
    BCPtr bc = this->bc(u1_exact, u2_exact, entireBoundary);
    
    RHSPtr rhs = this->rhs(f1,f2);
    Teuchos::RCP<ExactSolution> mySolution = Teuchos::rcp( new ExactSolution(_bf, bc, rhs) );
    mySolution->setSolutionFunction(u1, u1_exact);
    mySolution->setSolutionFunction(u2, u2_exact);
    //    mySolution->setSolutionFunction(p, p_exact);
    
    //      FunctionPtr pTerm = 0.5 * sigma11 + 0.5 * sigma12; // p = tr(sigma) / n
    mySolution->setSolutionFunction(sigma11, - p_exact + 2 * _mu * u1_exact->dx());
    mySolution->setSolutionFunction(sigma12, _mu * u1_exact->dy() + _mu * u2_exact->dx());
    mySolution->setSolutionFunction(sigma22, - p_exact + 2 * _mu * u2_exact->dy());
    
    return mySolution;
  }
  
  void primaryTrialIDs(vector<int> &fieldIDs) {
    // (u1,u2,p) 
    fieldIDs.clear();
    fieldIDs.push_back(u1->ID());
    fieldIDs.push_back(u2->ID());
  }
  void trialIDs(vector<int> &fieldIDs, vector<int> &correspondingTraceIDs, vector<string> &fileFriendlyNames) {
    // corr. ID == -1 if there isn't one
    int NONE = -1;
    fieldIDs.clear();  correspondingTraceIDs.clear();  fileFriendlyNames.clear();
    fieldIDs.push_back(u1->ID());
    fileFriendlyNames.push_back("u1");
    correspondingTraceIDs.push_back(u1hat->ID());
    fieldIDs.push_back(u2->ID());
    fileFriendlyNames.push_back("u2");
    correspondingTraceIDs.push_back(u2hat->ID());
    fieldIDs.push_back(sigma11->ID());
    fileFriendlyNames.push_back("sigma11");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(sigma12->ID());
    fileFriendlyNames.push_back("sigma12");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(sigma22->ID());
    fileFriendlyNames.push_back("sigma22");
    correspondingTraceIDs.push_back(NONE);
  }
};

// DDSP formulation: the DDS plus an algebraically defined pressure (p = - 0.5 * (sigma11 + sigma22) )
class DDSPStokesFormulation : public StokesFormulation {
  VarFactory varFactory;
  // fields:
  VarPtr u1, u2, sigma11, sigma12, sigma22;
  // pressure (algebraically defined)
  VarPtr p;
  // fluxes & traces:
  VarPtr u1hat, u2hat, sigma1n, sigma2n;
  // tests:
  VarPtr tau11, tau12, tau22, v1, v2;
  // test for pressure:
  VarPtr v3;
  BFPtr _bf;
  IPPtr _graphNorm;
  double _mu;
  
public:
  DDSPStokesFormulation(double mu) {
    _mu = mu;
    
    v1 = varFactory.testVar("v_1", HGRAD);
    v2 = varFactory.testVar("v_2", HGRAD);
    v3 = varFactory.testVar("v_3", HGRAD);
    
    tau11 = varFactory.testVar("\\tau_{11}", HGRAD);
    tau12 = varFactory.testVar("\\tau_{12}", HGRAD);
    tau22 = varFactory.testVar("\\tau_{22}", HGRAD);
    
    sigma1n = varFactory.fluxVar("\\widehat{\\sigma}_{1n}");
    sigma2n = varFactory.fluxVar("\\widehat{\\sigma}_{2n}");
    
    u1hat = varFactory.traceVar("\\widehat{u}_1");
    u2hat = varFactory.traceVar("\\widehat{u}_2");
    
    u1 = varFactory.fieldVar("u_1");
    u2 = varFactory.fieldVar("u_2");
    sigma11 = varFactory.fieldVar("\\sigma_{11}");
    sigma12 = varFactory.fieldVar("\\sigma_{12}");
    sigma22 = varFactory.fieldVar("\\sigma_{22}");
    p = varFactory.fieldVar("p");
    
    // construct bilinear form:
    _bf = Teuchos::rcp( new BF(varFactory) );
    
    // v1 terms:
    _bf->addTerm(-sigma11, v1->dx()); // (sigma1, grad v1)
    _bf->addTerm(-sigma12, v1->dy());
    _bf->addTerm(sigma1n, v1);
    
    // v2 terms:
    _bf->addTerm(-sigma12, v2->dx()); // (sigma2, grad v2)
    _bf->addTerm(-sigma22, v2->dy());
    _bf->addTerm(sigma2n, v2);
    
    // v3 terms:
    _bf->addTerm(p + 0.5 * sigma11 + 0.5 * sigma22, v3);
    
    // tau11 terms:
    _bf->addTerm(sigma11 - sigma22,tau11);
    _bf->addTerm(4 * mu * u1, tau11->dx());
    _bf->addTerm(-4 * mu * u1hat, tau11->times_normal_x());
    
    // tau12 terms:
    _bf->addTerm(sigma12,tau12);
    _bf->addTerm(mu * u2,tau12->dx());
    _bf->addTerm(mu * u1,tau12->dy());
    _bf->addTerm(-mu * u1hat, tau12->times_normal_y());
    _bf->addTerm(-mu * u2hat, tau12->times_normal_x());
    
    // tau22 terms:
    _bf->addTerm(sigma22 - sigma11,tau22);
    _bf->addTerm(4 * mu * u2,tau22->dy());
    _bf->addTerm(- 4 * mu * u2hat, tau22->times_normal_y());
    
    _graphNorm = Teuchos::rcp( new IP );
    _graphNorm->addTerm( v3 + tau11 - tau22 - v1->dx() );            // sigma11
    _graphNorm->addTerm( tau12 - v2->dx() - v1->dy() ); // sigma12
    _graphNorm->addTerm( v3 + tau22 - tau11 - v2->dy() );            // sigma22
    _graphNorm->addTerm( 4 * mu * tau11->dx() + mu * tau12->dy() );    // u1
    _graphNorm->addTerm( 4 * mu * tau22->dy() + mu * tau12->dx() );    // u2
    _graphNorm->addTerm( v1 );
    _graphNorm->addTerm( v2 );
    _graphNorm->addTerm( v3 );
    _graphNorm->addTerm( tau11 );
    _graphNorm->addTerm( tau12 );
    _graphNorm->addTerm( tau22 );
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
    FunctionPtr f1 = -p_exact->dx() + _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
    FunctionPtr f2 = -p_exact->dy() + _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
    
    BCPtr bc = this->bc(u1_exact, u2_exact, entireBoundary);
    
    RHSPtr rhs = this->rhs(f1,f2);
    Teuchos::RCP<ExactSolution> mySolution = Teuchos::rcp( new ExactSolution(_bf, bc, rhs) );
    mySolution->setSolutionFunction(u1, u1_exact);
    mySolution->setSolutionFunction(u2, u2_exact);
    mySolution->setSolutionFunction(p, p_exact);
    
    //      FunctionPtr pTerm = 0.5 * sigma11 + 0.5 * sigma12; // p = tr(sigma) / n
    mySolution->setSolutionFunction(sigma11, - p_exact + 2 * _mu * u1_exact->dx());
    mySolution->setSolutionFunction(sigma12, _mu * u1_exact->dy() + _mu * u2_exact->dx());
    mySolution->setSolutionFunction(sigma22, - p_exact + 2 * _mu * u2_exact->dy());
    
    return mySolution;
  }
  
  void primaryTrialIDs(vector<int> &fieldIDs) {
    // (u1,u2,p) 
    fieldIDs.clear();
    fieldIDs.push_back(u1->ID());
    fieldIDs.push_back(u2->ID());
    fieldIDs.push_back(p->ID());
  }
  void trialIDs(vector<int> &fieldIDs, vector<int> &correspondingTraceIDs, vector<string> &fileFriendlyNames) {
    // corr. ID == -1 if there isn't one
    int NONE = -1;
    fieldIDs.clear();  correspondingTraceIDs.clear();  fileFriendlyNames.clear();
    fieldIDs.push_back(u1->ID());
    fileFriendlyNames.push_back("u1");
    correspondingTraceIDs.push_back(u1hat->ID());
    fieldIDs.push_back(u2->ID());
    fileFriendlyNames.push_back("u2");
    correspondingTraceIDs.push_back(u2hat->ID());
    fieldIDs.push_back(sigma11->ID());
    fileFriendlyNames.push_back("sigma11");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(sigma12->ID());
    fileFriendlyNames.push_back("sigma12");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(sigma22->ID());
    fileFriendlyNames.push_back("sigma22");
    correspondingTraceIDs.push_back(NONE);
    fieldIDs.push_back(p->ID());
    fileFriendlyNames.push_back("pressure");
    correspondingTraceIDs.push_back(NONE);
  }
};

#endif
