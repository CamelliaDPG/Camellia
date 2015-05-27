#ifndef NS_FORMULATIONS
#define NS_FORMULATIONS

#include "StokesFormulations.h"

class NSFormulation
{
protected:
  Teuchos::RCP<Solution> _soln;
  double _mu;
public:
  NSFormulation( Teuchos::RCP<Solution> soln, double mu )
  {
    _soln = soln;
    _mu = mu;
  }
  virtual BFPtr bf() = 0;
  static BFPtr stokesBF(double mu) = 0;
  virtual RHSPtr rhs(FunctionPtr f1, FunctionPtr f2) = 0;
  // so far, only have support for BCs defined on the entire boundary (i.e. no outflow type conditions)
  virtual BCPtr bc(FunctionPtr u1, FunctionPtr u2, SpatialFilterPtr entireBoundary) = 0;
  virtual IPPtr graphNorm() = 0;
  virtual void primaryTrialIDs(vector<int> &fieldIDs) = 0; // used for best approximation error TeX output (u1,u2) or (u1,u2,p)
  virtual void trialIDs(vector<int> &fieldIDs, vector<int> &correspondingTraceIDs, vector<string> &fileFriendlyNames) = 0; // corr. ID == -1 if there isn't one
  virtual Teuchos::RCP<ExactSolution> exactSolution(FunctionPtr u1, FunctionPtr u2, FunctionPtr p,
      SpatialFilterPtr entireBoundary) = 0;
};

class VGP_NSFormulation : public NSFormulation
{
  BFPtr _bf;
  LinearTermPtr _nsRHSChange; // relative to the stokes RHS
  BFPtr _stokesBF;

  VarPtr u1, u2, v1, v2;
  LinearTermPtr du1_dx, du1_dy, du2_dx, du2_dy;
public:
  VGP_NSFormulation(Teuchos::RCP<Solution> soln, double mu) : NSFormulation(soln,mu)
  {
    // TODO: define BF and graphNorm
    VGPStokesFormulation stokesForm(mu);
    u1 = stokesForm.ui(0);
    u2 = stokesForm.ui(1);
    v1 = stokesForm.vi(0);
    v2 = stokesForm.vi(1);
    du1_dx = stokesForm.dui_dj(0,0);
    du1_dy = stokesForm.dui_dj(0,1);
    du2_dx = stokesForm.dui_dj(1,0);
    du2_dy = stokesForm.dui_dj(1,1);

    FunctionPtr prev_u1 = Teuchos::rcp( new PreviousSolutionFunction(soln, u1) );
    FunctionPtr prev_u2 = Teuchos::rcp( new PreviousSolutionFunction(soln, u2) );
    FunctionPtr prev_du1_dx = Teuchos::rcp( new PreviousSolutionFunction(soln, du1_dx) );
    FunctionPtr prev_du1_dy = Teuchos::rcp( new PreviousSolutionFunction(soln, du1_dy) );
    FunctionPtr prev_du2_dx = Teuchos::rcp( new PreviousSolutionFunction(soln, du2_dx) );
    FunctionPtr prev_du2_dy = Teuchos::rcp( new PreviousSolutionFunction(soln, du2_dy) );

    _stokesBF = stokesBF(mu);
    _bf = stokesBF(mu); // a second copy, which we'll now modify...

    _bf->addTerm( prev_du1_dx * u1 + prev_du1_dy * u2, v1 );
    _bf->addTerm( prev_du2_dx * u1 + prev_du2_dy * u2, v2 );
    _bf->addTerm( du1_dx * prev_u1 + du1_dy * prev_u2, v1 );
    _bf->addTerm( du2_dx * prev_u1 + du2_dy * prev_u2, v2 );

    // define _nsRHSChange
    LinearTermPtr prev_convection = (prev_du1_dx * prev_u1 + prev_du1_dy * prev_u2) * v1
                                    + (prev_du2_dx * prev_u1 + prev_du2_dy * prev_u2) * v2;
    _nsRHSChange = -_stokesBF->testFunctional(soln) - prev_convection;
  }
  BFPtr bf()
  {
    return _bf;
  }
  static BFPtr stokesBF(double mu)
  {
    return VGPStokesFormulation(mu).bf();
  }
  IPPtr graphNorm()
  {
    return _bf->graphNorm(); // NOTE: the automatic graph norm still needs to be tested...
  }
  RHSPtr rhs(FunctionPtr f1, FunctionPtr f2)
  {
    Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
    rhs->addTerm( f1 * v1 + f2 * v2 );
    return rhs;
  }
  BCPtr bc(FunctionPtr u1_fxn, FunctionPtr u2_fxn, SpatialFilterPtr entireBoundary)
  {
    Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
    bc->addDirichlet(u1hat, entireBoundary, u1_fxn);
    bc->addDirichlet(u2hat, entireBoundary, u2_fxn);
    bc->addZeroMeanConstraint(p);
    return bc;
  }
  Teuchos::RCP<ExactSolution> exactSolution(FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact,
      SpatialFilterPtr entireBoundary)
  {
    FunctionPtr f1 = p_exact->dx() - _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
    FunctionPtr f2 = p_exact->dy() - _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());

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

  void primaryTrialIDs(vector<int> &fieldIDs)
  {
    // (u1,u2,p)
    VGPStokesFormulation(_mu).primaryTrialIDs(fieldIDs);

  }
  void trialIDs(vector<int> &fieldIDs, vector<int> &correspondingTraceIDs, vector<string> &fileFriendlyNames)
  {
    VGPStokesFormulation(_mu).trialIDs(fieldIDs,correspondingTraceIDs,fileFriendlyNames);
  }
};


#endif