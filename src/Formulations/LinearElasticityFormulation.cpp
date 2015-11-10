
//  LinearElasticityFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#include "LinearElasticityFormulation.h"

#include "ConstantScalarFunction.h"
#include "Constraint.h"
#include "GMGSolver.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PoissonFormulation.h"
#include "PreviousSolutionFunction.h"
#include "SimpleFunction.h"
#include "SuperLUDistSolver.h"

using namespace Camellia;

const string LinearElasticityFormulation::S_U1 = "u_1";
const string LinearElasticityFormulation::S_U2 = "u_2";
const string LinearElasticityFormulation::S_U3 = "u_3";
const string LinearElasticityFormulation::S_W = "w";
const string LinearElasticityFormulation::S_SIGMA11 = "\\sigma_{11}";
const string LinearElasticityFormulation::S_SIGMA12 = "\\sigma_{12}";
const string LinearElasticityFormulation::S_SIGMA13 = "\\sigma_{13}";
const string LinearElasticityFormulation::S_SIGMA21 = "\\sigma_{21}";
const string LinearElasticityFormulation::S_SIGMA22 = "\\sigma_{22}";
const string LinearElasticityFormulation::S_SIGMA23 = "\\sigma_{23}";
const string LinearElasticityFormulation::S_SIGMA31 = "\\sigma_{31}";
const string LinearElasticityFormulation::S_SIGMA32 = "\\sigma_{32}";
const string LinearElasticityFormulation::S_SIGMA33 = "\\sigma_{33}";

const string LinearElasticityFormulation::S_U1_HAT = "\\widehat{u}_1";
const string LinearElasticityFormulation::S_U2_HAT = "\\widehat{u}_2";
const string LinearElasticityFormulation::S_U3_HAT = "\\widehat{u}_3";
const string LinearElasticityFormulation::S_TN1_HAT = "\\widehat{t}_{1n}";
const string LinearElasticityFormulation::S_TN2_HAT = "\\widehat{t}_{2n}";
const string LinearElasticityFormulation::S_TN3_HAT = "\\widehat{t}_{3n}";

const string LinearElasticityFormulation::S_V1 = "v_1";
const string LinearElasticityFormulation::S_V2 = "v_2";
const string LinearElasticityFormulation::S_V3 = "v_3";
const string LinearElasticityFormulation::S_TAU1 = "\\tau_{1}";
const string LinearElasticityFormulation::S_TAU2 = "\\tau_{2}";
const string LinearElasticityFormulation::S_TAU3 = "\\tau_{3}";
const string LinearElasticityFormulation::S_Q = "q";

static const int INITIAL_CONDITION_TAG = 1;

LinearElasticityFormulation LinearElasticityFormulation::steadyFormulation(int spaceDim, double lambda,
                                                                           double mu, bool useConformingTraces)
{
  Teuchos::ParameterList parameters;
  
  parameters.set("spaceDim", spaceDim);
  parameters.set("lambda",lambda);
  parameters.set("mu",mu);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  
  return LinearElasticityFormulation(parameters);
}

LinearElasticityFormulation::LinearElasticityFormulation(Teuchos::ParameterList &parameters)
{
  // basic parameters
  int spaceDim = parameters.get<int>("spaceDim");
  double mu = parameters.get<double>("mu",1.0);
  double lambda = parameters.get<double>("lambda",1.0);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces",false);
  
  TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument, "spaceDim != not yet supported");
  
  //////////////////////   DECLARE VARIABLES   ////////////////////////
  _vf = VarFactory::varFactory();
  
  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr u1, u2, u3;
  VarPtr w;
  VarPtr sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33;
  
  // traces
  VarPtr u1_hat, u2_hat, u3_hat;
  VarPtr t1n, t2n, t3n;
  
  // tests
  VarPtr v1, v2, v3;
  VarPtr tau1, tau2, tau3;
  VarPtr q;
  
  // trials:
  w       = _vf->fieldVar(S_W, L2);
  u1       = _vf->fieldVar(S_U1, L2);
  u2       = _vf->fieldVar(S_U2, L2);
  sigma11 = _vf->fieldVar(S_SIGMA11, L2);
  sigma12 = _vf->fieldVar(S_SIGMA12, L2);
  sigma22 = _vf->fieldVar(S_SIGMA22, L2);
  
  FunctionPtr n    = Function::normal();
  FunctionPtr one = Function::constant(1.0); // reuse Function to take advantage of accelerated BasisReconciliation (probably this is not the cleanest way to do this, but it should work)

  // H^1 traces
  Space uHatSpace = useConformingTraces ? HGRAD : L2;
  if (spaceDim > 0) u1_hat = _vf->traceVar(S_U1_HAT, one * u1, uHatSpace);
  if (spaceDim > 1) u2_hat = _vf->traceVar(S_U2_HAT, one * u2, uHatSpace);
  if (spaceDim > 2) u3_hat = _vf->traceVar(S_U3_HAT, one * u3, uHatSpace);

  // H(div) traces: (w/STRICTLY 2D termTraced definition)
  t1n = _vf->fluxVar(S_TN1_HAT, sigma11 * n->x() + sigma12 * n->y());
  t2n = _vf->fluxVar(S_TN2_HAT, sigma12 * n->x() + sigma22 * n->y());
  
  cout << "u1_hat->termTraced: " << u1_hat->termTraced()->displayString() << endl;
  cout << "u2_hat->termTraced: " << u2_hat->termTraced()->displayString() << endl;
  cout << "t1n->termTraced: " << t1n->termTraced()->displayString() << endl;
  cout << "t2n->termTraced: " << t2n->termTraced()->displayString() << endl;
  
  // tests:
  tau1 = _vf->testVar("\\tau_1", HDIV);
  tau2 = _vf->testVar("\\tau_2", HDIV);
  v1   = _vf->testVar("v_1", HGRAD);
  v2   = _vf->testVar("v_2", HGRAD);
  
  ////////////////    MISCELLANEOUS LOCAL VARIABLES    ////////////////
  
  auto kronDelta = [] (int i, int j) -> int
  {
    return (i == j) ? 1 : 0;
  };
  
  // Compliance Tensor
  int N = 2;
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 2; ++j)
    {
      for (int k = 0; k < 2; ++k)
      {
        for (int l = 0; l < 2; ++l)
        {
          _C[i][j][k][l] = 1/(2*mu)*(0.5*(kronDelta(i,k)*kronDelta(j,l)+kronDelta(i,l)*kronDelta(j,k))
                                    - lambda/(2*mu+N*lambda)*kronDelta(i,j)*kronDelta(k,l));
          // cout << "C(" << i << "," << j << "," << k << "," << l << ") = " << C[i][j][k][l] << endl;
        }
      }
    }
  }
  
  // Stiffness Tensor
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 2; ++j)
    {
      for (int k = 0; k < 2; ++k)
      {
        for (int l = 0; l < 2; ++l)
        {
          _E[i][j][k][l] = (2*mu)*0.5*(kronDelta(i,k)*kronDelta(j,l)+kronDelta(i,l)*kronDelta(j,k))
                         + lambda*kronDelta(i,j)*kronDelta(k,l);
          // cout << "C(" << i << "," << j << "," << k << "," << l << ") = " << C[i][j][k][l] << endl;
        }
      }
    }
  }
  
  LinearTermPtr sigma[2][2];
  sigma[0][0] = 1*sigma11;
  sigma[0][1] = 1*sigma12;
  sigma[1][0] = 1*sigma12;
  sigma[1][1] = 1*sigma22;
  
  LinearTermPtr tau[2][2];
  tau[0][0] = 1*tau1->x();
  tau[0][1] = 1*tau1->y();
  tau[1][0] = 1*tau2->x();
  tau[1][1] = 1*tau2->y();
  
  FunctionPtr zero = Function::zero();
  FunctionPtr x    = Function::xn(1);
  FunctionPtr y    = Function::yn(1);
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  _bf = Teuchos::rcp( new BF(_vf) );
  
  _bf->addTerm(u1,-tau1->div());
  _bf->addTerm(u2,-tau2->div());
  _bf->addTerm(w,-tau1->y());
  _bf->addTerm(w, tau2->x());
  _bf->addTerm(u1_hat, tau1->dot_normal());
  _bf->addTerm(u2_hat, tau2->dot_normal());
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 2; ++j)
    {
      for (int k = 0; k < 2; ++k)
      {
        for (int l = 0; l < 2; ++l)
        {
          if (abs(_C[i][j][k][l])>1e-14)
          {
            _bf->addTerm(sigma[k][l],-Function::constant(_C[i][j][k][l])*tau[i][j]);
          }
        }
      }
    }
  }
  
  _bf->addTerm(sigma11, v1->dx());
  _bf->addTerm(sigma12, v1->dy());
  _bf->addTerm(sigma12, v2->dx());
  _bf->addTerm(sigma22, v2->dy());
  // omega term missing
  _bf->addTerm(t1n,-v1);
  _bf->addTerm(t2n,-v2);
  
  // PRINT BILINEAR FORM
  // cout << bf->displayString() << endl;
    
  _spaceDim = spaceDim;
  _useConformingTraces = useConformingTraces;
  _mu = mu;
  _lambda = lambda;
  
  if ((spaceDim != 2) && (spaceDim != 3))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim must be 2 or 3");
  }
}

BFPtr LinearElasticityFormulation::bf()
{
  return _bf;
}

// ! compliance tensor
double LinearElasticityFormulation::C(int i, int j, int k, int l)
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  CHECK_VALID_COMPONENT(k);
  CHECK_VALID_COMPONENT(l);
  return _C[i-1][j-1][k-1][l-1];
}


// ! stiffness tensor
double LinearElasticityFormulation::E(int i, int j, int k, int l)
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  CHECK_VALID_COMPONENT(k);
  CHECK_VALID_COMPONENT(l);
  return _E[i-1][j-1][k-1][l-1];
}

void LinearElasticityFormulation::CHECK_VALID_COMPONENT(int i) // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)
{
  if ((i > _spaceDim) || (i < 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component indices must be at least 1 and less than or equal to _spaceDim");
  }
}

TFunctionPtr<double> LinearElasticityFormulation::forcingFunction(TFunctionPtr<double> u_exact)
{
  vector<FunctionPtr> f_vector(_spaceDim, Function::zero());
  for (int i=1; i<= _spaceDim; i++)
  {
    for (int j=1; j<= _spaceDim; j++)
    {
      for (int k=1; k<= _spaceDim; k++)
      {
        FunctionPtr u_k = u_exact->spatialComponent(k);
        for (int l=1; l<= _spaceDim; l++)
        {
          FunctionPtr u_k_lj = u_k->grad()->spatialComponent(l)->grad()->spatialComponent(j);
          double E_ijkl = this->E(i, j, k, l);
          //            cout << i << ", " << j << ", " << k << ", " << l << ": ";
          //            cout << -C_ijkl << " * " << u_k_lj->displayString() << endl;
          if (E_ijkl != 0)
            f_vector[i-1] = f_vector[i-1] -E_ijkl * u_k_lj;
          //            cout << f_vector[i-1]->displayString() << endl;
        }
      }
    }
    
    //      cout << "f[" << i << "]: " << f_vector[i-1]->displayString() << endl;
  }
  
  FunctionPtr f = Function::vectorize(f_vector);
  return f;
}

double LinearElasticityFormulation::lambda()
{
  return _lambda;
}


double LinearElasticityFormulation::mu()
{
  return _mu;
}

VarPtr LinearElasticityFormulation::w()
{
  return _vf->fieldVar(S_W);
}

RefinementStrategyPtr LinearElasticityFormulation::getRefinementStrategy()
{
  return _refinementStrategy;
}

void LinearElasticityFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void LinearElasticityFormulation::refine()
{
  _refinementStrategy->refine();
}

void LinearElasticityFormulation::hRefine()
{
  _hRefinementStrategy->refine();
}

void LinearElasticityFormulation::pRefine()
{
  _pRefinementStrategy->refine();
}

RHSPtr LinearElasticityFormulation::rhs(TFunctionPtr<double> f)
{
  RHSPtr rhs = RHS::rhs();

  VarPtr v1 = this->v(1);
  VarPtr v2 = this->v(2);
  VarPtr v3;
  if (_spaceDim==3) v3 = this->v(3);

  if (f != Teuchos::null)
  {
    rhs->addTerm( f->x() * v1 );
    rhs->addTerm( f->y() * v2 );
    if (_spaceDim == 3) rhs->addTerm( f->z() * v3 );
  }

  return rhs;
}

VarPtr LinearElasticityFormulation::sigma(int i, int j)
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  static const vector<vector<string>> sigmaStrings = {{S_SIGMA11, S_SIGMA12, S_SIGMA13},{S_SIGMA21, S_SIGMA22, S_SIGMA23},{S_SIGMA31, S_SIGMA32, S_SIGMA33}};
  
  return _vf->fieldVar(sigmaStrings[i-1][j-1]);
}

VarPtr LinearElasticityFormulation::u(int i)
{
  CHECK_VALID_COMPONENT(i);
  
  static const vector<string> uStrings = {S_U1,S_U2,S_U3};
  return _vf->fieldVar(uStrings[i-1]);
}

// traces:
VarPtr LinearElasticityFormulation::tn_hat(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> tnStrings = {S_TN1_HAT,S_TN2_HAT,S_TN3_HAT};
  return _vf->fluxVar(tnStrings[i-1]);
}

VarPtr LinearElasticityFormulation::u_hat(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> uHatStrings = {S_U1_HAT,S_U2_HAT,S_U3_HAT};
  return _vf->traceVar(uHatStrings[i-1]);
}

// test variables:
VarPtr LinearElasticityFormulation::tau(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION((i > _spaceDim) || (i < 1), std::invalid_argument, "i must be at least 1 and less than or equal to _spaceDim");
  vector<string> tauStrings = {S_TAU1,S_TAU2,S_TAU3};
  return _vf->testVar(tauStrings[i-1], HDIV);
}

int LinearElasticityFormulation::spaceDim()
{
  return _spaceDim;
}

void LinearElasticityFormulation::turnOffSuperLUDistOutput(Teuchos::RCP<GMGSolver> gmgSolver){
  Teuchos::RCP<GMGOperator> gmgOperator = gmgSolver->gmgOperator();
  while (gmgOperator->getCoarseOperator() != Teuchos::null)
  {
    gmgOperator = gmgOperator->getCoarseOperator();
  }
  SolverPtr coarseSolver = gmgOperator->getCoarseSolver();
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  SuperLUDistSolver* superLUSolver = dynamic_cast<SuperLUDistSolver*>(coarseSolver.get());
  if (superLUSolver)
  {
    superLUSolver->setRunSilent(true);
  }
#endif
}

VarPtr LinearElasticityFormulation::v(int i)
{
  if ((i > _spaceDim) || (i < 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be at least 1 and less than or equal to _spaceDim");
  }
  const static vector<string> vStrings = {S_V1,S_V2,S_V3};
  return _vf->testVar(vStrings[i-1], HGRAD);
}

const std::map<int,int> & LinearElasticityFormulation::getTrialVariablePolyOrderAdjustments()
{
  return _trialVariablePolyOrderAdjustments;
}