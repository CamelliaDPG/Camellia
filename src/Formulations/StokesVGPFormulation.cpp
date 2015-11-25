
//  StokesVGPFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#include "StokesVGPFormulation.h"

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

const string StokesVGPFormulation::S_U1 = "u_1";
const string StokesVGPFormulation::S_U2 = "u_2";
const string StokesVGPFormulation::S_U3 = "u_3";
const string StokesVGPFormulation::S_P = "p";
const string StokesVGPFormulation::S_SIGMA11 = "\\sigma_{11}";
const string StokesVGPFormulation::S_SIGMA12 = "\\sigma_{12}";
const string StokesVGPFormulation::S_SIGMA13 = "\\sigma_{13}";
const string StokesVGPFormulation::S_SIGMA21 = "\\sigma_{21}";
const string StokesVGPFormulation::S_SIGMA22 = "\\sigma_{22}";
const string StokesVGPFormulation::S_SIGMA23 = "\\sigma_{23}";
const string StokesVGPFormulation::S_SIGMA31 = "\\sigma_{31}";
const string StokesVGPFormulation::S_SIGMA32 = "\\sigma_{32}";
const string StokesVGPFormulation::S_SIGMA33 = "\\sigma_{33}";

const string StokesVGPFormulation::S_U1_HAT = "\\widehat{u}_1";
const string StokesVGPFormulation::S_U2_HAT = "\\widehat{u}_2";
const string StokesVGPFormulation::S_U3_HAT = "\\widehat{u}_3";
const string StokesVGPFormulation::S_TN1_HAT = "\\widehat{t}_{1n}";
const string StokesVGPFormulation::S_TN2_HAT = "\\widehat{t}_{2n}";
const string StokesVGPFormulation::S_TN3_HAT = "\\widehat{t}_{3n}";

const string StokesVGPFormulation::S_V1 = "v_1";
const string StokesVGPFormulation::S_V2 = "v_2";
const string StokesVGPFormulation::S_V3 = "v_3";
const string StokesVGPFormulation::S_TAU1 = "\\tau_{1}";
const string StokesVGPFormulation::S_TAU2 = "\\tau_{2}";
const string StokesVGPFormulation::S_TAU3 = "\\tau_{3}";
const string StokesVGPFormulation::S_Q = "q";

static const int INITIAL_CONDITION_TAG = 1;

StokesVGPFormulation StokesVGPFormulation::steadyFormulation(int spaceDim, double mu, bool useConformingTraces)
{
  Teuchos::ParameterList parameters;

  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",mu);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);

  return StokesVGPFormulation(parameters);
}

StokesVGPFormulation StokesVGPFormulation::steadyFormulationStrongConservation(int spaceDim, double mu, bool useConformingTraces)
{
  Teuchos::ParameterList parameters;

  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",mu);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", false);
  parameters.set("useStrongConservation",true);

  return StokesVGPFormulation(parameters);
}

StokesVGPFormulation StokesVGPFormulation::spaceTimeFormulation(int spaceDim, double mu, bool useConformingTraces,
                                                                bool includeVelocityTracesInFluxTerm)
{
  Teuchos::ParameterList parameters;

  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",mu);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", true);

  parameters.set("includeVelocityTracesInFluxTerm",includeVelocityTracesInFluxTerm); // a bit easier to visualize traces when false (when true, tn in space and uhat in time get lumped together, and these can have fairly different scales)
  parameters.set("t0",0.0);

  return StokesVGPFormulation(parameters);
}

StokesVGPFormulation StokesVGPFormulation::spaceTimeFormulationStrongConservation(int spaceDim, double mu, bool useConformingTraces,
                                                                                  bool includeVelocityTracesInFluxTerm)
{
  Teuchos::ParameterList parameters;

  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",mu);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", false);
  parameters.set("useSpaceTime", true);
  parameters.set("useStrongConservation",true);

  parameters.set("includeVelocityTracesInFluxTerm",includeVelocityTracesInFluxTerm); // a bit easier to visualize traces when false (when true, tn in space and uhat in time get lumped together, and these can have fairly different scales)
  parameters.set("t0",0.0);

  return StokesVGPFormulation(parameters);
}


StokesVGPFormulation StokesVGPFormulation::timeSteppingFormulation(int spaceDim, double mu, double dt,
                                                                   bool useConformingTraces, TimeStepType timeStepType)
{
  Teuchos::ParameterList parameters;

  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",mu);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", true);
  parameters.set("useSpaceTime", false);
  parameters.set("dt", dt);
  parameters.set("timeStepType", timeStepType);

  return StokesVGPFormulation(parameters);
}

StokesVGPFormulation StokesVGPFormulation::timeSteppingFormulationStrongConservation(int spaceDim, double mu, double dt,
                                                                                    bool useConformingTraces,
                                                                                    TimeStepType timeStepType)
{
  Teuchos::ParameterList parameters;

  parameters.set("spaceDim", spaceDim);
  parameters.set("mu",mu);
  parameters.set("useConformingTraces",useConformingTraces);
  parameters.set("useTimeStepping", true);
  parameters.set("useSpaceTime", false);
  parameters.set("dt", dt);
  parameters.set("timeStepType", timeStepType);

  parameters.set("useStrongConservation",true);

  return StokesVGPFormulation(parameters);
}


StokesVGPFormulation::StokesVGPFormulation(Teuchos::ParameterList &parameters)
{
  // basic parameters
  int spaceDim = parameters.get<int>("spaceDim");
  double mu = parameters.get<double>("mu",1.0);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces",false);

  // time-related parameters:
  bool useTimeStepping = parameters.get<bool>("useTimeStepping",false);
  double dt = parameters.get<double>("dt",1.0);
  bool useSpaceTime = parameters.get<bool>("useSpaceTime",false);
  TimeStepType timeStepType = parameters.get<TimeStepType>("timeStepType", BACKWARD_EULER); // Backward Euler is immune to oscillations (which Crank-Nicolson can/does exhibit)

  _spaceDim = spaceDim;
  _useConformingTraces = useConformingTraces;
  _mu = mu;
  _dt = ParameterFunction::parameterFunction(dt);
  _t = ParameterFunction::parameterFunction(0);
  _includeVelocityTracesInFluxTerm = parameters.get<bool>("includeVelocityTracesInFluxTerm",true);
  _t0 = parameters.get<double>("t0",0);

  _useStrongConservation = parameters.get<bool>("useStrongConservation",false);

  if (_useStrongConservation)
  {
    int rank = Teuchos::GlobalMPISession::getRank();
    if (rank==0)
      cout << "WARNING: using 'useStrongConservation' option in StokesVGPFormulation.  This is experimental; use with care!\n";
  }

  double thetaValue;
  switch (timeStepType) {
    case FORWARD_EULER:
      thetaValue = 0.0;
      break;
    case CRANK_NICOLSON:
      thetaValue = 0.5;
      break;
    case BACKWARD_EULER:
      thetaValue = 1.0;
      break;
  }

  _theta = ParameterFunction::parameterFunction(thetaValue);
  _timeStepping = useTimeStepping;
  _spaceTime = useSpaceTime;

  if ((spaceDim != 2) && (spaceDim != 3))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim must be 2 or 3");
  }

  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr u1, u2, u3;
  VarPtr p;
  VarPtr sigma11, sigma12, sigma13, sigma21, sigma22, sigma23, sigma31, sigma32, sigma33;

  // traces
  VarPtr u1_hat, u2_hat, u3_hat;
  VarPtr t1n, t2n, t3n;

  // tests
  VarPtr v1, v2, v3;
  VarPtr tau1, tau2, tau3;
  VarPtr q;

  _vf = VarFactory::varFactory();
  u1 = _vf->fieldVar(S_U1);
  u2 = _vf->fieldVar(S_U2);
  if (spaceDim==3) u3 = _vf->fieldVar(S_U3);

  vector<VarPtr> u(spaceDim);
  u[0] = u1;
  u[1] = u2;
  if (spaceDim==3) u[2] = u3;

  p = _vf->fieldVar(S_P);

  vector<vector<LinearTermPtr>> sigma(spaceDim,vector<LinearTermPtr>(spaceDim));
  sigma11 = _vf->fieldVar(S_SIGMA11);
  sigma12 = _vf->fieldVar(S_SIGMA12);
  sigma21 = _vf->fieldVar(S_SIGMA21);
  sigma[0][0] = 1.0 * sigma11;
  sigma[0][1] = 1.0 * sigma12;
  sigma[1][0] = 1.0 * sigma21;
  if (!_useStrongConservation || (spaceDim == 3))
  {
    sigma22 = _vf->fieldVar(S_SIGMA22);
    sigma[1][1] = 1.0 * sigma22;
  }
  else
    sigma[1][1] = - sigma11;
  if (spaceDim==3)
  {
    sigma13 = _vf->fieldVar(S_SIGMA13);
    sigma23 = _vf->fieldVar(S_SIGMA23);
    sigma31 = _vf->fieldVar(S_SIGMA31);
    sigma32 = _vf->fieldVar(S_SIGMA32);
    sigma[0][2] = 1.0 * sigma13;
    sigma[1][2] = 1.0 * sigma23;
    sigma[2][0] = 1.0 * sigma31;
    sigma[2][1] = 1.0 * sigma32;
    if (!_useStrongConservation)
    {
      sigma33 = _vf->fieldVar(S_SIGMA33);
      sigma[2][2] = 1.0 * sigma33;
    }
    else
      sigma[2][2] = - sigma11 - sigma22;
  }

  FunctionPtr one = Function::constant(1.0); // reuse Function to take advantage of accelerated BasisReconciliation (probably this is not the cleanest way to do this, but it should work)
  if (! _spaceTime)
  {
    Space uHatSpace = useConformingTraces ? HGRAD : L2;
    if (spaceDim > 0) u1_hat = _vf->traceVar(S_U1_HAT, one * u1, uHatSpace);
    if (spaceDim > 1) u2_hat = _vf->traceVar(S_U2_HAT, one * u2, uHatSpace);
    if (spaceDim > 2) u3_hat = _vf->traceVar(S_U3_HAT, one * u3, uHatSpace);
  }
  else
  {
    if (_includeVelocityTracesInFluxTerm)
    {
      Space uHatSpace = useConformingTraces ? HGRAD_SPACE_L2_TIME : L2;
      if (spaceDim > 0) u1_hat = _vf->traceVarSpaceOnly(S_U1_HAT, one * u1, uHatSpace);
      if (spaceDim > 1) u2_hat = _vf->traceVarSpaceOnly(S_U2_HAT, one * u2, uHatSpace);
      if (spaceDim > 2) u3_hat = _vf->traceVarSpaceOnly(S_U3_HAT, one * u3, uHatSpace);
    }
    else
    {
      Space uHatSpace = useConformingTraces ? HGRAD : L2;
      if (spaceDim > 0) u1_hat = _vf->traceVar(S_U1_HAT, one * u1, uHatSpace);
      if (spaceDim > 1) u2_hat = _vf->traceVar(S_U2_HAT, one * u2, uHatSpace);
      if (spaceDim > 2) u3_hat = _vf->traceVar(S_U3_HAT, one * u3, uHatSpace);
    }
  }

  vector<VarPtr> u_hat(spaceDim);
  u_hat[0] = u1_hat;
  u_hat[1] = u2_hat;
  if (spaceDim==3) u_hat[2] = u3_hat;

  TFunctionPtr<double> n = TFunction<double>::normal();
  TFunctionPtr<double> n_parity = n * TFunction<double>::sideParity();

  vector<LinearTermPtr> tn_lt(spaceDim);

//  LinearTermPtr t1n_lt, t2n_lt, t3n_lt;
  FunctionPtr minus_n_parity = - n_parity; // reuse Function to take advantage of accelerated BasisReconciliation (probably this is not the cleanest way to do this, but it should work)

  for (int d1=1; d1<=spaceDim; d1++)
  {
    tn_lt[d1-1] = p * n_parity->spatialComponent(d1);
    for (int d2=1; d2<=spaceDim; d2++)
    {
      tn_lt[d1-1] = tn_lt[d1-1] + sigma[d1-1][d2-1] * minus_n_parity->spatialComponent(d2);
    }
  }

  if (_spaceTime && _includeVelocityTracesInFluxTerm)
  {
    TFunctionPtr<double> n_spaceTime = TFunction<double>::normalSpaceTime();
    TFunctionPtr<double> n_t_parity = n_spaceTime->t() * TFunction<double>::sideParity();

    for (int d=1; d<=spaceDim; d++)
    {
      tn_lt[d-1] = tn_lt[d-1] + u[d-1] * n_t_parity;
    }
  }

  vector<string> fluxStrings = {S_TN1_HAT, S_TN2_HAT, S_TN3_HAT};

  if (_spaceTime)
  {
    if (_includeVelocityTracesInFluxTerm)
    {
      t1n = _vf->fluxVar(S_TN1_HAT, tn_lt[0]);
      t2n = _vf->fluxVar(S_TN2_HAT, tn_lt[1]);
      if (spaceDim == 3) t3n = _vf->fluxVar(S_TN3_HAT, tn_lt[2]);
    }
    else
    {
      t1n = _vf->fluxVarSpaceOnly(S_TN1_HAT, tn_lt[0]);
      t2n = _vf->fluxVarSpaceOnly(S_TN2_HAT, tn_lt[1]);
      if (spaceDim == 3) t3n = _vf->fluxVarSpaceOnly(S_TN3_HAT, tn_lt[2]);
    }
  }
  else
  {
    t1n = _vf->fluxVar(S_TN1_HAT, tn_lt[0]);
    t2n = _vf->fluxVar(S_TN2_HAT, tn_lt[1]);
    if (spaceDim == 3) t3n = _vf->fluxVar(S_TN3_HAT, tn_lt[2]);
  }

  v1 = _vf->testVar(S_V1, HGRAD);
  v2 = _vf->testVar(S_V2, HGRAD);
  if (spaceDim == 3) v3 = _vf->testVar(S_V3, HGRAD);

  tau1 = _vf->testVar(S_TAU1, HDIV);
  tau2 = _vf->testVar(S_TAU2, HDIV);
  if (spaceDim == 3) tau3 = _vf->testVar(S_TAU3, HDIV);

  // now that we have all our variables defined, process any adjustments
  map<int,VarPtr> trialVars = _vf->trialVars();
  for (auto entry : trialVars)
  {
    VarPtr var = entry.second;
    string lookupString = var->name() + "-polyOrderAdjustment";
    int adjustment = parameters.get<int>(lookupString,0);
    if (adjustment != 0)
    {
      _trialVariablePolyOrderAdjustments[var->ID()] = adjustment;
    }
  }

  _steadyStokesBF = Teuchos::rcp( new BF(_vf) );

  vector<VarPtr> tau = {tau1,tau2,tau3};
  for (int d1=1; d1<=spaceDim; d1++)
  {
    _steadyStokesBF->addTerm(u[d1-1], tau[d1-1]->div()); // (1/mu) * sigma_1 = grad u_1
    for (int d2=1; d2<=spaceDim; d2++)
    {
      _steadyStokesBF->addTerm(Function::constant(1.0 / _mu) * sigma[d1-1][d2-1], tau[d1-1]->spatialComponent(d2)); // (sigma_1, tau_1)
    }
    _steadyStokesBF->addTerm(-u_hat[d1-1], tau[d1-1] * n);
  }

  vector<VarPtr> v = {v1,v2,v3};
  vector<VarPtr> tn = {t1n,t2n,t3n};
  for (int d1=1; d1<=spaceDim; d1++)
  {
    for (int d2=1; d2<=spaceDim; d2++)
    {
      _steadyStokesBF->addTerm(sigma[d1-1][d2-1], v[d1-1]->di(d2)); // (sigma1, grad v1)
    }
    _steadyStokesBF->addTerm( - p, v[d1-1]->di(d1) );
    _steadyStokesBF->addTerm( tn[d1-1], v[d1-1] );
  }

  // q:
  bool skipWeakConservationIfUsingStrong = false; // we can save costs by setting this to true, but "strong" conservation only weakly applies to the velocity fields anyway, and we increase the strength of that effect by including this equation.  So if you're interested in local conservation, better to include the q terms.
  if (!_useStrongConservation || !skipWeakConservationIfUsingStrong)
  {
    q = _vf->testVar(S_Q, HGRAD);

    for (int d1=1; d1<=spaceDim; d1++)
    {
      _steadyStokesBF->addTerm(-u[d1-1],q->di(d1)); // (-u, grad q)
      _steadyStokesBF->addTerm(u_hat[d1-1] * n->spatialComponent(d1), q);
    }
  }

  if (!_spaceTime && !_timeStepping)
  {
    _stokesBF = _steadyStokesBF;
  }
  else if (_timeStepping)
  {
    _stokesBF = Teuchos::rcp( new BF(_vf) );

    for (int d1=1; d1<=spaceDim; d1++)
    {
      _stokesBF->addTerm(_theta * u[d1-1], tau[d1-1]->div());
      _stokesBF->addTerm(-u_hat[d1-1], tau[d1-1]->dot_normal());
      for (int d2=1; d2<=spaceDim; d2++)
      {
        FunctionPtr theta_mu_weight = ((FunctionPtr)_theta) / _mu;
        _stokesBF->addTerm(theta_mu_weight * sigma[d1-1][d2-1], tau[d1-1]->spatialComponent(d2)); // (sigma_d1, tau_d1)
      }
    }

    TFunctionPtr<double> dtFxn = _dt; // cast to allow use of TFunctionPtr<double> operator overloads
    TFunctionPtr<double> thetaFxn = _theta;

    for (int d1=1; d1<=spaceDim; d1++)
    {
      _stokesBF->addTerm((1.0 / dtFxn) * u[d1-1], v[d1-1]);
      _stokesBF->addTerm(-thetaFxn * p, v[d1-1]->di(d1) );
      _stokesBF->addTerm( tn[d1-1], v[d1-1] );

      for (int d2=1; d2<=spaceDim; d2++)
      {
        _stokesBF->addTerm(thetaFxn * sigma[d1-1][d2-1],v[d1-1]->di(d2)); // (sigma1, grad v1)
      }
    }

    if (!_useStrongConservation || !skipWeakConservationIfUsingStrong)
    {
      q = _vf->testVar(S_Q, HGRAD);

      for (int d1=1; d1<=spaceDim; d1++)
      {
        _steadyStokesBF->addTerm(_theta * (-u[d1-1]),q->di(d1)); // (-u, grad q)
        _steadyStokesBF->addTerm(u_hat[d1-1] * n->spatialComponent(d1), q);
      }
    }
  }
  else if (_spaceTime)
  {
    _stokesBF = Teuchos::rcp( new BF(*_steadyStokesBF) );

    TFunctionPtr<double> n_spaceTime = TFunction<double>::normalSpaceTime();

    for (int d1=1; d1<=spaceDim; d1++)
    {
      _stokesBF->addTerm(-u[d1-1], v[d1-1]->dt());
      if (!_includeVelocityTracesInFluxTerm)
      {
        _stokesBF->addTerm(u_hat[d1-1] * n_spaceTime->t(), v[d1-1]);
      }
    }
  }

  // define tractions (used in outflow conditions)
  // definition of traction: _mu * ( (\nabla u) + (\nabla u)^T ) n - p n
  //                      = (sigma + sigma^T) n - p n
  vector<LinearTermPtr> tractions(spaceDim);
  for (int d1=1; d1<=spaceDim; d1++)
  {
    tractions[d1-1] = - p * n->spatialComponent(d1);
    for (int d2=1; d2<=spaceDim; d2++)
    {
      tractions[d1-1] = tractions[d1-1] + sigma[d1-1][d2-1] * n->spatialComponent(d2);
      tractions[d1-1] = tractions[d1-1] + sigma[d2-1][d1-1] * n->spatialComponent(d2);
    }
  }
  _t1 = tractions[0];
  _t2 = tractions[1];
  if (_spaceDim == 3) _t3 = tractions[2];
}

// ! returns the sum of the absolute value of the mass flux through each element
double StokesVGPFormulation::absoluteMassFlux() const
{
  return computeMassFlux(true);
}

double StokesVGPFormulation::computeMassFlux(bool computeAbsoluteValuesOnEachCell) const
{
  vector<FunctionPtr> uhat_soln_vector(_spaceDim);
  for (int d1=1; d1<=_spaceDim; d1++)
  {
    uhat_soln_vector[d1-1] = Function::solution(this->u_hat(d1), this->solution());
  }

  FunctionPtr uhat_soln = Function::vectorize(uhat_soln_vector);

  FunctionPtr n = Function::normal();

  FunctionPtr massFlux = uhat_soln * n;

  MeshPtr mesh = _solution->mesh();

  double sumMassFlux = 0;
  set<GlobalIndexType> cellIDs = mesh->cellIDsInPartition();
  for (GlobalIndexType cellID : cellIDs)
  {
    bool testVsTest = false;
    int cubatureDegreeEnrichment = 0;
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest, cubatureDegreeEnrichment);
    ElementTypePtr elemType = mesh->getElementType(cellID);
    int numSides = elemType->cellTopoPtr->getSideCount();

    double cellIntegral = 0;
    for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
    {
      double sideIntegral = massFlux->integrate(basisCache->getSideBasisCache(sideOrdinal));
      cellIntegral += sideIntegral;
    }
    sumMassFlux += computeAbsoluteValuesOnEachCell ? abs(cellIntegral) : cellIntegral;
  }
  return MPIWrapper::sum(sumMassFlux);
}

void StokesVGPFormulation::addInflowCondition(SpatialFilterPtr inflowRegion, TFunctionPtr<double> u)
{
  VarPtr u1_hat = this->u_hat(1), u2_hat = this->u_hat(2);
  VarPtr u3_hat;
  if (_spaceDim==3) u3_hat = this->u_hat(3);

  if (! _timeStepping)
  {
    _solution->bc()->addDirichlet(u1_hat, inflowRegion, u->x());
    _solution->bc()->addDirichlet(u2_hat, inflowRegion, u->y());
    if (_spaceDim==3) _solution->bc()->addDirichlet(u3_hat, inflowRegion, u->z());
  }
  else // _timeStepping
  {
    TFunctionPtr<double> u1_hat_prev, u2_hat_prev, u3_hat_prev;
    TSolutionPtr<double> prevSolnWeakRef = Teuchos::rcp( _previousSolution.get(), false ); // avoid circular references

    u1_hat_prev = TFunction<double>::solution(this->u_hat(1), prevSolnWeakRef);
    u2_hat_prev = TFunction<double>::solution(this->u_hat(2), prevSolnWeakRef);
    if (_spaceDim==3) u3_hat_prev = TFunction<double>::solution(this->u_hat(3), prevSolnWeakRef);
    TFunctionPtr<double> thetaFxn = _theta; // cast to allow use of TFunctionPtr<double> operator overloads
    _solution->bc()->addDirichlet(u1_hat, inflowRegion, thetaFxn * u->x() + (1.-thetaFxn)* u1_hat_prev);
    _solution->bc()->addDirichlet(u2_hat, inflowRegion, thetaFxn * u->y() + (1.-thetaFxn)* u2_hat_prev);
    if (_spaceDim==3) _solution->bc()->addDirichlet(u3_hat, inflowRegion, thetaFxn * u->z() + (1.-thetaFxn)* u3_hat_prev);
  }
}

void StokesVGPFormulation::addOutflowCondition(SpatialFilterPtr outflowRegion, bool usePhysicalTractions)
{
//  for (int d=0; d<spaceDim; d++) {
//    VarPtr tn_hat = this->tn_hat(d+1);
//    _solution->bc()->addDirichlet(tn_hat, outflowRegion, TFunction<double>::zero());
//  }

  _haveOutflowConditionsImposed = true;

  // point pressure and zero-mean pressures are not compatible with outflow conditions:
  VarPtr p = this->p();
  if (_solution->bc()->shouldImposeZeroMeanConstraint(p->ID()))
  {
    cout << "Removing zero-mean constraint on pressure by virtue of outflow condition.\n";
    _solution->bc()->removeZeroMeanConstraint(p->ID());
  }

  if (_solution->bc()->singlePointBC(p->ID()))
  {
    cout << "Removing zero-point condition on pressure by virtue of outflow condition.\n";
    _solution->bc()->removeSinglePointBC(p->ID());
  }

  if (usePhysicalTractions)
  {
    // my favorite way to do outflow conditions is via penalty constraints imposing a zero traction
    Teuchos::RCP<LocalStiffnessMatrixFilter> filter_incr = _solution->filter();

    Teuchos::RCP< PenaltyConstraints > pcRCP;
    PenaltyConstraints* pc;

    if (filter_incr.get() != NULL)
    {
      pc = dynamic_cast<PenaltyConstraints*>(filter_incr.get());
      if (pc == NULL)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't add PenaltyConstraints when a non-PenaltyConstraints LocalStiffnessMatrixFilter already in place");
      }
    }
    else
    {
      pcRCP = Teuchos::rcp( new PenaltyConstraints );
      pc = pcRCP.get();
    }
    TFunctionPtr<double> zero = TFunction<double>::zero();
    pc->addConstraint(_t1==zero, outflowRegion);
    pc->addConstraint(_t2==zero, outflowRegion);
    if (_spaceDim==3) pc->addConstraint(_t3==zero, outflowRegion);

    if (pcRCP != Teuchos::null)   // i.e., we're not just adding to a prior PenaltyConstraints object
    {
      _solution->setFilter(pcRCP);
    }
  }
  else
  {
    TFunctionPtr<double> zero = TFunction<double>::zero();
    for (int d=1; d<=_spaceDim; d++)
    {
      _solution->bc()->addDirichlet(tn_hat(d), outflowRegion, zero);
    }
  }
}

void StokesVGPFormulation::addPointPressureCondition(vector<double> vertex)
{
  if (_haveOutflowConditionsImposed)
  {
    cout << "ERROR: can't add pressure point condition if there are outflow conditions imposed.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "");
  }

  VarPtr p = this->p();

  if (vertex.size() == 0)
  {
    vertex = _solution->mesh()->getTopology()->getVertex(0);
    if (_spaceTime) // then the last coordinate is time; drop it
    {
      vertex.pop_back();
    }
  }
  _solution->bc()->addSpatialPointBC(p->ID(), 0.0, vertex);

//  cout << "setting point pressure condition at point (";
//  for (int d=0; d<vertex.size(); d++)
//  {
//    cout << vertex[d];
//    if (d < vertex.size() - 1) cout << ", ";
//  }
//  cout << ")\n";

  if (_solution->bc()->shouldImposeZeroMeanConstraint(p->ID()))
  {
    _solution->bc()->removeZeroMeanConstraint(p->ID());
  }
}

void StokesVGPFormulation::addWallCondition(SpatialFilterPtr wall)
{
  vector<double> zero(_spaceDim, 0.0);
  addInflowCondition(wall, TFunction<double>::constant(zero));
}

void StokesVGPFormulation::addInitialCondition(double t0, vector<FunctionPtr> u0, FunctionPtr p0)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!_spaceTime, std::invalid_argument, "This method only supported for space-time formulations");
  TEUCHOS_TEST_FOR_EXCEPTION(u0.size() != _spaceDim, std::invalid_argument, "u0 should have length equal to the number of spatial dimensions");

  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(_solution->mesh()->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(!meshTopo, std::invalid_argument, "For the present, StokesFormulation only supports true MeshTopologies for its Solution object");
  meshTopo->applyTag(DIRICHLET_SET_TAG_NAME, INITIAL_CONDITION_TAG, meshTopo->getEntitySetInitialTime());

  for (int d=1; d<=_spaceDim; d++)
  {
    VarPtr var;
    FunctionPtr functionToImpose;
    if (!_includeVelocityTracesInFluxTerm)
    {
      var = this->u_hat(d);
      functionToImpose = u0[d-1];
    }
    else
    {
      var = this->tn_hat(d);
      FunctionPtr n_t = Function::normalSpaceTime()->t();  // under usual circumstances, n_t = -1
      functionToImpose = u0[d-1] * n_t;
    }
    _solution->bc()->addDirichlet(var, INITIAL_CONDITION_TAG, functionToImpose);
  }

  if (p0 != Teuchos::null)
  {
    MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(_solution->mesh()->getTopology().get());
    TEUCHOS_TEST_FOR_EXCEPTION(!meshTopo, std::invalid_argument, "For the present, StokesFormulation only supports true MeshTopologies for its Solution object");
    meshTopo->applyTag(DIRICHLET_SET_TAG_NAME, INITIAL_CONDITION_TAG, meshTopo->getEntitySetInitialTime());
    _solution->bc()->addDirichlet(p(), INITIAL_CONDITION_TAG, p0);
  }
}

void StokesVGPFormulation::addZeroInitialCondition(double t0)
{
  vector<FunctionPtr> zero(_spaceDim, Function::zero());
  addInitialCondition(t0, zero, Teuchos::null); // null: don't impose an initial condition on the pressure.
}

void StokesVGPFormulation::addZeroMeanPressureCondition()
{
  if (_spaceTime)
  {
    cout << "zero-mean constraints for pressure not yet supported for space-time.  Use point constraints instead.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "zero-mean constraints for pressure not yet supported for space-time.  Use point constraints instead.");
  }
  if (_haveOutflowConditionsImposed)
  {
    cout << "ERROR: can't add zero mean pressure condition if there are outflow conditions imposed.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "");
  }

  VarPtr p = this->p();

  _solution->bc()->addZeroMeanConstraint(p);

  if (_solution->bc()->singlePointBC(p->ID()))
  {
    _solution->bc()->removeSinglePointBC(p->ID());
  }
}

BFPtr StokesVGPFormulation::bf()
{
  return _stokesBF;
}

void StokesVGPFormulation::CHECK_VALID_COMPONENT(int i) const // throws exception on bad component value (should be between 1 and _spaceDim, inclusive)
{
  if ((i > _spaceDim) || (i < 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component indices must be at least 1 and less than or equal to _spaceDim");
  }
}

TFunctionPtr<double> StokesVGPFormulation::forcingFunction(TFunctionPtr<double> u_exact, TFunctionPtr<double> p_exact)
{
  // f1 and f2 are those for Navier-Stokes, but without the u \cdot \grad u term
  TFunctionPtr<double> u1_exact = u_exact->x();
  TFunctionPtr<double> u2_exact = u_exact->y();
  TFunctionPtr<double> u3_exact = u_exact->z();

  TFunctionPtr<double> f;

  if (_spaceDim == 2)
  {
    TFunctionPtr<double> f1, f2;
    f1 = p_exact->dx() - _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
    f2 = p_exact->dy() - _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
    if (_spaceTime)
    {
      f1 = f1 + u1_exact->dt();
      f2 = f2 + u2_exact->dt();
    }
    else if (_timeStepping)
    {
      // leave as is, for now.  The rhs() method will add some terms; this may be the right thing.
    }
    f = TFunction<double>::vectorize(f1, f2);
  }
  else
  {
    TFunctionPtr<double> f1, f2, f3;
    f1 = p_exact->dx() - _mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy() + u1_exact->dz()->dz());
    f2 = p_exact->dy() - _mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy() + u2_exact->dz()->dz());
    f3 = p_exact->dz() - _mu * (u3_exact->dx()->dx() + u3_exact->dy()->dy() + u3_exact->dz()->dz());
    if (_spaceTime)
    {
      f1 = f1 + u1_exact->dt();
      f2 = f2 + u2_exact->dt();
      f3 = f3 + u3_exact->dt();
    }
    else if (_timeStepping)
    {
      // leave as is, for now.  The rhs() method will add some terms; this may be the right thing.
    }

    f = TFunction<double>::vectorize(f1, f2, f3);
  }
  return f;
}

void StokesVGPFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
    TFunctionPtr<double> forcingFunction, int temporalPolyOrder)
{
  this->initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction, "", temporalPolyOrder);
}

void StokesVGPFormulation::initializeSolution(std::string filePrefix, int fieldPolyOrder, int delta_k,
    TFunctionPtr<double> forcingFunction, int temporalPolyOrder)
{
  this->initializeSolution(Teuchos::null, fieldPolyOrder, delta_k, forcingFunction, filePrefix, temporalPolyOrder);
}

void StokesVGPFormulation::initializeSolution(MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k,
    TFunctionPtr<double> forcingFunction, string savedSolutionAndMeshPrefix, int temporalPolyOrder)
{
  _haveOutflowConditionsImposed = false;
  BCPtr bc = BC::bc();

  vector<int> H1Order {fieldPolyOrder + 1};
  MeshPtr mesh;
  if (savedSolutionAndMeshPrefix == "")
  {
    if (_spaceTime) H1Order.push_back(temporalPolyOrder); // "H1Order" is a bit misleading for space-time; in fact in BasisFactory we ensure that the polynomial order in time is whatever goes in this slot, regardless of function space.  This is disanalogous to what happens in space, so we might want to revisit that at some point.
    mesh = Teuchos::rcp( new Mesh(meshTopo, _stokesBF, H1Order, delta_k, _trialVariablePolyOrderAdjustments) ) ;
    _solution = TSolution<double>::solution(mesh,bc);
    if (_timeStepping) _previousSolution = TSolution<double>::solution(mesh,bc);
  }
  else
  {
    mesh = MeshFactory::loadFromHDF5(_stokesBF, savedSolutionAndMeshPrefix+".mesh");
    _solution = TSolution<double>::solution(mesh, bc);
    _solution->loadFromHDF5(savedSolutionAndMeshPrefix+".soln");
    if (_timeStepping)
    {
      _previousSolution = TSolution<double>::solution(mesh,bc);
      _previousSolution->loadFromHDF5(savedSolutionAndMeshPrefix+"_previous.soln");
    }
  }

  RHSPtr rhs = this->rhs(forcingFunction); // in transient case, this will refer to _previousSolution
  IPPtr ip = _stokesBF->graphNorm();

//  cout << "graph norm for Stokes BF:\n";
//  ip->printInteractions();

  _solution->setRHS(rhs);
  _solution->setIP(ip);

  mesh->registerSolution(_solution); // will project both time steps during refinements...
  if (_timeStepping) mesh->registerSolution(_previousSolution);

  LinearTermPtr residual = rhs->linearTerm() - _stokesBF->testFunctional(_solution,false); // false: don't exclude boundary terms

  double energyThreshold = 0.2;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold ) );

  double maxDouble = std::numeric_limits<double>::max();
  double maxP = 20;
  _hRefinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold, 0, 0, false ) );
  _pRefinementStrategy = Teuchos::rcp( new RefinementStrategy( mesh, residual, ip, energyThreshold, maxDouble, maxP, true ) );

  _time = 0;
  _t->setTime(_time);

  if (_spaceDim==2)
  {
    // finally, set up a stream function solve for 2D
    _streamFormulation = Teuchos::rcp( new PoissonFormulation(_spaceDim,_useConformingTraces) );

    MeshPtr streamMesh;
    if (savedSolutionAndMeshPrefix == "")
    {
      MeshTopologyPtr streamMeshTopo = meshTopo->deepCopy();
      streamMesh = Teuchos::rcp( new Mesh(streamMeshTopo, _streamFormulation->bf(), H1Order, delta_k) ) ;
    }
    else
    {
      streamMesh = MeshFactory::loadFromHDF5(_streamFormulation->bf(), savedSolutionAndMeshPrefix+"_stream.mesh");
    }

    mesh->registerObserver(streamMesh); // refine streamMesh whenever mesh is refined

    LinearTermPtr u1_dy = (1.0 / _mu) * this->sigma(1,2);
    LinearTermPtr u2_dx = (1.0 / _mu) * this->sigma(2,1);

    TFunctionPtr<double> vorticity = Teuchos::rcp( new PreviousSolutionFunction<double>(_solution, u2_dx - u1_dy) );
    RHSPtr streamRHS = RHS::rhs();
    VarPtr q_stream = _streamFormulation->q();
    streamRHS->addTerm( -vorticity * q_stream );
    bool dontWarnAboutOverriding = true;
    ((PreviousSolutionFunction<double>*) vorticity.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);

    /* Stream function phi is such that
     *    d/dx phi = -u2
     *    d/dy phi =  u1
     * Therefore, psi = grad phi = (-u2, u1), and psi * n = u1 n2 - u2 n1
     */

    TFunctionPtr<double> u1_soln = Teuchos::rcp( new PreviousSolutionFunction<double>(_solution, this->u(1) ) );
    TFunctionPtr<double> u2_soln = Teuchos::rcp( new PreviousSolutionFunction<double>(_solution, this->u(2) ) );
    ((PreviousSolutionFunction<double>*) u1_soln.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);
    ((PreviousSolutionFunction<double>*) u2_soln.get())->setOverrideMeshCheck(true,dontWarnAboutOverriding);

    TFunctionPtr<double> n = TFunction<double>::normal();

    BCPtr streamBC = BC::bc();
    VarPtr phi = _streamFormulation->phi();
    streamBC->addZeroMeanConstraint(phi);

    VarPtr psi_n = _streamFormulation->psi_n_hat();
    streamBC->addDirichlet(psi_n, SpatialFilter::allSpace(), u1_soln * n->y() - u2_soln * n->x());

    IPPtr streamIP = _streamFormulation->bf()->graphNorm();
    _streamSolution = TSolution<double>::solution(streamMesh,streamBC,streamRHS,streamIP);

    if (savedSolutionAndMeshPrefix != "")
    {
      _streamSolution->loadFromHDF5(savedSolutionAndMeshPrefix + "_stream.soln");
    }
  }
}

bool StokesVGPFormulation::isSpaceTime() const
{
  return _spaceTime;
}

bool StokesVGPFormulation::isSteady() const
{
  return !_timeStepping && !_spaceTime;
}


bool StokesVGPFormulation::isTimeStepping() const
{
  return _timeStepping;
}

double StokesVGPFormulation::relativeL2NormOfTimeStep()
{
  TFunctionPtr<double>  p_current = TFunction<double>::solution( p(), _solution);
  TFunctionPtr<double> u1_current = TFunction<double>::solution(u(1), _solution);
  TFunctionPtr<double> u2_current = TFunction<double>::solution(u(2), _solution);
  TFunctionPtr<double>  p_prev = TFunction<double>::solution( p(), _previousSolution);
  TFunctionPtr<double> u1_prev = TFunction<double>::solution(u(1), _previousSolution);
  TFunctionPtr<double> u2_prev = TFunction<double>::solution(u(2), _previousSolution);

  TFunctionPtr<double> squaredSum = (p_current+p_prev) * (p_current+p_prev) + (u1_current+u1_prev) * (u1_current+u1_prev) + (u2_current + u2_prev) * (u2_current + u2_prev);
  // average would be each summand divided by 4
  double L2OfAverage = sqrt( 0.25 * squaredSum->integrate(_solution->mesh()));

  TFunctionPtr<double> squaredDiff = (p_current-p_prev) * (p_current-p_prev) + (u1_current-u1_prev) * (u1_current-u1_prev) + (u2_current - u2_prev) * (u2_current - u2_prev);

  double valSquared = squaredDiff->integrate(_solution->mesh());
  if (L2OfAverage < 1e-15) return sqrt(valSquared);

  return sqrt(valSquared) / L2OfAverage;
}

double StokesVGPFormulation::mu()
{
  return _mu;
}

// ! returns the sum of the signed mass flux through each element
double StokesVGPFormulation::netMassFlux() const
{
  return computeMassFlux(false);
}

VarPtr StokesVGPFormulation::p() const
{
  return _vf->fieldVar(S_P);
}

RefinementStrategyPtr StokesVGPFormulation::getRefinementStrategy()
{
  return _refinementStrategy;
}

void StokesVGPFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void StokesVGPFormulation::refine()
{
  _refinementStrategy->refine();
}

void StokesVGPFormulation::hRefine()
{
  _hRefinementStrategy->refine();
}

void StokesVGPFormulation::pRefine()
{
  _pRefinementStrategy->refine();
}

RHSPtr StokesVGPFormulation::rhs(TFunctionPtr<double> f)
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

  if (_timeStepping)
  {
    TFunctionPtr<double> u1_prev, u2_prev, u3_prev;
    u1_prev = TFunction<double>::solution(this->u(1), _previousSolution);
    u2_prev = TFunction<double>::solution(this->u(2), _previousSolution);
    if (_spaceDim==3) u3_prev = TFunction<double>::solution(this->u(3), _previousSolution);
    TFunctionPtr<double> dtFxn = _dt; // cast to allow use of TFunctionPtr<double> operator overloads
    rhs->addTerm(u1_prev / dtFxn * v1);
    rhs->addTerm(u2_prev / dtFxn * v2);
    if (_spaceDim==3) rhs->addTerm(u3_prev / dtFxn * v3);

    bool excludeFluxesAndTraces = true;
    LinearTermPtr prevTimeStepFunctional = _steadyStokesBF->testFunctional(_previousSolution,excludeFluxesAndTraces);
    TFunctionPtr<double> thetaFxn = _theta; // cast to allow use of TFunctionPtr<double> operator overloads
    rhs->addTerm((thetaFxn - 1.0) * prevTimeStepFunctional);
  }
  else if (_spaceTime)
  {
    //
  }

  return rhs;
}

VarPtr StokesVGPFormulation::sigma(int i, int j) const
{
  CHECK_VALID_COMPONENT(i);
  CHECK_VALID_COMPONENT(j);
  static const vector<vector<string>> sigmaStrings = {{S_SIGMA11, S_SIGMA12, S_SIGMA13},{S_SIGMA21, S_SIGMA22, S_SIGMA23},{S_SIGMA31, S_SIGMA32, S_SIGMA33}};


  if ((_spaceDim == 2) && _useStrongConservation)
  {
    // then (2,2) arguments not allowed
    TEUCHOS_TEST_FOR_EXCEPTION((i == 2) && (j == 2), std::invalid_argument, "sigma22 not defined when useStrongConservation = true");
  }

  if ((_spaceDim == 3) && _useStrongConservation)
  {
    // then (3,3) arguments not allowed
    TEUCHOS_TEST_FOR_EXCEPTION((i == 3) || (j == 3), std::invalid_argument, "sigma33 not defined when useStrongConservation = true");
  }

  return _vf->fieldVar(sigmaStrings[i-1][j-1]);
}

VarPtr StokesVGPFormulation::u(int i) const
{
  CHECK_VALID_COMPONENT(i);

  static const vector<string> uStrings = {S_U1,S_U2,S_U3};
  return _vf->fieldVar(uStrings[i-1]);
}

// traces:
VarPtr StokesVGPFormulation::tn_hat(int i) const
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> tnStrings = {S_TN1_HAT,S_TN2_HAT,S_TN3_HAT};
  return _vf->fluxVar(tnStrings[i-1]);
}

VarPtr StokesVGPFormulation::u_hat(int i) const
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> uHatStrings = {S_U1_HAT,S_U2_HAT,S_U3_HAT};
  return _vf->traceVar(uHatStrings[i-1]);
}

// test variables:
VarPtr StokesVGPFormulation::tau(int i) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((i > _spaceDim) || (i < 1), std::invalid_argument, "i must be at least 1 and less than or equal to _spaceDim");
  vector<string> tauStrings = {S_TAU1,S_TAU2,S_TAU3};
  return _vf->testVar(tauStrings[i-1], HDIV);
}

// test variables:
VarPtr StokesVGPFormulation::q() const
{
  return _vf->testVar(S_Q, HGRAD);
}

// ! Saves the solution(s) and mesh to an HDF5 format.
void StokesVGPFormulation::save(std::string prefixString)
{
  _solution->mesh()->saveToHDF5(prefixString+".mesh");
  _solution->saveToHDF5(prefixString+".soln");
  if (_timeStepping)
  {
    _previousSolution->saveToHDF5(prefixString+"_previous.soln");
  }
  if (_streamSolution != Teuchos::null)
  {
    _streamSolution->mesh()->saveToHDF5(prefixString+"_stream.mesh");
    _streamSolution->saveToHDF5(prefixString + "_stream.soln");
  }
}

// ! set current time step used for transient solve
void StokesVGPFormulation::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

// ! Returns the solution (at current time)
TSolutionPtr<double> StokesVGPFormulation::solution() const
{
  return _solution;
}

// ! Returns the solution (at previous time)
TSolutionPtr<double> StokesVGPFormulation::solutionPreviousTimeStep() const
{
  return _previousSolution;
}

// ! Solves
void StokesVGPFormulation::solve()
{
  _solution->solve();
}

// ! Solves iteratively
void StokesVGPFormulation::solveIteratively(int maxIters, double cgTol, int azOutputLevel, bool suppressSuperLUOutput)
{
  int kCoarse = 0;

  bool useCondensedSolve = _solution->usesCondensedSolve();

  vector<MeshPtr> meshes = GMGSolver::meshesForMultigrid(_solution->mesh(), kCoarse, 1);
  vector<MeshPtr> prunedMeshes;
  int minDofCount = 2000; // skip any coarse meshes that have fewer dofs than this
  for (int i=0; i<meshes.size()-2; i++) // leave the last two meshes, so we can guarantee there are at least two
  {
    MeshPtr mesh = meshes[i];
    GlobalIndexType numGlobalDofs;
    if (useCondensedSolve)
      numGlobalDofs = mesh->numFluxDofs(); // this might under-count, in case e.g. of pressure constraints.  But it's meant as a heuristic anyway.
    else
      numGlobalDofs = mesh->numGlobalDofs();

    if (numGlobalDofs > minDofCount)
    {
      prunedMeshes.push_back(mesh);
    }
  }
  prunedMeshes.push_back(meshes[meshes.size()-2]);
  prunedMeshes.push_back(meshes[meshes.size()-1]);

//  prunedMeshes = meshes;

  Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(_solution, prunedMeshes, maxIters, cgTol, GMGOperator::V_CYCLE,
                                                                  Solver::getDirectSolver(true), useCondensedSolve) );
  if (suppressSuperLUOutput)
    turnOffSuperLUDistOutput(gmgSolver);

  gmgSolver->setAztecOutput(azOutputLevel);

  _solution->solve(gmgSolver);
}

int StokesVGPFormulation::spaceDim()
{
  return _spaceDim;
}

PoissonFormulation & StokesVGPFormulation::streamFormulation()
{
  return *_streamFormulation;
}

VarPtr StokesVGPFormulation::streamPhi()
{
  if (_spaceDim == 2)
  {
    if (_streamFormulation == Teuchos::null)
    {
      cout << "ERROR: streamPhi() called before initializeSolution called.  Returning null.\n";
      return Teuchos::null;
    }
    return _streamFormulation->phi();
  }
  else
  {
    cout << "ERROR: stream function is only supported on 2D solutions.  Returning null.\n";
    return Teuchos::null;
  }
}

TSolutionPtr<double> StokesVGPFormulation::streamSolution()
{
  if (_spaceDim == 2)
  {
    if (_streamFormulation == Teuchos::null)
    {
      cout << "ERROR: streamPhi() called before initializeSolution called.  Returning null.\n";
      return Teuchos::null;
    }
    return _streamSolution;
  }
  else
  {
    cout << "ERROR: stream function is only supported on 2D solutions.  Returning null.\n";
    return Teuchos::null;
  }
}

// ! Takes a time step (assumes you have called solve() first)
void StokesVGPFormulation::takeTimeStep()
{
  TEUCHOS_TEST_FOR_EXCEPTION(!_timeStepping, std::invalid_argument, "takeTimeStep() only supported for time-stepping formulation.");
  ConstantScalarFunction<double>* dtValueFxn = dynamic_cast<ConstantScalarFunction<double>*>(_dt->getValue().get());

  double dt = dtValueFxn->value(0);
  _time += dt;
  _t->setValue(_time);

  // if we implemented some sort of value-replacement in Solution, that would be more efficient than this:
  _previousSolution->clear();
  _previousSolution->addSolution(_solution, 1.0);
}

// ! Returns the sum of the time steps taken thus far.
double StokesVGPFormulation::getTime()
{
  return _time;
}

TFunctionPtr<double> StokesVGPFormulation::getTimeFunction()
{
  return _t;
}

void StokesVGPFormulation::turnOffSuperLUDistOutput(Teuchos::RCP<GMGSolver> gmgSolver){
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

VarPtr StokesVGPFormulation::v(int i) const
{
  if ((i > _spaceDim) || (i < 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be at least 1 and less than or equal to _spaceDim");
  }
  const static vector<string> vStrings = {S_V1,S_V2,S_V3};
  return _vf->testVar(vStrings[i-1], HGRAD);
}

LinearTermPtr StokesVGPFormulation::getTraction(int i)
{
  CHECK_VALID_COMPONENT(i);
  switch (i)
  {
    case 1:
      return _t1;
    case 2:
      return _t2;
    case 3:
      return _t3;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

TFunctionPtr<double> StokesVGPFormulation::getPressureSolution()
{
  TFunctionPtr<double> p_soln = Function::solution(p(), _solution);
  return p_soln;
}

const std::map<int,int> & StokesVGPFormulation::getTrialVariablePolyOrderAdjustments()
{
  return _trialVariablePolyOrderAdjustments;
}

TFunctionPtr<double> StokesVGPFormulation::getVelocitySolution()
{
  vector<FunctionPtr> u_components;
  for (int d=1; d<=_spaceDim; d++)
  {
    u_components.push_back(Function::solution(u(d), _solution));
  }
  return Function::vectorize(u_components);
}

TFunctionPtr<double> StokesVGPFormulation::getVorticity()
{
  LinearTermPtr u1_dy = (1.0 / _mu) * this->sigma(1,2);
  LinearTermPtr u2_dx = (1.0 / _mu) * this->sigma(2,1);

  TFunctionPtr<double> vorticity = Teuchos::rcp( new PreviousSolutionFunction<double>(_solution, u2_dx - u1_dy) );
  return vorticity;
}
