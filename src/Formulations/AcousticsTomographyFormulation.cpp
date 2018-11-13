//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  AcousticsTomographyFormulation.cpp
//  Camellia
//
//  Created by Brendan Keith 11/18.
//
//

#include "AcousticsTomographyFormulation.h"

using namespace Camellia;

const string AcousticsTomographyFormulation::S_P1 = "p";
const string AcousticsTomographyFormulation::S_P2 = "i p";
const string AcousticsTomographyFormulation::S_Q1 = "q";
const string AcousticsTomographyFormulation::S_Q2 = "i q";

const string AcousticsTomographyFormulation::S_U1 = "\\vec{u}";
const string AcousticsTomographyFormulation::S_U2 = "i\\vec{u}";
const string AcousticsTomographyFormulation::S_V1 = "\\vec{v}";
const string AcousticsTomographyFormulation::S_V2 = "i\\vec{v}";

const string AcousticsTomographyFormulation::S_P1_HAT = "\\widehat{p}";
const string AcousticsTomographyFormulation::S_P2_HAT = "i\\widehat{p}";
const string AcousticsTomographyFormulation::S_U1_N_HAT = "\\widehat{u}_n";
const string AcousticsTomographyFormulation::S_U2_N_HAT = "i\\widehat{u}_n";

AcousticsTomographyFormulation::AcousticsTomographyFormulation(MeshTopologyPtr meshTopo, Teuchos::ParameterList &parameters)
// AcousticsTomographyFormulation::AcousticsTomographyFormulation(int spaceDim, bool useConformingTraces, AcousticsTomographyFormulationChoice formulationChoice, double angularFrequency)
{
  // basic parameters
  _spaceDim = parameters.get<int>("spaceDim");
  _omega = parameters.get<double>("omega",1.0);
  double slowness = parameters.get<double>("slowness",1.0);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces",false);

  _slowness = ParameterFunction::parameterFunction(slowness);


  Space uSpace = (_spaceDim > 1) ? VECTOR_L2 : L2;
  Space vSpace = (_spaceDim > 1) ? HDIV : HGRAD;
  Space p_hat_space = useConformingTraces ? HGRAD : L2;

  // fields
  VarPtr p1,p2,u1,u2;
  // vector<VarPtr> p(2),u(2);

  // traces
  VarPtr p1_hat,p2_hat,u1_n_hat,u2_n_hat;
  // vector<VarPtr> p_hat(2),u_n_hat(2);

  // tests
  VarPtr q1,q2,v1,v2;
  // vector<VarPtr> q(2),v(2);

  _vf = VarFactory::varFactory();
  p1 = _vf->fieldVar(S_P1);
  p2 = _vf->fieldVar(S_P2);
  u1 = _vf->fieldVar(S_U1, uSpace);
  u2 = _vf->fieldVar(S_U2, uSpace);
  // p[0] = p1; u[0] = u1;
  // p[1] = p2; u[1] = u2;

  TFunctionPtr<double> parity = TFunction<double>::sideParity();
  
  if (_spaceDim > 1)
  {
    p1_hat = _vf->traceVar(S_P1_HAT, p1, p_hat_space);
    p2_hat = _vf->traceVar(S_P2_HAT, p2, p_hat_space);
  }
  else
  {
    p1_hat = _vf->fluxVar(S_P1_HAT, p1 * (Function::normal_1D() * parity), p_hat_space);
    p2_hat = _vf->fluxVar(S_P2_HAT, p2 * (Function::normal_1D() * parity), p_hat_space);
  }
  // p_hat[0] = p1_hat;
  // p_hat[1] = p2_hat;

  TFunctionPtr<double> n = TFunction<double>::normal();

  if (_spaceDim > 1)
  {
    u1_n_hat = _vf->fluxVar(S_U1_N_HAT, u1 * (n * parity));
    u2_n_hat = _vf->fluxVar(S_U2_N_HAT, u2 * (n * parity));
  }
  else
  {
    u1_n_hat = _vf->fluxVar(S_U1_N_HAT, u1 * (Function::normal_1D() * parity)); // for _spaceDim==1, the "normal" component is in the flux-ness of phi_hat (it's a plus or minus 1)
    u2_n_hat = _vf->fluxVar(S_U2_N_HAT, u2 * (Function::normal_1D() * parity));
  }
  // u_n_hat[0] = u1_n_hat;
  // u_n_hat[1] = u2_n_hat;

  q1 = _vf->testVar(S_Q1, HGRAD);
  q2 = _vf->testVar(S_Q2, HGRAD);
  v1 = _vf->testVar(S_V1, vSpace);
  v2 = _vf->testVar(S_V2, vSpace);
  // vector<VarPtr> q(2), v(2);
  // q[0] = q1; v[0] = v1;
  // q[1] = q2; v[1] = v2;

  _acousticsBF = Teuchos::rcp( new BF(_vf) );

  if (_spaceDim==1)
  {
    // for _spaceDim==1, the "normal" component is in the flux-ness of phi_hat (it's a plus or minus 1)
    _acousticsBF->addTerm( _omega * u1, v1);
    _acousticsBF->addTerm(-p2, v1->dx());
    _acousticsBF->addTerm( p2_hat, v1);
    _acousticsBF->addTerm( _omega * u2, v2);
    _acousticsBF->addTerm( p1, v2->dx());
    _acousticsBF->addTerm(-p1_hat, v2);

    _acousticsBF->addTerm( _omega * p1, q1);
    _acousticsBF->addTerm(-u2, q1->dx());
    _acousticsBF->addTerm( u2_n_hat, q1);
    _acousticsBF->addTerm( _omega * p2, q2);
    _acousticsBF->addTerm( u1, q2->dx());
    _acousticsBF->addTerm(-u1_n_hat, q2);
  }
  else
  {
    _acousticsBF->addTerm( _omega * u1, v1);
    _acousticsBF->addTerm(-p2, v1->div());
    _acousticsBF->addTerm( p2_hat, v1->dot_normal());
    _acousticsBF->addTerm( _omega * u2, v2);
    _acousticsBF->addTerm( p1, v2->div());
    _acousticsBF->addTerm(-p1_hat, v2->dot_normal());

    _acousticsBF->addTerm( _omega * p1, q1);
    _acousticsBF->addTerm(-u2, q1->grad());
    _acousticsBF->addTerm( u2_n_hat, q1);
    _acousticsBF->addTerm( _omega * p2, q2);
    _acousticsBF->addTerm( u1, q2->grad());
    _acousticsBF->addTerm(-u1_n_hat, q2);
  }
}

BFPtr AcousticsTomographyFormulation::bf()
{
  return _acousticsBF;
}

RHSPtr AcousticsTomographyFormulation::rhs()
{

  _acousticsRHS = RHS::rhs();
  FunctionPtr zero = Function::zero();
  VarPtr q1 = this->q(1);
  _acousticsRHS->addTerm(zero * q1);
  return _acousticsRHS;
}

void AcousticsTomographyFormulation::CHECK_VALID_COMPONENT(int i) const // throws exception on bad component value (should be either 1 or 2)
{
  if ((i > 2) || (i < 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component indices must be either 1 or 2");
  }
}


// field variables:
VarPtr AcousticsTomographyFormulation::p(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> pStrings = {S_P1,S_P2};
  return _vf->fieldVar(pStrings[i-1]);
}

VarPtr AcousticsTomographyFormulation::u(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> uStrings = {S_U1,S_U2};
  return _vf->fieldVar(uStrings[i-1]);
}

VarPtr AcousticsTomographyFormulation::p_hat(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> pHatStrings = {S_P1_HAT,S_P2_HAT};
  return _vf->traceVar(pHatStrings[i-1]);
}

VarPtr AcousticsTomographyFormulation::u_n_hat(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> unHatStrings = {S_U1_N_HAT,S_U2_N_HAT};
  return _vf->traceVar(unHatStrings[i-1]);
}

// test variables:
VarPtr AcousticsTomographyFormulation::q(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> qStrings = {S_Q1,S_Q2};
  return _vf->testVar(qStrings[i-1],HGRAD);
}

VarPtr AcousticsTomographyFormulation::v(int i)
{
  CHECK_VALID_COMPONENT(i);
  static const vector<string> vStrings = {S_V1,S_V2};
  if (_spaceDim > 1)
    return _vf->testVar(vStrings[i-1], HDIV);
  else
    return _vf->testVar(vStrings[i-1], HGRAD);
}

void AcousticsTomographyFormulation::addImpedanceBCs(SpatialFilterPtr boundary, FunctionPtr g1, FunctionPtr g2)
{
  FunctionPtr meshSkeletonCharacteristic = Function::meshSkeletonCharacteristic(); // 1 on mesh skeleton, 0 elsewhere
  FunctionPtr boundaryFunction = Teuchos::rcp( new SpatiallyFilteredFunction<double>( meshSkeletonCharacteristic, boundary) );
  VarPtr p1_hat,p2_hat,q1,q2;
  p1_hat = this->p_hat(1);
  p2_hat = this->p_hat(2);
  q1     = this->q(1);
  q2     = this->q(2);
  _acousticsBF->addTerm( boundaryFunction * p2_hat, q1);
  _acousticsBF->addTerm(-boundaryFunction * p1_hat, q2);
  _acousticsRHS->addTerm( boundaryFunction * g2 * q1);
  _acousticsRHS->addTerm(-boundaryFunction * g1 * q2);
}