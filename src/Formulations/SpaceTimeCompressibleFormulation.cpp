//
//  SpaceTimeCompressibleFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#include "SpaceTimeCompressibleFormulation.h"
#include "MeshFactory.h"
#include "PenaltyConstraints.h"
#include "PoissonFormulation.h"
#include "PreviousSolutionFunction.h"
#include "CompressibleProblems.h"
#include <algorithm>

using namespace Camellia;

const string SpaceTimeCompressibleFormulation::s_rho = "rho";
const string SpaceTimeCompressibleFormulation::s_u1 = "u1";
const string SpaceTimeCompressibleFormulation::s_u2 = "u2";
const string SpaceTimeCompressibleFormulation::s_u3 = "u3";
const string SpaceTimeCompressibleFormulation::s_D11 = "D11";
const string SpaceTimeCompressibleFormulation::s_D12 = "D12";
const string SpaceTimeCompressibleFormulation::s_D13 = "D13";
const string SpaceTimeCompressibleFormulation::s_D21 = "D21";
const string SpaceTimeCompressibleFormulation::s_D22 = "D22";
const string SpaceTimeCompressibleFormulation::s_D23 = "D23";
const string SpaceTimeCompressibleFormulation::s_D31 = "D31";
const string SpaceTimeCompressibleFormulation::s_D32 = "D32";
const string SpaceTimeCompressibleFormulation::s_D33 = "D33";
const string SpaceTimeCompressibleFormulation::s_T = "T";
const string SpaceTimeCompressibleFormulation::s_q1 = "q1";
const string SpaceTimeCompressibleFormulation::s_q2 = "q2";
const string SpaceTimeCompressibleFormulation::s_q3 = "q3";

const string SpaceTimeCompressibleFormulation::s_u1hat = "u1hat";
const string SpaceTimeCompressibleFormulation::s_u2hat = "u2hat";
const string SpaceTimeCompressibleFormulation::s_u3hat = "u3hat";
const string SpaceTimeCompressibleFormulation::s_That = "That";
const string SpaceTimeCompressibleFormulation::s_tc = "tc";
const string SpaceTimeCompressibleFormulation::s_tm1 = "tm1";
const string SpaceTimeCompressibleFormulation::s_tm2hat = "tm2hat";
const string SpaceTimeCompressibleFormulation::s_tm3hat = "tm3hat";
const string SpaceTimeCompressibleFormulation::s_te = "te";

const string SpaceTimeCompressibleFormulation::s_vc = "vc";
const string SpaceTimeCompressibleFormulation::s_vm1 = "vm1";
const string SpaceTimeCompressibleFormulation::s_vm2 = "vm2";
const string SpaceTimeCompressibleFormulation::s_vm3 = "vm3";
const string SpaceTimeCompressibleFormulation::s_ve = "ve";
const string SpaceTimeCompressibleFormulation::s_S11 = "S11";
const string SpaceTimeCompressibleFormulation::s_S12 = "S12";
const string SpaceTimeCompressibleFormulation::s_S13 = "S13";
const string SpaceTimeCompressibleFormulation::s_S21 = "S21";
const string SpaceTimeCompressibleFormulation::s_S22 = "S22";
const string SpaceTimeCompressibleFormulation::s_S23 = "S23";
const string SpaceTimeCompressibleFormulation::s_S31 = "S31";
const string SpaceTimeCompressibleFormulation::s_S32 = "S32";
const string SpaceTimeCompressibleFormulation::s_S33 = "S33";
const string SpaceTimeCompressibleFormulation::s_tau = "tau";

SpaceTimeCompressibleFormulation::SpaceTimeCompressibleFormulation(Teuchos::RCP<CompressibleProblem> problem, Teuchos::ParameterList &parameters)
{
  int spaceDim = parameters.get<int>("spaceDim", 1);
  bool steady = parameters.get<bool>("steady", true);
  double mu = parameters.get<double>("mu", 1e-2);
  bool useConformingTraces = parameters.get<bool>("useConformingTraces", false);
  int fieldPolyOrder = parameters.get<int>("fieldPolyOrder", 2);
  int delta_p = parameters.get<int>("delta_p", 2);
  int numTElems = parameters.get<int>("numTElems", 1);
  string norm = parameters.get<string>("norm", "Graph");
  string savedSolutionAndMeshPrefix = parameters.get<string>("savedSolutionAndMeshPrefix", "");

  _spaceDim = spaceDim;
  _steady = steady;
  _mu = mu;
  _useConformingTraces = useConformingTraces;
  MeshTopologyPtr meshTopo = problem->meshTopology(numTElems);
  MeshGeometryPtr meshGeometry = problem->meshGeometry();
  double gamma = problem->gamma();
  double Pr = problem->Pr();
  double R = problem->R();
  double Cp = problem->Cp();
  double Cv = problem->Cv();

  if (!steady)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(meshTopo->getDimension() != _spaceDim + 1, std::invalid_argument, "MeshTopo must be space-time mesh for transient");
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(meshTopo->getDimension() != _spaceDim, std::invalid_argument, "MeshTopo must be spatial mesh for steady");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(mu==0, std::invalid_argument, "mu may not be 0!");
  // TEUCHOS_TEST_FOR_EXCEPTION(spaceDim==1, std::invalid_argument, "Compressible Navier-Stokes is trivial for spaceDim=1");
  TEUCHOS_TEST_FOR_EXCEPTION((spaceDim != 1) && (spaceDim != 2) && (spaceDim != 3), std::invalid_argument, "spaceDim must be 1, 2 or 3");


  Space traceSpace = useConformingTraces ? HGRAD : L2;

  FunctionPtr zero = Function::constant(1);
  FunctionPtr one = Function::constant(1);
  FunctionPtr n_x = TFunction<double>::normal(); // spatial normal
  // FunctionPtr n_x_parity = n_x * TFunction<double>::sideParity();
  FunctionPtr n_xt = TFunction<double>::normalSpaceTime();
  // FunctionPtr n_xt_parity = n_xt * TFunction<double>::sideParity();

  // declare all possible variables -- will only create the ones we need for spaceDim
  // fields
  VarPtr rho;
  VarPtr u1, u2, u3;
  VarPtr D11, D12, D13;
  VarPtr D22, D23;
  VarPtr T;
  VarPtr q1, q2, q3;

  // traces
  VarPtr u1hat, u2hat, u3hat;
  VarPtr That;
  VarPtr tc;
  VarPtr tm1, tm2hat, tm3hat;
  VarPtr te;

  // tests
  VarPtr vc;
  VarPtr vm1, vm2, vm3;
  VarPtr ve;
  VarPtr S11, S12, S13;
  VarPtr S22, S23;
  VarPtr tau;

  _vf = VarFactory::varFactory();

  rho = _vf->fieldVar(s_rho);
  T = _vf->fieldVar(s_T);
  // That = _vf->traceVarSpaceOnly(s_That, 1.0 * T, traceSpace);
  That = _vf->traceVar(s_That, 1.0 * T, traceSpace);
  tc = _vf->fluxVar(s_tc);
  te = _vf->fluxVar(s_te);
  vc = _vf->testVar(s_vc, HGRAD);
  ve = _vf->testVar(s_ve, HGRAD);
  if (spaceDim == 1)
  {
    u1 = _vf->fieldVar(s_u1);
    D11 = _vf->fieldVar(s_D11);
    q1 = _vf->fieldVar(s_q1);
    // u1hat = _vf->traceVarSpaceOnly(s_u1hat, 1.0 * u1, traceSpace);
    u1hat = _vf->traceVar(s_u1hat, 1.0 * u1, traceSpace);
    tm1 = _vf->fluxVar(s_tm1);
    vm1 = _vf->testVar(s_vm1, HGRAD);
    S11 = _vf->testVar(s_S11, HGRAD); // scalar
    tau = _vf->testVar(s_tau, HGRAD); // scalar
  }
  if (spaceDim == 2)
  {
    u1 = _vf->fieldVar(s_u1);
    u2 = _vf->fieldVar(s_u2);
    D11 = _vf->fieldVar(s_D11);
    D12 = _vf->fieldVar(s_D12);
    q1 = _vf->fieldVar(s_q1);
    q2 = _vf->fieldVar(s_q2);
    u1hat = _vf->traceVarSpaceOnly(s_u1hat, 1.0 * u1, traceSpace);
    u2hat = _vf->traceVarSpaceOnly(s_u2hat, 1.0 * u2, traceSpace);
    tm1 = _vf->fluxVar(s_tm1);
    tm2hat = _vf->fluxVar(s_tm2hat);
    vm1 = _vf->testVar(s_vm1, HGRAD);
    vm2 = _vf->testVar(s_vm2, HGRAD);
    S11 = _vf->testVar(s_S11, HGRAD); // scalar
    S12 = _vf->testVar(s_S12, HGRAD); // scalar
    tau = _vf->testVar(s_tau, HDIV); // vector
  }
  if (spaceDim == 3)
  {
    u1 = _vf->fieldVar(s_u1);
    u2 = _vf->fieldVar(s_u2);
    u3 = _vf->fieldVar(s_u3);
    D11 = _vf->fieldVar(s_D11);
    D12 = _vf->fieldVar(s_D12);
    D13 = _vf->fieldVar(s_D13);
    D22 = _vf->fieldVar(s_D22);
    D23 = _vf->fieldVar(s_D23);
    q1 = _vf->fieldVar(s_q1);
    q2 = _vf->fieldVar(s_q2);
    q3 = _vf->fieldVar(s_q3);
    u1hat = _vf->traceVarSpaceOnly(s_u1hat, 1.0 * u1, traceSpace);
    u2hat = _vf->traceVarSpaceOnly(s_u2hat, 1.0 * u2, traceSpace);
    u3hat = _vf->traceVarSpaceOnly(s_u3hat, 1.0 * u3, traceSpace);
    tm1 = _vf->fluxVar(s_tm1);
    tm2hat = _vf->fluxVar(s_tm2hat);
    tm3hat = _vf->fluxVar(s_tm3hat);
    vm1 = _vf->testVar(s_vm1, HGRAD);
    vm2 = _vf->testVar(s_vm2, HGRAD);
    vm3 = _vf->testVar(s_vm3, HGRAD);
    S11 = _vf->testVar(s_S11, HGRAD); // scalar
    S12 = _vf->testVar(s_S12, HGRAD); // scalar
    S13 = _vf->testVar(s_S13, HGRAD); // scalar
    S22 = _vf->testVar(s_S22, HGRAD); // scalar
    S23 = _vf->testVar(s_S23, HGRAD); // scalar
    tau = _vf->testVar(s_tau, HDIV); // vector
  }

  // LinearTermPtr tc_lt;
  // if (spaceDim == 1)
  // {
  //   tc_lt = beta->x()*n_x_parity->x()*u
  //     -D1 * n_x_parity->x()
  //     + u*n_xt_parity->t();
  // }
  // else if (spaceDim == 2)
  // {
  //   tc_lt = beta->x()*n_x_parity->x()*u
  //     + beta->y()*n_x_parity->y()*u
  //     - D1 * n_x_parity->x()
  //     - D2 * n_x_parity->y()
  //     + u*n_xt_parity->t();
  // }
  // else if (spaceDim == 3)
  // {
  //   tc_lt = beta->x()*n_x_parity->x()*u
  //     + beta->y()*n_x_parity->y()*u
  //     + beta->z()*n_x_parity->z()*u
  //     - D1 * n_x_parity->x()
  //     - D2 * n_x_parity->y()
  //     - D3 * n_x_parity->z()
  //     + u*n_xt_parity->t();
  // }
  // tc = _vf->fluxVar(s_tc, tc_lt);

  _bf = Teuchos::rcp( new BF(_vf) );
  _rhs = RHS::rhs();




  // Define mesh
  BCPtr bc = BC::bc();

  vector<int> H1Order(2);
  H1Order[0] = fieldPolyOrder + 1;
  H1Order[1] = fieldPolyOrder + 1; // for now, use same poly. degree for temporal bases...
  if (savedSolutionAndMeshPrefix == "")
  {
    MeshPtr proxyMesh = Teuchos::rcp( new Mesh(meshTopo->deepCopy(), _bf, H1Order, delta_p) ) ;
    _mesh = Teuchos::rcp( new Mesh(meshTopo, _bf, H1Order, delta_p) ) ;
    // if (meshGeometry != Teuchos::null)
    //   _mesh->setEdgeToCurveMap(meshGeometry->edgeToCurveMap());
    proxyMesh->registerObserver(_mesh);
    problem->preprocessMesh(proxyMesh);
    _mesh->enforceOneIrregularity();

    _solutionUpdate = Solution::solution(_bf, _mesh, bc);
    _solutionBackground = Solution::solution(_bf, _mesh, bc);
    map<int, FunctionPtr> initialGuess;
    initialGuess[rho()->ID()] = problem->rho_exact();
    initialGuess[u(1)->ID()] = problem->u1_exact();
    if (spaceDim > 1)
      initialGuess[u(2)->ID()] = problem->u2_exact();
    if (spaceDim > 2)
      initialGuess[u(3)->ID()] = problem->u3_exact();
    initialGuess[T()->ID()] = problem->T_exact();
    _solutionBackground->projectOntoMesh(initialGuess);
  }
  else
  {
    // // BFPTR version should be deprecated
    _mesh = MeshFactory::loadFromHDF5(_bf, savedSolutionAndMeshPrefix+".mesh");
    _solutionBackground = Solution::solution(_bf, _mesh, bc);
    _solutionBackground->loadFromHDF5(savedSolutionAndMeshPrefix+".soln");
    _solutionUpdate = Solution::solution(_bf, _mesh, bc);
  }

  FunctionPtr rho_prev = Function::solution(rho, _solutionBackground);
  FunctionPtr u1_prev, u2_prev, u3_prev;
  FunctionPtr T_prev = Function::solution(T, _solutionBackground);
  FunctionPtr D11_prev, D12_prev, D13_prev,
              D21_prev, D22_prev, D23_prev,
              D31_prev, D32_prev, D33_prev;
  FunctionPtr q1_prev, q2_prev, q3_prev;
  switch (_spaceDim)
  {
    case 1:
      u1_prev = Function::solution(u1, _solutionBackground);
      D11_prev = Function::solution(D11, _solutionBackground);
      q1_prev = Function::solution(q1, _solutionBackground);
      break;
    case 2:
      u1_prev = Function::solution(u1, _solutionBackground);
      u2_prev = Function::solution(u2, _solutionBackground);
      D11_prev = Function::solution(D11, _solutionBackground);
      D12_prev = Function::solution(D12, _solutionBackground);
      D21_prev = Function::solution(D12, _solutionBackground);
      D22_prev = -Function::solution(D11, _solutionBackground);
      q1_prev = Function::solution(q1, _solutionBackground);
      q2_prev = Function::solution(q2, _solutionBackground);
    case 3:
      u1_prev = Function::solution(u1, _solutionBackground);
      u2_prev = Function::solution(u2, _solutionBackground);
      u3_prev = Function::solution(u3, _solutionBackground);
      D11_prev = Function::solution(D11, _solutionBackground);
      D12_prev = Function::solution(D12, _solutionBackground);
      D13_prev = Function::solution(D13, _solutionBackground);
      D21_prev = Function::solution(D12, _solutionBackground);
      D22_prev = Function::solution(D22, _solutionBackground);
      D23_prev = Function::solution(D23, _solutionBackground);
      D31_prev = Function::solution(D13, _solutionBackground);
      D32_prev = Function::solution(D23, _solutionBackground);
      D33_prev = - Function::solution(D11, _solutionBackground) - Function::solution(D22, _solutionBackground);
      q1_prev = Function::solution(q1, _solutionBackground);
      q2_prev = Function::solution(q2, _solutionBackground);
      q3_prev = Function::solution(q3, _solutionBackground);
  }

  // FunctionPtr u_prev = Function::vectorize(u1_prev, u2_prev);

  // Nonlinear Residual Terms
  FunctionPtr Cc;
  FunctionPtr Cm1, Cm2, Cm3;
  FunctionPtr Ce;
  FunctionPtr Fc1, Fc2, Fc3;
  FunctionPtr Fm11, Fm12, Fm13;
  FunctionPtr Fm21, Fm22, Fm23;
  FunctionPtr Fm31, Fm32, Fm33;
  FunctionPtr Fe1, Fe2, Fe3;
  FunctionPtr Km11, Km12, Km13;
  FunctionPtr Km21, Km22, Km23;
  FunctionPtr Km31, Km32, Km33;
  FunctionPtr Ke1, Ke2, Ke3;
  FunctionPtr MD11, MD12, MD13;
  FunctionPtr MD21, MD22, MD23;
  FunctionPtr MD31, MD32, MD33;
  FunctionPtr Mq1, Mq2, Mq3;
  FunctionPtr GD1, GD2, GD3;
  FunctionPtr Gq;
  // Linearized Terms
  LinearTermPtr Cc_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Cm1_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Cm2_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Cm3_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Ce_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fc1_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fc2_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fc3_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fm11_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fm12_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fm13_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fm21_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fm22_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fm23_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fm31_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fm32_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fm33_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fe1_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fe2_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Fe3_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Km11_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Km12_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Km13_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Km21_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Km22_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Km23_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Km31_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Km32_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Km33_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Ke1_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Ke2_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Ke3_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr MD11_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr MD12_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr MD13_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr MD21_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr MD22_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr MD23_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr MD31_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr MD32_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr MD33_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Mq1_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Mq2_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Mq3_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr GD1_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr GD2_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr GD3_dU = Teuchos::rcp( new LinearTerm );
  LinearTermPtr Gq_dU = Teuchos::rcp( new LinearTerm );
  switch (_spaceDim)
  {
    case 1:
      // Nonlinear Residual Terms
      Cc = rho_prev;
      Cm1 = rho_prev*u1_prev;
      Ce = Cv*rho_prev*T_prev + 0.5*rho_prev*u1_prev*u1_prev;
      Fc1 = rho_prev*u1_prev;
      Fm11 = rho_prev*u1_prev*u1_prev + R*rho_prev*T_prev;
      Fe1 = Cv*rho_prev*u1_prev*T_prev + 0.5*rho_prev*u1_prev*u1_prev*u1_prev + R*rho_prev*u1_prev*T_prev;
      Km11 = D11_prev;
      Ke1 = -q1_prev + u1_prev*D11_prev;
      MD11 = 1./mu*D11_prev;
      Mq1 = Pr/(Cp*mu)*q1_prev;
      GD1 = 2*u1_prev;
      Gq = -T_prev;
      // Linearized Terms
      Cc_dU->addTerm( 1*rho );
      Cm1_dU->addTerm( rho_prev*u1 + u1_prev*rho );
      Ce_dU->addTerm( Cv*T_prev*rho + Cv*rho_prev*T + 0.5*u1_prev*u1_prev*rho + rho_prev*u1_prev*u1 );
      Fc1_dU->addTerm( u1_prev*rho + rho_prev*u1 );
      Fm11_dU->addTerm( u1_prev*u1_prev*rho + 2*rho_prev*u1_prev*u1 + R*T_prev*rho + R*rho_prev*T );
      Fe1_dU->addTerm( Cv*u1_prev*T_prev*rho + Cv*rho_prev*T_prev*u1 + Cv*rho_prev*u1_prev*T
            + 0.5*u1_prev*u1_prev*u1_prev*rho + 1.5*rho_prev*u1_prev*u1_prev*u1
            + R*rho_prev*T_prev*u1 + R*u1_prev*T_prev*rho + R*rho_prev*u1_prev*T );
      Km11_dU->addTerm( 1*D11 );
      Ke1_dU->addTerm( -q1 + D11_prev*u1 + u1_prev*D11 );
      MD11_dU->addTerm( 1./mu*D11 );
      Mq1_dU->addTerm( Pr/(Cp*mu)*q1 );
      GD1_dU->addTerm( 2*u1 );
      Gq_dU->addTerm( -T );
      break;
    case 2:
      break;
    case 3:
      break;
  }

  switch (_spaceDim)
  {
    case 1:
      // Bilinear Form
      // S terms:
      _bf->addTerm( MD11_dU, S11 );
      _bf->addTerm( GD1_dU, S11->dx() );
      _bf->addTerm( -2*u1hat, S11->times_normal_x() );

      // tau terms:
      _bf->addTerm( Mq1_dU, tau );
      _bf->addTerm( Gq_dU, tau->dx() );
      _bf->addTerm( That, tau->times_normal_x());

      // vc terms:
      _bf->addTerm( -Fc1_dU, vc->dx());
      if (!steady)
        _bf->addTerm( -Cc_dU, vc->dt());
      _bf->addTerm( tc, vc);

      // vm terms:
      _bf->addTerm( -Fm11_dU, vm1->dx());
      _bf->addTerm( Km11_dU, vm1->dx());
      if (!steady)
        _bf->addTerm( -Cm1_dU, vm1->dt());
      _bf->addTerm( tm1, vm1);

      // ve terms:
      _bf->addTerm( -Fe1_dU, ve->dx());
      _bf->addTerm( Ke1_dU, ve->dx());
      if (!steady)
        _bf->addTerm( -Ce_dU, ve->dt());
      _bf->addTerm( te, ve);

      // Residual
      // S terms:
      _rhs->addTerm( -MD11 * S11 );
      _rhs->addTerm( -GD1 * S11->dx() );

      // tau terms:
      _rhs->addTerm( -Mq1 * tau );
      _rhs->addTerm( -Gq * tau->dx() );

      // vc terms:
      _rhs->addTerm( Fc1 * vc->dx() );
      if (!steady)
        _rhs->addTerm( Cc * vc->dt() );

      // vm terms:
      _rhs->addTerm( Fm11 * vm1->dx() );
      _rhs->addTerm( -Km11 * vm1->dx() );
      if (!steady)
        _rhs->addTerm( Cm1 * vm1->dt() );

      // ve terms:
      _rhs->addTerm( Fe1 * ve->dx() );
      _rhs->addTerm( -Ke1 * ve->dx() );
      if (!steady)
        _rhs->addTerm( Ce * ve->dt() );
      break;
    case 2:
      break;
    case 3:
      break;
  }


  _ips["Graph"] = _bf->graphNorm();

  // _ips["CoupledRobust"] = Teuchos::rcp(new IP);
  // // _ips["CoupledRobust"]->addTerm(_beta*v->grad());
  // _ips["CoupledRobust"]->addTerm(u_prev*v1->grad());
  // _ips["CoupledRobust"]->addTerm(u_prev*v2->grad());
  // _ips["CoupledRobust"]->addTerm(u1_prev*v1->dx() + u2_prev*v2->dx());
  // _ips["CoupledRobust"]->addTerm(u1_prev*v1->dy() + u2_prev*v2->dy());
  // // _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau);
  // _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau1);
  // _ips["CoupledRobust"]->addTerm(Function::min(one/Function::h(),Function::constant(1./sqrt(_mu)))*tau2);
  // // _ips["CoupledRobust"]->addTerm(sqrt(_mu)*v->grad());
  // _ips["CoupledRobust"]->addTerm(sqrt(_mu)*v1->grad());
  // _ips["CoupledRobust"]->addTerm(sqrt(_mu)*v2->grad());
  // // _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v);
  // _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v1);
  // _ips["CoupledRobust"]->addTerm(Function::min(sqrt(_mu)*one/Function::h(),one)*v2);
  // // _ips["CoupledRobust"]->addTerm(tau->div() - v->dt() - beta*v->grad());
  // if (!steady)
  // {
  //   _ips["CoupledRobust"]->addTerm(tau1->div() -v1->dt() - u_prev*v1->grad() - u1_prev*v1->dx() - u2_prev*v2->dx());
  //   _ips["CoupledRobust"]->addTerm(tau2->div() -v2->dt() - u_prev*v2->grad() - u1_prev*v1->dy() - u2_prev*v2->dy());
  // }
  // else
  // {
  //   _ips["CoupledRobust"]->addTerm(tau1->div() - u_prev*v1->grad() - u1_prev*v1->dx() - u2_prev*v2->dx());
  //   _ips["CoupledRobust"]->addTerm(tau2->div() - u_prev*v2->grad() - u1_prev*v1->dy() - u2_prev*v2->dy());
  // }
  // // _ips["CoupledRobust"]->addTerm(v1->dx() + v2->dy());
  // _ips["CoupledRobust"]->addTerm(q->grad());
  // _ips["CoupledRobust"]->addTerm(q);


  // _ips["NSDecoupledH1"] = Teuchos::rcp(new IP);
  // // _ips["NSDecoupledH1"]->addTerm(one/Function::h()*tau);
  // _ips["NSDecoupledH1"]->addTerm(one/Function::h()*tau1);
  // _ips["NSDecoupledH1"]->addTerm(one/Function::h()*tau2);
  // // _ips["NSDecoupledH1"]->addTerm(v->grad());
  // _ips["NSDecoupledH1"]->addTerm(v1->grad());
  // _ips["NSDecoupledH1"]->addTerm(v2->grad());
  // // _ips["NSDecoupledH1"]->addTerm(_beta*v->grad()+v->dt());
  // if (!steady)
  // {
  //   _ips["NSDecoupledH1"]->addTerm(v1->dt() + u_prev*v1->grad() + u1_prev*v1->dx() + u2_prev*v2->dx());
  //   _ips["NSDecoupledH1"]->addTerm(v2->dt() + u_prev*v2->grad() + u1_prev*v1->dy() + u2_prev*v2->dy());
  // }
  // else
  // {
  //   _ips["NSDecoupledH1"]->addTerm(u_prev*v1->grad() + u1_prev*v1->dx() + u2_prev*v2->dx());
  //   _ips["NSDecoupledH1"]->addTerm(u_prev*v2->grad() + u1_prev*v1->dy() + u2_prev*v2->dy());
  // }
  // // _ips["NSDecoupledH1"]->addTerm(tau->div());
  // _ips["NSDecoupledH1"]->addTerm(tau1->div());
  // _ips["NSDecoupledH1"]->addTerm(tau2->div());
  // // _ips["NSDecoupledH1"]->addTerm(v);
  // _ips["NSDecoupledH1"]->addTerm(v1);
  // _ips["NSDecoupledH1"]->addTerm(v2);
  // // _ips["CoupledRobust"]->addTerm(v1->dx() + v2->dy());
  // _ips["NSDecoupledH1"]->addTerm(q->grad());
  // _ips["NSDecoupledH1"]->addTerm(q);

  IPPtr ip = _ips.at(norm);
  if (problem->forcingTerm != Teuchos::null)
    _rhs->addTerm(problem->forcingTerm);

  _solutionUpdate->setRHS(_rhs);
  _solutionUpdate->setIP(ip);

  _mesh->registerSolution(_solutionBackground);
  _mesh->registerSolution(_solutionUpdate);

  LinearTermPtr residual = _rhs->linearTerm() - _bf->testFunctional(_solutionUpdate,false); // false: don't exclude boundary terms

  double energyThreshold = 0.2;
  _refinementStrategy = Teuchos::rcp( new RefinementStrategy( _mesh, residual, ip, energyThreshold ) );
}

int SpaceTimeCompressibleFormulation::spaceDim()
{
  return _spaceDim;
}

VarFactoryPtr SpaceTimeCompressibleFormulation::vf()
{
  return _vf;
}

BFPtr SpaceTimeCompressibleFormulation::bf()
{
  return _bf;
}

IPPtr SpaceTimeCompressibleFormulation::ip(string normName)
{
  return _ips.at(normName);
}

double SpaceTimeCompressibleFormulation::mu()
{
  return _mu;
}

RefinementStrategyPtr SpaceTimeCompressibleFormulation::getRefinementStrategy()
{
  return _refinementStrategy;
}

void SpaceTimeCompressibleFormulation::setRefinementStrategy(RefinementStrategyPtr refStrategy)
{
  _refinementStrategy = refStrategy;
}

void SpaceTimeCompressibleFormulation::refine()
{
  _refinementStrategy->refine();
}

VarPtr SpaceTimeCompressibleFormulation::rho()
{
  return _vf->fieldVar(s_rho);
}

VarPtr SpaceTimeCompressibleFormulation::u(int i)
{
  switch (i)
  {
    case 1:
      return _vf->fieldVar(s_u1);
    case 2:
      return _vf->fieldVar(s_u2);
    case 3:
      return _vf->fieldVar(s_u3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr SpaceTimeCompressibleFormulation::T()
{
  return _vf->fieldVar(s_T);
}

VarPtr SpaceTimeCompressibleFormulation::D(int i, int j)
{
  if (i > _spaceDim || j > _spaceDim || i < 1 || j < 1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (_spaceDim)
  {
    case 1:
      return _vf->fieldVar(s_D11);
    case 2:
      switch (i)
      {
        case 1:
          switch (j)
          {
            case 1:
              return _vf->fieldVar(s_D11);
            case 2:
              return _vf->fieldVar(s_D12);
          }
        case 2:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only D11 and D12 are independent variables in 2D");
      }
    case 3:
      switch (i)
      {
        case 1:
          switch (j)
          {
            case 1:
              return _vf->fieldVar(s_D11);
            case 2:
              return _vf->fieldVar(s_D12);
            case 3:
              return _vf->fieldVar(s_D13);
          }
        case 2:
          switch (j)
          {
            case 1:
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "D21 is not independent in 3D");
            case 2:
              return _vf->fieldVar(s_D22);
            case 3:
              return _vf->fieldVar(s_D23);
          }
        case 3:
          switch (j)
          {
            case 1:
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "D31 is not independent in 3D");
            case 2:
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "D32 is not independent in 3D");
            case 3:
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "D33 is not independent in 3D");
          }
      }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i, j values");
}

VarPtr SpaceTimeCompressibleFormulation::q(int i)
{
  switch (i)
  {
    case 1:
      return _vf->fieldVar(s_q1);
    case 2:
      return _vf->fieldVar(s_q2);
    case 3:
      return _vf->fieldVar(s_q3);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

// traces:
VarPtr SpaceTimeCompressibleFormulation::uhat(int i)
{
  switch (i)
  {
    case 1:
      return _vf->traceVarSpaceOnly(s_u1hat);
    case 2:
      return _vf->traceVarSpaceOnly(s_u2hat);
    case 3:
      return _vf->traceVarSpaceOnly(s_u3hat);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

// traces:
VarPtr SpaceTimeCompressibleFormulation::That()
{
  return _vf->traceVarSpaceOnly(s_That);
}

VarPtr SpaceTimeCompressibleFormulation::tc()
{
  return _vf->fluxVar(s_tc);
}

VarPtr SpaceTimeCompressibleFormulation::tm(int i)
{
  switch (i)
  {
    case 1:
      return _vf->fluxVar(s_tm1);
    case 2:
      return _vf->fluxVar(s_tm2hat);
    case 3:
      return _vf->fluxVar(s_tm3hat);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr SpaceTimeCompressibleFormulation::te()
{
  return _vf->fluxVar(s_te);
}

VarPtr SpaceTimeCompressibleFormulation::vc()
{
  return _vf->testVar(s_vc, HGRAD);
}

VarPtr SpaceTimeCompressibleFormulation::vm(int i)
{
  switch (i)
  {
    case 1:
      return _vf->testVar(s_vm1, HGRAD);
    case 2:
      return _vf->testVar(s_vm2, HGRAD);
    case 3:
      return _vf->testVar(s_vm3, HGRAD);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i value");
}

VarPtr SpaceTimeCompressibleFormulation::ve()
{
  return _vf->testVar(s_ve, HGRAD);
}

VarPtr SpaceTimeCompressibleFormulation::S(int i, int j)
{
  if (i > _spaceDim || j > _spaceDim || i < 1 || j < 1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "i must be less than or equal to _spaceDim");
  }
  switch (_spaceDim)
  {
    case 1:
      return _vf->fieldVar(s_S11);
    case 2:
      switch (i)
      {
        case 1:
          switch (j)
          {
            case 1:
              return _vf->fieldVar(s_S11);
            case 2:
              return _vf->fieldVar(s_S12);
          }
        case 2:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only S11 and S12 are independent variables in 2D");
      }
    case 3:
      switch (i)
      {
        case 1:
          switch (j)
          {
            case 1:
              return _vf->fieldVar(s_S11);
            case 2:
              return _vf->fieldVar(s_S12);
            case 3:
              return _vf->fieldVar(s_S13);
          }
        case 2:
          switch (j)
          {
            case 1:
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "S21 is not independent in 3S");
            case 2:
              return _vf->fieldVar(s_S22);
            case 3:
              return _vf->fieldVar(s_S23);
          }
        case 3:
          switch (j)
          {
            case 1:
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "S31 is not independent in 3D");
            case 2:
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "S32 is not independent in 3D");
            case 3:
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "S33 is not independent in 3D");
          }
      }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled i, j values");
}

// test variables:
VarPtr SpaceTimeCompressibleFormulation::tau()
{
  return _vf->testVar(s_tau, HDIV);
}

set<int> SpaceTimeCompressibleFormulation::nonlinearVars()
{
  set<int> nonlinearVars;//{u(1)->ID(),u(2)->ID()};
  nonlinearVars.insert(rho()->ID());
  nonlinearVars.insert(u(1)->ID());
  // nonlinearVars.insert(u(2)->ID());
  nonlinearVars.insert(T()->ID());
  nonlinearVars.insert(D(1,1)->ID());
  nonlinearVars.insert(q(1)->ID());
  return nonlinearVars;
}

// ! Saves the solution(s) and mesh to an HDF5 format.
void SpaceTimeCompressibleFormulation::save(std::string prefixString)
{
  _solutionUpdate->mesh()->saveToHDF5(prefixString+".mesh");
  // _solutionUpdate->saveToHDF5(prefixString+"_update.soln");
  _solutionBackground->saveToHDF5(prefixString+".soln");
}

// ! Returns the solution
SolutionPtr SpaceTimeCompressibleFormulation::solutionUpdate()
{
  return _solutionUpdate;
}

// ! Returns the solution
SolutionPtr SpaceTimeCompressibleFormulation::solutionBackground()
{
  return _solutionBackground;
}

void SpaceTimeCompressibleFormulation::updateSolution()
{
  double alpha = 1;
  vector<int> trialIDs = _vf->trialIDs();
  set<int> trialIDSet(trialIDs.begin(), trialIDs.end());
  set<int> nlVars = nonlinearVars();
  set<int> lVars;
  set_difference(trialIDSet.begin(), trialIDSet.end(), nlVars.begin(), nlVars.end(),
      std::inserter(lVars, lVars.end()));
  _solutionBackground->addReplaceSolution(_solutionUpdate, alpha, nlVars, lVars);
}

// ! Solves
void SpaceTimeCompressibleFormulation::solve()
{
  _solutionUpdate->solve();
}
