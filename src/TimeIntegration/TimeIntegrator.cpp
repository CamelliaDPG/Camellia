#include "TimeIntegrator.h"
#include "DPGInnerProduct.h"

TimeIntegrator::TimeIntegrator(BFPtr steadyJacobian, SteadyResidual &steadyResidual, MeshPtr mesh,
    Teuchos::RCP<BCEasy> bc, IPPtr ip, map<int, FunctionPtr> initialCondition, bool nonlinear) :
  _steadyJacobian(steadyJacobian), _steadyResidual(steadyResidual), _bc(bc), _nonlinear(nonlinear)
{
  _t = 0;
  _dt = 1e-3;
  _timestep = 0;
  _nlTolerance = 1e-6;
  _nlIterationMax = 20;

  _rhs = Teuchos::rcp( new RHSEasy );
  _solution = Teuchos::rcp( new Solution(mesh, _bc, _rhs, ip) );

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  _prevTimeSolution = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );
  _prevTimeSolution->projectOntoMesh(initialCondition);
  if (_nonlinear)
  {
    _prevNLSolution = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );
    _prevNLSolution->setSolution(_prevTimeSolution);
  }

  _invDt = Teuchos::rcp( new InvDtFunction(_dt) );

  mesh->registerSolution(_prevTimeSolution);
  mesh->registerSolution(_prevNLSolution);

  if (_nonlinear)
  {
    _rhs->addTerm( -_steadyResidual.createResidual(_prevNLSolution, false) );
  }
}

FunctionPtr TimeIntegrator::invDt()
{
  return _invDt;
}

SolutionPtr TimeIntegrator::prevSolution()
{
  return _prevNLSolution;
}

SolutionPtr TimeIntegrator::solution()
{
  if (_nonlinear)
    return _prevNLSolution;
  else
    return _solution;
}

SolutionPtr TimeIntegrator::solutionUpdate()
{
  if (_nonlinear)
    return _solution;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "solution update only makes sense for nonlinear problems");
}

void TimeIntegrator::addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier)
{
  FunctionPtr trialPrevTime = Function::solution(trialVar, _prevTimeSolution);
  FunctionPtr trialPrevNL = Function::solution(trialVar, _prevNLSolution);
  _steadyJacobian->addTerm( _invDt*multiplier*trialVar, testVar );
  _rhs->addTerm( _invDt*multiplier*trialPrevTime*testVar );
  if (_nonlinear)
    _rhs->addTerm( -_invDt*multiplier*trialPrevNL*testVar );
}

void TimeIntegrator::calcNextTimeStep(double dt)
{
  dynamic_cast< InvDtFunction* >(_invDt.get())->setDt(dt);
  _bc->setTime(_t+dt);
  if (_nonlinear)
  {
    _nlL2Error = 1e10;
    _nlIteration = 1;
    while (_nlL2Error > _nlTolerance)
    {
      if (_nlIteration > _nlIterationMax)
      {
        cout << "Hit maximum number of iterations" << endl;
        break;
      }
      _solution->solve(false);
      _nlL2Error = _solution->L2NormOfSolution(0);
      _prevNLSolution->addSolution(_solution, 1, false, true);
      printNLMessage();
      _nlIteration++;
    }
    _prevTimeSolution->setSolution(_prevNLSolution);
  }
  else
  {
    _solution->solve(false);
    _prevTimeSolution->setSolution(_solution);
  }
  _t += dt;
  _timestep++;
}

void TimeIntegrator::printTimeStepMessage()
{
  cout << endl;
  cout << "timestep: " << _timestep << " t = " << _t+_dt << " dt = " << _dt << endl;
}

void TimeIntegrator::printNLMessage()
{
  cout << "    iteration: " << _nlIteration << " error = " << _nlL2Error << endl;
}

ImplicitEulerIntegrator::ImplicitEulerIntegrator(BFPtr steadyJacobian, SteadyResidual &steadyResidual, MeshPtr mesh,
    Teuchos::RCP<BCEasy> bc, IPPtr ip, map<int, FunctionPtr> initialCondition, bool nonlinear) :
  TimeIntegrator(steadyJacobian, steadyResidual, mesh, bc, ip, initialCondition, nonlinear) {}

void ImplicitEulerIntegrator::runToTime(double T, double dt)
{
  while (_t < T)
  {
    _dt = max(1e-9, min(dt, T-_t));
    printTimeStepMessage();
    calcNextTimeStep(_dt);
  }
}

ESDIRKIntegrator::ESDIRKIntegrator(BFPtr steadyJacobian, SteadyResidual &steadyResidual, MeshPtr mesh,
    Teuchos::RCP<BCEasy> bc, IPPtr ip, map<int, FunctionPtr> initialCondition, int numStages, bool nonlinear) :
  TimeIntegrator(steadyJacobian, steadyResidual, mesh, bc, ip, initialCondition, nonlinear),
  _numStages(numStages)
{
  a.resize(_numStages);
  b.resize(_numStages);
  c.resize(_numStages);
  for (int i = 0; i < _numStages; ++i)
    a[i].resize(_numStages);

  _stageSolution.resize(_numStages);
  _stageRHS.resize(_numStages);
  _steadyLinearTerm.resize(_numStages);

  switch (_numStages)
  {
    case 2:
      a[1][0] = 1./2;
      a[1][1] = 1./2;

      b[0] = 1./2;
      b[1] = 1./2;

      c[0] = 0;
      c[1] = 1;
      break;
    case 4:
      // Values from http://dx.doi.org/10.1006/jcph.2002.7059
      a[1][0] = 1767732205903./4055673282236;
      a[1][1] = 1767732205903./4055673282236;

      a[2][0] = 2746238789719./10658868560708;
      a[2][1] = -640167445237./6845629431997;
      a[2][2] = 1767732205903./4055673282236;

      a[3][0] = 1471266399579./7840856788654;
      a[3][1] = -4482444167858./7529755066697;
      a[3][2] = 11266239266428./11593286722821;
      a[3][3] = 1767732205903./4055673282236;

      b[0] = 1471266399579./7840856788654;
      b[1] = -4482444167858./7529755066697;
      b[2] = 11266239266428./11593286722821;
      b[3] = 1767732205903./4055673282236;

      c[0] = 0;
      c[1] = 1767732205903./2027836641118;
      c[2] = 3./5;
      c[3] = 1;
      break;
    case 6:
      // Values from http://utoronto-comp-aero.wikispaces.com/file/view/sammy_isono_masc.pdf
      a[1][0] = 1./4;
      a[1][1] = 1./4;

      a[2][0] = 8611./62500;
      a[2][1] = -1743./31250;
      a[2][2] = 1./4;

      a[3][0] = 5012029./34652500;
      a[3][1] = -654441./2922500;
      a[3][2] = 174375./388108;
      a[3][3] = 1./4;

      a[4][0] = 15267082809./155376265600;
      a[4][1] = -71443401./120774400;
      a[4][2] = 730878875./902184768;
      a[4][3] = 2285395./8070912;
      a[4][4] = 1./4;

      a[5][0] = 82889./524892;
      a[5][1] = 0;
      a[5][2] = 15625./83664;
      a[5][3] = 69875./102672;
      a[5][4] = -2260./8211;
      a[5][5] = 1./4;

      b[0] = 82889./524892;
      b[1] = 0;
      b[2] = 15625./83664;
      b[3] = 69875./102672;
      b[4] = -2260./8211;
      b[5] = 1./4;

      c[0] = 0;
      c[1] = 1./2;
      c[2] = 83./250;
      c[3] = 31./50;
      c[4] = 17./20;
      c[5] = 1;
      break;

    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid ESDIRK stage number");
  }

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);

  _stageSolution[0] = _prevTimeSolution;
  _stageRHS[0] = _rhs;

  for (int k=1; k < _numStages; k++)
  {
    _stageSolution[k] = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );
    _stageRHS[k] = Teuchos::rcp( new RHSEasy );
  }

  for (int k=0; k < _numStages-1; k++)
  {
    _steadyLinearTerm[k] = _steadyResidual.createResidual(_stageSolution[k], true);
  }

  for (int k=1; k < _numStages; k++)
  {
    if (_nonlinear)
    {
      _stageRHS[k]->addTerm( -_steadyResidual.createResidual(_prevNLSolution, false) );
    }
    for (int j=0; j < k; j++)
    {
      FunctionPtr aFunc = Function::constant(a[k][j]/a[k][k]);
      _stageRHS[k]->addTerm( -aFunc*_steadyLinearTerm[j] );
    }
  }
}

void ESDIRKIntegrator::addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier)
{
  FunctionPtr trialPrevTime = Function::solution(trialVar, _prevTimeSolution);
  FunctionPtr trialPrevNL = Function::solution(trialVar, _prevNLSolution);
  _steadyJacobian->addTerm( _invDt*multiplier*trialVar, testVar );
  for (int k=0; k < _numStages; k++)
  {
    _stageRHS[k]->addTerm( _invDt*multiplier*trialPrevTime*testVar );
    if (_nonlinear)
      _stageRHS[k]->addTerm( -_invDt*multiplier*trialPrevNL*testVar );
  }
}

void ESDIRKIntegrator::calcNextTimeStep(double dt)
{
  for (int k=1; k < _numStages; k++)
  {
    cout << "    stage " << k+1 << endl;
    _nlIteration = 1;
    dynamic_cast< InvDtFunction* >(_invDt.get())->setDt(a[k][k]*dt);
    _bc->setTime(_t+c[k]*dt);
    _solution->setRHS(_stageRHS[k]);
    if (_nonlinear)
    {
      _nlL2Error = 1e10;
      while (_nlL2Error > _nlTolerance)
      {
        _solution->solve(false);
        _nlL2Error = _solution->L2NormOfSolution(0);
        _prevNLSolution->addSolution(_solution, 1, false, true);
        printNLMessage();
        _nlIteration++;
        if (_nlIteration > _nlIterationMax)
        {
          cout << "Hit maximum number of iterations" << endl;
          break;
        }
      }
      _stageSolution[k]->setSolution(_prevNLSolution);
    }
    else
    {
      _solution->solve(false);
      _stageSolution[k]->setSolution(_solution);
    }
  }

  if (_nonlinear)
  {
    _prevTimeSolution->setSolution(_prevNLSolution);
  }
  else
  {
    _solution->setSolution(_stageSolution[_numStages-1]);
    _prevTimeSolution->setSolution(_solution);
  }
  _t += dt;
  _timestep++;
}

void ESDIRKIntegrator::runToTime(double T, double dt)
{
  // Use implicit Euler to start things out since most variables may not
  // be initialized correctly (which is not a problem for implicit Euler)
  if (_t == 0)
  {
    _dt = max(1e-9, 1e-3*min(dt, T-_t));
    printTimeStepMessage();
    TimeIntegrator::calcNextTimeStep(_dt);
  }
  // Continue with expected timestepping
  while (_t < T)
  {
    _dt = max(1e-9, min(dt, T-_t));
    printTimeStepMessage();
    calcNextTimeStep(_dt);
  }
}
