#include "TimeIntegrator.h"
#include "PreviousSolutionFunction.h"
#include "DPGInnerProduct.h"

TimeIntegrator::TimeIntegrator(BFPtr steadyBF,Teuchos::RCP<RHSEasy>  steadyRHS, MeshPtr mesh,
    SolutionPtr solution, map<int, FunctionPtr> initialCondition) :
  _bf(steadyBF), _rhs(steadyRHS), _mesh(mesh), _solution(solution)
{
  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  prevSolution = Teuchos::rcp(new Solution(_mesh, nullBC, nullRHS, nullIP) );
  prevSolution->projectOntoMesh(initialCondition);

  _invDt = Teuchos::rcp( new InvDtFunction(_dt) );

  _mesh->registerSolution(prevSolution);

  _t = 0;
  _dt = 1e-3;
  _timestep = 0;
}

void TimeIntegrator::addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier)
{
  FunctionPtr trialPrevTime = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, trialVar) );
  _bf->addTerm( _invDt*multiplier*trialVar, testVar );
  _rhs->addTerm( _invDt*multiplier*trialPrevTime*testVar );
}

void TimeIntegrator::calcNextTimeStep(double _dt)
{
  dynamic_cast< InvDtFunction* >(_invDt.get())->setDt(_dt);
  _solution->solve(false);
  prevSolution->setSolution(_solution);
}

void TimeIntegrator::printMessage()
{
  cout << "timestep: " << _timestep << " t = " << _t << " dt = " << _dt << endl;
}

ImplicitEulerIntegrator::ImplicitEulerIntegrator(BFPtr steadyBF,Teuchos::RCP<RHSEasy>  steadyRHS, MeshPtr mesh,
    SolutionPtr solution, map<int, FunctionPtr> initialCondition):
  TimeIntegrator(steadyBF, steadyRHS, mesh, solution, initialCondition) {}

void ImplicitEulerIntegrator::runToTime(double T, double dt)
{
  while (_t < T)
  {
    _dt = max(1e-9, min(dt, T));
    calcNextTimeStep(_dt);
    _t += _dt;
    _timestep++;
    printMessage();
  }
}

NonlinearImplicitEulerIntegrator::NonlinearImplicitEulerIntegrator(BFPtr steadyBF,Teuchos::RCP<RHSEasy>  steadyRHS, MeshPtr mesh,
    SolutionPtr solution, map<int, FunctionPtr> initialCondition):
  ImplicitEulerIntegrator(steadyBF, steadyRHS, mesh, solution, initialCondition)
{
  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  _prevNLSolution = Teuchos::rcp(new Solution(_mesh, nullBC, nullRHS, nullIP) );
  _nlRHS = Teuchos::rcp( new RHSEasy );

  _prevNLSolution->setSolution(prevSolution);
  _steadyPrevTimeTerm = _bf->testFunctional(prevSolution);
  _steadyPrevNLTerm = _bf->testFunctional(_prevNLSolution);
  _nlRHS->addTerm( _steadyPrevNLTerm );
}

void NonlinearImplicitEulerIntegrator::addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier)
{
  FunctionPtr trialPrevTime = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, trialVar) );
  FunctionPtr trialPrevNL = Teuchos::rcp( new PreviousSolutionFunction(_prevNLSolution, trialVar) );
  _bf->addTerm( _invDt*multiplier*trialVar, testVar );
  _rhs->addTerm( _invDt*multiplier*trialPrevTime*testVar );
  _nlRHS->addTerm( -_invDt*multiplier*trialPrevTime*testVar );
  _nlRHS->addTerm( _invDt*multiplier*trialPrevTime*testVar );
}

void NonlinearImplicitEulerIntegrator::calcNextTimeStep(double _dt)
{
  dynamic_cast< InvDtFunction* >(_invDt.get())->setDt(_dt);
  for (int i = 0; i < 1; i++)
  {
    _solution->solve(false);
    double L2 = _solution->L2NormOfSolution(0);
    cout << "L2 norm of update = " << L2 << endl;
    _prevNLSolution->addSolution(_solution, 1);
  }
  prevSolution->setSolution(_prevNLSolution);
}

ESDIRKIntegrator::ESDIRKIntegrator(BFPtr steadyBF,Teuchos::RCP<RHSEasy>  steadyRHS, MeshPtr mesh,
    SolutionPtr solution, map<int, FunctionPtr> initialCondition, int numStages):
  TimeIntegrator(steadyBF, steadyRHS, mesh, solution, initialCondition),
  _numStages(numStages)
{
  a.resize(_numStages);
  b.resize(_numStages);
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
      break;

    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid ESDIRK stage number");
  }

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);

  _stageSolution[0] = prevSolution;
  _stageRHS[0] = _rhs;
  for (int k=1; k < _numStages; k++)
  {
    _stageSolution[k] = Teuchos::rcp(new Solution(_mesh, nullBC, nullRHS, nullIP) );
    _stageRHS[k] = Teuchos::rcp( new RHSEasy );
  }

  for (int k=0; k < _numStages-1; k++)
  {
    _steadyLinearTerm[k] = _bf->testFunctional(_stageSolution[k]);
  }

  for (int k=1; k < _numStages; k++)
  {
    for (int j=0; j < k; j++)
    {
      FunctionPtr aFunc = Function::constant(a[k][j]/a[k][k]);
      _stageRHS[k]->addTerm( -aFunc*_steadyLinearTerm[j] );
    }
  }
}

void ESDIRKIntegrator::addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier)
{
  FunctionPtr trialPrevTime = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, trialVar) );
  _bf->addTerm( _invDt*multiplier*trialVar, testVar );
  // This will have issues if original rhs does not equal zero
  for (int k=0; k < _numStages; k++)
  {
    _stageRHS[k]->addTerm( _invDt*multiplier*trialPrevTime*testVar );
  }
}

void ESDIRKIntegrator::calcNextTimeStep(double _dt)
{
  for (int k=1; k < _numStages; k++)
  {
    dynamic_cast< InvDtFunction* >(_invDt.get())->setDt(a[k][k]*_dt);
    _solution->setRHS(_stageRHS[k]);
    _solution->solve(false);
    _stageSolution[k]->setSolution(_solution);
  }

  _solution->setSolution(_stageSolution[_numStages-1]);

  prevSolution->setSolution(_solution);
}

void ESDIRKIntegrator::runToTime(double T, double dt)
{
  // Use implicit Euler to start things out since most variables may not
  // be initialized correctly (which is not a problem for implicit Euler)
  if (_t == 0)
  {
    _dt = max(1e-9, 0.01*min(dt, T-_t));
    TimeIntegrator::calcNextTimeStep(_dt);

    _t += _dt;
    _timestep++;
    printMessage();
  }
  // Continue with expected timestepping
  while (_t < T)
  {
    _dt = max(1e-9, min(dt, T-_t));
    calcNextTimeStep(_dt);
    _t += _dt;
    _timestep++;
    printMessage();
  }
}
