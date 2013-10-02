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

RungeKuttaIntegrator::RungeKuttaIntegrator(BFPtr steadyBF,Teuchos::RCP<RHSEasy>  steadyRHS, MeshPtr mesh,
        SolutionPtr solution, map<int, FunctionPtr> initialCondition):
      TimeIntegrator(steadyBF, steadyRHS, mesh, solution, initialCondition)
{
   // _steadyLinearTerm = _bf->testFunctional(prevSolution);
}

void RungeKuttaIntegrator::runToTime(double T, double dt)
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

ESDIRK2Integrator::ESDIRK2Integrator(BFPtr steadyBF,Teuchos::RCP<RHSEasy>  steadyRHS, MeshPtr mesh,
        SolutionPtr solution, map<int, FunctionPtr> initialCondition):
      RungeKuttaIntegrator(steadyBF, steadyRHS, mesh, solution, initialCondition)
{
   a21 = 0.5;
   a22 = 0.5;
   b1 = 0.5;
   b2 = 0.5;

   BCPtr nullBC = Teuchos::rcp((BC*)NULL);
   RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
   IPPtr nullIP = Teuchos::rcp((IP*)NULL);
   _k2Solution = Teuchos::rcp(new Solution(_mesh, nullBC, nullRHS, nullIP) );
   _k2RHS = Teuchos::rcp( new RHSEasy );

   LinearTermPtr _steadyTerm = _bf->testFunctional(prevSolution);

   FunctionPtr a21Func = Function::constant(a21/a22);
   _k2RHS->addTerm( -a21Func*_steadyTerm );
}

void ESDIRK2Integrator::addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier)
{
   FunctionPtr trialPrevTime = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, trialVar) );
   _bf->addTerm( _invDt*multiplier*trialVar, testVar );
   // This will have issues if original rhs does not equal zero
   _rhs->addTerm( _invDt*multiplier*trialPrevTime*testVar );
   _k2RHS->addTerm( _invDt*multiplier*trialPrevTime*testVar );
}

void ESDIRK2Integrator::calcNextTimeStep(double _dt)
{
   dynamic_cast< InvDtFunction* >(_invDt.get())->setDt(a22*_dt);
   _solution->setRHS(_k2RHS);
   _solution->solve(false);
   _k2Solution->setSolution(_solution);

   _solution->setSolution(_k2Solution);

   prevSolution->setSolution(_solution);
}

ESDIRK4Integrator::ESDIRK4Integrator(BFPtr steadyBF,Teuchos::RCP<RHSEasy>  steadyRHS, MeshPtr mesh,
        SolutionPtr solution, map<int, FunctionPtr> initialCondition):
      RungeKuttaIntegrator(steadyBF, steadyRHS, mesh, solution, initialCondition)
{
   numStages = 6;
   // Values from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.28.8464&rep=rep1&type=pdf
   a[1][0] = 1./4;
   a[1][1] = 1./4;

   a[2][0] = 8611./62500;
   a[2][1] = -1743./31250;
   a[2][2] = 1./4;

   a[3][0] = 5012029./34652500;
   a[3][1] = -654441./2922500;
   a[3][2] = 174375./388108;
   a[3][3] = 2285395./8070912;

   a[4][0] = 1./4;
   a[4][1] = 82889./524892;
   a[4][2] = 0;
   a[4][3] = 15625./83664;
   a[4][4] = 69875./102672;

   a[5][0] = -2260./8211;
   a[5][1] = 1./4;
   a[5][2] = 4586570599./29645900160;
   a[5][3] = 0;
   a[5][4] = 178811875./945068544;
   a[5][5] = 814220225./1159782912;

   b[0] = -2260./8211;
   b[1] = 1./4;
   b[2] = 4586570599./29645900160;
   b[3] = 0;
   b[4] = 178811875./945068544;
   b[5] = 814220225./1159782912;

   BCPtr nullBC = Teuchos::rcp((BC*)NULL);
   RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
   IPPtr nullIP = Teuchos::rcp((IP*)NULL);

   _stageSolution[0] = prevSolution;
   _stageRHS[0] = _rhs;
   for (int k=1; k < numStages; k++)
   {
      _stageSolution[k] = Teuchos::rcp(new Solution(_mesh, nullBC, nullRHS, nullIP) );
      _stageRHS[k] = Teuchos::rcp( new RHSEasy );
   }

   for (int k=0; k < numStages-1; k++)
   {
      _steadyLinearTerm[k] = _bf->testFunctional(_stageSolution[k]);
   }

   for (int k=1; k < numStages; k++)
   {
      for (int j=0; j < k; j++)
      {
         FunctionPtr aFunc = Function::constant(a[k][j]/a[k][k]);
         _stageRHS[k]->addTerm( -aFunc*_steadyLinearTerm[j] );
      }
   }
}

void ESDIRK4Integrator::addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier)
{
   FunctionPtr trialPrevTime = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, trialVar) );
   _bf->addTerm( _invDt*multiplier*trialVar, testVar );
   // This will have issues if original rhs does not equal zero
   for (int k=0; k < numStages; k++)
   {
      _stageRHS[k]->addTerm( _invDt*multiplier*trialPrevTime*testVar );
   }
}

void ESDIRK4Integrator::calcNextTimeStep(double _dt)
{
   for (int k=1; k < numStages; k++)
   {
      dynamic_cast< InvDtFunction* >(_invDt.get())->setDt(a[k][k]*_dt);
      _solution->setRHS(_stageRHS[k]);
      _solution->solve(false);
      _stageSolution[k]->setSolution(_solution);
   }

   _solution->setSolution(_stageSolution[numStages-1]);

   prevSolution->setSolution(_solution);
}
