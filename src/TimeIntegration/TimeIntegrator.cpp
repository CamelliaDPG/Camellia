#include "TimeIntegrator.h"
#include "PreviousSolutionFunction.h"

TimeIntegrator::TimeIntegrator(BFPtr steadyBF,Teuchos::RCP<RHSEasy>  steadyRHS, MeshPtr _mesh,
        SolutionPtr _solution, map<int, FunctionPtr> initialCondition):
         _bf(steadyBF), _rhs(steadyRHS), _mesh(_mesh), _solution(_solution)
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
   cout << "timestep: " << _timestep << " t = " << _t << " _dt = " << _dt << endl;
}

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

void TrapezoidRuleIntegrator::addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier)
{
   steadyLinearTerm = _bf->testFunctional(prevSolution);

   TimeIntegrator::addTimeTerm(trialVar, testVar, multiplier);
}

void TrapezoidRuleIntegrator::runToTime(double T, double dt)
{
   if (_t == 0)
   {
      _dt = max(1e-9, min(dt, T));
      calcNextTimeStep(_dt);

      _rhs->addTerm( -steadyLinearTerm );

      _t += _dt;
      _timestep++;
      printMessage();
   }
   while (_t < T)
   {
      _dt = max(1e-9, min(dt, T));
      calcNextTimeStep(0.5*_dt);
      _t += _dt;
      _timestep++;
      printMessage();
   }
}
