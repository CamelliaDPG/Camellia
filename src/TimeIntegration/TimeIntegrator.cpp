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
   _steadyLinearTerm = _bf->testFunctional(prevSolution);
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

   _steadyLinearTerm = _bf->testFunctional(prevSolution);

   FunctionPtr a21Func = Function::constant(a21/a22);
   _k2RHS->addTerm( -a21Func*_steadyLinearTerm );
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
