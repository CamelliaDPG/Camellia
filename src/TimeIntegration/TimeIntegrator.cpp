#include "TimeIntegrator.h"
#include "PreviousSolutionFunction.h"
#include "SolutionExporter.h"

ImplicitEulerIntegrator::ImplicitEulerIntegrator(BFPtr steadyBF,Teuchos::RCP<RHSEasy>  steadyRHS, MeshPtr mesh,
        SolutionPtr solution): bf(steadyBF), rhs(steadyRHS), mesh(mesh), solution(solution)
{
   BCPtr nullBC = Teuchos::rcp((BC*)NULL);
   RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
   IPPtr nullIP = Teuchos::rcp((IP*)NULL);
   prevTimeFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );
   invDt = Teuchos::rcp( new InvDtFunction(0.1) );

   // mesh->registerSolution(solution);
   mesh->registerSolution(prevTimeFlow);

   t = 0;
   dt = 0.1;
   timestep = 0;
}


void ImplicitEulerIntegrator::addTimeTerm(VarPtr trialVar, VarPtr testVar, FunctionPtr multiplier)
{

   FunctionPtr trialPrevTime = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, trialVar) );
   bf->addTerm( invDt*multiplier*trialVar, testVar );
   rhs->addTerm( invDt*multiplier*trialPrevTime*testVar );
}

void ImplicitEulerIntegrator::runToTime(double T)
{
   while (t < T)
   {
      calcNextTimeStep(dt);
      t = t + dt;
      timestep += 1;
      cout << "timestep: " << timestep << " t = " << t << " dt = " << dt << endl;
   }
}

void ImplicitEulerIntegrator::calcNextTimeStep(double dt)
{
   // invDt->setDt(dt);
   solution->solve(false);
   prevTimeFlow->setSolution(solution);
}

void ImplicitEulerIntegrator::writeSolution()
{
  // VTKExporter exporter(solution, mesh, bf->varFactory());
}
