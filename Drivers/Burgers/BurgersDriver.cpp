#include "BurgersBilinearForm.h"

#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"

#include "PenaltyMethodFilter.h"
#include "BurgersProblem.h"
#include "Projector.h"
#include "BurgersInnerProduct.h"
#include "ZeroFunction.h"
#include "InitialGuess.h"

#include "RefinementStrategy.h"
#include "NonlinearStepSize.h"
#include "NonlinearSolveStrategy.h"

// Trilinos includes
#include "Epetra_Time.h"
#include "Intrepid_FieldContainer.hpp"
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

using namespace std;

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
#else
  int rank = 0;
  int numProcs = 1;
#endif
  int polyOrder = 3;
  int pToAdd = 2; // for tests
  
  // define our manufactured solution or problem bilinear form:
  double epsilon = 1e-2;
  bool useTriangles = false;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  int H1Order = polyOrder + 1;
  int horizontalCells = 2, verticalCells = 2;
  
  double energyThreshold = 0.2; // for mesh refinements
  double nonlinearStepSize = 0.5;
  double nonlinearRelativeEnergyTolerance = 0.015; // used to determine convergence of the nonlinear solution
  
  ////////////////////////////////////////////////////////////////////
  // SET UP PROBLEM 
  ////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<BurgersBilinearForm> bf = Teuchos::rcp(new BurgersBilinearForm(epsilon));
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, bf, H1Order, H1Order+pToAdd, useTriangles);
  //  mesh->setPartitionPolicy(Teuchos::rcp(new MeshPartitionPolicy()));
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
  //  mesh->setEnforceMultiBasisFluxContinuity(true); // experiment
  
  // ==================== SET INITIAL GUESS ==========================
  
  Teuchos::RCP<Solution> backgroundFlow = Teuchos::rcp(new Solution(mesh, Teuchos::rcp((BC*)NULL) , Teuchos::rcp((RHS*)NULL), Teuchos::rcp((DPGInnerProduct*)NULL))); // create null solution 
  
  mesh->registerSolution(backgroundFlow);
  
  map<int, Teuchos::RCP<AbstractFunction> > functionMap;
  functionMap[BurgersBilinearForm::U] = Teuchos::rcp(new InitialGuess());
  functionMap[BurgersBilinearForm::SIGMA_1] = Teuchos::rcp(new ZeroFunction());
  functionMap[BurgersBilinearForm::SIGMA_2] = Teuchos::rcp(new ZeroFunction());
  
  backgroundFlow->projectOntoMesh(functionMap);
  bf->setBackgroundFlow(backgroundFlow);
  
  // ==================== END SET INITIAL GUESS ==========================
  
  
  // define our inner product:
  Teuchos::RCP<BurgersInnerProduct> ip = Teuchos::rcp( new BurgersInnerProduct( bf, mesh ) );
  
  // create a solution object
  Teuchos::RCP<BurgersProblem> problem = Teuchos::rcp( new BurgersProblem(bf) );
  
  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, problem, problem, ip));
  mesh->registerSolution(solution);
  Teuchos::RCP<LocalStiffnessMatrixFilter> penaltyBC = Teuchos::rcp(new PenaltyMethodFilter(problem));
  solution->setFilter(penaltyBC);
  
  // define refinement strategy:
  Teuchos::RCP<RefinementStrategy> refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));
  
  // =================== END INITIALIZATION CODE ==========================
  
  //  solution->solve(false);
  //  backgroundFlow->addSolution(solution,.5);
  //  solution->solve(false);
  //  backgroundFlow->addSolution(solution,.5);
  //  return 0;
  
  int numRefs = 5;
  
  Teuchos::RCP<NonlinearStepSize> stepSize = Teuchos::rcp(new NonlinearStepSize(nonlinearStepSize));
  Teuchos::RCP<NonlinearSolveStrategy> solveStrategy = Teuchos::rcp( new NonlinearSolveStrategy(backgroundFlow,solution,stepSize,nonlinearRelativeEnergyTolerance)
  );
  
  for (int refIndex=0;refIndex<numRefs;refIndex++){    
    solveStrategy->solve(rank==0);
    refinementStrategy->refine(rank==0); // print to console on rank 0
  }
  
  // one more nonlinear solve on refined mesh
  int numNRSteps = 5;
  for (int i=0;i<numNRSteps;i++){
    solution->solve(false);    
    backgroundFlow->addSolution(solution,1.0);
  }
  
  if (rank==0){
    backgroundFlow->writeFieldsToFile(BurgersBilinearForm::U, "u_ref.m");
    backgroundFlow->writeFieldsToFile(BurgersBilinearForm::SIGMA_1, "sigmax.m");
    backgroundFlow->writeFieldsToFile(BurgersBilinearForm::SIGMA_2, "sigmay.m");
    solution->writeFluxesToFile(BurgersBilinearForm::U_HAT, "du_hat_ref.dat");
  }
  
  return 0;
  
}