#include "BurgersBilinearForm.h"

#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"

// added by Jesse
#include "PenaltyMethodFilter.h"
#include "BurgersProblem.h"
#include "Projector.h"
#include "BurgersInnerProduct.h"
#include "ZeroFunction.h"
#include "InitialGuess.h"

#include "SolutionTests.h"

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
  int horizontalCells = 4, verticalCells = 4;

  ////////////////////////////////////////////////////////////////////
  // SET UP PROBLEM 
  ////////////////////////////////////////////////////////////////////

  Teuchos::RCP<BurgersBilinearForm> bf = Teuchos::rcp(new BurgersBilinearForm(epsilon));

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, bf, H1Order, H1Order+pToAdd, useTriangles);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));

  // ==================== SET INITIAL GUESS ==========================

  Teuchos::RCP<Solution> backgroundFlow = Teuchos::rcp(new Solution(mesh, Teuchos::rcp((BC*)NULL) , Teuchos::rcp((RHS*)NULL), Teuchos::rcp((DPGInnerProduct*)NULL))); // create null solution 
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
  Teuchos::RCP<LocalStiffnessMatrixFilter> penaltyBC = Teuchos::rcp(new PenaltyMethodFilter(problem));
  solution->setFilter(penaltyBC);

  int numRefs = 3;
  int refIter = 0;
  for (int refIndex=0;refIndex<numRefs;refIndex++){    

    // initialize energyError stuff
    map<int, double> energyError;
    vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();
    vector< Teuchos::RCP< Element > >::iterator activeElemIt;

    int i = 0;    
    double prevError = 0.0;
    bool converged = false;
    while (!converged){ // while energy error has not stabilized

      solution->solve();

      backgroundFlow->addSolution(solution,1.0);

      // see if energy error has stabilized
      solution->energyError(energyError);    
      double totalError = 0.0;
      for (activeElemIt = activeElements.begin();activeElemIt != activeElements.end(); activeElemIt++){
	Teuchos::RCP< Element > current_element = *(activeElemIt);
	totalError += energyError[current_element->cellID()];
      }
      double relErrorDiff = abs(totalError-prevError)/max(totalError,prevError);
      if (rank==0){
	cout << "on iter = " << i  << ", relative change in energy error is " << relErrorDiff << endl;
      }

      double tol = .01; // if change is less than 1%, solve again
      if (relErrorDiff<tol){
	converged = true;
      } else {
	prevError=totalError; // reset previous error and continue
      } 
      i++;
    }

    // greedy refinement algorithm - mark cells for refinement
    vector<int> triangleCellsToRefine;
    vector<int> quadCellsToRefine;
    double maxError = 0.0;
    for (activeElemIt = activeElements.begin();activeElemIt != activeElements.end(); activeElemIt++){
      Teuchos::RCP< Element > current_element = *(activeElemIt);
      int cellID = current_element->cellID();
      maxError = max(energyError[cellID],maxError);
    }
    
    // do refinements on cells with error above threshold
    for (activeElemIt = activeElements.begin();activeElemIt != activeElements.end(); activeElemIt++){
      Teuchos::RCP< Element > current_element = *(activeElemIt);
      int cellID = current_element->cellID();
      if (energyError[cellID]>=.2*maxError){
	if (current_element->numSides()==3){
	  triangleCellsToRefine.push_back(cellID);
	}else if (current_element->numSides()==4){
	  quadCellsToRefine.push_back(cellID);
	}
      }
    }    

    if (rank==0){
      cout << "refining on iter " << refIter << endl;
    }
    refIter++;

    // reinitialize both background flow/solution data structures
    vector< Teuchos::RCP<Solution> > solutions;
    solutions.push_back(solution);
    solutions.push_back(backgroundFlow);

    cout << "refining/reinitializing dofs on rank " << rank << endl;
    mesh->hRefine(triangleCellsToRefine,RefinementPattern::regularRefinementPatternTriangle(),solutions);
    triangleCellsToRefine.clear();
    mesh->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad(),solutions);
    quadCellsToRefine.clear();
    cout << "enforcing one irregularity on rank " << rank << endl;

    mesh->enforceOneIrregularity(solutions);
    solutions.clear();

    cout << "discarding old cell coeffs on rank " << rank << endl;
    backgroundFlow->discardInactiveCellCoefficients();
    if (rank==0){
      cout << "proceeding to next refinement iteration" << endl;
    }
  }

  // one more nonlinear solve on refined mesh
  int numNRSteps = 6;
  for (int i=0;i<numNRSteps;i++){
    solution->solve(false);
    //    cout << "adding solution" << endl;
    if (rank==0){
      cout << "on iter = " << i << ", storage sizes agree = " << SolutionTests::storageSizesAgree(backgroundFlow,solution) << endl;
    }
    
    backgroundFlow->addSolution(solution,1.0);
  }
  

  if (rank==0){
    backgroundFlow->writeFieldsToFile(BurgersBilinearForm::U, "u_ref.m");
    solution->writeFluxesToFile(BurgersBilinearForm::U_HAT, "du_hat_ref.dat");
  }
  
  return 0;
 
}
