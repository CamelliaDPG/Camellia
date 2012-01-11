#include "ConfusionManufacturedSolution.h"
#include "ConfusionBilinearForm.h"
#include "ConfusionProblem.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"

#include "ZoltanMeshPartitionPolicy.h"

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

// Trilinos includes
#include "Intrepid_FieldContainer.hpp"

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
  int polyOrder = 2;
  int pToAdd = 2; // for tests
  
  // define our manufactured solution:
  double epsilon = 5e-2;
  double beta_x = 1.0, beta_y = 2.0;
  bool useTriangles = false;
  bool useEggerSchoeberl = false;
  ConfusionManufacturedSolution exactSolution(epsilon,beta_x,beta_y);
  
  // define our inner product:
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new OptimalInnerProduct( exactSolution.bilinearForm() ) );
  
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
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+pToAdd, useTriangles);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
//  mesh->setPartitionPolicy(Teuchos::rcp(new MeshPartitionPolicy()));
  mesh->setNumPartitions(numProcs);

  // create a solution object
  Teuchos::RCP<Solution> solution;
  if (useEggerSchoeberl)
    solution = Teuchos::rcp(new Solution(mesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip));
  else {
    Teuchos::RCP<ConfusionProblem> problem = Teuchos::rcp( new ConfusionProblem() );
    solution = Teuchos::rcp(new Solution(mesh, problem, problem, ip));
  }
//  vector<int> cellsToRef;
//  cellsToRef.push_back(0);
//  mesh->hRefine(cellsToRef,RefinementPattern::regularRefinementPatternQuad());

  // solve
  solution->solve(false); 

  cout << "Processor " << rank << " returned from solve()." << endl;
  
  if (rank == 0) {
    // save a data file for plotting in MATLAB
    solution->writeToFile(ConfusionBilinearForm::U, "Confusion_u_adaptive.dat");
    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "Confusion_u_hat_adaptive.dat");
    solution->writeFluxesToFile(ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT, "Confusion_beta_n_hat_adaptive.dat");

    cout << "Done solving and writing" << endl;
  }


  int numRefinements = 1;
  double thresholdFactor = 0.20;
  
  double totalEnergyErrorSquared;
  for (int i=0; i<numRefinements; i++) {
    vector<int> cellsToRefine;
    FieldContainer<double> energyError;
    solution->energyError(energyError);
/*
    int numActiveCells = energyError.dimension(0);
    double maxError = 0.0;
    totalEnergyErrorSquared = 0.0;
    for (int activeCellIndex=0; activeCellIndex<numActiveCells; activeCellIndex++) {
      cout << "energy error for cellID " << mesh->activeElements()[activeCellIndex]->cellID() << " is " << energyError(activeCellIndex) << endl;
      maxError = max(energyError(activeCellIndex),maxError);
      totalEnergyErrorSquared += energyError(activeCellIndex) * energyError(activeCellIndex);
    }

    for (int activeCellIndex=0; activeCellIndex<numActiveCells; activeCellIndex++) {
      if (energyError(activeCellIndex) >= thresholdFactor * maxError ) {
        int cellID = mesh->activeElements()[activeCellIndex]->cellID();
        cellsToRefine.push_back(cellID);
        cout << "refining cell ID " << cellID << " on proc " << rank << endl;
      }
    }

    if (useTriangles) {
      mesh->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternTriangle());
    } else {
      mesh->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternQuad());
    }
    cout << "Solving..." << endl;    
    solution->solve(false);
    cout << "Solved..." << endl;    
 */

  }
  
  if (useEggerSchoeberl) {
    // print out the L2 error of the solution:
    double l2error = exactSolution.L2NormOfError(*solution, ConfusionBilinearForm::U);
    cout << "L2 error: " << l2error << endl;
  }

  // save a data file for plotting in MATLAB
//  solution->writeToFile(ConfusionBilinearForm::U, "Confusion_u_adaptive.dat");
//  solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "Confusion_u_hat_adaptive.dat");
  
}
