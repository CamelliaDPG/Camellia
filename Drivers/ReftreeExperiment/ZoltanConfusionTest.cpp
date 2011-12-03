#include "ConfusionManufacturedSolution.h"
#include "ConfusionBilinearForm.h"
#include "ConfusionProblem.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"

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
  int polyOrder = 1;
  int pToAdd = 0; // for tests
  
  // define our manufactured solution:
  double epsilon = 1e-2;
  double beta_x = 1.0, beta_y = 2.0;
  bool useTriangles = false;
  bool useEggerSchoeberl = false;
  ConfusionManufacturedSolution exactSolution(epsilon,beta_x,beta_y);
  
  // define our inner product:
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new OptimalInnerProduct( exactSolution.bilinearForm() ) );

  exactSolution.bilinearForm()->printTrialTestInteractions();
  
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
  int horizontalCells = 1, verticalCells = 1;
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+pToAdd, useTriangles);

  // set partitioner
  string partitionType = "RANDOM";
  Teuchos::RCP< ZoltanMeshPartitionPolicy > ZoltanPartitionPolicy = Teuchos::rcp(new ZoltanMeshPartitionPolicy(partitionType));

  mesh->setNumPartitions(numProcs);
  mesh->setPartitionPolicy(ZoltanPartitionPolicy);

  cout << "refining cells---" << endl;
  vector<int> cellsToRefine;
  cellsToRefine.clear();

  // repeatedly refine the first element along the side shared with cellID 1
  int numRefinements = 4;
  for (int i=0; i<numRefinements; i++) {
    vector< pair<int,int> > descendents = mesh->elements()[0]->getDescendentsForSide(1);
    int numDescendents = descendents.size();
    cellsToRefine.clear();
    for (int j=0; j<numDescendents; j++ ) {
      int cellID = descendents[j].first;
      cellsToRefine.push_back(cellID);
    }
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    
    // same thing for north side
    descendents = mesh->elements()[0]->getDescendentsForSide(2);
    numDescendents = descendents.size();
    cellsToRefine.clear();
    for (int j=0; j<numDescendents; j++ ) {
      int cellID = descendents[j].first;
      cellsToRefine.push_back(cellID);
    }
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
  }
  
  //  FieldContainer<int> partitionedMesh(numProcs,mesh->numElements()); 
  //  ZoltanPartitionPolicy->partitionMesh(&(*mesh),numProcs,partitionedMesh);

  //  cout << "--- repartitioning" << endl;
  //  mesh->repartition();

  if (rank==0){
    mesh->writeMeshPartitionsToFile(); //visualize mesh partitions
  }
  return 0;

  //////////////////////////////////////////////////////////////////////////////////////


  // create a solution object
  Teuchos::RCP<Solution> solution;
  if (useEggerSchoeberl)
    solution = Teuchos::rcp(new Solution(mesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip));
  else {
    Teuchos::RCP<ConfusionProblem> problem = Teuchos::rcp( new ConfusionProblem() );
    solution = Teuchos::rcp(new Solution(mesh, problem, problem, ip));
  }
  // solve
  solution->solve(); 

  cout << "Processor " << rank << " returned from solve()." << endl;
  
  if (rank == 0) {
    // save a data file for plotting in MATLAB
    solution->writeToFile(ConfusionBilinearForm::U, "Confusion_u_adaptive.dat");
    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "Confusion_u_hat_adaptive.dat");
    solution->writeFluxesToFile(ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT, "Confusion_beta_n_hat_adaptive.dat");

    cout << "Done solving and writing" << endl;
  }

  mesh->writeMeshPartitionsToFile(); //visualize mesh partitions

  return 0;
    
  /*int numRefinements = 0;
  double thresholdFactor = 0.20;
  
  double totalEnergyErrorSquared;
  for (int i=0; i<numRefinements; i++) {
    vector<int> cellsToRefine;
    FieldContainer<double> energyError;
    solution->energyError(energyError);
    int numActiveCells = energyError.dimension(0);
    double maxError = 0.0;
    totalEnergyErrorSquared = 0.0;
    for (int activeCellIndex=0; activeCellIndex<numActiveCells; activeCellIndex++) {
      maxError = max(energyError(activeCellIndex),maxError);
      totalEnergyErrorSquared += energyError(activeCellIndex) * energyError(activeCellIndex);
    }
    cout << "Energy error: " << sqrt(totalEnergyErrorSquared) << endl;
    for (int activeCellIndex=0; activeCellIndex<numActiveCells; activeCellIndex++) {
      if (energyError(activeCellIndex) >= thresholdFactor * maxError ) {
        int cellID = mesh->activeElements()[activeCellIndex]->cellID();
        cellsToRefine.push_back(cellID);
      }
    }
    if (useTriangles) {
      mesh->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternTriangle());
    } else {
      mesh->hRefine(cellsToRefine,RefinementPattern::regularRefinementPatternQuad());
    }
    solution->solve();
  }
  
  if (useEggerSchoeberl) {
    // print out the L2 error of the solution:
    double l2error = exactSolution.L2NormOfError(*solution, ConfusionBilinearForm::U);
    cout << "L2 error: " << l2error << endl;
  }
  cout << "Energy error: " << sqrt(totalEnergyErrorSquared) << endl;

  // save a data file for plotting in MATLAB
  solution->writeToFile(ConfusionBilinearForm::U, "Confusion_u_adaptive.dat");
  solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "Confusion_u_hat_adaptive.dat");
  */
}
