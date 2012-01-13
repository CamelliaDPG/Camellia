#include "ConfusionManufacturedSolution.h"
#include "ConfusionBilinearForm.h"
#include "ConfusionProblem.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"
#include "MeshTestSuite.h"

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
  int polyOrder = 3;
  int pToAdd = 4; // for tests
  
  // define our manufactured solution:
  double epsilon = 1e-2;
  double beta_x = 1.0, beta_y = 1.5;
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
  int horizontalCells = 2, verticalCells = 2;  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+pToAdd, useTriangles);
  cout << "In driver, setting numProcs = " << numProcs << endl;
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
  mesh->setNumPartitions(numProcs);
  mesh->rebuildLookups();

  // create a solution object
  Teuchos::RCP<Solution> solution;
  if (useEggerSchoeberl)
    solution = Teuchos::rcp(new Solution(mesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip));
  else {
    Teuchos::RCP<ConfusionProblem> problem = Teuchos::rcp( new ConfusionProblem() );
    solution = Teuchos::rcp(new Solution(mesh, problem, problem, ip));
  }
 
  if (false){
    // ---------------------- TEST PARITY ----------------------------
    
    int numInitialElems = mesh->activeElements().size();
    vector<int> cells;  
    cells.push_back(0);
    cells.push_back(1);

    mesh->hRefine(cells,RefinementPattern::regularRefinementPatternQuad());
    cout << "mesh test suite consistency parity test returns (1st set of refs) " << MeshTestSuite::checkMeshConsistency(*mesh) << endl;
    cells.clear();
    
    cells.push_back(numInitialElems);  
    cells.push_back(numInitialElems+1);  
    cells.push_back(numInitialElems+2);  
    cells.push_back(numInitialElems+3);  
    mesh->hRefine(cells,RefinementPattern::regularRefinementPatternQuad());
    cout << "mesh test suite consistency parity test returns (2nd set of refs) " << MeshTestSuite::checkMeshConsistency(*mesh) << endl;

    // solve
    solution->solve(false); 

    if (rank==0){
      solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "multi_irregular.dat");
    }
    
    mesh->enforceOneIrregularity();
   
    // solve
    solution->solve(false);         

    if (rank==0){
      solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "irregular.dat");
    }

    cout << "mesh test suite consistency parity test returns (after making mesh regular) " << MeshTestSuite::checkMeshConsistency(*mesh) << endl;

    
    return 0;
    
    // ---------------------- END TEST PARITY ----------------------------
  }else{
    solution->solve(false);
    cout << "Processor " << rank << " returned from solve()." << endl;
  }

  bool limitIrregularity = true;
  int numRefinements = 3;
  double thresholdFactor = 0.20;
  int refIterCount = 0;  
  double totalEnergyErrorSquared;
  for (int i=0; i<numRefinements; i++) {
    map<int, double> energyError;
    solution->energyError(energyError);
    vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();
    vector< Teuchos::RCP< Element > >::iterator activeElemIt;

    // greedy refinement algorithm
    vector<int> triangleCellsToRefine;
    vector<int> quadCellsToRefine;
    double maxError = 0.0;
    for (activeElemIt = activeElements.begin();activeElemIt != activeElements.end(); activeElemIt++){
      Teuchos::RCP< Element > current_element = *(activeElemIt);
      int cellID = current_element->cellID();
      //      cout << "energy error for cellID " << cellID << " = " << energyError[cellID] << endl;      
      maxError = max(energyError[cellID],maxError);
      totalEnergyErrorSquared += energyError[cellID]*energyError[cellID];
    }
    cout << "For refinement number " << refIterCount << ", energy error = " << totalEnergyErrorSquared<<endl;

    //actually do refinements
    for (activeElemIt = activeElements.begin();activeElemIt != activeElements.end(); activeElemIt++){
      Teuchos::RCP< Element > current_element = *(activeElemIt);
      int cellID = current_element->cellID();
      if (energyError[cellID]>=thresholdFactor*maxError){
	if (current_element->numSides()==3){
	  triangleCellsToRefine.push_back(cellID);
	}else if (current_element->numSides()==4){
	  quadCellsToRefine.push_back(cellID);
	}
	if (rank==0){
	  cout << "refining cell ID " << cellID << endl;
	}
      }
    }    
    mesh->hRefine(triangleCellsToRefine,RefinementPattern::regularRefinementPatternTriangle());
    triangleCellsToRefine.clear();
    mesh->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad());
    quadCellsToRefine.clear();

    // enforce 1-irregularity if desired
    if (limitIrregularity){
      if (refIterCount==2){
	vector<int> manualCells;
	manualCells.push_back(12);
	mesh->hRefine(manualCells,RefinementPattern::regularRefinementPatternQuad());
      }else{    
	//	mesh->enforceOneIrregularity();
      }
    }
    
    refIterCount++;
    cout << "Solving on refinement iteration number " << refIterCount << "..." << endl;    
    solution->solve(false);
    cout << "Solved..." << endl;    
 } 
  
  if (useEggerSchoeberl) {
    // print out the L2 error of the solution:
    double l2error = exactSolution.L2NormOfError(*solution, ConfusionBilinearForm::U);
    cout << "L2 error: " << l2error << endl;
  }

  // save a data file for plotting in MATLAB
  if (rank==0){
    solution->writeToFile(ConfusionBilinearForm::U, "Confusion_u_adaptive.dat");
    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "Confusion_u_hat_adaptive.dat");
    cout << "Done writing soln to file." << endl;
  }
  if (rank==0){
    cout << "mesh test suite consistency parity test returns (after making mesh regular) " << MeshTestSuite::checkMeshConsistency(*mesh) << endl;
  }

  return 0;
}
