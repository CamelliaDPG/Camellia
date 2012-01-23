#include "ConfusionManufacturedSolution.h"
#include "ConfusionBilinearForm.h"
#include "ConfusionProblem.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"
#include "MeshTestSuite.h"

//#include "LocalStiffnessMatrixFilter.h"
// added by Jesse
#include "PenaltyMethodFilter.h"
#include "ConfectionProblem.h"
#include "ConfusionInnerProduct.h"


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
  int pToAdd = 3; // for tests
  
  // define our manufactured solution or problem bilinear form:
  double epsilon = 1e-3;
  double beta_x = 1.0, beta_y = 1.25;
  bool useTriangles = false;
  bool useEggerSchoeberl = false;
  ConfusionManufacturedSolution exactSolution(epsilon,beta_x,beta_y); 
  Teuchos::RCP<ConfusionBilinearForm> bf = Teuchos::rcp(new ConfusionBilinearForm(epsilon,beta_x,beta_y));

  
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
  //  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+pToAdd, useTriangles);
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, bf, H1Order, H1Order+pToAdd, useTriangles);
  cout << "In driver, setting numProcs = " << numProcs << endl;
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));

  // define our inner product:
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new ConfusionInnerProduct( bf, mesh ) );
  //  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new OptimalInnerProduct( bf ) );
  //  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct( bf ) );

  // create a solution object
  Teuchos::RCP<Solution> solution;
  if (useEggerSchoeberl)
    solution = Teuchos::rcp(new Solution(mesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip));
  else {
    Teuchos::RCP<ConfusionProblem> problem = Teuchos::rcp( new ConfusionProblem() );
    //    Teuchos::RCP<ConfectionProblem> problem = Teuchos::rcp( new ConfectionProblem(bf) );
    solution = Teuchos::rcp(new Solution(mesh, problem, problem, ip));
  }
 
  solution->solve(false);
  cout << "Processor " << rank << " returned from solve()." << endl;

  /*
  // save a data file for plotting in MATLAB
  if (rank==0){
    solution->writeToFile(ConfusionBilinearForm::U, "Confusion_u_adaptive.dat");
    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "Confusion_u_hat_adaptive.dat");
    cout << "Done writing soln to file." << endl;
  }
  return 0;
  */
  bool limitIrregularity = true;
  int numRefinements = 8;
  double thresholdFactor = 0.20;
  int refIterCount = 0;  
  vector<double> errorVector;
  for (int i=0; i<numRefinements; i++) {
    map<int, double> energyError;
    solution->energyError(energyError);
    vector< Teuchos::RCP< Element > > activeElements = mesh->activeElements();
    vector< Teuchos::RCP< Element > >::iterator activeElemIt;

    // greedy refinement algorithm - mark cells for refinement
    vector<int> triangleCellsToRefine;
    vector<int> quadCellsToRefine;
    double maxError = 0.0;
    double totalEnergyErrorSquared=0.0;
    for (activeElemIt = activeElements.begin();activeElemIt != activeElements.end(); activeElemIt++){
      Teuchos::RCP< Element > current_element = *(activeElemIt);
      int cellID = current_element->cellID();
      //      cout << "energy error for cellID " << cellID << " = " << energyError[cellID] << endl;      
      maxError = max(energyError[cellID],maxError);
      totalEnergyErrorSquared += energyError[cellID]*energyError[cellID];
    }
    if (rank==0){
      cout << "For refinement number " << refIterCount << ", energy error = " << totalEnergyErrorSquared<<endl;      
    }
    errorVector.push_back(totalEnergyErrorSquared);

    // do refinements on cells with error above threshold
    for (activeElemIt = activeElements.begin();activeElemIt != activeElements.end(); activeElemIt++){
      Teuchos::RCP< Element > current_element = *(activeElemIt);
      int cellID = current_element->cellID();
      if (energyError[cellID]>=thresholdFactor*maxError){
	if (current_element->numSides()==3){
	  triangleCellsToRefine.push_back(cellID);
	}else if (current_element->numSides()==4){
	  quadCellsToRefine.push_back(cellID);
	}
      }
    }    
    mesh->hRefine(triangleCellsToRefine,RefinementPattern::regularRefinementPatternTriangle());
    triangleCellsToRefine.clear();
    mesh->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad());
    quadCellsToRefine.clear();

    // enforce 1-irregularity if desired
    if (limitIrregularity){
      mesh->enforceOneIrregularity();
    }
    
    refIterCount++;
    if (rank==0){
      cout << "Solving on refinement iteration number " << refIterCount << "..." << endl;    
    }
    solution->solve(false);
    if (rank==0){
      cout << "Solved..." << endl;    
    }
  } 
  
  if (useEggerSchoeberl) {
    // print out the L2 error of the solution:
    double l2error = exactSolution.L2NormOfError(*solution, ConfusionBilinearForm::U);
    cout << "L2 error: " << l2error << endl;
  }

  // save a data file for plotting in MATLAB
  if (rank==0){
    //    solution->writeFieldsToFile(ConfusionBilinearForm::U, "Confusion_u_adaptive.dat");
    solution->writeFieldsToFile(ConfusionBilinearForm::U, "Confusion_u_adaptive.m");
    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "Confusion_u_hat_adaptive.dat");
    cout << "errors = [" << endl;
    for (int i = 0;i<errorVector.size();i++){
      cout << errorVector[i] << endl;
    }
    cout << "];" << endl;
    cout << "Done writing soln to file." << endl;
  }
  
  return 0;
}
