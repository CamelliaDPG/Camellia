#include "ConfusionBilinearForm.h"
#include "ConfusionProblem.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"

// added by Jesse
#include "PenaltyMethodFilter.h"
#include "ConfectionProblem.h"
#include "ConfectionManufacturedSolution.h"
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
  double beta_x = .8, beta_y = .6;
  bool useTriangles = false;
  Teuchos::RCP<ConfusionBilinearForm> bf = Teuchos::rcp(new ConfusionBilinearForm(epsilon,beta_x,beta_y));
  Teuchos::RCP<ConfectionManufacturedSolution> exactSolution = Teuchos::rcp(new ConfectionManufacturedSolution(epsilon,beta_x,beta_y));
  bool useExactSolution = true;
  
  FieldContainer<double> quadPoints(4,2);

  if (useExactSolution){
    quadPoints(0,0) = -1.0; // x1
    quadPoints(0,1) = -1.0; // y1
    quadPoints(1,0) = 1.0;
    quadPoints(1,1) = -1.0;
    quadPoints(2,0) = 1.0;
    quadPoints(2,1) = 1.0;
    quadPoints(3,0) = -1.0;
    quadPoints(3,1) = 1.0;  
  } else {
    quadPoints(0,0) = 0.0; // x1
    quadPoints(0,1) = 0.0; // y1
    quadPoints(1,0) = 1.0;
    quadPoints(1,1) = 0.0;
    quadPoints(2,0) = 1.0;
    quadPoints(2,1) = 1.0;
    quadPoints(3,0) = 0.0;
    quadPoints(3,1) = 1.0;  
  }
  
  int H1Order = polyOrder + 1;
  int horizontalCells = 4, verticalCells = 4;
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh;
  mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, bf, H1Order, H1Order+pToAdd, useTriangles);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));

  // define our inner product:
  Teuchos::RCP<ConfusionInnerProduct> ip = Teuchos::rcp( new ConfusionInnerProduct( bf, mesh ) );
  //  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new OptimalInnerProduct( bf ) );
  //  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct( bf ) );

  // create a solution object
  Teuchos::RCP<ConfectionProblem> problem = Teuchos::rcp( new ConfectionProblem(bf) );
  Teuchos::RCP<LocalStiffnessMatrixFilter> penaltyBC;
  Teuchos::RCP<Solution> solution;
  if (useExactSolution){
    solution = Teuchos::rcp(new Solution(mesh, exactSolution->bc(), exactSolution->ExactSolution::rhs(), ip));
    penaltyBC= Teuchos::rcp(new PenaltyMethodFilter(exactSolution));
  } else {
    solution = Teuchos::rcp(new Solution(mesh, problem, problem, ip));
    penaltyBC = Teuchos::rcp(new PenaltyMethodFilter(problem));
  }
  solution->setFilter(penaltyBC);
 
  solution->solve(false);
  cout << "Processor " << rank << " returned from solve()." << endl;
  if (rank==0){
    solution->writeFieldsToFile(ConfusionBilinearForm::U, "u.m");
    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "u_hat.dat");
  }
  int cubDegree = 15;
  double l2error = exactSolution->L2NormOfError(*solution, ConfusionBilinearForm::U,cubDegree);
  cout << "with epsilon " << bf->getEpsilon() << ", L2 error: " << l2error << endl;

  bf->setEpsilon(bf->getEpsilon()*.1);

  solution->solve(false);
  if (rank==0){
    solution->writeFieldsToFile(ConfusionBilinearForm::U, "u.m");
    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "u_hat.dat");
  }
  l2error = exactSolution->L2NormOfError(*solution, ConfusionBilinearForm::U,cubDegree);
  cout << "with epsilon " << bf->getEpsilon() << ", L2 error: " << l2error << endl;

  bf->setEpsilon(bf->getEpsilon()*.1);

  solution->solve(false);
  if (rank==0){
    solution->writeFieldsToFile(ConfusionBilinearForm::U, "u.m");
    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "u_hat.dat");
  }
  l2error = exactSolution->L2NormOfError(*solution, ConfusionBilinearForm::U,cubDegree);
  cout << "with epsilon " << bf->getEpsilon() << ", L2 error: " << l2error << endl;

  bf->setEpsilon(bf->getEpsilon()*.1);

  solution->solve(false);
  if (rank==0){
    solution->writeFieldsToFile(ConfusionBilinearForm::U, "u.m");
    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "u_hat.dat");
  }
  l2error = exactSolution->L2NormOfError(*solution, ConfusionBilinearForm::U,cubDegree);
  cout << "with epsilon " << bf->getEpsilon() << ", L2 error: " << l2error << endl;

  return 0;

  bool limitIrregularity = true;
  int numRefinements = 4;
  double thresholdFactor = 0.0;
  int refIterCount = 0;  
  vector<double> errorVector;
  vector<double> L2errorVector;
  vector<double> meshSizes; // assuming uniform meshes
  vector<int> dofVector;
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
      if (useExactSolution) {
	// print out the L2 error of the solution:
	int cubDegree = 15;
	double l2error = exactSolution->L2NormOfError(*solution, ConfusionBilinearForm::U,cubDegree);
	cout << "L2 error: " << l2error << endl;
	L2errorVector.push_back(l2error);
	meshSizes.push_back(pow(.5,refIterCount+1));
      }      	
    }
    errorVector.push_back(totalEnergyErrorSquared);
    dofVector.push_back(mesh->numGlobalDofs());

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
  
  // save a data file for plotting in MATLAB
  if (rank==0){
    //    solution->writeFieldsToFile(ConfusionBilinearForm::U, "Confusion_u_adaptive.dat");
    //    solution->writeToFile(ConfusionBilinearForm::U, "u.dat");
    solution->writeFieldsToFile(ConfusionBilinearForm::U, "u.m");
    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "u_hat.dat");
    solution->writeFluxesToFile(ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT, "sigma_hat.dat");

    ofstream fout1("errors.dat");
    fout1 << setprecision(15);
    for (int i = 0;i<errorVector.size();i++){
      fout1 << errorVector[i] << endl;
    }
    fout1.close();

    ofstream fout2("dofs.dat");
    for (int i = 0;i<dofVector.size();i++){
      fout2 << dofVector[i] << endl;
    }
    fout2.close();

    if (useExactSolution){
      string epsString("1e4");
      string L2errorFile("L2errors");      
      L2errorFile += epsString + string(".dat");
      ofstream fout3(L2errorFile.c_str());
      fout3 << setprecision(15);
      for (int i = 0;i<errorVector.size();i++){
	fout3 << L2errorVector[i] << endl;
      }
      fout3.close();
      string meshFile("meshSizes.dat");
      ofstream fout4(meshFile.c_str());
      fout4<< setprecision(15);
      for (int i = 0;i<errorVector.size();i++){
	fout4 << meshSizes[i] << endl;
      }
      fout4.close();      
    }

    cout << "Done writing soln to file." << endl;
  }
  
  return 0;
}
