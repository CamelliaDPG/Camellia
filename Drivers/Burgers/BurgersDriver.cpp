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
  int pToAdd = 3; // for tests
  
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

  /*
  solution->solve(false);
  backgroundFlow->addSolution(solution,1.0);
  if (rank==0){
    solution->writeFieldsToFile(BurgersBilinearForm::U, "du.m");
    solution->writeFluxesToFile(BurgersBilinearForm::U_HAT, "du_hat.dat");
    backgroundFlow->writeFieldsToFile(BurgersBilinearForm::U,"u.m");
  }
  solution->solve(false);
  backgroundFlow->addSolution(solution,1.0);
  if (rank==0){
    solution->writeFieldsToFile(BurgersBilinearForm::U, "du.m");
    solution->writeFluxesToFile(BurgersBilinearForm::U_HAT, "du_hat.dat");
    backgroundFlow->writeFieldsToFile(BurgersBilinearForm::U,"u.m");
  }
  return 0;
  */

  int numNRSteps = 8;
  for (int i=0;i<numNRSteps;i++){
    solution->solve(false);
    if (rank==0){
      cout << "solved on NR iter " << i << endl;
      ostringstream filename;
      filename << "u" << i << ".m";
      backgroundFlow->writeFieldsToFile(BurgersBilinearForm::U, filename.str());
      filename.clear();filename.str("");
      filename << "du" << i << ".m";
      solution->writeFieldsToFile(BurgersBilinearForm::U, filename.str());
      filename.clear();filename.str("");
      filename << "du_hat" << i << ".dat";
      solution->writeFluxesToFile(BurgersBilinearForm::U_HAT, filename.str());

      filename.clear();filename.str("");
      filename << "sigma_x" << i << ".m";
      solution->writeFieldsToFile(BurgersBilinearForm::SIGMA_1, filename.str());
      filename.clear();filename.str("");
      filename << "sigma_y" << i << ".m";
      solution->writeFieldsToFile(BurgersBilinearForm::SIGMA_1, filename.str());
      filename.clear();filename.str("");
      filename << "sigma_hat" << i << ".dat";
      solution->writeFluxesToFile(BurgersBilinearForm::BETA_N_U_MINUS_SIGMA_HAT, filename.str());
    }
    backgroundFlow->addSolution(solution,1.0);
  }

  return 0;

  ////////////////////////////////////////////////////////////////////
 
  solution->solve();

  bool limitIrregularity = true;
  int numRefinements = 2;
  double thresholdFactor = 0.2;
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
    solution->solve();
    if (rank==0){
      cout << "Solved..." << endl;    
    }
  } 
  
  // save a data file for plotting in MATLAB
  if (rank==0){
    solution->writeFieldsToFile(BurgersBilinearForm::U, "u.m");
    solution->writeFieldsToFile(BurgersBilinearForm::SIGMA_1, "sigma_x.m");
    solution->writeFieldsToFile(BurgersBilinearForm::SIGMA_2, "sigma_y.m");
    solution->writeFluxesToFile(BurgersBilinearForm::U_HAT, "u_hat.dat");
    solution->writeFluxesToFile(BurgersBilinearForm::BETA_N_U_MINUS_SIGMA_HAT, "sigma_hat.dat");
    
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

    cout << "Done writing soln to file." << endl;
  }
  
  return 0;
}
