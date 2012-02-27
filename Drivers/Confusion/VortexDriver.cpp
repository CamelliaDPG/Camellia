#include "ConfusionBilinearForm.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"
#include "RefinementStrategy.h"

// added by Jesse
#include "PenaltyMethodFilter.h"
//#include "ConfectionProblem.h"
#include "VortexProblem.h"
#include "ConfectionManufacturedSolution.h"
#include "VortexManufacturedSolution.h"

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
  int pToAdd = 3; // for tests
  
  // define our manufactured solution or problem bilinear form:
  double epsilon = 1e-4;
  bool useTriangles = false;
  Teuchos::RCP<ConfusionBilinearForm> bf = Teuchos::rcp(new ConfusionBilinearForm(epsilon));
  Teuchos::RCP<VortexManufacturedSolution> exactSolution = Teuchos::rcp(new VortexManufacturedSolution(epsilon));
  bool useExactSolution =true;

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

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, bf, H1Order, H1Order+pToAdd, useTriangles);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));

  // define our inner product:
  //  Teuchos::RCP<ConfusionInnerProduct> ip = Teuchos::rcp( new ConfusionInnerProduct( bf, mesh ) );
  Teuchos::RCP<MathInnerProduct> ip = Teuchos::rcp( new MathInnerProduct( bf ) );

  // create a solution object
  Teuchos::RCP<Solution> solution;
  Teuchos::RCP<LocalStiffnessMatrixFilter> penaltyBC;
  if (useExactSolution){
    solution = Teuchos::rcp(new Solution(mesh, exactSolution->bc(), exactSolution->ExactSolution::rhs(), ip));
    penaltyBC= Teuchos::rcp(new PenaltyMethodFilter(exactSolution));
  } else {
    Teuchos::RCP<VortexProblem> problem = Teuchos::rcp( new VortexProblem(bf) );
    solution = Teuchos::rcp(new Solution(mesh, problem, problem, ip));
    penaltyBC = Teuchos::rcp(new PenaltyMethodFilter(problem));
  }
  solution->setFilter(penaltyBC);

  ////////////////////////////////////////////////////////////////////

  // define refinement strategy:
  double energyThreshold = .2;
  Teuchos::RCP<RefinementStrategy> refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));

  solution->solve(false);
  cout << "Processor " << rank << " returned from solve()." << endl;
  if (rank==0){
    solution->writeFieldsToFile(ConfusionBilinearForm::U, "u.m");
    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "u_hat.dat");
    solution->writeFluxesToFile(ConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT, "sigma_hat.dat");
  }
  double l2error = exactSolution->L2NormOfError(*solution, ConfusionBilinearForm::U,15);
  cout << " L2 error: " << l2error << endl;

  bool limitIrregularity = true;
  int numRefinements = 2;
  double thresholdFactor = 0.2;
  int refIterCount = 0;  
  vector<double> errorVector;
  vector<double> L2errorVector;
  vector<int> dofVector;
  for (int i=0; i<numRefinements; i++) {

    double totalEnergyError = solution->energyErrorTotal();

    if (useExactSolution){
      // print out the L2 error of the solution:
      int cubDegree = 20;
      l2error = exactSolution->L2NormOfError(*solution, ConfusionBilinearForm::U,cubDegree);
      L2errorVector.push_back(l2error);
    }

    if (rank==0){
      if (useExactSolution) {
	cout << "L2 error: " << l2error << endl;
      }      	
    }
    errorVector.push_back(totalEnergyError);
    dofVector.push_back(mesh->numGlobalDofs());
    

    refinementStrategy->refine(rank==0);

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
    solution->writeFieldsToFile(ConfusionBilinearForm::U, "u.m");
    solution->writeFieldsToFile(ConfusionBilinearForm::SIGMA_1, "sigma_x.m");
    solution->writeFieldsToFile(ConfusionBilinearForm::SIGMA_2, "sigma_y.m");
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

    cout << "Done writing soln to file." << endl;

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
    }

  }
  
  return 0;
}
