#include "ConfusionBilinearForm.h"
#include "ConfusionProblem.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"
#include "RefinementStrategy.h"

// added by Jesse
#include "PenaltyMethodFilter.h"
#include "ConfectionProblem.h"
#include "EricksonProblem.h"
#include "ConfectionManufacturedSolution.h"
#include "ConfusionInnerProduct.h"
#include "EricksonManufacturedSolution.h"
#include "EricksonConfectionSolution.h" // discontinuous hat
#include "ZeroFunction.h"

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
  int pToAdd = 5; // for tests  

  // define our manufactured solution or problem bilinear form: 
  double epsilon = .1;
  if (argc > 1){
    epsilon = atof(argv[1]);
    cout << "eps set to " << epsilon << endl;
  }
  int numRefinements = 3;
  if (argc > 2){
    numRefinements = atoi(argv[2]);
    cout << "num refinements = " << numRefinements << endl;
  }

  double beta_x = 1.0, beta_y = 0.0;
  bool useTriangles = false;
  Teuchos::RCP<ConfusionBilinearForm> bf = Teuchos::rcp(new ConfusionBilinearForm(epsilon,beta_x,beta_y));
  Teuchos::RCP<EricksonManufacturedSolution> exactSolution = Teuchos::rcp(new EricksonManufacturedSolution(epsilon,beta_x,beta_y));
  //  Teuchos::RCP<EricksonConfectionSolution> exactSolution = Teuchos::rcp(new EricksonConfectionSolution(epsilon,beta_x,beta_y));
  bool useExactSolution = true;
  int cubDegree = 20;

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
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh;
  mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, bf, H1Order, H1Order+pToAdd, useTriangles);
  //  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));

  // define our inner product:
  Teuchos::RCP<ConfusionInnerProduct> ip = Teuchos::rcp( new ConfusionInnerProduct( bf, mesh ) );
  //  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new OptimalInnerProduct( bf ) );
  //  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct( bf ) );

  // create a solution object
  Teuchos::RCP<EricksonProblem> problem = Teuchos::rcp( new EricksonProblem(bf) );
  Teuchos::RCP<LocalStiffnessMatrixFilter> penaltyBC;
  Teuchos::RCP<Solution> solution;
  Teuchos::RCP<Solution> projectedSolution;
  if (useExactSolution){
    solution = Teuchos::rcp(new Solution(mesh, exactSolution->bc(), exactSolution->ExactSolution::rhs(), ip));
    penaltyBC= Teuchos::rcp(new PenaltyMethodFilter(exactSolution));
    projectedSolution = Teuchos::rcp(new Solution(mesh, exactSolution->bc(), exactSolution->ExactSolution::rhs(), ip));
  } else {
    solution = Teuchos::rcp(new Solution(mesh, problem, problem, ip));
    penaltyBC = Teuchos::rcp(new PenaltyMethodFilter(problem));
  }

  solution->setFilter(penaltyBC);

  mesh->registerSolution(solution);

  // also compute L2 projection of solution if exact solution given
  map<int, Teuchos::RCP<AbstractFunction> > functionMap;
  if (useExactSolution){
    mesh->registerSolution(projectedSolution);
    functionMap[ConfusionBilinearForm::U] = exactSolution;
    functionMap[ConfusionBilinearForm::SIGMA_1] = Teuchos::rcp(new ZeroFunction());
    functionMap[ConfusionBilinearForm::SIGMA_2] = Teuchos::rcp(new ZeroFunction());  
    projectedSolution->projectOntoMesh(functionMap);
  }

  // define refinement strategy:
  double energyThreshold = .2;
  Teuchos::RCP<RefinementStrategy> refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));

  solution->solve(false);

  bool limitIrregularity = true;
  int refIterCount = 0;  
  vector<double> errorVector;
  vector<double> L2errorVector;
  vector<double> uL2errorVector;
  vector<double> projErrorVector;
  vector<int> dofVector;
  double l2error;
  for (int i=0; i<numRefinements; i++) {
    double totalEnergyError = solution->energyErrorTotal();
    errorVector.push_back(totalEnergyError);
    dofVector.push_back(mesh->numGlobalDofs());

    if (useExactSolution){
      //      projectedSolution->addSolution(solution,-1.0); // subtract solution from projection
      //      double projError = projectedSolution->L2NormOfSolutionGlobal(ConfusionBilinearForm::U);
      double u_proj_error = exactSolution->L2NormOfError(*projectedSolution, ConfusionBilinearForm::U,cubDegree);
      projErrorVector.push_back(u_proj_error);      
      if (rank==0){
	cout << "Best approximation error: " << u_proj_error << endl;      
      }

      // print out the L2 error of the solution:
      double u_error = exactSolution->L2NormOfError(*solution, ConfusionBilinearForm::U,cubDegree);
      double s1_error = exactSolution->L2NormOfError(*solution, ConfusionBilinearForm::SIGMA_1,cubDegree);
      double s2_error = exactSolution->L2NormOfError(*solution, ConfusionBilinearForm::SIGMA_2,cubDegree);
      l2error = u_error*u_error + s1_error*s1_error + s2_error*s2_error;
      l2error = sqrt(l2error);
      L2errorVector.push_back(l2error);
      uL2errorVector.push_back(u_error);

      if (rank==0){
	cout << "L2 error: total = " << l2error << ", l2/err ratio " << u_error/totalEnergyError << ", proj ratio = " << u_proj_error/u_error << ", in u = " << u_error << ", in sigma1,2 = " << s1_error << ", " << s2_error << endl;
      }      
    }

    refinementStrategy->refine(rank==0);
    
    refIterCount++;
    if (rank==0){
      cout << "Solving on refinement iteration number " << refIterCount << "..." << endl;    
    }
    if (useExactSolution){
      projectedSolution->projectOntoMesh(functionMap);
    }
    solution->solve();
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
    
    string epsString("NoWeight");
    string energyErrorFile("errors");
    energyErrorFile+=epsString + string(".dat");
    ofstream fout1(energyErrorFile.c_str());
    fout1 << setprecision(15);
    for (int i = 0;i<errorVector.size();i++){
      fout1 << errorVector[i] << endl;
    }
    fout1.close();

    string dofFile("dofs");
    dofFile+=epsString + string(".dat");
    ofstream fout2(dofFile.c_str());
    for (int i = 0;i<dofVector.size();i++){
      fout2 << dofVector[i] << endl;
    }
    fout2.close();

    if (useExactSolution){
      projectedSolution->writeFieldsToFile(ConfusionBilinearForm::U, "projectedU.m");
      projectedSolution->addSolution(solution,-1.0);
      projectedSolution->writeFieldsToFile(ConfusionBilinearForm::U, "pointdiff.m");

      string L2errorFile("L2errors");      
      L2errorFile += epsString + string(".dat");
      ofstream fout3(L2errorFile.c_str());
      fout3 << setprecision(15);
      for (int i = 0;i<L2errorVector.size();i++){
	fout3 << L2errorVector[i] << endl;
      }
      fout3.close();

      string projErrorFile("projErrors");      
      projErrorFile += epsString + string(".dat");
      ofstream fout4(projErrorFile.c_str());
      fout4 << setprecision(15);
      for (int i = 0;i<projErrorVector.size();i++){
	fout4 << projErrorVector[i] << endl;
      }
      fout4.close();

      string uL2errorFile("uL2errors");      
      uL2errorFile += epsString + string(".dat");
      ofstream fout5(uL2errorFile.c_str());
      fout5 << setprecision(15);
      for (int i = 0;i<uL2errorVector.size();i++){
	fout5 << uL2errorVector[i] << endl;
      }
      fout5.close();


    }

    cout << "Done writing soln to file." << endl;
  }
  
  return 0;
}
