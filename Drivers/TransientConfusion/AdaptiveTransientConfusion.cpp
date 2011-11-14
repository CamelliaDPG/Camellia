#include "TransientConfusionBilinearForm.h"
#include "TransientConfusionProblem.h"
#include "ConfusionProblemFirstTimestep.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"

// Trilinos includes
#include "Intrepid_FieldContainer.hpp"

using namespace std;

int main(int argc, char *argv[]) {
  int polyOrder = 3;
  int pToAdd = 2; // for tests
  
  // define our problem (that'll define the solution)
  double epsilon = 1e-1;
  double beta_x = 1.0, beta_y = 2.0;
  double dt = .1; 
  bool useTriangles = false;
  Teuchos::RCP<TransientConfusionBilinearForm> bilinearForm = Teuchos::rcp(new TransientConfusionBilinearForm(epsilon,beta_x,beta_y,dt));
  
  bilinearForm->printTrialTestInteractions();
  
  // define our inner product:
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new OptimalInnerProduct( bilinearForm ) );
  //Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct( bilinearForm ) );
  
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
  int horizontalCells = 8, verticalCells = 8;
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, bilinearForm, H1Order, H1Order+pToAdd, useTriangles);

  // create a solution object
  Teuchos::RCP<ConfusionProblemFirstTimestep> initial_problem = Teuchos::rcp( new ConfusionProblemFirstTimestep(bilinearForm));
  Teuchos::RCP<Solution> previousTimeSolution = Teuchos::rcp(new Solution(mesh, initial_problem, initial_problem, ip)); 

  // solve first timestep
  double T;  
  previousTimeSolution->solve(); 
  previousTimeSolution->writeFluxesToFile(TransientConfusionBilinearForm::U_HAT, "Confusion_u_t0.dat");
  T = bilinearForm->increment_T();
  cout << "time is T = " << T << endl;

  // solve additional timesteps
  Teuchos::RCP<TransientConfusionProblem> problem = Teuchos::rcp( new TransientConfusionProblem(bilinearForm,previousTimeSolution));
  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, problem, problem, ip));    

  double Tend = 0*dt;
  while (T<Tend){
    // solve
    solution->solve(); 
    T = bilinearForm->increment_T();
  }

  // save a data file for plotting in MATLAB
  solution->writeToFile(TransientConfusionBilinearForm::U, "Confusion_u.dat");
  solution->writeFluxesToFile(TransientConfusionBilinearForm::U_HAT, "Confusion_u_hat.dat");
  solution->writeFluxesToFile(TransientConfusionBilinearForm::BETA_N_U_MINUS_SIGMA_HAT, "Confusion_beta_n_hat.dat");

  cout << "Done solving and writing" << endl;

  
}
