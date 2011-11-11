#include "PoissonExactSolution.h"
#include "PoissonBilinearForm.h"
#include "MathInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"

// Trilinos includes
#include "Intrepid_FieldContainer.hpp"

using namespace std;

int main(int argc, char *argv[]) {
  int polyOrder = 4;
  int pToAdd = 2; // for tests
  bool useConformingTraces = true;
  
  // define our manufactured solution:
  Teuchos::RCP<PoissonExactSolution> exactSolution = 
  Teuchos::rcp( new PoissonExactSolution(PoissonExactSolution::POLYNOMIAL, 
                                         polyOrder, useConformingTraces) );
  exactSolution->setUseSinglePointBCForPHI(true); // 

  // define our inner product:
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct( exactSolution->bilinearForm() ) );
  
  // define vertices for a pentagonal domain:
  vector< FieldContainer<double> > vertices;
  int spaceDim = 2;
  FieldContainer<double> vertex(spaceDim);
  vertex(0) = 0.0; vertex(1) = 0.0;
  vertices.push_back(vertex);
  vertex(0) = 1.0; vertex(1) = 0.0;
  vertices.push_back(vertex);
  vertex(0) = 1.25; vertex(1) = 1.0;
  vertices.push_back(vertex);
  vertex(0) = 0.50; vertex(1) = 1.25;
  vertices.push_back(vertex);
  vertex(0) = -0.25; vertex(1) = 1.0;
  vertices.push_back(vertex);
  
  // now define two elements, a triangle and a quad:
  // (we use the indices in vertices vector to define the elements)
  vector<int> triangle;
  triangle.push_back(0);
  triangle.push_back(1);
  triangle.push_back(2);
  
  vector<int> quad;
  quad.push_back(0);
  quad.push_back(2);
  quad.push_back(3);
  quad.push_back(4);
  
  vector< vector<int> > elements;
  elements.push_back(triangle);
  elements.push_back(quad);
  
  int H1Order = polyOrder + 1;
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Teuchos::rcp(new Mesh(vertices, elements, 
                                                  exactSolution->bilinearForm(), H1Order, pToAdd));
  // create a solution object
  Solution solution(mesh, exactSolution->bc(), exactSolution->ExactSolution::rhs(), ip);

  solution.solve();
  
  // print out the L2 error of the solution:
  double l2error = exactSolution->L2NormOfError(solution, PoissonBilinearForm::PHI);
  cout << "L2 error: " << l2error << endl;
  
  // save a data file for plotting in MATLAB
  solution.writeToFile(PoissonBilinearForm::PHI, "PoissonPentagon_phi.dat");
  
}