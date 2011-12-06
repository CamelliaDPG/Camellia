#include "ConfusionManufacturedSolution.h"
#include "ConfusionBilinearForm.h"
#include "ConfusionProblem.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"

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
  // first, build a simple mesh
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  // h-convergence
  int sqrtElements = 2;
  
  double epsilon = 1e-2;
  double beta_x = 1.0, beta_y = 1.0;
  ConfusionManufacturedSolution exactSolution(epsilon,beta_x,beta_y); // 0 doesn't mean constant, but a particular solution...
  
  int pOrder = 3;
  int H1Order = pOrder+1; int pToAdd = 2;
  int horizontalCells = 1; int verticalCells = 1;
  
  // before we hRefine, compute a solution for comparison after refinement
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(exactSolution.bilinearForm()));
  
  vector<int> cellsToRefine;
  cellsToRefine.clear();
  
  // start with a fresh 2x1 mesh:
  horizontalCells = 1; verticalCells = 1;
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, exactSolution.bilinearForm(), H1Order, H1Order+pToAdd);
  
  // repeatedly refine the first element along the side shared with cellID 1
  int numRefinements = 9;
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

  if (rank==0) cout << "Mesh globalDofs: " << mesh->numGlobalDofs() << endl;
  
  // the following line should not be necessary, but if Solution's data structures aren't rebuilt properly, it might be...
  Solution solution = Solution(mesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
  solution.solve(false);
  //cout << "Processor " << rank << " returned from solve()." << endl;

  double refinedError = exactSolution.L2NormOfError(solution,ConfusionBilinearForm::U);
  
  if (rank==0)
    cout << "L2 error in 'deeply' refined fine mesh: " << refinedError << endl;
  
  if (rank==0)
    solution.writeStatsToFile("scaling_stats.dat");
}
