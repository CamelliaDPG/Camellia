#include "ConfusionManufacturedSolution.h"
#include "ConfusionBilinearForm.h"
#include "ConfusionProblem.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"

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

#include <ostream>

using namespace std;

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
#else
  Epetra_SerialComm Comm;
  int rank = 0;
  int numProcs = 1;
#endif
  // first, build a simple mesh
  for (int numFinalUniformRefinements=0; numFinalUniformRefinements<5; numFinalUniformRefinements++) {
    Epetra_Time timer(Comm);
    double wallTimeStart = timer.WallTime();
    
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
    mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
    mesh->setNumPartitions(numProcs);
    
    // repeatedly refine the first element along the side shared with cellID 1
    int numRefinements = 7;
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
    
    for (int i=0; i<numFinalUniformRefinements; i++) {
      cellsToRefine.clear();
      int numActiveElements = mesh->activeElements().size();
      for (int j=0; j<numActiveElements; j++) {
        cellsToRefine.push_back(mesh->activeElements()[j]->cellID());
      }
      mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    }

    double wallTimeForMeshConstruction = timer.WallTime() - wallTimeStart;
    if (rank==0) cout << "time to construct mesh: " << wallTimeForMeshConstruction << endl;
    if (rank==0) cout << "Mesh globalDofs: " << mesh->numGlobalDofs() << endl;
    
    // the following line should not be necessary, but if Solution's data structures aren't rebuilt properly, it might be...
    Solution solution = Solution(mesh, exactSolution.bc(), exactSolution.ExactSolution::rhs(), ip);
    solution.solve(false);
    //cout << "Processor " << rank << " returned from solve()." << endl;

    double refinedError = exactSolution.L2NormOfError(solution,ConfusionBilinearForm::U);
  
    double wallTimeTotal = timer.WallTime() - wallTimeStart;
    if (rank==0) {
      cout << "L2 error in 'deeply' refined fine mesh: " << refinedError << endl;
      ostringstream fileName;
      fileName << "scaling_stats_" << numFinalUniformRefinements << "_ref_" << numProcs << "_mpi_nodes.dat" ;
      solution.writeStatsToFile(fileName.str(), 4);
      // record total wall time
      fileName.str("");
      fileName << "wall_time_" << numFinalUniformRefinements << "_ref_" << numProcs << "_mpi_nodes.dat" ;
      ofstream fout(fileName.str().c_str());
      fout << setprecision(4);
      fout << "total wall time for " << numRefinements << " north/east refinements and ";
      fout << numFinalUniformRefinements << " uniform refinements (";
      fout << mesh->numGlobalDofs() << " total dofs): ";
      fout << wallTimeTotal << " seconds.\n";
      fileName.str("");
      fileName << "mesh_partitions_" << numFinalUniformRefinements << "_ref_" << numProcs << "_mpi_nodes.m" ;
      if (numFinalUniformRefinements==0) {
        // write mesh partitions to file
        mesh->writeMeshPartitionsToFile(fileName.str());
      }
    }
  }
}
