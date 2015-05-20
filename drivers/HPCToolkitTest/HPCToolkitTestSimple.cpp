#include <mpi.h>
#include <vector>

using namespace std;

int main(int argc, char *argv[]) {
  int rank;
  int nProc;
  
  int mpierr = 0;
  
  // Initialize MPI
  mpierr = ::MPI_Init(&argc, (char ***) argv);
  if (mpierr != 0) {
    cout << "GlobalMPISession(): Error, MPI_Init() returned error code="
    << mpierr << "!=0, calling std::terminate()!\n"
    << std::flush;
    std::terminate();
  }
  
  mpierr = ::MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  mpierr = ::MPI_Comm_size( MPI_COMM_WORLD, &nProc );
  
  cout << "MPI: started processor with rank " << rank << "!" << std::endl;
  
  MPI_Barrier( MPI_COMM_WORLD ); // barrier for debugger

  std::vector<int> partitionDofCounts(nProc);

  int myDofs = 0;
  if (rank==0)
  {
    myDofs = 300;
  }
  
  partitionDofCounts[rank] = myDofs;
  
  std::vector<int> partitionDofCountsCopy = partitionDofCounts;
  MPI_Allreduce(&partitionDofCountsCopy[0], &partitionDofCounts[0], partitionDofCounts.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  int partitionDofOffset = 0; // add this to a local partition dof index to get the global dof index
  for (int i=0; i<rank; i++) {
    partitionDofOffset += partitionDofCounts[i];
  }
  int globalDofCount = partitionDofOffset;
  for (int i=rank; i<nProc; i++) {
    globalDofCount += partitionDofCounts[i];
  }

  int activeCellCount = 1;
  std::vector<int> globalCellIDDofOffsets(activeCellCount);
  
  std::vector<int> globalCellIDDofOffsetsCopy = globalCellIDDofOffsets;
  // global copy:
  MPI_Allreduce(&globalCellIDDofOffsetsCopy[0], &globalCellIDDofOffsets[0], globalCellIDDofOffsets.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  if (rank==0)
  {
    cout << "globalCellIDOffsets[0] = " << globalCellIDDofOffsets[0];
  }
  return 0;
}
