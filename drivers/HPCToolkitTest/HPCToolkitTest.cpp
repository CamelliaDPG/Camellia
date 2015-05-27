#include "MPIWrapper.h"
#include "TypeDefs.h"

// Epetra includes
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include <Intrepid_FieldContainer.hpp>
#include <Teuchos_GlobalMPISession.hpp>

using namespace Camellia;
using namespace Intrepid;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  int rank = Teuchos::GlobalMPISession::getRank();

#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif

  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  int numRanks = Teuchos::GlobalMPISession::getNProc();
  FieldContainer<IndexType> partitionDofCounts(256);

  int myDofs = 0;
  if (rank==0)
  {
    myDofs = 300;
  }

  partitionDofCounts[rank] = myDofs;
  MPIWrapper::entryWiseSum(partitionDofCounts);

  int partitionDofOffset = 0; // add this to a local partition dof index to get the global dof index
  for (int i=0; i<rank; i++)
  {
    partitionDofOffset += partitionDofCounts[i];
  }
  int globalDofCount = partitionDofOffset;
  for (int i=rank; i<numRanks; i++)
  {
    globalDofCount += partitionDofCounts[i];
  }

  int activeCellCount = 1;
  Intrepid::FieldContainer<int> globalCellIDDofOffsets(activeCellCount);

  // global copy:
  MPIWrapper::entryWiseSum(globalCellIDDofOffsets);

  return 0;
}
