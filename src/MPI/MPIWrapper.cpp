//
//  MPIWrapper.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 12/21/12.
//
//

#include "MPIWrapper.h"

// MPI includes
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

// sum the contents of inValues across all processors, and stores the result in outValues
// the rank of outValues determines the nature of the sum:
// if outValues has dimensions (D1,D2,D3), say, then inValues must agree in the first three dimensions,
// but may be of arbitrary shape beyond that.  All values on all processors with matching address
// (d1,d2,d3) will be summed and stored in outValues(d1,d2,d3).
void MPIWrapper::elementWiseSum(FieldContainer<double> &outValues, const FieldContainer<double> &inValues) {
  
}

void MPIWrapper::elementWiseSum(FieldContainer<double> &values) { // sums values element-wise across all processors
  
}
// sum the contents of valuesToSum across all processors, and returns the result:
// (valuesToSum may vary in length across processors)
double MPIWrapper::sum(const FieldContainer<double> &valuesToSum) {
  // this is fairly inefficient in the sense that the MPI overhead will dominate the cost here.
  // insofar as it's possible to group such calls into elementWiseSum() calls, this is preferred.
  double mySum = 0;
  for (int i=0; i<valuesToSum.size(); i++) {
    mySum += valuesToSum[i];
  }
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Comm.SumAll(&mySum, &mySum, 1);
#else
#endif
  return mySum;
}