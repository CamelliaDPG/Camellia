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

#include "Teuchos_GlobalMPISession.hpp"

void MPIWrapper::allGather(FieldContainer<int> &allValues, int myValue) {
  FieldContainer<int> myValueFC(1);
  myValueFC[0] = myValue;
  MPIWrapper::allGather(allValues, myValueFC);
}

void MPIWrapper::allGather(FieldContainer<int> &allValues, FieldContainer<int> &myValues) {
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  if (numProcs != allValues.dimension(0)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "allValues first dimension must be #procs");
  }
  if (allValues.size() / numProcs != myValues.size()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "myValues size invalid");
  }
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Comm.GatherAll(&myValues[0], &allValues[0], allValues.size()/numProcs);
#else
#endif
}

// sum the contents of inValues across all processors, and stores the result in outValues
// the rank of outValues determines the nature of the sum:
// if outValues has dimensions (D1,D2,D3), say, then inValues must agree in the first three dimensions,
// but may be of arbitrary shape beyond that.  All values on all processors with matching address
// (d1,d2,d3) will be summed and stored in outValues(d1,d2,d3).
//void MPIWrapper::entryWiseSum(FieldContainer<double> &outValues, const FieldContainer<double> &inValues) {
//  outValues.initialize();
//  int outRank = outValues.rank();
//  for (int i=0; i<outRank; i++) {
//    TEUCHOS_TEST_FOR_EXCEPTION(outValues.dimension(i) != inValues.dimension(i), std::invalid_argument, "inValues must match outValues in all outValues's dimensions");
//  }
//  double inEntriesPerOutEntry = 1;
//  for (int i=outRank; i<inValues.rank(); i++) {
//    inEntriesPerOutEntry *= inValues.dimension(i);
//  }
//  
//}

void MPIWrapper::entryWiseSum(FieldContainer<double> &values) { // sums values entry-wise across all processors
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  FieldContainer<double> valuesCopy = values; // it appears this copy is necessary
  Comm.SumAll(&valuesCopy[0], &values[0], values.size());
#else
#endif
}
// sum the contents of valuesToSum across all processors, and returns the result:
// (valuesToSum may vary in length across processors)
double MPIWrapper::sum(const FieldContainer<double> &valuesToSum) {
  // this is fairly inefficient in the sense that the MPI overhead will dominate the cost here.
  // insofar as it's possible to group such calls into entryWiseSum() calls, this is preferred.
  double mySum = 0;
  for (int i=0; i<valuesToSum.size(); i++) {
    mySum += valuesToSum[i];
  }
  
  return sum(mySum);
}

double MPIWrapper::sum(double mySum) {
#ifdef HAVE_MPI
  double mySumCopy = mySum;
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Comm.SumAll(&mySumCopy, &mySum, 1);
#else
#endif
  return mySum;
}

void MPIWrapper::entryWiseSum(FieldContainer<int> &values) {
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  FieldContainer<int> valuesCopy = values; // it appears this copy is necessary
  Comm.SumAll(&valuesCopy[0], &values[0], values.size());
#else
#endif
}
// sum the contents of valuesToSum across all processors, and returns the result:
// (valuesToSum may vary in length across processors)
int MPIWrapper::sum(const FieldContainer<int> &valuesToSum) {
  // this is fairly inefficient in the sense that the MPI overhead will dominate the cost here.
  // insofar as it's possible to group such calls into entryWiseSum() calls, this is preferred.
  int mySum = 0;
  for (int i=0; i<valuesToSum.size(); i++) {
    mySum += valuesToSum[i];
  }
  
  return sum(mySum);
}

int MPIWrapper::sum(int mySum) {
#ifdef HAVE_MPI
  int mySumCopy = mySum;
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Comm.SumAll(&mySumCopy, &mySum, 1);
  
#else
#endif
  return mySum;
}