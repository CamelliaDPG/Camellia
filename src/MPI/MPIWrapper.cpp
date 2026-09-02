//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  MPIWrapper.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 12/21/12.
//
//

#include "MPIWrapper.h"

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_Array.hpp"

using namespace Camellia;
using namespace std;

void MPIWrapper::allGather(Teuchos_CommPtr Comm, Camellia::FieldContainer<int> &allValues, int myValue)
{
  Camellia::FieldContainer<int> myValueFC(1);
  myValueFC[0] = myValue;
  MPIWrapper::allGather(Comm, allValues, myValueFC);
}

void MPIWrapper::allGather(Teuchos_CommPtr Comm, Camellia::FieldContainer<int> &allValues, Camellia::FieldContainer<int> &myValues)
{
  int numProcs = Comm->getSize();
  if (numProcs != allValues.dimension(0))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "allValues first dimension must be #procs");
  }
  if (allValues.size() / numProcs != myValues.size())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "myValues size invalid");
  }
#ifdef HAVE_MPI
  Teuchos::gatherAll(*Comm, allValues.size()/numProcs, &myValues[0], allValues.size(), &allValues[0]);
#else
#endif
}

Teuchos::RCP<MPIWrapperComm> MPIWrapper::CommSerial()
{
  static Teuchos::RCP<MPIWrapperComm> Comm = Teuchos::rcp(new MPIWrapperComm(TeuchosCommSerial()));
  return Comm;
}

Teuchos::RCP<MPIWrapperComm> MPIWrapper::CommWorld()
{
  static Teuchos::RCP<MPIWrapperComm> Comm = Teuchos::rcp(new MPIWrapperComm(TeuchosCommWorld()));
  return Comm;
}

int MPIWrapper::rank()
{
  return Teuchos::GlobalMPISession::getRank();
}

template<typename ScalarType>
void MPIWrapper::entryWiseSum(Camellia::FieldContainer<ScalarType> &values)
{
  entryWiseSum<ScalarType>(TeuchosCommWorld(), values);
}

void MPIWrapper::entryWiseSum(Camellia::FieldContainer<double> &values)   // sums values entry-wise across all processors
{
  entryWiseSum<double>(values);
}

void MPIWrapper::entryWiseSum(Teuchos_CommPtr Comm, std::vector<int> &values)
{
  entryWiseSum<int>(Comm, values);
}

void MPIWrapper::entryWiseSum(Teuchos_CommPtr Comm, std::vector<long> &values)
{
  entryWiseSum<long>(Comm, values);
}

void MPIWrapper::entryWiseSum(Teuchos_CommPtr Comm, std::vector<long long> &values)
{
  entryWiseSum<long long>(Comm, values);
}

void MPIWrapper::entryWiseSum(Teuchos_CommPtr Comm, std::vector<double> &values)
{
  entryWiseSum<double>(Comm, values);
}

bool MPIWrapper::globalAnd(Teuchos_CommPtr Comm, bool value)
{
  int myInt = value ? 0 : 1;
  return 0 == MPIWrapper::sum(Comm, myInt);
}

bool MPIWrapper::globalOr(Teuchos_CommPtr Comm, bool value)
{
  return !globalAnd(Comm, !value);
}

Teuchos::RCP<Tpetra::Distributor> MPIWrapper::getDistributor(Teuchos_CommPtr Comm)
{
  Teuchos::RCP<Tpetra::Distributor> distributor = Teuchos::rcp( new Tpetra::Distributor(Comm) );
  return distributor;
}

double MPIWrapper::scanSum(Teuchos_CommPtr Comm, double myValue)
{
  double sum;
  Teuchos::scan(*Comm, Teuchos::REDUCE_SUM, 1, &myValue, &sum);
  return sum;
}

int MPIWrapper::scanSum(Teuchos_CommPtr Comm, int myValue)
{
  int sum;
  Teuchos::scan(*Comm, Teuchos::REDUCE_SUM, 1, &myValue, &sum);
  return sum;
}

// sum the contents of valuesToSum across all processors, and returns the result:
// (valuesToSum may vary in length across processors)
double MPIWrapper::sum(const Camellia::FieldContainer<double> &valuesToSum)
{
  // this is fairly inefficient in the sense that the MPI overhead will dominate the cost here.
  // insofar as it's possible to group such calls into entryWiseSum() calls, this is preferred.
  double mySum = 0;
  for (int i=0; i<valuesToSum.size(); i++)
  {
    mySum += valuesToSum[i];
  }

  return sum(mySum);
}

double MPIWrapper::sum(Teuchos_CommPtr Comm, double mySum)
{
  double mySumCopy = mySum;
  Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &mySumCopy, &mySum);
  return mySum;
}

double MPIWrapper::sum(double mySum)
{
  return sum(TeuchosCommWorld(), mySum);
}

void MPIWrapper::entryWiseSum(Camellia::FieldContainer<int> &values)
{
  entryWiseSum<int>(TeuchosCommWorld(), values);
}

// sum the contents of valuesToSum across all processors, and returns the result:
// (valuesToSum may vary in length across processors)
int MPIWrapper::sum(const Camellia::FieldContainer<int> &valuesToSum)
{
  // this is fairly inefficient in the sense that the MPI overhead will dominate the cost here.
  // insofar as it's possible to group such calls into entryWiseSum() calls, this is preferred.
  int mySum = 0;
  for (int i=0; i<valuesToSum.size(); i++)
  {
    mySum += valuesToSum[i];
  }

  return sum(mySum);
}

int MPIWrapper::sum(Teuchos_CommPtr Comm, int mySum)
{
  int mySumCopy = mySum;
  Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &mySumCopy, &mySum);
  return mySum;
}

int MPIWrapper::sum(int mySum)
{
  return sum(TeuchosCommWorld(), mySum);
}

Teuchos_CommPtr MPIWrapper::TeuchosCommSerial()
{
  static Teuchos_CommPtr Comm = Teuchos::rcp( new Teuchos::SerialComm<int>() );
  return Comm;
}

Teuchos_CommPtr MPIWrapper::TeuchosCommWorld()
{
#ifdef HAVE_MPI
  static Teuchos_CommPtr Comm = Teuchos::rcp( new Teuchos::MpiComm<int> (MPI_COMM_WORLD) );
#else
  static Teuchos_CommPtr Comm = Teuchos::rcp( new Teuchos::SerialComm<int>() );
#endif
  return Comm;
}
