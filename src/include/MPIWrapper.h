// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//  MPIWrapper.h
//  Camellia
//
//  Created by Nathan Roberts on 12/21/12.
//
//

#ifndef __Camellia__MPIWrapper__
#define __Camellia__MPIWrapper__

#include "TypeDefs.h"

#include <iostream>

// MPI includes
#ifdef HAVE_MPI
#include "Teuchos_DefaultMpiComm.hpp"
#endif

#include "Teuchos_Comm.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_DefaultSerialComm.hpp"

#include <limits>

#include "Camellia_FieldContainer.hpp"
#include "Camellia_TeuchosSerializers.hpp"

#include <Tpetra_Distributor.hpp>

namespace Camellia
{
class MPIWrapperComm
{
  Teuchos_CommPtr _comm;
public:
  MPIWrapperComm(Teuchos_CommPtr comm) : _comm(comm) {}
  void Barrier() const { _comm->barrier(); }
  int MyPID() const { return _comm->getRank(); }
  int NumProc() const { return _comm->getSize(); }
  Teuchos_CommPtr TeuchosComm() const { return _comm; }
};

// static class to provide a FieldContainer-based interface to some common MPI tasks
// (Can be used even with MPI disabled)
class MPIWrapper
{
public:
  template<typename Scalar>
  static void allGather(Teuchos_CommPtr Comm, std::vector<Scalar> &allValues, Scalar myValue);
  
  template<typename Scalar>
  static void allGather(Teuchos_CommPtr Comm, std::vector<Scalar> &allValues, std::vector<Scalar> myValues);
  
  static void allGather(Teuchos_CommPtr Comm, Camellia::FieldContainer<int> &allValues, int myValue);
  static void allGather(Teuchos_CommPtr Comm, Camellia::FieldContainer<int> &values, Camellia::FieldContainer<int> &myValues); // assumes myValues is same size on every proc.

  template<typename DataType>
  static void allGatherVariable(Teuchos_CommPtr Comm,
                                std::vector<DataType> &gatheredValues,
                                std::vector<DataType> &myValues,
                                std::vector<int> &offsets);
  
  template<typename Scalar>
  static void allGatherVariable(Teuchos_CommPtr Comm,
                                Camellia::FieldContainer<Scalar> &gatheredValues,
                                Camellia::FieldContainer<Scalar> &myValues,
                                Camellia::FieldContainer<int> &offsets);
  
  static Teuchos::RCP<Tpetra::Distributor> getDistributor(Teuchos_CommPtr Comm);
  
  static Teuchos::RCP<MPIWrapperComm> CommSerial();
  static Teuchos::RCP<MPIWrapperComm> CommWorld();
  
  static Teuchos_CommPtr TeuchosCommSerial();
  static Teuchos_CommPtr TeuchosCommWorld();
  
  static int rank();

  //! sum values entry-wise across all processors in Comm
  template<typename ScalarType>
  static void entryWiseSum(Teuchos_CommPtr Comm, Camellia::FieldContainer<ScalarType> &values);
  
  template<typename ScalarType>
  static void entryWiseSum(Teuchos_CommPtr Comm, std::vector<ScalarType> &values);
  
  //! sum values entry-wise across all processors
  template<typename ScalarType>
  static void entryWiseSum(Camellia::FieldContainer<ScalarType> &values);
  
  //! sum values entry-wise across all processors
  template<typename ScalarType>
  static void entryWiseSum(std::vector<ScalarType> &values);
  
  //! sums values entry-wise across all processors
  static void entryWiseSum(Camellia::FieldContainer<double> &values);
  
  static void entryWiseSum(Teuchos_CommPtr Comm, std::vector<int> &values);
  static void entryWiseSum(Teuchos_CommPtr Comm, std::vector<long> &values);
  static void entryWiseSum(Teuchos_CommPtr Comm, std::vector<long long> &values);
  static void entryWiseSum(Teuchos_CommPtr Comm, std::vector<double> &values);
  
  static bool globalAnd(Teuchos_CommPtr Comm, bool value);
  static bool globalOr (Teuchos_CommPtr Comm, bool value);
  
  template<typename Scalar>
  static Scalar max(Teuchos_CommPtr Comm, Scalar value);
  
  template<typename Scalar>
  static Scalar min(Teuchos_CommPtr Comm, Scalar value);
    
  template<typename DataType>
  static void sendDataVectors(Teuchos_CommPtr Comm, const std::vector<int> &recipients,
                              const std::vector<std::vector<DataType>> &dataVectors,
                              std::vector<DataType> &receivedVector);
  template<typename DataType>
  static void sendDataVectors(Teuchos_CommPtr Comm,
                              const std::map<int,std::vector<DataType>> &recipientDataVectors,
                              std::vector<DataType> &receivedVector);
  
  template<typename KeyType, typename ValueType>
  static void sendDataMaps(Teuchos_CommPtr Comm,
                           const std::map<int,std::map<KeyType,ValueType>> &recipientDataMaps,
                           std::map<KeyType,ValueType> &receivedMap);
  
  static double scanSum(Teuchos_CommPtr Comm, double myValue);
  static int    scanSum(Teuchos_CommPtr Comm,    int myValue);
  
  // sum the contents of valuesToSum across all processors, and returns the result:
  // (valuesToSum may vary in length across processors)
  static double sum(const Camellia::FieldContainer<double> &valuesToSum);
  static double sum(double myValue);
  static double sum(Teuchos_CommPtr Comm, double mySum);
  
  static void entryWiseSum(Camellia::FieldContainer<int> &values); // sums values entry-wise across all processors
  // sum the contents of valuesToSum across all processors, and returns the result:
  // (valuesToSum may vary in length across processors)
  static int sum(const Camellia::FieldContainer<int> &valuesToSum);
  static int sum(int myValue);
  static int sum(Teuchos_CommPtr Comm, int mySum);
  
  //! sum value across communicator
  template<typename ScalarType>
  static ScalarType sum(Teuchos_CommPtr Comm, ScalarType mySum);
};

  template<typename Scalar>
  void MPIWrapper::allGather(Teuchos_CommPtr Comm, std::vector<Scalar> &allValues, Scalar myValue)
  {
    std::vector<Scalar> myValueVector(1);
    myValueVector[0] = myValue;
    MPIWrapper::allGather(Comm, allValues, myValueVector);
  }

  template<typename Scalar>
  void MPIWrapper::allGather(Teuchos_CommPtr Comm, std::vector<Scalar> &allValues, std::vector<Scalar> myValues)
  {
    int numProcs = Comm->getSize();
    if (allValues.size() / numProcs != myValues.size())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "myValues size invalid");
    }
  #ifdef HAVE_MPI
    if (numProcs > 1)
    {
      int byteCount = myValues.size() * sizeof(Scalar);
      MPI_Comm mpiComm = *Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(Comm)->getRawMpiComm();
      MPI_Allgather(myValues.data(), byteCount, MPI_BYTE, allValues.data(), byteCount, MPI_BYTE, mpiComm);
    }
    else
    {
      allValues = myValues;
    }
  //    Comm.GatherAll(&myValues[0], &allValues[0], allValues.size()/numProcs);
  #else
  #endif
  }

  //! sum values entry-wise across all processors
  template<typename ScalarType>
  void MPIWrapper::entryWiseSum(std::vector<ScalarType> &values)
  {
    entryWiseSum<ScalarType>(TeuchosCommWorld(), values);
  }
  
  template<typename ScalarType>
  void MPIWrapper::entryWiseSum(Teuchos_CommPtr Comm, std::vector<ScalarType> &values)
  {
    std::vector<ScalarType> valuesCopy = values; // it appears this copy is necessary
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, static_cast<int>(values.size()), valuesCopy.data(), values.data());
  }
  
  //! sum values entry-wise across all processors in Comm
  template<typename ScalarType>
  void MPIWrapper::entryWiseSum(Teuchos_CommPtr Comm, Camellia::FieldContainer<ScalarType> &values)
  {
    Camellia::FieldContainer<ScalarType> valuesCopy = values; // it appears this copy is necessary
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, static_cast<int>(values.size()), &valuesCopy[0], &values[0]);
  }
  
  template<typename ScalarType>
  ScalarType MPIWrapper::max(Teuchos_CommPtr Comm, ScalarType value)
  {
    ScalarType valueCopy = value;
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &valueCopy, &value);
    return value;
  }
  
  template<typename ScalarType>
  ScalarType MPIWrapper::min(Teuchos_CommPtr Comm, ScalarType value)
  {
    ScalarType valueCopy = value;
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MIN, 1, &valueCopy, &value);
    return value;
  }

  template<typename KeyType, typename ValueType>
  void MPIWrapper::sendDataMaps(Teuchos_CommPtr Comm,
                                const std::map<int,std::map<KeyType,ValueType>> &recipientDataMaps,
                                std::map<KeyType,ValueType> &receivedMap)
  {
    // convert to a vector, and send that.
    typedef std::vector<std::pair<KeyType,ValueType>> MapVector;
    std::vector<MapVector> dataVectors;
    std::vector<int> recipients;
    for (auto recipientDataMapEntry : recipientDataMaps)
    {
      int recipient = recipientDataMapEntry.first;
      recipients.push_back(recipient);
      auto dataMap = &recipientDataMapEntry.second;
      dataVectors.push_back(MapVector(dataMap->begin(),dataMap->end()));
    }
    MapVector receivedVector;
    sendDataVectors(Comm,recipients,dataVectors,receivedVector);
    receivedMap.insert(receivedVector.begin(),receivedVector.end());
  }

  template<typename DataType>
  void MPIWrapper::sendDataVectors(Teuchos_CommPtr Comm,
                                   const std::map<int,std::vector<DataType>> &recipientDataVectors,
                                   std::vector<DataType> &receivedVector)
  {
    int numRecipients = recipientDataVectors.size();
    std::vector<std::vector<DataType>> dataVectors(numRecipients);
    std::vector<int> recipients(numRecipients);
    int ordinal = 0;
    for (auto recipientVectorEntry : recipientDataVectors)
    {
      recipients[ordinal] = recipientVectorEntry.first;
      dataVectors[ordinal] = recipientVectorEntry.second;
      ordinal++;
    }
    sendDataVectors(Comm,recipients,dataVectors,receivedVector);
  }
  
  template<typename DataType>
  void MPIWrapper::sendDataVectors(Teuchos_CommPtr Comm, const std::vector<int> &recipients,
                                   const std::vector<std::vector<DataType>> &dataVectors,
                                   std::vector<DataType> &receivedVector)
  {
    const int numProcs = Comm->getSize();

    // check that recipients are in numerical order:
    size_t numRecipients = recipients.size();
    for (size_t i=1; i<numRecipients; i++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(recipients[i-1] >= recipients[i], std::invalid_argument, "recipients list must be in numerical order, and populated with distinct recipients");
    }
    TEUCHOS_TEST_FOR_EXCEPTION(dataVectors.size() != numRecipients, std::invalid_argument, "dataVectors and recipients list must be the same length");

    receivedVector.clear();

    std::vector<DataType> packedData;
    std::vector<int> sendCountsBytes(numProcs, 0);
    for (size_t i=0; i<numRecipients; i++)
    {
      const int pid = recipients[i];
      TEUCHOS_TEST_FOR_EXCEPTION((pid < 0) || (pid >= numProcs), std::invalid_argument, "recipient PID is out of range");

      const std::vector<DataType> & dataVector = dataVectors[i];
      const size_t byteCount = dataVector.size() * sizeof(DataType);
      TEUCHOS_TEST_FOR_EXCEPTION(byteCount > static_cast<size_t>(std::numeric_limits<int>::max()), std::invalid_argument, "MPIWrapper::sendDataVectors(): message is too large for MPI int counts");
      sendCountsBytes[pid] = static_cast<int>(byteCount);
      packedData.insert(packedData.end(), dataVector.begin(), dataVector.end());
    }

    if (numProcs == 1)
    {
      receivedVector = packedData;
      return;
    }

#ifdef HAVE_MPI
    auto mpiComm = Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(Comm);
    TEUCHOS_TEST_FOR_EXCEPTION(mpiComm.is_null(), std::invalid_argument, "MPIWrapper::sendDataVectors(): communicator has more than one rank but is not a Teuchos::MpiComm");
    MPI_Comm rawMpiComm = *mpiComm->getRawMpiComm();

    std::vector<int> recvCountsBytes(numProcs, 0);
    MPI_Alltoall(sendCountsBytes.data(), 1, MPI_INT, recvCountsBytes.data(), 1, MPI_INT, rawMpiComm);

    std::vector<int> sendDisplacementsBytes(numProcs, 0);
    std::vector<int> recvDisplacementsBytes(numProcs, 0);
    int totalSendBytes = 0;
    int totalReceiveBytes = 0;
    for (int pid=0; pid<numProcs; pid++)
    {
      sendDisplacementsBytes[pid] = totalSendBytes;
      recvDisplacementsBytes[pid] = totalReceiveBytes;
      totalSendBytes += sendCountsBytes[pid];
      totalReceiveBytes += recvCountsBytes[pid];
    }

    TEUCHOS_TEST_FOR_EXCEPTION((totalSendBytes % static_cast<int>(sizeof(DataType))) != 0, std::logic_error, "MPIWrapper::sendDataVectors(): internal send byte count is not an integral number of DataType entries");
    TEUCHOS_TEST_FOR_EXCEPTION((totalReceiveBytes % static_cast<int>(sizeof(DataType))) != 0, std::logic_error, "MPIWrapper::sendDataVectors(): received byte count is not an integral number of DataType entries");

    receivedVector.resize(static_cast<size_t>(totalReceiveBytes) / sizeof(DataType));

    void *sendBuffer = packedData.empty() ? NULL : static_cast<void*>(packedData.data());
    void *recvBuffer = receivedVector.empty() ? NULL : static_cast<void*>(receivedVector.data());
    MPI_Alltoallv(sendBuffer, sendCountsBytes.data(), sendDisplacementsBytes.data(), MPI_BYTE,
                  recvBuffer, recvCountsBytes.data(), recvDisplacementsBytes.data(), MPI_BYTE,
                  rawMpiComm);
#else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MPIWrapper::sendDataVectors(): MPI support is required for communicators with more than one rank");
#endif
  }
  
  template<typename ScalarType>
  ScalarType MPIWrapper::sum(Teuchos_CommPtr Comm, ScalarType value)
  {
    ScalarType valueCopy = value;
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &valueCopy, &value);
    return value;
  }

  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  template<typename DataType>
  void MPIWrapper::allGatherVariable(Teuchos_CommPtr Comm, std::vector<DataType> &gatheredValues,
                                     std::vector<DataType> &myValues, std::vector<int> &offsets)
  {
    std::vector<int> valuesSizes(Comm->getSize());
    
    // sizes will be in bytes
    int mySize = myValues.size() * sizeof(DataType);
    MPIWrapper::allGather(Comm, valuesSizes, mySize);
    
    offsets.resize(Comm->getSize());
    int offset = 0;
    for (int i=0; i<Comm->getSize(); i++)
    {
      offsets[i] = offset;
      offset += valuesSizes[i];
    }
    int totalSize = offset;
    gatheredValues.resize(totalSize / sizeof(DataType));
    
  #ifdef HAVE_MPI
    if (Comm->getSize() > 1)
    {
      if (Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(Comm) != Teuchos::null)
      {
        MPI_Comm mpiComm = *Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(Comm)->getRawMpiComm();
        char* myPtr = (myValues.size() > 0) ? reinterpret_cast<char*>(&myValues[0]) : NULL;
        MPI_Allgatherv(myPtr, mySize, MPI_CHAR, reinterpret_cast<char*>(gatheredValues.data()), valuesSizes.data(), offsets.data(), MPI_CHAR, mpiComm);
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "numProcs > 1, but Comm is not of type Teuchos::MpiComm<int>");
      }
    }
    else
    {
      gatheredValues = myValues;
    }
  #else
  #endif
    // now, go through offsets to make the sizes in terms of DataType rather than bytes
    for (int i=0; i<int(offsets.size()); i++)
    {
      offsets[i] /= sizeof(DataType);
    }
  }

  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  template<typename Scalar>
  void MPIWrapper::allGatherVariable(Teuchos_CommPtr Comm, Camellia::FieldContainer<Scalar> &gatheredValues,
                                     Camellia::FieldContainer<Scalar> &myValues, Camellia::FieldContainer<int> &offsets)
  {
    std::vector<int> valuesSizes(Comm->getSize());
    
    // sizes will be in bytes
    int mySize = myValues.size() * sizeof(Scalar);
    MPIWrapper::allGather(Comm, valuesSizes, mySize);
    
    offsets.resize(Comm->getSize());
    int offset = 0;
    for (int i=0; i<Comm->getSize(); i++)
    {
      offsets[i] = offset;
      offset += valuesSizes[i];
    }
    int totalSize = offset;
    gatheredValues.resize(totalSize / sizeof(Scalar));

  #ifdef HAVE_MPI
    if (Comm->getSize() > 1)
    {
      if (Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(Comm) != Teuchos::null)
      {
        MPI_Comm mpiComm = *Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int>>(Comm)->getRawMpiComm();
        auto myValuesPtr = (myValues.size() > 0) ? &myValues[0] : NULL;
        char* myValuesCharCast = reinterpret_cast<char*>(myValuesPtr);
        auto gatheredValuesPtr = (gatheredValues.size() > 0) ? &gatheredValues[0] : NULL;
        char* gatheredValuesCharCast = reinterpret_cast<char*>(gatheredValuesPtr);
        MPI_Allgatherv(myValuesCharCast, mySize, MPI_CHAR, gatheredValuesCharCast, &valuesSizes[0], &offsets[0], MPI_CHAR, mpiComm);
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "numProcs > 1, but Comm is not of type Teuchos::MpiComm<int>");
      }
    }
    else
    {
      gatheredValues = myValues;
    }
  #else
  #endif
    // now, go through offsets to make the sizes in Scalar rather than bytes
    for (int i=0; i<offsets.size(); i++)
    {
      offsets[i] /= sizeof(Scalar);
    }
  }
}

#endif /* defined(__Camellia_debug__MPIWrapper__) */
