//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  E_to_T.cpp
//  Camellia
//
//  Created by Nate Roberts on 7/29/25.
//
//

#include "E_to_T.hpp"

#include "MPIWrapper.h"

#include <Epetra_MultiVector.h>

using namespace Camellia;

TGraphPtr E_to_T::graph(const Epetra_CrsGraph &eGraph)
{
  TGraphPtr tGraph;
  using GO = TMatrix<double>::global_ordinal_type;
  // we are assuming both Tpetra and Epetra were built with 64-bit global ordinals:
  static_assert(sizeof(GO) == sizeof(long long));
  
  auto eRowMap = eGraph.RowMap();
  TMapPtr tRowMap = map(eRowMap);
  
  auto eColMap = eGraph.ColMap();
  TMapPtr tColMap = map(eColMap);
  
  const int numMyRows = eGraph.NumMyRows();
  
  std::vector<size_t> goIndicesPerRow(numMyRows);
  for (int i=0; i<numMyRows; i++)
  {
    goIndicesPerRow[i] = static_cast<size_t>(eGraph.NumMyIndices(i));
  }
    
  tGraph = Teuchos::rcp(new TGraph(tRowMap, tColMap, goIndicesPerRow) );
  
  int maxColCount = eGraph.MaxNumIndices();
  std::vector<GO> goIndicesT(maxColCount);
  
  for (int i = 0; i < numMyRows; i++)
  {
    int numEntries = 0;
    int* localIndices = nullptr;

    eGraph.ExtractMyRowView(i, numEntries, localIndices);
    
    for (int j=0; j<numEntries; j++)
    {
      goIndicesT[j] = static_cast<GO>(eColMap.GID64(localIndices[j]));
    }

    GO i_global = eGraph.RowMap().GID64(i);
    tGraph->insertGlobalIndices(i_global, numEntries, goIndicesT.data());
  }

  auto eDomainMap = eGraph.DomainMap();
  TMapPtr tDomainMap = map(eDomainMap);
  
  TMapPtr tRangeMap = tRowMap;
  
  tGraph->fillComplete(tDomainMap, tRangeMap);
  return tGraph;
}

TMapPtr E_to_T::map(const Epetra_BlockMap &eMap)
{
  using GO = TMatrix<double>::global_ordinal_type;
  
  GO numGlobalElements = static_cast<GO>(eMap.NumGlobalElements64());
  int numMyElements = eMap.NumMyElements();
  
  std::vector<GO> goMyIndices(numMyElements);
  for (int i=0; i<numMyElements; i++)
  {
    goMyIndices[i] = static_cast<GO>(eMap.GID64(i));
  }
  
  Epetra_CommPtr eComm = Teuchos::rcp(&const_cast<Epetra_Comm&>(eMap.Comm()),false);
  Teuchos_CommPtr comm = MPIWrapper::TeuchosComm(eComm); // NOTE: this is a collective call!
  
  TMapPtr tMap = Teuchos::rcp(new TMap(numGlobalElements, goMyIndices.data(), numMyElements, 0, comm));
  return tMap;
}

Teuchos::RCP< TMatrix<double> > E_to_T::matrix(Teuchos::RCP<const Epetra_CrsMatrix> eMatrix)
{
  Teuchos::RCP< TMatrix<double> > tMatrix;
  
  using GO = TMatrix<double>::global_ordinal_type;

  const Epetra_CrsGraph & eGraph = eMatrix->Graph();
  auto tGraph = graph(eGraph);
  
  tMatrix = Teuchos::rcp( new TMatrix<double>(tGraph) );
  
  auto eColMap = eGraph.ColMap();
  
  int numMyRows = eMatrix->NumMyRows();
  
  int maxColCount = eMatrix->MaxNumEntries();
  std::vector<GO> goIndicesT(maxColCount);
  
  for (int i = 0; i < numMyRows; ++i) {
    int numEntries = 0;
    double* values = nullptr;
    int* localIndices = nullptr;

    eMatrix->ExtractMyRowView(i, numEntries, values, localIndices);
    
    for (int j=0; j<numEntries; j++)
    {
      goIndicesT[j] = static_cast<GO>(eColMap.GID64(localIndices[j]));
    }
    GO i_global = eGraph.RowMap().GID64(i);
    tMatrix->replaceGlobalValues(i_global, numEntries, values, goIndicesT.data());
    
//    {
//      std::cout << "row " << i_global << ": ";
//      for (int j=0; j<numEntries; j++)
//      {
//        std::cout << goIndicesT[j] << " --> " << values[j] << ";";
//      }
//      std::cout << std::endl;
//    }
  }
  tMatrix->fillComplete();
  
  return tMatrix;
}

TVectorPtr<double> E_to_T::vector(Teuchos::RCP<const Epetra_MultiVector> eVector)
{
  auto tMap = map(eVector->Map());
  
  const int numVectors = eVector->NumVectors();
  
  auto tVector = Teuchos::rcp(new TVector<double>(tMap, numVectors));
  
  auto myRowCount = tMap->getLocalNumElements();
  for (int i=0; i<myRowCount; i++)
  {
    for (int j=0; j<numVectors; j++)
    {
      double value = (*eVector)[j][i];
      tVector->replaceLocalValue(i, j, value);
    }
  }

  if (tVector->need_sync_device())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This implementation does not yet sync to device.");
  }
  
  return tVector;
}
