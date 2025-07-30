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

using namespace Camellia;

TGraphPtr E_to_T::graph(const Epetra_CrsGraph &eGraph)
{
  TGraphPtr tGraph;
  using GO = TMatrix<double>::global_ordinal_type;
  
  auto eRowMap = eGraph.RowMap();
  TMapPtr tRowMap = map(eRowMap);
  
  const int numMyRows = eGraph.NumMyRows();
  
  std::vector<size_t> goIndicesPerRow(numMyRows);
  for (int i=0; i<numMyRows; i++)
  {
    goIndicesPerRow[i] = static_cast<size_t>(eGraph.NumMyIndices(i));
  }
    
  tGraph = Teuchos::rcp(new TGraph(tRowMap, goIndicesPerRow) );
  
  int maxColCount = eGraph.MaxNumIndices();
  std::vector<GO> goIndicesT(maxColCount);
  
  for (int i = 0; i < numMyRows; i++)
  {
    int numEntries = 0;
    int* indices = nullptr;

    eGraph.ExtractMyRowView(i, numEntries, indices);
    
    for (int j=0; j<numEntries; j++)
    {
      goIndicesT[j] = static_cast<GO>(indices[j]);
    }

    GO i_global = eGraph.RowMap().GID(i);
    tGraph->insertGlobalIndices(i_global, numEntries, &goIndicesT[0]);
  }

  tGraph->fillComplete();
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
    goMyIndices[i] = static_cast<GO>(eMap.GID(i));
  }
  
  Epetra_CommPtr eComm = Teuchos::rcp(&const_cast<Epetra_Comm&>(eMap.Comm()),false);
  Teuchos_CommPtr comm = MPIWrapper::TeuchosComm(eComm);
  
  TMapPtr tMap = Teuchos::rcp(new TMap(numGlobalElements, &goMyIndices[0], numMyElements, 0, comm));
  return tMap;
}

Teuchos::RCP< TMatrix<double> > E_to_T::matrix(Teuchos::RCP<Epetra_CrsMatrix> eMatrix)
{
  Teuchos::RCP< TMatrix<double> > tMatrix;
  
  using GO = TMatrix<double>::global_ordinal_type;

  const Epetra_CrsGraph & eGraph = eMatrix->Graph();
  auto tGraph = graph(eGraph);
  
  tMatrix = Teuchos::rcp( new TMatrix<double>(tGraph) );
  
  int numMyRows = eMatrix->NumMyRows();
  
  int maxColCount = eMatrix->MaxNumEntries();
  std::vector<GO> goIndicesT(maxColCount);
  
  for (int i = 0; i < numMyRows; ++i) {
    int numEntries = 0;
    double* values = nullptr;
    int* indices = nullptr;

    eMatrix->ExtractMyRowView(i, numEntries, values, indices);
    
    for (int j=0; j<numEntries; j++)
    {
      goIndicesT[j] = static_cast<GO>(indices[j]);
    }

    GO i_global = eGraph.RowMap().GID(i);
    tMatrix->replaceGlobalValues(i_global, numEntries, values, &goIndicesT[0]);
  }
  tMatrix->fillComplete();
  
  return tMatrix;
}
