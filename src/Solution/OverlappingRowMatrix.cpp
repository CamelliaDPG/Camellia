/*@HEADER
// ***********************************************************************
//
//                  Camellia Overlapping Row Matrix:
//
//  This code is largely copied from the IfPack found in Trilinos 11.12.1.
//  Modifications support definitions of overlap level in terms of a Camellia
//  Mesh.  Zero-level overlap means that the owner of each cell sees all the
//  degrees of freedom for that cell--including all the traces, even when
//  the cell owner does not own them.  One-level overlap means that the owner
//  of a cell sees all degrees of freedom belonging to the neighbors of that
//  cell (the cells that share sides with the owned cell).  Two-level overlap
//  extends this to neighbors of neighbors, etc.
//
//
// ***********************************************************************
//@HEADER
*/

#include "Ifpack_ConfigDefs.h"
#include "OverlappingRowMatrix.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Comm.h"
#include "Epetra_MultiVector.h"

#include "CamelliaDebugUtility.h"

#include "Teuchos_GlobalMPISession.hpp"

#include "Ifpack_LocalFilter.h"

#include "Epetra_SerialComm.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"

using namespace Teuchos;

using namespace Camellia;

// ======================================================================
template<typename int_type>
void OverlappingRowMatrix::BuildMap(int OverlapLevel_in, MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter,
                                    bool hierarchical)
{
  // Camellia revision/addition: determine the cell neighbors according to the overlap level
  int sideDim = mesh->getTopology()->getDimension() - 1;
  set<GlobalIndexType> allCells = overlappingCells(mesh->cellIDsInPartition(), mesh, OverlapLevel_in, hierarchical, sideDim);

  std::vector<GlobalIndexType> allCellsVector(allCells.begin(),allCells.end());
//  print("allCells on OverlappingRowMatrix",allCells);
  // Next, determine all global degrees of freedom belonging to those cells
  std::set<GlobalIndexType> globalDofIndices = dofInterpreter->importGlobalIndicesForCells(allCellsVector);
//  print("globalDofIndices (RowMap)", globalDofIndices);
  BuildMap<int_type>(globalDofIndices);
}

template<typename int_type>
void OverlappingRowMatrix::BuildMap(const set<GlobalIndexType> &rowIndices)
{// set up a map for all rows indicated...
  
  std::vector<int_type> ExtElements;
  int count = 0;
  
  for (GlobalIndexType globalDofIndex : rowIndices)
  {
    if (A().RowMatrixRowMap().LID((GlobalIndexTypeToCast)globalDofIndex) == -1)
    {
      typename std::vector<int_type>::iterator pos = find(ExtElements.begin(),ExtElements.end(),globalDofIndex);
      if (pos == ExtElements.end())
      {
        ExtElements.push_back(globalDofIndex);
        ++count;
      }
    }
  }
  
//  { // DEBUGGING:
//    int rank = Teuchos::GlobalMPISession::getRank();
//    if (rank==3)
//      Camellia::print("rank 3 ExtElements", ExtElements);
//  }
  
  // build the map containing all the nodes (original
  // matrix + extended matrix)
  std::vector<int_type> list(NumMyRowsA_ + ExtElements.size());
  for (int i = 0 ; i < NumMyRowsA_ ; ++i)
    list[i] = (int_type) A().RowMatrixRowMap().GID64(i);
  for (int i = 0 ; i < (int)ExtElements.size() ; ++i)
    list[i + NumMyRowsA_] = ExtElements[i];
  
  const int_type *listptr = NULL;
  if ( ! list.empty() ) listptr = &list[0];
  {
    Map_ = rcp( new Epetra_Map((int_type) -1, NumMyRowsA_ + ExtElements.size(),
                               listptr, 0, Comm()) );
  }
  
  // now build the map corresponding to all the external nodes
  // (with respect to A().RowMatrixRowMap().
  {
    const int_type * extelsptr = NULL;
    if ( ! ExtElements.empty() ) extelsptr = &ExtElements[0];
    ExtMap_ = rcp( new Epetra_Map((int_type) -1,ExtElements.size(),
                                  extelsptr,0,A().Comm()) );
  }
  
  rowIndices_ = rowIndices;
}

template<typename int_type>
void OverlappingRowMatrix::BuildMap(int OverlapLevel_in, const set<GlobalIndexType> &globalDofIndices, bool filterByRowIndices)
{
  RCP<Epetra_Map> TmpMap;
  RCP<Epetra_CrsMatrix> TmpMatrix;
  RCP<Epetra_Import> TmpImporter;

  // importing rows corresponding to elements that are
  // in ColMap, but not in RowMap
  const Epetra_Map *RowMap;
  const Epetra_Map *ColMap;

  std::vector<int_type> ExtElements;

  for (int overlap = 0 ; overlap <= OverlapLevel_in ; ++overlap)
  {
    if (TmpMatrix != Teuchos::null)
    {
      RowMap = &(TmpMatrix->RowMatrixRowMap());
      ColMap = &(TmpMatrix->RowMatrixColMap());
    }
    else
    {
      RowMap = &(A().RowMatrixRowMap());
      ColMap = &(A().RowMatrixColMap());
    }

    int size = ColMap->NumMyElements() - RowMap->NumMyElements();
    TEUCHOS_TEST_FOR_EXCEPTION(size < 0, std::invalid_argument, "size may not be negative");
    std::vector<int_type> list(size);

    int count = 0;

    // define the set of rows that are in ColMap but not in RowMap
    for (int i = 0 ; i < ColMap->NumMyElements() ; ++i)
    {
      int_type GID = (int_type) ColMap->GID64(i);
      if (filterByRowIndices && (globalDofIndices.find(GID) == globalDofIndices.end())) continue;
      if (A().RowMatrixRowMap().LID(GID) == -1)
      {
        typename std::vector<int_type>::iterator pos = find(ExtElements.begin(),ExtElements.end(),GID);
        if (pos == ExtElements.end())
        {
          ExtElements.push_back(GID);
          list[count] = GID;
          ++count;
        }
      }
    }

    const int_type *listptr = NULL;
    if ( ! list.empty() ) listptr = &list[0];
    TmpMap = rcp( new Epetra_Map(-1,count, listptr,0,Comm()) );

    TmpMatrix = rcp( new Epetra_CrsMatrix(::Copy,*TmpMap,0) );

    TmpImporter = rcp( new Epetra_Import(*TmpMap,A().RowMatrixRowMap()) );

    TmpMatrix->Import(A(),*TmpImporter,Insert);
    TmpMatrix->FillComplete(A().OperatorDomainMap(),*TmpMap);
  }

  // build the map containing all the nodes (original
  // matrix + extended matrix)
  std::vector<int_type> list(NumMyRowsA_ + ExtElements.size());
  for (int i = 0 ; i < NumMyRowsA_ ; ++i)
    list[i] = (int_type) A().RowMatrixRowMap().GID64(i);
  for (int i = 0 ; i < (int)ExtElements.size() ; ++i)
    list[i + NumMyRowsA_] = ExtElements[i];

//  int rank = Teuchos::GlobalMPISession::getRank();
//
//  if (rank==0) cout << "Overlap level: " << OverlapLevel_in << endl;
//
//  { // DEBUGGING
//    ostringstream rankLabel;
//    rankLabel << "rank " << rank << ", ExtElements";
//    Camellia::print(rankLabel.str(), ExtElements);
//
//    rankLabel.str("");
//    rankLabel << "rank " << rank << ", globalDofIndices";
//    Camellia::print(rankLabel.str(), globalDofIndices);
//
////    rankLabel.str("");
////    rankLabel << "rank " << rank << ", cells";
////    Camellia::print(rankLabel.str(), allCells);
//
//    rankLabel.str("");
//    rankLabel << "rank " << rank << ", list";
//    Camellia::print(rankLabel.str(), list);
//  }

  const int_type *listptr = NULL;
  if ( ! list.empty() ) listptr = &list[0];
  {
    Map_ = rcp( new Epetra_Map((int_type) -1, NumMyRowsA_ + ExtElements.size(),
                               listptr, 0, Comm()) );
  }

//  cout << "On rank " << rank << ", ExtElements.size() = " << ExtElements.size() << endl;

  // now build the map corresponding to all the external nodes
  // (with respect to A().RowMatrixRowMap().
  {
    const int_type * extelsptr = NULL;
    if ( ! ExtElements.empty() ) extelsptr = &ExtElements[0];
    ExtMap_ = rcp( new Epetra_Map((int_type) -1,ExtElements.size(),
                                  extelsptr,0,A().Comm()) );
  }

  if (filterByRowIndices)
  {
    rowIndices_ = globalDofIndices;
  }
  else
  {
    rowIndices_.insert(list.begin(), list.end());
  }
}


template<typename int_type>
void OverlappingRowMatrix::BuildMap(int OverlapLevel_in)
{
  rowIndices_ = set<GlobalIndexType>(); // empty set
  bool filter = false; // don't filter by row indices -- take all of them, just like the IfPack_OverlappingRowMatrix
  BuildMap<int_type>(OverlapLevel_in, rowIndices_, filter);
}

std::set<GlobalIndexType> OverlappingRowMatrix::overlappingCells(const std::set<GlobalIndexType> &cellIDs, MeshPtr mesh,
                                                                 int overlapLevel, bool hierarchical,
                                                                 int dimensionForNeighborRelation)
{
  set<GlobalIndexType> cellsInOverlap = cellIDs;
  for (GlobalIndexType cellID : cellIDs)
  {
    std::set<GlobalIndexType> cellHalo = OverlappingRowMatrix::overlappingCells(cellID, mesh, overlapLevel,
                                                                                hierarchical, dimensionForNeighborRelation);
    cellsInOverlap.insert(cellHalo.begin(), cellHalo.end());
  }
  return cellsInOverlap;
}

std::set<GlobalIndexType> OverlappingRowMatrix::overlappingCells(GlobalIndexType cellID, MeshPtr mesh,
                                                                 int overlapLevel, bool hierarchical,
                                                                 int dimensionForNeighborRelation)
{
  set<GlobalIndexType> cells = {cellID};
  
  // new version
  std::set<GlobalIndexType> lastNeighbors = cells;
  for (int overlap = 0 ; overlap < overlapLevel ; ++overlap)
  {
    std::set<GlobalIndexType> cellNeighbors;
    for (std::set<GlobalIndexType>::iterator cellIDIt = lastNeighbors.begin(); cellIDIt != lastNeighbors.end(); cellIDIt++)
    {
      CellPtr cell = mesh->getTopology()->getCell(*cellIDIt);
      
      bool useOldSideBasedNeighborRelation = true;
      
      if (useOldSideBasedNeighborRelation)
      {
        int numSides = cell->getSideCount();
        for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
        {
          pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal,mesh->getTopology());
          if (neighborInfo.first != -1)   // -1 indicates boundary/no neighbor
          {
            GlobalIndexType neighborCellID = neighborInfo.first;
            unsigned neighborSideOrdinal = neighborInfo.second;
            if (mesh->cellIsActive(neighborCellID))
            {
              cellNeighbors.insert(neighborCellID);
            }
            else
            {
              CellPtr neighborCell = mesh->getTopology()->getCell(neighborCellID);
              vector< pair< GlobalIndexType, unsigned> > activeDescendants = neighborCell->getDescendantsForSide(neighborSideOrdinal,
                                                                                                                 mesh->getTopology());
              for (auto descendantCellPair : activeDescendants)
              {
                cellNeighbors.insert(descendantCellPair.first);
              }
            }
          }
        }
      }
      else
      {
        // if dimensionForNeighborRelation < 0, then use sideDim.
        int sideDim = cell->topology()->getDimension() - 1;
        if (dimensionForNeighborRelation < 0) dimensionForNeighborRelation = sideDim;
        int numSubcells = cell->topology()->getSubcellCount(dimensionForNeighborRelation);
        for (int subcellOrdinal=0; subcellOrdinal<numSubcells; subcellOrdinal++)
        {
          IndexType subcellEntityIndex = cell->entityIndex(dimensionForNeighborRelation, subcellOrdinal);
          set< pair<IndexType, unsigned> > cellPairs = mesh->getTopology()->getCellsContainingEntity(dimensionForNeighborRelation, subcellEntityIndex);
          
          CellPtr ancestor = cell->ancestralCellForSubcell(dimensionForNeighborRelation, subcellOrdinal, mesh->getTopology());
          if (ancestor->cellIndex() != cell->cellIndex())
          {
            pair<unsigned, unsigned> ancestralInfo = cell->ancestralSubcellOrdinalAndDimension(dimensionForNeighborRelation,
                                                                                               subcellOrdinal, mesh->getTopology());
            IndexType ancestralEntityIndex = ancestor->entityIndex(ancestralInfo.second, ancestralInfo.first);
            set< pair<IndexType, unsigned> > ancestralCellPairs = mesh->getTopology()->getCellsContainingEntity(ancestralInfo.second, ancestralEntityIndex);
            cellPairs.insert(ancestralCellPairs.begin(),ancestralCellPairs.end());
          }
          
          while (cellPairs.size() > 0)
          {
            set< pair<IndexType, unsigned> > newPairs;
            for (pair<IndexType, unsigned> cellPair : cellPairs)
            {
              GlobalIndexType neighborCellID = cellPair.first;
              if (mesh->cellIsActive(neighborCellID))
              {
                cellNeighbors.insert(neighborCellID);
              }
              else
              {
                if (dimensionForNeighborRelation == sideDim)
                {
                  // find the descendants of neighbor along the side
                  unsigned neighborSideOrdinal = cellPair.second;
                  CellPtr neighborCell = mesh->getTopology()->getCell(neighborCellID);
                  vector< pair< GlobalIndexType, unsigned> > activeDescendants = neighborCell->getDescendantsForSide(neighborSideOrdinal, mesh->getTopology());
                  for (auto descendantCellPair : activeDescendants)
                  {
                    cellNeighbors.insert(descendantCellPair.first);
                  }
                }
                else
                {
                  static bool haveWarned = false;
                  if (!haveWarned)
                  {
                    haveWarned = true;
                    cout << "WARNING: In OverlappingRowMatrix::overlappingCells(), encountered hanging node when the dimension for the neighbor relation was not sideDim.  We still need to finish writing the code to find neighbors appropriately in this case!\n";
                  }
                  
                  // TODO: finish this
                  //       we need to proceed in two directions: find any descendants of subcellEntityIndex that have active cells that we haven't already seen, and any ancestors of subcellEntityIndex that have active cells that we haven't already seen.
                  // we insert all such neighbors into cellPairs.
                  // (to check what we've seen, should look both at cellNeighbors container and at cells container)
                  // NOTE: It may be worth implementing Cell::getActiveNeighborsForSubcell(subcdim, subcord);
                  //       -- This would allow us to replace everything to do with cellPairs above (though the new method would likely call getCellsContainingEntity)
                }
              }
            }
            cellPairs = newPairs;
          }
        }
      }
    }
    cells.insert(cellNeighbors.begin(), cellNeighbors.end());
    lastNeighbors = cellNeighbors;
  }
  if (hierarchical)
  {
    // get ancestors up to overlap level above our cells:
    std::set<GlobalIndexType> ancestors = {cellID};
    for (int overlap = 0 ; overlap < overlapLevel ; ++overlap)
    {
      std::set<GlobalIndexType> previousAncestors = ancestors;
      ancestors.clear();
      for (std::set<GlobalIndexType>::iterator cellIDIt = previousAncestors.begin(); cellIDIt != previousAncestors.end();
           cellIDIt++)
      {
        GlobalIndexType cellID = *cellIDIt;
        CellPtr cell = mesh->getTopology()->getCell(cellID);
        if (cell->getParent() == Teuchos::null)
        {
          ancestors.insert(cellID);
        }
        else
        {
          ancestors.insert(cell->getParent()->cellIndex());
        }
      }
      previousAncestors = ancestors;
    }
    // now, get all the descendants in ancestors:
    set<GlobalIndexType> descendantCells;
    for (std::set<GlobalIndexType>::iterator cellIDIt = ancestors.begin(); cellIDIt != ancestors.end();
         cellIDIt++)
    {
      GlobalIndexType cellID = *cellIDIt;
      CellPtr cell = mesh->getTopology()->getCell(cellID);
      std::set<GlobalIndexType> descendants = cell->getDescendants(mesh->getTopology());
      descendantCells.insert(descendants.begin(),descendants.end());
    }
    // finally, INTERSECT with the allCells given by standard neighbor thing
    set<GlobalIndexType> neighborCells = cells;
    cells.clear();
    for (GlobalIndexType descendantCell : descendantCells)
    {
      if (neighborCells.find(descendantCell) != neighborCells.end())
      {
        cells.insert(descendantCell);
      }
    }
  }
//  print("cells",cells);
  return cells;
}

OverlappingRowMatrix::
OverlappingRowMatrix(const Teuchos::RCP<const Epetra_RowMatrix>& Matrix_in, int OverlapLevel_in,
                     const std::set<GlobalIndexType> &rowIndicesForThisRank) :
  Matrix_(Matrix_in),
  OverlapLevel_(OverlapLevel_in)
{
  NumMyRowsA_ = A().NumMyRows();

  // construct the external matrix

#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  if(A().RowMatrixRowMap().GlobalIndicesInt())
  {
    BuildMap<int>(OverlapLevel_in, rowIndicesForThisRank);
  }
  else
#endif
#ifndef EPETRA_NO_64BIT_GLOBAL_INDICES
    if(A().RowMatrixRowMap().GlobalIndicesLongLong())
    {
      BuildMap<long long>(OverlapLevel_in, rowIndicesForThisRank);
    }
    else
#endif
      throw "OverlappingRowMatrix::OverlappingRowMatrix: GlobalIndices type unknown";

  ExtMatrix_ = rcp( new Epetra_CrsMatrix(::Copy,*ExtMap_,*Map_,0) );

  ExtImporter_ = rcp( new Epetra_Import(*ExtMap_,A().RowMatrixRowMap()) );
  ExtMatrix_->Import(A(),*ExtImporter_,Insert);
  ExtMatrix_->FillComplete(A().OperatorDomainMap(),*Map_);

  Importer_ = rcp( new Epetra_Import(*Map_,A().RowMatrixRowMap()) );

  // fix indices for overlapping matrix
  NumMyRowsB_ = B().NumMyRows();
  NumMyRows_ = NumMyRowsA_ + NumMyRowsB_;
  NumMyCols_ = NumMyRows_;

  NumMyDiagonals_ = A().NumMyDiagonals() + B().NumMyDiagonals();

  NumMyNonzeros_ = A().NumMyNonzeros() + B().NumMyNonzeros();
  long long NumMyNonzeros_tmp = NumMyNonzeros_;
  Comm().SumAll(&NumMyNonzeros_tmp,&NumGlobalNonzeros_,1);
  MaxNumEntries_ = A().MaxNumEntries();
  
  if (MaxNumEntries_ < B().MaxNumEntries())
    MaxNumEntries_ = B().MaxNumEntries();
}

//! Constructor for an exact match to IfPack_OverlappingMatrix's behavior.
OverlappingRowMatrix::
OverlappingRowMatrix(const Teuchos::RCP<const Epetra_RowMatrix>& Matrix_in, int OverlapLevel_in) :
  Matrix_(Matrix_in),
  OverlapLevel_(OverlapLevel_in)
{
  // nothing to do as well with one process
  if (Comm().NumProc() == 1)
    IFPACK_CHK_ERRV(-1);

  NumMyRowsA_ = A().NumMyRows();

  // construct the external matrix

#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  if(A().RowMatrixRowMap().GlobalIndicesInt())
  {
    BuildMap<int>(OverlapLevel_in);
  }
  else
#endif
#ifndef EPETRA_NO_64BIT_GLOBAL_INDICES
    if(A().RowMatrixRowMap().GlobalIndicesLongLong())
    {
      BuildMap<long long>(OverlapLevel_in);
    }
    else
#endif
      throw "OverlappingRowMatrix::OverlappingRowMatrix: GlobalIndices type unknown";

  ExtMatrix_ = rcp( new Epetra_CrsMatrix(::Copy,*ExtMap_,*Map_,0) );

  ExtImporter_ = rcp( new Epetra_Import(*ExtMap_,A().RowMatrixRowMap()) );
  ExtMatrix_->Import(A(),*ExtImporter_,Insert);
  ExtMatrix_->FillComplete(A().OperatorDomainMap(),*Map_);

  Importer_ = rcp( new Epetra_Import(*Map_,A().RowMatrixRowMap()) );

  // fix indices for overlapping matrix
  NumMyRowsB_ = B().NumMyRows();
  NumMyRows_ = NumMyRowsA_ + NumMyRowsB_;
  NumMyCols_ = NumMyRows_;

  NumMyDiagonals_ = A().NumMyDiagonals() + B().NumMyDiagonals();

  NumMyNonzeros_ = A().NumMyNonzeros() + B().NumMyNonzeros();
  long long NumMyNonzeros_tmp = NumMyNonzeros_;
  Comm().SumAll(&NumMyNonzeros_tmp,&NumGlobalNonzeros_,1);
  MaxNumEntries_ = A().MaxNumEntries();

  if (MaxNumEntries_ < B().MaxNumEntries())
    MaxNumEntries_ = B().MaxNumEntries();
}

// ======================================================================
// Constructor for the case of one core per subdomain (the only case supported by this Camellia version)
OverlappingRowMatrix::
OverlappingRowMatrix(const RCP<const Epetra_RowMatrix>& Matrix_in,
                     int OverlapLevel_in, MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter,
                     bool hierachical)  :
  Matrix_(Matrix_in),
  OverlapLevel_(OverlapLevel_in)
{
  NumMyRowsA_ = A().NumMyRows();

  // construct the external matrix

#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  if(A().RowMatrixRowMap().GlobalIndicesInt())
  {
    BuildMap<int>(OverlapLevel_in, mesh, dofInterpreter, hierachical);
  }
  else
#endif
#ifndef EPETRA_NO_64BIT_GLOBAL_INDICES
    if(A().RowMatrixRowMap().GlobalIndicesLongLong())
    {
      BuildMap<long long>(OverlapLevel_in, mesh, dofInterpreter, hierachical);
    }
    else
#endif
      throw "OverlappingRowMatrix::OverlappingRowMatrix: GlobalIndices type unknown";

  ExtMatrix_ = rcp( new Epetra_CrsMatrix(::Copy,*ExtMap_,*Map_,0) );

  ExtImporter_ = rcp( new Epetra_Import(*ExtMap_,A().RowMatrixRowMap()) );
  ExtMatrix_->Import(A(),*ExtImporter_,Insert);
  ExtMatrix_->FillComplete(A().OperatorDomainMap(),*Map_);

  Importer_ = rcp( new Epetra_Import(*Map_,A().RowMatrixRowMap()) );

  // fix indices for overlapping matrix
  NumMyRowsB_ = B().NumMyRows();
  NumMyRows_ = NumMyRowsA_ + NumMyRowsB_;
  NumMyCols_ = NumMyRows_;

  NumMyDiagonals_ = A().NumMyDiagonals() + B().NumMyDiagonals();

  NumMyNonzeros_ = A().NumMyNonzeros() + B().NumMyNonzeros();
  long long NumMyNonzeros_tmp = NumMyNonzeros_;
  Comm().SumAll(&NumMyNonzeros_tmp,&NumGlobalNonzeros_,1);
  MaxNumEntries_ = A().MaxNumEntries();

  if (MaxNumEntries_ < B().MaxNumEntries())
    MaxNumEntries_ = B().MaxNumEntries();
  
//  { // DEBUGGING
//
//    int rank = Teuchos::GlobalMPISession::getRank();
//    
//    EpetraExt::RowMatrixToMatrixMarketFile("A.dat",A(), NULL, NULL, false); // false: don't write header
//    printMapSummary(*ExtMap_, "ExtMap_");
//    printMapSummary(*Map_, "Map_");
//
//    cout << "On rank " << rank << ", ExtImporter_->NumSameIDs = " << ExtImporter_->NumSameIDs() << endl;
//    cout << "On rank " << rank << ", ExtImporter_->NumRemoteIDs = " << ExtImporter_->NumRemoteIDs() << endl;
//
//    cout << "On rank " << rank << ", Importer_->NumSameIDs = " << Importer_->NumSameIDs() << endl;
//    cout << "On rank " << rank << ", Importer_->NumRemoteIDs = " << Importer_->NumRemoteIDs() << endl;
//    cout << "OverlappingRowMatrix outputting on rank " << rank << endl;
//
//    if ((rank==3) && (NumMyRows_==11)) {
//      Epetra_SerialComm SerialComm;
//      Epetra_Map    localMap(NumMyRows_, 0, SerialComm);
//
////      Epetra_CrsMatrix localMatrix(Copy, localMap, NumMyRows_);
//
//      int nnz;
//      for (int row=0; row < NumMyRows_; row++) {
//        int numEntries;
//        this->NumMyRowEntries(row, numEntries);
//
//        vector<double> rowValues(numEntries);
//        vector<GlobalIndexTypeToCast> indices(numEntries);
//
//        ExtractMyRowCopy(row, rowValues.size(), nnz, &rowValues[0], &indices[0]);
//        ostringstream rowLabel;
//        rowLabel << "rank " << rank << ", row " << row;
//
//        Camellia::print(rowLabel.str().c_str(), rowValues);
//        rowLabel.str("");
//        rowLabel << "rank " << rank << ", row " << row << ", local indices";
//        Camellia::print(rowLabel.str().c_str(), indices);
//
//        vector<GlobalIndexTypeToCast> globalIndices(indices.size());
//        for (int col=0; col<indices.size(); col++) {
//          GlobalIndexTypeToCast colLID = indices[col];
//          GlobalIndexTypeToCast colGID;
////          if (row < NumMyRowsA_) {
////            colGID = Matrix_->RowMatrixColMap().GID(colLID);
////          } else {
//            colGID = this->RowMatrixColMap().GID(colLID);
////          }
//          globalIndices[col] = colGID;
//        }
//
//        rowLabel.str("");
//        rowLabel << "rank " << rank << ", row " << row << ", global indices";
//        Camellia::print(rowLabel.str().c_str(), globalIndices);
//
////        localMatrix.InsertGlobalValues(row, nnz, &rowValues[0], &indices[0]);
//      }
////      cout << "Calling localMatrix.FillComplete().\n";
////      localMatrix.FillComplete();
////      cout << "writing localMatrix to file.\n";
////      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/rank3_local.dat",localMatrix, NULL, NULL, false);
//    }
//
////    Teuchos::RCP<const Epetra_RowMatrix> this_A = Teuchos::rcp( &this->A(), false );
////    Teuchos::RCP<const Epetra_RowMatrix> A = Teuchos::rcp( new Ifpack_LocalFilter( this_A ) );
////    Teuchos::RCP<const Epetra_RowMatrix> this_B = Teuchos::rcp( ExtMatrix_.get(), false );
////    Teuchos::RCP<const Epetra_RowMatrix> B = Teuchos::rcp( new Ifpack_LocalFilter( this_B ) );
////
////    cout << "On rank " << rank << ", B->NumMyRows() = " << B->NumMyRows() << endl;
////    cout << "On rank " << rank << ", B->NumMyCols() = " << B->NumMyCols() << endl;
////
////    ostringstream rankLabel;
////    rankLabel << "/tmp/B_" << rank << ".dat";
////    EpetraExt::RowMatrixToMatrixMarketFile(rankLabel.str().c_str(),*B, NULL, NULL, false);
////
////    rankLabel.str("");
////    rankLabel << "/tmp/A_" << rank << ".dat";
////    EpetraExt::RowMatrixToMatrixMarketFile(rankLabel.str().c_str(),*A, NULL, NULL, false);
//  }
}

// ======================================================================
int OverlappingRowMatrix::
NumMyRowEntries(int MyRow, int & NumEntries) const
{
  if (MyRow < NumMyRowsA_)
    return(A().NumMyRowEntries(MyRow,NumEntries));
  else
    return(B().NumMyRowEntries(MyRow - NumMyRowsA_, NumEntries));
}

// ======================================================================
int OverlappingRowMatrix::
ExtractMyRowCopy(int MyRow, int Length, int & NumEntries, double *Values,
                 int * Indices) const
{
  int ierr;
  if (MyRow < NumMyRowsA_)
  {
    ierr = A().ExtractMyRowCopy(MyRow,Length,NumEntries,Values,Indices);
    // pretty sure we need to remap Indices here, like so:
    int offset = 0; // for when we delete entries
    for (int i=0; i<NumEntries; i++)
    {
      int lid_A = Indices[i];
      GlobalIndexTypeToCast gid_A = A().RowMatrixColMap().GID(lid_A);
      if (gid_A==-1)
      {
        cout << "Error: lid_A not found in A's column map!\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: lid_A not found in A's column map!");
      }

      int my_lid = this->RowMatrixColMap().LID(gid_A);
      if (my_lid == -1)
      {
        offset++;
      }
      else
      {
        Indices[i-offset] = my_lid;
        Values[i-offset] = Values[i];
      }
//      if (my_lid==-1) {
//        int rank = Teuchos::GlobalMPISession::getRank();
//
//        cout << "Error on rank " << rank <<  ": gid_A not found in this's column map!\n";
//        cout << "rank " << rank << ", row " << MyRow << ", i: " << i << ", lid_A: " << lid_A << ", gid_A: " << gid_A << ", my_lid: " << my_lid << endl;
//        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: gid_A not found in this's column map!");
//      }

    }
    NumEntries -= offset;
  }
  else
    ierr = B().ExtractMyRowCopy(MyRow - NumMyRowsA_,Length,NumEntries,
                                Values,Indices);

  IFPACK_RETURN(ierr);
}

// ======================================================================
int OverlappingRowMatrix::
ExtractDiagonalCopy(Epetra_Vector & Diagonal) const
{
  IFPACK_CHK_ERR(-1);
}


// ======================================================================
int OverlappingRowMatrix::
Multiply(bool TransA, const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  // In our present usage, I'm pretty sure this doesn't get invoked...
  // (We use this exclusively in the context of an IfPack_LocalFilter, which invokes ExtractMyRowCopy)
  cout << "Entered OverlappingRowMatrix::Multiply().\n";

  int NumVectors = X.NumVectors();
  std::vector<int> Ind(MaxNumEntries_);
  std::vector<double> Val(MaxNumEntries_);

  Y.PutScalar(0.0);

  // matvec with A (local rows)
  for (int i = 0 ; i < NumMyRowsA_ ; ++i)
  {
    for (int k = 0 ; k < NumVectors ; ++k)
    {
      int Nnz;
      IFPACK_CHK_ERR(A().ExtractMyRowCopy(i,MaxNumEntries_,Nnz,
                                          &Val[0], &Ind[0]));
      for (int j = 0 ; j < Nnz ; ++j)
      {
        Y[k][i] += Val[j] * X[k][Ind[j]];
      }
    }
  }

  // matvec with B (overlapping rows)
  for (int i = 0 ; i < NumMyRowsB_ ; ++i)
  {
    for (int k = 0 ; k < NumVectors ; ++k)
    {
      int Nnz;
      IFPACK_CHK_ERR(B().ExtractMyRowCopy(i,MaxNumEntries_,Nnz,
                                          &Val[0], &Ind[0]));
      for (int j = 0 ; j < Nnz ; ++j)
      {
        Y[k][i + NumMyRowsA_] += Val[j] * X[k][Ind[j]];
      }
    }
  }
  return(0);
}

// ======================================================================
int OverlappingRowMatrix::
Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  IFPACK_CHK_ERR(Multiply(UseTranspose(),X,Y));
  return(0);
}

// ======================================================================
int OverlappingRowMatrix::
ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  IFPACK_CHK_ERR(-1);
}

// ======================================================================
Epetra_RowMatrix& OverlappingRowMatrix::B() const
{
  return(*ExtMatrix_);
}
// ======================================================================
const Epetra_BlockMap& OverlappingRowMatrix::Map() const
{
  return(*Map_);
}

// ======================================================================
int OverlappingRowMatrix::
ImportMultiVector(const Epetra_MultiVector& X, Epetra_MultiVector& OvX,
                  Epetra_CombineMode CM)
{
  OvX.Import(X,*Importer_,CM);
  return(0);
}

// ======================================================================
int OverlappingRowMatrix::
ExportMultiVector(const Epetra_MultiVector& OvX, Epetra_MultiVector& X,
                  Epetra_CombineMode CM)
{
  X.Export(OvX,*Importer_,CM);
  return(0);
}

