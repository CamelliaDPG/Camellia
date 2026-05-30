//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
// This code is derived from source governed by the license LICENSE-Ifpack in the licenses directory.

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

#include "OverlappingRowMatrix.h"

#include "E_to_T.hpp"
#include "T_to_E.hpp"

#include "Epetra_RowMatrix.h"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Comm.h"
#include "Epetra_MultiVector.h"

#include "CamelliaDebugUtility.h"
#include "E_to_T.hpp"

#include "Teuchos_GlobalMPISession.hpp"

#include "Epetra_SerialComm.h"

#include <MatrixMarket_Tpetra.hpp>

using namespace Teuchos;

using namespace Camellia;

using Scalar = double;

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
    if (Matrix_->getRowMap()->getLocalElement(globalDofIndex) == -1)
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
  auto A_ptr = A();
  auto A_rowMap = A_ptr->RowMatrixRowMap();
  for (int i = 0 ; i < NumMyRowsA_; ++i)
    list[i] = (int_type) A_rowMap.GID64(i);
  for (int i = 0 ; i < (int)ExtElements.size() ; ++i)
    list[i + NumMyRowsA_] = ExtElements[i];
  
  const int_type *listptr = NULL;
  if ( ! list.empty() ) listptr = &list[0];
  {
    eMap_ = rcp( new Epetra_Map((int_type) -1, NumMyRowsA_ + ExtElements.size(),
                               listptr, 0, Comm()) );
    Map_  = E_to_T::map(*eMap_);
    
    /*
         auto unknownGlobalElementCount = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid();
         
         Map_ = Teuchos::rcp(new TMap(unknownGlobalElementCount, listptr, NumMyRowsA_ + ExtElements.size(), 0, Matrix_->getComm()));

     */
  }
  
  // now build the map corresponding to all the external nodes
  // (with respect to A()->RowMatrixRowMap().
  {
    const int_type * extelsptr = NULL;
    if ( ! ExtElements.empty() ) extelsptr = &ExtElements[0];
    ExtMap_ = rcp( new Epetra_Map((int_type) -1,ExtElements.size(),
                                  extelsptr,0,Comm()) );
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
  
  auto eA = A();

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
      RowMap = &(eA->RowMatrixRowMap());
      ColMap = &(eA->RowMatrixColMap());
    }

    int size = ColMap->NumMyElements() - RowMap->NumMyElements();
    TEUCHOS_TEST_FOR_EXCEPTION(size < 0, std::invalid_argument, "size may not be negative");
    std::vector<int_type> list(size);

    int count = 0;
    
    auto A_rowMap = eA->RowMatrixRowMap();

    // define the set of rows that are in ColMap but not in RowMap
    for (int i = 0 ; i < ColMap->NumMyElements() ; ++i)
    {
      int_type GID = (int_type) ColMap->GID64(i);
      if (filterByRowIndices && (globalDofIndices.find(GID) == globalDofIndices.end())) continue;
      if (A_rowMap.LID(GID) == -1)
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

    TmpImporter = rcp( new Epetra_Import(*TmpMap,eA->RowMatrixRowMap()) );

    TmpMatrix->Import(*eA,*TmpImporter,Insert);
    TmpMatrix->FillComplete(eA->OperatorDomainMap(),*TmpMap);
  }

  // build the map containing all the nodes (original
  // matrix + extended matrix)
  std::vector<int_type> list(NumMyRowsA_ + ExtElements.size());
  for (int i = 0 ; i < NumMyRowsA_ ; ++i)
    list[i] = (int_type) eA->RowMatrixRowMap().GID64(i);
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
    auto eMap = rcp( new Epetra_Map((int_type) -1, NumMyRowsA_ + ExtElements.size(),
                                    listptr, 0, Comm()) );
    Map_ = E_to_T::map(*eMap);
  }

//  cout << "On rank " << rank << ", ExtElements.size() = " << ExtElements.size() << endl;

  // now build the map corresponding to all the external nodes
  // (with respect to A()->RowMatrixRowMap().
  {
    const int_type * extelsptr = NULL;
    if ( ! ExtElements.empty() ) extelsptr = &ExtElements[0];
    ExtMap_ = rcp( new Epetra_Map((int_type) -1,ExtElements.size(),
                                  extelsptr,0,Comm()) );
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
  
  std::set<GlobalIndexType> lastNeighbors = cells;
  for (int overlap = 0 ; overlap < overlapLevel ; ++overlap)
  {
    std::set<GlobalIndexType> cellNeighbors;
    for (GlobalIndexType neighborCellID : lastNeighbors)
    {
      CellPtr cell = mesh->getTopology()->getCell(neighborCellID);
      // if dimensionForNeighborRelation < 0, then use sideDim.
      int sideDim = cell->topology()->getDimension() - 1;
      if (dimensionForNeighborRelation < 0) dimensionForNeighborRelation = sideDim;
      
      set<GlobalIndexType> thisCellNeighbors = cell->getActiveNeighborIndices(dimensionForNeighborRelation, mesh->getTopology());
      cellNeighbors.insert(thisCellNeighbors.begin(),thisCellNeighbors.end());
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
OverlappingRowMatrix(const TMatrixPtr<double> Matrix_in, int OverlapLevel_in,
                     const std::set<GlobalIndexType> &rowIndicesForThisRank) :
  Matrix_(Matrix_in),
  OverlapLevel_(OverlapLevel_in)
{
  _eComm = MPIWrapper::EpetraComm(Matrix_->getComm());
  _eMatrix = T_to_E::matrix(Matrix_);
  
  NumMyRowsA_ = Matrix_->getLocalNumRows();

  // construct the external matrix

  using go_type = TMatrix<double>::global_ordinal_type;
  BuildMap<go_type>(OverlapLevel_in, rowIndicesForThisRank);

  eMap_ = T_to_E::map(Map_);
  
  ExtMatrix_ = rcp( new Epetra_CrsMatrix(::Copy,*ExtMap_,*eMap_,0) );

  auto A_ptr = A();
  auto A_rowMap = A_ptr->RowMatrixRowMap();
  
  ExtImporter_ = rcp( new Epetra_Import(*ExtMap_,A_rowMap) );
  
  ExtMatrix_->Import(*A_ptr,*ExtImporter_,Insert);
  ExtMatrix_->FillComplete(A_ptr->OperatorDomainMap(),*eMap_);

  Importer_ = rcp( new Epetra_Import(*eMap_,A_rowMap) );

  // fix indices for overlapping matrix
  NumMyRowsB_ = B().NumMyRows();
  NumMyRows_ = NumMyRowsA_ + NumMyRowsB_;
  NumMyCols_ = NumMyRows_;

  NumMyDiagonals_ = A_ptr->NumMyDiagonals() + B().NumMyDiagonals();

  NumMyNonzeros_ = A_ptr->NumMyNonzeros() + B().NumMyNonzeros();
  long long NumMyNonzeros_tmp = NumMyNonzeros_;
  Comm().SumAll(&NumMyNonzeros_tmp,&NumGlobalNonzeros_,1);
  MaxNumEntries_ = A_ptr->MaxNumEntries();
  
  if (MaxNumEntries_ < B().MaxNumEntries())
    MaxNumEntries_ = B().MaxNumEntries();
}

//! Constructor for an exact match to IfPack_OverlappingMatrix's behavior.
OverlappingRowMatrix::
OverlappingRowMatrix(const TMatrixPtr<double> Matrix_in, int OverlapLevel_in) :
  Matrix_(Matrix_in),
  OverlapLevel_(OverlapLevel_in)
{
  _eComm = MPIWrapper::EpetraComm(Matrix_->getComm());
  _eMatrix = T_to_E::matrix(Matrix_);
  
  // nothing to do as well with one process
  if (Comm().NumProc() == 1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported");
  }

  auto A_ptr = A();
  
  NumMyRowsA_ = A_ptr->NumMyRows();

  // construct the external matrix
  using go_type = TMatrix<double>::global_ordinal_type;
  BuildMap<go_type>(OverlapLevel_in);

  eMap_ = T_to_E::map(Map_);
  ExtMatrix_ = rcp( new Epetra_CrsMatrix(::Copy,*ExtMap_,*eMap_,0) );

  ExtImporter_ = rcp( new Epetra_Import(*ExtMap_,A_ptr->RowMatrixRowMap()) );
  ExtMatrix_->Import(*A_ptr,*ExtImporter_,Insert);
  ExtMatrix_->FillComplete(A_ptr->OperatorDomainMap(),*eMap_);

  Importer_ = rcp( new Epetra_Import(*eMap_,A_ptr->RowMatrixRowMap()) );

  // fix indices for overlapping matrix
  NumMyRowsB_ = B().NumMyRows();
  NumMyRows_ = NumMyRowsA_ + NumMyRowsB_;
  NumMyCols_ = NumMyRows_;

  NumMyDiagonals_ = A_ptr->NumMyDiagonals() + B().NumMyDiagonals();

  NumMyNonzeros_ = A_ptr->NumMyNonzeros() + B().NumMyNonzeros();
  long long NumMyNonzeros_tmp = NumMyNonzeros_;
  Comm().SumAll(&NumMyNonzeros_tmp,&NumGlobalNonzeros_,1);
  MaxNumEntries_ = A_ptr->MaxNumEntries();

  if (MaxNumEntries_ < B().MaxNumEntries())
    MaxNumEntries_ = B().MaxNumEntries();
}

// ======================================================================
// Constructor for the case of one core per subdomain (the only case supported by this Camellia version)
OverlappingRowMatrix::
OverlappingRowMatrix(const TMatrixPtr<double> Matrix_in,
                     int OverlapLevel_in, MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter,
                     bool hierachical)  :
  Matrix_(Matrix_in),
  OverlapLevel_(OverlapLevel_in)
{
  _eComm = MPIWrapper::EpetraComm(Matrix_->getComm());
  _eMatrix = T_to_E::matrix(Matrix_);
  
  auto A_ptr = A();
  
  NumMyRowsA_ = A_ptr->NumMyRows();

  // construct the external matrix
  using go_type = TMatrix<double>::global_ordinal_type;
  BuildMap<go_type>(OverlapLevel_in, mesh, dofInterpreter, hierachical);

  eMap_ = T_to_E::map(Map_);
  ExtMatrix_ = rcp( new Epetra_CrsMatrix(::Copy,*ExtMap_,*eMap_,0) );

  ExtImporter_ = rcp( new Epetra_Import(*ExtMap_,A_ptr->RowMatrixRowMap()) );
  ExtMatrix_->Import(*A_ptr,*ExtImporter_,Insert);
  ExtMatrix_->FillComplete(A_ptr->OperatorDomainMap(),*eMap_);

  Importer_ = rcp( new Epetra_Import(*eMap_,A_ptr->RowMatrixRowMap()) );

  // fix indices for overlapping matrix
  NumMyRowsB_ = B().NumMyRows();
  NumMyRows_ = NumMyRowsA_ + NumMyRowsB_;
  NumMyCols_ = NumMyRows_;

  NumMyDiagonals_ = A_ptr->NumMyDiagonals() + B().NumMyDiagonals();

  NumMyNonzeros_ = A_ptr->NumMyNonzeros() + B().NumMyNonzeros();
  long long NumMyNonzeros_tmp = NumMyNonzeros_;
  Comm().SumAll(&NumMyNonzeros_tmp,&NumGlobalNonzeros_,1);
  MaxNumEntries_ = A_ptr->MaxNumEntries();

  if (MaxNumEntries_ < B().MaxNumEntries())
    MaxNumEntries_ = B().MaxNumEntries();
}

// ======================================================================
int OverlappingRowMatrix::
NumMyRowEntries(int MyRow, int & NumEntries) const
{
  if (MyRow < NumMyRowsA_)
    return(A()->NumMyRowEntries(MyRow,NumEntries));
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
    auto A_ptr = A();
    
    ierr = A_ptr->ExtractMyRowCopy(MyRow,Length,NumEntries,Values,Indices);
    // pretty sure we need to remap Indices here, like so:
    int offset = 0; // for when we delete entries
    for (int i=0; i<NumEntries; i++)
    {
      int lid_A = Indices[i];
      auto gid_A = A_ptr->RowMatrixColMap().GID64(lid_A);
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

  return ierr;
}

// ======================================================================
int OverlappingRowMatrix::
ExtractDiagonalCopy(Epetra_Vector & Diagonal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
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
  
  auto A_ptr = A();

  // matvec with A (local rows)
  for (int i = 0 ; i < NumMyRowsA_ ; ++i)
  {
    for (int k = 0 ; k < NumVectors ; ++k)
    {
      int Nnz;
      int info = A_ptr->ExtractMyRowCopy(i,MaxNumEntries_,Nnz,
                                         &Val[0], &Ind[0]);
      TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::invalid_argument, "ExtractMyRowCopy() failed with value " << info);
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
      int info = B().ExtractMyRowCopy(i,MaxNumEntries_,Nnz,
                                      &Val[0], &Ind[0]);
      TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::invalid_argument, "ExtractMyRowCopy() failed with value " << info);
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
  int info = Multiply(UseTranspose(),X,Y);
  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::invalid_argument, "Multiply() failed with value " << info);
  return(0);
}

// ======================================================================
int OverlappingRowMatrix::
ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  return -1;
}

// ======================================================================
Epetra_CrsMatrix& OverlappingRowMatrix::B() const
{
  return(*ExtMatrix_);
}
// ======================================================================
const Epetra_BlockMap& OverlappingRowMatrix::Map() const
{
  return(*eMap_);
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

