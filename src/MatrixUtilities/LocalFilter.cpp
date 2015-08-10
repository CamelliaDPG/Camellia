#include "Ifpack_ConfigDefs.h"

#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Map.h"
#include "Epetra_BlockMap.h"
#include "LocalFilter.h"

#include <vector>

#include "CamelliaDebugUtility.h"

using namespace Camellia;

// Generalization of IfPack_LocalFilter

//==============================================================================
LocalFilter::LocalFilter(const Teuchos::RefCountPtr<const Epetra_RowMatrix>& Matrix, std::vector<int> whichLocalRows) :
  Matrix_(Matrix),
  NumRows_(0),
  NumNonzeros_(0),
  MaxNumEntries_(0),
  MaxNumEntriesA_(0)
{
  sprintf(Label_,"%s","LocalFilter");
  _subdomainRows = whichLocalRows;
  
  // DEBUGGING/testing
  for (int localRow : whichLocalRows)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(localRow < 0, std::invalid_argument, "Invalid local row");
  }

#ifdef HAVE_MPI
  SerialComm_ = Teuchos::rcp( new Epetra_MpiComm(MPI_COMM_SELF) );
#else
  SerialComm_ = Teuchos::rcp( new Epetra_SerialComm );
#endif

//  Camellia::print("new LocalFilter's subdomain rows",whichLocalRows);
  
  _ordinalLookup = std::vector<int>(Matrix->NumMyRows(), -1);
  int i=0;
  for (int myRow : _subdomainRows)
  {
    _ordinalLookup[myRow] = i++;
  }
  
  // localized matrix has all the local rows of Matrix
  NumRows_ = _subdomainRows.size();

#if !defined(EPETRA_NO_32BIT_GLOBAL_INDICES) || !defined(EPETRA_NO_64BIT_GLOBAL_INDICES)
  // build a linear map, based on the serial communicator
  Map_ = Teuchos::rcp( new Epetra_Map(NumRows_,0,*SerialComm_) );
#endif

  // NumEntries_ will contain the actual number of nonzeros
  // for each localized row (that is, without external nodes,
  // and always with the diagonal entry)
  NumEntries_.resize(NumRows_);

  // want to store the diagonal vector. FIXME: am I really useful?
  Diagonal_ = Teuchos::rcp( new Epetra_Vector(*Map_) );
  if (Diagonal_ == Teuchos::null) IFPACK_CHK_ERRV(-5);
  
  // store this for future access to ExtractMyRowCopy().
  // This is the # of nonzeros in the non-local matrix
  MaxNumEntriesA_ = Matrix->MaxNumEntries();
  // tentative value for MaxNumEntries. This is the number of
  // nonzeros in the local matrix
  MaxNumEntries_ = Matrix->MaxNumEntries();

  // ExtractMyRowCopy() will use these vectors
  Indices_.resize(MaxNumEntries_);
  Values_.resize(MaxNumEntries_);

  // now compute:
  // - the number of nonzero per row
  // - the total number of nonzeros
  // - the diagonal entries

  // compute nonzeros (total and per-row), and store the
  // diagonal entries (already modified)
  int ActualMaxNumEntries = 0;

  int i_ordinal = 0;
  for (int i : _subdomainRows) {
    NumEntries_[i_ordinal] = 0;
    int Nnz, NewNnz = 0;
    IFPACK_CHK_ERRV(ExtractMyRowCopy(i_ordinal,MaxNumEntries_,Nnz,&Values_[0],&Indices_[0]));

//    {
//      // DEBUGGING:
//      std::cout << i_ordinal << ":";
//    }
    
    for (int j = 0 ; j < Nnz ; ++j) {
      ++NewNnz;
      
      if (Indices_[j] == i_ordinal)
        (*Diagonal_)[i_ordinal] = Values_[j];
      
//      {
//        // DEBUGGING:
//        std::cout << " " << Values_[j];
//      }
    }
    
//    {
//      // DEBUGGING:
//      std::cout << std::endl;
//    }

    if (NewNnz > ActualMaxNumEntries)
      ActualMaxNumEntries = NewNnz;

    NumNonzeros_ += NewNnz;
    NumEntries_[i_ordinal] = NewNnz;
    
    i_ordinal++;
  }
 
  
  
  MaxNumEntries_ = ActualMaxNumEntries;
}

//==============================================================================
int LocalFilter::
ExtractMyRowCopy(int MyRowOrdinal, int Length, int & NumEntries,
		 double *Values, int * Indices) const
{
  if ((MyRowOrdinal < 0) || (MyRowOrdinal >= NumRows_)) {
    IFPACK_CHK_ERR(-1); // range not valid
  }
  
  int MyRow = _subdomainRows[MyRowOrdinal];
  TEUCHOS_TEST_FOR_EXCEPTION(MyRow == -1, std::invalid_argument, "Invalid local row ordinal");

  if (Length < NumEntries_[MyRowOrdinal])
    return(-1);

  // always extract using the object Values_ and Indices_.
  // This is because I need more space than that given by
  // the user (for the external nodes)
  int Nnz;
  int ierr = Matrix_->ExtractMyRowCopy(MyRow,MaxNumEntriesA_,Nnz,
				       &Values_[0],&Indices_[0]);

  IFPACK_CHK_ERR(ierr);

  // populate the user's vectors, add diagonal if not found
  NumEntries = 0;

  for (int j = 0 ; j < Nnz ; ++j) {
    // only local indices
    if (_ordinalLookup[Indices_[j]] != -1 ) {
      Indices[NumEntries] = _ordinalLookup[Indices_[j]];
      Values[NumEntries] = Values_[j];
      ++NumEntries;
    }
  }
    
  return(0);

}

//==============================================================================
int LocalFilter::ExtractDiagonalCopy(Epetra_Vector & Diagonal) const
{
  if (!Diagonal.Map().SameAs(*Map_))
    IFPACK_CHK_ERR(-1);
  Diagonal = *Diagonal_;
  return(0);
}

//==============================================================================
int LocalFilter::Apply(const Epetra_MultiVector& X,
	  Epetra_MultiVector& Y) const 
{

  // skip expensive checks, I suppose input data are ok

  Y.PutScalar(0.0);
  int NumVectors = Y.NumVectors();

  double** X_ptr;
  double** Y_ptr;
  X.ExtractView(&X_ptr);
  Y.ExtractView(&Y_ptr);

  int i_ordinal = 0;
  for (int i : _subdomainRows) {
    
    int Nnz;
    int ierr = Matrix_->ExtractMyRowCopy(i,MaxNumEntriesA_,Nnz,&Values_[0],
                                         &Indices_[0]);
    IFPACK_CHK_ERR(ierr);

    for (int j = 0 ; j < Nnz ; ++j) {
      // include if
      if (_ordinalLookup[Indices_[j]] != -1) {
        for (int k = 0 ; k < NumVectors ; ++k)
          Y_ptr[k][i_ordinal] += Values_[j] * X_ptr[k][_ordinalLookup[Indices_[j]]];
      }
    }
    i_ordinal++;
  }

  return(0);
}

//==============================================================================
int LocalFilter::ApplyInverse(const Epetra_MultiVector& X,
		 Epetra_MultiVector& Y) const
{
  IFPACK_CHK_ERR(-1); // not implemented
}

//==============================================================================
const Epetra_BlockMap& LocalFilter::Map() const
{
  return(*Map_);
}
