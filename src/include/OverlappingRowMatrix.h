
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

// ***********************************************************************
//@HEADER
*/

#ifndef CAMELLIA_OVERLAPPINGROWMATRIX_H
#define CAMELLIA_OVERLAPPINGROWMATRIX_H

#include "Ifpack_ConfigDefs.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CombineMode.h"
#include "Teuchos_RCP.hpp"
#include "Epetra_Import.h"
#include "Epetra_Map.h"
#ifdef HAVE_IFPACK_PARALLEL_SUBDOMAIN_SOLVERS
#include "Epetra_IntVector.h"
#else
# ifdef IFPACK_NODE_AWARE_CODE
# include "Epetra_IntVector.h"
# endif
#endif

class Epetra_Map;
class Epetra_BlockMap;
class Epetra_CrsMatrix;
class Epetra_Comm;

//! OverlappingRowMatrix: matrix with ghost rows, based on Epetra_RowMatrix.  This class is essentially copied from IfPack_OverlappingRowMatrix, with modifications to support a mesh-based definition of overlap levels.
//
/*!
 Note: this class, and this documentation, are essentially copied from
 IfPack_OverlappingRowMatrix.
 
 Modifications support definitions of overlap level in terms of a Camellia
 Mesh.
 
 Zero-level overlap means that the owner of each cell sees all the
 degrees of freedom for that cell--including all the traces, even when
 the cell owner does not own them.
 
 Nonzero-level overlap has a different meaning depending on the value of 
 hierarchical provided on construction.  Using hierarchical = true is often
 appropriate when doing h-multigrid; while hierarchical = false would be
 appropriate for p-multigrid.
 
 When hierarchical is false, one-level overlap means that the owner of a
 cell sees all degrees of freedom belonging to the neighbors of that cell
 (the cells that share sides with the owned cell).  Two-level overlap extends
 this to neighbors of neighbors, etc.
 
 When hierarchical is true, one-level overlap means that the owner of a
 cell sees all degrees of freedom belonging to its siblings (the children
 of its parent).  Two-level overlap extends this to all descendants of its
 grandparent, etc.
 
 \author Nathan V. Roberts, ALCF.  Based on IfPack_OverlappingRowMatrix.
 
 \date Last modified on 26-Mar-2015.
 */

// Camellia includes:
#include "Mesh.h"
#include "DofInterpreter.h"

namespace Camellia {

  class OverlappingRowMatrix : public virtual Epetra_RowMatrix {

  public:

  //@{ Constructors/Destructors
  //! Constructor for mesh-based overlap levels, hierarchical and otherwise.
  OverlappingRowMatrix(const Teuchos::RCP<const Epetra_RowMatrix>& Matrix_in,
                       int OverlapLevel_in, MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter,
                       bool hierarchical = false);
  OverlappingRowMatrix(const Teuchos::RCP<const Epetra_RowMatrix>& Matrix_in, int OverlapLevel_in,
                       const std::set<GlobalIndexType> &rowIndicesForThisRank);

  //! Constructor for an exact match to IfPack_OverlappingMatrix's behavior.
  OverlappingRowMatrix(const Teuchos::RCP<const Epetra_RowMatrix>& Matrix_in, int OverlapLevel_in);
  ~OverlappingRowMatrix() {};
  //@}

  //@{ \name Matrix data extraction routines

  //! Returns the number of nonzero entries in MyRow.
  /*! 
    \param 
    MyRow - (In) Local row.
    \param 
    NumEntries - (Out) Number of nonzero values present.

    \return Integer error code, set to 0 if successful.
    */
  virtual int NumMyRowEntries(int MyRow, int & NumEntries) const;

  //! Returns the maximum of NumMyRowEntries() over all rows.
  virtual int MaxNumEntries() const
  {
    return(MaxNumEntries_);
  }

  //! Returns a copy of the specified local row in user-provided arrays.
  /*! 
    \param
    MyRow - (In) Local row to extract.
    \param
    Length - (In) Length of Values and Indices.
    \param
    NumEntries - (Out) Number of nonzero entries extracted.
    \param
    Values - (Out) Extracted values for this row.
    \param 
    Indices - (Out) Extracted global column indices for the corresponding values.

    \return Integer error code, set to 0 if successful.
    */
  virtual int ExtractMyRowCopy(int MyRow, int Length, int & NumEntries, double *Values, int * Indices) const;

  //! Returns a copy of the main diagonal in a user-provided vector.
  /*! 
    \param
    Diagonal - (Out) Extracted main diagonal.

    \return Integer error code, set to 0 if successful.
    */
  virtual int ExtractDiagonalCopy(Epetra_Vector & Diagonal) const;
  //@}

  //@{ \name Mathematical functions.

  //! Returns the result of a Epetra_RowMatrix multiplied by a Epetra_MultiVector X in Y.
  /*! 
    \param 
    TransA -(In) If true, multiply by the transpose of matrix, otherwise just use matrix.
    \param 
    X - (In) A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
    \param 
    Y -(Out) A Epetra_MultiVector of dimension NumVectorscontaining result.

    \return Integer error code, set to 0 if successful.
    */
  virtual int Multiply(bool TransA, const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

  //! Returns result of a local-only solve using a triangular Epetra_RowMatrix with Epetra_MultiVectors X and Y (NOT IMPLEMENTED).
  virtual int Solve(bool Upper, bool Trans, bool UnitDiagonal, const Epetra_MultiVector& X, 
		    Epetra_MultiVector& Y) const
  {
    IFPACK_RETURN(-1); // not implemented 
  }

  virtual int Apply(const Epetra_MultiVector& X,
		    Epetra_MultiVector& Y) const;

  virtual int ApplyInverse(const Epetra_MultiVector& X,
			   Epetra_MultiVector& Y) const;
  //! Computes the sum of absolute values of the rows of the Epetra_RowMatrix, results returned in x (NOT IMPLEMENTED).
  virtual int InvRowSums(Epetra_Vector& x) const
  {
    IFPACK_RETURN(-1); // not implemented
  }

  //! Scales the Epetra_RowMatrix on the left with a Epetra_Vector x (NOT IMPLEMENTED).
  virtual int LeftScale(const Epetra_Vector& x)
  {
    IFPACK_RETURN(-1); // not implemented
  }

  //! Computes the sum of absolute values of the columns of the Epetra_RowMatrix, results returned in x (NOT IMPLEMENTED).
  virtual int InvColSums(Epetra_Vector& x) const
  {
    IFPACK_RETURN(-1); // not implemented
  }


  //! Scales the Epetra_RowMatrix on the right with a Epetra_Vector x (NOT IMPLEMENTED).
  virtual int RightScale(const Epetra_Vector& x) 
  {
    IFPACK_RETURN(-1); // not implemented
  }

  //@}

  //@{ \name Attribute access functions

  //! If FillComplete() has been called, this query returns true, otherwise it returns false.
  virtual bool Filled() const
  {
    return(true);
  }

  //! Returns the infinity norm of the global matrix.
  /* Returns the quantity \f$ \| A \|_\infty\f$ such that
     \f[\| A \|_\infty = \max_{1\lei\len} \sum_{i=1}^m |a_{ij}| \f].
     */ 
  virtual double NormInf() const
  {
    return(A().NormInf());
  }

  //! Returns the one norm of the global matrix.
  /* Returns the quantity \f$ \| A \|_1\f$ such that
     \f[\| A \|_1= \max_{1\lej\len} \sum_{j=1}^n |a_{ij}| \f].
     */ 
  virtual double NormOne() const
  {
    return(A().NormOne());
  }

#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  //! Returns the number of nonzero entries in the global matrix.
  virtual int NumGlobalNonzeros() const
  {
    if(A().RowMatrixRowMap().GlobalIndicesInt())
       return (int) NumGlobalNonzeros_;
    else
       throw "OverlappingRowMatrix::NumGlobalNonzeros: Global indices not int";
  }

  //! Returns the number of global matrix rows.
  virtual int NumGlobalRows() const
  {
    return(A().NumGlobalRows());
  }

  //! Returns the number of global matrix columns.
  virtual int NumGlobalCols() const
  {
    return(A().NumGlobalCols());
  }

  //! Returns the number of global nonzero diagonal entries, based on global row/column index comparisons.
  virtual int NumGlobalDiagonals() const
  {
    return(A().NumGlobalDiagonals());
  }
#endif
  //! Returns the number of nonzero entries in the global matrix.
  virtual long long NumGlobalNonzeros64() const
  {
    return(NumGlobalNonzeros_);
  }

  //! Returns the number of global matrix rows.
  virtual long long NumGlobalRows64() const
  {
    return(A().NumGlobalRows64());
  }

  //! Returns the number of global matrix columns.
  virtual long long NumGlobalCols64() const
  {
    return(A().NumGlobalCols64());
  }

  //! Returns the number of global nonzero diagonal entries, based on global row/column index comparisons.
  virtual long long NumGlobalDiagonals64() const
  {
    return(A().NumGlobalDiagonals64());
  }

  //! Returns the number of nonzero entries in the calling processor's portion of the matrix.
  virtual int NumMyNonzeros() const
  {
    return(NumMyNonzeros_);
  }

  //! Returns the number of matrix rows owned by the calling processor.
  virtual int NumMyRows() const
  {
    return(NumMyRows_);
  }

  //! Returns the number of matrix columns owned by the calling processor.
  virtual int NumMyCols() const
  {
    return(NumMyCols_);
  }

  //! Returns the number of local nonzero diagonal entries, based on global row/column index comparisons.
  virtual int NumMyDiagonals() const
  {
    return(NumMyDiagonals_);
  }

  //! If matrix is lower triangular in local index space, this query returns true, otherwise it returns false.
  virtual bool LowerTriangular() const
  {
    return(A().LowerTriangular());
  }

  //! If matrix is upper triangular in local index space, this query returns true, otherwise it returns false.
  virtual bool UpperTriangular() const
  {
    return(A().UpperTriangular());
  }

  //! Returns the Epetra_Map object associated with the rows of this matrix.
  virtual const Epetra_Map & RowMatrixRowMap() const
  {
    return(*Map_);
  }

  //! Returns the Epetra_Map object associated with the columns of this matrix.
  virtual const Epetra_Map & RowMatrixColMap() const
  {
//    return A().RowMatrixColMap();
    return(*Map_);
  }

  //! Returns the Epetra_Import object that contains the import operations for distributed operations.
  virtual const Epetra_Import * RowMatrixImporter() const
  {
    return(&*Importer_);
  }
  //@}

  // following functions are required to derive Epetra_RowMatrix objects.

  //! Sets ownership.
  int SetOwnership(bool ownership)
  {
    IFPACK_RETURN(-1);
  }

  //! Sets use transpose (not implemented).
  int SetUseTranspose(bool UseTranspose_in)
  {
    UseTranspose_ = UseTranspose_in;
    return(0);
  }

  //! Returns the current UseTranspose setting.
  bool UseTranspose() const 
  {
    return(UseTranspose_);
  }

  //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
  bool HasNormInf() const
  {
    return(A().HasNormInf());
  }

  //! Returns a pointer to the Epetra_Comm communicator associated with this operator.
  const Epetra_Comm & Comm() const
  {
    return(A().Comm());
  }

  //! Returns the Epetra_Map object associated with the domain of this operator.
  const Epetra_Map & OperatorDomainMap() const 
  {
    return(*Map_);
  }

  //! Returns the Epetra_Map object associated with the range of this operator.
  const Epetra_Map & OperatorRangeMap() const 
  {
    return(*Map_);
  }
  //@}

const Epetra_BlockMap& Map() const;

const char* Label() const{
  return(Label_.c_str());
}

const set<GlobalIndexType> & RowIndices() {
  return rowIndices_;
}
    
int OverlapLevel() const
{
  return(OverlapLevel_);
}

int ImportMultiVector(const Epetra_MultiVector& X,
                      Epetra_MultiVector& OvX,
                      Epetra_CombineMode CM = Insert);

int ExportMultiVector(const Epetra_MultiVector& OvX,
                      Epetra_MultiVector& X,
                      Epetra_CombineMode CM = Add);

private: 
  inline const Epetra_RowMatrix& A() const
  {
    return(*Matrix_);
  }

  inline Epetra_RowMatrix& B() const;


  // Camellia additions:
//    MeshPtr mesh_;
//    Teuchos::RCP<DofInterpreter> dofInterpreter_;
    
    set<GlobalIndexType> rowIndices_; // for this rank
  // end of Camellia additions
  
  int NumMyRows_;
  int NumMyCols_;
  int NumMyDiagonals_;
  int NumMyNonzeros_;

  long long NumGlobalNonzeros_;
  int MaxNumEntries_;

  int NumMyRowsA_;
  int NumMyRowsB_;

  bool UseTranspose_;

  Teuchos::RCP<const Epetra_Map> Map_;
  Teuchos::RCP<const Epetra_Import> Importer_;

  Teuchos::RCP<const Epetra_RowMatrix> Matrix_;
  Teuchos::RCP<Epetra_CrsMatrix> ExtMatrix_;
  Teuchos::RCP<Epetra_Map> ExtMap_;
  Teuchos::RCP<Epetra_Import> ExtImporter_;

  int OverlapLevel_;
  string Label_;

  template<typename int_type>
  void BuildMap(int OverlapLevel_in);
    
  template<typename int_type>
  void BuildMap(int OverlapLevel_in, const set<GlobalIndexType> &rowIndices, bool filterByRowIndices = true);
  
  template<typename int_type>
  void BuildMap(int OverlapLevel_in, MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter, bool hierarchical);

}; // class OverlappingRowMatrix

} // namespace Camellia
#endif // CAMELLIA_OVERLAPPINGROWMATRIX_H
