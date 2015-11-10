/*
//@HEADER
// ***********************************************************************
//
//                  Camellia Additive Schwarz:
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

#ifndef CAMELLIA_ADDITIVESCHWARZ_H
#define CAMELLIA_ADDITIVESCHWARZ_H

#include "Ifpack_Condest.h"
#include "Ifpack_ConfigDefs.h"
#include "Ifpack_Preconditioner.h"
#include "Ifpack_ConfigDefs.h"
#include "Ifpack_Preconditioner.h"
#include "Ifpack_LocalFilter.h"
#include "Ifpack_NodeFilter.h"
#include "Ifpack_SingletonFilter.h"
#include "Ifpack_Utils.h"
#include "OverlappingRowMatrix.h"
#include "Epetra_CombineMode.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Map.h"
#include "Epetra_Comm.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Time.h"
#include "Epetra_LinearProblem.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "CamelliaDebugUtility.h"
#include "LocalFilter.h"
#include "Mesh.h"
#include "DofInterpreter.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"

#include "Epetra_Operator_to_Epetra_Matrix.h"

//! AdditiveSchwarz: a class to define Additive Schwarz preconditioners of Epetra_RowMatrix's; largely copied from IfPack_AdditiveSchwarz, then modified to use Camellia's OverlappingRowMatrix in place of the IfPack_OverlappingRowMatrix.

/*!
  Note: this class, and this documentation, are essentially copied from
  IfPack_AdditiveSchwarz.  The only changes are to support a slight modification
  of the definition of the overlapping matrix; we replace the IfPack_OverlappingRowMatrix
  with as OverlappingRowMatrix.

  Modifications support definitions of overlap level in terms of a Camellia
  Mesh.  Zero-level overlap means that the owner of each cell sees all the
  degrees of freedom for that cell--including all the traces, even when
  the cell owner does not own them.  One-level overlap means that the owner
  of a cell sees all degrees of freedom belonging to the neighbors of that
  cell (the cells that share sides with the owned cell).  Two-level overlap
  extends this to neighbors of neighbors, etc.

  Class AdditiveSchwarz enables the construction of Additive
  Schwarz (one-level overlapping domain decomposition) preconditioners,
  for a given Epetra_RowMatrix based on a Camellia Mesh.
  AdditiveSchwarz is derived from Ifpack_Preconditioner,
  itself derived from Epetra_Operator. An application of
  the Additive Schwarz preconditioner can be obtained
  by calling method ApplyInverse().

  One-level overlapping domain decomposition preconditioners use
  local solvers, of Dirichlet type. This means that the inverse of
  the local matrix (with minimal or wider overlap) is applied to
  the residual to be preconditioned.

  The preconditioner can be written as:
  \f[
  P_{AS}^{-1} = \sum_{i=1}^M P_i A_i^{-1} R_i ,
  \f]
  where \f$M\f$ is the number of subdomains (that is, the number of
  processors in
  the computation), \f$R_i\f$ is an operator that restricts the global
  vector to the vector lying on subdomain \f$i\f$, \f$P_i\f$ is the
  prolongator operator, and
  \f[
  A_i = R_i A P_i.
  \f]

  The construction of Schwarz preconditioners is mainly composed by
  two steps:
  - definition of the restriction and prolongation operator
  \f$R_i\f$ and \f$R_i^T\f$. If minimal overlap is chosen, their
  implementation is trivial, \f$R_i\f$ will return all the local
  components. For wider overlaps, instead, Epetra_Import and
  Epetra_Export will be used to import/export data. The user
  must provide both the matrix to be preconditioned (which is suppose
  to have minimal-overlap) and the matrix with wider overlap.
  - definition of a technique to apply the inverse of \f$A_i\f$.
  To solve on each subdomain, the user can adopt any class, derived
  from Ifpack_Preconditioner. This can be easily accomplished, as
  AdditiveSchwarz is templated with the solver for each subdomain.

  The local matrix \f$A_i\f$ can be filtered, to eliminate singletons, and
  reordered. At the present time, RCM and METIS can be used to reorder the
  local matrix.

  The complete list of supported parameters is reported in page \ref ifp_params.

  \author Nathan V. Roberts, ALCF.  Based on IfPack_AdditiveSchwarz by Marzio Sala, SNL 9214.

  \date Last modified on 17-Nov-2014.
*/

namespace Camellia
{

template<typename T>
class AdditiveSchwarz : public virtual Ifpack_Preconditioner
{

public:

  //@{ \name Constructors/Destructors
  //! AdditiveSchwarz constructor with given Epetra_RowMatrix.
  /*! Creates an AdditiveSchwarz preconditioner with overlap.
   * To use minimal-overlap, OverlappingMatrix is omitted
   * (as defaulted to 0).
   *
   * \param
   * Matrix - (In) Pointer to matrix to be preconditioned
   *
   * \param
   * OverlappingMatrix - (In) Pointer to the matrix extended with the
   *                     desired level of overlap.
   * \param
   * dofInterpreter - (In) the dofInterpreter for the problem
   *
   * \param
   * hierarchical - (In) if true, use parent relationships (in addition to neighbor relationships) to define Schwarz blocks.
   *
   * \param
   * dimensionForNeighborRelationship - (In) what dimension is used to determine neighbor relationship: 0 for vertex neighbors, d-1 for side neighbors, etc.
   *
   */
  AdditiveSchwarz(Epetra_RowMatrix* Matrix_in, int OverlapLevel_in, MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter,
                  bool hierarchical, int dimensionForNeighborRelationship);

  //! Destructor
  virtual ~AdditiveSchwarz() {};
  //@}

  //@{ \name Attribute set methods.

  //! If set true, transpose of this operator will be applied (not implemented).
  /*! This flag allows the transpose of the given operator to be used
   * implicitly.

  \param
   UseTranspose_in - (In) If true, multiply by the transpose of operator,
   otherwise just use operator.

  \return Integer error code, set to 0 if successful.  Set to -1 if this implementation does not support transpose.
  */
  virtual int SetUseTranspose(bool UseTranspose_in);
  //@}

  //@{ \name Mathematical functions.

  //! Applies the matrix to X, returns the result in Y.
  /*!
    \param
    X - (In) A Epetra_MultiVector of dimension NumVectors
       to multiply with matrix.
    \param
    Y -(Out) A Epetra_MultiVector of dimension NumVectors
       containing the result.

    \return Integer error code, set to 0 if successful.
    */
  virtual int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

  //! Applies the preconditioner to X, returns the result in Y.
  /*!
    \param
    X - (In) A Epetra_MultiVector of dimension NumVectors to be preconditioned.
    \param
    Y -(Out) A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.

    \warning In order to work with AztecOO, any implementation of this method
    must support the case where X and Y are the same object.
    */
  virtual int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

  //! Returns the infinity norm of the global matrix (not implemented)
  virtual double NormInf() const;
  //@}

  //@{ \name Attribute access functions

  //! Returns a character string describing the operator
  virtual const char * Label() const;

  //! Returns the current UseTranspose setting.
  virtual bool UseTranspose() const;

  //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
  virtual bool HasNormInf() const;

  //! Returns a pointer to the Epetra_Comm communicator associated with this operator.
  virtual const Epetra_Comm & Comm() const;

  //! Returns the Epetra_Map object associated with the domain of this operator.
  virtual const Epetra_Map & OperatorDomainMap() const;

  //! Returns the Epetra_Map object associated with the range of this operator.
  virtual const Epetra_Map & OperatorRangeMap() const;
  //@}

  //! Returns the Epetra_Map object for the overlapping row matrix associated with this operator.
  const Epetra_Map & OverlapMap() const;
  //@}

  //! Returns \c true if the preconditioner has been successfully initialized.
  virtual bool IsInitialized() const
  {
    return(IsInitialized_);
  }

  //! Returns \c true if the preconditioner has been successfully computed.
  virtual bool IsComputed() const
  {
    return(IsComputed_);
  }

  //! Sets the parameters.
  /*! Sets the parameter for the additive Schwarz preconditioner,
   * as well as for all the preconditioners that may need to
   * be defined on each subblock.
   * Parameters accepted by List are:
   * - \c "schwarz: combine mode" : It must be an Epetra_CombineMode.
   *     Default: \c Zero.
   *     It Can be assume of the following values:
   *   - Add: Components on the receiving processor will be added together;
   *   - Zero: Off-processor components will be ignored;
   *   - Insert: Off-processor components will be inserted into locations on
   *     receiving processor replacing existing values.
   *   - Average: Off-processor components will be averaged with existing;
   *   - AbsMax: Magnitudes of Off-processor components will be
   *     maxed with magnitudes of existing components on the receiving
   *     processor.
   * - \c "schwarz: compute condest" : if \c true, \c Compute() will
   *    estimate the condition number of the preconditioner.
   *    Default: \c true.
   */
  virtual int SetParameters(Teuchos::ParameterList& List);

  // @}

  // @{ Query methods

  //! Initialized the preconditioner.
  virtual int Initialize();

  //! Computes the preconditioner.
  virtual int Compute();

  //! Computes the estimated condition number and returns its value.
  virtual double Condest(const Ifpack_CondestType CT = Ifpack_Cheap,
                         const int MaxIters = 1550,
                         const double Tol = 1e-9,
                         Epetra_RowMatrix* Matrix_in = 0);

  //! Returns the estimated condition number, or -1.0 if not computed.
  virtual double Condest() const
  {
    return(Condest_);
  }

  //! Returns a refernence to the internally stored matrix.
  virtual const Epetra_RowMatrix& Matrix() const
  {
    return(*Matrix_);
  }

  //! Returns \c true is an overlapping matrix is present.
  virtual bool IsOverlapping() const
  {
    return(IsOverlapping_);
  }

  //! Prints major information about this preconditioner.
  virtual std::ostream& Print(std::ostream&) const;

//  virtual const T* Inverse() const
//  {
//    return(&*Inverse_);
//  }

  //! Returns the number of calls to Initialize().
  virtual int NumInitialize() const
  {
    return(NumInitialize_);
  }

  //! Returns the number of calls to Compute().
  virtual int NumCompute() const
  {
    return(NumCompute_);
  }

  //! Returns the number of calls to ApplyInverse().
  virtual int NumApplyInverse() const
  {
    return(NumApplyInverse_);
  }

  //! Returns the time spent in Initialize().
  virtual double InitializeTime() const
  {
    return(InitializeTime_);
  }

  //! Returns the time spent in Compute().
  virtual double ComputeTime() const
  {
    return(ComputeTime_);
  }

  //! Returns the time spent in ApplyInverse().
  virtual double ApplyInverseTime() const
  {
    return(ApplyInverseTime_);
  }

  //! Returns the number of flops in the initialization phase.
  virtual double InitializeFlops() const
  {
    return(InitializeFlops_);
  }

  virtual double ComputeFlops() const
  {
    return(ComputeFlops_);
  }

  virtual double ApplyInverseFlops() const
  {
    return(ApplyInverseFlops_);
  }

  //! Returns the level of overlap.
  virtual int OverlapLevel() const
  {
    return(OverlapLevel_);
  }

  //! Returns a reference to the internally stored list.
  virtual const Teuchos::ParameterList& List() const
  {
    return(List_);
  }

protected:

  // @}

  // @{ Internal methods.

  //! Copy constructor (should never be used)
  AdditiveSchwarz(const AdditiveSchwarz& RHS)
  { }

  //! Sets up the localized matrix and the singleton filter.
  int Setup();

  // @}

  // @{ Internal data.

  //! Camellia addition: the Mesh
  MeshPtr mesh_;

  //! Camellia addition: the DofInterpreter
  Teuchos::RCP<DofInterpreter> dofInterpreter_;

  //! Camellia addition: whether to use "hierarchical" overlap definition
  bool hierarchical_;

  //! Camellia addition: what dimension subcells to use to define neighbors
  int dimensionForNeighborRelationship_;

  //! Pointers to the matrix to be preconditioned.
  Teuchos::RCP<const Epetra_RowMatrix> Matrix_;
  //! Pointers to the overlapping matrix.
  Teuchos::RCP<OverlappingRowMatrix> OverlappingMatrix_;
  //! Localized version of Matrix_ or OverlappingMatrix_.
  /*
    //TODO if we choose to expose the node aware code, i.e., no ifdefs,
    //TODO then we should switch to this definition.
    Teuchos::RCP<Epetra_RowMatrix> LocalizedMatrix_;
  */

  //! Camellia modification: use our LocalFilter, and have a vector of them:
  std::vector<Teuchos::RCP<LocalFilter>> _LocalizedMatrices;
  
//  Teuchos::RCP<Ifpack_LocalFilter> LocalizedMatrix_;
  //! Contains the label of \c this object.
  string Label_;
  //! If true, the preconditioner has been successfully initialized.
  bool IsInitialized_;
  //! If true, the preconditioner has been successfully computed.
  bool IsComputed_;
  //! If \c true, solve with the transpose (not supported by all solvers).
  bool UseTranspose_;
  //! If true, overlapping is used
  bool IsOverlapping_;
  //! Level of overlap among the processors.
  int OverlapLevel_;
  //! Stores a copy of the list given in SetParameters()
  Teuchos::ParameterList List_;
  //! Combine mode for off-process elements (only if overlap is used)
  Epetra_CombineMode CombineMode_;
  //! Contains the estimated condition number.
  double Condest_;
  //! If \c true, compute the condition number estimate each time Compute() is called.
  bool ComputeCondest_;
  //! Contains the number of successful calls to Initialize().
  int NumInitialize_;
  //! Contains the number of successful call to Compute().
  int NumCompute_;
  //! Contains the number of successful call to ApplyInverse().
  mutable int NumApplyInverse_;
  //! Contains the time for all successful calls to Initialize().
  double InitializeTime_;
  //! Contains the time for all successful calls to Compute().
  double ComputeTime_;
  //! Contains the time for all successful calls to ApplyInverse().
  mutable double ApplyInverseTime_;
  //! Contains the number of flops for Initialize().
  double InitializeFlops_;
  //! Contains the number of flops for Compute().
  double ComputeFlops_;
  //! Contains the number of flops for ApplyInverse().
  mutable double ApplyInverseFlops_;
  //! Object used for timing purposes.
  Teuchos::RCP<Epetra_Time> Time_;
  //! Pointers to the local solvers.
  
  std::vector<Teuchos::RCP<T>> _Inverses;
}; // class AdditiveSchwarz<T>

//==============================================================================
template<typename T>
AdditiveSchwarz<T>::
AdditiveSchwarz(Epetra_RowMatrix* Matrix_in, int OverlapLevel_in, MeshPtr mesh,
                Teuchos::RCP<DofInterpreter> dofInterpreter, bool hierarchical,
                int dimensionForNeighborRelationship) :
  mesh_(mesh),
  dofInterpreter_(dofInterpreter),
  hierarchical_(hierarchical),
  dimensionForNeighborRelationship_(dimensionForNeighborRelationship),
  IsInitialized_(false),
  IsComputed_(false),
  UseTranspose_(false),
  IsOverlapping_(false),
  OverlapLevel_(OverlapLevel_in),
  CombineMode_(Zero),
  Condest_(-1.0),
  ComputeCondest_(true),
  NumInitialize_(0),
  NumCompute_(0),
  NumApplyInverse_(0),
  InitializeTime_(0.0),
  ComputeTime_(0.0),
  ApplyInverseTime_(0.0),
  InitializeFlops_(0.0),
  ComputeFlops_(0.0),
  ApplyInverseFlops_(0.0)
{
  // Construct a reference-counted pointer with the input matrix; don't manage the memory.
  Matrix_ = Teuchos::rcp( Matrix_in, false );

//  if (Matrix_->Comm().NumProc() == 1)
//    OverlapLevel_ = 0;

  // for us, even a 0-level overlap is overlapping, if there is more than one processor and there is more than one cell
  if ((Matrix_->Comm().NumProc() > 1) && (mesh_->numActiveElements() > 1))
    IsOverlapping_ = true;
  // Sets parameters to default values
  Teuchos::ParameterList List_in;
  SetParameters(List_in);
}

//==============================================================================
template<typename T>
int AdditiveSchwarz<T>::Setup()
{

  Epetra_RowMatrix* MatrixPtr;

  try
  {
    _LocalizedMatrices.clear();
    
    set<GlobalIndexType> myCellIDs = mesh_->cellIDsInPartition(); // we want a domain identified with each cell

    map<GlobalIndexType,set<GlobalIndexType>> overlapCellsForCell; // key: cell; value: the cells in its overlap region
    set<GlobalIndexType> allCellsOfInterestToMe;
    
    // find the overlap region for each cell.
    for (GlobalIndexType cellID : myCellIDs)
    {
      set<GlobalIndexType> cellsInOverlap = OverlappingRowMatrix::overlappingCells(cellID, mesh_, OverlapLevel_, hierarchical_,
                                                                                   dimensionForNeighborRelationship_);
      overlapCellsForCell[cellID] = cellsInOverlap;
      allCellsOfInterestToMe.insert(cellsInOverlap.begin(), cellsInOverlap.end());
    }
    
    // find the global indices that belong to each cell within the overlap regions
    map<GlobalIndexType,set<GlobalIndexType>> globalIndicesForCells; // key: cell; value: globalDofIndices
    globalIndicesForCells = dofInterpreter_->importGlobalIndicesMap(allCellsOfInterestToMe);
    
    for (GlobalIndexType cellID : myCellIDs)
    {
      set<GlobalIndexType> globalDofIndices; // global dof indices for the overlap region
      set<GlobalIndexType> cellsInOverlap = overlapCellsForCell[cellID];
      
      for (GlobalIndexType overlapCell : cellsInOverlap)
      {
        globalDofIndices.insert(globalIndicesForCells[overlapCell].begin(), globalIndicesForCells[overlapCell].end());
      }
      
      // find corresponding local rows
      vector<int> localRows;
      for (GlobalIndexType globalDofIndex : globalDofIndices)
      {
        localRows.push_back(OverlapMap().LID((GlobalIndexTypeToCast)globalDofIndex));
      }
      
      Teuchos::RCP<LocalFilter> localizedMatrix;
      
      if (OverlappingMatrix_ != Teuchos::null)
      {        
        localizedMatrix = Teuchos::rcp( new LocalFilter(OverlappingMatrix_, localRows) );
      }
      else
      {
        localizedMatrix = Teuchos::rcp( new LocalFilter(Matrix_, localRows) );
      }
      
      if (localizedMatrix == Teuchos::null)
        IFPACK_CHK_ERR(-5);
      
      _LocalizedMatrices.push_back(localizedMatrix);
      
      MatrixPtr = &*localizedMatrix;
      
      Teuchos::RCP<T> Inverse = Teuchos::rcp( new T(MatrixPtr) );
      
      if (Inverse == Teuchos::null)
        IFPACK_CHK_ERR(-5);

      _Inverses.push_back(Inverse);
    }
  }
  catch(...)
  {
    fprintf(stderr,"%s","AdditiveSchwarz Setup: problem creating local filter matrix.\n");
  }

  return(0);
}

//==============================================================================
template<typename T>
int AdditiveSchwarz<T>::SetParameters(Teuchos::ParameterList& List_in)
{
  // compute the condition number each time Compute() is invoked.
  ComputeCondest_ = List_in.get("schwarz: compute condest", ComputeCondest_);
  // combine mode
  if( Teuchos::ParameterEntry *combineModeEntry = List_in.getEntryPtr("schwarz: combine mode") )
  {
    if( typeid(std::string) == combineModeEntry->getAny().type() )
    {
      std::string mode = List_in.get("schwarz: combine mode", "Add");
      if (mode == "Add")
        CombineMode_ = Add;
//      else if (mode == "Zero")
//        CombineMode_ = Zero;
//      else if (mode == "Insert")
//        CombineMode_ = Insert;
//      else if (mode == "InsertAdd")
//        CombineMode_ = InsertAdd;
//      else if (mode == "Average")
//        CombineMode_ = Average;
//      else if (mode == "AbsMax")
//        CombineMode_ = AbsMax;
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(
          true,std::logic_error
          ,"Error, The (Epetra) combine mode of \""<<mode<<"\" is not valid!  Only the values"
//          " \"Add\", \"Zero\", \"Insert\", \"InsertAdd\", \"Average\", and \"AbsMax\" are accepted!"
           " \"Add\" is accepted!"
        );
      }
    }
    else if ( typeid(Epetra_CombineMode) == combineModeEntry->getAny().type() )
    {
      CombineMode_ = Teuchos::any_cast<Epetra_CombineMode>(combineModeEntry->getAny());
    }
    else
    {
      // Throw exception with good error message!
      Teuchos::getParameter<std::string>(List_in,"schwarz: combine mode");
    }
  }
  else
  {
    // Make the default be a string to be consistent with the valid parameters!
    List_in.get("schwarz: combine mode","Add");
  }

  // This copy may be needed by Amesos or other preconditioners.
  List_ = List_in;

  return(0);
}

//==============================================================================
template<typename T>
int AdditiveSchwarz<T>::Initialize()
{
//  cout << "Entered AdditiveSchwarz<T>::Initialize().\n";
  IsInitialized_ = false;
  IsComputed_ = false; // values required
  Condest_ = -1.0; // zero-out condest

  if (Time_ == Teuchos::null)
    Time_ = Teuchos::rcp( new Epetra_Time(Comm()) );

  Time_->ResetStartTime();

  // compute the overlapping matrix if necessary
  if (IsOverlapping_)
  {
    OverlappingMatrix_ = Teuchos::rcp( new OverlappingRowMatrix(Matrix_, OverlapLevel_, mesh_, dofInterpreter_, hierarchical_) );

    if (OverlappingMatrix_ == Teuchos::null)
    {
      IFPACK_CHK_ERR(-5);
    }
  }

  IFPACK_CHK_ERR(Setup());

  string inverseLabel;
  for (int i=0; i<_Inverses.size(); i++)
  {
    Teuchos::RCP<T> Inverse = _Inverses[i];
    Teuchos::RCP<LocalFilter> localizedMatrix = _LocalizedMatrices[i];
    
    if (Inverse == Teuchos::null)
      IFPACK_CHK_ERR(-5);
    
    if (localizedMatrix == Teuchos::null)
      IFPACK_CHK_ERR(-5);
    
    IFPACK_CHK_ERR(Inverse->SetUseTranspose(UseTranspose()));
    IFPACK_CHK_ERR(Inverse->SetParameters(List_));
    IFPACK_CHK_ERR(Inverse->Initialize());
    
    inverseLabel = Inverse->Label();
  }
  
  // Label is for Aztec-like solvers
  Label_ = "AdditiveSchwarz, ";
  if (UseTranspose())
    Label_ += ", transp";
  Label_ += ", ov = " + Ifpack_toString(OverlapLevel_)
            + ", local solver = \n\t\t***** `" + inverseLabel + "'";

  IsInitialized_ = true;
  ++NumInitialize_;
  InitializeTime_ += Time_->ElapsedTime();

#ifdef IFPACK_FLOPCOUNTERS
  // count flops by summing up all the InitializeFlops() in each
  // Inverse. Each Inverse() can only give its flops -- it acts on one
  // process only
  double partial = Inverse_->InitializeFlops();
  double total;
  Comm().SumAll(&partial, &total, 1);
  InitializeFlops_ += total;
#endif

  return(0);
}

//==============================================================================
template<typename T>
int AdditiveSchwarz<T>::Compute()
{
  if (IsInitialized() == false)
    IFPACK_CHK_ERR(Initialize());

  Time_->ResetStartTime();
  IsComputed_ = false;
  Condest_ = -1.0;

  string inverseLabel;
  for (int i=0; i<_Inverses.size(); i++)
  {
    Teuchos::RCP<T> Inverse = _Inverses[i];
    int err = Inverse->Compute();
    if (err != 0)
    {
      Teuchos::RCP<LocalFilter> localizedMatrix = _LocalizedMatrices[i];
      ostringstream matrixLocation;
      matrixLocation << Teuchos::GlobalMPISession::getRank() << "." << i << ".dat";
      EpetraExt::RowMatrixToMatrixMarketFile(matrixLocation.str().c_str(),*localizedMatrix, NULL, NULL, false); // false: don't write header
      cout << "AdditiveSchwarz: inverse computation returned error on rank " << Teuchos::GlobalMPISession::getRank() << ", local matrix number " << i;
      cout << "; wrote matrix to " << matrixLocation.str() << endl;
    }
    IFPACK_CHK_ERR(err);
    inverseLabel = Inverse->Label();
  }
  
  IsComputed_ = true; // need this here for Condest(Ifpack_Cheap)
  ++NumCompute_;
  ComputeTime_ += Time_->ElapsedTime();
  
#ifdef IFPACK_FLOPCOUNTERS
  // sum up flops
  double partial = Inverse->ComputeFlops();
  double total;
  Comm().SumAll(&partial, &total, 1);
  ComputeFlops_ += total;
#endif
  
  // reset the Label
  string R = "";
  
  if (ComputeCondest_)
    Condest(Ifpack_Cheap);
  
  // add Condest() to label
  Label_ = "AdditiveSchwarz, ov = " + Ifpack_toString(OverlapLevel_)
  + ", local solver = \n\t\t***** `" + string(inverseLabel) + "'"
  + "\n\t\t***** " + R + "Condition number estimate = "
  + Ifpack_toString(Condest_);

  return(0);
}

//==============================================================================
template<typename T>
int AdditiveSchwarz<T>::SetUseTranspose(bool UseTranspose_in)
{
  // store the flag -- it will be set in Initialize() if Inverse_ does not
  // exist.
  UseTranspose_ = UseTranspose_in;

  for (Teuchos::RCP<T> Inverse : _Inverses)
  {
    // If Inverse_ exists, pass it right now.
    if (Inverse!=Teuchos::null)
      IFPACK_CHK_ERR(Inverse->SetUseTranspose(UseTranspose_in));
  }
  
  return(0);
}

//==============================================================================
template<typename T>
int AdditiveSchwarz<T>::
Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  IFPACK_CHK_ERR(Matrix_->Apply(X,Y));
  return(0);
}

//==============================================================================
template<typename T>
double AdditiveSchwarz<T>::NormInf() const
{
  return(-1.0);
}

//==============================================================================
template<typename T>
const char * AdditiveSchwarz<T>::Label() const
{
  return(Label_.c_str());
}

//==============================================================================
template<typename T>
bool AdditiveSchwarz<T>::UseTranspose() const
{
  return(UseTranspose_);
}

//==============================================================================
template<typename T>
bool AdditiveSchwarz<T>::HasNormInf() const
{
  return(false);
}

//==============================================================================
template<typename T>
const Epetra_Comm & AdditiveSchwarz<T>::Comm() const
{
  return(Matrix_->Comm());
}

//==============================================================================
template<typename T>
const Epetra_Map & AdditiveSchwarz<T>::OperatorDomainMap() const
{
  return(Matrix_->OperatorDomainMap());
}

//==============================================================================
template<typename T>
const Epetra_Map & AdditiveSchwarz<T>::OperatorRangeMap() const
{
  return(Matrix_->OperatorRangeMap());
}

  //==============================================================================
  template<typename T>
  const Epetra_Map & AdditiveSchwarz<T>::OverlapMap() const
  {
    if (IsOverlapping_)
      return OverlappingMatrix_->OperatorRangeMap();
    else
      return Matrix_->OperatorRangeMap();
  }
  
//==============================================================================
template<typename T>
int AdditiveSchwarz<T>::
ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  // compute the preconditioner if not done by the user
  if (!IsComputed())
    IFPACK_CHK_ERR(-3);

  int NumVectors = X.NumVectors();

  if (NumVectors != Y.NumVectors())
    IFPACK_CHK_ERR(-2); // wrong input

  Time_->ResetStartTime();

  Teuchos::RCP<Epetra_MultiVector> OverlappingX;
  Teuchos::RCP<Epetra_MultiVector> OverlappingY;
  Teuchos::RCP<Epetra_MultiVector> Xtmp;

  // for flop count, see bottom of this function
#ifdef IFPACK_FLOPCOUNTERS
  double pre_partial = Inverse_->ApplyInverseFlops();
  double pre_total;
  Comm().SumAll(&pre_partial, &pre_total, 1);
#endif

  // process overlap, may need to create vectors and import data
  if (IsOverlapping())
  {
    OverlappingX = Teuchos::rcp( new Epetra_MultiVector(OverlappingMatrix_->RowMatrixRowMap(),
                                 X.NumVectors()) );
    OverlappingY = Teuchos::rcp( new Epetra_MultiVector(OverlappingMatrix_->RowMatrixRowMap(),
                                 Y.NumVectors()) );
    if (OverlappingY == Teuchos::null) IFPACK_CHK_ERR(-5);
    OverlappingY->PutScalar(0.0);
    OverlappingX->PutScalar(0.0);
    IFPACK_CHK_ERR(OverlappingMatrix_->ImportMultiVector(X,*OverlappingX,Insert));
    // FIXME: this will not work with overlapping and non-zero starting
    // solutions. The same for other cases below.
    // IFPACK_CHK_ERR(OverlappingMatrix_->ImportMultiVector(Y,*OverlappingY,Insert));
  }
  else
  {
    Xtmp = Teuchos::rcp( new Epetra_MultiVector(X) );
    OverlappingX = Xtmp;
    OverlappingY = Teuchos::rcp( &Y, false );
    OverlappingY->PutScalar(0.0);
  }
  
  Epetra_SerialComm SerialComm;
  
  // iterate over the Inverses, extracting appropriate part of OverlappingX and adding results into OverlappingY
  for (int i=0; i<_Inverses.size(); i++)
  {
    Teuchos::RCP<T> Inverse = _Inverses[i];
    Teuchos::RCP<LocalFilter> localFilter = _LocalizedMatrices[i];

    std::vector<int> subdomainRows = localFilter->LIDsInOriginalMatrix();
    
    Epetra_Map    localMap(subdomainRows.size(), subdomainRows.size(), &subdomainRows[0], 0, SerialComm);
    
    Epetra_MultiVector localX(localMap, NumVectors);
    Epetra_MultiVector localY(localMap, NumVectors);
    
    // extract the local X values:
    for (int vectorOrdinal=0; vectorOrdinal<NumVectors; vectorOrdinal++)
    {
      for (int subdomainLID=0; subdomainLID < subdomainRows.size(); subdomainLID++)
      {
        localX[vectorOrdinal][subdomainLID] = (*OverlappingX)[vectorOrdinal][subdomainRows[subdomainLID]];
      }
    }
    
    IFPACK_CHK_ERR(Inverse->ApplyInverse(localX,localY));

    // extract and sum into Y values:
    for (int vectorOrdinal=0; vectorOrdinal<NumVectors; vectorOrdinal++)
    {
      for (int subdomainLID=0; subdomainLID < subdomainRows.size(); subdomainLID++)
      {
        (*OverlappingY)[vectorOrdinal][subdomainRows[subdomainLID]] += localY[vectorOrdinal][subdomainLID];
      }
    }
  }

  if (IsOverlapping())
  {
    IFPACK_CHK_ERR(OverlappingMatrix_->ExportMultiVector(*OverlappingY,Y,
                   CombineMode_));
  }

#ifdef IFPACK_FLOPCOUNTERS
  // add flops. Note the we only have to add the newly counted
  // flops -- and each Inverse returns the cumulative sum
  double partial = Inverse_->ApplyInverseFlops();
  double total;
  Comm().SumAll(&partial, &total, 1);
  ApplyInverseFlops_ += total - pre_total;
#endif

  // FIXME: right now I am skipping the overlap and singletons
  ++NumApplyInverse_;
  ApplyInverseTime_ += Time_->ElapsedTime();

  return(0);
}

//==============================================================================
template<typename T>
std::ostream& AdditiveSchwarz<T>::
Print(std::ostream& os) const
{
#ifdef IFPACK_FLOPCOUNTERS
  double IF = InitializeFlops();
  double CF = ComputeFlops();
  double AF = ApplyInverseFlops();

  double IFT = 0.0, CFT = 0.0, AFT = 0.0;
  if (InitializeTime() != 0.0)
    IFT = IF / InitializeTime();
  if (ComputeTime() != 0.0)
    CFT = CF / ComputeTime();
  if (ApplyInverseTime() != 0.0)
    AFT = AF / ApplyInverseTime();
#endif

  if (Matrix().Comm().MyPID())
    return(os);

  os << endl;
  os << "================================================================================" << endl;
  os << "AdditiveSchwarz, overlap level = " << OverlapLevel_ << endl;
  if (CombineMode_ == Insert)
    os << "Combine mode                          = Insert" << endl;
  else if (CombineMode_ == Add)
    os << "Combine mode                          = Add" << endl;
  else if (CombineMode_ == Zero)
    os << "Combine mode                          = Zero" << endl;
  else if (CombineMode_ == Average)
    os << "Combine mode                          = Average" << endl;
  else if (CombineMode_ == AbsMax)
    os << "Combine mode                          = AbsMax" << endl;

  os << "Condition number estimate             = " << Condest_ << endl;
  os << "Global number of rows                 = " << Matrix_->NumGlobalRows64() << endl;

#ifdef HAVE_IFPACK_PARALLEL_SUBDOMAIN_SOLVERS
  os << endl;
  os << "================================================================================" << endl;
  os << "Subcommunicator stats" << endl;
  os << "Number of MPI processes in simulation: " << NumMpiProcs_ << endl;
  os << "Number of subdomains: " << NumSubdomains_ << endl;
  os << "Number of MPI processes per subdomain: " << NumMpiProcsPerSubdomain_ << endl;
#endif

  os << endl;
  os << "Phase           # calls   Total Time (s)       Total MFlops     MFlops/s" << endl;
  os << "-----           -------   --------------       ------------     --------" << endl;
  os << "Initialize()    "   << std::setw(5) << NumInitialize()
     << "  " << std::setw(15) << InitializeTime()
#ifdef IFPACK_FLOPCOUNTERS
     << "  " << std::setw(15) << 1.0e-6 * IF
     << "  " << std::setw(15) << 1.0e-6 * IFT
#endif
     << endl;
  os << "Compute()       "   << std::setw(5) << NumCompute()
     << "  " << std::setw(15) << ComputeTime()
#ifdef IFPACK_FLOPCOUNTERS
     << "  " << std::setw(15) << 1.0e-6 * CF
     << "  " << std::setw(15) << 1.0e-6 * CFT
#endif
     << endl;
  os << "ApplyInverse()  "   << std::setw(5) << NumApplyInverse()
     << "  " << std::setw(15) << ApplyInverseTime()
#ifdef IFPACK_FLOPCOUNTERS
     << "  " << std::setw(15) << 1.0e-6 * AF
     << "  " << std::setw(15) << 1.0e-6 * AFT
#endif
     << endl;
  os << "================================================================================" << endl;
  os << endl;

  return(os);
}

//==============================================================================
template<typename T>
double AdditiveSchwarz<T>::
Condest(const Ifpack_CondestType CT, const int MaxIters,
        const double Tol, Epetra_RowMatrix* Matrix_in)
{
  if (!IsComputed()) // cannot compute right now
    return(-1.0);

  Condest_ = Ifpack_Condest(*this, CT, MaxIters, Tol, Matrix_in);

  return(Condest_);
}

} // namespace Camellia

#endif // CAMELLIA_ADDITIVESCHWARZ_H
