//
//  GMGOperator.cpp
//  Camellia
//
//  Created by Nate Roberts on 7/3/14.
//
//

#include "GMGOperator.h"
#include "GlobalDofAssignment.h"
#include "BasisSumFunction.h"

#include "GDAMinimumRule.h"

#include "CamelliaCellTools.h"

// EpetraExt includes
#include "EpetraExt_MatrixMatrix.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "Epetra_SerialComm.h"

#include "Ifpack_BlockRelaxation.h"
#include "Ifpack_SparseContainer.h"
#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_PointRelaxation.h"
#include "Ifpack_Amesos.h"
#include "Ifpack_ILU.h"
#include "Ifpack_IC.h"
#include "Ifpack_Graph.h"
#include "Ifpack_Graph_Epetra_CrsGraph.h"
#include "Ifpack_Graph_Epetra_RowMatrix.h"

#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_BlockRelaxation.h"
#include "Ifpack_Graph_Epetra_RowMatrix.h"
#include "Ifpack_DenseContainer.h"

#include "CondensedDofInterpreter.h"

#include "Epetra_Operator_to_Epetra_Matrix.h"

#include "AdditiveSchwarz.h"

using namespace Intrepid;
using namespace Camellia;

#ifdef USE_HPCTW
extern "C" void HPM_Start(char *);
extern "C" void HPM_Stop(char *);
#endif

GMGOperator::GMGOperator(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP,
                         MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap,
                         Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation, bool fineSolverUsesDiagonalScaling) :  _finePartitionMap(finePartitionMap), _br(true) {
  _useStaticCondensation = useStaticCondensation;
  _fineDofInterpreter = fineDofInterpreter;
  _fineSolverUsesDiagonalScaling = false;

  _debugMode = false;

  _schwarzBlockFactorizationType = Direct;

  _applySmoothingOperator = true;

  _levelOfFill = 2;
  _fillRatio = 5.0;

  clearTimings();

#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  Epetra_Time constructionTimer(Comm);

  RHSPtr zeroRHS = RHS::rhs();
  _fineMesh = fineMesh;
  _coarseMesh = coarseMesh;
  _bc = zeroBCs;
  _coarseSolution = Teuchos::rcp( new TSolution<double>(coarseMesh, zeroBCs, zeroRHS, coarseIP) );
  _coarseSolution->setUseCondensedSolve(useStaticCondensation);

  _coarseSolver = coarseSolver;
  _haveSolvedOnCoarseMesh = false;

  setSmootherType(IFPACK_ADDITIVE_SCHWARZ); // default
  _smootherOverlap = 0;

  { // DEBUGGING
//    GlobalIndexType fineGlobalDofCount = _fineMesh->numGlobalDofs();
//    GlobalIndexType coarseGlobalDofCount = _coarseMesh->numGlobalDofs();
//    cout << "Global dof count for fine mesh: " << fineGlobalDofCount << endl;
//    cout << "Global dof count for coarse mesh: " << coarseGlobalDofCount << endl;
  }

  if (( coarseMesh->meshUsesMaximumRule()) || (! fineMesh->meshUsesMinimumRule()) ) {
    cout << "GMGOperator only supports minimum rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "GMGOperator only supports minimum rule.");
  }

//  cout << "Note: for debugging, GMGOperator writes coarse solution matrix to /tmp/A_coarse.dat.\n";
//  _coarseSolution->setWriteMatrixToFile(true, "/tmp/A_coarse.dat");

//  GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule *>(coarseMesh->globalDofAssignment().get());

//  int rank = Teuchos::GlobalMPISession::getRank();
//  if (rank==1)
//    minRule->printGlobalDofInfo();

  _coarseSolution->initializeLHSVector();
  _coarseSolution->initializeStiffnessAndLoad(); // actually don't need to initial stiffness anymore; we'll do this in computeCoarseStiffnessMatrix

  // now that:
  //   a) CondensedDofInterpreter can supply local stiffness matrices as needed, and
  //   b) we don't actually use a Solution-computed global stiffness matrix for coarse solves
  // I'm pretty sure there's no reason to call populateStiffnessAndLoad(), and it is extra work (we avoid at least global assembly)

//  if (_useStaticCondensation) {
//    // then, since the coarse solution does a condensed solve, we need to supply CondensedDofInterpreter with the
//    // local stiffness matrices on each coarse cell -- the easiest way to do this is just to invoke populateStiffnessAndLoad
//    // (this does a little extra work, but probably this is negligible)
//    _coarseSolution->populateStiffnessAndLoad();
//  }

  Epetra_Time prolongationTimer(Comm);
#ifdef HPCTW
  HPM_Start("constructProlongationOperator");
#endif
  constructProlongationOperator();
#ifdef HPCTW
  HPM_Stop("constructProlongationOperator");
#endif
  _timeProlongationOperatorConstruction = prolongationTimer.ElapsedTime();

  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) {
    cout << "Prolongation operator constructed in " << _timeProlongationOperatorConstruction << " seconds.\n";
  }

  _fineSolverUsesDiagonalScaling = false;
  setFineSolverUsesDiagonalScaling(fineSolverUsesDiagonalScaling);

  _timeConstruction += constructionTimer.ElapsedTime();
}

void GMGOperator::clearTimings() {
  _timeMapFineToCoarse = 0, _timeMapCoarseToFine = 0, _timeConstruction = 0, _timeCoarseSolve = 0, _timeCoarseImport = 0, _timeLocalCoefficientMapConstruction = 0, _timeComputeCoarseStiffnessMatrix = 0, _timeProlongationOperatorConstruction = 0;
}

void GMGOperator::computeCoarseStiffnessMatrix(Epetra_CrsMatrix *fineStiffnessMatrix) {
  int globalColCount = fineStiffnessMatrix->NumGlobalCols();
  if (_P.get() == NULL) {
    constructProlongationOperator();
  } else if (_P->NumGlobalRows() != globalColCount) {
    constructProlongationOperator();
    if (_P->NumGlobalRows() != globalColCount) {
      cout << "GMGOperator::computeCoarseStiffnessMatrix: Even after a fresh call to constructProlongationOperator, _P->NumGlobalRows() != globalColCount.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "_P->NumGlobalRows() != globalColCount");
    }
  }

#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  Epetra_Time coarseStiffnessTimer(Comm);

//  cout << "Writing fine stiffness to disk before setting up smoother.\n";
//  EpetraExt::RowMatrixToMatrixMarketFile("/tmp/A.dat",*fineStiffnessMatrix, NULL, NULL, false); // false: don't write header

  setUpSmoother(fineStiffnessMatrix);

//  EpetraExt::RowMatrixToMatrixMarketFile("/tmp/A.dat",*fineStiffnessMatrix, NULL, NULL, false); // false: don't write header

//  EpetraExt::RowMatrixToMatrixMarketFile("/tmp/P.dat",*_P, NULL, NULL, false); // false: don't write header

  int maxRowSize = _P->MaxNumEntries();

  // compute A * P
  Epetra_CrsMatrix AP(::Copy, _finePartitionMap, maxRowSize);

  int err = EpetraExt::MatrixMatrix::Multiply(*fineStiffnessMatrix, false, *_P, false, AP);
  if (err != 0) {
    cout << "ERROR: EpetraExt::MatrixMatrix::Multiply returned an error during computeCoarseStiffnessMatrix's computation of A * P.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "EpetraExt::MatrixMatrix::Multiply returned an error during computeCoarseStiffnessMatrix's computation of A * P.");
  }
//  EpetraExt::RowMatrixToMatrixMarketFile("/tmp/AP.dat", AP, NULL, NULL, false); // false: don't write header

  Epetra_Map domain_P = _P->DomainMap();
  Teuchos::RCP<Epetra_CrsMatrix> PT_A_P = Teuchos::rcp( new Epetra_CrsMatrix(::Copy, domain_P, maxRowSize) );

  // compute P^T * A * P
  err = EpetraExt::MatrixMatrix::Multiply(*_P, true, AP, false, *PT_A_P);
  if (err != 0) {
    cout << "WARNING: EpetraExt::MatrixMatrix::Multiply returned an error during computeCoarseStiffnessMatrix's computation of P^T * (A * P).\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "EpetraExt::MatrixMatrix::Multiply returned an error during computeCoarseStiffnessMatrix's computation of P^T * (A * P).");
  }

  PT_A_P->FillComplete();

//  string PT_A_P_path = "/tmp/PT_AP.dat";
//  cout << "Writing P^T * (A * P) to disk at " << PT_A_P_path << endl;
//  EpetraExt::RowMatrixToMatrixMarketFile(PT_A_P_path.c_str(), *PT_A_P, NULL, NULL, false); // false: don't write header
//
//  string P_path = "/tmp/P.dat";
//  cout << "Writing P to disk at " << P_path << endl;
//  EpetraExt::RowMatrixToMatrixMarketFile(P_path.c_str(),*_P, NULL, NULL, false); // false: don't write header

  Epetra_Map coarsePartitionMap = _coarseSolution->getPartitionMap();
  Epetra_Import  coarseImporter(coarsePartitionMap, PT_A_P->RowMap());
  Teuchos::RCP<Epetra_CrsMatrix> coarseStiffness;
  if (_coarseSolution->getZMCsAsGlobalLagrange() &&
      (_P->NumGlobalCols() > _coarseSolution->getDofInterpreter()->globalDofCount())) {
    // essentially: there are some lagrange constraints applied in coarse solve --> we shouldn't use the fused import,
    //              since this will call FillComplete()

    int numEntriesPerRow = 0; // sub-optimal, but easy
    coarseStiffness = Teuchos::rcp( new Epetra_CrsMatrix(::Copy, coarsePartitionMap, numEntriesPerRow) );
    coarseStiffness->Import(*PT_A_P,coarseImporter,::Insert);
    _coarseSolution->setStiffnessMatrix(coarseStiffness);
    _coarseSolution->imposeZMCsUsingLagrange(); // fills in the augmented matrix -- the ZMC rows that are at the end.
    // now can call FillComplete()
    coarseStiffness->FillComplete();
  } else {
    coarseStiffness = Teuchos::rcp( new Epetra_CrsMatrix(*PT_A_P, coarseImporter) );
    _coarseSolution->setStiffnessMatrix(coarseStiffness);
    _coarseSolution->imposeZMCsUsingLagrange(); // fills in the augmented matrix -- the ZMC rows that are at the end.
  }

  _timeComputeCoarseStiffnessMatrix = coarseStiffnessTimer.ElapsedTime();

  _haveSolvedOnCoarseMesh = false; // having recomputed coarseStiffness, any existing factorization is invalid
}

void GMGOperator::constructLocalCoefficientMaps() {
  Epetra_Time timer(Comm());

  set<GlobalIndexType> cellsInPartition = _fineMesh->globalDofAssignment()->cellsInPartition(-1); // rank-local

  for (set<GlobalIndexType>::iterator cellIDIt=cellsInPartition.begin(); cellIDIt != cellsInPartition.end(); cellIDIt++) {
    GlobalIndexType fineCellID = *cellIDIt;
    LocalDofMapperPtr fineMapper = getLocalCoefficientMap(fineCellID);
  }
  _timeLocalCoefficientMapConstruction += timer.ElapsedTime();
}

Teuchos::RCP<Epetra_CrsMatrix> GMGOperator::constructProlongationOperator() {
  // row indices belong to the fine grid, columns to the coarse
  // maps coefficients from coarse to fine
//  _globalStiffMatrix = Teuchos::rcp(new Epetra_FECrsMatrix(::Copy, partMap, maxRowSize));

  int maxRowSizeToPrescribe = _coarseMesh->rowSizeUpperBound();

  Teuchos::RCP<Epetra_FECrsMatrix> P = Teuchos::rcp( new Epetra_FECrsMatrix(::Copy, _finePartitionMap, maxRowSizeToPrescribe) );

  // by convention, all constraints come at the end.  (Element and global lagrange constraints, zero-mean constraints.)
  // There should be the same number in the coarse and the fine Solution objects, and the mapping is taken to be the identity.

  GlobalIndexType firstFineConstraintRowIndex = _fineDofInterpreter->globalDofCount();
  GlobalIndexType firstCoarseConstraintRowIndex = _coarseSolution->getDofInterpreter()->globalDofCount();

  // strategy: we iterate on the rank-local global dof indices on the fine mesh.
  //           we construct X, a canonical basis vector for the global dof index.
  //           we then determine the coarse mesh coefficients for X.
  //           This is a row of the matrix P.
  set<GlobalIndexType> cellsInPartition = _fineMesh->globalDofAssignment()->cellsInPartition(-1); // rank-local

  {
    Epetra_SerialComm SerialComm; // rank-local map

    GlobalIndexTypeToCast* myGlobalIndicesPtr;
    _finePartitionMap.MyGlobalElementsPtr(*&myGlobalIndicesPtr);

    map<GlobalIndexTypeToCast, set<GlobalIndexType> > cellsForGlobalDofOrdinal;
    vector<GlobalIndexTypeToCast> myGlobalIndices(_finePartitionMap.NumMyElements());
    {
      set<GlobalIndexTypeToCast> globalIndicesForRank;
      for  (int i=0;i <_finePartitionMap.NumMyElements(); i++) {
        globalIndicesForRank.insert(myGlobalIndicesPtr[i]);
        myGlobalIndices[i] = myGlobalIndicesPtr[i];
      }

      // for dof interpreter's sake, want to put 0's in slots for any seen-but-not-owned global coefficients
      set<GlobalIndexType> myCellIDs = _fineMesh->globalDofAssignment()->cellsInPartition(-1);
      for (set<GlobalIndexType>::iterator cellIDIt = myCellIDs.begin(); cellIDIt != myCellIDs.end(); cellIDIt++) {
        GlobalIndexType cellID = *cellIDIt;
        set<GlobalIndexType> globalDofsForCell = _fineDofInterpreter->globalDofIndicesForCell(cellID);
        for (set<GlobalIndexType>::iterator globalDofIt = globalDofsForCell.begin(); globalDofIt != globalDofsForCell.end(); globalDofIt++) {
          cellsForGlobalDofOrdinal[*globalDofIt].insert(cellID);
          if (globalIndicesForRank.find(*globalDofIt) == globalIndicesForRank.end()) {
            myGlobalIndices.push_back(*globalDofIt);
//            offRankGlobalIndicesForMyCells.insert(*globalDofIt);
          }
        }
      }
    }

    Epetra_Map    localXMap(myGlobalIndices.size(), myGlobalIndices.size(), &myGlobalIndices[0], 0, SerialComm);
    Teuchos::RCP<Epetra_Vector> XLocal = Teuchos::rcp( new Epetra_Vector(localXMap) );

    for (int localID=0; localID < _finePartitionMap.NumMyElements(); localID++) {
      GlobalIndexTypeToCast globalRow = _finePartitionMap.GID(localID);

      map<GlobalIndexTypeToCast, double> coarseXVectorLocal; // rank-local representation, so we just use an STL map.  Has the advantage of growing as we need it to.
      (*XLocal)[localID] = 1.0;

      if (globalRow >= firstFineConstraintRowIndex) {
        // belongs to a lagrange degree of freedom (zero-mean constraints are a special case), so we do a one-to-one map
        // NOTE: this isn't going to work properly for element Lagrange constraints in the context of h-multigrid, since
        //       for these we have one row per element, and the number of elements is reduced for the coarse mesh;
        //       it's not entirely clear to me what we should do in the general case, though for zero-mean constraints
        //       (even element-wise ones, like local conservation), probably simply weighting by relative element
        //       volume will be fine.  But we need a bit more information here; we don't have access to the fine Solution's
        //       Lagrange constraints on the present interface.
        int offset = globalRow - firstFineConstraintRowIndex;
        GlobalIndexType coarseGlobalRow = firstCoarseConstraintRowIndex + offset;
        coarseXVectorLocal[coarseGlobalRow] = 1.0;
      } else {

        set<GlobalIndexType> cells = cellsForGlobalDofOrdinal[globalRow];
        for (set<GlobalIndexType>::iterator cellIDIt=cells.begin(); cellIDIt != cells.end(); cellIDIt++) {
          GlobalIndexType fineCellID = *cellIDIt;
          int fineDofCount = _fineMesh->getElementType(fineCellID)->trialOrderPtr->totalDofs();
          FieldContainer<double> fineCellData(fineDofCount);
          _fineDofInterpreter->interpretGlobalCoefficients(fineCellID, fineCellData, *XLocal);
  //        if (globalRow==1) {
  //          cout << "fineCellData:\n" << fineCellData;
  //        }
          LocalDofMapperPtr fineMapper = getLocalCoefficientMap(fineCellID);
          GlobalIndexType coarseCellID = getCoarseCellID(fineCellID);
          int coarseDofCount = _coarseMesh->getElementType(coarseCellID)->trialOrderPtr->totalDofs();
          FieldContainer<double> coarseCellData(coarseDofCount);
          FieldContainer<double> mappedCoarseCellData(fineMapper->globalIndices().size());
          fineMapper->mapLocalDataVolume(fineCellData, mappedCoarseCellData, false);

          CellPtr fineCell = _fineMesh->getTopology()->getCell(fineCellID);
          int sideCount = fineCell->getSideCount();
          for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
            if (fineCell->ownsSide(sideOrdinal)) {
  //        cout << "fine cell " << fineCellID << " owns side " << sideOrdinal << endl;
              fineMapper->mapLocalDataSide(fineCellData, mappedCoarseCellData, false, sideOrdinal);
            }
          }

  //        if (globalRow==1) {
  //          cout << "mappedCoarseCellData:\n" << mappedCoarseCellData;
  //        }
          vector<GlobalIndexType> mappedCoarseDofIndices = fineMapper->globalIndices(); // "global" here means the coarse local
          for (int mappedCoarseDofOrdinal = 0; mappedCoarseDofOrdinal < mappedCoarseDofIndices.size(); mappedCoarseDofOrdinal++) {
            GlobalIndexType coarseDofIndex = mappedCoarseDofIndices[mappedCoarseDofOrdinal];
            coarseCellData[coarseDofIndex] = mappedCoarseCellData[mappedCoarseDofOrdinal];
          }
          FieldContainer<double> interpretedCoarseData;
          FieldContainer<GlobalIndexType> interpretedGlobalDofIndices;
          _coarseSolution->getDofInterpreter()->interpretLocalData(coarseCellID, coarseCellData, interpretedCoarseData, interpretedGlobalDofIndices);

          for (int interpretedCoarseGlobalDofOrdinal=0; interpretedCoarseGlobalDofOrdinal < interpretedGlobalDofIndices.size(); interpretedCoarseGlobalDofOrdinal++) {
            GlobalIndexType globalDofIndex = interpretedGlobalDofIndices[interpretedCoarseGlobalDofOrdinal];
            coarseXVectorLocal[globalDofIndex] += interpretedCoarseData[interpretedCoarseGlobalDofOrdinal];
          }
        }
      }

      FieldContainer<GlobalIndexTypeToCast> coarseGlobalIndices(coarseXVectorLocal.size());
      FieldContainer<double> coarseGlobalValues(coarseXVectorLocal.size());
      int nnz = 0; // nonzero entries
//      cout << "P global row " << globalRow << ": ";
      for (map<GlobalIndexTypeToCast, double>::iterator coarseXIt=coarseXVectorLocal.begin(); coarseXIt != coarseXVectorLocal.end(); coarseXIt++) {
        if (coarseXIt->second != 0.0) {
          coarseGlobalIndices[nnz] = coarseXIt->first;
          coarseGlobalValues[nnz] = coarseXIt->second;
//          cout << coarseGlobalIndices[nnz] << " --> " << coarseGlobalValues[nnz] << "; ";
          nnz++;
        }
      }
//      cout << endl;
      if (nnz > 0) {
        P->InsertGlobalValues(globalRow, nnz, &coarseGlobalValues[0], &coarseGlobalIndices[0]);
//        cout << "Inserting values for row " << globalRow << endl;
      }
      (*XLocal)[localID] = 0.0;
    }
  }

//  cout << "before FillComplete(), _P has " << _P->NumGlobalRows64() << " rows and " << _P->NumGlobalCols64() << " columns.\n";

  Epetra_Map coarseMap = _coarseSolution->getPartitionMap();
//  int rank = Teuchos::GlobalMPISession::getRank();
//  cout << "On rank " << rank << ", coarseMap has " << coarseMap.NumGlobalElements() << " global elements.\n";

  P->GlobalAssemble(coarseMap, _finePartitionMap);

  _P = P;

//  EpetraExt::RowMatrixToMatrixMarketFile("/tmp/P.dat",*_P, NULL, NULL, false); // false: don't write header

//  cout << "after FillComplete(),  _P has " << _P->NumGlobalRows64() << " rows and " << _P->NumGlobalCols64() << " columns.\n";

  return _P;
}

GlobalIndexType GMGOperator::getCoarseCellID(GlobalIndexType fineCellID) const {
  const set<IndexType>* coarseCellIDs = &_coarseMesh->getTopology()->getActiveCellIndices();
  CellPtr fineCell = _fineMesh->getTopology()->getCell(fineCellID);
  CellPtr ancestor = fineCell;
  RefinementBranch refBranch;
  while (coarseCellIDs->find(ancestor->cellIndex()) == coarseCellIDs->end()) {
    CellPtr parent = ancestor->getParent();
    if (parent.get() == NULL) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ancestor for fine cell not found in coarse mesh");
    }
    unsigned childOrdinal = parent->childOrdinal(ancestor->cellIndex());
    refBranch.insert(refBranch.begin(), make_pair(parent->refinementPattern().get(), childOrdinal));
    ancestor = parent;
  }
  return ancestor->cellIndex();
}

TSolutionPtr<double> GMGOperator::getCoarseSolution() {
  return _coarseSolution;
}

SolverPtr GMGOperator::getCoarseSolver() {
  return _coarseSolver;
}

LocalDofMapperPtr GMGOperator::getLocalCoefficientMap(GlobalIndexType fineCellID) const {
  const set<IndexType>* coarseCellIDs = &_coarseMesh->getTopology()->getActiveCellIndices();
  CellPtr fineCell = _fineMesh->getTopology()->getCell(fineCellID);
  CellPtr ancestor = fineCell;
  RefinementBranch refBranch;
  while (coarseCellIDs->find(ancestor->cellIndex()) == coarseCellIDs->end()) {
    CellPtr parent = ancestor->getParent();
    if (parent.get() == NULL) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ancestor for fine cell not found in coarse mesh");
    }
    unsigned childOrdinal = parent->childOrdinal(ancestor->cellIndex());
    refBranch.insert(refBranch.begin(), make_pair(parent->refinementPattern().get(), childOrdinal));
    ancestor = parent;
  }
  GlobalIndexType coarseCellID = ancestor->cellIndex();
  CellPtr coarseCell = ancestor;
  if (_fineMesh->globalDofAssignment()->getH1Order(fineCellID).size() > 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor-product H1Orders not yet supported in GMGOperator");
  }
  int fineOrder = _fineMesh->globalDofAssignment()->getH1Order(fineCellID)[0];
  int coarseOrder = _coarseMesh->globalDofAssignment()->getH1Order(coarseCellID)[0];

  DofOrderingPtr coarseTrialOrdering = _coarseMesh->getElementType(coarseCellID)->trialOrderPtr;
  DofOrderingPtr fineTrialOrdering = _fineMesh->getElementType(fineCellID)->trialOrderPtr;

  pair< pair<int,int>, RefinementBranch > key = make_pair(make_pair(fineOrder, coarseOrder), refBranch);

  CondensedDofInterpreter<double>* condensedDofInterpreter = NULL;

  if (_useStaticCondensation) {
    condensedDofInterpreter = dynamic_cast<CondensedDofInterpreter<double>*>(_fineDofInterpreter.get());
  }

  int fineSideCount = fineCell->getSideCount();
  int spaceDim = _fineMesh->getTopology()->getSpaceDim();
  int sideDim = spaceDim - 1;
  vector<unsigned> ancestralSideOrdinals(fineSideCount);
  vector< RefinementBranch > sideRefBranches(fineSideCount);
  for (int sideOrdinal=0; sideOrdinal<fineSideCount; sideOrdinal++) {
    ancestralSideOrdinals[sideOrdinal] = RefinementPattern::ancestralSubcellOrdinal(refBranch, sideDim, sideOrdinal);
    if (ancestralSideOrdinals[sideOrdinal] != -1) {
      sideRefBranches[sideOrdinal] = RefinementPattern::sideRefinementBranch(refBranch, sideOrdinal);
    }
  }

  if (_localCoefficientMap.find(key) == _localCoefficientMap.end()) {
    VarFactoryPtr vf = _fineMesh->bilinearForm()->varFactory();

    typedef vector< SubBasisDofMapperPtr > BasisMap; // taken together, these maps map a whole basis
    map< int, BasisMap > volumeMaps;

    vector< map< int, BasisMap > > sideMaps(fineSideCount);

    set<int> trialIDs = coarseTrialOrdering->getVarIDs();

    // for the moment, we skip the mapping from traces to fields based on traceTerm
    unsigned vertexNodePermutation = 0; // because we're "reconciling" to an ancestor, the views of the cells and sides are necessarily the same
    for (set<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++) {
      int trialID = *trialIDIt;

      VarPtr trialVar = vf->trialVars().find(trialID)->second;
      Space varSpace = trialVar->space();
      Camellia::EFunctionSpace varFS = efsForSpace(varSpace);
      if (! Camellia::functionSpaceIsDiscontinuous(varFS)) {
        int rank = Teuchos::GlobalMPISession::getRank();
        if (rank == 0)
          cout << "WARNING: function space for var " << trialVar->name() << " is not discontinuous, and GMGOperator does not yet support continuous variables, even continuous trace variables (i.e. all trace variables must be in L^2 or some other discontinuous space, like HGRAD_DISC).\n";
      }

      if (coarseTrialOrdering->getSidesForVarID(trialID).size() == 1) { // field variable
        if (condensedDofInterpreter != NULL) {
          if (condensedDofInterpreter->varDofsAreCondensible(trialID, 0, fineTrialOrdering)) continue;
        }
//        cout << "Warning: for debugging purposes, skipping projection of fields in GMGOperator.\n";
        BasisPtr coarseBasis = coarseTrialOrdering->getBasis(trialID);
        BasisPtr fineBasis = fineTrialOrdering->getBasis(trialID);
        SubBasisReconciliationWeights weights = _br.constrainedWeights(fineBasis, refBranch, coarseBasis, vertexNodePermutation);
        set<unsigned> fineDofOrdinals(weights.fineOrdinals.begin(),weights.fineOrdinals.end());

        vector<GlobalIndexType> coarseDofIndices;
        for (set<int>::iterator coarseOrdinalIt=weights.coarseOrdinals.begin(); coarseOrdinalIt != weights.coarseOrdinals.end(); coarseOrdinalIt++) {
          unsigned coarseDofIndex = coarseTrialOrdering->getDofIndex(trialID, *coarseOrdinalIt);
          coarseDofIndices.push_back(coarseDofIndex);
        }
        BasisMap basisMap(1,SubBasisDofMapper::subBasisDofMapper(fineDofOrdinals, coarseDofIndices, weights.weights));
        volumeMaps[trialID] = basisMap;
      } else { // flux/trace
//        cout << "Warning: for debugging purposes, skipping projection of fluxes and traces in GMGOperator.\n";
        for (int sideOrdinal=0; sideOrdinal<fineSideCount; sideOrdinal++) {
          if (condensedDofInterpreter != NULL) {
            if (condensedDofInterpreter->varDofsAreCondensible(trialID, sideOrdinal, fineTrialOrdering)) continue;
          }
          unsigned coarseSideOrdinal = ancestralSideOrdinals[sideOrdinal];
          BasisPtr coarseBasis, fineBasis;
          BasisMap basisMap;
          if (coarseSideOrdinal == -1) { // the fine side falls inside a coarse volume
            // we map trace to field using the traceTerm LinearTermPtr
            VarPtr trialVar = vf->trial(trialID);

            LinearTermPtr termTraced = trialVar->termTraced();
            if (termTraced.get() == NULL) // nothing we can do if we don't know what term we're tracing
              continue;

            if (! fineTrialOrdering->hasBasisEntry(trialID, sideOrdinal) ) continue;
            fineBasis = fineTrialOrdering->getBasis(trialID, sideOrdinal);

            set<int> varsTraced = termTraced->varIDs();
            for (set<int>::iterator varTracedIt = varsTraced.begin(); varTracedIt != varsTraced.end(); varTracedIt++) {
              int varTracedID = *varTracedIt;
              coarseBasis = coarseTrialOrdering->getBasis(varTracedID);

              unsigned coarseSubcellOrdinal = 0, coarseDomainOrdinal = 0; // the volume
              unsigned coarseSubcellPermutation = 0;
              unsigned fineSubcellOrdinalInFineDomain = 0; // the side is the whole fine domain...
              SubBasisReconciliationWeights weights = _br.computeConstrainedWeightsForTermTraced(termTraced, varTracedID,
                                                                                                 sideDim, fineBasis, fineSubcellOrdinalInFineDomain, refBranch, sideOrdinal,
                                                                                                 spaceDim, coarseBasis, coarseSubcellOrdinal, coarseDomainOrdinal, coarseSubcellPermutation);
              set<unsigned> fineDofOrdinals(weights.fineOrdinals.begin(),weights.fineOrdinals.end());
              vector<GlobalIndexType> coarseDofIndices;
              for (set<int>::iterator coarseOrdinalIt=weights.coarseOrdinals.begin(); coarseOrdinalIt != weights.coarseOrdinals.end(); coarseOrdinalIt++) {
                unsigned coarseDofIndex = coarseTrialOrdering->getDofIndex(varTracedID, *coarseOrdinalIt);
                coarseDofIndices.push_back(coarseDofIndex);
              }

              basisMap.push_back(SubBasisDofMapper::subBasisDofMapper(fineDofOrdinals, coarseDofIndices, weights.weights));
            }
          } else { // fine side maps to a coarse side
            if (! coarseTrialOrdering->hasBasisEntry(trialID, coarseSideOrdinal)) {
              cout << "ERROR: no entry for trial var " << trialVar->name() << " on side " << coarseSideOrdinal << " in coarse mesh.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Missing basis entry on coarse side!");
            }
            if (! fineTrialOrdering->hasBasisEntry(trialID, sideOrdinal)) {
              cout << "ERROR: no entry for trial var " << trialVar->name() << " on side " << sideOrdinal << " in fine mesh.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Missing basis entry on fine side!");
            }
            coarseBasis = coarseTrialOrdering->getBasis(trialID, coarseSideOrdinal);
            fineBasis = fineTrialOrdering->getBasis(trialID, sideOrdinal);
            SubBasisReconciliationWeights weights = _br.constrainedWeights(fineBasis, sideRefBranches[sideOrdinal], coarseBasis, vertexNodePermutation);

            set<unsigned> fineDofOrdinals(weights.fineOrdinals.begin(),weights.fineOrdinals.end());
            vector<GlobalIndexType> coarseDofIndices;
            for (set<int>::iterator coarseOrdinalIt=weights.coarseOrdinals.begin(); coarseOrdinalIt != weights.coarseOrdinals.end(); coarseOrdinalIt++) {
              unsigned coarseDofIndex = coarseTrialOrdering->getDofIndex(trialID, *coarseOrdinalIt, coarseSideOrdinal);
              coarseDofIndices.push_back(coarseDofIndex);
            }

            basisMap.push_back(SubBasisDofMapper::subBasisDofMapper(fineDofOrdinals, coarseDofIndices, weights.weights));
          }

          sideMaps[sideOrdinal][trialID] = basisMap;
        }
      }
    }

    int coarseDofCount = coarseTrialOrdering->totalDofs();
    set<GlobalIndexType> allCoarseDofs; // include these even when not mapped (which can happen with static condensation) to guarantee that the dof mapper interprets coarse cell data correctly...
    for (int coarseOrdinal=0; coarseOrdinal<coarseDofCount; coarseOrdinal++) {
      allCoarseDofs.insert(coarseOrdinal);
    }

    // I don't think we need to do any fitting, so we leave the "fittable" containers for LocalDofMapper empty

    set<GlobalIndexType> fittableGlobalDofOrdinalsInVolume;
    vector< set<GlobalIndexType> > fittableGlobalDofOrdinalsOnSides(fineSideCount);

    LocalDofMapperPtr dofMapper = Teuchos::rcp( new LocalDofMapper(fineTrialOrdering, volumeMaps, fittableGlobalDofOrdinalsInVolume,
                                                                   sideMaps, fittableGlobalDofOrdinalsOnSides, allCoarseDofs) );
    _localCoefficientMap[key] = dofMapper;
  }

  LocalDofMapperPtr dofMapper = _localCoefficientMap[key];

  // now, correct side parities in dofMapper if the ref space situation differs from the physical space one.
  FieldContainer<double> coarseCellSideParities = _coarseMesh->globalDofAssignment()->cellSideParitiesForCell(coarseCellID);
  FieldContainer<double> fineCellSideParities = _fineMesh->globalDofAssignment()->cellSideParitiesForCell(fineCellID);
  set<unsigned> fineSidesToCorrect;
  for (unsigned fineSideOrdinal=0; fineSideOrdinal<fineSideCount; fineSideOrdinal++) {
    unsigned coarseSideOrdinal = ancestralSideOrdinals[fineSideOrdinal];
    if (coarseSideOrdinal != -1) { // ancestor shares side
      double coarseParity = coarseCellSideParities(0,coarseSideOrdinal);
      double fineParity = fineCellSideParities(0,fineSideOrdinal);
      if (coarseParity != fineParity) {
        fineSidesToCorrect.insert(fineSideOrdinal);
      }
    } else {
      // when we have done a map from field to trace, no parities have been taken into account; the termTraced will involve
      // the outward facing normal, but not the potential negation of this to make neighbors agree.  Therefore, we need to "correct"
      // any interior fine side whose parity is -1.

      double fineParity = fineCellSideParities(0,fineSideOrdinal);
      if (fineParity < 0)
        fineSidesToCorrect.insert(fineSideOrdinal);
    }
  }

  if (fineSidesToCorrect.size() > 0) {
    // copy before changing dofMapper:
    dofMapper = Teuchos::rcp( new LocalDofMapper(*dofMapper.get()) );
    set<int> fluxIDs;
    VarFactoryPtr vf = _fineMesh->bilinearForm()->varFactory();
    vector<VarPtr> fluxVars = vf->fluxVars();
    for (vector<VarPtr>::iterator fluxIt = fluxVars.begin(); fluxIt != fluxVars.end(); fluxIt++) {
      fluxIDs.insert((*fluxIt)->ID());
    }
    dofMapper->reverseParity(fluxIDs, fineSidesToCorrect);
  }

  return dofMapper;
}

int GMGOperator::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported method.");
}

int GMGOperator::ApplyInverse(const Epetra_MultiVector& X_in, Epetra_MultiVector& Y) const {
//  cout << "GMGOperator::ApplyInverse.\n";
  int rank = Teuchos::GlobalMPISession::getRank();
  bool printVerboseOutput = (rank==0) && _debugMode;

  Epetra_Time timer(Comm());

  Epetra_MultiVector X(X_in); // looks like Y may be stored in the same location as X_in, so that changing Y will change X, too...
  if (_fineSolverUsesDiagonalScaling) {
    if (printVerboseOutput) cout << "multiplying X by _diag_sqrt\n";
    // Here, we assume symmetric diagonal scaling: D^-1/2 A D^-1/2, where A is the fine matrix.
    // (because the inverse that we otherwise approximate is A^-1, we now approximate D^1/2 A^-1 D^1/2)
    X.Multiply(1.0, *_diag_sqrt, X, 0);
    if (printVerboseOutput) cout << "finished multiplying X by _diag_sqrt\n";
  }

  if (printVerboseOutput) cout << "calling _coarseSolution->getRHSVector()\n";
  Teuchos::RCP<Epetra_FEVector> coarseRHSVector = _coarseSolution->getRHSVector();
  if (printVerboseOutput) cout << "returned from _coarseSolution->getRHSVector()\n";

  if (coarseRHSVector->GlobalLength() != _P->NumGlobalCols()) {
    // TODO: add support for coarseRHSVector that may have lagrange/zmc constraints applied even though fine solution neglects these...
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarseRHSVector->GlobalLength() != _P->NumGlobalCols()");
  }

  timer.ResetStartTime();

  if (printVerboseOutput) cout << "calling _P->Multiply(true, X, *coarseRHSVector);\n";
  _P->Multiply(true, X, *coarseRHSVector);
  if (printVerboseOutput) cout << "finished _P->Multiply(true, X, *coarseRHSVector);\n";
  _timeMapFineToCoarse += timer.ElapsedTime();

  timer.ResetStartTime();
  if (!_haveSolvedOnCoarseMesh) {
    if (printVerboseOutput) cout << "solving on coarse mesh\n";
    _coarseSolution->setProblem(_coarseSolver);
    _coarseSolution->solveWithPrepopulatedStiffnessAndLoad(_coarseSolver, false);
    if (printVerboseOutput) cout << "finished solving on coarse mesh\n";
    _haveSolvedOnCoarseMesh = true;
  } else {
    if (printVerboseOutput) cout << "resolving on coarse mesh\n";
    _coarseSolver->setRHS(coarseRHSVector);
    _coarseSolution->solveWithPrepopulatedStiffnessAndLoad(_coarseSolver, true); // call resolve() instead of solve() -- reuse factorization
    if (printVerboseOutput) cout << "finished resolving on coarse mesh\n";
  }
  _timeCoarseSolve += timer.ElapsedTime();

  timer.ResetStartTime();
  if (printVerboseOutput) cout << "calling _coarseSolution->getLHSVector()\n";
  Teuchos::RCP<Epetra_FEVector> coarseLHSVector = _coarseSolution->getLHSVector();

  if (printVerboseOutput) cout << "finished _coarseSolution->getLHSVector()\n";
  if (printVerboseOutput) cout << "calling _P->Multiply(false, *coarseLHSVector, Y)\n";
  _P->Multiply(false, *coarseLHSVector, Y);
  if (printVerboseOutput) cout << "finished _P->Multiply(false, *coarseLHSVector, Y)\n";
  _timeMapCoarseToFine += timer.ElapsedTime();

  // if _applySmoothingOperator is set, add S^(-1)X to Y.
  if (_applySmoothingOperator && (_smootherType != NONE)) {
    if (printVerboseOutput) cout << "copying X into X2\n";
    Epetra_MultiVector X2(X); // copy, since I'm not sure ApplyInverse is generally OK with X and Y in same location (though Aztec seems to make that assumption, so it probably is OK).
    if (printVerboseOutput) cout << "finished copying X into X2\n";
    if (printVerboseOutput) cout << "calling _smoother->ApplyInverse(X2, X)\n";
    int err = _smoother->ApplyInverse(X2, X);
    if (err != 0) {
      cout << "_smoother->ApplyInverse returned non-zero error code " << err << endl;
    }
    if (printVerboseOutput) cout << "finished _smoother->ApplyInverse(X2, X)\n";
    if (printVerboseOutput) cout << "calling Y.Update(1.0, X, 1.0)\n";
    Y.Update(1.0, X, 1.0);
    if (printVerboseOutput) cout << "finished Y.Update(1.0, X, 1.0)\n";
  } else {
    //    cout << "_diag is NULL.\n";
  }

  if (_fineSolverUsesDiagonalScaling) {
    if (printVerboseOutput) cout << "calling Y.Multiply(1.0, *_diag_sqrt, Y, 0)\n";
    Y.Multiply(1.0, *_diag_sqrt, Y, 0);
    if (printVerboseOutput) cout << "finished Y.Multiply(1.0, *_diag_sqrt, Y, 0)\n";
  }

  /*
   We zero out the lagrange multiplier solutions on the fine mesh, because we're relying on the coarse solve to impose these;
   we just use an identity block in the lower right of the fine matrix for the Lagrange constraints themselves.

   The argument for this, at least for zero mean constraints:
   If the coarse solution has zero mean, then the fact that the prolongation operator is exact (i.e. coarse mesh solution is
   exactly reproduced on fine mesh) means that the fine mesh solution will have zero mean.

   This neglects the smoothing operator.  If the smoothing operator isn't guaranteed to produce an update with zero mean,
   then this may not work.
   */
  GlobalIndexType firstFineConstraintRowIndex = _fineDofInterpreter->globalDofCount();
  for (GlobalIndexTypeToCast fineRowIndex=firstFineConstraintRowIndex; fineRowIndex<Y.GlobalLength(); fineRowIndex++) {
    int LID = Y.Map().LID(fineRowIndex);
    if (LID != -1) {
      Y[0][LID] = 0.0;
    }
  }

  return 0;
}

double GMGOperator::NormInf() const {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported method.");
}

const char * GMGOperator::Label() const {
  return "Camellia Geometric Multi-Grid Preconditioner";
}

int GMGOperator::SetUseTranspose(bool UseTranspose) {
  return -1; // not supported for now.  (wouldn't be hard, but I don't see the point.)
}

bool GMGOperator::UseTranspose() const {
  return false; // not supported for now.  (wouldn't be hard, but I don't see the point.)
}

//! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
bool GMGOperator::HasNormInf() const {
  return false;
}

//! Returns a pointer to the Epetra_Comm communicator associated with this operator.
const Epetra_Comm & GMGOperator::Comm() const {
  return _finePartitionMap.Comm();
}

//! Returns the Epetra_Map object associated with the domain of this operator.
const Epetra_Map & GMGOperator::OperatorDomainMap() const {
  return _finePartitionMap;
}

//! Returns the Epetra_Map object associated with the range of this operator.
const Epetra_Map & GMGOperator::OperatorRangeMap() const {
  return _finePartitionMap;
}

set<GlobalIndexTypeToCast> GMGOperator::setCoarseRHSVector(const Epetra_MultiVector &X, Epetra_FEVector &coarseRHSVector) const {
  // the data coming in (X) is in global dofs defined on the fine mesh.  First thing we'd like to do is map it to the fine mesh's local cells
  set<GlobalIndexType> cellsInPartition = _fineMesh->globalDofAssignment()->cellsInPartition(-1); // rank-local

  coarseRHSVector.PutScalar(0); // clear
  set<GlobalIndexTypeToCast> coarseDofIndicesToImport; // keep track of the coarse dof indices that this partition's fine cells talk to
  for (set<GlobalIndexType>::iterator cellIDIt=cellsInPartition.begin(); cellIDIt != cellsInPartition.end(); cellIDIt++) {
    GlobalIndexType fineCellID = *cellIDIt;
    int fineDofCount = _fineMesh->getElementType(fineCellID)->trialOrderPtr->totalDofs();
    FieldContainer<double> fineCellData(fineDofCount);
    _fineDofInterpreter->interpretGlobalCoefficients(fineCellID, fineCellData, X);
//    cout << "fineCellData:\n" << fineCellData;
    LocalDofMapperPtr fineMapper = getLocalCoefficientMap(fineCellID);
    GlobalIndexType coarseCellID = getCoarseCellID(fineCellID);
    int coarseDofCount = _coarseMesh->getElementType(coarseCellID)->trialOrderPtr->totalDofs();
    FieldContainer<double> coarseCellData(coarseDofCount);
    FieldContainer<double> mappedCoarseCellData(fineMapper->globalIndices().size());
    fineMapper->mapLocalDataVolume(fineCellData, mappedCoarseCellData, false);

    CellPtr fineCell = _fineMesh->getTopology()->getCell(fineCellID);
    int sideCount = fineCell->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
      if (fineCell->ownsSide(sideOrdinal)) {
//        cout << "fine cell " << fineCellID << " owns side " << sideOrdinal << endl;
        fineMapper->mapLocalDataSide(fineCellData, mappedCoarseCellData, false, sideOrdinal);
      }
    }

//    cout << "mappedCoarseCellData:\n" << mappedCoarseCellData;
    vector<GlobalIndexType> mappedCoarseDofIndices = fineMapper->globalIndices(); // "global" here means the coarse local
    for (int mappedCoarseDofOrdinal = 0; mappedCoarseDofOrdinal < mappedCoarseDofIndices.size(); mappedCoarseDofOrdinal++) {
      GlobalIndexType coarseDofIndex = mappedCoarseDofIndices[mappedCoarseDofOrdinal];
      coarseCellData[coarseDofIndex] = mappedCoarseCellData[mappedCoarseDofOrdinal];
    }
    FieldContainer<double> interpretedCoarseData;
    FieldContainer<GlobalIndexType> interpretedGlobalDofIndices;
    _coarseSolution->getDofInterpreter()->interpretLocalData(coarseCellID, coarseCellData, interpretedCoarseData, interpretedGlobalDofIndices);
//    cout << "interpretedCoarseData:\n" << interpretedCoarseData;
    FieldContainer<GlobalIndexTypeToCast> interpretedGlobalDofIndicesCast(interpretedGlobalDofIndices.size());
    for (int interpretedDofOrdinal=0; interpretedDofOrdinal < interpretedGlobalDofIndices.size(); interpretedDofOrdinal++) {
      interpretedGlobalDofIndicesCast[interpretedDofOrdinal] = (GlobalIndexTypeToCast) interpretedGlobalDofIndices[interpretedDofOrdinal];
      coarseDofIndicesToImport.insert(interpretedGlobalDofIndicesCast[interpretedDofOrdinal]);
    }
    coarseRHSVector.SumIntoGlobalValues(interpretedCoarseData.size(), &interpretedGlobalDofIndicesCast[0], &interpretedCoarseData[0]);
  }
  coarseRHSVector.GlobalAssemble();
  return coarseDofIndicesToImport;
}

TimeStatistics GMGOperator::getStatistics(double timeValue) const {
  TimeStatistics stats;
  int indexBase = 0;
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_Map timeMap(numProcs,indexBase,Comm());
  Epetra_Vector timeVector(timeMap);
  timeVector[0] = timeValue;

  int err = timeVector.Norm1( &stats.sum );
  err = timeVector.MeanValue( &stats.mean );
  err = timeVector.MinValue( &stats.min );
  err = timeVector.MaxValue( &stats.max );

  return stats;
}

void GMGOperator::reportTimings() const {
  //   mutable double _timeMapFineToCoarse, _timeMapCoarseToFine, _timeCoarseImport, _timeConstruction, _timeCoarseSolve;  // totals over the life of the object
  int rank = Teuchos::GlobalMPISession::getRank();

  map<string, double> reportValues;
  reportValues["construction time"] = _timeConstruction;
  reportValues["  construct prolongation operator"] = _timeProlongationOperatorConstruction;
  reportValues["construct local coefficient maps"] = _timeLocalCoefficientMapConstruction;
  reportValues["coarse import"] = _timeCoarseImport;
  reportValues["coarse solve"] = _timeCoarseSolve;
  reportValues["map coarse to fine"] = _timeMapCoarseToFine;
  reportValues["map fine to coarse"] = _timeMapFineToCoarse;
  reportValues["compute coarse stiffness matrix"] = _timeComputeCoarseStiffnessMatrix;

  for (map<string,double>::iterator reportIt = reportValues.begin(); reportIt != reportValues.end(); reportIt++) {
    TimeStatistics stats = getStatistics(reportIt->second);
    if (rank==0) {
      cout << reportIt->first << ":\n";
      cout <<  "mean = " << stats.mean << " seconds\n";
      cout << "max =  " << stats.max << " seconds\n";
      cout << "min =  " << stats.min << " seconds\n";
      cout << "sum =  " << stats.sum << " seconds\n";
    }
  }
}

void GMGOperator::setApplySmoothingOperator(bool value) {
  _applySmoothingOperator = value;
}

void GMGOperator::setCoarseSolver(SolverPtr coarseSolver) {
  _coarseSolver = coarseSolver;
}

void GMGOperator::setDebugMode(bool value) {
  _debugMode = value;
}

void GMGOperator::setFineMesh(MeshPtr fineMesh, Epetra_Map finePartitionMap) {
  _fineMesh = fineMesh;
  _finePartitionMap = finePartitionMap;

  constructProlongationOperator(); // _P
}

void GMGOperator::setFillRatio(double fillRatio) {
  _fillRatio = fillRatio;
//  cout << "fill ratio set to " << fillRatio << endl;
}

void GMGOperator::setFineSolverUsesDiagonalScaling(bool value) {
  if (value != _fineSolverUsesDiagonalScaling) {
    _fineSolverUsesDiagonalScaling = value;
  }
}

void GMGOperator::setLevelOfFill(int fillLevel) {
  _levelOfFill = fillLevel;
//  cout << "level of fill set to " << fillLevel << endl;
}

void GMGOperator::setSchwarzFactorizationType(FactorType choice) {
  _schwarzBlockFactorizationType = choice;
}

void GMGOperator::setStiffnessDiagonal(Teuchos::RCP< Epetra_MultiVector> diagonal) {
  // this should be the true diagonal (before scaling) of the fine stiffness matrix.
    _diag = diagonal;
  if (diagonal.get() != NULL) {
    // construct inverse, too.
    const Epetra_BlockMap* map = &_diag->Map();
    _diag_inv = Teuchos::rcp( new Epetra_MultiVector(*map, 1) );
    _diag_sqrt = Teuchos::rcp( new Epetra_MultiVector(*map, 1) );
    if (map->NumMyElements() > 0) {
      for (int lid = map->MinLID(); lid <= map->MaxLID(); lid++) {
        (*_diag_inv)[0][lid] = 1.0 / (*_diag)[0][lid];
        (*_diag_sqrt)[0][lid] = sqrt((*_diag)[0][lid]);
      }
    }
  } else {
    _diag_inv = Teuchos::rcp( (Epetra_MultiVector*) NULL);
    _diag_sqrt = Teuchos::rcp( (Epetra_MultiVector*) NULL);
  }

}

void GMGOperator::setSmootherOverlap(int overlap) {
  _smootherOverlap = overlap;
}

void GMGOperator::setSmootherType(GMGOperator::SmootherChoice smootherType) {
  if (smootherType != NONE) {
    setApplySmoothingOperator(true);
  } else {
    setApplySmoothingOperator(false);
  }
  _smootherType = smootherType;
}

void GMGOperator::setUpSmoother(Epetra_CrsMatrix *fineStiffnessMatrix) {
  SmootherChoice choice = _smootherType;

  Teuchos::ParameterList List;

  Teuchos::RCP<Ifpack_Preconditioner> smoother;

  switch (choice) {
    case NONE:
      _smoother = Teuchos::null;
      return;
    case POINT_JACOBI:
    {
      List.set("relaxation: type", "Jacobi");
      smoother = Teuchos::rcp(new Ifpack_PointRelaxation(fineStiffnessMatrix) );
    }
      break;
    case POINT_SYMMETRIC_GAUSS_SEIDEL:
    {
      List.set("relaxation: type", "symmetric Gauss-Seidel");
      smoother = Teuchos::rcp(new Ifpack_PointRelaxation(fineStiffnessMatrix) );
    }
      break;
    case BLOCK_JACOBI:
    {
      // TODO: work out how we're supposed to specify partitioning scheme.
      //      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Block Gauss-Seidel smoother not yet supported.");

      smoother = Teuchos::rcp(new Ifpack_BlockRelaxation<Ifpack_SparseContainer<Ifpack_Amesos> >(fineStiffnessMatrix) );
      Teuchos::ParameterList List;
      // TODO: work out what the various parameters do, and how they should depend on the problem...
      int overlapBlocks = _smootherOverlap;
      int sweeps = 2;
      int localParts = 4;

      List.set("relaxation: type", "Jacobi");
      List.set("relaxation: sweeps", sweeps);
      List.set("amesos: solver type", "Amesos_Klu");

      List.set("partitioner: overlap", overlapBlocks);
#ifdef HAVE_IFPACK_METIS
      // use METIS to create the blocks. This requires --enable-ifpack-metis.
      // If METIS is not installed, the user may select "linear".
      List.set("partitioner: type", "metis");
#else
      // or a simple greedy algorithm is METIS is not enabled
      List.set("partitioner: type", "greedy");
#endif
      // defines here the number of local blocks. If 1,
      // and only one process is used in the computation, then
      // the preconditioner must converge in one iteration.
      List.set("partitioner: local parts", localParts);
    }
      break;
    case BLOCK_SYMMETRIC_GAUSS_SEIDEL:
    {
      // TODO: work out how we're supposed to specify partitioning scheme.
//      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Block Gauss-Seidel smoother not yet supported.");

      smoother = Teuchos::rcp(new Ifpack_BlockRelaxation<Ifpack_SparseContainer<Ifpack_Amesos> >(fineStiffnessMatrix) );
      Teuchos::ParameterList List;
      // TODO: work out what the various parameters do, and how they should depend on the problem...
      int overlapBlocks = _smootherOverlap;
      int sweeps = 2;
      int localParts = 4;

      List.set("relaxation: type", "symmetric Gauss-Seidel");
      List.set("relaxation: sweeps", sweeps);
      List.set("amesos: solver type", "Amesos_Klu");

      List.set("partitioner: overlap", overlapBlocks);
#ifdef HAVE_IFPACK_METIS
      // use METIS to create the blocks. This requires --enable-ifpack-metis.
      // If METIS is not installed, the user may select "linear".
      List.set("partitioner: type", "metis");
#else
      // or a simple greedy algorithm if METIS is not enabled
      List.set("partitioner: type", "greedy");
#endif
      // defines here the number of local blocks. If 1,
      // and only one process is used in the computation, then
      // the preconditioner must converge in one iteration.
      List.set("partitioner: local parts", localParts);
    }
      break;
    case IFPACK_ADDITIVE_SCHWARZ:
    case CAMELLIA_ADDITIVE_SCHWARZ:
    {
//      cout << "Using additive Schwarz smoother.\n";
      int OverlapLevel = _smootherOverlap;

      if (choice==IFPACK_ADDITIVE_SCHWARZ) {
        switch (_schwarzBlockFactorizationType) {
          case Direct:
            smoother = Teuchos::rcp(new Ifpack_AdditiveSchwarz<Ifpack_Amesos>(fineStiffnessMatrix, OverlapLevel) );
            break;
          case ILU:
            smoother = Teuchos::rcp(new Ifpack_AdditiveSchwarz<Ifpack_ILU>(fineStiffnessMatrix, OverlapLevel) );
            List.set("fact: level-of-fill", _levelOfFill);
            break;
          case IC:
            smoother = Teuchos::rcp(new Ifpack_AdditiveSchwarz<Ifpack_IC>(fineStiffnessMatrix, OverlapLevel) );
            List.set("fact: ict level-of-fill", _fillRatio);
            break;
          default:
            break;
        }
      } else {
        switch (_schwarzBlockFactorizationType) {
          case Direct:
            smoother = Teuchos::rcp(new Camellia::AdditiveSchwarz<Ifpack_Amesos>(fineStiffnessMatrix, OverlapLevel, _fineMesh, _fineDofInterpreter) );
            break;
          case ILU:
            smoother = Teuchos::rcp(new Camellia::AdditiveSchwarz<Ifpack_ILU>(fineStiffnessMatrix, OverlapLevel, _fineMesh, _fineDofInterpreter) );
            List.set("fact: level-of-fill", _levelOfFill);
            break;
          case IC:
            smoother = Teuchos::rcp(new Camellia::AdditiveSchwarz<Ifpack_IC>(fineStiffnessMatrix, OverlapLevel, _fineMesh, _fineDofInterpreter) );
            List.set("fact: ict level-of-fill", _fillRatio);
            break;
          default:
            break;
        }
      }

      List.set("schwarz: combine mode", "Add"); // The PDF doc says to use "Insert" to maintain symmetry, but the HTML docs (which are more recent) say to use "Add".  http://trilinos.org/docs/r11.10/packages/ifpack/doc/html/index.html
    }
      break;

    default:
      break;
  }

  int err = smoother->SetParameters(List);
  if (err != 0) {
    cout << "WARNING: In GMGOperator, smoother->SetParameters() returned with err " << err << endl;
  }

//  int rank = Teuchos::GlobalMPISession::getRank();
//  if (rank == 0) {
//    cout << "Smoother info:\n";
//    cout << *smoother;
//  }

//  if (_smootherType != IFPACK_ADDITIVE_SCHWARZ) {
    // not real sure why, but in the doc examples, this isn't called for additive schwarz
    err = smoother->Initialize();
    if (err != 0) {
      cout << "WARNING: In GMGOperator, smoother->Initialize() returned with err " << err << endl;
    }
//  }
//  cout << "Calling smoother->Compute()\n";
  err = smoother->Compute();
//  cout << "smoother->Compute() completed\n";


  if (err != 0) {
    cout << "WARNING: In GMGOperator, smoother->Compute() returned with err = " << err << endl;
  }

//  int rank = Teuchos::GlobalMPISession::getRank();
//  {
//    // TEST CODE: compute and output condest
//    double condest = smoother->Condest(Ifpack_CG);
//
//    if (rank==0) {
//      cout << "smoother condest = " << condest << endl;
//    }
//  }

//  {
//    if (rank==0) cout << "Converting IfPack smoother to an Epetra_CrsMatrix.\n";
//    // TEST CODE: construct an Epetra_Matrix and output to disk:
//    Teuchos::RCP<Epetra_CrsMatrix> matrix = Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*smoother, _finePartitionMap);
//
//    string smoother_path = "/tmp/smoother.dat";
//    cout << "Writing smoother to disk at " << smoother_path << endl;
//    EpetraExt::RowMatrixToMatrixMarketFile(smoother_path.c_str(), *matrix, NULL, NULL, false); // false: don't write header
//  }

  _smoother = smoother;

}

Teuchos::RCP<Epetra_CrsMatrix> GMGOperator::getProlongationOperator() {
  return _P;
}

Teuchos::RCP<Epetra_CrsMatrix> GMGOperator::getSmootherAsMatrix() {
  return Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*_smoother, _finePartitionMap);
}

//! Returns the coarse stiffness matrix (an Epetra_CrsMatrix).
Teuchos::RCP<Epetra_CrsMatrix> GMGOperator::getCoarseStiffnessMatrix() {
  return _coarseSolution->getStiffnessMatrix();
}
