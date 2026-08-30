//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  GMGOperator.cpp
//  Camellia
//
//  Created by Nate Roberts on 7/3/14.
//
//

#include "GMGOperator.h"

#include "BasisFactory.h"
#include "DofOrdering.h"
#include "GlobalDofAssignment.h"
#include "BasisSumFunction.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "CondensedDofInterpreter.h"
#include "CubatureFactory.h"
#include "E_to_T.hpp"
#include "GDAMinimumRule.h"
#include "OverlappingRowMatrix.h"
#include "SerialDenseWrapper.h"
#include "T_to_E.hpp"
#include "TimeLogger.h"

#include <MatrixMarket_Tpetra.hpp>
#include <TpetraExt_MatrixMatrix.hpp>

#include "Epetra_SerialComm.h"

#include <Ifpack2_AdditiveSchwarz.hpp>
#include <Ifpack2_BlockRelaxation.hpp>
#include <Ifpack2_Factory.hpp>
#include <Ifpack2_Preconditioner.hpp>

#include "Epetra_Operator_to_Epetra_Matrix.h"
#include "Tpetra_Operator_to_Tpetra_Matrix.h"
#include "TpetraCamelliaAdditiveSchwarz.h"

using namespace Camellia;

#ifdef USE_HPCTW
extern "C" void HPM_Start(char *);
extern "C" void HPM_Stop(char *);
#endif

using Scalar = double;
using SerialDenseMatrix = Teuchos::SerialDenseMatrix<int,double>;

namespace {
// TODO: move everything in this (anonymous namespace) block to some more appropriate location (AdditiveSchwarzHelpers?)

std::set<GlobalIndexType> overlappingCells(GlobalIndexType cellID, MeshPtr mesh,
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

std::set<GlobalIndexType> overlappingCells(const std::set<GlobalIndexType> &cellIDs, MeshPtr mesh,
                                           int overlapLevel, bool hierarchical,
                                           int dimensionForNeighborRelation)
{
  set<GlobalIndexType> cellsInOverlap = cellIDs;
  for (GlobalIndexType cellID : cellIDs)
  {
    std::set<GlobalIndexType> cellHalo = overlappingCells(cellID, mesh, overlapLevel,
                                                          hierarchical, dimensionForNeighborRelation);
    cellsInOverlap.insert(cellHalo.begin(), cellHalo.end());
  }
  return cellsInOverlap;
}

int MaxGlobalOverlapSideNeighborCount(MeshPtr mesh, int overlap, bool hierarchical, int dimForNeighborRelationship)
{
  const set<GlobalIndexType>* myCellIndices = &mesh->cellIDsInPartition();
  Teuchos_CommPtr Comm = mesh->Comm();
  
  int localMaxNeighbors = 0;
  MeshTopologyViewPtr meshTopo = mesh->getTopology();
  
  int overlapNeighborRelationshipDimension = meshTopo->getDimension() - 1; // side dimension
  set<GlobalIndexType> remoteCells;
  
  // first: which neighbors are remote? (we assume that overlap region itself is locally known)
  for (GlobalIndexType cellIndex : *myCellIndices)
  {
    set<GlobalIndexType> subdomainCellsAndNeighbors = OverlappingRowMatrix::overlappingCells(cellIndex, mesh, overlap,
                                                                                             hierarchical,
                                                                                             dimForNeighborRelationship);
    for (GlobalIndexType subdomainCellIndex : subdomainCellsAndNeighbors)
    {
      if (!mesh->myCellsInclude(subdomainCellIndex))
      {
        remoteCells.insert(subdomainCellIndex);
      }
    }
  }
  
  // second: issue requests to owners for their list of neighbors
  // this is a fairly quick-and-dirty implementation; could be reworked to reduce redundant data sent
  std::map<GlobalIndexType,std::vector<GlobalIndexType>> remoteCellNeighbors;
  
  int myRank = Comm->getRank();
  std::map<int,std::vector<std::pair<int,GlobalIndexType>>> requests; // owning PID -> (myPID, cell whose neighbor lists we want)
  for (GlobalIndexType remoteCellID : remoteCells)
  {
    int rank = mesh->partitionForCellID(remoteCellID);
    requests[rank].push_back({myRank,remoteCellID});
  }
  
  std::vector<std::pair<int,GlobalIndexType>> requestsReceived;
  MPIWrapper::sendDataVectors(Comm, requests, requestsReceived);
  
  std::map<int,std::vector<std::pair<GlobalIndexType,GlobalIndexType>>> responsesToSend;
  std::vector<std::pair<GlobalIndexType,GlobalIndexType>> responsesReceived;
  for (auto request : requestsReceived)
  {
    int remotePID = request.first;
    GlobalIndexType myCellID = request.second;
    TEUCHOS_TEST_FOR_EXCEPTION(!mesh->myCellsInclude(myCellID),std::invalid_argument, "request received for non-owned cellID");
    CellPtr myCell = meshTopo->getCell(myCellID);
    std::set<GlobalIndexType> neighborIDs = myCell->getActiveNeighborIndices(meshTopo);
    for (GlobalIndexType neighborID : neighborIDs)
    {
      responsesToSend[remotePID].push_back({myCellID,neighborID});
    }
  }
  MPIWrapper::sendDataVectors(Comm, responsesToSend, responsesReceived);
  for (auto response : responsesReceived)
  {
    GlobalIndexType remoteCellID = response.first;
    GlobalIndexType neighborID = response.second;
    remoteCellNeighbors[remoteCellID].push_back(neighborID);
  }
  
  // third: count
  for (GlobalIndexType cellIndex : *myCellIndices)
  {
    set<GlobalIndexType> subdomainCellsAndNeighbors = OverlappingRowMatrix::overlappingCells(cellIndex, mesh, overlap,
                                                                                             hierarchical,
                                                                                             dimForNeighborRelationship);
    set<GlobalIndexType> subdomainNeighbors;
    for (GlobalIndexType subdomainCellIndex : subdomainCellsAndNeighbors)
    {
      if (mesh->myCellsInclude(subdomainCellIndex))
      {
        CellPtr subdomainCell = meshTopo->getCell(subdomainCellIndex);
        
        set<IndexType> cellNeighbors = subdomainCell->getActiveNeighborIndices(overlapNeighborRelationshipDimension, meshTopo);
        subdomainNeighbors.insert(cellNeighbors.begin(),cellNeighbors.end());
      }
      else
      {
        auto remoteEntry = remoteCellNeighbors.find(subdomainCellIndex);
        TEUCHOS_TEST_FOR_EXCEPTION(remoteEntry == remoteCellNeighbors.end(), std::invalid_argument, "Remote cell not found during send/receive");
        subdomainNeighbors.insert(remoteEntry->second.begin(), remoteEntry->second.end());
      }
    }
    
    subdomainCellsAndNeighbors.insert(subdomainNeighbors.begin(),subdomainNeighbors.end());
    localMaxNeighbors = max(localMaxNeighbors,(int)subdomainCellsAndNeighbors.size());
  }
  
  
  int globalMaxNeighbors;
  Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &localMaxNeighbors, &globalMaxNeighbors);
  return globalMaxNeighbors;
}

// TODO: move this somewhere appropriate, maybe to AdditiveSchwarz

using LO = TMatrix<double>::local_ordinal_type;
using GO = TMatrix<double>::global_ordinal_type;
using Node = TMatrix<double>::node_type;
// Note: this Ifpack2::OverlappingRowMatrix implementation is currently coming from
// Nathan V. Roberts' Trilinos fork/branch:
//   https://github.com/CamelliaDPG/Trilinos.git
//   branch: fixes-for-camellia
// This is already an accepted dependency of the target Camellia branch.
using OverlappingRowMatrix = Ifpack2::OverlappingRowMatrix< TRowMatrix<double> >;

using AdditiveSchwarzDetails = GMGOperator::AdditiveSchwarzDetails;

AdditiveSchwarzDetails
getAdditiveSchwarzDetails(MeshPtr mesh, int overlapLevel, bool hierarchical, int dimensionForNeighborRelationship,
                          Teuchos::RCP<DofInterpreter> fineDofInterpreter, TMatrixPtr<double> A)
{
  using Node = TMatrix<double>::node_type;
  
  GO numGlobalRows = A->getGlobalNumRows();
  
  std::set<GO> gidsThisRank; // all the GIDs locally visible in the overlapping row matrix
  
  std::vector< std::set<GO> > partsGlobalOrdinals; // one part per local cell -- start with GIDs; will map to LIDs below
  
  set<GlobalIndexType> myCellIDs = mesh->cellIDsInPartition(); // we want a domain identified with each cell
  
  map<GlobalIndexType,set<GlobalIndexType>> overlapCellsForCell; // key: cell; value: the cells in its overlap region
  set<GlobalIndexType> allCellsOfInterestToMe;
  
  // find the overlap region for each cell.
  for (GlobalIndexType cellID : myCellIDs)
  {
    set<GlobalIndexType> cellsInOverlap = overlappingCells(cellID, mesh, overlapLevel, hierarchical,
                                                           dimensionForNeighborRelationship);
    overlapCellsForCell[cellID] = cellsInOverlap;
    allCellsOfInterestToMe.insert(cellsInOverlap.begin(), cellsInOverlap.end());
  }
  
  // find the global indices that belong to each cell within the overlap regions
  map<GlobalIndexType,set<GlobalIndexType>> globalIndicesForCells; // key: cell; value: globalDofIndices
  globalIndicesForCells = fineDofInterpreter->importGlobalIndicesMap(allCellsOfInterestToMe);
  
  for (GlobalIndexType cellID : myCellIDs)
  {
    set<GO> globalDofIndices; // global dof indices for the overlap region
    set<GlobalIndexType> cellsInOverlap = overlapCellsForCell[cellID];
    
    for (GlobalIndexType overlapCell : cellsInOverlap)
    {
      globalDofIndices.insert(globalIndicesForCells[overlapCell].begin(), globalIndicesForCells[overlapCell].end());
    }
    partsGlobalOrdinals.push_back(globalDofIndices);
    gidsThisRank.insert(globalDofIndices.begin(),globalDofIndices.end());
  }
  
  Teuchos::Array<GO> externalGIDsThisRank;
  {
    auto rowMap = A->getRowMap();
    for (auto gid : gidsThisRank)
    {
      if (rowMap->getLocalElement(gid) == Teuchos::OrdinalTraits<LO>::invalid())
      {
        // external (ghost) GID
        externalGIDsThisRank.push_back(gid);
      }
    }
  }
  
  // a little unfortunate that we construct this twice: once here, and once within Ifpack2::AdditiveSchwarz
  Teuchos::RCP<OverlappingRowMatrix> overlapMatrix = Teuchos::rcp(new OverlappingRowMatrix(A, externalGIDsThisRank));
  
  auto overlapMap = overlapMatrix->getRowMap();
  Teuchos::Array<GO> overlapMapGlobalOrdinals;
  {
    auto overlapMapGIDsView = overlapMap->getMyGlobalIndices();
    overlapMapGlobalOrdinals.reserve(overlapMapGIDsView.size());
    for (size_t i = 0; i < overlapMapGIDsView.size(); i++)
    {
      overlapMapGlobalOrdinals.push_back(overlapMapGIDsView[i]);
    }
  }
  Teuchos::Array<Teuchos::ArrayRCP<LO> > partsLocalOrdinals; // one per (local) cell
  Teuchos::Array<Teuchos::ArrayRCP<GO> > partsGlobalOrdinalsForIfpack2; // same blocks, expressed as GIDs for Ifpack2::UserPartitioner
  std::vector<int> localIncidenceCounts(overlapMap->getLocalNumElements(), 0);
  for (auto gidSet : partsGlobalOrdinals)
  {
    const size_t numIDs = gidSet.size();
    std::vector<LO> lidVector(numIDs);
    Teuchos::ArrayRCP<LO> array(numIDs);
    Teuchos::ArrayRCP<GO> globalArray(numIDs);
    size_t lidOrdinal = 0;
    for (const auto &gid : gidSet)
    {
      LO lid = overlapMap->getLocalElement(gid);
      TEUCHOS_TEST_FOR_EXCEPTION(lid == Teuchos::OrdinalTraits<LO>::invalid(), std::invalid_argument,
                                 "getAdditiveSchwarzDetails(): overlap-map LID not found for GID " << gid);
      lidVector[lidOrdinal] = lid;
      array[lidOrdinal] = lid;
      globalArray[lidOrdinal] = gid;
      lidOrdinal++;
      localIncidenceCounts[lid]++;
    }
    partsLocalOrdinals.push_back(array);
    partsGlobalOrdinalsForIfpack2.push_back(globalArray);
//    size_t cellOrdinal = partsLocalOrdinals.size() - 1;
//    std::cout << "cell ordinal " << cellOrdinal << " has " << array.size() << " rows in its partition.\n";
//    Camellia::print("row LIDs", lidVector);
  }
  
  AdditiveSchwarzDetails details;
  details.externalGIDsThisRank = externalGIDsThisRank;
  details.partsLocalOrdinals = partsLocalOrdinals;
  details.partsGlobalOrdinals = partsGlobalOrdinalsForIfpack2;
  details.overlapMapGlobalOrdinals = overlapMapGlobalOrdinals;
  details.localIncidenceCounts = localIncidenceCounts;
  return details;
}

} // anonymous namespace

// TODO: change this to be CAMELLIA_ADDITIVE_SCHWARZ, if and when the Ifpack2-compatible version of that is implemented
GMGOperator::AdditiveSchwarzDetails
GMGOperator::getAdditiveSchwarzDetailsForTests(MeshPtr mesh,
                                               int overlapLevel,
                                               bool hierarchical,
                                               int dimensionForNeighborRelationship,
                                               Teuchos::RCP<DofInterpreter> fineDofInterpreter,
                                               TMatrixPtr<Scalar> A)
{
  return getAdditiveSchwarzDetails(mesh, overlapLevel, hierarchical,
                                   dimensionForNeighborRelationship,
                                   fineDofInterpreter, A);
}

GMGOperator::SmootherChoice GMGOperator::DEFAULT_ADDITIVE_SCHWARZ = GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ;

GMGOperator::GMGOperator(GMGOperator::MultigridStrategy multigridStrategy, BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP,
                         MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter, TMapPtr finePartitionMap,
                         Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation, Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation) :
Narrator("GMGOperator"),
_finePartitionMap(finePartitionMap), _br(sharedBasisReconciliation)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  
  if (useStaticCondensation)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(coarseIP == Teuchos::null, std::invalid_argument, "GMGOperator: coarseIP is required when useStaticCondensation = true");
  }
  
  // new 4/20/16: require that coarseMesh and fineMesh share a common base MeshTopology (ensures that distributed geometry does not interfere with the prolongation operator determination)
  
  const MeshTopology* coarseBaseTopo = coarseMesh->getTopology()->baseMeshTopology();
  const MeshTopology* fineBaseTopo = fineMesh->getTopology()->baseMeshTopology();
  TEUCHOS_TEST_FOR_EXCEPTION(fineBaseTopo != coarseBaseTopo, std::invalid_argument, "fine and coarse mesh must have a common base MeshTopology");
  
  _useStaticCondensation = useStaticCondensation;
  _fineDofInterpreter = fineDofInterpreter;

  _hierarchicalNeighborsForSchwarz = false; // default; can be overridden later
  _dimensionForSchwarzNeighborRelationship = fineMesh->getTopology()->getDimension() - 1; // side dimension default

  _debugMode = false;

  _schwarzBlockFactorizationType = Direct;

  _smootherApplicationCount = 1;
  
  // additive implies a two-level operator.
  _smootherApplicationType = MULTIPLICATIVE; // this field is deprecated
  _multigridStrategy = multigridStrategy;

  _levelOfFill = 2;
  _smootherWeight = 1.0;

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
  
  set<GlobalIndexType> offRankCellsToInclude; // ignored (and empty) when useStaticCondensation is false
  if (useStaticCondensation)
  {
    const set<GlobalIndexType>* fineCellIDsForRank = &_fineMesh->globalDofAssignment()->cellsInPartition(rank);
    const set<GlobalIndexType>* coarseCellIDsForRank = &_coarseMesh->globalDofAssignment()->cellsInPartition(rank);
    for (GlobalIndexType fineCellID : *fineCellIDsForRank)
    {
      GlobalIndexType coarseCellID = getCoarseCellID(fineCellID);
      if (coarseCellIDsForRank->find(coarseCellID) == coarseCellIDsForRank->end())
      {
        offRankCellsToInclude.insert(coarseCellID);
      }
    }
  }
  _coarseSolution->setUseCondensedSolve(useStaticCondensation, offRankCellsToInclude);

  _coarseSolver = coarseSolver;
  _haveSolvedOnCoarseMesh = false;

  setSmootherType(DEFAULT_ADDITIVE_SCHWARZ);
  _smootherOverlap = 0;
  _useSchwarzDiagonalWeight = false;
  _useSchwarzScalingWeight = true; // weight to ensure maximum eigenvalue of M*A <= 1.0, M being the Schwarz smoother.

  _coarseSolution->initializeLHSVector();
  _coarseSolution->initializeLoad(); // don't need to initialize stiffness anymore; we'll do this in computeCoarseStiffnessMatrix

  TimeLogger::sharedInstance()->createTimeEntry("map fluxes to fields to fluxes");
  TimeLogger::sharedInstance()->createTimeEntry("compute flux to field map");
  
//  constructProlongationOperator();

  _timeConstruction += constructionTimer.ElapsedTime();
}

GMGOperator::GMGOperator(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP,
                         MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter, TMapPtr finePartitionMap,
                         Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation, Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation)
: GMGOperator(V_CYCLE, zeroBCs, coarseMesh, coarseIP, fineMesh, fineDofInterpreter, finePartitionMap, coarseSolver, useStaticCondensation, sharedBasisReconciliation) {}


GMGOperator::GMGOperator(GMGOperator::MultigridStrategy multigridStrategy, BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP,
                         MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter, TMapPtr finePartitionMap,
                         bool useStaticCondensation, Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation)
: GMGOperator(multigridStrategy, zeroBCs, coarseMesh, coarseIP, fineMesh, fineDofInterpreter, finePartitionMap, Teuchos::null, useStaticCondensation, sharedBasisReconciliation) {}

GMGOperator::GMGOperator(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh,
                         Teuchos::RCP<DofInterpreter> fineDofInterpreter, TMapPtr finePartitionMap,
                         bool useStaticCondensation, Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation)
: GMGOperator(zeroBCs, coarseMesh, coarseIP, fineMesh, fineDofInterpreter, finePartitionMap, Teuchos::null, useStaticCondensation, sharedBasisReconciliation) {}

void GMGOperator::clearTimings()
{
  _timeMapFineToCoarse = 0, _timeMapCoarseToFine = 0, _timeConstruction = 0, _timeCoarseSolve = 0, _timeCoarseImport = 0, _timeLocalCoefficientMapConstruction = 0, _timeComputeCoarseStiffnessMatrix = 0, _timeProlongationOperatorConstruction = 0,
  _timeSetUpSmoother = 0, _timeUpdateCoarseOperator = 0, _timeApplyFineStiffness = 0, _timeApplySmoother = 0;
}

void GMGOperator::setClearFinestCondensedDofInterpreterAfterProlongation(bool value)
{
  _clearFinestCondensedDofInterpreterAfterProlongation = value;
}

void GMGOperator::computeCoarseStiffnessMatrix(TMatrixPtr<Scalar> tFineStiffnessMatrix)
{
  auto fineStiffnessMatrix = T_to_E::matrix(tFineStiffnessMatrix);
  narrate("computeCoarseStiffnessMatrix");
  int globalColCount = fineStiffnessMatrix->NumGlobalCols64();
  if (_P.get() == NULL)
  {
    constructProlongationOperator();
  }
  else if (prolongationRowCount() != globalColCount)
  {
    constructProlongationOperator();
    if (prolongationRowCount() != globalColCount)
    {
      cout << "GMGOperator::computeCoarseStiffnessMatrix: Even after a fresh call to constructProlongationOperator, _P->NumGlobalRows() != globalColCount (";
      cout << prolongationRowCount() << " != " << globalColCount << ").\n";

      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "prolongationRowCount() != globalColCount");
    }
  }

//  cout << "Writing fine stiffness to disk before setting up smoother.\n";
//  Tpetra::MatrixMarket::Writer<TMatrix<double> >::writeSparseFile("/tmp/A.dat", E_to_T::matrix(fineStiffnessMatrix));
  
  auto eComm = MPIWrapper::EpetraComm(Comm());
  Epetra_Time coarseStiffnessTimer(*eComm);

  int maxRowSize = 0; // _P->MaxNumEntries();
  
  // compute A * P
  TMatrix<double> AP(_finePartitionMap, maxRowSize);
  
  Tpetra::MatrixMatrix::Multiply(*tFineStiffnessMatrix, false, *_P, false, AP);
  
  auto domain_P = _P->getDomainMap();
  TMatrixPtr<double> PT_A_P = Teuchos::rcp( new TMatrix<double>(domain_P, maxRowSize) );
  
  // compute P^T * A * P
  Tpetra::MatrixMatrix::Multiply(*_P, true, AP, false, *PT_A_P); // calls fillComplete()
    
  auto ePT_A_P = T_to_E::matrix(PT_A_P);
  auto coarsePartitionMap = _coarseSolution->getPartitionMap();
  auto eCoarsePartitionMap = T_to_E::map(coarsePartitionMap);
  Epetra_Import  coarseImporter(*eCoarsePartitionMap, ePT_A_P->RowMap());
  
  Teuchos::RCP<Epetra_CrsMatrix> coarseStiffness;
  if (_coarseSolution->getZMCsAsGlobalLagrange() &&
      (prolongationColCount() > _coarseSolution->getDofInterpreter()->globalDofCount()))
  {
    // essentially: there are some lagrange constraints applied in coarse solve --> we shouldn't use the fused import,
    //              since this will call FillComplete()
    
    int numEntriesPerRow = 0; // sub-optimal, but easy
    coarseStiffness = Teuchos::rcp( new Epetra_CrsMatrix(::Copy, *eCoarsePartitionMap, numEntriesPerRow) );
    coarseStiffness->Import(*ePT_A_P,coarseImporter,::Insert);
    _coarseSolution->setStiffnessMatrix(coarseStiffness);
    _coarseSolution->imposeZMCsUsingLagrange(); // fills in the augmented matrix -- the ZMC rows that are at the end.
    // now can call FillComplete()
    coarseStiffness->FillComplete();
  }
  else
  {
    coarseStiffness = Teuchos::rcp( new Epetra_CrsMatrix(*ePT_A_P, coarseImporter) );
    
    _coarseSolution->setStiffnessMatrix(coarseStiffness);
    _coarseSolution->imposeZMCsUsingLagrange(); // fills in the augmented matrix -- the ZMC rows that are at the end.
  }

  _timeComputeCoarseStiffnessMatrix = coarseStiffnessTimer.ElapsedTime();

  _haveSolvedOnCoarseMesh = false; // having recomputed coarseStiffness, any existing factorization is invalid
}

// res should hold the RHS on entry
void GMGOperator::computeResidual(const TVector<Scalar>& Y, TVector<Scalar>& res, TVector<Scalar>& A_Y) const
{
  auto eComm = MPIWrapper::EpetraComm(Comm());
  Epetra_Time timer(*eComm);
  _fineStiffnessMatrix->apply(Y, A_Y);
  res.update(-1.0, A_Y, 1.0);
  _timeApplyFineStiffness += timer.ElapsedTime();
  if (_debugMode)
  {
    if (Teuchos::GlobalMPISession::getRank()==0) cout << "Updated residual after smoother application:\n";
    res.print(std::cout);
    Teuchos::RCP<Teuchos::FancyOStream> fos = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
    res.describe(*fos, Teuchos::VERB_EXTREME);
  }
}

// ! Computed as 1/(1+N), where N = max # of side neighbors of any cell's overlap region.
double GMGOperator::computeSchwarzSmootherWeight()
{
  // (For IfPack, this weight may not be exactly correct, but it's probably kinda close.  Likely we will deprecate support for
  // IFPACK_ADDITIVE_SCHWARZ soon.)
  
  // Suppose that E is a matrix with rows and columns corresponding to the elements, with 0s for disconnected elements, and 1s for
  // connected elements (an element is connected to itself and its face neighbors).
  // Conjecture: rho(E) is bounded above by the maximum side count of an element plus 1.
  // Based on results in Smith et al., I *think* that we can bound the maximum eigenvalue of S*A by 1 + rho(E)
  
  int globalMaxNeighbors = MaxGlobalOverlapSideNeighborCount(_fineMesh, _smootherOverlap,
                                                             _hierarchicalNeighborsForSchwarz,
                                                             _dimensionForSchwarzNeighborRelationship);
  
  double weight = 1.0 / (globalMaxNeighbors + 1); // aiming for a weight that guarantees max eig of weight * S * A is 1.0
  
  return weight;
}

void GMGOperator::constructLocalCoefficientMaps()
{
  narrate("constructLocalCoefficientMaps()");
  auto eComm = MPIWrapper::EpetraComm(Comm());
  Epetra_Time timer(*eComm);

  set<GlobalIndexType> cellsInPartition = _fineMesh->globalDofAssignment()->cellsInPartition(-1); // rank-local

  for (set<GlobalIndexType>::iterator cellIDIt=cellsInPartition.begin(); cellIDIt != cellsInPartition.end(); cellIDIt++)
  {
    GlobalIndexType fineCellID = *cellIDIt;
    LocalDofMapperPtr fineMapper = getLocalCoefficientMap(fineCellID);
  }
  _timeLocalCoefficientMapConstruction += timer.ElapsedTime();
}

Teuchos::RCP<Epetra_FECrsMatrix> GMGOperator::constructProlongationOperator(Teuchos::RCP<DofInterpreter> coarseDofInterpreter,
                                                                            Teuchos::RCP<DofInterpreter> fineDofInterpreter,
                                                                            bool useStaticCondensation,
                                                                            TMapPtr coarseMap, TMapPtr tFineMap,
                                                                            MeshPtr coarseMesh, MeshPtr fineMesh)
{
  // row indices belong to the fine grid, columns to the coarse
  // maps coefficients from coarse to fine
  
  int maxRowSizeToPrescribe = 0; //_coarseMesh->rowSizeUpperBound();
  
  CondensedDofInterpreter<double>* condensedDofInterpreterCoarse = NULL;
  CondensedDofInterpreter<double>* condensedDofInterpreterFine = NULL;
  
  auto eFineMap = T_to_E::map(tFineMap);
  
  set<int> varsToExclude;
  if (useStaticCondensation)
  {
    condensedDofInterpreterCoarse = dynamic_cast<CondensedDofInterpreter<double>*>(coarseDofInterpreter.get());
    condensedDofInterpreterCoarse->setCanSkipLocalFieldInInterpretGlobalCoefficients(true);
    condensedDofInterpreterFine = dynamic_cast<CondensedDofInterpreter<double>*>(fineDofInterpreter.get());
    condensedDofInterpreterFine->setCanSkipLocalFieldInInterpretGlobalCoefficients(true);
    varsToExclude = condensedDofInterpreterCoarse->condensibleVariableIDs();
  }
  
  Teuchos::RCP<Epetra_FECrsMatrix> P = Teuchos::rcp( new Epetra_FECrsMatrix(::Copy, *eFineMap, maxRowSizeToPrescribe) );
  
  // by convention, all constraints come at the end.  (Element and global lagrange constraints, zero-mean constraints.)
  // There should be the same number in the coarse and the fine Solution objects, and the mapping is taken to be the identity.
  
  GlobalIndexType firstFineConstraintRowIndex = fineDofInterpreter->globalDofCount();
  GlobalIndexType firstCoarseConstraintRowIndex = coarseDofInterpreter()->globalDofCount();
  
  // strategy: we iterate on the rank-local global dof indices on the fine mesh.
  //           we construct X, a canonical basis vector for the global dof index.
  //           we then determine the coarse mesh coefficients for X.
  //           This is a row of the matrix P.
  set<GlobalIndexType> cellsInPartition = fineMesh->globalDofAssignment()->cellsInPartition(-1); // rank-local
  
  {
    Epetra_SerialComm SerialComm; // rank-local map
    
    using GO = TMap::global_ordinal_type;
    GO* myGlobalIndicesPtr;
    eFineMap->MyGlobalElementsPtr(*&myGlobalIndicesPtr);
    
    map<GO, set<GlobalIndexType> > cellsForGlobalDofOrdinal;
    vector<GO> myGlobalIndices(eFineMap->NumMyElements());
    {
      set<GO> globalIndicesForRank;
      for  (int i=0; i<eFineMap->NumMyElements(); i++)
      {
        globalIndicesForRank.insert(myGlobalIndicesPtr[i]);
        myGlobalIndices[i] = myGlobalIndicesPtr[i];
      }
      
      // for dof interpreter's sake, want to put 0's in slots for any seen-but-not-owned global coefficients
      const set<GlobalIndexType>* myCellIDs = &fineMesh->globalDofAssignment()->cellsInPartition(-1);
      for (GlobalIndexType cellID : *myCellIDs)
      {
        set<GlobalIndexType> globalDofsForCell = fineDofInterpreter->globalDofIndicesForCell(cellID);
        for (GlobalIndexType globalDofIndex : globalDofsForCell)
        {
          cellsForGlobalDofOrdinal[globalDofIndex].insert(cellID);
          if (globalIndicesForRank.find(globalDofIndex) == globalIndicesForRank.end())
          {
            myGlobalIndices.push_back(globalDofIndex);
          }
        }
      }
    }

    // TODO: consider changing this to use an Epetra_Map with a single entry (given the usage below, this should be possible).
    Epetra_Map    localXMap(myGlobalIndices.size(), myGlobalIndices.size(), myGlobalIndices.data(), 0, SerialComm);
    Teuchos::RCP<Epetra_Vector> eXLocal = Teuchos::rcp( new Epetra_Vector(localXMap) );
    auto tXLocal = E_to_T::vector(eXLocal);
    
//    Epetra_Map    mapOneElement(1,1,&myGlobalIndices[0], 0, SerialComm);
//    Teuchos::RCP<Epetra_Vector> oneElementVector = Teuchos::rcp( new Epetra_Vector(mapOneElement) );
//    (*oneElementVector)[0] = 1.0;
    
    // to (possibly) reduce the cost of allocating these containers, declare them outside and resize inside as needed
    Camellia::FieldContainer<GO> coarseGlobalIndices;
    Camellia::FieldContainer<double> coarseGlobalValues;
    Camellia::FieldContainer<double> coarseCellCoefficients;
    Camellia::FieldContainer<double> mappedCoarseCellCoefficients;
    Camellia::FieldContainer<double> globalCoefficients;
    Camellia::FieldContainer<GlobalIndexType> globalDofIndices;
    Camellia::FieldContainer<double> fineCellCoefficients;
    Camellia::FieldContainer<double> basisCoefficients;
    
    auto XLocalViewHost = tXLocal->getLocalViewHost(Tpetra::Access::ReadWrite);
    
    const int vectorOrdinal = 0;
    
    for (int localID=0; localID < eFineMap->NumMyElements(); localID++)
    {
      auto globalRow = eFineMap->GID64(localID);
      map<GO, double> coarseXVectorLocal; // rank-local representation, so we just use an STL map.  Has the advantage of growing as we need it to.
      XLocalViewHost(localID, vectorOrdinal) = 1.0;
      
      if (globalRow >= firstFineConstraintRowIndex)
      {
        // belongs to a lagrange degree of freedom (zero-mean constraints are a special case), so we do a one-to-one map
        // NOTE: this isn't going to work properly for element Lagrange constraints in the context of h-multigrid, since
        //       for these we have one row per element, and the number of elements is reduced for the coarse mesh;
        //       it's not entirely clear to me what we should do in the general case, though for zero-mean constraints
        //       (even element-wise ones, like local conservation), probably simply weighting by relative element
        //       volume will be fine.  But we need a bit more information here; we don't have access to the fine Solution's
        //       Lagrange constraints on the present interface.
        GO offset = globalRow - firstFineConstraintRowIndex;
        GO coarseGlobalRow = firstCoarseConstraintRowIndex + offset;
        coarseXVectorLocal[coarseGlobalRow] = 1.0;
      }
      else
      {
        for (GlobalIndexType fineCellID : cellsForGlobalDofOrdinal[globalRow])
        {
          DofOrderingPtr fineTrialOrdering = fineMesh->getElementType(fineCellID)->trialOrderPtr;
          int fineDofCount = fineTrialOrdering->totalDofs();
          fineCellCoefficients.resize(fineDofCount);
        
//          {
//            // DEBUGGING
//            if (fineCellID == 1)
//            {
//              cout << "Debugging GMGOperator: fineCellID == 1.\n"; // set breakpoint here
//            }
//          }
          
          /*
           The following is a bit backwards; it should be that DofInterpreter itself could tell us which variables
           and sides belong to the provided global degree of freedom.  But adding this is a change that would take more
           effort than the below: here, we take the local coefficients and work out which variable matches.
           
           (For GDAMinimumRule, it wouldn't be *that* much effort to go from global dof index to variable; we have
           lookup tables and at worst we could do a brute force search through the lookup table for the fine cell.
           But in fact the brute force search wouldn't likely be much/any cheaper than variablesWithNonZeroEntries() below.)
           */
          fineDofInterpreter->interpretGlobalCoefficients(fineCellID, fineCellCoefficients, tXLocal);
          
//          cout << "fineCellCoefficients:\n" << fineCellCoefficients;
          
          vector<pair<int,vector<int>>> fineNonZeroVarIDs = fineTrialOrdering->variablesWithNonZeroEntries(fineCellCoefficients);
          if (fineNonZeroVarIDs.size() != 1)
          {
            GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(_fineMesh->globalDofAssignment().get());
            cout << "GlobalDofIndex " << globalRow;
            if (fineNonZeroVarIDs.size() != 0)
            {
              cout << " was interpreted to match several variables' ";
            }
            else
            {
              cout << ", interpreted, matched no variables' ";
            }
            cout << "local coefficients on cell " << fineCellID << endl;
            cout << "(useStaticCondensation = ";
            if (useStaticCondensation)
              cout << " true, ";
            else
              cout << " false, ";
            cout << "skipLocalFields = ";
            if (condensedDofInterpreterFine->canSkipLocalFieldInInterpretGlobalCoefficients())
              cout << " true)\n";
            else
              cout << " false)\n";
            
            minRule->printGlobalDofInfo();
            fineTrialOrdering->print(cout);
            cout << fineCellCoefficients;
            cout << "variables with non-zero entries: ";
            int numVars = fineNonZeroVarIDs.size();
            int varOrdinal = 0;
            for (auto entry : fineNonZeroVarIDs)
            {
              cout << "(" << entry.first << " : {";
              int numEntries = entry.second.size();
              for (int i=0; i<numEntries-1; i++)
              {
                cout << entry.second[i] << ",";
              }
              if (numEntries != 0) cout << entry.second[numEntries-1];
              cout << "} )";
              if (varOrdinal < numVars - 1) cout << ", ";
              varOrdinal++;
            }
            cout << "\n";
            TEUCHOS_TEST_FOR_EXCEPTION(fineNonZeroVarIDs.size() != 1, std::invalid_argument, "There should be exactly one variable corresponding to a given global dof ordinal in fine mesh");
          }
          
          LocalDofMapperPtr fineMapper = getLocalCoefficientMap(fineCellID);
          GlobalIndexType coarseCellID = getCoarseCellID(fineCellID);
          
          DofOrderingPtr coarseTrialOrdering = coarseMesh->getElementType(coarseCellID)->trialOrderPtr;
          int coarseDofCount = coarseMesh->getElementType(coarseCellID)->trialOrderPtr->totalDofs();
          coarseCellCoefficients.resize(coarseDofCount);
          mappedCoarseCellCoefficients.resize(fineMapper->globalIndices().size());
          
          coarseCellCoefficients = fineMapper->mapLocalData(fineCellCoefficients, true, fineNonZeroVarIDs);
          
//          cout << "coarseCellCoefficients:\n" << coarseCellCoefficients;
          
          vector<pair<int,vector<int>>> coarseNonZeroVarIDs = coarseTrialOrdering->variablesWithNonZeroEntries(coarseCellCoefficients);
          
          map<GlobalIndexType,double> fittedGlobalCoefficients;
          for (auto varIDSidesEntry : coarseNonZeroVarIDs)
          {
            int varID = varIDSidesEntry.first;
            for (int sideOrdinal : varIDSidesEntry.second)
            {
              const vector<int>* localDofIndices = &coarseTrialOrdering->getDofIndices(varID,sideOrdinal);
              basisCoefficients.resize(localDofIndices->size());
              for (int basisOrdinal=0; basisOrdinal < localDofIndices->size(); basisOrdinal++)
              {
                basisCoefficients[basisOrdinal] = coarseCellCoefficients[(*localDofIndices)[basisOrdinal]];
              }
              
              coarseDofInterpreter->interpretLocalBasisCoefficients(coarseCellID, varID, sideOrdinal, basisCoefficients,
                                                                    globalCoefficients, globalDofIndices);
              
//              cout << "basisCoefficients for varID " << varID << " on side " << sideOrdinal << ":\n" << basisCoefficients;
//              cout << "globalCoefficients:\n" << globalCoefficients;
              for (int i=0; i<globalCoefficients.size(); i++)
              {
                fittedGlobalCoefficients[globalDofIndices[i]] = globalCoefficients[i];
              }
            }
          }
          
//          Camellia::FieldContainer<double> interpretedCoarseData;
//          map<GlobalIndexType,double> fittedGlobalCoefficients;
//          coarseDofInterpreter->interpretLocalCoefficients(coarseCellID, coarseCellCoefficients, fittedGlobalCoefficients,
//                                                           varsToExclude);
          
//          { // DEBUGGING
//            if (globalRow == 4)
//            {
//              cout << "globalRow " << globalRow << ", fineCellCoefficients:\n" << fineCellCoefficients;
//              cout << "globalRow " << globalRow << ", coarseCellCoefficients:\n" << coarseCellCoefficients;
//              print("fittedGlobalCoefficients",fittedGlobalCoefficients);
//            }
//          }
          
          for (auto globalEntry : fittedGlobalCoefficients)
          {
            GlobalIndexType globalDofIndex = globalEntry.first;
            double value = globalEntry.second;
            coarseXVectorLocal[globalDofIndex] = value;
          }
        }
      }
      
      coarseGlobalIndices.resize(coarseXVectorLocal.size());
      coarseGlobalValues.resize(coarseXVectorLocal.size());
      int nnz = 0; // nonzero entries
      double tol = 1e-13; // count anything less as a zero entry
//            cout << "P global row " << globalRow << ": ";
      for (auto & entry : coarseXVectorLocal)
      {
        if (abs(entry.second) > tol)
        {
          coarseGlobalIndices[nnz] = entry.first;
          coarseGlobalValues[nnz] = entry.second;
//                    cout << coarseGlobalIndices[nnz] << " --> " << coarseGlobalValues[nnz] << "; ";
          nnz++;
        }
      }
//            cout << endl;
      if (nnz > 0)
      {
        P->InsertGlobalValues(globalRow, nnz, &coarseGlobalValues[0], &coarseGlobalIndices[0]);
//                cout << "Inserting values for row " << globalRow << endl;
      }
      XLocalViewHost(localID, vectorOrdinal) = 0.0;
    }
  }
  
//  cout << "Fine mesh has " << _fineMesh->numGlobalDofs() << " global dofs.\n";
//  
//    cout << "before FillComplete(), P has " << P->NumGlobalRows64() << " rows and " << P->NumGlobalCols64() << " columns.\n";
  
  auto eCoarseMap = T_to_E::map(coarseMap);
  P->GlobalAssemble(*eCoarseMap, *eFineMap);
  
//  cout << "after FillComplete(), P has " << P->NumGlobalRows64() << " rows and " << P->NumGlobalCols64() << " columns.\n";
  
  const static bool DEBUGGING = false;
  if (DEBUGGING)
  { // DEBUGGING
    // check whether all domain map (coarseMap) GIDs are present as column indices
    // (they should be, but it appears there are cases when they aren't)
    int numDomainElements = eCoarseMap->NumMyElements();
    
    int numMyRows = P->NumMyRows();
    
    vector<bool> localGID(numDomainElements,false); // track which LIDs we have found
    int numLocalColGIDs = 0;
    
    vector<int> colIndices;
    vector<double> values;
    for (int i=0; i<numMyRows; i++)
    {
      int numRowEntries = P->NumMyEntries(i);
      colIndices.resize(numRowEntries);
      values.resize(numRowEntries);
      P->ExtractMyRowCopy(i, numRowEntries, numRowEntries, &values[0], &colIndices[0]);
      for (int j=0; j<numRowEntries; j++)
      {
        int GID = colIndices[j];
        
        int LID = eCoarseMap->LID(GID);
        if (LID != -1)
        {
          bool previouslyFound = localGID[LID];
          if (!previouslyFound)
          {
            localGID[LID] = true;
            numLocalColGIDs++;
          }
        }
      }
    }
    cout << "Found " << numLocalColGIDs << " of " << numDomainElements << " in coarseMap.\n";
    
    if (numLocalColGIDs != numDomainElements)
    {
      cout << "Missing coarse GIDs: ";
      for (int colLID=0; colLID<numDomainElements; colLID++)
      {
        if (localGID[colLID]) continue;
        auto GID = eCoarseMap->GID64(colLID);
        cout << GID << " ";
      }
      cout << endl;
      
      GDAMinimumRule* gdaMinRuleCoarseMesh = dynamic_cast<GDAMinimumRule *>(_coarseMesh->globalDofAssignment().get());
      gdaMinRuleCoarseMesh->printGlobalDofInfo();
    }
  }
  
  if (condensedDofInterpreterFine != NULL)
  {
    condensedDofInterpreterFine->setCanSkipLocalFieldInInterpretGlobalCoefficients(false); //just true while in this call
    if (!_isFinest || _clearFinestCondensedDofInterpreterAfterProlongation)
      condensedDofInterpreterFine->clearStiffnessAndLoad();
  }
  
  if (condensedDofInterpreterCoarse != NULL)
  {
    condensedDofInterpreterCoarse->setCanSkipLocalFieldInInterpretGlobalCoefficients(false);
  }
  
  return P;
}

TMatrixPtr<Scalar> GMGOperator::constructProlongationOperator()
{
  auto eComm = MPIWrapper::EpetraComm(Comm());
  Epetra_Time prolongationTimer(*eComm);
  narrate("constructProlongationOperator");
  
  auto coarseMap = _coarseSolution->getPartitionMap();
  auto eP = this->constructProlongationOperator(_coarseSolution->getDofInterpreter(), _fineDofInterpreter, _useStaticCondensation,
                                                coarseMap, _finePartitionMap, _coarseMesh, _fineMesh);
  
//  std::cout << "Epetra P has " << eP->NumGlobalRows() << " rows and " << eP->NumGlobalCols() << " columns.\n";
  
  _P = E_to_T::matrix(eP);

//  std::cout << "Tpetra _P has " << _P->getGlobalNumRows() << " rows and " << _P->getGlobalNumCols() << " columns.\n";
  
  _timeProlongationOperatorConstruction = prolongationTimer.ElapsedTime();
  
  ostringstream prolongationTimingReport;
  prolongationTimingReport << "Prolongation operator constructed in " << _timeProlongationOperatorConstruction << " seconds.";
  narrate(prolongationTimingReport.str());
  
  return _P;
}

void GMGOperator::exportFunction()
{
  if (_functionExporter != Teuchos::null)
  {
    _functionExporter->exportFunction(_functionToExport, _functionToExportName, _exportTimeStep);
    _exportTimeStep++;
  }
}

Teuchos::RCP<SerialDenseMatrix> GMGOperator::fluxDuplicationMapForCoarseCell(GlobalIndexType coarseCellID) const
{
  // a null return value indicates that the operator is the identity (this happens for non-conforming ElementTypes)
 
  // otherwise, returns a matrix that can right-multiply the fluxToFieldMap
  
  DofOrdering* trialOrderRawPtr = _coarseMesh->getElementType(coarseCellID)->trialOrderPtr.get();
  if (_fluxDuplicationMap.find(trialOrderRawPtr) == _fluxDuplicationMap.end())
  {
    bool hasConformingTraces = false;
    for (int varID : trialOrderRawPtr->getVarIDs())
    {
      if (!trialOrderRawPtr->isVolumeVar(varID))
      {
        // trace or flux; use basis on first side as example
        EFunctionSpace fs = trialOrderRawPtr->getBasis(varID, trialOrderRawPtr->getSidesForVarID(varID)[0])->functionSpace();
        if (! functionSpaceIsDiscontinuous(fs))
        {
          hasConformingTraces = true;
          break;
        }
      }
    }
    if (!hasConformingTraces)
    {
      _fluxDuplicationMap[trialOrderRawPtr] = Teuchos::null;
      return _fluxDuplicationMap[trialOrderRawPtr];
    }
    
    CondensedDofInterpreter<double>* condensedDofInterpreterCoarse = NULL;
    
    if (_useStaticCondensation)
    {
      condensedDofInterpreterCoarse = dynamic_cast<CondensedDofInterpreter<double>*>(_coarseSolution->getDofInterpreter().get());
    }
    TEUCHOS_TEST_FOR_EXCEPTION(condensedDofInterpreterCoarse == NULL, std::invalid_argument, "fluxDuplicationMapForCoarseCell() requires a coarse condensed dof interpreter...");
    
    vector<int> fluxIndices = condensedDofInterpreterCoarse->fluxIndexLookupLocalCell(coarseCellID);
    
    Teuchos_CommPtr SerialComm = MPIWrapper::TeuchosCommSerial();
    MeshPtr coarseSingleCellMesh = Teuchos::rcp( new Mesh(_coarseMesh, coarseCellID, SerialComm) );
    
    // the fact that the local view does not enforce conformity means that for conforming traces
    // the coarseFluxToFieldMapMatrix does not do the right thing if we only map, say, one of two
    // dofs identified with a vertex.  Since this is the way we approach things below, here we manipulate
    // the matrix, essentially enforcing conformity:
    // (NOTE: the same thing could be accomplished at the abstract, reference-element level.  It would be more
    //  efficient to construct a matrix like this once for the element type, and then multiply matrices to enforce
    //  conformity...)
    Camellia::FieldContainer<double> localData(trialOrderRawPtr->totalDofs());
    Camellia::FieldContainer<double> localCoefficients(trialOrderRawPtr->totalDofs());
    Camellia::FieldContainer<double> globalCoefficients;
    Camellia::FieldContainer<GlobalIndexType> globalDofIndices;
    GlobalIndexType zeroCellID = 0;
    int indexBase = 0;
    int numDofsSingleElementMesh = coarseSingleCellMesh->numGlobalDofs();
    auto epetraSerialComm = MPIWrapper::EpetraComm(SerialComm);
    Epetra_Map serialGlobalDofMap(numDofsSingleElementMesh,indexBase,*epetraSerialComm);
    Epetra_MultiVector globalCoefficientsVector(serialGlobalDofMap,1,true);
    
    Teuchos::RCP<SerialDenseMatrix> fluxMap = Teuchos::rcp( new SerialDenseMatrix(fluxIndices.size(),fluxIndices.size()) );
    
    int fluxMapColumn = 0;
    for (int fluxIndex : fluxIndices)
    {
      localData.initialize(0.0);
      localData(fluxIndex) = 1.0;
      
      coarseSingleCellMesh->globalDofAssignment()->interpretLocalData(zeroCellID, localData, globalCoefficients, globalDofIndices);
      
      for (int i=0; i<globalCoefficients.size(); i++)
      {
        int globalDofIndex = globalDofIndices[i];
        globalCoefficientsVector[0][globalDofIndex] = globalCoefficients[i];
      }
      
      coarseSingleCellMesh->globalDofAssignment()->interpretGlobalCoefficients(zeroCellID, localCoefficients, globalCoefficientsVector);
      for (int fluxMapRow=0; fluxMapRow<fluxIndices.size(); fluxMapRow++)
      {
        (*fluxMap)(fluxMapRow,fluxMapColumn) = localCoefficients(fluxIndices[fluxMapRow]);
      }
      fluxMapColumn++;
    }
    _fluxDuplicationMap[trialOrderRawPtr] = fluxMap;
  }
  return _fluxDuplicationMap[trialOrderRawPtr];
}

//! Dimension that will be used for neighbor relationships for Schwarz blocks, if CAMELLIA_ADDITIVE_SCHWARZ is the smoother choice.
int GMGOperator::getDimensionForSchwarzNeighborRelationship()
{
  return _dimensionForSchwarzNeighborRelationship;
}

//! Set dimension to use for neighbor relationships for Schwarz blocks.  Requires CAMELLIA_ADDITIVE_SCHWARZ as the smoother choice.
void GMGOperator::setDimensionForSchwarzNeighborRelationship(int value)
{
  _dimensionForSchwarzNeighborRelationship = value;
}

GlobalIndexType GMGOperator::getCoarseCellID(GlobalIndexType fineCellID) const
{
  const set<IndexType>* coarseCellIDs = &_coarseMesh->getTopology()->getLocallyKnownActiveCellIndices();
  CellPtr fineCell = _fineMesh->getTopology()->getCell(fineCellID);
  CellPtr ancestor = fineCell;
  RefinementBranch refBranch;
  while (coarseCellIDs->find(ancestor->cellIndex()) == coarseCellIDs->end())
  {
    CellPtr parent = ancestor->getParent();
    if (parent.get() == NULL)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ancestor for fine cell not found in coarse mesh");
    }
    unsigned childOrdinal = parent->childOrdinal(ancestor->cellIndex());
    refBranch.insert(refBranch.begin(), make_pair(parent->refinementPattern().get(), childOrdinal));
    ancestor = parent;
  }
  return ancestor->cellIndex();
}

Teuchos::RCP<GMGOperator> GMGOperator::getCoarseOperator()
{
  return _coarseOperator;
}

TSolutionPtr<double> GMGOperator::getCoarseSolution()
{
  return _coarseSolution;
}

SolverPtr GMGOperator::getCoarseSolver()
{
  return _coarseSolver;
}

Teuchos::RCP<DofInterpreter> GMGOperator::getFineDofInterpreter()
{
  return _fineDofInterpreter;
}

MeshPtr GMGOperator::getFineMesh() const
{
  return _fineMesh;
}

TMatrixPtr<Scalar> GMGOperator::getFineStiffnessMatrix()
{
  return _fineStiffnessMatrix;
}

LocalDofMapperPtr GMGOperator::getLocalCoefficientMap(GlobalIndexType fineCellID) const
{
  GDAMinimumRule* gdaMinRuleFineMesh = dynamic_cast<GDAMinimumRule *>(_fineMesh->globalDofAssignment().get());

  // define lambda for filtering out fine dofs that don't belong:
  auto filterOutDuplicateFineDofs = [gdaMinRuleFineMesh, this] (BasisPtr fineBasis, CellPtr fineCell,
                                                                int sideOrdinal,
                                                                bool coarseInteriorDofsOnly,
                                                                RefinementBranch &refBranch,
                                                                SubBasisReconciliationWeights &weights) -> void
  {
    // Now: Zero out weights on fine dofs that are
    // - *not* interior to the coarse domain, or
    // - on interior subcells that are not owned by the child cell.
    GlobalIndexType fineCellID = fineCell->cellIndex();
    int minSubcellDimension = BasisReconciliation::minimumSubcellDimension(fineBasis);
    CellTopoPtr fineTopo = fineBasis->domainTopology();
    vector<vector<OwnershipInfo>> subcellOwnershipInfo = gdaMinRuleFineMesh->getCellConstraints(fineCellID).owningCellIDForSubcell;
    
    int sideDim = fineCell->topology()->getDimension() - 1;
    
    set<int> fineDofsToDiscard;
    for (int d=minSubcellDimension; d<=sideDim; d++)
    {
      for (int subcord=0; subcord<fineTopo->getSubcellCount(d); subcord++)
      {
        unsigned childCellSubcord = CamelliaCellTools::subcellOrdinalMap(fineCell->topology(), sideDim, sideOrdinal, d, subcord);
        bool shouldDiscard = false;
        if (subcellOwnershipInfo[d][childCellSubcord].cellID != fineCellID)
        {
          shouldDiscard = true;
        }
        else
        {
          if (d < sideDim)
          {
            // if we get here, this cell owns, but we need to check whether there is an earlier side on this cell that
            // also sees this subcell
            for (int earlierSideOrdinal = 0; earlierSideOrdinal < sideOrdinal; earlierSideOrdinal++)
            {
              CellTopoPtr earlierSide = fineCell->topology()->getSide(earlierSideOrdinal);
              int earlierSubcellCount = earlierSide->getSubcellCount(d);
              for (int earlierSubcord=0; earlierSubcord<earlierSubcellCount; earlierSubcord++)
              {
                unsigned earlierSubcordInChildCell = CamelliaCellTools::subcellOrdinalMap(fineCell->topology(),
                                                                                          sideDim, earlierSideOrdinal,
                                                                                          d, earlierSubcord);
                if (earlierSubcordInChildCell == childCellSubcord)
                {
                  shouldDiscard = true;
                  break;
                }
              }
            }
          }
          if (!shouldDiscard) // if shouldDiscard is already true, then we don't need to check whether subcell is interior
          {
            if (coarseInteriorDofsOnly)
            {
              pair<unsigned, unsigned> coarseSubcell = RefinementPattern::mapSubcellFromDescendantToAncestor(refBranch, d,
                                                                                                             childCellSubcord);
              unsigned coarseSubcellDim = coarseSubcell.first;
              if (coarseSubcellDim <= sideDim)
              {
                // *NOT* interior to the coarse cell
                shouldDiscard = true;
              }
            }
          }
        }
        if (shouldDiscard)
        {
          const vector<int>* dofOrdinals = &fineBasis->dofOrdinalsForSubcell(d, subcord);
          fineDofsToDiscard.insert(dofOrdinals->begin(),dofOrdinals->end());
        }
      }
    }
    int row = 0; // in weights container
    for (int fineDofOrdinal : weights.fineOrdinals)
    {
      if (fineDofsToDiscard.find(fineDofOrdinal) != fineDofsToDiscard.end())
      {
        // zero out
        for (int col=0; col<weights.weights.dimension(1); col++)
        {
          weights.weights(row,col) = 0;
        }
      }
      row++;
    }
    weights = BasisReconciliation::filterOutZeroRowsAndColumns(weights);
  };
  
  const set<IndexType>* coarseCellIDs = &_coarseMesh->getTopology()->getLocallyKnownActiveCellIndices();
  CellPtr fineCell = _fineMesh->getTopology()->getCell(fineCellID);
  CellPtr ancestor = fineCell;
  RefinementBranch refBranch;

  while (coarseCellIDs->find(ancestor->cellIndex()) == coarseCellIDs->end())
  {
    CellPtr parent = ancestor->getParent();
    if (parent.get() == NULL)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ancestor for fine cell not found in coarse mesh");
    }
    unsigned childOrdinal = parent->childOrdinal(ancestor->cellIndex());
    refBranch.insert(refBranch.begin(), make_pair(parent->refinementPattern().get(), childOrdinal));
    ancestor = parent;
  }
  GlobalIndexType coarseCellID = ancestor->cellIndex();
  CellPtr coarseCell = ancestor;
  if (_fineMesh->globalDofAssignment()->getH1Order(fineCellID).size() > 1)
  {
    // we have potentially multiple polynomial orders (one for space, one for time, say), whereas below we use just the
    // first polynomial order as a key.  This is OK, so long as there aren't anisotropic refinements in polynomial order
    // (as of this writing, we don't offer these).
    int rank = Teuchos::GlobalMPISession::getRank();
    static bool haveWarned = false;
    if ((rank==0) && (!haveWarned))
    {
      cout << "Note: using tensor-product polynomial orders in GMGOperator.  This is supported so long as no anisotropic";
      cout << " refinements are done in the polynomial order.  A small upgrade to GMGOperator's lookup tables would be";
      cout << " required to support anisotropic p-refinements.\n";
      haveWarned = true;
    }
  }
  int fineOrder = _fineMesh->globalDofAssignment()->getH1Order(fineCellID)[0];
  int coarseOrder = _coarseMesh->globalDofAssignment()->getH1Order(coarseCellID)[0];

  DofOrderingPtr coarseTrialOrdering = _coarseMesh->getElementType(coarseCellID)->trialOrderPtr;
  DofOrderingPtr fineTrialOrdering = _fineMesh->getElementType(fineCellID)->trialOrderPtr;

//  cout << "coarse trial ordering: \n";
//  coarseTrialOrdering->print(cout);
  
  pair< pair<int,int>, RefinementBranch > key = make_pair(make_pair(fineOrder, coarseOrder), refBranch);

  CondensedDofInterpreter<double>* condensedDofInterpreterCoarse = NULL;

  if (_useStaticCondensation)
  {
    condensedDofInterpreterCoarse = dynamic_cast<CondensedDofInterpreter<double>*>(_coarseSolution->getDofInterpreter().get());
  }
  
  // When doing static condensation for h-multigrid, the prolongation operator involves inversion of the
  // coarse local stiffness matrix to determine the fields on the coarse element; we take the traces of
  // these fields to determine the fine traces on the interior of the element.  Since these computations
  // may be spatially dependent, we can't reuse the prolongation that we determined for some other cell whose
  // refinement branch and element types match ours.
  bool cellProlongationCanMatchPatterns = (coarseCellID == fineCellID) || (condensedDofInterpreterCoarse == NULL);

  int fineSideCount = fineCell->getSideCount();
  int spaceDim = _fineMesh->getTopology()->getDimension();
  int sideDim = spaceDim - 1;
  vector<unsigned> ancestralSideOrdinals(fineSideCount);
  vector< RefinementBranch > sideRefBranches(fineSideCount);
  for (int sideOrdinal=0; sideOrdinal<fineSideCount; sideOrdinal++)
  {
    ancestralSideOrdinals[sideOrdinal] = RefinementPattern::ancestralSubcellOrdinal(refBranch, sideDim, sideOrdinal);
    if (ancestralSideOrdinals[sideOrdinal] != INVALID_CELLTOPOLOGY_SIDE_ORDINAL)
    {
      sideRefBranches[sideOrdinal] = RefinementPattern::sideRefinementBranch(refBranch, sideOrdinal);
    }
  }

  LocalDofMapperPtr dofMapper;
  if (cellProlongationCanMatchPatterns && (_localCoefficientMap.find(key) != _localCoefficientMap.end()))
  {
    dofMapper = _localCoefficientMap[key];
  }
  else
  {
    BasisCachePtr fineBasisCache   = BasisCache::basisCacheForCell(_fineMesh, fineCellID);
    BasisCachePtr coarseBasisCache = BasisCache::basisCacheForCell(_coarseMesh, coarseCellID);
    
    Teuchos::RCP<Teuchos::SerialDenseMatrix<int, Scalar> > coarseFluxToFieldMapMatrix;
    vector<int> fluxOrdinalToLocalDofIndex;
    set<int> condensibleVarIDs;
    MeshPtr coarseSingleCellMesh;
    if (!cellProlongationCanMatchPatterns)
    {
      int timerHandle = TimeLogger::sharedInstance()->startTimer("compute flux to field map");
      // then we're doing static condensation in context of h-refinement: we'll need to compute coarseFluxToFieldMap
      coarseFluxToFieldMapMatrix = condensedDofInterpreterCoarse->fluxToFieldMapForIterativeSolves(coarseCellID);
      fluxOrdinalToLocalDofIndex = condensedDofInterpreterCoarse->fluxIndexLookupLocalCell(coarseCellID);
      condensibleVarIDs = condensedDofInterpreterCoarse->condensibleVariableIDs();

      Teuchos::RCP<SerialDenseMatrix> fluxDuplicationMap = fluxDuplicationMapForCoarseCell(coarseCellID);
      if (fluxDuplicationMap != Teuchos::null)
      {
        Teuchos::RCP<Teuchos::SerialDenseMatrix<int,Scalar>> modifiedMatrix
        = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int,Scalar>(coarseFluxToFieldMapMatrix->numRows(),
                                                                   coarseFluxToFieldMapMatrix->numCols()));
        // resolves to:
        modifiedMatrix->multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1, *coarseFluxToFieldMapMatrix, *fluxDuplicationMap, 0.0);
        coarseFluxToFieldMapMatrix = modifiedMatrix;
      }
      TimeLogger::sharedInstance()->stopTimer(timerHandle);
    }
    
    VarFactoryPtr vf = _fineMesh->varFactory();

    typedef vector< SubBasisDofMapperPtr > BasisMap; // taken together, these maps map a whole basis
    map< int, BasisMap > volumeMaps;

    vector< map< int, BasisMap > > sideMaps(fineSideCount);

    set<int> trialIDs = coarseTrialOrdering->getVarIDs();

    // set up fittability: which coarse dof ordinals have support on volume and each side
    set<GlobalIndexType> fittableGlobalDofOrdinalsInVolume;
    vector< set<GlobalIndexType> > fittableGlobalDofOrdinalsOnSides(fineSideCount);
    
    int coarseSideCount = coarseTrialOrdering->cellTopology()->getSideCount();
    
    vector<vector<double>> fineSideReferenceUnitNormals(fineSideCount);   // lazily filled for interior sides when static condensation is enabled
    vector<vector<double>> coarseSideReferenceUnitNormals(coarseSideCount); // lazily filled for interior sides when static condensation is enabled
    
    // for the moment, we skip the mapping from traces to fields based on traceTerm
    unsigned vertexNodePermutation = 0; // because we're "reconciling" to an ancestor, the views of the cells and sides are necessarily the same
    for (set<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++)
    {
      int trialID = *trialIDIt;

      VarPtr trialVar = vf->trialVars().find(trialID)->second;

      if (coarseTrialOrdering->isVolumeVar(trialID))   // field variable
      {
        if (condensedDofInterpreterCoarse != NULL)
        {
          if (condensedDofInterpreterCoarse->varDofsAreCondensible(coarseCellID, trialID, 0, coarseTrialOrdering)) continue;
        }
//        cout << "Warning: for debugging purposes, skipping projection of fields in GMGOperator.\n";
        BasisPtr coarseBasis = coarseTrialOrdering->getBasis(trialID);
        BasisPtr fineBasis = fineTrialOrdering->getBasis(trialID);
        if (fineBasis->functionSpace() != coarseBasis->functionSpace())
        {
          // This will cause an error in BasisReconciliation; can we find equivalent continuous basis?
          // (necessary when using a non-conforming coarse mesh, though not at present an issue for fields, as here)
          coarseBasis = coarseBasis->continuousBasis();
          TEUCHOS_TEST_FOR_EXCEPTION(fineBasis->functionSpace() != coarseBasis->functionSpace(), std::invalid_argument, "Even after getting continuous basis for coarseBasis, fine and coarse function spaces disagree");
        }
        SubBasisReconciliationWeights weights = _br->constrainedWeights(fineBasis, refBranch, coarseBasis, vertexNodePermutation);
        
        vector<GlobalIndexType> coarseDofIndices;
        for (set<int>::iterator coarseOrdinalIt=weights.coarseOrdinals.begin(); coarseOrdinalIt != weights.coarseOrdinals.end(); coarseOrdinalIt++)
        {
          unsigned coarseDofIndex = coarseTrialOrdering->getDofIndex(trialID, *coarseOrdinalIt);
          coarseDofIndices.push_back(coarseDofIndex);
        }
        fittableGlobalDofOrdinalsInVolume.insert(coarseDofIndices.begin(),coarseDofIndices.end());
        BasisMap basisMap(1,SubBasisDofMapper::subBasisDofMapper(weights.fineOrdinals, coarseDofIndices, weights.weights));
        volumeMaps[trialID] = basisMap;
//        cout << "weights for trial ID " << trialID << ":\n" << weights.weights;
      }
      else     // flux/trace
      {
//        cout << "Warning: for debugging purposes, skipping projection of fluxes and traces in GMGOperator.\n";
        for (int sideOrdinal=0; sideOrdinal<fineSideCount; sideOrdinal++)
        {
          BasisCachePtr fineDomainCache = fineBasisCache->getSideBasisCache(sideOrdinal);
          
          if (! fineTrialOrdering->hasBasisEntry(trialID, sideOrdinal)) continue;
          
          if (condensedDofInterpreterCoarse != NULL)
          {
            if (condensedDofInterpreterCoarse->varDofsAreCondensible(coarseCellID, trialID, sideOrdinal, coarseTrialOrdering)) continue;
          }
          FunctionPtr sideParity = Function::sideParity();
          
          unsigned coarseSideOrdinal = ancestralSideOrdinals[sideOrdinal];
          BasisPtr coarseBasis, fineBasis;
          BasisMap basisMap;
          bool useTermTracedIfAvailable = true; // flag for debugging purposes (true is the production setting)
          if (coarseSideOrdinal == INVALID_CELLTOPOLOGY_SIDE_ORDINAL)   // the fine side falls inside a coarse volume
          {
            // we map trace to field using the traceTerm LinearTermPtr
            VarPtr trialVar = vf->trial(trialID);

            LinearTermPtr termTraced = trialVar->termTraced();
            if (!useTermTracedIfAvailable || (termTraced.get() == NULL)) // nothing we can do if we don't know what term we're tracing
              continue;

            if (! fineTrialOrdering->hasBasisEntry(trialID, sideOrdinal) ) continue;
            fineBasis = fineTrialOrdering->getBasis(trialID, sideOrdinal);

//            cout << "Processing termTraced for variable " << trialVar->name() << endl;
            
            set<int> varsTraced = termTraced->varIDs();
            for (set<int>::iterator varTracedIt = varsTraced.begin(); varTracedIt != varsTraced.end(); varTracedIt++)
            {
              int varTracedID = *varTracedIt;
              
              if ((condensedDofInterpreterCoarse == NULL) || (condensibleVarIDs.find(varTracedID) == condensibleVarIDs.end())) // the latter case is like the pressure for Stokes VGP
              {
                coarseBasis = coarseTrialOrdering->getBasis(varTracedID);
                
                unsigned coarseSubcellOrdinal = 0, coarseDomainOrdinal = 0; // the volume
                unsigned coarseSubcellPermutation = 0;
                unsigned fineSubcellOrdinalInFineDomain = 0; // the side is the whole fine domain...
                
                SubBasisReconciliationWeights weights;
                
                if (coarseBasis->functionSpace() == fineBasis->functionSpace())
                {
                  // then we can compute basis weights in reference space, and cache them
                  weights = _br->constrainedWeightsForTermTraced(termTraced, varTracedID,
                                                                 sideDim, fineBasis, fineSubcellOrdinalInFineDomain, refBranch, sideOrdinal,
                                                                 ancestor->topology(),
                                                                 spaceDim, coarseBasis, coarseSubcellOrdinal, coarseDomainOrdinal, coarseSubcellPermutation);
                }
                else
                {
                  weights = BasisReconciliation::computeConstrainedWeightsForTermTraced(termTraced, varTracedID, sideDim, fineBasis, fineSubcellOrdinalInFineDomain, refBranch, sideOrdinal, ancestor->topology(), spaceDim, coarseBasis, fineDomainCache, coarseBasisCache);
                }
                // Now: Zero out weights on fine dofs that are
                // - *not* interior to the coarse domain, or
                // - on interior subcells that are not owned by the child cell.
                filterOutDuplicateFineDofs(fineBasis, fineCell, sideOrdinal, true, refBranch, weights);
                // NOTE: applying the above filter is a bit fragile: since we are doing pattern-matching, it relies
                //       on ownership of the fine dofs on the interior of a refined element being assigned in the same
                //       way on every refined element.  This is in fact the case in GDAMinimumRule, and it seems a natural
                //       thing, but if it were to change, that would be a problem.
                
                // we may have filtered out all the fine dofs; if so, we can continue
                if (weights.weights.size() == 0)
                {
                  continue;
                }
                
                // **** end new weight-adjusting code ****
                
                vector<GlobalIndexType> coarseDofIndices;
                for (set<int>::iterator coarseOrdinalIt=weights.coarseOrdinals.begin(); coarseOrdinalIt != weights.coarseOrdinals.end(); coarseOrdinalIt++)
                {
                  unsigned coarseDofIndex = coarseTrialOrdering->getDofIndex(varTracedID, *coarseOrdinalIt);
                  coarseDofIndices.push_back(coarseDofIndex);
                  fittableGlobalDofOrdinalsOnSides[sideOrdinal].insert(coarseDofIndex);
  //                cout << "termTraced weights:\n" << weights.weights;
                }

                basisMap.push_back(SubBasisDofMapper::subBasisDofMapper(weights.fineOrdinals, coarseDofIndices, weights.weights));
              }
              else
              {
                // If we're doing static condensation, map from coarse flux to coarse field using CondensedDofInterpreter, then
                // back to fluxes using termTraced.
                
                int timerHandle = TimeLogger::sharedInstance()->startTimer("map fluxes to fields to fluxes");
                BasisPtr volumeBasis = coarseTrialOrdering->getBasis(varTracedID);
                CellTopoPtr volumeTopo = volumeBasis->domainTopology();
                
                unsigned volumeSubcellOrdinal = 0, volumeDomainOrdinal = 0;
                unsigned volumeSubcellPermutation = 0;
                unsigned fineSubcellOrdinalInFineDomain = 0; // the side is the whole fine domain...
                
                SubBasisReconciliationWeights weights;
                if (volumeBasis->functionSpace() == fineBasis->functionSpace())
                {
                  // then we can compute basis weights in reference space, and cache them
                  weights = _br->constrainedWeightsForTermTraced(termTraced, varTracedID,
                                                                 sideDim, fineBasis,
                                                                 fineSubcellOrdinalInFineDomain, refBranch,
                                                                 sideOrdinal, volumeTopo,
                                                                 spaceDim, volumeBasis, volumeSubcellOrdinal,
                                                                 volumeDomainOrdinal, volumeSubcellPermutation);
                }
                else
                {
                  weights = BasisReconciliation::computeConstrainedWeightsForTermTraced(termTraced, varTracedID, sideDim, fineBasis, fineSubcellOrdinalInFineDomain, refBranch, sideOrdinal, ancestor->topology(), spaceDim, volumeBasis, fineDomainCache, coarseBasisCache);
                }
                
                weights = BasisReconciliation::filterOutZeroRowsAndColumns(weights);
                
                // we may have filtered out all the fine dofs; if so, we can continue
                if (weights.weights.size() == 0)
                {
                  continue;
                }
                
                // TODO: treat the case (as with pressure in Stokes VGP) where termTraced includes some field variables that are not condensible.
                
//                print("weights, fineOrdinals", weights.fineOrdinals);
//                print("weights, coarseOrdinals", weights.coarseOrdinals);
//                cout << "weights, weights:\n" << weights.weights;
                
                // extract a SerialDenseMatrix corresponding to just the varTracedID dofs
                set<int> volumeBasisOrdinals = weights.coarseOrdinals;
                vector<int> fieldTracedIndices = condensedDofInterpreterCoarse->fieldRowIndices(coarseCellID, varTracedID);
                vector<int> volumeBasisOrdinalsForFields;
                for (int volumeBasisOrdinal : volumeBasisOrdinals)
                {
                  int fieldTracedIndex = fieldTracedIndices[volumeBasisOrdinal];
                  if (fieldTracedIndex != -1)
                  {
                    // we map this one
                    volumeBasisOrdinalsForFields.push_back(volumeBasisOrdinal);
                  }
                }
                
                SerialDenseMatrix coarseFluxToFieldTraced(volumeBasisOrdinalsForFields.size(),coarseFluxToFieldMapMatrix->numCols()); // columns here correspond to our coarse dof indices -- what we map to
                
                int reducedRowIndex = 0;
                for (int volumeBasisOrdinal : volumeBasisOrdinalsForFields)
                {
                  int fieldTracedIndex = fieldTracedIndices[volumeBasisOrdinal];
                  int numCols = coarseFluxToFieldMapMatrix->numCols();
                  for (int col = 0; col < numCols; col++)
                  {
                    coarseFluxToFieldTraced(reducedRowIndex,col) = (*coarseFluxToFieldMapMatrix)(fieldTracedIndex,col);
                  }
                  reducedRowIndex++;
                }
                int n = weights.weights.dimension(0);
                int m = weights.weights.dimension(1);
                if ((n==0) || (m==0))
                {
                  continue;
                }
                double *firstEntry = (double *) &weights.weights[0];
                SerialDenseMatrix fineBasisToCoarseFieldMatrix(Teuchos::Copy,firstEntry,m,m,n); // probably could do Teuchos::View here
                SerialDenseMatrix fineBasisToCoarseFluxMatrix(weights.fineOrdinals.size(),coarseFluxToFieldTraced.numCols());
                
                // TRANS: SDM col-major, FC row-major
                int success = fineBasisToCoarseFluxMatrix.multiply(Teuchos::TRANS,Teuchos::NO_TRANS, 1.0,
                                                                   fineBasisToCoarseFieldMatrix, coarseFluxToFieldTraced,
                                                                   0.0);
                
                vector<GlobalIndexType> coarseDofIndicesVector;
                
                // Look for zero columns in fineBasisToCoarseFluxMatrix
                vector<unsigned> nonZeroColumnIndices;
                map<GlobalIndexType,unsigned> columnOrdinalForCoarseDofIndex; // allows sorting by coarse dof index
                double tol = 1e-15;
                for (int j=0; j<fineBasisToCoarseFluxMatrix.numCols(); j++)
                {
//                  bool nonZero = false;
                  for (int i=0; i<fineBasisToCoarseFluxMatrix.numRows(); i++)
                  {
                    if (abs(fineBasisToCoarseFluxMatrix(i,j)) > tol)
                    {
//                      nonZero = true;
                      nonZeroColumnIndices.push_back(j);
                      columnOrdinalForCoarseDofIndex[fluxOrdinalToLocalDofIndex[j]] = coarseDofIndicesVector.size();
                      coarseDofIndicesVector.push_back(fluxOrdinalToLocalDofIndex[j]);
                      break;
                    }
                  }
                }
                
                vector<unsigned> permutedNonZeroColumnIndices;
                for (auto entry : columnOrdinalForCoarseDofIndex) {
                  permutedNonZeroColumnIndices.push_back(nonZeroColumnIndices[entry.second]);
                }
                
                weights.weights.resize(fineBasisToCoarseFluxMatrix.numRows(),nonZeroColumnIndices.size());
                for (int i=0; i<weights.weights.dimension(0); i++)
                {
                  for (int j=0; j<weights.weights.dimension(1); j++)
                  {
                    weights.weights(i,j) = fineBasisToCoarseFluxMatrix(i,permutedNonZeroColumnIndices[j]);
                  }
                }
                weights.coarseOrdinals.clear();
                weights.coarseOrdinals.insert(coarseDofIndicesVector.begin(),coarseDofIndicesVector.end());
                
//                print("coarseDofIndices", weights.coarseOrdinals);
//                cout << "weights:\n" << weights.weights;
                
                // Now: Zero out weights on fine dofs that are
                // - *not* interior to the coarse domain, or
                // - on interior subcells that are not owned by the child cell.
                filterOutDuplicateFineDofs(fineBasis, fineCell, sideOrdinal, true, refBranch, weights);
                // we may have filtered out all the fine dofs; if so, we can continue
                if (weights.weights.size() == 0)
                {
                  continue;
                }
                
                coarseDofIndicesVector.clear();
                coarseDofIndicesVector.insert(coarseDofIndicesVector.begin(),
                                              weights.coarseOrdinals.begin(), weights.coarseOrdinals.end());
                // **** end new weight-adjusting code ****
                
                
                basisMap.push_back(SubBasisDofMapper::subBasisDofMapper(weights.fineOrdinals, coarseDofIndicesVector, weights.weights));
                fittableGlobalDofOrdinalsOnSides[sideOrdinal].insert(coarseDofIndicesVector.begin(),coarseDofIndicesVector.end());
              
                TimeLogger::sharedInstance()->stopTimer(timerHandle);
              }
            }
          }
          else     // fine side maps to a coarse side
          {
            if (! coarseTrialOrdering->hasBasisEntry(trialID, coarseSideOrdinal))
            {
              cout << "ERROR: no entry for trial var " << trialVar->name() << " on side " << coarseSideOrdinal << " in coarse mesh.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Missing basis entry on coarse side!");
            }
            if (! fineTrialOrdering->hasBasisEntry(trialID, sideOrdinal))
            {
              cout << "ERROR: no entry for trial var " << trialVar->name() << " on side " << sideOrdinal << " in fine mesh.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Missing basis entry on fine side!");
            }
            coarseBasis = coarseTrialOrdering->getBasis(trialID, coarseSideOrdinal);
            fineBasis = fineTrialOrdering->getBasis(trialID, sideOrdinal);
            if (fineBasis->functionSpace() != coarseBasis->functionSpace())
            {
              // This will cause an error in BasisReconciliation; can we find equivalent continuous basis?
              // (necessary when using a non-conforming coarse mesh, though not at present an issue for fields, as here)
              coarseBasis = coarseBasis->continuousBasis();
              TEUCHOS_TEST_FOR_EXCEPTION(fineBasis->functionSpace() != coarseBasis->functionSpace(), std::invalid_argument, "Even after getting continuous basis for coarseBasis, fine and coarse function spaces disagree");
            }
            SubBasisReconciliationWeights weights;
            
            {
              // ancestralPermutation goes from canonical to cell's side's ancestor's ordering:
              unsigned ancestralCellPermutation = ancestor->subcellPermutation(sideDim, coarseSideOrdinal);
              // constrainingPermutation goes from canonical to the constraining side's ordering
              unsigned constrainingCellPermutation = fineCell->subcellPermutation(sideDim, sideOrdinal); // subcell permutation as seen from the perspective of the constraining cell's side
              
              // ancestralPermutationInverse goes from ancestral view to canonical
              unsigned ancestralPermutationInverse = CamelliaCellTools::permutationInverse(fineBasis->domainTopology(), ancestralCellPermutation);
              unsigned ancestralToConstrainedPermutation = CamelliaCellTools::permutationComposition(fineBasis->domainTopology(), ancestralPermutationInverse, constrainingCellPermutation);
              
              unsigned fineSubcellOrdinalInFineDomain = 0; // whole fine basis
              unsigned coarseSubcellOrdinalInCoarseDomain = 0; // whole coarse basis
              unsigned fineDomainOrdinalInRefinementLeaf = sideOrdinal;
              unsigned coarseDomainOrdinalInCoarseCellTopo = coarseSideOrdinal;
              
              weights = BasisReconciliation::computeConstrainedWeights(sideDim, fineBasis, fineSubcellOrdinalInFineDomain, refBranch,
                                                                       fineDomainOrdinalInRefinementLeaf,
                                                                       ancestor->topology(), sideDim,
                                                                       coarseBasis, coarseSubcellOrdinalInCoarseDomain,
                                                                       coarseDomainOrdinalInCoarseCellTopo,
                                                                       ancestralToConstrainedPermutation);
            }
            
            
            vector<GlobalIndexType> coarseDofIndices(weights.coarseOrdinals.size());
            int i = 0;
            for (int coarseOrdinal : weights.coarseOrdinals)
            {
              coarseDofIndices[i++] = coarseTrialOrdering->getDofIndex(trialID, coarseOrdinal, coarseSideOrdinal);
            }

            basisMap.push_back(SubBasisDofMapper::subBasisDofMapper(weights.fineOrdinals, coarseDofIndices, weights.weights));
          }

          sideMaps[sideOrdinal][trialID] = basisMap;
        }
      }
    }

    int coarseDofCount = coarseTrialOrdering->totalDofs();
    set<GlobalIndexType> allCoarseDofs; // include these even when not mapped (which can happen with static condensation) to guarantee that the dof mapper interprets coarse cell data correctly...
    for (int coarseOrdinal=0; coarseOrdinal<coarseDofCount; coarseOrdinal++)
    {
      allCoarseDofs.insert(coarseOrdinal);
    }
    
    // set up side maps from fine to coarse cell
    map<int,int> fineSideToCoarseSide;
    int fineSideCount = _fineMesh->getElementType(fineCellID)->cellTopoPtr->getSideCount();
    for (int fineSideOrdinal=0; fineSideOrdinal < fineSideCount; fineSideOrdinal++)
    {
      // is there a corresponding coarse side ordinal?
      RefinementBranch refBranch;
      GlobalIndexType ancestralCellID = fineCellID;
      while (ancestralCellID != coarseCellID) {
        CellPtr ancestralCell = _fineMesh->getTopology()->getCell(ancestralCellID);
        int childOrdinal = ancestralCell->getParent()->findChildOrdinal(ancestralCellID);
        TEUCHOS_TEST_FOR_EXCEPTION(childOrdinal==-1, std::invalid_argument, "internal error: cell not found in its own parent");
        refBranch.insert(refBranch.begin(),{ancestralCell->getParent()->refinementPattern().get(), childOrdinal});
        ancestralCellID = ancestralCell->getParent()->cellIndex();
      }
      int coarseSideOrdinal = RefinementPattern::mapSideOrdinalFromLeafToAncestor(fineSideOrdinal, refBranch);
      fineSideToCoarseSide[fineSideOrdinal] = coarseSideOrdinal; // -1 if there isn't a corresponding coarse side ordinal
    }
    
    for (int trialID : trialIDs)
    {
      VarPtr trialVar = vf->trial(trialID);

      const vector<int>& fineSidesForVarID = fineTrialOrdering->getSidesForVarID(trialID);
      if (fineTrialOrdering->isVolumeVar(trialID))   // field variable
      {
        if ((condensedDofInterpreterCoarse == NULL) || (condensedDofInterpreterCoarse->varDofsAreCondensible(coarseCellID, trialID, 0, fineTrialOrdering)))
        {
          vector<int> dofIndices = coarseTrialOrdering->getDofIndices(trialID);
          fittableGlobalDofOrdinalsInVolume.insert(dofIndices.begin(),dofIndices.end());
        }
      }
      else
      {
        for (const int & fineSideOrdinal : fineSidesForVarID)
        {
          int coarseSideOrdinal = fineSideToCoarseSide[fineSideOrdinal];
          if (coarseSideOrdinal != -1)
          {
            const vector<int> & dofIndices = coarseTrialOrdering->getDofIndices(trialID,coarseSideOrdinal);
            fittableGlobalDofOrdinalsOnSides[fineSideOrdinal].insert(dofIndices.begin(),dofIndices.end());
          }
        }
      }
    }

    dofMapper = Teuchos::rcp( new LocalDofMapper(fineTrialOrdering, volumeMaps, fittableGlobalDofOrdinalsInVolume,
                                                 sideMaps, fittableGlobalDofOrdinalsOnSides, allCoarseDofs) );
    
//    dofMapper->printMappingReport();
    if (cellProlongationCanMatchPatterns)
      _localCoefficientMap[key] = dofMapper;
  }

  // now, correct side parities in dofMapper if the ref space situation differs from the physical space one.
  Camellia::FieldContainer<double> coarseCellSideParities = _coarseMesh->globalDofAssignment()->cellSideParitiesForCell(coarseCellID);
  Camellia::FieldContainer<double> fineCellSideParities = _fineMesh->globalDofAssignment()->cellSideParitiesForCell(fineCellID);
//  cout << "fineCell parities:\n" << fineCellSideParities;
  set<unsigned> fineSidesToCorrect;
  for (unsigned fineSideOrdinal=0; fineSideOrdinal<fineSideCount; fineSideOrdinal++)
  {
    unsigned coarseSideOrdinal = ancestralSideOrdinals[fineSideOrdinal];
    if (coarseSideOrdinal != INVALID_CELLTOPOLOGY_SIDE_ORDINAL)   // ancestor shares side
    {
      double coarseParity = coarseCellSideParities(0,coarseSideOrdinal);
      double fineParity = fineCellSideParities(0,fineSideOrdinal);
      if (coarseParity != fineParity)
      {
        fineSidesToCorrect.insert(fineSideOrdinal);
//        cout << "fine side to correct on cell " << fineCellID << ": " << fineSideOrdinal << endl;
      }
    }
    else
    {
      // when we have done a map from field to trace, no parities have been taken into account; the termTraced will involve
      // the outward facing normal, but not the potential negation of this to make neighbors agree.  Therefore, we need to "correct"
      // any interior fine side whose parity is -1.

      double fineParity = fineCellSideParities(0,fineSideOrdinal);
      if (fineParity < 0)
      {
        fineSidesToCorrect.insert(fineSideOrdinal);
        //        cout << "fine side to correct on cell " << fineCellID << ": " << fineSideOrdinal << endl;
      }
    }
  }

  if (fineSidesToCorrect.size() > 0)
  {
    // copy before changing dofMapper:
    dofMapper = Teuchos::rcp( new LocalDofMapper(*dofMapper.get()) );
    set<int> fluxIDs;
    VarFactoryPtr vf = _fineMesh->bilinearForm()->varFactory();
    vector<VarPtr> fluxVars = vf->fluxVars();
    for (vector<VarPtr>::iterator fluxIt = fluxVars.begin(); fluxIt != fluxVars.end(); fluxIt++)
    {
      fluxIDs.insert((*fluxIt)->ID());
    }
    dofMapper->reverseParity(fluxIDs, fineSidesToCorrect);
  }

  return dofMapper;
}

int GMGOperator::getOperatorLevel() const
{
  if (_coarseOperator != Teuchos::null)
    return _coarseOperator->getOperatorLevel() + 1;
  else
    return 0;
}

int GMGOperator::getSmootherApplicationCount() const
{
  return _smootherApplicationCount;
}

int GMGOperator::getSmootherOverlap() const
{
  return _smootherOverlap;
}

GMGOperator::SmootherChoice GMGOperator::getSmootherType()
{
  return _smootherType;
}

Teuchos::RCP< Tpetra::Vector<double> > GMGOperator::getSmootherWeightVector()
{
  return _smootherDiagonalWeight;
}

void GMGOperator::apply(const TVector<Scalar> &tX, TVector<Scalar> &tY, Teuchos::ETransp mode, Scalar alpha, Scalar beta) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(alpha != 1, std::invalid_argument, "This implementation assumes alpha = 1");
  TEUCHOS_TEST_FOR_EXCEPTION(beta != 0, std::invalid_argument, "This implementation assumes beta = 0");
  int err = ApplyInverse(tX,tY);
  TEUCHOS_TEST_FOR_EXCEPTION(err !=0, std::invalid_argument, "ApplyInverse() returned non-zero error code");
}

int GMGOperator::ApplyInverse(const TVector<Scalar> &X_in, TVector<Scalar> &Y) const
{
  narrate("ApplyInverse");
  //  cout << "GMGOperator::ApplyInverse.\n";
  int rank = Teuchos::GlobalMPISession::getRank();
  bool printVerboseOutput = (rank==0) && _debugMode;
  
  auto eComm = MPIWrapper::EpetraComm(Comm());
  Epetra_Time timer(*eComm);
  
  if (_debugMode)
  {
    if (printVerboseOutput) std::cout << "X_in:\n";
    Teuchos::RCP<Teuchos::FancyOStream> fos = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
    X_in.describe(*fos, Teuchos::VERB_EXTREME);
  }
  
  TVector<Scalar> res(X_in, Teuchos::Copy); // Res: residual.  Starting with Y = 0, then this is just the RHS
  TVector<Scalar> f(X_in, Teuchos::Copy);   // f: the RHS.  Don't change this.
  
  // initialize Y (important to do this only after X_in has been copied--X and Y can be in the same location)
  Y.putScalar(0.0);
  TVector<Scalar> A_Y(Y.getMap(), Y.getNumVectors());
  
  if ((_multigridStrategy == FULL_MULTIGRID_V) || (_multigridStrategy == FULL_MULTIGRID_W))
  {
    // full multigrid takes the coarse operator applied to the RHS as its initial guess
    TVector<Scalar> Y2(Y.getMap(), Y.getNumVectors());
    this->ApplyInverseCoarseOperator(res, Y2);
    Y.update(1.0, Y2, 1.0);
    // recompute residual:
    res = f;
    computeResidual(Y,res,A_Y);
  }
  
  int numApplications;
  if ((_multigridStrategy == W_CYCLE) || (_multigridStrategy == FULL_MULTIGRID_W))
  {
    numApplications = 2;
  }
  else if (_multigridStrategy != SMOOTHER_ONLY)
  {
    numApplications = 1;
  }
  else
  {
    numApplications = 0;
  }
  
  if (_smootherType != NONE)
  {
    TVector<Scalar> B1_res(Y.getMap(), Y.getNumVectors()); // B1_res: the smoother applied to res.
    for (int i=0; i<_smootherApplicationCount; i++)
    {
      // if we have a smoother S, set Y = S^-1 f =: B1 * f
      ApplySmoother(res, B1_res, true); // true: apply weight on left
      Y.update(1.0, B1_res, 1.0);
      if (_debugMode)
      {
        if (printVerboseOutput) cout << "B1 * res:\n";
        Teuchos::RCP<Teuchos::FancyOStream> fos = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
        B1_res.describe(*fos, Teuchos::VERB_EXTREME);
      }
      
      if (_multigridStrategy == TWO_LEVEL)
      {
        if (_useSchwarzDiagonalWeight && (_smootherDiagonalWeight != Teuchos::null))
        {
          // then we need to average the left-application we already did with an application on the right
          ApplySmoother(res, B1_res, false); // false: apply weight on right
          Y.update(0.5, B1_res, 0.5); // 0.5: average
        }
        break; // don't apply multiple times if doing two-level (no point...)
      }
      else
      {
        // compute a new residual: res := f - A*Y = res - A*Y
        Tpetra::deep_copy(res, f);
        computeResidual(Y,res,A_Y);
      }
    }
  }
  
  for (int applicationOrdinal = 0; applicationOrdinal < numApplications; applicationOrdinal++)
  {
    TVector<Scalar> Y2(Y.getMap(), Y.getNumVectors());
    this->ApplyInverseCoarseOperator(res, Y2);
    Y.update(1.0, Y2, 1.0);
    
    if ((_smootherType != NONE) && (_multigridStrategy != TWO_LEVEL))
    {
      TVector<Scalar> B1_res(Y.getMap(), Y.getNumVectors());
      
      for (int i=0; i<_smootherApplicationCount; i++)
      {
        // compute Y + B1 * (f - A*y)
        Tpetra::deep_copy(res, f);
        if (_debugMode)
        {
          if (printVerboseOutput) cout << "res after res = f:\n";
          Teuchos::RCP<Teuchos::FancyOStream> fos = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
          res.describe(*fos, Teuchos::VERB_EXTREME);
        }
        computeResidual(Y, res, A_Y);
        
        if (((_useSchwarzDiagonalWeight && (_smootherDiagonalWeight != Teuchos::null)))
            && (numApplications > 1) && (applicationOrdinal < numApplications-1))
        {
          // then, to maintain symmetry, apply smoother twice, once with left weighting, and once with right:
          ApplySmoother(res, B1_res, true);
          Y.update(1.0, B1_res, 1.0);
          Tpetra::deep_copy(res,f);
          computeResidual(Y, res, A_Y);
          ApplySmoother(res, B1_res, false);
          Y.update(1.0, B1_res, 1.0);
        }
        else
        {
          // just apply once, on the right
          ApplySmoother(res, B1_res, false);
          Y.update(1.0, B1_res, 1.0);
        }
      }
    }
    
    if (_debugMode)
    {
      if (printVerboseOutput) cout << "Y:\n";
      Teuchos::RCP<Teuchos::FancyOStream> fos = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
      Y.describe(*fos, Teuchos::VERB_EXTREME);
    }
    if (applicationOrdinal < numApplications-1)
    {
      // another application will follow, so let's recompute the residual
      res = f;
      computeResidual(Y, res, A_Y);
    }
  }
  
//  {
//    // TESTING SOMETHING:
//    if (_isFinest)
//    {
//      res = f;
//      computeResidual(Y, res, A_Y);
//      Epetra_MultiVector B1_res(Y.Map(), Y.NumVectors());
//      ApplySmoother(res, B1_res, true); // true: apply weight on left
//      double normValues[Y.NumVectors()];
//      res.Dot(B1_res, &normValues[0]);
//      
//      if (rank == 0) cout << "ApplyInverse: residual on finest GMGOperator, measured in norm induced by Schwarz operator: " << normValues[0] << endl;
//    }
//  }
  
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
  auto YHostView = Y.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (GlobalIndexTypeToCast fineRowIndex=firstFineConstraintRowIndex; fineRowIndex<Y.getGlobalLength(); fineRowIndex++)
  {
    int LID = Y.getMap()->getLocalElement(fineRowIndex);
    if (LID != -1)
    {
      YHostView(LID, 0) = 0.0;
    }
  }
  Kokkos::deep_copy(Y.getLocalViewDevice(Tpetra::Access::ReadWrite), YHostView);
  
  return 0;
}


int GMGOperator::ApplyInverseCoarseOperator(const TVector<Scalar> &res, TVector<Scalar> &Y) const
{
  int rank = Teuchos::GlobalMPISession::getRank();
  bool printVerboseOutput = (rank==0) && _debugMode;

  Teuchos::RCP<Epetra_FEVector> coarseRHSVector = _coarseSolution->getRHSVector();
  auto tCoarseRHSVector = E_to_T::vector(coarseRHSVector);
  
//  if (rank == 0)
//  {
//    std::cout << "coarseRHSVector->GlobalLength(): " << coarseRHSVector->GlobalLength() << std::endl;
//    std::cout << "prolongationColCount(): " << prolongationColCount() << std::endl;
//  }
  if (coarseRHSVector->GlobalLength() != prolongationColCount())
  {
    // TODO: add support for coarseRHSVector that may have lagrange/zmc constraints applied even though fine solution neglects these...
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarseRHSVector->GlobalLength() != _P->NumGlobalCols()");
  }
  
  auto eComm = MPIWrapper::EpetraComm(Comm());
  Epetra_Time timer(*eComm);
  
  narrate("multiply _P * coarseRHSVector");
  if (printVerboseOutput) cout << "calling _P->Multiply(true, res, *coarseRHSVector);\n";
  if (_debugMode)
  {
    if (rank==0) cout << "res:\n";
    Teuchos::RCP<Teuchos::FancyOStream> fos = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
    res.describe(*fos, Teuchos::VERB_EXTREME);
  }
  _P->apply(res, *tCoarseRHSVector, Teuchos::TRANS, 1.0, 0.0);
//  _P->Multiply(true, res, coarseRHSVector);
  if (printVerboseOutput) cout << "finished _P->Multiply(true, X, *coarseRHSVector);\n";
  if (_debugMode)
  {
    if (rank==0) cout << "coarseRHSVector:\n";
    Teuchos::RCP<Teuchos::FancyOStream> fos = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
    tCoarseRHSVector->describe(*fos, Teuchos::VERB_EXTREME);
  }
  _timeMapFineToCoarse += timer.ElapsedTime();
  
  Teuchos::RCP<Epetra_FEVector> coarseLHSVector;
  
  timer.ResetStartTime();
  if (_coarseSolver == Teuchos::null)
  {
    // then we must be at a finer level than the coarsest, and the appropriate thing is to simply apply
    // our _coarseOperator
    TEUCHOS_TEST_FOR_EXCEPTION(_coarseOperator == Teuchos::null, std::invalid_argument, "GMGOperator internal error: _coarseOperator and _coarseSolver are both null");
    auto tLHSVector = E_to_T::vector(_coarseSolution->getLHSVector());
    _coarseOperator->ApplyInverse(*tCoarseRHSVector, *tLHSVector);
    auto eResult = T_to_E::vector(tLHSVector);
    _coarseSolution->getLHSVector()->Update(1.0, *eResult, 0.0);
  }
  else if (!_haveSolvedOnCoarseMesh)
  {
    if (printVerboseOutput) cout << "solving on coarse mesh\n";
    narrate("_coarseSolution->solveWithPrepopulatedStiffnessAndLoad(_coarseSolver, false)");
    auto eCoarseRHSVector = T_to_E::vector(tCoarseRHSVector);
    coarseRHSVector->Update(1.0, *eCoarseRHSVector, 0.0);
    _coarseSolution->setProblem(_coarseSolver);
    _coarseSolution->solveWithPrepopulatedStiffnessAndLoad(_coarseSolver, false);
    if (printVerboseOutput) cout << "finished solving on coarse mesh\n";
    _haveSolvedOnCoarseMesh = true;
  }
  else
  {
    if (printVerboseOutput) cout << "re-solving on coarse mesh\n";
    narrate("_coarseSolution->solveWithPrepopulatedStiffnessAndLoad(_coarseSolver, true)");
    auto eCoarseRHSVector = T_to_E::vector(tCoarseRHSVector);
    coarseRHSVector->Update(1.0, *eCoarseRHSVector, 0.0);
    _coarseSolver->setRHS(coarseRHSVector);
    _coarseSolution->solveWithPrepopulatedStiffnessAndLoad(_coarseSolver, true); // call resolve() instead of solve() -- reuse factorization
    if (printVerboseOutput) cout << "finished re-solving on coarse mesh\n";
  }
  _timeCoarseSolve += timer.ElapsedTime();
  
  timer.ResetStartTime();
  if (printVerboseOutput) cout << "calling _coarseSolution->getLHSVector()\n";
  coarseLHSVector = _coarseSolution->getLHSVector();
  auto tCoarseLHSVector = E_to_T::vector(_coarseSolution->getLHSVector());
  
  if (_debugMode)
  {
    if (rank==0) cout << "coarseLHSVector (Epetra):\n";
    cout << *coarseLHSVector;
    if (rank==0) cout << "coarseLHSVector (Tpetra):\n";
    Teuchos::RCP<Teuchos::FancyOStream> fos = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
    tCoarseLHSVector->describe(*fos, Teuchos::VERB_EXTREME);
  }
  
  if (printVerboseOutput) cout << "finished _coarseSolution->getLHSVector()\n";
  if (printVerboseOutput) cout << "calling _P->Multiply(false, *coarseLHSVector, Y)\n";
  narrate("multiply _P * coarseLHSVector");

  _P->apply(*tCoarseLHSVector, Y, Teuchos::NO_TRANS, 1.0, 0.0);
//  _P->Multiply(false, tCoarseLHSVector, Y);
  if (printVerboseOutput) cout << "finished _P->Multiply(false, *coarseLHSVector, Y)\n";
  _timeMapCoarseToFine += timer.ElapsedTime();
  
  if (_debugMode)
  {
    if (rank==0) cout << "_P * coarseLHSVector:\n";
    Teuchos::RCP<Teuchos::FancyOStream> fos = Teuchos::getFancyOStream(Teuchos::rcpFromRef(std::cout));
    Y.describe(*fos, Teuchos::VERB_EXTREME);
  }
  return 0;
}

int GMGOperator::ApplySmoother(const TVector<Scalar> &res, TVector<Scalar> &Y, bool weightOnLeft) const
{
  narrate("ApplySmoother()");
  auto eComm = MPIWrapper::EpetraComm(Comm());
  Epetra_Time timer(*eComm);
  
  if ((_smootherDiagonalWeight == Teuchos::null) || !_useSchwarzDiagonalWeight)
  {
    _smoother->apply(res, Y);
  }
  else
  {
    TVector<double> temp(res.getMap(),res.getNumVectors());
    if (!weightOnLeft)
      temp.elementWiseMultiply(1.0, *_smootherDiagonalWeight, res, 0.0);
    else
      temp = res;
        
    _smoother->apply(temp, Y);
    
    if (weightOnLeft)
    {
      temp = Y;
      Y.elementWiseMultiply(1.0, *_smootherDiagonalWeight, temp, 0.0);
    }
  }
  Y.scale(_smootherWeight);
    
  _timeApplySmoother += timer.ElapsedTime();
  
  return 0;
}

double GMGOperator::NormInf() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported method.");
}

const char * GMGOperator::Label() const
{
  return "Camellia Geometric Multi-Grid Preconditioner";
}

int GMGOperator::SetUseTranspose(bool UseTranspose)
{
  return -1; // not supported for now.  (wouldn't be hard, but I don't see the point.)
}

bool GMGOperator::UseTranspose() const
{
  return false; // not supported for now.  (wouldn't be hard, but I don't see the point.)
}

//! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
bool GMGOperator::HasNormInf() const
{
  return false;
}

//! Returns a pointer to the communicator associated with this operator.
const Teuchos_CommPtr GMGOperator::Comm() const
{
  return _fineMesh->Comm();
}

//! Returns the Map object associated with the domain of this operator.
TMapConstPtr GMGOperator::getDomainMap() const
{
  return _finePartitionMap;
}

//! Returns the Map object associated with the range of this operator.
TMapConstPtr GMGOperator::getRangeMap() const
{
  return _finePartitionMap;
}

int GMGOperator::prolongationColCount() const
{
  return static_cast<int>(_P->getGlobalNumCols());
}

int GMGOperator::prolongationRowCount() const
{
  return static_cast<int>(_P->getGlobalNumRows());
}

TimeStatistics GMGOperator::getStatistics(double timeValue) const
{
  TimeStatistics stats;
  
  auto comm = Comm();
  stats.sum  = MPIWrapper::sum(comm, timeValue);
  stats.mean = stats.sum / comm->getSize();
  stats.min = MPIWrapper::min(comm, timeValue);
  stats.max = MPIWrapper::max(comm, timeValue);

  return stats;
}

void GMGOperator::reportTimings(StatisticChoice whichStat) const
{
  reportTimings(whichStat,false);
}

void GMGOperator::reportTimings(StatisticChoice whichStat, bool sumAllOperators) const
{
  //   mutable double _timeMapFineToCoarse, _timeMapCoarseToFine, _timeCoarseImport, _timeConstruction, _timeCoarseSolve;  // totals over the life of the object
  int rank = Teuchos::GlobalMPISession::getRank();
  
  map<string, double> reportValues = sumAllOperators ? timingReportSumOfOperators() : timingReport() ;
  
  if (rank == 0)
  {
    if (sumAllOperators)
      cout << "Sum of all grid levels, ";
    switch (whichStat)
    {
      case MIN:
        cout << "Timing MINIMUM values:\n";
        break;
      case MAX:
        cout << "Timing MAXIMUM values:\n";
        break;
      case MEAN:
        cout << "Timing mean values:\n";
        break;
      case SUM:
        cout << "SUM of timing values:\n";
        break;
      case ALL:
        break;
    }
  }
  for (map<string,double>::iterator reportIt = reportValues.begin(); reportIt != reportValues.end(); reportIt++)
  {
    TimeStatistics stats = getStatistics(reportIt->second);
    if (rank==0)
    {
      cout << setprecision(2);
      cout << std::scientific;
      cout << setw(35) << reportIt->first << ": ";
      switch (whichStat) {
        case ALL:
          cout << endl;
          cout <<  "mean = " << stats.mean << " seconds\n";
          cout << "max =  " << stats.max << " seconds\n";
          cout << "min =  " << stats.min << " seconds\n";
          cout << "sum =  " << stats.sum << " seconds\n";
          break;
        case MIN:
          cout << stats.min << " seconds\n";
          break;
        case MAX:
          cout << stats.max << " seconds\n";
          break;
        case MEAN:
          cout << stats.mean << " seconds\n";
          break;
        case SUM:
          cout << stats.sum << " seconds\n";
          break;
      }
    }
  }
}

void GMGOperator::reportTimingsSumOfOperators(StatisticChoice whichStat) const
{
  reportTimings(whichStat,true);
}

std::map<string, double> GMGOperator::timingReport() const
{
  map<string, double> reportValues;
  reportValues["apply fine stiffness"] = _timeApplyFineStiffness;
  reportValues["apply smoother"] = _timeApplySmoother;
  reportValues["total construction time"] = _timeConstruction;
  reportValues["construct prolongation operator"] = _timeProlongationOperatorConstruction;
  reportValues["construct local coefficient maps"] = _timeLocalCoefficientMapConstruction;
  reportValues["coarse import"] = _timeCoarseImport;
  reportValues["coarse solve"] = _timeCoarseSolve;
  reportValues["map coarse to fine"] = _timeMapCoarseToFine;
  reportValues["map fine to coarse"] = _timeMapFineToCoarse;
  reportValues["compute coarse stiffness matrix"] = _timeComputeCoarseStiffnessMatrix;
  reportValues["set up smoother"] = _timeSetUpSmoother;
  reportValues["update coarse operator"] = _timeUpdateCoarseOperator;
  return reportValues;
}

std::map<string, double> GMGOperator::timingReportSumOfOperators() const
{
  map<string, double> reportValues = timingReport();
  Teuchos::RCP<GMGOperator> coarseOperator = _coarseOperator;
  while (coarseOperator != Teuchos::null)
  {
    map<string,double> coarseReportValues = coarseOperator->timingReport();
    for (pair<string,double> entry : coarseReportValues)
    {
      // the "coarse solve" entry we should take from the coarsest operator, and not sum
      if (entry.first != "coarse solve")
      {
        reportValues[entry.first] += entry.second;
      }
      else
      {
        reportValues[entry.first] = entry.second;
      }
    }
    coarseOperator = coarseOperator->getCoarseOperator();
  }
  
  reportValues.erase("update coarse operator"); // this one's costs are accounted for in other entries from the coarse operators (set up smoother and compute coarse stiffness)
  return reportValues;
}

void GMGOperator::setChebyshevDegree(int chebyshevDegree)
{
  _chebyshevDegree = chebyshevDegree;
}

void GMGOperator::setChebyshevRatio(double chebyshevRatio)
{
  _chebyshevEigenAutocomputeRatio = chebyshevRatio;
}

void GMGOperator::setCoarseOperator(Teuchos::RCP<GMGOperator> coarseOperator)
{
  _coarseOperator = coarseOperator;
  _coarseOperator->setIsFinest(false);
}

void GMGOperator::setCoarseSolver(SolverPtr coarseSolver)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_coarseOperator != Teuchos::null, std::invalid_argument, "coarseSolver may not be set when _coarseOperator is set");
  _coarseSolver = coarseSolver;
}

void GMGOperator::setDebugMode(bool value)
{
  _debugMode = value;
}

void GMGOperator::setFineStiffnessMatrix(TMatrixPtr<Scalar> fineStiffness)
{
  _fineStiffnessMatrix = fineStiffness;
  _eFineStiffnessMatrix = T_to_E::matrix(_fineStiffnessMatrix);
  if (_multigridStrategy != SMOOTHER_ONLY)
  {
    computeCoarseStiffnessMatrix(fineStiffness);
  }
  setUpSmoother(fineStiffness);
  
  if (_coarseOperator != Teuchos::null)
  {
    auto eComm = MPIWrapper::EpetraComm(Comm());
    Epetra_Time coarseOperatorTimer(*eComm);
    // then our changed coarse matrix is coarse operator's fine matrix, and we need to ask coarse operator recompute *its* coarse matrix and smoother
    _coarseOperator->setFineStiffnessMatrix(getCoarseStiffnessMatrix());
    _timeUpdateCoarseOperator += coarseOperatorTimer.ElapsedTime();
  }
}

void GMGOperator::setFillRatio(double fillRatio)
{
  _fillRatio = fillRatio;
//  cout << "fill ratio set to " << fillRatio << endl;
}

void GMGOperator::setFunctionExporter(Teuchos::RCP<HDF5Exporter> exporter, FunctionPtr function, string functionName)
{
  _functionExporter = exporter;
  _functionToExport = function;
  _functionToExportName = functionName;
  _exportTimeStep = 0;
}

void GMGOperator::setIsFinest(bool value)
{
  _isFinest = value;
}

void GMGOperator::setLevelOfFill(int fillLevel)
{
  _levelOfFill = fillLevel;
//  cout << "level of fill set to " << fillLevel << endl;
}

void GMGOperator::setMultigridStrategy(MultigridStrategy choice)
{
  _multigridStrategy = choice;
}

void GMGOperator::setSchwarzFactorizationType(FactorType choice)
{
  _schwarzBlockFactorizationType = choice;
}

void GMGOperator::setSmootherApplicationCount(int count)
{
  _smootherApplicationCount = count;
}

void GMGOperator::setSmootherApplicationType(SmootherApplicationType applicationType)
{
  // smoother application type is deprecated
  // TODO: eliminate it (in favor of MultigridStrategy)
  _smootherApplicationType = applicationType;
  if (_smootherApplicationType == ADDITIVE)
    _multigridStrategy = TWO_LEVEL;
  else
    _multigridStrategy = V_CYCLE;
}

void GMGOperator::setSmootherOverlap(int overlap)
{
  _smootherOverlap = overlap;
}

void GMGOperator::setSmootherType(GMGOperator::SmootherChoice smootherType)
{
  _smootherType = smootherType;
}

double GMGOperator::getSmootherWeight()
{
  return _smootherWeight;
}

void GMGOperator::setSmootherWeight(double weight)
{
  _smootherWeight = weight;
  if (_smootherWeight != -1.0)
  {
    // prevent overwriting this value later
    cout << "Turning OFF _useSchwarzScalingWeight, because smootherWeight value is provided.\n";
    setUseSchwarzScalingWeight(false);
  }
}

void GMGOperator::setUpSmoother(TMatrixPtr<double> fineStiffnessMatrix)
{
  narrate("setUpSmoother()");
  auto eComm = MPIWrapper::EpetraComm(Comm());
  Epetra_Time smootherSetupTimer(*eComm);
  
  using RowMatrix = Tpetra::RowMatrix<Scalar>;
  
  SmootherChoice choice = _smootherType;

  using BlockRelaxation = Ifpack2::BlockRelaxation< RowMatrix >;
  
  Teuchos::ParameterList List;
  
  Teuchos::RCP< Tpetra::Operator<Scalar> > smoother;
  bool useILUTFallbackForDegenerateLocalCase = false;
  bool bypassCustomOuterPartitioningForDegenerateCase = false;
  std::string schwarzInnerPrecName = "Block Relaxation";

  int rank = Comm()->getRank();
  
  switch (choice)
  {
  case NONE:
    _smoother = Teuchos::null;
    return;
  case POINT_JACOBI:
  {
    List.set("relaxation: type", "Jacobi");
    List.set("relaxation: sweeps", 1);
    smoother = Ifpack2::Factory::create< RowMatrix >("RELAXATION", fineStiffnessMatrix);
  }
  break;
  case POINT_SYMMETRIC_GAUSS_SEIDEL:
  {
    List.set("relaxation: type", "Symmetric Gauss-Seidel");
    smoother = Teuchos::rcp(new BlockRelaxation(fineStiffnessMatrix) );
  }
  break;
  case BLOCK_JACOBI:
  {
    smoother = Teuchos::rcp(new BlockRelaxation(fineStiffnessMatrix) );
    Teuchos::ParameterList List;
    int overlapBlocks = _smootherOverlap;
    int sweeps = 2;
    int localParts = 4;

    List.set("block relaxation: type", "Jacobi");
    List.set("block relaxation: sweeps", sweeps);
    List.set("Amesos2: Solver", "KLU2");

    List.set("partitioner: overlap level", overlapBlocks);
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
  case BLOCK_SYMMETRIC_GAUSS_SEIDEL:
  {
    smoother = Teuchos::rcp(new BlockRelaxation(fineStiffnessMatrix) );
    Teuchos::ParameterList List;
    int overlapBlocks = _smootherOverlap;
    int sweeps = 2;
    int localParts = 4;

    List.set("relaxation: type", "symmetric Gauss-Seidel");
    List.set("relaxation: sweeps", sweeps);
    List.set("amesos: solver type", "Amesos_Klu"); // default

    List.set("partitioner: overlap level", overlapBlocks);
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
  {
    smoother = Ifpack2::Factory::create<RowMatrix>("Schwarz", fineStiffnessMatrix);

    // Match the legacy Epetra Ifpack_AdditiveSchwarz setup more closely.
    // Overlapped additive Schwarz corrections should be accumulated rather
    // than inserted/overwritten, which is also important for the symmetry
    // checks in the GMG matrix-representation tests.
    List.set("schwarz: combine mode", "Add");

    // Ifpack2's non-overlapping MPI path is fragile for the very small,
    // degenerate distributions used by some GMGSolver tests on many ranks.
    // Use the ordinary overlapping Schwarz path in parallel when Camellia's
    // requested overlap is zero.
    const int overlapLevel = (_smootherOverlap == 0 && Comm()->getSize() > 1) ? 1 : _smootherOverlap;
    List.set("schwarz: overlap level", overlapLevel);

    List.set("subdomain solver name", "BLOCK_RELAXATION");
    Teuchos::ParameterList &subdomainSolverParameters = List.sublist("subdomain solver parameters");
    subdomainSolverParameters.set("relaxation: type", "Jacobi");
    subdomainSolverParameters.set("relaxation: sweeps", 1);
    subdomainSolverParameters.set("relaxation: container", "Dense");
    subdomainSolverParameters.set("partitioner: type", "linear");
    subdomainSolverParameters.set("partitioner: local parts", 1);
  }
  break;
  case CAMELLIA_ADDITIVE_SCHWARZ:
  {
    using LocalOrdinal = typename TMatrix<Scalar>::local_ordinal_type;
    using GlobalOrdinal = typename TMatrix<Scalar>::global_ordinal_type;
    using Node = typename TMatrix<Scalar>::node_type;
    using TpetraSchwarz = Camellia::TpetraCamelliaAdditiveSchwarz<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    smoother = Teuchos::rcp(new TpetraSchwarz(fineStiffnessMatrix,
                                              _smootherOverlap,
                                              _fineMesh,
                                              _fineDofInterpreter,
                                              _hierarchicalNeighborsForSchwarz,
                                              _dimensionForSchwarzNeighborRelationship));
    schwarzInnerPrecName = "Tpetra Camellia additive Schwarz";
  }
  break;
    case CHEBYSHEV:
    {
      smoother = Ifpack2::Factory::create<RowMatrix>("CHEBYSHEV", fineStiffnessMatrix);
      /*
       EigRatio_             = List.get("chebyshev: ratio eigenvalue", EigRatio_);
       LambdaMin_            = List.get("chebyshev: min eigenvalue", LambdaMin_);
       LambdaMax_            = List.get("chebyshev: max eigenvalue", LambdaMax_);
       PolyDegree_           = List.get("chebyshev: degree",PolyDegree_);
       EigMaxIters_          = List.get("chebyshev: eigenvalue max iterations",EigMaxIters_);
       */
//      List.set("chebyshev: eigenvalue autocompute ratio",_chebyshevEigenAutocomputeRatio); //Ifpack2?
      List.set("chebyshev: ratio eigenvalue", _chebyshevEigenAutocomputeRatio);
      List.set("chebyshev: eigenvalue max iterations",_chebyshevMaxEigenIterations);
      List.set("chebyshev: degree",_chebyshevDegree);
    }
      break;

  default:
    break;
  }

  auto ifpack2Smoother = Teuchos::rcp_dynamic_cast< Ifpack2::Preconditioner<Scalar> >(smoother);
  if (!ifpack2Smoother.is_null())
  {
    ifpack2Smoother->setParameters(List);
  }

//  if (choice == CAMELLIA_ADDITIVE_SCHWARZ)
//  {
//    const int myRank = Comm()->getRank();
//    if (myRank == 0)
//    {
//      std::cout << "GMGOperator::setUpSmoother(): using Tpetra-native Camellia additive Schwarz smoother\n";
//    }
//  }
  
//  int rank = Teuchos::GlobalMPISession::getRank();
//  if (rank == 0) {
//    cout << "Smoother info:\n";
//    cout << *smoother;
//  }

  if (!ifpack2Smoother.is_null())
  {
    ifpack2Smoother->initialize();
    ifpack2Smoother->compute();
  }

  if ((choice == CAMELLIA_ADDITIVE_SCHWARZ) && _useSchwarzDiagonalWeight)
  {
    using LocalOrdinal = typename TMatrix<Scalar>::local_ordinal_type;
    using GlobalOrdinal = typename TMatrix<Scalar>::global_ordinal_type;
    using Node = typename TMatrix<Scalar>::node_type;
    using TpetraSchwarz = Camellia::TpetraCamelliaAdditiveSchwarz<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

    auto tpetraSmoother = Teuchos::rcp_dynamic_cast<TpetraSchwarz>(smoother);
    TEUCHOS_TEST_FOR_EXCEPTION(tpetraSmoother.is_null(), std::invalid_argument,
                               "GMGOperator::setUpSmoother(): CAMELLIA_ADDITIVE_SCHWARZ expected a TpetraCamelliaAdditiveSchwarz smoother");

    const auto &incidenceCounts = tpetraSmoother->getLocalIncidenceCounts();
    auto overlapMap = tpetraSmoother->getOverlapMap();
    auto ownedMap = fineStiffnessMatrix->getRowMap();

    Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node> overlapCounts(overlapMap);
    TEUCHOS_TEST_FOR_EXCEPTION(incidenceCounts.size() != overlapMap->getLocalNumElements(), std::invalid_argument,
                               "GMGOperator::setUpSmoother(): local incidence count size does not match overlap-map row count");
    {
      auto overlapCountsView = overlapCounts.getLocalViewHost(Tpetra::Access::ReadWrite);
      for (size_t i = 0; i < incidenceCounts.size(); ++i)
      {
        overlapCountsView(i, 0) = static_cast<Scalar>(incidenceCounts[i]);
      }
    }

    Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node> exporter(overlapMap, ownedMap);
    _smootherDiagonalWeight = Teuchos::rcp(new Tpetra::Vector<Scalar, LocalOrdinal, GlobalOrdinal, Node>(ownedMap));
    _smootherDiagonalWeight->putScalar(0.0);
    _smootherDiagonalWeight->doExport(overlapCounts, exporter, Tpetra::ADD);

    auto weightView = _smootherDiagonalWeight->getLocalViewHost(Tpetra::Access::ReadWrite);
    const size_t numMyElements = ownedMap->getLocalNumElements();
    for (size_t lid = 0; lid < numMyElements; lid++)
    {
      const Scalar value = weightView(lid, 0);
      TEUCHOS_TEST_FOR_EXCEPTION(value == Scalar(0), std::invalid_argument,
                                 "GMGOperator::setUpSmoother(): multiplicity value should never be 0");
      weightView(lid, 0) = Scalar(1) / value;
    }
  }
  
  if (_useSchwarzScalingWeight && (_smootherType == CAMELLIA_ADDITIVE_SCHWARZ))
  {
    _smootherWeight = computeSchwarzSmootherWeight();
  }
  
  _smoother = smoother;
  _timeSetUpSmoother = smootherSetupTimer.ElapsedTime();
}

void GMGOperator::setUseSchwarzDiagonalWeight(bool value)
{
  _useSchwarzDiagonalWeight = value;
}

void GMGOperator::setUseSchwarzScalingWeight(bool value)
{
  _useSchwarzScalingWeight = value;
}

std::string GMGOperator::smootherString(SmootherChoice choice)
{
  switch(choice)
  {
    case IFPACK_ADDITIVE_SCHWARZ:
      return "Ifpack additive Schwarz";
    case CAMELLIA_ADDITIVE_SCHWARZ:
      return "Camellia additive Schwarz";
    case NONE:
      return "None";
    case BLOCK_JACOBI:
      return "Block Jacobi";
    case BLOCK_SYMMETRIC_GAUSS_SEIDEL:
      return "Block Symmetric Gauss-Seidel";
    case POINT_JACOBI:
      return "Point Jacobi";
    case POINT_SYMMETRIC_GAUSS_SEIDEL:
      return "Point Symmetric Gauss-Seidel";
    case CHEBYSHEV:
      return "Chebyshev";
    default:
      return "Unknown smoother type";
  }
}

TMatrixPtr<double> GMGOperator::getMatrixRepresentation()
{
  Teuchos::RCP<TOperator<double> > op = Teuchos::rcp(static_cast<TOperator<double> *>(this), false);
  return Tpetra_Operator_to_Tpetra_Matrix::constructInverseMatrix(op, _finePartitionMap);
}

void GMGOperator::setUseHierarchicalNeighborsForSchwarz(bool value)
{
  _hierarchicalNeighborsForSchwarz = value;
}

TMatrixPtr<double> GMGOperator::getProlongationOperator()
{
  return _P;
}

Teuchos::RCP< Tpetra::Operator<double> > GMGOperator::getSmoother() const
{
  return _smoother;
}

TMatrixPtr<double> GMGOperator::getSmootherAsMatrix()
{
  narrate("getSmootherAsMatrix()");
  auto tMatrix = Tpetra_Operator_to_Tpetra_Matrix::constructInverseMatrix(_smoother, _finePartitionMap);
  return tMatrix;
}

//! Returns the coarse stiffness matrix (a Tpetra CrsMatrix).
TMatrixPtr<double> GMGOperator::getCoarseStiffnessMatrix()
{
  auto eMatrix = _coarseSolution->getStiffnessMatrix();
  return E_to_T::matrix(eMatrix);
}
