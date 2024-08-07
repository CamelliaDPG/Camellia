//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  GMGSolver.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 7/7/14.
//
//

#include "CamelliaDebugUtility.h"
#include "ConvergenceTestOpNorm.hpp"
#include "GDAMinimumRule.h"
#include "GMGSolver.h"
#include "MPIWrapper.h"

#include <BelosEpetraAdapter.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosSolverFactory.hpp>

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include <Teuchos_ParameterList.hpp>

// hard-code some constants from AztecOO.h that are also used by us and/or Belos:
const int AZ_warnings = -3; /* Print out only warning messages              */

const int AZ_r0  = 0; /* ||r||_2 / ||r^{(0)}||_2                      */
const int AZ_rhs = 1; /* ||r||_2 / ||b||_2                            */

using namespace Camellia;

GMGSolver::GMGSolver(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP,
                     MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap,
                     int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation, int cubatureEnrichmentDegree) :
  Narrator("GMGSolver"),
  _finePartitionMap(finePartitionMap)
{
  Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation = Teuchos::rcp( new BasisReconciliation(true) );
  _gmgOperator = Teuchos::rcp(new GMGOperator(zeroBCs,coarseMesh,coarseIP,fineMesh,fineDofInterpreter,
                                              finePartitionMap,coarseSolver, useStaticCondensation, sharedBasisReconciliation));
  _gmgOperator->getCoarseSolution()->setCubatureEnrichmentDegree(cubatureEnrichmentDegree);
  
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;

  _computeCondest = false;
  _azOutput = AZ_warnings;

  _useCG = true;
  _azConvergenceOption = AZ_rhs;
}

GMGSolver::GMGSolver(TSolutionPtr<double> fineSolution, MeshPtr coarseMesh, int maxIters, double tol,
                     Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation) :
  Narrator("GMGSolver"),
  _finePartitionMap(fineSolution->getPartitionMap())
{
  Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation = Teuchos::rcp( new BasisReconciliation(true) );
  _gmgOperator = Teuchos::rcp(new GMGOperator(fineSolution->bc()->copyImposingZero(),coarseMesh,
                                              fineSolution->ip(), fineSolution->mesh(), fineSolution->getDofInterpreter(),
                                              _finePartitionMap, coarseSolver, useStaticCondensation, sharedBasisReconciliation));
  int cubatureEnrichment = fineSolution->cubatureEnrichmentDegree();
  _gmgOperator->getCoarseSolution()->setCubatureEnrichmentDegree(cubatureEnrichment);
  
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;

  _computeCondest = true;
  _azOutput = AZ_warnings;

  _useCG = true;
  _azConvergenceOption = AZ_rhs;
}

GMGSolver::GMGSolver(TSolutionPtr<double> fineSolution, int maxIters, double tol, int H1OrderCoarse,
                     Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation) :
Narrator("GMGSolver"),
_finePartitionMap(fineSolution->getPartitionMap())
{
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;
  
  _computeCondest = true;
  _azOutput = AZ_warnings;
  
  _useCG = true;
  _azConvergenceOption = AZ_rhs;
  
  // notion here is that we build a hierarchy of meshes in some intelligent way
  // for now, we jump in p from whatever it is on the fine mesh to H1OrderCoarse, and then
  // do single h-coarsening steps until we reach the coarsest topology.
  
  vector<MeshPtr> meshesCoarseToFine;
  MeshPtr fineMesh = fineSolution->mesh();
  meshesCoarseToFine.push_back(fineMesh);
  
  VarFactoryPtr vf = fineMesh->bilinearForm()->varFactory();
  int delta_k = 1; // this shouldn't matter for meshes outside the finest
  MeshPtr mesh_pCoarsened = Teuchos::rcp( new Mesh(fineMesh->getTopology(), vf, H1OrderCoarse, delta_k) );
  
  meshesCoarseToFine.insert(meshesCoarseToFine.begin(), mesh_pCoarsened);

  // TODO: finish implementing this
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This GMGSolver constructor not yet completed!");
  
  // once we have a list of meshes, use gmgOperatorFromMeshSequence to build the operator
}

GMGSolver::GMGSolver(TSolutionPtr<double> fineSolution, const std::vector<MeshPtr> &meshesCoarseToFine, int maxIters,
                     double tol, GMGOperator::MultigridStrategy multigridStrategy, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation,
                     bool useDiagonalSchwarzWeighting) :
Narrator("GMGSolver"),
_finePartitionMap(fineSolution->getPartitionMap())
{
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;
  
  _computeCondest = true;
  _azOutput = AZ_warnings;
  
  if ((multigridStrategy == GMGOperator::FULL_MULTIGRID_V) && (multigridStrategy == GMGOperator::FULL_MULTIGRID_W))
    _useCG = false; // Full multigrid is not symmetric
  else
    _useCG = true;
  _azConvergenceOption = AZ_rhs;
  
  Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation = Teuchos::rcp( new BasisReconciliation(true) );
  
  _gmgOperator = gmgOperatorFromMeshSequence(meshesCoarseToFine, fineSolution, multigridStrategy, coarseSolver, useStaticCondensation,
                                             useDiagonalSchwarzWeighting, sharedBasisReconciliation);
}

Teuchos::RCP<GMGSolver> GMGSolver::cgSolver(SolutionPtr soln, double tol, int maxIters)
{
  return GMGSolver::gmgSolver(soln, true, tol, maxIters);
}

double GMGSolver::condest()
{
  return _condest;
}

int GMGSolver::getGridLevelCount() const
{
  return _gmgOperator->getOperatorLevel() + 1;
}

vector<int> GMGSolver::getIterationCountLog()
{
  return _iterationCountLog;
}

int GMGSolver::iterationCount()
{
  return _iterationCount;
}

Teuchos::RCP<GMGOperator> GMGSolver::gmgOperatorFromMeshSequence(const std::vector<MeshPtr> &meshesCoarseToFine, SolutionPtr fineSolution,
                                                                 GMGOperator::MultigridStrategy multigridStrategy,
                                                                 SolverPtr coarseSolver, bool useStaticCondensationInCoarseSolve,
                                                                 bool useDiagonalSchwarzWeighting,
                                                                 Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation)
{
  if (multigridStrategy != GMGOperator::SMOOTHER_ONLY)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(meshesCoarseToFine.size() < 2, std::invalid_argument, "meshesCoarseToFine must have at least two meshes, unless multigrid strategy is SMOOTHER_ONLY");
  }
  
  Teuchos::RCP<GMGOperator> coarseOperator = Teuchos::null, finerOperator = Teuchos::null, finestOperator = Teuchos::null;
  
  Teuchos::RCP<DofInterpreter> fineDofInterpreter = fineSolution->getDofInterpreter();
  IPPtr ip = fineSolution->ip();
  BCPtr zeroBCs = fineSolution->bc()->copyImposingZero();
  Epetra_Map finePartitionMap = fineSolution->getPartitionMap();
  
  if (multigridStrategy == GMGOperator::SMOOTHER_ONLY)
  {
    MeshPtr mesh = meshesCoarseToFine[0];
    return Teuchos::rcp(new GMGOperator(GMGOperator::SMOOTHER_ONLY, zeroBCs, mesh, ip, mesh, fineDofInterpreter, finePartitionMap,
                                        useStaticCondensationInCoarseSolve, sharedBasisReconciliation));
  }
  
  int cubatureEnrichmentDegree = fineSolution->cubatureEnrichmentDegree();
  
  // for now, leaving these two parameters as the historical defaults (1 and 1)
  // it may be that iteration counts would remain more stable under refinements if we increased these
  // (but our major expense is constructing prolongation operator, so this is not likely to make a big difference in runtime)
  int coarseSmootherApplications = 1;
  int fineSmootherApplications = 1;
  
  const static int SMOOTHER_OVERLAP_FOR_LOWEST_ORDER_P = 0; // new; old approach would have had 0 here...
  
  bool hRefinedPrevious = false; // assumption is that we do h-refinements on a coarse poly mesh, and then p refinements.
  for (int i=meshesCoarseToFine.size()-1; i>0; i--)
  {
    MeshPtr fineMesh = meshesCoarseToFine[i];
    MeshPtr coarseMesh = meshesCoarseToFine[i-1];
    if (i>1)
    {
      coarseOperator = Teuchos::rcp(new GMGOperator(zeroBCs, coarseMesh, ip, fineMesh, fineDofInterpreter, finePartitionMap,
                                                    useStaticCondensationInCoarseSolve, sharedBasisReconciliation));
    }
    else
    {
      coarseOperator = Teuchos::rcp(new GMGOperator(zeroBCs, coarseMesh, ip, fineMesh, fineDofInterpreter, finePartitionMap,
                                                    coarseSolver, useStaticCondensationInCoarseSolve, sharedBasisReconciliation));
    }
    coarseOperator->setSmootherType(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
    coarseOperator->setUseSchwarzScalingWeight(true);
    coarseOperator->setMultigridStrategy(multigridStrategy);
    bool hRefined = fineMesh->numActiveElements() > coarseMesh->numActiveElements();
    coarseOperator->setUseHierarchicalNeighborsForSchwarz(false); // changed Dec. 22 2015, after tests revealed that hiarchical truncation seems to be a uniformly bad idea...
    coarseOperator->getCoarseSolution()->setCubatureEnrichmentDegree(cubatureEnrichmentDegree);
    
    int smootherOverlap;
    if (hRefined)
    {
      smootherOverlap = 1;
      coarseOperator->setUseSchwarzDiagonalWeight(useDiagonalSchwarzWeighting); // empirically, this doesn't work well for h-multigrid -- but that was with the square-root-symmetrized weight, which James Lottes says is not a good idea
    }
    else
    {
      smootherOverlap = 0;
      coarseOperator->setUseSchwarzDiagonalWeight(useDiagonalSchwarzWeighting); // not sure which is better; use false for now
    }

    coarseOperator->setSmootherOverlap(smootherOverlap);
    coarseOperator->setSmootherApplicationCount(coarseSmootherApplications);
    
    if (finerOperator != Teuchos::null)
    {
      finerOperator->setCoarseOperator(coarseOperator);
      if (hRefined && !hRefinedPrevious)
      {
        // at the border between h and p refinements, use smootherOverlap of 1
        finerOperator->setSmootherOverlap(SMOOTHER_OVERLAP_FOR_LOWEST_ORDER_P);
      }
    }
    else
    {
      finestOperator = coarseOperator;
    }
    if ((i == 1) && !hRefined)
    {
      coarseOperator->setSmootherOverlap(SMOOTHER_OVERLAP_FOR_LOWEST_ORDER_P);
    }
    finerOperator = coarseOperator;
    finePartitionMap = finerOperator->getCoarseSolution()->getPartitionMap();
    fineDofInterpreter = finerOperator->getCoarseSolution()->getDofInterpreter();
    hRefinedPrevious = hRefined;
  }
  finestOperator->setSmootherApplicationCount(fineSmootherApplications);
  return finestOperator;
}

Teuchos::RCP<GMGSolver> GMGSolver::gmgSolver(SolutionPtr soln, bool useCG, double tol, int maxIters)
{
  Teuchos::ParameterList pl;
  pl.set("kCoarse", 0);
  int delta_k = soln->mesh()->testSpaceEnrichment(); // might matter in cases where GMGOperator actually computes some local stiffness matrices on its own (currently, this can happen on coarse meshes when static condensation is employed).
  pl.set("delta_k", delta_k);
  pl.set("jumpToCoarsePolyOrder", true);
  vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(soln->mesh(), pl);
  
  vector<MeshPtr> prunedMeshes;
  int minDofCount = 2000; // skip any coarse meshes that have fewer dofs than this
  for (int i=0; i<meshesCoarseToFine.size()-2; i++) // leave the last two meshes, so we can guarantee there are at least two
  {
    MeshPtr mesh = meshesCoarseToFine[i];
    GlobalIndexType numGlobalDofs;
    if (soln->usesCondensedSolve())
      numGlobalDofs = mesh->numFluxDofs(); // this might under-count, in case e.g. of pressure constraints.  But it's meant as a heuristic anyway.
    else
      numGlobalDofs = mesh->numGlobalDofs();
    
    if (numGlobalDofs > minDofCount)
    {
      prunedMeshes.push_back(mesh);
    }
  }
  prunedMeshes.push_back(meshesCoarseToFine[meshesCoarseToFine.size()-2]);
  prunedMeshes.push_back(meshesCoarseToFine[meshesCoarseToFine.size()-1]);
  
  bool saveFactorization = true;
  auto coarseSolver = Solver::getDirectSolver(saveFactorization);
  
  Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(new GMGSolver(soln, prunedMeshes, maxIters, tol, GMGOperator::V_CYCLE,
                                                                 coarseSolver, soln->usesCondensedSolve(), false));
  
  gmgSolver->setAztecOutput(0);
  gmgSolver->setComputeConditionNumberEstimate(false);
  
  gmgSolver->setUseConjugateGradient(useCG);
  
  return gmgSolver;
}

Teuchos::RCP<GMGSolver> GMGSolver::gmgSolver(TSolutionPtr<double> fineSolution, bool useConjugateGradient,
                                             const std::vector<MeshPtr> &meshesCoarseToFine, int maxIters, double tol,
                                             GMGOperator::MultigridStrategy multigridStrategy,
                                             Teuchos::RCP<Solver> coarseSolver)
{
  auto solver = Teuchos::rcp( new GMGSolver(fineSolution, meshesCoarseToFine, maxIters, tol, multigridStrategy,
                                            coarseSolver, fineSolution->usesCondensedSolve()) );
  solver->setUseConjugateGradient(useConjugateGradient);
  return solver;
}

Teuchos::RCP<GMGSolver> GMGSolver::gmresSolver(SolutionPtr soln, double tol, int maxIters)
{
  return GMGSolver::gmgSolver(soln, false, tol, maxIters);
}

vector<MeshPtr> GMGSolver::meshesForMultigrid(MeshPtr fineMesh, int kCoarse, int delta_k)
{
  Teuchos::ParameterList pl;
  
  if (delta_k == -1) // then use fine mesh's delta_k
  {
    delta_k = fineMesh->globalDofAssignment()->getTestOrderEnrichment();
  }
  
  pl.set("kCoarse", kCoarse);
  pl.set("delta_k", delta_k);
  pl.set("jumpToCoarsePolyOrder",true);
  
  return meshesForMultigrid(fineMesh, pl);
}

vector<MeshPtr> GMGSolver::meshesForMultigrid(MeshPtr fineMesh, Teuchos::ParameterList &parameters)
{
  // for now, we do the following (later, we may introduce a ParameterList to give finer-grained control):
  /*
   coarsest h-mesh with k=kCoarse
   once-refined h-mesh with k=kCoarse (not necessarily uniformly refined--maximally within fine mesh)
   twice-refined h-mesh with k=kCoarse
   ...
   fine h-mesh with k=kCoarse
   fine h-mesh with k=kFine
   */
  
  bool jumpToCoarsePolyOrder = parameters.get<bool>("jumpToCoarsePolyOrder",true);
  int kCoarse = parameters.get<int>("kCoarse", 0);
  int delta_k = parameters.get<int>("delta_k",1);
  
  MeshPtr curvilinearFineMesh = Teuchos::null;
  if (fineMesh->getTransformationFunction() != Teuchos::null)
  {
    /*
     There is an issue here: GMGOperator now insists that baseMeshTopology() for fine and coarse meshes
     points to the same object.  Two possibilities:
     1) Let the straight-edged variant be a view.  I.e. MeshTopologyView now has the possibility of overriding the edge-to-curve map.
     2) Do something more drastic.
     */
    // TODO: work something out here.
    curvilinearFineMesh = fineMesh;
    fineMesh = curvilinearFineMesh->deepCopy();
    map<pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr> edgeToCurveMap;
    fineMesh->setEdgeToCurveMap(edgeToCurveMap); // should result in a null transformation function (straight edges)
    TEUCHOS_TEST_FOR_EXCEPTION(fineMesh->getTransformationFunction() != Teuchos::null, std::invalid_argument,
                               "Internal error: attempt to set null transformation function failed");
  }
  
  MeshTopologyViewPtr fineMeshTopo = fineMesh->getTopology();
  set<GlobalIndexType> thisLevelCellIndices = fineMeshTopo->getRootCellIndicesLocal();
  GlobalIndexType thisLevelNumCells = 0;

  int myTensorialDegree = -1;
  if (thisLevelCellIndices.size() > 0)
  {
    IndexType cellIndex = *thisLevelCellIndices.begin();
    myTensorialDegree = fineMeshTopo->getCell(cellIndex)->topology()->getTensorialDegree();
  }
  int tensorialDegree;
  fineMesh->Comm()->MaxAll(&myTensorialDegree, &tensorialDegree, 1);
  // if we have no elements, trust the global guy.  Otherwise, check that the global guy matches ours.
  if (thisLevelCellIndices.size() > 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(myTensorialDegree != tensorialDegree, std::invalid_argument, "rank-local tensorial degree does not agree with global tensorial degree.");
  }
  
  GlobalIndexType fineMeshNumCells = fineMeshTopo->activeCellCount();
  
  vector<MeshPtr> meshesCoarseToFine;
  
  map<int,int> trialOrderEnhancements = fineMesh->getDofOrderingFactory().getTrialOrderEnhancements();
  
  vector<int> H1Order_coarse(tensorialDegree + 1, kCoarse + 1);
  BFPtr bf = fineMesh->bilinearForm();
  
  do {
    MeshTopologyViewPtr thisLevelMeshTopo = fineMeshTopo->getView(thisLevelCellIndices);
    // let the fine mesh induce partitioning
    MeshPartitionPolicyPtr inducedPartitionPolicy = MeshPartitionPolicy::inducedPartitionPolicyFromRefinedMesh(thisLevelMeshTopo, fineMesh);
    
    // create the actual Mesh object:
    MeshPtr thisLevelMesh = Teuchos::rcp(new Mesh(thisLevelMeshTopo, bf, H1Order_coarse, delta_k,
                                                  trialOrderEnhancements, map<int,int>(), inducedPartitionPolicy));
    meshesCoarseToFine.push_back(thisLevelMesh);
    
    set<GlobalIndexType> nextLevelCellIndices;
    for (GlobalIndexType cellIndex : thisLevelCellIndices)
    {
      CellPtr cell = fineMeshTopo->getCell(cellIndex);
      if (cell->isParent(fineMeshTopo))
      {
        vector<IndexType> childIndices = cell->getChildIndices(fineMeshTopo);
        for (IndexType childCellID : childIndices)
        {
          if (fineMeshTopo->isValidCellIndex(childCellID))
          {
            nextLevelCellIndices.insert(childCellID);
          }
        }
      }
      else
      {
        nextLevelCellIndices.insert(cellIndex);
      }
    }
    
    thisLevelNumCells = thisLevelMeshTopo->activeCellCount();
    
    thisLevelCellIndices = nextLevelCellIndices;
  } while (fineMeshNumCells > thisLevelNumCells);
  
  const set<GlobalIndexType>* myFineCellIndices = &fineMesh->cellIDsInPartition();
  
  set<GlobalIndexType> locallyKnownFineCellIndices = fineMeshTopo->getLocallyKnownActiveCellIndices();
  MeshTopologyViewPtr fineMeshTopoView = fineMeshTopo->getView(locallyKnownFineCellIndices);
  MeshPartitionPolicyPtr inducedPartitionPolicy = MeshPartitionPolicy::inducedPartitionPolicy(fineMesh);
  if (! jumpToCoarsePolyOrder)
  {
    // NOTE: this option is not very well-supported for space-time meshes, because of our lack of anisotropic p-refinement
    //       support; essentially, for this to work, you need to have the same temporal poly order as spatial.
    
    bool someCellWasRefined = true;

    map<GlobalIndexType,int> pRefinements; // relative to H1OrderCoarse

    MeshPtr meshToPRefine;
    BFPtr bf = fineMesh->bilinearForm();
    if (bf != Teuchos::null)
    {
      meshToPRefine = Teuchos::rcp( new Mesh(fineMeshTopoView, bf, H1Order_coarse, delta_k, trialOrderEnhancements, map<int,int>(), inducedPartitionPolicy) );
    }
    else
    {
      VarFactoryPtr vf = fineMesh->varFactory();
      meshToPRefine = Teuchos::rcp( new Mesh(fineMeshTopoView, vf, H1Order_coarse, delta_k, trialOrderEnhancements, map<int,int>(), inducedPartitionPolicy) );
    }
    
    while (someCellWasRefined)
    {
      meshToPRefine = meshToPRefine->deepCopy();
      meshToPRefine->globalDofAssignment()->setCellPRefinements(pRefinements);
      
      someCellWasRefined = false;

      vector<pair<GlobalIndexTypeToCast,int>> pRefinementsVector; // only store for non-zero p-refinements
      for (GlobalIndexType cellIndex : *myFineCellIndices)
      {
        ElementTypePtr coarseElemType = meshToPRefine->getElementType(cellIndex);
        int kForCoarseCell = coarseElemType->trialOrderPtr->maxBasisDegreeForVolume();
        
        ElementTypePtr fineElemType = fineMesh->getElementType(cellIndex);
        int kForFineCell = fineElemType->trialOrderPtr->maxBasisDegreeForVolume();
        
        int kToAdd;
        if (kForCoarseCell == 0)
        {
          kToAdd = 1;
        }
        else
        {
          kToAdd = min(kForCoarseCell,kForFineCell - kForCoarseCell);
        }
        
        if (kForCoarseCell + kToAdd < kForFineCell) // if kForCoarseCell + kToAdd == kForFineCell, then the fine mesh will do the job
        {
          int pRefinement = pRefinements[cellIndex] + kToAdd;
          pRefinements[cellIndex] = pRefinement;
          pRefinementsVector.push_back({cellIndex,pRefinement});
          someCellWasRefined = true;
        }
        else
        {
          auto foundEntry = pRefinements.find(cellIndex);
          if (foundEntry != pRefinements.end())
          {
            pRefinementsVector.push_back({cellIndex,foundEntry->second});
          }
        }
      }
      
      someCellWasRefined = MPIWrapper::globalOr(*meshToPRefine->Comm(),someCellWasRefined);
      
      if (someCellWasRefined)
      {
        // gather the pRefinements
        vector<int> offsets;
        vector<pair<GlobalIndexTypeToCast,int>> pRefinementsGathered;
        MPIWrapper::allGatherVariable(*meshToPRefine->Comm(), pRefinementsGathered, pRefinementsVector, offsets);
        pRefinements.clear();
        pRefinements.insert(pRefinementsGathered.begin(),pRefinementsGathered.end());
        
        meshToPRefine->globalDofAssignment()->setCellPRefinements(pRefinements);
        MeshPartitionPolicyPtr inducedPartitionPolicy = MeshPartitionPolicy::inducedPartitionPolicy(meshToPRefine,fineMesh);
        meshToPRefine->setPartitionPolicy(inducedPartitionPolicy);
        
        meshesCoarseToFine.push_back(meshToPRefine);
      }
    }
  }
  meshesCoarseToFine.push_back(fineMesh);
  
  if (curvilinearFineMesh != Teuchos::null) meshesCoarseToFine.push_back(curvilinearFineMesh);
  
  return meshesCoarseToFine;
}

void GMGSolver::setPrintToConsole(bool printToConsole)
{
  _printToConsole = printToConsole;
}

void GMGSolver::setTolerance(double tol)
{
  _tol = tol;
}

int GMGSolver::resolve()
{
  bool buildCoarseStiffness = false; // won't have changed since solve() was called
  return solve(buildCoarseStiffness);
}

int GMGSolver::solve()
{
  bool buildCoarseStiffness = true;
  return solve(buildCoarseStiffness);
}

int GMGSolver::solve(bool buildCoarseStiffness)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  
  int solveResult;
  
  using namespace Teuchos;
  typedef Epetra_MultiVector MV;
  typedef double Scalar;
  typedef Epetra_Operator OP;
  typedef Belos::LinearProblem<Scalar, MV, OP> BelosProblem;
  typedef RCP<BelosProblem> BelosProblemPtr;
  BelosProblemPtr problem = rcp( new BelosProblem(_stiffnessMatrix, _lhs, _rhs) );
  
  Belos::SolverFactory<Scalar, MV, OP> factory;
  RCP<Belos::SolverManager<Scalar, MV, OP> > solver;
  
  if (buildCoarseStiffness)
  {
    _gmgOperator->setFineStiffnessMatrix(_stiffnessMatrix.get());
  }
  
  RCP<ParameterList> solverParams = parameterList();
  
  string resScaling = "Norm of RHS";
  if (_azConvergenceOption == AZ_rhs)
  {
    resScaling = "Norm of RHS";
  }
  else if (_azConvergenceOption == AZ_r0)
  {
    resScaling = "Norm of Initial Residual";
  }
  
  solverParams->set ("Maximum Iterations", _maxIters);
  solverParams->set ("Convergence Tolerance", _tol);
  solverParams->set ("Estimate Condition Number", _computeCondest);
  solverParams->set ("Residual Scaling", resScaling);
  
  if (_azOutput > 0)
  {
    solverParams->set("Verbosity",Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
    solverParams->set("Output Frequency",_azOutput);
    solverParams->set("Output Style",Belos::Brief);
  }
  
  Belos::PseudoBlockCGSolMgr<Scalar,MV,OP>* cgSolver;
//    Belos::PseudoBlockGmresSolMgr<Scalar,MV,OP>* gmresSolver;
  
  if (_useCG)
  {
    solver = factory.create("CG", solverParams);
    cgSolver = dynamic_cast<Belos::PseudoBlockCGSolMgr<Scalar,MV,OP>*>(solver.get());
  }
  else
  {
    solverParams->set ("Num Blocks", 200); // is this equivalent to AZ_kspace??
    solver = factory.create("GMRES", solverParams);
//      gmresSolver = dynamic_cast<Belos::PseudoBlockGmresSolMgr<Scalar,MV,OP>*>(solver.get());
  }
  
//    {
//      // TRYING SOMETHING
//      Epetra_MultiVector B_times_rhs(_rhs->Map(), _rhs->NumVectors());
//      _gmgOperator->ApplySmoother(*_rhs, B_times_rhs, true); // true: apply weight on left
//      double normValues[_rhs->NumVectors()];
//      _rhs->Dot(B_times_rhs, &normValues[0]);
//
//      if (rank == 0) cout << "GMGSolver: Schwarz-induced norm of _rhs: " << normValues[0] << "\n";
//    }
  
  problem->setRightPrec(_gmgOperator);
  problem->setProblem();
  solver->setProblem(problem);

  if (_exportFullOperators)
  {
    ostringstream path;
    path << _pathForExport << "M.dat";
    if (rank == 0) cout << "Writing preconditioner to " << path.str() << endl;
    EpetraExt::RowMatrixToMatrixMarketFile(path.str().c_str(),*_gmgOperator->getMatrixRepresentation(), NULL, NULL, false);
    path.str("");
    path << _pathForExport << "A.dat";
    if (rank == 0) cout << "Writing system matrix to " << path.str() << endl;
    EpetraExt::RowMatrixToMatrixMarketFile(path.str().c_str(),*_stiffnessMatrix, NULL, NULL, false);
  }
  
//    cout << "NOTE: setting user convergence status test.\n";
//    Teuchos::RCP<Belos::StatusTest<Scalar,MV,OP> > smootherTestNorm = Teuchos::rcp(new ConvergenceTestOpNorm<Scalar, MV, OP>(gmgOperator()->getSmoother(), _tol));
//    cgSolver->setUserConvStatusTest(smootherTestNorm);
  
  Belos::ReturnType belosResult = solver->solve();
  
  if (belosResult == Belos::Converged)
  {
    solveResult = 0; // success
  }
  else if ((solver->getNumIters() == _maxIters) && !_returnErrorIfMaxItersReached)
  {
    solveResult = 0;
  }
  else
  {
    solveResult = 1;  // failure (not converged, and user indicates that should be an error)
  }
  
  if (_computeCondest && _useCG)  // right now, gmresSolver does not support condest
  {
    _condest = cgSolver->getConditionEstimate();
    if ((rank==0) && (_azOutput > 0))
    {
      cout << "Condition number estimate: " << _condest << endl;
    }
  }
  
  // TODO: try some more sophisticated stopping criterion
  
  _iterationCount = solver->getNumIters();

  
  _iterationCountLog.push_back(_iterationCount);

  return solveResult;
}

void GMGSolver::setAztecConvergenceOption(int value)
{
  _azConvergenceOption = value;
}

void GMGSolver::setAztecOutput(int value)
{
  _azOutput = value;
}

void GMGSolver::setComputeConditionNumberEstimate(bool value)
{
  _computeCondest = value;
}

void GMGSolver::setExportFullOperatorsOnSolve(bool exportOnSolve, string path)
{
  _exportFullOperators = exportOnSolve;
  _pathForExport = path;
}

void GMGSolver::setPrintIterationCount(bool value)
{
  _printIterationCountIfNoAzOutput = value;
}

void GMGSolver::setReturnErrorIfMaxItersReached(bool value)
{
  _returnErrorIfMaxItersReached = value;
}

vector<Teuchos::RCP<GMGOperator>> GMGSolver::getOperatorStack(bool coarseToFine)
{
  Teuchos::RCP<GMGOperator> op = _gmgOperator;
  vector<Teuchos::RCP<GMGOperator>> opStack = {op};
  while (op != Teuchos::null)
  {
    if (coarseToFine)
    {
      opStack.insert(opStack.begin(), op);
    }
    else
    {
      opStack.push_back(op);
    }
    op = op->getCoarseOperator();
  }
  return opStack;
}

// ! Sets the number of times the smoother will be applied when it is applied.  (For multigrid strategies other than TWO_LEVEL, it is applied this many times before and after each application of the coarse operator.)
void GMGSolver::setSmootherApplicationCount(int count)
{
  auto opStack = getOperatorStack(false);
  for (auto op : opStack)
  {
    op->setSmootherApplicationCount(count);
  }
}

void GMGSolver::setSmootherType(GMGOperator::SmootherChoice smootherType)
{
  auto opStack = getOperatorStack(false);
  for (auto op : opStack)
  {
    op->setSmootherType(smootherType);
  }
}

void GMGSolver::setUseConjugateGradient(bool value)
{
  _useCG = value;
}
