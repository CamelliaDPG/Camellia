//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "ConfigurationSpaceFormulation.h"
#include "ErrorIndicator.h"
#include "ExpFunction.h"
#include "FixedCountRefinementStrategy.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "Projector.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"
#include "VlasovBackwardEulerFormulation.h"

#include <sys/stat.h>
#include <sys/types.h>

using namespace Camellia;

class Maxwellian1V : public TFunction<double>
{
  double _u;
  double _sigma;
  double _n;
public:
  Maxwellian1V(double u, double sigma, double n)
  :
  _u(u),
  _sigma(sigma),
  _n(n)
  {}
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & phaseSpacePoints = basisCache->getPhysicalCubaturePointsFC();
    const int numCells  = phaseSpacePoints.dimension(0);
    const int numPoints = phaseSpacePoints.dimension(1);
    
    const double weight = _n / (sqrt(2. * M_PI) * _sigma);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,1); // vx is the second dimension for 1D1V
        const double exponent = - 0.5 * (vx - _u) * (vx - _u) / (_sigma * _sigma);
        values(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
      }
    }
  }
};

static const std::string S_TWO_STREAM = "two-stream";

void setPoissonBC(VlasovBackwardEulerFormulation &form, const double &x0)
{
  // we have periodic BCs; we just need to pin the potential at a point -- we choose the x0 (left) point.
  const int poissonTagLeft = form.createConfigurationMeshSideSetTag(SpatialFilter::matchingX(x0));
  
  auto VE_hat = form.configurationSpaceFormulation()->VE_hat();
  form.addConfigurationSpaceBC(poissonTagLeft, VE_hat, Function::constant(0.0));

  // DEBUG implementation below -- imitates the original Boltzmann 1D1V pretty closely; can be switched on if and when we're suspicious of the createConfigurationMeshSideSetTag() flow above.
  {
//  auto intervalMeshTopology = form.configurationSpaceMeshTopology();
//  EntitySetPtr x0EntitySet = intervalMeshTopology->createEntitySet();
//  IndexType x0Vertex;
//  bool x0Found = intervalMeshTopology->getVertexIndex({x0}, x0Vertex);
//  TEUCHOS_TEST_FOR_EXCEPTION(!x0Found, std::invalid_argument, "x0 not found");
//
//  x0EntitySet->addEntity(0, x0Vertex); // add entity of dimension 0 with ID x0Vertex to the x0 entity set.
//
//  static const int POISSON_TAG_X0 = 0;
//  intervalMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X0, x0EntitySet);
//
//  BCPtr poissonBC = BC::bc();
//  poissonBC->addDirichlet(form.configurationSpaceFormulation()->VE_hat(), POISSON_TAG_X0, Function::constant(0.0));
//
//  form.configurationSpaceSolution()->setBC(poissonBC);
  }
}

enum ProblemChoice
{
  TWO_STREAM
};

static const std::map<std::string, ProblemChoice> stringToProblemChoice {{S_TWO_STREAM,TWO_STREAM}};
static const std::map<ProblemChoice, std::string> problemChoiceToString {{TWO_STREAM,S_TWO_STREAM}};

ProblemChoice getProblemChoice(const std::string &problemChoiceString)
{
  const auto &entry = stringToProblemChoice.find(problemChoiceString);
  if (entry == stringToProblemChoice.end())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unrecognized problem choice string");
  }
  else
  {
    return entry->second;
  }
}

std::string getProblemChoiceString(ProblemChoice problem)
{
  const auto &entry = problemChoiceToString.find(problem);
  if (entry == problemChoiceToString.end())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unrecognized problem choice");
  }
  else
  {
    return entry->second;
  }
}

// the following is somewhat specific to our situation; it assumes:
// - the two meshes share a coarse initial mesh (i.e., cell levels are comparable)
// - all refinements are isotropic
// - the meshes are not distributed (all geometry is locally known)
class MeshMap
{
  std::map<GlobalIndexType,int> _coarseLevelForFineCell;
  std::map<GlobalIndexType,GlobalIndexType> _coarseCellForFineCell;
  
  IndexType coarseCellMatchingVertices(MeshTopologyViewPtr coarseMeshTopo, const std::vector<IndexType> &vertices)
  {
    bool firstVertex = true;
    unsigned vertexDim = 0;
    set<IndexType> matchingCells;
    for (const auto & vertexIndex : vertices)
    {
      set< pair<IndexType, unsigned> > matchingCellPairs = coarseMeshTopo->getCellsContainingEntity(vertexDim, vertexIndex);
      set<IndexType> matchingCellsIntersection;
      for (const auto & cellPairEntry : matchingCellPairs)
      {
        IndexType cellID = cellPairEntry.first;
        if (firstVertex)
        {
          matchingCellsIntersection.insert(cellID);
        }
        else
        {
          if (matchingCells.find(cellID) != matchingCells.end())
          {
            matchingCellsIntersection.insert(cellID);
          }
        }
      }
      matchingCells = matchingCellsIntersection;
      firstVertex = false;
    }
    if (matchingCells.size() == 0)
    {
      return -1; // not found
    }
    else if (matchingCells.size() == 1)
    {
      return *matchingCells.begin();
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Expected at most one matching cells; got several");
    }
  }
public:
  MeshMap(MeshPtr coarseMesh, MeshPtr fineMesh)
  {
    auto fineMeshTopo   = fineMesh->getTopology();
    auto coarseMeshTopo = coarseMesh->getTopology();
    
    const auto & myFineCells = fineMesh->cellIDsInPartition();
    for (auto fineCellID : myFineCells)
    {
      auto fineCell         = fineMeshTopo->getCell(fineCellID);
      auto fineCellAncestor = fineCell;
      while (fineCellAncestor != Teuchos::null)
      {
        const auto & fineCellVertexIndices = fineCellAncestor->vertices();
        std::vector<IndexType> coarseCellVertexIndices(fineCellVertexIndices.size(), -1);
        int vertexOrdinal = 0;
        bool allVerticesFound = true;
        for (const auto & fineCellVertexIndex : fineCellVertexIndices)
        {
          const auto &vertexCoords = fineMeshTopo->getVertex(fineCellVertexIndex);
          bool vertexFound = coarseMeshTopo->getVertexIndex(vertexCoords, coarseCellVertexIndices[vertexOrdinal]);
          vertexOrdinal++;
          if (!vertexFound)
          {
            allVerticesFound = false;
            break;
          }
        }
        if (allVerticesFound)
        {
          IndexType coarseCellID = coarseCellMatchingVertices(coarseMeshTopo, coarseCellVertexIndices);
          if (coarseCellID != -1)
          {
            _coarseLevelForFineCell[fineCellID] = fineCellAncestor->level();
            _coarseCellForFineCell[fineCellID]  = coarseCellID;
            break; // cell found; break out of the while loop
          }
        }
        fineCellAncestor = fineCellAncestor->getParent();
      }
    }
  }
  
  int coarseLevelForFineCell(GlobalIndexType fineCellIndex) const
  {
    auto entryIt = _coarseLevelForFineCell.find(fineCellIndex);
    TEUCHOS_TEST_FOR_EXCEPTION(entryIt == _coarseLevelForFineCell.end(), std::invalid_argument, "fine cell not found");
    return entryIt->second;
  }
  
  //! Note that the coarse cell may correspond to an ancestor of fineCellIndex; use coarseLevelForFineCell() to determine how many levels to go up
  GlobalIndexType coarseCellForFineCell(GlobalIndexType fineCellIndex) const
  {
    auto entryIt = _coarseCellForFineCell.find(fineCellIndex);
    TEUCHOS_TEST_FOR_EXCEPTION(entryIt == _coarseCellForFineCell.end(), std::invalid_argument, "fine cell not found");
    return entryIt->second;
  }
};

RefinementStrategyPtr constructRefStrategy(VlasovBackwardEulerFormulation &form, double refinementThreshold, int maxElements, int maxRefinementLevel, ErrorIndicatorPtr unrefinementErrorIndicator)
{
  auto vlasovSoln  = form.vlasovSolution();
  auto refStrategy = RefinementStrategy::energyErrorRefinementStrategy(vlasovSoln, refinementThreshold);

  refStrategy->setMaxElements(maxElements);
  refStrategy->setMaximumRefinementLevel(maxRefinementLevel);
  refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
  const int H1Order = vlasovSoln->mesh()->globalDofAssignment()->getInitialH1Order()[0];
  refStrategy->setMaxP(H1Order); // disallow p refinements (these would require revision of VelocitySpaceExtrusion)
  return refStrategy;
}

void performRefinements(double & lastRefinementMaxElementError, VlasovBackwardEulerFormulation &form, RefinementStrategyPtr refStrategy, bool gateRefinements, double refinementThreshold, int maxElements)
{
  if (refStrategy == Teuchos::null) return;
  bool shouldRefine;
  if (gateRefinements)
  {
    // we compute error indicator first; if no refinement-eligible element has error greater than what would have been refined last time, then we don't do a refinement step now.
    
    auto localCellsToExclude = refStrategy->localCellsIneligibleForRefinement();
    
    ErrorIndicatorPtr errorIndicator = refStrategy->errorIndicator();
    errorIndicator->measureError();
    const double maxError = errorIndicator->maxError(localCellsToExclude);
    
    shouldRefine = maxError > refinementThreshold * lastRefinementMaxElementError;
    
    if (shouldRefine)
    {
      lastRefinementMaxElementError = maxError;
    }
    else
    {
      // let's still record the measured error in RefinementStrategy's records
      const double totalEnergyError = errorIndicator->totalError();
      const int numElements = form.vlasovMesh()->numActiveElements();
      const int numDofs     = form.vlasovMesh()->numGlobalDofs();
      refStrategy->recordResults(numElements, numDofs, totalEnergyError);
    }
  }
  else
  {
    shouldRefine = true;
  }
  
  if (shouldRefine)
  {
    // for now, print refinement info to console on rank 0.
    const bool printToConsole = form.vlasovMesh()->Comm()->MyPID() == 0;
    const bool doUnrefinements = (maxElements < std::numeric_limits<int>::max());
    refStrategy->refineAndUnrefine(doUnrefinements, printToConsole);
    
    // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
    form.configurationSpaceMesh()->repartitionAndRebuild();
  }
}

void interpolateToFineSolution(VarPtr var, SolutionPtr coarseSolution, SolutionPtr fineSolution, MeshMap fineToCoarseMap, BasisReconciliation &br)
{
  MeshPtr coarseMesh = coarseSolution->mesh();
  MeshPtr fineMesh = fineSolution->mesh();
  
  const auto & myFineCells = fineMesh->cellIDsInPartition();
  
  // the coarse cell distribution over MPI ranks need not have anything to do with the fine distribution; the fine distribution drives things
  // therefore, we need to do MPI communication for coarse cell solutions; we request the coarse cell IDs that correspond to local fine cells
  std::set<GlobalIndexType> coarseCellsToRequest;
  for (const auto & fineCellID : myFineCells)
  {
    auto coarseCellID = fineToCoarseMap.coarseCellForFineCell(fineCellID);
    if (! coarseMesh->myCellsInclude(coarseCellID))
    {
      coarseCellsToRequest.insert(coarseCellID);
    }
  }
  coarseSolution->importSolutionForOffRankCells(coarseCellsToRequest);
  
  Camellia::FieldContainer<double> coarseBasisCoefficients, fineBasisCoefficients;
  
  for (const auto & fineCellID : myFineCells)
  {
    auto coarseCellID = fineToCoarseMap.coarseCellForFineCell(fineCellID);
    auto coarseLevel  = fineToCoarseMap.coarseLevelForFineCell(fineCellID);
       
    int dummyCubatureDegree = 1;
    BasisCachePtr coarseRefCellCache = BasisCache::basisCacheForReferenceCell(coarseMesh->getTopology()->getCell(coarseCellID)->topology(), dummyCubatureDegree);
    
    auto coarseDofOrdering = coarseMesh->getElementType(coarseCellID)->trialOrderPtr;
    const bool warnAboutOffRankImports = false; // we did this import for coarse solution above, so it's fresh, and we don't need the warning…
    const auto & coarseCellAllCoefficients = coarseSolution->allCoefficientsForCellID(coarseCellID, warnAboutOffRankImports);
    
    auto coarseBasis = coarseDofOrdering->getBasis(var->ID());
    const int coarseBasisCardinality = coarseBasis->getCardinality();
    coarseBasisCoefficients.resize(coarseBasisCardinality);
    const auto & coarseDofIndices = coarseDofOrdering->getDofIndices(var->ID());
    
    for (auto dofOrdinal=0; dofOrdinal<coarseBasisCardinality; dofOrdinal++)
    {
      const auto & dofIndex = coarseDofIndices[dofOrdinal];
      coarseBasisCoefficients(dofOrdinal) = coarseCellAllCoefficients(dofIndex);
    }
    
    auto fineDofOrdering = fineMesh->getElementType(fineCellID)->trialOrderPtr;
    auto fineBasis = fineDofOrdering->getBasis(var->ID());
    const auto & fineDofIndices = fineDofOrdering->getDofIndices(var->ID());
  
    CellPtr fineCell         = fineMesh->getTopology()->getCell(fineCellID);
    
    RefinementBranch fineCellRefBranch = fineCell->refinementBranch(); // starts at level 0
    RefinementBranch fineCellRefBranchToAncestor(fineCellRefBranch.begin() + coarseLevel, fineCellRefBranch.end()); // starts at ancestor corresponding to coarse cell
    
    const SubBasisReconciliationWeights & weights = br.constrainedWeights(fineBasis, fineCellRefBranchToAncestor, coarseBasis, 0);
    
    // it should be the case that weights has entries for each coarse coefficient
    TEUCHOS_TEST_FOR_EXCEPTION(int(weights.coarseOrdinals.size()) != coarseBasisCardinality, std::invalid_argument, "invalid coarseOrdinals set");
    
    fineBasisCoefficients.resize(fineBasis->getCardinality()); // 1: cell dimension (projection below could happen for multiple cells; we're doing just one.)
    fineBasisCoefficients.initialize(0.0);
    
    const auto & fineDofOrdinals = weights.fineOrdinals;
    for (int coarseDofOrdinal=0; coarseDofOrdinal<coarseBasisCardinality; coarseDofOrdinal++)
    {
      const double & coarseCoefficient = coarseBasisCoefficients(coarseDofOrdinal);
      auto fineIter = fineDofOrdinals.begin();
      for (int i=0; i<int(fineDofOrdinals.size()); i++)
      {
        int fineDofOrdinal = *fineIter++;
        fineBasisCoefficients(fineDofOrdinal) += coarseCoefficient * weights.weights(i,coarseDofOrdinal);
      }
    }
    
    auto fineCellAllCoefficients = fineSolution->allCoefficientsForCellID(fineCellID);
    for (auto dofOrdinal=0; dofOrdinal<int(fineDofIndices.size()); dofOrdinal++)
    {
      const auto & dofIndex = fineDofIndices[dofOrdinal];
      fineCellAllCoefficients(dofIndex) = fineBasisCoefficients(0,dofOrdinal);
    }
    fineSolution->setLocalCoefficientsForCell(fineCellID, fineCellAllCoefficients);
  }
}

std::map<GlobalIndexType, std::vector<double> > readOverkillSolutionCoefficients(VlasovBackwardEulerFormulation &fineForm, const int timeStepNumber, const std::string &overkillSolutionDir)
{
  std::map<GlobalIndexType, std::vector<double> > mySolnCoefficients;

  using std::cout;
  using std::cerr;
  using std::endl;
  using std::flush;
  using std::string;
  using std::to_string;
  
  auto comm = fineForm.vlasovMesh()->Comm();
  
  const int rank = comm->MyPID();
  
  if (rank == 0)
  {
    // does directory already exist?
    struct stat sb;
    int stat_result = stat(overkillSolutionDir.c_str(), &sb);
    if (stat_result == 0 && S_ISDIR(sb.st_mode))
    {
      // file exists and is directory
    }
    else if (stat_result == 0 && ! S_ISDIR(sb.st_mode))
    {
      // file exists but is not a directory
      cerr << "The specified path for the overkill solution " << overkillSolutionDir << " exists, but is not a directory!\n";
      exit(1);
    }
    else
    {
      cerr << "Directory " << overkillSolutionDir << " does not exist; exiting.\n";
      exit(1);
    }
  }

  auto f = fineForm.f();

  // file that stores f coefficients in our proprietary format
  string solnFile_f = overkillSolutionDir + "/OverkillSolution_" + to_string(timeStepNumber) + ".soln_f";
  
  // collect all solution data onto rank 0
  auto allActiveCellIDs = fineForm.vlasovMesh()->getActiveCellIDsGlobal();
    
  // send from rank 0 to all ranks -- memory inefficient, but a simple implementation
  int numRecipients = (rank == 0) ? comm->NumProc() : 0;
  
  std::vector<int> recipients(numRecipients);
  for (int i=0; i<numRecipients; i++)
  {
    recipients[i] = i;
  }
  
  std::vector< std::vector<double> > packedSolutions(numRecipients);
  
  if (rank == 0)
  {
    std::ifstream solnFileStream(solnFile_f, std::ios::in | std::ios::binary);
    if (!solnFileStream)
    {
      cout << "Failed to open file " << solnFile_f << endl;
      exit(1);
    }
    
    GlobalIndexType cellCount;
    solnFileStream.read((char*) &cellCount, sizeof(cellCount));
    
    TEUCHOS_TEST_FOR_EXCEPTION(cellCount != int(allActiveCellIDs.size()), std::invalid_argument, "Invalid cell count");
    
    for (int cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++)
    {
      GlobalIndexType cellID;
      solnFileStream.read((char*) &cellID, sizeof(cellID));
      
      TEUCHOS_TEST_FOR_EXCEPTION(allActiveCellIDs.find(cellID) == allActiveCellIDs.end(), std::invalid_argument, "Invalid cell ID");
      
      auto dofIndices = fineForm.vlasovMesh()->getElementType(cellID)->trialOrderPtr->getDofIndices(f->ID());
      
      int dofCount;
      solnFileStream.read((char*) &dofCount, sizeof(dofCount));
      
      TEUCHOS_TEST_FOR_EXCEPTION(dofCount != int(dofIndices.size()), std::invalid_argument, "Invalid dof count");
      
      std::vector<double> coefficients(dofCount);
      
      solnFileStream.read((char*) &coefficients[0], dofCount * sizeof(double));
      
      int partitionForCell = fineForm.vlasovMesh()->partitionForCellID(cellID);
      packedSolutions[partitionForCell].insert(packedSolutions[partitionForCell].end(), coefficients.begin(), coefficients.end());
    }
    
    solnFileStream.close();
  }
  
  // send from rank 0 to all ranks
  std::vector<double> receivedPackedSolution;
  MPIWrapper::sendDataVectors(comm, recipients, packedSolutions, receivedPackedSolution);
  
  int offset = 0;
  const auto &myCellIDs = fineForm.vlasovMesh()->cellIDsInPartition();
  for (const auto &cellID : myCellIDs)
  {
    int dofCount = fineForm.vlasovMesh()->getElementType(cellID)->trialOrderPtr->getBasisCardinality(f->ID(), VOLUME_INTERIOR_SIDE_INDEX);
    
    TEUCHOS_TEST_FOR_EXCEPTION(dofCount + offset > int(receivedPackedSolution.size()), std::invalid_argument, "out of bounds");
    
    std::vector<double> coefficients(receivedPackedSolution.begin() + offset, receivedPackedSolution.begin() + offset + dofCount);
    
    offset += dofCount;
    
    mySolnCoefficients[cellID] = coefficients;
  }
  
  return mySolnCoefficients;
}

void convertOverkillSolution(VlasovBackwardEulerFormulation &fineForm, const int numTimeSteps, const std::string &overkillSolutionDir)
{
  using std::cout;
  using std::cerr;
  using std::endl;
  using std::flush;
  using std::string;
  using std::to_string;
  
  const int rank = fineForm.vlasovMesh()->Comm()->MyPID();
  
  const bool readOverkillFromFile = true;
  if (readOverkillFromFile)
  {
    if (rank == 0)
    {
      cout << "Will convert overkill solution in directory: " << overkillSolutionDir << endl;
    }
  }
  
  if (rank == 0)
  {
    // does directory already exist?
    struct stat sb;
    int stat_result = stat(overkillSolutionDir.c_str(), &sb);
    if (stat_result == 0 && S_ISDIR(sb.st_mode))
    {
      // file exists and is directory
    }
    else if (stat_result == 0 && ! S_ISDIR(sb.st_mode))
    {
      // file exists but is not a directory
      cerr << "The specified path for the overkill solution " << overkillSolutionDir << " exists, but is not a directory!\n";
      exit(1);
    }
    else
    {
      int success = mkdir(overkillSolutionDir.c_str(), S_IRWXU | S_IRWXG);
      if (success != 0)
      {
        cerr << "Failed to create directory " << overkillSolutionDir << "; exiting.\n";
        return;
      }
    }
  }
  
  int timeStepNumber = 0;
  string solnFile = overkillSolutionDir + "/OverkillSolution_" + to_string(timeStepNumber) + ".soln";
  fineForm.vlasovSolution()->loadFromHDF5(solnFile);
  
  FunctionPtr f_fine     = Function::solution(fineForm.f(),     fineForm.vlasovSolution());
  
  auto f = fineForm.f();
  
  // output just the f basis coefficients
  string solnFile_f = overkillSolutionDir + "/OverkillSolution_" + to_string(timeStepNumber) + ".soln_f";
  
  // collect all solution data onto rank 0
  auto allActiveCellIDs = fineForm.vlasovMesh()->getActiveCellIDsGlobal();
  std::set<GlobalIndexType> noCellIDs;
  auto & cellIDsToImport = (rank == 0) ? allActiveCellIDs : noCellIDs;
  
  auto writeSolution = [&] () -> void
  {
    fineForm.vlasovSolution()->importSolutionForOffRankCells(cellIDsToImport);
    
    if (rank == 0)
    {
      std::ofstream solnFileStream(solnFile_f, std::ios::out | std::ios::binary);
      if (!solnFileStream)
      {
        cout << "Failed to open file " << solnFile_f << endl;
        exit(1);
      }
      
      GlobalIndexType cellCount = allActiveCellIDs.size();
      solnFileStream.write((char*) &cellCount, sizeof(cellCount));
      
      for (const auto &cellID : allActiveCellIDs)
      {
        auto dofIndices = fineForm.vlasovMesh()->getElementType(cellID)->trialOrderPtr->getDofIndices(f->ID());
        
        solnFileStream.write((char*) &cellID, sizeof(cellID));
        
        int dofCount = dofIndices.size();
        solnFileStream.write((char*) &dofCount, sizeof(dofCount));
        
        bool warnAboutOffRank = false; // don't warn
        const auto & solnCoefficients = fineForm.vlasovSolution()->allCoefficientsForCellID(cellID, warnAboutOffRank);
        
        for (const auto &dofIndex : dofIndices)
        {
          solnFileStream.write((char*) &solnCoefficients(dofIndex), sizeof(double));
        }
      }
      
      solnFileStream.close();
      
      if (rank == 0)
      {
        cerr << "Completed conversion for time step number: " << timeStepNumber << "/" << numTimeSteps << "\r" << flush;
      }
    }
  };
  
  writeSolution();
    
  for (timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
  {
    solnFile = overkillSolutionDir + "/OverkillSolution_" + to_string(timeStepNumber) + ".soln";
    fineForm.vlasovSolution()->loadFromHDF5(solnFile);
    
    solnFile_f = overkillSolutionDir + "/OverkillSolution_" + to_string(timeStepNumber) + ".soln_f";
    
    writeSolution();
    
    if (rank == 0)
    {
      cerr << "Completed conversion for time step number: " << timeStepNumber << "/" << numTimeSteps << "\r" << flush;
    }
  }
  cerr << endl;
  
  if (rank == 0)
  {
    cout << "File conversion completed.\n";
  }
}

double overkillSolutionErrorL2Norm(VarPtr fieldVar, SolutionPtr interpolatedSolutionOnOverkillMesh, const std::map<GlobalIndexType, std::vector<double> > &localOverkillSolution, bool overkillOnly = false)
{
  // we rely on the fact that the overkill mesh is uniform; we can precompute the integrals of pairs of basis functions to accelerate the error evaluation on each cell
  
  auto mesh = interpolatedSolutionOnOverkillMesh->mesh();
  
  const auto & myCellIDs = mesh->cellIDsInPartition();
  
  // we tolerate an empty localOverkillSolution, but we don't tolerate one that is non-empty but misses owned cells
  bool haveOverkillSolution = (localOverkillSolution.size() > 0);
  
  double myErrorSquaredSum = 0;
  if (myCellIDs.size() > 0)
  {
    GlobalIndexType representativeCellID = *myCellIDs.begin();
    auto varBasis =  mesh->getElementType(representativeCellID)->trialOrderPtr->getBasis(fieldVar->ID());
    int dofCount = varBasis->getCardinality();
    
    FieldContainer<double> representativeIntegrals(dofCount,dofCount);
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, representativeCellID);
    
    FieldContainer<double> basisCoefficients_i(dofCount), basisCoefficients_j(dofCount);
    for (int i=0; i<dofCount; i++)
    {
      basisCoefficients_i.initialize(0);
      basisCoefficients_i[i] = 1.0;
      
      FunctionPtr fxn_i = Teuchos::rcp(new BasisSumFunction(varBasis, basisCoefficients_i));
      
      for (int j=0; j<dofCount; j++)
      {
        basisCoefficients_j.initialize(0);
        basisCoefficients_j[j] = 1.0;
        
        FunctionPtr fxn_j = Teuchos::rcp(new BasisSumFunction(varBasis, basisCoefficients_j));
        
        double integral = (fxn_i * fxn_j)->integrate(basisCache);
        representativeIntegrals(i,j) = integral;
      }
    }
    
    auto & dofIndices = mesh->getElementType(representativeCellID)->trialOrderPtr->getDofIndices(fieldVar->ID());
    FieldContainer<double> errorCoefficients(dofCount);
    for (const auto &cellID : myCellIDs)
    {
      if (haveOverkillSolution && !overkillOnly)
      {
        const auto &allCoefficients = interpolatedSolutionOnOverkillMesh->allCoefficientsForCellID(cellID);
        
        auto overkillEntry = localOverkillSolution.find(cellID);
        TEUCHOS_TEST_FOR_EXCEPTION(overkillEntry == localOverkillSolution.end(), std::invalid_argument, "entry for overkill solution not found");
        
        const auto &overkillSoln    = overkillEntry->second;
        TEUCHOS_TEST_FOR_EXCEPTION(int(overkillSoln.size()) != dofCount, std::invalid_argument, "overkill solution sized incorrectly");
        
        for (int dofOrdinal=0; dofOrdinal<dofCount; dofOrdinal++)
        {
          errorCoefficients(dofOrdinal) = overkillSoln[dofOrdinal] - allCoefficients(dofIndices[dofOrdinal]);
        }
      }
      else if (haveOverkillSolution)
      {
        auto overkillEntry = localOverkillSolution.find(cellID);
        TEUCHOS_TEST_FOR_EXCEPTION(overkillEntry == localOverkillSolution.end(), std::invalid_argument, "entry for overkill solution not found");
        
        const auto &overkillSoln    = overkillEntry->second;
        TEUCHOS_TEST_FOR_EXCEPTION(int(overkillSoln.size()) != dofCount, std::invalid_argument, "overkill solution sized incorrectly");
        
        for (int dofOrdinal=0; dofOrdinal<dofCount; dofOrdinal++)
        {
          errorCoefficients(dofOrdinal) = overkillSoln[dofOrdinal];
        }
      }
      else
      {
        const auto &allCoefficients = interpolatedSolutionOnOverkillMesh->allCoefficientsForCellID(cellID);
        for (int dofOrdinal=0; dofOrdinal<dofCount; dofOrdinal++)
        {
          errorCoefficients(dofOrdinal) = allCoefficients(dofIndices[dofOrdinal]);
        }
      }
      
      for (int i=0; i<dofCount; i++)
      {
        for (int j=0; j<dofCount; j++)
        {
          myErrorSquaredSum += errorCoefficients[i] * errorCoefficients[j] * representativeIntegrals(i,j);
        }
      }
    }
  }
  double globalErrorSquaredSum = MPIWrapper::sum(*mesh->Comm(), myErrorSquaredSum);
  return std::sqrt(globalErrorSquaredSum);
}

double l2NormOfUniformSolution(VarPtr fieldVar, SolutionPtr interpolatedSolutionOnOverkillMesh)
{
  std::map<GlobalIndexType, std::vector<double> > emptyOverkillSolution;
  
  return overkillSolutionErrorL2Norm(fieldVar, interpolatedSolutionOnOverkillMesh, emptyOverkillSolution);
}

void runAdaptiveComparisonExperiment(VlasovBackwardEulerFormulation &adaptiveForm, VlasovBackwardEulerFormulation &fineForm, VlasovBackwardEulerFormulation &modestForm,
                                     double refinementThreshold, bool gateRefinements, int maxElements, int maximumRefinementLevel, FunctionPtr f0, const int numTimeSteps, const double dt,
                                     const std::string &outputDir, const bool writeOverkillSolutionAndQuit, const std::string &overkillSolutionDir, bool saveViz,
                                     bool useEnergyForUnrefinements, int solutionOutputFrequency, int solutionComparisonFrequency, int overkillTimeStepsBetweenComparisons)
{
  using std::cout;
  using std::cerr;
  using std::endl;
  using std::flush;
  using std::string;
  using std::to_string;
  
  auto comm = adaptiveForm.vlasovMesh()->Comm();
  const int rank = comm->MyPID();
  
  if (writeOverkillSolutionAndQuit)
  {
    int numFineElements     = fineForm.vlasovMesh()->numActiveElements();
    int numFineDofs         = fineForm.vlasovMesh()->numGlobalDofs();
    
    if (rank == 0)
    {
      // does directory already exist?
      struct stat sb;
      int stat_result = stat(overkillSolutionDir.c_str(), &sb);
      if (stat_result == 0 && S_ISDIR(sb.st_mode))
      {
        // file exists and is directory
      }
      else if (stat_result == 0 && ! S_ISDIR(sb.st_mode))
      {
        // file exists but is not a directory
        cerr << "The specified path for the overkill solution " << overkillSolutionDir << " exists, but is not a directory!\n";
        exit(1);
      }
      else
      {
        int success = mkdir(overkillSolutionDir.c_str(), S_IRWXU | S_IRWXG);
        if (success != 0)
        {
          cerr << "Failed to create directory " << overkillSolutionDir << "; exiting.\n";
          return;
        }
      }
      
      cout << "Computing overkill solution and writing to disk in directory: " << overkillSolutionDir << "\n";
      cout << "Overkill mesh has " << numFineElements << " elements and " << numFineDofs << " degrees of freedom.\n";
      cout << "Simulation will run for " << numTimeSteps << " time steps with dt=" << dt << endl;
    }
    fineForm.setInitialConditions(f0);
    
    auto mesh = fineForm.vlasovMesh();
    auto soln = fineForm.vlasovSolution();
          
    const int minSubsteps = 3;
    const int maxSubsteps = 15;
    fineForm.setFixedPointMinSubsteps(minSubsteps);
    fineForm.setFixedPointMaxSubsteps(maxSubsteps);
    fineForm.setFixedPointTolerance(1e-6);
    
    fineForm.setUseGeometricMultigrid(true);
    
    auto f = fineForm.f();
    int timeStepNumber = 0;
    int solutionNumber = 0;
    int numSolutions = numTimeSteps / solutionOutputFrequency;
    
    // output just the f basis coefficients
    string solnFile_f = overkillSolutionDir + "/OverkillSolution_" + to_string(solutionNumber) + ".soln_f";
    
    // collect all solution data onto rank 0
    auto allActiveCellIDs = fineForm.vlasovMesh()->getActiveCellIDsGlobal();
    std::set<GlobalIndexType> noCellIDs;
    auto & cellIDsToImport = (rank == 0) ? allActiveCellIDs : noCellIDs;
    
    auto writeSolution = [&] () -> void
    {
      fineForm.vlasovSolution()->importSolutionForOffRankCells(cellIDsToImport);
      
      if (rank == 0)
      {
        std::ofstream solnFileStream(solnFile_f, std::ios::out | std::ios::binary);
        if (!solnFileStream)
        {
          cout << "Failed to open file " << solnFile_f << endl;
          exit(1);
        }
        
        GlobalIndexType cellCount = allActiveCellIDs.size();
        solnFileStream.write((char*) &cellCount, sizeof(cellCount));
        
        for (const auto &cellID : allActiveCellIDs)
        {
          auto dofIndices = fineForm.vlasovMesh()->getElementType(cellID)->trialOrderPtr->getDofIndices(f->ID());
          
          solnFileStream.write((char*) &cellID, sizeof(cellID));
          
          int dofCount = dofIndices.size();
          solnFileStream.write((char*) &dofCount, sizeof(dofCount));
          
          bool warnAboutOffRank = false; // don't warn
          const auto & solnCoefficients = fineForm.vlasovSolution()->allCoefficientsForCellID(cellID, warnAboutOffRank);
          
          for (const auto &dofIndex : dofIndices)
          {
            solnFileStream.write((char*) &solnCoefficients(dofIndex), sizeof(double));
          }
        }
        
        solnFileStream.close();
        
        if (rank == 0)
        {
          cerr << "Wrote solution number: " << solutionNumber << "/" << numSolutions << "\r" << flush;
        }
      }
    };
    
    writeSolution();
    
    for (timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
    {
      fineForm.takeTimeStep();
      
      if (timeStepNumber % solutionOutputFrequency == 0)
      {
        solutionNumber = timeStepNumber / solutionOutputFrequency;
        solnFile_f = overkillSolutionDir + "/OverkillSolution_" + to_string(solutionNumber) + ".soln_f";
        writeSolution();
      }
      
      if (rank == 0)
      {
        cerr << "Wrote solution number: " << solutionNumber << "/" << numSolutions << "\r" << flush;
      }
    }
    if (rank == 0)
    {
      cerr << "\n";
      cout << "Finished writing out overkill solution.\n";
    }
    return;
  }
  
  if (rank == 0)
  {
    cout << "Will read overkill solution from directory: " << overkillSolutionDir << endl;
  }
  
  if (rank == 0)
  {
    // does directory already exist?
    struct stat sb;
    int stat_result = stat(outputDir.c_str(), &sb);
    if (stat_result == 0 && S_ISDIR(sb.st_mode))
    {
      // file exists and is directory
    }
    else if (stat_result == 0 && ! S_ISDIR(sb.st_mode))
    {
      // file exists but is not a directory
      cerr << "The specified path for output " << outputDir << " exists, but is not a directory!\n";
      exit(1);
    }
    else
    {
      int success = mkdir(outputDir.c_str(), S_IRWXU | S_IRWXG);
      if (success != 0)
      {
        cerr << "Failed to create directory " << outputDir << "; exiting.\n";
        return;
      }
    }
  }
  
  std::string errorFilePath          = outputDir + "/fineAdaptiveErrorCompared.dat";
  std::string iterationCountFilePath = outputDir + "/fineAdaptiveIterationCounts.dat";
  std::fstream errorStream, iterationCountStream;
  if (rank == 0)
  {
    errorStream.open(errorFilePath, std::ios::out);
    iterationCountStream.open(iterationCountFilePath, std::ios::out);
    if (!errorStream.is_open())
    {
      std::cout << "Failed to open file " << errorFilePath << " on rank 0.\n";
      return;
    }
    else
    {
      std::cout << "Will write L^2 error data to " << errorFilePath << "\n";
    }
    if (!iterationCountStream.is_open())
    {
      std::cout << "Failed to open file " << iterationCountFilePath << " on rank 0.\n";
      return;
    }
    else
    {
      std::cout << "Will write iteration counts to " << iterationCountFilePath << "\n";
    }
  }
  
  MeshMap fineToModestMap(modestForm.vlasovMesh(), fineForm.vlasovMesh());
  
  // set initial conditions on adaptive form:
  adaptiveForm.setInitialConditions(f0);

  ErrorIndicatorPtr unrefinementErrorIndicator;
  if (!useEnergyForUnrefinements)
  {
    unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(adaptiveForm.vlasovSolution(), adaptiveForm.f());
  }
  RefinementStrategyPtr adaptiveRefStrategy = constructRefStrategy(adaptiveForm, refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator);
  
  auto vlasovMesh = adaptiveForm.vlasovMesh();
  auto poissonMesh     = adaptiveForm.configurationSpaceMesh();
  
  bool outputOverkillSolution = false;
  
  Teuchos::RCP<HDF5Exporter> vlasovFineViz, vlasovModestViz, vlasovAdaptiveViz;
  if (saveViz)
  {
    if (outputOverkillSolution)
    {
      vlasovFineViz     = Teuchos::rcp( new HDF5Exporter (fineForm.vlasovMesh(),   "vlasovFine",     outputDir));
    }
    vlasovModestViz   = Teuchos::rcp( new HDF5Exporter (modestForm.vlasovMesh(),   "vlasovModest",   outputDir));
    vlasovAdaptiveViz = Teuchos::rcp( new HDF5Exporter (adaptiveForm.vlasovMesh(), "vlasovAdaptive", outputDir));
  }
  
  BasisReconciliation br;
    
  // adapt to capture initial data:
  {
    FunctionPtr f_soln = Function::solution(adaptiveForm.f(), adaptiveForm.vlasovSolution());
    
    FunctionPtr f0_err = f0 - f_soln;
    int quadratureEnrichment = 4;
    
    auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(vlasovMesh, f0_err, quadratureEnrichment);
    
    double refinementThresholdForInitialData = 0.20;
    auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
    refStrategy->setMaxElements(maxElements);
    refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
    refStrategy->setMaximumRefinementLevel(maximumRefinementLevel);
    
    double f0_norm = f0->l2norm(vlasovMesh, quadratureEnrichment);
    double relErr = f0_err->l2norm(vlasovMesh, quadratureEnrichment) / f0_norm;
    
    int numInitialRefinements = 0;
    while (vlasovMesh->numActiveElements() < maxElements * 0.90)
    {
      if (rank==0)
      {
        std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << vlasovMesh->numActiveElements() << " elements and relErr is " << relErr << ".\n";
        std::cout << "Physical mesh has " << poissonMesh->numActiveElements() << " elements.\n";
      }
      const bool unrefine = true;
      const bool printToConsole = (rank == 0);
      refStrategy->refineAndUnrefine(unrefine, printToConsole);
      adaptiveForm.setInitialConditions(f0); // do projection again
      numInitialRefinements++;
      relErr = f0_err->l2norm(vlasovMesh, quadratureEnrichment) / f0_norm;
      
      if (vlasovMesh->numActiveElements() > maxElements)
      {
        // do one more refine + unrefine, and then switch to unrefinements only
        refStrategy->refineAndUnrefine(unrefine, printToConsole);
        while (vlasovMesh->numActiveElements() > maxElements)
        {
          // repeat until unrefinement gets us down to maxElements or below
          if (rank==0)
          {
            std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << vlasovMesh->numActiveElements() << " elements and relErr is " << relErr << ".\n";
            std::cout << "Physical mesh has " << poissonMesh->numActiveElements() << " elements.\n";
          }
          
          refStrategy->setRelativeErrorThreshold(1.1); // ensure one unrefinement-only pass
          refStrategy->refineAndUnrefine(unrefine, printToConsole);
          refStrategy->setRelativeErrorThreshold(refinementThresholdForInitialData);
          
          adaptiveForm.setInitialConditions(f0); // do projection again
          numInitialRefinements++;
          relErr = f0_err->l2norm(vlasovMesh, quadratureEnrichment) / f0_norm;
          
          if (numInitialRefinements > 25)
          {
            // that seems like too many; break
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Could not get initial mesh down to maxElements through refine/unrefine after 25 adaptive steps");
          }
        }
      }
    }
    if (rank==0)
    {
      std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << vlasovMesh->numActiveElements() << " elements and relErr is " << relErr << ".\n";
      std::cout << "Physical mesh has " << poissonMesh->numActiveElements() << " elements.\n";
    }
    
  }
  
  // set initial conditions for the other two forms:
  modestForm.setInitialConditions(f0);
  
  auto f = fineForm.f();
  int timeStepNumber = 0;
  int overkillStepNumber = 0;
  
  std::map<GlobalIndexType, std::vector<double> > localOverkillCoefficients;
  
  localOverkillCoefficients = readOverkillSolutionCoefficients(fineForm, overkillStepNumber, overkillSolutionDir);
  
  FunctionPtr f_modest   = Function::solution(modestForm.f(),   modestForm.vlasovSolution());
  FunctionPtr f_adaptive = Function::solution(adaptiveForm.f(), adaptiveForm.vlasovSolution());

  SolutionPtr projectedModestSoln   = Solution::solution(fineForm.vlasovMesh()->bilinearForm(), fineForm.vlasovMesh());
  SolutionPtr projectedAdaptiveSoln = Solution::solution(fineForm.vlasovMesh()->bilinearForm(), fineForm.vlasovMesh());
  
  MeshMap fineToAdaptiveMap(adaptiveForm.vlasovMesh(), fineForm.vlasovMesh());
  
  interpolateToFineSolution(f, modestForm.vlasovSolution(),   projectedModestSoln,   fineToModestMap,   br);
  interpolateToFineSolution(f, adaptiveForm.vlasovSolution(), projectedAdaptiveSoln, fineToAdaptiveMap, br);
  
  double adaptiveErr = overkillSolutionErrorL2Norm(f, projectedAdaptiveSoln, localOverkillCoefficients);
  double modestErr   = overkillSolutionErrorL2Norm(f, projectedModestSoln, localOverkillCoefficients);
  
  double adaptiveSolnNorm = l2NormOfUniformSolution(f, projectedAdaptiveSoln);
  double modestSolnNorm   = l2NormOfUniformSolution(f, projectedModestSoln);
  double fineSolnNorm     = overkillSolutionErrorL2Norm(f, fineForm.vlasovSolution(), localOverkillCoefficients, true);
  
  if (rank == 0)
  {
    std::cout << "Initial data, L^2 norm of adaptive solution: " << adaptiveSolnNorm << std::endl;
    std::cout << "Initial data, L^2 norm of modest solution:   " << modestSolnNorm << std::endl;
    std::cout << "Initial data, L^2 norm of fine solution:     " << fineSolnNorm << std::endl;
    
    std::cout << "Initial data, relative error in adaptive solution: " << adaptiveErr / fineSolnNorm << std::endl;
    std::cout << "Initial data, relative error in modest solution:   " << modestErr / fineSolnNorm << std::endl;
  }
  
  double t = 0.0;
  if (saveViz)
  {
    vlasovAdaptiveViz->exportSolution(adaptiveForm.vlasovSolution(), t);
    vlasovModestViz->exportSolution(  modestForm.vlasovSolution(),   t);
    
    if (outputOverkillSolution)
    {
      vlasovFineViz->exportSolution(    fineForm.vlasovSolution(),     t);
    }
  }
  
  int numAdaptiveElements = adaptiveForm.vlasovMesh()->numActiveElements();
  int numAdaptiveDofs     = adaptiveForm.vlasovMesh()->numGlobalDofs();
  
  int numModestElements   = modestForm.vlasovMesh()->numActiveElements();
  int numModestDofs       = modestForm.vlasovMesh()->numGlobalDofs();
  
  int numFineElements     = fineForm.vlasovMesh()->numActiveElements();
  int numFineDofs         = fineForm.vlasovMesh()->numGlobalDofs();
  
  if (rank == 0)
  {
    using std::cout;
    using std::endl;
    
    cout << "Fine mesh has     " << numFineElements     << " elements and " << numFineDofs     << " degrees of freedom.\n";
    cout << "Modest mesh has   " << numModestElements   << " elements and " << numModestDofs   << " degrees of freedom.\n";
    cout << "Adaptive mesh has " << numAdaptiveElements << " elements and " << numAdaptiveDofs << " degrees of freedom.\n";
  }
  
  if (rank == 0)
  {
    errorStream << "Step\tUniform\tAdaptive\n";
    errorStream << 0 << "\t" << modestErr / fineSolnNorm << "\t" << adaptiveErr / fineSolnNorm << "\n";
    errorStream.flush();
    
    iterationCountStream << "Time Step\tSubstep\tAdaptive\tUniform\n";
    iterationCountStream.flush();
  }
  
  const auto & adaptiveIterationCounts = adaptiveForm.iterationCounts();
  const auto & uniformIterationCounts  = modestForm.iterationCounts();
  
  const int minSubsteps = 3;
  const int maxSubsteps = 15;
  const double fixedPointTol = 1e-6;
  adaptiveForm.setFixedPointMinSubsteps(minSubsteps);
  adaptiveForm.setFixedPointMaxSubsteps(maxSubsteps);
  adaptiveForm.setFixedPointTolerance(fixedPointTol);
  modestForm.setFixedPointMinSubsteps(minSubsteps);
  modestForm.setFixedPointMaxSubsteps(maxSubsteps);
  modestForm.setFixedPointTolerance(fixedPointTol);
  fineForm.setFixedPointMinSubsteps(minSubsteps);
  fineForm.setFixedPointMaxSubsteps(maxSubsteps);
  fineForm.setFixedPointTolerance(fixedPointTol);
  
  // set up timers
  int numProcs = comm->NumProc();
  int indexBase = 0;
  Epetra_Time adaptiveTimer(*comm);
  Epetra_Time modestTimer  (*comm);
  
  double adaptiveTime = 0.;
  double modestTime   = 0.;
  Epetra_Map timeMap(numProcs,indexBase,*comm);
    
  double lastRefinementMaxElementError = 0.0;
  
  adaptiveForm.setUseGeometricMultigrid(true);
  modestForm.setUseGeometricMultigrid(true);
  
  for (timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
  {
    adaptiveTimer.ResetStartTime();
    adaptiveForm.takeTimeStep();
    
    performRefinements(lastRefinementMaxElementError, adaptiveForm, adaptiveRefStrategy, gateRefinements, refinementThreshold, maxElements);
    {
      double myAdaptiveElapsedTime        = adaptiveTimer.ElapsedTime();
      double globalMaxAdaptiveElapsedTime = 0.0;
      comm->MaxAll(&myAdaptiveElapsedTime, &globalMaxAdaptiveElapsedTime, 1);
      adaptiveTime += globalMaxAdaptiveElapsedTime;
    }
    
    numAdaptiveElements = adaptiveForm.vlasovMesh()->numActiveElements();
    numAdaptiveDofs     = adaptiveForm.vlasovMesh()->numGlobalDofs();
    if (rank == 0)
    {
      std::cout << "After time step " << timeStepNumber << ", adaptive mesh has " << numAdaptiveElements << " elements and " << numAdaptiveDofs << " degrees of freedom.\n";
    }
    
    modestTimer.ResetStartTime();
    modestForm.takeTimeStep();
    {
      double myModestElapsedTime        = modestTimer.ElapsedTime();
      double globalMaxModestElapsedTime = 0.0;
      comm->MaxAll(&myModestElapsedTime, &globalMaxModestElapsedTime, 1);
      modestTime += globalMaxModestElapsedTime;
    }
    
    t += dt;
    
    bool doComparison = (timeStepNumber % solutionComparisonFrequency == 0);
    if (doComparison)
    {
      overkillStepNumber += overkillTimeStepsBetweenComparisons;
      localOverkillCoefficients = readOverkillSolutionCoefficients(fineForm, overkillStepNumber, overkillSolutionDir);
      
      // adaptive mesh will have changed; build a fresh map.  (We could do this in a more granular fashion, by making MeshMap a RefinementObserver and updating on refinement, unrefinement.)
      fineToAdaptiveMap = MeshMap(adaptiveForm.vlasovMesh(), fineForm.vlasovMesh());
      
      interpolateToFineSolution(f, modestForm.vlasovSolution(),   projectedModestSoln,   fineToModestMap,   br);
      interpolateToFineSolution(f, adaptiveForm.vlasovSolution(), projectedAdaptiveSoln, fineToAdaptiveMap, br);
      
      if (saveViz)
      {
        vlasovAdaptiveViz->exportSolution(adaptiveForm.vlasovSolution(), t);
        vlasovModestViz->exportSolution(  modestForm.vlasovSolution(),   t);
        
        if (outputOverkillSolution)
        {
          vlasovFineViz->exportSolution(fineForm.vlasovSolution(), t);
        }
      }
          
      adaptiveErr  = overkillSolutionErrorL2Norm(f, projectedAdaptiveSoln, localOverkillCoefficients);
      modestErr    = overkillSolutionErrorL2Norm(f, projectedModestSoln, localOverkillCoefficients);
      fineSolnNorm = overkillSolutionErrorL2Norm(f, fineForm.vlasovSolution(), localOverkillCoefficients, true);
      
      if (rank == 0)
      {
        std::cout << "Time step " << timeStepNumber << ", relative error in adaptive solution: " << adaptiveErr / fineSolnNorm << std::endl;
        std::cout << "Time step " << timeStepNumber << ", relative error in modest solution:   " << modestErr / fineSolnNorm << std::endl;
        
        errorStream << timeStepNumber << "\t" << modestErr / fineSolnNorm << "\t" << adaptiveErr / fineSolnNorm << "\n";
        errorStream.flush();
      }
    }
        
    if (rank == 0)
    {
      iterationCountStream << timeStepNumber << "\t";
      const int numAdaptiveSubsteps = adaptiveIterationCounts[timeStepNumber-1].size();
      const int numUniformSubsteps = uniformIterationCounts[timeStepNumber-1].size();
      const int maxSubsteps = std::max(numAdaptiveSubsteps,numUniformSubsteps);
      for (int substep=0; substep<maxSubsteps; substep++)
      {
        iterationCountStream << substep << "\t";
        if (substep < numAdaptiveSubsteps)
          iterationCountStream << adaptiveIterationCounts[timeStepNumber-1][substep] << "\t";
        else
          iterationCountStream << -1 << "\t";
        if (substep < numUniformSubsteps)
          iterationCountStream << uniformIterationCounts[timeStepNumber-1][substep]  << "\n";
        else
          iterationCountStream << -1  << "\n";
        iterationCountStream.flush();
      }
    }
  }
  if (rank == 0)
  {
    errorStream.close();
  }
  
  // write summary to file
  if (rank == 0)
  {
    std::string summaryFilePath        = outputDir + "/overkillComparisonSummary.txt";
    std::fstream summaryStream;
    if (rank == 0)
    {
      summaryStream.open(summaryFilePath, std::ios::out);
      if (!summaryStream.is_open())
      {
        std::cout << "Failed to open file " << summaryFilePath << " on rank 0.\n";
        return;
      }
    }
    
    summaryStream << "Ran for " << numTimeSteps << " time steps with dt=" << dt << endl;
    summaryStream << "Overkill mesh:        " << numFineElements << " elements, " << numFineDofs << " degrees of freedom.\n";
    summaryStream << "Uniform mesh:         " << numModestElements << " elements, " << numModestDofs << " degrees of freedom.\n";
    summaryStream << "Final adaptive mesh:  " << numAdaptiveElements << " elements, " << numAdaptiveDofs << " degrees of freedom.\n";
    summaryStream << "Field polynomial order of non-overkill meshes (one less than H^1 order): " << adaptiveForm.vlasovMesh()->globalDofAssignment()->getInitialH1Order()[0]-1 << endl;
    summaryStream << "Relative L^2 error for adaptive solution at final time: " << adaptiveErr / fineSolnNorm << endl;
    summaryStream << "Relative L^2 error for uniform solution at final time:  " << modestErr / fineSolnNorm << endl;
    summaryStream << "Total solve time for adaptive solution (max across MPI ranks): " << adaptiveTime << endl;
    summaryStream << "Total solve time for uniform solution (max across MPI ranks):  " << modestTime << endl;
    
    summaryStream.close();
    std::cout << "Wrote run summary to " << summaryFilePath << "\n";
  }
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  const int rank = mpiSession.getRank();
  
  // block off variables so that everything gets destructed before the finalize() call at the end.
  {
    Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
    
    mpiSession.barrier();
    
    const bool useSerendipity = false; // this driver does not support the use of Serendipity bases
    
    double k_x    = 0.2; // wavelength
    double u0     = 3.0; // initial drift velocity
    double gamma  = 1e-3; // initial perturbation
    double sigma  = 1.0; // temperature
    double n0     = 1.0; // density
    
    double q      =  1.0;
    double eps0   =  1.0;
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  2 * M_PI / k_x;
    
    double v0     = std::numeric_limits<double>::infinity();
    double vWidth = std::numeric_limits<double>::infinity();
    
    bool enforceQuasiNeutrality = true;
    
    bool useAdaptivity = false;
  //  double refinementThreshold   = 0.999999; // essentially, refine just the maximum error element (and any that thanks to solution symmetry have same error)
  //  double unrefinementThreshold = 1.000001; // similarly, unrefine just the minimum-error parent element
    double refinementThreshold = 0.80; // refine elements with at least 80% of the maximum per-element error
    int maxElements = 64 * 64; // do unrefinements to keep total mesh size smaller than this
    bool gateRefinements = true;
    
    int xElements = 8;
    int vElements = 8;
    int numUniformRefinements = 0;
    int delta_k   = 2;
    double dt_max = -1.0;
    int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
    int minSubsteps = 5;  // for fixed-point iteration.
    int maxSubsteps = 15; // for fixed-point iteration.
    double fixedPointTol = 1e-8;
    int numTimeSteps = 500;
    int outputFrequency = 10; // number of time steps to take before outputting viz.
    std::string problemChoiceString = "two-stream"; // "bump-on-tail", "cold-diode" the other options
    ProblemChoice problemChoice = TWO_STREAM;
    int maximumRefinementLevel = std::numeric_limits<int>::max(); // default to no maximum refinement level
    bool runOverkillVersusAdaptiveComparison = false;
    bool useEnergyForUnrefinements = false;
    bool saveOverkillSolutionAndQuit = false;
    bool convertOverkillSolutionFiles = false;
    std::string overkillSolutionDir = "";
    int overkillRefinementLevel = 5; // would take a coarse 8x8 mesh to 256x256
    int overkillH1Order = 5;
    int uniformMeshRefinementLevel = 3; // would take a coarse 8x8 mesh to 64x64 (this is used for overkill experiment)
    bool saveViz = true;
    std::string outputDir = "/tmp";
    int solutionOutputFrequency = 1;
    int overkillTimeStepsBetweenComparisons = 1;
    int solutionComparisonFrequency = 1;
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-8;
    double iterativeSolveMaxIters  = 10000;
    int maxDofsForDirectSolve = 20000;
    int fixedCountRefinementTarget = -1; // set to positive number to used fixed-count ref strategy instead of greedy strategy
    
    int vlasovTraceEnrichment = -1; // if left unset by user, will default to 0 for all problems but cold diode, where it is 1 (at least for now).
    double coldDiodeSigma = -1.0; // will be set automatically below if not specified by user
    double bcResolutionTolerance = -1;  // only applies if doing an adaptive solve (and so far only for cold diode); -1 indicates to spend the whole element budget to resolve; unrefinements at later time steps will tend to reduce the BC resolution over time, though experimentally the loss is often negligible (but this will depend on precise settings!).
    std::string resultsFilePrefix = "";
    
    std::string resultsFile = "";
    
    cmdp.setOption("resultsFile", &resultsFile);
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("sigma", &coldDiodeSigma); // sigma for cold diode -- ignored for other problems
    cmdp.setOption("numUniformRefinements", &numUniformRefinements);
    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);
    cmdp.setOption("dt", &dt_max );
    cmdp.setOption("resultsFilePrefix", &resultsFilePrefix);
    cmdp.setOption("u0", &u0);
    cmdp.setOption("v0", &v0, "minimum velocity coordinate in domain");
    cmdp.setOption("vWidth", &vWidth, "span of velocity coordinates in domain");
    cmdp.setOption("maxElements", &maxElements);
    cmdp.setOption("maximumRefinementLevel", &maximumRefinementLevel);
    cmdp.setOption("minSubsteps", &minSubsteps);
    cmdp.setOption("maxSubsteps", &maxSubsteps);
    cmdp.setOption("fixedPointTol", &fixedPointTol);
    cmdp.setOption("viz", "noViz", &saveViz);
    cmdp.setOption("gateRefinements", "dontGateRefinements", &gateRefinements);
    cmdp.setOption("numTimeSteps", &numTimeSteps);
    cmdp.setOption("outputFrequency", &outputFrequency);
    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    cmdp.setOption("problem", &problemChoiceString);
    cmdp.setOption("refinementThreshold", &refinementThreshold);
    cmdp.setOption("fixedCountRefTarget", &fixedCountRefinementTarget);
    cmdp.setOption("useEnergyForUnrefinements", "useL2ProjectionForUnrefinements", &useEnergyForUnrefinements);
    cmdp.setOption("runOverkillVersusAdaptiveComparison", "runStandard", &runOverkillVersusAdaptiveComparison);
    cmdp.setOption("saveOverkillSolutionAndQuit", "dontSolveOverkill", &saveOverkillSolutionAndQuit);
    cmdp.setOption("solutionOutputFrequency", &solutionOutputFrequency);
    cmdp.setOption("solutionComparisonFrequency", &solutionComparisonFrequency);
    cmdp.setOption("overkillTimeStepsBetweenComparisons", &overkillTimeStepsBetweenComparisons);
    cmdp.setOption("convertOverkillSolution", "dontConvertOverkillSolution", &convertOverkillSolutionFiles);
    cmdp.setOption("overkillSolutionDir", &overkillSolutionDir);
    cmdp.setOption("overkillMeshRefinementLevel", &overkillRefinementLevel);
    cmdp.setOption("overkillH1Order", &overkillH1Order);
    cmdp.setOption("uniformMeshRefinementLevel", &uniformMeshRefinementLevel);
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
    cmdp.setOption("maxDofsForDirectSolve", &maxDofsForDirectSolve);
    cmdp.setOption("vlasovTraceEnrichment", &vlasovTraceEnrichment);
    cmdp.setOption("bcResolutionTolerance", &bcResolutionTolerance);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
  #ifdef HAVE_MPI
      MPI_Finalize();
  #endif
      return -1;
    }
    
    problemChoice = getProblemChoice(problemChoiceString);
    
    // for cold diode, we non-dimensionalize; we modify the below accordingly so that we can output dimensional values to results file.
    double velocityScaling = 1.;
    double spatialScaling  = 1.;
    double EScaling        = 1.;
    double nScaling         = 1.;
    double potentialScaling = 1.;
        
    if (vWidth == std::numeric_limits<double>::infinity())
    {
      vWidth = 20.0;
    }
    if (v0 == std::numeric_limits<double>::infinity())
    {
      v0 = -10.0;
    }
    
    bool usePeriodicBCs = true;
    
    if (dt_max < 0)
    {
      dt_max = 1e-1;
    }
    
    auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
    std::vector<double> vxVertices;
    double vx = v0;
    for (int i=0; i<=vElements; i++)
    {
      vx = v0 + (i * vWidth) / vElements;
      vxVertices.push_back(vx);
    }
    
    double dt = dt_max;
    VlasovBackwardEulerFormulation form(q, m, useSerendipity, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k);
//    poissonConfig->configurationSpaceMesh()->bilinearForm()->printTrialTestInteractions();
    form.refineUniformly(numUniformRefinements);
    form.setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    setPoissonBC(form, x0);
    
    RefinementStrategyPtr formRefStrategy;
    
    form.setIterativeSolveOutputLevel(iterativeSolveOutputLevel);
    form.setIterativeSolveTolerance(iterativeSolveTolerance);
    std::vector<double> cgTols(maxSubsteps,iterativeSolveTolerance);
    if (maxSubsteps > 3)
    {
      // use looser tolerances for the first couple steps
      cgTols[0] = iterativeSolveTolerance * 100.0;
      cgTols[1] = iterativeSolveTolerance * 10.0;
      form.setIterativeSolveTolerances(cgTols);
    }
    form.setIterativeSolveMaxIterations(iterativeSolveMaxIters);
    form.setFixedPointTolerance(fixedPointTol);
    
    FunctionPtr f0;
    FunctionPtr cos_kx       = Teuchos::rcp(new Cos_ax(k_x));
    
    // the following flags are considered only if useAdaptivity is true:
    bool adaptToCaptureInitialBCs  = false; // true for cold diode
    bool adaptToCaptureInitialData = true;
    if (problemChoice == TWO_STREAM)
    {
      double n = 0.5 * n0;
      FunctionPtr fM_u0        = Teuchos::rcp(new Maxwellian1V( u0, sigma, n) );
      FunctionPtr fM_minus_u0  = Teuchos::rcp(new Maxwellian1V(-u0, sigma, n) );
      f0                       = (fM_u0 + fM_minus_u0) * (1. + gamma * cos_kx);
      adaptToCaptureInitialData = true;
    }
    
    if (convertOverkillSolutionFiles)
    {
      auto finePoissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
      auto fineSpatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
      VlasovBackwardEulerFormulation fineForm(q, m, useSerendipity, finePoissonConfig, fineSpatialMeshTopo, dt, vxVertices, overkillH1Order, delta_k);
      fineForm.refineUniformly(numUniformRefinements + overkillRefinementLevel);
      fineForm.setEnforceQuasiNeutrality(enforceQuasiNeutrality);
      setPoissonBC(fineForm, x0);
      convertOverkillSolution(fineForm, numTimeSteps, overkillSolutionDir);
      return 0; // exit
    }
    else if (runOverkillVersusAdaptiveComparison)
    {
      auto     finePoissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
      auto adaptivePoissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
      auto   modestPoissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
      
      // although these MeshTopo's begin identical, they get modified during uniform refinements; we want a distinct one for each formulation instance.
      auto fineSpatialMeshTopo     = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
      auto adaptiveSpatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
      auto modestSpatialMeshTopo   = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
      VlasovBackwardEulerFormulation     fineForm(q, m, useSerendipity,     finePoissonConfig, fineSpatialMeshTopo,     dt, vxVertices, overkillH1Order, delta_k);
      VlasovBackwardEulerFormulation adaptiveForm(q, m, useSerendipity, adaptivePoissonConfig, adaptiveSpatialMeshTopo, dt, vxVertices, overkillH1Order, delta_k);
      VlasovBackwardEulerFormulation   modestForm(q, m, useSerendipity,   modestPoissonConfig, modestSpatialMeshTopo,   dt, vxVertices, overkillH1Order, delta_k);
      
      fineForm.refineUniformly(numUniformRefinements + overkillRefinementLevel);
      fineForm.setEnforceQuasiNeutrality(enforceQuasiNeutrality);
      setPoissonBC(fineForm, x0);
      
      adaptiveForm.refineUniformly(numUniformRefinements);
      adaptiveForm.setEnforceQuasiNeutrality(enforceQuasiNeutrality);
      setPoissonBC(adaptiveForm, x0);
      
      modestForm.refineUniformly(numUniformRefinements + uniformMeshRefinementLevel);
      modestForm.setEnforceQuasiNeutrality(enforceQuasiNeutrality);
      setPoissonBC(modestForm, x0);
      
      fineForm.setIterativeSolveMaxIterations(iterativeSolveMaxIters);
      fineForm.setIterativeSolveOutputLevel(0); // run quietly
      fineForm.setIterativeSolveTolerance(iterativeSolveTolerance);
      
      modestForm.setIterativeSolveMaxIterations(iterativeSolveMaxIters);
      modestForm.setIterativeSolveOutputLevel(0); // run quietly
      modestForm.setIterativeSolveTolerance(iterativeSolveTolerance);
      
      adaptiveForm.setIterativeSolveMaxIterations(iterativeSolveMaxIters);
      adaptiveForm.setIterativeSolveOutputLevel(0); // run quietly
      adaptiveForm.setIterativeSolveTolerance(iterativeSolveTolerance);
      
      const int maxAdaptiveMeshElements = modestForm.vlasovMesh()->numActiveElements();
      
      if (overkillSolutionDir == "")
      {
        std::cout << "overkillSolutionDir must be specified!\n";
        exit(1);
      }
      
      runAdaptiveComparisonExperiment(adaptiveForm, fineForm, modestForm, refinementThreshold, gateRefinements, maxAdaptiveMeshElements, numUniformRefinements+overkillRefinementLevel, f0, numTimeSteps, dt, outputDir, saveOverkillSolutionAndQuit, overkillSolutionDir, saveViz, useEnergyForUnrefinements, solutionOutputFrequency, solutionComparisonFrequency, overkillTimeStepsBetweenComparisons );
      
      const bool printAllocationReport = true;
      Camellia::finalize(printAllocationReport);
      
      return 0; // exit
    }
    
    using std::vector;
    using std::pair;
    vector<pair<pair<FunctionPtr,FunctionPtr>,int> > bcFunctionsToResolve; // < <exact function, solution function>, tag> -- only used when useAdaptivity is true, and only for cold diode
            
    form.setInitialConditions(f0);
    
    // adapt to capture initial data:
    if (useAdaptivity && adaptToCaptureInitialData)
    {
      FunctionPtr f_soln = Function::solution(form.f(), form.vlasovSolution());
      
      FunctionPtr f0_err = f0 - f_soln;
      int quadratureEnrichment = 4;
      
      auto vlasovMesh = form.vlasovMesh();
      auto poissonMesh = form.configurationSpaceMesh();
      auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(vlasovMesh, f0_err, quadratureEnrichment);
      
      double refinementThresholdForInitialData = 0.20;
      auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
      refStrategy->setMaxElements(maxElements);
      ErrorIndicatorPtr unrefinementErrorIndicator;
      if (!useEnergyForUnrefinements)
      {
        unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(form.vlasovSolution(), form.f());
      }
      refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
      refStrategy->setMaximumRefinementLevel(maximumRefinementLevel);
      refStrategy->setMaxP(H1Order);
      
      double f0_norm =     f0->l2norm(vlasovMesh, quadratureEnrichment);
      double relErr  = f0_err->l2norm(vlasovMesh, quadratureEnrichment) / f0_norm;
      
      int numInitialRefinements = 0;
      while (vlasovMesh->numActiveElements() < maxElements * 0.90)
      {
        if (rank==0)
        {
          std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << vlasovMesh->numActiveElements() << " elements and relErr is " << relErr << ".\n";
          std::cout << "Physical mesh has " << poissonMesh->numActiveElements() << " elements.\n";
        }
        const bool unrefine = true;
        const bool printToConsole = (rank == 0);
        refStrategy->refineAndUnrefine(unrefine, printToConsole);
        // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
        form.configurationSpaceMesh()->repartitionAndRebuild();
        
        form.setInitialConditions(f0); // do projection again
        numInitialRefinements++;
        relErr = f0_err->l2norm(vlasovMesh, quadratureEnrichment) / f0_norm;
        
        if (vlasovMesh->numActiveElements() > maxElements)
        {
          // do one more refine + unrefine, and then switch to unrefinements only
          refStrategy->refineAndUnrefine(unrefine, printToConsole);
          while (vlasovMesh->numActiveElements() > maxElements)
          {
            // repeat until unrefinement gets us down to maxElements or below
            if (rank==0)
            {
              std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << vlasovMesh->numActiveElements() << " elements and relErr is " << relErr << ".\n";
              std::cout << "Physical mesh has " << poissonMesh->numActiveElements() << " elements.\n";
            }
            
            refStrategy->setRelativeErrorThreshold(1.1); // ensure one unrefinement-only pass
            refStrategy->refineAndUnrefine(unrefine, printToConsole);
            refStrategy->setRelativeErrorThreshold(refinementThresholdForInitialData);
            
            form.setInitialConditions(f0); // do projection again
            numInitialRefinements++;
            relErr = f0_err->l2norm(vlasovMesh, quadratureEnrichment) / f0_norm;
            
            if (numInitialRefinements > 25)
            {
              // that seems like too many; break
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Could not get initial mesh down to maxElements through refine/unrefine after 25 adaptive steps");
            }
          }
        }
      }
      if (rank==0)
      {
        std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << vlasovMesh->numActiveElements() << " elements and relErr is " << relErr << ".\n";
        std::cout << "Physical mesh has " << poissonMesh->numActiveElements() << " elements.\n";
      }
      // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
      poissonMesh->repartitionAndRebuild();
      
      if (fixedCountRefinementTarget > 0)
      {
        ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(form.vlasovSolution());
        formRefStrategy = Teuchos::rcp(new FixedCountRefinementStrategy(errorIndicator, fixedCountRefinementTarget) );
        gateRefinements = false;
      }
      else
      {
        formRefStrategy = constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator);
      }
    }
    else if (useAdaptivity && adaptToCaptureInitialBCs)
    {
      for (auto bcPair : bcFunctionsToResolve)
      {
        FunctionPtr bcFxnExact = bcPair.first.first;
        FunctionPtr bcFxnSoln  = bcPair.first.second;
        int bcTag = bcPair.second;
        
        const std::string & tagName = DIRICHLET_SET_TAG_NAME;
        FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
        
        // project onto vlasovSolution:
        std::map<int,FunctionPtr> bcMap;
        bcMap[form.t_n_hat()->ID()] = bcFxnExact;
        form.vlasovSolution()->projectOntoMesh(bcMap);
        
        FunctionPtr bc_err = bcFxnExact * bcRestriction - bcFxnSoln * bcRestriction;
        int quadratureEnrichment = 4;
        
        auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(form.vlasovMesh(), bc_err, quadratureEnrichment);
        
        double refinementThresholdForInitialData = 0.80;
        auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
  //      refStrategy->setMaxElements(maxElements);
        refStrategy->setMaxP(H1Order);
        
        double bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
        double relErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
        
        int numBCRefinements = 0;
        int numVlasovElements = form.vlasovMesh()->numActiveElements();
        while ((relErr > bcResolutionTolerance) && (numVlasovElements < maxElements))
        {
          if (rank==0)
          {
            std::cout << "After " << numBCRefinements++ << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
            std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
          }
          
          refStrategy->refine();
          
          // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
          form.configurationSpaceMesh()->repartitionAndRebuild();
          
          // project the BC onto the mesh again
          form.vlasovSolution()->projectOntoMesh(bcMap);
          
          bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
          relErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
          
          numVlasovElements = form.vlasovMesh()->numActiveElements();
        }
        if (rank==0)
        {
          std::cout << "After " << numBCRefinements << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
          std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
        }
      }
      
      ErrorIndicatorPtr unrefinementErrorIndicator;
      if (!useEnergyForUnrefinements)
      {
        unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(form.vlasovSolution(), form.f());
      }
      
      if (fixedCountRefinementTarget > 0)
      {
        ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(form.vlasovSolution());
        formRefStrategy = Teuchos::rcp(new FixedCountRefinementStrategy(errorIndicator, fixedCountRefinementTarget) );
        gateRefinements = false;
      }
      else
      {
        formRefStrategy = constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator);
      }
    }
    
    std::ostringstream suffix;
    suffix << "u0" << u0;
    if (useAdaptivity)
    {
      suffix << "_AMR_threshold" << refinementThreshold;
    }
    suffix << "_dt" << dt_max;
    suffix << "_H1Order" << H1Order;
    suffix << "_delta_k" << delta_k;

    // adjust x and v element counts according to the number of uniform refinements requested.
    xElements <<= numUniformRefinements;
    vElements <<= numUniformRefinements;

    suffix << "_" << xElements << "x" << vElements;

    const std::string problemPrefix = "TwoStream";
    
    std::ostringstream vlasovSS;
    vlasovSS << problemPrefix << "Vlasov" << suffix.str();

    std::ostringstream poissonSS;
    poissonSS << problemPrefix << "Poisson" << suffix.str();
    
    if (resultsFile == "")
    {
      std::ostringstream resultsFileStream;
      resultsFileStream << outputDir << "/";
      if (resultsFilePrefix == "")
      {
        resultsFileStream << "results_" << problemPrefix << "_" << suffix.str();
      }
      else
      {
        resultsFileStream << resultsFilePrefix;
      }
      
      resultsFileStream << "_" << numTimeSteps << "_timeSteps";
      resultsFileStream << ".dat";
      resultsFile = resultsFileStream.str();
    }
    
    auto getSpatialMeshFilePath = [&]() {
      std::ostringstream spatialMeshInfoPathStringStream;
      spatialMeshInfoPathStringStream << outputDir << "/";
      spatialMeshInfoPathStringStream << "mesh1D_" << problemPrefix << suffix.str();
      
      spatialMeshInfoPathStringStream << numTimeSteps << "_timeSteps";
      spatialMeshInfoPathStringStream << ".dat";
      
      return spatialMeshInfoPathStringStream.str();
    };
    
    auto getTimeStepReportFilePath = [&]() {
      std::ostringstream timeStepReportPathStringStream;
      timeStepReportPathStringStream << outputDir << "/";
      timeStepReportPathStringStream << "timeStepReport_" << problemPrefix << suffix.str();
      
      timeStepReportPathStringStream << numTimeSteps << "_timeSteps";
      timeStepReportPathStringStream << ".dat";
      
      return timeStepReportPathStringStream.str();
    };
    
    auto refReportFilePath = getTimeStepReportFilePath();
    std::ofstream refReportFileStream(refReportFilePath, std::ios::out);
    
    refReportFileStream << "Vlasov Elements\tVlasov Dofs\tPoisson Elements\tPoisson Dofs\tBC err\tEnergy Error\n";
    refReportFileStream.close();
    
    auto writeTimeStepReport = [&](const int numElementsVlasov, const int numDofsVlasov, const int numElementsPoisson, const int numDofsPoisson, const double bcErr, const double energyError) {
      auto refReportFilePath = getTimeStepReportFilePath();
      std::ofstream refReportFileStream(refReportFilePath, std::ios_base::app);
      refReportFileStream << numElementsVlasov << "\t" << numDofsVlasov << "\t" << numElementsPoisson << "\t" << numDofsPoisson << "\t" << bcErr << "\t" << energyError << std::endl;
      refReportFileStream.close();
    };
    
    auto getBCResolution = [&]() {
      double bcRelErr = 0.0;
      for (auto bcPair : bcFunctionsToResolve)
      {
        FunctionPtr bcFxnExact = bcPair.first.first;
        FunctionPtr bcFxnSoln  = bcPair.first.second;
        int bcTag = bcPair.second;
        
        const std::string & tagName = DIRICHLET_SET_TAG_NAME;
        FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
        
        FunctionPtr bc_err = bcFxnExact * bcRestriction - bcFxnSoln * bcRestriction;
        int quadratureEnrichment = 4;
        
        double bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
        bcRelErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
      }
      return bcRelErr;
    };
    
    auto reportBCResolutionAndTimeStepComplete = [&](const int &timeStepNumber, const double &bcRelErr, const double &energyError) {
      auto numVlasovDofs       = form.vlasovMesh()->numGlobalDofs();
      auto vlasovElementCount  = form.vlasovMesh()->numActiveElements();
      auto poissonElementCount = form.configurationSpaceMesh()->numActiveElements();
      auto poissonDofCount     = form.configurationSpaceMesh()->numGlobalDofs();
      
      if (rank == 0)
      {
        std::cout << "BC resolution, relative error in L^2 norm: " << bcRelErr << std::endl;
        writeTimeStepReport(vlasovElementCount, numVlasovDofs, poissonElementCount, poissonDofCount, bcRelErr, energyError);
        std::cout << "Time step " << timeStepNumber << " complete.\n";
      }
    };
   
    
    const std::string directoryPath = outputDir;
    
    if (resultsFile == "")
    {
      std::ostringstream resultsFileStream;
      resultsFileStream << directoryPath << "/" << problemPrefix << "_results" << suffix.str() << ".dat";
      resultsFile = resultsFileStream.str();
    }
    
    Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz;
    if (saveViz)
    {
      vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovSS.str(), directoryPath));
      poissonViz = Teuchos::rcp(new HDF5Exporter (form.configurationSpaceMesh(), poissonSS.str(), directoryPath) );
    }
    
    std::ostringstream E_magnitudeFilePath;
    E_magnitudeFilePath << directoryPath << "/" << "E_magnitude" << suffix.str() << ".dat";
    std::fstream E_magnitudeOut;
    if ((rank == 0) && (problemChoice == TWO_STREAM))
    {
      E_magnitudeOut.open(E_magnitudeFilePath.str(), std::ios::out);
      if (!E_magnitudeOut.is_open())
      {
        std::cout << "Failed to open file " << E_magnitudeFilePath.str() << " on rank 0.\n";
        return -1;
      }
      else
      {
        std::cout << "Will write integral of |E|^2 to " << E_magnitudeFilePath.str() << "\n";
      }
    }
    
    double t = 0.0;
    if (saveViz)
    {
      poissonViz->exportSolution(form.configurationSpaceSolution(),t);
      vlasovViz->exportSolution(form.vlasovSolution(),t);
    }
    
    FunctionPtr E_soln = Function::solution(form.configurationSpaceFormulation()->E_fieldVar(0), form.configurationSpaceSolution());
    FunctionPtr E_squared = E_soln * E_soln;
    FunctionPtr abs_E_soln = Teuchos::rcp( new AbsFunction(E_soln) );
    
    const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
    auto solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form.vlasovSolution(), form.f(), form.configurationSpaceMesh()) );
        
    auto density  = solutionMoments->solutionMomentFunction(0);
    auto momentum = solutionMoments->solutionMomentFunction(1);
    auto velocity = momentum / density;
    
    bool previousSolveUsedGeometricMultigrid = false;
    bool previousSolveUsedDirect = false;
    
    double lastRefinementMaxElementError = 0.0;
    
    for (int timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
    {
      auto numVlasovDofs = form.vlasovSolution()->mesh()->numGlobalDofs();
      if (numVlasovDofs > maxDofsForDirectSolve)
      {
        if ((rank == 0) && (!previousSolveUsedGeometricMultigrid))
        {
          std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
        }
        form.setUseGeometricMultigrid(true);
        previousSolveUsedGeometricMultigrid = true;
        previousSolveUsedDirect = false;
      }
      else
      {
        if ((rank == 0) && (previousSolveUsedGeometricMultigrid))
        {
          std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
        }
        form.setUseGeometricMultigrid(false);
        previousSolveUsedGeometricMultigrid = false;
        previousSolveUsedDirect = true;
      }
      
      form.takeTimeStep();
      t += dt;
      
      const double bcRelErr = getBCResolution(); // returns -1 if BCs to resolve weren't specified
      performRefinements(lastRefinementMaxElementError, form, formRefStrategy, gateRefinements, refinementThreshold, maxElements);
      const double energyError = (formRefStrategy != Teuchos::null) ? formRefStrategy->getEnergyError(formRefStrategy()->getNumRefinements() - 1) : -1;
      
      if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
      {
        if (saveViz)
        {
          solutionMoments->update(); // update to get latest density, velocity
          vlasovViz->exportSolution(form.vlasovSolution(),t);
          poissonViz->exportSolution(form.configurationSpaceSolution(), {density,velocity}, {"n_e","v"},t);
        }
        
        reportBCResolutionAndTimeStepComplete(timeStepNumber,bcRelErr,energyError);
        
        double E_mag = E_squared->integrate(form.configurationSpaceMesh());
        if ((rank == 0) && (problemChoice == TWO_STREAM))
        {
          std::cout << "|E|^2 integral = " << E_mag << std::endl;
          E_magnitudeOut << t << "\t" << E_mag << std::endl << std::flush;
        }
      }
    } // end time step loop
    if ((rank == 0) && (problemChoice == TWO_STREAM))
    {
      E_magnitudeOut.close();
    }
    
    FunctionPtr potential = Function::solution(form.configurationSpaceFormulation()->VE_fieldVar(), form.configurationSpaceSolution());
        
    // output to results file:
    {
      // output (physical/dimensional) results
      // ultimately, we want E-field, velocity, and some other things
      // to begin with, let's output the E-field values computed at quadrature points at final time
            
      auto allActivePhysicalCells = form.configurationSpaceMesh()->getActiveCellIDsGlobal();
      
      form.configurationSpaceSolution()->importSolutionForOffRankCells(allActivePhysicalCells);
      
      // update solutionMoments once more and gather data to rank 0, so we can access it in the block below
      solutionMoments->update();
      solutionMoments->gatherOnRankZero();
      
      if (rank == 0)
      {
        std::map<double,double> E_values; // map from x coordinate to final E field value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> ne_values; // map from x coordinate to final density value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> velocity_values; // map from x coordinate to final velocity value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> potential_values; // map from x coordinate to final potential value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> pointWeight_values; // map from x coordinate to cell measure-weighted quadrature weight
        GlobalIndexType firstPhysicalCellID = *allActivePhysicalCells.begin();
        
        const int cubatureEnrichmentDegree = 5; // over-integrate to make up for undermeasurement on coarse initial meshes
        auto basisCache = BasisCache::basisCacheForCell(form.configurationSpaceMesh(), firstPhysicalCellID, false, cubatureEnrichmentDegree); // false: not test versus test
        std::vector<GlobalIndexType> physicalCellIDsVector(allActivePhysicalCells.begin(),allActivePhysicalCells.end());
        
        const int numPoints = basisCache->getRefCellPoints().extent_int(0);
        auto & cubWeights = basisCache->getCubatureWeightsFC();
        
        auto physicalCellNodes = form.configurationSpaceMesh()->physicalCellNodes(physicalCellIDsVector);
        basisCache->setPhysicalCellNodes(physicalCellNodes, physicalCellIDsVector, false);
        
        FieldContainer<double> E_values_FC(physicalCellIDsVector.size(), numPoints);
        E_soln->values(E_values_FC, basisCache);
        
        FieldContainer<double> potential_values_FC(physicalCellIDsVector.size(), numPoints);
        potential->values(potential_values_FC, basisCache);
        
        FieldContainer<double> density_values_FC(physicalCellIDsVector.size(), numPoints);
        density->values(density_values_FC, basisCache);
        
        FieldContainer<double> velocity_values_FC(physicalCellIDsVector.size(), numPoints);
        velocity->values(velocity_values_FC, basisCache);
        
        const FieldContainer<double> & physicalQuadraturePoints = basisCache->getPhysicalCubaturePointsFC();
        
        const int numCells         = physicalQuadraturePoints.dimension(0);
        
        for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
        {
          const double cellWidth = physicalCellNodes(cellOrdinal,1,0) - physicalCellNodes(cellOrdinal,0,0);
          for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
          {
            const double & x = physicalQuadraturePoints(cellOrdinal,pointOrdinal,0);
            const double pointWeight = cubWeights(pointOrdinal) * cellWidth / 2.0; // 2.0: width of the reference line.
            
            E_values[x]           = E_values_FC(cellOrdinal,pointOrdinal);
            potential_values[x]   = potential_values_FC(cellOrdinal,pointOrdinal);
            ne_values[x]          = density_values_FC(cellOrdinal,pointOrdinal);
            velocity_values[x]    = velocity_values_FC(cellOrdinal,pointOrdinal);
            pointWeight_values[x] = pointWeight;
          }
        }
        
        std::ofstream resultsFileStream(resultsFile, std::ios::out);
        
        resultsFileStream << "x\tquadrature weight\tE\tphi\tne\tvx\n";
        resultsFileStream << setprecision(15);
        
        for (const auto & pointEntry : pointWeight_values)
        {
          const auto x           = pointEntry.first * spatialScaling;
          const auto pointWeight = pointEntry.second * spatialScaling;
          const auto E           = E_values[pointEntry.first]         * EScaling;
          const auto potential   = potential_values[pointEntry.first] * potentialScaling;
          const auto ne          = ne_values[pointEntry.first]        * nScaling;
          const auto velocity    = velocity_values[pointEntry.first]  * velocityScaling;
          
          resultsFileStream << x           << "\t";
          resultsFileStream << pointWeight << "\t";
          resultsFileStream << E           << "\t";
          resultsFileStream << potential   << "\t";
          resultsFileStream << ne          << "\t";
          resultsFileStream << velocity    << "\n";
        }
        
        resultsFileStream.close();
        
        auto spatialMeshFile = getSpatialMeshFilePath();
        std::ofstream spatialMeshFileStream(spatialMeshFile, std::ios::out);
        // let's collect the spatial nodes and sort them
        std::set<double> spatialNodes;
        for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
        {
          spatialNodes.insert(physicalCellNodes(cellOrdinal,0,0)); // (min x, min t)
          spatialNodes.insert(physicalCellNodes(cellOrdinal,1,0)); // (max x, min t)
        }
        resultsFileStream << "x\n";
        for (const auto & node : spatialNodes)
        {
          spatialMeshFileStream << node << "\n";
        }
        spatialMeshFileStream.close();
      }
    }
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
