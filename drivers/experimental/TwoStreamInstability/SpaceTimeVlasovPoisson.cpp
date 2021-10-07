//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "VlasovPoissonSpaceTimeFormulation.h"
#include "ErrorIndicator.h"
#include "ERKTableau.h"
#include "ExpFunction.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "Projector.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"

#include <sys/stat.h>
#include <sys/types.h>

using namespace Camellia;

class Maxwellian1T1V : public TFunction<double>
{
  double _u;
  double _sigma;
  double _n;
public:
  Maxwellian1T1V(double u, double sigma, double n)
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
//    std::cout << "weight: " << weight << std::endl;
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,2); // vx is the third dimension for 1D1T1V
        const double exponent = - 0.5 * (vx - _u) * (vx - _u) / (_sigma * _sigma);
        values(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
      }
    }
  }
};

static const std::string S_TWO_STREAM   = "two-stream";
static const std::string S_BUMP_ON_TAIL = "bump-on-tail";
static const std::string S_COLD_DIODE   = "cold-diode";

enum ProblemChoice
{
  TWO_STREAM,
  BUMP_ON_TAIL,
  COLD_DIODE
};

static const std::map<std::string, ProblemChoice> stringToProblemChoice {{S_BUMP_ON_TAIL,BUMP_ON_TAIL},{S_TWO_STREAM,TWO_STREAM}, {S_COLD_DIODE,COLD_DIODE}};
static const std::map<ProblemChoice, std::string> problemChoiceToString {{BUMP_ON_TAIL,S_BUMP_ON_TAIL},{TWO_STREAM,S_TWO_STREAM},{COLD_DIODE,S_COLD_DIODE}};

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

void checkMeshParities( MeshPtr mesh)
{
  const auto & myCellIDs = mesh->cellIDsInPartition();
  
  auto globalDofAssignment = mesh->globalDofAssignment();
  
  for (auto & cellID : myCellIDs)
  {
    CellPtr cell = mesh->getTopology()->getCell(cellID);
    int sideCount = cell->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      auto neighborInfo = cell->getNeighborInfo(sideOrdinal, mesh->getTopology());
      const IndexType & neighborCellID = neighborInfo.first;
      const auto & mySideOrdinalInNeighbor = neighborInfo.second;
      if (neighborCellID != -1)
      {
        int myParity       = globalDofAssignment->sideParity(cellID, sideOrdinal);
        int neighborParity = globalDofAssignment->sideParity(neighborCellID, mySideOrdinalInNeighbor);
        TEUCHOS_TEST_FOR_EXCEPTION(myParity != -neighborParity, std::invalid_argument, "inconsistent parities");
      }
    }
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
   
//    std::cout << "fine cell: " << fineCellID << "; coarse cell: " << coarseCellID << " (level: " << coarseLevel << ")\n";
    
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
    
//    std::cout << "coarseBasisCoefficients:\n" << coarseBasisCoefficients;
    
    auto fineDofOrdering = fineMesh->getElementType(fineCellID)->trialOrderPtr;
    auto fineBasis = fineDofOrdering->getBasis(var->ID());
    const auto & fineDofIndices = fineDofOrdering->getDofIndices(var->ID());
  
    CellPtr fineCell         = fineMesh->getTopology()->getCell(fineCellID);
    
    RefinementBranch fineCellRefBranch = fineCell->refinementBranch(); // starts at level 0
    RefinementBranch fineCellRefBranchToAncestor(fineCellRefBranch.begin() + coarseLevel, fineCellRefBranch.end()); // starts at ancestor corresponding to coarse cell
    
    const SubBasisReconciliationWeights & weights = br.constrainedWeights(fineBasis, fineCellRefBranchToAncestor, coarseBasis, 0);
    
    // it should be the case that weights has entries for each coarse coefficient
    TEUCHOS_TEST_FOR_EXCEPTION(weights.coarseOrdinals.size() != coarseBasisCardinality, std::invalid_argument, "invalid coarseOrdinals set");
    
    fineBasisCoefficients.resize(fineBasis->getCardinality()); // 1: cell dimension (projection below could happen for multiple cells; we're doing just one.)
    fineBasisCoefficients.initialize(0.0);
    
    const auto & fineDofOrdinals = weights.fineOrdinals;
    for (int coarseDofOrdinal=0; coarseDofOrdinal<coarseBasisCardinality; coarseDofOrdinal++)
    {
      const double & coarseCoefficient = coarseBasisCoefficients(coarseDofOrdinal);
      auto fineIter = fineDofOrdinals.begin();
      for (int i=0; i<fineDofOrdinals.size(); i++)
      {
        int fineDofOrdinal = *fineIter++;
        fineBasisCoefficients(fineDofOrdinal) += coarseCoefficient * weights.weights(i,coarseDofOrdinal);
      }
    }
    
    auto fineCellAllCoefficients = fineSolution->allCoefficientsForCellID(fineCellID);
    for (auto dofOrdinal=0; dofOrdinal<fineDofIndices.size(); dofOrdinal++)
    {
      const auto & dofIndex = fineDofIndices[dofOrdinal];
      fineCellAllCoefficients(dofIndex) = fineBasisCoefficients(0,dofOrdinal);
    }
    fineSolution->setLocalCoefficientsForCell(fineCellID, fineCellAllCoefficients);
  }
}

std::map<GlobalIndexType, std::vector<double> > readOverkillSolutionCoefficients(VlasovPoissonSpaceTimeFormulation &fineForm, const std::string &overkillSolutionDir)
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
  string solnFile_f = overkillSolutionDir + "/OverkillSolution.soln_f";
  
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
    
    TEUCHOS_TEST_FOR_EXCEPTION(cellCount != allActiveCellIDs.size(), std::invalid_argument, "Invalid cell count");
    
    for (int cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++)
    {
      GlobalIndexType cellID;
      solnFileStream.read((char*) &cellID, sizeof(cellID));
      
      TEUCHOS_TEST_FOR_EXCEPTION(allActiveCellIDs.find(cellID) == allActiveCellIDs.end(), std::invalid_argument, "Invalid cell ID");
      
      auto dofIndices = fineForm.vlasovMesh()->getElementType(cellID)->trialOrderPtr->getDofIndices(f->ID());
      
      int dofCount;
      solnFileStream.read((char*) &dofCount, sizeof(dofCount));
      
      TEUCHOS_TEST_FOR_EXCEPTION(dofCount != dofIndices.size(), std::invalid_argument, "Invalid dof count");
      
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
    
    TEUCHOS_TEST_FOR_EXCEPTION(dofCount + offset > receivedPackedSolution.size(), std::invalid_argument, "out of bounds");
    
    std::vector<double> coefficients(receivedPackedSolution.begin() + offset, receivedPackedSolution.begin() + offset + dofCount);
    
    offset += dofCount;
    
    mySolnCoefficients[cellID] = coefficients;
  }
  
  return mySolnCoefficients;
}

void convertOverkillSolution(VlasovPoissonSpaceTimeFormulation &fineForm, const std::string &overkillSolutionDir)
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
    }
  };
  
  writeSolution();
  
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
        TEUCHOS_TEST_FOR_EXCEPTION(overkillSoln.size() != dofCount, std::invalid_argument, "overkill solution sized incorrectly");
        
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
        TEUCHOS_TEST_FOR_EXCEPTION(overkillSoln.size() != dofCount, std::invalid_argument, "overkill solution sized incorrectly");
        
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

void runAdaptiveComparisonExperiment(VlasovPoissonSpaceTimeFormulation &adaptiveForm,
                                     VlasovPoissonSpaceTimeFormulation &fineForm,
                                     VlasovPoissonSpaceTimeFormulation &modestForm,
                                     double refinementThreshold, bool gateRefinements,
                                     int maxElements, int maximumRefinementLevel,
                                     FunctionPtr f0,
                                     const std::string &outputDir, const bool writeOverkillSolutionAndQuit,
                                     const std::string &overkillSolutionDir, bool saveViz,
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
    }
    fineForm.setVlasovInitialConditions(f0);
    
    auto mesh = fineForm.vlasovMesh();
    auto soln = fineForm.vlasovSolution();
          
    const int numSubsteps = 3;
    fineForm.setSubstepsForFixedPoint(numSubsteps);
    
    auto f = fineForm.f();
    int timeStepNumber = 0;
    int solutionNumber = 0;
    
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
      }
    };
    
    solnFile_f = overkillSolutionDir + "/OverkillSolution.soln_f";
    fineForm.solve();
    
    writeSolution();
    
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
  std::fstream errorStream;
  if (rank == 0)
  {
    errorStream.open(errorFilePath, std::ios::out);
    if (!errorStream.is_open())
    {
      std::cout << "Failed to open file " << errorFilePath << " on rank 0.\n";
      return;
    }
    else
    {
      std::cout << "Will write L^2 error data to " << errorFilePath << "\n";
    }
  }
  
  MeshMap fineToModestMap(modestForm.vlasovMesh(), fineForm.vlasovMesh());
  
  // set initial conditions on adaptive form:
  adaptiveForm.setVlasovInitialConditions(f0);

  ErrorIndicatorPtr unrefinementErrorIndicator;
  if (!useEnergyForUnrefinements)
  {
    unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(adaptiveForm.vlasovSolution(), adaptiveForm.f());
  }
  adaptiveForm.setRefinementThresholds(refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator, gateRefinements);
  
  auto vlasovMesh  = adaptiveForm.vlasovMesh();
  auto poissonMesh = adaptiveForm.poissonMesh();
  
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
      adaptiveForm.setVlasovInitialConditions(f0); // do projection again
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
          
          adaptiveForm.setVlasovInitialConditions(f0); // do projection again
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
  modestForm.setVlasovInitialConditions(f0);
  
  auto f = fineForm.f();
  
  std::map<GlobalIndexType, std::vector<double> > localOverkillCoefficients;
  
  localOverkillCoefficients = readOverkillSolutionCoefficients(fineForm, overkillSolutionDir);
  
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
  
  const int numSubsteps = 3;
  adaptiveForm.setSubstepsForFixedPoint(numSubsteps);
  modestForm.setSubstepsForFixedPoint(numSubsteps);
  fineForm.setSubstepsForFixedPoint(numSubsteps);
  
  // set up timers
  int numProcs = comm->NumProc();
  int indexBase = 0;
  Epetra_Time adaptiveTimer(*comm);
  Epetra_Time modestTimer  (*comm);
  
  double adaptiveTime = 0.;
  double modestTime   = 0.;
  Epetra_Map timeMap(numProcs,indexBase,*comm);
  
  adaptiveTimer.ResetStartTime();
  adaptiveForm.solve();
  {
    double myAdaptiveElapsedTime        = adaptiveTimer.ElapsedTime();
    double globalMaxAdaptiveElapsedTime = 0.0;
    comm->MaxAll(&myAdaptiveElapsedTime, &globalMaxAdaptiveElapsedTime, 1);
    adaptiveTime += globalMaxAdaptiveElapsedTime;
  }
  
  modestTimer.ResetStartTime();
  modestForm.solve();
  {
    double myModestElapsedTime        = modestTimer.ElapsedTime();
    double globalMaxModestElapsedTime = 0.0;
    comm->MaxAll(&myModestElapsedTime, &globalMaxModestElapsedTime, 1);
    modestTime += globalMaxModestElapsedTime;
  }
  
  // adaptive mesh will have changed; build a fresh map.  (We could do this in a more granular fashion, by making MeshMap a RefinementObserver and updating on refinement, unrefinement.)
  fineToAdaptiveMap = MeshMap(adaptiveForm.vlasovMesh(), fineForm.vlasovMesh());
  
  interpolateToFineSolution(f, modestForm.vlasovSolution(),   projectedModestSoln,   fineToModestMap,   br);
  interpolateToFineSolution(f, adaptiveForm.vlasovSolution(), projectedAdaptiveSoln, fineToAdaptiveMap, br);
  
  if (saveViz)
  {
    vlasovAdaptiveViz->exportSolution(adaptiveForm.vlasovSolution());
    vlasovModestViz->exportSolution(  modestForm.vlasovSolution());
    
    if (outputOverkillSolution)
    {
      vlasovFineViz->exportSolution(fineForm.vlasovSolution());
    }
  }
  
  adaptiveErr  = overkillSolutionErrorL2Norm(f, projectedAdaptiveSoln, localOverkillCoefficients);
  modestErr    = overkillSolutionErrorL2Norm(f, projectedModestSoln, localOverkillCoefficients);
  fineSolnNorm = overkillSolutionErrorL2Norm(f, fineForm.vlasovSolution(), localOverkillCoefficients, true);
  
  if (rank == 0)
  {
    std::cout << "Relative error in adaptive solution: " << adaptiveErr / fineSolnNorm << std::endl;
    std::cout << "Relative error in modest solution:   " << modestErr / fineSolnNorm << std::endl;
    
    errorStream << modestErr / fineSolnNorm << "\t" << adaptiveErr / fineSolnNorm << "\n";
    errorStream.flush();
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
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  mpiSession.barrier();
  
  bool implicit = true;
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
  double t0     =  0.0;
  double tWidth =  1.0;
  double v0     = -10.0;
  double vWidth =  20.0;
  
  bool enforceQuasiNeutrality = true;
  
  bool useAdaptivity = false;
//  double refinementThreshold   = 0.999999; // essentially, refine just the maximum error element (and any that thanks to solution symmetry have same error)
//  double unrefinementThreshold = 1.000001; // similarly, unrefine just the minimum-error parent element
  double refinementThreshold = 0.80; // refine elements with at least 80% of the maximum per-element error
  int maxElements = 64 * 64; // do unrefinements to keep total mesh size smaller than this
  bool gateRefinements = true;
  
  int xElements = 8;
  int vElements = 8;
  int tElements = 10;
  int delta_k   = 2;
  int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  int outputFrequency = 10; // number of time steps to take before outputting viz.
  double timeStepC0 = -1.0;
  bool useL2Trace = true;
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
  bool useGeometricMultigrid = true;
  int numUniformRefinements = 0;
  int numSubsteps = 5; // for fixed-point iteration.  (May want more for coarser meshes?)
  int vlasovTraceEnrichment = -1; // if left unset by user, will default to 0 for all problems but cold diode, where it is 1 (at least for now).
  
  std::string resultsFile = "";
  
  cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
  cmdp.setOption("xElements", &xElements);
  cmdp.setOption("tElements", &tElements);
  cmdp.setOption("vElements", &vElements);
  cmdp.setOption("numUniformRefinements", &numUniformRefinements);
  cmdp.setOption("H1Order", &H1Order);
  cmdp.setOption("resultsFile", &resultsFile);
  cmdp.setOption("u0", &u0);
  cmdp.setOption("maxElements", &maxElements);
  cmdp.setOption("maximumRefinementLevel", &maximumRefinementLevel);
  cmdp.setOption("numSubsteps", &numSubsteps);
  cmdp.setOption("viz", "noViz", &saveViz);
  cmdp.setOption("gateRefinements", "dontGateRefinements", &gateRefinements);
  cmdp.setOption("outputFrequency", &outputFrequency);
  cmdp.setOption("outputDir", &outputDir);
  cmdp.setOption("timeStepC0", &timeStepC0);
  cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
  cmdp.setOption("problem", &problemChoiceString);
  cmdp.setOption("refinementThreshold", &refinementThreshold);
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
  cmdp.setOption("useGeometricMultigrid", "useDirectSolver", &useGeometricMultigrid);
  cmdp.setOption("vlasovTraceEnrichment", &vlasovTraceEnrichment);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  const int rank = mpiSession.getRank();
  
  problemChoice = getProblemChoice(problemChoiceString);
  
  // for cold diode, we non-dimensionalize; we modify the below accordingly so that we can output dimensional values to results file.
  double velocityScaling = 1.;
  double timeScaling     = 1.;
  double spatialScaling  = 1.;
  double EScaling        = 1.;
  
  if (vlasovTraceEnrichment == -1)
  {
    if (problemChoice == COLD_DIODE)
    {
      // this helps resolve the very fine BCs
      vlasovTraceEnrichment = 1;
    }
    else
    {
      // theory says that H^1 order is enough for convergence rates (if BCs are resolved)
      vlasovTraceEnrichment = 0;
    }
  }
  
  bool usePeriodicBCs = true;
  if (problemChoice == BUMP_ON_TAIL)
  {
    k_x = 0.3;
    xWidth =  2 * M_PI / k_x;
    u0 = 4.5;
    n0 = 0.1;
    usePeriodicBCs = true;
  }
  else if (problemChoice == COLD_DIODE)
  {
    const double e               = 1.602176634e-19; // electron's electric charge in Coulombs
    const double m_e             =  9.10938356e-31;  // m_e, mass of electron in kg
    
    const double W               = 1e4;       // beam intensity from SAND report [eV]
    const double J               = 9.33582e4; // current density from SAND report [A/m^2]
    
    // v_beam is defined by 1/2 m v^2 = W ==> v_beam = sqrt( 2 W /m ):
    const double v_beam          = sqrt(2 * W * e / m_e ); // e converts from eV to Joules
    
    const double q0              = e;
    const double m0              = m_e;
    
    velocityScaling = v_beam;
    timeScaling     = 1.97485663509e-9; // final time from SAND report [s]
    spatialScaling  = velocityScaling * timeScaling; // [m]
    EScaling        = m0 * velocityScaling / (q0 * timeScaling);
    
    std::cout << "Characteristic E-field strength E_0 : " << EScaling << std::endl;
    std::cout << "spatial scaling : " << spatialScaling << std::endl;
    
    const double permittivity    = 8.85418782e-12; // permittivity of free space
    
    const double nScaling        = J / (e * v_beam); // number density of plasma
    
    const double omega_p = sqrt( q0 * q0 * nScaling / (m0 * permittivity));
    
    xWidth = 0.01 / spatialScaling;
    u0     = 1.0;
    n0     = 1.0;
    sigma  = 1e6 / velocityScaling;
    q      = -1.0;
    
    // eps0 is a weight that enters VlasovPoissonSpaceTimeFormulation; we use this to scale the problem:
    eps0   = 1.0 / (omega_p * omega_p * timeScaling * timeScaling);
    usePeriodicBCs = false;
    enforceQuasiNeutrality = false;
    v0     = 0.80;
    vWidth = 0.40;
    
    std::cout << "sigma = " << sigma << std::endl;
    std::cout << "u0 = " << u0 << std::endl;
    std::cout << "n0 = " << n0 << std::endl;
  }
  
  VlasovPoissonSpaceTimeFormulation::VlasovPoissonSpaceTimeFormulationChoice formulationChoice;
  using FormulationChoice = Camellia::VlasovPoissonSpaceTimeFormulation::VlasovPoissonSpaceTimeFormulationChoice;
  formulationChoice = FormulationChoice::ULTRAWEAK;
  
  VlasovPoissonSpaceTimeFormulation form(formulationChoice, q, eps0, m, x0, xWidth, xElements, t0, tWidth, tElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, vlasovTraceEnrichment);
  
  form.setUseGeometricMultigrid(useGeometricMultigrid);
  
  if (numUniformRefinements > 0)
  {
    for (int i=0; i<numUniformRefinements; i++)
    {
      const auto & activeCells  = form.vlasovMesh()->getActiveCellIDsGlobal();
      form.vlasovMesh()->hRefine(activeCells);
      
      xElements *= 2;
      tElements *= 2;
      vElements *= 2;
      
      // it should be the case that Poisson mesh is automatically refined when the Vlasov mesh is
    }
  }
  
  FunctionPtr f0;
  FunctionPtr cos_kx       = Teuchos::rcp(new Cos_ax(k_x));
  
  bool adaptToCaptureInitialData = true;
  if (problemChoice == TWO_STREAM)
  {
    double n = 0.5 * n0;
    FunctionPtr fM_u0        = Teuchos::rcp(new Maxwellian1T1V( u0, sigma, n) );
    FunctionPtr fM_minus_u0  = Teuchos::rcp(new Maxwellian1T1V(-u0, sigma, n) );
    f0                       = (fM_u0 + fM_minus_u0) * (1. + gamma * cos_kx);
    adaptToCaptureInitialData = true;
  }
  else if (problemChoice == BUMP_ON_TAIL)
  {
    FunctionPtr fM_0 = Teuchos::rcp(new Maxwellian1T1V( 0.0, 1.0, 1.0-n0) );
    FunctionPtr fM_b = Teuchos::rcp(new Maxwellian1T1V(  u0, 0.5,     n0) );
    f0               = (fM_0 + 0.5 * fM_b) * (1. + gamma * cos_kx);
    std::cout << "f0 for bump-on-tail: " << f0->displayString() << "\n";
    adaptToCaptureInitialData = true;
  }
  else if (problemChoice == COLD_DIODE)
  {
    // experimentally: start with a beam that goes all the way across
//    FunctionPtr fM_u0 = Teuchos::rcp(new Maxwellian1T1V( u0, sigma, n0) );
//    f0 = fM_u0;
//    adaptToCaptureInitialData = true;
    
    f0 = Function::zero(); // start with vacuum
    adaptToCaptureInitialData = false; // zero solution is always exactly captured…
    // TODO: we *will* want to refine to capture the (rather narrow) inflow BC.  But that will require something different than the adaptation for f0 for the other problems.
  }
  
  if (convertOverkillSolutionFiles)
  {
    VlasovPoissonSpaceTimeFormulation fineForm(formulationChoice, q, eps0, m, x0, xWidth, xElements, t0, tWidth, tElements, v0, vWidth, vElements, overkillH1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, vlasovTraceEnrichment);
    // TODO: do uniform refinements on fineForm meshes: numUniformRefinements+overkillRefinementLevel
    
    convertOverkillSolution(fineForm, overkillSolutionDir);
    return 0; // exit
  }
  else if (runOverkillVersusAdaptiveComparison)
  {
    VlasovPoissonSpaceTimeFormulation fineForm    (formulationChoice, q, eps0, m, x0, xWidth, xElements, t0, tWidth, tElements, v0, vWidth, vElements, overkillH1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, vlasovTraceEnrichment);
    // TODO: do uniform refinements on fineForm meshes: numUniformRefinements+overkillRefinementLevel
    VlasovPoissonSpaceTimeFormulation adaptiveForm(formulationChoice, q, eps0, m, x0, xWidth, xElements, t0, tWidth, tElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, vlasovTraceEnrichment);
    // TODO: do uniform refinements on adaptiveForm meshes: numUniformRefinements
    VlasovPoissonSpaceTimeFormulation modestForm  (formulationChoice, q, eps0, m, x0, xWidth, xElements, t0, tWidth, tElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, vlasovTraceEnrichment);
    // TODO: do uniform refinements on modestForm meshes: numUniformRefinements+uniformMeshRefinementLevel
    
//    {
//      // DEBUGGING/PROFILING
//      // how much memory does the fine form MeshTopology take up?
//      if (rank == 0)
//      {
//        std::cout << "overkill mesh, approximate footprint for base MeshTopology:\n";
//        fineForm.vlasovMesh()->getTopology()->baseMeshTopology()->printApproximateMemoryReport();
//      }
//    }
    
    const int maxAdaptiveMeshElements = modestForm.vlasovMesh()->numActiveElements();
    
    if (overkillSolutionDir == "")
    {
      std::cout << "overkillSolutionDir must be specified!\n";
      exit(1);
    }
    
    runAdaptiveComparisonExperiment(adaptiveForm, fineForm, modestForm, refinementThreshold, gateRefinements, maxAdaptiveMeshElements, numUniformRefinements+overkillRefinementLevel, f0, outputDir, saveOverkillSolutionAndQuit, overkillSolutionDir, saveViz, useEnergyForUnrefinements, solutionOutputFrequency, solutionComparisonFrequency, overkillTimeStepsBetweenComparisons );
    
    const bool printAllocationReport = true;
    Camellia::finalize(printAllocationReport);
    
    return 0; // exit
  }
  
  if (!usePeriodicBCs)
  {
    if (problemChoice == COLD_DIODE)
    {
      // pin potential to zero at left and right:
      form.setPotentialBCs(Function::zero(), Function::zero());
      // set a Maxwellian at the inflow:
      FunctionPtr fM_u0 = Teuchos::rcp(new Maxwellian1T1V( u0, sigma, n0) );
      form.setVlasovInitialAndInflowConditions(Function::zero(), fM_u0, Function::zero()); // the right BC is only for negative velocity, so won't actually get triggered on our mesh
    }
  }
  else
  {
    form.setVlasovInitialConditions(f0);
  }
  
  std::ostringstream suffix;
  if (useL2Trace)
  {
    suffix << "_L2Trace";
  }
  else
  {
    suffix << "_H1Trace";
  }
  suffix << "_u0" << u0;
  suffix << "_H1Order" << H1Order;
  suffix << "_" << xElements << "x" << tElements << "x" << vElements;
  
  const std::string problemPrefix = problemChoiceString;
  
  if (resultsFile == "")
  {
    std::ostringstream resultsFileStream;
    resultsFileStream << problemPrefix << "_results" << suffix.str() << ".dat";
    resultsFile = resultsFileStream.str();
  }
  
  std::ostringstream vlasovSS;
  vlasovSS << problemPrefix << "Vlasov" << suffix.str();
  
  std::ostringstream poissonSS;
  poissonSS << problemPrefix << "Poisson" << suffix.str();
  
//  std::cout << "vlasov name: " << vlasovSS.str() << std::endl;
//  std::cout << "poisson name: " << poissonSS.str() << std::endl;
  
  const std::string directoryPath = outputDir;
  
  Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz;
  if (saveViz)
  {
    vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovSS.str(), directoryPath));
    poissonViz = Teuchos::rcp(new HDF5Exporter (form.poissonMesh(), poissonSS.str(), directoryPath) );
  }
    
  FunctionPtr E_soln = Function::solution(form.E(), form.poissonSolution());
  FunctionPtr E_squared = E_soln * E_soln;
  FunctionPtr abs_E_soln = Teuchos::rcp( new AbsFunction(E_soln) );
  
  form.setSubstepsForFixedPoint(numSubsteps);
  
//  form.vlasovSolution()->setWriteMatrixToFile(true, "/tmp/A_Vlasov.dat");
  
  form.solve();
  
  if (saveViz)
  {
    poissonViz->exportSolution(form.poissonSolution());
    vlasovViz->exportSolution(form.vlasovSolution());
  }
  
  // output (physical/dimensional) results
  // ultimately, we want E-field, velocity, and some other things
  // to begin with, let's output the E-field values computed at quadrature points at final time

  {
    // find elements that match final time
    
    // the di() methods are 1-based:
    static const int SPATIAL_INDEX  = 1;
    static const int TEMPORAL_INDEX = 2;
    static const int VELOCITY_INDEX = 3;
    
    // SpatialFilter's components are zero-based
    static const int SPATIAL_INDEX_ZERO_BASED  = SPATIAL_INDEX  - 1;
    static const int TEMPORAL_INDEX_ZERO_BASED = TEMPORAL_INDEX - 1;
    static const int VELOCITY_INDEX_ZERO_BASED = VELOCITY_INDEX - 1;
    
    SpatialFilterPtr finalTimeFilter = SpatialFilter::matchingComponent(TEMPORAL_INDEX_ZERO_BASED, t0 + tWidth);
    
    auto poissonMeshTopo = form.poissonMesh()->getTopology()->baseMeshTopology();
    auto finalTimeSideIndices = poissonMeshTopo->getBoundarySidesThatMatch(finalTimeFilter);
    std::set<GlobalIndexType> physicalCellsFinalTime;
    
    for (const auto & sideEntityIndex : finalTimeSideIndices)
    {
      IndexType physicalCellID = poissonMeshTopo->getFirstCellForSide(sideEntityIndex).first;
      if (!poissonMeshTopo->isParent(physicalCellID))
      {
        physicalCellsFinalTime.insert(physicalCellID);
      }
    }
    
    form.poissonSolution()->importSolutionForOffRankCells(physicalCellsFinalTime);
    
    if (rank == 0)
    {
      std::map<double,double> E_values; // map from x coordinate to final E field value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
      std::map<double,double> pointWeight_values; // map from x coordinate to cell measure-weighted quadrature weight
      GlobalIndexType firstPhysicalCellID = *physicalCellsFinalTime.begin();
      
      auto basisCache = BasisCache::basisCacheForCell(form.poissonMesh(), firstPhysicalCellID);
      std::vector<GlobalIndexType> physicalCellIDsVector(physicalCellsFinalTime.begin(),physicalCellsFinalTime.end());
      
      auto cubWeights = basisCache->getCubatureWeights();
      auto cubPoints  = basisCache->getRefCellPoints();
      
      // we want to rewrite cubPoints/weights so that the temporal dimension only extends over a thin slice of the element
      // for now, we assume that the temporal extent of each element at final time is the same
      auto spatialCubWeights = cubWeights.getTensorComponent(0);
      auto spatialPoints     = cubPoints.getTensorComponent(0);
      auto temporalWeights   = cubWeights.getTensorComponent(1);
      auto temporalPoints    = cubPoints.getTensorComponent(1);
      
      using Data       = Intrepid2::Data<double, Camellia::DeviceType>;
      using DataHost   = Intrepid2::Data<double, Kokkos::HostSpace>;
      using ScalarView = Kokkos::DynRankView<double, Camellia::DeviceType>;
      
      auto temporalWeightsView = temporalWeights.getUnderlyingView<1>();
      
      const int numTemporalPoints = temporalPoints.extent_int(0); // (P,D)
      Kokkos::parallel_for(numTemporalPoints, KOKKOS_LAMBDA(const int &pointOrdinal)
      {
        // map points from (-1,1) to (.98,1)
        // x_new = 0.01 * x_old + 0.99
        auto x_old = temporalPoints(pointOrdinal,0);
        temporalPoints(pointOrdinal,0) = .01 * x_old + 0.99;
        
        // quadrature weights should likewise be scaled by .01
        temporalWeightsView(pointOrdinal) *= .01;
      });
      
      DataHost spatialCubatureWeightsHost(spatialCubWeights);
      DataHost temporalCubatureWeightsHost(temporalWeights);
      
      std::vector<ScalarView> refPointsComponents {spatialPoints, temporalPoints};
      std::vector<Data> cubWeightsComponents {spatialCubWeights, temporalWeights};
      Intrepid2::TensorPoints<double, Camellia::DeviceType> finalTimeRefPoints(refPointsComponents);
      Intrepid2::TensorData<double, Camellia::DeviceType> finalTimeCubWeights(cubWeightsComponents);
      
      basisCache->setRefCellPoints(finalTimeRefPoints, finalTimeCubWeights);
      
      auto physicalCellNodes = form.poissonMesh()->physicalCellNodes(physicalCellIDsVector);
      basisCache->setPhysicalCellNodes(physicalCellNodes, physicalCellIDsVector, false);
      
      FieldContainer<double> E_values_FC(physicalCellIDsVector.size(), finalTimeRefPoints.extent_int(0));
      E_soln->values(E_values_FC, basisCache);
      
      const FieldContainer<double> & physicalQuadraturePoints = basisCache->getPhysicalCubaturePointsFC();
      
      const int numCells         = physicalQuadraturePoints.dimension(0);
      const int numSpatialPoints = spatialPoints.extent_int(0);
      
      for (int cellOrdinal=0; cellOrdinal<physicalCellIDsVector.size(); cellOrdinal++)
      {
        const double cellWidth = physicalCellNodes(cellOrdinal,1,0) - physicalCellNodes(cellOrdinal,0,0);
        const double temporalWidth = physicalCellNodes(cellOrdinal,2,1) - physicalCellNodes(cellOrdinal,0,1);
        for (int spatialPointOrdinal=0; spatialPointOrdinal<numSpatialPoints; spatialPointOrdinal++)
        {
          const int firstTensorPointOrdinal = spatialPointOrdinal; // formula is temporalPointOrdinal * numSpatialPoints + spatialPointOrdinal; this is the point ordinal corresponding to temporalPointOrdinal = 0.
          const double & x = physicalQuadraturePoints(cellOrdinal,firstTensorPointOrdinal,0);
          double E_integral = 0.0;   // integral in time
          double one_integral = 0.0; // integrate 1 so that we can divide by that to get the average
          for (int temporalPointOrdinal=0; temporalPointOrdinal < numTemporalPoints; temporalPointOrdinal++)
          {
            const int pointOrdinal = temporalPointOrdinal * numSpatialPoints + spatialPointOrdinal;
            const double & E = E_values_FC(cellOrdinal,pointOrdinal);
            
            const double & temporalWeight = temporalCubatureWeightsHost(temporalPointOrdinal);
            E_integral   += E * temporalWeight;
            one_integral += temporalWeight;
          }
          const double pointWeight = spatialCubatureWeightsHost(spatialPointOrdinal) * cellWidth;
          
          E_values[x]           = E_integral / one_integral;
          pointWeight_values[x] = pointWeight;
        }
      }
      
      std::ofstream resultsFileStream(resultsFile, std::ios::out);
      
      resultsFileStream << "x\tquadrature weight\tE\n";
      resultsFileStream << setprecision(15);
      
      for (const auto & pointEntry : pointWeight_values)
      {
        const auto x           = pointEntry.first * spatialScaling;
        const auto pointWeight = pointEntry.second * spatialScaling;
        const auto E           = E_values[pointEntry.first] * EScaling;
        
        resultsFileStream << x           << "\t";
        resultsFileStream << pointWeight << "\t";
        resultsFileStream << E           << "\n";
      }
      
      resultsFileStream.close();
    }
  }
  
  BasisCachePtr physicalBasisCache = BasisCache::basisCacheForCell(form.poissonMesh(), 0);
  
  return 0;
}
