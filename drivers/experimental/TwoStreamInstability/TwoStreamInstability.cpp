//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "Boltzmann1D1VFormulation.h"
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
//    std::cout << "weight: " << weight << std::endl;
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

void checkMesh( MeshPtr mesh)
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

std::map<GlobalIndexType, std::vector<double> > readOverkillSolutionCoefficients(Boltzmann1D1VFormulation &fineForm, const int timeStepNumber, const std::string &overkillSolutionDir)
{
  std::map<GlobalIndexType, std::vector<double> > mySolnCoefficients;

  using std::cout;
  using std::cerr;
  using std::endl;
  using std::flush;
  using std::string;
  using std::to_string;
  
  auto comm = fineForm.boltzmannMesh()->Comm();
  
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
  auto allActiveCellIDs = fineForm.boltzmannMesh()->getActiveCellIDsGlobal();
    
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
      
      auto dofIndices = fineForm.boltzmannMesh()->getElementType(cellID)->trialOrderPtr->getDofIndices(f->ID());
      
      int dofCount;
      solnFileStream.read((char*) &dofCount, sizeof(dofCount));
      
      TEUCHOS_TEST_FOR_EXCEPTION(dofCount != dofIndices.size(), std::invalid_argument, "Invalid dof count");
      
      std::vector<double> coefficients(dofCount);
      
      solnFileStream.read((char*) &coefficients[0], dofCount * sizeof(double));
      
      int partitionForCell = fineForm.boltzmannMesh()->partitionForCellID(cellID);
      packedSolutions[partitionForCell].insert(packedSolutions[partitionForCell].end(), coefficients.begin(), coefficients.end());
    }
    
    solnFileStream.close();
  }
  
  // send from rank 0 to all ranks
  std::vector<double> receivedPackedSolution;
  MPIWrapper::sendDataVectors(comm, recipients, packedSolutions, receivedPackedSolution);
  
  int offset = 0;
  const auto &myCellIDs = fineForm.boltzmannMesh()->cellIDsInPartition();
  for (const auto &cellID : myCellIDs)
  {
    int dofCount = fineForm.boltzmannMesh()->getElementType(cellID)->trialOrderPtr->getBasisCardinality(f->ID(), VOLUME_INTERIOR_SIDE_INDEX);
    
    TEUCHOS_TEST_FOR_EXCEPTION(dofCount + offset > receivedPackedSolution.size(), std::invalid_argument, "out of bounds");
    
    std::vector<double> coefficients(receivedPackedSolution.begin() + offset, receivedPackedSolution.begin() + offset + dofCount);
    
    offset += dofCount;
    
    mySolnCoefficients[cellID] = coefficients;
  }
  
  return mySolnCoefficients;
}

void convertOverkillSolution(Boltzmann1D1VFormulation &fineForm, const int numTimeSteps, const std::string &overkillSolutionDir)
{
  using std::cout;
  using std::cerr;
  using std::endl;
  using std::flush;
  using std::string;
  using std::to_string;
  
  const int rank = fineForm.boltzmannMesh()->Comm()->MyPID();
  
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
  fineForm.boltzmannSolution()->loadFromHDF5(solnFile);
  
  FunctionPtr f_fine     = Function::solution(fineForm.f(),     fineForm.boltzmannSolution());
  
  auto f = fineForm.f();
  
  // output just the f basis coefficients
  string solnFile_f = overkillSolutionDir + "/OverkillSolution_" + to_string(timeStepNumber) + ".soln_f";
  
  // collect all solution data onto rank 0
  auto allActiveCellIDs = fineForm.boltzmannMesh()->getActiveCellIDsGlobal();
  std::set<GlobalIndexType> noCellIDs;
  auto & cellIDsToImport = (rank == 0) ? allActiveCellIDs : noCellIDs;
  
  auto writeSolution = [&] () -> void
  {
    fineForm.boltzmannSolution()->importSolutionForOffRankCells(cellIDsToImport);
    
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
        auto dofIndices = fineForm.boltzmannMesh()->getElementType(cellID)->trialOrderPtr->getDofIndices(f->ID());
        
        solnFileStream.write((char*) &cellID, sizeof(cellID));
        
        int dofCount = dofIndices.size();
        solnFileStream.write((char*) &dofCount, sizeof(dofCount));
        
        bool warnAboutOffRank = false; // don't warn
        const auto & solnCoefficients = fineForm.boltzmannSolution()->allCoefficientsForCellID(cellID, warnAboutOffRank);
        
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
    fineForm.boltzmannSolution()->loadFromHDF5(solnFile);
    
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

void runAdaptiveComparisonExperiment(Boltzmann1D1VFormulation &adaptiveForm, Boltzmann1D1VFormulation &fineForm, Boltzmann1D1VFormulation &modestForm,
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
  
  auto comm = adaptiveForm.boltzmannMesh()->Comm();
  const int rank = comm->MyPID();
  
  if (writeOverkillSolutionAndQuit)
  {
    int numFineElements     = fineForm.boltzmannMesh()->numActiveElements();
    int numFineDofs         = fineForm.boltzmannMesh()->numGlobalDofs();
    
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
    
    auto mesh = fineForm.boltzmannMesh();
    auto soln = fineForm.boltzmannSolution();
          
    const int minSubsteps = 3;
    const int maxSubsteps = 15;
    fineForm.setFixedPointMinSubstepsForImplicit(minSubsteps);
    fineForm.setFixedPointMaxSubstepsForImplicit(maxSubsteps);
    fineForm.setFixedPointTolerance(1e-6);
    
    auto f = fineForm.f();
    int timeStepNumber = 0;
    int solutionNumber = 0;
    int numSolutions = numTimeSteps / solutionOutputFrequency;
    
    // output just the f basis coefficients
    string solnFile_f = overkillSolutionDir + "/OverkillSolution_" + to_string(solutionNumber) + ".soln_f";
    
    // collect all solution data onto rank 0
    auto allActiveCellIDs = fineForm.boltzmannMesh()->getActiveCellIDsGlobal();
    std::set<GlobalIndexType> noCellIDs;
    auto & cellIDsToImport = (rank == 0) ? allActiveCellIDs : noCellIDs;
    
    auto writeSolution = [&] () -> void
    {
      fineForm.boltzmannSolution()->importSolutionForOffRankCells(cellIDsToImport);
      
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
          auto dofIndices = fineForm.boltzmannMesh()->getElementType(cellID)->trialOrderPtr->getDofIndices(f->ID());
          
          solnFileStream.write((char*) &cellID, sizeof(cellID));
          
          int dofCount = dofIndices.size();
          solnFileStream.write((char*) &dofCount, sizeof(dofCount));
          
          bool warnAboutOffRank = false; // don't warn
          const auto & solnCoefficients = fineForm.boltzmannSolution()->allCoefficientsForCellID(cellID, warnAboutOffRank);
          
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
  
  MeshMap fineToModestMap(modestForm.boltzmannMesh(), fineForm.boltzmannMesh());
  
  // set initial conditions on adaptive form:
  adaptiveForm.setInitialConditions(f0);

  ErrorIndicatorPtr unrefinementErrorIndicator;
  if (!useEnergyForUnrefinements)
  {
    unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(adaptiveForm.boltzmannSolution(), adaptiveForm.f());
  }
  adaptiveForm.setRefinementThresholds(refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator, gateRefinements);
  
  auto boltzmannMesh = adaptiveForm.boltzmannMesh();
  auto gaussMesh     = adaptiveForm.gaussMesh();
  
  bool outputOverkillSolution = false;
  
  Teuchos::RCP<HDF5Exporter> boltzmannFineViz, boltzmannModestViz, boltzmannAdaptiveViz;
  if (saveViz)
  {
    if (outputOverkillSolution)
    {
      boltzmannFineViz     = Teuchos::rcp( new HDF5Exporter (fineForm.boltzmannMesh(),   "boltzmannFine",     outputDir));
    }
    boltzmannModestViz   = Teuchos::rcp( new HDF5Exporter (modestForm.boltzmannMesh(),   "boltzmannModest",   outputDir));
    boltzmannAdaptiveViz = Teuchos::rcp( new HDF5Exporter (adaptiveForm.boltzmannMesh(), "boltzmannAdaptive", outputDir));
  }
  
  BasisReconciliation br;
    
  // adapt to capture initial data:
  {
    FunctionPtr f_soln = Function::solution(adaptiveForm.f(), adaptiveForm.boltzmannSolution());
    
    FunctionPtr f0_err = f0 - f_soln;
    int quadratureEnrichment = 4;
    
    auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(boltzmannMesh, f0_err, quadratureEnrichment);
    
    double refinementThresholdForInitialData = 0.20;
    auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
    refStrategy->setMaxElements(maxElements);
    refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
    refStrategy->setMaximumRefinementLevel(maximumRefinementLevel);
    
    double f0_norm = f0->l2norm(boltzmannMesh, quadratureEnrichment);
    double relErr = f0_err->l2norm(boltzmannMesh, quadratureEnrichment) / f0_norm;
    
    int numInitialRefinements = 0;
    while (boltzmannMesh->numActiveElements() < maxElements * 0.90)
    {
      if (rank==0)
      {
        std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << boltzmannMesh->numActiveElements() << " elements and relErr is " << relErr << ".\n";
        std::cout << "Physical mesh has " << gaussMesh->numActiveElements() << " elements.\n";
      }
      const bool unrefine = true;
      const bool printToConsole = (rank == 0);
      refStrategy->refineAndUnrefine(unrefine, printToConsole);
      adaptiveForm.setInitialConditions(f0); // do projection again
      numInitialRefinements++;
      relErr = f0_err->l2norm(boltzmannMesh, quadratureEnrichment) / f0_norm;
      
      if (boltzmannMesh->numActiveElements() > maxElements)
      {
        // do one more refine + unrefine, and then switch to unrefinements only
        refStrategy->refineAndUnrefine(unrefine, printToConsole);
        while (boltzmannMesh->numActiveElements() > maxElements)
        {
          // repeat until unrefinement gets us down to maxElements or below
          if (rank==0)
          {
            std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << boltzmannMesh->numActiveElements() << " elements and relErr is " << relErr << ".\n";
            std::cout << "Physical mesh has " << gaussMesh->numActiveElements() << " elements.\n";
          }
          
          refStrategy->setRelativeErrorThreshold(1.1); // ensure one unrefinement-only pass
          refStrategy->refineAndUnrefine(unrefine, printToConsole);
          refStrategy->setRelativeErrorThreshold(refinementThresholdForInitialData);
          
          adaptiveForm.setInitialConditions(f0); // do projection again
          numInitialRefinements++;
          relErr = f0_err->l2norm(boltzmannMesh, quadratureEnrichment) / f0_norm;
          
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
      std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << boltzmannMesh->numActiveElements() << " elements and relErr is " << relErr << ".\n";
      std::cout << "Physical mesh has " << gaussMesh->numActiveElements() << " elements.\n";
    }
    
  }
  
  // set initial conditions for the other two forms:
  modestForm.setInitialConditions(f0);
  
  auto f = fineForm.f();
  int timeStepNumber = 0;
  int overkillStepNumber = 0;
  
  std::map<GlobalIndexType, std::vector<double> > localOverkillCoefficients;
  
  localOverkillCoefficients = readOverkillSolutionCoefficients(fineForm, overkillStepNumber, overkillSolutionDir);
  
  FunctionPtr f_modest   = Function::solution(modestForm.f(),   modestForm.boltzmannSolution());
  FunctionPtr f_adaptive = Function::solution(adaptiveForm.f(), adaptiveForm.boltzmannSolution());

  SolutionPtr projectedModestSoln   = Solution::solution(fineForm.boltzmannMesh()->bilinearForm(), fineForm.boltzmannMesh());
  SolutionPtr projectedAdaptiveSoln = Solution::solution(fineForm.boltzmannMesh()->bilinearForm(), fineForm.boltzmannMesh());
  
  MeshMap fineToAdaptiveMap(adaptiveForm.boltzmannMesh(), fineForm.boltzmannMesh());
  
  interpolateToFineSolution(f, modestForm.boltzmannSolution(),   projectedModestSoln,   fineToModestMap,   br);
  interpolateToFineSolution(f, adaptiveForm.boltzmannSolution(), projectedAdaptiveSoln, fineToAdaptiveMap, br);
  
  double adaptiveErr = overkillSolutionErrorL2Norm(f, projectedAdaptiveSoln, localOverkillCoefficients);
  double modestErr   = overkillSolutionErrorL2Norm(f, projectedModestSoln, localOverkillCoefficients);
  
  double adaptiveSolnNorm = l2NormOfUniformSolution(f, projectedAdaptiveSoln);
  double modestSolnNorm   = l2NormOfUniformSolution(f, projectedModestSoln);
  double fineSolnNorm     = overkillSolutionErrorL2Norm(f, fineForm.boltzmannSolution(), localOverkillCoefficients, true);
  
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
    boltzmannAdaptiveViz->exportSolution(adaptiveForm.boltzmannSolution(), t);
    boltzmannModestViz->exportSolution(  modestForm.boltzmannSolution(),   t);
    
    if (outputOverkillSolution)
    {
      boltzmannFineViz->exportSolution(    fineForm.boltzmannSolution(),     t);
    }
  }
  
  int numAdaptiveElements = adaptiveForm.boltzmannMesh()->numActiveElements();
  int numAdaptiveDofs     = adaptiveForm.boltzmannMesh()->numGlobalDofs();
  
  int numModestElements   = modestForm.boltzmannMesh()->numActiveElements();
  int numModestDofs       = modestForm.boltzmannMesh()->numGlobalDofs();
  
  int numFineElements     = fineForm.boltzmannMesh()->numActiveElements();
  int numFineDofs         = fineForm.boltzmannMesh()->numGlobalDofs();
  
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
  adaptiveForm.setFixedPointMinSubstepsForImplicit(minSubsteps);
  adaptiveForm.setFixedPointMaxSubstepsForImplicit(maxSubsteps);
  adaptiveForm.setFixedPointTolerance(fixedPointTol);
  modestForm.setFixedPointMinSubstepsForImplicit(minSubsteps);
  modestForm.setFixedPointMaxSubstepsForImplicit(maxSubsteps);
  modestForm.setFixedPointTolerance(fixedPointTol);
  fineForm.setFixedPointMinSubstepsForImplicit(minSubsteps);
  fineForm.setFixedPointMaxSubstepsForImplicit(maxSubsteps);
  fineForm.setFixedPointTolerance(fixedPointTol);
  
  // set up timers
  int numProcs = comm->NumProc();
  int indexBase = 0;
  Epetra_Time adaptiveTimer(*comm);
  Epetra_Time modestTimer  (*comm);
  
  double adaptiveTime = 0.;
  double modestTime   = 0.;
  Epetra_Map timeMap(numProcs,indexBase,*comm);
    
  for (timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
  {
    adaptiveTimer.ResetStartTime();
    adaptiveForm.takeTimeStep();
    {
      double myAdaptiveElapsedTime        = adaptiveTimer.ElapsedTime();
      double globalMaxAdaptiveElapsedTime = 0.0;
      comm->MaxAll(&myAdaptiveElapsedTime, &globalMaxAdaptiveElapsedTime, 1);
      adaptiveTime += globalMaxAdaptiveElapsedTime;
    }
    
    numAdaptiveElements = adaptiveForm.boltzmannMesh()->numActiveElements();
    numAdaptiveDofs     = adaptiveForm.boltzmannMesh()->numGlobalDofs();
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
      fineToAdaptiveMap = MeshMap(adaptiveForm.boltzmannMesh(), fineForm.boltzmannMesh());
      
      interpolateToFineSolution(f, modestForm.boltzmannSolution(),   projectedModestSoln,   fineToModestMap,   br);
      interpolateToFineSolution(f, adaptiveForm.boltzmannSolution(), projectedAdaptiveSoln, fineToAdaptiveMap, br);
      
      if (saveViz)
      {
        boltzmannAdaptiveViz->exportSolution(adaptiveForm.boltzmannSolution(), t);
        boltzmannModestViz->exportSolution(  modestForm.boltzmannSolution(),   t);
        
        if (outputOverkillSolution)
        {
          boltzmannFineViz->exportSolution(fineForm.boltzmannSolution(), t);
        }
      }
          
      adaptiveErr  = overkillSolutionErrorL2Norm(f, projectedAdaptiveSoln, localOverkillCoefficients);
      modestErr    = overkillSolutionErrorL2Norm(f, projectedModestSoln, localOverkillCoefficients);
      fineSolnNorm = overkillSolutionErrorL2Norm(f, fineForm.boltzmannSolution(), localOverkillCoefficients, true);
      
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
    summaryStream << "Field polynomial order of non-overkill meshes (one less than H^1 order): " << adaptiveForm.boltzmannMesh()->globalDofAssignment()->getInitialH1Order()[0]-1 << endl;
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
    int numUniformRefinements = 0;
    int delta_k   = 2;
    double dt_max = -1.0;
    int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
    int minSubsteps = 5;  // for fixed-point iteration.
    int maxSubsteps = 15; // for fixed-point iteration.
    double fixedPointTol = 1e-8;
    int numTimeSteps = 500;
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
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    double iterativeSolveMaxIters  = 10000;
    
    int vlasovTraceEnrichment = -1; // if left unset by user, will default to 0 for all problems but cold diode, where it is 1 (at least for now).
    double coldDiodeSigma = -1.0; // will be set automatically below if not specified by user
    double bcResolutionTolerance = -1;  // only applies if doing an adaptive solve (and so far only for cold diode); -1 indicates to spend the whole element budget to resolve; unrefinements at later time steps will tend to reduce the BC resolution over time, though experimentally the loss is often negligible (but this will depend on precise settings!).
    std::string resultsFilePrefix = "";
    
    std::string resultsFile = "";
    
    cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
    cmdp.setOption("implicit", "explicit", &implicit);
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
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
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
    double timeScaling     = 1.;
    double spatialScaling  = 1.;
    double EScaling        = 1.;
    double nScaling         = 1.;
    double potentialScaling = 1.;
    
    if (vlasovTraceEnrichment == -1)
    {
      if (problemChoice == COLD_DIODE)
      {
        // we no longer think it's a good idea to enrich the traces separately; refine to resolve the BCs; don't change the polynomial order of traces globally
        vlasovTraceEnrichment = 0;
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
      
      velocityScaling  = v_beam;
      timeScaling      = 1.97485663509e-9; // final time from SAND report [s]
      spatialScaling   = velocityScaling * timeScaling; // [m]
      EScaling         = m0 * velocityScaling / (q0 * timeScaling);
      potentialScaling = EScaling * spatialScaling;
      
      if (rank == 0)
      {
        std::cout << "Characteristic E-field strength E_0 : " << EScaling << std::endl;
        std::cout << "spatial scaling : " << spatialScaling << std::endl;
      }
      
      const double permittivity = 8.85418782e-12; // permittivity of free space
      
      nScaling     = J / (e * v_beam); // number density of plasma
      
      const double omega_p = sqrt( q0 * q0 * nScaling / (m0 * permittivity));
      
      xWidth = 0.01 / spatialScaling;
      u0     = 1.0;
      n0     = 1.0;
      q      = -1.0;
      
      if (coldDiodeSigma == -1)
      {
        sigma = 0.05; // default sigma value
      }
      else
      {
        sigma = coldDiodeSigma;
      }
      
      // eps0 is a weight that enters BoltmannFormulation; we use this to scale the problem:
      eps0   = 1.0 / (omega_p * omega_p * timeScaling * timeScaling);
      usePeriodicBCs = false;
      enforceQuasiNeutrality = false;
      v0     = 0.50;
      vWidth = 1.00;
      
      if (rank == 0)
      {
        std::cout << "sigma = " << sigma << std::endl;
        std::cout << "u0 = " << u0 << std::endl;
        std::cout << "n0 = " << n0 << std::endl;
      }
    }
    
    if (dt_max < 0)
    {
      dt_max = 1e-1;
    }
    
    Boltzmann1D1VFormulation::Boltzmann1D1VFormulationChoice formulationChoice;
    using FormulationChoice = Camellia::Boltzmann1D1VFormulation::Boltzmann1D1VFormulationChoice;
    if (implicit)
    {
      if (useL2Trace)
      {
        formulationChoice = FormulationChoice::ULTRAWEAK_IMPLICIT_L2_TRACE;
      }
      else
      {
        formulationChoice = FormulationChoice::ULTRAWEAK_IMPLICIT_H1_TRACE;
      }
    }
    else
    {
      if (useL2Trace)
      {
        formulationChoice = FormulationChoice::ULTRAWEAK_EXPLICIT_L2_TRACE;
      }
      else
      {
        formulationChoice = FormulationChoice::ULTRAWEAK_EXPLICIT_H1_TRACE;
      }
    }
    
    ERKTableauPtr erkTableau;
    if (!implicit)
    {
      erkTableau = ERKTableau::ssprk3();
    }
    
    double dt = dt_max;
    Boltzmann1D1VFormulation form(formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, erkTableau, vlasovTraceEnrichment, numUniformRefinements);
    
//    form.setUseGeometricMultigrid(useGeometricMultigrid);
    form.setIterativeSolveOutputLevel(iterativeSolveOutputLevel);
//    form.setIterativeSolveTolerance(iterativeSolveTolerance);
    std::vector<double> cgTols(maxSubsteps,iterativeSolveTolerance);
    if (maxSubsteps > 3)
    {
      // use looser tolerances for the first couple steps
      cgTols[0] = iterativeSolveTolerance * 100.0;
      cgTols[1] = iterativeSolveTolerance * 10.0;
    }
    form.setIterativeSolveTolerances(cgTols);
    form.setIterativeSolveMaxIterations(iterativeSolveMaxIters);

    
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
    else if (problemChoice == BUMP_ON_TAIL)
    {
      FunctionPtr fM_0 = Teuchos::rcp(new Maxwellian1V( 0.0, 1.0, 1.0-n0) );
      FunctionPtr fM_b = Teuchos::rcp(new Maxwellian1V(  u0, 0.5,     n0) );
      f0               = (fM_0 + 0.5 * fM_b) * (1. + gamma * cos_kx);
      std::cout << "f0 for bump-on-tail: " << f0->displayString() << "\n";
      adaptToCaptureInitialData = true;
    }
    else if (problemChoice == COLD_DIODE)
    {
      f0 = Function::zero(); // start with vacuum
      adaptToCaptureInitialData = false; // zero solution is always exactly captured…
      adaptToCaptureInitialBCs  = true;
    }
    
    if (convertOverkillSolutionFiles)
    {
      Boltzmann1D1VFormulation fineForm    (formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, overkillH1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, erkTableau, vlasovTraceEnrichment, numUniformRefinements+overkillRefinementLevel);
      convertOverkillSolution(fineForm, numTimeSteps, overkillSolutionDir);
      return 0; // exit
    }
    else if (runOverkillVersusAdaptiveComparison)
    {
      Boltzmann1D1VFormulation fineForm    (formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, overkillH1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, erkTableau, vlasovTraceEnrichment, numUniformRefinements+overkillRefinementLevel);
      Boltzmann1D1VFormulation adaptiveForm(formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, H1Order,         delta_k, usePeriodicBCs, enforceQuasiNeutrality, erkTableau, vlasovTraceEnrichment, numUniformRefinements  );
      Boltzmann1D1VFormulation modestForm  (formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, H1Order,         delta_k, usePeriodicBCs, enforceQuasiNeutrality, erkTableau, vlasovTraceEnrichment, numUniformRefinements+uniformMeshRefinementLevel);
      
  //    {
  //      // DEBUGGING/PROFILING
  //      // how much memory does the fine form MeshTopology take up?
  //      if (rank == 0)
  //      {
  //        std::cout << "overkill mesh, approximate footprint for base MeshTopology:\n";
  //        fineForm.boltzmannMesh()->getTopology()->baseMeshTopology()->printApproximateMemoryReport();
  //      }
  //    }
      
      const int maxAdaptiveMeshElements = modestForm.boltzmannMesh()->numActiveElements();
      
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
    
    if (!usePeriodicBCs)
    {
      if (problemChoice == COLD_DIODE)
      {
        // pin potential to zero at left and right:
        form.setGaussBCs(Function::zero(), Function::zero());
        // set a Maxwellian at the inflow:
        FunctionPtr fM_u0 = Teuchos::rcp(new Maxwellian1V( u0, sigma, n0) );
        form.setBoltzmannInflow(fM_u0, Function::zero());  // the right BC is only for negative velocity, so won't actually get triggered on our mesh
        
        const bool weightFluxesBySideParity = true;
        FunctionPtr tn_soln = Function::solution(form.t_n_hat(), form.boltzmannSolution(), weightFluxesBySideParity);
        
        const std::string & tagName = DIRICHLET_SET_TAG_NAME;
        int inflowTag = form.tagBoltzmannLeftInflow();
        FunctionPtr inflowRestriction = Function::meshSideSetCharacteristic(tagName, inflowTag);
        
        auto leftInflowTag = form.tagBoltzmannLeftInflow();
        
        FunctionPtr sgn = Function::sideParity();
        FunctionPtr n = Function::normal();
        FunctionPtr n_x = n->spatialComponent(1); // 1-based
        FunctionPtr vx = form.vx();
        
        FunctionPtr tn_leftInflowExact = fM_u0 * sgn * n_x * vx;
        bcFunctionsToResolve.push_back({{tn_leftInflowExact, tn_soln},leftInflowTag});
      }
    }
    form.setInitialConditions(f0);
    
    // adapt to capture initial data:
    if (useAdaptivity && adaptToCaptureInitialData)
    {
      FunctionPtr f_soln = Function::solution(form.f(), form.boltzmannSolution());
      
      FunctionPtr f0_err = f0 - f_soln;
      int quadratureEnrichment = 4;
      
      auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(form.boltzmannMesh(), f0_err, quadratureEnrichment);
      
      double refinementThresholdForInitialData = 0.20;
      auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
      refStrategy->setMaxElements(maxElements);
      ErrorIndicatorPtr unrefinementErrorIndicator;
      if (!useEnergyForUnrefinements)
      {
        unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(form.boltzmannSolution(), form.f());
      }
      refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
      refStrategy->setMaximumRefinementLevel(maximumRefinementLevel);
      refStrategy->setMaxP(H1Order);
      
      double f0_norm = f0->l2norm(form.boltzmannMesh(), quadratureEnrichment);
      double relErr = f0_err->l2norm(form.boltzmannMesh(), quadratureEnrichment) / f0_norm;
      
      int numInitialRefinements = 0;
      while (form.boltzmannMesh()->numActiveElements() < maxElements * 0.90)
      {
        if (rank==0)
        {
          std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << form.boltzmannMesh()->numActiveElements() << " elements and relErr is " << relErr << ".\n";
          std::cout << "Physical mesh has " << form.gaussMesh()->numActiveElements() << " elements.\n";
        }
        const bool unrefine = true;
        const bool printToConsole = (rank == 0);
        refStrategy->refineAndUnrefine(unrefine, printToConsole);
        form.setInitialConditions(f0); // do projection again
        numInitialRefinements++;
        relErr = f0_err->l2norm(form.boltzmannMesh(), quadratureEnrichment) / f0_norm;
        
        if (form.boltzmannMesh()->numActiveElements() > maxElements)
        {
          // do one more refine + unrefine, and then switch to unrefinements only
          refStrategy->refineAndUnrefine(unrefine, printToConsole);
          while (form.boltzmannMesh()->numActiveElements() > maxElements)
          {
            // repeat until unrefinement gets us down to maxElements or below
            if (rank==0)
            {
              std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << form.boltzmannMesh()->numActiveElements() << " elements and relErr is " << relErr << ".\n";
              std::cout << "Physical mesh has " << form.gaussMesh()->numActiveElements() << " elements.\n";
            }
            
            refStrategy->setRelativeErrorThreshold(1.1); // ensure one unrefinement-only pass
            refStrategy->refineAndUnrefine(unrefine, printToConsole);
            refStrategy->setRelativeErrorThreshold(refinementThresholdForInitialData);
            
            form.setInitialConditions(f0); // do projection again
            numInitialRefinements++;
            relErr = f0_err->l2norm(form.boltzmannMesh(), quadratureEnrichment) / f0_norm;
            
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
        std::cout << "After " << numInitialRefinements << " refinements, phase space mesh has " << form.boltzmannMesh()->numActiveElements() << " elements and relErr is " << relErr << ".\n";
        std::cout << "Physical mesh has " << form.gaussMesh()->numActiveElements() << " elements.\n";
      }
      form.setRefinementThresholds(refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator, gateRefinements);
      
      // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
      form.gaussMesh()->repartitionAndRebuild();
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
        form.boltzmannSolution()->projectOntoMesh(bcMap);
        
        FunctionPtr bc_err = bcFxnExact * bcRestriction - bcFxnSoln * bcRestriction;
        int quadratureEnrichment = 4;
        
        auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(form.boltzmannMesh(), bc_err, quadratureEnrichment);
        
        double refinementThresholdForInitialData = 0.80;
        auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
  //      refStrategy->setMaxElements(maxElements);
        refStrategy->setMaxP(H1Order);
        
        double bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.boltzmannMesh(), quadratureEnrichment);
        double relErr = bc_err->l2norm(form.boltzmannMesh(), quadratureEnrichment) / bc_norm;
        
        int numBCRefinements = 0;
        int numVlasovElements = form.boltzmannMesh()->numActiveElements();
        while ((relErr > bcResolutionTolerance) && (numVlasovElements < maxElements))
        {
          if (rank==0)
          {
            std::cout << "After " << numBCRefinements++ << " refinements, phase space mesh has " << form.boltzmannMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
            std::cout << "Physical mesh has " << form.gaussMesh()->numActiveElements() << " elements.\n";
          }
          
          refStrategy->refine();
          
          // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
          form.gaussMesh()->repartitionAndRebuild();
          
          // project the BC onto the mesh again
          form.boltzmannSolution()->projectOntoMesh(bcMap);
          
          bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.boltzmannMesh(), quadratureEnrichment);
          relErr = bc_err->l2norm(form.boltzmannMesh(), quadratureEnrichment) / bc_norm;
          
          numVlasovElements = form.boltzmannMesh()->numActiveElements();
        }
        if (rank==0)
        {
          std::cout << "After " << numBCRefinements << " refinements, phase space mesh has " << form.boltzmannMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
          std::cout << "Physical mesh has " << form.gaussMesh()->numActiveElements() << " elements.\n";
        }
      }
      
      ErrorIndicatorPtr unrefinementErrorIndicator;
      if (!useEnergyForUnrefinements)
      {
        unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(form.boltzmannSolution(), form.f());
      }
      
      form.setRefinementThresholds(refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator, gateRefinements);
    }
    
    std::ostringstream suffix;
    if (implicit)
    {
      suffix << "Implicit";
    }
    if (useL2Trace)
    {
      suffix << "_L2Trace";
    }
    else
    {
      suffix << "_H1Trace";
    }
    if (problemChoice != COLD_DIODE)
    {
      suffix << "_u0" << u0;
    }
    else
    {
      suffix << "_sigma" << sigma;
    }
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

    std::string problemPrefix;
    if (problemChoice == TWO_STREAM)
    {
      problemPrefix = "TwoStream";
    }
    else if (problemChoice == BUMP_ON_TAIL)
    {
      problemPrefix = "BumpOnTail";
    }
    else
    {
      problemPrefix = "ColdDiode";
    }

    std::ostringstream boltzmannSS;
    boltzmannSS << problemPrefix <<"Boltzmann" << suffix.str();

    std::ostringstream gaussSS;
    gaussSS << problemPrefix << "Gauss" << suffix.str();
    
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
      
      resultsFileStream << numTimeSteps << "_timeSteps";
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
      double bcRelErr = -1;
      for (auto bcPair : bcFunctionsToResolve)
      {
        FunctionPtr bcFxnExact = bcPair.first.first;
        FunctionPtr bcFxnSoln  = bcPair.first.second;
        int bcTag = bcPair.second;
        
        const std::string & tagName = DIRICHLET_SET_TAG_NAME;
        FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
        
        FunctionPtr bc_err = bcFxnExact * bcRestriction - bcFxnSoln * bcRestriction;
        int quadratureEnrichment = 4;
        
        double bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.boltzmannMesh(), quadratureEnrichment);
        bcRelErr = bc_err->l2norm(form.boltzmannMesh(), quadratureEnrichment) / bc_norm;
      }
      return bcRelErr;
    };
    
    auto reportBCResolutionAndTimeStepComplete = [&](const int &timeStepNumber, const double &bcRelErr, const double &energyError) {
      auto numVlasovDofs       = form.boltzmannMesh()->numGlobalDofs();
      auto vlasovElementCount  = form.boltzmannMesh()->numActiveElements();
      auto poissonElementCount = form.gaussMesh()->numActiveElements();
      auto poissonDofCount     = form.gaussMesh()->numGlobalDofs();
      
      if (rank == 0)
      {
        std::cout << "BC resolution, relative error in L^2 norm: " << bcRelErr << std::endl;
        writeTimeStepReport(vlasovElementCount, numVlasovDofs, poissonElementCount, poissonDofCount, bcRelErr, energyError);
        std::cout << "Time step " << timeStepNumber << " complete.\n";
      }
    };
   
    
  //  std::cout << "boltzmann name: " << boltzmannSS.str() << std::endl;
  //  std::cout << "gauss name: " << gaussSS.str() << std::endl;
    
    const std::string directoryPath = outputDir;
    
    if (resultsFile == "")
    {
      std::ostringstream resultsFileStream;
      resultsFileStream << directoryPath << "/" << problemPrefix << "_results" << suffix.str() << ".dat";
      resultsFile = resultsFileStream.str();
    }
    
    Teuchos::RCP<HDF5Exporter> boltzmannViz, gaussViz;
    if (saveViz)
    {
      boltzmannViz = Teuchos::rcp(new HDF5Exporter(form.boltzmannMesh(), boltzmannSS.str(), directoryPath));
      gaussViz = Teuchos::rcp(new HDF5Exporter (form.gaussMesh(), gaussSS.str(), directoryPath) );
    }
    
    std::ostringstream E_magnitudeFilePath;
    E_magnitudeFilePath << directoryPath << "/" << "E_magnitude" << suffix.str() << ".dat";
    std::fstream E_magnitudeOut;
    if (rank == 0)
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
      gaussViz->exportSolution(form.gaussSolution(),t);
      boltzmannViz->exportSolution(form.boltzmannSolution(),t);
    }
    
    FunctionPtr E_soln = Function::solution(form.E(), form.gaussSolution());
    FunctionPtr E_squared = E_soln * E_soln;
    FunctionPtr abs_E_soln = Teuchos::rcp( new AbsFunction(E_soln) );
    
    auto refStrategy = form.refinementStrategy();
    
    form.setPerformRefinementsDuringTimeSteps(false); //
    
    for (int timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
    {
      if (timeStepC0 > 0)
      {
        // adapt time step
        double E_max = abs_E_soln->maximumValue(form.gaussMesh());
        double v_max = std::max(std::abs(v0), std::abs(v0+vWidth));
        double h_x   = xWidth / xElements;
        double h_v   = vWidth / vElements;
        double cfl_dt_max = 1.0 / (v_max / h_x + (q/m) * E_max / h_v) * timeStepC0;
        double dt_new = std::min(dt_max, cfl_dt_max);
        if (dt != dt_new)
        {
          std::cout << "Set dt = " << dt_new << std::endl;
          dt = dt_new;
          form.setTimeStep(dt);
        }
      }
      
      form.takeTimeStep();
      t += dt;
      
      checkMesh(form.boltzmannMesh());
      
      const double bcRelErr = getBCResolution(); // returns -1 if BCs to resolve weren't specified
      form.performRefinements(); // only does refinements if setRefinementThresholds() has been called
      const double energyError = (refStrategy != Teuchos::null) ? refStrategy->getEnergyError(refStrategy()->getNumRefinements() - 1) : -1;
      
      const bool computeMinimum_bn = false; // average value on a side
      if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
      {
        double min_bn = 1000;
        if (computeMinimum_bn)
        {
          // v * n->x() + (q/m) * E * n->y()
          auto E = form.E_extruded();
          auto v = Function::yn(1);
          
          const double xElemWidth = xWidth / xElements;
          const double vElemWidth = vWidth / vElements;
          
          auto myCellIDs = form.boltzmannMesh()->cellIDsInPartition();
          double localMin_bn = min_bn;
          for (auto cellID : myCellIDs)
          {
            auto basisCache = BasisCache::basisCacheForCell(form.boltzmannMesh(), cellID);
            for (int sideOrdinal=0; sideOrdinal<4; sideOrdinal++)
            {
              const bool sideParallelToXAxis = (sideOrdinal % 2 == 0);
              FunctionPtr f = sideParallelToXAxis ? (q/m) * E : v;
              const double integralLength = sideParallelToXAxis ? xElemWidth : vElemWidth;
              
              auto sideCache = basisCache->getSideBasisCache(sideOrdinal);
              double integral = std::abs(f->integrate(sideCache));
              double averageValue = integral / integralLength;
              
              localMin_bn = std::min(localMin_bn, averageValue);
            }
          }
          MPIWrapper::CommWorld()->MinAll(&localMin_bn, &min_bn, 1);
        }
        
        if (saveViz)
        {
          boltzmannViz->exportSolution(form.boltzmannSolution(),t);
          gaussViz->exportSolution(form.gaussSolution(),t);
        }
        
        reportBCResolutionAndTimeStepComplete(timeStepNumber,bcRelErr,energyError);
        
  //      double E_mag = E_squared->integrate(form.gaussMesh());
  //      if (rank == 0)
  //      {
  //        std::cout << "Completed time step " << timeStepNumber << " (t = " << t << ")" << std::endl;
  //        std::cout << "|E|^2 integral = " << E_mag << std::endl;
  //        E_magnitudeOut << t << "\t" << E_mag << std::endl << std::flush;
  //        if (computeMinimum_bn)
  //        {
  //          std::cout << "Minimum average value for b_n on a side: " << min_bn << std::endl;
  //        }
  //      }
      }
    } // end time step loop
    if (rank == 0)
    {
      E_magnitudeOut.close();
    }
    
    const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
    auto solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form.boltzmannSolution(), form.f(), form.gaussMesh()) );
        
    FunctionPtr potential = Function::solution(form.VE(), form.gaussSolution());
    
    auto density  = solutionMoments->solutionMomentFunction(0);
    auto momentum = solutionMoments->solutionMomentFunction(1);
    
    auto velocity = momentum / density;
    
    // output to results file:
    {
      // output (physical/dimensional) results
      // ultimately, we want E-field, velocity, and some other things
      // to begin with, let's output the E-field values computed at quadrature points at final time
            
      // the di() methods are 1-based:
      static const int SPATIAL_INDEX  = 1;
      static const int VELOCITY_INDEX = 1;
      
      // SpatialFilter's components are zero-based
      static const int SPATIAL_INDEX_ZERO_BASED  = SPATIAL_INDEX  - 1;
      static const int VELOCITY_INDEX_ZERO_BASED = VELOCITY_INDEX - 1;
      
      auto allActivePhysicalCells = form.gaussMesh()->getActiveCellIDsGlobal();
      
      form.gaussSolution()->importSolutionForOffRankCells(allActivePhysicalCells);
      
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
        auto basisCache = BasisCache::basisCacheForCell(form.gaussMesh(), firstPhysicalCellID, false, cubatureEnrichmentDegree); // false: not test versus test
        std::vector<GlobalIndexType> physicalCellIDsVector(allActivePhysicalCells.begin(),allActivePhysicalCells.end());
        
        const int numPoints = basisCache->getRefCellPoints().extent_int(0);
        auto & cubWeights = basisCache->getCubatureWeightsFC();
        
        auto physicalCellNodes = form.gaussMesh()->physicalCellNodes(physicalCellIDsVector);
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
        
        for (int cellOrdinal=0; cellOrdinal<physicalCellIDsVector.size(); cellOrdinal++)
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
        for (int cellOrdinal=0; cellOrdinal<physicalCellIDsVector.size(); cellOrdinal++)
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
    
  //  // output to results file:
  //  auto activePhysicalCells = form.gaussMesh()->getActiveCellIDsGlobal();
  //  form.gaussSolution()->importSolutionForOffRankCells(activePhysicalCells);
  //
  //  if (rank == 0)
  //  {
  //    std::map<double,double> E_values; // map from x coordinate to final E field value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
  //    std::map<double,double> pointWeight_values; // map from x coordinate to cell measure-weighted quadrature weight
  //    GlobalIndexType firstPhysicalCellID = *activePhysicalCells.begin();
  //
  //    auto basisCache = BasisCache::basisCacheForCell(form.gaussMesh(), firstPhysicalCellID);
  //    std::vector<GlobalIndexType> physicalCellIDsVector(activePhysicalCells.begin(),activePhysicalCells.end());
  //
  //    auto physicalCellNodes = form.gaussMesh()->physicalCellNodes(physicalCellIDsVector);
  //    basisCache->setPhysicalCellNodes(physicalCellNodes, physicalCellIDsVector, false);
  //
  //    const FieldContainer<double> & physicalQuadraturePoints = basisCache->getPhysicalCubaturePointsFC();
  //    const FieldContainer<double> & cubatureWeights = basisCache->getCubatureWeightsFC();
  //
  //    const int numCells  = physicalQuadraturePoints.dimension(0);
  //    const int numPoints = physicalQuadraturePoints.dimension(1);
  //
  //    FieldContainer<double> E_values_FC(physicalCellIDsVector.size(), numPoints);
  //    E_soln->values(E_values_FC, basisCache);
  //
  //    for (int cellOrdinal=0; cellOrdinal<physicalCellIDsVector.size(); cellOrdinal++)
  //    {
  //      const double cellWidth = physicalCellNodes(cellOrdinal,1,0) - physicalCellNodes(cellOrdinal,0,0);
  //      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
  //      {
  //        const double & x = physicalQuadraturePoints(cellOrdinal,pointOrdinal,0);
  //        const double & E = E_values_FC(cellOrdinal,pointOrdinal);
  //        const double pointWeight = cubatureWeights(pointOrdinal) * cellWidth;
  //
  //        E_values[x]           = E;
  //        pointWeight_values[x] = pointWeight;
  //      }
  //    }
  //
  //    std::ofstream resultsFileStream(resultsFile, std::ios::out);
  //
  //    resultsFileStream << "x\tquadrature weight\tE\n";
  //    resultsFileStream << setprecision(15);
  //
  //    for (const auto & pointEntry : pointWeight_values)
  //    {
  //      const auto x           = pointEntry.first * spatialScaling;
  //      const auto pointWeight = pointEntry.second * spatialScaling;
  //      const auto E           = E_values[pointEntry.first] * EScaling;
  //
  //      resultsFileStream << x           << "\t";
  //      resultsFileStream << pointWeight << "\t";
  //      resultsFileStream << E           << "\n";
  //    }
  //
  //    resultsFileStream.close();
  //
  //    std::cout << "Wrote results to " << resultsFile << std::endl;
  //  }
      
    const bool outputInflowValues = false; // note that "true" may cause a crash on distributed runs
    if (outputInflowValues && (problemChoice == COLD_DIODE))
    {
      // output t_n / v along inflow boundary (which is x=0)
      const int X_COMP_ZERO_BASED  = 0;
      const int VX_COMP_ZERO_BASED = 1;
      SpatialFilterPtr inflowFilter = SpatialFilter::matchingComponent(X_COMP_ZERO_BASED, x0);
      
      auto vlasovMeshTopo = form.boltzmannMesh()->getTopology()->baseMeshTopology();
      auto inflowSideIndices = vlasovMeshTopo->getBoundarySidesThatMatch(inflowFilter);
      std::set<GlobalIndexType> phaseInflowCells;
      
      const unsigned leftSideOrdinal = 2;
      const unsigned sideDim = vlasovMeshTopo->getDimension() - 1;
      
      for (const auto & sideEntityIndex : inflowSideIndices)
      {
        IndexType phaseCellID = vlasovMeshTopo->getFirstCellForSide(sideEntityIndex).first;
        if (!vlasovMeshTopo->isParent(phaseCellID))
        {
          phaseInflowCells.insert(phaseCellID);
        }
        // assert that the sideEntityIndex we found is in fact the leftSideOrdinal
        assert( sideEntityIndex == vlasovMeshTopo->getCell(phaseCellID)->entityIndex(sideDim, leftSideOrdinal) );
      }
      
      form.boltzmannSolution()->importSolutionForOffRankCells(phaseInflowCells);
      
      if (rank == 0)
      {
        std::map<double,double> f_values; // map from v_x coordinate to a computed f = t_n / v value; we use a map so it's sorted by v_x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        GlobalIndexType firstPhaseCellID = *phaseInflowCells.begin();
        
        const bool testVsTest = false;
        const int cubatureEnrichment = 10; // get a bunch of extra points
        auto basisCache = BasisCache::basisCacheForCell(form.boltzmannMesh(), firstPhaseCellID, testVsTest, cubatureEnrichment)->getSideBasisCache(leftSideOrdinal);
        std::vector<GlobalIndexType> phaseCellIDsVector(phaseInflowCells.begin(),phaseInflowCells.end());
        
        const auto & cubPointsFC  = basisCache->getRefCellPointsFC();
        const int numPoints = cubPointsFC.dimension(0); // (P,D)
        
        auto physicalCellNodes = form.boltzmannMesh()->physicalCellNodes(phaseCellIDsVector);
        basisCache->setPhysicalCellNodes(physicalCellNodes, phaseCellIDsVector, false);
        
        FunctionPtr tn_soln = Function::solution(form.t_n_hat(), form.boltzmannSolution(), false);
        
        FieldContainer<double> tn_values_FC(phaseCellIDsVector.size(), numPoints);
        tn_soln->values(tn_values_FC, basisCache);
        
        const FieldContainer<double> & physicalQuadraturePoints = basisCache->getPhysicalCubaturePointsFC();
        
        const int numCells  = physicalQuadraturePoints.dimension(0);
        
        std::ostringstream gnuPlotStringStream;
        gnuPlotStringStream << directoryPath << "/" << problemPrefix << "_inflowBC" << suffix.str() << ".gnuplot";
        std::string gnuPlotFile = gnuPlotStringStream.str();
        
        std::ofstream gnuPlotFileStream(gnuPlotFile, std::ios::out);
        
        gnuPlotFileStream << setprecision(15);
        gnuPlotFileStream << "set term png\n";
        gnuPlotFileStream << "set output \"" << problemPrefix << "_inflowBC" << suffix.str() << ".png\"\n";
        gnuPlotFileStream << "plot '-' using 1:2 t \"f inflow\" smooth unique\n";
        gnuPlotFileStream << "set term png\n";
        
        /*
         set term png
         set output "inflowBCs.png"
         plot '-' using 1:2 t "f inflow" smooth unique
         */
        gnuPlotFileStream << "plot '-' using 1:2 smooth unique\n"; // inline data
        for (int cellOrdinal=0; cellOrdinal<phaseCellIDsVector.size(); cellOrdinal++)
        {
          for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
          {
            const double & v_x = physicalQuadraturePoints(cellOrdinal,pointOrdinal,VX_COMP_ZERO_BASED);
            const double & t_n = tn_values_FC(cellOrdinal,pointOrdinal);
            
            const double f = t_n / v_x;
            gnuPlotFileStream << "  " << v_x << " " << f << std::endl;
          }
        }
        gnuPlotFileStream << "  e\n";
        gnuPlotFileStream.close();
      }
    }
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
