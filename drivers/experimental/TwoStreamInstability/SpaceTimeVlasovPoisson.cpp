//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "CamelliaDataHelpers.h"
#include "DoerflerRefinementStrategy.h"
#include "VlasovPoissonSpaceTimeFormulation.h"
#include "EnergyErrorFunction.h"
#include "ErrorIndicator.h"
#include "ERKTableau.h"
#include "ExpFunction.h"
#include "GDAMinimumRule.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MonomialFunctions.h"
#include "PhysicalSolutionExtrusion.hpp"
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
  
  TFunctionPtr<double> dx() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dy() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dz() override
  {
    auto vx = MonomialFunction::xi_n(2, 1);
    FunctionPtr f_M = Teuchos::rcp( new Maxwellian1T1V(_u,_sigma,_n));
    return (_u - vx) / (_sigma * _sigma) * f_M;
  }
  
  TFunctionPtr<double> dt() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> di(int i) override
  {
    if (i == 3) // 1-based
    {
      return dz();
    }
    else
    {
      return Function::zero();
    }
  }
};

class TemporalRamp1D1T : public TFunction<double>
{
  double _rampWidth; // amount of time before the value is 1.0
public:
  TemporalRamp1D1T(double rampWidth)
  :
  _rampWidth(rampWidth)
  {}
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & spaceTimePoints = basisCache->getPhysicalCubaturePointsFC();
    const int numCells  = spaceTimePoints.dimension(0);
    const int numPoints = spaceTimePoints.dimension(1);
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double t = spaceTimePoints(cellOrdinal,pointOrdinal,1); // t is the second dimension for 1D1T
        double value;
        if (t < _rampWidth)
        {
          value = t / _rampWidth;
        }
        else
        {
          value = 1.0;
        }
        values(cellOrdinal,pointOrdinal) = value;
      }
    }
  }
  
  TFunctionPtr<double> dx() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dy() override
  {
    auto heavisideT = Function::heavisideY(_rampWidth); // 1 for y < _rampWidth, 0 otherwise
    return (1.0 / _rampWidth) * heavisideT;
  }
  
  TFunctionPtr<double> dz() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dt() override
  {
    return dy();
  }
  
  TFunctionPtr<double> di(int i) override
  {
    if (i == 2) // 1-based
    {
      return dt();
    }
    else
    {
      return Function::zero();
    }
  }
};

class CellIDFunction : public Function
{
public:
  CellIDFunction() : Function(0) {}

  using Function::values;
  
  virtual void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();

    for (int cellOrdinal=0; cellOrdinal<cellIDs.size(); cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<values.dimension(1); pointOrdinal++)
      {
        values(cellOrdinal,pointOrdinal) = cellIDs[cellOrdinal];
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
  
  const int sideDim = mesh->getDimension() - 1;
  
  for (auto & cellID : myCellIDs)
  {
    CellPtr cell = mesh->getTopology()->getCell(cellID);
    int sideCount = cell->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      // CHECK PARITY CONSISTENCY
      auto neighborInfo = cell->getNeighborInfo(sideOrdinal, mesh->getTopology());
      const IndexType & neighborCellID = neighborInfo.first;
      const auto & mySideOrdinalInNeighbor = neighborInfo.second;
      if (neighborCellID != -1)
      {
        int myParity       = globalDofAssignment->sideParity(cellID, sideOrdinal);
        int neighborParity = globalDofAssignment->sideParity(neighborCellID, mySideOrdinalInNeighbor);
        TEUCHOS_TEST_FOR_EXCEPTION(myParity != -neighborParity, std::invalid_argument, "inconsistent parities");
      }
      else
      {
        int myParity       = globalDofAssignment->sideParity(cellID, sideOrdinal);
        TEUCHOS_TEST_FOR_EXCEPTION(myParity != 1, std::invalid_argument, "incorrect parity for boundary side");
      }
      
      // CHECK SIDE VERTEX CONSISTENCY
      auto vertices = cell->vertices();
      auto cellTopo = cell->topology();
      
      auto sideTopo = cellTopo->getSide(sideOrdinal);
      
      int sideVertexCount = sideTopo->getVertexCount();
      
      int sideIndex = cell->entityIndex(sideDim, sideOrdinal);
      auto sideVertices = mesh->getTopology()->getEntityVertexIndices(sideDim, sideIndex);
      
      for (int v_ord = 0; v_ord < sideVertexCount; v_ord++)
      {
        int v_ord_in_cell = cellTopo->getNodeMap(sideDim, sideOrdinal, v_ord);
        
        int vertexIndexCell = vertices[v_ord_in_cell];
        int vertexIndexSide = sideVertices[v_ord];
        
        TEUCHOS_TEST_FOR_EXCEPTION(vertexIndexCell != vertexIndexSide, std::invalid_argument, "cell and side have inconsistent vertex indices");
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
    fineForm.setFixedPointMinSubsteps(numSubsteps);
    fineForm.setFixedPointMaxSubsteps(numSubsteps);
    
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
  adaptiveForm.setFixedPointMinSubsteps(numSubsteps);
  modestForm.setFixedPointMinSubsteps(numSubsteps);
  fineForm.setFixedPointMinSubsteps(numSubsteps);
  adaptiveForm.setFixedPointMaxSubsteps(numSubsteps);
  modestForm.setFixedPointMaxSubsteps(numSubsteps);
  fineForm.setFixedPointMaxSubsteps(numSubsteps);
  
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

//! This method is for setting up a specific set of refinements that generate some issues I'm trying to debug.
void performDebuggingRefinementSeries(VlasovPoissonSpaceTimeFormulation &form)
{
  const int rank = form.vlasovMesh()->Comm()->MyPID();
  
  auto vlasovMesh  = form.vlasovMesh();
  auto poissonMesh = form.poissonMesh();
  
  enum DebugRefinementSeries
  {
    ORIGINAL,
    SIMPLIFIED_ORIGINAL,
    POST_TIME_RAMP_ONE_REF,
    POST_TIME_RAMP_TWO_REFS,
    POST_TIME_RAMP_THREE_REFS,
    POST_TIME_RAMP_FOUR_REFS,
    POST_TIME_RAMP_FOUR_REFS_SIMPLIFIED
  };
  
  std::map<DebugRefinementSeries,std::string> refSeriesNames;
  refSeriesNames[ORIGINAL] = "original";
  refSeriesNames[SIMPLIFIED_ORIGINAL] = "simplified original";
  refSeriesNames[POST_TIME_RAMP_ONE_REF] = "post-time-ramp one ref";
  refSeriesNames[POST_TIME_RAMP_TWO_REFS] = "post-time-ramp two refs";
  refSeriesNames[POST_TIME_RAMP_THREE_REFS] = "post-time-ramp three refs";
  refSeriesNames[POST_TIME_RAMP_FOUR_REFS] = "post-time-ramp four refs";
  const DebugRefinementSeries series = POST_TIME_RAMP_FOUR_REFS;
  
  vector<vector<GlobalIndexType>> refinementSequence;
  
  switch (series) {
    case ORIGINAL:
    {
      vector<GlobalIndexType> firstRefinement {6,7,8,9,10,11};
      vector<GlobalIndexType> secondRefinement {21,23,24,25,26,27,28,30,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,51,52,54,61,63};
      vector<GlobalIndexType> thirdRefinement {53,60,65,69,71,72,73,74,84,85,87,92,93,96,97,99,116,120,121,132,133,136,137,138,144,164,165,167,168,169,172,173,176,177,196,197,198,200,201,202,204,205,206,207,208,212,214,216,218,220,221,222,224,225,228,230,232,235,252,253,256,257,259,260,261,262,263,264,265,267,269,271,273,275,276,292,293,294,295,296,297,299,305};
      refinementSequence = {firstRefinement, secondRefinement, thirdRefinement};
    }
      break;
    case SIMPLIFIED_ORIGINAL:
    {
      vector<GlobalIndexType> firstRefinement {6,7,8,9};
      vector<GlobalIndexType> secondRefinement {49,51};
      refinementSequence = {firstRefinement, secondRefinement};
    }
      break;
    case POST_TIME_RAMP_ONE_REF:
    case POST_TIME_RAMP_TWO_REFS:
    case POST_TIME_RAMP_THREE_REFS:
    case POST_TIME_RAMP_FOUR_REFS:
    case POST_TIME_RAMP_FOUR_REFS_SIMPLIFIED:
    {
      vector<GlobalIndexType> firstRefinement {4,6,7,10,11};
      vector<GlobalIndexType> secondRefinement {5,8,9,16,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,49,51};
      vector<GlobalIndexType> thirdRefinement {60,62,63,64,66,75,116,117,118,122,123,135,140,141,142,149,151,156,158,160,161,162,163,164,165,166,167,168,169,170,172,176,177,178,179,180,181,182,183,184,186,196,197,199,206,213,215,220,222,224,225,227,228,229,232,233,234,235,236,237,238,239,241,243,248,249,250,251,252,253,254,255,256,258,264,266,273,275,281,283};
      vector<GlobalIndexType> fourthRefinement {61,65,72,73,94,95,101,102,103,104,108,109,110,111,112,113,114,115,119,120,121,124,125,126,127,133,137,139,143,153,155,157,159,171,173,174,175,185,187,188,189,190,191,192,194,198,204,208,210,212,214,226,230,231,240,242,244,245,246,247,257,272,277,326,327,328,330,331,332,333,334,335,338,340,341,342,343,344,345,346,347,352,353,354,355,360,361,362,363,390,413,415,419,420,421,422,428,429,430,431,433,435,438,439,445,447,453,455,457,459,460,461,462,463,465,467,468,469,470,471,475,476,477,479,480,481,482,483,484,485,486,487,493,494,495,496,497,498,499,500,501,502,503,504,505,506,507,508,510,512,514,516,520,521,522,523,524,525,526,527,528,529,530,531,532,534,536,537,538,539,540,541,544,545,548,549,552,553,554,555,556,558,559,560,561,572,573,574,575,576,577,578,579,580,581,582,583,589,591,592,593,594,595,596,597,598,599,600,601,602,603,604,605,606,607,608,609,610,611,612,614,616,617,618,619,620,621,622,623,624,625,626,627,628,630,632,633,634,635,636,637,638,639,640,641,642,643,644,645,646,647,648,649,650,651,652,653,654,655,658,660,661,662,663,664,666,668,669,670,671,676,678,680,682,684,685,686,687,688,690,692,693,694,695,700,702,708,710,712,713,715,716,717,720,721,722,723,724,725,726,727,728,729,730,731,732,733,734,735,736,737,744,745,747,752,754,755,756,757,758,759,766,768,769,770,771,772,773,774,775,780,782,783,784,785,786,787,788,789,791,792,793,794,795,796,797,798,799,800,801,802,803,805,807,808,809,810,811,812,813,814,815,816,817,818,819,820,821,822,823,824,825,826,827,828,829,830,831,832,833,834,835,836,837,838,839,844,845,846,847,848,849,850,851,852,853,854,855,860,861,862,863,864,865,866,867,869,871,872,873,874,875,876,877,878,879,880,881,882,883,885,887,888,889,890,891,892,893,894,895,904,906,907,910,912,913,914,915,916,918,920,921,922,923,924,926,928,929,930,931,933,935,937,938,939,943,944,945,946,947,949,951,952,953,954,955,957,959,1016,1017,1025,1027};
      if (series == POST_TIME_RAMP_ONE_REF)
      {
        refinementSequence = {firstRefinement};
      }
      else if (series == POST_TIME_RAMP_TWO_REFS)
      {
        refinementSequence = {firstRefinement, secondRefinement};
      }
      else if (series == POST_TIME_RAMP_THREE_REFS)
      {
        refinementSequence = {firstRefinement, secondRefinement, thirdRefinement};
      }
      else if (series == POST_TIME_RAMP_FOUR_REFS)
      {
        refinementSequence = {firstRefinement, secondRefinement, thirdRefinement, fourthRefinement};
      }
      else if (series == POST_TIME_RAMP_FOUR_REFS_SIMPLIFIED)
      {
        // the commented-out fourth refinement series definitely fails for patch test 12.  The one below it represents my effort to reduce the number of refinements…
//        vector<GlobalIndexType> fourthRefinementSimplified {61,65,72,73,94,95,101,102,103,104,108,109,110,111,112,113,114,115,119,120,121,124,125,126,127,133,137,139,143,153,155,157,159,171,173,174,175,185,187,188,189,190,191,192,194,198,204,208,210,212,214};
        vector<GlobalIndexType> fourthRefinementSimplified {61,65,72,73,101,102,103,104,108,109,110,111,112,157,159,171,173,174,175,185,187,188,189,190,191,192,194,198,204,208,210,212,214};
        refinementSequence = {firstRefinement, secondRefinement, thirdRefinement, fourthRefinementSimplified};
      }
    }
      break;
      
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported ref sequence choice");
      break;
  }
    
  auto cellTopo = CellTopology::cellTopology(CellTopology::line(), 2);
  auto refPattern = RefinementPattern::regularRefinementPattern(cellTopo);
  for (const auto &cellsToRefine : refinementSequence)
  {
    vlasovMesh->hRefine(cellsToRefine, refPattern, false); // false: don't rebuild yet (wait for 1-irregularity enforcement)
    vlasovMesh->enforceOneIrregularity(true);
    
    poissonMesh->repartitionAndRebuild();
  }
//  {
//    // DEBUGGING:
//    std::vector<GlobalIndexType> cellsOfInterest {931,937,813,799,1037};
//    auto vlasovTopo = vlasovMesh->getTopology();
//    for (GlobalIndexType cellID : cellsOfInterest)
//    {
//      auto cell = vlasovTopo->getCell(cellID);
//      if (rank == 0)
//      {
//        cell->printNeighborInfo(vlasovTopo);
//      }
//    }
//    for (auto cellID : cellsOfInterest)
//    {
//      CellPtr cell = vlasovMesh->getTopology()->getCell(cellID);
//      auto parent = cell->getParent();
//      auto ordinalInParent = parent->childOrdinal(cellID);
////      auto grandparent = parent->getParent();
////      auto ordinalInGrandParent = grandparent->childOrdinal(parent->cellIndex());
//
//      if (rank == 0)
//      {
//        std::cout << "cell " << cellID << " is child " << ordinalInParent << " of cell " << parent->cellIndex() << std::endl;
//      }
//      // << ", which is child " << ordinalInGrandParent << " of cell " << grandparent->cellIndex();
//    }
//  }
  
//  {
//    // DEBUGGING:
//    vector<GlobalIndexType> cellIDs = refinementSequence[refinementSequence.size()-1];
//    for (auto cellID : cellIDs)
//    {
//      CellPtr cell = vlasovMesh->getTopology()->getCell(cellID);
//      auto parent = cell->getParent();
//      auto ordinalInParent = parent->childOrdinal(cellID);
////      auto grandparent = parent->getParent();
////      auto ordinalInGrandParent = grandparent->childOrdinal(parent->cellIndex());
//
//      if (rank == 0)
//      {
//        std::cout << "cell " << cellID << " is child " << ordinalInParent << " of cell " << parent->cellIndex() << std::endl;
//      }
//      // << ", which is child " << ordinalInGrandParent << " of cell " << grandparent->cellIndex();
//    }
//  }

  
//  {
//    // DEBUGGING:
//    vector<GlobalIndexType> cellIDs = secondRefinement;
//    for (auto cellID : cellIDs)
//    {
//      CellPtr cell = vlasovMesh->getTopology()->getCell(cellID);
//      auto parent = cell->getParent();
//      auto ordinalInParent = parent->childOrdinal(cellID);
////      auto grandparent = parent->getParent();
////      auto ordinalInGrandParent = grandparent->childOrdinal(parent->cellIndex());
//
//      std::cout << "cell " << cellID << " is child " << ordinalInParent << " of cell " << parent->cellIndex() << std::endl;
//      // << ", which is child " << ordinalInGrandParent << " of cell " << grandparent->cellIndex();
//    }
//  }
  
  if (rank == 0)
  {
    std::cout << "Performed debugging refinement series.  After refinements, Vlasov mesh has ";
    std::cout << vlasovMesh->numActiveElements() << " active elements; Poisson mesh has ";
    std::cout << poissonMesh->numActiveElements() << " active elements.\n";
  }
  
  if (rank == 0) std::cout << "Checking mesh parities for Poisson mesh.\n";
  checkMesh(poissonMesh);
  if (rank == 0) std::cout << "Checking mesh parities for Vlasov mesh.\n";
  checkMesh(vlasovMesh);
}

LinearTermPtr vlasovRHSForExactSoln(VlasovPoissonSpaceTimeFormulation form, FunctionPtr f_exact, FunctionPtr E_exact, const double q,
                                    const double m)
{
  // want an RHS so that f_exact is the exact solution
  auto w = form.w(); // test function for Vlasov RHS
  const int spaceDim              = form.poissonMesh()->getDimension() - 1; // space, not space-time, dim
  const int velocityDim           = form.vlasovMesh()->getDimension() - spaceDim - 1;
  const int timeCompIndexOneBased = spaceDim + 1;
  const int vxCompIndexOneBased   = spaceDim + 2;
  FunctionPtr lhsAtExactSoln      = Function::zero();
  // spatial derivatives:
  for (int i=0; i<spaceDim; i++)
  {
    auto vComp     = vxCompIndexOneBased + i;
    auto vi        = MonomialFunction::xi_n(vComp-1,1); // Annoyingly, confusingly, MonomialFunction's xi is zero-based.  TODO: make di and xi_n match in terms of one- or zero-basedness.
    lhsAtExactSoln = lhsAtExactSoln + (f_exact * vi)->di(i+1);
  }
  // temporal derivative:
  lhsAtExactSoln   = lhsAtExactSoln + f_exact->di(timeCompIndexOneBased);
  // velocity derivatives:
  vector<FunctionPtr> E_soln_vector {E_exact};
  for (int i=0; i<velocityDim; i++)
  {
    auto vComp     = vxCompIndexOneBased + i;
    lhsAtExactSoln = lhsAtExactSoln + ((q / m) * E_soln_vector[i] * f_exact)->di(vComp);
  }
  return lhsAtExactSoln * w;
}

int patchTestNumberOffset = 0;
void performDebuggingPatchTest(VlasovPoissonSpaceTimeFormulation &form, FunctionPtr f_exact, FunctionPtr E_exact, FunctionPtr VE_exact,
                               const double q, const double m, const bool overrideVlasovBCs, const bool zeroInitialConditions,
                               const bool addSourceTerm = true, const bool vlasovSolveOnly = false)
{
  const double tol = 1e-7;
  
  form.clearSolution();
  
  auto vlasovSoln  = form.vlasovSolution();
  auto poissonSoln = form.poissonSolution();
  
  auto gda = vlasovSoln->mesh()->globalDofAssignment();
  GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(gda.get());
  
//  minRule->printGlobalDofInfo();
  vlasovSoln->setWriteMatrixToMatrixMarketFile(true, "/tmp/A.dat");
  vlasovSoln->setWriteRHSToMatrixMarketFile(true, "/tmp/b.dat");
  
  auto vlasovRHS = RHS::rhs();
  vlasovSoln->setRHS(vlasovRHS); // zero RHS for vlasov, prior to any source terms from exact solution…
  
  form.setUseGeometricMultigrid(true);
  form.setIterativeSolveTolerance(tol);
  form.setIterativeSolveOutputLevel(100);
  form.setFixedPointTolerance(tol);
  
  FunctionPtr vx = form.vx();

  FunctionPtr E_soln = Function::solution(form.E(), poissonSoln);
  FunctionPtr f_soln = Function::solution(form.f(), vlasovSoln);
  
  if (addSourceTerm)
  {
    vlasovSoln->rhs()->addTerm(vlasovRHSForExactSoln(form, f_exact, E_exact, q, m));
  }
  
  if (! zeroInitialConditions)
  {
    form.setVlasovInitialAndInflowConditions(f_exact, f_exact, f_exact);
  }
  else
  {
    form.setVlasovInitialAndInflowConditions(Function::zero(), f_exact, f_exact);
  }
  form.setPotentialBCs(VE_exact, VE_exact);
  
  if (overrideVlasovBCs)
  {
    auto vlasovBC = BC::bc();
    VarPtr traceVar = form.t_n_hat();
    
    const int initialTimeTag = form.tagVlasovInitialTime();
    
    if (!zeroInitialConditions)
    {
      vlasovBC->addDirichlet(traceVar, initialTimeTag, -f_exact); // minus sign thanks to outward normal on temporal interface
    }
    else
    {
      vlasovBC->addDirichlet(traceVar, initialTimeTag, Function::zero());
    }
    
    const int minVelocityTag = form.tagVlasovMinVelocity();
    const int maxVelocityTag = form.tagVlasovMaxVelocity();
    
    vlasovBC->addDirichlet(traceVar, minVelocityTag, -f_exact * (q/m) * E_exact); // outward normal is negative
    vlasovBC->addDirichlet(traceVar, maxVelocityTag,  f_exact * (q/m) * E_exact); // outward normal is positive
    
    const int  leftInflowTag = form.tagVlasovLeftInflow();
    const int rightInflowTag = form.tagVlasovRightInflow();
      
    FunctionPtr leftTrace  = - f_exact * vx; // minus sign is thanks to outward normal
    FunctionPtr rightTrace =   f_exact * vx;
    vlasovBC->addDirichlet(traceVar, leftInflowTag,   leftTrace);
    vlasovBC->addDirichlet(traceVar, rightInflowTag, rightTrace);
    vlasovSoln->setBC(vlasovBC);
  }
  
  auto cellTopo = CellTopology::cellTopology(CellTopology::line(), 2);
  auto vlasovMesh  = vlasovSoln->mesh();
  auto poissonMesh = poissonSoln->mesh();
    
  FunctionPtr sgn = Function::sideParity();
  FunctionPtr n   = Function::normal();
  FunctionPtr tn_0 = f_exact * (vx * n->x() + n->y() + (q/m) * E_exact * n->z()) * sgn;
    
  // TODO: TRY A CALL TO updateElectricField() prior to doing the Vlasov solve within form.solve() -- may improve fixed-point convergence…
  
  if (!vlasovSolveOnly)
  {
    {
      auto E_extruded = form.E_extruded();
      
      // make sure E_extruded is ready to evaluate…
      static_cast<PhysicalSolutionExtrusion*>(E_extruded.get())->updateValues();
    }
    
    form.solve();
  }
  else
  {
    // project the E field onto Poisson solution
    std::map<int, FunctionPtr> functionMap;
    functionMap[form.E()->ID()] = E_exact;
    poissonSoln->projectOntoMesh(functionMap);
    
    {
      auto E_extruded = form.E_extruded();
      
      // make sure E_extruded is ready to evaluate…
      static_cast<PhysicalSolutionExtrusion*>(E_extruded.get())->updateValues();
    }
    
    const int rank = vlasovMesh->Comm()->MyPID();
    auto numVlasovDofs = form.vlasovSolution()->mesh()->numGlobalDofs();
    if (numVlasovDofs > 20000)
    {
      if (rank == 0)
      {
        std::cout << "numVlasovDofs = " << numVlasovDofs << "; using GMG solver.\n";
      }
      // solve vlasov using GMG solver from form:
      vlasovSoln->solve(form.constructGMGSolver());
      
    }
    else
    {
      if (rank == 0)
      {
        std::cout << "numVlasovDofs = " << numVlasovDofs << "; using direct solver.\n";
      }
      // solve vlasov using default (direct) solver:
      vlasovSoln->solve();
    }
  }
  
  FunctionPtr tn_soln = Function::solution(form.t_n_hat(), vlasovSoln, false);
  FunctionPtr tn_err = tn_0 - tn_soln;
  
  FunctionPtr VE_soln = Function::solution(form.VE(), poissonSoln);
  
  static int patchTestNumber = 0;
  patchTestNumber++;
  
  std::string vizPrefix = "debuggingPatchTest";

  FunctionPtr cellIDFunction = Teuchos::rcp(new CellIDFunction);
  
  std::ostringstream vlasovVizPath;
  vlasovVizPath << vizPrefix << patchTestNumber + patchTestNumberOffset << "_Vlasov";
  auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovVizPath.str(), "/tmp"));
  vlasovViz->exportSolution(form.vlasovSolution(), std::vector<FunctionPtr> {f_soln-f_exact, f_exact, tn_soln-tn_0, tn_0, cellIDFunction}, std::vector<string>{"f err", "f exact", "t_n err", "t_n exact", "Vlasov cell ID"});
  
  std::ostringstream poissonVizPath;
  poissonVizPath << vizPrefix << patchTestNumber + patchTestNumberOffset << "_Poisson";
  
  auto poissonViz = Teuchos::rcp(new HDF5Exporter(form.poissonMesh(), poissonVizPath.str(), "/tmp"));
  poissonViz->exportSolution(form.poissonSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact, cellIDFunction}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact", "Poisson cell ID"});
  
  double tn_err_L2 = tn_err->l2norm(vlasovSoln->mesh());
  double f_err_L2 = (f_soln-f_exact)->l2norm(vlasovSoln->mesh());
  double f_L2     = f_exact->l2norm(vlasovSoln->mesh());
  double tn_L2    = tn_0->l2norm(vlasovSoln->mesh());
  
  int rank = vlasovMesh->Comm()->MyPID();
  if (rank == 0)
  {
    std::cout << "Performed debugging patch test " << patchTestNumber + patchTestNumberOffset << std::endl;
    std::cout << "tn_err_L2 = " << tn_err_L2 << std::endl;
    std::cout << "f_err_L2  = " << f_err_L2 << std::endl;
    
    std::cout << "tn rel err = " << tn_err_L2 / tn_L2 << std::endl;
    std::cout << "f rel err  = " << f_err_L2 / f_L2 << std::endl;
    
    static std::map<int,double> f_errs;
    f_errs[patchTestNumber+patchTestNumberOffset] = f_err_L2;
    
    for (auto f_err : f_errs)
    {
      std::cout << "Patch test " << f_err.first  << " f_err:                    " << f_err.second << std::endl;
    }
  }
  
  {
    std::ostringstream vlasovBestApproximationVizPath;
    vlasovBestApproximationVizPath << vizPrefix << patchTestNumber + patchTestNumberOffset << "BestApproximation_Vlasov";
    // project f, t_n onto vlasov solution
    std::map<int, FunctionPtr> functionMap;
    functionMap[form.f()->ID()] = f_exact;
    functionMap[form.t_n_hat()()->ID()] = tn_0;
    vlasovSoln->projectOntoMesh(functionMap);
    auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovBestApproximationVizPath.str(), "/tmp"));
    vlasovViz->exportSolution(form.vlasovSolution());
    
    double tn_err_L2 = tn_err->l2norm(vlasovSoln->mesh());
    double f_err_L2 = (f_soln-f_exact)->l2norm(vlasovSoln->mesh());

    static std::map<int,double> f_best_errs;
    static std::map<int,double> tn_best_errs;
    f_best_errs[patchTestNumber+patchTestNumberOffset]  = f_err_L2;
    tn_best_errs[patchTestNumber+patchTestNumberOffset] = tn_err_L2;
    
    if (rank == 0)
    {
      for (auto f_err : f_best_errs)
      {
        std::cout << "Patch test " << f_err.first  << " best approximation f_err: " << f_err.second << std::endl;
      }
      
      for (auto tn_err : tn_best_errs)
      {
        std::cout << "Patch test " << tn_err.first  << " best approximation tn_err: " << tn_err.second << std::endl;
      }
    }
  }
  
}

int patchTestNumberOffsetWithSeparateBCs = 0;
void performDebuggingPatchTestWithSeparateBCs(VlasovPoissonSpaceTimeFormulation &form, FunctionPtr f_exact, FunctionPtr E_exact, FunctionPtr VE_exact,
                                              FunctionPtr f_initialCondition, FunctionPtr f_xInflowCondition, FunctionPtr f_vMinBoundaryCondition, FunctionPtr f_vMaxBoundaryCondition,
                                              const double q, const double m,
                                              const bool addSourceTerm, const bool vlasovSolveOnly)
{
//  const double tol = 1e-7;
  
  form.clearSolution();
  
  auto vlasovSoln  = form.vlasovSolution();
  auto poissonSoln = form.poissonSolution();
  
  auto gda = vlasovSoln->mesh()->globalDofAssignment();
  GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(gda.get());
  
  minRule->printGlobalDofInfo();
  vlasovSoln->setWriteMatrixToMatrixMarketFile(true, "/tmp/A_sep.dat");
  vlasovSoln->setWriteRHSToMatrixMarketFile(true, "/tmp/b_sep.dat");
  
  auto vlasovRHS = RHS::rhs();
  vlasovSoln->setRHS(vlasovRHS); // zero RHS for vlasov, prior to any source terms from exact solution…
  
//  form.setUseGeometricMultigrid(true);
//  form.setIterativeSolveTolerance(tol);
//  form.setIterativeSolveOutputLevel(100);
//  form.setSubstepsForFixedPoint(1);
  form.setFixedPointTolerance(1e-7);
  
  FunctionPtr vx = form.vx();

  FunctionPtr E_soln = Function::solution(form.E(), poissonSoln);
  FunctionPtr f_soln = Function::solution(form.f(), vlasovSoln);
  
  if (addSourceTerm)
  {
    vlasovSoln->rhs()->addTerm(vlasovRHSForExactSoln(form, f_exact, E_exact, q, m));
  }
  
  form.setPotentialBCs(VE_exact, VE_exact);
  
  // set vlasov bcs:
  {
    auto vlasovBC = BC::bc();
    VarPtr traceVar = form.t_n_hat();
    
    const int initialTimeTag = form.tagVlasovInitialTime();
    
    vlasovBC->addDirichlet(traceVar, initialTimeTag, -f_initialCondition); // minus sign thanks to outward normal on temporal interface
    
    const int minVelocityTag = form.tagVlasovMinVelocity();
    const int maxVelocityTag = form.tagVlasovMaxVelocity();
    
    vlasovBC->addDirichlet(traceVar, minVelocityTag, -f_vMinBoundaryCondition * (q/m) * E_exact); // outward normal is negative
    vlasovBC->addDirichlet(traceVar, maxVelocityTag,  f_vMaxBoundaryCondition * (q/m) * E_exact); // outward normal is positive
    
    const int  leftInflowTag = form.tagVlasovLeftInflow();
    const int rightInflowTag = form.tagVlasovRightInflow();
      
    FunctionPtr leftTrace  = - f_xInflowCondition * vx; // minus sign is thanks to outward normal
    FunctionPtr rightTrace =   f_xInflowCondition * vx;
    vlasovBC->addDirichlet(traceVar, leftInflowTag,   leftTrace);
    vlasovBC->addDirichlet(traceVar, rightInflowTag, rightTrace);
    vlasovSoln->setBC(vlasovBC);
  }
  
  auto cellTopo = CellTopology::cellTopology(CellTopology::line(), 2);
  auto vlasovMesh  = vlasovSoln->mesh();
  auto poissonMesh = poissonSoln->mesh();
    
  FunctionPtr sgn = Function::sideParity();
  FunctionPtr n   = Function::normal();
  FunctionPtr tn_0 = f_exact * (vx * n->x() + n->y() + (q/m) * E_exact * n->z()) * sgn;
    
  // TODO: TRY A CALL TO updateElectricField() prior to doing the Vlasov solve within form.solve() -- may improve fixed-point convergence…
  
  if (!vlasovSolveOnly)
  {
    {
      auto E_extruded = form.E_extruded();
      
      // make sure E_extruded is ready to evaluate…
      static_cast<PhysicalSolutionExtrusion*>(E_extruded.get())->updateValues();
    }
    
    form.solve();
  }
  else
  {
    // project the E field onto Poisson solution
    std::map<int, FunctionPtr> functionMap;
    functionMap[form.E()->ID()] = E_exact;
    poissonSoln->projectOntoMesh(functionMap);
    
    {
      auto E_extruded = form.E_extruded();
      
      // make sure E_extruded is ready to evaluate…
      static_cast<PhysicalSolutionExtrusion*>(E_extruded.get())->updateValues();
    }
    
    // solve vlasov using default (direct) solver:
    vlasovSoln->solve();
  }
  
  FunctionPtr tn_soln = Function::solution(form.t_n_hat(), vlasovSoln, false);
  FunctionPtr tn_err = tn_0 - tn_soln;
  
  FunctionPtr VE_soln = Function::solution(form.VE(), poissonSoln);
  
  static int patchTestNumber = 0;
  patchTestNumber++;
  
  std::string vizPrefix = "debuggingPatchTestWithSeparateBCs";

  FunctionPtr cellIDFunction = Teuchos::rcp(new CellIDFunction);
  
  std::ostringstream vlasovVizPath;
  vlasovVizPath << vizPrefix << patchTestNumber + patchTestNumberOffsetWithSeparateBCs << "_Vlasov";
  auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovVizPath.str(), "/tmp"));
  vlasovViz->exportSolution(form.vlasovSolution(), std::vector<FunctionPtr> {f_soln-f_exact, f_exact, tn_soln-tn_0, tn_0, cellIDFunction}, std::vector<string>{"f err", "f exact", "t_n err", "t_n exact", "Vlasov cell ID"});
  
  std::ostringstream poissonVizPath;
  poissonVizPath << vizPrefix << patchTestNumber + patchTestNumberOffsetWithSeparateBCs << "_Poisson";
  
  auto poissonViz = Teuchos::rcp(new HDF5Exporter(form.poissonMesh(), poissonVizPath.str(), "/tmp"));
  poissonViz->exportSolution(form.poissonSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact, cellIDFunction}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact", "Poisson cell ID"});
  
  double tn_err_L2 = tn_err->l2norm(vlasovSoln->mesh());
  double f_err_L2 = (f_soln-f_exact)->l2norm(vlasovSoln->mesh());
  double f_L2     = f_exact->l2norm(vlasovSoln->mesh());
  double tn_L2    = tn_0->l2norm(vlasovSoln->mesh());
  
  int rank = vlasovMesh->Comm()->MyPID();
  if (rank == 0)
  {
    std::cout << "Performed debugging patch test (with separate BCs) " << patchTestNumber + patchTestNumberOffsetWithSeparateBCs << std::endl;
    std::cout << "tn_err_L2 = " << tn_err_L2 << std::endl;
    std::cout << "f_err_L2  = " << f_err_L2 << std::endl;
    
    std::cout << "tn rel err = " << tn_err_L2 / tn_L2 << std::endl;
    std::cout << "f rel err  = " << f_err_L2 / f_L2 << std::endl;
    
    static std::map<int,double> f_errs;
    f_errs[patchTestNumber+patchTestNumberOffsetWithSeparateBCs] = f_err_L2;
    
    for (auto f_err : f_errs)
    {
      std::cout << "Patch test " << f_err.first  << " f_err:                    " << f_err.second << std::endl;
    }
  }
  
  {
    std::ostringstream vlasovBestApproximationVizPath;
    vlasovBestApproximationVizPath << vizPrefix << patchTestNumber + patchTestNumberOffsetWithSeparateBCs << "BestApproximation_Vlasov";
    // project f, t_n onto vlasov solution
    std::map<int, FunctionPtr> functionMap;
    functionMap[form.f()->ID()] = f_exact;
    functionMap[form.t_n_hat()()->ID()] = tn_0;
    vlasovSoln->projectOntoMesh(functionMap);
    auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovBestApproximationVizPath.str(), "/tmp"));
    vlasovViz->exportSolution(form.vlasovSolution());
    
    double tn_err_L2 = tn_err->l2norm(vlasovSoln->mesh());
    double f_err_L2 = (f_soln-f_exact)->l2norm(vlasovSoln->mesh());

    static std::map<int,double> f_best_errs;
    static std::map<int,double> tn_best_errs;
    f_best_errs[patchTestNumber+patchTestNumberOffsetWithSeparateBCs]  = f_err_L2;
    tn_best_errs[patchTestNumber+patchTestNumberOffsetWithSeparateBCs] = tn_err_L2;
    
    for (auto f_err : f_best_errs)
    {
      std::cout << "Patch test " << f_err.first  << " best approximation f_err: " << f_err.second << std::endl;
    }
    
    for (auto tn_err : tn_best_errs)
    {
      std::cout << "Patch test " << tn_err.first  << " best approximation tn_err: " << tn_err.second << std::endl;
    }
  }
  
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  const int rank = mpiSession.getRank();
  
  {
    mpiSession.barrier();
    
    bool implicit = true;
    double k_x    = 0.2; // wavelength
    double u0     = 3.0; // initial drift velocity
    double gamma  = 1e-3; // initial perturbation
    double sigma  = 1.0; // temperature
    double n0     = 1.0; // computational density
    
    double q      =  1.0;
    double eps0   =  1.0;
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  2 * M_PI / k_x;
    double t0     =  0.0;
    double tWidth =  1.0;
    
    const double DEFAULT_V0     = 0.50;
    const double DEFAULT_VWIDTH = 1.00;
    double v0     = std::numeric_limits<double>::infinity();
    double vWidth = std::numeric_limits<double>::infinity();
    
    bool enforceQuasiNeutrality = true;
    bool useSerendipityBasis = true;
    
    bool useAdaptivity = false;
    int numRefinements = 3;
    
  //  double refinementThreshold   = 0.999999; // essentially, refine just the maximum error element (and any that thanks to solution symmetry have same error)
  //  double unrefinementThreshold = 1.000001; // similarly, unrefine just the minimum-error parent element
    double refinementThreshold = 0.80; // refine elements with at least 80% of the maximum per-element error
    int maxElements = 64 * 64; // do unrefinements to keep total mesh size smaller than this
    bool gateRefinements = true;
    double doerflerTarget = 0; // if set to positive value by user, use Doerfler instead of greedy strategy
    
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
//    bool useGeometricMultigrid = true;
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    double iterativeSolveMaxIters  = 10000;
    
    double initialTimeDivision = -1;
    
    int maxDofsForDirectSolve = 20000;
    int numUniformRefinements = 0;
    int minSubsteps = 5;  // for fixed-point iteration.
    int maxSubsteps = 15; // for fixed-point iteration.
    double fixedPointTol = 1e-8;
    int vlasovTraceEnrichment = -1; // if left unset by user, will default to 0 for all problems but cold diode, where it is 1 (at least for now).
    double coldDiodeSigma = -1.0; // will be set automatically below if not specified by user
    bool performDebugRefinementSeries = false;
    bool runPatchTests = false;
    int patchTestToRun = -1; // -1 indicates ALL if runPatchTests is true
    double bcResolutionTolerance = 1e-3;  // only applies if doing an adaptive solve
    
    std::string resultsFilePrefix = "";
    
    cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("tElements", &tElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("bcResolutionTolerance", &bcResolutionTolerance);
    cmdp.setOption("numUniformRefinements", &numUniformRefinements);
    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);
    cmdp.setOption("resultsFilePrefix", &resultsFilePrefix);
    cmdp.setOption("u0", &u0);
    cmdp.setOption("v0", &v0, "minimum velocity coordinate in domain");
    cmdp.setOption("vWidth", &vWidth, "span of velocity coordinates in domain");
    cmdp.setOption("sigma", &coldDiodeSigma); // sigma for cold diode -- ignored for other problems
    cmdp.setOption("maxElements", &maxElements);
    cmdp.setOption("maximumRefinementLevel", &maximumRefinementLevel);
    cmdp.setOption("minSubsteps", &minSubsteps);
    cmdp.setOption("maxSubsteps", &maxSubsteps);
    cmdp.setOption("fixedPointTol", &fixedPointTol);
    cmdp.setOption("viz", "noViz", &saveViz);
    cmdp.setOption("initialTimeDivision", &initialTimeDivision);
    cmdp.setOption("gateRefinements", "dontGateRefinements", &gateRefinements);
    cmdp.setOption("outputFrequency", &outputFrequency);
    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("timeStepC0", &timeStepC0);
    cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    cmdp.setOption("numRefinements", &numRefinements);
    cmdp.setOption("problem", &problemChoiceString);
    cmdp.setOption("doerflerTarget", &doerflerTarget);
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
//    cmdp.setOption("useGeometricMultigrid", "useDirectSolver", &useGeometricMultigrid);
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
    cmdp.setOption("maxDofsForDirectSolve", &maxDofsForDirectSolve);
    cmdp.setOption("vlasovTraceEnrichment", &vlasovTraceEnrichment);
    cmdp.setOption("performDebugRefinementSeries", "dontPerformDebugRefinementSeries", &performDebugRefinementSeries);
    cmdp.setOption("runPatchTests", "dontRunPatchTests", &runPatchTests);
    cmdp.setOption("patchTestToRun", &patchTestToRun);
    cmdp.setOption("useSerendipityBasis", "useFullTensorBasis", &useSerendipityBasis);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    problemChoice = getProblemChoice(problemChoiceString);
    
    // for cold diode, we non-dimensionalize; we modify the below accordingly so that we can output dimensional values to results file.
    double velocityScaling  = 1.;
    double timeScaling      = 1.;
    double spatialScaling   = 1.;
    double EScaling         = 1.;
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
    
    if (problemChoice != COLD_DIODE)
    {
      if (vWidth == std::numeric_limits<double>::infinity())
      {
        vWidth = 20.0;
      }
      if (v0 == std::numeric_limits<double>::infinity())
      {
        v0 = -10.0;
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
      
      const double permittivity    = 8.85418782e-12; // permittivity of free space
      
      nScaling = J / (e * v_beam); // number density of plasma
      
      const double omega_p = sqrt( q0 * q0 * nScaling / (m0 * permittivity));
      
      xWidth = 0.01 / spatialScaling;
      
      if (vWidth == std::numeric_limits<double>::infinity())
      {
        vWidth = 1.00;
      }
      if (v0 == std::numeric_limits<double>::infinity())
      {
        v0 = 0.50;
      }
      
      u0     = 1.0;
      
      int vElementsAfterUniformRefinements = vElements << numUniformRefinements;
      
      if (coldDiodeSigma == -1)
      {
        sigma  = vWidth / vElementsAfterUniformRefinements; // let sigma be the width of one element
      //    sigma  = 1e6 / velocityScaling;
      }
      else
      {
        sigma = coldDiodeSigma;
      }
      
      q      = -1.0;
      
      // eps0 is a weight that enters VlasovPoissonSpaceTimeFormulation; we use this to scale the problem:
      eps0   = 1.0 / (omega_p * omega_p * timeScaling * timeScaling);
      usePeriodicBCs = false;
      enforceQuasiNeutrality = false;
      
      if (initialTimeDivision < 0)
      {
        initialTimeDivision = 0.25;
      }
      
      if (rank == 0)
      {
        std::cout << "E scaling :         " << EScaling << std::endl;
        std::cout << "potential scaling : " << potentialScaling << std::endl;
        std::cout << "spatial scaling :   " << spatialScaling << std::endl;
        std::cout << "n scaling :         " << nScaling << std::endl;
        std::cout << "temporal scaling :  " << timeScaling << std::endl;
        std::cout << "velocity scaling :  " << velocityScaling << std::endl;
        std::cout << "computational domain width: " << xWidth << std::endl;
        std::cout << "physical domain width: " << xWidth * spatialScaling << std::endl;
        std::cout << "initial time division: " << initialTimeDivision << std::endl;
        
        std::cout << "sigma = " << sigma << std::endl;
        std::cout << "u0 = " << u0 << std::endl;
        std::cout << "n0 = " << n0 << std::endl;
      }
    }
    
    if (initialTimeDivision < 0)
    {
      initialTimeDivision = 0; // no extra initial time elements (can be used with a time ramp if positive)
    }
    
    VlasovPoissonSpaceTimeFormulation::VlasovPoissonSpaceTimeFormulationChoice formulationChoice;
    using FormulationChoice = Camellia::VlasovPoissonSpaceTimeFormulation::VlasovPoissonSpaceTimeFormulationChoice;
    formulationChoice = FormulationChoice::ULTRAWEAK;
    
    VlasovPoissonSpaceTimeFormulation form(formulationChoice, q, eps0, m, x0, xWidth, xElements, t0, tWidth, tElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, useSerendipityBasis, vlasovTraceEnrichment, initialTimeDivision);
    
    if (performDebugRefinementSeries)
    {
      performDebuggingRefinementSeries(form);
    }
    
    if (runPatchTests)
    {
      std::map<int, bool> runPatchTestNumber;
      if (patchTestToRun == -1)
      {
        const int MAX_PATCH_TEST_NUMBER = 17;
        for (int i=1; i<=MAX_PATCH_TEST_NUMBER; i++)
        {
          runPatchTestNumber[i] = true;
        }
      }
      else
      {
        runPatchTestNumber[patchTestToRun] = true;
        patchTestNumberOffset                = patchTestToRun - 1;
        patchTestNumberOffsetWithSeparateBCs = patchTestToRun - 1;
      }
      
      // TEST 1
      if (runPatchTestNumber[1])
      {
        FunctionPtr x       = Function::xn(1);

        FunctionPtr f_exact   = Function::constant(0.0);
        FunctionPtr E_exact = Function::constant(1.0);
        FunctionPtr V_E_exact = -x;

        const bool overrideVlasovBCs = false;    // required if f is nonzero at velocity extrema
        const bool zeroInitialConditions = true; // for f
        performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions);
      }

      // TEST 2
      if (runPatchTestNumber[2])
      {
        FunctionPtr x         = Function::xn(1);
        FunctionPtr f_exact   = Function::constant(1.0);
        FunctionPtr E_exact   =   (q / eps0 * vWidth) * x;
        FunctionPtr V_E_exact = - (q / eps0 * vWidth) / 2.0 * x * x;

        const bool overrideVlasovBCs = true;     // required if f is nonzero at velocity extrema
        const bool zeroInitialConditions = true; // for f
        performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions);
      }

      // TEST 3
      if (runPatchTestNumber[3])
    {
      // a non-patch exact solution

      FunctionPtr x         = Function::xn(1);
      FunctionPtr v         = Function::zn(1);
      FunctionPtr sin_fxn   = TrigFunctions<double>::sin(2.0 * M_PI * (v - 0.5)); // 0 at the v boundaries
      FunctionPtr f_exact   = sin_fxn * x / xWidth;
      FunctionPtr E_exact   = Function::zero();
      FunctionPtr V_E_exact = Function::zero();

      const bool overrideVlasovBCs = false;     // required if f is nonzero at velocity extrema
      const bool zeroInitialConditions = false; // for f
      performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions);
      
      const bool addSourceTerm = true;
      const bool vlasovSolveOnly = false;
      auto f_initialConditions   = f_exact; // f_exact does not vary in time
      auto f_xInflowConditions   = Function::zero(); // f is zero on the left
      auto f_vBoundaryConditions = Function::zero(); // f is zero at v boundaries
      performDebuggingPatchTestWithSeparateBCs(form, f_exact, E_exact, V_E_exact, f_initialConditions, f_xInflowConditions, f_vBoundaryConditions, f_vBoundaryConditions, q, m, addSourceTerm, vlasovSolveOnly);
    }
      
    // TEST 4
      if (runPatchTestNumber[4])
    {
      // a non-patch exact solution with E non-zero

      FunctionPtr x         = Function::xn(1);
      FunctionPtr v         = Function::zn(1);
      FunctionPtr sin_fxn   = TrigFunctions<double>::sin(2.0 * M_PI * (v - 0.5)); // 0 at the v boundaries
      FunctionPtr f_exact   = sin_fxn * sin_fxn * x / xWidth; // integral in v of this is x / (2 * xWidth)
      FunctionPtr E_exact   = q / (4. * eps0 * xWidth) * x * x  - (q * xWidth) / (12. * eps0); // integral in x of (rho / eps0), with the constant selected such that V_E is 0 at left and right boundaries
      // set V_E to give 0 value at left and right boundaries:
      FunctionPtr V_E_exact = - q / (12. * eps0 * xWidth) * (x * x * x) + (q * xWidth) / (12. * eps0) * x;

      const bool overrideVlasovBCs = false;     // required if f is nonzero at velocity extrema
      const bool zeroInitialConditions = false; // for f
      performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions);
    }
    
    // TEST 5
      if (runPatchTestNumber[5])
    {
      // a non-patch exact solution with sinusoidal E

      FunctionPtr x           = Function::xn(1);
      FunctionPtr v           = Function::zn(1);
      FunctionPtr sin_v_fxn   = TrigFunctions<double>::sin(2.0 * M_PI * (v - 0.5)); // 0 at the v boundaries
      FunctionPtr cos_x_fxn   = TrigFunctions<double>::cos(M_PI / (2. * xWidth) * x);
      FunctionPtr f_exact     = sin_v_fxn * sin_v_fxn * cos_x_fxn; // integral in v of this is cos_x_fxn / 2

      FunctionPtr sin_x_fxn   = TrigFunctions<double>::sin(M_PI / (2. * xWidth) * x);
      const double C = - 2. * q * xWidth / (eps0 * M_PI * M_PI);
      const double D = C * xWidth;
      FunctionPtr E_exact   = (q / eps0) * (xWidth / M_PI) * sin_x_fxn + C; // integral in x of (rho / eps0), with the constant selected such that V_E is 0 at left and right boundaries
      // set V_E to give 0 value at left and right boundaries:
      FunctionPtr V_E_exact = (q / eps0) * (2. * xWidth * xWidth / (M_PI * M_PI) ) * cos_x_fxn  - C * x + D;

      {
        // DEBUGGING
        // is E = - d/dx (V_E)?
        auto mesh = form.poissonMesh();
        double err = (E_exact + V_E_exact->dx())->l2norm(mesh);
        if (rank == 0) std::cout << "err for E_exact, V_E_exact relation: " << err << std::endl;

        // is (d/dx E) = rho / eps0?
        FunctionPtr rho = (q / 2.0) * cos_x_fxn;
        err = (E_exact->dx() - rho / eps0)->l2norm(mesh);
        if (rank == 0) std::cout << "err for E_exact, rho relation: " << err << std::endl;

        // is V_E_exact(0) = 0?
        double V_L = V_E_exact->evaluate(0);
        double V_R = V_E_exact->evaluate(xWidth);
        if (rank == 0) std::cout << "V_E(0)   = " << V_L << std::endl;
        if (rank == 0) std::cout << "V_E(x_w) = " << V_R << std::endl;

      }

      const bool overrideVlasovBCs = false;     // required if f is nonzero at velocity extrema
      const bool zeroInitialConditions = false; // for f
      performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions);
    }

      // TEST 6
      if (runPatchTestNumber[6])
      {
        // a patch exact solution, zero at the v boundaries.  (Requires H^1 order of 4 for exact recovery.)

        FunctionPtr x           = Function::xn(1);
        FunctionPtr t           = Function::yn(1);
        FunctionPtr v           = Function::zn(1);
        FunctionPtr f_exact     = (v-0.5) *(1.5-v) * (x + xWidth) * t / xWidth; // integral in v of this is 1/(6 x_w) * (x + x_w) * t

        // set C, D so that V_E has 0 value at left boundary, 1.0 at right boundary:
        FunctionPtr C         = - q / (9. * eps0) * xWidth * t - 1. / xWidth;
        const double D        = 0.0;
        FunctionPtr E_exact   = q / (12 * eps0 * xWidth) * x * x * t + q / (6.*eps0) * x * t + C;
        FunctionPtr V_E_exact = - q / (36 * eps0 * xWidth) * x * x * x * t - q / (12. * eps0) * x * x * t - C * x + D;

        {
          // DEBUGGING
          // is E = - d/dx (V_E)?
          auto mesh = form.poissonMesh();
          double err = (E_exact + V_E_exact->dx())->l2norm(mesh);
          if (rank == 0) std::cout << "err for E_exact, V_E_exact relation: " << err << std::endl;

          // is (d/dx E) = rho / eps0?
          FunctionPtr rho = q / (6.0 * xWidth) * (x + xWidth) * t;
          err = (E_exact->dx() - rho / eps0)->l2norm(mesh);
          if (rank == 0) std::cout << "err for E_exact, rho relation: " << err << std::endl;

          // is V_E_exact(0) = 0?
          double V_L = V_E_exact->evaluate(0,0); // time 0
          double V_R = V_E_exact->evaluate(xWidth,0);
          if (rank == 0) std::cout << "V_E(0,0)   = " << V_L << std::endl;
          if (rank == 0) std::cout << "V_E(x_w,0) = " << V_R << std::endl;
          V_L = V_E_exact->evaluate(0,1); // time 1
          V_R = V_E_exact->evaluate(xWidth,1);
          if (rank == 0) std::cout << "V_E(0,1)   = " << V_L << std::endl;
          if (rank == 0) std::cout << "V_E(x_w,1) = " << V_R << std::endl;
        }

        const bool overrideVlasovBCs = false; // f is 0 at v_max, v_min: we don't need to override…
        const bool zeroInitialConditions = false;
        performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions);
      }
      
      // TEST 7
      if (runPatchTestNumber[7])
      {
        // a non-patch exact solution with zero E-field (but non-zero q).

        FunctionPtr x           = Function::xn(1);
        FunctionPtr t           = Function::yn(1);
        FunctionPtr v           = Function::zn(1);
        FunctionPtr sin_v_fxn   = TrigFunctions<double>::sin(2.0 * M_PI * (v - 0.5)); // 0 at the v boundaries
        FunctionPtr cos_x_fxn   = TrigFunctions<double>::cos(M_PI / (2. * xWidth) * x);
        FunctionPtr f_exact     = sin_v_fxn * cos_x_fxn * (1 - t); // integral in v of this is 0

        FunctionPtr E_exact   = Function::zero();
        // set V_E to give 0 value at left and right boundaries:
        FunctionPtr V_E_exact = Function::zero();

        {
          // DEBUGGING
          // is E = - d/dx (V_E)?
          auto mesh = form.poissonMesh();
          double err = (E_exact + V_E_exact->dx())->l2norm(mesh);
          if (rank == 0) std::cout << "err for E_exact, V_E_exact relation: " << err << std::endl;

          // is (d/dx E) = rho / eps0?
          FunctionPtr rho = Function::zero();
          err = (E_exact->dx() - rho / eps0)->l2norm(mesh);
          if (rank == 0) std::cout << "err for E_exact, rho relation: " << err << std::endl;

          // is V_E_exact(0) = 0?
          double V_L = V_E_exact->evaluate(0);
          double V_R = V_E_exact->evaluate(xWidth);
          if (rank == 0) std::cout << "V_E(0)   = " << V_L << std::endl;
          if (rank == 0) std::cout << "V_E(x_w) = " << V_R << std::endl;

        }

        const bool overrideVlasovBCs = false;     // required if f is nonzero at velocity extrema
        const bool zeroInitialConditions = false; // for f
        performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions);
        
        const bool addSourceTerm = true;
        const bool vlasovSolveOnly = false;
        auto f_initialConditions   = sin_v_fxn * cos_x_fxn;
        auto f_xInflowConditions   = sin_v_fxn * (1 - t); // f evaluated at x=0
        auto f_vBoundaryConditions = Function::zero(); // f is zero at v boundaries
        performDebuggingPatchTestWithSeparateBCs(form, f_exact, E_exact, V_E_exact, f_initialConditions, f_xInflowConditions, f_vBoundaryConditions, f_vBoundaryConditions, q, m, addSourceTerm, vlasovSolveOnly);
      }
      
      if (runPatchTestNumber[8])
      {
        // TEST 8
        // a non-patch exact solution with time-varying sinusoidal E

        FunctionPtr x           = Function::xn(1);
        FunctionPtr t           = Function::yn(1);
        FunctionPtr v           = Function::zn(1);
        FunctionPtr sin_v_fxn   = TrigFunctions<double>::sin(2.0 * M_PI * (v - 0.5)); // 0 at the v boundaries
        FunctionPtr cos_x_fxn   = TrigFunctions<double>::cos(M_PI / (2. * xWidth) * x);
        FunctionPtr f_exact     = sin_v_fxn * sin_v_fxn * cos_x_fxn * (1 - t); // integral in v of this is cos_x_fxn * (1 - t) / 2

        FunctionPtr sin_x_fxn   = TrigFunctions<double>::sin(M_PI / (2. * xWidth) * x);
        const double C = - 2. * q * xWidth / (eps0 * M_PI * M_PI);
        const double D = C * xWidth;
        FunctionPtr E_exact   = (q / eps0) * (xWidth / M_PI) * sin_x_fxn * (1-t) + C * (1-t); // integral in x of (rho / eps0), with the constant selected such that V_E is 0 at left and right boundaries
        // set V_E to give 0 value at left and right boundaries:
        FunctionPtr V_E_exact = (q / eps0) * (2. * xWidth * xWidth / (M_PI * M_PI) ) * cos_x_fxn * (1-t)  - C * x * (1-t) + D * (1-t);

        {
          // DEBUGGING
          // is E = - d/dx (V_E)?
          auto mesh = form.poissonMesh();
          double err = (E_exact + V_E_exact->dx())->l2norm(mesh);
          if (rank == 0) std::cout << "err for E_exact, V_E_exact relation: " << err << std::endl;

          // is (d/dx E) = rho / eps0?
          FunctionPtr rho = (q / 2.0) * cos_x_fxn * (1-t);
          err = (E_exact->dx() - rho / eps0)->l2norm(mesh);
          if (rank == 0) std::cout << "err for E_exact, rho relation: " << err << std::endl;

          // is V_E_exact(0) = 0?
          double V_L = V_E_exact->evaluate(0);
          double V_R = V_E_exact->evaluate(xWidth);
          if (rank == 0) std::cout << "V_E(0)   = " << V_L << std::endl;
          if (rank == 0) std::cout << "V_E(x_w) = " << V_R << std::endl;

        }

        const bool overrideVlasovBCs = false;     // required if f is nonzero at velocity extrema
        const bool zeroInitialConditions = false; // for f
        performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions);
      }
      
      if (runPatchTestNumber[9])
      {
        // TEST 9
        // a non-patch exact solution with q = 0, varying in all dimensions

        VlasovPoissonSpaceTimeFormulation form_q0(formulationChoice, 0, eps0, m, x0, xWidth, xElements, t0, tWidth, tElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, useSerendipityBasis, vlasovTraceEnrichment);

        performDebuggingRefinementSeries(form_q0);

        FunctionPtr x           = Function::xn(1);
        FunctionPtr t           = Function::yn(1);
        FunctionPtr v           = Function::zn(1);
        FunctionPtr sin_v_fxn   = TrigFunctions<double>::sin(2.0 * M_PI * (v - 0.5)); // 0 at the v boundaries
        FunctionPtr cos_x_fxn   = TrigFunctions<double>::cos(M_PI / (2. * xWidth) * x);
        FunctionPtr f_exact     = sin_v_fxn * sin_v_fxn * cos_x_fxn * (1 - t); // integral in v of this is cos_x_fxn * (1 - t) / 2

        FunctionPtr sin_x_fxn   = TrigFunctions<double>::sin(M_PI / (2. * xWidth) * x);
        FunctionPtr E_exact   = Function::zero();
        // set V_E to give 0 value at left and right boundaries:
        FunctionPtr V_E_exact = Function::zero();

        const bool overrideVlasovBCs = false;     // required if f is nonzero at velocity extrema
        const bool zeroInitialConditions = false; // for f
        performDebuggingPatchTest(form_q0, f_exact, E_exact, V_E_exact, 0, m, overrideVlasovBCs, zeroInitialConditions); // 0: q value to eliminate electric field
      }
      
      if (runPatchTestNumber[10])
      {
        // TEST 10
        // one more non-patch exact solution with time-varying sinusoidal E

        FunctionPtr x           = Function::xn(1);
        FunctionPtr t           = Function::yn(1);
        FunctionPtr v           = Function::zn(1);
        FunctionPtr sin_v_fxn   = TrigFunctions<double>::sin(2.0 * M_PI * (v - 0.5)); // 0 at the v boundaries
        FunctionPtr cos_x_fxn   = TrigFunctions<double>::cos(M_PI / (2. * xWidth) * x);
        FunctionPtr f_exact     = sin_v_fxn * sin_v_fxn * cos_x_fxn * (1 - t) + t; // integral in v of this is cos_x_fxn * (1 - t) / 2 + t

        FunctionPtr sin_x_fxn   = TrigFunctions<double>::sin(M_PI / (2. * xWidth) * x);
        FunctionPtr C = -q / (2.0 * eps0) * t - 2. * q * xWidth / (eps0 * M_PI * M_PI) * (1.-t);
        FunctionPtr D = -2. * q * xWidth * xWidth / (eps0 * M_PI * M_PI) * (1-t);
        FunctionPtr E_exact   = (q / eps0) * (xWidth / M_PI) * sin_x_fxn * (1-t) + (q / eps0) * x * t + C; // integral in x of (rho / eps0), with the constant selected such that V_E is 0 at left and right boundaries
        // set V_E to give 0 value at left and right boundaries:
        FunctionPtr V_E_exact = (q / eps0) * (2. * xWidth * xWidth / (M_PI * M_PI) ) * cos_x_fxn * (1-t)  - q/(2. * eps0) * x * x * t - C * x + D;

        {
          // DEBUGGING
          // is E = - d/dx (V_E)?
          auto mesh = form.poissonMesh();
          double err = (E_exact + V_E_exact->dx())->l2norm(mesh);
          if (rank == 0) std::cout << "err for E_exact, V_E_exact relation: " << err << std::endl;

          // is (d/dx E) = rho / eps0?
          FunctionPtr rho = (q / 2.0) * cos_x_fxn * (1-t) + q * t;
          err = (E_exact->dx() - rho / eps0)->l2norm(mesh);
          if (rank == 0) std::cout << "err for E_exact, rho relation: " << err << std::endl;

          // is V_E_exact(0) = 0?
          double V_L = V_E_exact->evaluate(0);
          double V_R = V_E_exact->evaluate(xWidth);
          if (rank == 0) std::cout << "V_E(0)   = " << V_L << std::endl;
          if (rank == 0) std::cout << "V_E(x_w) = " << V_R << std::endl;

        }

        const bool overrideVlasovBCs = true;     // required if f is nonzero at velocity extrema
        const bool zeroInitialConditions = false; // for f
        performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions);
      }
      if (runPatchTestNumber[11])
      {
        // TEST 11
        // not a manufactured solution: try doing the standard cold diode in this context, just to rule out the possibility that we're
        // doing something different in the patch tests…
        bool overrideVlasovBCs = true; // need this true if f(v_max) != 0 or f(v_min) != 0.
        
//        FunctionPtr temporalRamp = Teuchos::rcp( new TemporalRamp1D1T(0.1) ); // ramp to 1 from t=0 to t=0.1
        FunctionPtr f = Teuchos::rcp( new Maxwellian1T1V(u0, sigma, n0) ); // BC for f on the left
        FunctionPtr E = Function::zero();
        FunctionPtr V_E = Function::zero();
        
        {
          // DEBUGGING
          auto vlasovMesh = form.vlasovMesh();
          const auto & phaseCellIDs = vlasovMesh->cellIDsInPartition();
          bool success = true;
          
          if (rank == 0) std::cout << "Performing data path versus view path comparison for Maxwellian.\n";
          for (auto phaseCellID : phaseCellIDs)
          {
            BasisCachePtr basisCache = BasisCache::basisCacheForCell(vlasovMesh, phaseCellID);
            Teuchos::basic_FancyOStream<char> out(Teuchos::rcp(&std::cout,false));
            testFunctionDataPathMatchesViewPath1T1V(f, basisCache, 1e-12, out, success);
          }
          if (rank == 0)
          {
            if (success)
              std::cout << "SUCCESS: Data path matched view path.\n";
            else
              std::cout << "FAILURE: Data path did not match view path.\n";
          }
        }
        
        const bool zeroInitialConditions = true;
        const bool addSourceTerm = false;
        performDebuggingPatchTest(form, f, E, V_E, q, m, overrideVlasovBCs, zeroInitialConditions, addSourceTerm);
      }
      
      if (runPatchTestNumber[12])
      {
        // TEST 12
        // PURE VLASOV SOLVE
        // Set a zero E field, and a patch solution for f.
        FunctionPtr x           = Function::xn(1);
        FunctionPtr t           = Function::yn(1);
        FunctionPtr v           = Function::zn(1);
        FunctionPtr f_exact     = x - v * t;  // v df / dx + df / dt = v - v = 0

        FunctionPtr E_exact   = Function::zero();
        FunctionPtr V_E_exact = Function::zero();

        const bool overrideVlasovBCs = false; // f is 0 at v_max, v_min: we don't need to override…
        const bool zeroInitialConditions = false;
        const bool addSourceTerm = false;
        const bool pureVlasovSolve = true;
        performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions, addSourceTerm, pureVlasovSolve);
      }
      
      if (runPatchTestNumber[13])
      {
        // TEST 13
        // PURE VLASOV SOLVE
        // Set a constant E field, and a patch solution for f.
        FunctionPtr x           = Function::xn(1);
        FunctionPtr t           = Function::yn(1);
        FunctionPtr v           = Function::zn(1);
        FunctionPtr E_exact     = Function::constant(m/q);
        FunctionPtr f_exact     = 2 * x + v - t - v * v;  // v df / dx + df / dt + q / m E df/dv = 2 v - 1 + 1  - 2 v = 0

        FunctionPtr V_E_exact = Function::zero(); // pure Vlasov solve: V_E_exact does not actually enter the solve…

        const bool overrideVlasovBCs = true; // f is NOT 0 at v_max, v_min: we need to override…
        const bool zeroInitialConditions = false;
        const bool addSourceTerm = false;
        const bool pureVlasovSolve = true;
        performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions, addSourceTerm, pureVlasovSolve);
      }
      
      if (runPatchTestNumber[14])
      {
        // TEST 14
        // PURE VLASOV SOLVE
        // Set a variable E field, and a patch solution for f.
        FunctionPtr x           = Function::xn(1);
        FunctionPtr t           = Function::yn(1);
        FunctionPtr v           = Function::zn(1);
        FunctionPtr E_exact     = (m/q) * (x + t);
        FunctionPtr f_exact     = - 2 * v + 2 * x * t - v * v + x * x + t * t;  // v df / dx + df / dt + q / m E df/dv = 2tv + 2xv + 2x + 2t - 2x - 2t - 2xv - 2tv = 0

        FunctionPtr V_E_exact = Function::zero(); // pure Vlasov solve: V_E_exact does not actually enter the solve…

        const bool overrideVlasovBCs = true; // f is NOT 0 at v_max, v_min: we need to override…
        const bool zeroInitialConditions = false;
        const bool addSourceTerm = false;
        const bool pureVlasovSolve = true;
        performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions, addSourceTerm, pureVlasovSolve);
        
        auto f_initialConditions   = - 2 * v - v * v + x * x; // f evaluated at t=0
        auto f_xInflowConditions   = - 2 * v - v * v + t * t; // f evaluated at x=0
        const auto v_min = v0;
        const auto v_max = v0 + vWidth;
        auto f_vMinBoundaryConditions = - 2 * v_min + 2 * x * t - v_min * v_min + x * x + t * t;
        auto f_vMaxBoundaryConditions = - 2 * v_max + 2 * x * t - v_max * v_max + x * x + t * t;
        performDebuggingPatchTestWithSeparateBCs(form, f_exact, E_exact, V_E_exact, f_initialConditions, f_xInflowConditions, f_vMinBoundaryConditions, f_vMaxBoundaryConditions, q, m, addSourceTerm, pureVlasovSolve);
      }
      
      if (runPatchTestNumber[15])
      {
        // TEST 15
        // not a manufactured solution: try doing the standard cold diode in this context, just to rule out the possibility that we're
        // doing something different in the patch tests…
        // LIKE TEST 11, but without using our custom function subclass
        bool overrideVlasovBCs = true; // need this true if f(v_max) != 0 or f(v_min) != 0.
        
        double weight = n0 / sqrt (2. * M_PI) * sigma;
        FunctionPtr vx       = Function::zn(1); // 1D1T1V: v_x is the "z" dimension
        FunctionPtr exponent = - 0.5 * (vx - u0) * (vx - u0) / (sigma * sigma);
        FunctionPtr f = weight * Exp<double>::exp(exponent);
        FunctionPtr E = Function::zero();
        FunctionPtr V_E = Function::zero();
        
        {
          // DEBUGGING
          auto vlasovMesh = form.vlasovMesh();
          const auto & phaseCellIDs = vlasovMesh->cellIDsInPartition();
          bool success = true;
          
          if (rank == 0) std::cout << "Performing data path versus view path comparison for Maxwellian.\n";
          for (auto phaseCellID : phaseCellIDs)
          {
            BasisCachePtr basisCache = BasisCache::basisCacheForCell(vlasovMesh, phaseCellID);
            Teuchos::basic_FancyOStream<char> out(Teuchos::rcp(&std::cout,false));
            testFunctionDataPathMatchesViewPath1T1V(f, basisCache, 1e-12, out, success);
          }
          if (rank == 0)
          {
            if (success)
              std::cout << "SUCCESS: Data path matched view path.\n";
            else
              std::cout << "FAILURE: Data path did not match view path.\n";
          }
        }
        
        const bool zeroInitialConditions = true;
        const bool addSourceTerm = false;
        performDebuggingPatchTest(form, f, E, V_E, q, m, overrideVlasovBCs, zeroInitialConditions, addSourceTerm);
      }
      
      if (runPatchTestNumber[16])
      {
        // TEST 16
        // not a manufactured solution: try doing the first Vlasov solve of standard cold diode in this context, just to rule out the possibility that we're
        // doing something different in the patch tests…
        bool overrideVlasovBCs = true; // need this true if f(v_max) != 0 or f(v_min) != 0.
        
        FunctionPtr temporalRamp = Teuchos::rcp( new TemporalRamp1D1T(0.25) ); // ramp to 1 from t=0 to t=0.1
        FunctionPtr f = Teuchos::rcp( new Maxwellian1T1V(u0, sigma, n0) ); // BC for f on the left
        FunctionPtr E = Function::zero();
        FunctionPtr V_E = Function::zero();
        
        {
          // DEBUGGING
          auto vlasovMesh = form.vlasovMesh();
          const auto & phaseCellIDs = vlasovMesh->cellIDsInPartition();
          bool success = true;
          
          if (rank == 0) std::cout << "Performing data path versus view path comparison for Maxwellian.\n";
          for (auto phaseCellID : phaseCellIDs)
          {
            BasisCachePtr basisCache = BasisCache::basisCacheForCell(vlasovMesh, phaseCellID);
            Teuchos::basic_FancyOStream<char> out(Teuchos::rcp(&std::cout,false));
            testFunctionDataPathMatchesViewPath1T1V(f, basisCache, 1e-12, out, success);
          }
          if (rank == 0)
          {
            if (success)
              std::cout << "SUCCESS: Data path matched view path.\n";
            else
              std::cout << "FAILURE: Data path did not match view path.\n";
          }
        }
        
        const bool zeroInitialConditions = true;
        const bool addSourceTerm = false;
        const bool pureVlasovSolve = true;
        performDebuggingPatchTest(form, f * temporalRamp, E, V_E, q, m, overrideVlasovBCs, zeroInitialConditions, addSourceTerm, pureVlasovSolve);
      }
      
      if (runPatchTestNumber[17])
      {
        // TEST 17
        // PURE VLASOV SOLVE
        // Set a zero E field, and a patch solution for f.
        FunctionPtr x           = Function::xn(1);
        FunctionPtr t           = Function::yn(1);
        FunctionPtr v           = Function::zn(1);
        FunctionPtr f_exact     = x + t;

        FunctionPtr E_exact   = Function::zero();
        FunctionPtr V_E_exact = Function::zero();

        const bool overrideVlasovBCs = true;
        const bool zeroInitialConditions = false;
        const bool addSourceTerm = true;
        const bool pureVlasovSolve = true;
        performDebuggingPatchTest(form, f_exact, E_exact, V_E_exact, q, m, overrideVlasovBCs, zeroInitialConditions, addSourceTerm, pureVlasovSolve);
      }
      
    // TODO: try testing Vlasov solve only, with a projected E field and zero forcing
      exit(0);
    }
    
    using std::vector;
    using std::pair;
    vector<pair<pair<FunctionPtr,FunctionPtr>,int> > bcFunctionsToResolve; // < <exact function, solution function>, tag> -- only used when useAdaptivity is true
    
    // we'll use this function to allow labeling of visualization output
    FunctionPtr cellIDFunction = Teuchos::rcp(new CellIDFunction);
    
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
      VlasovPoissonSpaceTimeFormulation fineForm(formulationChoice, q, eps0, m, x0, xWidth, xElements, t0, tWidth, tElements, v0, vWidth, vElements, overkillH1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, useSerendipityBasis, vlasovTraceEnrichment);
      // TODO: do uniform refinements on fineForm meshes: numUniformRefinements+overkillRefinementLevel
      
      convertOverkillSolution(fineForm, overkillSolutionDir);
      return 0; // exit
    }
    else if (runOverkillVersusAdaptiveComparison)
    {
      VlasovPoissonSpaceTimeFormulation fineForm    (formulationChoice, q, eps0, m, x0, xWidth, xElements, t0, tWidth, tElements, v0, vWidth, vElements, overkillH1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, useSerendipityBasis, vlasovTraceEnrichment);
      // TODO: do uniform refinements on fineForm meshes: numUniformRefinements+overkillRefinementLevel
      VlasovPoissonSpaceTimeFormulation adaptiveForm(formulationChoice, q, eps0, m, x0, xWidth, xElements, t0, tWidth, tElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, useSerendipityBasis, vlasovTraceEnrichment);
      // TODO: do uniform refinements on adaptiveForm meshes: numUniformRefinements
      VlasovPoissonSpaceTimeFormulation modestForm  (formulationChoice, q, eps0, m, x0, xWidth, xElements, t0, tWidth, tElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, useSerendipityBasis, vlasovTraceEnrichment);
      // TODO: do uniform refinements on modestForm meshes: numUniformRefinements+uniformMeshRefinementLevel
            
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
        FunctionPtr timeRamp = Teuchos::rcp( new TemporalRamp1D1T(initialTimeDivision) );
        
        FunctionPtr f_leftInflow;
        if (initialTimeDivision > 0)
        {
          if (rank == 0)
          {
            std::cout << "Using time ramp of width " << initialTimeDivision << std::endl;
          }
          f_leftInflow = fM_u0 * timeRamp;
        }
        else
        {
          f_leftInflow = fM_u0;
        }
        form.setVlasovInitialAndInflowConditions(Function::zero(), f_leftInflow, Function::zero()); // the right BC is only for negative velocity, so won't actually get triggered on our mesh
        
        if (useAdaptivity)
        {
          auto leftInflowTag = form.tagVlasovLeftInflow();
          
          FunctionPtr sgn = Function::sideParity();
          FunctionPtr n = Function::normal();
          FunctionPtr n_x = n->spatialComponent(1); // 1-based
          FunctionPtr vx = form.vx();
          
          FunctionPtr tn_leftInflowExact = f_leftInflow * sgn * n_x * vx;
          const bool weightFluxesBySideParity = true;
          FunctionPtr tn_soln = Function::solution(form.t_n_hat(), form.vlasovSolution(), weightFluxesBySideParity);
          bcFunctionsToResolve.push_back({{tn_leftInflowExact, tn_soln},leftInflowTag});
        }
      }
    }
    else
    {
      form.setVlasovInitialConditions(f0);
    }
    
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
          std::cout << "Physical mesh has " << form.poissonMesh()->numActiveElements() << " elements.\n";
        }
        
        refStrategy->refine();
        
        // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
        form.poissonMesh()->repartitionAndRebuild();
        
        // project the BC onto the mesh again
        form.vlasovSolution()->projectOntoMesh(bcMap);
        
        bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
        relErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
        
        numVlasovElements = form.vlasovMesh()->numActiveElements();
      }
      if (rank==0)
      {
        std::cout << "After " << numBCRefinements << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
        std::cout << "Physical mesh has " << form.poissonMesh()->numActiveElements() << " elements.\n";
      }
    }
    
    std::ostringstream suffix;
    if (!useL2Trace)
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
    if (useSerendipityBasis)
    {
      suffix << "_serendipity";
    }
    else
    {
      suffix << "_fullTensorBasis";
    }
    suffix << "_p" << H1Order;
    suffix << "_dp" << delta_k;
    suffix << "_" << xElements << "x" << tElements << "x" << vElements;
    if (initialTimeDivision > 0)
    {
      suffix << "_timeRamp" << initialTimeDivision;
    }
    
    const std::string problemPrefix = problemChoiceString;
    
    std::string AMR_suffix = "";
    if (useAdaptivity)
    {
      std::ostringstream AMR_suffix_ss;
      AMR_suffix_ss << "_AMR_";
      if (doerflerTarget > 0)
      {
        AMR_suffix_ss << "Doerfler" << doerflerTarget << "_";
      }
      else
      {
        AMR_suffix_ss << "theta" << refinementThreshold << "_";
      }
      AMR_suffix = AMR_suffix_ss.str();
    }
    
    auto getAMRSuffixForRefNumber = [&](int refNumber = -1) {
      std::ostringstream AMR_suffix_ss;
      AMR_suffix_ss << AMR_suffix;
      if (useAdaptivity)
      {
        AMR_suffix_ss << refNumber << "_refs";
      }
      return AMR_suffix_ss.str();
    };
    
    auto getResultsFilePath = [&](int refNumber = -1) {
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
      resultsFileStream << getAMRSuffixForRefNumber(refNumber);
      resultsFileStream << ".dat";
      
      return resultsFileStream.str();
    };
    
    auto getSpatialMeshFilePath = [&](int refNumber = -1) {
      std::ostringstream spatialMeshInfoPathStringStream;
      spatialMeshInfoPathStringStream << outputDir << "/";
      spatialMeshInfoPathStringStream << problemPrefix << "_mesh1D" << suffix.str();
      
      spatialMeshInfoPathStringStream << getAMRSuffixForRefNumber(refNumber);
      spatialMeshInfoPathStringStream << ".dat";
      
      return spatialMeshInfoPathStringStream.str();
    };
    
    auto getRefinementReportFilePath = [&]() {
      std::ostringstream refReportPathStringStream;
      refReportPathStringStream << outputDir << "/";
      refReportPathStringStream << problemPrefix << "_refReport" << suffix.str();
      
      refReportPathStringStream << getAMRSuffixForRefNumber(numRefinements);
      refReportPathStringStream << ".dat";
      
      return refReportPathStringStream.str();
    };
    
    std::ostringstream vlasovSS;
    vlasovSS << problemPrefix << "Vlasov" << suffix.str();
    
    std::ostringstream poissonSS;
    poissonSS << problemPrefix << "Poisson" << suffix.str();
    
    if (useAdaptivity)
    {
      vlasovSS  << getAMRSuffixForRefNumber(numRefinements);
      poissonSS << getAMRSuffixForRefNumber(numRefinements);
    }
    
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
    
    form.setFixedPointTolerance(fixedPointTol);
    form.setFixedPointMinSubsteps(minSubsteps);
    form.setFixedPointMaxSubsteps(maxSubsteps);
    
  //  form.vlasovSolution()->setWriteMatrixToFile(true, "/tmp/A_Vlasov.dat");
    
    const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
    auto solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form.vlasovSolution(), form.f(), form.poissonMesh()) );
        
    FunctionPtr potential = Function::solution(form.VE(), form.poissonSolution());
    
    auto density  = solutionMoments->solutionMomentFunction(0);
    auto momentum = solutionMoments->solutionMomentFunction(1);
    
    auto velocity = momentum / density;
    
    auto refReportFilePath = getRefinementReportFilePath();
    std::ofstream refReportFileStream(refReportFilePath, std::ios::out);
    
    refReportFileStream << "Vlasov Elements\tVlasov Dofs\tPoisson Elements\tPoisson Dofs\tBC err\tEnergy Error\n";
    refReportFileStream.close();
    
    auto writeRefReport = [&](const int numElementsVlasov, const int numDofsVlasov, const int numElementsPoisson, const int numDofsPoisson, const double bcErr, const double energyError) {
      auto refReportFilePath = getRefinementReportFilePath();
      std::ofstream refReportFileStream(refReportFilePath, std::ios_base::app);
      refReportFileStream << numElementsVlasov << "\t" << numDofsVlasov << "\t" << numElementsPoisson << "\t" << numDofsPoisson << "\t" << bcErr << "\t" << energyError << std::endl;
      refReportFileStream.close();
    };
    
    auto writeResults = [&](const int refinementNumber) {
      
      // output (physical/dimensional) results
      // ultimately, we want E-field, velocity, and some other things
      // to begin with, let's output the E-field values computed at quadrature points at final time
            
      // the di() methods are 1-based:
      static const int SPATIAL_INDEX  = 1;
      static const int TEMPORAL_INDEX = 2;
      static const int VELOCITY_INDEX = 3;
      
      // SpatialFilter's components are zero-based
      static const int SPATIAL_INDEX_ZERO_BASED  = SPATIAL_INDEX  - 1;
      static const int TEMPORAL_INDEX_ZERO_BASED = TEMPORAL_INDEX - 1;
      static const int VELOCITY_INDEX_ZERO_BASED = VELOCITY_INDEX - 1;
      
      // find elements that match final time

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
      
      // update solutionMoments once more and gather data to rank 0, so we can access it in the block below
      solutionMoments->update();
      solutionMoments->gatherOnRankZero();
      
      if (rank == 0)
      {
        std::map<double,double> E_values; // map from x coordinate to time-averaged E field value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> ne_values; // map from x coordinate to time-averaged density value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> velocity_values; // map from x coordinate to time-averaged velocity value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> potential_values; // map from x coordinate to time-averaged potential value; we use a map so it's sorted by x, regardless of the cell numbering (which for adaptive meshes may be out of order)
        std::map<double,double> pointWeight_values; // map from x coordinate to cell measure-weighted quadrature weight
        
        std::vector<GlobalIndexType> physicalCellIDsVector(physicalCellsFinalTime.begin(),physicalCellsFinalTime.end());
        
        for (const auto &physicalCellID : physicalCellIDsVector)
        {
          const int cubatureEnrichmentDegree = 5; // over-integrate to make up for undermeasurement on coarse initial meshes
          auto basisCache = BasisCache::basisCacheForCell(form.poissonMesh(), physicalCellID, false, cubatureEnrichmentDegree); // false: not test versus test
          
          const auto & physicalCellNodes = basisCache->getPhysicalCellNodes();
          
          const int cellOrdinal = 0;
          const double cellWidth = physicalCellNodes(cellOrdinal,1,0) - physicalCellNodes(cellOrdinal,0,0);
          const double temporalWidth = physicalCellNodes(cellOrdinal,2,1) - physicalCellNodes(cellOrdinal,0,1);
          
          auto cubWeights = basisCache->getCubatureWeights();
          auto cubPoints  = basisCache->getRefCellPoints();
          
          // we want to rewrite cubPoints/weights so that the temporal dimension only extends over a thin slice of the element
          // we scale the temporal dimension so that the extent we integrate in time is .001
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
            // map points from (-1,1) to the reference-space preimage of (.999,1.0) in physical space
            // ref space scaling goes from (-1,1) to (1.0 - temporalWidth, 1.0)
            // formula is t_physical = (t_ref + 1) * w/2 + 1 - w
            //                 t_ref = 1 + (t_phys - 1) * 2 / w
            // pre-image of (.999, 1.0) is thus (1 - .002 / w, 1.0)
            // we scale by .001 / w, and shift by 1 - .001 / w
            const double scale = .001 / temporalWidth;
            const double shift = 1 - scale;
            
            auto t_old = temporalPoints(pointOrdinal,0);
            temporalPoints(pointOrdinal,0) = scale * t_old + shift;
            
            // quadrature weights should likewise be scaled
            temporalWeightsView(pointOrdinal) *= scale;
          });
          
          DataHost spatialCubatureWeightsHost(spatialCubWeights);
          DataHost temporalCubatureWeightsHost(temporalWeights);
          
          std::vector<ScalarView> refPointsComponents {spatialPoints, temporalPoints};
          std::vector<Data> cubWeightsComponents {spatialCubWeights, temporalWeights};
          Intrepid2::TensorPoints<double, Camellia::DeviceType> finalTimeRefPoints(refPointsComponents);
          Intrepid2::TensorData<double, Camellia::DeviceType> finalTimeCubWeights(cubWeightsComponents);
          
          basisCache->setRefCellPoints(finalTimeRefPoints, finalTimeCubWeights);
          
          const int numCells = 1;
          FieldContainer<double> E_values_FC(numCells, finalTimeRefPoints.extent_int(0));
          E_soln->values(E_values_FC, basisCache);
          
          FieldContainer<double> potential_values_FC(numCells, finalTimeRefPoints.extent_int(0));
          potential->values(potential_values_FC, basisCache);
          
          FieldContainer<double> density_values_FC(numCells, finalTimeRefPoints.extent_int(0));
          density->values(density_values_FC, basisCache);
          
          FieldContainer<double> velocity_values_FC(numCells, finalTimeRefPoints.extent_int(0));
          velocity->values(velocity_values_FC, basisCache);
          
          const FieldContainer<double> & physicalQuadraturePoints = basisCache->getPhysicalCubaturePointsFC();
          
          const int numSpatialPoints = spatialPoints.extent_int(0);
          
          for (int spatialPointOrdinal=0; spatialPointOrdinal<numSpatialPoints; spatialPointOrdinal++)
          {
            const int firstTensorPointOrdinal = spatialPointOrdinal; // formula is temporalPointOrdinal * numSpatialPoints + spatialPointOrdinal; this is the point ordinal corresponding to temporalPointOrdinal = 0.
            const double & x = physicalQuadraturePoints(cellOrdinal,firstTensorPointOrdinal,0);
            double E_integral         = 0.0;   // integral in time
            double potential_integral = 0.0;
            double density_integral   = 0.0;
            double velocity_integral  = 0.0;
            double one_integral = 0.0; // integrate 1 so that we can divide by that to get the average
            for (int temporalPointOrdinal=0; temporalPointOrdinal < numTemporalPoints; temporalPointOrdinal++)
            {
              const int pointOrdinal = temporalPointOrdinal * numSpatialPoints + spatialPointOrdinal;
              const double & E         = E_values_FC        (cellOrdinal,pointOrdinal);
              const double & potential = potential_values_FC(cellOrdinal,pointOrdinal);
              const double & density   = density_values_FC  (cellOrdinal,pointOrdinal);
              const double & velocity  = velocity_values_FC (cellOrdinal,pointOrdinal);
              
              const double & temporalWeight = temporalCubatureWeightsHost(temporalPointOrdinal);
              E_integral         += E         * temporalWeight;
              potential_integral += potential * temporalWeight;
              density_integral   += density   * temporalWeight;
              velocity_integral  += velocity  * temporalWeight;
              one_integral += temporalWeight;
            }
            const double pointWeight = spatialCubatureWeightsHost(spatialPointOrdinal) * cellWidth / 2.0; // 2.0: width of the reference line.
            
            E_values[x]           = E_integral / one_integral;
            potential_values[x]   = potential_integral / one_integral;
            ne_values[x]          = density_integral / one_integral;
            velocity_values[x]    = velocity_integral / one_integral;
            pointWeight_values[x] = pointWeight;
          }
        }
        
        auto resultsFile = getResultsFilePath(refinementNumber);
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
        
        auto physicalCellNodes = form.poissonMesh()->physicalCellNodes(physicalCellIDsVector);
        auto spatialMeshFile = getSpatialMeshFilePath(refinementNumber);
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
    };
    
    auto reportBCResolutionAndRefinementComplete = [&](const int &refinementNumber, const double energyError) {
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
        
        double bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
        bcRelErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
                  
        if (rank==0)
        {
          std::cout << "BC resolution, relative error in L^2 norm: " << bcRelErr << std::endl;
        }
      }
      auto numVlasovDofs = form.vlasovSolution()->mesh()->numGlobalDofs();
      auto vlasovElementCount  = form.vlasovMesh()->numActiveElements();
      auto poissonElementCount = form.poissonMesh()->numActiveElements();
      auto poissonDofCount     = form.poissonMesh()->numGlobalDofs();
            
      if (rank == 0)
      {
        writeRefReport(vlasovElementCount, numVlasovDofs, poissonElementCount, poissonDofCount, bcRelErr, energyError);
        std::cout << "******************************  Refinement " << refinementNumber << " complete. ******************************\n";
      }
    };
    
    if (!useAdaptivity)
    {
      auto numVlasovDofs = form.vlasovSolution()->mesh()->numGlobalDofs();
      if (numVlasovDofs > maxDofsForDirectSolve)
      {
        if (rank == 0)
        {
          std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
        }
        form.setUseGeometricMultigrid(true);
      }
      else
      {
        if (rank == 0)
        {
          std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
        }
        form.setUseGeometricMultigrid(false);
      }
      
      form.solve();
      
      solutionMoments->update();
      
      if (saveViz)
      {
        poissonViz->exportSolution(form.poissonSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"});
        vlasovViz->exportSolution(form.vlasovSolution(), {cellIDFunction}, {"cell ID (Vlasov)"});
      }
      writeResults(0);
    }
    else
    {
      RefinementStrategyPtr refStrategy;
      
      if (doerflerTarget <= 0)
      {
        refStrategy = Teuchos::rcp(new RefinementStrategy(form.vlasovSolution(), refinementThreshold));
        refStrategy->setMaxElements(maxElements); // allow unrefinements
      }
      else
      {
        ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(form.vlasovSolution());
        
        refStrategy = Teuchos::rcp(new DoerflerRefinementStrategy(errorIndicator, doerflerTarget) );
      }
      
      double energyError;
      auto energyErrorFunction = EnergyErrorFunction::energyErrorFunction(form.vlasovSolution());
      for (int refinementNumber=0; refinementNumber<numRefinements; refinementNumber++)
      {
        auto numVlasovDofs = form.vlasovSolution()->mesh()->numGlobalDofs();
        if (numVlasovDofs > maxDofsForDirectSolve)
        {
          if (rank == 0)
          {
            std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
          }
          form.setUseGeometricMultigrid(true);
        }
        else
        {
          if (rank == 0)
          {
            std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
          }
          form.setUseGeometricMultigrid(false);
        }
        
//        {
//          if (rank == 0)
//          {
//            std::cout << "zeroing electric field between adaptive solves; this is primarily for debugging.\n";
//          }
//          std::map<int, FunctionPtr> functionMap;
//          functionMap[form.E()->ID()] = Function::zero();
//          form.poissonSolution()->projectOntoMesh(functionMap);
//
//          {
//            auto E_extruded = form.E_extruded();
//
//            // make sure E_extruded is ready to evaluate…
//            static_cast<PhysicalSolutionExtrusion*>(E_extruded.get())->updateValues();
//          }
//        }
        
        form.solve();
        
        solutionMoments->update();
        
        writeResults(refinementNumber);
        
        energyError = refStrategy->computeTotalEnergyError(); // forces update to energyErrorFunction
        reportBCResolutionAndRefinementComplete(refinementNumber, energyError);
        
        if (saveViz)
        {
          poissonViz->exportSolution(form.poissonSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"}, refinementNumber);
          vlasovViz->exportSolution(form.vlasovSolution(), {energyErrorFunction,cellIDFunction}, {"energy error","cell ID (Vlasov)"}, refinementNumber);
        }
        
        refStrategy->refine(rank==0);
        
        // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
        form.poissonMesh()->repartitionAndRebuild();
        auto poissonElementCount = form.poissonMesh()->numActiveElements();
        auto poissonDofCount     = form.poissonMesh()->numGlobalDofs();
        if (rank == 0)
        {
          std::cout << "Poisson mesh has " << poissonElementCount << " active elements and " << poissonDofCount << " global degrees of freedom.\n";
        }
      }
      auto numVlasovDofs = form.vlasovSolution()->mesh()->numGlobalDofs();
      if (numVlasovDofs > maxDofsForDirectSolve)
      {
        if (rank == 0)
        {
          std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
        }
        form.setUseGeometricMultigrid(true);
      }
      else
      {
        if (rank == 0)
        {
          std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
        }
        form.setUseGeometricMultigrid(false);
      }
      
//      {
//        if (rank == 0)
//        {
//          std::cout << "zeroing electric field between adaptive solves; this is primarily for debugging.\n";
//        }
//        std::map<int, FunctionPtr> functionMap;
//        functionMap[form.E()->ID()] = Function::zero();
//        form.poissonSolution()->projectOntoMesh(functionMap);
//
//        {
//          auto E_extruded = form.E_extruded();
//
//          // make sure E_extruded is ready to evaluate…
//          static_cast<PhysicalSolutionExtrusion*>(E_extruded.get())->updateValues();
//        }
//      }
      
      form.solve();
      energyError = refStrategy->computeTotalEnergyError(); // forces update to energyErrorFunction

      solutionMoments->update();
      
      if (saveViz)
      {
        poissonViz->exportSolution(form.poissonSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"}, numRefinements);
        vlasovViz->exportSolution(form.vlasovSolution(), {energyErrorFunction,cellIDFunction}, {"energy error","cell ID (Vlasov)"}, numRefinements);
      }
      writeResults(numRefinements);
      
      reportBCResolutionAndRefinementComplete(numRefinements, energyError);
    }
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
