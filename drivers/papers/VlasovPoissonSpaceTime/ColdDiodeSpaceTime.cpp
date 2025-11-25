//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "CamelliaDataHelpers.h"
#include "ConfigurationSpaceFormulation.h"
#include "ConfigurationSpaceFormulationPoisson.h"
#include "DoerflerRefinementStrategy.h"
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
#include "VlasovSpaceTimeFormulation.h"

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

Teuchos::RCP<VlasovSpaceTimeFormulation> getFormulation1D1T1V(const double x0, const double xWidth,
                                                              const double t0, const double tWidth,
                                                              const double v0, const double vWidth,
                                                              const int xElements, const int tElements, const int vElements,
                                                              double q, double eps0, double m,
                                                              const int H1Order, const int delta_k,
                                                              const bool usePeriodicBCs, const bool useSerendipity,
                                                              const int numUniformRefinements)
{
  Teuchos::RCP<ConfigurationSpaceFormulationPoisson> configForm = Teuchos::rcp(new ConfigurationSpaceFormulationPoisson(q, eps0, SPACE_TIME) );
  
  MeshTopologyPtr spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElements, usePeriodicBCs);
  
  std::vector<double> temporalVertices(tElements + 1);
  for (int i=0; i<=tElements; i++)
  {
    const double t = double(i) / double(tElements) * tWidth + t0;
    temporalVertices[i] = t;
  }
  
  std::vector<double> velocitySpaceVertices_x(vElements+1);
  for (int i=0; i<=vElements; i++)
  {
    const double v = double(i) / double(vElements) * vWidth + v0;
    velocitySpaceVertices_x[i] = v;
  }
  
  std::vector<double> velocitySpaceVertices_y; // empty
  std::vector<double> velocitySpaceVertices_z; // empty
  
  Teuchos::RCP<VlasovSpaceTimeFormulation> form;
  
  EBasisFamily basisFamily = useSerendipity ? BASIS_FAMILY_SERENDIPITY : BASIS_FAMILY_HIERARCHICAL;
  
  form = Teuchos::rcp(new VlasovSpaceTimeFormulation(q, m, configForm, spatialMeshTopo,
                                                     temporalVertices,
                                                     velocitySpaceVertices_x,
                                                     velocitySpaceVertices_y,
                                                     velocitySpaceVertices_z,
                                                     H1Order, delta_k,
                                                     numUniformRefinements,
                                                     basisFamily));
  return form;
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

std::map<GlobalIndexType, std::vector<double> > readOverkillSolutionCoefficients(Teuchos::RCP<VlasovSpaceTimeFormulation> fineForm, const std::string &overkillSolutionDir)
{
  std::map<GlobalIndexType, std::vector<double> > mySolnCoefficients;

  using std::cout;
  using std::cerr;
  using std::endl;
  using std::flush;
  using std::string;
  using std::to_string;
  
  auto comm = fineForm->vlasovMesh()->Comm();
  
  const int rank = comm->getRank();
  
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

  auto f = fineForm->f();

  // file that stores f coefficients in our proprietary format
  string solnFile_f = overkillSolutionDir + "/OverkillSolution.soln_f";
  
  // collect all solution data onto rank 0
  auto allActiveCellIDs = fineForm->vlasovMesh()->getActiveCellIDsGlobal();
    
  // send from rank 0 to all ranks -- memory inefficient, but a simple implementation
  int numRecipients = (rank == 0) ? comm->getSize() : 0;
  
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
      
      auto dofIndices = fineForm->vlasovMesh()->getElementType(cellID)->trialOrderPtr->getDofIndices(f->ID());
      
      int dofCount;
      solnFileStream.read((char*) &dofCount, sizeof(dofCount));
      
      TEUCHOS_TEST_FOR_EXCEPTION(dofCount != dofIndices.size(), std::invalid_argument, "Invalid dof count");
      
      std::vector<double> coefficients(dofCount);
      
      solnFileStream.read((char*) &coefficients[0], dofCount * sizeof(double));
      
      int partitionForCell = fineForm->vlasovMesh()->partitionForCellID(cellID);
      packedSolutions[partitionForCell].insert(packedSolutions[partitionForCell].end(), coefficients.begin(), coefficients.end());
    }
    
    solnFileStream.close();
  }
  
  // send from rank 0 to all ranks
  std::vector<double> receivedPackedSolution;
  MPIWrapper::sendDataVectors(comm, recipients, packedSolutions, receivedPackedSolution);
  
  int offset = 0;
  const auto &myCellIDs = fineForm->vlasovMesh()->cellIDsInPartition();
  for (const auto &cellID : myCellIDs)
  {
    int dofCount = fineForm->vlasovMesh()->getElementType(cellID)->trialOrderPtr->getBasisCardinality(f->ID(), VOLUME_INTERIOR_SIDE_INDEX);
    
    TEUCHOS_TEST_FOR_EXCEPTION(dofCount + offset > receivedPackedSolution.size(), std::invalid_argument, "out of bounds");
    
    std::vector<double> coefficients(receivedPackedSolution.begin() + offset, receivedPackedSolution.begin() + offset + dofCount);
    
    offset += dofCount;
    
    mySolnCoefficients[cellID] = coefficients;
  }
  
  return mySolnCoefficients;
}

void convertOverkillSolution(Teuchos::RCP<VlasovSpaceTimeFormulation> fineForm, const std::string &overkillSolutionDir)
{
  using std::cout;
  using std::cerr;
  using std::endl;
  using std::flush;
  using std::string;
  using std::to_string;
  
  const int rank = fineForm->vlasovMesh()->Comm()->getRank();
  
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
  fineForm->vlasovSolution()->loadFromHDF5(solnFile);
  
  FunctionPtr f_fine     = Function::solution(fineForm->f(),     fineForm->vlasovSolution());
  
  auto f = fineForm->f();
  
  // output just the f basis coefficients
  string solnFile_f = overkillSolutionDir + "/OverkillSolution_" + to_string(timeStepNumber) + ".soln_f";
  
  // collect all solution data onto rank 0
  auto allActiveCellIDs = fineForm->vlasovMesh()->getActiveCellIDsGlobal();
  std::set<GlobalIndexType> noCellIDs;
  auto & cellIDsToImport = (rank == 0) ? allActiveCellIDs : noCellIDs;
  
  auto writeSolution = [&] () -> void
  {
    fineForm->vlasovSolution()->importSolutionForOffRankCells(cellIDsToImport);
    
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
        auto dofIndices = fineForm->vlasovMesh()->getElementType(cellID)->trialOrderPtr->getDofIndices(f->ID());
        
        solnFileStream.write((char*) &cellID, sizeof(cellID));
        
        int dofCount = dofIndices.size();
        solnFileStream.write((char*) &dofCount, sizeof(dofCount));
        
        bool warnAboutOffRank = false; // don't warn
        const auto & solnCoefficients = fineForm->vlasovSolution()->allCoefficientsForCellID(cellID, warnAboutOffRank);
        
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
  double globalErrorSquaredSum = MPIWrapper::sum(mesh->Comm(), myErrorSquaredSum);
  return std::sqrt(globalErrorSquaredSum);
}

double l2NormOfUniformSolution(VarPtr fieldVar, SolutionPtr interpolatedSolutionOnOverkillMesh)
{
  std::map<GlobalIndexType, std::vector<double> > emptyOverkillSolution;
  
  return overkillSolutionErrorL2Norm(fieldVar, interpolatedSolutionOnOverkillMesh, emptyOverkillSolution);
}

void runAdaptiveComparisonExperiment(Teuchos::RCP<VlasovSpaceTimeFormulation> adaptiveForm,
                                     Teuchos::RCP<VlasovSpaceTimeFormulation> fineForm,
                                     Teuchos::RCP<VlasovSpaceTimeFormulation> modestForm,
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
  
  auto comm = adaptiveForm->vlasovMesh()->Comm();
  const int rank = comm->getRank();
  
  if (writeOverkillSolutionAndQuit)
  {
    int numFineElements     = fineForm->vlasovMesh()->numActiveElements();
    int numFineDofs         = fineForm->vlasovMesh()->numGlobalDofs();
    
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
    fineForm->setInitialConditions(f0);
    
    auto mesh = fineForm->vlasovMesh();
    auto soln = fineForm->vlasovSolution();
          
    const int numSubsteps = 3;
    fineForm->setFixedPointMinSubsteps(numSubsteps);
    fineForm->setFixedPointMaxSubsteps(numSubsteps);
    
    auto f = fineForm->f();
    int timeStepNumber = 0;
    int solutionNumber = 0;
    
    // output just the f basis coefficients
    string solnFile_f = overkillSolutionDir + "/OverkillSolution_" + to_string(solutionNumber) + ".soln_f";
    
    // collect all solution data onto rank 0
    auto allActiveCellIDs = fineForm->vlasovMesh()->getActiveCellIDsGlobal();
    std::set<GlobalIndexType> noCellIDs;
    auto & cellIDsToImport = (rank == 0) ? allActiveCellIDs : noCellIDs;
    
    auto writeSolution = [&] () -> void
    {
      fineForm->vlasovSolution()->importSolutionForOffRankCells(cellIDsToImport);
      
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
          auto dofIndices = fineForm->vlasovMesh()->getElementType(cellID)->trialOrderPtr->getDofIndices(f->ID());
          
          solnFileStream.write((char*) &cellID, sizeof(cellID));
          
          int dofCount = dofIndices.size();
          solnFileStream.write((char*) &dofCount, sizeof(dofCount));
          
          bool warnAboutOffRank = false; // don't warn
          const auto & solnCoefficients = fineForm->vlasovSolution()->allCoefficientsForCellID(cellID, warnAboutOffRank);
          
          for (const auto &dofIndex : dofIndices)
          {
            solnFileStream.write((char*) &solnCoefficients(dofIndex), sizeof(double));
          }
        }
        
        solnFileStream.close();
      }
    };
    
    solnFile_f = overkillSolutionDir + "/OverkillSolution.soln_f";
    fineForm->solve();
    
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
  
  MeshMap fineToModestMap(modestForm->vlasovMesh(), fineForm->vlasovMesh());
  
  // set initial conditions on adaptive form:
  adaptiveForm->setInitialConditions(f0);

  ErrorIndicatorPtr unrefinementErrorIndicator;
  if (!useEnergyForUnrefinements)
  {
    unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(adaptiveForm->vlasovSolution(), adaptiveForm->f());
  }
//  adaptiveForm->setRefinementThresholds(refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator, gateRefinements);
  
  auto vlasovMesh  = adaptiveForm->vlasovMesh();
  auto poissonMesh = adaptiveForm->configurationSpaceMesh();
  
  bool outputOverkillSolution = false;
  
  Teuchos::RCP<HDF5Exporter> vlasovFineViz, vlasovModestViz, vlasovAdaptiveViz;
  if (saveViz)
  {
    if (outputOverkillSolution)
    {
      vlasovFineViz     = Teuchos::rcp( new HDF5Exporter (fineForm->vlasovMesh(),   "vlasovFine",     outputDir));
    }
    vlasovModestViz   = Teuchos::rcp( new HDF5Exporter (modestForm->vlasovMesh(),   "vlasovModest",   outputDir));
    vlasovAdaptiveViz = Teuchos::rcp( new HDF5Exporter (adaptiveForm->vlasovMesh(), "vlasovAdaptive", outputDir));
  }
  
  BasisReconciliation br;
    
  // adapt to capture initial data:
  {
    FunctionPtr f_soln = Function::solution(adaptiveForm->f(), adaptiveForm->vlasovSolution());
    
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
      adaptiveForm->setInitialConditions(f0); // do projection again
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
          
          adaptiveForm->setInitialConditions(f0); // do projection again
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
  modestForm->setInitialConditions(f0);
  
  auto f = fineForm->f();
  
  std::map<GlobalIndexType, std::vector<double> > localOverkillCoefficients;
  
  localOverkillCoefficients = readOverkillSolutionCoefficients(fineForm, overkillSolutionDir);
  
  FunctionPtr f_modest   = Function::solution(modestForm->f(),   modestForm->vlasovSolution());
  FunctionPtr f_adaptive = Function::solution(adaptiveForm->f(), adaptiveForm->vlasovSolution());

  SolutionPtr projectedModestSoln   = Solution::solution(fineForm->vlasovMesh()->bilinearForm(), fineForm->vlasovMesh());
  SolutionPtr projectedAdaptiveSoln = Solution::solution(fineForm->vlasovMesh()->bilinearForm(), fineForm->vlasovMesh());
  
  MeshMap fineToAdaptiveMap(adaptiveForm->vlasovMesh(), fineForm->vlasovMesh());
  
  interpolateToFineSolution(f, modestForm->vlasovSolution(),   projectedModestSoln,   fineToModestMap,   br);
  interpolateToFineSolution(f, adaptiveForm->vlasovSolution(), projectedAdaptiveSoln, fineToAdaptiveMap, br);
  
  double adaptiveErr = overkillSolutionErrorL2Norm(f, projectedAdaptiveSoln, localOverkillCoefficients);
  double modestErr   = overkillSolutionErrorL2Norm(f, projectedModestSoln, localOverkillCoefficients);
  
  double adaptiveSolnNorm = l2NormOfUniformSolution(f, projectedAdaptiveSoln);
  double modestSolnNorm   = l2NormOfUniformSolution(f, projectedModestSoln);
  double fineSolnNorm     = overkillSolutionErrorL2Norm(f, fineForm->vlasovSolution(), localOverkillCoefficients, true);
  
  double t = 0.0;
  if (saveViz)
  {
    vlasovAdaptiveViz->exportSolution(adaptiveForm->vlasovSolution(), t);
    vlasovModestViz->exportSolution(  modestForm->vlasovSolution(),   t);
    
    if (outputOverkillSolution)
    {
      vlasovFineViz->exportSolution(    fineForm->vlasovSolution(),     t);
    }
  }
  
  int numAdaptiveElements = adaptiveForm->vlasovMesh()->numActiveElements();
  int numAdaptiveDofs     = adaptiveForm->vlasovMesh()->numGlobalDofs();
  
  int numModestElements   = modestForm->vlasovMesh()->numActiveElements();
  int numModestDofs       = modestForm->vlasovMesh()->numGlobalDofs();
  
  int numFineElements     = fineForm->vlasovMesh()->numActiveElements();
  int numFineDofs         = fineForm->vlasovMesh()->numGlobalDofs();
  
  if (rank == 0)
  {
    using std::cout;
    using std::endl;
    
    cout << "Fine mesh has     " << numFineElements     << " elements and " << numFineDofs     << " degrees of freedom.\n";
    cout << "Modest mesh has   " << numModestElements   << " elements and " << numModestDofs   << " degrees of freedom.\n";
    cout << "Adaptive mesh has " << numAdaptiveElements << " elements and " << numAdaptiveDofs << " degrees of freedom.\n";
  }
  
  const int numSubsteps = 3;
  adaptiveForm->setFixedPointMinSubsteps(numSubsteps);
  modestForm->setFixedPointMinSubsteps(numSubsteps);
  fineForm->setFixedPointMinSubsteps(numSubsteps);
  adaptiveForm->setFixedPointMaxSubsteps(numSubsteps);
  modestForm->setFixedPointMaxSubsteps(numSubsteps);
  fineForm->setFixedPointMaxSubsteps(numSubsteps);
  
  double adaptiveTime = 0;
  {
    auto adaptiveTimer = Teuchos::TimeMonitor::getNewTimer("adaptive solve");
    adaptiveTimer->start();
    adaptiveForm->solve();
    adaptiveTimer->stop();
    adaptiveTime += MPIWrapper::max(comm, adaptiveTimer->totalElapsedTime());
  }
  
  double modestTime = 0;
  {
    auto modestTimer = Teuchos::TimeMonitor::getNewTimer("modest solve");
    modestTimer->start();
    modestForm->solve();
    modestTimer->stop();
    modestTime += MPIWrapper::max(comm, modestTimer->totalElapsedTime());
  }
  
  // adaptive mesh will have changed; build a fresh map.  (We could do this in a more granular fashion, by making MeshMap a RefinementObserver and updating on refinement, unrefinement.)
  fineToAdaptiveMap = MeshMap(adaptiveForm->vlasovMesh(), fineForm->vlasovMesh());
  
  interpolateToFineSolution(f, modestForm->vlasovSolution(),   projectedModestSoln,   fineToModestMap,   br);
  interpolateToFineSolution(f, adaptiveForm->vlasovSolution(), projectedAdaptiveSoln, fineToAdaptiveMap, br);
  
  if (saveViz)
  {
    vlasovAdaptiveViz->exportSolution(adaptiveForm->vlasovSolution());
    vlasovModestViz->exportSolution(  modestForm->vlasovSolution());
    
    if (outputOverkillSolution)
    {
      vlasovFineViz->exportSolution(fineForm->vlasovSolution());
    }
  }
  
  adaptiveErr  = overkillSolutionErrorL2Norm(f, projectedAdaptiveSoln, localOverkillCoefficients);
  modestErr    = overkillSolutionErrorL2Norm(f, projectedModestSoln, localOverkillCoefficients);
  fineSolnNorm = overkillSolutionErrorL2Norm(f, fineForm->vlasovSolution(), localOverkillCoefficients, true);
  
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
    summaryStream << "Field polynomial order of non-overkill meshes (one less than H^1 order): " << adaptiveForm->vlasovMesh()->globalDofAssignment()->getInitialH1Order()[0]-1 << endl;
    summaryStream << "Relative L^2 error for adaptive solution at final time: " << adaptiveErr / fineSolnNorm << endl;
    summaryStream << "Relative L^2 error for uniform solution at final time:  " << modestErr / fineSolnNorm << endl;
    summaryStream << "Total solve time for adaptive solution (max across MPI ranks): " << adaptiveTime << endl;
    summaryStream << "Total solve time for uniform solution (max across MPI ranks):  " << modestTime << endl;
    
    summaryStream.close();
    std::cout << "Wrote run summary to " << summaryFilePath << "\n";
  }
}

LinearTermPtr vlasovRHSForExactSoln(Teuchos::RCP<VlasovSpaceTimeFormulation> form, FunctionPtr f_exact, FunctionPtr E_exact, const double q,
                                    const double m)
{
  // want an RHS so that f_exact is the exact solution
  auto w = form->w(); // test function for Vlasov RHS
  const int spaceDim              = form->configurationSpaceMesh()->getDimension() - 1; // space, not space-time, dim
  const int velocityDim           = form->vlasovMesh()->getDimension() - spaceDim - 1;
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

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  const int rank = mpiSession.getRank();
  
  {
    mpiSession.barrier();

    const double e                = 1.602176634e-19;                                          // electron's electric charge in Coulombs
    const double m_e              =  9.10938356e-31;                                          // m_e, mass of electron in kg
    const double W                = 1e4;                                                      // beam intensity from SAND report [eV]
    const double J                = 9.33582e4;                                                // current density from SAND report [A/m^2]
    const double permittivity     = 8.85418782e-12;                                           // permittivity of free space
    const double v_beam           = sqrt(2 * W * e / m_e );                                   // v_beam is defined by 1/2 m v^2 = W ==> v_beam = sqrt( 2 W /m ), e converts from eV to Joules
    const double velocityScaling  = v_beam;
    const double timeScaling      = 1.97485663509e-9;                                         // final time from SAND report [s]
    const double spatialScaling   = velocityScaling * timeScaling;                            // [m]
    const double EScaling         = m_e * velocityScaling / (e * timeScaling);
    const double nScaling         = J / (e * v_beam);                                         // number density of plasma
    const double potentialScaling = EScaling * spatialScaling;
    const double omega_p          = sqrt( e * e * nScaling / (m_e * permittivity));
    const double q                = -1.0;
    const double eps0             =  1.0 / (omega_p * omega_p * timeScaling * timeScaling);
    const double m                =  1.0;

    bool usePeriodicBCs         = false;
    bool enforceQuasiNeutrality = false;
    bool useL2Trace             = true;
    bool useSerendipityBasis    = true;
    
    cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
    cmdp.setOption("useSerendipityBasis", "useFullTensorBasis", &useSerendipityBasis);
    
    int delta_k   = 2;
    int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1

    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);

    int xElements = 4;
    int vElements = 40;
    int tElements = 5;
    int numUniformRefinements = 0;

    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("tElements", &tElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("numUniformRefinements", &numUniformRefinements, "number of uniform refinements applied to the base mesh");

    double x0        = 0.0;
    double xWidth    = 0.01 / spatialScaling;
    double t0        = 0.0;
    double tWidth    = 1.0;
    double v0        = 0.50;
    double vWidth    = 1.00;
    
    cmdp.setOption("v0", &v0, "minimum velocity coordinate in domain");
    cmdp.setOption("vWidth", &vWidth, "span of velocity coordinates in domain");
    
    bool useTimeSlabs = false;
    int numTimeSlabs = 1;

    cmdp.setOption("useTimeSlabs", "solveForAllTime", &useTimeSlabs);
    cmdp.setOption("numTimeSlabs", &numTimeSlabs);

    double u0     = 1.0;   // beam velocity
    double sigma  = 0.025; // themral velocity
    double n0     = 1.0;   // computational density

    cmdp.setOption("u0", &u0);
    cmdp.setOption("sigma", &sigma); // sigma for cold diode -- ignored for other problems
    
    bool useAdaptivity = false;
    int numRefinements = 3;
    double refinementThreshold = 0.80; // refine elements with at least 80% of the maximum per-element error
    int maxElements = 64 * 64; // do unrefinements to keep total mesh size smaller than this
    bool gateRefinements = true;
    double doerflerTarget = 0; // if set to positive value by user, use Doerfler instead of greedy strategy
    bool useEnergyForUnrefinements = false;

    cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    cmdp.setOption("numRefinements", &numRefinements);
    cmdp.setOption("refinementThreshold", &refinementThreshold);
    cmdp.setOption("maxElements", &maxElements);
    cmdp.setOption("gateRefinements", "dontGateRefinements", &gateRefinements);
    cmdp.setOption("doerflerTarget", &doerflerTarget);
    cmdp.setOption("useEnergyForUnrefinements", "useL2ProjectionForUnrefinements", &useEnergyForUnrefinements);

    std::string outputDir = "/tmp";
    bool saveViz = true;
    std::string resultsFilePrefix = "";

    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("viz", "noViz", &saveViz);
    cmdp.setOption("resultsFilePrefix", &resultsFilePrefix);

    bool convertOverkillSolutionFiles = false;          // mutually exclusive with runOverkillVersusAdaptiveComparison
    bool runOverkillVersusAdaptiveComparison = false;   // mutually exclusive with convertOverkillSolutionFiles
    bool saveOverkillSolutionAndQuit = false;           // only used if runOverkillVersusAdaptiveComparison is true
    int modestRefinementLevel = 3;                      // only used if runOverkillVersusAdaptiveComparison is true; would take a coarse 8x8 mesh to 64x64 (this is used for overkill experiment)
    int overkillRefinementLevel = 5;                    // only used if convertOverkillSolutionFiles or runOverkillVersusAdaptiveComparison is true; would take a coarse 8x8 mesh to 256x256
    int overkillTimeStepsBetweenComparisons = 1;        // only used if runOverkillVersusAdaptiveComparison is true
    int solutionOutputFrequency = 1;                    // only used if runOverkillVersusAdaptiveComparison is true
    int solutionComparisonFrequency = 1;                // only used if runOverkillVersusAdaptiveComparison is true
    std::string overkillSolutionDir = "";

    cmdp.setOption("convertOverkillSolution", "dontConvertOverkillSolution", &convertOverkillSolutionFiles);
    cmdp.setOption("runOverkillVersusAdaptiveComparison", "runStandard", &runOverkillVersusAdaptiveComparison);
    cmdp.setOption("saveOverkillSolutionAndQuit", "dontSolveOverkill", &saveOverkillSolutionAndQuit);
    cmdp.setOption("modestMeshRefinementLevel", &modestRefinementLevel, "modest refinement over numUniformRefinements");
    cmdp.setOption("overkillMeshRefinementLevel", &overkillRefinementLevel, "overkill refinement over numUniformRefinements");
    cmdp.setOption("overkillTimeStepsBetweenComparisons", &overkillTimeStepsBetweenComparisons);
    cmdp.setOption("solutionOutputFrequency", &solutionOutputFrequency);
    cmdp.setOption("solutionComparisonFrequency", &solutionComparisonFrequency);
    cmdp.setOption("overkillSolutionDir", &overkillSolutionDir);

    //    bool useGeometricMultigrid = true;
    
    int maxDofsForDirectSolve = 20000;
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    double iterativeSolveMaxIters  = 10000;

    cmdp.setOption("maxDofsForDirectSolve", &maxDofsForDirectSolve);
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);

    int minSubsteps = 5;  // for fixed-point iteration.
    int maxSubsteps = 50; // for fixed-point iteration.
    double fixedPointTol = 1e-8;

    cmdp.setOption("minSubsteps", &minSubsteps);
    cmdp.setOption("maxSubsteps", &maxSubsteps);
    cmdp.setOption("fixedPointTol", &fixedPointTol);

    double bcResolutionTolerance = 1e-3;  // only applies if doing an adaptive solve
    cmdp.setOption("bcResolutionTolerance", &bcResolutionTolerance);
    
    double initialTimeDivision = 0;
    cmdp.setOption("initialTimeDivision", &initialTimeDivision);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      return -1;
    }
    
    if (useTimeSlabs)
    {
      tWidth /= numTimeSlabs;
    }
    
    if (rank == 0)
    {
      std::cout << "Running cold diode problem.\n";
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
    
    auto form = getFormulation1D1T1V(x0, xWidth, t0, tWidth, v0, vWidth, xElements, tElements, vElements, q, eps0, m, H1Order, delta_k, usePeriodicBCs,
                                     useSerendipityBasis, numUniformRefinements);
    form->setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    
    std::vector<std::pair<std::pair<FunctionPtr,FunctionPtr>,int> > bcFunctionsToResolve; // < <exact function, solution function>, tag> -- only used when useAdaptivity is true
    
    // we'll use this function to allow labeling of visualization output
    FunctionPtr cellIDFunction = Teuchos::rcp(new CellIDFunction);
    
    //    form->setUseGeometricMultigrid(useGeometricMultigrid);
    form->setIterativeSolveOutputLevel(iterativeSolveOutputLevel);
    //    form->setIterativeSolveTolerance(iterativeSolveTolerance);
    std::vector<double> cgTols(maxSubsteps,iterativeSolveTolerance);
    if (maxSubsteps > 3)
    {
      // use looser tolerances for the first couple steps
      cgTols[0] = iterativeSolveTolerance * 100.0;
      cgTols[1] = iterativeSolveTolerance * 10.0;
    }
    form->setIterativeSolveTolerances(cgTols);
    form->setIterativeSolveMaxIterations(iterativeSolveMaxIters);
    
    if (numUniformRefinements > 0)
    {
      for (int i=0; i<numUniformRefinements; i++)
      {
        const auto & activeCells  = form->vlasovMesh()->getActiveCellIDsGlobal();
        form->vlasovMesh()->hRefine(activeCells);
        
        xElements *= 2;
        tElements *= 2;
        vElements *= 2;
        
        // it should be the case that Poisson mesh is automatically refined when the Vlasov mesh is
      }
    }
    
    FunctionPtr f0;
    
    bool adaptToCaptureInitialData = true;
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
      auto fineForm = getFormulation1D1T1V(x0, xWidth, t0, tWidth, v0, vWidth, xElements, tElements, vElements, q, eps0, m, H1Order, delta_k, usePeriodicBCs,
                                           useSerendipityBasis, numUniformRefinements + overkillRefinementLevel);
      
      //VlasovPoissonSpaceTimeFormulation fineForm(formulationChoice, q, eps0, m, x0, xWidth, xElements, t0, tWidth, tElements, v0, vWidth, vElements, overkillH1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, useSerendipityBasis, vlasovTraceEnrichment);
      //// TODO: do uniform refinements on fineForm meshes: numUniformRefinements+overkillRefinementLevel
      
      convertOverkillSolution(fineForm, overkillSolutionDir);
      return 0; // exit
    }
    else if (runOverkillVersusAdaptiveComparison)
    {
      auto fineForm     = getFormulation1D1T1V(x0, xWidth, t0, tWidth, v0, vWidth, xElements, tElements, vElements, q, eps0, m, H1Order, delta_k, usePeriodicBCs,
                                               useSerendipityBasis, numUniformRefinements + overkillRefinementLevel);
      auto adaptiveForm = getFormulation1D1T1V(x0, xWidth, t0, tWidth, v0, vWidth, xElements, tElements, vElements, q, eps0, m, H1Order, delta_k, usePeriodicBCs,
                                               useSerendipityBasis, numUniformRefinements);
      auto modestForm   = getFormulation1D1T1V(x0, xWidth, t0, tWidth, v0, vWidth, xElements, tElements, vElements, q, eps0, m, H1Order, delta_k, usePeriodicBCs,
                                               useSerendipityBasis, numUniformRefinements + modestRefinementLevel);
      const int maxAdaptiveMeshElements = modestForm->vlasovMesh()->numActiveElements();
      
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
    {
      {
        // pin potential to zero at left and right:
        SpatialFilterPtr fullBoundaryFilter  = SpatialFilter::allSpace();
        bool restrictToBoundary = true; // cold diode: not periodic
        auto xBoundaryTag = form->createConfigurationMeshSideSetTag(fullBoundaryFilter, restrictToBoundary);
        auto VE_traceVar = form->VE_configurationSpaceTrace();
        form->addConfigurationSpaceBC(xBoundaryTag, VE_traceVar, Function::zero());
        
        // set a Maxwellian at the inflow:
        FunctionPtr fM_u0 = Teuchos::rcp(new Maxwellian1T1V( u0, sigma, n0) );
        FunctionPtr f_leftInflow = fM_u0;
        form->setInitialAndInflowConditions(Function::zero(), f_leftInflow, initialTimeDivision); // no right BC is only for negative velocity, so won't actually get triggered on our mesh
        
        if (useAdaptivity)
        {
          auto inflowTag = form->tagVlasovSpatialInflow();
          
          FunctionPtr sgn = Function::sideParity();
          FunctionPtr n = Function::normal();
          FunctionPtr n_x = n->spatialComponent(1); // 1-based
          FunctionPtr vx = form->v_i(0);
          
          FunctionPtr tn_leftInflowExact = f_leftInflow * sgn * n_x * vx;
          const bool weightFluxesBySideParity = true;
          FunctionPtr tn_soln = Function::solution(form->t_n_hat(), form->vlasovSolution(), weightFluxesBySideParity);
          bcFunctionsToResolve.push_back({{tn_leftInflowExact, tn_soln},inflowTag});
        }
      }
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
      bcMap[form->t_n_hat()->ID()] = bcFxnExact;
      form->vlasovSolution()->projectOntoMesh(bcMap);
      
      FunctionPtr bc_err = bcFxnExact * bcRestriction - bcFxnSoln * bcRestriction;
      int quadratureEnrichment = 4;
      
      auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(form->vlasovMesh(), bc_err, quadratureEnrichment);
      
      double refinementThresholdForInitialData = 0.80;
      auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
      //      refStrategy->setMaxElements(maxElements);
      refStrategy->setMaxP(H1Order);
      
      double bc_norm = (bcFxnExact * bcRestriction)->l2norm(form->vlasovMesh(), quadratureEnrichment);
      double relErr = bc_err->l2norm(form->vlasovMesh(), quadratureEnrichment) / bc_norm;
      
      int numBCRefinements = 0;
      int numVlasovElements = form->vlasovMesh()->numActiveElements();
      while ((relErr > bcResolutionTolerance) && (numVlasovElements < maxElements))
      {
        if (rank==0)
        {
          std::cout << "After " << numBCRefinements++ << " refinements, phase space mesh has " << form->vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
          std::cout << "Physical mesh has " << form->configurationSpaceMesh()->numActiveElements() << " elements.\n";
        }
        
        refStrategy->refine();
        
        // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
        form->configurationSpaceMesh()->repartitionAndRebuild();
        
        // project the BC onto the mesh again
        form->vlasovSolution()->projectOntoMesh(bcMap);
        
        bc_norm = (bcFxnExact * bcRestriction)->l2norm(form->vlasovMesh(), quadratureEnrichment);
        relErr = bc_err->l2norm(form->vlasovMesh(), quadratureEnrichment) / bc_norm;
        
        numVlasovElements = form->vlasovMesh()->numActiveElements();
      }
      if (rank==0)
      {
        std::cout << "After " << numBCRefinements << " refinements, phase space mesh has " << form->vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
        std::cout << "Physical mesh has " << form->configurationSpaceMesh()->numActiveElements() << " elements.\n";
      }
    }
    
    std::ostringstream suffix;
    if (!useL2Trace)
    {
      suffix << "_H1Trace";
    }
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
    
    const std::string problemPrefix = "cold-diode";
    
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
      vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovSS.str(), directoryPath));
      poissonViz = Teuchos::rcp(new HDF5Exporter (form->configurationSpaceMesh(), poissonSS.str(), directoryPath) );
    }
    
    FunctionPtr E_soln = Function::solution(form->configurationSpaceFormulation()->E_fieldVar(0), form->configurationSpaceSolution());
    FunctionPtr E_squared = E_soln * E_soln;
    FunctionPtr abs_E_soln = Teuchos::rcp( new AbsFunction(E_soln) );
    
    form->setFixedPointTolerance(fixedPointTol);
    form->setFixedPointMinSubsteps(minSubsteps);
    form->setFixedPointMaxSubsteps(maxSubsteps);
    
    //  form->vlasovSolution()->setWriteMatrixToFile(true, "/tmp/A_Vlasov.dat");
    
    const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
    auto solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form->vlasovSolution(), form->f(), form->configurationSpaceMesh()) );
    
    FunctionPtr potential = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), form->configurationSpaceSolution());
    
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
      
      auto poissonMeshTopo = form->configurationSpaceMesh()->getTopology()->baseMeshTopology();
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
      
      form->configurationSpaceSolution()->importSolutionForOffRankCells(physicalCellsFinalTime);
      
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
          auto basisCache = BasisCache::basisCacheForCell(form->configurationSpaceMesh(), physicalCellID, false, cubatureEnrichmentDegree); // false: not test versus test
          
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
        
        auto physicalCellNodes = form->configurationSpaceMesh()->physicalCellNodes(physicalCellIDsVector);
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
        
        double bc_norm = (bcFxnExact * bcRestriction)->l2norm(form->vlasovMesh(), quadratureEnrichment);
        bcRelErr = bc_err->l2norm(form->vlasovMesh(), quadratureEnrichment) / bc_norm;
        
        if (rank==0)
        {
          std::cout << "BC resolution, relative error in L^2 norm: " << bcRelErr << std::endl;
        }
      }
      auto numVlasovDofs = form->vlasovSolution()->mesh()->numGlobalDofs();
      auto vlasovElementCount  = form->vlasovMesh()->numActiveElements();
      auto poissonElementCount = form->configurationSpaceMesh()->numActiveElements();
      auto poissonDofCount     = form->configurationSpaceMesh()->numGlobalDofs();
      
      if (rank == 0)
      {
        writeRefReport(vlasovElementCount, numVlasovDofs, poissonElementCount, poissonDofCount, bcRelErr, energyError);
        std::cout << "******************************  Refinement " << refinementNumber << " complete. ******************************\n";
      }
    };
    
    FunctionPtr inflowRestriction = Function::meshSideSetCharacteristic(DIRICHLET_SET_TAG_NAME, form->tagVlasovSpatialInflow());
    const bool weightFluxesBySideParity = true;
    FunctionPtr tn_soln = Function::solution(form->t_n_hat(), form->vlasovSolution(), weightFluxesBySideParity);
    FunctionPtr tn_inflow = tn_soln * inflowRestriction;
    
    if (useTimeSlabs)
    {
      if (!useAdaptivity)
      {
        for (int timeSlabOrdinal=0; timeSlabOrdinal<numTimeSlabs; timeSlabOrdinal++)
        {
          if (rank == 0)
          {
            std::cout << "*******************************************************************\n";
            std::cout << "*                     Starting Time Slab " << timeSlabOrdinal;
            if      (timeSlabOrdinal < 10)   std::cout << "                        *\n";
            else if (timeSlabOrdinal < 100)  std::cout << "                       *\n";
            else if (timeSlabOrdinal < 1000) std::cout << "                      *\n";
            else                             std::cout << "                     *\n";
            std::cout << "*******************************************************************\n";
          }
          auto numVlasovDofs = form->vlasovSolution()->mesh()->numGlobalDofs();
          if (numVlasovDofs > maxDofsForDirectSolve)
          {
            if (rank == 0)
            {
              std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
            }
            form->setUseGeometricMultigrid(true);
          }
          else
          {
            if (rank == 0)
            {
              std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
            }
            form->setUseGeometricMultigrid(false);
          }
          
          form->solve();
          
          solutionMoments->update();
          
          if (saveViz)
          {
            std::string  vlasovSlabStr =  vlasovSS.str() + "_slab" + to_string(timeSlabOrdinal);
            std::string poissonSlabStr = poissonSS.str() + "_slab" + to_string(timeSlabOrdinal);
            
            auto vlasovVizSlab  = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovSlabStr, directoryPath));
            auto poissonVizSlab = Teuchos::rcp(new HDF5Exporter (form->configurationSpaceMesh(), poissonSlabStr, directoryPath) );
            
            poissonVizSlab->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"});
            vlasovVizSlab->exportSolution(form->vlasovSolution(), {cellIDFunction, tn_inflow}, {"cell ID (Vlasov)", "t_n (inflow boundary)"});
          }
          // TODO: figure out what to do with writeResults() in context of time slabs
          // writeResults(0);
          
          // next time slab
          form = form->nextTimeSlab(tWidth, tElements);
          
          {
            // solutions have been recreated for the new slab.  We need to update anything that depends on the solution
            const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
            solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form->vlasovSolution(), form->f(), form->configurationSpaceMesh()) );
            
            potential = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), form->configurationSpaceSolution());
            
            density  = solutionMoments->solutionMomentFunction(0);
            momentum = solutionMoments->solutionMomentFunction(1);
            
            velocity = momentum / density;
          }
        }
      }
      else // useAdaptivity = true
      {
        for (int timeSlabOrdinal=0; timeSlabOrdinal<numTimeSlabs; timeSlabOrdinal++)
        {
          Teuchos::RCP<HDF5Exporter> vlasovVizSlab, poissonVizSlab;
          
          if (saveViz)
          {
            std::string  vlasovSlabStr =  vlasovSS.str() + "_slab" + to_string(timeSlabOrdinal);
            std::string poissonSlabStr = poissonSS.str() + "_slab" + to_string(timeSlabOrdinal);
            
            vlasovVizSlab  = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovSlabStr, directoryPath));
            poissonVizSlab = Teuchos::rcp(new HDF5Exporter (form->configurationSpaceMesh(), poissonSlabStr, directoryPath) );
          }
          
          if (rank == 0)
          {
            std::cout << "*******************************************************************\n";
            std::cout << "*                     Starting Time Slab " << timeSlabOrdinal;
            if      (timeSlabOrdinal < 10)   std::cout << "                        *\n";
            else if (timeSlabOrdinal < 100)  std::cout << "                       *\n";
            else if (timeSlabOrdinal < 1000) std::cout << "                      *\n";
            else                             std::cout << "                     *\n";
            std::cout << "*******************************************************************\n";
          }
          RefinementStrategyPtr refStrategy;
          
          if (doerflerTarget <= 0)
          {
            refStrategy = Teuchos::rcp(new RefinementStrategy(form->vlasovSolution(), refinementThreshold));
            refStrategy->setMaxElements(maxElements); // allow unrefinements
          }
          else
          {
            ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(form->vlasovSolution());
            
            refStrategy = Teuchos::rcp(new DoerflerRefinementStrategy(errorIndicator, doerflerTarget) );
          }
          
          double energyError;
          auto energyErrorFunction = EnergyErrorFunction::energyErrorFunction(form->vlasovSolution());
          
          for (int refinementNumber=0; refinementNumber<numRefinements; refinementNumber++)
          {
            auto numVlasovDofs = form->vlasovSolution()->mesh()->numGlobalDofs();
            if (numVlasovDofs > maxDofsForDirectSolve)
            {
              if (rank == 0)
              {
                std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
              }
              form->setUseGeometricMultigrid(true);
            }
            else
            {
              if (rank == 0)
              {
                std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
              }
              form->setUseGeometricMultigrid(false);
            }
            
            form->solve();
            
            solutionMoments->update();
            
            energyError = refStrategy->computeTotalEnergyError(); // forces update to energyErrorFunction
            // TODO: revise reportBCResolutionAndRefinementComplete() to make it compatible with time-slabbing, or create a new reporting mechanism.
//            reportBCResolutionAndRefinementComplete(refinementNumber, energyError);
            
            if (saveViz)
            {
              poissonVizSlab->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"}, refinementNumber);
              vlasovVizSlab->exportSolution(form->vlasovSolution(), {cellIDFunction, tn_inflow}, {"cell ID (Vlasov)", "t_n (inflow boundary)"}, refinementNumber);
            }
            // TODO: figure out what to do with writeResults() in context of time slabs
            // writeResults(refinementNumber);
            
            refStrategy->refine(rank==0);
            
            // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
            form->configurationSpaceMesh()->repartitionAndRebuild();
            auto poissonElementCount = form->configurationSpaceMesh()->numActiveElements();
            auto poissonDofCount     = form->configurationSpaceMesh()->numGlobalDofs();
            if (rank == 0)
            {
              std::cout << "Poisson mesh has " << poissonElementCount << " active elements and " << poissonDofCount << " global degrees of freedom.\n";
            }
          } // refinement loop
          
          // next time slab
          form = form->nextTimeSlab(tWidth, tElements);
          
          {
            // solutions have been recreated for the new slab.  We need to update anything that depends on the solution
            const int maxMomentNumber = 1; // we want both density (moment 0) and momentum (moment 1)
            solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, form->vlasovSolution(), form->f(), form->configurationSpaceMesh()) );
            
            potential = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), form->configurationSpaceSolution());
            
            density  = solutionMoments->solutionMomentFunction(0);
            momentum = solutionMoments->solutionMomentFunction(1);
            
            velocity = momentum / density;
          }
        }
      } // end time slab + useAdaptivity
    }
    else // useTimeSlabs = false
    {
      if (!useAdaptivity)
      {
        auto numVlasovDofs = form->vlasovSolution()->mesh()->numGlobalDofs();
        if (numVlasovDofs > maxDofsForDirectSolve)
        {
          if (rank == 0)
          {
            std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
          }
          form->setUseGeometricMultigrid(true);
        }
        else
        {
          if (rank == 0)
          {
            std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
          }
          form->setUseGeometricMultigrid(false);
        }
        
        form->solve();
        
        solutionMoments->update();
        
        if (saveViz)
        {
          poissonViz->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"});
          vlasovViz->exportSolution(form->vlasovSolution(), {cellIDFunction, tn_inflow}, {"cell ID (Vlasov)", "t_n (inflow boundary)"});
        }
        writeResults(0);
      }
      else
      {
        RefinementStrategyPtr refStrategy;
        
        if (doerflerTarget <= 0)
        {
          refStrategy = Teuchos::rcp(new RefinementStrategy(form->vlasovSolution(), refinementThreshold));
          refStrategy->setMaxElements(maxElements); // allow unrefinements
        }
        else
        {
          ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(form->vlasovSolution());
          
          refStrategy = Teuchos::rcp(new DoerflerRefinementStrategy(errorIndicator, doerflerTarget) );
        }
        
        double energyError;
        auto energyErrorFunction = EnergyErrorFunction::energyErrorFunction(form->vlasovSolution());
        for (int refinementNumber=0; refinementNumber<numRefinements; refinementNumber++)
        {
          auto numVlasovDofs = form->vlasovSolution()->mesh()->numGlobalDofs();
          if (numVlasovDofs > maxDofsForDirectSolve)
          {
            if (rank == 0)
            {
              std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
            }
            form->setUseGeometricMultigrid(true);
          }
          else
          {
            if (rank == 0)
            {
              std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
            }
            form->setUseGeometricMultigrid(false);
          }
          
          //        {
          //          if (rank == 0)
          //          {
          //            std::cout << "zeroing electric field between adaptive solves; this is primarily for debugging.\n";
          //          }
          //          std::map<int, FunctionPtr> functionMap;
          //          functionMap[form->configurationSpaceFormulation()->E_fieldVar(0)->ID()] = Function::zero();
          //          form->configurationSpaceSolution()->projectOntoMesh(functionMap);
          //
          //          {
          //            auto E_extruded = form->configurationSpaceFormulation()->E_i(0);
          //
          //            // make sure E_extruded is ready to evaluate…
          //            static_cast<PhysicalSolutionExtrusion*>(E_extruded.get())->updateValues();
          //          }
          //        }
          
          form->solve();
          
          solutionMoments->update();
          
          writeResults(refinementNumber);
          
          energyError = refStrategy->computeTotalEnergyError(); // forces update to energyErrorFunction
          reportBCResolutionAndRefinementComplete(refinementNumber, energyError);
          
          if (saveViz)
          {
            poissonViz->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"}, refinementNumber);
            vlasovViz->exportSolution(form->vlasovSolution(), {energyErrorFunction,cellIDFunction,tn_inflow}, {"energy error","cell ID (Vlasov)", "t_n (inflow boundary)"}, refinementNumber);
          }
          
          refStrategy->refine(rank==0);
          
          // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
          form->configurationSpaceMesh()->repartitionAndRebuild();
          auto poissonElementCount = form->configurationSpaceMesh()->numActiveElements();
          auto poissonDofCount     = form->configurationSpaceMesh()->numGlobalDofs();
          if (rank == 0)
          {
            std::cout << "Poisson mesh has " << poissonElementCount << " active elements and " << poissonDofCount << " global degrees of freedom.\n";
          }
        }
        auto numVlasovDofs = form->vlasovSolution()->mesh()->numGlobalDofs();
        if (numVlasovDofs > maxDofsForDirectSolve)
        {
          if (rank == 0)
          {
            std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
          }
          form->setUseGeometricMultigrid(true);
        }
        else
        {
          if (rank == 0)
          {
            std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
          }
          form->setUseGeometricMultigrid(false);
        }
        
        //      {
        //        if (rank == 0)
        //        {
        //          std::cout << "zeroing electric field between adaptive solves; this is primarily for debugging.\n";
        //        }
        //        std::map<int, FunctionPtr> functionMap;
        //        functionMap[form->configurationSpaceFormulation()->E_fieldVar(0)->ID()] = Function::zero();
        //        form->configurationSpaceSolution()->projectOntoMesh(functionMap);
        //
        //        {
        //          auto E_extruded = form->configurationSpaceFormulation()->E_i(0);
        //
        //          // make sure E_extruded is ready to evaluate…
        //          static_cast<PhysicalSolutionExtrusion*>(E_extruded.get())->updateValues();
        //        }
        //      }
        
        form->solve();
        energyError = refStrategy->computeTotalEnergyError(); // forces update to energyErrorFunction
        
        solutionMoments->update();
        
        if (saveViz)
        {
          poissonViz->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"}, numRefinements);
          vlasovViz->exportSolution(form->vlasovSolution(), {energyErrorFunction,cellIDFunction,tn_inflow}, {"energy error","cell ID (Vlasov)", "t_n (inflow boundary)"}, numRefinements);
        }
        writeResults(numRefinements);
        
        reportBCResolutionAndRefinementComplete(numRefinements, energyError);
      }
    }
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
