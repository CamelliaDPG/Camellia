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
#include "RelativisticVlasovSpaceTimeFormulation.h"
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

class ErrorIndicatorFunction : public Function
{
  ErrorIndicatorPtr _errorIndicator;
public:
  ErrorIndicatorFunction(ErrorIndicatorPtr errorIndicator)
  :
  Function("ErrorIndicatorFunction", 0, Function::VARIES_IN_NO_DIMENSION), // constant on each cell
  _errorIndicator(errorIndicator)
  {}

  using Function::values;
  
  virtual void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    // we assume that measureError() has been called before values() is called
    // we also assume that values() will only be called for MPI-local cells
    
    const auto & errorMeasures = _errorIndicator->localErrorMeasures();
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();

    int numCells = cellIDs.size();
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      const auto &cellID = cellIDs[cellOrdinal];
      auto entry = errorMeasures.find(cellID);
      if (entry == errorMeasures.end())
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID not found in local errorMeasures");
      }
      const double error = entry->second;
      
      for (int pointOrdinal=0; pointOrdinal<values.dimension(1); pointOrdinal++)
      {
        values(cellOrdinal,pointOrdinal) = error;
      }
    }
  }
  
  void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache)
  {
    auto underlyingView1 = values.template getUnderlyingView<1>();
    
    const int numCells = underlyingView1.extent_int(0);
    
    const auto & errorMeasures = _errorIndicator->localErrorMeasures();
    
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.size() != numCells, std::invalid_argument, "underlying view doesn't have the expected length");
    
    auto underlyingViewHost = Kokkos::create_mirror_view(underlyingView1);
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      const auto &cellID = cellIDs[cellOrdinal];
      auto entry = errorMeasures.find(cellID);
      if (entry == errorMeasures.end())
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID not found in local errorMeasures");
      }
      const double error = entry->second;
      
      underlyingViewHost(cellOrdinal) = error;
    }
    
    Kokkos::deep_copy(underlyingView1, underlyingViewHost);
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

Teuchos::RCP<VlasovSpaceTimeFormulation> getFormulation1D1T1V(const double c,
                                                              const double x0, const double xWidth,
                                                              const double t0, const double tWidth,
                                                              const double v0, const double vWidth,
                                                              const int xElements, const int tElements, const int vElements,
                                                              double q, double eps0, double m,
                                                              const int H1Order, const int delta_k,
                                                              const double L2WeightGraph,
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
    
  form = Teuchos::rcp(new RelativisticVlasovSpaceTimeFormulation(c, q, m, configForm, spatialMeshTopo,
                                                                 temporalVertices,
                                                                 velocitySpaceVertices_x,
                                                                 velocitySpaceVertices_y,
                                                                 velocitySpaceVertices_z,
                                                                 H1Order, delta_k,
                                                                 numUniformRefinements,
                                                                 L2WeightGraph,
                                                                 basisFamily));
  return form;
}

std::vector<std::pair<std::pair<FunctionPtr,FunctionPtr>,int> > bcFunctions(Teuchos::RCP<VlasovSpaceTimeFormulation> form, FunctionPtr f_leftInflow)
{
  std::vector<std::pair<std::pair<FunctionPtr,FunctionPtr>,int> > functions;
  
  auto inflowTag = form->tagVlasovSpatialInflow();
  
  FunctionPtr sgn = Function::sideParity();
  FunctionPtr n = Function::normal();
  FunctionPtr n_x = n->spatialComponent(1); // 1-based
  FunctionPtr vx = form->v_i(0);
  
  FunctionPtr tn_leftInflowExact = f_leftInflow * vx; //  * sgn * n_x are also in there, but those are trivially resolved.
            
  const bool weightFluxesBySideParity = false;
  FunctionPtr tn_soln = Function::solution(form->t_n_hat(), form->vlasovSolutionPreviousSubstep(), weightFluxesBySideParity);
  functions.push_back({{tn_leftInflowExact, tn_soln},inflowTag});
  return functions;
}

void refineForBC(SolutionPtr soln, VarPtr bcVar, std::vector<std::pair<std::pair<FunctionPtr,FunctionPtr>,int> > &bcFunctionsToResolve,
                 int maxElements, double bcResolutionTolerance)
{
  MeshPtr mesh = soln->mesh();
  int rank = mesh->Comm()->getRank();
  int H1Order = mesh->globalDofAssignment()->getInitialH1Order()[0];
  
  for (auto bcPair : bcFunctionsToResolve)
  {
    FunctionPtr bcFxnExact = bcPair.first.first;
    FunctionPtr bcFxnSoln  = bcPair.first.second;
    int bcTag = bcPair.second;
    
    const std::string & tagName = DIRICHLET_SET_TAG_NAME;
    FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
    
    // project onto soln:
    std::map<int,FunctionPtr> bcMap;
    bcMap[bcVar->ID()] = bcFxnExact;
    soln->projectOntoMesh(bcMap);
    
    FunctionPtr bc_err = bcFxnExact * bcRestriction - bcFxnSoln * bcRestriction;
    int quadratureEnrichment = 4;
    
    auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(mesh, bc_err, quadratureEnrichment);
    errorIndicator->measureError();
    
    double refinementThresholdForInitialData = 0.80;
    auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
    refStrategy->setMaxP(H1Order);
    
    double bc_norm = (bcFxnExact * bcRestriction)->l2norm(mesh, quadratureEnrichment);
    double relErr = bc_err->l2norm(mesh, quadratureEnrichment) / bc_norm;
    
    if (rank == 0)
    {
      std::cout << "L^2 norm of BC: " << bc_norm << std::endl;
    }
    
    FunctionPtr bcErrIndicator = Teuchos::rcp( new ErrorIndicatorFunction(errorIndicator) );
    
    int numBCRefinements = 0;
    int numElements = mesh->numActiveElements();
    
    while ((relErr > bcResolutionTolerance) && (numElements < maxElements))
    {
      errorIndicator->measureError();
//      bcViz->exportSolution(soln, {cellIDFunction, bcFxnExact, bcFxnSoln * bcRestriction, bcFxnExact * bcRestriction, bc_err * bc_err, bcErrIndicator * bcRestriction}, {"cellID", "bcFxnExact (volume)", "bcFxnSoln", "bcFxnExact", "bcErr^2", "bcErrIndicator"}, numBCRefinements);
      
      if (rank==0)
      {
        std::cout << "After " << numBCRefinements << " refinements, phase space mesh has " << numElements << " elements and BC's relErr is " << relErr << ".\n";
      }
      numBCRefinements++;
      
      refStrategy->refine(rank==0);
      
      // project the BC onto the mesh again
      soln->projectOntoMesh(bcMap);
      
      bc_norm = (bcFxnExact * bcRestriction)->l2norm(mesh, quadratureEnrichment);
      relErr = bc_err->l2norm(mesh, quadratureEnrichment) / bc_norm;
      
      numElements = mesh->numActiveElements();
      
      if (rank==0) std::cout << "\n\n *************** BC refinement " << numBCRefinements << " complete *************** \n\n";
      
//        form->vlasovMesh()->getTopology()->baseMeshTopology()->printAllEntities();
    }
    errorIndicator->measureError();
//    bcViz->exportSolution(soln, {cellIDFunction, bcFxnExact, bcFxnSoln * bcRestriction, bcFxnExact * bcRestriction, bc_err * bc_err, bcErrIndicator * bcRestriction}, {"cellID", "bcFxnExact (volume)", "bcFxnSoln", "bcFxnExact", "bcErr^2", "bcErrIndicator"}, numBCRefinements);
    if (rank==0)
    {
      std::cout << "After " << numBCRefinements << " refinements, phase space mesh has " << mesh->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
//      std::cout << "Physical mesh has " << form->configurationSpaceMesh()->numActiveElements() << " elements.\n";
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

    const double e                = 1.602176634e-19;                                          // electron's electric charge in Coulombs
    const double m_e              =  9.10938356e-31;                                          // m_e, mass of electron in kg
    
//    const double J                = 9.33582e4;                                              // current density from SAND report [A/m^2]
    const double permittivity     = 8.85418782e-12;                                           // permittivity of free space (epsilon_0)
    const double hypergeom2F1     = 0.9222654768973;                                          // computed by Wolfram Alpha as: 2F1(1/4,3/4,7/4,-9.784756070E-01)
    const double D                = 0.01;
    const double c                = 2.997924580e8;
    const double V                = 1.e6;
    const double J_r              = 4./9. * permittivity * sqrt(2*e / m_e) * pow(V,1.5)/(D*D) * hypergeom2F1 * hypergeom2F1;
    
    double W                      = 1e4;                                                      // beam intensity from SAND report [eV]
    cmdp.setOption("W", &W, "beam intensity [eV]");
    
    double L2WeightGraph    = 1e4; // this is larger than our usual 1.0 factor, but it appears to be helpful with cold diode on coarse mesh.
    cmdp.setOption("L2WeightGraph", &L2WeightGraph, "Weight for the L^2 part of the graph norm");
    
    // proper velocity domain, specified as fraction of speed of light:
    double beta0                = 0.00;
    double betaMax              = 0.995;
    double cScaleFactor         = 0.5; // velocity will be scaled by cScaleFactor * c.
    
    double CCR                  = 0.5; // critical current ratio: < 1 means not space-charge limited
    bool usePeriodicBCs         = false;
    bool enforceQuasiNeutrality = false;
    bool useL2Trace             = true;
    bool useSerendipityBasis    = false;
    
    cmdp.setOption("ratio", &CCR, "fraction of J_r to use for injection current");
    
    cmdp.setOption("useL2Trace", "useH1Trace", &useL2Trace);
    cmdp.setOption("useSerendipityBasis", "useFullTensorBasis", &useSerendipityBasis);
    
    int delta_k = 2;
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
    double t0        = 0.0;
    double tWidth    = 1.0;
//    double v0        = 0.50;
//    double vWidth    = 1.00;
    
    cmdp.setOption("beta0", &beta0, "minimum speed-of-light fraction in velocity domain");
    cmdp.setOption("betaMax", &betaMax, "maximum speed-of-light fraction in velocity domain");
    cmdp.setOption("cScaleFactor", &cScaleFactor, "velocity scaling, specified as a fraction of the speed of light (i.e., 1.0 value corresponds to velocity scaling of c).");
    
    bool useTimeSlabs = true;
    int numTimeSlabs = 1;

    cmdp.setOption("numTimeSlabs", &numTimeSlabs);
    cmdp.setOption("tWidth", &tWidth);

    double sigma        = -1.0;  // if negative (i.e. left unset by user), set using sigmaFactor
    cmdp.setOption("sigma", &sigma, "(proper) thermal velocity at inflow, in mesh proper velocity units");
    double sigmaFactor  = 0.025; // (proper) thermal velocity = sigmaFactor * u0
    cmdp.setOption("sigmaFactor", &sigmaFactor, "(proper) thermal velocity at inflow as a fraction of u0");
    
    bool useAdaptivity = false;
    int maxRefinements = 3;
    int maxRefinementsFinalSlab = -1; // defaults to same as maxRefinements
    double refinementThreshold = 0.80; // refine elements with at least 80% of the maximum per-element error
    int maxElements = 100000; // do unrefinements to keep total mesh size smaller than this
    bool gateRefinements = true;
    double doerflerTarget = 0; // if set to positive value by user, use Doerfler instead of greedy strategy
    bool useEnergyForUnrefinements = false;
    double relErrorTarget = 0.01; // can stop adaptivity one refinement after this is met
    double relErrorTargetFinalSlab = -1.0; // defaults to same as relErrorTarget

    cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    cmdp.setOption("maxRefinements", &maxRefinements);
    cmdp.setOption("maxRefinementsFinalSlab", &maxRefinementsFinalSlab);
    cmdp.setOption("relErrorTarget", &relErrorTarget);
    cmdp.setOption("relErrorTargetFinalSlab", &relErrorTargetFinalSlab);
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

    int solutionOutputFrequency = 1;                    // only used if runOverkillVersusAdaptiveComparison is true
    int solutionComparisonFrequency = 1;                // only used if runOverkillVersusAdaptiveComparison is true
    
    cmdp.setOption("solutionOutputFrequency", &solutionOutputFrequency);
    cmdp.setOption("solutionComparisonFrequency", &solutionComparisonFrequency);
    
    int maxDofsForDirectSolve = 20000;
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    int iterativeSolveMaxIters  = 10000;

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
    
    // in the context of adaptivity, we can use looser tolerances for fixed point and iterative solves, prior to final solution
    double coarseIterativeSolveTolerance = 1e-4;
    int coarseIterativeSolveMaxIters  = 2000;
    cmdp.setOption("coarseIterativeSolveTolerance", &coarseIterativeSolveTolerance);
    cmdp.setOption("coarseIterativeSolveMaxIters",  &coarseIterativeSolveMaxIters);
    
    int coarseMinSubsteps = 2;
    int coarseMaxSubsteps = 10;
    double coarseFixedPointTol = 1e-2;
    
    cmdp.setOption("coarseMinSubsteps",   &coarseMinSubsteps);
    cmdp.setOption("coarseMaxSubsteps",   &coarseMaxSubsteps);
    cmdp.setOption("coarseFixedPointTol", &coarseFixedPointTol);

    double bcResolutionTolerance = 1e-1;  // only applies if doing an adaptive solve
    cmdp.setOption("bcResolutionTolerance", &bcResolutionTolerance);
    
    double initialTimeDivision = 0;
    cmdp.setOption("initialTimeDivision", &initialTimeDivision);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      return -1;
    }
    
    if (maxRefinementsFinalSlab < 0)
    {
      maxRefinementsFinalSlab = maxRefinements;
    }
    if (relErrorTargetFinalSlab < 0)
    {
      relErrorTargetFinalSlab = relErrorTarget;
    }
    
    const double gamma_beam       = 1 + W * e / (m_e * c * c);
    const double beta_beam        = sqrt(1. - 1./(gamma_beam * gamma_beam));
    const double v_beam           = c * beta_beam;
    const double velocityScaling  = cScaleFactor * c;
    const double spatialScaling   = 0.01;                             // [m]
    const double timeScaling      = spatialScaling / velocityScaling; // [s] // 1.97485663509e-9; // final time from SAND report [s]
    const double EScaling         = m_e * velocityScaling / (e * timeScaling);
    const double potentialScaling = EScaling * spatialScaling;
    const double q                = -1.0;
    const double m                =  1.0;
    const double cScaled          = c / velocityScaling;
    
    double v0     = v_beam / velocityScaling;
    double u0     = v0 * gamma_beam;
    double n0     = velocityScaling / v_beam; // computational density -- see nScaling below…
    
    if (sigma < 0)
    {
      sigma = sigmaFactor * u0;
    }
    else
    {
      sigmaFactor = sigma / u0;
    }
    
    const double J        = CCR * J_r;
    const double nScaling = J / (e * velocityScaling);                                         // number density of plasma
    const double omega_p  = sqrt( e * e * nScaling / (m_e * permittivity));
    const double eps0     =  1.0 / (omega_p * omega_p * timeScaling * timeScaling);
        
    double xWidth = D / spatialScaling;
    
    const double uMin = beta0   * cScaled / sqrt(1-beta0*beta0);
    const double uMax = betaMax * cScaled / sqrt(1-betaMax*betaMax);
    
    if (rank == 0)
    {
      std::cout << "Running relativistic cold diode problem.\n";
      std::cout << "J:                  " << J << " [A/m^2]\n";
      std::cout << "J_r:                " << J_r << " [A/m^2]\n";
      std::cout << "E scaling :         " << EScaling << std::endl;
      std::cout << "potential scaling : " << potentialScaling << std::endl;
      std::cout << "spatial scaling :   " << spatialScaling << std::endl;
      std::cout << "n scaling :         " << nScaling << std::endl;
      std::cout << "temporal scaling :  " << timeScaling << std::endl;
      std::cout << "velocity scaling :  " << velocityScaling << std::endl;
      std::cout << "c scaled:           " << cScaled  << std::endl;
      std::cout << "computational domain width: " << xWidth << std::endl;
      std::cout << "physical domain width: " << xWidth * spatialScaling << std::endl;
      std::cout << "initial time division: " << initialTimeDivision << std::endl;
      
      std::cout << "v_beam: " << v_beam << " [m/s] (" << v_beam / c  << " c)" << std::endl;
      std::cout << "uMin: " << uMin << " (" << beta0 << " c)\n";
      std::cout << "uMax: " << uMax << " (" << betaMax << " c)\n";
      std::cout << "sigma = " << sigma << std::endl;
      std::cout << "u0 = " << u0 << std::endl;
      std::cout << "v0 = " << v0 << std::endl;
      std::cout << "gamma0 = " << gamma_beam << std::endl;
      std::cout << "n0 = " << n0 << std::endl;
    }
    
    auto form = getFormulation1D1T1V(cScaled, x0, xWidth, t0, tWidth, uMin, uMax-uMin, xElements, tElements, vElements, q, eps0, m, H1Order, delta_k, L2WeightGraph, usePeriodicBCs,
                                     useSerendipityBasis, numUniformRefinements);
    form->setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    
    std::vector<std::pair<std::pair<FunctionPtr,FunctionPtr>,int> > bcFunctionsToResolve; // < <exact function, solution function>, tag> -- only used when useAdaptivity is true
    
    // we'll use this function to allow labeling of visualization output
    FunctionPtr cellIDFunction = Teuchos::rcp(new CellIDFunction);
    
    //    form->setUseGeometricMultigrid(useGeometricMultigrid);
    form->setIterativeSolveOutputLevel(iterativeSolveOutputLevel);
    //    form->setIterativeSolveTolerance(iterativeSolveTolerance);
    std::vector<double> cgTols(maxSubsteps,iterativeSolveTolerance);
    if ((maxSubsteps > 3) && !useAdaptivity) // for adaptive solves, after the 0 refinement, we'll already be pretty close to the right solution, and looser tolerances will tend to push us away.  Not worth it to special-case the 0 refinement, I think.  So we only use the looser CG tolerances below for non-adaptive solves.
    {
      // use looser tolerances for the first couple steps
      cgTols[0] = iterativeSolveTolerance * 10000.0;
      cgTols[1] = iterativeSolveTolerance * 100.0;
    }
    form->setIterativeSolveTolerances(cgTols);
    form->setIterativeSolveMaxIterations(iterativeSolveMaxIters);
    
    std::vector<double> cgTolsCoarse(coarseMaxSubsteps,coarseIterativeSolveTolerance);
    if ((coarseMaxSubsteps > 3) && !useAdaptivity) // for adaptive solves, after the 0 refinement, we'll already be pretty close to the right solution, and looser tolerances will tend to push us away.  Not worth it to special-case the 0 refinement, I think.  So we only use the looser CG tolerances below for non-adaptive solves.
    {
      // use looser tolerances for the first couple steps
      cgTolsCoarse[0] = coarseIterativeSolveTolerance * 100.0;
      cgTolsCoarse[1] = coarseIterativeSolveTolerance * 10.0;
    }
    
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
    
    FunctionPtr f0 = Function::zero(); // start with vacuum
    FunctionPtr f_leftInflow;
    
    {
      {
        // pin potential to zero at left, 1 megavolt at right:
        SpatialFilterPtr fullBoundaryFilter  = SpatialFilter::allSpace();
        bool restrictToBoundary = true; // cold diode: not periodic
        SpatialFilterPtr  leftBoundaryFilter = SpatialFilter::matchingX(x0);
        SpatialFilterPtr rightBoundaryFilter = SpatialFilter::matchingX(x0+xWidth);
        auto  leftBoundaryTag = form->createConfigurationMeshSideSetTag( leftBoundaryFilter, restrictToBoundary);
        auto rightBoundaryTag = form->createConfigurationMeshSideSetTag(rightBoundaryFilter, restrictToBoundary);
        auto VE_traceVar = form->VE_configurationSpaceTrace();
        form->addConfigurationSpaceBC( leftBoundaryTag, VE_traceVar, Function::zero());
        const double V_scaled = V / potentialScaling;
        form->addConfigurationSpaceBC(rightBoundaryTag, VE_traceVar, Function::constant(V_scaled));
        
        // set a Maxwellian at the inflow:
        FunctionPtr fM_u0 = Teuchos::rcp(new Maxwellian1T1V( u0, sigma, n0) );
        FunctionPtr positiveRestriction = Function::heaviside(0,2); // don't impose right-going conditions for negative velocities, even when Maxwellian extends past v=0
        f_leftInflow = fM_u0 * positiveRestriction;
        
        form->setInitialAndInflowConditions(f0, f_leftInflow, initialTimeDivision); // no right BC is only for negative velocity, so won't actually get triggered on our mesh
        
        if (useAdaptivity)
        {
          bcFunctionsToResolve = bcFunctions(form, f_leftInflow);
        }
      }
    }
    
    refineForBC(form->vlasovSolutionPreviousSubstep(), form->t_n_hat(), bcFunctionsToResolve, maxElements, bcResolutionTolerance);
    // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
    form->configurationSpaceMesh()->repartitionAndRebuild();
    if (rank == 0)
    {
      std::cout << "Physical mesh has " << form->configurationSpaceMesh()->numActiveElements() << " elements.\n";
    }
    
    std::ostringstream suffix;
    if (!useL2Trace)
    {
      suffix << "_H1Trace";
    }
    {
      suffix << "_relSigma" << sigmaFactor;
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
    suffix << "_CCR" << CCR;
    if (initialTimeDivision > 0)
    {
      suffix << "_timeRamp" << initialTimeDivision;
    }
    
    const std::string problemPrefix = "relativistic-cold-diode";
    
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
      
      refReportPathStringStream << getAMRSuffixForRefNumber(maxRefinements);
      refReportPathStringStream << ".dat";
      
      return refReportPathStringStream.str();
    };
    
    std::ostringstream vlasovSS;
    vlasovSS << problemPrefix << "Vlasov" << suffix.str();
    
    std::ostringstream poissonSS;
    poissonSS << problemPrefix << "Poisson" << suffix.str();
    
    if (useAdaptivity)
    {
      vlasovSS  << getAMRSuffixForRefNumber(maxRefinements);
      poissonSS << getAMRSuffixForRefNumber(maxRefinements);
    }
    
    const std::string directoryPath = outputDir;
    
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
        
//          if (saveViz)
//          {
//            std::string  vlasovSlabStr =  vlasovSS.str() + "_slab" + to_string(timeSlabOrdinal);
//            std::string poissonSlabStr = poissonSS.str() + "_slab" + to_string(timeSlabOrdinal);
//
//            auto vlasovVizSlab  = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovSlabStr, directoryPath));
//            auto poissonVizSlab = Teuchos::rcp(new HDF5Exporter (form->configurationSpaceMesh(), poissonSlabStr, directoryPath) );
//
//            poissonVizSlab->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"});
//            vlasovVizSlab->exportSolution(form->vlasovSolution(), {cellIDFunction}, {"cell ID (Vlasov)"});
//          }
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
        Teuchos::RCP<HDF5Exporter> vlasovVizSlabFinalRef, poissonVizSlabFinalRef;
        
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
        
        if (timeSlabOrdinal == numTimeSlabs - 1)
        {
          // final time slab
          maxRefinements = maxRefinementsFinalSlab;
          relErrorTarget = relErrorTargetFinalSlab;
        }
        
        for (int refinementNumber=0; refinementNumber<=maxRefinements; refinementNumber++)
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
          if (refinementNumber < maxRefinements)
          {
            // at least one refinement left; use coarse tolerances
            form->setIterativeSolveTolerances(cgTolsCoarse);
            form->setIterativeSolveMaxIterations(coarseIterativeSolveMaxIters);
            form->setFixedPointTolerance(coarseFixedPointTol);
            form->setFixedPointMinSubsteps(coarseMinSubsteps);
            form->setFixedPointMaxSubsteps(coarseMaxSubsteps);
          }
          else
          {
            form->setIterativeSolveTolerances(cgTols);
            form->setIterativeSolveMaxIterations(iterativeSolveMaxIters);
            form->setFixedPointTolerance(fixedPointTol);
            form->setFixedPointMinSubsteps(minSubsteps);
            form->setFixedPointMaxSubsteps(maxSubsteps);
          }
          
          form->solve();
          
          solutionMoments->update();
          
          energyError = refStrategy->computeTotalEnergyError(); // forces update to energyErrorFunction
          
          RieszRepPtr f_rieszRep;
          {
            std::map<int, FunctionPtr> f_map;
            auto f_var = form->f(0); // 0: species ordinal
            f_map[f_var->ID()] = Function::solution(f_var, form->vlasovSolution());
            auto f_functional = form->vlasovBF()->testFunctional(f_map); // bilinear form evaluated at the f solution
            f_rieszRep = RieszRep::rieszRep(form->vlasovMesh(), form->vlasovSolution()->ip(), f_functional);
          }
          double f_norm = f_rieszRep->getNorm();
          double relEnergyError = energyError / f_norm;
          
          bool errorTargetMet = (relEnergyError < relErrorTarget);
          
          if (rank == 0)
          {
            std::cout << "\n********************************************************************************\n";
            std::cout << "Time slab " << timeSlabOrdinal << ", ";
            std::cout << "solution after refinement " << refinementNumber << ": relative error " << energyError / f_norm * 100.0 << "%" << std::endl;
            if (errorTargetMet && (refinementNumber < maxRefinements))
            {
              std::cout << "Energy error target met; will perform one more solve with tightened tolerances.\n";
            }
            std::cout << "********************************************************************************\n\n";
          }
          
          if (errorTargetMet && (refinementNumber < maxRefinements))
          {
            form->setIterativeSolveTolerances(cgTols);
            form->setIterativeSolveMaxIterations(iterativeSolveMaxIters);
            form->setFixedPointTolerance(fixedPointTol);
            form->setFixedPointMinSubsteps(minSubsteps);
            form->setFixedPointMaxSubsteps(maxSubsteps);
            form->solve();
            
            energyError = refStrategy->computeTotalEnergyError(); // forces update to energyErrorFunction
            f_rieszRep->computeRieszRep(); // recompute now that we've solved to tighter tolerance
            f_norm = f_rieszRep->getNorm();
            relEnergyError = energyError / f_norm;
            errorTargetMet = (relEnergyError < relErrorTarget);
            
            if (rank == 0)
            {
              std::cout << "\n*******************************************************************\n";
              std::cout << "After re-solve, relative error " << energyError / f_norm * 100.0 << "%" << std::endl;
              if ( !errorTargetMet )
              {
                std::cout << "New error does not meet target; will not yet exit refinement loop early.\n";
              }
              std::cout << "*******************************************************************\n\n";
            }
          }
          
          // TODO: revise reportBCResolutionAndRefinementComplete() to make it compatible with time-slabbing, or create a new reporting mechanism.
//            reportBCResolutionAndRefinementComplete(refinementNumber, energyError);
          
          if (saveViz)
          {
            poissonVizSlab->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"}, refinementNumber);
            vlasovVizSlab->exportSolution(form->vlasovSolution(), {cellIDFunction, energyErrorFunction}, {"cell ID (Vlasov)", "energy error"}, refinementNumber);
          }
          // TODO: figure out what to do with writeResults() in context of time slabs
          // writeResults(refinementNumber);
          
          if (!errorTargetMet && (refinementNumber < maxRefinements))
          {
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
          else
          {
            if (saveViz)
            {
              std::string  vlasovSlabStr = "finalRef" + vlasovSS.str() + "_slab" + to_string(timeSlabOrdinal);
              std::string poissonSlabStr = "finalRef" + poissonSS.str() + "_slab" + to_string(timeSlabOrdinal);
              
              auto vlasovVizSlabFinalRef  = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovSlabStr, directoryPath));
              auto poissonVizSlabFinalRef = Teuchos::rcp(new HDF5Exporter (form->configurationSpaceMesh(), poissonSlabStr, directoryPath) );
              
              const int refNumber = 0; // "time" for ParaView
              const int defaultNum1DPoints = 2 * H1Order;
              poissonVizSlabFinalRef->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"},
                                                     defaultNum1DPoints);
              vlasovVizSlabFinalRef->exportSolution(form->vlasovSolution(), {cellIDFunction, energyErrorFunction}, {"cell ID (Vlasov)", "energy error"}, defaultNum1DPoints);
            }
            break; // exits loop early if error target met
          }
        } // refinement loop
        
        if (timeSlabOrdinal < numTimeSlabs - 1)
        {
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
            
            bcFunctionsToResolve = bcFunctions(form, f_leftInflow);
          }
          
          // refine the new slab to ensure resolution of the BC.
          refineForBC(form->vlasovSolutionPreviousSubstep(), form->t_n_hat(), bcFunctionsToResolve, maxElements, bcResolutionTolerance);
          // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
          form->configurationSpaceMesh()->repartitionAndRebuild();
          if (rank == 0)
          {
            std::cout << "Physical mesh has " << form->configurationSpaceMesh()->numActiveElements() << " elements.\n";
          }
        }
      } // timeSlabOrdinal loop
      
    } // end time slab + useAdaptivity
//    else // useTimeSlabs = false
//    {
//      if (!useAdaptivity)
//      {
//        auto numVlasovDofs = form->vlasovSolution()->mesh()->numGlobalDofs();
//        if (numVlasovDofs > maxDofsForDirectSolve)
//        {
//          if (rank == 0)
//          {
//            std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
//          }
//          form->setUseGeometricMultigrid(true);
//        }
//        else
//        {
//          if (rank == 0)
//          {
//            std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
//          }
//          form->setUseGeometricMultigrid(false);
//        }
//        
//        form->solve();
//        
//        solutionMoments->update();
//        
//        if (saveViz)
//        {
//          poissonViz->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"});
//          vlasovViz->exportSolution(form->vlasovSolution(), {cellIDFunction}, {"cell ID (Vlasov)"});
//        }
//        writeResults(0);
//      }
//      else
//      {
//        RefinementStrategyPtr refStrategy;
//        
//        if (doerflerTarget <= 0)
//        {
//          refStrategy = Teuchos::rcp(new RefinementStrategy(form->vlasovSolution(), refinementThreshold));
//          refStrategy->setMaxElements(maxElements); // allow unrefinements
//        }
//        else
//        {
//          ErrorIndicatorPtr errorIndicator = ErrorIndicator::energyErrorIndicator(form->vlasovSolution());
//          
//          refStrategy = Teuchos::rcp(new DoerflerRefinementStrategy(errorIndicator, doerflerTarget) );
//        }
//        
//        double energyError;
//        auto energyErrorFunction = EnergyErrorFunction::energyErrorFunction(form->vlasovSolution());
//        for (int refinementNumber=0; refinementNumber<maxRefinements; refinementNumber++)
//        {
//          auto numVlasovDofs = form->vlasovSolution()->mesh()->numGlobalDofs();
//          if (numVlasovDofs > maxDofsForDirectSolve)
//          {
//            if (rank == 0)
//            {
//              std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
//            }
//            form->setUseGeometricMultigrid(true);
//          }
//          else
//          {
//            if (rank == 0)
//            {
//              std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
//            }
//            form->setUseGeometricMultigrid(false);
//          }
//          
//          form->solve();
//          
//          solutionMoments->update();
//          
//          writeResults(refinementNumber);
//          
//          energyError = refStrategy->computeTotalEnergyError(); // forces update to energyErrorFunction
//          reportBCResolutionAndRefinementComplete(refinementNumber, energyError);
//          
//          if (saveViz)
//          {
//            poissonViz->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"}, refinementNumber);
//            vlasovViz->exportSolution(form->vlasovSolution(), {energyErrorFunction,cellIDFunction}, {"energy error","cell ID (Vlasov)"}, refinementNumber);
//          }
//          
//          refStrategy->refine(rank==0);
//          
//          // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
//          form->configurationSpaceMesh()->repartitionAndRebuild();
//          auto poissonElementCount = form->configurationSpaceMesh()->numActiveElements();
//          auto poissonDofCount     = form->configurationSpaceMesh()->numGlobalDofs();
//          if (rank == 0)
//          {
//            std::cout << "Poisson mesh has " << poissonElementCount << " active elements and " << poissonDofCount << " global degrees of freedom.\n";
//          }
//        }
//        auto numVlasovDofs = form->vlasovSolution()->mesh()->numGlobalDofs();
//        if (numVlasovDofs > maxDofsForDirectSolve)
//        {
//          if (rank == 0)
//          {
//            std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
//          }
//          form->setUseGeometricMultigrid(true);
//        }
//        else
//        {
//          if (rank == 0)
//          {
//            std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
//          }
//          form->setUseGeometricMultigrid(false);
//        }
//        
//        form->solve();
//        energyError = refStrategy->computeTotalEnergyError(); // forces update to energyErrorFunction
//        
//        solutionMoments->update();
//        
//        if (saveViz)
//        {
//          poissonViz->exportSolution(form->configurationSpaceSolution(), {density,velocity,cellIDFunction}, {"n_e","v","cell ID (Poisson)"}, maxRefinements);
//          vlasovViz->exportSolution(form->vlasovSolution(), {energyErrorFunction,cellIDFunction}, {"energy error","cell ID (Vlasov)"}, maxRefinements);
//        }
//        writeResults(maxRefinements);
//        
//        reportBCResolutionAndRefinementComplete(maxRefinements, energyError);
//      }
//    }
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
