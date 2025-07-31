//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"

#include "AbsFunction.h"
#include "BasisSumFunction.h"
#include "CamelliaDebugUtility.h"
#include "ConfigurationSpaceFormulation.h"
#include "EnergyErrorFunction.h"
#include "ErrorIndicator.h"
#include "ExpFunction.h"
#include "FixedCountRefinementStrategy.h"
#include "GDAMinimumRule.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MaxwellianFunction.h"
#include "MeshTestUtility.h"
#include "Projector.h"
#include "RefinementHistory.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "TimeRampFunction.h"
#include "TrigFunctions.h"
#include "VlasovBackwardEulerFormulation.h"

#include "Teuchos_LocalTestingHelpers.hpp"

#include <sys/stat.h>
#include <sys/types.h>

#include <cfenv>        // for std::fenv_t
#include <cmath>        // for sqrt()
#include <csignal>      // for signal()
#include <fenv.h>       // for fegetenv(), fesetenv()
#include <iostream>

void fpe_signal_handler(int /*signal*/) {
  std::cerr << "Floating point exception!\n";
  exit(1);
}

void enable_floating_point_exceptions() {
 std::fenv_t env;
 fegetenv(&env);
 env.__fpcr = env.__fpcr | __fpcr_trap_invalid | __fpcr_trap_divbyzero | __fpcr_trap_overflow;
 fesetenv(&env);
 signal(SIGFPE, fpe_signal_handler);
}

using namespace Camellia;

bool checkMeshParitiesAndVertices( MeshPtr mesh )
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
  return true; // we'll have thrown an exception if we failed
}

//! TODO: put this into the main Camellia source (i.e. part of the library)
class CellIDFunction : public Function
{
public:
  CellIDFunction() : Function(0) {}

  using Function::values;
  
  virtual void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();

    int numCells = cellIDs.size();
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<values.dimension(1); pointOrdinal++)
      {
        values(cellOrdinal,pointOrdinal) = cellIDs[cellOrdinal];
      }
    }
  }
};

// this communication strategy is wasteful, but we don't expect this to be a major bottleneck…
double getMaxMagnitudeOnCell(FunctionPtr fxn, MeshPtr mesh, GlobalIndexType &cellID)
{
  auto meshTopo = mesh->getTopology();
  auto cell = meshTopo->getCell(cellID);
  auto activeCells = cell->getDescendants(meshTopo);
  double maxValue = 0;
  
//  {
//    // DEBUGGING
//    int rank = mesh->Comm()->MyPID();
//    if (rank == 0)
//    {
//      std::cout << "Physical cell " << cellID;
//      if (activeCells.size() > 1)
//      {
//        std::cout << " has descendants: ";
//        Camellia::print("", activeCells);
//      }
//      else
//      {
//        std::cout << " is active.\n";
//      }
//    }
//  }
  for (auto activeCellID : activeCells)
  {
    double myMax = 0;
    if (mesh->myCellsInclude(activeCellID))
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, activeCellID);
      int numPoints = basisCache->getRefCellPointsFC().dimension(0);
      FieldContainer<double> values(1, numPoints);
      fxn->values(values, basisCache);
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        double pointValue = values(0, pointOrdinal);
        myMax = std::max(std::abs(pointValue), myMax);
      }
    }
    double globalMax = MPIWrapper::sum(*mesh->Comm(), myMax);
//    std::cout << "max value on cell " << activeCellID << ": " << globalMax << std::endl;
    maxValue = std::max(maxValue, globalMax);
  }
  return maxValue;
}

namespace Camellia
{
  class AnisotropicCFLRefinementStrategy : public TRefinementStrategy<double>
  {
  protected:
    MeshPtr _configurationSpaceMesh;
    double _cfl;
    double _dt;
    FunctionPtr _velocity_v; // i.e. the acceleration, defined on configurationSpaceMesh
  public:
    AnisotropicCFLRefinementStrategy(ErrorIndicatorPtr errorIndicator, double relativeErrorThreshold, MeshPtr configurationSpaceMesh, FunctionPtr velocity_v, double cfl, double dt):
    _configurationSpaceMesh(configurationSpaceMesh),
    _cfl(cfl),
    _velocity_v(velocity_v),
    _dt(dt),
    TRefinementStrategy<double>( errorIndicator, relativeErrorThreshold )
    {
      ExtrudedMeshTopology* extrudedMeshTopo = dynamic_cast<ExtrudedMeshTopology*>(mesh()->getTopology().get());
      TEUCHOS_TEST_FOR_EXCEPTION(extrudedMeshTopo == NULL, std::invalid_argument, "Phase-space MeshTopology must be an ExtrudedMeshTopology");
    }

    virtual void refineCells(vector<GlobalIndexType> &cellIDs) override
    {
      bool repartitionAndRebuild = false;
      
      // for now, we only refine isotropically or just in x; allowing v-anisotropic refinements would not add much value for this problem, and would add a bit of complexity in terms of the rules for refinements
      set<GlobalIndexType> cellsForIsotropicRefinement;
      set<GlobalIndexType> cellsForXRefinement;
      
      auto meshTopo = mesh()->getTopology();
      ExtrudedMeshTopology* extrudedMeshTopo = dynamic_cast<ExtrudedMeshTopology*>(meshTopo.get());
      TEUCHOS_TEST_FOR_EXCEPTION(extrudedMeshTopo == NULL, std::invalid_argument, "Phase-space MeshTopology must be an ExtrudedMeshTopology");
      
      bool printToConsole = (this->mesh()->Comm()->MyPID() == 0);
      
      auto physicalMeshTopo = extrudedMeshTopo->physicalMeshTopology();
      
      auto cellTopo = CellTopology::lineTensorTopology(CellTopology::line()); // line, once extruded
      auto isotropicRefinement = RefinementPattern::anisotropicRefinementPattern(cellTopo, {1,1});
      auto         xRefinement = RefinementPattern::xAnisotropicRefinementPatternExtrudedLine();
      
      auto canRefine = [&](const GlobalIndexType &cellID, bool &inX, bool &inV)
      {
        double v0, v1;
        extrudedMeshTopo->extrusionVerticesForPhaseSpaceCell(v0, v1, cellID, 0);
        
        using std::max;
        using std::abs;
        const double vx = max(abs(v0),abs(v1));
        
        GlobalIndexType physicalCellID = extrudedMeshTopo->physicalCellIDForPhaseSpaceCellID(cellID);
        
        const double vv = getMaxMagnitudeOnCell(_velocity_v, _configurationSpaceMesh, physicalCellID);
        double hv = abs(v1-v0);
        
        double x0, x1;
        
        CellPtr physicalCell = physicalMeshTopo->getCell(physicalCellID);
        const auto & vertexIndices = physicalCell->vertices();
        x0 = physicalMeshTopo->getVertex(vertexIndices[0])[0];
        x1 = physicalMeshTopo->getVertex(vertexIndices[1])[0];
        
        double hx = abs(x1-x0);
        
        inX = _cfl * (hx / 2) > _dt * vx;
        inV = _cfl * (hv / 2) > _dt * vv;
        
        const bool debuggingOutput = false;
        if (printToConsole && debuggingOutput)
        {
          std::cout << "cell " << cellID << " (x0,x1) x (v0,v1) = (" << x0 << "," << x1 << ") x (" << v0 << "," << v1 << "):\n";
          std::cout << "physical cell: " << physicalCellID << std::endl;
          std::cout << "hx = " << hx << std::endl;
          std::cout << "hv = " << hv << std::endl;
          std::cout << "vx = " << vx << std::endl;
          std::cout << "vv = " << vv << std::endl;
          
          if (inX)
          {
            std::cout << "CAN refine in x.\n";
          }
          else
          {
            std::cout << "CAN'T refine in x.\n";
          }
          if (inV)
          {
            std::cout << "CAN refine in v.\n";
          }
          else
          {
            std::cout << "CAN'T refine in v.\n";
          }
        }
      };
      
      auto refinementHalo = [&](auto &&refinementHalo, const GlobalIndexType &cellID, const bool &xRefinement, const bool &vRefinement, set<GlobalIndexType> &halo_x, set<GlobalIndexType> &halo_xv) -> void
      {
        // original caller should place the central cellID into halo_x and halo_v, if isotropic refinement of cellID is desired
        
        // halo_x:  those we should check for x-anisotropic refinement admissibility
        // halo_xv: those we should check for     isotropic refinement admissibility
        
        const int SOUTH_EDGE = 0;
        const int NORTH_EDGE = 1;
        const int WEST_EDGE  = 2;
        const int EAST_EDGE  = 3;
        vector<bool> checkEdgeNeighbor(4, false);
        if (xRefinement)
        {
          checkEdgeNeighbor[SOUTH_EDGE] = true;
          checkEdgeNeighbor[NORTH_EDGE] = true;
        }
        if (vRefinement)
        {
          checkEdgeNeighbor[ WEST_EDGE] = true;
          checkEdgeNeighbor[ EAST_EDGE] = true;
        }
        for (int i=0; i<4; i++)
        {
          if (checkEdgeNeighbor[i])
          {
            const int edgeDim = 1;
            CellPtr cell = extrudedMeshTopo->getCell(cellID);
            
            const int refBranchSize = int(cell->refinementBranchForSubcell(edgeDim, i, meshTopo).size()); // 0 if unconstrained
            
            if (refBranchSize > 0)
            {
              // then we'll want to refine neighbor as well
              auto neighborCellIndex = cell->getNeighborInfo(i, meshTopo).first;
              if ((i == WEST_EDGE) || (i == EAST_EDGE))
              {
                if ((halo_xv.find(neighborCellIndex) == halo_xv.end()) && (neighborCellIndex >= 0))
                {
                  // since we never do anisotropic v refinements, we need to assume an isotropic refinement of this neighbor
                  halo_xv.insert(neighborCellIndex);
                  refinementHalo(refinementHalo, neighborCellIndex, xRefinement, vRefinement, halo_x, halo_xv);
                }
              }
              else
              {
                if ((halo_x.find(neighborCellIndex) == halo_x.end()) && (neighborCellIndex >= 0))
                {
                  static const bool NO_V_REFINEMENT = false;
                  halo_x.insert(neighborCellIndex);
                  refinementHalo(refinementHalo, neighborCellIndex, xRefinement, NO_V_REFINEMENT, halo_x, halo_xv);
                }
              }
            }
          }
        }
      };
      
      set<GlobalIndexType> processedCells;
      
      auto cellHasBeenProcessed = [&](const GlobalIndexType &cellID) -> bool {
        return processedCells.find(cellID) != processedCells.end();
      };
      
      auto processCell = [&](const GlobalIndexType &cellID, std::set<GlobalIndexType> &additionalCellsToRefine) {
        processedCells.insert(cellID);
        additionalCellsToRefine.erase(cellID);
        
        set<GlobalIndexType> halo_x, halo_xv;
        
        bool xRefinementIsAllowed, vRefinementIsAllowed;
        canRefine(cellID, xRefinementIsAllowed, vRefinementIsAllowed);
        
        if (!xRefinementIsAllowed && !vRefinementIsAllowed)
        {
          return;
        }
        
        halo_x.insert(cellID);
        if (vRefinementIsAllowed)
        {
          halo_xv.insert(cellID);
        }
        
        refinementHalo(refinementHalo, cellID, xRefinementIsAllowed, vRefinementIsAllowed, halo_x, halo_xv);
        
        bool allHaloCellsAllowRefinement = true;
        for (auto xHaloCellID : halo_x)
        {
          // here, we only require that x refinement is allowed
          
          bool haloCellAllowsXRefinement, haloCellAllowsVRefinement;
          canRefine(xHaloCellID, haloCellAllowsXRefinement, haloCellAllowsVRefinement);
          
          allHaloCellsAllowRefinement &= haloCellAllowsXRefinement;
        }
        
        for (auto xvHaloCellID : halo_xv)
        {
          // here, we require that x and v refinement are allowed
          
          bool haloCellAllowsXRefinement, haloCellAllowsVRefinement;
          canRefine(xvHaloCellID, haloCellAllowsXRefinement, haloCellAllowsVRefinement);
          
          allHaloCellsAllowRefinement &= haloCellAllowsXRefinement && haloCellAllowsVRefinement;
        }
        
        if (allHaloCellsAllowRefinement)
        {
          if (xRefinementIsAllowed && vRefinementIsAllowed)
          {
            cellsForIsotropicRefinement.insert(cellID);
            
            for (auto haloCellID : halo_xv)
            {
              if (!cellHasBeenProcessed(haloCellID))
              {
                additionalCellsToRefine.insert(haloCellID);
              }
            }
            for (auto haloCellID : halo_x)
            {
              if (!cellHasBeenProcessed(haloCellID))
              {
                additionalCellsToRefine.insert(haloCellID);
              }
            }
          }
          else if (xRefinementIsAllowed)
          {
            cellsForXRefinement.insert(cellID);
            
            for (auto haloCellID : halo_x)
            {
              if (!cellHasBeenProcessed(haloCellID))
              {
                additionalCellsToRefine.insert(haloCellID);
              }
            }
          }
        }
        else if (vRefinementIsAllowed)
        {
          // we tried with both x and v and struck out; can we do just x?
          vRefinementIsAllowed = false;
          
          halo_x.insert(cellID);
          halo_xv.clear();
          
          refinementHalo(refinementHalo, cellID, xRefinementIsAllowed, vRefinementIsAllowed, halo_x, halo_xv);
          
          allHaloCellsAllowRefinement = true;
          for (auto xHaloCellID : halo_x)
          {
            // here, we only require that x refinement is allowed
            
            bool haloCellAllowsXRefinement, haloCellAllowsVRefinement;
            canRefine(xHaloCellID, haloCellAllowsXRefinement, haloCellAllowsVRefinement);
            
            allHaloCellsAllowRefinement &= haloCellAllowsXRefinement;
          }
          
          if (allHaloCellsAllowRefinement)
          {
            cellsForXRefinement.insert(cellID);
            for (auto haloCellID : halo_x)
            {
              if (!cellHasBeenProcessed(haloCellID))
              {
                additionalCellsToRefine.insert(haloCellID);
              }
            }
          }
        }
      };
      
      set<GlobalIndexType> additionalCellsToRefine;
      for (auto cellID : cellIDs)
      {
        processCell(cellID, additionalCellsToRefine);
      }
      
      while (additionalCellsToRefine.size() > 0)
      {
        set<GlobalIndexType> previousCellsToRefine = additionalCellsToRefine;
        additionalCellsToRefine.clear();
        
        for (auto cellID : previousCellsToRefine)
        {
          processCell(cellID, additionalCellsToRefine);
        }
      }
      
      if (printToConsole)
      {
        if (cellsForIsotropicRefinement.size() > 0) Camellia::print("cells for isotropic refinement",     cellsForIsotropicRefinement);
        if (cellsForXRefinement.size() > 0)         Camellia::print("cells for x-anisotropic refinement", cellsForXRefinement);
      }
      this->mesh()->hRefine(cellsForIsotropicRefinement, isotropicRefinement, repartitionAndRebuild);
      this->mesh()->hRefine(cellsForXRefinement, xRefinement, repartitionAndRebuild);
      if (printToConsole)
      {
        std::cout << "Vlasov mesh has " << this->mesh()->getTopology()->activeCellCount() << " active elements and " << this->mesh()->globalDofCount() << " degrees of freedom.\n";
      }
      return static_cast<int>(cellsForXRefinement.size() + cellsForIsotropicRefinement.size());
    }
  };

  class FluxLimitingErrorIndicator : public ErrorIndicator
  {
    VlasovBackwardEulerFormulation & _form;
  public:
    FluxLimitingErrorIndicator(VlasovBackwardEulerFormulation &form)
    :
    _form(form),
    ErrorIndicator(form.vlasovMesh())
    {
    }
    
    void measureError() override
    {
      auto meshTopo = mesh()->getTopology();
      ExtrudedMeshTopology* extrudedMeshTopo = dynamic_cast<ExtrudedMeshTopology*>(meshTopo.get());
      
      TEUCHOS_TEST_FOR_EXCEPTION(extrudedMeshTopo == NULL, std::invalid_argument, "Phase-space MeshTopology must be an ExtrudedMeshTopology");
      auto physicalMeshTopo = extrudedMeshTopo->physicalMeshTopology();
      auto cellTopo = CellTopology::lineTensorTopology(CellTopology::line()); // line, once extruded
      
      _localErrorMeasures.clear();
      
      auto myCells = this->mesh()->cellIDsInPartition();
      for (auto cellID : myCells)
      {
        double v0, v1;
        extrudedMeshTopo->extrusionVerticesForPhaseSpaceCell(v0, v1, cellID, 0);
        double hv = abs(v1-v0);
        
        GlobalIndexType physicalCellID = extrudedMeshTopo->physicalCellIDForPhaseSpaceCellID(cellID);
        double x0, x1;
        CellPtr physicalCell = physicalMeshTopo->getCell(physicalCellID);
        const auto & vertexIndices = physicalCell->vertices();
        x0 = physicalMeshTopo->getVertex(vertexIndices[0])[0];
        x1 = physicalMeshTopo->getVertex(vertexIndices[1])[0];
        double hx = abs(x1-x0);
        
        double cellMeasure = hx * hv;
        
        double theta   = _form.fluxLimitingWeightForCell(cellID);
        double exp_psi = _form.expPsiForCell(cellID);
        
        if (exp_psi < 1e-16)
        {
          // count that as 0
          exp_psi = 0;
        }
        double err = (1.0 - theta) * exp_psi * cellMeasure;
        
        _localErrorMeasures[cellID] = err;
        
//        if (err > 0)
//        {
//          std::cout << "error measure for cell " << cellID << ": " << err << std::endl;
//        }
      }
    }
  };
}

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

bool performRefinements(VlasovBackwardEulerFormulation &form, RefinementStrategyPtr refStrategy, RefinementStrategyPtr unrefinementStrategy,
                        int maxElements)
{
  if (refStrategy == Teuchos::null) return false;
  
  bool meshChanged = false;
  
  const bool doUnrefinements = (maxElements < std::numeric_limits<int>::max());
  if (doUnrefinements)
  {
    // for now, print refinement info to console on rank 0.
    const bool printToConsole = form.vlasovMesh()->Comm()->MyPID() == 0;
    unrefinementStrategy->setRelativeErrorThreshold(1.1); // ensure that we only do unrefinements
    int maxUnrefinementSteps = 1; // In case we end up unable to unrefine to maxElements for some reason, cap the number of unrefinements
    int numUnrefinementStepsTaken = 0;
    int unrefinementTarget = maxElements * .90;
    unrefinementStrategy->setMaxElements(unrefinementTarget);
    while ((form.vlasovMesh()->numActiveElements() > unrefinementTarget) && (numUnrefinementStepsTaken < maxUnrefinementSteps))
    {
      unrefinementStrategy->refineAndUnrefine(doUnrefinements, printToConsole);
      form.configurationSpaceMesh()->repartitionAndRebuild();
      form.reconstructGMGSolver();
      numUnrefinementStepsTaken++;
      int numRefinements = unrefinementStrategy->getNumRefinements();
      if (form.vlasovMesh()->numActiveElements() != unrefinementStrategy->getNumElements(numRefinements - 1))
      {
        form.finishTimeStep();
        meshChanged = true;
      }
    }
        
    if (numUnrefinementStepsTaken > 0)
    {
      // update solution to ensure that the errors measured are up to date post-refinement
      form.finishTimeStep();
    }
  }
  
  // for now, print refinement info to console on rank 0.
  const bool printToConsole = false; // form.vlasovMesh()->Comm()->MyPID() == 0;
  refStrategy->refineAndUnrefine(false, printToConsole); // false: don't unrefine
  
  int numRefinements = refStrategy->getNumRefinements();
  if (form.vlasovMesh()->numActiveElements() != refStrategy->getNumElements(numRefinements - 1))
  {
    meshChanged = true;
  }
  
  if (meshChanged)
  {
    // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
    form.configurationSpaceMesh()->repartitionAndRebuild();
    
    // also reconstruct GMG solver, if one is in use:
    form.reconstructGMGSolver();
  }
  return meshChanged;
}

//! Sets a homogeneous BC at the x0 point, and parameter-based BC at the x1 point.  Returns a pointer to the parameter function; the caller may then set it to an appropriate value before (any/each) Poisson solve.
ParameterFunctionPtr setPoissonBC(VlasovBackwardEulerFormulation &form, const double &x0, const double &x1)
{
  const int poissonTagLeft  = form.createConfigurationMeshSideSetTag(SpatialFilter::matchingX(x0));
  const int poissonTagRight = form.createConfigurationMeshSideSetTag(SpatialFilter::matchingX(x1));
  
  auto VE_hat = form.configurationSpaceFormulation()->VE_hat();
  // homogeneous on left:
  form.addConfigurationSpaceBC(poissonTagLeft, VE_hat, Function::constant(0.0));
  
  // parameter on right:
  ParameterFunctionPtr rightBCFunction = ParameterFunction::parameterFunction(0.0);
  form.addConfigurationSpaceBC(poissonTagRight, VE_hat, rightBCFunction);
  
  return rightBCFunction;
}

//! MPI-collective method for 1D meshes.  Evaluates f_soln on whatever rank owns the point x, and communicates the result to every rank.  Returns the value.
double getSolutionFunctionValue(MeshPtr solnMesh, FunctionPtr f_soln, const double x)
{
  FieldContainer<double> points(1,1);
  points(0,0) = x;
  
  auto cellIDs = solnMesh->cellIDsForPoints(points);
  
  double value = 0.0;
  if (cellIDs.size() > 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.size() != 1, std::invalid_argument, "Found more than one cellID for point x.");
    const auto cellID = cellIDs[0];
    
    if (solnMesh->myCellsInclude(cellID))
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(solnMesh, cellID);
      
      if (f_soln->boundaryValueOnly())
      {
        // i.e., a trace
        // 1D trace: sides are just points.  Does x match one of the sides?
        const auto & nodes = basisCache->getPhysicalCellNodes();
        double leftNode  = nodes(0,0,0); // (C,N,D)
        double rightNode = nodes(0,1,0);
        BasisCachePtr sideCache;
        if (std::abs(x-leftNode) < 1e-14)
        {
          sideCache = basisCache->getSideBasisCache(0);
        }
        else if (std::abs(x-rightNode) < 1e-14)
        {
          sideCache = basisCache->getSideBasisCache(1);
        }
        else
        {
          std::cout << "ERROR: x is not a mesh vertex, but f_soln is a trace solution.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertex not found");
        }
        
        FieldContainer<double> fValue(1,1); // (C,P)
        
        f_soln->values(fValue, sideCache);
        
        value = fValue(0,0);
      }
      else
      {
        // volume variable
        FieldContainer<double> refPoint(1,1,1);
        FieldContainer<double> physPoint(1,1,1);
        physPoint(0,0,0) = x;
        
        int cubatureDegree = 1; // not curvilinear…
        
        CamelliaCellTools::mapToReferenceFrame( refPoint, physPoint, solnMesh->getTopology(), cellID, cubatureDegree);
        
        refPoint.resize(1,1);
        
        basisCache->setRefCellPointsFC(refPoint);
        
        FieldContainer<double> fValue(1,1); // (C,P)
        
        f_soln->values(fValue, basisCache);
        
        value = fValue(0,0);
      }
    }
  }
  
  // all but the MPI-owner have a zero value, so we can sum to communicate the value to all ranks.
  value = MPIWrapper::sum(value);
  return value;
}

double inflowErrorOnBoundary(VlasovBackwardEulerFormulation &form, std::vector<FunctionPtr> f_in, const int bcTag, const int p_max)
{
  const std::string & tagName = DIRICHLET_SET_TAG_NAME;
  FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
  
  const int delta_k = form.vlasovMesh()->globalDofAssignment()->getTestOrderEnrichment();
  const int quadratureDegree = p_max * 2 + delta_k;
  const int quadratureEnrichment = (4 + quadratureDegree > Intrepid2::Parameters::MaxCubatureDegreeEdge) ? (Intrepid2::Parameters::MaxCubatureDegreeEdge - quadratureDegree) : 4;
  
  double errSquared = 0;
  double bcNormSquared = 0;
  for (int speciesOrdinal=0; speciesOrdinal<int(f_in.size()); speciesOrdinal++)
  {
    FunctionPtr f_soln = Function::solution(form.f(speciesOrdinal), form.vlasovSolution());
    FunctionPtr bc_err = (f_in[speciesOrdinal] - f_soln) * bcRestriction;
    
    double bc_norm = (f_in[speciesOrdinal] * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
    bcNormSquared += bc_norm * bc_norm;
    double err = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment);
    errSquared += err * err;
  }
  
  return sqrt(errSquared / bcNormSquared);
}

void setUseGMG(VlasovBackwardEulerFormulation &form, int maxDofsForDirectSolve, bool &previousSolveUsedGeometricMultigrid, bool &previousSolveUsedDirect, bool logChangeToConsole)
{
  auto numVlasovDofs = form.vlasovSolution()->mesh()->numGlobalDofs();
  if (numVlasovDofs > maxDofsForDirectSolve)
  {
    if (logChangeToConsole && (!previousSolveUsedGeometricMultigrid))
    {
      std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to true.\n";
    }
    form.setUseGeometricMultigrid(true);
    previousSolveUsedGeometricMultigrid = true;
    previousSolveUsedDirect = false;
  }
  else
  {
    if (logChangeToConsole && (previousSolveUsedGeometricMultigrid))
    {
      std::cout << "numVlasovDofs = " << numVlasovDofs << "; setting useGeometricMultigrid to false.\n";
    }
    form.setUseGeometricMultigrid(false);
    previousSolveUsedGeometricMultigrid = false;
    previousSolveUsedDirect = true;
  }
}

void patchTestVlasovPoisson(std::vector<FunctionPtr> f, // f exact at first n time steps (0 being the initial state)
                            FunctionPtr f_inflow,
                            FunctionPtr E_exact, FunctionPtr VE_exact,
                            std::vector<FunctionPtr> rhs_extra_load, // difference between the LHS and RHS of the temporally discretized strong form, evaluated at the exact solution at time k+1 (on the left) and at time k (on the right), for k = 0 to n-1
                            const double q,
                            VlasovBackwardEulerFormulation &form,
                            const bool overrideVlasovBCs,
                            const bool useCrankNicolson,
                            const double tol,
                            const double x0,
                            const double xWidth,
                            const double m,
                            Teuchos::FancyOStream &out, bool &success,
                            const bool outputViz=false, const string vizPrefix="")
{
  form.setUseGeometricMultigrid(true);
  form.setIterativeSolveTolerance(1e-15);
//    form.setIterativeSolveOutputLevel(10);
  form.setFixedPointMaxSubsteps(5);
  form.setFixedPointMinSubsteps(2);
  
  form.configurationSpaceSolution()->setBC(BC::bc()); // clear any previously set BCs
  
  FunctionPtr vx = form.v_i(0);

  auto vlasovSoln = form.vlasovSolution();
  
  auto poissonForm = form.configurationSpaceFormulation();
  auto poissonSoln = form.configurationSpaceSolution();
  FunctionPtr E_soln = Function::solution(poissonForm->E_fieldVar(0), poissonSoln);
  FunctionPtr f_soln = Function::solution(form.f(), vlasovSoln);
  
  SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(x0);
  SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(x0 + xWidth);
  
  auto xLeftRightTag = form.createConfigurationMeshSideSetTag(xLeftFilter | xRightFilter);
  auto VE_traceVar = form.VE_configurationSpaceTrace();
  form.addConfigurationSpaceBC(xLeftRightTag, VE_traceVar, VE_exact);
  
  // to do the "pure" Poisson solve, we do need to have an f solution defined, since that enters the RHS
  std::map<int, FunctionPtr> initialGuess_Vlasov; // project the exact field solution onto the mesh
  initialGuess_Vlasov[form.f()->ID()] = f[1];
//    initialGuess_Vlasov[form.t_n_hat()->ID()] = tn_0;
  vlasovSoln->projectOntoMesh(initialGuess_Vlasov);
  
  double f_globalIntegral = f[0]->integrate(form.vlasovMesh());
  std::cout << "f_globalIntegral: " << f_globalIntegral << std::endl;
  
//    const double m    = TEST_M;
//
//    FunctionPtr sgn = Function::sideParity();
//    FunctionPtr n   = Function::normal();
//    FunctionPtr tn_0 = f_BC * (vx * n->x() + (q/m) * E_exact * n->y()) * sgn;
//    // check that tn_0 can be represented on the mesh, at least to within tol
//    FunctionPtr tn_soln = Function::solution(form.t_n_hat(), vlasovSoln, false);
//    FunctionPtr tn_err = tn_0 - tn_soln;
//    double tn_projection_err = tn_err->l2norm(vlasovSoln->mesh());
//    double tn_soln_magnitude = tn_soln->l2norm(vlasovSoln->mesh());
//    out << "tn_soln_magnitude = " << tn_soln_magnitude << std::endl;
//    TEST_COMPARE(tn_projection_err, <=, tol * tn_soln_magnitude);
  
  
  
  form.configurationSpaceFormulation()->solve(true); // true: ensure that SolutionMomentFunction is properly set with the value that we just projected.
//    poissonSoln->solve();

  out << "Checking E_soln after pure Poisson solve.\n";
  double E_soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
  double E_err_l2 = (E_soln-E_exact)->l2norm(poissonSoln->mesh());

  TEST_COMPARE(E_err_l2, <=, max(tol * E_soln_magnitude, tol));

  out << "Checking VE_soln after pure Poisson solve.\n";
  
  FunctionPtr VE_soln = Function::solution(form.configurationSpaceFormulation()->VE_fieldVar(), poissonSoln);
  double VE_soln_magnitude = VE_soln->l2norm(poissonSoln->mesh());
  double VE_err_l2 = (VE_soln-VE_exact)->l2norm(poissonSoln->mesh());

  TEST_COMPARE(VE_err_l2, <=, max(tol * VE_soln_magnitude, tol));
  
  // initialize viz outputs, if using
  Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz, purePoissonViz, vlasovPriorTimeViz;
  if (outputViz)
  {
    std::ostringstream purePoissonVizPath;
    purePoissonVizPath << vizPrefix << "_PurePoisson";
    
    purePoissonViz = Teuchos::rcp(new HDF5Exporter(form.configurationSpaceMesh(), purePoissonVizPath.str(), "/tmp"));
    
    std::ostringstream vlasovVizPath;
    vlasovVizPath << vizPrefix << "_Vlasov";
    vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovVizPath.str(), "/tmp"));
    
    std::ostringstream vlasovPriorTimeVizPath;
    vlasovPriorTimeVizPath << vizPrefix << "_VlasovPriorTime";
    vlasovPriorTimeViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovPriorTimeVizPath.str(), "/tmp"));
    
    std::ostringstream poissonVizPath;
    poissonVizPath << vizPrefix << "_Poisson";
    
    poissonViz = Teuchos::rcp(new HDF5Exporter(form.configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
  }
  
  if (outputViz)
  {
    purePoissonViz->exportSolution(form.configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"});
  }

  // reset Vlasov solution:
  initialGuess_Vlasov[form.f()->ID()]       = Function::zero();
  initialGuess_Vlasov[form.t_n_hat()->ID()] = Function::zero();
  vlasovSoln->projectOntoMesh(initialGuess_Vlasov);
      
  out << "Setting initial condition: f = " << f[0]->displayString() << std::endl;
  out << "Setting inflow  condition: f = " << f_inflow->displayString() << std::endl;
  form.setInitialAndInflowConditions(f[0], f_inflow);
  
  if (outputViz)
  {
    vlasovViz->exportSolution(form.vlasovSolution(), std::vector<FunctionPtr> {f_soln-f[0], f[0], f_inflow, Function::zero(), Function::zero()}, std::vector<string>{"f err", "f exact", "f inflow", "t_n err", "t_n exact"}, 0);
    
    poissonViz->exportSolution(form.configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"}, 0);
  }
  
  auto w = form.w(); // test function for Vlasov RHS
  for (int k=0; k<int(f.size()) - 1; k++) // k: previous time step number (k+1 is current time step number)
  {
    FunctionPtr f_k   = f[k];
    FunctionPtr f_kp1 = f[k+1];
    
    FunctionPtr rhs_extra_load_k = (k >= 1) ? rhs_extra_load[k] - rhs_extra_load[k-1] : rhs_extra_load[k];
    
//      std::cout << "rhs_extra_load[" << k << "]: " << rhs_extra_load_k->displayString() << std::endl;
    
    vlasovSoln->rhs()->addTerm(rhs_extra_load_k * w);
    
    // for Crank-Nicolson, we let the trace be the average of the solution at time k and that at k+1.
    FunctionPtr f_BC = useCrankNicolson ? (0.5 * f_kp1 + 0.5 * f_k) : f_kp1;
    
    if (overrideVlasovBCs)
    {
      auto vlasovBC = BC::bc();
      VarPtr traceVar = form.t_n_hat();
      
      const int VX_COMPONENT = 0;
      
      const int minVelocityTag = form.tagVlasovMinVelocity(VX_COMPONENT);
      const int maxVelocityTag = form.tagVlasovMaxVelocity(VX_COMPONENT);
      
      vlasovBC->addDirichlet(traceVar, minVelocityTag, -f_BC * (q/m) * E_exact); // outward normal is negative
      vlasovBC->addDirichlet(traceVar, maxVelocityTag,  f_BC * (q/m) * E_exact); // outward normal is positive
      
      const int inflowTag = form.tagVlasovSpatialInflow();
      
      FunctionPtr n = Function::normal();
      
      FunctionPtr traceValue = f_BC * vx * n->x();
      vlasovBC->addDirichlet(traceVar, inflowTag,  traceValue);
      vlasovSoln->setBC(vlasovBC);
    }
    
    form.takeTimeStep();
    
    out << "Comparing f[" << k+1 << "] with f_soln after Vlasov-Poisson solve.\n";
    double f_soln_magnitude = f_soln->l2norm(vlasovSoln->mesh());
    double f_err_l2 = (f_soln-f_kp1)->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(f_err_l2, <=, tol * f_soln_magnitude);
    
    out << "Checking E_soln after Vlasov-Poisson solve.\n";
    E_soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
    E_err_l2 = (E_soln-E_exact)->l2norm(poissonSoln->mesh());
    
    TEST_COMPARE(E_err_l2, <=, tol * E_soln_magnitude);
    
    out << "Checking VE_soln after Vlasov-Poisson solve.\n";
    VE_soln_magnitude = VE_soln->l2norm(poissonSoln->mesh());
    VE_err_l2 = (VE_soln-VE_exact)->l2norm(poissonSoln->mesh());
    
    TEST_COMPARE(VE_err_l2, <=, tol * VE_soln_magnitude);

    out << "Comparing tn_0 with t_n_soln after Vlasov-Poisson solve.\n";

    FunctionPtr sgn = Function::sideParity();
    FunctionPtr n   = Function::normal();
    FunctionPtr tn_0 = f_BC * (vx * n->x() + (q/m) * E_exact * n->y()) * sgn;
    // check that tn_0 can be represented on the mesh, at least to within tol
    FunctionPtr tn_soln = Function::solution(form.t_n_hat(), vlasovSoln, false);
    double tn_soln_magnitude = tn_soln->l2norm(vlasovSoln->mesh());

    FunctionPtr tn_err = tn_0 - tn_soln;
    double tn_err_overall = tn_err->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(tn_err_overall, <=, tol * tn_soln_magnitude);

    TEST_ASSERT(tn_0->boundaryValueOnly() || tn_0->isZero()); // if zero, we allow it not to claim to be boundary-only.  (This is an artifact of the logic regarding Function products: zero times anything is zero, and we don't retain the boundary-valuedness in the zero result.  We could fix this, but it's not a high priority.)
    TEST_ASSERT(tn_soln->boundaryValueOnly());

    tn_soln_magnitude = tn_0->l2norm(vlasovSoln->mesh());
    double tn_err_l2 = (tn_soln-tn_0)->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(tn_err_l2, <=, tol * tn_soln_magnitude);
    
    if (outputViz)
    {
      double dt = form.timeStep();
      vlasovViz->exportSolution(form.vlasovSolution(), std::vector<FunctionPtr> {f_soln-f_kp1, f_kp1, f_inflow, tn_soln-tn_0, tn_0}, std::vector<string>{"f err", "f exact", "f inflow", "t_n err", "t_n exact"}, (k+1) * dt);
      
      vlasovPriorTimeViz->exportSolution(form.vlasovSolutionPriorTime(), k * dt); // output such that prior-time and current-time viz *should* agree at the same time; facilitates comparision in ParaView
      
      poissonViz->exportSolution(form.configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"}, (k+1) * dt);
    }
  }
}

void runTests(double vWidth, double xWidth,
              double x0,
              double dt, bool useCrankNicolson,
              double q, double m,
              VlasovBackwardEulerFormulation &form,
              std::string testTimingDescription) // e.g. "initial mesh"
{
  std::cout << "********* Running test on " << testTimingDescription << "*********\n";
  
  double expectedArea = vWidth * xWidth;
  FunctionPtr one = Function::constant(1.0);
  double actualArea = one->integrate(form.vlasovMesh());
  
  double areaTol = 1e-14;
  double areaErr = std::fabs(expectedArea - actualArea);
  double relAreaErr = areaErr / expectedArea;
  bool areaTest = relAreaErr < areaTol;
  
  if (areaTest)
  {
    std::cout << "PASS: Mesh passed area test.\n";
  }
  else
  {
    std::cout << "FAIL: Mesh passed area test: expected " << expectedArea << " but got " << actualArea << std::endl;
  }
  
  bool localGlobalConsistencyTest = MeshTestUtility::checkLocalGlobalConsistency(form.vlasovMesh());
  if (localGlobalConsistencyTest)
  {
    std::cout << "PASS: Mesh satisfied local-global consistency check.\n";
  }
  else
  {
    std::cout << "FAIL: Mesh failed local-global consistency check.\n";
  }
  bool paritiesAndVerticesTest = checkMeshParitiesAndVertices(form.vlasovMesh());
  if (paritiesAndVerticesTest)
  {
    std::cout << "PASS: Mesh satisfied parities/vertices consistency check.\n";
  }
  else
  {
    std::cout << "FAIL: Mesh failed parities/vertices consistency check.\n";
  }
  
  bool success = true;
  Teuchos::basic_FancyOStream<char> out(Teuchos::rcp(&std::cout,false));
  
  FunctionPtr x  = Function::xn(1);
  FunctionPtr v  = Function::yn(1);
  FunctionPtr v2 = v * v;
  FunctionPtr v3 = v * v * v;
  FunctionPtr t  = form.t();
  
  FunctionPtr f_inflow  = x * v * (1 + t + dt); // f evaluated at t + Delta t
  
  double theta = useCrankNicolson ? 0.5 : 1.0;
  
  std::vector<FunctionPtr> f;
  std::vector<FunctionPtr> rhs_extra_load;
  int numTimeSteps = 5;
  for (int k=0; k<=numTimeSteps; k++)
  {
    double t_k = dt * k;
    FunctionPtr f_k = x * v * (1 + t_k);
    f.push_back(f_k);
    if (k<numTimeSteps)
    {
      FunctionPtr rhs = x * v + (1 + t_k + theta * dt) * v2;
      rhs_extra_load.push_back(rhs);
    }
  }
  
  FunctionPtr E_exact = Function::zero();
  FunctionPtr VE_exact = Function::zero();
  
  const bool overrideVlasovBCs = false; // since E is zero, t_n will also be zero at the velocity extrema
  const double tol = 1e-12;
  
  // DEBUGGING
  bool outputViz = true;
  std::string vizPrefix = useCrankNicolson ? "Patch_CN" : "Patch_BE";

  patchTestVlasovPoisson(f, f_inflow, E_exact, VE_exact, rhs_extra_load, q,
                         form, overrideVlasovBCs, useCrankNicolson,
                         tol, x0, xWidth, m,
                         out, success, outputViz, vizPrefix);
  
  if (success)
  {
    std::cout << "PASS: Before final refinements, patch test succeeded.\n";
  }
  else
  {
    std::cout << "FAIL: Before final refinements, patch test failed.\n";
  }
}

int main(int argc, char *argv[])
{
  // enable_floating_point_exceptions();
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  const int rank = mpiSession.getRank();
  
  // block off variables so that everything gets destructed before the finalize() call at the end.
  {
    Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
    
    mpiSession.barrier();
    
    const int spaceDim = 1; // 1D1V
    const int velocityDim = 1; // 1D1V
    
    double q_e_scaled    = -1.0;
    double eps0          =  1.0; // overwritten below
    double q_ion_scaled  = -q_e_scaled;
    double x0     =  0.0;
    double xWidth =  1.0; // non-dimensionalized in some appropriate way
    double potentialBC = 12000; // in Volts (physical units); overridden to 0 if disableElectricField is set to true
    double voltageBCDamping = 1.; // a multiplier for the applied voltage; using this for diagnosis
    
    const double permittivity = 8.85418782e-12; // permittivity of free space
    
    double v0     = std::numeric_limits<double>::infinity();
    double vWidth = std::numeric_limits<double>::infinity();
    
    bool enforceQuasiNeutrality = false;
    bool enforcePositivity = true;
    
    bool useAdaptivity = false;
    int maxElements = 64 * 64; // do unrefinements to keep total mesh size smaller than this
    
    int xElements = 16;
    int vElements = 20;
    int numUniformRefinements = 2;
    int delta_k   = 2;
    double dt_max = -1.0;
    double timeRampWidth = 2.0; // assuming domain width of 1.0, velocity of 1.0, this fully turns on BCs after two transit times.  Modestly aggressive; could increase to 3.0 if there appear to be artifacts.
    int H1Order = 3; // this gives f, E (which are in L^2) a polynomial order of 2
    int minSubsteps = 2;  // for fixed-point iteration.
    int maxSubsteps = 15; // for fixed-point iteration.
    std::string normChoice = "graph"; // "conservative", "L2" are other choices
    double fixedPointTol = 1e-8;
    int outputFrequency = 1; // number of time steps to take before outputting viz.
    int maximumRefinementLevel = std::numeric_limits<int>::max(); // default to no maximum refinement level
    bool useEnergyForUnrefinements = false;
    bool saveViz = true;
    std::string outputDir = "/private/viz";
    int solutionOutputFrequency = 1;
    int iterativeSolveOutputLevel = 100; // output every one hundred CG iterations
    double iterativeSolveTolerance = 1e-9;
    double iterativeSolveMaxIters  = 10000;
    int maxDofsForDirectSolve = 80000;
    bool useSerendipityBasis = false;
    bool includeIons = true;
    bool includeElectrons = true;
    bool disableElectricField = false;
    bool disableMagneticField = false;
    double emFieldStartTime = 0.0; // allows running for some initial time before we start to turn on electric, magnetic fields
    double overrideEps0 = -1.; // if set from CLI, will override the computed eps0 value.
    bool useCrankNicolson = false;
    bool useConservativeNorm = false; // if true, override Vlasov IP to use one for which we can demonstrate local (mass) conservation
    double CFL_max = 1.0;      // CFL is used, together with dt and solution data, to set the minimum h
    double CFL_for_BCs = -1.0; // Allow user to set a different CFL for the initial BC refinements.  Not sure this is a good idea!
    double energyErrorTarget = 0.01; // try to refine to get below this threshold
    
    double physicalGapWidth = .00216; // [m], from Christenson and Lau (1994)
    double criticalCurrentRatio = 0.95; // I/I_C.  0.95 is case C2; 1.1 is case C3
    
    double alpha0 = 1.0;
    double alphaMultiplier = 1.0;
    double electronThermalVelocity = 0.25; // let thermal velocity be 25% of drift velocity
    double ionThermalVelocity = 0.25; // let thermal velocity be 25% of drift velocity
    
    double adaptiveRefinementThreshold     = 0.005; // refine any cell that has more than 0.5% of the maximum element error
    double fluxLimitingRefinementThreshold = 0.01; // refine any cells that have any appreciable error in terms of (1-theta) * exp psi * cell volume
    
    std::string resultsFilePrefix = "";
    
    std::string resultsFile = "";
    
    std::string refStrategyString = "anisotropicCFL"; // "greedy" or "fixed-count" alternatives
    
    cmdp.setOption("resultsFile", &resultsFile);
    cmdp.setOption("xElements", &xElements);
    cmdp.setOption("vElements", &vElements);
    cmdp.setOption("numUniformRefinements", &numUniformRefinements);
    cmdp.setOption("H1Order", &H1Order);
    cmdp.setOption("delta_k", &delta_k);
    cmdp.setOption("dt", &dt_max );
    cmdp.setOption("CFL", &CFL_max);
    cmdp.setOption("CFL_BC", &CFL_for_BCs, "CFL to use for initial BC-resolving refinements.  Default (-1) indicates to use the same CFL as for in-situ refinements.  (HIGHLY EXPERIMENTAL)");
    cmdp.setOption("electronThermalVelocity", &electronThermalVelocity, "electron thermal velocity as a fraction of electron drift velocity");
    cmdp.setOption("ionThermalVelocity", &ionThermalVelocity, "ion thermal velocity as a fraction of ion drift velocity");
    cmdp.setOption("timeRampWidth", &timeRampWidth);
    cmdp.setOption("resultsFilePrefix", &resultsFilePrefix);
    cmdp.setOption("v0", &v0, "minimum velocity coordinate in domain");
    cmdp.setOption("vWidth", &vWidth, "span of velocity coordinates in domain");
    cmdp.setOption("maxElements", &maxElements);
    cmdp.setOption("maximumRefinementLevel", &maximumRefinementLevel);
    cmdp.setOption("minSubsteps", &minSubsteps);
    cmdp.setOption("maxSubsteps", &maxSubsteps);
    cmdp.setOption("norm", &normChoice);
    cmdp.setOption("fixedPointTol", &fixedPointTol);
    cmdp.setOption("viz", "noViz", &saveViz);
    cmdp.setOption("outputFrequency", &outputFrequency);
    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    cmdp.setOption("refinementThreshold", &adaptiveRefinementThreshold);
    cmdp.setOption("refinementStrategy", &refStrategyString);
    cmdp.setOption("fluxLimitingRefinementThreshold", &fluxLimitingRefinementThreshold);
    cmdp.setOption("useEnergyForUnrefinements", "useL2ProjectionForUnrefinements", &useEnergyForUnrefinements);
    cmdp.setOption("energyErrorTarget", &energyErrorTarget);
    cmdp.setOption("solutionOutputFrequency", &solutionOutputFrequency);
    cmdp.setOption("useSerendipityBasis", "useFullTensorBasis", &useSerendipityBasis);
    cmdp.setOption("iterativeSolveOutputLevel", &iterativeSolveOutputLevel);
    cmdp.setOption("iterativeSolveMaxIters", &iterativeSolveMaxIters);
    cmdp.setOption("iterativeSolveTolerance", &iterativeSolveTolerance);
    cmdp.setOption("maxDofsForDirectSolve", &maxDofsForDirectSolve);
    cmdp.setOption("overrideEps0", &overrideEps0);
    cmdp.setOption("physicalGapWidth", &physicalGapWidth);
    cmdp.setOption("ions", "no-ions", &includeIons);
    cmdp.setOption("electrons", "no-electrons", &includeElectrons);
    cmdp.setOption("disableElectricField", "includeElectricField", &disableElectricField);
    cmdp.setOption("disableMagneticField", "includeMagneticField", &disableMagneticField);
    cmdp.setOption("emFieldStartTime", &emFieldStartTime);
    cmdp.setOption("enforcePositivity", "dontEnforcePositivity", &enforcePositivity);
    cmdp.setOption("alpha0", &alpha0);
    cmdp.setOption("alphaMultiplier", &alphaMultiplier);
    cmdp.setOption("useCrankNicolson", "useBackwardEuler", &useCrankNicolson);
    cmdp.setOption("useConservativeNorm", "useGraphNorm", &useConservativeNorm);
    cmdp.setOption("voltageBCDamping", &voltageBCDamping);
    cmdp.setOption("criticalCurrentRatio", &criticalCurrentRatio);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    bool useGreedyStrategy = false;
    bool useAnisotropicCFLRefStrategy = false;
    bool useFixedCountStrategy = false;
    
    if (refStrategyString == "anisotropicCFL")
    {
      useAnisotropicCFLRefStrategy = true;
    }
    else if (refStrategyString == "greedy")
    {
      useGreedyStrategy = true;
    }
    else if (refStrategyString == "fixed-count")
    {
      useFixedCountStrategy = true;
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unrecognized refinement strategy");
    }
    
    if (useConservativeNorm)
    {
      normChoice = "conservative";
    }
    
    if (CFL_for_BCs < 0)
    {
      CFL_for_BCs = CFL_max;
    }
    
    // Trying a velocity scaling that goes from -0.3c to +0.3c.
    const double DEFAULT_V0     = -1.00;
    const double DEFAULT_VWIDTH =  2.00;
    
    const double c         = 299792458;
    double velocityScaling = 0.30 * c;
    //    const double v_beam    = 0.00139891 * c;
    // scale so that our domain width is 1.0
    double spatialScaling  = physicalGapWidth;
    double timeScaling     = spatialScaling / velocityScaling;
    
    double EScaling        = 1.;
    double n0              = 1.;
    double potentialScaling = 1.;
    double currentScaling   = 1.;
    double BScaling         = 1.;
    double currentDensityScaling = 1.;
    
    double B = 0.27; // z-direction magnetic field
    
    FunctionPtr extraAccelerationFormula; // function of x
    
    double m_e_scaled;
    double m_ion_scaled;
    
    // determine scalings
    {
      const double e   = 1.602176634e-19; // electron's electric charge in Coulombs
      const double m_e =  9.10938356e-31;  // m_e, mass of electron in kg
      const double Omega = B * e / m_e; // cyclotron frequency in Hz
      const double OmegaScaling = 1.0 / timeScaling;
      potentialScaling = potentialBC;
      
      const double qScaling = - e / q_e_scaled; // charge scaling
      EScaling = potentialScaling / spatialScaling;
      if (rank == 0)
        cout << "potentialScaling: " << potentialScaling << std::endl;
      
      const double mScaling = EScaling * qScaling * timeScaling / velocityScaling;
      
      m_e_scaled   = m_e / mScaling;
      m_ion_scaled = m_e_scaled * 1836.15;
      
      xWidth = physicalGapWidth / spatialScaling;
      BScaling = potentialScaling * timeScaling / (spatialScaling * spatialScaling);
      
      FunctionPtr x = Function::xn(1);  // x^1
      const double B_scaled = B / BScaling;
      const double Omega_scaled = Omega / OmegaScaling;
      if (rank == 0)
        cout << "extra acceleration term: " << B_scaled * Omega_scaled << " x.\n";
      extraAccelerationFormula = B_scaled * Omega_scaled * x;
      
      const double J_C = 2.1e5; // A / m^2; critical value from Christenson & Lau
      n0 = J_C / (qScaling * velocityScaling); // number density of plasma
      
      const double omega_p = qScaling * sqrt( n0 / (mScaling * permittivity));
      //      std::cout << "omega_p = sqrt(n0) * " << q0 * sqrt( 1.0 / (m0 * permittivity)) << std::endl;
      
      // eps0 is a weight that enters PoissonFormulation; we use this to scale the problem:
      eps0 = 1.0 / (omega_p * omega_p * timeScaling * timeScaling);
      
      if (overrideEps0 != -1.0)
      {
        if (rank == 0)
        {
          std::cout << "NOTE: overriding eps0 (which was " << eps0 << ") with value " << overrideEps0 << std::endl;
        }
        eps0 = overrideEps0;
      }
      
      // we know that total current (density) is Iem + eps0 * dE / dt
      // we have a scaling for eps0: it's our eps0 divided by the physical permittivity of free space
      // we also have a scaling for E: EScaling
      // and for t: timeScaling
      // so the currentScaling should be (permittivity / eps0) * EScaling / timeScaling
      
      currentDensityScaling = (permittivity / eps0) * EScaling / timeScaling; // should match J_C
      currentScaling        = currentDensityScaling * spatialScaling * spatialScaling;
      
      if (rank == 0)
      {
        std::cout << "test norm:           " << normChoice << std::endl;
        std::cout << "E-field scaling:     " << EScaling << std::endl;
        std::cout << "B-field scaling:     " << BScaling << std::endl;
        std::cout << "V scaling:           " << potentialScaling << std::endl;
        std::cout << "physical gap width:  " << physicalGapWidth << std::endl;
        std::cout << "spatial scaling:     " << spatialScaling << std::endl;
        std::cout << "temporal scaling:    " << timeScaling << std::endl;
        std::cout << "domain width:        " << xWidth << std::endl;
        std::cout << "n0:                  " << n0 << std::endl;
        std::cout << "current scaling:     " << currentScaling << std::endl;
        std::cout << "J scaling:           " << currentDensityScaling << std::endl;
        std::cout << "eps0:                " << eps0 << std::endl;
      }
      enforceQuasiNeutrality = false;
    }
    
    if (vWidth == std::numeric_limits<double>::infinity())
    {
      vWidth = DEFAULT_VWIDTH;
    }
    if (v0 == std::numeric_limits<double>::infinity())
    {
      v0 = DEFAULT_V0;
    }
    
    bool usePeriodicBCs = false;
    
    double x1 = x0 + xWidth;
    
    std::vector<double> q_values;
    std::vector<double> m_values;
    std::vector<string> speciesLabels;
    if (includeElectrons)
    {
      q_values.push_back(q_e_scaled);
      m_values.push_back(m_e_scaled);
      speciesLabels.push_back("e");
    }
    if (includeIons)
    {
      q_values.push_back(q_ion_scaled);
      m_values.push_back(m_ion_scaled);
      speciesLabels.push_back("i");
    }
    const int numSpecies = int(q_values.size());
    
    if (disableElectricField)
    {
      potentialBC = 0;
    }
    
    // use parameter function to allow us to turn magnetic field on with a delay.
    ParameterFunctionPtr extraAccelerationParameter = ParameterFunction::parameterFunction(extraAccelerationFormula);
    
    auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q_values, eps0, TIME_MARCHING);
    if (!disableMagneticField)
    {
      poissonConfig->setExtraAcceleration({extraAccelerationParameter});
    }
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x1, xElements, usePeriodicBCs);
    std::vector<double> vxVertices;
    double vx = v0;
    for (int i=0; i<=vElements; i++)
    {
      vx = v0 + (i * vWidth) / vElements;
      vxVertices.push_back(vx);
    }
    Camellia::print("vxVertices", vxVertices);
    
    double dt = dt_max;
    int maxMomentNumber = 1; // we want to compute current for circuit solve.
    
    VlasovBackwardEulerFormulation form(q_values, m_values, speciesLabels, useSerendipityBasis, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k, maxMomentNumber, useCrankNicolson, enforcePositivity);
    
    form.setAlpha(alpha0, alphaMultiplier);
    
    form.setTestIP(normChoice);
    if (rank == 0)
    {
      std::cout << "Test space IP:\n";
      form.vlasovSolution()->ip()->printInteractions();
    }
    
//    if (disableElectricField)
//    {
//      // set the RHS to 0 -- in combination with the zero potential BC we set above, this forces the electric field to be zero.
//      // (This needs to be done after the VBEFormulation has been constructed, since it calls poissonConfig->setMoments(), which
//      //  is where the RHS for Poisson gets set up.)
//      poissonConfig->solution()->setRHS(RHS::rhs());
//    }
//    
//    // save off the RHS
//    RHSPtr poissonRHS = poissonConfig->solution()->rhs();
//    if (emFieldStartTime > 0)
//    {
//      // turn things off for now
//      poissonConfig->solution()->setRHS(RHS::rhs());
//      extraAccelerationParameter->setValue(0);
//    }
  
//    runTests(vWidth, xWidth, x0, dt, useCrankNicolson, q_values[0], m_values[0], form, "initial mesh");
    
    form.refineUniformly(numUniformRefinements);
    
//    runTests(vWidth, xWidth, x0, dt, useCrankNicolson, q_values[0], m_values[0], form, "uniformly refined mesh");
    
    auto refHistory = Teuchos::rcp( new RefinementHistory);
    refHistory->loadFromFile("/Users/nvrober/secondary-camellia/drivers/experimental/Ions/refHistoryForBug.txt");
    
    form.vlasovMesh()->registerObserver(refHistory);
    refHistory->playback(form.vlasovMesh());
    
    runTests(vWidth, xWidth, x0, dt, useCrankNicolson, q_values[0], m_values[0], form, "after ref history playback");
    
//    auto voltageParameter = setPoissonBC(form, x0, x1);
//    
//    auto moments_k   = form.solutionMomentsPriorTime(); // we need to call update() on this after each time step
//    auto moments_kp1 = form.solutionMoments();
//    
//    // voltageParameter is a "ParameterFunction" -- we need fill it in with an appropriate value
//    // we allow delay of EM fields turning on; if delayed, we ramp the EM field more quickly so that it comes fully on at the same time as the inflow BC.
//    bool useRampForEMField = false;
//    double emFieldFullyOnTime = useRampForEMField ? timeRampWidth : emFieldStartTime;
//    auto timeRampFunction = TimeRampFunction::timeRamp(form.t(), emFieldFullyOnTime, emFieldStartTime);
//    double voltageValue = potentialBC / potentialScaling * voltageBCDamping;  // 12000 V
//    voltageParameter->setValue(voltageValue * timeRampFunction->value()); // ramp up the voltage just as we do the inflow BC
//    
//    FunctionPtr velocityExtremaRestriction =   Function::meshSideSetCharacteristic(DIRICHLET_SET_TAG_NAME, form.tagVlasovMaxVelocity(0))
//    + Function::meshSideSetCharacteristic(DIRICHLET_SET_TAG_NAME, form.tagVlasovMinVelocity(0));
//    
//    std::vector<FunctionPtr> fAtExtrema(numSpecies);
//    for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
//    {
//      fAtExtrema[speciesOrdinal] = Function::solution(form.f(speciesOrdinal), form.vlasovSolution()) * velocityExtremaRestriction;
//    }
//    
//    if (rank == 0)
//    {
//      if (useCrankNicolson) std::cout << "Using Crank-Nicolson in time.\n";
//      std::cout << "Setting non-dimensional voltage at x=1 to " << potentialBC / potentialScaling << std::endl;
//    }
    
    auto configMesh = poissonConfig->configurationSpaceMesh();
    
//    // electrons flow in from left
//    double u0_e;
//    {
//      const double e    = 1.602176634e-19; // electron's electric charge in Coulombs
//      const double m_e  =  9.10938356e-31;  // m_e, mass of electron in kg
//      const double W    = 0.5; // [eV] beam intensity from Christenson & Lau
//      
//      // v_e is defined by 1/2 m v^2 = W ==> v_beam = sqrt( 2 W /m ):
//      const double v_e = sqrt(2 * W * e / m_e ); // e converts from eV to Joules
//      u0_e = v_e / velocityScaling;
//      if (rank == 0)
//      {
//        std::cout << "u0_e: " << u0_e << std::endl;
//      }
//    }
    
//    double sigmaSquared_e = electronThermalVelocity * electronThermalVelocity * u0_e * u0_e;
//    double n = 1.0;
//    const double firstMomentOfMaxwellian_e = MaxwellianFunction::momentNormalized1D(u0_e, sigmaSquared_e, 1); // will return u0_e
//    const double J_inflow_physical_e = criticalCurrentRatio * 2.1e5; // Amps // criticalCurrentRatio: CLI parameter (default of 0.95 is Case C2 from C & L; use 1.1 for Case C3)
//    const double J_inflow_scaled_e   = J_inflow_physical_e / currentDensityScaling;
//    const double inflow_weight_e     = J_inflow_scaled_e / firstMomentOfMaxwellian_e;
//    if (rank == 0)
//    {
//      std::cout << "sigma_e / u0_e: " << sqrt(sigmaSquared_e / (u0_e * u0_e) ) << std::endl;
//      std::cout << "inflow_weight_e: " << inflow_weight_e << std::endl;
//    }
    
//    FunctionPtr positiveRestriction = Function::heavisideY(0); // don't impose right-going conditions for negative velocities, even when Maxwellian extends past v=0
//    FunctionPtr fM_u0_e = inflow_weight_e * MaxwellianFunction::maxwellian(1, 1, {u0_e}, sigmaSquared_e, n) * positiveRestriction;
//    
//    // ions flow in from right
//    double u0_ion = -u0_e / sqrt(1836.15); // ratio of electron mass to ion mass (assigning them both the same energy)
//    double sigmaSquared_ion = ionThermalVelocity * ionThermalVelocity * u0_ion * u0_ion;
//    FunctionPtr negativeRestriction = 1 - Function::heavisideY(0); // don't impose right-going conditions for negative velocities, even when Maxwellian extends past v=0
//    FunctionPtr fM_u0_ion = MaxwellianFunction::maxwellian(1, 1, {u0_ion}, sigmaSquared_ion, n) * negativeRestriction;
//    
//    FunctionPtr f_e_in = fM_u0_e;
//    FunctionPtr f_i_in = fM_u0_ion;
//    
//    ErrorIndicatorPtr unrefinementErrorIndicator;
//    if (!useEnergyForUnrefinements)
//    {
//      unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(form.vlasovSolution(), form.f());
//    }
//    double refinementThreshold = 1.1; // > 1 ==> no refinements will take place
//    RefinementStrategyPtr unrefinementStrategy = constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator);
//    
//    form.setFixedPointMaxSubsteps(maxSubsteps);
//    form.setFixedPointMinSubsteps(minSubsteps);
//    form.setFixedPointTolerance(fixedPointTol);
//    
//    form.setIterativeSolveOutputLevel(iterativeSolveOutputLevel);
//    form.setIterativeSolveTolerance(iterativeSolveTolerance);
//    std::vector<double> cgTols(maxSubsteps,iterativeSolveTolerance);
//    if (maxSubsteps > 3)
//    {
//      // use looser tolerances for the first couple steps
//      cgTols[0] = iterativeSolveTolerance * 100.0;
//      cgTols[1] = iterativeSolveTolerance * 10.0;
//      form.setIterativeSolveTolerances(cgTols);
//    }
//    form.setIterativeSolveMaxIterations(iterativeSolveMaxIters);
//    form.setFixedPointTolerance(fixedPointTol);
//    
//    using std::vector;
//    using std::tuple;
//    vector<tuple<VarPtr,FunctionPtr,FunctionPtr,int> > bcFunctionsToResolve; // (var, exact function, solution function, tag)
//    
//    std::vector<FunctionPtr> vlasovFunctionsToExport;
//    std::vector<std::string> vlasovFunctionsToExportNames;
//    
//    int speciesOrdinal = 0;
//    std::vector<FunctionPtr> f0, f_in;
//    if (includeElectrons)
//    {
//      f0.push_back(Function::zero()); // vacuum initial conditions for both species
//      f_in.push_back(f_e_in);
//      
//      VarPtr tn_e = form.t_n_hat(speciesOrdinal);
//      
//      auto inflowTag = form.tagVlasovSpatialInflow();
//      
//      FunctionPtr sgn = Function::sideParity();
//      FunctionPtr n = Function::normal();
//      FunctionPtr n_x = n->spatialComponent(1); // 1-based
//      FunctionPtr vx = form.v_i(0);
//      
//      vlasovFunctionsToExportNames.push_back("f_e_in");
//      vlasovFunctionsToExport.push_back(f_e_in);
//      
//      FunctionPtr tn_inflowExact = f_e_in * sgn * n_x * vx;
//      const bool weightFluxesBySideParity = true;
//      FunctionPtr tn_soln = Function::solution(tn_e, form.vlasovSolution(), weightFluxesBySideParity);
//      bcFunctionsToResolve.push_back({tn_e,tn_inflowExact,tn_soln,inflowTag});
//      speciesOrdinal++;
//    }
//    if (includeIons)
//    {
//      f0.push_back(Function::zero()); // vacuum initial conditions for both species
//      f_in.push_back(f_i_in);
//      
//      VarPtr tn_i = form.t_n_hat(speciesOrdinal);
//      
//      auto inflowTag = form.tagVlasovSpatialInflow();
//      
//      FunctionPtr sgn = Function::sideParity();
//      FunctionPtr n = Function::normal();
//      FunctionPtr n_x = n->spatialComponent(1); // 1-based
//      FunctionPtr vx = form.v_i(0);
//      
//      vlasovFunctionsToExportNames.push_back("f_i_in");
//      vlasovFunctionsToExport.push_back(f_i_in);
//      
//      FunctionPtr tn_inflowExact = f_i_in * sgn * n_x * vx;
//      const bool weightFluxesBySideParity = true;
//      FunctionPtr tn_soln = Function::solution(tn_i, form.vlasovSolution(), weightFluxesBySideParity);
//      bcFunctionsToResolve.push_back({tn_i,tn_inflowExact,tn_soln,inflowTag});
//    }
//    form.setInitialAndInflowConditions(f0, f_in, timeRampWidth);
//    
//    FunctionPtr Ex = Function::solution(poissonConfig->E_fieldVar(0), form.configurationSpaceSolution());
//    
//    FunctionPtr vxAcceleration = Ex + extraAccelerationFormula;
//    double maxAcceleration_vx = vxAcceleration->maximumValue(poissonConfig->configurationSpaceMesh());
//    if (rank == 0)
//    {
//      std::cout << "Max acceleration on initial solution is " << maxAcceleration_vx << std::endl;
//    }
//    //    double u_mesh = max(maxAcceleration_vx, v0 + vWidth);
//    //    double h_min =  u_mesh * dt / CFL_max;
//    int    p_max = H1Order;  // for now, no p refinements.  (May want to revisit; if we do, can set p_max = 10 - delta_k -- p is limited by Intrepid2 quadrature rules at the moment.)
//    
//    ErrorIndicatorPtr errorIndicator;
//    if (!enforcePositivity)
//    {
//      errorIndicator = ErrorIndicator::energyErrorIndicator(form.vlasovSolution());
//    }
//    else
//    {
//      // want adaptivity to refine where the *original* problem has large residuals
//      // the second equation (defining the latent variable) in particular may distort the error
//      
//      const int indexOffsetSpaceOneBased    = 1;
//      const int indexOffsetVelocityOneBased = spaceDim + 1;
//      
//      LinearTermPtr residual;
//      
//      for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
//      {
//        FunctionPtr f_i = Function::solution(form.f(speciesOrdinal), form.vlasovSolution());
//        VarPtr w_i  = form.w(speciesOrdinal);
//        residual = residual + f_i * w_i;
//      }
//      
//      std::vector<VarPtr> w(numSpecies);
//      std::vector<FunctionPtr> t_n_hat_soln(numSpecies);
//      std::vector<FunctionPtr> f_soln(numSpecies);
//      std::vector<FunctionPtr> t_n_hat_soln_prev(numSpecies);
//      std::vector<FunctionPtr> f_soln_prev(numSpecies);
//      auto vlasovSolnWeakPtr = Teuchos::rcp(form.vlasovSolution().get(), false);
//      auto vlasovSolnPriorTimeWeakPtr = Teuchos::rcp(form.vlasovSolutionPriorTime().get(), false);
//      for (int i=0; i<numSpecies; i++)
//      {
//        w[i]            = form.w(i);
//        f_soln[i]       = Function::solution(form.f(i), vlasovSolnWeakPtr);
//        bool weightFluxesBySideParity = true;
//        t_n_hat_soln[i] = Function::solution(form.t_n_hat(), vlasovSolnWeakPtr, weightFluxesBySideParity);
//        
//        f_soln_prev[i]  = Function::solution(form.f(i), vlasovSolnPriorTimeWeakPtr);
//        t_n_hat_soln_prev[i] = Function::solution(form.t_n_hat(), vlasovSolnPriorTimeWeakPtr, weightFluxesBySideParity);
//      }
//      
//      std::vector<LinearTermPtr> w_against_f_i(numSpecies);
//      
//      const double theta = useCrankNicolson ? 0.5 : 1.0;
//      
//      const bool l2Trace = true;
//      for (int i=0; i<numSpecies; i++)
//      {
//        w_against_f_i[i] = w[i] / dt;
//        
//        residual = residual + t_n_hat_soln[i] * w[i];
//        
//        for (int d=0; d<spaceDim; d++)
//        {
//          auto v_d = form.v_i(d);
//          
//          w_against_f_i[i] = w_against_f_i[i] - theta * v_d * w[i]->di(indexOffsetSpaceOneBased + d); // dx_d
//        }
//      }
//      
//      for (int i=0; i<numSpecies; i++)
//      {
//        auto q_i = form.q(i);
//        auto m_i = form.m(i);
//        for (int d=0; d<velocityDim; d++)
//        {
//          auto weight = form.a()[d];
//          w_against_f_i[i] = w_against_f_i[i] - theta * (q_i/m_i) * weight * w[i]->di(indexOffsetVelocityOneBased + d); // d/dv_d
//        }
//      }
//      // now add in the f terms:
//      for (int i=0; i<numSpecies; i++)
//      {
//        residual = residual + (f_soln[i] - f_soln_prev[i]) * w_against_f_i[i];
//      }
//      
//      IPPtr ip = IP::ip();
//      for (int i=0; i<numSpecies; i++)
//      {
//        ip->addTerm(w[i]);
//        ip->addTerm(w_against_f_i[i]);
//        
//        ip->addTerm(form.phi(i)); // this term and its weight should not matter; the phi coefficients will always be 0, because they do not enter the residual.
//      }
//      
////      std::cout << "residual for error indicator: " << residual->displayString() << std::endl;
////
////      std::cout << "ip for error indicator:\n";
////      ip->printInteractions();
//      
//      // EnergyErrorIndicator(MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip, int cubatureEnrichmentDegree = 0);
//      errorIndicator = ErrorIndicator::energyErrorIndicator(form.vlasovMesh(), residual, ip, 0);
//    }
    
    form.vlasovMesh()->Comm()->Barrier();
    
//    const bool refineToResolveBCs = useAdaptivity;
//    double bcResolutionTolerance = 1e-6;
//    if (refineToResolveBCs)
//    {
//      for (auto bcTuple : bcFunctionsToResolve)
//      {
//        using std::get;
//        VarPtr var = get<0>(bcTuple);
//        FunctionPtr bcFxnExact = get<1>(bcTuple);
//        FunctionPtr bcFxnSoln  = get<2>(bcTuple);
//        int bcTag = get<3>(bcTuple);
//        
//        const std::string & tagName = DIRICHLET_SET_TAG_NAME;
//        FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
//        
//        // project onto vlasovSolution:
//        std::map<int,FunctionPtr> bcMap;
//        bcMap[var->ID()] = bcFxnExact;
//        form.vlasovSolution()->projectOntoMesh(bcMap);
//        
//        FunctionPtr bc_err = bcFxnExact * bcRestriction - bcFxnSoln * bcRestriction;
//        int quadratureDegree = p_max * 2 + delta_k;
//        int quadratureEnrichment = (4 + quadratureDegree > Intrepid2::Parameters::MaxCubatureDegreeEdge) ? (Intrepid2::Parameters::MaxCubatureDegreeEdge - quadratureDegree) : 4;
//        
//        auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(form.vlasovMesh(), bc_err, quadratureEnrichment);
//        
//        double refinementThresholdForInitialData = 0.80;
//        
//        RefinementStrategyPtr refStrategy;
//        if (useAnisotropicCFLRefStrategy)
//        {
//          refStrategy = Teuchos::rcp(new AnisotropicCFLRefinementStrategy(errorIndicator, refinementThresholdForInitialData, poissonConfig->configurationSpaceMesh(), vxAcceleration, CFL_for_BCs, dt) );
//          refStrategy->setEnforceOneIrregularity(false);
//        }
//        else
//        {
//          refStrategy = Teuchos::rcp( new RefinementStrategy( errorIndicator, refinementThresholdForInitialData, 0, H1Order, false));
//          refStrategy->setMaximumRefinementLevel(maximumRefinementLevel);
//          refStrategy->setEnforceOneIrregularity(true);
//        }
//        
//        //      auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
//        //      refStrategy->setMaxElements(maxElements);
//        //      refStrategy->setMaxP(p_max);
//        //      refStrategy->setMinH(h_min);
//        
//        double bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
//        double relErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
//        
//        int numBCRefinements = 0;
//        int numVlasovElements = form.vlasovMesh()->numActiveElements();
//        int prevNumVlasovElements = numVlasovElements;
//        int numGlobalDofs     = form.vlasovMesh()->globalDofCount();
//        int prevNumGlobalDofs = 0;
//        while ((relErr > bcResolutionTolerance) && (numVlasovElements < maxElements))
//        {
//          // recompute h_min
//          //        maxAcceleration_vx = vxAcceleration->maximumValue(poissonConfig->configurationSpaceMesh());
//          //        u_mesh = max(maxAcceleration_vx, v0 + vWidth);
//          //        h_min =  u_mesh * dt / CFL_for_BCs;
//          //        refStrategy->setMinH(h_min);
//          
//          if (rank==0)
//          {
//            std::cout << "After " << numBCRefinements++ << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
//            std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
//            //          std::cout << "h_min is " << h_min << std::endl;
//          }
//          
//          refStrategy->refine();
//          
//          // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
//          form.configurationSpaceMesh()->repartitionAndRebuild();
//          
//          // project the BC onto the mesh again
//          form.vlasovSolution()->projectOntoMesh(bcMap);
//          
//          bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
//          relErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
//          
//          numGlobalDofs = form.vlasovMesh()->globalDofCount();
//          numVlasovElements = form.vlasovMesh()->numActiveElements();
//          
//          if ((numVlasovElements == prevNumVlasovElements) && (numGlobalDofs == prevNumGlobalDofs))
//          {
//            if (rank == 0)
//            {
//              std::cout << "No refinements performed on this pass; exiting refinement loop early.\n";
//            }
//            break;
//          }
//          
//          prevNumVlasovElements = numVlasovElements;
//          prevNumGlobalDofs = numGlobalDofs;
//        }
//        if (rank==0)
//        {
//          std::cout << "After " << numBCRefinements << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
//          std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
//        }
//      }
//  }
////    form.vlasovMesh()->getTopology()->printAllEntitiesInBaseMeshTopology();
//    GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(form.vlasovMesh()->globalDofAssignment().get());
////    std::cout << "dof indices for cell 1024:\n";
////    minRule->getGlobalDofIndices(1024).print(std::cout);
////    
////    std::cout << "dof indices for cell 992:\n";
////    minRule->getGlobalDofIndices(992).print(std::cout);
//    
//    std::ostringstream suffix;
//    suffix << "_UR" << numUniformRefinements; // This is mainly for distinguishing between cases while we diagnose a possible bug in refinements
//    if (useAdaptivity)
//    {
//      suffix << "_budget" << maxElements;
//      suffix << "_cfl" << CFL_max;
//      if (CFL_max != CFL_for_BCs) suffix << "_cfl_bc" << CFL_for_BCs;
//    }
//    suffix << "_dt" << dt_max;
//    suffix << "_H1Order" << H1Order;
//    suffix << "_delta_k" << delta_k;
//    suffix << "_gap" << physicalGapWidth;
//    suffix << "_" << numTimeSteps << "_timeSteps";
//    if (includeElectrons)
//    {
//      suffix << "_eTherm" << electronThermalVelocity;
//    }
//    if (includeIons)
//    {
//      suffix << "_iTherm" << ionThermalVelocity;
//    }
//    if (enforcePositivity)
//    {
//      suffix << "_proximal";
//      suffix << "_alpha0" << alpha0;
//      suffix << "_mult" << alphaMultiplier;
//    }
//
//    // adjust x and v element counts according to the number of uniform refinements requested.
//    xElements <<= numUniformRefinements;
//    vElements <<= numUniformRefinements;
//
//    suffix << "_" << xElements << "x" << vElements;
//    if (useCrankNicolson)
//    {
//      suffix << "_CN";
//    }
//    suffix << "_I_IC_" << criticalCurrentRatio;
//    suffix << "_" << normChoice << "Norm";
//
//    std::string problemPrefix = "Ions";
//    
//    std::ostringstream vlasovSS;
//    vlasovSS << problemPrefix << "Vlasov" << suffix.str();
//
//    std::ostringstream poissonSS;
//    poissonSS << problemPrefix << "Poisson" << suffix.str();
//    
//    if (resultsFile == "")
//    {
//      std::ostringstream resultsFileStream;
//      resultsFileStream << outputDir << "/";
//      if (resultsFilePrefix == "")
//      {
//        resultsFileStream << "results_" << problemPrefix << "_" << suffix.str();
//      }
//      else
//      {
//        resultsFileStream << resultsFilePrefix;
//      }
//      
//      resultsFileStream << "_" << numTimeSteps << "_timeSteps";
//      resultsFileStream << ".dat";
//      resultsFile = resultsFileStream.str();
//    }
//    
//    auto getSpatialMeshFilePath = [&]() {
//      std::ostringstream spatialMeshInfoPathStringStream;
//      spatialMeshInfoPathStringStream << outputDir << "/";
//      spatialMeshInfoPathStringStream << "mesh1D_" << problemPrefix << suffix.str();
//      
//      spatialMeshInfoPathStringStream << numTimeSteps << "_timeSteps";
//      spatialMeshInfoPathStringStream << ".dat";
//      
//      return spatialMeshInfoPathStringStream.str();
//    };
//    
//    auto getTimeStepReportFilePath = [&]() {
//      std::ostringstream timeStepReportPathStringStream;
//      timeStepReportPathStringStream << outputDir << "/";
//      timeStepReportPathStringStream << "timeStepReport_" << problemPrefix << suffix.str();
//      
//      timeStepReportPathStringStream << numTimeSteps << "_timeSteps";
//      timeStepReportPathStringStream << ".dat";
//      
//      return timeStepReportPathStringStream.str();
//    };
//    
//    auto refReportFilePath = getTimeStepReportFilePath();
//    const std::string directoryPath = outputDir;
//    
//    std::ofstream refReportFileStream(refReportFilePath, std::ios::out);
//    
//    refReportFileStream << "Vlasov Elements\tVlasov Dofs\tPoisson Elements\tPoisson Dofs\tEnergy Error\n";
//    refReportFileStream.close();
//    
//    auto writeTimeStepReport = [&](const int numElementsVlasov, const int numDofsVlasov, const int numElementsPoisson, const int numDofsPoisson, const double energyError) {
//      auto refReportFilePath = getTimeStepReportFilePath();
//      std::ofstream refReportFileStream(refReportFilePath, std::ios_base::app);
//      refReportFileStream << numElementsVlasov << "\t" << numDofsVlasov << "\t" << numElementsPoisson << "\t" << numDofsPoisson << "\t" << energyError << std::endl;
//      refReportFileStream.close();
//    };
//    
//    auto reportTimeStepComplete = [&](const int &timeStepNumber, const double &energyError) {
//      auto numVlasovDofs       = form.vlasovMesh()->numGlobalDofs();
//      auto vlasovElementCount  = form.vlasovMesh()->numActiveElements();
//      auto poissonElementCount = form.configurationSpaceMesh()->numActiveElements();
//      auto poissonDofCount     = form.configurationSpaceMesh()->numGlobalDofs();
//      
//      if (rank == 0)
//      {
//        writeTimeStepReport(vlasovElementCount, numVlasovDofs, poissonElementCount, poissonDofCount, energyError);
//        std::cout << "Time step " << timeStepNumber << " complete.\n";
//      }
//    };
//    
//    if (resultsFile == "")
//    {
//      std::ostringstream resultsFileStream;
//      resultsFileStream << directoryPath << "/" << problemPrefix << "_results" << suffix.str() << ".dat";
//      resultsFile = resultsFileStream.str();
//    }
//    
//    Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz;
//    if (saveViz)
//    {
//      vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovSS.str(), directoryPath));
//      poissonViz = Teuchos::rcp(new HDF5Exporter (form.configurationSpaceMesh(), poissonSS.str(), directoryPath) );
//    }
//    
//    const int num1DPointsPoisson = 3; // default is 4
//    
//    {
//      FunctionPtr energyErrorFunction = EnergyErrorFunction::energyErrorFunction(form.vlasovSolution());
//      vlasovFunctionsToExport.push_back(energyErrorFunction);
//      vlasovFunctionsToExportNames.push_back("Vlasov energy error");
//      
//      FunctionPtr cellIDFunction = Teuchos::rcp(new CellIDFunction);
//      vlasovFunctionsToExport.push_back(cellIDFunction);
//      vlasovFunctionsToExportNames.push_back("Vlasov cell ID");
//    }
//    
//    double t = 0.0;
//    if (saveViz)
//    {
//      poissonViz->exportSolution(form.configurationSpaceSolution(),t,num1DPointsPoisson);
//      vlasovViz->exportSolution(form.vlasovSolution(),vlasovFunctionsToExport,vlasovFunctionsToExportNames,t);
//    }
//    
//    FunctionPtr E_soln = Function::solution(form.configurationSpaceFormulation()->E_fieldVar(0), form.configurationSpaceSolution());
//            
//    auto solutionMoments = form.solutionMoments();
//    auto density  = solutionMoments->solutionMomentFunction(0);
//    auto momentum = solutionMoments->solutionMomentFunction(1); // aka current
//    auto velocity = momentum / density;
//            
//    bool previousSolveUsedGeometricMultigrid = false;
//    bool previousSolveUsedDirect = false;
//    
//    bool emFieldIsOn = (t >= emFieldStartTime);
//    
////    refStrategy->setMinH(h_min);
////    refStrategy->setMaxP(p_max);
////
////    auto solnRep = RieszRep::rieszRep(form.vlasovMesh(), form.vlasovSolution()->ip(), form.vlasovSolution()->bf()->testFunctional(form.vlasovSolution()));
//    
//    double solnNorm = RieszRep::rieszRep(form.vlasovMesh(), form.vlasovSolution()->ip(), form.vlasovSolution()->bf()->testFunctional(form.vlasovSolutionPriorTime()))->getNorm();
//    if (rank == 0)
//    {
//      std::cout << "Initial solution, energy norm: " << solnNorm << std::endl;
//    }
//    
//    if (rank == 0)
//    {
//      refHistory->saveToFile("/tmp/refHistory.txt");
//    }
//    
//    numTimeSteps = 6;
//    for (int timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
//    {
//      setUseGMG(form, maxDofsForDirectSolve, previousSolveUsedGeometricMultigrid, previousSolveUsedDirect, rank == 0);
//      
//      form.startTimeStep();
//
//      // refresh voltage value, which may be ramped, and depend on the current time.
//      voltageParameter->setValue(voltageValue * timeRampFunction->value());
//      extraAccelerationParameter->setValue(extraAccelerationFormula * timeRampFunction->value());
//      
//      if (useAdaptivity)
//      {
//        int maxRefinementSteps = 1; // just do 1 pass on every time step after the first
//        int minElements = 0;   // don't gate on the element count except for the first time step
//        bool meshChanged = false; // track whether the mesh has changed since the last call to form.finishTimeStep() -- if it has, we should call again, to re-finish it.  (Time gets advanced in startTimeStep().)
//        for (int refNumber=0; refNumber<maxRefinementSteps; refNumber++)
//        {
//          form.finishTimeStep();
//          
//          maxAcceleration_vx = vxAcceleration->maximumValue(poissonConfig->configurationSpaceMesh());
//          
//          // recompute h_min
////          u_mesh = max(maxAcceleration_vx, v0 + vWidth);
////          h_min =  u_mesh * dt / CFL_max;
////          refStrategy->setMinH(h_min);
//          
//          if (performRefinements(form, refStrategy, unrefinementStrategy, maxElements) )
//          {
//            meshChanged = true;
//          }
//          
//          if (enforcePositivity)
//          {
//            if (meshChanged)
//            {
//              form.finishTimeStep();
//              meshChanged = false;
//            }
//            
//            if ( performRefinements(form, fluxLimitingRefStrategy, unrefinementStrategy, maxElements) )
//            {
//              meshChanged = true;
//            }
//          }
//          
//          setUseGMG(form, maxDofsForDirectSolve, previousSolveUsedGeometricMultigrid, previousSolveUsedDirect, rank == 0);
//          
//          if (form.vlasovMesh()->numActiveElements() >= minElements)
//          {
//            break;
//          }
//        }
//        if (meshChanged)
//        {
//          // then solve again
//          form.finishTimeStep();
//        }
//      }
//      else
//      {
//        form.finishTimeStep();
//      }
//      
//      t += dt;
//      
//      std::vector<double> f_disagreement(numSpecies);
//      for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
//      {
//        double min_f_at_extrema = fAtExtrema[speciesOrdinal]->minimumValue(form.vlasovMesh());
//        double max_f_at_extrema = fAtExtrema[speciesOrdinal]->maximumValue(form.vlasovMesh());
//        f_disagreement[speciesOrdinal] = std::max(std::abs(min_f_at_extrema), std::abs(max_f_at_extrema));
//      }
//      
//      solnNorm = RieszRep::rieszRep(form.vlasovMesh(), form.vlasovSolution()->ip(), form.vlasovSolution()->bf()->testFunctional(form.vlasovSolutionPriorTime()))->getNorm();
//      double totalEnergyError = refStrategy->computeTotalEnergyError();
//      if (rank == 0)
//      {
//        double relErr = (solnNorm > 0) ? totalEnergyError / solnNorm : 1.0;
//        std::cout << "Solution's energy norm: " << solnNorm << std::endl;
//        std::cout << "Relative energy error: " << relErr << std::endl;
//        for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
//        {
//          std::cout << "Species " << speciesOrdinal << ", max disagreement with homogeneous BCs at velocity extrema: " << f_disagreement[speciesOrdinal] << std::endl;
//        }
//      }
//      
//      if (rank == 0)
//      {
//        std::cout << "Max acceleration after " << timeStepNumber << " time steps is " << maxAcceleration_vx << std::endl;
////        std::cout << "h_min: " << h_min << std::endl;
//      }
//      
//      if (!emFieldIsOn)
//      {
//        // check if it should turn on now
//        emFieldIsOn = (t >= emFieldStartTime);
//        if (emFieldIsOn)
//        {
//          poissonConfig->solution()->setRHS(poissonRHS);
//        }
//      }
//      
//      {
//        // compute, report BC error for inflow:
//        const double relErr = inflowErrorOnBoundary(form, f_in, form.tagVlasovSpatialInflow(), p_max);
//        if (rank == 0)
//        {
//          std::cout << "Relative L^2 error in f compared to f_in at inflow: " << relErr << std::endl;
//        }
//      }
//      
//      moments_kp1->update();
//      if (rank == 0) std::cout << "moments_kp1 updated.\n";
//      moments_k->update();
//      if (rank == 0) std::cout << "moments_k   updated.\n";
//      
//      double energyError = -1.;
//      if ((refStrategy != Teuchos::null) && (refStrategy->getNumRefinements() > 0))
//      {
//        energyError = refStrategy->getEnergyError(refStrategy->getNumRefinements() - 1);
//      }
//      
//      if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
//      {
//        if (saveViz)
//        {
//          solutionMoments->update(); // update to get latest density, velocity
//          vlasovViz->exportSolution(form.vlasovSolution(),vlasovFunctionsToExport,vlasovFunctionsToExportNames,t);
//          poissonViz->exportSolution(form.configurationSpaceSolution(), {density,velocity}, {"n_e","v"},t,num1DPointsPoisson);
//          
////          FunctionPtr E_x = Function::solution(poissonConfig->E_fieldVar(0), form.configurationSpaceSolution());
////          poissonViz->exportFunction({"E_x"}, {E_x});
//        }
//        
//        reportTimeStepComplete(timeStepNumber,energyError);
//      }
//    } // end time step loop
//    
    if (rank == 0)
    {
      refHistory->saveToFile("/tmp/refHistory.txt");
    }
    
    if (rank == 0)
    {
      std::cout << "Performing artificial refinements to capture bug.\n";
    }
    
    auto cellTopo = CellTopology::lineTensorTopology(CellTopology::line()); // line, once extruded
    auto isotropicRefinement = RefinementPattern::anisotropicRefinementPattern(cellTopo, {1,1});
    auto         xRefinement = RefinementPattern::xAnisotropicRefinementPatternExtrudedLine();
    
    std::vector<std::pair< std::set<int>, RefinementPatternPtr> > bcRefs, refs;
    {
//      bcRefs.push_back({std::set<int>{ 106  }, isotropicRefinement});
//      bcRefs.push_back({std::set<int>{ 84, 170  }, isotropicRefinement});
//      bcRefs.push_back({std::set<int>{ 176, 178  }, isotropicRefinement});
//      bcRefs.push_back({std::set<int>{ 184, 186  }, isotropicRefinement});
//      bcRefs.push_back({std::set<int>{ 108, 172, 180, 188, 192, 194  }, isotropicRefinement});
//      bcRefs.push_back({std::set<int>{ 196  }, isotropicRefinement});
//      bcRefs.push_back({std::set<int>{ 210  }, isotropicRefinement});
//      bcRefs.push_back({std::set<int>{ 222  }, isotropicRefinement});
//      bcRefs.push_back({std::set<int>{ 216, 218, 220  }, isotropicRefinement});
//      bcRefs.push_back({std::set<int>{ 206, 212, 224  }, isotropicRefinement});
//      bcRefs.push_back({std::set<int>{ 226  }, isotropicRefinement});
//      bcRefs.push_back({std::set<int>{ 228  }, isotropicRefinement});
//      bcRefs.push_back({std::set<int>{ 230, 240, 242  }, isotropicRefinement});
//      
      
//      refs.push_back({std::set<int>{ 107, 171, 179, 187, 195, 197, 219, 221, 223, 231, 232, 236, 238, 241, 243, 244, 254, 256, 266, 268, 269, 270, 272, 273, 274, 275, 276, 277  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 245, 318, 324, 338, 340, 341, 350, 356, 383, 384  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 225, 233, 255, 258, 267, 271, 319, 320, 321, 326, 328, 339, 342, 343, 346, 348, 351, 352, 353, 354, 355, 357, 362, 363, 364, 365, 366, 367, 368, 369, 374, 375, 376, 377, 382, 385, 394, 395, 396, 397, 400, 402, 403, 404, 405, 406, 407, 408, 409, 414, 415, 416, 417, 418, 419, 420, 421, 422, 424, 426, 427, 428, 429  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 82, 174, 182, 190, 214, 234, 239, 250, 262, 264, 316, 322, 329, 344, 358, 386, 398, 423, 434, 436, 438, 442, 446, 455, 457, 458, 460, 462, 463, 466, 471, 472, 473, 483, 486, 488, 490, 492, 495, 498, 499, 500, 501, 508, 509, 511, 516, 518, 519, 521, 526, 559, 561, 567, 568, 572, 574, 575, 577, 583, 584, 585, 590, 594, 595, 604, 605, 610, 612, 613, 618, 619, 620, 621, 626, 628, 634, 636, 637, 642, 643, 644, 646, 650, 652, 653, 655, 657, 660, 666, 667, 668, 669, 674, 675, 676, 677, 679, 681  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 257, 260, 327, 345, 347, 349, 370, 372, 444, 459, 461, 465, 467, 468, 469, 470, 478, 479, 480, 481, 484, 487, 493, 494, 496, 502, 506, 507, 510, 512, 514, 520, 527, 528, 529, 534, 535, 536, 537, 538, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 556, 558, 560, 562, 564, 566, 569, 570, 576, 578, 580, 582, 586, 588, 592, 596, 597, 598, 600, 602, 603, 606, 608, 611, 614, 616, 622, 624, 627, 629, 630, 631, 635, 645, 647, 648, 649, 651, 654, 656, 662, 664, 670, 671, 672, 673, 678, 680, 730, 734, 735, 736, 766, 768, 782, 783, 784, 785, 786, 787, 788, 789, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 810, 818, 819, 820, 822, 824, 826, 828, 830, 831, 832, 833, 838, 839, 840, 841, 846, 847, 848, 849, 854, 855, 856, 857, 858, 859, 860, 861, 866, 868, 870, 872, 882, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 900, 901, 902, 906, 907, 908, 909, 910, 911, 912, 914, 915, 918, 919, 922, 923, 924, 925, 926, 927, 928, 934, 935, 936, 937, 938, 939, 940, 941, 942, 944, 946, 947, 948, 949, 950, 952, 954, 956, 959, 961, 962, 964, 967, 970, 972, 975  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 978, 979, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 998, 999, 1000, 1001, 1002, 1004, 1005, 1006, 1007, 1008, 1009, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1071, 1073  }, xRefinement});
//      refs.push_back({std::set<int>{ 85, 177, 185, 189, 193, 211, 217, 227, 237, 259, 314, 325, 360, 401, 425, 440, 443, 456, 482, 517, 532, 539, 589, 591, 593, 659, 663, 665, 722, 728, 746, 748, 758, 760, 762, 797, 867, 930, 932, 1080, 1086, 1088, 1090, 1099, 1126, 1127, 1137, 1141, 1158, 1160, 1204, 1210, 1235, 1237, 1326, 1354, 1355, 1356, 1357, 1490, 1546, 1566, 1567, 1569  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 1458, 1459, 1460, 1461, 1463, 1465, 1638, 1718, 1720, 1722, 1724, 2002  }, xRefinement});
//      refs.push_back({std::set<int>{ 809, 871, 904, 913, 916, 917, 920, 943, 945, 951, 953, 955, 957, 958, 960, 969, 1110, 1114, 1128, 1130, 1131, 1132, 1133, 1134, 1136, 1138, 1139, 1140, 1142, 1143, 1144, 1150, 1152, 1182, 1184, 1202, 1212, 1213, 1218, 1219, 1220, 1221, 1222, 1224, 1226, 1227, 1228, 1229, 1230, 1232, 1233, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1318, 1319, 1320, 1321, 1322, 1324, 1325, 1328, 1330, 1331, 1332, 1333, 1342, 1343, 1344, 1345, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1368, 1370, 1372, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1384, 1386, 1388, 1390, 1391, 1392, 1393, 1394, 1396, 1494, 1496, 1502, 1504, 1506, 1508, 1510, 1512, 1514, 1515, 1516, 1518, 1522, 1524, 1530, 1531, 1532, 1533, 1538, 1539, 1540, 1541, 1547, 1548, 1549, 1554, 1555, 1556, 1557, 1561, 1578, 1579, 1580, 1581, 1582, 1583, 1586, 1587, 1588, 1589, 1594, 1596, 1602, 1604, 1606, 1608, 1610, 1612, 1614, 1616, 1618, 1620, 1622, 1624, 1630, 1632, 2248, 2270, 2272, 2274, 2276, 2311, 2313, 2342, 2344, 2350, 2352, 2353  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 1036, 1462, 1639, 1640, 1641, 1646, 1647, 1648, 1649, 1654, 1655, 1656, 1657, 1662, 1663, 1664, 1665, 1670, 1671, 1672, 1673, 1678, 1679, 1680, 1681, 1710, 1712, 1714, 1716, 1778, 1779, 1780, 1781, 1786, 1787, 1788, 1789, 1798, 1799, 1800, 1801, 1806, 1807, 1808, 1809, 1842, 1843, 1844, 1845, 1850, 1851, 1852, 1853, 1856, 1870, 1871, 1872, 1873, 1874, 1876, 1878, 1879, 1880, 1881, 1882, 1884, 1886, 1888, 1894, 1896, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1974, 1975, 1976, 1978, 1979, 1980, 1982, 1983, 1984, 1986, 1987, 1988, 1990, 1991, 1992, 1994, 1995, 1996, 1998, 1999, 2000, 2003, 2004, 2008, 2009, 2010, 2012, 2013, 2014, 2022, 2023, 2024, 2026, 2027, 2028, 2030, 2031, 2032, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2054, 2057, 2058, 2061, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2322, 2324, 2326, 2328, 2362, 2363, 2364, 2365, 2378, 2383, 2390, 2391, 2392, 2394, 2396, 2397, 2398, 2399, 2400, 2401  }, xRefinement});
//      refs.push_back({std::set<int>{ 229, 246, 252, 261, 263, 310, 312, 317, 332, 361, 371, 380, 388, 399, 435, 437, 452, 464, 515, 565, 571, 661, 706, 710, 712, 715, 718, 720, 723, 725, 749, 750, 754, 755, 756, 761, 764, 775, 793, 794, 796, 814, 933, 1078, 1083, 1085, 1087, 1091, 1092, 1103, 1135, 1155, 1157, 1201, 1216, 1317, 1327, 1550, 1551, 1553, 1559, 1584, 2158, 2160, 2164, 2171, 2174, 2176, 2196, 2197, 2215, 2217, 2218, 2220, 2235, 2240, 2242, 2244, 2246, 2247, 2249, 2250, 2260, 2282, 2283, 2288, 2290, 2292, 2293, 2298, 2300, 2312, 2314, 2316, 2318, 2320, 2346, 2347, 2348, 2351, 2355, 2356, 2357, 2358, 2487, 2499, 2513, 3178, 3179, 3180  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 1026, 1027, 1033, 1037, 1075, 1439, 1444, 1464, 1481, 1823, 2081, 2085, 2119, 2120, 2121, 2222, 2226, 2228, 2229, 2230, 2233, 2266, 2339, 2379, 2381, 2384, 2388, 2418, 2420, 2423, 2425, 2426, 2428, 2574, 2791, 2796, 2850, 2854, 2856, 3042, 3043, 3045, 3071, 3072, 3091, 3093, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3200, 3202, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3214, 3216, 3218, 3220, 3221, 3222, 3223, 3225, 3614  }, xRefinement});
//      refs.push_back({std::set<int>{ 489, 491, 737, 821, 823, 825, 827, 873, 966, 968, 977, 1082, 1112, 1145, 1151, 1153, 1183, 1185, 1203, 1205, 1398, 1495, 1497, 1503, 1505, 1517, 1523, 1525, 1595, 1597, 1603, 1605, 1607, 1609, 1611, 1613, 1615, 2470, 2472, 2482, 2483, 2484, 2485, 2490, 2491, 2492, 2493, 2506, 2507, 2508, 2509, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536, 2537, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2972, 2974, 2976, 2978, 2980, 2982, 2984  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 1782, 1784, 1802, 1804, 1810, 1812, 2011, 2015, 2025, 2029, 2033, 2055, 2056, 2059, 2060, 2434, 2436, 2438, 2440, 2442, 2444, 2446, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2548, 2550, 2552, 2558, 2560, 2566, 2568, 2582, 2584, 2590, 2591, 2592, 2593, 2610, 2611, 2612, 2613, 2618, 2619, 2620, 2621, 2624, 2642, 2643, 2644, 2645, 2646, 2648, 2650, 2651, 2652, 2653, 2654, 2656, 2657, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2802, 2803, 2804, 2805, 2806, 2808, 2810, 2811, 2812, 2813, 2814, 2816, 2818, 2819, 2820, 2821, 2822, 2824, 2826, 2827, 2828, 2829, 2830, 2834, 2835, 2836, 2837, 2838, 2840, 2842, 2843, 2844, 2845, 2846, 2848, 2882, 2883, 2884, 2885, 2886, 2888, 2890, 2891, 2892, 2893, 2894, 2896, 2986, 2987, 2988, 2989, 2994, 2995, 2996, 2997, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3018, 3019, 3020, 3021, 3026, 3027, 3028, 3029, 3034, 3035, 3036, 3037, 3044, 3046, 3047, 3048, 3049, 3050, 3052, 3054, 3055, 3056, 3057, 3058, 3060, 3062, 3063, 3064, 3065, 3066, 3068, 3074, 3075, 3076, 3077, 3082, 3083, 3084, 3085, 3098, 3100, 3106, 3108, 3114, 3115, 3116, 3118, 3120, 3122, 3124, 3126, 3128, 3130, 3132, 3134, 3136, 3138, 3140, 3142, 3144, 3146, 3148, 3150, 3152, 3154, 3156, 3158, 3160, 3162, 3164, 3166, 3168, 3170, 3172, 3174, 3176, 3230, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3278, 3280, 3282, 3284, 3285, 3286, 3288, 3289, 3290, 3292, 3293, 3294, 3296, 3297, 3298, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3366, 3367, 3368, 3370, 3371, 3372, 3374, 3375, 3376, 3378, 3379, 3380, 3384, 3385, 3386, 3388, 3389, 3390, 3392, 3393, 3394, 3396, 3397, 3398, 3400, 3401, 3402, 3404, 3405, 3406, 3408, 3409, 3410, 3412, 3413, 3414, 3416, 3417, 3418, 3420, 3421, 3422, 3424, 3425, 3426, 3428, 3429, 3430, 3432, 3433, 3434, 3436, 3437, 3438, 3440, 3441, 3442, 3444, 3445, 3446, 3448, 3449, 3450, 3451, 3452, 3454, 3455, 3456, 3457, 3458, 3460, 3461, 3462, 3463, 3464, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3530, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3540, 3541, 3542, 3543, 3544, 3545, 3546, 3548, 3549, 3552, 3553, 3556, 3557, 3558, 3559, 3560, 3564, 3565, 3566, 3567, 3568, 3572, 3573, 3574, 3575, 3576, 3580, 3581, 3582, 3583, 3584, 3588, 3589, 3590, 3591, 3592, 3596, 3597, 3598, 3599, 3600, 3604, 3606, 3608, 3609, 3610, 3611, 3612, 3613, 3616, 3617, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3636, 3637, 3639, 3640, 3641, 3642, 3643  }, xRefinement});
//      refs.push_back({std::set<int>{ 265, 315, 323, 392, 430, 439, 445, 449, 485, 704, 717, 719, 724, 726, 729, 732, 747, 757, 759, 763, 777, 779, 781, 791, 792, 795, 815, 816, 817, 869, 1093, 1122, 1123, 1124, 1125, 1149, 1159, 1492, 1499, 1552, 1568, 1575, 2150, 2161, 2162, 2165, 2168, 2175, 2177, 2188, 2194, 2195, 2214, 2237, 2238, 2241, 2243, 2245, 2251, 2252, 2254, 2258, 2261, 2262, 2263, 2265, 2284, 2285, 2291, 2308, 2309, 2315, 2349, 2354, 3181, 3652, 3656, 3660, 3670, 3672, 3674, 3698, 3700, 3705, 3713, 3724, 3736, 3738, 3745, 3747, 3752, 3754, 3756, 3757, 3759, 3760, 3761, 3762, 3767, 3773, 3776, 3777, 3778, 3796, 3797, 3798, 3804, 3805, 3806, 3807, 3808, 3809, 3810, 3811, 3828, 3834, 3896, 3898, 3899, 3900, 3902, 3903, 3908, 3911, 3912, 3913, 3914, 3915, 3917, 3918, 3919, 3921, 3922, 3924, 3925, 3926, 3927, 3941, 3943, 3944, 3945, 3946, 3947, 3948, 3949, 3950, 3951, 3952, 3954, 3955, 3956, 3957, 3958, 3959, 3971, 3972, 3973, 3976, 3977, 3979, 3980, 3981, 3982, 3983, 3986, 3988, 3993, 3996, 3997, 4358  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 1028, 1031, 1034, 1035, 1072, 1077, 1479, 1488, 1819, 1820, 1830, 1832, 1838, 1900, 1912, 2069, 2078, 2231, 2268, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2380, 2382, 2385, 2386, 2576, 2852, 2862, 2908, 3051, 3070, 3213, 3219, 3224, 3226, 3227, 3229, 3358, 3359, 3382, 3383, 3563, 3571, 3618, 3620, 3621, 3622, 3623, 3858, 3862, 3869, 3872, 3874, 3876, 3878, 4011, 4012, 4014, 4016, 4018, 4026, 4027, 4042, 4043, 4044, 4046, 4048, 4049, 4050, 4052, 4054, 4055, 4057, 4060, 4070, 4072, 4073, 4074, 4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4085, 4089, 4091, 4093, 4101, 4105, 4108, 4109, 4112, 4119, 4125, 4126, 4130, 4132, 4134, 4138, 4139, 4140, 4141, 4146, 4147, 4148, 4149, 4150, 4151, 4156, 4157, 4158, 4159, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4171, 4172, 4173, 4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 4196, 4197, 4198, 4199, 4200, 4201, 4202, 4203, 4204, 4207, 4208, 4209, 4210, 4214, 4216, 4217, 4218, 4220, 4221, 4222, 4224, 4225, 4226, 4228, 4230, 4232, 4233, 4234, 4235, 4236, 4238, 4241, 4242, 4243, 4244, 4272, 4276, 4278, 4279, 4280, 4281, 4282, 4284, 5040, 5041, 5042, 5043, 5146, 5154, 5155, 5156, 5157, 5166, 5167, 5169, 5179, 5883  }, xRefinement});
//      refs.push_back({std::set<int>{ 504, 767, 829, 883, 1211, 1507, 1617, 1619, 1621, 1623, 1625, 1631, 1633, 2971, 4362, 4364, 4462, 4464, 4466  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 1430, 1432, 1446, 1448, 1674, 1676, 1682, 1684, 1846, 1848, 1854, 1977, 1981, 1985, 1989, 1993, 1997, 2001, 2005, 2016, 2018, 2046, 2050, 2448, 2832, 2990, 2992, 3014, 3016, 3022, 3024, 3030, 3032, 3038, 3040, 3117, 3231, 3332, 3333, 3381, 3387, 3391, 3395, 3399, 3403, 3407, 3411, 3415, 3419, 3423, 3427, 3431, 3435, 3439, 3443, 3447, 3453, 3459, 3465, 3531, 3539, 3547, 4382, 4384, 4414, 4416, 4418, 4420, 4422, 4424, 4426, 4428, 4430, 4470, 4472, 4478, 4480, 4481, 4486, 4487, 4488, 4489, 4490, 4492, 4494, 4495, 4496, 4497, 4498, 4500, 4502, 4503, 4504, 4505, 4506, 4508, 4510, 4511, 4512, 4513, 4514, 4518, 4519, 4520, 4521, 4526, 4527, 4528, 4529, 4534, 4535, 4536, 4537, 4538, 4540, 4542, 4543, 4544, 4545, 4546, 4548, 4550, 4551, 4552, 4553, 4554, 4556, 4558, 4559, 4560, 4561, 4562, 4564, 4566, 4567, 4568, 4569, 4570, 4572, 4574, 4575, 4576, 4577, 4578, 4580, 4582, 4583, 4584, 4585, 4586, 4588, 4590, 4591, 4592, 4593, 4594, 4596, 4598, 4599, 4600, 4601, 4602, 4604, 4606, 4607, 4608, 4609, 4610, 4612, 4614, 4615, 4616, 4617, 4618, 4620, 4622, 4623, 4624, 4625, 4626, 4628, 4630, 4631, 4632, 4633, 4634, 4636, 4638, 4639, 4640, 4641, 4642, 4644, 4646, 4647, 4648, 4649, 4650, 4652, 4654, 4655, 4656, 4657, 4658, 4660, 4662, 4664, 4666, 4668, 4670, 4672, 4674, 4676, 4678, 4680, 4682, 4684, 4686, 4688, 4690, 4692, 4698, 4700, 4702, 4704, 4706, 4707, 4708, 4709, 4710, 4711, 4712, 4713, 4714, 4715, 4716, 4717, 4718, 4720, 4721, 4722, 4738, 4739, 4740, 4741, 4742, 4743, 4744, 4745, 4746, 4747, 4748, 4749, 4750, 4751, 4752, 4753, 4754, 4756, 4758, 4760, 4762, 4764, 4766, 4768, 4770, 4772, 4774, 4778, 4782, 4786, 4787, 4790, 4791, 4794, 4795, 4800, 4801, 4804, 4805, 4812, 4813, 4816, 4817, 4826, 4827, 4830, 4831, 4842, 4843, 4846, 4847, 4858, 4859, 4862, 4863, 4874, 4875, 4878, 4879, 4890, 4891, 4894, 4895, 4906, 4907, 4910, 4911, 4912, 4922, 4923, 4924, 4926, 4927, 4928, 4938, 4939, 4940, 4942, 4943, 4944, 4954, 4955, 4956, 4958, 4959, 4960, 4970, 4971, 4972, 4974, 4975, 4976, 4986, 4987, 4988, 4989, 4990, 4991, 4992, 4993, 4994, 4996, 4998, 4999, 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5008, 5010, 5011, 5012, 5013, 5014, 5015, 5016, 5017, 5018, 5020, 5022, 5023, 5024, 5025, 5026, 5027, 5028, 5029, 5030, 5031, 5032, 5033, 5034, 5036, 5037, 5038, 5044, 5045, 5046, 5048, 5049, 5050, 5056, 5057, 5058, 5060, 5061, 5062, 5068, 5069, 5070, 5072, 5073, 5074, 5080, 5081, 5082, 5083, 5084, 5085, 5086, 5087, 5088, 5089, 5090, 5091, 5092, 5093, 5094, 5095, 5096, 5097, 5098, 5099, 5100, 5101, 5102, 5103, 5104, 5105, 5106, 5107, 5108, 5109, 5110, 5111, 5112, 5113, 5114, 5115, 5116, 5117, 5118, 5119, 5120, 5121, 5122, 5123, 5124, 5125, 5126, 5127, 5128, 5129, 5130, 5131, 5132, 5133, 5134, 5135, 5136, 5137, 5138, 5139, 5140, 5141, 5142, 5143, 5150, 5158, 5162, 5170, 5174, 5175, 5182, 5183, 5186, 5187, 5190, 5191, 5194, 5195, 5198, 5199, 5200, 5201, 5202, 5204, 5206, 5210, 5212, 5214, 5216, 5218, 5220, 5222, 5224, 5226, 5228, 5230, 5232, 5234, 5236, 5238, 5240, 5242, 5244, 5246, 5248, 5250, 5252, 5254, 5256, 5258, 5260, 5262, 5264, 5266, 5268, 5270, 5272, 5273, 5274, 5275, 5276, 5277, 5278, 5279, 5280, 5282, 5283, 5284, 5285, 5286, 5287, 5288, 5290, 5291, 5292, 5293, 5294, 5295, 5296, 5298, 5299, 5300, 5301, 5302, 5303, 5304, 5306, 5307, 5308, 5309, 5310, 5311, 5312, 5314, 5315, 5316, 5317, 5318, 5319, 5320, 5321, 5322, 5323, 5324, 5325, 5326, 5327, 5328, 5329, 5330, 5331, 5332, 5333, 5334, 5335, 5336, 5337, 5338, 5339, 5340, 5341, 5342, 5343, 5344, 5345, 5346, 5347, 5348, 5349, 5350, 5351, 5352, 5353, 5354, 5355, 5356, 5357, 5358, 5359, 5360, 5361, 5362, 5363, 5364, 5365, 5366, 5367, 5368, 5369, 5370, 5371, 5372, 5373, 5374, 5376, 5377, 5378, 5379, 5380, 5382, 5383, 5384, 5385, 5386, 5388, 5389, 5390, 5391, 5392, 5394, 5395, 5396, 5397, 5398, 5399, 5400, 5401, 5402, 5403, 5404, 5405, 5406, 5407, 5408, 5409, 5410, 5411, 5412, 5413, 5414, 5415, 5416, 5417, 5418, 5419, 5420, 5421, 5422, 5423, 5424, 5425, 5426, 5427, 5428, 5429, 5430, 5431, 5432, 5433, 5434, 5435, 5436, 5437, 5438, 5439, 5440, 5441, 5442, 5443, 5444, 5445, 5446, 5447, 5448, 5449, 5450, 5451, 5452, 5453, 5454, 5455, 5456, 5457, 5458, 5459, 5460, 5461, 5462, 5466, 5467, 5468, 5469, 5470, 5474, 5475, 5476, 5477, 5478, 5482, 5483, 5484, 5485, 5486, 5490, 5491, 5492, 5493, 5494, 5496, 5497, 5498, 5499, 5500, 5502, 5503, 5504, 5505, 5506, 5508, 5509, 5510, 5511, 5512, 5514, 5515, 5516, 5517, 5518, 5519, 5520, 5521, 5522, 5523, 5524, 5525, 5526, 5527, 5528, 5529, 5530, 5531, 5532, 5533, 5534, 5535, 5536, 5537, 5538, 5539, 5540, 5541, 5542, 5543, 5544, 5545, 5546, 5547, 5548, 5549, 5550, 5551, 5552, 5553, 5554, 5555, 5556, 5557, 5558, 5559, 5560, 5561, 5562, 5563, 5564, 5565, 5566, 5567, 5568, 5569, 5570, 5571, 5572, 5573, 5574, 5575, 5576, 5577, 5578, 5579, 5580, 5581, 5582, 5583, 5584, 5585, 5586, 5587, 5588, 5589, 5590, 5591, 5592, 5593, 5594, 5595, 5596, 5597, 5598, 5599, 5600, 5601, 5602, 5603, 5604, 5605, 5606, 5607, 5608, 5609, 5610, 5611, 5612, 5613, 5614, 5615, 5616, 5617, 5618, 5619, 5620, 5621, 5622, 5623, 5624, 5625, 5626, 5627, 5628, 5629, 5630, 5631, 5632, 5633, 5634, 5635, 5636, 5637, 5638, 5639, 5640, 5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5650, 5651, 5652, 5653, 5654, 5655, 5656, 5657, 5658, 5659, 5660, 5661, 5662, 5663, 5664, 5665, 5666, 5667, 5668, 5669, 5670, 5671, 5672, 5673, 5674, 5675, 5676, 5677, 5678, 5679, 5680, 5681, 5682, 5683, 5684, 5685, 5686, 5687, 5688, 5689, 5690, 5691, 5692, 5693, 5694, 5695, 5696, 5697, 5698, 5699, 5700, 5701, 5702, 5703, 5704, 5705, 5706, 5707, 5708, 5709, 5710, 5711, 5712, 5713, 5714, 5715, 5716, 5717, 5718, 5719, 5720, 5721, 5722, 5723, 5724, 5725, 5726, 5727, 5728, 5729, 5730, 5731, 5732, 5733, 5734, 5735, 5736, 5737, 5738, 5739, 5740, 5741, 5742, 5743, 5744, 5745, 5746, 5747, 5748, 5749, 5750, 5751, 5752, 5753, 5754, 5755, 5756, 5757, 5758, 5759, 5760, 5761, 5762, 5763, 5764, 5765, 5766, 5767, 5768, 5770, 5771, 5772, 5773, 5774, 5775, 5776, 5777, 5778, 5779, 5780, 5781, 5782, 5784, 5785, 5786, 5787, 5788, 5789, 5790, 5791, 5792, 5793, 5794, 5795, 5796, 5798, 5799, 5800, 5801, 5802, 5803, 5804, 5805, 5806, 5807, 5808, 5809, 5812, 5813, 5816, 5817, 5818, 5819, 5822, 5823, 5826, 5827, 5828, 5829, 5830, 5833, 5836, 5837, 5838, 5839, 5840, 5843, 5846, 5847, 5848, 5849, 5850, 5853, 5856, 5857, 5858, 5859, 5860, 5861, 5863, 5866, 5867, 5868, 5869, 5870, 5871, 5874, 5875, 5878, 5879, 5880, 5881, 5882, 5886, 5887, 5888, 5889, 5890, 5891, 5892, 5893, 5894, 5895, 5898, 5899, 5902, 5903, 5906, 5907, 5912, 5913, 5914, 5915, 5918, 5919, 7006  }, xRefinement});
//      refs.push_back({std::set<int>{ 213, 251, 253, 441, 711, 713, 716, 721, 752, 778, 1089, 1214, 1329, 1501, 2152, 2159, 2219, 2239, 2256, 2259, 2264, 2511, 3653, 3657, 3658, 3662, 3668, 3696, 3701, 3702, 3715, 3737, 3739, 3740, 3741, 3743, 3748, 3750, 3758, 3763, 3764, 3766, 3772, 3774, 3775, 3779, 3780, 3782, 3788, 3832, 3836, 3837, 3897, 3901, 3910, 3953, 3960, 3961, 3962, 3969, 3974, 3975, 3989, 3990, 3994, 3999, 5920, 5926, 5940, 5942, 5947, 5953, 5961, 5963, 5965, 5967, 5969, 5971, 5986, 5988, 6003, 6004, 6005, 6007, 6020, 6021, 6022, 6023, 6025, 6027, 6033, 6034, 6035, 6048, 6049, 6051, 6054, 6092, 6102, 6103, 6112, 6114, 6123, 6127, 6138, 6140, 6142, 6149, 6164, 6165, 6166, 6167, 6195, 6220, 6221, 6224, 6225, 6227, 6230, 6236, 6238, 6240, 6244, 6246, 6250, 6251, 6264, 6266, 6268, 6270, 6280, 6282, 6284, 6286, 6288, 6289, 6290, 6291, 6292, 6293, 6294, 6295, 6297, 6300, 6301, 6302, 6303, 6304, 6306, 6307, 6314, 6317, 6318, 6319, 6320, 6324, 6326, 6330, 6376, 6390, 6392, 6394, 6404, 6406, 6407, 6410, 6414, 6417, 6418, 6423, 6427, 6428, 6430, 6431, 6482, 6483, 6484, 6485, 6486, 6487, 6489, 6491, 7091, 7094  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 1486, 1793, 1818, 1898, 1918, 2065, 2066, 2082, 2097, 2338, 2341, 2578, 2858, 2864, 2906, 3053, 3215, 3217, 3228, 3356, 3360, 3361, 3364, 3365, 3860, 3871, 4020, 4024, 4040, 4041, 4051, 4068, 4099, 4107, 4128, 4131, 4135, 4137, 4160, 4205, 4206, 4211, 4227, 4237, 4240, 4245, 4248, 4270, 4274, 4474, 5052, 5144, 5145, 5148, 5168, 5181, 6172, 6173, 6174, 6175, 6176, 6177, 6178, 6179, 6214, 6215, 6217, 6218, 6219, 6260, 6351, 6360, 6432, 6438, 6443, 6444, 6445, 6453, 6455, 6468, 6498, 6502, 6516, 6518, 6524, 6525, 6526, 6530, 6531, 6540, 6541, 6542, 6543, 6544, 6552, 6554, 6557, 6560, 6561, 6564, 6565, 6567, 6580, 6582, 6592, 6593, 6594, 6595, 6604, 6605, 6614, 6615, 6619, 6621, 6626, 6627, 6628, 6629, 6630, 6631, 6632, 6635, 6653, 6665, 6670, 6671, 6672, 6673, 6675, 6676, 6677, 6678, 6679, 6683, 6684, 6685, 6686, 6689, 6690, 6693, 6694, 6696, 6699, 6701, 6706, 6710, 6711, 6714, 6715, 6718, 6719, 6721, 6722, 6723, 6726, 6727, 6732, 6733, 6734, 6735, 6736, 6737, 6739, 6740, 6741, 6744, 6747, 6753, 6754, 6755, 6756, 6757, 6760, 6761, 6768, 6769, 6770, 6771, 6772, 6773, 6774, 6775, 6776, 6777, 6786, 6788, 6790, 6791, 6792, 6804, 6815, 6818, 6819, 6826, 6830, 6840, 6841, 6842, 6845, 6846, 6848, 6851, 6852, 6853, 6854, 6855, 6859, 6862, 6863, 6865, 6872, 6873, 6874, 6876, 6877, 6880, 6881, 6882, 6884, 6885, 6886, 6888, 6889, 6890, 6891, 6892, 6893, 6894, 6895, 6896, 6897, 6898, 6899, 6900, 6901, 6902, 6903, 6904, 6905, 6906, 6907, 6908, 6909, 6910, 6912, 6913, 6915, 6916, 6917, 6920, 6921, 6923, 6924, 6925, 6928, 6933, 6934, 6935, 6938, 6941, 6944, 6950, 6956, 6958, 6959, 6960, 6961, 6962, 6966, 6967, 6970, 6971, 6972, 6973, 6974, 6976, 6977, 6978, 6979, 6980, 6982, 6984, 6987, 6989, 6991, 6998, 6999, 7002, 7003, 7008, 7009, 7010, 7011, 7012, 7013, 7014, 7015, 7016, 7017, 7018, 7019, 7020, 7021, 7022, 7023, 7040, 7914, 7915, 8503, 8504, 9311  }, xRefinement});
//      refs.push_back({std::set<int>{ 769, 1084, 1111, 1509, 1511, 1513, 1598, 1600, 1626, 1628, 1634, 1636, 2471, 2973, 2975, 4326, 4328, 4330, 4332, 4334, 4336, 4338, 4340, 4360, 4410, 4412, 4468, 7106, 7108, 7118  }, isotropicRefinement});
//      refs.push_back({std::set<int>{ 996, 1410, 1412, 1414, 1416, 1426, 1428, 1642, 1644, 1650, 1652, 1658, 1660, 1666, 1668, 1857, 2006, 2047, 2051, 2860, 2991, 2993, 3015, 3017, 3023, 3025, 3031, 3033, 3039, 3041, 3119, 3362, 3363, 3369, 3373, 3377, 4256, 4258, 4260, 4262, 4264, 4378, 4380, 4415, 4417, 4419, 4421, 4423, 4425, 4427, 4429, 4431, 4433, 4435, 4437, 4439, 4441, 4539, 4541, 4547, 4549, 4555, 4557, 4563, 4565, 4571, 4573, 4579, 4581, 4587, 4589, 4595, 4597, 4603, 4605, 4611, 4613, 4619, 4621, 4627, 4629, 4635, 4637, 4643, 4645, 4651, 4653, 4659, 4661, 4699, 4701, 4703, 4705, 4779, 4783, 4995, 4997, 5007, 5009, 5019, 5021, 5039, 5047, 5051, 5059, 5063, 5071, 5075, 5281, 5289, 5297, 5305, 5313, 5463, 5471, 5479, 5487, 5495, 5501, 5507, 5513, 5810, 5811, 5820, 5821, 5824, 5831, 5832, 5834, 5841, 5842, 5844, 5851, 5852, 5854, 5862, 5864, 5916, 5917, 7162, 7164, 7166, 7168, 7170, 7172, 7174, 7176, 7178, 7180, 7181, 7182, 7184, 7186, 7188, 7190, 7191, 7192, 7193, 7194, 7195, 7196, 7198, 7199, 7200, 7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7210, 7211, 7212, 7213, 7214, 7215, 7216, 7217, 7218, 7219, 7222, 7223, 7224, 7225, 7226, 7227, 7228, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7236, 7237, 7238, 7239, 7240, 7241, 7242, 7243, 7246, 7247, 7248, 7249, 7250, 7251, 7264, 7266, 7268, 7270, 7272, 7274, 7276, 7278, 7280, 7282, 7284, 7286, 7287, 7288, 7289, 7290, 7291, 7298, 7300, 7302, 7303, 7304, 7305, 7306, 7307, 7308, 7309, 7310, 7311, 7312, 7313, 7320, 7322, 7324, 7326, 7330, 7334, 7342, 7346, 7354, 7358, 7366, 7370, 7376, 7380, 7384, 7388, 7392, 7393, 7394, 7395, 7396, 7397, 7398, 7399, 7400, 7401, 7402, 7403, 7404, 7405, 7406, 7407, 7408, 7409, 7410, 7411, 7412, 7413, 7414, 7415, 7416, 7417, 7418, 7419, 7420, 7421, 7422, 7423, 7424, 7425, 7426, 7427, 7428, 7429, 7430, 7431, 7432, 7433, 7434, 7435, 7436, 7437, 7438, 7439, 7440, 7441, 7442, 7443, 7444, 7445, 7446, 7447, 7448, 7449, 7450, 7451, 7452, 7453, 7454, 7455, 7456, 7457, 7458, 7459, 7460, 7461, 7462, 7463, 7464, 7465, 7466, 7467, 7468, 7469, 7470, 7471, 7472, 7473, 7474, 7475, 7476, 7477, 7478, 7479, 7480, 7481, 7482, 7483, 7484, 7485, 7487, 7488, 7489, 7490, 7492, 7493, 7494, 7497, 7499, 7500, 7502, 7504, 7509, 7511, 7512, 7516, 7521, 7523, 7524, 7528, 7536, 7540, 7548, 7552, 7560, 7564, 7572, 7576, 7584, 7586, 7588, 7590, 7592, 7594, 7596, 7598, 7600, 7602, 7604, 7606, 7608, 7610, 7612, 7614, 7616, 7617, 7618, 7619, 7620, 7621, 7622, 7623, 7626, 7627, 7631, 7635, 7639, 7643, 7644, 7645, 7646, 7647, 7650, 7651, 7652, 7653, 7656, 7657, 7658, 7659, 7660, 7661, 7662, 7663, 7664, 7665, 7666, 7667, 7668, 7669, 7670, 7671, 7672, 7673, 7674, 7675, 7676, 7677, 7678, 7679, 7680, 7681, 7682, 7683, 7684, 7685, 7686, 7687, 7688, 7689, 7690, 7691, 7692, 7693, 7694, 7695, 7696, 7697, 7698, 7699, 7700, 7701, 7702, 7703, 7704, 7705, 7706, 7707, 7708, 7709, 7710, 7711, 7712, 7713, 7714, 7715, 7718, 7719, 7722, 7723, 7724, 7726, 7727, 7728, 7730, 7731, 7732, 7734, 7735, 7736, 7738, 7739, 7740, 7742, 7743, 7744, 7746, 7747, 7748, 7750, 7751, 7752, 7754, 7755, 7756, 7758, 7759, 7760, 7762, 7763, 7764, 7766, 7767, 7768, 7770, 7771, 7772, 7774, 7775, 7776, 7778, 7779, 7780, 7781, 7782, 7783, 7784, 7785, 7786, 7787, 7788, 7789, 7792, 7793, 7794, 7795, 7798, 7799, 7800, 7801, 7804, 7805, 7806, 7807, 7810, 7811, 7812, 7813, 7816, 7817, 7818, 7819, 7822, 7823, 7824, 7825, 7828, 7829, 7830, 7831, 7834, 7835, 7836, 7837, 7840, 7841, 7842, 7843, 7844, 7845, 7846, 7847, 7848, 7849, 7850, 7851, 7852, 7853, 7854, 7855, 7856, 7857, 7858, 7859, 7860, 7861, 7862, 7863, 7864, 7865, 7866, 7867, 7868, 7869, 7870, 7871, 7872, 7873, 7874, 7875, 7876, 7877, 7878, 7879, 7880, 7881, 7882, 7883, 7884, 7885, 7886, 7887, 7888, 7889, 7890, 7891, 7892, 7893, 7894, 7895, 7896, 7897, 7898, 7899, 7900, 7901, 7902, 7903, 7904, 7905, 7906, 7907, 7908, 7909, 7910, 7911, 7912, 7913, 7916, 7917, 7918, 7919, 7920, 7921, 7922, 7923, 7924, 7925, 7926, 7927, 7928, 7929, 7930, 7931, 7932, 7933, 7934, 7935, 7936, 7937, 7938, 7939, 7940, 7941, 7942, 7943, 7944, 7945, 7946, 7947, 7948, 7949, 7950, 7951, 7952, 7953, 7954, 7955, 7956, 7957, 7958, 7959, 7960, 7961, 7962, 7963, 7964, 7965, 7966, 7967, 7968, 7969, 7970, 7971, 7973, 7974, 7975, 7976, 7977, 7978, 7979, 7981, 7982, 7983, 7984, 7985, 7986, 7987, 7988, 7990, 7991, 7992, 7993, 7994, 7995, 7996, 7998, 7999, 8000, 8001, 8002, 8003, 8004, 8006, 8007, 8008, 8009, 8010, 8011, 8012, 8014, 8015, 8016, 8017, 8018, 8019, 8020, 8022, 8023, 8024, 8025, 8026, 8027, 8028, 8030, 8031, 8032, 8033, 8034, 8035, 8036, 8039, 8040, 8041, 8042, 8043, 8044, 8047, 8048, 8049, 8050, 8051, 8052, 8055, 8056, 8057, 8058, 8059, 8060, 8063, 8064, 8066, 8067, 8068, 8071, 8072, 8074, 8075, 8076, 8079, 8080, 8082, 8083, 8084, 8088, 8090, 8091, 8096, 8098, 8100, 8102, 8104, 8108, 8112, 8116, 8120, 8124, 8128, 8132, 8134, 8136, 8138, 8140, 8142, 8144, 8146, 8148, 8150, 8152, 8154, 8156, 8158, 8160, 8162, 8164, 8166, 8168, 8170, 8172, 8174, 8176, 8178, 8180, 8182, 8184, 8186, 8188, 8190, 8192, 8194, 8196, 8198, 8200, 8201, 8202, 8203, 8204, 8205, 8206, 8207, 8208, 8209, 8210, 8211, 8212, 8213, 8214, 8215, 8216, 8217, 8218, 8219, 8220, 8221, 8222, 8223, 8224, 8225, 8226, 8227, 8228, 8229, 8230, 8231, 8232, 8233, 8234, 8235, 8236, 8237, 8238, 8239, 8240, 8241, 8242, 8243, 8244, 8245, 8246, 8247, 8248, 8249, 8250, 8251, 8252, 8253, 8254, 8255, 8256, 8257, 8258, 8259, 8260, 8261, 8262, 8263, 8264, 8265, 8266, 8267, 8268, 8269, 8270, 8271, 8272, 8273, 8274, 8275, 8276, 8277, 8278, 8280, 8281, 8282, 8284, 8285, 8286, 8287, 8288, 8290, 8291, 8292, 8293, 8294, 8296, 8297, 8298, 8300, 8301, 8302, 8303, 8304, 8306, 8307, 8308, 8309, 8310, 8311, 8314, 8315, 8316, 8318, 8319, 8320, 8321, 8322, 8323, 8324, 8325, 8326, 8327, 8328, 8330, 8331, 8332, 8335, 8336, 8337, 8338, 8339, 8340, 8341, 8342, 8343, 8344, 8346, 8347, 8348, 8349, 8352, 8353, 8354, 8355, 8356, 8357, 8358, 8359, 8360, 8363, 8364, 8365, 8368, 8369, 8370, 8371, 8374, 8375, 8378, 8379, 8382, 8383, 8386, 8387, 8388, 8389, 8390, 8391, 8392, 8396, 8397, 8398, 8399, 8400, 8401, 8402, 8403, 8406, 8407, 8408, 8409, 8410, 8411, 8412, 8413, 8416, 8417, 8418, 8419, 8420, 8421, 8422, 8423, 8426, 8427, 8428, 8429, 8432, 8433, 8434, 8435, 8436, 8437, 8438, 8442, 8443, 8444, 8445, 8446, 8448, 8449, 8450, 8451, 8452, 8453, 8454, 8455, 8457, 8458, 8459, 8460, 8461, 8462, 8465, 8466, 8467, 8468, 8469, 8470, 8471, 8474, 8475, 8476, 8477, 8478, 8481, 8482, 8483, 8484, 8485, 8486, 8487, 8488, 8490, 8494, 8495, 8496, 8500, 8501, 8506, 8510, 8511, 8512, 8516, 8517, 8518, 8520, 8521, 8522, 8526, 8527, 8528, 8533, 8534, 8536, 8537, 8538, 8543, 8544, 8549, 8550, 8553, 8554, 8555, 8556, 8557, 8558, 8559, 8560, 8562, 8564, 8565, 8566, 8567, 8568, 8569, 8570, 8572, 8574, 8575, 8576, 8577, 8578, 8579, 8580, 8582, 8584, 8585, 8586, 8587, 8588, 8589, 8590, 8591, 8593, 8594, 8595, 8596, 8597, 8598, 8599, 8600, 8601, 8602, 8603, 8604, 8605, 8606, 8607, 8608, 8609, 8610, 8611, 8612, 8613, 8614, 8615, 8616, 8617, 8618, 8619, 8620, 8621, 8622, 8623, 8624, 8625, 8626, 8627, 8628, 8629, 8630, 8631, 8632, 8633, 8634, 8635, 8636, 8637, 8638, 8639, 8640, 8641, 8642, 8643, 8646, 8647, 8648, 8649, 8650, 8651, 8652, 8653, 8654, 8655, 8656, 8658, 8659, 8660, 8661, 8662, 8663, 8664, 8665, 8666, 8667, 8668, 8670, 8671, 8672, 8673, 8674, 8675, 8676, 8677, 8678, 8679, 8680, 8682, 8683, 8684, 8685, 8686, 8687, 8688, 8689, 8690, 8691, 8692, 8694, 8695, 8696, 8697, 8698, 8699, 8700, 8701, 8702, 8703, 8704, 8706, 8707, 8708, 8709, 8710, 8711, 8712, 8713, 8714, 8715, 8716, 8717, 8718, 8719, 8720, 8721, 8722, 8723, 8724, 8725, 8726, 8727, 8728, 8729, 8730, 8731, 8732, 8733, 8734, 8735, 8736, 8737, 8738, 8739, 8740, 8741, 8742, 8743, 8744, 8745, 8746, 8747, 8748, 8749, 8750, 8751, 8752, 8753, 8754, 8755, 8756, 8757, 8758, 8759, 8760, 8761, 8762, 8763, 8764, 8765, 8766, 8767, 8768, 8769, 8770, 8771, 8772, 8773, 8774, 8775, 8776, 8777, 8778, 8779, 8780, 8781, 8782, 8783, 8784, 8785, 8786, 8787, 8788, 8789, 8790, 8791, 8792, 8793, 8794, 8795, 8796, 8797, 8798, 8799, 8800, 8801, 8802, 8803, 8804, 8805, 8806, 8807, 8808, 8809, 8810, 8811, 8812, 8813, 8814, 8815, 8816, 8817, 8818, 8819, 8820, 8821, 8822, 8823, 8824, 8825, 8826, 8827, 8828, 8829, 8833, 8834, 8835, 8836, 8841, 8842, 8843, 8844, 8845, 8846, 8847, 8848, 8849, 8853, 8854, 8855, 8856, 8862, 8863, 8864, 8865, 8866, 8867, 8868, 8869, 8870, 8874, 8875, 8876, 8882, 8883, 8884, 8885, 8886, 8887, 8888, 8889, 8890, 8894, 8895, 8896, 8897, 8902, 8903, 8904, 8905, 8906, 8907, 8908, 8909, 8910, 8911, 8912, 8913, 8914, 8915, 8919, 8920, 8921, 8927, 8929, 8930, 8931, 8932, 8933, 8934, 8935, 8936, 8937, 8938, 8939, 8943, 8944, 8945, 8953, 8954, 8955, 8956, 8957, 8958, 8959, 8960, 8961, 8962, 8963, 8964, 8968, 8969, 8978, 8979, 8980, 8981, 8984, 8985, 8986, 8998, 9008, 9009, 9010, 9022, 9033, 9034, 9035, 9046, 9058, 9059, 9070, 9083, 9094, 9107, 9108, 9118, 9132, 9144, 9163, 9164, 9165, 9166, 9167, 9168, 9170, 9189, 9190, 9191, 9192, 9193, 9194, 9196, 9197, 9198, 9199, 9202, 9203, 9204, 9205, 9206, 9207, 9211, 9212, 9213, 9214, 9215, 9216, 9219, 9220, 9224, 9225, 9226, 9227, 9228, 9231, 9236, 9237, 9238, 9239, 9240, 9244, 9245, 9248, 9249, 9250, 9251, 9252, 9253, 9256, 9257, 9260, 9261, 9262, 9263, 9264, 9265, 9268, 9269, 9272, 9273, 9274, 9275, 9276, 9277, 9278, 9281, 9282, 9286, 9287, 9290, 9291, 9294, 9295, 9296, 9298, 9299, 9300, 9302, 9303, 9304, 9305, 9312, 9313, 9314, 9315, 9316, 9317, 9318, 9323, 9324, 9328, 9329, 9330, 9332, 9333, 9334, 9336, 9337, 9338, 9340, 9341, 9342, 9344, 10118, 10119, 10139, 10140, 10347, 10348, 10490, 10491, 10492, 10707, 10708  }, xRefinement});
//      refs.push_back({std::set<int>{ 10, 11, 12, 13, 14, 15, 16, 17, 19, 22, 23, 24, 25, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 541, 547, 761, 775, 1072, 1103, 1235, 1237, 1410, 1439, 1563, 1613, 1615, 1823, 2171, 2222, 2298, 2300, 2346, 2347, 2348, 2359, 2360, 2366, 2367, 2368, 2369, 2370, 2372, 2374, 2376, 2423, 2425, 2480, 2486, 2499, 2687, 2689, 2711, 2713, 2721, 2745, 2751, 2753, 2761, 2767, 2769, 2783, 2785, 2796, 3071, 3115, 3639, 3705, 3737, 3739, 3741, 3743, 3745, 3747, 3796, 3797, 3798, 3804, 3806, 3809, 3810, 3811, 3849, 3851, 3853, 3855, 3900, 3916, 3920, 3922, 3925, 3927, 3938, 3940, 3943, 3948, 3954, 3956, 3960, 3961, 3962, 3964, 3966, 3969, 3971, 3975, 3976, 3982, 3993, 3996, 4027, 4044, 4046, 4055, 4057, 4076, 4077, 4078, 4079, 4085, 4089, 4108, 4109, 4112, 4119, 4126, 4132, 4134, 4135, 4137, 4146, 4147, 4148, 4149, 4171, 4172, 4173, 4203, 4264, 4284, 4310, 4437, 4439, 4441, 4489, 4490, 4492, 4497, 4498, 4503, 4657, 4912, 4924, 4928, 4940, 4944, 4956, 5175, 5374, 5435, 5443, 5451, 5487, 5646, 5658, 5670, 5691, 5700, 5712, 5717, 5729, 5737, 5743, 5746, 5791, 5811, 5821, 5842, 5881, 5893, 5913, 5914, 5940, 5942, 5961, 5963, 5965, 5967, 6003, 6004, 6005, 6007, 6020, 6021, 6025, 6027, 6028, 6033, 6034, 6035, 6048, 6049, 6051, 6063, 6073, 6092, 6125, 6138, 6149, 6156, 6165, 6172, 6173, 6174, 6175, 6176, 6177, 6178, 6179, 6217, 6220, 6221, 6227, 6230, 6244, 6246, 6250, 6251, 6266, 6268, 6270, 6280, 6282, 6284, 6286, 6288, 6289, 6290, 6291, 6292, 6293, 6294, 6295, 6297, 6300, 6301, 6302, 6303, 6304, 6306, 6307, 6317, 6318, 6319, 6320, 6324, 6326, 6330, 6351, 6360, 6376, 6380, 6390, 6392, 6394, 6404, 6406, 6410, 6414, 6417, 6418, 6423, 6428, 6432, 6443, 6444, 6445, 6453, 6455, 6468, 6482, 6483, 6484, 6485, 6486, 6487, 6489, 6491, 6502, 6516, 6518, 6524, 6525, 6526, 6540, 6541, 6542, 6543, 6544, 6552, 6554, 6557, 6560, 6561, 6564, 6565, 6567, 6592, 6593, 6594, 6595, 6630, 6653, 6685, 6699, 6701, 6710, 6711, 6714, 6715, 6722, 6723, 6726, 6732, 6733, 6734, 6735, 6736, 6737, 6739, 6740, 6741, 6753, 6754, 6755, 6760, 6761, 6770, 6771, 6772, 6773, 6775, 6776, 6777, 6788, 6815, 6818, 6819, 6826, 6830, 6840, 6841, 6842, 6848, 6851, 6852, 6853, 6855, 6862, 6874, 6876, 6882, 6884, 6890, 6891, 6892, 6894, 6895, 6898, 6899, 6900, 6901, 6902, 6903, 6906, 6907, 6908, 6909, 6910, 6915, 6916, 6917, 6923, 6924, 6933, 6934, 6935, 6941, 6959, 6961, 6972, 6973, 6974, 6976, 6977, 6980, 6982, 6984, 6989, 6991, 7008, 7009, 7010, 7012, 7013, 7014, 7015, 7016, 7017, 7018, 7020, 7021, 7022, 7023, 7040, 7091, 7164, 7166, 7168, 7170, 7172, 7199, 7204, 7287, 7397, 7404, 7405, 7408, 7409, 7420, 7422, 7428, 7430, 7432, 7433, 7434, 7441, 7442, 7445, 7446, 7452, 7453, 7454, 7456, 7457, 7458, 7459, 7464, 7465, 7466, 7467, 7468, 7469, 7476, 7477, 7480, 7481, 7484, 7489, 7490, 7493, 7494, 7500, 7502, 7504, 7512, 7516, 7524, 7528, 7536, 7540, 7548, 7552, 7560, 7564, 7572, 7586, 7588, 7590, 7592, 7594, 7596, 7598, 7600, 7602, 7604, 7606, 7608, 7610, 7612, 7614, 7645, 7647, 7651, 7653, 7689, 7691, 7709, 7713, 7715, 7724, 7728, 7739, 7743, 7770, 7776, 7780, 7781, 7784, 7785, 7788, 7789, 7806, 7812, 7813, 7819, 7825, 7859, 7880, 7888, 7900, 7976, 7978, 7984, 7986, 7992, 7994, 8002, 8008, 8009, 8010, 8016, 8019, 8024, 8026, 8027, 8033, 8034, 8035, 8040, 8041, 8043, 8048, 8049, 8050, 8051, 8056, 8057, 8058, 8059, 8060, 8066, 8067, 8068, 8074, 8075, 8076, 8079, 8080, 8082, 8083, 8084, 8088, 8090, 8091, 8136, 8138, 8140, 8142, 8144, 8146, 8148, 8150, 8152, 8154, 8156, 8158, 8160, 8162, 8164, 8166, 8168, 8170, 8172, 8174, 8178, 8180, 8182, 8186, 8188, 8190, 8196, 8198, 8211, 8240, 8254, 8262, 8270, 8276, 8288, 8290, 8292, 8306, 8307, 8309, 8319, 8322, 8325, 8330, 8346, 8358, 8370, 8371, 8374, 8375, 8379, 8383, 8390, 8391, 8392, 8398, 8400, 8401, 8402, 8403, 8411, 8412, 8413, 8421, 8423, 8433, 8482, 8484, 8490, 8494, 8500, 8510, 8511, 8516, 8522, 8526, 8527, 8528, 8538, 8543, 8544, 8582, 8595, 8596, 8605, 8606, 8614, 8615, 8616, 8618, 8625, 8626, 8627, 8628, 8632, 8635, 8636, 8637, 8638, 8639, 8640, 8642, 8646, 8647, 8648, 8649, 8650, 8651, 8654, 8656, 8659, 8660, 8661, 8662, 8663, 8664, 8666, 8668, 8671, 8672, 8673, 8674, 8675, 8676, 8678, 8679, 8680, 8682, 8684, 8685, 8686, 8687, 8688, 8690, 8691, 8692, 8695, 8696, 8697, 8698, 8699, 8700, 8701, 8703, 8704, 8707, 8708, 8709, 8710, 8711, 8712, 8713, 8715, 8717, 8718, 8720, 8721, 8722, 8723, 8724, 8726, 8727, 8728, 8729, 8730, 8731, 8732, 8733, 8734, 8735, 8736, 8737, 8738, 8739, 8740, 8741, 8742, 8743, 8744, 8745, 8746, 8747, 8748, 8749, 8751, 8752, 8753, 8755, 8757, 8758, 8759, 8760, 8761, 8762, 8763, 8764, 8765, 8769, 8770, 8771, 8772, 8773, 8775, 8776, 8777, 8781, 8782, 8783, 8785, 8787, 8788, 8789, 8794, 8797, 8800, 8801, 8806, 8807, 8810, 8812, 8813, 8825, 8828, 8829, 8835, 8836, 8841, 8842, 8843, 8844, 8845, 8848, 8849, 8855, 8856, 8863, 8864, 8865, 8866, 8869, 8876, 8883, 8884, 8885, 8886, 8887, 8889, 8895, 8896, 8897, 8902, 8903, 8904, 8905, 8909, 8910, 8911, 8914, 8921, 8929, 8930, 8933, 8935, 8936, 8938, 8944, 8945, 8953, 8954, 8955, 8959, 8969, 8978, 8979, 8986, 9008, 9009, 9010, 9033, 9034, 9035, 9046, 9058, 9059, 9070, 9083, 9094, 9107, 9108, 9118, 9132, 9163, 9164, 9167, 9189, 9190, 9191, 9193, 9194, 9197, 9198, 9203, 9206, 9214, 9220, 9249, 9257, 9261, 9269, 9273, 9274, 9286, 9300, 9303, 9304, 9305, 9313, 9314, 9324, 9333, 10491, 10708  }, Teuchos::null});
//      refs.push_back({std::set<int>{ 714, 3644, 3648, 3654, 3659, 3661, 3666, 3675, 3737, 3739, 3741, 3742, 3954, 3956, 3996, 5922, 6094, 6220, 6221, 6223, 6229, 6241, 6305, 6395, 6487, 9387, 9389, 9455, 9457, 9624  }, isotropicRefinement});
      refs.push_back({std::set<int>{ 2366, 2898, 3059, 3067, 5717, 5743, 6432, 6444, 6497, 6664, 6688, 6708, 6709, 6861, 8038, 8046, 8054, 8062, 8329, 8330, 8345, 8362, 8440, 8441, 8456, 8472, 8473, 8505, 8519, 8848, 8869, 8914, 9016, 9042, 9090, 9091, 9116, 9139, 9210, 9257, 9266, 9793, 9800, 10125, 10359, 10513, 10553, 10569, 10583, 10710, 10711, 10927, 11134, 11135, 11204, 11258, 11294, 11302, 11411, 11420, 11692, 11730, 11731, 11747, 11763, 11764, 11779, 11780, 12035, 12051, 12056, 12092, 12153, 12285, 12380, 12381, 12406, 12570, 12632, 12768, 12769, 12848, 12926, 13063, 13064, 13065, 13081, 13092, 13100, 13665, 13692, 13880, 13881, 13885, 13896, 13930, 13932, 13960, 13961, 13978, 13979, 14024, 14058, 14066, 14074  }, xRefinement});
    }
    
    bool repartitionAndRebuild = false;
    for (auto bcRef : bcRefs)
    {
      // all isotropic, and need to have 1-irregularity enforced afterward
      auto cells = bcRef.first;
      auto refinementType = bcRef.second;
      
      form.vlasovMesh()->hRefine(cells, refinementType, repartitionAndRebuild);
      
      if (rank == 0)
      {
        Camellia::print("cells for isotropic refinement", cells);
      }
      
      form.vlasovMesh()->enforceOneIrregularity(true);
    }
    repartitionAndRebuild = true;
    for (auto ref : refs)
    {
      auto cells = ref.first;
      auto refinementType = ref.second;
      
      if (refinementType != Teuchos::null)
      {
        form.vlasovMesh()->hRefine(cells, refinementType, repartitionAndRebuild);
        if (rank == 0)
        {
          if (refinementType->numChildren() == 4)
          {
            Camellia::print("cells for isotropic refinement", cells);
          }
          else
          {
            Camellia::print("cells for x-anisotropic refinement", cells);
          }
        }
      }
      else
      {
        form.vlasovMesh()->hUnrefine(cells, repartitionAndRebuild);
        if (rank == 0)
        {
          Camellia::print("cells for unrefinement", cells);
        }
      }
      if (rank == 0)
      {
        refHistory->saveToFile("/tmp/refHistory.txt");
      }
    }
    
    if (rank == 0)
    {
      std::cout << "Completed refinement series.\n";
    }
    
    runTests(vWidth, xWidth, x0, dt, useCrankNicolson, q_values[0], m_values[0], form, "after final refinements");
  }
  
  const bool printAllocationReport = (rank == 0);
  Camellia::finalize(printAllocationReport);
  
  return 0;
}
