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
#include "Projector.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "TimeRampFunction.h"
#include "TrigFunctions.h"
#include "VlasovBackwardEulerFormulation.h"

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

//! TODO: put this into the main Camellia source (i.e. part of the library)
class CellIDFunction : public Function
{
public:
  CellIDFunction() : Function("CellIDFunction", 0, Function::VARIES_IN_NO_DIMENSION) {}

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
  
  void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache)
  {
    auto underlyingView1 = values.template getUnderlyingView<1>();
    
    const int numCells = underlyingView1.extent_int(0);
    
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.size() != numCells, std::invalid_argument, "underlying view doesn't have the expected length");
    
    auto underlyingViewHost = Kokkos::create_mirror_view(underlyingView1);
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      underlyingViewHost(cellOrdinal) = cellIDs[cellOrdinal];
    }
    
    Kokkos::deep_copy(underlyingView1, underlyingViewHost);
  }
};

//! TODO: put this into the main Camellia source (i.e. part of the library)
class CellPolyOrderFunction : public Function
{
public:
  CellPolyOrderFunction() : Function("CellPolyOrderFunction", 0, Function::VARIES_IN_NO_DIMENSION) {}

  using Function::values;
  
  virtual void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();

    int numCells = cellIDs.size();
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      int polyOrder = basisCache->mesh()->cellPolyOrder(cellIDs[cellOrdinal]);
      
      for (int pointOrdinal=0; pointOrdinal<values.dimension(1); pointOrdinal++)
      {
        values(cellOrdinal,pointOrdinal) = polyOrder;
      }
    }
  }
  
  void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache)
  {
    auto underlyingView1 = values.template getUnderlyingView<1>();
    
    const int numCells = underlyingView1.extent_int(0);
    
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.size() != numCells, std::invalid_argument, "underlying view doesn't have the expected length");
    
    auto underlyingViewHost = Kokkos::create_mirror_view(underlyingView1);
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      underlyingViewHost(cellOrdinal) = basisCache->mesh()->cellPolyOrder(cellIDs[cellOrdinal]);
    }
    
    Kokkos::deep_copy(underlyingView1, underlyingViewHost);
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
    FunctionPtr _v; // i.e. the acceleration, defined on configurationSpaceMesh
    FunctionPtr _velocity_v; // i.e. the acceleration, defined on configurationSpaceMesh
    bool _useGlobalConstraint = true;
  public:
    AnisotropicCFLRefinementStrategy(ErrorIndicatorPtr errorIndicator, double relativeErrorThreshold, MeshPtr configurationSpaceMesh, FunctionPtr v, FunctionPtr velocity_v, double cfl, double dt, bool useGlobalConstraint=true, int maxP=10):
    _configurationSpaceMesh(configurationSpaceMesh),
    _cfl(cfl),
    _v(v),
    _velocity_v(velocity_v),
    _dt(dt),
    _useGlobalConstraint(useGlobalConstraint),
    TRefinementStrategy<double>( errorIndicator, relativeErrorThreshold, 0, maxP )
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
      set<GlobalIndexType> cellsForPRefinement;
      
      auto meshTopo = mesh()->getTopology();
      ExtrudedMeshTopology* extrudedMeshTopo = dynamic_cast<ExtrudedMeshTopology*>(meshTopo.get());
      TEUCHOS_TEST_FOR_EXCEPTION(extrudedMeshTopo == NULL, std::invalid_argument, "Phase-space MeshTopology must be an ExtrudedMeshTopology");
      
      bool printToConsole = (this->mesh()->Comm()->MyPID() == 0);
      
      auto physicalMeshTopo = extrudedMeshTopo->physicalMeshTopology();
      
      auto cellTopo = CellTopology::lineTensorTopology(CellTopology::line()); // line, once extruded
      auto isotropicRefinement = RefinementPattern::anisotropicRefinementPattern(cellTopo, {1,1});
      auto         xRefinement = RefinementPattern::xAnisotropicRefinementPatternExtrudedLine();
      
      const auto & cellPRefinements = mesh()->globalDofAssignment()->getCellPRefinements();
      int initialH1Order = mesh()->globalDofAssignment()->getInitialH1Order()[0]; // assuming isotropic
      
      const double vv_global = _useGlobalConstraint ? _velocity_v->maximumValue(_configurationSpaceMesh) : 0;
      const double  v_global = _useGlobalConstraint ?          _v->maximumValue(mesh()) : 0;
      
      auto canRefine = [&](const GlobalIndexType &cellID, bool &inX, bool &inV, bool &inP)
      {
        double v0, v1;
        extrudedMeshTopo->extrusionVerticesForPhaseSpaceCell(v0, v1, cellID, 0);
        
        int p = this->mesh()->globalDofAssignment()->getH1Order(cellID)[0]; // we're assuming isotropic in p
        inP = (p < _max_p);
        
        using std::max;
        using std::abs;
        const double vx = _useGlobalConstraint ? v_global : max(abs(v0),abs(v1));
        
        GlobalIndexType physicalCellID = extrudedMeshTopo->physicalCellIDForPhaseSpaceCellID(cellID);
        
        const double vv = _useGlobalConstraint ? vv_global : getMaxMagnitudeOnCell(_velocity_v, _configurationSpaceMesh, physicalCellID);
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
          if (inP)
          {
            std::cout << "CAN refine in p.\n";
          }
          else
          {
            std::cout << "CAN'T refine in p.\n";
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
      
      auto pRefinementHalo = [&](auto &&pRefinementHalo, const GlobalIndexType &cellID, map<GlobalIndexType,int> &cell_dp) -> void
      {
        // original caller should place the central (cellID, dp) into cell_dp, with the difference between target H^1 order and the initial H^1 order of the mesh as dp.
        
        int dp = cell_dp[cellID];
        if (dp == 0) return;
        
        int minNeighbor_dp = dp - 1;
        if (cellPRefinements.find(cellID) != cellPRefinements.end())
        {
          minNeighbor_dp += cellPRefinements.find(cellID)->second;
        }
        
        for (int i=0; i<4; i++)
        {
          const int edgeDim = 1;
          CellPtr cell = extrudedMeshTopo->getCell(cellID);
          
          auto neighborInfo        = cell->getNeighborInfo(i, meshTopo);
          auto neighborCellIndex   = neighborInfo.first;
          auto neighborSideOrdinal = neighborInfo.second;
          
          if (neighborCellIndex == MESHTOPOLOGY_INVALID_CELL_INDEX) continue; // no neighbor (boundary side)
          
          // if neighbor is a parent, then we want to check children along this side
          
          CellPtr neighbor = extrudedMeshTopo->getCell(neighborCellIndex);
          const bool leafNodesOnly = true; // this only matters if there are multiple levels of refinement (i.e. mesh is not 1-irregular, which it should be)
          vector<pair<GlobalIndexType,unsigned>> childrenForSide = neighbor->getDescendantsForSide(neighborSideOrdinal, meshTopo, leafNodesOnly); // if cell is not a parent, returns the cell itself
          
          for (auto activeNeighborEntry : childrenForSide)
          {
            auto & activeNeighborID = activeNeighborEntry.first;
            auto pRefEntry = cellPRefinements.find(activeNeighborID);
            
            int current_dp = (pRefEntry != cellPRefinements.end()) ? pRefEntry->second : 0;
            if (current_dp < minNeighbor_dp)
            {
              if ((cell_dp.find(activeNeighborID) == cell_dp.end()) || (cell_dp[activeNeighborID] < minNeighbor_dp))
              {
                // then we need to update cell_dp, and check activeNeighborID's neighbors as well
                cell_dp[activeNeighborID] = minNeighbor_dp - current_dp;
                pRefinementHalo(pRefinementHalo, activeNeighborID, cell_dp);
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
        
        bool xRefinementIsAllowed, vRefinementIsAllowed, pRefinementIsAllowed;
        canRefine(cellID, xRefinementIsAllowed, vRefinementIsAllowed, pRefinementIsAllowed);
        
        if (!xRefinementIsAllowed && !vRefinementIsAllowed && !pRefinementIsAllowed)
        {
          return;
        }
        bool onlyPRefinementAllowed = pRefinementIsAllowed && !xRefinementIsAllowed && !vRefinementIsAllowed;
        
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
          
          bool haloCellAllowsXRefinement, haloCellAllowsVRefinement, haloCellAllowsPRefinement;
          canRefine(xHaloCellID, haloCellAllowsXRefinement, haloCellAllowsVRefinement, haloCellAllowsPRefinement);
          
          allHaloCellsAllowRefinement &= haloCellAllowsXRefinement;
        }
        
        for (auto xvHaloCellID : halo_xv)
        {
          // here, we require that x and v refinement are allowed
          
          bool haloCellAllowsXRefinement, haloCellAllowsVRefinement, haloCellAllowsPRefinement;
          canRefine(xvHaloCellID, haloCellAllowsXRefinement, haloCellAllowsVRefinement, haloCellAllowsPRefinement);
          
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
          else if (pRefinementIsAllowed)
          {
            cellsForPRefinement.insert(cellID);
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
            
            bool haloCellAllowsXRefinement, haloCellAllowsVRefinement, haloCellAllowsPRefinement;
            canRefine(xHaloCellID, haloCellAllowsXRefinement, haloCellAllowsVRefinement, haloCellAllowsPRefinement);
            
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
          else if (pRefinementIsAllowed)
          {
            cellsForPRefinement.insert(cellID);
          }
        }
        else if (pRefinementIsAllowed)
        {
          cellsForPRefinement.insert(cellID);
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
      
      map<GlobalIndexType, int> cell_dp;
      for (auto cellID : cellsForPRefinement)
      {
        cell_dp[cellID] = 1;
      }
      
      for (auto cellID : cellsForPRefinement)
      {
        pRefinementHalo(pRefinementHalo, cellID, cell_dp);
      }
      
      map<int, set<GlobalIndexType> > pRefinements;
      for (auto entry : cell_dp)
      {
        pRefinements[entry.second].insert(entry.first);
      }
      
      if (printToConsole)
      {
        if (cellsForIsotropicRefinement.size() > 0) Camellia::print("cells for isotropic refinement",     cellsForIsotropicRefinement);
        if (cellsForXRefinement.size() > 0)         Camellia::print("cells for x-anisotropic refinement", cellsForXRefinement);
        if (cellsForPRefinement.size() > 0)
        {
          for (const auto & entry : pRefinements)
          {
            int dp = entry.first;
            string label = string("cells for p-refinement (dp=") + to_string(dp) + string(")");
            Camellia::print(label, entry.second);
          }
        }
      }
      for (const auto & entry : pRefinements)
      {
        int dp = entry.first;
        this->mesh()->pRefine(entry.second, dp, repartitionAndRebuild);
      }
      this->mesh()->hRefine(cellsForIsotropicRefinement, isotropicRefinement, repartitionAndRebuild);
      this->mesh()->hRefine(cellsForXRefinement, xRefinement, repartitionAndRebuild);
      
//      this->mesh()->globalDofAssignment()->rebuildLookups();
      if (printToConsole)
      {
        std::cout << "Vlasov mesh has " << this->mesh()->getTopology()->activeCellCount() << " active elements and " << this->mesh()->globalDofCount() << " degrees of freedom.\n";
      }
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

RefinementStrategyPtr constructRefStrategy(VlasovBackwardEulerFormulation &form, double refinementThreshold, int maxElements, int maxRefinementLevel, ErrorIndicatorPtr unrefinementErrorIndicator, int maxP)
{
  auto vlasovSoln  = form.vlasovSolution();
  auto refStrategy = RefinementStrategy::energyErrorRefinementStrategy(vlasovSoln, refinementThreshold);

  refStrategy->setMaxElements(maxElements);
  refStrategy->setMaximumRefinementLevel(maxRefinementLevel);
  refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
  const int H1Order = vlasovSoln->mesh()->globalDofAssignment()->getInitialH1Order()[0];
  refStrategy->setMaxP(maxP);
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
      if (form.vlasovMesh()->globalDofCount() != unrefinementStrategy->getNumDofs(numRefinements - 1))
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
  if (form.vlasovMesh()->numGlobalDofs() != refStrategy->getNumDofs(numRefinements - 1))
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
    
    double beamIntensity_eV = 0.5; // [eV] beam intensity from Christenson & Lau
    
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
    bool enforceInflowBCsWeakly = false;
    
    bool useAdaptivity = false;
    int maxElements = std::numeric_limits<int>::max();
    
    int xElements = 16;
    int vElements = 20;
    int numUniformRefinements = 2;
    int delta_k   = 2;
    double dt_max = -1.0;
    double timeRampWidth = 2.0; // assuming domain width of 1.0, velocity of 1.0, this fully turns on BCs after two transit times.  Modestly aggressive; could increase to 3.0 if there appear to be artifacts.
    int H1Order = 3; // this gives f, E (which are in L^2) a polynomial order of 2
    int maxP = -1;
    int minSubsteps = 2;  // for fixed-point iteration.
    int maxSubsteps = 15; // for fixed-point iteration.
    std::string normChoice = "graph"; // "conservative", "L2" are other choices
    double fixedPointTol = 1e-8;
    int numTimeSteps = 80;
    int outputFrequency = 1; // number of time steps to take before outputting viz.
    int maximumRefinementLevel = std::numeric_limits<int>::max(); // default to no maximum refinement level
    bool useEnergyForUnrefinements = true;
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
    bool useGlobalCFLConstraint = true; // if false, we allow more refinements where acceleration, velocity are lower
    double energyErrorTarget = 0.01; // try to refine to get below this threshold
    int cubatureEnrichmentDegree = 0;
    
    int greedyUnrefinementFrequency = 0;       // set to a positive number N to do a greedy unrefinement every N time steps
    double greedyUnrefinementThreshold = 1e-4; // unrefinements of parent cells with error below this threshold times the max element error are allowed
    
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
    cmdp.setOption("maxP", &maxP, "maximum H^1 order to p-refine to (-1 indicates no p refinements)");
    cmdp.setOption("dt", &dt_max );
    cmdp.setOption("CFL", &CFL_max);
    cmdp.setOption("CFL_BC", &CFL_for_BCs, "CFL to use for initial BC-resolving refinements.  Default (-1) indicates to use the same CFL as for in-situ refinements.  (HIGHLY EXPERIMENTAL)");
    cmdp.setOption("useGlobalCFLConstraint", "useLocalCFLConstraint", &useGlobalCFLConstraint);
    cmdp.setOption("beamIntensity", &beamIntensity_eV, "electron beam intensity in [eV] (used to compute drift velocity)"); // [eV] beam intensity from Christenson & Lau
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
    cmdp.setOption("numTimeSteps", &numTimeSteps);
    cmdp.setOption("outputFrequency", &outputFrequency);
    cmdp.setOption("outputDir", &outputDir);
    cmdp.setOption("useAdaptivity", "useFixedMesh", &useAdaptivity);
    cmdp.setOption("greedyUnrefinementFrequency", &greedyUnrefinementFrequency);
    cmdp.setOption("greedyUnrefinementThreshold", &greedyUnrefinementThreshold);
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
    cmdp.setOption("enforceBCsWeakly", "enforceBCsStrongly", &enforceInflowBCsWeakly);
    cmdp.setOption("alpha0", &alpha0);
    cmdp.setOption("alphaMultiplier", &alphaMultiplier);
    cmdp.setOption("useCrankNicolson", "useBackwardEuler", &useCrankNicolson);
    cmdp.setOption("useConservativeNorm", "useGraphNorm", &useConservativeNorm);
    cmdp.setOption("voltageBCDamping", &voltageBCDamping);
    cmdp.setOption("criticalCurrentRatio", &criticalCurrentRatio);
    cmdp.setOption("cubatureEnrichmentDegree", &cubatureEnrichmentDegree);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    if (maxP < 1)
    {
      maxP = H1Order;
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
    
    if (dt_max < 0)
    {
      dt_max = 1e-8; // experimentally, this appears to be about the biggest (initial) time step we can get away with, for adaptive mesh with backward Euler.  Could be that we could increase this by using Crank-Nicolson.  Could also be we should do something more sophisticated with the test space inner product.  Something to consider: What would a *Newton* iteration look like?  What does the linearized system look like?  Can we specify a test space that reflects that underlying system, rather than one step of the fixed point iteration that we're actually solving??
      // (Here's something very weird: during initial mesh refinement, the error increases with every refinement, seemingly without bound.  Is this a feature of the problem setup, or indicative of a general problem with the error measurement?  Notably, we do *not* generally see this once the actual run starts.)
    }
    
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
    double vx_coord = v0;
    for (int i=0; i<=vElements; i++)
    {
      vx_coord = v0 + (i * vWidth) / vElements;
      vxVertices.push_back(vx_coord);
    }
    
    double dt = dt_max;
    int maxMomentNumber = 1; // we want to compute current for circuit solve.
    
    VlasovBackwardEulerFormulation form(q_values, m_values, speciesLabels, useSerendipityBasis, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k, maxMomentNumber, useCrankNicolson, enforcePositivity);
    
    form.vlasovSolution()->setCubatureEnrichmentDegree(cubatureEnrichmentDegree);
    
    form.setAlpha(alpha0, alphaMultiplier);
    
    form.setTestIP(normChoice);
    if (rank == 0)
    {
      std::cout << "Test space IP:\n";
      form.vlasovSolution()->ip()->printInteractions();
    }
    
    if (disableElectricField)
    {
      // set the RHS to 0 -- in combination with the zero potential BC we set above, this forces the electric field to be zero.
      // (This needs to be done after the VBEFormulation has been constructed, since it calls poissonConfig->setMoments(), which
      //  is where the RHS for Poisson gets set up.)
      poissonConfig->solution()->setRHS(RHS::rhs());
    }
    
    // save off the RHS
    RHSPtr poissonRHS = poissonConfig->solution()->rhs();
    if (emFieldStartTime > 0)
    {
      // turn things off for now
      poissonConfig->solution()->setRHS(RHS::rhs());
      extraAccelerationParameter->setValue(0);
    }
    
    form.refineUniformly(numUniformRefinements);
    form.setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    
    auto voltageParameter = setPoissonBC(form, x0, x1);
    
    auto moments_k   = form.solutionMomentsPriorTime(); // we need to call update() on this after each time step
    auto moments_kp1 = form.solutionMoments();
    
    // voltageParameter is a "ParameterFunction" -- we need fill it in with an appropriate value
    // we allow delay of EM fields turning on; if delayed, we ramp the EM field more quickly so that it comes fully on at the same time as the inflow BC.
    bool useRampForEMField = false;
    double emFieldFullyOnTime = useRampForEMField ? timeRampWidth : emFieldStartTime;
    auto timeRampFunction = TimeRampFunction::timeRamp(form.t(), emFieldFullyOnTime, emFieldStartTime);
    double voltageValue = potentialBC / potentialScaling * voltageBCDamping;  // 12000 V
    voltageParameter->setValue(voltageValue * timeRampFunction->value()); // ramp up the voltage just as we do the inflow BC
    
    FunctionPtr velocityExtremaRestriction =   Function::meshSideSetCharacteristic(DIRICHLET_SET_TAG_NAME, form.tagVlasovMaxVelocity(0))
    + Function::meshSideSetCharacteristic(DIRICHLET_SET_TAG_NAME, form.tagVlasovMinVelocity(0));
    
    std::vector<FunctionPtr> fAtExtrema(numSpecies);
    for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
    {
      fAtExtrema[speciesOrdinal] = Function::solution(form.f(speciesOrdinal), form.vlasovSolution()) * velocityExtremaRestriction;
    }
    
    if (rank == 0)
    {
      if (useCrankNicolson) std::cout << "Using Crank-Nicolson in time.\n";
      std::cout << "Setting non-dimensional voltage at x=1 to " << potentialBC / potentialScaling << std::endl;
    }
    
    auto configMesh = poissonConfig->configurationSpaceMesh();
    
    // electrons flow in from left
    double u0_e;
    {
      const double e    = 1.602176634e-19; // electron's electric charge in Coulombs
      const double m_e  =  9.10938356e-31;  // m_e, mass of electron in kg
      const double W    = beamIntensity_eV; // default is 0.5 [eV], from Christenson & Lau; user may override
      
      // v_e is defined by 1/2 m v^2 = W ==> v_beam = sqrt( 2 W /m ):
      const double v_e = sqrt(2 * W * e / m_e ); // e converts from eV to Joules
      u0_e = v_e / velocityScaling;
      if (rank == 0)
      {
        std::cout << "u0_e: " << u0_e << std::endl;
        if (enforceInflowBCsWeakly)
        {
          std::cout << "Enforcing inflow BCs weakly (through extra RHS term).\n";
        }
        else
        {
          std::cout << "Enforcing inflow BCs strongly (via Dirichlet conditions on t_n).\n";
        }
      }
    }
    
    double sigmaSquared_e = electronThermalVelocity * electronThermalVelocity * u0_e * u0_e;
    double n = 1.0;
    const double firstMomentOfMaxwellian_e = MaxwellianFunction::momentNormalized1D(u0_e, sigmaSquared_e, 1); // will return u0_e
    const double J_inflow_physical_e = criticalCurrentRatio * 2.1e5; // Amps // criticalCurrentRatio: CLI parameter (default of 0.95 is Case C2 from C & L; use 1.1 for Case C3)
    const double J_inflow_scaled_e   = J_inflow_physical_e / currentDensityScaling;
    const double inflow_weight_e     = J_inflow_scaled_e / firstMomentOfMaxwellian_e;
    if (rank == 0)
    {
      std::cout << "sigma_e / u0_e: " << sqrt(sigmaSquared_e / (u0_e * u0_e) ) << std::endl;
      std::cout << "inflow_weight_e: " << inflow_weight_e << std::endl;
    }
    
    FunctionPtr positiveRestriction = Function::heavisideY(0); // don't impose right-going conditions for negative velocities, even when Maxwellian extends past v=0
    FunctionPtr fM_u0_e = inflow_weight_e * MaxwellianFunction::maxwellian(1, 1, {u0_e}, sigmaSquared_e, n) * positiveRestriction;
    
    // ions flow in from right
    double u0_ion = -u0_e / sqrt(1836.15); // ratio of electron mass to ion mass (assigning them both the same energy)
    double sigmaSquared_ion = ionThermalVelocity * ionThermalVelocity * u0_ion * u0_ion;
    FunctionPtr negativeRestriction = 1 - Function::heavisideY(0); // don't impose right-going conditions for negative velocities, even when Maxwellian extends past v=0
    FunctionPtr fM_u0_ion = MaxwellianFunction::maxwellian(1, 1, {u0_ion}, sigmaSquared_ion, n) * negativeRestriction;
    
    FunctionPtr f_e_in = fM_u0_e;
    FunctionPtr f_i_in = fM_u0_ion;
    
    ErrorIndicatorPtr unrefinementErrorIndicator;
    if (!useEnergyForUnrefinements)
    {
      if (greedyUnrefinementFrequency > 0)
      {
        std::cout << "Error: for greedy unrefinements, must have useEnergyForUnrefinements=true.\n";
        return -1;
      }
      unrefinementErrorIndicator = ErrorIndicator::unrefinementProjectionErrorIndicator(form.vlasovSolution(), form.f());
    }
    double refinementThreshold = 1.1; // > 1 ==> no refinements will take place
    RefinementStrategyPtr unrefinementStrategy = constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, unrefinementErrorIndicator, maxP);
    
    form.setFixedPointMaxSubsteps(maxSubsteps);
    form.setFixedPointMinSubsteps(minSubsteps);
    form.setFixedPointTolerance(fixedPointTol);
    
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
    
    using std::vector;
    using std::tuple;
    vector<tuple<VarPtr,FunctionPtr,FunctionPtr,int> > bcFunctionsToResolve; // (var, exact function, solution function, tag)
    
    std::vector<FunctionPtr> vlasovFunctionsToExport;
    std::vector<std::string> vlasovFunctionsToExportNames;
    
    std::vector<FunctionPtr> poissonFunctionsToExport;
    std::vector<std::string> poissonFunctionsToExportNames;
    
    FunctionPtr vx = form.v_i(0);
    int speciesOrdinal = 0;
    std::vector<FunctionPtr> f0, f_in;
    if (includeElectrons)
    {
      f0.push_back(Function::zero()); // vacuum initial conditions for both species
      f_in.push_back(f_e_in);
      
      VarPtr tn_e = form.t_n_hat(speciesOrdinal);
      
      auto inflowTag = form.tagVlasovSpatialInflow();
      
      FunctionPtr sgn = Function::sideParity();
      FunctionPtr n = Function::normal();
      FunctionPtr n_x = n->spatialComponent(1); // 1-based
      
      vlasovFunctionsToExportNames.push_back("f_e_in");
      vlasovFunctionsToExport.push_back(f_e_in);
      
      FunctionPtr tn_inflowExact = f_e_in * sgn * n_x * vx;
      const bool weightFluxesBySideParity = true;
      FunctionPtr tn_soln = Function::solution(tn_e, form.vlasovSolution(), weightFluxesBySideParity);
      bcFunctionsToResolve.push_back({tn_e,tn_inflowExact,tn_soln,inflowTag});
      speciesOrdinal++;
    }
    if (includeIons)
    {
      f0.push_back(Function::zero()); // vacuum initial conditions for both species
      f_in.push_back(f_i_in);
      
      VarPtr tn_i = form.t_n_hat(speciesOrdinal);
      
      auto inflowTag = form.tagVlasovSpatialInflow();
      
      FunctionPtr sgn = Function::sideParity();
      FunctionPtr n = Function::normal();
      FunctionPtr n_x = n->spatialComponent(1); // 1-based
      
      vlasovFunctionsToExportNames.push_back("f_i_in");
      vlasovFunctionsToExport.push_back(f_i_in);
      
      FunctionPtr tn_inflowExact = f_i_in * sgn * n_x * vx;
      const bool weightFluxesBySideParity = true;
      FunctionPtr tn_soln = Function::solution(tn_i, form.vlasovSolution(), weightFluxesBySideParity);
      bcFunctionsToResolve.push_back({tn_i,tn_inflowExact,tn_soln,inflowTag});
    }
    form.setInitialAndInflowConditions(f0, f_in, timeRampWidth, enforceInflowBCsWeakly);
    
    FunctionPtr Ex = Function::solution(poissonConfig->E_fieldVar(0), form.configurationSpaceSolution());
    
    FunctionPtr vxAcceleration = Ex + extraAccelerationFormula;
    double maxAcceleration_vx = vxAcceleration->maximumValue(poissonConfig->configurationSpaceMesh());
    if (rank == 0)
    {
      std::cout << "Max acceleration on initial solution is " << maxAcceleration_vx << std::endl;
    }
    //    double u_mesh = max(maxAcceleration_vx, v0 + vWidth);
    //    double h_min =  u_mesh * dt / CFL_max;
    int    p_max = H1Order;  // for now, no p refinements.  (May want to revisit; if we do, can set p_max = 10 - delta_k -- p is limited by Intrepid2 quadrature rules at the moment.)
    
    ErrorIndicatorPtr errorIndicator;
    if (!enforcePositivity)
    {
      errorIndicator = ErrorIndicator::energyErrorIndicator(form.vlasovSolution());
    }
    else
    {
      // want adaptivity to refine where the *original* problem has large residuals
      // the second equation (defining the latent variable) in particular may distort the error
      
      const int indexOffsetSpaceOneBased    = 1;
      const int indexOffsetVelocityOneBased = spaceDim + 1;
      
      LinearTermPtr residual;
      
      for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
      {
        FunctionPtr f_i = Function::solution(form.f(speciesOrdinal), form.vlasovSolution());
        VarPtr w_i  = form.w(speciesOrdinal);
        residual = residual + f_i * w_i;
      }
      
      std::vector<VarPtr> w(numSpecies);
      std::vector<FunctionPtr> t_n_hat_soln(numSpecies);
      std::vector<FunctionPtr> f_soln(numSpecies);
      std::vector<FunctionPtr> t_n_hat_soln_prev(numSpecies);
      std::vector<FunctionPtr> f_soln_prev(numSpecies);
      auto vlasovSolnWeakPtr = Teuchos::rcp(form.vlasovSolution().get(), false);
      auto vlasovSolnPriorTimeWeakPtr = Teuchos::rcp(form.vlasovSolutionPriorTime().get(), false);
      for (int i=0; i<numSpecies; i++)
      {
        w[i]            = form.w(i);
        f_soln[i]       = Function::solution(form.f(i), vlasovSolnWeakPtr);
        bool weightFluxesBySideParity = true;
        t_n_hat_soln[i] = Function::solution(form.t_n_hat(), vlasovSolnWeakPtr, weightFluxesBySideParity);
        
        f_soln_prev[i]  = Function::solution(form.f(i), vlasovSolnPriorTimeWeakPtr);
        t_n_hat_soln_prev[i] = Function::solution(form.t_n_hat(), vlasovSolnPriorTimeWeakPtr, weightFluxesBySideParity);
      }
      
      std::vector<LinearTermPtr> w_against_f_i(numSpecies);
      
      const double theta = useCrankNicolson ? 0.5 : 1.0;
      
      const bool l2Trace = true;
      for (int i=0; i<numSpecies; i++)
      {
        w_against_f_i[i] = w[i] / dt;
        
        residual = residual + t_n_hat_soln[i] * w[i];
        
        for (int d=0; d<spaceDim; d++)
        {
          auto v_d = form.v_i(d);
          
          w_against_f_i[i] = w_against_f_i[i] - theta * v_d * w[i]->di(indexOffsetSpaceOneBased + d); // dx_d
        }
      }
      
      for (int i=0; i<numSpecies; i++)
      {
        auto q_i = form.q(i);
        auto m_i = form.m(i);
        for (int d=0; d<velocityDim; d++)
        {
          auto weight = form.a()[d];
          w_against_f_i[i] = w_against_f_i[i] - theta * (q_i/m_i) * weight * w[i]->di(indexOffsetVelocityOneBased + d); // d/dv_d
        }
      }
      // now add in the f terms:
      for (int i=0; i<numSpecies; i++)
      {
        residual = residual + (f_soln[i] - f_soln_prev[i]) * w_against_f_i[i];
      }
      
      IPPtr ip = IP::ip();
      for (int i=0; i<numSpecies; i++)
      {
        ip->addTerm(w[i]);
        ip->addTerm(w_against_f_i[i]);
        
        ip->addTerm(form.phi(i)); // this term and its weight should not matter; the phi coefficients will always be 0, because they do not enter the residual.
      }
      
//      std::cout << "residual for error indicator: " << residual->displayString() << std::endl;
//
//      std::cout << "ip for error indicator:\n";
//      ip->printInteractions();
      
      // EnergyErrorIndicator(MeshPtr mesh, TLinearTermPtr<Scalar> residual, TIPPtr<Scalar> ip, int cubatureEnrichmentDegree = 0);
      errorIndicator = ErrorIndicator::energyErrorIndicator(form.vlasovMesh(), residual, ip, 0);
    }
    const int numElementsToRefine = std::max(int(.05 / 3. * maxElements), 1);
    RefinementStrategyPtr fixedCountRefStrategy = Teuchos::rcp( new FixedCountRefinementStrategy(errorIndicator, numElementsToRefine) );
        
    RefinementStrategyPtr refStrategy;
    if (useGreedyStrategy)
    {
      refStrategy = constructRefStrategy(form, adaptiveRefinementThreshold, maxElements, maximumRefinementLevel, Teuchos::null, maxP);
      refStrategy->setEnforceOneIrregularity(true);
    }
    else if (useAnisotropicCFLRefStrategy)
    {
      if (rank == 0) std::cout << "CFL: " << CFL_max << std::endl;
      refStrategy = Teuchos::rcp(new AnisotropicCFLRefinementStrategy(errorIndicator, adaptiveRefinementThreshold, poissonConfig->configurationSpaceMesh(), vx, vxAcceleration, CFL_max, dt, useGlobalCFLConstraint, maxP) ); // constructRefStrategy(form, refinementThreshold, maxElements, maximumRefinementLevel, Teuchos::null);
      refStrategy->setEnforceOneIrregularity(false); // anisotropic CFL strategy has its own approach to enforcing 1-irregularity
    }
    else
    {
      refStrategy = fixedCountRefStrategy;
    }
    
    RefinementStrategyPtr fluxLimitingRefStrategy;
    if (enforcePositivity)
    {
      ErrorIndicatorPtr fluxLimitingErrorIndicator =  Teuchos::rcp( new FluxLimitingErrorIndicator(form) );
       
      if (useAnisotropicCFLRefStrategy)
      {
        fluxLimitingRefStrategy = Teuchos::rcp(new AnisotropicCFLRefinementStrategy(fluxLimitingErrorIndicator, fluxLimitingRefinementThreshold, poissonConfig->configurationSpaceMesh(), vx, vxAcceleration, CFL_max, dt, useGlobalCFLConstraint, maxP) );
      }
      else
      {
        fluxLimitingRefStrategy = Teuchos::rcp( new RefinementStrategy( fluxLimitingErrorIndicator, fluxLimitingRefinementThreshold, 0, maxP, false));
        fluxLimitingRefStrategy->setMaximumRefinementLevel(maximumRefinementLevel);
        fluxLimitingRefStrategy->setEnforceOneIrregularity(true);
      }
    }
    
    const bool refineToResolveBCs = useAdaptivity;
    double bcResolutionTolerance = 1e-1;
    if (refineToResolveBCs)
    {
      for (auto bcTuple : bcFunctionsToResolve)
      {
        using std::get;
        VarPtr var = get<0>(bcTuple);
        FunctionPtr bcFxnExact = get<1>(bcTuple);
        FunctionPtr bcFxnSoln  = get<2>(bcTuple);
        int bcTag = get<3>(bcTuple);
        
        const std::string & tagName = DIRICHLET_SET_TAG_NAME;
        FunctionPtr bcRestriction = Function::meshSideSetCharacteristic(tagName, bcTag);
        
        // project onto vlasovSolution:
        std::map<int,FunctionPtr> bcMap;
        bcMap[var->ID()] = bcFxnExact;
        form.vlasovSolution()->projectOntoMesh(bcMap);
        
        FunctionPtr bc_err = bcFxnExact * bcRestriction - bcFxnSoln * bcRestriction;
        int quadratureDegree = p_max * 2 + delta_k;
        int quadratureEnrichment = (4 + quadratureDegree > Intrepid2::Parameters::MaxCubatureDegreeEdge) ? (Intrepid2::Parameters::MaxCubatureDegreeEdge - quadratureDegree) : 4;
        
        auto errorIndicator = ErrorIndicator::functionErrorIndicator<double>(form.vlasovMesh(), bc_err, quadratureEnrichment);
        
        double refinementThresholdForInitialData = 0.80;
        
        RefinementStrategyPtr refStrategy;
        if (useAnisotropicCFLRefStrategy)
        {
          refStrategy = Teuchos::rcp(new AnisotropicCFLRefinementStrategy(errorIndicator, refinementThresholdForInitialData, poissonConfig->configurationSpaceMesh(), vx, vxAcceleration, CFL_for_BCs, dt, useGlobalCFLConstraint, maxP) );
          refStrategy->setEnforceOneIrregularity(false);
        }
        else
        {
          refStrategy = Teuchos::rcp( new RefinementStrategy( errorIndicator, refinementThresholdForInitialData, 0, maxP, false));
          refStrategy->setMaximumRefinementLevel(maximumRefinementLevel);
          refStrategy->setEnforceOneIrregularity(true);
        }
        
        //      auto refStrategy = Teuchos::rcp(new TRefinementStrategy<double>(errorIndicator, refinementThresholdForInitialData));
        //      refStrategy->setMaxElements(maxElements);
        //      refStrategy->setMaxP(p_max);
        //      refStrategy->setMinH(h_min);
        
        double bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
        double relErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
        
        int numBCRefinements = 0;
        int numVlasovElements = form.vlasovMesh()->numActiveElements();
        int numGlobalDofs     = form.vlasovMesh()->globalDofCount();
        while ((relErr > bcResolutionTolerance) && (numVlasovElements < maxElements))
        {
          // recompute h_min
          //        maxAcceleration_vx = vxAcceleration->maximumValue(poissonConfig->configurationSpaceMesh());
          //        u_mesh = max(maxAcceleration_vx, v0 + vWidth);
          //        h_min =  u_mesh * dt / CFL_for_BCs;
          //        refStrategy->setMinH(h_min);
          
          int prevNumVlasovElements = numVlasovElements;
          int prevNumGlobalDofs = numGlobalDofs;
          
          if (rank==0)
          {
            std::cout << "After " << numBCRefinements++ << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
            std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
            //          std::cout << "h_min is " << h_min << std::endl;
          }
          
          refStrategy->refine();
          
          // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
          form.configurationSpaceMesh()->repartitionAndRebuild();
          
          // project the BC onto the mesh again
          form.vlasovSolution()->projectOntoMesh(bcMap);
          
          bc_norm = (bcFxnExact * bcRestriction)->l2norm(form.vlasovMesh(), quadratureEnrichment);
          relErr = bc_err->l2norm(form.vlasovMesh(), quadratureEnrichment) / bc_norm;
          
          numGlobalDofs = form.vlasovMesh()->globalDofCount();
          numVlasovElements = form.vlasovMesh()->numActiveElements();
          
          if ((numVlasovElements == prevNumVlasovElements) && (numGlobalDofs == prevNumGlobalDofs))
          {
            if (rank == 0)
            {
              std::cout << "No refinements performed on this pass; exiting refinement loop early.\n";
            }
            break;
          }
        }
        if (rank==0)
        {
          std::cout << "After " << numBCRefinements << " refinements, phase space mesh has " << form.vlasovMesh()->numActiveElements() << " elements and BC's relErr is " << relErr << ".\n";
          std::cout << "Physical mesh has " << form.configurationSpaceMesh()->numActiveElements() << " elements.\n";
        }
      }
  }
//    form.vlasovMesh()->getTopology()->printAllEntitiesInBaseMeshTopology();
    GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(form.vlasovMesh()->globalDofAssignment().get());
//    std::cout << "dof indices for cell 1024:\n";
//    minRule->getGlobalDofIndices(1024).print(std::cout);
//    
//    std::cout << "dof indices for cell 992:\n";
//    minRule->getGlobalDofIndices(992).print(std::cout);
    
    std::ostringstream suffix;
    suffix << "_UR" << numUniformRefinements; // This is mainly for distinguishing between cases while we diagnose a possible bug in refinements
    if (useAdaptivity)
    {
      suffix << "_budget" << maxElements;
      suffix << "_cfl" << CFL_max;
      if (CFL_max != CFL_for_BCs) suffix << "_cfl_bc" << CFL_for_BCs;
    }
    suffix << "_dt" << dt_max;
    suffix << "_H1Order" << H1Order;
    suffix << "_delta_k" << delta_k;
    suffix << "_gap" << physicalGapWidth;
    suffix << "_" << beamIntensity_eV << "eV";
    suffix << "_" << numTimeSteps << "_timeSteps";
    if (includeElectrons)
    {
      suffix << "_eTherm" << electronThermalVelocity;
    }
    if (includeIons)
    {
      suffix << "_iTherm" << ionThermalVelocity;
    }
    if (enforcePositivity)
    {
      suffix << "_proximal";
      suffix << "_alpha0" << alpha0;
      suffix << "_mult" << alphaMultiplier;
    }

    // adjust x and v element counts according to the number of uniform refinements requested.
    xElements <<= numUniformRefinements;
    vElements <<= numUniformRefinements;

    suffix << "_" << xElements << "x" << vElements;
    if (useCrankNicolson)
    {
      suffix << "_CN";
    }
    suffix << "_I_IC_" << criticalCurrentRatio;
    suffix << "_" << normChoice << "Norm";

    std::string problemPrefix = "Ions";
    
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
    const std::string directoryPath = outputDir;
    
    std::ofstream refReportFileStream(refReportFilePath, std::ios::out);
    
    refReportFileStream << "Vlasov Elements\tVlasov Dofs\tPoisson Elements\tPoisson Dofs\tEnergy Error\n";
    refReportFileStream.close();
    
    auto writeTimeStepReport = [&](const int numElementsVlasov, const int numDofsVlasov, const int numElementsPoisson, const int numDofsPoisson, const double energyError) {
      auto refReportFilePath = getTimeStepReportFilePath();
      std::ofstream refReportFileStream(refReportFilePath, std::ios_base::app);
      refReportFileStream << numElementsVlasov << "\t" << numDofsVlasov << "\t" << numElementsPoisson << "\t" << numDofsPoisson << "\t" << energyError << std::endl;
      refReportFileStream.close();
    };
    
    auto reportTimeStepComplete = [&](const int &timeStepNumber, const double &energyError) {
      auto numVlasovDofs       = form.vlasovMesh()->numGlobalDofs();
      auto vlasovElementCount  = form.vlasovMesh()->numActiveElements();
      auto poissonElementCount = form.configurationSpaceMesh()->numActiveElements();
      auto poissonDofCount     = form.configurationSpaceMesh()->numGlobalDofs();
      
      if (rank == 0)
      {
        writeTimeStepReport(vlasovElementCount, numVlasovDofs, poissonElementCount, poissonDofCount, energyError);
        std::cout << "Time step " << timeStepNumber << " complete.\n";
      }
    };
    
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
      
      if (rank == 0)
      {
        std::cout << "Visualizations will be placed in directory " << directoryPath << " under subdirectories:\n";
        std::cout <<  vlasovSS.str() << std::endl;
        std::cout << poissonSS.str() << std::endl;
      }
    }
    
    const int num1DPointsPoisson = maxP * 2; // default is 4
    const int num1DPointsVlasov  = maxP * 2; // default is 4
    
    {
      FunctionPtr energyErrorFunction = EnergyErrorFunction::energyErrorFunction(form.vlasovSolution());
      vlasovFunctionsToExport.push_back(energyErrorFunction);
      vlasovFunctionsToExportNames.push_back("Vlasov energy error");
      
      FunctionPtr cellIDFunction = Teuchos::rcp(new CellIDFunction);
      vlasovFunctionsToExport.push_back(cellIDFunction);
      vlasovFunctionsToExportNames.push_back("Vlasov cell ID");
      
      FunctionPtr cellPolyOrderFunction = Teuchos::rcp(new CellPolyOrderFunction);
      vlasovFunctionsToExport.push_back(cellPolyOrderFunction);
      vlasovFunctionsToExportNames.push_back("Vlasov poly order");
      
      poissonFunctionsToExport.push_back(cellPolyOrderFunction);
      poissonFunctionsToExportNames.push_back("Poisson poly order");
    }
    
    double t = 0.0;
    if (saveViz)
    {
      poissonViz->exportSolution(form.configurationSpaceSolution(),poissonFunctionsToExport,poissonFunctionsToExportNames,t,num1DPointsPoisson);
      vlasovViz->exportSolution(form.vlasovSolution(),vlasovFunctionsToExport,vlasovFunctionsToExportNames,t,num1DPointsVlasov);
    }
    
    FunctionPtr E_soln = Function::solution(form.configurationSpaceFormulation()->E_fieldVar(0), form.configurationSpaceSolution());
            
    auto solutionMoments = form.solutionMoments();
    auto density  = solutionMoments->solutionMomentFunction(0);
    auto momentum = solutionMoments->solutionMomentFunction(1); // aka current
    auto velocity = momentum / density;
            
    bool previousSolveUsedGeometricMultigrid = false;
    bool previousSolveUsedDirect = false;
    
    bool emFieldIsOn = (t >= emFieldStartTime);
    
//    refStrategy->setMinH(h_min);
//    refStrategy->setMaxP(p_max);
//
//    auto solnRep = RieszRep::rieszRep(form.vlasovMesh(), form.vlasovSolution()->ip(), form.vlasovSolution()->bf()->testFunctional(form.vlasovSolution()));
    
    // set up a residual to measure that is independent of whether we
    // are doing proximal Galerkin or not
    using std::map;
    using std::set;
    map<int, FunctionPtr> solnMap;
    set<int> testVarIDsToExclude;
    LinearTermPtr rhsTerm;
    for (int i=0; i<numSpecies; i++)
    {
      SolutionPtr vlasovSoln     = form.vlasovSolution();
      VarPtr f_i  = form.f(i);
      VarPtr tn_i = form.t_n_hat(i);
      solnMap[f_i->ID()]  = Function::solution(f_i,  form.vlasovSolution());
      const bool weightFluxesBySideParity = true;
      solnMap[tn_i->ID()] = Function::solution(tn_i, form.vlasovSolution(), weightFluxesBySideParity);
      
      VarPtr w_i = form.w(i);
      rhsTerm = rhsTerm + form.vlasovSolution()->rhs()->linearTerm()->getPartMatchingVariable(w_i);
      if (enforcePositivity)
      {
        // the test variable for the extra equation in proximal Galerkin is phi_i
        VarPtr phi_i = form.phi(i);
        testVarIDsToExclude.insert(phi_i->ID());
        
        // the rhs term will have included psi_prev / alpha * w_i, so we should subtract that off: don't want psi or alpha to enter into it.
        VarPtr psi_i = form.psi(i);
        SolutionPtr vlasovSolnPrev = form.vlasovSolutionPrevious();
        FunctionPtr psi_solnPrev = Function::solution(psi_i, vlasovSolnPrev);
        FunctionPtr alpha        = form.alpha();
        rhsTerm->addTerm(-(psi_solnPrev / alpha) * w_i);
      }
    }
    auto lhsTerm = form.vlasovSolution()->bf()->testFunctional(solnMap, testVarIDsToExclude);
    auto residual = lhsTerm - rhsTerm;
    
    double solnNorm = RieszRep::rieszRep(form.vlasovMesh(), form.vlasovSolution()->ip(), lhsTerm)->getNorm();
    if (rank == 0)
    {
      std::cout << "Initial solution, energy norm: " << solnNorm << std::endl;
    }
    
    {
      poissonFunctionsToExport.push_back(density);
      poissonFunctionsToExportNames.push_back("n_e");
      
      poissonFunctionsToExport.push_back(velocity);
      poissonFunctionsToExportNames.push_back("v");
    }
    
    for (int timeStepNumber=1; timeStepNumber<=numTimeSteps; timeStepNumber++)
    {
      setUseGMG(form, maxDofsForDirectSolve, previousSolveUsedGeometricMultigrid, previousSolveUsedDirect, rank == 0);
      
      form.startTimeStep();

      // refresh voltage value, which may be ramped, and depend on the current time.
      voltageParameter->setValue(voltageValue * timeRampFunction->value());
      extraAccelerationParameter->setValue(extraAccelerationFormula * timeRampFunction->value());
      
      if (useAdaptivity)
      {
        int maxRefinementSteps = 1; // just do 1 pass on every time step after the first
        int minElements = 0;   // don't gate on the element count except for the first time step
        bool meshChanged = false; // track whether the mesh has changed since the last call to form.finishTimeStep() -- if it has, we should call again, to re-finish it.  (Time gets advanced in startTimeStep().)
        for (int refNumber=0; refNumber<maxRefinementSteps; refNumber++)
        {
          form.finishTimeStep();
          
          maxAcceleration_vx = vxAcceleration->maximumValue(poissonConfig->configurationSpaceMesh());
          
          // recompute h_min
//          u_mesh = max(maxAcceleration_vx, v0 + vWidth);
//          h_min =  u_mesh * dt / CFL_max;
//          refStrategy->setMinH(h_min);
          
          if (performRefinements(form, refStrategy, unrefinementStrategy, maxElements) )
          {
            meshChanged = true;
          }
          
          if (enforcePositivity)
          {
            if (meshChanged)
            {
              form.finishTimeStep();
              meshChanged = false;
            }
            
            if ( performRefinements(form, fluxLimitingRefStrategy, unrefinementStrategy, maxElements) )
            {
              meshChanged = true;
            }
          }
          
          if (greedyUnrefinementFrequency > 0)
          {
            if ((timeStepNumber % greedyUnrefinementFrequency) == 0)
            {
              int numUnrefinementsThisPass = 0;
              do
              {
                if (meshChanged)
                {
//                  {
//                    // DEBUGGING: output viz at (t+dt) to allow diagnosis when we fail to solve in finishTimeStep() call
//                    solutionMoments->update(); // update to get latest density, velocity
//                    form.configurationSpaceSolve();
//                    form.configurationSpaceFormulation()->updateConfigurationSpaceFunctions();
//                    
//                    auto Ex_extruded = form.configurationSpaceFormulation()->E_i(0);
//                    
//                    auto phaseSpaceFunctionNames = vlasovFunctionsToExportNames;
//                    phaseSpaceFunctionNames.push_back("Ex_extruded");
//                    auto phaseSpaceFunctions     = vlasovFunctionsToExport;
//                    phaseSpaceFunctions.push_back(Ex_extruded);
//                    
//                    vlasovViz->exportSolution(form.vlasovSolution(),phaseSpaceFunctions,phaseSpaceFunctionNames,t+dt,num1DPointsVlasov);
//                    poissonViz->exportSolution(form.configurationSpaceSolution(),poissonFunctionsToExport,poissonFunctionsToExportNames,t+dt,num1DPointsPoisson);
//                  }
                  // solve again so we can legitimately measure error again
                  form.finishTimeStep();
                  meshChanged = false;
                }
                
                bool printToConsole = (form.vlasovMesh()->Comm()->MyPID() == 0);
                numUnrefinementsThisPass = unrefinementStrategy->greedyUnrefinement(greedyUnrefinementThreshold, printToConsole);
                
                if (numUnrefinementsThisPass > 0)
                {
                  meshChanged = true;
                  
                  // need to repartition and rebuild the physical mesh now (RefStrategy was repartitioning and rebuilding phase space mesh without giving physical mesh a chance to do so).
                  form.configurationSpaceMesh()->repartitionAndRebuild();
                  
                  // also reconstruct GMG solver, if one is in use:
                  form.reconstructGMGSolver();
                }
              } while (numUnrefinementsThisPass > 0);
            }
          }
          
          setUseGMG(form, maxDofsForDirectSolve, previousSolveUsedGeometricMultigrid, previousSolveUsedDirect, rank == 0);
          
          if (form.vlasovMesh()->numActiveElements() >= minElements)
          {
            break;
          }
        }
        if (meshChanged)
        {
          // then solve again
          form.finishTimeStep();
        }
      }
      else
      {
        form.finishTimeStep();
      }
      
      t += dt;
      
      std::vector<double> f_disagreement(numSpecies);
      for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
      {
        double min_f_at_extrema = fAtExtrema[speciesOrdinal]->minimumValue(form.vlasovMesh());
        double max_f_at_extrema = fAtExtrema[speciesOrdinal]->maximumValue(form.vlasovMesh());
        f_disagreement[speciesOrdinal] = std::max(std::abs(min_f_at_extrema), std::abs(max_f_at_extrema));
      }
      
      solnNorm = RieszRep::rieszRep(form.vlasovMesh(), form.vlasovSolution()->ip(), lhsTerm)->getNorm();
      double totalEnergyError = RieszRep::rieszRep(form.vlasovMesh(), form.vlasovSolution()->ip(), residual)->getNorm();
      if (rank == 0)
      {
        double relErr = (solnNorm > 0) ? totalEnergyError / solnNorm : 1.0;
        std::cout << "Solution's energy norm: " << solnNorm << std::endl;
        std::cout << "Relative energy error: " << relErr << std::endl;
        for (int speciesOrdinal=0; speciesOrdinal<numSpecies; speciesOrdinal++)
        {
          std::cout << "Species " << speciesOrdinal << ", max disagreement with homogeneous BCs at velocity extrema: " << f_disagreement[speciesOrdinal] << std::endl;
        }
      }
      
      if (rank == 0)
      {
        std::cout << "Max acceleration after " << timeStepNumber << " time steps is " << maxAcceleration_vx << std::endl;
//        std::cout << "h_min: " << h_min << std::endl;
      }
      
      if (!emFieldIsOn)
      {
        // check if it should turn on now
        emFieldIsOn = (t >= emFieldStartTime);
        if (emFieldIsOn)
        {
          poissonConfig->solution()->setRHS(poissonRHS);
        }
      }
      
      {
        // compute, report BC error for inflow:
        const double relErr = inflowErrorOnBoundary(form, f_in, form.tagVlasovSpatialInflow(), p_max);
        if (rank == 0)
        {
          std::cout << "Relative L^2 error in f compared to f_in at inflow: " << relErr << std::endl;
        }
      }
      
      moments_kp1->update();
      if (rank == 0) std::cout << "moments_kp1 updated.\n";
      moments_k->update();
      if (rank == 0) std::cout << "moments_k   updated.\n";
      
      if ((timeStepNumber % outputFrequency == 0) || (timeStepNumber == numTimeSteps))
      {
        if (saveViz)
        {
          solutionMoments->update(); // update to get latest density, velocity
          vlasovViz->exportSolution(form.vlasovSolution(),vlasovFunctionsToExport,vlasovFunctionsToExportNames,t,num1DPointsVlasov);
          poissonViz->exportSolution(form.configurationSpaceSolution(),poissonFunctionsToExport,poissonFunctionsToExportNames,t,num1DPointsPoisson);
          
//          FunctionPtr E_x = Function::solution(poissonConfig->E_fieldVar(0), form.configurationSpaceSolution());
//          poissonViz->exportFunction({"E_x"}, {E_x});
        }
        
        reportTimeStepComplete(timeStepNumber,totalEnergyError);
      }
    } // end time step loop
        
    FunctionPtr potential = Function::solution(form.configurationSpaceFormulation()->VE_fieldVar(), form.configurationSpaceSolution());
        
    // output to results file:
    bool outputResultsFile = false; // turn off until we can fix an issue re. different elem types in physical mesh.
    if (outputResultsFile)
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
        
        // TODO: partition physicalCellIDs according to element type.  C.f. PhysicalSolutionExtrusion.
        
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
        
        const int numCells = physicalCellIDsVector.size();
        
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
          const auto ne          = ne_values[pointEntry.first]        * n0; // TODO: determine if this scaling is correct; we probably need to divide by spatialScaling^3.
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
