//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  Boltzmann1D1VFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/3/2020.
//
//

#include "Boltzmann1D1VFormulation.h"
#include "CamelliaMemoryTools.h"
#include "CubatureFactory.h"
#include "ERKTableau.h"
#include "ExtrudedMeshTopology.h"
#include "GlobalDofAssignment.h"
#include "GMGSolver.h"
#include "MeshFactory.h"
#include "ParameterFunction.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "SolutionStage.h"
#include "TypeDefs.h"

#include "CamelliaDataHelpers.h"

using namespace Camellia;

const string Boltzmann1D1VFormulation::S_F = "f";
const string Boltzmann1D1VFormulation::S_E = "E";

const string Boltzmann1D1VFormulation::S_F_HAT = "\\widehat{f}";
const string Boltzmann1D1VFormulation::S_T_N_HAT = "\\widehat{t}_n";
const string Boltzmann1D1VFormulation::S_E_HAT = "\\widehat{E}";

const string Boltzmann1D1VFormulation::S_W = "w";
const string Boltzmann1D1VFormulation::S_TAU = "\\tau";

/*
 Worth noting that VelocitySpaceExtrusion codifies an inefficient use of memory:
 it is simply duplicating values in the velocity dimension.  It would be better to
 avoid the duplication by adding some facilities to Function; I like the idea of
 adding a method
    bool variesInDimension(int dim);
 and then having some mechanism by which the arrays that Functions that are constant
 in some dimension can only specify values for the dimensions in which they vary.
 Doing this more or less requires reworking the way points are stored, though, to
 express tensor-product points.  We have some loose plans to do something like this
 in Intrepid2.
 */
class VelocitySpaceExtrusion : public TFunction<double>
{
  FunctionPtr _g_1D; // arbitrary function in physical space -- can depend on mesh/physical solution
  
  Teuchos::RCP<PhasePhysicalMeshMap> _phaseSpacePhysicalMeshMap;
public:
  VelocitySpaceExtrusion( FunctionPtr g_1D, Teuchos::RCP<PhasePhysicalMeshMap> phaseSpacePhysicalMeshMap)
  :
  Function("VelocitySpaceExtrusion",g_1D->rank(),VARIES_IN_DIMENSION_0),
  _g_1D(g_1D),
  _phaseSpacePhysicalMeshMap(phaseSpacePhysicalMeshMap)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_g_1D->rank() != 0, std::invalid_argument, "Only scalar-valued functions supported right now");
  }
  
  void updateValues()
  {
    auto & phaseSpaceCellIDs = _phaseSpacePhysicalMeshMap->phaseMesh()->cellIDsInPartition();
    
    const set<GlobalIndexType> physicalCellIDs = _phaseSpacePhysicalMeshMap->physicalCellIDsForPhaseSpaceCells(phaseSpaceCellIDs);
    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalValuesMap;
    
    const std::vector<GlobalIndexType> physicalCellIDsVector(physicalCellIDs.begin(), physicalCellIDs.end());
    _g_1D->importCellData(physicalCellIDsVector);
  }
  
  using TFunction<double>::values;
  void values(Intrepid2::Data<double, DeviceType> values, BasisCachePtr basisCache) override
  {
    // there are two use cases to consider:
    // 1. fully tensor-product cubature points (non-trivial PhysicalTensorPoints)
    // 2. cubature points without a specified tensor-product structure
    //
    // (1) is the principal use case.  (2) is a holdover, just there until we can upgrade BasisCache to manage tensor points on sides…
    
    // if we are in use case (1), there will be 2 tensor components in physical cubature points; otherwise, there will be just 1
    const bool dataExtentMatchesPhysicalPoints = basisCache->getPhysicalCubaturePoints().numTensorComponents() == 2; // true for use case (1)
    
    // the code that actually fills in values below assumes that values has shape (C,P1,P2) in case (1), and shape (C,P) in use case 2
    const int numPoints = basisCache->getPhysicalCubaturePoints().extent_int(1);
    if (dataExtentMatchesPhysicalPoints)
    {
      const int numPhysicalPoints = basisCache->getPhysicalCubaturePoints().getTensorComponent(0).extent_int(1);
      TEUCHOS_TEST_FOR_EXCEPTION(numPhysicalPoints != values.extent_int(1), std::invalid_argument, "numPhysicalPoints should match the extent in dimension 1 of values");
      const int numVelocityPoints = numPoints / numPhysicalPoints;
      TEUCHOS_TEST_FOR_EXCEPTION(numVelocityPoints != values.extent_int(2), std::invalid_argument, "numVelocityPoints should match the extent in dimension 2 of values");
      TEUCHOS_TEST_FOR_EXCEPTION(values.getVariationModulus(2) != 1, std::invalid_argument, "values should not vary in velocity dimension");
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(numPoints != values.extent_int(1), std::invalid_argument, "numPoints should match the extent in dimension 1 of values");
    }
    
    if (dataExtentMatchesPhysicalPoints)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 3, std::invalid_argument, "Expect values to have shape (C,P0,P1)");
      int extrusionExtentCount = 1;
      for (int dataIndex=2; dataIndex<values.rank(); dataIndex++)
      {
        extrusionExtentCount *= values.getDataExtent(dataIndex);
      }
      TEUCHOS_TEST_FOR_EXCEPTION(extrusionExtentCount != 1, std::invalid_argument, "Data container has more than the expected extent in extrusion dimensions");
    }

    // implementation adapted from the DeviceView values() implementation below
    
    using HostExecSpace = Kokkos::HostSpace::execution_space;
    Intrepid2::Data<double,HostExecSpace> valuesHost(values);
    
    // at present, we require that this be called on every MPI rank that owns cells in the phase space mesh
    // we could avoid this requirement by adding an explicit, separate communication step.
    
    // It is also worth noting that creating basisCache_1D below is an inefficiency: those basis values won't
    // be reused beyond the body of this method…
    
    bool isSideCache = (basisCache->getSideIndex() != -1);
    int physicalSpaceSideOrdinal;
    if (!isSideCache)
    {
      physicalSpaceSideOrdinal = -1;
    }
    else // isSideCache == true
    {
      const int phaseSpaceSideOrdinal = basisCache->getSideIndex();
      
      // what follows assumes 1D1V with line x line elements; let's throw an exception if we're not on such an element
      auto line_x_line = CellTopology::cellTopology(CellTopology::line(), 1);
      TEUCHOS_TEST_FOR_EXCEPTION(basisCache->cellTopology()->getKey() != line_x_line->getKey(), std::invalid_argument, "Only phase-space line x line topologies are supported");
      bool sideIsParallelToPhysicalSpace = !line_x_line->sideIsSpatial(phaseSpaceSideOrdinal);
      
      if (sideIsParallelToPhysicalSpace)
      {
        // this side does not map to a physical side; runs parallel to physical space
        // therefore, it instead maps to a physical cell interior (in adaptive context, possibly multiple -- this is not yet supported)
        if (_g_1D->boundaryValueOnly())
        {
          // if _g_1D is a skeleton variable, then we can't evaluate it on a cell interior; we define the extrusion to be zero in this case
          Kokkos::deep_copy(values.getUnderlyingView(), 0.0);
          return;
        }
        else
        {
          // we can and should evaluate _g_1D on the spatial element interior
          physicalSpaceSideOrdinal = -1;
        }
      }
      else
      {
        physicalSpaceSideOrdinal = line_x_line->getSpatialComponentSideOrdinal(phaseSpaceSideOrdinal);
      }
    }
    
    using std::map;
    using std::pair;
    using std::set;
    using std::vector;
    
    // determine physical cell IDs we need to compute on
    const vector<GlobalIndexType> & phaseSpaceCellIDs = basisCache->cellIDs();
    const vector<GlobalIndexType> physicalCellIDs = _phaseSpacePhysicalMeshMap->physicalCellIDsForPhaseSpaceCells(phaseSpaceCellIDs);
    
    using PhysicalValuesHost = Kokkos::DynRankView<double, Kokkos::HostSpace>;
    map<GlobalIndexType, PhysicalValuesHost> physicalValuesMapHost;
    map<GlobalIndexType, DeviceView<double> > physicalValuesMap;
//    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalPointsMap; // this is used for debugging; we could save something by dropping this later.
    
    const bool testVsTest = basisCache->testVsTest();
    const int phaseSpaceQuadratureEnrichment = basisCache->cubatureEnrichmentDegree();

    int basisCache1DQuadratureEnrichment;
    if (testVsTest)
    {
      // test order in both 1D and 2D meshes is relative to the H^1 order, which will match in both
      basisCache1DQuadratureEnrichment = phaseSpaceQuadratureEnrichment;
    }
    else
    {
      // trial order is 1 higher in 2D mesh because the t_n_hat/f_hat discretization has order matching H1Order
      basisCache1DQuadratureEnrichment = phaseSpaceQuadratureEnrichment + 1;
    }
    
    map< GlobalIndexType , pair<BasisCachePtr, vector<int> > > physicalBasisCachePairs; // index: physical cell ID
    for ( auto & physicalCellID : physicalCellIDs )
    {
      physicalBasisCachePairs[physicalCellID] = _phaseSpacePhysicalMeshMap->physicalCellBasisCache(physicalCellID, testVsTest, basisCache1DQuadratureEnrichment);
    }
    
    const bool inPhysicalVolume = (physicalSpaceSideOrdinal == -1);
    for ( auto & physicalCellID : physicalCellIDs )
    {
      auto basisCachePair = physicalBasisCachePairs[physicalCellID];
      auto physicalBasisCache = basisCachePair.first;
      auto offsets            = basisCachePair.second;
      
      BasisCachePtr basisCacheToUse = inPhysicalVolume ? physicalBasisCache : physicalBasisCache->getSideBasisCache(physicalSpaceSideOrdinal);
      
      const int numCells  = 1;
      const int numPoints = basisCacheToUse->getRefCellPointsFC().dimension(0);
      
      auto physicalValuesView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numPoints}, "VelocitySpaceExtrusion1D1V physicalValuesView");
      
      _g_1D->values(physicalValuesView, ASSIGN, basisCacheToUse);
      physicalValuesMap[physicalCellID] = physicalValuesView;
      
      PhysicalValuesHost physicalValuesViewHost = Kokkos::create_mirror_view_and_copy(HostExecSpace::memory_space(), physicalValuesView);
      physicalValuesMapHost[physicalCellID] = physicalValuesViewHost;
    }

    if (phaseSpaceCellIDs.size() == 0)
    {
      return;
    }
 
    const int numPhaseSpacePoints = basisCache->getRefCellPointsFC().dimension(0);
        
    for (int phaseSpaceCellOrdinal = 0; phaseSpaceCellOrdinal < phaseSpaceCellIDs.size(); phaseSpaceCellOrdinal++)
    {
      auto physicalPointIterator = _phaseSpacePhysicalMeshMap->getPhysicalPointIterator(basisCache, phaseSpaceCellIDs[phaseSpaceCellOrdinal], physicalBasisCachePairs);
      
      int numPhysicalPoints = physicalPointIterator.numPhysicalPoints();
      const int numVelocityPoints   = numPhaseSpacePoints / numPhysicalPoints;
      
      TEUCHOS_TEST_FOR_EXCEPTION(numVelocityPoints * numPhysicalPoints != numPhaseSpacePoints, std::invalid_argument, "physical points do not evenly divide phase space points");
      
      // For the moment, we do the copying below on host; it'll take a bit of effort to manage the maps, etc. efficiently on device.
      // (It seems likely that this will not be a performance bottleneck.)
      for (; physicalPointIterator.isValid(); physicalPointIterator.increment() )
      {
        auto physicalCellID             = physicalPointIterator.physicalCellID();
        auto & values1D                 = physicalValuesMapHost[physicalCellID];
        auto physicalPointOrdinal       = physicalPointIterator.pointOrdinal();
        auto pointOrdinalInPhysicalCell = physicalPointIterator.pointOrdinalInPhysicalCell();
        double & value                  = values1D(0, pointOrdinalInPhysicalCell);
        
        if (dataExtentMatchesPhysicalPoints)
        {
          // then we only need to fill in one entry
          valuesHost.getWritableEntry(phaseSpaceCellOrdinal, physicalPointOrdinal) = value;
        }
        else
        {
          // we need to iterate over all the velocity-space points
          for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
          {
            const int tensorPointOrdinal = getTensorPointOrdinal(physicalPointOrdinal, velocityPointOrdinal, numPhysicalPoints);
            valuesHost.getWritableEntry(phaseSpaceCellOrdinal, tensorPointOrdinal) = value;
          }
        }
      }
    }
     
    // check the views we checked out back in
    for ( auto & physicalCellID : physicalCellIDs )
    {
      SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(physicalValuesMap[physicalCellID]);
    }
    
    // copy from host to device
    // because layouts may differ, we use a copy mechanism built into Intrepid2::Data (which may be inefficient)
    // (we could speed things up by ensuring that layouts match, or at least checking whether they do, and doing
    //  a deep_copy if they do), though I'm inclined to doubt this will be a performance bottleneck…
    copyHostDataToDevice(values, valuesHost);
  }
  
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    // at present, we require that this be called on every MPI rank that owns cells in the phase space mesh
    // we could avoid this requirement by adding an explicit, separate communication step.
    
    // It is also worth noting that creating basisCache_1D below is an inefficiency: those basis values won't
    // be reused beyond the body of this method…
    
    const int rank = MPIWrapper::CommWorld()->MyPID();
    
    TEUCHOS_TEST_FOR_EXCEPTION(basisCache == Teuchos::null, std::invalid_argument, "On rank " + std::to_string(rank) + ", null pointer on line " + std::to_string(__LINE__) );
    TEUCHOS_TEST_FOR_EXCEPTION(_g_1D == Teuchos::null, std::invalid_argument, "On rank " + std::to_string(rank) + ", null pointer on line " + std::to_string(__LINE__) );
    TEUCHOS_TEST_FOR_EXCEPTION(_phaseSpacePhysicalMeshMap == Teuchos::null, std::invalid_argument, "On rank " + std::to_string(rank) + ", null pointer on line " + std::to_string(__LINE__) );
    
    bool isSideCache = (basisCache->getSideIndex() != -1);
    int physicalSpaceSideOrdinal;
    if (!isSideCache)
    {
      physicalSpaceSideOrdinal = -1;
    }
    else // isSideCache == true
    {
      const int phaseSpaceSideOrdinal = basisCache->getSideIndex();
      // what follows assumes 1D1V with line x line elements; let's throw an exception if we're not on such an element
      auto line_x_line = CellTopology::cellTopology(CellTopology::line(), 1);
      TEUCHOS_TEST_FOR_EXCEPTION(basisCache->cellTopology()->getKey() != line_x_line->getKey(), std::invalid_argument, "Only phase-space line x line topologies are supported");
      bool sideIsParallelToPhysicalSpace = !line_x_line->sideIsSpatial(phaseSpaceSideOrdinal);
      if (sideIsParallelToPhysicalSpace)
      {
        // this side does not map to a physical side; runs parallel to physical space
        // therefore, it instead maps to a physical cell interior (in adaptive context, possibly multiple -- this is not yet supported)
        if (_g_1D->boundaryValueOnly())
        {
          // if _g_1D is a skeleton variable, then we can't evaluate it on a cell interior; we define the extrusion to be zero in this case
          values.initialize(0.0);
          return;
        }
        else
        {
          // we can and should evaluate _g_1D on the spatial element interior
          physicalSpaceSideOrdinal = -1;
        }
      }
      else
      {
        physicalSpaceSideOrdinal = line_x_line->getSpatialComponentSideOrdinal(phaseSpaceSideOrdinal);
      }
    }
    
    using std::map;
    using std::pair;
    using std::set;
    using std::vector;
    
    // determine physical cell IDs we need to compute on
    const vector<GlobalIndexType> & phaseSpaceCellIDs = basisCache->cellIDs();
    const vector<GlobalIndexType> physicalCellIDs = _phaseSpacePhysicalMeshMap->physicalCellIDsForPhaseSpaceCells(phaseSpaceCellIDs);
    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalValuesMap;
    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalPointsMap; // this is used for debugging; we could save something by dropping this later.
    
    const bool testVsTest = basisCache->testVsTest();
    const int phaseSpaceQuadratureEnrichment = basisCache->cubatureEnrichmentDegree();

    int basisCache1DQuadratureEnrichment;
    if (testVsTest)
    {
      // test order in both 1D and 2D meshes is relative to the H^1 order, which will match in both
      basisCache1DQuadratureEnrichment = phaseSpaceQuadratureEnrichment;
    }
    else
    {
      // trial order is 1 higher in 2D mesh because the t_n_hat/f_hat discretization has order matching H1Order
      basisCache1DQuadratureEnrichment = phaseSpaceQuadratureEnrichment + 1;
    }
    
    map< GlobalIndexType , pair<BasisCachePtr, vector<int> > > physicalBasisCachePairs; // index: physical cell ID
    for ( auto & physicalCellID : physicalCellIDs )
    {
      physicalBasisCachePairs[physicalCellID] = _phaseSpacePhysicalMeshMap->physicalCellBasisCache(physicalCellID, testVsTest, basisCache1DQuadratureEnrichment);
    }
    
    const bool inPhysicalVolume = (physicalSpaceSideOrdinal == -1);
    for ( auto & physicalCellID : physicalCellIDs )
    {
      auto basisCachePair = physicalBasisCachePairs[physicalCellID];
      auto physicalBasisCache = basisCachePair.first;
      auto offsets            = basisCachePair.second;
      
      BasisCachePtr basisCacheToUse = inPhysicalVolume ? physicalBasisCache : physicalBasisCache->getSideBasisCache(physicalSpaceSideOrdinal);
      
      TEUCHOS_TEST_FOR_EXCEPTION(basisCacheToUse == Teuchos::null, std::invalid_argument, "On rank " + std::to_string(rank) + ", null pointer on line " + std::to_string(__LINE__) );
      
      const int numCells  = 1;
      const int numPoints = basisCacheToUse->getRefCellPointsFC().dimension(0);
      Camellia::FieldContainer<double> values(numCells,numPoints);
      
      _g_1D->values(values, basisCacheToUse);
      physicalValuesMap[physicalCellID] = values;
      physicalPointsMap[physicalCellID] = basisCacheToUse->getPhysicalCubaturePointsFC();
    }

    if (phaseSpaceCellIDs.size() == 0)
    {
      return;
    }
 
    const int numPhaseSpacePoints = basisCache->getRefCellPointsFC().dimension(0);
    
    for (int phaseSpaceCellOrdinal = 0; phaseSpaceCellOrdinal < phaseSpaceCellIDs.size(); phaseSpaceCellOrdinal++)
    {
      auto physicalPointIterator = _phaseSpacePhysicalMeshMap->getPhysicalPointIterator(basisCache, phaseSpaceCellIDs[phaseSpaceCellOrdinal], physicalBasisCachePairs);
      
      int numPhysicalPoints = physicalPointIterator.numPhysicalPoints();
      const int numVelocityPoints   = numPhaseSpacePoints / numPhysicalPoints;
      
      TEUCHOS_TEST_FOR_EXCEPTION(numVelocityPoints * numPhysicalPoints != numPhaseSpacePoints, std::invalid_argument, "physical points do not evenly divide phase space points");
      
      for (; physicalPointIterator.isValid(); physicalPointIterator.increment() )
      {
        auto physicalCellID             = physicalPointIterator.physicalCellID();
        auto & values1D                 = physicalValuesMap[physicalCellID];
        auto physicalPointOrdinal       = physicalPointIterator.pointOrdinal();
        auto pointOrdinalInPhysicalCell = physicalPointIterator.pointOrdinalInPhysicalCell();
        double & value                  = values1D(0, pointOrdinalInPhysicalCell);
        
        for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
        {
          const int tensorPointOrdinal = getTensorPointOrdinal(physicalPointOrdinal, velocityPointOrdinal, numPhysicalPoints);
          values(phaseSpaceCellOrdinal,tensorPointOrdinal) = value;
        }
      }
    }
  }
  
  bool boundaryValueOnly() override
  {
    return _g_1D->boundaryValueOnly();
  }
};

Boltzmann1D1VFormulation::Boltzmann1D1VFormulation(Boltzmann1D1VFormulationChoice formulationChoice, double q, double eps0, double m, double dt,
                                                   const double x0, const double xWidth, const int xElementsCoarse,
                                                   const double v0, const double vWidth, const int vElementsCoarse,
                                                   const int H1Order, const int delta_k, const bool usePeriodicBCs,
                                                   const bool enforceQuasiNeutrality, ERKTableauPtr erkTableau,
                                                   const int vlasovTraceEnrichment,
                                                   const int numUniformRefinements)
:
_formulationChoice(formulationChoice),
_usePeriodicBCs(usePeriodicBCs),
_enforceQuasiNeutrality(enforceQuasiNeutrality)
{
//  {
//    double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//    std::cout << "At start of Boltzmann1D1VFormulation construction, memory HWM:      " << memoryUsageInGigabytes << " GB.\n";
//  }
  
  const int spaceDim = 1;
  _delta_k = delta_k;
  _tn_enrichment = vlasovTraceEnrichment;
  
  SolutionStagePtr solnStage = Teuchos::null;
  _haveTableau = (erkTableau != Teuchos::null);
  if (_haveTableau)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(this->usesImplicitSolve(), std::invalid_argument, "ERK Tableau is only supported for explicit formulations...");
    solnStage = SolutionStage::solutionStage(erkTableau, dt);
    _dt = solnStage->dt();
  }
  else
  {
    _dt   = ParameterFunction::parameterFunction(dt);
  }
  
  bool useConformingTraces = true; // governs E_hat space (though in 1D this is just a point value regardless)
  
  _q    = ParameterFunction::parameterFunction(q);
  _eps0 = ParameterFunction::parameterFunction(eps0);
  _m    = ParameterFunction::parameterFunction(m);
  
  _n0   = ParameterFunction::parameterFunction(0.0); // will be set/updated during solve if quasi-neutrality enforced
  
  auto intervalMeshTopology      = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElementsCoarse, usePeriodicBCs);
  
  MeshTopologyPtr extrudedMeshTopology = MeshFactory::extrudedMeshTopology(intervalMeshTopology, {v0}, {vWidth}, {vElementsCoarse});
  
  for (int i=0; i<numUniformRefinements; i++)
  {
    const auto & activeQuadCells = extrudedMeshTopology->getActiveCellIndicesGlobal();
    auto line_x_line = CellTopology::cellTopology(CellTopology::line(), 1);
    auto refPattern = RefinementPattern::regularRefinementPattern(line_x_line->getKey());
    for (const auto & activeQuadCellIndex : activeQuadCells)
    {
      extrudedMeshTopology->refineCell(activeQuadCellIndex, refPattern, extrudedMeshTopology->nextCellID());
    }
    const auto & activeIntervalCells = intervalMeshTopology->getActiveCellIndicesGlobal();
    auto lineRefPattern = RefinementPattern::regularRefinementPatternLine();
    for (const auto & activeIntervalCellIndex : activeIntervalCells)
    {
      intervalMeshTopology->refineCell(activeIntervalCellIndex, lineRefPattern, intervalMeshTopology->nextCellID());
    }
  }
  
  const bool implicit  = usesImplicitSolve();
  const bool l2Trace   = hasL2Trace();
  const bool ultraweak = _formulationChoice == ULTRAWEAK_EXPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_EXPLICIT_H1_TRACE
                      || _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_IMPLICIT_H1_TRACE;
  
  if (ultraweak)
  {
    Space tauSpace    = (spaceDim > 1)      ? HDIV : HGRAD;
    Space E_hat_space = useConformingTraces ? HGRAD : L2;
    Space ESpace      = (spaceDim > 1)      ? VECTOR_L2 : L2;

    // fields
    VarPtr f; // Boltzmann
    VarPtr E; // Gauss

    // traces
    VarPtr E_hat, t_n_hat, f_hat; // t_n_hat: L2 trace, f_hat: H^1 trace

    // tests
    VarPtr w;   // Boltzmann test function
    
    _poissonForm = Teuchos::rcp(new PoissonFormulation(spaceDim,true,PoissonFormulation::ULTRAWEAK,
                                                       "V_E", "E", "\\hat{V_E}", "\\hat{E}", true));
    _gaussBF = _poissonForm->bf();
    E     = _poissonForm->sigma(); // the negative gradient of the potential
    E_hat = _poissonForm->sigma_n_hat();
    
    Epetra_CommPtr comm = Teuchos::null;
    const bool allowPruning = false;
    MeshPtr gaussMesh = MeshFactory::minRuleMesh(intervalMeshTopology, _gaussBF, H1Order, delta_k, comm, allowPruning);
    
    _gaussSoln = Solution::solution(_gaussBF, gaussMesh);
    auto gaussSolnWeakPtr = Teuchos::rcp(_gaussSoln.get(), false);
    
    FunctionPtr dt = FunctionPtr(_dt);
    FunctionPtr E_soln = Function::solution(E, gaussSolnWeakPtr);
    
    FunctionPtr v = Function::yn(1); // we are using the y dimension for velocity space
    FunctionPtr m = FunctionPtr(_m);
    
    VarFactoryPtr boltzmannVF = VarFactory::varFactory();
    
    f = boltzmannVF->fieldVar(S_F);
    w = boltzmannVF->testVar(S_W, HGRAD); // v lies in HGRAD of the whole 1D1V space
    
    if (l2Trace)
    {
      t_n_hat = boltzmannVF->fluxVar(S_T_N_HAT);
    }
    else
    {
      f_hat = boltzmannVF->traceVar(S_F_HAT);
    }
    
    _boltzmannBF = BF::bf(boltzmannVF);
    
    map<int,int> trialOrderEnhancements;
    map<int,int> testOrderEnhancements;
    if (l2Trace)
    {
      trialOrderEnhancements[t_n_hat->ID()] = 1 + _tn_enrichment;
    }
    
    MeshPartitionPolicyPtr partitionPolicy = Teuchos::null; // use default partition policy
    MeshPtr boltzmannMesh = Teuchos::rcp( new Mesh(extrudedMeshTopology, _boltzmannBF, H1Order, delta_k, trialOrderEnhancements, testOrderEnhancements, partitionPolicy, comm, allowPruning) );
    
    _phaseSpacePhysicalMeshMap = Teuchos::rcp( new PhasePhysicalMeshMap( gaussMesh, boltzmannMesh ) );
    auto meshMapWeakPtr        = Teuchos::rcp( _phaseSpacePhysicalMeshMap.get(), false);
    
    _E_extruded = Teuchos::rcp( new VelocitySpaceExtrusion(E_soln, meshMapWeakPtr) );

    _boltzmannBF->addTerm(f / _dt, w);
    
    if (l2Trace)
    {
      _boltzmannBF->addTerm(t_n_hat, w); // for explicit, t_n should be understood as the trace of the solution at prior time…
    }
    else
    {
      FunctionPtr n = Function::normal();
      _boltzmannBF->addTerm((v * n->x() + (q/m) * _E_extruded * n->y()) * f_hat, w); // for explicit, f_hat should be understood as the trace of the solution at prior time…
    }

    if (implicit)
    {
      _boltzmannBF->addTerm(-v * f, w->dx());
      _boltzmannBF->addTerm(-(q/m)*_E_extruded*f, w->dy());
    }
    
    _boltzmannSoln     = Solution::solution(_boltzmannBF,boltzmannMesh);
    if (_haveTableau)
    {
      // then previous solution is in the same Solution object as the "current" solution
      _boltzmannSoln->setSolutionStage(solnStage);
      _boltzmannSolnPrev = _boltzmannSoln;
      // register for notifications
      auto thisPtr = Teuchos::rcp(this,false);
      solnStage->addObserver(thisPtr);
    }
    else
    {
      _boltzmannSolnPrev = Solution::solution(_boltzmannBF,boltzmannMesh);
    }
    
    const int maxMomentNumber = 0; // for now, we only do the zeroth moment
    _solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, _boltzmannSoln, f, gaussMesh) );
    
    _f_velocityIntegral = _solutionMoments->solutionMomentFunction(0);
    
    // set RHS for the two problems
    // casts to allow FunctionPtr overloads to be used:
    auto qFunction = FunctionPtr(_q);
    auto mFunction = FunctionPtr(_m);
    auto epsFxn = FunctionPtr(_eps0);
    auto n0Fxn  = _enforceQuasiNeutrality ? FunctionPtr(_n0) : Function::zero(); // _n0 will be initialized when initial values are specified, if quasi-neutrality is enforced
    auto f_velocityIntegralFxn = FunctionPtr(_f_velocityIntegral);
    auto gaussRHS = _poissonForm->rhs(qFunction / epsFxn * (f_velocityIntegralFxn - n0Fxn));
    _gaussSoln->setRHS(gaussRHS);
    
    auto boltzmannRHS = RHS::rhs();
    auto boltzmannPrevSolnWeakPtr = Teuchos::rcp(_boltzmannSolnPrev.get(), false);
    
    const bool weightFluxesBySideParity = true;
    const SolutionComponent solnComponent = STANDARD_SOLUTION;
    const int stageOrdinal = _haveTableau ? -1 : 0;
    auto f_previous = Function::solution(f, boltzmannPrevSolnWeakPtr, weightFluxesBySideParity,
                                         solnComponent, stageOrdinal, std::string("k"));
    
    boltzmannRHS->addTerm(f_previous / dt * w);
    
    if (!implicit)
    {
      boltzmannRHS->addTerm(v * f_previous * w->dx());
      boltzmannRHS->addTerm((q/m)*_E_extruded*f_previous * w->dy());
    }
    
    _boltzmannSoln->setRHS(boltzmannRHS);
    
    // set the test-space inner products.
    _gaussSoln->setIP(_gaussBF->naiveNorm(spaceDim));
    
    gaussMesh->registerSolution(_gaussSoln);
    boltzmannMesh->registerSolution(_boltzmannSoln);
    boltzmannMesh->registerSolution(_boltzmannSolnPrev);
    
    IPPtr boltzmannTestNorm = IP::ip();
    const bool useSimpleL2Norm = false;
    if (useSimpleL2Norm)
    {
      boltzmannTestNorm->addTerm(w);
      std::cout << "NOTE: using simple L^2 norm for explicit Boltzmann...\n";
    }
    else
    {
      const bool useExperimentalNormForExplicit = false;
      if (!useExperimentalNormForExplicit || implicit)
      {
        // experiments so far suggest that the norm below is the most robust for explicit time-stepping
        boltzmannTestNorm->addTerm(Function::sqrtFunction((dt + 1)/dt) * w);
        boltzmannTestNorm->addTerm(v * w->dx() + (q/m)*_E_extruded * w->dy());
      }
      else
      {
        std::cout << "NOTE: using experimental test norm for explicit Boltzmann...\n";
        boltzmannTestNorm->addTerm(Function::sqrtFunction((dt + 1)/dt) * w);
        boltzmannTestNorm->addTerm(v * w->dx() + 1e-3 * (q/m)*_E_extruded * w->dy());
      }
    }
    _boltzmannSoln->setIP(boltzmannTestNorm);
    
    // create entity sets for BCs:
    {
      // 1D mesh entity sets:
      // create an empty entity set.  In higher dimensions, this could be an edge or face; on refinement,
      // the entity set contains the descendants of the original.
      EntitySetPtr x0EntitySet = intervalMeshTopology->createEntitySet();
      EntitySetPtr x1EntitySet = intervalMeshTopology->createEntitySet();
      
      // vertex lookup.  There are quite a few ways to look up entities in the MeshTopology.
      IndexType x0Vertex;
      bool x0Found = intervalMeshTopology->getVertexIndex({x0}, x0Vertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!x0Found, std::invalid_argument, "x0 not found");
      
      IndexType x1Vertex;
      bool x1Found = intervalMeshTopology->getVertexIndex({x0+xWidth}, x1Vertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!x1Found, std::invalid_argument, "x1 not found");
      
      x0EntitySet->addEntity(0, x0Vertex); // add entity of dimension 0 with ID x0Vertex to the x0 entity set.
      x1EntitySet->addEntity(0, x1Vertex); // add entity of dimension 0 with ID x1Vertex to the x1 entity set.
      
      intervalMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, GAUSS_TAG_X0, x0EntitySet);
      intervalMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, GAUSS_TAG_X1, x1EntitySet);
    }
    {
      // 2D mesh entity sets:
      auto bottomSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> bottomSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(SpatialFilter::matchingY(v0));
      for (const auto & bottomSideIndex : bottomSideIndices)
      {
        bottomSideSet->addEntity(1, bottomSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, BOLTZMANN_TAG_VMIN, bottomSideSet);
      auto topSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> topSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(SpatialFilter::matchingY(v0+vWidth));
      for (const auto & topSideIndex : topSideIndices)
      {
        topSideSet->addEntity(1, topSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, BOLTZMANN_TAG_VMAX, topSideSet);
      
      SpatialFilterPtr inflowLeftFilter  = SpatialFilter::matchingX(x0)          & SpatialFilter::greaterThanOrEqualToY(0.0);
      SpatialFilterPtr inflowRightFilter = SpatialFilter::matchingX(x0 + xWidth) & SpatialFilter::lessThanOrEqualToY   (0.0);
      
      auto inflowLeftSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> inflowLeftSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(inflowLeftFilter);
      for (const auto & leftSideIndex : inflowLeftSideIndices)
      {
        inflowLeftSideSet->addEntity(1, leftSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, BOLTZMANN_TAG_INFLOW_LEFT, inflowLeftSideSet);
      auto inflowRightSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> inflowRightSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(inflowRightFilter);
      for (const auto & inflowRightSideIndex : inflowRightSideIndices)
      {
        inflowRightSideSet->addEntity(1, inflowRightSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, BOLTZMANN_TAG_INFLOW_RIGHT, inflowRightSideSet);
    }

    // set BCs
    // set homogeneous conditions on the v boundaries
    // if we are not periodic in x, then the user must add conditions for left and right boundaries
    auto boltzmannBC = BC::bc();
    auto traceVar = l2Trace ? t_n_hat : f_hat;
    boltzmannBC->addDirichlet(traceVar, BOLTZMANN_TAG_VMIN, Function::zero());
    boltzmannBC->addDirichlet(traceVar, BOLTZMANN_TAG_VMAX, Function::zero());
    _boltzmannSoln->setBC(boltzmannBC);
    
    auto gaussBC = BC::bc();
    
    if (usePeriodicBCs)
    {
      gaussBC->addDirichlet(this->VE_hat(), GAUSS_TAG_X0, Function::constant(0.0));
    }
    else
    {
      gaussBC->addDirichlet(this->VE_hat(), GAUSS_TAG_X0, Function::constant(0.0)); // pin the potential to 0 at x0
      gaussBC->addDirichlet(this->VE_hat(), GAUSS_TAG_X1, Function::constant(0.0)); // pin the potential to 0 at x0+xWidth
    }
    _gaussSoln->setBC(gaussBC);
    
    // static condensation does not work properly with GMG due to details of VelocitySpaceExtrusion; it also does not appear to speed things up for the direct solve
//    if (!_haveTableau)
//    {
//      _boltzmannSoln->setUseCondensedSolve(true); // should work without issue given no tableau; we need to test with tableaus before enabling with those
//    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported Boltzmann1D1VFormulationChoice");
  }
//  {
//    double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//    std::cout << "At conclusion of Boltzmann1D1VFormulation construction, memory HWM: " << memoryUsageInGigabytes << " GB.\n";
//  }
  PhysicalMeshUpdater::registerForUpdates(_boltzmannSoln->mesh(), _gaussSoln->mesh());
}

MeshPtr Boltzmann1D1VFormulation::boltzmannMesh()
{
  return _boltzmannSoln->mesh();
}

SolutionPtr Boltzmann1D1VFormulation::boltzmannPreviousSolution()
{
  return _boltzmannSolnPrev;
}

SolutionPtr Boltzmann1D1VFormulation::boltzmannSolution()
{
  return _boltzmannSoln;
}

FunctionPtr Boltzmann1D1VFormulation::E_extruded()
{
  return _E_extruded;
}

VarPtr Boltzmann1D1VFormulation::f()
{
  VarFactoryPtr vf = _boltzmannBF->varFactory();
  return vf->trialVar(S_F);
}

VarPtr Boltzmann1D1VFormulation::f_hat()
{
  VarFactoryPtr vf = _boltzmannSoln->bf()->varFactory();
  return vf->trialVar(S_F_HAT);
}

MeshPtr Boltzmann1D1VFormulation::gaussMesh()
{
  return _gaussSoln->mesh();
}

SolutionPtr Boltzmann1D1VFormulation::gaussSolution()
{
  return _gaussSoln;
}

bool Boltzmann1D1VFormulation::hasL2Trace() const {
  return _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_EXPLICIT_L2_TRACE;
}

VarPtr Boltzmann1D1VFormulation::E()
{
  return _poissonForm->sigma();
}

VarPtr Boltzmann1D1VFormulation::E_hat()
{
  return _poissonForm->sigma_n_hat();
}

const std::vector< std::vector<int> > & Boltzmann1D1VFormulation::iterationCounts() const
{
  return _iterationCounts;
}

const std::map<GlobalIndexType, std::set<GlobalIndexType> > & Boltzmann1D1VFormulation::phaseSpaceCellIDToSpatialCellIDs()
{
  return _phaseSpaceCellIDToSpatialCellID;
}

Teuchos::RCP<PhasePhysicalMeshMap> Boltzmann1D1VFormulation::phaseSpacePhysicalMeshMap()
{
  return _phaseSpacePhysicalMeshMap;
}

void Boltzmann1D1VFormulation::setBoltzmannInflow(FunctionPtr f_left, FunctionPtr f_right)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_usePeriodicBCs, std::invalid_argument, "setting inflow BCs is not allowed for periodic meshes");

  auto boltzmannBC = BC::bc();
  const bool l2Trace = hasL2Trace();
  VarPtr traceVar = l2Trace ? this->t_n_hat() : this->f_hat();
  boltzmannBC->addDirichlet(traceVar, BOLTZMANN_TAG_VMIN, Function::zero());
  boltzmannBC->addDirichlet(traceVar, BOLTZMANN_TAG_VMAX, Function::zero());
  
  if (l2Trace)
  {
    FunctionPtr v = Function::yn(1);
    FunctionPtr leftTrace  = - f_left * v; // minus sign is thanks to outward normal
    FunctionPtr rightTrace =  f_right * v;
    boltzmannBC->addDirichlet(traceVar, BOLTZMANN_TAG_INFLOW_LEFT,   leftTrace);
    boltzmannBC->addDirichlet(traceVar, BOLTZMANN_TAG_INFLOW_RIGHT, rightTrace);
  }
  else
  {
    boltzmannBC->addDirichlet(traceVar, BOLTZMANN_TAG_INFLOW_LEFT,   f_left);
    boltzmannBC->addDirichlet(traceVar, BOLTZMANN_TAG_INFLOW_RIGHT, f_right);
  }
  _boltzmannSoln->setBC(boltzmannBC);
}

void Boltzmann1D1VFormulation::setGaussBCs(FunctionPtr VE_left, FunctionPtr VE_right)
{
  auto gaussBC = BC::bc();
  
  TEUCHOS_TEST_FOR_EXCEPTION(_usePeriodicBCs, std::invalid_argument, "setting left and right Gauss BCs is not allowed for periodic meshes");
  
  gaussBC->addDirichlet(this->VE_hat(), GAUSS_TAG_X0, VE_left);  // set the left potential
  gaussBC->addDirichlet(this->VE_hat(), GAUSS_TAG_X1, VE_right); // set the right potential
  
  _gaussSoln->setBC(gaussBC);
}

void Boltzmann1D1VFormulation::setInitialConditions(FunctionPtr f0)
{
  _boltzmannMeshMeasure = Function::constant(1.0)->integrate(_boltzmannSoln->mesh());
  
  std::map<int,FunctionPtr> projectionMap;
  projectionMap[this->f()->ID()] = f0;
  if ( !hasL2Trace() )
  {
    projectionMap[this->f_hat()->ID()] = f0;
  }
  
  const int solutionOrdinal = 0;
  _boltzmannSoln->projectOntoMesh(projectionMap, solutionOrdinal);
  if (!_haveTableau)
  {
    _boltzmannSolnPrev->setSolution(_boltzmannSoln);
  }
  
  _gaussMeshMeasure = Function::constant(1.0)->integrate(_gaussSoln->mesh());

  updateElectricField();
  
  // set initial tn value:
  auto gaussSolnWeakPtr = Teuchos::rcp(_gaussSoln.get(), false);
  
  if (hasL2Trace())
  {
    FunctionPtr sgn = Function::sideParity();
    FunctionPtr q(_q);
    FunctionPtr m(_m);
    FunctionPtr v = Function::yn(1);
    FunctionPtr n = Function::normal();
    FunctionPtr tn_0 = f0 * (v * n->x() + (q/m) * _E_extruded * n->y()) * sgn;
    projectionMap[this->t_n_hat()->ID()] = tn_0;
    _boltzmannSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    if (!_haveTableau)
    {
      _boltzmannSolnPrev->setSolution(_boltzmannSoln);
    }
  }
  _t = 0;
}

void Boltzmann1D1VFormulation::setIterativeSolveOutputLevel(int azOutputLevel)
{
  _aztecOutputLevel = azOutputLevel;
}

void Boltzmann1D1VFormulation::setRefinementThresholds(double refinementThreshold, int maxElements, int maxRefinementLevel, ErrorIndicatorPtr unrefinementErrorIndicator, bool gateRefinements)
{
  _refinementThreshold = refinementThreshold;
  _maxElements = maxElements;
  _maxRefinementLevel = maxRefinementLevel;
  _lastRefinementMaxElementError = -1.0;
  _gateRefinements = gateRefinements;
  
  if (_refStrategy == Teuchos::null)
  {
    _refStrategy = RefinementStrategy::energyErrorRefinementStrategy(_boltzmannSoln, _refinementThreshold);
  }
  _refStrategy->setRelativeErrorThreshold(_refinementThreshold);
  _refStrategy->setMaxElements(_maxElements);
  _refStrategy->setMaximumRefinementLevel(_maxRefinementLevel);
  _refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
  const int H1Order = _boltzmannSoln->mesh()->globalDofAssignment()->getInitialH1Order()[0];
  _refStrategy->setMaxP(H1Order); // disallow p refinements (these would require revision of VelocitySpaceExtrusion)
}

void Boltzmann1D1VFormulation::setStage(const int stageNumber)
{
  updateElectricField();
}

void Boltzmann1D1VFormulation::setTimeStep(double dt)
{
  _dt->setValue(dt);
}

void Boltzmann1D1VFormulation::setSubstepsForImplicit(int numSubsteps)
{
  _numSubstepsForImplicit = numSubsteps;
}

double Boltzmann1D1VFormulation::t() const
{
  return _t;
}

void Boltzmann1D1VFormulation::takeTimeStep()
{
  // for implicit solves, we cycle through the f solve and the E solve a few times;
  // for explicit, we do each solve just once per time step.
  const bool implicit = _formulationChoice == ULTRAWEAK_IMPLICIT_H1_TRACE || _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE;
  int numSubsteps = implicit ? _numSubstepsForImplicit : 1;
//  int numSubsteps = implicit ? 1 : 1; // 1 for implicit for faster debugging
  
  const bool useAdaptivity = _refStrategy != Teuchos::null;
  
  if (_haveTableau && useAdaptivity)
  {
    // the logic below only does adaptivity for the non-tableau (implicit, right now) case; probably user mis-specified something
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "adaptivity is only supported for implicit time-stepping");
  }
  
  bool useGMG = true;
  if (useGMG && (_gmgSolver == Teuchos::null))
  {
//    {
//      double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//      std::cout << "Before GMG construction, memory HWM: " << memoryUsageInGigabytes << " GB.\n";
//    }
    
    int kCoarse = 0;
    const double cgTol = 1e-6;
    const int maxIters = 10000;

    bool useCondensedSolve = _boltzmannSoln->usesCondensedSolve();

    //    auto multigridStrategy = GMGOperator::V_CYCLE;
    auto multigridStrategy = GMGOperator::SMOOTHER_ONLY;
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::IFPACK_ADDITIVE_SCHWARZ;
    GMGOperator::SmootherChoice smootherChoice = GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ; // CAMELLIA_ADDITIVE_SCHWARZ defines cells as blocks; Ifpack's defines all cells on the MPI rank as a block, making MPI scaling less predictable.  (Camellia's implementation is likely less well-optimized, though.)
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL; // as configured, this is very slow; I think it's treating all data on an MPI rank as a block…
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL;
    vector<MeshPtr> meshes;
    
    if (multigridStrategy != GMGOperator::SMOOTHER_ONLY)
    {
      meshes = GMGSolver::meshesForMultigrid(_boltzmannSoln->mesh(), kCoarse, 1);
    }
    else
    {
      meshes = vector<MeshPtr> {_boltzmannSoln->mesh()};
    }
    
    _gmgSolver = Teuchos::rcp( new GMGSolver(_boltzmannSoln, meshes, maxIters, cgTol, multigridStrategy,
                                             Solver::getDirectSolver(true), useCondensedSolve) );
    
    _gmgSolver->setSmootherType(smootherChoice);
    
    _gmgSolver->setAztecOutput(_aztecOutputLevel);
    
//    {
//      double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//      std::cout << "After GMG construction, memory HWM:  " << memoryUsageInGigabytes << " GB.\n";
//    }
  }
  
  std::vector<int> substepIterationCounts(numSubsteps);
  
  for (int substepOrdinal=0; substepOrdinal < numSubsteps; substepOrdinal++)
  {
    // solve for f:
    if (_gmgSolver == Teuchos::null)
    {
      _boltzmannSoln->solve();
      substepIterationCounts[substepOrdinal] = -1; // -1 to indicate not an iterative solver
    }
    else
    {
      _boltzmannSoln->solve(_gmgSolver);
      substepIterationCounts[substepOrdinal] = _gmgSolver->iterationCount();
    }
    
    const bool correctConservation = false;
    
    if (correctConservation)
    {
      // EXPERIMENT: check "conservation" and correct
      auto mesh = _boltzmannSoln->mesh();
      std::map<int, FunctionPtr> testMap;
      testMap[this->w()->ID()] = Function::constant(1.0);
      auto trialFunctional = _boltzmannBF->trialFunctional(testMap);
      std::map<int, FunctionPtr> trialMap;
      auto f_soln = Function::solution(this->f(), _boltzmannSoln, std::string("k+1"));
      trialMap[this->f()->ID()] = f_soln;
      auto lhsFunction = trialFunctional->evaluate(trialMap);
      
      auto rhsLT = _boltzmannSoln->rhs()->linearTerm();
      auto rhsFunctionBoundaryPart = rhsLT->getBoundaryOnlyPart()->evaluate(testMap);
      auto rhsFunctionVolumePart = rhsLT->getNonBoundaryOnlyPart()->evaluate(testMap);
      
      double rhsVolumeIntegral = rhsFunctionVolumePart->integrate(mesh);
      double rhsTraceIntegral  = rhsFunctionBoundaryPart->integrate(mesh);
      double rhsIntegral = rhsVolumeIntegral + rhsTraceIntegral;
      
      double lhsIntegral = lhsFunction->integrate(mesh);
      double residualIntegral = lhsIntegral - rhsIntegral;
      
      double error = _dt->getConstantValue() * residualIntegral;
      std::cout << "Conservation error before correction: " << error << std::endl;
      double correctionFactor = rhsIntegral / lhsIntegral;
      std::cout << "correctionFactor = 1 + " << (correctionFactor-1.0) << std::endl;
      FunctionPtr f_soln_corrected = f_soln * correctionFactor;
      
      std::map<int,FunctionPtr> correctedSolnMap;
      correctedSolnMap[this->f()->ID()] = f_soln_corrected;
      _boltzmannSoln->projectOntoMesh(correctedSolnMap);
      
      lhsIntegral      = lhsFunction->integrate(mesh);
      
      rhsVolumeIntegral = rhsFunctionVolumePart->integrate(mesh);
      rhsTraceIntegral  = rhsFunctionBoundaryPart->integrate(mesh);
      rhsIntegral = rhsVolumeIntegral + rhsTraceIntegral;
      
      error = _dt->getConstantValue() * (lhsIntegral  - rhsIntegral);
      std::cout << "Conservation error after correction: " << error << std::endl;
    }
    if ( !_haveTableau )
    {
      if (substepOrdinal == numSubsteps-1)
      {
        const bool useAdaptivity = _refStrategy != Teuchos::null;
        
        if (useAdaptivity)
        {
          bool shouldRefine;
          if (_gateRefinements)
          {
            // we compute error indicator first; if no refinement-eligible element has error greater than what would have been refined last time, then we don't do a refinement step now.
            
            auto localCellsToExclude = _refStrategy->localCellsIneligibleForRefinement();
            
            ErrorIndicatorPtr errorIndicator = _refStrategy->errorIndicator();
            errorIndicator->measureError();
            const double maxError = errorIndicator->maxError(localCellsToExclude);
            
            shouldRefine = maxError > _refinementThreshold * _lastRefinementMaxElementError;
            
            if (shouldRefine)
            {
              _lastRefinementMaxElementError = maxError;
            }
          }
          else
          {
            shouldRefine = true;
          }
          
          if (shouldRefine)
          {
            // for now, print refinement info to console on rank 0.
            const bool printToConsole = _boltzmannSoln->mesh()->Comm()->MyPID() == 0;
            const bool doUnrefinements = (_maxElements < std::numeric_limits<int>::max());
            _refStrategy->refineAndUnrefine(doUnrefinements, printToConsole);
          }
        }
        
        // last substep
        _boltzmannSolnPrev->setSolution(_boltzmannSoln);
      }
      // if we have a tableau, then electric field will be updated when the solution stage is set.
      updateElectricField();
    }
  }
  
  _iterationCounts.push_back(substepIterationCounts);
  
  _t += _dt->getConstantValue();
}

const std::map<GlobalIndexType, std::vector<GlobalIndexType> > & Boltzmann1D1VFormulation::spatialCellIDToPhaseSpaceCellIDsMap()
{
  return _spatialCellIDToPhaseSpaceCellIDs;
}

void Boltzmann1D1VFormulation::updateElectricField()
{
  // update f integral, n0
  _solutionMoments->update();
  
  double n0;
  if (_enforceQuasiNeutrality)
  {
    double f_integral   = _f_velocityIntegral->integrate(_gaussSoln->mesh());
    n0 = f_integral / _gaussMeshMeasure;
//    std::cout << "1D1V n0 = " << n0 << std::endl;
  }
  else
  {
    n0 = 0;
  }
  _n0->setValue(n0);
//  {
//    // DEBUGGING
//    std::cout << "f_integral = " << f_integral << std::endl;
//    std::cout << "n0 = " << n0 << std::endl;
//    
//    std::cout << "L^2 norm of _f_velocityIntegral: " << _f_velocityIntegral->l2norm(_gaussSoln->mesh()) << std::endl;
//    
//    const auto physicalMesh = _gaussSoln->mesh();
//    const auto & myCells = physicalMesh->cellIDsInPartition();
//    for (const auto & physicalCellID : myCells)
//    {
//      BasisCachePtr basisCache = BasisCache::basisCacheForCell(physicalMesh, physicalCellID);
//      bool success = true;
//      Teuchos::basic_FancyOStream<char> out(Teuchos::rcp(&std::cout,false));
//      testFunctionDataPathMatchesViewPath1V(_f_velocityIntegral, basisCache, 1e-14, out, success);
//      if (!success)
//      {
//        std::cout << "testFunctionDataPathMatchesViewPath1V(): failed on physical cell " << physicalCellID << std::endl;
//      }
//    }
//    
//    _gaussSoln->setWriteRHSToMatrixMarketFile(true, "gaussRHS.dat");
//    _gaussSoln->setWriteMatrixToFile(true, "gaussStiffness.dat");
//  }
  
  // solve for E, E_hat:
  _gaussSoln->solve();
    
  static_cast<VelocitySpaceExtrusion*>(_E_extruded.get())->updateValues();
}

bool Boltzmann1D1VFormulation::usesImplicitSolve() const
{
  return _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE || _formulationChoice == ULTRAWEAK_IMPLICIT_L2_TRACE;
}

// test variables:
VarPtr Boltzmann1D1VFormulation::w()
{
  VarFactoryPtr vf = _boltzmannBF->varFactory();
  return vf->testVar(S_W, HGRAD);
}

VarPtr Boltzmann1D1VFormulation::tau()
{
  VarFactoryPtr vf = _gaussBF->varFactory();
  return vf->testVar(S_TAU);
}

VarPtr Boltzmann1D1VFormulation::t_n_hat()
{
  VarFactoryPtr vf = _boltzmannSoln->bf()->varFactory();
  return vf->trialVar(S_T_N_HAT);
}

VarPtr Boltzmann1D1VFormulation::VE_hat()
{
  return _poissonForm->u_hat();
}

// below, we set up unit tests against the private classes above
#include "Teuchos_UnitTestHarness.hpp"

void testVelocitySpace1DIntegral(const bool &doRefinement, Teuchos::FancyOStream &out, bool &success)
{
  double tol = 1e-15;
  double q      = 1.0;
  double eps0   = 1.0;
  double m      = 1.0;
  double x0     = 0.0;
  double xWidth = 1.0;
  double v0     = 0.0;
  double vWidth = 1.0;
  
  const int xElements = 4;
  const int vElements = 4;
  const double dt = 0.1;
  const int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  const int delta_k = 2;
  const bool usePeriodicBCs = true;
  const bool enforceQuasiNeutrality = true;
  
  using FormulationChoice = Camellia::Boltzmann1D1VFormulation::Boltzmann1D1VFormulationChoice;
  std::vector<FormulationChoice> formulationChoices {
    FormulationChoice::ULTRAWEAK_EXPLICIT_H1_TRACE,
    FormulationChoice::ULTRAWEAK_EXPLICIT_L2_TRACE,
    FormulationChoice::ULTRAWEAK_IMPLICIT_H1_TRACE,
    FormulationChoice::ULTRAWEAK_IMPLICIT_L2_TRACE
  };
  
  set<GlobalIndexType> boltzmannCellsToRefine;
  if (doRefinement)
  {
    boltzmannCellsToRefine = {0,5,8};
  }
  
  for (auto formulationChoice : formulationChoices)
  {
    const int numUniformRefinements = 0;
    const int vlasovTraceEnrichment = 0;
    Boltzmann1D1VFormulation form(formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, Teuchos::null, vlasovTraceEnrichment, numUniformRefinements);
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr y  = Function::yn(1);
    FunctionPtr xy = x * y;
    // integral of xy dy = x * y^2 / 2
    // we integrate from v0 to v0 + vWidth
    const double expectedIntegralY = (0.5 * (v0 + vWidth) * (v0 + vWidth) - 0.5 * v0 * v0);
    FunctionPtr expectedIntegral = x * expectedIntegralY;
    const double expectedIntegralXY = (0.5 * (x0 + xWidth) * (x0 + xWidth) - 0.5 * x0 * x0) * expectedIntegralY;
    const double actualIntegralXY   = expectedIntegral->integrate(form.gaussMesh());
    
    TEST_FLOATING_EQUALITY(expectedIntegralXY, actualIntegralXY, tol);
    
    if (boltzmannCellsToRefine.size() > 0)
    {
      auto boltzmannMesh = form.boltzmannMesh();
      auto line_x_line = CellTopology::cellTopology(CellTopology::line(), 1);
      auto refPattern = RefinementPattern::regularRefinementPattern(line_x_line->getKey());
      boltzmannMesh->hRefine(boltzmannCellsToRefine, refPattern);
    }
    
    auto f = form.f();
    std::map<int, FunctionPtr> projectionMap;
    projectionMap[f->ID()] = xy;
    
    auto boltzmannSoln = form.boltzmannSolution();
    const int solutionOrdinal = 0;
    boltzmannSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    
    auto gaussMesh = form.gaussMesh();
    auto spatialToPhaseSpaceMap = form.spatialCellIDToPhaseSpaceCellIDsMap();
    
    const int maxMomentNumber = 0; // for now, we only do the zeroth moment
    SolutionMoments solutionMoments(maxMomentNumber, boltzmannSoln, f, gaussMesh);
    auto velocitySpaceIntegral = solutionMoments.solutionMomentFunction(0); // one argument because 1V
    FunctionPtr err = FunctionPtr(velocitySpaceIntegral) - expectedIntegral;
    
    solutionMoments.update();
    
    double magnitudeExpected = expectedIntegral->l2norm(gaussMesh);
    double err_l2 = err->l2norm(gaussMesh);
    
    TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
  }
}

TEUCHOS_UNIT_TEST( SolutionMomentFunctionBoltzmann1D1VFormulation, AnalyticIntegralMatches )
{
  const bool doRefinement = false;
  testVelocitySpace1DIntegral(doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( SolutionMomentFunctionBoltzmann1D1VFormulation, AnalyticIntegralMatchesWithRefinement )
{
  const bool doRefinement = true;
  testVelocitySpace1DIntegral(doRefinement, out, success);
}

class SideRestriction : public TFunction<double>
{
  std::vector<int> _sidesToMatch;
public:
  SideRestriction(std::vector<int> sidesToMatch)
  :
  Function("SideRestriction",0),
  _sidesToMatch(sidesToMatch)
  {}
  
  bool boundaryValueOnly() override
  {
    return true;
  }
  
  using TFunction<double>::values;
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    const int sideOrdinal = basisCache->getSideIndex();
    bool sideMatches = false;
    for (auto sideToMatch : _sidesToMatch)
    {
      if (sideToMatch == sideOrdinal)
      {
        sideMatches = true;
        break;
      }
    }
    const double value = sideMatches ? 1.0 : 0.0;
    values.initialize(value);
  }
};

void testVelocityExtrusion(bool volumeVariable, bool doRefinement, Teuchos::FancyOStream &out, bool &success)
{
  double tol = 1e-15;
  // these constants correspond to what we will be testing in the two-stream instability.  They should not affect this test, except maybe by introducing a little roundoff error here or there.
  double q      = 1.0;
  double eps0   = 1e-4;
  double m      = 1.0;
  double x0     = 0.0;
  double k_x    = 0.2; // wavelength
  double xWidth = 2 * M_PI / k_x;
  double v0     = 0.0;
  double vWidth = 20.0;
  
  const int xElements = 2;
  const int vElements = 2;
  const double dt = 0.1;
  const int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  const int delta_k = 2;
  const bool usePeriodicBCs = true;
  const bool enforceQuasiNeutrality = true;
  
  using FormulationChoice = Camellia::Boltzmann1D1VFormulation::Boltzmann1D1VFormulationChoice;
  std::vector<FormulationChoice> formulationChoices {
    FormulationChoice::ULTRAWEAK_EXPLICIT_H1_TRACE,
    FormulationChoice::ULTRAWEAK_EXPLICIT_L2_TRACE,
    FormulationChoice::ULTRAWEAK_IMPLICIT_H1_TRACE,
    FormulationChoice::ULTRAWEAK_IMPLICIT_L2_TRACE
  };
  
  for (auto formulationChoice : formulationChoices)
  {
    const int numUniformRefinements = 0;
    const int vlasovTraceEnrichment = 0;
    Boltzmann1D1VFormulation form(formulationChoice, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, Teuchos::null, vlasovTraceEnrichment, numUniformRefinements);
    
    FunctionPtr x  = Function::xn(1);
    
    FunctionPtr soln_1D;
    
    auto gaussSoln = form.gaussSolution();
    std::map<int, FunctionPtr> projectionMap;
    if (volumeVariable)
    {
      auto E = form.E();
      projectionMap[E->ID()] = x;
      soln_1D = Function::solution(E, gaussSoln);
    }
    else
    {
      auto E     = form.E();
      auto E_hat = form.E_hat();
      projectionMap[E->ID()] = x * (xWidth - x);
      projectionMap[E_hat->ID()] = x * (xWidth - x); // for skeleton values, due to periodic BCs, want something that agrees on left/right side: having zeros there makes this simpler
      soln_1D = Function::solution(E_hat, gaussSoln, false); // false: don't weight by parity (makes uniquely-valued)
    }
    
    TEST_ASSERT(soln_1D->boundaryValueOnly() == !volumeVariable);
    
    auto gaussMesh = form.gaussMesh();
    auto boltzmannMesh = form.boltzmannMesh();
    
    auto meshMap = Teuchos::rcp( new PhasePhysicalMeshMap( gaussMesh, boltzmannMesh ) );
    auto extrusion = Teuchos::rcp( new VelocitySpaceExtrusion(soln_1D, meshMap) );
    
    auto cellTopo = CellTopology::cellTopology(CellTopology::line(), 1);
    
    if (doRefinement)
    {
      const set<GlobalIndexType> cellIDs_2D {0,1};
      auto line_x_line = CellTopology::cellTopology(CellTopology::line(), 1);
      auto refPattern = RefinementPattern::regularRefinementPattern(line_x_line->getKey());
      boltzmannMesh->hRefine(cellIDs_2D, refPattern);
    }
    
    const int solutionOrdinal =0;
    gaussSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    
    extrusion->updateValues();
    
    FunctionPtr expected;
    if (volumeVariable)
    {
      expected = x;
    }
    else
    {
      // restrict x to the spatial sides of the mesh skeleton
      // these are the sides of the hypercube corresponding to left and right sides of the physical cell (sides that do not vary in x)
      // because of the way tensor cell topologies are numbered, the final sides are the ones of interest: here, the final two sides;
      const int sideCount = cellTopo->getSideCount();
      std::vector<int> spatialSides { sideCount - 2, sideCount - 1 };
      TEUCHOS_TEST_FOR_EXCEPTION( !cellTopo->sideIsSpatial(spatialSides[0]), std::logic_error, "Spatial sides are not enumerated as expected");
      TEUCHOS_TEST_FOR_EXCEPTION( !cellTopo->sideIsSpatial(spatialSides[1]), std::logic_error, "Spatial sides are not enumerated as expected");
      FunctionPtr sideRestriction = Teuchos::rcp(new SideRestriction(spatialSides) );
      expected = x * (xWidth - x) * sideRestriction;
    }
    
    FunctionPtr err = FunctionPtr(extrusion) - expected;
    
  //  double magnitudeActual = extrusion->l2norm(boltzmannMesh);
  //  std::cout << "magnitudeActual = " << magnitudeActual << std::endl;
    
    double magnitudeExpected = expected->l2norm(boltzmannMesh);
  //  std::cout << "magnitudeExpected = " << magnitudeExpected << std::endl;
    double err_l2 = err->l2norm(boltzmannMesh);
    
    TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
    
    const auto & phaseCellIDs = boltzmannMesh->cellIDsInPartition();
    for (auto phaseCellID : phaseCellIDs)
    {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(boltzmannMesh, phaseCellID);
      if (volumeVariable)
      {
        testFunctionDataPathMatchesViewPath1V(extrusion, basisCache, tol, out, success);
      }
      else
      {
        const int numSides = basisCache->cellTopology()->getSideCount();
        for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
        {
          auto sideCache = basisCache->getSideBasisCache(sideOrdinal);
          testFunctionDataPathMatchesViewPath1V(extrusion, sideCache, tol, out, success);
        }
      }
    }
  }
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionBoltzmann1D1VFormulation, SkeletonExtrusion )
{
  const bool volumeVariable = false;
  const bool doRefinement = false;
  testVelocityExtrusion(volumeVariable, doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionBoltzmann1D1VFormulation, SkeletonExtrusionWithRefinement )
{
  const bool volumeVariable = false;
  const bool doRefinement = true;
  testVelocityExtrusion(volumeVariable, doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionBoltzmann1D1VFormulation, VolumeExtrusion )
{
  const bool volumeVariable = true;
  const bool doRefinement = false;
  testVelocityExtrusion(volumeVariable, doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionBoltzmann1D1VFormulation, VolumeExtrusionWithRefinement )
{
  const bool volumeVariable = true;
  const bool doRefinement = true;
  testVelocityExtrusion(volumeVariable, doRefinement, out, success);
}
