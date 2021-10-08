//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  VlasovPoissonSpaceTimeFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/3/2020.
//
//

#include "VlasovPoissonSpaceTimeFormulation.h"
#include "CamelliaMemoryTools.h"
#include "CubatureFactory.h"
#include "ERKTableau.h"
#include "ExtrudedMeshTopology.h"
#include "GlobalDofAssignment.h"
#include "GMGSolver.h"
#include "MeshFactory.h"
#include "MonomialFunctions.h"
#include "ParameterFunction.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "Solution.h"
#include "SolutionStage.h"
#include "TypeDefs.h"

#include "CamelliaDataHelpers.h"

using namespace Camellia;

const string VlasovPoissonSpaceTimeFormulation::S_F = "f";
const string VlasovPoissonSpaceTimeFormulation::S_E = "E";
const string VlasovPoissonSpaceTimeFormulation::S_VE = "V_E";

const string VlasovPoissonSpaceTimeFormulation::S_F_HAT = "\\widehat{f}";
const string VlasovPoissonSpaceTimeFormulation::S_T_N_HAT = "\\widehat{t}_n";
const string VlasovPoissonSpaceTimeFormulation::S_E_HAT = "\\widehat{E}";
const string VlasovPoissonSpaceTimeFormulation::S_VE_HAT = "\\widehat{V}_E";

const string VlasovPoissonSpaceTimeFormulation::S_W = "w";

// Poisson test functions
const string VlasovPoissonSpaceTimeFormulation::S_V_POISSON = "v_{\rm Poisson}";
const string VlasovPoissonSpaceTimeFormulation::S_TAU_POISSON = "\\tau_{\rm Poisson}";

class PhysicalSolutionExtrusion : public TFunction<double>
{
  SolutionPtr _physicalSoln;    // the solution on the physical mesh
  VarPtr      _fieldVar;        // the physical field var whose solution we extrude.
  FunctionPtr _physicalSolnFxn; // the solution function for the field var, defined on the physical mesh.
  
  Teuchos::RCP<PhasePhysicalMeshMap> _phaseSpacePhysicalMeshMap; // TODO: eliminate this member variable; we don't use the PhasePhysicalMeshMap except to allow reference to phase and physical meshes individually.
  static int variationDimensions(Teuchos::RCP<PhasePhysicalMeshMap> phaseSpacePhysicalMeshMap)
  {
    auto physicalMeshDims = phaseSpacePhysicalMeshMap->physicalMesh()->getDimension();
    int variationDimensions = VARIES_IN_NO_DIMENSION;
    for (int d=0; d<physicalMeshDims; d++)
    {
      int varDim = VARIES_IN_DIMENSION_0 << d; // will give VARIES_IN_DIMENSION_1 for d == 1, etc.
      variationDimensions |= varDim;
    }
    return variationDimensions;
  }
public:
  PhysicalSolutionExtrusion( SolutionPtr physicalSoln, VarPtr fieldVar,
                            Teuchos::RCP<PhasePhysicalMeshMap> phaseSpacePhysicalMeshMap)
  :
  Function("PhysicalSolutionExtrusion",fieldVar->rank(),variationDimensions(phaseSpacePhysicalMeshMap)), // varies in space-time, but not in velocity dimensions
  _physicalSoln(physicalSoln),
  _fieldVar(fieldVar),
  _phaseSpacePhysicalMeshMap(phaseSpacePhysicalMeshMap)
  {
    auto physicalSolnWeakPtr = Teuchos::rcp(_physicalSoln.get(), false);
    _physicalSolnFxn = Function::solution(_fieldVar, physicalSolnWeakPtr);
    
    TEUCHOS_TEST_FOR_EXCEPTION(_physicalSolnFxn->rank() != 0, std::invalid_argument, "Only scalar-valued functions supported");
  }
  
  void updateValues()
  {
    auto & phaseSpaceCellIDs = _phaseSpacePhysicalMeshMap->phaseMesh()->cellIDsInPartition();
        
    // determine all active descendants of the physical cells needed locally; import *their* data
    set<GlobalIndexType> activePhysicalCellIDDescendants;
    
    vector< set<GlobalIndexType> > parentCellsForLevel;
    auto physicalMeshTopo = _phaseSpacePhysicalMeshMap->physicalMesh()->getTopology();
    ExtrudedMeshTopology* phaseSpaceMeshTopo = dynamic_cast<ExtrudedMeshTopology*>( _phaseSpacePhysicalMeshMap->phaseMesh()->getTopology().get());
    
    TEUCHOS_TEST_FOR_EXCEPTION(phaseSpaceMeshTopo == NULL, std::invalid_argument, "Phase-space MeshTopology must be an ExtrudedMeshTopology");
    
    set<GlobalIndexType> physicalCellIDs;
    for (const auto & phaseSpaceCellID : phaseSpaceCellIDs)
    {
      auto physicalCellID = phaseSpaceMeshTopo->physicalCellIDForPhaseSpaceCellID(phaseSpaceCellID);
      physicalCellIDs.insert(physicalCellID);
    }
    
    for (const auto &physicalCellID : physicalCellIDs)
    {
      auto physicalCell = physicalMeshTopo->getCell(physicalCellID);
      const bool leafNodesOnly = true; // only the active descendants
      auto activeDescendants = physicalCell->getDescendants(physicalMeshTopo, leafNodesOnly);
      activePhysicalCellIDDescendants.insert(activeDescendants.begin(),activeDescendants.end());
      
      if (physicalCell->isParent(physicalMeshTopo))
      {
        auto allDescendants = physicalCell->getDescendants(physicalMeshTopo, false); // false: not just the leaf nodes
        for (const auto & descendantID : allDescendants)
        {
          if (descendantID == physicalCellID) continue; // getDescendants() includes the cell itself, too, when leafNodesOnly = false.  (We only want proper descendants.)
          auto descendant = physicalMeshTopo->getCell(descendantID);
          auto parentCellID = descendant->getParent()->cellIndex();
          
          int descendantLevel = descendant->level();
          if (descendantLevel >= parentCellsForLevel.size())
          {
            parentCellsForLevel.resize(descendantLevel);
          }
          parentCellsForLevel[descendantLevel - 1].insert(parentCellID);
        }
      }
    }
    
    _physicalSoln->discardInactiveCellCoefficients(); // clear any previously computed parent data -- this is mostly to avoid hiding bugs due to using stale data
    
    const std::vector<GlobalIndexType> physicalCellIDsVector(activePhysicalCellIDDescendants.begin(), activePhysicalCellIDDescendants.end());
    _physicalSolnFxn->importCellData(physicalCellIDsVector);
    
    int maxLevel = parentCellsForLevel.size() - 1;
    for (int level=maxLevel; level >= 0; level--)
    {
      const auto & parentCellIDsForLevel = parentCellsForLevel[level];
      for (const auto & parentCellID : parentCellIDsForLevel)
      {
        _physicalSoln->projectChildrenOntoParent(parentCellID);
      }
    }
  }
  
  using TFunction<double>::values;
  void values(Intrepid2::Data<double, DeviceType> values, BasisCachePtr basisCache) override
  {
    // We only support fully tensor-product cubature points (non-trivial PhysicalTensorPoints)
    
    auto physicalMesh = _phaseSpacePhysicalMeshMap->physicalMesh();
    auto phaseMesh    = _phaseSpacePhysicalMeshMap->phaseMesh();
    
    const int numPhysicalDimensions = physicalMesh->getDimension();
    const int numPhaseDimensions    = phaseMesh->getDimension();
    const int numVelocityDimensions = numPhaseDimensions - numPhysicalDimensions;
    
    const auto &tensorPoints = basisCache->getPhysicalCubaturePoints();
    
    const auto &phaseRefPoints = basisCache->getRefCellPoints(); // TODO: figure out whether we actually need the physical points (tensorPoints above), or whether we can just use tensorRefPoints everywhere instead…
    
    const int numPhysicalPointComponents = phaseRefPoints.numTensorComponents() - numVelocityDimensions;
    
    using DynRankView = Kokkos::DynRankView<double, DeviceType>;
    std::vector< DynRankView > physicalRefPointComponents(numPhysicalPointComponents);
    
    for (int r=0; r<numPhysicalPointComponents; r++)
    {
      physicalRefPointComponents[r] = phaseRefPoints.getTensorComponent(r);
    }
    
    const Intrepid2::TensorPoints<double,DeviceType> physicalRefPoints(physicalRefPointComponents);
    
    // if cubature points are fully tensor-product, there will be at least numVelocityDimensions + 1 tensor components in physical cubature points; otherwise, there will be just 1
    const bool velocityPointsAreFullTensorProduct = tensorPoints.numTensorComponents() >= numVelocityDimensions + 1;
    
    TEUCHOS_TEST_FOR_EXCEPTION(!velocityPointsAreFullTensorProduct, std::invalid_argument, "Unsupported point structure; we require fully tensor-product physical points in the extrusion dimensions");
    
    const int numPoints = tensorPoints.extent_int(1);
    
    std::vector<int> physicalPointCounts(tensorPoints.numTensorComponents() - numVelocityDimensions);
    
    int numPhysicalPoints = 1;
    int numPhysicalPointsInValues = 1;
    for (int r=0; r<tensorPoints.numTensorComponents() - numVelocityDimensions; r++)
    {
      physicalPointCounts[r] = tensorPoints.getTensorComponent(r).extent_int(1);
      numPhysicalPoints *= physicalPointCounts[r];
      numPhysicalPointsInValues *= values.extent_int(1+r);
    }
    
    TEUCHOS_TEST_FOR_EXCEPTION(numPhysicalPoints != numPhysicalPointsInValues, std::invalid_argument, "numPhysicalPoints should match the product of the extents in dimension 1 of values");
    const int numVelocityPoints = numPoints / numPhysicalPoints;
    TEUCHOS_TEST_FOR_EXCEPTION(numVelocityPoints != values.extent_int(3), std::invalid_argument, "numVelocityPoints should match the extent in dimension 3 of values");
    TEUCHOS_TEST_FOR_EXCEPTION(values.getVariationModulus(3) != 1, std::invalid_argument, "values should not vary in velocity dimension");
    
    TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 1 + numPhaseDimensions, std::invalid_argument, "Expect values to have shape (C,P0,P1,…,Pn) for n = # phase dimensions - 1");
    int extrusionExtentCount = 1;
    for (int dataIndex=numPhysicalDimensions+1; dataIndex<values.rank(); dataIndex++) // expect variation in C, plus each physical dimension (including time, for space-time)
    {
      extrusionExtentCount *= values.getDataExtent(dataIndex);
    }
    TEUCHOS_TEST_FOR_EXCEPTION(extrusionExtentCount != 1, std::invalid_argument, "Data container has more than the expected extent in extrusion dimensions");

    // implementation adapted from the DeviceView values() implementation below
    
    using HostExecSpace = Kokkos::HostSpace::execution_space;
    Intrepid2::Data<double,HostExecSpace> valuesHost(values);
    
    // at present, we require that this be called on every MPI rank that owns cells in the phase space mesh
    // we could avoid this requirement by adding an explicit, separate communication step.
    
    // It is also worth noting that creating basisCache_1D below is an inefficiency: those basis values won't
    // be reused beyond the body of this method…
    
    const bool isSideCache = (basisCache->getSideIndex() != -1);
    TEUCHOS_TEST_FOR_EXCEPTION(isSideCache, std::invalid_argument, "PhaseSpaceExtrusion doesn't support evaluation on sides");

    int physicalSpaceSideOrdinal = -1;
    
    auto extrudedMeshTopology = dynamic_cast<ExtrudedMeshTopology*>(phaseMesh->getTopology().get());
    TEUCHOS_TEST_FOR_EXCEPTION(extrudedMeshTopology == NULL, std::invalid_argument, "phase-space mesh must use an ExtrudedMeshTopology");
    
    auto physicalMeshTopology = physicalMesh->getTopology();
    
    const vector<GlobalIndexType> & phaseSpaceCellIDs = basisCache->cellIDs();
    
    for (int phaseSpaceCellOrdinal = 0; phaseSpaceCellOrdinal < phaseSpaceCellIDs.size(); phaseSpaceCellOrdinal++)
    {
      const auto & phaseSpaceCellID = phaseSpaceCellIDs[phaseSpaceCellOrdinal];
      
      // rule is that phase space mesh is nowhere finer than physical mesh: we can "cover" the phase space cell's physical dimensions with active physical cells.  And there is a unique (possibly refined) physical cell that corresponds exactly to the physical component of the phase-space cell.
      GlobalIndexType physicalCellID = extrudedMeshTopology->physicalCellIDForPhaseSpaceCellID(phaseSpaceCellID);
      
      CellPtr physicalCell = physicalMeshTopology->getCell(physicalCellID);
      
      const int cubatureDegree = basisCache->cubatureDegree();

      FieldContainer<double> physicalCellNodes = physicalMesh->physicalCellNodesForCell(physicalCellID);
      BasisCachePtr physicalBasisCache = Teuchos::rcp(new BasisCache(physicalCellNodes, physicalCell->topology(), physicalRefPoints) );
      physicalBasisCache->setMesh(physicalMesh);
      physicalBasisCache->setCellIDs({physicalCellID});
      
      auto g_dataValues = _physicalSolnFxn->checkoutValuesData(physicalBasisCache);
      
      _physicalSolnFxn->values(g_dataValues, physicalBasisCache);
      
      // copy g_dataValues into values
      using ExecutionSpace = Camellia::DeviceExecutionSpace;
      if ((numPhysicalPointComponents == 2) && (numVelocityDimensions == 1))
      {
        // this is the 1D1T1V case (space-time 1D1V)
        auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>>({0,0},{physicalPointCounts[0],physicalPointCounts[1]});
        
        Kokkos::parallel_for("SolutionMomentFunction_values", policy,
        KOKKOS_LAMBDA (const int &x0, const int &x1)
        {
          values.getWritableEntry(phaseSpaceCellOrdinal,x0,x1,0) = g_dataValues(0,x0,x1);
        });
      }
      else
      {
        // TODO: add support for other physical/velocity component counts
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported physical/velocity component count");
      }
      
      _physicalSolnFxn->checkinValuesData(g_dataValues);
    }
  }
  
  bool boundaryValueOnly() override
  {
    // should always be false
    return _physicalSolnFxn->boundaryValueOnly();
  }
};

///*
// Worth noting that VelocitySpaceExtrusion codifies an inefficient use of memory:
// it is simply duplicating values in the velocity dimension.  It would be better to
// avoid the duplication by adding some facilities to Function; I like the idea of
// adding a method
//    bool variesInDimension(int dim);
// and then having some mechanism by which the arrays that Functions that are constant
// in some dimension can only specify values for the dimensions in which they vary.
// Doing this more or less requires reworking the way points are stored, though, to
// express tensor-product points.  We have some loose plans to do something like this
// in Intrepid2.
// */
//class VelocitySpaceExtrusion1D1T1V : public TFunction<double>
//{
//  FunctionPtr _g_1D1T; // arbitrary function in physical space-time -- can depend on mesh/physical solution
//  
//  Teuchos::RCP<PhasePhysicalMeshMap> _phaseSpacePhysicalMeshMap;
//public:
//  VelocitySpaceExtrusion1D1T1V( FunctionPtr g_1D, Teuchos::RCP<PhasePhysicalMeshMap> phaseSpacePhysicalMeshMap)
//  :
//  Function("VelocitySpaceExtrusion",g_1D->rank(),VARIES_IN_DIMENSION_0),
//  _g_1D1T(g_1D),
//  _phaseSpacePhysicalMeshMap(phaseSpacePhysicalMeshMap)
//  {
//    TEUCHOS_TEST_FOR_EXCEPTION(_g_1D1T->rank() != 0, std::invalid_argument, "Only scalar-valued functions supported right now");
//  }
//  
//  void updateValues()
//  {
//    auto & phaseSpaceCellIDs = _phaseSpacePhysicalMeshMap->phaseMesh()->cellIDsInPartition();
//    
//    const set<GlobalIndexType> physicalCellIDs = _phaseSpacePhysicalMeshMap->physicalCellIDsForPhaseSpaceCells(phaseSpaceCellIDs);
//    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalValuesMap;
//    
//    const std::vector<GlobalIndexType> physicalCellIDsVector(physicalCellIDs.begin(), physicalCellIDs.end());
//    _g_1D1T->importCellData(physicalCellIDsVector);
//  }
//  
//  using TFunction<double>::values;
//  void values(Intrepid2::Data<double, DeviceType> values, BasisCachePtr basisCache) override
//  {
//    // there are two use cases to consider:
//    // 1. fully tensor-product cubature points (non-trivial PhysicalTensorPoints)
//    // 2. cubature points without a specified tensor-product structure
//    //
//    // (1) is the principal use case.  (2) is a holdover, just there until we can upgrade BasisCache to manage tensor points on sides…
//    
//    const int numVelocityDimensions = 1;
//    
//    const auto &tensorPoints = basisCache->getPhysicalCubaturePoints();
//    
//    // if we are in use case (1), there will be 3 tensor components in physical cubature points (for 1D1T1V); otherwise, there will be just 1
//    const bool dataExtentMatchesPhysicalPoints = tensorPoints.numTensorComponents() > 1; // true for use case (1)
//    
//    // the code that actually fills in values below assumes that values has shape (C,P1,P2,P3) in case (1), and shape (C,P) in use case 2
//    const int numPoints = tensorPoints.extent_int(1);
//    int numSpatialPoints  = -1; // stored if the data extent matches that of physical points
//    int numTemporalPoints = -1;
//
//    if (dataExtentMatchesPhysicalPoints)
//    {
//      int numPhysicalPoints = 1;
//      int numPhysicalPointsInValues = 1;
//      for (int r=0; r<tensorPoints.numTensorComponents() - numVelocityDimensions; r++)
//      {
//        numPhysicalPoints *= tensorPoints.getTensorComponent(r).extent_int(1);
//        numPhysicalPointsInValues *= values.extent_int(1+r);
//      }
//      // these two lines are specific to the 1D1T case:
//      numSpatialPoints  = tensorPoints.getTensorComponent(0).extent_int(1);
//      numTemporalPoints = tensorPoints.getTensorComponent(1).extent_int(1);
//      
////      std::cout << "tensorPoints.getTensorComponent(0).extent_int(0): " << tensorPoints.getTensorComponent(0).extent_int(0) << std::endl;
////      std::cout << "tensorPoints.getTensorComponent(0).extent_int(1): " << tensorPoints.getTensorComponent(0).extent_int(1) << std::endl;
////      std::cout << "tensorPoints.getTensorComponent(1).extent_int(0): " << tensorPoints.getTensorComponent(1).extent_int(0) << std::endl;
////      std::cout << "tensorPoints.getTensorComponent(1).extent_int(1): " << tensorPoints.getTensorComponent(1).extent_int(1) << std::endl;
//      
//      TEUCHOS_TEST_FOR_EXCEPTION(numPhysicalPoints != numPhysicalPointsInValues, std::invalid_argument, "numPhysicalPoints should match the product of the extents in dimension 1 of values");
//      const int numVelocityPoints = numPoints / numPhysicalPoints;
//      TEUCHOS_TEST_FOR_EXCEPTION(numVelocityPoints != values.extent_int(3), std::invalid_argument, "numVelocityPoints should match the extent in dimension 3 of values");
//      TEUCHOS_TEST_FOR_EXCEPTION(values.getVariationModulus(3) != 1, std::invalid_argument, "values should not vary in velocity dimension");
//    }
//    else
//    {
//      TEUCHOS_TEST_FOR_EXCEPTION(numPoints != values.extent_int(1), std::invalid_argument, "numPoints should match the extent in dimension 1 of values");
//    }
//    
//    
//    if (dataExtentMatchesPhysicalPoints)
//    {
//      TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 4, std::invalid_argument, "Expect values to have shape (C,P0,P1,P2)");
//      int extrusionExtentCount = 1;
//      for (int dataIndex=2; dataIndex<values.rank(); dataIndex++)
//      {
//        extrusionExtentCount *= values.getDataExtent(dataIndex);
//      }
//      TEUCHOS_TEST_FOR_EXCEPTION(extrusionExtentCount != 1, std::invalid_argument, "Data container has more than the expected extent in extrusion dimensions");
//    }
//
//    // implementation adapted from the DeviceView values() implementation below
//    
//    using HostExecSpace = Kokkos::HostSpace::execution_space;
//    Intrepid2::Data<double,HostExecSpace> valuesHost(values);
//    
//    // at present, we require that this be called on every MPI rank that owns cells in the phase space mesh
//    // we could avoid this requirement by adding an explicit, separate communication step.
//    
//    // It is also worth noting that creating basisCache_1D below is an inefficiency: those basis values won't
//    // be reused beyond the body of this method…
//    
//    bool isSideCache = (basisCache->getSideIndex() != -1);
//    int physicalSpaceSideOrdinal;
//    if (!isSideCache)
//    {
//      physicalSpaceSideOrdinal = -1;
//    }
//    else // isSideCache == true
//    {
//      const int phaseSpaceSideOrdinal = basisCache->getSideIndex();
//      
//      // what follows assumes 1D1T1V with line x line x line elements; let's throw an exception if we're not on such an element
//      auto line_x_line_x_line = CellTopology::cellTopology(CellTopology::line(), 2);
//      TEUCHOS_TEST_FOR_EXCEPTION(basisCache->cellTopology()->getKey() != line_x_line_x_line->getKey(),
//                                 std::invalid_argument, "Only phase-space line x line x line topologies are supported");
//      bool sideIsParallelToPhysicalSpace = !line_x_line_x_line->sideIsSpatial(phaseSpaceSideOrdinal);
//      
//      if (sideIsParallelToPhysicalSpace)
//      {
//        // this side does not map to a physical side; runs parallel to physical space
//        // therefore, it instead maps to a physical cell interior (in adaptive context, possibly multiple -- this is not yet supported)
//        if (_g_1D1T->boundaryValueOnly())
//        {
//          // if _g_1D1T is a skeleton variable, then we can't evaluate it on a cell interior; we define the extrusion to be zero in this case
//          Kokkos::deep_copy(values.getUnderlyingView(), 0.0);
//          return;
//        }
//        else
//        {
//          // we can and should evaluate _g_1D1T on the spatial element interior
//          physicalSpaceSideOrdinal = -1;
//        }
//      }
//      else
//      {
//        physicalSpaceSideOrdinal = line_x_line_x_line->getSpatialComponentSideOrdinal(phaseSpaceSideOrdinal);
//      }
//    }
//    
//    using std::map;
//    using std::pair;
//    using std::set;
//    using std::vector;
//    
//    // determine physical cell IDs we need to compute on
//    const vector<GlobalIndexType> & phaseSpaceCellIDs = basisCache->cellIDs();
//    const vector<GlobalIndexType> physicalCellIDs = _phaseSpacePhysicalMeshMap->physicalCellIDsForPhaseSpaceCells(phaseSpaceCellIDs);
//    
//    using PhysicalValuesHost = Kokkos::DynRankView<double, Kokkos::HostSpace>;
//    map<GlobalIndexType, PhysicalValuesHost> physicalValuesMapHost;
//    map<GlobalIndexType, DeviceView<double> > physicalValuesMap;
////    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalPointsMap; // this is used for debugging; we could save something by dropping this later.
//    
//    const bool testVsTest = basisCache->testVsTest();
//    const int phaseSpaceQuadratureEnrichment = basisCache->cubatureEnrichmentDegree();
//
//    int basisCache1DQuadratureEnrichment;
//    if (testVsTest)
//    {
//      // test order in both 1D and 2D meshes is relative to the H^1 order, which will match in both
//      basisCache1DQuadratureEnrichment = phaseSpaceQuadratureEnrichment;
//    }
//    else
//    {
//      // trial order is 1 higher in 2D mesh because the t_n_hat/f_hat discretization has order matching H1Order
//      basisCache1DQuadratureEnrichment = phaseSpaceQuadratureEnrichment + 1;
//    }
//    
//    map< GlobalIndexType , pair<BasisCachePtr, vector<int> > > physicalBasisCachePairs; // index: physical cell ID
//    for ( auto & physicalCellID : physicalCellIDs )
//    {
//      physicalBasisCachePairs[physicalCellID] = _phaseSpacePhysicalMeshMap->physicalCellBasisCache(physicalCellID, testVsTest, basisCache1DQuadratureEnrichment);
//    }
//    
//    const bool inPhysicalVolume = (physicalSpaceSideOrdinal == -1);
//    for ( auto & physicalCellID : physicalCellIDs )
//    {
//      auto basisCachePair = physicalBasisCachePairs[physicalCellID];
//      auto physicalBasisCache = basisCachePair.first;
//      auto offsets            = basisCachePair.second;
//      
//      BasisCachePtr basisCacheToUse = inPhysicalVolume ? physicalBasisCache : physicalBasisCache->getSideBasisCache(physicalSpaceSideOrdinal);
//      
//      const int numCells  = 1;
//      const int numPoints = basisCacheToUse->getRefCellPointsFC().dimension(0);
//      
//      auto physicalValuesView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numPoints}, "VelocitySpaceExtrusion1D1V physicalValuesView");
//      
//      _g_1D1T->values(physicalValuesView, ASSIGN, basisCacheToUse);
//      physicalValuesMap[physicalCellID] = physicalValuesView;
//      
//      PhysicalValuesHost physicalValuesViewHost = Kokkos::create_mirror_view_and_copy(HostExecSpace::memory_space(), physicalValuesView);
//      physicalValuesMapHost[physicalCellID] = physicalValuesViewHost;
//    }
//
//    if (phaseSpaceCellIDs.size() == 0)
//    {
//      return;
//    }
// 
//    const int numPhaseSpacePoints = basisCache->getRefCellPointsFC().dimension(0);
//        
//    for (int phaseSpaceCellOrdinal = 0; phaseSpaceCellOrdinal < phaseSpaceCellIDs.size(); phaseSpaceCellOrdinal++)
//    {
//      auto physicalPointIterator = _phaseSpacePhysicalMeshMap->getPhysicalPointIterator(basisCache, phaseSpaceCellIDs[phaseSpaceCellOrdinal], physicalBasisCachePairs);
//      
//      int numPhysicalPoints = physicalPointIterator.numPhysicalPoints();
//      const int numVelocityPoints   = numPhaseSpacePoints / numPhysicalPoints;
//      
//      TEUCHOS_TEST_FOR_EXCEPTION(numVelocityPoints * numPhysicalPoints != numPhaseSpacePoints, std::invalid_argument, "physical points do not evenly divide phase space points");
//      
//      // For the moment, we do the copying below on host; it'll take a bit of effort to manage the maps, etc. efficiently on device.
//      // (It seems likely that this will not be a performance bottleneck.)
//      for (; physicalPointIterator.isValid(); physicalPointIterator.increment() )
//      {
//        auto physicalCellID             = physicalPointIterator.physicalCellID();
//        auto & values1D                 = physicalValuesMapHost[physicalCellID];
//        auto physicalPointOrdinal       = physicalPointIterator.pointOrdinal(); // spatialPointOrdinal + numSpatialPoints * temporalPointOrdinal;
//        auto spatialPointOrdinal        = physicalPointOrdinal % numSpatialPoints;
//        auto temporalPointOrdinal       = physicalPointOrdinal / numSpatialPoints;
//        auto pointOrdinalInPhysicalCell = physicalPointIterator.pointOrdinalInPhysicalCell();
//        double & value                  = values1D(0, pointOrdinalInPhysicalCell);
//        
//        if (dataExtentMatchesPhysicalPoints)
//        {
//          // then we only need to fill in one entry
//          valuesHost.getWritableEntry(phaseSpaceCellOrdinal, spatialPointOrdinal, temporalPointOrdinal) = value;
//        }
//        else
//        {
//          // we need to iterate over all the velocity-space points
//          for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
//          {
//            const int tensorPointOrdinal = getTensorPointOrdinal(physicalPointOrdinal, velocityPointOrdinal, numPhysicalPoints);
//            valuesHost.getWritableEntry(phaseSpaceCellOrdinal, tensorPointOrdinal) = value;
//          }
//        }
//      }
//    }
//     
//    // check the views we checked out back in
//    for ( auto & physicalCellID : physicalCellIDs )
//    {
//      SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(physicalValuesMap[physicalCellID]);
//    }
//    
//    // copy from host to device
//    // because layouts may differ, we use a copy mechanism built into Intrepid2::Data (which may be inefficient)
//    // (we could speed things up by ensuring that layouts match, or at least checking whether they do, and doing
//    //  a deep_copy if they do), though I'm inclined to doubt this will be a performance bottleneck…
//    copyHostDataToDevice(values, valuesHost);
//  }
//  
//  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
//  {
//    // at present, we require that this be called on every MPI rank that owns cells in the phase space mesh
//    // we could avoid this requirement by adding an explicit, separate communication step.
//    
//    // It is also worth noting that creating basisCache_1D below is an inefficiency: those basis values won't
//    // be reused beyond the body of this method…
//    
//    const int rank = MPIWrapper::CommWorld()->MyPID();
//    
//    TEUCHOS_TEST_FOR_EXCEPTION(basisCache == Teuchos::null, std::invalid_argument, "On rank " + std::to_string(rank) + ", null pointer on line " + std::to_string(__LINE__) );
//    TEUCHOS_TEST_FOR_EXCEPTION(_g_1D1T == Teuchos::null, std::invalid_argument, "On rank " + std::to_string(rank) + ", null pointer on line " + std::to_string(__LINE__) );
//    TEUCHOS_TEST_FOR_EXCEPTION(_phaseSpacePhysicalMeshMap == Teuchos::null, std::invalid_argument, "On rank " + std::to_string(rank) + ", null pointer on line " + std::to_string(__LINE__) );
//    
//    bool isSideCache = (basisCache->getSideIndex() != -1);
//    int physicalSpaceSideOrdinal;
//    if (!isSideCache)
//    {
//      physicalSpaceSideOrdinal = -1;
//    }
//    else // isSideCache == true
//    {
//      const int phaseSpaceSideOrdinal = basisCache->getSideIndex();
//      // what follows assumes 1D1T1V with line x line x line elements; let's throw an exception if we're not on such an element
//      auto line_x_line_x_line = CellTopology::cellTopology(CellTopology::line(), 2);
//      TEUCHOS_TEST_FOR_EXCEPTION(basisCache->cellTopology()->getKey() != line_x_line_x_line->getKey(),
//                                 std::invalid_argument, "Only phase-space line x line x line topologies are supported");
//      
//      TEUCHOS_TEST_FOR_EXCEPTION(basisCache->cellTopology()->getKey() != line_x_line_x_line->getKey(), std::invalid_argument, "Only phase-space line x line topologies are supported");
//      bool sideIsParallelToPhysicalSpace = !line_x_line_x_line->sideIsSpatial(phaseSpaceSideOrdinal);
//      if (sideIsParallelToPhysicalSpace)
//      {
//        // this side does not map to a physical side; runs parallel to physical space
//        // therefore, it instead maps to a physical cell interior (in adaptive context, possibly multiple -- this is not yet supported)
//        if (_g_1D1T->boundaryValueOnly())
//        {
//          // if _g_1D1T is a skeleton variable, then we can't evaluate it on a cell interior; we define the extrusion to be zero in this case
//          values.initialize(0.0);
//          return;
//        }
//        else
//        {
//          // we can and should evaluate _g_1D1T on the spatial element interior
//          physicalSpaceSideOrdinal = -1;
//        }
//      }
//      else
//      {
//        physicalSpaceSideOrdinal = line_x_line_x_line->getSpatialComponentSideOrdinal(phaseSpaceSideOrdinal);
//      }
//    }
//    
//    using std::map;
//    using std::pair;
//    using std::set;
//    using std::vector;
//    
//    // determine physical cell IDs we need to compute on
//    const vector<GlobalIndexType> & phaseSpaceCellIDs = basisCache->cellIDs();
//    const vector<GlobalIndexType> physicalCellIDs = _phaseSpacePhysicalMeshMap->physicalCellIDsForPhaseSpaceCells(phaseSpaceCellIDs);
//    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalValuesMap;
//    map<GlobalIndexType, Camellia::FieldContainer<double> > physicalPointsMap; // this is used for debugging; we could save something by dropping this later.
//    
//    const bool testVsTest = basisCache->testVsTest();
//    const int phaseSpaceQuadratureEnrichment = basisCache->cubatureEnrichmentDegree();
//
//    int basisCachePhysicalQuadratureEnrichment;
//    if (testVsTest)
//    {
//      // test order in both 2D and 3D meshes is relative to the H^1 order, which will match in both
//      basisCachePhysicalQuadratureEnrichment = phaseSpaceQuadratureEnrichment;
//    }
//    else
//    {
//      // trial order is 1 higher in 3D mesh because the t_n_hat/f_hat discretization has order matching H1Order
//      basisCachePhysicalQuadratureEnrichment = phaseSpaceQuadratureEnrichment + 1;
//    }
//    
//    map< GlobalIndexType , pair<BasisCachePtr, vector<int> > > physicalBasisCachePairs; // index: physical cell ID
//    for ( auto & physicalCellID : physicalCellIDs )
//    {
//      physicalBasisCachePairs[physicalCellID] = _phaseSpacePhysicalMeshMap->physicalCellBasisCache(physicalCellID, testVsTest, basisCachePhysicalQuadratureEnrichment);
//    }
//    
//    const bool inPhysicalVolume = (physicalSpaceSideOrdinal == -1);
//    for ( auto & physicalCellID : physicalCellIDs )
//    {
//      auto basisCachePair = physicalBasisCachePairs[physicalCellID];
//      auto physicalBasisCache = basisCachePair.first;
//      auto offsets            = basisCachePair.second;
//      
//      BasisCachePtr basisCacheToUse = inPhysicalVolume ? physicalBasisCache : physicalBasisCache->getSideBasisCache(physicalSpaceSideOrdinal);
//      
//      TEUCHOS_TEST_FOR_EXCEPTION(basisCacheToUse == Teuchos::null, std::invalid_argument, "On rank " + std::to_string(rank) + ", null pointer on line " + std::to_string(__LINE__) );
//      
//      const int numCells  = 1;
//      const int numPoints = basisCacheToUse->getRefCellPointsFC().dimension(0);
//      Camellia::FieldContainer<double> values(numCells,numPoints);
//      
//      _g_1D1T->values(values, basisCacheToUse);
//      physicalValuesMap[physicalCellID] = values;
//      physicalPointsMap[physicalCellID] = basisCacheToUse->getPhysicalCubaturePointsFC();
//    }
//
//    if (phaseSpaceCellIDs.size() == 0)
//    {
//      return;
//    }
// 
//    const int numPhaseSpacePoints = basisCache->getRefCellPointsFC().dimension(0);
//    
//    for (int phaseSpaceCellOrdinal = 0; phaseSpaceCellOrdinal < phaseSpaceCellIDs.size(); phaseSpaceCellOrdinal++)
//    {
//      auto physicalPointIterator = _phaseSpacePhysicalMeshMap->getPhysicalPointIterator(basisCache, phaseSpaceCellIDs[phaseSpaceCellOrdinal], physicalBasisCachePairs);
//      
//      int numPhysicalPoints = physicalPointIterator.numPhysicalPoints();
//      const int numVelocityPoints   = numPhaseSpacePoints / numPhysicalPoints;
//      
//      TEUCHOS_TEST_FOR_EXCEPTION(numVelocityPoints * numPhysicalPoints != numPhaseSpacePoints, std::invalid_argument, "physical points do not evenly divide phase space points");
//      
//      for (; physicalPointIterator.isValid(); physicalPointIterator.increment() )
//      {
//        auto physicalCellID             = physicalPointIterator.physicalCellID();
//        auto & physicalValues           = physicalValuesMap[physicalCellID];
//        auto physicalPointOrdinal       = physicalPointIterator.pointOrdinal();
//        auto pointOrdinalInPhysicalCell = physicalPointIterator.pointOrdinalInPhysicalCell();
//        double & value                  = physicalValues(0, pointOrdinalInPhysicalCell);
//        
//        for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
//        {
//          const int tensorPointOrdinal = getTensorPointOrdinal(physicalPointOrdinal, velocityPointOrdinal, numPhysicalPoints);
//          values(phaseSpaceCellOrdinal,tensorPointOrdinal) = value;
//        }
//      }
//    }
//  }
//  
//  bool boundaryValueOnly() override
//  {
//    return _g_1D1T->boundaryValueOnly();
//  }
//};

VlasovPoissonSpaceTimeFormulation::VlasovPoissonSpaceTimeFormulation(VlasovPoissonSpaceTimeFormulationChoice formulationChoice,
                                                                     double q, double eps0, double m,
                                                                     const double x0, const double xWidth, const int xElementsCoarse,
                                                                     const double t0, const double tFinal, const int tElementsCoarse,
                                                                     const double v0, const double vWidth, const int vElementsCoarse,
                                                                     const int H1Order, const int delta_k,
                                                                     const bool usePeriodicBCs, const bool enforceQuasiNeutrality,
                                                                     const int vlasovTraceEnrichment)
:
_formulationChoice(formulationChoice),
_usePeriodicBCs(usePeriodicBCs),
_enforceQuasiNeutrality(enforceQuasiNeutrality)
{
//  {
//    double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//    std::cout << "At start of VlasovPoissonSpaceTimeFormulation construction, memory HWM:      " << memoryUsageInGigabytes << " GB.\n";
//  }
  
  const int spaceDim = 1;
  const int velocityDim = 1;
  _delta_k = delta_k;
  _tn_enrichment = vlasovTraceEnrichment;
  
  _q    = ParameterFunction::parameterFunction(q);
  _eps0 = ParameterFunction::parameterFunction(eps0);
  _m    = ParameterFunction::parameterFunction(m);
  
  _n0   = ParameterFunction::parameterFunction(0.0); // will be set/updated during solve if quasi-neutrality enforced
  
  auto intervalMeshTopology = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElementsCoarse, usePeriodicBCs);
  auto intervalMeshTopologySpaceTime = MeshFactory::extrudedMeshTopology(intervalMeshTopology, {t0}, {tFinal-t0}, {tElementsCoarse});
  
  MeshTopologyPtr extrudedMeshTopology = MeshFactory::extrudedMeshTopology(intervalMeshTopologySpaceTime, {v0}, {vWidth}, {vElementsCoarse});
  
  const bool l2Trace   = true; // non-conforming
  const bool ultraweak = _formulationChoice == ULTRAWEAK;
  
  if (ultraweak)
  {
    Space tauSpace    = (spaceDim > 1) ? HDIV : HGRAD;
    Space E_hat_space = L2; // non-conforming
    Space ESpace      = (spaceDim > 1) ? VECTOR_L2 : L2;

    // fields
    VarPtr f; // Vlasov
    VarPtr E; // Poisson
    VarPtr VE; // potential (Poisson)

    // traces
    VarPtr E_hat, VE_hat, t_n_hat, f_hat; // t_n_hat: L2 trace, f_hat: H^1 trace

    // tests
    VarPtr w;   // Vlasov test function

    VarFactoryPtr poissonVF = VarFactory::varFactory();
    E  = poissonVF->fieldVarSpaceTime(S_E,  ESpace, L2); // L2 in time, too
    VE = poissonVF->fieldVarSpaceTime(S_VE, L2,     L2); // L2 in time, too
        
    TFunctionPtr<double> parity = TFunction<double>::sideParity();
    TFunctionPtr<double> n = TFunction<double>::normal();

    if (spaceDim > 1)
      E_hat = poissonVF->fluxVarSpaceOnly(S_E_HAT, E * (n * parity));
    else
      E_hat = poissonVF->fluxVarSpaceOnly(S_E_HAT, E * (Function::normal_1D() * parity));
    
    if (spaceDim > 1)
      VE_hat = poissonVF->traceVarSpaceOnly(S_VE_HAT, VE, L2);
    else
      VE_hat = poissonVF->fluxVarSpaceOnly(S_VE_HAT, VE * (Function::normal_1D() * parity), L2); // for spaceDim==1, the "normal" component is in the flux-ness of u_hat (it's a plus or minus 1)
    
    VarPtr v_Poisson   = poissonVF->testVarSpaceTime(S_V_POISSON, HGRAD, HGRAD);   // HGRAD in space, HGRAD in time
    VarPtr tau_Poisson = poissonVF->testVarSpaceTime(S_TAU_POISSON, tauSpace, HGRAD); // HDIV or HGRAD in space, HGRAD in time
    
    _poissonBF = BF::bf(poissonVF);
    
    if (spaceDim==1)
    {
      // for spaceDim==1, the "normal" component is in the flux-ness of u_hat (it's a plus or minus 1)
      _poissonBF->addTerm(VE, tau_Poisson->di(1));
      _poissonBF->addTerm( -E, tau_Poisson);
      _poissonBF->addTerm(-VE_hat, tau_Poisson);

      _poissonBF->addTerm(-E, v_Poisson->di(1));
      _poissonBF->addTerm(E_hat, v_Poisson);
    }
    else
    {
      _poissonBF->addTerm(VE, tau_Poisson->div());
      _poissonBF->addTerm(-E, tau_Poisson);
      _poissonBF->addTerm(-VE_hat, tau_Poisson->dot_normal());

      _poissonBF->addTerm(-E, v_Poisson->grad());
      _poissonBF->addTerm(E_hat, v_Poisson);
      
      // the above won't *quite* work for space-time: note the use of (spatial) div(), grad(), and dot_normal() operators.  We'll want to separate out the components of E, and do (-Ex, v_Poisson->di(1)), etc.  So if we get here, let's throw an exception:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Incorrect implementation; need to fix the spatial operators to work in a space-time context…");
    }
    
    Epetra_CommPtr comm = Teuchos::null;
    const bool allowPruning = false;
    map<int,int> trialOrderEnhancements;
    map<int,int> testOrderEnhancements;
    trialOrderEnhancements[ E_hat->ID()] = 1;
    trialOrderEnhancements[VE_hat->ID()] = 1;
    MeshPartitionPolicyPtr partitionPolicy = Teuchos::null; // use default partition policy
    MeshPtr poissonMesh = Teuchos::rcp( new Mesh(intervalMeshTopologySpaceTime, _poissonBF, H1Order, delta_k,
                                                 trialOrderEnhancements, testOrderEnhancements, partitionPolicy,
                                                 comm, allowPruning) );
    
    _poissonSoln = Solution::solution(_poissonBF, poissonMesh);
    auto poissonSolnWeakPtr = Teuchos::rcp(_poissonSoln.get(), false);
    
    FunctionPtr E_soln = Function::solution(E, poissonSolnWeakPtr);
    
    FunctionPtr v = this->vx();
    FunctionPtr m = FunctionPtr(_m);
    
    // Vlasov equation/mesh will not "know" about "space-time" as such -- no variables will be marked as space-time.
    // We can do this because the flux variable (t_n_hat) is defined on every interface, so we don't need the weird
    // only-supported-on-spatial-interfaces thing that we need for space-time Poisson.
    VarFactoryPtr vlasovVF = VarFactory::varFactory();
    
    f = vlasovVF->fieldVar(S_F);
    w = vlasovVF->testVar(S_W, HGRAD); // w lies in HGRAD of the whole 1D1T1V space
    
    if (l2Trace)
    {
      t_n_hat = vlasovVF->fluxVar(S_T_N_HAT);
    }
    else
    {
      f_hat = vlasovVF->traceVar(S_F_HAT);
    }
    
    _vlasovBF = BF::bf(vlasovVF);
    
    trialOrderEnhancements.clear();
    testOrderEnhancements.clear();
    if (l2Trace)
    {
      trialOrderEnhancements[t_n_hat->ID()] = 1 + _tn_enrichment; // +1 for non-conforming, + whatever user requested
    }
    
    MeshPtr vlasovMesh = Teuchos::rcp( new Mesh(extrudedMeshTopology, _vlasovBF, H1Order, delta_k, trialOrderEnhancements, testOrderEnhancements, partitionPolicy, comm, allowPruning) );
    
    _phaseSpacePhysicalMeshMap = Teuchos::rcp( new PhasePhysicalMeshMap( poissonMesh, vlasovMesh ) );
    auto meshMapWeakPtr        = Teuchos::rcp( _phaseSpacePhysicalMeshMap.get(), false);
    
    _E_extruded = Teuchos::rcp( new PhysicalSolutionExtrusion(_poissonSoln, E, meshMapWeakPtr) );

    // the di() methods are 1-based:
    static const int SPATIAL_INDEX  = 1;
    static const int TEMPORAL_INDEX = 2;
    static const int VELOCITY_INDEX = 3;
    
    // SpatialFilter's components are zero-based
    static const int SPATIAL_INDEX_ZERO_BASED  = SPATIAL_INDEX  - 1;
    static const int TEMPORAL_INDEX_ZERO_BASED = TEMPORAL_INDEX - 1;
    static const int VELOCITY_INDEX_ZERO_BASED = VELOCITY_INDEX - 1;
    
    _vlasovBF->addTerm(              -v * f, w->di(SPATIAL_INDEX));  // dx
    _vlasovBF->addTerm(                  -f, w->di(TEMPORAL_INDEX)); // dt
    _vlasovBF->addTerm(-(q/m)*_E_extruded*f, w->di(VELOCITY_INDEX)); // dv
    
    if (l2Trace)
    {
      _vlasovBF->addTerm(t_n_hat, w);
    }
    else
    {
      FunctionPtr n = Function::normal(); // TODO: make the normal function use Intrepid2::Data to store only a single value on each side (rather than one per point), and to "know" about the tensor product structures present (i.e., make it fast on the Vlasov mesh).
      
      // b: convective direction
      FunctionPtr b_dot_normal = Function::zero();
      b_dot_normal = b_dot_normal + n->spatialComponent(SPATIAL_INDEX) * v;
      b_dot_normal = b_dot_normal + n->spatialComponent(TEMPORAL_INDEX);
      b_dot_normal = b_dot_normal + n->spatialComponent(VELOCITY_INDEX) * (q/m) * _E_extruded;
      
      _vlasovBF->addTerm(b_dot_normal * f_hat, w);
    }

    _vlasovSoln     = Solution::solution(_vlasovBF,vlasovMesh);
    
    const int maxMomentNumber = 0; // for now, we only do the zeroth moment
    _solutionMoments = Teuchos::rcp( new SolutionMoments(maxMomentNumber, _vlasovSoln, f, poissonMesh) );
    
    // note: although the temporal dimension here is also an extrusion dimension, we do NOT indicate it as a
    //       "surviving" extrusion dimension; the extrusions referred to in SolutionMomentFunction are extrusions
    //       in the higher-dimensional *relative* to the lower-dimensional one.  Here, there is only one such dimension,
    //       the velocity dimension.
    _f_velocityIntegral = _solutionMoments->solutionMomentFunction(0);
    
    // set RHS for the two problems
    // casts to allow FunctionPtr overloads to be used:
    auto qFunction = FunctionPtr(_q);
    auto mFunction = FunctionPtr(_m);
    auto epsFxn = FunctionPtr(_eps0);
    auto n0Fxn  = _enforceQuasiNeutrality ? FunctionPtr(_n0) : Function::zero(); // _n0 will be initialized when initial values are specified, if quasi-neutrality is enforced
    auto f_velocityIntegralFxn = FunctionPtr(_f_velocityIntegral);
    auto poissonRHS = RHS::rhs();
    poissonRHS->addTerm( qFunction / epsFxn * f_velocityIntegralFxn * v_Poisson);
    _poissonSoln->setRHS(poissonRHS);
    
    auto vlasovRHS = RHS::rhs();
    _vlasovSoln->setRHS(vlasovRHS); // zero RHS for vlasov
    
    // set the test-space inner products.
    _poissonSoln->setIP(_poissonBF->graphNorm());
    
    poissonMesh->registerSolution(_poissonSoln);
    vlasovMesh->registerSolution(_vlasovSoln);
    
    IPPtr vlasovTestNorm = _vlasovBF->graphNorm();
    _vlasovSoln->setIP(vlasovTestNorm);
    
    SpatialFilterPtr leftSideFilter    = SpatialFilter::matchingX(x0);
    SpatialFilterPtr rightSideFilter   = SpatialFilter::matchingX(x0+xWidth);
    SpatialFilterPtr initialTimeFilter = SpatialFilter::matchingComponent(TEMPORAL_INDEX_ZERO_BASED, t0);
    SpatialFilterPtr minVelocityFilter = SpatialFilter::matchingComponent(VELOCITY_INDEX_ZERO_BASED, v0);
    SpatialFilterPtr maxVelocityFilter = SpatialFilter::matchingComponent(VELOCITY_INDEX_ZERO_BASED, v0+vWidth);
    
    // create entity sets for BCs:
    {
      // physical (space-time) mesh entity sets:
      // create an empty entity set.
      EntitySetPtr leftSideSet  = intervalMeshTopologySpaceTime->createEntitySet();
      EntitySetPtr rightSideSet = intervalMeshTopologySpaceTime->createEntitySet();
      
      // here we use getSidesThatMatch instead of getBoundarySidesThatMatch -- this allows matches even when periodic BCs are present.
      std::vector<IndexType> leftSideIndices  = intervalMeshTopologySpaceTime->getSidesThatMatch(leftSideFilter);
      std::vector<IndexType> rightSideIndices = intervalMeshTopologySpaceTime->getSidesThatMatch(rightSideFilter);
      
      const int sideDim = spaceDim; // sides in space-time mesh have dimension equal to spatial mesh
      
      for (const auto & leftSideIndex : leftSideIndices)
      {
        leftSideSet->addEntity(sideDim, leftSideIndex);
      }
      intervalMeshTopologySpaceTime->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X0, leftSideSet);
      
      for (const auto & rightSideIndex : rightSideIndices)
      {
        rightSideSet->addEntity(sideDim, rightSideIndex);
      }
      intervalMeshTopologySpaceTime->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X1, rightSideSet);
    }
    {
      // Vlasov side sets
      const int sideDim = spaceDim + velocityDim; // phase space, space-time: sides have dimension of (fixed-time) phase space
      auto initialTimeSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> initialTimeSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(initialTimeFilter);
      for (const auto & initialTimeSideIndex : initialTimeSideIndices)
      {
        initialTimeSideSet->addEntity(sideDim, initialTimeSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_TMIN, initialTimeSideSet);
      
      auto bottomSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> bottomSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(minVelocityFilter);
      for (const auto & bottomSideIndex : bottomSideIndices)
      {
        bottomSideSet->addEntity(sideDim, bottomSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_VMIN, bottomSideSet);
      auto topSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> topSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(maxVelocityFilter);
      for (const auto & topSideIndex : topSideIndices)
      {
        topSideSet->addEntity(sideDim, topSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_VMAX, topSideSet);
      
      SpatialFilterPtr inflowLeftFilter  = SpatialFilter::matchingX(x0)          & SpatialFilter::greaterThanOrEqualToZ(0.0);
      SpatialFilterPtr inflowRightFilter = SpatialFilter::matchingX(x0 + xWidth) & SpatialFilter::lessThanOrEqualToZ   (0.0);
      
      auto inflowLeftSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> inflowLeftSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(inflowLeftFilter);
      for (const auto & leftSideIndex : inflowLeftSideIndices)
      {
        inflowLeftSideSet->addEntity(sideDim, leftSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_INFLOW_LEFT, inflowLeftSideSet);
      auto inflowRightSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> inflowRightSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(inflowRightFilter);
      for (const auto & inflowRightSideIndex : inflowRightSideIndices)
      {
        inflowRightSideSet->addEntity(sideDim, inflowRightSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_INFLOW_RIGHT, inflowRightSideSet);
    }

    // set BCs
    // set homogeneous conditions on the v boundaries
    // if we are not periodic in x, then the user must add conditions for left and right boundaries
    auto vlasovBC = BC::bc();
    auto traceVar = l2Trace ? t_n_hat : f_hat;
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_VMIN, Function::zero());
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_VMAX, Function::zero());
    _vlasovSoln->setBC(vlasovBC);
    
    auto poissonBC = BC::bc();
    
    if (usePeriodicBCs)
    {
      poissonBC->addDirichlet(this->VE_hat(), POISSON_TAG_X0, Function::constant(0.0));
    }
    else
    {
      poissonBC->addDirichlet(this->VE_hat(), POISSON_TAG_X0, Function::constant(0.0)); // pin the potential to 0 at x0
      poissonBC->addDirichlet(this->VE_hat(), POISSON_TAG_X1, Function::constant(0.0)); // pin the potential to 0 at x0+xWidth
    }
    _poissonSoln->setBC(poissonBC);
    
    // static condensation does not work properly with GMG due to details of VelocitySpaceExtrusion; it also does not appear to speed things up for the direct solve
//    if (!_haveTableau)
//    {
//      _vlasovSoln->setUseCondensedSolve(true); // should work without issue given no tableau; we need to test with tableaus before enabling with those
//    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported VlasovPoissonSpaceTimeFormulationChoice");
  }
//  {
//    double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//    std::cout << "At conclusion of VlasovPoissonSpaceTimeFormulation construction, memory HWM: " << memoryUsageInGigabytes << " GB.\n";
//  }
  PhysicalMeshUpdater::registerForUpdates(_vlasovSoln->mesh(), _poissonSoln->mesh());
}

MeshPtr VlasovPoissonSpaceTimeFormulation::vlasovMesh()
{
  return _vlasovSoln->mesh();
}

SolutionPtr VlasovPoissonSpaceTimeFormulation::vlasovSolution()
{
  return _vlasovSoln;
}

FunctionPtr VlasovPoissonSpaceTimeFormulation::E_extruded()
{
  return _E_extruded;
}

VarPtr VlasovPoissonSpaceTimeFormulation::f()
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->trialVar(S_F);
}

VarPtr VlasovPoissonSpaceTimeFormulation::f_hat()
{
  VarFactoryPtr vf = _vlasovSoln->bf()->varFactory();
  return vf->trialVar(S_F_HAT);
}

MeshPtr VlasovPoissonSpaceTimeFormulation::poissonMesh()
{
  return _poissonSoln->mesh();
}

SolutionPtr VlasovPoissonSpaceTimeFormulation::poissonSolution()
{
  return _poissonSoln;
}

bool VlasovPoissonSpaceTimeFormulation::hasL2Trace() const {
  return true;
}

VarPtr VlasovPoissonSpaceTimeFormulation::E()
{
  return _poissonBF->varFactory()->trialVar(S_E);
}

VarPtr VlasovPoissonSpaceTimeFormulation::E_hat()
{
  return _poissonBF->varFactory()->trialVar(S_E_HAT);
}

Teuchos::RCP<PhasePhysicalMeshMap> VlasovPoissonSpaceTimeFormulation::phaseSpacePhysicalMeshMap()
{
  return _phaseSpacePhysicalMeshMap;
}

void VlasovPoissonSpaceTimeFormulation::setPotentialBCs(FunctionPtr VE_left, FunctionPtr VE_right)
{
  auto poissonBC = BC::bc();
  
  TEUCHOS_TEST_FOR_EXCEPTION(_usePeriodicBCs, std::invalid_argument, "setting left and right potential BCs is not allowed for periodic meshes");
  
  poissonBC->addDirichlet(this->VE_hat(), POISSON_TAG_X0, VE_left);  // set the left potential
  poissonBC->addDirichlet(this->VE_hat(), POISSON_TAG_X1, VE_right); // set the right potential
  
  _poissonSoln->setBC(poissonBC);
}

void VlasovPoissonSpaceTimeFormulation::setIterativeSolveOutputLevel(int azOutputLevel)
{
  _aztecOutputLevel = azOutputLevel;
}

void VlasovPoissonSpaceTimeFormulation::setRefinementThresholds(double refinementThreshold, int maxElements, int maxRefinementLevel, ErrorIndicatorPtr unrefinementErrorIndicator, bool gateRefinements)
{
  _refinementThreshold = refinementThreshold;
  _maxElements = maxElements;
  _maxRefinementLevel = maxRefinementLevel;
  _lastRefinementMaxElementError = -1.0;
  _gateRefinements = gateRefinements;
  
  if (_refStrategy == Teuchos::null)
  {
    _refStrategy = RefinementStrategy::energyErrorRefinementStrategy(_vlasovSoln, _refinementThreshold);
  }
  _refStrategy->setRelativeErrorThreshold(_refinementThreshold);
  _refStrategy->setMaxElements(_maxElements);
  _refStrategy->setMaximumRefinementLevel(_maxRefinementLevel);
  _refStrategy->setUnrefinementErrorIndicator(unrefinementErrorIndicator);
  const int H1Order = _vlasovSoln->mesh()->globalDofAssignment()->getInitialH1Order()[0];
  _refStrategy->setMaxP(H1Order); // disallow p refinements (these would require revision of VelocitySpaceExtrusion)
}

void VlasovPoissonSpaceTimeFormulation::setSubstepsForFixedPoint(int numSubsteps)
{
  _numSubsteps = numSubsteps;
}

void VlasovPoissonSpaceTimeFormulation::solve()
{
  const bool useAdaptivity = _refStrategy != Teuchos::null;
  
  if (_useGeometricMultigrid && (_gmgSolver == Teuchos::null))
  {
//    {
//      double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//      std::cout << "Before GMG construction, memory HWM: " << memoryUsageInGigabytes << " GB.\n";
//    }
    
    int kCoarse = 0;
    const double cgTol = 1e-6;
    const int maxIters = 10000;

    bool useCondensedSolve = _vlasovSoln->usesCondensedSolve();

    auto multigridStrategy = GMGOperator::V_CYCLE;
//    auto multigridStrategy = GMGOperator::SMOOTHER_ONLY;
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::IFPACK_ADDITIVE_SCHWARZ;
    GMGOperator::SmootherChoice smootherChoice = GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ; // CAMELLIA_ADDITIVE_SCHWARZ defines cells as blocks; Ifpack's defines all cells on the MPI rank as a block, making MPI scaling less predictable.  (Camellia's implementation is likely less well-optimized, though.)
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL; // as configured, this is very slow; I think it's treating all data on an MPI rank as a block…
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL;
    vector<MeshPtr> meshes;
    
    if (multigridStrategy != GMGOperator::SMOOTHER_ONLY)
    {
      meshes = GMGSolver::meshesForMultigrid(_vlasovSoln->mesh(), kCoarse, 1);
    }
    else
    {
      meshes = vector<MeshPtr> {_vlasovSoln->mesh()};
    }
    
    _gmgSolver = Teuchos::rcp( new GMGSolver(_vlasovSoln, meshes, maxIters, cgTol, multigridStrategy,
                                             Solver::getDirectSolver(true), useCondensedSolve) );
    
    _gmgSolver->setSmootherType(smootherChoice);
    
    _gmgSolver->setAztecOutput(_aztecOutputLevel);
    
//    {
//      double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//      std::cout << "After GMG construction, memory HWM:  " << memoryUsageInGigabytes << " GB.\n";
//    }
  }
  
  std::vector<int> substepIterationCounts(_numSubsteps);
  
  Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz;
  const bool exportViz = false; // for debugging of fixed-point iteration, set to true.  TODO: make this subject to external control.
  
  if (exportViz)
  {
    std::string prefix = "VlasovPoissonSpaceTimeFormulation_";
    std::ostringstream vlasovSS;
    vlasovSS << prefix << "Vlasov";
    
    std::ostringstream poissonSS;
    poissonSS << prefix << "Poisson";
      
    const std::string directoryPath = "/tmp";
    
    vlasovViz = Teuchos::rcp(new HDF5Exporter(this->vlasovMesh(), vlasovSS.str(), directoryPath));
    poissonViz = Teuchos::rcp(new HDF5Exporter (this->poissonMesh(), poissonSS.str(), directoryPath) );
  }
  
  // get the PhysicalSolutionExtrusion ready to evaluate (in context of Vlasov BF)
  static_cast<PhysicalSolutionExtrusion*>(_E_extruded.get())->updateValues();
  for (int substepOrdinal=0; substepOrdinal < _numSubsteps; substepOrdinal++)
  {
    std::cout << "Solving substep " << substepOrdinal+1 << " of " << _numSubsteps << std::endl;
    // solve for f:
    if (_gmgSolver == Teuchos::null)
    {
      _vlasovSoln->solve();
      substepIterationCounts[substepOrdinal] = -1; // -1 to indicate not an iterative solver
    }
    else
    {
      _vlasovSoln->solve(_gmgSolver);
      substepIterationCounts[substepOrdinal] = _gmgSolver->iterationCount();
    }
    updateElectricField();
    
    if (exportViz)
    {
      poissonViz->exportSolution(poissonSolution(),substepOrdinal);
      vlasovViz->exportSolution(vlasovSolution(),substepOrdinal);
    }
  }
  
  _iterationCounts.push_back(substepIterationCounts);
}

void VlasovPoissonSpaceTimeFormulation::updateElectricField()
{
  // update f integral, n0
  _solutionMoments->update();
  
  double n0;
  if (_enforceQuasiNeutrality)
  {
    double f_integral = _f_velocityIntegral->integrate(_poissonSoln->mesh());
    n0 = f_integral / _poissonMeshMeasure;
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
//    std::cout << "L^2 norm of _f_velocityIntegral: " << _f_velocityIntegral->l2norm(_poissonSoln->mesh()) << std::endl;
//    
//    const auto physicalMesh = _poissonSoln->mesh();
//    const auto & myCells = physicalMesh->cellIDsInPartition();
//    for (const auto & physicalCellID : myCells)
//    {
//      BasisCachePtr basisCache = BasisCache::basisCacheForCell(physicalMesh, physicalCellID);
//      bool success = true;
//      Teuchos::basic_FancyOStream<char> out(Teuchos::rcp(&std::cout,false));
//      testFunctionDataPathMatchesViewPath1T1V(_f_velocityIntegral, basisCache, 1e-14, out, success);
//      if (!success)
//      {
//        std::cout << "testFunctionDataPathMatchesViewPath1T1V(): failed on physical cell " << physicalCellID << std::endl;
//      }
//    }
//    
//    _poissonSoln->setWriteRHSToMatrixMarketFile(true, "poissonRHS.dat");
//    _poissonSoln->setWriteMatrixToFile(true, "poissonStiffness.dat");
//  }
  
  // solve for E, E_hat:
  _poissonSoln->solve();
    
  static_cast<PhysicalSolutionExtrusion*>(_E_extruded.get())->updateValues();
}

void VlasovPoissonSpaceTimeFormulation::setVlasovInitialConditions(FunctionPtr f0)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!_usePeriodicBCs, std::invalid_argument, "setting initial conditions without setting inflow BCs is not allowed for non-periodic meshes");

  auto vlasovBC = BC::bc();
  const bool l2Trace = hasL2Trace();
  VarPtr traceVar = l2Trace ? this->t_n_hat() : this->f_hat();
  vlasovBC->addDirichlet(traceVar, VLASOV_TAG_VMIN, Function::zero());
  vlasovBC->addDirichlet(traceVar, VLASOV_TAG_VMAX, Function::zero());
  
  if (l2Trace)
  {
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_TMIN, -f0); // minus sign thanks to outward normal on temporal interface
  }
  else
  {
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_TMIN,  f0);
  }
  _vlasovSoln->setBC(vlasovBC);
}

void VlasovPoissonSpaceTimeFormulation::setVlasovInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_left, FunctionPtr f_right)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_usePeriodicBCs, std::invalid_argument, "setting inflow BCs is not allowed for periodic meshes");

  auto vlasovBC = BC::bc();
  const bool l2Trace = hasL2Trace();
  VarPtr traceVar = l2Trace ? this->t_n_hat() : this->f_hat();
  vlasovBC->addDirichlet(traceVar, VLASOV_TAG_VMIN, Function::zero());
  vlasovBC->addDirichlet(traceVar, VLASOV_TAG_VMAX, Function::zero());
  
  if (l2Trace)
  {
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_TMIN, -f0); // minus sign thanks to outward normal on temporal interface
    
    const int VELOCITY_INDEX_ZERO_BASED = 2; // 1D1T1V: velocity is the final dimension
    
    FunctionPtr v = MonomialFunction::xi_n(VELOCITY_INDEX_ZERO_BASED, 1);
    FunctionPtr leftTrace  = - f_left * v; // minus sign is thanks to outward normal
    FunctionPtr rightTrace =  f_right * v;
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_INFLOW_LEFT,   leftTrace);
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_INFLOW_RIGHT, rightTrace);
  }
  else
  {
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_TMIN, f0);
    
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_INFLOW_LEFT,   f_left);
    vlasovBC->addDirichlet(traceVar, VLASOV_TAG_INFLOW_RIGHT, f_right);
  }
  _vlasovSoln->setBC(vlasovBC);
}

Teuchos::RCP<SolutionMoments> VlasovPoissonSpaceTimeFormulation::solutionMoments()
{
  return _solutionMoments;
}

// test variables:
VarPtr VlasovPoissonSpaceTimeFormulation::w()
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->testVar(S_W, HGRAD);
}

void VlasovPoissonSpaceTimeFormulation::setUseGeometricMultigrid(bool useGeometricMultigrid)
{
  _useGeometricMultigrid = useGeometricMultigrid;
}

VarPtr VlasovPoissonSpaceTimeFormulation::tau()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->testVar(S_TAU_POISSON);
}

int VlasovPoissonSpaceTimeFormulation::tagVlasovInitialTime() const
{
  return VLASOV_TAG_TMIN;
}

int VlasovPoissonSpaceTimeFormulation::tagVlasovLeftInflow() const
{
  return VLASOV_TAG_INFLOW_LEFT;
}

int VlasovPoissonSpaceTimeFormulation::tagVlasovRightInflow() const
{
  return VLASOV_TAG_INFLOW_RIGHT;
}

int VlasovPoissonSpaceTimeFormulation::tagVlasovMaxVelocity() const
{
  return VLASOV_TAG_VMAX;
}

int VlasovPoissonSpaceTimeFormulation::tagVlasovMinVelocity() const
{
  return VLASOV_TAG_VMIN;
}

VarPtr VlasovPoissonSpaceTimeFormulation::t_n_hat()
{
  VarFactoryPtr vf = _vlasovSoln->bf()->varFactory();
  return vf->trialVar(S_T_N_HAT);
}

VarPtr VlasovPoissonSpaceTimeFormulation::VE()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->trialVar(S_VE);
}

VarPtr VlasovPoissonSpaceTimeFormulation::VE_hat()
{
  VarFactoryPtr vf = _poissonBF->varFactory();
  return vf->trialVar(S_VE_HAT);
}

FunctionPtr VlasovPoissonSpaceTimeFormulation::vx() const
{
  // the vx component is the first one after the dimension of the physical mesh
  int vx_component_zero_based = _poissonSoln->mesh()->getDimension();
  return MonomialFunction::xi_n(vx_component_zero_based, 1);
}

FunctionPtr VlasovPoissonSpaceTimeFormulation::zerothMoment()
{
  return _f_velocityIntegral;
}

// below, we set up unit tests against the private classes above
#include "Teuchos_UnitTestHarness.hpp"

void testVelocitySpace1D1TIntegral(const bool &doRefinement, Teuchos::FancyOStream &out, bool &success)
{
  double tol = 1e-15;
  double q      = 1.0;
  double eps0   = 1.0;
  double m      = 1.0;
  double x0     = 0.0;
  double xWidth = 1.0;
  double t0     = 0.0;
  double tWidth = 1.0;
  double v0     = 0.0;
  double vWidth = 1.0;
  
  const int xElements = 4;
  const int vElements = 4;
  const int tElements = 1;
  const int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  const int delta_k = 2;
  const bool usePeriodicBCs = true;
  const bool enforceQuasiNeutrality = true;
  
  using FormulationChoice = Camellia::VlasovPoissonSpaceTimeFormulation::VlasovPoissonSpaceTimeFormulationChoice;
  std::vector<FormulationChoice> formulationChoices {
    FormulationChoice::ULTRAWEAK
  };
  
  set<GlobalIndexType> vlasovCellsToRefine;
  if (doRefinement)
  {
    vlasovCellsToRefine = {0,5,8};
  }
  
  for (auto formulationChoice : formulationChoices)
  {
    const int vlasovTraceEnrichment = 0;
    VlasovPoissonSpaceTimeFormulation form(formulationChoice,
                                           q, eps0, m,
                                           x0, xWidth, xElements,
                                           t0, tWidth, tElements,
                                           v0, vWidth, vElements,
                                           H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality,
                                           vlasovTraceEnrichment);
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr z  = Function::zn(1);
    FunctionPtr xz = x * z;
    // integral of xz dz = x * z^2 / 2
    // we integrate from v0 to v0 + vWidth
    const double expectedIntegralZ = (0.5 * (v0 + vWidth) * (v0 + vWidth) - 0.5 * v0 * v0);
    FunctionPtr expectedIntegral = x * expectedIntegralZ;
    const double expectedIntegralXZ = (0.5 * (x0 + xWidth) * (x0 + xWidth) - 0.5 * x0 * x0) * expectedIntegralZ;
    const double actualIntegralXZ   = expectedIntegral->integrate(form.poissonMesh());
    
    TEST_FLOATING_EQUALITY(expectedIntegralXZ, actualIntegralXZ, tol);
    
    if (vlasovCellsToRefine.size() > 0)
    {
      auto vlasovMesh = form.vlasovMesh();
      auto line_x_line_x_line = CellTopology::cellTopology(CellTopology::line(), 2);
      auto refPattern = RefinementPattern::regularRefinementPattern(line_x_line_x_line->getKey());
      vlasovMesh->hRefine(vlasovCellsToRefine, refPattern);
    }
    
    auto f = form.f();
    std::map<int, FunctionPtr> projectionMap;
    projectionMap[f->ID()] = xz;
    
    auto vlasovSoln = form.vlasovSolution();
    const int solutionOrdinal = 0;
    vlasovSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    
    auto poissonMesh = form.poissonMesh();
    
    SolutionMoments solutionMoments(0, vlasovSoln, f, poissonMesh);
    auto velocitySpaceIntegral = solutionMoments.solutionMomentFunction(0);
    FunctionPtr err = FunctionPtr(velocitySpaceIntegral) - expectedIntegral;
    
    solutionMoments.update();
    
    double magnitudeExpected = expectedIntegral->l2norm(poissonMesh);
    double err_l2 = err->l2norm(poissonMesh);
    
    TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
  }
}

TEUCHOS_UNIT_TEST( SolutionMomentFunctionVlasovPoissonSpaceTimeFormulation, AnalyticIntegralMatches )
{
  const bool doRefinement = false;
  testVelocitySpace1D1TIntegral(doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( SolutionMomentFunctionVlasovPoissonSpaceTimeFormulation, AnalyticIntegralMatchesWithRefinement )
{
  const bool doRefinement = true;
  testVelocitySpace1D1TIntegral(doRefinement, out, success);
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

void testVelocityExtrusion1D1T(bool doRefinement, Teuchos::FancyOStream &out, bool &success)
{
  double tol = 1e-14;
  // these constants correspond to what we will be testing in the two-stream instability.  They should not affect this test, except maybe by introducing a little roundoff error here or there.
  double q      = 1.0;
  double eps0   = 1e-4;
  double m      = 1.0;
  double x0     = 0.0;
  double k_x    = 0.2; // wavelength
  double xWidth = 2 * M_PI / k_x;
  double t0     = 0.0;
  double tWidth = 1.0;
  double v0     = 0.0;
  double vWidth = 20.0;
  
  const int xElements = 2;
  const int tElements = 1;
  const int vElements = 2;
  const int H1Order = 2; // this gives f, E (which are in L^2) a polynomial order of 1
  const int delta_k = 2;
  const bool usePeriodicBCs = true;
  const bool enforceQuasiNeutrality = true;
  
  using FormulationChoice = Camellia::VlasovPoissonSpaceTimeFormulation::VlasovPoissonSpaceTimeFormulationChoice;
  std::vector<FormulationChoice> formulationChoices {
    FormulationChoice::ULTRAWEAK
  };
  
  for (auto formulationChoice : formulationChoices)
  {
    const int vlasovTraceEnrichment = 0;
    VlasovPoissonSpaceTimeFormulation form(formulationChoice,
                                           q, eps0, m,
                                           x0, xWidth, xElements,
                                           t0, tWidth, tElements,
                                           v0, vWidth, vElements,
                                           H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality,
                                           vlasovTraceEnrichment);
    
    FunctionPtr xt = Function::xn(1) * Function::yn(1);
    
    FunctionPtr physicalSoln;
    
    auto poissonSoln = form.poissonSolution();
    std::map<int, FunctionPtr> projectionMap;
    
    auto E = form.E();
    projectionMap[E->ID()] = xt;
    physicalSoln = Function::solution(E, poissonSoln);
    
    auto poissonMesh = form.poissonMesh();
    auto vlasovMesh = form.vlasovMesh();
    
    auto meshMap = Teuchos::rcp( new PhasePhysicalMeshMap( poissonMesh, vlasovMesh ) );
    auto extrusion = Teuchos::rcp( new PhysicalSolutionExtrusion(poissonSoln, E, meshMap) );
    
    auto cellTopo = CellTopology::cellTopology(CellTopology::line(), 2);
    
    if (doRefinement)
    {
      const set<GlobalIndexType> cellIDs_2D {0,1};
      auto line_x_line_x_line = CellTopology::cellTopology(CellTopology::line(), 2);
      auto refPattern = RefinementPattern::regularRefinementPattern(line_x_line_x_line->getKey());
      vlasovMesh->hRefine(cellIDs_2D, refPattern);
    }
    
    const int solutionOrdinal = 0;
    poissonSoln->projectOntoMesh(projectionMap, solutionOrdinal);
    
    extrusion->updateValues();
    
    FunctionPtr expected = xt;
    
    FunctionPtr err = FunctionPtr(extrusion) - expected;
    
  //  double magnitudeActual = extrusion->l2norm(vlasovMesh);
  //  std::cout << "magnitudeActual = " << magnitudeActual << std::endl;
    
    double magnitudeExpected = expected->l2norm(vlasovMesh);
  //  std::cout << "magnitudeExpected = " << magnitudeExpected << std::endl;
    double err_l2 = err->l2norm(vlasovMesh);
    
    TEST_COMPARE(err_l2, <, tol * magnitudeExpected);
    
//    if (!success)
//    {
//      Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz;
//      std::string prefix = "VlasovPoissonSpaceTimeFormulation_";
//      std::ostringstream vlasovSS;
//      vlasovSS << prefix << "Vlasov";
//
//      std::ostringstream poissonSS;
//      poissonSS << prefix << "Poisson";
//
//      const std::string directoryPath = "/tmp";
//
//      vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovSS.str(), directoryPath));
//      poissonViz = Teuchos::rcp(new HDF5Exporter (form.poissonMesh(), poissonSS.str(), directoryPath) );
//
//      poissonViz->exportSolution(form.poissonSolution(),0);
//      // right now, can't export E extrusion because it only implements a Data-based values(), which HDF5Exporter does not support.
////      vlasovViz->exportSolution(form.vlasovSolution(),{expected, extrusion, err},{"exact E extrusion","actual E extrusion", "err"},0);
//    }
    
    // we now only implement data path, so we can't test that View path matches it…
//    const auto & phaseCellIDs = vlasovMesh->cellIDsInPartition();
//    for (auto phaseCellID : phaseCellIDs)
//    {
//      BasisCachePtr basisCache = BasisCache::basisCacheForCell(vlasovMesh, phaseCellID);
//      if (volumeVariable)
//      {
//        testFunctionDataPathMatchesViewPath1T1V(extrusion, basisCache, tol * magnitudeExpected, out, success);
//      }
//      else
//      {
//        const int numSides = basisCache->cellTopology()->getSideCount();
//        for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
//        {
//          auto sideCache = basisCache->getSideBasisCache(sideOrdinal);
//          testFunctionDataPathMatchesViewPath1T1V(extrusion, sideCache, tol * magnitudeExpected, out, success);
//        }
//      }
//    }
  }
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionVlasovPoissonSpaceTimeFormulation, VolumeExtrusion )
{
  const bool doRefinement = false;
  testVelocityExtrusion1D1T(doRefinement, out, success);
}

TEUCHOS_UNIT_TEST( VelocitySpaceExtrusionVlasovPoissonSpaceTimeFormulation, VolumeExtrusionWithRefinement )
{
  const bool doRefinement = true;
  testVelocityExtrusion1D1T(doRefinement, out, success);
}
