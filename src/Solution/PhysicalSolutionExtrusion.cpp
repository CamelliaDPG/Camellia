//
//  PhysicalSolutionExtrusion.cpp
//  BoltzmannTests
//
//  Created by Roberts, Nathan V on 12/6/21.
//

#include "PhysicalSolutionExtrusion.hpp"

#include "ExtrudedMeshTopology.h"
#include "GlobalDofAssignment.h"

using namespace Camellia;

PhysicalSolutionExtrusion::PhysicalSolutionExtrusion( SolutionPtr physicalSoln, VarPtr fieldVar, MeshPtr phaseMesh)
:
Function("PhysicalSolutionExtrusion",fieldVar->rank(),variationDimensions(physicalSoln)), // varies in space(-time), but not in velocity dimensions
_physicalSoln(physicalSoln),
_fieldVar(fieldVar),
_phaseMesh(phaseMesh)
{
  auto physicalSolnWeakPtr = Teuchos::rcp(_physicalSoln.get(), false);
  _physicalSolnFxn = Function::solution(_fieldVar, physicalSolnWeakPtr);
  
  TEUCHOS_TEST_FOR_EXCEPTION(_physicalSolnFxn->rank() != 0, std::invalid_argument, "Only scalar-valued functions supported");
}

std::string PhysicalSolutionExtrusion::displayString()
{
  return _fieldVar->displayString();
}

void PhysicalSolutionExtrusion::updateValues()
{
  auto & phaseSpaceCellIDs = _phaseMesh->cellIDsInPartition();
      
  // determine all active descendants of the physical cells needed locally; import *their* data
  set<GlobalIndexType> activePhysicalCellIDDescendants;
  
  vector< set<GlobalIndexType> > parentCellsForLevel;
  auto physicalMeshTopo = _physicalSoln->mesh()->getTopology();
  ExtrudedMeshTopology* phaseSpaceMeshTopo = dynamic_cast<ExtrudedMeshTopology*>(_phaseMesh->getTopology().get());
  
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
        if (descendantLevel >= int(parentCellsForLevel.size()))
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

void PhysicalSolutionExtrusion::values(Intrepid2::Data<double, DeviceType> &values, BasisCachePtr basisCache)
{
  // We only support fully tensor-product cubature points (non-trivial PhysicalTensorPoints)
  
  auto physicalMesh = _physicalSoln->mesh();
  
  const int numPhysicalDimensions = physicalMesh->getDimension();
  const int numPhaseDimensions    = _phaseMesh->getDimension();
  const int numVelocityDimensions = numPhaseDimensions - numPhysicalDimensions;
  
  const auto &tensorPoints = basisCache->getPhysicalCubaturePoints();
  
  const bool isSideCache = basisCache->isSideCache();
  const auto &phaseRefPoints = isSideCache ? basisCache->getSideRefCellPointsInVolumeCoordinates() : basisCache->getRefCellPoints(); // TODO: figure out whether we actually need the physical points (tensorPoints above), or whether we can just use tensorRefPoints everywhere instead…
  
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
    
  std::vector<int> physicalPointCounts(tensorPoints.numTensorComponents() - numVelocityDimensions);
  
  int numPhysicalPoints = 1;
  int numPhysicalPointsInValues = 1;
  int numTensorPhysicalPointComponents = tensorPoints.numTensorComponents() - numVelocityDimensions;
  for (int r=0; r<numTensorPhysicalPointComponents; r++)
  {
    physicalPointCounts[r] = tensorPoints.getTensorComponent(r).extent_int(1);
    numPhysicalPoints *= physicalPointCounts[r];
    numPhysicalPointsInValues *= values.extent_int(1+r);
  }
  
  for (int r=numTensorPhysicalPointComponents; r<tensorPoints.numTensorComponents(); r++)
  {
    const int numPointsInTensorComponent = tensorPoints.getTensorComponent(r).extent_int(1);
    const int numPointsInValuesExtent    = values.extent_int(r+1); // (C,P0,P1,…,Pn)
    TEUCHOS_TEST_FOR_EXCEPTION(numPointsInTensorComponent != numPointsInValuesExtent, std::invalid_argument, "tensorPoints and values should match in point extents");
    TEUCHOS_TEST_FOR_EXCEPTION(values.getVariationModulus(r+1) != 1, std::invalid_argument, "values should not vary in velocity dimension(s)");
  }
  
  TEUCHOS_TEST_FOR_EXCEPTION(numPhysicalPoints != numPhysicalPointsInValues, std::invalid_argument, "numPhysicalPoints should match the product of the extents in the first dimensions of values");
  
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
  
  
  auto extrudedMeshTopology = dynamic_cast<ExtrudedMeshTopology*>(_phaseMesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(extrudedMeshTopology == NULL, std::invalid_argument, "phase-space mesh must use an ExtrudedMeshTopology");
  
  auto physicalMeshTopology = physicalMesh->getTopology();
  
  const vector<GlobalIndexType> & phaseSpaceCellIDs = basisCache->cellIDs();
  const int numCells = int(phaseSpaceCellIDs.size());
  
  if (numCells == 0) return;
  
  set<GlobalIndexType> physicalCellIDs;
  map<GlobalIndexType,vector<int> > phaseSpaceCellOrdinalsForPhysicalCellID; // vector value because the same physical cell ordinal may recur
  
  auto & scratch = SharedScratchStorage<GlobalIndexType>::sharedScratchStorage();
  
  // checkout scratch storage to allow us, for each element type, to set up device-side lookups from the physical cell ordinals to the phase-space cell ordinals
  DeviceView<GlobalIndexType> fullListOrdinals = scratch.checkoutScratchStorage({numCells},"fullListOrdinals");
  
  DeviceView<GlobalIndexType> partitionedOrdinalToFullListOffset = scratch.checkoutScratchStorage({numCells},"partitionedOrdinalToFullListOffset");

  DeviceView<GlobalIndexType> partitionedOrdinalToFullListCount = scratch.checkoutScratchStorage({numCells},"partitionedOrdinalToFullListCount");
  
  auto partitionedOrdinalToFullListOffsetHost = Kokkos::create_mirror_view(partitionedOrdinalToFullListOffset);
  auto partitionedOrdinalToFullListCountHost  = Kokkos::create_mirror_view(partitionedOrdinalToFullListCount);
  auto fullListOrdinalsHost                   = Kokkos::create_mirror_view(fullListOrdinals);
  
  for (int cellOrdinal = 0; cellOrdinal < numCells; cellOrdinal++)
  {
    auto physicalCellID = extrudedMeshTopology->physicalCellIDForPhaseSpaceCellID(phaseSpaceCellIDs[cellOrdinal]);
    physicalCellIDs.insert(physicalCellID);
    extrudedMeshTopology->physicalCellIDForPhaseSpaceCellID(phaseSpaceCellIDs[cellOrdinal]);
    phaseSpaceCellOrdinalsForPhysicalCellID[physicalCellID].push_back(cellOrdinal);
  }
  
  auto physicalGDA = physicalMesh->globalDofAssignment();
  auto elemTypeToPhysicalCellMap = physicalGDA->getElementTypePartitionForCells(physicalCellIDs);
  // TODO: worry about what happens if the physical mesh has tag-filtered variables
  
  for (auto elemTypePhysicalCellsEntry : elemTypeToPhysicalCellMap)
  {
    const auto & elemTypeKey                   = elemTypePhysicalCellsEntry.first;
    const auto & physicalCellIDsForElementType = elemTypePhysicalCellsEntry.second;
    const auto   elemType = physicalGDA->getElementTypeForKey(elemTypeKey);
    
    int numCellsForElemType = int(physicalCellIDsForElementType.size());
    auto physicalCellIDIt = physicalCellIDsForElementType.begin();
    int offset = 0;
    for (int partitionCellOrdinal=0; partitionCellOrdinal<numCellsForElemType; partitionCellOrdinal++)
    {
      const auto physicalCellID = *physicalCellIDIt++;
      const auto &fullListOrdinals = phaseSpaceCellOrdinalsForPhysicalCellID[physicalCellID];
      int count = int(fullListOrdinals.size());
      
      partitionedOrdinalToFullListCountHost(partitionCellOrdinal)  = count;
      partitionedOrdinalToFullListOffsetHost(partitionCellOrdinal) = offset;
      
      for (int i=0; i<count; i++)
      {
        auto fullListOrdinal = fullListOrdinals[i];
        fullListOrdinalsHost(offset + i) = fullListOrdinal;
      }
      
      offset += count;
    }
    
    Kokkos::deep_copy(partitionedOrdinalToFullListOffset, partitionedOrdinalToFullListOffsetHost);
    Kokkos::deep_copy(partitionedOrdinalToFullListCount, partitionedOrdinalToFullListCountHost);
    Kokkos::deep_copy(fullListOrdinals, fullListOrdinalsHost);
    
    vector<GlobalIndexType> physicalCellIDsVector(physicalCellIDsForElementType.begin(),physicalCellIDsForElementType.end());
    FieldContainer<double> physicalCellNodes = physicalMesh->physicalCellNodes(physicalCellIDsVector);
    BasisCachePtr physicalBasisCache = BasisCache::basisCacheForCellType(physicalMesh, elemType);
    physicalBasisCache->setRefCellPoints(physicalRefPoints);
    bool createSideCache = false;
    physicalBasisCache->setPhysicalCellNodes(physicalCellNodes, physicalCellIDsVector, createSideCache);
      
    auto g_dataValues = _physicalSolnFxn->checkoutValuesData(physicalBasisCache);
      
    _physicalSolnFxn->values(g_dataValues, physicalBasisCache);
      
    // copy g_dataValues into values
    using ExecutionSpace = Camellia::DeviceExecutionSpace;
    if ((numPhysicalPointComponents == 1) && (numVelocityDimensions == 1))
    {
      // 1D1V case
      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>>({0,0},{numCellsForElemType,physicalPointCounts[0]});
      
      Kokkos::parallel_for("PhysicalSolutionExtrusion::values", policy,
      KOKKOS_LAMBDA (const int &cellOrdinal, const int &x0)
      {
        const auto & value = g_dataValues(cellOrdinal,x0);
        int numPhaseSpaceCellOrdinals   = partitionedOrdinalToFullListCount(cellOrdinal);
        int phaseSpaceCellOrdinalOffset = partitionedOrdinalToFullListOffset(cellOrdinal);
        
        for (int i=0; i<numPhaseSpaceCellOrdinals; i++)
        {
          auto & phaseSpaceCellOrdinal = fullListOrdinals(i+phaseSpaceCellOrdinalOffset);
          values.getWritableEntry(phaseSpaceCellOrdinal,x0,0) = value;
        }
      });
    }
    else if ((numPhysicalPointComponents == 2) && (numVelocityDimensions == 1))
    {
      // 2D1V or 1D1T1V case (space-time 1D1V)
      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{numCellsForElemType,physicalPointCounts[0],physicalPointCounts[1]});
      
      Kokkos::parallel_for("PhysicalSolutionExtrusion::values", policy,
      KOKKOS_LAMBDA (const int &cellOrdinal, const int &x0, const int &x1)
      {
        const auto & value = g_dataValues(cellOrdinal,x0,x1);
        int numPhaseSpaceCellOrdinals   = partitionedOrdinalToFullListCount(cellOrdinal);
        int phaseSpaceCellOrdinalOffset = partitionedOrdinalToFullListOffset(cellOrdinal);
        
        for (int i=0; i<numPhaseSpaceCellOrdinals; i++)
        {
          auto & phaseSpaceCellOrdinal = fullListOrdinals(i+phaseSpaceCellOrdinalOffset);
          values.getWritableEntry(phaseSpaceCellOrdinal,x0,x1,0) = value;
        }
      });
    }
    else if ((numPhysicalPointComponents == 2) && (numVelocityDimensions == 2))
    {
      // 2D2V or 1D1T2V case (space-time 1D2V)
      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{numCellsForElemType,physicalPointCounts[0],physicalPointCounts[1]});
      
      Kokkos::parallel_for("PhysicalSolutionExtrusion::values", policy,
      KOKKOS_LAMBDA (const int &cellOrdinal, const int &x0, const int &x1)
      {
        const auto & value = g_dataValues(cellOrdinal,x0,x1);
        int numPhaseSpaceCellOrdinals   = partitionedOrdinalToFullListCount(cellOrdinal);
        int phaseSpaceCellOrdinalOffset = partitionedOrdinalToFullListOffset(cellOrdinal);
        
        for (int i=0; i<numPhaseSpaceCellOrdinals; i++)
        {
          auto & phaseSpaceCellOrdinal = fullListOrdinals(i+phaseSpaceCellOrdinalOffset);
          values.getWritableEntry(phaseSpaceCellOrdinal,x0,x1,0,0) = value;
        }
      });
    }
    else
    {
      // TODO: add support for other physical/velocity component counts
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported physical/velocity component count");
    }
    
    _physicalSolnFxn->checkinValuesData(g_dataValues);
  }
  SharedScratchStorage<GlobalIndexType>::sharedScratchStorage().checkinScratchStorage(fullListOrdinals);
  SharedScratchStorage<GlobalIndexType>::sharedScratchStorage().checkinScratchStorage(partitionedOrdinalToFullListOffset);
  SharedScratchStorage<GlobalIndexType>::sharedScratchStorage().checkinScratchStorage(partitionedOrdinalToFullListCount);
}

bool PhysicalSolutionExtrusion::boundaryValueOnly()
{
  // should always be false
  return _physicalSolnFxn->boundaryValueOnly();
}

