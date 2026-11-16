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
  // the meshes or the configuration solution may have changed
  _configurationCellsForUnmatchedCell.clear();
  _configurationBasisCaches.clear();

  auto & phaseSpaceCellIDs = _phaseMesh->cellIDsInPartition();
      
  // determine all active descendants of the physical cells needed locally; import *their* data
  set<GlobalIndexType> activePhysicalCellIDDescendants;
  
  vector< set<GlobalIndexType> > parentCellsForLevel;
  auto physicalMeshTopo = _physicalSoln->mesh()->getTopology();
  const PhaseSpaceTopology* phaseSpaceMeshTopo = PhaseSpaceTopology::phaseSpaceTopology(_phaseMesh->getTopology().get());
  
  TEUCHOS_TEST_FOR_EXCEPTION(phaseSpaceMeshTopo == NULL, std::invalid_argument, "Phase-space MeshTopology must be an ExtrudedMeshTopology or a BoxMeshTopology");
  
  set<GlobalIndexType> physicalCellIDs;
  for (const auto & phaseSpaceCellID : phaseSpaceCellIDs)
  {
    auto physicalCellID = phaseSpaceMeshTopo->phaseSpaceConfigurationCellID(phaseSpaceCellID);
    if (physicalCellID == -1)
    {
      // The phase-space cell's projection is not a single configuration cell (see valuesForUnmatchedCells()):
      // we evaluate on the active configuration cells beneath it, and need their data.
      for (IndexType activeCellID : phaseSpaceMeshTopo->phaseSpaceActiveConfigurationCells(phaseSpaceCellID))
      {
        activePhysicalCellIDDescendants.insert(activeCellID);
      }
      continue;
    }
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
  
  
  auto extrudedMeshTopology = PhaseSpaceTopology::phaseSpaceTopology(_phaseMesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(extrudedMeshTopology == NULL, std::invalid_argument, "phase-space mesh must use an ExtrudedMeshTopology or a BoxMeshTopology");
  
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
  
  std::vector<int> unmatchedCellOrdinals; // phase-space cells whose projection is not a single configuration cell
  for (int cellOrdinal = 0; cellOrdinal < numCells; cellOrdinal++)
  {
    auto physicalCellID = extrudedMeshTopology->phaseSpaceConfigurationCellID(phaseSpaceCellIDs[cellOrdinal]);
    if (physicalCellID == -1)
    {
      unmatchedCellOrdinals.push_back(cellOrdinal);
      continue;
    }
    physicalCellIDs.insert(physicalCellID);
    phaseSpaceCellOrdinalsForPhysicalCellID[physicalCellID].push_back(cellOrdinal);
  }
  if (unmatchedCellOrdinals.size() > 0)
  {
    valuesForUnmatchedCells(values, basisCache, unmatchedCellOrdinals, physicalRefPointComponents, physicalPointCounts);
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

void PhysicalSolutionExtrusion::valuesForUnmatchedCells(Intrepid2::Data<double, DeviceType> &values, BasisCachePtr basisCache,
                                                        const std::vector<int> &cellOrdinals,
                                                        const std::vector< Kokkos::DynRankView<double, DeviceType> > &physicalRefPointComponents,
                                                        const std::vector<int> &physicalPointCounts)
{
  /*
   For phase-space cells whose projection onto configuration space is not a single configuration cell but a union of
   active ones (the configuration mesh being at least as fine as the phase-space mesh): the configuration solution
   is piecewise polynomial over the projection, and we evaluate it point by point, in the configuration cell that
   contains each point.  (Points are nudged toward the centre of the projection before they are located, so that a
   point on the boundary of the phase-space cell, as for a side cache, takes its value from inside the cell.)

   This is a host implementation that writes into values directly; it assumes host-accessible device memory.
   */
  auto physicalMesh = _physicalSoln->mesh();
  auto physicalMeshTopology = physicalMesh->getTopology();
  const int numPhysicalDimensions = physicalMesh->getDimension();
  const int numPhaseDimensions = _phaseMesh->getDimension();
  const int numVelocityDimensions = numPhaseDimensions - numPhysicalDimensions;
  const int numComponents = int(physicalRefPointComponents.size());
  TEUCHOS_TEST_FOR_EXCEPTION((numComponents != 2) || (numPhysicalDimensions != 2), std::invalid_argument,
                             "phase-space cells that do not match a configuration cell are supported for two configuration dimensions (as 1D1T) only");
  TEUCHOS_TEST_FOR_EXCEPTION((numVelocityDimensions != 1) && (numVelocityDimensions != 2), std::invalid_argument,
                             "unsupported number of velocity dimensions");
  // reference coordinates of the points in each configuration direction (one 1D component per direction)
  std::vector<std::vector<double>> refCoordinates(numComponents);
  for (int r=0; r<numComponents; r++)
  {
    auto componentHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), physicalRefPointComponents[r]);
    TEUCHOS_TEST_FOR_EXCEPTION(componentHost.extent_int(1) != 1, std::invalid_argument,
                               "expected one tensor component of reference points per configuration direction");
    refCoordinates[r].resize(componentHost.extent_int(0));
    for (int i=0; i<componentHost.extent_int(0); i++) refCoordinates[r][i] = componentHost(i,0);
  }
  const int count0 = int(refCoordinates[0].size()), count1 = int(refCoordinates[1].size());
  TEUCHOS_TEST_FOR_EXCEPTION((count0 != physicalPointCounts[0]) || (count1 != physicalPointCounts[1]), std::invalid_argument, "point counts do not match");

  const vector<GlobalIndexType> &phaseSpaceCellIDs = basisCache->cellIDs();
  for (int cellOrdinal : cellOrdinals)
  {
    const GlobalIndexType phaseSpaceCellID = phaseSpaceCellIDs[cellOrdinal];
    // the cell's extent in the configuration directions
    // (verticesForCell() expects the container to be sized already)
    Camellia::FieldContainer<double> phaseVertices(int(_phaseMesh->getTopology()->getCell(phaseSpaceCellID)->vertices().size()), numPhaseDimensions);
    _phaseMesh->getTopology()->verticesForCell(phaseVertices, phaseSpaceCellID);
    double lo[2], hi[2];
    for (int k=0; k<2; k++)
    {
      lo[k] = hi[k] = phaseVertices(0,k);
      for (int n=1; n<phaseVertices.dimension(0); n++)
      {
        lo[k] = std::min(lo[k], phaseVertices(n,k));
        hi[k] = std::max(hi[k], phaseVertices(n,k));
      }
    }
    // locate each point
    const int numPoints = count0 * count1;
    Camellia::FieldContainer<double> points(numPoints, 2), nudgedPoints(numPoints, 2);
    for (int i0=0; i0<count0; i0++)
    {
      for (int i1=0; i1<count1; i1++)
      {
        const int pointOrdinal = i0 * count1 + i1;
        const double ref[2] = {refCoordinates[0][i0], refCoordinates[1][i1]};
        for (int k=0; k<2; k++)
        {
          points(pointOrdinal,k) = lo[k] + 0.5 * (ref[k] + 1.0) * (hi[k] - lo[k]);
          const double centre = 0.5 * (lo[k] + hi[k]);
          nudgedPoints(pointOrdinal,k) = points(pointOrdinal,k) + 1e-9 * (centre - points(pointOrdinal,k));
        }
      }
    }
    // The active configuration cells that tile this cell's projection, with their extents (cached until the next
    // updateValues()): they are axis-aligned, so points are located by comparing coordinates.
    auto tilingEntry = _configurationCellsForUnmatchedCell.find(phaseSpaceCellID);
    if (tilingEntry == _configurationCellsForUnmatchedCell.end())
    {
      std::vector<ConfigurationCellBox> tiling;
      const PhaseSpaceTopology* phaseSpaceTopology = PhaseSpaceTopology::phaseSpaceTopology(_phaseMesh->getTopology().get());
      for (IndexType configurationCellID : phaseSpaceTopology->phaseSpaceActiveConfigurationCells(phaseSpaceCellID))
      {
        Camellia::FieldContainer<double> configurationVertices(int(physicalMeshTopology->getCell(configurationCellID)->vertices().size()), numPhysicalDimensions);
        physicalMeshTopology->verticesForCell(configurationVertices, configurationCellID);
        ConfigurationCellBox box;
        box.cellID = configurationCellID;
        for (int k=0; k<2; k++)
        {
          box.lo[k] = box.hi[k] = configurationVertices(0,k);
          for (int n=1; n<configurationVertices.dimension(0); n++)
          {
            box.lo[k] = std::min(box.lo[k], configurationVertices(n,k));
            box.hi[k] = std::max(box.hi[k], configurationVertices(n,k));
          }
        }
        tiling.push_back(box);
      }
      tilingEntry = _configurationCellsForUnmatchedCell.emplace(phaseSpaceCellID, tiling).first;
    }
    const std::vector<ConfigurationCellBox> &tiling = tilingEntry->second;
    std::map<int, std::vector<int>> pointsForTilingOrdinal;
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      int tilingOrdinal = -1;
      for (int i=0; (i<int(tiling.size())) && (tilingOrdinal < 0); i++)
      {
        bool inside = true;
        for (int k=0; k<2; k++) inside = inside && (nudgedPoints(pointOrdinal,k) >= tiling[i].lo[k]) && (nudgedPoints(pointOrdinal,k) <= tiling[i].hi[k]);
        if (inside) tilingOrdinal = i;
      }
      TEUCHOS_TEST_FOR_EXCEPTION(tilingOrdinal < 0, std::invalid_argument, "no configuration cell found for a phase-space point");
      pointsForTilingOrdinal[tilingOrdinal].push_back(pointOrdinal);
    }
    for (const auto &entry : pointsForTilingOrdinal)
    {
      const ConfigurationCellBox &box = tiling[entry.first];
      const IndexType configurationCellID = box.cellID;
      const std::vector<int> &pointOrdinals = entry.second;
      const double* cellLo = box.lo;
      const double* cellHi = box.hi;
      const int numCellPoints = int(pointOrdinals.size());
      Camellia::FieldContainer<double> refPoints(numCellPoints, 2);
      for (int i=0; i<numCellPoints; i++)
      {
        for (int k=0; k<2; k++)
        {
          refPoints(i,k) = -1.0 + 2.0 * (points(pointOrdinals[i],k) - cellLo[k]) / (cellHi[k] - cellLo[k]);
          refPoints(i,k) = std::max(-1.0, std::min(1.0, refPoints(i,k)));
        }
      }
      // one BasisCache per configuration cell, reused until the next updateValues()
      auto cacheEntry = _configurationBasisCaches.find(configurationCellID);
      if (cacheEntry == _configurationBasisCaches.end())
      {
        cacheEntry = _configurationBasisCaches.emplace(configurationCellID, BasisCache::basisCacheForCell(physicalMesh, configurationCellID)).first;
      }
      BasisCachePtr configurationBasisCache = cacheEntry->second;
      configurationBasisCache->setRefCellPointsFC(refPoints);
      Camellia::FieldContainer<double> cellValues(1, numCellPoints);
      _physicalSolnFxn->values(cellValues, configurationBasisCache);
      for (int i=0; i<numCellPoints; i++)
      {
        const int i0 = pointOrdinals[i] / count1, i1 = pointOrdinals[i] % count1;
        if (numVelocityDimensions == 1) values.getWritableEntry(cellOrdinal, i0, i1, 0) = cellValues(0,i);
        else                            values.getWritableEntry(cellOrdinal, i0, i1, 0, 0) = cellValues(0,i);
      }
    }
  }
}

bool PhysicalSolutionExtrusion::boundaryValueOnly()
{
  // should always be false
  return _physicalSolnFxn->boundaryValueOnly();
}

