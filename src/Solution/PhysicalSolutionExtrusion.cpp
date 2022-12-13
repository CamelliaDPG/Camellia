//
//  PhysicalSolutionExtrusion.cpp
//  BoltzmannTests
//
//  Created by Roberts, Nathan V on 12/6/21.
//

#include "PhysicalSolutionExtrusion.hpp"

#include "ExtrudedMeshTopology.h"

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

void PhysicalSolutionExtrusion::values(Intrepid2::Data<double, DeviceType> values, BasisCachePtr basisCache)
{
  // We only support fully tensor-product cubature points (non-trivial PhysicalTensorPoints)
  
  auto physicalMesh = _physicalSoln->mesh();
  
  const int numPhysicalDimensions = physicalMesh->getDimension();
  const int numPhaseDimensions    = _phaseMesh->getDimension();
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
  const int numVelocityPoints = numPoints / numPhysicalPoints;
  
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
  
  auto extrudedMeshTopology = dynamic_cast<ExtrudedMeshTopology*>(_phaseMesh->getTopology().get());
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
    if ((numPhysicalPointComponents == 1) && (numVelocityDimensions == 1))
    {
      // 1D1V case
      auto policy = Kokkos::RangePolicy<ExecutionSpace>(0,physicalPointCounts[0]);
      
      Kokkos::parallel_for("SolutionMomentFunction_values", policy,
      KOKKOS_LAMBDA (const int &x0)
      {
        values.getWritableEntry(phaseSpaceCellOrdinal,x0,0) = g_dataValues(0,x0);
      });
    }
    else if ((numPhysicalPointComponents == 2) && (numVelocityDimensions == 1))
    {
      // 2D1V or 1D1T1V case (space-time 1D1V)
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

bool PhysicalSolutionExtrusion::boundaryValueOnly()
{
  // should always be false
  return _physicalSolnFxn->boundaryValueOnly();
}

