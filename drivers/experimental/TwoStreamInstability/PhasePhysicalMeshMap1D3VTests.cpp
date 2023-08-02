//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  Boltzmann1D3VFormulationTests.cpp
//
//  Created by Nate Roberts on 3/4/2020.
//
//

#include "Boltzmann1D3VFormulation.h" // defines getTensorPointOrdinal()
#include "GlobalDofAssignment.h"
#include "MeshFactory.h"
#include "PhasePhysicalMeshMap.h"
#include "PoissonFormulation.h"

#include "Teuchos_LocalTestingHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

using namespace Camellia;

namespace
{
  MeshPtr poissonUniformMesh(vector<int> elementWidths, int H1Order, bool useConformingTraces)
  {
    const int totalDim = elementWidths.size();
    const int testSpaceEnrichment = totalDim;
    double span = 1.0; // in each spatial dimension
    
    vector<double> dimensions(totalDim,span);
    
    PoissonFormulation poissonForm(totalDim, useConformingTraces, PoissonFormulation::CONTINUOUS_GALERKIN);
    
    vector<double> x0(totalDim, 0.0);
    map<int,int> emptyOrderEnhancementsMap;
    Epetra_CommPtr nullComm = Teuchos::null;
    bool allowMeshTopologyPruning = (totalDim > 1); // we allow pruning the phase space mesh, but not the physical mesh
    
    auto intervalMeshTopology = MeshFactory::intervalMeshTopology(x0[0], x0[0]+span, elementWidths[0]);
    
    const int physicalDim = 1;
    const int velocityDim = totalDim - physicalDim;
    vector<double> v0(velocityDim, 0.0);
    vector<double> velocityExtents(velocityDim, span);
    vector<int> vElementWidths;
    for (int d_v=0; d_v<velocityDim; d_v++)
    {
      vElementWidths.push_back(elementWidths[d_v+physicalDim]);
    }
    MeshTopologyPtr extrudedMeshTopology = MeshFactory::extrudedMeshTopology(intervalMeshTopology, v0, velocityExtents, vElementWidths);
    
    MeshPartitionPolicyPtr nullPartitionPolicy;
    MeshPtr mesh = Teuchos::rcp( new Mesh(extrudedMeshTopology, poissonForm.bf(), H1Order, testSpaceEnrichment,
                                          emptyOrderEnhancementsMap, emptyOrderEnhancementsMap,
                                          nullPartitionPolicy, nullComm, allowMeshTopologyPruning) );
    return mesh;
  }

  CellTopoPtr sampleCellTopology1D3V()
  {
    auto line = CellTopology::line();
    const int numExtrusionDimensions = 3;
    return CellTopology::cellTopology(line, numExtrusionDimensions);
  }

  void testPhysicalPointsAgree(const PhasePhysicalMeshMap &phasePhysicalMap, Teuchos::FancyOStream &out, bool &success)
  {
    // test each active phase space element, both volume and side BasisCache
    const int quadratureEnrichment = 2;
    const bool testVsTest = false;
    
    auto mesh_1D = phasePhysicalMap.physicalMesh();
    auto mesh_4D = phasePhysicalMap.phaseMesh();
    
    auto phaseSpaceCellIDs = mesh_4D->cellIDsInPartition();
    
    // determine 1D quadrature enrichment that will match 2D quadrature
    int basisCache1DQuadratureEnrichment = quadratureEnrichment;
    if (phaseSpaceCellIDs.size() > 0) // if no phase space cells locally, no testing will take place, and basisCache1DQuadratureEnrichment value doesn't matter
    {
      GlobalIndexType samplePhaseSpaceCellID = *phaseSpaceCellIDs.begin();
      const int maxTestDegree4D  = mesh_4D->getElementType(samplePhaseSpaceCellID)->testOrderPtr->maxBasisDegree();
      const int maxTrialDegree4D = mesh_4D->getElementType(samplePhaseSpaceCellID)->trialOrderPtr->maxBasisDegree();
      
      GlobalIndexType samplePhysicalCellID = 0; // physical mesh is replicated on every rank, so cell 0 is guaranteed to be available
      const int maxTestDegree1D  = mesh_1D->getElementType(samplePhysicalCellID)->testOrderPtr->maxBasisDegree();
      const int maxTrialDegree1D = mesh_1D->getElementType(samplePhysicalCellID)->trialOrderPtr->maxBasisDegree();
      
      if (testVsTest)
      {
        basisCache1DQuadratureEnrichment = quadratureEnrichment + (maxTestDegree4D * 2) - (maxTestDegree1D * 2);
      }
      else
      {
        // trial order for 1D mesh is one lower than that for 4D because in 4D we have a conforming trace
        // that matches the H^1 order, but in 1D the maximum order is the L^2 order
        basisCache1DQuadratureEnrichment = quadratureEnrichment + (maxTestDegree4D + maxTrialDegree4D) - (maxTestDegree1D + maxTrialDegree1D);;
      }
    }
    
    const set<GlobalIndexType> physicalCellIDs = phasePhysicalMap.physicalCellIDsForPhaseSpaceCells(phaseSpaceCellIDs);
    
    map< GlobalIndexType , pair<BasisCachePtr, vector<int> > > physicalBasisCachePairs; // index: physical cell ID
    for ( auto & physicalCellID : physicalCellIDs )
    {
      physicalBasisCachePairs[physicalCellID] = phasePhysicalMap.physicalCellBasisCache(physicalCellID, testVsTest, basisCache1DQuadratureEnrichment);
    }
    
    auto hypercube_4D = CellTopology::cellTopology(CellTopology::line(), 3); // 3: 3 tensorial dimensions
    auto leftSideOrdinal1D3V  = hypercube_4D->getExtrudedSideOrdinal(0);
    auto rightSideOrdinal1D3V = hypercube_4D->getExtrudedSideOrdinal(1);
  
    std::vector<int> phaseSpaceSideOrdinals {-1}; // -1: volume
    
    for (int sideOrdinal=0; sideOrdinal<hypercube_4D->getSideCount(); sideOrdinal++)
    {
      phaseSpaceSideOrdinals.push_back(sideOrdinal);
    }
    
    for (auto phaseSpaceCellID : phaseSpaceCellIDs)
    {
      auto phaseSpaceVolumeCache = BasisCache::basisCacheForCell(mesh_4D, phaseSpaceCellID, testVsTest, quadratureEnrichment);
      
      for (auto phaseSpaceSideOrdinal : phaseSpaceSideOrdinals)
      {
        BasisCachePtr basisCache;
        if (phaseSpaceSideOrdinal == -1)
        {
          basisCache = phaseSpaceVolumeCache;
        }
        else
        {
          basisCache = phaseSpaceVolumeCache->getSideBasisCache(phaseSpaceSideOrdinal);
        }
        
        int physicalSpaceSideOrdinal = -1;
        if (phaseSpaceSideOrdinal == rightSideOrdinal1D3V)
        {
          physicalSpaceSideOrdinal = 1;
        }
        else if (phaseSpaceSideOrdinal == leftSideOrdinal1D3V)
        {
          physicalSpaceSideOrdinal = 0;
        }
        const bool inPhysicalVolume = (physicalSpaceSideOrdinal == -1);
        auto physicalPointIterator = phasePhysicalMap.getPhysicalPointIterator(basisCache, phaseSpaceCellID, physicalBasisCachePairs);
        
        const int numPhysicalPoints   = physicalPointIterator.numPhysicalPoints();
        const int numPhaseSpacePoints = basisCache->getRefCellPointsFC().dimension(0);
        const int numVelocityPoints   = numPhaseSpacePoints / numPhysicalPoints;
        
        TEUCHOS_TEST_FOR_EXCEPTION(numVelocityPoints * numPhysicalPoints != numPhaseSpacePoints, std::invalid_argument, "physical points do not evenly divide phase space points");
        
        for (; physicalPointIterator.isValid(); physicalPointIterator.increment() )
        {
          auto physicalCellID             = physicalPointIterator.physicalCellID();
          auto physicalPointOrdinal       = physicalPointIterator.pointOrdinal();
          auto pointOrdinalInPhysicalCell = physicalPointIterator.pointOrdinalInPhysicalCell();
          
          BasisCachePtr physicalBasisCache = physicalBasisCachePairs[physicalCellID].first;
          physicalBasisCache = inPhysicalVolume ? physicalBasisCache : physicalBasisCache->getSideBasisCache(physicalSpaceSideOrdinal);
          
          bool didFailForPointIterator = false;
          
          auto &phaseSpacePhysCubPoints = basisCache->getPhysicalCubaturePointsFC();
          auto &physicalPhysCubPoints   = physicalBasisCache->getPhysicalCubaturePointsFC();
          
          for (int velocityPointOrdinal=0; velocityPointOrdinal<numVelocityPoints; velocityPointOrdinal++)
          {
            const int tensorPointOrdinal = getTensorPointOrdinal(physicalPointOrdinal, velocityPointOrdinal, numPhysicalPoints);
            {
              double x_phaseSpace = phaseSpacePhysCubPoints(0,tensorPointOrdinal,0);
              double x_physical   = physicalPhysCubPoints(0,pointOrdinalInPhysicalCell,0);
              double tol = 1e-8;
              if (std::fabs(x_phaseSpace) < tol)
              {
                // small; don't use magnitude to scale the tolerance
                if (std::fabs(x_physical) > tol)
                {
                  out << "physical/phase points do not match: " << x_physical << " ≠ " << x_phaseSpace << std::endl;;
                  success = false;
                  didFailForPointIterator = true;
                }
              }
              else
              {
                double mag = std::max(std::fabs(x_phaseSpace), std::fabs(x_physical));
                if (std::abs(x_physical-x_phaseSpace) > tol * mag)
                {
                  success = false;
                  didFailForPointIterator = true;
                  out << "physical/phase points do not match: " << x_physical << " ≠ " << x_phaseSpace << std::endl;;
                }
              }
            }
          }
          
          if (didFailForPointIterator)
          {
            out << "phase space cubature points:\n" << phaseSpacePhysCubPoints;
            out << "physical cubature points:\n" << physicalPhysCubPoints;
            
            out << "phase space reference points:\n" << basisCache->getRefCellPointsFC();
            out << "phase space reference points in volume:\n" << basisCache->getSideRefCellPointsInVolumeCoordinatesFC();
            out << "physical ref points:\n" << physicalBasisCache->getRefCellPointsFC();

          }
        }
      }
    }
  }
}

TEUCHOS_UNIT_TEST( PhasePhysicalMeshMap1D3V, RefinedMeshes )
{
  const int meshWidth = 2;
  const int H1Order = 2;
  const bool useConformingTraces = true;
  MeshPtr mesh_1D = poissonUniformMesh({meshWidth}, H1Order, useConformingTraces);
  MeshPtr mesh_4D = poissonUniformMesh({meshWidth, meshWidth, meshWidth, meshWidth}, H1Order, useConformingTraces);
  
  PhasePhysicalMeshMap phasePhysicalMap(mesh_1D, mesh_4D);
  
  std::set<GlobalIndexType> cellsToRefine4D {0};
  auto sampleCellTopo = sampleCellTopology1D3V();
  auto refPattern = RefinementPattern::regularRefinementPattern(sampleCellTopo);
  mesh_4D->hRefine(cellsToRefine4D, refPattern);
  
  // the 1D mesh should be automatically refined by phasePhysicalMap; confirm this:
  TEST_EQUALITY(meshWidth+1, mesh_1D->numActiveElements());
  
  testPhysicalPointsAgree(phasePhysicalMap, out, success);
}

TEUCHOS_UNIT_TEST( PhasePhysicalMeshMap1D3V, UniformMeshes )
{
  const int meshWidth = 2;
  const int H1Order = 2;
  const bool useConformingTraces = true;
  MeshPtr mesh_1D = poissonUniformMesh({meshWidth}, H1Order, useConformingTraces);
  MeshPtr mesh_4D = poissonUniformMesh({meshWidth, meshWidth, meshWidth, meshWidth}, H1Order, useConformingTraces);
  
  PhasePhysicalMeshMap phasePhysicalMap(mesh_1D, mesh_4D);
  
  testPhysicalPointsAgree(phasePhysicalMap, out, success);
}

TEUCHOS_UNIT_TEST( PhasePhysicalMeshMap1D3V, UnrefinedMeshes )
{
  const int meshWidth = 2;
  const int H1Order = 2;
  const bool useConformingTraces = true;
  MeshPtr mesh_1D = poissonUniformMesh({meshWidth}, H1Order, useConformingTraces);
  MeshPtr mesh_4D = poissonUniformMesh({meshWidth, meshWidth, meshWidth, meshWidth}, H1Order, useConformingTraces);
  
  PhasePhysicalMeshMap phasePhysicalMap(mesh_1D, mesh_4D);
  
  // mesh is [0,1]^4, with two divisions in each direction.
  // refine the cells with centroids: (0.25, 0.25, 0.25, 0.25), (0.25, 0.75, 0.25, 0.25)
  // (this should induce exactly one refinement in the 1D mesh)
  std::vector< std::vector<double> > centroids;
  centroids.push_back(std::vector<double>{0.25, 0.25, 0.25, 0.25});
  centroids.push_back(std::vector<double>{0.25, 0.75, 0.25, 0.25});
  
  std::vector<GlobalIndexType> cellsToRefine4D = mesh_4D->getTopology()->cellIDsWithCentroids(centroids);
  auto sampleCellTopo = sampleCellTopology1D3V();
  auto refPattern = RefinementPattern::regularRefinementPattern(sampleCellTopo);
  
  mesh_4D->hRefine(cellsToRefine4D, refPattern);
  
  // the 1D mesh should be automatically refined by phasePhysicalMap; confirm this:
  TEST_EQUALITY(meshWidth+1, mesh_1D->numActiveElements());
  
  // reverse the refinement of one of the cells
  std::set<GlobalIndexType> cellsToUnrefine4D {0};
  mesh_4D->hUnrefine(cellsToUnrefine4D);
  
  testPhysicalPointsAgree(phasePhysicalMap, out, success);
}
