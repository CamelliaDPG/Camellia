//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  Boltzmann1D1VFormulationTests.cpp
//
//  Created by Nate Roberts on 3/4/2020.
//
//

#include "Boltzmann1D1VFormulation.h" // defines getTensorPointOrdinal()
#include "GlobalDofAssignment.h"
#include "MeshFactory.h"
#include "PhasePhysicalMeshMap.h"
#include "PoissonFormulation.h"

#include "Teuchos_LocalTestingHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

using namespace Camellia;

Boltzmann1D1VFormulation boltzmannFormulationUniformMesh(vector<int> elementWidths, int H1Order)
{
  const double q = 1.0;
  const double eps0 = 1.0;
  const double m = 1.0;
  const double dt = 0.1;
  const double x0 = 0.0;
  const double v0 = 0.0;
  const double xWidth = 1.0;
  const double vWidth = 1.0;
  const double xElements = elementWidths[0];
  const double vElements = elementWidths[1];
  const int delta_k = 1;
  const bool usePeriodicBCs = true;
  const bool enforceQuasiNeutrality = true;
  
  auto formulationChoice = Boltzmann1D1VFormulation::ULTRAWEAK_IMPLICIT_L2_TRACE;
  
  const int vlasovTraceEnrichment = 0;
  const int numUniformRefinements = 0;
  Boltzmann1D1VFormulation form(formulationChoice, q, eps0, m, dt,
                                x0, xWidth, xElements,
                                v0, vWidth, vElements,
                                H1Order, delta_k, usePeriodicBCs, enforceQuasiNeutrality, Teuchos::null,
                                vlasovTraceEnrichment, numUniformRefinements);
  
  return form;
}

void testPhysicalPointsAgree(const PhasePhysicalMeshMap &phasePhysicalMap, Teuchos::FancyOStream &out, bool &success)
{
  // test each active phase space element, both volume and side BasisCache
  const int quadratureEnrichment = 2;
  const bool testVsTest = false;
  
  auto mesh_1D = phasePhysicalMap.physicalMesh();
  auto mesh_2D = phasePhysicalMap.phaseMesh();
  
  auto phaseSpaceCellIDs = mesh_2D->cellIDsInPartition();
  
  // determine 1D quadrature enrichment that will match 2D quadrature
  int basisCache1DQuadratureEnrichment = quadratureEnrichment;
  if (phaseSpaceCellIDs.size() > 0) // if no phase space cells locally, no testing will take place, and basisCache1DQuadratureEnrichment value doesn't matter
  {
    GlobalIndexType samplePhaseSpaceCellID = *phaseSpaceCellIDs.begin();
    const int maxTestDegree2D  = mesh_2D->getElementType(samplePhaseSpaceCellID)->testOrderPtr->maxBasisDegree();
    const int maxTrialDegree2D = mesh_2D->getElementType(samplePhaseSpaceCellID)->trialOrderPtr->maxBasisDegree();
    
    GlobalIndexType samplePhysicalCellID = 0; // physical mesh is replicated on every rank, so cell 0 is guaranteed to be available
    const int maxTestDegree1D  = mesh_1D->getElementType(samplePhysicalCellID)->testOrderPtr->maxBasisDegree();
    const int maxTrialDegree1D = mesh_1D->getElementType(samplePhysicalCellID)->trialOrderPtr->maxBasisDegree();
    
    if (testVsTest)
    {
      basisCache1DQuadratureEnrichment = quadratureEnrichment + (maxTestDegree2D * 2) - (maxTestDegree1D * 2);
    }
    else
    {
      // trial order for 1D mesh is one lower than that for 2D because in 2D we have a conforming trace
      // that matches the H^1 order, but in 1D the maximum order is the L^2 order
      basisCache1DQuadratureEnrichment = quadratureEnrichment + (maxTestDegree2D + maxTrialDegree2D) - (maxTestDegree1D + maxTrialDegree1D);;
    }
  }
//  int basisCache1DQuadratureEnrichment;
//  if (testVsTest)
//  {
//    // test order in both 1D and 2D meshes is relative to the H^1 order, which will match in both
//    basisCache1DQuadratureEnrichment = quadratureEnrichment;
//  }
//  else
//  {
//    // trial order is 1 higher in 2D mesh because the t_n_hat/f_hat discretization has order matching H1Order
//    basisCache1DQuadratureEnrichment = quadratureEnrichment + 1;
//  }
  
  const set<GlobalIndexType> physicalCellIDs = phasePhysicalMap.physicalCellIDsForPhaseSpaceCells(phaseSpaceCellIDs);
  
  map< GlobalIndexType , pair<BasisCachePtr, vector<int> > > physicalBasisCachePairs; // index: physical cell ID
  for ( auto & physicalCellID : physicalCellIDs )
  {
    physicalBasisCachePairs[physicalCellID] = phasePhysicalMap.physicalCellBasisCache(physicalCellID, testVsTest, basisCache1DQuadratureEnrichment);
  }
  
  const std::vector<int> phaseSpaceSideOrdinals {-1, 0, 1, 2, 3}; // -1: volume
  
  for (auto phaseSpaceCellID : phaseSpaceCellIDs)
  {
    auto phaseSpaceVolumeCache = BasisCache::basisCacheForCell(mesh_2D, phaseSpaceCellID, testVsTest, quadratureEnrichment);
    
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
      
      int physicalSpaceSideOrdinal;
      auto line_x_line = CellTopology::cellTopology(CellTopology::line(), 1);
      if (line_x_line->sideIsSpatial(phaseSpaceSideOrdinal))
      {
        physicalSpaceSideOrdinal = line_x_line->getSpatialComponentSideOrdinal(phaseSpaceSideOrdinal);
      }
      else
      {
        physicalSpaceSideOrdinal = -1;
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
          out << "phase space reference points in volume:\n" << basisCache->getSideRefCellPointsInVolumeCoordinates();
          out << "physical ref points:\n" << physicalBasisCache->getRefCellPointsFC();

        }
      }
    }
  }
}

TEUCHOS_UNIT_TEST( PhasePhysicalMeshMap1D1V, RefinedMeshes )
{
  const int meshWidth = 2;
  const int H1Order = 2;
  auto form = boltzmannFormulationUniformMesh({meshWidth,meshWidth}, H1Order);
  MeshPtr mesh_1D = form.gaussMesh();
  MeshPtr mesh_2D = form.boltzmannMesh();
  
  PhasePhysicalMeshMap phasePhysicalMap(mesh_1D, mesh_2D);
  
  std::set<GlobalIndexType> cellsToRefine2D {0};
  auto line_x_line = CellTopology::cellTopology(CellTopology::line(), 1);
  auto refPattern = RefinementPattern::regularRefinementPattern(line_x_line->getKey());
  mesh_2D->hRefine(cellsToRefine2D, refPattern);
  
  // the 1D mesh should be automatically refined by phasePhysicalMap; confirm this:
  TEST_EQUALITY(meshWidth+1, mesh_1D->numActiveElements());
  
  testPhysicalPointsAgree(phasePhysicalMap, out, success);
}

TEUCHOS_UNIT_TEST( PhasePhysicalMeshMap1D1V, UniformMeshes )
{
  const int meshWidth = 2;
  const int H1Order = 2;
  auto form = boltzmannFormulationUniformMesh({meshWidth,meshWidth}, H1Order);
  MeshPtr mesh_1D = form.gaussMesh();
  MeshPtr mesh_2D = form.boltzmannMesh();
  
  PhasePhysicalMeshMap phasePhysicalMap(mesh_1D, mesh_2D);
  
  testPhysicalPointsAgree(phasePhysicalMap, out, success);
}

TEUCHOS_UNIT_TEST( PhasePhysicalMeshMap1D1V, UnrefinedMeshes )
{
  const int meshWidth = 2;
  const int H1Order = 2;
  auto form = boltzmannFormulationUniformMesh({meshWidth,meshWidth}, H1Order);
  MeshPtr mesh_1D = form.gaussMesh();
  MeshPtr mesh_2D = form.boltzmannMesh();
  
  PhasePhysicalMeshMap phasePhysicalMap(mesh_1D, mesh_2D);
  
  // refine two cells in the phase space mesh, each corresponding to the leftmost physical element
  const double x_centroid = 0.25; // two physical elements, total mesh width of 1.0
  using std::vector;
  vector<double> bottomCell {x_centroid, 0.25}; // phase space mesh has extent of two cells in v dimension, with span 1.0
  vector<double> topCell    {x_centroid, 0.75};
  
  vector<vector<double>> cellCentroids {bottomCell, topCell};
  vector<GlobalIndexType> cellsToRefine2D = mesh_2D->getTopology()->cellIDsWithCentroids(cellCentroids);

  auto line_x_line = CellTopology::cellTopology(CellTopology::line(), 1);
  auto refPattern = RefinementPattern::regularRefinementPattern(line_x_line->getKey());
  mesh_2D->hRefine(cellsToRefine2D, refPattern);
  
  // the 1D mesh should be automatically refined by phasePhysicalMap; confirm this:
  TEST_EQUALITY(meshWidth+1, mesh_1D->numActiveElements());
  
  // reverse the refinement of one of the cells
  std::set<GlobalIndexType> cellsToUnrefine2D {0};
  mesh_2D->hUnrefine(cellsToUnrefine2D);
  
  testPhysicalPointsAgree(phasePhysicalMap, out, success);
}
