// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  TensorVlasovMomentSampling1D2V.cpp
//  Camellia
//

#include "TensorVlasovMomentSampling1D2V.h"

#include "BasisCache.h"
#include "BoxMeshTopology.h"
#include "CamelliaCellTools.h"
#include "Camellia_FieldContainerConversion.hpp"
#include "CubatureFactory.h"
#include "ExtrudedMeshTopology.h"
#include "GlobalDofAssignment.h"
#include "Mesh.h"
#include "MPIWrapper.h"
#include "Solution.h"
#include "TensorMaxwell1D2VMomentReductionPipeline.h"
#include "Var.h"

#include <algorithm>
#include <functional>
#include <map>

using namespace Camellia;

namespace
{
  // 1D Gauss rule on the reference line [-1,1].
  void referenceLineRule(int degree, std::vector<double> &points, std::vector<double> &weights)
  {
    CubatureFactory cubatureFactory;
    auto cubature = cubatureFactory.create(CellTopology::line(), std::vector<int>{degree});
    auto refPoints = cubature->allocateCubaturePoints();
    auto refWeights = cubature->allocateCubatureWeights();
    cubature->getCubature(refPoints, refWeights);
    auto refPointsFC = FieldContainerConversion<double>::fromTensorPoints(refPoints);
    auto refWeightsFC = FieldContainerConversion<double>::fromTensorData(refWeights);
    const int pointCount = refPointsFC.dimension(0);
    points.resize(pointCount);
    weights.resize(pointCount);
    for (int i=0; i<pointCount; i++)
    {
      points[i] = refPointsFC(i,0);
      weights[i] = refWeightsFC(i);
    }
  }

  // physical (x,t) cubature points for a configuration cell, as (P,2)
  Camellia::FieldContainer<double> configurationCubaturePoints(MeshPtr configurationMesh, GlobalIndexType cellID)
  {
    auto basisCache = BasisCache::basisCacheForCell(configurationMesh, cellID);
    const auto &points = basisCache->getPhysicalCubaturePointsFC(); // (1,P,2)
    const int pointCount = points.dimension(1);
    TEUCHOS_TEST_FOR_EXCEPTION(points.dimension(2) != 2, std::invalid_argument,
                               "computeVlasovMomentsByPhaseSpaceCell1D2V expects a 1D space-time configuration mesh.");
    Camellia::FieldContainer<double> cellPoints(pointCount, 2);
    for (int p=0; p<pointCount; p++)
    {
      cellPoints(p,0) = points(0,p,0);
      cellPoints(p,1) = points(0,p,1);
    }
    return cellPoints;
  }

  // physical (x,t) images, as (P,2), of the reference points (P,2) on a configuration cell
  Camellia::FieldContainer<double> configurationPhysicalPoints(MeshPtr configurationMesh, GlobalIndexType cellID,
                                                               const Camellia::FieldContainer<double> &referencePoints)
  {
    const int pointCount = referencePoints.dimension(0);
    Camellia::FieldContainer<double> physicalPoints(1, pointCount, 2);
    Camellia::FieldContainer<double> cellNodes = configurationMesh->physicalCellNodesForCell(cellID);
    CellTopoPtr cellTopo = configurationMesh->getElementType(cellID)->cellTopoPtr;
    CamelliaCellTools::mapToPhysicalFrame(physicalPoints, referencePoints, cellNodes, cellTopo);
    physicalPoints.resize(pointCount, 2);
    return physicalPoints;
  }
}

int Camellia::defaultVlasovMomentVelocityQuadratureDegree1D2V(SolutionPtr phaseSpaceSolution)
{
  TEUCHOS_TEST_FOR_EXCEPTION(phaseSpaceSolution == Teuchos::null, std::invalid_argument, "phaseSpaceSolution is null");
  MeshPtr phaseSpaceMesh = phaseSpaceSolution->mesh();
  int velocityQuadratureDegree = 1;
  for (GlobalIndexType phaseSpaceCellID : phaseSpaceMesh->cellIDsInPartition())
  {
    velocityQuadratureDegree = std::max(velocityQuadratureDegree,
                                        phaseSpaceMesh->globalDofAssignment()->getCubatureDegree(phaseSpaceCellID));
  }
  return MPIWrapper::max(phaseSpaceMesh->Comm(), velocityQuadratureDegree);
}

namespace
{
  using ConfigurationPointsFunction = std::function<Camellia::FieldContainer<double>(MeshPtr, GlobalIndexType)>;

  TensorVlasovMoments1D2V computeMoments(const std::vector<GlobalIndexType> &configurationCellIDs,
                                         MeshPtr configurationMesh,
                                         SolutionPtr phaseSpaceSolution,
                                         VarPtr fVar,
                                         int velocityQuadratureDegree,
                                         bool useRelativisticCurrentMoments,
                                         double lightSpeed,
                                         const ConfigurationPointsFunction &configurationPointsFunction)
{
  using Pipeline = TensorMaxwell1D2VMomentReductionPipeline;
  TEUCHOS_TEST_FOR_EXCEPTION(configurationMesh == Teuchos::null, std::invalid_argument, "configurationMesh is null");
  TEUCHOS_TEST_FOR_EXCEPTION(phaseSpaceSolution == Teuchos::null, std::invalid_argument, "phaseSpaceSolution is null");
  TEUCHOS_TEST_FOR_EXCEPTION(fVar == Teuchos::null, std::invalid_argument, "fVar is null");
  TEUCHOS_TEST_FOR_EXCEPTION(velocityQuadratureDegree < 0, std::invalid_argument, "velocityQuadratureDegree must be non-negative");
  const bool perCellVelocityQuadrature = (velocityQuadratureDegree == 0);

  MeshPtr phaseSpaceMesh = phaseSpaceSolution->mesh();
  auto phaseSpaceTopology = PhaseSpaceTopology::phaseSpaceTopology(phaseSpaceMesh->getTopology().get());
  const BoxMeshTopology* boxTopology = dynamic_cast<const BoxMeshTopology*>(phaseSpaceMesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(phaseSpaceTopology == NULL, std::invalid_argument,
                             "computeVlasovMomentsByPhaseSpaceCell1D2V requires a phase-space mesh topology (ExtrudedMeshTopology or BoxMeshTopology).");
  auto extrudedTopology = dynamic_cast<const ExtrudedMeshTopology*>(phaseSpaceMesh->getTopology()->baseMeshTopology()); // may be NULL
  TEUCHOS_TEST_FOR_EXCEPTION(phaseSpaceTopology->phaseSpaceVelocityDimensionCount() != 2, std::invalid_argument,
                             "computeVlasovMomentsByPhaseSpaceCell1D2V requires exactly two velocity extrusions.");

  // 1D velocity rules, by degree
  std::map<int,std::pair<std::vector<double>,std::vector<double>>> lineRules;
  auto lineRule = [&lineRules](int degree) -> const std::pair<std::vector<double>,std::vector<double>> & {
    auto entry = lineRules.find(degree);
    if (entry != lineRules.end()) return entry->second;
    std::pair<std::vector<double>,std::vector<double>> rule;
    referenceLineRule(degree, rule.first, rule.second);
    return lineRules.emplace(degree, rule).first->second;
  };
  // Velocity rule degrees (u_x, u_z) for a phase-space cell.  Per cell, the cell's own cubature degree, lowered in a
  // velocity direction by two per order that direction is below the cell's highest order (trial and test each lose
  // one).  On a mesh of uniform order this is the global maximum that defaultVlasovMomentVelocityQuadratureDegree1D2V
  // gives, so the result is unchanged there.
  auto velocityDegrees = [&](GlobalIndexType phaseSpaceCellID) -> std::pair<int,int> {
    if (!perCellVelocityQuadrature) return {velocityQuadratureDegree, velocityQuadratureDegree};
    auto gda = phaseSpaceMesh->globalDofAssignment();
    const int cellDegree = gda->getCubatureDegree(phaseSpaceCellID);
    const std::vector<int> orders = gda->getH1Order(phaseSpaceCellID);
    const int dim = phaseSpaceMesh->getDimension();
    if ((int(orders.size()) != dim) || (dim < 2)) return {cellDegree, cellDegree};
    const int maxOrder = *std::max_element(orders.begin(), orders.end());
    const int uxDegree = std::max(1, cellDegree - 2 * (maxOrder - orders[dim-2]));
    const int uzDegree = std::max(1, cellDegree - 2 * (maxOrder - orders[dim-1]));
    return {uxDegree, uzDegree};
  };

  // Global, rank-independent indexing of active configuration cells; partial moments are summed in this layout.
  std::set<GlobalIndexType> activeConfigurationCells = configurationMesh->getActiveCellIDsGlobal();
  std::map<GlobalIndexType,int> configurationCellOrdinal;
  for (GlobalIndexType cellID : activeConfigurationCells)
  {
    const int ordinal = static_cast<int>(configurationCellOrdinal.size());
    configurationCellOrdinal[cellID] = ordinal;
  }
  const int globalConfigurationCellCount = static_cast<int>(configurationCellOrdinal.size());

  // Uniform cubature point count is assumed (as in the Maxwell RHS adapter); determined from the first cell
  // whose points we evaluate, and checked thereafter.  Agreed across ranks below.
  int pointsPerConfigurationCell = -1;
  std::map<GlobalIndexType,Camellia::FieldContainer<double>> configurationPointsCache;
  auto configurationPointsForCell = [&](GlobalIndexType cellID) -> const Camellia::FieldContainer<double> & {
    auto entry = configurationPointsCache.find(cellID);
    if (entry != configurationPointsCache.end()) return entry->second;
    auto cellPoints = configurationPointsFunction(configurationMesh, cellID);
    if (pointsPerConfigurationCell < 0) pointsPerConfigurationCell = cellPoints.dimension(0);
    TEUCHOS_TEST_FOR_EXCEPTION(cellPoints.dimension(0) != pointsPerConfigurationCell, std::invalid_argument,
                               "computeVlasovMomentsByPhaseSpaceCell1D2V requires a uniform configuration cubature point count.");
    return configurationPointsCache.emplace(cellID, cellPoints).first->second;
  };
  for (GlobalIndexType cellID : configurationCellIDs) configurationPointsForCell(cellID);
  pointsPerConfigurationCell = MPIWrapper::max(phaseSpaceMesh->Comm(), pointsPerConfigurationCell);

  // (globalCellOrdinal, point, moment) with moment = density, x moment, z moment
  const int momentCount = 3;
  std::vector<double> globalMoments(static_cast<size_t>(globalConfigurationCellCount) * std::max(pointsPerConfigurationCell,0) * momentCount, 0.0);

  const int fID = fVar->ID();
  const set<GlobalIndexType> &ownedPhaseSpaceCells = phaseSpaceMesh->cellIDsInPartition();
  for (GlobalIndexType phaseSpaceCellID : ownedPhaseSpaceCells)
  {
    double ux0, ux1, uz0, uz1;
    phaseSpaceTopology->phaseSpaceVelocityExtents(ux0, ux1, phaseSpaceCellID, 0);
    phaseSpaceTopology->phaseSpaceVelocityExtents(uz0, uz1, phaseSpaceCellID, 1);
    const double uxCenter = 0.5 * (ux0 + ux1), uxScale = 0.5 * std::abs(ux1 - ux0);
    const double uzCenter = 0.5 * (uz0 + uz1), uzScale = 0.5 * std::abs(uz1 - uz0);
    const std::pair<int,int> degrees = velocityDegrees(phaseSpaceCellID);
    const auto &uxRule = lineRule(degrees.first);
    const auto &uzRule = lineRule(degrees.second);
    const int uxPointCount = static_cast<int>(uxRule.first.size());
    const int uzPointCount = static_cast<int>(uzRule.first.size());
    const int velocityPointsPerCell = uxPointCount * uzPointCount;
    // velocity points/weights for this cell, in the (ux-major, uz-minor) order used by the reduction pipeline
    Pipeline::HostMatrix cellVelocityPoints("velocity cell points", velocityPointsPerCell, 2);
    Pipeline::HostVector cellVelocityWeights("velocity cell weights", velocityPointsPerCell);
    for (int i=0; i<uxPointCount; i++)
    {
      for (int j=0; j<uzPointCount; j++)
      {
        const int v = i * uzPointCount + j;
        cellVelocityPoints(v,0) = uxCenter + uxScale * uxRule.first[i];
        cellVelocityPoints(v,1) = uzCenter + uzScale * uzRule.first[j];
        cellVelocityWeights(v) = uxScale * uxRule.second[i] * uzScale * uzRule.second[j];
      }
    }

    // The phase-space cell's configuration cell may have been refined (via other phase-space cells over it); then this
    // phase-space cell spans several active configuration cells, and contributes at each of their cubature points.
    std::vector<GlobalIndexType> activeConfigurationCells;
    for (IndexType configurationCellID : phaseSpaceTopology->phaseSpaceActiveConfigurationCells(phaseSpaceCellID))
    {
      activeConfigurationCells.push_back(configurationCellID);
    }

    BasisCachePtr basisCache = BasisCache::basisCacheForCell(phaseSpaceMesh, phaseSpaceCellID);
    const int cubatureDegree = phaseSpaceMesh->globalDofAssignment()->getCubatureDegree(phaseSpaceCellID);
    for (GlobalIndexType activeConfigurationCellID : activeConfigurationCells)
    {
      auto ordinalEntry = configurationCellOrdinal.find(activeConfigurationCellID);
      TEUCHOS_TEST_FOR_EXCEPTION(ordinalEntry == configurationCellOrdinal.end(), std::invalid_argument,
                                 "phase-space cell's configuration cell has no active configuration-mesh descendant");
      const auto &configurationPoints = configurationPointsForCell(activeConfigurationCellID);
      const int configurationPointCount = configurationPoints.dimension(0);

      // physical points in this phase-space cell: (configuration point) x (velocity point)
      const int cellPointCount = configurationPointCount * velocityPointsPerCell;
      Camellia::FieldContainer<double> physicalPoints(1, cellPointCount, 4);
      for (int c=0; c<configurationPointCount; c++)
      {
        for (int v=0; v<velocityPointsPerCell; v++)
        {
          const int p = c * velocityPointsPerCell + v;
          physicalPoints(0,p,0) = configurationPoints(c,0);
          physicalPoints(0,p,1) = configurationPoints(c,1);
          physicalPoints(0,p,2) = cellVelocityPoints(v,0);
          physicalPoints(0,p,3) = cellVelocityPoints(v,1);
        }
      }

      // One batched inverse map and one evaluation per (phase-space cell, configuration cell): the points are known
      // to lie in this phase-space cell.
      Camellia::FieldContainer<double> refPoints(1, cellPointCount, 4);
      if (boxTopology != NULL)
      {
        // box cells are affine images of the reference cell: invert the map directly (the Newton iteration costs much more)
        const BoxMeshTopology::Extents extents = boxTopology->cellExtents(phaseSpaceCellID);
        for (int p=0; p<cellPointCount; p++)
        {
          for (int d=0; d<4; d++)
          {
            refPoints(0,p,d) = 2.0 * (physicalPoints(0,p,d) - extents[d].first) / (extents[d].second - extents[d].first) - 1.0;
          }
        }
      }
      else
      {
        CamelliaCellTools::mapToReferenceFrame(refPoints, physicalPoints, phaseSpaceMesh->getTopology(), phaseSpaceCellID, cubatureDegree);
      }
      refPoints.resize(cellPointCount, 4);
      basisCache->setRefCellPointsFC(refPoints);
      Camellia::FieldContainer<double> fValues(1, cellPointCount);
      phaseSpaceSolution->solutionValues(fValues, fID, basisCache);

      Pipeline::HostMatrix cellFValues("phase-space cell f values", configurationPointCount, velocityPointsPerCell);
      for (int c=0; c<configurationPointCount; c++)
      {
        for (int v=0; v<velocityPointsPerCell; v++)
        {
          cellFValues(c,v) = fValues(0, c * velocityPointsPerCell + v);
        }
      }
      auto cellMoments = Pipeline::reduceSpeciesMoments(cellFValues, cellVelocityPoints, cellVelocityWeights,
                                                        useRelativisticCurrentMoments, lightSpeed);
      const size_t cellOffset = static_cast<size_t>(ordinalEntry->second) * pointsPerConfigurationCell * momentCount;
      for (int c=0; c<configurationPointCount; c++)
      {
        globalMoments[cellOffset + c * momentCount + 0] += cellMoments._density[c];
        globalMoments[cellOffset + c * momentCount + 1] += cellMoments._xProperVelocityMoment[c];
        globalMoments[cellOffset + c * momentCount + 2] += cellMoments._zProperVelocityMoment[c];
      }
    }
  }

  MPIWrapper::entryWiseSum(phaseSpaceMesh->Comm(), globalMoments);

  TensorVlasovMoments1D2V result;
  result.numConfigurationCells = static_cast<int>(configurationCellIDs.size());
  result.numPointsPerConfigurationCell = pointsPerConfigurationCell;
  // informational; defined only when the velocity mesh is a product of 1D meshes and one velocity rule is used throughout
  int uniformVelocityPointsPerCell = 0;
  if ((extrudedTopology != NULL) && !perCellVelocityQuadrature)
  {
    const int linePointCount = static_cast<int>(lineRule(velocityQuadratureDegree).first.size());
    uniformVelocityPointsPerCell = linePointCount * linePointCount;
  }
  result.numVelocityPoints = (extrudedTopology == NULL) ? 0
                           : static_cast<int>(extrudedTopology->extrusionMeshTopology(0)->activeCellCount()
                                              * extrudedTopology->extrusionMeshTopology(1)->activeCellCount()) * uniformVelocityPointsPerCell;
  const int entryCount = result.numConfigurationCells * std::max(pointsPerConfigurationCell,0);
  std::vector<double> density(entryCount), xMoment(entryCount), zMoment(entryCount);
  for (int cellOrdinal=0; cellOrdinal<result.numConfigurationCells; cellOrdinal++)
  {
    const size_t cellOffset = static_cast<size_t>(configurationCellOrdinal[configurationCellIDs[cellOrdinal]]) * pointsPerConfigurationCell * momentCount;
    for (int c=0; c<pointsPerConfigurationCell; c++)
    {
      const int entry = cellOrdinal * pointsPerConfigurationCell + c;
      density[entry] = globalMoments[cellOffset + c * momentCount + 0];
      xMoment[entry] = globalMoments[cellOffset + c * momentCount + 1];
      zMoment[entry] = globalMoments[cellOffset + c * momentCount + 2];
    }
  }
  result.moments = TensorPlanar1D2VSpeciesMoments(density, xMoment, zMoment);
  return result;
}
}

TensorVlasovMoments1D2V Camellia::computeVlasovMomentsByPhaseSpaceCell1D2V(const std::vector<GlobalIndexType> &configurationCellIDs,
                                                                           MeshPtr configurationMesh,
                                                                           SolutionPtr phaseSpaceSolution,
                                                                           VarPtr fVar,
                                                                           int velocityQuadratureDegree,
                                                                           bool useRelativisticCurrentMoments,
                                                                           double lightSpeed)
{
  return computeMoments(configurationCellIDs, configurationMesh, phaseSpaceSolution, fVar, velocityQuadratureDegree,
                        useRelativisticCurrentMoments, lightSpeed, configurationCubaturePoints);
}

TensorVlasovMoments1D2V Camellia::computeVlasovMomentsByPhaseSpaceCell1D2V(const std::vector<GlobalIndexType> &configurationCellIDs,
                                                                           MeshPtr configurationMesh,
                                                                           SolutionPtr phaseSpaceSolution,
                                                                           VarPtr fVar,
                                                                           int velocityQuadratureDegree,
                                                                           bool useRelativisticCurrentMoments,
                                                                           double lightSpeed,
                                                                           const Camellia::FieldContainer<double> &configurationReferencePoints)
{
  TEUCHOS_TEST_FOR_EXCEPTION((configurationReferencePoints.rank() != 2) || (configurationReferencePoints.dimension(1) != 2),
                             std::invalid_argument, "configurationReferencePoints must have shape (P,2)");
  auto mappedPoints = [&configurationReferencePoints](MeshPtr mesh, GlobalIndexType cellID) {
    return configurationPhysicalPoints(mesh, cellID, configurationReferencePoints);
  };
  return computeMoments(configurationCellIDs, configurationMesh, phaseSpaceSolution, fVar, velocityQuadratureDegree,
                        useRelativisticCurrentMoments, lightSpeed, mappedPoints);
}
