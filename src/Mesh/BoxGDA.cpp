//
//  BoxGDA.cpp
//  Camellia
//
//  See BoxGDA.h.
//

#include "BoxGDA.h"

#include "InducedMeshPartitionPolicy.h"
#include "MPIWrapper.h"
#include "Mesh.h"
#include "Solution.h"
#include "VarFactory.h"

#include <Teuchos_LAPACK.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <cmath>
#include <limits>

using namespace Camellia;
using namespace std;

namespace
{
  using FC = Camellia::FieldContainer<double>;

  double legendre(int i, double x)
  {
    double P0 = 1, P1 = x;
    if (i == 0) return P0;
    for (int k=2; k<=i; k++)
    {
      double P2 = ((2*k-1) * x * P1 - (k-1) * P0) / k;
      P0 = P1;
      P1 = P2;
    }
    return P1;
  }

}

MeshPtr BoxGDA::mesh(Teuchos::RCP<BoxMeshTopology> topology, BFPtr bf, int H1Order, int delta_k,
                     MeshPartitionPolicyPtr partitionPolicy, Teuchos_CommPtr Comm,
                     std::map<int,int> trialOrderEnhancements, std::map<int,int> testOrderEnhancements)
{
  Mesh::GDAFactory factory = [](MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory,
                                MeshPartitionPolicyPtr policy, std::vector<int> H1OrderVector, int pToAddTest) -> Teuchos::RCP<GlobalDofAssignment>
  {
    return Teuchos::rcp(new BoxGDA(mesh, varFactory, dofOrderingFactory, policy, H1OrderVector, pToAddTest));
  };
  return Teuchos::rcp(new Mesh(topology, bf, vector<int>{H1Order}, delta_k, factory, trialOrderEnhancements, testOrderEnhancements, partitionPolicy, Comm));
}

BoxGDA::BoxGDA(MeshPtr mesh, VarFactoryPtr varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
               vector<int> initialH1OrderTrial, int testOrderEnhancement)
: GlobalDofAssignment(mesh, varFactory, dofOrderingFactory, partitionPolicy, initialH1OrderTrial, testOrderEnhancement, false)
{
  _boxTopology = dynamic_cast<BoxMeshTopology*>(_meshTopology.get());
  TEUCHOS_TEST_FOR_EXCEPTION(_boxTopology == NULL, std::invalid_argument, "BoxGDA requires a BoxMeshTopology");
  TEUCHOS_TEST_FOR_EXCEPTION(varFactory->traceVars().size() > 0, std::invalid_argument,
                             "BoxGDA supports field and flux trial variables only (no continuous traces)");
  _allowMeshTopologyPruning = false;
  _fieldVars = varFactory->fieldVars();
  _fluxVars = varFactory->fluxVars();
}

int BoxGDA::cellOrder(GlobalIndexType cellID)
{
  return getH1Order(cellID)[0];
}

const BoxGDA::OrderData &BoxGDA::orderData(int H1Order) const
{
  auto entry = _orderData.find(H1Order);
  TEUCHOS_TEST_FOR_EXCEPTION(entry == _orderData.end(), std::logic_error, "BoxGDA: no data for H^1 order " << H1Order);
  return entry->second;
}

const BoxGDA::OrderData &BoxGDA::orderDataForCell(GlobalIndexType cellID)
{
  const int order = cellOrder(cellID);
  auto existing = _orderData.find(order);
  if (existing != _orderData.end()) return existing->second;
  OrderData &data = _orderData[order];
  ElementTypePtr elemType = elementType(cellID);
  data.elemType = elemType;
  DofOrderingPtr trialOrdering = elemType->trialOrderPtr;
  const int dim = _boxTopology->getDimension();
  const int sideCount = _boxTopology->cellTopology()->getSideCount();
  for (VarPtr field : _fieldVars)
  {
    data.fieldOffsets.push_back(data.fieldDofs);
    data.fieldDofs += trialOrdering->getBasisCardinality(field->ID(), VOLUME_INTERIOR_SIDE_INDEX);
  }
  for (VarPtr flux : _fluxVars)
  {
    const int cardinality = trialOrdering->getBasisCardinality(flux->ID(), 0);
    // Identify the flux space: find which products of Legendre polynomials (one degree per tangential direction) the
    // side basis reproduces.  They must account for the whole basis, and the set of multi-indices must be downward
    // closed, which makes the space invariant under the axis-aligned affine maps between a patch and its faces.
    // (Q_p and Camellia's serendipity spaces are both of this kind.)
    const int tangentialCount = dim - 1;
    std::vector<std::vector<int>> indexSet;
    int degree = 0;
    {
      const IndexType sampleCellID = cellID;
      const int maxDegree = order + 2; // comfortably above the flux degree
      // enough quadrature points that a product the basis does not contain cannot match it at all of them
      const int cubatureEnrichment = 2 * maxDegree;
      BasisCachePtr sampleCache = Teuchos::rcp(new BasisCache(elemType, MeshPtr(Teuchos::null), false, cubatureEnrichment));
      FC sampleNodes = _boxTopology->physicalCellNodesForCell(sampleCellID, true);
      sampleCache->setPhysicalCellNodes(sampleNodes, {GlobalIndexType(sampleCellID)}, true);
      const BoxMeshTopology::Extents extents = _boxTopology->cellExtents(sampleCellID);
      const int side = 0, normal = _boxTopology->faceForSideOrdinal(side) / 2;
      BasisCachePtr sideCache = sampleCache->getSideBasisCache(side);
      BasisPtr basis = trialOrdering->getBasis(flux->ID(), side);
      auto basisValues = sideCache->getTransformedValues(basis, OP_VALUE);
      const FC &points = sideCache->getPhysicalCubaturePointsFC();
      const FC &weightedMeasures = sideCache->getWeightedMeasures();
      const int P = points.dimension(1), n = cardinality;
      std::vector<double> M(n*n, 0.0), g(P), c(n);
      for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
          for (int q=0; q<P; q++)
            M[j*n+i] += (*basisValues)(0,i,q) * (*basisValues)(0,j,q) * weightedMeasures(0,q);
      Teuchos::LAPACK<int,double> lapack;
      int info;
      lapack.POTRF('U', n, M.data(), n, &info);
      TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxGDA: could not factor the flux side mass matrix");
      std::vector<int> index(tangentialCount, 0);
      while (true) // the first tangential direction is the most significant digit
      {
        double maxValue = 0;
        for (int q=0; q<P; q++)
        {
          double value = 1.0;
          int t = 0;
          for (int k=0; k<dim; k++)
          {
            if (k == normal) continue;
            value *= legendre(index[t++], -1 + 2 * (points(0,q,k) - extents[k].first) / (extents[k].second - extents[k].first));
          }
          g[q] = value;
          maxValue = std::max(maxValue, std::abs(value));
        }
        std::fill(c.begin(), c.end(), 0.0);
        for (int i=0; i<n; i++) for (int q=0; q<P; q++) c[i] += (*basisValues)(0,i,q) * g[q] * weightedMeasures(0,q);
        lapack.POTRS('U', n, 1, M.data(), n, c.data(), n, &info);
        double maxResidual = 0;
        for (int q=0; q<P; q++)
        {
          double value = 0;
          for (int i=0; i<n; i++) value += c[i] * (*basisValues)(0,i,q);
          maxResidual = std::max(maxResidual, std::abs(value - g[q]));
        }
        if (maxResidual < 1e-9 * maxValue)
        {
          indexSet.push_back(index);
          for (int i : index) degree = std::max(degree, i);
        }
        int t = tangentialCount - 1;
        for (; t>=0; t--)
        {
          if (++index[t] <= maxDegree) break;
          index[t] = 0;
        }
        if (t < 0) break;
      }
      TEUCHOS_TEST_FOR_EXCEPTION(int(indexSet.size()) != cardinality, std::invalid_argument,
                                 "BoxGDA: the flux side basis is not spanned by products of Legendre polynomials (found "
                                 << indexSet.size() << " such products in a basis of cardinality " << cardinality << ")");
      std::set<std::vector<int>> members(indexSet.begin(), indexSet.end());
      for (const std::vector<int> &member : indexSet)
      {
        for (int t=0; t<tangentialCount; t++)
        {
          if (member[t] == 0) continue;
          std::vector<int> lower = member;
          lower[t]--;
          TEUCHOS_TEST_FOR_EXCEPTION(members.find(lower) == members.end(), std::invalid_argument,
                                     "BoxGDA: the flux polynomial space is not invariant under restriction to sub-boxes");
        }
      }
    }
    for (int side=1; side<sideCount; side++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(trialOrdering->getBasisCardinality(flux->ID(), side) != cardinality, std::invalid_argument,
                                 "BoxGDA: flux basis cardinality varies from side to side");
    }
    data.fluxDegrees.push_back(degree);
    data.fluxIndexSets.push_back(indexSet);
    data.fluxOffsets.push_back(data.fluxDofs);
    data.fluxDofs += cardinality;
  }
  // the flux spaces of successive orders must be nested, for the minimum rule
  for (const auto &[otherOrder, otherData] : _orderData)
  {
    if (otherOrder == order) continue;
    const OrderData &lower = (otherOrder < order) ? otherData : data, &higher = (otherOrder < order) ? data : otherData;
    for (size_t fluxOrdinal=0; fluxOrdinal<_fluxVars.size(); fluxOrdinal++)
    {
      std::set<std::vector<int>> higherSet(higher.fluxIndexSets[fluxOrdinal].begin(), higher.fluxIndexSets[fluxOrdinal].end());
      for (const auto &index : lower.fluxIndexSets[fluxOrdinal])
      {
        TEUCHOS_TEST_FOR_EXCEPTION(higherSet.find(index) == higherSet.end(), std::invalid_argument,
                                   "BoxGDA: flux spaces of different orders are not nested");
      }
    }
  }
  data.basisCache = Teuchos::rcp(new BasisCache(elemType));
  data.parentBasisCache = Teuchos::rcp(new BasisCache(elemType));
  return data;
}

GlobalDofAssignmentPtr BoxGDA::deepCopy()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BoxGDA::deepCopy() is not supported");
}

void BoxGDA::didChangePartitionPolicy() { rebuildLookups(); }

void BoxGDA::didHRefine(const set<GlobalIndexType> &parentCellIDs)
{
  for (GlobalIndexType parentCellID : parentCellIDs)
  {
    const int parentDeltaP = getPRefinementDegree(parentCellID);
    if (parentDeltaP == 0) continue;
    for (IndexType childID : _boxTopology->getCell(parentCellID)->getChildIndices(_meshTopology))
    {
      setPRefinementDegree(childID, std::max(parentDeltaP, getPRefinementDegree(childID)));
    }
  }
  this->GlobalDofAssignment::didHRefine(parentCellIDs);
}

void BoxGDA::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP)
{
  // as GDAMinimumRule: record the old element types, raise the orders, then project the registered solutions
  std::map<GlobalIndexType, ElementTypePtr> oldTypes;
  const std::set<IndexType> &knownCells = _boxTopology->getLocallyKnownActiveCellIndices();
  for (GlobalIndexType cellID : cellIDs)
  {
    if (knownCells.find(cellID) != knownCells.end()) oldTypes[cellID] = elementType(cellID);
  }
  this->GlobalDofAssignment::didPRefine(cellIDs, deltaP);
  _cellMaps.clear();
  Teuchos::LAPACK<int,double> lapack;
  for (const auto &[cellID, oldType] : oldTypes)
  {
    // every cell for which a solution holds coefficients: owners, and other ranks that imported them
    bool anyCoefficients = false;
    for (SolutionPtr solution : _registeredSolutions)
      for (int ordinal=0; ordinal<solution->numSolutions(); ordinal++)
        if (solution->solutionForCellID()[ordinal].count(cellID) > 0) anyCoefficients = true;
    if (!anyCoefficients) continue;
    // (Solution's generic projection needs RefinementPatterns.)  The old bases lie in the new spaces, so the L2
    // projection onto the new basis is exact, on the volume for fields and on each side for fluxes.
    const OrderData &newData = orderDataForCell(cellID);
    DofOrderingPtr oldOrdering = oldType->trialOrderPtr, newOrdering = newData.elemType->trialOrderPtr;
    const BasisCachePtr &cache = newData.basisCache;
    FC nodes = _boxTopology->physicalCellNodesForCell(cellID, true);
    cache->setPhysicalCellNodes(nodes, {cellID}, true);
    // (old basis -> new basis) coefficient matrix for one basis pair on one cache: C (n_new x n_old), column-major
    auto transfer = [&](BasisCachePtr basisCache, BasisPtr oldBasis, BasisPtr newBasis, std::vector<double> &C)
    {
      auto newValues = basisCache->getTransformedValues(newBasis, OP_VALUE);
      auto oldValues = basisCache->getTransformedValues(oldBasis, OP_VALUE);
      const FC &weightedMeasures = basisCache->getWeightedMeasures();
      const int n = newBasis->getCardinality(), m = oldBasis->getCardinality(), P = weightedMeasures.dimension(1);
      std::vector<double> M(n*n, 0.0);
      C.assign(n*m, 0.0);
      for (int i=0; i<n; i++)
      {
        for (int j=0; j<n; j++)
          for (int q=0; q<P; q++) M[j*n+i] += (*newValues)(0,i,q) * (*newValues)(0,j,q) * weightedMeasures(0,q);
        for (int j=0; j<m; j++)
          for (int q=0; q<P; q++) C[j*n+i] += (*newValues)(0,i,q) * (*oldValues)(0,j,q) * weightedMeasures(0,q);
      }
      int info;
      lapack.POSV('U', n, m, M.data(), n, C.data(), n, &info);
      TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxGDA::didPRefine: projection failed");
    };
    struct Block { std::vector<int> oldIndices, newIndices; std::vector<double> C; };
    std::vector<Block> blocks;
    for (VarPtr field : _fieldVars)
    {
      Block block;
      block.oldIndices = oldOrdering->getDofIndices(field->ID());
      block.newIndices = newOrdering->getDofIndices(field->ID());
      transfer(cache, oldOrdering->getBasis(field->ID()), newOrdering->getBasis(field->ID()), block.C);
      blocks.push_back(block);
    }
    const int sideCount = _boxTopology->cellTopology()->getSideCount();
    for (int side=0; side<sideCount; side++)
    {
      for (VarPtr flux : _fluxVars)
      {
        Block block;
        block.oldIndices = oldOrdering->getDofIndices(flux->ID(), side);
        block.newIndices = newOrdering->getDofIndices(flux->ID(), side);
        transfer(cache->getSideBasisCache(side), oldOrdering->getBasis(flux->ID(), side), newOrdering->getBasis(flux->ID(), side), block.C);
        blocks.push_back(block);
      }
    }
    for (SolutionPtr solution : _registeredSolutions)
    {
      for (int ordinal=0; ordinal<solution->numSolutions(); ordinal++)
      {
        const auto &coefficientsForCell = solution->solutionForCellID()[ordinal];
        auto entry = coefficientsForCell.find(cellID);
        if (entry == coefficientsForCell.end()) continue;
        const FC &oldCoefficients = entry->second;
        TEUCHOS_TEST_FOR_EXCEPTION(int(oldCoefficients.size()) != oldOrdering->totalDofs(), std::logic_error,
                                   "BoxGDA::didPRefine: solution coefficients do not match the old element type");
        FC newCoefficients(newOrdering->totalDofs());
        for (const Block &block : blocks)
        {
          const int n = int(block.newIndices.size()), m = int(block.oldIndices.size());
          for (int i=0; i<n; i++)
          {
            double value = 0;
            for (int j=0; j<m; j++) value += block.C[j*n+i] * oldCoefficients[block.oldIndices[j]];
            newCoefficients[block.newIndices[i]] = value;
          }
        }
        solution->setLocalCoefficientsForCell(cellID, newCoefficients, ordinal);
      }
    }
  }
}

void BoxGDA::setAllowMeshTopologyPruning(bool allowMeshTopologyPruning)
{
  // the topology is replicated: nothing to prune
  _allowPruning = false;
  _allowMeshTopologyPruning = false;
}

bool BoxGDA::allowMeshTopologyPruning() { return false; }

IndexType BoxGDA::patchCount() const { return IndexType(_fluxDofOffsetForPatch.size()); }

void BoxGDA::rebuildLookups()
{
  TEUCHOS_FUNC_TIME_MONITOR("BoxGDA::rebuildLookups");
  _cellMaps.clear();
  _cellSideParitiesForCellID.clear(); // boundary status and hence parities are a function of the topology alone, but be safe

  const BoxForest::Skeleton &skeleton = _boxTopology->skeleton();
  const int numPatches = int(skeleton.patches.size());
  const int numPartitions = int(_partitions.size());

  vector<int> partitionForCell(_boxTopology->cellCount(), -1);
  for (int partition=0; partition<numPartitions; partition++)
  {
    for (GlobalIndexType cellID : _partitions[partition]) partitionForCell[cellID] = partition;
  }
  vector<vector<int>> patchesForPartition(numPartitions);
  for (int patchOrdinal=0; patchOrdinal<numPatches; patchOrdinal++)
  {
    const BoxForest::Patch &patch = skeleton.patches[patchOrdinal];
    const int owningCell = patch.minusFaces.empty() ? patch.plusFaces[0].cell : patch.minusFaces[0].cell;
    const int partition = partitionForCell[owningCell];
    TEUCHOS_TEST_FOR_EXCEPTION(partition < 0, std::logic_error, "BoxGDA: active cell without a partition");
    patchesForPartition[partition].push_back(patchOrdinal);
  }

  // every rank knows every cell's order (p-refinements are applied on all ranks)
  for (IndexType cellID : _boxTopology->getLocallyKnownActiveCellIndices()) orderDataForCell(cellID);
  _patchOrder.assign(numPatches, 0);
  for (int patchOrdinal=0; patchOrdinal<numPatches; patchOrdinal++)
  {
    const BoxForest::Patch &patch = skeleton.patches[patchOrdinal];
    int order = std::numeric_limits<int>::max();
    for (const auto *faces : {&patch.minusFaces, &patch.plusFaces})
      for (const BoxForest::Face &face : *faces) order = std::min(order, cellOrder(face.cell));
    _patchOrder[patchOrdinal] = order;
  }

  _fieldDofOffsetForCell.assign(_boxTopology->cellCount(), -1);
  _fluxDofOffsetForPatch.assign(numPatches, -1);
  _partitionDofOffsets.assign(numPartitions + 1, 0);
  const int rank = _partitionPolicy->Comm()->getRank();
  GlobalIndexType offset = 0;
  for (int partition=0; partition<numPartitions; partition++)
  {
    _partitionDofOffsets[partition] = offset;
    for (GlobalIndexType cellID : _partitions[partition])
    {
      _fieldDofOffsetForCell[cellID] = offset;
      offset += orderData(cellOrder(cellID)).fieldDofs;
    }
    const GlobalIndexType fluxStart = offset;
    for (int patchOrdinal : patchesForPartition[partition])
    {
      _fluxDofOffsetForPatch[patchOrdinal] = offset;
      offset += orderData(_patchOrder[patchOrdinal]).fluxDofs;
    }
    if (partition == rank)
    {
      _myFieldDofCount = fluxStart - _partitionDofOffsets[partition];
      _myFluxDofCount = offset - fluxStart;
    }
  }
  _partitionDofOffsets[numPartitions] = offset;
}

GlobalIndexType BoxGDA::globalDofCount()
{
  TEUCHOS_TEST_FOR_EXCEPTION(_partitionDofOffsets.size() == 0, std::invalid_argument, "BoxGDA::globalDofCount() called before rebuildLookups()");
  return _partitionDofOffsets.back();
}

set<GlobalIndexType> BoxGDA::globalDofIndicesForPartition(PartitionIndexType partitionNumber)
{
  if (partitionNumber == -1) partitionNumber = _partitionPolicy->Comm()->getRank();
  set<GlobalIndexType> indices;
  for (GlobalIndexType i=_partitionDofOffsets[partitionNumber]; i<_partitionDofOffsets[partitionNumber+1]; i++) indices.insert(indices.end(), i);
  return indices;
}

bool BoxGDA::isLocallyOwnedGlobalDofIndex(GlobalIndexType globalDofIndex) const
{
  const int rank = _partitionPolicy->Comm()->getRank();
  return (globalDofIndex >= _partitionDofOffsets[rank]) && (globalDofIndex < _partitionDofOffsets[rank+1]);
}

PartitionIndexType BoxGDA::partitionForGlobalDofIndex(GlobalIndexType globalDofIndex)
{
  auto upperBound = std::upper_bound(_partitionDofOffsets.begin(), _partitionDofOffsets.end(), globalDofIndex);
  const int partition = int(upperBound - _partitionDofOffsets.begin()) - 1;
  TEUCHOS_TEST_FOR_EXCEPTION((partition < 0) || (partition >= int(_partitionDofOffsets.size()) - 1), std::invalid_argument, "partition for globalDofIndex not found");
  return partition;
}

GlobalIndexType BoxGDA::numPartitionOwnedGlobalFieldIndices() { return _myFieldDofCount; }
GlobalIndexType BoxGDA::numPartitionOwnedGlobalFluxIndices() { return _myFluxDofCount; }
GlobalIndexType BoxGDA::numPartitionOwnedGlobalTraceIndices() { return 0; }
IndexType BoxGDA::localDofCount() { return IndexType(_myFieldDofCount + _myFluxDofCount); }

void BoxGDA::patchBasisValues(const BoxForest::Patch &patch, const OrderData &patchData, int fluxOrdinal, const double *x, vector<double> &values) const
{
  // products of Legendre polynomials in the tangential physical coordinates, on the patch's bounding box, for the
  // multi-indices of the flux variable's space at the patch's order
  const int dim = _boxTopology->getDimension();
  const int degree = patchData.fluxDegrees[fluxOrdinal];
  const std::vector<std::vector<int>> &indexSet = patchData.fluxIndexSets[fluxOrdinal];
  double legendreValues[BoxForest::MAX_DIM][32];
  int t = 0;
  for (int k=0; k<dim; k++)
  {
    if (k == patch.normal) continue;
    const double a = _boxTopology->coordinate(k, patch.lo[k]), b = _boxTopology->coordinate(k, patch.hi[k]);
    const double xi = -1 + 2 * (x[k] - a) / (b - a);
    for (int i=0; i<=degree; i++) legendreValues[t][i] = legendre(i, xi);
    t++;
  }
  values.resize(indexSet.size());
  for (size_t j=0; j<indexSet.size(); j++)
  {
    double value = 1.0;
    for (int tangential=0; tangential<t; tangential++) value *= legendreValues[tangential][indexSet[j][tangential]];
    values[j] = value;
  }
}

const BoxGDA::CellMap &BoxGDA::cellMap(GlobalIndexType cellID)
{
  auto existing = _cellMaps.find(cellID);
  if (existing != _cellMaps.end()) return existing->second;
  TEUCHOS_TEST_FOR_EXCEPTION(_fieldDofOffsetForCell[cellID] < 0, std::invalid_argument, "BoxGDA: dof map requested for an inactive cell");
  TEUCHOS_FUNC_TIME_MONITOR("BoxGDA::cellMap construction");

  const BoxForest &forest = _boxTopology->forest();
  const BoxForest::Skeleton &skeleton = _boxTopology->skeleton();
  DofOrderingPtr trialOrdering = elementType(cellID)->trialOrderPtr;
  const int numTrial = trialOrdering->totalDofs();
  const int dim = _boxTopology->getDimension();
  const int leafOrdinal = forest.leafOrdinal(cellID);

  const OrderData &cellData = orderDataForCell(cellID);
  const int myOrder = cellOrder(cellID);
  CellMap &map = _cellMaps[cellID];
  map.rows.assign(numTrial, {});
  for (size_t fieldOrdinal=0; fieldOrdinal<_fieldVars.size(); fieldOrdinal++)
  {
    const int varID = _fieldVars[fieldOrdinal]->ID();
    BasisMap &basisMap = map.basisMaps[{varID, VOLUME_INTERIOR_SIDE_INDEX}];
    basisMap.localDofIndices = trialOrdering->getDofIndices(varID);
    const GlobalIndexType offset = _fieldDofOffsetForCell[cellID] + cellData.fieldOffsets[fieldOrdinal];
    for (size_t i=0; i<basisMap.localDofIndices.size(); i++)
    {
      map.rows[basisMap.localDofIndices[i]].push_back({int(map.globalIndices.size()), 1.0});
      map.globalIndices.push_back(offset + GlobalIndexType(i));
      basisMap.globalIndices.push_back(offset + GlobalIndexType(i));
    }
  }

  FC nodes = _boxTopology->physicalCellNodesForCell(cellID, true);
  vector<GlobalIndexType> cellIDs = {cellID};
  cellData.basisCache->setPhysicalCellNodes(nodes, cellIDs, true);

  vector<double> psi, phiValues, psiValues, weights, M;
  Teuchos::LAPACK<int,double> lapack;
  for (int face=0; face<2*dim; face++)
  {
    const int side = _boxTopology->sideOrdinalForFace(face);
    const int patchOrdinal = skeleton.patchForFace[forest.faceOrdinal(leafOrdinal, face)];
    const BoxForest::Patch &patch = skeleton.patches[patchOrdinal];
    const int patchOrder = _patchOrder[patchOrdinal];
    const OrderData &patchData = orderData(patchOrder);
    BasisCachePtr sideCache = cellData.basisCache->getSideBasisCache(side);
    const FC &points = sideCache->getPhysicalCubaturePointsFC();
    const FC &weightedMeasures = sideCache->getWeightedMeasures();
    const int P = points.dimension(1);
    weights.resize(P);
    for (int q=0; q<P; q++) weights[q] = weightedMeasures(0,q);

    for (size_t fluxOrdinal=0; fluxOrdinal<_fluxVars.size(); fluxOrdinal++)
    {
      const int varID = _fluxVars[fluxOrdinal]->ID();
      BasisPtr fluxBasis = trialOrdering->getBasis(varID, side);
      auto fluxValues = sideCache->getTransformedValues(fluxBasis, OP_VALUE);
      const int n = fluxBasis->getCardinality(), m = int(patchData.fluxIndexSets[fluxOrdinal].size());
      phiValues.resize(n*P);
      psiValues.resize(m*P);
      for (int q=0; q<P; q++)
      {
        for (int i=0; i<n; i++) phiValues[i*P+q] = (*fluxValues)(0,i,q);
        patchBasisValues(patch, patchData, int(fluxOrdinal), &points(0,q,0), psi);
        for (int j=0; j<m; j++) psiValues[j*P+q] = psi[j];
      }
      // weights(i,j): patch basis function j = sum_i weights(i,j) * side basis function i, on this side (L2 projection, exact)
      BasisMap &basisMap = map.basisMaps[{varID, side}];
      M.assign(n*n, 0.0);
      basisMap.weights.assign(n*m, 0.0);
      for (int i=0; i<n; i++)
      {
        for (int j=0; j<n; j++)
          for (int q=0; q<P; q++)
            M[j*n+i] += phiValues[i*P+q] * phiValues[j*P+q] * weights[q];
        for (int j=0; j<m; j++)
          for (int q=0; q<P; q++)
            basisMap.weights[j*n+i] += phiValues[i*P+q] * psiValues[j*P+q] * weights[q];
      }
      if (m < n) basisMap.sideMass = M; // for interpretLocalBasisCoefficients()
      int info;
      lapack.POSV('U', n, m, M.data(), n, basisMap.weights.data(), n, &info);
      TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxGDA: side-to-patch projection failed");
      if (_verifiedFluxSpaces.find({myOrder, patchOrder}) == _verifiedFluxSpaces.end())
      {
        // once: the side basis must reproduce the patch basis functions exactly (it does if the flux space is what
        // orderDataForCell() took it to be)
        double maxResidual = 0, maxValue = 0;
        for (int q=0; q<P; q++)
        {
          for (int j=0; j<m; j++)
          {
            double value = 0;
            for (int i=0; i<n; i++) value += basisMap.weights[j*n+i] * phiValues[i*P+q];
            maxResidual = std::max(maxResidual, std::abs(value - psiValues[j*P+q]));
            maxValue = std::max(maxValue, std::abs(psiValues[j*P+q]));
          }
        }
        TEUCHOS_TEST_FOR_EXCEPTION(maxResidual > 1e-8 * maxValue, std::invalid_argument,
                                   "BoxGDA: the flux side basis does not span the expected polynomial space (Q_p or serendipity S_p)");
      }

      basisMap.localDofIndices = trialOrdering->getDofIndices(varID, side);
      const GlobalIndexType offset = _fluxDofOffsetForPatch[patchOrdinal] + patchData.fluxOffsets[fluxOrdinal];
      // two sides of one cell can belong to the same patch only through periodicity, which is not supported: no repeats
      const int base = int(map.globalIndices.size());
      for (int j=0; j<m; j++)
      {
        map.globalIndices.push_back(offset + j);
        basisMap.globalIndices.push_back(offset + j);
      }
      for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
          if (std::abs(basisMap.weights[j*n+i]) > 1e-14) map.rows[basisMap.localDofIndices[i]].push_back({base + j, basisMap.weights[j*n+i]});
    }
    // (verified for every flux variable on this side)
  }
  for (int face=0; face<2*dim; face++)
  {
    const int patchOrdinal = skeleton.patchForFace[forest.faceOrdinal(leafOrdinal, face)];
    _verifiedFluxSpaces.insert({myOrder, _patchOrder[patchOrdinal]});
  }
  return map;
}

void BoxGDA::interpretLocalData(GlobalIndexType cellID, const FC &localData, FC &globalData,
                                Camellia::FieldContainer<GlobalIndexType> &globalDofIndices)
{
  TEUCHOS_FUNC_TIME_MONITOR("BoxGDA::interpretLocalData");
  // as GDAMinimumRule: localData may be a vector, trial x trial, trial x other, or other x trial
  const CellMap &map = cellMap(cellID);
  const int numTrial = int(map.rows.size()), numGlobal = int(map.globalIndices.size());
  globalDofIndices.resize(numGlobal);
  for (int i=0; i<numGlobal; i++) globalDofIndices(i) = map.globalIndices[i];

  if (localData.rank() == 1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(localData.dimension(0) != numTrial, std::invalid_argument, "localData must match the number of trial dofs on the cell");
    globalData.resize(numGlobal);
    globalData.initialize(0.0);
    for (int i=0; i<numTrial; i++)
      for (const auto &[position, weight] : map.rows[i])
        globalData(position) += weight * localData(i);
    return;
  }
  const bool trialRows = (localData.dimension(0) == numTrial), trialColumns = (localData.dimension(1) == numTrial);
  TEUCHOS_TEST_FOR_EXCEPTION(!trialRows && !trialColumns, std::invalid_argument, "One of the dimensions of localData must match the number of trial dofs on the cell");
  if (trialRows && trialColumns)
  {
    globalData.resize(numGlobal, numGlobal);
    globalData.initialize(0.0);
    for (int i=0; i<numTrial; i++)
    {
      for (int j=0; j<numTrial; j++)
      {
        const double value = localData(i,j);
        if (value == 0.0) continue;
        for (const auto &[positionI, weightI] : map.rows[i])
          for (const auto &[positionJ, weightJ] : map.rows[j])
            globalData(positionI, positionJ) += weightI * value * weightJ;
      }
    }
  }
  else if (trialRows)
  {
    const int other = localData.dimension(1);
    globalData.resize(numGlobal, other);
    globalData.initialize(0.0);
    for (int i=0; i<numTrial; i++)
      for (const auto &[position, weight] : map.rows[i])
        for (int j=0; j<other; j++)
          globalData(position, j) += weight * localData(i,j);
  }
  else
  {
    const int other = localData.dimension(0);
    globalData.resize(other, numGlobal);
    globalData.initialize(0.0);
    for (int j=0; j<numTrial; j++)
      for (const auto &[position, weight] : map.rows[j])
        for (int i=0; i<other; i++)
          globalData(i, position) += localData(i,j) * weight;
  }
}

void BoxGDA::interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const FC &basisCoefficients,
                                             FC &globalCoefficients, Camellia::FieldContainer<GlobalIndexType> &globalDofIndices)
{
  TEUCHOS_FUNC_TIME_MONITOR("BoxGDA::interpretLocalBasisCoefficients");
  const CellMap &map = cellMap(cellID);
  auto entry = map.basisMaps.find({varID, sideOrdinal});
  TEUCHOS_TEST_FOR_EXCEPTION(entry == map.basisMaps.end(), std::invalid_argument, "BoxGDA: no basis for the variable on the specified side");
  const BasisMap &basisMap = entry->second;
  const int n = int(basisMap.localDofIndices.size()), m = int(basisMap.globalIndices.size());
  TEUCHOS_TEST_FOR_EXCEPTION(int(basisCoefficients.size()) != n, std::invalid_argument, "basisCoefficients should be sized to match the basis cardinality");
  globalCoefficients.resize(m);
  globalDofIndices.resize(m);
  for (int j=0; j<m; j++) globalDofIndices(j) = basisMap.globalIndices[j];
  if (basisMap.weights.empty())
  {
    for (int j=0; j<m; j++) globalCoefficients(j) = basisCoefficients[j];
    return;
  }
  // the patch polynomial restricted to this side is weights * g; the restriction is injective
  Teuchos::LAPACK<int,double> lapack;
  int info;
  if (m == n)
  {
    // solve weights * g = c
    vector<double> W = basisMap.weights, g(n);
    for (int i=0; i<n; i++) g[i] = basisCoefficients[i];
    vector<int> pivots(n);
    lapack.GESV(n, 1, W.data(), n, pivots.data(), g.data(), n, &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxGDA: could not interpret side coefficients");
    for (int j=0; j<m; j++) globalCoefficients(j) = g[j];
    return;
  }
  // the side basis is richer than the patch's (a higher-order cell on a lower-order patch): the L2 projection of the
  // side function onto the patch space, (W^T M W) g = W^T M c
  const vector<double> &W = basisMap.weights, &M = basisMap.sideMass;
  vector<double> Mc(n, 0.0), MW(n*m, 0.0), A(m*m, 0.0), g(m, 0.0);
  for (int i=0; i<n; i++)
    for (int k=0; k<n; k++)
    {
      Mc[i] += M[k*n+i] * basisCoefficients[k];
      for (int j=0; j<m; j++) MW[j*n+i] += M[k*n+i] * W[j*n+k];
    }
  for (int j=0; j<m; j++)
  {
    for (int i=0; i<n; i++) g[j] += W[j*n+i] * Mc[i];
    for (int l=0; l<m; l++)
      for (int i=0; i<n; i++) A[l*m+j] += W[j*n+i] * MW[l*n+i];
  }
  lapack.POSV('U', m, 1, A.data(), m, g.data(), m, &info);
  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxGDA: could not interpret side coefficients");
  for (int j=0; j<m; j++) globalCoefficients(j) = g[j];
}

void BoxGDA::interpretGlobalCoefficients(GlobalIndexType cellID, FC &localCoefficients, const TVectorPtr<double> globalCoefficients)
{
  TEUCHOS_FUNC_TIME_MONITOR("BoxGDA::interpretGlobalCoefficients");
  using LO = TMap::local_ordinal_type;
  using GO = TMap::global_ordinal_type;
  const CellMap &map = cellMap(cellID);
  const int numTrial = int(map.rows.size());
  auto view = globalCoefficients->getLocalViewHost(Tpetra::Access::ReadOnly);
  const int numVectors = int(globalCoefficients->getNumVectors());
  auto tpetraMap = globalCoefficients->getMap();
  const LO invalid = Teuchos::OrdinalTraits<LO>::invalid();
  vector<LO> localIDs(map.globalIndices.size());
  for (size_t i=0; i<localIDs.size(); i++) localIDs[i] = tpetraMap->getLocalElement(GO(map.globalIndices[i]));
  if (numVectors == 1)
  {
    if (int(localCoefficients.size()) != numTrial) localCoefficients.resize(numTrial);
  }
  for (int vectorOrdinal=0; vectorOrdinal<numVectors; vectorOrdinal++)
  {
    for (int i=0; i<numTrial; i++)
    {
      double value = 0;
      for (const auto &[position, weight] : map.rows[i])
      {
        if (localIDs[position] != invalid) value += weight * view(localIDs[position], vectorOrdinal);
      }
      if (numVectors == 1) localCoefficients[i] = value;
      else localCoefficients(vectorOrdinal, i) = value;
    }
  }
}

set<GlobalIndexType> BoxGDA::globalDofIndicesForCell(GlobalIndexType cellID)
{
  const CellMap &map = cellMap(cellID);
  return set<GlobalIndexType>(map.globalIndices.begin(), map.globalIndices.end());
}

set<GlobalIndexType> BoxGDA::globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned dim, unsigned subcellOrdinal,
                                                             bool upgradeToContinuousBasis, bool tensorTopologyMeansSpaceTime)
{
  // field dofs belong to the cell interior; flux dofs to sides
  const CellMap &map = cellMap(cellID);
  const unsigned cellDim = _boxTopology->getDimension();
  int sideOrdinal;
  if (dim == cellDim) sideOrdinal = VOLUME_INTERIOR_SIDE_INDEX;
  else if (dim == cellDim - 1) sideOrdinal = int(subcellOrdinal);
  else return set<GlobalIndexType>();
  auto entry = map.basisMaps.find({varID, sideOrdinal});
  if (entry == map.basisMaps.end()) return set<GlobalIndexType>();
  return set<GlobalIndexType>(entry->second.globalIndices.begin(), entry->second.globalIndices.end());
}

vector<GlobalIndexType> BoxGDA::globalDofIndicesForFieldVariable(GlobalIndexType cellID, int varID)
{
  const CellMap &map = cellMap(cellID);
  auto entry = map.basisMaps.find({varID, VOLUME_INTERIOR_SIDE_INDEX});
  TEUCHOS_TEST_FOR_EXCEPTION(entry == map.basisMaps.end(), std::invalid_argument, "BoxGDA: field variable not found");
  return entry->second.globalIndices;
}

bool BoxGDA::projectParentCoefficientsOntoChildren(GlobalIndexType parentCellID, const FC &parentCoefficients,
                                                   const vector<GlobalIndexType> &childIDs, vector<FC> &childCoefficients)
{
  TEUCHOS_FUNC_TIME_MONITOR("BoxGDA::projectParentCoefficientsOntoChildren");
  // p-refinement ("child" = the cell itself, at a new order): Solution's generic projection handles it
  if ((childIDs.size() == 1) && (childIDs[0] == parentCellID)) return false;
  const OrderData &data = orderDataForCell(parentCellID);
  for (GlobalIndexType childID : childIDs)
  {
    // children take the parent's order (didHRefine()); otherwise let Solution's generic projection handle it
    if ((childID != GlobalIndexType(-1)) && (cellOrder(childID) != cellOrder(parentCellID))) return false;
  }
  DofOrderingPtr trialOrdering = data.elemType->trialOrderPtr; // the children's ordering too
  const int numTrial = trialOrdering->totalDofs();
  TEUCHOS_TEST_FOR_EXCEPTION(int(parentCoefficients.size()) != numTrial, std::invalid_argument,
                             "BoxGDA::projectParentCoefficientsOntoChildren: cell " << parentCellID << " has " << parentCoefficients.size()
                             << " coefficients, but its element type has " << numTrial << " (a solution not projected when the cell was p-refined?)");
  const int dim = _boxTopology->getDimension();
  const BoxForest &forest = _boxTopology->forest();
  const BoxMeshTopology::Extents parentExtents = _boxTopology->cellExtents(parentCellID);
  const BoxForest::Box &parentBox = forest.node(parentCellID).box;

  // solves (phi, phi) c = (phi, values): the coefficients c of the L2 projection onto the basis with values phi (n x P)
  Teuchos::LAPACK<int,double> lapack;
  auto project = [&](const vector<double> &phi, const vector<double> &values, const vector<double> &weights, int n, vector<double> &c)
  {
    const int P = int(weights.size());
    vector<double> M(n*n, 0.0);
    c.assign(n, 0.0);
    for (int i=0; i<n; i++)
    {
      for (int j=0; j<n; j++)
        for (int q=0; q<P; q++)
          M[j*n+i] += phi[i*P+q] * phi[j*P+q] * weights[q];
      for (int q=0; q<P; q++) c[i] += phi[i*P+q] * values[q] * weights[q];
    }
    int info;
    lapack.POSV('U', n, 1, M.data(), n, c.data(), n, &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxGDA: projection onto child failed");
  };

  // The parent's flux on each of its sides, as coefficients of the tensor-Legendre basis on that side's box.
  // (This avoids evaluating the parent's side basis at the children's points.)
  FC parentNodes = _boxTopology->physicalCellNodesForCell(parentCellID, true);
  const BasisCachePtr &parentBasisCache = data.parentBasisCache, &basisCache = data.basisCache;
  parentBasisCache->setPhysicalCellNodes(parentNodes, {parentCellID}, true);
  const int sideCount = _boxTopology->cellTopology()->getSideCount();
  vector<BoxForest::Patch> sideBox(sideCount);
  vector<vector<vector<double>>> parentSideLegendre(sideCount, vector<vector<double>>(_fluxVars.size()));
  vector<double> psi, phiValues, psiValues, values, weights, c;
  for (int side=0; side<sideCount; side++)
  {
    const int face = _boxTopology->faceForSideOrdinal(side), k = face / 2;
    BoxForest::Patch &patch = sideBox[side];
    patch.normal = k;
    for (int j=0; j<dim; j++) { patch.lo[j] = parentBox.lo[j]; patch.hi[j] = parentBox.hi(j); }
    BasisCachePtr sideCache = parentBasisCache->getSideBasisCache(side);
    const FC &points = sideCache->getPhysicalCubaturePointsFC();
    const FC &weightedMeasures = sideCache->getWeightedMeasures();
    const int P = points.dimension(1);
    weights.resize(P);
    for (int q=0; q<P; q++) weights[q] = weightedMeasures(0,q);
    for (size_t fluxOrdinal=0; fluxOrdinal<_fluxVars.size(); fluxOrdinal++)
    {
      const int varID = _fluxVars[fluxOrdinal]->ID();
      BasisPtr basis = trialOrdering->getBasis(varID, side);
      auto basisValues = sideCache->getTransformedValues(basis, OP_VALUE);
      const vector<int> &dofIndices = trialOrdering->getDofIndices(varID, side);
      const int n = basis->getCardinality();
      psiValues.resize(n*P);
      values.assign(P, 0.0);
      for (int q=0; q<P; q++)
      {
        for (int i=0; i<n; i++) values[q] += parentCoefficients[dofIndices[i]] * (*basisValues)(0,i,q);
        patchBasisValues(patch, data, int(fluxOrdinal), &points(0,q,0), psi);
        for (int j=0; j<n; j++) psiValues[j*P+q] = psi[j];
      }
      project(psiValues, values, weights, n, parentSideLegendre[side][fluxOrdinal]);
    }
  }

  childCoefficients.assign(childIDs.size(), FC());
  for (size_t childOrdinal=0; childOrdinal<childIDs.size(); childOrdinal++)
  {
    const GlobalIndexType childID = childIDs[childOrdinal];
    if (childID == -1) continue;
    FC &coefficients = childCoefficients[childOrdinal];
    coefficients.resize(numTrial);
    coefficients.initialize(0.0);
    const BoxForest::Box &childBox = forest.node(childID).box;

    FC childNodes = _boxTopology->physicalCellNodesForCell(childID, true);
    basisCache->setPhysicalCellNodes(childNodes, {childID}, true);

    // fields: evaluate the parent's basis at the child's cubature points, mapped to the parent's reference cell
    {
      const FC &points = basisCache->getPhysicalCubaturePointsFC();
      const FC &weightedMeasures = basisCache->getWeightedMeasures();
      const int P = points.dimension(1);
      FC parentRefPoints(P, dim);
      for (int q=0; q<P; q++)
        for (int k=0; k<dim; k++)
          parentRefPoints(q,k) = -1 + 2 * (points(0,q,k) - parentExtents[k].first) / (parentExtents[k].second - parentExtents[k].first);
      weights.resize(P);
      for (int q=0; q<P; q++) weights[q] = weightedMeasures(0,q);
      for (VarPtr field : _fieldVars)
      {
        BasisPtr basis = trialOrdering->getBasis(field->ID());
        const vector<int> &dofIndices = trialOrdering->getDofIndices(field->ID());
        const int n = basis->getCardinality();
        auto childValues = basisCache->getTransformedValues(basis, OP_VALUE);
        phiValues.resize(n*P);
        for (int i=0; i<n; i++) for (int q=0; q<P; q++) phiValues[i*P+q] = (*childValues)(0,i,q);
        // the reference points of parentBasisCache are replaced; its physical nodes are still the parent's
        parentBasisCache->setRefCellPointsFC(parentRefPoints);
        auto parentValues = parentBasisCache->getTransformedValues(basis, OP_VALUE);
        values.assign(P, 0.0);
        for (int q=0; q<P; q++)
          for (int i=0; i<n; i++)
            values[q] += parentCoefficients[dofIndices[i]] * (*parentValues)(0,i,q);
        project(phiValues, values, weights, n, c);
        for (int i=0; i<n; i++) coefficients[dofIndices[i]] = c[i];
      }
    }

    // fluxes, on child sides that lie on the same side of the parent
    for (int side=0; side<sideCount; side++)
    {
      const int face = _boxTopology->faceForSideOrdinal(side), k = face / 2;
      const bool onParentSide = (face % 2 == 1) ? (childBox.hi(k) == parentBox.hi(k)) : (childBox.lo[k] == parentBox.lo[k]);
      if (!onParentSide) continue;
      BasisCachePtr sideCache = basisCache->getSideBasisCache(side);
      const FC &points = sideCache->getPhysicalCubaturePointsFC();
      const FC &weightedMeasures = sideCache->getWeightedMeasures();
      const int P = points.dimension(1);
      weights.resize(P);
      for (int q=0; q<P; q++) weights[q] = weightedMeasures(0,q);
      for (size_t fluxOrdinal=0; fluxOrdinal<_fluxVars.size(); fluxOrdinal++)
      {
        const int varID = _fluxVars[fluxOrdinal]->ID();
        BasisPtr basis = trialOrdering->getBasis(varID, side);
        const vector<int> &dofIndices = trialOrdering->getDofIndices(varID, side);
        const int n = basis->getCardinality();
        auto childValues = sideCache->getTransformedValues(basis, OP_VALUE);
        phiValues.resize(n*P);
        values.assign(P, 0.0);
        for (int q=0; q<P; q++)
        {
          for (int i=0; i<n; i++) phiValues[i*P+q] = (*childValues)(0,i,q);
          patchBasisValues(sideBox[side], data, int(fluxOrdinal), &points(0,q,0), psi);
          for (int j=0; j<n; j++) values[q] += parentSideLegendre[side][fluxOrdinal][j] * psi[j];
        }
        project(phiValues, values, weights, n, c);
        for (int i=0; i<n; i++) coefficients[dofIndices[i]] = c[i];
      }
    }
  }
  return true;
}

// ---------------------------------------------------------------- BoxConfigurationMeshUpdater

BoxConfigurationMeshUpdater::BoxConfigurationMeshUpdater(MeshPtr phaseSpaceMesh, MeshPtr configurationMesh)
{
  _phaseSpaceMesh = Teuchos::rcp(phaseSpaceMesh.get(), false);
  _configurationMesh = Teuchos::rcp(configurationMesh.get(), false);
  _boxTopology = dynamic_cast<BoxMeshTopology*>(phaseSpaceMesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(_boxTopology == NULL, std::invalid_argument, "phaseSpaceMesh must use a BoxMeshTopology");
  TEUCHOS_TEST_FOR_EXCEPTION(_boxTopology->configurationTopology() == Teuchos::null, std::invalid_argument,
                             "the BoxMeshTopology has no configuration topology set");
}

void BoxConfigurationMeshUpdater::registerForUpdates(MeshPtr phaseSpaceMesh, MeshPtr configurationMesh)
{
  Teuchos::RCP<RefinementObserver> updater = Teuchos::rcp(new BoxConfigurationMeshUpdater(phaseSpaceMesh, configurationMesh));
  phaseSpaceMesh->registerObserver(updater);
}

void BoxConfigurationMeshUpdater::didHRefineInDirections(const std::map<GlobalIndexType,unsigned> &cellDirectionMasks, bool didRepartitionAndRebuild)
{
  const int configurationDimensionCount = _boxTopology->configurationDimensionCount();
  const unsigned configurationMask = (1u << configurationDimensionCount) - 1;
  MeshTopologyViewPtr configurationTopology = _configurationMesh->getTopology();
  // The configuration mesh is refined regularly, so its cells are at one level in every direction.  Under each
  // refined phase-space cell, it must reach the finest level the children have in any configuration direction.
  bool refinedSome = false;
  while (true)
  {
    std::set<GlobalIndexType> configurationCellsToRefine;
    for (const auto &entry : cellDirectionMasks)
    {
      const unsigned mask = entry.second & configurationMask;
      if (mask == 0) continue; // velocity directions only
      const BoxForest::Box &box = _boxTopology->forest().node(entry.first).box;
      int requiredLevel = 0;
      for (int k=0; k<configurationDimensionCount; k++)
      {
        requiredLevel = std::max(requiredLevel, int(box.level[k]) + int((mask >> k) & 1u));
      }
      for (IndexType configurationCellID : _boxTopology->phaseSpaceActiveConfigurationCells(entry.first))
      {
        if (configurationTopology->getCell(configurationCellID)->level() < requiredLevel) configurationCellsToRefine.insert(configurationCellID);
      }
    }
    if (configurationCellsToRefine.empty()) break;
    _configurationMesh->hRefine(configurationCellsToRefine, false);
    refinedSome = true;
  }
  if (refinedSome)
  {
    _configurationMesh->enforceOneIrregularity(false);
    _configurationMesh->repartitionAndRebuild();
  }
}

// ---------------------------------------------------------------- matching two box meshes at a boundary

void BoxGDA::hRefineToMatchAtBoundary(std::map<IndexType,IndexType> &cellIDMap, std::map<int,int> &sideOrdinalMap,
                                      MeshPtr meshFrom, MeshPtr meshTo, int direction)
{
  BoxMeshTopology* fromTopology = dynamic_cast<BoxMeshTopology*>(meshFrom->getTopology().get());
  BoxMeshTopology* toTopology = dynamic_cast<BoxMeshTopology*>(meshTo->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION((fromTopology == NULL) || (toTopology == NULL), std::invalid_argument, "both meshes must use a BoxMeshTopology");
  const int dim = fromTopology->getDimension();
  TEUCHOS_TEST_FOR_EXCEPTION(int(toTopology->getDimension()) != dim, std::invalid_argument, "mesh dimensions must match");

  // meshTo's upper bound in the direction
  double toUpperBound = 0;
  bool first = true;
  for (IndexType root : toTopology->getRootCellIndicesLocal())
  {
    const double upper = toTopology->cellExtents(root)[direction].second;
    toUpperBound = first ? upper : std::max(toUpperBound, upper);
    first = false;
  }

  const unsigned lowerSide = unsigned(fromTopology->sideOrdinalForFace(2 * direction));
  auto counterpart = [&](IndexType fromCellID)
  {
    std::vector<double> centroid = fromTopology->getCellCentroid(fromCellID);
    FC point(1, dim);
    for (int k=0; k<dim; k++) point(0,k) = (k == direction) ? toUpperBound : centroid[k];
    const IndexType toCellID = toTopology->cellIDsForPoints(point)[0];
    TEUCHOS_TEST_FOR_EXCEPTION(toCellID == -1, std::invalid_argument, "no matching cell found across the boundary; do the meshes share root cells?");
    return toCellID;
  };

  while (true)
  {
    std::map<GlobalIndexType,unsigned> fromMasks, toMasks;
    for (IndexType fromCellID : fromTopology->getLocallyKnownActiveCellIndices())
    {
      if (!fromTopology->sideIsOnBoundary(fromCellID, lowerSide)) continue;
      const IndexType toCellID = counterpart(fromCellID);
      const BoxForest::Box &fromBox = fromTopology->forest().node(fromCellID).box;
      const BoxForest::Box &toBox = toTopology->forest().node(toCellID).box;
      for (int j=0; j<dim; j++)
      {
        if (j == direction) continue;
        if (fromBox.level[j] < toBox.level[j]) fromMasks[fromCellID] |= (1u << j);
        else if (toBox.level[j] < fromBox.level[j]) toMasks[toCellID] |= (1u << j);
      }
    }
    if (fromMasks.empty() && toMasks.empty()) break;
    meshFrom->hRefine(fromMasks);
    meshTo->hRefine(toMasks);
  }

  cellIDMap.clear();
  for (IndexType fromCellID : fromTopology->getLocallyKnownActiveCellIndices())
  {
    if (fromTopology->sideIsOnBoundary(fromCellID, lowerSide)) cellIDMap[fromCellID] = counterpart(fromCellID);
  }
  // orders: meshFrom's boundary cells are raised to their counterparts' orders (with the halo that keeps neighbours
  // within one order); meshTo is left as it is
  {
    std::map<GlobalIndexType,int> targetOrder;
    for (const auto &[fromCellID, toCellID] : cellIDMap)
    {
      const int toOrder = meshTo->globalDofAssignment()->getH1Order(toCellID)[0];
      if (toOrder > meshFrom->globalDofAssignment()->getH1Order(fromCellID)[0]) targetOrder[fromCellID] = toOrder;
    }
    HPRefinements raise;
    raise.pIncrements = orderIncrementsWithHalo(meshFrom, targetOrder);
    if (!raise.pIncrements.empty()) hpRefine(meshFrom, raise);
  }
  sideOrdinalMap.clear();
  sideOrdinalMap[int(lowerSide)] = toTopology->sideOrdinalForFace(2 * direction + 1);
}

// ---------------------------------------------------------------- multigrid

std::vector<MeshPtr> BoxGDA::meshesForMultigrid(MeshPtr fineMesh, int kCoarse, int delta_k)
{
  TEUCHOS_FUNC_TIME_MONITOR("BoxGDA::meshesForMultigrid");
  BoxMeshTopology* fineTopology = dynamic_cast<BoxMeshTopology*>(fineMesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(fineTopology == NULL, std::invalid_argument, "fineMesh must use a BoxMeshTopology");
  if (delta_k == -1) delta_k = fineMesh->globalDofAssignment()->getTestOrderEnrichment();
  const int dim = fineTopology->getDimension();
  const int depth = fineTopology->refinementDepth();
  BFPtr bf = fineMesh->bilinearForm();
  map<int,int> trialOrderEnhancements = fineMesh->getDofOrderingFactory().getTrialOrderEnhancements();

  // the coarse levels have one order, no higher than the lowest on the fine mesh (so each coarse space is in the next)
  int fineMinOrder = std::numeric_limits<int>::max();
  for (IndexType cellID : fineTopology->getLocallyKnownActiveCellIndices())
  {
    fineMinOrder = std::min(fineMinOrder, fineMesh->globalDofAssignment()->getH1Order(cellID)[0]);
  }
  const int coarseH1Order = std::min(kCoarse + 1, fineMinOrder);
  std::vector<MeshPtr> meshesCoarseToFine;
  for (int level=0; level<=depth; level++)
  {
    Teuchos::RCP<BoxMeshTopology> levelTopology = fineTopology->copyTruncatedAtDepth(level);
    // a coarse cell goes to the rank owning the fine cell at its lower corner
    map<GlobalIndexType,GlobalIndexType> cellIDMap;
    for (IndexType cellID : levelTopology->getLocallyKnownActiveCellIndices())
    {
      BoxMeshTopology::Extents extents = levelTopology->cellExtents(cellID);
      FC point(1, dim);
      for (int k=0; k<dim; k++) point(0,k) = extents[k].first + 1e-9 * (extents[k].second - extents[k].first);
      cellIDMap[cellID] = fineTopology->cellIDsForPoints(point)[0];
    }
    MeshPartitionPolicyPtr partitionPolicy = Teuchos::rcp(new InducedMeshPartitionPolicy(fineMesh, cellIDMap));
    meshesCoarseToFine.push_back(BoxGDA::mesh(levelTopology, bf, coarseH1Order, delta_k, partitionPolicy, Teuchos::null,
                                              trialOrderEnhancements, map<int,int>()));
  }
  meshesCoarseToFine.push_back(fineMesh);
  return meshesCoarseToFine;
}

TMatrixPtr<double> BoxGDA::prolongationOperator(MeshPtr coarseMesh, TMapPtr coarseMap, TMapPtr fineMap)
{
  TEUCHOS_FUNC_TIME_MONITOR("BoxGDA::prolongationOperator");
  using GO = TMap::global_ordinal_type;
  BoxGDA* coarseGDA = dynamic_cast<BoxGDA*>(coarseMesh->globalDofAssignment().get());
  TEUCHOS_TEST_FOR_EXCEPTION(coarseGDA == NULL, std::invalid_argument, "the coarse mesh must also have a BoxGDA");
  BoxMeshTopology* coarseTopology = coarseGDA->_boxTopology;
  TEUCHOS_TEST_FOR_EXCEPTION((_fieldVars.size() != coarseGDA->_fieldVars.size()) || (_fluxVars.size() != coarseGDA->_fluxVars.size()),
                             std::invalid_argument, "coarse and fine meshes must have the same variables");

  const int dim = _boxTopology->getDimension();
  const BoxForest &fineForest = _boxTopology->forest();
  const BoxForest &coarseForest = coarseTopology->forest();
  const BoxForest::Skeleton &fineSkeleton = _boxTopology->skeleton();
  const BoxForest::Skeleton &coarseSkeleton = coarseTopology->skeleton();
  const int rank = _partitionPolicy->Comm()->getRank();

  const GlobalIndexType fineDofCount = globalDofCount(), coarseDofCount = coarseGDA->globalDofCount();
  const size_t numOwnedRows = fineMap->getLocalNumElements();
  std::vector<std::vector<GO>> rowIndices(numOwnedRows);
  std::vector<std::vector<double>> rowValues(numOwnedRows);
  auto addEntry = [&](GlobalIndexType fineDof, GlobalIndexType coarseDof, double value)
  {
    if (std::abs(value) < 1e-13) return;
    const auto localRow = fineMap->getLocalElement(GO(fineDof));
    TEUCHOS_TEST_FOR_EXCEPTION(localRow == Teuchos::OrdinalTraits<TMap::local_ordinal_type>::invalid(), std::logic_error,
                               "BoxGDA::prolongationOperator: an owned fine dof is not in the fine map");
    rowIndices[localRow].push_back(GO(coarseDof));
    rowValues[localRow].push_back(value);
  };

  auto coarseCellContaining = [&](GlobalIndexType fineCellID)
  {
    std::vector<double> centroid = _boxTopology->getCellCentroid(fineCellID);
    FC point(1, dim);
    for (int k=0; k<dim; k++) point(0,k) = centroid[k];
    const IndexType coarseCellID = coarseTopology->cellIDsForPoints(point)[0];
    TEUCHOS_TEST_FOR_EXCEPTION(coarseCellID == -1, std::invalid_argument, "no coarse cell found for fine cell");
    return GlobalIndexType(coarseCellID);
  };

  // ---- fields: the coarse cell's basis functions, expressed in the fine cell's basis (L2 projection; exact)
  std::map<int, BasisCachePtr> coarseBasisCaches; // by coarse order (their reference points are replaced below)
  Teuchos::LAPACK<int,double> lapack;
  std::vector<double> phi, psi, weights, M, C;
  for (GlobalIndexType fineCellID : _partitions[rank])
  {
    const GlobalIndexType coarseCellID = coarseCellContaining(fineCellID);
    DofOrderingPtr fineOrdering = elementType(fineCellID)->trialOrderPtr;
    DofOrderingPtr coarseOrdering = coarseGDA->elementType(coarseCellID)->trialOrderPtr;
    const BoxMeshTopology::Extents coarseExtents = coarseTopology->cellExtents(coarseCellID);

    const OrderData &fineData = orderDataForCell(fineCellID);
    TEUCHOS_TEST_FOR_EXCEPTION(coarseGDA->cellOrder(coarseCellID) > cellOrder(fineCellID), std::invalid_argument,
                               "BoxGDA::prolongationOperator: a coarse cell has a higher order than a fine cell in it");
    const BasisCachePtr &fineBasisCache = fineData.basisCache;
    FC fineNodes = _boxTopology->physicalCellNodesForCell(fineCellID, true);
    fineBasisCache->setPhysicalCellNodes(fineNodes, {fineCellID}, false);
    const FC &points = fineBasisCache->getPhysicalCubaturePointsFC();
    const FC &weightedMeasures = fineBasisCache->getWeightedMeasures();
    const int P = points.dimension(1);
    weights.resize(P);
    for (int q=0; q<P; q++) weights[q] = weightedMeasures(0,q);
    FC coarseRefPoints(P, dim);
    for (int q=0; q<P; q++)
      for (int k=0; k<dim; k++)
        coarseRefPoints(q,k) = -1 + 2 * (points(0,q,k) - coarseExtents[k].first) / (coarseExtents[k].second - coarseExtents[k].first);
    BasisCachePtr &coarseBasisCache = coarseBasisCaches[coarseGDA->cellOrder(coarseCellID)];
    if (coarseBasisCache == Teuchos::null) coarseBasisCache = Teuchos::rcp(new BasisCache(coarseGDA->elementType(coarseCellID)));
    FC coarseNodes = coarseTopology->physicalCellNodesForCell(coarseCellID, true);
    coarseBasisCache->setPhysicalCellNodes(coarseNodes, {coarseCellID}, false);
    coarseBasisCache->setRefCellPointsFC(coarseRefPoints);

    for (size_t fieldOrdinal=0; fieldOrdinal<_fieldVars.size(); fieldOrdinal++)
    {
      const int varID = _fieldVars[fieldOrdinal]->ID();
      BasisPtr fineBasis = fineOrdering->getBasis(varID), coarseBasis = coarseOrdering->getBasis(varID);
      const int n = fineBasis->getCardinality(), m = coarseBasis->getCardinality();
      auto fineValues = fineBasisCache->getTransformedValues(fineBasis, OP_VALUE);
      auto coarseValues = coarseBasisCache->getTransformedValues(coarseBasis, OP_VALUE);
      M.assign(n*n, 0.0);
      C.assign(n*m, 0.0);
      for (int i=0; i<n; i++)
      {
        for (int j=0; j<n; j++)
          for (int q=0; q<P; q++)
            M[j*n+i] += (*fineValues)(0,i,q) * (*fineValues)(0,j,q) * weights[q];
        for (int j=0; j<m; j++)
          for (int q=0; q<P; q++)
            C[j*n+i] += (*fineValues)(0,i,q) * (*coarseValues)(0,j,q) * weights[q];
      }
      int info;
      lapack.POSV('U', n, m, M.data(), n, C.data(), n, &info);
      TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxGDA::prolongationOperator: field projection failed");
      const GlobalIndexType fineOffset = _fieldDofOffsetForCell[fineCellID] + fineData.fieldOffsets[fieldOrdinal];
      const GlobalIndexType coarseOffset = coarseGDA->_fieldDofOffsetForCell[coarseCellID]
                                         + coarseGDA->orderData(coarseGDA->cellOrder(coarseCellID)).fieldOffsets[fieldOrdinal];
      for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
          addEntry(fineOffset + i, coarseOffset + j, C[j*n+i]);
    }
  }

  // ---- fluxes: the coarse patch's polynomial restricted to the fine patch, where the fine patch is on the coarse skeleton
  const int numPartitions = int(_partitions.size());
  std::vector<int> partitionForCell(_boxTopology->cellCount(), -1);
  for (int partition=0; partition<numPartitions; partition++)
    for (GlobalIndexType cellID : _partitions[partition]) partitionForCell[cellID] = partition;

  for (int patchOrdinal=0; patchOrdinal<int(fineSkeleton.patches.size()); patchOrdinal++)
  {
    const BoxForest::Patch &finePatch = fineSkeleton.patches[patchOrdinal];
    const BoxForest::Face &face = finePatch.minusFaces.empty() ? finePatch.plusFaces[0] : finePatch.minusFaces[0];
    if (partitionForCell[face.cell] != rank) continue; // not mine
    const int normal = finePatch.normal;
    const GlobalIndexType coarseCellID = coarseCellContaining(face.cell);
    // is the fine face on the coarse cell's boundary?  Compare physical coordinates of the two cells' faces.
    const bool upper = (face.side % 2 == 1);
    const BoxMeshTopology::Extents fineExtents = _boxTopology->cellExtents(face.cell);
    const BoxMeshTopology::Extents coarseExtents = coarseTopology->cellExtents(coarseCellID);
    const double fineCoordinate = upper ? fineExtents[normal].second : fineExtents[normal].first;
    const double coarseCoordinate = upper ? coarseExtents[normal].second : coarseExtents[normal].first;
    const double tolerance = 1e-12 * std::max(1.0, std::abs(coarseExtents[normal].second - coarseExtents[normal].first));
    if (std::abs(fineCoordinate - coarseCoordinate) > tolerance) continue; // inside a coarse cell: no coarse flux here

    const int coarsePatchOrdinal = coarseSkeleton.patchForFace[coarseForest.faceOrdinal(coarseForest.leafOrdinal(coarseCellID), face.side)];
    const BoxForest::Patch &coarsePatch = coarseSkeleton.patches[coarsePatchOrdinal];
    const OrderData &finePatchData = orderData(_patchOrder[patchOrdinal]);
    const OrderData &coarsePatchData = coarseGDA->orderData(coarseGDA->_patchOrder[coarsePatchOrdinal]);

    for (size_t fluxOrdinal=0; fluxOrdinal<_fluxVars.size(); fluxOrdinal++)
    {
      const int fineDegree = finePatchData.fluxDegrees[fluxOrdinal], coarseDegree = coarsePatchData.fluxDegrees[fluxOrdinal];
      TEUCHOS_TEST_FOR_EXCEPTION(_patchOrder[patchOrdinal] < coarseGDA->_patchOrder[coarsePatchOrdinal], std::invalid_argument,
                                 "BoxGDA::prolongationOperator: a coarse patch has a higher order than a fine patch on it");
      // one 1D restriction matrix per tangential direction: fine coefficients from coarse ones
      std::vector<std::vector<double>> R;
      for (int k=0; k<dim; k++)
      {
        if (k == normal) continue;
        const double coarseLo = coarseTopology->coordinate(k, coarsePatch.lo[k]), coarseHi = coarseTopology->coordinate(k, coarsePatch.hi[k]);
        const double fineLo = _boxTopology->coordinate(k, finePatch.lo[k]), fineHi = _boxTopology->coordinate(k, finePatch.hi[k]);
        const double a = -1 + 2 * (fineLo - coarseLo) / (coarseHi - coarseLo), b = -1 + 2 * (fineHi - coarseLo) / (coarseHi - coarseLo);
        R.push_back(BoxForest::legendreRestriction(fineDegree, a, b)); // (fineDegree+1)^2, row-major
      }
      const int tangentialCount = dim - 1, nf = fineDegree + 1;
      const std::vector<std::vector<int>> &fineIndexSet = finePatchData.fluxIndexSets[fluxOrdinal];
      const std::vector<std::vector<int>> &coarseIndexSet = coarsePatchData.fluxIndexSets[fluxOrdinal];
      const GlobalIndexType fineOffset = _fluxDofOffsetForPatch[patchOrdinal] + finePatchData.fluxOffsets[fluxOrdinal];
      const GlobalIndexType coarseOffset = coarseGDA->_fluxDofOffsetForPatch[coarsePatchOrdinal] + coarsePatchData.fluxOffsets[fluxOrdinal];
      // The coarse function, restricted, has a unique expansion in the full tensor Legendre basis of the fine box, with
      // coefficients that are products of the 1D restrictions; it lies in the fine space, so those are its coefficients there.
      for (size_t i=0; i<fineIndexSet.size(); i++)
      {
        for (size_t j=0; j<coarseIndexSet.size(); j++)
        {
          double value = 1.0;
          for (int t=0; t<tangentialCount; t++) value *= R[t][fineIndexSet[i][t] * nf + coarseIndexSet[j][t]];
          addEntry(fineOffset + GlobalIndexType(i), coarseOffset + GlobalIndexType(j), value);
        }
      }
    }
  }

  // ---- rows beyond the dof count (Lagrange constraints): one to one, as GMGOperator does
  auto fineGIDs = fineMap->getMyGlobalIndices();
  for (size_t localRow=0; localRow<numOwnedRows; localRow++)
  {
    const GO globalRow = fineGIDs[localRow];
    if (globalRow >= GO(fineDofCount))
    {
      rowIndices[localRow].push_back(GO(coarseDofCount) + (globalRow - GO(fineDofCount)));
      rowValues[localRow].push_back(1.0);
    }
  }

  Teuchos::Array<size_t> numEntriesPerRow(numOwnedRows);
  for (size_t localRow=0; localRow<numOwnedRows; localRow++) numEntriesPerRow[localRow] = rowIndices[localRow].size();
  TMatrixPtr<double> P = Teuchos::rcp(new TMatrix<double>(fineMap, numEntriesPerRow()));
  for (size_t localRow=0; localRow<numOwnedRows; localRow++)
  {
    if (rowIndices[localRow].empty()) continue;
    const Teuchos::ArrayView<const GO> indices(rowIndices[localRow].data(), rowIndices[localRow].size());
    const Teuchos::ArrayView<const double> values(rowValues[localRow].data(), rowValues[localRow].size());
    P->insertGlobalValues(fineGIDs[localRow], indices, values);
  }
  P->fillComplete(coarseMap, fineMap);
  return P;
}

// ---------------------------------------------------------------- choosing refinement directions

namespace
{
  void gaussLegendre(int n, std::vector<double> &x, std::vector<double> &w)
  {
    x.resize(n);
    w.resize(n);
    for (int q=0; q<n; q++)
    {
      double xi = std::cos(M_PI * (q + 0.75) / (n + 0.5));
      double dP = 1;
      for (int iteration=0; iteration<100; iteration++)
      {
        const double Pn = legendre(n, xi), Pnm1 = legendre(n-1, xi);
        dP = n * (xi * Pn - Pnm1) / (xi * xi - 1.0);
        const double dx = Pn / dP;
        xi -= dx;
        if (std::abs(dx) < 1e-15) break;
      }
      const double Pn = legendre(n, xi), Pnm1 = legendre(n-1, xi);
      dP = n * (xi * Pn - Pnm1) / (xi * xi - 1.0);
      x[q] = xi;
      w[q] = 2.0 / ((1.0 - xi * xi) * dP * dP);
    }
  }
}

std::map<GlobalIndexType,unsigned> BoxGDA::anisotropicRefinementMasks(MeshPtr mesh, FunctionPtr f, const std::vector<GlobalIndexType> &cellIDs,
                                                                      double threshold, unsigned allowedDirections)
{
  TEUCHOS_FUNC_TIME_MONITOR("BoxGDA::anisotropicRefinementMasks");
  BoxMeshTopology* topology = dynamic_cast<BoxMeshTopology*>(mesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(topology == NULL, std::invalid_argument, "mesh must use a BoxMeshTopology");
  const int dim = topology->getDimension();
  const unsigned allDirections = (1u << dim) - 1;
  allowedDirections &= allDirections;
  TEUCHOS_TEST_FOR_EXCEPTION(allowedDirections == 0, std::invalid_argument, "no refinement directions allowed");

  std::vector<std::pair<GlobalIndexType,GlobalIndexType>> myMasks; // (cellID, mask), for cells this rank owns
  const std::set<GlobalIndexType> &myCells = mesh->cellIDsInPartition();
  const int sideCount = topology->cellTopology()->getSideCount();

  // Piecewise-constant cells have no high modes: for them, the jumps of f's cell means across faces.  The means of the
  // marked p = 0 cells and their neighbours, gathered from their owners.  (Orders and topology are replicated, so all
  // ranks agree on whether this is needed.)
  std::map<GlobalIndexType,double> cellMeans;
  {
    std::set<GlobalIndexType> cellsNeedingMeans;
    for (GlobalIndexType cellID : cellIDs)
    {
      if (mesh->globalDofAssignment()->getH1Order(cellID)[0] > 1) continue;
      cellsNeedingMeans.insert(cellID);
      for (int side=0; side<sideCount; side++)
        for (IndexType neighborID : topology->sideNeighbors(cellID, side)) cellsNeedingMeans.insert(neighborID);
    }
    if (!cellsNeedingMeans.empty())
    {
      std::vector<std::pair<GlobalIndexType,double>> myMeans;
      for (GlobalIndexType cellID : cellsNeedingMeans)
      {
        if (myCells.find(cellID) == myCells.end()) continue;
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
        const FC &weightedMeasures = basisCache->getWeightedMeasures();
        FC values(1, weightedMeasures.dimension(1));
        f->values(values, basisCache);
        double integral = 0, measure = 0;
        for (int q=0; q<weightedMeasures.dimension(1); q++)
        {
          integral += values(0,q) * weightedMeasures(0,q);
          measure += weightedMeasures(0,q);
        }
        myMeans.push_back({cellID, integral / measure});
      }
      std::vector<std::pair<GlobalIndexType,double>> gatheredMeans;
      std::vector<int> meanOffsets;
      MPIWrapper::allGatherVariable(mesh->Comm(), gatheredMeans, myMeans, meanOffsets);
      for (const auto &entry : gatheredMeans) cellMeans[entry.first] = entry.second;
    }
  }

  for (GlobalIndexType cellID : cellIDs)
  {
    if (myCells.find(cellID) == myCells.end()) continue;
    const int p = std::max(mesh->globalDofAssignment()->getH1Order(cellID)[0] - 1, 0); // degree of L2 fields
    unsigned mask = allowedDirections;
    if (p == 0)
    {
      std::vector<double> eta(dim, 0.0);
      const double myMean = cellMeans.at(cellID);
      for (int side=0; side<sideCount; side++)
      {
        const int k = topology->faceForSideOrdinal(side) / 2;
        for (IndexType neighborID : topology->sideNeighbors(cellID, side)) eta[k] = std::max(eta[k], std::abs(cellMeans.at(neighborID) - myMean));
      }
      double maxEta = 0;
      for (int k=0; k<dim; k++) if (allowedDirections & (1u << k)) maxEta = std::max(maxEta, eta[k]);
      if (maxEta > 0)
      {
        mask = 0;
        for (int k=0; k<dim; k++) if ((allowedDirections & (1u << k)) && (eta[k] >= threshold * maxEta)) mask |= (1u << k);
      }
    }
    else
    {
      // sample f at tensor Gauss points, and find its Legendre coefficients c(i_0, ..., i_{d-1}) on the cell
      const int n = p + 1;
      std::vector<double> x, w;
      gaussLegendre(n, x, w);
      int numPoints = 1;
      for (int k=0; k<dim; k++) numPoints *= n;
      FC refPoints(numPoints, dim);
      std::vector<int> index(dim);
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        int remainder = pointOrdinal;
        for (int k=dim-1; k>=0; k--) { index[k] = remainder % n; remainder /= n; }
        for (int k=0; k<dim; k++) refPoints(pointOrdinal,k) = x[index[k]];
      }
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      basisCache->setRefCellPointsFC(refPoints);
      FC values(1, numPoints);
      f->values(values, basisCache);
      std::vector<double> coefficients(numPoints), next(numPoints);
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++) coefficients[pointOrdinal] = values(0,pointOrdinal);
      // A(i,q) = (2i+1)/2 w_q P_i(x_q): values at the points -> Legendre coefficients, one direction at a time
      std::vector<double> A(n*n);
      for (int i=0; i<n; i++) for (int q=0; q<n; q++) A[i*n+q] = 0.5 * (2*i+1) * w[q] * legendre(i, x[q]);
      int stride = numPoints;
      for (int k=0; k<dim; k++)
      {
        stride /= n; // direction k is digit k, the first being the most significant
        std::fill(next.begin(), next.end(), 0.0);
        for (int ordinal=0; ordinal<numPoints; ordinal++)
        {
          const int digit = (ordinal / stride) % n;
          const int base = ordinal - digit * stride;
          for (int i=0; i<n; i++) next[base + i * stride] += A[i*n+digit] * coefficients[ordinal];
        }
        coefficients.swap(next);
      }
      // eta_k^2: the L2 norm (on the reference cell) of the part of f in the highest Legendre modes of direction k: the
      // top two for p >= 3, the top one below that (so that the linear mode, which says nothing about resolution, never
      // counts once there is a quadratic one)
      std::vector<double> eta(dim, 0.0);
      const int lowestHighMode = std::max(std::min(p, 2), p - 1); // p = 1: 1; p = 2: 2; p = 3: 2; p = 4: 3; ...
      for (int ordinal=0; ordinal<numPoints; ordinal++)
      {
        int remainder = ordinal;
        double weight = 1.0;
        for (int k=dim-1; k>=0; k--) { index[k] = remainder % n; remainder /= n; weight *= 2.0 / (2 * index[k] + 1); }
        const double contribution = coefficients[ordinal] * coefficients[ordinal] * weight;
        for (int k=0; k<dim; k++) if (index[k] >= lowestHighMode) eta[k] += contribution;
      }
      double maxEta = 0;
      for (int k=0; k<dim; k++) if (allowedDirections & (1u << k)) maxEta = std::max(maxEta, std::sqrt(eta[k]));
      if (maxEta > 0)
      {
        mask = 0;
        for (int k=0; k<dim; k++)
        {
          if ((allowedDirections & (1u << k)) && (std::sqrt(eta[k]) >= threshold * maxEta)) mask |= (1u << k);
        }
      }
    }
    myMasks.push_back({cellID, GlobalIndexType(mask)});
  }
  std::vector<std::pair<GlobalIndexType,GlobalIndexType>> gatheredMasks;
  std::vector<int> offsets;
  MPIWrapper::allGatherVariable(mesh->Comm(), gatheredMasks, myMasks, offsets);
  std::map<GlobalIndexType,unsigned> masks;
  for (const auto &entry : gatheredMasks) masks[entry.first] = unsigned(entry.second);
  return masks;
}

std::map<GlobalIndexType,unsigned> BoxGDA::masksTangentialToTaggedSides(MeshPtr mesh, const std::string &tagName, int tagID,
                                                                        const std::vector<GlobalIndexType> &cellIDs)
{
  BoxMeshTopology* topology = dynamic_cast<BoxMeshTopology*>(mesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(topology == NULL, std::invalid_argument, "mesh must use a BoxMeshTopology");
  const unsigned allDirections = (1u << topology->getDimension()) - 1;
  std::map<GlobalIndexType,unsigned> masks;
  for (GlobalIndexType cellID : cellIDs)
  {
    unsigned mask = allDirections;
    for (unsigned side : topology->getSidesMatchingTag(cellID, tagName, tagID))
    {
      mask &= ~(1u << (topology->faceForSideOrdinal(side) / 2)); // not in the direction normal to the tagged side
    }
    masks[cellID] = (mask != 0) ? mask : allDirections;
  }
  return masks;
}

// ---------------------------------------------------------------- hp-refinement

BoxGDA::HPRefinements BoxGDA::hpRefinements(MeshPtr mesh, const std::map<GlobalIndexType,unsigned> &directionMasks,
                                            const std::vector<double> &minWidths, int maxH1Order)
{
  TEUCHOS_FUNC_TIME_MONITOR("BoxGDA::hpRefinements");
  BoxMeshTopology* topology = dynamic_cast<BoxMeshTopology*>(mesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(topology == NULL, std::invalid_argument, "mesh must use a BoxMeshTopology");
  const int dim = topology->getDimension();
  TEUCHOS_TEST_FOR_EXCEPTION(int(minWidths.size()) != dim, std::invalid_argument, "minWidths needs one entry per direction");
  GlobalDofAssignmentPtr gda = mesh->globalDofAssignment();
  auto order = [&](GlobalIndexType cellID) { return gda->getH1Order(cellID)[0]; };

  HPRefinements refinements;
  std::map<GlobalIndexType,int> targetOrder; // cells to raise, with their new orders
  for (const auto &[cellID, mask] : directionMasks)
  {
    const BoxMeshTopology::Extents extents = topology->cellExtents(cellID);
    unsigned hMask = 0;
    for (int k=0; k<dim; k++)
    {
      if (!(mask & (1u << k))) continue;
      const double halfWidth = 0.5 * (extents[k].second - extents[k].first);
      if (halfWidth >= minWidths[k] * (1 - 1e-12)) hMask |= (1u << k);
    }
    if (hMask != 0) refinements.hMasks[cellID] = hMask;
    else if (order(cellID) < maxH1Order) targetOrder[cellID] = order(cellID) + 1;
  }

  refinements.pIncrements = orderIncrementsWithHalo(mesh, targetOrder);
  return refinements;
}

std::map<GlobalIndexType,int> BoxGDA::orderIncrementsWithHalo(MeshPtr mesh, std::map<GlobalIndexType,int> targetOrder)
{
  BoxMeshTopology* topology = dynamic_cast<BoxMeshTopology*>(mesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(topology == NULL, std::invalid_argument, "mesh must use a BoxMeshTopology");
  GlobalDofAssignmentPtr gda = mesh->globalDofAssignment();
  auto order = [&](GlobalIndexType cellID) { return gda->getH1Order(cellID)[0]; };
  auto newOrder = [&](GlobalIndexType cellID)
  {
    auto entry = targetOrder.find(cellID);
    return (entry == targetOrder.end()) ? order(cellID) : std::max(order(cellID), entry->second);
  };
  // neighbours' orders may differ by at most one: raise neighbours more than one below a raised cell, and so on
  std::vector<GlobalIndexType> queue;
  for (const auto &entry : targetOrder) queue.push_back(entry.first);
  const int sideCount = topology->cellTopology()->getSideCount();
  while (!queue.empty())
  {
    const GlobalIndexType cellID = queue.back();
    queue.pop_back();
    const int minimumNeighborOrder = newOrder(cellID) - 1;
    for (int side=0; side<sideCount; side++)
    {
      for (IndexType neighborID : topology->sideNeighbors(cellID, side))
      {
        if (newOrder(neighborID) < minimumNeighborOrder)
        {
          targetOrder[neighborID] = minimumNeighborOrder;
          queue.push_back(neighborID);
        }
      }
    }
  }
  std::map<GlobalIndexType,int> increments;
  for (const auto &[cellID, target] : targetOrder)
  {
    if (target > order(cellID)) increments[cellID] = target - order(cellID);
  }
  return increments;
}

void BoxGDA::hpRefine(MeshPtr mesh, const HPRefinements &refinements, bool repartitionAndRebuild)
{
  std::map<int, std::set<GlobalIndexType>> cellsForIncrement;
  for (const auto &[cellID, increment] : refinements.pIncrements) cellsForIncrement[increment].insert(cellID);
  for (const auto &[increment, cellIDs] : cellsForIncrement) mesh->pRefine(cellIDs, increment, false);
  mesh->hRefine(refinements.hMasks, false);
  if (repartitionAndRebuild) mesh->repartitionAndRebuild();
}
