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

namespace
{
  //! the largest entry of any multi-index in the set
  int indexSetDegree(const std::vector<std::vector<int>> &indexSet)
  {
    int degree = 0;
    for (const auto &index : indexSet) for (int i : index) degree = std::max(degree, i);
    return degree;
  }
}

namespace
{
  //! the cell's H^1 order in each direction
  std::vector<int> ordersPerDirection(MeshPtr mesh, GlobalIndexType cellID, int dim)
  {
    std::vector<int> order = mesh->globalDofAssignment()->getH1Order(cellID);
    if (int(order.size()) == 1) order = std::vector<int>(dim, order[0]); // (not assign(): its value argument would refer into the vector being reassigned)
    return order;
  }
  bool directionalOrdersFor(MeshPtr mesh, bool anisotropicP)
  {
    BoxGDA* gda = dynamic_cast<BoxGDA*>(mesh->globalDofAssignment().get());
    return anisotropicP && (gda != NULL) && gda->supportsDirectionalOrders();
  }
}

bool BoxGDA::supportsDirectionalOrders() const
{
  // one tensorial component per direction: Line x Line x ... (BoxMeshTopology uses these from 4D up)
  CellTopoPtr cellTopo = _boxTopology->cellTopology();
  return (cellTopo->getTensorialDegree() + 1 == int(_boxTopology->getDimension())) && (_boxTopology->getDimension() > 1);
}

std::vector<int> BoxGDA::getH1Order(GlobalIndexType cellID)
{
  std::vector<int> order = GlobalDofAssignment::getH1Order(cellID);
  auto entry = _directionalIncrements.find(cellID);
  if (entry == _directionalIncrements.end()) return order;
  const int dim = _boxTopology->getDimension();
  if (int(order.size()) == 1) order = std::vector<int>(dim, order[0]); // (not assign(): its value argument would refer into the vector being reassigned)
  for (int k=0; k<dim; k++) order[k] += entry->second[k];
  return order;
}

ElementTypePtr BoxGDA::elementType(GlobalIndexType cellID)
{
  if (_directionalIncrements.find(cellID) == _directionalIncrements.end()) return GlobalDofAssignment::elementType(cellID);
  std::vector<int> order = getH1Order(cellID); // (DofOrderingFactory takes non-const references)
  ElementTypePtr &elemType = _directionalElementTypes[order];
  if (elemType == Teuchos::null)
  {
    std::vector<int> testOrder = order;
    for (int &p : testOrder) p += _testOrderEnhancement;
    CellTopoPtr cellTopo = _boxTopology->cellTopology();
    DofOrderingPtr trialOrdering = _dofOrderingFactory->trialOrdering(order, cellTopo, _enforceConformityLocally);
    DofOrderingPtr testOrdering = _dofOrderingFactory->testOrdering(testOrder, cellTopo);
    elemType = _elementTypeFactory.getElementType(trialOrdering, testOrdering, cellTopo);
  }
  return elemType;
}

std::vector<GlobalIndexType> BoxGDA::cellIDsOfElementType(PartitionIndexType partitionNumber, ElementTypePtr elemTypePtr)
{
  if (_directionalIncrements.empty()) return GlobalDofAssignment::cellIDsOfElementType(partitionNumber, elemTypePtr);
  if (partitionNumber == -1) partitionNumber = _partitionPolicy->Comm()->getRank();
  std::vector<GlobalIndexType> cellIDs;
  for (GlobalIndexType cellID : _partitions[partitionNumber])
  {
    ElementTypePtr cellType = elementType(cellID);
    if ((cellType.get() == elemTypePtr.get()) || (*cellType == *elemTypePtr)) cellIDs.push_back(cellID);
  }
  return cellIDs;
}

std::vector<int> BoxGDA::cellOrder(GlobalIndexType cellID)
{
  std::vector<int> order = getH1Order(cellID);
  const int dim = _boxTopology->getDimension();
  if (int(order.size()) == 1) order = std::vector<int>(dim, order[0]); // (not assign(): its value argument would refer into the vector being reassigned)
  return order;
}

const BoxGDA::OrderData &BoxGDA::orderDataForCell(GlobalIndexType cellID)
{
  const std::vector<int> order = cellOrder(cellID);
  auto existing = _orderData.find(order);
  if (existing != _orderData.end()) return existing->second;
  OrderData &data = _orderData[order];
  ElementTypePtr elemType = elementType(cellID);
  data.elemType = elemType;
  DofOrderingPtr trialOrdering = elemType->trialOrderPtr;
  const int dim = _boxTopology->getDimension();
  for (VarPtr field : _fieldVars)
  {
    data.fieldOffsets.push_back(data.fieldDofs);
    data.fieldDofs += trialOrdering->getBasisCardinality(field->ID(), VOLUME_INTERIOR_SIDE_INDEX);
  }
  const int maxOrder = *std::max_element(order.begin(), order.end());
  const int tangentialCount = dim - 1;
  const int maxDegree = maxOrder + 2; // comfortably above the flux degree
  // (the side sample points are set explicitly below, so this cache needs no cubature enrichment; a large enrichment
  // asked Intrepid2 for side cubature degrees one macOS build did not provide)
  BasisCachePtr sampleCache = Teuchos::rcp(new BasisCache(elemType));
  FC sampleNodes = _boxTopology->physicalCellNodesForCell(cellID, true);
  sampleCache->setPhysicalCellNodes(sampleNodes, {GlobalIndexType(cellID)}, true);
  const BoxMeshTopology::Extents extents = _boxTopology->cellExtents(cellID);
  Teuchos::LAPACK<int,double> lapack;
  data.sideIndexSets.assign(dim, std::vector<IndexSet>(_fluxVars.size()));
  // Sample points on each side: a tensor Gauss grid in the side's reference coordinates, set explicitly (rather than
  // relying on the side cache's own cubature, which on one macOS build came back without points), with enough points
  // per direction that a product the basis does not contain cannot match it at all of them.
  const int sideDim = dim - 1, pointsPerDirection = maxDegree + 2;
  std::vector<double> gaussPoints, gaussWeights;
  gaussLegendre(pointsPerDirection, gaussPoints, gaussWeights);
  int sidePointCount = 1;
  for (int k=0; k<sideDim; k++) sidePointCount *= pointsPerDirection;
  FC sideRefPoints(sidePointCount, sideDim), sideRefWeights(sidePointCount);
  for (int q=0; q<sidePointCount; q++)
  {
    int remainder = q;
    double weight = 1.0;
    for (int k=sideDim-1; k>=0; k--)
    {
      const int i = remainder % pointsPerDirection;
      remainder /= pointsPerDirection;
      sideRefPoints(q,k) = gaussPoints[i];
      weight *= gaussWeights[i];
    }
    sideRefWeights(q) = weight;
  }
  for (int normal=0; normal<dim; normal++)
  {
    const int side = _boxTopology->sideOrdinalForFace(2 * normal);
    BasisCachePtr sideCache = sampleCache->getSideBasisCache(side);
    sideCache->setRefCellPointsFC(sideRefPoints, sideRefWeights);
    const FC &points = sideCache->getPhysicalCubaturePointsFC();
    const FC &weightedMeasures = sideCache->getWeightedMeasures();
    const int P = points.dimension(1);
    for (size_t fluxOrdinal=0; fluxOrdinal<_fluxVars.size(); fluxOrdinal++)
    {
      VarPtr flux = _fluxVars[fluxOrdinal];
      // Identify the flux space: find which products of Legendre polynomials (one degree per tangential direction)
      // the side basis reproduces.  They must account for the whole basis, and the set of multi-indices must be
      // downward closed, which makes the space invariant under the axis-aligned affine maps between a patch and its
      // faces.  (Q_p, Camellia's serendipity spaces, and their anisotropic versions are all of this kind.)
      const int cardinality = trialOrdering->getBasisCardinality(flux->ID(), side);
      TEUCHOS_TEST_FOR_EXCEPTION(trialOrdering->getBasisCardinality(flux->ID(), _boxTopology->sideOrdinalForFace(2 * normal + 1)) != cardinality,
                                 std::invalid_argument, "BoxGDA: flux basis cardinality differs between opposite sides");
      IndexSet &indexSet = data.sideIndexSets[normal][fluxOrdinal];
      BasisPtr basis = trialOrdering->getBasis(flux->ID(), side);
      auto basisValues = sideCache->getTransformedValues(basis, OP_VALUE);
      const int n = cardinality;
      std::vector<double> M(n*n, 0.0), g(P), c(n);
      for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
          for (int q=0; q<P; q++)
            M[j*n+i] += (*basisValues)(0,i,q) * (*basisValues)(0,j,q) * weightedMeasures(0,q);
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
        if (maxResidual < 1e-9 * maxValue) indexSet.push_back(index);
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
    auto directional = _directionalIncrements.find(parentCellID);
    for (IndexType childID : _boxTopology->getCell(parentCellID)->getChildIndices(_meshTopology))
    {
      if (parentDeltaP != 0) setPRefinementDegree(childID, std::max(parentDeltaP, getPRefinementDegree(childID)));
      if (directional != _directionalIncrements.end()) _directionalIncrements[childID] = directional->second;
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
  projectSolutionsOntoNewOrders(oldTypes);
}

void BoxGDA::pRefineInDirections(MeshPtr mesh, const std::map<GlobalIndexType, std::vector<int>> &increments)
{
  if (increments.empty()) return;
  BoxGDA* gda = dynamic_cast<BoxGDA*>(mesh->globalDofAssignment().get());
  TEUCHOS_TEST_FOR_EXCEPTION(gda == NULL, std::invalid_argument, "mesh must use a BoxGDA");
  const int dim = gda->_boxTopology->getDimension();
  if (!gda->supportsDirectionalOrders())
  {
    // orders cannot differ by direction: raise each cell by its largest increment
    std::map<int, std::set<GlobalIndexType>> cellsForIncrement;
    for (const auto &[cellID, increment] : increments)
    {
      const int largest = *std::max_element(increment.begin(), increment.end());
      if (largest > 0) cellsForIncrement[largest].insert(cellID);
    }
    for (const auto &[increment, cellIDs] : cellsForIncrement) mesh->pRefine(cellIDs, increment, false);
    return;
  }
  std::map<GlobalIndexType, ElementTypePtr> oldTypes;
  const std::set<IndexType> &knownCells = gda->_boxTopology->getLocallyKnownActiveCellIndices();
  for (const auto &entry : increments)
  {
    if (knownCells.find(entry.first) != knownCells.end()) oldTypes[entry.first] = gda->elementType(entry.first);
  }
  for (const auto &[cellID, increment] : increments)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(int(increment.size()) != dim, std::invalid_argument, "increments need one entry per direction");
    std::vector<int> directional = gda->_directionalIncrements.count(cellID) ? gda->_directionalIncrements[cellID] : std::vector<int>(dim, 0);
    for (int k=0; k<dim; k++) directional[k] += increment[k];
    // what is common to every direction is kept as an isotropic p-refinement, so that a cell raised in every direction
    // has the same element type as one p-refined isotropically
    const int common = *std::min_element(directional.begin(), directional.end());
    if (common != 0)
    {
      gda->setPRefinementDegree(cellID, gda->getPRefinementDegree(cellID) + common);
      for (int &d : directional) d -= common;
    }
    if (std::all_of(directional.begin(), directional.end(), [](int d) { return d == 0; })) gda->_directionalIncrements.erase(cellID);
    else gda->_directionalIncrements[cellID] = directional;
  }
  gda->_cellMaps.clear();
  gda->projectSolutionsOntoNewOrders(oldTypes);
}

void BoxGDA::projectSolutionsOntoNewOrders(const std::map<GlobalIndexType, ElementTypePtr> &oldTypes)
{
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
  // Each patch's flux space: the intersection, over the cells whose faces make it up, of their side index sets (the
  // minimum rule).  Distinct spaces are stored once.
  _patchSpaceForPatch.assign(numPatches, -1);
  for (int patchOrdinal=0; patchOrdinal<numPatches; patchOrdinal++)
  {
    const BoxForest::Patch &patch = skeleton.patches[patchOrdinal];
    std::vector<IndexSet> indexSets;
    bool first = true;
    for (const auto *faces : {&patch.minusFaces, &patch.plusFaces})
    {
      for (const BoxForest::Face &face : *faces)
      {
        const std::vector<IndexSet> &cellSets = orderDataForCell(face.cell).sideIndexSets[patch.normal];
        if (first)
        {
          indexSets = cellSets;
          first = false;
          continue;
        }
        for (size_t fluxOrdinal=0; fluxOrdinal<indexSets.size(); fluxOrdinal++)
        {
          // index sets are in lexicographic order (as found by orderDataForCell()), so set_intersection applies
          IndexSet intersection;
          std::set_intersection(indexSets[fluxOrdinal].begin(), indexSets[fluxOrdinal].end(),
                                cellSets[fluxOrdinal].begin(), cellSets[fluxOrdinal].end(), std::back_inserter(intersection));
          indexSets[fluxOrdinal].swap(intersection);
        }
      }
    }
    auto entry = _patchSpaceIndex.find(indexSets);
    if (entry == _patchSpaceIndex.end())
    {
      PatchSpace space;
      space.indexSets = indexSets;
      for (const IndexSet &indexSet : indexSets)
      {
        int degree = 0;
        for (const auto &index : indexSet) for (int i : index) degree = std::max(degree, i);
        space.degrees.push_back(degree);
        space.offsets.push_back(space.dofs);
        space.dofs += int(indexSet.size());
      }
      entry = _patchSpaceIndex.emplace(indexSets, int(_patchSpaces.size())).first;
      _patchSpaces.push_back(space);
    }
    _patchSpaceForPatch[patchOrdinal] = entry->second;
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
      offset += orderDataForCell(cellID).fieldDofs;
    }
    const GlobalIndexType fluxStart = offset;
    for (int patchOrdinal : patchesForPartition[partition])
    {
      _fluxDofOffsetForPatch[patchOrdinal] = offset;
      offset += patchSpace(patchOrdinal).dofs;
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

void BoxGDA::patchBasisValues(const BoxForest::Patch &patch, const IndexSet &indexSet, int degree, const double *x, vector<double> &values) const
{
  // products of Legendre polynomials in the tangential physical coordinates, on the patch's bounding box, for the
  // multi-indices of the index set
  const int dim = _boxTopology->getDimension();
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
  const std::vector<int> myOrder = cellOrder(cellID);
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
    const int patchSpaceOrdinal = _patchSpaceForPatch[patchOrdinal];
    const PatchSpace &patchData = _patchSpaces[patchSpaceOrdinal];
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
      const int n = fluxBasis->getCardinality(), m = int(patchData.indexSets[fluxOrdinal].size());
      phiValues.resize(n*P);
      psiValues.resize(m*P);
      for (int q=0; q<P; q++)
      {
        for (int i=0; i<n; i++) phiValues[i*P+q] = (*fluxValues)(0,i,q);
        patchBasisValues(patch, patchData.indexSets[fluxOrdinal], patchData.degrees[fluxOrdinal], &points(0,q,0), psi);
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
      if (_verifiedFluxSpaces.find({myOrder, patchSpaceOrdinal}) == _verifiedFluxSpaces.end())
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
      const GlobalIndexType offset = _fluxDofOffsetForPatch[patchOrdinal] + patchData.offsets[fluxOrdinal];
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
    _verifiedFluxSpaces.insert({myOrder, _patchSpaceForPatch[patchOrdinal]});
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
        patchBasisValues(patch, data.sideIndexSets[k][fluxOrdinal], indexSetDegree(data.sideIndexSets[k][fluxOrdinal]), &points(0,q,0), psi);
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
          patchBasisValues(sideBox[side], data.sideIndexSets[k][fluxOrdinal], indexSetDegree(data.sideIndexSets[k][fluxOrdinal]), &points(0,q,0), psi);
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
  // orders: meshFrom's boundary cells are raised to their counterparts' orders, direction by direction (with the halo
  // that keeps neighbors within one order); meshTo is left as it is
  {
    std::map<GlobalIndexType, std::vector<int>> targetOrders;
    for (const auto &[fromCellID, toCellID] : cellIDMap)
    {
      targetOrders[fromCellID] = ordersPerDirection(meshTo, toCellID, dim);
    }
    const auto increments = orderIncrementsWithHaloInDirections(meshFrom, targetOrders);
    if (!increments.empty())
    {
      pRefineInDirections(meshFrom, increments);
      meshFrom->repartitionAndRebuild();
    }
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
    for (int order : fineMesh->globalDofAssignment()->getH1Order(cellID)) fineMinOrder = std::min(fineMinOrder, order);
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
  std::map<std::vector<int>, BasisCachePtr> coarseBasisCaches; // by coarse order (their reference points are replaced below)
  Teuchos::LAPACK<int,double> lapack;
  std::vector<double> phi, psi, weights, M, C;
  for (GlobalIndexType fineCellID : _partitions[rank])
  {
    const GlobalIndexType coarseCellID = coarseCellContaining(fineCellID);
    DofOrderingPtr fineOrdering = elementType(fineCellID)->trialOrderPtr;
    DofOrderingPtr coarseOrdering = coarseGDA->elementType(coarseCellID)->trialOrderPtr;
    const BoxMeshTopology::Extents coarseExtents = coarseTopology->cellExtents(coarseCellID);

    const OrderData &fineData = orderDataForCell(fineCellID);
    {
      const std::vector<int> coarseOrder = coarseGDA->cellOrder(coarseCellID), fineOrder = cellOrder(fineCellID);
      for (size_t k=0; k<coarseOrder.size(); k++)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(coarseOrder[k] > fineOrder[k], std::invalid_argument,
                                   "BoxGDA::prolongationOperator: a coarse cell has a higher order than a fine cell in it");
      }
    }
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
                                         + coarseGDA->orderDataForCell(coarseCellID).fieldOffsets[fieldOrdinal];
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
    const PatchSpace &finePatchData = patchSpace(patchOrdinal);
    const PatchSpace &coarsePatchData = coarseGDA->patchSpace(coarsePatchOrdinal);

    for (size_t fluxOrdinal=0; fluxOrdinal<_fluxVars.size(); fluxOrdinal++)
    {
      const int fineDegree = finePatchData.degrees[fluxOrdinal];
      {
        // the coarse space must lie in the fine one (restriction keeps each multi-index or lowers its entries, and the
        // fine index set is downward closed)
        const std::set<std::vector<int>> fineSet(finePatchData.indexSets[fluxOrdinal].begin(), finePatchData.indexSets[fluxOrdinal].end());
        for (const auto &index : coarsePatchData.indexSets[fluxOrdinal])
        {
          TEUCHOS_TEST_FOR_EXCEPTION(fineSet.find(index) == fineSet.end(), std::invalid_argument,
                                     "BoxGDA::prolongationOperator: a coarse patch's flux space is not in the fine patch's");
        }
      }
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
      const std::vector<std::vector<int>> &fineIndexSet = finePatchData.indexSets[fluxOrdinal];
      const std::vector<std::vector<int>> &coarseIndexSet = coarsePatchData.indexSets[fluxOrdinal];
      const GlobalIndexType fineOffset = _fluxDofOffsetForPatch[patchOrdinal] + finePatchData.offsets[fluxOrdinal];
      const GlobalIndexType coarseOffset = coarseGDA->_fluxDofOffsetForPatch[coarsePatchOrdinal] + coarsePatchData.offsets[fluxOrdinal];
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

}

std::map<GlobalIndexType,unsigned> BoxGDA::anisotropicRefinementMasks(MeshPtr mesh, FunctionPtr f, const std::vector<GlobalIndexType> &cellIDs,
                                                                      double threshold, unsigned allowedDirections,
                                                                      const std::map<int,FunctionPtr> &boundaryData)
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
  // marked p = 0 cells and their neighbors, gathered from their owners.  (Orders and topology are replicated, so all
  // ranks agree on whether this is needed.)
  std::map<GlobalIndexType,double> cellMeans;
  {
    std::set<GlobalIndexType> cellsNeedingMeans;
    for (GlobalIndexType cellID : cellIDs)
    {
      const std::vector<int> order = ordersPerDirection(mesh, cellID, dim);
      if (*std::min_element(order.begin(), order.end()) > 1) continue; // no direction with p = 0
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
    // degree of the L2 fields in each direction
    std::vector<int> p = ordersPerDirection(mesh, cellID, dim);
    for (int &pk : p) pk = std::max(pk - 1, 0);
    const bool piecewiseConstant = std::all_of(p.begin(), p.end(), [](int pk) { return pk == 0; });
    unsigned mask = allowedDirections;
    // the largest jump of f's cell mean across the cell's faces normal to each direction
    auto jumps = [&]()
    {
      std::vector<double> jump(dim, 0.0);
      const double myMean = cellMeans.at(cellID);
      for (int side=0; side<sideCount; side++)
      {
        const int face = topology->faceForSideOrdinal(side), k = face / 2;
        for (IndexType neighborID : topology->sideNeighbors(cellID, side)) jump[k] = std::max(jump[k], std::abs(cellMeans.at(neighborID) - myMean));
        auto data = boundaryData.find(face);
        if ((data != boundaryData.end()) && topology->sideIsOnBoundary(cellID, side))
        {
          // the jump against the data on this boundary face: their mean over the face
          BasisCachePtr sideCache = BasisCache::basisCacheForCell(mesh, cellID)->getSideBasisCache(side);
          const FC &weightedMeasures = sideCache->getWeightedMeasures();
          FC values(1, weightedMeasures.dimension(1));
          data->second->values(values, sideCache);
          double integral = 0, measure = 0;
          for (int q=0; q<weightedMeasures.dimension(1); q++)
          {
            integral += values(0,q) * weightedMeasures(0,q);
            measure += weightedMeasures(0,q);
          }
          if (measure > 0) jump[k] = std::max(jump[k], std::abs(integral / measure - myMean));
        }
      }
      return jump;
    };
    if (piecewiseConstant)
    {
      std::vector<double> eta = jumps();
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
      // sample f at tensor Gauss points (p_k + 1 in direction k), and find its Legendre coefficients c(i_0, ..., i_{d-1})
      std::vector<int> n(dim);
      std::vector<std::vector<double>> x(dim), w(dim);
      int numPoints = 1;
      for (int k=0; k<dim; k++)
      {
        n[k] = p[k] + 1;
        gaussLegendre(n[k], x[k], w[k]);
        numPoints *= n[k];
      }
      // ordinals are mixed-radix numbers, direction 0 the most significant digit
      std::vector<int> stride(dim);
      {
        int product = 1;
        for (int k=dim-1; k>=0; k--) { stride[k] = product; product *= n[k]; }
      }
      FC refPoints(numPoints, dim);
      std::vector<int> index(dim);
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        for (int k=0; k<dim; k++) index[k] = (pointOrdinal / stride[k]) % n[k];
        for (int k=0; k<dim; k++) refPoints(pointOrdinal,k) = x[k][index[k]];
      }
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      basisCache->setRefCellPointsFC(refPoints);
      FC values(1, numPoints);
      f->values(values, basisCache);
      std::vector<double> coefficients(numPoints), next(numPoints);
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++) coefficients[pointOrdinal] = values(0,pointOrdinal);
      // A_k(i,q) = (2i+1)/2 w_q P_i(x_q): values at the points -> Legendre coefficients, one direction at a time
      for (int k=0; k<dim; k++)
      {
        std::vector<double> A(n[k]*n[k]);
        for (int i=0; i<n[k]; i++) for (int q=0; q<n[k]; q++) A[i*n[k]+q] = 0.5 * (2*i+1) * w[k][q] * legendre(i, x[k][q]);
        std::fill(next.begin(), next.end(), 0.0);
        for (int ordinal=0; ordinal<numPoints; ordinal++)
        {
          const int digit = (ordinal / stride[k]) % n[k];
          const int base = ordinal - digit * stride[k];
          for (int i=0; i<n[k]; i++) next[base + i * stride[k]] += A[i*n[k]+digit] * coefficients[ordinal];
        }
        coefficients.swap(next);
      }
      // eta_k^2: the L2 norm (on the reference cell) of the part of f in the highest Legendre modes of direction k: the
      // top two for p_k >= 3, the top one below that (so that the linear mode, which says nothing about resolution,
      // never counts once there is a quadratic one)
      std::vector<double> eta(dim, 0.0);
      for (int ordinal=0; ordinal<numPoints; ordinal++)
      {
        double weight = 1.0;
        for (int k=0; k<dim; k++) { index[k] = (ordinal / stride[k]) % n[k]; weight *= 2.0 / (2 * index[k] + 1); }
        const double contribution = coefficients[ordinal] * coefficients[ordinal] * weight;
        for (int k=0; k<dim; k++)
        {
          if (p[k] == 0) continue;
          const int lowestHighMode = std::max(std::min(p[k], 2), p[k] - 1); // p = 1: 1; p = 2: 2; p = 3: 2; p = 4: 3; ...
          if (index[k] >= lowestHighMode) eta[k] += contribution;
        }
      }
      // In a direction with p_k = 0 (orders raised in other directions only) there are no high modes: there, the jump
      // of the cell mean across the faces normal to k, as the linear mode it implies (coefficient half the jump), in
      // the same norm.
      if (std::any_of(p.begin(), p.end(), [](int pk) { return pk == 0; }))
      {
        const std::vector<double> jump = jumps();
        for (int k=0; k<dim; k++)
        {
          if (p[k] != 0) continue;
          eta[k] = 0.25 * jump[k] * jump[k] * (2.0 / 3.0) * std::ldexp(1.0, dim - 1);
        }
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

namespace
{
  //! the directions in the mask in which half the cell's width is at least the minimum width
  unsigned bisectableDirections(const BoxMeshTopology *topology, GlobalIndexType cellID, unsigned mask, const std::vector<double> &minWidths)
  {
    const BoxMeshTopology::Extents extents = topology->cellExtents(cellID);
    unsigned hMask = 0;
    for (int k=0; k<int(extents.size()); k++)
    {
      if (!(mask & (1u << k))) continue;
      const double halfWidth = 0.5 * (extents[k].second - extents[k].first);
      if (halfWidth >= minWidths[k] * (1 - 1e-12)) hMask |= (1u << k);
    }
    return hMask;
  }
}

std::set<GlobalIndexType> BoxGDA::hpSaturatedCells(MeshPtr mesh, const std::map<GlobalIndexType,unsigned> &directionMasks,
                                                   const std::vector<double> &minWidths, int maxH1Order, bool anisotropicP)
{
  BoxMeshTopology* topology = dynamic_cast<BoxMeshTopology*>(mesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(topology == NULL, std::invalid_argument, "mesh must use a BoxMeshTopology");
  const int dim = topology->getDimension();
  TEUCHOS_TEST_FOR_EXCEPTION(int(minWidths.size()) != dim, std::invalid_argument, "minWidths needs one entry per direction");
  const bool directional = directionalOrdersFor(mesh, anisotropicP);
  std::set<GlobalIndexType> saturated;
  for (const auto &[cellID, mask] : directionMasks)
  {
    if (bisectableDirections(topology, cellID, mask, minWidths) != 0) continue;
    const std::vector<int> order = ordersPerDirection(mesh, cellID, dim);
    bool atMaximum = true;
    if (directional)
    {
      // at the maximum in every chosen direction
      for (int k=0; k<dim; k++) if ((mask & (1u << k)) && (order[k] < maxH1Order)) atMaximum = false;
    }
    else
    {
      atMaximum = (*std::max_element(order.begin(), order.end()) >= maxH1Order);
    }
    if (atMaximum) saturated.insert(cellID);
  }
  return saturated;
}

BoxGDA::HPRefinements BoxGDA::hpRefinements(MeshPtr mesh, const std::map<GlobalIndexType,unsigned> &directionMasks,
                                            const std::vector<double> &minWidths, int maxH1Order, bool anisotropicP)
{
  TEUCHOS_FUNC_TIME_MONITOR("BoxGDA::hpRefinements");
  BoxMeshTopology* topology = dynamic_cast<BoxMeshTopology*>(mesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(topology == NULL, std::invalid_argument, "mesh must use a BoxMeshTopology");
  const int dim = topology->getDimension();
  TEUCHOS_TEST_FOR_EXCEPTION(int(minWidths.size()) != dim, std::invalid_argument, "minWidths needs one entry per direction");
  HPRefinements refinements;
  if (directionalOrdersFor(mesh, anisotropicP))
  {
    // each chosen direction: bisect if the cell is wide enough there, else raise the order in that direction
    std::map<GlobalIndexType, std::vector<int>> targetOrders;
    for (const auto &[cellID, mask] : directionMasks)
    {
      const unsigned hMask = bisectableDirections(topology, cellID, mask, minWidths);
      if (hMask != 0) refinements.hMasks[cellID] = hMask;
      std::vector<int> target = ordersPerDirection(mesh, cellID, dim);
      bool raise = false;
      for (int k=0; k<dim; k++)
      {
        if ((mask & (1u << k)) && !(hMask & (1u << k)) && (target[k] < maxH1Order))
        {
          target[k]++;
          raise = true;
        }
      }
      if (raise) targetOrders[cellID] = target;
    }
    refinements.pIncrementsInDirections = orderIncrementsWithHaloInDirections(mesh, targetOrders);
    return refinements;
  }
  auto order = [&](GlobalIndexType cellID)
  {
    const std::vector<int> orders = ordersPerDirection(mesh, cellID, dim);
    return *std::max_element(orders.begin(), orders.end());
  };
  std::map<GlobalIndexType,int> targetOrder; // cells to raise, with their new orders
  for (const auto &[cellID, mask] : directionMasks)
  {
    const unsigned hMask = bisectableDirections(topology, cellID, mask, minWidths);
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
  const int dim = topology->getDimension();
  auto order = [&](GlobalIndexType cellID)
  {
    const std::vector<int> orders = ordersPerDirection(mesh, cellID, dim);
    return *std::max_element(orders.begin(), orders.end());
  };
  auto newOrder = [&](GlobalIndexType cellID)
  {
    auto entry = targetOrder.find(cellID);
    return (entry == targetOrder.end()) ? order(cellID) : std::max(order(cellID), entry->second);
  };
  // neighbors' orders may differ by at most one: raise neighbors more than one below a raised cell, and so on
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

std::map<GlobalIndexType, std::vector<int>> BoxGDA::orderIncrementsWithHaloInDirections(MeshPtr mesh, std::map<GlobalIndexType, std::vector<int>> targetOrders)
{
  BoxMeshTopology* topology = dynamic_cast<BoxMeshTopology*>(mesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(topology == NULL, std::invalid_argument, "mesh must use a BoxMeshTopology");
  const int dim = topology->getDimension();
  auto newOrder = [&](GlobalIndexType cellID)
  {
    std::vector<int> order = ordersPerDirection(mesh, cellID, dim);
    auto entry = targetOrders.find(cellID);
    if (entry != targetOrders.end()) for (int k=0; k<dim; k++) order[k] = std::max(order[k], entry->second[k]);
    return order;
  };
  // in each direction, face neighbors' orders may differ by at most one
  std::vector<GlobalIndexType> queue;
  for (const auto &entry : targetOrders) queue.push_back(entry.first);
  const int sideCount = topology->cellTopology()->getSideCount();
  while (!queue.empty())
  {
    const GlobalIndexType cellID = queue.back();
    queue.pop_back();
    const std::vector<int> cellOrder = newOrder(cellID);
    for (int side=0; side<sideCount; side++)
    {
      for (IndexType neighborID : topology->sideNeighbors(cellID, side))
      {
        std::vector<int> neighborOrder = newOrder(neighborID);
        bool raised = false;
        for (int k=0; k<dim; k++)
        {
          if (neighborOrder[k] < cellOrder[k] - 1)
          {
            neighborOrder[k] = cellOrder[k] - 1;
            raised = true;
          }
        }
        if (raised)
        {
          targetOrders[neighborID] = neighborOrder;
          queue.push_back(neighborID);
        }
      }
    }
  }
  std::map<GlobalIndexType, std::vector<int>> increments;
  for (const auto &[cellID, target] : targetOrders)
  {
    const std::vector<int> order = ordersPerDirection(mesh, cellID, dim);
    std::vector<int> increment(dim, 0);
    bool any = false;
    for (int k=0; k<dim; k++)
    {
      increment[k] = std::max(target[k] - order[k], 0);
      if (increment[k] > 0) any = true;
    }
    if (any) increments[cellID] = increment;
  }
  return increments;
}

void BoxGDA::hpRefine(MeshPtr mesh, const HPRefinements &refinements, bool repartitionAndRebuild)
{
  std::map<int, std::set<GlobalIndexType>> cellsForIncrement;
  for (const auto &[cellID, increment] : refinements.pIncrements) cellsForIncrement[increment].insert(cellID);
  for (const auto &[increment, cellIDs] : cellsForIncrement) mesh->pRefine(cellIDs, increment, false);
  pRefineInDirections(mesh, refinements.pIncrementsInDirections);
  mesh->hRefine(refinements.hMasks, false);
  if (repartitionAndRebuild) mesh->repartitionAndRebuild();
}

// ---------------------------------------------------------------- recording and replaying a mesh

std::vector<BoxGDA::CellRecord> BoxGDA::cellRecords(MeshPtr mesh)
{
  BoxMeshTopology* topology = dynamic_cast<BoxMeshTopology*>(mesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(topology == NULL, std::invalid_argument, "mesh must use a BoxMeshTopology");
  std::vector<CellRecord> records;
  for (IndexType cellID : topology->getLocallyKnownActiveCellIndices())
  {
    records.push_back({topology->forest().node(int(cellID)).box, ordersPerDirection(mesh, cellID, topology->getDimension())});
  }
  return records;
}

void BoxGDA::refineToContain(MeshPtr mesh, const std::vector<CellRecord> &records)
{
  TEUCHOS_FUNC_TIME_MONITOR("BoxGDA::refineToContain");
  BoxMeshTopology* topology = dynamic_cast<BoxMeshTopology*>(mesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(topology == NULL, std::invalid_argument, "mesh must use a BoxMeshTopology");
  const int dim = topology->getDimension();
  auto boxBounds = [dim](const BoxForest::Box &box, std::array<BoxForest::Coord,BoxForest::MAX_DIM> &lo, std::array<BoxForest::Coord,BoxForest::MAX_DIM> &hi)
  {
    for (int k=0; k<dim; k++) { lo[k] = box.lo[k]; hi[k] = box.hi(k); }
  };
  // h: bisect each cell in the directions in which it is coarser than a recorded box it overlaps, until none is
  bool refined = false;
  while (true)
  {
    std::map<GlobalIndexType,unsigned> masks;
    std::array<BoxForest::Coord,BoxForest::MAX_DIM> lo, hi;
    for (const CellRecord &record : records)
    {
      boxBounds(record.box, lo, hi);
      topology->forest().forEachLeafOverlapping(lo, hi, [&](int leaf)
      {
        const BoxForest::Box &leafBox = topology->forest().node(leaf).box;
        unsigned mask = 0;
        for (int k=0; k<dim; k++) if (leafBox.level[k] < record.box.level[k]) mask |= (1u << k);
        if (mask != 0) masks[GlobalIndexType(leaf)] |= mask;
      });
    }
    if (masks.empty()) break;
    // one level at a time: a cell several levels coarser than a record is bisected again on the next pass
    mesh->hRefine(masks, false);
    refined = true;
  }
  // p: each cell to at least the orders of the recorded boxes it lies in, direction by direction
  std::map<GlobalIndexType, std::vector<int>> targetOrders;
  std::array<BoxForest::Coord,BoxForest::MAX_DIM> lo, hi;
  for (const CellRecord &record : records)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(int(record.H1Orders.size()) != dim, std::invalid_argument, "records need one order per direction");
    boxBounds(record.box, lo, hi);
    topology->forest().forEachLeafOverlapping(lo, hi, [&](int leaf)
    {
      std::vector<int> &target = targetOrders[GlobalIndexType(leaf)];
      if (target.empty()) target.assign(dim, 0);
      for (int k=0; k<dim; k++) target[k] = std::max(target[k], record.H1Orders[k]);
    });
  }
  const auto increments = orderIncrementsWithHaloInDirections(mesh, targetOrders);
  pRefineInDirections(mesh, increments);
  if (refined || !increments.empty()) mesh->repartitionAndRebuild();
}

// ---------------------------------------------------------------- traces between meshes

Camellia::FieldContainer<double> BoxGDA::transferSideCoefficients(MeshPtr fromMesh, GlobalIndexType fromCellID, int fromSide,
                                                                  MeshPtr toMesh, GlobalIndexType toCellID, int toSide,
                                                                  int fromVarID, int toVarID, const Camellia::FieldContainer<double> &fromCoefficients)
{
  BoxGDA* from = dynamic_cast<BoxGDA*>(fromMesh->globalDofAssignment().get());
  BoxGDA* to = dynamic_cast<BoxGDA*>(toMesh->globalDofAssignment().get());
  TEUCHOS_TEST_FOR_EXCEPTION((from == NULL) || (to == NULL), std::invalid_argument, "both meshes must have a BoxGDA");
  const int normal = from->_boxTopology->faceForSideOrdinal(fromSide) / 2;
  TEUCHOS_TEST_FOR_EXCEPTION(to->_boxTopology->faceForSideOrdinal(toSide) / 2 != normal, std::invalid_argument, "the sides must have the same normal direction");
  int fluxOrdinal = -1;
  for (size_t i=0; i<from->_fluxVars.size(); i++) if (from->_fluxVars[i]->ID() == fromVarID) fluxOrdinal = int(i);
  TEUCHOS_TEST_FOR_EXCEPTION(fluxOrdinal < 0, std::invalid_argument, "fromVarID is not a flux variable");
  const int dim = from->_boxTopology->getDimension();
  Teuchos::LAPACK<int,double> lapack;
  // solves (phi, phi) c = (phi, values) for the L2 projection onto the functions phi (n x P, point index fastest)
  auto project = [&](const std::vector<double> &phi, const std::vector<double> &values, const FC &weightedMeasures, int n)
  {
    const int P = weightedMeasures.dimension(1);
    std::vector<double> M(n*n, 0.0), c(n, 0.0);
    for (int i=0; i<n; i++)
    {
      for (int j=0; j<n; j++) for (int q=0; q<P; q++) M[j*n+i] += phi[i*P+q] * phi[j*P+q] * weightedMeasures(0,q);
      for (int q=0; q<P; q++) c[i] += phi[i*P+q] * values[q] * weightedMeasures(0,q);
    }
    int info;
    lapack.POSV('U', n, 1, M.data(), n, c.data(), n, &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxGDA::transferSideCoefficients: projection failed");
    return c;
  };
  // the from-side function as tensor-Legendre coefficients on the from cell's side box
  const OrderData &fromData = from->orderDataForCell(fromCellID);
  const IndexSet &indexSet = fromData.sideIndexSets[normal][fluxOrdinal];
  const int degree = indexSetDegree(indexSet);
  BoxForest::Patch faceBox;
  faceBox.normal = normal;
  const BoxForest::Box &fromBox = from->_boxTopology->forest().node(int(fromCellID)).box;
  for (int k=0; k<dim; k++) { faceBox.lo[k] = fromBox.lo[k]; faceBox.hi[k] = fromBox.hi(k); }
  std::vector<double> legendreCoefficients;
  {
    FC nodes = from->_boxTopology->physicalCellNodesForCell(fromCellID, true);
    fromData.basisCache->setPhysicalCellNodes(nodes, {fromCellID}, true);
    BasisCachePtr sideCache = fromData.basisCache->getSideBasisCache(fromSide);
    BasisPtr basis = fromData.elemType->trialOrderPtr->getBasis(fromVarID, fromSide);
    auto basisValues = sideCache->getTransformedValues(basis, OP_VALUE);
    const FC &points = sideCache->getPhysicalCubaturePointsFC();
    const int P = points.dimension(1), n = basis->getCardinality(), m = int(indexSet.size());
    TEUCHOS_TEST_FOR_EXCEPTION(int(fromCoefficients.size()) != n, std::invalid_argument, "fromCoefficients must match the from-side basis");
    std::vector<double> values(P, 0.0), psiValues(m*P), psi;
    for (int q=0; q<P; q++)
    {
      for (int i=0; i<n; i++) values[q] += fromCoefficients[i] * (*basisValues)(0,i,q);
      from->patchBasisValues(faceBox, indexSet, degree, &points(0,q,0), psi);
      for (int j=0; j<m; j++) psiValues[j*P+q] = psi[j];
    }
    legendreCoefficients = project(psiValues, values, sideCache->getWeightedMeasures(), m);
  }
  // evaluated on the to-side and projected onto its basis (the tangential coordinates of the two sides coincide; the
  // Legendre products do not depend on the normal coordinate)
  const OrderData &toData = to->orderDataForCell(toCellID);
  FC nodes = to->_boxTopology->physicalCellNodesForCell(toCellID, true);
  toData.basisCache->setPhysicalCellNodes(nodes, {toCellID}, true);
  BasisCachePtr sideCache = toData.basisCache->getSideBasisCache(toSide);
  BasisPtr basis = toData.elemType->trialOrderPtr->getBasis(toVarID, toSide);
  auto basisValues = sideCache->getTransformedValues(basis, OP_VALUE);
  const FC &points = sideCache->getPhysicalCubaturePointsFC();
  const int P = points.dimension(1), n = basis->getCardinality(), m = int(indexSet.size());
  std::vector<double> values(P, 0.0), phiValues(n*P), psi;
  for (int q=0; q<P; q++)
  {
    from->patchBasisValues(faceBox, indexSet, degree, &points(0,q,0), psi);
    for (int j=0; j<m; j++) values[q] += legendreCoefficients[j] * psi[j];
    for (int i=0; i<n; i++) phiValues[i*P+q] = (*basisValues)(0,i,q);
  }
  const std::vector<double> c = project(phiValues, values, sideCache->getWeightedMeasures(), n);
  FC toCoefficients(n);
  for (int i=0; i<n; i++) toCoefficients[i] = c[i];
  return toCoefficients;
}

