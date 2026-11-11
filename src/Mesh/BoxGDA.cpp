//
//  BoxGDA.cpp
//  Camellia
//
//  See BoxGDA.h.
//

#include "BoxGDA.h"

#include "InducedMeshPartitionPolicy.h"
#include "Mesh.h"
#include "VarFactory.h"

#include <Teuchos_LAPACK.hpp>

#include <cmath>

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

  int intPower(int base, int exponent)
  {
    int result = 1;
    for (int i=0; i<exponent; i++) result *= base;
    return result;
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

void BoxGDA::initializeVariables()
{
  if (_basisCache != Teuchos::null) return;
  ElementTypePtr elemType = elementType(*_boxTopology->getLocallyKnownActiveCellIndices().begin());
  DofOrderingPtr trialOrdering = elemType->trialOrderPtr;
  const int dim = _boxTopology->getDimension();
  const int sideCount = _boxTopology->cellTopology()->getSideCount();
  _fieldDofsPerCell = 0;
  _fluxDofsPerPatch = 0;
  for (VarPtr field : _fieldVars)
  {
    _fieldOffsets.push_back(_fieldDofsPerCell);
    _fieldDofsPerCell += trialOrdering->getBasisCardinality(field->ID(), VOLUME_INTERIOR_SIDE_INDEX);
  }
  for (VarPtr flux : _fluxVars)
  {
    const int cardinality = trialOrdering->getBasisCardinality(flux->ID(), 0);
    int degree = 0;
    while (intPower(degree + 1, dim - 1) < cardinality) degree++;
    TEUCHOS_TEST_FOR_EXCEPTION(intPower(degree + 1, dim - 1) != cardinality, std::invalid_argument,
                               "BoxGDA: flux basis is not a full tensor-product polynomial space");
    for (int side=1; side<sideCount; side++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(trialOrdering->getBasisCardinality(flux->ID(), side) != cardinality, std::invalid_argument,
                                 "BoxGDA: flux basis cardinality varies from side to side");
    }
    _fluxDegrees.push_back(degree);
    _fluxOffsets.push_back(_fluxDofsPerPatch);
    _fluxDofsPerPatch += cardinality;
  }
  _basisCache = Teuchos::rcp(new BasisCache(elemType));
  _parentBasisCache = Teuchos::rcp(new BasisCache(elemType));
}

GlobalDofAssignmentPtr BoxGDA::deepCopy()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BoxGDA::deepCopy() is not supported");
}

void BoxGDA::didChangePartitionPolicy() { rebuildLookups(); }

void BoxGDA::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP)
{
  TEUCHOS_TEST_FOR_EXCEPTION(deltaP != 0, std::invalid_argument, "BoxGDA does not support p-refinement yet");
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
  initializeVariables();
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
      offset += _fieldDofsPerCell;
    }
    for (int patchOrdinal : patchesForPartition[partition])
    {
      _fluxDofOffsetForPatch[patchOrdinal] = offset;
      offset += _fluxDofsPerPatch;
    }
    if (partition == rank)
    {
      _myFieldDofCount = GlobalIndexType(_partitions[partition].size()) * _fieldDofsPerCell;
      _myFluxDofCount = GlobalIndexType(patchesForPartition[partition].size()) * _fluxDofsPerPatch;
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

void BoxGDA::patchBasisValues(const BoxForest::Patch &patch, int degree, const double *x, vector<double> &values) const
{
  // tensor Legendre in the tangential physical coordinates, on the patch's bounding box
  const int dim = _boxTopology->getDimension();
  values.assign(1, 1.0);
  vector<double> next;
  for (int k=0; k<dim; k++)
  {
    if (k == patch.normal) continue;
    const double a = _boxTopology->coordinate(k, patch.lo[k]), b = _boxTopology->coordinate(k, patch.hi[k]);
    const double xi = -1 + 2 * (x[k] - a) / (b - a);
    next.resize(values.size() * (degree + 1));
    for (size_t j=0; j<values.size(); j++)
      for (int i=0; i<=degree; i++)
        next[j*(degree+1)+i] = values[j] * legendre(i, xi);
    values.swap(next);
  }
}

const BoxGDA::CellMap &BoxGDA::cellMap(GlobalIndexType cellID)
{
  auto existing = _cellMaps.find(cellID);
  if (existing != _cellMaps.end()) return existing->second;
  TEUCHOS_TEST_FOR_EXCEPTION(_fieldDofOffsetForCell[cellID] < 0, std::invalid_argument, "BoxGDA: dof map requested for an inactive cell");

  const BoxForest &forest = _boxTopology->forest();
  const BoxForest::Skeleton &skeleton = _boxTopology->skeleton();
  DofOrderingPtr trialOrdering = elementType(cellID)->trialOrderPtr;
  const int numTrial = trialOrdering->totalDofs();
  const int dim = _boxTopology->getDimension();
  const int leafOrdinal = forest.leafOrdinal(cellID);

  CellMap &map = _cellMaps[cellID];
  map.rows.assign(numTrial, {});
  for (size_t fieldOrdinal=0; fieldOrdinal<_fieldVars.size(); fieldOrdinal++)
  {
    const int varID = _fieldVars[fieldOrdinal]->ID();
    BasisMap &basisMap = map.basisMaps[{varID, VOLUME_INTERIOR_SIDE_INDEX}];
    basisMap.localDofIndices = trialOrdering->getDofIndices(varID);
    const GlobalIndexType offset = _fieldDofOffsetForCell[cellID] + _fieldOffsets[fieldOrdinal];
    for (size_t i=0; i<basisMap.localDofIndices.size(); i++)
    {
      map.rows[basisMap.localDofIndices[i]].push_back({int(map.globalIndices.size()), 1.0});
      map.globalIndices.push_back(offset + GlobalIndexType(i));
      basisMap.globalIndices.push_back(offset + GlobalIndexType(i));
    }
  }

  FC nodes = _boxTopology->physicalCellNodesForCell(cellID, true);
  vector<GlobalIndexType> cellIDs = {cellID};
  _basisCache->setPhysicalCellNodes(nodes, cellIDs, true);

  vector<double> psi, phiValues, psiValues, weights, M;
  Teuchos::LAPACK<int,double> lapack;
  for (int face=0; face<2*dim; face++)
  {
    const int side = _boxTopology->sideOrdinalForFace(face);
    const int patchOrdinal = skeleton.patchForFace[forest.faceOrdinal(leafOrdinal, face)];
    const BoxForest::Patch &patch = skeleton.patches[patchOrdinal];
    BasisCachePtr sideCache = _basisCache->getSideBasisCache(side);
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
      const int n = fluxBasis->getCardinality(), m = n;
      phiValues.resize(n*P);
      psiValues.resize(m*P);
      for (int q=0; q<P; q++)
      {
        for (int i=0; i<n; i++) phiValues[i*P+q] = (*fluxValues)(0,i,q);
        patchBasisValues(patch, _fluxDegrees[fluxOrdinal], &points(0,q,0), psi);
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
      int info;
      lapack.POSV('U', n, m, M.data(), n, basisMap.weights.data(), n, &info);
      TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxGDA: side-to-patch projection failed");

      basisMap.localDofIndices = trialOrdering->getDofIndices(varID, side);
      const GlobalIndexType offset = _fluxDofOffsetForPatch[patchOrdinal] + _fluxOffsets[fluxOrdinal];
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
  }
  return map;
}

void BoxGDA::interpretLocalData(GlobalIndexType cellID, const FC &localData, FC &globalData,
                                Camellia::FieldContainer<GlobalIndexType> &globalDofIndices)
{
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
  // the patch polynomial restricted to this side is weights * g; the restriction is injective, so solve weights * g = c
  vector<double> W = basisMap.weights, g(n);
  for (int i=0; i<n; i++) g[i] = basisCoefficients[i];
  vector<int> pivots(n);
  Teuchos::LAPACK<int,double> lapack;
  int info;
  lapack.GESV(n, 1, W.data(), n, pivots.data(), g.data(), n, &info);
  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxGDA: could not interpret side coefficients");
  for (int j=0; j<m; j++) globalCoefficients(j) = g[j];
}

void BoxGDA::interpretGlobalCoefficients(GlobalIndexType cellID, FC &localCoefficients, const TVectorPtr<double> globalCoefficients)
{
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
  initializeVariables();
  DofOrderingPtr trialOrdering = elementType(parentCellID)->trialOrderPtr; // uniform order: the children's ordering too
  const int numTrial = trialOrdering->totalDofs();
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
  _parentBasisCache->setPhysicalCellNodes(parentNodes, {parentCellID}, true);
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
    BasisCachePtr sideCache = _parentBasisCache->getSideBasisCache(side);
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
        patchBasisValues(patch, _fluxDegrees[fluxOrdinal], &points(0,q,0), psi);
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
    _basisCache->setPhysicalCellNodes(childNodes, {childID}, true);

    // fields: evaluate the parent's basis at the child's cubature points, mapped to the parent's reference cell
    {
      const FC &points = _basisCache->getPhysicalCubaturePointsFC();
      const FC &weightedMeasures = _basisCache->getWeightedMeasures();
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
        auto childValues = _basisCache->getTransformedValues(basis, OP_VALUE);
        phiValues.resize(n*P);
        for (int i=0; i<n; i++) for (int q=0; q<P; q++) phiValues[i*P+q] = (*childValues)(0,i,q);
        // the reference points of _parentBasisCache are replaced; its physical nodes are still the parent's
        _parentBasisCache->setRefCellPointsFC(parentRefPoints);
        auto parentValues = _parentBasisCache->getTransformedValues(basis, OP_VALUE);
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
      BasisCachePtr sideCache = _basisCache->getSideBasisCache(side);
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
          patchBasisValues(sideBox[side], _fluxDegrees[fluxOrdinal], &points(0,q,0), psi);
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
  // repeat, because a configuration cell may need more than one level to catch up with the phase-space children
  bool refinedSome = false;
  while (true)
  {
    std::set<GlobalIndexType> configurationCellsToRefine;
    for (const auto &entry : cellDirectionMasks)
    {
      const unsigned mask = entry.second & configurationMask;
      if (mask == 0) continue; // velocity directions only
      TEUCHOS_TEST_FOR_EXCEPTION(mask != configurationMask, std::invalid_argument,
                                 "phase-space cells must be bisected in all configuration directions together, or in none");
      // the configuration cell matching the refined phase-space cell must itself be refined
      const IndexType configurationCellID = _boxTopology->phaseSpaceConfigurationCellID(entry.first);
      if (!_configurationMesh->getTopology()->isParent(configurationCellID)) configurationCellsToRefine.insert(configurationCellID);
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

  // phase-space meshes bisect in all configuration directions together
  auto completeMask = [dim](const BoxMeshTopology* topology, unsigned mask)
  {
    if (topology->configurationTopology() == Teuchos::null) return mask;
    const unsigned configurationMask = (1u << topology->configurationDimensionCount()) - 1;
    return (mask & configurationMask) ? (mask | configurationMask) : mask;
  };

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
    for (auto &entry : fromMasks) entry.second = completeMask(fromTopology, entry.second);
    for (auto &entry : toMasks) entry.second = completeMask(toTopology, entry.second);
    meshFrom->hRefine(fromMasks);
    meshTo->hRefine(toMasks);
  }

  cellIDMap.clear();
  for (IndexType fromCellID : fromTopology->getLocallyKnownActiveCellIndices())
  {
    if (fromTopology->sideIsOnBoundary(fromCellID, lowerSide)) cellIDMap[fromCellID] = counterpart(fromCellID);
  }
  sideOrdinalMap.clear();
  sideOrdinalMap[int(lowerSide)] = toTopology->sideOrdinalForFace(2 * direction + 1);
}

// ---------------------------------------------------------------- multigrid

std::vector<MeshPtr> BoxGDA::meshesForMultigrid(MeshPtr fineMesh, int kCoarse, int delta_k)
{
  BoxMeshTopology* fineTopology = dynamic_cast<BoxMeshTopology*>(fineMesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(fineTopology == NULL, std::invalid_argument, "fineMesh must use a BoxMeshTopology");
  if (delta_k == -1) delta_k = fineMesh->globalDofAssignment()->getTestOrderEnrichment();
  const int dim = fineTopology->getDimension();
  const int depth = fineTopology->refinementDepth();
  BFPtr bf = fineMesh->bilinearForm();
  map<int,int> trialOrderEnhancements = fineMesh->getDofOrderingFactory().getTrialOrderEnhancements();

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
    meshesCoarseToFine.push_back(BoxGDA::mesh(levelTopology, bf, kCoarse + 1, delta_k, partitionPolicy, Teuchos::null,
                                              trialOrderEnhancements, map<int,int>()));
  }
  meshesCoarseToFine.push_back(fineMesh);
  return meshesCoarseToFine;
}

TMatrixPtr<double> BoxGDA::prolongationOperator(MeshPtr coarseMesh, TMapPtr coarseMap, TMapPtr fineMap)
{
  using GO = TMap::global_ordinal_type;
  BoxGDA* coarseGDA = dynamic_cast<BoxGDA*>(coarseMesh->globalDofAssignment().get());
  TEUCHOS_TEST_FOR_EXCEPTION(coarseGDA == NULL, std::invalid_argument, "the coarse mesh must also have a BoxGDA");
  BoxMeshTopology* coarseTopology = coarseGDA->_boxTopology;
  initializeVariables();
  coarseGDA->initializeVariables();
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
  ElementTypePtr coarseElemType = coarseGDA->elementType(*coarseTopology->getLocallyKnownActiveCellIndices().begin());
  BasisCachePtr coarseBasisCache = Teuchos::rcp(new BasisCache(coarseElemType));
  Teuchos::LAPACK<int,double> lapack;
  std::vector<double> phi, psi, weights, M, C;
  for (GlobalIndexType fineCellID : _partitions[rank])
  {
    const GlobalIndexType coarseCellID = coarseCellContaining(fineCellID);
    DofOrderingPtr fineOrdering = elementType(fineCellID)->trialOrderPtr;
    DofOrderingPtr coarseOrdering = coarseGDA->elementType(coarseCellID)->trialOrderPtr;
    const BoxMeshTopology::Extents coarseExtents = coarseTopology->cellExtents(coarseCellID);

    FC fineNodes = _boxTopology->physicalCellNodesForCell(fineCellID, true);
    _basisCache->setPhysicalCellNodes(fineNodes, {fineCellID}, false);
    const FC &points = _basisCache->getPhysicalCubaturePointsFC();
    const FC &weightedMeasures = _basisCache->getWeightedMeasures();
    const int P = points.dimension(1);
    weights.resize(P);
    for (int q=0; q<P; q++) weights[q] = weightedMeasures(0,q);
    FC coarseRefPoints(P, dim);
    for (int q=0; q<P; q++)
      for (int k=0; k<dim; k++)
        coarseRefPoints(q,k) = -1 + 2 * (points(0,q,k) - coarseExtents[k].first) / (coarseExtents[k].second - coarseExtents[k].first);
    FC coarseNodes = coarseTopology->physicalCellNodesForCell(coarseCellID, true);
    coarseBasisCache->setPhysicalCellNodes(coarseNodes, {coarseCellID}, false);
    coarseBasisCache->setRefCellPointsFC(coarseRefPoints);

    for (size_t fieldOrdinal=0; fieldOrdinal<_fieldVars.size(); fieldOrdinal++)
    {
      const int varID = _fieldVars[fieldOrdinal]->ID();
      BasisPtr fineBasis = fineOrdering->getBasis(varID), coarseBasis = coarseOrdering->getBasis(varID);
      const int n = fineBasis->getCardinality(), m = coarseBasis->getCardinality();
      auto fineValues = _basisCache->getTransformedValues(fineBasis, OP_VALUE);
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
      const GlobalIndexType fineOffset = _fieldDofOffsetForCell[fineCellID] + _fieldOffsets[fieldOrdinal];
      const GlobalIndexType coarseOffset = coarseGDA->_fieldDofOffsetForCell[coarseCellID] + coarseGDA->_fieldOffsets[fieldOrdinal];
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

    for (size_t fluxOrdinal=0; fluxOrdinal<_fluxVars.size(); fluxOrdinal++)
    {
      const int fineDegree = _fluxDegrees[fluxOrdinal], coarseDegree = coarseGDA->_fluxDegrees[fluxOrdinal];
      TEUCHOS_TEST_FOR_EXCEPTION(coarseDegree > fineDegree, std::invalid_argument, "the coarse mesh must not have a higher polynomial order than the fine mesh");
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
      const int tangentialCount = dim - 1, nf = fineDegree + 1, nc = coarseDegree + 1;
      int fineCardinality = 1, coarseCardinality = 1;
      for (int t=0; t<tangentialCount; t++) { fineCardinality *= nf; coarseCardinality *= nc; }
      const GlobalIndexType fineOffset = _fluxDofOffsetForPatch[patchOrdinal] + _fluxOffsets[fluxOrdinal];
      const GlobalIndexType coarseOffset = coarseGDA->_fluxDofOffsetForPatch[coarsePatchOrdinal] + coarseGDA->_fluxOffsets[fluxOrdinal];
      // multi-indices: the first tangential direction is the most significant digit (as in patchBasisValues())
      std::vector<int> fineIndex(tangentialCount), coarseIndex(tangentialCount);
      for (int i=0; i<fineCardinality; i++)
      {
        int remainder = i;
        for (int t=tangentialCount-1; t>=0; t--) { fineIndex[t] = remainder % nf; remainder /= nf; }
        for (int j=0; j<coarseCardinality; j++)
        {
          remainder = j;
          for (int t=tangentialCount-1; t>=0; t--) { coarseIndex[t] = remainder % nc; remainder /= nc; }
          double value = 1.0;
          for (int t=0; t<tangentialCount; t++) value *= R[t][fineIndex[t] * nf + coarseIndex[t]];
          addEntry(fineOffset + i, coarseOffset + j, value);
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
