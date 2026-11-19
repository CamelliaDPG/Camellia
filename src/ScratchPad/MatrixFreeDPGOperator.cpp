//
//  MatrixFreeDPGOperator.cpp
//  Camellia
//

#include "MatrixFreeDPGOperator.h"

#include "Boundary.h"
#include "GlobalDofAssignment.h"
#include "Mesh.h"

#include "Teuchos_BLAS.hpp"
#include "Teuchos_LAPACK.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

using namespace Camellia;

namespace
{
  using GO = TMap::global_ordinal_type;
  using LO = TMap::local_ordinal_type;
}

MatrixFreeDPGOperator::MatrixFreeDPGOperator(MeshPtr mesh, Teuchos::RCP<BoxCellDPGOperator> local, TMapConstPtr map,
                                             const std::set<GlobalIndexType> &dirichletDofs)
: _mesh(mesh), _local(local), _map(map)
{
  const std::set<GlobalIndexType> &cells = mesh->cellIDsInPartition();
  _cells.assign(cells.begin(), cells.end());
  GlobalDofAssignmentPtr gda = mesh->globalDofAssignment();
  std::set<GlobalIndexType> overlapGlobalIndices;
  for (GlobalIndexType cellID : _cells)
  {
    std::string reason;
    TEUCHOS_TEST_FOR_EXCEPTION(!local->supports(cellID, &reason), std::invalid_argument,
                               "MatrixFreeDPGOperator: cell " << cellID << " is not supported: " << reason);
    const std::set<GlobalIndexType> cellIndices = gda->globalDofIndicesForCell(cellID);
    overlapGlobalIndices.insert(cellIndices.begin(), cellIndices.end());
  }
  std::vector<GO> overlapGIDs(overlapGlobalIndices.begin(), overlapGlobalIndices.end());
  _overlapMap = Teuchos::rcp(new TMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), overlapGIDs.data(),
                                      overlapGIDs.size(), map->getIndexBase(), map->getComm()));
  _importer = Teuchos::rcp(new Tpetra::Import<>(_map, _overlapMap));
  _exporter = Teuchos::rcp(new Tpetra::Export<>(_overlapMap, _map));
  for (GlobalIndexType dof : dirichletDofs)
  {
    const LO lid = map->getLocalElement(static_cast<GO>(dof));
    TEUCHOS_TEST_FOR_EXCEPTION(lid == Teuchos::OrdinalTraits<LO>::invalid(), std::invalid_argument,
                               "MatrixFreeDPGOperator: Dirichlet dof " << dof << " is not owned by this rank");
    _dirichletLocal.push_back(lid);
  }
}

void MatrixFreeDPGOperator::dirichletData(MeshPtr mesh, BCPtr bc, std::vector<GlobalIndexType> &indices, std::vector<double> &values)
{
  Camellia::FieldContainer<GlobalIndexType> indicesFC;
  Camellia::FieldContainer<double> valuesFC;
  mesh->boundary().bcsToImpose(indicesFC, valuesFC, *bc, mesh->globalDofAssignment().get());
  indices.resize(indicesFC.size());
  values.resize(valuesFC.size());
  for (int i=0; i<indicesFC.size(); i++) indices[i] = indicesFC[i];
  for (int i=0; i<valuesFC.size(); i++) values[i] = valuesFC[i];
}

template <typename CellAction>
void MatrixFreeDPGOperator::assemble(const MV *X, MV &Y, CellAction cellAction) const
{
  const size_t vectorCount = Y.getNumVectors();
  GlobalDofAssignmentPtr gda = _mesh->globalDofAssignment();
  Teuchos::RCP<MV> Xoverlap;
  if (X != nullptr)
  {
    Xoverlap = Teuchos::rcp(new MV(_overlapMap, vectorCount));
    Xoverlap->doImport(*X, *_importer, Tpetra::INSERT);
  }
  MV Yoverlap(_overlapMap, vectorCount);
  Yoverlap.putScalar(0.0);
  for (size_t column=0; column<vectorCount; column++)
  {
    TVectorPtr<double> xColumn;
    if (X != nullptr) xColumn = Xoverlap->getVectorNonConst(column);
    for (GlobalIndexType cellID : _cells)
    {
      const int trialCount = _local->trialDofCount(cellID);
      Camellia::FieldContainer<double> x(trialCount), y(trialCount);
      if (X != nullptr) gda->interpretGlobalCoefficients(cellID, x, xColumn);
      cellAction(cellID, &x[0], &y[0]);
      Camellia::FieldContainer<double> globalData;
      Camellia::FieldContainer<GlobalIndexType> globalIndices;
      gda->interpretLocalData(cellID, y, globalData, globalIndices);
      for (int i=0; i<globalIndices.size(); i++) Yoverlap.sumIntoGlobalValue(static_cast<GO>(globalIndices[i]), column, globalData[i]);
    }
  }
  Y.putScalar(0.0);
  Y.doExport(Yoverlap, *_exporter, Tpetra::ADD);
}

void MatrixFreeDPGOperator::applyUnconstrained(const MV &X, MV &Y) const
{
  assemble(&X, Y, [this](GlobalIndexType cellID, const double *x, double *y) { _local->applySchur(cellID, x, y); });
}

void MatrixFreeDPGOperator::apply(const MV &X, MV &Y, Teuchos::ETransp /*mode: S is symmetric*/, double alpha, double beta) const
{
  const size_t vectorCount = X.getNumVectors();
  MV result(_map, vectorCount);
  if (_dirichletLocal.empty())
  {
    applyUnconstrained(X, result);
  }
  else
  {
    MV Xfree(X, Teuchos::Copy);
    {
      auto xView = Xfree.getLocalViewHost(Tpetra::Access::ReadWrite);
      for (LO lid : _dirichletLocal)
        for (size_t c=0; c<vectorCount; c++) xView(lid,c) = 0.0;
    }
    applyUnconstrained(Xfree, result);
    auto resultView = result.getLocalViewHost(Tpetra::Access::ReadWrite);
    auto xView = X.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (LO lid : _dirichletLocal)
      for (size_t c=0; c<vectorCount; c++) resultView(lid,c) = xView(lid,c);
  }
  Y.update(alpha, result, beta);
}

void MatrixFreeDPGOperator::load(RHSPtr rhs, MV &L) const
{
  if (rhs == Teuchos::null)
  {
    L.putScalar(0.0);
    return;
  }
  assemble(nullptr, L, [this, rhs](GlobalIndexType cellID, const double * /*x*/, double *y) { _local->applyLoad(cellID, rhs, y); });
}

void MatrixFreeDPGOperator::dirichletRightHandSide(RHSPtr rhs, const std::vector<GlobalIndexType> &dirichletIndices,
                                                   const std::vector<double> &dirichletValues, MV &b) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(dirichletIndices.size() != dirichletValues.size(), std::invalid_argument, "indices and values differ in number");
  const size_t vectorCount = b.getNumVectors();
  MV xD(_map, vectorCount), SxD(_map, vectorCount);
  xD.putScalar(0.0);
  for (size_t i=0; i<dirichletIndices.size(); i++)
    for (size_t c=0; c<vectorCount; c++) xD.replaceGlobalValue(static_cast<GO>(dirichletIndices[i]), c, dirichletValues[i]);
  applyUnconstrained(xD, SxD);
  load(rhs, b);
  b.update(-1.0, SxD, 1.0);
  auto bView = b.getLocalViewHost(Tpetra::Access::ReadWrite);
  auto xView = xD.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t i=0; i<dirichletIndices.size(); i++)
  {
    const LO lid = _map->getLocalElement(static_cast<GO>(dirichletIndices[i]));
    for (size_t c=0; c<vectorCount; c++) bView(lid,c) = xView(lid,c);
  }
}

void MatrixFreeDPGOperator::diagonal(MV &d) const
{
  GlobalDofAssignmentPtr gda = _mesh->globalDofAssignment();
  MV dOverlap(_overlapMap, 1);
  dOverlap.putScalar(0.0);
  for (GlobalIndexType cellID : _cells)
  {
    // P_K e_g for each global dof g touching the cell, from the interpretation of each local unit vector (row l of P_K)
    const int trialCount = _local->trialDofCount(cellID);
    std::map<GlobalIndexType, std::vector<std::pair<int,double>>> columns;
    Camellia::FieldContainer<double> unit(trialCount), globalData;
    Camellia::FieldContainer<GlobalIndexType> globalIndices;
    for (int l=0; l<trialCount; l++)
    {
      unit.initialize(0.0);
      unit[l] = 1.0;
      gda->interpretLocalData(cellID, unit, globalData, globalIndices);
      for (int i=0; i<globalIndices.size(); i++)
        if (globalData[i] != 0.0) columns[globalIndices[i]].push_back({l, globalData[i]});
    }
    std::vector<std::vector<std::pair<int,double>>> vectors;
    std::vector<GlobalIndexType> globals;
    for (auto &entry : columns)
    {
      globals.push_back(entry.first);
      vectors.push_back(std::move(entry.second));
    }
    std::vector<double> values(vectors.size());
    _local->schurQuadraticForms(cellID, vectors, values.data());
    for (size_t i=0; i<globals.size(); i++) dOverlap.sumIntoGlobalValue(static_cast<GO>(globals[i]), 0, values[i]);
  }
  d.putScalar(0.0);
  d.doExport(dOverlap, *_exporter, Tpetra::ADD);
  auto dView = d.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (LO lid : _dirichletLocal) dView(lid,0) = 1.0;
}

DiagonalScalingOperator::DiagonalScalingOperator(Teuchos::RCP<const MV> diagonal)
{
  TEUCHOS_TEST_FOR_EXCEPTION(diagonal->getNumVectors() != 1, std::invalid_argument, "the diagonal must have one column");
  _inverseDiagonal = Teuchos::rcp(new MV(diagonal->getMap(), 1));
  _inverseDiagonal->reciprocal(*diagonal);
}

void DiagonalScalingOperator::apply(const MV &X, MV &Y, Teuchos::ETransp /*mode*/, double alpha, double beta) const
{
  const size_t vectorCount = X.getNumVectors();
  auto inverse = _inverseDiagonal->getLocalViewHost(Tpetra::Access::ReadOnly);
  auto x = X.getLocalViewHost(Tpetra::Access::ReadOnly);
  auto y = Y.getLocalViewHost(Tpetra::Access::ReadWrite);
  const size_t localLength = X.getLocalLength();
  for (size_t c=0; c<vectorCount; c++)
    for (size_t i=0; i<localLength; i++)
      y(i,c) = ((beta == 0.0) ? 0.0 : beta * y(i,c)) + alpha * inverse(i,0) * x(i,c);
}

namespace
{
  //! block Jacobi / additive Schwarz: y = beta y + alpha sum_b R_b^T A_b^{-1} R_b x (Cholesky-factored blocks, which may
  //! overlap), the identity on dofs in no block
  class BlockJacobiOperator : public TOperator<double>
  {
  public:
    using MV = TVector<double>;
    struct Block
    {
      std::vector<LO> lids;
      std::vector<double> factor; // column-major Cholesky factor (lower)
    };
    BlockJacobiOperator(TMapConstPtr map, std::vector<Block> blocks) : _map(map), _blocks(std::move(blocks))
    {
      _inBlock.assign(map->getLocalNumElements(), false);
      for (const Block &block : _blocks) for (LO lid : block.lids) _inBlock[lid] = true;
    }
    Teuchos::RCP<const TMap> getDomainMap() const override { return _map; }
    Teuchos::RCP<const TMap> getRangeMap() const override { return _map; }
    bool hasTransposeApply() const override { return true; }
    void apply(const MV &X, MV &Y, Teuchos::ETransp, double alpha, double beta) const override
    {
      Teuchos::LAPACK<int,double> lapack;
      auto x = X.getLocalViewHost(Tpetra::Access::ReadOnly);
      MV result(_map, X.getNumVectors());
      auto r = result.getLocalViewHost(Tpetra::Access::ReadWrite);
      for (size_t c=0; c<X.getNumVectors(); c++)
      {
        for (size_t i=0; i<_inBlock.size(); i++) r(i,c) = _inBlock[i] ? 0.0 : x(i,c);
        std::vector<double> rhs;
        for (const Block &block : _blocks)
        {
          const int n = int(block.lids.size());
          rhs.resize(n);
          for (int i=0; i<n; i++) rhs[i] = x(block.lids[i],c);
          int info = 0;
          lapack.POTRS('L', n, 1, block.factor.data(), n, rhs.data(), n, &info);
          TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::logic_error, "BlockJacobiOperator: POTRS failed");
          for (int i=0; i<n; i++) r(block.lids[i],c) += rhs[i]; // additive (overlapping blocks sum)
        }
      }
      Y.update(alpha, result, beta);
    }
  private:
    TMapConstPtr _map;
    std::vector<Block> _blocks;
    std::vector<bool> _inBlock;
  };
}

Teuchos::RCP<TOperator<double>> MatrixFreeDPGOperator::traceBlockJacobi() const
{
  GlobalDofAssignmentPtr gda = _mesh->globalDofAssignment();
  std::set<LO> dirichlet(_dirichletLocal.begin(), _dirichletLocal.end());
  // block keys: an element's field dofs ({-1, cellID}), or a face's flux dofs (its centroid, rounded)
  std::map<std::vector<long long>, int> blockOfKey;
  std::map<LO, int> blockOfDof;
  std::vector<std::vector<LO>> blockDofs;
  std::vector<std::map<std::pair<LO,LO>, double>> blockEntries;
  for (GlobalIndexType cellID : _cells)
  {
    const int trialCount = _local->trialDofCount(cellID);
    std::map<GlobalIndexType, std::vector<std::pair<int,double>>> columns;
    Camellia::FieldContainer<double> unit(trialCount), globalData;
    Camellia::FieldContainer<GlobalIndexType> globalIndices;
    for (int l=0; l<trialCount; l++)
    {
      unit.initialize(0.0);
      unit[l] = 1.0;
      gda->interpretLocalData(cellID, unit, globalData, globalIndices);
      for (int i=0; i<globalIndices.size(); i++)
        if (globalData[i] != 0.0) columns[globalIndices[i]].push_back({l, globalData[i]});
    }
    // the cell's box, for face centroids
    Camellia::FieldContainer<double> nodes = _mesh->physicalCellNodesForCell(cellID); // (1, N, D)
    const int dim = nodes.dimension(2);
    std::vector<double> lo(dim), hi(dim);
    for (int k=0; k<dim; k++)
    {
      lo[k] = hi[k] = nodes(0,0,k);
      for (int n=1; n<nodes.dimension(1); n++) { lo[k] = std::min(lo[k], nodes(0,n,k)); hi[k] = std::max(hi[k], nodes(0,n,k)); }
    }
    double scale = 0;
    for (int k=0; k<dim; k++) scale = std::max(scale, hi[k] - lo[k]);
    std::vector<std::vector<std::pair<int,double>>> vectors;
    std::vector<int> block; // per vector
    std::vector<LO> lids;
    for (auto &entry : columns)
    {
      const LO lid = _map->getLocalElement(static_cast<GO>(entry.first));
      if ((lid == Teuchos::OrdinalTraits<LO>::invalid()) || (dirichlet.count(lid) > 0)) continue; // not owned, or Dirichlet
      auto found = blockOfDof.find(lid);
      int b;
      if (found != blockOfDof.end()) b = found->second;
      else
      {
        const int side = _local->dofSide(cellID, entry.second.front().first);
        std::vector<long long> key;
        if (side < 0) key = {-1, static_cast<long long>(cellID)};
        else
        {
          int normal, endpoint;
          _local->sideNormal(cellID, side, normal, endpoint);
          key.push_back(normal);
          for (int k=0; k<dim; k++)
          {
            const double x = (k == normal) ? (endpoint ? hi[k] : lo[k]) : 0.5 * (lo[k] + hi[k]);
            key.push_back(std::llround(x / scale * 1e9));
          }
        }
        auto keyFound = blockOfKey.find(key);
        if (keyFound != blockOfKey.end()) b = keyFound->second;
        else
        {
          b = int(blockDofs.size());
          blockOfKey[key] = b;
          blockDofs.push_back({});
          blockEntries.push_back({});
        }
        blockOfDof[lid] = b;
        blockDofs[b].push_back(lid);
      }
      vectors.push_back(entry.second);
      block.push_back(b);
      lids.push_back(lid);
    }
    std::vector<double> w;
    int stride = 0;
    _local->schurFactorCoordinates(cellID, vectors, w, stride);
    const int count = int(vectors.size());
    for (int a=0; a<count; a++)
      for (int c=0; c<count; c++)
      {
        if (block[a] != block[c]) continue;
        double value = 0;
        for (int i=0; i<stride; i++) value += w[size_t(a) * stride + i] * w[size_t(c) * stride + i];
        blockEntries[block[a]][{lids[a], lids[c]}] += value;
      }
  }
  // factor
  Teuchos::LAPACK<int,double> lapack;
  std::vector<BlockJacobiOperator::Block> blocks(blockDofs.size());
  for (size_t b=0; b<blockDofs.size(); b++)
  {
    const int n = int(blockDofs[b].size());
    std::map<LO,int> position;
    for (int i=0; i<n; i++) position[blockDofs[b][i]] = i;
    blocks[b].lids = blockDofs[b];
    blocks[b].factor.assign(size_t(n) * n, 0.0);
    for (const auto &entry : blockEntries[b])
      blocks[b].factor[size_t(position[entry.first.second]) * n + position[entry.first.first]] = entry.second;
    int info = 0;
    lapack.POTRF('L', n, blocks[b].factor.data(), n, &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::logic_error, "traceBlockJacobi: a diagonal block is not positive definite");
  }
  return Teuchos::rcp(new BlockJacobiOperator(_map, std::move(blocks)));
}

Camellia::ConjugateGradientResult Camellia::conjugateGradient(const TOperator<double> &A, const TOperator<double> *M, const TVector<double> &b,
                                                    TVector<double> &x, double tolerance, int maxIterations)
{
  using MV = TVector<double>;
  ConjugateGradientResult result;
  auto map = b.getMap();
  MV r(b, Teuchos::Copy), z(map, 1), p(map, 1), Ap(map, 1);
  x.putScalar(0.0);
  if (M == nullptr) z.assign(r); else M->apply(r, z);
  p.assign(z);
  double rz = r.getVector(0)->dot(*z.getVector(0));
  const double bNorm = b.getVector(0)->norm2();
  std::vector<double> alphas, betas;
  double residual = r.getVector(0)->norm2();
  while ((result.iterations < maxIterations) && (residual > tolerance * bNorm))
  {
    A.apply(p, Ap);
    const double alpha = rz / p.getVector(0)->dot(*Ap.getVector(0));
    x.update(alpha, p, 1.0);
    r.update(-alpha, Ap, 1.0);
    if (M == nullptr) z.assign(r); else M->apply(r, z);
    const double rzNew = r.getVector(0)->dot(*z.getVector(0));
    const double beta = rzNew / rz;
    p.update(1.0, z, beta);
    rz = rzNew;
    alphas.push_back(alpha);
    betas.push_back(beta);
    residual = r.getVector(0)->norm2();
    result.iterations++;
  }
  result.relativeResidual = residual / bNorm;
  // the Lanczos tridiagonal of the iteration: T_jj = 1/alpha_j + beta_{j-1}/alpha_{j-1}, T_j,j+1 = sqrt(beta_j)/alpha_j
  const int n = int(alphas.size());
  if (n > 0)
  {
    std::vector<double> d(n), e(std::max(n - 1, 1), 0.0), work(std::max(1, 2 * n - 2));
    for (int j=0; j<n; j++)
    {
      d[j] = 1.0 / alphas[j] + ((j > 0) ? betas[j-1] / alphas[j-1] : 0.0);
      if (j < n - 1) e[j] = std::sqrt(betas[j]) / alphas[j];
    }
    Teuchos::LAPACK<int,double> lapack;
    int info = 0;
    double dummy = 0;
    lapack.STEQR('N', n, d.data(), e.data(), &dummy, 1, work.data(), &info);
    if (info == 0)
    {
      result.lambdaMin = *std::min_element(d.begin(), d.end());
      result.lambdaMax = *std::max_element(d.begin(), d.end());
    }
  }
  return result;
}

TMatrixPtr<double> MatrixFreeDPGOperator::galerkinCoarseOperator(TMatrixPtr<double> P) const
{
  GlobalDofAssignmentPtr gda = _mesh->globalDofAssignment();
  std::set<LO> dirichlet(_dirichletLocal.begin(), _dirichletLocal.end());
  auto colMap = P->getColMap();
  // P's row of a fine dof: (coarse global index, value)
  auto prolongationRow = [&](LO fineLid, std::vector<std::pair<GO,double>> &row) {
    row.clear();
    typename TMatrix<double>::local_inds_host_view_type indices;
    typename TMatrix<double>::values_host_view_type values;
    P->getLocalRowView(fineLid, indices, values);
    for (size_t i=0; i<indices.extent(0); i++)
      if (values(i) != 0.0) row.push_back({colMap->getGlobalElement(indices(i)), values(i)});
  };
  std::map<GO, std::map<GO, double>> entries;
  std::vector<std::pair<GO,double>> row;
  for (GlobalIndexType cellID : _cells)
  {
    const int trialCount = _local->trialDofCount(cellID);
    // the cell's local coefficients of each coarse basis function: sum over fine dofs g of P_gc P_K e_g
    std::map<GO, std::map<int,double>> coarseColumns;
    Camellia::FieldContainer<double> unit(trialCount), globalData;
    Camellia::FieldContainer<GlobalIndexType> globalIndices;
    for (int l=0; l<trialCount; l++)
    {
      unit.initialize(0.0);
      unit[l] = 1.0;
      gda->interpretLocalData(cellID, unit, globalData, globalIndices);
      for (int i=0; i<globalIndices.size(); i++)
      {
        if (globalData[i] == 0.0) continue;
        const LO lid = _map->getLocalElement(static_cast<GO>(globalIndices[i]));
        TEUCHOS_TEST_FOR_EXCEPTION(lid == Teuchos::OrdinalTraits<LO>::invalid(), std::invalid_argument,
                                   "galerkinCoarseOperator: only single-rank (all dofs owned) is supported");
        if (dirichlet.count(lid) > 0) continue; // the Dirichlet dofs are masked out of the S part
        prolongationRow(lid, row);
        for (const auto &entry : row) coarseColumns[entry.first][l] += entry.second * globalData[i];
      }
    }
    std::vector<std::vector<std::pair<int,double>>> vectors;
    std::vector<GO> coarse;
    for (const auto &entry : coarseColumns)
    {
      std::vector<std::pair<int,double>> vector;
      for (const auto &localEntry : entry.second) if (localEntry.second != 0.0) vector.push_back(localEntry);
      if (vector.empty()) continue;
      coarse.push_back(entry.first);
      vectors.push_back(vector);
    }
    std::vector<double> w;
    int stride = 0;
    _local->schurFactorCoordinates(cellID, vectors, w, stride);
    for (size_t a=0; a<coarse.size(); a++)
      for (size_t b=0; b<coarse.size(); b++)
      {
        double value = 0;
        for (int i=0; i<stride; i++) value += w[a * stride + i] * w[b * stride + i];
        entries[coarse[a]][coarse[b]] += value;
      }
  }
  // the identity on the Dirichlet dofs: sum_g P_gc P_gc'
  for (LO lid : _dirichletLocal)
  {
    prolongationRow(lid, row);
    for (const auto &a : row)
      for (const auto &b : row) entries[a.first][b.first] += a.second * b.second;
  }
  auto coarseMap = P->getDomainMap();
  size_t maxRowLength = 1;
  for (const auto &entry : entries) maxRowLength = std::max(maxRowLength, entry.second.size());
  TMatrixPtr<double> Sc = Teuchos::rcp(new TMatrix<double>(coarseMap, maxRowLength));
  for (size_t i=0; i<coarseMap->getLocalNumElements(); i++)
  {
    const GO gid = coarseMap->getGlobalElement(i);
    std::vector<GO> columns;
    std::vector<double> values;
    auto found = entries.find(gid);
    if (found != entries.end())
      for (const auto &entry : found->second) { columns.push_back(entry.first); values.push_back(entry.second); }
    if (columns.empty()) { columns.push_back(gid); values.push_back(1.0); } // (a coarse dof seen by no cell)
    Sc->insertGlobalValues(gid, Teuchos::ArrayView<const GO>(columns), Teuchos::ArrayView<const double>(values));
  }
  Sc->fillComplete(coarseMap, coarseMap);
  return Sc;
}

void MatrixFreeDPGOperator::elementBlocks(std::vector<std::vector<LO>> &lids, std::vector<std::vector<double>> &factors,
                                          std::vector<std::vector<CellDof>> *cellDofs, const std::vector<int> *fluxOwner) const
{
  GlobalDofAssignmentPtr gda = _mesh->globalDofAssignment();
  std::set<LO> dirichlet(_dirichletLocal.begin(), _dirichletLocal.end());
  // pass 1: each cell's (non-Dirichlet, owned) dofs, as sparse local vectors
  std::vector<std::vector<LO>> blockDofs(_cells.size()), cellLids(_cells.size());
  std::vector<std::vector<std::vector<std::pair<int,double>>>> cellVectors(_cells.size());
  std::map<LO, std::vector<int>> blocksOfDof; // dof -> the cells (blocks) that contain it
  if (cellDofs != nullptr) cellDofs->assign(_cells.size(), {});
  for (size_t k=0; k<_cells.size(); k++)
  {
    const GlobalIndexType cellID = _cells[k];
    const int trialCount = _local->trialDofCount(cellID);
    std::map<LO, std::vector<std::pair<int,double>>> columns;
    Camellia::FieldContainer<double> unit(trialCount), globalData;
    Camellia::FieldContainer<GlobalIndexType> globalIndices;
    for (int l=0; l<trialCount; l++)
    {
      unit.initialize(0.0);
      unit[l] = 1.0;
      gda->interpretLocalData(cellID, unit, globalData, globalIndices);
      for (int i=0; i<globalIndices.size(); i++)
      {
        if (globalData[i] == 0.0) continue;
        const LO lid = _map->getLocalElement(static_cast<GO>(globalIndices[i]));
        if ((lid == Teuchos::OrdinalTraits<LO>::invalid()) || (dirichlet.count(lid) > 0)) continue;
        columns[lid].push_back({l, globalData[i]});
        if (cellDofs != nullptr) (*cellDofs)[k].push_back({l, lid, globalData[i]});
      }
    }
    for (auto &entry : columns)
    {
      cellLids[k].push_back(entry.first);
      cellVectors[k].push_back(std::move(entry.second));
      const int owner = (fluxOwner == nullptr) ? -1 : (*fluxOwner)[entry.first];
      if ((owner >= 0) && (owner != int(k))) continue; // a flux owned by another cell
      blockDofs[k].push_back(entry.first);
      blocksOfDof[entry.first].push_back(int(k));
    }
  }
  // pass 2: each cell's pair entries go to every block containing both dofs
  std::vector<std::map<std::pair<LO,LO>, double>> blockEntries(_cells.size());
  for (size_t k=0; k<_cells.size(); k++)
  {
    std::vector<double> w;
    int stride = 0;
    _local->schurFactorCoordinates(_cells[k], cellVectors[k], w, stride);
    const std::vector<LO> &dofs = cellLids[k];
    for (size_t a=0; a<dofs.size(); a++)
    {
      const std::vector<int> &blocksA = blocksOfDof[dofs[a]];
      for (size_t b=0; b<dofs.size(); b++)
      {
        double value = 0;
        for (int i=0; i<stride; i++) value += w[a * stride + i] * w[b * stride + i];
        const std::vector<int> &blocksB = blocksOfDof[dofs[b]];
        for (int block : blocksA)
          if (std::find(blocksB.begin(), blocksB.end(), block) != blocksB.end()) blockEntries[block][{dofs[a], dofs[b]}] += value;
      }
    }
  }
  Teuchos::LAPACK<int,double> lapack;
  lids = blockDofs;
  factors.assign(_cells.size(), {});
  for (size_t k=0; k<_cells.size(); k++)
  {
    const int n = int(blockDofs[k].size());
    std::map<LO,int> position;
    for (int i=0; i<n; i++) position[blockDofs[k][i]] = i;
    factors[k].assign(size_t(n) * n, 0.0);
    for (const auto &entry : blockEntries[k])
      factors[k][size_t(position[entry.first.second]) * n + position[entry.first.first]] = entry.second;
    blockEntries[k].clear();
    int info = 0;
    if (n > 0) lapack.POTRF('L', n, factors[k].data(), n, &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::logic_error, "elementBlocks: an element block is not positive definite");
  }
}

Teuchos::RCP<TOperator<double>> MatrixFreeDPGOperator::elementAdditiveSchwarz() const
{
  std::vector<std::vector<LO>> lids;
  std::vector<std::vector<double>> factors;
  elementBlocks(lids, factors, nullptr);
  std::vector<BlockJacobiOperator::Block> blocks(_cells.size());
  for (size_t k=0; k<_cells.size(); k++)
  {
    blocks[k].lids = std::move(lids[k]);
    blocks[k].factor = std::move(factors[k]);
  }
  return Teuchos::rcp(new BlockJacobiOperator(_map, std::move(blocks)));
}

Teuchos::RCP<ElementMultiplicativeSchwarz> MatrixFreeDPGOperator::elementMultiplicativeSchwarz(const std::vector<int> *fluxOwner) const
{
  std::vector<std::vector<LO>> lids;
  std::vector<std::vector<double>> factors;
  std::vector<std::vector<CellDof>> cellDofs;
  elementBlocks(lids, factors, &cellDofs, fluxOwner);
  // each cell's dense S_K in its local trial dofs, from the shared-eigenbasis coordinates of the unit vectors
  std::vector<ElementMultiplicativeSchwarz::CellMatrix> cells(_cells.size());
  Teuchos::BLAS<int,double> blas;
  for (size_t k=0; k<_cells.size(); k++)
  {
    ElementMultiplicativeSchwarz::CellMatrix &cell = cells[k];
    const int n = _local->trialDofCount(_cells[k]);
    cell.n = n;
    for (const CellDof &dof : cellDofs[k])
    {
      cell.local.push_back(dof.local);
      cell.lid.push_back(dof.lid);
      cell.coefficient.push_back(dof.coefficient);
    }
    std::vector<std::vector<std::pair<int,double>>> units(n);
    for (int l=0; l<n; l++) units[l] = {{l, 1.0}};
    std::vector<double> w;
    int stride = 0;
    _local->schurFactorCoordinates(_cells[k], units, w, stride); // w: n rows of length stride
    cell.S.assign(size_t(n) * n, 0.0);
    // S = W W^T with W (n x stride, row-major) = W^T (stride x n, column-major): S = (W^T)^T (W^T)
    blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS, n, n, stride, 1.0, w.data(), stride, w.data(), stride, 0.0, cell.S.data(), n);
  }
  return Teuchos::rcp(new ElementMultiplicativeSchwarz(_map, std::move(lids), std::move(factors), std::move(cells), _dirichletLocal));
}

void MatrixFreeDPGOperator::cellMatrices(std::vector<std::vector<CellDof>> &cellDofs, std::vector<std::vector<double>> &matrices) const
{
  GlobalDofAssignmentPtr gda = _mesh->globalDofAssignment();
  std::set<LO> dirichlet(_dirichletLocal.begin(), _dirichletLocal.end());
  Teuchos::BLAS<int,double> blas;
  cellDofs.assign(_cells.size(), {});
  matrices.assign(_cells.size(), {});
  for (size_t k=0; k<_cells.size(); k++)
  {
    const GlobalIndexType cellID = _cells[k];
    const int n = _local->trialDofCount(cellID);
    Camellia::FieldContainer<double> unit(n), globalData;
    Camellia::FieldContainer<GlobalIndexType> globalIndices;
    for (int l=0; l<n; l++)
    {
      unit.initialize(0.0);
      unit[l] = 1.0;
      gda->interpretLocalData(cellID, unit, globalData, globalIndices);
      for (int i=0; i<globalIndices.size(); i++)
      {
        if (globalData[i] == 0.0) continue;
        const LO lid = _map->getLocalElement(static_cast<GO>(globalIndices[i]));
        TEUCHOS_TEST_FOR_EXCEPTION(lid == Teuchos::OrdinalTraits<LO>::invalid(), std::invalid_argument, "cellMatrices: one rank only");
        if (dirichlet.count(lid) > 0) continue;
        cellDofs[k].push_back({l, lid, globalData[i]});
      }
    }
    std::vector<std::vector<std::pair<int,double>>> units(n);
    for (int l=0; l<n; l++) units[l] = {{l, 1.0}};
    std::vector<double> w;
    int stride = 0;
    _local->schurFactorCoordinates(cellID, units, w, stride);
    matrices[k].assign(size_t(n) * n, 0.0);
    blas.GEMM(Teuchos::TRANS, Teuchos::NO_TRANS, n, n, stride, 1.0, w.data(), stride, w.data(), stride, 0.0, matrices[k].data(), n);
  }
}

namespace
{
  //! P = R^{-1} W^{-1} R^{-T}: see MatrixFreeDPGOperator::transportSweep()
  class TransportSweepOperator : public TOperator<double>
  {
  public:
    using MV = TVector<double>;
    struct Cell
    {
      std::vector<LO> owned, inflow;  // map lids
      std::vector<double> factor;     // Cholesky factor (lower, column-major) of S_uu, in global coordinates
      std::vector<double> transport;  // T (owned x inflow, column-major), in global coordinates
    };
    struct FaceBlock
    {
      std::vector<LO> lids;
      std::vector<double> factor;
    };
    TransportSweepOperator(TMapConstPtr map, std::vector<Cell> cells, std::vector<int> order, std::vector<LO> dirichlet)
    : _map(map), _cells(std::move(cells)), _order(std::move(order)), _dirichlet(std::move(dirichlet)) {}
    Teuchos::RCP<const TMap> getDomainMap() const override { return _map; }
    Teuchos::RCP<const TMap> getRangeMap() const override { return _map; }
    bool hasTransposeApply() const override { return true; }
    void apply(const MV &X, MV &Y, Teuchos::ETransp, double alpha, double beta) const override
    {
      const size_t localCount = _map->getLocalNumElements();
      MV result(_map, X.getNumVectors());
      auto xView = X.getLocalViewHost(Tpetra::Access::ReadOnly);
      std::vector<double> r(localCount), z(localCount);
      for (size_t column=0; column<X.getNumVectors(); column++)
      {
        for (size_t i=0; i<localCount; i++) r[i] = xView(i,column);
        std::fill(z.begin(), z.end(), 0.0);
        if (!_multiplicative || _faces.empty())
        {
          sweep(r, z);
          faceSolve(r, z);
        }
        else
        {
          // symmetric product: face blocks, sweep, face blocks, with residual updates by the operator
          faceSolve(r, z);
          std::vector<double> residual = residualOf(r, z);
          sweep(residual, z);
          residual = residualOf(r, z);
          faceSolve(residual, z);
        }
        for (LO lid : _dirichlet) z[lid] = r[lid];
        auto resultView = result.getLocalViewHost(Tpetra::Access::ReadWrite);
        for (size_t i=0; i<localCount; i++) resultView(i,column) = z[i];
      }
      Y.update(alpha, result, beta);
    }
    //! [cell][inflow entry]: whether its owner comes later in the order (a lagged face), to be dropped from R
    void setLagged(std::vector<std::vector<bool>> lagged) { _lagged = std::move(lagged); }
    void setFaces(std::vector<FaceBlock> faces, const TOperator<double> *op, bool multiplicative)
    {
      _faces = std::move(faces); _op = op; _multiplicative = multiplicative;
    }
  private:
    TMapConstPtr _map;
    std::vector<Cell> _cells;
    std::vector<int> _order;
    std::vector<LO> _dirichlet;
    std::vector<std::vector<bool>> _lagged;
    std::vector<FaceBlock> _faces;
    const TOperator<double> *_op = nullptr;
    bool _multiplicative = false;

    //! z += P r on the sweep dofs (r is not changed)
    void sweep(const std::vector<double> &r, std::vector<double> &z) const
    {
      Teuchos::BLAS<int,double> blas;
      Teuchos::LAPACK<int,double> lapack;
      std::vector<double> y(r), x(r.size(), 0.0), a, b;
      // backward sweep, R^T y = r: y_K = r_K + sum over downwind K' of (T_K'^T y_K')|_K
      for (auto it = _order.rbegin(); it != _order.rend(); ++it)
      {
        const Cell &cell = _cells[*it];
        const int nu = int(cell.owned.size()), ni = int(cell.inflow.size());
        if ((nu == 0) || (ni == 0)) continue;
        a.resize(nu); b.assign(ni, 0.0);
        for (int i=0; i<nu; i++) a[i] = y[cell.owned[i]];
        blas.GEMV(Teuchos::TRANS, nu, ni, 1.0, cell.transport.data(), nu, a.data(), 1, 0.0, b.data(), 1);
        for (int i=0; i<ni; i++) if (_lagged.empty() || !_lagged[*it][i]) y[cell.inflow[i]] += b[i];
      }
      // cell solves z_K = S_uu^{-1} y_K, then the forward sweep x_K = z_K + T_K x_{i,K}
      for (int k : _order)
      {
        const Cell &cell = _cells[k];
        const int nu = int(cell.owned.size()), ni = int(cell.inflow.size());
        if (nu == 0) continue;
        a.resize(nu);
        for (int i=0; i<nu; i++) a[i] = y[cell.owned[i]];
        int info = 0;
        lapack.POTRS('L', nu, 1, cell.factor.data(), nu, a.data(), nu, &info);
        if (ni > 0)
        {
          b.resize(ni);
          for (int i=0; i<ni; i++) b[i] = (_lagged.empty() || !_lagged[k][i]) ? x[cell.inflow[i]] : 0.0;
          blas.GEMV(Teuchos::NO_TRANS, nu, ni, 1.0, cell.transport.data(), nu, b.data(), 1, 1.0, a.data(), 1);
        }
        for (int i=0; i<nu; i++) x[cell.owned[i]] = a[i];
      }
      for (const Cell &cell : _cells) for (LO lid : cell.owned) z[lid] += x[lid];
    }
    //! z += D_c^{-1} r on the characteristic-face dofs
    void faceSolve(const std::vector<double> &r, std::vector<double> &z) const
    {
      Teuchos::LAPACK<int,double> lapack;
      std::vector<double> a;
      for (const FaceBlock &face : _faces)
      {
        const int n = int(face.lids.size());
        a.resize(n);
        for (int i=0; i<n; i++) a[i] = r[face.lids[i]];
        int info = 0;
        lapack.POTRS('L', n, 1, face.factor.data(), n, a.data(), n, &info);
        for (int i=0; i<n; i++) z[face.lids[i]] += a[i];
      }
    }
    //! r - S z (Dirichlet rows: zero, since z is zero there)
    std::vector<double> residualOf(const std::vector<double> &r, const std::vector<double> &z) const
    {
      MV zv(_map, 1), Sz(_map, 1);
      {
        auto view = zv.getLocalViewHost(Tpetra::Access::ReadWrite);
        for (size_t i=0; i<z.size(); i++) view(i,0) = z[i];
        for (LO lid : _dirichlet) view(lid,0) = 0.0;
      }
      _op->apply(zv, Sz);
      auto view = Sz.getLocalViewHost(Tpetra::Access::ReadOnly);
      std::vector<double> residual(r.size());
      for (size_t i=0; i<r.size(); i++) residual[i] = r[i] - view(i,0);
      for (LO lid : _dirichlet) residual[lid] = 0.0;
      return residual;
    }
  };
}

Teuchos::RCP<TOperator<double>> MatrixFreeDPGOperator::transportSweep(const std::vector<int> &order, const std::vector<int> &fluxOwner,
                                                                      const std::vector<int> *characteristicFace,
                                                                      bool characteristicMultiplicative,
                                                                      const std::vector<int> *blockOf) const
{
  std::vector<std::vector<CellDof>> cellDofs;
  std::vector<std::vector<double>> matrices;
  cellMatrices(cellDofs, matrices);
  const int cellCount = int(_cells.size());
  // the blocks, and their sweep order
  std::vector<int> block(cellCount);
  int blockCount = 0;
  std::vector<int> blockOrder;
  if (blockOf != nullptr)
  {
    block = *blockOf;
    for (int b : block) blockCount = std::max(blockCount, b + 1);
    for (int b=0; b<blockCount; b++) blockOrder.push_back(b);
  }
  else
  {
    for (int k=0; k<cellCount; k++) block[k] = k;
    blockCount = cellCount;
    blockOrder = order;
  }
  std::vector<int> position(blockCount);
  for (int i=0; i<blockCount; i++) position[blockOrder[i]] = i;
  std::vector<std::vector<int>> blockCells(blockCount);
  for (int k=0; k<cellCount; k++) blockCells[block[k]].push_back(k);

  Teuchos::LAPACK<int,double> lapack;
  std::set<LO> dirichlet(_dirichletLocal.begin(), _dirichletLocal.end());
  // the characteristic faces' blocks
  std::map<int, std::vector<LO>> faceLids;
  if (characteristicFace != nullptr)
    for (size_t lid=0; lid<characteristicFace->size(); lid++)
      if (((*characteristicFace)[lid] >= 0) && (dirichlet.count(LO(lid)) == 0)) faceLids[(*characteristicFace)[lid]].push_back(LO(lid));
  std::map<int, std::vector<double>> faceMatrices;
  for (const auto &entry : faceLids) faceMatrices[entry.first].assign(entry.second.size() * entry.second.size(), 0.0);

  std::vector<TransportSweepOperator::Cell> blocks(blockCount);
  std::vector<std::vector<bool>> lagged(blockCount);
  bool anyLagged = false;
  for (int b=0; b<blockCount; b++)
  {
    TransportSweepOperator::Cell &sweepBlock = blocks[b];
    // owned (u) and inflow (i) lids of the block
    std::map<LO,int> ownedPosition, inflowPosition;
    for (int k : blockCells[b])
      for (const CellDof &dof : cellDofs[k])
      {
        const int owner = fluxOwner[dof.lid];
        if (owner == -2) continue;
        if ((owner == -1) || (block[owner] == b))
        {
          if (ownedPosition.insert({dof.lid, int(sweepBlock.owned.size())}).second) sweepBlock.owned.push_back(dof.lid);
        }
        else if (inflowPosition.insert({dof.lid, int(sweepBlock.inflow.size())}).second)
        {
          sweepBlock.inflow.push_back(dof.lid);
          const bool isLagged = position[block[owner]] > position[b];
          lagged[b].push_back(isLagged);
          anyLagged = anyLagged || isLagged;
        }
      }
    const int nu = int(sweepBlock.owned.size()), ni = int(sweepBlock.inflow.size());
    std::vector<double> Suu(size_t(nu) * nu, 0.0), Sui(size_t(nu) * std::max(ni,1), 0.0);
    // assemble the block's S_uu and S_ui in global coordinates (x_local = c x_global: S^g_ab = c_a S_ab c_b)
    for (int k : blockCells[b])
    {
      const int n = _local->trialDofCount(_cells[k]);
      const std::vector<double> &S = matrices[k];
      std::vector<int> seen(n, 0);
      for (const CellDof &dof : cellDofs[k])
        TEUCHOS_TEST_FOR_EXCEPTION(seen[dof.local]++ > 0, std::invalid_argument, "transportSweep: a local dof with several global dofs (constraints) is not supported");
      for (const CellDof &a : cellDofs[k])
      {
        auto ua = ownedPosition.find(a.lid);
        if (ua == ownedPosition.end()) continue;
        for (const CellDof &c : cellDofs[k])
        {
          const double value = a.coefficient * S[size_t(c.local) * n + a.local] * c.coefficient;
          auto uc = ownedPosition.find(c.lid);
          if (uc != ownedPosition.end()) { Suu[size_t(uc->second) * nu + ua->second] += value; continue; }
          auto ic = inflowPosition.find(c.lid);
          if (ic != inflowPosition.end()) Sui[size_t(ic->second) * nu + ua->second] += value;
        }
      }
      // the characteristic-face blocks: this cell's contributions
      if (characteristicFace != nullptr)
        for (const CellDof &a : cellDofs[k])
        {
          if (fluxOwner[a.lid] != -2) continue;
          const int face = (*characteristicFace)[a.lid];
          const std::vector<LO> &lids = faceLids[face];
          const size_t m = lids.size();
          const size_t pa = std::find(lids.begin(), lids.end(), a.lid) - lids.begin();
          for (const CellDof &c : cellDofs[k])
          {
            if ((fluxOwner[c.lid] != -2) || ((*characteristicFace)[c.lid] != face)) continue;
            const size_t pc = std::find(lids.begin(), lids.end(), c.lid) - lids.begin();
            faceMatrices[face][pc * m + pa] += a.coefficient * S[size_t(c.local) * n + a.local] * c.coefficient;
          }
        }
    }
    int info = 0;
    if (nu > 0) lapack.POTRF('L', nu, Suu.data(), nu, &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::logic_error, "transportSweep: a block's owned part S_uu is not positive definite");
    if ((nu > 0) && (ni > 0))
    {
      lapack.POTRS('L', nu, ni, Suu.data(), nu, Sui.data(), nu, &info);
      for (double &value : Sui) value = -value;
      sweepBlock.transport = std::move(Sui);
    }
    sweepBlock.factor = std::move(Suu);
  }
  matrices.clear();
  auto sweep = Teuchos::rcp(new TransportSweepOperator(_map, std::move(blocks), blockOrder, _dirichletLocal));
  if (anyLagged) sweep->setLagged(std::move(lagged));
  if (characteristicFace != nullptr)
  {
    std::vector<TransportSweepOperator::FaceBlock> faces;
    for (auto &entry : faceLids)
    {
      TransportSweepOperator::FaceBlock face;
      face.lids = entry.second;
      face.factor = std::move(faceMatrices[entry.first]);
      const int m = int(face.lids.size());
      int info = 0;
      lapack.POTRF('L', m, face.factor.data(), m, &info);
      TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::logic_error, "transportSweep: a characteristic face block is not positive definite");
      faces.push_back(std::move(face));
    }
    sweep->setFaces(std::move(faces), this, characteristicMultiplicative);
  }
  return sweep;
}

std::vector<int> MatrixFreeDPGOperator::upwindCellOrder(int *cyclesBroken, int *laggedFaces, std::vector<int> *fluxOwner,
                                                       std::vector<int> *characteristicFace, double characteristicTolerance,
                                                       std::vector<int> *component, bool weightedCycleBreaking,
                                                       double *laggedWeightFraction) const
{
  GlobalDofAssignmentPtr gda = _mesh->globalDofAssignment();
  const int cellCount = int(_cells.size());
  // the cells' average beta_k (the negated averages of the coefficients against d_k v)
  std::vector<std::vector<double>> beta(cellCount);
  for (int k=0; k<cellCount; k++)
  {
    beta[k] = _local->testDerivativeCoefficientAverages(_cells[k]);
    for (double &b : beta[k]) b = -b;
  }
  // neighbors through shared flux dofs: lid -> (cell, side)
  std::map<LO, std::vector<std::pair<int,int>>> fluxDofCells;
  for (int k=0; k<cellCount; k++)
  {
    const GlobalIndexType cellID = _cells[k];
    const int trialCount = _local->trialDofCount(cellID);
    Camellia::FieldContainer<double> unit(trialCount), globalData;
    Camellia::FieldContainer<GlobalIndexType> globalIndices;
    for (int l=0; l<trialCount; l++)
    {
      const int side = _local->dofSide(cellID, l);
      if (side < 0) continue;
      unit.initialize(0.0);
      unit[l] = 1.0;
      gda->interpretLocalData(cellID, unit, globalData, globalIndices);
      for (int i=0; i<globalIndices.size(); i++)
      {
        if (globalData[i] == 0.0) continue;
        const LO lid = _map->getLocalElement(static_cast<GO>(globalIndices[i]));
        if (lid == Teuchos::OrdinalTraits<LO>::invalid()) continue;
        auto &entries = fluxDofCells[lid];
        if (std::find(entries.begin(), entries.end(), std::make_pair(k, side)) == entries.end()) entries.push_back({k, side});
      }
    }
  }
  // directed edges upwind -> downwind, one per neighboring pair
  std::set<std::pair<int,int>> edges;
  std::map<std::pair<int,int>, double> edgeWeight; // |beta . n| / h_n of the downstream cell
  std::set<std::pair<int,int>> pairs;
  std::vector<std::vector<double>> widths(cellCount);
  for (int k=0; k<cellCount; k++)
  {
    Camellia::FieldContainer<double> nodes = _mesh->physicalCellNodesForCell(_cells[k]);
    const int dim = nodes.dimension(2);
    widths[k].assign(dim, 0.0);
    for (int d=0; d<dim; d++)
    {
      double lo = nodes(0,0,d), hi = nodes(0,0,d);
      for (int node=1; node<nodes.dimension(1); node++) { lo = std::min(lo, nodes(0,node,d)); hi = std::max(hi, nodes(0,node,d)); }
      widths[k][d] = hi - lo;
    }
  }
  if (fluxOwner != nullptr) fluxOwner->assign(_map->getLocalNumElements(), -1);
  if (characteristicFace != nullptr) characteristicFace->assign(_map->getLocalNumElements(), -1);
  std::map<std::pair<int,int>, int> faceIDs; // (cell, -1 - side) for a boundary face, (k1, k2) for an interior one
  auto betaScale = [&](int k) { double scale = 0; for (double b : beta[k]) scale = std::max(scale, std::abs(b)); return scale; };
  auto markCharacteristic = [&](LO lid, std::pair<int,int> key) {
    if (characteristicFace == nullptr) return;
    auto found = faceIDs.find(key);
    const int id = (found == faceIDs.end()) ? (faceIDs[key] = int(faceIDs.size())) : found->second;
    (*characteristicFace)[lid] = id;
    if (fluxOwner != nullptr) (*fluxOwner)[lid] = -2;
  };
  for (const auto &entry : fluxDofCells)
  {
    const auto &cellSides = entry.second;
    if (fluxOwner != nullptr) (*fluxOwner)[entry.first] = cellSides[0].first; // a boundary face: its only cell
    if ((cellSides.size() == 1) && (characteristicFace != nullptr))
    {
      const int k = cellSides[0].first;
      int normal, endpoint;
      _local->sideNormal(_cells[k], cellSides[0].second, normal, endpoint);
      if (std::abs(beta[k][normal]) <= characteristicTolerance * betaScale(k)) markCharacteristic(entry.first, {k, -1 - cellSides[0].second});
    }
    for (size_t a=0; a<cellSides.size(); a++)
      for (size_t b=a+1; b<cellSides.size(); b++)
      {
        int k1 = cellSides[a].first, k2 = cellSides[b].first;
        if (k1 == k2) continue;
        int normal, endpoint;
        _local->sideNormal(_cells[k1], cellSides[a].second, normal, endpoint);
        // k1's side is its upper face (endpoint 1) in direction normal: the face normal from k1 to k2 is +e_normal
        const double betaN = (beta[k1][normal] + beta[k2][normal]) * ((endpoint == 1) ? 1.0 : -1.0); // beta . n_{k1 -> k2}
        if ((characteristicFace != nullptr) && (0.5 * std::abs(betaN) <= characteristicTolerance * std::max(betaScale(k1), betaScale(k2))))
        {
          markCharacteristic(entry.first, {std::min(k1,k2), std::max(k1,k2)});
          continue; // no ownership and no dependency: the face is in neither cell's sweep split
        }
        if (fluxOwner != nullptr) (*fluxOwner)[entry.first] = (betaN >= 0) ? k1 : k2;
        if (!pairs.insert({std::min(k1,k2), std::max(k1,k2)}).second) continue;
        const std::pair<int,int> edge = (betaN >= 0) ? std::make_pair(k1, k2) : std::make_pair(k2, k1);
        edges.insert(edge);
        edgeWeight[edge] = 0.5 * std::abs(betaN) / widths[edge.second][normal];
      }
  }
  std::vector<std::vector<int>> downwind(cellCount);
  std::vector<int> inDegree(cellCount, 0);
  for (const auto &edge : edges) { downwind[edge.first].push_back(edge.second); inDegree[edge.second]++; }
  if (component != nullptr)
  {
    // Tarjan's strongly connected components (iterative); Tarjan finds them in reverse topological order
    component->assign(cellCount, -1);
    std::vector<int> index(cellCount, -1), low(cellCount, 0), stack;
    std::vector<bool> onStack(cellCount, false);
    int counter = 0, components = 0;
    for (int root=0; root<cellCount; root++)
    {
      if (index[root] >= 0) continue;
      std::vector<std::pair<int,size_t>> call{{root, 0}};
      index[root] = low[root] = counter++; stack.push_back(root); onStack[root] = true;
      while (!call.empty())
      {
        int v = call.back().first;
        size_t &next = call.back().second;
        if (next < downwind[v].size())
        {
          const int w = downwind[v][next++];
          if (index[w] < 0)
          {
            index[w] = low[w] = counter++; stack.push_back(w); onStack[w] = true;
            call.push_back({w, 0});
          }
          else if (onStack[w]) low[v] = std::min(low[v], index[w]);
        }
        else
        {
          if (low[v] == index[v])
          {
            int w;
            do { w = stack.back(); stack.pop_back(); onStack[w] = false; (*component)[w] = components; } while (w != v);
            components++;
          }
          call.pop_back();
          if (!call.empty()) low[call.back().first] = std::min(low[call.back().first], low[v]);
        }
      }
    }
    for (int &c : *component) c = components - 1 - c; // upwind first
  }
  // Kahn's algorithm; on a cycle, take the remaining cell with the fewest unvisited upwind neighbors (or the least
  // unvisited in-edge weight)
  std::vector<double> inWeight(cellCount, 0.0);
  double totalWeight = 0, laggedWeight = 0;
  for (const auto &entry : edgeWeight) { inWeight[entry.first.second] += entry.second; totalWeight += entry.second; }
  auto key = [&](int k) { return std::make_pair(inDegree[k] > 0 ? (weightedCycleBreaking ? inWeight[k] : double(inDegree[k])) : -1.0, k); };
  std::set<std::pair<double,int>> remaining; // (0 if ready, else the cycle-breaking key; cell)
  for (int k=0; k<cellCount; k++) remaining.insert(key(k));
  std::vector<int> order;
  std::vector<bool> visited(cellCount, false);
  int broken = 0, lagged = 0;
  while (!remaining.empty())
  {
    auto first = remaining.begin();
    const int k = first->second;
    if (inDegree[k] > 0) { broken++; lagged += inDegree[k]; laggedWeight += inWeight[k]; }
    remaining.erase(first);
    visited[k] = true;
    order.push_back(k);
    for (int next : downwind[k])
    {
      if (visited[next]) continue;
      remaining.erase(key(next));
      inDegree[next]--;
      inWeight[next] -= edgeWeight[{k, next}];
      remaining.insert(key(next));
    }
  }
  if (laggedWeightFraction != nullptr) *laggedWeightFraction = (totalWeight > 0) ? laggedWeight / totalWeight : 0.0;
  if (cyclesBroken != nullptr) *cyclesBroken = broken;
  if (laggedFaces != nullptr) *laggedFaces = lagged;
  return order;
}

ElementMultiplicativeSchwarz::ElementMultiplicativeSchwarz(TMapConstPtr map, std::vector<std::vector<LO>> blockLids,
                                                           std::vector<std::vector<double>> blockFactors,
                                                           std::vector<CellMatrix> cells, std::vector<LO> dirichletLids)
: _map(map), _blockLids(std::move(blockLids)), _blockFactors(std::move(blockFactors)), _cells(std::move(cells)),
  _dirichletLids(std::move(dirichletLids))
{
  std::map<LO, std::vector<int>> cellsOfDof;
  for (int c=0; c<int(_cells.size()); c++)
    for (LO lid : _cells[c].lid)
    {
      auto &list = cellsOfDof[lid];
      if (list.empty() || (list.back() != c)) list.push_back(c);
    }
  _touching.resize(_blockLids.size());
  for (size_t b=0; b<_blockLids.size(); b++)
  {
    std::set<int> touching;
    for (LO lid : _blockLids[b]) for (int c : cellsOfDof[lid]) touching.insert(c);
    _touching[b].assign(touching.begin(), touching.end());
  }
  _order.resize(_blockLids.size());
  for (size_t b=0; b<_order.size(); b++) _order[b] = int(b);
}

void ElementMultiplicativeSchwarz::correct(int block, std::vector<double> &r, std::vector<double> &z, std::vector<double> &scratch) const
{
  Teuchos::LAPACK<int,double> lapack;
  Teuchos::BLAS<int,double> blas;
  const std::vector<LO> &lids = _blockLids[block];
  const int n = int(lids.size());
  if (n == 0) return;
  std::vector<double> delta(n);
  for (int i=0; i<n; i++) delta[i] = r[lids[i]];
  int info = 0;
  lapack.POTRS('L', n, 1, _blockFactors[block].data(), n, delta.data(), n, &info);
  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::logic_error, "ElementMultiplicativeSchwarz: POTRS failed");
  for (int i=0; i<n; i++) { z[lids[i]] += delta[i]; scratch[lids[i]] = delta[i]; }
  // r -= S R^T delta, cell by cell
  std::vector<double> x, y;
  for (int c : _touching[block])
  {
    const CellMatrix &cell = _cells[c];
    x.assign(cell.n, 0.0);
    bool any = false;
    for (size_t e=0; e<cell.lid.size(); e++)
    {
      const double value = scratch[cell.lid[e]];
      if (value != 0.0) { x[cell.local[e]] += cell.coefficient[e] * value; any = true; }
    }
    if (!any) continue;
    y.assign(cell.n, 0.0);
    blas.GEMV(Teuchos::NO_TRANS, cell.n, cell.n, 1.0, cell.S.data(), cell.n, x.data(), 1, 0.0, y.data(), 1);
    for (size_t e=0; e<cell.lid.size(); e++) r[cell.lid[e]] -= cell.coefficient[e] * y[cell.local[e]];
  }
  for (int i=0; i<n; i++) scratch[lids[i]] = 0.0;
}

void ElementMultiplicativeSchwarz::apply(const MV &X, MV &Y, Teuchos::ETransp /*mode*/, double alpha, double beta) const
{
  Teuchos::LAPACK<int,double> lapack;
  const size_t localCount = _map->getLocalNumElements();
  MV result(_map, X.getNumVectors());
  auto xView = X.getLocalViewHost(Tpetra::Access::ReadOnly);
  auto resultView = result.getLocalViewHost(Tpetra::Access::ReadWrite);
  std::vector<bool> isDirichlet(localCount, false);
  for (LO lid : _dirichletLids) isDirichlet[lid] = true;
  for (size_t column=0; column<X.getNumVectors(); column++)
  {
    std::vector<double> r(localCount), z(localCount, 0.0), scratch(localCount, 0.0);
    for (size_t i=0; i<localCount; i++) r[i] = xView(i,column);
    for (LO lid : _dirichletLids) { z[lid] = r[lid]; r[lid] = 0.0; } // identity rows and columns
    if (_additive)
    {
      for (size_t b=0; b<_blockLids.size(); b++)
      {
        const std::vector<LO> &lids = _blockLids[b];
        const int n = int(lids.size());
        if (n == 0) continue;
        std::vector<double> delta(n);
        for (int i=0; i<n; i++) delta[i] = r[lids[i]];
        int info = 0;
        lapack.POTRS('L', n, 1, _blockFactors[b].data(), n, delta.data(), n, &info);
        for (int i=0; i<n; i++) z[lids[i]] += delta[i];
      }
    }
    else
    {
      for (int b : _order) correct(b, r, z, scratch);
      if (!_forwardOnly)
        for (auto it = _order.rbegin(); it != _order.rend(); ++it) correct(*it, r, z, scratch);
    }
    for (size_t i=0; i<localCount; i++) resultView(i,column) = z[i];
  }
  Y.update(alpha, result, beta);
}
