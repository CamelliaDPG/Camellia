//
//  MatrixFreeDPGOperator.cpp
//  Camellia
//

#include "MatrixFreeDPGOperator.h"

#include "Boundary.h"
#include "GlobalDofAssignment.h"
#include "Mesh.h"

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
  //! block Jacobi: y = beta y + alpha M^{-1} x, M block diagonal (Cholesky-factored blocks), the identity elsewhere
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
        for (size_t i=0; i<_inBlock.size(); i++) if (!_inBlock[i]) r(i,c) = x(i,c);
        std::vector<double> rhs;
        for (const Block &block : _blocks)
        {
          const int n = int(block.lids.size());
          rhs.resize(n);
          for (int i=0; i<n; i++) rhs[i] = x(block.lids[i],c);
          int info = 0;
          lapack.POTRS('L', n, 1, block.factor.data(), n, rhs.data(), n, &info);
          TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::logic_error, "BlockJacobiOperator: POTRS failed");
          for (int i=0; i<n; i++) r(block.lids[i],c) = rhs[i];
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
