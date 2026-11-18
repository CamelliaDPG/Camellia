//
//  MatrixFreeDPGOperator.cpp
//  Camellia
//

#include "MatrixFreeDPGOperator.h"

#include "Boundary.h"
#include "GlobalDofAssignment.h"
#include "Mesh.h"

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
