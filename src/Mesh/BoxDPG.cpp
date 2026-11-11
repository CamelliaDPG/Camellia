//
//  BoxDPG.cpp
//  Camellia
//
//  PROTOTYPE; see BoxDPG.h.
//

#include "BoxDPG.h"

#include "CamelliaCellTools.h"
#include "DofOrderingFactory.h"
#include "VarFactory.h"

#include <Amesos2.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include <cmath>

using namespace Camellia;
using namespace std;

namespace
{
  using FC = Camellia::FieldContainer<double>;

  class BoxPointFunction : public TFunction<double>
  {
    BoxDPG::PointFunction _f;
  public:
    BoxPointFunction(BoxDPG::PointFunction f) : TFunction<double>(0), _f(f) {}
    using TFunction<double>::values;
    void values(FC &values, BasisCachePtr basisCache) override
    {
      const FC &points = basisCache->getPhysicalCubaturePointsFC();
      const int numCells = points.dimension(0), numPoints = points.dimension(1);
      for (int c=0; c<numCells; c++)
        for (int p=0; p<numPoints; p++)
          values(c,p) = _f(&points(c,p,0));
    }
  };

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

FunctionPtr BoxDPG::function(PointFunction f)
{
  return Teuchos::rcp(new BoxPointFunction(f));
}

BoxDPG::BoxDPG(const vector<vector<double>> &rootVertices, BFPtr bf, int H1Order, int delta_k)
: _forest([&]{ vector<int> counts; for (const auto &v : rootVertices) counts.push_back(int(v.size()) - 1); return counts; }()),
  _rootVertices(rootVertices), _dim(int(rootVertices.size())), _bf(bf)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_dim < 2, std::invalid_argument, "BoxDPG requires dimension >= 2");
  VarFactoryPtr vf = bf->varFactory();
  TEUCHOS_TEST_FOR_EXCEPTION(vf->traceVars().size() > 0, std::invalid_argument,
                             "BoxDPG supports field and flux trial variables only (no continuous traces)");
  _fieldVars = vf->fieldVars();
  _fluxVars = vf->fluxVars();

  if (_dim == 2) _cellTopo = CellTopology::quad();
  else if (_dim == 3) _cellTopo = CellTopology::hexahedron();
  else _cellTopo = CellTopology::cellTopology(CellTopology::line(), _dim - 1);

  DofOrderingFactory factory(bf);
  vector<int> trialOrder = {H1Order}, testOrder = {H1Order + delta_k};
  const bool conformingVertices = false;
  DofOrderingPtr trialOrdering = factory.trialOrdering(trialOrder, _cellTopo, conformingVertices);
  DofOrderingPtr testOrdering = factory.testOrdering(testOrder, _cellTopo);
  _elemType = Teuchos::rcp(new ElementType(trialOrdering, testOrdering, _cellTopo));
  _basisCache = Teuchos::rcp(new BasisCache(_elemType));
  const bool testVsTest = true;
  _ipBasisCache = Teuchos::rcp(new BasisCache(_elemType, MeshPtr(Teuchos::null), testVsTest));

  // match the topology's side ordinals with (direction, lower/upper)
  _refNodes.resize(_cellTopo->getNodeCount(), _dim);
  CamelliaCellTools::refCellNodesForTopology(_refNodes, _cellTopo);
  _sideForFace.assign(2 * _dim, -1);
  for (int side=0; side<int(_cellTopo->getSideCount()); side++)
  {
    const int nodeCount = _cellTopo->getSide(side)->getNodeCount();
    for (int k=0; k<_dim; k++)
    {
      const double value = _refNodes(_cellTopo->getNodeMap(_dim-1, side, 0), k);
      bool constant = true;
      for (int n=1; n<nodeCount; n++)
      {
        if (std::abs(_refNodes(_cellTopo->getNodeMap(_dim-1, side, n), k) - value) > 1e-12) constant = false;
      }
      if (constant) _sideForFace[2*k + (value > 0 ? 1 : 0)] = side;
    }
  }
  for (int side : _sideForFace) TEUCHOS_TEST_FOR_EXCEPTION(side < 0, std::logic_error, "BoxDPG: could not identify the sides of the cell topology");

  for (VarPtr field : _fieldVars)
  {
    _fieldOffsets.push_back(_fieldDofsPerCell);
    _fieldDofsPerCell += trialOrdering->getBasisCardinality(field->ID(), VOLUME_INTERIOR_SIDE_INDEX);
  }
  for (VarPtr flux : _fluxVars)
  {
    const int cardinality = trialOrdering->getBasisCardinality(flux->ID(), 0);
    int degree = 0;
    while (intPower(degree + 1, _dim - 1) < cardinality) degree++;
    TEUCHOS_TEST_FOR_EXCEPTION(intPower(degree + 1, _dim - 1) != cardinality, std::invalid_argument,
                               "BoxDPG: flux basis is not a full tensor-product polynomial space");
    for (int side=1; side<int(_cellTopo->getSideCount()); side++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(trialOrdering->getBasisCardinality(flux->ID(), side) != cardinality, std::invalid_argument,
                                 "BoxDPG: flux basis cardinality varies from side to side");
    }
    _fluxDegrees.push_back(degree);
    _fluxOffsets.push_back(_fluxDofsPerPatch);
    _fluxDofsPerPatch += cardinality;
  }
  _ip = bf->graphNorm();
  _rhs = RHS::rhs();
}

BoxForest &BoxDPG::forest()
{
  _numberingValid = false; // the caller may refine
  return _forest;
}

const BoxForest::Skeleton &BoxDPG::skeleton()
{
  if (!_numberingValid)
  {
    _skeleton = _forest.skeleton();
    _numberingValid = true;
  }
  return _skeleton;
}

void BoxDPG::setIP(IPPtr ip) { _ip = ip; }
void BoxDPG::setRHS(RHSPtr rhs) { _rhs = rhs; }

void BoxDPG::addFluxBC(VarPtr flux, BoundarySelector where, BoundaryValue value)
{
  _fluxBCs.push_back({flux->ID(), where, value});
}

double BoxDPG::coordinate(int k, BoxForest::Coord c) const
{
  const int i = int(c >> BoxForest::MAX_LEVEL);
  if (i == int(_rootVertices[k].size()) - 1) return _rootVertices[k][i];
  const double fraction = std::ldexp(double(c - (BoxForest::Coord(i) << BoxForest::MAX_LEVEL)), -BoxForest::MAX_LEVEL);
  return _rootVertices[k][i] + fraction * (_rootVertices[k][i+1] - _rootVertices[k][i]);
}

FC BoxDPG::physicalCellNodes(int nodeIndex) const
{
  const auto &box = _forest.node(nodeIndex).box;
  FC nodes(1, _cellTopo->getNodeCount(), _dim);
  for (int n=0; n<int(_cellTopo->getNodeCount()); n++)
  {
    for (int k=0; k<_dim; k++)
    {
      nodes(0,n,k) = (_refNodes(n,k) > 0) ? coordinate(k, box.hi(k)) : coordinate(k, box.lo[k]);
    }
  }
  return nodes;
}

GlobalIndexType BoxDPG::globalDofCount()
{
  return GlobalIndexType(_forest.leaves().size()) * _fieldDofsPerCell + GlobalIndexType(skeleton().patches.size()) * _fluxDofsPerPatch;
}

GlobalIndexType BoxDPG::fieldDofOffset(int leafOrdinal, VarPtr field)
{
  for (size_t i=0; i<_fieldVars.size(); i++)
  {
    if (_fieldVars[i]->ID() == field->ID()) return GlobalIndexType(leafOrdinal) * _fieldDofsPerCell + _fieldOffsets[i];
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "field variable not found");
}

GlobalIndexType BoxDPG::fluxDofOffset(int patchOrdinal, VarPtr flux)
{
  for (size_t i=0; i<_fluxVars.size(); i++)
  {
    if (_fluxVars[i]->ID() == flux->ID())
    {
      return GlobalIndexType(_forest.leaves().size()) * _fieldDofsPerCell + GlobalIndexType(patchOrdinal) * _fluxDofsPerPatch + _fluxOffsets[i];
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "flux variable not found");
}

void BoxDPG::patchBasisValues(const BoxForest::Patch &patch, int degree, const double *x, vector<double> &values) const
{
  // tensor Legendre in the tangential physical coordinates, on the patch's bounding box
  values.assign(1, 1.0);
  vector<double> next;
  for (int k=0; k<_dim; k++)
  {
    if (k == patch.normal) continue;
    const double a = coordinate(k, patch.lo[k]), b = coordinate(k, patch.hi[k]);
    const double xi = -1 + 2 * (x[k] - a) / (b - a);
    next.resize(values.size() * (degree + 1));
    for (size_t j=0; j<values.size(); j++)
      for (int i=0; i<=degree; i++)
        next[j*(degree+1)+i] = values[j] * legendre(i, xi);
    values.swap(next);
  }
}

void BoxDPG::project(const vector<double> &phi, const vector<double> &psi, const vector<double> &weights, int n, int m,
                     vector<double> &coefficients)
{
  // phi: n x P; psi: m x P.  coefficients (n x m, column-major): psi_j ~ sum_i coefficients(i,j) phi_i in weighted L2
  const int P = int(weights.size());
  vector<double> M(n*n, 0.0);
  coefficients.assign(n*m, 0.0);
  for (int i=0; i<n; i++)
  {
    for (int j=0; j<n; j++)
      for (int q=0; q<P; q++)
        M[j*n+i] += phi[i*P+q] * phi[j*P+q] * weights[q];
    for (int j=0; j<m; j++)
      for (int q=0; q<P; q++)
        coefficients[j*n+i] += phi[i*P+q] * psi[j*P+q] * weights[q];
  }
  Teuchos::LAPACK<int,double> lapack;
  int info;
  lapack.POSV('U', n, m, M.data(), n, coefficients.data(), n, &info);
  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxDPG: projection failed");
}

void BoxDPG::setCell(int leafOrdinal, bool createSideCaches)
{
  FC nodes = physicalCellNodes(_forest.leaves()[leafOrdinal]);
  vector<GlobalIndexType> cellIDs = {GlobalIndexType(leafOrdinal)};
  _basisCache->setPhysicalCellNodes(nodes, cellIDs, createSideCaches);
  _ipBasisCache->setPhysicalCellNodes(nodes, cellIDs, false);
}

void BoxDPG::computeCellData(int leafOrdinal, CellData &cellData)
{
  const BoxForest::Skeleton &skel = skeleton();
  DofOrderingPtr trialOrdering = _elemType->trialOrderPtr, testOrdering = _elemType->testOrderPtr;
  const int numTrial = trialOrdering->totalDofs(), numTest = testOrdering->totalDofs();

  setCell(leafOrdinal, true);
  // the flux is that of the cell below the patch's plane (and of the boundary cell, on the boundary)
  FC parities(1, _cellTopo->getSideCount());
  for (int face=0; face<2*_dim; face++)
  {
    const auto &patch = skel.patches[skel.patchForFace[_forest.faceOrdinal(leafOrdinal, face)]];
    parities(0, _sideForFace[face]) = (patch.isBoundary() || (face % 2 == 1)) ? 1.0 : -1.0;
  }
  _basisCache->setCellSideParities(parities);

  cellData.B.resize(1, numTrial, numTest);
  cellData.G.resize(1, numTest, numTest);
  cellData.l.resize(1, numTest);
  const bool rowMajor = true, checkForZeroCols = false;
  _bf->stiffnessMatrix(cellData.B, _elemType, parities, _basisCache, rowMajor, checkForZeroCols);
  _ip->computeInnerProductMatrix(cellData.G, testOrdering, _ipBasisCache);
  _rhs->integrateAgainstStandardBasis(cellData.l, testOrdering, _basisCache);

  cellData.globalIndices.clear();
  cellData.rows.assign(numTrial, {});
  for (size_t fieldOrdinal=0; fieldOrdinal<_fieldVars.size(); fieldOrdinal++)
  {
    const auto &dofIndices = trialOrdering->getDofIndices(_fieldVars[fieldOrdinal]->ID());
    const GlobalIndexType offset = GlobalIndexType(leafOrdinal) * _fieldDofsPerCell + _fieldOffsets[fieldOrdinal];
    for (size_t i=0; i<dofIndices.size(); i++)
    {
      cellData.rows[dofIndices[i]].push_back({int(cellData.globalIndices.size()), 1.0});
      cellData.globalIndices.push_back(offset + GlobalIndexType(i));
    }
  }

  vector<double> psi, phiValues, psiValues, weights, coefficients;
  for (int face=0; face<2*_dim; face++)
  {
    const int side = _sideForFace[face];
    const int patchOrdinal = skel.patchForFace[_forest.faceOrdinal(leafOrdinal, face)];
    const auto &patch = skel.patches[patchOrdinal];
    BasisCachePtr sideCache = _basisCache->getSideBasisCache(side);
    const FC &points = sideCache->getPhysicalCubaturePointsFC();
    const FC &weightedMeasures = sideCache->getWeightedMeasures();
    const int P = points.dimension(1);
    weights.resize(P);
    for (int q=0; q<P; q++) weights[q] = weightedMeasures(0,q);

    for (size_t fluxOrdinal=0; fluxOrdinal<_fluxVars.size(); fluxOrdinal++)
    {
      const int fluxID = _fluxVars[fluxOrdinal]->ID();
      BasisPtr fluxBasis = trialOrdering->getBasis(fluxID, side);
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
      project(phiValues, psiValues, weights, n, m, coefficients);
      const auto &dofIndices = trialOrdering->getDofIndices(fluxID, side);
      const int base = int(cellData.globalIndices.size());
      const GlobalIndexType offset = fluxDofOffset(patchOrdinal, _fluxVars[fluxOrdinal]);
      for (int j=0; j<m; j++) cellData.globalIndices.push_back(offset + j);
      for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
          if (std::abs(coefficients[j*n+i]) > 1e-14) cellData.rows[dofIndices[i]].push_back({base + j, coefficients[j*n+i]});
    }
  }
}

void BoxDPG::computeBoundaryValues(map<GlobalIndexType,double> &bcValues)
{
  const BoxForest::Skeleton &skel = skeleton();
  vector<double> psi, psiValues, weights, g, coefficients;
  for (int patchOrdinal=0; patchOrdinal<int(skel.patches.size()); patchOrdinal++)
  {
    const auto &patch = skel.patches[patchOrdinal];
    if (!patch.isBoundary()) continue;
    const BoxForest::Face &face = patch.minusFaces.empty() ? patch.plusFaces[0] : patch.minusFaces[0];
    const int direction = face.side / 2;
    const bool upper = (face.side % 2 == 1);
    bool cellIsSet = false;
    for (const FluxBC &bc : _fluxBCs)
    {
      if (!bc.where(direction, upper)) continue;
      size_t fluxOrdinal = 0;
      while (_fluxVars[fluxOrdinal]->ID() != bc.varID) fluxOrdinal++;
      if (!cellIsSet)
      {
        setCell(_forest.leafOrdinal(face.cell), true);
        cellIsSet = true;
      }
      BasisCachePtr sideCache = _basisCache->getSideBasisCache(_sideForFace[face.side]);
      const FC &points = sideCache->getPhysicalCubaturePointsFC();
      const FC &weightedMeasures = sideCache->getWeightedMeasures();
      const int P = points.dimension(1);
      const int m = intPower(_fluxDegrees[fluxOrdinal] + 1, _dim - 1);
      weights.resize(P);
      g.resize(P);
      psiValues.resize(m*P);
      for (int q=0; q<P; q++)
      {
        weights[q] = weightedMeasures(0,q);
        g[q] = bc.value(&points(0,q,0), direction, upper);
        patchBasisValues(patch, _fluxDegrees[fluxOrdinal], &points(0,q,0), psi);
        for (int j=0; j<m; j++) psiValues[j*P+q] = psi[j];
      }
      project(psiValues, g, weights, m, 1, coefficients);
      const GlobalIndexType offset = fluxDofOffset(patchOrdinal, _fluxVars[fluxOrdinal]);
      for (int j=0; j<m; j++) bcValues[offset + j] = coefficients[j];
    }
  }
}

void BoxDPG::solve()
{
  using Matrix = Tpetra::CrsMatrix<double>;
  using Vector = Tpetra::MultiVector<double>;
  using Map = Tpetra::Map<>;
  using LO = Map::local_ordinal_type;
  using GO = Map::global_ordinal_type;

  const GlobalIndexType N = globalDofCount();
  const int numLeaves = int(_forest.leaves().size());
  const int numTrial = _elemType->trialOrderPtr->totalDofs(), numTest = _elemType->testOrderPtr->totalDofs();

  map<GlobalIndexType,double> bcValues;
  computeBoundaryValues(bcValues);
  vector<char> isBC(N, 0);
  vector<double> bcValue(N, 0.0);
  for (const auto &entry : bcValues) { isBC[entry.first] = 1; bcValue[entry.first] = entry.second; }

  vector<map<GlobalIndexType,double>> matrixRows(N);
  vector<double> F(N, 0.0);
  Teuchos::LAPACK<int,double> lapack;
  int info;
  CellData cellData;
  vector<double> Gmatrix, X, Klocal(size_t(numTrial) * numTrial), Flocal(numTrial);
  for (int leaf=0; leaf<numLeaves; leaf++)
  {
    computeCellData(leaf, cellData);
    // X = G^{-1} [B^T  l];  K = B G^{-1} B^T,  f = B G^{-1} l
    Gmatrix.resize(size_t(numTest) * numTest);
    X.resize(size_t(numTest) * (numTrial + 1));
    for (int i=0; i<numTest; i++) for (int j=0; j<numTest; j++) Gmatrix[size_t(j)*numTest+i] = cellData.G(0,i,j);
    for (int j=0; j<numTrial; j++) for (int i=0; i<numTest; i++) X[size_t(j)*numTest+i] = cellData.B(0,j,i);
    for (int i=0; i<numTest; i++) X[size_t(numTrial)*numTest+i] = cellData.l(0,i);
    lapack.POSV('U', numTest, numTrial + 1, Gmatrix.data(), numTest, X.data(), numTest, &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxDPG: Gram matrix factorization failed");
    for (int i=0; i<numTrial; i++)
    {
      for (int j=0; j<=numTrial; j++)
      {
        double sum = 0;
        for (int t=0; t<numTest; t++) sum += cellData.B(0,i,t) * X[size_t(j)*numTest+t];
        if (j < numTrial) Klocal[size_t(i)*numTrial+j] = sum;
        else Flocal[i] = sum;
      }
    }
    for (int i=0; i<numTrial; i++)
    {
      for (const auto &[positionI, weightI] : cellData.rows[i])
      {
        const GlobalIndexType globalI = cellData.globalIndices[positionI];
        if (isBC[globalI]) continue;
        F[globalI] += weightI * Flocal[i];
        for (int j=0; j<numTrial; j++)
        {
          const double kij = Klocal[size_t(i)*numTrial+j];
          if (kij == 0.0) continue;
          for (const auto &[positionJ, weightJ] : cellData.rows[j])
          {
            const GlobalIndexType globalJ = cellData.globalIndices[positionJ];
            const double value = weightI * kij * weightJ;
            if (isBC[globalJ]) F[globalI] -= value * bcValue[globalJ];
            else matrixRows[globalI][globalJ] += value;
          }
        }
      }
    }
  }
  for (GlobalIndexType i=0; i<N; i++)
  {
    if (isBC[i])
    {
      matrixRows[i][i] = 1.0;
      F[i] = bcValue[i];
    }
  }

  auto comm = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_SELF));
  auto tpetraMap = Teuchos::rcp(new Map(Tpetra::global_size_t(N), 0, comm));
  Teuchos::Array<size_t> entriesPerRow(N);
  for (GlobalIndexType i=0; i<N; i++) entriesPerRow[i] = matrixRows[i].size();
  auto A = Teuchos::rcp(new Matrix(tpetraMap, entriesPerRow()));
  vector<GO> columns;
  vector<double> values;
  for (GlobalIndexType i=0; i<N; i++)
  {
    columns.clear();
    values.clear();
    for (const auto &entry : matrixRows[i]) { columns.push_back(entry.first); values.push_back(entry.second); }
    A->insertGlobalValues(GO(i), LO(columns.size()), values.data(), columns.data());
    matrixRows[i].clear();
  }
  A->fillComplete();
  auto x = Teuchos::rcp(new Vector(tpetraMap, 1)), b = Teuchos::rcp(new Vector(tpetraMap, 1));
  for (GlobalIndexType i=0; i<N; i++) b->replaceLocalValue(LO(i), 0, F[i]);
  auto solver = Amesos2::create<Matrix,Vector>("KLU2", A, x, b);
  solver->symbolicFactorization().numericFactorization().solve();
  _solution.resize(N);
  auto xData = x->getData(0);
  for (GlobalIndexType i=0; i<N; i++) _solution[i] = xData[i];
}

vector<double> BoxDPG::cellCoefficients(int leafOrdinal)
{
  CellData cellData;
  computeCellData(leafOrdinal, cellData);
  vector<double> coefficients(cellData.rows.size(), 0.0);
  for (size_t i=0; i<cellData.rows.size(); i++)
    for (const auto &[position, weight] : cellData.rows[i])
      coefficients[i] += weight * _solution[cellData.globalIndices[position]];
  return coefficients;
}

double BoxDPG::L2Error(VarPtr field, PointFunction exact)
{
  BasisPtr basis = _elemType->trialOrderPtr->getBasis(field->ID());
  const int cardinality = basis->getCardinality();
  double errorSquared = 0;
  for (int leaf=0; leaf<int(_forest.leaves().size()); leaf++)
  {
    setCell(leaf, false);
    auto basisValues = _basisCache->getTransformedValues(basis, OP_VALUE);
    const FC &points = _basisCache->getPhysicalCubaturePointsFC();
    const FC &weights = _basisCache->getWeightedMeasures();
    const GlobalIndexType offset = fieldDofOffset(leaf, field);
    for (int q=0; q<points.dimension(1); q++)
    {
      double value = 0;
      for (int i=0; i<cardinality; i++) value += _solution[offset + i] * (*basisValues)(0,i,q);
      const double diff = value - exact(&points(0,q,0));
      errorSquared += diff * diff * weights(0,q);
    }
  }
  return std::sqrt(errorSquared);
}

vector<double> BoxDPG::energyErrors()
{
  const int numLeaves = int(_forest.leaves().size());
  const int numTrial = _elemType->trialOrderPtr->totalDofs(), numTest = _elemType->testOrderPtr->totalDofs();
  vector<double> errors(numLeaves);
  Teuchos::LAPACK<int,double> lapack;
  int info;
  CellData cellData;
  vector<double> Gmatrix(size_t(numTest) * numTest), residual(numTest), representation(numTest);
  for (int leaf=0; leaf<numLeaves; leaf++)
  {
    computeCellData(leaf, cellData);
    vector<double> coefficients(numTrial, 0.0);
    for (int i=0; i<numTrial; i++)
      for (const auto &[position, weight] : cellData.rows[i])
        coefficients[i] += weight * _solution[cellData.globalIndices[position]];
    for (int t=0; t<numTest; t++)
    {
      residual[t] = cellData.l(0,t);
      for (int i=0; i<numTrial; i++) residual[t] -= cellData.B(0,i,t) * coefficients[i];
      for (int s=0; s<numTest; s++) Gmatrix[size_t(s)*numTest+t] = cellData.G(0,t,s);
    }
    representation = residual;
    lapack.POSV('U', numTest, 1, Gmatrix.data(), numTest, representation.data(), numTest, &info);
    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, std::runtime_error, "BoxDPG: Gram matrix factorization failed");
    double errorSquared = 0;
    for (int t=0; t<numTest; t++) errorSquared += residual[t] * representation[t];
    errors[leaf] = std::sqrt(std::max(errorSquared, 0.0));
  }
  return errors;
}
