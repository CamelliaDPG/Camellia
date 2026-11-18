//
//  BoxCellDPGOperator.cpp
//  Camellia
//

#include "BoxCellDPGOperator.h"

#include "BasisCache.h"
#include "CamelliaCellTools.h"
#include "ConstantScalarFunction.h"
#include "ElementType.h"
#include "LinearTerm.h"
#include "Mesh.h"

#include <algorithm>
#include <cmath>
#include <numeric>

extern "C" void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const double *alpha,
                       const double *a, const int *lda, const double *b, const int *ldb, const double *beta, double *c,
                       const int *ldc);

using namespace Camellia;

namespace
{
  //! n-point Gauss-Legendre rule on [-1,1]
  void gaussLegendreRule(int n, std::vector<double> &x, std::vector<double> &w)
  {
    x.resize(n);
    w.resize(n);
    for (int i=0; i<n; i++)
    {
      double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), dp = 0;
      for (int iteration=0; iteration<100; iteration++)
      {
        double p0 = 1, p1 = z;
        for (int k=2; k<=n; k++) { const double p2 = ((2*k - 1) * z * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
        if (n == 1) { p1 = z; p0 = 1; }
        dp = n * (z * p1 - p0) / (z * z - 1);
        const double dz = p1 / dp;
        z -= dz;
        if (std::abs(dz) < 1e-15) break;
      }
      x[i] = z;
      w[i] = 2.0 / ((1 - z * z) * dp * dp);
    }
  }

  int product(const std::vector<int> &dims)
  {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
  }

  /**
   y = the mode-k product of x (dims, first index fastest) with A or A^T, A row-major rows x cols:
   !transpose: dims[k] == cols, and y has dims[k] = rows;  transpose: dims[k] == rows, and y has dims[k] = cols.
   dims is updated.
   */
  void modeProduct(std::vector<int> &dims, int k, const std::vector<double> &A, int rows, int cols, bool transpose,
                   const double *x, std::vector<double> &y)
  {
    const int nIn = transpose ? rows : cols, nOut = transpose ? cols : rows;
    TEUCHOS_TEST_FOR_EXCEPTION(dims[k] != nIn, std::logic_error, "modeProduct: dimension mismatch");
    int inner = 1, outer = 1;
    for (int j=0; j<k; j++) inner *= dims[j];
    for (int j=k+1; j<int(dims.size()); j++) outer *= dims[j];
    y.assign(size_t(inner) * nOut * outer, 0.0);
    const double one = 1.0, zero = 0.0;
    // column-major, the row-major A is M = A^T (cols x rows, leading dimension cols)
    if (inner == 1)
    {
      // Y (nOut x outer) = op(A) X (nIn x outer): op(A) = A = M^T, or A^T = M
      const char opM = transpose ? 'N' : 'T';
      dgemm_(&opM, "N", &nOut, &outer, &nIn, &one, A.data(), &cols, x, &nIn, &zero, y.data(), &nOut);
    }
    else
    {
      // per slab: Y_o (inner x nOut) = X_o (inner x nIn) op(A)^T, op(A)^T = A^T = M, or A = M^T
      const char opM = transpose ? 'T' : 'N';
      for (int o=0; o<outer; o++)
      {
        dgemm_("N", &opM, &inner, &nOut, &nIn, &one, x + size_t(o) * inner * nIn, &inner, A.data(), &cols, &zero,
               y.data() + size_t(o) * inner * nOut, &inner);
      }
    }
    dims[k] = nOut;
  }

  //! y = x with its axes reordered: new axis a is old axis order[a] (first index fastest); dims is updated
  void permuteAxes(std::vector<int> &dims, const std::vector<int> &order, const std::vector<double> &x, std::vector<double> &y)
  {
    const int D = int(dims.size());
    std::vector<int> oldStride(D, 1), newDims(D);
    for (int j=1; j<D; j++) oldStride[j] = oldStride[j-1] * dims[j-1];
    for (int a=0; a<D; a++) newDims[a] = dims[order[a]];
    const int N = product(dims);
    y.resize(N);
    std::vector<int> index(D, 0);
    for (int n=0; n<N; n++)
    {
      int oldIndex = 0;
      for (int a=0; a<D; a++) oldIndex += index[a] * oldStride[order[a]];
      y[n] = x[oldIndex];
      for (int a=0; a<D; a++) { if (++index[a] < newDims[a]) break; index[a] = 0; }
    }
    dims = newDims;
  }

  //! the value of a constant Function (a ConstantScalarFunction), or false
  bool constantValue(FunctionPtr f, double &value)
  {
    auto constant = dynamic_cast<ConstantScalarFunction<double>*>(f.get());
    if (constant == NULL) return false;
    value = constant->value();
    return true;
  }
}

BoxCellDPGOperator::BoxCellDPGOperator(BFPtr bf, Teuchos::RCP<SeparableTensorIP> ip, MeshPtr mesh, int quadratureEnrichment)
: _bf(bf), _ip(ip), _mesh(mesh), _quadratureEnrichment(quadratureEnrichment)
{
  TEUCHOS_TEST_FOR_EXCEPTION(quadratureEnrichment < 0, std::invalid_argument, "quadratureEnrichment must be nonnegative");
}

BoxCellDPGOperator::TypeData &BoxCellDPGOperator::typeData(GlobalIndexType cellID)
{
  ElementTypePtr elemType = _mesh->getElementType(cellID);
  auto entry = _typeData.find(elemType.get());
  if (entry != _typeData.end()) return entry->second;
  TypeData &data = _typeData[elemType.get()];
  initializeTypeData(data, elemType);
  return data;
}

void BoxCellDPGOperator::initializeTypeData(TypeData &data, ElementTypePtr elemType)
{
  auto decline = [&](const std::string &why) { data.supported = false; data.reason = why; };
  DofOrderingPtr testOrder = elemType->testOrderPtr, trialOrder = elemType->trialOrderPtr;
  CellTopoPtr cellTopo = elemType->cellTopoPtr;
  const int dim = cellTopo->getDimension();
  data.dim = dim;
  data.trialDofCount = trialOrder->totalDofs();
  if (!_bf->getJumpTerms().empty()) return decline("the bilinear form has jump terms");

  // the test space: one variable, a full tensor-product H^1 basis
  const std::set<int> &testIDs = testOrder->getVarIDs();
  if (testIDs.size() != 1) return decline("more than one test variable");
  data.testVarID = *testIDs.begin();
  data.testBasis = TensorBasisDecomposition::decompose(testOrder->getBasis(data.testVarID), dim);
  if (!data.testBasis.valid) return decline("the test basis is not a tensor product of one-dimensional bases");
  if (!data.testBasis.fullTensor) return decline("the test basis is not a full tensor-product basis");
  for (const auto &component : data.testBasis.components)
    if (component->getFunctionSpace() != Intrepid2::FUNCTION_SPACE_HGRAD) return decline("the test basis is not H^1");

  // the quadrature: q_k = n_k + ceil(enrichment / 2) points per direction
  data.points.resize(dim);
  data.weights.resize(dim);
  data.testValues.resize(dim);
  data.testDerivatives.resize(dim);
  data.testEndpoints.resize(dim);
  std::vector<int> q(dim);
  for (int k=0; k<dim; k++)
  {
    q[k] = data.testBasis.componentCardinality[k] + (_quadratureEnrichment + 1) / 2;
    gaussLegendreRule(q[k], data.points[k], data.weights[k]);
    data.testBasis.componentValues(k, data.points[k], data.testValues[k], &data.testDerivatives[k]);
    data.testBasis.componentValues(k, {-1.0, 1.0}, data.testEndpoints[k]);
  }
  data.testDofIndex.assign(data.testBasis.fullSize, -1);
  for (int a=0; a<int(data.testBasis.tensorIndex.size()); a++)
    data.testDofIndex[data.testBasis.tensorIndex[a]] = testOrder->getDofIndex(data.testVarID, a);
  const int P = product(q);
  {
    std::vector<Kokkos::DynRankView<double,Camellia::DeviceType>> components(dim);
    for (int k=0; k<dim; k++)
    {
      components[k] = Kokkos::DynRankView<double,Camellia::DeviceType>("BoxCellDPGOperator 1D points", q[k], 1);
      auto host = Kokkos::create_mirror_view(components[k]);
      for (int i=0; i<q[k]; i++) host(i,0) = data.points[k][i];
      Kokkos::deep_copy(components[k], host);
    }
    data.tensorRefPoints = Intrepid2::TensorPoints<double,Camellia::DeviceType>(components);
  }
  data.refPoints.resize(P, dim);
  {
    std::vector<int> index(dim, 0);
    for (int p=0; p<P; p++)
    {
      for (int k=0; k<dim; k++) data.refPoints(p,k) = data.points[k][index[k]];
      for (int k=0; k<dim; k++) { if (++index[k] < q[k]) break; index[k] = 0; }
    }
  }

  // the terms
  std::map<std::pair<int,int>, int> groupOrdinal; // (trial var, test direction) -> group
  std::map<int, double> fluxCoefficients;          // flux var -> coefficient
  for (const auto &term : _bf->getTerms())
  {
    LinearTermPtr trialTerm = term.first, testTerm = term.second;
    for (const auto &trialSummand : trialTerm->summands())
    {
      for (const auto &testSummand : testTerm->summands())
      {
        const VarPtr trialVar = trialSummand.second, testVar = testSummand.second;
        if (testVar->ID() != data.testVarID) return decline("a term's test variable is not the test variable");
        if (trialVar->op() != OP_VALUE) return decline("a term applies an operator to a trial variable");
        int testDirection;
        const int op = testVar->op();
        if (op == OP_VALUE) testDirection = -1;
        else if ((op >= OP_DI1) && (op < OP_DI1 + dim)) testDirection = op - OP_DI1;
        else if ((op >= OP_DX) && (op <= OP_DZ) && (op - OP_DX < dim)) testDirection = op - OP_DX;
        else return decline("a test operator is neither the value nor a first derivative");
        if (trialVar->varType() == FIELD)
        {
          const auto key = std::make_pair(trialVar->ID(), testDirection);
          if (groupOrdinal.find(key) == groupOrdinal.end())
          {
            groupOrdinal[key] = int(data.groups.size());
            data.groups.push_back({trialVar->ID(), testDirection, {}});
          }
          data.groups[groupOrdinal[key]].coefficients.push_back(trialSummand.first * testSummand.first);
        }
        else if (trialVar->varType() == FLUX)
        {
          if (testDirection != -1) return decline("a flux term tests a derivative");
          double trialCoefficient, testCoefficient;
          if (!constantValue(trialSummand.first, trialCoefficient) || !constantValue(testSummand.first, testCoefficient))
            return decline("a flux term's coefficient is not constant");
          fluxCoefficients[trialVar->ID()] += trialCoefficient * testCoefficient;
        }
        else
        {
          return decline("a trial variable is neither a field nor a flux");
        }
      }
    }
  }

  // the field variables' bases
  std::set<int> fieldIDs;
  for (const FieldGroup &group : data.groups) fieldIDs.insert(group.trialVarID);
  for (int fieldID : fieldIDs)
  {
    FieldData field;
    field.trialVarID = fieldID;
    BasisPtr fieldBasis = trialOrder->getBasis(fieldID);
    field.hvol = (fieldBasis->functionSpace() == Camellia::FUNCTION_SPACE_HVOL);
    field.basis = TensorBasisDecomposition::decompose(fieldBasis, dim);
    if (!field.basis.valid) return decline("a field basis is not a (subset of a) tensor product of one-dimensional bases");
    field.values.resize(dim);
    for (int k=0; k<dim; k++) field.basis.componentValues(k, data.points[k], field.values[k]);
    const int cardinality = int(field.basis.tensorIndex.size());
    field.dofIndex.resize(cardinality);
    for (int b=0; b<cardinality; b++) field.dofIndex[b] = trialOrder->getDofIndex(fieldID, b);
    data.fields.push_back(field);
  }

  // the sides: each is a face x_normal = +-1, its coordinates a signed permutation of the cell's other coordinates
  const int sideCount = cellTopo->getSideCount();
  for (int s=0; s<sideCount; s++)
  {
    Camellia::FieldContainer<double> sidePoints(dim, dim - 1), cellPoints(dim, dim);
    for (int k=0; k<dim-1; k++) sidePoints(k+1, k) = 1.0; // the origin, then the unit vectors
    CamelliaCellTools::mapToReferenceSubcell(cellPoints, sidePoints, dim - 1, s, cellTopo);
    SideData side;
    side.side = s;
    side.normal = -1;
    for (int k=0; k<dim; k++)
      if (std::abs(std::abs(cellPoints(0,k)) - 1.0) < 1e-12)
      {
        bool fixed = true;
        for (int i=1; i<dim; i++) fixed = fixed && (std::abs(cellPoints(i,k) - cellPoints(0,k)) < 1e-12);
        if (fixed) side.normal = k;
      }
    if (side.normal < 0) return decline("a side is not a face x_k = +-1");
    side.endpoint = (cellPoints(0, side.normal) > 0) ? 1 : 0;
    for (int i=0; i<dim-1; i++)
    {
      int direction = -1, sign = 0;
      for (int k=0; k<dim; k++)
      {
        const double difference = cellPoints(i+1,k) - cellPoints(0,k);
        if (std::abs(difference) < 1e-12) continue;
        if ((direction >= 0) || (std::abs(std::abs(difference) - 1.0) > 1e-12) || (std::abs(cellPoints(0,k)) > 1e-12))
          return decline("a side's reference map is not a signed permutation");
        direction = k;
        sign = (difference > 0) ? 1 : -1;
      }
      if ((direction < 0) || (direction == side.normal)) return decline("a side's reference map is degenerate");
      side.cellDirection.push_back(direction);
      side.sign.push_back(sign);
    }
    data.sides.push_back(side);
  }

  // the flux variables' side bases
  for (const auto &entry : fluxCoefficients)
  {
    const int fluxID = entry.first;
    for (int s : trialOrder->getSidesForVarID(fluxID))
    {
      FluxSideData fluxSide;
      fluxSide.fluxVarID = fluxID;
      fluxSide.side = s;
      fluxSide.coefficient = entry.second;
      fluxSide.basis = TensorBasisDecomposition::decompose(trialOrder->getBasis(fluxID, s), dim - 1);
      if (!fluxSide.basis.valid) return decline("a flux basis is not a (subset of a) tensor product of one-dimensional bases");
      const SideData &side = data.sides[s];
      fluxSide.values.resize(dim - 1);
      for (int i=0; i<dim-1; i++)
      {
        std::vector<double> sidePointsOfDirection = data.points[side.cellDirection[i]];
        for (double &x : sidePointsOfDirection) x *= side.sign[i];
        fluxSide.basis.componentValues(i, sidePointsOfDirection, fluxSide.values[i]);
      }
      const int cardinality = int(fluxSide.basis.tensorIndex.size());
      fluxSide.dofIndex.resize(cardinality);
      for (int b=0; b<cardinality; b++) fluxSide.dofIndex[b] = trialOrder->getDofIndex(fluxID, b, s);
      data.fluxSides.push_back(fluxSide);
    }
  }
  data.supported = true;
}

BoxCellDPGOperator::CellData &BoxCellDPGOperator::cellData(GlobalIndexType cellID, const TypeData &type)
{
  auto entry = _cellData.find(cellID);
  if (entry != _cellData.end()) return entry->second;
  CellData &data = _cellData[cellID];
  const int dim = type.dim;

  // geometry: an axis-aligned box (diagonal, constant Jacobian)
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh, cellID);
  basisCache->setRefCellPoints(type.tensorRefPoints);
  const Camellia::FieldContainer<double> &jacobian = basisCache->getJacobian(); // (1,P,D,D)
  const int P = type.refPoints.dimension(0);
  data.halfWidths.resize(dim);
  for (int k=0; k<dim; k++) data.halfWidths[k] = jacobian(0,0,k,k);
  for (int p=0; p<P; p++)
    for (int i=0; i<dim; i++)
      for (int j=0; j<dim; j++)
      {
        const double expected = (i == j) ? data.halfWidths[i] : 0.0;
        TEUCHOS_TEST_FOR_EXCEPTION(std::abs(jacobian(0,p,i,j) - expected) > 1e-12 * std::abs(data.halfWidths[i]), std::invalid_argument,
                                   "BoxCellDPGOperator: cell is not an axis-aligned box");
      }
  double J = 1.0;
  for (int k=0; k<dim; k++) J *= data.halfWidths[k];

  // side parities (flux values are multiplied by them, as in LinearTerm::integrate)
  Camellia::FieldContainer<double> parities = _mesh->cellSideParitiesForCell(cellID);
  data.sideParities.resize(type.sides.size());
  for (int s=0; s<int(type.sides.size()); s++) data.sideParities[s] = (parities.rank() == 2) ? parities(0,s) : parities(s);

  // G_K^{-1}
  {
    BasisCachePtr ipBasisCache = BasisCache::basisCacheForCell(_mesh, cellID, true);
    std::vector<SeparableTensorIP::CellInverse> inverses;
    const bool invertible = _ip->cellInverses(_mesh->getElementType(cellID)->testOrderPtr, ipBasisCache, inverses);
    TEUCHOS_TEST_FOR_EXCEPTION(!invertible || (inverses.size() != 1), std::invalid_argument,
                               "BoxCellDPGOperator: the inner product has no fast inverse on cell " << cellID);
    data.gramInverse = inverses[0];
  }

  // the flux bases' value transforms on the sides (e.g. 1 / |det J J^-T n| for L^2 fluxes), measured
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  data.fluxTransform.resize(type.fluxSides.size());
  for (int fs=0; fs<int(type.fluxSides.size()); fs++)
  {
    const FluxSideData &fluxSide = type.fluxSides[fs];
    BasisPtr sideBasis = trialOrder->getBasis(fluxSide.fluxVarID, fluxSide.side);
    BasisCachePtr sideCache = BasisCache::basisCacheForCell(_mesh, cellID)->getSideBasisCache(fluxSide.side);
    auto transformed = sideCache->getTransformedValues(sideBasis, OP_VALUE); // (C, F, P)
    // the reference values at the side cache's points, from the one-dimensional components (the Intrepid2 serendipity
    // basis does not support the FieldContainer getValues())
    const Camellia::FieldContainer<double> &sidePoints = sideCache->getRefCellPointsFC(); // (P, dim - 1)
    const int F = transformed->dimension(1), sidePointCount = transformed->dimension(2);
    TEUCHOS_TEST_FOR_EXCEPTION(sidePoints.dimension(0) != sidePointCount, std::logic_error, "BoxCellDPGOperator: unexpected side points");
    std::vector<std::vector<double>> componentTables(dim - 1);
    for (int i=0; i<dim-1; i++)
    {
      std::vector<double> coordinates(sidePointCount);
      for (int p=0; p<sidePointCount; p++) coordinates[p] = sidePoints(p,i);
      fluxSide.basis.componentValues(i, coordinates, componentTables[i]);
    }
    auto reference = [&](int f, int p) {
      double value = 1.0;
      for (int i=0; i<dim-1; i++)
        value *= componentTables[i][size_t(p) * fluxSide.basis.componentCardinality[i] + fluxSide.basis.componentOrdinals[f][i]];
      return value;
    };
    int fMax = 0, pMax = 0;
    double largest = 0;
    for (int f=0; f<F; f++)
      for (int p=0; p<sidePointCount; p++)
        if (std::abs(reference(f,p)) > largest) { largest = std::abs(reference(f,p)); fMax = f; pMax = p; }
    TEUCHOS_TEST_FOR_EXCEPTION(largest == 0, std::logic_error, "BoxCellDPGOperator: flux basis values vanish on a side");
    const double ratio = (*transformed)(0,fMax,pMax) / reference(fMax,pMax);
    for (int f=0; f<F; f++)
      for (int p=0; p<sidePointCount; p++)
        TEUCHOS_TEST_FOR_EXCEPTION(std::abs((*transformed)(0,f,p) - ratio * reference(f,p)) > 1e-10 * std::abs(ratio) * largest,
                                   std::invalid_argument, "BoxCellDPGOperator: the flux basis's transform on a side is not a constant scaling");
    data.fluxTransform[fs] = ratio;
  }

  // the coefficients at the quadrature points, times the weights, the Jacobian, and the derivative scaling 1/(h_k/2)
  std::vector<double> weights(P, J);
  {
    std::vector<int> index(dim, 0);
    for (int p=0; p<P; p++)
    {
      for (int k=0; k<dim; k++) weights[p] *= type.weights[k][index[k]];
      for (int k=0; k<dim; k++) { if (++index[k] < int(type.points[k].size())) break; index[k] = 0; }
    }
  }
  data.groupCoefficients.resize(type.groups.size());
  Camellia::FieldContainer<double> values(1, P);
  for (int g=0; g<int(type.groups.size()); g++)
  {
    const FieldGroup &group = type.groups[g];
    std::vector<double> &c = data.groupCoefficients[g];
    c.assign(P, 0.0);
    for (FunctionPtr coefficient : group.coefficients)
    {
      values.initialize(0.0);
      coefficient->values(values, basisCache);
      for (int p=0; p<P; p++) c[p] += values(0,p);
    }
    double scale = (group.testDirection >= 0) ? 1.0 / data.halfWidths[group.testDirection] : 1.0;
    for (const FieldData &field : type.fields)
      if ((field.trialVarID == group.trialVarID) && field.hvol) scale /= J; // L^2 values: reference values / det J
    for (int p=0; p<P; p++) c[p] *= weights[p] * scale;
  }
  return data;
}

bool BoxCellDPGOperator::supports(GlobalIndexType cellID, std::string *reason)
{
  const TypeData &type = typeData(cellID);
  if (reason != nullptr) *reason = type.reason;
  return type.supported;
}

int BoxCellDPGOperator::trialDofCount(GlobalIndexType cellID)
{
  return typeData(cellID).trialDofCount;
}

int BoxCellDPGOperator::testSize(GlobalIndexType cellID)
{
  return typeData(cellID).testBasis.fullSize;
}

const std::vector<int> &BoxCellDPGOperator::testDofIndex(GlobalIndexType cellID)
{
  return typeData(cellID).testDofIndex;
}

void BoxCellDPGOperator::applyB(GlobalIndexType cellID, const double *u, double *Bu)
{
  const TypeData &type = typeData(cellID);
  TEUCHOS_TEST_FOR_EXCEPTION(!type.supported, std::invalid_argument, "BoxCellDPGOperator: unsupported cell: " + type.reason);
  const CellData &cell = cellData(cellID, type);
  const int dim = type.dim, N = type.testBasis.fullSize;
  const std::vector<int> &n = type.testBasis.componentCardinality;
  std::fill(Bu, Bu + N, 0.0);
  std::vector<double> x, y;

  // field terms: the field at the quadrature points, times each group's coefficients, tested
  for (const FieldData &field : type.fields)
  {
    x.assign(field.basis.fullSize, 0.0);
    for (int b=0; b<int(field.dofIndex.size()); b++) x[field.basis.tensorIndex[b]] = u[field.dofIndex[b]];
    std::vector<int> dims = field.basis.componentCardinality;
    for (int k=0; k<dim; k++)
    {
      modeProduct(dims, k, field.values[k], int(type.points[k].size()), field.basis.componentCardinality[k], false, x.data(), y);
      x.swap(y);
    }
    const std::vector<double> fieldValues = x;
    for (int g=0; g<int(type.groups.size()); g++)
    {
      const FieldGroup &group = type.groups[g];
      if (group.trialVarID != field.trialVarID) continue;
      const std::vector<double> &c = cell.groupCoefficients[g];
      x.resize(fieldValues.size());
      for (size_t p=0; p<x.size(); p++) x[p] = fieldValues[p] * c[p];
      std::vector<int> gridDims = dims;
      for (int k=0; k<dim; k++)
      {
        const std::vector<double> &A = (group.testDirection == k) ? type.testDerivatives[k] : type.testValues[k];
        modeProduct(gridDims, k, A, int(type.points[k].size()), n[k], true, x.data(), y);
        x.swap(y);
      }
      for (int i=0; i<N; i++) Bu[i] += x[i];
    }
  }

  // flux terms: on each side, the flux at the side's quadrature points, times weights, parity, coefficient, tested
  std::vector<int> testStride(dim, 1);
  for (int k=1; k<dim; k++) testStride[k] = testStride[k-1] * n[k-1];
  for (int fs=0; fs<int(type.fluxSides.size()); fs++)
  {
    const FluxSideData &fluxSide = type.fluxSides[fs];
    const SideData &side = type.sides[fluxSide.side];
    x.assign(fluxSide.basis.fullSize, 0.0);
    for (int b=0; b<int(fluxSide.dofIndex.size()); b++) x[fluxSide.basis.tensorIndex[b]] = u[fluxSide.dofIndex[b]];
    std::vector<int> dims = fluxSide.basis.componentCardinality;
    for (int i=0; i<dim-1; i++)
    {
      modeProduct(dims, i, fluxSide.values[i], int(type.points[side.cellDirection[i]].size()), fluxSide.basis.componentCardinality[i],
                  false, x.data(), y);
      x.swap(y);
    }
    // weights (in side order), parity, coefficient, side Jacobian
    // side measure prod h/2, times the flux basis's value transform on the side, the coefficient and the parity
    double scale = fluxSide.coefficient * cell.sideParities[fluxSide.side] * cell.fluxTransform[fs];
    for (int i=0; i<dim-1; i++) scale *= cell.halfWidths[side.cellDirection[i]];
    {
      std::vector<int> index(dim - 1, 0);
      for (size_t p=0; p<x.size(); p++)
      {
        double w = scale;
        for (int i=0; i<dim-1; i++) w *= type.weights[side.cellDirection[i]][index[i]];
        x[p] *= w;
        for (int i=0; i<dim-1; i++) { if (++index[i] < dims[i]) break; index[i] = 0; }
      }
    }
    // to the cell's order of the other directions
    std::vector<int> others;
    for (int k=0; k<dim; k++) if (k != side.normal) others.push_back(k);
    std::vector<int> order(dim - 1);
    for (int a=0; a<dim-1; a++)
      order[a] = int(std::find(side.cellDirection.begin(), side.cellDirection.end(), others[a]) - side.cellDirection.begin());
    permuteAxes(dims, order, x, y);
    x.swap(y);
    for (int a=0; a<dim-1; a++)
    {
      const int k = others[a];
      modeProduct(dims, a, type.testValues[k], int(type.points[k].size()), n[k], true, x.data(), y);
      x.swap(y);
    }
    // times the test functions' values at the side, in the normal direction
    const double *endpointValues = &type.testEndpoints[side.normal][side.endpoint * n[side.normal]];
    for (int i=0; i<N; i++)
    {
      int reduced = 0, reducedStride = 1, aNormal = 0;
      for (int k=0; k<dim; k++)
      {
        const int ak = (i / testStride[k]) % n[k];
        if (k == side.normal) { aNormal = ak; continue; }
        reduced += ak * reducedStride;
        reducedStride *= n[k];
      }
      Bu[i] += endpointValues[aNormal] * x[reduced];
    }
  }
}

void BoxCellDPGOperator::applyBTranspose(GlobalIndexType cellID, const double *v, double *BTv)
{
  const TypeData &type = typeData(cellID);
  TEUCHOS_TEST_FOR_EXCEPTION(!type.supported, std::invalid_argument, "BoxCellDPGOperator: unsupported cell: " + type.reason);
  const CellData &cell = cellData(cellID, type);
  const int dim = type.dim, N = type.testBasis.fullSize;
  const std::vector<int> &n = type.testBasis.componentCardinality;
  std::fill(BTv, BTv + type.trialDofCount, 0.0);
  std::vector<double> x, y;

  for (const FieldData &field : type.fields)
  {
    std::vector<double> h;
    std::vector<int> gridDims;
    for (int g=0; g<int(type.groups.size()); g++)
    {
      const FieldGroup &group = type.groups[g];
      if (group.trialVarID != field.trialVarID) continue;
      x.assign(v, v + N);
      std::vector<int> dims = n;
      for (int k=0; k<dim; k++)
      {
        const std::vector<double> &A = (group.testDirection == k) ? type.testDerivatives[k] : type.testValues[k];
        modeProduct(dims, k, A, int(type.points[k].size()), n[k], false, x.data(), y);
        x.swap(y);
      }
      const std::vector<double> &c = cell.groupCoefficients[g];
      if (h.empty()) { h.assign(x.size(), 0.0); gridDims = dims; }
      for (size_t p=0; p<x.size(); p++) h[p] += x[p] * c[p];
    }
    if (h.empty()) continue;
    for (int k=0; k<dim; k++)
    {
      modeProduct(gridDims, k, field.values[k], int(type.points[k].size()), field.basis.componentCardinality[k], true, h.data(), y);
      h.swap(y);
    }
    for (int b=0; b<int(field.dofIndex.size()); b++) BTv[field.dofIndex[b]] += h[field.basis.tensorIndex[b]];
  }

  std::vector<int> testStride(dim, 1);
  for (int k=1; k<dim; k++) testStride[k] = testStride[k-1] * n[k-1];
  for (int fs=0; fs<int(type.fluxSides.size()); fs++)
  {
    const FluxSideData &fluxSide = type.fluxSides[fs];
    const SideData &side = type.sides[fluxSide.side];
    std::vector<int> others;
    for (int k=0; k<dim; k++) if (k != side.normal) others.push_back(k);
    // the test function's trace on the side, in the cell's order of the other directions
    const double *endpointValues = &type.testEndpoints[side.normal][side.endpoint * n[side.normal]];
    std::vector<int> dims;
    for (int k : others) dims.push_back(n[k]);
    x.assign(product(dims), 0.0);
    for (int i=0; i<N; i++)
    {
      int reduced = 0, reducedStride = 1, aNormal = 0;
      for (int k=0; k<dim; k++)
      {
        const int ak = (i / testStride[k]) % n[k];
        if (k == side.normal) { aNormal = ak; continue; }
        reduced += ak * reducedStride;
        reducedStride *= n[k];
      }
      x[reduced] += endpointValues[aNormal] * v[i];
    }
    for (int a=0; a<dim-1; a++)
    {
      const int k = others[a];
      modeProduct(dims, a, type.testValues[k], int(type.points[k].size()), n[k], false, x.data(), y);
      x.swap(y);
    }
    // to the side's coordinate order
    std::vector<int> order(dim - 1);
    for (int i=0; i<dim-1; i++)
      order[i] = int(std::find(others.begin(), others.end(), side.cellDirection[i]) - others.begin());
    permuteAxes(dims, order, x, y);
    x.swap(y);
    // side measure prod h/2, times the flux basis's value transform on the side, the coefficient and the parity
    double scale = fluxSide.coefficient * cell.sideParities[fluxSide.side] * cell.fluxTransform[fs];
    for (int i=0; i<dim-1; i++) scale *= cell.halfWidths[side.cellDirection[i]];
    {
      std::vector<int> index(dim - 1, 0);
      for (size_t p=0; p<x.size(); p++)
      {
        double w = scale;
        for (int i=0; i<dim-1; i++) w *= type.weights[side.cellDirection[i]][index[i]];
        x[p] *= w;
        for (int i=0; i<dim-1; i++) { if (++index[i] < dims[i]) break; index[i] = 0; }
      }
    }
    for (int i=0; i<dim-1; i++)
    {
      modeProduct(dims, i, fluxSide.values[i], int(type.points[side.cellDirection[i]].size()), fluxSide.basis.componentCardinality[i],
                  true, x.data(), y);
      x.swap(y);
    }
    for (int b=0; b<int(fluxSide.dofIndex.size()); b++) BTv[fluxSide.dofIndex[b]] += x[fluxSide.basis.tensorIndex[b]];
  }
}

void BoxCellDPGOperator::applyInverseGram(GlobalIndexType cellID, double *v)
{
  const TypeData &type = typeData(cellID);
  TEUCHOS_TEST_FOR_EXCEPTION(!type.supported, std::invalid_argument, "BoxCellDPGOperator: unsupported cell: " + type.reason);
  SeparableTensorIP::applyCellInverse(cellData(cellID, type).gramInverse, v);
}

void BoxCellDPGOperator::applySchur(GlobalIndexType cellID, const double *u, double *Su)
{
  std::vector<double> v(testSize(cellID));
  applyB(cellID, u, v.data());
  applyInverseGram(cellID, v.data());
  applyBTranspose(cellID, v.data(), Su);
}

void BoxCellDPGOperator::applyLoad(GlobalIndexType cellID, RHSPtr rhs, double *load)
{
  const TypeData &type = typeData(cellID);
  TEUCHOS_TEST_FOR_EXCEPTION(!type.supported, std::invalid_argument, "BoxCellDPGOperator: unsupported cell: " + type.reason);
  const int N = type.testBasis.fullSize;
  DofOrderingPtr testOrder = _mesh->getElementType(cellID)->testOrderPtr;
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh, cellID, false, _quadratureEnrichment);
  Camellia::FieldContainer<double> l(1, testOrder->totalDofs());
  rhs->integrateAgainstStandardBasis(l, testOrder, basisCache);
  std::vector<double> v(N);
  for (int i=0; i<N; i++) v[i] = l(0, type.testDofIndex[i]);
  applyInverseGram(cellID, v.data());
  applyBTranspose(cellID, v.data(), load);
}
