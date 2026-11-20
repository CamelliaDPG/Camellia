//
//  SeparableTensorIP.cpp
//  Camellia
//

#include "SeparableTensorIP.h"

#include "BasisCache.h"
#include "Function.h"
#include "LinearTerm.h"

#include "Intrepid2_SerendipityBasis.hpp"
#include "Intrepid2_TensorBasis.hpp"

#include <cmath>

using namespace Camellia;

namespace
{
  using Intrepid2Basis = Intrepid2::Basis<Camellia::DeviceType,double,double>;

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

  //! reference mass and derivative matrices of a one-dimensional basis on [-1,1]: M(i,j) = int phi_i phi_j,
  //! D(i,j) = int phi_i' phi_j'
  void oneDimensionalMatrices(Teuchos::RCP<Intrepid2Basis> basis, std::vector<double> &mass, std::vector<double> &derivative)
  {
    const int n = basis->getCardinality();
    const int pointCount = basis->getDegree() + 2; // exact for products of degree <= 2 * degree + 2
    std::vector<double> x, w;
    gaussLegendreRule(pointCount, x, w);
    using PointView = typename Intrepid2Basis::PointViewType;
    using OutputView = typename Intrepid2Basis::OutputViewType;
    // allocate with the default (contiguous) layout: PointViewType is LayoutStride, and a LayoutStride view built from
    // extents alone has no valid strides (its zero-fill can run past the allocation)
    Kokkos::DynRankView<double,Camellia::DeviceType> pointsAllocation("1D points", pointCount, 1);
    PointView points = pointsAllocation;
    auto pointsHost = Kokkos::create_mirror_view(points);
    for (int q=0; q<pointCount; q++) pointsHost(q,0) = x[q];
    Kokkos::deep_copy(points, pointsHost);
    // views of the shape the basis writes (as the basis allocates them)
    OutputView values = basis->allocateOutputView(pointCount, Intrepid2::OPERATOR_VALUE);
    OutputView gradients = basis->allocateOutputView(pointCount, Intrepid2::OPERATOR_GRAD);
    TEUCHOS_TEST_FOR_EXCEPTION((values.extent_int(0) != n) || (values.extent_int(1) != pointCount) || (gradients.extent_int(0) != n)
                               || (gradients.extent_int(1) != pointCount), std::logic_error, "unexpected 1D basis value shapes");
    basis->getValues(values, points, Intrepid2::OPERATOR_VALUE);
    basis->getValues(gradients, points, Intrepid2::OPERATOR_GRAD);
    auto valuesHost = Kokkos::create_mirror_view(values);
    auto gradientsHost = Kokkos::create_mirror_view(gradients);
    Kokkos::deep_copy(valuesHost, values);
    Kokkos::deep_copy(gradientsHost, gradients);
    auto gradient = [&](int i, int q) { return (gradientsHost.rank() == 3) ? gradientsHost(i,q,0) : gradientsHost(i,q); };
    mass.assign(n * n, 0.0);
    derivative.assign(n * n, 0.0);
    for (int i=0; i<n; i++)
      for (int j=0; j<n; j++)
        for (int q=0; q<pointCount; q++)
        {
          mass[i*n+j] += w[q] * valuesHost(i,q) * valuesHost(j,q);
          derivative[i*n+j] += w[q] * gradient(i,q) * gradient(j,q);
        }
  }
}

namespace
{
  //! the values of a one-dimensional basis at -1 and +1: 2 x n, row-major
  std::vector<double> oneDimensionalEndpointValues(Teuchos::RCP<Intrepid2Basis> basis)
  {
    const int n = basis->getCardinality();
    using PointView = typename Intrepid2Basis::PointViewType;
    using OutputView = typename Intrepid2Basis::OutputViewType;
    Kokkos::DynRankView<double,Camellia::DeviceType> pointsAllocation("1D endpoints", 2, 1); // (contiguous; see above)
    PointView points = pointsAllocation;
    auto pointsHost = Kokkos::create_mirror_view(points);
    pointsHost(0,0) = -1.0;
    pointsHost(1,0) = 1.0;
    Kokkos::deep_copy(points, pointsHost);
    OutputView values = basis->allocateOutputView(2, Intrepid2::OPERATOR_VALUE);
    basis->getValues(values, points, Intrepid2::OPERATOR_VALUE);
    auto valuesHost = Kokkos::create_mirror_view(values);
    Kokkos::deep_copy(valuesHost, values);
    std::vector<double> endpoints(2 * n);
    for (int i=0; i<n; i++) { endpoints[i] = valuesHost(i,0); endpoints[n + i] = valuesHost(i,1); }
    return endpoints;
  }
}

const SeparableTensorIP::TensorDecomposition &SeparableTensorIP::decomposition(BasisPtr basis, int spaceDim)
{
  std::lock_guard<std::mutex> lock(_decompositionMutex);
  auto entry = _decompositions.find(basis.get());
  if (entry != _decompositions.end()) return entry->second;
  TensorDecomposition &d = _decompositions[basis.get()];
  d.basis = basis;
  Teuchos::RCP<Intrepid2Basis> intrepid2Basis = basis->intrepid2Basis();
  if (intrepid2Basis == Teuchos::null) return d;
  // a serendipity basis is a subset of a full tensor basis: its ordinals in the full basis, as SerendipityBasis
  // chooses them (superlinear H^1 degree at most the basis's H^1 degree)
  std::vector<int> fullOrdinals;
  Teuchos::RCP<Intrepid2Basis> fullBasis = intrepid2Basis;
  using Serendipity = Intrepid2::SerendipityBasis<Intrepid2Basis>;
  if (auto serendipity = dynamic_cast<const Serendipity*>(intrepid2Basis.get()))
  {
    fullBasis = serendipity->getUnderlyingBasis();
    const int basisH1Degree = (serendipity->getFunctionSpace() == Intrepid2::FUNCTION_SPACE_HVOL) ? serendipity->getDegree() + 1 : serendipity->getDegree();
    for (int i=0; i<fullBasis->getCardinality(); i++)
    {
      int superlinearDegree = 0;
      for (int p : fullBasis->getH1PolynomialDegreeOfFieldAsVector(i)) if (p > 1) superlinearDegree += p;
      if (superlinearDegree <= basisH1Degree) fullOrdinals.push_back(i);
    }
    if (int(fullOrdinals.size()) != intrepid2Basis->getCardinality()) return d; // not the selection expected
  }
  else
  {
    for (int i=0; i<intrepid2Basis->getCardinality(); i++) fullOrdinals.push_back(i);
  }
  using TensorBasis = Intrepid2::Basis_TensorBasis<Intrepid2Basis>;
  auto tensorBasis = dynamic_cast<const TensorBasis*>(fullBasis.get());
  if (tensorBasis == NULL) return d;
  auto components = tensorBasis->getTensorBasisComponents();
  if (int(components.size()) != spaceDim) return d;
  for (const auto &component : components)
  {
    if (component->getDomainDimension() != 1) return d;
    if (component->getFunctionSpace() != Intrepid2::FUNCTION_SPACE_HGRAD) return d;
  }
  // ordinals: the first component fastest (Basis_TensorBasis: composite = ordinal2 * cardinality1 + ordinal1)
  d.componentCardinality.resize(spaceDim);
  d.mass.resize(spaceDim);
  d.derivative.resize(spaceDim);
  d.endpointValues.resize(spaceDim);
  for (int k=0; k<spaceDim; k++)
  {
    d.componentCardinality[k] = components[k]->getCardinality();
    oneDimensionalMatrices(components[k], d.mass[k], d.derivative[k]);
    d.endpointValues[k] = oneDimensionalEndpointValues(components[k]);
  }
  d.componentOrdinals.resize(fullOrdinals.size(), std::vector<int>(spaceDim));
  for (size_t i=0; i<fullOrdinals.size(); i++)
  {
    int remainder = fullOrdinals[i];
    for (int k=0; k<spaceDim; k++)
    {
      d.componentOrdinals[i][k] = remainder % d.componentCardinality[k];
      remainder /= d.componentCardinality[k];
    }
  }
  int fullCardinality = 1;
  for (int k=0; k<spaceDim; k++) fullCardinality *= d.componentCardinality[k];
  d.tensorIndex = fullOrdinals; // the composite ordinal of Basis_TensorBasis is the first-direction-fastest index
  d.fullTensor = (int(fullOrdinals.size()) == fullCardinality);
  if (d.fullTensor)
  {
    try
    {
      d.inverse = std::make_shared<SeparableGramInverse>(d.mass, d.derivative);
    }
    catch (const std::invalid_argument &)
    {
      d.inverse = nullptr; // (a singular one-dimensional mass matrix; not expected)
    }
  }
  d.valid = true;
  return d;
}

bool SeparableTensorIP::separableCellData(DofOrderingPtr dofOrdering, BasisCachePtr basisCache, SeparableCellData &data)
{
  if (!getBoundaryTerms().empty() || !getZeroMeanTerms().empty() || hasBCsImposed()) return false;
  if (basisCache->isSideCache()) return false;
  const std::set<int> &varIDs = dofOrdering->getVarIDs();
  if (varIDs.size() != 1) return false;
  const int varID = *varIDs.begin();
  const int spaceDim = basisCache->getSpaceDim();

  // the terms: c_t * v or c_t * d_k v
  std::vector<FunctionPtr> coefficients;
  std::vector<int> directions; // -1: the value
  for (const auto &lt : getLinearTerms())
  {
    const auto &summands = lt->summands();
    if (summands.size() != 1) return false;
    const VarPtr var = summands[0].second;
    if (var->ID() != varID) return false;
    const int op = var->op();
    int direction;
    if (op == OP_VALUE) direction = -1;
    else if ((op >= OP_DI1) && (op < OP_DI1 + spaceDim)) direction = op - OP_DI1;
    else return false;
    coefficients.push_back(summands[0].first);
    directions.push_back(direction);
  }
  if (coefficients.empty()) return false;

  BasisPtr basis = dofOrdering->getBasis(varID);
  const TensorDecomposition &d = decomposition(basis, spaceDim);
  if (!d.valid) return false;

  // axis-aligned boxes: diagonal, positive, constant Jacobian
  const Camellia::FieldContainer<double> &jacobian = basisCache->getJacobian(); // (C,P,D,D)
  const int numCells = jacobian.dimension(0), numPoints = jacobian.dimension(1);
  std::vector<std::vector<double>> halfWidths(numCells, std::vector<double>(spaceDim));
  for (int c=0; c<numCells; c++)
  {
    for (int k=0; k<spaceDim; k++)
    {
      halfWidths[c][k] = jacobian(c,0,k,k);
      if (!(halfWidths[c][k] > 0)) return false;
    }
    for (int p=0; p<numPoints; p++)
      for (int i=0; i<spaceDim; i++)
        for (int j=0; j<spaceDim; j++)
        {
          const double expected = (i == j) ? halfWidths[c][i] : 0.0;
          if (std::abs(jacobian(c,p,i,j) - expected) > 1e-12 * halfWidths[c][i]) return false;
        }
  }
  // the coefficients, constant on each cell
  std::vector<std::vector<double>> coefficientSquared(coefficients.size(), std::vector<double>(numCells));
  for (size_t t=0; t<coefficients.size(); t++)
  {
    Camellia::FieldContainer<double> values(numCells, numPoints);
    coefficients[t]->values(values, basisCache);
    for (int c=0; c<numCells; c++)
    {
      const double value = values(c,0);
      for (int p=1; p<numPoints; p++) if (std::abs(values(c,p) - value) > 1e-12 * std::max(1.0, std::abs(value))) return false;
      coefficientSquared[t][c] = value * value;
    }
  }
  // the outflow-face terms: a weight constant on each cell, and the side by the sign of the cell average of side
  data.outflowWeight.assign(_outflowTerms.size(), std::vector<double>(numCells, 0.0));
  data.outflowSide.assign(_outflowTerms.size(), std::vector<int>(numCells, 0));
  for (size_t t=0; t<_outflowTerms.size(); t++)
  {
    const OutflowTerm &term = _outflowTerms[t];
    if ((term.direction < 0) || (term.direction >= spaceDim)) return false;
    Camellia::FieldContainer<double> weights(numCells, numPoints), sides(numCells, numPoints);
    term.weight->values(weights, basisCache);
    term.side->values(sides, basisCache);
    for (int c=0; c<numCells; c++)
    {
      const double value = weights(c,0);
      for (int p=1; p<numPoints; p++) if (std::abs(weights(c,p) - value) > 1e-12 * std::max(1.0, std::abs(value))) return false;
      TEUCHOS_TEST_FOR_EXCEPTION(value < 0.0, std::invalid_argument, "an outflow-face weight is negative");
      double sideSum = 0;
      for (int p=0; p<numPoints; p++) sideSum += sides(c,p);
      data.outflowWeight[t][c] = value;
      data.outflowSide[t][c] = (sideSum > 0) ? 1 : ((sideSum < 0) ? -1 : 0);
    }
  }
  data.varID = varID;
  data.decomposition = &d;
  data.directions = directions;
  data.halfWidths = std::move(halfWidths);
  data.coefficientSquared = std::move(coefficientSquared);
  data.numCells = numCells;
  return true;
}

bool SeparableTensorIP::tryTensorAssembly(Camellia::FieldContainer<double> &innerProduct, DofOrderingPtr dofOrdering, BasisCachePtr basisCache)
{
  if (!_useTensorAssembly) return false;
  SeparableCellData data;
  if (!separableCellData(dofOrdering, basisCache, data)) return false;
  const TensorDecomposition &d = *data.decomposition;
  const int varID = data.varID, spaceDim = int(d.componentCardinality.size());
  const int basisCardinality = int(d.componentOrdinals.size());

  // G(a,b) = sum_t c_t^2 prod_k F_{t,k}(a_k, b_k)
  std::vector<int> dofIndex(basisCardinality);
  for (int a=0; a<basisCardinality; a++) dofIndex[a] = dofOrdering->getDofIndex(varID, a);
  innerProduct.initialize(0.0);
  std::vector<std::vector<double>> factor(spaceDim); // per direction: the scaled mass matrix, then the scaled derivative matrix
  for (int c=0; c<data.numCells; c++)
  {
    for (size_t t=0; t<data.directions.size(); t++)
    {
      const double coefficient = data.coefficientSquared[t][c];
      if (coefficient == 0) continue;
      for (int k=0; k<spaceDim; k++)
      {
        const int n = d.componentCardinality[k];
        const bool differentiated = (data.directions[t] == k);
        const double scale = differentiated ? 1.0 / data.halfWidths[c][k] : data.halfWidths[c][k];
        const std::vector<double> &reference = differentiated ? d.derivative[k] : d.mass[k];
        factor[k].resize(n * n);
        for (int i=0; i<n*n; i++) factor[k][i] = scale * reference[i];
      }
      for (int a=0; a<basisCardinality; a++)
      {
        const std::vector<int> &ia = d.componentOrdinals[a];
        for (int b=a; b<basisCardinality; b++)
        {
          const std::vector<int> &ib = d.componentOrdinals[b];
          double product = coefficient;
          for (int k=0; (k<spaceDim) && (product != 0); k++) product *= factor[k][ia[k] * d.componentCardinality[k] + ib[k]];
          innerProduct(c, dofIndex[a], dofIndex[b]) += product;
        }
      }
    }
    // the outflow-face terms: weight int_F v^2 = weight prod_{k != d} (h_k/2) M_k (x) E_side in direction d
    for (size_t t=0; t<_outflowTerms.size(); t++)
    {
      const int side = data.outflowSide[t][c];
      const double weight = data.outflowWeight[t][c];
      if ((side == 0) || (weight == 0.0)) continue;
      const int direction = _outflowTerms[t].direction;
      for (int k=0; k<spaceDim; k++)
      {
        const int n = d.componentCardinality[k];
        factor[k].resize(n * n);
        if (k == direction)
        {
          const double *phi = d.endpointValues[k].data() + ((side > 0) ? n : 0);
          for (int i=0; i<n; i++) for (int j=0; j<n; j++) factor[k][i*n+j] = phi[i] * phi[j];
        }
        else
        {
          for (int i=0; i<n*n; i++) factor[k][i] = data.halfWidths[c][k] * d.mass[k][i];
        }
      }
      for (int a=0; a<basisCardinality; a++)
      {
        const std::vector<int> &ia = d.componentOrdinals[a];
        for (int b=a; b<basisCardinality; b++)
        {
          const std::vector<int> &ib = d.componentOrdinals[b];
          double product = weight;
          for (int k=0; (k<spaceDim) && (product != 0); k++) product *= factor[k][ia[k] * d.componentCardinality[k] + ib[k]];
          innerProduct(c, dofIndex[a], dofIndex[b]) += product;
        }
      }
    }
    for (int a=0; a<basisCardinality; a++)
      for (int b=a+1; b<basisCardinality; b++)
        innerProduct(c, dofIndex[b], dofIndex[a]) = innerProduct(c, dofIndex[a], dofIndex[b]);
  }
  _tensorCellCount += data.numCells;
  return true;
}

bool SeparableTensorIP::cellInverses(DofOrderingPtr dofOrdering, BasisCachePtr basisCache, std::vector<CellInverse> &inverses)
{
  if (!_useFastInverse) return false;
  SeparableCellData data;
  if (!separableCellData(dofOrdering, basisCache, data)) return false;
  const TensorDecomposition &d = *data.decomposition;
  if (!d.fullTensor || (d.inverse == nullptr)) return false;
  const SeparableGramInverse &inverse = *d.inverse;
  const int spaceDim = inverse.directionCount(), N = inverse.size();
  if (dofOrdering->totalDofs() != N) return false;

  // G = J ( a M^{(x)} + sum_k b_k K_(k) ) on the reference factors: a mass factor is scaled by h_k/2, a derivative factor
  // by 2/h_k, so with J = prod_k h_k/2, a = sum of value terms' c_t^2 and b_k = sum of d_k terms' c_t^2 / (h_k/2)^2
  std::vector<CellInverse> result(data.numCells);
  for (int c=0; c<data.numCells; c++)
  {
    CellInverse &cell = result[c];
    cell.inverse = d.inverse;
    SeparableGramInverse::Scales &s = cell.scales;
    s.jacobian = 1.0;
    for (int k=0; k<spaceDim; k++) s.jacobian *= data.halfWidths[c][k];
    s.valueWeight = 0.0;
    s.derivativeWeights.assign(spaceDim, 0.0);
    for (size_t t=0; t<data.directions.size(); t++)
    {
      const int k = data.directions[t];
      if (k < 0) s.valueWeight += data.coefficientSquared[t][c];
      else s.derivativeWeights[k] += data.coefficientSquared[t][c] / (data.halfWidths[c][k] * data.halfWidths[c][k]);
    }
    if (!(s.valueWeight > 0)) return false; // G singular on the constants (or not positive definite)
    if (!_outflowTerms.empty())
    {
      // the outflow-face terms: weight / (h_d/2) E_side in direction d, relative to J; as K_d + r E_side with r = eta / b_d
      std::vector<double> eta(spaceDim, 0.0);
      std::vector<int> sides(spaceDim, 0);
      for (size_t t=0; t<_outflowTerms.size(); t++)
      {
        const int k = _outflowTerms[t].direction, side = data.outflowSide[t][c];
        const double weight = data.outflowWeight[t][c];
        if ((side == 0) || (weight == 0.0)) continue;
        TEUCHOS_TEST_FOR_EXCEPTION((sides[k] != 0) && (sides[k] != side), std::invalid_argument,
                                   "outflow-face terms on both faces of one direction are not supported by the fast inverse");
        sides[k] = side;
        eta[k] += weight / data.halfWidths[c][k];
      }
      std::vector<std::pair<int,double>> pattern(spaceDim, {0, 0.0});
      bool any = false;
      for (int k=0; k<spaceDim; k++)
      {
        if (sides[k] == 0) continue;
        any = true;
        if (s.derivativeWeights[k] > 0)
        {
          // r, rounded to 12 significant digits so that cells with the same ratio share the inverse
          const double r = eta[k] / s.derivativeWeights[k];
          const double magnitude = std::pow(10.0, std::floor(std::log10(r)) - 11);
          pattern[k] = {sides[k], std::round(r / magnitude) * magnitude};
        }
        else
        {
          pattern[k] = {sides[k], -1.0}; // no derivative term: the endpoint term alone, with weight eta
          s.derivativeWeights[k] = eta[k];
        }
      }
      if (any) cell.inverse = outflowInverse(d, pattern);
    }
  }

  // The constant-mode weight (setConstantModeWeight()) makes the Gram matrix G - c m m^T on each cell; its inverse is
  // the Sherman-Morrison update of G^{-1}:
  //   (G - c m m^T)^{-1} x = y + (c m^T y / (1 - c m^T z)) z,   y = G^{-1} x,  z = G^{-1} m,
  // one more fast-diagonalization solve per cell.  (1 - c m^T z > 0 exactly when G - c m m^T is positive definite.)
  Camellia::FieldContainer<double> m;
  std::vector<double> coefficient;
  if (constantModeCorrection(dofOrdering, basisCache, m, coefficient))
  {
    for (int c=0; c<data.numCells; c++)
    {
      if (coefficient[c] == 0.0) continue;
      CellInverse &cell = result[c];
      cell.c = coefficient[c];
      cell.m.resize(N);
      for (int a=0; a<N; a++) cell.m[d.tensorIndex[a]] = m(c, dofOrdering->getDofIndex(data.varID, a));
      cell.z = cell.m;
      cell.inverse->applyInverse(cell.scales, cell.z.data(), 1);
      double mz = 0;
      for (int i=0; i<N; i++) mz += cell.m[i] * cell.z[i];
      cell.denominator = 1.0 - cell.c * mz;
      TEUCHOS_TEST_FOR_EXCEPTION(!(cell.denominator > 0), std::invalid_argument,
                                 "the constant-mode weight made the Gram matrix indefinite (1 - c m^T G^{-1} m <= 0)");
    }
  }
  inverses = std::move(result);
  _inverseCellCount += data.numCells;
  return true;
}

std::shared_ptr<SeparableGramInverse> SeparableTensorIP::outflowInverse(const TensorDecomposition &d,
                                                                       const std::vector<std::pair<int,double>> &pattern)
{
  std::lock_guard<std::mutex> lock(_decompositionMutex);
  TensorDecomposition &mutableD = const_cast<TensorDecomposition &>(d);
  auto found = mutableD.outflowInverses.find(pattern);
  if (found != mutableD.outflowInverses.end()) return found->second;
  std::vector<std::vector<double>> stiffness = d.derivative;
  for (size_t k=0; k<pattern.size(); k++)
  {
    const int side = pattern[k].first;
    if (side == 0) continue;
    const int n = d.componentCardinality[k];
    const double *phi = d.endpointValues[k].data() + ((side > 0) ? n : 0);
    const double r = pattern[k].second;
    if (r < 0) stiffness[k].assign(n * n, 0.0); // the endpoint term alone (scaled by the derivative weight eta)
    const double scale = (r < 0) ? 1.0 : r;
    for (int i=0; i<n; i++) for (int j=0; j<n; j++) stiffness[k][i*n+j] += scale * phi[i] * phi[j];
  }
  auto inverse = std::make_shared<SeparableGramInverse>(d.mass, stiffness);
  mutableD.outflowInverses[pattern] = inverse;
  return inverse;
}

void SeparableTensorIP::addOutflowFaceTerm(int direction, FunctionPtr weight, FunctionPtr side)
{
  TEUCHOS_TEST_FOR_EXCEPTION((weight == Teuchos::null) || (side == Teuchos::null), std::invalid_argument, "addOutflowFaceTerm: null function");
  _outflowTerms.push_back({direction, weight, side});
}

void SeparableTensorIP::applyCellInverse(const CellInverse &cell, double *x, int vectorCount)
{
  const int N = cell.inverse->size();
  cell.inverse->applyInverse(cell.scales, x, vectorCount);
  if (cell.c == 0.0) return;
  for (int r=0; r<vectorCount; r++)
  {
    double *y = x + size_t(r) * N;
    double my = 0;
    for (int i=0; i<N; i++) my += cell.m[i] * y[i];
    const double scale = cell.c * my / cell.denominator;
    for (int i=0; i<N; i++) y[i] += scale * cell.z[i];
  }
}

void SeparableTensorIP::inverseFactorCoordinates(const CellInverse &cell, double *b, int vectorCount, std::vector<double> &w)
{
  const int N = cell.inverse->size();
  w.assign(size_t(vectorCount) * (N + 1), 0.0);
  if (cell.c != 0.0)
  {
    const double scale = std::sqrt(cell.c / cell.denominator);
    for (int r=0; r<vectorCount; r++)
    {
      double zb = 0;
      for (int i=0; i<N; i++) zb += cell.z[i] * b[size_t(r) * N + i];
      w[size_t(r) * (N + 1) + N] = scale * zb;
    }
  }
  const std::vector<double> D = cell.inverse->eigenvalueDiagonal(cell.scales);
  cell.inverse->applyEigenbasisTranspose(b, vectorCount);
  for (int r=0; r<vectorCount; r++)
    for (int i=0; i<N; i++) w[size_t(r) * (N + 1) + i] = b[size_t(r) * N + i] / std::sqrt(D[i]);
}

void SeparableTensorIP::inverseQuadraticForms(const CellInverse &cell, double *b, int vectorCount, double *values)
{
  const int N = cell.inverse->size();
  std::vector<double> w;
  inverseFactorCoordinates(cell, b, vectorCount, w);
  for (int r=0; r<vectorCount; r++)
  {
    double value = 0;
    for (int i=0; i<=N; i++) value += w[size_t(r) * (N + 1) + i] * w[size_t(r) * (N + 1) + i];
    values[r] = value;
  }
}

bool SeparableTensorIP::applyInverseInnerProductMatrix(Camellia::FieldContainer<double> &values,
                                                       Teuchos::RCP<DofOrdering> dofOrdering,
                                                       Teuchos::RCP<BasisCache> basisCache)
{
  std::vector<CellInverse> inverses;
  if (!cellInverses(dofOrdering, basisCache, inverses)) return false;
  const int N = dofOrdering->totalDofs(), numCells = int(inverses.size());
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 3, std::invalid_argument, "values must have shape (C, R, N)");
  TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(0) != numCells, std::invalid_argument, "values must have one entry per cell of basisCache");
  TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(2) != N, std::invalid_argument, "the last dimension of values must be the number of test dofs");
  const int varID = *dofOrdering->getVarIDs().begin();
  const TensorDecomposition &d = decomposition(dofOrdering->getBasis(varID), basisCache->getSpaceDim());

  // gather each row into the tensor ordering, apply, scatter back
  const int rowCount = values.dimension(1);
  std::vector<int> dofIndex(N);
  for (int a=0; a<N; a++) dofIndex[a] = dofOrdering->getDofIndex(varID, a);
  std::vector<double> rows(size_t(rowCount) * N);
  for (int c=0; c<numCells; c++)
  {
    for (int r=0; r<rowCount; r++)
      for (int a=0; a<N; a++) rows[size_t(r) * N + d.tensorIndex[a]] = values(c, r, dofIndex[a]);
    applyCellInverse(inverses[c], rows.data(), rowCount);
    for (int r=0; r<rowCount; r++)
      for (int a=0; a<N; a++) values(c, r, dofIndex[a]) = rows[size_t(r) * N + d.tensorIndex[a]];
  }
  return true;
}

void SeparableTensorIP::setConstantModeWeight(VarPtr var, FunctionPtr alpha, double ratio)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!(ratio > 0.0) || (ratio > 1.0), std::invalid_argument, "the constant-mode ratio must lie in (0, 1]");
  TEUCHOS_TEST_FOR_EXCEPTION((ratio < 1.0) && ((var == Teuchos::null) || (alpha == Teuchos::null)), std::invalid_argument,
                             "the constant-mode weight needs the test variable and its L^2 weight");
  _constantModeVar = var;
  _constantModeAlpha = alpha;
  _constantModeRatio = ratio;
}

bool SeparableTensorIP::constantModeCorrection(DofOrderingPtr dofOrdering, BasisCachePtr basisCache,
                                               Camellia::FieldContainer<double> &m, std::vector<double> &coefficient)
{
  if (_constantModeRatio >= 1.0) return false;
  if (basisCache->isSideCache()) return false;
  const int varID = _constantModeVar->ID();
  if (!dofOrdering->hasBasisEntry(varID, VOLUME_INTERIOR_SIDE_INDEX)) return false;
  const Camellia::FieldContainer<double> &weightedMeasures = basisCache->getWeightedMeasures(); // (C,P)
  const int numCells = weightedMeasures.dimension(0), numPoints = weightedMeasures.dimension(1);
  // m(c,i) = int_K phi_i, as a linear term integrated against the basis
  LinearTermPtr valueTerm = 1.0 * _constantModeVar;
  m.resize(numCells, dofOrdering->totalDofs());
  valueTerm->integrate(m, dofOrdering, basisCache);
  // c = (1 - ratio) / int_K 1/alpha
  Camellia::FieldContainer<double> alphaValues(numCells, numPoints);
  _constantModeAlpha->values(alphaValues, basisCache);
  coefficient.resize(numCells);
  for (int c=0; c<numCells; c++)
  {
    double inverseAlphaIntegral = 0;
    for (int p=0; p<numPoints; p++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!(alphaValues(c,p) > 0), std::invalid_argument, "the L^2 weight must be positive");
      inverseAlphaIntegral += weightedMeasures(c,p) / alphaValues(c,p);
    }
    coefficient[c] = (1.0 - _constantModeRatio) / inverseAlphaIntegral;
  }
  return true;
}

void SeparableTensorIP::applyConstantModeWeight(Camellia::FieldContainer<double> &innerProduct, DofOrderingPtr dofOrdering, BasisCachePtr basisCache)
{
  Camellia::FieldContainer<double> m;
  std::vector<double> coefficient;
  if (!constantModeCorrection(dofOrdering, basisCache, m, coefficient)) return;
  const int varID = _constantModeVar->ID();
  const int basisCardinality = dofOrdering->getBasis(varID)->getCardinality();
  std::vector<int> dofIndex(basisCardinality);
  for (int a=0; a<basisCardinality; a++) dofIndex[a] = dofOrdering->getDofIndex(varID, a);
  for (int c=0; c<int(coefficient.size()); c++)
  {
    for (int a=0; a<basisCardinality; a++)
    {
      const double ma = m(c, dofIndex[a]);
      if (ma == 0) continue;
      for (int b=0; b<basisCardinality; b++) innerProduct(c, dofIndex[a], dofIndex[b]) -= coefficient[c] * ma * m(c, dofIndex[b]);
    }
  }
}

void SeparableTensorIP::computeInnerProductMatrix(Camellia::FieldContainer<double> &innerProduct,
                                                  Teuchos::RCP<DofOrdering> dofOrdering,
                                                  Teuchos::RCP<BasisCache> basisCache)
{
  if (!tryTensorAssembly(innerProduct, dofOrdering, basisCache))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(!_outflowTerms.empty(), std::invalid_argument,
                               "SeparableTensorIP: outflow-face terms need the one-dimensional assembly path (box cells, tensor-product test bases)");
    _fallbackCellCount += basisCache->getPhysicalCubaturePointsFC().dimension(0);
    TIP<double>::computeInnerProductMatrix(innerProduct, dofOrdering, basisCache);
  }
  applyConstantModeWeight(innerProduct, dofOrdering, basisCache);
}
