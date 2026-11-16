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
    PointView points("1D points", pointCount, 1);
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
  for (int k=0; k<spaceDim; k++)
  {
    d.componentCardinality[k] = components[k]->getCardinality();
    oneDimensionalMatrices(components[k], d.mass[k], d.derivative[k]);
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
  d.valid = true;
  return d;
}

bool SeparableTensorIP::tryTensorAssembly(Camellia::FieldContainer<double> &innerProduct, DofOrderingPtr dofOrdering, BasisCachePtr basisCache)
{
  if (!_useTensorAssembly) return false;
  if (!getBoundaryTerms().empty() || !getZeroMeanTerms().empty() || hasBCsImposed()) return false;
  if (basisCache->isSideCache()) return false;
  const std::set<int> &varIDs = dofOrdering->getVarIDs();
  if (varIDs.size() != 1) return false;
  const int varID = *varIDs.begin();
  const int spaceDim = basisCache->getSpaceDim();

  // the terms: c_t * v or c_t * d_k v
  struct Term { FunctionPtr coefficient; int direction; }; // direction -1: the value
  std::vector<Term> terms;
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
    terms.push_back({summands[0].first, direction});
  }
  if (terms.empty()) return false;

  BasisPtr basis = dofOrdering->getBasis(varID);
  const TensorDecomposition &d = decomposition(basis, spaceDim);
  if (!d.valid) return false;
  const int basisCardinality = basis->getCardinality();

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
  std::vector<std::vector<double>> coefficientSquared(terms.size(), std::vector<double>(numCells));
  for (size_t t=0; t<terms.size(); t++)
  {
    Camellia::FieldContainer<double> values(numCells, numPoints);
    terms[t].coefficient->values(values, basisCache);
    for (int c=0; c<numCells; c++)
    {
      const double value = values(c,0);
      for (int p=1; p<numPoints; p++) if (std::abs(values(c,p) - value) > 1e-12 * std::max(1.0, std::abs(value))) return false;
      coefficientSquared[t][c] = value * value;
    }
  }

  // G(a,b) = sum_t c_t^2 prod_k F_{t,k}(a_k, b_k)
  std::vector<int> dofIndex(basisCardinality);
  for (int a=0; a<basisCardinality; a++) dofIndex[a] = dofOrdering->getDofIndex(varID, a);
  innerProduct.initialize(0.0);
  std::vector<std::vector<double>> factor(spaceDim); // per direction: the scaled mass matrix, then the scaled derivative matrix
  for (int c=0; c<numCells; c++)
  {
    for (size_t t=0; t<terms.size(); t++)
    {
      const double coefficient = coefficientSquared[t][c];
      if (coefficient == 0) continue;
      for (int k=0; k<spaceDim; k++)
      {
        const int n = d.componentCardinality[k];
        const bool differentiated = (terms[t].direction == k);
        const double scale = differentiated ? 1.0 / halfWidths[c][k] : halfWidths[c][k];
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
    for (int a=0; a<basisCardinality; a++)
      for (int b=a+1; b<basisCardinality; b++)
        innerProduct(c, dofIndex[b], dofIndex[a]) = innerProduct(c, dofIndex[a], dofIndex[b]);
  }
  _tensorCellCount += numCells;
  return true;
}

void SeparableTensorIP::computeInnerProductMatrix(Camellia::FieldContainer<double> &innerProduct,
                                                  Teuchos::RCP<DofOrdering> dofOrdering,
                                                  Teuchos::RCP<BasisCache> basisCache)
{
  if (tryTensorAssembly(innerProduct, dofOrdering, basisCache)) return;
  _fallbackCellCount += basisCache->getPhysicalCubaturePointsFC().dimension(0);
  TIP<double>::computeInnerProductMatrix(innerProduct, dofOrdering, basisCache);
}
