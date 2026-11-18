//
//  TensorBasisDecomposition.cpp
//  Camellia
//

#include "TensorBasisDecomposition.h"

#include "Basis.h"

#include "Intrepid2_SerendipityBasis.hpp"
#include "Intrepid2_TensorBasis.hpp"

using namespace Camellia;

TensorBasisDecomposition TensorBasisDecomposition::decompose(BasisPtr basis, int spaceDim)
{
  TensorBasisDecomposition d;
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
  // the one-dimensional components; a basis on a side of a tensor-product cell can also carry a component on a point
  // (domain dimension 0, cardinality 1), which leaves the ordinals unchanged and is dropped
  std::vector<Teuchos::RCP<Intrepid2Basis>> components;
  for (const auto &component : tensorBasis->getTensorBasisComponents())
  {
    if ((component->getDomainDimension() == 0) && (component->getCardinality() == 1)) continue;
    if (component->getDomainDimension() != 1) return d;
    components.push_back(component);
  }
  if (int(components.size()) != spaceDim) return d;
  d.components.assign(components.begin(), components.end());
  d.componentCardinality.resize(spaceDim);
  d.fullSize = 1;
  for (int k=0; k<spaceDim; k++)
  {
    d.componentCardinality[k] = components[k]->getCardinality();
    d.fullSize *= d.componentCardinality[k];
  }
  // ordinals: the first component fastest (Basis_TensorBasis: composite = ordinal2 * cardinality1 + ordinal1)
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
  d.tensorIndex = fullOrdinals;
  d.fullTensor = (int(fullOrdinals.size()) == d.fullSize);
  d.valid = true;
  return d;
}

void TensorBasisDecomposition::componentValues(int direction, const std::vector<double> &points, std::vector<double> &values,
                                               std::vector<double> *derivatives) const
{
  const Teuchos::RCP<Intrepid2Basis> &basis = components[direction];
  const int n = basis->getCardinality(), pointCount = int(points.size());
  using PointView = typename Intrepid2Basis::PointViewType;
  using OutputView = typename Intrepid2Basis::OutputViewType;
  // allocate with the default (contiguous) layout: PointViewType is LayoutStride, and a LayoutStride view built from
  // extents alone has no valid strides
  Kokkos::DynRankView<double,Camellia::DeviceType> pointsAllocation("TensorBasisDecomposition points", pointCount, 1);
  PointView pointView = pointsAllocation;
  auto pointsHost = Kokkos::create_mirror_view(pointView);
  for (int q=0; q<pointCount; q++) pointsHost(q,0) = points[q];
  Kokkos::deep_copy(pointView, pointsHost);

  OutputView valueView = basis->allocateOutputView(pointCount, Intrepid2::OPERATOR_VALUE);
  basis->getValues(valueView, pointView, Intrepid2::OPERATOR_VALUE);
  auto valuesHost = Kokkos::create_mirror_view(valueView);
  Kokkos::deep_copy(valuesHost, valueView);
  values.resize(size_t(pointCount) * n);
  for (int q=0; q<pointCount; q++)
    for (int i=0; i<n; i++) values[size_t(q) * n + i] = valuesHost(i,q);

  if (derivatives != nullptr)
  {
    OutputView gradientView = basis->allocateOutputView(pointCount, Intrepid2::OPERATOR_GRAD);
    basis->getValues(gradientView, pointView, Intrepid2::OPERATOR_GRAD);
    auto gradientsHost = Kokkos::create_mirror_view(gradientView);
    Kokkos::deep_copy(gradientsHost, gradientView);
    derivatives->resize(size_t(pointCount) * n);
    for (int q=0; q<pointCount; q++)
      for (int i=0; i<n; i++) (*derivatives)[size_t(q) * n + i] = (gradientsHost.rank() == 3) ? gradientsHost(i,q,0) : gradientsHost(i,q);
  }
}
