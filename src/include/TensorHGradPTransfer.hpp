#pragma once

#include <Intrepid2_DerivedBasisFamily.hpp>
#include <Intrepid2_HierarchicalBasisFamily.hpp>
#include <Intrepid2_HierarchicalBasisProlongator.hpp>

#include <Kokkos_Core.hpp>

#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace Camellia
{
  template<class Scalar>
  struct TensorHGradPTransfer
  {
    int dim = 0;
    int fineP = 0;
    int coarseP = 0;
    int fineRows = 0;
    int coarseRows = 0;
    std::vector<Scalar> prolongation; // row-major: fineOrdinal * coarseRows + coarseOrdinal
    std::vector<Scalar> restriction;  // row-major: coarseOrdinal * fineRows + fineOrdinal; restriction = prolongation^T
  };

  inline int tensorHGradPTransferCheckedProduct(const std::vector<int> &extents)
  {
    int product = 1;
    for (const int extent : extents)
    {
      if (extent <= 0)
        throw std::invalid_argument("tensorHGradPTransferCheckedProduct requires positive extents");
      product *= extent;
    }
    return product;
  }

  inline int tensorHGradPTransferOrdinalFromMultiIndex(const std::vector<int> &index,
                                                       const std::vector<int> &extents)
  {
    if (index.size() != extents.size())
      throw std::invalid_argument("tensorHGradPTransferOrdinalFromMultiIndex rank mismatch");

    int ordinal = 0;
    int stride = 1;
    for (size_t d=0; d<index.size(); d++)
    {
      if ((index[d] < 0) || (index[d] >= extents[d]))
        throw std::invalid_argument("tensorHGradPTransferOrdinalFromMultiIndex index out of bounds");
      ordinal += index[d] * stride;
      stride *= extents[d];
    }
    return ordinal;
  }

  inline std::vector<int> tensorHGradPTransferMultiIndexFromOrdinal(const int ordinal,
                                                                    const std::vector<int> &extents)
  {
    if (ordinal < 0)
      throw std::invalid_argument("tensorHGradPTransferMultiIndexFromOrdinal requires nonnegative ordinal");

    std::vector<int> index(extents.size(), 0);
    int remaining = ordinal;
    for (size_t d=0; d<extents.size(); d++)
    {
      if (extents[d] <= 0)
        throw std::invalid_argument("tensorHGradPTransferMultiIndexFromOrdinal requires positive extents");
      index[d] = remaining % extents[d];
      remaining /= extents[d];
    }

    if (remaining != 0)
      throw std::invalid_argument("tensorHGradPTransferMultiIndexFromOrdinal ordinal out of bounds");

    return index;
  }

  template<class DeviceType, class Scalar>
  TensorHGradPTransfer<Scalar> buildTensorHGradPTransfer(const int dim, const int fineP,
                                                         const int coarseP)
  {
    if ((dim < 1) || (fineP < 2) || (coarseP < 1) || (coarseP >= fineP))
      throw std::invalid_argument("buildTensorHGradPTransfer requires dim >= 1, fineP >= 2, and 1 <= coarseP < fineP");

    TensorHGradPTransfer<Scalar> transfer;
    transfer.dim = dim;
    transfer.fineP = fineP;
    transfer.coarseP = coarseP;

    const std::vector<int> fineExtents(static_cast<size_t>(dim), fineP + 1);
    const std::vector<int> coarseExtents(static_cast<size_t>(dim), transfer.coarseP + 1);
    transfer.fineRows = tensorHGradPTransferCheckedProduct(fineExtents);
    transfer.coarseRows = tensorHGradPTransferCheckedProduct(coarseExtents);

    using BasisFamily = Intrepid2::HierarchicalBasisFamily<DeviceType>;
    auto coarseBasis = Intrepid2::getHypercubeBasis_HGRAD<BasisFamily>(transfer.coarseP, dim);
    auto fineBasis = Intrepid2::getHypercubeBasis_HGRAD<BasisFamily>(fineP, dim);

    if (coarseBasis->getCardinality() != transfer.coarseRows)
      throw std::runtime_error("buildTensorHGradPTransfer coarse basis cardinality did not match tensor-product row count");
    if (fineBasis->getCardinality() != transfer.fineRows)
      throw std::runtime_error("buildTensorHGradPTransfer fine basis cardinality did not match tensor-product row count");

    Intrepid2::HierarchicalBasisProlongator<DeviceType,Scalar> prolongator(coarseBasis, fineBasis);

    using View2D = Kokkos::View<Scalar**,DeviceType>;
    View2D coarseIdentity("TensorHGradPTransfer coarse identity", transfer.coarseRows, transfer.coarseRows);
    View2D fineProlongedIdentity("TensorHGradPTransfer fine prolonged identity", transfer.coarseRows, transfer.fineRows);

    Kokkos::deep_copy(coarseIdentity, Scalar(0.0));
    auto coarseIdentityHost = Kokkos::create_mirror_view(coarseIdentity);
    for (int coarseOrdinal=0; coarseOrdinal<transfer.coarseRows; coarseOrdinal++)
      coarseIdentityHost(coarseOrdinal, coarseOrdinal) = Scalar(1.0);
    Kokkos::deep_copy(coarseIdentity, coarseIdentityHost);

    prolongator.apply(fineProlongedIdentity, coarseIdentity);
    Kokkos::fence();

    auto fineProlongedIdentityHost =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fineProlongedIdentity);

    transfer.prolongation.assign(static_cast<size_t>(transfer.fineRows) * static_cast<size_t>(transfer.coarseRows),
                                 Scalar(0.0));
    transfer.restriction.assign(static_cast<size_t>(transfer.coarseRows) * static_cast<size_t>(transfer.fineRows),
                                Scalar(0.0));

    for (int coarseOrdinal=0; coarseOrdinal<transfer.coarseRows; coarseOrdinal++)
    {
      for (int fineOrdinal=0; fineOrdinal<transfer.fineRows; fineOrdinal++)
      {
        const Scalar value = fineProlongedIdentityHost(coarseOrdinal, fineOrdinal);
        transfer.prolongation[static_cast<size_t>(fineOrdinal) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(coarseOrdinal)] = value;
        transfer.restriction[static_cast<size_t>(coarseOrdinal) * static_cast<size_t>(transfer.fineRows) + static_cast<size_t>(fineOrdinal)] = value;
      }
    }

    return transfer;
  }

  template<class DeviceType, class Scalar>
  TensorHGradPTransfer<Scalar> buildTensorHGradPTransfer(const int dim, const int fineP)
  {
    return buildTensorHGradPTransfer<DeviceType,Scalar>(dim, fineP, fineP - 1);
  }

  template<class DeviceType, class Scalar>
  const TensorHGradPTransfer<Scalar> &cachedTensorHGradPTransfer(const int dim, const int fineP,
                                                                const int coarseP)
  {
    using CacheKey = std::tuple<int,int,int>;
    static std::map<CacheKey,TensorHGradPTransfer<Scalar>> cache;
    const CacheKey key(dim, fineP, coarseP);
    auto found = cache.find(key);
    if (found != cache.end()) return found->second;

    auto inserted = cache.emplace(key, buildTensorHGradPTransfer<DeviceType,Scalar>(dim, fineP, coarseP));
    return inserted.first->second;
  }

  template<class DeviceType, class Scalar>
  const TensorHGradPTransfer<Scalar> &cachedTensorHGradPTransfer(const int dim, const int fineP)
  {
    return cachedTensorHGradPTransfer<DeviceType,Scalar>(dim, fineP, fineP - 1);
  }

  template<class Scalar>
  void applyTensorHGradPTransferProlongation(const TensorHGradPTransfer<Scalar> &transfer,
                                             const std::vector<Scalar> &coarseVector,
                                             std::vector<Scalar> &fineVector)
  {
    if (static_cast<int>(coarseVector.size()) != transfer.coarseRows)
      throw std::runtime_error("p-multigrid prolongation coarse vector has incorrect length");

    fineVector.assign(static_cast<size_t>(transfer.fineRows), Scalar(0.0));
    for (int fineOrdinal=0; fineOrdinal<transfer.fineRows; fineOrdinal++)
    {
      for (int coarseOrdinal=0; coarseOrdinal<transfer.coarseRows; coarseOrdinal++)
      {
        fineVector[static_cast<size_t>(fineOrdinal)] +=
            transfer.prolongation[static_cast<size_t>(fineOrdinal) * static_cast<size_t>(transfer.coarseRows) + static_cast<size_t>(coarseOrdinal)] *
            coarseVector[static_cast<size_t>(coarseOrdinal)];
      }
    }
  }

  template<class Scalar>
  void applyTensorHGradPTransferRestriction(const TensorHGradPTransfer<Scalar> &transfer,
                                            const std::vector<Scalar> &fineVector,
                                            std::vector<Scalar> &coarseVector)
  {
    if (static_cast<int>(fineVector.size()) != transfer.fineRows)
      throw std::runtime_error("p-multigrid restriction fine vector has incorrect length");

    coarseVector.assign(static_cast<size_t>(transfer.coarseRows), Scalar(0.0));
    for (int coarseOrdinal=0; coarseOrdinal<transfer.coarseRows; coarseOrdinal++)
    {
      for (int fineOrdinal=0; fineOrdinal<transfer.fineRows; fineOrdinal++)
      {
        coarseVector[static_cast<size_t>(coarseOrdinal)] +=
            transfer.restriction[static_cast<size_t>(coarseOrdinal) * static_cast<size_t>(transfer.fineRows) + static_cast<size_t>(fineOrdinal)] *
            fineVector[static_cast<size_t>(fineOrdinal)];
      }
    }
  }
}
