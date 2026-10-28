#include "TensorCoefficientValueEvaluator.h"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

TensorCoefficientValueEvaluator::OwnedCoefficientValues::OwnedCoefficientValues(const std::pair<IndexType,IndexType> &ownedRange,
                                                                               const std::vector<double> &ownedCoefficients,
                                                                               const std::map<IndexType,double> &ghostCoefficientsByGlobalIndex)
: ownedRange(ownedRange),
  ownedCoefficients(ownedCoefficients),
  ghostCoefficientsByGlobalIndex(ghostCoefficientsByGlobalIndex)
{
  populateDeviceMirrors();
}

bool TensorCoefficientValueEvaluator::OwnedCoefficientValues::hasDeviceMirrors() const
{
  return (static_cast<size_t>(ownedCoefficientsDevice.extent(0)) == ownedCoefficients.size()) &&
         (static_cast<size_t>(ghostGlobalIndicesDevice.extent(0)) == ghostCoefficientsByGlobalIndex.size()) &&
         (static_cast<size_t>(ghostCoefficientsDevice.extent(0)) == ghostCoefficientsByGlobalIndex.size());
}

void TensorCoefficientValueEvaluator::OwnedCoefficientValues::populateDeviceMirrors()
{
  ownedCoefficientsDevice = DeviceScalarView("tensor owned coefficient values", ownedCoefficients.size());
  auto ownedHostMirror = Kokkos::create_mirror_view(ownedCoefficientsDevice);
  for (size_t ordinal=0; ordinal<ownedCoefficients.size(); ordinal++)
  {
    ownedHostMirror(ordinal) = ownedCoefficients[ordinal];
  }
  Kokkos::deep_copy(ownedCoefficientsDevice, ownedHostMirror);

  ghostGlobalIndicesDevice = DeviceIndexView("tensor ghost coefficient global indices", ghostCoefficientsByGlobalIndex.size());
  ghostCoefficientsDevice = DeviceScalarView("tensor ghost coefficient values", ghostCoefficientsByGlobalIndex.size());
  auto ghostIndexHostMirror = Kokkos::create_mirror_view(ghostGlobalIndicesDevice);
  auto ghostValueHostMirror = Kokkos::create_mirror_view(ghostCoefficientsDevice);
  size_t ghostOrdinal = 0;
  for (const auto &ghostEntry : ghostCoefficientsByGlobalIndex)
  {
    ghostIndexHostMirror(ghostOrdinal) = ghostEntry.first;
    ghostValueHostMirror(ghostOrdinal) = ghostEntry.second;
    ghostOrdinal++;
  }
  Kokkos::deep_copy(ghostGlobalIndicesDevice, ghostIndexHostMirror);
  Kokkos::deep_copy(ghostCoefficientsDevice, ghostValueHostMirror);
}

namespace
{
  void validateVolumeCoefficientMap(const TensorCoefficientMap &coefficientMap)
  {
    const auto &layout = coefficientMap.variableLayout();
    TEUCHOS_TEST_FOR_EXCEPTION(layout.hasTraceComponentOrdinal(), std::invalid_argument,
                               "TensorCoefficientValueEvaluator currently supports only volume variables.");
    for (int d=0; d<layout.componentCount(); d++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(layout.isTraceSupportComponent(d), std::invalid_argument,
                                 "TensorCoefficientValueEvaluator cannot evaluate trace/skeleton-supported variables as volume values.");
    }
  }

  void validateOwnedCoefficientValues(const TensorCoefficientValueEvaluator::OwnedCoefficientValues &ownedValues)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(ownedValues.ownedRange.second < ownedValues.ownedRange.first, std::invalid_argument,
                               "TensorCoefficientValueEvaluator owned range must be a valid half-open interval.");
    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(ownedValues.ownedCoefficients.size()) != ownedValues.ownedRange.second - ownedValues.ownedRange.first, std::invalid_argument,
                               "TensorCoefficientValueEvaluator owned coefficient vector size does not match owned range length.");
  }

  double coefficientValue(IndexType globalIndex,
                          const TensorCoefficientValueEvaluator::OwnedCoefficientValues &ownedValues)
  {
    if ((ownedValues.ownedRange.first <= globalIndex) && (globalIndex < ownedValues.ownedRange.second))
    {
      return ownedValues.ownedCoefficients[static_cast<size_t>(globalIndex - ownedValues.ownedRange.first)];
    }
    auto entry = ownedValues.ghostCoefficientsByGlobalIndex.find(globalIndex);
    TEUCHOS_TEST_FOR_EXCEPTION(entry == ownedValues.ghostCoefficientsByGlobalIndex.end(), std::invalid_argument,
                               "TensorCoefficientValueEvaluator missing ghost coefficient for a nonowned cell-local DOF.");
    return entry->second;
  }

  std::vector<int> componentPointOrdinals(IndexType tensorPointOrdinal,
                                          const TensorBasisCacheND &basisCache)
  {
    const int componentCount = basisCache.componentCount();
    std::vector<int> pointOrdinals(componentCount, 0);
    for (int d=0; d<componentCount; d++)
    {
      const int pointCount = basisCache.componentCache(d).pointCount();
      pointOrdinals[d] = static_cast<int>((tensorPointOrdinal / basisCache.pointStride(d)) % pointCount);
    }
    return pointOrdinals;
  }

  IndexType packedOrdinalForComponents(const std::vector<int> &selectedComponents,
                                       const std::vector<int> &allComponentPointOrdinals,
                                       const TensorBasisCacheND &basisCache)
  {
    IndexType ordinal = 0;
    IndexType stride = 1;
    for (int componentOrdinal : selectedComponents)
    {
      ordinal += stride * allComponentPointOrdinals[componentOrdinal];
      stride *= basisCache.componentCache(componentOrdinal).pointCount();
    }
    return ordinal;
  }

  IndexType pointCountForComponents(const std::vector<int> &components,
                                    const TensorBasisCacheND &basisCache)
  {
    IndexType pointCount = 1;
    for (int componentOrdinal : components)
    {
      pointCount *= basisCache.componentCache(componentOrdinal).pointCount();
    }
    return pointCount;
  }

  void validateComponentSplit(const std::vector<int> &configurationComponentOrdinals,
                              const std::vector<int> &velocityComponentOrdinals,
                              int componentCount)
  {
    std::vector<bool> covered(componentCount, false);
    auto markComponents = [&](const std::vector<int> &components)
    {
      for (int componentOrdinal : components)
      {
        TEUCHOS_TEST_FOR_EXCEPTION((componentOrdinal < 0) || (componentOrdinal >= componentCount), std::invalid_argument,
                                   "TensorCoefficientValueEvaluator component split contains an out-of-range component ordinal.");
        TEUCHOS_TEST_FOR_EXCEPTION(covered[componentOrdinal], std::invalid_argument,
                                   "TensorCoefficientValueEvaluator component split contains a duplicate component ordinal.");
        covered[componentOrdinal] = true;
      }
    };

    markComponents(configurationComponentOrdinals);
    markComponents(velocityComponentOrdinals);

    for (int d=0; d<componentCount; d++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(!covered[d], std::invalid_argument,
                                 "TensorCoefficientValueEvaluator component split must cover every tensor component exactly once.");
    }
  }
}

TensorCoefficientValueEvaluator::HostVector
TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(const TensorCoefficientMap &coefficientMap,
                                                          const TensorBasisCacheND &basisCache,
                                                          const std::vector<double> &globalCoefficients,
                                                          const TensorMeshTopology::TensorCellTuple &cellTuple)
{
  validateVolumeCoefficientMap(coefficientMap);
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(cellTuple.size()) != basisCache.componentCount(), std::invalid_argument,
                             "TensorCoefficientValueEvaluator cell tuple and basis cache component counts do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(globalCoefficients.size()) < coefficientMap.coefficientCount(), std::invalid_argument,
                             "TensorCoefficientValueEvaluator global coefficient vector is too short.");

  const auto localToGlobal = coefficientMap.indicesForVolumeCell(cellTuple);
  IndexType expectedLocalDofCount = 1;
  for (int d=0; d<basisCache.componentCount(); d++)
  {
    expectedLocalDofCount *= basisCache.componentCache(d).basisCardinality();
  }
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(localToGlobal.size()) != expectedLocalDofCount, std::invalid_argument,
                             "TensorCoefficientValueEvaluator basis cache and coefficient map local DOF counts do not match.");

  HostVector values("tensor coefficient values on tensor cell", basisCache.tensorPointCount());
  for (IndexType tensorPointOrdinal=0; tensorPointOrdinal<basisCache.tensorPointCount(); tensorPointOrdinal++)
  {
    const auto pointOrdinals = componentPointOrdinals(tensorPointOrdinal, basisCache);
    double value = 0.0;
    for (IndexType localOrdinal=0; localOrdinal<expectedLocalDofCount; localOrdinal++)
    {
      IndexType workingLocalOrdinal = localOrdinal;
      double basisProduct = 1.0;
      for (int d=0; d<basisCache.componentCount(); d++)
      {
        const auto &componentCache = basisCache.componentCache(d);
        const int basisOrdinal = static_cast<int>(workingLocalOrdinal % componentCache.basisCardinality());
        workingLocalOrdinal /= componentCache.basisCardinality();
        basisProduct *= componentCache.valuesAtQuadrature().view_host()(basisOrdinal, pointOrdinals[d]);
      }
      value += globalCoefficients[localToGlobal[localOrdinal]] * basisProduct;
    }
    values(tensorPointOrdinal) = value;
  }
  return values;
}

TensorCoefficientValueEvaluator::HostVector
TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(const TensorCoefficientMap &coefficientMap,
                                                          const TensorBasisCacheND &basisCache,
                                                          const std::pair<IndexType,IndexType> &ownedRange,
                                                          const std::vector<double> &ownedCoefficients,
                                                          const std::map<IndexType,double> &ghostCoefficientsByGlobalIndex,
                                                          const TensorMeshTopology::TensorCellTuple &cellTuple)
{
  return volumeValuesOnTensorCell(coefficientMap,
                                  basisCache,
                                  OwnedCoefficientValues(ownedRange, ownedCoefficients, ghostCoefficientsByGlobalIndex),
                                  cellTuple);
}

TensorCoefficientValueEvaluator::HostVector
TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(const TensorCoefficientMap &coefficientMap,
                                                          const TensorBasisCacheND &basisCache,
                                                          const OwnedCoefficientValues &ownedValues,
                                                          const TensorMeshTopology::TensorCellTuple &cellTuple)
{
  validateVolumeCoefficientMap(coefficientMap);
  validateOwnedCoefficientValues(ownedValues);
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(cellTuple.size()) != basisCache.componentCount(), std::invalid_argument,
                             "TensorCoefficientValueEvaluator cell tuple and basis cache component counts do not match.");

  const auto localToGlobal = coefficientMap.indicesForVolumeCell(cellTuple);
  IndexType expectedLocalDofCount = 1;
  for (int d=0; d<basisCache.componentCount(); d++)
  {
    expectedLocalDofCount *= basisCache.componentCache(d).basisCardinality();
  }
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(localToGlobal.size()) != expectedLocalDofCount, std::invalid_argument,
                             "TensorCoefficientValueEvaluator basis cache and coefficient map local DOF counts do not match.");

  HostVector values("tensor coefficient values on tensor cell from owned plus ghost coefficients", basisCache.tensorPointCount());
  for (IndexType tensorPointOrdinal=0; tensorPointOrdinal<basisCache.tensorPointCount(); tensorPointOrdinal++)
  {
    const auto pointOrdinals = componentPointOrdinals(tensorPointOrdinal, basisCache);
    double value = 0.0;
    for (IndexType localOrdinal=0; localOrdinal<expectedLocalDofCount; localOrdinal++)
    {
      IndexType workingLocalOrdinal = localOrdinal;
      double basisProduct = 1.0;
      for (int d=0; d<basisCache.componentCount(); d++)
      {
        const auto &componentCache = basisCache.componentCache(d);
        const int basisOrdinal = static_cast<int>(workingLocalOrdinal % componentCache.basisCardinality());
        workingLocalOrdinal /= componentCache.basisCardinality();
        basisProduct *= componentCache.valuesAtQuadrature().view_host()(basisOrdinal, pointOrdinals[d]);
      }
      value += coefficientValue(localToGlobal[localOrdinal], ownedValues) * basisProduct;
    }
    values(tensorPointOrdinal) = value;
  }
  return values;
}

TensorCoefficientValueEvaluator::DeviceVector
TensorCoefficientValueEvaluator::volumeValuesOnTensorCellDevice(const TensorCoefficientMap &coefficientMap,
                                                               const TensorBasisCacheND &basisCache,
                                                               const OwnedCoefficientValues &ownedValues,
                                                               const TensorMeshTopology::TensorCellTuple &cellTuple)
{
  validateVolumeCoefficientMap(coefficientMap);
  validateOwnedCoefficientValues(ownedValues);
  TEUCHOS_TEST_FOR_EXCEPTION(!ownedValues.hasDeviceMirrors(), std::invalid_argument,
                             "TensorCoefficientValueEvaluator device evaluation requires populated owned-plus-ghost device mirrors.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(cellTuple.size()) != basisCache.componentCount(), std::invalid_argument,
                             "TensorCoefficientValueEvaluator cell tuple and basis cache component counts do not match.");

  const auto localToGlobal = coefficientMap.indicesForVolumeCell(cellTuple);
  const int componentCount = basisCache.componentCount();
  IndexType expectedLocalDofCount = 1;
  int maxBasisCardinality = 0;
  int maxPointCount = 0;
  for (int d=0; d<componentCount; d++)
  {
    expectedLocalDofCount *= basisCache.componentCache(d).basisCardinality();
    maxBasisCardinality = std::max(maxBasisCardinality, basisCache.componentCache(d).basisCardinality());
    maxPointCount = std::max(maxPointCount, basisCache.componentCache(d).pointCount());
  }
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(localToGlobal.size()) != expectedLocalDofCount, std::invalid_argument,
                             "TensorCoefficientValueEvaluator basis cache and coefficient map local DOF counts do not match.");

  for (IndexType gid : localToGlobal)
  {
    if ((gid < ownedValues.ownedRange.first) || (ownedValues.ownedRange.second <= gid))
    {
      TEUCHOS_TEST_FOR_EXCEPTION(ownedValues.ghostCoefficientsByGlobalIndex.find(gid) == ownedValues.ghostCoefficientsByGlobalIndex.end(),
                                 std::invalid_argument,
                                 "TensorCoefficientValueEvaluator missing ghost coefficient for a nonowned cell-local DOF.");
    }
  }

  DeviceIndexVector localToGlobalDevice("tensor coefficient local-to-global device indices", localToGlobal.size());
  auto localToGlobalHost = Kokkos::create_mirror_view(localToGlobalDevice);
  for (size_t localOrdinal=0; localOrdinal<localToGlobal.size(); localOrdinal++)
  {
    localToGlobalHost(localOrdinal) = localToGlobal[localOrdinal];
  }
  Kokkos::deep_copy(localToGlobalDevice, localToGlobalHost);

  DeviceIntVector basisCardinalities("tensor coefficient basis cardinalities", componentCount);
  DeviceIntVector pointCounts("tensor coefficient point counts", componentCount);
  DeviceIndexVector pointStrides("tensor coefficient point strides", componentCount);
  DeviceBasisValueTensor basisValues("tensor coefficient basis values", componentCount, maxBasisCardinality, maxPointCount);
  auto basisCardinalitiesHost = Kokkos::create_mirror_view(basisCardinalities);
  auto pointCountsHost = Kokkos::create_mirror_view(pointCounts);
  auto pointStridesHost = Kokkos::create_mirror_view(pointStrides);
  auto basisValuesHost = Kokkos::create_mirror_view(basisValues);
  for (int d=0; d<componentCount; d++)
  {
    const auto &componentCache = basisCache.componentCache(d);
    basisCardinalitiesHost(d) = componentCache.basisCardinality();
    pointCountsHost(d) = componentCache.pointCount();
    pointStridesHost(d) = basisCache.pointStride(d);
    for (int basisOrdinal=0; basisOrdinal<componentCache.basisCardinality(); basisOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<componentCache.pointCount(); pointOrdinal++)
      {
        basisValuesHost(d,basisOrdinal,pointOrdinal) = componentCache.valuesAtQuadrature().view_host()(basisOrdinal, pointOrdinal);
      }
    }
  }
  Kokkos::deep_copy(basisCardinalities, basisCardinalitiesHost);
  Kokkos::deep_copy(pointCounts, pointCountsHost);
  Kokkos::deep_copy(pointStrides, pointStridesHost);
  Kokkos::deep_copy(basisValues, basisValuesHost);

  DeviceVector values("tensor coefficient device values on tensor cell", basisCache.tensorPointCount());
  auto ownedCoefficientsDevice = ownedValues.ownedCoefficientsDevice;
  auto ghostGlobalIndicesDevice = ownedValues.ghostGlobalIndicesDevice;
  auto ghostCoefficientsDevice = ownedValues.ghostCoefficientsDevice;
  const IndexType ownedBegin = ownedValues.ownedRange.first;
  const IndexType ownedEnd = ownedValues.ownedRange.second;
  const int ghostCount = static_cast<int>(ownedValues.ghostGlobalIndicesDevice.extent(0));
  const IndexType tensorPointCount = basisCache.tensorPointCount();

  Kokkos::parallel_for("TensorCoefficientValueEvaluator::volumeValuesOnTensorCellDevice",
                       static_cast<int>(tensorPointCount),
                       KOKKOS_LAMBDA(const int tensorPointOrdinal)
                       {
                         double value = 0.0;
                         for (IndexType localOrdinal=0; localOrdinal<expectedLocalDofCount; localOrdinal++)
                         {
                           IndexType workingLocalOrdinal = localOrdinal;
                           double basisProduct = 1.0;
                           for (int d=0; d<componentCount; d++)
                           {
                             const int basisOrdinal = static_cast<int>(workingLocalOrdinal % basisCardinalities(d));
                             workingLocalOrdinal /= basisCardinalities(d);
                             const int pointOrdinal = static_cast<int>((static_cast<IndexType>(tensorPointOrdinal) / pointStrides(d)) % pointCounts(d));
                             basisProduct *= basisValues(d,basisOrdinal,pointOrdinal);
                           }

                           const IndexType globalIndex = localToGlobalDevice(localOrdinal);
                           double coefficient = 0.0;
                           if ((ownedBegin <= globalIndex) && (globalIndex < ownedEnd))
                           {
                             coefficient = ownedCoefficientsDevice(globalIndex - ownedBegin);
                           }
                           else
                           {
                             for (int ghostOrdinal=0; ghostOrdinal<ghostCount; ghostOrdinal++)
                             {
                               if (ghostGlobalIndicesDevice(ghostOrdinal) == globalIndex)
                               {
                                 coefficient = ghostCoefficientsDevice(ghostOrdinal);
                               }
                             }
                           }
                           value += coefficient * basisProduct;
                         }
                         values(tensorPointOrdinal) = value;
                       });
  Kokkos::fence();
  return values;
}

TensorCoefficientValueEvaluator::HostMatrix
TensorCoefficientValueEvaluator::volumeValuesAsConfigurationVelocityMatrix1D2V(const TensorCoefficientMap &coefficientMap,
                                                                               const TensorBasisCacheND &basisCache,
                                                                               const std::vector<double> &globalCoefficients,
                                                                               const TensorMeshTopology::TensorCellTuple &cellTuple,
                                                                               const std::vector<int> &configurationComponentOrdinals,
                                                                               const std::vector<int> &velocityComponentOrdinals)
{
  validateComponentSplit(configurationComponentOrdinals, velocityComponentOrdinals, basisCache.componentCount());
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(velocityComponentOrdinals.size()) != 2, std::invalid_argument,
                             "TensorCoefficientValueEvaluator 1D2V matrix output requires exactly two velocity components.");

  auto tensorValues = volumeValuesOnTensorCell(coefficientMap, basisCache, globalCoefficients, cellTuple);
  const IndexType configurationEntryCount = pointCountForComponents(configurationComponentOrdinals, basisCache);
  const IndexType velocityEntryCount = pointCountForComponents(velocityComponentOrdinals, basisCache);
  HostMatrix values("tensor coefficient configuration-velocity values", configurationEntryCount, velocityEntryCount);

  for (IndexType tensorPointOrdinal=0; tensorPointOrdinal<basisCache.tensorPointCount(); tensorPointOrdinal++)
  {
    const auto pointOrdinals = componentPointOrdinals(tensorPointOrdinal, basisCache);
    const IndexType configurationOrdinal = packedOrdinalForComponents(configurationComponentOrdinals, pointOrdinals, basisCache);
    const IndexType velocityOrdinal = packedOrdinalForComponents(velocityComponentOrdinals, pointOrdinals, basisCache);
    values(configurationOrdinal, velocityOrdinal) = tensorValues(tensorPointOrdinal);
  }
  return values;
}

TensorCoefficientValueEvaluator::HostMatrix
TensorCoefficientValueEvaluator::volumeValuesAsConfigurationVelocityMatrix1D2V(const TensorCoefficientMap &coefficientMap,
                                                                               const TensorBasisCacheND &basisCache,
                                                                               const OwnedCoefficientValues &ownedValues,
                                                                               const TensorMeshTopology::TensorCellTuple &cellTuple,
                                                                               const std::vector<int> &configurationComponentOrdinals,
                                                                               const std::vector<int> &velocityComponentOrdinals)
{
  validateComponentSplit(configurationComponentOrdinals, velocityComponentOrdinals, basisCache.componentCount());
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(velocityComponentOrdinals.size()) != 2, std::invalid_argument,
                             "TensorCoefficientValueEvaluator 1D2V matrix output requires exactly two velocity components.");

  auto tensorValues = volumeValuesOnTensorCell(coefficientMap, basisCache, ownedValues, cellTuple);
  const IndexType configurationEntryCount = pointCountForComponents(configurationComponentOrdinals, basisCache);
  const IndexType velocityEntryCount = pointCountForComponents(velocityComponentOrdinals, basisCache);
  HostMatrix values("tensor coefficient configuration-velocity values from owned plus ghost coefficients", configurationEntryCount, velocityEntryCount);

  for (IndexType tensorPointOrdinal=0; tensorPointOrdinal<basisCache.tensorPointCount(); tensorPointOrdinal++)
  {
    const auto pointOrdinals = componentPointOrdinals(tensorPointOrdinal, basisCache);
    const IndexType configurationOrdinal = packedOrdinalForComponents(configurationComponentOrdinals, pointOrdinals, basisCache);
    const IndexType velocityOrdinal = packedOrdinalForComponents(velocityComponentOrdinals, pointOrdinals, basisCache);
    values(configurationOrdinal, velocityOrdinal) = tensorValues(tensorPointOrdinal);
  }
  return values;
}
