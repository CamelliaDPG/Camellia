#ifndef Camellia_TensorCoefficientValueEvaluator_h
#define Camellia_TensorCoefficientValueEvaluator_h

#include <map>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "TensorBasisCache.h"
#include "TensorCoefficientMap.h"
// TensorMaxwell1D2VMomentReductionPipeline intentionally includes this evaluator;
// keep this header independent to avoid an include cycle.

namespace Camellia
{
  /**
   * \brief Host-side bridge from tensor coefficient storage to quadrature value data.
   *
   * This is a deliberately narrow, read-only adapter for the first tensor-native
   * Vlasov moment handoff.  It evaluates one volume tensor variable on one tensor
   * cell using the existing TensorCoefficientMap numbering and TensorBasisCacheND
   * basis/quadrature data, and can reshape the flat tensor-point values into the
   * configuration-by-velocity matrix consumed by TensorMaxwell1D2VMomentReductionPipeline.
   */
  class TensorCoefficientValueEvaluator
  {
  public:
    using HostVector = Kokkos::View<double*, Kokkos::HostSpace>;
    using HostMatrix = Kokkos::View<double**, Kokkos::HostSpace>;
    using DeviceVector = Kokkos::View<double*>;
    using DeviceIndexVector = Kokkos::View<IndexType*>;
    using DeviceIntVector = Kokkos::View<int*>;
    using DeviceBasisValueTensor = Kokkos::View<double***>;

    /**
     * \brief Rank-local coefficient ownership view prepared by the distributed import layer.
     *
     * This is intentionally a plain host-side contract object: callers own any Tpetra/MPI
     * import/export work needed to populate `ghostCoefficientsByGlobalIndex`.  The evaluator
     * consumes only this explicit owned-plus-ghost view and still fails fast if a cell-local
     * nonowned coefficient was not imported.
     */
    struct OwnedCoefficientValues
    {
      using DeviceScalarView = Kokkos::View<double*>;
      using DeviceIndexView = Kokkos::View<IndexType*>;

      std::pair<IndexType,IndexType> ownedRange;
      std::vector<double> ownedCoefficients;
      std::map<IndexType,double> ghostCoefficientsByGlobalIndex;

      DeviceScalarView ownedCoefficientsDevice;
      DeviceIndexView ghostGlobalIndicesDevice;
      DeviceScalarView ghostCoefficientsDevice;

      OwnedCoefficientValues() = default;
      OwnedCoefficientValues(const std::pair<IndexType,IndexType> &ownedRange,
                             const std::vector<double> &ownedCoefficients,
                             const std::map<IndexType,double> &ghostCoefficientsByGlobalIndex);

      bool hasDeviceMirrors() const;
      void populateDeviceMirrors();
    };

    static HostVector volumeValuesOnTensorCell(const TensorCoefficientMap &coefficientMap,
                                               const TensorBasisCacheND &basisCache,
                                               const std::vector<double> &globalCoefficients,
                                               const TensorMeshTopology::TensorCellTuple &cellTuple);

    /**
     * \brief Evaluate using production-style coefficient ownership semantics.
     *
     * `ownedRange` is the half-open global range owned by the caller and `ownedCoefficients`
     * is indexed relative to `ownedRange.first`.  Any cell-local coefficient outside that
     * range must be present in `ghostCoefficientsByGlobalIndex`.  Missing ghost values fail
     * immediately rather than silently reading from workset-ordered or all-global storage.
     */
    static HostVector volumeValuesOnTensorCell(const TensorCoefficientMap &coefficientMap,
                                               const TensorBasisCacheND &basisCache,
                                               const std::pair<IndexType,IndexType> &ownedRange,
                                               const std::vector<double> &ownedCoefficients,
                                               const std::map<IndexType,double> &ghostCoefficientsByGlobalIndex,
                                               const TensorMeshTopology::TensorCellTuple &cellTuple);

    static HostVector volumeValuesOnTensorCell(const TensorCoefficientMap &coefficientMap,
                                               const TensorBasisCacheND &basisCache,
                                               const OwnedCoefficientValues &ownedValues,
                                               const TensorMeshTopology::TensorCellTuple &cellTuple);

    static DeviceVector volumeValuesOnTensorCellDevice(const TensorCoefficientMap &coefficientMap,
                                                       const TensorBasisCacheND &basisCache,
                                                       const OwnedCoefficientValues &ownedValues,
                                                       const TensorMeshTopology::TensorCellTuple &cellTuple);

    static HostMatrix volumeValuesAsConfigurationVelocityMatrix1D2V(const TensorCoefficientMap &coefficientMap,
                                                                    const TensorBasisCacheND &basisCache,
                                                                    const std::vector<double> &globalCoefficients,
                                                                    const TensorMeshTopology::TensorCellTuple &cellTuple,
                                                                    const std::vector<int> &configurationComponentOrdinals,
                                                                    const std::vector<int> &velocityComponentOrdinals);

    static HostMatrix volumeValuesAsConfigurationVelocityMatrix1D2V(const TensorCoefficientMap &coefficientMap,
                                                                    const TensorBasisCacheND &basisCache,
                                                                    const OwnedCoefficientValues &ownedValues,
                                                                    const TensorMeshTopology::TensorCellTuple &cellTuple,
                                                                    const std::vector<int> &configurationComponentOrdinals,
                                                                    const std::vector<int> &velocityComponentOrdinals);
  };
}

#endif
