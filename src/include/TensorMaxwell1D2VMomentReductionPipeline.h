#ifndef Camellia_TensorMaxwell1D2VMomentReductionPipeline_h
#define Camellia_TensorMaxwell1D2VMomentReductionPipeline_h

#include <vector>

#include "TensorCoefficientValueEvaluator.h"
#include "TensorMaxwellRHSMomentData.h"
#include "TensorMomentReduction.h"
#include "TensorSolution.h"

namespace Camellia
{
  /**
   * \brief Narrow planar 1D2V pipeline from tensor-native Vlasov coefficient values
   * to the typed Maxwell RHS moment handoff container.
   *
   * Each species entry supplies a configuration-by-velocity host matrix of sampled
   * Vlasov solution coefficients.  This pipeline performs the same density,
   * x-proper-velocity, and retained z-proper-velocity reductions recorded by
   * TensorMomentDescriptor and then applies TensorMaxwell1D2VMomentAdapter's
   * Maxwell-side accumulation algebra.  It deliberately remains below the production
   * FunctionPtr/RHS layer; the output is explicit per-configuration-point scalar data.
   */
  class TensorMaxwell1D2VMomentReductionPipeline
  {
  public:
    using HostVector = TensorMomentReduction::HostVector;
    using HostMatrix = TensorMomentReduction::HostMatrix;

    static TensorPlanar1D2VSpeciesMoments reduceSpeciesMoments(HostMatrix vlasovSolutionValues,
                                                               HostMatrix velocityPoints,
                                                               HostVector velocityWeights)
    {
      const int configurationEntryCount = static_cast<int>(vlasovSolutionValues.extent(0));
      HostVector density("TensorMaxwell1D2VMomentReductionPipeline density", configurationEntryCount);
      HostVector xMoment("TensorMaxwell1D2VMomentReductionPipeline x moment", configurationEntryCount);
      HostVector zMoment("TensorMaxwell1D2VMomentReductionPipeline z moment", configurationEntryCount);

      TensorMomentReduction::reduceHost(TensorMomentDescriptor::planar1D2VDensity(),
                                        vlasovSolutionValues,
                                        velocityPoints,
                                        velocityWeights,
                                        density);
      TensorMomentReduction::reduceHost(TensorMomentDescriptor::planar1D2VXProperVelocityMoment(),
                                        vlasovSolutionValues,
                                        velocityPoints,
                                        velocityWeights,
                                        xMoment);
      TensorMomentReduction::reduceHost(TensorMomentDescriptor::planar1D2VSecondProperVelocityMoment(),
                                        vlasovSolutionValues,
                                        velocityPoints,
                                        velocityWeights,
                                        zMoment);

      std::vector<double> densityValues(configurationEntryCount);
      std::vector<double> xMomentValues(configurationEntryCount);
      std::vector<double> zMomentValues(configurationEntryCount);
      for (int entry=0; entry<configurationEntryCount; entry++)
      {
        densityValues[entry] = density(entry);
        xMomentValues[entry] = xMoment(entry);
        zMomentValues[entry] = zMoment(entry);
      }

      return TensorPlanar1D2VSpeciesMoments(densityValues, xMomentValues, zMomentValues);
    }

    static TensorMaxwellRHSMomentData buildRHSMomentData(int numConfigurationCells,
                                                         int numPointsPerConfigurationCell,
                                                         const std::vector<double> &speciesCharges,
                                                         double eps0,
                                                         const std::vector<HostMatrix> &speciesVlasovSolutionValues,
                                                         HostMatrix velocityPoints,
                                                         HostVector velocityWeights,
                                                         const std::vector<double> &n0Values)
    {
      if (speciesCharges.size() != speciesVlasovSolutionValues.size())
      {
        throw std::invalid_argument("TensorMaxwell1D2VMomentReductionPipeline species charge/value count mismatch.");
      }

      std::vector<TensorPlanar1D2VSpeciesMoments> speciesMoments;
      speciesMoments.reserve(speciesVlasovSolutionValues.size());
      for (const auto &values : speciesVlasovSolutionValues)
      {
        speciesMoments.push_back(reduceSpeciesMoments(values, velocityPoints, velocityWeights));
      }

      return TensorMaxwell1D2VMomentAdapter::buildRHSMomentData(numConfigurationCells,
                                                                numPointsPerConfigurationCell,
                                                                speciesCharges,
                                                                eps0,
                                                                speciesMoments,
                                                                n0Values);
    }

    static TensorMaxwellRHSMomentData buildRHSMomentDataFromOwnedCoefficientValues(
        int numConfigurationCells,
        int numPointsPerConfigurationCell,
        const std::vector<double> &speciesCharges,
        double eps0,
        const std::vector<TensorCoefficientValueEvaluator::OwnedCoefficientValues> &speciesOwnedCoefficientValues,
        const TensorCoefficientMap &coefficientMap,
        const TensorBasisCacheND &basisCache,
        const TensorMeshTopology::TensorCellTuple &cellTuple,
        const std::vector<int> &configurationComponentOrdinals,
        const std::vector<int> &velocityComponentOrdinals,
        HostMatrix velocityPoints,
        HostVector velocityWeights,
        const std::vector<double> &n0Values)
    {
      std::vector<HostMatrix> speciesVlasovSolutionValues;
      speciesVlasovSolutionValues.reserve(speciesOwnedCoefficientValues.size());
      for (const auto &ownedValues : speciesOwnedCoefficientValues)
      {
        speciesVlasovSolutionValues.push_back(TensorCoefficientValueEvaluator::volumeValuesAsConfigurationVelocityMatrix1D2V(coefficientMap,
                                                                                                                             basisCache,
                                                                                                                             ownedValues,
                                                                                                                             cellTuple,
                                                                                                                             configurationComponentOrdinals,
                                                                                                                             velocityComponentOrdinals));
      }

      return buildRHSMomentData(numConfigurationCells,
                                numPointsPerConfigurationCell,
                                speciesCharges,
                                eps0,
                                speciesVlasovSolutionValues,
                                velocityPoints,
                                velocityWeights,
                                n0Values);
    }

    static TensorMaxwellRHSMomentData buildRHSMomentDataFromTensorSolutions(
        int numConfigurationCells,
        int numPointsPerConfigurationCell,
        const std::vector<double> &speciesCharges,
        double eps0,
        const std::vector<TensorSolutionPtr> &speciesSolutions,
        TensorSpaceRole role,
        const std::vector<std::string> &speciesVariableNames,
        const TensorBasisCacheND &basisCache,
        const TensorMeshTopology::TensorCellTuple &cellTuple,
        const std::vector<int> &configurationComponentOrdinals,
        const std::vector<int> &velocityComponentOrdinals,
        HostMatrix velocityPoints,
        HostVector velocityWeights,
        const std::vector<double> &n0Values)
    {
      if (speciesCharges.size() != speciesSolutions.size())
      {
        throw std::invalid_argument("TensorMaxwell1D2VMomentReductionPipeline species charge/TensorSolution count mismatch.");
      }
      if (speciesCharges.size() != speciesVariableNames.size())
      {
        throw std::invalid_argument("TensorMaxwell1D2VMomentReductionPipeline species charge/variable-name count mismatch.");
      }

      std::vector<HostMatrix> speciesVlasovSolutionValues;
      speciesVlasovSolutionValues.reserve(speciesSolutions.size());
      for (size_t speciesOrdinal=0; speciesOrdinal<speciesSolutions.size(); speciesOrdinal++)
      {
        if (speciesSolutions[speciesOrdinal] == Teuchos::null)
        {
          throw std::invalid_argument("TensorMaxwell1D2VMomentReductionPipeline requires non-null TensorSolution species inputs.");
        }
        speciesVlasovSolutionValues.push_back(speciesSolutions[speciesOrdinal]->volumeValuesAsConfigurationVelocityMatrix1D2V(role,
                                                                                                                              speciesVariableNames[speciesOrdinal],
                                                                                                                              basisCache,
                                                                                                                              cellTuple,
                                                                                                                              configurationComponentOrdinals,
                                                                                                                              velocityComponentOrdinals));
      }

      return buildRHSMomentData(numConfigurationCells,
                                numPointsPerConfigurationCell,
                                speciesCharges,
                                eps0,
                                speciesVlasovSolutionValues,
                                velocityPoints,
                                velocityWeights,
                                n0Values);
    }
  };
}

#endif
