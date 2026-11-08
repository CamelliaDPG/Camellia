#ifndef Camellia_TensorFixedPointLifecycle_h
#define Camellia_TensorFixedPointLifecycle_h

#include <stdexcept>

namespace Camellia
{
  enum TensorFixedPointInitialGuessPolicy
  {
    TENSOR_FIXED_POINT_INITIAL_GUESS_ZERO,
    TENSOR_FIXED_POINT_INITIAL_GUESS_PREVIOUS_SLAB
  };

  enum TensorCoefficientCacheScope
  {
    TENSOR_COEFFICIENT_CACHE_PER_FIXED_POINT_SUBSTEP,
    TENSOR_COEFFICIENT_CACHE_PER_TIME_SLAB
  };

  /**
   * \brief Metadata-only contract for Vlasov/Maxwell fixed-point coupling.
   *
   * This object records the lifecycle assumptions needed by solution-dependent tensor
   * coefficients without owning any solution data or cache storage.  It mirrors the
   * Planar MITL driver knobs: minimum/maximum substeps and relative L2 tolerance in f.
   * The default tensor-native contract starts from a zero field guess and scopes
   * solution-dependent coefficient caches to one fixed-point substep, invalidating them
   * when the Maxwell/configuration-space solution changes.
   */
  class TensorFixedPointLifecycle
  {
    TensorFixedPointInitialGuessPolicy _initialGuessPolicy = TENSOR_FIXED_POINT_INITIAL_GUESS_ZERO;
    int _minSubsteps = 1;
    int _maxSubsteps = 1;
    double _relativeL2Tolerance = 0.0;
    TensorCoefficientCacheScope _coefficientCacheScope = TENSOR_COEFFICIENT_CACHE_PER_FIXED_POINT_SUBSTEP;
    bool _invalidateCoefficientCachesWhenConfigurationSolutionChanges = true;

    void validate() const
    {
      if (_minSubsteps < 0) throw std::invalid_argument("TensorFixedPointLifecycle minSubsteps must be nonnegative.");
      if (_maxSubsteps < _minSubsteps) throw std::invalid_argument("TensorFixedPointLifecycle maxSubsteps must be at least minSubsteps.");
      if (_relativeL2Tolerance < 0.0) throw std::invalid_argument("TensorFixedPointLifecycle tolerance must be nonnegative.");
    }

  public:
    TensorFixedPointLifecycle() = default;

    TensorFixedPointLifecycle(TensorFixedPointInitialGuessPolicy initialGuessPolicy,
                              int minSubsteps,
                              int maxSubsteps,
                              double relativeL2Tolerance,
                              TensorCoefficientCacheScope coefficientCacheScope = TENSOR_COEFFICIENT_CACHE_PER_FIXED_POINT_SUBSTEP,
                              bool invalidateCoefficientCachesWhenConfigurationSolutionChanges = true)
    : _initialGuessPolicy(initialGuessPolicy),
      _minSubsteps(minSubsteps),
      _maxSubsteps(maxSubsteps),
      _relativeL2Tolerance(relativeL2Tolerance),
      _coefficientCacheScope(coefficientCacheScope),
      _invalidateCoefficientCachesWhenConfigurationSolutionChanges(invalidateCoefficientCachesWhenConfigurationSolutionChanges)
    {
      validate();
    }

    TensorFixedPointInitialGuessPolicy initialGuessPolicy() const { return _initialGuessPolicy; }
    int minSubsteps() const { return _minSubsteps; }
    int maxSubsteps() const { return _maxSubsteps; }
    double relativeL2Tolerance() const { return _relativeL2Tolerance; }
    TensorCoefficientCacheScope coefficientCacheScope() const { return _coefficientCacheScope; }
    bool invalidateCoefficientCachesWhenConfigurationSolutionChanges() const { return _invalidateCoefficientCachesWhenConfigurationSolutionChanges; }

    bool startsFromZeroFieldGuess() const { return _initialGuessPolicy == TENSOR_FIXED_POINT_INITIAL_GUESS_ZERO; }
    bool cachesAreScopedToFixedPointSubstep() const { return _coefficientCacheScope == TENSOR_COEFFICIENT_CACHE_PER_FIXED_POINT_SUBSTEP; }

    static TensorFixedPointLifecycle planarMITLDefaults(int minSubsteps,
                                                        int maxSubsteps,
                                                        double relativeL2Tolerance)
    {
      return TensorFixedPointLifecycle(TENSOR_FIXED_POINT_INITIAL_GUESS_ZERO,
                                       minSubsteps,
                                       maxSubsteps,
                                       relativeL2Tolerance,
                                       TENSOR_COEFFICIENT_CACHE_PER_FIXED_POINT_SUBSTEP,
                                       true);
    }
  };
}

#endif
