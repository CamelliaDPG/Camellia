#ifndef Camellia_TensorMomentDescriptor_h
#define Camellia_TensorMomentDescriptor_h

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace Camellia
{
  /**
   * \brief Metadata-only description of a tensor-native reduction of a Vlasov solution.
   *
   * This descriptor is intentionally not an evaluator.  It records the exact tensor-product
   * quadrature intent, the velocity-component powers used by legacy solutionMomentFunction(...)
   * calls, and the optional relativistic velocity/current map needed by Planar MITL diagnostics
   * and Maxwell right-hand-side assembly.  The actual gather/evaluate/cache path remains a
   * follow-on adapter task.
   */
  class TensorMomentDescriptor
  {
    std::string _key;
    std::vector<int> _velocityComponentPowers;
    bool _usesRelativisticVelocityMap = false;
    int _relativisticVelocityComponentOrdinal = -1;
    double _chargeWeight = 1.0;
    bool _requiresExactTensorQuadrature = true;

    void validate() const
    {
      if (_key.empty()) throw std::invalid_argument("TensorMomentDescriptor requires a non-empty key.");
      if (_velocityComponentPowers.empty()) throw std::invalid_argument("TensorMomentDescriptor requires at least one velocity component.");
      for (const int power : _velocityComponentPowers)
      {
        if (power < 0) throw std::invalid_argument("TensorMomentDescriptor velocity powers must be nonnegative.");
      }
      if (_usesRelativisticVelocityMap)
      {
        if ((_relativisticVelocityComponentOrdinal < 0) ||
            (_relativisticVelocityComponentOrdinal >= static_cast<int>(_velocityComponentPowers.size())))
        {
          throw std::invalid_argument("TensorMomentDescriptor relativistic velocity component is out of range.");
        }
      }
    }

  public:
    TensorMomentDescriptor() = default;

    TensorMomentDescriptor(const std::string &key,
                           const std::vector<int> &velocityComponentPowers,
                           bool usesRelativisticVelocityMap = false,
                           int relativisticVelocityComponentOrdinal = -1,
                           double chargeWeight = 1.0,
                           bool requiresExactTensorQuadrature = true)
    : _key(key),
      _velocityComponentPowers(velocityComponentPowers),
      _usesRelativisticVelocityMap(usesRelativisticVelocityMap),
      _relativisticVelocityComponentOrdinal(relativisticVelocityComponentOrdinal),
      _chargeWeight(chargeWeight),
      _requiresExactTensorQuadrature(requiresExactTensorQuadrature)
    {
      validate();
    }

    const std::string &key() const { return _key; }
    int velocityDim() const { return static_cast<int>(_velocityComponentPowers.size()); }
    const std::vector<int> &velocityComponentPowers() const { return _velocityComponentPowers; }
    int velocityComponentPower(int velocityComponentOrdinal) const { return _velocityComponentPowers.at(velocityComponentOrdinal); }
    int totalVelocityMomentOrder() const { return std::accumulate(_velocityComponentPowers.begin(), _velocityComponentPowers.end(), 0); }
    bool isDensityMoment() const { return totalVelocityMomentOrder() == 0 && !_usesRelativisticVelocityMap; }
    bool usesRelativisticVelocityMap() const { return _usesRelativisticVelocityMap; }
    int relativisticVelocityComponentOrdinal() const { return _relativisticVelocityComponentOrdinal; }
    double chargeWeight() const { return _chargeWeight; }
    bool requiresExactTensorQuadrature() const { return _requiresExactTensorQuadrature; }

    static TensorMomentDescriptor density(const int velocityDim,
                                          const std::string &key = "density")
    {
      return TensorMomentDescriptor(key, std::vector<int>(velocityDim, 0));
    }

    static TensorMomentDescriptor properVelocityMoment(const std::string &key,
                                                       const std::vector<int> &velocityComponentPowers)
    {
      return TensorMomentDescriptor(key, velocityComponentPowers);
    }

    static TensorMomentDescriptor relativisticCurrent(const std::string &key,
                                                      const std::vector<int> &velocityComponentPowers,
                                                      const int velocityComponentOrdinal,
                                                      const double chargeWeight)
    {
      return TensorMomentDescriptor(key, velocityComponentPowers, true, velocityComponentOrdinal, chargeWeight);
    }

    static TensorMomentDescriptor planar1D2VDensity()
    {
      return density(2, "density");
    }

    static TensorMomentDescriptor planar1D2VXProperVelocityMoment()
    {
      return properVelocityMoment("ux_moment", {1, 0});
    }

    static TensorMomentDescriptor planar1D2VSecondProperVelocityMoment()
    {
      return properVelocityMoment("uy_or_uz_moment", {0, 1});
    }

    static TensorMomentDescriptor planar1D2VCurrentX(const double chargeWeight)
    {
      return relativisticCurrent("Jx", {1, 0}, 0, chargeWeight);
    }

    static TensorMomentDescriptor planar1D2VCurrentZ(const double chargeWeight)
    {
      return relativisticCurrent("Jz", {0, 1}, 1, chargeWeight);
    }
  };
}

#endif
