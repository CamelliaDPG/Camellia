#ifndef Camellia_TensorMomentReduction_h
#define Camellia_TensorMomentReduction_h

#include <cmath>
#include <stdexcept>

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

#include "TensorMomentDescriptor.h"

namespace Camellia
{
  class TensorMomentReduction
  {
  public:
    using HostVector = Kokkos::View<double*, Kokkos::HostSpace>;
    using HostMatrix = Kokkos::View<double**, Kokkos::HostSpace>;
    using DeviceVector = Kokkos::View<double*>;
    using DeviceMatrix = Kokkos::View<double**>;

    static double velocityWeight(const TensorMomentDescriptor &descriptor,
                                 HostMatrix velocityPoints,
                                 int velocityPointOrdinal,
                                 double lightSpeed = 1.0)
    {
      if (descriptor.usesRelativisticVelocityMap())
      {
        if (lightSpeed <= 0.0) throw std::invalid_argument("TensorMomentReduction requires positive light speed for relativistic moments.");
        double uSquared = 0.0;
        for (int d=0; d<descriptor.velocityDim(); d++)
        {
          const double u = velocityPoints(velocityPointOrdinal,d);
          uSquared += u * u;
        }
        const double gamma = std::sqrt(1.0 + uSquared / (lightSpeed * lightSpeed));
        return descriptor.chargeWeight() * velocityPoints(velocityPointOrdinal, descriptor.relativisticVelocityComponentOrdinal()) / gamma;
      }

      double weight = 1.0;
      for (int d=0; d<descriptor.velocityDim(); d++)
      {
        const double u = velocityPoints(velocityPointOrdinal,d);
        for (int p=0; p<descriptor.velocityComponentPower(d); p++) weight *= u;
      }
      return weight;
    }

    static void reduceHost(const TensorMomentDescriptor &descriptor,
                           HostMatrix vlasovSolutionValues,
                           HostMatrix velocityPoints,
                           HostVector velocityWeights,
                           HostVector momentValuesOut,
                           double lightSpeed = 1.0)
    {
      const int configurationPointCount = static_cast<int>(vlasovSolutionValues.extent(0));
      const int velocityPointCount = static_cast<int>(vlasovSolutionValues.extent(1));
      if (static_cast<int>(velocityPoints.extent(0)) != velocityPointCount) throw std::invalid_argument("TensorMomentReduction velocity-point count mismatch.");
      if (static_cast<int>(velocityPoints.extent(1)) != descriptor.velocityDim()) throw std::invalid_argument("TensorMomentReduction velocity dimension mismatch.");
      if (static_cast<int>(velocityWeights.extent(0)) != velocityPointCount) throw std::invalid_argument("TensorMomentReduction velocity-weight count mismatch.");
      if (static_cast<int>(momentValuesOut.extent(0)) != configurationPointCount) throw std::invalid_argument("TensorMomentReduction output extent mismatch.");

      for (int c=0; c<configurationPointCount; c++)
      {
        double reducedValue = 0.0;
        for (int v=0; v<velocityPointCount; v++)
        {
          reducedValue += vlasovSolutionValues(c,v) * velocityWeights(v) * velocityWeight(descriptor, velocityPoints, v, lightSpeed);
        }
        momentValuesOut(c) = reducedValue;
      }
    }

    static void reduceDevicePlanar1D2V(const TensorMomentDescriptor &descriptor,
                                       DeviceMatrix vlasovSolutionValues,
                                       DeviceMatrix velocityPoints,
                                       DeviceVector velocityWeights,
                                       DeviceVector momentValuesOut,
                                       double lightSpeed = 1.0)
    {
      if (descriptor.velocityDim() != 2) throw std::invalid_argument("TensorMomentReduction::reduceDevicePlanar1D2V currently supports only planar 1D2V descriptors.");
      if (descriptor.usesRelativisticVelocityMap() && (lightSpeed <= 0.0)) throw std::invalid_argument("TensorMomentReduction requires positive light speed for relativistic moments.");

      const int configurationPointCount = static_cast<int>(vlasovSolutionValues.extent(0));
      const int velocityPointCount = static_cast<int>(vlasovSolutionValues.extent(1));
      if (static_cast<int>(velocityPoints.extent(0)) != velocityPointCount) throw std::invalid_argument("TensorMomentReduction velocity-point count mismatch.");
      if (static_cast<int>(velocityPoints.extent(1)) != descriptor.velocityDim()) throw std::invalid_argument("TensorMomentReduction velocity dimension mismatch.");
      if (static_cast<int>(velocityWeights.extent(0)) != velocityPointCount) throw std::invalid_argument("TensorMomentReduction velocity-weight count mismatch.");
      if (static_cast<int>(momentValuesOut.extent(0)) != configurationPointCount) throw std::invalid_argument("TensorMomentReduction output extent mismatch.");

      const int velocityPower0 = descriptor.velocityComponentPower(0);
      const int velocityPower1 = descriptor.velocityComponentPower(1);
      const bool usesRelativisticVelocityMap = descriptor.usesRelativisticVelocityMap();
      const int relativisticComponentOrdinal = descriptor.relativisticVelocityComponentOrdinal();
      const double chargeWeight = descriptor.chargeWeight();

      Kokkos::parallel_for("TensorMomentReduction::reduceDevicePlanar1D2V",
                           configurationPointCount,
                           KOKKOS_LAMBDA(const int c)
                           {
                             double reducedValue = 0.0;
                             for (int v=0; v<velocityPointCount; v++)
                             {
                               double momentWeight = 1.0;
                               if (usesRelativisticVelocityMap)
                               {
                                 const double u0 = velocityPoints(v,0);
                                 const double u1 = velocityPoints(v,1);
                                 const double gamma = Kokkos::sqrt(1.0 + (u0*u0 + u1*u1) / (lightSpeed * lightSpeed));
                                 momentWeight = chargeWeight * velocityPoints(v,relativisticComponentOrdinal) / gamma;
                               }
                               else
                               {
                                 const double u0 = velocityPoints(v,0);
                                 const double u1 = velocityPoints(v,1);
                                 for (int p=0; p<velocityPower0; p++) momentWeight *= u0;
                                 for (int p=0; p<velocityPower1; p++) momentWeight *= u1;
                               }
                               reducedValue += vlasovSolutionValues(c,v) * velocityWeights(v) * momentWeight;
                             }
                             momentValuesOut(c) = reducedValue;
                           });
      Kokkos::fence();
    }
  };
}

#endif
