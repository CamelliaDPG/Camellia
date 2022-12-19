// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//  ExpFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_MaxwellianFunction_h
#define Camellia_MaxwellianFunction_h

#include "Function.h"
#include "SimpleFunction.h"
#include "TypeDefs.h"

namespace Camellia
{
  class MaxwellianFunction : public TFunction<double>
  {
    int _firstVelocityDimension;
    int _numVelocityDims;
    
    Kokkos::Array<double,3> _u;
    double _sigmaSquared;
    double _n;
  public:
    MaxwellianFunction(const int firstVelocityDimensionZeroBased, const int numVelocityDims, std::vector<double> u, double sigmaSquared, double n);
    
    using TFunction<double>::values;
    void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
    
    //! derivative of this function in the one-based i dimension.  Will be zero for dimensions below (_firstVelocityDimension + 1), and non-zero for dimensions (_firstVelocityDimension + 1) through (_firstVelocityDimension + _numVelocityDims).
    TFunctionPtr<double> di(int i) override;
    
    static FunctionPtr maxwellian(const int firstVelocityDimensionZeroBased, const int numVelocityDims, std::vector<double> u, double sigmaSquared, double n);
    
    static inline double momentNormalized1D(const double ux, const double sigmaSquared, const int momentNumber)
    {
      // see https://en.wikipedia.org/wiki/Normal_distribution#Moments
      using std::pow;
      switch(momentNumber)
      {
        case 0: return 1;
        case 1: return ux;
        case 2: return pow(ux,2.) + sigmaSquared;
        case 3: return pow(ux,3.) +  3. * pow(ux,1.) * sigmaSquared;
        case 4: return pow(ux,4.) +  6. * pow(ux,2.) * sigmaSquared +   3. * pow(sigmaSquared,2.);
        case 5: return pow(ux,5.) + 10. * pow(ux,3.) * sigmaSquared +  15. * pow(ux,1.) * pow(sigmaSquared,2.);
        case 6: return pow(ux,6.) + 15. * pow(ux,4.) * sigmaSquared +  45. * pow(ux,2.) * pow(sigmaSquared,2.) +  15. * pow(sigmaSquared,3.);
        case 7: return pow(ux,7.) + 21. * pow(ux,5.) * sigmaSquared + 105. * pow(ux,3.) * pow(sigmaSquared,2.) + 105. * pow(ux,1.) * pow(sigmaSquared,3.);
        case 8: return pow(ux,8.) + 28. * pow(ux,6.) * sigmaSquared + 210. * pow(ux,4.) * pow(sigmaSquared,2.) + 420. * pow(ux,2.) * pow(sigmaSquared,3.) + 105. * pow(sigmaSquared,4.);
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled moment number");
      }
    }

    static inline double moment(const double n, const double ux, const double uy, const double uz, const double sigmaSquared, const int a, const int b, const int c)
    {
      const double xMoment = momentNormalized1D(ux, sigmaSquared, a);
      const double yMoment = momentNormalized1D(uy, sigmaSquared, b);
      const double zMoment = momentNormalized1D(uz, sigmaSquared, c);
      
      return n * xMoment * yMoment * zMoment;
    }
  };
}
#endif
