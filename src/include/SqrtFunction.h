// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//  SqrtFunction.h
//  Camellia
//
//  Created by Nate Roberts on 3/17/16.
//
//

#ifndef Camellia_SqrtFunction_h
#define Camellia_SqrtFunction_h

#include "Function.h"

namespace Camellia {
  template <typename Scalar>
  class SqrtFunction : public TFunction<Scalar>
  {
    TFunctionPtr<Scalar> _f;
  public:
    SqrtFunction(TFunctionPtr<Scalar> f)
    : TFunction<Scalar>("SqrtFunction")
    {
      _f = f;
    }
    ~SqrtFunction() {}
    using TFunction<Scalar>::values;
    void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache) override
    {
      _f->values(values, basisCache);
      auto sqrtOp = KOKKOS_LAMBDA(const double &a) -> double
      {
#ifndef KOKKOS_ENABLE_CUDA
        using std::sqrt;
#endif
        return sqrt(a);
      };
      values.applyOperator(sqrtOp);
    }
    void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache) override
    {
      _f->values(values,basisCache);
      for (int i=0; i<values.size(); i++)
      {
        values[i] = sqrt(values[i]);
      }
    }
    std::vector<Intrepid2::DimensionInfo> valuesDataDims(BasisCachePtr basisCache) override
    {
      return _f->valuesDataDims(basisCache);
    }
    TFunctionPtr<Scalar> dx() override
    {
      TFunctionPtr<Scalar> sqrt_f = Teuchos::rcp( new SqrtFunction(_f) );
      return 0.5 * _f->dx() / sqrt_f;
    }
    TFunctionPtr<Scalar> dy() override
    {
      TFunctionPtr<Scalar> sqrt_f = Teuchos::rcp( new SqrtFunction(_f) );
      return 0.5 * _f->dy() / sqrt_f;
    }
    TFunctionPtr<Scalar> dz() override
    {
      TFunctionPtr<Scalar> sqrt_f = Teuchos::rcp( new SqrtFunction(_f) );
      return 0.5 * _f->dz() / sqrt_f;
    }
  };
}


#endif
