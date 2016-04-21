//
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
    {
      _f = f;
    }
    ~SqrtFunction() {}
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
    {
      _f->values(values,basisCache);
      for (int i=0; i<values.size(); i++)
      {
        values[i] = sqrt(values[i]);
      }
    }
    TFunctionPtr<Scalar> dx()
    {
      TFunctionPtr<Scalar> sqrt_f = Teuchos::rcp( new SqrtFunction(_f) );
      return 0.5 * _f->dx() / sqrt_f;
    }
    TFunctionPtr<Scalar> dy()
    {
      TFunctionPtr<Scalar> sqrt_f = Teuchos::rcp( new SqrtFunction(_f) );
      return 0.5 * _f->dy() / sqrt_f;
    }
    TFunctionPtr<Scalar> dz()
    {
      TFunctionPtr<Scalar> sqrt_f = Teuchos::rcp( new SqrtFunction(_f) );
      return 0.5 * _f->dz() / sqrt_f;
    }
  };
}


#endif
