//
//  AbsFunction.h
//  Camellia
//
//  Created by Nate Roberts on 9/12/16.
//
//

#ifndef __Camellia__AbsFunction__
#define __Camellia__AbsFunction__

#include "Function.h"

namespace Camellia
{
  class AbsFunction : public TFunction<double>
  {
    TFunctionPtr<double> _f;
  public:
    AbsFunction(TFunctionPtr<double> f);
    
    TFunctionPtr<double> dx() override;
    TFunctionPtr<double> dy() override;
    TFunctionPtr<double> dz() override;
    
    using TFunction<double>::values;
    void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
    bool boundaryValueOnly() override;
    
    string displayString() override;
    
    std::vector< TFunctionPtr<double> > memberFunctions() override;
    
    static FunctionPtr abs(TFunctionPtr<double> f);
  };
}

#endif /* defined(__Camellia__AbsFunction__) */
