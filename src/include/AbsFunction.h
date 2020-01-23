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
    
    TFunctionPtr<double> dx();
    TFunctionPtr<double> dy();
    TFunctionPtr<double> dz();
    
    void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache);
    void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
    bool boundaryValueOnly();
    
    string displayString();
    
    std::vector< TFunctionPtr<double> > memberFunctions();
  };
}

#endif /* defined(__Camellia__AbsFunction__) */
