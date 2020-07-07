//
//  FunctionErrorIndicator.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 7/2/20.
//

#ifndef FunctionErrorIndicator_hpp
#define FunctionErrorIndicator_hpp

#include "ErrorIndicator.h"
#include "Mesh.h"
#include "TypeDefs.h"

namespace Camellia
{
  template <typename Scalar>
  class FunctionErrorIndicator : public ErrorIndicator
  {
    MeshPtr _mesh;
    TFunctionPtr<Scalar> _errorFunction;
    int _quadratureEnrichment;
  public:
    //! Uses L^2 norm of the specified function as error indicator.
    FunctionErrorIndicator(MeshPtr mesh, TFunctionPtr<Scalar> f, int quadratureEnrichment);

    //! determine rank-local error measures.  Populates ErrorIndicator::_localErrorMeasures.
    virtual void measureError();
  };
}


#endif /* FunctionErrorIndicator_hpp */
