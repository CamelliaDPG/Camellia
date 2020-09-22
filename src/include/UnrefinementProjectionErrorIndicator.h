//
//  UnrefinementProjectionErrorIndicator.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 8/13/20.
//

#ifndef UnrefinementProjectionErrorIndicator_hpp
#define UnrefinementProjectionErrorIndicator_hpp

#include "ErrorIndicator.h"

namespace Camellia
{
  class UnrefinementProjectionErrorIndicator : public ErrorIndicator
  {
    SolutionPtr _soln;
    VarPtr _var;
  public:
    UnrefinementProjectionErrorIndicator(SolutionPtr soln, VarPtr var);
    
    void gatherParentError(const std::set<GlobalIndexType> &excludedChildren) override;
    
    //! measureError() throws an exception if called; this ErrorIndicator should only be used for unrefinements (parent error)
    void measureError() override;
  };
}

#endif /* UnrefinementProjectionErrorIndicator_hpp */
