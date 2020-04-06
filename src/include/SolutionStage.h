//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  SolutionStage.h
//  Camellia
//
//  Created by Roberts, Nathan V on 3/25/20.
//

#ifndef SolutionStage_h
#define SolutionStage_h

#include "TypeDefs.h"

namespace Camellia
{
  class SolutionStage {
    int _solutionStage;
    ERKTableauPtr _tableau;
    ParameterFunctionPtr _t;
    ParameterFunctionPtr _dt;
  public:
    SolutionStage(ERKTableauPtr tableau, double initialTimeStep);
    
    //! 1-based stage
    int getStage() const;
    
    //! set 1-based stage
    void setStage(int solutionStage);
    
    ERKTableauPtr tableau() const;
    
    //! Dynamic time function
    FunctionPtr t() const;
    double getTime() const;
    void setTime(double t);
    
    //! Dynamic time step (will always be accurate even if time step changes)
    FunctionPtr dt() const;
    double getTimeStep() const;
    void setTimeStep(double dt);
    
    static SolutionStagePtr solutionStage(ERKTableauPtr tableau, double initialTimeStep);
  };
}



#endif /* SolutionStage_h */
