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
  class SolutionStageObserver
  {
  public:
    virtual void setStage(const int stageNumber) = 0;
  };
  using SolutionStageObserverPtr = Teuchos::RCP<SolutionStageObserver>;
  
  class SolutionStage {
    int _solutionStage;
    ERKTableauPtr _tableau;
    ParameterFunctionPtr _t;
    ParameterFunctionPtr _dt;
    std::vector<SolutionStageObserverPtr> _solnStageObservers;
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
    ParameterFunctionPtr dt() const;
    double getTimeStep() const;
    void setTimeStep(double dt);
    
    //! add observer; will be called when setStage() is called.
    void addObserver(SolutionStageObserverPtr solnStageObserver);
    
    static SolutionStagePtr solutionStage(ERKTableauPtr tableau, double initialTimeStep);
  };
}



#endif /* SolutionStage_h */
