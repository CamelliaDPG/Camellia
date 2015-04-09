//
//  PolarizedFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/9/15.
//
//

#ifndef Camellia_PolarizedFunction_h
#define Camellia_PolarizedFunction_h

#include "Function.h"

namespace Camellia {
  class PolarizedFunction : public Function { // takes a 2D Function of x and y, interpreting it as function of r and theta
    // i.e. to implement f(r,theta) = r sin theta
    // pass in a Function f(x,y) = x sin y.
    // Given the implementation, it is important that f depend *only* on x and y, and not on the mesh, etc.
    // (the only method in BasisCache that f may call is getPhysicalCubaturePoints())
    FunctionPtr _f;
  public:
    PolarizedFunction( FunctionPtr f_of_xAsR_yAsTheta );
    void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
    
    FunctionPtr dx();
    FunctionPtr dy();
    
    Teuchos::RCP<PolarizedFunction> dtheta();
    Teuchos::RCP<PolarizedFunction> dr();
    
    virtual string displayString(); // for PolarizedFunction, this should be _f->displayString() + "(r,theta)";
    
    bool isZero();
    
    static Teuchos::RCP<PolarizedFunction> r();
    static Teuchos::RCP<PolarizedFunction> sin_theta();
    static Teuchos::RCP<PolarizedFunction> cos_theta();
  };
}

#endif
