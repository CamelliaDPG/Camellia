// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

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

namespace Camellia
{
template <typename Scalar>
class PolarizedFunction : public TFunction<Scalar>   // takes a 2D Function of x and y, interpreting it as function of r and theta
{
  // i.e. to implement f(r,theta) = r sin theta
  // pass in a Function f(x,y) = x sin y.
  // Given the implementation, it is important that f depend *only* on x and y, and not on the mesh, etc.
  // (the only method in BasisCache that f may call is getPhysicalCubaturePointsFC())
  TFunctionPtr<Scalar> _f;
public:
  PolarizedFunction( TFunctionPtr<Scalar> f_of_xAsR_yAsTheta );
  
  using TFunction<Scalar>::values;
  void values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;

  TFunctionPtr<Scalar> dx() override;
  TFunctionPtr<Scalar> dy() override;

  Teuchos::RCP<PolarizedFunction<Scalar> > dtheta();
  Teuchos::RCP<PolarizedFunction<Scalar> > dr();
  
  std::vector< TFunctionPtr<Scalar> > memberFunctions() override;

  virtual string displayString() override; // for PolarizedFunction, this should be _f->displayString() + "(r,theta)";

  using TFunction<double>::isZero;
  bool isZero() override;

  static Teuchos::RCP<PolarizedFunction<double> > r();
  static Teuchos::RCP<PolarizedFunction<double> > sin_theta();
  static Teuchos::RCP<PolarizedFunction<double> > cos_theta();
};
}

#endif
