//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include <iostream>

#include "ConstantScalarFunction.h"
#include "ParameterFunction.h"

using namespace Camellia;

ParameterFunction::ParameterFunction(double value) : TFunction<double>("ParameterFunction",0)
{
  setValue(value);
}

ParameterFunction::ParameterFunction(TFunctionPtr<double> fxn) : TFunction<double>("ParameterFunction",fxn->rank())
{
  setValue(fxn);
}

double ParameterFunction::getConstantValue() const
{
  auto scalarFxn = dynamic_cast<ConstantScalarFunction<double>*>(_fxn.get());
  if (scalarFxn == NULL)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getConstantValue() only supported for _fxn that is a subclass of ConstantScalarFunction<double>");
  }
  
  return scalarFxn->value();
}

TFunctionPtr<double> ParameterFunction::getValue() const
{
  return _fxn;
}

void ParameterFunction::setValue(TFunctionPtr<double> fxn)
{
  if ((_fxn.get() == NULL) || (fxn->rank() == _fxn->rank()))
  {
    _fxn = fxn;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ParameterFunction can't change rank!");
  }
}

void ParameterFunction::setValue(double value)
{
  _fxn = Function::constant(value);
}

TFunctionPtr<double> ParameterFunction::x()
{
  return _fxn->x();
}
TFunctionPtr<double> ParameterFunction::y()
{
  return _fxn->y();
}
TFunctionPtr<double> ParameterFunction::z()
{
  return _fxn->z();
}

TFunctionPtr<double> ParameterFunction::dx()
{
  return _fxn->dx();
}
TFunctionPtr<double> ParameterFunction::dy()
{
  return _fxn->dy();
}
TFunctionPtr<double> ParameterFunction::dz()
{
  return _fxn->dz();
}

TFunctionPtr<double> ParameterFunction::grad(int numComponents)
{
  return _fxn->grad(numComponents);
}
TFunctionPtr<double> ParameterFunction::div()
{
  return _fxn->div();
}

std::vector< TFunctionPtr<double> > ParameterFunction::memberFunctions()
{
  return {{_fxn}};
}

void ParameterFunction::values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("ParameterFunction values()");
  _fxn->values(values,modType,basisCache);
}

std::vector<Intrepid2::DimensionInfo> ParameterFunction::valuesDataDims(BasisCachePtr basisCache)
{
  return _fxn->valuesDataDims(basisCache);
}

void ParameterFunction::values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache)
{
  _fxn->values(values, basisCache);
}

bool ParameterFunction::boundaryValueOnly()
{
  return _fxn->boundaryValueOnly();
}

string ParameterFunction::displayString()
{
  if (_name == "")
  {
    return _fxn->displayString();
  }
  else
  {
    return _name;
  }
}

Teuchos::RCP<ParameterFunction> ParameterFunction::parameterFunction(double value)
{
  return Teuchos::rcp( new ParameterFunction(value) );
}
Teuchos::RCP<ParameterFunction> ParameterFunction::parameterFunction(TFunctionPtr<double> fxn)
{
  return Teuchos::rcp( new ParameterFunction(fxn) );
}

void ParameterFunction::setName(const std::string &name)
{
  _name = name;
}
