//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "VarFunction.h"

#include "BasisCache.h"
#include "LinearTerm.h"
#include "Solution.h"

using namespace Camellia;
using namespace std;

template <typename Scalar>
VarFunction<Scalar>::VarFunction(VarPtr var) : TFunction<Scalar>(var->rank())
{
  _var = var;
}

template <typename Scalar>
TFunctionPtr<Scalar> VarFunction<Scalar>::abstractFunction(TLinearTermPtr<Scalar> lt)
{
  auto rank = lt->rank();
  auto sum = TFunction<Scalar>::zero(rank);
  const std::vector< TLinearSummand<Scalar> >& summands = lt->summands();
  for (auto & summand : summands)
  {
    auto weight = summand.first;
    auto var = summand.second;
    auto varFunction = VarFunction<Scalar>::abstractFunction(var);
    sum = sum + weight * varFunction;
  }
  return sum;
}

template <typename Scalar>
TFunctionPtr<Scalar> VarFunction<Scalar>::abstractFunction(VarPtr var)
{
  return Teuchos::rcp(new VarFunction(var));
}

template <typename Scalar>
bool VarFunction<Scalar>::boundaryValueOnly()
{
  return (_var->varType() == FLUX) || (_var->varType() == TRACE);
}

template <typename Scalar>
string VarFunction<Scalar>::displayString()
{
  ostringstream str;
  str << _var->displayString();
  return str.str();
}

//! evaluates, filling in _var values using soln
template <typename Scalar>
TFunctionPtr<Scalar> VarFunction<Scalar>::evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap)
{
  auto entry = valueMap.find(_var->ID());
  if (entry != valueMap.end())
  {
    return entry->second;
  }
  else
  {
    return TFunction<Scalar>::zero(this->rank());
  }
}

//! returns the LinearTerm corresponding to _var
template <typename Scalar>
TLinearTermPtr<Scalar> VarFunction<Scalar>::jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap)
{
  return 1.0 * _var;
}

//! returns true
template <typename Scalar>
bool VarFunction<Scalar>::isAbstract()
{
  return true;
}

template <typename Scalar>
TFunctionPtr<Scalar> VarFunction<Scalar>::dx()
{
  return Teuchos::rcp( new VarFunction(_var->dx()) );
}

template <typename Scalar>
TFunctionPtr<Scalar> VarFunction<Scalar>::dy()
{
  return Teuchos::rcp( new VarFunction(_var->dy()) );
}

template <typename Scalar>
TFunctionPtr<Scalar> VarFunction<Scalar>::dz()
{
  return Teuchos::rcp( new VarFunction(_var->dz()) );
}

template <typename Scalar>
void VarFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  // This is an abstract function -- becomes concrete when evaluated at a Solution object.
  // Therefore, we should not ever call values on this...
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "VarFunction is abstract; values() should not be called.");
}

template <typename Scalar>
TFunctionPtr<Scalar> VarFunction<Scalar>::x()
{
  return Teuchos::rcp( new VarFunction(_var->x()) );
}

template <typename Scalar>
TFunctionPtr<Scalar> VarFunction<Scalar>::y()
{
  return Teuchos::rcp( new VarFunction(_var->y()) );
}

template <typename Scalar>
TFunctionPtr<Scalar> VarFunction<Scalar>::z()
{
  return Teuchos::rcp( new VarFunction(_var->z()) );
}

namespace Camellia
{
template class VarFunction<double>;
}

