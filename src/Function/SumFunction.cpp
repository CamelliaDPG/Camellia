//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
#include "SumFunction.h"

#include "BasisCache.h"

#include "CamelliaDebugUtility.h"

using namespace Camellia;

template <typename Scalar>
SumFunction<Scalar>::SumFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2) : TFunction<Scalar>("SumFunction", f1->rank())
{
  TEUCHOS_TEST_FOR_EXCEPTION( f1->rank() != f2->rank(), std::invalid_argument, "summands must be of like rank.");
  TEUCHOS_TEST_FOR_EXCEPTION( f1->boundaryValueOnly() != f2->boundaryValueOnly(), std::invalid_argument, "summands must agree on boundaryValueOnly()" );
  _f1 = f1;
  _f2 = f2;
}

template <typename Scalar>
bool SumFunction<Scalar>::boundaryValueOnly()
{
  // we enforce agreement on boundaryValueOnly() in constructor
  return _f1->boundaryValueOnly();
}

template <typename Scalar>
string SumFunction<Scalar>::displayString()
{
  ostringstream ss;
  ss << "(" << _f1->displayString() << " + " << _f2->displayString() << ")";
  return ss.str();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap)
{
  auto f1 = TFunction<Scalar>::evaluateFunctionAt(_f1, valueMap);
  auto f2 = TFunction<Scalar>::evaluateFunctionAt(_f2, valueMap);
  return f1 + f2;
}

template <typename Scalar>
TLinearTermPtr<Scalar> SumFunction<Scalar>::jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap)
{
  auto df1 = _f1->jacobian(valueMap);
  auto df2 = _f2->jacobian(valueMap);
  return df1 + df2;
}

template <typename Scalar>
int SumFunction<Scalar>::numVectorComponents() const
{
  if (this->rank() == 0)
  {
    return 0;
  }
  else
  {
    const int f1Components = _f1->numVectorComponents();
    const int f2Components = _f2->numVectorComponents();
    return std::max(f1Components, f2Components);
  }
}

template <typename Scalar>
std::vector<Intrepid2::DimensionInfo> SumFunction<Scalar>::valuesDataDims(BasisCachePtr basisCache)
{
  auto f1Dims = _f1->valuesDataDims(basisCache);
  auto f2Dims = _f2->valuesDataDims(basisCache);
  
  TEUCHOS_TEST_FOR_EXCEPTION(f1Dims.size() != f2Dims.size(), std::invalid_argument, "summands do not have like rank!");
  
  const int numDims = f1Dims.size();
  std::vector<Intrepid2::DimensionInfo> combinedDims(numDims);
  for (int d=0; d<numDims; d++)
  {
    combinedDims[d] = Intrepid2::combinedDimensionInfo(f1Dims[d], f2Dims[d]);
  }
  return combinedDims;
}

template <typename Scalar>
void SumFunction<Scalar>::values(Intrepid2::Data<Scalar, Camellia::DeviceType> &values, BasisCachePtr basisCache)
{
  auto f1Data = _f1->checkoutValuesData(basisCache);
  auto f2Data = _f2->checkoutValuesData(basisCache);
  
  _f1->values(f1Data,basisCache);
  _f2->values(f2Data,basisCache);
  
  values.storeInPlaceSum(f1Data, f2Data);
  
  _f1->checkinValuesData(f1Data);
  _f2->checkinValuesData(f2Data);

//  {
//    // DEBUGGING
//    const int numPoints = values.extent_int(1);
//    if (numPoints == 64)
//    {
//      cout << "SumFunction: ";
//      cout << _f1->getClassName() << " + " << _f2->getClassName() << ":\n";
//      if (values.rank() == 2)
//      {
//        Camellia::print2(cout, "SumFunction values (Data path)", values);
//      }
//      else if (values.rank() == 3)
//      {
//        Camellia::print3(cout, "SumFunction values (Data path)", values);
//      }
//      else if (values.rank() == 4)
//      {
//        Camellia::print4(cout, "SumFunction values (Data path)", values);
//      }
//    }
//  }
}

template <typename Scalar>
void SumFunction<Scalar>::values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType,  BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("SumFunction values()");
  if (modType == ASSIGN)
  {
    _f1->values(values, ValueModificationType::ASSIGN, basisCache);
    Kokkos::fence();
    _f2->values(values, ValueModificationType::ADD, basisCache);
  }
  else
  {
    auto sumValues = basisCache->checkoutScratchStorage<Scalar>(values);
    _f1->values(sumValues, ValueModificationType::ASSIGN, basisCache);
    Kokkos::fence();
    _f2->values(sumValues, ValueModificationType::ADD, basisCache);
    Kokkos::fence();
    assignValues<Scalar>(values, modType, sumValues);
    basisCache->checkinScratchStorage<Scalar>(sumValues);
  }
//  {
//    // DEBUGGING
//    const int numPoints = values.extent_int(1);
//    if (numPoints == 64)
//    {
//      cout << "SumFunction: ";
//      cout << _f1->getClassName() << " + " << _f2->getClassName() << ":\n";
//      if (values.rank() == 2)
//      {
//        Camellia::print2(cout, "SumFunction values (View path)", values);
//      }
//      else if (values.rank() == 3)
//      {
//        Camellia::print3(cout, "SumFunction values (View path)", values);
//      }
//      else if (values.rank() == 4)
//      {
//        Camellia::print4(cout, "SumFunction values (View path)", values);
//      }
//    }
//  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::x()
{
  if ( (_f1->x() == Teuchos::null) || (_f2->x() == Teuchos::null) )
  {
    return this->null();
  }
  return _f1->x() + _f2->x();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::y()
{
  if ( (_f1->y() == Teuchos::null) || (_f2->y() == Teuchos::null) )
  {
    return this->null();
  }
  return _f1->y() + _f2->y();
}
template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::z()
{
  if ( (_f1->z() == Teuchos::null) || (_f2->z() == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return _f1->z() + _f2->z();
}
template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::t()
{
  if ( (_f1->t() == Teuchos::null) || (_f2->t() == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return _f1->t() + _f2->t();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::dx()
{
  if ( (_f1->dx() == Teuchos::null) || (_f2->dx() == Teuchos::null) )
  {
    return this->null();
  }
  return _f1->dx() + _f2->dx();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::dy()
{
  if ( (_f1->dy() == Teuchos::null) || (_f2->dy() == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return _f1->dy() + _f2->dy();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::dz()
{
  if ( (_f1->dz() == Teuchos::null) || (_f2->dz() == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return _f1->dz() + _f2->dz();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::di(int i) // 1 for dx(), 2 for dy(), 3 for dz()
{
  auto f1_di = _f1->di(i);
  auto f2_di = _f2->di(i);
  if ( (f1_di == Teuchos::null) || (f2_di == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return f1_di + f2_di;
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::dt()
{
  if ( (_f1->dt() == Teuchos::null) || (_f2->dt() == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return _f1->dt() + _f2->dt();
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::grad(int numComponents)
{
  if ( (_f1->grad(numComponents) == Teuchos::null) || (_f2->grad(numComponents) == Teuchos::null) )
  {
    return Teuchos::null;
  }
  else
  {
    return _f1->grad(numComponents) + _f2->grad(numComponents);
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> SumFunction<Scalar>::div()
{
  if ( (_f1->div() == Teuchos::null) || (_f2->div() == Teuchos::null) )
  {
    return Teuchos::null;
  }
  else
  {
    return _f1->div() + _f2->div();
  }
}

template <typename Scalar>
std::vector<TFunctionPtr<Scalar>> SumFunction<Scalar>::memberFunctions()
{
  return {{_f1, _f2}};
}

namespace Camellia
{
template class SumFunction<double>;
}

