//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "ProductFunction.h"

#include "BasisCache.h"
#include "ConstantScalarFunction.h"

#include "CamelliaDebugUtility.h"

using namespace Camellia;

template <typename Scalar>
string ProductFunction<Scalar>::displayString()
{
  ostringstream ss;
  string times_string = " * ";
  if ((_f1->rank() == 1) && _f2->rank() == 1) times_string = " \\cdot ";
  
  ConstantScalarFunction<Scalar>* f1_constant = dynamic_cast<ConstantScalarFunction<Scalar>*> (_f1.get());
  if (f1_constant && (f1_constant->value() == -1.0))
  {
    // then just show "-f2"
    ss << "-" << _f2->displayString();
  }
  else
  {
    ss << _f1->displayString() << times_string << _f2->displayString();
  }
  return ss.str();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::di(int i) // 1 for dx(), 2 for dy(), 3 for dz()
{
  auto f1_di = _f1->di(i);
  auto f2_di = _f2->di(i);
  if ( (f1_di == Teuchos::null) || (f2_di == Teuchos::null) )
  {
    return Teuchos::null;
  }
  return _f1 * f2_di + f1_di * _f2;
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::dx()
{
  if ( (_f1->dx().get() == NULL) || (_f2->dx().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dx() + _f2 * _f1->dx();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::dy()
{
  if ( (_f1->dy().get() == NULL) || (_f2->dy().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dy() + _f2 * _f1->dy();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::dz()
{
  if ( (_f1->dz().get() == NULL) || (_f2->dz().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dz() + _f2 * _f1->dz();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::dt()
{
  if ( (_f1->dt().get() == NULL) || (_f2->dt().get() == NULL) )
  {
    return this->null();
  }
  // otherwise, apply product rule:
  return _f1 * _f2->dt() + _f2 * _f1->dt();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::evaluateAt(const map<int, TFunctionPtr<Scalar> > &valueMap)
{
  auto f1 = TFunction<Scalar>::evaluateFunctionAt(_f1, valueMap);
  auto f2 = TFunction<Scalar>::evaluateFunctionAt(_f2, valueMap);
  return f1 * f2;
}

template <typename Scalar>
bool ProductFunction<Scalar>::isZero()
{
  return _f1->isZero() || _f2->isZero();
}

template <typename Scalar>
bool ProductFunction<Scalar>::isZero(BasisCachePtr basisCache)
{
  return _f1->isZero(basisCache) || _f2->isZero(basisCache);
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::f1()
{
  return _f1;
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::f2()
{
  return _f2;
}

template <typename Scalar>
TLinearTermPtr<Scalar> ProductFunction<Scalar>::jacobian(const map<int, TFunctionPtr<Scalar> > &valueMap)
{
  auto f1 = TFunction<Scalar>::evaluateFunctionAt(_f1, valueMap);
  auto f2 = TFunction<Scalar>::evaluateFunctionAt(_f2, valueMap);
  auto df1 = _f1->jacobian(valueMap);
  auto df2 = _f2->jacobian(valueMap);
  return f1 * df2 + df1 * f2;
}

template <typename Scalar>
int ProductFunction<Scalar>::numVectorComponents() const
{
  if (this->rank() == 0) return 0;
  else return _f2->numVectorComponents();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::x()
{
  if (this->rank() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take x component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (this->isNull(_f2->x()))
  {
    return this->null();
  }
  return _f1 * _f2->x();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::y()
{
  if (this->rank() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take y component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (this->isNull(_f2->y()))
  {
    return this->null();
  }
  return _f1 * _f2->y();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::z()
{
  if (this->rank() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take z component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (this->isNull(_f2->z()))
  {
    return this->null();
  }
  return _f1 * _f2->z();
}

template <typename Scalar>
TFunctionPtr<Scalar> ProductFunction<Scalar>::t()
{
  if (this->rank() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can't take t component of scalar function.");
  }
  // otherwise, _f2 is the rank > 0 function
  if (this->isNull(_f2->t()))
  {
    return this->null();
  }
  return _f1 * _f2->t();
}

template <typename Scalar>
int ProductFunction<Scalar>::productRank(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2)
{
  if (f1->rank() == f2->rank()) return 0;
  if (f1->rank() == 0) return f2->rank();
  if (f2->rank() == 0) return f1->rank();
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank pairing for function product.");
//  return -1;
}

template <typename Scalar>
ProductFunction<Scalar>::ProductFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2) : TFunction<Scalar>("ProductFunction", productRank(f1,f2) )
{
  // for simplicity of values() code, ensure that rank of f1 ≤ rank of f2:
  if ( f1->rank() <= f2->rank() )
  {
    _f1 = f1;
    _f2 = f2;
  }
  else
  {
    _f1 = f2;
    _f2 = f1;
  }
  if (_f1->rank() != 0)
  {
    // New 5-13-21: consider a rule that we *don't* do tensor contractions within ProductFunction; instead, these should be reduced analytically to sums of scalars.
    cout << "Warning: creating a ProductFunction in which both multiplicands have positive rank.\n";
  }
  // the following should be false for all the automatic products.  Added the test for debugging...
  if ((_f1->isZero()) || (_f2->isZero()))
  {
    cout << "Warning: creating a ProductFunction where one of the multiplicands is zero." << endl;
  }
}

template <typename Scalar>
bool ProductFunction<Scalar>::boundaryValueOnly()
{
  return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
}

template <typename Scalar>
std::vector<TFunctionPtr<Scalar>> ProductFunction<Scalar>::memberFunctions()
{
  return {{_f1, _f2}};
}

template <typename Scalar>
std::vector<Intrepid2::DimensionInfo> ProductFunction<Scalar>::valuesDataDims(BasisCachePtr basisCache)
{
  auto f1Dims = _f1->valuesDataDims(basisCache);
  auto f2Dims = _f2->valuesDataDims(basisCache);
 
  bool tensorContraction = ( _f2->rank() > 0) && (this->rank() == 0); // tensor contraction resulting in scalar value
  int numContractionDims = tensorContraction ? _f2->rank() : 0; // dims that are contracted away don't affect variation dims (i.e., we don't need to include them in combinedDims below)
  
  // if _f1 and _f2 have differing rank, by construction _f2 is the higher-rank
  
  const int numDims = f1Dims.size() - numContractionDims;
  std::vector<Intrepid2::DimensionInfo> combinedDims(f2Dims.size()-numContractionDims);
  for (int d=0; d<numDims; d++)
  {
    combinedDims[d] = Intrepid2::combinedDimensionInfo(f1Dims[d], f2Dims[d]);
  }
  for (int d=numDims; d<combinedDims.size(); d++)
  {
    combinedDims[d] = f2Dims[d];
  }
  return combinedDims;
}

template <typename Scalar>
void ProductFunction<Scalar>::values(Intrepid2::Data<Scalar, Camellia::DeviceType> &values, BasisCachePtr basisCache)
{
  auto f1Data = _f1->checkoutValuesData(basisCache);
  auto f2Data = _f2->checkoutValuesData(basisCache);
  
  _f1->values(f1Data,basisCache);
  _f2->values(f2Data,basisCache);
  
  using ExecutionSpace = Camellia::DeviceExecutionSpace;
  if ((_f1->rank() == 0) && (_f2->rank() == 0))
  {
    // scalar multiplicands
    values.storeInPlaceProduct(f1Data, f2Data);
  }
  else if (_f1->rank() == 0)
  {
    // scalar times a higher-rank function
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unfinished implementation");
  }
  else if (this->rank() == 0)
  {
    // higher-rank multiplicands, contracted to a scalar
    if (values.rank() == 2)
    {
      // (C,P)
      TEUCHOS_TEST_FOR_EXCEPTION(f1Data.rank() != 3, std::invalid_argument, "Support for functions with rank > 1 not yet implemented");
      int d_extent = f1Data.extent_int(2);
      auto policy = Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>>({0,0},{values.getDataExtent(0),values.getDataExtent(1)});
      Kokkos::parallel_for("compute contraction", policy,
      KOKKOS_LAMBDA (const int &cellOrdinal, const int &pointOrdinal) {
        auto & val = values.getWritableEntry(cellOrdinal, pointOrdinal);
        val = 0;
        for (int d=0; d<d_extent; d++)
        {
          val += f1Data(cellOrdinal,pointOrdinal,d) * f2Data(cellOrdinal,pointOrdinal,d);
        }
      });
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unfinished implementation");
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank combination");
  }
  
  _f1->checkinValuesData(f1Data);
  _f2->checkinValuesData(f2Data);
  
//  {
//    // DEBUGGING
//    const int numPoints = values.extent_int(1);
//    if (numPoints == 64)
//    {
//      cout << "ProductFunction: ";
//      cout << _f1->getClassName() << " x " << _f2->getClassName() << ":\n";
//      
//      if (values.rank() == 2)
//      {
//        Camellia::print2(cout, "ProductFunction values (Data path)", values);
//      }
//      else if (values.rank() == 3)
//      {
//        Camellia::print3(cout, "ProductFunction values (Data path)", values);
//      }
//    }
//  }
}

template <typename Scalar>
void ProductFunction<Scalar>::values(typename FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("ProductFunction values()");
  if (modType == ASSIGN)
  {
    if (( _f2->rank() > 0) && (this->rank() == 0)) // tensor contraction resulting in scalar value
    {
      _f2->valuesDottedWithTensor(values, _f1, basisCache);
    }
    else // scalar multiplication by f1, then
    {
      _f2->values(values, ValueModificationType::ASSIGN, basisCache);
      Kokkos::fence();
      _f1->scalarMultiplyFunctionValues(values, basisCache);
    }
  }
  else
  {
    auto valuesScratch = basisCache->checkoutScratchStorage<Scalar>(values);
    if (( _f2->rank() > 0) && (this->rank() == 0)) // tensor contraction resulting in scalar value
    {
      _f2->valuesDottedWithTensor(valuesScratch, _f1, basisCache);
    }
    else // scalar multiplication by f1, then
    {
      _f2->values(valuesScratch, ValueModificationType::ASSIGN, basisCache);
      Kokkos::fence();
      _f1->scalarMultiplyFunctionValues(valuesScratch, basisCache);
    }
    Kokkos::fence();
    assignValues<Scalar>(values, modType, valuesScratch);
    basisCache->checkinScratchStorage<Scalar>(valuesScratch);
  }
//  {
//    // DEBUGGING
//    const int numPoints = values.extent_int(1);
//    if (numPoints == 64)
//    {
//      cout << "ProductFunction: ";
//      cout << _f1->getClassName() << " x " << _f2->getClassName() << ":\n";
//      
//      if (values.rank() == 2)
//      {
//        Camellia::print2(cout, "ProductFunction values (View path)", values);
//      }
//      else if (values.rank() == 3)
//      {
//        Camellia::print3(cout, "ProductFunction values (View path)", values);
//      }
//    }
//  }
}

namespace Camellia
{
template class ProductFunction<double>;
}
