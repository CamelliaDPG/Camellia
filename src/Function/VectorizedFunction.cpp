#include "VectorizedFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

template <typename Scalar>
VectorizedFunction<Scalar>::VectorizedFunction(const vector< TFunctionPtr<Scalar> > &fxns) : TFunction<Scalar>(fxns[0]->rank() + 1) {
  _fxns = fxns;
}
template <typename Scalar>
VectorizedFunction<Scalar>::VectorizedFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2) : TFunction<Scalar>(f1->rank() + 1) {
  TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
  _fxns.push_back(f1);
  _fxns.push_back(f2);
}
template <typename Scalar>
VectorizedFunction<Scalar>::VectorizedFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2, TFunctionPtr<Scalar> f3) : TFunction<Scalar>(f1->rank() + 1) {
  TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
  TEUCHOS_TEST_FOR_EXCEPTION(f3->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
  _fxns.push_back(f1);
  _fxns.push_back(f2);
  _fxns.push_back(f3);
}
template <typename Scalar>
VectorizedFunction<Scalar>::VectorizedFunction(TFunctionPtr<Scalar> f1, TFunctionPtr<Scalar> f2, TFunctionPtr<Scalar> f3, TFunctionPtr<Scalar> f4) : TFunction<Scalar>(f1->rank() + 1) {
  TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
  TEUCHOS_TEST_FOR_EXCEPTION(f2->rank() != f3->rank(), std::invalid_argument, "function ranks do not match");
  TEUCHOS_TEST_FOR_EXCEPTION(f3->rank() != f4->rank(), std::invalid_argument, "function ranks do not match");
  _fxns.push_back(f1);
  _fxns.push_back(f2);
  _fxns.push_back(f3);
  _fxns.push_back(f4);
}
template <typename Scalar>
string VectorizedFunction<Scalar>::displayString() {
  ostringstream str;
  str << "(";
  for (int i=0; i<_fxns.size(); i++) {
    if (i > 0) str << ",";
    str << _fxns[i]->displayString();
  }
  str << ")";
  return str.str();
}
template <typename Scalar>
int VectorizedFunction<Scalar>::dim() {
  return _fxns.size();
}

template <typename Scalar>
void VectorizedFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
  this->CHECK_VALUES_RANK(values);
  // this is not going to be particularly efficient, because values from the components need to be interleaved...
  Teuchos::Array<int> dims;
  values.dimensions(dims);
  int numComponents = dims[dims.size()-1];
  TEUCHOS_TEST_FOR_EXCEPTION( numComponents > _fxns.size(), std::invalid_argument, "too many components requested" );
  if (numComponents != _fxns.size()) {
    // we're asking for fewer components than we have functions.  We're going to say that's OK so long as the
    // unused functions are 0.
    for (int i=numComponents; i<_fxns.size(); i++) {
      if (!_fxns[i]->isZero()) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "_fxns outnumber components and some of those extra Functions aren't zero!");
      }
    }
  }
  dims.pop_back(); // remove the last, dimensions argument
  Intrepid::FieldContainer<Scalar> compValues(dims);
  int valuesPerComponent = compValues.size();

  for (int comp=0; comp < numComponents; comp++) {
    TFunctionPtr<Scalar> fxn = _fxns[comp];
    fxn->values(compValues, basisCache);
    for (int i=0; i < valuesPerComponent; i++) {
      values[ numComponents * i + comp ] = compValues[ i ];
    }
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> VectorizedFunction<Scalar>::x() {
  return _fxns[0];
}

template <typename Scalar>
TFunctionPtr<Scalar> VectorizedFunction<Scalar>::y() {
  return _fxns[1];
}

template <typename Scalar>
TFunctionPtr<Scalar> VectorizedFunction<Scalar>::z() {
  if (dim() >= 3) {
    return _fxns[2];
  } else {
    return TFunction<Scalar>::null();
  }
}

template <typename Scalar>
TFunctionPtr<Scalar> VectorizedFunction<Scalar>::t() {
  return _fxns[dim()-1];
}

template <typename Scalar>
TFunctionPtr<Scalar> VectorizedFunction<Scalar>::di(int i) {
  // derivative in the ith coordinate direction
  Camellia::EOperator op;
  switch (i) {
    case 0:
      op = Camellia::OP_DX;
      break;
    case 1:
      op = Camellia::OP_DY;
      break;
    case 2:
      op = Camellia::OP_DZ;
      break;
    case 3:
      op = Camellia::OP_DT; // in all cases except when spaceDim = 3, OP_DY or OP_DZ should mean the same as OP_DT
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid coordinate direction");
      break;
  }
  vector< TFunctionPtr<Scalar> > fxns;
  for (int j = 0; j< dim(); j++) {
    TFunctionPtr<Scalar> fj_di = TFunction<Scalar>::op(_fxns[j], op);
    if (this->isNull(fj_di)) {
      return TFunction<Scalar>::null();
    }
    fxns.push_back(fj_di);
  }
  // if we made it this far, then all components aren't null:
  return Teuchos::rcp( new VectorizedFunction<Scalar>(fxns) );
}

template <typename Scalar>
TFunctionPtr<Scalar> VectorizedFunction<Scalar>::dx() {
  return di(0);
}
template <typename Scalar>
TFunctionPtr<Scalar> VectorizedFunction<Scalar>::dy() {
  return di(1);
}
template <typename Scalar>
TFunctionPtr<Scalar> VectorizedFunction<Scalar>::dz() {
  return di(2);
}
template <typename Scalar>
TFunctionPtr<Scalar> VectorizedFunction<Scalar>::dt() {
  return di(dim()-1);
}

template <typename Scalar>
bool VectorizedFunction<Scalar>::isZero() {
  // vector function is zero if each of its components is zero.
  for (typename vector< TFunctionPtr<Scalar> >::iterator fxnIt = _fxns.begin(); fxnIt != _fxns.end(); fxnIt++) {
    if (! (*fxnIt)->isZero() ) {
      return false;
    }
  }
  return true;
}

template class VectorizedFunction<double>;

