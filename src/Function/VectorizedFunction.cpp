#include "VectorizedFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

VectorizedFunction::VectorizedFunction(const vector< FunctionPtr > &fxns) : Function(fxns[0]->rank() + 1) {
  _fxns = fxns;
}
VectorizedFunction::VectorizedFunction(FunctionPtr f1, FunctionPtr f2) : Function(f1->rank() + 1) {
  TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
  _fxns.push_back(f1);
  _fxns.push_back(f2);
}
VectorizedFunction::VectorizedFunction(FunctionPtr f1, FunctionPtr f2, FunctionPtr f3) : Function(f1->rank() + 1) {
  TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
  TEUCHOS_TEST_FOR_EXCEPTION(f3->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
  _fxns.push_back(f1);
  _fxns.push_back(f2);
  _fxns.push_back(f3);
}
VectorizedFunction::VectorizedFunction(FunctionPtr f1, FunctionPtr f2, FunctionPtr f3, FunctionPtr f4) : Function(f1->rank() + 1) {
  TEUCHOS_TEST_FOR_EXCEPTION(f1->rank() != f2->rank(), std::invalid_argument, "function ranks do not match");
  TEUCHOS_TEST_FOR_EXCEPTION(f2->rank() != f3->rank(), std::invalid_argument, "function ranks do not match");
  TEUCHOS_TEST_FOR_EXCEPTION(f3->rank() != f4->rank(), std::invalid_argument, "function ranks do not match");
  _fxns.push_back(f1);
  _fxns.push_back(f2);
  _fxns.push_back(f3);
  _fxns.push_back(f4);
}
string VectorizedFunction::displayString() {
  ostringstream str;
  str << "(";
  for (int i=0; i<_fxns.size(); i++) {
    if (i > 0) str << ",";
    str << _fxns[i]->displayString();
  }
  str << ")";
  return str.str();
}
int VectorizedFunction::dim() {
  return _fxns.size();
}

void VectorizedFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
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
  Intrepid::FieldContainer<double> compValues(dims);
  int valuesPerComponent = compValues.size();
  
  for (int comp=0; comp < numComponents; comp++) {
    FunctionPtr fxn = _fxns[comp];
    fxn->values(compValues, basisCache);
    for (int i=0; i < valuesPerComponent; i++) {
      values[ numComponents * i + comp ] = compValues[ i ];
    }
  }
}

FunctionPtr VectorizedFunction::x() {
  return _fxns[0];
}

FunctionPtr VectorizedFunction::y() {
  return _fxns[1];
}

FunctionPtr VectorizedFunction::z() {
  if (dim() >= 3) {
    return _fxns[2];
  } else {
    return Function::null();
  }
}

FunctionPtr VectorizedFunction::t() {
  return _fxns[dim()-1];
}

FunctionPtr VectorizedFunction::di(int i) {
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
  vector< FunctionPtr > fxns;
  for (int j = 0; j< dim(); j++) {
    FunctionPtr fj_di = Function::op(_fxns[j], op);
    if (isNull(fj_di)) {
      return Function::null();
    }
    fxns.push_back(fj_di);
  }
  // if we made it this far, then all components aren't null:
  return Teuchos::rcp( new VectorizedFunction(fxns) );
}

FunctionPtr VectorizedFunction::dx() {
  return di(0);
}
FunctionPtr VectorizedFunction::dy() {
  return di(1);
}
FunctionPtr VectorizedFunction::dz() {
  return di(2);
}
FunctionPtr VectorizedFunction::dt() {
  return di(dim()-1);
}

bool VectorizedFunction::isZero() {
  // vector function is zero if each of its components is zero.
  for (vector< FunctionPtr >::iterator fxnIt = _fxns.begin(); fxnIt != _fxns.end(); fxnIt++) {
    if (! (*fxnIt)->isZero() ) {
      return false;
    }
  }
  return true;
}
