#include "MonomialFunctions.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

string Xn::displayString() {
  ostringstream ss;
  if ((_n != 1) && (_n != 0)) {
    ss << "x^" << _n ;
  } else if (_n == 1) {
    ss << "x";
  } else {
    ss << "(1)";
  }
  return ss.str();
}
Xn::Xn(int n) {
  _n = n;
}
double Xn::value(double x) {
  return pow(x,_n);
}
FunctionPtr Xn::dx() {
  if (_n == 0) {
    return Function::zero();
  }
  FunctionPtr x_n_minus = Teuchos::rcp( new Xn(_n-1) );
  return _n * x_n_minus;
}
FunctionPtr Xn::dy() {
  return Function::zero();
}
FunctionPtr Xn::dz() {
  return Function::zero();
}
FunctionPtr Xn::dt() {
  return Function::zero();
}

string Yn::displayString() {
  ostringstream ss;
  if ((_n != 1) && (_n != 0)) {
    ss << "y^" << _n ;
  } else if (_n == 1) {
    ss << "y";
  } else {
    ss << "(1)";
  }
  return ss.str();
}
Yn::Yn(int n) {
  _n = n;
}
double Yn::value(double x, double y) {
  return pow(y,_n);
}

FunctionPtr Yn::dx() {
  return Function::zero();
}
FunctionPtr Yn::dy() {
  if (_n == 0) {
    return Function::zero();
  }
  FunctionPtr y_n_minus = Teuchos::rcp( new Yn(_n-1) );
  return _n * y_n_minus;
}
FunctionPtr Yn::dz() {
  return Function::zero();
}
FunctionPtr Yn::dt() {
  return Function::zero();
}

string Zn::displayString() {
  ostringstream ss;
  if ((_n != 1) && (_n != 0)) {
    ss << "z^" << _n ;
  } else if (_n == 1) {
    ss << "z";
  } else {
    ss << "(1)";
  }
  return ss.str();
}
Zn::Zn(int n) {
  _n = n;
}
double Zn::value(double x, double y, double z) {
  return pow(z,_n);
}

FunctionPtr Zn::dx() {
  return Function::zero();
}
FunctionPtr Zn::dy() {
  return Function::zero();
}
FunctionPtr Zn::dz() {
  if (_n == 0) {
    return Function::zero();
  }
  FunctionPtr z_n_minus = Teuchos::rcp( new Zn(_n-1) );
  return _n * z_n_minus;
}
FunctionPtr Zn::dt() {
  return Function::zero();
}

string Tn::displayString() {
  ostringstream ss;
  if ((_n != 1) && (_n != 0)) {
    ss << "t^" << _n ;
  } else if (_n == 1) {
    ss << "t";
  } else {
    ss << "(1)";
  }
  return ss.str();
}
Tn::Tn(int n) {
  _n = n;
}
double Tn::value(double x, double t) {
  return pow(t,_n);
}
double Tn::value(double x, double y, double t) {
  return pow(t,_n);
}
double Tn::value(double x, double y, double z, double t) {
  return pow(t,_n);
}

FunctionPtr Tn::dx() {
  return Function::zero();
}
FunctionPtr Tn::dy() {
  return Function::zero();
}
FunctionPtr Tn::dz() {
  return Function::zero();
}
FunctionPtr Tn::dt() {
  if (_n == 0) {
    return Function::zero();
  }
  FunctionPtr t_n_minus = Teuchos::rcp( new Tn(_n-1) );
  return _n * t_n_minus;
}