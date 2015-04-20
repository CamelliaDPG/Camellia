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
TFunctionPtr<double> Xn::dx() {
  if (_n == 0) {
    return TFunction<double>::zero();
  }
  TFunctionPtr<double> x_n_minus = Teuchos::rcp( new Xn(_n-1) );
  return (double)_n * x_n_minus;
}
TFunctionPtr<double> Xn::dy() {
  return TFunction<double>::zero();
}
TFunctionPtr<double> Xn::dz() {
  return TFunction<double>::zero();
}
TFunctionPtr<double> Xn::dt() {
  return TFunction<double>::zero();
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

TFunctionPtr<double> Yn::dx() {
  return TFunction<double>::zero();
}
TFunctionPtr<double> Yn::dy() {
  if (_n == 0) {
    return TFunction<double>::zero();
  }
  TFunctionPtr<double> y_n_minus = Teuchos::rcp( new Yn(_n-1) );
  return (double)_n * y_n_minus;
}
TFunctionPtr<double> Yn::dz() {
  return TFunction<double>::zero();
}
TFunctionPtr<double> Yn::dt() {
  return TFunction<double>::zero();
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

TFunctionPtr<double> Zn::dx() {
  return TFunction<double>::zero();
}
TFunctionPtr<double> Zn::dy() {
  return TFunction<double>::zero();
}
TFunctionPtr<double> Zn::dz() {
  if (_n == 0) {
    return TFunction<double>::zero();
  }
  TFunctionPtr<double> z_n_minus = Teuchos::rcp( new Zn(_n-1) );
  return (double)_n * z_n_minus;
}
TFunctionPtr<double> Zn::dt() {
  return TFunction<double>::zero();
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

TFunctionPtr<double> Tn::dx() {
  return TFunction<double>::zero();
}
TFunctionPtr<double> Tn::dy() {
  return TFunction<double>::zero();
}
TFunctionPtr<double> Tn::dz() {
  return TFunction<double>::zero();
}
TFunctionPtr<double> Tn::dt() {
  if (_n == 0) {
    return TFunction<double>::zero();
  }
  TFunctionPtr<double> t_n_minus = Teuchos::rcp( new Tn(_n-1) );
  return (double)_n * t_n_minus;
}
