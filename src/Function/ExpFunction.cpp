#include "ExpFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

string Exp_x::displayString() {
  return "e^x";
}
double Exp_x::value(double x, double y) {
  return exp(x);
}
FunctionPtr Exp_x::dx() {
  return Teuchos::rcp( new Exp_x );
}
FunctionPtr Exp_x::dy() {
  return Function::zero();
}
FunctionPtr Exp_x::dz() {
  return Function::zero();
}

string Exp_y::displayString() {
  return "e^y";
}
double Exp_y::value(double x, double y) {
  return exp(y);
}
FunctionPtr Exp_y::dx() {
  return Function::zero();
}
FunctionPtr Exp_y::dy() {
  return Teuchos::rcp( new Exp_y );
}
FunctionPtr Exp_y::dz() {
  return Function::zero();
}

string Exp_z::displayString() {
  return "e^z";
}
double Exp_z::value(double x, double y, double z) {
  return exp(z);
}
FunctionPtr Exp_z::dx() {
  return Function::zero();
}
FunctionPtr Exp_z::dy() {
  return Function::zero();
}
FunctionPtr Exp_z::dz() {
  return Teuchos::rcp( new Exp_z );
}

Exp_ax::Exp_ax(double a) {
  _a = a;
}
double Exp_ax::value(double x, double y) {
  return exp( _a * x);
}
FunctionPtr Exp_ax::dx() {
  return _a * (FunctionPtr) Teuchos::rcp(new Exp_ax(_a));
}
FunctionPtr Exp_ax::dy() {
  return Function::zero();
}
string Exp_ax::displayString() {
  ostringstream ss;
  ss << "\\exp( " << _a << " x )";
  return ss.str();
}

Exp_ay::Exp_ay(double a) {
  _a = a;
}
double Exp_ay::value(double x, double y) {
  return exp( _a * y);
}
FunctionPtr Exp_ay::dx() {
  return Function::zero();
}
FunctionPtr Exp_ay::dy() {
  return _a * (FunctionPtr) Teuchos::rcp(new Exp_ay(_a));
}
string Exp_ay::displayString() {
  ostringstream ss;
  ss << "\\exp( " << _a << " y )";
  return ss.str();
}