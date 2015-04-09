#include "PolarizedFunction.h"

#include "BasisCache.h"
#include "MonomialFunctions.h"
#include "PhysicalPointCache.h"
#include "TrigFunctions.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

PolarizedFunction::PolarizedFunction( FunctionPtr f_of_xAsR_yAsTheta ) : Function(f_of_xAsR_yAsTheta->rank()) {
  _f = f_of_xAsR_yAsTheta;
}

Teuchos::RCP<PolarizedFunction> PolarizedFunction::r() {
  static Teuchos::RCP<PolarizedFunction> _r = Teuchos::rcp( new PolarizedFunction( Teuchos::rcp( new Xn(1) ) ) );
  return _r;
}

Teuchos::RCP<PolarizedFunction> PolarizedFunction::sin_theta() {
  static Teuchos::RCP<PolarizedFunction> _sin_theta = Teuchos::rcp( new PolarizedFunction( Teuchos::rcp( new Sin_y ) ) );
  return _sin_theta;
}

Teuchos::RCP<PolarizedFunction> PolarizedFunction::cos_theta() {
  static Teuchos::RCP<PolarizedFunction> _cos_theta = Teuchos::rcp( new PolarizedFunction( Teuchos::rcp( new Cos_y ) ) );
  return _cos_theta;
}

void findAndReplace(string &str, const string &findStr, const string &replaceStr) {
  size_t found = str.find( findStr );
  while (found!=string::npos) {
    str.replace( found, findStr.length(), replaceStr );
    found = str.find( findStr );
  }
}

string PolarizedFunction::displayString() {
  string displayString = _f->displayString();
  findAndReplace(displayString, "x", "r");
  findAndReplace(displayString, "y", "\\theta");
  return displayString;
  //  ostringstream ss( _f->displayString());
  //  ss << "(r,\\theta)";
  //  return ss.str();
}

FunctionPtr PolarizedFunction::dx() {
  // cast everything to FunctionPtrs:
  FunctionPtr sin_theta_fxn = sin_theta();
  FunctionPtr dtheta_fxn = dtheta();
  FunctionPtr dr_fxn = dr();
  FunctionPtr r_fxn = r();
  FunctionPtr cos_theta_fxn = cos_theta();
  return dr_fxn * cos_theta_fxn - dtheta_fxn * sin_theta_fxn / r_fxn;
}
FunctionPtr PolarizedFunction::dy() {
  FunctionPtr sin_theta_fxn = sin_theta();
  FunctionPtr dtheta_fxn = dtheta();
  FunctionPtr dr_fxn = dr();
  FunctionPtr r_fxn = r();
  FunctionPtr cos_theta_fxn = cos_theta();
  return dr_fxn * sin_theta_fxn + dtheta_fxn * cos_theta_fxn / r_fxn;
}

Teuchos::RCP<PolarizedFunction> PolarizedFunction::dtheta() {
  return Teuchos::rcp( new PolarizedFunction( _f->dy() ) );
}

Teuchos::RCP<PolarizedFunction> PolarizedFunction::dr() {
  return Teuchos::rcp( new PolarizedFunction( _f->dx() ) );
}

bool PolarizedFunction::isZero() {
  return _f->isZero();
}

void PolarizedFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  static const double PI  = 3.141592653589793238462;
  
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  
  const Intrepid::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
  Intrepid::FieldContainer<double> polarPoints = *points;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double x = (*points)(cellIndex,ptIndex,0);
      double y = (*points)(cellIndex,ptIndex,1);
      double r = sqrt(x * x + y * y);
      double theta = (r != 0) ? acos(x/r) : 0;
      // now x = r cos theta, but need to guarantee that y = r sin theta (might differ in sign)
      // according to the acos docs, theta will be in [0, pi], so the rule is: (y < 0) ==> theta := 2 pi - theta;
      if (y < 0) theta = 2*PI-theta;
      
      polarPoints(cellIndex, ptIndex, 0) = r;
      polarPoints(cellIndex, ptIndex, 1) = theta;
      //      if (r == 0) {
      //        cout << "r == 0!" << endl;
      //      }
    }
  }
  BasisCachePtr dummyBasisCache = Teuchos::rcp( new PhysicalPointCache( polarPoints ) );
  _f->values(values,dummyBasisCache);
  if (_f->isZero()) {
    cout << "Warning: in PolarizedFunction, we are being asked for values when _f is zero.  This shouldn't happen.\n";
  }
  //  cout << "polarPoints: \n" << polarPoints;
  //  cout << "PolarizedFunction, values: \n" << values;
}