//
//  ERKTableau.cpp
//  AcousticsDriver
//
//  Created by Roberts, Nathan V on 3/25/20.
//

#include "ERKTableau.h"

using namespace Camellia;

ERKTableau::ERKTableau(const int numStages, const FieldContainer<double> &a, const FieldContainer<double> &b, const FieldContainer<double> &c, const double expectedRate)
{
  TEUCHOS_TEST_FOR_EXCEPTION(a.rank() != 2, std::invalid_argument, "a must have dimensions numStages x numStages");
  TEUCHOS_TEST_FOR_EXCEPTION(a.dimension(0) != numStages, std::invalid_argument, "a must have dimensions numStages x numStages");
  TEUCHOS_TEST_FOR_EXCEPTION(a.dimension(1) != numStages, std::invalid_argument, "a must have dimensions numStages x numStages");
  TEUCHOS_TEST_FOR_EXCEPTION(b.rank() != 1, std::invalid_argument, "b must have dimension numStages");
  TEUCHOS_TEST_FOR_EXCEPTION(b.dimension(0) != numStages, std::invalid_argument, "b must have dimension numStages");
  TEUCHOS_TEST_FOR_EXCEPTION(c.rank() != 1, std::invalid_argument, "c must have dimension numStages");
  TEUCHOS_TEST_FOR_EXCEPTION(c.dimension(0) != numStages, std::invalid_argument, "c must have dimension numStages");
  _a = std::vector<double>(numStages*(numStages-1)/2);
  _b = std::vector<double>(numStages);
  _c = std::vector<double>(numStages);
  _s = numStages;
  _expectedRate = expectedRate;
  int a_index = 0;
  for (int i=0; i<numStages; i++)
  {
    _b[i] = b(i);
    _c[i] = c(i);
    for (int j=0; j<i; j++)
    {
      _a[a_index] = a(i,j);
      a_index++;
    }
  }
}
double ERKTableau::a(int i, int j) const
{
  if (j >= i) return 0.0; // upper triangle of a is 0
  int i_offset = i * (i-1) / 2; // number of entries that belong to prior i values
  return _a[i_offset + j];
}
double ERKTableau::b(int i) const
{
  return _b[i];
}
double ERKTableau::c(int i) const
{
  return _c[i];
}

double ERKTableau::expectedRate() const
{
  return _expectedRate;
}

ERKTableauPtr ERKTableau::forwardEuler()
{
  const int numStages = 1;
  const double expectedRate = 1.0;
  FieldContainer<double> a(numStages,numStages);
  FieldContainer<double> b(numStages);
  FieldContainer<double> c(numStages);
  a(0,0) = 0.0;
  b(0)   = 1.0;
  c(0)   = 0.0;
  return Teuchos::rcp(new ERKTableau(numStages,a,b,c,expectedRate));
}

ERKTableauPtr ERKTableau::heun()
{
  const int numStages = 2;
  const double expectedRate = 2.0;
  FieldContainer<double> a(numStages,numStages);
  FieldContainer<double> b(numStages);
  FieldContainer<double> c(numStages);
  a.initialize(0);
  a(1,0) = 1.;
  
  b(0)   = 1./2.;
  b(1)   = 1./2.;
  
  c(0)   = 0.0;
  c(1)   = 1.;
  
  return Teuchos::rcp(new ERKTableau(numStages,a,b,c,expectedRate));
}

int ERKTableau::numStages() const
{
  return _s;
}

ERKTableauPtr ERKTableau::rk4()
{
  const int numStages = 4;
  const double expectedRate = 4.0;
  FieldContainer<double> a(numStages,numStages);
  FieldContainer<double> b(numStages);
  FieldContainer<double> c(numStages);
  a.initialize(0);
  a(1,0) = 1./2.;
  a(2,1) = 1./2.;
  a(3,2) = 1.0;
  
  b(0)   = 1./6.;
  b(1)   = 1./3.;
  b(2)   = 1./3.;
  b(3)   = 1./6.;
  
  c(0)   = 0.0;
  c(1)   = 1./2.;
  c(2)   = 1./2.;
  c(3)   = 1.0;
  
  return Teuchos::rcp(new ERKTableau(numStages,a,b,c,expectedRate));
}

/**
 * A third-order, three-stage explicit SSPRK scheme.
 *
 * Gottlieb, Sigal, and Chi-Wang Shu. "Total variation diminishing Runge-Kutta schemes."
 * Mathematics of computation of the American Mathematical Society 67, no. 221 (1998): 73-85.
 *
 * (See Prop. 3.2 for the scheme.  Note that this is not expressed as a Butcher tableau; some
 *  algebra is required to establish the equivalence of this with what is implemented below.)
 */

ERKTableauPtr ERKTableau::ssprk3()
{
  const int numStages = 3;
  const double expectedRate = 3.0;
  FieldContainer<double> a(numStages,numStages);
  FieldContainer<double> b(numStages);
  FieldContainer<double> c(numStages);
  a.initialize(0);
  a(1,0) = 1.;
  a(2,0) = 1./4.;
  a(2,1) = 1./4.;
  
  b(0)   = 1./6.;
  b(1)   = 1./6.;
  b(2)   = 2./3.;
  
  c(0)   = 0.0;
  c(1)   = 1.;
  c(2)   = 1./2.;
  
  return Teuchos::rcp(new ERKTableau(numStages,a,b,c,expectedRate));
}
