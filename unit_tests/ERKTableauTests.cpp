//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  ERKTableauTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/25/20.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia_FieldContainer.hpp"
#include "ERKTableau.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( ERKTableau, ConstructionAndValueLookup )
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
      
    ERKTableau tableau(numStages,a,b,c,expectedRate);
    
    double tol = 1e-15;
    for (int i=0; i<numStages; i++)
    {
      TEST_FLOATING_EQUALITY(c(i), tableau.c(i), tol);
      TEST_FLOATING_EQUALITY(b(i), tableau.b(i), tol);
      for (int j=0; j<numStages; j++)
      {
        TEST_FLOATING_EQUALITY(a(i,j), tableau.a(i,j), tol);
      }
    }
  }
} // namespace
