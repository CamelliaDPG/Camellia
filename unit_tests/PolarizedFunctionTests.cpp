//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  MinMaxFunctionsTests
//  Camellia
//
//  Created by Stefan Henneking on 1/24/20.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "PolarizedFunction.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( PolarizedFunction, Polar_values )
  {
    FunctionPtr cos_y = Teuchos::rcp(new Cos_ay(1));
    FunctionPtr sin_y = Teuchos::rcp(new Sin_ay(1));
    FunctionPtr cos_theta = Teuchos::rcp(new PolarizedFunction<double>(cos_y));
    FunctionPtr sin_theta = Teuchos::rcp(new PolarizedFunction<double>(sin_y));
    
    bool conformingTraces = true; int spaceDim = 2;
    PoissonFormulation form(spaceDim,conformingTraces);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, {1,1}, 1);
    
    double tol = 1.0e-14;
    
    for (int cellID : mesh->cellIDsInPartition()) {
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      const Camellia::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePointsFC());
      int numPoints = points->dimension(1);
      auto fc_cos = FieldContainer<double>(1,numPoints);
      auto fc_sin = FieldContainer<double>(1,numPoints);
      cos_theta->values(fc_cos, basisCache);
      sin_theta->values(fc_sin, basisCache);
      for (int j = 0; j < numPoints; j++) {
        double x = (*points)(0,j,0);
        double y = (*points)(0,j,1);
        double r = sqrt(x*x + y*y);
        double cos_t = fc_cos(0,j);
        double sin_t = fc_sin(0,j);
        TEST_FLOATING_EQUALITY(x,r*cos_t,tol);
        TEST_FLOATING_EQUALITY(y,r*sin_t,tol);
      }
    }
  }
} // namespace

