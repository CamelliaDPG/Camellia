//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  BasisFamilyTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 9/11/23.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "CamelliaCellTools.h"
#include "CellTopology.h"
#include "BasisFamily.h"
#include "MeshFactory.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( BasisFamily, HierarchicalDerivativeBasis1D )
  {
    auto line = CellTopology::line();
    
    auto basisFamily = BasisFamily::hierarchicalBasisFamily();
    
    const double  h = 0.25;
    const double x0 = 5.0;
    const double x1 = x0 + h;
    const int numElements = 1;
    
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x0, x1, numElements);
    
    int max_p = 10;
    BasisCachePtr basisCache = BasisCache::basisCache1D(x0, x1, max_p);
    
    const double absTol = 1e-14;
    const double relTol = 1e-14;
    for (int p=1; p<=max_p; p++)
    {
      auto basisH1 = basisFamily->getBasis(line, FUNCTION_SPACE_HGRAD, p);
      auto basisL2_with_weights = basisFamily->getDerivativeBasis(line, {FUNCTION_SPACE_HGRAD}, {p}, 0);
      auto basisL2 = basisL2_with_weights.first;
      auto weights = basisL2_with_weights.second;
      
      auto valuesH1_dx = basisCache->getTransformedValues(basisH1, OP_DX);
      auto valuesL2    = basisCache->getTransformedValues(basisL2, OP_VALUE);
      
      const int numPoints = valuesH1_dx->dimension(2); // (C,F,P)
      const int cellOrdinal = 0;
      
      for (int i=0; i<basisH1->getCardinality(); i++)
      {
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          double valueH1_dx = (*valuesH1_dx)(cellOrdinal,i,pointOrdinal);
          
          double actualL2ExpansionValue = 0.0;
          for (int j=0; j<basisL2->getCardinality(); j++)
          {
            actualL2ExpansionValue += weights[i][j] * (*valuesL2)(cellOrdinal,j,pointOrdinal);
          }
          
          if ((std::abs(valueH1_dx) > absTol) || (std::abs(actualL2ExpansionValue) > absTol))
          {
            TEST_FLOATING_EQUALITY(valueH1_dx, actualL2ExpansionValue, relTol);
          }
          
        }
      }
    }
  }

} // anonymous namespace
