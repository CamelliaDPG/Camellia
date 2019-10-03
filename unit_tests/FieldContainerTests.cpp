//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  FieldContainerTests
//  Camellia
//
//  Created by Nate Roberts on 10/3/19.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "Camellia_FieldContainer.hpp"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( FieldContainer, ViewKeptInSync )
  {
    Camellia::FieldContainer<double> testContainer(3,3,3);
    testContainer[4] = 300.0;
    std::cout << "testContainer: " << testContainer;
    testContainer.resize(3);
    
  }
} // namespace
