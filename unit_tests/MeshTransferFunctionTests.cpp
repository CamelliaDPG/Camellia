//
//  MeshTransferFunctionTests
//  Camellia
//
//  Created by Nate Roberts on 11/25/14.
//
//

// empty test file.  Copy (naming "MyClassTests.cpp", typically) and then add your tests below.

#include "Teuchos_UnitTestHarness.hpp"
namespace {
  TEUCHOS_UNIT_TEST( MeshTransferFunction, CellMap)
  {
    // TODO: write this test
    // test to check that the cell mapping is correct
    // try it first with some MeshFactory-generated quad meshes
    // then try with some arbitrarily permuted cell numberings
  }
  
  TEUCHOS_UNIT_TEST( MeshTransferFunction, FunctionValues)
  {
    // TODO: write this test
    // test to check that functions are correctly valued
    
    // try with some functions that simply return the cellID
    // and check that this matches the cell map.
    
    // important to try this test on multiple MPI ranks...
  }
//  TEUCHOS_UNIT_TEST( Int, Basic )
//  {
//    int i1 = 5;
//    TEST_EQUALITY_CONST( i1, 5 );
//  }
//  TEUCHOS_UNIT_TEST( Int, Assignment )
//  {
//    int i1 = 4;
//    int i2 = i1;
//    TEST_EQUALITY( i2, i1 );
//  }
} // namespace