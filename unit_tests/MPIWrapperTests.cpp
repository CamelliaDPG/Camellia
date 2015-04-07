//
//  MPIWrapperTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/25/15.
//
//

#include "MPIWrapper.h"

#include "Intrepid_FieldContainer.hpp"
#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;
using namespace Intrepid;

#include "Teuchos_UnitTestHarness.hpp"
namespace {
  TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( MPIWrapper, AllGatherCompact, Scalar )
  {
    int myRank = Teuchos::GlobalMPISession::getRank();
    int numProcs = Teuchos::GlobalMPISession::getNProc();
    
    FieldContainer<int> expectedOffsets(numProcs);
    int numGlobalEntries = 0; // counts the total number of entries
    for (int i=0; i < numProcs; i++) {
      expectedOffsets[i] = numGlobalEntries;
      for (int j=0; j <= i; j++) {
        numGlobalEntries++;
      }
    }
    
    int myOffset = expectedOffsets[myRank];
    
    // put a variable number of values on each processor
    FieldContainer<Scalar> myValues(myRank+1);
    for (int i=0; i<=myRank; i++) {
      myValues[i] = myOffset + i;
    }
    
//    std::cout << "rank " << myRank << " values:";
//    for (int i=0; i<=myRank; i++) {
//      std::cout << " " << myValues[i];
//    }
//    std::cout << std::endl;
    
    FieldContainer<Scalar> allValuesExpected(numGlobalEntries);
    for (int i=0; i<numGlobalEntries; i++) {
      allValuesExpected[i] = i;
    }
    
    FieldContainer<Scalar> allValues;
    FieldContainer<int> offsets;
    MPIWrapper::allGatherCompact(allValues,myValues,offsets);

    TEST_COMPARE_ARRAYS(expectedOffsets, offsets);
    
    TEST_COMPARE_ARRAYS(allValuesExpected, allValues);
  }
  
  TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( MPIWrapper, AllGatherCompact, int );
  TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( MPIWrapper, AllGatherCompact, double );

} // namespace
