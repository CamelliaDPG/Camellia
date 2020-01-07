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

#include "Epetra_SerialDenseMatrix.h"
#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "Camellia_FieldContainer.hpp"
#include "SerialDenseWrapper.h"

using namespace Camellia;

namespace
{
  void getDataCopy(Camellia::FieldContainer<double>& dataCopy, const double *dataPtr, int n, int m)
  {
    Teuchos::Array<int> dim(2);
    dim[0] = m;
    dim[1] = n;
    double * firstEntry = (double *) dataPtr; // casting away the const.  OK since we copy below.
    dataCopy = Camellia::FieldContainer<double>(dim,firstEntry,true); // true: copy
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, convertSDMToFC_DataOwnership )
  {
    // here, we are attempting to isolate an issue in which (it appears) the optimized build
    // does something different with respect to data ownership during a call to solveSPDSystemMultipleRHS()
    // -- the optimized version does not have the final FieldContainer owning the data, even though it was
    // produced via a copy.  The somewhat convoluted construction below is an artifact of trying to replicate the
    // code path that leads to this issue.
    const int M = 40;
    const int N = 15;
    Epetra_SerialDenseMatrix dataSDM(M,N);
    
    Teuchos::Array<int> dataDim(2);
    dataDim[0] = N;
    dataDim[1] = M;
    
    Camellia::FieldContainer<double> data(1,N,M);
    for (int i=0; i<M; i++)
    {
      for (int j=0; j<N; j++)
      {
        dataSDM(i,j) = i*j;
      }
    }
    bool shouldDeepCopy = false;
    bool shouldOwnData = false;
    
    Camellia::FieldContainer<double> dataFC(dataDim, &data(0,0,0), shouldDeepCopy, shouldOwnData); // this is the line we had to change to get things to pass.  We had been using 'false' (default) values for the latter two arguments.  I'm not entirely sure why this should matter (i.e., there may be a bug lurking in the FC construction), but the extra deep copy here probably won't hurt performance too much, and may avoid a memory leak.
    
    bool ownsMemory = dataFC.getData().has_ownership();
    TEST_EQUALITY(ownsMemory, shouldOwnData);
    
    SerialDenseWrapper::convertSDMToFC(dataFC, dataSDM);
    
    ownsMemory = dataFC.getData().has_ownership();
    TEST_EQUALITY(ownsMemory, shouldOwnData);
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, ViewKeptInSync )
  {
    Camellia::FieldContainer<double> testContainer(3,3,3);
    testContainer[4] = 300.0;
    testContainer.resize(3);
    
    // TODO: finish this test
  }
} // namespace
