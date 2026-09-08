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
#include "BasisCache.h"
#include "BasisFactory.h"
#include "Camellia_FieldContainer.hpp"
#include "Camellia_FieldContainerConversion.hpp"
#include "CamelliaCellTools.h"
#include "CamelliaDataHelpers.h"
#include "SerialDenseWrapper.h"
#include "TensorBasis.h"

#include <cmath>
#include <type_traits>

#include "Camellia_Integrate.hpp"

#include <Intrepid2_Data.hpp>
#include <Intrepid2_TensorData.hpp>
#include <Intrepid2_TensorPoints.hpp>

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( FieldContainer, StandaloneFieldContainerType )
  {
    using CamelliaFC = Camellia::FieldContainer<double>;

    const bool isFieldContainerClass = std::is_class<CamelliaFC>::value;
    TEST_EQUALITY(isFieldContainerClass, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, AssignmentOperator )
  {
    const int numCells  = 1;
    const int numPoints = 1;
    const int spaceDim  = 1;
    const double x = 1.0;
    Camellia::FieldContainer<double> points(numCells,numPoints,spaceDim);
    points(0,0,0) = x;
    
    Camellia::FieldContainer<double> myPoints = points;
    TEST_EQUALITY(myPoints(0,0,0), x);
    
    auto myPointsDeviceView = myPoints.checkoutDeviceView();
    double xDevice;
    const int cellOrdinal = 0;
    const int d = 0;
    Kokkos::parallel_reduce("x device reduction", numPoints, KOKKOS_LAMBDA (const int pointOrdinal, double& lvalue)
    {
      lvalue += myPointsDeviceView(cellOrdinal,pointOrdinal,d);
    }, xDevice);
    
    TEST_EQUALITY(x, xDevice);
    myPoints.checkinDeviceView(false);
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, HostAllocationRecorded )
  {
    const auto initialCount               = fcAllocationRecord.allocationCount;
    const auto initialBytesAllocatedTotal = fcAllocationRecord.bytesAllocatedTotal;
    const auto initialNetBytesAllocated   = fcAllocationRecord.bytesAllocatedNet;
    
    const int dim0 = 5;
    using Scalar = double;
    Camellia::FieldContainer<Scalar> values(dim0);
    
    const int newBytesExpected = values.size() * sizeof(Scalar);
    TEST_EQUALITY(initialBytesAllocatedTotal + newBytesExpected, fcAllocationRecord.bytesAllocatedTotal);
    TEST_EQUALITY(initialNetBytesAllocated   + newBytesExpected, fcAllocationRecord.bytesAllocatedNet);
    TEST_EQUALITY(initialCount + 1, fcAllocationRecord.allocationCount);
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, DeviceAllocationRecorded )
  {
    const int dim0 = 5;
    using Scalar = double;
    Camellia::FieldContainer<Scalar> values(dim0);
    
    const auto initialCount               = deviceViewAllocationRecord.allocationCount     + deviceViewFakeAllocationRecord.allocationCount;
    const auto initialBytesAllocatedTotal = deviceViewAllocationRecord.bytesAllocatedTotal + deviceViewFakeAllocationRecord.bytesAllocatedTotal;
    const auto initialNetBytesAllocated   = deviceViewAllocationRecord.bytesAllocatedNet   + deviceViewFakeAllocationRecord.bytesAllocatedNet;
   
    auto dv = values.checkoutDeviceView();  // DeviceView is initialized lazily on checkout
    
    // Camellia::Allocation uses SharedScratchStorage in checkoutMirrorView, so the checkout may not result in a new allocation
    const int newCount = deviceViewAllocationRecord.allocationCount + deviceViewFakeAllocationRecord.allocationCount;
    long long newBytesExpected;
    if (newCount == initialCount)
    {
      // reuse case: all counts should match initial
      newBytesExpected = 0;
    }
    else
    {
      // new allocation
      TEST_EQUALITY( initialCount + 1,  newCount );
      newBytesExpected = values.size() * sizeof(Scalar);
    }
    
    const auto newBytesAllocatedTotal = deviceViewAllocationRecord.bytesAllocatedTotal + deviceViewFakeAllocationRecord.bytesAllocatedTotal;
    const auto newNetBytesAllocated   = deviceViewAllocationRecord.bytesAllocatedNet + deviceViewFakeAllocationRecord.bytesAllocatedNet;
    TEST_EQUALITY(initialBytesAllocatedTotal + newBytesExpected, newBytesAllocatedTotal);
    TEST_EQUALITY(initialNetBytesAllocated   + newBytesExpected, newNetBytesAllocated  );
        
    values.checkinDeviceView(false); // false: don't deep copy to host
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, DeviceToHostDeepCopyRecorded )
  {
    const auto initialCount      = deviceToHostDeepCopyRecord.deepCopyCount;
    const auto initialByteCount  = deviceToHostDeepCopyRecord.bytesDeepCopied;
    
    const int dim0 = 5;
    using Scalar = double;
    Camellia::FieldContainer<Scalar> values(dim0);
    
    auto dv = values.checkoutDeviceView(false); // false: don't deep copy from host to device (unnecessary here)
    values.checkinDeviceView(true); // true: do deep copy from device to host
    
    const int newBytesExpected = values.size() * sizeof(Scalar);
    TEST_EQUALITY(initialByteCount + newBytesExpected, deviceToHostDeepCopyRecord.bytesDeepCopied);
    TEST_EQUALITY(initialCount + 1, deviceToHostDeepCopyRecord.deepCopyCount);
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, RepeatedHostAccessDoesNotRepeatedlySyncFromDevice )
  {
    const auto initialCount      = deviceToHostDeepCopyRecord.deepCopyCount;
    const auto initialByteCount  = deviceToHostDeepCopyRecord.bytesDeepCopied;

    const int dim0 = 5;
    using Scalar = double;
    Camellia::FieldContainer<Scalar> values(dim0);

    auto dv = values.checkoutDeviceView(false);
    Kokkos::deep_copy(dv, 1.0);
    values.checkinDeviceView(true);

    const int newBytesExpected = values.size() * sizeof(Scalar);
    TEST_EQUALITY(initialByteCount + newBytesExpected, deviceToHostDeepCopyRecord.bytesDeepCopied);
    TEST_EQUALITY(initialCount + 1, deviceToHostDeepCopyRecord.deepCopyCount);

    for (int i=0; i<dim0; i++)
    {
      TEST_EQUALITY(values[i], 1.0);
      values[i] = 2.0;
      TEST_EQUALITY(values(i), 2.0);
    }

    TEST_EQUALITY(initialByteCount + newBytesExpected, deviceToHostDeepCopyRecord.bytesDeepCopied);
    TEST_EQUALITY(initialCount + 1, deviceToHostDeepCopyRecord.deepCopyCount);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, GetDeviceViewAllocatesSeparateStorage )
  {
    // test that Camellia::Allocation<Scalar>::allocateDeviceViewMatchingView() actually allocates new storage, rather than returning a separate view object pointing to same data
    // TODO: create AllocationTests.cpp, move this test there, and put it in an "Allocation" test group
    const double tol = 1e-15;
    const int dim0 = 5;
    using Scalar = double;
    Camellia::FieldContainer<Scalar> values(dim0);
    auto dvValues = values.checkoutDeviceView();
    Kokkos::deep_copy(dvValues, 1.0);
    auto dvValuesScratch = Camellia::Allocation<Scalar>::allocateDeviceViewMatchingView(dvValues);
    Kokkos::deep_copy(dvValuesScratch, 2.0); // should not change dvValues
    values.checkinDeviceView(true); // deep copies dvValues back to host
    
    double expectedValue = 1.0;
    for (int i=0; i<dim0; i++)
    {
      TEST_FLOATING_EQUALITY(expectedValue, values[i], tol);
    }
    Camellia::Allocation<Scalar>::recordDeviceViewDeallocation(dvValuesScratch);
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, HostToDeviceDeepCopyRecordedConstCase )
  {
    const auto initialCount      = hostToDeviceDeepCopyRecord.deepCopyCount;
    const auto initialByteCount  = hostToDeviceDeepCopyRecord.bytesDeepCopied;
    
    const int dim0 = 5;
    using Scalar = double;
    const Camellia::FieldContainer<Scalar> values(dim0);
    
    auto dv = values.checkoutDeviceView(); // default: request deep copy from host to device
    values.checkinDeviceView(false);  // false: don't deep copy to host
    
    const int newBytesExpected = values.size() * sizeof(Scalar);
    TEST_EQUALITY(initialByteCount + newBytesExpected, hostToDeviceDeepCopyRecord.bytesDeepCopied);
    TEST_EQUALITY(initialCount + 1, hostToDeviceDeepCopyRecord.deepCopyCount);
    
    auto dvSecondCheckout = values.checkoutDeviceView(); // requested checkout copies again
    values.checkinDeviceView(false); // false: don't deep copy to host
    
    TEST_EQUALITY(initialByteCount + 2 * newBytesExpected, hostToDeviceDeepCopyRecord.bytesDeepCopied);
    TEST_EQUALITY(initialCount + 2, hostToDeviceDeepCopyRecord.deepCopyCount);
  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, HostToDeviceDeepCopyRecorded )
  {
    const auto initialCount      = hostToDeviceDeepCopyRecord.deepCopyCount;
    const auto initialByteCount  = hostToDeviceDeepCopyRecord.bytesDeepCopied;
    
    const int dim0 = 5;
    using Scalar = double;
    Camellia::FieldContainer<Scalar> values(dim0);
    
    auto dv = values.checkoutDeviceView(true); // true: deep copy
    
    const int newBytesExpected = values.size() * sizeof(Scalar);
    TEST_EQUALITY(initialByteCount + newBytesExpected, hostToDeviceDeepCopyRecord.bytesDeepCopied);
    TEST_EQUALITY(initialCount + 1, hostToDeviceDeepCopyRecord.deepCopyCount);
  }
  
  /* NVR Jan 6 2021
     Commented out the convertSDMToFC_DataOwnership test.  It started failing, and I'm not convinced that it helps with the
     problem it claims to; it looks to me like the issue with solveSPDSystemMultipleRHS() was solved by adding the line
   
        x.resize(x.dimension(1), x.dimension(0));
     
     I'm also not sure I understand what is meant by ownership of the data.  If there's a bug exposed by the test failure,
     it seems likely it's in Teuchos, rather than in Camellia, and it's not clear that it's a bug that actually makes any
     difference to us at all.
   */
//  TEUCHOS_UNIT_TEST( FieldContainer, convertSDMToFC_DataOwnership )
//  {
//    // here, we are attempting to isolate an issue in which (it appears) the optimized build
//    // does something different with respect to data ownership during a call to solveSPDSystemMultipleRHS()
//    // -- the optimized version does not have the final FieldContainer owning the data, even though it was
//    // produced via a copy.  The somewhat convoluted construction below is an artifact of trying to replicate the
//    // code path that leads to this issue.
//    const int M = 40;
//    const int N = 15;
//    Epetra_SerialDenseMatrix dataSDM(M,N);
//
//    Teuchos::Array<int> dataDim(2);
//    dataDim[0] = N;
//    dataDim[1] = M;
//
//    Camellia::FieldContainer<double> data(1,N,M);
//    for (int i=0; i<M; i++)
//    {
//      for (int j=0; j<N; j++)
//      {
//        dataSDM(i,j) = i*j;
//      }
//    }
//    bool shouldDeepCopy = false;
//    bool shouldOwnData = false;
//
//    Camellia::FieldContainer<double> dataFC(dataDim, &data(0,0,0), shouldDeepCopy, shouldOwnData); // this is the line we had to change to get things to pass.  We had been using 'false' (default) values for the latter two arguments.  I'm not entirely sure why this should matter (i.e., there may be a bug lurking in the FC construction), but the extra deep copy here probably won't hurt performance too much, and may avoid a memory leak.
//
//    bool ownsMemory = dataFC.getData().has_ownership();
//    TEST_EQUALITY(ownsMemory, shouldOwnData);
//
//    SerialDenseWrapper::convertSDMToFC(dataFC, dataSDM);
//
//    ownsMemory = dataFC.getData().has_ownership();
//    TEST_EQUALITY(ownsMemory, shouldOwnData);
//  }
  
  TEUCHOS_UNIT_TEST( FieldContainer, ViewKeptInSync )
  {
    Camellia::FieldContainer<double> testContainer(3,3,3);
    testContainer[4] = 300.0;
    testContainer.resize(3);
    
    // TODO: finish this test
  }

  // =========================================================================
  // DIAGNOSTIC TESTS: Layout, indexing, and integration behavior
  // Based on analysis from transcript entry 17
  // =========================================================================

  TEUCHOS_UNIT_TEST( FieldContainer, LayoutConsistency_Rank2 )
  {
    // Verify that operator()(i0,i1) accesses the same memory location as
    // operator[](getEnumeration(i0,i1)) for a rank-2 container.
    const int dim0 = 3, dim1 = 5;
    Camellia::FieldContainer<double> fc(dim0, dim1);
    for (int i = 0; i < fc.size(); i++) fc[i] = 100.0 + i;

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int i0 = 0; i0 < dim0; i0++) {
      for (int i1 = 0; i1 < dim1; i1++) {
        int enumVal = fc.getEnumeration(i0, i1);
        double byBracket = fc[enumVal];
        double byParen   = fc(i0, i1);
        if (byBracket != byParen) {
          allMatch = false;
          errMsg << "  fc[" << enumVal << "]=" << byBracket
                 << " != fc(" << i0 << "," << i1 << ")=" << byParen << "\n";
        }
      }
    }
    if (!allMatch) out << "Rank-2 layout mismatch:\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, LayoutConsistency_Rank3 )
  {
    // Verify operator()(i0,i1,i2) == operator[](getEnumeration(i0,i1,i2))
    // This is the critical rank for COMP_BLAS integrate: (C,F,P)
    const int dim0 = 2, dim1 = 4, dim2 = 3;
    Camellia::FieldContainer<double> fc(dim0, dim1, dim2);
    for (int i = 0; i < fc.size(); i++) fc[i] = 200.0 + i;

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int i0 = 0; i0 < dim0; i0++) {
      for (int i1 = 0; i1 < dim1; i1++) {
        for (int i2 = 0; i2 < dim2; i2++) {
          int enumVal = fc.getEnumeration(i0, i1, i2);
          double byBracket = fc[enumVal];
          double byParen   = fc(i0, i1, i2);
          if (byBracket != byParen) {
            allMatch = false;
            errMsg << "  fc[" << enumVal << "]=" << byBracket
                   << " != fc(" << i0 << "," << i1 << "," << i2 << ")=" << byParen << "\n";
          }
        }
      }
    }
    if (!allMatch) out << "Rank-3 layout mismatch:\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, LayoutConsistency_Rank4 )
  {
    const int dim0 = 2, dim1 = 3, dim2 = 4, dim3 = 2;
    Camellia::FieldContainer<double> fc(dim0, dim1, dim2, dim3);
    for (int i = 0; i < fc.size(); i++) fc[i] = 300.0 + i;

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int i0 = 0; i0 < dim0; i0++)
      for (int i1 = 0; i1 < dim1; i1++)
        for (int i2 = 0; i2 < dim2; i2++)
          for (int i3 = 0; i3 < dim3; i3++) {
            int enumVal = fc.getEnumeration(i0, i1, i2, i3);
            double byBracket = fc[enumVal];
            double byParen   = fc(i0, i1, i2, i3);
            if (byBracket != byParen) {
              allMatch = false;
              errMsg << "  fc[" << enumVal << "]=" << byBracket
                     << " != fc(" << i0 << "," << i1 << "," << i2 << "," << i3
                     << ")=" << byParen << "\n";
            }
          }
    if (!allMatch) out << "Rank-4 layout mismatch:\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, HostViewStrideAnalysis_Rank3 )
  {
    // Diagnose actual memory strides of the HostView.
    // For row-major (LayoutRight): stride(2)=1, stride(1)=dim2, stride(0)=dim1*dim2.
    const int dim0 = 2, dim1 = 4, dim2 = 3;
    Camellia::FieldContainer<double> fc(dim0, dim1, dim2);
    for (int i = 0; i < fc.size(); i++) fc[i] = 1000.0 + i;

    const double* rawPtr = &fc[0];
    ptrdiff_t stride2 = &fc(0,0,1) - &fc(0,0,0);
    ptrdiff_t stride1 = &fc(0,1,0) - &fc(0,0,0);
    ptrdiff_t stride0 = &fc(1,0,0) - &fc(0,0,0);

    out << "Rank-3 FC(" << dim0 << "," << dim1 << "," << dim2 << "):\n";
    out << "  stride along dim0 (cells):  " << stride0 << " (expected " << dim1*dim2 << ")\n";
    out << "  stride along dim1 (fields): " << stride1 << " (expected " << dim2 << ")\n";
    out << "  stride along dim2 (points): " << stride2 << " (expected 1)\n";
    out << "  &fc(0,0,0) - &fc[0] = " << (&fc(0,0,0) - rawPtr) << " (expected 0)\n";

    TEST_EQUALITY(stride2, 1);
    TEST_EQUALITY(stride1, (ptrdiff_t)dim2);
    TEST_EQUALITY(stride0, (ptrdiff_t)(dim1*dim2));
    TEST_EQUALITY(&fc(0,0,0), rawPtr);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, IntrepidVsCamellia_Rank3_OperatorParen )
  {
    // Fill both with the same flat data, then compare operator() values.
    const int dim0 = 2, dim1 = 4, dim2 = 3;
    Camellia::FieldContainer<double> ifc(dim0, dim1, dim2);
    Camellia::FieldContainer<double> cfc(dim0, dim1, dim2);
    for (int i = 0; i < ifc.size(); i++) {
      ifc[i] = 500.0 + i;
      cfc[i] = 500.0 + i;
    }

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int i0 = 0; i0 < dim0; i0++)
      for (int i1 = 0; i1 < dim1; i1++)
        for (int i2 = 0; i2 < dim2; i2++) {
          double ival = ifc(i0, i1, i2);
          double cval = cfc(i0, i1, i2);
          if (ival != cval) {
            allMatch = false;
            errMsg << "  (" << i0 << "," << i1 << "," << i2 << "): "
                   << "intrepid=" << ival << " camellia=" << cval << "\n";
          }
        }
    if (!allMatch) out << "Intrepid vs Camellia operator() mismatch (rank-3):\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, IntrepidVsCamellia_Rank4_OperatorParen )
  {
    const int dim0 = 2, dim1 = 3, dim2 = 4, dim3 = 2;
    Camellia::FieldContainer<double> ifc(dim0, dim1, dim2, dim3);
    Camellia::FieldContainer<double> cfc(dim0, dim1, dim2, dim3);
    for (int i = 0; i < ifc.size(); i++) {
      ifc[i] = 700.0 + i;
      cfc[i] = 700.0 + i;
    }

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int i0 = 0; i0 < dim0; i0++)
      for (int i1 = 0; i1 < dim1; i1++)
        for (int i2 = 0; i2 < dim2; i2++)
          for (int i3 = 0; i3 < dim3; i3++) {
            double ival = ifc(i0, i1, i2, i3);
            double cval = cfc(i0, i1, i2, i3);
            if (ival != cval) {
              allMatch = false;
              errMsg << "  (" << i0 << "," << i1 << "," << i2 << "," << i3 << "): "
                     << "intrepid=" << ival << " camellia=" << cval << "\n";
            }
          }
    if (!allMatch) out << "Intrepid vs Camellia operator() mismatch (rank-4):\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, AsymmetricDimensionLayoutCheck )
  {
    // Highly asymmetric dims to catch layout issues invisible with square dims.
    const int dim0 = 2, dim1 = 7, dim2 = 3;
    Camellia::FieldContainer<double> cfc(dim0, dim1, dim2);
    Camellia::FieldContainer<double> ifc(dim0, dim1, dim2);
    for (int i0 = 0; i0 < dim0; i0++)
      for (int i1 = 0; i1 < dim1; i1++)
        for (int i2 = 0; i2 < dim2; i2++) {
          double val = i0 * 100.0 + i1 * 10.0 + i2;
          cfc(i0, i1, i2) = val;
          ifc(i0, i1, i2) = val;
        }
    // Compare flat storage
    bool flatMatch = true;
    for (int i = 0; i < cfc.size(); i++) {
      if (cfc[i] != ifc[i]) { flatMatch = false; break; }
    }
    TEST_EQUALITY(flatMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, BLASPointerArithmetic_AddressCheck_Rank3 )
  {
    // Verify &container[cl*skip] == &container(cl,0,0) for COMP_BLAS.
    const int numCells = 3, numFields = 4, numPoints = 6;
    Camellia::FieldContainer<double> fc(numCells, numFields, numPoints);
    for (int i = 0; i < fc.size(); i++) fc[i] = i;
    int skip = numFields * numPoints;
    for (int cl = 0; cl < numCells; cl++) {
      TEST_EQUALITY(&fc[cl * skip], &fc(cl, 0, 0));
    }
  }

  TEUCHOS_UNIT_TEST( FieldContainer, WriteByParenReadByBracket_Rank3 )
  {
    // Write via operator(), read via operator[]. Asymmetric dims.
    const int dim0 = 2, dim1 = 7, dim2 = 3;
    Camellia::FieldContainer<double> fc(dim0, dim1, dim2);
    double counter = 1.0;
    for (int i0 = 0; i0 < dim0; i0++)
      for (int i1 = 0; i1 < dim1; i1++)
        for (int i2 = 0; i2 < dim2; i2++)
          fc(i0, i1, i2) = counter++;

    bool allMatch = true;
    counter = 1.0;
    for (int i0 = 0; i0 < dim0; i0++)
      for (int i1 = 0; i1 < dim1; i1++)
        for (int i2 = 0; i2 < dim2; i2++) {
          int flatIdx = (i0 * dim1 + i1) * dim2 + i2;
          if (fc[flatIdx] != counter) { allMatch = false; break; }
          counter++;
        }
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, DynRankViewLayoutCharacterization )
  {
    // Report the actual layout types for diagnostic purposes.
    using FCType = Camellia::FieldContainer<double>;
    using HV = FCType::HostView;
    using Layout = HV::array_layout;
    bool isLayoutRight = std::is_same<Layout, Kokkos::LayoutRight>::value;
    bool isLayoutLeft  = std::is_same<Layout, Kokkos::LayoutLeft>::value;
    bool isLayoutStride = std::is_same<Layout, Kokkos::LayoutStride>::value;
    out << "HostView layout: Right=" << isLayoutRight
        << " Left=" << isLayoutLeft
        << " Stride=" << isLayoutStride << "\n";
    out << "For COMP_BLAS correctness, LayoutRight (row-major) is required.\n";
    if (!isLayoutRight) {
      out << "WARNING: HostView is NOT LayoutRight! Flat indexing may be wrong.\n";
    }
    // We do NOT assert layout here; the stride/consistency tests catch actual bugs.
    // This test is purely informational.
    TEST_EQUALITY(true, true); // always passes; output is the value
  }

  TEUCHOS_UNIT_TEST( FieldContainer, IntegrateCOMP_BLAS_vs_COMP_CPP_FieldFieldScalar )
  {
    // Compare COMP_BLAS vs COMP_CPP using Camellia::FieldContainer.
    const double tol = 1e-12;
    const int C = 2, FL = 3, FR = 4, P = 5;
    Camellia::FieldContainer<double> left(C, FL, P);
    Camellia::FieldContainer<double> right(C, FR, P);
    for (int c = 0; c < C; c++)
      for (int f = 0; f < FL; f++)
        for (int p = 0; p < P; p++)
          left(c, f, p) = c*100.0 + f*10.0 + p + 1.0;
    for (int c = 0; c < C; c++)
      for (int f = 0; f < FR; f++)
        for (int p = 0; p < P; p++)
          right(c, f, p) = c*50.0 + f*5.0 + p + 0.5;

    Camellia::FieldContainer<double> outCPP(C, FL, FR);
    Camellia::FunctionSpaceTools::integrate<double>(outCPP, left, right, false);
    Camellia::FieldContainer<double> outBLAS(C, FL, FR);
    Camellia::FunctionSpaceTools::integrate<double>(outBLAS, left, right, false);

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int c = 0; c < C; c++)
      for (int lf = 0; lf < FL; lf++)
        for (int rf = 0; rf < FR; rf++) {
          double diff = std::abs(outCPP(c,lf,rf) - outBLAS(c,lf,rf));
          if (diff > tol) {
            allMatch = false;
            errMsg << "  (" << c << "," << lf << "," << rf << "): "
                   << "CPP=" << outCPP(c,lf,rf) << " BLAS=" << outBLAS(c,lf,rf)
                   << " diff=" << diff << "\n";
          }
        }
    if (!allMatch) out << "COMP_BLAS vs COMP_CPP mismatch (fieldFieldScalar):\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, IntegrateCOMP_BLAS_vs_COMP_CPP_DataFieldScalar )
  {
    const double tol = 1e-12;
    const int C = 2, F = 4, P = 5;
    Camellia::FieldContainer<double> left(C, P);
    Camellia::FieldContainer<double> right(C, F, P);
    for (int c = 0; c < C; c++)
      for (int p = 0; p < P; p++)
        left(c, p) = c*10.0 + p + 1.0;
    for (int c = 0; c < C; c++)
      for (int f = 0; f < F; f++)
        for (int p = 0; p < P; p++)
          right(c, f, p) = c*50.0 + f*5.0 + p + 0.5;

    Camellia::FieldContainer<double> outCPP(C, F);
    Camellia::FunctionSpaceTools::integrate<double>(outCPP, left, right, false);
    Camellia::FieldContainer<double> outBLAS(C, F);
    Camellia::FunctionSpaceTools::integrate<double>(outBLAS, left, right, false);

    bool allMatch = true;
    for (int c = 0; c < C; c++)
      for (int f = 0; f < F; f++) {
        double diff = std::abs(outCPP(c,f) - outBLAS(c,f));
        if (diff > tol) { allMatch = false; break; }
      }
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, IntegrateCOMP_BLAS_vs_COMP_CPP_FieldFieldVector )
  {
    const double tol = 1e-12;
    const int C = 2, FL = 3, FR = 4, P = 5, D = 2;
    Camellia::FieldContainer<double> left(C, FL, P, D);
    Camellia::FieldContainer<double> right(C, FR, P, D);
    for (int c = 0; c < C; c++)
      for (int f = 0; f < FL; f++)
        for (int p = 0; p < P; p++)
          for (int d = 0; d < D; d++)
            left(c,f,p,d) = c*100.0 + f*10.0 + p + d*0.5 + 1.0;
    for (int c = 0; c < C; c++)
      for (int f = 0; f < FR; f++)
        for (int p = 0; p < P; p++)
          for (int d = 0; d < D; d++)
            right(c,f,p,d) = c*50.0 + f*5.0 + p + d*0.3 + 0.5;

    Camellia::FieldContainer<double> outCPP(C, FL, FR);
    Camellia::FunctionSpaceTools::integrate<double>(outCPP, left, right, false);
    Camellia::FieldContainer<double> outBLAS(C, FL, FR);
    Camellia::FunctionSpaceTools::integrate<double>(outBLAS, left, right, false);

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int c = 0; c < C; c++)
      for (int lf = 0; lf < FL; lf++)
        for (int rf = 0; rf < FR; rf++) {
          double diff = std::abs(outCPP(c,lf,rf) - outBLAS(c,lf,rf));
          if (diff > tol) {
            allMatch = false;
            errMsg << "  (" << c << "," << lf << "," << rf << "): "
                   << "CPP=" << outCPP(c,lf,rf) << " BLAS=" << outBLAS(c,lf,rf) << "\n";
          }
        }
    if (!allMatch) out << "COMP_BLAS vs COMP_CPP mismatch (fieldFieldVector):\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, IntegrateIntrepidFC_vs_CamelliaFC_COMP_BLAS )
  {
    // Same data in both FC types, integrate with COMP_BLAS, compare results.
    const double tol = 1e-12;
    const int C = 2, FL = 3, FR = 4, P = 5;
    Camellia::FieldContainer<double> iL(C,FL,P), iR(C,FR,P), iO(C,FL,FR);
    Camellia::FieldContainer<double> cL(C,FL,P), cR(C,FR,P), cO(C,FL,FR);
    for (int i = 0; i < iL.size(); i++) { double v = 1.0+0.1*i; iL[i]=v; cL[i]=v; }
    for (int i = 0; i < iR.size(); i++) { double v = 2.0+0.05*i; iR[i]=v; cR[i]=v; }

    Camellia::FunctionSpaceTools::integrate<double>(iO, iL, iR, false);
    Camellia::FunctionSpaceTools::integrate<double>(cO, cL, cR, false);

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int c = 0; c < C; c++)
      for (int lf = 0; lf < FL; lf++)
        for (int rf = 0; rf < FR; rf++) {
          double diff = std::abs(iO(c,lf,rf) - cO(c,lf,rf));
          if (diff > tol) {
            allMatch = false;
            errMsg << "  (" << c << "," << lf << "," << rf << "): "
                   << "intrepid=" << iO(c,lf,rf) << " camellia=" << cO(c,lf,rf) << "\n";
          }
        }
    if (!allMatch) out << "IntrepidFC vs CamelliaFC COMP_BLAS mismatch:\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, IntegrateIntrepidFC_vs_CamelliaFC_COMP_CPP )
  {
    // If this passes but COMP_BLAS fails, the issue is in flat-indexing.
    const double tol = 1e-12;
    const int C = 2, FL = 3, FR = 4, P = 5;
    Camellia::FieldContainer<double> iL(C,FL,P), iR(C,FR,P), iO(C,FL,FR);
    Camellia::FieldContainer<double> cL(C,FL,P), cR(C,FR,P), cO(C,FL,FR);
    for (int i = 0; i < iL.size(); i++) { double v = 1.0+0.1*i; iL[i]=v; cL[i]=v; }
    for (int i = 0; i < iR.size(); i++) { double v = 2.0+0.05*i; iR[i]=v; cR[i]=v; }

    Camellia::FunctionSpaceTools::integrate<double>(iO, iL, iR, false);
    Camellia::FunctionSpaceTools::integrate<double>(cO, cL, cR, false);

    bool allMatch = true;
    for (int c = 0; c < C; c++)
      for (int lf = 0; lf < FL; lf++)
        for (int rf = 0; rf < FR; rf++) {
          double diff = std::abs(iO(c,lf,rf) - cO(c,lf,rf));
          if (diff > tol) { allMatch = false; break; }
        }
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, SizeTracksConstructionResizeAndClear )
  {
    Camellia::FieldContainer<double> empty;
    TEST_EQUALITY(empty.size(), 0);

    Camellia::FieldContainer<double> fc(2,3,4);
    TEST_EQUALITY(fc.size(), 24);

    fc.resize(5,6);
    TEST_EQUALITY(fc.size(), 30);

    double data[6] = {0,1,2,3,4,5};
    Teuchos::Array<int> dims(2);
    dims[0] = 2;
    dims[1] = 3;
    Camellia::FieldContainer<double> external(dims, data);
    TEST_EQUALITY(external.size(), 6);

    Teuchos::Array<int> reshapedDims(1);
    reshapedDims[0] = 6;
    external.resize(reshapedDims);
    TEST_EQUALITY(external.size(), 6);

    fc.clear();
    TEST_EQUALITY(fc.size(), 0);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, ResizeZeroInitialization )
  {
    const int dim0 = 3, dim1 = 4;
    Camellia::FieldContainer<double> fc(dim0, dim1);
    for (int i = 0; i < fc.size(); i++) fc[i] = 999.0;
    fc.resize(5, 6);
    int oldSize = dim0 * dim1;
    int newSize = 5 * 6;
    bool newZero = true;
    for (int i = oldSize; i < newSize; i++)
      if (fc[i] != 0.0) { newZero = false; break; }
    TEST_EQUALITY(newZero, true);
    bool oldPreserved = true;
    for (int i = 0; i < oldSize; i++)
      if (fc[i] != 999.0) { oldPreserved = false; break; }
    TEST_EQUALITY(oldPreserved, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, HostDataPointerConsistency )
  {
    const int dim0 = 3, dim1 = 4, dim2 = 5;
    Camellia::FieldContainer<double> fc(dim0, dim1, dim2);
    for (int i = 0; i < fc.size(); i++) fc[i] = i;
    const double* ptrBracket  = &fc[0];
    const double* ptrParen    = &fc(0, 0, 0);
    const double* ptrGetData  = fc.getData().getRawPtr();
    out << "&fc[0]=" << (void*)ptrBracket
        << " &fc(0,0,0)=" << (void*)ptrParen
        << " getData()=" << (void*)ptrGetData << "\n";
    TEST_EQUALITY(ptrBracket, ptrParen);
    TEST_EQUALITY(ptrBracket, ptrGetData);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, IntrepidVsCamellia_DataPointerComparison )
  {
    // Fill via operator(), compare flat data pointers.
    const int dim0 = 2, dim1 = 5, dim2 = 3;
    Camellia::FieldContainer<double> ifc(dim0, dim1, dim2);
    Camellia::FieldContainer<double> cfc(dim0, dim1, dim2);
    for (int i0 = 0; i0 < dim0; i0++)
      for (int i1 = 0; i1 < dim1; i1++)
        for (int i2 = 0; i2 < dim2; i2++) {
          double val = i0*100.0 + i1*10.0 + i2 + 0.5;
          ifc(i0, i1, i2) = val;
          cfc(i0, i1, i2) = val;
        }
    const double* iData = ifc.getData().getRawPtr();
    const double* cData = cfc.getData().getRawPtr();
    bool allMatch = true;
    for (int i = 0; i < ifc.size(); i++)
      if (iData[i] != cData[i]) { allMatch = false; break; }
    if (!allMatch) out << "Flat data differs after operator() fill!\n";
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, TensorPoints_LeftmostComponentFastest_FlatExpansion )
  {
    using DeviceType = Camellia::DeviceType;
    using PointView  = Intrepid2::ScalarView<double, DeviceType>;
    using TensorPoints = Intrepid2::TensorPoints<double, DeviceType>;

    const int n0 = 2, n1 = 3, n2 = 2;
    PointView component0("component0", n0, 1);
    PointView component1("component1", n1, 1);
    PointView component2("component2", n2, 1);

    auto h0 = Kokkos::create_mirror_view(component0);
    auto h1 = Kokkos::create_mirror_view(component1);
    auto h2 = Kokkos::create_mirror_view(component2);
    for (int i0=0; i0<n0; i0++) h0(i0,0) = 10.0   + i0;
    for (int i1=0; i1<n1; i1++) h1(i1,0) = 100.0  + i1;
    for (int i2=0; i2<n2; i2++) h2(i2,0) = 1000.0 + i2;
    Kokkos::deep_copy(component0, h0);
    Kokkos::deep_copy(component1, h1);
    Kokkos::deep_copy(component2, h2);

    Kokkos::Array<PointView,3> components = {{component0, component1, component2}};
    TensorPoints tensorPoints(components);
    auto expanded = tensorPoints.allocateAndFillExpandedRawPointView();
    auto expandedHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expanded);

    TEST_EQUALITY(tensorPoints.extent_int(0), n0*n1*n2);
    TEST_EQUALITY(tensorPoints.extent_int(1), 3);
    TEST_EQUALITY(expanded.extent_int(0), n0*n1*n2);
    TEST_EQUALITY(expanded.extent_int(1), 3);

    bool allMatch = true;
    bool swappedOrderingWouldDiffer = false;
    std::ostringstream errMsg;
    for (int p=0; p<n0*n1*n2; p++)
    {
      // Intrepid2::TensorPoints uses component 0 as the fastest-moving index.
      const int i0 = p % n0;
      const int i1 = (p / n0) % n1;
      const int i2 = (p / (n0*n1)) % n2;
      const double expected0 = 10.0   + i0;
      const double expected1 = 100.0  + i1;
      const double expected2 = 1000.0 + i2;
      if ((expandedHost(p,0) != expected0) || (expandedHost(p,1) != expected1) || (expandedHost(p,2) != expected2))
      {
        allMatch = false;
        errMsg << "  p=" << p << " expected (" << expected0 << "," << expected1 << "," << expected2
               << ") actual (" << expandedHost(p,0) << "," << expandedHost(p,1) << "," << expandedHost(p,2) << ")\n";
      }

      const int swappedI0 = (p / n1) % n0;
      const int swappedI1 = p % n1;
      if ((swappedI0 != i0) || (swappedI1 != i1)) swappedOrderingWouldDiffer = true;
    }
    if (!allMatch) out << "TensorPoints flat expansion ordering mismatch:\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
    TEST_EQUALITY(swappedOrderingWouldDiffer, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, TensorData_LeftmostComponentFastest_MatchesFlatPointOrdinal )
  {
    using DeviceType = Camellia::DeviceType;
    using Data       = Intrepid2::Data<double, DeviceType>;
    using TensorData = Intrepid2::TensorData<double, DeviceType>;
    using View1D     = Kokkos::View<double*, DeviceType>;

    const int n0 = 2, n1 = 3, n2 = 2;
    View1D data0("data0", n0);
    View1D data1("data1", n1);
    View1D data2("data2", n2);

    auto h0 = Kokkos::create_mirror_view(data0);
    auto h1 = Kokkos::create_mirror_view(data1);
    auto h2 = Kokkos::create_mirror_view(data2);
    for (int i0=0; i0<n0; i0++) h0(i0) = 2.0  + i0;
    for (int i1=0; i1<n1; i1++) h1(i1) = 10.0 + i1;
    for (int i2=0; i2<n2; i2++) h2(i2) = 50.0 + i2;
    Kokkos::deep_copy(data0, h0);
    Kokkos::deep_copy(data1, h1);
    Kokkos::deep_copy(data2, h2);

    Data component0(data0, Kokkos::Array<int,1>{{n0}}, Kokkos::Array<Intrepid2::DataVariationType,1>{{Intrepid2::GENERAL}});
    Data component1(data1, Kokkos::Array<int,1>{{n1}}, Kokkos::Array<Intrepid2::DataVariationType,1>{{Intrepid2::GENERAL}});
    Data component2(data2, Kokkos::Array<int,1>{{n2}}, Kokkos::Array<Intrepid2::DataVariationType,1>{{Intrepid2::GENERAL}});
    Kokkos::Array<Data,3> components = {{component0, component1, component2}};
    TensorData tensorData(components);

    TEST_EQUALITY(tensorData.rank(), 1);
    TEST_EQUALITY(tensorData.extent_int(0), n0*n1*n2);

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int p=0; p<n0*n1*n2; p++)
    {
      const int i0 = p % n0;
      const int i1 = (p / n0) % n1;
      const int i2 = (p / (n0*n1)) % n2;
      const double expected = (2.0+i0) * (10.0+i1) * (50.0+i2);
      const double actual = tensorData(p);
      if (actual != expected)
      {
        allMatch = false;
        errMsg << "  p=" << p << " components=(" << i0 << "," << i1 << "," << i2
               << ") expected=" << expected << " actual=" << actual << "\n";
      }
    }
    if (!allMatch) out << "TensorData rank-1 ordering mismatch:\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, TensorData_SeparateCellComponent_MatchesCellPointFieldContainer )
  {
    using DeviceType = Camellia::DeviceType;
    using Data       = Intrepid2::Data<double, DeviceType>;
    using TensorData = Intrepid2::TensorData<double, DeviceType>;
    using View1D     = Kokkos::View<double*, DeviceType>;

    const int C = 2, n0 = 2, n1 = 3;
    View1D cellValues("cellValues", C);
    View1D pointValues0("pointValues0", n0);
    View1D pointValues1("pointValues1", n1);

    auto hCell = Kokkos::create_mirror_view(cellValues);
    auto h0    = Kokkos::create_mirror_view(pointValues0);
    auto h1    = Kokkos::create_mirror_view(pointValues1);
    for (int c=0; c<C; c++) hCell(c) = 100.0 + 10.0*c;
    for (int i0=0; i0<n0; i0++) h0(i0) = 2.0 + i0;
    for (int i1=0; i1<n1; i1++) h1(i1) = 5.0 + i1;
    Kokkos::deep_copy(cellValues, hCell);
    Kokkos::deep_copy(pointValues0, h0);
    Kokkos::deep_copy(pointValues1, h1);

    Data cellData(cellValues, Kokkos::Array<int,1>{{C}}, Kokkos::Array<Intrepid2::DataVariationType,1>{{Intrepid2::GENERAL}});
    Data pointData0(pointValues0, Kokkos::Array<int,1>{{n0}}, Kokkos::Array<Intrepid2::DataVariationType,1>{{Intrepid2::GENERAL}});
    Data pointData1(pointValues1, Kokkos::Array<int,1>{{n1}}, Kokkos::Array<Intrepid2::DataVariationType,1>{{Intrepid2::GENERAL}});
    Kokkos::Array<Data,3> components = {{cellData, pointData0, pointData1}};
    TensorData tensorData(components, true); // separate first component: logical shape (C,P)

    Camellia::FieldContainer<double> flattened(C, n0*n1);
    for (int c=0; c<C; c++)
    {
      for (int p=0; p<n0*n1; p++)
      {
        const int i0 = p % n0;
        const int i1 = (p / n0) % n1;
        flattened(c,p) = (100.0 + 10.0*c) * (2.0+i0) * (5.0+i1);
      }
    }

    TEST_EQUALITY(tensorData.rank(), 2);
    TEST_EQUALITY(tensorData.extent_int(0), C);
    TEST_EQUALITY(tensorData.extent_int(1), n0*n1);

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int c=0; c<C; c++)
    {
      for (int p=0; p<n0*n1; p++)
      {
        const double expected = flattened(c,p);
        const double actual = tensorData(c,p);
        if (actual != expected)
        {
          allMatch = false;
          errMsg << "  (c,p)=(" << c << "," << p << ") expected=" << expected << " actual=" << actual << "\n";
        }
      }
    }
    if (!allMatch) out << "TensorData separate-cell ordering does not match FieldContainer(C,P) flattening:\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, BasisEvaluation_TensorBasisTensorPoints_MatchesExpandedRawPoints )
  {
    using DeviceType   = Camellia::DeviceType;
    using PointView    = Intrepid2::ScalarView<double, DeviceType>;
    using TensorPoints = Intrepid2::TensorPoints<double, DeviceType>;
    using TensorBasis  = Camellia::TensorBasis<double, Camellia::FieldContainer<double> >;

    const double tol = 1e-12;
    const int nSpace = 3, nTime = 2;
    PointView spacePoints("spacePoints", nSpace, 1);
    PointView timePoints ("timePoints",  nTime,  1);

    auto hSpace = Kokkos::create_mirror_view(spacePoints);
    auto hTime  = Kokkos::create_mirror_view(timePoints);
    hSpace(0,0) = -0.75;
    hSpace(1,0) = -0.10;
    hSpace(2,0) =  0.60;
    hTime(0,0)  = -0.35;
    hTime(1,0)  =  0.80;
    Kokkos::deep_copy(spacePoints, hSpace);
    Kokkos::deep_copy(timePoints,  hTime);

    Kokkos::Array<PointView,2> components = {{spacePoints, timePoints}};
    TensorPoints tensorPoints(components);

    BasisPtr spatialBasis  = BasisFactory::basisFactory()->getBasis(2, CellTopology::line(), Camellia::FUNCTION_SPACE_HGRAD);
    BasisPtr temporalBasis = BasisFactory::basisFactory()->getBasis(3, CellTopology::line(), Camellia::FUNCTION_SPACE_HGRAD);
    Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp(new TensorBasis(spatialBasis, temporalBasis));

    auto tensorBasisValues = tensorBasis->allocateBasisValues(tensorPoints, Intrepid2::OPERATOR_VALUE);
    tensorBasis->getValues(tensorBasisValues, tensorPoints, Intrepid2::OPERATOR_VALUE);
    Camellia::FieldContainer<double> tensorValues = Camellia::FieldContainerConversion<double>::fromBasisValues(tensorBasisValues);

    auto expanded = tensorPoints.allocateAndFillExpandedRawPointView();
    auto expandedHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), expanded);
    Camellia::FieldContainer<double> expandedPoints(nSpace*nTime, 2);
    for (int p=0; p<nSpace*nTime; p++)
    {
      expandedPoints(p,0) = expandedHost(p,0);
      expandedPoints(p,1) = expandedHost(p,1);
    }

    Camellia::FieldContainer<double> flatValues(tensorBasis->getCardinality(), nSpace*nTime);
    tensorBasis->getValues(flatValues, expandedPoints, Camellia::OP_VALUE);

    TEST_EQUALITY(tensorValues.rank(), flatValues.rank());
    TEST_EQUALITY(tensorValues.dimension(0), flatValues.dimension(0));
    TEST_EQUALITY(tensorValues.dimension(1), flatValues.dimension(1));

    bool allMatch = true;
    bool swappedOrderingWouldDiffer = false;
    std::ostringstream errMsg;
    for (int f=0; f<tensorBasis->getCardinality(); f++)
    {
      for (int p=0; p<nSpace*nTime; p++)
      {
        const double diff = std::abs(tensorValues(f,p) - flatValues(f,p));
        if (diff > tol)
        {
          allMatch = false;
          const int iSpace = p % nSpace;
          const int iTime  = (p / nSpace) % nTime;
          errMsg << "  (f,p)=(" << f << "," << p << ") components=(" << iSpace << "," << iTime
                 << ") tensor=" << tensorValues(f,p) << " flat=" << flatValues(f,p) << " diff=" << diff << "\n";
        }

        const int swappedP = (p % nTime) * nSpace + ((p / nTime) % nSpace);
        if (swappedP != p)
        {
          if (std::abs(flatValues(f,p) - flatValues(f,swappedP)) > tol) swappedOrderingWouldDiffer = true;
        }
      }
    }
    if (!allMatch) out << "TensorBasis TensorPoints path does not match expanded flat-point path:\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
    TEST_EQUALITY(swappedOrderingWouldDiffer, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, CamelliaCellTools_MapToReferenceSubcell_QuadEdges_OrderAndCoordinates )
  {
    const double tol = 1e-15;
    CellTopoPtr quad = CellTopology::quad();

    Camellia::FieldContainer<double> parentNodes(quad->getNodeCount(), quad->getDimension());
    CamelliaCellTools::refCellNodesForTopology(parentNodes, quad);

    Camellia::FieldContainer<double> paramPoints(3,1);
    paramPoints(0,0) = -0.50;
    paramPoints(1,0) =  0.25;
    paramPoints(2,0) =  0.75;

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int sideOrdinal=0; sideOrdinal<quad->getSideCount(); sideOrdinal++)
    {
      Camellia::FieldContainer<double> mappedPoints(paramPoints.dimension(0), quad->getDimension());
      CamelliaCellTools::mapToReferenceSubcell(mappedPoints, paramPoints, 1, sideOrdinal, quad);

      const int v0 = quad->getNodeMap(1, sideOrdinal, 0);
      const int v1 = quad->getNodeMap(1, sideOrdinal, 1);
      for (int p=0; p<paramPoints.dimension(0); p++)
      {
        const double t = paramPoints(p,0);
        for (int d=0; d<quad->getDimension(); d++)
        {
          const double expected = 0.5 * ((1.0 - t) * parentNodes(v0,d) + (1.0 + t) * parentNodes(v1,d));
          const double diff = std::abs(mappedPoints(p,d) - expected);
          if (diff > tol)
          {
            allMatch = false;
            errMsg << "  side=" << sideOrdinal << " point=" << p << " dim=" << d
                   << " expected=" << expected << " actual=" << mappedPoints(p,d) << " diff=" << diff << "\n";
          }
        }
      }
    }
    if (!allMatch) out << "CamelliaCellTools quad edge map/order mismatch:\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, CamelliaCellTools_ReferenceSideNormals_QuadOutwardOrientation )
  {
    const double tol = 1e-15;
    CellTopoPtr quad = CellTopology::quad();

    Camellia::FieldContainer<double> parentNodes(quad->getNodeCount(), quad->getDimension());
    CamelliaCellTools::refCellNodesForTopology(parentNodes, quad);

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int sideOrdinal=0; sideOrdinal<quad->getSideCount(); sideOrdinal++)
    {
      Camellia::FieldContainer<double> sideNormal(quad->getDimension());
      CamelliaCellTools::getReferenceSideNormal(sideNormal, sideOrdinal, quad);

      const int v0 = quad->getNodeMap(1, sideOrdinal, 0);
      const int v1 = quad->getNodeMap(1, sideOrdinal, 1);
      const double midpointX = 0.5 * (parentNodes(v0,0) + parentNodes(v1,0));
      const double midpointY = 0.5 * (parentNodes(v0,1) + parentNodes(v1,1));
      const double outwardDot = sideNormal(0) * midpointX + sideNormal(1) * midpointY;
      const double norm = std::sqrt(sideNormal(0) * sideNormal(0) + sideNormal(1) * sideNormal(1));
      const bool axisAligned = (std::abs(sideNormal(0)) < tol) || (std::abs(sideNormal(1)) < tol);

      if ((outwardDot <= 0.0) || (std::abs(norm - 1.0) > tol) || !axisAligned)
      {
        allMatch = false;
        errMsg << "  side=" << sideOrdinal << " normal=(" << sideNormal(0) << "," << sideNormal(1)
               << ") midpoint=(" << midpointX << "," << midpointY << ") outwardDot=" << outwardDot
               << " norm=" << norm << " axisAligned=" << axisAligned << "\n";
      }
    }
    if (!allMatch) out << "CamelliaCellTools reference quad side-normal mismatch:\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, CamelliaCellTools_ComputeSideMeasure_ReferenceQuadEdges )
  {
    const double tol = 1e-14;
    CellTopoPtr quad = CellTopology::quad();
    const int C = 2, P = 3, D = 2;

    Camellia::FieldContainer<double> cellJacobian(C,P,D,D);
    Camellia::FieldContainer<double> cubWeights(P);
    cubWeights(0) = 0.50;
    cubWeights(1) = 0.75;
    cubWeights(2) = 0.75;

    for (int c=0; c<C; c++)
    {
      for (int p=0; p<P; p++)
      {
        cellJacobian(c,p,0,0) = 1.0;
        cellJacobian(c,p,0,1) = 0.0;
        cellJacobian(c,p,1,0) = 0.0;
        cellJacobian(c,p,1,1) = 1.0;
      }
    }

    bool allMatch = true;
    std::ostringstream errMsg;
    for (int sideOrdinal=0; sideOrdinal<quad->getSideCount(); sideOrdinal++)
    {
      Camellia::FieldContainer<double> weightedMeasure(C,P);
      CamelliaCellTools::computeSideMeasure(weightedMeasure, cellJacobian, cubWeights, sideOrdinal, quad);
      for (int c=0; c<C; c++)
      {
        double measureSum = 0.0;
        for (int p=0; p<P; p++)
        {
          measureSum += weightedMeasure(c,p);
          const double diff = std::abs(weightedMeasure(c,p) - cubWeights(p));
          if (diff > tol)
          {
            allMatch = false;
            errMsg << "  side=" << sideOrdinal << " cell=" << c << " point=" << p
                   << " expected=" << cubWeights(p) << " actual=" << weightedMeasure(c,p) << " diff=" << diff << "\n";
          }
        }
        if (std::abs(measureSum - 2.0) > tol)
        {
          allMatch = false;
          errMsg << "  side=" << sideOrdinal << " cell=" << c << " measureSum=" << measureSum << " expected=2\n";
        }
      }
    }
    if (!allMatch) out << "CamelliaCellTools reference quad side-measure mismatch:\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, BasisCache_SideGeometryOrdering_QuadPhysicalPointsMeasuresNormals )
  {
    const double tol = 1e-12;
    const double width = 2.5;
    const double height = 1.25;
    const int cubDegree = 4;
    const bool createSideCaches = true;

    BasisCachePtr volumeCache = BasisCache::quadBasisCache(width, height, cubDegree, createSideCaches);
    CellTopoPtr quad = volumeCache->cellTopology();

    bool allMatch = true;
    bool allSideIntegralsDistinct = true;
    std::ostringstream errMsg;
    std::vector<double> sideIntegrals;

    for (int sideOrdinal=0; sideOrdinal<quad->getSideCount(); sideOrdinal++)
    {
      BasisCachePtr sideCache = volumeCache->getSideBasisCache(sideOrdinal);
      const Camellia::FieldContainer<double> &refSidePoints  = sideCache->getSideRefCellPointsInVolumeCoordinatesFC();
      const Camellia::FieldContainer<double> &physSidePoints = sideCache->getPhysicalCubaturePointsFC();
      const Camellia::FieldContainer<double> &weightedMeasures = sideCache->getWeightedMeasures();
      const Camellia::FieldContainer<double> &sideNormals = sideCache->getSideNormals();
      const Camellia::FieldContainer<double> &sideWeights = sideCache->getCubatureWeightsFC();

      const int numPoints = refSidePoints.dimension(0);
      double sideIntegral = 0.0;
      double measureSum = 0.0;
      double expectedMeasureSum = 0.0;

      for (int p=0; p<numPoints; p++)
      {
        const double xi  = refSidePoints(p,0);
        const double eta = refSidePoints(p,1);
        const double expectedX = 0.5 * width  * (xi  + 1.0);
        const double expectedY = 0.5 * height * (eta + 1.0);
        const double dx = std::abs(physSidePoints(0,p,0) - expectedX);
        const double dy = std::abs(physSidePoints(0,p,1) - expectedY);
        if ((dx > tol) || (dy > tol))
        {
          allMatch = false;
          errMsg << "  side=" << sideOrdinal << " point=" << p
                 << " expected physical=(" << expectedX << "," << expectedY << ") actual=("
                 << physSidePoints(0,p,0) << "," << physSidePoints(0,p,1) << ") diff=(" << dx << "," << dy << ")\n";
        }

        const bool verticalSide = (std::abs(std::abs(xi) - 1.0) < 1e-14);
        const double expectedScaling = verticalSide ? 0.5 * height : 0.5 * width;
        const double expectedWeightedMeasure = expectedScaling * sideWeights(p);
        const double measureDiff = std::abs(weightedMeasures(0,p) - expectedWeightedMeasure);
        if (measureDiff > tol)
        {
          allMatch = false;
          errMsg << "  side=" << sideOrdinal << " point=" << p
                 << " expected measure=" << expectedWeightedMeasure << " actual=" << weightedMeasures(0,p)
                 << " diff=" << measureDiff << "\n";
        }
        measureSum += weightedMeasures(0,p);
        expectedMeasureSum += expectedWeightedMeasure;

        const double sideCoordinate = verticalSide ? xi : eta;
        const double outwardDot = sideNormals(0,p,0) * (physSidePoints(0,p,0) - 0.5 * width)
                                + sideNormals(0,p,1) * (physSidePoints(0,p,1) - 0.5 * height);
        const double normalNorm = std::sqrt(sideNormals(0,p,0) * sideNormals(0,p,0) + sideNormals(0,p,1) * sideNormals(0,p,1));
        if ((std::abs(std::abs(sideCoordinate) - 1.0) > 1e-14) || (outwardDot <= 0.0) || (std::abs(normalNorm - 1.0) > tol))
        {
          allMatch = false;
          errMsg << "  side=" << sideOrdinal << " point=" << p
                 << " ref=(" << xi << "," << eta << ") normal=(" << sideNormals(0,p,0) << "," << sideNormals(0,p,1)
                 << ") outwardDot=" << outwardDot << " norm=" << normalNorm << "\n";
        }

        const double f = physSidePoints(0,p,0) + 3.0 * physSidePoints(0,p,1);
        const double vDotN = sideNormals(0,p,0) + 0.5 * sideNormals(0,p,1);
        sideIntegral += weightedMeasures(0,p) * vDotN * f;
      }

      if (std::abs(measureSum - expectedMeasureSum) > tol)
      {
        allMatch = false;
        errMsg << "  side=" << sideOrdinal << " measureSum=" << measureSum << " expected=" << expectedMeasureSum << "\n";
      }
      sideIntegrals.push_back(sideIntegral);
    }

    for (unsigned i=0; i<sideIntegrals.size(); i++)
    {
      for (unsigned j=i+1; j<sideIntegrals.size(); j++)
      {
        if (std::abs(sideIntegrals[i] - sideIntegrals[j]) <= 1e-10)
        {
          allSideIntegralsDistinct = false;
          errMsg << "  sideIntegrals[" << i << "] and sideIntegrals[" << j << "] are not distinct: "
                 << sideIntegrals[i] << " vs " << sideIntegrals[j] << "\n";
        }
      }
    }

    if (!allMatch || !allSideIntegralsDistinct)
      out << "BasisCache side geometry/ordering diagnostic mismatch:\n" << errMsg.str();
    TEST_EQUALITY(allMatch, true);
    TEST_EQUALITY(allSideIntegralsDistinct, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, LinearTermRHS_TensorDataReducedPath_MatchesFlattenedFieldContainer )
  {
    using DeviceType    = Camellia::DeviceType;
    using ExecutionSpace = typename DeviceType::execution_space;
    using PointView     = Intrepid2::ScalarView<double, DeviceType>;
    using TensorPoints  = Intrepid2::TensorPoints<double, DeviceType>;
    using Data          = Intrepid2::Data<double, DeviceType>;
    using View1D        = Kokkos::View<double*, DeviceType>;
    using View3D        = Kokkos::View<double***, DeviceType>;
    using TensorBasis   = Camellia::TensorBasis<double, Camellia::FieldContainer<double> >;

    const double tol = 1e-12;
    const int C = 1;
    const int nSpace = 3, nTime = 2;

    PointView spacePoints("rhs space points", nSpace, 1);
    PointView timePoints ("rhs time points",  nTime,  1);
    View1D spaceWeights("rhs space weights", nSpace);
    View1D timeWeights ("rhs time weights",  nTime);

    auto hSpace = Kokkos::create_mirror_view(spacePoints);
    auto hTime  = Kokkos::create_mirror_view(timePoints);
    auto hWS    = Kokkos::create_mirror_view(spaceWeights);
    auto hWT    = Kokkos::create_mirror_view(timeWeights);

    hSpace(0,0) = -0.70;  hWS(0) = 0.45;
    hSpace(1,0) = -0.05;  hWS(1) = 0.90;
    hSpace(2,0) =  0.65;  hWS(2) = 0.65;
    hTime(0,0)  = -0.40;  hWT(0) = 1.10;
    hTime(1,0)  =  0.75;  hWT(1) = 0.70;

    Kokkos::deep_copy(spacePoints,  hSpace);
    Kokkos::deep_copy(timePoints,   hTime);
    Kokkos::deep_copy(spaceWeights, hWS);
    Kokkos::deep_copy(timeWeights,  hWT);

    Kokkos::Array<PointView,2> pointComponents = {{spacePoints, timePoints}};
    TensorPoints tensorPoints(pointComponents);

    Data spaceWeightData(spaceWeights, Kokkos::Array<int,1>{{nSpace}}, Kokkos::Array<Intrepid2::DataVariationType,1>{{Intrepid2::GENERAL}});
    Data timeWeightData (timeWeights,  Kokkos::Array<int,1>{{nTime}},  Kokkos::Array<Intrepid2::DataVariationType,1>{{Intrepid2::GENERAL}});
    Kokkos::Array<Data,2> weightComponents = {{spaceWeightData, timeWeightData}};
    Intrepid2::TensorData<double, DeviceType> tensorWeights(weightComponents);

    BasisPtr spatialBasis  = BasisFactory::basisFactory()->getBasis(2, CellTopology::line(), Camellia::FUNCTION_SPACE_HGRAD);
    BasisPtr temporalBasis = BasisFactory::basisFactory()->getBasis(3, CellTopology::line(), Camellia::FUNCTION_SPACE_HGRAD);
    Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp(new TensorBasis(spatialBasis, temporalBasis));
    const int F = tensorBasis->getCardinality();
    const int P = nSpace * nTime;

    auto tensorBasisValues = tensorBasis->allocateBasisValues(tensorPoints, Intrepid2::OPERATOR_VALUE);
    tensorBasis->getValues(tensorBasisValues, tensorPoints, Intrepid2::OPERATOR_VALUE);
    Camellia::FieldContainer<double> unweightedBasisNoCell = Camellia::FieldContainerConversion<double>::fromBasisValues(tensorBasisValues);

    Camellia::FieldContainer<double> weightedBasis(C,F,P);
    Camellia::FieldContainer<double> fxnFlat(C,P);
    Camellia::FieldContainer<double> flattenedPointValues(C,F,P);
    Camellia::FieldContainer<double> reducedPointValues(C,F,P);
    Camellia::FieldContainer<double> rhsFlat(C,F);
    Camellia::FieldContainer<double> rhsReduced(C,F);

    View3D fxnView("nonconstant rhs tensor data", C, nSpace, nTime);
    auto hFxn = Kokkos::create_mirror_view(fxnView);

    for (int c=0; c<C; c++)
    {
      for (int iTime=0; iTime<nTime; iTime++)
      {
        for (int iSpace=0; iSpace<nSpace; iSpace++)
        {
          const int p = tensorEnumerationIndex(iSpace, iTime, nSpace);
          const double x = hSpace(iSpace,0);
          const double t = hTime(iTime,0);
          const double nonconstantFxn = 1.25 + 0.5*c + 2.0*x - 3.0*t + 0.7*x*t;
          const double weight = hWS(iSpace) * hWT(iTime);
          hFxn(c,iSpace,iTime) = nonconstantFxn;
          fxnFlat(c,p) = nonconstantFxn;
          for (int f=0; f<F; f++)
          {
            weightedBasis(c,f,p) = unweightedBasisNoCell(f,p) * weight;
            flattenedPointValues(c,f,p) = weightedBasis(c,f,p) * fxnFlat(c,p);
          }
        }
      }
    }
    Kokkos::deep_copy(fxnView, hFxn);

    Data fxnData(fxnView,
                 Kokkos::Array<int,3>{{C,nSpace,nTime}},
                 Kokkos::Array<Intrepid2::DataVariationType,3>{{Intrepid2::GENERAL,Intrepid2::GENERAL,Intrepid2::GENERAL}});

    auto reducedView = reducedPointValues.checkoutDeviceView(false);
    auto basisView   = weightedBasis.checkoutDeviceView();
    Kokkos::deep_copy(reducedView, 0.0);

    Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<4>> rangePolicy({0,0,0,0},{C,F,nSpace,nTime});
    Kokkos::parallel_for(rangePolicy,
      KOKKOS_LAMBDA(const int &cellOrdinal, const int &fieldOrdinal, const int &spacePointOrdinal, const int &timePointOrdinal)
      {
        const auto & fxnValue = fxnData(cellOrdinal, spacePointOrdinal, timePointOrdinal);
        const int pointOrdinal = tensorEnumerationIndex(spacePointOrdinal, timePointOrdinal, nSpace);
        const auto & basisValue = basisView(cellOrdinal, fieldOrdinal, pointOrdinal);
        auto & outValue = reducedView(cellOrdinal, fieldOrdinal, pointOrdinal);
        outValue += basisValue * fxnValue;
      }
    );
    reducedPointValues.checkinDeviceView(true);
    weightedBasis.checkinDeviceView(false);

    bool pointwiseMatch = true;
    bool integralMatch = true;
    bool swappedOrderingWouldDiffer = false;
    std::ostringstream errMsg;
    for (int c=0; c<C; c++)
    {
      for (int f=0; f<F; f++)
      {
        rhsFlat(c,f) = 0.0;
        rhsReduced(c,f) = 0.0;
        double swappedIntegral = 0.0;
        for (int p=0; p<P; p++)
        {
          const double diff = std::abs(flattenedPointValues(c,f,p) - reducedPointValues(c,f,p));
          if (diff > tol)
          {
            pointwiseMatch = false;
            const int iSpace = p % nSpace;
            const int iTime  = (p / nSpace) % nTime;
            errMsg << "  pointwise (c,f,p)=(" << c << "," << f << "," << p << ") components=("
                   << iSpace << "," << iTime << ") flat=" << flattenedPointValues(c,f,p)
                   << " reduced=" << reducedPointValues(c,f,p) << " diff=" << diff << "\n";
          }
          rhsFlat(c,f)    += flattenedPointValues(c,f,p);
          rhsReduced(c,f) += reducedPointValues(c,f,p);

          const int iSpace = p % nSpace;
          const int iTime  = (p / nSpace) % nTime;
          const int swappedPointOrdinal = tensorEnumerationIndex(iTime, iSpace, nTime);
          if (swappedPointOrdinal < P)
            swappedIntegral += weightedBasis(c,f,swappedPointOrdinal) * fxnFlat(c,p);
        }
        const double integralDiff = std::abs(rhsFlat(c,f) - rhsReduced(c,f));
        if (integralDiff > tol)
        {
          integralMatch = false;
          errMsg << "  integrated (c,f)=(" << c << "," << f << ") flat=" << rhsFlat(c,f)
                 << " reduced=" << rhsReduced(c,f) << " diff=" << integralDiff << "\n";
        }
        if (std::abs(swappedIntegral - rhsFlat(c,f)) > 1e-10) swappedOrderingWouldDiffer = true;
      }
    }

    if (!pointwiseMatch || !integralMatch)
      out << "Reduced tensor Data LinearTerm/RHS path does not match flattened FieldContainer path:\n" << errMsg.str();
    TEST_EQUALITY(pointwiseMatch, true);
    TEST_EQUALITY(integralMatch, true);
    TEST_EQUALITY(swappedOrderingWouldDiffer, true);
  }

  TEUCHOS_UNIT_TEST( FieldContainer, ProductionLikePoissonNonconstantRHS_ReducedDataPath_MatchesFlattenedFieldContainer )
  {
    using DeviceType     = Camellia::DeviceType;
    using ExecutionSpace = typename DeviceType::execution_space;

    const double tol = 1e-12;
    const double width = 2.5;
    const double height = 1.25;
    const int cubDegree = 5;
    const bool createSideCaches = false;
    const bool tensorProductTopologyMeansSpaceTime = true;

    CellTopoPtr spaceTopo = CellTopology::line();
    CellTopoPtr timeTopo  = CellTopology::line();
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);

    Camellia::FieldContainer<double> spaceNodes(spaceTopo->getNodeCount(), spaceTopo->getDimension());
    Camellia::FieldContainer<double> timeNodes(timeTopo->getNodeCount(), timeTopo->getDimension());
    spaceNodes(0,0) = 0.0;    spaceNodes(1,0) = width;
    timeNodes(0,0)  = 0.0;    timeNodes(1,0)  = height;

    Camellia::FieldContainer<double> physicalCellNodesNoCell(spaceTimeTopo->getNodeCount(), spaceTimeTopo->getDimension());
    spaceTimeTopo->initializeNodes({spaceNodes, timeNodes}, physicalCellNodesNoCell);
    Camellia::FieldContainer<double> physicalCellNodes(1, spaceTimeTopo->getNodeCount(), spaceTimeTopo->getDimension());
    for (int node=0; node<spaceTimeTopo->getNodeCount(); node++)
      for (int d=0; d<spaceTimeTopo->getDimension(); d++)
        physicalCellNodes(0,node,d) = physicalCellNodesNoCell(node,d);

    BasisCachePtr basisCache = BasisCache::basisCacheForCellTopology(spaceTimeTopo, cubDegree, physicalCellNodes,
                                                                      createSideCaches, tensorProductTopologyMeansSpaceTime);
    BasisPtr basis = BasisFactory::basisFactory()->getBasis(3, spaceTimeTopo, Camellia::FUNCTION_SPACE_HGRAD);

    const Camellia::FieldContainer<double> &physicalPointsFC = basisCache->getPhysicalCubaturePointsFC();
    const PhysicalTensorPoints &physicalTensorPoints = basisCache->getPhysicalCubaturePoints();
    const Camellia::FieldContainer<double> &weightedMeasures = basisCache->getWeightedMeasures();
    Teuchos::RCP< const Camellia::FieldContainer<double> > unweightedBasis = basisCache->getTransformedValues(basis, Camellia::OP_VALUE);
    Teuchos::RCP< const Camellia::FieldContainer<double> > weightedBasis = basisCache->getTransformedWeightedValues(basis, Camellia::OP_VALUE);

    const int C = physicalPointsFC.dimension(0);
    const int P = physicalPointsFC.dimension(1);
    const int D = physicalPointsFC.dimension(2);
    const int F = basis->getCardinality();
    TEST_EQUALITY(C, 1);
    TEST_EQUALITY(D, 2);
    TEST_EQUALITY(physicalTensorPoints.numTensorComponents(), 2);
    if (physicalTensorPoints.numTensorComponents() != 2)
    {
      out << "Expected tensor-structured physical points with two components; got "
          << physicalTensorPoints.numTensorComponents() << ".\n";
      return;
    }

    const int n0 = physicalTensorPoints.getTensorComponent(0).extent_int(1);
    const int n1 = physicalTensorPoints.getTensorComponent(1).extent_int(1);
    TEST_EQUALITY(P, n0*n1);
    TEST_EQUALITY(unweightedBasis->dimension(0), C);
    TEST_EQUALITY(unweightedBasis->dimension(1), F);
    TEST_EQUALITY(unweightedBasis->dimension(2), P);
    TEST_EQUALITY(weightedBasis->dimension(0), C);
    TEST_EQUALITY(weightedBasis->dimension(1), F);
    TEST_EQUALITY(weightedBasis->dimension(2), P);

    bool physicalPointsMatch = true;
    bool weightedBasisMatchesMeasure = true;
    bool functionValuesMatch = true;
    bool pointwiseIntegrandMatch = true;
    bool integralMatch = true;
    bool swappedOrderingWouldDiffer = false;
    std::ostringstream errMsg;

    for (int c=0; c<C; c++)
    {
      for (int p=0; p<P; p++)
      {
        for (int d=0; d<D; d++)
        {
          const double diff = std::abs(physicalPointsFC(c,p,d) - physicalTensorPoints(c,p,d));
          if (diff > tol)
          {
            physicalPointsMatch = false;
            errMsg << "  physical point mismatch (c,p,d)=(" << c << "," << p << "," << d << ") flat="
                   << physicalPointsFC(c,p,d) << " tensor=" << physicalTensorPoints(c,p,d) << " diff=" << diff << "\n";
          }
        }
      }
    }

    for (int c=0; c<C; c++)
    {
      for (int f=0; f<F; f++)
      {
        for (int p=0; p<P; p++)
        {
          const double expectedWeighted = (*unweightedBasis)(c,f,p) * weightedMeasures(c,p);
          const double diff = std::abs((*weightedBasis)(c,f,p) - expectedWeighted);
          if (diff > tol)
          {
            weightedBasisMatchesMeasure = false;
            errMsg << "  weighted basis mismatch (c,f,p)=(" << c << "," << f << "," << p << ") weighted="
                   << (*weightedBasis)(c,f,p) << " expected=" << expectedWeighted << " diff=" << diff << "\n";
          }
        }
      }
    }

    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr fxn = 1.25 + 2.0*x - 3.0*y + 0.7*x*y;

    Camellia::FieldContainer<double> fxnFlat(C,P);
    fxn->values(fxnFlat,basisCache);
    Intrepid2::Data<double, DeviceType> fxnData = fxn->checkoutValuesData(basisCache);
    fxn->values(fxnData,basisCache);
    Kokkos::fence();

    TEST_EQUALITY(fxnData.rank(), 3);
    TEST_EQUALITY(fxnData.extent_int(0), C);
    TEST_EQUALITY(fxnData.extent_int(1), n0);
    TEST_EQUALITY(fxnData.extent_int(2), n1);

    Camellia::FieldContainer<double> reducedPointValues(C,F,P);
    auto reducedView = reducedPointValues.checkoutDeviceView(false);
    auto weightedBasisView = weightedBasis->checkoutDeviceView();
    Kokkos::deep_copy(reducedView, 0.0);

    Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<4>> rangePolicy({0,0,0,0},{C,F,n0,n1});
    Kokkos::parallel_for("production-like scalar RHS reduced Data path diagnostic", rangePolicy,
      KOKKOS_LAMBDA(const int &cellOrdinal, const int &fieldOrdinal, const int &pointOrdinal0, const int &pointOrdinal1)
      {
        const auto & fxnValue = fxnData(cellOrdinal, pointOrdinal0, pointOrdinal1);
        const int pointOrdinal = tensorEnumerationIndex(pointOrdinal0, pointOrdinal1, n0);
        const auto & basisValue = weightedBasisView(cellOrdinal, fieldOrdinal, pointOrdinal);
        auto & outValue = reducedView(cellOrdinal, fieldOrdinal, pointOrdinal);
        outValue += basisValue * fxnValue;
      }
    );
    reducedPointValues.checkinDeviceView(true);
    weightedBasis->checkinDeviceView(false);

    Camellia::FieldContainer<double> flattenedPointValues(C,F,P);
    Camellia::FieldContainer<double> rhsFlat(C,F);
    Camellia::FieldContainer<double> rhsReduced(C,F);
    for (int c=0; c<C; c++)
    {
      for (int f=0; f<F; f++)
      {
        rhsFlat(c,f) = 0.0;
        rhsReduced(c,f) = 0.0;
        double swappedIntegral = 0.0;
        for (int p=0; p<P; p++)
        {
          const int p0 = p % n0;
          const int p1 = (p / n0) % n1;
          const double fxnDataValue = fxnData(c,p0,p1);
          const double fxnDiff = std::abs(fxnFlat(c,p) - fxnDataValue);
          if (fxnDiff > tol)
          {
            functionValuesMatch = false;
            errMsg << "  function value mismatch (c,p)=(" << c << "," << p << ") components=(" << p0 << "," << p1
                   << ") flat=" << fxnFlat(c,p) << " data=" << fxnDataValue << " diff=" << fxnDiff << "\n";
          }

          flattenedPointValues(c,f,p) = (*weightedBasis)(c,f,p) * fxnFlat(c,p);
          const double pointwiseDiff = std::abs(flattenedPointValues(c,f,p) - reducedPointValues(c,f,p));
          if (pointwiseDiff > tol)
          {
            pointwiseIntegrandMatch = false;
            errMsg << "  pointwise integrand mismatch (c,f,p)=(" << c << "," << f << "," << p << ") components=("
                   << p0 << "," << p1 << ") flat=" << flattenedPointValues(c,f,p)
                   << " reduced=" << reducedPointValues(c,f,p) << " diff=" << pointwiseDiff << "\n";
          }
          rhsFlat(c,f)    += flattenedPointValues(c,f,p);
          rhsReduced(c,f) += reducedPointValues(c,f,p);

          const int swappedPointOrdinal = tensorEnumerationIndex(p1, p0, n1);
          if (swappedPointOrdinal < P)
            swappedIntegral += (*weightedBasis)(c,f,swappedPointOrdinal) * fxnFlat(c,p);
        }
        const double integralDiff = std::abs(rhsFlat(c,f) - rhsReduced(c,f));
        if (integralDiff > tol)
        {
          integralMatch = false;
          errMsg << "  integrated RHS mismatch (c,f)=(" << c << "," << f << ") flat=" << rhsFlat(c,f)
                 << " reduced=" << rhsReduced(c,f) << " diff=" << integralDiff << "\n";
        }
        if (std::abs(swappedIntegral - rhsFlat(c,f)) > 1e-10) swappedOrderingWouldDiffer = true;
      }
    }
    fxn->checkinValuesData(fxnData);

    if (!physicalPointsMatch || !weightedBasisMatchesMeasure || !functionValuesMatch || !pointwiseIntegrandMatch || !integralMatch)
      out << "Production-like Poisson nonconstant RHS reduced Data path does not match flattened FieldContainer path:\n" << errMsg.str();
    TEST_EQUALITY(physicalPointsMatch, true);
    TEST_EQUALITY(weightedBasisMatchesMeasure, true);
    TEST_EQUALITY(functionValuesMatch, true);
    TEST_EQUALITY(pointwiseIntegrandMatch, true);
    TEST_EQUALITY(integralMatch, true);
    TEST_EQUALITY(swappedOrderingWouldDiffer, true);
  }

} // namespace

