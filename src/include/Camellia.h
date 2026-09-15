// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//@HEADER

// Header file suitable for #including in drivers that use Camellia.

#ifndef Camellia_Camellia_h
#define Camellia_Camellia_h

#include "BasisFactory.h"
#include "BC.h"
#include "BF.h"
#include "CamelliaCellTools.h"
#include "Camellia_FieldContainer.hpp"
#include "CondensedDofInterpreter.h"
#include "ConvectionDiffusionFormulation.h"
#include "ExpFunction.h"
#include "Function.h"
#include "GMGOperator.h"
#include "GMGSolver.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "GLVisExporter.h"
#include "LinearTerm.h"
#include "Mesh.h"
#include "MeshFactory.h"
#include "NavierStokesVGPFormulation.h"
#include "PoissonFormulation.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "SharedScratchStorage.h"
#include "SimpleFunction.h"
#include "Solution.h"
#include "Solver.h"
#include "StokesVGPFormulation.h"
#include "TimeLogger.h"
#include "TrigFunctions.h"
#include "TypeDefs.h"
#include "Var.h"
#include "VarFactory.h"

namespace Camellia
{
  inline void finalize(bool printAllocationReports=false)
  {
    CamelliaCellTools::clearStaticMaps();
    RefinementPattern::clearStaticMaps();
    clearSharedScratchStorage();
    
    auto sharedScratchStorageHitCount = SharedScratchStorage<double>::sharedScratchStorage().hitCount();
    
    BasisFactory::deleteBasisFactory();
    if (printAllocationReports)
    {
      std::cout << "SharedScratchStorage avoided " << sharedScratchStorageHitCount << " device allocations.\n";
      
      std::string fcAllocationReport         = allocationReport(fcAllocationRecord);
      std::string deviceViewAllocationReport = allocationReport(deviceViewAllocationRecord);
      std::cout << "\n***********************  FieldContainer allocation report ***********************\n";
      std::cout << fcAllocationReport << std::endl;
      
      constexpr bool memorySpacesMatch = std::is_same<HostMemorySpace, DeviceMemorySpace>::value;
      if (memorySpacesMatch)
      {
        std::cout << "**************************** \"Device\" allocation report ************************************\n";
        std::cout << "** (These are allocations that really occurred on host, but would be on device for CUDA ) **\n";
        std::cout << deviceViewAllocationReport << std::endl;
        AllocationRecord totalDeviceAllocationReport;
        totalDeviceAllocationReport.allocationCount = deviceViewAllocationRecord.allocationCount + deviceViewFakeAllocationRecord.allocationCount;
        totalDeviceAllocationReport.bytesAllocatedNet = deviceViewAllocationRecord.bytesAllocatedNet + deviceViewFakeAllocationRecord.bytesAllocatedNet;
        totalDeviceAllocationReport.bytesAllocatedNetHighWaterMark = deviceViewAllocationRecord.bytesAllocatedNetHighWaterMark + deviceViewFakeAllocationRecord.bytesAllocatedNetHighWaterMark;
        totalDeviceAllocationReport.bytesAllocatedTotal = deviceViewAllocationRecord.bytesAllocatedTotal + deviceViewFakeAllocationRecord.bytesAllocatedTotal;
        std::string totalDeviceViewAllocationReport = allocationReport(totalDeviceAllocationReport);
        std::cout << "**************************** Total Device allocation report *****************************************\n";
        std::cout << "** ( These are allocations that would be on device for CUDA, some of which were avoided on host  ) **\n";
        std::cout << "** (Note that due to SharedScratchStorage, the actual numbers on CUDA may be substantially lower ) **\n";
        std::cout << totalDeviceViewAllocationReport << std::endl;
      }
      else
      {
        std::cout << "**************************  Device allocation report ****************************\n";
        std::cout << "**                                                                             **\n";
        std::cout << deviceViewAllocationReport << std::endl;
      }
      
      // TODO: fix deep copy report (and records) to reflect a real/fake deep copy distinction
      std::cout << "\n************************** Deep Copy Report ******************************\n";
      std::cout <<   "** (Note: only counts calls to Kokkos::deep_copy() from FieldContainer) **\n";
      std::cout << "device-to-host: " << deviceToHostDeepCopyRecord.deepCopyCount << " deep copies, ";
      std::cout << deviceToHostDeepCopyRecord.bytesDeepCopied << " bytes copied.\n";
      std::cout << "host-to-device: " << hostToDeviceDeepCopyRecord.deepCopyCount << " deep copies, ";
      std::cout << hostToDeviceDeepCopyRecord.bytesDeepCopied << " bytes copied.\n\n";

      if (recordOutstandingFCAllocationNumbers)
      {
        std::cout << "There are " << outstandingFCAllocationNumbers.size() << " outstanding FC allocation numbers.\n";
        std::cout << "Outstanding FC allocation numbers: ";
        for (auto allocationNumber : outstandingFCAllocationNumbers)
        {
          std::cout << allocationNumber << " ";
        }
        std::cout << std::endl;
      }
      
      const int maxNumEntries = 10;
      std::string noDevicePathReport = functionsWithoutDevicePathImplementationsReport(maxNumEntries);
      if (callsToFunctionsWithoutDevicePathImplementations.size() > 0)
      {
        std::cout << noDevicePathReport;
      }
      
      // Teuchos::TimeMonitor::summarize(); // prints to console.  TODO: output to a log file
    }
  }
}

#endif
