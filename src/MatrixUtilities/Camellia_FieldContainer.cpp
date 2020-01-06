// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

/*
 *  Camellia_FieldContainer.cpp
 *
 *  Created by Nathan Roberts on 10/28/19.
 *
 */

#include "Camellia_FieldContainer.hpp"

namespace Camellia
{
  const std::string DEFAULT_VIEW_NAME = "Camellia FieldContainer View";
  
  AllocationRecord fcAllocationRecord;
  AllocationRecord deviceViewAllocationRecord;
  
  std::string allocationReport(AllocationRecord &record)
  {
    std::ostringstream allocationReportStream;
    
    double totalInMB  = double(record.bytesAllocatedTotal) / 1024.0 / 1024.0;
    double netInMB    = double(record.bytesAllocatedNet) / 1024.0 / 1024.0;
    double netHWMInMB = double(record.bytesAllocatedNetHighWaterMark) / 1024.0 / 1024.0;
    
    allocationReportStream << record.allocationCount << " Views/FieldContainers have been allocated.\n";
    allocationReportStream << "Total bytes allocated:                " << record.bytesAllocatedTotal << " (" << totalInMB << " MB)\n";
    allocationReportStream << "Net bytes allocated:                  " << record.bytesAllocatedNet   << " (" << netInMB   << " MB)\n";
    allocationReportStream << "Net bytes allocated, high water mark: " << record.bytesAllocatedNetHighWaterMark << " (" << netHWMInMB << " MB)\n";
    return allocationReportStream.str();
  }
} // namespace Camellia
