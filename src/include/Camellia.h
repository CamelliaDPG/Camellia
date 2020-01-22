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
#include "LinearTerm.h"
#include "Mesh.h"
#include "MeshFactory.h"
#include "NavierStokesVGPFormulation.h"
#include "PoissonFormulation.h"
#include "RefinementStrategy.h"
#include "RHS.h"
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
    
    BasisFactory::deleteBasisFactory();
    if (printAllocationReports)
    {
      std::string fcAllocationReport         = allocationReport(fcAllocationRecord);
      std::string deviceViewAllocationReport = allocationReport(deviceViewAllocationRecord);
      std::cout << "\n***********************  FieldContainer allocation report ***********************\n";
      std::cout << fcAllocationReport << std::endl;
      std::cout << "**************************  Device allocation report ****************************\n";
      std::cout << "** (Note: these are generally not real allocations for non-CUDA memory spaces) **\n";
      std::cout << deviceViewAllocationReport << std::endl;

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
    }
  }
}

#endif
