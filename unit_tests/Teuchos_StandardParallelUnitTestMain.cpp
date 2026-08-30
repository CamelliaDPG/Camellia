// @HEADER
// ***********************************************************************
//
//                    Teuchos: Common Tools Package
//                 Copyright (2004) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
// @HEADER


/*! \file Teuchos_StandardUnitTestMain.cpp

\brief Standard Parallel Unit testing main program.

This file is ment to be used as a standard main program for a unit test
executable for parallel unit tests.

NOTE: This file should *not* be built and included as part of the Teuchos
library.  It is instead to be directly included in the build files for
specific unit test suites.

*/

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_StackedTimer.hpp"
#include "Teuchos_UnitTestRepository.hpp"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "Kokkos_Core.hpp"

#include "Camellia.h"

int main( int argc, char* argv[] )
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  Teuchos::UnitTestRepository::setGloballyReduceTestResult(true);

  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  Teuchos::RCP<const Teuchos::Comm<int>> comm = Teuchos::DefaultComm<int>::getComm();

  std::string timingsFilePath = "./test-timings.log";
  // parse CLI input to look for timing output request
  {
    auto & cmdp = Teuchos::UnitTestRepository::getCLP();
    cmdp.setOption("test-timings-log", &timingsFilePath, "File for stacked timer log output; will overwrite existing");
  }
  
  comm->barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  auto timer = Teuchos::TimeMonitor::getStackedTimer();
  int result = Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);
  Teuchos::StackedTimer::OutputOptions options;
  options.output_histogram = true;
  options.output_fraction  = true;
  options.output_minmax    = true;
  timer->stop ("Teuchos::StackedTimer");
  
  if ( timingsFilePath != "" )
  {
    std::ofstream fout(timingsFilePath.c_str());
    timer->report(fout, comm, options);
  }

  int rank = comm->getRank();
  bool printAllocationReport = false; // (rank == 0);
  
  Camellia::finalize(printAllocationReport);
  Kokkos::finalize();

  return result;
  
}
