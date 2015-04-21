// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of
// conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

/*
 *  TestingUtilities.h
 *
 *
 */

#ifndef TEST_UTIL
#define TEST_UTIL

#include "TypeDefs.h"

// Epetra includes
#include <Epetra_Map.h>
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "Mesh.h"
#include "Solution.h"

namespace Camellia {
  class TestingUtilities {
   public:
    static bool isBCDof(GlobalIndexType dof, TSolutionPtr<double> solution);
    static bool isFluxOrTraceDof(MeshPtr mesh, GlobalIndexType globalDofIndex);
    static void initializeSolnCoeffs(TSolutionPtr<double> solution);
    static void setSolnCoeffForGlobalDofIndex(TSolutionPtr<double> solution, double solnCoeff, GlobalIndexType dofIndex);
    static void getGlobalFieldFluxDofInds(MeshPtr mesh, map<GlobalIndexType,set<GlobalIndexType> > &fluxIndices, map<GlobalIndexType,set<GlobalIndexType> > &fieldIndices);
    //  static void getDofIndices(MeshPtr mesh, set<int> &allFluxInds, map<int,vector<int> > &globalFluxInds, map<int, vector<int> > &globalFieldInds, map<int,vector<int> > &localFluxInds, map<int,vector<int> > &localFieldInds);
    //  static void getFieldFluxDofInds(MeshPtr mesh, map<int,set<int> > &localFluxInds, map<int,set<int> > &localFieldInds);


    static TSolutionPtr<double> makeNullSolution(MeshPtr mesh){
      BCPtr nullBC = Teuchos::rcp((BC*)NULL);
      RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
      IPPtr nullIP = Teuchos::rcp((IP*)NULL);
      return Teuchos::rcp(new TSolution<double>(mesh, nullBC, nullRHS, nullIP) );
    }
    static double zero(){
      return 0.0;
    }
  };
}

#endif
