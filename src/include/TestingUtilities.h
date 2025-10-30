// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
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

namespace Camellia
{
class TestingUtilities
{
public:
  static bool isBCDof(GlobalIndexType dof, TSolutionPtr<double> solution);
  static bool isFluxOrTraceDof(MeshPtr mesh, GlobalIndexType globalDofIndex);
  static void getGlobalFieldFluxDofInds(MeshPtr mesh, map<GlobalIndexType,set<GlobalIndexType> > &fluxIndices, map<GlobalIndexType,set<GlobalIndexType> > &fieldIndices);
  //  static void getDofIndices(MeshPtr mesh, set<int> &allFluxInds, map<int,vector<int> > &globalFluxInds, map<int, vector<int> > &globalFieldInds, map<int,vector<int> > &localFluxInds, map<int,vector<int> > &localFieldInds);
  //  static void getFieldFluxDofInds(MeshPtr mesh, map<int,set<int> > &localFluxInds, map<int,set<int> > &localFieldInds);


  static TSolutionPtr<double> makeNullSolution(MeshPtr mesh)
  {
    BCPtr nullBC = Teuchos::rcp((BC*)NULL);
    RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
    IPPtr nullIP = Teuchos::rcp((IP*)NULL);
    return Teuchos::rcp(new TSolution<double>(mesh, nullBC, nullRHS, nullIP) );
  }
  static double zero()
  {
    return 0.0;
  }
};
}

#endif
