#ifndef FD_UTIL
#define FD_UTIL

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

/*
 *  FiniteDifferenceUtilities.h
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */

// Epetra includes
#include <Epetra_Map.h>
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "RieszRep.h"
#include "Solution.h"
#include "Mesh.h"

namespace Camellia
{
class FiniteDifferenceUtilities
{
public:
  static double finiteDifferenceGradient(MeshPtr mesh, RieszRepPtr residual, TSolutionPtr<double> backgroundSoln, int dofIndex);
  static Camellia::FieldContainer<double> getDPGGradient(); // assumes you only linearize in the field variables
};
}

#endif
