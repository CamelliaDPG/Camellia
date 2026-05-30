// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//  MeshTestUtility.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/17/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_MeshTestUtility_h
#define Camellia_debug_MeshTestUtility_h

#include "GDAMinimumRule.h"
#include "Mesh.h"

namespace Camellia
{
class MeshTestUtility
{
  static bool determineRefTestPointsForNeighbors(MeshTopologyViewPtr meshTopo, CellPtr fineCell, unsigned sideOrdinal,
      Camellia::FieldContainer<double> &fineSideRefPoints, Camellia::FieldContainer<double> &fineCellRefPoints,
      Camellia::FieldContainer<double> &coarseSideRefPoints, Camellia::FieldContainer<double> &coarseCellRefPoints); // returns false if neighbor at sideOrdinal is broken

public:
  static bool checkLocalGlobalConsistency(MeshPtr mesh, double tol=1e-10);

  // ! Method for checking consistency of all constraints in minimum-rule mesh.  Checks on present rank only (non-communicating method)
  static bool checkConstraintConsistency(MeshPtr meshMinimumRule);
  
  // ! Returns true if the indicated constrainingEntity is a (generalized) ancestor of the constrained entity in meshTopo.
  static bool constraintIsConsistent(MeshTopologyViewPtr meshTopo, AnnotatedEntity &constrainingEntity,
                                     int d, IndexType constrainedEntityIndex, bool requireConstrainingEntityBelongToActiveCell);
  
  static bool fcsAgree(const Camellia::FieldContainer<double> &fc1, const Camellia::FieldContainer<double> &fc2, double tol, double &maxDiff); // redundant with / copied from TestSuite::fcsAgree()
  static bool neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh);

//  static bool neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh, Epetra_MultiVector &globalSolutionCoefficients);
  
  static bool neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh, TVectorPtr<double> globalSolutionCoefficients);
};
}

#endif
