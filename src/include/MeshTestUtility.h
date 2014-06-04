//
//  MeshTestUtility.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/17/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_MeshTestUtility_h
#define Camellia_debug_MeshTestUtility_h

#include "Mesh.h"

class MeshTestUtility {
  static bool determineRefTestPointsForNeighbors(MeshTopologyPtr meshTopo, CellPtr fineCell, unsigned sideOrdinal,
                                                 FieldContainer<double> &fineSideRefPoints, FieldContainer<double> &fineCellRefPoints,
                                                 FieldContainer<double> &coarseSideRefPoints, FieldContainer<double> &coarseCellRefPoints); // returns false if neighbor at sideOrdinal is broken
  
public:
  static bool checkMeshDofConnectivities(Teuchos::RCP<Mesh> mesh);
  static bool checkMeshConsistency(Teuchos::RCP<Mesh> mesh);
  static bool fcsAgree(const FieldContainer<double> &fc1, const FieldContainer<double> &fc2, double tol, double &maxDiff); // redundant with / copied from TestSuite::fcsAgree()
  static bool neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh);
  
  static bool neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh, Epetra_Vector &globalSolutionCoefficients);
};

#endif
