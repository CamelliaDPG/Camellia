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

namespace Camellia {
	class MeshTestUtility {
	  static bool determineRefTestPointsForNeighbors(MeshTopologyPtr meshTopo, CellPtr fineCell, unsigned sideOrdinal,
	                                                 Intrepid::FieldContainer<double> &fineSideRefPoints, Intrepid::FieldContainer<double> &fineCellRefPoints,
	                                                 Intrepid::FieldContainer<double> &coarseSideRefPoints, Intrepid::FieldContainer<double> &coarseCellRefPoints); // returns false if neighbor at sideOrdinal is broken
	  
	public:
    static bool checkLocalGlobalConsistency(MeshPtr mesh, double tol=1e-10);
    
    // checkMeshDofConnectivities() and checkMeshConsistency() only support 2D Maximum rule meshes
    static bool checkMeshDofConnectivities(Teuchos::RCP<Mesh> mesh2DMaximumRule);
	  static bool checkMeshConsistency(Teuchos::RCP<Mesh> mesh2DMaximumRule);
    
    static bool fcsAgree(const Intrepid::FieldContainer<double> &fc1, const Intrepid::FieldContainer<double> &fc2, double tol, double &maxDiff); // redundant with / copied from TestSuite::fcsAgree()
	  static bool neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh);
	  
	  static bool neighborBasesAgreeOnSides(Teuchos::RCP<Mesh> mesh, Epetra_MultiVector &globalSolutionCoefficients);
	};
}

#endif
