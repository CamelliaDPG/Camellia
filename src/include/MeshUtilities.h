#ifndef MESH_UTIL
#define MESH_UTIL

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

#include "Mesh.h"
#include "BilinearForm.h"
#include "SpatialFilter.h"

#include "IP.h"

class MeshUtilities {
 public:

  static SpatialFilterPtr rampBoundary(double rampHeight);
  static MeshPtr buildRampMesh(double rampHeight, Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pTest);

  static SpatialFilterPtr longRampBoundary(double rampHeight);
  static MeshPtr buildLongRampMesh(double rampHeight, Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pTest);

  static MeshPtr buildFrontFacingStep(Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pTest);

  static MeshPtr buildUnitQuadMesh(int horizontalCells, int verticalCells, Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pTest);
  
  static MeshPtr buildUnitQuadMesh(int nCells, Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pTest);

  static double computeMaxLocalConditionNumber(IPPtr ip, MeshPtr mesh, bool jacobiScaling=true, string sparseFileToWriteTo="");
  
};

#endif
