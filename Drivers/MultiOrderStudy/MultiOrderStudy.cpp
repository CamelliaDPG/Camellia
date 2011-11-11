// @HEADER
//
// Copyright © 2011 Nathan Roberts. All Rights Reserved.
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
// THIS SOFTWARE IS PROVIDED BY NATHAN ROBERTS "AS IS" AND ANY 
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


//
//  MixedOrderStudy.cpp
//  DPGTrilinos
//
//  Created by Nate Roberts on 9/1/11.
//

#include "MultiOrderStudy.h"

Teuchos::RCP<Mesh> MultiOrderStudy::makeMultiOrderMesh16x16(const FieldContainer<double> &quadBoundaryPoints,
                                                            Teuchos::RCP<BilinearForm> bilinearForm,
                                                            int lowH1Order, int pToAdd,
                                                            bool useTriangles) {
  // first, make a vanilla 16x16 mesh:
  int horizontalElements = 16, verticalElements = 16;
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadBoundaryPoints, 
                                                horizontalElements, verticalElements,
                                                bilinearForm, lowH1Order, lowH1Order + pToAdd, useTriangles);
  /* refinement pattern goes like
   
     4 4 4 4  1 1 1 1  2 2 2 2  3 3 3 3
     3 3 3 3  4 4 4 4  1 1 1 1  2 2 2 2
     2 2 2 2  3 3 3 3  4 4 4 4  1 1 1 1
     1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4
  
   and repeats vertically (4x)
   
   if i counts from bottom to top, j from left to right (i,j 0-based):
   numRefinements = ((i % 4) + j / 4) % 4
   
   */
  
  FieldContainer<int> cellIDs;
  if (useTriangles) {
    cellIDs.resize(horizontalElements,verticalElements,2);
  } else {
    cellIDs.resize(horizontalElements,verticalElements);
  }
  
  // get the cellIDs as they're laid out in the quad mesh returned by buildQuadMesh:
  Mesh::quadMeshCellIDs(cellIDs, horizontalElements, verticalElements, useTriangles);
  
  vector<int> cellsToPRefine, cellsToHRefine;
  
  //cout << cellIDs;
  
  for (int i=0; i<horizontalElements; i++) {
    for (int j=0; j<verticalElements; j++) {
      int numRefinements = ((i % 4) + j / 4) % 4;
      for (int k=0; k<numRefinements; k++) {
        if (useTriangles) {
          cellsToPRefine.push_back(cellIDs(i,j,0));
          cellsToPRefine.push_back(cellIDs(i,j,1));
        } else {
          cellsToPRefine.push_back(cellIDs(i,j));
        }
      }
    }
  }
  mesh->refine(cellsToPRefine,cellsToHRefine);
  return mesh;
}