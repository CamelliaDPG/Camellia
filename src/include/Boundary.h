#ifndef DPG_BOUNDARY
#define DPG_BOUNDARY

// @HEADER
//
// Original version Copyright © 2011 Sandia Corporation. All Rights Reserved.
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
 *  Boundary.h
 *
 */

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Intrepid_FieldContainer.hpp"

#include "Element.h"

#include "IndexType.h"

class Mesh;
class BC;

using namespace Intrepid;

class Boundary {
  set< pair< GlobalIndexType, unsigned > > _boundaryElements; // first arg is cellID, second arg is sideOrdinal
  // rebuild the next two structures whenever element types have changed (after mesh build or refinement)
  map< ElementType*, vector< pair< GlobalIndexType, int > > > _boundaryElementsByType; // entries in vector are indices into mesh's
                                                                     // enumeration of elements of ElementType, paired
                                                                     // with the sideIndex of the boundary in the element
  map< ElementType*, vector< GlobalIndexType > > _boundaryCellIDs; // ordering matches the pairs in _boundaryElementsByType
  Mesh *_mesh;
public:
  Boundary();
  void setMesh(Mesh* mesh);
  bool boundaryElement( GlobalIndexType cellID );
  bool boundaryElement( GlobalIndexType cellID, int sideIndex );
  vector< pair<GlobalIndexType, int > > boundaryElements(Teuchos::RCP< ElementType > elemTypePtr);
  void bcsToImpose(FieldContainer<GlobalIndexType> &globalIndices, FieldContainer<double> &globalValues, BC &bc, set<GlobalIndexType>& globalIndexFilter);
  void bcsToImpose(FieldContainer<GlobalIndexType> &globalIndices, FieldContainer<double> &globalValues, BC &bc);
  void bcsToImpose( map< GlobalIndexType, double > &globalDofIndicesAndValues, BC &bc, Teuchos::RCP< ElementType > elemTypePtr,
                   map <int, bool> &isSingleton);
  void buildLookupTables();
  //bool cellIsBoundaryElement(int cellID);
  //void elementChangedType( Teuchos::RCP< Element > elemPtr, Teuchos::RCP< ElementType > oldType,
  //                        Teuchos::RCP< ElementType > newType);
};

#endif