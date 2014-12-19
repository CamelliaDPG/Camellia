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
 *  ElementTypeFactory.cpp
 *
 */

#include "ElementTypeFactory.h"
#include "CellTopology.h"

using namespace Camellia;

ElementTypePtr ElementTypeFactory::getElementType( DofOrderingPtr trialOrderPtr, DofOrderingPtr testOrderPtr, CellTopoPtr cellTopoPtr) {
  pair< CellTopologyKey, pair<  DofOrdering*, DofOrdering* > > key; // int is cellTopo key
  key = make_pair( cellTopoPtr->getKey(), make_pair( trialOrderPtr.get(), testOrderPtr.get() ) );
  if ( _elementTypes.find(key) == _elementTypes.end() ) {
    Teuchos::RCP< ElementType > typePtr = Teuchos::rcp( new ElementType( trialOrderPtr, testOrderPtr, cellTopoPtr ) );
    _elementTypes[key] = typePtr;
  }
  return _elementTypes[key];
}

ElementTypePtr ElementTypeFactory::getElementType(DofOrderingPtr trialOrderPtr, DofOrderingPtr testOrderPtr, CellTopoPtrLegacy shardsTopo) {
  CellTopoPtr cellTopo = CellTopology::cellTopology(*shardsTopo);
  return getElementType(trialOrderPtr, testOrderPtr, cellTopo);
}