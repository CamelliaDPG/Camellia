#ifndef DPG_ELEMENT_TYPE_FACTORY
#define DPG_ELEMENT_TYPE_FACTORY

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
 *  ElementTypeFactory.h
 *
 */

#include "TypeDefs.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "ElementType.h"

#include "CellTopology.h"

namespace Camellia
{
class ElementTypeFactory
{
  std::map< std::pair< Camellia::CellTopologyKey, std::pair<  DofOrdering*, DofOrdering* > >,
      Teuchos::RCP< ElementType > > _elementTypes;

public:
  Teuchos::RCP< ElementType > getElementType(DofOrderingPtr trialOrderPtr, DofOrderingPtr testOrderPtr, CellTopoPtr cellTopoPtr);
  Teuchos::RCP< ElementType > getElementType(DofOrderingPtr trialOrderPtr, DofOrderingPtr testOrderPtr, CellTopoPtrLegacy cellTopoPtr);
};
}

#endif