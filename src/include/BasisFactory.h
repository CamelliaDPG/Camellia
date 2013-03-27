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

#ifndef BASIS_FACTORY
#define BASIS_FACTORY

// Intrepid includes
#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Basis.h"

#include "MultiBasis.h"
#include "PatchBasis.h"
#include "VectorizedBasis.h"

#include "CamelliaIntrepidExtendedTypes.h"

using namespace std;
using namespace IntrepidExtendedTypes;

typedef Teuchos::RCP< Camellia::Basis<> > BasisPtr;

class BasisFactory {
private:
  static map< pair< pair<int,int>, IntrepidExtendedTypes::EFunctionSpaceExtended >, BasisPtr >
              _existingBasis; // keys are ((polyOrder,cellTopoKey),fs))
  
  // the following maps let us remember what arguments were used to create a basis:
  // (this is useful to, say, create a basis again, but now with polyOrder+1)
  static map< Camellia::Basis<>*, int > _polyOrders; // allows lookup of poly order used to create basis
//  static map< Camellia::Basis<>*, int > _ranks; // allows lookup of basis rank
  static map< Camellia::Basis<>*, IntrepidExtendedTypes::EFunctionSpaceExtended > _functionSpaces; // allows lookup of function spaces
  static map< Camellia::Basis<>*, int > _cellTopoKeys; // allows lookup of cellTopoKeys
  static set< Camellia::Basis<>*> _multiBases;
  static map< vector< Camellia::Basis<>* >, MultiBasisPtr > _multiBasesMap;
  static map< pair<Camellia::Basis<>*, vector<double> >, PatchBasisPtr > _patchBases;
  static set< Camellia::Basis<>* > _patchBasisSet;
  
  static bool _useEnrichedTraces; // i.e. p+1, not p (default is true: this is what we need to prove optimal convergence)
public:
  static BasisPtr getBasis( int polyOrder, unsigned cellTopoKey, IntrepidExtendedTypes::EFunctionSpaceExtended fs);
  static BasisPtr getBasis(int &basisRank, int polyOrder, unsigned cellTopoKey, IntrepidExtendedTypes::EFunctionSpaceExtended fs);
  static MultiBasisPtr getMultiBasis(vector< BasisPtr > &bases);
  static PatchBasisPtr getPatchBasis(BasisPtr parent, FieldContainer<double> &patchNodesInParentRefCell, unsigned cellTopoKey = shards::Line<2>::key);

  static BasisPtr addToPolyOrder(BasisPtr basis, int pToAdd);
  static BasisPtr setPolyOrder(BasisPtr basis, int polyOrderToSet);
  
  static int basisPolyOrder(BasisPtr basis);
  static int getBasisRank(BasisPtr basis);
  static IntrepidExtendedTypes::EFunctionSpaceExtended getBasisFunctionSpace(BasisPtr basis);
  
  static bool basisKnown(BasisPtr basis);
  static bool isMultiBasis(BasisPtr basis);
  static bool isPatchBasis(BasisPtr basis);
  
  static void registerBasis( BasisPtr basis, int basisRank, int polyOrder, int cellTopoKey, IntrepidExtendedTypes::EFunctionSpaceExtended fs );
  
  static void setUseEnrichedTraces( bool value );
  
  // the following convenience methods belong in Basis or perhaps a wrapper thereof
  static set<int> sideFieldIndices( BasisPtr basis, bool includeSideSubcells = true); // includeSideSubcells: e.g. include vertices as part of quad sides
  
};

#endif
