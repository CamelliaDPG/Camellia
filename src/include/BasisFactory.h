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

#include "BilinearForm.h"
#include "MultiBasis.h"
#include "PatchBasis.h"
#include "Vectorized_Basis.hpp"

using namespace Intrepid;
using namespace std;

/*
 NOTES on what needs to be done to support arbitrary CellTopology (e.g. curvilinear elements):
 Basically, we need to remove the dependence in BasisFactor on cellTopoKeys, and replace that with CellTopology pointers.
 In order to allow optimal reuse, we'll then have to implement a CellTopologyFactory that makes sure that the pointers to
 a particular CellTopology are the same.
 
 Actually, it occurs to me that we can isolate many of the changes to current code to BasisFactory: for calls involving the
 cellTopoKeys, we simply use this to call the appropriate CellTopologyFactory method to get the pointer to the right CellTopology.
 
 */

class BasisFactory {
private:
  typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
  typedef Teuchos::RCP< MultiBasis > MultiBasisPtr;
  typedef Teuchos::RCP< PatchBasis > PatchBasisPtr;
  typedef Teuchos::RCP<Vectorized_Basis<double, FieldContainer<double> > > VectorBasisPtr;
  static map< pair< pair<int,int>, EFunctionSpaceExtended >, BasisPtr >
              _existingBasis; // keys are ((polyOrder,cellTopoKey),fs))
  
  // the following maps let us remember what arguments were used to create a basis:
  // (this is useful to, say, create a basis again, but now with polyOrder+1)
  static map< Basis<double,FieldContainer<double> >*, int > _polyOrders; // allows lookup of poly order used to create basis
  static map< Basis<double,FieldContainer<double> >*, int > _ranks; // allows lookup of basis rank
  static map< Basis<double,FieldContainer<double> >*, EFunctionSpaceExtended > _functionSpaces; // allows lookup of function spaces
  static map< Basis<double,FieldContainer<double> >*, int > _cellTopoKeys; // allows lookup of cellTopoKeys
  static set< Basis<double,FieldContainer<double> >*> _multiBases;
  static map< vector< Basis<double,FieldContainer<double> >* >, MultiBasisPtr > _multiBasesMap;
  static map< pair<Basis<double,FieldContainer<double> >*, vector<double> >, PatchBasisPtr > _patchBasesLines;
public:
  static BasisPtr getBasis( int polyOrder, unsigned cellTopoKey, EFunctionSpaceExtended fs);
  static BasisPtr getBasis(int &basisRank, int polyOrder, unsigned cellTopoKey, EFunctionSpaceExtended fs);
  static MultiBasisPtr getMultiBasis(vector< BasisPtr > &bases);
  static PatchBasisPtr getPatchBasis(BasisPtr parent, FieldContainer<double> &patchNodesInParentRefCell, unsigned cellTopoKey = shards::Line<2>::key);

  static BasisPtr addToPolyOrder(BasisPtr basis, int pToAdd);
  static BasisPtr setPolyOrder(BasisPtr basis, int polyOrderToSet);
  
  static int basisPolyOrder(BasisPtr basis);
  static int getBasisRank(BasisPtr basis);
  static EFunctionSpaceExtended getBasisFunctionSpace(BasisPtr basis);
  
  static bool basisKnown(BasisPtr basis);
  static bool isMultiBasis(BasisPtr basis);
  
  static void registerBasis( BasisPtr basis, int basisRank, int polyOrder, int cellTopoKey, EFunctionSpaceExtended fs );
};

#endif
