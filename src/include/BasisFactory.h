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

#include "CellTopology.h"

typedef Teuchos::RCP< Camellia::Basis<> > BasisPtr;

class BasisFactory {
  typedef Camellia::EFunctionSpace FSE;
private:
  map< pair< pair<int,int>, Camellia::EFunctionSpace >, BasisPtr >
              _existingBases; // keys are ((polyOrder,cellTopoKey),fs))
  map< pair< pair<int,int>, Camellia::EFunctionSpace >, BasisPtr >
              _conformingBases; // keys are ((polyOrder,cellTopoKey),fs))
  map< pair< pair< Camellia::Basis<>*, int>, Camellia::EFunctionSpace>, BasisPtr >
              _spaceTimeBases; // keys are (shards Topo basis, temporal degree, temporal function space)
  map< pair< pair< Camellia::Basis<>*, int>, Camellia::EFunctionSpace>, BasisPtr >
              _conformingSpaceTimeBases; // keys are (shards Topo basis, temporal degree, temporal function space)

  
  // the following maps let us remember what arguments were used to create a basis:
  // (this is useful to, say, create a basis again, but now with polyOrder+1)
  map< Camellia::Basis<>*, int > _polyOrders; // allows lookup of poly order used to create basis
//  static map< Camellia::Basis<>*, int > _ranks; // allows lookup of basis rank
  map< Camellia::Basis<>*, Camellia::EFunctionSpace > _functionSpaces; // allows lookup of function spaces
  map< Camellia::Basis<>*, int > _cellTopoKeys; // allows lookup of cellTopoKeys
  set< Camellia::Basis<>*> _multiBases;
  map< vector< Camellia::Basis<>* >, Camellia::MultiBasisPtr > _multiBasesMap;
  map< pair< Camellia::Basis<>*, vector<double> >, PatchBasisPtr > _patchBases;
  set< Camellia::Basis<>* > _patchBasisSet;
  
  bool _useEnrichedTraces; // i.e. p+1, not p (default is true: this is what we need to prove optimal convergence)
  bool _useLobattoForQuadHGRAD;
  bool _useLobattoForQuadHDIV;
  bool _useLobattoForLineHGRAD;
  bool _useLegendreForLineHVOL;
public:
  BasisFactory();
  
  // new getBasis: (handles 0 or 1 temporal dimensions; calls the other version)
  BasisPtr getBasis(int H1Order, CellTopoPtr cellTopo, FSE functionSpaceForSpatialTopology, int temporalH1Order = 2,
                    FSE functionSpaceForTemporalTopology = Camellia::FUNCTION_SPACE_HVOL);
  BasisPtr getBasis( int polyOrder, unsigned cellTopoKey, FSE fs);
//  static BasisPtr getBasis(int &basisRank, int polyOrder, unsigned cellTopoKey, Camellia::EFunctionSpace fs);
  BasisPtr getConformingBasis( int polyOrder, unsigned cellTopoKey, FSE fs );
  BasisPtr getConformingBasis( int polyOrder, CellTopoPtr cellTopo, FSE fs, int temporalPolyOrder = 1,
                              FSE functionSpaceForTemporalTopology = Camellia::FUNCTION_SPACE_HVOL);
  
  BasisPtr getNodalBasisForCellTopology(CellTopoPtr cellTopo);
  BasisPtr getNodalBasisForCellTopology(unsigned cellTopoKey);
  
  Camellia::MultiBasisPtr getMultiBasis(vector< BasisPtr > &bases);
  PatchBasisPtr getPatchBasis(BasisPtr parent, FieldContainer<double> &patchNodesInParentRefCell, unsigned cellTopoKey = shards::Line<2>::key);

  BasisPtr addToPolyOrder(BasisPtr basis, int pToAdd);
  BasisPtr setPolyOrder(BasisPtr basis, int polyOrderToSet);
  
  int basisPolyOrder(BasisPtr basis);
  int getBasisRank(BasisPtr basis);
  Camellia::EFunctionSpace getBasisFunctionSpace(BasisPtr basis);
  
  bool basisKnown(BasisPtr basis);
  bool isMultiBasis(BasisPtr basis);
  bool isPatchBasis(BasisPtr basis);
  
  void registerBasis( BasisPtr basis, int basisRank, int polyOrder, int cellTopoKey, FSE fs );
  
  void setUseEnrichedTraces( bool value );
  
  // the following convenience methods belong in Basis or perhaps a wrapper thereof
  set<int> sideFieldIndices( BasisPtr basis, bool includeSideSubcells = true); // includeSideSubcells: e.g. include vertices as part of quad sides
  
  void setUseLobattoForQuadHGrad(bool value);
  void setUseLobattoForQuadHDiv(bool value);
  
  static Teuchos::RCP<BasisFactory> basisFactory(); // shared, global BasisFactory
};

typedef Teuchos::RCP<BasisFactory> BasisFactoryPtr;

#endif
