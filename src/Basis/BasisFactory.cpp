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

#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid_HDIV_QUAD_In_FEM.hpp"
#include "Intrepid_HCURL_QUAD_In_FEM.hpp"

#include "Intrepid_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid_HDIV_TRI_In_FEM.hpp"
#include "Intrepid_HCURL_TRI_In_FEM.hpp"

#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"

#include "Basis_HGRAD_QUAD_C0_FEM.hpp"

#include "Vectorized_Basis.hpp"
#include "Basis_HGRAD_2D_Cn_FEM.hpp"
#include "BasisFactory.h"

typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
typedef Teuchos::RCP< MultiBasis > MultiBasisPtr;
typedef Teuchos::RCP<Vectorized_Basis<double, FieldContainer<double> > > VectorBasisPtr;
typedef Teuchos::RCP< PatchBasis > PatchBasisPtr;

//define the static maps:
map< pair< pair<int,int>, IntrepidExtendedTypes::EFunctionSpaceExtended >, BasisPtr > BasisFactory::_existingBasis;
map< Basis<double,FieldContainer<double> >*, int > BasisFactory::_polyOrders; // allows lookup of poly order used to create basis
map< Basis<double,FieldContainer<double> >*, int > BasisFactory::_ranks; // allows lookup of basis rank
map< Basis<double,FieldContainer<double> >*, IntrepidExtendedTypes::EFunctionSpaceExtended > BasisFactory::_functionSpaces; // allows lookup of function spaces
map< Basis<double,FieldContainer<double> >*, int > BasisFactory::_cellTopoKeys; // allows lookup of cellTopoKeys
set< Basis<double,FieldContainer<double> >* > BasisFactory::_multiBases;
map< vector< Basis<double,FieldContainer<double> >* >, MultiBasisPtr > BasisFactory::_multiBasesMap;
map< pair<Basis<double,FieldContainer<double> >*, vector<double> >, PatchBasisPtr > BasisFactory::_patchBases;
set< Basis<double,FieldContainer<double> >* > BasisFactory::_patchBasisSet;

BasisPtr BasisFactory::getBasis( int polyOrder, unsigned cellTopoKey, IntrepidExtendedTypes::EFunctionSpaceExtended fs) {
  int basisRank; // to discard
  return getBasis(basisRank,polyOrder,cellTopoKey,fs);
}

BasisPtr BasisFactory::getBasis(int &basisRank,
                                int polyOrder, unsigned cellTopoKey, IntrepidExtendedTypes::EFunctionSpaceExtended fs) {

  if (fs != IntrepidExtendedTypes::FUNCTION_SPACE_ONE) {
    TEST_FOR_EXCEPTION(polyOrder == 0, std::invalid_argument, "polyOrder = 0 unsupported");
  }
  
  BasisPtr basis;
  pair< pair<int,int>, IntrepidExtendedTypes::EFunctionSpaceExtended > key = make_pair( make_pair(polyOrder, cellTopoKey), fs );
  
  if ( _existingBasis.find(key) != _existingBasis.end() ) {
    basis = _existingBasis[key];
    basisRank = _ranks[basis.get()];
    return basis;
  }
  
  if (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) {
    //BasisPtr componentBasis = getBasis(basisRank, polyOrder, cellTopoKey, IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
    // TODO: support more than just 2 dimensions here
    basis = Teuchos::rcp(new Basis_HGRAD_2D_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,cellTopoKey));
    basisRank = 1;
  } else { 
  
    switch (cellTopoKey) {
      case shards::Quadrilateral<4>::key:
        switch(fs) {
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD):
            //if (polyOrder==0) {
            //  basis = Teuchos::rcp( new Basis_HGRAD_QUAD_C0_FEM<double, Intrepid::FieldContainer<double> >() ) ;
            //} else {
              basis = Teuchos::rcp( new Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL) ) ;
            //}
            basisRank = 0;
          break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HDIV):
            basis = Teuchos::rcp( new Basis_HDIV_QUAD_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL) );
            basisRank = 1;
          break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HCURL):
            basis = Teuchos::rcp( new Basis_HCURL_QUAD_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL) );
            basisRank = 1;
          break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HVOL):
            basis = Teuchos::rcp( new Intrepid::Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL) );
            basisRank = 0;
          break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_ONE):
            basis = Teuchos::rcp( new Basis_HGRAD_QUAD_C0_FEM<double, Intrepid::FieldContainer<double> >() ) ;
          break;
          default:
            TEST_FOR_EXCEPTION( ( (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) &&
                                  (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HDIV) &&
                                  (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HCURL) &&
                                  (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) ),
                               std::invalid_argument,
                              "Unhandled function space for quad_4. Please use HGRAD, HDIV, HCURL, or HVOL.");
        }
      break;
      case shards::Triangle<3>::key:
        switch(fs) {
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD):
            basis = Teuchos::rcp( new Basis_HGRAD_TRI_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND) ) ;
            basisRank = 0;
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HDIV):
            basis = Teuchos::rcp( new Basis_HDIV_TRI_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND) );
            basisRank = 1;
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HCURL):
            basis = Teuchos::rcp( new Basis_HCURL_TRI_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND) );
            basisRank = 1;
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HVOL):
            basis = Teuchos::rcp( new Intrepid::Basis_HGRAD_TRI_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_WARPBLEND) );
            basisRank = 0;
            break;
          default:
            TEST_FOR_EXCEPTION( ( (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) &&
                                 (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HDIV) &&
                                 (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HCURL) &&
                                 (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) ),
                               std::invalid_argument,
                               "Unhandled function space for tri_3. Please use HGRAD, HDIV, HCURL, or HVOL.");
        }
        break;
      case shards::Line<2>::key:
        switch(fs) {
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD):
            basis = Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL) );
            basisRank = 0;
          break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HVOL):
            basis = Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL) );
            basisRank = 0;
          break;
          default:        
            TEST_FOR_EXCEPTION( ( (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) &&
                                  (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) ),
                               std::invalid_argument,
                              "Unhandled function space for line_2. Please use HGRAD or HVOL.");
        }
      break;
      default:
        TEST_FOR_EXCEPTION( ( (cellTopoKey != shards::Quadrilateral<4>::key)        &&
                              (cellTopoKey != shards::Triangle<3>::key)             &&
                              (cellTopoKey != shards::Line<2>::key) ),
                               std::invalid_argument,
                              "Unknown cell topology for basis selction. Please use Line_2, Quadrilateral_4, or Triangle_3.");

     }
  }
  _existingBasis[key] = basis;
  _polyOrders[basis.get()] = polyOrder;
  _functionSpaces[basis.get()] = fs;
  _cellTopoKeys[basis.get()] = cellTopoKey;
  _ranks[basis.get()] = basisRank;
  return basis;
}

MultiBasisPtr BasisFactory::getMultiBasis(vector< BasisPtr > &bases) {
  vector< Basis<double,FieldContainer<double> >* > key;
  int numBases = bases.size();
  for (int i=0; i<numBases; i++) {
    key.push_back(bases[i].get());
  }
  if (_multiBasesMap.find(key) != _multiBasesMap.end() ) {
    return _multiBasesMap[key];
  }
  
  int polyOrder = 0;
  IntrepidExtendedTypes::EFunctionSpaceExtended fs;
  unsigned cellTopoKey;
  int basisRank;
  
  if (numBases != 2) {
    TEST_FOR_EXCEPTION(true,std::invalid_argument,"BasisFactory only supports lines divided in two right now.");
  }
  
  for (int i=0; i<numBases; i++) {
    if (! basisKnown(bases[i]) ) {
      TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisFactory can only make MultiBasis from registered Basis pointers.");
    }
    
    polyOrder = max(polyOrder,_polyOrders[bases[i].get()]);
    fs = _functionSpaces[bases[i].get()];
    cellTopoKey = _cellTopoKeys[bases[i].get()];
    basisRank = _ranks[bases[i].get()];
  }
  
  if ((_cellTopoKeys[bases[0].get()] != shards::Line<2>::key)
    || (_cellTopoKeys[bases[1].get()] != shards::Line<2>::key) )  {
    TEST_FOR_EXCEPTION(true,std::invalid_argument,"BasisFactory only supports lines divided in two right now.");
  }
  int spaceDim=1, numNodesPerSubRefCell = 2; 
  FieldContainer<double> subRefNodes(numBases,numNodesPerSubRefCell,spaceDim);
  subRefNodes(0,0,0) = -1.0;
  subRefNodes(0,1,0) = 0.0;
  subRefNodes(1,0,0) = 0.0;
  subRefNodes(1,1,0) = 1.0;
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  MultiBasisPtr multiBasis = Teuchos::rcp( new MultiBasis(bases, subRefNodes, line_2) );
  _multiBasesMap[key] = multiBasis;
  _multiBases.insert(multiBasis.get());
  
  _polyOrders[multiBasis.get()] = polyOrder;
  _functionSpaces[multiBasis.get()] = fs;
  _cellTopoKeys[multiBasis.get()] = cellTopoKey;
  _ranks[multiBasis.get()] = basisRank;
  
  return multiBasis;
}

PatchBasisPtr BasisFactory::getPatchBasis(BasisPtr parent, FieldContainer<double> &patchNodesInParentRefCell, unsigned cellTopoKey) {
  TEST_FOR_EXCEPTION(cellTopoKey != shards::Line<2>::key, std::invalid_argument, "getPatchBasis only supports lines right now.");
  TEST_FOR_EXCEPTION(patchNodesInParentRefCell.dimension(0) != 2, std::invalid_argument, "should be just 2 points in patchNodes.");
  TEST_FOR_EXCEPTION(patchNodesInParentRefCell.dimension(1) != 1, std::invalid_argument, "patchNodes.dimension(1) != 1.");
  TEST_FOR_EXCEPTION(! basisKnown(parent), std::invalid_argument, "parentBasis not registered with BasisFactory.");
  vector<double> points;
  for (int i=0; i<patchNodesInParentRefCell.size(); i++) {
    points.push_back(patchNodesInParentRefCell[i]);
  }
  pair<Basis<double,FieldContainer<double> >*, vector<double> > key = make_pair( parent.get(), points );
  map< pair<Basis<double,FieldContainer<double> >*, vector<double> >, PatchBasisPtr >::iterator entry = _patchBases.find(key);
  if ( entry != _patchBases.end() ) {
    return entry->second;
  }
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  PatchBasisPtr patchBasis = Teuchos::rcp( new PatchBasis(parent, patchNodesInParentRefCell, line_2));
  
  _patchBasisSet.insert(patchBasis.get());
  _patchBases[key] = patchBasis;
  _polyOrders[patchBasis.get()] = _polyOrders[parent.get()];
  _functionSpaces[patchBasis.get()] = _functionSpaces[parent.get()];
  _cellTopoKeys[patchBasis.get()] = cellTopoKey;
  _ranks[patchBasis.get()] = _ranks[parent.get()];
  
  return patchBasis;
}


void BasisFactory::registerBasis( BasisPtr basis, int basisRank, int polyOrder, int cellTopoKey, IntrepidExtendedTypes::EFunctionSpaceExtended fs ) {
  pair< pair<int,int>, IntrepidExtendedTypes::EFunctionSpaceExtended > key = make_pair( make_pair(polyOrder, cellTopoKey), fs );
  if ( _existingBasis.find(key) != _existingBasis.end() ) {
    TEST_FOR_EXCEPTION(true,std::invalid_argument, "Can't register a basis for which there's already an entry...");
  }
  _existingBasis[key] = basis;
  _polyOrders[basis.get()] = polyOrder;
  _functionSpaces[basis.get()] = fs;
  _cellTopoKeys[basis.get()] = cellTopoKey;
  _ranks[basis.get()] = basisRank;
}

BasisPtr BasisFactory::addToPolyOrder(BasisPtr basis, int pToAdd) {
  int polyOrder = _polyOrders[basis.get()] + pToAdd;
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = _functionSpaces[basis.get()];
  int cellTopoKey = _cellTopoKeys[basis.get()];
  int rank;
  return getBasis(rank, polyOrder, cellTopoKey, fs);
}

BasisPtr BasisFactory::setPolyOrder(BasisPtr basis, int pToSet) {
  if (isMultiBasis(basis)) {
    // set each sub-basis to pToSet:
    MultiBasis* mb = (MultiBasis*) basis.get();
    int numSubBases = mb->numSubBases();
    vector< BasisPtr > upgradedSubBases;
    for (int basisIndex=0; basisIndex<numSubBases; basisIndex++) {
      BasisPtr subBasis = mb->getSubBasis(basisIndex);
      upgradedSubBases.push_back( BasisFactory::setPolyOrder(subBasis, pToSet) );
    }
    return getMultiBasis(upgradedSubBases);
  }
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = _functionSpaces[basis.get()];
  int cellTopoKey = _cellTopoKeys[basis.get()];
  int rank;
  return getBasis(rank, pToSet, cellTopoKey, fs);
}

int BasisFactory::getBasisRank(BasisPtr basis) {
  return _ranks[basis.get()];
}

EFunctionSpaceExtended BasisFactory::getBasisFunctionSpace(BasisPtr basis) {
  return _functionSpaces[basis.get()];
}

int BasisFactory::basisPolyOrder(BasisPtr basis) {
  return _polyOrders[basis.get()];
}

bool BasisFactory::basisKnown(BasisPtr basis) {
  // look it up in one of our universal maps...
  return _polyOrders.find(basis.get()) != _polyOrders.end();
}

bool BasisFactory::isMultiBasis(BasisPtr basis) {
  return _multiBases.find(basis.get()) != _multiBases.end();
}

bool BasisFactory::isPatchBasis(BasisPtr basis) {
  return _patchBasisSet.find(basis.get()) != _patchBasisSet.end();
}