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
#include "BasisFactory.h"

#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid_HDIV_QUAD_In_FEM.hpp"
#include "Intrepid_HCURL_QUAD_In_FEM.hpp"

#include "Intrepid_HGRAD_TRI_Cn_FEM.hpp"
#include "Intrepid_HDIV_TRI_In_FEM.hpp"
#include "Intrepid_HCURL_TRI_In_FEM.hpp"

#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"

#include "Basis_HGRAD_QUAD_C0_FEM.hpp"

#include "VectorizedBasis.h"

#include "LobattoHGRAD_Quad.h"

//define the static maps:
map< pair< pair<int,int>, IntrepidExtendedTypes::EFunctionSpaceExtended >, BasisPtr > BasisFactory::_conformingBases;
map< pair< pair<int,int>, IntrepidExtendedTypes::EFunctionSpaceExtended >, BasisPtr > BasisFactory::_existingBases;
map< Camellia::Basis<>*, int > BasisFactory::_polyOrders; // allows lookup of poly order used to create basis
map< Camellia::Basis<>*, IntrepidExtendedTypes::EFunctionSpaceExtended > BasisFactory::_functionSpaces; // allows lookup of function spaces
map< Camellia::Basis<>*, int > BasisFactory::_cellTopoKeys; // allows lookup of cellTopoKeys
set< Camellia::Basis<>* > BasisFactory::_multiBases;
map< vector< Camellia::Basis<>* >, MultiBasisPtr > BasisFactory::_multiBasesMap;
map< pair<Camellia::Basis<>*, vector<double> >, PatchBasisPtr > BasisFactory::_patchBases;
set< Camellia::Basis<>* > BasisFactory::_patchBasisSet;

bool BasisFactory::_useEnrichedTraces = true;

bool BasisFactory::_useLobattoForQuadH1 = false;

using namespace Camellia;

BasisPtr BasisFactory::getBasis( int polyOrder, unsigned cellTopoKey, IntrepidExtendedTypes::EFunctionSpaceExtended fs) {
  if (fs != IntrepidExtendedTypes::FUNCTION_SPACE_ONE) {
    TEUCHOS_TEST_FOR_EXCEPTION(polyOrder == 0, std::invalid_argument, "polyOrder = 0 unsupported");
  }
  
  BasisPtr basis;
  pair< pair<int,int>, IntrepidExtendedTypes::EFunctionSpaceExtended > key = make_pair( make_pair(polyOrder, cellTopoKey), fs );
  
  if ( _existingBases.find(key) != _existingBases.end() ) {
    basis = _existingBases[key];
    return basis;
  }
  
  int spaceDim;
  bool twoD = (cellTopoKey == shards::Quadrilateral<4>::key) || (cellTopoKey == shards::Triangle<3>::key);
  bool oneD = (cellTopoKey == shards::Line<2>::key);
  if (oneD) {
    spaceDim = 1;
  } else if (twoD) {
    spaceDim = 2;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
  }
  
  int scalarRank = 0, vectorRank = 1;
  
  if (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) {
    BasisPtr componentBasis = BasisFactory::getBasis(polyOrder, cellTopoKey, IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
    basis = Teuchos::rcp( new VectorizedBasis<>(componentBasis,spaceDim) ); // 3-21-13: changed behavior for 1D vectors, but I don't think we use these right now.
  } else if (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL) {
    BasisPtr componentBasis = BasisFactory::getBasis(polyOrder, cellTopoKey, IntrepidExtendedTypes::FUNCTION_SPACE_HVOL);
    basis = Teuchos::rcp( new VectorizedBasis<>(componentBasis,spaceDim) );
  } else { 
    switch (cellTopoKey) {
      case shards::Quadrilateral<4>::key:
        switch(fs) {
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD):
            //if (polyOrder==0) {
            //  basis = Teuchos::rcp( new Basis_HGRAD_QUAD_C0_FEM<double, Intrepid::FieldContainer<double> >() ) ;
            //} else {
            if (! _useLobattoForQuadH1) {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                      spaceDim, scalarRank) );
            } else {
              basis = Teuchos::rcp( new LobattoHGRAD_Quad<double, Intrepid::FieldContainer<double> >(polyOrder) );
            }
            //}
          break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HDIV):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HDIV_QUAD_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                  spaceDim, vectorRank)
                                 );
          break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HCURL):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HCURL_QUAD_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                              spaceDim, vectorRank)
                                                             );
          break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HVOL):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Intrepid::Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL)),
                                  spaceDim, scalarRank)
                                 );
          break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_ONE):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HGRAD_QUAD_C0_FEM<double, Intrepid::FieldContainer<double> >()),
                                                             spaceDim, scalarRank)
                                 ) ;
          break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION( ( (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) &&
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
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HGRAD_TRI_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, scalarRank)
                                 ) ;
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HDIV):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HDIV_TRI_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, vectorRank)
                                 );
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HCURL):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HCURL_TRI_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, vectorRank)
                                 );
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HVOL):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Intrepid::Basis_HGRAD_TRI_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_WARPBLEND)),
                                                             spaceDim, scalarRank)
                                 );
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION( ( (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) &&
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
            if (_useEnrichedTraces) {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                               spaceDim, scalarRank)
                                   );
            } else {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL)),
                                                               spaceDim, scalarRank)
                                   );
            }
          break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HVOL):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL)),
                                                             spaceDim, scalarRank)
                                 );
          break;
          default:        
            TEUCHOS_TEST_FOR_EXCEPTION( ( (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) &&
                                  (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) ),
                               std::invalid_argument,
                              "Unhandled function space for line_2. Please use HGRAD or HVOL.");
        }
      break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION( ( (cellTopoKey != shards::Quadrilateral<4>::key)        &&
                              (cellTopoKey != shards::Triangle<3>::key)             &&
                              (cellTopoKey != shards::Line<2>::key) ),
                               std::invalid_argument,
                              "Unknown cell topology for basis selction. Please use Line_2, Quadrilateral_4, or Triangle_3.");

     }
  }
  _existingBases[key] = basis;
  _polyOrders[basis.get()] = polyOrder;
  _functionSpaces[basis.get()] = fs;
  _cellTopoKeys[basis.get()] = cellTopoKey;
  return basis;
}

BasisPtr BasisFactory::getConformingBasis( int polyOrder, unsigned cellTopoKey, IntrepidExtendedTypes::EFunctionSpaceExtended fs ) {
  // this method is fairly redundant with getBasis(), but it provides the chance to offer different bases when a conforming basis is
  // required.
  
  if (fs != IntrepidExtendedTypes::FUNCTION_SPACE_ONE) {
    TEUCHOS_TEST_FOR_EXCEPTION(polyOrder == 0, std::invalid_argument, "polyOrder = 0 unsupported");
  }
  
  BasisPtr basis;
  pair< pair<int,int>, IntrepidExtendedTypes::EFunctionSpaceExtended > key = make_pair( make_pair(polyOrder, cellTopoKey), fs );
  
  // First, we call getBasis(), and if the one that it returns is conforming, we just use that.
  BasisPtr standardBasis = getBasis(polyOrder, cellTopoKey, fs);
  if (standardBasis->isConforming()) {
    _conformingBases[key] = standardBasis;
    return standardBasis;
  }
  
  if ( _conformingBases.find(key) != _conformingBases.end() ) {
    basis = _conformingBases[key];
    return basis;
  }
  
  int spaceDim;
  bool twoD = (cellTopoKey == shards::Quadrilateral<4>::key) || (cellTopoKey == shards::Triangle<3>::key);
  bool oneD = (cellTopoKey == shards::Line<2>::key);
  if (oneD) {
    spaceDim = 1;
  } else if (twoD) {
    spaceDim = 2;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
  }
  
  int scalarRank = 0, vectorRank = 1;
  
  if (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) {
    BasisPtr componentBasis = BasisFactory::getConformingBasis(polyOrder, cellTopoKey, IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
    basis = Teuchos::rcp( new VectorizedBasis<>(componentBasis,spaceDim) ); // 3-21-13: changed behavior for 1D vectors, but I don't think we use these right now.
  } else if (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL) {
    BasisPtr componentBasis = BasisFactory::getConformingBasis(polyOrder, cellTopoKey, IntrepidExtendedTypes::FUNCTION_SPACE_HVOL);
    basis = Teuchos::rcp( new VectorizedBasis<>(componentBasis,spaceDim) );
  } else {
    switch (cellTopoKey) {
      case shards::Quadrilateral<4>::key:
        switch(fs) {
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD):
            //if (polyOrder==0) {
            //  basis = Teuchos::rcp( new Basis_HGRAD_QUAD_C0_FEM<double, Intrepid::FieldContainer<double> >() ) ;
            //} else {
            if (! _useLobattoForQuadH1) {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                               spaceDim, scalarRank) );
            } else {
              basis = Teuchos::rcp( new LobattoHGRAD_Quad<double, Intrepid::FieldContainer<double> >(polyOrder) );
            }
            //}
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HDIV):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HDIV_QUAD_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                             spaceDim, vectorRank)
                                 );
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HCURL):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HCURL_QUAD_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                             spaceDim, vectorRank)
                                 );
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HVOL):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Intrepid::Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL)),
                                                             spaceDim, scalarRank)
                                 );
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_ONE):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HGRAD_QUAD_C0_FEM<double, Intrepid::FieldContainer<double> >()),
                                                             spaceDim, scalarRank)
                                 ) ;
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION( ( (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) &&
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
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HGRAD_TRI_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, scalarRank)
                                 ) ;
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HDIV):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HDIV_TRI_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, vectorRank)
                                 );
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HCURL):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HCURL_TRI_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, vectorRank)
                                 );
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HVOL):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Intrepid::Basis_HGRAD_TRI_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_WARPBLEND)),
                                                             spaceDim, scalarRank)
                                 );
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION( ( (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) &&
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
            if (_useEnrichedTraces) {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                               spaceDim, scalarRank)
                                   );
            } else {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL)),
                                                               spaceDim, scalarRank)
                                   );
            }
            break;
          case(IntrepidExtendedTypes::FUNCTION_SPACE_HVOL):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL)),
                                                             spaceDim, scalarRank)
                                 );
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION( ( (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) &&
                                         (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) ),
                                       std::invalid_argument,
                                       "Unhandled function space for line_2. Please use HGRAD or HVOL.");
        }
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION( ( (cellTopoKey != shards::Quadrilateral<4>::key)        &&
                                     (cellTopoKey != shards::Triangle<3>::key)             &&
                                     (cellTopoKey != shards::Line<2>::key) ),
                                   std::invalid_argument,
                                   "Unknown cell topology for basis selction. Please use Line_2, Quadrilateral_4, or Triangle_3.");
        
    }
  }
  _conformingBases[key] = basis;
  _polyOrders[basis.get()] = polyOrder;
  _functionSpaces[basis.get()] = fs;
  _cellTopoKeys[basis.get()] = cellTopoKey;
  return basis;
}

MultiBasisPtr BasisFactory::getMultiBasis(vector< BasisPtr > &bases) {
  vector< Camellia::Basis<>* > key;
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
  
  if (numBases != 2) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"BasisFactory only supports lines divided in two right now.");
  }
  
  for (int i=0; i<numBases; i++) {
    if (! basisKnown(bases[i]) ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisFactory can only make MultiBasis from registered Basis pointers.");
    }
    
    polyOrder = max(polyOrder,_polyOrders[bases[i].get()]);
    fs = _functionSpaces[bases[i].get()];
    cellTopoKey = _cellTopoKeys[bases[i].get()];
  }
  
  if ((_cellTopoKeys[bases[0].get()] != shards::Line<2>::key)
    || (_cellTopoKeys[bases[1].get()] != shards::Line<2>::key) )  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"BasisFactory only supports lines divided in two right now.");
  }
  int spaceDim=1, numNodesPerSubRefCell = 2; 
  FieldContainer<double> subRefNodes(numBases,numNodesPerSubRefCell,spaceDim);
  subRefNodes(0,0,0) = -1.0;
  subRefNodes(0,1,0) = 0.0;
  subRefNodes(1,0,0) = 0.0;
  subRefNodes(1,1,0) = 1.0;
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  MultiBasisPtr multiBasis = Teuchos::rcp( new MultiBasis<>(bases, subRefNodes, line_2) );
  _multiBasesMap[key] = multiBasis;
  _multiBases.insert(multiBasis.get());
  
  _polyOrders[multiBasis.get()] = polyOrder;
  _functionSpaces[multiBasis.get()] = fs;
  _cellTopoKeys[multiBasis.get()] = cellTopoKey;
  
  return multiBasis;
}

PatchBasisPtr BasisFactory::getPatchBasis(BasisPtr parent, FieldContainer<double> &patchNodesInParentRefCell, unsigned cellTopoKey) {
  TEUCHOS_TEST_FOR_EXCEPTION(cellTopoKey != shards::Line<2>::key, std::invalid_argument, "getPatchBasis only supports lines right now.");
  TEUCHOS_TEST_FOR_EXCEPTION(patchNodesInParentRefCell.dimension(0) != 2, std::invalid_argument, "should be just 2 points in patchNodes.");
  TEUCHOS_TEST_FOR_EXCEPTION(patchNodesInParentRefCell.dimension(1) != 1, std::invalid_argument, "patchNodes.dimension(1) != 1.");
  TEUCHOS_TEST_FOR_EXCEPTION(! basisKnown(parent), std::invalid_argument, "parentBasis not registered with BasisFactory.");
  vector<double> points;
  for (int i=0; i<patchNodesInParentRefCell.size(); i++) {
    points.push_back(patchNodesInParentRefCell[i]);
  }
  pair<Camellia::Basis<>*, vector<double> > key = make_pair( parent.get(), points );
  map< pair<Camellia::Basis<>*, vector<double> >, PatchBasisPtr >::iterator entry = _patchBases.find(key);
  if ( entry != _patchBases.end() ) {
    return entry->second;
  }
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  PatchBasisPtr patchBasis = Teuchos::rcp( new PatchBasis<>(parent, patchNodesInParentRefCell, line_2));
  
  _patchBasisSet.insert(patchBasis.get());
  _patchBases[key] = patchBasis;
  _polyOrders[patchBasis.get()] = _polyOrders[parent.get()];
  _functionSpaces[patchBasis.get()] = _functionSpaces[parent.get()];
  _cellTopoKeys[patchBasis.get()] = cellTopoKey;
  
  return patchBasis;
}


void BasisFactory::registerBasis( BasisPtr basis, int basisRank, int polyOrder, int cellTopoKey, IntrepidExtendedTypes::EFunctionSpaceExtended fs ) {
  pair< pair<int,int>, IntrepidExtendedTypes::EFunctionSpaceExtended > key = make_pair( make_pair(polyOrder, cellTopoKey), fs );
  if ( _existingBases.find(key) != _existingBases.end() ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "Can't register a basis for which there's already an entry...");
  }
  _existingBases[key] = basis;
  _polyOrders[basis.get()] = polyOrder;
  _functionSpaces[basis.get()] = fs;
  _cellTopoKeys[basis.get()] = cellTopoKey;
}

BasisPtr BasisFactory::addToPolyOrder(BasisPtr basis, int pToAdd) {
  int polyOrder = _polyOrders[basis.get()] + pToAdd;
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = _functionSpaces[basis.get()];
  int cellTopoKey = _cellTopoKeys[basis.get()];
  if (basis->isConforming()) {
    return getConformingBasis(polyOrder, cellTopoKey, fs);
  } else {
    return getBasis(polyOrder, cellTopoKey, fs);
  }
}

BasisPtr BasisFactory::setPolyOrder(BasisPtr basis, int pToSet) {
  if (isMultiBasis(basis)) {
    // for now anyway, we don't set poly order for MultiBasis
    // (the rule now is that MultiBasis is exactly the broken neighbor's "natural" bases)
    return basis;
//    // set each sub-basis to pToSet:
//    MultiBasis* mb = (MultiBasis*) basis.get();
//    int numSubBases = mb->numSubBases();
//    vector< BasisPtr > upgradedSubBases;
//    for (int basisIndex=0; basisIndex<numSubBases; basisIndex++) {
//      BasisPtr subBasis = mb->getSubBasis(basisIndex);
//      upgradedSubBases.push_back( BasisFactory::setPolyOrder(subBasis, pToSet) );
//    }
//    return getMultiBasis(upgradedSubBases);
  }
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = _functionSpaces[basis.get()];
  int cellTopoKey = _cellTopoKeys[basis.get()];
  if (basis->isConforming()) {
    return getConformingBasis(pToSet, cellTopoKey, fs);
  } else {
    return getBasis(pToSet, cellTopoKey, fs);
  }
}

int BasisFactory::getBasisRank(BasisPtr basis) {
  return basis->rangeRank();
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

void BasisFactory::setUseEnrichedTraces( bool value ) {
  _useEnrichedTraces = value;
}

set<int> BasisFactory::sideFieldIndices( BasisPtr basis, bool includeSideSubcells ) { // includeSideSubcells: e.g. include vertices as part of quad sides
  shards::CellTopology cellTopo = basis->domainTopology();
  int dim = cellTopo.getDimension();
  int sideDim = dim - 1;
  if (sideDim==2) {
    return basis->dofOrdinalsForFaces(includeSideSubcells);
  } else if (sideDim==1) {
    return basis->dofOrdinalsForEdges(includeSideSubcells);
  } else if (sideDim==0) {
    return basis->dofOrdinalsForVertices();
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "can't get side field indices for sideDim != 0, 1, or 2.");
  }
}
