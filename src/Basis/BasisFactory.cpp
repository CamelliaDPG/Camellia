// @HEADER
//
// Original version copyright © 2011 Sandia Corporation. All Rights Reserved.
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

#include "Intrepid_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid_HDIV_HEX_In_FEM.hpp"
#include "Intrepid_HCURL_HEX_In_FEM.hpp"

#include "Intrepid_HGRAD_TET_Cn_FEM.hpp"
#include "Intrepid_HDIV_TET_In_FEM.hpp"
#include "Intrepid_HCURL_TET_In_FEM.hpp"

#include "Basis_HVOL_QUAD_C0_FEM.hpp"

#include "VectorizedBasis.h"

#include "LobattoHGRAD_QuadBasis.h"
#include "LobattoHDIV_QuadBasis.h"
#include "LobattoHDIV_QuadBasis_separable.h"
#include "LegendreHVOL_LineBasis.h"
#include "LobattoHGRAD_LineBasis.h"

#include "PointBasis.h"

#include "TensorBasis.h"

BasisFactory::BasisFactory() {
  _useEnrichedTraces = true;
  _useLobattoForQuadHGRAD = false;
  _useLobattoForQuadHDIV = false;
  _useLobattoForLineHGRAD = false;
  _useLegendreForLineHVOL = false;
}

using namespace Camellia;

BasisPtr BasisFactory::getBasis(std::vector<int> &H1Order, CellTopoPtr cellTopo, Camellia::EFunctionSpace functionSpaceForSpatialTopology,
                                Camellia::EFunctionSpace functionSpaceForTemporalTopology) {
  if (H1Order.size() == 0) {
    cout << "H1Order cannot be empty.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "H1Order cannot be empty.");
  }
  if (H1Order.size() > 2) {
    cout << "BasisFactory does not yet support tensor product bases of tensorial degree > 1.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisFactory does not yet support tensor product bases of tensorial degree > 1.");
  }
  if (H1Order.size() == 1) {
    return getBasis(H1Order[0], cellTopo, functionSpaceForSpatialTopology);
  } else {
    return getBasis(H1Order[0], cellTopo, functionSpaceForSpatialTopology, H1Order[1], functionSpaceForTemporalTopology);
  }
}

BasisPtr BasisFactory::getBasis(int H1Order, CellTopoPtr cellTopo, Camellia::EFunctionSpace functionSpaceForSpatialTopology,
                                int temporalH1Order, Camellia::EFunctionSpace functionSpaceForTemporalTopology) {
  if (cellTopo->getTensorialDegree() > 1) {
    cout << "BasisFactory::getBasis() only handles 0 or 1 tensorial degree elements.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisFactory::getBasis() only handles 0 or 1 tensorial degree elements.");
  }
  
  BasisPtr basisForShardsTopo = getBasis(H1Order, cellTopo->getShardsTopology().getKey(), functionSpaceForSpatialTopology);
  
  if (cellTopo->getTensorialDegree() == 0) return basisForShardsTopo;
  
  // if we get here, have tensorial degree exactly 1.
  
  unsigned lineKey = shards::Line<2>::key;
  pair< pair<Camellia::Basis<>*, int>, Camellia::EFunctionSpace> key = make_pair( make_pair(basisForShardsTopo.get(), temporalH1Order), functionSpaceForTemporalTopology );
  
  if (_spaceTimeBases.find(key) != _spaceTimeBases.end()) return _spaceTimeBases[key];
  
  BasisPtr temporalBasis = getBasis(temporalH1Order, lineKey, functionSpaceForTemporalTopology);
  
  typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;
  Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(basisForShardsTopo, temporalBasis) );

  _spaceTimeBases[key] = tensorBasis;

  return tensorBasis;
}

BasisPtr BasisFactory::getBasis( int polyOrder, unsigned cellTopoKey, Camellia::EFunctionSpace fs) {
  if (fs != Camellia::FUNCTION_SPACE_REAL_SCALAR) {
    TEUCHOS_TEST_FOR_EXCEPTION(polyOrder == 0, std::invalid_argument, "polyOrder = 0 unsupported");
  }
  
  BasisPtr basis;
  pair< pair<int,int>, Camellia::EFunctionSpace > key = make_pair( make_pair(polyOrder, cellTopoKey), fs );
  
  if ( _existingBases.find(key) != _existingBases.end() ) {
    basis = _existingBases[key];
    return basis;
  }
  
  int spaceDim;
  bool threeD = (cellTopoKey == shards::Hexahedron<8>::key) || (cellTopoKey == shards::Tetrahedron<4>::key);
  bool twoD = (cellTopoKey == shards::Quadrilateral<4>::key) || (cellTopoKey == shards::Triangle<3>::key);
  bool oneD = (cellTopoKey == shards::Line<2>::key);
  bool zeroD = (cellTopoKey == shards::Node::key);
  if (zeroD) {
    spaceDim = 0;
  } else if (oneD) {
    spaceDim = 1;
  } else if (twoD) {
    spaceDim = 2;
  } else if (threeD) {
    spaceDim = 3;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
  }
  
  int scalarRank = 0, vectorRank = 1;
  
  if ((fs == Camellia::FUNCTION_SPACE_VECTOR_HGRAD) || (fs == Camellia::FUNCTION_SPACE_VECTOR_HGRAD_DISC)) {
    BasisPtr componentBasis = BasisFactory::getBasis(polyOrder, cellTopoKey, Camellia::FUNCTION_SPACE_HGRAD);
    basis = Teuchos::rcp( new VectorizedBasis<>(componentBasis,spaceDim) ); // 3-21-13: changed behavior for 1D vectors, but I don't think we use these right now.
  } else if (fs == Camellia::FUNCTION_SPACE_VECTOR_HVOL) {
    BasisPtr componentBasis = BasisFactory::getBasis(polyOrder, cellTopoKey, Camellia::FUNCTION_SPACE_HVOL);
    basis = Teuchos::rcp( new VectorizedBasis<>(componentBasis,spaceDim) );
  } else { 
    switch (cellTopoKey) {
      case shards::Node::key: // point topology
        switch (fs) {
          case Camellia::FUNCTION_SPACE_HGRAD:
          case Camellia::FUNCTION_SPACE_HGRAD_DISC:
          case Camellia::FUNCTION_SPACE_HVOL:
                    basis = Teuchos::rcp( new PointBasis<>() );
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported function space for point topology.");
        }

        break;
      case shards::Hexahedron<8>::key:
        switch(fs) {
          case Camellia::FUNCTION_SPACE_HGRAD:
          case Camellia::FUNCTION_SPACE_HGRAD_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_HEX_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                             spaceDim, scalarRank, fs) );
            break;
          case Camellia::FUNCTION_SPACE_HDIV:
          case Camellia::FUNCTION_SPACE_HDIV_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HDIV_HEX_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                             spaceDim, vectorRank, fs) );
            break;
          case Camellia::FUNCTION_SPACE_HCURL:
          case Camellia::FUNCTION_SPACE_HCURL_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HCURL_HEX_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                             spaceDim, vectorRank, fs) );
            break;
          case Camellia::FUNCTION_SPACE_HVOL:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_HEX_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL)),
                                                             spaceDim, scalarRank, fs) );
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported function space for topology Hexahedron<8>");
        }
        break;
      case shards::Tetrahedron<4>::key:
        switch(fs) {
          case Camellia::FUNCTION_SPACE_HGRAD:
          case Camellia::FUNCTION_SPACE_HGRAD_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_TET_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, scalarRank, fs) );
            break;
          case Camellia::FUNCTION_SPACE_HDIV:
          case Camellia::FUNCTION_SPACE_HDIV_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HDIV_TET_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, vectorRank, fs) );
            break;
          case Camellia::FUNCTION_SPACE_HCURL:
          case Camellia::FUNCTION_SPACE_HCURL_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HCURL_TET_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, vectorRank, fs) );
            break;
          case Camellia::FUNCTION_SPACE_HVOL:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_TET_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_WARPBLEND)),
                                                             spaceDim, scalarRank, fs) );
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported function space for topology Tetrahedron<4>");
        }
        break;
      case shards::Quadrilateral<4>::key:
        switch(fs) {
          case Camellia::FUNCTION_SPACE_HGRAD:
          case Camellia::FUNCTION_SPACE_HGRAD_DISC:
            //if (polyOrder==0) {
            //  basis = Teuchos::rcp( new Basis_HVOL_QUAD_C0_FEM<double, Intrepid::FieldContainer<double> >() ) ;
            //} else {
            if (! _useLobattoForQuadHGRAD) {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                      spaceDim, scalarRank, fs) );
            } else {
              bool conformingFalse = false;
              basis = Teuchos::rcp( new LobattoHGRAD_QuadBasis<double, Intrepid::FieldContainer<double> >(polyOrder,conformingFalse) );
            }
            //}
            break;
          case Camellia::FUNCTION_SPACE_HDIV:
          case Camellia::FUNCTION_SPACE_HDIV_DISC:
            if (! _useLobattoForQuadHDIV ) {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HDIV_QUAD_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                               spaceDim, vectorRank, fs)
                                   );
            } else {
              bool conformingFalse = false;
              basis = Teuchos::rcp( new LobattoHDIV_QuadBasis<double, Intrepid::FieldContainer<double> >(polyOrder,conformingFalse) );
            }
            break;
          case Camellia::FUNCTION_SPACE_HDIV_FREE:
          {
            bool conformingFalse = false;
            bool divFreeTrue = true;
            basis = Teuchos::rcp( new LobattoHDIV_QuadBasis_separable<double, Intrepid::FieldContainer<double> >(polyOrder,conformingFalse,divFreeTrue) );
          }
            break;
          case Camellia::FUNCTION_SPACE_HCURL:
          case Camellia::FUNCTION_SPACE_HCURL_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HCURL_QUAD_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                              spaceDim, vectorRank, fs)
                                                             );
          break;
          case(Camellia::FUNCTION_SPACE_HVOL):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Intrepid::Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL)),
                                  spaceDim, scalarRank, fs)
                                 );
          break;
          case(Camellia::FUNCTION_SPACE_REAL_SCALAR):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HVOL_QUAD_C0_FEM<double, Intrepid::FieldContainer<double> >()),
                                                             spaceDim, scalarRank, fs)
                                 ) ;
          break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION( ( (fs != Camellia::FUNCTION_SPACE_HGRAD) &&
                                  (fs != Camellia::FUNCTION_SPACE_HDIV) &&
                                  (fs != Camellia::FUNCTION_SPACE_HCURL) &&
                                  (fs != Camellia::FUNCTION_SPACE_HVOL) ),
                               std::invalid_argument,
                              "Unhandled function space for quad_4. Please use HGRAD, HDIV, HCURL, or HVOL.");
        }
      break;
      case shards::Triangle<3>::key:
        switch(fs) {
          case Camellia::FUNCTION_SPACE_HGRAD:
          case Camellia::FUNCTION_SPACE_HGRAD_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HGRAD_TRI_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, scalarRank, fs)
                                 ) ;
            break;
          case Camellia::FUNCTION_SPACE_HDIV:
          case Camellia::FUNCTION_SPACE_HDIV_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HDIV_TRI_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, vectorRank, fs)
                                 );
            break;
          case Camellia::FUNCTION_SPACE_HCURL:
          case Camellia::FUNCTION_SPACE_HCURL_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HCURL_TRI_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, vectorRank, fs)
                                 );
            break;
          case Camellia::FUNCTION_SPACE_HVOL:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Intrepid::Basis_HGRAD_TRI_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_WARPBLEND)),
                                                             spaceDim, scalarRank, fs)
                                 );
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION( ( (fs != Camellia::FUNCTION_SPACE_HGRAD) &&
                                 (fs != Camellia::FUNCTION_SPACE_HDIV) &&
                                 (fs != Camellia::FUNCTION_SPACE_HCURL) &&
                                 (fs != Camellia::FUNCTION_SPACE_HVOL) ),
                               std::invalid_argument,
                               "Unhandled function space for tri_3. Please use HGRAD, HDIV, HCURL, or HVOL.");
        }
        break;
      case shards::Line<2>::key:
        switch(fs) {
          case Camellia::FUNCTION_SPACE_HGRAD:
          case Camellia::FUNCTION_SPACE_HGRAD_DISC:
          {
            int basisPolyOrder = _useEnrichedTraces ? polyOrder : polyOrder - 1;
            
            if (_useLobattoForLineHGRAD) {
              // we use Legendre for HGRAD and HVOL both on the line--since we don't actually 
              // take derivatives of traces, this makes sense.
              // but I do have some concern that there may be logic errors to do with the basis's functionSpace()
              // return value: the Intrepid guys will always say HGRAD, and the Legendre HVOL.  Need to look at the
              // way this gets used to see if the conflation will make a difference in e.g. p-refinements.
              bool conformingFalse = false;
              basis = Teuchos::rcp( new LobattoHGRAD_LineBasis<>(basisPolyOrder, false) );
            } else {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(basisPolyOrder,POINTTYPE_SPECTRAL)),
                                                               spaceDim, scalarRank, fs)
                                   );
            }
          }
          break;
          case Camellia::FUNCTION_SPACE_HVOL:
            if (_useLegendreForLineHVOL) {
              basis = Teuchos::rcp( new LegendreHVOL_LineBasis<>(polyOrder-1) );
            } else {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL)),
                                                               spaceDim, scalarRank, fs)
                                   );
            }
          break;
          default:        
            TEUCHOS_TEST_FOR_EXCEPTION( ( (fs != Camellia::FUNCTION_SPACE_HGRAD) &&
                                  (fs != Camellia::FUNCTION_SPACE_HVOL) ),
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

BasisPtr BasisFactory::getConformingBasis(std::vector<int> &H1Order, CellTopoPtr cellTopo, Camellia::EFunctionSpace functionSpaceForSpatialTopology,
                                          Camellia::EFunctionSpace functionSpaceForTemporalTopology) {
  if (H1Order.size() == 0) {
    cout << "H1Order cannot be empty.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "H1Order cannot be empty.");
  }
  if (H1Order.size() > 2) {
    cout << "BasisFactory does not yet support tensor product bases of tensorial degree > 1.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisFactory does not yet support tensor product bases of tensorial degree > 1.");
  }
  if (H1Order.size() == 1) {
    return getConformingBasis(H1Order[0], cellTopo, functionSpaceForSpatialTopology);
  } else {
    return getConformingBasis(H1Order[0], cellTopo, functionSpaceForSpatialTopology, H1Order[1], functionSpaceForTemporalTopology);
  }
}

BasisPtr BasisFactory::getConformingBasis( int polyOrder, CellTopoPtr cellTopo, Camellia::EFunctionSpace fs,
                                          int temporalPolyOrder, FSE functionSpaceForTemporalTopology) {
  // this method is fairly redundant with getBasis(), but it provides the chance to offer different bases when a conforming basis is
  // required.
  
  BasisPtr basisForShardsTopo = getConformingBasis(polyOrder, cellTopo->getShardsTopology().getKey(), fs);
  
  if (cellTopo->getTensorialDegree() == 0) return basisForShardsTopo;
  
  // if we get here, have tensorial degree exactly 1.
  
  unsigned lineKey = shards::Line<2>::key;
  pair< pair<Camellia::Basis<>*, int>, Camellia::EFunctionSpace> key = make_pair( make_pair(basisForShardsTopo.get(), temporalPolyOrder), functionSpaceForTemporalTopology );
  
  if (_conformingSpaceTimeBases.find(key) != _conformingSpaceTimeBases.end()) return _conformingSpaceTimeBases[key];
  
  BasisPtr temporalBasis = getConformingBasis(temporalPolyOrder + 1, lineKey, functionSpaceForTemporalTopology);
  
  typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;
  Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(basisForShardsTopo, temporalBasis) );
  
  _conformingSpaceTimeBases[key] = tensorBasis;
  
  return tensorBasis;
}

BasisPtr BasisFactory::getConformingBasis( int polyOrder, unsigned cellTopoKey, Camellia::EFunctionSpace fs ) {
  // this method is fairly redundant with getBasis(), but it provides the chance to offer different bases when a conforming basis is
  // required.
  
  if (fs != Camellia::FUNCTION_SPACE_REAL_SCALAR) {
    TEUCHOS_TEST_FOR_EXCEPTION(polyOrder == 0, std::invalid_argument, "polyOrder = 0 unsupported");
  }
  
  BasisPtr basis;
  pair< pair<int,int>, Camellia::EFunctionSpace > key = make_pair( make_pair(polyOrder, cellTopoKey), fs );
  
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
  bool zeroD = (cellTopoKey == shards::Node::key);

  if (zeroD) {
    spaceDim = 0;
  } else if (oneD) {
    spaceDim = 1;
  } else if (twoD) {
    spaceDim = 2;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
  }
  
  int scalarRank = 0, vectorRank = 1;
  
  if (fs == Camellia::FUNCTION_SPACE_VECTOR_HGRAD) {
    BasisPtr componentBasis = BasisFactory::getConformingBasis(polyOrder, cellTopoKey, Camellia::FUNCTION_SPACE_HGRAD);
    basis = Teuchos::rcp( new VectorizedBasis<>(componentBasis,spaceDim) ); // 3-21-13: changed behavior for 1D vectors, but I don't think we use these right now.
  } else if (fs == Camellia::FUNCTION_SPACE_VECTOR_HVOL) {
    BasisPtr componentBasis = BasisFactory::getConformingBasis(polyOrder, cellTopoKey, Camellia::FUNCTION_SPACE_HVOL);
    basis = Teuchos::rcp( new VectorizedBasis<>(componentBasis,spaceDim) );
  } else {
    switch (cellTopoKey) {
      case shards::Quadrilateral<4>::key:
        switch(fs) {
          case Camellia::FUNCTION_SPACE_HGRAD:
          case Camellia::FUNCTION_SPACE_HGRAD_DISC:
            //if (polyOrder==0) {
            //  basis = Teuchos::rcp( new Basis_HVOL_QUAD_C0_FEM<double, Intrepid::FieldContainer<double> >() ) ;
            //} else {
            if (! _useLobattoForQuadHGRAD) {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                               spaceDim, scalarRank, fs) );
            } else {
              bool conformingTrue = true;
              basis = Teuchos::rcp( new LobattoHGRAD_QuadBasis<double, Intrepid::FieldContainer<double> >(polyOrder,conformingTrue) );
            }
            //}
            break;
          case Camellia::FUNCTION_SPACE_HDIV:
          case Camellia::FUNCTION_SPACE_HDIV_DISC:
            if (! _useLobattoForQuadHDIV) {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HDIV_QUAD_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                             spaceDim, vectorRank, fs)
                                 );
            } else {
              bool conformingTrue = true;
              basis = Teuchos::rcp( new LobattoHDIV_QuadBasis<double, Intrepid::FieldContainer<double> >(polyOrder,conformingTrue) );
            }
            break;
          case Camellia::FUNCTION_SPACE_HDIV_FREE:
          {
            bool conformingTrue = true;
            bool divFreeTrue = true;
            basis = Teuchos::rcp( new LobattoHDIV_QuadBasis<double, Intrepid::FieldContainer<double> >(polyOrder,conformingTrue,divFreeTrue) );
          }
            break;
          case Camellia::FUNCTION_SPACE_HCURL:
          case Camellia::FUNCTION_SPACE_HCURL_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HCURL_QUAD_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_SPECTRAL)),
                                                             spaceDim, vectorRank, fs)
                                 );
            break;
          case Camellia::FUNCTION_SPACE_HVOL:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Intrepid::Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL)),
                                                             spaceDim, scalarRank, fs)
                                 );
            break;
          case Camellia::FUNCTION_SPACE_REAL_SCALAR:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HVOL_QUAD_C0_FEM<double, Intrepid::FieldContainer<double> >()),
                                                             spaceDim, scalarRank, fs)
                                 ) ;
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION( ( (fs != Camellia::FUNCTION_SPACE_HGRAD) &&
                                         (fs != Camellia::FUNCTION_SPACE_HDIV) &&
                                         (fs != Camellia::FUNCTION_SPACE_HCURL) &&
                                         (fs != Camellia::FUNCTION_SPACE_HVOL) ),
                                       std::invalid_argument,
                                       "Unhandled function space for quad_4. Please use HGRAD, HDIV, HCURL, or HVOL.");
        }
        break;
      case shards::Triangle<3>::key:
        switch(fs) {
          case Camellia::FUNCTION_SPACE_HGRAD:
          case Camellia::FUNCTION_SPACE_HGRAD_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Basis_HGRAD_TRI_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, scalarRank, fs)
                                 ) ;
            break;
          case Camellia::FUNCTION_SPACE_HDIV:
          case Camellia::FUNCTION_SPACE_HDIV_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HDIV_TRI_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, vectorRank, fs)
                                 );
            break;
          case Camellia::FUNCTION_SPACE_HCURL:
          case Camellia::FUNCTION_SPACE_HCURL_DISC:
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HCURL_TRI_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,POINTTYPE_WARPBLEND)),
                                                             spaceDim, vectorRank, fs)
                                 );
            break;
          case(Camellia::FUNCTION_SPACE_HVOL):
            basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Intrepid::Basis_HGRAD_TRI_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_WARPBLEND)),
                                                             spaceDim, scalarRank, fs)
                                 );
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION( ( (fs != Camellia::FUNCTION_SPACE_HGRAD) &&
                                         (fs != Camellia::FUNCTION_SPACE_HDIV) &&
                                         (fs != Camellia::FUNCTION_SPACE_HCURL) &&
                                         (fs != Camellia::FUNCTION_SPACE_HVOL) ),
                                       std::invalid_argument,
                                       "Unhandled function space for tri_3. Please use HGRAD, HDIV, HCURL, or HVOL.");
        }
        break;
      case shards::Line<2>::key:
        switch(fs) {
          case Camellia::FUNCTION_SPACE_HGRAD:
          case Camellia::FUNCTION_SPACE_HGRAD_DISC:
          {
            int basisPolyOrder = _useEnrichedTraces ? polyOrder : polyOrder - 1;
            
            if (_useLobattoForLineHGRAD) {
              // we use Legendre for HGRAD and HVOL both on the line--since we don't actually
              // take derivatives of traces, this makes sense.
              // but I do have some concern that there may be logic errors to do with the basis's functionSpace()
              // return value: the Intrepid guys will always say HGRAD, and the Legendre HVOL.  Need to look at the
              // way this gets used to see if the conflation will make a difference in e.g. p-refinements.
              bool conformingTrue = true;
              basis = Teuchos::rcp( new LobattoHGRAD_LineBasis<>(basisPolyOrder, conformingTrue) );
            } else {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(basisPolyOrder,POINTTYPE_SPECTRAL)),
                                                               spaceDim, scalarRank, fs)
                                   );
            }
          }
            break;
          case Camellia::FUNCTION_SPACE_HVOL:
            if (_useLegendreForLineHVOL) { // not actually conforming, except in the L^2 sense...
              basis = Teuchos::rcp( new LegendreHVOL_LineBasis<>(polyOrder-1) );
            } else {
              basis = Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder-1,POINTTYPE_SPECTRAL)),
                                                               spaceDim, scalarRank, fs)
                                   );
            }
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION( ( (fs != Camellia::FUNCTION_SPACE_HGRAD) &&
                                         (fs != Camellia::FUNCTION_SPACE_HVOL) ),
                                       std::invalid_argument,
                                       "Unhandled function space for line_2. Please use HGRAD or HVOL.");
        }
        break;
      case shards::Node::key: // point topology
        switch (fs) {
          case Camellia::FUNCTION_SPACE_HGRAD:
          case Camellia::FUNCTION_SPACE_HGRAD_DISC:
          case Camellia::FUNCTION_SPACE_HVOL:
            basis = Teuchos::rcp( new PointBasis<>() );
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported function space for point topology.");
        }
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION( ( (cellTopoKey != shards::Quadrilateral<4>::key)        &&
                                      (cellTopoKey != shards::Triangle<3>::key)             &&
                                      (cellTopoKey != shards::Line<2>::key)                 &&
                                      (cellTopoKey != shards::Node::key)),
                                    std::invalid_argument,
                                    "Unknown cell topology for basis selction. Please use Node, Line_2, Quadrilateral_4, or Triangle_3.");
        
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
  Camellia::EFunctionSpace fs;
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

BasisPtr BasisFactory::getNodalBasisForCellTopology(CellTopoPtr cellTopo) {
  BasisPtr shardsNodalBasis = BasisFactory::basisFactory()->getNodalBasisForCellTopology(cellTopo->getShardsTopology().getKey());
  
  if (cellTopo->getTensorialDegree() == 0) return shardsNodalBasis;
  
  BasisPtr lineNodalBasis = BasisFactory::basisFactory()->getNodalBasisForCellTopology(shards::Line<2>::key);
  
  typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;
  
  BasisPtr nodalBasis = shardsNodalBasis;
  for (int i=0; i<cellTopo->getTensorialDegree(); i++) {
    nodalBasis = Teuchos::rcp( new TensorBasis(nodalBasis, lineNodalBasis) );
  }
  return nodalBasis;
}

BasisPtr BasisFactory::getNodalBasisForCellTopology(unsigned int cellTopoKey) {
  // used by CamelliaCellTools for computing Jacobians, etc.
  static const int ONE_D = 1, TWO_D = 2, THREE_D = 3;
  static const int SCALAR_RANK = 0;
  switch( cellTopoKey ){
      // Standard Base topologies (number of cellWorkset = number of vertices)
    case shards::Node::key: // point topology
      return Teuchos::rcp( new PointBasis<>() );
    case shards::Line<2>::key:
      return Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_LINE_C1_FEM<double, FieldContainer<double> >()),
                                                      ONE_D, SCALAR_RANK, Camellia::FUNCTION_SPACE_HGRAD) );
    case shards::Triangle<3>::key:
      return Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_TRI_C1_FEM<double, FieldContainer<double> >()),
                                                      TWO_D, SCALAR_RANK, Camellia::FUNCTION_SPACE_HGRAD) );
    case shards::Quadrilateral<4>::key:
      return Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_QUAD_C1_FEM<double, FieldContainer<double> >()),
                                                      TWO_D, SCALAR_RANK, Camellia::FUNCTION_SPACE_HGRAD) );
    case shards::Tetrahedron<4>::key:
      return Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_TET_C1_FEM<double, FieldContainer<double> >()),
                                                      THREE_D, SCALAR_RANK, Camellia::FUNCTION_SPACE_HGRAD) );
    case shards::Hexahedron<8>::key:
      return Teuchos::rcp( new IntrepidBasisWrapper<>( Teuchos::rcp( new Basis_HGRAD_HEX_C1_FEM<double, FieldContainer<double> >()),
                                                      THREE_D, SCALAR_RANK, Camellia::FUNCTION_SPACE_HGRAD) );
      
/*
    case shards::Wedge<6>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_WEDGE_C1_FEM<Scalar, FieldContainer<Scalar> >() );
      break;
      
    case shards::Pyramid<5>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_PYR_C1_FEM<Scalar, FieldContainer<Scalar> >() );
      break;
      
      // Standard Extended topologies
    case shards::Triangle<6>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_TRI_C2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;
      
    case shards::Quadrilateral<9>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_QUAD_C2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;
      
    case shards::Tetrahedron<10>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_TET_C2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;
      
    case shards::Tetrahedron<11>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_TET_COMP12_FEM<Scalar, FieldContainer<Scalar> >() );
      break;
      
    case shards::Hexahedron<20>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_HEX_I2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;
      
    case shards::Hexahedron<27>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_HEX_C2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;
      
    case shards::Wedge<15>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_WEDGE_I2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;
      
    case shards::Wedge<18>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_WEDGE_C2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;
      
    case shards::Pyramid<13>::key:
      HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_PYR_I2_FEM<Scalar, FieldContainer<Scalar> >() );
      break;
      
      // These extended topologies are not used for mapping purposes
    case shards::Quadrilateral<8>::key:
      TEUCHOS_TEST_FOR_EXCEPTION( (true), std::invalid_argument,
                                 ">>> ERROR (Intrepid::CellTools::mapToPhysicalFrame): Cell topology not supported. ");
      break;
      
      // Base and Extended Line, Beam and Shell topologies
    case shards::Line<3>::key:
    case shards::Beam<2>::key:
    case shards::Beam<3>::key:
    case shards::ShellLine<2>::key:
    case shards::ShellLine<3>::key:
    case shards::ShellTriangle<3>::key:
    case shards::ShellTriangle<6>::key:
    case shards::ShellQuadrilateral<4>::key:
    case shards::ShellQuadrilateral<8>::key:
    case shards::ShellQuadrilateral<9>::key:
      TEUCHOS_TEST_FOR_EXCEPTION( (true), std::invalid_argument,
                                 ">>> ERROR (Intrepid::CellTools::mapToPhysicalFrame): Cell topology not supported. ");
      break;*/
    default:
      TEUCHOS_TEST_FOR_EXCEPTION( true, std::invalid_argument,
                                 ">>> ERROR (BasisFactory::getNodalBasisForCellTopology): Cell topology not supported.");
  }// switch
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


void BasisFactory::registerBasis( BasisPtr basis, int basisRank, int polyOrder, int cellTopoKey, Camellia::EFunctionSpace fs ) {
  pair< pair<int,int>, Camellia::EFunctionSpace > key = make_pair( make_pair(polyOrder, cellTopoKey), fs );
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
  Camellia::EFunctionSpace fs = _functionSpaces[basis.get()];
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
  Camellia::EFunctionSpace fs = _functionSpaces[basis.get()];
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

Camellia::EFunctionSpace BasisFactory::getBasisFunctionSpace(BasisPtr basis) {
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
  MultiBasis<>* multiBasis = dynamic_cast< MultiBasis<>*>(basis.get());
  return multiBasis != NULL;
}

bool BasisFactory::isPatchBasis(BasisPtr basis) {
  return _patchBasisSet.find(basis.get()) != _patchBasisSet.end();
}

void BasisFactory::setUseEnrichedTraces( bool value ) {
  _useEnrichedTraces = value;
}

set<int> BasisFactory::sideFieldIndices( BasisPtr basis, bool includeSideSubcells ) { // includeSideSubcells: e.g. include vertices as part of quad sides
  CellTopoPtr cellTopo = basis->domainTopology();
  int dim = cellTopo->getDimension();
  int sideDim = dim - 1;
  if (sideDim==2) {
    return basis->dofOrdinalsForFaces(includeSideSubcells);
  } else if (sideDim==1) {
    return basis->dofOrdinalsForEdges(includeSideSubcells);
  } else if (sideDim==0) {
    return basis->dofOrdinalsForVertices();
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "can't get side field indices for sideDim != 0, 1, or 2.");
    return set<int>();
  }
}

void BasisFactory::setUseLobattoForQuadHGrad(bool value) {
  _useLobattoForQuadHGRAD = value;
}
void BasisFactory::setUseLobattoForQuadHDiv(bool value) {
  _useLobattoForQuadHDIV = value;
}

Teuchos::RCP<BasisFactory> BasisFactory::basisFactory() { // shared/static instance
  static Teuchos::RCP<BasisFactory> basisFactory = Teuchos::rcp( new BasisFactory() );
  return basisFactory;
}