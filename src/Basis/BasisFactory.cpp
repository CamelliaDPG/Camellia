// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#include "BasisFactory.h"

#include "doubleBasisConstruction.h"

#include "Basis_HVOL_QUAD_C0_FEM.hpp"
#include "Basis_HVOL_TRI_C0_FEM.hpp"

#include "VectorizedBasis.h"

#include "LobattoHGRAD_QuadBasis.h"
#include "LobattoHDIV_QuadBasis.h"
#include "LobattoHDIV_QuadBasis_separable.h"
#include "LegendreHVOL_LineBasis.h"
#include "LegendreHVOL_QuadBasis.h"
#include "LobattoHGRAD_LineBasis.h"

#include "PointBasis.h"
#include "PolynomialOrder.hpp"

#include "TensorBasis.h"

using namespace Camellia;

using ArrayScalar = Intrepid::FieldContainer<double>;

BasisFactory::BasisFactory()
{
  _defaultTemporalPolyOrder = 1; // linear in time by default
  
  _useEnrichedTraces = true;
  _useLobattoForQuadHGRAD = false;
  _useLobattoForQuadHDIV = false;
  _useLobattoForLineHGRAD = false;
  _useLegendreForLineHVOL = false;
  _useLegendreForQuadHVOL = false;
}

BasisPtr BasisFactory::getBasis(const std::vector<int> &H1Orders, CellTopoPtr cellTopo, const std::vector<FSE> &functionSpaces)
{
  PolynomialOrder::checkPolyOrderValidity(H1Orders, cellTopo);
  TEUCHOS_TEST_FOR_EXCEPTION(functionSpaces.size() != H1Orders.size(), std::invalid_argument, "functionSpaces.size() must match H1Orders.size()");
  
  if (H1Orders.size() == 1)
  {
    const int &polyOrder = H1Orders[0];
    const auto shardsKey = cellTopo->getShardsTopology().getKey();
    const auto &fs       = functionSpaces[0];
    
    pair< pair<int,int>, Camellia::EFunctionSpace > key = make_pair( make_pair(polyOrder, shardsKey), fs );
    if (_existingBases.find(key) == _existingBases.end())
    {
      // First, we call getBasis(), and if the one that it returns is conforming, we just use that.
      BasisPtr basis = getBasis(polyOrder, shardsKey, fs);
      
      _existingBases[key] = basis;
      _polyOrders[basis.get()] = polyOrder;
      _functionSpaces[basis.get()] = fs;
      _cellTopoKeys[basis.get()] = shardsKey;
    }
    return _existingBases[key];
  }
  else // more than one tensorial component
  {
    const std::vector<int> H1OrdersTruncated(H1Orders.begin(),H1Orders.end()-1);
    const std::vector<FSE> functionSpacesTruncated(functionSpaces.begin(),functionSpaces.end()-1);
    CellTopoPtr componentTopo = CellTopology::cellTopology(cellTopo->getShardsTopology(), cellTopo->getTensorialDegree() - 1);
    BasisPtr componentBasis = getBasis(H1OrdersTruncated, componentTopo, functionSpacesTruncated);
    
    const std::vector<int> H1OrdersFinal {H1Orders[H1Orders.size()-1]};
    const std::vector<FSE> functionSpaceFinal {functionSpaces[functionSpaces.size()-1]};
    
    auto key = make_tuple(componentBasis.get(), H1OrdersFinal[0], functionSpaceFinal[0]);
    if (_tensorBases.find(key) == _tensorBases.end())
    {
      BasisPtr extrusionBasis = getBasis(H1OrdersFinal, CellTopology::line(), functionSpaceFinal);
      
//      bool rangeDimensionIsSum = true; // This means, e.g., OPERATOR_GRAD is evaluated for all dimensions (contrast with space-time basis)
      // whereever rangeDimensionIsSum was true in construction of TensorBasis, we now want to use Intrepid2::Basis_TensorBasis instead.
      _tensorBases[key] = intrepid2TensorBasis(componentBasis, extrusionBasis);
    }
    return _tensorBases[key];
  }
}

BasisPtr BasisFactory::getBasis( int polyOrder, CellTopoPtr cellTopo, Camellia::EFunctionSpace fs )
{
  if (cellTopo->getTensorialDegree() > 0)
  {
    // then we need to call the getBasis() that takes std::vector arguments
    std::vector<int> H1Orders(cellTopo->getTensorialDegree()+1,polyOrder);
    std::vector<Camellia::EFunctionSpace> functionSpaces(cellTopo->getTensorialDegree()+1,fs);
    return getBasis(H1Orders, cellTopo, functionSpaces);
  }
  else
  {
    // we can call the one that takes a shards key
    return getBasis(polyOrder, cellTopo->getShardsTopology().getKey(), fs);
  }
}

BasisPtr BasisFactory::getBasis( int polyOrder, unsigned cellTopoKey, Camellia::EFunctionSpace fs)
{
  if ((fs != Camellia::FUNCTION_SPACE_REAL_SCALAR) && (cellTopoKey != shards::Node::key))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(polyOrder == 0, std::invalid_argument, "polyOrder = 0 unsupported, except for constant basis");
  }

  BasisPtr basis;
  pair< pair<int,int>, Camellia::EFunctionSpace > key = make_pair( make_pair(polyOrder, cellTopoKey), fs );

  if ( _existingBases.find(key) != _existingBases.end() )
  {
    basis = _existingBases[key];
    return basis;
  }

  int spaceDim;
  bool threeD = (cellTopoKey == shards::Hexahedron<8>::key) || (cellTopoKey == shards::Tetrahedron<4>::key);
  bool twoD = (cellTopoKey == shards::Quadrilateral<4>::key) || (cellTopoKey == shards::Triangle<3>::key);
  bool oneD = (cellTopoKey == shards::Line<2>::key);
  bool zeroD = (cellTopoKey == shards::Node::key);
  if (zeroD)
  {
    spaceDim = 0;
  }
  else if (oneD)
  {
    spaceDim = 1;
  }
  else if (twoD)
  {
    spaceDim = 2;
  }
  else if (threeD)
  {
    spaceDim = 3;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
  }

  int scalarRank = 0, vectorRank = 1;

  if ((fs == Camellia::FUNCTION_SPACE_VECTOR_HGRAD) || (fs == Camellia::FUNCTION_SPACE_VECTOR_HGRAD_DISC))
  {
    BasisPtr componentBasis = BasisFactory::getBasis(polyOrder, cellTopoKey, Camellia::FUNCTION_SPACE_HGRAD);
    basis = Teuchos::rcp( new VectorizedBasis<>(componentBasis,spaceDim) ); // 3-21-13: changed behavior for 1D vectors, but I don't think we use these right now.
  }
  else if (fs == Camellia::FUNCTION_SPACE_VECTOR_HVOL)
  {
    BasisPtr componentBasis = BasisFactory::getBasis(polyOrder, cellTopoKey, Camellia::FUNCTION_SPACE_HVOL);
    basis = Teuchos::rcp( new VectorizedBasis<>(componentBasis,spaceDim) );
  }
  else
  {
    switch (cellTopoKey)
    {
    case shards::Node::key: // point topology
      switch (fs)
      {
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
      switch(fs)
      {
      case Camellia::FUNCTION_SPACE_HGRAD:
          basis = intrepid2HexHGRAD(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HGRAD_DISC:
          basis = intrepid2HexHGRAD_DISC(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HDIV:
          basis = intrepid2HexHDIV(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HDIV_DISC:
          basis = intrepid2HexHDIV_DISC(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HCURL:
          basis = intrepid2HexHCURL(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HCURL_DISC:
          basis = intrepid2HexHCURL_DISC(polyOrder);
        break;
      case Camellia::FUNCTION_SPACE_HVOL:
          basis = intrepid2HexHVOL(polyOrder-1);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported function space for topology Hexahedron<8>");
      }
      break;
    case shards::Tetrahedron<4>::key:
      switch(fs)
      {
      case Camellia::FUNCTION_SPACE_HGRAD:
          basis = intrepid2TetHGRAD(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HGRAD_DISC:
          basis = intrepid2TetHGRAD_DISC(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HDIV:
          basis = intrepid2TetHDIV(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HDIV_DISC:
          basis = intrepid2TetHDIV_DISC(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HCURL:
          basis = intrepid2TetHCURL(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HCURL_DISC:
          basis = intrepid2TetHDIV_DISC(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HVOL:
          basis = intrepid2TetHVOL(polyOrder-1);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported function space for topology Tetrahedron<4>");
      }
      break;
    case shards::Quadrilateral<4>::key:
      switch(fs)
      {
      case Camellia::FUNCTION_SPACE_HGRAD:
      case Camellia::FUNCTION_SPACE_HGRAD_DISC:
        //if (polyOrder==0) {
        //  basis = Teuchos::rcp( new Basis_HVOL_QUAD_C0_FEM<double, ArrayScalar >() ) ;
        //} else {
        if (! _useLobattoForQuadHGRAD)
        {
          if (fs == FUNCTION_SPACE_HGRAD)
          {
            basis = intrepid2QuadHGRAD(polyOrder);
          }
          else
          {
            basis = intrepid2QuadHGRAD_DISC(polyOrder);
          }
        }
        else
        {
          bool conformingFalse = false;
          basis = Teuchos::rcp( new LobattoHGRAD_QuadBasis<double, Intrepid::FieldContainer<double> >(polyOrder,conformingFalse) );
        }
        //}
        break;
      case Camellia::FUNCTION_SPACE_HDIV:
      case Camellia::FUNCTION_SPACE_HDIV_DISC:
        if (! _useLobattoForQuadHDIV )
        {
          basis = intrepid2QuadHDIV(polyOrder);
        }
        else
        {
          bool conformingFalse = false;
          basis = Teuchos::rcp( new LobattoHDIV_QuadBasis<double, Intrepid::FieldContainer<double> >(polyOrder,conformingFalse) );
        }
        break;
      case Camellia::FUNCTION_SPACE_HDIV_FREE:
      {
        bool conformingFalse = false;
        bool divFreeTrue = true;
        basis = Teuchos::rcp( new LobattoHDIV_QuadBasis_separable<double, ArrayScalar >(polyOrder,conformingFalse,divFreeTrue) );
      }
      break;
      case Camellia::FUNCTION_SPACE_HCURL:
          basis = intrepid2QuadHCURL(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HCURL_DISC:
          basis = intrepid2QuadHCURL_DISC(polyOrder);
          break;
      case(Camellia::FUNCTION_SPACE_HVOL):
          if (!_useLegendreForQuadHVOL)
          {
            basis = intrepid2QuadHVOL(polyOrder-1);
          }
          else
          {
            bool conformingFalse = false;
            basis = Teuchos::rcp( new LegendreHVOL_QuadBasis<>(polyOrder-1,conformingFalse) );
          }
        break;
      case(Camellia::FUNCTION_SPACE_REAL_SCALAR):
          // TODO: replace this with an Intrepid2-derived basis.
          basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Intrepid::Basis_HVOL_QUAD_C0_FEM<double, ArrayScalar >()),
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
      switch(fs)
      {
      case Camellia::FUNCTION_SPACE_HGRAD:
          TEUCHOS_TEST_FOR_EXCEPTION(polyOrder <= 0, std::invalid_argument, "Constant basis not supported by Intrepid::Basis_HGRAD_TRI_Cn_FEM");
          basis = intrepid2TriHGRAD(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HGRAD_DISC:
          TEUCHOS_TEST_FOR_EXCEPTION(polyOrder <= 0, std::invalid_argument, "Constant basis not supported by Intrepid::Basis_HGRAD_TRI_Cn_FEM");
          basis = intrepid2TriHGRAD_DISC(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HDIV:
          basis = intrepid2TriHDIV(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HDIV_DISC:
          basis = intrepid2TriHDIV_DISC(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HCURL:
          basis = intrepid2TriHCURL(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HCURL_DISC:
          basis = intrepid2TriHCURL_DISC(polyOrder);
          break;
      case Camellia::FUNCTION_SPACE_HVOL:
          TEUCHOS_TEST_FOR_EXCEPTION(polyOrder < 1, std::invalid_argument, "Invalid H^1 order");
          basis = intrepid2TriHVOL(polyOrder-1);
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
      switch(fs)
      {
      case Camellia::FUNCTION_SPACE_HGRAD:
      case Camellia::FUNCTION_SPACE_HGRAD_DISC:
      {
        int basisPolyOrder = _useEnrichedTraces ? polyOrder : polyOrder - 1;

        if (_useLobattoForLineHGRAD)
        {
          // we use Legendre for HGRAD and HVOL both on the line--since we don't actually
          // take derivatives of traces, this makes sense.
          // but I do have some concern that there may be logic errors to do with the basis's functionSpace()
          // return value: the Intrepid guys will always say HGRAD, and the Legendre HVOL.  Need to look at the
          // way this gets used to see if the conflation will make a difference in e.g. p-refinements.
          bool conformingFalse = false;
          basis = Teuchos::rcp( new LobattoHGRAD_LineBasis<>(basisPolyOrder, conformingFalse) );
        }
        else
        {
          basis = intrepid2LineHGRAD(basisPolyOrder);
        }
      }
      break;
      case Camellia::FUNCTION_SPACE_HVOL:
        if (_useLegendreForLineHVOL)
        {
          basis = Teuchos::rcp( new LegendreHVOL_LineBasis<>(polyOrder-1) );
        }
        else
        {
          basis = intrepid2LineHVOL(polyOrder-1);
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

BasisPtr BasisFactory::getConformingBasis( const std::vector<int> &polyOrder, CellTopoPtr cellTopo, const Camellia::EFunctionSpace fs)
{
  const int requiredPolyOrders = cellTopo->getTensorialDegree() + 1;
  if (polyOrder.size() == 1)
  {
    std::vector<int> polyOrderReplicated(requiredPolyOrders, polyOrder[0]);
    std::vector<Camellia::EFunctionSpace> functionSpaces(requiredPolyOrders, fs);
    return getConformingBasis(polyOrderReplicated, cellTopo, functionSpaces);
  }
  else
  {
    std::vector<Camellia::EFunctionSpace> functionSpaces(polyOrder.size(), fs);
    return getConformingBasis(polyOrder, cellTopo, functionSpaces);
  }
}

BasisPtr BasisFactory::getConformingBasis(const std::vector<int> &H1OrdersIn, CellTopoPtr cellTopo, const std::vector<FSE> &functionSpacesIn)
{
  PolynomialOrder::checkPolyOrderValidity(H1OrdersIn, cellTopo);
  TEUCHOS_TEST_FOR_EXCEPTION(functionSpacesIn.size() != H1OrdersIn.size(), std::invalid_argument, "functionSpaces.size() must match H1Orders.size()");
  
  std::vector<int> H1Orders;
  std::vector<Camellia::EFunctionSpace> functionSpaces;
  if (H1OrdersIn.size() == 1)
  {
    // isotropic; replicate to match tensorial degree
    const int requiredPolyOrders = cellTopo->getTensorialDegree() + 1;
    H1Orders = std::vector<int>(requiredPolyOrders, H1OrdersIn[0]);
    functionSpaces = std::vector<Camellia::EFunctionSpace>(requiredPolyOrders, functionSpacesIn[0]);
  }
  else
  {
    H1Orders = H1OrdersIn;
    functionSpaces = functionSpacesIn;
  }
  
  if (cellTopo->getTensorialDegree() == 0)
  {
    const int &polyOrder = H1Orders[0];
    const auto shardsKey = cellTopo->getShardsTopology().getKey();
    const auto &fs       = functionSpaces[0];
    
    // NOTE: the logic here does not support anisotropic polynomial orders for shards quadrilaterals and hexahedra because these have zero tensorial degree.
    // (We may want to change this once we are more fully using Intrepid2 basis functions.)
    
    pair< pair<int,int>, Camellia::EFunctionSpace > key = make_pair( make_pair(polyOrder, shardsKey), fs );
    if (_conformingBases.find(key) == _conformingBases.end())
    {
      BasisPtr basis;
      const int spaceDim = cellTopo->getDimension();
      
      // First, we call getBasis(), and if the one that it returns is conforming, we just use that.
      BasisPtr standardBasis = getBasis(polyOrder, shardsKey, fs);
      
      if (standardBasis->isConforming())
      {
        basis = standardBasis;
      }
      else
      {
        // confirm that what is requested is isotropic in H1Order:
        for (int d=0; d<H1Orders.size(); d++)
        {
          TEUCHOS_TEST_FOR_EXCEPTION(H1Orders[d] != polyOrder, std::invalid_argument, "Anisotropic polynomial order not (yet) supported here.");
        }
        
        if (fs == Camellia::FUNCTION_SPACE_VECTOR_HGRAD)
        {
          BasisPtr componentBasis = BasisFactory::getConformingBasis(polyOrder, shardsKey, Camellia::FUNCTION_SPACE_HGRAD);
          basis = Teuchos::rcp( new VectorizedBasis<>(componentBasis,spaceDim) ); // 3-21-13: changed behavior for 1D vectors, but I don't think we use these right now.
        }
        else if (fs == Camellia::FUNCTION_SPACE_VECTOR_HVOL)
        {
          BasisPtr componentBasis = BasisFactory::getConformingBasis(polyOrder, shardsKey, Camellia::FUNCTION_SPACE_HVOL);
          basis = Teuchos::rcp( new VectorizedBasis<>(componentBasis,spaceDim) );
        }
        else
        {
          const int scalarRank = 0, vectorRank = 1;
          switch (shardsKey)
          {
          case shards::Quadrilateral<4>::key:
            switch(fs)
            {
            case Camellia::FUNCTION_SPACE_HGRAD:
            case Camellia::FUNCTION_SPACE_HGRAD_DISC:
              //if (polyOrder==0) {
              //  basis = Teuchos::rcp( new Basis_HVOL_QUAD_C0_FEM<double, ArrayScalar >() ) ;
              //} else {
              if (! _useLobattoForQuadHGRAD)
              {
                if (fs == FUNCTION_SPACE_HGRAD)
                {
                  basis = intrepid2QuadHGRAD(polyOrder);
                }
                else
                {
                  basis = intrepid2QuadHGRAD_DISC(polyOrder);
                }
              }
              else
              {
                bool conformingTrue = true;
                basis = Teuchos::rcp( new LobattoHGRAD_QuadBasis<double, Intrepid::FieldContainer<double> >(polyOrder,conformingTrue) );
              }
              //}
              break;
            case Camellia::FUNCTION_SPACE_HDIV:
            case Camellia::FUNCTION_SPACE_HDIV_DISC:
              if (! _useLobattoForQuadHDIV)
              {
                basis = (fs == FUNCTION_SPACE_HDIV) ? intrepid2QuadHDIV(polyOrder) : intrepid2QuadHDIV_DISC(polyOrder);
              }
              else
              {
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
                basis = intrepid2QuadHCURL(polyOrder);
              break;
            case Camellia::FUNCTION_SPACE_HCURL_DISC:
                basis = intrepid2QuadHCURL_DISC(polyOrder);
              break;
            case Camellia::FUNCTION_SPACE_HVOL:
                basis =  intrepid2QuadHVOL(polyOrder-1);
              break;
            case Camellia::FUNCTION_SPACE_REAL_SCALAR:
                basis = Teuchos::rcp( new IntrepidBasisWrapper<>(Teuchos::rcp( new Intrepid::Basis_HVOL_QUAD_C0_FEM<double, ArrayScalar >()),
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
            switch(fs)
            {
            case Camellia::FUNCTION_SPACE_HGRAD:
            case Camellia::FUNCTION_SPACE_HGRAD_DISC:
              if (fs == Camellia::FUNCTION_SPACE_HGRAD)
              {
                basis = intrepid2TriHGRAD(polyOrder);
              }
              else
              {
                basis = intrepid2TriHGRAD_DISC(polyOrder);
              }
              break;
            case Camellia::FUNCTION_SPACE_HDIV:
            case Camellia::FUNCTION_SPACE_HDIV_DISC:
              if (fs == Camellia::FUNCTION_SPACE_HDIV)
              {
                basis = intrepid2TriHDIV(polyOrder);
              }
              else
              {
                basis = intrepid2TriHDIV_DISC(polyOrder);
              }
              break;
            case Camellia::FUNCTION_SPACE_HCURL:
                basis = intrepid2TriHCURL(polyOrder);
              break;
            case Camellia::FUNCTION_SPACE_HCURL_DISC:
                basis = intrepid2TriHCURL_DISC(polyOrder);
              break;
            case(Camellia::FUNCTION_SPACE_HVOL):
              basis = intrepid2TriHGRAD_DISC(polyOrder-1); // TODO: check whether we can switch to intrepid2TriHVOL() here (a different basis).
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
            switch(fs)
            {
            case Camellia::FUNCTION_SPACE_HGRAD:
            case Camellia::FUNCTION_SPACE_HGRAD_DISC:
            {
              int basisPolyOrder = _useEnrichedTraces ? polyOrder : polyOrder - 1;

              if (_useLobattoForLineHGRAD)
              {
                // we use Legendre for HGRAD and HVOL both on the line--since we don't actually
                // take derivatives of traces, this makes sense.
                // but I do have some concern that there may be logic errors to do with the basis's functionSpace()
                // return value: the Intrepid guys will always say HGRAD, and the Legendre HVOL.  Need to look at the
                // way this gets used to see if the conflation will make a difference in e.g. p-refinements.
                bool conformingTrue = true;
                basis = Teuchos::rcp( new LobattoHGRAD_LineBasis<>(basisPolyOrder, conformingTrue) );
              }
              else
              {
                if (fs == Camellia::FUNCTION_SPACE_HGRAD) basis = intrepid2LineHGRAD(basisPolyOrder);
                else                                      basis = intrepid2LineHGRAD_DISC(basisPolyOrder);
              }
            }
            break;
            case Camellia::FUNCTION_SPACE_HVOL:
              if (_useLegendreForLineHVOL)   // not actually conforming, except in the L^2 sense...
              {
                basis = Teuchos::rcp( new LegendreHVOL_LineBasis<>(polyOrder-1) );
              }
              else
              {
                basis = intrepid2LineHVOL(polyOrder-1);
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
            switch (fs)
            {
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
            TEUCHOS_TEST_FOR_EXCEPTION( ( (shardsKey != shards::Quadrilateral<4>::key)        &&
                                          (shardsKey != shards::Triangle<3>::key)             &&
                                          (shardsKey != shards::Line<2>::key)                 &&
                                          (shardsKey != shards::Node::key)),
                                        std::invalid_argument,
                                        "Unknown cell topology for basis selection. Please use Node, Line_2, Quadrilateral_4, or Triangle_3.");

          }
        }
      }
      
      _conformingBases[key] = basis;
      _polyOrders[basis.get()] = polyOrder;
      _functionSpaces[basis.get()] = fs;
      _cellTopoKeys[basis.get()] = shardsKey;
    }
    return _conformingBases[key];
  }
  else // more than one tensorial component
  {
    const std::vector<int> H1OrdersTruncated(H1Orders.begin(),H1Orders.end()-1);
    const std::vector<FSE> functionSpacesTruncated(functionSpaces.begin(),functionSpaces.end()-1);
    CellTopoPtr componentTopo = CellTopology::cellTopology(cellTopo->getShardsTopology(), cellTopo->getTensorialDegree() - 1);
    BasisPtr componentBasis = getConformingBasis(H1OrdersTruncated, componentTopo, functionSpacesTruncated);
    
    const std::vector<int> H1OrdersFinal {H1Orders[H1Orders.size()-1]};
    const std::vector<FSE> functionSpaceFinal {functionSpaces[functionSpaces.size()-1]};
    
    
    auto key = make_tuple(componentBasis.get(), H1OrdersFinal[0], functionSpaceFinal[0]);
    if (_conformingTensorBases.find(key) == _conformingTensorBases.end())
    {
      BasisPtr extrusionBasis = getConformingBasis(H1OrdersFinal, CellTopology::line(), functionSpaceFinal);
//      bool rangeDimensionIsSum = true; // This means, e.g., OPERATOR_GRAD is evaluated for all dimensions (contrast with space-time basis)
      // whereever rangeDimensionIsSum was true in construction of TensorBasis, we now want to use Intrepid2::Basis_TensorBasis instead.
      _conformingTensorBases[key] = intrepid2TensorBasis(componentBasis, extrusionBasis);
    }
    return _conformingTensorBases[key];
  }
}

BasisPtr BasisFactory::getConformingBasis( int polyOrder, CellTopoPtr cellTopo, Camellia::EFunctionSpace fs )
{
  if (cellTopo->getTensorialDegree() > 0)
  {
    // then we need to call the getConformingBasis() that takes std::vector arguments
    std::vector<int> H1Orders(cellTopo->getTensorialDegree()+1,polyOrder);
    std::vector<Camellia::EFunctionSpace> functionSpaces(cellTopo->getTensorialDegree()+1,fs);
    return getConformingBasis(H1Orders, cellTopo, functionSpaces);
  }
  else
  {
    // we can call the one that takes a shards key
    return getConformingBasis(polyOrder, cellTopo->getShardsTopology().getKey(), fs);
  }
}

BasisPtr BasisFactory::getConformingBasis( int polyOrder, unsigned cellTopoKey, Camellia::EFunctionSpace fs )
{
  // This method provides the chance to offer different bases when a conforming basis is required.
  // Specifically, getBasis() returns non-conforming basis for the relevant cell topo/function space combinations when:
  // - _useLobattoForQuadHGRAD is true
  // - _useLobattoForQuadHDIV  is true
  // - _useLegendreForQuadHVOL is true
  // - _useLobattoForLineHGRAD is true
  // and for FUNCTION_SPACE_HDIV_FREE no matter what.

  BasisPtr basis;
  pair< pair<int,int>, Camellia::EFunctionSpace > key = make_pair( make_pair(polyOrder, cellTopoKey), fs );

  // First, we call getBasis(), and if the one that it returns is conforming, we just use that.
  BasisPtr standardBasis = getBasis(polyOrder, cellTopoKey, fs);
  if (standardBasis->isConforming())
  {
    _conformingBases[key] = standardBasis;
    return standardBasis;
  }

  if ( _conformingBases.find(key) != _conformingBases.end() )
  {
    basis = _conformingBases[key];
    return basis;
  }
  
    switch (cellTopoKey)
    {
    case shards::Quadrilateral<4>::key:
      switch(fs)
      {
      case Camellia::FUNCTION_SPACE_HGRAD:
      case Camellia::FUNCTION_SPACE_HGRAD_DISC:
        if (_useLobattoForQuadHGRAD)
        {
          bool conformingTrue = true;
          basis = Teuchos::rcp( new LobattoHGRAD_QuadBasis<double, Intrepid::FieldContainer<double> >(polyOrder,conformingTrue) );
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "non-conforming basis should only be returned by getBasis() here if _useLobattoForQuadHGRAD is true");
        }
        break;
      case Camellia::FUNCTION_SPACE_HDIV:
      case Camellia::FUNCTION_SPACE_HDIV_DISC:
        if (_useLobattoForQuadHDIV)
        {
          bool conformingTrue = true;
          basis = Teuchos::rcp( new LobattoHDIV_QuadBasis<double, Intrepid::FieldContainer<double> >(polyOrder,conformingTrue) );
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "non-conforming basis should only be returned by getBasis() here if _useLobattoForQuadHDIV is true");
        }
        break;
      case Camellia::FUNCTION_SPACE_HDIV_FREE:
      {
        bool conformingTrue = true;
        bool divFreeTrue = true;
        basis = Teuchos::rcp( new LobattoHDIV_QuadBasis<double, Intrepid::FieldContainer<double> >(polyOrder,conformingTrue,divFreeTrue) );
      }
      break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION( true, std::invalid_argument, "Unhandled case for quad_4.");
      }
      break;
    case shards::Line<2>::key:
      switch(fs)
      {
      case Camellia::FUNCTION_SPACE_HGRAD:
      case Camellia::FUNCTION_SPACE_HGRAD_DISC:
      {
        int basisPolyOrder = _useEnrichedTraces ? polyOrder : polyOrder - 1;

        if (_useLobattoForLineHGRAD)
        {
          // we use Legendre for HGRAD and HVOL both on the line--since we don't actually
          // take derivatives of traces, this makes sense.
          // but I do have some concern that there may be logic errors to do with the basis's functionSpace()
          // return value: the Intrepid guys will always say HGRAD, and the Legendre HVOL.  Need to look at the
          // way this gets used to see if the conflation will make a difference in e.g. p-refinements.
          bool conformingTrue = true;
          basis = Teuchos::rcp( new LobattoHGRAD_LineBasis<>(basisPolyOrder, conformingTrue) );
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "non-conforming basis should only be returned by getBasis() here if _useLobattoForLineHGRAD is true");
        }
      }
      break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION( true, std::invalid_argument, "Unhandled case for line_2.");
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION( true, std::invalid_argument, "Unhandled cell topology.");
    }
  
  _conformingBases[key] = basis;
  _polyOrders[basis.get()] = polyOrder;
  _functionSpaces[basis.get()] = fs;
  _cellTopoKeys[basis.get()] = cellTopoKey;
  return basis;
}

MultiBasisPtr BasisFactory::getMultiBasis(vector< BasisPtr > &bases)
{
  vector< Camellia::Basis<>* > key;
  int numBases = bases.size();
  for (int i=0; i<numBases; i++)
  {
    key.push_back(bases[i].get());
  }
  if (_multiBasesMap.find(key) != _multiBasesMap.end() )
  {
    return _multiBasesMap[key];
  }

  int polyOrder = 0;
  Camellia::EFunctionSpace fs;
  unsigned cellTopoKey;

  if (numBases != 2)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"BasisFactory only supports lines divided in two right now.");
  }

  for (int i=0; i<numBases; i++)
  {
    if (! basisKnown(bases[i]) )
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisFactory can only make MultiBasis from registered Basis pointers.");
    }

    polyOrder = max(polyOrder,_polyOrders[bases[i].get()]);
    fs = _functionSpaces[bases[i].get()];
    cellTopoKey = _cellTopoKeys[bases[i].get()];
  }

  if ((_cellTopoKeys[bases[0].get()] != shards::Line<2>::key)
      || (_cellTopoKeys[bases[1].get()] != shards::Line<2>::key) )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"BasisFactory only supports lines divided in two right now.");
  }
  int spaceDim=1, numNodesPerSubRefCell = 2;
  Camellia::FieldContainer<double> subRefNodes(numBases,numNodesPerSubRefCell,spaceDim);
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

BasisPtr BasisFactory::getNodalBasisForCellTopology(CellTopoPtr cellTopo)
{
  if ((cellTopo->getBaseTopology()->getDimension() == 0) && (cellTopo->getDimension() > 0))
  {
    // tensor topo with base as node; get the equivalent guy with the line as base
    int numExtrusions = cellTopo->getTensorialDegree() - 1;
    auto equivalentCellTopo = CellTopology::cellTopology(CellTopology::line(), numExtrusions);
    return getNodalBasisForCellTopology(equivalentCellTopo);
  }
  
  if (_nodalBasisForTopology.find(cellTopo->getKey()) == _nodalBasisForTopology.end())
  {
    BasisPtr shardsNodalBasis = BasisFactory::basisFactory()->getNodalBasisForCellTopology(cellTopo->getShardsTopology().getKey());
    
    if (cellTopo->getTensorialDegree() == 0)
    {
      _nodalBasisForTopology[cellTopo->getKey()] = shardsNodalBasis;
    }
    else
    {
      BasisPtr lineNodalBasis = BasisFactory::basisFactory()->getNodalBasisForCellTopology(shards::Line<2>::key);
      
      typedef Camellia::TensorBasis<double, Intrepid::FieldContainer<double> > TensorBasis;
      
      BasisPtr nodalBasis = shardsNodalBasis;
//      bool rangeDimensionIsSum = true;
      for (int i=0; i<cellTopo->getTensorialDegree(); i++)
      {
        // whereever rangeDimensionIsSum was true in construction of TensorBasis, we now want to use Intrepid2::Basis_TensorBasis instead.
        nodalBasis = intrepid2TensorBasis(nodalBasis, lineNodalBasis);
//        nodalBasis = Teuchos::rcp( new TensorBasis(nodalBasis, lineNodalBasis, rangeDimensionIsSum) );
      }
      _nodalBasisForTopology[cellTopo->getKey()] = nodalBasis;
    }
  }
  return _nodalBasisForTopology[cellTopo->getKey()];
}

BasisPtr BasisFactory::getNodalBasisForCellTopology(unsigned int cellTopoKey)
{
  // used by CamelliaCellTools for computing Jacobians, etc.
  
  if (_nodalBasisForShardsTopology.find(cellTopoKey) == _nodalBasisForShardsTopology.end())
  {
    static const int TWO_D = 2, THREE_D = 3;
    static const int SCALAR_RANK = 0;
    switch( cellTopoKey )
    {
        // Standard Base topologies (number of cellWorkset = number of vertices)
      case shards::Node::key: // point topology
        _nodalBasisForShardsTopology[cellTopoKey] = Teuchos::rcp( new PointBasis<>() );
        break;
      case shards::Line<2>::key:
      case shards::Triangle<3>::key:
      case shards::Quadrilateral<4>::key:
      case shards::Tetrahedron<4>::key:
      case shards::Hexahedron<8>::key:
        _nodalBasisForShardsTopology[cellTopoKey] = intrepid2HGRADLowestOrder(cellTopoKey);
        break;
        /*
         case shards::Wedge<6>::key:
         HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_WEDGE_C1_FEM<Scalar, Camellia::FieldContainer<Scalar> >() );
         break;
         
         case shards::Pyramid<5>::key:
         HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_PYR_C1_FEM<Scalar, Camellia::FieldContainer<Scalar> >() );
         break;
         
         // Standard Extended topologies
         case shards::Triangle<6>::key:
         HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_TRI_C2_FEM<Scalar, Camellia::FieldContainer<Scalar> >() );
         break;
         
         case shards::Quadrilateral<9>::key:
         HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_QUAD_C2_FEM<Scalar, Camellia::FieldContainer<Scalar> >() );
         break;
         
         case shards::Tetrahedron<10>::key:
         HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_TET_C2_FEM<Scalar, Camellia::FieldContainer<Scalar> >() );
         break;
         
         case shards::Tetrahedron<11>::key:
         HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_TET_COMP12_FEM<Scalar, Camellia::FieldContainer<Scalar> >() );
         break;
         
         case shards::Hexahedron<20>::key:
         HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_HEX_I2_FEM<Scalar, Camellia::FieldContainer<Scalar> >() );
         break;
         
         case shards::Hexahedron<27>::key:
         HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_HEX_C2_FEM<Scalar, Camellia::FieldContainer<Scalar> >() );
         break;
         
         case shards::Wedge<15>::key:
         HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_WEDGE_I2_FEM<Scalar, Camellia::FieldContainer<Scalar> >() );
         break;
         
         case shards::Wedge<18>::key:
         HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_WEDGE_C2_FEM<Scalar, Camellia::FieldContainer<Scalar> >() );
         break;
         
         case shards::Pyramid<13>::key:
         HGRAD_Basis = Teuchos::rcp( new Basis_HGRAD_PYR_I2_FEM<Scalar, Camellia::FieldContainer<Scalar> >() );
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
  return _nodalBasisForShardsTopology[cellTopoKey];
}

BasisPtr BasisFactory::getSpaceTimeConformingBasis( const std::vector<int> &polyOrderSpace, const int polyOrderTime,
                                                   CellTopoPtr spaceTimeCellTopo, const Camellia::EFunctionSpace spatialFS, const Camellia::EFunctionSpace temporalFS)
{
  auto spatialTopo = spaceTimeCellTopo->getTensorialComponent();
  const int requiredPolyOrders = spatialTopo->getTensorialDegree() + 1;
  if (polyOrderSpace.size() == 1)
  {
    std::vector<int> polyOrder(requiredPolyOrders, polyOrderSpace[0]);
    std::vector<Camellia::EFunctionSpace> functionSpaces(requiredPolyOrders, spatialFS);
    return getSpaceTimeConformingBasis(polyOrder, polyOrderTime, spaceTimeCellTopo, functionSpaces, temporalFS);
  }
  else
  {
    std::vector<Camellia::EFunctionSpace> functionSpaces(requiredPolyOrders, spatialFS);
    return getSpaceTimeConformingBasis(polyOrderSpace, polyOrderTime, spaceTimeCellTopo, functionSpaces, temporalFS);
  }
}

BasisPtr BasisFactory::getSpaceTimeConformingBasis( const std::vector<int> &polyOrderSpace, const int polyOrderTime,
                                                   CellTopoPtr spaceTimeCellTopo, const std::vector<Camellia::EFunctionSpace> &spatialFS, const Camellia::EFunctionSpace temporalFS)
{
  const int tensorialDegree = spaceTimeCellTopo->getTensorialDegree();
  TEUCHOS_TEST_FOR_EXCEPTION(spaceTimeCellTopo->getTensorialDegree() == 0, std::invalid_argument, "Space-time basis requires tensorial degree of at least 1");
  
  std::vector<int> H1Orders = polyOrderSpace;
  H1Orders.push_back(polyOrderTime);
  
  std::vector<EFunctionSpace> functionSpaces = spatialFS;
  functionSpaces.push_back(temporalFS);
  
  return getConformingBasis(H1Orders, spaceTimeCellTopo, functionSpaces);
}

BasisPtr BasisFactory::getSpaceTimeConformingBasis( const int polyOrderSpace, const int polyOrderTime,
                                                   CellTopoPtr spaceTimeCellTopo, const FSE spatialFS, const FSE temporalFS)
{
  // Line has tensorial degree zero (and there we get to choose 1 polynomial order, and 1 function space).
  // We get one more choice for each tensorial extrusion in space.
  // The number of choices we get for poly order, function space in *spatial* dimensions is thus exactly the tensorial degree of the *space-time* topology.
  const int tensorialDegree = spaceTimeCellTopo->getTensorialDegree();
  std::vector<int> polyOrderSpaceVector(tensorialDegree,polyOrderSpace);
  std::vector<FSE> spatialFunctionSpaces(tensorialDegree,spatialFS);
  return getSpaceTimeConformingBasis(polyOrderSpaceVector, polyOrderTime, spaceTimeCellTopo, spatialFunctionSpaces, temporalFS);
}

void BasisFactory::registerBasis( BasisPtr basis, int basisRank, int polyOrder, int cellTopoKey, Camellia::EFunctionSpace fs )
{
  pair< pair<int,int>, Camellia::EFunctionSpace > key = make_pair( make_pair(polyOrder, cellTopoKey), fs );
  if ( _existingBases.find(key) != _existingBases.end() )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "Can't register a basis for which there's already an entry...");
  }
  _existingBases[key] = basis;
  _polyOrders[basis.get()] = polyOrder;
  _functionSpaces[basis.get()] = fs;
  _cellTopoKeys[basis.get()] = cellTopoKey;
}

BasisPtr BasisFactory::addToPolyOrder(BasisPtr basis, int pToAdd)
{
  int polyOrder = _polyOrders[basis.get()] + pToAdd;
  Camellia::EFunctionSpace fs = _functionSpaces[basis.get()];
  int cellTopoKey = _cellTopoKeys[basis.get()];
  if (basis->isConforming())
  {
    return getConformingBasis(polyOrder, cellTopoKey, fs);
  }
  else
  {
    return getBasis(polyOrder, cellTopoKey, fs);
  }
}

BasisPtr BasisFactory::setPolyOrder(BasisPtr basis, int pToSet)
{
  if (isMultiBasis(basis))
  {
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
  if (basis->isConforming())
  {
    return getConformingBasis(pToSet, cellTopoKey, fs);
  }
  else
  {
    return getBasis(pToSet, cellTopoKey, fs);
  }
}

int BasisFactory::getBasisRank(BasisPtr basis, EOperator op)
{
  switch (op)
  {
    case OP_VALUE:
    case OP_LAPLACIAN:
    case OP_DX:
    case OP_DY:
    case OP_DZ:
    case OP_DXDX:
    case OP_DXDY:
    case OP_DXDZ:
    case OP_DYDY:
    case OP_DYDZ:
    case OP_DZDZ:
      return basis->rangeRank();
    case OP_X:
    case OP_Y:
    case OP_Z:
    case OP_DIV:
      return basis->rangeRank() - 1;
    case OP_GRAD:
      return basis->rangeRank() + 1;
    case OP_CURL:
      if (basis->rangeDimension() == 2)
      {
        // then vectors go to scalars, and vice versa
        if (basis->rangeRank() == 0)
          return 1;
        else if (basis->rangeRank() == 1)
          return 0;
        else
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled rangeRank() for OP_CURL");
      }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled operator");
  }
}

Camellia::EFunctionSpace BasisFactory::getBasisFunctionSpace(BasisPtr basis)
{
  return _functionSpaces[basis.get()];
}

int BasisFactory::basisPolyOrder(BasisPtr basis)
{
  return _polyOrders[basis.get()];
}

bool BasisFactory::basisKnown(BasisPtr basis)
{
  // look it up in one of our universal maps...
  return _polyOrders.find(basis.get()) != _polyOrders.end();
}

bool BasisFactory::isMultiBasis(BasisPtr basis)
{
  MultiBasis<>* multiBasis = dynamic_cast< MultiBasis<>*>(basis.get());
  return multiBasis != NULL;
}

void BasisFactory::setUseEnrichedTraces( bool value )
{
  _useEnrichedTraces = value;
}

set<int> BasisFactory::sideFieldIndices( BasisPtr basis, bool includeSideSubcells )   // includeSideSubcells: e.g. include vertices as part of quad sides
{
  CellTopoPtr cellTopo = basis->domainTopology();
  int dim = cellTopo->getDimension();
  int sideDim = dim - 1;
  if (sideDim==2)
  {
    return basis->dofOrdinalsForFaces(includeSideSubcells);
  }
  else if (sideDim==1)
  {
    return basis->dofOrdinalsForEdges(includeSideSubcells);
  }
  else if (sideDim==0)
  {
    return basis->dofOrdinalsForVertices();
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "can't get side field indices for sideDim != 0, 1, or 2.");
  }
}

void BasisFactory::setUseLegendreForQuadHVol(bool value)
{
  _useLegendreForQuadHVOL = value;
}

void BasisFactory::setUseLobattoForQuadHGrad(bool value)
{
  _useLobattoForQuadHGRAD = value;
}

void BasisFactory::setUseLobattoForQuadHDiv(bool value)
{
  _useLobattoForQuadHDIV = value;
}

namespace Camellia
{
  static Teuchos::RCP<BasisFactory> sharedBasisFactory;
}

Teuchos::RCP<BasisFactory> BasisFactory::basisFactory()   // shared/static instance
{
  if (sharedBasisFactory == Teuchos::null)
  {
    Camellia::sharedBasisFactory = Teuchos::rcp( new BasisFactory() );
  }
  return sharedBasisFactory;
}

void BasisFactory::deleteBasisFactory()   // delete shared/static instance
{
  int strong_count = sharedBasisFactory.strong_count();
  TEUCHOS_ASSERT(strong_count <= 1); // by the time deleteBasisFactory() is called, there should be only one (could be zero if sharedBasisFactory has never been used)
  sharedBasisFactory = Teuchos::null;
}
