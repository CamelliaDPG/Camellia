//
//  PolynomialOrder.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 1/25/21.
//

#ifndef PolynomialOrder_h
#define PolynomialOrder_h

#include "Camellia.h"

#include "CellTopology.h"

namespace Camellia {
  class PolynomialOrder {
  public:
    /**
      \brief private method to check that the polyOrder given is valid.
     A single-entry polyOrder is always valid, and taken as isotropic when the cell topology has tensor structure.
     Throws std::invalid_argument if not valid.
     */
    static void checkPolyOrderValidity(const vector<int> &polyOrder, CellTopoPtr cellTopo)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(polyOrder.size() == 0, std::invalid_argument, "Empty polyOrder vector not valid!");
      if (polyOrder.size() == 1) return; // single-entry is always valid
      if ((polyOrder.size() == 2) && (cellTopo->getTensorialDegree() >= 1)) return; // can be understood as space-time (isotropic in space)
      if (polyOrder.size() == cellTopo->getTensorialDegree() + 1) return; // one entry for base dimension, one per tensorial extrusion
      if (cellTopo->getTensorialDegree() == 0)
      {
        if (cellTopo->getShardsTopology().getKey() == shards::Quadrilateral<>::key)
        {
          // tensorial degree is technically 0, but we support 2-entry anisotropic polyOrder
          TEUCHOS_TEST_FOR_EXCEPTION(polyOrder.size() != 2, std::invalid_argument, "polyOrder for quadrilateral must have 1 entry (isotropic) or 2 (anisotropic)");
        }
        else if (cellTopo->getShardsTopology().getKey() == shards::Hexahedron<>::key)
        {
          // tensorial degree is technically 0, but we support 3-entry anisotropic polyOrder
          TEUCHOS_TEST_FOR_EXCEPTION(polyOrder.size() != 3, std::invalid_argument, "polyOrder for hexahedron must have 1 entry (isotropic) or 3 (anisotropic)");
        }
        // TODO: support anisotropic tensorial extrusions of shards quadrilateral, hexahedron.
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "polyOrder is not valid for the cell topology!");
      }
    }
    
    /**
          \brief Given a possibly anisotropic polynomial order on a volume, determine the polynomial order of the restriction to the specified side.
     */
    static std::vector<int> sidePolynomialOrder(const vector<int> &polyOrder, CellTopoPtr cellTopo, int sideOrdinal)
    {
      std::vector<int> sideOrder;
      const int tensorialDegree = cellTopo->getTensorialDegree();
      const unsigned shardsKey = cellTopo->getShardsTopology().getKey();
      if (tensorialDegree == 0)
      {
        // the base shards topology dictates everything:
        switch (shardsKey)
        {
          case shards::Node::key:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Node does not have sides; asking for its sidePolynomialOrder is not supported");
          case shards::Line<>::key:
            // sides are nodes
            // nodes only support 0-order polynomials (constants)
            sideOrder = std::vector<int>{0};
            break;
          case shards::Triangle<>::key:
          case shards::Tetrahedron<>::key:
            TEUCHOS_TEST_FOR_EXCEPTION(polyOrder.size() != 1, std::invalid_argument, "For topologies that support only isotropic polynomial order, polyOrder must have only one entry.");
            sideOrder = std::vector<int>{polyOrder[0]};
            break;
          
          case shards::Quadrilateral<>::key:
            if (polyOrder.size() == 1)
            {
              // understood as isotropic in p
              sideOrder = std::vector<int>{polyOrder[0]};
            }
            else
            {
              // in x-y plane, sides are numbered: bottom, right, top, left
              if ((sideOrdinal == 0) || (sideOrdinal == 2))
              {
                // x polyOrder:
                sideOrder = std::vector<int>{polyOrder[0]};
              }
              else
              {
                sideOrder = std::vector<int>{polyOrder[1]};
              }
            }
            break;
            
          case shards::Hexahedron<>::key:
            if (polyOrder.size() == 1)
            {
              // understood as isotropic in p
              sideOrder = std::vector<int>{polyOrder[0]};
            }
            else
            {
              TEUCHOS_TEST_FOR_EXCEPTION(polyOrder.size() != 3, std::invalid_argument, "For p-anisotropic hexahedron, polyOrder must have size 3.");
              // sides go in order:
              // bottom (xy): 0,
              // top (xy):    1,
              // front (xz):  2,
              // right (yz):  3,
              // back (xz):   4,
              // left (yz):   5
              const int & x = polyOrder[0];
              const int & y = polyOrder[1];
              const int & z = polyOrder[2];
              switch (sideOrdinal)
              {
                case 0:
                case 1:
                  sideOrder = std::vector<int>{x,y};
                  break;
                case 2:
                case 4:
                  sideOrder = std::vector<int>{x,z};
                  break;
                case 3:
                case 5:
                  sideOrder = std::vector<int>{y,z};
                  break;
                
                default:
                  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Side ordinal is out of bounds");
              }
            }
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported/unhandled cell type");
        }
      }
      else // tensorialDegree >= 1
      {
        if (polyOrder.size() == 1)
        {
          // understood as isotropic in p
          sideOrder = std::vector<int>{polyOrder[0]};
        }
        else
        {
          const bool sideIsExtrudedInFinalDimension = cellTopo->sideIsExtrudedInFinalDimension(sideOrdinal);
          const int& polyOrderInFinalDimension = polyOrder[polyOrder.size()-1];
          
          // lop off the last entry in polyOrder to get the polynomial order for the non-extruded ("component") sides
          auto componentSideOrder = std::vector<int>(polyOrder.begin(), polyOrder.end()-1);
          if (!sideIsExtrudedInFinalDimension)
          {
            // this side just lops off the last dimension; sideOrder is just the first N-1 entries of polyOrder
            sideOrder = componentSideOrder;
          }
          else
          {
            const auto componentSideOrdinal = cellTopo->getSpatialComponentSideOrdinal(sideOrdinal);
            CellTopoPtr componentTopo = cellTopo->getTensorialComponent();
            sideOrder = sidePolynomialOrder(componentSideOrder, componentTopo, componentSideOrdinal);
            sideOrder.push_back(polyOrderInFinalDimension);
          }
        }
      }
      return sideOrder;
    }
    
    /**
      \brief determines the temporal component of the given polyOrder.  If there is only one entry, then the poly order is understood to be isotropic; otherwise, it is the last entry in polyOrder.
     */
    static int temporalPolyOrder(const vector<int> &polyOrder)
    {
      if (polyOrder.size() == 1) return polyOrder[0]; // understood as isotropic in space-time
      else return polyOrder[polyOrder.size()-1]; // last entry corresponds to temporal poly order
    }
  };
}


#endif /* PolynomialOrder_h */
