// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER
//
//  IntrepidBasisConstruction.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/10/13.
//
//

#ifndef Camellia_debug_doubleBasisConstruction_h
#define Camellia_debug_doubleBasisConstruction_h

#include "TypeDefs.h"

#include <Shards_CellTopology.hpp>

namespace Camellia
{
BasisPtr lobattoQuadHGRAD(int polyOrder, bool conforming=false);

BasisPtr intrepid2LineHGRAD(int polyOrder);
BasisPtr intrepid2LineHGRAD_DISC(int polyOrder);
BasisPtr intrepid2LineHVOL(int polyOrder); // polyOrder is the degree of the polynomial space; the H^1 order is one higher than this (contrast BasisFactory)

BasisPtr intrepid2TriHGRAD(int polyOrder);
BasisPtr intrepid2TriHDIV(int polyOrder);
BasisPtr intrepid2TriHDIV_DISC(int polyOrder);

BasisPtr intrepid2QuadHGRAD(int polyOrder);
BasisPtr intrepid2QuadHGRAD_DISC(int polyOrder);
BasisPtr intrepid2QuadHVOL(int polyOrder);

BasisPtr intrepid2HexHGRAD(int polyOrder);
BasisPtr intrepid2HexHGRAD_DISC(int polyOrder);
BasisPtr intrepid2HexHVOL(int polyOrder);

BasisPtr intrepid2TetHGRAD(int polyOrder);
BasisPtr intrepid2TetHGRAD_DISC(int polyOrder);

BasisPtr intrepid2HGRAD(const shards::CellTopology &cellTopo, int polyOrder);
BasisPtr intrepid2HGRADLowestOrder(const shards::CellTopology &cellTopo);
BasisPtr intrepid2HGRADLowestOrder(const unsigned &cellTopoKey);

BasisPtr intrepid2TensorBasis(BasisPtr firstBasis, BasisPtr extrusionBasis);

BasisPtr intrepidQuadHGRAD(int polyOrder);
BasisPtr intrepidQuadHDIV(int polyOrder);

BasisPtr intrepidHexHGRAD(int polyOrder);
BasisPtr intrepidHexHDIV(int polyOrder);

}

#endif
