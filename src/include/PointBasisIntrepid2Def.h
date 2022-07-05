// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  PointBasisIntrepid2Def.h
//
//  Created by Nathan Roberts on 4/12/22.
//
//

#ifndef Camellia_PointBasisIntrepid2Def_h
#define Camellia_PointBasisIntrepid2Def_h

#include "CamelliaIntrepidExtendedTypes.h"

#include "CellTopology.h"

namespace Camellia
{
  template<typename DT, typename OT, typename PT>
  PointBasisIntrepid2<DT,OT,PT>::
  PointBasisIntrepid2()
  {
    using Intrepid2::ordinal_type;
    const ordinal_type spaceDim = 0;

    this->basisCardinality_  = 1;
    this->basisDegree_       = 0;
    this->basisCellTopology_ = shards::getCellTopologyData<shards::Node >();
    this->basisType_         = Intrepid2::BASIS_FEM_DEFAULT;
    this->basisCoordinates_  = Intrepid2::COORDINATES_CARTESIAN;
    this->functionSpace_     = Intrepid2::FUNCTION_SPACE_HVOL;

    // initialize tags
    {
      // Basis-dependent intializations
      const ordinal_type tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
      const ordinal_type posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
      const ordinal_type posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
      const ordinal_type posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell

      // An array with local DoF tags assigned to the basis functions, in the order of their local enumeration
      ordinal_type tags[4] = { spaceDim, 0, 0, 1 };

      OrdinalTypeArray1DHost tagView(&tags[0], 4);

      this->setOrdinalTagData(this->tagToOrdinal_,
                              this->ordinalToTag_,
                              tagView,
                              this->basisCardinality_,
                              tagSize,
                              posScDim,
                              posScOrd,
                              posDfOrd);
    }
  }
}

#endif
