// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
// ***********************************************************************
//
//                  Camellia Cubature Factory:
//
//  A wrapper around Intrepid's DefaultCubatureFactory that adds support
//  for Camellia's tensor-product topologies.
//
// ***********************************************************************
//@HEADER

#ifndef CAMELLIA_CUBATUREFACTORY_H
#define CAMELLIA_CUBATUREFACTORY_H

//! CubatureFactory: wrapper around Intrepid's DefaultCubatureFactory with support for Camellia's CellTopology.
//
/*!

 \author Nathan V. Roberts, ALCF.

 \date Last modified on 11-Dec-2014.
 */

// Intrepid includes:
#include "Intrepid2_DefaultCubatureFactory.hpp"

// Camellia includes:
#include "CellTopology.h"

namespace Camellia
{

class CubatureFactory
{
  using Cubature = Intrepid2::Cubature<DeviceType,double,double>;
public:

  //@{ Constructors/Destructors
  CubatureFactory() {}
  ~CubatureFactory() {}
  //@}

  //@{ \name Cubature creation methods

  //! Creates cubature object with identical cubature degree in all dimensions
  /*!
    \param
    cellTopo  - (In) The Camellia CellTopology on which cubature is desired.
    \param
    cubDegree - (In) The cubature degree required.

    \return Cubature object.
    */
  Teuchos::RCP<Cubature> create(CellTopoPtr cellTopo, int cubDegree);

  //! Creates cubature object cubature degree that may vary according to dimension
  /*!
   \param
   cellTopo - (In) The Camellia CellTopology on which cubature is desired.
   \param
   cubDegree - (In) The cubature degree required in each dimension.  The size of this object may be equal either to the tensorial degree of the cellTopo or, for a cellTopo whose shards CellTopology is itself a hypercube topology, the spatial dimension of the cellTopo.

   \return Cubature object.
   */
  Teuchos::RCP<Cubature> create(CellTopoPtr cellTopo, std::vector<int> cubDegree);
  //@}
}; // class CubatureFactory

} // namespace Camellia
#endif // CAMELLIA_CUBATUREFACTORY_H
