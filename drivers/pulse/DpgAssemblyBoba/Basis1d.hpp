#ifndef BASIS_1D_HPP
#define BASIS_1D_HPP

#include "Camellia.h"

using Basis1d = Camellia::Basis<double, Intrepid::FieldContainer<double,0>>;

using Basis1dPtr = Camellia::BasisPtr;

inline Basis1dPtr newBasis1d(Camellia::Space space, std::size_t h1Order) {
  static auto cellTopo = Camellia::CellTopology::line();
  static auto basisFamily = Camellia::BasisFamily::nodalBasisFamily();
  return basisFamily->getBasis(cellTopo, Camellia::efsForSpace(space), h1Order);
}

#endif
