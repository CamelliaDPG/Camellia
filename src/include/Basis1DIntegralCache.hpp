//
//  Basis1DIntegralCache.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 1/28/25.
//

#ifndef Basis1DIntegralCache_hpp
#define Basis1DIntegralCache_hpp

#include <Intrepid2_Basis.hpp>

#include <TypeDefs.h>

namespace Camellia
{

template <typename Scalar>
class Basis1DIntegralCache
{
  using     ViewType = Kokkos::View<Scalar*, Camellia::DeviceType>;
  using HostViewType = Kokkos::View<Scalar*, Kokkos::HostSpace>;
  using       View2D = Kokkos::View<Scalar**, Camellia::DeviceType>;
  using   HostView2D = typename View2D::HostMirror;
  
  using BasisKey = std::tuple<std::string,bool,int>; // bool: whether to differentiate basis; int: polynomial degree, as reported by basis
  using RefIntegralKey = std::tuple<std::vector<BasisKey>,int,bool>; // int: the power of x degree to integrate against bases; bool: true indicates LayoutLeft (first basis function has contiguous memory); false LayoutRight
  std::map<RefIntegralKey,ViewType> _refSpaceIntegrals;
  
  View2D computeBasisValues(BasisPtr basis, bool differentiate, Intrepid2::TensorPoints<double,DeviceType> points);
public:
  ViewType getReferenceSpaceIntegral(BasisPtr basis1, bool differentiateBasis1,
                                     BasisPtr basis2, bool differentiateBasis2,
                                     int monomialWeightDegree = 0, int quadratureDegree = -1, bool layoutLeft = true);
  
  ViewType getReferenceSpaceIntegral(BasisPtr basis1, bool differentiateBasis1,
                                     BasisPtr basis2, bool differentiateBasis2,
                                     BasisPtr basis3, bool differentiateBasis3,
                                     int monomialWeightDegree = 0, int quadratureDegree = -1, bool layoutLeft = true);
  
  ViewType getReferenceSpaceIntegral(BasisPtr basis1, bool differentiateBasis1,
                                     BasisPtr basis2, bool differentiateBasis2,
                                     BasisPtr basis3, bool differentiateBasis3,
                                     BasisPtr basis4, bool differentiateBasis4,
                                     int monomialWeightDegree = 0, int quadratureDegree = -1, bool layoutLeft = true);
};

}

#endif /* Basis1DIntegralCache_hpp */
