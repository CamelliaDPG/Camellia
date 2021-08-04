//
//  PhysicalTensorPoints.h
//  Camellia
//
//  Created by Roberts, Nathan V on 5/1/21.
//

#ifndef PhysicalTensorPoints_h
#define PhysicalTensorPoints_h

#include <Intrepid2_Data.hpp>
#include <Intrepid2_TensorPoints.hpp>

namespace Camellia {
  class PhysicalPointComponent;
  class PhysicalPointComponent1D;
  /**
   \class PhysicalTensorPoints
   With operator() interface, presents shape: (C,P,D), where the number of point dimensions corresponds to the number of tensorial components present in each cell (for hypercubes, this is the number of diagonal blocks within the Jacobian).
   
   Also presents an accessComponent() interface, with shape (C,T,PT,DT), where T is the tensorial component number, PT is the point ordinal within the tensorial component, and DT is the relative spatial dimension within the tensorial component.
   
   For example, suppose we have a triangle that is orthogonally extruded into the third dimension.  We can express the physical quadrature points as a tensor product of quadrature on the triangle and the line.
   Suppose that we have 6 quadrature points for the triangle, and 2 for the line, a total of 12 3D quadrature points.  The (C,P,D) shape for operator() has a P extent of 12, one for each 3D point, and a D extent of 3 (for all three dimensions).
   The (C,T,PT,DT) has T extent 2, for the two tensorial components.
   The PT extent depends on the T argument; it is 6 for T=0, and 2 for T=1.
   The DT extent also depends on the T argument; for T=0, the DT extent is 2 (two dimensions for the triangle); for T=1, the DT extent is 1 (for the one dimension of the line).
   The product of all the PT extents is P; the sum of all DT extents is D.
   
   The initial implementation simply uses a standard, expanded physical points container.  This could be made more memory-efficient by introducing a special representation for 1D physical points; for these, we need just
   two scalars to specify the scaling and shifting relative to reference space.  (This initial implementation is chosen for its simplicity.)
   */
  class PhysicalTensorPoints
  {
    using PointsView = Intrepid2::ScalarView<double,Camellia::DeviceType>;
    using ordinal_type = Intrepid2::ordinal_type;
    static constexpr auto MaxTensorComponents = Intrepid2::Parameters::MaxTensorComponents;
    
    Kokkos::Array<PointsView,MaxTensorComponents> tensorComponents_; // each with shape (C,PT,DT)
    Kokkos::Array<int,3> extents_ {0,0,0}; // (C,P,D)
    
    int numTensorComponents_ = 0; // this says how many of the slots in _tensorComponents (a fixed-size array) are filled
    
    Kokkos::Array<ordinal_type, MaxTensorComponents> dimToComponent_;
    Kokkos::Array<ordinal_type, MaxTensorComponents> dimToComponentDim_;
    Kokkos::Array<ordinal_type, MaxTensorComponents> pointModulus_;
    Kokkos::Array<ordinal_type, MaxTensorComponents> pointDivisor_;
    
    /**
     \brief Initialize members based on constructor parameters.
    */
    void initialize()
    {
      INTREPID2_TEST_FOR_EXCEPTION(numTensorComponents_ > MaxTensorComponents, std::invalid_argument, "Can't have more than MaxTensorComponents components");
      
      int cellCount = tensorComponents_[0].extent_int(0);
      
      int totalPointCount = 1;
      ordinal_type pointDivisor = 1;
      
      ordinal_type d=0;
      ordinal_type dimsSoFar = 0;
      for (ordinal_type r=0; r<numTensorComponents_; r++)
      {
        const int componentCellCount  = tensorComponents_[r].extent_int(0);
        const int componentPointCount = tensorComponents_[r].extent_int(1);
        const int componentDim        = tensorComponents_[r].extent_int(2);
        
        INTREPID2_TEST_FOR_EXCEPTION(componentCellCount != cellCount, std::invalid_argument, "All components must agree in their first (cell) dimension.");
        
        for (int i=0; i<componentDim; i++)
        {
          dimToComponent_[d]    = r;
          dimToComponentDim_[d] = d - dimsSoFar;
          d++;
        }
        
        totalPointCount *= componentPointCount;
        dimsSoFar       += componentDim;
        
        pointModulus_[r] = componentPointCount;
        pointDivisor_[r] = pointDivisor;
        pointDivisor *= componentPointCount;
      }
      extents_[0] = cellCount;
      extents_[1] = totalPointCount;
      extents_[2] = dimsSoFar;
    }
  public:
    PhysicalTensorPoints(std::vector<PointsView> tensorComponents = std::vector<PointsView>())
    :
    numTensorComponents_(tensorComponents.size())
    {
      int cellExtent = (tensorComponents.size() > 0) ? tensorComponents[0].extent_int(0) : 0;
      for (int r=0; r<tensorComponents.size(); r++)
      {
        tensorComponents_[r] = tensorComponents[r];
      }
      initialize();
    }
    
    KOKKOS_INLINE_FUNCTION double
    operator()(const int& cell_ordinal, const int& point_ordinal, const int& dim) const {
      const ordinal_type component = dimToComponent_[dim];
      const ordinal_type d = dimToComponentDim_[dim];
      const ordinal_type componentPointOrdinal = (point_ordinal / pointDivisor_[component]) % pointModulus_[component];
      return tensorComponents_[component](cell_ordinal,componentPointOrdinal,d);
    }
    
    KOKKOS_INLINE_FUNCTION
    int extent_int(const int& d) const {
      if (d < 3)
      {
        return extents_[d];
      }
      else
      {
        return 1;
      }
    }
    
    const PointsView & getTensorComponent(const int & r) const
    {
      return tensorComponents_[r];
    }
    
    KOKKOS_INLINE_FUNCTION
    const int & numTensorComponents() const
    {
      return numTensorComponents_;
    }
    
    //! Return the rank of the container, which is 3.
    KOKKOS_INLINE_FUNCTION
    constexpr ordinal_type rank() const
    {
      return 3; // shape is (C,P,D)
    }
    
    KOKKOS_INLINE_FUNCTION bool isValid() const
    {
      return numTensorComponents_ > 0;
    }
  };
}

#endif /* PhysicalTensorPoints_h */
