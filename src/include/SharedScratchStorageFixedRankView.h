//
//  SharedScratchStorageFixedRankView.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/28/20.
//

#ifndef SharedScratchStorageFixedRankView_h
#define SharedScratchStorageFixedRankView_h

#include <map>
#include <vector>

#include "TypeDefs.h"

#include <Intrepid2_Data.hpp>
#include <Intrepid2_Utils.hpp>

namespace Camellia {
  template<typename Scalar, int rank>
  class SharedScratchStorageFixedRankView
  {
    long _hitCount = 0; // number of times we reuse a device view
  public:
    using ScalarPointer = typename Intrepid2::RankExpander<Scalar,rank>::value_type; // Scalar* for rank 1, Scalar** for rank 2, etc.
  
    using FixedView = Kokkos::View<ScalarPointer,::Camellia::DeviceType>;
    using HostView   = Camellia::HostView<Scalar>;
        
    using ScratchStorageDimensions = std::vector<int>;
    using ScratchStorageEntry = std::tuple<FixedView, bool, std::string>; // true means checked out right now
    using ScratchStorageCache = std::map<ScratchStorageDimensions, std::vector<ScratchStorageEntry> >;
    
    FixedView checkoutScratchStorage(const ScratchStorageDimensions &dims, std::string description="");
    FixedView checkoutScratchStorageMatchingFixedView(const FixedView &deviceView, std::string description="");
    FixedView checkoutScratchStorageMatchingHostView(const HostView &hostView, std::string description="");
    
    void checkinScratchStorage(const FixedView &scratchView);
    
    void clear();
    
    long hitCount();
        
    template<typename ViewType>
    static ScratchStorageDimensions viewDimensions(ViewType &view)
    {
      ScratchStorageDimensions dim(rank);
      for (int i=0; i<rank; i++)
      {
        dim[i] = view.extent_int(i);
      }
      return dim;
    }
  private:
    FixedView allocateView(const ScratchStorageDimensions &dims, const std::string &label) const;
    
    ScratchStorageCache _scratchStorageCache;
  };
} // namespace Camellia
#endif /* SharedScratchStorageFixedRankView_h */
