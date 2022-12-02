//
//  SharedScratchStorageDynRankView.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/28/20.
//

#ifndef SharedScratchStorageDynRankView_h
#define SharedScratchStorageDynRankView_h

#include <map>
#include <vector>

#include "TypeDefs.h"

#include <Intrepid2_Data.hpp>
#include <Intrepid2_Utils.hpp>

namespace Camellia {
  template<typename Scalar>
  class SharedScratchStorageDynRankView
  {
    long _hitCount = 0; // number of times we reuse a device view
  public:
    using DynRankView = Camellia::DeviceView<Scalar>;
    using HostView    = Camellia::HostView<Scalar>;
        
    using ScratchStorageDimensions = std::vector<int>;
    using ScratchStorageEntry = std::tuple<DynRankView, bool, std::string>; // true means checked out right now
    using ScratchStorageCache = std::map<ScratchStorageDimensions, std::vector<ScratchStorageEntry> >;
    
    DynRankView checkoutScratchStorage(const ScratchStorageDimensions &dims, std::string description="");
    DynRankView checkoutScratchStorageMatchingDynRankView(DynRankView &deviceView, std::string description="");
    DynRankView checkoutScratchStorageMatchingHostView(HostView &hostView, std::string description="");
    
    void checkinScratchStorage(DynRankView &scratchView);
    
    void clear();
    
    long hitCount();
    
    static SharedScratchStorageDynRankView<Scalar> & sharedScratchStorage(); // returns singleton instance
    
    template<typename ViewType>
    static ScratchStorageDimensions viewDimensions(ViewType &view)
    {
      ScratchStorageDimensions dim(view.rank());
      const int viewRank = view.rank();
      for (int i=0; i<viewRank; i++)
      {
        dim[i] = view.extent_int(i);
      }
      return dim;
    }
  private:
    DynRankView allocateView(const ScratchStorageDimensions &dims, const std::string &label) const;
    
    ScratchStorageCache _scratchStorageCache;
  };
  void clearSharedScratchStorageDynRankView();
} // namespace Camellia
#endif /* SharedScratchStorageDynRankView_h */
