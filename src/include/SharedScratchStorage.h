//
//  SharedScratchStorage.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/28/20.
//

#ifndef SharedScratchStorage_h
#define SharedScratchStorage_h

#include <map>
#include <vector>

#include "TypeDefs.h"

namespace Camellia {
  template<typename Scalar>
  class SharedScratchStorage
  {
    long _hitCount = 0; // number of times we reuse a device view
    
  public:
    using DeviceView = Camellia::DeviceView<Scalar>;
    using HostView   = Camellia::HostView<Scalar>;
    
    using ScratchStorageDimensions = std::vector<int>;
    using ScratchStorageEntry = std::tuple<DeviceView, bool, std::string>; // true means checked out right now
    using ScratchStorageCache = std::map<ScratchStorageDimensions, std::vector<ScratchStorageEntry> >;
    
    DeviceView checkoutScratchStorage(const ScratchStorageDimensions &dims, std::string description="");
    DeviceView checkoutScratchStorageMatchingDeviceView(DeviceView &deviceView, std::string description="");
    DeviceView checkoutScratchStorageMatchingHostView(HostView &hostView, std::string description="");
    
    void checkinScratchStorage(DeviceView &scratchView);
    
    void clear();
    
    long hitCount();
    
    static SharedScratchStorage & sharedScratchStorage(); // returns singleton instance
    
    template<typename ViewType>
    static ScratchStorageDimensions viewDimensions(ViewType &view)
    {
      ScratchStorageDimensions dim(view.rank());
      for (int i=0; i<view.rank(); i++)
      {
        dim[i] = view.extent_int(i);
      }
      return dim;
    }
  private:
    DeviceView allocateView(const ScratchStorageDimensions &dims, const std::string &label) const;
    
    ScratchStorageCache _scratchStorageCache;
  };
  
  void clearSharedScratchStorage(); // should be called during finalize()
  
} // namespace Camellia
#endif /* SharedScratchStorage_h */
