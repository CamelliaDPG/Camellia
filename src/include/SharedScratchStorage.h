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

#include <Intrepid2_Data.hpp>
#include <Intrepid2_Utils.hpp>

#include "SharedScratchStorageDynRankView.h"
#include "SharedScratchStorageFixedRankView.h"

namespace Camellia {

  template<typename Scalar>
  class SharedScratchStorage
  {
    SharedScratchStorageFixedRankView<Scalar, 1> _fixedRank1;
    SharedScratchStorageFixedRankView<Scalar, 2> _fixedRank2;
    SharedScratchStorageFixedRankView<Scalar, 3> _fixedRank3;
    SharedScratchStorageFixedRankView<Scalar, 4> _fixedRank4;
    SharedScratchStorageFixedRankView<Scalar, 5> _fixedRank5;
    SharedScratchStorageFixedRankView<Scalar, 6> _fixedRank6;
    SharedScratchStorageFixedRankView<Scalar, 7> _fixedRank7;
    
    SharedScratchStorageDynRankView<Scalar> _dynRank;
  public:
    using DeviceView = Camellia::DeviceView<Scalar>;
    using HostView   = Camellia::HostView<Scalar>;
    
    using Data = Intrepid2::Data<Scalar, DeviceType>;
    
    using ScratchStorageDimensions = std::vector<int>;
    using ScratchStorageEntry = std::tuple<DeviceView, bool, std::string>; // true means checked out right now
    using ScratchStorageCache = std::map<ScratchStorageDimensions, std::vector<ScratchStorageEntry> >;
    
    DeviceView checkoutScratchStorage(const ScratchStorageDimensions &dims, std::string description="");
    DeviceView checkoutScratchStorageMatchingDeviceView(DeviceView &deviceView, std::string description="");
    DeviceView checkoutScratchStorageMatchingHostView(HostView &hostView, std::string description="");
    
    DeviceView checkoutScratchStorage(FunctionPtr f, BasisCachePtr basisCache, std::string description="");
    
    using ScratchDataDimensions = std::vector<Intrepid2::DimensionInfo>;
    Data checkoutScratchData(const ScratchDataDimensions &dataDims, std::string description="");
    void checkinScratchData(const Data &data);
    
    void checkinScratchStorage(DeviceView &scratchView);
    
    void clear();
    
    long hitCount();
    
    static SharedScratchStorage & sharedScratchStorage(); // returns singleton instance
    static Teuchos::RCP<SharedScratchStorage> sharedScratchStoragePtr(); // returns rcp to singleton instance
  };
  
  void clearSharedScratchStorage(); // should be called during finalize()
  
} // namespace Camellia
#endif /* SharedScratchStorage_h */
