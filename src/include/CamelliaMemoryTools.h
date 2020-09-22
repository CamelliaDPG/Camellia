//
//  CamelliaMemoryTools.h
//  Camellia
//
//  Created by Roberts, Nathan V on 8/17/20.
//

#ifndef CamelliaMemoryTools_h
#define CamelliaMemoryTools_h

#include <sys/resource.h>

namespace Camellia
{

  inline double memoryHighWaterMarkInMegabytes()
  {
    struct rusage sys_resources;
    getrusage(RUSAGE_SELF, &sys_resources);
    
    // sys_resources.ru_maxrss is memory high-water mark in KB; divide by 1024 for MB.
    double usage_in_MB = double(sys_resources.ru_maxrss)/1024.0;
    
    // MacOS is off by a factor of 1024 (appears to report in bytes)
  #if __APPLE__
    usage_in_MB /= 1024.0;
  #endif
    return usage_in_MB;
  }

  inline double memoryHighWaterMarkInGigabytes()
  {
    return memoryHighWaterMarkInMegabytes() / 1024.0;
  }
}

#endif /* CamelliaMemoryTools_h */
