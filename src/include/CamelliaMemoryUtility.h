// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  CamelliaMemoryUtility.h
//  Camellia
//
//  Created by Nate Roberts on 6/28/16.
//
//

#ifndef Camellia_CamelliaMemoryUtility_h
#define Camellia_CamelliaMemoryUtility_h

namespace Camellia {
  const static int MAP_NODE_OVERHEAD = 32;  // according to http://info.prelert.com/blog/stl-container-memory-usage, this appears to be basically universal
  
  // LLVM memory approximations come from http://info.prelert.com/blog/stl-container-memory-usage
  template<typename A, typename B>
  long long approximateMapSizeLLVM(const std::map<A,B> &someMap)   // in bytes
  {
    // 24 bytes for the map itself; nodes are 32 bytes + sizeof(pair<A,B>) each
    // if A and B are containers, this won't count their contents...
    
    std::map<int, int> emptyMap;
    
    int MAP_OVERHEAD = sizeof(emptyMap);
    
    return MAP_OVERHEAD + (MAP_NODE_OVERHEAD + sizeof(pair<A,B>)) * someMap.size();
  }
  
  template<typename A>
  long long approximateSetSizeLLVM(const std::set<A> &someSet)   // in bytes
  {
    // 48 bytes for the set itself; nodes are 32 bytes + sizeof(pair<A,B>) each
    // if A and B are containers, this won't count their contents...
    
    std::set<int> emptySet;
    int SET_OVERHEAD = sizeof(emptySet);
    
    return SET_OVERHEAD + (MAP_NODE_OVERHEAD + sizeof(A)) * someSet.size();
  }
  
  template<typename A>
  long long approximateVectorSizeLLVM(const std::vector<A> &someVector)   // in bytes
  {
    // 24 bytes for the vector itself; entries are sizeof(A) each
    // if A is a container, this won't count its contents...
    std::vector<int> emptyVector;
    int VECTOR_OVERHEAD = sizeof(someVector);
    
    return VECTOR_OVERHEAD + sizeof(A) * someVector.size();
  }
}

#endif
