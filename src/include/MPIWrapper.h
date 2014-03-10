//
//  MPIWrapper.h
//  Camellia
//
//  Created by Nathan Roberts on 12/21/12.
//
//

#ifndef __Camellia__MPIWrapper__
#define __Camellia__MPIWrapper__

#include <iostream>

#include "Intrepid_FieldContainer.hpp"
using namespace Intrepid;

#include "IndexType.h"

// static class to provide a FieldContainer-based interface to some common MPI tasks
// (Can be used even with MPI disabled)
class MPIWrapper {
public:
  // sum the contents of inValues across all processors, and stores the result in outValues
  // the rank of outValues determines the nature of the sum:
  // if outValues has dimensions (D1,D2,D3), say, then inValues must agree in the first three dimensions,
  // but may be of arbitrary shape beyond that.  All values on all processors with matching address
  // (d1,d2,d3) will be summed and stored in outValues(d1,d2,d3).
//  static void entryWiseSum(FieldContainer<double> &outValues, const FieldContainer<double> &inValues);

  static void allGather(FieldContainer<int> &allValues, int myValue);
  static void allGather(FieldContainer<int> &values, FieldContainer<int> &myValues);
  
  static int rank();
  
  static void entryWiseSum(FieldContainer<double> &values); // sums values entry-wise across all processors
  // sum the contents of valuesToSum across all processors, and returns the result:
  // (valuesToSum may vary in length across processors)
  static double sum(const FieldContainer<double> &valuesToSum);
  static double sum(double myValue);
  
  static void entryWiseSum(FieldContainer<int> &values); // sums values entry-wise across all processors
  // sum the contents of valuesToSum across all processors, and returns the result:
  // (valuesToSum may vary in length across processors)
  static int sum(const FieldContainer<int> &valuesToSum);
  static int sum(int myValue);
  
  static void entryWiseSum(FieldContainer<GlobalIndexType> &values); // sums values entry-wise across all processors
  // sum the contents of valuesToSum across all processors, and returns the result:
  // (valuesToSum may vary in length across processors)
  static GlobalIndexType sum(const FieldContainer<GlobalIndexType> &valuesToSum);
  static GlobalIndexType sum(GlobalIndexType myValue);
};

#endif /* defined(__Camellia_debug__MPIWrapper__) */
