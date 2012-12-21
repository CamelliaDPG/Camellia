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

// static class to provide a FieldContainer-based interface to some common MPI tasks
// (Can be used even with MPI disabled)
class MPIWrapper {
public:
  // sum the contents of inValues across all processors, and stores the result in outValues
  // the rank of outValues determines the nature of the sum:
  // if outValues has dimensions (D1,D2,D3), say, then inValues must agree in the first three dimensions,
  // but may be of arbitrary shape beyond that.  All values on all processors with matching address
  // (d1,d2,d3) will be summed and stored in outValues(d1,d2,d3).
//  static void elementWiseSum(FieldContainer<double> &outValues, const FieldContainer<double> &inValues);

  static void elementWiseSum(FieldContainer<double> &values); // sums values element-wise across all processors
  // sum the contents of valuesToSum across all processors, and returns the result:
  // (valuesToSum may vary in length across processors)
  static double sum(const FieldContainer<double> &valuesToSum);
};

#endif /* defined(__Camellia_debug__MPIWrapper__) */
