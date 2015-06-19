//
//  MPIWrapper.h
//  Camellia
//
//  Created by Nathan Roberts on 12/21/12.
//
//

#ifndef __Camellia__MPIWrapper__
#define __Camellia__MPIWrapper__

#include "TypeDefs.h"

#include <iostream>

// MPI includes
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "Intrepid_FieldContainer.hpp"

namespace Camellia
{
// static class to provide a FieldContainer-based interface to some common MPI tasks
// (Can be used even with MPI disabled)
class MPIWrapper
{
private:
  template<typename Scalar>
  static void allGatherCompact(Intrepid::FieldContainer<Scalar> &gatheredValues,
                               Intrepid::FieldContainer<Scalar> &myValues,
                               Intrepid::FieldContainer<int> &offsets);
public:
  // sum the contents of inValues across all processors, and stores the result in outValues
  // the rank of outValues determines the nature of the sum:
  // if outValues has dimensions (D1,D2,D3), say, then inValues must agree in the first three dimensions,
  // but may be of arbitrary shape beyond that.  All values on all processors with matching address
  // (d1,d2,d3) will be summed and stored in outValues(d1,d2,d3).
  //  static void entryWiseSum(FieldContainer<double> &outValues, const FieldContainer<double> &inValues);

  static void allGather(Intrepid::FieldContainer<int> &allValues, int myValue);
  static void allGatherHomogeneous(Intrepid::FieldContainer<int> &values, Intrepid::FieldContainer<int> &myValues); // assumes myValues is same size on every proc.

  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  //        Not necessarily super-efficient in terms of communication, but avoids allocating a big array like allGatherHomogeneous would.
  static void allGatherCompact(Intrepid::FieldContainer<int> &gatheredValues,
                               Intrepid::FieldContainer<int> &myValues,
                               Intrepid::FieldContainer<int> &offsets);

  // \brief Resizes gatheredValues to be the size of the sum of the myValues containers, and fills it with the values from those containers.
  //        Not necessarily super-efficient in terms of communication, but avoids allocating a big array like allGatherHomogeneous would.
  static void allGatherCompact(Intrepid::FieldContainer<double> &gatheredValues,
                               Intrepid::FieldContainer<double> &myValues,
                               Intrepid::FieldContainer<int> &offsets);

  static int rank();

  //! sum values entry-wise across all processors in Comm
  template<typename ScalarType>
  static void entryWiseSum(const Epetra_Comm &Comm, Intrepid::FieldContainer<ScalarType> &values);
  
  //! sum values entry-wise across all processors
  template<typename ScalarType>
  static void entryWiseSum(Intrepid::FieldContainer<ScalarType> &values);
  
  static void entryWiseSum(const Epetra_Comm &Comm, Intrepid::FieldContainer<double> &values); // sums values entry-wise across all processors
  
  static void entryWiseSum(Intrepid::FieldContainer<double> &values); // sums values entry-wise across all processors
  // sum the contents of valuesToSum across all processors, and returns the result:
  // (valuesToSum may vary in length across processors)
  static double sum(const Intrepid::FieldContainer<double> &valuesToSum);
  static double sum(double myValue);

  static void entryWiseSum(Intrepid::FieldContainer<int> &values); // sums values entry-wise across all processors
  // sum the contents of valuesToSum across all processors, and returns the result:
  // (valuesToSum may vary in length across processors)
  static int sum(const Intrepid::FieldContainer<int> &valuesToSum);
  static int sum(int myValue);

  static void entryWiseSum(Intrepid::FieldContainer<GlobalIndexType> &values); // sums values entry-wise across all processors
  // sum the contents of valuesToSum across all processors, and returns the result:
  // (valuesToSum may vary in length across processors)
  static GlobalIndexType sum(const Intrepid::FieldContainer<GlobalIndexType> &valuesToSum);
  static GlobalIndexType sum(GlobalIndexType myValue);
};
}

#endif /* defined(__Camellia_debug__MPIWrapper__) */
