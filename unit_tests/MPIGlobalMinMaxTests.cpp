//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  MPIGlobalMinMaxTests
//  Camellia
//
//  Created by Nate Roberts on 7-16-20.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"
#include "MPIGlobalMinMax.h"

using namespace Camellia;

namespace
{

  std::map<GlobalIndexType,double> globalData()
  {
    std::map<GlobalIndexType,double> data;
    const int numElements = 10000;
    const int firstCellID = 3500;
    for (int cellOrdinal=0; cellOrdinal<numElements; cellOrdinal++)
    {
      int cellID = firstCellID + cellOrdinal;
      // an arbitrary formula to generate some values:
      double value = double(cellOrdinal * cellOrdinal - firstCellID * cellOrdinal);
//      double value = cellID;
      data[cellID] = value;
    }
    return data;
  }

  struct MinComparator
  {
    bool operator() (const std::pair<GlobalIndexType,double> &lhs, const std::pair<GlobalIndexType,double> &rhs) const
    {
      if ( lhs.second != rhs.second ) return (lhs.second < rhs.second);
      else                            return (lhs.first < rhs.first);
    }
  };

  struct MaxComparator
  {
    bool operator() (const std::pair<GlobalIndexType,double> &lhs, const std::pair<GlobalIndexType,double> &rhs) const
    {
      if ( lhs.second != rhs.second ) return (lhs.second > rhs.second);
      else                            return (lhs.first > rhs.first);
    }
  };

  std::vector< std::pair<GlobalIndexType,double> > leastEntries(const int numEntries, const std::map<GlobalIndexType,double> &entries)
  {
    std::vector< std::pair<GlobalIndexType,double> > sortedEntries(entries.begin(),entries.end());
    MinComparator comparator;
    std::sort(sortedEntries.begin(), sortedEntries.end(), comparator);
    sortedEntries.erase(sortedEntries.begin() + numEntries, sortedEntries.end());
    return sortedEntries;
  }

  std::vector< std::pair<GlobalIndexType,double> > greatestEntries(const int numEntries, const std::map<GlobalIndexType,double> &entries)
  {
    std::vector< std::pair<GlobalIndexType,double> > sortedEntries(entries.begin(),entries.end());
    MaxComparator comparator;
    std::sort(sortedEntries.begin(), sortedEntries.end(), comparator);
    sortedEntries.erase(sortedEntries.begin() + numEntries, sortedEntries.end());
    return sortedEntries;
  }

  std::map<GlobalIndexType,double> myData(const int myRank, const int numRanks)
  {
    const auto allData = globalData();
    int perRankElementCount = allData.size() / numRanks;
    int remainderElements = allData.size() - numRanks * perRankElementCount;
    int myElementOffset;
    int myElementCount;
    
    if (myRank < remainderElements)
    {
      // first remainderElements ranks get one extra element
      myElementCount  = perRankElementCount + 1;
      myElementOffset = myRank * (perRankElementCount + 1);
    }
    else
    {
      myElementCount  = perRankElementCount;
      myElementOffset = myRank * perRankElementCount + remainderElements;
    }
    
    std::map<GlobalIndexType,double> myEntries;
    int i=0;
    for (const auto & entry : allData)
    {
      if (i >= myElementOffset)
      {
        myEntries[entry.first] = entry.second;
      }
      i++;
      if (i >= myElementOffset + myElementCount)
      {
        break;
      }
    }
    
    // DEBUGGING
//    if (myRank == 0)
//    {
//      std::cout << "On rank 0:\n";
//      std::cout << "perRankElementCount: " << perRankElementCount << std::endl;
//      std::cout << "remainderElements: "   << remainderElements << std::endl;
//      std::cout << "myElementCount: "      << myElementCount << std::endl;
//      std::cout << "myElementOffset: "     << myElementOffset << std::endl;
//    }
    
    return myEntries;
  }

  TEUCHOS_UNIT_TEST( MPIGlobalMinMax, GlobalMaximum )
  {
    auto comm = MPIWrapper::TeuchosCommWorld();
    
    auto globalEntries = globalData();
    int numEntries = globalEntries.size() / 10;
    
    auto expectedGreatestEntries = greatestEntries(numEntries, globalEntries);
    
    auto myEntries = myData(comm->getRank(), comm->getSize());
    
    // sanity check on myData(): make sure that the sum of local entry counts equals globalEntries.size()
    int localCount = myEntries.size();
    out << "localCount: " << localCount << endl;
    int globalCount = MPIWrapper::sum(comm, localCount);
    TEST_EQUALITY(globalEntries.size(), globalCount);
    
    const int maxGatherCount = numEntries * 4;
    
    auto actualGreatestEntries = MPIGlobalMinMax::globalMaximums(comm, myEntries, numEntries, maxGatherCount);
    
    TEST_EQUALITY(expectedGreatestEntries.size(), numEntries);
    TEST_EQUALITY(actualGreatestEntries.size(),   numEntries);
    
    for (int i=0; i<numEntries; i++)
    {
      TEST_EQUALITY( expectedGreatestEntries[i].first, actualGreatestEntries[i].first );
      TEST_EQUALITY( expectedGreatestEntries[i].second, actualGreatestEntries[i].second );
    }
  }
        
  TEUCHOS_UNIT_TEST( MPIGlobalMinMax, GlobalMinimum )
  {
    auto comm = MPIWrapper::TeuchosCommWorld();
    
    auto globalEntries = globalData();
    int numEntries = globalEntries.size() / 10;
    
    auto expectedLeastEntries = leastEntries(numEntries, globalEntries);
    
    auto myEntries = myData(comm->getRank(), comm->getSize());
    
    // sanity check on myData(): make sure that the sum of local entry counts equals globalEntries.size()
    int localCount = myEntries.size();
    out << "localCount: " << localCount << endl;
    int globalCount = MPIWrapper::sum(comm, localCount);
    TEST_EQUALITY(globalEntries.size(), globalCount);
    
    const int maxGatherCount = numEntries * 4;
    
    auto actualLeastEntries = MPIGlobalMinMax::globalMinimums(comm, myEntries, numEntries, maxGatherCount);
    
    TEST_EQUALITY(expectedLeastEntries.size(), numEntries);
    TEST_EQUALITY(actualLeastEntries.size(),   numEntries);
    
    for (int i=0; i<numEntries; i++)
    {
      TEST_EQUALITY( expectedLeastEntries[i].first, actualLeastEntries[i].first );
      TEST_EQUALITY( expectedLeastEntries[i].second, actualLeastEntries[i].second );
    }
  }
} // namespace
