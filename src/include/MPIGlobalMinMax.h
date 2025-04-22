//
//  MPIGlobalMinMax.hpp
//  Camellia
//
//  Created by Roberts, Nathan V on 7/15/20.
//

#ifndef MPIGlobalMinMax_hpp
#define MPIGlobalMinMax_hpp

#include <map>

#include "MPIWrapper.h"
#include "TypeDefs.h"

namespace Camellia
{
  class MPIGlobalMinMax
  {
    // comparator for putting the least elements first
    template<class less_or_greater >
    struct Comparator
    {
      bool operator() (const std::pair<GlobalIndexType,double> &lhs, const std::pair<GlobalIndexType,double> &rhs) const
      {
        less_or_greater comparator;
        
        if ( lhs.second != rhs.second )
        {
          return comparator(lhs.second, rhs.second);
        }
        else
        {
          return comparator(lhs.first, rhs.first);
        }
      }
    };
    
    template<class less_or_greater >
    static std::vector< std::pair<GlobalIndexType,double> > globalMinOrMax(Epetra_CommPtr Comm, const std::map<GlobalIndexType,double> &localEntries, const int numValues, const int maxGatherCount)
      {
        Comparator<less_or_greater> entry_comparator;
        less_or_greater double_comparator;
        
        const bool computingMinimum = double_comparator(0,1);

        TEUCHOS_TEST_FOR_EXCEPTION(maxGatherCount < numValues, std::invalid_argument, "maxGatherCount must be at least as large as numValues requested");
        
        int localEntryCount = localEntries.size();
        int globalEntryCount = 0;
        Comm->SumAll(&localEntryCount, &globalEntryCount, 1);
        
        std::vector< std::pair<GlobalIndexType,double> > localData;
        int localMatchCount;
        int globalMatchCount;
        
        if (globalEntryCount <= maxGatherCount)
        {
          // then we can skip the thresholding; just gather everything
          localMatchCount = localEntryCount;
          globalMatchCount = globalEntryCount;
          localData = std::vector< std::pair<GlobalIndexType,double> >(localEntries.begin(), localEntries.end());
        }
        else //if (globalEntryCount > maxGatherCount)
        {
          std::map<double,std::vector<GlobalIndexType>, less_or_greater> reverseLocalMap;
          double minLocal = std::numeric_limits<double>::max();
          double maxLocal = std::numeric_limits<double>::lowest();
          for (const auto & entry : localEntries)
          {
            reverseLocalMap[entry.second].push_back(entry.first);
            maxLocal = std::fmax(maxLocal, entry.first);
            minLocal = std::fmin(minLocal, entry.first);
          }
          double localThreshold = std::numeric_limits<double>::lowest();
          int matchesThusFar = 0;
          for (const auto & entry : reverseLocalMap)
          {
            localThreshold = entry.first;
            matchesThusFar += entry.second.size();
            if (matchesThusFar >= numValues)
            {
              break;
            }
          }
                  
          double globalMax = maxLocal;
          double globalMin = minLocal;
          Comm->MaxAll(&maxLocal, &globalMax, 1);
          Comm->MinAll(&minLocal, &globalMin, 1);
          
          double globalThreshold = localThreshold;
          if (computingMinimum)
          {
            Comm->MinAll(&localThreshold, &globalThreshold, 1);
          }
          else
          {
            Comm->MaxAll(&localThreshold, &globalThreshold, 1);
          }
        
          // keep track of global thresholds that yield too many/too few matches
          double tooMany = computingMinimum ? globalMax : globalMin;
          double tooFew  = computingMinimum ? globalMin : globalMax;
          
          int maxCyclesAllowed = 30;
          int numCycles = 0;
          while (true)
          {
            // find local match count for current global threshold
            localMatchCount = 0;
            for (const auto & entry : reverseLocalMap)
            {
              if (double_comparator(entry.first, globalThreshold) || (entry.first == globalThreshold))
              {
                localMatchCount += entry.second.size();
              }
              else
              {
                break;
              }
            }
            
            // determine global match count
            globalMatchCount = 0;
            Comm->SumAll(&localMatchCount, &globalMatchCount, 1);
            
            if ((globalMatchCount <= maxGatherCount) && (globalMatchCount >= numValues))
            {
              // then we're ready to do the all gather
              break;
            }
            else if (globalMatchCount > maxGatherCount)
            {
              // threshold is too permissive
              tooMany = globalThreshold;
            }
            else // globalMatchCount < numValues
            {
              // threshold is not permissive enough
              tooFew = globalThreshold;
            }
            globalThreshold = (tooMany + tooFew) / 2.0;
            if (++numCycles > maxCyclesAllowed)
            {
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "max cycles exceeded");
            }
          }
          
          localData = std::vector< std::pair<GlobalIndexType,double> >(localMatchCount);
          auto entryPointer = reverseLocalMap.begin();
          for (int i=0; i<localMatchCount;)
          {
            const auto &value = entryPointer->first;
            const auto &matchingCellIDs = entryPointer->second;
            for (const auto &cellID : matchingCellIDs)
            {
              localData[i] = std::pair<GlobalIndexType,double>{cellID,value};
              i++;
            }
            entryPointer++;
          }
        }
        
        std::vector<int> offsets;
        std::vector< std::pair<GlobalIndexType,double> > gatheredData(globalMatchCount);

        MPIWrapper::allGatherVariable(*Comm, gatheredData, localData, offsets);
        
        std::sort( gatheredData.begin(), gatheredData.end(), entry_comparator );
        
        if (globalMatchCount > numValues)
        {
          // delete everything after the first numValues
          gatheredData.erase( gatheredData.begin() + numValues, gatheredData.end() );
        }
        
        return gatheredData;
      }
    
    template<class less_or_greater >
    static std::vector< std::pair<GlobalIndexType,double> > globalMinOrMax(Teuchos_CommPtr Comm, const std::map<GlobalIndexType,double> &localEntries, const int numValues, const int maxGatherCount)
      {
        Comparator<less_or_greater> entry_comparator;
        less_or_greater double_comparator;
        
        const bool computingMinimum = double_comparator(0,1);

        TEUCHOS_TEST_FOR_EXCEPTION(maxGatherCount < numValues, std::invalid_argument, "maxGatherCount must be at least as large as numValues requested");
        
        int localEntryCount = localEntries.size();
        int globalEntryCount = MPIWrapper::sum(Comm, localEntryCount);
        
        std::vector< std::pair<GlobalIndexType,double> > localData;
        int localMatchCount;
        int globalMatchCount;
        
        if (globalEntryCount <= maxGatherCount)
        {
          // then we can skip the thresholding; just gather everything
          localMatchCount = localEntryCount;
          globalMatchCount = globalEntryCount;
          localData = std::vector< std::pair<GlobalIndexType,double> >(localEntries.begin(), localEntries.end());
        }
        else //if (globalEntryCount > maxGatherCount)
        {
          std::map<double,std::vector<GlobalIndexType>, less_or_greater> reverseLocalMap;
          double minLocal = std::numeric_limits<double>::max();
          double maxLocal = std::numeric_limits<double>::lowest();
          for (const auto & entry : localEntries)
          {
            reverseLocalMap[entry.second].push_back(entry.first);
            maxLocal = std::fmax(maxLocal, entry.first);
            minLocal = std::fmin(minLocal, entry.first);
          }
          double localThreshold = std::numeric_limits<double>::lowest();
          int matchesThusFar = 0;
          for (const auto & entry : reverseLocalMap)
          {
            localThreshold = entry.first;
            matchesThusFar += entry.second.size();
            if (matchesThusFar >= numValues)
            {
              break;
            }
          }
                  
          double globalMin = MPIWrapper::min(Comm, minLocal);
          double globalMax = MPIWrapper::max(Comm, maxLocal);
          
          double globalThreshold = localThreshold;
          if (computingMinimum)
          {
            globalThreshold = MPIWrapper::min(Comm, localThreshold);
          }
          else
          {
            globalThreshold = MPIWrapper::max(Comm, localThreshold);
          }
        
          // keep track of global thresholds that yield too many/too few matches
          double tooMany = computingMinimum ? globalMax : globalMin;
          double tooFew  = computingMinimum ? globalMin : globalMax;
          
          int maxCyclesAllowed = 30;
          int numCycles = 0;
          while (true)
          {
            // find local match count for current global threshold
            localMatchCount = 0;
            for (const auto & entry : reverseLocalMap)
            {
              if (double_comparator(entry.first, globalThreshold) || (entry.first == globalThreshold))
              {
                localMatchCount += entry.second.size();
              }
              else
              {
                break;
              }
            }
            
            // determine global match count
            globalMatchCount = MPIWrapper::sum(Comm, localMatchCount);
            
            if ((globalMatchCount <= maxGatherCount) && (globalMatchCount >= numValues))
            {
              // then we're ready to do the all gather
              break;
            }
            else if (globalMatchCount > maxGatherCount)
            {
              // threshold is too permissive
              tooMany = globalThreshold;
            }
            else // globalMatchCount < numValues
            {
              // threshold is not permissive enough
              tooFew = globalThreshold;
            }
            globalThreshold = (tooMany + tooFew) / 2.0;
            if (++numCycles > maxCyclesAllowed)
            {
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "max cycles exceeded");
            }
          }
          
          localData = std::vector< std::pair<GlobalIndexType,double> >(localMatchCount);
          auto entryPointer = reverseLocalMap.begin();
          for (int i=0; i<localMatchCount;)
          {
            const auto &value = entryPointer->first;
            const auto &matchingCellIDs = entryPointer->second;
            for (const auto &cellID : matchingCellIDs)
            {
              localData[i] = std::pair<GlobalIndexType,double>{cellID,value};
              i++;
            }
            entryPointer++;
          }
        }
        
        std::vector<int> offsets;
        std::vector< std::pair<GlobalIndexType,double> > gatheredData(globalMatchCount);

        MPIWrapper::allGatherVariable(Comm, gatheredData, localData, offsets);
        
        std::sort( gatheredData.begin(), gatheredData.end(), entry_comparator );
        
        if (globalMatchCount > numValues)
        {
          // delete everything after the first numValues
          gatheredData.erase( gatheredData.begin() + numValues, gatheredData.end() );
        }
        
        return gatheredData;
      }
    public:
    /// Returns the numValues entries with greatest value in the union of localEntries, sorted by the values
    static std::vector< std::pair<GlobalIndexType,double> > globalMaximums(Epetra_CommPtr Comm, const std::map<GlobalIndexType,double> &localEntries, const int numValues, const int maxGatherCount)
    {
      return globalMinOrMax< std::greater<double> >(Comm, localEntries, numValues, maxGatherCount);
    }
    
    /// Returns the numValues entries with greatest value in the union of localEntries, sorted by the values
    static std::vector< std::pair<GlobalIndexType,double> > globalMaximums(Teuchos_CommPtr Comm, const std::map<GlobalIndexType,double> &localEntries, const int numValues, const int maxGatherCount)
    {
      return globalMinOrMax< std::greater<double> >(Comm, localEntries, numValues, maxGatherCount);
    }

    /// Returns the numValues entries with least value in the union of localEntries, sorted by the values
    static std::vector< std::pair<GlobalIndexType,double> > globalMinimums(Epetra_CommPtr Comm, const std::map<GlobalIndexType,double> &localEntries, const int numValues, const int maxGatherCount)
    {
      return globalMinOrMax< std::less<double> >(Comm, localEntries, numValues, maxGatherCount);
    }
    
    /// Returns the numValues entries with least value in the union of localEntries, sorted by the values
    static std::vector< std::pair<GlobalIndexType,double> > globalMinimums(Teuchos_CommPtr Comm, const std::map<GlobalIndexType,double> &localEntries, const int numValues, const int maxGatherCount)
    {
      return globalMinOrMax< std::less<double> >(Comm, localEntries, numValues, maxGatherCount);
    }
  };
} // end namespace Camellia

#endif /* MPIGlobalMinMax_hpp */
