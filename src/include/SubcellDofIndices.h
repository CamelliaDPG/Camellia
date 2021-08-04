// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  SubcellDofIndices.h
//  Camellia
//
//  Created by Nate Roberts on 4/18/16.
//
//

#ifndef Camellia_SubcellDofIndices_h
#define Camellia_SubcellDofIndices_h

#include <map>
#include <set>
#include <vector>

#include "CamelliaMemoryUtility.h"
#include "RangeList.h"
#include "Var.h"
#include "TimeLogger.h"
#include "TypeDefs.h"

#include "Teuchos_TestForException.hpp"

namespace Camellia {
  class SubcellDofIndices
  {
  public:
    typedef std::vector< std::pair<int,std::pair<GlobalIndexType,int>> > VarIDToDofIndices; // key: varID
    typedef std::vector<VarIDToDofIndices> SubCellOrdinalToMap; // key: subcell ordinal
    std::vector<SubCellOrdinalToMap> subcellDofIndices; // index to vector: subcell dimension
  public:
    SubcellDofIndices() {}
    SubcellDofIndices(CellTopoPtr cellTopo)
    {
      int dim = cellTopo->getDimension();
      subcellDofIndices.resize(dim + 1);
      for (int d=0; d<=dim; d++)
      {
        int scCount = cellTopo->getSubcellCount(d);
        subcellDofIndices[d].resize(scCount);
      }
    }
    
    long long approximateMemoryFootprint() // in bytes
    {
      int myFootprint = sizeof(subcellDofIndices); // for the overhead of vector
      for (auto outerMap : subcellDofIndices)
      {
        myFootprint += sizeof(outerMap); // overhead for the NewSubCellOrdinalToMap vector
        for (auto outerMapEntry : outerMap)
        {
          myFootprint += sizeof(outerMapEntry); // overhead for NewVarIDToDofIndices vector
          for (auto innerMapEntry : outerMapEntry)
          {
            myFootprint += sizeof(innerMapEntry);
          }
        }
      }
      return myFootprint;
    }
    
    void addOffsetToDofIndices(GlobalIndexType dofIndexOffset)
    {
      for (int d=0; d<subcellDofIndices.size(); d++)
      {
        std::vector<VarIDToDofIndices>* entriesForSubcellOrdinal = &subcellDofIndices[d];
        int scCount = entriesForSubcellOrdinal->size();
        for (int scord=0; scord < scCount; scord++)
        {
          VarIDToDofIndices* entryList = &(*entriesForSubcellOrdinal)[scord];
          int numEntries = entryList->size();
          for (int entryOrdinal = 0; entryOrdinal < numEntries; entryOrdinal++)
          {
            (*entryList)[entryOrdinal].second.first += dofIndexOffset;
          }
        }
      }
    }
    
    std::pair<GlobalIndexType,int> & dofIndexRangeForVarOnSubcell(unsigned d, unsigned scord, int varID)
    {
      int entryCount = subcellDofIndices[d][scord].size();
      for (int entryOrdinal=0; entryOrdinal<entryCount; entryOrdinal++)
      {
        int entryVarID = subcellDofIndices[d][scord][entryOrdinal].first;
        if (varID == entryVarID)
        {
          return subcellDofIndices[d][scord][entryOrdinal].second;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Entry not found");
    }
    
    std::vector<GlobalIndexType> dofIndicesForVarOnSubcell(unsigned d, unsigned scord, int varID)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(d >= subcellDofIndices.size(), std::invalid_argument, "d is out of range");
      TEUCHOS_TEST_FOR_EXCEPTION(scord >= subcellDofIndices[d].size(), std::invalid_argument, "scord is out of range");
      std::vector<GlobalIndexType> dofIndices;
      int entryCount = subcellDofIndices[d][scord].size();
      if (varID != -1)
      {
        for (int entryOrdinal=0; entryOrdinal<entryCount; entryOrdinal++)
        {
          int entryVarID = subcellDofIndices[d][scord][entryOrdinal].first;
          if (varID == entryVarID)
          {
            GlobalIndexType firstDofIndex = subcellDofIndices[d][scord][entryOrdinal].second.first;
            int dofCount = subcellDofIndices[d][scord][entryOrdinal].second.second;
            dofIndices.resize(dofCount);
            for (int dofOrdinal=0; dofOrdinal<dofCount; dofOrdinal++)
            {
              dofIndices[dofOrdinal] = firstDofIndex + dofOrdinal;
            }
            break; // we require that there only be one entry per varID
          }
        }
      }
      else
      {
        for (int entryOrdinal=0; entryOrdinal<entryCount; entryOrdinal++)
        {
          GlobalIndexType firstDofIndex = subcellDofIndices[d][scord][entryOrdinal].second.first;
          int dofCount = subcellDofIndices[d][scord][entryOrdinal].second.second;
          for (int dofOrdinal=0; dofOrdinal<dofCount; dofOrdinal++)
          {
            dofIndices.push_back(firstDofIndex + dofOrdinal);
          }
        }
      }
      return dofIndices;
    }
    
    void print(std::ostream &os) const
    {
      for (int d=0; d<subcellDofIndices.size(); d++)
      {
        os << "****** dimension " << d << " *******\n";
        int scCount = subcellDofIndices[d].size();
        for (int scord=0; scord<scCount; scord++)
        {
          os << "  scord " << scord << ":\n";
          int entryCount = subcellDofIndices[d][scord].size();
          for (int entryOrdinal=0; entryOrdinal<entryCount; entryOrdinal++)
          {
            pair<int,pair<GlobalIndexType,int>> varEntry = subcellDofIndices[d][scord][entryOrdinal];
            int varID = varEntry.first;
            GlobalIndexType firstGlobalDofOrdinal = varEntry.second.first;
            int dofCount = varEntry.second.second;
            GlobalIndexType lastGlobalDofOrdinal = firstGlobalDofOrdinal + dofCount - 1;
            if (dofCount == 1)
            {
              os << "     var " << varID << ", global dofs: " << firstGlobalDofOrdinal << endl;
            }
            else if (dofCount > 0)
            {
              os << "     var " << varID << ", global dofs: " << firstGlobalDofOrdinal << "-" << lastGlobalDofOrdinal << endl;
            }
          }
        }
      }
    }
    
    void setDofIndicesForVarOnSubcell(unsigned d, unsigned scord, int varID, std::pair<GlobalIndexType, int> range)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(d >= subcellDofIndices.size(), std::invalid_argument, "d is out of range");
      TEUCHOS_TEST_FOR_EXCEPTION(scord >= subcellDofIndices[d].size(), std::invalid_argument, "scord is out of range");
      int entryCount = subcellDofIndices[d][scord].size();
      bool found = false;
      for (int entryOrdinal=0; entryOrdinal<entryCount; entryOrdinal++)
      {
        int entryVarID = subcellDofIndices[d][scord][entryOrdinal].first;
        if (varID == entryVarID)
        {
          subcellDofIndices[d][scord][entryOrdinal] = {varID, range};
          found = true;
          break; // we require that there be only be one entry per varID
        }
      }
      // if none found, add to the end of the entries for the scord
      if (!found)
      {
        subcellDofIndices[d][scord].push_back({varID,range});
      }
    }
    
//    typedef std::map<int, std::vector<GlobalIndexType> > VarIDToDofIndices; // key: varID
//    typedef std::map<unsigned, VarIDToDofIndices> SubCellOrdinalToMap; // key: subcell ordinal
//    std::vector<SubCellOrdinalToMap> subcellDofIndices; // index to vector: subcell dimension
//    
//    long long approximateMemoryFootprint() // in bytes
//    {
//      int myFootprint = sizeof(subcellDofIndices); // for the overhead of subcellDofIndices
//      for (auto outerMap : subcellDofIndices)
//      {
//        myFootprint += sizeof(outerMap); // overhead for the SubCellOrdinalToMap map
//        for (auto outerMapEntry : outerMap)
//        {
//          myFootprint += MAP_NODE_OVERHEAD;
//          myFootprint += sizeof(outerMapEntry);
//          for (auto innerMapEntry : outerMapEntry.second)
//          {
//            myFootprint += MAP_NODE_OVERHEAD;
//            myFootprint += sizeof(innerMapEntry);
//            myFootprint += innerMapEntry.second.size() * sizeof(GlobalIndexType);
//          }
//        }
//      }
//      return myFootprint;
//    }
//    
//    long long approximateMemoryFootprintAlternativeDataStructure(CellTopoPtr cellTopo) // in bytes
//    {
//      // try this:
//      typedef std::vector< pair<int,pair<GlobalIndexType,int>> > NewVarIDToDofIndices; // inner pair: (starting global dof index, count)
//      typedef std::vector<NewVarIDToDofIndices> NewSubCellOrdinalToMap;
//      std::vector<NewSubCellOrdinalToMap> newSubcellDofIndices(cellTopo->getDimension() + 1);
//      
//      for (int d=0; d<=cellTopo->getDimension(); d++)
//      {
//        int scCount = cellTopo->getSubcellCount(d);
//        newSubcellDofIndices[d].resize(scCount);
//        for (int scord=0; scord<scCount; scord++)
//        {
//          if (subcellDofIndices[d].find(scord) != subcellDofIndices[d].end())
//          {
//            auto varIDToDofIndicesMap = &subcellDofIndices[d][scord];
//            newSubcellDofIndices[d][scord].resize(varIDToDofIndicesMap->size());
//            int entryOrdinal = 0;
//            for (pair<int, vector<GlobalIndexType>> entry : *varIDToDofIndicesMap)
//            {
//              int varID = entry.first;
//              vector<GlobalIndexType> globalDofIndices = entry.second;
//              int dofCount = entry.second.size();
//              GlobalIndexType startingDofIndex = (dofCount == 0) ? -1 : entry.second[0];
//              newSubcellDofIndices[d][scord][entryOrdinal] = {varID, {startingDofIndex, dofCount}};
//              entryOrdinal++;
//            }
//          }
//        }
//      }
//      
//      int myFootprint = sizeof(newSubcellDofIndices); // for the overhead of vector
//      for (auto outerMap : newSubcellDofIndices)
//      {
//        myFootprint += sizeof(outerMap); // overhead for the NewSubCellOrdinalToMap vector
//        for (auto outerMapEntry : outerMap)
//        {
//          myFootprint += sizeof(outerMapEntry); // overhead for NewVarIDToDofIndices vector
//          for (auto innerMapEntry : outerMapEntry)
//          {
//            myFootprint += sizeof(innerMapEntry);
//          }
//        }
//      }
//      return myFootprint;
//    }
    
    int dofIndexCountForVariable(VarPtr var)
    {
      RangeList<GlobalIndexType> dofIndices;
      for (int d=0; d<subcellDofIndices.size(); d++)
      {
        std::vector<VarIDToDofIndices>* entriesForSubcellOrdinal = &subcellDofIndices[d];
        int scCount = entriesForSubcellOrdinal->size();
        for (int scord=0; scord < scCount; scord++)
        {
          VarIDToDofIndices* entryList = &(*entriesForSubcellOrdinal)[scord];
          int numEntries = entryList->size();
          for (int entryOrdinal = 0; entryOrdinal < numEntries; entryOrdinal++)
          {
            int entryVarID = (*entryList)[entryOrdinal].first;
            if (entryVarID == var->ID())
            {
              pair<GlobalIndexType, int> dofIndexRange = (*entryList)[entryOrdinal].second;
              dofIndices.insertRange(dofIndexRange.first, dofIndexRange.second);
            }
          }
        }
      }
      return dofIndices.size();
    }
    
    int dataSize() const
    {
      int size = 0;
      int numDimensions = subcellDofIndices.size();
      size += sizeof(numDimensions);
      for (int d=0; d<numDimensions; d++)
      {
        int scCount = subcellDofIndices[d].size();
        size += sizeof(scCount);
        for (int scord = 0; scord < scCount; scord++)
        {
          int numEntries = subcellDofIndices[d][scord].size();
          size += sizeof(numEntries);
          for (int entryOrdinal=0; entryOrdinal<numEntries; entryOrdinal++)
          {
            pair<int,pair<GlobalIndexType,int>> entry = subcellDofIndices[d][scord][entryOrdinal];
            size += sizeof(entry);
          }
        }
      }
      return size;
    }
    
    void read(const char* &dataLocation, int size)
    {
      int timerHandle = TimeLogger::sharedInstance()->startTimer("read SubcellDofIndices");
      subcellDofIndices.clear();
      int numDimensions;
      memcpy(&numDimensions, dataLocation, sizeof(numDimensions));
      dataLocation += sizeof(numDimensions);
      subcellDofIndices.resize(numDimensions);
      for (int d=0; d<numDimensions; d++)
      {
        int scCount;
        memcpy(&scCount, dataLocation, sizeof(scCount));
        dataLocation += sizeof(scCount);
        subcellDofIndices[d].resize(scCount);
        for (int scord=0; scord<scCount; scord++)
        {
          int numEntries;
          memcpy(&numEntries, dataLocation, sizeof(numEntries));
          dataLocation += sizeof(numEntries);
          subcellDofIndices[d][scord].resize(numEntries);
          for (int entryOrdinal=0; entryOrdinal<numEntries; entryOrdinal++)
          {
            pair<int,pair<GlobalIndexType,int>> entry;
            memcpy(&entry, dataLocation, sizeof(entry));
            dataLocation += sizeof(entry);
            subcellDofIndices[d][scord][entryOrdinal] = entry;
          }
        }
      }
      TimeLogger::sharedInstance()->stopTimer(timerHandle);
    }
    
    void write(char* &dataLocation, int bufferSize) const
    {
      int timerHandle = TimeLogger::sharedInstance()->startTimer("write SubcellDofIndices");
      int numDimensions = subcellDofIndices.size();
      memcpy(dataLocation, &numDimensions, sizeof(numDimensions));
      dataLocation += sizeof(numDimensions);
      for (int d=0; d<numDimensions; d++)
      {
        int scCount = subcellDofIndices[d].size();
        memcpy(dataLocation, &scCount, sizeof(scCount));
        dataLocation += sizeof(scCount);
        for (int scord=0; scord<scCount; scord++)
        {
          int numEntries = subcellDofIndices[d][scord].size();
          memcpy(dataLocation, &numEntries, sizeof(numEntries));
          dataLocation += sizeof(numEntries);
          for (int entryOrdinal=0; entryOrdinal<numEntries; entryOrdinal++)
          {
            pair<int,pair<GlobalIndexType,int>> entry = subcellDofIndices[d][scord][entryOrdinal];
            memcpy(dataLocation, &entry, sizeof(entry));
            dataLocation += sizeof(entry);
          }
        }
      }
      TimeLogger::sharedInstance()->stopTimer(timerHandle);
    }
  };
}
#endif
