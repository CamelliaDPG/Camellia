// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  CellDataMigration.h
//  Camellia-debug
//
//  Created by Nate Roberts on 7/15/14.
//
//

#ifndef __Camellia_debug__CellDataMigration__
#define __Camellia_debug__CellDataMigration__

#include "Mesh.h"

namespace Camellia
{
  typedef pair<RefinementBranch,vector<GlobalIndexType>> LabeledRefinementBranch; // first cellID indicates the root cell ID; after that, each cellID indicates first child of each refinement
  typedef pair<LabeledRefinementBranch, vector<vector<double>> > RootedLabeledRefinementBranch; // second contains vertex coordinates for root cell
  
  // RefinementLevel: pairs are (parentCellID, firstChildCellID).
  typedef std::map<RefinementPatternKey,vector<pair<GlobalIndexType,GlobalIndexType>>> RefinementLevel;
  
  void inline readAndAdvance(void *writeLocation, const char* &readLocation, size_t size)
  {
    memcpy(writeLocation, readLocation, size);
    readLocation += size;
  }
  
  void inline writeAndAdvance(char* &writeLocation, const void *readLocation, size_t size)
  {
    memcpy(writeLocation, readLocation, size);
    writeLocation += size;
  }
  
  struct MeshGeometryInfo
  {
    GlobalIndexType globalActiveCellCount;
    GlobalIndexType globalCellCount;
    std::vector<std::vector<std::vector<double>>> rootVertices;
    std::vector<GlobalIndexType> rootCellIDs;
    std::vector<CellTopologyKey> rootCellTopos;
    std::vector<RefinementLevel> refinementLevels;
    std::vector<GlobalIndexType> myCellIDs;
  };
  
class CellDataMigration
{
  // ! solutionCoefficients stores solution data that has been migrated to this MPI rank, but not yet stored to the appropriate Solution object.
  // ! The method processSolutionCoefficients() will process these, and clear the container.  (The deferral is necessary because we may not
  // ! have assigned global degrees of freedom to the cell in question by the time we receive the solution coefficients for the cell.)
  // ! This is not an ideal design, for many reasons.  It would be better to communicate solution coefficients independently of Zoltan; that
  // ! way, we only communicate them when we are ready to use them.
  static std::vector<std::pair<std::pair<GlobalIndexType,bool>,Camellia::FieldContainer<double>>> solutionCoefficients; // ((cellID, coeffsBelongToParent), coefficients)
public:
  // methods for Zoltan's benefit
  static int dataSize(Mesh* mesh, GlobalIndexType cellID, bool willPackParentDofs);
  static void packData(Mesh* mesh, GlobalIndexType cellID, bool packParentDofs, char *dataBuffer, int size);
  static void unpackData(Mesh* mesh, GlobalIndexType cellID, const char *dataBuffer, int size);
  
  // methods used in the above, made public for testing
  static int geometryDataSize(Mesh* mesh, GlobalIndexType cellID);
  static void packGeometryData(Mesh* mesh, GlobalIndexType cellID, char* &dataLocation, int size);
  static void unpackGeometryData(Mesh* mesh, GlobalIndexType cellID, const char* &dataLocation, int size,
                                 vector<RootedLabeledRefinementBranch> &rootedLabeledBranches);
  
  static int solutionDataSize(Mesh* mesh, GlobalIndexType cellID, bool willPackParentDofs);
  static void packSolutionData(Mesh* mesh, GlobalIndexType cellID, bool packParentDofs, char* &dataLocation, int size);
  static void unpackSolutionData(Mesh* mesh, GlobalIndexType cellID, const char* &dataLocation, int size);
  
  static void getCellGeometry(const MeshTopology* mesh, GlobalIndexType cellID, set<GlobalIndexType> &knownCells,
                              RootedLabeledRefinementBranch &rootedLabeledRefBranch);
  static void getCellHaloGeometry(Mesh *mesh, GlobalIndexType cellID, vector<RootedLabeledRefinementBranch> &cellHaloBranches);
  static void getCellHaloGeometry(const MeshTopology* meshTopo, unsigned minDimensionForContinuity,
                                  GlobalIndexType cellID, vector<RootedLabeledRefinementBranch> &cellHaloBranches);
  
  static void addMigratedGeometry(MeshTopology* meshTopo, const vector<RootedLabeledRefinementBranch> &rootedLabeledBranches);
  
  static void getGeometry(const MeshTopologyView* meshTopo, MeshGeometryInfo &geometryInfo);
  static int getGeometryDataSize(const MeshGeometryInfo &geometryInfo);
  static void writeGeometryData(const MeshGeometryInfo &geometryInfo, char* &dataLocation, int bufferSize);
  // ! Reads 0 or more serialized geometryInfo objects into the geometryInfo structure provided.  (Reads until end of buffer.)
  static void readGeometryData(const char* &dataLocation, int bufferSize, MeshGeometryInfo &geometryInfo);
  
  static void processSolutionCoefficients(Mesh* mesh);
};
}

#endif /* defined(__Camellia_debug__CellDataMigration__) */
