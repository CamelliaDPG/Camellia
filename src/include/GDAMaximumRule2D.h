//
//  GDAMaximumRule2D.h
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#ifndef __Camellia_debug__GDAMaximumRule2D__
#define __Camellia_debug__GDAMaximumRule2D__

#include <iostream>

#include "GlobalDofAssignment.h"
#include "ElementTypeFactory.h"
#include "Element.h"

class GDAMaximumRule2D : public GlobalDofAssignment {
  // much of this code copied from Mesh
  
  vector< vector<int> > _cellSideParitiesForCellID;
  
  ElementTypeFactory _elementTypeFactory;
  Teuchos::RCP< BilinearForm > _bilinearForm;
  Teuchos::RCP< MeshPartitionPolicy > _partitionPolicy;
  int _numPartitions;
  int _numInitialElements;
  vector< ElementPtr > _elements;
  vector< ElementPtr > _activeElements;
  vector< vector< ElementPtr > > _partitions;
  
  // keep track of upgrades to the sides of cells since the last rebuild:
  // (used to remap solution coefficients)
  map< int, pair< ElementTypePtr, ElementTypePtr > > _cellSideUpgrades; // cellID --> (oldType, newType)
  
  map< pair<int,int>, pair<int,int> > _dofPairingIndex; // key/values are (cellID,localDofIndex)
  // note that the FieldContainer for cellSideParities has dimensions (numCellsForType,numSidesForType),
  // and that the values are 1.0 or -1.0.  These are weights to account for the fact that fluxes are defined in
  // terms of an outward normal, and thus one cell's idea about the flux is the negative of its neighbor's.
  // We decide parity by cellID: the neighbor with the lower cellID gets +1, the higher gets -1.
  
  // call buildTypeLookups to rebuild the elementType data structures:
  vector< map< ElementType*, vector<int> > > _cellIDsForElementType;
  map< ElementType*, map<int, int> > _globalCellIndexToCellID;
  vector< vector< ElementTypePtr > > _elementTypesForPartition;
  vector< ElementTypePtr > _elementTypes;
  map<int, int> _partitionForCellID;
  map<int, int> _partitionForGlobalDofIndex;
  map<int, int> _partitionLocalIndexForGlobalDofIndex;
  vector< map< ElementType*, FieldContainer<double> > > _partitionedPhysicalCellNodesForElementType;
  vector< map< ElementType*, FieldContainer<double> > > _partitionedCellSideParitiesForElementType;
  map< ElementType*, FieldContainer<double> > _physicalCellNodesForElementType; // for uniform mesh, just a single entry..
  vector< set<int> > _partitionedGlobalDofIndices;
  
  map< pair<int,int> , int> _localToGlobalMap; // pair<cellID, localDofIndex>
  
  map<unsigned, unsigned> getGlobalVertexIDs(const FieldContainer<double> &vertexCoordinates);

  void addDofPairing(int cellID1, int dofIndex1, int cellID2, int dofIndex2);
  void buildTypeLookups();
  void buildLocalToGlobalMap();
  void determineActiveElements();
  void determineDofPairings();
  void determinePartitionDofIndices();
  
  void verticesForCells(FieldContainer<double>& vertices, vector<int> &cellIDs);
  void verticesForCell(FieldContainer<double>& vertices, int cellID);
  
  bool _enforceMBFluxContinuity;
  
  int _numGlobalDofs;
  ElementPtr _nullPtr;
  
  static int neighborChildPermutation(int childIndex, int numChildrenInSide);
  static int neighborDofPermutation(int dofIndex, int numDofsForSide);
public:
  GDAMaximumRule2D(MeshTopologyPtr meshTopology, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                   bool enforceMBFluxContinuity = false);
  
  void didHRefine(set<int> &parentCellIDs);
  void didPRefine(set<int> &cellIDs);
  void didHUnrefine(set<int> &parentCellIDs);
  void didPUnrefine(set<int> &cellIDs);
  ElementTypePtr elementType(unsigned cellID);
  unsigned globalDofCount();
  unsigned localDofCount(); // local to the MPI node
};

#endif /* defined(__Camellia_debug__GDAMaximumRule2D__) */
