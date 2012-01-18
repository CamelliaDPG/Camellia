#ifndef DPG_MESH
#define DPG_MESH

// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER 

/*
 *  Mesh.h
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */

// Intrepid includes
#include "Intrepid_FieldContainer.hpp"

// Epetra includes
#include <Epetra_Map.h>
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "ElementType.h"
#include "ElementTypeFactory.h"
#include "Element.h"
#include "Boundary.h"
#include "BilinearForm.h"
#include "DofOrderingFactory.h"
#include "RefinementPattern.h"
#include "MeshPartitionPolicy.h"

class Mesh {
  typedef Teuchos::RCP< ElementType > ElementTypePtr;
  typedef Teuchos::RCP< Element > ElementPtr;
  
  int _pToAddToTest;
  bool _usePatchBasis; // use MultiBasis if this is false.
  // for now, just a uniform mesh, with a rectangular boundary and elements.
  Boundary _boundary;
  DofOrderingFactory _dofOrderingFactory;
  ElementTypeFactory _elementTypeFactory;
  Teuchos::RCP< BilinearForm > _bilinearForm;
  Teuchos::RCP< MeshPartitionPolicy > _partitionPolicy;
  int _numPartitions;
  int _numInitialElements;
  vector< ElementPtr > _elements;
  vector< ElementPtr > _activeElements;
  vector< vector< ElementPtr > > _partitions;
  vector< vector<int> > _verticesForCellID;
  vector< FieldContainer<double> > _vertices;
  map < vector<float>, vector<int> > _verticesMap; // key: coordinates; value: index to _vertices
  //set< pair<int,int> > _edges;
  map< pair<int,int>, vector< pair<int, int> > > _edgeToCellIDs; //keys are (vertexIndex1, vertexIndex2)
                                                                  //values are (cellID, sideIndex)
                                                                  //( will need to do something else in 3D )
  vector< vector<int> > _cellSideParitiesForCellID;
  
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
  map<int, int> _partitionLocalIndexForGlobalDofIndex;
  map<int, int> _partitionForGlobalDofIndex;
  vector< map< ElementType*, FieldContainer<double> > > _partitionedPhysicalCellNodesForElementType;
  vector< map< ElementType*, FieldContainer<double> > > _partitionedCellSideParitiesForElementType;
  map< ElementType*, FieldContainer<double> > _physicalCellNodesForElementType; // for uniform mesh, just a single entry..
  map< ElementType*, FieldContainer<double> > _cellSideParitiesForElementType; // for uniform mesh, just a single entry..
  
  vector< set<int> > _partitionedGlobalDofIndices;
  
  map< pair<int,int> , int> _localToGlobalMap; // pair<cellID, localDofIndex>
  void buildTypeLookups();
  void buildLocalToGlobalMap();
  void determineActiveElements();
  void determineDofPairings();
  void determinePartitionDofIndices();
  void addDofPairing(int cellID1, int dofIndex1, int cellID2, int dofIndex2);
  int _numGlobalDofs;
  ElementPtr _nullPtr;
  ElementPtr addElement(const vector<int> & vertexIndices, ElementTypePtr elemType);
  void addChildren(ElementPtr parent, vector< vector<int> > &children, 
                   vector< vector< pair< int, int> > > &childrenForSide);
  void setNeighbor(ElementPtr elemPtr, int elemSide, ElementPtr neighborPtr, int neighborSide);
  
  // simple utility functions:
  static bool colinear(double x0, double y0, double x1, double y1, double x2, double y2);
  static double distance(double x0, double y0, double x1, double y1);
public:
  Mesh(const vector<FieldContainer<double> > &vertices, vector< vector<int> > &elementVertices,
       Teuchos::RCP< BilinearForm > bilinearForm, int H1Order, int pToAddTest);

  int numInitialElements();

  vector<double> getCellCentroid(int cellID);
  
  static Teuchos::RCP<Mesh> buildQuadMesh(const FieldContainer<double> &quadBoundaryPoints, 
                                          int horizontalElements, int verticalElements,
                                          Teuchos::RCP< BilinearForm > bilinearForm, 
                                          int H1Order, int pTest, bool triangulate=false);
  static Teuchos::RCP<Mesh> buildQuadMeshHybrid(const FieldContainer<double> &quadBoundaryPoints, 
                                                int horizontalElements, int verticalElements,
                                                Teuchos::RCP< BilinearForm > bilinearForm, 
                                                int H1Order, int pTest);
  static void quadMeshCellIDs(FieldContainer<int> &cellIDs, 
                              int horizontalElements, int verticalElements,
                              bool useTriangles);
  
  FieldContainer<double> & physicalCellNodes( ElementTypePtr elemType, int partitionNumber = -1 );
  FieldContainer<double> & cellSideParities( ElementTypePtr elemTypePtr, int partitionNumber = -1 );

  void refine(vector<int> cellIDsForPRefinements, vector<int> cellIDsForHRefinements);
  
  void hRefine(vector<int> cellIDs, Teuchos::RCP<RefinementPattern> refPattern);
  
  BilinearForm & bilinearForm();
  
  vector<ElementPtr> elementsForPoints(FieldContainer<double> &physicalPoints);
  bool elementContainsPoint(ElementPtr elem, double x, double y);
  
  vector< Teuchos::RCP< ElementType > > elementTypes(int partitionNumber=-1); // returns *all* elementTypes by default
  
  ElementPtr getElement(int cellID);
  
  int cellID(Teuchos::RCP< ElementType > elemTypePtr, int cellIndex, int partitionNumber=-1);
  
  void enforceOneIrregularity();

  Epetra_Map getCellIDPartitionMap(int rank, Epetra_Comm* Comm); 
  
  int globalDofIndex(int cellID, int localDofIndex);
  
  set<int> globalDofIndicesForPartition(int partitionNumber);
  
  Boundary &boundary();
  
  int rowSizeUpperBound(); // accounts for multiplicity, but isn't a tight bound
  
  ElementPtr ancestralNeighborForSide(ElementPtr elem, int sideIndex, int &elemSideIndexInNeighbor);
  
  void matchNeighbor(const ElementPtr &elem, int sideIndex);
  
  static int neighborChildPermutation(int childIndex, int numChildrenInSide);
  static int neighborDofPermutation(int dofIndex, int numDofsForSide);
  
  int numGlobalDofs();
  
  int numElements();
  
  int numElementsOfType( Teuchos::RCP< ElementType > elemTypePtr );
  
  void repartition();
  void setNumPartitions(int numPartitions);
  void setPartitionPolicy(  Teuchos::RCP< MeshPartitionPolicy > partitionPolicy );

  vector< Teuchos::RCP< Element > > & activeElements();
  vector< Teuchos::RCP< Element > > & elements();
  vector< Teuchos::RCP< Element > > elementsOfType(int partitionNumber, ElementTypePtr elemTypePtr);
  
  vector< ElementPtr > elementsInPartition(int partitionNumber);

  map< int, BasisPtr > multiBasisUpgradeMap(ElementPtr parent, int sideIndex);
  void getMultiBasisOrdering(DofOrderingPtr &originalNonParentOrdering,
                             ElementPtr parent, int sideIndex, int parentSideIndexInNeighbor,
                             ElementPtr nonParent);
  
  DofOrderingFactory & getDofOrderingFactory();
  ElementTypeFactory & getElementTypeFactory();

  vector<int> vertexIndicesForCell(int cellID);
  FieldContainer<double> vertexCoordinates(int vertexIndex);
  
  void verticesForCell(FieldContainer<double>& vertices, int cellID);
  void verticesForElementType(FieldContainer<double>& vertices, ElementTypePtr elemTypePtr);
  void verticesForSide(FieldContainer<double>& vertices, int cellID, int sideIndex);
  
  int parityForSide(int cellID, int sideIndex);
  int partitionForGlobalDofIndex( int globalDofIndex );
  int partitionLocalIndexForGlobalDofIndex( int globalDofIndex );
  
  Epetra_Map getPartitionMap(); // returns map for current processor's local-to-global dof indices
  
  void rebuildLookups();

  void writeMeshPartitionsToFile(const string & fileName);
};

#endif
