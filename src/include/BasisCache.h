#ifndef CAMELLIA_BASIS_CACHE
#define CAMELLIA_BASIS_CACHE

/*
 *  BasisCache.h
 *
 */

// @HEADER
//
// Original Version Copyright © 2011 Sandia Corporation. All Rights Reserved.
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
//
// @HEADER 

// only works properly with bases obtained from the BasisFactory.
#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "ElementType.h"

#include "DofOrdering.h"

#include "CamelliaIntrepidExtendedTypes.h"

#include "Basis.h"

#include "IndexType.h"

#include "Mesh.h"

#include "Function.h"

class Mesh;
class Function; // Function.h and BasisCache.h refer to each other...
class BasisCache;

typedef Teuchos::RCP<Function> FunctionPtr;
typedef Teuchos::RCP<BasisCache> BasisCachePtr;

class BasisCache {
private:
  IndexType _numCells;
  int _spaceDim;
  bool _isSideCache;
  int _sideIndex;
  
  int _maxPointsPerCubaturePhase; // default: -1 (infinite)
  int _cubaturePhase; // index of the cubature phase; defaults to 0
  int _cubaturePhaseCount; // how many phases to get through all the points
  std::vector<int> _phasePointOrdinalOffsets;
  
  Teuchos::RCP<Mesh> _mesh;
  Intrepid::FieldContainer<double> _cubPoints, _cubWeights;
  Intrepid::FieldContainer<double> _allCubPoints, _allCubWeights; // when using phased cubature points, these store the whole set
  
  Intrepid::FieldContainer<double> _cellJacobian;
  Intrepid::FieldContainer<double> _cellJacobInv;
  Intrepid::FieldContainer<double> _cellJacobDet;
  Intrepid::FieldContainer<double> _weightedMeasure;
  Intrepid::FieldContainer<double> _physCubPoints;
  Intrepid::FieldContainer<double> _cellSideParities;
  Intrepid::FieldContainer<double> _physicalCellNodes;
  
  FunctionPtr _transformationFxn;
  bool _composeTransformationFxnWithMeshTransformation;
  // bool: compose with existing ref-to-mesh-cell transformation. (false means that the function goes from ref to the physical geometry;
  //                                                                true means it goes from the straight-edge mesh to the curvilinear one)
    
  std::vector<GlobalIndexType> _cellIDs; // the list of cell IDs corresponding to the physicalCellNodes
  
  // we use *EITHER* _cubDegree or _cubDegrees
  // if _cubDegree is -1, use _cubDegrees
  // (_cubDegree == -1) <=> (_cubDegrees.size() > 0)
  int _cubDegree;
  vector<int> _cubDegrees;
  
  // containers specifically for sides:
  Intrepid::FieldContainer<double> _cubPointsSideRefCell; // the _cubPoints is the one in the side coordinates; this one in volume coords
  Intrepid::FieldContainer<double> _sideNormals;
  Intrepid::FieldContainer<double> _sideNormalsSpaceTime; // for space-time CellTopologies, a copy of _sideNormals that includes the temporal component
  
  CellTopoPtr _cellTopo;

  void initCubatureDegree(int maxTrialDegree, int maxTestDegree);
  void initCubatureDegree(std::vector<int> &maxTrialDegrees, std::vector<int> &maxTestDegrees);
  
  void init(bool createSideCacheToo, bool interpretTensorTopologyAsSpaceTime);

  void determineJacobian();
  void determinePhysicalPoints();
  
  int maxTestDegree();
  
  void findMaximumDegreeBasisForSides(DofOrdering &trialOrdering);
  
  void recomputeMeasures();
protected:
  BasisCache() { _isSideCache = false; } // for the sake of some hackish subclassing
  
  map< pair< Camellia::Basis<>*, Camellia::EOperator >,
  Teuchos::RCP< const Intrepid::FieldContainer<double> > > _knownValues;
  
  map< pair< Camellia::Basis<>*, Camellia::EOperator >,
  Teuchos::RCP< const Intrepid::FieldContainer<double> > > _knownValuesTransformed;
  
  map< pair< Camellia::Basis<>*, Camellia::EOperator >,
  Teuchos::RCP< const Intrepid::FieldContainer<double> > > _knownValuesTransformedDottedWithNormal;
  
  map< pair< Camellia::Basis<>*, Camellia::EOperator >,
  Teuchos::RCP< const Intrepid::FieldContainer<double> > > _knownValuesTransformedWeighted;
  
  map< pair< Camellia::Basis<>*, Camellia::EOperator >,
  Teuchos::RCP< const Intrepid::FieldContainer<double> > > _knownValuesTransformedWeightedDottedWithNormal;
  
  std::vector< BasisCachePtr > _basisCacheSides;
  BasisCachePtr _basisCacheVolume;
  
  virtual void createSideCaches();
  
  // protected side cache constructor:
  BasisCache(int sideIndex, BasisCachePtr volumeCache, int trialDegree, int testDegree, BasisPtr multiBasisIfAny);
  
  // protected "fake" side cache constructor:
  BasisCache(int fakeSideOrdinal, BasisCachePtr volumeCache, const FieldContainer<double> &volumeRefPoints,
             const FieldContainer<double> &sideNormals, const FieldContainer<double> &cellSideParities);
  
  // protected constructor basically for the sake of the SpaceTimeBasisCache subclass, which wants to disable side cache creation during construction.
  BasisCache(ElementTypePtr elemType, MeshPtr mesh, bool testVsTest,
             int cubatureDegreeEnrichment, bool tensorProductTopologyMeansSpaceTime,
             bool createSideCacheToo);
  
  std::vector< BasisPtr > _maxDegreeBasisForSide; // stored in volume cache so we can get cubature right on sides, including broken sides (if this is a multiBasis)
  int _maxTestDegree, _maxTrialDegree;
  
  void cubatureDegreeForElementType(ElementTypePtr elemType, bool testVsTest, int &cubatureDegree);
  void cubatureDegreeForElementType(ElementTypePtr elemType, bool testVsTest, int &cubatureDegreeSpace, int &cubatureDegreeTime);
public:
  BasisCache(ElementTypePtr elemType, Teuchos::RCP<Mesh> mesh = Teuchos::rcp( (Mesh*) NULL ), bool testVsTest=false,
             int cubatureDegreeEnrichment = 0, bool tensorProductTopologyMeansSpaceTime = true); // use testVsTest=true for test space inner product
  
  BasisCache(CellTopoPtr cellTopo, int cubDegree, bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime=true);
  BasisCache(shards::CellTopology &cellTopo, int cubDegree, bool createSideCacheToo);
  
  BasisCache(const Intrepid::FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo, int cubDegree, bool createSideCacheToo = false);
  BasisCache(const Intrepid::FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo, int cubDegree, bool createSideCacheToo = false, bool tensorProductTopologyMeansSpaceTime=true);

  BasisCache(const Intrepid::FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo,
             DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo = false, bool tensorProductTopologyMeansSpaceTime=true);
  BasisCache(const Intrepid::FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo,
             DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo = false);
  virtual ~BasisCache() {}
  
  Intrepid::FieldContainer<double> & getWeightedMeasures();
  Intrepid::FieldContainer<double> getCellMeasures();
  
  virtual Teuchos::RCP< const Intrepid::FieldContainer<double> > getValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  virtual Teuchos::RCP< const Intrepid::FieldContainer<double> > getTransformedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  virtual Teuchos::RCP< const Intrepid::FieldContainer<double> > getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  
  // side variants:
  virtual Teuchos::RCP< const Intrepid::FieldContainer<double> > getValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  virtual Teuchos::RCP< const Intrepid::FieldContainer<double> > getTransformedValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  virtual Teuchos::RCP< const Intrepid::FieldContainer<double> > getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  
  bool isSideCache();
  BasisCachePtr getSideBasisCache(int sideOrdinal);
  BasisCachePtr getVolumeBasisCache(); // from sideCache
  
  const std::vector<GlobalIndexType> & cellIDs();
  
  CellTopoPtr cellTopology();
  
  int cubatureDegree();
  
  int getCubaturePhaseCount();
  void setMaxPointsPerCubaturePhase(int maxPoints);
  void setCubaturePhase(int phaseOrdinal);
  
  Teuchos::RCP<Mesh> mesh();
  void setMesh(Teuchos::RCP<Mesh> mesh);
  
  void discardPhysicalNodeInfo(); // discards physicalNodes and all transformed basis values.
  
  const Intrepid::FieldContainer<double> & getJacobian();
  const Intrepid::FieldContainer<double> & getJacobianDet();
  const Intrepid::FieldContainer<double> & getJacobianInv();
  
  Intrepid::FieldContainer<double> computeParametricPoints();
  
  virtual const Intrepid::FieldContainer<double> & getPhysicalCubaturePoints();
  const Intrepid::FieldContainer<double> & getPhysicalCubaturePointsForSide(int sideOrdinal);
  const Intrepid::FieldContainer<double> & getCellSideParities();
  
  const Intrepid::FieldContainer<double> & getCubatureWeights();
  
  const Intrepid::FieldContainer<double> & getSideUnitNormals(int sideOrdinal);
  
  const Intrepid::FieldContainer<double> &getPhysicalCellNodes();
  void setPhysicalCellNodes(const Intrepid::FieldContainer<double> &physicalCellNodes, const std::vector<GlobalIndexType> &cellIDs, bool createSideCacheToo);
  
  /*** Methods added for BC support below ***/
  // setRefCellPoints overwrites _cubPoints -- for when cubature is not your interest
  // (this comes up in imposeBC)
  virtual void setRefCellPoints(const Intrepid::FieldContainer<double> &pointsRefCell);
  virtual void setRefCellPoints(const Intrepid::FieldContainer<double> &pointsRefCell,
                                const Intrepid::FieldContainer<double> &cubatureWeights);
  const Intrepid::FieldContainer<double> &getRefCellPoints();
  const Intrepid::FieldContainer<double> &getSideRefCellPointsInVolumeCoordinates();
  
  // physicalPoints: (P,D).  cellIndex indexes into BasisCache's physicalCellNodes
  Intrepid::FieldContainer<double> getRefCellPointsForPhysicalPoints(const Intrepid::FieldContainer<double> &physicalPoints, int cellIndex=0);

  /** \brief  Returns an Intrepid::FieldContainer<double> populated with the side normals; dimensions are (C,P,D) or (C,P,D-1).  Tensor-product topologies are interpreted as space-time elements; in this context, the side normals provided will be the spatial part of the space-time normal.
   */
  const Intrepid::FieldContainer<double> & getSideNormals();
  void setSideNormals(Intrepid::FieldContainer<double> &sideNormals);
  void setCellSideParities(const Intrepid::FieldContainer<double> &cellSideParities);

  /** \brief  Returns an Intrepid::FieldContainer<double> populated with the full space-time side normals; dimensions are (C,P,D).  For non-tensor-product topologies, throws an exception.
   */
  const Intrepid::FieldContainer<double> & getSideNormalsSpaceTime();
  
  int getMaxCubatureDegree();
  
  int getSideIndex() const; // -1 if not sideCache
  
  virtual int getSpaceDim();
  
  void setMaxCubatureDegree(int value);
  
  void setTransformationFunction(FunctionPtr fxn, bool composeWithMeshTransformation = true);
  
  // static convenience constructors:
  static BasisCachePtr parametric1DCache(int cubatureDegree);
  static BasisCachePtr parametricQuadCache(int cubatureDegree);
  static BasisCachePtr parametricQuadCache(int cubatureDegree, const Intrepid::FieldContainer<double> &refCellPoints, int sideCacheIndex=-1);
  static BasisCachePtr basisCache1D(double x0, double x1, int cubatureDegree); // x0 and x1: physical space endpoints
  static BasisCachePtr basisCacheForCell(Teuchos::RCP<Mesh> mesh, GlobalIndexType cellID, bool testVsTest = false,
                                         int cubatureDegreeEnrichment = 0, bool tensorProductTopologyMeansSpaceTime=true);
  static BasisCachePtr basisCacheForCellType(Teuchos::RCP<Mesh> mesh, ElementTypePtr elemType, bool testVsTest = false,
                                             int cubatureDegreeEnrichment = 0, bool tensorProductTopologyMeansSpaceTime=true); // for cells on the local MPI node
  static BasisCachePtr basisCacheForReferenceCell(shards::CellTopology &cellTopo, int cubatureDegree, bool createSideCacheToo=false);
  static BasisCachePtr basisCacheForRefinedReferenceCell(shards::CellTopology &cellTopo, int cubatureDegree, RefinementBranch refinementBranch, bool createSideCacheToo=false);

  static BasisCachePtr basisCacheForCellTopology(CellTopoPtr cellTopo, int cubatureDegree,
                                                 const FieldContainer<double> &physicalCellNodes,
                                                 bool createSideCacheToo=false,
                                                 bool tensorProductTopologyMeansSpaceTime=true);
  
  static BasisCachePtr basisCacheForReferenceCell(CellTopoPtr cellTopo, int cubatureDegree, bool createSideCacheToo=false,
                                                  bool tensorProductTopologyMeansSpaceTime=true);
  static BasisCachePtr basisCacheForRefinedReferenceCell(CellTopoPtr cellTopo, int cubatureDegree, RefinementBranch refinementBranch,
                                                         bool createSideCacheToo=false, bool tensorProductTopologyMeansSpaceTime=true);
  
  static BasisCachePtr quadBasisCache(double width, double height, int cubDegree, bool createSideCacheToo=false);
  
  // note that this does not inform the volumeCache about the created side cache:
  // Intended for cases where you just want to create a BasisCache for one of the sides, not all of them.
  // If you want one for all of them, you should pass createSideCacheToo = true to an appropriate volumeCache method.
  static BasisCachePtr sideBasisCache(BasisCachePtr volumeCache, int sideIndex);
  
  // ! As the name suggests, this method is not meant for widespread use.  Intended mainly for flux-to-field mappings
  static BasisCachePtr fakeSideCache(int fakeSideOrdinal, BasisCachePtr volumeCache, const FieldContainer<double> &volumeRefPoints,
                                     const FieldContainer<double> &sideNormals, const FieldContainer<double> &cellSideParities);
};

#endif
