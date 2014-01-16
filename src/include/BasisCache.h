#ifndef CAMELLIA_BASIS_CACHE
#define CAMELLIA_BASIS_CACHE

/*
 *  BasisCache.h
 *
 */

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

// only works properly with bases obtained from the BasisFactory.
#include "Intrepid_CellTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "ElementType.h"

#include "DofOrdering.h"

#include "CamelliaIntrepidExtendedTypes.h"

#include "Basis.h"

using namespace std;
using namespace IntrepidExtendedTypes;

using namespace Camellia;

class Mesh;
class Function; // Function.h and BasisCache.h refer to each other...
class BasisCache;

typedef Teuchos::RCP<Function> FunctionPtr;
typedef Teuchos::RCP<BasisCache> BasisCachePtr;

class BasisCache {
private:
  int _numCells, _spaceDim;
  int _numSides;
  bool _isSideCache;
  int _sideIndex;
  Teuchos::RCP<Mesh> _mesh;
  vector< Teuchos::RCP<BasisCache> > _basisCacheSides;
  Teuchos::RCP<BasisCache> _basisCacheVolume;
  Intrepid::FieldContainer<double> _cubPoints, _cubWeights;
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
    
  vector<int> _cellIDs; // the list of cell IDs corresponding to the physicalCellNodes
  
  int _cubDegree;
  
  // containers specifically for sides:
  Intrepid::FieldContainer<double> _cubPointsSideRefCell; // the _cubPoints is the one in the side coordinates; this one in volume coords
  Intrepid::FieldContainer<double> _sideNormals;
  
  shards::CellTopology _cellTopo;
  
  map< pair< Camellia::Basis<>*, IntrepidExtendedTypes::EOperatorExtended >,
  Teuchos::RCP< const Intrepid::FieldContainer<double> > > _knownValues;
  
  map< pair< Camellia::Basis<>*, IntrepidExtendedTypes::EOperatorExtended >,
  Teuchos::RCP< const Intrepid::FieldContainer<double> > > _knownValuesTransformed;
  
  map< pair< Camellia::Basis<>*, IntrepidExtendedTypes::EOperatorExtended >,
  Teuchos::RCP< const Intrepid::FieldContainer<double> > > _knownValuesTransformedDottedWithNormal;
  
  map< pair< Camellia::Basis<>*, IntrepidExtendedTypes::EOperatorExtended >,
  Teuchos::RCP< const Intrepid::FieldContainer<double> > > _knownValuesTransformedWeighted;
  
  map< pair< Camellia::Basis<>*, IntrepidExtendedTypes::EOperatorExtended >,
  Teuchos::RCP< const Intrepid::FieldContainer<double> > > _knownValuesTransformedWeightedDottedWithNormal;
  
  void init(shards::CellTopology &cellTopo, int maxTrialDegree, int maxTestDegree, bool createSideCacheToo);

  void determineJacobian();
  void determinePhysicalPoints();
  
  // (private) side cache constructor:
  BasisCache(int sideIndex, Teuchos::RCP<BasisCache> volumeCache, int trialDegree, int testDegree, BasisPtr multiBasisIfAny);
  
  int maxTestDegree();
  void createSideCaches();
  
  void findMaximumDegreeBasisForSides(DofOrdering &trialOrdering);
protected:
  BasisCache() { _isSideCache = false; } // for the sake of some hackish subclassing
  
  vector< BasisPtr > _maxDegreeBasisForSide; // stored in volume cache so we can get cubature right on sides, including broken sides (if this is a multiBasis)
  int _maxTestDegree, _maxTrialDegree;
public:
  BasisCache(ElementTypePtr elemType, Teuchos::RCP<Mesh> mesh = Teuchos::rcp( (Mesh*) NULL ), bool testVsTest=false,
             int cubatureDegreeEnrichment = 0); // use testVsTest=true for test space inner product
  BasisCache(shards::CellTopology &cellTopo, int cubDegree, bool createSideCacheToo);
  BasisCache(const Intrepid::FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo, int cubDegree, bool createSideCacheToo = false);
  BasisCache(const Intrepid::FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo,
             DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo = false);
  virtual ~BasisCache() {}
  
  Teuchos::RCP< const Intrepid::FieldContainer<double> > getValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, bool useCubPointsSideRefCell = false);
  Intrepid::FieldContainer<double> & getWeightedMeasures();
  Intrepid::FieldContainer<double> getCellMeasures();
  Teuchos::RCP< const Intrepid::FieldContainer<double> > getTransformedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, bool useCubPointsSideRefCell = false);
  Teuchos::RCP< const Intrepid::FieldContainer<double> > getTransformedWeightedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, bool useCubPointsSideRefCell = false);
  
  // side variants:
  Teuchos::RCP< const Intrepid::FieldContainer<double> > getValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  Teuchos::RCP< const Intrepid::FieldContainer<double> > getTransformedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  Teuchos::RCP< const Intrepid::FieldContainer<double> > getTransformedWeightedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  
  bool isSideCache();
  Teuchos::RCP<BasisCache> getSideBasisCache(int sideOrdinal);
  Teuchos::RCP<BasisCache> getVolumeBasisCache(); // from sideCache
  
  const vector<int> & cellIDs();
  
  shards::CellTopology cellTopology();
  
  int cubatureDegree();
  
  Teuchos::RCP<Mesh> mesh();
  
  void discardPhysicalNodeInfo(); // discards physicalNodes and all transformed basis values.
  
  const Intrepid::FieldContainer<double> & getJacobian();
  const Intrepid::FieldContainer<double> & getJacobianDet();
  const Intrepid::FieldContainer<double> & getJacobianInv();
  
  Intrepid::FieldContainer<double> computeParametricPoints();
  
  virtual const Intrepid::FieldContainer<double> & getPhysicalCubaturePoints();
  const Intrepid::FieldContainer<double> & getPhysicalCubaturePointsForSide(int sideOrdinal);
  const Intrepid::FieldContainer<double> & getCellSideParities();
  
  const Intrepid::FieldContainer<double> & getSideUnitNormals(int sideOrdinal);
  
  void setPhysicalCellNodes(const Intrepid::FieldContainer<double> &physicalCellNodes, const vector<int> &cellIDs, bool createSideCacheToo);
  
  /*** Methods added for BC support below ***/
  // setRefCellPoints overwrites _cubPoints -- for when cubature is not your interest
  // (this comes up in imposeBC)
  void setRefCellPoints(const Intrepid::FieldContainer<double> &pointsRefCell);
  const Intrepid::FieldContainer<double> &getRefCellPoints();
  const Intrepid::FieldContainer<double> &getSideRefCellPointsInVolumeCoordinates();
  
  // physicalPoints: (P,D).  cellIndex indexes into BasisCache's physicalCellNodes
  Intrepid::FieldContainer<double> getRefCellPointsForPhysicalPoints(const Intrepid::FieldContainer<double> &physicalPoints, int cellIndex=0);

  const Intrepid::FieldContainer<double> & getSideNormals();
  void setSideNormals(Intrepid::FieldContainer<double> &sideNormals);
  void setCellSideParities(const Intrepid::FieldContainer<double> &cellSideParities);
  
  int getMaxCubatureDegree();
  
  int getSideIndex(); // -1 if not sideCache
  
  int getSpaceDim();
  
  void setMaxCubatureDegree(int value);
  
  void setTransformationFunction(FunctionPtr fxn);
    
  // static convenience constructors:
  static BasisCachePtr parametric1DCache(int cubatureDegree);
  static BasisCachePtr parametricQuadCache(int cubatureDegree);
  static BasisCachePtr parametricQuadCache(int cubatureDegree, const Intrepid::FieldContainer<double> &refCellPoints, int sideCacheIndex=-1);
  static BasisCachePtr basisCache1D(double x0, double x1, int cubatureDegree); // x0 and x1: physical space endpoints
  static BasisCachePtr basisCacheForCell(Teuchos::RCP<Mesh> mesh, int cellID, bool testVsTest = false,
                                         int cubatureDegreeEnrichment = 0);
  static BasisCachePtr basisCacheForCellType(Teuchos::RCP<Mesh> mesh, ElementTypePtr elemType, bool testVsTest = false,
                                             int cubatureDegreeEnrichment = 0); // for cells on the local MPI node
  static BasisCachePtr quadBasisCache(double width, double height, int cubDegree, bool createSideCacheToo=false);
  
  // note that this does not inform the volumeCache about the created side cache:
  // Intended for cases where you just want to create a BasisCache for one of the sides, not all of them.
  // If you want one for all of them, you should pass createSideCacheToo = true to an appropriate volumeCache method.
  static BasisCachePtr sideBasisCache(Teuchos::RCP<BasisCache> volumeCache, int sideIndex);
};

#endif
