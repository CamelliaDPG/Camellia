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

using namespace Intrepid;
using namespace std;
using namespace IntrepidExtendedTypes;

class Mesh;
class Function; // Function.h and BasisCache.h refer to each other...
typedef Teuchos::RCP<Function> FunctionPtr;
typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;
typedef Teuchos::RCP<ElementType> ElementTypePtr;

class BasisCache {
  typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
private:
  int _numCells, _spaceDim;
  int _numSides;
  bool _isSideCache;
  int _sideIndex;
  Teuchos::RCP<Mesh> _mesh;
  vector< Teuchos::RCP<BasisCache> > _basisCacheSides;
  Teuchos::RCP<BasisCache> _basisCacheVolume;
  FieldContainer<double> _cubPoints, _cubWeights;
  FieldContainer<double> _cellJacobian;
  FieldContainer<double> _cellJacobInv;
  FieldContainer<double> _cellJacobDet;
  FieldContainer<double> _weightedMeasure;
  FieldContainer<double> _physCubPoints;
  FieldContainer<double> _cellSideParities;
  FieldContainer<double> _physicalCellNodes;
  
  FunctionPtr _transformationFxn;
  bool _composeTransformationFxnWithMeshTransformation;
  // bool: compose with existing ref-to-mesh-cell transformation. (false means that the function goes from ref to the physical geometry;
  //                                                                true means it goes from the straight-edge mesh to the curvilinear one)
  
  // eventually, will likely want to have _testOrdering, too--and RCP's would be better than copies (need to change constructors)
  DofOrdering _trialOrdering;
  
  vector<int> _cellIDs; // the list of cell IDs corresponding to the physicalCellNodes
  
  int _cubDegree, _maxTestDegree;
  
  // containers specifically for sides:
  FieldContainer<double> _cubPointsSideRefCell; // the _cubPoints is the one in the side coordinates; this one in volume coords
  FieldContainer<double> _sideNormals;
  
  shards::CellTopology _cellTopo;
  
  map< pair< Basis<double,FieldContainer<double> >*, IntrepidExtendedTypes::EOperatorExtended >,
  Teuchos::RCP< const FieldContainer<double> > > _knownValues;
  
  map< pair< Basis<double,FieldContainer<double> >*, IntrepidExtendedTypes::EOperatorExtended >,
  Teuchos::RCP< const FieldContainer<double> > > _knownValuesTransformed;
  
  map< pair< Basis<double,FieldContainer<double> >*, IntrepidExtendedTypes::EOperatorExtended >,
  Teuchos::RCP< const FieldContainer<double> > > _knownValuesTransformedDottedWithNormal;
  
  map< pair< Basis<double,FieldContainer<double> >*, IntrepidExtendedTypes::EOperatorExtended >,
  Teuchos::RCP< const FieldContainer<double> > > _knownValuesTransformedWeighted;
  
  map< pair< Basis<double,FieldContainer<double> >*, IntrepidExtendedTypes::EOperatorExtended >,
  Teuchos::RCP< const FieldContainer<double> > > _knownValuesTransformedWeightedDottedWithNormal;
  
  // Intrepid::EOperator relatedOperator(EOperatorExtended op, int &componentOfInterest);
  //  Teuchos::RCP< const FieldContainer<double> > getComponentOfInterest(Teuchos::RCP< const FieldContainer<double> > values,
  //                                                                int componentOfInterest);
  void init(shards::CellTopology &cellTopo, DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo);

  void determineJacobian();
  void determinePhysicalPoints();
  
  // (private) side cache constructor:
  BasisCache(int sideIndex, Teuchos::RCP<BasisCache> volumeCache, BasisPtr maxDegreeBasis);
  
  int maxTestDegree();
  void createSideCaches();
protected:
  BasisCache() {} // for the sake of some hackish subclassing
public:
  BasisCache(ElementTypePtr elemType, Teuchos::RCP<Mesh> mesh = Teuchos::rcp( (Mesh*) NULL ), bool testVsTest=false, int cubatureDegreeEnrichment = 0); // use testVsTest=true for test space inner product
  BasisCache(const FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo, int cubDegree);
  BasisCache(const FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo,
             DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo = false);
  
  Teuchos::RCP< const FieldContainer<double> > getValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, bool useCubPointsSideRefCell = false);
  FieldContainer<double> & getWeightedMeasures();
  FieldContainer<double> getCellMeasures();
  Teuchos::RCP< const FieldContainer<double> > getTransformedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, bool useCubPointsSideRefCell = false);
  Teuchos::RCP< const FieldContainer<double> > getTransformedWeightedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, bool useCubPointsSideRefCell = false);
  
  // side variants:
  Teuchos::RCP< const FieldContainer<double> > getValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  Teuchos::RCP< const FieldContainer<double> > getTransformedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  Teuchos::RCP< const FieldContainer<double> > getTransformedWeightedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  
  // side cache accessor: (new, pretty untested!)
  bool isSideCache();
  Teuchos::RCP<BasisCache> getSideBasisCache(int sideOrdinal);
  Teuchos::RCP<BasisCache> getVolumeBasisCache(); // from sideCache
  
  const vector<int> & cellIDs();
  
  shards::CellTopology cellTopology();
  
  int cubatureDegree();
  
  Teuchos::RCP<Mesh> mesh();
  
  void discardPhysicalNodeInfo(); // discards physicalNodes and all transformed basis values.
  
  virtual const FieldContainer<double> & getPhysicalCubaturePoints();
  const FieldContainer<double> & getPhysicalCubaturePointsForSide(int sideOrdinal);
  const FieldContainer<double> & getCellSideParities();
  
  const FieldContainer<double> & getSideUnitNormals(int sideOrdinal);
  
  void setPhysicalCellNodes(const FieldContainer<double> &physicalCellNodes, const vector<int> &cellIDs, bool createSideCacheToo);
  
  /*** Methods added for BC support below ***/
  // setRefCellPoints overwrites _cubPoints -- for when cubature is not your interest
  // (this comes up in imposeBC)
  void setRefCellPoints(const FieldContainer<double> &pointsRefCell);
  const FieldContainer<double> getRefCellPoints(); 

  const FieldContainer<double> & getSideNormals();
  void setSideNormals(FieldContainer<double> &sideNormals);
  void setCellSideParities(const FieldContainer<double> &cellSideParities);
  
  int getSideIndex(); // -1 if not sideCache
  
  void setTransformationFunction(FunctionPtr fxn, bool composeWithMeshTransformation);
};

typedef Teuchos::RCP<BasisCache> BasisCachePtr;

#endif
