#ifndef DPG_BASIS_VALUE_CACHE
#define DPG_BASIS_VALUE_CACHE

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

#include "BilinearForm.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "DofOrdering.h"

using namespace Intrepid;
using namespace std;

class BasisCache {
  typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
private:
  int _numCells, _spaceDim;
  int _numSides;
  bool _isSideCache;
  vector< Teuchos::RCP<BasisCache> > _basisCacheSides;
  FieldContainer<double> _cubPoints, _cubWeights;
  FieldContainer<double> _cellJacobian;
  FieldContainer<double> _cellJacobInv;
  FieldContainer<double> _cellJacobDet;
  FieldContainer<double> _weightedMeasure;
  FieldContainer<double> _physCubPoints;
  
  // containers specifically for sides:
  FieldContainer<double> _cubPointsSideRefCell; // the _cubPoints is the one in the side coordinates; this one in volume coords
  FieldContainer<double> _sideNormals;
  
  shards::CellTopology _cellTopo;
  
  map< pair< Basis<double,FieldContainer<double> >*, EOperatorExtended >,
        Teuchos::RCP< const FieldContainer<double> > > _knownValues;
  
  map< pair< Basis<double,FieldContainer<double> >*, EOperatorExtended >,
        Teuchos::RCP< const FieldContainer<double> > > _knownValuesTransformed;
  
  map< pair< Basis<double,FieldContainer<double> >*, EOperatorExtended >,
  Teuchos::RCP< const FieldContainer<double> > > _knownValuesTransformedDottedWithNormal;
  
  map< pair< Basis<double,FieldContainer<double> >*, EOperatorExtended >,
        Teuchos::RCP< const FieldContainer<double> > > _knownValuesTransformedWeighted;
  
  map< pair< Basis<double,FieldContainer<double> >*, EOperatorExtended >,
  Teuchos::RCP< const FieldContainer<double> > > _knownValuesTransformedWeightedDottedWithNormal;
  
  // Intrepid::EOperator relatedOperator(EOperatorExtended op, int &componentOfInterest);
//  Teuchos::RCP< const FieldContainer<double> > getComponentOfInterest(Teuchos::RCP< const FieldContainer<double> > values,
//                                                                int componentOfInterest);
  void init(const FieldContainer<double> &physicalCellNodes, 
            shards::CellTopology &cellTopo,
            DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo);
public:
  BasisCache(const FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo, int cubDegree);
  BasisCache(const FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo,
                  DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo = false);
  // side cache constructor:
  BasisCache(shards::CellTopology &cellTopo, int numCells, int spaceDim, FieldContainer<double> &cubPointsSidePhysical,
                  FieldContainer<double> &cubPointsSide, FieldContainer<double> &cubPointsSideRefCell, 
                  FieldContainer<double> &cubWeightsSide, FieldContainer<double> &sideMeasure,
                  FieldContainer<double> &sideNormals, FieldContainer<double> &jacobianSideRefCell,
                  FieldContainer<double> &jacobianInvSideRefCell, FieldContainer<double> &jacobianDetSideRefCell);
    
  Teuchos::RCP< const FieldContainer<double> > getValues(BasisPtr basis, EOperatorExtended op, bool useCubPointsSideRefCell = false);
  FieldContainer<double> getCellMeasures();
  Teuchos::RCP< const FieldContainer<double> > getTransformedValues(BasisPtr basis, EOperatorExtended op, bool useCubPointsSideRefCell = false);
  Teuchos::RCP< const FieldContainer<double> > getTransformedWeightedValues(BasisPtr basis, EOperatorExtended op, bool useCubPointsSideRefCell = false);
  
  // side variants:
  Teuchos::RCP< const FieldContainer<double> > getValues(BasisPtr basis, EOperatorExtended op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  Teuchos::RCP< const FieldContainer<double> > getTransformedValues(BasisPtr basis, EOperatorExtended op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  Teuchos::RCP< const FieldContainer<double> > getTransformedWeightedValues(BasisPtr basis, EOperatorExtended op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  
  const FieldContainer<double> & getPhysicalCubaturePoints();
  const FieldContainer<double> & getPhysicalCubaturePointsForSide(int sideOrdinal);

  const FieldContainer<double> & getSideUnitNormals(int sideOrdinal);
};

#endif
