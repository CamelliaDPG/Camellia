//
//  LinearTerm.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "LinearTerm.h"

typedef pair< FunctionPtr, VarPtr > LinearSummand;

const vector< LinearSummand > & LinearTerm::summands() const { 
  return _summands; 
}

LinearTerm::LinearTerm() {
  _rank = -1;
  _termType = UNKNOWN_TYPE;
}

LinearTerm::LinearTerm(FunctionPtr weight, VarPtr var) {
  _rank = -1;
  _termType = UNKNOWN_TYPE;
  addVar(weight,var);
}

LinearTerm::LinearTerm(double weight, VarPtr var) {
  _rank = -1;
  _termType = UNKNOWN_TYPE;
  addVar(weight,var);
}

LinearTerm::LinearTerm(vector<double> weight, VarPtr var) {
  _rank = -1;
  _termType = UNKNOWN_TYPE;
  addVar(weight,var);
}

LinearTerm::LinearTerm( VarPtr v ) {
  _rank = -1;
  _termType = UNKNOWN_TYPE;
  addVar( 1.0, v);
}

// copy constructor:
LinearTerm::LinearTerm( const LinearTerm &a ) {
  _rank = a.rank();
  _termType = a.termType();
  _summands = a.summands();
  _varIDs = a.varIDs();
}

void LinearTerm::addVar(FunctionPtr weight, VarPtr var) {
  // check ranks:
  int rank; // rank of weight * var
  if (weight->rank() == var->rank() ) { // then we dot like terms together, getting a scalar
    rank = 0;
  } else if ( weight->rank() == 0 || var->rank() == 0) { // then we multiply each term by scalar
    rank = (weight->rank() == 0) ? var->rank() : weight->rank(); // rank is the non-zero one
  } else {
    TEST_FOR_EXCEPTION( true, std::invalid_argument, "Unhandled rank combination.");
  }
  if (_rank == -1) { // LinearTerm's rank is unassigned
    _rank = rank;
  }
  if (_rank != rank) {
    TEST_FOR_EXCEPTION( true, std::invalid_argument, "Attempting to add terms of unlike rank." );
  }
  // check type:
  if (_termType == UNKNOWN_TYPE) {
    _termType = var->varType();
  }
  if (_termType != var->varType() ) {
    TEST_FOR_EXCEPTION( true, std::invalid_argument, "Attempting to add terms of differing type." );
  }
  _summands.push_back( make_pair( weight, var ) );
  _varIDs.insert(var->ID());
}

void LinearTerm::addVar(double weight, VarPtr var) {
  FunctionPtr weightFn = Teuchos::rcp( new ConstantScalarFunction(weight) );
  addVar( weightFn, var );
}

void LinearTerm::addVar(vector<double> vector_weight, VarPtr var) { // dots weight vector with vector var, makes a vector out of a scalar var
  FunctionPtr weightFn = Teuchos::rcp( new ConstantVectorFunction(vector_weight) );
  addVar( weightFn, var );
}

const set<int> & LinearTerm::varIDs() const {
  return _varIDs;
}

VarType LinearTerm::termType() const { 
  return _termType; 
}

void LinearTerm::integrate(FieldContainer<double> &values, DofOrderingPtr thisOrdering,
                           FunctionPtr scalarWeight, BasisCachePtr basisCache,
                           bool forceBoundaryTerm) {
  // values has dimensions (numCells, thisFields)
  set<int> varIDs = this->varIDs();
  
  bool thisFluxOrTrace  = (this->termType() == FLUX) || (this->termType() == TRACE);
  bool boundaryTerm = thisFluxOrTrace || forceBoundaryTerm;
  
  int numSides = boundaryTerm ? basisCache->cellTopology().getSideCount() : 1;

  for (int sideIndex = 0; sideIndex < numSides; sideIndex++ ) {
  
    int numCells  = basisCache->getPhysicalCubaturePoints().dimension(0);
    int numPoints = boundaryTerm ? basisCache->getPhysicalCubaturePointsForSide(sideIndex).dimension(1)
                                 : basisCache->getPhysicalCubaturePoints().dimension(1);
    
    Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > basis;
    
    Teuchos::Array<int> ltValueDim;
    ltValueDim.push_back(numCells);
    ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
    ltValueDim.push_back(numPoints);
    FieldContainer<double> ltValues;
    
    TEST_FOR_EXCEPTION(scalarWeight->rank() != 0, std::invalid_argument, "scalarWeight must be scalar!");
    
    FieldContainer<double> scalarWeightValues(numCells,numPoints);
    if (boundaryTerm) {
      scalarWeight->values(scalarWeightValues,basisCache->getSideBasisCache(sideIndex));
    } else {
      scalarWeight->values(scalarWeightValues,basisCache);
    }
  
    for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++) {
      int varID = *varIt;
      basis = thisFluxOrTrace ? thisOrdering->getBasis(varID,sideIndex)
      : thisOrdering->getBasis(varID);
      int basisCardinality = basis->getCardinality();
      ltValueDim[1] = basisCardinality;
      ltValues.resize(ltValueDim);
      
      if (! boundaryTerm ) {
        this->values(ltValues, varID, basis, basisCache, true); // true: applyCubatureWeights
      } else {
        this->values(ltValues, varID, basis, basisCache, true, sideIndex); // true: applyCubatureWeights
        if ( this->termType() == FLUX ) {
          // we need to multiply ltValues' entries by the parity of the normal, since
          // the trial implicitly contains an outward normal, and we need to adjust for the fact
          // that the neighboring cells have opposite normal
          // thisValues should have dimensions (numCells,numFields,numCubPointsSide)
          int numFields = ltValues.dimension(1);
          int numPoints = ltValues.dimension(2);
          for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            double parity = basisCache->getCellSideParities()(cellIndex,sideIndex);
            if (parity != 1.0) {  // otherwise, we can just leave things be...
              for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
                for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
                  ltValues(cellIndex,fieldIndex,ptIndex) *= parity;
                }
              }
            }
          }
        }
      }
      
      vector<int> varDofIndices = thisFluxOrTrace ? thisOrdering->getDofIndices(varID,sideIndex)
      : thisOrdering->getDofIndices(varID);
      // compute integrals:
      for (int cellIndex = 0; cellIndex<numCells; cellIndex++) {
        for (int basisOrdinal = 0; basisOrdinal < basisCardinality; basisOrdinal++) {
          int varDofIndex = varDofIndices[basisOrdinal];
          for (int ptIndex = 0; ptIndex < numPoints; ptIndex++) {
            values(cellIndex,varDofIndex) += ltValues(cellIndex,basisOrdinal,ptIndex) * scalarWeightValues(cellIndex,ptIndex);
          }
        }
      }
    }
  }
}

void LinearTerm::integrate(FieldContainer<double> &values, DofOrderingPtr thisOrdering,
                           BasisCachePtr basisCache, bool forceBoundaryTerm) {
  // values has dimensions (numCells, thisFields)
  set<int> varIDs = this->varIDs();
  
  int numCells  = basisCache->getPhysicalCubaturePoints().dimension(0);
  
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > basis;
  bool thisFluxOrTrace  = (this->termType() == FLUX) || (this->termType() == TRACE);
  bool boundaryTerm = thisFluxOrTrace || forceBoundaryTerm;
  
  int numSides = boundaryTerm ? basisCache->cellTopology().getSideCount() : 1;
  
  for (int sideIndex = 0; sideIndex < numSides; sideIndex++ ) {
    int numPoints = boundaryTerm ? basisCache->getPhysicalCubaturePointsForSide(sideIndex).dimension(1)
                                 : basisCache->getPhysicalCubaturePoints().dimension(1);
    
    Teuchos::Array<int> ltValueDim;
    ltValueDim.push_back(numCells);
    ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
    ltValueDim.push_back(numPoints);
    FieldContainer<double> ltValues;

    for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++) {
      int varID = *varIt;
      basis = thisFluxOrTrace ? thisOrdering->getBasis(varID,sideIndex)
                              : thisOrdering->getBasis(varID);
      int basisCardinality = basis->getCardinality();
      ltValueDim[1] = basisCardinality;
      ltValues.resize(ltValueDim);
      
      if (! boundaryTerm ) {
        this->values(ltValues, varID, basis, basisCache, true); // true: applyCubatureWeights
      } else {
        this->values(ltValues, varID, basis, basisCache, true, sideIndex); // true: applyCubatureWeights
        if ( this->termType() == FLUX ) {
          // we need to multiply ltValues' entries by the parity of the normal, since
          // the trial implicitly contains an outward normal, and we need to adjust for the fact
          // that the neighboring cells have opposite normal
          // thisValues should have dimensions (numCells,numFields,numCubPointsSide)
          int numFields = ltValues.dimension(1);
          int numPoints = ltValues.dimension(2);
          for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            double parity = basisCache->getCellSideParities()(cellIndex,sideIndex);
            if (parity != 1.0) {  // otherwise, we can just leave things be...
              for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
                for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
                  ltValues(cellIndex,fieldIndex,ptIndex) *= parity;
                }
              }
            }
          }
        }
      }
      
      vector<int> varDofIndices = thisFluxOrTrace ? thisOrdering->getDofIndices(varID,sideIndex)
                                                  : thisOrdering->getDofIndices(varID);
      // compute integrals:
      for (int cellIndex = 0; cellIndex<numCells; cellIndex++) {
        for (int basisOrdinal = 0; basisOrdinal < basisCardinality; basisOrdinal++) {
          int varDofIndex = varDofIndices[basisOrdinal];
          for (int ptIndex = 0; ptIndex < numPoints; ptIndex++) {
            values(cellIndex,varDofIndex) += ltValues(cellIndex,basisOrdinal,ptIndex);
          }
        }
      }
    }
  }
}

void LinearTerm::integrate(FieldContainer<double> &values, DofOrderingPtr thisOrdering, 
                           LinearTermPtr otherTerm, DofOrderingPtr otherOrdering, 
                           BasisCachePtr basisCache, bool forceBoundaryTerm) {
  // values has dimensions (numCells, thisFields, otherFields)
  
  int numCells = values.dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  set<int> otherIDs = otherTerm->varIDs();
  
  int rank = this->rank();
  TEST_FOR_EXCEPTION( rank != otherTerm->rank(), std::invalid_argument, "other and this ranks disagree." );
  
  set<int>::iterator thisIt;
  set<int>::iterator otherIt;
  
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > thisBasis, otherBasis;
  
  bool thisFluxOrTrace  = (     this->termType() == FLUX) || (     this->termType() == TRACE);
  bool otherFluxOrTrace = (otherTerm->termType() == FLUX) || (otherTerm->termType() == TRACE);

  bool boundaryTerm = thisFluxOrTrace || otherFluxOrTrace || forceBoundaryTerm;
  
  Teuchos::Array<int> ltValueDim;
  ltValueDim.push_back(numCells);
  ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
  ltValueDim.push_back(numPoints);
  
  // num "sides" for volume integral: 1...
  int numSides = boundaryTerm ? basisCache->cellTopology().getSideCount() : 1;
  for (int sideIndex = 0; sideIndex < numSides; sideIndex++) {
    int numPointsSide;
    if (boundaryTerm) { 
      numPointsSide = basisCache->getPhysicalCubaturePointsForSide(sideIndex).dimension(1);
    } else {
      numPointsSide = -1;      
    }
    
    for (otherIt= otherIDs.begin(); otherIt != otherIDs.end(); otherIt++) {
      int otherID = *otherIt;
      otherBasis = otherFluxOrTrace ? otherOrdering->getBasis(otherID,sideIndex) : otherOrdering->getBasis(otherID);
      int otherBasisCardinality = otherBasis->getCardinality();
      
      // set up values container for other
      Teuchos::Array<int> ltValueDim1 = ltValueDim;
      ltValueDim1[1] = otherBasisCardinality;
      for (int d=0; d<rank; d++) {
        ltValueDim1.push_back(spaceDim);
      }
      ltValueDim1[2] = boundaryTerm ? numPointsSide : numPoints;
      FieldContainer<double> otherValues(ltValueDim1);
      bool applyCubatureWeights = true;
      if (! boundaryTerm) {
        otherTerm->values(otherValues,otherID,otherBasis,basisCache,applyCubatureWeights);
      } else {
        otherTerm->values(otherValues,otherID,otherBasis,basisCache,applyCubatureWeights,sideIndex);
      }
      
      for (thisIt= _varIDs.begin(); thisIt != _varIDs.end(); thisIt++) {
        int thisID = *thisIt;
        thisBasis = thisFluxOrTrace ? thisOrdering->getBasis(thisID,sideIndex) : thisOrdering->getBasis(thisID);
        int thisBasisCardinality = thisBasis->getCardinality();
        
        // set up values container this term:
        Teuchos::Array<int> ltValueDim2 = ltValueDim1;
        ltValueDim2[1] = thisBasisCardinality;
        
        FieldContainer<double> thisValues(ltValueDim2);
        
        if (! boundaryTerm ) {
          this->values(thisValues,thisID,thisBasis,basisCache);
        } else {
          this->values(thisValues,thisID,thisBasis,basisCache,false,sideIndex); // false: don't apply cubature weights
          if ( this->termType() == FLUX ) {
            // we need to multiply thisValues' entries by the parity of the normal, since
            // the trial implicitly contains an outward normal, and we need to adjust for the fact
            // that the neighboring cells have opposite normal
            // thisValues should have dimensions (numCells,numFields,numCubPointsSide)
            int numFields = thisValues.dimension(1);
            int numPoints = thisValues.dimension(2);
            for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
              double parity = basisCache->getCellSideParities()(cellIndex,sideIndex);
              if (parity != 1.0) {  // otherwise, we can just leave things be...
                for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
                  for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
                    thisValues(cellIndex,fieldIndex,ptIndex) *= parity;
                  }
                }
              }
            }
          }
          // same thing, for otherTerm:
          if ( otherTerm->termType() == FLUX ) {
            int numFields = otherValues.dimension(1);
            int numPoints = otherValues.dimension(2);
            for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
              double parity = basisCache->getCellSideParities()(cellIndex,sideIndex);
              if (parity != 1.0) {  // otherwise, we can just leave things be...
                for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
                  for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
                    otherValues(cellIndex,fieldIndex,ptIndex) *= parity;
                  }
                }
              }
            }
          }
        }
        
        FieldContainer<double> miniMatrix( numCells, otherBasisCardinality, thisBasisCardinality );
        
        FunctionSpaceTools::integrate<double>(miniMatrix,otherValues,thisValues,COMP_CPP);
        
        vector<int> thisDofIndices = thisFluxOrTrace ? thisOrdering->getDofIndices(thisID,sideIndex)
                                                     : thisOrdering->getDofIndices(thisID);
        
        vector<int> otherDofIndices = otherFluxOrTrace ? otherOrdering->getDofIndices(otherID,sideIndex)
                                                       : otherOrdering->getDofIndices(otherID);
        
        // there may be a more efficient way to do this copying:
        for (int i=0; i < otherBasisCardinality; i++) {
//          int otherDofIndex = otherFluxOrTrace ? otherOrdering->getDofIndex(otherID,i,sideIndex)
//                                               : otherOrdering->getDofIndex(otherID,i);
          int otherDofIndex = otherDofIndices[i];
          for (int j=0; j < thisBasisCardinality; j++) {
            int thisDofIndex = thisDofIndices[j];
//            int thisDofIndex = thisFluxOrTrace ? thisOrdering->getDofIndex(thisID,j,sideIndex)
//                                               : thisOrdering->getDofIndex(thisID,j);
            for (unsigned k=0; k < numCells; k++) {
              double value = miniMatrix(k,i,j); // separate line for debugger inspection
              values(k,otherDofIndex,thisDofIndex) += value;
            }
          }
        }
      }
    }
  }
}

// compute the value of linearTerm for solution at the BasisCache points
// values shape: (C,P), (C,P,D), or (C,P,D,D)
void LinearTerm::evaluate(FieldContainer<double> &values, SolutionPtr solution, BasisCachePtr basisCache, 
                          bool applyCubatureWeights) {
  int sideIndex = basisCache->getSideIndex();
//  bool boundaryTerm = (sideIndex != -1);
  
  int valuesRankExpected = _rank + 2; // 2 for scalar, 3 for vector, etc.
  TEST_FOR_EXCEPTION( valuesRankExpected != values.rank(), std::invalid_argument,
                     "values FC does not have the expected rank" );
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  values.initialize(0.0);
  Teuchos::Array<int> scalarFunctionValueDim;
  scalarFunctionValueDim.append(numCells);
  scalarFunctionValueDim.append(numPoints);
  
  // (could tune things by pre-allocating this storage)
  FieldContainer<double> fValues;
  FieldContainer<double> solnValues;
  
  Teuchos::Array<int> vectorFunctionValueDim = scalarFunctionValueDim;
  vectorFunctionValueDim.append(spaceDim);
  Teuchos::Array<int> tensorFunctionValueDim = vectorFunctionValueDim;
  tensorFunctionValueDim.append(spaceDim);
  
  TEST_FOR_EXCEPTION( numCells != basisCache->getPhysicalCubaturePoints().dimension(0),
                     std::invalid_argument, "values FC numCells disagrees with cubature points container");
  TEST_FOR_EXCEPTION( numPoints != basisCache->getPhysicalCubaturePoints().dimension(1),
                     std::invalid_argument, "values FC numPoints disagrees with cubature points container");
  for (vector< LinearSummand >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
    LinearSummand ls = *lsIt;
    FunctionPtr f = ls.first;
    VarPtr var = ls.second;
    if (f->rank() == 0) {
      fValues.resize(scalarFunctionValueDim);
    } else if (f->rank() == 1) {
      fValues.resize(vectorFunctionValueDim);
    } else if (f->rank() == 2) {
      fValues.resize(tensorFunctionValueDim);
    } else {
      Teuchos::Array<int> fDim = tensorFunctionValueDim;
      for (int d=3; d < f->rank(); d++) {
        fDim.append(spaceDim);
      }
      fValues.resize(fDim);
    }
    
    if (var->rank() == 0) {
      solnValues.resize(scalarFunctionValueDim);
    } else if (var->rank() == 1) {
      solnValues.resize(vectorFunctionValueDim);
    } else if (var->rank() == 2) {
      solnValues.resize(tensorFunctionValueDim);
    } else {
      Teuchos::Array<int> solnDim = tensorFunctionValueDim;
      for (int d=3; d < var->rank(); d++) {
        solnDim.append(spaceDim);
      }
      solnValues.resize(solnDim);
    }
      
    f->values(fValues,basisCache);
    solution->solutionValues(solnValues,var->ID(),basisCache,
                             applyCubatureWeights,var->op());
    
    Teuchos::Array<int> fDim(fValues.rank());
    Teuchos::Array<int> solnDim(solnValues.rank());

    int entriesPerPoint = 1;
    bool scalarF = f->rank() == 0;
    int resultRank = scalarF ? var->rank() : f->rank();
    for (int d=0; d<resultRank; d++) {
      entriesPerPoint *= spaceDim;
    }
    Teuchos::Array<int> vDim( values.rank() );
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      fDim[0] = cellIndex; solnDim[0] = cellIndex; vDim[0] = cellIndex;
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        fDim[1] = ptIndex; solnDim[1] = ptIndex; vDim[1] = ptIndex;
        const double *fValue = &fValues[fValues.getEnumeration(fDim)];
        const double *solnValue = &solnValues[solnValues.getEnumeration(solnDim)];
        
        double *value = &values[values.getEnumeration(vDim)];
        
        for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
          *value += *fValue * *solnValue;
          value++;
          if (resultRank == 0) {
            // resultRank == 0 --> "dot" product; march along both f and soln
            fValue++;
            solnValue++;
          } else {
            // resultRank != 0 --> scalar guy stays fixed while we march over the higher-rank values
            if (scalarF) {
              solnValue++;
            } else {
              fValue++;
            }
          }
        }
      }
    }
  }
}

// compute the value of linearTerm for non-zero varID at the cubature points, for each basis function in basis
// values shape: (C,F,P), (C,F,P,D), or (C,F,P,D,D)
void LinearTerm::values(FieldContainer<double> &values, int varID, BasisPtr basis, BasisCachePtr basisCache, 
                        bool applyCubatureWeights, int sideIndex) {
  bool boundaryTerm = (sideIndex != -1);
  
  int valuesRankExpected = _rank + 3; // 3 for scalar, 4 for vector, etc.
  TEST_FOR_EXCEPTION( valuesRankExpected != values.rank(), std::invalid_argument,
                     "values FC does not have the expected rank" );
  int numCells = values.dimension(0);
  int numFields = values.dimension(1);
  int numPoints = values.dimension(2);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  values.initialize(0.0);
  Teuchos::Array<int> scalarFunctionValueDim;
  scalarFunctionValueDim.append(numCells);
  scalarFunctionValueDim.append(numPoints);
  
  // (could tune things by pre-allocating this storage)
  FieldContainer<double> fValues;
  
  Teuchos::Array<int> vectorFunctionValueDim = scalarFunctionValueDim;
  vectorFunctionValueDim.append(spaceDim);
  Teuchos::Array<int> tensorFunctionValueDim = vectorFunctionValueDim;
  tensorFunctionValueDim.append(spaceDim);
  
  TEST_FOR_EXCEPTION( numCells != basisCache->getPhysicalCubaturePoints().dimension(0),
                     std::invalid_argument, "values FC numCells disagrees with cubature points container");
  TEST_FOR_EXCEPTION( numFields != basis->getCardinality(),
                     std::invalid_argument, "values FC numFields disagrees with basis cardinality");
  if (! boundaryTerm) {
    TEST_FOR_EXCEPTION( numPoints != basisCache->getPhysicalCubaturePoints().dimension(1),
                       std::invalid_argument, "values FC numPoints disagrees with cubature points container");
  } else {
    TEST_FOR_EXCEPTION( numPoints != basisCache->getPhysicalCubaturePointsForSide(sideIndex).dimension(1),
                       std::invalid_argument, "values FC numPoints disagrees with cubature points container");      
  }
  for (vector< LinearSummand >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
    LinearSummand ls = *lsIt;
    if (ls.second->ID() == varID) {
      constFCPtr basisValues;
      if (applyCubatureWeights) {
        if (sideIndex == -1) {
          basisValues = basisCache->getTransformedWeightedValues(basis, ls.second->op());
        } else {
          // on sides, we use volume coords for test values
          bool useVolumeCoords = ls.second->varType() == TEST;
          basisValues = basisCache->getTransformedWeightedValues(basis, ls.second->op(), sideIndex, useVolumeCoords);
        }
      } else {
        if (sideIndex == -1) {
          basisValues = basisCache->getTransformedValues(basis, ls.second->op());
        } else {
          // on sides, we use volume coords for test values
          bool useVolumeCoords = ls.second->varType() == TEST;
          basisValues = basisCache->getTransformedValues(basis, ls.second->op(), sideIndex, useVolumeCoords);
        }
      }

      if ( ls.first->rank() == 0 ) { // scalar function -- we can speed things along in this case...
        // E.g. ConstantFunction::scalarMultiplyBasisValues() knows not to do anything at all if its value is 1.0...
        FieldContainer<double> weightedBasisValues = *basisValues; // weighted by the scalar function
        if (sideIndex == -1) {
          ls.first->scalarMultiplyBasisValues(weightedBasisValues,basisCache);
        } else {
          ls.first->scalarMultiplyBasisValues(weightedBasisValues,basisCache->getSideBasisCache(sideIndex));
        }
        for (int i=0; i<values.size(); i++) {
          values[i] += weightedBasisValues[i];
        }
        continue;
      }
      
      if (ls.first->rank() == 0) {
        fValues.resize(scalarFunctionValueDim);
      } else if (ls.first->rank() == 1) {
        fValues.resize(vectorFunctionValueDim);
      } else if (ls.first->rank() == 2) {
        fValues.resize(tensorFunctionValueDim);
      } else {
        Teuchos::Array<int> fDim = tensorFunctionValueDim;
        for (int d=3; d < ls.first->rank(); d++) {
          fDim.append(spaceDim);
        }
        fValues.resize(fDim);
      }
      
      if (sideIndex == -1) {
        ls.first->values(fValues,basisCache);
      } else {
        ls.first->values(fValues,basisCache->getSideBasisCache(sideIndex));
      }
      int numFields = basis->getCardinality();
      
      Teuchos::Array<int> fDim(fValues.rank());
      Teuchos::Array<int> bDim(basisValues->rank());
      
      // compute f * basisValues
      if ( ls.first->rank() == ls.second->rank() ) { // scalar result
        int entriesPerPoint = 1;
        int fRank = ls.first->rank();
        for (int d=0; d<fRank; d++) {
          entriesPerPoint *= spaceDim;
        }
        for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
          fDim[0] = cellIndex; bDim[0] = cellIndex;
          for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
            fDim[1] = ptIndex; bDim[2] = ptIndex;
            for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
              const double *fValue = &fValues[fValues.getEnumeration(fDim)];
              bDim[1] = fieldIndex;
              const double *bValue = &((*basisValues)[basisValues->getEnumeration(bDim)]);
              for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
                values(cellIndex,fieldIndex,ptIndex) += *fValue * *bValue;
                
                fValue++;
                bValue++;
              }
            }
          }
        }
      } else { // vector/tensor result
        // could pretty easily fold the scalar case above into the code below
        // (just change the logic in the pointer increments)
        int entriesPerPoint = 1;
        // now that we've changed so that we handle scalar function multiplication separately,
        // we don't hit this code for scalar functions.  I.e. scalarF == false always.
        bool scalarF = ls.first->rank() == 0;
        int resultRank = scalarF ? ls.second->rank() : ls.first->rank();
        for (int d=0; d<resultRank; d++) {
          entriesPerPoint *= spaceDim;
        }
        Teuchos::Array<int> vDim( values.rank() );
        for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
          fDim[0] = cellIndex; bDim[0] = cellIndex; vDim[0] = cellIndex;
          for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
            fDim[1] = ptIndex; bDim[2] = ptIndex; vDim[2] = ptIndex;
            for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
              const double *fValue = &fValues[fValues.getEnumeration(fDim)];
              bDim[1] = fieldIndex; vDim[1] = fieldIndex;
              const double *bValue = &(*basisValues)[basisValues->getEnumeration(bDim)];
              
              double *value = &values[values.getEnumeration(vDim)];
              
              for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
                *value += *fValue * *bValue;
                value++; 
                if (scalarF) {
                  bValue++;
                } else {
                  fValue++;
                }
              }
            }
          }
        }
      }
    }
  }
}

int LinearTerm::rank() const {   // 0 for scalar, 1 for vector, etc.
  return _rank; 
}

// operator overloading niceties:

LinearTerm& LinearTerm::operator=(const LinearTerm &rhs) {
  if ( this == &rhs ) {
    return *this;
  }
  _rank = rhs.rank();
  _summands = rhs.summands();
  _varIDs = rhs.varIDs();
  return *this;
}    

LinearTerm& LinearTerm::operator+=(const LinearTerm &rhs) {
  if (_rank == -1) { // we're empty -- adopt rhs's rank
    _rank = rhs.rank();
  }
  if (_rank != rhs.rank()) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "attempting to add terms of unlike rank.");
  }
  if (_termType == UNKNOWN_TYPE) {
    _termType = rhs.termType();
  }
  if (_termType != rhs.termType()) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "attempting to add terms of unlike type.");
  }
  _summands.insert(_summands.end(), rhs.summands().begin(), rhs.summands().end());
  _varIDs.insert( rhs.varIDs().begin(), rhs.varIDs().end() );
  return *this;
}

LinearTerm& LinearTerm::operator+=(VarPtr v) {
  this->addVar(1.0, v);
  return *this;
}

// operator overloading for syntax sugar:
LinearTermPtr operator+(LinearTermPtr a1, LinearTermPtr a2) {
  LinearTermPtr sum = Teuchos::rcp( new LinearTerm(*a1) );
  //  cout << "sum->rank(): " << sum->rank() << endl;
  //  cout << "sum->summands.size() before adding a2: " << sum->summands().size() << endl;
  *sum += *a2;
  //  cout << "sum->summands.size() after adding a2: " << sum->summands().size() << endl;
  return sum;
}

LinearTermPtr operator+(VarPtr v, LinearTermPtr a) {
  LinearTermPtr sum = Teuchos::rcp( new LinearTerm(*a) );
  *sum += v;
  return sum;
}

LinearTermPtr operator+(LinearTermPtr a, VarPtr v) {
  return v + a;
}

LinearTermPtr operator*(FunctionPtr f, VarPtr v) {
  return Teuchos::rcp( new LinearTerm(f, v) );
}

LinearTermPtr operator*(double weight, VarPtr v) {
  return Teuchos::rcp( new LinearTerm(weight, v) );
}

LinearTermPtr operator*(VarPtr v, double weight) {
  return weight * v;
}

LinearTermPtr operator*(vector<double> weight, VarPtr v) {
  return Teuchos::rcp( new LinearTerm(weight, v) );
}

LinearTermPtr operator*(VarPtr v, vector<double> weight) {
  return weight * v;
}

LinearTermPtr operator*(FunctionPtr f, LinearTermPtr a) {
  LinearTermPtr lt = Teuchos::rcp( new LinearTerm );
  
  for (vector< LinearSummand >::const_iterator lsIt = a->summands().begin(); lsIt != a->summands().end(); lsIt++) {
    LinearSummand ls = *lsIt;
    FunctionPtr lsWeight = ls.first;
    FunctionPtr newWeight = f * lsWeight;
    VarPtr var = ls.second;
    *lt += *(newWeight * var);
  }
  return lt;
}

LinearTermPtr operator/(LinearTermPtr a, FunctionPtr f) {
  LinearTermPtr lt = Teuchos::rcp( new LinearTerm );
  
  for (vector< LinearSummand >::const_iterator lsIt = lt->summands().begin(); lsIt != lt->summands().end(); lsIt++) {
    LinearSummand ls = *lsIt;
    FunctionPtr lsWeight = ls.first;
    FunctionPtr newWeight = lsWeight / f;
    VarPtr var = ls.second;
    *lt += *(newWeight * var);
  }
  return lt;
}

LinearTermPtr operator/(VarPtr v, FunctionPtr scalarFunction) {
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  return (one / scalarFunction) * v;
}

LinearTermPtr operator+(VarPtr v1, VarPtr v2) {
  return 1.0 * v1 + 1.0 * v2;
}

LinearTermPtr operator/(VarPtr v, double weight) {
  return (1.0 / weight) * v;
}

LinearTermPtr operator-(VarPtr v1, VarPtr v2) {
  return v1 + (-1.0) * v2;
}

LinearTermPtr operator-(VarPtr v) {
  return (-1.0) * v;
}

LinearTermPtr operator-(LinearTermPtr a) {
  return Teuchos::rcp( new ConstantScalarFunction(-1.0) ) * a;
}

LinearTermPtr operator-(LinearTermPtr a, VarPtr v) {
  return a + (-1.0) * v;
}

LinearTermPtr operator-(LinearTermPtr a1, LinearTermPtr a2) {
  return a1 + -a2;
}