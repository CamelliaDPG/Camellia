//
//  LinearTerm.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "LinearTerm.h"
#include "Mesh.h"
#include "Solution.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#endif

#include "Function.h"

typedef pair< FunctionPtr, VarPtr > LinearSummand;

bool linearSummandIsBoundaryValueOnly(LinearSummand &ls) {
  bool opInvolvesNormal = (ls.second->op() == IntrepidExtendedTypes::OP_TIMES_NORMAL)   || 
  (ls.second->op() == IntrepidExtendedTypes::OP_TIMES_NORMAL_X) || 
  (ls.second->op() == IntrepidExtendedTypes::OP_TIMES_NORMAL_Y) || 
  (ls.second->op() == IntrepidExtendedTypes::OP_TIMES_NORMAL_Z) || 
  (ls.second->op() == IntrepidExtendedTypes::OP_CROSS_NORMAL)   || 
  (ls.second->op() == IntrepidExtendedTypes::OP_DOT_NORMAL);
  bool boundaryOnlyFunction = ls.first->boundaryValueOnly();
  return boundaryOnlyFunction || (ls.second->varType()==FLUX) || (ls.second->varType()==TRACE) || opInvolvesNormal;
}

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
    TEUCHOS_TEST_FOR_EXCEPTION( true, std::invalid_argument, "Unhandled rank combination.");
  }
  if (_rank == -1) { // LinearTerm's rank is unassigned
    _rank = rank;
  }
  if (_rank != rank) {
    TEUCHOS_TEST_FOR_EXCEPTION( true, std::invalid_argument, "Attempting to add terms of unlike rank." );
  }
  // check type:
  if (_termType == UNKNOWN_TYPE) {
    _termType = var->varType();
  }
  if (_termType != var->varType() ) {
    TEUCHOS_TEST_FOR_EXCEPTION( true, std::invalid_argument, "Attempting to add terms of differing type." );
  }
  _varIDs.insert(var->ID());
  if (weight->isZero()) return; // in that case, we can skip the actual adding...
  _summands.push_back( make_pair( weight, var ) );
}

void LinearTerm::addVar(double weight, VarPtr var) {
  FunctionPtr weightFn = (weight != 1.0) ? Teuchos::rcp( new ConstantScalarFunction(weight) ) 
                                         : Teuchos::rcp( new ConstantScalarFunction(weight, "") );
  // (suppress display of 1.0 weights)
  addVar( weightFn, var );
}

void LinearTerm::addVar(vector<double> vector_weight, VarPtr var) { // dots weight vector with vector var, makes a vector out of a scalar var
  FunctionPtr weightFn = Teuchos::rcp( new ConstantVectorFunction(vector_weight) );
  addVar( weightFn, var );
}

string LinearTerm::displayString() {
  ostringstream dsStream;
  bool first = true;
  for (vector< LinearSummand >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
    if ( ! first ) {
      dsStream << " + ";
    }
    LinearSummand ls = *lsIt;
    FunctionPtr f = ls.first;
    VarPtr var = ls.second;
    dsStream << f->displayString() << " " << var->displayString();
    first = false;
  }
  return dsStream.str();
}

const set<int> & LinearTerm::varIDs() const {
  return _varIDs;
}

VarType LinearTerm::termType() const { 
  return _termType; 
}

// some notes on the design of integrate:
// each linear summand can be either an element-boundary-only term, or one that's defined on the
// whole element.  For boundary-only terms, we integrate along each side.  The summands that are
// defined on the whole element are integrated over the element interior, unless forceBoundaryTerm
// is set to true, in which case these too will be integrated over the boundary.  One place where 
// this is appropriate is when a test function LinearTerm is being integrated against a trace or
// flux in a bilinear form: the test function is defined on the whole element, but the traces and
// fluxes are only defined on the element boundary.
// TODO: make the code below match the description above.
void LinearTerm::integrate(FieldContainer<double> &values, DofOrderingPtr thisOrdering,
                           BasisCachePtr basisCache, bool forceBoundaryTerm, bool sumInto) {
  // values has dimensions (numCells, thisFields)
  if (!sumInto) values.initialize();
  
  set<int> varIDs = this->varIDs();
  
  int numCells  = basisCache->getPhysicalCubaturePoints().dimension(0);
  
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > basis;
  bool thisFluxOrTrace  = (this->termType() == FLUX) || (this->termType() == TRACE);
  bool boundaryTerm = thisFluxOrTrace || forceBoundaryTerm || basisCache->isSideCache();
  
  // boundaryTerm ==> even volume summands should be restricted to boundary
  // (if thisFluxOrTrace, there won't be any volume summands, so the above will hold trivially)
  // note that ! boundaryTerm does NOT imply that all terms are volume terms.  boundaryTerm means that
  // we ONLY evaluate on the boundary; !boundaryTerm allows the possibility of "mixed-type" terms
  int numSides = basisCache->cellTopology().getSideCount();
  
  Teuchos::Array<int> ltValueDim;
  ltValueDim.push_back(numCells);
  ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
  ltValueDim.push_back(0); // # points -- empty until we know whether we're on side
  FieldContainer<double> ltValues;
  
  for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++) {
    int varID = *varIt;
    if (! boundaryTerm ) {
      vector<int> varDofIndices = thisOrdering->getDofIndices(varID);
      
      // first, compute volume integral
      int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
      
      basis = thisOrdering->getBasis(varID);
      int basisCardinality = basis->getCardinality();
      ltValueDim[1] = basisCardinality;
      ltValueDim[2] = numPoints;
      ltValues.resize(ltValueDim);
      bool applyCubatureWeights = true;
      this->values(ltValues, varID, basis, basisCache, applyCubatureWeights);
      
      // compute integrals:
      for (int cellIndex = 0; cellIndex<numCells; cellIndex++) {
        for (int basisOrdinal = 0; basisOrdinal < basisCardinality; basisOrdinal++) {
          int varDofIndex = varDofIndices[basisOrdinal];
          for (int ptIndex = 0; ptIndex < numPoints; ptIndex++) {
            values(cellIndex,varDofIndex) += ltValues(cellIndex,basisOrdinal,ptIndex);
          }
        }
      }
      
      // now, compute boundary integrals
      for (int sideIndex = 0; sideIndex < numSides; sideIndex++ ) {
        BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideIndex);
        if ( sideBasisCache.get() != NULL ) {
          numPoints = sideBasisCache->getPhysicalCubaturePoints().dimension(1);
          ltValueDim[2] = numPoints;
          ltValues.resize(ltValueDim);
          if (thisOrdering->getNumSidesForVarID(varID) > 1) {
            varDofIndices = thisOrdering->getDofIndices(varID,sideIndex);
          }
          bool naturalBoundaryValuesOnly = true; // don't restrict volume summands to boundary
          this->values(ltValues, varID, basis, sideBasisCache, applyCubatureWeights, naturalBoundaryValuesOnly);
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
    } else {
      BasisCachePtr volumeCache;
      int startSideIndex, endSideIndex;
      if ( basisCache->isSideCache() ) {
        // just one side, then:
        startSideIndex = basisCache->getSideIndex();
        endSideIndex = startSideIndex;
        volumeCache = basisCache->getVolumeBasisCache();
      } else {
        // all sides:
        startSideIndex = 0;
        endSideIndex = numSides - 1;
        volumeCache = basisCache;
      }
      for (int sideIndex = startSideIndex; sideIndex <= endSideIndex; sideIndex++ ) {
        int numPoints = volumeCache->getPhysicalCubaturePointsForSide(sideIndex).dimension(1);
        
        basis = thisFluxOrTrace ? thisOrdering->getBasis(varID,sideIndex) 
                                : thisOrdering->getBasis(varID);
        int basisCardinality = basis->getCardinality();
        ltValueDim[1] = basisCardinality;
        ltValueDim[2] = numPoints;
        ltValues.resize(ltValueDim);
        BasisCachePtr sideBasisCache = volumeCache->getSideBasisCache(sideIndex);
        bool applyCubatureWeights = true;
        bool naturalBoundaryValuesOnly = false; // DO include volume summands restricted to boundary
        this->values(ltValues, varID, basis, sideBasisCache, applyCubatureWeights, naturalBoundaryValuesOnly);
        if ( this->termType() == FLUX ) {
          multiplyFluxValuesByParity(ltValues, sideBasisCache);
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
}

void LinearTerm::integrate(FieldContainer<double> &values, 
                           LinearTermPtr u, DofOrderingPtr uOrdering, 
                           LinearTermPtr v, DofOrderingPtr vOrdering, 
                           BasisCachePtr basisCache, bool sumInto) {
  // will integrate either in volume or along a side, depending on the type of BasisCache passed in
  if (!sumInto) values.initialize();
  
  if (u->isZero() || v->isZero()) return; // nothing to do, then.
  
  TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(1) != uOrdering->totalDofs(), std::invalid_argument, "values.dim(1) != uOrdering->totalDofs()");
  TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(2) != vOrdering->totalDofs(), std::invalid_argument, "values.dim(2) != vOrdering->totalDofs()");
  
  // values has dimensions (numCells, uFields, vFields)
  int numCells = values.dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  
  Teuchos::Array<int> ltValueDim;
  ltValueDim.push_back(numCells);
  ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
  ltValueDim.push_back(numPoints);
  
  for (int i=0; i<u->rank(); i++) {
    ltValueDim.push_back(spaceDim);
  }
  
  set<int> uIDs = u->varIDs();
  set<int> vIDs = v->varIDs();
  set<int>::iterator uIt, vIt;
  for (uIt = uIDs.begin(); uIt != uIDs.end(); uIt++) {
    int uID = *uIt;
    // the DofOrdering needs a sideIndex argument; this is 0 for volume bases.
    bool uVolVar = (uOrdering->getNumSidesForVarID(uID) == 1);
    int uSideIndex = uVolVar ? 0 : basisCache->getSideIndex();
    
    BasisPtr uBasis = uOrdering->getBasis(uID,uSideIndex);
    int uBasisCardinality = uBasis->getCardinality();
    ltValueDim[1] = uBasisCardinality;
    FieldContainer<double> uValues(ltValueDim);
    bool applyCubatureWeights = true, dontApplyCubatureWeights = false;
    u->values(uValues,uID,uBasis,basisCache,applyCubatureWeights);
    
//    cout << "uValues for u = " << u->displayString() << ":" << endl;
//    cout << uValues;
//    double debugSum = 0.0;
//    for (int i=0; i<uValues.size(); i++) {
//      debugSum += uValues[i];
//    }
//    cout << "uValues debug sum for u = " << u->displayString() << ": " << debugSum << endl;
    
    if ( u->termType() == FLUX ) {
      // we need to multiply uValues' entries by the parity of the normal, since
      // the trial implicitly contains an outward normal, and we need to adjust for the fact
      // that the neighboring cells have opposite normal...
      multiplyFluxValuesByParity(uValues, basisCache); // basisCache had better be a side cache!
    }
    
    for (vIt = vIDs.begin(); vIt != vIDs.end(); vIt++) {
      int vID = *vIt;
      bool vVolVar = (vOrdering->getNumSidesForVarID(vID) == 1);
      int vSideIndex = vVolVar ? 0 : basisCache->getSideIndex();
      BasisPtr vBasis = vOrdering->getBasis(vID,vSideIndex);
      int vBasisCardinality = vBasis->getCardinality();
      ltValueDim[1] = vBasisCardinality;
      FieldContainer<double> vValues(ltValueDim);
      v->values(vValues, vID, vBasis, basisCache, dontApplyCubatureWeights);
      
//      cout << "vValues for v = " << v->displayString() << ":" << endl;
//      cout << vValues;
      
      // same flux consideration, for the vValues
      if ( v->termType() == FLUX ) {
        multiplyFluxValuesByParity(vValues, basisCache);
      }
      
      FieldContainer<double> miniMatrix( numCells, uBasisCardinality, vBasisCardinality );
      
      FunctionSpaceTools::integrate<double>(miniMatrix,uValues,vValues,COMP_CPP);
      
//      cout << "uValues:" << endl << uValues;
//      cout << "vValues:" << endl << vValues;
//      cout << "miniMatrix:" << endl << miniMatrix;
      
      vector<int> uDofIndices = uOrdering->getDofIndices(uID,uSideIndex);
      vector<int> vDofIndices = vOrdering->getDofIndices(vID,vSideIndex);
      
      // there may be a more efficient way to do this copying:
      for (int i=0; i < uBasisCardinality; i++) {
        int uDofIndex = uDofIndices[i];
        for (int j=0; j < vBasisCardinality; j++) {
          int vDofIndex = vDofIndices[j];
          for (unsigned k=0; k < numCells; k++) {
            double value = miniMatrix(k,i,j); // separate line for debugger inspection
            values(k,uDofIndex,vDofIndex) += value;
//            cout << "values(" << k << ", " << uDofIndex << ", " << vDofIndex << ") += " << value << endl;
          }
        }
      }
    }
  }
}

void LinearTerm::integrate(FieldContainer<double> &values, DofOrderingPtr thisOrdering, 
                           LinearTermPtr otherTerm, DofOrderingPtr otherOrdering, 
                           BasisCachePtr basisCache, bool forceBoundaryTerm, bool sumInto) {
  // values has dimensions (numCells, otherFields, thisFields)
  // note that this means when we call the private integrate, we need to use otherTerm as the first LinearTerm argument
  if (!sumInto) values.initialize();
  
  // define variables:
  //  u - the non-boundary-only part of this
  // du - the boundary-only part of this
  //  v - the non-boundary-only part of otherTerm
  // dv - the boundary-only part of otherTerm
  
  // we are then computing (u + du, v + dv) where e.g. (du,v) will be integrated along boundary
  // so the only non-boundary term is (u,v), and that only comes into it if forceBoundaryTerm is false.
  
  if (basisCache->isSideCache() && !forceBoundaryTerm) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: forceBoundaryTerm is false but basisCache is a sideBasisCache...");
  }
  
  LinearTermPtr thisPtr = Teuchos::rcp( this, false );
  
  if (basisCache->isSideCache()) {
    // then we just integrate along the one side:
    integrate(values, otherTerm, otherOrdering, thisPtr, thisOrdering, basisCache);
    return;
  } else if (forceBoundaryTerm) {
    // then we don't need to worry about splitting into boundary and non-boundary parts,
    // but we do need to loop over the sides:
    int numSides = basisCache->cellTopology().getSideCount();
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      integrate(values, otherTerm, otherOrdering, thisPtr, thisOrdering, basisCache->getSideBasisCache(sideIndex));
    }
  } else {
    LinearTermPtr thisBoundaryOnly = this->getBoundaryOnlyPart();
    LinearTermPtr thisNonBoundaryOnly = this->getNonBoundaryOnlyPart();
    LinearTermPtr otherBoundaryOnly = otherTerm->getBoundaryOnlyPart();
    LinearTermPtr otherNonBoundaryOnly = otherTerm->getNonBoundaryOnlyPart();
    
    // volume integration first:  ( (u,v) from above )
    integrate(values, otherNonBoundaryOnly, otherOrdering, thisNonBoundaryOnly, thisOrdering, basisCache);
    
    // sides:
    // (u + du, v + dv) - (u,v) = (u + du, dv) + (du, v)
    int numSides = basisCache->cellTopology().getSideCount();
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      // (u + du, dv)
      integrate(values, otherBoundaryOnly, otherOrdering, thisPtr, thisOrdering, basisCache->getSideBasisCache(sideIndex));
      // (du, v)
      integrate(values, otherNonBoundaryOnly, otherOrdering, thisBoundaryOnly, thisOrdering, basisCache->getSideBasisCache(sideIndex));
    }
  }
}

// integrate this against otherTerm, where otherVar == fxn
void LinearTerm::integrate(FieldContainer<double> &values, DofOrderingPtr thisOrdering, 
                           LinearTermPtr otherTerm, VarPtr otherVar, FunctionPtr fxn, 
                           BasisCachePtr basisCache, bool forceBoundaryTerm) {
  // values has dimensions (numCells, thisFields)
  
  map<int, FunctionPtr> otherVarMap;
  otherVarMap[otherVar->ID()] = fxn;
  FunctionPtr otherFxnBoundaryPart = otherTerm->evaluate(otherVarMap, true);
  FunctionPtr otherFxnVolumePart = otherTerm->evaluate(otherVarMap, false);
  
  LinearTermPtr thisPtr = Teuchos::rcp(this, false);
  LinearTermPtr lt = otherFxnVolumePart * thisPtr + otherFxnBoundaryPart * thisPtr;
  
  lt->integrate(values, thisOrdering, basisCache, forceBoundaryTerm);
}

bool LinearTerm::isZero() const { // true if the LinearTerm is identically zero
  // DEBUGGING test: pretend we're NEVER zero
  //return false;
  for (vector< LinearSummand >::const_iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
    LinearSummand ls = *lsIt;
    FunctionPtr f = ls.first;
    if (! f->isZero() ) {
      return false;
    }
  }
  return true;
}

// compute the value of linearTerm for solution at the BasisCache points
// values shape: (C,P), (C,P,D), or (C,P,D,D)
// TODO: consider rewriting this to set up a map varID->simpleSolutionFxn (where SimpleSolutionFunction is a
//       new class that is just the Solution evaluated at a given varID, might support non-value op as well).
// (The SimpleSolutionFunction class is now written, though untested.  What remains is to add something such that
//  LinearTerm can determine its OP_VALUE VarPtrs so that it can construct the right map.  It would probably suffice
//  if Var learned about its "base", and then in addition to tracking the varIDs set here, we also tracked vars.)
void LinearTerm::evaluate(FieldContainer<double> &values, SolutionPtr solution, BasisCachePtr basisCache, 
                          bool applyCubatureWeights) {
  int sideIndex = basisCache->getSideIndex();
//  bool boundaryTerm = (sideIndex != -1);
  
  int valuesRankExpected = _rank + 2; // 2 for scalar, 3 for vector, etc.
  TEUCHOS_TEST_FOR_EXCEPTION( valuesRankExpected != values.rank(), std::invalid_argument,
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
  
  TEUCHOS_TEST_FOR_EXCEPTION( numCells != basisCache->getPhysicalCubaturePoints().dimension(0),
                     std::invalid_argument, "values FC numCells disagrees with cubature points container");
  TEUCHOS_TEST_FOR_EXCEPTION( numPoints != basisCache->getPhysicalCubaturePoints().dimension(1),
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

LinearTermPtr LinearTerm::getBoundaryOnlyPart() {
  return this->getPart(true);
}

LinearTermPtr LinearTerm::getNonBoundaryOnlyPart() {
  return this->getPart(false);
}

LinearTermPtr LinearTerm::getPart(bool boundaryOnlyPart) {
  LinearTermPtr lt = Teuchos::rcp( new LinearTerm );
  for (vector< LinearSummand >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
    LinearSummand ls = *lsIt;
    if ( linearSummandIsBoundaryValueOnly(ls) && !boundaryOnlyPart) {
      continue;
    } else if (!linearSummandIsBoundaryValueOnly(ls) && boundaryOnlyPart) {
      continue;
    }
    FunctionPtr f = ls.first;
    VarPtr var = ls.second;
    
    if (lt.get()) {
      lt = lt + f * var;
    } else {
      lt = f * var;
    }
  }
  return lt;
}

LinearTermPtr LinearTerm::getPartMatchingVariable( VarPtr varToMatch ) {
  LinearTermPtr lt = Teuchos::rcp( new LinearTerm );
  for (vector< LinearSummand >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
    LinearSummand ls = *lsIt;
    
    FunctionPtr f = ls.first;
    VarPtr var = ls.second;
    
    if (var->ID() == varToMatch->ID()) {
      if (lt.get()) {
        lt = lt + f * var;
      } else {
        lt = f * var;
      }
    }
  }
  return lt;
}

FunctionPtr LinearTerm::evaluate(map< int, FunctionPtr> &varFunctions, bool boundaryPart) {
  // NOTE that if boundaryPart is false, then we exclude terms that are defined only on the boundary
  // and if boundaryPart is true, then we exclude terms that are defined everywhere
  // so that the whole LinearTerm is the sum of the two options
  FunctionPtr fxn = Function::null();
  vector< LinearSummand > summands = this->getPart(boundaryPart)->summands();
  for (vector< LinearSummand >::iterator lsIt = summands.begin(); lsIt != summands.end(); lsIt++) {
    LinearSummand ls = *lsIt;
    FunctionPtr f = ls.first;
    VarPtr var = ls.second;
    
    // if there isn't an entry for var, we take it to be zero:
    if (varFunctions.find(var->ID()) == varFunctions.end()) continue;
    
    FunctionPtr varEvaluation = Function::op(varFunctions[var->ID()],var->op());
    if (fxn.get()) {
      fxn = fxn + f * varEvaluation;
    } else {
      fxn = f * varEvaluation;
    }
  }
  if (!fxn.get() && (this->rank()==0)) {
    fxn = Function::zero();
  }
  return fxn;
}

void LinearTerm::multiplyFluxValuesByParity(FieldContainer<double> &fluxValues, BasisCachePtr sideBasisCache) {
  int numCells  = fluxValues.dimension(0);
  int numFields = fluxValues.dimension(1);
  int numPoints = fluxValues.dimension(2);
  int sideIndex = sideBasisCache->getSideIndex();
  BasisCachePtr volumeCache = sideBasisCache->getVolumeBasisCache();
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    double parity = volumeCache->getCellSideParities()(cellIndex,sideIndex);
    if (parity != 1.0) {  // otherwise, we can just leave things be...
      for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          fluxValues(cellIndex,fieldIndex,ptIndex) *= parity;
        }
      }
    }
  }
}

// compute the value of linearTerm for non-zero varID at the cubature points, for each basis function in basis
// values shape: (C,F,P), (C,F,P,D), or (C,F,P,D,D)
void LinearTerm::values(FieldContainer<double> &values, int varID, BasisPtr basis, BasisCachePtr basisCache, 
                        bool applyCubatureWeights, bool naturalBoundaryTermsOnly) {
  int sideIndex = basisCache->getSideIndex();
  
  int valuesRankExpected = _rank + 3; // 3 for scalar, 4 for vector, etc.
  TEUCHOS_TEST_FOR_EXCEPTION( valuesRankExpected != values.rank(), std::invalid_argument,
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
  
  TEUCHOS_TEST_FOR_EXCEPTION( numCells != basisCache->getPhysicalCubaturePoints().dimension(0),
                     std::invalid_argument, "values FC numCells disagrees with cubature points container");
  TEUCHOS_TEST_FOR_EXCEPTION( numFields != basis->getCardinality(),
                     std::invalid_argument, "values FC numFields disagrees with basis cardinality");
  TEUCHOS_TEST_FOR_EXCEPTION( numPoints != basisCache->getPhysicalCubaturePoints().dimension(1),
                     std::invalid_argument, "values FC numPoints disagrees with cubature points container");
  for (vector< LinearSummand >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
    LinearSummand ls = *lsIt;
    // skip if this is a volume term, and we're only interested in the pure-boundary terms
    if (naturalBoundaryTermsOnly && !linearSummandIsBoundaryValueOnly(ls)) {
      continue;
    }
    // skip if this is a boundary term, and we're doing a volume integration:
    if ((sideIndex == -1) && linearSummandIsBoundaryValueOnly(ls)) {
      continue;
    }
    if (ls.second->ID() == varID) {
      constFCPtr basisValues;
      if (applyCubatureWeights) {
        // on sides, we use volume coords for test and field values
        bool useVolumeCoords = (sideIndex != -1) && ((ls.second->varType() == TEST) || (ls.second->varType() == FIELD));
        basisValues = basisCache->getTransformedWeightedValues(basis, ls.second->op(), useVolumeCoords);
      } else {
        // on sides, we use volume coords for test and field values
        bool useVolumeCoords = (sideIndex != -1) && ((ls.second->varType() == TEST) || (ls.second->varType() == FIELD));
        basisValues = basisCache->getTransformedValues(basis, ls.second->op(), useVolumeCoords);
      }

      if ( ls.first->rank() == 0 ) { // scalar function -- we can speed things along in this case...
        // E.g. ConstantFunction::scalarMultiplyBasisValues() knows not to do anything at all if its value is 1.0...
        FieldContainer<double> weightedBasisValues = *basisValues; // weighted by the scalar function
        ls.first->scalarMultiplyBasisValues(weightedBasisValues,basisCache);
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
      
      ls.first->values(fValues,basisCache);
      
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

// compute the value of linearTerm for non-zero varID at the cubature points, for var == fxn
// values shape: (C,P), (C,P,D), or (C,P,D,D)
// TODO: refactor this and the other values() to reduce code redundancy
void LinearTerm::values(FieldContainer<double> &values, int varID, FunctionPtr fxn, BasisCachePtr basisCache, 
                        bool applyCubatureWeights, bool naturalBoundaryTermsOnly) {
  int sideIndex = basisCache->getSideIndex();
  
  int valuesRankExpected = _rank + 2; // 2 for scalar, 3 for vector, etc.
  TEUCHOS_TEST_FOR_EXCEPTION( valuesRankExpected != values.rank(), std::invalid_argument,
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
  
  Teuchos::Array<int> vectorFunctionValueDim = scalarFunctionValueDim;
  vectorFunctionValueDim.append(spaceDim);
  Teuchos::Array<int> tensorFunctionValueDim = vectorFunctionValueDim;
  tensorFunctionValueDim.append(spaceDim);
  
  int numFields = 1; // when we're pretending to be a basis
  Teuchos::Array<int> fxnValueDim, fxnValueAsBasisDim; // adds numFields = 1 to imitate what happens as basis
  if (fxn->rank() == 0) {
    fxnValueDim = scalarFunctionValueDim;
  } else if (fxn->rank() == 1) {
    fxnValueDim = vectorFunctionValueDim;
  } else if (fxn->rank() == 2) {
    fxnValueDim = tensorFunctionValueDim;
  }
  fxnValueAsBasisDim = fxnValueDim;
  fxnValueAsBasisDim.insert(fxnValueDim.begin()+1, numFields); // (C, F, ...)
  
  TEUCHOS_TEST_FOR_EXCEPTION( numCells != basisCache->getPhysicalCubaturePoints().dimension(0),
                     std::invalid_argument, "values FC numCells disagrees with cubature points container");
  TEUCHOS_TEST_FOR_EXCEPTION( numPoints != basisCache->getPhysicalCubaturePoints().dimension(1),
                     std::invalid_argument, "values FC numPoints disagrees with cubature points container");
  for (vector< LinearSummand >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
    LinearSummand ls = *lsIt;
    // skip if this is a volume term, and we're only interested in the pure-boundary terms
    if (naturalBoundaryTermsOnly && !linearSummandIsBoundaryValueOnly(ls)) {
      continue;
    }
    // skip if this is a boundary term, and we're doing a volume integration:
    if ((sideIndex == -1) && linearSummandIsBoundaryValueOnly(ls)) {
      continue;
    }
    if (ls.second->ID() == varID) {
      FieldContainer<double> fxnValues(fxnValueDim);
      fxn->values(fxnValues, ls.second->op(), basisCache); // should always use the volume coords (compare with other LinearTerm::values() function)
      if (applyCubatureWeights) {
        // TODO: apply cubature weights!!
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "still need to implement cubature weighting");
      }
      
      if ( ls.first->rank() == 0 ) { // scalar function -- we can speed things along in this case...
        // E.g. ConstantFunction::scalarMultiplyBasisValues() knows not to do anything at all if its value is 1.0...
        fxnValues.resize(fxnValueAsBasisDim);
        ls.first->scalarMultiplyBasisValues(fxnValues,basisCache);
        for (int i=0; i<values.size(); i++) {
          values[i] += fxnValues[i];
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
      
      ls.first->values(fValues,basisCache);
            
      Teuchos::Array<int> fDim(fValues.rank()); // f is the functional weight -- fxn is the function substituted for the variable
      Teuchos::Array<int> fxnDim(fxnValues.rank());
      
      // compute f * basisValues
      if ( ls.first->rank() == ls.second->rank() ) { // scalar result
        int entriesPerPoint = 1;
        int fRank = ls.first->rank();
        for (int d=0; d<fRank; d++) {
          entriesPerPoint *= spaceDim;
        }
        for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
          fDim[0] = cellIndex; fxnDim[0] = cellIndex;
          for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
            fDim[1] = ptIndex; fxnDim[1] = ptIndex;
            const double *fValue = &fValues[fValues.getEnumeration(fDim)];
            const double *fxnValue = &(fxnValues[fxnValues.getEnumeration(fxnDim)]);
            for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
              values(cellIndex,ptIndex) += *fValue * *fxnValue;
              
              fValue++;
              fxnValue++;
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
          fDim[0] = cellIndex; fxnDim[0] = cellIndex; vDim[0] = cellIndex;
          for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
            fDim[1] = ptIndex; fxnDim[1] = ptIndex; vDim[1] = ptIndex;
            const double *fValue = &fValues[fValues.getEnumeration(fDim)];
            const double *fxnValue = &(fxnValues)[fxnValues.getEnumeration(fxnDim)];
            
            double *value = &values[values.getEnumeration(vDim)];
            
            for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
              *value += *fValue * *fxnValue;
              value++; 
              if (scalarF) {
                fxnValue++;
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

int LinearTerm::rank() const {   // 0 for scalar, 1 for vector, etc.
  return _rank; 
}

// added by Jesse --------------------

LinearTermPtr LinearTerm::rieszRep(VarPtr v){
  LinearTermPtr errorComponent = Teuchos::rcp( new LinearTerm);
  return errorComponent; // WARNING - FINISH THIS
}

void LinearTerm::computeRieszRep(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<DPGInnerProduct> ip){
  int numProcs=1;
  int rank=0;
  
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif

  computeRieszRHS(mesh);

  vector<ElementTypePtr> elemTypes = mesh->elementTypes(rank);
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    BasisCachePtr ipBasisCache = Teuchos::rcp(new BasisCache(elemTypePtr,mesh,true));
    
    Teuchos::RCP<DofOrdering> testOrdering = elemTypePtr->testOrderPtr;
    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodes(elemTypePtr);
    
    vector<Teuchos::RCP<Element> > elemsInPartitionOfType = mesh->elementsOfType(rank, elemTypePtr);
    int numCells = physicalCellNodes.dimension(0);
    int numTestDofs = testOrdering->totalDofs();
    
    FieldContainer<double> ipMatrix(numCells,numTestDofs,numTestDofs);
    
    // determine cellIDs
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      int cellID = mesh->cellID(elemTypePtr, cellIndex, rank);
      cellIDs.push_back(cellID);
    }    
    ipBasisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,ip->hasBoundaryTerms());
    
    ip->computeInnerProductMatrix(ipMatrix,testOrdering, ipBasisCache);
    FieldContainer<double> rieszRepresentation(numCells,numTestDofs);
    
    Epetra_SerialDenseSolver solver;
    
    for (int localCellIndex=0; localCellIndex<numCells; localCellIndex++ ) {
      
      Epetra_SerialDenseMatrix ipMatrixT(Copy, &ipMatrix(localCellIndex,0,0),
                                         ipMatrix.dimension(2), // stride -- fc stores in row-major order (a.o.t. SDM)
                                         ipMatrix.dimension(2),ipMatrix.dimension(1));
      
      Epetra_SerialDenseMatrix rhs(Copy, & (_rieszRHSForElementType[elemTypePtr.get()](localCellIndex,0)),
                                   _rieszRHSForElementType[elemTypePtr.get()].dimension(1), // stride
                                   _rieszRHSForElementType[elemTypePtr.get()].dimension(1), 1);
     
      Epetra_SerialDenseMatrix representationMatrix(numTestDofs,1);
      
      solver.SetMatrix(ipMatrixT);
      int success = solver.SetVectors(representationMatrix, rhs);
            
      bool equilibrated = false;
      if ( solver.ShouldEquilibrate() ) {
        solver.EquilibrateMatrix();
        solver.EquilibrateRHS();
        equilibrated = true;
      }
      
      success = solver.Solve();      
      
      if (equilibrated) {
        success = solver.UnequilibrateLHS();
      }    
      
      for (int i=0; i<numTestDofs; i++) {
        rieszRepresentation(localCellIndex,i) = representationMatrix(i,0);
      }
    }
    _rieszRepresentationForElementType[elemTypePtr.get()] = rieszRepresentation;
  }
}

void LinearTerm::computeRieszRHS(Teuchos::RCP<Mesh> mesh){
  int numProcs=1;
  int rank=0;
  
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  vector<ElementTypePtr> elemTypes = mesh->elementTypes(rank);  
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    
    Teuchos::RCP<DofOrdering> trialOrdering = elemTypePtr->trialOrderPtr;
    Teuchos::RCP<DofOrdering> testOrdering = elemTypePtr->testOrderPtr;

    vector< Teuchos::RCP< Element > > elemsInPartitionOfType = mesh->elementsOfType(rank, elemTypePtr);
    
    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodes(elemTypePtr);
    FieldContainer<double> cellSideParities  = mesh->cellSideParities(elemTypePtr);
    
    int numTrialDofs = trialOrdering->totalDofs();
    int numTestDofs  = testOrdering->totalDofs();
    int numCells = physicalCellNodes.dimension(0); // partition-local cells
      
    // determine cellIDs
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      int cellID = mesh->cellID(elemTypePtr, cellIndex, rank);
      cellIDs.push_back(cellID);
    }
    
    TEUCHOS_TEST_FOR_EXCEPTION( numCells!=elemsInPartitionOfType.size(), std::invalid_argument, "in computeResiduals::numCells does not match number of elems in partition.");    
     
    // prepare basisCache and cellIDs
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,mesh));
    bool createSideCacheToo = true;
    FieldContainer<double> rhs(numCells,numTestDofs);
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
    this->integrate(rhs, testOrdering, basisCache);

    _rieszRHSForElementType[elemTypePtr.get()] = rhs;
  }
  //  _residualsComputed = true;  
}

const map<int,double> & LinearTerm::energyNorm(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<DPGInnerProduct> ip) { 
  int numProcs=1;
  int rank=0;
  
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif  
  
  int numActiveElements = mesh->activeElements().size();

  computeRieszRep(mesh,ip);  
  
  // initialize error array to -1 (cannot have negative index...) 
  int localCellIDArray[numActiveElements];
  double localNormArray[numActiveElements];  
  for (int globalCellIndex=0;globalCellIndex<numActiveElements;globalCellIndex++){
    localCellIDArray[globalCellIndex] = -1;    
    localNormArray[globalCellIndex] = -1.0;        
  }  
  
  vector<ElementTypePtr> elemTypes = mesh->elementTypes(rank);   
  vector<ElementTypePtr>::iterator elemTypeIt;  
  int globalCellIndex = 0;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);    
    
    vector< Teuchos::RCP< Element > > elemsInPartitionOfType = mesh->elementsOfType(rank, elemTypePtr);
    
    // for error rep v_e, residual res, energyError = sqrt ( ve_^T * res)
    FieldContainer<double> rhs = _rieszRHSForElementType[elemTypePtr.get()];
    FieldContainer<double> rieszReps = _rieszRepresentationForElementType[elemTypePtr.get()];
    int numTestDofs = rhs.dimension(1);    
    int numCells = rhs.dimension(0);    
    TEUCHOS_TEST_FOR_EXCEPTION( numCells!=elemsInPartitionOfType.size(), std::invalid_argument, "In energyError::numCells does not match number of elems in partition.");    
    
    for (int cellIndex=0;cellIndex<numCells;cellIndex++){
      double normSquared = 0.0;
      for (int i=0; i<numTestDofs; i++) {      
        normSquared += rhs(cellIndex,i) * rieszReps(cellIndex,i);
      }
      localNormArray[globalCellIndex] = sqrt(normSquared);
      int cellID = mesh->cellID(elemTypePtr,cellIndex,rank);
      localCellIDArray[globalCellIndex] = cellID; 
      globalCellIndex++;
    }   
  } // end of loop thru element types
  
  // mpi communicate all energy errors
  double normArray[numProcs][numActiveElements];  
  int cellIDArray[numProcs][numActiveElements];    
#ifdef HAVE_MPI
  if (numProcs>1){
    MPI::COMM_WORLD.Allgather(localNormArray,numActiveElements, MPI::DOUBLE, normArray, numActiveElements , MPI::DOUBLE);      
    MPI::COMM_WORLD.Allgather(localCellIDArray,numActiveElements, MPI::INT, cellIDArray, numActiveElements , MPI::INT);        
  }else{
#else
#endif
    for (int globalCellIndex=0;globalCellIndex<numActiveElements;globalCellIndex++){    
      cellIDArray[0][globalCellIndex] = localCellIDArray[globalCellIndex];
      normArray[0][globalCellIndex] = localNormArray[globalCellIndex];
    }
#ifdef HAVE_MPI
  }
#endif
  // copy back to energyError container 
  for (int procIndex=0;procIndex<numProcs;procIndex++){
    for (int globalCellIndex=0;globalCellIndex<numActiveElements;globalCellIndex++){
      if (cellIDArray[procIndex][globalCellIndex]!=-1){
        _energyNormForCellIDGlobal[cellIDArray[procIndex][globalCellIndex]] = normArray[procIndex][globalCellIndex];
      }
    }
  }
 
  return _energyNormForCellIDGlobal;
}

double LinearTerm::energyNormTotal(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<DPGInnerProduct> ip){
  double energyNormSquared = 0.0;
  const map<int,double>* energyNormPerCell = &(energyNorm(mesh, ip));  
  for (map<int,double>::const_iterator cellEnergyIt = energyNormPerCell->begin(); 
       cellEnergyIt != energyNormPerCell->end(); cellEnergyIt++) {
    energyNormSquared += (cellEnergyIt->second) * (cellEnergyIt->second);
  }
  return sqrt(energyNormSquared);

}

// end of added by Jesse --------------------

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

void LinearTerm::addTerm(const LinearTerm &a, bool overrideTypeCheck) {
  if (_rank == -1) { // we're empty -- adopt rhs's rank
    _rank = a.rank();
  }
  if (_rank != a.rank()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "attempting to add terms of unlike rank.");
  }
  if (_termType == UNKNOWN_TYPE) {
    _termType = a.termType();
  }
  if (_termType != a.termType()) {
    if (!overrideTypeCheck) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "attempting to add terms of unlike type.");
    } else {
      _termType = MIXED_TYPE;
    }
  }
  _varIDs.insert( a.varIDs().begin(), a.varIDs().end() );
  if (a.isZero()) return; // we can skip the actual adding in this case
  _summands.insert(_summands.end(), a.summands().begin(), a.summands().end());
}

void LinearTerm::addTerm(LinearTermPtr aPtr, bool overrideTypeCheck) {
  this->addTerm(*aPtr, overrideTypeCheck);
}

LinearTerm& LinearTerm::operator+=(const LinearTerm &rhs) {
  this->addTerm(rhs);
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
    bool bypassTypeCheck = true; // unless user bypassed it, there will already have been a type check in the construction of a.  If the user did bypass, we should bypass, too.
    lt->addTerm(newWeight * var, bypassTypeCheck);
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
    bool bypassTypeCheck = true; // unless user bypassed it, there will already have been a type check in the construction of a.  If the user did bypass, we should bypass, too.
    lt->addTerm(newWeight * var, bypassTypeCheck);
  }
  return lt;
}

LinearTermPtr operator/(VarPtr v, FunctionPtr scalarFunction) {
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  return (one / scalarFunction) * v;
}

LinearTermPtr operator+(VarPtr v1, VarPtr v2) {
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0, "") );
  return one * v1 + one * v2;
}

LinearTermPtr operator/(VarPtr v, double weight) {
  return (1.0 / weight) * v;
}

LinearTermPtr operator-(VarPtr v1, VarPtr v2) {
  FunctionPtr minus_one = Teuchos::rcp( new ConstantScalarFunction(-1.0, "-") );
  return v1 + minus_one * v2;
}

LinearTermPtr operator-(VarPtr v) {
  FunctionPtr minus_one = Teuchos::rcp( new ConstantScalarFunction(-1.0, "-") );
  return minus_one * v;
}

LinearTermPtr operator-(LinearTermPtr a) {
  return Teuchos::rcp( new ConstantScalarFunction(-1.0, "-") ) * a;
}

LinearTermPtr operator-(LinearTermPtr a, VarPtr v) {
  FunctionPtr minus_one = Teuchos::rcp( new ConstantScalarFunction(-1.0, "-") );
  return a + minus_one * v;
}

LinearTermPtr operator-(VarPtr v, LinearTermPtr a) {
  FunctionPtr minus_one = Teuchos::rcp( new ConstantScalarFunction(-1.0, "-") );
  return v + minus_one * a;
}

LinearTermPtr operator-(LinearTermPtr a1, LinearTermPtr a2) {
  return a1 + -a2;
}

