//
//  LinearTerm.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "CamelliaCellTools.h"
#include "ConstantScalarFunction.h"
#include "ConstantVectorFunction.h"
#include "Function.h"
#include "LinearTerm.h"
#include "Mesh.h"
#include "MPIWrapper.h"
#include "RieszRep.h"
#include "Solution.h"
#include "TensorBasis.h"

#include "Epetra_CrsMatrix.h"
#include "Intrepid_FunctionSpaceTools.hpp"
#include <Teuchos_GlobalMPISession.hpp>

namespace Camellia {

  template<typename Scalar>
  bool linearSummandIsBoundaryValueOnly(TLinearSummand<Scalar> &ls) {
    bool opInvolvesNormal = (ls.second->op() == Camellia::OP_TIMES_NORMAL)   ||
    (ls.second->op() == Camellia::OP_TIMES_NORMAL_X) ||
    (ls.second->op() == Camellia::OP_TIMES_NORMAL_Y) ||
    (ls.second->op() == Camellia::OP_TIMES_NORMAL_Z) ||
    (ls.second->op() == Camellia::OP_CROSS_NORMAL)   ||
    (ls.second->op() == Camellia::OP_DOT_NORMAL);
    bool boundaryOnlyFunction = ls.first->boundaryValueOnly();
    return boundaryOnlyFunction || (ls.second->varType()==FLUX) || (ls.second->varType()==TRACE) || opInvolvesNormal;
  }

  template<typename Scalar>
  const vector< TLinearSummand<Scalar> > & TLinearTerm<Scalar>::summands() const {
    return _summands;
  }

  template<typename Scalar>
  TLinearTerm<Scalar>::TLinearTerm() {
    _rank = -1;
    _termType = UNKNOWN_TYPE;
  }

  template<typename Scalar>
  TLinearTerm<Scalar>::TLinearTerm(TFunctionPtr<Scalar> weight, VarPtr var) {
    _rank = -1;
    _termType = UNKNOWN_TYPE;
    addVar(weight,var);
  }

  template<typename Scalar>
  TLinearTerm<Scalar>::TLinearTerm(Scalar weight, VarPtr var) {
    _rank = -1;
    _termType = UNKNOWN_TYPE;
    addVar(weight,var);
  }

  template<typename Scalar>
  TLinearTerm<Scalar>::TLinearTerm(vector<Scalar> weight, VarPtr var) {
    _rank = -1;
    _termType = UNKNOWN_TYPE;
    addVar(weight,var);
  }

  template<typename Scalar>
  TLinearTerm<Scalar>::TLinearTerm( VarPtr v ) {
    _rank = -1;
    _termType = UNKNOWN_TYPE;
    addVar( 1.0, v);
  }

  // copy constructor:
  template<typename Scalar>
  TLinearTerm<Scalar>::TLinearTerm( const TLinearTerm &a ) {
    _rank = a.rank();
    _termType = a.termType();
    _summands = a.summands();
    _varIDs = a.varIDs();
  }

  // destructor:
  template<typename Scalar>
  TLinearTerm<Scalar>::~TLinearTerm() {
//    cout << "destroying TLinearTerm: " << this->displayString() << endl;
  }

  template<typename Scalar>
  void TLinearTerm<Scalar>::addVar(TFunctionPtr<Scalar> weight, VarPtr var) {
    // check ranks:
    int rank; // rank of weight * var
    if (weight->rank() == var->rank() ) { // then we dot like terms together, getting a scalar
      rank = 0;
    } else if ( weight->rank() == 0 || var->rank() == 0) { // then we multiply each term by scalar
      rank = (weight->rank() == 0) ? var->rank() : weight->rank(); // rank is the non-zero one
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION( true, std::invalid_argument, "Unhandled rank combination.");
    }
    if (_rank == -1) { // TLinearTerm's rank is unassigned
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

  template<typename Scalar>
  void TLinearTerm<Scalar>::addVar(Scalar weight, VarPtr var) {
    TFunctionPtr<Scalar> weightFn = (weight != 1.0) ? Teuchos::rcp( new ConstantScalarFunction<Scalar>(weight) )
                                           : Teuchos::rcp( new ConstantScalarFunction<Scalar>(weight, "") );
    // (suppress display of 1.0 weights)
    addVar( weightFn, var );
  }

  template<typename Scalar>
  void TLinearTerm<Scalar>::addVar(vector<Scalar> vector_weight, VarPtr var) { // dots weight vector with vector var, makes a vector out of a scalar var
    TFunctionPtr<Scalar> weightFn = Teuchos::rcp( new ConstantVectorFunction<Scalar>(vector_weight) );
    addVar( weightFn, var );
  }

  template<typename Scalar>
  double TLinearTerm<Scalar>::computeNorm(IPPtr ip, MeshPtr mesh) {
    TLinearTermPtr<Scalar> thisPtr = Teuchos::rcp(this, false);
    Teuchos::RCP<RieszRep> rieszRep = Teuchos::rcp( new RieszRep(mesh, ip, thisPtr) );
    rieszRep->computeRieszRep();
    return rieszRep->getNorm();
  }

  template<typename Scalar>
  string TLinearTerm<Scalar>::displayString() const {
    ostringstream dsStream;
    bool first = true;
    for (typename vector< TLinearSummand<Scalar> >::const_iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
      if ( ! first ) {
        dsStream << " + ";
      }
      TLinearSummand<Scalar> ls = *lsIt;
      TFunctionPtr<Scalar> f = ls.first;
      VarPtr var = ls.second;
      dsStream << f->displayString() << " " << var->displayString();
      first = false;
    }
    return dsStream.str();
  }

  template<typename Scalar>
  const set<int> & TLinearTerm<Scalar>::varIDs() const {
    return _varIDs;
  }

  template<typename Scalar>
  VarType TLinearTerm<Scalar>::termType() const {
    return _termType;
  }

  // some notes on the design of integrate:
  // each linear summand can be either an element-boundary-only term, or one that's defined on the
  // whole element.  For boundary-only terms, we integrate along each side.  The summands that are
  // defined on the whole element are integrated over the element interior, unless forceBoundaryTerm
  // is set to true, in which case these too will be integrated over the boundary.  One place where
  // this is appropriate is when a test function TLinearTerm is being integrated against a trace or
  // flux in a bilinear form: the test function is defined on the whole element, but the traces and
  // fluxes are only defined on the element boundary.
  // TODO: make the code below match the description above.
  template<typename Scalar>
  void TLinearTerm<Scalar>::integrate(Intrepid::FieldContainer<Scalar> &values, DofOrderingPtr thisOrdering,
                             BasisCachePtr basisCache, bool forceBoundaryTerm, bool sumInto) {
    // values has dimensions (numCells, thisFields)
    if (!sumInto) values.initialize();

    set<int> varIDs = this->varIDs();

    int numCells  = basisCache->getPhysicalCubaturePoints().dimension(0);

    BasisPtr basis;
    bool thisFluxOrTrace  = (this->termType() == FLUX) || (this->termType() == TRACE);
    bool boundaryTerm = thisFluxOrTrace || forceBoundaryTerm || basisCache->isSideCache();

    // boundaryTerm ==> even volume summands should be restricted to boundary
    // (if thisFluxOrTrace, there won't be any volume summands, so the above will hold trivially)
    // note that ! boundaryTerm does NOT imply that all terms are volume terms.  boundaryTerm means that
    // we ONLY evaluate on the boundary; !boundaryTerm allows the possibility of "mixed-type" terms
    int numSides = basisCache->cellTopology()->getSideCount();

    Teuchos::Array<int> ltValueDim;
    ltValueDim.push_back(numCells);
    ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
    ltValueDim.push_back(0); // # points -- empty until we know whether we're on side
    Intrepid::FieldContainer<Scalar> ltValues;

    for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++) {
      int varID = *varIt;
      if (! boundaryTerm ) {
        vector<int> varDofIndices = thisOrdering->getDofIndices(varID);

        // first, compute volume integral
        int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);

        const vector<int>* sidesForVar = &thisOrdering->getSidesForVarID(varID);

        bool applyCubatureWeights = true;
        int basisCardinality = -1;
        if (sidesForVar->size() == 1) { // volume variable
          basis = thisOrdering->getBasis(varID);
          basisCardinality = basis->getCardinality();
          ltValueDim[1] = basisCardinality;
          ltValueDim[2] = numPoints;
          ltValues.resize(ltValueDim);
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
        }

        // now, compute boundary integrals
        for (int sideIndex = 0; sideIndex < numSides; sideIndex++ ) {
          BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideIndex);
          if ( sideBasisCache.get() != NULL ) {
            if (sidesForVar->size() > 1) {
              if (! thisOrdering->hasBasisEntry(varID, sideIndex)) continue;
              basis = thisOrdering->getBasis(varID, sideIndex);
              basisCardinality = basis->getCardinality();
              ltValueDim[1] = basisCardinality;
              varDofIndices = thisOrdering->getDofIndices(varID,sideIndex);
            }
            numPoints = sideBasisCache->getPhysicalCubaturePoints().dimension(1);
            ltValueDim[2] = numPoints;
            ltValues.resize(ltValueDim);
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

//          bool DEBUGGING = true;
//          if (DEBUGGING) {
//            if (basisCache->cellIDs().size() > 0) {
//              if (basisCache->cellIDs()[0]==0) {
//                if (basisCardinality==1) {
//                  int cellIndex = 0;
//                  Intrepid::FieldContainer<double> valueOnSide(1, basisCardinality);
//                  for (int basisOrdinal = 0; basisOrdinal < basisCardinality; basisOrdinal++) {
//                    for (int ptIndex = 0; ptIndex < numPoints; ptIndex++) {
//                      valueOnSide(cellIndex,basisOrdinal) += ltValues(cellIndex,basisOrdinal,ptIndex);
//                    }
//                  }
//                  cout << "TLinearTerm<Scalar>::integrate: For cellID 0 on side " << sideIndex << ": " << valueOnSide(0,0) << endl;
//                  cout << "TLinearTerm<Scalar>::integrate: For cellID 0 on side " << sideIndex << ", ltValues:\n";
//                  for (int ptIndex = 0; ptIndex < numPoints; ptIndex++) {
//                    cout << ptIndex << ": " << ltValues(0,0,ptIndex) << endl;
//                  }
//                }
//              }
//            }
//          }
          }
        }
      } else {
        BasisCachePtr volumeCache;
        vector<int> sideOrdinals;
        if ( basisCache->isSideCache() ) {
          // just one side, then:
          sideOrdinals = {basisCache->getSideIndex()};
          volumeCache = basisCache->getVolumeBasisCache();
        } else {
          // determine all applicable sides
          if (thisOrdering->getNumSidesForVarID(varID)==1) // volume
          {
            for (int i=0; i<numSides; i++)
            {
              sideOrdinals.push_back(i);
            }
          }
          else
          {
            sideOrdinals = thisOrdering->getSidesForVarID(varID);
          }
          volumeCache = basisCache;
        }
        for (int sideOrdinal : sideOrdinals ) {
          int numPoints = volumeCache->getPhysicalCubaturePointsForSide(sideOrdinal).dimension(1);

          if (thisFluxOrTrace) {
            if (! thisOrdering->hasBasisEntry(varID, sideOrdinal)) continue;
            basis = thisOrdering->getBasis(varID,sideOrdinal);
          } else {
            basis = thisOrdering->getBasis(varID);
          }

          int basisCardinality = basis->getCardinality();
          ltValueDim[1] = basisCardinality;
          ltValueDim[2] = numPoints;
          ltValues.resize(ltValueDim);
          BasisCachePtr sideBasisCache = volumeCache->getSideBasisCache(sideOrdinal);
          bool applyCubatureWeights = true;
          bool naturalBoundaryValuesOnly = false; // DO include volume summands restricted to boundary
          this->values(ltValues, varID, basis, sideBasisCache, applyCubatureWeights, naturalBoundaryValuesOnly);
          if ( this->termType() == FLUX ) {
            multiplyFluxValuesByParity(ltValues, sideBasisCache);
          }
          vector<int> varDofIndices = thisFluxOrTrace ? thisOrdering->getDofIndices(varID,sideOrdinal)
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
  //        bool DEBUGGING = true;
  //        if (DEBUGGING) {
  //          if (basisCache->cellIDs().size() > 0) {
  //            if (basisCache->cellIDs()[0]==0) {
  //              if (basisCardinality==1) {
  //                int cellIndex = 0;
  //                Intrepid::FieldContainer<double> valueOnSide(1, basisCardinality);
  //                for (int basisOrdinal = 0; basisOrdinal < basisCardinality; basisOrdinal++) {
  //                  for (int ptIndex = 0; ptIndex < numPoints; ptIndex++) {
  //                    valueOnSide(cellIndex,basisOrdinal) += ltValues(cellIndex,basisOrdinal,ptIndex);
  //                  }
  //                }
  //                cout << "TLinearTerm<Scalar>::integrate: For cellID 0 on side " << sideOrdinal << ": " << valueOnSide(0,0) << endl;
  //                cout << "TLinearTerm<Scalar>::integrate: For cellID 0 on side " << sideOrdinal << ", ltValues:\n";
  //                for (int ptIndex = 0; ptIndex < numPoints; ptIndex++) {
  //                  cout << ptIndex << ": " << ltValues(0,0,ptIndex) << endl;
  //                }
  //              }
  //            }
  //          }
  //        }
        }
      }
    }
  }

  template<typename Scalar>
  void TLinearTerm<Scalar>::integrate(Intrepid::FieldContainer<Scalar> &values,
                             TLinearTermPtr<Scalar> u, DofOrderingPtr uOrdering,
                             TLinearTermPtr<Scalar> v, DofOrderingPtr vOrdering,
                             BasisCachePtr basisCache, bool sumInto) {
    integrate(NULL, values, u, uOrdering, v, vOrdering, basisCache, sumInto);
  }

  template<typename Scalar>
  void TLinearTerm<Scalar>::integrate(Epetra_CrsMatrix* valuesCrsMatrix, Intrepid::FieldContainer<double> &valuesFC,
                             TLinearTermPtr<double> u, DofOrderingPtr uOrdering,
                             TLinearTermPtr<double> v, DofOrderingPtr vOrdering,
                             BasisCachePtr basisCache, bool sumInto) {
    if (!sumInto) valuesFC.initialize();
    if (u->isZero() || v->isZero()) return;
    int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
    // will integrate either in volume or along a side, depending on the type of BasisCache passed in
    if (valuesCrsMatrix==NULL) { // then the FC is the one to fill
      TEUCHOS_TEST_FOR_EXCEPTION(valuesFC.dimension(0) != numCells, std::invalid_argument, "values.dim(0) != numCells");
      TEUCHOS_TEST_FOR_EXCEPTION(valuesFC.dimension(1) != uOrdering->totalDofs(), std::invalid_argument, "values.dim(1) != uOrdering->totalDofs()");
      TEUCHOS_TEST_FOR_EXCEPTION(valuesFC.dimension(2) != vOrdering->totalDofs(), std::invalid_argument, "values.dim(2) != vOrdering->totalDofs()");
    } else {
      if (numCells != 1) {
        cout << "CrsMatrix version of integrate() requires that numCells = 1.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CrsMatrix version of integrate() requires that numCells = 1.");
      }
    }

    bool symmetric = false; // turning off a so far buggy attempt at optimization
    //  bool symmetric = (u.get()==v.get()) && (uOrdering.get() == vOrdering.get());

    // values has dimensions (numCells, uFields, vFields)
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    int spaceDim = basisCache->getSpaceDim();

    Teuchos::Array<int> ltValueDim;
    ltValueDim.push_back(numCells);
    ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
    ltValueDim.push_back(numPoints);

    for (int i=0; i<u->rank(); i++) {
      ltValueDim.push_back(spaceDim);
    }

    set<int> uIDs = u->varIDs();
    set<int> vIDs = v->varIDs();
    vector<int> uIDVector = vector<int>(uIDs.begin(),uIDs.end());
    vector<int> vIDVector = vector<int>(vIDs.begin(),vIDs.end());

    for (int uOrdinal=0; uOrdinal < uIDVector.size(); uOrdinal++) {
      int uID = uIDVector[uOrdinal];
      //    cout << "uID: " << uID << endl;
      // the DofOrdering needs a sideIndex argument; this is 0 for volume bases.
      bool uVolVar = (uOrdering->getNumSidesForVarID(uID) == 1);
      int uSideIndex = uVolVar ? 0 : basisCache->getSideIndex();

      if (! uOrdering->hasBasisEntry(uID, uSideIndex) && (uSideIndex == basisCache->getSideIndex())) {
        // this variable doesn't live on this side, so its contribution is 0...
        continue;
      }
      if (! uOrdering->hasBasisEntry(uID, uSideIndex) ) {
        // this is a bit of a mess: a hack to allow us to do projections on side bases
        // we could avoid this if either TLinearTerm or DofOrdering did things better, if either
        // 1. TLinearTerm knew about VarPtrs instead of merely varIDs--then we could learn that u was a trace or flux
        //     OR
        // 2. DofOrdering used a sideIndex of -1 for volume variables, instead of the ambiguous 0.
        if (basisCache->isSideCache()) {
          uSideIndex = basisCache->getSideIndex();
        }
        // now, test again, and throw an exception if the issue wasn't corrected:
        if (! uOrdering->hasBasisEntry(uID, uSideIndex) ) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "no entry for uSideIndex");
        }
      }
      BasisPtr uBasis = uOrdering->getBasis(uID,uSideIndex);
      int uBasisCardinality = uBasis->getCardinality();
      ltValueDim[1] = uBasisCardinality;
      Intrepid::FieldContainer<double> uValues(ltValueDim);
      bool applyCubatureWeights = true, dontApplyCubatureWeights = false;
      u->values(uValues,uID,uBasis,basisCache,applyCubatureWeights);


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

      int vStartOrdinal = symmetric ? uOrdinal : 0;

      for (int vOrdinal = vStartOrdinal; vOrdinal < vIDVector.size(); vOrdinal++) {
        int vID = vIDVector[vOrdinal];
        //      cout << "vID: " << vID << endl;
        bool vVolVar = (vOrdering->getNumSidesForVarID(vID) == 1);
        int vSideIndex = vVolVar ? 0 : basisCache->getSideIndex();
        if (! vOrdering->hasBasisEntry(vID, vSideIndex) && (vSideIndex == basisCache->getSideIndex())) {
          // this variable doesn't live on this side, so its contribution is 0...
          continue;
        }
        if (! vOrdering->hasBasisEntry(vID, vSideIndex) ) {
          // this is a bit of a mess: a hack to allow us to do projections on side bases
          // we could avoid this if either TLinearTerm or DofOrdering did things better, if either
          // 1. TLinearTerm knew about VarPtrs instead of merely varIDs--then we could learn that u was a trace or flux
          //     OR
          // 2. DofOrdering used a sideIndex of -1 for volume variables, instead of the ambiguous 0.
          if (basisCache->isSideCache()) {
            vSideIndex = basisCache->getSideIndex();
          }
          // now, test again, and throw an exception if the issue wasn't corrected:
          if (! vOrdering->hasBasisEntry(vID, vSideIndex) ) {
            cout << "No entry for vID = " << vID << " with vSideIndex = " << vSideIndex << " in DofOrdering: \n";
            cout << *vOrdering;
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "no entry for vSideIndex");
          }
        }

        BasisPtr vBasis = vOrdering->getBasis(vID,vSideIndex);
        int vBasisCardinality = vBasis->getCardinality();
        ltValueDim[1] = vBasisCardinality;
        Intrepid::FieldContainer<double> vValues(ltValueDim);
        v->values(vValues, vID, vBasis, basisCache, dontApplyCubatureWeights);

        //      cout << "vValues (without cubature weights applied) for v = " << v->displayString() << ":" << endl;
        //      cout << vValues;

        // same flux consideration, for the vValues
        if ( v->termType() == FLUX ) {
          multiplyFluxValuesByParity(vValues, basisCache);
        }

        Intrepid::FieldContainer<double> miniMatrix( numCells, uBasisCardinality, vBasisCardinality );

        Intrepid::FunctionSpaceTools::integrate<double>(miniMatrix,uValues,vValues,Intrepid::COMP_BLAS);

        //      cout << "uValues:" << endl << uValues;
        //      cout << "vValues:" << endl << vValues;
        //      cout << "miniMatrix:" << endl << miniMatrix;

        vector<int> uDofIndices = uOrdering->getDofIndices(uID,uSideIndex);
        vector<int> vDofIndices = vOrdering->getDofIndices(vID,vSideIndex);

        if (valuesCrsMatrix==NULL) {
          // there may be a more efficient way to do this copying:
          for (unsigned k=0; k < numCells; k++) {
            for (int i=0; i < uBasisCardinality; i++) {
              int uDofIndex = uDofIndices[i];
              for (int j=0; j < vBasisCardinality; j++) {
                int vDofIndex = vDofIndices[j];
                double value = miniMatrix(k,i,j); // separate line for debugger inspection
                valuesFC(k,uDofIndex,vDofIndex) += value;
                if ((symmetric) && (uOrdinal != vOrdinal)) {  // pretty sure this point is where the bug in symmetric accumulation comes in.  Pretty sure we'll get some double-accumulation.  I'm not sure how to fix it just yet, though.
                  valuesFC(k,vDofIndex,uDofIndex) += value;
                }
                //            cout << "values(" << k << ", " << uDofIndex << ", " << vDofIndex << ") += " << value << endl;
              }
            }
          }
        } else { // CrsMatrix version
          Intrepid::FieldContainer<int> uDofIndicesFC(uDofIndices.size()), vDofIndicesFC(vDofIndices.size());
          for (int i=0; i < uBasisCardinality; i++) {
            uDofIndicesFC[i] = uDofIndices[i];
          }
          for (int j=0; j < vBasisCardinality; j++) {
            vDofIndicesFC[j] = vDofIndicesFC[j];
          }
          for (int i=0; i < uBasisCardinality; i++) {
            int uDofIndex = uDofIndices[i];
            valuesCrsMatrix->SumIntoGlobalValues(uDofIndex, vBasisCardinality, &miniMatrix(0,i,0), &vDofIndicesFC[0]);
          }
        }
      }
    }
    //  cout << "Integrate complete.\n";
  }

  template<typename Scalar>
  void TLinearTerm<Scalar>::integrate(Epetra_CrsMatrix *values, DofOrderingPtr thisOrdering,
                             TLinearTermPtr<double> otherTerm, DofOrderingPtr otherOrdering,
                             BasisCachePtr basisCache, bool forceBoundaryTerm, bool sumInto) {
    static Intrepid::FieldContainer<double> emptyValues;
    integrate(values, emptyValues, thisOrdering, otherTerm, otherOrdering, basisCache, forceBoundaryTerm, sumInto);
  }

  template<typename Scalar>
  void TLinearTerm<Scalar>::integrate(Intrepid::FieldContainer<Scalar> &values, DofOrderingPtr thisOrdering,
                             TLinearTermPtr<Scalar> otherTerm, DofOrderingPtr otherOrdering,
                             BasisCachePtr basisCache, bool forceBoundaryTerm, bool sumInto) {
    integrate(NULL, values, thisOrdering, otherTerm, otherOrdering, basisCache, forceBoundaryTerm, sumInto);
  }

  template<typename Scalar>
  void TLinearTerm<Scalar>::integrate(Epetra_CrsMatrix *valuesCrsMatrix, Intrepid::FieldContainer<double> &valuesFC, DofOrderingPtr thisOrdering,
                             TLinearTermPtr<double> otherTerm, DofOrderingPtr otherOrdering,
                             BasisCachePtr basisCache, bool forceBoundaryTerm, bool sumInto) {
    // values has dimensions (numCells, otherFields, thisFields)
    // note that this means when we call the private integrate, we need to use otherTerm as the first TLinearTerm argument
    if (!sumInto) valuesFC.initialize();

    // define variables:
    //  u - the non-boundary-only part of this
    // du - the boundary-only part of this
    //  v - the non-boundary-only part of otherTerm
    // dv - the boundary-only part of otherTerm

    // we are then computing (u + du, v + dv) where e.g. (du,v) will be integrated along boundary
    // so the only non-boundary term is (u,v), and that only comes into it if forceBoundaryTerm is false.

    bool symmetric = (thisOrdering.get() == otherOrdering.get()) && (this == otherTerm.get());

    if (basisCache->isSideCache() && !forceBoundaryTerm) {
      // if sideCache, then we'd better forceBoundaryTerm
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: forceBoundaryTerm is false but basisCache is a sideBasisCache...");
    }

    TLinearTermPtr<Scalar> thisPtr = Teuchos::rcp( this, false );

    if (basisCache->isSideCache()) {
      // then we just integrate along the one side:
      integrate(valuesCrsMatrix, valuesFC, otherTerm, otherOrdering, thisPtr, thisOrdering, basisCache);
      return;
    } else if (forceBoundaryTerm) {
      // then we don't need to worry about splitting into boundary and non-boundary parts,
      // but we do need to loop over the sides:
      int numSides = basisCache->cellTopology()->getSideCount();
      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
        integrate(valuesCrsMatrix, valuesFC, otherTerm, otherOrdering, thisPtr, thisOrdering, basisCache->getSideBasisCache(sideIndex));
      }
    } else {
      TLinearTermPtr<Scalar> thisBoundaryOnly = this->getBoundaryOnlyPart();
      TLinearTermPtr<Scalar> thisNonBoundaryOnly = this->getNonBoundaryOnlyPart();
      TLinearTermPtr<Scalar> otherBoundaryOnly;
      TLinearTermPtr<Scalar> otherNonBoundaryOnly;

      if (symmetric) {
        otherBoundaryOnly = thisBoundaryOnly;
        otherNonBoundaryOnly = thisNonBoundaryOnly;
      } else {
        otherBoundaryOnly = otherTerm->getBoundaryOnlyPart();
        otherNonBoundaryOnly = otherTerm->getNonBoundaryOnlyPart();
      }

      // volume integration first:  ( (u,v) from above )
      integrate(valuesCrsMatrix, valuesFC, otherNonBoundaryOnly, otherOrdering, thisNonBoundaryOnly, thisOrdering, basisCache);

      // sides:
      // (u + du, v + dv) - (u,v) = (u + du, dv) + (du, v)
      int numSides = basisCache->cellTopology()->getSideCount();
      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
        BasisCachePtr sideCache = basisCache->getSideBasisCache(sideIndex);
        // (u + du, dv)
        integrate(valuesCrsMatrix, valuesFC, otherBoundaryOnly, otherOrdering, thisPtr, thisOrdering, sideCache);
        // (du, v)
        integrate(valuesCrsMatrix, valuesFC, otherNonBoundaryOnly, otherOrdering, thisBoundaryOnly, thisOrdering, sideCache);
      }
    }
  }

  // integrate this against otherTerm, where otherVar == fxn
  template<typename Scalar>
  void TLinearTerm<Scalar>::integrate(Intrepid::FieldContainer<Scalar> &values, DofOrderingPtr thisOrdering,
                             TLinearTermPtr<Scalar> otherTerm, VarPtr otherVar, TFunctionPtr<Scalar> fxn,
                             BasisCachePtr basisCache, bool forceBoundaryTerm) {
    // values has dimensions (numCells, thisFields)

    if (!fxn.get()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fxn cannot be null!");
    }

    map<int, TFunctionPtr<Scalar>> otherVarMap;
    otherVarMap[otherVar->ID()] = fxn;
    TFunctionPtr<Scalar> otherFxnBoundaryPart = otherTerm->evaluate(otherVarMap, true);
    TFunctionPtr<Scalar> otherFxnVolumePart = otherTerm->evaluate(otherVarMap, false);

    TLinearTermPtr<Scalar> thisPtr = Teuchos::rcp(this, false);
    TLinearTermPtr<Scalar> lt = otherFxnVolumePart * thisPtr + otherFxnBoundaryPart * thisPtr;

    lt->integrate(values, thisOrdering, basisCache, forceBoundaryTerm);
  }

  template<typename Scalar>
  bool TLinearTerm<Scalar>::isZero() const { // true if the TLinearTerm is identically zero
    // DEBUGGING test: pretend we're NEVER zero
    //return false;
    for (typename vector< TLinearSummand<Scalar> >::const_iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
      TLinearSummand<Scalar> ls = *lsIt;
      TFunctionPtr<Scalar> f = ls.first;
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
  //  TLinearTerm can determine its OP_VALUE VarPtrs so that it can construct the right map.  It would probably suffice
  //  if Var learned about its "base", and then in addition to tracking the varIDs set here, we also tracked vars.)
  template<typename Scalar>
  void TLinearTerm<Scalar>::evaluate(Intrepid::FieldContainer<Scalar> &values, TSolutionPtr<Scalar> solution, BasisCachePtr basisCache,
                            bool applyCubatureWeights) {
    // int sideIndex = basisCache->getSideIndex();
    //  bool boundaryTerm = (sideIndex != -1);

    int valuesRankExpected = _rank + 2; // 2 for scalar, 3 for vector, etc.
    TEUCHOS_TEST_FOR_EXCEPTION( valuesRankExpected != values.rank(), std::invalid_argument,
                               "values FC does not have the expected rank" );
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int spaceDim = basisCache->getSpaceDim();

    values.initialize(0.0);
    Teuchos::Array<int> scalarFunctionValueDim;
    scalarFunctionValueDim.append(numCells);
    scalarFunctionValueDim.append(numPoints);

    // (could tune things by pre-allocating this storage)
    Intrepid::FieldContainer<Scalar> fValues;
    Intrepid::FieldContainer<Scalar> solnValues;

    Teuchos::Array<int> vectorFunctionValueDim = scalarFunctionValueDim;
    vectorFunctionValueDim.append(spaceDim);
    Teuchos::Array<int> tensorFunctionValueDim = vectorFunctionValueDim;
    tensorFunctionValueDim.append(spaceDim);

    TEUCHOS_TEST_FOR_EXCEPTION( numCells != basisCache->getPhysicalCubaturePoints().dimension(0),
                               std::invalid_argument, "values FC numCells disagrees with cubature points container");
    TEUCHOS_TEST_FOR_EXCEPTION( numPoints != basisCache->getPhysicalCubaturePoints().dimension(1),
                               std::invalid_argument, "values FC numPoints disagrees with cubature points container");
    for (typename vector< TLinearSummand<Scalar> >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
      TLinearSummand<Scalar> ls = *lsIt;
      TFunctionPtr<Scalar> f = ls.first;
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
          const Scalar *fValue = &fValues[fValues.getEnumeration(fDim)];
          const Scalar *solnValue = &solnValues[solnValues.getEnumeration(solnDim)];

          Scalar *value = &values[values.getEnumeration(vDim)];

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

  template<typename Scalar>
  TLinearTermPtr<Scalar> TLinearTerm<Scalar>::getBoundaryOnlyPart() {
    return this->getPart(true);
  }

  template<typename Scalar>
  TLinearTermPtr<Scalar> TLinearTerm<Scalar>::getNonBoundaryOnlyPart() {
    return this->getPart(false);
  }

  template<typename Scalar>
  TLinearTermPtr<Scalar> TLinearTerm<Scalar>::getPart(bool boundaryOnlyPart) {
    TLinearTermPtr<Scalar> lt = Teuchos::rcp( new TLinearTerm<Scalar> );
    for (typename vector< TLinearSummand<Scalar> >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
      TLinearSummand<Scalar> ls = *lsIt;
      if ( linearSummandIsBoundaryValueOnly(ls) && !boundaryOnlyPart) {
        continue;
      } else if (!linearSummandIsBoundaryValueOnly(ls) && boundaryOnlyPart) {
        continue;
      }
      TFunctionPtr<Scalar> f = ls.first;
      VarPtr var = ls.second;

      if (lt.get()) {
        lt = lt + f * var;
      } else {
        lt = f * var;
      }
    }
    return lt;
  }

  template<typename Scalar>
  TLinearTermPtr<Scalar> TLinearTerm<Scalar>::getPartMatchingVariable( VarPtr varToMatch ) {
    TLinearTermPtr<Scalar> lt = Teuchos::rcp( new TLinearTerm<Scalar> );
    for (typename vector< TLinearSummand<Scalar> >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
      TLinearSummand<Scalar> ls = *lsIt;

      TFunctionPtr<Scalar> f = ls.first;
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

  template<typename Scalar>
  TFunctionPtr<Scalar> TLinearTerm<Scalar>::evaluate(const map< int, TFunctionPtr<Scalar>> &varFunctions) {
    return evaluate(varFunctions,false) + evaluate(varFunctions, true);
  }

  template<typename Scalar>
  TFunctionPtr<Scalar> TLinearTerm<Scalar>::evaluate(const map< int, TFunctionPtr<Scalar>> &varFunctions, bool boundaryPart) {
    // NOTE that if boundaryPart is false, then we exclude terms that are defined only on the boundary
    // and if boundaryPart is true, then we exclude terms that are defined everywhere
    // so that the whole TLinearTerm is the sum of the two options
    TFunctionPtr<Scalar> fxn = TFunction<Scalar>::null();
    vector< TLinearSummand<Scalar> > summands = this->getPart(boundaryPart)->summands();
    for (typename vector< TLinearSummand<Scalar> >::iterator lsIt = summands.begin(); lsIt != summands.end(); lsIt++) {
      TLinearSummand<Scalar> ls = *lsIt;
      TFunctionPtr<Scalar> f = ls.first;
      VarPtr var = ls.second;

      // if there isn't an entry for var, we take it to be zero:
      if (varFunctions.find(var->ID()) == varFunctions.end()) continue;

      TFunctionPtr<Scalar> varFunction = varFunctions.find(var->ID())->second;

      if (!varFunction.get()) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "varFunctions entries cannot be null!");
      }

      TFunctionPtr<Scalar> varEvaluation = TFunction<Scalar>::op(varFunction,var->op());
      {
        // DEBUGGING CODE:
        if (varEvaluation.get() == NULL) {
          // try that again, so we can step into the code
          TFunctionPtr<Scalar> varEvaluation = TFunction<Scalar>::op(varFunction,var->op());
        }
      }

      if (fxn.get()) {
        fxn = fxn + f * varEvaluation;
      } else {
        fxn = f * varEvaluation;
      }
    }
    if (!fxn.get()) {
      fxn = TFunction<Scalar>::zero(this->rank());
    }
    return fxn;
  }

  template<typename Scalar>
  void TLinearTerm<Scalar>::multiplyFluxValuesByParity(Intrepid::FieldContainer<Scalar> &fluxValues, BasisCachePtr sideBasisCache) {
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
  template<typename Scalar>
  void TLinearTerm<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, int varID, BasisPtr basis, BasisCachePtr basisCache,
                          bool applyCubatureWeights, bool naturalBoundaryTermsOnly) {
    int sideIndex = basisCache->getSideIndex();

    int valuesRankExpected = _rank + 3; // 3 for scalar, 4 for vector, etc.
    TEUCHOS_TEST_FOR_EXCEPTION( valuesRankExpected != values.rank(), std::invalid_argument,
                               "values FC does not have the expected rank" );
    int numCells = values.dimension(0);
    int numFields = values.dimension(1);
    int numPoints = values.dimension(2);
    int basisRangeDimension = basis->rangeDimension();

    values.initialize(0.0);
    Teuchos::Array<int> scalarFunctionValueDim;
    scalarFunctionValueDim.append(numCells);
    scalarFunctionValueDim.append(numPoints);

    // (could tune things by pre-allocating this storage)
    Intrepid::FieldContainer<Scalar> fValues;

    Teuchos::Array<int> vectorFunctionValueDim = scalarFunctionValueDim;
    vectorFunctionValueDim.append(basisRangeDimension);
    Teuchos::Array<int> tensorFunctionValueDim = vectorFunctionValueDim;
    tensorFunctionValueDim.append(basisRangeDimension);

    TEUCHOS_TEST_FOR_EXCEPTION( numCells != basisCache->getPhysicalCubaturePoints().dimension(0),
                               std::invalid_argument, "values FC numCells disagrees with cubature points container");
    TEUCHOS_TEST_FOR_EXCEPTION( numFields != basis->getCardinality(),
                               std::invalid_argument, "values FC numFields disagrees with basis cardinality");
    TEUCHOS_TEST_FOR_EXCEPTION( numPoints != basisCache->getPhysicalCubaturePoints().dimension(1),
                               std::invalid_argument, "values FC numPoints disagrees with cubature points container");
    for (typename vector< TLinearSummand<Scalar> >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
      TLinearSummand<Scalar> ls = *lsIt;
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

        if (basisValues->dimension(2) != numPoints) {
          {
            // DEBUGGING: repeat the call, so we can watch it:
            if (applyCubatureWeights) {
              // on sides, we use volume coords for test and field values
              bool useVolumeCoords = (sideIndex != -1) && ((ls.second->varType() == TEST) || (ls.second->varType() == FIELD));
              basisValues = basisCache->getTransformedWeightedValues(basis, ls.second->op(), useVolumeCoords);
            } else {
              // on sides, we use volume coords for test and field values
              bool useVolumeCoords = (sideIndex != -1) && ((ls.second->varType() == TEST) || (ls.second->varType() == FIELD));
              basisValues = basisCache->getTransformedValues(basis, ls.second->op(), useVolumeCoords);
            }
          }

          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: transformed basisValues doesn't have the correct # of points.");
        }

        if ( ls.first->rank() == 0 ) { // scalar function -- we can speed things along in this case...
          // E.g. ConstantTFunction<double>::scalarMultiplyBasisValues() knows not to do anything at all if its value is 1.0...
          Intrepid::FieldContainer<double> weightedBasisValues = *basisValues; // weighted by the scalar function
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
            fDim.append(basisRangeDimension);
          }
          fValues.resize(fDim);
        }

        ls.first->values(fValues,basisCache);

        int numFields = basis->getCardinality();

//        { // DEBUGGING:
//          typedef Camellia::TensorBasis<double, Intrepid::FieldContainer<double> > TensorBasis;
//          TensorBasis* tensorBasis = dynamic_cast<TensorBasis*>(basis.get());
//          if (tensorBasis != NULL) {
//            BasisPtr spatialBasis = tensorBasis->getSpatialBasis();
//            BasisPtr temporalBasis = tensorBasis->getTemporalBasis();
//            cout << "Tensor basis has a degree-" << spatialBasis->getDegree() << " spatial basis and a degree-";
//            cout << temporalBasis->getDegree() << " temporal basis.\n";
//            cout << "Basis Values:\n" << *basisValues;
//          }
//        }

        Teuchos::Array<int> fDim(fValues.rank());
        Teuchos::Array<int> bDim(basisValues->rank());

        bool usePointerArithmetic = false; // slightly faster, but skips some bounds checks
        if (usePointerArithmetic) { // do some bounds checks here
          // TODO: do some bounds checks here
        }

        // compute f * basisValues
        if ( ls.first->rank() == ls.second->rank() ) { // scalar result
          int entriesPerPoint = 1;
          int fRank = ls.first->rank();
          for (int d=0; d<fRank; d++) {
            entriesPerPoint *= basisRangeDimension;
          }

          for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            fDim[0] = cellIndex; bDim[0] = cellIndex;
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
              fDim[1] = ptIndex; bDim[2] = ptIndex;
              for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
                if (usePointerArithmetic) {
                  const Scalar *fValue = &fValues[fValues.getEnumeration(fDim)];
                  bDim[1] = fieldIndex;
                  const double *bValue = &((*basisValues)[basisValues->getEnumeration(bDim)]);
                  for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
                    values(cellIndex,fieldIndex,ptIndex) += *fValue * *bValue;

                    fValue++;
                    bValue++;
                  }
                } else {
                  int fValueEnum = fValues.getEnumeration(fDim);
                  bDim[1] = fieldIndex;
                  int bValueEnum = basisValues->getEnumeration(bDim);
//                  cout << ls.first->displayString() << " " << ls.second->displayString();
//                  cout << ", basis cardinality: " << numFields << "; entriesPerPoint: " << entriesPerPoint << endl;
                  for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
//                    cout << "bValue: " << (*basisValues)[bValueEnum] << endl;
                    values(cellIndex,fieldIndex,ptIndex) += fValues[fValueEnum] * (*basisValues)[bValueEnum];

                    fValueEnum++;
                    bValueEnum++;
                  }
                }
              }
            }
          }
        } else { // vector/tensor result
          // could pretty easily fold the scalar case above into the code below
          // (just change the logic in the pointer increments)
          int entriesPerPoint = 1;

          //        cout << "basisValues:\n" << basisValues;
          //        cout << "fValues:\n" << fValues;

          // now that we've changed so that we handle scalar function multiplication separately,
          // we don't hit this code for scalar functions.  I.e. scalarF == false always.
          bool scalarF = ls.first->rank() == 0;
          int resultRank = scalarF ? ls.second->rank() : ls.first->rank();
          for (int d=0; d<resultRank; d++) {
            entriesPerPoint *= basisRangeDimension;
          }
          Teuchos::Array<int> vDim( values.rank() );
          for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            fDim[0] = cellIndex; bDim[0] = cellIndex; vDim[0] = cellIndex;
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
              fDim[1] = ptIndex; bDim[2] = ptIndex; vDim[2] = ptIndex;
              for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
                if (usePointerArithmetic) {
                  const Scalar *fValue = &fValues[fValues.getEnumeration(fDim)];
                  bDim[1] = fieldIndex; vDim[1] = fieldIndex;
                  const double *bValue = &(*basisValues)[basisValues->getEnumeration(bDim)];

                  Scalar *value = &values[values.getEnumeration(vDim)];

                  for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
                    *value += *fValue * *bValue;
                    value++;
                    if (scalarF) {
                      bValue++;
                    } else {
                      fValue++;
                    }
                  }
                } else {
                  int fValueEnum = fValues.getEnumeration(fDim);
                  bDim[1] = fieldIndex; vDim[1] = fieldIndex;
                  int bValueEnum = basisValues->getEnumeration(bDim);
                  int valueEnum = values.getEnumeration(vDim);

  //                double *value = &values[values.getEnumeration(vDim)];

                  for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
                    values[valueEnum] += fValues[fValueEnum] * (*basisValues)[bValueEnum];
                    valueEnum++;
                    if (scalarF) {
                      bValueEnum++;
                    } else {
                      fValueEnum++;
                    }
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
  template<typename Scalar>
  void TLinearTerm<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, int varID, TFunctionPtr<Scalar> fxn, BasisCachePtr basisCache,
                          bool applyCubatureWeights, bool naturalBoundaryTermsOnly) {
    int sideIndex = basisCache->getSideIndex();

    int valuesRankExpected = _rank + 2; // 2 for scalar, 3 for vector, etc.
    TEUCHOS_TEST_FOR_EXCEPTION( valuesRankExpected != values.rank(), std::invalid_argument,
                               "values FC does not have the expected rank" );
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int spaceDim = basisCache->getSpaceDim();

    values.initialize(0.0);
    Teuchos::Array<int> scalarFunctionValueDim;
    scalarFunctionValueDim.append(numCells);
    scalarFunctionValueDim.append(numPoints);

    // (could tune things by pre-allocating this storage)
    Intrepid::FieldContainer<Scalar> fValues;

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
    for (typename vector< TLinearSummand<Scalar> >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
      TLinearSummand<Scalar> ls = *lsIt;
      // skip if this is a volume term, and we're only interested in the pure-boundary terms
      if (naturalBoundaryTermsOnly && !linearSummandIsBoundaryValueOnly(ls)) {
        continue;
      }
      // skip if this is a boundary term, and we're doing a volume integration:
      if ((sideIndex == -1) && linearSummandIsBoundaryValueOnly(ls)) {
        continue;
      }
      if (ls.second->ID() == varID) {
        Intrepid::FieldContainer<Scalar> fxnValues(fxnValueDim);
        fxn->values(fxnValues, ls.second->op(), basisCache); // should always use the volume coords (compare with other TLinearTerm<Scalar>::values() function)
        if (applyCubatureWeights) {
          // TODO: apply cubature weights!!
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "still need to implement cubature weighting");
        }

        if ( ls.first->rank() == 0 ) { // scalar function -- we can speed things along in this case...
          // E.g. ConstantTFunction<double>::scalarMultiplyBasisValues() knows not to do anything at all if its value is 1.0...
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
              const Scalar *fValue = &fValues[fValues.getEnumeration(fDim)];
              const Scalar *fxnValue = &(fxnValues[fxnValues.getEnumeration(fxnDim)]);
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
          //        cout << "fxnValues:\n" << fxnValues;
          //        cout << "fValues:\n" << fValues;

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
              const Scalar *fValue = &fValues[fValues.getEnumeration(fDim)];
              const Scalar *fxnValue = &(fxnValues)[fxnValues.getEnumeration(fxnDim)];

              Scalar *value = &values[values.getEnumeration(vDim)];

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

  template<typename Scalar>
  int TLinearTerm<Scalar>::rank() const {   // 0 for scalar, 1 for vector, etc.
    return _rank;
  }

  // operator overloading niceties:

  template<typename Scalar>
  TLinearTerm<Scalar>& TLinearTerm<Scalar>::operator=(const TLinearTerm<Scalar> &rhs) {
    if ( this == &rhs ) {
      return *this;
    }
    _rank = rhs.rank();
    _summands = rhs.summands();
    _varIDs = rhs.varIDs();
    return *this;
  }

  template<typename Scalar>
  void TLinearTerm<Scalar>::addTerm(const TLinearTerm<Scalar> &a, bool overrideTypeCheck) {
    if (_rank == -1) { // we're empty -- adopt rhs's rank
      _rank = a.rank();
    }
    if (a.isZero()) return; // we can skip the actual adding in this case
    if (_rank != a.rank()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "attempting to add terms of unlike rank.");
    }
    if (_termType == UNKNOWN_TYPE) {
      _termType = a.termType();
    }
    if (_termType != a.termType()) {
      if (!overrideTypeCheck) {
        cout << "ERROR: attempting to add terms of unlike type.\n";
        cout << "Attempting to add term of type " << _termType << " to one of type " << a.termType();
        cout << "; terms are " << this->displayString() << " and " << a.displayString() << endl;
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "attempting to add terms of unlike type.");
      } else {
        _termType = MIXED_TYPE;
      }
    }
    _varIDs.insert( a.varIDs().begin(), a.varIDs().end() );
    _summands.insert(_summands.end(), a.summands().begin(), a.summands().end());
  }

  template<typename Scalar>
  void TLinearTerm<Scalar>::addTerm(TLinearTermPtr<Scalar> aPtr, bool overrideTypeCheck) {
    this->addTerm(*aPtr, overrideTypeCheck);
  }

  template<typename Scalar>
  TLinearTerm<Scalar>& TLinearTerm<Scalar>::operator+=(const TLinearTerm<Scalar> &rhs) {
    this->addTerm(rhs);
    return *this;
  }

  template<typename Scalar>
  TLinearTerm<Scalar>& TLinearTerm<Scalar>::operator+=(VarPtr v) {
    this->addVar(1.0, v);
    return *this;
  }

  template class TLinearTerm<double>;

  // operator overloading for syntax sugar:
  TLinearTermPtr<double> operator+(TLinearTermPtr<double> a1, TLinearTermPtr<double> a2) {
    TLinearTermPtr<double> sum = Teuchos::rcp( new TLinearTerm<double>(*a1) );
    //  cout << "sum->rank(): " << sum->rank() << endl;
    //  cout << "sum->summands.size() before adding a2: " << sum->summands().size() << endl;
    *sum += *a2;
    //  cout << "sum->summands.size() after adding a2: " << sum->summands().size() << endl;
    return sum;
  }

  TLinearTermPtr<double> operator+(VarPtr v, TLinearTermPtr<double> a) {
    TLinearTermPtr<double> sum = Teuchos::rcp( new TLinearTerm<double>(*a) );
    *sum += v;
    return sum;
  }

  TLinearTermPtr<double> operator+(TLinearTermPtr<double> a, VarPtr v) {
    return v + a;
  }

  TLinearTermPtr<double> operator+(VarPtr v1, VarPtr v2) {
    TFunctionPtr<double> one = Teuchos::rcp( new ConstantScalarFunction<double>(1.0, "") );
    return one * v1 + one * v2;
  }

  TLinearTermPtr<double> operator*(TFunctionPtr<double> f, VarPtr v) {
    return Teuchos::rcp( new TLinearTerm<double>(f, v) );
  }

  TLinearTermPtr<double> operator*(VarPtr v, TFunctionPtr<double> f) {
    return f * v;
  }

  TLinearTermPtr<double> operator*(double weight, VarPtr v) {
    return Teuchos::rcp( new TLinearTerm<double>(weight, v) );
  }

  TLinearTermPtr<double> operator*(VarPtr v, double weight) {
    return weight * v;
  }

  TLinearTermPtr<double> operator*(vector<double> weight, VarPtr v) {
    return Teuchos::rcp( new TLinearTerm<double>(weight, v) );
  }

  TLinearTermPtr<double> operator*(VarPtr v, vector<double> weight) {
    return weight * v;
  }

  TLinearTermPtr<double> operator*(TFunctionPtr<double> f, TLinearTermPtr<double> a) {
    TLinearTermPtr<double> lt = Teuchos::rcp( new TLinearTerm<double> );

    for (typename vector< TLinearSummand<double> >::const_iterator lsIt = a->summands().begin(); lsIt != a->summands().end(); lsIt++) {
      TLinearSummand<double> ls = *lsIt;
      TFunctionPtr<double> lsWeight = ls.first;
      TFunctionPtr<double> newWeight = f * lsWeight;
      VarPtr var = ls.second;
      bool bypassTypeCheck = true; // unless user bypassed it, there will already have been a type check in the construction of a.  If the user did bypass, we should bypass, too.
      lt->addTerm(newWeight * var, bypassTypeCheck);
    }
    return lt;
  }

  TLinearTermPtr<double> operator*(TLinearTermPtr<double> a, TFunctionPtr<double> f)
  {
    TLinearTermPtr<double> lt = Teuchos::rcp( new TLinearTerm<double> );

    for (typename vector< TLinearSummand<double> >::const_iterator lsIt = a->summands().begin(); lsIt != a->summands().end(); lsIt++) {
      TLinearSummand<double> ls = *lsIt;
      TFunctionPtr<double> lsWeight = ls.first;
      TFunctionPtr<double> newWeight = lsWeight * f;
      VarPtr var = ls.second;
      bool bypassTypeCheck = true; // unless user bypassed it, there will already have been a type check in the construction of a.  If the user did bypass, we should bypass, too.
      lt->addTerm(newWeight * var, bypassTypeCheck);
    }
    return lt;
  }

  TLinearTermPtr<double> operator/(TLinearTermPtr<double> a, TFunctionPtr<double> f) {
    TLinearTermPtr<double> lt = Teuchos::rcp( new TLinearTerm<double> );

    for (typename vector< TLinearSummand<double> >::const_iterator lsIt = lt->summands().begin(); lsIt != lt->summands().end(); lsIt++) {
      TLinearSummand<double> ls = *lsIt;
      TFunctionPtr<double> lsWeight = ls.first;
      TFunctionPtr<double> newWeight = lsWeight / f;
      VarPtr var = ls.second;
      bool bypassTypeCheck = true; // unless user bypassed it, there will already have been a type check in the construction of a.  If the user did bypass, we should bypass, too.
      lt->addTerm(newWeight * var, bypassTypeCheck);
    }
    return lt;
  }

  TLinearTermPtr<double> operator/(VarPtr v, TFunctionPtr<double> scalarFunction) {
    TFunctionPtr<double> one = Teuchos::rcp( new ConstantScalarFunction<double>(1.0) );
    return (one / scalarFunction) * v;
  }

  TLinearTermPtr<double> operator/(VarPtr v, double weight) {
    return (1.0 / weight) * v;
  }

  TLinearTermPtr<double> operator-(TLinearTermPtr<double> a) {
    TFunctionPtr<double> minus_one = Teuchos::rcp( new ConstantScalarFunction<double>(-1.0, "-") );
    return minus_one * a;
  }

  TLinearTermPtr<double> operator-(TLinearTermPtr<double> a, VarPtr v) {
    TFunctionPtr<double> minus_one = Teuchos::rcp( new ConstantScalarFunction<double>(-1.0, "-") );
    return a + minus_one * v;
  }

  TLinearTermPtr<double> operator-(VarPtr v, TLinearTermPtr<double> a) {
    TFunctionPtr<double> minus_one = Teuchos::rcp( new ConstantScalarFunction<double>(-1.0, "-") );
    return v + minus_one * a;
  }

  TLinearTermPtr<double> operator-(TLinearTermPtr<double> a1, TLinearTermPtr<double> a2) {
    return a1 + -a2;
  }

  TLinearTermPtr<double> operator-(VarPtr v1, VarPtr v2) {
    TFunctionPtr<double> minus_one = Teuchos::rcp( new ConstantScalarFunction<double>(-1.0, "-") );
    return v1 + minus_one * v2;
  }

  TLinearTermPtr<double> operator-(VarPtr v) {
    TFunctionPtr<double> minus_one = Teuchos::rcp( new ConstantScalarFunction<double>(-1.0, "-") );
    return minus_one * v;
  }







  // template<typename Scalar>
  // TLinearTerm<Scalar>& TLinearTerm<Scalar>::operator+=(const TLinearTerm<Scalar> &rhs) {
  //   this->addTerm(rhs);
  //   return *this;
  // }

  // template<typename Scalar>
  // TLinearTerm<Scalar>& TLinearTerm<Scalar>::operator+=(VarPtr v) {
  //   this->addVar(1.0, v);
  //   return *this;
  // }

  // // operator overloading for syntax sugar:
  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator+(TLinearTermPtr<Scalar> a1, TLinearTermPtr<Scalar> a2) {
  //   TLinearTermPtr<Scalar> sum = Teuchos::rcp( new TLinearTerm<Scalar>(*a1) );
  //   //  cout << "sum->rank(): " << sum->rank() << endl;
  //   //  cout << "sum->summands.size() before adding a2: " << sum->summands().size() << endl;
  //   *sum += *a2;
  //   //  cout << "sum->summands.size() after adding a2: " << sum->summands().size() << endl;
  //   return sum;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator+(VarPtr v, TLinearTermPtr<Scalar> a) {
  //   TLinearTermPtr<Scalar> sum = Teuchos::rcp( new TLinearTerm<Scalar>(*a) );
  //   *sum += v;
  //   return sum;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator+(TLinearTermPtr<Scalar> a, VarPtr v) {
  //   return v + a;
  // }

  // TLinearTermPtr<double> operator+(VarPtr v1, VarPtr v2) {
  //   TFunctionPtr<double> one = Teuchos::rcp( new ConstantScalarFunction<double>(1.0, "") );
  //   return one * v1 + one * v2;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator*(TFunctionPtr<Scalar> f, VarPtr v) {
  //   return Teuchos::rcp( new TLinearTerm<Scalar>(f, v) );
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator*(VarPtr v, TFunctionPtr<Scalar> f) {
  //   return f * v;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator*(Scalar weight, VarPtr v) {
  //   return Teuchos::rcp( new TLinearTerm<Scalar>(weight, v) );
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator*(VarPtr v, Scalar weight) {
  //   return weight * v;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator*(vector<Scalar> weight, VarPtr v) {
  //   return Teuchos::rcp( new TLinearTerm<Scalar>(weight, v) );
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator*(VarPtr v, vector<Scalar> weight) {
  //   return weight * v;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator*(TFunctionPtr<Scalar> f, TLinearTermPtr<Scalar> a) {
  //   TLinearTermPtr<Scalar> lt = Teuchos::rcp( new TLinearTerm<Scalar> );

  //   for (typename vector< TLinearSummand<Scalar> >::const_iterator lsIt = a->summands().begin(); lsIt != a->summands().end(); lsIt++) {
  //     TLinearSummand<Scalar> ls = *lsIt;
  //     TFunctionPtr<Scalar> lsWeight = ls.first;
  //     TFunctionPtr<Scalar> newWeight = f * lsWeight;
  //     VarPtr var = ls.second;
  //     bool bypassTypeCheck = true; // unless user bypassed it, there will already have been a type check in the construction of a.  If the user did bypass, we should bypass, too.
  //     lt->addTerm(newWeight * var, bypassTypeCheck);
  //   }
  //   return lt;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator*(TLinearTermPtr<Scalar> a, TFunctionPtr<Scalar> f)
  // {
  //   TLinearTermPtr<Scalar> lt = Teuchos::rcp( new TLinearTerm<Scalar> );

  //   for (typename vector< TLinearSummand<Scalar> >::const_iterator lsIt = a->summands().begin(); lsIt != a->summands().end(); lsIt++) {
  //     TLinearSummand<Scalar> ls = *lsIt;
  //     TFunctionPtr<Scalar> lsWeight = ls.first;
  //     TFunctionPtr<Scalar> newWeight = lsWeight * f;
  //     VarPtr var = ls.second;
  //     bool bypassTypeCheck = true; // unless user bypassed it, there will already have been a type check in the construction of a.  If the user did bypass, we should bypass, too.
  //     lt->addTerm(newWeight * var, bypassTypeCheck);
  //   }
  //   return lt;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator/(TLinearTermPtr<Scalar> a, TFunctionPtr<Scalar> f) {
  //   TLinearTermPtr<Scalar> lt = Teuchos::rcp( new TLinearTerm<Scalar> );

  //   for (typename vector< TLinearSummand<Scalar> >::const_iterator lsIt = lt->summands().begin(); lsIt != lt->summands().end(); lsIt++) {
  //     TLinearSummand<Scalar> ls = *lsIt;
  //     TFunctionPtr<Scalar> lsWeight = ls.first;
  //     TFunctionPtr<Scalar> newWeight = lsWeight / f;
  //     VarPtr var = ls.second;
  //     bool bypassTypeCheck = true; // unless user bypassed it, there will already have been a type check in the construction of a.  If the user did bypass, we should bypass, too.
  //     lt->addTerm(newWeight * var, bypassTypeCheck);
  //   }
  //   return lt;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator/(VarPtr v, TFunctionPtr<Scalar> scalarFunction) {
  //   TFunctionPtr<Scalar> one = Teuchos::rcp( new ConstantScalarFunction<Scalar>(1.0) );
  //   return (one / scalarFunction) * v;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator/(VarPtr v, Scalar weight) {
  //   return (1.0 / weight) * v;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator-(TLinearTermPtr<Scalar> a) {
  //   TFunctionPtr<Scalar> minus_one = Teuchos::rcp( new ConstantScalarFunction<Scalar>(-1.0, "-") );
  //   return minus_one * a;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator-(TLinearTermPtr<Scalar> a, VarPtr v) {
  //   TFunctionPtr<Scalar> minus_one = Teuchos::rcp( new ConstantScalarFunction<Scalar>(-1.0, "-") );
  //   return a + minus_one * v;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator-(VarPtr v, TLinearTermPtr<Scalar> a) {
  //   TFunctionPtr<Scalar> minus_one = Teuchos::rcp( new ConstantScalarFunction<Scalar>(-1.0, "-") );
  //   return v + minus_one * a;
  // }

  // template <typename Scalar>
  // TLinearTermPtr<Scalar> operator-(TLinearTermPtr<Scalar> a1, TLinearTermPtr<Scalar> a2) {
  //   return a1 + -a2;
  // }

  // TLinearTermPtr<double> operator-(VarPtr v1, VarPtr v2) {
  //   TFunctionPtr<double> minus_one = Teuchos::rcp( new ConstantScalarFunction<double>(-1.0, "-") );
  //   return v1 + minus_one * v2;
  // }

  // TLinearTermPtr<double> operator-(VarPtr v) {
  //   TFunctionPtr<double> minus_one = Teuchos::rcp( new ConstantScalarFunction<double>(-1.0, "-") );
  //   return minus_one * v;
  // }

  // template TLinearTermPtr<double> operator+(TLinearTermPtr<double> a1, TLinearTermPtr<double> a2);
  // template TLinearTermPtr<double> operator+(VarPtr v, TLinearTermPtr<double> a);
  // template TLinearTermPtr<double> operator+(TLinearTermPtr<double> a, VarPtr v);
  // // template TLinearTermPtr<double> operator+(VarPtr v1, VarPtr v2);
  // template TLinearTermPtr<double> operator*(TFunctionPtr<double> f, VarPtr v);
  // template TLinearTermPtr<double> operator*(VarPtr v, TFunctionPtr<double> f);
  // template TLinearTermPtr<double> operator*(double weight, VarPtr v);
  // template TLinearTermPtr<double> operator*(VarPtr v, double weight);
  // template TLinearTermPtr<double> operator*(vector<double> weight, VarPtr v);
  // template TLinearTermPtr<double> operator*(VarPtr v, vector<double> weight);
  // template TLinearTermPtr<double> operator*(TFunctionPtr<double> f, TLinearTermPtr<double> a);
  // template TLinearTermPtr<double> operator*(TLinearTermPtr<double> a, TFunctionPtr<double> f);
  // template TLinearTermPtr<double> operator/(VarPtr v, double weight);
  // template TLinearTermPtr<double> operator/(VarPtr v, TFunctionPtr<double> f);
  // template TLinearTermPtr<double> operator-(TLinearTermPtr<double> a);
  // template TLinearTermPtr<double> operator-(TLinearTermPtr<double> a, VarPtr v);
  // template TLinearTermPtr<double> operator-(VarPtr v, TLinearTermPtr<double> a);
  // template TLinearTermPtr<double> operator-(TLinearTermPtr<double> a1, TLinearTermPtr<double> a2);
  // // template TLinearTermPtr<double> operator-(VarPtr v1, VarPtr v2);
  // // template TLinearTermPtr<double> operator-(VarPtr v);
}
