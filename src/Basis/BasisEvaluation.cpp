//
//  BasisEvaluation.cpp
//  DPGTrilinos
//
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
//

#include <iostream>

#include "BasisEvaluation.h"
#include "BasisFactory.h"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

typedef Teuchos::RCP< FieldContainer<double> > FCPtr;
typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;

FCPtr BasisEvaluation::getValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op,
                                 const FieldContainer<double> &refPoints) {
  int numPoints = refPoints.dimension(0);
  int spaceDim = refPoints.dimension(1);  // points dimensions are (numPoints, spaceDim)
  int basisCardinality = basis->getCardinality();
  int spaceDimOut = spaceDim; // for now, we assume basis values are in the same spaceDim as points (e.g. vector 1D has just 1 component)
  // test to make sure that the basis is known by BasisFactory--otherwise, throw exception
  int componentOfInterest = -1;
  // otherwise, lookup to see whether a related value is already known
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = basis->functionSpace();
  EOperator relatedOp = relatedOperator(op, fs, componentOfInterest);
  
  if ((EOperatorExtended)relatedOp != op) {
      // we can assume relatedResults has dimensions (numPoints,basisCardinality,spaceDimOut)
    FCPtr relatedResults = Teuchos::rcp(new FieldContainer<double>(basisCardinality,numPoints,spaceDimOut));
    basis->getValues(*(relatedResults.get()), refPoints, (EOperator)relatedOp);
    FCPtr result = getComponentOfInterest(relatedResults,op,fs,componentOfInterest);
    if ( result.get() == 0 ) {
      result = relatedResults;
    }
    return result;
  }
  // if we get here, we should have a standard Intrepid operator, in which case we should
  // be able to: size a FieldContainer appropriately, and then call basis->getValues
  
  // But let's do just check that we have a standard Intrepid operator
  if ( (op >= IntrepidExtendedTypes::OP_X) || (op <  IntrepidExtendedTypes::OP_VALUE) ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unknown operator.");
  }
  
  // result dimensions should be either (numPoints,basisCardinality) or (numPoints,basisCardinality,spaceDimOut);
  Teuchos::Array<int> dimensions;
  dimensions.push_back(basisCardinality);
  dimensions.push_back(numPoints);
  int basisRank = basis->rangeRank();
  if ( ( ( basisRank == 1) && (op ==  IntrepidExtendedTypes::OP_VALUE) ) ){
    dimensions.push_back(spaceDimOut);
  } else if (
      ( ( basisRank == 0) && (op == IntrepidExtendedTypes::OP_GRAD) )
      ||
      ( ( basisRank == 0) && (op == IntrepidExtendedTypes::OP_CURL) ) )
  {
    dimensions.push_back(spaceDim);
  } else if ( (basis->rangeRank() == 1) && (op == IntrepidExtendedTypes::OP_GRAD) ) {
    // grad of vector: a tensor
    dimensions.push_back(spaceDim);
    dimensions.push_back(spaceDimOut);
  }
  FCPtr result = Teuchos::rcp(new FieldContainer<double>(dimensions));
  basis->getValues(*(result.get()), refPoints, (EOperator)op);
  return result;
}

FCPtr BasisEvaluation::getTransformedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op,
                                            const FieldContainer<double> &referencePoints,
                                            const FieldContainer<double> &cellJacobian, 
                                            const FieldContainer<double> &cellJacobianInv,
                                            const FieldContainer<double> &cellJacobianDet) {
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = basis->functionSpace();
  int componentOfInterest;
  Intrepid::EOperator relatedOp;
  relatedOp = relatedOperator(op, fs, componentOfInterest);
  
  FCPtr referenceValues = getValues(basis,(EOperatorExtended) relatedOp, referencePoints);
  return getTransformedValuesWithBasisValues(basis,op,referenceValues,cellJacobian,cellJacobianInv,cellJacobianDet);
}

FCPtr BasisEvaluation::getTransformedVectorValuesWithComponentBasisValues(VectorBasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op,
                                                                          constFCPtr componentReferenceValuesTransformed) {
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = basis->functionSpace();
  bool vectorizedBasis = (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) || (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL);
  if ( !vectorizedBasis || 
      ((op !=  IntrepidExtendedTypes::OP_VALUE) && (op != IntrepidExtendedTypes::OP_CROSS_NORMAL) )) {
    TEUCHOS_TEST_FOR_EXCEPTION( !vectorizedBasis || (op !=  IntrepidExtendedTypes::OP_VALUE),
                       std::invalid_argument, "Only Vector HGRAD/HVOL with OP_VALUE supported by getTransformedVectorValuesWithComponentBasisValues.  Please use getTransformedValuesWithBasisValues instead.");
  }
  BasisPtr componentBasis = basis->getComponentBasis();
  Teuchos::Array<int> dimensions;
  componentReferenceValuesTransformed->dimensions(dimensions);
  dimensions[1] = basis->getCardinality(); // dimensions are (C,F,P,D)
  dimensions.push_back(basis->getNumComponents());
  Teuchos::RCP<FieldContainer<double> > transformedValues = Teuchos::rcp(new FieldContainer<double>(dimensions));
  int fieldIndex = 1;
  basis->getVectorizedValues(*transformedValues, *componentReferenceValuesTransformed,fieldIndex);
  return transformedValues;
}

FCPtr BasisEvaluation::getTransformedValuesWithBasisValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op,
                                                           constFCPtr referenceValues,
                                                           const FieldContainer<double> &cellJacobian, 
                                                           const FieldContainer<double> &cellJacobianInv,
                                                           const FieldContainer<double> &cellJacobianDet) {
  typedef FunctionSpaceTools fst;
  int numCells = cellJacobian.dimension(0);
  int spaceDim = cellJacobian.dimension(2);
  int componentOfInterest;
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = basis->functionSpace();
  Intrepid::EOperator relatedOp = relatedOperator(op,fs, componentOfInterest);
  Teuchos::Array<int> dimensions;
  referenceValues->dimensions(dimensions);
  dimensions.insert(dimensions.begin(), numCells);
  Teuchos::RCP<FieldContainer<double> > transformedValues = Teuchos::rcp(new FieldContainer<double>(dimensions));
  bool vectorizedBasis = functionSpaceIsVectorized(fs);
  if (vectorizedBasis && (op ==  IntrepidExtendedTypes::OP_VALUE)) {
    TEUCHOS_TEST_FOR_EXCEPTION( vectorizedBasis && (op ==  IntrepidExtendedTypes::OP_VALUE),
                       std::invalid_argument, "Vector HGRAD/HVOL with OP_VALUE not supported by getTransformedValuesWithBasisValues.  Please use getTransformedVectorValuesWithComponentBasisValues instead.");
  }
  switch(relatedOp) {
    case(Intrepid::OPERATOR_VALUE):
      switch(fs) {
        case IntrepidExtendedTypes::FUNCTION_SPACE_ONE:
//          cout << "Reference values for FUNCTION_SPACE_ONE: " << *referenceValues;
        case IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD:
          fst::HGRADtransformVALUE<double>(*transformedValues,*referenceValues);
          break;
        case IntrepidExtendedTypes::FUNCTION_SPACE_HCURL:
          fst::HCURLtransformVALUE<double>(*transformedValues,cellJacobianInv,*referenceValues);
          break;
        case IntrepidExtendedTypes::FUNCTION_SPACE_HDIV:
          fst::HDIVtransformVALUE<double>(*transformedValues,cellJacobian,cellJacobianDet,*referenceValues);
          break;
        case IntrepidExtendedTypes::FUNCTION_SPACE_HVOL:
          // for the moment, use the fact that we know the HVOL basis is always an HGRAD basis:
          fst::HGRADtransformVALUE<double>(*transformedValues,*referenceValues);
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
          break;
      }
      break;
    case(IntrepidExtendedTypes::OP_GRAD):
      switch(fs) {
        case IntrepidExtendedTypes::FUNCTION_SPACE_HVOL:
        case IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD: // HGRAD is the only space that supports the GRAD operator...
          fst::HGRADtransformGRAD<double>(*transformedValues,cellJacobianInv,*referenceValues);
          break;
        case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL:
        case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD:
          // referenceValues has dimensions (F,P,D1,D2).  D1 is our component dimension, and D2 is the one that came from the gradient.
          // HGRADtransformGRAD expects (F,P,D) for input, and (C,F,P,D) for output.
          // If we split referenceValues into (F,P,D1,D2=0) and (F,P,D1,D2=1), then we can transform each of those, and then interleave the results…
        { // block off so we can create new stuff inside the switch case
          Teuchos::Array<int> dimensions;
          referenceValues->dimensions(dimensions);
          int D2 = dimensions[dimensions.size() - 1];
          dimensions.pop_back(); // get rid of D2
          FieldContainer<double> refValuesSlice(dimensions);
          dimensions.insert(dimensions.begin(),numCells);
          FieldContainer<double> transformedValuesSlice(dimensions);
          
          int numEntriesPerSlice = refValuesSlice.size();
          int numEntriesPerTransformedSlice = transformedValuesSlice.size();
          for (int compIndex=0; compIndex<D2; compIndex++) {
            for (int i=0; i<numEntriesPerSlice; i++) {
              refValuesSlice[i] = (*referenceValues)[i*D2 + compIndex];
            }
            fst::HGRADtransformGRAD<double>(transformedValuesSlice,cellJacobianInv,refValuesSlice);
            for (int i=0; i<numEntriesPerTransformedSlice; i++) {
              (*transformedValues)[i*D2 + compIndex] = transformedValuesSlice[i];
            }
          }
        }
          
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
          break;
      }
      break;
    case(IntrepidExtendedTypes::OP_CURL):
      switch(fs) {
        case IntrepidExtendedTypes::FUNCTION_SPACE_HCURL:
          if (spaceDim == 2) {
            // TODO: confirm that this is correct
//            static bool warningIssued = false;
//            if ( ! warningIssued ) {
//              cout << "WARNING: for HCURL in 2D, transforming with HVOLtransformVALUE. Need to confirm this is correct.\n";
//              warningIssued = true;
//            }
            fst::HVOLtransformVALUE<double>(*transformedValues, cellJacobianDet, *referenceValues);
          } else {
            fst::HCURLtransformCURL<double>(*transformedValues,cellJacobian,cellJacobianDet,*referenceValues);
          }
          break;
        case IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD:
          // in 2D, HGRADtransformCURL == HDIVtransformVALUE (because curl(H1) --> H(div))
          fst::HDIVtransformVALUE<double>(*transformedValues,cellJacobian,cellJacobianDet,*referenceValues);
          break;
        case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL:
        case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD: // shouldn't take the transform so late
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
          break;
      }
      break;
    case(IntrepidExtendedTypes::OP_DIV):
      switch(fs) {
        case IntrepidExtendedTypes::FUNCTION_SPACE_HDIV:
          fst::HDIVtransformDIV<double>(*transformedValues,cellJacobianDet,*referenceValues);
          break;
        case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL:
        case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD: // shouldn't take the transform so late
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
          break;
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
      break;
  }

  FCPtr result = getComponentOfInterest(transformedValues,op,fs,componentOfInterest);
  if ( result.get() == 0 ) {
    result = transformedValues;
  }
  return result;
}

Intrepid::EOperator BasisEvaluation::relatedOperator(EOperatorExtended op, IntrepidExtendedTypes::EFunctionSpaceExtended fs, int &componentOfInterest) {
  Intrepid::EOperator relatedOp = (Intrepid::EOperator) op;
  componentOfInterest = -1;

  bool vectorizedBasis = (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) || (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL);

  if (   vectorizedBasis
      && ( (op == IntrepidExtendedTypes::OP_CURL) || (op == IntrepidExtendedTypes::OP_DIV) ) ) {
    relatedOp = Intrepid::OPERATOR_GRAD;
  } else if ((op==OP_X) || (op==OP_Y) || (op==OP_Z)) {
    componentOfInterest = op - OP_X;
    relatedOp = Intrepid::OPERATOR_VALUE;
  } else if ((op==OP_DX) || (op==OP_DY) || (op==OP_DZ)) {
    componentOfInterest = op - OP_DX;
    relatedOp = Intrepid::OPERATOR_GRAD;
  } else if (   (op==OP_CROSS_NORMAL)   || (op==OP_DOT_NORMAL)     || (op==OP_TIMES_NORMAL)
             || (op==OP_TIMES_NORMAL_X) || (op==OP_TIMES_NORMAL_Y) || (op==OP_TIMES_NORMAL_Z) 
             || (op==OP_VECTORIZE_VALUE)) {
    relatedOp = Intrepid::OPERATOR_VALUE;
  }
  return relatedOp;
}

FCPtr BasisEvaluation::getComponentOfInterest(constFCPtr values, IntrepidExtendedTypes::EOperatorExtended op, IntrepidExtendedTypes::EFunctionSpaceExtended fs, int componentOfInterest) {
  FCPtr result;
  bool vectorizedBasis = (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) || (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL);
  if (   vectorizedBasis
      && ( (op == IntrepidExtendedTypes::OP_CURL) || (op == IntrepidExtendedTypes::OP_DIV)) ) {
    TEUCHOS_TEST_FOR_EXCEPTION(values->rank() != 5, std::invalid_argument, "rank of values must be 5 for VECTOR_HGRAD_GRAD");
    int numCells  = values->dimension(0);
    int numFields = values->dimension(1);
    int numPoints = values->dimension(2);
    result = Teuchos::rcp(new FieldContainer<double>(numCells,numFields,numPoints));
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int field=0; field<numFields; field++) {
        for (int point=0; point<numPoints; point++) {
          if (op==IntrepidExtendedTypes::OP_DIV) {
            (*result)(cellIndex,field,point) = (*values)(cellIndex,field,point,0,0) + (*values)(cellIndex,field,point,1,1); 
          } else if (op==IntrepidExtendedTypes::OP_CURL) {
            // curl of 2D vector: d(Fx)/dy - d(Fy)/dx
            (*result)(cellIndex,field,point) = (*values)(cellIndex,field,point,1,0) - (*values)(cellIndex,field,point,0,1); 
          }
        }
      }
    }
    return result;
  } else if (componentOfInterest < 0) { // then just return values
    // the copy is a bit unfortunate, but can't be avoided unless we change a bunch of constFCPtrs to FCPtrs (or vice versa)
    // in the API...
//    cout << "values:\n" << *values;
    return Teuchos::rcp( new FieldContainer<double>(*values));
  }
  Teuchos::Array<int> dimensions;
  values->dimensions(dimensions);
  int spaceDim = dimensions[dimensions.size()-1];
  dimensions.pop_back(); // get rid of last, spatial dimension
  result = Teuchos::rcp(new FieldContainer<double>(dimensions));
  int numPoints = dimensions[0];
  int basisCardinality = dimensions[1];
  int size = result->size();
  int enumeratedLocation;
  if (values->rank() == 3) {
    enumeratedLocation = values->getEnumeration(0,0,componentOfInterest);
  } else if (values->rank() == 4) {
    enumeratedLocation = values->getEnumeration(0,0,0,componentOfInterest);
  } else if (values->rank() == 5) {
    enumeratedLocation = values->getEnumeration(0,0,0,0,componentOfInterest);
  } else {
    // TODO: consider computing the enumerated location in a rank-independent way.
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported values container rank.");
  }
  for (int i=0; i<size; i++) {
    (*result)[i] = (*values)[enumeratedLocation];
    enumeratedLocation += spaceDim;
  }
  return result;
}

FCPtr BasisEvaluation::getValuesCrossedWithNormals(constFCPtr values,const FieldContainer<double> &sideNormals) {
  // values should have dimensions (C,basisCardinality,P,D)
  int numCells = sideNormals.dimension(0);
  int numPoints = sideNormals.dimension(1);
  int spaceDim = sideNormals.dimension(2);
  int basisCardinality = values->dimension(1);
  
  if (spaceDim != 2) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "crossing with normal only supported for 2D right now");
  }
  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1))
      || (numPoints != values->dimension(2)) || (spaceDim != values->dimension(3)) 
      ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P,D)");
  }
  
  Teuchos::RCP< FieldContainer<double> > result = Teuchos::rcp(new FieldContainer<double>(numCells,basisCardinality,numPoints));
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
      for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
        double n1 = sideNormals(cellIndex,pointIndex,0);
        double n2 = sideNormals(cellIndex,pointIndex,1);
        double xValue = (*values)(cellIndex,basisOrdinal,pointIndex,0);
        double yValue = (*values)(cellIndex,basisOrdinal,pointIndex,1);
        /*cout << "(n1,n2) = (" << n1 << ", " << n2 << ")" << endl;
        cout << "(x,y) = (" << xValue << ", " << yValue << ")" << endl;
        cout << "(x,y) x n = " << xValue*n2 - yValue*n1 << endl;*/
//        cout << "WARNING: pretty sure that we've reversed the sign of OP_CROSS_NORMAL.\n";
        (*result)(cellIndex,basisOrdinal,pointIndex) = yValue*n1 - xValue*n2;
      }
    }
  }
  return result;
}

FCPtr BasisEvaluation::getValuesDottedWithNormals(constFCPtr values,const FieldContainer<double> &sideNormals) {
  // values should have dimensions (C,basisCardinality,P,D)
  int numCells = sideNormals.dimension(0);
  int numPoints = sideNormals.dimension(1);
  int spaceDim = sideNormals.dimension(2);
  int basisCardinality = values->dimension(1);
  
  if (spaceDim != 2) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "dotting with normal only supported for 2D right now");
  }
  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1))
      || (numPoints != values->dimension(2)) || (spaceDim != values->dimension(3)) 
      ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P,D)");
  }
  
  Teuchos::RCP< FieldContainer<double> > result = Teuchos::rcp(new FieldContainer<double>(numCells,basisCardinality,numPoints));
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
      for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
        double n1 = sideNormals(cellIndex,pointIndex,0);
        double n2 = sideNormals(cellIndex,pointIndex,1);
        double xValue = (*values)(cellIndex,basisOrdinal,pointIndex,0);
        double yValue = (*values)(cellIndex,basisOrdinal,pointIndex,1);
        (*result)(cellIndex,basisOrdinal,pointIndex) = xValue*n1 + yValue*n2;
      }
    }
  }
  return result;
}

FCPtr BasisEvaluation::getValuesTimesNormals(constFCPtr values,const FieldContainer<double> &sideNormals, int normalComponent) {
  // TODO: write tests against this version of getValuesTimesNormals
  // values should have dimensions (C,basisCardinality,P)
  int numCells = sideNormals.dimension(0);
  int numPoints = sideNormals.dimension(1);
  int spaceDim = sideNormals.dimension(2);
  int basisCardinality = values->dimension(1);
  
  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1))
      || (numPoints != values->dimension(2))) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P)");
  }
  
  Teuchos::RCP< FieldContainer<double> > result = Teuchos::rcp(new FieldContainer<double>(numCells,basisCardinality,numPoints));
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
      for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
        double n_i = sideNormals(cellIndex,pointIndex,normalComponent);
        double value = (*values)(cellIndex,basisOrdinal,pointIndex);
        (*result)(cellIndex,basisOrdinal,pointIndex) = value * n_i;
      }
    }
  }
  return result;
}

FCPtr BasisEvaluation::getValuesTimesNormals(constFCPtr values,const FieldContainer<double> &sideNormals) {
  // values should have dimensions (C,basisCardinality,P)
  int numCells = sideNormals.dimension(0);
  int numPoints = sideNormals.dimension(1);
  int spaceDim = sideNormals.dimension(2);
  int basisCardinality = values->dimension(1);
  
  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1))
      || (numPoints != values->dimension(2))) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P)");
  }
  
  Teuchos::RCP< FieldContainer<double> > result = Teuchos::rcp(new FieldContainer<double>(numCells,basisCardinality,numPoints,spaceDim));
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
      for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
        for (int dim=0; dim<spaceDim; dim++) {
          double n_dim = sideNormals(cellIndex,pointIndex,dim);
          double value = (*values)(cellIndex,basisOrdinal,pointIndex);
          (*result)(cellIndex,basisOrdinal,pointIndex,dim) = value * n_dim;
        }
      }
    }
  }
  return result;
}

FCPtr BasisEvaluation::getVectorizedValues(constFCPtr values, int spaceDim) {
  // values should have dimensions (C,basisCardinality,P)
  int numCells = values->dimension(0);
  int basisCardinality = values->dimension(1);
  int numPoints = values->dimension(2);
  
  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1)) || (numPoints != values->dimension(2))) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P)");
  }
  
  Teuchos::RCP< FieldContainer<double> > result = Teuchos::rcp(new FieldContainer<double>(numCells,basisCardinality,numPoints,spaceDim));
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
      for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
        for (int dim=0; dim<spaceDim; dim++) {
          (*result)(cellIndex,basisOrdinal,pointIndex,dim) = (*values)(cellIndex,basisOrdinal,pointIndex);
        }
      }
    }
  }
  return result;
}
