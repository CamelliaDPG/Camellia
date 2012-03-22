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
#include "Vectorized_Basis.hpp"

typedef Teuchos::RCP< FieldContainer<double> > FCPtr;
typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;
typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
typedef Teuchos::RCP<Vectorized_Basis<double, FieldContainer<double> > > VectorBasisPtr;

FCPtr BasisEvaluation::getValues(BasisPtr basis, EOperatorExtended op,
                                 const FieldContainer<double> &refPoints) {
  int numPoints = refPoints.dimension(0);
  int spaceDim = refPoints.dimension(1);  // points dimensions are (numPoints, spaceDim)
  int basisCardinality = basis->getCardinality();
  // test to make sure that the basis is known by BasisFactory--otherwise, throw exception
  if (! BasisFactory::basisKnown(basis) ) {
    TEST_FOR_EXCEPTION(true,std::invalid_argument,
                       "Unknown basis.  BasisCache only works for bases created by BasisFactory");
  }
  int componentOfInterest = -1;
  // otherwise, lookup to see whether a related value is already known
  EFunctionSpaceExtended fs = BasisFactory::getBasisFunctionSpace(basis);
  EOperator relatedOp = relatedOperator(op, fs, componentOfInterest);
  
  if ((EOperatorExtended)relatedOp != op) {
      // we can assume relatedResults has dimensions (numPoints,basisCardinality,spaceDim)
    FCPtr relatedResults = Teuchos::rcp(new FieldContainer<double>(basisCardinality,numPoints,spaceDim));
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
  if ( (op >= IntrepidExtendedTypes::OPERATOR_X) || (op < IntrepidExtendedTypes::OPERATOR_VALUE) ) {
    TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unknown operator.");
  }
  
  // result dimensions should be either (numPoints,basisCardinality) or (numPoints,basisCardinality,spaceDim);
  Teuchos::Array<int> dimensions;
  dimensions.push_back(basisCardinality);
  dimensions.push_back(numPoints);
  if ( ( ( BasisFactory::getBasisRank(basis) == 1) && (op == IntrepidExtendedTypes::OPERATOR_VALUE) )
      ||
      ( ( BasisFactory::getBasisRank(basis) == 0) && (op == IntrepidExtendedTypes::OPERATOR_GRAD) ) )
  {
    dimensions.push_back(spaceDim);
  } else if ( (BasisFactory::getBasisRank(basis) == 1) && (op == IntrepidExtendedTypes::OPERATOR_GRAD) ) {
    // grad of vector: a tensor
    dimensions.push_back(spaceDim);
    dimensions.push_back(spaceDim);
  }
  FCPtr result = Teuchos::rcp(new FieldContainer<double>(dimensions));
  basis->getValues(*(result.get()), refPoints, (EOperator)op);
  return result;
}

FCPtr BasisEvaluation::getTransformedValues(BasisPtr basis, EOperatorExtended op,
                                            const FieldContainer<double> &referencePoints,
                                            const FieldContainer<double> &cellJacobian, 
                                            const FieldContainer<double> &cellJacobianInv,
                                            const FieldContainer<double> &cellJacobianDet) {
  EFunctionSpaceExtended fs = BasisFactory::getBasisFunctionSpace(basis);
  int componentOfInterest;
  Intrepid::EOperator relatedOp;
  relatedOp = relatedOperator(op, fs, componentOfInterest);
  
  FCPtr referenceValues = getValues(basis,(EOperatorExtended) relatedOp, referencePoints);
  return getTransformedValuesWithBasisValues(basis,op,referenceValues,cellJacobian,cellJacobianInv,cellJacobianDet);
}

FCPtr BasisEvaluation::getTransformedVectorValuesWithComponentBasisValues(VectorBasisPtr basis, EOperatorExtended op,
                                                                          constFCPtr componentReferenceValuesTransformed) {
  EFunctionSpaceExtended fs = BasisFactory::getBasisFunctionSpace(basis);
  if ((fs != IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) || 
      ((op != IntrepidExtendedTypes::OPERATOR_VALUE) && (op != IntrepidExtendedTypes::OPERATOR_CROSS_NORMAL) )) {
    TEST_FOR_EXCEPTION( (fs != IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) || (op != IntrepidExtendedTypes::OPERATOR_VALUE),
                       std::invalid_argument, "Only Vector HGRAD with OPERATOR_VALUE supported by getTransformedVectorValuesWithComponentBasisValues.  Please use getTransformedValuesWithBasisValues instead.");
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

FCPtr BasisEvaluation::getTransformedValuesWithBasisValues(BasisPtr basis, EOperatorExtended op,
                                                           constFCPtr referenceValues,
                                                           const FieldContainer<double> &cellJacobian, 
                                                           const FieldContainer<double> &cellJacobianInv,
                                                           const FieldContainer<double> &cellJacobianDet) {
  typedef FunctionSpaceTools fst;
  int numCells = cellJacobian.dimension(0);
  int componentOfInterest;
  EFunctionSpaceExtended fs = BasisFactory::getBasisFunctionSpace(basis);
  Intrepid::EOperator relatedOp = relatedOperator(op,fs, componentOfInterest);
  Teuchos::Array<int> dimensions;
  referenceValues->dimensions(dimensions);
  dimensions.insert(dimensions.begin(), numCells);
  Teuchos::RCP<FieldContainer<double> > transformedValues = Teuchos::rcp(new FieldContainer<double>(dimensions));
  if ((fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) && (op == IntrepidExtendedTypes::OPERATOR_VALUE)) {
    TEST_FOR_EXCEPTION( (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) && (op == IntrepidExtendedTypes::OPERATOR_VALUE),
                       std::invalid_argument, "Vector HGRAD with OPERATOR_VALUE not supported by getTransformedValuesWithBasisValues.  Please use getTransformedVectorValuesWithComponentBasisValues instead.");
  }
  switch(relatedOp) {
    case(Intrepid::OPERATOR_VALUE):
      switch(fs) {
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
        case IntrepidExtendedTypes::CURL_HGRAD_FOR_CONSERVATION:
          // TODO: figure out the right thing here...
          TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
          break;
        default:
          TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
          break;
      }
      break;
    case(IntrepidExtendedTypes::OPERATOR_GRAD):
      switch(fs) {
        case IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD: // HGRAD is the only space that supports the GRAD operator...
          fst::HGRADtransformGRAD<double>(*transformedValues,cellJacobianInv,*referenceValues);
          break;
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
          TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
          break;
      }
      break;
    case(IntrepidExtendedTypes::OPERATOR_CURL):
      switch(fs) {
        case IntrepidExtendedTypes::FUNCTION_SPACE_HCURL:
          fst::HCURLtransformCURL<double>(*transformedValues,cellJacobian,cellJacobianDet,*referenceValues);
          break;
        case IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD:
          // in 2D, HGRADtransformCURL == HDIVtransformVALUE (because curl(H1) --> H(div))
          fst::HDIVtransformVALUE<double>(*transformedValues,cellJacobian,cellJacobianDet,*referenceValues);
          break;
        case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD: // shouldn't take the transform so late
        default:
          TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
          break;
      }
      break;
    case(IntrepidExtendedTypes::OPERATOR_DIV):
      switch(fs) {
        case IntrepidExtendedTypes::CURL_HGRAD_FOR_CONSERVATION: // these live in HDIV...
        case IntrepidExtendedTypes::FUNCTION_SPACE_HDIV:
          fst::HDIVtransformDIV<double>(*transformedValues,cellJacobianDet,*referenceValues);
          break;
        case IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD: // shouldn't take the transform so late
        default:
          TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
          break;
      }
      break;
    default:
      TEST_FOR_EXCEPTION(true,std::invalid_argument, "unhandled transformation");
      break;
  }

  FCPtr result = getComponentOfInterest(transformedValues,op,fs,componentOfInterest);
  if ( result.get() == 0 ) {
    result = transformedValues;
  }
  return result;
}

Intrepid::EOperator BasisEvaluation::relatedOperator(EOperatorExtended op, EFunctionSpaceExtended fs, int &componentOfInterest) {
  Intrepid::EOperator relatedOp = (Intrepid::EOperator) op;
  componentOfInterest = -1;
  if (   ( fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) 
      && ( (op == IntrepidExtendedTypes::OPERATOR_CURL) || (op == IntrepidExtendedTypes::OPERATOR_DIV) ) ) {
    relatedOp = Intrepid::OPERATOR_GRAD;
  } else if ((op==OPERATOR_X) || (op==OPERATOR_Y) || (op==OPERATOR_Z)) {
    componentOfInterest = op - OPERATOR_X;
    relatedOp = Intrepid::OPERATOR_VALUE;
  } else if ((op==OPERATOR_DX) || (op==OPERATOR_DY) || (op==OPERATOR_DZ)) {
    componentOfInterest = op - OPERATOR_DX;
    relatedOp = Intrepid::OPERATOR_GRAD;
  } else if (   (op==OPERATOR_CROSS_NORMAL) || (op==OPERATOR_DOT_NORMAL) || (op==OPERATOR_TIMES_NORMAL)
             || (op==OPERATOR_VECTORIZE_VALUE) ) {
    relatedOp = Intrepid::OPERATOR_VALUE;
  }
  return relatedOp;
}

FCPtr BasisEvaluation::getComponentOfInterest(constFCPtr values, EOperatorExtended op, EFunctionSpaceExtended fs, int componentOfInterest) {
  FCPtr result;
  if (   (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD)
      && ( (op == IntrepidExtendedTypes::OPERATOR_CURL) || (op == IntrepidExtendedTypes::OPERATOR_DIV)) ) {
    TEST_FOR_EXCEPTION(values->rank() != 5, std::invalid_argument, "rank of values must be 5 for VECTOR_HGRAD_GRAD");
    int numCells  = values->dimension(0);
    int numFields = values->dimension(1);
    int numPoints = values->dimension(2);
    result = Teuchos::rcp(new FieldContainer<double>(numCells,numFields,numPoints));
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int field=0; field<numFields; field++) {
        for (int point=0; point<numPoints; point++) {
          if (op==IntrepidExtendedTypes::OPERATOR_DIV) {
            (*result)(cellIndex,field,point) = (*values)(cellIndex,field,point,0,0) + (*values)(cellIndex,field,point,1,1); 
          } else if (op==IntrepidExtendedTypes::OPERATOR_CURL) {
            // curl of 2D vector: d(Fx)/dy - d(Fy)/dx
            (*result)(cellIndex,field,point) = (*values)(cellIndex,field,point,1,0) - (*values)(cellIndex,field,point,0,1); 
          }
        }
      }
    }
    return result;
  } else if (componentOfInterest < 0) { // then just return values
    return result;
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
  } else {
    TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported values container rank.");
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
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "crossing with normal only supported for 2D right now");
  }
  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1))
      || (numPoints != values->dimension(2)) || (spaceDim != values->dimension(3)) 
      ) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P,D)");
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
        (*result)(cellIndex,basisOrdinal,pointIndex) = xValue*n2 - yValue*n1;
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
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "dotting with normal only supported for 2D right now");
  }
  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1))
      || (numPoints != values->dimension(2)) || (spaceDim != values->dimension(3)) 
      ) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P,D)");
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
  // values should have dimensions (C,basisCardinality,P)
  int numCells = sideNormals.dimension(0);
  int numPoints = sideNormals.dimension(1);
  int spaceDim = sideNormals.dimension(2);
  int basisCardinality = values->dimension(1);
  
  if ( (numCells != values->dimension(0)) || (basisCardinality != values->dimension(1))
      || (numPoints != values->dimension(2))) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P)");
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
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P)");
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
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "values should have dimensions (C,basisCardinality,P)");
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
