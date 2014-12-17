//
//  TensorBasisDef.h
//  Camellia
//
//  Created by Nathan Roberts on 11/12/14.
//
//

#define TENSOR_FIELD_ORDINAL(spaceFieldOrdinal,timeFieldOrdinal) timeFieldOrdinal * _spatialBasis->getCardinality() + spaceFieldOrdinal;

namespace Camellia {
  template<class Scalar, class ArrayScalar>
  TensorBasis<Scalar,ArrayScalar>::TensorBasis(Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > spatialBasis, Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > temporalBasis) {
    _spatialBasis = spatialBasis;
    _temporalBasis = temporalBasis;
    
    if (temporalBasis->domainTopology()->getDimension() != 1) {
      cout << "temporalBasis must have a line topology as its domain.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "temporalBasis must have a line topology as its domain.");
    }
    
    if (temporalBasis->rangeRank() > 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "only scalar temporal bases are supported by TensorBasis at present.");
    }
    
    if (_spatialBasis->domainTopology()->getTensorialDegree() > 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "only spatial bases defined on toplogies with 0 tensorial degree are supported by TensorBasis at present.");
    }
    
    int tensorialDegree = 1;
    this->_domainTopology = CellTopology::cellTopology(_spatialBasis->domainTopology()->getShardsTopology(), tensorialDegree);
  }

  template<class Scalar, class ArrayScalar>
  int TensorBasis<Scalar,ArrayScalar>::getCardinality() const {
    return _spatialBasis->getCardinality() * _temporalBasis->getCardinality();
  }

  template<class Scalar, class ArrayScalar>
  const Teuchos::RCP< Camellia::Basis<Scalar, ArrayScalar> > TensorBasis<Scalar, ArrayScalar>::getComponentBasis(int tensorialBasisRank) const {
    if (tensorialBasisRank==0) return _spatialBasis;
    if (tensorialBasisRank==1) return _temporalBasis;
    
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensorialBasisRank exceeds the tensorial degree of basis.");
  }

  template<class Scalar, class ArrayScalar>
  int TensorBasis<Scalar,ArrayScalar>::getDegree() const {
    return _spatialBasis->getDegree();
  }

  template<class Scalar, class ArrayScalar>
  void TensorBasis<Scalar,ArrayScalar>::getTensorValues(ArrayScalar& outputValues, std::vector< const ArrayScalar> & componentOutputValuesVector, std::vector<EOperator> operatorTypes) const {
    // outputValues can have dimensions (C,F,P,...) or (F,P,...)
    
    if (operatorTypes.size() != 2) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "only two-component tensor bases supported right now");
    }
    
    EOperator spatialOperator = operatorTypes[0];
    int rankAdjustment;
    switch (spatialOperator) {
      case(OPERATOR_VALUE):
        rankAdjustment = 0;
        break;
      case(OPERATOR_GRAD):
        rankAdjustment = 1;
        break;
      case(OPERATOR_CURL):
        if (this->rangeRank() == 1)
        break;
      case(OPERATOR_DIV):
        rankAdjustment = -1;
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled operator type.");
    }
    
    int valuesRank = this->rangeRank() + rankAdjustment; // should have 2 additional dimensions, plus optionally the cell dimension
    int fieldIndex;
    if (outputValues.rank() == valuesRank + 3) {
      fieldIndex = 1; // (C,F,...)
    } else if (outputValues.rank() == valuesRank + 2) {
      fieldIndex = 0; // (F,...)
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "outputValues shape must be either (C,F,P,...) or (F,P,...)");
    }
    
    const ArrayScalar* spatialValues = &componentOutputValuesVector[0];
    const ArrayScalar* temporalValues = &componentOutputValuesVector[1];
    
    //cout << "componentOutputValues: \n" << componentOutputValues;
    TEUCHOS_TEST_FOR_EXCEPTION( outputValues.dimension(fieldIndex) != this->getCardinality(),
                               std::invalid_argument, "outputValues.dimension(fieldIndex) != this->getCardinality()");
    TEUCHOS_TEST_FOR_EXCEPTION( spatialValues->dimension(fieldIndex) != _spatialBasis->getCardinality(),
                               std::invalid_argument, "spatialValues->dimension(fieldIndex) != _spatialBasis->getCardinality()");
    TEUCHOS_TEST_FOR_EXCEPTION( temporalValues->dimension(fieldIndex) != _temporalBasis->getCardinality(),
                               std::invalid_argument, "temporalValues->dimension(fieldIndex) != _temporalBasis->getCardinality()");
    int pointIndex = fieldIndex+1;
    TEUCHOS_TEST_FOR_EXCEPTION( outputValues.dimension(pointIndex) != spatialValues->dimension(pointIndex) * temporalValues->dimension(pointIndex),
                               std::invalid_argument, "outputValues.dimension(pointIndex) != spatialValues->dimension(pointIndex) * temporalValues->dimension(pointIndex)");
    Teuchos::Array<int> dimensions;
    outputValues.dimensions(dimensions);
    outputValues.initialize(0.0);
    int numPoints = dimensions[fieldIndex+1];
    int numComponents = dimensions[fieldIndex+2];
    // TODO: finish writing this...
    /*
    if (_numComponents != numComponents) {
      TEUCHOS_TEST_FOR_EXCEPTION ( _numComponents != numComponents, std::invalid_argument,
                                  "fieldIndex+2 dimension of outputValues must match the number of vector components.");
    }
    int componentCardinality = _componentBasis->getCardinality();
    
    for (int i=fieldIndex+3; i<dimensions.size(); i++) {
      dimensions[i] = 0;
    }
    
    int numCells = (fieldIndex==1) ? dimensions[0] : 1;
    
    int compValuesPerPoint = componentOutputValues.size() / (numCells * componentCardinality * numPoints);
    
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      if (fieldIndex==1) {
        dimensions[0] = cellIndex;
      }
      for (int field=0; field<componentCardinality; field++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          dimensions[fieldIndex+1] = ptIndex;
          dimensions[fieldIndex+2] = 0; // component dimension
          Teuchos::Array<int> compIndexArray = dimensions;
          compIndexArray.pop_back(); // removes the last 0 (conceptually it's fieldIndex + 2 that corresponds to the comp dimension, but it's all 0s from fieldIndex+2 on...)
          compIndexArray[fieldIndex] = field;
          int compEnumerationOffset = componentOutputValues.getEnumeration(compIndexArray);
          for (int comp=0; comp<numComponents; comp++) {
            dimensions[fieldIndex] = field + componentCardinality * comp;
            dimensions[fieldIndex+2] = comp;
            int outputEnumerationOffset = outputValues.getEnumeration(dimensions);
            const double *compValue = &componentOutputValues[compEnumerationOffset];
            double *outValue = &outputValues[outputEnumerationOffset];
            for (int i=0; i<compValuesPerPoint; i++) {
              *outValue++ = *compValue++;
            }
          }
        }
      }
    }*/
    //    cout << "getVectorizedValues: componentOutputValues:\n" << componentOutputValues;
    //    cout << "getVectorizedValues: outputValues:\n" << outputValues;
  }

  // range info for basis values:
  template<class Scalar, class ArrayScalar>
  int TensorBasis<Scalar,ArrayScalar>::rangeDimension() const {
    return _spatialBasis->rangeDimension();
  }
  
  template<class Scalar, class ArrayScalar>
  int TensorBasis<Scalar,ArrayScalar>::rangeRank() const {
    return _spatialBasis->rangeRank();
  }
  
  template<class Scalar, class ArrayScalar>
  void TensorBasis<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints,
                                                  Intrepid::EOperator operatorType) const {
    Intrepid::EOperator temporalOperator = OPERATOR_VALUE; // default
    if (operatorType==OPERATOR_GRAD) {
      if (values.rank() == 3) { // F, P, D
        if (_spatialBasis->rangeDimension() + _temporalBasis->rangeDimension() == values.dimension(2)) {
          // then we can/should use OPERATOR_GRAD for the temporal operator as well
          temporalOperator = OPERATOR_GRAD;
        }
      }
    }
    getValues(values, refPoints, operatorType, temporalOperator);
  }
  
  template<class Scalar, class ArrayScalar>
  void TensorBasis<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints,
                                                  EOperator spatialOperatorType, EOperator temporalOperatorType) const {
    if ((spatialOperatorType==OPERATOR_GRAD) && (temporalOperatorType==OPERATOR_GRAD)) {
      // i.e., we are taking the gradient of the whole, not just a time derivative or a spatial gradient.
      cout << "WARNING: support for taking OPERATOR_GRAD on both space and time not yet fully implemented.\n";
    } else {
      this->CHECK_VALUES_ARGUMENTS(values,refPoints,spatialOperatorType);
    }
    
    int numPoints = refPoints.dimension(0);
    
    Teuchos::Array<int> spaceTimePointsDimensions;
    refPoints.dimensions(spaceTimePointsDimensions);
    
    spaceTimePointsDimensions[1] -= 1; // spatial dimension is 1 less than space-time
    if (spaceTimePointsDimensions[1]==0) spaceTimePointsDimensions[1] = 1; // degenerate case: we still want points defined in the 0-dimensional case...
    ArrayScalar refPointsSpatial(spaceTimePointsDimensions);
    
    spaceTimePointsDimensions[1] = 1; // time is 1-dimensional
    ArrayScalar refPointsTemporal(spaceTimePointsDimensions);
    
    int spaceDim = _spatialBasis->domainTopology()->getDimension();
    // copy the points out:
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++) {
      for (int d=0; d<spaceDim; d++) {
        refPointsSpatial(pointOrdinal,d) = refPoints(pointOrdinal,d);
      }
      refPointsTemporal(pointOrdinal,0) = refPoints(pointOrdinal,spaceDim);
    }
    
    Teuchos::Array<int> spaceTimeValuesDimensions;
    values.dimensions(spaceTimeValuesDimensions);
    
    int valuesPerPoint = 1;
    for (int d=2; d<spaceTimeValuesDimensions.size(); d++) { // for vector and tensor-valued bases, take the spatial range dimension in each tensorial rank
      spaceTimeValuesDimensions[d] = _spatialBasis->rangeDimension();
      valuesPerPoint *= _spatialBasis->rangeDimension();
    }
    spaceTimeValuesDimensions[0] = _spatialBasis->getCardinality(); // field dimension
    ArrayScalar spatialValues(spaceTimeValuesDimensions);
    _spatialBasis->getValues(spatialValues, refPointsSpatial, spatialOperatorType);
    
    ArrayScalar temporalValues(_temporalBasis->getCardinality(), numPoints);
    _temporalBasis->getValues(temporalValues, refPointsTemporal, temporalOperatorType);
    
//    cout << "refPointsTemporal:\n" << refPointsTemporal;
    
//    cout << "spatialValues:\n" << spatialValues;
//    cout << "temporalValues:\n" << temporalValues;
    
    Teuchos::Array<int> spaceTimeValueCoordinate(spaceTimeValuesDimensions.size(), 0);
    Teuchos::Array<int> spatialValueCoordinate(spaceTimeValuesDimensions.size(), 0);
    
    // combine values:
    for (int spaceFieldOrdinal=0; spaceFieldOrdinal<_spatialBasis->getCardinality(); spaceFieldOrdinal++) {
      spatialValueCoordinate[0] = spaceFieldOrdinal;
      for (int timeFieldOrdinal=0; timeFieldOrdinal<_temporalBasis->getCardinality(); timeFieldOrdinal++) {
        int spaceTimeFieldOrdinal = TENSOR_FIELD_ORDINAL(spaceFieldOrdinal, timeFieldOrdinal);
        spaceTimeValueCoordinate[0] = spaceTimeFieldOrdinal;
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++) {
          spaceTimeValueCoordinate[1] = pointOrdinal;
          spatialValueCoordinate[1] = pointOrdinal;
          double temporalValue = temporalValues(timeFieldOrdinal,pointOrdinal);
          int spaceTimeValueEnumeration = values.getEnumeration(spaceTimeValueCoordinate);
          int spatialValueEnumeration = spatialValues.getEnumeration(spatialValueCoordinate);
          for (int offset=0; offset<valuesPerPoint; offset++) {
            double spatialValue = spatialValues[spatialValueEnumeration+offset];
            values[spaceTimeValueEnumeration+offset] = spatialValue * temporalValue;
          }
        }
      }
    }
  }
  
  template<class Scalar, class ArrayScalar>
  int TensorBasis<Scalar, ArrayScalar>::getDofOrdinalFromComponentDofOrdinals(std::vector<int> componentDofOrdinals) const {
    if (componentDofOrdinals.size() != 2) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "componentDofOrdinals.size() must equal 2, in present implementation");
    }
    int spaceFieldOrdinal = componentDofOrdinals[0];
    int timeFieldOrdinal = componentDofOrdinals[1];
    return TENSOR_FIELD_ORDINAL(spaceFieldOrdinal, timeFieldOrdinal);
  }

  template<class Scalar, class ArrayScalar>
  void TensorBasis<Scalar,ArrayScalar>::initializeTags() const {
    // TODO: implement this
    
    // get the component basis's tag data:
//    const std::vector<std::vector<std::vector<int> > > compTagToOrdinal = _componentBasis->getDofOrdinalData();
//    const std::vector<std::vector<int> > compOrdinalToTag = _componentBasis->getAllDofTags();
//    
//    int tagSize = 4;
//    int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
//    int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
//    int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
//    
//    std::vector<int> tags( tagSize * this->getCardinality() ); // flat array
//    
//    int componentCardinality = _componentBasis->getCardinality();
//    // ordinalToTag_:
//    for (int comp=0; comp<_numComponents; comp++) {
//      for (int compFieldIndex=0; compFieldIndex<componentCardinality; compFieldIndex++) {
//        int i=comp*componentCardinality + compFieldIndex; // i is the ordinal in the vector basis
//        vector<int> tagData = compOrdinalToTag[compFieldIndex];
//        tags[tagSize*i]   = tagData[0]; // spaceDim
//        tags[tagSize*i+1] = tagData[1]; // subcell ordinal
//        tags[tagSize*i+2] = tagData[2] + comp * tagData[3];  // ordinal of the specified DoF relative to the subcell (shifted)
//        tags[tagSize*i+3] = tagData[3] * _numComponents;     // total number of DoFs associated with the subcell
//      }
//    }
//    
//    // call basis-independent method (declared in IntrepidUtil.hpp) to set up the data structures
//    Intrepid::setOrdinalTagData(this -> _tagToOrdinal,
//                                this -> _ordinalToTag,
//                                &(tags[0]),
//                                this -> getCardinality(),
//                                tagSize,
//                                posScDim,
//                                posScOrd,
//                                posDfOrd);
  }

} // namespace Camellia