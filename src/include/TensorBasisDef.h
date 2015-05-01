//
//  TensorBasisDef.h
//  Camellia
//
//  Created by Nathan Roberts on 11/12/14.
//
//

#include "CamelliaCellTools.h"

#define TENSOR_POINT_ORDINAL(spacePointOrdinal,timePointOrdinal,numSpacePoints) timePointOrdinal * numSpacePoints + spacePointOrdinal
#define TENSOR_FIELD_ORDINAL(spaceFieldOrdinal,timeFieldOrdinal) timeFieldOrdinal * _spatialBasis->getCardinality() + spaceFieldOrdinal
#define TENSOR_DOF_OFFSET_ORDINAL(spaceDofOffsetOrdinal,timeDofOffsetOrdinal,spaceDofsForSubcell) timeDofOffsetOrdinal * spaceDofsForSubcell + spaceDofOffsetOrdinal

namespace Camellia {
  template<class Scalar, class ArrayScalar>
  TensorBasis<Scalar,ArrayScalar>::TensorBasis(Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > spatialBasis,
                                               Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > temporalBasis,
                                               bool rangeDimensionIsSum) {
    _spatialBasis = spatialBasis;
    _temporalBasis = temporalBasis;
    _rangeDimensionIsSum = rangeDimensionIsSum;
    
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
    
    if (_spatialBasis->domainTopology()->getDimension() == 0) {
      this->_functionSpace = _temporalBasis->functionSpace();
    } else if (_spatialBasis->functionSpace() == _temporalBasis->functionSpace()) {
      // I doubt this would be right if the function spaces were HDIV or HCURL, but
      // for HVOL AND HGRAD, it does hold, and since our temporal bases are always defined
      // on line topologies, HVOL and HGRAD are the only ones for which we'll use this...
      this->_functionSpace = _spatialBasis->functionSpace();
    } else {
      this->_functionSpace = FUNCTION_SPACE_UNKNOWN;
    }
    
    int tensorialDegree = 1;
    this->_domainTopology = CellTopology::cellTopology(_spatialBasis->domainTopology()->getShardsTopology(), tensorialDegree);
    
    this->_basisCardinality = _spatialBasis->getCardinality() * _temporalBasis->getCardinality();
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
  const Teuchos::RCP< Camellia::Basis<Scalar, ArrayScalar> > TensorBasis<Scalar, ArrayScalar>::getSpatialBasis() const {
    return this->getComponentBasis(0);
  }
  
  template<class Scalar, class ArrayScalar>
  const Teuchos::RCP< Camellia::Basis<Scalar, ArrayScalar> > TensorBasis<Scalar, ArrayScalar>::getTemporalBasis() const {
    return this->getComponentBasis(1);
  }

  template<class Scalar, class ArrayScalar>
  void TensorBasis<Scalar, ArrayScalar>::getTensorPoints(ArrayScalar& tensorPoints, const ArrayScalar & spatialPoints,
                                                         const ArrayScalar & temporalPoints) const {
    bool hasCellRank;
    if ((spatialPoints.rank() != tensorPoints.rank()) || (temporalPoints.rank() != tensorPoints.rank())) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "temporalPoints and spatialPoints must have same rank as tensorPoints");
    }
    
    if (tensorPoints.rank() == 3) {
      hasCellRank = true;

    } else if (tensorPoints.rank() == 2) {
      hasCellRank = false;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"unsupported rank in tensorPoints");
    }
    
    int numCells = hasCellRank ? tensorPoints.dimension(0) : 1;
    
    int pointIndex = hasCellRank ? 1 : 0;
    int spaceDimIndex = hasCellRank ? 2 : 1;
    int numPointsSpace = spatialPoints.dimension(pointIndex);
    int numPointsTime = temporalPoints.dimension(pointIndex);
    
    int spaceDim = spatialPoints.dimension(spaceDimIndex);
    int timeDim = temporalPoints.dimension(spaceDimIndex);
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++) {
      for (int timePointOrdinal=0; timePointOrdinal<numPointsTime; timePointOrdinal++) {
        for (int spacePointOrdinal=0; spacePointOrdinal<numPointsSpace; spacePointOrdinal++) {
          int spaceTimePointOrdinal = TENSOR_POINT_ORDINAL(spacePointOrdinal, timePointOrdinal, numPointsSpace);
          for (int d=0; d<spaceDim; d++) {
            if (hasCellRank) {
              tensorPoints(cellOrdinal,spaceTimePointOrdinal,d) = spatialPoints(cellOrdinal,spacePointOrdinal,d);
            } else {
              tensorPoints(spaceTimePointOrdinal,d) = spatialPoints(spacePointOrdinal,d);
            }
          }
          for (int d=spaceDim; d<spaceDim+timeDim; d++) {
            if (hasCellRank) {
              tensorPoints(cellOrdinal,spaceTimePointOrdinal,d) = temporalPoints(cellOrdinal,timePointOrdinal,d-spaceDim);
            } else {
              tensorPoints(spaceTimePointOrdinal,d) = temporalPoints(timePointOrdinal,d-spaceDim);
            }
          }
        }
      }
    }
  }

  
  template<class Scalar, class ArrayScalar>
  void TensorBasis<Scalar,ArrayScalar>::getTensorValues(ArrayScalar& outputValues, std::vector< ArrayScalar> & componentOutputValuesVector,
                                                        std::vector<Intrepid::EOperator> operatorTypes) const {
    // outputValues can have dimensions (C,F,P,...) or (F,P,...)
    
    if (operatorTypes.size() != 2) {
      TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument, "only two-component tensor bases supported right now");
    }
    
    Intrepid::EOperator spaceOp = operatorTypes[0], timeOp = operatorTypes[1];
    
    if (timeOp == Intrepid::OPERATOR_GRAD) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "getTensorBasisValues() doesn't support taking gradient of temporal basis.");
    }
    
    std::map<Intrepid::EOperator, int> rankAdjustmentForOperator;
    
    rankAdjustmentForOperator[Intrepid::OPERATOR_VALUE] = 0;
    rankAdjustmentForOperator[Intrepid::OPERATOR_GRAD] = 1;
    rankAdjustmentForOperator[Intrepid::OPERATOR_DIV] = -1;
    rankAdjustmentForOperator[Intrepid::OPERATOR_CURL] = 0; // in 2D, this toggles between +1 and -1, depending on the current rank (scalar --> vector, vector --> scalar)
    
    int valuesRank = _spatialBasis->rangeRank() + rankAdjustmentForOperator[spaceOp];
    if ((_spatialBasis->rangeDimension() == 2) && (spaceOp==Intrepid::OPERATOR_CURL)) {
      if (_spatialBasis->rangeRank() == 0) valuesRank += 1;
      if (_spatialBasis->rangeRank() == 1) valuesRank -= 1;
    }

    const ArrayScalar* spatialValues = &componentOutputValuesVector[0];
    const ArrayScalar* temporalValues = &componentOutputValuesVector[1];
    
    int fieldIndex;
    if (outputValues.rank() == valuesRank + 3) {
      fieldIndex = 1; // (C,F,...)
    } else if (outputValues.rank() == valuesRank + 2) {
      fieldIndex = 0; // (F,...)
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "outputValues shape must be either (C,F,P,...) or (F,P,...)");
    }
    int valuesPerPointSpace = 1;
    for (int rankOrdinal=0; rankOrdinal < valuesRank; rankOrdinal++) {
      valuesPerPointSpace *= spatialValues->dimension(rankOrdinal + fieldIndex + 2); // start with the dimension after the point dimension
    }

    //cout << "componentOutputValues: \n" << componentOutputValues;
    TEUCHOS_TEST_FOR_EXCEPTION( outputValues.dimension(fieldIndex) != this->getCardinality(),
                               std::invalid_argument, "outputValues.dimension(fieldIndex) != this->getCardinality()");
    if (spatialValues->dimension(fieldIndex) != _spatialBasis->getCardinality()) {
      TEUCHOS_TEST_FOR_EXCEPTION( spatialValues->dimension(fieldIndex) != _spatialBasis->getCardinality(),
                                 std::invalid_argument, "spatialValues->dimension(fieldIndex) != _spatialBasis->getCardinality()");
    }
    if (temporalValues->dimension(fieldIndex) != _temporalBasis->getCardinality()) {
      TEUCHOS_TEST_FOR_EXCEPTION( temporalValues->dimension(fieldIndex) != _temporalBasis->getCardinality(),
                                 std::invalid_argument, "temporalValues->dimension(fieldIndex) != _temporalBasis->getCardinality()");
    }
    int pointIndex = fieldIndex+1;
    TEUCHOS_TEST_FOR_EXCEPTION( outputValues.dimension(pointIndex) != spatialValues->dimension(pointIndex) * temporalValues->dimension(pointIndex),
                               std::invalid_argument, "outputValues.dimension(pointIndex) != spatialValues->dimension(pointIndex) * temporalValues->dimension(pointIndex)");
    Teuchos::Array<int> dimensions;
    outputValues.dimensions(dimensions);
    outputValues.initialize(0.0);
    int numPointsSpace = spatialValues->dimension(fieldIndex + 1);
    int numPointsTime = temporalValues->dimension(fieldIndex + 1);
    
    Teuchos::Array<int> spaceTimeValueCoordinate(outputValues.rank(), 0);
    Teuchos::Array<int> spatialValueCoordinate(outputValues.rank(), 0);
    
    int numCells = (fieldIndex==0) ? 1 : outputValues.dimension(0);
    
    for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++) {
      if (fieldIndex==1) { // have a cell index
        spaceTimeValueCoordinate[0] = cellOrdinal;
        spatialValueCoordinate[0] = cellOrdinal;
      }
      // combine values:
      for (int spaceFieldOrdinal=0; spaceFieldOrdinal<_spatialBasis->getCardinality(); spaceFieldOrdinal++) {
        spatialValueCoordinate[fieldIndex] = spaceFieldOrdinal;
        for (int timeFieldOrdinal=0; timeFieldOrdinal<_temporalBasis->getCardinality(); timeFieldOrdinal++) {
          int spaceTimeFieldOrdinal = TENSOR_FIELD_ORDINAL(spaceFieldOrdinal, timeFieldOrdinal);
          spaceTimeValueCoordinate[fieldIndex] = spaceTimeFieldOrdinal;
          for (int timePointOrdinal=0; timePointOrdinal<numPointsTime; timePointOrdinal++) {
            double temporalValue = (fieldIndex==0) ? (*temporalValues)(timeFieldOrdinal,timePointOrdinal) : (*temporalValues)(cellOrdinal,timeFieldOrdinal,timePointOrdinal);
            for (int spacePointOrdinal=0; spacePointOrdinal<numPointsSpace; spacePointOrdinal++) {
              int spaceTimePointOrdinal = TENSOR_POINT_ORDINAL(spacePointOrdinal, timePointOrdinal, numPointsSpace);
              spatialValueCoordinate[fieldIndex+1] = spacePointOrdinal;
              spaceTimeValueCoordinate[fieldIndex+1] = spaceTimePointOrdinal;
              
              int spatialValueEnumeration = spatialValues->getEnumeration(spatialValueCoordinate);
              int spaceTimeValueEnumeration = outputValues.getEnumeration(spaceTimeValueCoordinate);
              
              for (int offset=0; offset<valuesPerPointSpace; offset++) {
                double spatialValue = (*spatialValues)[spatialValueEnumeration+offset];
                outputValues[spaceTimeValueEnumeration+offset] = spatialValue * temporalValue;
              }
            }
          }
        }
      }
    }
  }

  // range info for basis values:
  template<class Scalar, class ArrayScalar>
  int TensorBasis<Scalar,ArrayScalar>::rangeDimension() const {
    // NVR changed this 4-14-15 to depend on the (new) _rangeDimensionIsSum member.
    // Default is false; in general, when doing space-time, vector lengths are those of the spatial basis.
    if (!_rangeDimensionIsSum) {
      return _spatialBasis->rangeDimension();
    } else {
      return _spatialBasis->rangeDimension() + _temporalBasis->rangeDimension();
    }
  }
  
  template<class Scalar, class ArrayScalar>
  int TensorBasis<Scalar,ArrayScalar>::rangeRank() const {
    return _spatialBasis->rangeRank();
  }
  
  template<class Scalar, class ArrayScalar>
  void TensorBasis<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints,
                                                  Intrepid::EOperator operatorType) const {
    Intrepid::EOperator temporalOperator = Intrepid::OPERATOR_VALUE; // default
    if (operatorType==Intrepid::OPERATOR_GRAD) {
      if (values.rank() == 3) { // F, P, D
        if (_spatialBasis->rangeDimension() + _temporalBasis->rangeDimension() == values.dimension(2)) {
          // then we can/should use OPERATOR_GRAD for the temporal operator as well
          temporalOperator = Intrepid::OPERATOR_GRAD;
        }
      }
    }
    getValues(values, refPoints, operatorType, temporalOperator);
  }
  
  template<class Scalar, class ArrayScalar>
  void TensorBasis<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints,
                                                  Intrepid::EOperator spatialOperatorType, Intrepid::EOperator temporalOperatorType) const {
    bool gradInBoth = (spatialOperatorType==Intrepid::OPERATOR_GRAD) && (temporalOperatorType==Intrepid::OPERATOR_GRAD);
    this->CHECK_VALUES_ARGUMENTS(values,refPoints,spatialOperatorType);
    
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
    
    Teuchos::Array<int> valuesDim; // will use to size our space and time values arrays
    values.dimensions(valuesDim); // F, P[,D,...]
    
    int valuesPerPointSpace = 1;
    for (int d=2; d<valuesDim.size(); d++) { // for vector and tensor-valued bases, take the spatial range dimension in each tensorial rank
      valuesDim[d] = max(_spatialBasis->rangeDimension(), 1); // ensure that for 0-dimensional topologies, we still define a value
      valuesPerPointSpace *= valuesDim[d];
    }
    valuesDim[0] = _spatialBasis->getCardinality(); // field dimension
    ArrayScalar spatialValues(valuesDim);
    _spatialBasis->getValues(spatialValues, refPointsSpatial, spatialOperatorType);
    
    ArrayScalar temporalValues(_temporalBasis->getCardinality(), numPoints);
    if (temporalOperatorType==Intrepid::OPERATOR_GRAD) {
      temporalValues.resize(_temporalBasis->getCardinality(), numPoints, _temporalBasis->rangeDimension());
    }
    _temporalBasis->getValues(temporalValues, refPointsTemporal, temporalOperatorType);
    
    ArrayScalar spatialValues_opValue;
    ArrayScalar temporalValues_opValue;
    if (gradInBoth) {
      spatialValues_opValue.resize(_spatialBasis->getCardinality(), numPoints);
      temporalValues_opValue.resize(_temporalBasis->getCardinality(), numPoints);
      _spatialBasis->getValues(spatialValues_opValue, refPointsSpatial, Intrepid::OPERATOR_VALUE);
      _temporalBasis->getValues(temporalValues_opValue, refPointsTemporal, Intrepid::OPERATOR_VALUE);
    }
    
//    cout << "refPointsTemporal:\n" << refPointsTemporal;
    
//    cout << "spatialValues:\n" << spatialValues;
//    cout << "temporalValues:\n" << temporalValues;
    
    Teuchos::Array<int> spaceTimeValueCoordinate(valuesDim.size(), 0);
    Teuchos::Array<int> spatialValueCoordinate(valuesDim.size(), 0);
    
    // combine values:
    for (int spaceFieldOrdinal=0; spaceFieldOrdinal<_spatialBasis->getCardinality(); spaceFieldOrdinal++) {
      spatialValueCoordinate[0] = spaceFieldOrdinal;
      for (int timeFieldOrdinal=0; timeFieldOrdinal<_temporalBasis->getCardinality(); timeFieldOrdinal++) {
        int spaceTimeFieldOrdinal = TENSOR_FIELD_ORDINAL(spaceFieldOrdinal, timeFieldOrdinal);
        spaceTimeValueCoordinate[0] = spaceTimeFieldOrdinal;
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++) {
          spaceTimeValueCoordinate[1] = pointOrdinal;
          spatialValueCoordinate[1] = pointOrdinal;
          double temporalValue;
          if (temporalOperatorType!=Intrepid::OPERATOR_GRAD)
            temporalValue = temporalValues(timeFieldOrdinal,pointOrdinal);
          else
            temporalValue = temporalValues(timeFieldOrdinal,pointOrdinal, 0);
          int spatialValueEnumeration = spatialValues.getEnumeration(spatialValueCoordinate);
          
          if (! gradInBoth) {
            int spaceTimeValueEnumeration = values.getEnumeration(spaceTimeValueCoordinate);
            for (int offset=0; offset<valuesPerPointSpace; offset++) {
              double spatialValue = spatialValues[spatialValueEnumeration+offset];
              values[spaceTimeValueEnumeration+offset] = spatialValue * temporalValue;
            }
          } else {
            double spatialValue_opValue = spatialValues_opValue(spaceFieldOrdinal,pointOrdinal);
            double temporalValue_opValue = temporalValues_opValue(timeFieldOrdinal,pointOrdinal);
            
            // product rule: first components are spatial gradient times temporal value; next components are spatial value times temporal gradient
            // first, handle spatial gradients
            spaceTimeValueCoordinate[2] = 0;
            int spaceTimeValueEnumeration = values.getEnumeration(spaceTimeValueCoordinate);
            for (int offset=0; offset<valuesPerPointSpace; offset++) {
              double spatialGradValue = spatialValues[spatialValueEnumeration+offset];
              double spaceTimeValue = spatialGradValue * temporalValue_opValue;

              values[spaceTimeValueEnumeration+offset] = spaceTimeValue;
            }

            // next, temporal gradients
            spaceTimeValueCoordinate[2] = _spatialBasis->rangeDimension();
            spaceTimeValueEnumeration = values.getEnumeration(spaceTimeValueCoordinate);
            
            double temporalGradValue = temporalValues(timeFieldOrdinal,pointOrdinal,0);
            double spaceTimeValue = spatialValue_opValue * temporalGradValue;

            values[spaceTimeValueEnumeration] = spaceTimeValue;
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
    const std::vector<std::vector<std::vector<int> > > spatialTagToOrdinal = _spatialBasis->getDofOrdinalData();
    const std::vector<std::vector<std::vector<int> > > temporalTagToOrdinal = _temporalBasis->getDofOrdinalData();
    
    const std::vector<std::vector<int> > spatialOrdinalToTag = _spatialBasis->getAllDofTags();
    const std::vector<std::vector<int> > temporalOrdinalToTag = _temporalBasis->getAllDofTags();
    
    int tagSize = 4;
    int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
    int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
    int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
    int posDfCnt = 3;        // position in the tag, counting from 0, of DoF ordinal count for subcell
    
    std::vector<int> tags( tagSize * this->getCardinality() ); // flat array
    
    CellTopoPtr spaceTimeTopo = this->domainTopology();
    
    int sideDim = spaceTimeTopo->getDimension() - 1;
    
    for (int spaceFieldOrdinal=0; spaceFieldOrdinal<_spatialBasis->getCardinality(); spaceFieldOrdinal++) {
      std::vector<int> spaceTagData = spatialOrdinalToTag[spaceFieldOrdinal];
      unsigned spaceSubcellDim = spaceTagData[posScDim];
      unsigned spaceSubcellOrd = spaceTagData[posScOrd];
      for (int timeFieldOrdinal=0; timeFieldOrdinal<_temporalBasis->getCardinality(); timeFieldOrdinal++) {
        std::vector<int> timeTagData = temporalOrdinalToTag[timeFieldOrdinal];
        unsigned timeSubcellDim = timeTagData[posScDim];
        unsigned timeSubcellOrd = timeTagData[posScOrd];
        
        unsigned spaceTimeSubcellDim = spaceSubcellDim + timeSubcellDim;
        unsigned spaceTimeSubcellOrd;
        if (timeSubcellDim == 0) {
          // vertex node in time; the subcell is not extruded in time but belongs to one of the two "copies"
          // of the spatial topology
          unsigned spaceTimeSideOrdinal = this->domainTopology()->getTemporalSideOrdinal(timeSubcellOrd); // timeSubcellOrd is a "side" of the line topology
          spaceTimeSubcellOrd = CamelliaCellTools::subcellOrdinalMap(spaceTimeTopo, sideDim, spaceTimeSideOrdinal,
                                                                     spaceSubcellDim, spaceSubcellOrd);
        } else {
          // line subcell in time; the subcell *is* extruded in time
          spaceTimeSubcellOrd = spaceTimeTopo->getExtrudedSubcellOrdinal(spaceSubcellDim, spaceSubcellOrd);
          if (spaceTimeSubcellOrd == (unsigned)-1) {
            cout << "ERROR: -1 subcell ordinal.\n";
            spaceTimeSubcellOrd = spaceTimeTopo->getExtrudedSubcellOrdinal(spaceSubcellDim, spaceSubcellOrd);
          }
        }
        
        int i = TENSOR_FIELD_ORDINAL(spaceFieldOrdinal, timeFieldOrdinal);
//        cout << "(" << spaceFieldOrdinal << "," << timeFieldOrdinal << ") --> " << i << endl;
        int spaceDofOffsetOrdinal = spaceTagData[posDfOrd];
        int timeDofOffsetOrdinal = timeTagData[posDfOrd];
        int spaceDofsForSubcell = spaceTagData[posDfCnt];
        int spaceTimeDofOffsetOrdinal = TENSOR_DOF_OFFSET_ORDINAL(spaceDofOffsetOrdinal, timeDofOffsetOrdinal, spaceDofsForSubcell);
        tags[tagSize*i + posScDim] = spaceTimeSubcellDim; // subcellDim
        tags[tagSize*i + posScOrd] = spaceTimeSubcellOrd; // subcell ordinal
        tags[tagSize*i + posDfOrd] = spaceTimeDofOffsetOrdinal;  // ordinal of the specified DoF relative to the subcell
        tags[tagSize*i + posDfCnt] = spaceTagData[posDfCnt] * timeTagData[posDfCnt];     // total number of DoFs associated with the subcell
      }
    }
    
    // call basis-independent method (declared in IntrepidUtil.hpp) to set up the data structures
    Intrepid::setOrdinalTagData(this -> _tagToOrdinal,
                                this -> _ordinalToTag,
                                &(tags[0]),
                                this -> getCardinality(),
                                tagSize,
                                posScDim,
                                posScOrd,
                                posDfOrd);
  }

} // namespace Camellia