//
//  VectorizedBasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/22/13.
//
//
template<class Scalar, class ArrayScalar>
VectorizedBasis<Scalar,ArrayScalar>::VectorizedBasis(BasisPtr basis, int numComponents) {
  _componentBasis = basis;
  _numComponents = numComponents;
  if (basis->functionSpace() == IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD) {
    this->_functionSpace = IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD;
  } else if (basis->functionSpace() == IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) {
    this->_functionSpace = IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL;
  } else this->_functionSpace = basis->functionSpace();
}

template<class Scalar, class ArrayScalar>
int VectorizedBasis<Scalar,ArrayScalar>::getCardinality() const {
  return _componentBasis->getCardinality() * _numComponents;
}

template<class Scalar, class ArrayScalar>
const Teuchos::RCP< Camellia::Basis<Scalar, ArrayScalar> > VectorizedBasis<Scalar, ArrayScalar>::getComponentBasis() const {
  return _componentBasis;
}

template<class Scalar, class ArrayScalar>
int VectorizedBasis<Scalar,ArrayScalar>::getDegree() const {
  return _componentBasis->getDegree();
}

// domain info on which the basis is defined:

template<class Scalar, class ArrayScalar>
shards::CellTopology VectorizedBasis<Scalar,ArrayScalar>::domainTopology() const {
  return _componentBasis->domainTopology();
}

// dof ordinal subsets:
template<class Scalar, class ArrayScalar>
std::set<int> VectorizedBasis<Scalar,ArrayScalar>::dofOrdinalsForSubcells(int subcellDim, bool includeLesserDimensions) const {
  std::set<int> dofOrdinals;
  std::set<int> componentDofOrdinals = _componentBasis->dofOrdinalsForSubcells(subcellDim, includeLesserDimensions);

  for (int compIndex = 0; compIndex < _numComponents; compIndex++) {
    for (std::set<int>::iterator compDofOrdinalIt = componentDofOrdinals.begin();
         compDofOrdinalIt != componentDofOrdinals.end(); compDofOrdinalIt++) {
      int dofOrdinal = this->getDofOrdinalFromComponentDofOrdinal(*compDofOrdinalIt,compIndex);
      dofOrdinals.insert(dofOrdinal);
    }
  }
  
  return dofOrdinals;
}

//template<class Scalar, class ArrayScalar>
//std::set<int> VectorizedBasis<Scalar,ArrayScalar>::dofOrdinalsForEdges(bool includeVertices) const {
//  std::set<int> dofOrdinals;
//  std::set<int> componentDofOrdinals = _componentBasis->dofOrdinalsForEdges(includeVertices);
//  for (int compIndex = 0; compIndex < _numComponents; compIndex++) {
//    
//  }
//  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method.");
//}
//template<class Scalar, class ArrayScalar>
//std::set<int> VectorizedBasis<Scalar,ArrayScalar>::dofOrdinalsForFaces(bool includeVerticesAndEdges) const {
//  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method.");
//}
//template<class Scalar, class ArrayScalar>
//std::set<int> VectorizedBasis<Scalar,ArrayScalar>::dofOrdinalsForInterior() const {
//  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method.");
//}
//template<class Scalar, class ArrayScalar>
//std::set<int> VectorizedBasis<Scalar,ArrayScalar>::dofOrdinalsForVertices() const {
//  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method.");
//}

template<class Scalar, class ArrayScalar>
void VectorizedBasis<Scalar, ArrayScalar>::getVectorizedValues(ArrayScalar& outputValues,
                                                               const ArrayScalar & componentOutputValues,
                                                               int fieldIndex) const {
  // fieldIndex argument because sometimes outputValues has dimensions (C,F,P,#comp,...)
  // and other times (F,P,#comp,...)
  
  TEUCHOS_TEST_FOR_EXCEPTION((fieldIndex != 0) && (fieldIndex != 1), std::invalid_argument,
                             "fieldIndex must be 0 or 1");
  
  //cout << "componentOutputValues: \n" << componentOutputValues;
  TEUCHOS_TEST_FOR_EXCEPTION( outputValues.dimension(fieldIndex) != this->getCardinality(),
                             std::invalid_argument, "outputValues.dimension(fieldIndex) != this->getCardinality()");
  TEUCHOS_TEST_FOR_EXCEPTION( componentOutputValues.dimension(fieldIndex) != _componentBasis->getCardinality(),
                             std::invalid_argument, "componentOutputValues.dimension(fieldIndex) != _componentBasis->getCardinality()");
  int pointIndex = fieldIndex+1;
  TEUCHOS_TEST_FOR_EXCEPTION( outputValues.dimension(pointIndex) != componentOutputValues.dimension(pointIndex),
                             std::invalid_argument, "outputValues.dimension(pointIndex) != componentOutputValues.dimension(pointIndex)");
  Teuchos::Array<int> dimensions;
  outputValues.dimensions(dimensions);
  outputValues.initialize(0.0);
  int numFields = dimensions[fieldIndex];
  int numPoints = dimensions[fieldIndex+1];
  int numComponents = dimensions[fieldIndex+2];
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
  }
  //    cout << "getVectorizedValues: componentOutputValues:\n" << componentOutputValues;
  //    cout << "getVectorizedValues: outputValues:\n" << outputValues;
}

// range info for basis values:
template<class Scalar, class ArrayScalar>
int VectorizedBasis<Scalar,ArrayScalar>::rangeDimension() const {
  return _componentBasis->rangeDimension();
}
template<class Scalar, class ArrayScalar>
int VectorizedBasis<Scalar,ArrayScalar>::rangeRank() const {
  return _componentBasis->rangeRank() + 1;
}
template<class Scalar, class ArrayScalar>
void VectorizedBasis<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints,
                                                    Intrepid::EOperator operatorType) const {
  this->CHECK_VALUES_ARGUMENTS(values,refPoints,operatorType);
  Teuchos::Array<int> dimensions;
  values.dimensions(dimensions);
  int numComponents = dimensions[dimensions.size() - 1];
  if (_numComponents != numComponents) {
    TEUCHOS_TEST_FOR_EXCEPTION ( _numComponents != numComponents, std::invalid_argument,
                                "final dimension of outputValues must match the number of vector components.");
  }
  // get rid of last dimension:
  dimensions.pop_back();
  // this is now the size of the array of vector-values.
  
  Teuchos::Array<int> componentDimensions = dimensions;
  int componentCardinality = _componentBasis->getCardinality();
  componentDimensions[0] = componentCardinality;
  
  ArrayScalar componentOutputValues(componentDimensions);
  _componentBasis->getValues(componentOutputValues,refPoints,operatorType);
  
  int fieldIndex = 0;
  this->getVectorizedValues(values,componentOutputValues,fieldIndex);
}

template<class Scalar, class ArrayScalar>
int VectorizedBasis<Scalar, ArrayScalar>::getDofOrdinalFromComponentDofOrdinal(int componentDofOrdinal, int componentIndex) const {
  int compCardinality = _componentBasis->getCardinality();
  return componentIndex * compCardinality + componentDofOrdinal;
}

template<class Scalar, class ArrayScalar>
void VectorizedBasis<Scalar,ArrayScalar>::initializeTags() const {
  // get the component basis's tag data:
  const std::vector<std::vector<std::vector<int> > > compTagToOrdinal = _componentBasis->getDofOrdinalData();
  const std::vector<std::vector<int> > compOrdinalToTag = _componentBasis->getAllDofTags();
  
  int tagSize = 4;
  int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
  int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
  int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell
  
  std::vector<int> tags( tagSize * this->getCardinality() ); // flat array
  
  int componentCardinality = _componentBasis->getCardinality();
  // ordinalToTag_:
  for (int comp=0; comp<_numComponents; comp++) {
    for (int compFieldIndex=0; compFieldIndex<componentCardinality; compFieldIndex++) {
      int i=comp*componentCardinality + compFieldIndex; // i is the ordinal in the vector basis
      vector<int> tagData = compOrdinalToTag[compFieldIndex];
      tags[tagSize*i]   = tagData[0]; // spaceDim
      tags[tagSize*i+1] = tagData[1]; // subcell ordinal
      tags[tagSize*i+2] = tagData[2] + comp * tagData[3];  // ordinal of the specified DoF relative to the subcell (shifted)
      tags[tagSize*i+3] = tagData[3] * _numComponents;     // total number of DoFs associated with the subcell
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
