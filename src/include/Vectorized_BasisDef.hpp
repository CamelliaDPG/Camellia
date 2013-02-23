#ifndef DPGTrilinos_Vectorized_BasisDef
#define DPGTrilinos_Vectorized_BasisDef


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

using namespace std;

namespace Intrepid {
  
  template<class Scalar, class ArrayScalar>
  Vectorized_Basis<Scalar, ArrayScalar>::Vectorized_Basis(Teuchos::RCP< Basis<Scalar, ArrayScalar> > basis, int numComponents)
  {
    this -> _componentBasis = basis;
    this -> _numComponents = numComponents;
    int componentCardinality = basis->getCardinality();
    this -> basisCardinality_  = numComponents * componentCardinality;
    this -> basisDegree_       = basis->getDegree();
    this -> basisCellTopology_ = basis->getBaseCellTopology();
    this -> basisType_         = basis->getBasisType();
    this -> basisCoordinates_  = basis->getCoordinateSystem();
    this -> basisTagsAreSet_   = false;
  }
  
  template<class Scalar, class ArrayScalar>
  void Vectorized_Basis<Scalar, ArrayScalar>::initializeTags() {
    
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
    setOrdinalTagData(this -> tagToOrdinal_,
                      this -> ordinalToTag_,
                      &(tags[0]),
                      this -> basisCardinality_,
                      tagSize,
                      posScDim,
                      posScOrd,
                      posDfOrd);
  }
  
  template<class Scalar, class ArrayScalar>
  void Vectorized_Basis<Scalar, ArrayScalar>::getValues(ArrayScalar &        outputValues,
                                                        const ArrayScalar &  inputPoints,
                                                        const EOperator      operatorType) const {
    Teuchos::Array<int> dimensions;
    outputValues.dimensions(dimensions);
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
    _componentBasis->getValues(componentOutputValues,inputPoints,operatorType);
    
    int fieldIndex = 0;
    this->getVectorizedValues(outputValues,componentOutputValues,fieldIndex);
  }
  
  template<class Scalar, class ArrayScalar>
  void Vectorized_Basis<Scalar, ArrayScalar>::getVectorizedValues(ArrayScalar& outputValues,
                                                                  const ArrayScalar & componentOutputValues,
                                                                  int fieldIndex) const {
    // fieldIndex argument because sometimes outputValues has dimensions (C,F,P,#comp,...)
    // and other times (F,P,#comp,...)
    
    TEUCHOS_TEST_FOR_EXCEPTION((fieldIndex != 0) && (fieldIndex != 1), std::invalid_argument,
                               "fieldIndex must be 0 or 1");
    
    //cout << "componentOutputValues: \n" << componentOutputValues;
    TEUCHOS_TEST_FOR_EXCEPTION( outputValues.dimension(fieldIndex) != this->basisCardinality_,
                               std::invalid_argument, "outputValues.dimension(fieldIndex) != this->basisCardinality_");
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
  
  template<class Scalar, class ArrayScalar>
  void Vectorized_Basis<Scalar, ArrayScalar>::getValues(ArrayScalar&           outputValues,
                                                        const ArrayScalar &    inputPoints,
                                                        const ArrayScalar &    cellVertices,
                                                        const EOperator        operatorType) const {
    // TODO: implement this
  }
  
  template<class Scalar, class ArrayScalar>
  const Teuchos::RCP< Basis<Scalar, ArrayScalar> > Vectorized_Basis<Scalar, ArrayScalar>::getComponentBasis() const {
    return _componentBasis;
  }
  
  template<class Scalar, class ArrayScalar>
  int Vectorized_Basis<Scalar, ArrayScalar>::getDofOrdinalFromComponentDofOrdinal(int componentDofOrdinal, int componentIndex) const {
    int compCardinality = _componentBasis->getCardinality();
    return componentIndex * compCardinality + componentDofOrdinal;
  }
  
}// namespace Intrepid

#endif