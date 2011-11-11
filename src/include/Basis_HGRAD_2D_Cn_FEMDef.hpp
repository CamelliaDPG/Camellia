//
//  Basis_HGRAD_C0_FEM.h
//  DPGTrilinos
//


#ifndef DPGTrilinos_Basis_HGRAD_2D_Cn_FEMDef
#define DPGTrilinos_Basis_HGRAD_2D_Cn_FEMDef

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

/** \file   Intrepid_HGRAD_QUAD_C0_FEMDef.hpp
 \brief  Definition file for bi-linear FEM basis functions for H(grad) functions on QUAD cells.
 \author Created by P. Bochev and D. Ridzal.
 */

#include "BasisFactory.h"

namespace Intrepid {
  
  template<class Scalar, class ArrayScalar>
  Basis_HGRAD_2D_Cn_FEM<Scalar, ArrayScalar>::Basis_HGRAD_2D_Cn_FEM(int order, int cellTopoKey)
  : Vectorized_Basis<Scalar, ArrayScalar>( BasisFactory::getBasis(order, cellTopoKey,
                                                                  IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD), 2)
  {
    _spaceDim = 2;
  }
  
  template<class Scalar, class ArrayScalar>
  void Basis_HGRAD_2D_Cn_FEM<Scalar, ArrayScalar>::getValues(ArrayScalar &        outputValues,
                                                             const ArrayScalar &  inputPoints,
                                                             const EOperator      operatorType) const {
    
    int numFields = outputValues.dimension(0);
    int numPoints = outputValues.dimension(1);
    TEST_FOR_EXCEPTION( numPoints != inputPoints.dimension(0), std::invalid_argument, "outputValues.dimension(1) != inputPoints.dimension(0)");
    switch (operatorType) {
      case OPERATOR_DIV:
      case OPERATOR_CURL:
      {
        TEST_FOR_EXCEPTION(true,std::invalid_argument,
                           "Error: should use BasisEvaluation for DIV, CURL of VECTOR_HGRAD.");
        // the above exception because we haven't figured out the right way to scale
        // the results coming from below.  (We need to do the HGRAD_transform_GRAD, and compute the DIV or CURL from this...)
        // grad values will have dimensions (F,P,D,D)
        Teuchos::Array<int> gradValueDims(4);
        gradValueDims[0] = numFields;
        gradValueDims[1] = numPoints;
        gradValueDims[2] = _spaceDim;
        gradValueDims[3] = _spaceDim;
        ArrayScalar gradValues(gradValueDims);
        this->Vectorized_Basis<Scalar, ArrayScalar>::getValues(gradValues,inputPoints,OPERATOR_GRAD);

        // outputValues will have dimensions (F,P)
        for (int field=0; field<numFields; field++) {
          for (int point=0; point<numPoints; point++) {
            if (operatorType==OPERATOR_DIV) {
              outputValues(field,point) = gradValues(field,point,0,0) + gradValues(field,point,1,1); 
//              cout << "for OPERATOR_DIV, outputValues(field=" << field << "," << "point=" << point << "): ";
//              cout << outputValues(field,point) << endl;
            } else if (operatorType==OPERATOR_CURL) {
              // curl of 2D vector: d(Fx)/dy - d(Fy)/dx
              outputValues(field,point) = gradValues(field,point,1,0) - gradValues(field,point,0,1); 
              
//              cout << "for OPERATOR_CURL, outputValues(field=" << field << "," << "point=" << point << "): ";
//              cout << outputValues(field,point) << endl;
            }
          }
        }
      }
        break;
      default:
        this->Vectorized_Basis<Scalar, ArrayScalar>::getValues(outputValues,inputPoints,operatorType);
        break;    
    }
    //cout << "Basis_HGRAD_2D_Cn_FEM: getValues: inputPoints:\n" << inputPoints;
    //cout << "Basis_HGRAD_2D_Cn_FEM: getValues: outputValues:\n" << outputValues;
  }
  
  template<class Scalar, class ArrayScalar>
  void Basis_HGRAD_2D_Cn_FEM<Scalar, ArrayScalar>::getValues(ArrayScalar&           outputValues,
                                                               const ArrayScalar &    inputPoints,
                                                               const ArrayScalar &    cellVertices,
                                                               const EOperator        operatorType) const {
    TEST_FOR_EXCEPTION( (true), std::logic_error,
                       ">>> ERROR (Basis_HGRAD_2D_Cn_FEM): FEM Basis calling an FVD member function");
  }

  
}// namespace Intrepid

#endif