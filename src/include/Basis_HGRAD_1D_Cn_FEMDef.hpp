//
//  Basis_HGRAD_C0_FEM.h
//  DPGTrilinos
//


#ifndef DPGTrilinos_Basis_HGRAD_1D_Cn_FEMDef
#define DPGTrilinos_Basis_HGRAD_1D_Cn_FEMDef


/** \file   Intrepid_HGRAD_QUAD_C0_FEMDef.hpp
 \brief  Definition file for vector FEM basis functions for H(grad) functions on LINE cells.
 \author Created by N. Roberts.
 */

#include "BasisFactory.h"

namespace Intrepid {
  
  const static int NUM_COMPONENTS = 2;
  
  template<class Scalar, class ArrayScalar>
  Basis_HGRAD_1D_Cn_FEM<Scalar, ArrayScalar>::Basis_HGRAD_1D_Cn_FEM(int order, int cellTopoKey)
  : Vectorized_Basis<Scalar, ArrayScalar>( BasisFactory::getBasis(order, cellTopoKey,
                                                                  IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD), NUM_COMPONENTS)
  {
    _spaceDim = 1;
  }
  
  template<class Scalar, class ArrayScalar>
  void Basis_HGRAD_1D_Cn_FEM<Scalar, ArrayScalar>::getValues(ArrayScalar &        outputValues,
                                                             const ArrayScalar &  inputPoints,
                                                             const EOperator      operatorType) const {
    
    int numFields = outputValues.dimension(0);
    int numPoints = outputValues.dimension(1);
    TEUCHOS_TEST_FOR_EXCEPTION( numPoints != inputPoints.dimension(0), std::invalid_argument, "outputValues.dimension(1) != inputPoints.dimension(0)");
    switch (operatorType) {
      case OPERATOR_DIV:
      case OPERATOR_CURL:
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,
                                   "Error: should use BasisEvaluation for DIV, CURL of VECTOR_HGRAD.");
        // the above exception because we haven't figured out the right way to scale these, and in 
        // practice for traces and fluxes, we shouldn't be taking derivatives anyway...  (See the 2D version of this
        // for a start on coding this, if it's desired--right now, we do throw an exception there as well, though.)
      }
        break;
      default:
        this->Vectorized_Basis<Scalar, ArrayScalar>::getValues(outputValues,inputPoints,operatorType);
        break;    
    }
  }
  
  template<class Scalar, class ArrayScalar>
  void Basis_HGRAD_1D_Cn_FEM<Scalar, ArrayScalar>::getValues(ArrayScalar&           outputValues,
                                                             const ArrayScalar &    inputPoints,
                                                             const ArrayScalar &    cellVertices,
                                                             const EOperator        operatorType) const {
    TEUCHOS_TEST_FOR_EXCEPTION( (true), std::logic_error,
                               ">>> ERROR (Basis_HGRAD_1D_Cn_FEM): FEM Basis calling an FVD member function");
  }
  
  
}// namespace Intrepid

#endif