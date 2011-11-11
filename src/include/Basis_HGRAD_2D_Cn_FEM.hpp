#ifndef DPGTrilinos_Basis_HGRAD_2D_Cn_FEM
#define DPGTrilinos_Basis_HGRAD_2D_Cn_FEM

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


/** \file   
 \brief  Header file for the Intrepid::G_QUAD_C0_FEM class.
 \author Created by N. Roberts
 */

#include "Intrepid_Basis.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

using namespace Intrepid;
namespace Intrepid {
  
  /** \class  Intrepid::Basis_HGRAD_QUAD_C0_FEM
   \brief  Implementation of the default H(grad)-compatible FEM basis of degree 0 on Quadrilateral cell
   
   Implements Lagrangian basis of degree 1 on the reference Quadrilateral cell. The basis has
   cardinality 1 and spans a COMPLETE bi-linear polynomial space. Basis functions are dual 
   to a unisolvent set of degrees-of-freedom (DoF) defined and enumerated as follows:
   
   \verbatim
   =================================================================================================
   |         |           degree-of-freedom-tag table                    |                           |
   |   DoF   |----------------------------------------------------------|      DoF definition       |
   | ordinal |  subc dim    | subc ordinal | subc DoF ord |subc num DoF |                           |
   |=========|==============|==============|==============|=============|===========================|
   |=========|==============|==============|==============|=============|===========================|
   |   MAX   |  maxScDim=0  |  maxScOrd=3  |  maxDfOrd=0  |     -       |                           |
   |=========|==============|==============|==============|=============|===========================|
   \endverbatim
   */
  template<class Scalar, class ArrayScalar> 
  class Basis_HGRAD_2D_Cn_FEM : public Vectorized_Basis<Scalar, ArrayScalar> {
  private:
    int _spaceDim;
    
  public:
    
    /** \brief Constructor.
     */
    Basis_HGRAD_2D_Cn_FEM(int order, int cellTopoKey);
    
    
    /** \brief  FEM basis evaluation on a <strong>reference Quadrilateral or Triangle</strong> cell. 
     
     Returns values of <var>operatorType</var> acting on FEM basis functions for a set of
     points in the <strong>reference Quadrilateral</strong> cell. For rank and dimensions of 
     I/O array arguments see Section \ref basis_md_array_sec .
     
     \param  outputValues      [out] - rank-2 or 3 array with the computed basis values
     \param  inputPoints       [in]  - rank-2 array with dimensions (P,D) containing reference points  
     \param  operatorType      [in]  - operator applied to basis functions        
     */
    void getValues(ArrayScalar &          outputValues,
                   const ArrayScalar &    inputPoints,
                   const EOperator        operatorType) const;
    
    
    /**  \brief  FVD basis evaluation: invocation of this method throws an exception.
     */
    void getValues(ArrayScalar &          outputValues,
                   const ArrayScalar &    inputPoints,
                   const ArrayScalar &    cellVertices,
                   const EOperator        operatorType = OPERATOR_VALUE) const;
  };
}// namespace Intrepid

#include "Basis_HGRAD_2D_Cn_FEMDef.hpp"

#endif