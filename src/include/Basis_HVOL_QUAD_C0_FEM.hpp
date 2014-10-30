//
//  Basis_HGRAD_C0_FEM.h
//  DPGTrilinos
//
//  Created by Nate Roberts on 8/9/11.
//

#ifndef DPGTrilinos_Basis_HGRAD_C0_FEM_h
#define DPGTrilinos_Basis_HGRAD_C0_FEM_h

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

/** \file   Intrepid_G_QUAD_C0_FEM.hpp
 \brief  Header file for the Intrepid::G_QUAD_C0_FEM class.
 \author Created by P. Bochev and D. Ridzal.
 */

#include "Intrepid_Basis.hpp"

using namespace Intrepid;
namespace Intrepid {
  
  /** \class  Intrepid::Basis_HVOL_QUAD_C0_FEM
   \brief  Implementation of the default H(VOL)-compatible FEM basis of degree 0 on Quadrilateral cell
   
   Implements constant (degree 0) basis on the reference Quadrilateral cell. The basis has
   cardinality 1 and spans a COMPLETE bilinear polynomial space. Basis functions are dual
   to a unisolvent set of degrees-of-freedom (DoF) defined and enumerated as follows:
   
   \verbatim
   =================================================================================================
   |         |           degree-of-freedom-tag table                    |                           |
   |   DoF   |----------------------------------------------------------|      DoF definition       |
   | ordinal |  subc dim    | subc ordinal | subc DoF ord |subc num DoF |                           |
   |=========|==============|==============|==============|=============|===========================|
   |    0    |       0      |       0      |       0      |      1      |   L_0(u) = u(-1,-1)       |
   |=========|==============|==============|==============|=============|===========================|
   |   MAX   |  maxScDim=0  |  maxScOrd=3  |  maxDfOrd=0  |     -       |                           |
   |=========|==============|==============|==============|=============|===========================|
   \endverbatim
   */
  template<class Scalar, class ArrayScalar> 
  class Basis_HVOL_QUAD_C0_FEM : public Basis<Scalar, ArrayScalar>, public DofCoordsInterface<ArrayScalar> {
  private:
    
    /** \brief Initializes <var>tagToOrdinal_</var> and <var>ordinalToTag_</var> lookup arrays.
     */
    void initializeTags();
    
  public:
    
    /** \brief Constructor.
     */
    Basis_HVOL_QUAD_C0_FEM();
    
    
    /** \brief  FEM basis evaluation on a <strong>reference Quadrilateral</strong> cell. 
     
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
    
    /** \brief  Returns spatial locations (coordinates) of degrees of freedom on a
     <strong>reference Quadrilateral</strong>.
     
     \param  DofCoords      [out] - array with the coordinates of degrees of freedom,
     dimensioned (F,D)
     */
    void getDofCoords(ArrayScalar & DofCoords) const;
    
  };
}// namespace Intrepid

#include "Basis_HVOL_QUAD_C0_FEMDef.hpp"

#endif