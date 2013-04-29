
//
//  DPGTrilinos
//
//  Created by Nate Roberts on 8/9/11.
//

#ifndef DPGTrilinos_Vectorized_Basis
#define DPGTrilinos_Vectorized_Basis


// @HEADER
//
// Original version Copyright © 2011 Sandia Corporation. All Rights Reserved.
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

/** \file   Vectorized_Basis.hpp
 \brief  Header file for the Vectorized_Basis class.
 \author Created by N. Roberts
 */

#include "Intrepid_Basis.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

using namespace Intrepid;
namespace Intrepid {
  
  /** \class  Intrepid::Vectorized_Basis
   \brief  Makes a vector basis out of any Intrepid Basis.  Operators simply apply to the individual components of the vector.
   The vector basis consists of functions (e_i, e_j) where e_i and e_j are functions in the original Intrepid basis.  Basis functions are ordered lexicographically: ij=00, ij=01, etc.
   */
  template<class Scalar, class ArrayScalar> 
  class Vectorized_Basis : public Basis<Scalar, ArrayScalar> {
  private:
    Teuchos::RCP< Basis<Scalar, ArrayScalar> > _componentBasis;
    int _numComponents;
    
    /** \brief Initializes <var>tagToOrdinal_</var> and <var>ordinalToTag_</var> lookup arrays.
     */
    void initializeTags();
    
  public:
    
    /** \brief Constructor.
     */
    Vectorized_Basis(Teuchos::RCP< Basis<Scalar, ArrayScalar> > basis, int numComponents = 2);
    
    
    /** \brief  
     \param  outputValues      [out] - rank-3 or 4 array with the computed basis values
     \param  inputPoints       [in]  - rank-3 array with dimensions (P,D,V) containing reference points (V the vector component)
     \param  operatorType      [in]  - operator applied to basis functions
     */
    virtual void getValues(ArrayScalar &          outputValues,
                   const ArrayScalar &    inputPoints,
                   const EOperator        operatorType) const;
    
    
    /**  \brief  FVD basis evaluation: invocation of this method throws an exception if the components are not FVD bases.
     */
    virtual void getValues(ArrayScalar &          outputValues,
                   const ArrayScalar &    inputPoints,
                   const ArrayScalar &    cellVertices,
                   const EOperator        operatorType = OPERATOR_VALUE) const;
    
    void getVectorizedValues(ArrayScalar& outputValues, 
                             const ArrayScalar & componentOutputValues,
                             int fieldIndex) const;
    
    const Teuchos::RCP< Basis<Scalar, ArrayScalar> > getComponentBasis() const;
    int getNumComponents() const {
      return _numComponents;
    }
    
    int getDofOrdinalFromComponentDofOrdinal(int componentDofOrdinal, int componentIndex) const;
  };
}// namespace Intrepid

#include "Vectorized_BasisDef.hpp"

#endif