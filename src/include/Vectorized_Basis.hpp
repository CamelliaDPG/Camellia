// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-Intrepid in the licenses directory.
//
// @HEADER

//
//  DPGTrilinos
//
//  Created by Nate Roberts on 8/9/11.
//

#ifndef DPGTrilinos_Vectorized_Basis
#define DPGTrilinos_Vectorized_Basis

/** \file   Vectorized_Basis.hpp
 \brief  Header file for the Vectorized_Basis class.
 \author Created by N. Roberts
 */

#include "Intrepid_Basis.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

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
                   const EOperator        operatorType = Intrepid::OPERATOR_VALUE) const;
    
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
