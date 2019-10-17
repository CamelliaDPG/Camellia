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
//  Basis_HGRAD_C0_FEM.h
//  DPGTrilinos
//
//  Created by Nate Roberts on 8/9/11.
//

#ifndef DPGTrilinos_Basis_HGRAD_C0_FEM_h
#define DPGTrilinos_Basis_HGRAD_C0_FEM_h

/** \file   Intrepid_G_QUAD_C0_FEM.hpp
 \brief  Header file for the Intrepid::G_QUAD_C0_FEM class.
 \author Created by P. Bochev and D. Ridzal.
 */

#include "Intrepid_Basis.hpp"

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
                   const EOperator        operatorType = Intrepid::OPERATOR_VALUE) const;
    
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
