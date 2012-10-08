#ifndef DPGTrilinos_Basis_HGRAD_1D_Cn_FEM
#define DPGTrilinos_Basis_HGRAD_1D_Cn_FEM

/** \file   
 \brief  Header file for the Intrepid::G_QUAD_C0_FEM class.
 \author Created by N. Roberts
 */

#include "Intrepid_Basis.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

using namespace Intrepid;
namespace Intrepid {
  
  /** \class  Intrepid::Basis_HGRAD_1D_Cn_FEM
   \brief  Implementation of the default H(grad)-compatible FEM basis of degree n on Line cell
   
   Implements Lagrangian basis of degree n on the reference Line. The basis has
   cardinality n+1 and spans a COMPLETE bi-linear polynomial space. Basis functions are dual 
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
  class Basis_HGRAD_1D_Cn_FEM : public Vectorized_Basis<Scalar, ArrayScalar> {
  private:
    int _spaceDim;
    
  public:
    
    /** \brief Constructor.
     */
    Basis_HGRAD_1D_Cn_FEM(int order, int cellTopoKey);
    
    
    /** \brief  FEM basis evaluation on a <strong>reference Line</strong> cell. 
     
     Returns values of <var>operatorType</var> acting on FEM basis functions for a set of
     points in the <strong>reference Line</strong> cell. For rank and dimensions of 
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

#include "Basis_HGRAD_1D_Cn_FEMDef.hpp"

#endif