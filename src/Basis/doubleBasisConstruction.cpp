//
//  doubleBasisConstruction.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/11/13.
//
//

#include "doubleBasisConstruction.h"

#include "LobattoHGRAD_QuadBasis.h"
#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"
#include "Intrepid_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid_HDIV_QUAD_In_FEM.hpp"
#include "Intrepid_HDIV_HEX_In_FEM.hpp"

namespace Camellia {

  BasisPtr lobattoQuadHGRAD(int polyOrder, bool conforming=false) {
    return Teuchos::rcp( new LobattoHGRAD_QuadBasis<>(polyOrder,conforming) );
  }

  BasisPtr intrepidLineHGRAD(int polyOrder) {
    IntrepidExtendedTypes::EFunctionSpaceExtended fs = IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
    int scalarRank = 0;
    int spaceDim = 1;
    return Teuchos::rcp( new IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >( Teuchos::rcp( new Intrepid::Basis_HGRAD_LINE_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,Intrepid::POINTTYPE_SPECTRAL)), spaceDim, scalarRank, fs) );
  }
  
  BasisPtr intrepidQuadHGRAD(int polyOrder) {
    IntrepidExtendedTypes::EFunctionSpaceExtended fs = IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
    int scalarRank = 0;
    int spaceDim = 2;
    return Teuchos::rcp( new IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >( Teuchos::rcp( new Intrepid::Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,Intrepid::POINTTYPE_SPECTRAL)), spaceDim, scalarRank, fs) );
  }

  BasisPtr intrepidQuadHDIV(int polyOrder) {
    IntrepidExtendedTypes::EFunctionSpaceExtended fs = IntrepidExtendedTypes::FUNCTION_SPACE_HDIV;
    int vectorRank = 1;
    int spaceDim = 2;
    return Teuchos::rcp( new IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >( Teuchos::rcp( new Intrepid::Basis_HDIV_QUAD_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,Intrepid::POINTTYPE_SPECTRAL)), spaceDim, vectorRank, fs) );
  }
  
  BasisPtr intrepidHexHGRAD(int polyOrder) {
    IntrepidExtendedTypes::EFunctionSpaceExtended fs = IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD;
    int scalarRank = 0;
    int spaceDim = 3;
    return Teuchos::rcp( new IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >( Teuchos::rcp( new Intrepid::Basis_HGRAD_HEX_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,Intrepid::POINTTYPE_SPECTRAL)), spaceDim, scalarRank, fs) );
  }
  
  BasisPtr intrepidHexHDIV(int polyOrder) {
    IntrepidExtendedTypes::EFunctionSpaceExtended fs = IntrepidExtendedTypes::FUNCTION_SPACE_HDIV;
    int vectorRank = 1;
    int spaceDim = 3;
    return Teuchos::rcp( new IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >( Teuchos::rcp( new Intrepid::Basis_HDIV_HEX_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,Intrepid::POINTTYPE_SPECTRAL)), spaceDim, vectorRank, fs) );
  }
  
  

}