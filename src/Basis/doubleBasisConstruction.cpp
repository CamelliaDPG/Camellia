//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "doubleBasisConstruction.h"

#include "Basis.h"

#include "LobattoHGRAD_QuadBasis.h"
#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid_HDIV_QUAD_In_FEM.hpp"
#include "Intrepid_HDIV_HEX_In_FEM.hpp"

#include <Intrepid2_HGRAD_LINE_Cn_FEM.hpp>
#include <Intrepid2_HVOL_LINE_Cn_FEM.hpp>
#include <Intrepid2_HGRAD_LINE_C1_FEM.hpp>

#include <Intrepid2_HGRAD_TRI_Cn_FEM.hpp>
#include <Intrepid2_HDIV_TRI_In_FEM.hpp>
#include <Intrepid2_HCURL_TRI_In_FEM.hpp>
#include <Intrepid2_HVOL_TRI_Cn_FEM.hpp>

#include <Intrepid2_HGRAD_TET_Cn_FEM.hpp>
#include <Intrepid2_HCURL_TET_In_FEM.hpp>
#include <Intrepid2_HDIV_TET_In_FEM.hpp>
#include <Intrepid2_HGRAD_TET_C1_FEM.hpp>

#include <Intrepid2_HGRAD_QUAD_Cn_FEM.hpp>
#include <Intrepid2_HCURL_QUAD_In_FEM.hpp>
#include <Intrepid2_HDIV_QUAD_In_FEM.hpp>
#include <Intrepid2_HGRAD_QUAD_C1_FEM.hpp>

#include <Intrepid2_HGRAD_HEX_Cn_FEM.hpp>
#include <Intrepid2_HCURL_HEX_In_FEM.hpp>
#include <Intrepid2_HDIV_HEX_In_FEM.hpp>
#include <Intrepid2_HGRAD_HEX_C1_FEM.hpp>

#include <Intrepid2_TensorBasis.hpp>

#include "TensorBasis.h"

namespace Camellia
{

BasisPtr lobattoQuadHGRAD(int polyOrder, bool conforming)
{
  return Teuchos::rcp( new LobattoHGRAD_QuadBasis<>(polyOrder,conforming) );
}

BasisPtr intrepidQuadHGRAD(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD;
  int scalarRank = 0;
  int spaceDim = 2;
  return Teuchos::rcp( new IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >( Teuchos::rcp( new Intrepid::Basis_HGRAD_QUAD_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,Intrepid::POINTTYPE_SPECTRAL)), spaceDim, scalarRank, fs) );
}

BasisPtr intrepidQuadHDIV(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV;
  int vectorRank = 1;
  int spaceDim = 2;
  return Teuchos::rcp( new IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >( Teuchos::rcp( new Intrepid::Basis_HDIV_QUAD_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,Intrepid::POINTTYPE_SPECTRAL)), spaceDim, vectorRank, fs) );
}

BasisPtr intrepidHexHGRAD(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD;
  int scalarRank = 0;
  int spaceDim = 3;
  return Teuchos::rcp( new IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >( Teuchos::rcp( new Intrepid::Basis_HGRAD_HEX_Cn_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,Intrepid::POINTTYPE_SPECTRAL)), spaceDim, scalarRank, fs) );
}

BasisPtr intrepidHexHDIV(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV;
  int vectorRank = 1;
  int spaceDim = 3;
  return Teuchos::rcp( new IntrepidBasisWrapper< double, Intrepid::FieldContainer<double> >( Teuchos::rcp( new Intrepid::Basis_HDIV_HEX_In_FEM<double, Intrepid::FieldContainer<double> >(polyOrder,Intrepid::POINTTYPE_SPECTRAL)), spaceDim, vectorRank, fs) );
}

BasisPtr intrepid2LineHGRAD(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD;
  int scalarRank = 0;
  int spaceDim = 1;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_LINE_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2LineHGRAD_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD_DISC;
  int scalarRank = 0;
  int spaceDim = 1;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_LINE_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2LineHVOL(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HVOL;
  int scalarRank = 0;
  int spaceDim = 1;
  
  // TODO: switch to Intrepid2 HVOL basis (right now, using lower-order HGRAD, and mapping dof ordinals to belong to volume)
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_LINE_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2TriHGRAD(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD;
  int scalarRank = 0;
  int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_TRI_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2TriHGRAD_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD_DISC;
  int scalarRank = 0;
  int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_TRI_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2TriHCURL(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HCURL;
  const int vectorRank = 1;
  const int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HCURL_TRI_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                      spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2TriHCURL_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HCURL_DISC;
  const int vectorRank = 1;
  const int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HCURL_TRI_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2TriHDIV(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV;
  const int vectorRank = 1;
  const int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HDIV_TRI_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                      spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2TriHDIV_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV_DISC;
  const int vectorRank = 1;
  const int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HDIV_TRI_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2TriHVOL(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HVOL;
  const int scalarRank = 0;
  const int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HVOL_TRI_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                      spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2QuadHGRAD(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD;
  int scalarRank = 0;
  int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_QUAD_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2QuadHGRAD_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD_DISC;
  int scalarRank = 0;
  int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_QUAD_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2QuadHCURL(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HCURL;
  int vectorRank = 1;
  int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HCURL_QUAD_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2QuadHCURL_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HCURL_DISC;
  int vectorRank = 1;
  int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HCURL_QUAD_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2QuadHDIV(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV;
  int vectorRank = 1;
  int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HDIV_QUAD_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2QuadHDIV_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV_DISC;
  int vectorRank = 1;
  int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HDIV_QUAD_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2QuadHVOL(int polyOrder)
{
  // historically, we've used the HGRAD basis from intrepid for HVOL, on the idea that specific HVOL bases were not available in Intrepid.  They are available in Intrepid2, and we should perhaps use these instead.  But for now we stick with the earlier setup, using HGRAD, marked with an HVOL function space.  (Where it gets tricky to use different bases for HVOL and HGRAD is that in some places we "upgrade" from discontinuous to continuous bases; not sure whether we do this with HVOL ever, but we do it for HGRAD_DISC; we'd need to check...)
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HVOL;
  int scalarRank = 0;
  int spaceDim = 2;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_QUAD_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2HexHGRAD(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD;
  int scalarRank = 0;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_HEX_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2HexHGRAD_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD_DISC;
  int scalarRank = 0;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_HEX_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2HexHCURL(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HCURL;
  int vectorRank = 1;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HCURL_HEX_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2HexHCURL_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HCURL_DISC;
  int vectorRank = 1;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HCURL_HEX_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2HexHDIV(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV;
  int vectorRank = 1;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HDIV_HEX_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2HexHDIV_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV_DISC;
  int vectorRank = 1;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HDIV_HEX_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2HexHVOL(int polyOrder)
{
  // historically, we've used the HGRAD basis from intrepid for HVOL, on the idea that specific HVOL bases were not available in Intrepid.  They are available in Intrepid2, and we should perhaps use these instead.  But for now we stick with the earlier setup, using HGRAD, marked with an HVOL function space.  (Where it gets tricky to use different bases for HVOL and HGRAD is that in some places we "upgrade" from discontinuous to continuous bases; not sure whether we do this with HVOL ever, but we do it for HGRAD_DISC; we'd need to check...)
  
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HVOL;
  int scalarRank = 0;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_HEX_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}


BasisPtr intrepid2TetHGRAD(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD;
  int scalarRank = 0;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_TET_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2TetHGRAD_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD_DISC;
  int scalarRank = 0;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_TET_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2TetHCURL(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HCURL;
  int vectorRank = 1;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HCURL_TET_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2TetHCURL_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HCURL_DISC;
  int vectorRank = 1;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HCURL_TET_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2TetHDIV(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV;
  int vectorRank = 1;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HDIV_TET_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2TetHDIV_DISC(int polyOrder)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HDIV_DISC;
  int vectorRank = 1;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HDIV_TET_In_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, vectorRank, fs)
  );
}

BasisPtr intrepid2TetHVOL(int polyOrder)
{
  // historically, we've used the HGRAD basis from intrepid for HVOL, on the idea that specific HVOL bases were not available in Intrepid.  They are available in Intrepid2, and we should perhaps use these instead.  But for now we stick with the earlier setup, using HGRAD, marked with an HVOL function space.  (Where it gets tricky to use different bases for HVOL and HGRAD is that in some places we "upgrade" from discontinuous to continuous bases; not sure whether we do this with HVOL ever, but we do it for HGRAD_DISC; we'd need to check...)
 
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HVOL;
  int scalarRank = 0;
  int spaceDim = 3;
  
  return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_TET_Cn_FEM<Camellia::DeviceType, double, double >(polyOrder,Intrepid2::POINTTYPE_WARPBLEND)),
                                                   spaceDim, scalarRank, fs)
  );
}

BasisPtr intrepid2HGRAD(const shards::CellTopology &cellTopo, int polyOrder)
{
  switch (cellTopo.getBaseKey())
  {
    case shards::Line<>::key:          return intrepid2LineHGRAD(polyOrder);
    case shards::Triangle<>::key:      return intrepid2TriHGRAD(polyOrder);
    case shards::Quadrilateral<>::key: return intrepid2QuadHGRAD(polyOrder);
    case shards::Tetrahedron<>::key:   return intrepid2TetHGRAD(polyOrder);
    case shards::Hexahedron<>::key:    return intrepid2HexHGRAD(polyOrder);
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported CellTopology");
  }
}

BasisPtr intrepid2HGRADLowestOrder(const shards::CellTopology &cellTopo)
{
  return intrepid2HGRADLowestOrder(cellTopo.getBaseKey());
}

BasisPtr intrepid2HGRADLowestOrder(const unsigned &cellTopoKey)
{
  Camellia::EFunctionSpace fs = Camellia::FUNCTION_SPACE_HGRAD;
  const int scalarRank = 0;
  switch (cellTopoKey)
  {
    case shards::Line<>::key:
    {
      const int spaceDim = 1;
      return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_LINE_C1_FEM<Camellia::DeviceType, double, double >()),
                                                       spaceDim, scalarRank, fs)
      );
    }
    case shards::Triangle<>::key:
    {
      const int spaceDim = 2;
      return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_TRI_C1_FEM<Camellia::DeviceType, double, double >()),
                                                       spaceDim, scalarRank, fs)
      );
    }
    case shards::Quadrilateral<>::key:
    {
      const int spaceDim = 2;
      return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<Camellia::DeviceType, double, double >()),
                                                       spaceDim, scalarRank, fs)
      );
    }
    case shards::Tetrahedron<>::key:
    {
      const int spaceDim = 3;
      return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_TET_C1_FEM<Camellia::DeviceType, double, double >()),
                                                       spaceDim, scalarRank, fs)
      );
    }
    case shards::Hexahedron<>::key:
    {
      const int spaceDim = 3;
      return Teuchos::rcp( new Intrepid2BasisWrapper<>( Teuchos::rcp( new Intrepid2::Basis_HGRAD_HEX_C1_FEM<Camellia::DeviceType, double, double >()),
                                                       spaceDim, scalarRank, fs)
      );
    }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported CellTopology");
  }
}

//using BasisBase   = Intrepid2::Basis<Camellia::DeviceType,double,double>;
//using TensorBasis = Intrepid2::Basis_TensorBasis<BasisBase>;
//
////! Just provides a required override mapping operators to component bases; ours just handles GRAD and VALUE
//class SimpleTensorBasis : public TensorBasis
//{
//  int _spaceDimFirstBasis;
//  int _spaceDimSecondBasis;
//public:
//  using OutputViewType         = typename TensorBasis::OutputViewType;
//  using PointViewType          = typename TensorBasis::PointViewType;
//  using Intrepid2BasisPtr      = Teuchos::RCP< Intrepid2::Basis<Camellia::DeviceType,double,double> >;
//
//  SimpleTensorBasis(Intrepid2BasisPtr firstBasis, Intrepid2BasisPtr extrusionBasis, Intrepid2::EFunctionSpace fsIntrepid2)
//  :
//  TensorBasis(firstBasis, extrusionBasis, fsIntrepid2)
//  {
//    _spaceDimFirstBasis = firstBasis->getBaseCellTopology().getDimension();
//    _spaceDimSecondBasis = extrusionBasis->getBaseCellTopology().getDimension();
//  }
//
//  virtual void getValues(OutputViewType outputValues, const Intrepid2::EOperator operatorType,
//                         const PointViewType  inputPoints1, const PointViewType  inputPoints2,
//                         bool tensorPoints) const override
//  {
//    // this code adapted from Intrepid2's Basis_Derived_HGRAD_HEX
//    Intrepid2::EOperator op1, op2;
//    if (operatorType == Intrepid2::OPERATOR_VALUE)
//    {
//      op1 = Intrepid2::OPERATOR_VALUE;
//      op2 = Intrepid2::OPERATOR_VALUE;
//
//      this->TensorBasis::getValues(outputValues,
//                                   inputPoints1, op1,
//                                   inputPoints2, op2, tensorPoints);
//    }
//    else if (operatorType == Intrepid2::OPERATOR_GRAD)
//    {
//      // to evaluate gradient, we actually need both OP_VALUE and OP_GRAD (thanks to product rule)
//      // for 1D line x line, we will put derivative * value in first component, and value * derivative in second
//
//      // outputValues1 and outputValues2 are computed by basis1 and basis2 -- these are tensorial components
//      // outputValuesComponent1 is a slice of the final output container (similarly, outputValuesComponent2)
//      // when the component basis is 1D, it expects not to have a "dimension" component in the output container
//      // the int argument in the dimension component creates a subview that skips the dimension component; the std::pair argument retains it
//      const bool component1IsVector = (_spaceDimFirstBasis > 1);
//      const bool component2IsVector = (_spaceDimSecondBasis > 1);
//
//      // compute first component -- derivative happens in first basis, and value taken in second
//      op1 = Intrepid2::OPERATOR_GRAD;
//      op2 = Intrepid2::OPERATOR_VALUE;
//
//      if (component1IsVector)
//      {
//        auto outputValuesComponent1 = Kokkos::subview(outputValues,Kokkos::ALL(),Kokkos::ALL(),std::make_pair(0,_spaceDimFirstBasis));
//        this->TensorBasis::getValues(outputValuesComponent1,
//                                     inputPoints1, op1,
//                                     inputPoints2, op2, tensorPoints);
//      }
//      else
//      {
//        auto outputValuesComponent1 = Kokkos::subview(outputValues,Kokkos::ALL(),Kokkos::ALL(),_spaceDimFirstBasis);
//        this->TensorBasis::getValues(outputValuesComponent1,
//                                     inputPoints1, op1,
//                                     inputPoints2, op2, tensorPoints);
//      }
//
//      // second component -- value in first, derivative in second
//      op1 = Intrepid2::OPERATOR_VALUE;
//      op2 = Intrepid2::OPERATOR_GRAD;
//
//      if (component2IsVector)
//      {
//        auto outputValuesComponent2 = Kokkos::subview(outputValues,Kokkos::ALL(),Kokkos::ALL(),std::make_pair(_spaceDimFirstBasis,_spaceDimSecondBasis+_spaceDimFirstBasis));
//        this->TensorBasis::getValues(outputValuesComponent2,
//                                     inputPoints1, op1,
//                                     inputPoints2, op2, tensorPoints);
//      }
//      else
//      {
//        auto outputValuesComponent2 = Kokkos::subview(outputValues,Kokkos::ALL(),Kokkos::ALL(),_spaceDimFirstBasis);
//
//        this->TensorBasis::getValues(outputValuesComponent2,
//                                     inputPoints1, op1,
//                                     inputPoints2, op2, tensorPoints);
//      }
//    }
//    else
//    {
//      INTREPID2_TEST_FOR_EXCEPTION(true,std::invalid_argument,"operator not yet supported");
//    }
//  }
//
//  /** \brief Returns a simple decomposition of the specified operator: what operator(s) should be applied to basis1, and what operator(s) to basis2.  A one-element vector corresponds to a single TensorData entry; a multiple-element vector corresponds to a VectorData object with axialComponents = false.
//  */
//  virtual Intrepid2::OperatorTensorDecomposition getSimpleOperatorDecomposition(const Intrepid2::EOperator operatorType) const override
//  {
//    const Intrepid2::EOperator VALUE = Intrepid2::OPERATOR_VALUE;
//    const Intrepid2::EOperator GRAD  = Intrepid2::OPERATOR_GRAD;
//
//    if (operatorType == VALUE)
//    {
//      return Intrepid2::OperatorTensorDecomposition(VALUE,VALUE);
//    }
//    else if (operatorType == GRAD)
//    {
//      // to evaluate gradient, we need both OP_VALUE and OP_GRAD (thanks to product rule)
//      // for quad x line, we will put derivative * value in first component, and value * derivative in second
//
//      std::vector< std::vector<Intrepid2::EOperator> > ops;
//      ops.push_back(std::vector<Intrepid2::EOperator>{GRAD,  VALUE});
//      ops.push_back(std::vector<Intrepid2::EOperator>{VALUE, GRAD});
//
//      std::vector<double> weights(ops.size(), 1.0);
//
//      return Intrepid2::OperatorTensorDecomposition(ops, weights);
//    }
//    else
//    {
//      INTREPID2_TEST_FOR_EXCEPTION(true,std::invalid_argument,"operator not yet supported");
//    }
//  }
//};

BasisPtr intrepid2TensorBasis(BasisPtr firstBasis, BasisPtr extrusionBasis)
{
  auto firstBasis_intrepid2 = firstBasis->intrepid2Basis();
  auto extrusion_intrepid2  = extrusionBasis->intrepid2Basis();
  
  TEUCHOS_TEST_FOR_EXCEPTION(firstBasis_intrepid2 == Teuchos::null, std::invalid_argument, "Arguments to intrepid2TensorBasis must define an Intrepid2 basis");
  TEUCHOS_TEST_FOR_EXCEPTION( extrusion_intrepid2 == Teuchos::null, std::invalid_argument, "Arguments to intrepid2TensorBasis must define an Intrepid2 basis");
  
  using BasisBase   = Intrepid2::Basis<Camellia::DeviceType,double,double>;
  using TensorBasis = Intrepid2::Basis_TensorBasis<BasisBase>;
  
  const int spaceDim = firstBasis->rangeDimension() + extrusionBasis->rangeDimension();
  const int rangeRank = std::max(firstBasis->rangeRank(), extrusionBasis->rangeRank()); // 0 for scalar-valued bases, 1 for vector, etc.

  Intrepid2::EFunctionSpace fs_Intrepid2;

  EFunctionSpace fs_Camellia;
  auto fsFirst = firstBasis->functionSpace();
  auto fsExtrusion = extrusionBasis->functionSpace();
  if ((fsFirst == FUNCTION_SPACE_HGRAD) && (fsExtrusion == FUNCTION_SPACE_HGRAD))
  {
    fs_Intrepid2 = Intrepid2::FUNCTION_SPACE_HGRAD;
    fs_Camellia  = FUNCTION_SPACE_HGRAD;
  }
  else if ((fsFirst == FUNCTION_SPACE_HGRAD_DISC) && (fsExtrusion == FUNCTION_SPACE_HGRAD_DISC))
  {
    fs_Intrepid2 = Intrepid2::FUNCTION_SPACE_HGRAD;
    fs_Camellia  = FUNCTION_SPACE_HGRAD_DISC;
  }
  else if ((fsFirst == FUNCTION_SPACE_HVOL) && (fsExtrusion == FUNCTION_SPACE_HVOL))
  {
    fs_Intrepid2 = Intrepid2::FUNCTION_SPACE_HVOL;
    fs_Camellia  = FUNCTION_SPACE_HVOL;
  }
  else if (firstBasis->domainTopology()->getDimension() == 0)
  {
    fs_Intrepid2 = toIntrepid2FunctionSpace(fsExtrusion);
    fs_Camellia  = fsExtrusion;
  }
  else
  {
    fs_Intrepid2 = Intrepid2::FUNCTION_SPACE_MAX;
    fs_Camellia  = EFunctionSpace::FUNCTION_SPACE_UNKNOWN;
  }
  auto tensorBasis_Intrepid2 = Teuchos::rcp(new TensorBasis(firstBasis_intrepid2, extrusion_intrepid2, fs_Intrepid2));

  return Teuchos::rcp( new Intrepid2BasisWrapper<>(tensorBasis_Intrepid2, spaceDim, rangeRank, fs_Camellia));
}

}
