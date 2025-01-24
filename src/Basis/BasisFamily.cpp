//
//  BasisFamily.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 3/14/22.
//

#include "BasisFamily.h"

#include "Basis.h"
#include "CamelliaIntrepidExtendedTypes.h"
#include "CellTopology.h"
#include "PointBasis.h"

#include <Intrepid2_HierarchicalBasisFamily.hpp>
#include <Intrepid2_NodalBasisFamily.hpp>

using namespace Camellia;

namespace {
  BasisPtr intrepid2TensorBasis(BasisPtr firstBasis, BasisPtr extrusionBasis)
  {
    using BasisBase   = Intrepid2::Basis<Camellia::DeviceType,double,double>;
    using TensorBasis = Intrepid2::Basis_TensorBasis<BasisBase>;

    if (firstBasis->domainTopology()->getDimension() == 0)
    {
      // firstBasis is defined on a Node, and therefore isn't actually an Intrepid2 Basis.
      // We assume that it's just a unit-valued constant, so that we can without too much loss
      // simply return extrusionBasis.
      return extrusionBasis;
    }
    
    const int spaceDim  = firstBasis->rangeDimension() + extrusionBasis->rangeDimension();
    const int rangeRank = firstBasis->rangeRank();
    
    Camellia::EFunctionSpace  fs = Camellia::FUNCTION_SPACE_UNKNOWN;
    Intrepid2::EFunctionSpace fs_Intrepid2 = Intrepid2::FUNCTION_SPACE_MAX;

    auto fsFirst = firstBasis->functionSpace();
    auto fsExtrusion = extrusionBasis->functionSpace();
    if (fsFirst == fsExtrusion)
    {
      if ((fsFirst == FUNCTION_SPACE_HGRAD) || (fsFirst == FUNCTION_SPACE_HGRAD_DISC))
      {
        fs_Intrepid2 = Intrepid2::FUNCTION_SPACE_HGRAD;
        fs = fsFirst;
      }
      else
      {
        fs_Intrepid2 = Intrepid2::FUNCTION_SPACE_HVOL;
        fs = fsFirst;
      }
    }
    
    auto firstBasis_intrepid2 = firstBasis->intrepid2Basis();
    auto extrusion_intrepid2  = extrusionBasis->intrepid2Basis();
    
    auto tensorBasis_Intrepid2 = Teuchos::rcp(new TensorBasis(firstBasis_intrepid2, extrusion_intrepid2, fs_Intrepid2));

    return Teuchos::rcp( new Intrepid2BasisWrapper<>(tensorBasis_Intrepid2, spaceDim, rangeRank, fs));
  }

}; // anonymous namespace

class HierarchicalBasisFamily : public BasisFamily
{
public:
  HierarchicalBasisFamily();
  
  using BasisFamily::getBasis;
  BasisPtr getBasis(CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs_in, const std::vector<int> &p_H1_in) override
  {
    auto fs = fs_in;
    std::vector<int> p_H1 = p_H1_in;
    {
      // FUNCTION_SPACE_REAL_SCALAR: means we want elementwise constant basis
      // We allow the fs argument to be one long; if so, it will apply to each dimension of the p_H1 argument
      if (fs[0] == FUNCTION_SPACE_REAL_SCALAR)
      {
        fs[0]   = FUNCTION_SPACE_HVOL;
        p_H1[0] = 1; // H^1 order 1 --> L^2 order 0
        if (fs.size() == 1)
        {
          for (int r=1; r<p_H1.size(); r++)
          {
            p_H1[r] = 1;
          }
        }
      }
      for (int r=1; r<fs.size(); r++)
      {
        if (fs[r] == FUNCTION_SPACE_REAL_SCALAR)
        {
          fs[r]   = FUNCTION_SPACE_HVOL;
          p_H1[r] = 1;
        }
      }
    }
    auto basis = getCachedBasis(cellTopo,fs,p_H1);
    if (basis == Teuchos::null)
    {
      if (cellTopo->getTensorialDegree() == 0)
      {
        using Intrepid2BasisFamily = Intrepid2::HierarchicalBasisFamily<Camellia::DeviceType>;
        using Intrepid2BasisPtr = Intrepid2BasisFamily::BasisPtr;
        Intrepid2::EFunctionSpace intrepid2FS;
        int rank;
        // TODO: factor out this switch statement (will be reused)
        switch (fs[0])
        {
          case FUNCTION_SPACE_HGRAD:      intrepid2FS = Intrepid2::FUNCTION_SPACE_HGRAD; rank = 0; break;
          case FUNCTION_SPACE_HCURL:      intrepid2FS = Intrepid2::FUNCTION_SPACE_HCURL; rank = 1; break;
          case FUNCTION_SPACE_HDIV:       intrepid2FS = Intrepid2::FUNCTION_SPACE_HDIV;  rank = 1; break;
          case FUNCTION_SPACE_HVOL:       intrepid2FS = Intrepid2::FUNCTION_SPACE_HVOL;  rank = 0; break;
          case FUNCTION_SPACE_HGRAD_DISC: intrepid2FS = Intrepid2::FUNCTION_SPACE_HGRAD; rank = 0; break;
          case FUNCTION_SPACE_HCURL_DISC: intrepid2FS = Intrepid2::FUNCTION_SPACE_HCURL; rank = 1; break;
          case FUNCTION_SPACE_HDIV_DISC:  intrepid2FS = Intrepid2::FUNCTION_SPACE_HDIV;  rank = 1; break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported EFunctionSpace");
        }
        
        const int spaceDim = cellTopo->getDimension();
        if (spaceDim == 0)
        {
          basis = PointBasis<>::pointBasis();
        }
        else if (p_H1.size() == 1)
        {
          int polyOrder = (intrepid2FS == Intrepid2::FUNCTION_SPACE_HVOL) ? p_H1[0] - 1 : p_H1[0];
          Intrepid2BasisPtr intrepid2Basis = Intrepid2::getBasis<Intrepid2BasisFamily>(cellTopo->getShardsTopology(), intrepid2FS, polyOrder);
          basis = Teuchos::rcp( new Intrepid2BasisWrapper<>(intrepid2Basis, spaceDim, rank, fs[0]) );
        }
        else
        {
          if (cellTopo->isHypercube())
          {
            Intrepid2BasisPtr intrepid2Basis;
            if (spaceDim == 3)
            {
              intrepid2Basis = Intrepid2::getHexahedronBasis<Intrepid2BasisFamily>(intrepid2FS, p_H1[0], p_H1[1], p_H1[2]);
            }
            else if (spaceDim == 2)
            {
              intrepid2Basis = Intrepid2::getQuadrilateralBasis<Intrepid2BasisFamily>(intrepid2FS, p_H1[0], p_H1[1]);
            }
            else if (spaceDim == 1)
            {
              intrepid2Basis = Intrepid2::getLineBasis<Intrepid2BasisFamily>(intrepid2FS, p_H1[0]);
            }
            basis = Teuchos::rcp( new Intrepid2BasisWrapper<>(intrepid2Basis, spaceDim, rank, fs[0]) );
          }
          else
          {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported CellTopo for anisotropic poly order");
          }
        }
      }
      else
      {
        std::vector<EFunctionSpace> fsComponentVector;
        std::vector<int> p_H1_component;
        EFunctionSpace fsExtrusion;
        int p_H1_extrusion;
        if (fs.size() == 1)
        {
          fsComponentVector = fs;
          fsExtrusion = fs[0];
        }
        else
        {
          fsComponentVector.insert(fsComponentVector.begin(), fs.begin(), fs.end() - 1);
          fsExtrusion = fs[fs.size()-1];
        }
        if (p_H1.size() == 1)
        {
          p_H1_component = p_H1;
          p_H1_extrusion = p_H1[0];
        }
        else
        {
          p_H1_component.insert(p_H1_component.begin(), p_H1.begin(), p_H1.end() - 1);
          p_H1_extrusion = p_H1[p_H1.size() - 1];
        }
        auto tensorialComponentBasis = getBasis(cellTopo->getTensorialComponent(), fsComponentVector, p_H1_component);
        auto lineBasis = getBasis(CellTopology::line(), fsExtrusion, p_H1_extrusion);
        
        basis = intrepid2TensorBasis(tensorialComponentBasis, lineBasis);
      }
      setCachedBasis(cellTopo,fs,p_H1,basis);
    }
    return basis;
  }
};

class NodalBasisFamily : public BasisFamily
{
public:
  NodalBasisFamily();
  
  using BasisFamily::getBasis;
  BasisPtr getBasis(CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1) override
  {
    auto basis = getCachedBasis(cellTopo,fs,p_H1);
    if (basis == Teuchos::null)
    {
      if (cellTopo->getTensorialDegree() == 0)
      {
        using Intrepid2BasisFamily = Intrepid2::DerivedNodalBasisFamily<Camellia::DeviceType>;
        using Intrepid2BasisPtr = Intrepid2BasisFamily::BasisPtr;
        Intrepid2::EFunctionSpace intrepid2FS;
        int rank;
        // TODO: factor out this switch statement (will be reused)
        switch (fs[0])
        {
          case FUNCTION_SPACE_HGRAD:      intrepid2FS = Intrepid2::FUNCTION_SPACE_HGRAD; rank = 0; break;
          case FUNCTION_SPACE_HCURL:      intrepid2FS = Intrepid2::FUNCTION_SPACE_HCURL; rank = 1; break;
          case FUNCTION_SPACE_HDIV:       intrepid2FS = Intrepid2::FUNCTION_SPACE_HDIV;  rank = 1; break;
          case FUNCTION_SPACE_HVOL:       intrepid2FS = Intrepid2::FUNCTION_SPACE_HVOL;  rank = 0; break;
          case FUNCTION_SPACE_HGRAD_DISC: intrepid2FS = Intrepid2::FUNCTION_SPACE_HGRAD; rank = 0; break;
          case FUNCTION_SPACE_HCURL_DISC: intrepid2FS = Intrepid2::FUNCTION_SPACE_HCURL; rank = 1; break;
          case FUNCTION_SPACE_HDIV_DISC:  intrepid2FS = Intrepid2::FUNCTION_SPACE_HDIV;  rank = 1; break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported EFunctionSpace");
        }
        
        const int spaceDim = cellTopo->getDimension();
        if (spaceDim == 0)
        {
          basis = PointBasis<>::pointBasis();
        }
        else if (p_H1.size() == 1)
        {
          int polyOrder = (intrepid2FS == Intrepid2::FUNCTION_SPACE_HVOL) ? p_H1[0] - 1 : p_H1[0];
          Intrepid2BasisPtr intrepid2Basis = Intrepid2::getBasis<Intrepid2BasisFamily>(cellTopo->getShardsTopology(), intrepid2FS, polyOrder);
          basis = Teuchos::rcp( new Intrepid2BasisWrapper<>(intrepid2Basis, spaceDim, rank, fs[0]) );
        }
        else
        {
          if (cellTopo->isHypercube())
          {
            Intrepid2BasisPtr intrepid2Basis;
            if (spaceDim == 3)
            {
              intrepid2Basis = Intrepid2::getHexahedronBasis<Intrepid2BasisFamily>(intrepid2FS, p_H1[0], p_H1[1], p_H1[2]);
            }
            else if (spaceDim == 2)
            {
              intrepid2Basis = Intrepid2::getQuadrilateralBasis<Intrepid2BasisFamily>(intrepid2FS, p_H1[0], p_H1[1]);
            }
            else if (spaceDim == 1)
            {
              intrepid2Basis = Intrepid2::getLineBasis<Intrepid2BasisFamily>(intrepid2FS, p_H1[0]);
            }
            basis = Teuchos::rcp( new Intrepid2BasisWrapper<>(intrepid2Basis, spaceDim, rank, fs[0]) );
          }
          else
          {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported CellTopo for anisotropic poly order");
          }
        }
      }
      else
      {
        std::vector<EFunctionSpace> fsComponentVector;
        std::vector<int> p_H1_component;
        EFunctionSpace fsExtrusion;
        int p_H1_extrusion;
        if (fs.size() == 1)
        {
          fsComponentVector = fs;
          fsExtrusion = fs[0];
        }
        else
        {
          fsComponentVector.insert(fsComponentVector.begin(), fs.begin(), fs.end() - 1);
          fsExtrusion = fs[fs.size()-1];
        }
        if (p_H1.size() == 1)
        {
          p_H1_component = p_H1;
          p_H1_extrusion = p_H1[0];
        }
        else
        {
          p_H1_component.insert(p_H1_component.begin(), p_H1.begin(), p_H1.end() - 1);
          p_H1_extrusion = p_H1[p_H1.size() - 1];
        }
        auto tensorialComponentBasis = getBasis(cellTopo->getTensorialComponent(), fsComponentVector, p_H1_component);
        auto lineBasis = getBasis(CellTopology::line(), fsExtrusion, p_H1_extrusion);
        
        basis = intrepid2TensorBasis(tensorialComponentBasis, lineBasis);
      }
      setCachedBasis(cellTopo,fs,p_H1,basis);
    }
    return basis;
  }
};

class SerendipityBasisFamily : public BasisFamily
{
public:
  SerendipityBasisFamily();
  
  using BasisFamily::getBasis;
  BasisPtr getBasis(CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1) override
  {
    auto basis = getCachedBasis(cellTopo,fs,p_H1);
    if (basis == Teuchos::null)
    {
      auto hierarchicalBasis = hierarchicalBasisFamily()->getBasis(cellTopo, fs, p_H1);
      using Intrepid2Basis = Intrepid2::Basis<DeviceType,double,double>;
      Teuchos::RCP<Intrepid2Basis> serendipityIntrepid2 = Teuchos::rcp( new Intrepid2::SerendipityBasis<Intrepid2Basis>(hierarchicalBasis->intrepid2Basis()) );
      
      basis = Teuchos::rcp(new Intrepid2BasisWrapper<>(serendipityIntrepid2, serendipityIntrepid2->getDomainDimension(), hierarchicalBasis->rangeRank(),
                                                       hierarchicalBasis->functionSpace()) );
      
      setCachedBasis(cellTopo,fs,p_H1,basis);
    }
    return basis;
  }
};

class SingletonBasisFamily : public BasisFamily
{
  BasisPtr _basis;
public:
  SingletonBasisFamily(BasisPtr basis);
  
  using BasisFamily::getBasis;
  BasisPtr getBasis(CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1) override
  {
    // TODO: consider adding some input checks.  At minimum, the cellTopo should match that of the basis.  I can imagine use cases where the basis does not match the fs or poly order passed in, though.  (E.g. where a variable is always discretized with a constant.)
    return _basis;
  }
};

BasisFamily::BasisFamily(const std::string &name, bool pushFinalizeHookToClearCachedBases)
:
_name(name)
{
  if (pushFinalizeHookToClearCachedBases)
  {
    Kokkos::push_finalize_hook( [=] {
      _cachedBases.clear();
    });
  }
}

BasisFamilyPtr BasisFamily::basisFamily(const EBasisFamily &bf)
{
  switch (bf)
  {
    case BASIS_FAMILY_HIERARCHICAL: return hierarchicalBasisFamily();
    case BASIS_FAMILY_SERENDIPITY:  return serendipityBasisFamily();
    case BASIS_FAMILY_NODAL:        return nodalBasisFamily();
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported EBasisFamily choice");
  }
}

BasisPtr BasisFamily::getBasis(CellTopoPtr cellTopo, EFunctionSpace fs, const int p_H1)
{
  std::vector<EFunctionSpace> fsVector {fs};
  std::vector<int> pVector {p_H1};
  return getBasis(cellTopo, fsVector, pVector);
}

BasisPtr BasisFamily::getCachedBasis(CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1)
{
  using Key = std::tuple<CellTopologyKey, std::vector<EFunctionSpace>, std::vector<int> >; // vector<int>: p_H1
  Key key = std::make_tuple(cellTopo->getKey(), fs, p_H1);
  if (_cachedBases.find(key) != _cachedBases.end())
  {
    return _cachedBases[key];
  }
  return Teuchos::null;
}

//! returns a basis that corresponds to taking a derivative d/dx_i of the basis returned by getBasis(cellTopo, fs, p_H1), alongside a matrix of coefficients specifying the 1D basis weights.  The outer (first) index for the weights is the basis ordinal in the original 1D component basis; the inner/second index is the basis ordinal in the derivative 1D component basis.  (We specify it this way, rather than in terms of the full tensor basis, to maintain maximum flexibility for downstream implementations to exploit the tensor structure.)
//! So far, this method supports only very limited use cases; it requires a hierarchical basis, and only supports fully-decomposed tensorial function spaces. The function space in the derivative dimension must be HGRAD or HGRAD_DISC.
std::pair<BasisPtr, std::vector<std::vector<double> > > BasisFamily::getDerivativeBasis( CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1, const int &i_zero_based)
{
  TEUCHOS_TEST_FOR_EXCEPTION(int(fs.size()) < i_zero_based, std::invalid_argument, "we require tensor-product fs with at least i+1 components");
  
  TEUCHOS_TEST_FOR_EXCEPTION((fs[i_zero_based] != FUNCTION_SPACE_HGRAD) && (fs[i_zero_based] != FUNCTION_SPACE_HGRAD_DISC), std::invalid_argument, "fs in the derivative dimension must be H^1");
  
  // copy fs container
  auto fs_dxi = fs;
  fs_dxi[i_zero_based] = FUNCTION_SPACE_HVOL;
  
  auto basis_dxi = getBasis(cellTopo,fs_dxi,p_H1);
  
  using std::vector;
  const int H1_cardinality = p_H1[i_zero_based] + 1;
  const int L2_cardinality = H1_cardinality - 1;
  vector<vector<double> > weights(H1_cardinality, vector<double>(L2_cardinality, 0.0));
  
  // on the line:
  // - first H^1 basis function is 1-x
  // - second H^1 basis function is x.
  // - first L^2 basis function is 1/2.
  // - For i > 0, the (i-1)th L^2 basis function is 1/2 the derivative of the ith H^1 basis function.
  // (the 1/2 values have to do with the fact that our ref element is (-1,1), and has width 2.)
  weights[0][0] = -0.5;
  for (int i=1; i<H1_cardinality; i++)
  {
    weights[i][i-1] = 0.5;
  }
  
  return {basis_dxi,weights};
}

BasisFamilyPtr BasisFamily::hierarchicalBasisFamily()
{
  static BasisFamilyPtr basisFamily = Teuchos::rcp(new HierarchicalBasisFamily);
  return basisFamily;
}

const std::string & BasisFamily::name()
{
  return _name;
}

BasisFamilyPtr BasisFamily::nodalBasisFamily()
{
  static BasisFamilyPtr basisFamily = Teuchos::rcp(new NodalBasisFamily);
  return basisFamily;
}


BasisFamilyPtr BasisFamily::serendipityBasisFamily()
{
  static BasisFamilyPtr basisFamily = Teuchos::rcp(new SerendipityBasisFamily);
  return basisFamily;
}

BasisFamilyPtr BasisFamily::singletonBasisFamily(BasisPtr basis)
{
  BasisFamilyPtr basisFamily = Teuchos::rcp(new SingletonBasisFamily(basis));
  return basisFamily;
}

void BasisFamily::setCachedBasis(CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1, BasisPtr basis)
{
  using Key = std::tuple<CellTopologyKey, std::vector<EFunctionSpace>, std::vector<int> >; // vector<int>: p_H1
  Key key = std::make_tuple(cellTopo->getKey(), fs, p_H1);
  
  _cachedBases[key] = basis;
}

HierarchicalBasisFamily::HierarchicalBasisFamily()
: BasisFamily("Intrepid2 Hierarchical", true) // true because this only ever gets constructed as a static that persists through Kokkos::finalize()
{}

NodalBasisFamily::NodalBasisFamily()
: BasisFamily("Intrepid2 Nodal", true) // true because this only ever gets constructed as a static that persists through Kokkos::finalize()
{}

SerendipityBasisFamily::SerendipityBasisFamily()
: BasisFamily("Intrepid2 Serendipity", true) // true because this only ever gets constructed as a static that persists through Kokkos::finalize()
{}

SingletonBasisFamily::SingletonBasisFamily(BasisPtr basis)
:
_basis(basis),
BasisFamily("Singleton", false) // false because we don't store this as a static that persists through Kokkos::finalize()
{}
