//
//  BasisFamily.h
//  Camellia
//
//  Created by Roberts, Nathan V on 3/14/22.
//

#ifndef BasisFamily_h
#define BasisFamily_h

#include "TypeDefs.h"

#include "CamelliaIntrepidExtendedTypes.h"
#include "CellTopology.h"

namespace Camellia
{
  class BasisFamily;
  using BasisFamilyPtr = Teuchos::RCP<BasisFamily>;
  
  enum EBasisFamily
  {
    BASIS_FAMILY_HIERARCHICAL,
    BASIS_FAMILY_SERENDIPITY,
    BASIS_FAMILY_NODAL
  };

  class BasisFamily
  {
    const std::string _name;
    
    using Key = std::tuple<CellTopologyKey, std::vector<EFunctionSpace>, std::vector<int> >; // vector<int>: p_H1
    std::map<Key, BasisPtr> _cachedBases;
  protected:
    BasisPtr getCachedBasis(CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1);
    void setCachedBasis(CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1, BasisPtr basis);
  public:
    BasisFamily(const std::string &name, bool pushFinalizeHookToClearCachedBases);
    
    virtual BasisPtr getBasis(CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1) = 0;
    BasisPtr getBasis(CellTopoPtr cellTopo, EFunctionSpace fs, const int p_H1);
    
    //! returns a basis that corresponds to taking a derivative d/dx_i of the basis returned by getBasis(cellTopo, fs, p_H1), alongside a matrix of coefficients specifying the 1D basis weights.  The outer (first) index for the weights is the basis ordinal in the original 1D component basis; the inner/second index is the basis ordinal in the derivative 1D component basis.  (We specify it this way, rather than in terms of the full tensor basis, to maintain maximum flexibility for downstream implementations to exploit the tensor structure.)
    //! So far, this method supports only very limited use cases; it requires a hierarchical basis, and only supports fully-decomposed tensorial function spaces. The function space in the derivative dimension must be HGRAD or HGRAD_DISC.
    virtual std::pair<BasisPtr, std::vector<std::vector<double> > > getDerivativeBasis( CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1, const int &i_zero_based);
    
    const std::string & name();
    
    static BasisFamilyPtr hierarchicalBasisFamily();
    static BasisFamilyPtr nodalBasisFamily();
    static BasisFamilyPtr serendipityBasisFamily();
    
    static BasisFamilyPtr basisFamily(const EBasisFamily &bf);
    
    static BasisFamilyPtr singletonBasisFamily(BasisPtr basis);
  };
  
} // namespace Camellia

#endif /* BasisFamily_hpp */
