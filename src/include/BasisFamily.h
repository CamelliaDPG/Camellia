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
  
  class BasisFamily
  {
    const std::string _name;
    
    using Key = std::tuple<CellTopologyKey, std::vector<EFunctionSpace>, std::vector<int> >; // vector<int>: p_H1
    std::map<Key, BasisPtr> _cachedBases;
  protected:
    BasisPtr getCachedBasis(CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1);
    void setCachedBasis(CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1, BasisPtr basis);
  public:
    BasisFamily(const std::string &name);
    
    virtual BasisPtr getBasis(CellTopoPtr cellTopo, const std::vector<EFunctionSpace> &fs, const std::vector<int> &p_H1) = 0;
    BasisPtr getBasis(CellTopoPtr cellTopo, EFunctionSpace fs, const int p_H1);
    
    const std::string & name();
    
    static BasisFamilyPtr hierarchicalBasisFamily();
    static BasisFamilyPtr nodalBasisFamily();
    static BasisFamilyPtr serendipityBasisFamily();
  };
  
} // namespace Camellia

#endif /* BasisFamily_hpp */
