//
//  Basis.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/21/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_Basis_h
#define Camellia_debug_Basis_h

#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Shards_CellTopology.hpp"

namespace Camellia {
  template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<Scalar> > class Basis;
  
  template<class Scalar, class ArrayScalar> class Basis {
  public:
    virtual int getCardinality() = 0;
    
    // domain info on which the basis is defined:
    virtual shards::CellTopology domainTopology();
    
    // dof ordinal subsets:
    virtual std::set<int> dofOrdinalsForEdges(bool includeVertices = true) = 0;
    virtual std::set<int> dofOrdinalsForFaces(bool includeVerticesAndEdges = true) = 0;
    virtual std::set<int> dofOrdinalsForInterior() = 0;
    virtual std::set<int> dofOrdinalsForVertices() = 0;
    
    // range info for basis values:
    virtual int rangeDimension() = 0;
    virtual int rangeRank() = 0;
    
    virtual void values(ArrayScalar &values, const ArrayScalar &refPoints) = 0;
    
    virtual void CHECK_VALUES_ARGUMENTS(const ArrayScalar &values, const ArrayScalar &refPoints);
  };
  
  template<class Scalar, class ArrayScalar> class IntrepidBasisWrapper : public Basis<Scalar,ArrayScalar> {
  private:
    Intrepid::Basis<Scalar,ArrayScalar> _intrepidBasis;
    int _rangeDimension;
    int _rangeRank;
    
    std::set<int> getSubcellDofs(int subcellDimStart, int subcellDimEnd);
  public:
    IntrepidBasisWrapper(Intrepid::Basis<Scalar,ArrayScalar> intrepidBasis, int rangeDimension, int rangeRank);
    
    int getCardinality();
    
    // domain info on which the basis is defined:
    shards::CellTopology domainTopology();
    
    // dof ordinal subsets:
    std::set<int> dofOrdinalsForEdges(bool includeVertices = true);
    std::set<int> dofOrdinalsForFaces(bool includeVerticesAndEdges = true);
    std::set<int> dofOrdinalsForInterior();
    std::set<int> dofOrdinalsForVertices();
    
    // range info for basis values:
    int rangeDimension();
    int rangeRank();
    
    void values(ArrayScalar &values, const ArrayScalar &refPoints);
  };
}

#endif
