//
//  VectorizedBasis.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/21/13.
//
//

#ifndef __Camellia_debug__VectorizedBasis__
#define __Camellia_debug__VectorizedBasis__

#include "Basis.h"

typedef Intrepid::EOperator EOperator;

namespace Camellia {
  template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class VectorizedBasis;
  template<class Scalar, class ArrayScalar> class VectorizedBasis : public Basis<Scalar,ArrayScalar> {
  private:
    Teuchos::RCP< Basis<Scalar, ArrayScalar> > _componentBasis;
    int _numComponents;
  protected:
    void initializeTags() const;
  public:
    VectorizedBasis(BasisPtr basis, int numComponents = 2);
    virtual ~VectorizedBasis() {}
    
    int getCardinality() const;
    int getDegree() const;
    
    int getDofOrdinalFromComponentDofOrdinal(int componentDofOrdinal, int componentIndex) const;
    void getVectorizedValues(ArrayScalar& outputValues, const ArrayScalar & componentOutputValues,
                             int fieldIndex) const;
    
    const Teuchos::RCP< Basis<Scalar, ArrayScalar> > getComponentBasis() const;
    int getNumComponents() const {
      return _numComponents;
    }
    
    // domain info on which the basis is defined:
    CellTopoPtr domainTopology() const;
    
    // dof ordinal subsets:
    std::set<int> dofOrdinalsForSubcells(int subcellDim, bool includeLesserDimensions) const;
  //  std::set<int> dofOrdinalsForEdges(bool includeVertices = true) const;
  //  std::set<int> dofOrdinalsForFaces(bool includeVerticesAndEdges = true) const;
  //  std::set<int> dofOrdinalsForInterior() const;
  //  std::set<int> dofOrdinalsForVertices() const;
    
    // range info for basis values:
    int rangeDimension() const;
    int rangeRank() const;
    
    void getValues(ArrayScalar &values, const ArrayScalar &refPoints, EOperator operatorType) const;
  };
  typedef Teuchos::RCP<VectorizedBasis<> > VectorBasisPtr;

} //namespace Camellia

#include "VectorizedBasisDef.h"

#endif /* defined(__Camellia_debug__VectorizedBasis__) */
