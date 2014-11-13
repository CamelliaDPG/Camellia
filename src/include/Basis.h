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

#include "CellTopology.h"
#include "CamelliaIntrepidExtendedTypes.h"

namespace Camellia {
  template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<Scalar> > class Basis;
  
  template<class Scalar, class ArrayScalar> class Basis {
  protected:
    Basis();
    
    int _basisCardinality;
    int _basisDegree;
    
    int _rangeDimension;
    int _rangeRank;
    
    IntrepidExtendedTypes::EFunctionSpaceExtended _functionSpace;
    
    virtual void initializeTags() const = 0;
    /** \brief  "true" if <var>_tagToOrdinal</var> and <var>_ordinalToTag</var> have been initialized
     */
    mutable bool _basisTagsAreSet;
    
    /** \brief  DoF ordinal to tag lookup table.
     
     Rank-2 array with dimensions (basisCardinality_, 4) containing the DoF tags. This array
     is left empty at instantiation and filled by initializeTags() only when tag data is
     requested.
     
     \li     ordinalToTag_[DofOrd][0] = dim. of the subcell associated with the specified DoF
     \li     ordinalToTag_[DofOrd][1] = ordinal of the subcell defined in the cell topology
     \li     ordinalToTag_[DodOrd][2] = ordinal of the specified DoF relative to the subcell
     \li     ordinalToTag_[DofOrd][3] = total number of DoFs associated with the subcell
     */
    mutable std::vector<std::vector<int> > _ordinalToTag;
    
    CellTopoPtr _domainTopology;
    
    /** \brief  DoF tag to ordinal lookup table.
     
     Rank-3 array with dimensions (maxScDim + 1, maxScOrd + 1, maxDfOrd + 1), i.e., the
     columnwise maximums of the 1st three columns in the DoF tag table for the basis plus 1.
     For every triple (subscDim, subcOrd, subcDofOrd) that is valid DoF tag data this array
     stores the corresponding DoF ordinal. If the triple does not correspond to tag data,
     the array stores -1. This array is left empty at instantiation and filled by
     initializeTags() only when tag data is requested.
     
     \li     tagToOrdinal_[subcDim][subcOrd][subcDofOrd] = Degree-of-freedom ordinal
     */
    mutable std::vector<std::vector<std::vector<int> > > _tagToOrdinal;
  public:
    virtual int getCardinality() const;
    virtual int getDegree() const;
    
    /** \brief  Returns the topology on which the basis is defined.
     */
    virtual CellTopoPtr domainTopology() const;
    
    // dof ordinal subsets:
    virtual std::set<int> dofOrdinalsForEdges(bool includeVertices = true) const;
    virtual std::set<int> dofOrdinalsForFaces(bool includeVerticesAndEdges = true) const;
    virtual std::set<int> dofOrdinalsForInterior() const;
    virtual std::set<int> dofOrdinalsForVertices() const;
    virtual std::set<int> dofOrdinalsForSubcells(int subcellDim, bool includeLesserDimensions) const;

    virtual std::set<int> dofOrdinalsForSubcell(int subcellDim, int subcellIndex, int minimumSubSubcellDimension) const; // e.g. can get vertex, edge, and face dofs for a side by specifying subcellDim=2, minimumSubSubcellDimension = 0
    virtual std::set<int> dofOrdinalsForSubcell(int subcellDim, int subcellIndex) const;
    virtual std::set<int> dofOrdinalsForVertex(int vertexIndex) const;
    virtual std::set<int> dofOrdinalsForEdge(int edgeIndex) const;
    
    virtual int getDofOrdinal(const int subcDim, const int subcOrd, const int subcDofOrd) const;
    virtual const std::vector<std::vector<std::vector<int> > > &getDofOrdinalData( ) const;
    virtual const std::vector<int>& getDofTag(int dofOrd) const;
    virtual const std::vector<std::vector<int> > & getAllDofTags() const;
    
    virtual IntrepidExtendedTypes::EFunctionSpaceExtended functionSpace() const;
    virtual IntrepidExtendedTypes::EFunctionSpaceExtended functionSpace(int tensorialRank) const; // 0 for space, 1 for time, in space-time bases
    
    // methods identifying the type of basis:
    virtual bool isConforming() const; // defaults to false (true for the Intrepid bases)
    virtual bool isNodal() const; // defaults to false (true for the Intrepid bases)
    
    // range info for basis values:
    virtual int rangeDimension() const;
    virtual int rangeRank() const;
    
    virtual void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const = 0;
    
    virtual void CHECK_VALUES_ARGUMENTS(const ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const;
    
    virtual ~Basis() {}
  };
  
                /***************** IntrepidBasisWrapper ******************/
  template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class IntrepidBasisWrapper;
  template<class Scalar, class ArrayScalar> class IntrepidBasisWrapper : public Basis<Scalar,ArrayScalar> {
  private:
    Teuchos::RCP< Intrepid::Basis<Scalar,ArrayScalar> > _intrepidBasis;
  protected:
    void initializeTags() const;
  public:
    IntrepidBasisWrapper(Teuchos::RCP< Intrepid::Basis<Scalar,ArrayScalar> > intrepidBasis, int rangeDimension, int rangeRank,
                         IntrepidExtendedTypes::EFunctionSpaceExtended functionSpace);
    
    Teuchos::RCP< Intrepid::Basis<Scalar,ArrayScalar> > intrepidBasis();
    
    virtual bool isConforming() const; // true for Intrepid bases
    virtual bool isNodal() const;      // true for the Intrepid bases
    
    void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const;
  };
} // namespace Camellia

typedef Teuchos::RCP< Camellia::Basis<> > BasisPtr;

#include "BasisDef.h"

#endif
