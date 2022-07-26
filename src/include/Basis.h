// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//  Basis.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/21/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_Basis_h
#define Camellia_debug_Basis_h

#include "Intrepid_Basis.hpp"
#include "Camellia_FieldContainer.hpp"
#include "Shards_CellTopology.hpp"

#include "Intrepid2_Basis.hpp"

#include "CellTopology.h"
#include "CamelliaIntrepidExtendedTypes.h"

#include <Intrepid2_TensorBasis.hpp>

namespace Camellia
{
template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<Scalar> > class Basis;

template<class Scalar, class ArrayScalar> class Basis
{
  /** \brief  "true" if <var>_tagToOrdinal</var> and <var>_ordinalToTag</var> have been initialized and initializeTagsAndTrim() has been called.
   */
  mutable bool _basisTagsAreSet;
  void initializeTagsAndTrim() const;
  
  using DeviceType = Camellia::DeviceType;
  
  // The (stored) result of casting to Intrepid2 TensorBasis and separating out into components.  Just a single weak ref to this if the cast fails.
  std::vector< Teuchos::RCP< ::Camellia::Basis<Scalar, ArrayScalar> > > _intrepid2TensorBasisComponents;
protected:
  Basis();
  
  Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > _continuousBasis; // continuous version of the same basis, if this basis is discontinuous

  int _basisCardinality;
  int _basisDegree;

  int _rangeDimension;
  int _rangeRank;

  Camellia::EFunctionSpace _functionSpace;

  virtual void initializeTags() const = 0;

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

   Rank-3 container with arguments (scdim, scord, subcDofOrd).  Subclasses (in particular IntrepidBasisWrapper) often fill some empty slots with -1s, but these are eliminated during initializeTagsAndTrim().

   \li     tagToOrdinal_[subcDim][subcOrd][subcDofOrd] = Degree-of-freedom ordinal
   */
  mutable std::vector<std::vector<std::vector<int> > > _tagToOrdinal;
public:
  //! allocation method for Intrepid2-style getValues() output argument.  (For non-Intrepid2 bases, throws an exception.)
  virtual Intrepid2::BasisValues<Scalar,DeviceType> allocateBasisValues( Intrepid2::TensorPoints<Scalar,DeviceType> points,
                                                                         const Intrepid2::EOperator op = Intrepid2::OPERATOR_VALUE) const;
  
  //! Intrepid2-style getValues() method.  This is part of the API that will be performance-optimized.  (Allows use of Intrepid2's sum factorization facilities, etc.)  Non-Intrepid2 bases throw an exception.
  virtual void getValues( Intrepid2::BasisValues<Scalar,DeviceType> outputValues,
                         const Intrepid2::TensorPoints<Scalar,DeviceType>  inputPoints,
                         const Intrepid2::EOperator op = Intrepid2::OPERATOR_VALUE ) const;
  
  virtual int getCardinality() const;
  virtual int getDegree() const;

  /** \brief  Returns the topology on which the basis is defined.
   */
  virtual CellTopoPtr domainTopology() const;

  // dof ordinal subsets:
  virtual std::set<int> dofOrdinalsForEdges(bool includeVertices = true) const;
  virtual std::set<int> dofOrdinalsForFaces(bool includeVerticesAndEdges = true) const;
  virtual std::set<int> dofOrdinalsForVertices() const;
  virtual std::set<int> dofOrdinalsForSubcells(int subcellDim, bool includeLesserDimensions) const;

  // ! includes lesser dimensions (e.g. vertices on the side), and is considered as a "continuous" basis;
  // ! i.e., if you have HGRAD_DISC or HVOL, you'll get the dofs that have support on the side, even though
  // ! continuity would not usually be enforced along the side for these.
  virtual std::set<int> dofOrdinalsForSide(int sideOrdinal) const;
  
  virtual const std::vector<int> &dofOrdinalsForInterior() const;
  
  virtual std::vector<int> dofOrdinalsForSubcell(int subcellDim, int subcellIndex, int minimumSubSubcellDimension) const; // e.g. can get vertex, edge, and face dofs for a side by specifying subcellDim=2, minimumSubSubcellDimension = 0
  virtual const std::vector<int> &dofOrdinalsForSubcell(int subcellDim, int subcellIndex) const;
  virtual const std::vector<int> &dofOrdinalsForVertex(int vertexIndex) const;
  virtual const std::vector<int> &dofOrdinalsForEdge(int edgeIndex) const;
  
//  virtual void dofOrdinalsForSubcell(int subcellDim, int subcellIndex, std::vector<int> &dofOrdinals) const;

  virtual int getDofOrdinal(const int subcDim, const int subcOrd, const int subcDofOrd) const;
  virtual const std::vector<std::vector<std::vector<int> > > &getDofOrdinalData( ) const;
  virtual const std::vector<int>& getDofTag(int dofOrd) const;
  virtual const std::vector<std::vector<int> > & getAllDofTags() const;

  /** \brief returns the number of tensorial components.
   */
  virtual int getNumTensorialComponents() const;
  
  virtual Camellia::EFunctionSpace functionSpace() const;
  virtual Camellia::EFunctionSpace functionSpace(int tensorialComponent) const; // 0 for space, 1 for time, in space-time bases
  virtual Camellia::EFunctionSpace functionSpace(int subcDim, int subcellOrdinal) const; // particularly relevant for space-time bases

  virtual Teuchos::RCP< Intrepid2::Basis<Camellia::DeviceType,Scalar,Scalar> > intrepid2Basis() const;
  
  virtual std::vector< Teuchos::RCP< ::Camellia::Basis<Scalar, ArrayScalar> > > intrepid2TensorBasisComponents();
  
  //! returns a (weak) pointer to this if the basis is already continuous.  If it belongs to a broken space, returns an unbroken basis (with the same cardinality and member bases).
  Teuchos::RCP< ::Camellia::Basis<Scalar, ArrayScalar> > continuousBasis();
  
  // methods identifying the type of basis:
  virtual bool isConforming() const; // defaults to false (true for the Intrepid bases)
  virtual bool isModal() const; // defaults to false (true for the Lobatto bases)
  virtual bool isNodal() const; // defaults to false (true for the Intrepid bases)

  // range info for basis values:
  virtual int rangeDimension() const;
  virtual int rangeRank() const;
  
  //! returns true if the basis accepts the specified operator (modulo Intrepid or Intrepid2 enumeration conversion) in getValues().  This does not mean that the basis will not throw an exception, etc., but rather that the operator specified is "known to" the basis.  Some of the extended Camellia operators are not known to Intrepid/Intrepid2 bases.
  virtual bool supportsOperator(::Camellia::EOperator &op) const;

  virtual void getValues(ArrayScalar &values, const ArrayScalar &refPoints, ::Camellia::EOperator operatorType) const = 0;

  virtual void CHECK_VALUES_ARGUMENTS(const ArrayScalar &values, const ArrayScalar &refPoints, ::Camellia::EOperator operatorType) const;

  //! Fills the provided array with (F,D)-shaped nodal coordinates corresponding to the basis functions.
  //! May throw an exception if the basis is not nodal.
  virtual void getDofCoords(ArrayScalar & DofCoords) const;
  
  virtual ~Basis() {}
};

/***************** IntrepidBasisWrapper ******************/
template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class IntrepidBasisWrapper;
template<class Scalar, class ArrayScalar> class IntrepidBasisWrapper : public Basis<Scalar,ArrayScalar>
{
private:
  Teuchos::RCP< Intrepid::Basis<Scalar,ArrayScalar> > _intrepidBasis;
protected:
  void initializeTags() const;
public:
  IntrepidBasisWrapper(Teuchos::RCP< Intrepid::Basis<Scalar,ArrayScalar> > intrepidBasis, int rangeDimension, int rangeRank,
                       Camellia::EFunctionSpace functionSpace);

  Teuchos::RCP< Intrepid::Basis<Scalar,ArrayScalar> > intrepidBasis();

  virtual bool isConforming() const; // true for Intrepid bases
  virtual bool isNodal() const;      // true for the Intrepid bases

  virtual std::set<int> dofOrdinalsForSide(int sideOrdinal) const;
  
  using Basis<Scalar,ArrayScalar>::getValues;
  void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Camellia::EOperator operatorType) const;
  
  void getDofCoords(ArrayScalar & DofCoords) const;
};

template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class Intrepid2BasisWrapper;
template<class Scalar, class ArrayScalar> class Intrepid2BasisWrapper : public Basis<Scalar,ArrayScalar>
{
  using DeviceType = Camellia::DeviceType;
private:
  Teuchos::RCP< Intrepid2::Basis<DeviceType,Scalar,Scalar> > _intrepid2Basis;
protected:
  void initializeTags() const override;
public:
  Intrepid2BasisWrapper(Teuchos::RCP< Intrepid2::Basis<Camellia::DeviceType,Scalar,Scalar> > intrepid2Basis,
                        int rangeDimension, int rangeRank, Camellia::EFunctionSpace functionSpace);

  Teuchos::RCP< Intrepid2::Basis<Camellia::DeviceType,Scalar,Scalar> > intrepid2Basis() const override;

  virtual bool isConforming() const override;
  virtual bool isNodal() const override;

  virtual std::set<int> dofOrdinalsForSide(int sideOrdinal) const override;
  
  void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Camellia::EOperator operatorType) const override;
  
  //! allocation method for Intrepid2-style getValues() output argument.
  Intrepid2::BasisValues<Scalar,DeviceType> allocateBasisValues( Intrepid2::TensorPoints<Scalar,DeviceType> points,
                                                                const Intrepid2::EOperator op = Intrepid2::OPERATOR_VALUE) const override;
  
  //! Intrepid2-style getValues() method.  This is part of the API that will be performance-optimized.  (Allows use of Intrepid2's sum factorization facilities, etc.)
  void getValues( Intrepid2::BasisValues<Scalar,DeviceType> outputValues,
                  const Intrepid2::TensorPoints<Scalar,DeviceType>  inputPoints,
                  const Intrepid2::EOperator op = Intrepid2::OPERATOR_VALUE ) const override;
  
  void getDofCoords(ArrayScalar & DofCoords) const override;
  
  static Intrepid2::EOperator intrepidOpToIntrepid2Op(const Intrepid::EOperator &op)
  {
    switch (op)
    {
      case Intrepid::OPERATOR_VALUE: return Intrepid2::OPERATOR_VALUE;
      case Intrepid::OPERATOR_GRAD:  return Intrepid2::OPERATOR_GRAD;
      case Intrepid::OPERATOR_CURL:  return Intrepid2::OPERATOR_CURL;
      case Intrepid::OPERATOR_DIV:   return Intrepid2::OPERATOR_DIV;
      case Intrepid::OPERATOR_D1:    return Intrepid2::OPERATOR_D1;
      case Intrepid::OPERATOR_D2:    return Intrepid2::OPERATOR_D2;
      case Intrepid::OPERATOR_D3:    return Intrepid2::OPERATOR_D3;
      case Intrepid::OPERATOR_D4:    return Intrepid2::OPERATOR_D4;
      case Intrepid::OPERATOR_D5:    return Intrepid2::OPERATOR_D5;
      case Intrepid::OPERATOR_D6:    return Intrepid2::OPERATOR_D6;
      case Intrepid::OPERATOR_D7:    return Intrepid2::OPERATOR_D7;
      case Intrepid::OPERATOR_D8:    return Intrepid2::OPERATOR_D8;
      case Intrepid::OPERATOR_D9:    return Intrepid2::OPERATOR_D9;
      case Intrepid::OPERATOR_D10:   return Intrepid2::OPERATOR_D10;
      case Intrepid::OPERATOR_MAX:   return Intrepid2::OPERATOR_MAX;
      default:                       return Intrepid2::OPERATOR_MAX;
    }
  }
};
} // namespace Camellia

#include "BasisDef.h"

#endif
