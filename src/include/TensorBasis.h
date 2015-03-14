//
//  TensorBasis.h
//  Camellia
//
//  Created by Nathan Roberts on 11/12/14.
//
// @HEADER

/** \file   TensorBasis.h
 \brief  Header file for the Camellia::TensorBasis class.
 \author Created by Nathan V. Roberts.
 */

#ifndef __Camellia__TensorBasis__
#define __Camellia__TensorBasis__

#include "Basis.h"

typedef Intrepid::EOperator EOperator;

namespace Camellia {

  /** \class Camellia::TensorBasis
   \brief Defines a basis on a tensor-product topology in terms of Camellia::Basis objects defined on the
          tensorial components.  Initial implementation supports just two tensorial components, the idea
          being that the first belongs to a spatial topology, while the second is defined on a temporal
          topology (a line), though some of the method interfaces anticipate a more general tensor basis
          definition, which could allow for fast quadrature on hypercubes, e.g.
   */
  
template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class TensorBasis;
template<class Scalar, class ArrayScalar> class TensorBasis : public Camellia::Basis<Scalar,ArrayScalar> {
private:
  Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > _spatialBasis, _temporalBasis;
protected:
  void initializeTags() const;
public:
  /** \brief  Constructor
   
   Returns values of <var>operatorType</var> acting on FEM basis functions for a set of
   points in the <strong>reference cell</strong> for which the basis is defined.
   
   \param  spatialBasis      [in] - Basis for the spatial topology
   \param  temporalBasis     [in] - Basis for the temporal topology (a line)
   
   At present, we anticipate only scalar bases being used in the temporal dimension, but bases of arbitrary rank might be used for the spatial basis.
   
   */
  TensorBasis(Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > spatialBasis, Teuchos::RCP< Camellia::Basis<Scalar,ArrayScalar> > temporalBasis);
  /** \brief Destructor.
   */
  virtual ~TensorBasis() {}

  /** \brief Returns the cardinality of the basis, computed as the product of the cardinalities of the component bases.
   */
  int getCardinality() const;
  
  /** \brief Returns the spatial degree of the basis.
   */
  int getDegree() const;
  
  /** \brief  Given a vector of basis ordinal choices (one for each tensorial rank), returns the ordinal of the corresponding 
              basis function in the tensor basis.
   
   \param  componentDofOrdinals [in] - vector of selections of basis ordinals in each component.
   
   */
  int getDofOrdinalFromComponentDofOrdinals(std::vector<int> componentDofOrdinals) const;

  /** \brief  Given a vector of component value arrays (one for space, one for time), generates the tensor product value array.
   
   \param  tensorPoints  [out] - tensor product point array.  Ordered (P,D1+D2).
   \param  spatialPoints  [in] - spatial point array.  Should be ordered (P,D1).
   \param temporalPoints  [in] - spatial point array.  Should be ordered (P,D2).
   
   */
  void getTensorPoints(ArrayScalar& tensorPoints, const ArrayScalar & spatialPoints, const ArrayScalar & temporalPoints) const;
  
  /** \brief  Given a vector of component value arrays (one for space, one for time), generates the tensor product value array.
   
   \param  outputValues          [out] - tensor product value array.  Ordered (C,F,P,D,...) or (F,P,D,...).
   \param  componentOutputValues  [in] - values for each component.  Should be ordered (C,F,P,D,...) or (F,P,D,...).
   
   */
  void getTensorValues(ArrayScalar& outputValues, std::vector< ArrayScalar> & componentOutputValues,
                       std::vector<Intrepid::EOperator> operatorTypes) const;

  /** \brief  Returns the basis corresponding to the provided tensorial rank.
   
   \param  tensorialBasisRank     [in] - tensorial rank of the desired component basis.  0 for space, 1 for time.
   
   */
  const Teuchos::RCP< Camellia::Basis<Scalar, ArrayScalar> > getComponentBasis(int tensorialBasisRank) const;
  
  /** \brief  Returns the spatial basis.
   */
  const Teuchos::RCP< Camellia::Basis<Scalar, ArrayScalar> > getSpatialBasis() const;

  /** \brief  Returns the temporal basis.
   */
  const Teuchos::RCP< Camellia::Basis<Scalar, ArrayScalar> > getTemporalBasis() const;
  
  /** \brief  Returns the range dimension--i.e. 3 for vectors or tensors in 3D, 2 in 2D, etc.
              For scalar-valued bases, returns the dimension of the domain.
   */
  int rangeDimension() const;
  
  /** \brief  Tensorial rank of the basis values -- 0 for scalars, 1 for vectors, etc.
   */
  int rangeRank() const;

  /** \brief  Computes the values of the basis at the provided points.
   
   \param  values        [out] - the basis values, ordered as (F, P, D, ...)
   \param  refPoints      [in] - points at which to evaluate the basis, dimensions (P,D).
   \param  operatorType   [in] - the operator to use when evaluating the spatial basis.  (OPERATOR_VALUE assumed for the temporal basis)

   */
  void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const;
  
  /** \brief  Computes the values of the basis at the provided points.
   
   \param  values               [out] - the basis values, ordered as (F,P,D, ...)
   \param  refPoints             [in] - points at which to evaluate the basis, dimensions (P,D).
   \param  spatialOperatorType   [in] - the operator to use when evaluating the spatial basis.
   \param  temporalOperatorType  [in] - the operator to use when evaluating the temporal basis.
   
   */
  void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator spatialOperatorType, Intrepid::EOperator temporalOperatorType) const;
};

typedef Teuchos::RCP<TensorBasis<> > TensorBasisPtr;
  
} // namespace Camellia

#include "TensorBasisDef.h"

#endif /* defined(__Camellia__TensorBasis__) */
