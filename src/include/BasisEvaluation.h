// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

//
//  BasisEvaluation.h
//  DPGTrilinos
//

#ifndef DPGTrilinos_BasisEvaluation_h
#define DPGTrilinos_BasisEvaluation_h

#include "TypeDefs.h"

#include "Camellia_FieldContainer.hpp"

#include "Basis.h"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "CamelliaIntrepidExtendedTypes.h"

namespace Camellia
{
class BasisEvaluation
{
public:
  static FCPtr getValues(BasisPtr basis, Camellia::EOperator op,
                         const Camellia::FieldContainer<double> &refPoints);
  
  //! Intrepid2-style getValues()
  static Intrepid2::BasisValues<double,DeviceType> getValues(Intrepid2BasisWrapper<double, Camellia::FieldContainer<double,0> >* intrepid2Basis,
                                                             Intrepid2::EOperator op,
                                                             const Intrepid2::TensorPoints<double,DeviceType> &inputPoints);
  
  static FCPtr getTransformedValues(BasisPtr basis, Camellia::EOperator op,
                                    const Camellia::FieldContainer<double> &refPoints,
                                    int numCells,
                                    BasisCache* basisCache);
  
  // ! Deprecated.  Will be deleted once MultiBasis no longer depends on this
  static FCPtr getTransformedValues(BasisPtr basis, Camellia::EOperator op,
                                    const Camellia::FieldContainer<double> &refPoints,
                                    int numCells,
                                    const Camellia::FieldContainer<double> &cellJacobian,
                                    const Camellia::FieldContainer<double> &cellJacobianInv,
                                    const Camellia::FieldContainer<double> &cellJacobianDet);
  static FCPtr getTransformedVectorValuesWithComponentBasisValues(Camellia::VectorBasisPtr basis,
                                                                  Camellia::EOperator op,
                                                                  constFCPtr componentReferenceValuesTransformed);
  static FCPtr getTransformedValuesWithBasisValues(BasisPtr basis, Camellia::EOperator op, int spaceDim,
                                                   constFCPtr referenceValues, int numCells,
                                                   const Camellia::FieldContainer<double> &cellJacobian,
                                                   const Camellia::FieldContainer<double> &cellJacobianInv,
                                                   const Camellia::FieldContainer<double> &cellJacobianDet);
  static FCPtr getTransformedValuesWithBasisValues(BasisPtr basis, Camellia::EOperator op,
                                                   constFCPtr referenceValues, int numCells,
                                                   BasisCache* basisCache);
  static FCPtr getValuesCrossedWithNormals(constFCPtr values,const Camellia::FieldContainer<double> &sideNormals);
  static FCPtr getValuesDottedWithNormals(constFCPtr values,const Camellia::FieldContainer<double> &sideNormals);
  static FCPtr getValuesTimesNormals(constFCPtr values,const Camellia::FieldContainer<double> &sideNormals);
  static FCPtr getValuesTimesNormals(constFCPtr values,const Camellia::FieldContainer<double> &sideNormals, int normalComponent);
  static FCPtr getVectorizedValues(constFCPtr values, int spaceDim);
  
  //! Returns a related operator that the provided basis can evaluate directly (will be the same operator if the basis supports that).
  static Camellia::EOperator relatedOperator(BasisPtr basis, Camellia::EOperator op, int &componentOfInterest);
  static FCPtr getComponentOfInterest(constFCPtr values, Camellia::EOperator op, Camellia::EFunctionSpace fs, int componentOfInterest);
};
}

#endif
