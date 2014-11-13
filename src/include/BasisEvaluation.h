//
//  BasisEvaluation.h
//  DPGTrilinos
//
// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER 
//

#ifndef DPGTrilinos_BasisEvaluation_h
#define DPGTrilinos_BasisEvaluation_h

#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

#include "BilinearForm.h"
#include "VectorizedBasis.h"
#include "Basis.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "CamelliaIntrepidExtendedTypes.h"

class BasisEvaluation { 
  typedef Teuchos::RCP< FieldContainer<double> > FCPtr;
  typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;
  
public:
  static FCPtr getValues(BasisPtr basis, EOperatorExtended op,
                         const FieldContainer<double> &refPoints);
  static FCPtr getTransformedValues(BasisPtr basis, EOperatorExtended op,
                                   const FieldContainer<double> &refPoints,
                                   const FieldContainer<double> &cellJacobian, 
                                   const FieldContainer<double> &cellJacobianInv,
                                   const FieldContainer<double> &cellJacobianDet);
  static FCPtr getTransformedVectorValuesWithComponentBasisValues(Camellia::VectorBasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op,
                                                                  constFCPtr componentReferenceValuesTransformed);
  static FCPtr getTransformedValuesWithBasisValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op,
                                                   constFCPtr referenceValues,                                 
                                                   const FieldContainer<double> &cellJacobian, 
                                                   const FieldContainer<double> &cellJacobianInv,
                                                   const FieldContainer<double> &cellJacobianDet);
  static FCPtr getValuesCrossedWithNormals(constFCPtr values,const FieldContainer<double> &sideNormals);
  static FCPtr getValuesDottedWithNormals(constFCPtr values,const FieldContainer<double> &sideNormals);
  static FCPtr getValuesTimesNormals(constFCPtr values,const FieldContainer<double> &sideNormals);
  static FCPtr getValuesTimesNormals(constFCPtr values,const FieldContainer<double> &sideNormals, int normalComponent);
  static FCPtr getVectorizedValues(constFCPtr values, int spaceDim);
  static Intrepid::EOperator relatedOperator(EOperatorExtended op, IntrepidExtendedTypes::EFunctionSpaceExtended fs, int &componentOfInterest);
  static FCPtr getComponentOfInterest(constFCPtr values, IntrepidExtendedTypes::EOperatorExtended op, IntrepidExtendedTypes::EFunctionSpaceExtended fs, int componentOfInterest);
};

#endif
