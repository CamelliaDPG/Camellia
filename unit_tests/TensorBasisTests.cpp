//
//  BasisTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/10/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Basis.h"
#include "Intrepid_HGRAD_LINE_Cn_FEM.hpp"

#include "Intrepid_HGRAD_QUAD_Cn_FEM.hpp"
#include "Intrepid_HDIV_QUAD_In_FEM.hpp"

#include "Intrepid_HGRAD_HEX_Cn_FEM.hpp"
#include "Intrepid_HDIV_HEX_In_FEM.hpp"

#include "doubleBasisConstruction.h"
#include "CamelliaCellTools.h"

#include "Intrepid_FieldContainer.hpp"

#include "Basis.h"

#include "TensorBasis.h"

#include "BasisFactory.h"

namespace {
  TEUCHOS_UNIT_TEST( TensorBasis, TensorBasisBasicValues ) {
    int spatialPolyOrder = 1;
    
    BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
    
    // set up some spatial bases to test against
    std::vector< BasisPtr > spatialBases;
    {
      int H1Order = spatialPolyOrder + 1;
      BasisPtr basis = basisFactory->getBasis(H1Order, shards::Line<2>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Line<2>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HVOL);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Quadrilateral<4>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Quadrilateral<4>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HCURL);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Quadrilateral<4>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HDIV);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Quadrilateral<4>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HVOL);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Hexahedron<8>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Hexahedron<8>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HCURL);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Hexahedron<8>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HDIV);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Hexahedron<8>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HVOL);
      spatialBases.push_back(basis);
    }
    
    int timePolyOrder = 2;
    BasisPtr timeBasis = basisFactory->getBasis(timePolyOrder + 1, shards::Line<2>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HVOL);
    
    typedef Intrepid::FieldContainer<double> FC;
    int numSpacePoints = 3, numTimePoints = 3;
    FC spatialPoints1D(numSpacePoints,1), spatialPoints2D(numSpacePoints,2), spatialPoints3D(numSpacePoints,3);
    
    spatialPoints1D(0,0) = -0.05;
    spatialPoints1D(1,0) = 0.33;
    spatialPoints1D(2,0) = 1.00;
    
    spatialPoints2D(0,0) = -1.0;
    spatialPoints2D(0,1) = 0.0;
    spatialPoints2D(1,0) = 0.5;
    spatialPoints2D(1,1) = -0.33;
    spatialPoints2D(2,0) = 1.0;
    spatialPoints2D(2,1) = 1.0;
    
    spatialPoints3D(0,0) = -1.0;
    spatialPoints3D(0,1) = 0.0;
    spatialPoints3D(0,2) = 0.0;
    spatialPoints3D(1,0) = 0.5;
    spatialPoints3D(1,1) = -0.33;
    spatialPoints3D(1,2) = 1.0;
    spatialPoints3D(2,0) = 1.0;
    spatialPoints3D(2,1) = 1.0;
    spatialPoints3D(2,2) = 1.0;
    
    vector< FC > spacePointsForDimension(3), tensorPointsForDimension(3);
    spacePointsForDimension[0] = spatialPoints1D;
    spacePointsForDimension[1] = spatialPoints2D;
    spacePointsForDimension[2] = spatialPoints3D;
    
    FC temporalPoints(numTimePoints, 1);
    temporalPoints(0,0) = -0.5; // these are in reference space; we don't actually have negative time values
    temporalPoints(1,0) = 0.33;
    temporalPoints(2,0) = 1.0;
    
    int numTensorPoints = numSpacePoints * numTimePoints;
    for (int spaceDim=1; spaceDim<=3; spaceDim++) {
      FC spatialPoints = spacePointsForDimension[spaceDim - 1];
      FC tensorPoints = FC(numTensorPoints, spaceDim + 1);
      for (int i=0; i<numSpacePoints; i++) {
        FC spaceTimePoint(spaceDim + 1);
        for (int d=0; d<spaceDim; d++) {
          spaceTimePoint(d) = spatialPoints(i,d);
        }
        for (int j=0; j<numTimePoints; j++) {
          spaceTimePoint(spaceDim) = temporalPoints(j,0);
          int pointOrdinal = i * numTimePoints + j;
          for (int d=0; d<spaceDim+1; d++) {
            tensorPoints(pointOrdinal,d) = spaceTimePoint(d);
          }
        }
      }
      tensorPointsForDimension[spaceDim-1] = tensorPoints;
    }
    
    std::map<EOperator, int> rankAdjustmentForOperator;
    
    rankAdjustmentForOperator[OPERATOR_VALUE] = 0;
    rankAdjustmentForOperator[OPERATOR_GRAD] = 1;
    rankAdjustmentForOperator[OPERATOR_DIV] = -1;
    rankAdjustmentForOperator[OPERATOR_CURL] = 0; // in 2D, this toggles between +1 and -1, depending on the current rank (scalar --> vector, vector --> scalar)
    
    for (int i=0; i<spatialBases.size(); i++) {
      BasisPtr spatialBasis = spatialBases[i];
      typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;
      Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(spatialBasis, timeBasis) );
      FC spatialValues, temporalValues(timeBasis->getCardinality(), numTimePoints), tensorValues;
      EOperator op = OPERATOR_VALUE;
      EOperator timeOp = OPERATOR_VALUE;
      
      int spaceDim = spatialBasis->domainTopology()->getDimension();
      
      int tensorCardinality = spatialBasis->getCardinality() * timeBasis->getCardinality();
      
      int rank = spatialBasis->rangeRank() + rankAdjustmentForOperator[op];
      if ((spatialBasis->rangeDimension() == 2) && (op==OPERATOR_CURL)) {
        if (spaceDim == 0) rank += 1;
        if (spaceDim == 1) rank -= 1;
      }
      
      FC spatialPoints = spacePointsForDimension[spaceDim-1];
      FC tensorPoints = tensorPointsForDimension[spaceDim-1];
      
      if (rank == 0) { // scalar
        spatialValues.resize(spatialBasis->getCardinality(), numSpacePoints);
        tensorValues.resize(tensorCardinality, numTensorPoints);
      } else if (rank == 1) { // vector
        spatialValues.resize(spatialBasis->getCardinality(), numSpacePoints, spaceDim);
        tensorValues.resize(tensorCardinality, numTensorPoints, spaceDim);
      } else if (rank == 2) { // tensor
        spatialValues.resize(spatialBasis->getCardinality(), numSpacePoints, spaceDim, spaceDim);
        tensorValues.resize(tensorCardinality, numTensorPoints, spaceDim, spaceDim);
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled tensorial degree");
      }
      spatialBasis->getValues(spatialValues, spatialPoints, op);
      timeBasis->getValues(temporalValues, temporalPoints, timeOp);
      
      tensorBasis->getValues(tensorValues, tensorPoints, op, timeOp);
      
      vector<int> fieldCoord(2);
      for (int fieldOrdinal_i = 0; fieldOrdinal_i < spatialBasis->getCardinality(); fieldOrdinal_i++) {
        fieldCoord[0] = fieldOrdinal_i;
        for (int fieldOrdinal_j = 0; fieldOrdinal_j < timeBasis->getCardinality(); fieldOrdinal_j++) {
          fieldCoord[1] = fieldOrdinal_j;
          int fieldOrdinal_tensor = tensorBasis->getDofOrdinalFromComponentDofOrdinals(fieldCoord);
          for (int pointOrdinal_i = 0; pointOrdinal_i < numSpacePoints; pointOrdinal_i++) {
            vector<double> spaceValue;
            if (rank == 0) {
              spaceValue.push_back(spatialValues(fieldOrdinal_i, pointOrdinal_i));
            } else if (rank == 1) {
              for (int d=0; d<spaceDim; d++) {
                spaceValue.push_back(spatialValues(fieldOrdinal_i, pointOrdinal_i, d));
              }
            } else if (rank == 2) {
              for (int d1=0; d1<spaceDim; d1++) {
                for (int d2=0; d2<spaceDim; d2++) {
                  spaceValue.push_back(spatialValues(fieldOrdinal_i, pointOrdinal_i, d1, d2));
                }
              }
            }
            for (int pointOrdinal_j = 0; pointOrdinal_j < numTimePoints; pointOrdinal_j++) {
              double timeValue = temporalValues(pointOrdinal_j, 0);
              int pointOrdinal_tensor = pointOrdinal_i * numTimePoints + pointOrdinal_j;
              vector<double> tensorValue;
              if (rank == 0) {
                tensorValue.push_back(tensorValues(fieldOrdinal_tensor, pointOrdinal_tensor));
              } else if (rank == 1) {
                for (int d=0; d<spaceDim; d++) {
                  tensorValue.push_back(tensorValues(fieldOrdinal_tensor, pointOrdinal_tensor, d));
                }
              } else if (rank == 2) {
                for (int d1=0; d1<spaceDim; d1++) {
                  for (int d2=0; d2<spaceDim; d2++) {
                    tensorValue.push_back(tensorValues(fieldOrdinal_tensor, pointOrdinal_tensor, d1, d2));
                  }
                }
              }
              
              if (spaceValue.size() != tensorValue.size()) {
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal test error: tensorValue size does not match spaceValue size");
              }
              double tol = 1e-15;
              for (int i=0; i<spaceValue.size(); i++) {
                double expectedValue = spaceValue[i] * timeValue;
                double actualValue = tensorValue[i];
                TEST_FLOATING_EQUALITY(expectedValue,actualValue,tol);
              }
            }
          }
        }
      }
      
      // TODO: test that the values are correct
    }
  }

} // namespace