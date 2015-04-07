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
#include "CellTopology.h"

using namespace Camellia;
using namespace Intrepid;

namespace {
  typedef Intrepid::FieldContainer<double> FC;
  
  std::vector<BasisPtr> getSpatialBases(int spatialPolyOrder) {
    BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
    
    // set up some spatial bases to test against
    std::vector< BasisPtr > spatialBases;
    {
      int H1Order = spatialPolyOrder + 1;
      BasisPtr basis = basisFactory->getBasis(H1Order, shards::Line<2>::key, Camellia::FUNCTION_SPACE_HVOL);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Line<2>::key, Camellia::FUNCTION_SPACE_HGRAD);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Quadrilateral<4>::key, Camellia::FUNCTION_SPACE_HGRAD);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Quadrilateral<4>::key, Camellia::FUNCTION_SPACE_HCURL);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Quadrilateral<4>::key, Camellia::FUNCTION_SPACE_HDIV);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Quadrilateral<4>::key, Camellia::FUNCTION_SPACE_HVOL);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Hexahedron<8>::key, Camellia::FUNCTION_SPACE_HGRAD);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Hexahedron<8>::key, Camellia::FUNCTION_SPACE_HCURL);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Hexahedron<8>::key, Camellia::FUNCTION_SPACE_HDIV);
      spatialBases.push_back(basis);
      basis = basisFactory->getBasis(H1Order, shards::Hexahedron<8>::key, Camellia::FUNCTION_SPACE_HVOL);
      spatialBases.push_back(basis);
    }
    return spatialBases;
  }
  
  BasisPtr getTimeBasis(int timePolyOrder) {
    BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
    BasisPtr timeBasis = basisFactory->getBasis(timePolyOrder + 1, shards::Line<2>::key, Camellia::FUNCTION_SPACE_HVOL);
    return timeBasis;
  }
  
  FC spacePointsForDimension(int spaceDim) {
    int numSpacePoints = 3;
    FC spatialPoints(numSpacePoints,spaceDim);
    
    switch (spaceDim) {
      case 1:
        spatialPoints(0,0) = -0.05;
        spatialPoints(1,0) = 0.33;
        spatialPoints(2,0) = 1.00;
        break;
      case 2:
        spatialPoints(0,0) = -1.0;
        spatialPoints(0,1) = 0.0;
        spatialPoints(1,0) = 0.5;
        spatialPoints(1,1) = -0.33;
        spatialPoints(2,0) = 1.0;
        spatialPoints(2,1) = 1.0;
        break;
      case 3:
        spatialPoints(0,0) = -1.0;
        spatialPoints(0,1) = 0.0;
        spatialPoints(0,2) = 0.0;
        spatialPoints(1,0) = 0.5;
        spatialPoints(1,1) = -0.33;
        spatialPoints(1,2) = 1.0;
        spatialPoints(2,0) = 1.0;
        spatialPoints(2,1) = 1.0;
        spatialPoints(2,2) = 1.0;
        break;
      default:
        break;
    }
    return spatialPoints;
   }
  
  FC timePoints() {
    int numTimePoints = 3;
    
    FC temporalPoints(numTimePoints, 1);
    temporalPoints(0,0) = -0.5; // these are in reference space; we don't actually have negative time values
    temporalPoints(1,0) = 0.33; // 0.33;
    temporalPoints(2,0) = 1.0; // 1.0;
    return temporalPoints;
  }
  
  FC tensorPointsForDimension(int spaceDim) {
    FC spatialPoints = spacePointsForDimension(spaceDim);
    FC temporalPoints = timePoints();
    
    int numSpacePoints = spatialPoints.dimension(0);
    int numTimePoints = temporalPoints.dimension(0);
    
    int numTensorPoints = numSpacePoints * numTimePoints;
    FC tensorPoints = FC(numTensorPoints, spaceDim + 1);
    for (int i=0; i<numSpacePoints; i++) {
      FC spaceTimePoint(spaceDim + 1);
      for (int d=0; d<spaceDim; d++) {
        spaceTimePoint(d) = spatialPoints(i,d);
      }
      for (int j=0; j<numTimePoints; j++) {
        spaceTimePoint(spaceDim) = temporalPoints(j,0);
        int pointOrdinal = i + j * numSpacePoints;
        for (int d=0; d<spaceDim+1; d++) {
          tensorPoints(pointOrdinal,d) = spaceTimePoint(d);
        }
      }
    }
    return tensorPoints;
  }
  
  void sizeFCForBasisValues(FC &values, BasisPtr basis, int numPoints, Intrepid::EOperator op) {
    std::map<Intrepid::EOperator, int> rankAdjustmentForOperator;
    
    rankAdjustmentForOperator[OPERATOR_VALUE] = 0;
    rankAdjustmentForOperator[OPERATOR_GRAD] = 1;
    rankAdjustmentForOperator[OPERATOR_DIV] = -1;
    rankAdjustmentForOperator[OPERATOR_CURL] = 0; // in 2D, this toggles between +1 and -1, depending on the current rank (scalar --> vector, vector --> scalar)
    
    int spaceDim = basis->domainTopology()->getDimension();
    
    int rank = basis->rangeRank() + rankAdjustmentForOperator[op];
    if ((basis->rangeDimension() == 2) && (op==OPERATOR_CURL)) {
      if (basis->rangeRank() == 0) rank += 1;
      if (basis->rangeRank() == 1) rank -= 1;
    }
      
    if (rank == 0) { // scalar
      values.resize(basis->getCardinality(), numPoints);
    } else if (rank == 1) { // vector
      values.resize(basis->getCardinality(), numPoints, spaceDim);
    } else if (rank == 2) { // tensor
      values.resize(basis->getCardinality(), numPoints, spaceDim, spaceDim);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled tensorial degree");
    }
  }

  TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell ) {
    int timeH1Order = 2, spaceH1Order = 2;
    
    CellTopoPtr line = CellTopology::line();
    
    BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
    BasisPtr lineBasis = basisFactory->getBasis(timeH1Order, line, Camellia::FUNCTION_SPACE_HGRAD);
    
    CellTopoPtr spaceTopo = CellTopology::line();
    BasisPtr spaceBasis = basisFactory->getBasis(spaceH1Order, spaceTopo, Camellia::FUNCTION_SPACE_HGRAD);
    
    typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;
    Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(spaceBasis, lineBasis) );
    
    // sanity check: tensorBasis should have cardinality = spaceBasis->getCardinality() * lineBasis->getCardinality()
    TEST_EQUALITY(tensorBasis->getCardinality(), spaceBasis->getCardinality() * lineBasis->getCardinality());
    
    // if we get all dofOrdinals for all subcells, that should cover the whole tensor basis
    int allSubcellDofOrdinalsCount = tensorBasis->dofOrdinalsForSubcells(spaceTopo->getDimension() + line->getDimension(), true).size();
    TEST_EQUALITY(tensorBasis->getCardinality(), allSubcellDofOrdinalsCount);
    
    CellTopoPtr spaceTimeTopo = tensorBasis->domainTopology();
    
    map<unsigned, pair<unsigned,unsigned> > basisOrdinalMap; // maps spaceTime --> (space, time) basis ordinals for nodes
    // check nodes
    int vertexDim = 0;
    int subcellDofOrdinal = 0;
    for (unsigned spaceNode=0; spaceNode < spaceTopo->getNodeCount(); spaceNode++) {
      unsigned spaceBasisOrdinal = spaceBasis->getDofOrdinal(vertexDim, spaceNode, subcellDofOrdinal);
      for (unsigned timeNode=0; timeNode < line->getNodeCount(); timeNode++) {
        unsigned timeBasisOrdinal = lineBasis->getDofOrdinal(vertexDim, timeNode, subcellDofOrdinal);
        vector<unsigned> componentNodes(2);
        componentNodes[0] = spaceNode;
        componentNodes[1] = timeNode;
        unsigned spaceTimeNode = spaceTimeTopo->getNodeFromTensorialComponentNodes(componentNodes);
        unsigned spaceTimeBasisOrdinal = tensorBasis->getDofOrdinal(vertexDim, spaceTimeNode, subcellDofOrdinal);
        
        if ((spaceBasisOrdinal == (unsigned) -1) || (timeBasisOrdinal == (unsigned)-1)) {
          // expect that spaceTimeBasisOrdinal is -1
          TEST_EQUALITY((unsigned)-1, spaceTimeBasisOrdinal);
        } else {
          // otherwise, expect it's not
          TEST_INEQUALITY((unsigned)-1, spaceTimeBasisOrdinal);
        }
        
        basisOrdinalMap[spaceTimeBasisOrdinal] = make_pair(spaceBasisOrdinal, timeBasisOrdinal);
      }
    }
    
    // test that the basisOrdinalMap entries are unique
    int spaceNodeOrdinalCount = spaceBasis->dofOrdinalsForVertices().size();
    int timeNodeOrdinalCount = lineBasis->dofOrdinalsForVertices().size();
    
    TEST_EQUALITY(basisOrdinalMap.size(), spaceNodeOrdinalCount * timeNodeOrdinalCount);
  }
  
  TEUCHOS_UNIT_TEST( TensorBasis, GetTensorValues ) {
    int spatialPolyOrder = 1;
    std::vector< BasisPtr > spatialBases = getSpatialBases(spatialPolyOrder);
    
    int timePolyOrder = 1;
    BasisPtr timeBasis = getTimeBasis(timePolyOrder);
    
    double tol = 1e-15;
    for (int i=0; i<spatialBases.size(); i++) {
      BasisPtr spatialBasis = spatialBases[i];
      int spaceDim = spatialBasis->domainTopology()->getDimension();
      
      FC temporalPoints = timePoints();
      FC spatialPoints = spacePointsForDimension(spaceDim);
      FC tensorPoints = tensorPointsForDimension(spaceDim);
      
      int numSpacePoints = spatialPoints.dimension(0);
      int numTimePoints = temporalPoints.dimension(0);
      int numTensorPoints = tensorPoints.dimension(0);
      
      typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;
      Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(spatialBasis, timeBasis) );
      FC spatialValues, temporalValues(timeBasis->getCardinality(), numTimePoints), tensorValuesExpected, tensorValuesActual;
      Intrepid::EOperator op = OPERATOR_VALUE;
      Intrepid::EOperator timeOp = OPERATOR_VALUE;
      
      sizeFCForBasisValues(spatialValues, spatialBasis, numSpacePoints, op);
      sizeFCForBasisValues(tensorValuesExpected, tensorBasis, numTensorPoints, op);
    
      spatialBasis->getValues(spatialValues, spatialPoints, op);
      timeBasis->getValues(temporalValues, temporalPoints, timeOp);
      tensorBasis->getValues(tensorValuesExpected, tensorPoints, op, timeOp);
      
      out << "spatialValues:\n" << spatialValues;
      out << "temporalValues:\n" << temporalValues;
      
      sizeFCForBasisValues(tensorValuesActual, tensorBasis, numTensorPoints, op);
      vector<FC> componentValues(2);
      componentValues[0] = spatialValues;
      componentValues[1] = temporalValues;
      
      vector<Intrepid::EOperator> componentOps(2);
      componentOps[0] = op;
      componentOps[1] = timeOp;
    
      tensorBasis->getTensorValues(tensorValuesActual, componentValues, componentOps);
      
      TEST_COMPARE_FLOATING_ARRAYS(tensorValuesActual, tensorValuesExpected, tol);
    }
  }
  
  TEUCHOS_UNIT_TEST( TensorBasis, TensorBasisFieldOrderingAgreesWithTensorTopologyOrdering ) {
    int H1Order = 1;
    
    BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
    CellTopoPtr line = CellTopology::line();
    
    BasisPtr basis = basisFactory->getBasis(H1Order, line, Camellia::FUNCTION_SPACE_HGRAD);
    
    typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;
    Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(basis, basis) );
    
    int tensorialDegree = 1;
    CellTopoPtr tensorTopo = CellTopology::cellTopology(line->getShardsTopology(), tensorialDegree);
    
    int numSpatialNodes = line->getNodeCount();
    int numTimeNodes = line->getNodeCount();
    vector<unsigned> compNodes(2);
    vector<int> compBasisOrdinals(2);
    for (int spatialNode=0; spatialNode<numSpatialNodes; spatialNode++) {
      compNodes[0] = spatialNode;
      compBasisOrdinals[0] = spatialNode;
      for (int timeNode=0; timeNode<numTimeNodes; timeNode++) {
        compNodes[1] = timeNode;
        compBasisOrdinals[1] = timeNode;
        unsigned topoOrdinal = tensorTopo->getNodeFromTensorialComponentNodes(compNodes);
        unsigned basisOrdinal = tensorBasis->getDofOrdinalFromComponentDofOrdinals(compBasisOrdinals);
        TEST_EQUALITY(topoOrdinal, basisOrdinal);
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( TensorBasis, TensorBasisPointOrderingAgreesWithTensorTopologyOrdering ) {
    // check that the way points are ordered in tensor basis is the same as the way they're ordered in
    // CellTopology
    int H1Order = 1;
    
    BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
    CellTopoPtr timeTopo = CellTopology::line();
    CellTopoPtr spaceTopo = CellTopology::triangle();
    
    BasisPtr spaceBasis = basisFactory->getBasis(H1Order, spaceTopo, Camellia::FUNCTION_SPACE_HGRAD);
    BasisPtr timeBasis = basisFactory->getBasis(H1Order, timeTopo, Camellia::FUNCTION_SPACE_HGRAD);
    
    typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;
    Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(spaceBasis, timeBasis) );
    
    int tensorialDegree = 1;
    CellTopoPtr tensorTopo = CellTopology::cellTopology(spaceTopo->getShardsTopology(), tensorialDegree);
    
    FC spaceNodes(spaceTopo->getNodeCount(), spaceTopo->getDimension());
    FC timeNodes(timeTopo->getNodeCount(), timeTopo->getDimension());
    
    CamelliaCellTools::refCellNodesForTopology(spaceNodes, spaceTopo);
    CamelliaCellTools::refCellNodesForTopology(timeNodes, timeTopo);
    
    vector<FC> tensorComponentNodes;
    tensorComponentNodes.push_back(spaceNodes);
    tensorComponentNodes.push_back(timeNodes);
    
    FC tensorNodesExpected(spaceNodes.dimension(0) * timeNodes.dimension(0), spaceNodes.dimension(1) + timeNodes.dimension(1));
    tensorTopo->initializeNodes(tensorComponentNodes, tensorNodesExpected);
    
    FC tensorNodesActual(spaceNodes.dimension(0) * timeNodes.dimension(0), spaceNodes.dimension(1) + timeNodes.dimension(1));
    tensorBasis->getTensorPoints(tensorNodesActual, spaceNodes, timeNodes);
    
    double tol = 1e-15;
    TEST_COMPARE_FLOATING_ARRAYS(tensorNodesActual, tensorNodesExpected, tol);
  }
  
  TEUCHOS_UNIT_TEST( TensorBasis, TensorBasisBasicValues ) {
    int spatialPolyOrder = 1;
    std::vector< BasisPtr > spatialBases = getSpatialBases(spatialPolyOrder);
    
    int timePolyOrder = 1;
    BasisPtr timeBasis = getTimeBasis(timePolyOrder);
    
    for (int i=0; i<spatialBases.size(); i++) {
      BasisPtr spatialBasis = spatialBases[i];
      int spaceDim = spatialBasis->domainTopology()->getDimension();

      FC temporalPoints = timePoints();
      FC spatialPoints = spacePointsForDimension(spaceDim);
      FC tensorPoints = tensorPointsForDimension(spaceDim);

      int numSpacePoints = spatialPoints.dimension(0);
      int numTimePoints = temporalPoints.dimension(0);
      int numTensorPoints = tensorPoints.dimension(0);
      
      typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;
      Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(spatialBasis, timeBasis) );
      FC spatialValues, temporalValues(timeBasis->getCardinality(), numTimePoints), tensorValues;
      Intrepid::EOperator op = OPERATOR_VALUE;
      Intrepid::EOperator timeOp = OPERATOR_VALUE;
      
      sizeFCForBasisValues(spatialValues, spatialBasis, numSpacePoints, op);
      sizeFCForBasisValues(tensorValues, tensorBasis, numTensorPoints, op);
      
      spatialBasis->getValues(spatialValues, spatialPoints, op);
      timeBasis->getValues(temporalValues, temporalPoints, timeOp);
      tensorBasis->getValues(tensorValues, tensorPoints, op, timeOp);
      
      int rank = tensorValues.rank() - 2; // (F,P,D,D,...) where # of D's is the rank.
      
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
              double timeValue = temporalValues(fieldOrdinal_j, pointOrdinal_j);
              int pointOrdinal_tensor = pointOrdinal_i + pointOrdinal_j * numSpacePoints;
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
    }
  }
  
} // namespace
