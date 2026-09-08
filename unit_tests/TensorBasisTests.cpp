//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  BasisTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/10/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Camellia_FieldContainer.hpp"

#include "doubleBasisConstruction.h"
#include "Basis.h"
#include "BasisCache.h"
#include "BasisFactory.h"
#include "CamelliaCellTools.h"
#include "CellTopology.h"
#include "TensorBasis.h"

// includes specific to Projection tests (may want to move to a different test group)
#include "MeshFactory.h"
#include "Solution.h"

using namespace Camellia;

namespace
{
typedef Camellia::FieldContainer<double> FC;

BasisPtr getSpatialBasis(CellTopoPtr cellTopo, Camellia::EFunctionSpace fs, int spatialPolyOrder)
{
  BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
  int H1Order = spatialPolyOrder + 1;
  BasisPtr basis = basisFactory->getBasis(H1Order, cellTopo, fs);
  return basis;
}

std::vector<BasisPtr> getSpatialBases(int spatialPolyOrder)
{
  BasisFactoryPtr basisFactory = BasisFactory::basisFactory();

  // set up some spatial bases to test against
  std::vector< BasisPtr > spatialBases;
  {
    BasisPtr basis = getSpatialBasis(CellTopology::line(), Camellia::FUNCTION_SPACE_HVOL, spatialPolyOrder);
    spatialBases.push_back(basis);
    basis = getSpatialBasis(CellTopology::line(), Camellia::FUNCTION_SPACE_HGRAD, spatialPolyOrder);
    spatialBases.push_back(basis);
    basis = getSpatialBasis(CellTopology::quad(), Camellia::FUNCTION_SPACE_HGRAD, spatialPolyOrder);
    spatialBases.push_back(basis);
    basis = getSpatialBasis(CellTopology::quad(), Camellia::FUNCTION_SPACE_HCURL, spatialPolyOrder);
    spatialBases.push_back(basis);
    basis = getSpatialBasis(CellTopology::quad(), Camellia::FUNCTION_SPACE_HDIV, spatialPolyOrder);
    spatialBases.push_back(basis);
    basis = getSpatialBasis(CellTopology::quad(), Camellia::FUNCTION_SPACE_HVOL, spatialPolyOrder);
    spatialBases.push_back(basis);
    basis = getSpatialBasis(CellTopology::hexahedron(), Camellia::FUNCTION_SPACE_HGRAD, spatialPolyOrder);
    spatialBases.push_back(basis);
    basis = getSpatialBasis(CellTopology::hexahedron(), Camellia::FUNCTION_SPACE_HCURL, spatialPolyOrder);
    spatialBases.push_back(basis);
    basis = getSpatialBasis(CellTopology::hexahedron(), Camellia::FUNCTION_SPACE_HDIV, spatialPolyOrder);
    spatialBases.push_back(basis);
    basis = getSpatialBasis(CellTopology::hexahedron(), Camellia::FUNCTION_SPACE_HVOL, spatialPolyOrder);
    spatialBases.push_back(basis);
  }
  return spatialBases;
}

BasisPtr getTimeBasis(int timePolyOrder)
{
  BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
  BasisPtr timeBasis = basisFactory->getBasis(timePolyOrder + 1, shards::Line<2>::key, Camellia::FUNCTION_SPACE_HVOL);
  return timeBasis;
}

FC spacePointsForDimension(int spaceDim)
{
  int numSpacePoints = 3;
  FC spatialPoints(numSpacePoints,spaceDim);

  switch (spaceDim)
  {
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

FC timePoints()
{
  int numTimePoints = 3;

  FC temporalPoints(numTimePoints, 1);
  temporalPoints(0,0) = -0.5; // these are in reference space; we don't actually have negative time values
  temporalPoints(1,0) = 0.33; // 0.33;
  temporalPoints(2,0) = 1.0; // 1.0;
  return temporalPoints;
}

FC tensorPointsForDimension(int spaceDim)
{
  FC spatialPoints = spacePointsForDimension(spaceDim);
  FC temporalPoints = timePoints();

  int numSpacePoints = spatialPoints.dimension(0);
  int numTimePoints = temporalPoints.dimension(0);

  int numTensorPoints = numSpacePoints * numTimePoints;
  FC tensorPoints = FC(numTensorPoints, spaceDim + 1);
  for (int i=0; i<numSpacePoints; i++)
  {
    FC spaceTimePoint(spaceDim + 1);
    for (int d=0; d<spaceDim; d++)
    {
      spaceTimePoint(d) = spatialPoints(i,d);
    }
    for (int j=0; j<numTimePoints; j++)
    {
      spaceTimePoint(spaceDim) = temporalPoints(j,0);
      int pointOrdinal = i + j * numSpacePoints;
      for (int d=0; d<spaceDim+1; d++)
      {
        tensorPoints(pointOrdinal,d) = spaceTimePoint(d);
      }
    }
  }
  return tensorPoints;
}

void sizeFCForBasisValues(FC &values, BasisPtr basis, int numPoints, ::Camellia::EOperator op)
{
  std::map<Camellia::EOperator, int> rankAdjustmentForOperator;

  rankAdjustmentForOperator[OP_VALUE] = 0;
  rankAdjustmentForOperator[OP_GRAD] = 1;
  rankAdjustmentForOperator[OP_DIV] = -1;
  rankAdjustmentForOperator[OP_CURL] = 0; // in 2D, this toggles between +1 and -1, depending on the current rank (scalar --> vector, vector --> scalar)

  int spaceDim = basis->rangeDimension();

  int rank = basis->rangeRank() + rankAdjustmentForOperator[op];
  if ((basis->rangeDimension() == 2) && (op==OP_CURL))
  {
    if (basis->rangeRank() == 0) rank += 1;
    if (basis->rangeRank() == 1) rank -= 1;
  }

  if (rank == 0)   // scalar
  {
    values.resize(basis->getCardinality(), numPoints);
  }
  else if (rank == 1)     // vector
  {
    values.resize(basis->getCardinality(), numPoints, spaceDim);
  }
  else if (rank == 2)     // tensor
  {
    values.resize(basis->getCardinality(), numPoints, spaceDim, spaceDim);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled tensorial degree");
  }
}

void testBasisOrdinalsForSubcell(CellTopoPtr spaceTopo, bool useHGRADInTime, int spaceH1Order, int timeH1Order,
                                 Teuchos::FancyOStream &out, bool &success)
{
  CellTopoPtr line = CellTopology::line();

  BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
  Camellia::EFunctionSpace timeFS = useHGRADInTime ? Camellia::FUNCTION_SPACE_HGRAD : Camellia::FUNCTION_SPACE_HVOL;
  BasisPtr lineBasis = basisFactory->getBasis(timeH1Order, line, timeFS);
  BasisPtr spaceBasis = basisFactory->getBasis(spaceH1Order, spaceTopo, Camellia::FUNCTION_SPACE_HGRAD);

  typedef Camellia::TensorBasis<double, Camellia::FieldContainer<double> > TensorBasis;
  Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(spaceBasis, lineBasis) );

  // sanity check: tensorBasis should have cardinality = spaceBasis->getCardinality() * lineBasis->getCardinality()
  TEST_EQUALITY(tensorBasis->getCardinality(), spaceBasis->getCardinality() * lineBasis->getCardinality());

  // if we get all dofOrdinals for all subcells, that should cover the whole tensor basis
  int allSubcellDofOrdinalsCount = tensorBasis->dofOrdinalsForSubcells(spaceTopo->getDimension() + line->getDimension(), true).size();
  TEST_EQUALITY(tensorBasis->getCardinality(), allSubcellDofOrdinalsCount);

  CellTopoPtr spaceTimeTopo = tensorBasis->domainTopology();

  int spaceNodeOrdinalCount = spaceBasis->dofOrdinalsForVertices().size();
  int timeNodeOrdinalCount = lineBasis->dofOrdinalsForVertices().size();

  /* ******** NODE TESTS ******** */
  if ((timeNodeOrdinalCount != 0) && (spaceNodeOrdinalCount != 0))
  {
    map<unsigned, pair<unsigned,unsigned> > basisOrdinalMap; // maps spaceTime --> (space, time) basis ordinals for nodes
    // check nodes
    int vertexDim = 0;
    int subcellDofOrdinal = 0;

    for (unsigned spaceNode=0; spaceNode < spaceTopo->getNodeCount(); spaceNode++)
    {
      unsigned spaceBasisOrdinal = spaceBasis->getDofOrdinal(vertexDim, spaceNode, subcellDofOrdinal);
      for (unsigned timeNode=0; timeNode < line->getNodeCount(); timeNode++)
      {
        unsigned timeBasisOrdinal = lineBasis->getDofOrdinal(vertexDim, timeNode, subcellDofOrdinal);
        vector<unsigned> componentNodes(2);
        componentNodes[0] = spaceNode;
        componentNodes[1] = timeNode;
        unsigned spaceTimeNode = spaceTimeTopo->getNodeFromTensorialComponentNodes(componentNodes);
        unsigned spaceTimeBasisOrdinal = tensorBasis->getDofOrdinal(vertexDim, spaceTimeNode, subcellDofOrdinal);

        if ((spaceBasisOrdinal == (unsigned) -1) || (timeBasisOrdinal == (unsigned)-1))
        {
          // expect that spaceTimeBasisOrdinal is -1
          TEST_EQUALITY((unsigned)-1, spaceTimeBasisOrdinal);
        }
        else
        {
          // otherwise, expect it's not
          TEST_INEQUALITY((unsigned)-1, spaceTimeBasisOrdinal);
        }

        basisOrdinalMap[spaceTimeBasisOrdinal] = make_pair(spaceBasisOrdinal, timeBasisOrdinal);
      }
    }

    // check that the basisOrdinalMap entries are unique
    TEST_EQUALITY(int(basisOrdinalMap.size()), spaceNodeOrdinalCount * timeNodeOrdinalCount);
  }
  else // if either timeNodeOrdinalCount = 0 or spaceNodeOrdinalCount = 0, expect no vertex ordinals for tensorBasis
  {
    TEST_EQUALITY(0, tensorBasis->dofOrdinalsForVertices().size());
  }

  {
    /* ********* TEST SUPPORT ********* */
    // If a basis claims that a dof ordinal belongs to a subcell, then we expect it to have support on that subcell.
    // If it has support on the subcell, then there should be at least one nonzero point among the cubature points
    // on the subcell.
    int cubDegree = max(timeH1Order, spaceH1Order);

    double tol = 1e-15;
    for (unsigned subcellDim = 0; subcellDim < spaceTimeTopo->getDimension(); subcellDim++)
    {
      int subcellCount = spaceTimeTopo->getSubcellCount(subcellDim);
      for (int subcellOrdinal=0; subcellOrdinal<subcellCount; subcellOrdinal++)
      {
        CellTopoPtr subcellTopo = spaceTimeTopo->getSubcell(subcellDim, subcellOrdinal);
        // lazy way to get the cubature points for subcell:
        BasisCachePtr subcellCache = BasisCache::basisCacheForReferenceCell(subcellTopo, cubDegree);
        Camellia::FieldContainer<double> subcellCubPoints = subcellCache->getRefCellPointsFC();

        int numPoints = subcellCubPoints.dimension(0);
        Camellia::FieldContainer<double> subcellPointsInParent(numPoints,spaceTimeTopo->getDimension());
        CamelliaCellTools::mapToReferenceSubcell(subcellPointsInParent, subcellCubPoints, subcellDim, subcellOrdinal, spaceTimeTopo);

        vector<int> dofOrdinals = tensorBasis->dofOrdinalsForSubcell(subcellDim, subcellOrdinal);
        Camellia::FieldContainer<double> values(tensorBasis->getCardinality(), numPoints);
        tensorBasis->getValues(values, subcellPointsInParent, OP_VALUE);

        for (auto dofOrdinal : dofOrdinals)
        {
          bool hasSupport = false;
          for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
          {
            if (abs(values(dofOrdinal,ptOrdinal)) > tol)
            {
              hasSupport = true;
              break;
            }
          }
          if (!hasSupport)
          {
            out << "basis for " << spaceTimeTopo->getName() << " claims dof ordinal " << dofOrdinal << " has support on ";
            out << "subcell " << subcellOrdinal << " of dimension " << subcellDim << ", but none found.\n";
          }
          TEST_ASSERT(hasSupport);
        }
      }
    }
  }
}

void testTensorBasisFillsFieldContainer(BasisPtr spatialBasis, BasisPtr temporalBasis,
                                        Teuchos::FancyOStream &out, bool &success)
{
  int spaceDim = spatialBasis->rangeDimension();
  FC tensorPoints = tensorPointsForDimension(spaceDim);
  FC myValues;
  Camellia::EOperator op = OP_VALUE;
  typedef Camellia::TensorBasis<double, Camellia::FieldContainer<double> > TensorBasis;
  Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(spatialBasis, temporalBasis) );

  sizeFCForBasisValues(myValues, tensorBasis, tensorPoints.dimension(0), op);
  double specialValue = 3.141592;
  myValues.initialize(specialValue);

  tensorBasis->getValues(myValues, tensorPoints, op);

  for (int valueOrdinal=0; valueOrdinal<myValues.size(); valueOrdinal++)
  {
    TEST_INEQUALITY(myValues[valueOrdinal], specialValue);
  }
}

void testTensorBasisValuesAreTensorProduct(BasisPtr spatialBasis, BasisPtr timeBasis,
    Teuchos::FancyOStream &out, bool &success)
{
  int spaceDim = spatialBasis->domainTopology()->getDimension();

  FC temporalPoints = timePoints();
  FC spatialPoints = spacePointsForDimension(spaceDim);
  FC tensorPoints = tensorPointsForDimension(spaceDim);

  int numSpacePoints = spatialPoints.dimension(0);
  int numTimePoints = temporalPoints.dimension(0);
  int numTensorPoints = tensorPoints.dimension(0);

  typedef Camellia::TensorBasis<double, Camellia::FieldContainer<double> > TensorBasis;
  Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(spatialBasis, timeBasis) );
  FC spatialValues, temporalValues(timeBasis->getCardinality(), numTimePoints), tensorValues;
  ::Camellia::EOperator op = OP_VALUE;
  ::Camellia::EOperator timeOp = OP_VALUE;

  sizeFCForBasisValues(spatialValues, spatialBasis, numSpacePoints, op);
  sizeFCForBasisValues(tensorValues, tensorBasis, numTensorPoints, op);

  spatialBasis->getValues(spatialValues, spatialPoints, op);
  timeBasis->getValues(temporalValues, temporalPoints, timeOp);
  tensorBasis->getValues(tensorValues, tensorPoints, op, timeOp);

  int rank = tensorValues.rank() - 2; // (F,P,D,D,...) where # of D's is the rank.

  vector<int> fieldCoord(2);
  for (int fieldOrdinal_i = 0; fieldOrdinal_i < spatialBasis->getCardinality(); fieldOrdinal_i++)
  {
    fieldCoord[0] = fieldOrdinal_i;
    for (int fieldOrdinal_j = 0; fieldOrdinal_j < timeBasis->getCardinality(); fieldOrdinal_j++)
    {
      fieldCoord[1] = fieldOrdinal_j;
      int fieldOrdinal_tensor = tensorBasis->getDofOrdinalFromComponentDofOrdinals(fieldCoord);
      for (int pointOrdinal_i = 0; pointOrdinal_i < numSpacePoints; pointOrdinal_i++)
      {
        vector<double> spaceValue;
        if (rank == 0)
        {
          spaceValue.push_back(spatialValues(fieldOrdinal_i, pointOrdinal_i));
        }
        else if (rank == 1)
        {
          for (int d=0; d<spaceDim; d++)
          {
            spaceValue.push_back(spatialValues(fieldOrdinal_i, pointOrdinal_i, d));
          }
        }
        else if (rank == 2)
        {
          for (int d1=0; d1<spaceDim; d1++)
          {
            for (int d2=0; d2<spaceDim; d2++)
            {
              spaceValue.push_back(spatialValues(fieldOrdinal_i, pointOrdinal_i, d1, d2));
            }
          }
        }
        for (int pointOrdinal_j = 0; pointOrdinal_j < numTimePoints; pointOrdinal_j++)
        {
          double timeValue = temporalValues(fieldOrdinal_j, pointOrdinal_j);
          int pointOrdinal_tensor = pointOrdinal_i + pointOrdinal_j * numSpacePoints;
          vector<double> tensorValue;
          if (rank == 0)
          {
            tensorValue.push_back(tensorValues(fieldOrdinal_tensor, pointOrdinal_tensor));
          }
          else if (rank == 1)
          {
            for (int d=0; d<spaceDim; d++)
            {
              tensorValue.push_back(tensorValues(fieldOrdinal_tensor, pointOrdinal_tensor, d));
            }
          }
          else if (rank == 2)
          {
            for (int d1=0; d1<spaceDim; d1++)
            {
              for (int d2=0; d2<spaceDim; d2++)
              {
                tensorValue.push_back(tensorValues(fieldOrdinal_tensor, pointOrdinal_tensor, d1, d2));
              }
            }
          }

          if (spaceValue.size() != tensorValue.size())
          {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal test error: tensorValue size does not match spaceValue size");
          }
          double tol = 1e-15;
          int numSpaceValues = spaceValue.size();
          for (int i=0; i<numSpaceValues; i++)
          {
            double expectedValue = spaceValue[i] * timeValue;
            double actualValue = tensorValue[i];
            TEST_FLOATING_EQUALITY(expectedValue,actualValue,tol);
          }
        }
      }
    }
  }
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell_HGRADInTime_Point )
{
  CellTopoPtr spaceTopo = CellTopology::point();
  bool useHGRADForTime = true;
  int spaceH1Order = 2, timeH1Order = 2;
  testBasisOrdinalsForSubcell(spaceTopo, useHGRADForTime, spaceH1Order, timeH1Order, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell_HGRADInTime_Line )
{
  CellTopoPtr spaceTopo = CellTopology::line();
  bool useHGRADForTime = true;
  int spaceH1Order = 2, timeH1Order = 2;
  testBasisOrdinalsForSubcell(spaceTopo, useHGRADForTime, spaceH1Order, timeH1Order, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell_HGRADInTime_Quad )
{
  CellTopoPtr spaceTopo = CellTopology::quad();
  bool useHGRADForTime = true;
  int spaceH1Order = 2, timeH1Order = 2;
  testBasisOrdinalsForSubcell(spaceTopo, useHGRADForTime, spaceH1Order, timeH1Order, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell_HGRADInTime_Triangle )
{
  CellTopoPtr spaceTopo = CellTopology::triangle();
  bool useHGRADForTime = true;
  int spaceH1Order = 2, timeH1Order = 2;
  testBasisOrdinalsForSubcell(spaceTopo, useHGRADForTime, spaceH1Order, timeH1Order, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell_HGRADInTime_Hexahedron )
{
  CellTopoPtr spaceTopo = CellTopology::hexahedron();
  bool useHGRADForTime = true;
  int spaceH1Order = 2, timeH1Order = 2;
  testBasisOrdinalsForSubcell(spaceTopo, useHGRADForTime, spaceH1Order, timeH1Order, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell_HGRADInTime_Tetrahedron )
{
  CellTopoPtr spaceTopo = CellTopology::tetrahedron();
  bool useHGRADForTime = true;
  int spaceH1Order = 2, timeH1Order = 2;
  testBasisOrdinalsForSubcell(spaceTopo, useHGRADForTime, spaceH1Order, timeH1Order, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell_HVOLInTime_Point )
{
  CellTopoPtr spaceTopo = CellTopology::point();
  bool useHGRADForTime = false;
  int spaceH1Order = 2, timeH1Order = 2;
  testBasisOrdinalsForSubcell(spaceTopo, useHGRADForTime, spaceH1Order, timeH1Order, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell_HVOLInTime_Line )
{
  CellTopoPtr spaceTopo = CellTopology::line();
  bool useHGRADForTime = false;
  int spaceH1Order = 2, timeH1Order = 2;
  testBasisOrdinalsForSubcell(spaceTopo, useHGRADForTime, spaceH1Order, timeH1Order, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell_HVOLInTime_Quad )
{
  CellTopoPtr spaceTopo = CellTopology::quad();
  bool useHGRADForTime = false;
  int spaceH1Order = 2, timeH1Order = 2;
  testBasisOrdinalsForSubcell(spaceTopo, useHGRADForTime, spaceH1Order, timeH1Order, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell_HVOLInTime_Triangle )
{
  CellTopoPtr spaceTopo = CellTopology::triangle();
  bool useHGRADForTime = false;
  int spaceH1Order = 2, timeH1Order = 2;
  testBasisOrdinalsForSubcell(spaceTopo, useHGRADForTime, spaceH1Order, timeH1Order, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell_HVOLInTime_Hexahedron )
{
  CellTopoPtr spaceTopo = CellTopology::hexahedron();
  bool useHGRADForTime = false;
  int spaceH1Order = 2, timeH1Order = 2;
  testBasisOrdinalsForSubcell(spaceTopo, useHGRADForTime, spaceH1Order, timeH1Order, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisOrdinalsForSubcell_HVOLInTime_Tetrahedron )
{
  CellTopoPtr spaceTopo = CellTopology::tetrahedron();
  bool useHGRADForTime = false;
  int spaceH1Order = 2, timeH1Order = 2;
  testBasisOrdinalsForSubcell(spaceTopo, useHGRADForTime, spaceH1Order, timeH1Order, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, GetTensorValues )
{
  int spatialPolyOrder = 1;
  std::vector< BasisPtr > spatialBases = getSpatialBases(spatialPolyOrder);

  int timePolyOrder = 1;
  BasisPtr timeBasis = getTimeBasis(timePolyOrder);

  double tol = 1e-15;
  int numSpatialBases = spatialBases.size();
  for (int i=0; i<numSpatialBases; i++)
  {
    BasisPtr spatialBasis = spatialBases[i];
    int spaceDim = spatialBasis->domainTopology()->getDimension();

    FC temporalPoints = timePoints();
    FC spatialPoints = spacePointsForDimension(spaceDim);
    FC tensorPoints = tensorPointsForDimension(spaceDim);

    int numSpacePoints = spatialPoints.dimension(0);
    int numTimePoints = temporalPoints.dimension(0);
    int numTensorPoints = tensorPoints.dimension(0);

    typedef Camellia::TensorBasis<double, Camellia::FieldContainer<double> > TensorBasis;
    Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(spatialBasis, timeBasis) );
    FC spatialValues, temporalValues(timeBasis->getCardinality(), numTimePoints), tensorValuesExpected, tensorValuesActual;
    ::Camellia::EOperator op = OP_VALUE;
    ::Camellia::EOperator timeOp = OP_VALUE;

    sizeFCForBasisValues(spatialValues, spatialBasis, numSpacePoints, op);
    sizeFCForBasisValues(tensorValuesExpected, tensorBasis, numTensorPoints, op);

    spatialBasis->getValues(spatialValues, spatialPoints, op);
    timeBasis->getValues(temporalValues, temporalPoints, timeOp);
    tensorBasis->getValues(tensorValuesExpected, tensorPoints, op, timeOp);

    out << "spatialValues:\n" << spatialValues;
    out << "temporalValues:\n" << temporalValues;

    sizeFCForBasisValues(tensorValuesActual, tensorBasis, numTensorPoints, op);
    vector< Camellia::FieldContainer<double> > componentValues(2);
    componentValues[0] = spatialValues;
    componentValues[1] = temporalValues;

    vector<::Camellia::EOperator> componentOps(2);
    componentOps[0] = op;
    componentOps[1] = timeOp;

    tensorBasis->getTensorValues(tensorValuesActual, componentValues, componentOps);

    TEST_COMPARE_FLOATING_ARRAYS(tensorValuesActual, tensorValuesExpected, tol);
  }
}

TEUCHOS_UNIT_TEST( TensorBasis, TensorBasisFieldOrderingAgreesWithTensorTopologyOrdering )
{
  int H1Order = 1;

  BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
  CellTopoPtr line = CellTopology::line();

  BasisPtr basis = basisFactory->getBasis(H1Order, line, Camellia::FUNCTION_SPACE_HGRAD);

  typedef Camellia::TensorBasis<double, Camellia::FieldContainer<double> > TensorBasis;
  Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(basis, basis) );

  int tensorialDegree = 1;
  CellTopoPtr tensorTopo = CellTopology::cellTopology(line->getShardsTopology(), tensorialDegree);

  int numSpatialNodes = line->getNodeCount();
  int numTimeNodes = line->getNodeCount();
  vector<unsigned> compNodes(2);
  vector<int> compBasisOrdinals(2);
  for (int spatialNode=0; spatialNode<numSpatialNodes; spatialNode++)
  {
    compNodes[0] = spatialNode;
    compBasisOrdinals[0] = spatialNode;
    for (int timeNode=0; timeNode<numTimeNodes; timeNode++)
    {
      compNodes[1] = timeNode;
      compBasisOrdinals[1] = timeNode;
      unsigned topoOrdinal = tensorTopo->getNodeFromTensorialComponentNodes(compNodes);
      unsigned basisOrdinal = tensorBasis->getDofOrdinalFromComponentDofOrdinals(compBasisOrdinals);
      TEST_EQUALITY(topoOrdinal, basisOrdinal);
    }
  }
}

TEUCHOS_UNIT_TEST( TensorBasis, TensorBasisPointOrderingAgreesWithTensorTopologyOrdering )
{
  // check that the way points are ordered in tensor basis is the same as the way they're ordered in
  // CellTopology
  int H1Order = 1;

  BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
  CellTopoPtr timeTopo = CellTopology::line();
  CellTopoPtr spaceTopo = CellTopology::triangle();

  BasisPtr spaceBasis = basisFactory->getBasis(H1Order, spaceTopo, Camellia::FUNCTION_SPACE_HGRAD);
  BasisPtr timeBasis = basisFactory->getBasis(H1Order, timeTopo, Camellia::FUNCTION_SPACE_HGRAD);

  typedef Camellia::TensorBasis<double, Camellia::FieldContainer<double> > TensorBasis;
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

TEUCHOS_UNIT_TEST( TensorBasis, ValuesEqualTensorProduct )
{
  int spatialPolyOrder = 1;
  std::vector< BasisPtr > spatialBases = getSpatialBases(spatialPolyOrder);

  int timePolyOrder = 1;
  BasisPtr timeBasis = getTimeBasis(timePolyOrder);
  
  int numSpatialBases = spatialBases.size();
  for (int i=0; i<numSpatialBases; i++)
  {
    BasisPtr spatialBasis = spatialBases[i];
    testTensorBasisValuesAreTensorProduct(spatialBasis, timeBasis, out, success);
  }
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisFillsContainer )
{
  int spatialPolyOrder = 1;
  std::vector< BasisPtr > spatialBases = getSpatialBases(spatialPolyOrder);

  int timePolyOrder = 1;
  BasisPtr timeBasis = getTimeBasis(timePolyOrder);

  int numSpatialBases = spatialBases.size();
  for (int i=0; i<numSpatialBases; i++)
  {
    BasisPtr spatialBasis = spatialBases[i];
    testTensorBasisFillsFieldContainer(spatialBasis, timeBasis, out, success);
  }
}

TEUCHOS_UNIT_TEST( TensorBasis, BasisFillsContainerQuadHDIV_2D )
{
  int spatialPolyOrder = 3;
  BasisPtr spatialBasis = getSpatialBasis(CellTopology::quad(), Camellia::FUNCTION_SPACE_HDIV, spatialPolyOrder);

  int timePolyOrder = 2;
  BasisPtr timeBasis = getTimeBasis(timePolyOrder);

  testTensorBasisFillsFieldContainer(spatialBasis, timeBasis, out, success);
}

TEUCHOS_UNIT_TEST( TensorBasis, Projection_2D_x )
{
  const double tol = 1e-15;
  
  const int baseDim          = 1; // corresponds to the fact that the initial mesh topo is 1D.
  const int numExtrusionDims = 1;
  const int totalDim         = baseDim + numExtrusionDims;
  
  
  // 1. Construct extruded 2D mesh and corresponding variable, solution
  double x0 = 0.0, xWidth = 1.0;
  
  const int xElements = 2;
  bool usePeriodicBCs            = true;
  auto intervalMeshTopology      = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElements, usePeriodicBCs);
  
  std::vector<double> v0(numExtrusionDims,0.0);
  std::vector<double> vWidths(numExtrusionDims,1.0);
  std::vector<int> vMeshWidths(numExtrusionDims,1);
  MeshTopologyPtr extrudedMeshTopology = MeshFactory::extrudedMeshTopology(intervalMeshTopology, v0, vWidths, vMeshWidths);
  
  VarFactoryPtr vf = VarFactory::varFactory();
  VarPtr u = vf->fieldVar("u");
  
  const int H1Order = 2;
  const int delta_k = 0;
  
  MeshPtr mesh = Teuchos::rcp(new Mesh(extrudedMeshTopology, vf, H1Order, delta_k));
  TEST_EQUALITY(false, mesh->tensorTopologyMeansSpaceTime());
  
  // get the basis for u:
  auto uBasis = mesh->getElementType(0)->trialOrderPtr->getBasis(u->ID());
  // confirm that the range dimension for u is correct:
  TEST_EQUALITY(totalDim, uBasis->rangeDimension());
  
  SolutionPtr soln = Solution::solution(Teuchos::null, mesh);
  soln->setCubatureEnrichmentDegree(H1Order); // we do this to get around a quirk in 
  FunctionPtr u_soln = Function::solution(u, soln);
  
  // 2. Project x onto that variable
  FunctionPtr x = Function::xn(1);
  std::map<int, FunctionPtr> xMap;
  xMap[u->ID()] = x;
  soln->projectOntoMesh(xMap);

  // 3. Check error between x and the solution
  double err_l2 = (u_soln - x)->l2norm(mesh);
  TEST_COMPARE(err_l2, <, tol);
  
//  // 4. Check d/dy of solution is zero
//  double u_dy_l2 = u_soln->dy()->l2norm(mesh);
//  TEST_COMPARE(u_dy_l2, <, tol);
//  
//  // check the more general way of expressing d/dy:
//  u_dy_l2 = u_soln->di(1)->l2norm(mesh);
//  TEST_COMPARE(u_dy_l2, <, tol);
}

} // namespace
