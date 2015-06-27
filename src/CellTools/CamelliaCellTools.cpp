//
//  CamelliaCellTools.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 6/6/14.
//
//

#include "CamelliaCellTools.h"
#include "BasisCache.h"

#include "BasisFactory.h"
#include "TensorBasis.h"

#include "MeshTransformationFunction.h"

#include "SerialDenseWrapper.h"

#include "CellTopology.h"

using namespace Intrepid;
using namespace Camellia;

CellTopoPtr CamelliaCellTools::cellTopoForKey(Camellia::CellTopologyKey key)
{
  CellTopoPtrLegacy shardsTopo = cellTopoForKey(key.first);
  return CellTopology::cellTopology(*shardsTopo, key.second);
}

CellTopoPtrLegacy CamelliaCellTools::cellTopoForKey(unsigned key)
{
  static CellTopoPtrLegacy node, line, triangle, quad, tet, hex;

  switch (key)
  {
  case shards::Node::key:
    if (node.get()==NULL)
    {
      node = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Node >() ));
    }
    return node;
    break;
  case shards::Line<2>::key:
    if (line.get()==NULL)
    {
      line = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Line<2> >() ));
    }
    return line;
    break;
  case shards::Triangle<3>::key:
    if (triangle.get()==NULL)
    {
      triangle = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Triangle<3> >() ));
    }
    return triangle;
    break;
  case shards::Quadrilateral<4>::key:
    if (quad.get()==NULL)
    {
      quad = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Quadrilateral<4> >() ));
    }
    return quad;
    break;
  case shards::Tetrahedron<4>::key:
    if (tet.get()==NULL)
    {
      tet = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Tetrahedron<4> >() ));
    }
    return tet;
    break;
  case shards::Hexahedron<8>::key:
    if (hex.get()==NULL)
    {
      hex = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData< shards::Hexahedron<8> >() ));
    }
    return hex;
    break;
  default:
    cout << "Unhandled CellTopology.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled CellTopology.");
  }
}

int CamelliaCellTools::checkPointInclusion(const double*                 point,
                                           const int                     pointDim,
                                           CellTopoPtr                   cellTopo,
                                           const double &                threshold)
{
  TEUCHOS_TEST_FOR_EXCEPTION( !(pointDim == (int)cellTopo->getDimension() ), std::invalid_argument,
                             ">>> ERROR (Intrepid::CellTools::checkPointInclusion): Point and cell dimensions do not match. ");
  if (cellTopo->getTensorialDegree() == 0)
  {
    return Intrepid::CellTools<double>::checkPointInclusion(point, pointDim, cellTopo->getShardsTopology(), threshold);
  }
  else
  {
    // first entries of point belong to the shards topology -- check whether these lie inside it
    int shardsDim = cellTopo->getShardsTopology().getDimension();
    int shardsInclusion = Intrepid::CellTools<double>::checkPointInclusion(point, shardsDim, cellTopo->getShardsTopology(), threshold);
    if (shardsInclusion != 1) return shardsInclusion;
    // if included in shards, then check whether the tensorial dimensions lie inside reference cell (-1,1)
    for (int tensorDim=shardsDim; tensorDim<pointDim; tensorDim++)
    {
      if ((point[tensorDim] < -1.0 - threshold) || (point[tensorDim] > 1.0 + threshold)) return 0; // not included
    }
    return 1;
  }
}

bool CamelliaCellTools::jacobianIsOrthogonal(const FieldContainer<double> &cellJacobian, int d1, int d2, double tol)
{
  TEUCHOS_TEST_FOR_EXCEPTION(cellJacobian.rank() != 4, std::invalid_argument, "");
  int numCells = cellJacobian.dimension(0);
  int numPoints = cellJacobian.dimension(1);
  int spaceDim = cellJacobian.dimension(2);
  TEUCHOS_TEST_FOR_EXCEPTION(cellJacobian.dimension(3) != spaceDim, std::invalid_argument, "");

  for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
  {
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      if ((abs(cellJacobian(cellOrdinal,pointOrdinal,d1,d2)) > tol) || (abs(cellJacobian(cellOrdinal,pointOrdinal,d2,d1)) > tol))
      {
        // DEBUGGING:
        cout << pointOrdinal << " " << d1 << " " << d2 << endl;
        cout << abs(cellJacobian(cellOrdinal,pointOrdinal,d1,d2)) << " " << abs(cellJacobian(cellOrdinal,pointOrdinal,d2,d1)) << endl;
        return false;
      }
    }
  }
  return true;
}

void CamelliaCellTools::computeSideMeasure(FieldContainer<double> &weightedMeasure, const FieldContainer<double> &cellJacobian, const FieldContainer<double> &cubWeights,
    int sideOrdinal, CellTopoPtr parentCell)
{
  int spaceDim = parentCell->getDimension();
  int numCells = cellJacobian.dimension(0);
  int numPoints = cellJacobian.dimension(1);

  if (parentCell->getTensorialDegree() == 0)
  {
    if (spaceDim < 2)   // side topology is then a point; just copy the cubature weights
    {
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
          weightedMeasure(cellOrdinal,ptOrdinal) = cubWeights(ptOrdinal);
        }
      }
    }
    else     // spaceDim >= 2
    {
      if (spaceDim == 2)
      {
        // compute weighted edge measure
        FunctionSpaceTools::computeEdgeMeasure<double>(weightedMeasure, cellJacobian, cubWeights, sideOrdinal, parentCell->getShardsTopology());
      }
      else if (spaceDim == 3)
      {
        FunctionSpaceTools::computeFaceMeasure<double>(weightedMeasure, cellJacobian, cubWeights, sideOrdinal, parentCell->getShardsTopology());
      }
      else
      {
        cout << "ERROR: CamelliaCellTools::computeSideMeasure() does not yet support dimension > 3 for shards topologies.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "recomputeMeasures() does not yet support tensorial degree > 0.");
      }
    }
  }
  else     // tensorialDegree > 0
  {
    // for the below to work, we require the transformation preserves orthogonality of space and time.  Check that the Jacobian provided satisfies this:
    FieldContainer<double> spatialCellJacobian(numCells,numPoints,spaceDim-1,spaceDim-1);
    FieldContainer<double> temporalCellJacobianAbs(numCells,numPoints);
    
    for (int d1=0; d1<spaceDim-1; d1++)
    {
      int d2 = spaceDim - 1;
      const double tol = 1e-13;
      if (!jacobianIsOrthogonal(cellJacobian, d1, d2, tol))
      {
        cout << "CamelliaCellTools::computeSideMeasure requires the transformation to be orthogonal in space and time.\n";
        cout << "cellJacobian:\n" << cellJacobian;
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CamelliaCellTools::computeSideMeasure requires the transformation to be orthogonal in space and time.");
      }
    }
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        for (int d1=0; d1<spaceDim-1; d1++)
        {
          for (int d2=0; d2<spaceDim-1; d2++)
          {
            spatialCellJacobian(cellOrdinal,ptOrdinal,d1,d2) = cellJacobian(cellOrdinal,ptOrdinal,d1,d2);
          }
        }
        temporalCellJacobianAbs(cellOrdinal,ptOrdinal) = abs(cellJacobian(cellOrdinal,ptOrdinal,spaceDim-1,spaceDim-1));
      }
    }
    // if we get here, cellJacobian satisfies the orthogonality requirement and spatialCellJacobian contains the appropriate submatrix of cellJacobian.
    if (parentCell->sideIsSpatial(sideOrdinal))
    {
      int spatialSideOrdinal = parentCell->getSpatialComponentSideOrdinal(sideOrdinal);
      // then the space-time side is comprised of space side x temporal line, and we can recurse:
      CellTopoPtr spatialCell = CellTopology::cellTopology(parentCell->getShardsTopology(), parentCell->getTensorialDegree()-1);
      computeSideMeasure(weightedMeasure, spatialCellJacobian, cubWeights, spatialSideOrdinal, spatialCell);
      // next, we multiply the weights uniformly by the absolute value of the temporal Jacobian (which is just a scalar value for each point)
      ArrayTools::scalarMultiplyDataData<double>(weightedMeasure, weightedMeasure, temporalCellJacobianAbs);
    }
    else
    {
      // here, the space-time side is an instance of the spatial cell, so we want to use FunctionSpaceTools::computeCellMeasure
      FieldContainer<double> spatialCellJacobianDet(numCells,numPoints);
      Intrepid::CellTools<double>::setJacobianDet(spatialCellJacobianDet, spatialCellJacobian );
      FunctionSpaceTools::computeCellMeasure<double>(weightedMeasure, spatialCellJacobianDet, cubWeights);
    }
  }
}

string CamelliaCellTools::entityTypeString(unsigned entityDimension)   // vertex, edge, face, solid, hypersolid
{
  switch (entityDimension)
  {
  case 0:
    return "vertex";
  case 1:
    return "edge";
  case 2:
    return "face";
  case 3:
    return "solid";
  case 4:
    return "hypersolid";
  default:
    return "unknown entity type";
  }
}

void CamelliaCellTools::getReferenceSideNormal(FieldContainer<double> &refSideNormal, int sideOrdinal, CellTopoPtr parentCell)
{
  if ((sideOrdinal < 0) || (sideOrdinal >= parentCell->getSideCount()))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sideOrdinal out of bounds");
  }
  if (refSideNormal.rank() != 1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refSideNormal must have dimensions (D)");
  }

  int spaceDim = parentCell->getDimension();
  if (refSideNormal.dimension(0) != spaceDim)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "refSideNormal must have dimensions (D), where D is the parent cell's spatial dimension");
  }

  if (spaceDim == 1)
  {
    if (sideOrdinal == 0)
    {
      refSideNormal[0] = -1;
    }
    else
    {
      refSideNormal[0] =  1;
    }
  }
  else if (parentCell->getTensorialDegree() == 0)
  {
    if ((spaceDim != 2) && (spaceDim != 3))
    {
      // shouldn't get here, except for point topology.  Can't get side normals for point topology...
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "When tensorial degree == 0, spaceDim must be 2 or 3");
    }
    Intrepid::CellTools<double>::getReferenceSideNormal(refSideNormal, sideOrdinal, parentCell->getShardsTopology());
  }
  else
  {
    refSideNormal.initialize(0.0);
    if (! parentCell->sideIsSpatial(sideOrdinal))
    {
      // then it's a temporal side.  First temporal side has normal (0,…,-1); second has normal (0,…,1)
      int temporalOrdinal = parentCell->getTemporalComponentSideOrdinal(sideOrdinal);
      if (temporalOrdinal == 0)   // first temporal side
      {
        refSideNormal[spaceDim-1] = -1;
      }
      else     // second temporal side
      {
        refSideNormal[spaceDim-1] =  1;
      }
    }
    else
    {
      // spatial side: normal will be same as that of the spatial topology, with a 0 tacked on for the temporal dimension (since we initialize to 0 above, the latter is taken care of)
      Teuchos::Array<int> dim(1,spaceDim-1);
      FieldContainer<double> spatialRefSideNormal(dim,&refSideNormal[0]); // FC pointing to the spatial part of the refSideNormal
      CellTopoPtr spatialParentCell = CellTopology::cellTopology(parentCell->getShardsTopology(), parentCell->getTensorialDegree() - 1);
      unsigned spatialSideOrdinal = parentCell->getSpatialComponentSideOrdinal(sideOrdinal);
      getReferenceSideNormal(spatialRefSideNormal, spatialSideOrdinal, spatialParentCell);
    }
  }

}

int CamelliaCellTools::getSideCount(const shards::CellTopology &cellTopo)
{
  // unlike shards itself, defines vertices as sides for Line topo
  return (cellTopo.getDimension() > 1) ? cellTopo.getSideCount() : cellTopo.getVertexCount();
}

void CamelliaCellTools::getUnitSideNormals(FieldContainer<double> &unitSideNormals, int sideOrdinal, const FieldContainer<double> &inCellJacobian, CellTopoPtr parentCell)
{
  int numCells = unitSideNormals.dimension(0);
  int numPoints = unitSideNormals.dimension(1);
  int spaceDim = unitSideNormals.dimension(2);

  if (spaceDim==0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim==0 not supported in CamelliaCellTools::getUnitSideNormals.");
  }
  else if (spaceDim==1)     // Line
  {
    // If it's the first side, the normal points left; if it's the second, it points right.
    // the direction is flipped if the cell jacobian is negative...
    double refSpaceNormal = (sideOrdinal == 0) ? -1 : 1;
    unitSideNormals.initialize(0);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        double jacobianDirection = (inCellJacobian(cellOrdinal,ptOrdinal,spaceDim-1,spaceDim-1) > 0) ? 1.0 : -1.0;
        double normal = refSpaceNormal * jacobianDirection;
        unitSideNormals(cellOrdinal,ptOrdinal,0) = normal;
      }
    }
  }
  else if (parentCell->getTensorialDegree() == 0)
  {
    Intrepid::CellTools<double>::getPhysicalSideNormals(unitSideNormals, inCellJacobian, sideOrdinal, parentCell->getShardsTopology());      // make unit length
    FieldContainer<double> normalLengths(numCells, numPoints);
    RealSpaceTools<double>::vectorNorm(normalLengths, unitSideNormals, NORM_TWO);
    FunctionSpaceTools::scalarMultiplyDataData<double>(unitSideNormals, normalLengths, unitSideNormals, true); // true: divide
  }
  else
  {
    if (parentCell->sideIsSpatial(sideOrdinal))
    {
      FieldContainer<double> spaceUnitSideNormals(numCells,numPoints,spaceDim-1);
      CellTopoPtr spaceTopo = CellTopology::cellTopology(parentCell->getShardsTopology(), parentCell->getTensorialDegree()-1);
      // for the below to work, we require the transformation preserves orthogonality of space and time.  Check that the Jacobian provided satisfies this:
      unsigned spatialSideOrdinal = parentCell->getSpatialComponentSideOrdinal(sideOrdinal);
      FieldContainer<double> spatialCellJacobian(numCells,numPoints,spaceDim-1,spaceDim-1);
      FieldContainer<double> temporalCellJacobianAbs(numCells,numPoints);
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
          for (int d1=0; d1<spaceDim-1; d1++)
          {
            int d2 = spaceDim - 1;
            const double tol = 1e-14;
            if ((abs(inCellJacobian(cellOrdinal,ptOrdinal,d1,d2)) > tol) || (abs(inCellJacobian(cellOrdinal,ptOrdinal,d2,d1)) > tol))
            {
              cout << "CamelliaCellTools::getUnitSideNormals requires the transformation to be orthogonal in space and time.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CamelliaCellTools::getUnitSideNormals requires the transformation to be orthogonal in space and time.");
            }
            for (int d2=0; d2<spaceDim-1; d2++)
            {
              spatialCellJacobian(cellOrdinal,ptOrdinal,d1,d2) = inCellJacobian(cellOrdinal,ptOrdinal,d1,d2);
            }
          }
          temporalCellJacobianAbs(cellOrdinal,ptOrdinal) = abs(inCellJacobian(cellOrdinal,ptOrdinal,spaceDim-1,spaceDim-1));
        }
      }
      CamelliaCellTools::getUnitSideNormals(spaceUnitSideNormals, spatialSideOrdinal, spatialCellJacobian, spaceTopo);
      unitSideNormals.initialize(0);
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
          for (int d=0; d<spaceDim-1; d++)
          {
            unitSideNormals(cellOrdinal,ptOrdinal,d) = spaceUnitSideNormals(cellOrdinal,ptOrdinal,d);
          }
        }
      }
    }
    else     // side is not spatial; it's temporal.  If it's the first side, the normal points downward; if it's the second, it points upward.
    {
      // the direction is flipped if the cell jacobian is negative...
      unsigned temporalNodeOrdinal = parentCell->getTemporalComponentSideOrdinal(sideOrdinal);
      double refSpaceNormal = (temporalNodeOrdinal == 0) ? -1 : 1;
      unitSideNormals.initialize(0);
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
          double jacobianDirection = (inCellJacobian(cellOrdinal,ptOrdinal,spaceDim-1,spaceDim-1) > 0) ? 1.0 : -1.0;
          double temporalNormal = refSpaceNormal * jacobianDirection;
          unitSideNormals(cellOrdinal,ptOrdinal,spaceDim-1) = temporalNormal;
        }
      }
    }
  }
}

void CamelliaCellTools::refCellNodesForTopology(FieldContainer<double> &cellNodes, const shards::CellTopology &cellTopo, unsigned permutation)   // 0 permutation is the identity
{
  // really, ref cell *vertices* for topology.  (shards/Intrepid distinguish between vertices and nodes; we don't.)

  int vertexCount = cellTopo.getVertexCount();
  if (cellNodes.dimension(0) != vertexCount)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellNodes must be sized (N,D) where N=vertex count, D=space dim.");
  }
  if (cellNodes.dimension(1) != cellTopo.getDimension())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellNodes must be sized (N,D) where N=vertex count, D=space dim.");
  }
  int dim = cellTopo.getDimension();

  for (int vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++)
  {
    const double* v = Intrepid::CellTools<double>::getReferenceVertex(cellTopo, vertexOrdinal);
    for (int d=0; d<dim; d++)
    {
      cellNodes(vertexOrdinal,d) = v[d];
    }
  }

  if ( permutation != 0 )
  {
    FieldContainer<double> cellNodesCopy = cellNodes;
    unsigned nodeCount = cellNodes.dimension(0);
    unsigned spaceDim = cellNodes.dimension(1);
    for (int n = 0; n<nodeCount; n++)
    {
      int n_permuted = cellTopo.getNodePermutation(permutation, n);
      for (int d = 0; d<spaceDim; d++)
      {
        cellNodes(n,d) = cellNodesCopy(n_permuted,d);
      }
    }
  }
}

void CamelliaCellTools::refCellNodesForTopology(FieldContainer<double> &cellNodes, CellTopoPtr cellTopo, unsigned permutation)
{
  if ((cellNodes.dimension(0) != cellTopo->getNodeCount()) && (cellTopo->getDimension() != 0) )   // shards and Camellia disagree on the node count for points (0 vs. 1), so we accept either as dimensions for cellNodes, which is a size 0 container in any case...
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellNodes must be sized (N,D) where N=node count, D=space dim.");
  }
  if (cellNodes.dimension(1) != cellTopo->getDimension())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellNodes must be sized (N,D) where N=node count, D=space dim.");
  }

//  if (cellTopo->getTensorialDegree() == 0) {
//    // then the other variant of refCellNodesForTopology will do the trick
//    refCellNodesForTopology(cellNodes, cellTopo->getShardsTopology(), permutation);
//    return;
//  }

  shards::CellTopology shardsTopology = cellTopo->getShardsTopology();
  FieldContainer<double> shardsCellNodes(shardsTopology.getNodeCount(), shardsTopology.getDimension());

  refCellNodesForTopology(shardsCellNodes, shardsTopology);

  vector< FieldContainer<double> > tensorComponentNodes;
  tensorComponentNodes.push_back(shardsCellNodes);

  FieldContainer<double> lineNodes(2,1);
  shards::CellTopology shardsLine(shards::getCellTopologyData<shards::Line<2> >() );
  refCellNodesForTopology(lineNodes, shardsLine);

  for (int degreeOrdinal=0; degreeOrdinal<cellTopo->getTensorialDegree(); degreeOrdinal++)
  {
    tensorComponentNodes.push_back(lineNodes);
  }

  cellNodes.resize(cellTopo->getNodeCount(),cellTopo->getDimension());

  cellTopo->initializeNodes(tensorComponentNodes, cellNodes);

  if ( permutation != 0 )
  {
    FieldContainer<double> cellNodesCopy = cellNodes;
    unsigned nodeCount = cellNodes.dimension(0);
    unsigned spaceDim = cellNodes.dimension(1);
    for (int n = 0; n<nodeCount; n++)
    {
      int n_permuted = cellTopo->getNodePermutation(permutation, n);
      for (int d = 0; d<spaceDim; d++)
      {
        cellNodes(n,d) = cellNodesCopy(n_permuted,d);
      }
    }
  }
}

void CamelliaCellTools::refCellNodesForTopology(std::vector< vector<double> > &cellNodes, CellTopoPtr cellTopo, unsigned permutation)
{
  int vertexCount = cellTopo->getVertexCount();
  int spaceDim = cellTopo->getDimension();
  FieldContainer<double> cellNodesFC(vertexCount, spaceDim);
  refCellNodesForTopology(cellNodesFC,cellTopo,permutation);
  cellNodes.resize(vertexCount);
  for (int vertexOrdinal=0; vertexOrdinal < vertexCount; vertexOrdinal++)
  {
    cellNodes[vertexOrdinal].resize(cellTopo->getDimension());
    for (int d=0; d<spaceDim; d++)
    {
      cellNodes[vertexOrdinal][d] = cellNodesFC(vertexOrdinal,d);
    }
  }
}

void CamelliaCellTools::mapToPhysicalFrame(FieldContainer<double> &physPoints, const FieldContainer<double> &refPoints, const FieldContainer<double> &cellWorkset,
    CellTopoPtr cellTopo, const int & whichCell)
{
  BasisPtr nodalBasis = BasisFactory::basisFactory()->getNodalBasisForCellTopology(cellTopo);

  int basisCardinality = nodalBasis->getCardinality();
  int numCells = cellWorkset.dimension(0);
  int numNodes = cellWorkset.dimension(1);
  int numPoints = (refPoints.rank() == 2) ? refPoints.dimension(0) : refPoints.dimension(1);

  if (numNodes != cellTopo->getNodeCount())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Second dimension of cellWorkset does not match cellTopo->getNodeCount()!");
  }
  if (cellTopo->getNodeCount() != nodalBasis->getCardinality())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: cellTopo node count does not match nodalBasis->getCardinality()!");
  }

  FieldContainer<double> basisValues(basisCardinality, numPoints);

  physPoints.initialize(0.0);
  int spaceDim = cellTopo->getDimension();

  // handle separately rank-2 (P,D) and rank-3 (C,P,D) cases of refPoints
  switch(refPoints.rank())
  {
  case 2:
  {
    nodalBasis->getValues(basisValues, refPoints, Intrepid::OPERATOR_VALUE);
    // If whichCell = -1, ref pt. set is mapped to all cells, otherwise, the set is mapped to one cell only
    int cellLoop = (whichCell == -1) ? numCells : 1 ;

    // Compute the map F(refPoints) = sum node_coordinate*basis(refPoints)
    for(int cellOrdinal = 0; cellOrdinal < cellLoop; cellOrdinal++)
    {
      for(int pointOrdinal = 0; pointOrdinal < numPoints; pointOrdinal++)
      {
        for(int d = 0; d < spaceDim; d++)
        {
          for(int basisOrdinal = 0; basisOrdinal < basisCardinality; basisOrdinal++)
          {

            if(whichCell == -1)
            {
              physPoints(cellOrdinal, pointOrdinal, d) += cellWorkset(cellOrdinal, basisOrdinal, d)*basisValues(basisOrdinal, pointOrdinal);
            }
            else
            {
              physPoints(pointOrdinal, d) += cellWorkset(whichCell, basisOrdinal, d)*basisValues(basisOrdinal, pointOrdinal);
            }
          } // basisOrdinal
        }// d
      }// pointOrdinal
    }//cellOrdinal
  }// case 2
  break;

  // refPoints is (C,P,D): multiple sets of ref. points are mapped to matching number of physical cells.
  case 3:
  {
    FieldContainer<double> refPointsForCell(numPoints, spaceDim);
    // Compute the map F(refPoints) = sum node_coordinate*basis(refPoints)
    for(int cellOrdinal = 0; cellOrdinal < numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        for (int d=0; d<spaceDim; d++)
        {
          refPointsForCell(pointOrdinal,d) = refPoints(cellOrdinal,pointOrdinal,d);
        }
      }

      // Compute basis values for this set of ref. points
      nodalBasis -> getValues(basisValues, refPointsForCell, Intrepid::OPERATOR_VALUE);

      for(int pointOrdinal = 0; pointOrdinal < numPoints; pointOrdinal++)
      {
        for(int d = 0; d < spaceDim; d++)
        {
          for(int basisOrdinal = 0; basisOrdinal < basisCardinality; basisOrdinal++)
          {

            physPoints(cellOrdinal, pointOrdinal, d) += cellWorkset(cellOrdinal, basisOrdinal, d)*basisValues(basisOrdinal, pointOrdinal);

          } // basisOrdinal
        }// d
      }// pointOrdinal
    }//cellOrdinal
  }// case 3
  break;

  default:
    TEUCHOS_TEST_FOR_EXCEPTION( !( (refPoints.rank() == 2) && (refPoints.rank() == 3) ), std::invalid_argument,
                                ">>> ERROR (CamelliaCellTools::mapToPhysicalFrame): rank 2 or 3 required for refPoints array. ");
  }
}

unsigned CamelliaCellTools::permutationMatchingOrder( CellTopoPtr cellTopo, const vector<unsigned> &fromOrder, const vector<unsigned> &toOrder)
{
  if (cellTopo->getDimension() == 0)
  {
    return 0;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(cellTopo->getNodeCount() != fromOrder.size(), std::invalid_argument, "fromOrder.size() != node count");
  TEUCHOS_TEST_FOR_EXCEPTION(cellTopo->getNodeCount() != toOrder.size(), std::invalid_argument, "toOrder.size() != node count");
  unsigned permutationCount = cellTopo->getNodePermutationCount();
  unsigned nodeCount = fromOrder.size();
  for (unsigned permutation=0; permutation<permutationCount; permutation++)
  {
    bool matches = true;
    for (unsigned fromIndex=0; fromIndex<nodeCount; fromIndex++)
    {
      unsigned toIndex = cellTopo->getNodePermutation(permutation, fromIndex);
      if (fromOrder[fromIndex] != toOrder[toIndex])
      {
        matches = false;
        break;
      }
    }
    if (matches) return permutation;
  }
  cout << "No matching permutation found.\n";
  Camellia::print("fromOrder", fromOrder);
  Camellia::print("toOrder", toOrder);

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No matching permutation found");
  return permutationCount; // an impossible (out of bounds) answer: this line just to satisfy compilers that warn about missing return values.
}

unsigned CamelliaCellTools::permutationMatchingOrder( const shards::CellTopology &cellTopo, const vector<unsigned> &fromOrder, const vector<unsigned> &toOrder)
{
  if (cellTopo.getDimension() == 0)
  {
    return 0;
  }
  unsigned permutationCount = cellTopo.getNodePermutationCount();
  unsigned nodeCount = fromOrder.size();
  for (unsigned permutation=0; permutation<permutationCount; permutation++)
  {
    bool matches = true;
    for (unsigned fromIndex=0; fromIndex<nodeCount; fromIndex++)
    {
      unsigned toIndex = cellTopo.getNodePermutation(permutation, fromIndex);
      if (fromOrder[fromIndex] != toOrder[toIndex])
      {
        matches = false;
        break;
      }
    }
    if (matches) return permutation;
  }
  cout << "No matching permutation found.\n";
  Camellia::print("fromOrder", fromOrder);
  Camellia::print("toOrder", toOrder);

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No matching permutation found");
  return permutationCount; // an impossible (out of bounds) answer: this line just to satisfy compilers that warn about missing return values.
}

unsigned CamelliaCellTools::permutationComposition( const shards::CellTopology &shardsTopo, unsigned a_permutation, unsigned b_permutation )
{
  CellTopoPtr cellTopo = Camellia::CellTopology::cellTopology(shardsTopo);
  return permutationComposition(cellTopo, a_permutation, b_permutation);
}

unsigned CamelliaCellTools::permutationComposition( CellTopoPtr cellTopo, unsigned a_permutation, unsigned b_permutation )
{
  // returns the permutation ordinal for a composed with b -- the lookup table is determined in a fairly brute force way (treating CellTopo as a black box), but we just do this once per topology.

  typedef CellTopologyKey CellTopoKey;
  typedef unsigned Permutation;
  typedef pair<Permutation, Permutation> PermutationPair;
  static map< CellTopoKey, map< PermutationPair, Permutation > > compositionMap;

  if (cellTopo->getKey() == CellTopology::point()->getKey())
  {
    if ((a_permutation==0) && (b_permutation==0))
    {
      return 0;
    }
  }

  if (compositionMap.find(cellTopo->getKey()) == compositionMap.end())   // build lookup table
  {
    int permCount = cellTopo->getNodePermutationCount();
    int nodeCount = cellTopo->getNodeCount();
    vector<unsigned> identityOrder;
    for (unsigned node=0; node<nodeCount; node++)
    {
      identityOrder.push_back(node);
    }
    for (int i=0; i<permCount; i++)
    {
      for (int j=0; j<permCount; j++)
      {
        vector<unsigned> composedOrder(nodeCount);
        PermutationPair ijPair = make_pair(i,j);
        for (unsigned node=0; node<nodeCount; node++)
        {
          unsigned j_of_node = cellTopo->getNodePermutation(j, node);
          unsigned i_of_j_of_node = cellTopo->getNodePermutation(i, j_of_node);
          composedOrder[node] = i_of_j_of_node;
        }
        compositionMap[cellTopo->getKey()][ijPair] = permutationMatchingOrder(cellTopo, identityOrder, composedOrder);
      }
    }
  }
  PermutationPair abPair = make_pair(a_permutation, b_permutation);

  if (compositionMap[cellTopo->getKey()].find(abPair) == compositionMap[cellTopo->getKey()].end())
  {
    cout << "Permutation pair not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Permutation pair not found");
  }
  return compositionMap[cellTopo->getKey()][abPair];
}

unsigned CamelliaCellTools::permutationFromSubsubcellToParent(CellTopoPtr cellTopo, unsigned subcdim, unsigned subcord,
    unsigned subsubcdim, unsigned subsubcord)
{
  if ((cellTopo->getDimension()==subcdim) && (subcord == 0)) // subcell and cellTopo are the same
  {
    return 0;
  }
  if (subsubcdim == 0) // vertex
  {
    return 0;
  }

  CellTopoPtr subcell = cellTopo->getSubcell(subcdim, subcord);
  int subsubcellNodeCount = subcell->getNodeCount(subsubcdim,subsubcord);
  vector<unsigned> subcellOrder(subsubcellNodeCount), parentOrder(subsubcellNodeCount);
  unsigned subsubcordInParent = subcellOrdinalMap(cellTopo, subcdim, subcord, subsubcdim, subsubcord);
  for (unsigned node=0; node<subsubcellNodeCount; node++)
  {
    parentOrder[node] = cellTopo->getNodeMap(subsubcdim, subsubcordInParent, node);
    unsigned nodeInSubcell = subcell->getNodeMap(subsubcdim, subsubcord, node);
    subcellOrder[node] = cellTopo->getNodeMap(subcdim, subcord, nodeInSubcell);
  }
  CellTopoPtr subsubcell = subcell->getSubcell(subsubcdim, subsubcord);
  return permutationMatchingOrder(subsubcell, subcellOrder, parentOrder);
}

unsigned CamelliaCellTools::permutationInverse( CellTopoPtr cellTopo, unsigned permutation )
{
  // 2-12-15: the code below copied from the version of permutationInverse that takes a shards::CellTopology as argument,
  //          modified slightly to accommodate Camellia::CellTopology
  typedef unsigned Permutation;
  static map< CellTopologyKey, map< Permutation, Permutation > > inverseMap;

  if (permutation==0) return 0;  // identity

  if (inverseMap.find(cellTopo->getKey()) == inverseMap.end())   // build lookup table
  {
    int permCount = cellTopo->getNodePermutationCount();
    int nodeCount = cellTopo->getNodeCount();
    vector<unsigned> identityOrder;
    for (unsigned node=0; node<nodeCount; node++)
    {
      identityOrder.push_back(node);
    }
    for (int i=0; i<permCount; i++)
    {
      vector<unsigned> inverseOrder(nodeCount);
      for (unsigned node=0; node<nodeCount; node++)
      {
        unsigned i_inverse_of_node = cellTopo->getNodePermutationInverse(i, node);
        inverseOrder[node] = i_inverse_of_node;
      }
      inverseMap[cellTopo->getKey()][i] = permutationMatchingOrder(cellTopo, identityOrder, inverseOrder);
    }
    if (cellTopo->getKey() == CellTopology::point()->getKey())
    {
      // for consistency of interface, we treat this a bit differently than shards -- we support the identity permutation
      // (we also consider a Node to have one node, index 0)
      inverseMap[cellTopo->getKey()][0] = 0;
    }
  }

  if (inverseMap[cellTopo->getKey()].find(permutation) == inverseMap[cellTopo->getKey()].end())
  {
    cout << "Permutation not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Permutation not found");
  }

  return inverseMap[cellTopo->getKey()][permutation];
}

unsigned CamelliaCellTools::permutationInverse( const shards::CellTopology &cellTopo, unsigned permutation )
{
  // returns the permutation ordinal for the inverse of this permutation -- the lookup table is determined in a fairly brute force way (treating CellTopo as a black box), but we just do this once per topology.  (CellTopology lets you execute an inverse, but doesn't give any way to determine the ordinal of the inverse.)

  typedef unsigned CellTopoKey;
  typedef unsigned Permutation;
  static map< CellTopoKey, map< Permutation, Permutation > > inverseMap;

  if (permutation==0) return 0;  // identity

  if (inverseMap.find(cellTopo.getKey()) == inverseMap.end())   // build lookup table
  {
    int permCount = cellTopo.getNodePermutationCount();
    int nodeCount = cellTopo.getNodeCount();
    vector<unsigned> identityOrder;
    for (unsigned node=0; node<nodeCount; node++)
    {
      identityOrder.push_back(node);
    }
    for (int i=0; i<permCount; i++)
    {
      vector<unsigned> inverseOrder(nodeCount);
      for (unsigned node=0; node<nodeCount; node++)
      {
        unsigned i_inverse_of_node = cellTopo.getNodePermutationInverse(i, node);
        inverseOrder[node] = i_inverse_of_node;
      }
      inverseMap[cellTopo.getKey()][i] = permutationMatchingOrder(cellTopo, identityOrder, inverseOrder);
    }
    if (cellTopo.getKey() == shards::Node::key)
    {
      // for consistency of interface, we treat this a bit differently than shards -- we support the identity permutation
      // (we also consider a Node to have one node, index 0)
      inverseMap[cellTopo.getKey()][0] = 0;
    }
  }

  if (inverseMap[cellTopo.getKey()].find(permutation) == inverseMap[cellTopo.getKey()].end())
  {
    cout << "Permutation not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Permutation not found");
  }

  //    // debugging line added because I suspect that in our tests we aren't running into a case where we have a permutation which is not its own inverse...
  //    if (inverseMap[cellTopo.getKey()][permutation] != permutation) {
  //      cout << "permutation encountered which is not its own inverse.\n";
  //    }

  return inverseMap[cellTopo.getKey()][permutation];
}

// ! Note that permutedPoints container must be different from refPoints.
void CamelliaCellTools::permutedReferenceCellPoints(const shards::CellTopology &cellTopo, unsigned int permutation,
    const FieldContainer<double> &refPoints, FieldContainer<double> &permutedPoints)
{
  FieldContainer<double> permutedNodes(cellTopo.getNodeCount(),cellTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(permutedNodes, cellTopo, permutation);

  permutedNodes.resize(1,permutedNodes.dimension(0), permutedNodes.dimension(1));
  int whichCell = 0;
  Intrepid::CellTools<double>::mapToPhysicalFrame(permutedPoints,refPoints,permutedNodes,cellTopo, whichCell);
}

void CamelliaCellTools::permutedReferenceCellPoints(CellTopoPtr cellTopo, unsigned int permutation,
    const FieldContainer<double> &refPoints, FieldContainer<double> &permutedPoints)
{
  if (cellTopo->getDimension()==0)
  {
    permutedPoints = refPoints;
    return;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(refPoints.dimension(1) != cellTopo->getDimension(), std::invalid_argument, "refPoints must have shape (P,D), where D = cellTopo->getDimension()");
  FieldContainer<double> permutedNodes(cellTopo->getNodeCount(),cellTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(permutedNodes, cellTopo, permutation);

  permutedNodes.resize(1,permutedNodes.dimension(0), permutedNodes.dimension(1));
  int whichCell = 0;
  CamelliaCellTools::mapToPhysicalFrame(permutedPoints,refPoints,permutedNodes,cellTopo, whichCell);
}

void CamelliaCellTools::setJacobian(FieldContainer<double> &jacobian, const FieldContainer<double> &points, const FieldContainer<double> &cellWorkset, CellTopoPtr cellTopo, const int &whichCell)
{
  int spaceDim  = (int)cellTopo->getDimension();
  int numCells  = cellWorkset.dimension(0);
  //points can be rank-2 (P,D), or rank-3 (C,P,D)
  int numPoints = (points.rank() == 2) ? points.dimension(0) : points.dimension(1);

  // Jacobian is computed using gradients of an appropriate H(grad) basis function, nodalBasis

  BasisPtr shardsNodalBasis = BasisFactory::basisFactory()->getNodalBasisForCellTopology(cellTopo->getShardsTopology().getKey());

  BasisPtr lineNodalBasis = BasisFactory::basisFactory()->getNodalBasisForCellTopology(shards::Line<2>::key);

  typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;

  BasisPtr nodalBasis = shardsNodalBasis;
  bool rangeDimensionIsSum = true;
  for (int i=0; i<cellTopo->getTensorialDegree(); i++)
  {
    nodalBasis = Teuchos::rcp( new TensorBasis(nodalBasis, lineNodalBasis, rangeDimensionIsSum) );
  }

  int basisCardinality = nodalBasis -> getCardinality();
  FieldContainer<double> basisGrads(basisCardinality, numPoints, spaceDim);

  // Initialize jacobian
  jacobian.initialize(0);

  // Handle separately rank-2 (P,D) and rank-3 (C,P,D) cases of points arrays.
  switch(points.rank())
  {
  // refPoints is (P,D): a single or multiple cell jacobians computed for a single set of ref. points
  case 2:
  {
    nodalBasis -> getValues(basisGrads, points, Intrepid::OPERATOR_GRAD);

    // The outer loops select the multi-index of the Jacobian entry: cell, point, row, col
    // If whichCell = -1, all jacobians are computed, otherwise a single cell jacobian is computed
    int cellLoop = (whichCell == -1) ? numCells : 1 ;

    if(whichCell == -1)
    {
      for(int cellOrd = 0; cellOrd < cellLoop; cellOrd++)
      {
        for(int pointOrd = 0; pointOrd < numPoints; pointOrd++)
        {
          for(int row = 0; row < spaceDim; row++)
          {
            for(int col = 0; col < spaceDim; col++)
            {

              // The entry is computed by contracting the basis index. Number of basis functions and vertices must be the same.
              for(int bfOrd = 0; bfOrd < basisCardinality; bfOrd++)
              {
                jacobian(cellOrd, pointOrd, row, col) += cellWorkset(cellOrd, bfOrd, row)*basisGrads(bfOrd, pointOrd, col);
              } // bfOrd
            } // col
          } // row
        } // pointOrd
      } // cellOrd
    }
    else
    {
      for(int cellOrd = 0; cellOrd < cellLoop; cellOrd++)
      {
        for(int pointOrd = 0; pointOrd < numPoints; pointOrd++)
        {
          for(int row = 0; row < spaceDim; row++)
          {
            for(int col = 0; col < spaceDim; col++)
            {

              // The entry is computed by contracting the basis index. Number of basis functions and vertices must be the same.
              for(int bfOrd = 0; bfOrd < basisCardinality; bfOrd++)
              {
                jacobian(pointOrd, row, col) += cellWorkset(whichCell, bfOrd, row)*basisGrads(bfOrd, pointOrd, col);
              } // bfOrd
            } // col
          } // row
        } // pointOrd
      } // cellOrd
    } // if whichcell
  }// case 2
  break;

  // points is (C,P,D): multiple jacobians computed at multiple point sets, one jacobian per cell
  case 3:
  {
    // getValues requires rank-2 (P,D) input array, refPoints cannot be used as argument: need temp (P,D) array
    FieldContainer<double> tempPoints( points.dimension(1), points.dimension(2) );

    for(int cellOrd = 0; cellOrd < numCells; cellOrd++)
    {

      // Copy point set corresponding to this cell oridinal to the temp (P,D) array
      for(int pt = 0; pt < points.dimension(1); pt++)
      {
        for(int dm = 0; dm < points.dimension(2) ; dm++)
        {
          tempPoints(pt, dm) = points(cellOrd, pt, dm);
        }//dm
      }//pt

      // Compute gradients of basis functions at this set of ref. points
      nodalBasis -> getValues(basisGrads, tempPoints, Intrepid::OPERATOR_GRAD);

      // Compute jacobians for the point set corresponding to the current cellordinal
      for(int pointOrd = 0; pointOrd < numPoints; pointOrd++)
      {
        for(int row = 0; row < spaceDim; row++)
        {
          for(int col = 0; col < spaceDim; col++)
          {

            // The entry is computed by contracting the basis index. Number of basis functions and vertices must be the same
            for(int bfOrd = 0; bfOrd < basisCardinality; bfOrd++)
            {
              jacobian(cellOrd, pointOrd, row, col) += cellWorkset(cellOrd, bfOrd, row)*basisGrads(bfOrd, pointOrd, col);
            } // bfOrd
          } // col
        } // row
      } // pointOrd
    }//cellOrd
  }// case 3

  break;

  default:
    TEUCHOS_TEST_FOR_EXCEPTION( !( (points.rank() == 2) && (points.rank() == 3) ), std::invalid_argument,
                                ">>> ERROR (CamelliaCellTools::setJacobian): rank 2 or 3 required for points array. ");
  }//switch
  
  // DEBUGGING
//  if (cellTopo->getTensorialDegree() == 1)
//  {
//    cout << "Note: debugging in CamelliaCellTools::setJacobian.\n";
//    // check that Jacobian is orthogonal:
//    int timeDim = cellTopo->getDimension() - 1;
//    for (int d=0; d<timeDim; d++)
//    {
//      if (! jacobianIsOrthogonal(jacobian, d, timeDim))
//      {
//        cout << "Jacobian is not orthogonal in space dim " << d << " and time dim " << timeDim << endl;
//        cout << "Jacobian:\n" << jacobian;
//        cout << "CellWorkset:\n" << cellWorkset;
//      }
//    }
//  }
  
}

const FieldContainer<double>& CamelliaCellTools::getSubcellParametrization(const int subcellDim, CellTopoPtr parentCell)
{
  // Coefficients of the coordinate functions defining the parametrization maps are stored in
  // rank-3 arrays with dimensions (SC, PCD, COEF) where:
  //  - SC    is the subcell count of subcells with the specified dimension in the parent cell
  //  - PCD   is Parent Cell Dimension, which gives the number of coordinate functions in the map:
  //          PCD = 2 for standard 2D cells and non-standard 2D cells: shell line and beam
  //          PCD = 3 for standard 3D cells and non-standard 3D cells: shell Tri and Quad
  //  - COEF  is number of coefficients needed to specify a coordinate function:
  //          COEFF = 2 for edge parametrizations
  //          COEFF = 3 for both Quad and Tri face parametrizations. Because all Quad reference faces
  //          are affine, the coefficient of the bilinear term u*v is zero and is not stored, i.e.,
  //          3 coefficients are sufficient to store Quad face parameterization maps.
  //
  // Arrays are sized and filled only when parametrization of a particular subcell is requested
  // by setSubcellParametrization.

  static std::vector< std::map< Camellia::CellTopologyKey, FieldContainer<double> > > subcellMapsForDimension;
  if (subcellMapsForDimension.size() < subcellDim + 1) subcellMapsForDimension.resize(subcellDim + 1);

  if (subcellMapsForDimension[subcellDim].find(parentCell->getKey()) == subcellMapsForDimension[subcellDim].end())
  {
    int subcellCount = parentCell->getSubcellCount(subcellDim);
    int parentDim = parentCell->getDimension();
    int numCoefficients = subcellDim + 1; // I believe this suffices for all affine cases (all our reference cells have subcells that are affine maps of the subcell topology's reference cell.)
    FieldContainer<double> subcellMap(subcellCount,parentDim,numCoefficients);
    FieldContainer<double> parentRefNodes(parentCell->getVertexCount(), parentCell->getDimension());
    refCellNodesForTopology(parentRefNodes, parentCell);

    for (int scOrd=0; scOrd<subcellCount; scOrd++)
    {
      CellTopoPtr scTopo = parentCell->getSubcell(subcellDim, scOrd);
      int scNodeCount = scTopo->getVertexCount();
      FieldContainer<double> scTopoRefNodes(scNodeCount, scTopo->getDimension());
      refCellNodesForTopology(scTopoRefNodes, scTopo);

      // A gets built up out of identity blocks multiplied by the weights in scTopoRefNodes
      FieldContainer<double> A(scNodeCount, subcellDim+1);
      FieldContainer<double> b(scNodeCount, parentDim);

      for (int scNode = 0; scNode < scNodeCount; scNode++)
      {
        int parentNode = parentCell->getNodeMap(subcellDim, scOrd, scNode);

        // we'll solve for each parent dimension separately:
        for (int d_parent = 0; d_parent < parentDim; d_parent++)
        {
          b(scNode, d_parent) = parentRefNodes(parentNode,d_parent);

          A(scNode, 0) = 1;
          for (int d_sc = 0; d_sc < subcellDim; d_sc++)
          {
            A(scNode, d_sc + 1) = scTopoRefNodes(scNode, d_sc);
          }
        }
      }

      // In general, A will correspond to an overdetermined system.
      FieldContainer<double> x(subcellDim+1,parentDim);
      int result = SerialDenseWrapper::solveSystemLeastSquares(x, A, b);

      if (result != 0)
      {
        cout << "ERROR: getSubcellParametrization failed to solve for subcell parameters.\n";
        cout << "A:\n" << A;
        cout << "b:\n" << b;
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "solve failed");
      }

      for (int d_parent=0; d_parent<parentDim; d_parent++)
      {
        for (int coeffOrdinal=0; coeffOrdinal < numCoefficients; coeffOrdinal++)
        {
          subcellMap(scOrd,d_parent,coeffOrdinal) = x(coeffOrdinal,d_parent);
        }
      }
    }
    subcellMapsForDimension[subcellDim][parentCell->getKey()] = subcellMap;
  }
  return subcellMapsForDimension[subcellDim][parentCell->getKey()];
}

unsigned CamelliaCellTools::subcellOrdinalMap(CellTopoPtr cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcord)
{
  // maps from a subcell's ordering of its subcells (the sub-subcells) to the cell topology's ordering of those subcells.
  typedef unsigned SubcellOrdinal;
  typedef unsigned SubcellDimension;
  typedef unsigned SubSubcellOrdinal;
  typedef unsigned SubSubcellDimension;
  typedef unsigned SubSubcellOrdinalInCellTopo;
  typedef pair< SubcellDimension, SubcellOrdinal > SubcellIdentifier;    // dim, ord in cellTopo
  typedef pair< SubSubcellDimension, SubSubcellOrdinal > SubSubcellIdentifier; // dim, ord in subcell
  typedef map< SubcellIdentifier, map< SubSubcellIdentifier, SubSubcellOrdinalInCellTopo > > OrdinalMap;
  static map< CellTopologyKey, OrdinalMap > ordinalMaps;

  if (subsubcdim==subcdim)
  {
    if (subsubcord==0)   // i.e. the "subsubcell" is really just the subcell
    {
      return subcord;
    }
    else
    {
      cout << "request for subsubcell of the same dimension as subcell, but with subsubcord > 0.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "request for subsubcell of the same dimension as subcell, but with subsubcord > 0.");
    }
  }

  if (subcdim==cellTopo->getDimension())
  {
    if (subcord==0)   // i.e. the subcell is the cell itself
    {
      return subsubcord;
    }
    else
    {
      cout << "request for subcell of the same dimension as cell, but with subsubcord > 0.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "request for subcell of the same dimension as cell, but with subsubcord > 0.");
    }
  }

  CellTopologyKey key = cellTopo->getKey();
  if (ordinalMaps.find(key) == ordinalMaps.end())
  {
    // then we construct the map for this cellTopo
    OrdinalMap ordinalMap;
    unsigned sideDim = cellTopo->getDimension() - 1;
    typedef unsigned NodeOrdinal;
    map< set<NodeOrdinal>, SubcellIdentifier > subcellMap; // given set of nodes in cellTopo, what subcell is it?)

    for (unsigned d=1; d<=sideDim; d++)   // only things of dimension >= 1 will have subcells
    {
      unsigned subcellCount = cellTopo->getSubcellCount(d);
      for (unsigned subcellOrdinal=0; subcellOrdinal<subcellCount; subcellOrdinal++)
      {

        set<NodeOrdinal> nodes;
        unsigned nodeCount = cellTopo->getNodeCount(d, subcellOrdinal);
        for (NodeOrdinal subcNode=0; subcNode<nodeCount; subcNode++)
        {
          nodes.insert(cellTopo->getNodeMap(d, subcellOrdinal, subcNode));
        }
        SubcellIdentifier subcell = make_pair(d, subcellOrdinal);
        subcellMap[nodes] = subcell;

        CellTopoPtr subcellTopo = cellTopo->getSubcell(d, subcellOrdinal);
        // now, go over all the subsubcells, and look them up...
        for (unsigned subsubcellDim=0; subsubcellDim<d; subsubcellDim++)
        {
          unsigned subsubcellCount = subcellTopo->getSubcellCount(subsubcellDim);
          for (unsigned subsubcellOrdinal=0; subsubcellOrdinal<subsubcellCount; subsubcellOrdinal++)
          {
            SubSubcellIdentifier subsubcell = make_pair(subsubcellDim,subsubcellOrdinal);
            if (subsubcellDim==0)   // treat vertices separately
            {
              ordinalMap[subcell][subsubcell] = cellTopo->getNodeMap(subcell.first, subcell.second, subsubcellOrdinal);
              continue;
            }
            unsigned nodeCount = subcellTopo->getNodeCount(subsubcellDim, subsubcellOrdinal);
            set<NodeOrdinal> subcellNodes; // NodeOrdinals index into cellTopo, though!
            for (NodeOrdinal subsubcNode=0; subsubcNode<nodeCount; subsubcNode++)
            {
              NodeOrdinal subcNode = subcellTopo->getNodeMap(subsubcellDim, subsubcellOrdinal, subsubcNode);
              NodeOrdinal node = cellTopo->getNodeMap(d, subcellOrdinal, subcNode);
              subcellNodes.insert(node);
            }

            SubcellIdentifier subsubcellInCellTopo = subcellMap[subcellNodes];
            ordinalMap[ subcell ][ subsubcell ] = subsubcellInCellTopo.second;
            //              cout << "ordinalMap( (" << subcell.first << "," << subcell.second << "), (" << subsubcell.first << "," << subsubcell.second << ") ) ";
            //              cout << " ---> " << subsubcellInCellTopo.second << endl;
          }
        }
      }
    }
    ordinalMaps[key] = ordinalMap;
  }
  SubcellIdentifier subcell = make_pair(subcdim, subcord);
  SubSubcellIdentifier subsubcell = make_pair(subsubcdim, subsubcord);
  if (ordinalMaps[key][subcell].find(subsubcell) != ordinalMaps[key][subcell].end())
  {
    return ordinalMaps[key][subcell][subsubcell];
  }
  else
  {
    cout << "For topology " << cellTopo->getName() << " and subcell " << subcord << " of dim " << subcdim;
    cout << ", subsubcell " << subsubcord << " of dim " << subsubcdim << " not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "subsubcell not found");
    return -1; // NOT FOUND
  }
}

// this caches the lookup tables it builds.  Well worth it, since we'll have just one per cell topology
unsigned CamelliaCellTools::subcellOrdinalMap(const shards::CellTopology &cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcord)
{
  // maps from a subcell's ordering of its subcells (the sub-subcells) to the cell topology's ordering of those subcells.
  typedef unsigned CellTopoKey;
  typedef unsigned SubcellOrdinal;
  typedef unsigned SubcellDimension;
  typedef unsigned SubSubcellOrdinal;
  typedef unsigned SubSubcellDimension;
  typedef unsigned SubSubcellOrdinalInCellTopo;
  typedef pair< SubcellDimension, SubcellOrdinal > SubcellIdentifier;    // dim, ord in cellTopo
  typedef pair< SubSubcellDimension, SubSubcellOrdinal > SubSubcellIdentifier; // dim, ord in subcell
  typedef map< SubcellIdentifier, map< SubSubcellIdentifier, SubSubcellOrdinalInCellTopo > > OrdinalMap;
  static map< CellTopoKey, OrdinalMap > ordinalMaps;

  if (subsubcdim==subcdim)
  {
    if (subsubcord==0)   // i.e. the "subsubcell" is really just the subcell
    {
      return subcord;
    }
    else
    {
      cout << "request for subsubcell of the same dimension as subcell, but with subsubcord > 0.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "request for subsubcell of the same dimension as subcell, but with subsubcord > 0.");
    }
  }

  if (subcdim==cellTopo.getDimension())
  {
    if (subcord==0)   // i.e. the subcell is the cell itself
    {
      return subsubcord;
    }
    else
    {
      cout << "request for subcell of the same dimension as cell, but with subsubcord > 0.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "request for subcell of the same dimension as cell, but with subsubcord > 0.");
    }
  }

  CellTopoKey key = cellTopo.getKey();
  if (ordinalMaps.find(key) == ordinalMaps.end())
  {
    // then we construct the map for this cellTopo
    OrdinalMap ordinalMap;
    unsigned sideDim = cellTopo.getDimension() - 1;
    typedef unsigned NodeOrdinal;
    map< set<NodeOrdinal>, SubcellIdentifier > subcellMap; // given set of nodes in cellTopo, what subcell is it?)

    for (unsigned d=1; d<=sideDim; d++)   // only things of dimension >= 1 will have subcells
    {
      unsigned subcellCount = cellTopo.getSubcellCount(d);
      for (unsigned subcellOrdinal=0; subcellOrdinal<subcellCount; subcellOrdinal++)
      {

        set<NodeOrdinal> nodes;
        unsigned nodeCount = cellTopo.getNodeCount(d, subcellOrdinal);
        for (NodeOrdinal subcNode=0; subcNode<nodeCount; subcNode++)
        {
          nodes.insert(cellTopo.getNodeMap(d, subcellOrdinal, subcNode));
        }
        SubcellIdentifier subcell = make_pair(d, subcellOrdinal);
        subcellMap[nodes] = subcell;

        shards::CellTopology subcellTopo = cellTopo.getCellTopologyData(d, subcellOrdinal);
        // now, go over all the subsubcells, and look them up...
        for (unsigned subsubcellDim=0; subsubcellDim<d; subsubcellDim++)
        {
          unsigned subsubcellCount = subcellTopo.getSubcellCount(subsubcellDim);
          for (unsigned subsubcellOrdinal=0; subsubcellOrdinal<subsubcellCount; subsubcellOrdinal++)
          {
            SubSubcellIdentifier subsubcell = make_pair(subsubcellDim,subsubcellOrdinal);
            if (subsubcellDim==0)   // treat vertices separately
            {
              ordinalMap[subcell][subsubcell] = cellTopo.getNodeMap(subcell.first, subcell.second, subsubcellOrdinal);
              continue;
            }
            unsigned nodeCount = subcellTopo.getNodeCount(subsubcellDim, subsubcellOrdinal);
            set<NodeOrdinal> subcellNodes; // NodeOrdinals index into cellTopo, though!
            for (NodeOrdinal subsubcNode=0; subsubcNode<nodeCount; subsubcNode++)
            {
              NodeOrdinal subcNode = subcellTopo.getNodeMap(subsubcellDim, subsubcellOrdinal, subsubcNode);
              NodeOrdinal node = cellTopo.getNodeMap(d, subcellOrdinal, subcNode);
              subcellNodes.insert(node);
            }

            SubcellIdentifier subsubcellInCellTopo = subcellMap[subcellNodes];
            ordinalMap[ subcell ][ subsubcell ] = subsubcellInCellTopo.second;
            //              cout << "ordinalMap( (" << subcell.first << "," << subcell.second << "), (" << subsubcell.first << "," << subsubcell.second << ") ) ";
            //              cout << " ---> " << subsubcellInCellTopo.second << endl;
          }
        }
      }
    }
    ordinalMaps[key] = ordinalMap;
  }
  SubcellIdentifier subcell = make_pair(subcdim, subcord);
  SubSubcellIdentifier subsubcell = make_pair(subsubcdim, subsubcord);
  if (ordinalMaps[key][subcell].find(subsubcell) != ordinalMaps[key][subcell].end())
  {
    return ordinalMaps[key][subcell][subsubcell];
  }
  else
  {
    cout << "For topology " << cellTopo.getName() << " and subcell " << subcord << " of dim " << subcdim;
    cout << ", subsubcell " << subsubcord << " of dim " << subsubcdim << " not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "subsubcell not found");
    return -1; // NOT FOUND
  }
}

unsigned CamelliaCellTools::subcellReverseOrdinalMap(CellTopoPtr cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcordInCell)
{
  // looks for the ordinal of a sub-sub-cell in the subcell
  CellTopoPtr subcellTopo = cellTopo->getSubcell(subcdim, subcord);
  int subsubcCount = subcellTopo->getSubcellCount(subsubcdim);
  //    cout << "For cellTopo " << cellTopo.getName() << ", subcell dim " << subcdim << ", ordinal " << subcord;
  //    cout << ", and subsubcdim " << subsubcdim << ":\n";
  for (int subsubcOrdinal = 0; subsubcOrdinal < subsubcCount; subsubcOrdinal++)
  {
    unsigned mapped_subsubcOrdinal = subcellOrdinalMap(cellTopo, subcdim, subcord, subsubcdim, subsubcOrdinal);
    //      cout << "subsubcOrdinal " << subsubcOrdinal << " --> subcord " << mapped_subsubcOrdinal << endl;
    if (mapped_subsubcOrdinal == subsubcordInCell)
    {
      return subsubcOrdinal;
    }
  }
  cout << "ERROR: subcell " << subsubcordInCell << " not found in subcellReverseOrdinalMap.\n";
  cout << "For topology " << cellTopo->getName() << ", looking for subcell of dimension " << subsubcdim << " with ordinal " << subsubcordInCell << " in cell.\n";
  cout << "Looking in subcell of dimension " << subcdim << " with ordinal " << subcord << ".\n";

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "subcell not found in subcellReverseOrdinalMap.");
  return -1; // NOT FOUND
}

unsigned CamelliaCellTools::subcellReverseOrdinalMap(const shards::CellTopology &cellTopo, unsigned subcdim, unsigned subcord, unsigned subsubcdim, unsigned subsubcordInCell)
{
  // looks for the ordinal of a sub-sub-cell in the subcell
  const shards::CellTopology subcellTopo = cellTopo.getCellTopologyData(subcdim, subcord);
  int subsubcCount = subcellTopo.getSubcellCount(subsubcdim);
  //    cout << "For cellTopo " << cellTopo.getName() << ", subcell dim " << subcdim << ", ordinal " << subcord;
  //    cout << ", and subsubcdim " << subsubcdim << ":\n";
  for (int subsubcOrdinal = 0; subsubcOrdinal < subsubcCount; subsubcOrdinal++)
  {
    unsigned mapped_subsubcOrdinal = subcellOrdinalMap(cellTopo, subcdim, subcord, subsubcdim, subsubcOrdinal);
    //      cout << "subsubcOrdinal " << subsubcOrdinal << " --> subcord " << mapped_subsubcOrdinal << endl;
    if (mapped_subsubcOrdinal == subsubcordInCell)
    {
      return subsubcOrdinal;
    }
  }
  cout << "ERROR: subcell " << subsubcordInCell << " not found in subcellReverseOrdinalMap.\n";
  cout << "For topology " << cellTopo.getName() << ", looking for subcell of dimension " << subsubcdim << " with ordinal " << subsubcordInCell << " in cell.\n";
  cout << "Looking in subcell of dimension " << subcdim << " with ordinal " << subcord << ".\n";

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "subcell not found in subcellReverseOrdinalMap.");
  return -1; // NOT FOUND
}

void CamelliaCellTools::getTensorPoints(Intrepid::FieldContainer<double>& tensorPoints, const Intrepid::FieldContainer<double> & spatialPoints,
                                        const Intrepid::FieldContainer<double> & temporalPoints)
{
  bool hasCellRank;
  if ( spatialPoints.rank() != temporalPoints.rank() )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "temporalPoints and spatialPoints must have same rank");
  }

  if (tensorPoints.rank() == 3)
  {
    hasCellRank = true;
  }
  else if (tensorPoints.rank() == 2)
  {
    hasCellRank = false;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"unsupported rank in tensorPoints");
  }

  int numCells = hasCellRank ? tensorPoints.dimension(0) : 1;

  int pointIndex = hasCellRank ? 1 : 0;
  int spaceDimIndex = hasCellRank ? 2 : 1;
  int numPointsSpace = spatialPoints.dimension(pointIndex);
  int numPointsTime = temporalPoints.dimension(pointIndex);

  int spaceDim = spatialPoints.dimension(spaceDimIndex);
  int timeDim = temporalPoints.dimension(spaceDimIndex);

  for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
  {
    for (int timePointOrdinal=0; timePointOrdinal<numPointsTime; timePointOrdinal++)
    {
      for (int spacePointOrdinal=0; spacePointOrdinal<numPointsSpace; spacePointOrdinal++)
      {
        int spaceTimePointOrdinal = TENSOR_POINT_ORDINAL(spacePointOrdinal, timePointOrdinal, numPointsSpace);
        for (int d=0; d<spaceDim; d++)
        {
          if (hasCellRank)
          {
            tensorPoints(cellOrdinal,spaceTimePointOrdinal,d) = spatialPoints(cellOrdinal,spacePointOrdinal,d);
          }
          else
          {
            tensorPoints(spaceTimePointOrdinal,d) = spatialPoints(spacePointOrdinal,d);
          }
        }
        for (int d=spaceDim; d<spaceDim+timeDim; d++)
        {
          if (hasCellRank)
          {
            tensorPoints(cellOrdinal,spaceTimePointOrdinal,d) = temporalPoints(cellOrdinal,timePointOrdinal,d-spaceDim);
          }
          else
          {
            tensorPoints(spaceTimePointOrdinal,d) = temporalPoints(timePointOrdinal,d-spaceDim);
          }
        }
      }
    }
  }
}

// copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry
void CamelliaCellTools::mapToReferenceFrameInitGuess(      FieldContainer<double>  &        refPoints,
    const FieldContainer<double>  &        initGuess,
    const FieldContainer<double>  &        physPoints,
    BasisCachePtr basisCache)
{
  int spaceDim  = basisCache->cellTopology()->getDimension();
  int numPoints;
  int numCells=physPoints.dimension(0);

  // Temp arrays for Newton iterates and Jacobians. Resize according to rank of ref. point array
  FieldContainer<double> xOld;
  FieldContainer<double> xTem;
  FieldContainer<double> error;
  FieldContainer<double> cellCenter(spaceDim);

  // Default: map (C,P,D) array of physical pt. sets to (C,P,D) array. Requires (C,P,D) temp arrays and (C,P,D,D) Jacobians.
  numPoints = physPoints.dimension(1);
  xOld.resize(numCells, numPoints, spaceDim);
  xTem.resize(numCells, numPoints, spaceDim);
  error.resize(numCells,numPoints);
  // Set initial guess to xOld
  for(int c = 0; c < numCells; c++)
  {
    for(int p = 0; p < numPoints; p++)
    {
      for(int d = 0; d < spaceDim; d++)
      {
        xOld(c, p, d) = initGuess(c, p, d);
      }// d
    }// p
  }// c

  // Newton method to solve the equation F(refPoints) - physPoints = 0:
  // refPoints = xOld - DF^{-1}(xOld)*(F(xOld) - physPoints) = xOld + DF^{-1}(xOld)*(physPoints - F(xOld))
  for(int iter = 0; iter < INTREPID_MAX_NEWTON; ++iter)
  {

    // compute Jacobians at the old iterates and their inverses.
    xOld.resize(numPoints,spaceDim); // BasisCache expects (P,D) sizing...
    basisCache->setRefCellPoints(xOld);
    xOld.resize(numCells,numPoints,spaceDim);

    // The Newton step.
    xTem = basisCache->getPhysicalCubaturePoints();                    // xTem <- F(xOld)
    RealSpaceTools<double>::subtract( xTem, physPoints, xTem );        // xTem <- physPoints - F(xOld)
    RealSpaceTools<double>::matvec( refPoints, basisCache->getJacobianInv(), xTem);        // refPoints <- DF^{-1}( physPoints - F(xOld) )
    RealSpaceTools<double>::add( refPoints, xOld );                    // refPoints <- DF^{-1}( physPoints - F(xOld) ) + xOld

    // l2 error (Euclidean distance) between old and new iterates: |xOld - xNew|
    RealSpaceTools<double>::subtract( xTem, xOld, refPoints );
    RealSpaceTools<double>::vectorNorm( error, xTem, NORM_TWO );

    // Average L2 error for a multiple sets of physical points: error is rank-2 (C,P) array
    double totalError;
    FieldContainer<double> cellWiseError(numCells);
    // error(C,P) -> cellWiseError(P)
    RealSpaceTools<double>::vectorNorm( cellWiseError, error, NORM_ONE );
    totalError = RealSpaceTools<double>::vectorNorm( cellWiseError, NORM_ONE );

    // Stopping criterion:
    if (totalError < INTREPID_TOL)
    {
      break;
    }
    else if ( iter > INTREPID_MAX_NEWTON)
    {
      INTREPID_VALIDATE(std::cout << " CamelliaCellTools::mapToReferenceFrameInitGuess failed to converge to desired tolerance within "
                        << INTREPID_MAX_NEWTON  << " iterations\n" );
      break;
    }

    // initialize next Newton step
    xOld = refPoints;
  } // for(iter)
}

// copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry
void CamelliaCellTools::mapToReferenceFrameInitGuess(       FieldContainer<double>  &        refPoints,
    const FieldContainer<double>  &        initGuess,
    const FieldContainer<double>  &        physPoints,
    MeshTopologyViewPtr meshTopo, IndexType cellID, int cubatureDegree)
{
  CellPtr cell = meshTopo->getCell(cellID);

  BasisCachePtr basisCache = BasisCache::basisCacheForReferenceCell(cell->topology(), cubatureDegree);

  if (meshTopo->transformationFunction() != Teuchos::null)
  {
    TFunctionPtr<double> transformFunction = meshTopo->transformationFunction();
    basisCache->setTransformationFunction(transformFunction, true);
  }
  std::vector<GlobalIndexType> cellIDs;
  cellIDs.push_back(cellID);
  bool includeCellDimension = true;
  basisCache->setPhysicalCellNodes(meshTopo->physicalCellNodesForCell(cellID, includeCellDimension), cellIDs, false);

  mapToReferenceFrameInitGuess(refPoints, initGuess, physPoints, basisCache);
}

// copied from Intrepid's CellTools and specialized to allow use when we have curvilinear geometry
void CamelliaCellTools::mapToReferenceFrame(      FieldContainer<double>      &        refPoints,
    const FieldContainer<double>      &        physPoints,
    MeshTopologyViewPtr meshTopo, IndexType cellID, int cubatureDegree)
{
  CellPtr cell = meshTopo->getCell(cellID);
  CellTopoPtr cellTopo = cell->topology();
  int spaceDim  = cellTopo->getDimension();
  int numPoints;
  int numCells;

  FieldContainer<double> cellCenter(cellTopo->getDimension());

  FieldContainer<double> refCellNodes(cellTopo->getNodeCount(),cellTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);

  int nodeCount = cellTopo->getNodeCount();
  for (int node=0; node < nodeCount; node++)
  {
    for (int d=0; d<cellTopo->getDimension(); d++)
    {
      cellCenter(d) += refCellNodes(node,d) / nodeCount;
    }
  }

  // Resize initial guess depending on the rank of the physical points array
  FieldContainer<double> initGuess;

  // Default: map (C,P,D) array of physical pt. sets to (C,P,D) array. Requires (C,P,D) initial guess.
  numPoints = physPoints.dimension(1);
  numCells = 1;
  initGuess.resize(numCells, numPoints, spaceDim);
  // Set initial guess:
  for(int c = 0; c < numCells; c++)
  {
    for(int p = 0; p < numPoints; p++)
    {
      for(int d = 0; d < spaceDim; d++)
      {
        initGuess(c, p, d) = cellCenter(d);
      }// d
    }// p
  }// c

  // Call method with initial guess
  mapToReferenceFrameInitGuess(refPoints, initGuess, physPoints, meshTopo, cellID, cubatureDegree);
}

void CamelliaCellTools::mapToReferenceSubcell(FieldContainer<double>       &refSubcellPoints,
    const FieldContainer<double> &paramPoints,
    const int                     subcellDim,
    const int                     subcellOrd,
    CellTopoPtr                   parentCell)
{
  TEUCHOS_TEST_FOR_EXCEPTION(subcellDim > parentCell->getDimension(), std::invalid_argument,
                             "subcellDim cannot exceed parentCell dimension.");
  if (parentCell->getDimension() == 0)
  {
    refSubcellPoints = paramPoints; // likely a (1,1)-sized container with a zero in it, but whatever it is, we copy it
    return;
  }
  const FieldContainer<double>& subcellMap = getSubcellParametrization(subcellDim, parentCell);
  if (subcellDim != 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(paramPoints.dimension(1) != subcellDim, std::invalid_argument, "paramPoints should have shape (P,D), where D is is the subcellDim, unless subcellDim = 0");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(subcellOrd >= subcellMap.dimension(0), std::invalid_argument, "subcellOrd is out of bounds");

  int numPoints = paramPoints.dimension(0);
  int parentDim = parentCell->getDimension();
  // Apply the parametrization map to every point in parameter domain
  for(int pt = 0; pt < numPoints; pt++)
  {
    for(int  d_parent = 0; d_parent < parentDim; d_parent++)
    {
      refSubcellPoints(pt, d_parent) = subcellMap(subcellOrd, d_parent, 0);
      for (int d_sc = 0; d_sc < subcellDim; d_sc++)
      {
        refSubcellPoints(pt, d_parent) += subcellMap(subcellOrd, d_parent, d_sc + 1) * paramPoints(pt, d_sc);
      }
    }
  }
}

void CamelliaCellTools::mapToReferenceSubcell(FieldContainer<double>       &refSubcellPoints,
    const FieldContainer<double> &paramPoints,
    const int                     subcellDim,
    const int                     subcellOrd,
    const shards::CellTopology   &parentCell)
{
  // for cells that Intrepid's CellTools supports, we just use that
  int cellDim = parentCell.getDimension();
  if ((subcellDim > 0) && ((cellDim == 2) || (cellDim == 3)) )
  {
    Intrepid::CellTools<double>::mapToReferenceSubcell(refSubcellPoints, paramPoints, subcellDim, subcellOrd, parentCell);
  }
  else if (subcellDim == 0)
  {
    // just looking for a vertex; neglect paramPoints argument here
    FieldContainer<double> refCellNodes(parentCell.getNodeCount(),cellDim);
    refCellNodesForTopology(refCellNodes,parentCell);

    // we do assume that refSubcellPoints is appropriately sized
    int numPoints = refSubcellPoints.dimension(0);
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      for (int d=0; d<cellDim; d++)
      {
        refSubcellPoints(ptIndex,d) = refCellNodes(subcellOrd,d);
      }
    }
  }
  else
  {
    // TODO: add support for 4D elements.
    cout << "CamelliaCellTools::mapToReferenceSubcell -- unsupported arguments.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CamelliaCellTools::mapToReferenceSubcell -- unsupported arguments.");
  }
}

void CamelliaCellTools::pointsVectorFromFC(std::vector< vector<double> > &pointsVector, const FieldContainer<double> &pointsFC)
{
  int numPoints = pointsFC.dimension(0);
  int spaceDim = pointsFC.dimension(1);
  pointsVector.resize(numPoints);

  for (int pointOrdinal=0; pointOrdinal < numPoints; pointOrdinal++)
  {
    pointsVector[pointOrdinal].resize(spaceDim);
    for (int d=0; d<spaceDim; d++)
    {
      pointsVector[pointOrdinal][d] = pointsFC(pointOrdinal,d);
    }
  }
}

void CamelliaCellTools::pointsFCFromVector(FieldContainer<double> &pointsFC,
    const std::vector< vector<double> > &pointsVector)
{
  int numPoints = pointsVector.size();
  if (numPoints==0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "pointsVector can't be empty");
  }
  int spaceDim = pointsVector[0].size();
  pointsFC.resize(numPoints,spaceDim);
  for (int pointOrdinal=0; pointOrdinal < numPoints; pointOrdinal++)
  {
    for (int d=0; d<spaceDim; d++)
    {
      pointsFC(pointOrdinal,d) = pointsVector[pointOrdinal][d];
    }
  }
}
