//
//  BasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/22/13.
//
//
#include "Teuchos_TestForException.hpp"

#include "Lobatto.hpp"

namespace Camellia
{
template<class Scalar, class ArrayScalar>
int LobattoHGRAD_QuadBasis<Scalar,ArrayScalar>::dofOrdinalMap(int i, int j) const   // i the xDofOrdinal, j the yDofOrdinal
{
  int yCardinality = _degree_y + 1;
  return i * yCardinality + j;
}

template<class Scalar, class ArrayScalar>
void LobattoHGRAD_QuadBasis<Scalar,ArrayScalar>::initializeL2normValues()
{
  _legendreL2normsSquared.resize(this->_basisDegree+1);
  _lobattoL2normsSquared.resize(this->_basisDegree+1);
  _legendreL2normsSquared[0] = 0; // not actually Legendre: the squared L^2 norm of the derivative of the first Lobatto polynomial...
  for (int i=1; i<=this->_basisDegree; i++)
  {
    _legendreL2normsSquared[i] = 2.0 / (2*i-1); // the squared L^2 norm of the (i-1)th Legendre polynomial...
  }
  Lobatto<Scalar,ArrayScalar>::l2norms(_lobattoL2normsSquared,this->_basisDegree,_conforming);
  // square the L^2 norms:
  for (int i=0; i<=this->_basisDegree; i++)
  {
    _lobattoL2normsSquared[i] = _lobattoL2normsSquared[i] * _lobattoL2normsSquared[i];
  }
//    cout << "Lobatto L^2 norms squared:\n"  << _lobattoL2normsSquared;
//    cout << "Legendre L^2 norms squared:\n"  << _legendreL2normsSquared;
}

template<class Scalar, class ArrayScalar>
LobattoHGRAD_QuadBasis<Scalar,ArrayScalar>::LobattoHGRAD_QuadBasis(int degree, bool conforming)
{
  _degree_x = degree;
  _degree_y = degree;
  this->_basisDegree = degree;
  _conforming = conforming;

  this->_functionSpace = Camellia::FUNCTION_SPACE_HGRAD;
  this->_rangeDimension = 2; // 2 space dim
  this->_rangeRank = 0; // scalar
  this->_domainTopology = CellTopology::cellTopology( shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
  this->_basisCardinality = (_degree_x + 1) * (_degree_y + 1);
  initializeL2normValues();
}

template<class Scalar, class ArrayScalar>
LobattoHGRAD_QuadBasis<Scalar,ArrayScalar>::LobattoHGRAD_QuadBasis(int degree_x, int degree_y, bool conforming)
{
  this->_basisDegree = max(degree_x, degree_y);
  _degree_x = degree_x;
  _degree_y = degree_y;
  _conforming = conforming;

  this->_functionSpace = Camellia::FUNCTION_SPACE_HGRAD;
  this->_rangeDimension = 2; // 2 space dim
  this->_rangeRank = 0; // scalar
  this->_domainTopology = CellTopology::cellTopology( shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
  this->_basisCardinality = (_degree_x + 1) * (_degree_y + 1);
  initializeL2normValues();
}

template<class Scalar, class ArrayScalar>
void LobattoHGRAD_QuadBasis<Scalar,ArrayScalar>::initializeTags() const
{
  if (!_conforming)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "initializeTags() called for non-conforming Lobatto basis.");
  }
  // ordering of vertices: 0 at (-1,-1), 1 at (1,-1), 2 at (1,1), 3 at (-1,1)
  const int numVertices = 4;
  const int numEdges = 4;
  int vertexOrdinals[numVertices];
  vertexOrdinals[0] = dofOrdinalMap(0,0);
  vertexOrdinals[1] = dofOrdinalMap(1,0);
  vertexOrdinals[2] = dofOrdinalMap(1,1);
  vertexOrdinals[3] = dofOrdinalMap(0,1);
  // ordering of edges: 0 south, 1 east, 2 north, 3 west
  vector<int> edgeOrdinals[numEdges];
  for (int i=2; i<_degree_x+1; i++)
  {
    edgeOrdinals[0].push_back(dofOrdinalMap(i,0));
  }
  for (int j=2; j<_degree_y+1; j++)
  {
    edgeOrdinals[1].push_back(dofOrdinalMap(1,j));
  }
  for (int i=2; i<_degree_x+1; i++)
  {
    edgeOrdinals[2].push_back(dofOrdinalMap(i,1));
  }
  for (int j=2; j<_degree_y+1; j++)
  {
    edgeOrdinals[3].push_back(dofOrdinalMap(0,j));
  }

  vector<int> interiorOrdinals;
  for (int i=2; i<_degree_x+1; i++)
  {
    for (int j=2; j<_degree_y+1; j++)
    {
      interiorOrdinals.push_back(dofOrdinalMap(i,j));
    }
  }

  int tagSize = 4;
  int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
  int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
  int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell

  std::vector<int> tags( tagSize * this->getCardinality() ); // flat array

  for (int vertexIndex=0; vertexIndex<numVertices; vertexIndex++)
  {
    int vertexSpaceDim = 0;
    int ordinalRelativeToSubcell = 0; // just one in the subcell
    int subcellDofTotal = 1;
    int i = vertexOrdinals[vertexIndex];
    tags[tagSize*i]   = vertexSpaceDim; // spaceDim
    tags[tagSize*i+1] = vertexIndex; // subcell ordinal
    tags[tagSize*i+2] = ordinalRelativeToSubcell;  // ordinal of the specified DoF relative to the subcell
    tags[tagSize*i+3] = subcellDofTotal;     // total number of DoFs associated with the subcell
  }

  for (int edgeIndex=0; edgeIndex<numEdges; edgeIndex++)
  {
    int numDofsForEdge = edgeOrdinals[edgeIndex].size();
    int edgeSpaceDim = 1;
    for (int edgeDofOrdinal=0; edgeDofOrdinal<numDofsForEdge; edgeDofOrdinal++)
    {
      int i = edgeOrdinals[edgeIndex][edgeDofOrdinal];
      tags[tagSize*i]   = edgeSpaceDim; // spaceDim
      tags[tagSize*i+1] = edgeIndex; // subcell ordinal
      tags[tagSize*i+2] = edgeDofOrdinal;  // ordinal of the specified DoF relative to the subcell
      tags[tagSize*i+3] = numDofsForEdge;     // total number of DoFs associated with the subcell
    }
  }

  int numDofsForInterior = interiorOrdinals.size();
  for (int interiorIndex=0; interiorIndex<numDofsForInterior; interiorIndex++)
  {
    int i=interiorOrdinals[interiorIndex];
    int interiorSpaceDim = 2;
    tags[tagSize*i]   = interiorSpaceDim; // spaceDim
    tags[tagSize*i+1] = interiorIndex; // subcell ordinal
    tags[tagSize*i+2] = interiorIndex;  // ordinal of the specified DoF relative to the subcell
    tags[tagSize*i+3] = numDofsForInterior;     // total number of DoFs associated with the subcell
  }

  // call basis-independent method (declared in IntrepidUtil.hpp) to set up the data structures
  Intrepid::setOrdinalTagData(this -> _tagToOrdinal,
                              this -> _ordinalToTag,
                              &(tags[0]),
                              this -> getCardinality(),
                              tagSize,
                              posScDim,
                              posScOrd,
                              posDfOrd);
}

template<class Scalar, class ArrayScalar>
void LobattoHGRAD_QuadBasis<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const
{
  this->CHECK_VALUES_ARGUMENTS(values,refPoints,operatorType);

  ArrayScalar lobattoValues_x(_degree_x+1), lobattoValues_y(_degree_y+1);
  ArrayScalar lobattoValues_dx(_degree_x+1), lobattoValues_dy(_degree_y+1);

  int numPoints = refPoints.dimension(0);
  for (int pointIndex=0; pointIndex < numPoints; pointIndex++)
  {
    double x = refPoints(pointIndex,0);
    double y = refPoints(pointIndex,1);

    Lobatto<Scalar,ArrayScalar>::values(lobattoValues_x,lobattoValues_dx, x,_degree_x,_conforming);
    Lobatto<Scalar,ArrayScalar>::values(lobattoValues_y,lobattoValues_dy, y,_degree_y,_conforming);
    for (int i=0; i<_degree_x+1; i++)
    {
      for (int j=0; j<_degree_y+1; j++)
      {
        int fieldIndex = dofOrdinalMap(i,j);
        double scalingFactor = _legendreL2normsSquared(i) * _lobattoL2normsSquared(j)
                               + _legendreL2normsSquared(j) * _lobattoL2normsSquared(i);
//          cout << "scaling factor squared for " << i << " " << j << ": " << scalingFactor << endl;
        if (scalingFactor==0) scalingFactor = 1; // the (0,0) scaling factor will be 0 because we're scaling according to (grad e_ij, grad e_ij)--and e_00 = 1.
        scalingFactor = sqrt(scalingFactor);

        switch (operatorType)
        {
        case Intrepid::OPERATOR_VALUE:
          values(fieldIndex,pointIndex) = lobattoValues_x(i) * lobattoValues_y(j) / scalingFactor;
          break;
        case Intrepid::OPERATOR_GRAD:
          values(fieldIndex,pointIndex,0) = lobattoValues_dx(i) * lobattoValues_y(j) / scalingFactor;
          values(fieldIndex,pointIndex,1) = lobattoValues_x(i) * lobattoValues_dy(j) / scalingFactor;
          break;
        case Intrepid::OPERATOR_CURL:
          values(fieldIndex,pointIndex,0) =  lobattoValues_x(i) * lobattoValues_dy(j) / scalingFactor;
          values(fieldIndex,pointIndex,1) = -lobattoValues_dx(i) * lobattoValues_y(j) / scalingFactor;
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported operatorType");
          break;
        }
      }
    }
  }

}
} // namespace Camellia