//
//  BasisCacheFactory.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 10/10/18.
//

#include "BasisCacheFactory.hpp"

#include "BasisCache.h"
#include "CamelliaCellTools.h"
#include "Camellia_matrix.hpp"
#include "Function.h"
#include "RefinementPattern.h"
#include "SerialDenseWrapper.h"

using namespace Camellia;
using namespace std;

namespace Camellia {
  template<typename Scalar>
  class UnionBasisCacheSubdomainFunctionWrapper : public Function {
    std::vector<BasisCachePtr> _subdomainCaches;
    TFunctionPtr<Scalar> _f;
  public:
    UnionBasisCacheSubdomainFunctionWrapper(TFunctionPtr<Scalar> f, const std::vector<BasisCachePtr> &subdomainCaches)
    : Function(f->rank()), _subdomainCaches(subdomainCaches), _f(f)
    {
      
    }
    
    void values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
    {
      // Note that we could avoid copies here by appropriate Intrepid::FieldContainer games using pointers and offsets into the passed-in values container
      // For the moment, though, we accept the cost for ease/certainty of implementation
      int pointOffset = 0;
      Teuchos::Array<int> dimensions(values.rank());
      values.dimensions(dimensions);
      
      int numCells = dimensions[0];
      int numTotalPoints = dimensions[1];
      
      int numValuesPerPoint = values.size() / (numCells * numTotalPoints);
      
      Teuchos::Array<int> unionMultiIndex(values.rank()); // keep track of where we are
      Teuchos::Array<int> subdomainMultiIndex(values.rank()); // keep track of where we are
      for (auto subdomainCache : _subdomainCaches)
      {
        int numPoints = subdomainCache->getRefCellPoints().dimension(0);
        dimensions[1] = numPoints;
        Intrepid::FieldContainer<Scalar> subdomainValues(dimensions);
        _f->values(subdomainValues, subdomainCache);
        
        // copy values back
        for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
        {
          unionMultiIndex[0] = cellOrdinal;
          subdomainMultiIndex[0] = cellOrdinal;
          for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
          {
            unionMultiIndex[1] = ptOrdinal + pointOffset;
            subdomainMultiIndex[1] = ptOrdinal;
            
            int unionIndex = values.getEnumeration(unionMultiIndex);
            int subdomainIndex = subdomainValues.getEnumeration(subdomainMultiIndex);
            
            for (int entryOrdinal=0; entryOrdinal<numValuesPerPoint; entryOrdinal++)
            {
              values[unionIndex+entryOrdinal] = subdomainValues[subdomainIndex+entryOrdinal];
            }
          }
        }
        
        pointOffset += numPoints;
      }
    }
  };
}

template<int D>
void multiplyMatrices(const Intrepid::FieldContainer<double> &matrices1, const Intrepid::FieldContainer<double> &matrices2, Intrepid::FieldContainer<double> &result)
{
  int numCells = matrices1.dimension(0);
  int numPoints = matrices1.dimension(1);
  
  // we allow matrix1 to be of smaller dimension; we augment with the identity before multiplying
  int matrix1TrueDim = matrices1.dimension(2);
  
  for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
  {
    for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
    {
      Matrix<double,D,D> matrix1, matrix2, resultMatrix;
      for (int d1=0; d1<matrix1TrueDim; d1++)
      {
        for (int d2=0; d2<matrix1TrueDim; d2++)
        {
          matrix1[d1][d2] = matrices1(cellOrdinal,ptOrdinal,d1,d2);
        }
      }
      for (int d1=0; d1<D; d1++)
      {
        for (int d2=matrix1TrueDim; d2<D; d2++)
        {
          matrix1[d1][d2] = 0.0;
          matrix1[d2][d1] = 0.0;
          matrix1[d2][d2] = 1.0;
        }
      }
      for (int d1=0; d1<D; d1++)
      {
        for (int d2=0; d2<D; d2++)
        {
          matrix2[d1][d2] = matrices2(cellOrdinal,ptOrdinal,d1,d2);
        }
      }
//      cout << "matrix1 = \n";
//      for (int d1=0; d1<D; d1++)
//      {
//        for (int d2=0; d2<D; d2++)
//        {
//          cout << matrix1[d1][d2] << " ";
//        }
//        cout << endl;
//      }
//      cout << "matrix2 = \n";
//      for (int d1=0; d1<D; d1++)
//      {
//        for (int d2=0; d2<D; d2++)
//        {
//          cout << matrix2[d1][d2] << " ";
//        }
//        cout << endl;
//      }
      
      resultMatrix = matrix1 * matrix2;
      
//      cout << "matrix1 * matrix2 = \n";
//      for (int d1=0; d1<D; d1++)
//      {
//        for (int d2=0; d2<D; d2++)
//        {
//          cout << resultMatrix[d1][d2] << " ";
//        }
//        cout << endl;
//      }
      
      for (int d1=0; d1<D; d1++)
      {
        for (int d2=0; d2<D; d2++)
        {
          result(cellOrdinal,ptOrdinal,d1,d2) = resultMatrix[d1][d2];
        }
      }
    }
  }
}

void multiplyMatrices(int dim, const Intrepid::FieldContainer<double> &matrices1, const Intrepid::FieldContainer<double> &matrices2, Intrepid::FieldContainer<double> &result)
{
  if (dim == 1)
  {
    multiplyMatrices<1>(matrices1, matrices2, result);
  }
  else if (dim == 2)
  {
    multiplyMatrices<2>(matrices1, matrices2, result);
  }
  else if (dim == 3)
  {
    multiplyMatrices<3>(matrices1, matrices2, result);
  }
  else if (dim == 4)
  {
    multiplyMatrices<4>(matrices1, matrices2, result);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported dim");
  }
}

template<typename Scalar>
BasisCachePtr BasisCacheFactory<Scalar>::unionBasisCacheForParentCell(RefinementPatternPtr refPattern, const std::vector<BasisCachePtr> &subdomainCaches)
{
  TEUCHOS_TEST_FOR_EXCEPTION(refPattern->numChildren() != subdomainCaches.size(), std::invalid_argument, "refPattern must have the same number of children as there are subdomain caches provided");

  CellTopoPtr parentTopo = refPattern->parentTopology();
  
  int cubDegree = 0;
  bool createSideCacheToo = false;
  BasisCachePtr unionCache = Teuchos::rcp(new BasisCache(parentTopo, cubDegree, createSideCacheToo) );
  
  // volumeDim will be used for Jacobians and physical points; topoDim will be used for ref points
  int volumeDim = subdomainCaches[0]->getJacobian().dimension(2);
  
  int topoDim = parentTopo->getDimension();
  int numPoints = 0;
  int numCells = subdomainCaches[0]->getJacobian().dimension(0);
  for (auto & basisCache : subdomainCaches)
  {
    numPoints += basisCache->getRefCellPoints().dimension(0);
    int subdomainNumCells = basisCache->getJacobian().dimension(0);
    TEUCHOS_TEST_FOR_EXCEPTION(subdomainNumCells != numCells, std::invalid_argument, "subdomainCaches must agree on the number of cells");
  }
  
  Intrepid::FieldContainer<double> parentPhysicalPoints(numCells, numPoints, volumeDim);
  Intrepid::FieldContainer<double> parentRefCellPoints(numPoints, topoDim);
  
  Intrepid::FieldContainer<double> parentCubatureWeights(numPoints);
  Intrepid::FieldContainer<double> parentJacobian(numCells,numPoints,volumeDim,volumeDim);
  Intrepid::FieldContainer<double> parentJacobianInverse(numCells,numPoints,volumeDim,volumeDim);
  Intrepid::FieldContainer<double> parentJacobianDeterminant(numCells,numPoints);
  
  auto refinementMeshTopo = refPattern->refinementMeshTopology();

  int childIDInMeshTopo = 0; // 0 is the parent; start numbering children at 1
  int pointOffset = 0;
  for (auto & basisCache : subdomainCaches)
  {
    childIDInMeshTopo++;
    
    auto & childRefCellPoints = basisCache->getRefCellPoints();
    
    int numChildPoints = childRefCellPoints.dimension(0);
    
    RefinementBranch refBranch;
    refBranch.push_back({refPattern.get(),childIDInMeshTopo-1});
    
    bool includeCellDimension = true;
    auto childNodes = refinementMeshTopo->physicalCellNodesForCell(childIDInMeshTopo,includeCellDimension);
    
    // compute Jacobian of the reference-space transformation from the reference cell for child to the refinement pattern's geometry for the child
    Intrepid::FieldContainer<double> parentToChildJacobian(numCells, numChildPoints, topoDim, topoDim);
    CamelliaCellTools::setJacobian(parentToChildJacobian, basisCache->getRefCellPoints(), childNodes, refPattern->childTopology(childIDInMeshTopo-1));
    
    // Jacobian values that basisCache has for points are for the transformation from reference cell for child to physical space
    // we want to transform these to Jacobian values for the transformation from reference cell for *parent* to physical space
    // We therefore want the *inverse* of the parent-to-child Jacobian applied to the Jacobian that basisCache has
    Intrepid::FieldContainer<double> parentToChildJacobianInverse(numCells, numChildPoints, topoDim, topoDim);
    Intrepid::FieldContainer<double> parentToChildJacobianDeterminant(numCells, numChildPoints);
    SerialDenseWrapper::determinantAndInverse(parentToChildJacobianDeterminant, parentToChildJacobianInverse, parentToChildJacobian);
    
    Intrepid::FieldContainer<double> parentJacobianForChildPoints(numCells, numChildPoints, volumeDim, volumeDim);
    Intrepid::FieldContainer<double> parentJacobianInverseForChildPoints(numCells, numChildPoints, volumeDim, volumeDim);
    Intrepid::FieldContainer<double> parentJacobianDeterminantForChildPoints(numCells, numChildPoints);
    
    multiplyMatrices(volumeDim, parentToChildJacobianInverse, basisCache->getJacobian(), parentJacobianForChildPoints);
    
    SerialDenseWrapper::determinantAndInverse(parentJacobianDeterminantForChildPoints, parentJacobianInverseForChildPoints, parentJacobianForChildPoints);
    
    Intrepid::FieldContainer<double> parentRefCellPointsForChild(numChildPoints, topoDim);
    RefinementPattern::mapRefCellPointsToAncestor(refBranch, basisCache->getRefCellPoints(), parentRefCellPointsForChild);
    
    auto & childCubatureWeights = basisCache->getCubatureWeights();
    
    for (int ptOrdinal=0; ptOrdinal<numChildPoints; ptOrdinal++)
    {
      for (int d1=0; d1<topoDim; d1++)
      {
        parentRefCellPoints(ptOrdinal+pointOffset,d1) = parentRefCellPointsForChild(ptOrdinal,d1);
      }
      parentCubatureWeights(ptOrdinal+pointOffset) = childCubatureWeights(ptOrdinal) * parentJacobianDeterminantForChildPoints(0,ptOrdinal);
    }
    
    auto & childPhysicalPoints = basisCache->getPhysicalCubaturePoints();
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int ptOrdinal=0; ptOrdinal<numChildPoints; ptOrdinal++)
      {
        parentJacobianDeterminant(cellOrdinal,ptOrdinal+pointOffset) = parentJacobianDeterminantForChildPoints(cellOrdinal,ptOrdinal);
        for (int d1=0; d1<volumeDim; d1++)
        {
          parentPhysicalPoints(cellOrdinal,ptOrdinal+pointOffset,d1) = childPhysicalPoints(cellOrdinal,ptOrdinal,d1);
          for (int d2=0; d2<volumeDim; d2++)
          {
            parentJacobian(cellOrdinal,ptOrdinal+pointOffset,d1,d2) = parentJacobianForChildPoints(cellOrdinal,ptOrdinal,d1,d2);
            parentJacobianInverse(cellOrdinal,ptOrdinal+pointOffset,d1,d2) = parentJacobianInverseForChildPoints(cellOrdinal,ptOrdinal,d1,d2);
          }
        }
      }
    }
    pointOffset += numChildPoints;
  }
  
  unionCache->setRefCellPoints(parentRefCellPoints, parentCubatureWeights);
  unionCache->setPhysicalCubaturePoints(parentPhysicalPoints);
  unionCache->setJacobian(parentJacobian);
  unionCache->setJacobianInv(parentJacobianInverse);
  unionCache->setJacobianDet(parentJacobianDeterminant);
  
  return unionCache;
}

template<typename Scalar>
TFunctionPtr<Scalar> BasisCacheFactory<Scalar>::unionBasisCacheSubdomainFunctionWrapper(TFunctionPtr<Scalar> subdomainFunction, const std::vector<BasisCachePtr> &subdomainCaches)
{
  return Teuchos::rcp( new UnionBasisCacheSubdomainFunctionWrapper<Scalar>(subdomainFunction, subdomainCaches));
}

namespace Camellia
{
  template class BasisCacheFactory<double>;
  template class UnionBasisCacheSubdomainFunctionWrapper<double>;
}
