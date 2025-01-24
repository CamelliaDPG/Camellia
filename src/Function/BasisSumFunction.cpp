//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "BasisFactory.h"
#include "BasisSumFunction.h"
#include "CamelliaCellTools.h"
#include "Camellia_KokkosAllocations.h"
#include "TypeDefs.h"

using namespace Camellia;

BasisSumFunction::BasisSumFunction(BasisPtr basis, const Camellia::FieldContainer<double> &basisCoefficients,
                                   BasisCachePtr overridingBasisCache, Camellia::EOperator op, bool boundaryValueOnly)
:
TFunction<double>("BasisSumFunction", BasisFactory::basisFactory()->getBasisRank(basis,op) )
{
  _coefficients = basisCoefficients;
  _overridingBasisCache = overridingBasisCache;
  if (_coefficients.rank()==1)
  {
    _coefficients.resize(1,_coefficients.dimension(0));
  }
  else if (_coefficients.rank() != 2)
  {
    cout << "basisCoefficients must be rank 1 or 2!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "basisCoefficients must be rank 1 or 2");
  }
  _boundaryValueOnly = boundaryValueOnly;
  _basis = basis; // note - _basis->getBaseCellTopology
  _op = op;
  int cardinality = basis->getCardinality();
  TEUCHOS_TEST_FOR_EXCEPTION( _coefficients.dimension(1) != cardinality,
                              std::invalid_argument,
                              "BasisSumFunction: coefficients passed in do not match cardinality of basis.");
}

// Old host implementation
//void BasisSumFunction::values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
//{
//  CHECK_VALUES_RANK(values);
//  if (_overridingBasisCache.get() != NULL)
//  {
//    // we want to transform the physical "cubature" points given by basisCache into reference points on the _overridingBasisCache,
//    // set the points there, and then replace basisCache with _overridingBasisCache
//    // we implicitly assume that the points given lie inside the physical cell nodes for _overridingBasisCache
//    // Note that this transformation does not take curvilinearity into account.
//
//    CellTopoPtr cellTopo = basisCache->cellTopology();
//    if (cellTopo->getDimension() > 0)
//    {
//      const Camellia::FieldContainer<double>* physicalCellNodes = &basisCache->getPhysicalCellNodes();
//      int numCells = physicalCellNodes->dimension(0);
//      int numNodes = physicalCellNodes->dimension(1);
//      int spaceDim = physicalCellNodes->dimension(2);
//      Camellia::FieldContainer<double> relativeReferenceCellNodes(numCells, numNodes, spaceDim);
//      CellTopoPtr domainTopo = _basis->domainTopology();
//
//      Camellia::FieldContainer<double> cellCenter(cellTopo->getDimension());
//
//      Camellia::FieldContainer<double> refCellNodes(cellTopo->getNodeCount(),cellTopo->getDimension());
//      CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);
//
//      int nodeCount = cellTopo->getNodeCount();
//      for (int node=0; node < nodeCount; node++)
//      {
//        for (int d=0; d<cellTopo->getDimension(); d++)
//        {
//          cellCenter(d) += refCellNodes(node,d) / nodeCount;
//        }
//      }
//
//      // Resize initial guess depending on the rank of the physical points array
//      Camellia::FieldContainer<double> initGuess;
//
//      // Default: map (C,P,D) array of physical pt. sets to (C,P,D) array. Requires (C,P,D) initial guess.
//      initGuess.resize(numCells, numNodes, spaceDim);
//      // Set initial guess:
//      for(int c = 0; c < numCells; c++)
//      {
//        for(int p = 0; p < numNodes; p++)
//        {
//          for(int d = 0; d < spaceDim; d++)
//          {
//            initGuess(c, p, d) = cellCenter(d);
//          }// d
//        }// p
//      }// c
//
//      CamelliaCellTools::mapToReferenceFrameInitGuess(relativeReferenceCellNodes, initGuess, *physicalCellNodes, _overridingBasisCache);
//
//      Camellia::FieldContainer<double> oneCellRelativeReferenceNodes(1,numNodes,spaceDim);
//      for (int n=0; n<numNodes; n++)
//      {
//        for (int d=0; d<spaceDim; d++)
//        {
//          oneCellRelativeReferenceNodes(0,n,d) = relativeReferenceCellNodes(0,n,d);
//        }
//      }
//      bool cachesAgreeOnSideness = basisCache->isSideCache() == _overridingBasisCache->isSideCache();
//      Camellia::FieldContainer<double> relativeReferencePoints = cachesAgreeOnSideness ? basisCache->getRefCellPointsFC() : basisCache->getSideRefCellPointsInVolumeCoordinatesFC();
//      Camellia::FieldContainer<double> refPoints(1,relativeReferencePoints.dimension(0),relativeReferencePoints.dimension(1));
//      CamelliaCellTools::mapToPhysicalFrame(refPoints, relativeReferencePoints, oneCellRelativeReferenceNodes, basisCache->cellTopology());
//      refPoints.resize(refPoints.dimension(1),refPoints.dimension(2)); // strip cell dimension
//      _overridingBasisCache->setRefCellPointsFC(refPoints, basisCache->getCubatureWeightsFC());
//      basisCache = _overridingBasisCache;
//    }
//    else // if (cellTopo->getDimension() == 0)
//    {
//      _overridingBasisCache->setRefCellPointsFC(basisCache->getRefCellPointsFC(), basisCache->getCubatureWeightsFC());
//    }
//  }
//
//  int numDofs = _basis->getCardinality();
//
//  int spaceDim = basisCache->cellTopology()->getDimension();
//
//  bool basisIsVolumeBasis = _basis->domainTopology()->getDimension() == spaceDim;
//
//  bool useCubPointsSideRefCell = basisIsVolumeBasis && basisCache->isSideCache();
//
//  constFCPtr transformedValues = basisCache->getTransformedValues(_basis, _op, useCubPointsSideRefCell);
//
//  // transformedValues has dimensions (C,F,P,[D,D])
//  // therefore, the rank of the sum is transformedValues->rank() - 3
//  int rank = transformedValues->rank() - 3;
//  TEUCHOS_TEST_FOR_EXCEPTION(rank != values.rank()-2, std::invalid_argument, "values rank is incorrect.");
//
//  values.initialize(0.0);
//  bool singleCoefficientVector = _coefficients.dimension(0) == 1;
//  int numCells = values.dimension(0);
//  int numPoints = values.dimension(1);
//  int entriesPerPoint = values.size() / (numCells * numPoints);
//  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
//  {
//    for (int i=0; i<numDofs; i++)
//    {
//      double weight = singleCoefficientVector ? _coefficients(0,i) : _coefficients(cellIndex,i);
//      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
//      {
//        int valueIndex = (cellIndex*numPoints + ptIndex)*entriesPerPoint;
//        int basisValueIndex = (cellIndex*numPoints*numDofs + i*numPoints + ptIndex) * entriesPerPoint;
//        double *value = &values[valueIndex];
//        const double *basisValue = &((*transformedValues)[basisValueIndex]);
//        for (int j=0; j<entriesPerPoint; j++)
//        {
//          *value++ += *basisValue++ * weight;
//        }
//      }
//    }
//  }
//}

int BasisSumFunction::numVectorComponents() const
{
  if (this->rank() == 0) return 0;
  else
  {
    return _basis->domainTopology()->getDimension();
  }
}

// New device implementation
void BasisSumFunction::values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("BasisSumFunction values()");
  if (_overridingBasisCache.get() != NULL)
  {
    // we want to transform the physical "cubature" points given by basisCache into reference points on the _overridingBasisCache,
    // set the points there, and then replace basisCache with _overridingBasisCache
    // we implicitly assume that the points given lie inside the physical cell nodes for _overridingBasisCache
    // Note that this transformation does not take curvilinearity into account.
    CellTopoPtr cellTopo = basisCache->cellTopology();
    if (_overridingBasisCache->isSideCache())
    {
      // we require a few things in this case:
      // 1. basisCache is also a side cache
      // 2. overridingBasisCache and basisCache each have cellIDs/meshes defined, and there is exactly one cellID for each.
      // 3. overridingBasisCache must belong to an ancestor of the cell for basisCache
      
      TEUCHOS_TEST_FOR_EXCEPTION(! basisCache->isSideCache(), std::invalid_argument, "if _overridingBasisCache is a side cache, then basisCache must also be a side cache.");
      
      TEUCHOS_TEST_FOR_EXCEPTION(_overridingBasisCache->cellIDs().size() != 1, std::invalid_argument, "for side impositions, _overridingBasisCache must have exactly one cell");
      
      TEUCHOS_TEST_FOR_EXCEPTION(basisCache->cellIDs().size() != 1, std::invalid_argument, "for side impositions, basisCache must have exactly one cell");
      
      const auto overridingCellID = *_overridingBasisCache->cellIDs().begin();
      const auto cellID = *basisCache->cellIDs().begin();
      
      auto meshTopo = basisCache->mesh()->getTopology();
      
      auto cell = meshTopo->getCell(cellID);
      const bool cellIsDescendant = cell->isDescendant(overridingCellID);
      
      TEUCHOS_TEST_FOR_EXCEPTION(!cellIsDescendant, std::invalid_argument, "for BasisSumFunction on side basisCache, the cell for basisCache must be a descendant of the cell for _overridingBasisCache");
      
      auto volumeRefBranch = cell->refinementBranch(overridingCellID);
      
      auto sideRefBranch = RefinementPattern::sideRefinementBranch(volumeRefBranch, basisCache->getSideIndex());
      
      const auto &refPointsFC = basisCache->getRefCellPointsFC();
      
      Camellia::FieldContainer<double> overridingRefPoints(refPointsFC.dimension(0), refPointsFC.dimension(1));
      
      RefinementPattern::mapRefCellPointsToAncestor(sideRefBranch, refPointsFC, overridingRefPoints);
      
      _overridingBasisCache->setRefCellPointsFC(overridingRefPoints, basisCache->getCubatureWeightsFC());
      basisCache = _overridingBasisCache;
    }
    else if (cellTopo->getDimension() > 0)
    {
      const Camellia::FieldContainer<double>* physicalCellNodes = &basisCache->getPhysicalCellNodes();
      const int numCells = physicalCellNodes->dimension(0);
      const int numNodes = physicalCellNodes->dimension(1);
      const int spaceDim = physicalCellNodes->dimension(2);
      Camellia::FieldContainer<double> relativeReferenceCellNodes(numCells, numNodes, spaceDim);
      CellTopoPtr domainTopo = _basis->domainTopology();

      const int nodeCount = cellTopo->getNodeCount();
      const int dim = cellTopo->getDimension();
      
      // refCellNodes
      Camellia::FieldContainer<double> refCellNodes(nodeCount, dim);
      CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);
      auto dv_refCellNodes = refCellNodes.checkoutDeviceView();
      
      // cellCenter
      Camellia::FieldContainer<double> cellCenter(dim);
      
      // TODO: Optimize by performing Kokkos parallel reduction over the array (DeviceView of cellCenter)
      for (int d = 0; d < dim; d++)
      {
        Kokkos::parallel_reduce("BasisSumFunction_cellCenter", nodeCount, KOKKOS_LAMBDA (const int node, double& lvalue)
        {
          lvalue += dv_refCellNodes(node,d) / nodeCount;
        }, cellCenter(d));
      }
      Kokkos::fence();
      refCellNodes.checkinDeviceView(false);
      auto dv_cellCenter = cellCenter.checkoutDeviceView();
      
      // Resize initial guess depending on the rank of the physical points array
      // Default: map (C,P,D) array of physical pt. sets to (C,P,D) array. Requires (C,P,D) initial guess.
      Camellia::FieldContainer<double> initGuess(numCells, numNodes, spaceDim);
      auto dv_initGuess = initGuess.checkoutDeviceView(false); // will be overwritten
      
      // Set initial guess:
      using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<2> >::array_index_type;
      Kokkos::parallel_for("BasisSumFunction_initGuess", FieldContainer<double>::getMDRangePolicy2(dv_initGuess),
      KOKKOS_LAMBDA (const array_index_type c, const array_index_type p)
      {
        for(int d = 0; d < spaceDim; d++)
        {
          dv_initGuess(c,p,d) = dv_cellCenter(d);
        }
      });
      Kokkos::fence();
      cellCenter.checkinDeviceView(false);
      initGuess.checkinDeviceView(true);

      CamelliaCellTools::mapToReferenceFrameInitGuess(relativeReferenceCellNodes, initGuess, *physicalCellNodes, _overridingBasisCache);
      auto dv_relativeReferenceCellNodes = relativeReferenceCellNodes.checkoutDeviceView();

      Camellia::FieldContainer<double> oneCellRelativeReferenceNodes(1,numNodes,spaceDim);
      auto dv_oneCellRelativeReferenceNodes = oneCellRelativeReferenceNodes.checkoutDeviceView(false); // will be overwritten
      Kokkos::parallel_for("BasisSumFunction_oneCellRelativeReferenceNodes", numNodes, KOKKOS_LAMBDA (const int n)
      {
        for (int d = 0; d < spaceDim; d++)
        {
          dv_oneCellRelativeReferenceNodes(0,n,d) = dv_relativeReferenceCellNodes(0,n,d);
        }
      });
      Kokkos::fence();
      relativeReferenceCellNodes.checkinDeviceView(false);
      oneCellRelativeReferenceNodes.checkinDeviceView(true);
      
      bool cachesAgreeOnSideness = basisCache->isSideCache() == _overridingBasisCache->isSideCache();
      Camellia::FieldContainer<double> relativeReferencePoints = cachesAgreeOnSideness ? basisCache->getRefCellPointsFC() : basisCache->getSideRefCellPointsInVolumeCoordinatesFC();
      Camellia::FieldContainer<double> refPoints(1,relativeReferencePoints.dimension(0),relativeReferencePoints.dimension(1));
      CamelliaCellTools::mapToPhysicalFrame(refPoints, relativeReferencePoints, oneCellRelativeReferenceNodes, basisCache->cellTopology());
      refPoints.resize(refPoints.dimension(1),refPoints.dimension(2)); // strip cell dimension
      _overridingBasisCache->setRefCellPointsFC(refPoints, basisCache->getCubatureWeightsFC());
      basisCache = _overridingBasisCache;
    }
    else // if (cellTopo->getDimension() == 0)
    {
      _overridingBasisCache->setRefCellPointsFC(basisCache->getRefCellPointsFC(), basisCache->getCubatureWeightsFC());
    }
  }

  const int numDofs = _basis->getCardinality();
  const int spaceDim = basisCache->cellTopology()->getDimension();

  bool basisIsVolumeBasis = _basis->domainTopology()->getDimension() == spaceDim;
  bool useCubPointsSideRefCell = basisIsVolumeBasis && basisCache->isSideCache();

  constFCPtr transformedValues = basisCache->getTransformedValues(_basis, _op, useCubPointsSideRefCell);
  
  // transformedValues has dimensions (C,F,P,[D,D])
  // therefore, the rank of the sum is transformedValues->rank() - 3
  int rank = transformedValues->rank() - 3;
  TEUCHOS_TEST_FOR_EXCEPTION(rank != values.rank()-2, std::invalid_argument, "values rank is incorrect.");
  auto dv_transformedValues = transformedValues->checkoutDeviceView();
  
  bool singleCoefficientVector = _coefficients.dimension(0) == 1;
  auto dv_coefficients = _coefficients.checkoutDeviceView();
  
  if (modType == ASSIGN)
  {
    Kokkos::deep_copy(values, 0.0); // initialize
    // Note: values.rank() <= 6, because values.rank < transformedValues.rank
    using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type;
    Kokkos::parallel_for("BasisSumFunction_values", FieldContainer<double>::getMDRangePolicy6(values),
    KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2,
                   const array_index_type i3, const array_index_type i4, const array_index_type i5)
    {
      for (int dof = 0; dof < numDofs; dof++)
      {
        double weight = singleCoefficientVector ? dv_coefficients(0,dof) : dv_coefficients(i0,dof);
        values.access(i0,i1,i2,i3,i4,i5,0) += dv_transformedValues.access(i0,dof,i1,i2,i3,i4,i5) * weight;
      }
    });
  }
  else
  {
    auto basisSumValues = FieldContainer<double>::getDeviceView(values);
    Kokkos::deep_copy(basisSumValues, 0.0); // initialize
    // Note: values.rank() <= 6, because values.rank < transformedValues.rank
    using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type;
    Kokkos::parallel_for("BasisSumFunction_values", FieldContainer<double>::getMDRangePolicy6(basisSumValues),
    KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2,
                   const array_index_type i3, const array_index_type i4, const array_index_type i5)
    {
      for (int dof = 0; dof < numDofs; dof++)
      {
        double weight = singleCoefficientVector ? dv_coefficients(0,dof) : dv_coefficients(i0,dof);
        basisSumValues.access(i0,i1,i2,i3,i4,i5,0) += dv_transformedValues.access(i0,dof,i1,i2,i3,i4,i5) * weight;
      }
    });
    Kokkos::fence();
    Kokkos::parallel_for("BasisSumFunction_assign_values", FieldContainer<double>::getMDRangePolicy6(values),
    KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2,
                   const array_index_type i3, const array_index_type i4, const array_index_type i5)
    {
      double & entry =         values.access(i0,i1,i2,i3,i4,i5,0);
      double & value = basisSumValues.access(i0,i1,i2,i3,i4,i5,0);
      assignValue(entry, modType, value);
    });
    Allocation<double>::recordDeviceViewDeallocation(basisSumValues);
  }
  _coefficients.checkinDeviceView(false);
  transformedValues->checkinDeviceView(true);
}

TFunctionPtr<double> BasisSumFunction::x()
{
  if (_op != OP_VALUE)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component evaluation only supported for BasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new BasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_X));
}
TFunctionPtr<double> BasisSumFunction::y()
{
  if (_op != OP_VALUE)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component evaluation only supported for BasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new BasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_Y));
}
TFunctionPtr<double> BasisSumFunction::z()
{
  if (_op != OP_VALUE)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "component evaluation only supported for BasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new BasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_Z));
}

TFunctionPtr<double> BasisSumFunction::dx()
{
  if (_op != OP_VALUE)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "derivatives only supported for BasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new BasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_DX));
}

TFunctionPtr<double> BasisSumFunction::dy()
{
  if (_op != OP_VALUE)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "derivatives only supported for BasisSumFunction with op = OP_VALUE");
  }
  return Teuchos::rcp( new BasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_DY));
}

TFunctionPtr<double> BasisSumFunction::dz()
{
  if (_op != OP_VALUE)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "derivatives only supported for BasisSumFunction with op = OP_VALUE");
  }
  // a bit of a hack: if the topology defined in 3D, then we'll define a derivative there...
  if (_basis->domainTopology()->getDimension() > 2)
  {
    return Teuchos::rcp( new BasisSumFunction(_basis, _coefficients, _overridingBasisCache, OP_DZ));
  }
  else
  {
    return TFunction<double>::null();
  }
}

bool BasisSumFunction::boundaryValueOnly()
{
  return _boundaryValueOnly;
}

TFunctionPtr<double> BasisSumFunction::basisSumFunction(BasisPtr basis, const Camellia::FieldContainer<double> &basisCoefficients, Camellia::EOperator op)
{
  return Teuchos::rcp( new BasisSumFunction(basis,basisCoefficients,Teuchos::null,op) );
}
