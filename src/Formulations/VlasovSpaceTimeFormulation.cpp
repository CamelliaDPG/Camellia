//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  VlasovSpaceTimeFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 1/11/22.
//
//

#include "VlasovSpaceTimeFormulation.h"
#include "CamelliaMemoryTools.h"
#include "CubatureFactory.h"
#include "ERKTableau.h"
#include "ExtrudedMeshTopology.h"
#include "GlobalDofAssignment.h"
#include "GMGSolver.h"
#include "MeshFactory.h"
#include "MeshTools.h"
#include "MonomialFunctions.h"
#include "MPIGlobalMinMax.h"
#include "ParameterFunction.h"
#include "PhysicalSolutionExtrusion.hpp"
#include "PreviousSolutionFunction.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "SimpleSolutionFunction.h"
#include "Solution.h"
#include "SolutionStage.h"
#include "TypeDefs.h"

#include "CamelliaDataHelpers.h"

#include "Intrepid2_BasisValues.hpp"
#include "Intrepid2_DerivedBasisFamily.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_IntegrationTools.hpp"
#include "Intrepid2_NodalBasisFamily.hpp"

using namespace Camellia;

using std::string;
using std::vector;

const string VlasovSpaceTimeFormulation::S_F = "f";

const string VlasovSpaceTimeFormulation::S_F_HAT = "\\widehat{f}";
const string VlasovSpaceTimeFormulation::S_T_N_HAT = "\\widehat{t}_n";

const string VlasovSpaceTimeFormulation::S_W = "w";

namespace
{
  //! private class determines regions in phase-space mesh where v \cdot n_x < 0
  class SpatialFilterSpatialInflow : public SpatialFilter
  {
    int _spaceDim;
    int _vDimStart;
  public:
    SpatialFilterSpatialInflow(int spaceDim, int vDimStart)
    :
    _spaceDim(spaceDim),
    _vDimStart(vDimStart)
    {}
    
    bool matchesPoints(Camellia::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) override
    {
      // TODO: Kokkos implementation
      const auto & normals = basisCache->getSideNormals(); // shape: C,P,D
      
      const auto & points = basisCache->getPhysicalCubaturePoints();
      
      const int numCells  = normals.dimension(0);
      const int numPoints = normals.dimension(1);
      FieldContainer<double> convectionComponentValues(normals.dimension(0),normals.dimension(1)); // C,P
      FieldContainer<double> convectionDottedWithNormal(normals.dimension(0),normals.dimension(1)); // C,P
      bool somePointMatches = false;
      for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
      {
        for (int pointOrdinal=0; pointOrdinal < numPoints; pointOrdinal++)
        {
          double v_dot_n = 0.0;
          for (int d=0; d<_spaceDim; d++)
          {
            double v_d = points(cellOrdinal,pointOrdinal,_vDimStart + d);
            double n_d = normals(cellOrdinal,pointOrdinal,d);
            v_dot_n += v_d * n_d;
          }
          if (v_dot_n < 0)
          {
            pointsMatch(cellOrdinal,pointOrdinal) = true;
            somePointMatches = true;
          }
          else
          {
            pointsMatch(cellOrdinal,pointOrdinal) = false;
          }
        }
      }
      
      return somePointMatches;
    }
    
    void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double>) override
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Not supported");
    }
  };

  Intrepid2::Data<double, DeviceType> getJacobianData(Teuchos::RCP<BasisCache> basisCache, int numConfigSpaceDims)
  {
    DeviceView<double> data;
    const int rank = 4; // C,P,D,D
    const int CELL_DIM  = 0;
    const int POINT_DIM = 1;
    const int D1_DIM    = 2;
    const int D2_DIM    = 3;
    
    const int spaceDim        = basisCache->getSpaceDim();
    const int numCellsWorkset = basisCache->cellIDs().size();
    const int pointsPerCell   = basisCache->getRefCellPoints().extent_int(0);
    int physicalSpaceDim      = numConfigSpaceDims - 1; // strip off temporal dimension
    
    using Intrepid2::CONSTANT;
    using Intrepid2::GENERAL;
    using Intrepid2::BLOCK_PLUS_DIAGONAL;
    
    Kokkos::Array<int,7>                          extents        { numCellsWorkset, pointsPerCell, spaceDim, spaceDim,        1,        1,        1 };
    Kokkos::Array<Intrepid2::DataVariationType,7> variationType  {        CONSTANT,      CONSTANT, CONSTANT, CONSTANT, CONSTANT, CONSTANT, CONSTANT };
    
    int blockPlusDiagonalLastNonDiagonal = physicalSpaceDim - 1;
    variationType[D1_DIM]    = BLOCK_PLUS_DIAGONAL;
    variationType[D2_DIM]    = BLOCK_PLUS_DIAGONAL;
    
    // check if grid is uniform AND entirely formed by orthogonal extrusions:
    bool gridIsUniform = true;
    std::vector<double> cellWidths(spaceDim,0);
    try
    {
      double tol = 1e-14;
      for (int cellOrdinal=0; cellOrdinal<numCellsWorkset; cellOrdinal++)
      {
        for (int d=0; d<spaceDim; d++)
        {
          double cellWidth = basisCache->extrusionWidth(d, cellOrdinal); // throws exception if the cell is not of uniform width in specified dimension
          if (cellWidths[d] == 0)
            cellWidths[d] = cellWidth;
          else
          {
            const double s1Abs  = fabs(cellWidths[d]);
            const double s2Abs  = fabs(cellWidth);
            const double maxAbs = (s1Abs > s2Abs) ? s1Abs : s2Abs;
            const double relErr = fabs( cellWidths[d] - cellWidth ) / ( tol + maxAbs );
            if (relErr > tol)
            {
              // not uniform
              gridIsUniform = false;
              break;
            }
          }
        }
        if (!gridIsUniform) break;
      }
    }
    catch (...)
    {
      // not uniform, then
      gridIsUniform = false;
    }
    int jacobianSize = gridIsUniform ? spaceDim : physicalSpaceDim * physicalSpaceDim + spaceDim - physicalSpaceDim;
    
    bool affine = gridIsUniform; // gridIsUniform ==> affine (due to what we mean by gridIsUniform)
    if (spaceDim == 3)
    {
      // 2 orthogonal extrusions ==> affine elements
      affine = true;
    }
    
    if (affine && !gridIsUniform)
    {
      variationType[CELL_DIM]  = GENERAL;
      variationType[POINT_DIM] = CONSTANT;
    }
    
    if (gridIsUniform) variationType[CELL_DIM] = CONSTANT;
          
    if (gridIsUniform)
    {
      // diagonal Jacobian
      blockPlusDiagonalLastNonDiagonal = -1;
      
      data = DeviceView<double>("VlasovSpaceTimeIP: Jacobian data", jacobianSize);
    }
    else
    {
      variationType[CELL_DIM]  = GENERAL;
      blockPlusDiagonalLastNonDiagonal = numConfigSpaceDims - 2; // 0-based index, and temporal extrusion is diagonal
      if (affine)
      {
        // no point variation
        variationType[POINT_DIM] = CONSTANT;
        data = DeviceView<double>("VlasovSpaceTimeIP: Jacobian data", numCellsWorkset, jacobianSize);
        
      }
      else
      {
        variationType[POINT_DIM] = GENERAL;
        data = DeviceView<double>("VlasovSpaceTimeIP: Jacobian data", numCellsWorkset, pointsPerCell, jacobianSize);
      }
    }
    
    if (gridIsUniform)
    {
      auto dataHost = Kokkos::create_mirror_view(HostMemorySpace(), data);
      for (int d=0; d<spaceDim; d++)
      {
        dataHost(d) = cellWidths[d] / 2.0;
      }
      Kokkos::deep_copy(data, dataHost);
    }
    else
    {
      using CellTools = Intrepid2::CellTools<DeviceType>;
      
      using BasisFamily = Intrepid2::DerivedNodalBasisFamily<DeviceType,double,double>;
      const int linearPolyOrder = 1;
      CellTopoPtr physicalCellTopo = basisCache->cellTopology()->getBaseTopology();
      if (physicalCellTopo->getDimension() < physicalSpaceDim)
      {
        // then extrude to get the physical cell topo:
        physicalCellTopo = CellTopology::cellTopology(physicalCellTopo, physicalSpaceDim - physicalCellTopo->getDimension());
      }
      else if (physicalCellTopo->getDimension() > physicalSpaceDim)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Base cell topology has more than the expected number of dimensions");
      }
      
      auto basisForPhysicalNodes = Intrepid2::getBasis<BasisFamily>(physicalCellTopo->getShardsTopology(), Intrepid2::FUNCTION_SPACE_HGRAD, linearPolyOrder);
      
      {
        int numFields = basisForPhysicalNodes->getCardinality();
        
        auto & scratchStorage = SharedScratchStorage<double>::sharedScratchStorage();
        auto dataViewPhysicalCell = scratchStorage.checkoutScratchStorage({numCellsWorkset, pointsPerCell, physicalSpaceDim, physicalSpaceDim});
        auto fullNodesView = basisCache->getPhysicalCellNodes().checkoutDeviceView();
        auto nodesPhysicalCell = Kokkos::subview(fullNodesView, Kokkos::ALL, numFields, physicalSpaceDim);
        
        auto cellExtrusionJacobians = scratchStorage.checkoutScratchStorage({numCellsWorkset, spaceDim - physicalSpaceDim});
        auto cellExtrusionJacobiansHost = Kokkos::create_mirror_view(HostMemorySpace(), cellExtrusionJacobians);
        
        for (int cellOrdinal=0; cellOrdinal<numCellsWorkset; cellOrdinal++)
        {
          for (int d=physicalSpaceDim; d<spaceDim; d++)
          {
            cellExtrusionJacobiansHost(cellOrdinal,d) = basisCache->extrusionWidth(d-physicalSpaceDim, cellOrdinal) / 2.0;
          }
        }
        Kokkos::deep_copy(cellExtrusionJacobians, cellExtrusionJacobiansHost);
        
        TEUCHOS_TEST_FOR_EXCEPTION(basisForPhysicalNodes == Teuchos::null, std::invalid_argument, "basisForNodes must not be null");
        
        auto fullPoints = basisCache->getRefCellPoints();
        
        std::vector<int> physicalDims;
        int numDimsCaptured = 0;
        for (int d=0; d<physicalSpaceDim; d++)
        {
          physicalDims.push_back(d);
          numDimsCaptured += fullPoints.getTensorComponent(d).extent_int(1); // (P,D)
          if (numDimsCaptured == physicalSpaceDim) break;
        }
        auto points = Intrepid2::TensorPoints<double,DeviceType>(basisCache->getRefCellPoints(), physicalDims);
        std::vector<int> otherDims;
        for (int d=points.numTensorComponents(); d<spaceDim; d++)
        {
          otherDims.push_back(d);
        }
        auto extrusionPoints = Intrepid2::TensorPoints<double,DeviceType>(basisCache->getRefCellPoints(), otherDims);
        int numExtrusionPoints = extrusionPoints.extent_int(0);
        
        auto basisGradients = basisForPhysicalNodes->allocateBasisValues(points, Intrepid2::OPERATOR_GRAD);
        basisForPhysicalNodes->getValues(basisGradients, points, Intrepid2::OPERATOR_GRAD);
        
        CellTools::setJacobian(dataViewPhysicalCell, nodesPhysicalCell, basisGradients);
        
        int numPhysicalPoints = points.extent_int(0);
        
        Kokkos::MDRangePolicy< Kokkos::Rank<2> > rangePolicy({0,0},{numCellsWorkset,numPhysicalPoints});
        
        int numNondiagonalEntries = (blockPlusDiagonalLastNonDiagonal + 1) * (blockPlusDiagonalLastNonDiagonal + 1);
        // copy dataViewPhysicalCell into data, adding the extrusion (Jacobian diagonal) data
        Kokkos::parallel_for("VlasovSpaceTimeIP: set Jacobian data", rangePolicy,
        KOKKOS_LAMBDA (const int cellOrdinal, const int physicalPointOrdinal) {
          int pointIndex = physicalPointOrdinal;
          for (int extrusionPointOrdinal=0; extrusionPointOrdinal<numExtrusionPoints; extrusionPointOrdinal++)
          {
            // TODO: find a way to avoid all this redundant computation and storage in higher dimensions (same data stored for each extrusionPointOrdinal)
            for (int d1=0; d1<physicalSpaceDim; d1++)
            {
              for (int d2=0; d2<physicalSpaceDim; d2++)
              {
                const int flatIndex = d1 * (blockPlusDiagonalLastNonDiagonal + 1) + d2;
                data(cellOrdinal, pointIndex, flatIndex) = dataViewPhysicalCell(cellOrdinal,physicalPointOrdinal,d1,d2);
              }
            }
            for (int d=physicalSpaceDim; d<spaceDim; d++)
            {
              const int flatIndex = numNondiagonalEntries + d - physicalSpaceDim;
              data(cellOrdinal, pointIndex, flatIndex) = cellExtrusionJacobians(cellOrdinal, d - physicalSpaceDim);
            }
            pointIndex += numExtrusionPoints;
          }
        });
        
        scratchStorage.checkinScratchStorage(dataViewPhysicalCell);
        scratchStorage.checkinScratchStorage(cellExtrusionJacobians);
      }
    }
    
    Intrepid2::Data<double,DeviceType> jacobianData(data,rank,extents,variationType,blockPlusDiagonalLastNonDiagonal);
    
    return jacobianData;
  }

  Intrepid2::TensorData<double,DeviceType> allocateCellMeasure( const Intrepid2::Data<double,DeviceType> & jacobianDet, const Intrepid2::TensorData<double,DeviceType> & cubatureWeights )
  {
    // Output possibilities for a cubatureWeights with N components:
    // 1. For AFFINE elements (jacobianDet cell-wise constant), returns a container with N+1 tensorial components; the first component corresponds to cells
    // 2. Otherwise, returns a container with 1 tensorial component
    
    using Intrepid2::CONSTANT;
    using Intrepid2::GENERAL;
    using Intrepid2::Data;
    using Intrepid2::DataVariationType;
    using Intrepid2::TensorData;
    
    INTREPID2_TEST_FOR_EXCEPTION(cubatureWeights.rank() != 1, std::invalid_argument, "cubatureWeights container must have shape (P)");
    
    bool affine = jacobianDet.getVariationTypes()[1] == Intrepid2::CONSTANT; // affine if Jacobian does not vary in point dimension
    
    const int numTensorComponents = affine ? cubatureWeights.numTensorComponents() + 1 : 1;
    std::vector< Data<double,DeviceType> > tensorComponents(numTensorComponents);
    
    if (affine)
    {
      const int cellExtent = jacobianDet.extent_int(0);
      Kokkos::Array<Intrepid2::DataVariationType,7> cellVariationTypes {jacobianDet.getVariationTypes()[0], CONSTANT, CONSTANT, CONSTANT, CONSTANT, CONSTANT, CONSTANT};
      const int cellDataDim = jacobianDet.getDataExtent(0);
      Kokkos::Array<int,7> cellExtents{cellExtent,1,1,1,1,1,1};
      
      DeviceView<double> detDataView ("cell relative volumes", cellDataDim);
      tensorComponents[0] = Data<double,DeviceType>(detDataView,1,cellExtents,cellVariationTypes);
      
      for (int cubTensorComponent=0; cubTensorComponent<numTensorComponents-1; cubTensorComponent++)
      {
        auto cubatureComponent = cubatureWeights.getTensorComponent(cubTensorComponent);
        const auto cubatureExtents        = cubatureComponent.getExtents();
        const auto cubatureVariationTypes = cubatureComponent.getVariationTypes();
        const int numPoints               = cubatureComponent.getDataExtent(0);
        DeviceView<double> cubatureWeightView ("cubature component weights", numPoints);
        const int pointComponentRank = 1;
        tensorComponents[cubTensorComponent+1] = Data<double,DeviceType>(cubatureWeightView,pointComponentRank,cubatureExtents,cubatureVariationTypes);
      }
    }
    else
    {
      const int cellExtent = jacobianDet.extent_int(0);
      Kokkos::Array<DataVariationType,7> variationTypes {jacobianDet.getVariationTypes()[0], GENERAL, CONSTANT, CONSTANT, CONSTANT, CONSTANT, CONSTANT};
      const int cellDataDim = jacobianDet.getDataExtent(0);
      
      const int numPoints = cubatureWeights.extent_int(0);
      Kokkos::Array<int,7> extents{cellExtent,numPoints,1,1,1,1,1};
      
      DeviceView<double> cubatureWeightView;
      if (variationTypes[0] != CONSTANT)
      {
        cubatureWeightView = DeviceView<double>("cell measure", cellDataDim, numPoints);
      }
      else
      {
        cubatureWeightView = DeviceView<double>("cell measure", numPoints);
      }
      const int cellMeasureRank = 2;
      tensorComponents[0] = Data<double,DeviceType>(cubatureWeightView,cellMeasureRank,extents,variationTypes);
    }
    const bool separateFirstComponent = (numTensorComponents > 1);
    return TensorData<double,DeviceType>(tensorComponents, separateFirstComponent);
  }
    
  void computeCellMeasure( Intrepid2::TensorData<double,DeviceType> &cellMeasure,
                           const Intrepid2::Data<double,DeviceType> & jacobianDet,
                           const Intrepid2::TensorData<double,DeviceType> & cubatureWeights )
  {
    // Output possibilities for a cubatureWeights with N components:
    // 1. For AFFINE elements (jacobianDet constant on each cell), returns a container with N+1 tensorial components; the first component corresponds to cells
    // 2. Otherwise, returns a container with 1 tensorial component
    
    INTREPID2_TEST_FOR_EXCEPTION((cellMeasure.numTensorComponents() != cubatureWeights.numTensorComponents() + 1) && (cellMeasure.numTensorComponents() != 1), std::invalid_argument,
                                 "cellMeasure must either have a tensor component count of 1 or a tensor component count that is one higher than that of cubatureWeights");
    
    INTREPID2_TEST_FOR_EXCEPTION(cubatureWeights.rank() != 1, std::invalid_argument, "cubatureWeights container must have shape (P)");
    
    if (cellMeasure.numTensorComponents() == cubatureWeights.numTensorComponents() + 1)
    {
      // affine case; the first component should contain the cell volume divided by ref cell volume; this should be stored in jacobianDet
      Kokkos::deep_copy(cellMeasure.getTensorComponent(0).getUnderlyingView1(), jacobianDet.getUnderlyingView1()); // copy point-invariant data from jacobianDet to the first tensor component of cell measure container
      const int numTensorDimensions = cubatureWeights.numTensorComponents();
      for (int i=1; i<numTensorDimensions+1; i++)
      {
        Kokkos::deep_copy(cellMeasure.getTensorComponent(i).getUnderlyingView1(), cubatureWeights.getTensorComponent(i-1).getUnderlyingView1());
      }
    }
    else
    {
      auto detVaries  = jacobianDet.getVariationTypes();
      
      using Intrepid2::CONSTANT;
      
      const bool detCellVaries  = detVaries[0] != CONSTANT;
      const bool detPointVaries = detVaries[1] != CONSTANT;
      
      if (detCellVaries && detPointVaries)
      {
        auto cellMeasureData = cellMeasure.getTensorComponent(0).getUnderlyingView2();
        auto detData = jacobianDet.getUnderlyingView2();
        const int numCells = detData.extent_int(0);
        const int numPoints = detData.extent_int(1);
        INTREPID2_TEST_FOR_EXCEPTION(numCells != cellMeasureData.extent_int(0), std::invalid_argument, "cellMeasureData doesn't match jacobianDet in cell dimension");
        INTREPID2_TEST_FOR_EXCEPTION(numPoints != cellMeasureData.extent_int(1), std::invalid_argument, "cellMeasureData doesn't match jacobianDet in point dimension");

        using ExecutionSpace = typename DeviceType::execution_space;
        Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> rangePolicy({0,0},{numCells,numPoints});
        Kokkos::parallel_for(rangePolicy,
                             KOKKOS_LAMBDA(int cellOrdinal, int pointOrdinal) {
          cellMeasureData(cellOrdinal,pointOrdinal) = detData(cellOrdinal,pointOrdinal) * cubatureWeights(pointOrdinal);
        });
      }
      else if (detCellVaries && !detPointVaries)
      {
        auto cellMeasureData = cellMeasure.getTensorComponent(0).getUnderlyingView2();
        auto detData = jacobianDet.getUnderlyingView1();
        using ExecutionSpace = typename DeviceType::execution_space;
        Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>>({0,0},{detData.extent_int(0),cubatureWeights.extent_int(0)}),
        KOKKOS_LAMBDA (int cellOrdinal, int pointOrdinal) {
          cellMeasureData(cellOrdinal,pointOrdinal) = detData(cellOrdinal) * cubatureWeights(pointOrdinal);
        });
      }
      else
      {
        // constant jacobian det case
        // cell measure data has shape (P)
        auto cellMeasureData = cellMeasure.getTensorComponent(0).getUnderlyingView1();
        auto detData = jacobianDet.getUnderlyingView1();
        using ExecutionSpace = typename DeviceType::execution_space;
        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0,cellMeasureData.extent_int(0)),
        KOKKOS_LAMBDA (const int &pointOrdinal) {
          cellMeasureData(pointOrdinal) = detData(0) * cubatureWeights(pointOrdinal);
        });
      }
    }
  }

  KOKKOS_INLINE_FUNCTION int pointOrdinalFromComponents(const Kokkos::Array<int,7> &cpppExtents,
                                                        const int &i0, const int &i1, const int &i2,
                                                        const int &i3, const int &i4, const int &i5)
  {
    return i0 + cpppExtents[1] * (i1 + cpppExtents[2] * (i2 + cpppExtents[3] * (i3 + cpppExtents[4] * (i4 + cpppExtents[5] * i5))));
  }

  class VlasovSpaceTimeBF : public TBF<double>
  {
    vector<FunctionPtr> _acceleration;   // index: space dimension
    vector<double> _accelerationWeights; // index: species
    VlasovSpaceTimeFormulation & _form;
    int _numConfigSpaceDims;
    int _numVelocitySpaceDims;
  public:
    VlasovSpaceTimeBF(VarFactoryPtr varFactory, VlasovSpaceTimeFormulation & form) :
    TBF<double>( varFactory ),
    _form(form)
    {
      _numConfigSpaceDims   = form.configurationSpaceMesh()->getDimension(); // note this includes time dimension
      _numVelocitySpaceDims = form.velocityDim();
      
      _acceleration = _form.a(-1);
      for (int speciesOrdinal = 0; speciesOrdinal < _form.numSpecies(); speciesOrdinal++)
      {
        _accelerationWeights[speciesOrdinal] = form.q(speciesOrdinal) / form.m(speciesOrdinal);
      }
    }
    
    void localStiffnessMatrixAndRHS(Camellia::FieldContainer<double> &localStiffness,
                                    Camellia::FieldContainer<double> &rhsVector,
                                    TIPPtr<double> ip, BasisCachePtr ipBasisCache,
                                    TRHSPtr<double> rhs,  BasisCachePtr basisCache) override
    {
      Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation = _form.configurationSpaceFormulation();
      
      const vector<GlobalIndexType>* cellIDs = &basisCache->cellIDs();
      const int numCells = cellIDs->size();
      if (numCells != localStiffness.dimension(0))
      {
        cout << "localStiffnessMatrix requires basisCache->cellIDs() to have the same # of cells as the first dimension of localStiffness\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffnessMatrix requires basisCache->cellIDs() to have the same # of cells as the first dimension of localStiffness");
      }
      
      ElementTypePtr elemType = _form.vlasovMesh()->getElementType((*cellIDs)[0]); // we assume all cells provided are of the same type
      DofOrderingPtr trialOrder = elemType->trialOrderPtr;
      DofOrderingPtr testOrder = elemType->testOrderPtr;
      int numTestDofs = testOrder->totalDofs();
      int numTrialDofs = trialOrder->totalDofs();
      if ((numTrialDofs != localStiffness.dimension(1)) || (numTrialDofs != localStiffness.dimension(2)))
      {
        cout << "localStiffness should have dimensions (C,numTrialFields,numTrialFields).\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localStiffness should have dimensions (C,numTrialFields,numTrialFields).");
      }
      
      const int spaceDim = _numConfigSpaceDims;
      const int velocityDim = _numVelocitySpaceDims;
      
      // the di() methods are 1-based:
      const int indexOffsetSpaceOneBased    = 1;
      const int indexOffsetTimeOneBased     = 1 + spaceDim;
      const int indexOffsetVelocityOneBased = indexOffsetTimeOneBased + 1;
        
//      // The below assumes that spaceDim <= velocityDim; enforce that here:
//      TEUCHOS_TEST_FOR_EXCEPTION(spaceDim > velocityDim, std::invalid_argument, "velocityDim must be at least as large as spaceDim");
//      std::vector<FunctionPtr> v;
//      for (int vd=0; vd<velocityDim; vd++)
//      {
//        v.push_back(this->v_i(vd));
//      }
//      
//      for (int i=0; i<_numSpecies; i++)
//      {
//        VarPtr f = this->f(i);
//        VarPtr w = this->w(i);
//        VarPtr t_n_hat = this->t_n_hat(i);
//        
//        for (int d=0; d<spaceDim; d++)
//        {
//          vlasovBF->addTerm( -v[d] * f, w->di(indexOffsetSpaceOneBased + d));  // dx_d
//        }
//        vlasovBF->addTerm( -f, w->di(indexOffsetTimeOneBased)); // dt
//        
//        auto a = this->a(); // acceleration term
//        for (int vd=0; vd<velocityDim; vd++)
//        {
//          // grad v of f
//          vlasovBF->addTerm(-a[vd] * f, w->di(indexOffsetVelocityOneBased + vd));
//        }
//        
//        vlasovBF->addTerm(t_n_hat, w);
//      }
      
      // the dof ordering should have just one entry, for a single H^1 test function w
      TEUCHOS_TEST_FOR_EXCEPTION(testOrder->getVarIDs().size() != 1, std::invalid_argument, "Unexpected number of test variables");
      auto w0_ID = _form.w(0)->ID(); // we assume basis for all w_i is the same
      auto f0_ID = _form.f(0)->ID(); // we assume basis for all f_i is the same
      
      // evaluate acceleration at quadrature points
      // we could speed this up for Poisson by only evaluating at physical points
      // we could also speed it up for Maxwell by evaluating B and v separately, and doing the cross product, etc.
      
      auto refPoints = basisCache->getRefCellPoints();
      
      const int numPoints = refPoints.extent_int(0); // (P,D)
      
      auto & scratchStorage = SharedScratchStorage<double>::sharedScratchStorage();
      
//      Kokkos::Array< DeviceView<double>, 3> accelerationValues; // at most 3 components
      Kokkos::Array< Intrepid2::Data<double, DeviceType>, 3> accelerationValues; // at most 3 components
      
      auto numAccelerationComponents = _acceleration.size();
      for (int vd=0; vd<numAccelerationComponents; vd++)
      {
        accelerationValues[vd] = _acceleration[vd]->checkoutValuesData(basisCache);
        _acceleration[vd]->values(accelerationValues[vd], basisCache);
      }
      
      Kokkos::Array< DeviceView<double>, 3> velocityValues; // at most 3 components
      for (int vd=0; vd<_numVelocitySpaceDims; vd++)
      {
        velocityValues[vd] = scratchStorage.checkoutScratchStorage({numCells, numPoints}, "velocityValues");
        auto vdFxn = _form.v_i(vd);
        vdFxn->values(velocityValues[vd], ASSIGN, basisCache);
      }
      
      auto wBasis =  testOrder->getBasis(w0_ID)->intrepid2Basis();
      auto fBasis = trialOrder->getBasis(f0_ID)->intrepid2Basis();
      
      const int w_basisCardinality = wBasis->getCardinality();
      const int f_basisCardinality = fBasis->getCardinality();
      
      auto w_basisValues    = wBasis->allocateBasisValues(refPoints, Intrepid2::OPERATOR_VALUE);
      auto w_basisGradients = wBasis->allocateBasisValues(refPoints, Intrepid2::OPERATOR_GRAD);
      
      wBasis->getValues(w_basisValues,    refPoints, Intrepid2::OPERATOR_VALUE);
      wBasis->getValues(w_basisGradients, refPoints, Intrepid2::OPERATOR_GRAD);
      
      auto f_basisValues = fBasis->allocateBasisValues(refPoints, Intrepid2::OPERATOR_VALUE);
      fBasis->getValues(f_basisValues, refPoints, Intrepid2::OPERATOR_VALUE);
      
      auto jacobian       = getJacobianData(basisCache, _numConfigSpaceDims);
      auto jacobianDet    = Intrepid2::CellTools<DeviceType>::allocateJacobianDet(jacobian);
      auto jacobianInv    = Intrepid2::CellTools<DeviceType>::allocateJacobianInv(jacobian);
      auto jacobianDetInv = Intrepid2::CellTools<DeviceType>::allocateJacobianDet(jacobian);
      Intrepid2::CellTools<DeviceType>::setJacobianDet(jacobianDet, jacobian);
      Intrepid2::CellTools<DeviceType>::setJacobianInv(jacobianInv, jacobian);
      Intrepid2::CellTools<DeviceType>::setJacobianDetInv(jacobianDetInv, jacobian);
      
      using fst = Intrepid2::FunctionSpaceTools<DeviceType>;
      
      auto w_transformedBasisValues = fst::getHGRADtransformVALUE(numCells,       w_basisValues);
      auto f_transformedBasisValues = fst::getHVOLtransformVALUE (jacobianDetInv, f_basisValues);
      
      auto cellMeasures = allocateCellMeasure(jacobianDet, basisCache->getCubatureWeights());
      computeCellMeasure(cellMeasures, jacobianDet, basisCache->getCubatureWeights());
      
      //   (- f_i, v \cdot grad_x w_i + d/dt w_i + a \cdot grad_v w_i)
      // = (  f_i, (-v -1 -a) \cdot grad w_i )
      // allocate container for [v 1 a], with shape (C,P,D)
      auto gradientWeightsView = scratchStorage.checkoutScratchStorage({numCells,numPoints,spaceDim}, "Vlasov gradient vector weights");
      
      for (int speciesOrdinal=0; speciesOrdinal<_form.numSpecies(); speciesOrdinal++)
      {
        // fill container:
        using ExecutionSpace = typename DeviceType::execution_space;
        int numVelocitySpaceDims = _numVelocitySpaceDims;
        auto extents_7 = accelerationValues[0].getExtents();
        double species_a_weight = _accelerationWeights[speciesOrdinal];
        Kokkos::Array<int,6> extents_6 {extents_7[0],extents_7[1],extents_7[2],extents_7[3],extents_7[4],extents_7[5]};
        Kokkos::parallel_for(Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<6>>({0,0,0,0,0,0},extents_6),
        KOKKOS_LAMBDA (const int &cellOrdinal, const int &i0, const int &i1, const int &i2, const int &i3, const int &i4)
        {
          for (int i5=0; i5<extents_7[6]; i5++)
          {
            const int pointOrdinal = pointOrdinalFromComponents(extents_7, i0, i1, i2, i3, i4, i5);
            // velocity first (weights spatial gradient):
            for (int d=0; d<numVelocitySpaceDims; d++)
            {
              const auto & vd = velocityValues[d](cellOrdinal,pointOrdinal);
              gradientWeightsView(cellOrdinal,pointOrdinal,d) = -vd;
            }
            // temporal gradient weight is unit:
            gradientWeightsView(cellOrdinal,pointOrdinal,numVelocitySpaceDims) = -1.0;
            // acceleration weights velocity gradient
            for (int d=0; d<numAccelerationComponents; d++)
            {
              const auto & ad = accelerationValues[d](cellOrdinal, i0, i1, i2, i3, i4, i5);
              gradientWeightsView(cellOrdinal,pointOrdinal,d+numVelocitySpaceDims+1) = -ad * species_a_weight;
            }
          }
        });
        
        // compute mat-vec result between gradientWeightsView and the jacobianInv.  Be careful about the direction of the multiplication.
        
        Intrepid2::Data<double,DeviceType> gradientWeightsData(gradientWeightsView);
        
        auto transformVectorView = scratchStorage.checkoutScratchStorage({numCells,numPoints,spaceDim}, "Vlasov transform vector");
        Intrepid2::Data<double,DeviceType> transformVectorData(transformVectorView);
        
        transformVectorData.storeMatVec(jacobianInv, gradientWeightsData, true); // true: transpose jacobianInv when multiplying
        
        Intrepid2::TransformedBasisValues<double, DeviceType> w_transformedBasisGradients(transformVectorData, w_basisGradients);
        
        bool sumInto = false;
        auto integralDataFieldGrad = Intrepid2::IntegrationTools<DeviceType>::allocateIntegralData(f_transformedBasisValues, cellMeasures, w_transformedBasisGradients);
        Intrepid2::IntegrationTools<DeviceType>::integrate(integralDataFieldGrad, f_transformedBasisValues, cellMeasures, w_transformedBasisGradients, sumInto);
        
        // TODO: 1. place (field, test) values into a rectangular stiffness matrix
        // TODO: 2. compute (trace, test) values; place these in the same rectangular stiffness matrix
        // TODO: 3. place (test, test) values into a Gram matrix
        // TODO: 4. compute B^T G^-1 B (via Cholesky, e.g.), and store into square stiffness matrix
        // TODO: 5. compute RHS against enriched test space
        // TODO: 6. contract enriched RHS against optimal tests G^-1 B; copy into rhsVector.
        // TODO: 7. set options for skipping 4,6, and instead placing in a larger "mixed method"/saddle-point formulation.
      } // speciesOrdinal
      
      scratchStorage.checkinScratchStorage(gradientWeightsView);
    }
  };

  class VlasovSpaceTimeIP : public TIP<double>
  {
    vector<FunctionPtr> _acceleration;
    VlasovSpaceTimeFormulation & _form;
    int _numConfigSpaceDims;
    int _numVelocitySpaceDims;
    
    std::map<std::pair<int,int>,Intrepid2::Data<double, DeviceType> > _integralDataValueValue; // key: (basis polyOrder, numCells)
    std::map<std::pair<int,int>,Intrepid2::Data<double, DeviceType> > _integralDataGradGrad; // key: (basis polyOrder, numCells)
    
    std::function<void(int numElements, double timeG_integrate,
                                                                double time_computeCellMeasure, double time_getJacobianData, double time_configSpaceFunction, double time_intrepid2JacobianAllocAndCompute)> _gramAssemblyTimingCallback;
  public:
    VlasovSpaceTimeIP(VlasovSpaceTimeFormulation & form)
    :
    TIP<double>(),
    _form(form)
    {
      _numConfigSpaceDims   = form.configurationSpaceMesh()->getDimension(); // note this includes time dimension
      _numVelocitySpaceDims = form.velocityDim();
      
      if (form.numSpecies() != 1)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "VlasovSpaceTimeIP does not yet support multiple species");
      }
      int speciesOrdinal = 0;
      _acceleration = form.a(speciesOrdinal);
    }
    
    void computeInnerProductMatrix(Camellia::FieldContainer<double> &innerProduct,
                                   Teuchos::RCP<DofOrdering> dofOrdering,
                                   Teuchos::RCP<BasisCache> basisCache) override
    {
      // the dof ordering should have just one entry, for a single H^1 test function w
      TEUCHOS_TEST_FOR_EXCEPTION(dofOrdering->getVarIDs().size() != 1, std::invalid_argument, "Unexpected number of test variables");
      auto wID = *dofOrdering->getVarIDs().begin();
      
      Epetra_Time timer(*MPIWrapper::CommSerial());
      
      // evaluate acceleration at quadrature points
      // we could speed this up for Poisson by only evaluating at physical points
      // we could also speed it up for Maxwell by evaluating B and v separately, and doing the cross product, etc.
      
      const auto & refPoints = basisCache->getRefCellPoints();
      
      const int numCells  = innerProduct.dimension(0); // (C,F,F)
      const int numPoints = refPoints.extent_int(0); // (P,D)
      const int spaceDim  = refPoints.extent_int(1);
      
      auto & scratchStorage = SharedScratchStorage<double>::sharedScratchStorage();
      
//      Kokkos::Array< DeviceView<double>, 3> accelerationValues; // at most 3 components
      Kokkos::Array< Intrepid2::Data<double, DeviceType>, 3> accelerationValues; // at most 3 components
      
      auto numAccelerationComponents = _acceleration.size();
      for (int vd=0; vd<numAccelerationComponents; vd++)
      {
        accelerationValues[vd] = _acceleration[vd]->checkoutValuesData(basisCache);
        _acceleration[vd]->values(accelerationValues[vd], basisCache);
      }
      
      Kokkos::Array< DeviceView<double>, 3> velocityValues; // at most 3 components
      for (int vd=0; vd<_numVelocitySpaceDims; vd++)
      {
        velocityValues[vd] = scratchStorage.checkoutScratchStorage({numCells, numPoints}, "velocityValues");
        auto vdFxn = _form.v_i(vd);
        vdFxn->values(velocityValues[vd], ASSIGN, basisCache);
      }
      
      auto wBasis = dofOrdering->getBasis(wID);
//      auto wBasis = wBasisWrapper->intrepid2Basis();
      const int basisCardinality = wBasis->getCardinality();
  
      const auto & basisValues    = basisCache->getValues(wBasis, Intrepid2::OPERATOR_VALUE, false);
      const auto & basisGradients = basisCache->getValues(wBasis, Intrepid2::OPERATOR_GRAD,  false);
      
//      auto basisValues    = wBasis->allocateBasisValues(refPoints, Intrepid2::OPERATOR_VALUE);
//      auto basisGradients = wBasis->allocateBasisValues(refPoints, Intrepid2::OPERATOR_GRAD);
//      
//      wBasis->getValues(basisValues,    refPoints, Intrepid2::OPERATOR_VALUE);
//      wBasis->getValues(basisGradients, refPoints, Intrepid2::OPERATOR_GRAD);
      
      auto jacobian    = getJacobianData(basisCache, _numConfigSpaceDims);
      
      auto jacobianDet = Intrepid2::CellTools<DeviceType>::allocateJacobianDet(jacobian);
      auto jacobianInv = Intrepid2::CellTools<DeviceType>::allocateJacobianInv(jacobian);
      Intrepid2::CellTools<DeviceType>::setJacobianDet(jacobianDet, jacobian);
      Intrepid2::CellTools<DeviceType>::setJacobianInv(jacobianInv, jacobian);
      
      using fst = Intrepid2::FunctionSpaceTools<DeviceType>;
      auto transformedBasisValues    = fst::getHGRADtransformVALUE(numCells, basisValues);
      
      auto cellMeasures = allocateCellMeasure(jacobianDet, basisCache->getCubatureWeights());
      computeCellMeasure(cellMeasures, jacobianDet, basisCache->getCubatureWeights());
      
      std::pair<int,int> integralDataKey {wBasis->getDegree(),numCells};
      if (_integralDataValueValue.find(integralDataKey) == _integralDataValueValue.end())
      {
        _integralDataValueValue[integralDataKey] = Intrepid2::IntegrationTools<DeviceType>::allocateIntegralData(transformedBasisValues, cellMeasures, transformedBasisValues);
      }
      auto & integralDataValueValue = _integralDataValueValue[integralDataKey];
//      auto integralDataValueValue = Intrepid2::IntegrationTools<DeviceType>::allocateIntegralData(transformedBasisValues, cellMeasures, transformedBasisValues);
      
      double time_block1 = timer.ElapsedTime();
      timer.ResetStartTime();
      
      double coreIntegrationTime = 0;
      
      double flopEstimateValueValue = 0;
      bool sumInto = false;
      // (w1, w2)
      timer.ResetStartTime();
      Intrepid2::IntegrationTools<DeviceType>::integrate(integralDataValueValue, transformedBasisValues, cellMeasures, transformedBasisValues, sumInto, &flopEstimateValueValue);
      coreIntegrationTime = timer.ElapsedTime();
      timer.ResetStartTime();
      
      // (v \cdot grad_x w1 + d/dt w1 + a \cdot grad_v w1, v \cdot grad_x w2 + d/dt w2 + a \cdot grad_v w2)
      // allocate container for [v 1 a], with shape (C,P,D)
      auto gradientWeightsView = scratchStorage.checkoutScratchStorage({numCells,numPoints,spaceDim}, "Vlasov gradient vector weights");
      // fill container:
      using ExecutionSpace = typename DeviceType::execution_space;
      int numVelocitySpaceDims = _numVelocitySpaceDims;
      auto extents_7 = accelerationValues[0].getExtents();
      Kokkos::Array<int,6> extents_6 {extents_7[0],extents_7[1],extents_7[2],extents_7[3],extents_7[4],extents_7[5]};
      Kokkos::parallel_for(
        Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<6>>({0,0,0,0,0,0},extents_6),
        KOKKOS_LAMBDA (const int &cellOrdinal, const int &i0, const int &i1, const int &i2, const int &i3, const int &i4)
        {
          for (int i5=0; i5<extents_7[6]; i5++)
          {
            const int pointOrdinal = pointOrdinalFromComponents(extents_7, i0, i1, i2, i3, i4, i5);
            // velocity first (weights spatial gradient):
            for (int d=0; d<numVelocitySpaceDims; d++)
            {
              const auto & vd = velocityValues[d](cellOrdinal,pointOrdinal);
              gradientWeightsView(cellOrdinal,pointOrdinal,d) = vd;
            }
            // temporal gradient weight is unit:
            gradientWeightsView(cellOrdinal,pointOrdinal,numVelocitySpaceDims) = 1.0;
            // acceleration weights velocity gradient
            for (int d=0; d<numAccelerationComponents; d++)
            {
              const auto & ad = accelerationValues[d](cellOrdinal, i0, i1, i2, i3, i4, i5);
              gradientWeightsView(cellOrdinal,pointOrdinal,d+numVelocitySpaceDims+1) = ad;
            }
          }
        });
      
      // compute mat-vec result between gradientWeightsView and the jacobianInv.  Be careful about the direction of the multiplication.
      
      Intrepid2::Data<double,DeviceType> gradientWeightsData(gradientWeightsView);
      
      auto transformVectorView = scratchStorage.checkoutScratchStorage({numCells,numPoints,spaceDim}, "Vlasov transform vector");
      Intrepid2::Data<double,DeviceType> transformVectorData(transformVectorView);
      
      transformVectorData.storeMatVec(jacobianInv, gradientWeightsData, true); // true: transpose jacobianInv when multiplying
      
      Intrepid2::TransformedBasisValues<double, DeviceType> transformedBasisGradients(transformVectorData, basisGradients);
      
      if (_integralDataGradGrad.find(integralDataKey) == _integralDataGradGrad.end())
      {
        _integralDataGradGrad[integralDataKey] = Intrepid2::IntegrationTools<DeviceType>::allocateIntegralData(transformedBasisGradients, cellMeasures, transformedBasisGradients);
      }
      auto & integralDataGradGrad = _integralDataGradGrad[integralDataKey];
//      auto integralDataGradGrad = Intrepid2::IntegrationTools<DeviceType>::allocateIntegralData(transformedBasisGradients, cellMeasures, transformedBasisGradients);
      double time_block2 = timer.ElapsedTime();
      timer.ResetStartTime();
      
      sumInto = false;
      double flopEstimateGradGrad = 0;
      timer.ResetStartTime();
      Intrepid2::IntegrationTools<DeviceType>::integrate(integralDataGradGrad, transformedBasisGradients, cellMeasures, transformedBasisGradients, sumInto, &flopEstimateGradGrad);
      coreIntegrationTime += timer.ElapsedTime();
      timer.ResetStartTime();
//      std::cout << "flop estimate for (value, value): " << flopEstimateValueValue << std::endl;
//      std::cout << "flop estimate for ( grad,  grad): " << flopEstimateGradGrad   << std::endl;

      auto ipView = innerProduct.checkoutDeviceView();
      Kokkos::parallel_for(
                           Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>>({0,0,0},{numCells,basisCardinality,basisCardinality}),
      KOKKOS_LAMBDA (int cellOrdinal, int i, int j) {
//        std::cout << "integralDataValueValue(" << cellOrdinal << "," << i << "," << j << ") = " << integralDataValueValue(cellOrdinal,i,j) << std::endl;
//        std::cout << "  integralDataGradGrad(" << cellOrdinal << "," << i << "," << j << ") = " << integralDataGradGrad(cellOrdinal,i,j) << std::endl;
        ipView(cellOrdinal,i,j) = integralDataValueValue(cellOrdinal,i,j) + integralDataGradGrad(cellOrdinal,i,j);
      });
      Kokkos::fence();
      double time_block3 = timer.ElapsedTime();
      timer.ResetStartTime();
      
      innerProduct.checkinDeviceView(true); // true: deep copy to host.  If we could guarantee that caller wouldn't try to access on host, we could eliminate this.  But we can't, and probably shouldn't until we've changed the method signature here to have a DeviceView passed in.  Basically the contract for Camellia::FieldContainer is that it should always be accessible on host.

      for (int vd=0; vd<_acceleration.size(); vd++)
      {
        _acceleration[vd]->checkinValuesData(accelerationValues[vd]);
      }
      
      for (int vd=0; vd<_numVelocitySpaceDims; vd++)
      {
        scratchStorage.checkinScratchStorage(velocityValues[vd]);
      }
      scratchStorage.checkinScratchStorage(gradientWeightsView);
      scratchStorage.checkinScratchStorage(transformVectorView);
      
      double time_block4 = timer.ElapsedTime();
      
      if (_gramAssemblyTimingCallback)
      {
        _gramAssemblyTimingCallback(numCells,coreIntegrationTime, time_block1, time_block2, time_block3, time_block4);
      }
    }
    
    void setVlasovGramAssemblyTimingCallback(std::function<void(int numElements, double timeG_integrate,
                                                                double time_computeCellMeasure, double time_getJacobianData, double time_configSpaceFunction, double time_intrepid2JacobianAllocAndCompute)> &gramAssemblyTimingCallback)
    {
      _gramAssemblyTimingCallback = gramAssemblyTimingCallback;
    }
  };
}

VlasovSpaceTimeFormulation::VlasovSpaceTimeFormulation(double q, double m,
                                                       Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                       MeshTopologyPtr spatialMeshTopology,
                                                       std::vector<double> temporalVertices,
                                                       std::vector<double> velocitySpaceVertices_x,
                                                       std::vector<double> velocitySpaceVertices_y,
                                                       std::vector<double> velocitySpaceVertices_z,
                                                       const int H1Order, const int delta_k,
                                                       const int numUniformRefinements,
                                                       const EBasisFamily basisFamily,
                                                       const int maxMomentNumberForDriver,
                                                       const bool deferInitialization)
:
VlasovSpaceTimeFormulation(vector<double>{q}, vector<double>{m}, vector<string>{""},
                           configurationSpaceFormulation,
                           spatialMeshTopology,
                           temporalVertices,
                           velocitySpaceVertices_x,
                           velocitySpaceVertices_y,
                           velocitySpaceVertices_z,
                           H1Order, delta_k,
                           numUniformRefinements,
                           basisFamily,
                           maxMomentNumberForDriver,
                           deferInitialization)
{}

VlasovSpaceTimeFormulation::VlasovSpaceTimeFormulation(std::vector<double> q, std::vector<double> m, vector<string> speciesLabels,
                                                       Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                       MeshTopologyPtr spatialMeshTopology,
                                                       std::vector<double> temporalVertices,
                                                       std::vector<double> velocitySpaceVertices_x,
                                                       std::vector<double> velocitySpaceVertices_y,
                                                       std::vector<double> velocitySpaceVertices_z,
                                                       const int H1Order, const int delta_k,
                                                       const int numUniformRefinements,
                                                       const EBasisFamily basisFamily,
                                                       const int maxMomentNumberForDriver,
                                                       const bool deferInitialization)
:
_spaceDim(spatialMeshTopology->getDimension()),
_basisFamily(basisFamily),
_H1Order(H1Order),
_delta_k(delta_k),
_maxMomentNumberForDriver(maxMomentNumberForDriver),
_configurationSpaceFormulation(configurationSpaceFormulation),
_numSpecies(q.size()),
_q(q),
_m(m),
_speciesLabels(speciesLabels),
_temporalVertices(temporalVertices),
_velocitySpaceVertices_x(velocitySpaceVertices_x),
_velocitySpaceVertices_y(velocitySpaceVertices_y),
_velocitySpaceVertices_z(velocitySpaceVertices_z),
_spatialMeshTopology(spatialMeshTopology),
_numUniformRefinements(numUniformRefinements)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_q.size() != _m.size(), std::invalid_argument, "q and m arguments must agree in length");
  TEUCHOS_TEST_FOR_EXCEPTION(_numSpecies == 0, std::invalid_argument, "Must have at least one species");
  
  if (_basisFamily == BASIS_FAMILY_SERENDIPITY)
  {
    // at the moment, Intrepid2's sum factorization does not allow us to use Serendipity basis
    _useSumFactorizedAssembly = false;
  }
  
  if (_speciesLabels.size() == 0)
  {
    if (_numSpecies == 1)
    {
      _speciesLabels = {""}; // no suffix for single-species
    }
    else
    {
      for (int i=1; i<=_numSpecies; i++)
      {
        _speciesLabels.push_back(std::to_string(i));
      }
    }
  }
  _density.resize(_numSpecies);
  _momentum.resize(_numSpecies);
  _energy.resize(_numSpecies);
  
  _f_names.resize(_numSpecies);
  _w_names.resize(_numSpecies);
  _tn_hat_names.resize(_numSpecies,"");
  _f_hat_names.resize(_numSpecies, "");
  
  for (int i=0; i<_numSpecies; i++)
  {
    std::string suffix = (_speciesLabels[i].length() > 0) ? "_" + _speciesLabels[i] : "";
    _f_names[i]      = S_F + suffix;
    _w_names[i]      = S_W + suffix;
    _tn_hat_names[i] = S_T_N_HAT + suffix;
  }
  
  if (!deferInitialization)
  {
    initialize();
  }
}

MeshPtr VlasovSpaceTimeFormulation::vlasovMesh()
{
  return _vlasovSoln->mesh();
}

SolutionPtr VlasovSpaceTimeFormulation::vlasovSolution()
{
  return _vlasovSoln;
}

vector<FunctionPtr> VlasovSpaceTimeFormulation::a(int i)
{
  // returns the acceleration term for species i; if i == -1, returns an acceleration term that should be weighted by (q / m) for the species
  if (i == -1)
  {
    return _a;
  }
  else
  {
    vector<FunctionPtr> a_i(_a.size());
    for (unsigned vd=0; vd<_a.size(); vd++)
    {
      a_i[vd] = (_q[i] / _m[i]) * _a[vd];
    }
    return a_i;
  }
}

void VlasovSpaceTimeFormulation::addConfigurationSpaceBC(const int tag, VarPtr configVariable, FunctionPtr dirichletValue)
{
  if (_configurationSpaceFormulation->solution()->bc() == Teuchos::null)
  {
    // create new BC object
    _configurationSpaceFormulation->solution()->setBC(BC::bc());
  }
  _configurationSpaceFormulation->solution()->bc()->addDirichlet(configVariable, tag, dirichletValue);
  
  _configurationSpaceBCs.push_back(std::make_tuple(tag,configVariable,dirichletValue));
}

VarPtr VlasovSpaceTimeFormulation::B_configurationSpaceTrace(const int zeroBasedComp)
{
  return _configurationSpaceFormulation->B_hat(zeroBasedComp);
}

EBasisFamily VlasovSpaceTimeFormulation::basisFamily() const
{
  return _basisFamily;
}

void VlasovSpaceTimeFormulation::clearConfigurationSpaceBCs()
{
  _configurationSpaceFormulation->solution()->setBC(BC::bc());
}

void VlasovSpaceTimeFormulation::clearSolution()
{
  _vlasovSoln->clear();
}

void VlasovSpaceTimeFormulation::computeConservationDiagnostics(bool updateSolutionMoments)
{
  if (updateSolutionMoments)
  {
    for (auto & solnMoments : _solutionMoments)
    {
      solnMoments->update();
    }
  }
  
  // record conservation diagnostics
  int slabNumber = this->slabOrdinal();
  
  // ensure that our containers are big enough:
  _massIntegrals.resize(_numSpecies);
  _momentumIntegrals.resize(_numSpecies);
  _energyIntegrals.resize(_numSpecies);
  for (int i=0; i<_numSpecies; i++)
  {
    _massIntegrals[i].resize(slabNumber+1, Teuchos::null);
    _momentumIntegrals[i].resize(_spaceDim);
    for (int d=0; d<_spaceDim; d++)
    {
      _momentumIntegrals[i][d].resize(slabNumber+1, Teuchos::null);
    }
    _energyIntegrals[i].resize(slabNumber+1, Teuchos::null);
  }
  
  if (_maxMomentForConservationDiagnostics >= 0)
  {
    // create Solution object
    VarFactoryPtr varFactory = VarFactory::varFactory();
    
    vector<VarPtr> massVar(_numSpecies);
    vector<vector<VarPtr>> momentumVar(_numSpecies, vector<VarPtr>(_spaceDim));
    vector<VarPtr> energyVar(_numSpecies);
    
    std::map<int, FunctionPtr> projectionMap;
    for (int i=0; i<_numSpecies; i++)
    {
      std::string suffix = (_speciesLabels[i].length() > 0) ? "_" + _speciesLabels[i] : "";
      if (_density[i] != Teuchos::null)
      {
        massVar[i] = varFactory->fieldVar("mass" + suffix);
        projectionMap[massVar[i]->ID()] = _density[i];
      }
      for (int d=0; d<_spaceDim; d++)
      {
        if (_momentum[i][d] != Teuchos::null)
        {
          momentumVar[i][d] = varFactory->fieldVar("momentum" + suffix + std::to_string(d));
          projectionMap[momentumVar[i][d]->ID()] = _momentum[i][d];
        }
      }
      if (_energy[i] != Teuchos::null)
      {
        energyVar[i] = varFactory->fieldVar("energy" + suffix);
        projectionMap[energyVar[i]->ID()] = _energy[i];
      }
    }
    
    auto H1Order = _configurationSpaceFormulation->configurationSpaceMesh()->globalDofAssignment()->getInitialH1Order();
    auto delta_k = 0; // since we're not solving, no enrichment required.  (Also, we don't have any test variables.)
    MeshPtr conservationDiagnosticsMesh = Teuchos::rcp(new Mesh(_configurationSpaceMeshTopology, varFactory, H1Order, delta_k) );
    
    SolutionPtr diagnosticSolution = Solution::solution(conservationDiagnosticsMesh);
    diagnosticSolution->projectOntoMesh(projectionMap);
            
    FunctionPtr one = Function::constant(1.0);
    const int quadratureEnrichment = 0;
    for (int i=0; i<_numSpecies; i++)
    {
      _massIntegrals[i][slabNumber] = Teuchos::rcp( new SpatialIntegralOfSpaceTimeSolution(diagnosticSolution, massVar[i], one, quadratureEnrichment) );
      
      if (_maxMomentForConservationDiagnostics >= 1)
      {
        for (int d=0; d<_spaceDim; d++)
        {
          _momentumIntegrals[i][d][slabNumber] = Teuchos::rcp( new SpatialIntegralOfSpaceTimeSolution(diagnosticSolution, momentumVar[i][d], one, quadratureEnrichment) );
        }
      }
      
      if (_maxMomentForConservationDiagnostics >= 2)
      {
        _energyIntegrals[i][slabNumber] = Teuchos::rcp( new SpatialIntegralOfSpaceTimeSolution(diagnosticSolution, energyVar[i], one, quadratureEnrichment) );
      }
    }
  }
  
  // element-wise integrals ("local conservation", as opposed to temporally pointwise conservation)
  _elementwiseMassIntegrals.resize(_numSpecies);
  for (int i=0; i<_numSpecies; i++)
  {
    recordElementwiseConservationDiagnostics(i, _elementwiseMassIntegrals[i], slabOrdinal(), testForMassConservation(i));
    // TODO: add records for momentum, energy
  }
  
  // global measurement of the residual of the w (aka w1, in context of mimetic) equation:
  _w1ResidualNorm.resize(_numSpecies);
  for (int i=0; i<_numSpecies; i++)
  {
    auto w1 = this->w(i);
    // want to exclude all test IDs other than w1
    std::vector<int> testIDs = this->vlasovBF()->varFactory()->testIDs();
    std::set<int> otherTestIDs(testIDs.begin(),testIDs.end());
    otherTestIDs.erase(w1->ID());
    
    auto residual_w1 = _vlasovBF->testFunctional(_vlasovSoln,false,SolutionComponent::STANDARD_SOLUTION,0,otherTestIDs);
    
  //  std::cout << "residual_w1:\n" << residual_w1->displayString() << std::endl;
    auto ip_l2 = _vlasovSoln->bf()->l2Norm();
    auto rieszRep_residual = RieszRep::rieszRep(_vlasovSoln->mesh(), ip_l2, residual_w1);
    const double norm = rieszRep_residual->getNorm();
    
    if (int(_w1ResidualNorm[i].size()) > slabNumber)
    {
      _w1ResidualNorm[i][slabNumber] = norm;
    }
    else
    {
      _w1ResidualNorm[i].push_back(norm);
    }
  }
}

Teuchos::RCP<ConfigurationSpaceFormulation> VlasovSpaceTimeFormulation::configurationSpaceFormulation()
{
  return _configurationSpaceFormulation;
}

MeshPtr VlasovSpaceTimeFormulation::configurationSpaceMesh()
{
  return _configurationSpaceFormulation->configurationSpaceMesh();
}

SolutionPtr VlasovSpaceTimeFormulation::configurationSpaceSolution()
{
  return _configurationSpaceFormulation->solution();
}

void VlasovSpaceTimeFormulation::configurationSpaceSolve()
{
  for (int i=0; i<_numSpecies; i++)
  {
    _solutionMoments[i]->update();
  }
  {
    double n0;
    if (_enforceQuasiNeutrality)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_numSpecies > 1, std::invalid_argument, "quasi-neutrality enforcement assumes single species");
      FunctionPtr zeroMoment   = _solutionMoments[0]->solutionMomentFunction(0);
      auto configurationMesh   = _configurationSpaceFormulation->configurationSpaceMesh();
      const double meshMeasure = Function::constant(1.0)->integrate(configurationMesh);
      const double f_integral  = zeroMoment->integrate(configurationMesh);
      n0 = f_integral / meshMeasure;
    }
    else
    {
      n0 = 0;
    }
    _n0->setValue(n0);
  }
  _configurationSpaceFormulation->solve(false); // false: don't update solution moments (we just did so, above)
  _configurationSpaceFormulation->updateConfigurationSpaceFunctions();
}

const double &VlasovSpaceTimeFormulation::conservationLawResidualNorm(int slabOrdinal, int speciesOrdinal)
{
  return _w1ResidualNorm[speciesOrdinal][slabOrdinal];
}

Teuchos::RCP<GMGSolver> VlasovSpaceTimeFormulation::constructGMGSolver()
{
//    {
//      double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//      std::cout << "Before GMG construction, memory HWM: " << memoryUsageInGigabytes << " GB.\n";
//    }
    
    int kCoarse = 0;

    bool useCondensedSolve = _vlasovSoln->usesCondensedSolve();

    auto multigridStrategy = GMGOperator::V_CYCLE;
//    auto multigridStrategy = GMGOperator::SMOOTHER_ONLY;
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::IFPACK_ADDITIVE_SCHWARZ;
    GMGOperator::SmootherChoice smootherChoice = GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ; // CAMELLIA_ADDITIVE_SCHWARZ defines cells as blocks; Ifpack's defines all cells on the MPI rank as a block, making MPI scaling less predictable.  (Camellia's implementation is likely less well-optimized, though.)
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::BLOCK_SYMMETRIC_GAUSS_SEIDEL; // as configured, this is very slow; I think it's treating all data on an MPI rank as a block…
//    GMGOperator::SmootherChoice smootherChoice = GMGOperator::POINT_SYMMETRIC_GAUSS_SEIDEL;
    vector<MeshPtr> meshes;
    
    if (multigridStrategy != GMGOperator::SMOOTHER_ONLY)
    {
      meshes = GMGSolver::meshesForMultigrid(_vlasovSoln->mesh(), kCoarse, 1);
    }
    else
    {
      meshes = vector<MeshPtr> {_vlasovSoln->mesh()};
    }
    
    auto gmgSolver = Teuchos::rcp( new GMGSolver(_vlasovSoln, meshes, _maxIters, _cgTol, multigridStrategy,
                                                 Solver::getDirectSolver(true), useCondensedSolve) );
    
    gmgSolver->setSmootherType(smootherChoice);
    
    gmgSolver->setAztecOutput(_aztecOutputLevel);
    
//    {
//      double memoryUsageInGigabytes = memoryHighWaterMarkInGigabytes();
//      std::cout << "After GMG construction, memory HWM:  " << memoryUsageInGigabytes << " GB.\n";
//    }
  return gmgSolver;
}

IPPtr VlasovSpaceTimeFormulation::constructVlasovInnerProduct(const bool useSumFactorizedAssembly)
{
  if (useSumFactorizedAssembly)
  {
    auto ip = Teuchos::rcp( new VlasovSpaceTimeIP(*this) );
    return ip;
  }
  else
  {
    auto vlasovBF = this->vlasovSolution()->bf();
    return vlasovBF->graphNorm();
  }
}

VarFactoryPtr VlasovSpaceTimeFormulation::constructVlasovVarFactory()
{
  // Vlasov equation/mesh will not "know" about "space-time" as such -- no variables will be marked as space-time.
  // We can do this because the flux variable (t_n_hat) is defined on every interface, so we don't need the weird
  // only-supported-on-spatial-interfaces thing that we need for space-time Poisson.
  VarFactoryPtr vlasovVF = VarFactory::varFactory();
  
  BasisFamilyPtr basisFamily = BasisFamily::basisFamily(_basisFamily);
  
  for (int i=0; i<_numSpecies; i++)
  {
    VarPtr f_i = vlasovVF->fieldVar(_f_names[i]);            // distribution function (field variable)
    VarPtr w_i = vlasovVF->testVar(_w_names[i], HGRAD);      // test function w; lies in HGRAD of the whole space-time phase space
    VarPtr t_n_hat_i = vlasovVF->fluxVar(_tn_hat_names[i]); // t_n_hat: L^2 (non-conforming) trace
  }
  
  vlasovVF->setTrialBasisFamily(basisFamily);
  vlasovVF->setTestBasisFamily(basisFamily);
  
  return vlasovVF;
}

void VlasovSpaceTimeFormulation::configureNextSlab(Teuchos::RCP<VlasovSpaceTimeFormulation> nextSlabForm)
{
  auto nextSlabVlasovMesh = nextSlabForm->vlasovMesh(); // "from" in cellIDMap
  auto thisSlabVlasovMesh = this->vlasovMesh();
  
  std::map<IndexType,IndexType> cellIDMap;
  std::map<int,int> sideOrdinalMap;
  
  MeshTools::hRefineToMatchAtBoundary(cellIDMap, sideOrdinalMap, nextSlabVlasovMesh, VLASOV_TAG_TMIN, thisSlabVlasovMesh);
  
  for (auto sideSetTagFilter : _configurationSpaceSideSetFilters)
  {
    nextSlabForm->createConfigurationMeshSideSetTag(sideSetTagFilter.first,sideSetTagFilter.second);
  }
  
  for (auto entry : _configurationSpaceBCs)
  {
    int tag           = std::get<0>(entry);
    VarPtr var        = std::get<1>(entry);
    FunctionPtr value = std::get<2>(entry);
    
    nextSlabForm->addConfigurationSpaceBC(tag, var, value);
  }
  
  // copy my settings:
  nextSlabForm->setIterativeSolveOutputLevel(_aztecOutputLevel);
  nextSlabForm->setIterativeSolveTolerance(_cgTol);
  nextSlabForm->setIterativeSolveTolerances(_cgTols);
  nextSlabForm->setUseGeometricMultigrid(_useGeometricMultigrid);
  nextSlabForm->setEnforceQuasiNeutrality(_enforceQuasiNeutrality);
  nextSlabForm->setIterativeSolveMaxIterations(_maxIters);
  nextSlabForm->setFixedPointMaxSubsteps(_maxSubsteps);
  nextSlabForm->setFixedPointMinSubsteps(_minSubsteps);
  nextSlabForm->setFixedPointTolerance(_fixedPointTolerance);
  nextSlabForm->setExportViz(_exportViz,_vizOutputDir);
  nextSlabForm->setMaxMomentForConservationDiagnostics(_maxMomentForConservationDiagnostics);
  
  // copy slab/diagnostic history:
  nextSlabForm->setSlabHistory(_slabTemporalCoordinates, _massIntegrals, _momentumIntegrals, _energyIntegrals,
                               _elementwiseMassIntegrals, _elementwiseMomentumIntegrals, _elementwiseEnergyIntegrals,
                               _w1ResidualNorm);
  
  // import solution data so that it's available for BC setting
  std::set<GlobalIndexType> currentCellIDsForRank;
  const auto &myNextCellIDs = nextSlabVlasovMesh->cellIDsInPartition();
  for (const auto &myNextCellID : myNextCellIDs)
  {
    auto mapEntry = cellIDMap.find(myNextCellID); // all rank-local cells which lie on the final-time boundary should have an entry
    if (mapEntry != cellIDMap.end())
    {
      currentCellIDsForRank.insert(mapEntry->second);
    }
  }
  _vlasovSoln->importSolutionForOffRankCells(currentCellIDsForRank);
  
  // set up Functions that can be evaluated on the next slab with reference to the current (will be projected onto next below)
  std::vector<FunctionPtr>  f0_next(_numSpecies);
  for (int i=0; i<_numSpecies; i++)
  {
    VarPtr tn = this->t_n_hat(i);
    bool weightFluxesBySideParity = false; // should not matter: boundary parity is 1
    SolutionPtr thisSlabVlasovSoln = this->vlasovSolution();
    
    // use a weak pointer because the solution will have this reference as part of its own BCs, which would result in a ref-counted pointer cycle.
    SolutionPtr nextSlabVlasovSolnWeakPtr = Teuchos::rcp(nextSlabForm->vlasovSolution().get(), false);
    FunctionPtr tnFxnNextSlab = Teuchos::rcp( new SimpleSolutionFunction<double>(tn, nextSlabVlasovSolnWeakPtr, weightFluxesBySideParity) );
    f0_next[i] = -tnFxnNextSlab; // at temporal inflow, the normal in "tn" is negative, so f = - tn.
  }
  
  // find cells on next slab's initial-time boundary
  auto initialTimeTag = nextSlabForm->tagVlasovInitialTime();
  auto meshTopo = Teuchos::rcp(nextSlabForm->vlasovMesh()->getTopology()->baseMeshTopology(), false);
  auto entitySets = meshTopo->getEntitySetsForTagID(DIRICHLET_SET_TAG_NAME, initialTimeTag);
  
  const auto & myCells = nextSlabForm->vlasovMesh()->cellIDsInPartition();
  std::set<IndexType> myInitialTimeCells; // those owned by this MPI rank
  for (auto entitySet : entitySets)
  {
    std::set<IndexType> matchingCells = entitySet->cellIDsThatMatch(meshTopo, myCells);
    myInitialTimeCells.insert(matchingCells.begin(),matchingCells.end());
  }
  
  auto nextVlasovSoln = nextSlabForm->vlasovSolution();
  auto nextVlasovMesh = nextSlabForm->vlasovMesh();
  
  nextVlasovSoln->initializeLHSVector(); // ensures that setSolnCoeffsForCellID() sets not only the local solnCoefficients but also the corresponding entries in the global solution vector.
  
  for (const auto & nextCellID : myInitialTimeCells)
  {
    const auto elemType = nextVlasovMesh->getElementType(nextCellID);
    
    IndexType currentCellID = cellIDMap[nextCellID];
    
    int currentSideIndex;
    int nextSideIndex;
    if (sideOrdinalMap.size() == 1)
    {
      // if all the cells agree on their side ordinals (which will be true if the spatial cells all have the same topology), then it's easy to figure out which sides we're talking about here.  The maps are built to be able to take a cell ID from the "next" slab and find the corresponding cell ID in the previous one.
      nextSideIndex    = sideOrdinalMap.begin()->first;  // values are the sides in the next slab
      currentSideIndex = sideOrdinalMap.begin()->second;
    }
    else
    {
      // TODO: write some logic to determine which is the initial-time side for the cell; might look at MeshTools::hRefineToMatchAtBoundary() for inspiration.  (Then again, that method might also assume that the cell topologies are the same, and might need to be revised to support this use case.)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Support for multiple spatial cell topologies is not yet implemented in configureNextSlab()");
    }
    
    for (int i=0; i<_numSpecies; i++)
    {
      VarPtr tn = this->t_n_hat(i);
      
      int basisCardinality = nextVlasovMesh->getElementType(nextCellID)->trialOrderPtr->getBasisCardinality(tn->ID(), nextSideIndex);
      FieldContainer<double> coefficients(basisCardinality);
      _vlasovSoln->solnCoeffsForCellID(coefficients, currentCellID, tn->ID(), currentSideIndex);
      // because tn has an outward-facing normal included, we need to reverse the sign of the coefficients for the cell in the next slab
      for (int i=0; i<basisCardinality; i++)
      {
        coefficients[i] *= -1;
      }
      nextVlasovSoln->setSolnCoeffsForCellID(coefficients, nextCellID, tn->ID(), nextSideIndex);
    }
  }
  
  // NOTE: for the conforming case (not yet supported here: this will be the case when we use "f_hat" instead of "tn_hat"), in which some initial-condition data will be shared between neighboring cells, we will need to do an import at this point.
  
  if (_f_inflow.size() != 0)
  {
    nextSlabForm->setInitialAndInflowConditions(f0_next, _f_inflow);
  }
  else
  {
    nextSlabForm->setInitialConditions(f0_next);
  }
  _vlasovSoln->setBC(Teuchos::null);
}

int VlasovSpaceTimeFormulation::createConfigurationMeshSideSetTag(SpatialFilterPtr regionToMatch, const bool restrictToBoundary)
{
  const int thisTag = _nextConfigurationSpaceTag++;
  
  const int sideDim = _spaceDim; // space-time configuration-space mesh has dimension _spaceDim + 1; sides have dimension _spaceDim
  auto matchingSideSet = _configurationSpaceMeshTopology->createEntitySet();
  std::vector<IndexType> matchingSideIndices = restrictToBoundary ? _configurationSpaceMeshTopology->getBoundarySidesThatMatch(regionToMatch)
                                                                  : _configurationSpaceMeshTopology->getSidesThatMatch        (regionToMatch);
  for (const auto & matchingSideIndex : matchingSideIndices)
  {
    matchingSideSet->addEntity(sideDim, matchingSideIndex);
  }
  _configurationSpaceMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, thisTag, matchingSideSet);
  
  _configurationSpaceSideSetFilters.push_back({regionToMatch,restrictToBoundary});
  
  return thisTag;
}

int VlasovSpaceTimeFormulation::delta_k() const
{
  return _delta_k;
}

VarPtr VlasovSpaceTimeFormulation::E_configurationSpaceTrace(const int zeroBasedComp)
{
  return _configurationSpaceFormulation->E_hat(zeroBasedComp);
}

bool VlasovSpaceTimeFormulation::enforceQuasiNeutrality() const
{
  return _enforceQuasiNeutrality;
}

bool VlasovSpaceTimeFormulation::exportViz() const
{
  return _exportViz;
}

VarPtr VlasovSpaceTimeFormulation::f(int i)
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->trialVar(_f_names[i]);
}

VarPtr VlasovSpaceTimeFormulation::f_hat(int i)
{
  VarFactoryPtr vf = _vlasovSoln->bf()->varFactory();
  return vf->trialVar(_f_hat_names[i]);
}

int VlasovSpaceTimeFormulation::fixedPointMaxSubsteps() const
{
  return _maxSubsteps;
}

int VlasovSpaceTimeFormulation::fixedPointMinSubsteps() const
{
  return _minSubsteps;
}

double VlasovSpaceTimeFormulation::fixedPointTolerance() const
{
  return _fixedPointTolerance;
}

bool VlasovSpaceTimeFormulation::hasL2Trace() const {
  return true;
}

int VlasovSpaceTimeFormulation::H1Order() const
{
  return _H1Order;
}

void VlasovSpaceTimeFormulation::initialize()
{
  TEUCHOS_TEST_FOR_EXCEPTION(_initialized, std::invalid_argument, "initialize() should only be called once, during construction");
  
  _t0 = _temporalVertices[0];
  _t1 = _temporalVertices[_temporalVertices.size()-1];
  
  _slabTemporalCoordinates = {_t0, _t1}; // will be overwritten by setSlabHistory(), if that's called (i.e., if this is not the initial slab).
  
  _configurationSpaceMeshTopology = MeshFactory::extrudedMeshTopology(_spatialMeshTopology, {_temporalVertices});
  _configurationSpaceFormulation->initializeMesh(_configurationSpaceMeshTopology, _H1Order, _delta_k);
  
  std::vector<std::vector<double> > velocitySpaceVertices;
  
  TEUCHOS_TEST_FOR_EXCEPTION(_velocitySpaceVertices_x.size() == 0, std::invalid_argument, "velocitySpaceVertices_x may not be empty");
  velocitySpaceVertices.push_back(_velocitySpaceVertices_x);
  if (_velocitySpaceVertices_y.size() > 0)
  {
    velocitySpaceVertices.push_back(_velocitySpaceVertices_y);
    if (_velocitySpaceVertices_z.size() > 0)
    {
      velocitySpaceVertices.push_back(_velocitySpaceVertices_z);
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_velocitySpaceVertices_y.size() > 0, std::invalid_argument, "if velocitySpaceVertices_y is empty, then velocitySpaceVertices_z must also be empty");
  }
  
  MeshTopologyPtr extrudedMeshTopology = MeshFactory::extrudedMeshTopology(_configurationSpaceMeshTopology, velocitySpaceVertices);
  
  _spaceDim = _spatialMeshTopology->getDimension();
  _velocityDim = velocitySpaceVertices.size();
  
  _configurationSpaceFormulation->setDimensions(_spaceDim, _velocityDim);

  TFunctionPtr<double> parity = TFunction<double>::sideParity();
  TFunctionPtr<double> n = TFunction<double>::normal();
  
  // Vlasov equation/mesh will not "know" about "space-time" as such -- no variables will be marked as space-time.
  // We can do this because the flux variable (t_n_hat) is defined on every interface, so we don't need the weird
  // only-supported-on-spatial-interfaces thing that we need for space-time Poisson.
  VarFactoryPtr vlasovVF = constructVlasovVarFactory();
  _vlasovBF = BF::bf(vlasovVF);
  
  // fields
  std::vector<VarPtr> f(_numSpecies); // Vlasov distributions
  
  // traces
  std::vector<VarPtr> t_n_hat(_numSpecies); // t_n_hat: L^2 (non-conforming) trace
  std::vector<VarPtr> f_hat(_numSpecies);   // f_hat:   conforming trace (unused in present implementation -- see l2Trace flag below)

  // tests
  std::vector<VarPtr> w(_numSpecies);   // Vlasov test functions
  
  for (int i=0; i<_numSpecies; i++)
  {
    std::string suffix = (_speciesLabels[i].length() > 0) ? "_" + _speciesLabels[i] : "";
    VarPtr f_i = vlasovVF->fieldVar(S_F + suffix);
    VarPtr w_i = vlasovVF->testVar(S_W + suffix, HGRAD); // w lies in HGRAD of phase space
    
    VarPtr t_n_hat_i = vlasovVF->fluxVar(S_T_N_HAT + suffix);
    
    f[i]       = f_i;
    w[i]       = w_i;
    t_n_hat[i] = t_n_hat_i;
  }
  
  _vlasovBF = BF::bf(vlasovVF);
  
  const bool allowPruning = false;
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::null; // use default partition policy
  Epetra_CommPtr comm = MPIWrapper::CommWorld();
  MeshPtr vlasovMesh = Teuchos::rcp( new Mesh(extrudedMeshTopology, _vlasovBF, _H1Order, _delta_k, trialOrderEnhancements(), testOrderEnhancements(), partitionPolicy, comm, allowPruning) );
  
  auto vlasovMeshWeakPtr = Teuchos::rcp(vlasovMesh.get(), false);
  
  _vlasovSoln     = Solution::solution(_vlasovBF,vlasovMesh);
  _vlasovSolnPrev = Solution::solution(_vlasovBF,vlasovMesh); // will copy out after fixed-point solve
  
  const int maxMomentNumber = std::max(_configurationSpaceFormulation->maxMomentNumber(), _maxMomentNumberForDriver);
  _solutionMoments.resize(_numSpecies);
  for (int i=0; i<_numSpecies; i++)
  {
    _solutionMoments[i] = Teuchos::rcp( new SolutionMoments(maxMomentNumber, _vlasovSoln, f[i], _configurationSpaceFormulation->configurationSpaceMesh()) );
  }
  
  _n0 = ParameterFunction::parameterFunction(0); // we'll update this when initial conditions are set, if _enforceQuasiNeutrality is true
  _configurationSpaceFormulation->setMoments(_solutionMoments,_n0);
  
  // the di() methods are 1-based:
  const int indexOffsetTimeOneBased     = 1 + _spaceDim;
  const int indexOffsetVelocityOneBased = indexOffsetTimeOneBased + 1;
  
  // SpatialFilter's components are zero-based
  const int indexOffsetTimeZeroBased     = indexOffsetTimeOneBased - 1;
  const int indexOffsetVelocityZeroBased = indexOffsetVelocityOneBased - 1;
  
  // initialize acceleration term, _a
  {
    int velocityDim = this->velocityDim();
    std::vector<FunctionPtr> v(velocityDim);
    for (int i=0; i<velocityDim; i++)
    {
      v.push_back(this->v_i(i));
    }
    _a = std::vector<FunctionPtr>(velocityDim); // this gets dotted with grad_v of f (in the strong form)
    
    std::vector<FunctionPtr> E(3,Function::zero());
    std::vector<FunctionPtr> B(3,Function::zero());
    
    for (int i=0; i<velocityDim; i++)
    {
      E[i] = _configurationSpaceFormulation->E_i(i);
      B[i] = _configurationSpaceFormulation->B_i(i);
    }
    
    // we only include the v_cross_B term for 3V case
    // TODO: determine whether v_cross_B ever makes sense for 1V or 2V cases; if so, add something here.
    std::vector<FunctionPtr> v_cross_B(3,Function::zero());
    if (velocityDim == 3)
    {
      v_cross_B[0] = v[1] * B[2] - v[2] * B[1];
      v_cross_B[1] = v[2] * B[0] - v[0] * B[2];
      v_cross_B[2] = v[0] * B[1] - v[1] * B[0];
    }
    
    auto extraAcceleration = _configurationSpaceFormulation->extraAcceleration();
    for (int i=0; i<velocityDim; i++)
    {
      _a[i] = E[i] + v_cross_B[i] + extraAcceleration[i]; // q/m weight is applied based on the species within the a() implementation.
    }
  }
  
  initializeVlasovBilinearForm(_vlasovBF);

  auto vlasovSolnWeakPtr     = Teuchos::rcp(_vlasovSoln.get(),     false);
  auto vlasovSolnPrevWeakPtr = Teuchos::rcp(_vlasovSolnPrev.get(), false);
  
  _fSoln.resize(_numSpecies);
  _fDiff.resize(_numSpecies);
  for (int i=0; i<_numSpecies; i++)
  {
    _fSoln[i]     = Function::solution(f[i], vlasovSolnWeakPtr);
    FunctionPtr f_solnPrev = Function::solution(f[i], vlasovSolnPrevWeakPtr);
    
    _fDiff[i] = _fSoln[i] - f_solnPrev;
  }
  
  
    
  auto vlasovRHS = RHS::rhs();
  _vlasovSoln->setRHS(vlasovRHS); // zero RHS for vlasov, until we add collisions
    
  IPPtr vlasovTestNorm = this->constructVlasovInnerProduct(_useSumFactorizedAssembly);
  _vlasovSoln->setIP(vlasovTestNorm);
  
  SpatialFilterPtr inflowFilter = Teuchos::rcp(new SpatialFilterSpatialInflow(_spaceDim,indexOffsetVelocityZeroBased) );
  
  SpatialFilterPtr initialTimeFilter = SpatialFilter::matchingComponent(indexOffsetTimeZeroBased, _t0);
  
  std::vector<double> v0,vMax;
  
  std::vector<SpatialFilterPtr> minVelocityFilters, maxVelocityFilters;
  for (int i=0; i<_velocityDim; i++)
  {
    auto & dimVertices = velocitySpaceVertices[i];
    const int numVertices = dimVertices.size();
    v0.push_back(dimVertices[0]);
    vMax.push_back(dimVertices[numVertices-1]);
    
    SpatialFilterPtr minVelocityFilter = SpatialFilter::matchingComponent(indexOffsetVelocityZeroBased + i, v0[i]  );
    SpatialFilterPtr maxVelocityFilter = SpatialFilter::matchingComponent(indexOffsetVelocityZeroBased + i, vMax[i]);
    
    minVelocityFilters.push_back(minVelocityFilter);
    maxVelocityFilters.push_back(maxVelocityFilter);
  }
  
  std::vector<int> minVelocityTags {VLASOV_TAG_VXMIN, VLASOV_TAG_VYMIN, VLASOV_TAG_VZMIN};
  std::vector<int> maxVelocityTags {VLASOV_TAG_VXMAX, VLASOV_TAG_VYMAX, VLASOV_TAG_VZMAX};
  
  // create entity sets for BCs:
  {
    // Vlasov side sets
    
    // initial conditions
    const int sideDim = _spaceDim + _velocityDim; // phase space, space-time: sides have dimension of (fixed-time) phase space
    auto initialTimeSideSet = extrudedMeshTopology->createEntitySet();
    std::vector<IndexType> initialTimeSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(initialTimeFilter);
    for (const auto & initialTimeSideIndex : initialTimeSideIndices)
    {
      initialTimeSideSet->addEntity(sideDim, initialTimeSideIndex);
    }
    extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_TMIN, initialTimeSideSet);
    
    // inflow conditions
    auto inflowSideSet = extrudedMeshTopology->createEntitySet();
    std::vector<IndexType> inflowSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(inflowFilter, vlasovMesh);
    for (const auto & inflowSideIndex : inflowSideIndices)
    {
      inflowSideSet->addEntity(sideDim, inflowSideIndex);
    }
    extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, VLASOV_TAG_SPATIAL_INFLOW, inflowSideSet);

    for (int i=0; i<_velocityDim; i++)
    {
      auto minSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> minSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(minVelocityFilters[i]);
      for (const auto & minSideIndex : minSideIndices)
      {
        minSideSet->addEntity(sideDim, minSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, minVelocityTags[i], minSideSet);
      auto maxSideSet = extrudedMeshTopology->createEntitySet();
      std::vector<IndexType> maxSideIndices = extrudedMeshTopology->getBoundarySidesThatMatch(maxVelocityFilters[i]);
      for (const auto & maxSideIndex : maxSideIndices)
      {
        maxSideSet->addEntity(sideDim, maxSideIndex);
      }
      extrudedMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, maxVelocityTags[i], maxSideSet);
    }
  }
  // create initial-time tag for configuration-space mesh
  {
    const int sideDim = _spaceDim; // space-time configuration-space mesh has dimension _spaceDim + 1; sides have dimension _spaceDim
    auto initialTimeSideSet = _configurationSpaceMeshTopology->createEntitySet();
    std::vector<IndexType> initialTimeSideIndices = _configurationSpaceMeshTopology->getBoundarySidesThatMatch(initialTimeFilter);
    for (const auto & initialTimeSideIndex : initialTimeSideIndices)
    {
      initialTimeSideSet->addEntity(sideDim, initialTimeSideIndex);
    }
    _configurationSpaceMeshTopology->applyTag(DIRICHLET_SET_TAG_NAME, CONFIGURATION_TAG_TMIN, initialTimeSideSet);
  }
  
  PhysicalMeshUpdater::registerForUpdates(_vlasovSoln->mesh(), _configurationSpaceFormulation->configurationSpaceMesh());
  
  for (int uniformRefinementNumber=0; uniformRefinementNumber < _numUniformRefinements; uniformRefinementNumber++)
  {
    const auto & activeCells  = _vlasovSoln->mesh()->getActiveCellIDsGlobal();
    _vlasovSoln->mesh()->hRefine(activeCells);
    // (the configuration-space mesh will automatically be refined when the Vlasov mesh is, by virtue of the registerForUpdates() call above)
  }
  
  vlasovMesh->registerSolution(_vlasovSoln);
  
  _initialized = true;
}

void VlasovSpaceTimeFormulation::initializeVlasovBilinearForm(BFPtr vlasovBF)
{
  Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation = this->configurationSpaceFormulation();
  
  const int spaceDim = this->spaceDim();
  const int velocityDim = this->velocityDim();
  
  // the di() methods are 1-based:
  const int indexOffsetSpaceOneBased    = 1;
  const int indexOffsetTimeOneBased     = 1 + spaceDim;
  const int indexOffsetVelocityOneBased = indexOffsetTimeOneBased + 1;
    
  // The below assumes that spaceDim <= velocityDim; enforce that here:
  TEUCHOS_TEST_FOR_EXCEPTION(spaceDim > velocityDim, std::invalid_argument, "velocityDim must be at least as large as spaceDim");
  std::vector<FunctionPtr> v;
  for (int vd=0; vd<velocityDim; vd++)
  {
    v.push_back(this->v_i(vd));
  }
  
  for (int i=0; i<_numSpecies; i++)
  {
    VarPtr f = this->f(i);
    VarPtr w = this->w(i);
    VarPtr t_n_hat = this->t_n_hat(i);
    
    for (int d=0; d<spaceDim; d++)
    {
      vlasovBF->addTerm( -v[d] * f, w->di(indexOffsetSpaceOneBased + d));  // dx_d
    }
    vlasovBF->addTerm( -f, w->di(indexOffsetTimeOneBased)); // dt
    
    auto a = this->a(); // acceleration term
    for (int vd=0; vd<velocityDim; vd++)
    {
      // grad v of f
      vlasovBF->addTerm(-a[vd] * f, w->di(indexOffsetVelocityOneBased + vd));
    }
    
    vlasovBF->addTerm(t_n_hat, w);
  }
}

int VlasovSpaceTimeFormulation::iterativeSolveMaxIterations() const
{
  return _maxIters;
}

int VlasovSpaceTimeFormulation::iterativeSolveOutputLevel() const
{
  return _aztecOutputLevel;
}

double VlasovSpaceTimeFormulation::iterativeSolveTolerance() const
{
  return _cgTol;
}

std::vector<double> VlasovSpaceTimeFormulation::iterativeSolveTolerances() const
{
  return _cgTols;
}

const VlasovSpaceTimeFormulation::ElementwiseConservationDiagnostics & VlasovSpaceTimeFormulation::localMassConservationDiagnostics(int slabOrdinal, int speciesOrdinal)
{
  return _elementwiseMassIntegrals[speciesOrdinal][slabOrdinal];
}

const VlasovSpaceTimeFormulation::ElementwiseConservationDiagnostics & VlasovSpaceTimeFormulation::localXMomentumConservationDiagnostics(int slabOrdinal, int speciesOrdinal)
{
  return _elementwiseMomentumIntegrals[speciesOrdinal][0][slabOrdinal];
}

const VlasovSpaceTimeFormulation::ElementwiseConservationDiagnostics & VlasovSpaceTimeFormulation::localYMomentumConservationDiagnostics(int slabOrdinal, int speciesOrdinal)
{
  return _elementwiseMomentumIntegrals[speciesOrdinal][1][slabOrdinal];
}

const VlasovSpaceTimeFormulation::ElementwiseConservationDiagnostics & VlasovSpaceTimeFormulation::localZMomentumConservationDiagnostics(int slabOrdinal, int speciesOrdinal)
{
  return _elementwiseMomentumIntegrals[speciesOrdinal][2][slabOrdinal];
}

const VlasovSpaceTimeFormulation::ElementwiseConservationDiagnostics & VlasovSpaceTimeFormulation::localEnergyConservationDiagnostics(int slabOrdinal, int speciesOrdinal)
{
  return _elementwiseEnergyIntegrals[speciesOrdinal][slabOrdinal];
}

double VlasovSpaceTimeFormulation::m(int i) const
{
  return _m[i];
}

int VlasovSpaceTimeFormulation::maxMomentForConservationDiagnostics() const
{
  return _maxMomentForConservationDiagnostics;
}

Teuchos::RCP<VlasovSpaceTimeFormulation> VlasovSpaceTimeFormulation::nextTimeSlab(double tWidth, int numTemporalElements)
{
  auto temporalVertices = MeshTools::verticesFor1DIntervalMesh(_t1, tWidth, numTemporalElements);
  
  // create a new spatial MeshTopology based on the root of the current _spatialMeshTopology
  auto rootSpatialMeshTopology = MeshFactory::rootMeshTopologyCopy(_spatialMeshTopology);
  
  auto nextSlabForm = Teuchos::rcp(new VlasovSpaceTimeFormulation(_q, _m, _speciesLabels, _configurationSpaceFormulation, rootSpatialMeshTopology,
                                                                  temporalVertices, _velocitySpaceVertices_x, _velocitySpaceVertices_y, _velocitySpaceVertices_z,
                                                                  _H1Order, _delta_k, _numUniformRefinements, _basisFamily));
  
  configureNextSlab(nextSlabForm);
  
  return nextSlabForm;
}

//! Returns the number of velocity dimensions in the mesh.
int VlasovSpaceTimeFormulation::numSpecies() const
{
  return _numSpecies;
}

int VlasovSpaceTimeFormulation::numUniformRefinements() const
{
  return _numUniformRefinements;
}

void VlasovSpaceTimeFormulation::printLocalConservationReportForSlab(int slabOrdinal)
{
  for (int i=0; i<_numSpecies; i++)
  {
    std::string prefix = (_speciesLabels[i].size() == 0) ? "" : "species " + _speciesLabels[i] + ": ";
    auto & massIntegral = _elementwiseMassIntegrals[i][slabOrdinal];
    std::cout << prefix << "max mass conservation failure occured in element " << massIntegral.elementWithMaxConservationFailure << ": " << massIntegral.maxElementConservationFailure << std::endl;
    
    auto & residualNorm = conservationLawResidualNorm(slabOrdinal);
    std::cout << prefix << "enriched L^2-dual norm of the conservation law residual: " << residualNorm << std::endl;
  }
}

double VlasovSpaceTimeFormulation::q(int i) const
{
  return _q[i];
}

void VlasovSpaceTimeFormulation::recordElementwiseConservationDiagnostics(int speciesOrdinal, std::vector< ElementwiseConservationDiagnostics > &diagnosticHistory, int slabOrdinal, const std::map<int,FunctionPtr> &testFunction)
{
  auto trialFunctional = _vlasovBF->trialFunctional(testFunction); // the BF evaluated at the test function
  
//  { // DEBUGGING
//    if (MPIWrapper::CommWorld()->MyPID() == 0)
//    {
//      std::cout << "Conservation diagnostic: " << trialFunctional->displayString() << std::endl;
//    }
//  }
  
  map<int,VarPtr> trialVarMap = _vlasovBF->varFactory()->trialVars();
  
  vector<VarPtr> trialVars;
  for (auto entry : trialVarMap)
  {
    trialVars.push_back(entry.second);
  }
  
  // because BF::trialFunctional already weights by parity, we should not weight by it again here:
  bool weightFluxesByParity = false;
  auto solnMap = PreviousSolutionFunction<double>::functionMap(trialVars, _vlasovSoln, STANDARD_SOLUTION, 0, weightFluxesByParity);
  
  FunctionPtr conservationIntegrandVolume   = trialFunctional->evaluate(solnMap, false);
  FunctionPtr conservationIntegrandSkeleton = trialFunctional->evaluate(solnMap, true);
  
  const auto &myCellIDs = _vlasovSoln->mesh()->cellIDsInPartition();
  
  ElementwiseConservationDiagnostics diagnostics;
  
  double myMaxFailure = 0.0;
  IndexType myCellIDWithMax = -1;
  double localSum = 0.0;
  double localAbsSum = 0.0;
  
  for (const auto &cellID : myCellIDs)
  {
    const double valueVolume   = conservationIntegrandVolume->integrate(cellID, _vlasovSoln->mesh());
    const double valueSkeleton = conservationIntegrandSkeleton->integrate(cellID, _vlasovSoln->mesh());
    const double value         = valueVolume + valueSkeleton;
    const double absValue = std::abs(value);
    
    if (absValue > myMaxFailure)
    {
      myMaxFailure = absValue;
      myCellIDWithMax = cellID;
    }
    
    localSum    += value;
    localAbsSum += absValue;
  }
  
  double globalSum    = MPIWrapper::sum(localSum);
  double globalAbsSum = MPIWrapper::sum(localAbsSum);
  
  std::map<GlobalIndexType, double> localMaximum;
  localMaximum[myCellIDWithMax] = myMaxFailure;
  
  const int numEntries = 1;
  const int maxGatherCount = MPIWrapper::CommWorld()->NumProc();
  
  auto actualGreatestEntries = MPIGlobalMinMax::globalMaximums(MPIWrapper::CommWorld(), localMaximum, numEntries, maxGatherCount);
  
  IndexType globalMaxFailureCellID = actualGreatestEntries.begin()->first;
  double globalMaxFailure = actualGreatestEntries.begin()->second;
  
  diagnostics.elementWithMaxConservationFailure = globalMaxFailureCellID;
  diagnostics.maxElementConservationFailure = globalMaxFailure;
  
  diagnostics.netGlobalConservationFailure = globalSum;
  diagnostics.sumOfAbsoluteConservationFailures = globalAbsSum;
  
  if (int(diagnosticHistory.size()) <= slabOrdinal)
  {
    diagnosticHistory.push_back(diagnostics);
  }
  else
  {
    diagnosticHistory[slabOrdinal] = diagnostics;
  }
}

void VlasovSpaceTimeFormulation::setEnforceQuasiNeutrality(bool enforceQuasiNeutrality)
{
  _enforceQuasiNeutrality = enforceQuasiNeutrality;
}

void VlasovSpaceTimeFormulation::setExportViz(bool exportViz, const std::string outputDir)
{
  _exportViz = exportViz;
  _vizOutputDir = outputDir;
}

void VlasovSpaceTimeFormulation::setFixedPointMaxSubsteps(const int maxSubsteps)
{
  _maxSubsteps = maxSubsteps;
}

void VlasovSpaceTimeFormulation::setFixedPointMinSubsteps(const int minSubsteps)
{
  _minSubsteps = minSubsteps;
}

void VlasovSpaceTimeFormulation::setFixedPointTolerance(const double fTol)
{
  _fixedPointTolerance = fTol;
}

void VlasovSpaceTimeFormulation::setIterativeSolveOutputLevel(int azOutputLevel)
{
  _aztecOutputLevel = azOutputLevel;
}

void VlasovSpaceTimeFormulation::setIterativeSolveMaxIterations(const int maxIters)
{
  _maxIters = maxIters;
}

void VlasovSpaceTimeFormulation::setIterativeSolveTolerance(const double cgTol)
{
  _cgTol = cgTol;
}

void VlasovSpaceTimeFormulation::setIterativeSolveTolerances(const std::vector<double> cgTols)
{
  _cgTols = cgTols;
}

void VlasovSpaceTimeFormulation::solve()
{
  const int rank = MPIWrapper::CommWorld()->MyPID();
  
  Teuchos::RCP<GMGSolver> gmgSolver;
  
  std::vector<int>    substepIterationCounts(_maxSubsteps);
  std::vector<double> substepStepSizesVlasov;
  
  Teuchos::RCP<HDF5Exporter> vlasovViz, configurationViz;
  
  if (_exportViz)
  {
    static int solveNumber = 0;
    std::string prefix = "VlasovSpaceTimeFormulation_";
    std::ostringstream vlasovSS;
    vlasovSS << prefix << "Vlasov" << solveNumber;
    
    std::ostringstream configurationSS;
    configurationSS << prefix << _configurationSpaceFormulation->name() << solveNumber;
      
    const std::string directoryPath = "/tmp";
    
    vlasovViz = Teuchos::rcp(new HDF5Exporter(this->vlasovMesh(), vlasovSS.str(), directoryPath));
    configurationViz = Teuchos::rcp(new HDF5Exporter (_configurationSpaceFormulation->configurationSpaceMesh(), configurationSS.str(), directoryPath) );
    solveNumber++;
  }
  
  // get the PhysicalSolutionExtrusions ready to evaluate (in context of Vlasov BF)
  // updateConfigurationSpaceFunctions() will do MPI communication of configuration-space solution values so they're where they need to be for phase space mesh:
  _configurationSpaceFormulation->updateConfigurationSpaceFunctions();
  for (int substepOrdinal=0; substepOrdinal < _maxSubsteps; substepOrdinal++)
  {
    if (rank == 0)
    {
      std::cout << "\n\nSolving substep " << substepOrdinal+1 << " of up to " << _maxSubsteps << " (with early stop when f update is below " << _fixedPointTolerance << ")" << std::endl;
    }
    // solve for f:
    if (!_useGeometricMultigrid)
    {
      _vlasovSoln->solve();
      substepIterationCounts[substepOrdinal] = -1; // -1 to indicate not an iterative solver
    }
    else
    {
      Teuchos::RCP<GMGSolver> gmgSolver = constructGMGSolver();
      if (_cgTols.size() > 0) // then we should have one per substep
      {
        gmgSolver->setTolerance(_cgTols[substepOrdinal]);
      }
      _vlasovSoln->solve(gmgSolver);
      substepIterationCounts[substepOrdinal] = gmgSolver->iterationCount();
    }
    for (auto & solnMoments : _solutionMoments)
    {
      solnMoments->update();
    }
    {
      double n0;
      if (_enforceQuasiNeutrality)
      {
        // _enforceQuasiNeutrality ==  true only supported for single-species, so we use _solutionMoments[0]
        FunctionPtr zeroMoment   = _solutionMoments[0]->solutionMomentFunction(0);
        auto configurationMesh   = _configurationSpaceFormulation->configurationSpaceMesh();
        const double meshMeasure = Function::constant(1.0)->integrate(configurationMesh);
        const double f_integral  = zeroMoment->integrate(configurationMesh);
        n0 = f_integral / meshMeasure;
      }
      else
      {
        n0 = 0;
      }
      _n0->setValue(n0);
    }
    _configurationSpaceFormulation->solve(false); // false: don't update solution moments (we just did so, above)
    _configurationSpaceFormulation->updateConfigurationSpaceFunctions(); // update/MPI-communicate (Poisson/Maxwell) values to make available for phase-space solution
    
    // Copy vlasov, Poisson solution to Prev solutions.  For substepOrdinal > 0, measure difference between the two solutions first.
    if (substepOrdinal > 0)
    {
      auto vlasovMesh  = _vlasovSoln->mesh();
      double fDiff_L2_squared = 0, f_L2_squared = 0;
      for (int i=0; i<_numSpecies; i++)
      {
        double fDiff_L2_i = _fDiff[i]->l2norm(vlasovMesh);
        double f_L2_i     = _fSoln[i]->l2norm(vlasovMesh);
        fDiff_L2_squared += fDiff_L2_i * fDiff_L2_i;
        f_L2_squared     += f_L2_i * f_L2_i;
      }
      const double fDiff_L2 = sqrt(fDiff_L2_squared);
      const double f_L2 = sqrt(f_L2_squared);
      const double fRelDiff = fDiff_L2 / f_L2;
      
      substepStepSizesVlasov.push_back(fRelDiff);
      
      if (rank == 0)
      {
        std::cout << "Substep " << substepOrdinal+1 << ", relative L^2 norm of difference in f: " << fRelDiff << std::endl;
      }
      
      if ((fRelDiff < _fixedPointTolerance) && (substepOrdinal >= _minSubsteps-1))
      {
        // tolerance reached, min substeps taken
        break;
      }
    }
    
    _vlasovSolnPrev->setSolution(_vlasovSoln);
    
    if (_exportViz)
    {
      configurationViz->exportSolution(_configurationSpaceFormulation->solution(),substepOrdinal);
      vlasovViz->exportSolution(vlasovSolution(),substepOrdinal);
    }
  }
  
  _iterationCounts.push_back(substepIterationCounts);
  _fixedPointL2StepSizesVlasov.push_back(substepStepSizesVlasov);
  
  computeConservationDiagnostics(false); // false: don't update solutionMoments; it was just updated above
}

void VlasovSpaceTimeFormulation::setInitialConditions(FunctionPtr f0)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_numSpecies > 1, std::invalid_argument, "For multi-species formulations, must call the multi-species setInitialConditions");
  this->setInitialConditions(vector<FunctionPtr>{f0});
}

void VlasovSpaceTimeFormulation::setInitialConditions(vector<FunctionPtr> f0)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_numSpecies != int(f0.size()), std::invalid_argument, "number of f0 components must match the number of species in the formulation");
  
  _f0 = f0;
  
  auto vlasovBC = BC::bc();
  
  for (int i=0; i<_numSpecies; i++)
  {
    VarPtr t_n = this->t_n_hat(i);
    for (int vd=0; vd<_velocityDim; vd++)
    {
      const int tagMin = this->tagVlasovMinVelocity(vd);
      const int tagMax = this->tagVlasovMaxVelocity(vd);
      vlasovBC->addDirichlet(t_n, tagMin, Function::zero());
      vlasovBC->addDirichlet(t_n, tagMax, Function::zero());
    }
    int tagInitialTime = this->tagVlasovInitialTime();
    vlasovBC->addDirichlet(t_n, tagInitialTime, -f0[i]); // minus sign thanks to outward normal on temporal interface
  }
  
  _vlasovSoln->setBC(vlasovBC);
  
  double n0;
  if (_enforceQuasiNeutrality)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_numSpecies > 1, std::invalid_argument, "For multi-species formulations, enforceQuasiNeutrality=true is not supported");
    auto configurationMesh   = _configurationSpaceFormulation->configurationSpaceMesh();
    const double meshMeasure = Function::constant(1.0)->integrate(configurationMesh);
    const double f_integral  = f0[0]->integrate(_vlasovSoln->mesh());
    n0 = f_integral / meshMeasure;
  }
  else
  {
    n0 = 0;
  }
  _n0->setValue(n0);
}

void VlasovSpaceTimeFormulation::setInitialAndInflowConditions(FunctionPtr f0, FunctionPtr f_in, double timeRampWidthForInitialConditions)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_numSpecies > 1, std::invalid_argument, "For multi-species formulations, must call the multi-species setInitialConditions");
  this->setInitialAndInflowConditions(vector<FunctionPtr>{f0}, vector<FunctionPtr>{f_in}, timeRampWidthForInitialConditions);
}

class TemporalRamp : public TFunction<double>
{
  double _rampWidth; // amount of time before the value is 1.0
  int _timeDim; // 0-based
public:
  TemporalRamp(double rampWidth, int timeDimension)
  :
  _rampWidth(rampWidth),
  _timeDim(timeDimension)
  {}
  
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
  {
    auto & spaceTimePoints = basisCache->getPhysicalCubaturePointsFC();
    const int numCells  = spaceTimePoints.dimension(0);
    const int numPoints = spaceTimePoints.dimension(1);
    
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double t = spaceTimePoints(cellOrdinal,pointOrdinal,_timeDim);
        double value;
        if (t < _rampWidth)
        {
          value = t / _rampWidth;
        }
        else
        {
          value = 1.0;
        }
        values(cellOrdinal,pointOrdinal) = value;
      }
    }
  }
  
  TFunctionPtr<double> dx() override
  {
    return Function::zero();
  }
  
  TFunctionPtr<double> dy() override
  {
    return di(1);
  }
  
  TFunctionPtr<double> dz() override
  {
    return di(2);
  }
  
  TFunctionPtr<double> dt() override
  {
    return di(_timeDim + 1);
  }
  
  TFunctionPtr<double> di(int i) override
  {
    if (i == _timeDim + 1) // i is 1-based; _timeDim is 0-based
    {
      auto heavisideT = Function::heaviside(_rampWidth,2); // 1 for z < _rampWidth, 0 otherwise
      return (1.0 / _rampWidth) * heavisideT;
    }
    else
    {
      return Function::zero();
    }
  }
};

void VlasovSpaceTimeFormulation::setInitialAndInflowConditions(vector<FunctionPtr> f0, vector<FunctionPtr> f_in, double timeRampWidthForInitialConditions)
{
  _f0 = f0;
  _f_inflow = f_in;
  
  auto vlasovBC = BC::bc();
  
  for (int i=0; i<_numSpecies; i++)
  {
    VarPtr t_n = this->t_n_hat(i);
    for (int vd=0; vd<_velocityDim; vd++)
    {
      const int tagMin = this->tagVlasovMinVelocity(vd);
      const int tagMax = this->tagVlasovMaxVelocity(vd);
      vlasovBC->addDirichlet(t_n, tagMin, Function::zero());
      vlasovBC->addDirichlet(t_n, tagMax, Function::zero());
    }
    
    const int tagInitialTime = this->tagVlasovInitialTime();
    vlasovBC->addDirichlet(t_n, tagInitialTime, -f0[i]); // minus sign thanks to outward normal on temporal interface
  }
  
  FunctionPtr n = Function::normal();
  FunctionPtr v_dot_nx = Function::zero();
  for (int vd=0; vd<std::min(_velocityDim,_spaceDim); vd++)
  {
    FunctionPtr v_d = this->v_i(vd);
    FunctionPtr n_d = n->spatialComponent(vd+1); // 1-based index for Function::spatialComponent();
    v_dot_nx = v_dot_nx + v_d * n_d;
  }
  
  const int tagSpatialInflow = this->tagVlasovSpatialInflow();
  
  const int tCompZeroBased = _spaceDim;
  FunctionPtr timeRamp = Teuchos::rcp(new TemporalRamp(timeRampWidthForInitialConditions, tCompZeroBased));
  for (int i=0; i<_numSpecies; i++)
  {
    if (timeRampWidthForInitialConditions != 0.0)
    {
      
      f_in[i] = f_in[i] * timeRamp;
    }
    
    VarPtr t_n = this->t_n_hat(i);
    vlasovBC->addDirichlet(t_n, tagSpatialInflow, v_dot_nx * f_in[i]);
  }

  _vlasovSoln->setBC(vlasovBC);
  
  double n0;
  if (_enforceQuasiNeutrality)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_numSpecies > 1, std::invalid_argument, "For multi-species formulations, enforceQuasiNeutrality=true is not supported");
    auto configurationMesh   = _configurationSpaceFormulation->configurationSpaceMesh();
    const double meshMeasure = Function::constant(1.0)->integrate(configurationMesh);
    const double f_integral  = f0[0]->integrate(_vlasovSoln->mesh());
    n0 = f_integral / meshMeasure;
  }
  else
  {
    n0 = 0;
  }
  _n0->setValue(n0);
}

void VlasovSpaceTimeFormulation::setMaxMomentForConservationDiagnostics(const int maxMomentForConservationDiagnostics)
{
  _maxMomentForConservationDiagnostics = maxMomentForConservationDiagnostics;
  
  bool didReconstructSolutionMoments = false;
  for (int i=0; i<_numSpecies; i++)
  {
    if (_solutionMoments[i]->maxMomentNumber() < maxMomentForConservationDiagnostics)
    {
      // need to re-create _solutionMoments
      _solutionMoments[i] = Teuchos::rcp( new SolutionMoments(maxMomentForConservationDiagnostics, _vlasovSoln, f(i), _configurationSpaceFormulation->configurationSpaceMesh()) );
      didReconstructSolutionMoments = true;
    }
  }
  if (didReconstructSolutionMoments)
  {
    _configurationSpaceFormulation->setMoments(_solutionMoments,_n0);
  }
  
  for (int i=0; i<_numSpecies; i++)
  {
    _momentum[i].resize(_spaceDim);
    if (_spaceDim == 1)
    {
      _density[i]     = (maxMomentForConservationDiagnostics >= 0) ? _solutionMoments[i]->solutionMomentFunction(0) : Teuchos::null;
      _momentum[i][0] = (maxMomentForConservationDiagnostics >= 1) ? _solutionMoments[i]->solutionMomentFunction(1) : Teuchos::null;
      _energy[i]      = (maxMomentForConservationDiagnostics >= 2) ? _solutionMoments[i]->solutionMomentFunction(2) : Teuchos::null;
    }
    else if (_spaceDim == 2)
    {
      _density[i]     = (maxMomentForConservationDiagnostics >= 0) ? _solutionMoments[i]->solutionMomentFunction(0,0) : Teuchos::null;
      _momentum[i][0] = (maxMomentForConservationDiagnostics >= 1) ? _solutionMoments[i]->solutionMomentFunction(1,0) : Teuchos::null;
      _momentum[i][1] = (maxMomentForConservationDiagnostics >= 1) ? _solutionMoments[i]->solutionMomentFunction(0,1) : Teuchos::null;
      if (maxMomentForConservationDiagnostics < 2)
      {
        _energy[i] = Teuchos::null;
      }
      else
      {
        _energy[i] = _solutionMoments[i]->solutionMomentFunction(2,0) + _solutionMoments[i]->solutionMomentFunction(0,2);
      }
    }
    else if (_spaceDim == 3)
    {
      _density[i]     = (maxMomentForConservationDiagnostics >= 0) ? _solutionMoments[i]->solutionMomentFunction(0,0,0) : Teuchos::null;
      _momentum[i][0] = (maxMomentForConservationDiagnostics >= 1) ? _solutionMoments[i]->solutionMomentFunction(1,0,0) : Teuchos::null;
      _momentum[i][1] = (maxMomentForConservationDiagnostics >= 1) ? _solutionMoments[i]->solutionMomentFunction(0,1,0) : Teuchos::null;
      _momentum[i][2] = (maxMomentForConservationDiagnostics >= 1) ? _solutionMoments[i]->solutionMomentFunction(0,0,1) : Teuchos::null;
      if (maxMomentForConservationDiagnostics < 2)
      {
        _energy[i] = Teuchos::null;
      }
      else
      {
        _energy[i] = _solutionMoments[i]->solutionMomentFunction(2,0,0) + _solutionMoments[i]->solutionMomentFunction(0,2,0) + _solutionMoments[i]->solutionMomentFunction(0,0,2);
      }
    }
  }
}

void VlasovSpaceTimeFormulation::setVlasovGramAssemblyTimingCallback(std::function<void(int numElements, double timeG_integrate,
                                                                                        double time_computeCellMeasure, double time_getJacobianData, double time_configSpaceFunction, double time_intrepid2JacobianAllocAndCompute)> &gramAssemblyTimingCallback)
{
  VlasovSpaceTimeIP* ip = dynamic_cast<VlasovSpaceTimeIP*>(_vlasovSoln->ip().get());
  if (ip != NULL)
  {
    ip->setVlasovGramAssemblyTimingCallback(gramAssemblyTimingCallback);
  }
}

Teuchos::RCP<SolutionMoments> VlasovSpaceTimeFormulation::solutionMoments(int i)
{
  return _solutionMoments[i];
}

int VlasovSpaceTimeFormulation::spaceDim() const
{
  return _spaceDim;
}

void VlasovSpaceTimeFormulation::setSlabHistory(const std::vector<double>                                                                       &slabTemporalCoordinates,
                                                const std::vector<std::vector< Teuchos::RCP<SpatialIntegralOfSpaceTimeSolution> > >             &massIntegrals,
                                                const std::vector<std::vector<std::vector< Teuchos::RCP<SpatialIntegralOfSpaceTimeSolution> >>> &momentumIntegrals,
                                                const std::vector<std::vector< Teuchos::RCP<SpatialIntegralOfSpaceTimeSolution> > >             &energyIntegrals,
                                                const std::vector<std::vector< ElementwiseConservationDiagnostics > >                           &elementwiseMassIntegrals,
                                                const std::vector<std::vector< std::vector< ElementwiseConservationDiagnostics > > >            &elementwiseMomentumIntegrals,
                                                const std::vector<std::vector< ElementwiseConservationDiagnostics > >                           &elementwiseEnergyIntegrals,
                                                const std::vector<std::vector< double> >                                                        &w1ResidualNorms)
{
  _slabTemporalCoordinates = slabTemporalCoordinates;
  _massIntegrals = massIntegrals;
  _momentumIntegrals = momentumIntegrals;
  _energyIntegrals = energyIntegrals;
  
  _elementwiseMassIntegrals     = elementwiseMassIntegrals;
  _elementwiseMomentumIntegrals = elementwiseMomentumIntegrals;
  _elementwiseEnergyIntegrals   = elementwiseEnergyIntegrals;
  
  // last listed temporal coordinate should match our start coordinate
  double tol = 1e-14;
  double t0Diff = std::abs(_slabTemporalCoordinates[_slabTemporalCoordinates.size()-1] - _t0);
  TEUCHOS_TEST_FOR_EXCEPTION(t0Diff > tol, std::invalid_argument, "slabTemporalCoordinates: final entry must match this slab's start time");
  _slabTemporalCoordinates.push_back(_t1);
}

int VlasovSpaceTimeFormulation::slabOrdinal() const
{
  return _slabTemporalCoordinates.size() - 2;
}

// test variables:
VarPtr VlasovSpaceTimeFormulation::w(int i)
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->testVar(_w_names[i], HGRAD);
}

void VlasovSpaceTimeFormulation::setUseGeometricMultigrid(bool useGeometricMultigrid)
{
  _useGeometricMultigrid = useGeometricMultigrid;
}

MeshTopologyPtr VlasovSpaceTimeFormulation::spatialMeshTopology()
{
  return _spatialMeshTopology;
}

FunctionPtr VlasovSpaceTimeFormulation::t() const
{
  int tCompZeroBased = _spaceDim;
  return MonomialFunction::xi_n(tCompZeroBased, 1);
}

double VlasovSpaceTimeFormulation::t0() const
{
  return _t0;
}

double VlasovSpaceTimeFormulation::t1() const
{
  return _t1;
}

int VlasovSpaceTimeFormulation::tagConfigurationInitialTime() const
{
  return CONFIGURATION_TAG_TMIN;
}

int VlasovSpaceTimeFormulation::tagVlasovInitialTime() const
{
  return VLASOV_TAG_TMIN;
}

int VlasovSpaceTimeFormulation::tagVlasovSpatialInflow() const
{
  return VLASOV_TAG_SPATIAL_INFLOW;
}

int VlasovSpaceTimeFormulation::tagVlasovMaxVelocity(int zeroBasedComp) const
{
  switch (zeroBasedComp)
  {
    case 0: return VLASOV_TAG_VXMAX;
    case 1: return VLASOV_TAG_VYMAX;
    case 2: return VLASOV_TAG_VZMAX;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "zeroBasedComp must be 0, 1, 2")
  }
}

int VlasovSpaceTimeFormulation::tagVlasovMinVelocity(int zeroBasedComp) const
{
  switch (zeroBasedComp)
  {
    case 0: return VLASOV_TAG_VXMIN;
    case 1: return VLASOV_TAG_VYMIN;
    case 2: return VLASOV_TAG_VZMIN;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "zeroBasedComp must be 0, 1, 2")
  }
}

map<int,FunctionPtr> VlasovSpaceTimeFormulation::testForMassConservation(int i)
{
  VarPtr w = this->w(i);
  map<int,FunctionPtr> testFunction;
  testFunction[w->ID()] = Function::constant(1.0);
  
  return testFunction;
}

map<int,FunctionPtr> VlasovSpaceTimeFormulation::testForEnergyConservation(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
}

map<int,FunctionPtr> VlasovSpaceTimeFormulation::testForXMomentumConservation(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
}

map<int,FunctionPtr> VlasovSpaceTimeFormulation::testForYMomentumConservation(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
}

map<int,FunctionPtr> VlasovSpaceTimeFormulation::testForZMomentumConservation(int i)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented");
}

map<int,int> VlasovSpaceTimeFormulation::testOrderEnhancements()
{
  map<int,int> orderEnhancements;
  return orderEnhancements;
}

map<int,int> VlasovSpaceTimeFormulation::trialOrderEnhancements()
{
  map<int,int> orderEnhancements;
  for (int i=0; i<_numSpecies; i++)
  {
    orderEnhancements[t_n_hat(i)->ID()] = 1; // +1 for non-conforming
  }
  return orderEnhancements;
}

vector<pair<double,double>> VlasovSpaceTimeFormulation::timeSeriesForSpatialIntegral(Teuchos::RCP<SpatialIntegralOfSpaceTimeSolution> integral,
                                                                                     const double &t0, const double &dt, const int &numValues) const
{
  auto temporalSolution         = integral->temporalSolution();
  auto temporalSolutionFunction = integral->temporalFunction();
  auto mesh = temporalSolution->mesh();
  auto activeCells = mesh->getTopology()->getLocallyKnownActiveCellIndices(); // we assume mesh geometry is not distributed
  temporalSolution->importSolutionForOffRankCells(activeCells);
  
  // confirm that the mesh geometry is not distributed:
  const int numActiveCells = activeCells.size();
  TEUCHOS_TEST_FOR_EXCEPTION(mesh->numActiveElements() != numActiveCells, std::invalid_argument, "Error: not all active cell indices are locally known");
  
  const double tol = 1e-10;
  
  std::map<IndexType, vector<double> > pointsForCell;
  
  const double t1 = t0 + dt * (numValues - 1);
  
  vector<double> allPoints;
  {
    double t = t0;
    int t_ordinal = 0;
    while (t <= t1 + tol)
    {
      allPoints.push_back(t);
      t_ordinal++;
      t = t0 + t_ordinal * dt;
    }
  }
  const int numPoints = allPoints.size();
  
  vector<pair<double,double>> fxnValues(numPoints);
  if (numPoints == 0) return fxnValues;

  BasisCachePtr basisCache = BasisCache::basisCacheForReferenceCell(CellTopology::line(), 1);
  basisCache->setMesh(mesh);

  auto meshTopo = mesh->getTopology();
  
  FieldContainer<double> vertices(2,1); // (N,D)
  for (const auto &cellID : activeCells)
  {
    meshTopo->verticesForCell(vertices, cellID);
    double t_left  = vertices(0,0);
    double t_right = vertices(1,0);
    if (t_right != t1)
    {
      // rightmost cell owns its final point; intermediate cells do not
      t_right -= tol * 2;
    }

    // determine the first matching point that intersects the cell (if any)
    int t_firstOrdinal = (t_left - t0 + tol) / dt; // in allPoints
    if (allPoints[t_firstOrdinal] + tol < t_left)
    {
      // then go to the next point
      t_firstOrdinal = (t_left - t0 + dt + tol) / dt; // in allPoints
    }
    if (t_firstOrdinal >= numPoints) continue; // can't have any matches in this cell, then
    TEUCHOS_TEST_FOR_EXCEPTION(allPoints[t_firstOrdinal] + tol < t_left, std::logic_error, "logic error determining t_firstOrdinal");

    vector<double> points;

    double t = dt * t_firstOrdinal + t0;
    while (t < t_right + tol)
    {
      points.push_back(t);
      t += dt;
    }
    pointsForCell[cellID] = points;

    double refCellScaling = 2.0 / (t_right - t_left);
    int numPointsForCell = points.size();
    
    if (numPointsForCell == 0) continue;
    
    FieldContainer<double> refCellPoints(numPointsForCell,1);
    for (int pointOrdinal=0; pointOrdinal<numPointsForCell; pointOrdinal++)
    {
      const auto &point = points[pointOrdinal];
      double refPoint = (point - t_left) * refCellScaling - 1.0; // -1.0 is the image of t_left on the reference cell
      refCellPoints(pointOrdinal,0) = refPoint;
    }

    FieldContainer<double> cellFxnValues(1,numPointsForCell);
    auto physicalCellNodes = mesh->physicalCellNodesForCell(cellID);

    basisCache->setRefCellPointsFC(refCellPoints);
    basisCache->setPhysicalCellNodes(physicalCellNodes, {cellID}, false);
    temporalSolutionFunction->values(cellFxnValues, basisCache);

    for (int pointOrdinal=0; pointOrdinal<numPointsForCell; pointOrdinal++)
    {
      fxnValues[t_firstOrdinal + pointOrdinal] = {points[pointOrdinal],cellFxnValues(0,pointOrdinal)};
    }
  }
  return fxnValues;
}

vector<pair<double,double>> VlasovSpaceTimeFormulation::timeSeriesForSlabSpatialIntegrals(const std::vector<Teuchos::RCP<SpatialIntegralOfSpaceTimeSolution>> &integrals,
                                                                                          const double &t0, const double &dt, const int &numValues) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(integrals.size() != _slabTemporalCoordinates.size() - 1, std::invalid_argument, "integrals container is not the expected size.  computeConservationDiagnostics() may not have been called.");
  vector<pair<double,double>> allValues;
  const int numSlabs = _slabTemporalCoordinates.size() - 1;
  double t = t0;
  double tol = 1e-14;
  
  for (int slabNumber=0; slabNumber<numSlabs; slabNumber++)
  {
    double slabStart = _slabTemporalCoordinates[slabNumber];
    double slabEnd   = _slabTemporalCoordinates[slabNumber+1];
    double slabWidth = slabEnd - slabStart;
    
    double initialTimeOffset = t - slabStart;
    TEUCHOS_TEST_FOR_EXCEPTION(initialTimeOffset < -tol, std::invalid_argument, "Start time is prior to slab start");
//    while (initialTimeOffset < -tol) // t is before slab start (should only happen for initial slab, and then only if called with a t0 prior to start time)
//    {
//      // pad with zeros for unknown values prior to slab start
//      allValues.push_back({t,0});
//      t += dt;
//      initialTimeOffset = t - slabStart;
//    }
    if (initialTimeOffset >= slabWidth) continue; // we're into the next slab, at least
    
    int slabNumValues;
    if (slabNumber < numSlabs - 1)
    {
      slabNumValues = (slabWidth - initialTimeOffset) / dt;
    }
    else
    {
      slabNumValues = numValues - allValues.size();
    }
    
    auto slabValues = timeSeriesForSpatialIntegral(integrals[slabNumber], t, dt, slabNumValues);
    
    allValues.insert(allValues.end(), slabValues.begin(), slabValues.end());
    t += slabNumValues * dt;
  }
  return allValues;
}

VarPtr VlasovSpaceTimeFormulation::t_n_hat(int i)
{
  VarFactoryPtr vf = _vlasovBF->varFactory();
  return vf->trialVar(_tn_hat_names[i]);
}

vector< pair<double,double> > VlasovSpaceTimeFormulation::totalEnergyTimeSeries(const double t0, const double dt, const int numValues, const int speciesOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_maxMomentForConservationDiagnostics < 2, std::invalid_argument, "maxMomentForConservationDiagnostics must have been set to 2 to enable totalEnergyTimeSeries()");
  return timeSeriesForSlabSpatialIntegrals(_energyIntegrals[speciesOrdinal], t0, dt, numValues);
}

vector< pair<double,double> > VlasovSpaceTimeFormulation::totalMassTimeSeries(const double t0, const double dt, const int numValues, const int speciesOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_maxMomentForConservationDiagnostics < 0, std::invalid_argument, "maxMomentForConservationDiagnostics must have been set >= 0 to enable totalMassTimeSeries()");
  return timeSeriesForSlabSpatialIntegrals(_massIntegrals[speciesOrdinal], t0, dt, numValues);
}

vector< pair<double,double> > VlasovSpaceTimeFormulation::totalXMomentumTimeSeries(const double t0, const double dt, const int numValues, const int speciesOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_maxMomentForConservationDiagnostics < 1, std::invalid_argument, "maxMomentForConservationDiagnostics must have been set >= 1 to enable totalXMomentumTimeSeries()");
  return timeSeriesForSlabSpatialIntegrals(_momentumIntegrals[speciesOrdinal][0], t0, dt, numValues);
}

vector< pair<double,double> > VlasovSpaceTimeFormulation::totalYMomentumTimeSeries(const double t0, const double dt, const int numValues, const int speciesOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim < 2, std::invalid_argument, "spatial dimension must be >= 2 for totalYMomentumTimeSeries()");
  TEUCHOS_TEST_FOR_EXCEPTION(_maxMomentForConservationDiagnostics < 1, std::invalid_argument, "maxMomentForConservationDiagnostics must have been set >= 1 to enable totalYMomentumTimeSeries()");
  return timeSeriesForSlabSpatialIntegrals(_momentumIntegrals[speciesOrdinal][1], t0, dt, numValues);
}

vector< pair<double,double> > VlasovSpaceTimeFormulation::totalZMomentumTimeSeries(const double t0, const double dt, const int numValues, const int speciesOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim < 3, std::invalid_argument, "spatial dimension must be 3 for totalZMomentumTimeSeries()");
  TEUCHOS_TEST_FOR_EXCEPTION(_maxMomentForConservationDiagnostics < 1, std::invalid_argument, "maxMomentForConservationDiagnostics must have been set >= 1 to enable totalZMomentumTimeSeries()");
  return timeSeriesForSlabSpatialIntegrals(_momentumIntegrals[speciesOrdinal][2], t0, dt, numValues);
}

bool VlasovSpaceTimeFormulation::useGeometricMultigrid() const
{
  return _useGeometricMultigrid;
}

FunctionPtr VlasovSpaceTimeFormulation::v_i(int compZeroBased) const
{
  int vx_component_zero_based = _spaceDim + 1; // plus 1 for temporal dimension
  return MonomialFunction::xi_n(vx_component_zero_based + compZeroBased, 1);
}

VarPtr VlasovSpaceTimeFormulation::VE_configurationSpaceTrace()
{
  return _configurationSpaceFormulation->VE_hat();
}

int VlasovSpaceTimeFormulation::velocityDim() const
{
  return _velocityDim;
}

vector<double> VlasovSpaceTimeFormulation::velocitySpaceVertices_x() const
{
  return _velocitySpaceVertices_x;
}

vector<double> VlasovSpaceTimeFormulation::velocitySpaceVertices_y() const
{
  return _velocitySpaceVertices_y;
}

vector<double> VlasovSpaceTimeFormulation::velocitySpaceVertices_z() const
{
  return _velocitySpaceVertices_z;
}

std::string VlasovSpaceTimeFormulation::vizOutputDir() const
{
  return _vizOutputDir;
}

BFPtr VlasovSpaceTimeFormulation::vlasovBF()
{
  return _vlasovBF;
}

FunctionPtr VlasovSpaceTimeFormulation::x_i(const int zeroBasedComp) const
{
  return MonomialFunction::xi_n(zeroBasedComp, 1);
}
