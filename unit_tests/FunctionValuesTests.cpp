//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  FunctionValuesTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/3/24.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "CamelliaDataHelpers.h"
#include "CamelliaTestingHelpers.h"
#include "Function.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "SharedScratchStorage.h"

#include <Intrepid2_Data.hpp>
#include <Intrepid2_TensorArgumentIterator.hpp>

using namespace Camellia;

namespace
{
  class TestFunction : public TFunction<double>
  {
    bool _overrideValuesWithDataArg;
    bool _overrideValuesWithFCArg;
    bool _overrideValuesWithViewArg;
    
  public:
    TestFunction(bool overrideValuesWithDataArg,
                 bool overrideValuesWithFCArg,
                 bool overrideValuesWithViewArg,
                 int variationDimensions,
                 int rank)
    :
    TFunction<double>("FunctionValuesTests::TestFunction", rank, variationDimensions),
    _overrideValuesWithDataArg(overrideValuesWithDataArg),
    _overrideValuesWithFCArg(overrideValuesWithFCArg),
    _overrideValuesWithViewArg(overrideValuesWithViewArg)
    {}
    
    KOKKOS_INLINE_FUNCTION
    void setDataEntry(const Intrepid2::Data<double, Camellia::DeviceType> &data,
                      Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> &dataArgs,
                      const int &numDataArgs,
                      const double &value)
    {
      double* data_entry;
      switch (numDataArgs)
      {
        case 1: data_entry = &data.getWritableEntry(dataArgs[0]); break;
        case 2: data_entry = &data.getWritableEntry(dataArgs[0],dataArgs[1]); break;
        case 3: data_entry = &data.getWritableEntry(dataArgs[0],dataArgs[1],dataArgs[2]); break;
        case 4: data_entry = &data.getWritableEntry(dataArgs[0],dataArgs[1],dataArgs[2],dataArgs[3]); break;
        case 5: data_entry = &data.getWritableEntry(dataArgs[0],dataArgs[1],dataArgs[2],dataArgs[3],dataArgs[4]); break;
        case 6: data_entry = &data.getWritableEntry(dataArgs[0],dataArgs[1],dataArgs[2],dataArgs[3],dataArgs[4],dataArgs[5]); break;
        case 7: data_entry = &data.getWritableEntry(dataArgs[0],dataArgs[1],dataArgs[2],dataArgs[3],dataArgs[4],dataArgs[5],dataArgs[6]); break;
      }
      *data_entry = value;
    }
    
    // Data-arg values()
    void values(Intrepid2::Data<double, Camellia::DeviceType> values, BasisCachePtr basisCache) override
    {
      if (! _overrideValuesWithDataArg)
      {
        this->TFunction<double>::values(values, basisCache);
        return;
      }
      else
      {
        const auto & points = basisCache->getPhysicalCubaturePoints(); // (C,P,D) TensorPoints container
        const int numCells  = points.extent_int(0);
        const int numPoints = points.extent_int(1);
        const int spaceDim = points.extent_int(2);
        auto rangePolicy = FieldContainer<double>::getMDRangePolicy<2>({numCells,numPoints});
       
        Kokkos::Array<int,7> variesInDimensionEnumValues;
        variesInDimensionEnumValues[0] = VARIES_IN_DIMENSION_0;
        variesInDimensionEnumValues[1] = VARIES_IN_DIMENSION_1;
        variesInDimensionEnumValues[2] = VARIES_IN_DIMENSION_2;
        variesInDimensionEnumValues[3] = VARIES_IN_DIMENSION_3;
        variesInDimensionEnumValues[4] = VARIES_IN_DIMENSION_4;
        variesInDimensionEnumValues[5] = VARIES_IN_DIMENSION_5;
        variesInDimensionEnumValues[6] = VARIES_IN_DIMENSION_6;
        
        int variationDimensions = _variationDimensions;
        
        Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> pointBoundsArray;
        for (int d=0; d<spaceDim; d++)
        {
          pointBoundsArray[d] = points.getTensorComponent(d).extent_int(1); // each component has shape (C,P,D)
        }
        for (int d=spaceDim; d<Intrepid2::Parameters::MaxTensorComponents; d++)
        {
          pointBoundsArray[d] = 1;
        }
        
        const int rank = _rank;
        if (rank == 0)
        {
          Kokkos::parallel_for("TestFunction values() functor", rangePolicy,
                               KOKKOS_LAMBDA (const int i0, const int i1) {
            double value = 1.0;
            for (int d=0; d<spaceDim; d++)
            {
              bool variesInDimension = variationDimensions & variesInDimensionEnumValues[d];
              const auto xi = points(i0,i1,d);
              const double xiFactor = variesInDimension ? xi : 1.0;
              value *= xiFactor;
            }
            
            Intrepid2::TensorArgumentIterator pi(pointBoundsArray); // point iterator; values has shape (C,P1[,P2,…][,D,…]) where P = P1 * P2 * …
            
            pi.setEnumerationIndex(i1); // pointOrdinal
            
            Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> dataArgs;
            for (int d=0; d<Intrepid2::Parameters::MaxTensorComponents; d++)
            {
              dataArgs[d] = 0;
            }
            dataArgs[0] = i0;
            for (int d=1; d<spaceDim+1; d++)
            {
              dataArgs[d] = pi.argument(d-1);
            }
            const int numDataArgs = spaceDim + 1 + rank;
            setDataEntry(values, dataArgs, numDataArgs, value);
          });
        }
        else if (rank == 1)
        {
          Kokkos::parallel_for("TestFunction values() functor", rangePolicy,
                               KOKKOS_LAMBDA (const int i0, const int i1) {
            Intrepid2::TensorArgumentIterator pi(pointBoundsArray); // point iterator; values has shape (C,P1[,P2,…][,D,…]) where P = P1 * P2 * …
            
            pi.setEnumerationIndex(i1); // pointOrdinal
            
            Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> dataArgs;
            for (int d=0; d<Intrepid2::Parameters::MaxTensorComponents; d++)
            {
              dataArgs[d] = 0;
            }
            dataArgs[0] = i0;
            for (int d=1; d<spaceDim+1; d++)
            {
              dataArgs[d] = pi.argument(d-1);
            }
            const int numDataArgs = spaceDim + 1 + rank;
            
            double value = 1.0;
            for (int d=0; d<spaceDim; d++)
            {
              bool variesInDimension = variationDimensions & variesInDimensionEnumValues[d];
              const auto xi = points(i0,i1,d);
              const double xiFactor = variesInDimension ? xi : 1.0;
              value *= xiFactor;
            }
            for (int comp=0; comp<spaceDim; comp++)
            {
              double value1 = pow(value,comp+1);
              
              dataArgs[spaceDim+1] = comp;
              setDataEntry(values, dataArgs, numDataArgs, value1);
            }
          });
        }
        else if (rank == 2)
        {
          Kokkos::parallel_for("TestFunction values() functor", rangePolicy,
                               KOKKOS_LAMBDA (const int i0, const int i1) {
            Intrepid2::TensorArgumentIterator pi(pointBoundsArray); // point iterator; values has shape (C,P1[,P2,…][,D,…]) where P = P1 * P2 * …
            
            pi.setEnumerationIndex(i1); // pointOrdinal
            
            Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> dataArgs;
            for (int d=0; d<Intrepid2::Parameters::MaxTensorComponents; d++)
            {
              dataArgs[d] = 0;
            }
            dataArgs[0] = i0;
            for (int d=1; d<spaceDim+1; d++)
            {
              dataArgs[d] = pi.argument(d-1);
            }
            const int numDataArgs = spaceDim + 1 + rank;
            
            double value = 1.0;
            for (int d=0; d<spaceDim; d++)
            {
              bool variesInDimension = variationDimensions & variesInDimensionEnumValues[d];
              const auto xi = points(i0,i1,d);
              const double xiFactor = variesInDimension ? xi : 1.0;
              value *= xiFactor;
            }
            
            for (int comp1=0; comp1<spaceDim; comp1++)
            {
              dataArgs[spaceDim+1] = comp1;
              const double value1 = pow(value,comp1+1);
              
              for (int comp2=0; comp2<spaceDim; comp2++)
              {
                dataArgs[spaceDim+2] = comp2;
                setDataEntry(values, dataArgs, numDataArgs, value1 + comp2);
              }
            }
          });
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported function rank");
        }
      }
    }
    
    // FC-arg values()
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      if (! _overrideValuesWithFCArg)
      {
        this->TFunction<double>::values(values, basisCache);
        return;
      }
      else
      {
        const auto & pointsFC = basisCache->getPhysicalCubaturePointsFC(); // (C,P,D) FieldContainer
        const int numCells  = pointsFC.dimension(0);
        const int numPoints = pointsFC.dimension(1);
        const int spaceDim  = pointsFC.dimension(2);
        
        Kokkos::Array<int,7> variesInDimensionEnumValues;
        variesInDimensionEnumValues[0] = VARIES_IN_DIMENSION_0;
        variesInDimensionEnumValues[1] = VARIES_IN_DIMENSION_1;
        variesInDimensionEnumValues[2] = VARIES_IN_DIMENSION_2;
        variesInDimensionEnumValues[3] = VARIES_IN_DIMENSION_3;
        variesInDimensionEnumValues[4] = VARIES_IN_DIMENSION_4;
        variesInDimensionEnumValues[5] = VARIES_IN_DIMENSION_5;
        variesInDimensionEnumValues[6] = VARIES_IN_DIMENSION_6;
        
        int variationDimensions = _variationDimensions;
        
        if (_rank == 0)
        {
          for (int i0=0; i0<numCells; i0++)
          {
            for (int i1=0; i1<numPoints; i1++)
            {
              double value = 1.0;
              for (int d=0; d<spaceDim; d++)
              {
                bool variesInDimension = variationDimensions & variesInDimensionEnumValues[d];
                const auto xi = pointsFC(i0,i1,d);
                const double xiFactor = variesInDimension ? xi : 1.0;
                value *= xiFactor;
              }
              
              values(i0,i1) = value;
            }
          }
        }
        else if (_rank == 1)
        {
          for (int i0=0; i0<numCells; i0++)
          {
            for (int i1=0; i1<numPoints; i1++)
            {
              for (int comp=0; comp<spaceDim; comp++)
              {
                double value = 1.0;
                for (int d=0; d<spaceDim; d++)
                {
                  bool variesInDimension = variationDimensions & variesInDimensionEnumValues[d];
                  const auto xi = pointsFC(i0,i1,d);
                  const double xiFactor = variesInDimension ? xi : 1.0;
                  value *= xiFactor;
                }
                value = pow(value,comp+1);
                
                values(i0,i1,comp) = value;
              }
            }
          }
        }
        else if (_rank == 2)
        {
          for (int i0=0; i0<numCells; i0++)
          {
            for (int i1=0; i1<numPoints; i1++)
            {
              for (int comp1=0; comp1<spaceDim; comp1++)
              {
                for (int comp2=0; comp2<spaceDim; comp2++)
                {
                  double value = 1.0;
                  for (int d=0; d<spaceDim; d++)
                  {
                    bool variesInDimension = variationDimensions & variesInDimensionEnumValues[d];
                    const auto xi = pointsFC(i0,i1,d);
                    const double xiFactor = variesInDimension ? xi : 1.0;
                    value *= xiFactor;
                  }
                  value = pow(value,comp1+1);
                  value += comp2;
                  
                  values(i0,i1,comp1,comp2) = value;
                }
              }
            }
          }
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported function rank");
        }
      }
    }
    
    void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override
    {
      if (! _overrideValuesWithViewArg)
      {
        this->TFunction<double>::values(values, modType, basisCache);
        return;
      }
      else
      {
        const auto & points = basisCache->getPhysicalCubaturePoints(); // (C,P,D) TensorPoints container
        const int numCells  = points.extent_int(0);
        const int numPoints = points.extent_int(1);
        const int spaceDim  = points.extent_int(2);
        auto rangePolicy = FieldContainer<double>::getMDRangePolicy<2>({numCells,numPoints});
       
        Kokkos::Array<int,7> variesInDimensionEnumValues;
        variesInDimensionEnumValues[0] = VARIES_IN_DIMENSION_0;
        variesInDimensionEnumValues[1] = VARIES_IN_DIMENSION_1;
        variesInDimensionEnumValues[2] = VARIES_IN_DIMENSION_2;
        variesInDimensionEnumValues[3] = VARIES_IN_DIMENSION_3;
        variesInDimensionEnumValues[4] = VARIES_IN_DIMENSION_4;
        variesInDimensionEnumValues[5] = VARIES_IN_DIMENSION_5;
        variesInDimensionEnumValues[6] = VARIES_IN_DIMENSION_6;
        
        int variationDimensions = _variationDimensions;
        
        if (_rank == 0)
        {
          Kokkos::parallel_for("TestFunction values() functor", rangePolicy,
                               KOKKOS_LAMBDA (const int i0, const int i1) {
            auto & entry = values(i0,i1); // (C,P)
            double value = 1.0;
            for (int d=0; d<spaceDim; d++)
            {
              bool variesInDimension = variationDimensions & variesInDimensionEnumValues[d];
              const auto xi = points(i0,i1,d);
              const double xiFactor = variesInDimension ? xi : 1.0;
              value *= xiFactor;
            }
            assignValue(entry, modType, value);
          });
        }
        else if (_rank == 1)
        {
          Kokkos::parallel_for("TestFunction values() functor", rangePolicy,
                               KOKKOS_LAMBDA (const int i0, const int i1) {
            for (int comp=0; comp<spaceDim; comp++)
            {
              auto & entry = values(i0,i1,comp); // (C,P,D)
              double value = 1.0;
              for (int d=0; d<spaceDim; d++)
              {
                bool variesInDimension = variationDimensions & variesInDimensionEnumValues[d];
                const auto xi = points(i0,i1,d);
                const double xiFactor = variesInDimension ? xi : 1.0;
                value *= xiFactor;
              }
              value = pow(value,comp+1);
              
              assignValue(entry, modType, value);
            }
          });
        }
        else if (_rank == 2)
        {
          Kokkos::parallel_for("TestFunction values() functor", rangePolicy,
                               KOKKOS_LAMBDA (const int i0, const int i1) {
            for (int comp1=0; comp1<spaceDim; comp1++)
            {
              for (int comp2=0; comp2<spaceDim; comp2++)
              {
                auto & entry = values(i0,i1,comp1,comp2); // (C,P,D1,D2)
                double value = 1.0;
                for (int d=0; d<spaceDim; d++)
                {
                  bool variesInDimension = variationDimensions & variesInDimensionEnumValues[d];
                  const auto xi = points(i0,i1,d);
                  const double xiFactor = variesInDimension ? xi : 1.0;
                  value *= xiFactor;
                }
                value = pow(value,comp1+1);
                value += comp2;
                
                assignValue(entry, modType, value);
              }
            }
          });
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported function rank");
        }
      }
    }
  };

  template<int spaceDim, int rank>
  void testFunctionValues(bool overrideValuesWithDataArg,
                          bool overrideValuesWithFCArg,
                          bool overrideValuesWithViewArg,
                          int variationDimensions,
                          int meshWidth,
                          double tol,
                          Teuchos::FancyOStream &out, bool &success)
  {
    // baselineFunction overrides all the values variants
    auto baselineFunction = Teuchos::rcp(new TestFunction(true, true, true, variationDimensions, rank));
    auto functionUnderTest = Teuchos::rcp(new TestFunction(overrideValuesWithDataArg, overrideValuesWithFCArg,
                                                           overrideValuesWithViewArg, variationDimensions, rank));
    
    // hypercube test mesh with 2^n elements
    PoissonFormulation poissonForm(spaceDim, true, PoissonFormulation::PRIMAL);
    std::vector<double> meshDims(spaceDim, 1.0);
    std::vector<int> elementCounts(spaceDim, meshWidth);
    int H1Order = 1;
    int delta_k = 1;
    auto meshTopo = MeshFactory::hypercubeMeshTopology(meshDims, elementCounts);
    auto mesh     = MeshFactory::minRuleMesh(meshTopo, poissonForm.bf(), H1Order, delta_k);
    auto elemType = mesh->elementTypes()[0];
    BasisCachePtr basisCache = BasisCache::basisCacheForCellType(mesh, elemType);
    
    int numCells   = basisCache->getPhysicalCubaturePoints().extent_int(0);
    int numPoints  = basisCache->getPhysicalCubaturePoints().extent_int(1);
    
    Teuchos::Array<int> dimensions(2 + rank);
    dimensions[0] = numCells;
    dimensions[1] = numPoints;
    for (int r=0; r<rank; r++)
    {
      dimensions[2+r] = spaceDim;
    }
    
    auto ssStorage = SharedScratchStorage<double>::sharedScratchStoragePtr();
    
    auto baselineData = baselineFunction->checkoutValuesData(basisCache);
    FieldContainer<double> baselineFC(dimensions);
    auto baselineView = ssStorage->checkoutScratchStorage(baselineFunction, basisCache);
    
    baselineFunction->values(baselineData, basisCache);
    baselineFunction->values(  baselineFC, basisCache);
    baselineFunction->values(baselineView, ASSIGN, basisCache);
    
    auto testFunctionData = functionUnderTest->checkoutValuesData(basisCache);
    FieldContainer<double> testFunctionFC(dimensions);
    auto testFunctionView = ssStorage->checkoutScratchStorage(functionUnderTest, basisCache);
    
    functionUnderTest->values(testFunctionData, basisCache);
    functionUnderTest->values(  testFunctionFC, basisCache);
    functionUnderTest->values(testFunctionView, ASSIGN, basisCache);
    
    constexpr int expandedRank = spaceDim+1+rank;
    
    //template<class Scalar, class DataContainer, int expandedRank, int startCollapseDim, int lastCollapseDim>
    using ViewInterpreter = ViewCollapseDimensions<double,DeviceView<double>,expandedRank,1,spaceDim>; // maps (C,P0,…,PD[,D0,…]) --> (C,P[,D0,…])
    
    static_assert(Intrepid2::supports_rank<ViewInterpreter, expandedRank>::value, "ViewInterpreter should support rank expandedRank");
    
    Kokkos::Array<int,spaceDim> pointComponentBounds;
    for (int d=1; d<=spaceDim; d++)
    {
      pointComponentBounds[d-1] = baselineData.extent_int(d);
    }
    
    // 1. sanity check: the baseline implementations agree
    // data nominal shape is (C,P0,…,PD); we interpret the (C,P) views
    ViewInterpreter baselineViewExpanded(baselineView, pointComponentBounds);
    testFloatingEquality<expandedRank>(baselineViewExpanded, baselineData, tol, tol, out, success, "baselineView", "baselineData");
    
    auto baselineFCView = baselineFC.checkoutDeviceView();
    ViewInterpreter baselineFCViewExpanded(baselineFCView, pointComponentBounds);
    testFloatingEquality<expandedRank>(baselineFCViewExpanded, baselineData, tol, tol, out, success, "baselineFCView", "baselineData");
    
    testFloatingEquality<rank+2>(baselineFCView, baselineView, tol, tol, out, success, "baselineFC", "baselineView");
    
    // 2. test that the testFunction* containers agree with each other
    ViewInterpreter testFunctionViewExpanded(testFunctionView, pointComponentBounds);
    testFloatingEquality<expandedRank>(testFunctionViewExpanded, testFunctionData, tol, tol, out, success, "testFunctionView", "testFunctionData");
    
    auto testFunctionFCView = testFunctionFC.checkoutDeviceView();
    ViewInterpreter testFunctionFCViewExpanded(testFunctionFCView, pointComponentBounds);
    testFloatingEquality<expandedRank>(testFunctionFCViewExpanded, testFunctionData, tol, tol, out, success, "testFunctionFCView", "testFunctionData");
    
    testFloatingEquality<rank+2>(testFunctionFCView, testFunctionView, tol, tol, out, success, "testFunctionFC", "testFunctionView");
    
    // 3. test that the testFunction* containers agree with their baseline counterparts
    testFloatingEquality<expandedRank>(baselineData,   testFunctionData, tol, tol, out, success,   "baselineData",   "testFunctionData");
    testFloatingEquality<rank+2>(      baselineView,   testFunctionView, tol, tol, out, success,   "baselineView",   "testFunctionView");
    testFloatingEquality<rank+2>(    baselineFCView, testFunctionFCView, tol, tol, out, success, "baselineFCView", "testFunctionFCView");
        
    baselineFunction->checkinValuesData(baselineData);
    ssStorage->checkinScratchStorage(baselineView);
    baselineFC.checkinDeviceView(false);
    
    functionUnderTest->checkinValuesData(testFunctionData);
    ssStorage->checkinScratchStorage(testFunctionView);
    testFunctionFC.checkinDeviceView(false);
  }

  TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL(FunctionValues, Exhaustive, DimTag, RankTag)
  {
    const int spaceDim  = DimTag::spaceDim;
    const int      rank = RankTag::rank;
    
    const double tol = 1e-14;
    
    const int meshWidth = (spaceDim <= 3) ? 2 : 1;
    
    Kokkos::Array<int,7> variesInDimensionEnumValues;
    variesInDimensionEnumValues[0] = Function::VARIES_IN_DIMENSION_0;
    variesInDimensionEnumValues[1] = Function::VARIES_IN_DIMENSION_1;
    variesInDimensionEnumValues[2] = Function::VARIES_IN_DIMENSION_2;
    variesInDimensionEnumValues[3] = Function::VARIES_IN_DIMENSION_3;
    variesInDimensionEnumValues[4] = Function::VARIES_IN_DIMENSION_4;
    variesInDimensionEnumValues[5] = Function::VARIES_IN_DIMENSION_5;
    variesInDimensionEnumValues[6] = Function::VARIES_IN_DIMENSION_6;
    
    int variationDimensionsMax = Function::VARIES_IN_NO_DIMENSION;
    for (int d=0; d<spaceDim; d++)
    {
      variationDimensionsMax |= variesInDimensionEnumValues[d];
    }
    
    // we require at least one of the methods to be overridden, and up to all three may be
    const bool DATA_ON = true, DATA_OFF = false;
    const bool   FC_ON = true,   FC_OFF = false;
    const bool VIEW_ON = true, VIEW_OFF = false;
    
    struct TestCase
    {
      bool supports_data;
      bool supports_fc;
      bool supports_view;
    };
    
    std::vector<TestCase> testCases { {DATA_OFF, FC_OFF, VIEW_ON },
                                      {DATA_OFF, FC_ON,  VIEW_OFF},
                                      {DATA_OFF, FC_ON,  VIEW_ON },
                                      { DATA_ON, FC_OFF, VIEW_OFF},
                                      { DATA_ON, FC_OFF, VIEW_ON },
                                      { DATA_ON,  FC_ON, VIEW_ON } };
    
    for (auto tc : testCases)
    {
      out << "******* Data: ";
      if (tc.supports_data) out << " ON"; else out << "OFF";
      out << " / FC: ";
      if (tc.supports_fc)   out << " ON"; else out << "OFF";
      out << " / View: ";
      if (tc.supports_view) out << " ON"; else out << "OFF";
      out << " *******\n";
      
      out.flush();
      
      for (int variationDimensions=0; variationDimensions<=variationDimensionsMax; variationDimensions++)
      {
        out << "variationDimensions: ";
        for (int d=0; d<spaceDim; d++)
        {
          bool varies_in_d = variationDimensions & variesInDimensionEnumValues[d];
          if (varies_in_d) out << "1"; else out << "0";
        }
        out << std::endl;
        out.flush();
        testFunctionValues<spaceDim, rank>(tc.supports_data, tc.supports_fc, tc.supports_view, variationDimensions, meshWidth, tol, out, success);
      }
    }
  }

  TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL(FunctionValues, Selective, DimTag, RankTag)
  {
    const int spaceDim  = DimTag::spaceDim;
    const int      rank = RankTag::rank;
    
    const int meshWidth = (spaceDim <= 3) ? 2 : 1;
    const double tol = 1e-14;
    
    Kokkos::Array<int,7> variesInDimensionEnumValues;
    variesInDimensionEnumValues[0] = Function::VARIES_IN_DIMENSION_0;
    variesInDimensionEnumValues[1] = Function::VARIES_IN_DIMENSION_1;
    variesInDimensionEnumValues[2] = Function::VARIES_IN_DIMENSION_2;
    variesInDimensionEnumValues[3] = Function::VARIES_IN_DIMENSION_3;
    variesInDimensionEnumValues[4] = Function::VARIES_IN_DIMENSION_4;
    variesInDimensionEnumValues[5] = Function::VARIES_IN_DIMENSION_5;
    variesInDimensionEnumValues[6] = Function::VARIES_IN_DIMENSION_6;
    
    int variationDimensionsNone = Function::VARIES_IN_NO_DIMENSION;
    int variationDimensionsAll  = Function::VARIES_IN_NO_DIMENSION;
    for (int d=0; d<spaceDim; d++)
    {
      variationDimensionsAll |= variesInDimensionEnumValues[d];
    }
    
    std::vector<int> selectiveVariations;
    selectiveVariations.push_back(variationDimensionsNone);
    for (int d=0; d<spaceDim; d++)
    {
      selectiveVariations.push_back(variesInDimensionEnumValues[d]);
    }
    selectiveVariations.push_back(variationDimensionsAll);
    
    // we require at least one of the methods to be overridden, and up to all three may be
    const bool DATA_ON = true, DATA_OFF = false;
    const bool   FC_ON = true,   FC_OFF = false;
    const bool VIEW_ON = true, VIEW_OFF = false;
    
    struct TestCase
    {
      bool supports_data;
      bool supports_fc;
      bool supports_view;
    };
    
    std::vector<TestCase> testCases { {DATA_OFF, FC_OFF, VIEW_ON },
                                      {DATA_OFF, FC_ON,  VIEW_OFF},
                                      { DATA_ON, FC_OFF, VIEW_OFF} };
    
    for (auto tc : testCases)
    {
      out << "******* Data: ";
      if (tc.supports_data) out << " ON"; else out << "OFF";
      out << " / FC: ";
      if (tc.supports_fc)   out << " ON"; else out << "OFF";
      out << " / View: ";
      if (tc.supports_view) out << " ON"; else out << "OFF";
      out << " *******\n";
      
      out.flush();
      
      for (int variationDimensions : selectiveVariations)
      {
        out << "variationDimensions: ";
        for (int d=0; d<spaceDim; d++)
        {
          bool varies_in_d = variationDimensions & variesInDimensionEnumValues[d];
          if (varies_in_d) out << "1"; else out << "0";
        }
        out << std::endl;
        out.flush();
        testFunctionValues<spaceDim, rank>(tc.supports_data, tc.supports_fc, tc.supports_view, variationDimensions, meshWidth, tol, out, success);
      }
    }
  }

//  TEUCHOS_UNIT_TEST( FunctionValues, D3_R0_OFF_ON_OFF_001 )
//  {
//    const int spaceDim = 3;
//    const int rank = 0;
//    const double tol = 1e-14;
//    
//    int variationDimensions = Function::VARIES_IN_DIMENSION_2;
//    bool overrideValuesWithDataArg = false;
//    bool overrideValuesWithFCArg   = true;
//    bool overrideValuesWithViewArg = false;
//    
//    testFunctionValues<spaceDim, rank>(overrideValuesWithDataArg, overrideValuesWithFCArg, overrideValuesWithViewArg, variationDimensions, tol, out, success);
//  }

  TEUCHOS_UNIT_TEST( FunctionValues, D4_R2_OFF_ON_OFF_0000 )
  {
    const int spaceDim = 4;
    const int rank = 2;
    const int meshWidth = 2;
    const double tol = 1e-14;

    int variationDimensions = Function::VARIES_IN_DIMENSION_2;
    bool overrideValuesWithDataArg = false;
    bool overrideValuesWithFCArg   = true;
    bool overrideValuesWithViewArg = false;

    testFunctionValues<spaceDim, rank>(overrideValuesWithDataArg, overrideValuesWithFCArg, overrideValuesWithViewArg, variationDimensions, meshWidth, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( FunctionValues, D5_R0_OFF_ON_OFF_11111 )
  {
    const int spaceDim = 4;
    const int rank = 2;
    const int meshWidth = 1;
    const double tol = 1e-14;

    int variationDimensions = Function::VARIES_IN_DIMENSION_0 | Function::VARIES_IN_DIMENSION_1 | Function::VARIES_IN_DIMENSION_2 | Function::VARIES_IN_DIMENSION_3 | Function::VARIES_IN_DIMENSION_4;
    bool overrideValuesWithDataArg = false;
    bool overrideValuesWithFCArg   = true;
    bool overrideValuesWithViewArg = false;

    testFunctionValues<spaceDim, rank>(overrideValuesWithDataArg, overrideValuesWithFCArg, overrideValuesWithViewArg, variationDimensions, meshWidth, tol, out, success);
  }

  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Exhaustive, D1, R0)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Exhaustive, D1, R1)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Exhaustive, D1, R2)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Exhaustive, D2, R0)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Exhaustive, D2, R1)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Exhaustive, D2, R2)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Exhaustive, D3, R0)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Exhaustive, D3, R1)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Exhaustive, D3, R2)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Exhaustive, D4, R0)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Exhaustive, D4, R1)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Exhaustive, D4, R2)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Selective, D5, R0)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Selective, D5, R1)
//  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Selective, D5, R2)
  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Selective, D6, R0)
//  TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT(FunctionValues, Selective, D6, R1)
// right now, Intrepid2::supports_rank only supports up to rank 7.  Would be an easy enough patch to Intrepid2 to get up to rank 8; beyond that, Kokkos support is going to be limited…
} // anonymous namespace
