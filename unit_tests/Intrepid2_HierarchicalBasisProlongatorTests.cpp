// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_HierarchicalBasisProlongatorTests.cpp
    \brief  Tests against Intrepid2::HierarchicalBasisProlongator
    \author Created by N.V. Roberts.
 */

#include "Teuchos_UnitTestHarness.hpp"

#include <Intrepid2_TestUtils.hpp>

#include <Intrepid2_HierarchicalBasisProlongator.hpp>

#include <Intrepid2_CellTopology.hpp>
#include <Intrepid2_DerivedBasisFamily.hpp>

using namespace Intrepid2;

namespace
{
using CellTopoPtr = CellTopology::CellTopoPtr;

//! Tag definitions
class P1
{
public:
  static const int polyOrder = 1;
};
class P2
{
public:
  static const int polyOrder = 2;
};
class P3
{
public:
  static const int polyOrder = 3;
};
class P4
{
public:
  static const int polyOrder = 4;
};
class P5
{
public:
  static const int polyOrder = 5;
};
class P6
{
public:
  static const int polyOrder = 6;
};
class P7
{
public:
  static const int polyOrder = 7;
};
class P8
{
public:
  static const int polyOrder = 8;
};
class P9
{
public:
  static const int polyOrder = 9;
};

class LINE
{
public:
  static const inline CellTopoPtr cellTopoPtr = CellTopology::line();
};
class QUAD
{
public:
  static const inline CellTopoPtr cellTopoPtr = CellTopology::quad();
};
class TRI
{
public:
  static const inline CellTopoPtr cellTopoPtr = CellTopology::triangle();
};
class HEX
{
public:
  static const inline CellTopoPtr cellTopoPtr = CellTopology::hexahedron();
};
class TET
{
public:
  static const inline CellTopoPtr cellTopoPtr = CellTopology::tetrahedron();
};
class WEDGE
{
public:
  static const inline CellTopoPtr cellTopoPtr = CellTopology::wedge();
};

class HGRAD
{
public:
  static const Intrepid2::EFunctionSpace functionSpace = Intrepid2::FUNCTION_SPACE_HGRAD;
};
class HDIV
{
public:
  static const Intrepid2::EFunctionSpace functionSpace = Intrepid2::FUNCTION_SPACE_HDIV;
};
class HCURL
{
public:
  static const Intrepid2::EFunctionSpace functionSpace = Intrepid2::FUNCTION_SPACE_HCURL;
};
class HVOL
{
public:
  static const Intrepid2::EFunctionSpace functionSpace = Intrepid2::FUNCTION_SPACE_HVOL;
};

template<class BasisCoeffsViewType, int rankToTest, bool vectorValued>
void testProlongation(shards::CellTopology &cellTopo, const EFunctionSpace &fs,
                      const ordinal_type &pCoarse, const ordinal_type &pFine,
                      Teuchos::FancyOStream &out, bool &success)
{
  using DeviceType = typename BasisCoeffsViewType::device_type;
  using Scalar     = typename BasisCoeffsViewType::value_type;
  using BasisFamily = HierarchicalBasisFamily<DeviceType>;
  
  const double relTol=1e-13;
  const double absTol=1e-13;
  
  auto coarseBasis = getBasis< BasisFamily >(cellTopo, fs, pCoarse);
  auto fineBasis   = getBasis< BasisFamily >(cellTopo, fs, pFine);
  
  auto prolongator = HierarchicalBasisProlongator<DeviceType,Scalar>(coarseBasis, fineBasis);
  
  auto testPoints = getInputPointsView<Scalar, DeviceType>(cellTopo, pFine + 1);
  
  const ordinal_type coarseCardinality = coarseBasis->getCardinality();
  const ordinal_type   fineCardinality =   fineBasis->getCardinality();
  const ordinal_type numCells = coarseCardinality;
  
  using ExecutionSpace = typename DeviceType::execution_space;
  
  const ordinal_type numPoints = testPoints.extent_int(0);
  const ordinal_type spaceDim  = cellTopo.getDimension();
  
  Kokkos::DynRankView<Scalar, DeviceType> coarseValues, fineValues;
  
  if constexpr (vectorValued)
  {
    coarseValues = Kokkos::DynRankView<Scalar, DeviceType>("coarseValues", coarseCardinality, numPoints, spaceDim);
    fineValues   = Kokkos::DynRankView<Scalar, DeviceType>(  "fineValues",   fineCardinality, numPoints, spaceDim);
  }
  else
  {
    coarseValues = Kokkos::DynRankView<Scalar, DeviceType>("coarseValues", coarseCardinality, numPoints);
    fineValues   = Kokkos::DynRankView<Scalar, DeviceType>(  "fineValues",   fineCardinality, numPoints);
  }

  coarseBasis->getValues(coarseValues, testPoints);
  fineBasis  ->getValues(  fineValues, testPoints);
  
  if constexpr (rankToTest == 2)
  {
    // (C,F) use case
    // set up one cell per coarse dof, each with weight 1.0
    BasisCoeffsViewType coarseCoefficients("coarseCoefficients", numCells, coarseCardinality);
    BasisCoeffsViewType   fineCoefficients(  "fineCoefficients", numCells,   fineCardinality);
    
    // to check that fineCoefficients gets cleared appropriately by prolongator, fill it initially with 1's.
    Kokkos::deep_copy(fineCoefficients, 1.0);
    
    // coarseCoefficients will be 1 where cellOrdinal == dofOrdinal, 0 elsewhere
    Kokkos::deep_copy(coarseCoefficients, 0.0);
    
    Kokkos::RangePolicy<ExecutionSpace> rangePolicy(0,coarseCardinality);
    Kokkos::parallel_for(rangePolicy,
                         KOKKOS_LAMBDA(const ordinal_type &cellOrdinal)
                         {
      const ordinal_type &dofOrdinal = cellOrdinal;
      coarseCoefficients(cellOrdinal,dofOrdinal) = 1.0;
    });
    
    prolongator.apply(fineCoefficients, coarseCoefficients); // exercises (C,F) use case, with sumInto = false
    
    Kokkos::DynRankView<Scalar, DeviceType> coarsePointEvaluations, finePointEvaluations;
    
    if (vectorValued)
    {
      coarsePointEvaluations = Kokkos::DynRankView<Scalar, DeviceType>("coarseValues", numCells, numPoints, spaceDim);
      finePointEvaluations   = Kokkos::DynRankView<Scalar, DeviceType>(  "fineValues", numCells, numPoints, spaceDim);
    }
    else
    {
      coarsePointEvaluations = Kokkos::DynRankView<Scalar, DeviceType>("coarsePointEvaluations ", numCells, numPoints);
      finePointEvaluations   = Kokkos::DynRankView<Scalar, DeviceType>("finePointEvaluations   ", numCells, numPoints);
    }
    
    Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> pointRangePolicy({0,0},{numCells,numPoints});
    if constexpr (!vectorValued)
    {
      Kokkos::parallel_for(pointRangePolicy,
                           KOKKOS_LAMBDA(const ordinal_type &cellOrdinal, const ordinal_type &pointOrdinal)
                           {
        Scalar coarseValue = 0.0;
        for (ordinal_type i=0; i<coarseCardinality; i++)
        {
          coarseValue += coarseCoefficients(cellOrdinal,i) * coarseValues(i,pointOrdinal);
        }
        coarsePointEvaluations(cellOrdinal,pointOrdinal) = coarseValue;
        
        Scalar fineValue = 0.0;
        for (ordinal_type i=0; i<fineCardinality; i++)
        {
          fineValue += fineCoefficients(cellOrdinal,i) * fineValues(i,pointOrdinal);
        }
        finePointEvaluations(cellOrdinal,pointOrdinal) = fineValue;
      });
      
      testFloatingEquality2(coarsePointEvaluations, finePointEvaluations, relTol, absTol, out, success);
    }
    else
    {
      Kokkos::parallel_for(pointRangePolicy,
                           KOKKOS_LAMBDA(const ordinal_type &cellOrdinal, const ordinal_type &pointOrdinal)
                           {
        for (int d=0; d<spaceDim; d++)
        {
          Scalar coarseValue = 0.0;
          for (ordinal_type i=0; i<coarseCardinality; i++)
          {
            coarseValue += coarseCoefficients(cellOrdinal,i) * coarseValues(i,pointOrdinal,d);
          }
          coarsePointEvaluations(cellOrdinal,pointOrdinal,d) = coarseValue;
          
          Scalar fineValue = 0.0;
          for (ordinal_type i=0; i<fineCardinality; i++)
          {
            fineValue += fineCoefficients(cellOrdinal,i) * fineValues(i,pointOrdinal,d);
          }
          finePointEvaluations(cellOrdinal,pointOrdinal,d) = fineValue;
        }
      });
      
      testFloatingEquality3(coarsePointEvaluations, finePointEvaluations, relTol, absTol, out, success);
    }
    
    prolongator.apply(fineCoefficients, coarseCoefficients, true); // exercises (C,F) use case, with sumInto = true
    // now we expect the fineCoefficients to be double what they were before
    // we'll do the comparison as above, except now we'll just double the coarsePointEvaluations.
    if constexpr (!vectorValued)
    {
      Kokkos::parallel_for(pointRangePolicy,
                           KOKKOS_LAMBDA(const ordinal_type &cellOrdinal, const ordinal_type &pointOrdinal)
                           {
        coarsePointEvaluations(cellOrdinal,pointOrdinal) *= 2;
        
        Scalar fineValue = 0.0;
        for (ordinal_type i=0; i<fineCardinality; i++)
        {
          fineValue += fineCoefficients(cellOrdinal,i) * fineValues(i,pointOrdinal);
        }
        finePointEvaluations(cellOrdinal,pointOrdinal) = fineValue;
      });
      
      testFloatingEquality2(coarsePointEvaluations, finePointEvaluations, relTol, absTol, out, success);
    }
    else
    {
      Kokkos::parallel_for(pointRangePolicy,
                           KOKKOS_LAMBDA(const ordinal_type &cellOrdinal, const ordinal_type &pointOrdinal)
                           {
        for (int d=0; d<spaceDim; d++)
        {
          coarsePointEvaluations(cellOrdinal,pointOrdinal,d) *= 2.;
          
          Scalar fineValue = 0.0;
          for (ordinal_type i=0; i<fineCardinality; i++)
          {
            fineValue += fineCoefficients(cellOrdinal,i) * fineValues(i,pointOrdinal,d);
          }
          finePointEvaluations(cellOrdinal,pointOrdinal,d) = fineValue;
        }
      });
      
      testFloatingEquality3(coarsePointEvaluations, finePointEvaluations, relTol, absTol, out, success);
    }
  }
  else if constexpr (rankToTest == 1)
  {
    // (F) use case
    for (int coarseDofOrdinal=0; coarseDofOrdinal<coarseCardinality; coarseDofOrdinal++)
    {
      // set up one cell per coarse dof, each with weight 1.0
      BasisCoeffsViewType coarseCoefficients("coarseCoefficients", coarseCardinality);
      BasisCoeffsViewType   fineCoefficients(  "fineCoefficients",   fineCardinality);
      
      // to check that fineCoefficients gets cleared appropriately by prolongator, fill it initially with 1's.
      Kokkos::deep_copy(fineCoefficients, 1.0);
      
      // coarseCoefficients will be 1 where coarseDofOrdinal == dofOrdinal, 0 elsewhere
      Kokkos::deep_copy(coarseCoefficients, 0.0);
      
      auto coarseDof = Kokkos::subview(coarseCoefficients, coarseDofOrdinal);
      Kokkos::deep_copy(coarseDof, 1.0);
      
      prolongator.apply(fineCoefficients, coarseCoefficients); // exercises (F) use case, with sumInto = false
      
      Kokkos::DynRankView<Scalar, DeviceType> coarsePointEvaluations, finePointEvaluations;
      
      if (vectorValued)
      {
        coarsePointEvaluations = Kokkos::DynRankView<Scalar, DeviceType>("coarseValues", numPoints, spaceDim);
        finePointEvaluations   = Kokkos::DynRankView<Scalar, DeviceType>(  "fineValues", numPoints, spaceDim);
      }
      else
      {
        coarsePointEvaluations = Kokkos::DynRankView<Scalar, DeviceType>("coarsePointEvaluations ", numPoints);
        finePointEvaluations   = Kokkos::DynRankView<Scalar, DeviceType>("finePointEvaluations   ", numPoints);
      }
      
      if constexpr (!vectorValued)
      {
        Kokkos::RangePolicy<ExecutionSpace> pointRangePolicy(0,numPoints);
        Kokkos::parallel_for(pointRangePolicy,
                             KOKKOS_LAMBDA(const ordinal_type &pointOrdinal)
                             {
          Scalar coarseValue = 0.0;
          for (ordinal_type i=0; i<coarseCardinality; i++)
          {
            coarseValue += coarseCoefficients(i) * coarseValues(i,pointOrdinal);
          }
          coarsePointEvaluations(pointOrdinal) = coarseValue;
          
          Scalar fineValue = 0.0;
          for (ordinal_type i=0; i<fineCardinality; i++)
          {
            fineValue += fineCoefficients(i) * fineValues(i,pointOrdinal);
          }
          finePointEvaluations(pointOrdinal) = fineValue;
        });
        
        testFloatingEquality1(coarsePointEvaluations, finePointEvaluations, relTol, absTol, out, success);
      }
      else
      {
        Kokkos::RangePolicy<ExecutionSpace> pointRangePolicy(0,numPoints);
        Kokkos::parallel_for(pointRangePolicy,
                             KOKKOS_LAMBDA(const ordinal_type &pointOrdinal)
                             {
          for (int d=0; d<spaceDim; d++)
          {
            Scalar coarseValue = 0.0;
            for (ordinal_type i=0; i<coarseCardinality; i++)
            {
              coarseValue += coarseCoefficients(i) * coarseValues(i,pointOrdinal,d);
            }
            coarsePointEvaluations(pointOrdinal,d) = coarseValue;
            
            Scalar fineValue = 0.0;
            for (ordinal_type i=0; i<fineCardinality; i++)
            {
              fineValue += fineCoefficients(i) * fineValues(i,pointOrdinal,d);
            }
            finePointEvaluations(pointOrdinal,d) = fineValue;
          }
        });
        
        testFloatingEquality2(coarsePointEvaluations, finePointEvaluations, relTol, absTol, out, success);
      }
      
      prolongator.apply(fineCoefficients, coarseCoefficients, true); // exercises (F) use case, with sumInto = true
      // now we expect the fineCoefficients to be double what they were before
      // we'll do the comparison as above, except now we'll just double the coarsePointEvaluations.
      if constexpr (!vectorValued)
      {
        Kokkos::RangePolicy<ExecutionSpace> pointRangePolicy(0,numPoints);
        Kokkos::parallel_for(pointRangePolicy,
                             KOKKOS_LAMBDA(const ordinal_type &pointOrdinal)
                             {
          coarsePointEvaluations(pointOrdinal) *= 2.;
          
          Scalar fineValue = 0.0;
          for (ordinal_type i=0; i<fineCardinality; i++)
          {
            fineValue += fineCoefficients(i) * fineValues(i,pointOrdinal);
          }
          finePointEvaluations(pointOrdinal) = fineValue;
        });
        
        testFloatingEquality1(coarsePointEvaluations, finePointEvaluations, relTol, absTol, out, success);
      }
      else
      {
        Kokkos::RangePolicy<ExecutionSpace> pointRangePolicy(0,numPoints);
        Kokkos::parallel_for(pointRangePolicy,
                             KOKKOS_LAMBDA(const ordinal_type &pointOrdinal)
                             {
          for (int d=0; d<spaceDim; d++)
          {
            coarsePointEvaluations(pointOrdinal,d) *= 2.;
            
            Scalar fineValue = 0.0;
            for (ordinal_type i=0; i<fineCardinality; i++)
            {
              fineValue += fineCoefficients(i) * fineValues(i,pointOrdinal,d);
            }
            finePointEvaluations(pointOrdinal,d) = fineValue;
          }
        });
        
        testFloatingEquality2(coarsePointEvaluations, finePointEvaluations, relTol, absTol, out, success);
      }
    }
  }
  else if constexpr (rankToTest == 3)
  {
    // (C0,C1,F) use case
    // set up one cell per coarse dof, each with weight 1.0
    BasisCoeffsViewType coarseCoefficients("coarseCoefficients", 1, numCells, coarseCardinality);
    BasisCoeffsViewType   fineCoefficients(  "fineCoefficients", 1, numCells,   fineCardinality);
    
    // to check that fineCoefficients gets cleared appropriately by prolongator, fill it initially with 1's.
    Kokkos::deep_copy(fineCoefficients, 1.0);
    
    // coarseCoefficients will be 1 where cellOrdinal == dofOrdinal, 0 elsewhere
    Kokkos::deep_copy(coarseCoefficients, 0.0);
    
    Kokkos::RangePolicy<ExecutionSpace> rangePolicy(0,coarseCardinality);
    Kokkos::parallel_for(rangePolicy,
                         KOKKOS_LAMBDA(const ordinal_type &cellOrdinal)
                         {
      const ordinal_type &dofOrdinal = cellOrdinal;
      coarseCoefficients(0,cellOrdinal,dofOrdinal) = 1.0;
    });
    
    prolongator.apply(fineCoefficients, coarseCoefficients); // exercises (C,F) use case, with sumInto = false
    
    Kokkos::DynRankView<Scalar, DeviceType> coarsePointEvaluations, finePointEvaluations;
    
    if (vectorValued)
    {
      coarsePointEvaluations = Kokkos::DynRankView<Scalar, DeviceType>("coarseValues", 1, numCells, numPoints, spaceDim);
      finePointEvaluations   = Kokkos::DynRankView<Scalar, DeviceType>(  "fineValues", 1, numCells, numPoints, spaceDim);
    }
    else
    {
      coarsePointEvaluations = Kokkos::DynRankView<Scalar, DeviceType>("coarsePointEvaluations ", 1, numCells, numPoints);
      finePointEvaluations   = Kokkos::DynRankView<Scalar, DeviceType>("finePointEvaluations   ", 1, numCells, numPoints);
    }
    
    Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> pointRangePolicy({0,0},{numCells,numPoints});
    if constexpr (!vectorValued)
    {
      Kokkos::parallel_for(pointRangePolicy,
                           KOKKOS_LAMBDA(const ordinal_type &cellOrdinal, const ordinal_type &pointOrdinal)
                           {
        Scalar coarseValue = 0.0;
        for (ordinal_type i=0; i<coarseCardinality; i++)
        {
          coarseValue += coarseCoefficients(0, cellOrdinal,i) * coarseValues(i,pointOrdinal);
        }
        coarsePointEvaluations(0, cellOrdinal,pointOrdinal) = coarseValue;
        
        Scalar fineValue = 0.0;
        for (ordinal_type i=0; i<fineCardinality; i++)
        {
          fineValue += fineCoefficients(0, cellOrdinal,i) * fineValues(i,pointOrdinal);
        }
        finePointEvaluations(0, cellOrdinal,pointOrdinal) = fineValue;
      });
      
      testFloatingEquality3(coarsePointEvaluations, finePointEvaluations, relTol, absTol, out, success);
    }
    else
    {
      Kokkos::parallel_for(pointRangePolicy,
                           KOKKOS_LAMBDA(const ordinal_type &cellOrdinal, const ordinal_type &pointOrdinal)
                           {
        for (int d=0; d<spaceDim; d++)
        {
          Scalar coarseValue = 0.0;
          for (ordinal_type i=0; i<coarseCardinality; i++)
          {
            coarseValue += coarseCoefficients(0,cellOrdinal,i) * coarseValues(i,pointOrdinal,d);
          }
          coarsePointEvaluations(0,cellOrdinal,pointOrdinal,d) = coarseValue;
          
          Scalar fineValue = 0.0;
          for (ordinal_type i=0; i<fineCardinality; i++)
          {
            fineValue += fineCoefficients(0,cellOrdinal,i) * fineValues(i,pointOrdinal,d);
          }
          finePointEvaluations(0,cellOrdinal,pointOrdinal,d) = fineValue;
        }
      });
      
      testFloatingEquality4(coarsePointEvaluations, finePointEvaluations, relTol, absTol, out, success);
    }
    
    prolongator.apply(fineCoefficients, coarseCoefficients, true); // exercises (C,F) use case, with sumInto = true
    // now we expect the fineCoefficients to be double what they were before
    // we'll do the comparison as above, except now we'll just double the coarsePointEvaluations.
    if constexpr (!vectorValued)
    {
      Kokkos::parallel_for(pointRangePolicy,
      KOKKOS_LAMBDA(const ordinal_type &cellOrdinal, const ordinal_type &pointOrdinal)
      {
        coarsePointEvaluations(0,cellOrdinal,pointOrdinal) *= 2;
        
        Scalar fineValue = 0.0;
        for (ordinal_type i=0; i<fineCardinality; i++)
        {
          fineValue += fineCoefficients(0,cellOrdinal,i) * fineValues(i,pointOrdinal);
        }
        finePointEvaluations(0,cellOrdinal,pointOrdinal) = fineValue;
      });
      
      testFloatingEquality3(coarsePointEvaluations, finePointEvaluations, relTol, absTol, out, success);
    }
    else
    {
      Kokkos::parallel_for(pointRangePolicy,
      KOKKOS_LAMBDA(const ordinal_type &cellOrdinal, const ordinal_type &pointOrdinal)
      {
        for (int d=0; d<spaceDim; d++)
        {
          coarsePointEvaluations(0,cellOrdinal,pointOrdinal,d) *= 2.;
          
          Scalar fineValue = 0.0;
          for (ordinal_type i=0; i<fineCardinality; i++)
          {
            fineValue += fineCoefficients(0,cellOrdinal,i) * fineValues(i,pointOrdinal,d);
          }
          finePointEvaluations(0,cellOrdinal,pointOrdinal,d) = fineValue;
        }
      });
      
      testFloatingEquality4(coarsePointEvaluations, finePointEvaluations, relTol, absTol, out, success);
    }
  }
}

template<class BasisCoeffsViewType, int rankToTest>
void testProlongationTranspose(shards::CellTopology &cellTopo, const EFunctionSpace &fs,
                               const ordinal_type &pCoarse, const ordinal_type &pFine,
                               Teuchos::FancyOStream &out, bool &success)
{
  using DeviceType = typename BasisCoeffsViewType::device_type;
  using Scalar     = typename BasisCoeffsViewType::value_type;
  using BasisFamily = HierarchicalBasisFamily<DeviceType>;
  
  const double relTol=1e-13;
  const double absTol=1e-13;
  
  auto coarseBasis = getBasis< BasisFamily >(cellTopo, fs, pCoarse);
  auto fineBasis   = getBasis< BasisFamily >(cellTopo, fs, pFine);
  
  auto prolongator = HierarchicalBasisProlongator<DeviceType,Scalar>(coarseBasis, fineBasis);
  
  auto testPoints = getInputPointsView<Scalar, DeviceType>(cellTopo, pFine + 1);
  
  const ordinal_type coarseCardinality = coarseBasis->getCardinality();
  const ordinal_type   fineCardinality =   fineBasis->getCardinality();
  const ordinal_type numCells = coarseCardinality;
  
  using ExecutionSpace = typename DeviceType::execution_space;
  
  const ordinal_type numPoints = testPoints.extent_int(0);
  const ordinal_type spaceDim  = cellTopo.getDimension();
  
  if constexpr (rankToTest == 2)
  {
    // (C,F) use case
    // set up one cell per coarse dof, each with weight 1.0
    BasisCoeffsViewType coarseCoefficients("coarseCoefficients", numCells, coarseCardinality);
    BasisCoeffsViewType   fineCoefficients(  "fineCoefficients", numCells,   fineCardinality);
    
    // to check that coarseCoefficients gets cleared appropriately by prolongator, fill it initially with 1's.
    Kokkos::deep_copy(coarseCoefficients, 1.0);
    
    // coarseCoefficientsExpected will be 1 where cellOrdinal == dofOrdinal, 0 elsewhere
    BasisCoeffsViewType coarseCoefficientsExpected("coarseCoefficientsExpected", numCells, coarseCardinality);
    Kokkos::deep_copy(coarseCoefficientsExpected, 0.0);
    
    Kokkos::RangePolicy<ExecutionSpace> rangePolicy(0,coarseCardinality);
    Kokkos::parallel_for(rangePolicy,
    KOKKOS_LAMBDA(const ordinal_type &cellOrdinal)
    {
      const ordinal_type &dofOrdinal = cellOrdinal;
      coarseCoefficientsExpected(cellOrdinal,dofOrdinal) = 1.0;
    });
    prolongator.apply(fineCoefficients, coarseCoefficientsExpected);
    prolongator.applyTranspose(coarseCoefficients, fineCoefficients); // exercises (C,F) use case, with sumInto = false
    
    testFloatingEquality2(coarseCoefficientsExpected, coarseCoefficients, relTol, absTol, out, success);
    
    prolongator.applyTranspose(coarseCoefficients, fineCoefficients, true); // exercises (C,F) use case, with sumInto = true
    
    Kokkos::parallel_for(rangePolicy,
    KOKKOS_LAMBDA(const ordinal_type &cellOrdinal)
    {
      const ordinal_type &dofOrdinal = cellOrdinal;
      coarseCoefficientsExpected(cellOrdinal,dofOrdinal) = 2.0;
    });
    
    testFloatingEquality2(coarseCoefficientsExpected, coarseCoefficients, relTol, absTol, out, success);
  }
  else if constexpr (rankToTest == 1)
  {
    // (F) use case
    for (int coarseDofOrdinal=0; coarseDofOrdinal<coarseCardinality; coarseDofOrdinal++)
    {
      // set up one cell per coarse dof, each with weight 1.0
      BasisCoeffsViewType coarseCoefficients("coarseCoefficients", coarseCardinality);
      BasisCoeffsViewType   fineCoefficients(  "fineCoefficients",   fineCardinality);
      
      // to check that coarseCoefficients gets cleared appropriately by prolongator, fill it initially with 1's.
      Kokkos::deep_copy(coarseCoefficients, 1.0);
      
      // coarseCoefficientsExpected will be 1 where coarseDofOrdinal == dofOrdinal, 0 elsewhere
      BasisCoeffsViewType coarseCoefficientsExpected("coarseCoefficientsExpected", coarseCardinality);
      Kokkos::deep_copy(coarseCoefficientsExpected, 0.0);
      
      auto coarseDof = Kokkos::subview(coarseCoefficientsExpected, coarseDofOrdinal);
      Kokkos::deep_copy(coarseDof, 1.0);
      
      prolongator.apply(fineCoefficients, coarseCoefficientsExpected);
      prolongator.applyTranspose(coarseCoefficients, fineCoefficients);
      
      testFloatingEquality1(coarseCoefficientsExpected, coarseCoefficients, relTol, absTol, out, success);
      
      prolongator.applyTranspose(coarseCoefficients, fineCoefficients, true); // exercises (F) use case, with sumInto = true
      // now we expect the coarseCoefficients to be double what they were before
      // we'll do the comparison as above, except now we'll just double the finePointEvaluations.
      Kokkos::deep_copy(coarseDof, 2.0); // copies into coarseCoefficientsExpected
      testFloatingEquality1(coarseCoefficientsExpected, coarseCoefficients, relTol, absTol, out, success);
    }
  }
  else if constexpr (rankToTest == 3)
  {
    // (C0,C1,F) use case
    // set up one cell per coarse dof, each with weight 1.0
    BasisCoeffsViewType coarseCoefficientsExpected("coarseCoefficientsExpected", 1, numCells, coarseCardinality);
    BasisCoeffsViewType coarseCoefficients("coarseCoefficients", 1, numCells, coarseCardinality);
    BasisCoeffsViewType   fineCoefficients(  "fineCoefficients", 1, numCells,   fineCardinality);
    
    // to check that fineCoefficients gets cleared appropriately by prolongator, fill it initially with 1's.
    Kokkos::deep_copy(fineCoefficients, 1.0);
    
    // coarseCoefficients will be 1 where cellOrdinal == dofOrdinal, 0 elsewhere
    Kokkos::deep_copy(coarseCoefficientsExpected, 0.0);
    
    Kokkos::RangePolicy<ExecutionSpace> rangePolicy(0,coarseCardinality);
    Kokkos::parallel_for(rangePolicy,
    KOKKOS_LAMBDA(const ordinal_type &cellOrdinal)
    {
      const ordinal_type &dofOrdinal = cellOrdinal;
      coarseCoefficientsExpected(0,cellOrdinal,dofOrdinal) = 1.0;
    });
    
    prolongator.apply(fineCoefficients, coarseCoefficientsExpected);
    prolongator.applyTranspose(coarseCoefficients, fineCoefficients); // exercises (C0,C1,F) use case, with sumInto = false
    
    testFloatingEquality3(coarseCoefficientsExpected, coarseCoefficients, relTol, absTol, out, success);
    
    prolongator.applyTranspose(coarseCoefficients, fineCoefficients, true); // exercises (C0,C1,F) use case, with sumInto = true
    // now we expect the coarseCoefficients to be double what they were before
    
    Kokkos::parallel_for(rangePolicy,
    KOKKOS_LAMBDA(const ordinal_type &cellOrdinal)
    {
      const ordinal_type &dofOrdinal = cellOrdinal;
      coarseCoefficientsExpected(0,cellOrdinal,dofOrdinal) = 2.0;
    });
    
    testFloatingEquality3(coarseCoefficientsExpected, coarseCoefficients, relTol, absTol, out, success);
  }
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(HierarchicalBasisProlongator, Prolongation, CellTopoTag, FSTag, PCoarseTag, PFineTag)
{
  using DeviceType  = DefaultTestDeviceType;
  
  using Scalar = double;
  
  const EFunctionSpace    fs = FSTag::functionSpace;
  const ordinal_type pCoarse = PCoarseTag::polyOrder;
  const ordinal_type pFine   = PFineTag::polyOrder;
  
  auto cellTopo = CellTopoTag::cellTopoPtr->getBaseTopology();
  
  using ViewType1 = Kokkos::View<Scalar*,DeviceType>;
  using ViewType2 = Kokkos::View<Scalar**,DeviceType>;
  using ViewType3 = Kokkos::View<Scalar***,DeviceType>;
  
  constexpr bool vectorValued = (fs == Intrepid2::FUNCTION_SPACE_HCURL) || (fs == Intrepid2::FUNCTION_SPACE_HDIV);
  
  testProlongation<ViewType1, 1, vectorValued>(cellTopo, fs, pCoarse, pFine, out, success);
  testProlongation<ViewType2, 2, vectorValued>(cellTopo, fs, pCoarse, pFine, out, success);
  testProlongation<ViewType3, 3, vectorValued>(cellTopo, fs, pCoarse, pFine, out, success);
  
  using DynRankView = Kokkos::DynRankView<Scalar, DeviceType>;
  testProlongation<DynRankView, 1, vectorValued>(cellTopo, fs, pCoarse, pFine, out, success);
  testProlongation<DynRankView, 2, vectorValued>(cellTopo, fs, pCoarse, pFine, out, success);
  testProlongation<DynRankView, 3, vectorValued>(cellTopo, fs, pCoarse, pFine, out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(HierarchicalBasisProlongator, Restriction, CellTopoTag, FSTag, PCoarseTag, PFineTag)
{
  using DeviceType  = DefaultTestDeviceType;
  
  using Scalar = double;
  
  const EFunctionSpace    fs = FSTag::functionSpace;
  const ordinal_type pCoarse = PCoarseTag::polyOrder;
  const ordinal_type pFine   = PFineTag::polyOrder;
  
  auto cellTopo = CellTopoTag::cellTopoPtr->getBaseTopology();
  
  using ViewType1 = Kokkos::View<Scalar*,DeviceType>;
  using ViewType2 = Kokkos::View<Scalar**,DeviceType>;
  using ViewType3 = Kokkos::View<Scalar***,DeviceType>;
    
  testProlongationTranspose<ViewType1, 1>(cellTopo, fs, pCoarse, pFine, out, success);
  testProlongationTranspose<ViewType2, 2>(cellTopo, fs, pCoarse, pFine, out, success);
  testProlongationTranspose<ViewType3, 3>(cellTopo, fs, pCoarse, pFine, out, success);
  
  using DynRankView = Kokkos::DynRankView<Scalar, DeviceType>;
  testProlongationTranspose<DynRankView, 1>(cellTopo, fs, pCoarse, pFine, out, success);
  testProlongationTranspose<DynRankView, 2>(cellTopo, fs, pCoarse, pFine, out, success);
  testProlongationTranspose<DynRankView, 3>(cellTopo, fs, pCoarse, pFine, out, success);
}

TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P1, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P1, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P1, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P2, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P2, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P2, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P3, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P3, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P3, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P4, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P4, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P4, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P5, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P5, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P5, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P6, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P6, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P6, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P7, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P7, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P7, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P8, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P8, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HGRAD, P9, P9)

TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P1, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P1, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P1, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P2, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P2, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P2, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P3, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P3, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P3, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P4, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P4, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P4, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P5, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P5, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P5, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P6, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P6, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P6, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P7, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P7, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P7, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P8, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P8, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, LINE, HVOL, P9, P9)

TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HGRAD, P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HCURL, P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HDIV,  P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, QUAD, HVOL,  P6, P6)

TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HGRAD, P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HCURL, P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HDIV,  P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, TRI, HVOL,  P6, P6)

TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HGRAD, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HGRAD, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HGRAD, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HGRAD, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HCURL, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HCURL, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HCURL, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HCURL, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HDIV,  P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HDIV,  P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HDIV,  P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HDIV,  P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HVOL,  P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HVOL,  P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HVOL,  P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Prolongation, HEX, HVOL,  P1, P4)

TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P1, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P1, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P1, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P2, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P2, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P2, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P3, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P3, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P3, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P4, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P4, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P4, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P5, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P5, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P5, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P6, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P6, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P6, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P7, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P7, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P7, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P8, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P8, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HGRAD, P9, P9)

TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P1, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P1, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P1, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P2, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P2, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P2, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P3, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P3, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P3, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P4, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P4, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P4, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P5, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P5, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P5, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P6, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P6, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P6, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P7, P7)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P7, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P7, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P8, P8)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P8, P9)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, LINE, HVOL, P9, P9)

TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HGRAD, P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HCURL, P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HDIV,  P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, QUAD, HVOL,  P6, P6)

TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HGRAD, P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HCURL, P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HDIV,  P6, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P1, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P1, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P2, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P2, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P2, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P2, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P2, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P3, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P3, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P3, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P3, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P4, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P4, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P4, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P5, P5)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P5, P6)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, TRI, HVOL,  P6, P6)

TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HGRAD, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HGRAD, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HGRAD, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HGRAD, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HCURL, P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HCURL, P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HCURL, P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HCURL, P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HDIV,  P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HDIV,  P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HDIV,  P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HDIV,  P1, P4)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HVOL,  P1, P1)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HVOL,  P1, P2)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HVOL,  P1, P3)
TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(HierarchicalBasisProlongator, Restriction, HEX, HVOL,  P1, P4)

} // namespace

