// @HEADER
// *****************************************************************************
//                           Intrepid2 Package
//
// Copyright 2007 NTESS and the Intrepid2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

/** \file   Intrepid2_HierarchicalBasisProlongator.hpp
    \brief  Header file for the Intrepid2 HierarchicalBasisProlongator class.
    \author Created by N. V. Roberts.
 */

#ifndef __INTREPID2_BASIS_PROLONGATOR_HPP__
#define __INTREPID2_BASIS_PROLONGATOR_HPP__

#include "Intrepid2_ConfigDefs.hpp"
#include "Intrepid2_Types.hpp"
#include "Intrepid2_Utils.hpp"

#include "Intrepid2_Basis.hpp"

namespace Intrepid2 {

template<typename DeviceType,
         typename OutputScalar = double,
         typename PointScalar  = double>
class HierarchicalBasisProlongator {
  Kokkos::View<ordinal_type*,DeviceType> _fineDofOrdinals; // index: coarse dof ordinal; value: fine dof ordinal
  ordinal_type _fineBasisCardinality;
public:
  using BPtr = BasisPtr<DeviceType,OutputScalar,PointScalar>;
  
  HierarchicalBasisProlongator(BPtr coarseBasis, BPtr fineBasis)
  {
    INTREPID2_TEST_FOR_EXCEPTION(coarseBasis->getBasisType() != BASIS_FEM_HIERARCHICAL, std::invalid_argument,
                                 "coarseBasis and fineBasis must have basis type BASIS_FEM_HIERARCHICAL");
    
    INTREPID2_TEST_FOR_EXCEPTION(fineBasis->getBasisType() != BASIS_FEM_HIERARCHICAL, std::invalid_argument,
                                 "coarseBasis and fineBasis must have basis type BASIS_FEM_HIERARCHICAL");
    
    INTREPID2_TEST_FOR_EXCEPTION(coarseBasis->getFunctionSpace() != fineBasis->getFunctionSpace(), std::invalid_argument,
                                 "coarseBasis and fineBasis must agree on function space");
    
    const ordinal_type coarseCardinality = coarseBasis->getCardinality();
    _fineBasisCardinality = fineBasis->getCardinality();
    
    const auto fs = fineBasis->getFunctionSpace();
    
    auto vectorsMatch = [](std::vector<int> lhs,std::vector<int> rhs)
    {
      if (lhs.size() != rhs.size()) return false;
      for (unsigned i=0; i<lhs.size(); i++)
      {
        if (lhs[i] != rhs[i]) return false;
      }
      return true;
    };
    
    ordinal_type degreeLength = coarseBasis->getPolynomialDegreeLength();
    std::vector<ordinal_type> coarseDegree(degreeLength,0);
    for (ordinal_type i=0; i<coarseCardinality; i++)
    {
      auto polyDegree = coarseBasis->getPolynomialDegreeOfField(i);
      
      for (int d=0; d<degreeLength; d++)
      {
        coarseDegree[d] = std::max(coarseDegree[d], polyDegree[d]);
      }
    }
    
    auto allLowerOrderFieldOrdinals = fineBasis->getFieldOrdinalsForDegree(coarseDegree);
    // for H(curl) and H(div), allLowerOrderFieldOrdinals may have some extra entries beyond what the subBasis has
    // (this has to do with the fact that the different families have different degrees in each dimension)
    // we filter according to polynomial degree of the fields, and place the result in lowerOrderFieldOrdinals
    std::vector<int> lowerOrderFieldOrdinals;
    if ((fs == Intrepid2::FUNCTION_SPACE_HCURL) || (fs == Intrepid2::FUNCTION_SPACE_HDIV))
    {
      unsigned allFieldOrdinalIndex = 0;
      for (int subBasisFieldOrdinal=0; subBasisFieldOrdinal < coarseBasis->getCardinality(); subBasisFieldOrdinal++)
      {
        while (   (allFieldOrdinalIndex<allLowerOrderFieldOrdinals.size())
               && (
                   !vectorsMatch(
                                 fineBasis  ->getPolynomialDegreeOfFieldAsVector(allLowerOrderFieldOrdinals[allFieldOrdinalIndex]),
                                 coarseBasis->getPolynomialDegreeOfFieldAsVector(subBasisFieldOrdinal) // same poly order in each dimension
                                 )
                   )
               )
        {
          allFieldOrdinalIndex++;
        }
        if (allFieldOrdinalIndex < allLowerOrderFieldOrdinals.size())
        {
          // then the subcell dim and subcell ordinal must match -- add this into our filtered list
          lowerOrderFieldOrdinals.push_back(allLowerOrderFieldOrdinals[allFieldOrdinalIndex]);
        }
        // if all has gone well, we've established a correspondence between subBasisFieldOrdinal and allLowerOrderFieldOrdinals[allFieldOrdinalIndex]
        // we therefore should increment allFieldOrdinalIndex
        allFieldOrdinalIndex++;
      }
    }
    else
    {
      lowerOrderFieldOrdinals = allLowerOrderFieldOrdinals;
    }
    if (lowerOrderFieldOrdinals.size() != unsigned(coarseCardinality))
    {
      INTREPID2_TEST_FOR_EXCEPTION(lowerOrderFieldOrdinals.size() != unsigned(coarseBasis->getCardinality()),
                                   std::invalid_argument, "Error: could not establish mapping between coarse basis and fine basis.");
    }
    
    _fineDofOrdinals = Kokkos::View<ordinal_type*,DeviceType>("HierarchicalBasisProlongator fineDofOrdinals",lowerOrderFieldOrdinals.size());
    auto fineDofOrdinalsHost = Kokkos::create_mirror_view(_fineDofOrdinals);
    
    for (ordinal_type i=0; i<coarseCardinality; i++)
    {
      fineDofOrdinalsHost(i) = lowerOrderFieldOrdinals[i];
    }
  }
  
  template<class OutViewType, class InViewType>
  void apply(OutViewType &fineBasisView, const InViewType &coarseBasisView, const bool sumInto = false)
  {
    auto fineDofOrdinals = _fineDofOrdinals; // avoid implicit reference to this in Kokkos::parallel_for's below.
    
    INTREPID2_TEST_FOR_EXCEPTION(fineBasisView.rank() != coarseBasisView.rank(), std::invalid_argument,
                                 "coarseBasisView and fineBasisView must agree on rank");
    
    const ordinal_type finalDim = fineBasisView.rank() - 1;
    
    INTREPID2_TEST_FOR_EXCEPTION(fineBasisView.extent_int(finalDim) != _fineBasisCardinality, std::invalid_argument,
                                 "Final dimension of fineBasisView must have extent equal to fine basis cardinality");
    
    const ordinal_type coarseBasisCardinality = _fineDofOrdinals.extent_int(0); // _fineDofOrdinals has one entry per coarse dof
    INTREPID2_TEST_FOR_EXCEPTION(coarseBasisView.extent_int(finalDim) != _fineDofOrdinals.extent_int(0), std::invalid_argument,
                                 "Final dimension of coarseBasisView must have extent equal to coarse basis cardinality");
    
    using ExecutionSpace = typename DeviceType::execution_space;
    
    // three use cases: (F) to (F); (C,F) to (C,F); (C0,C1,F) to (C0,C1,F)
    if (fineBasisView.rank() == 1)
    {
      // guard with constexpr for fixed-rank view cases, for which the lambdas below otherwise would not compile
      constexpr bool rank1 = supports_rank_1<OutViewType>::value;
      if constexpr (rank1)
      {
        Kokkos::RangePolicy<ExecutionSpace> policy(0,coarseBasisCardinality);
        
        if (sumInto)
        {
          Kokkos::parallel_for("apply HierarchicalBasisProlongator (F) (sumInto)", policy,
          KOKKOS_LAMBDA(const int &coarseOrdinal)
          {
            const ordinal_type & fineOrdinal = fineDofOrdinals(coarseOrdinal);
            fineBasisView(fineOrdinal) += coarseBasisView(coarseOrdinal);
          });
        }
        else
        {
          Kokkos::deep_copy(fineBasisView, 0.0);
          Kokkos::parallel_for("apply HierarchicalBasisProlongator (F)", policy,
          KOKKOS_LAMBDA(const int &coarseOrdinal)
          {
            const ordinal_type & fineOrdinal = fineDofOrdinals(coarseOrdinal);
            fineBasisView(fineOrdinal) = coarseBasisView(coarseOrdinal);
          });
        }
      }
    }
    else if (fineBasisView.rank() == 2)
    {
      // guard with constexpr for fixed-rank view cases, for which the lambdas below otherwise would not compile
      constexpr bool rank2 = supports_rank_2<OutViewType>::value;
      if constexpr (rank2)
      {
        INTREPID2_TEST_FOR_EXCEPTION(fineBasisView.extent_int(0) != coarseBasisView.extent_int(0), std::invalid_argument,
                                     "fineBasisView and coarseBasisView must agree in the cell dimension(s)");
        const ordinal_type numCells = fineBasisView.extent_int(0);
        
        Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> policy({0,0},{numCells,coarseBasisCardinality});
        
        if (sumInto)
        {
          Kokkos::parallel_for("apply HierarchicalBasisProlongator (C,F)", policy,
          KOKKOS_LAMBDA(const int &cellOrdinal, const int &coarseOrdinal)
          {
            const ordinal_type & fineOrdinal = fineDofOrdinals(coarseOrdinal);
            fineBasisView(cellOrdinal,fineOrdinal) += coarseBasisView(cellOrdinal,coarseOrdinal);
          });
        }
        else
        {
          Kokkos::deep_copy(fineBasisView, 0.0);
          Kokkos::parallel_for("apply HierarchicalBasisProlongator (C,F)", policy,
          KOKKOS_LAMBDA(const int &cellOrdinal, const int &coarseOrdinal)
          {
            const ordinal_type & fineOrdinal = fineDofOrdinals(coarseOrdinal);
            fineBasisView(cellOrdinal,fineOrdinal) = coarseBasisView(cellOrdinal,coarseOrdinal);
          });
        }
      }
    }
    else if (fineBasisView.rank() == 3)
    {
      // guard with constexpr for fixed-rank view cases, for which the lambdas below otherwise would not compile
      constexpr bool rank3 = supports_rank_3<OutViewType>::value;
      if constexpr (rank3)
      {
        INTREPID2_TEST_FOR_EXCEPTION(fineBasisView.extent_int(0) != coarseBasisView.extent_int(0), std::invalid_argument,
                                     "fineBasisView and coarseBasisView must agree in the cell dimensions");
        INTREPID2_TEST_FOR_EXCEPTION(fineBasisView.extent_int(1) != coarseBasisView.extent_int(1), std::invalid_argument,
                                     "fineBasisView and coarseBasisView must agree in the cell dimensions");
        const ordinal_type C0 = fineBasisView.extent_int(0);
        const ordinal_type C1 = fineBasisView.extent_int(1);
        
        Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>> policy({0,0,0},{C0,C1,coarseBasisCardinality});
        
        if (sumInto)
        {
          Kokkos::parallel_for("apply HierarchicalBasisProlongator (C0,C1,F)", policy,
          KOKKOS_LAMBDA(const int &c0, const int &c1, const int &coarseOrdinal)
          {
            const ordinal_type & fineOrdinal = fineDofOrdinals(coarseOrdinal);
            fineBasisView(c0,c1,fineOrdinal) += coarseBasisView(c0,c1,coarseOrdinal);
          });
        }
        else
        {
          Kokkos::deep_copy(fineBasisView, 0.0);
          Kokkos::parallel_for("apply HierarchicalBasisProlongator (C0,C1,F)", policy,
          KOKKOS_LAMBDA(const int &c0, const int &c1, const int &coarseOrdinal)
          {
            const ordinal_type & fineOrdinal = fineDofOrdinals(coarseOrdinal);
            fineBasisView(c0,c1,fineOrdinal) = coarseBasisView(c0,c1,coarseOrdinal);
          });
        }
      }
    }
    else
    {
      INTREPID2_TEST_FOR_EXCEPTION((fineBasisView.rank() != 1) && (fineBasisView.rank() != 2) && (fineBasisView.rank() != 3), std::invalid_argument,
                                   "ranks of Views must both be 1, 2, or 3");
    }
  }
  
  template<class OutViewType, class InViewType>
  void applyTranspose(OutViewType &coarseBasisView, const InViewType &fineBasisView, const bool sumInto = false)
  {
    auto fineDofOrdinals = _fineDofOrdinals; // avoid implicit reference to this in Kokkos::parallel_for's below.
    
    INTREPID2_TEST_FOR_EXCEPTION(fineBasisView.rank() != coarseBasisView.rank(), std::invalid_argument,
                                 "coarseBasisView and fineBasisView must agree on rank");
    
    const ordinal_type finalDim = fineBasisView.rank() - 1;
    
    INTREPID2_TEST_FOR_EXCEPTION(fineBasisView.extent_int(finalDim) != _fineBasisCardinality, std::invalid_argument,
                                 "Final dimension of fineBasisView must have extent equal to fine basis cardinality");
    
    const ordinal_type coarseBasisCardinality = _fineDofOrdinals.extent_int(0); // _fineDofOrdinals has one entry per coarse dof
    INTREPID2_TEST_FOR_EXCEPTION(coarseBasisView.extent_int(finalDim) != _fineDofOrdinals.extent_int(0), std::invalid_argument,
                                 "Final dimension of coarseBasisView must have extent equal to coarse basis cardinality");
    
    using ExecutionSpace = typename DeviceType::execution_space;
    
    // three use cases: (F) to (F); (C,F) to (C,F); (C0,C1,F) to (C0,C1,F)
    if (fineBasisView.rank() == 1)
    {
      // guard with constexpr for fixed-rank view cases, for which the lambdas below otherwise would not compile
      constexpr bool rank1 = supports_rank_1<OutViewType>::value;
      if constexpr (rank1)
      {
        Kokkos::RangePolicy<ExecutionSpace> policy(0,coarseBasisCardinality);
        
        if (sumInto)
        {
          Kokkos::parallel_for("apply HierarchicalBasisProlongator (F) (sumInto)", policy,
          KOKKOS_LAMBDA(const int &coarseOrdinal)
          {
            const ordinal_type & fineOrdinal = fineDofOrdinals(coarseOrdinal);
            coarseBasisView(coarseOrdinal) += fineBasisView(fineOrdinal);
          });
        }
        else
        {
          Kokkos::deep_copy(coarseBasisView, 0.0);
          Kokkos::parallel_for("apply HierarchicalBasisProlongator (F)", policy,
          KOKKOS_LAMBDA(const int &coarseOrdinal)
          {
            const ordinal_type & fineOrdinal = fineDofOrdinals(coarseOrdinal);
            coarseBasisView(coarseOrdinal) = fineBasisView(fineOrdinal);
          });
        }
      }
    }
    else if (fineBasisView.rank() == 2)
    {
      // guard with constexpr for fixed-rank view cases, for which the lambdas below otherwise would not compile
      constexpr bool rank2 = supports_rank_2<OutViewType>::value;
      if constexpr (rank2)
      {
        INTREPID2_TEST_FOR_EXCEPTION(fineBasisView.extent_int(0) != coarseBasisView.extent_int(0), std::invalid_argument,
                                     "fineBasisView and coarseBasisView must agree in the cell dimension(s)");
        const ordinal_type numCells = fineBasisView.extent_int(0);
        
        Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> policy({0,0},{numCells,coarseBasisCardinality});
        
        if (sumInto)
        {
          Kokkos::parallel_for("apply HierarchicalBasisProlongator (C,F)", policy,
          KOKKOS_LAMBDA(const int &cellOrdinal, const int &coarseOrdinal)
          {
            const ordinal_type & fineOrdinal = fineDofOrdinals(coarseOrdinal);
            coarseBasisView(cellOrdinal,coarseOrdinal) += fineBasisView(cellOrdinal,fineOrdinal);
          });
        }
        else
        {
          Kokkos::deep_copy(coarseBasisView, 0.0);
          Kokkos::parallel_for("apply HierarchicalBasisProlongator (C,F)", policy,
          KOKKOS_LAMBDA(const int &cellOrdinal, const int &coarseOrdinal)
          {
            const ordinal_type & fineOrdinal = fineDofOrdinals(coarseOrdinal);
            coarseBasisView(cellOrdinal,coarseOrdinal) = fineBasisView(cellOrdinal,fineOrdinal);
          });
        }
      }
    }
    else if (fineBasisView.rank() == 3)
    {
      // guard with constexpr for fixed-rank view cases, for which the lambdas below otherwise would not compile
      constexpr bool rank3 = supports_rank_3<OutViewType>::value;
      if constexpr (rank3)
      {
        INTREPID2_TEST_FOR_EXCEPTION(fineBasisView.extent_int(0) != coarseBasisView.extent_int(0), std::invalid_argument,
                                     "fineBasisView and coarseBasisView must agree in the cell dimensions");
        INTREPID2_TEST_FOR_EXCEPTION(fineBasisView.extent_int(1) != coarseBasisView.extent_int(1), std::invalid_argument,
                                     "fineBasisView and coarseBasisView must agree in the cell dimensions");
        const ordinal_type C0 = fineBasisView.extent_int(0);
        const ordinal_type C1 = fineBasisView.extent_int(1);
        
        Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<3>> policy({0,0,0},{C0,C1,coarseBasisCardinality});
        
        if (sumInto)
        {
          Kokkos::parallel_for("apply HierarchicalBasisProlongator (C0,C1,F)", policy,
          KOKKOS_LAMBDA(const int &c0, const int &c1, const int &coarseOrdinal)
          {
            const ordinal_type & fineOrdinal = fineDofOrdinals(coarseOrdinal);
            coarseBasisView(c0,c1,coarseOrdinal) += fineBasisView(c0,c1,fineOrdinal);
          });
        }
        else
        {
          Kokkos::deep_copy(coarseBasisView, 0.0);
          Kokkos::parallel_for("apply HierarchicalBasisProlongator (C0,C1,F)", policy,
          KOKKOS_LAMBDA(const int &c0, const int &c1, const int &coarseOrdinal)
          {
            const ordinal_type & fineOrdinal = fineDofOrdinals(coarseOrdinal);
            coarseBasisView(c0,c1,coarseOrdinal) = fineBasisView(c0,c1,fineOrdinal);
          });
        }
      }
    }
    else
    {
      INTREPID2_TEST_FOR_EXCEPTION((fineBasisView.rank() != 1) && (fineBasisView.rank() != 2) && (fineBasisView.rank() != 3), std::invalid_argument,
                                   "ranks of Views must both be 1, 2, or 3");
    }
  }
};

}// namespace Intrepid2

#endif
