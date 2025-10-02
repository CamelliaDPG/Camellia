// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#ifndef TypeDefs_h
#define TypeDefs_h

#include "Kokkos_DynRankView.hpp"

// #include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_RCP.hpp>
// #include <Teuchos_GlobalMPISession.hpp>
// #include <Teuchos_oblackholestream.hpp>
// #include <Teuchos_Tuple.hpp>
// #include <Teuchos_VerboseObject.hpp>
// #include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>

// Epetra includes
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

namespace Intrepid
{
  template<class Scalar, int ArrayTypeId>
  class FieldContainer;
} // namespace Intrepid

namespace Camellia
{
  //! SolutionComponent is used to distinguish between the usual solution and the influence (goal-oriented) solution.
  enum SolutionComponent : short
  {
    STANDARD_SOLUTION,
    INFLUENCE_SOLUTION
  };
  
// Basic typedefs
typedef int IndexType;
typedef int GlobalIndexType;
typedef int PartitionIndexType; // for partition numbering
typedef int GlobalIndexTypeToCast; // for constructing Epetra_Maps, etc.  (these like either int or long long)
  
// MOAB-compatible definitions:
#define DIRICHLET_SET_TAG_NAME   "DIRICHLET_SET" // identical to MOAB's DIRICHLET_SET_TAG_NAME
#define MATERIAL_SET_TAG_NAME   "MATERIAL_SET"
#define NEUMANN_SET_TAG_NAME   "NEUMANN_SET"
typedef unsigned long EntityHandle; // identical to MOAB's EntityHandle type.  We define our own because we don't want to require MOAB, at least not yet.
  
  typedef Teuchos::RCP<const Epetra_Comm> Epetra_CommPtr;
  typedef Teuchos::RCP<const Teuchos::Comm<int>> Teuchos_CommPtr;

// Camellia forward declarations and typedefs
class BasisCache;
class BasisFactory;
class Cell;
class DofOrdering;
class DofOrderingFactory;
class Element;
class ElementType;
class EntitySet;
class ErrorIndicator;
class ERKTableau;
class GlobalDofAssignment;
class LagrangeConstraints;
class Mesh;
class MeshPartitionPolicy;
class MeshTopology;
class MeshTopologyView;
class ParameterFunction;
class RefinementPattern;
class SolutionStage;
class SpatialFilter;
class Var;
class VarFactory;
// templates
template<class Scalar, class ArrayScalar>
class Basis;
template <typename Scalar=double>
class TBC;
template <typename Scalar=double>
class TBF;
template <typename Scalar=double>
class TIP;
template <typename Scalar=double>
class TFunction;
template <typename Scalar=double>
class TLinearTerm;
template <typename Scalar=double>
class TRefinementStrategy;
template <typename Scalar=double>
class TRHS;
template <typename Scalar=double>
class TRieszRep;
template <typename Scalar=double>
class TSolution;
template <typename Scalar=double>
class TSolver;
template<class Scalar, class ArrayScalar>
class VectorizedBasis;

typedef Teuchos::RCP< Camellia::Basis<double, Intrepid::FieldContainer<double,0> > > BasisPtr;
typedef Teuchos::RCP<BasisCache> BasisCachePtr;
typedef Teuchos::RCP<BasisFactory> BasisFactoryPtr;
typedef Teuchos::RCP<Cell> CellPtr;
typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;
typedef Teuchos::RCP<DofOrderingFactory> DofOrderingFactoryPtr;
typedef Teuchos::RCP<Element> ElementPtr;
typedef Teuchos::RCP<ElementType> ElementTypePtr;
typedef Teuchos::RCP<ERKTableau> ERKTableauPtr;
typedef Teuchos::RCP<EntitySet> EntitySetPtr;
typedef Teuchos::RCP<ErrorIndicator> ErrorIndicatorPtr;
typedef Teuchos::RCP<GlobalDofAssignment> GlobalDofAssignmentPtr;
typedef Teuchos::RCP<Mesh> MeshPtr;
typedef Teuchos::RCP<MeshPartitionPolicy> MeshPartitionPolicyPtr;
typedef Teuchos::RCP<MeshTopology> MeshTopologyPtr;
typedef Teuchos::RCP<MeshTopologyView> MeshTopologyViewPtr;
typedef Teuchos::RCP<ParameterFunction> ParameterFunctionPtr;
typedef Teuchos::RCP<RefinementPattern> RefinementPatternPtr;
typedef Teuchos::RCP<SolutionStage> SolutionStagePtr;
typedef Teuchos::RCP<SpatialFilter> SpatialFilterPtr;
typedef Teuchos::RCP<Var> VarPtr;
typedef Teuchos::RCP<VarFactory> VarFactoryPtr;
typedef Teuchos::RCP<VectorizedBasis< double, Intrepid::FieldContainer<double,0> > > VectorBasisPtr;
  
  typedef Teuchos::RCP<const MeshTopology> ConstMeshTopologyPtr;
  typedef Teuchos::RCP<const MeshTopologyView> ConstMeshTopologyViewPtr;
  
// templates
template <typename Scalar>
using TBCPtr = Teuchos::RCP<TBC<Scalar> >;
typedef TBC<double> BC;
typedef TBCPtr<double> BCPtr;

template <typename Scalar>
using TBFPtr = Teuchos::RCP<TBF<Scalar> >;
typedef TBF<double> BF;
typedef TBFPtr<double> BFPtr;

template <typename Scalar>
using TIPPtr = Teuchos::RCP<TIP<Scalar> >;
typedef TIP<double> IP;
typedef TIPPtr<double> IPPtr;

template <typename Scalar>
using TFunctionPtr = Teuchos::RCP<TFunction<Scalar> >;
typedef TFunction<double> Function;
typedef TFunctionPtr<double> FunctionPtr;

template <typename Scalar>
using TLinearTermPtr = Teuchos::RCP<TLinearTerm<Scalar> >;
typedef TLinearTerm<double> LinearTerm;
typedef TLinearTermPtr<double> LinearTermPtr;

template <typename Scalar>
using TRefinementStrategyPtr = Teuchos::RCP<TRefinementStrategy<Scalar> >;
typedef TRefinementStrategy<double> RefinementStrategy;
typedef TRefinementStrategyPtr<double> RefinementStrategyPtr;

template <typename Scalar>
using TRHSPtr = Teuchos::RCP<TRHS<Scalar> >;
typedef TRHS<double> RHS;
typedef TRHSPtr<double> RHSPtr;

template <typename Scalar>
using TRieszRepPtr = Teuchos::RCP<TRieszRep<Scalar> >;
typedef TRieszRep<double> RieszRep;
typedef TRieszRepPtr<double> RieszRepPtr;

template <typename Scalar>
using TSolutionPtr = Teuchos::RCP<TSolution<Scalar> >;
typedef TSolution<double> Solution;
typedef TSolutionPtr<double> SolutionPtr;

template <typename Scalar>
using TSolverPtr = Teuchos::RCP<TSolver<Scalar> >;
typedef TSolver<double> Solver;
typedef TSolverPtr<double> SolverPtr;

// minor typedefs
template <typename Scalar>
using TLinearSummand = std::pair<TFunctionPtr<Scalar>, VarPtr>;
typedef TLinearSummand<double> LinearSummand;
template <typename Scalar>
using TBilinearTerm = std::pair<TLinearTermPtr<Scalar>,TLinearTermPtr<Scalar>>;
typedef TBilinearTerm<double> BilinearTerm;
template <typename Scalar>
using TDirichletBC = std::pair<SpatialFilterPtr,TFunctionPtr<Scalar>>;
typedef TDirichletBC<double> DirichletBC;

template <typename Scalar=double>
class TAmesos2Solver;
typedef TAmesos2Solver<double> Amesos2Solver;

typedef Teuchos::RCP< Tpetra::Map<> > MapPtr;
typedef Teuchos::RCP< const Tpetra::Map<> > ConstMapPtr;
template <typename Scalar>
using TMatrix = Tpetra::CrsMatrix<Scalar>;
template <typename Scalar>
using TMatrixPtr = Teuchos::RCP< TMatrix<Scalar> >;
using TGraph = TMatrix<double>::crs_graph_type;
using TGraphPtr = Teuchos::RCP< TGraph >;
using TGraphConstPtr = Teuchos::RCP< const TGraph >;
using TMap = TMatrix<double>::map_type;
using TMapPtr = Teuchos::RCP< TMap >;
using TMapConstPtr = Teuchos::RCP< const TMap >;
template <typename Scalar>
using TVector = Tpetra::MultiVector<Scalar>;
template <typename Scalar>
using TVectorPtr = Teuchos::RCP< TVector<Scalar> >;
template <typename Scalar>
using TVectorConstPtr = Teuchos::RCP< const TVector<Scalar> >;
// typedef TVectorPtr<double> VectorPtr;
template <typename Scalar>
using TOperator = Tpetra::Operator<Scalar>;
  
#ifdef KOKKOS_ENABLE_CUDA
  using DeviceType = Kokkos::Device<Kokkos::Cuda,Kokkos::CudaSpace>;  // we want to avoid UVM; Kokkos default may be Kokkos::CudaUVMSpace
#elif defined(KOKKOS_ENABLE_HIP)
  using DeviceType = Kokkos::Device<Kokkos::HIP,Kokkos::HIPSpace>;
#else
  using DeviceType = typename Kokkos::DefaultExecutionSpace::device_type;
#endif
  using DeviceMemorySpace    = DeviceType::memory_space;
  using DeviceExecutionSpace = DeviceType::execution_space;
  using HostMemorySpace      = Kokkos::HostSpace::memory_space;
  
  template<typename Scalar>
  using HostView   = Kokkos::DynRankView<Scalar, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  
  template<typename Scalar>
  using HostViewManaged = Kokkos::View<Scalar*, Kokkos::HostSpace >;

  template<typename Scalar>
  using ViewTraits = typename HostView<Scalar>::drvtraits;
  
  #ifndef KOKKOS_ENABLE_HIP
  template<typename Scalar>
  using DeviceView = Kokkos::DynRankView<typename ViewTraits<Scalar>::value_type, typename ViewTraits<Scalar>::array_layout, DeviceType>;
  template<typename Scalar>
  using DeviceViewConst = Kokkos::DynRankView<const typename ViewTraits<Scalar>::value_type, typename ViewTraits<Scalar>::array_layout, DeviceType>;
  #else
  template<typename Scalar>
  using DeviceView = Kokkos::DynRankView<typename ViewTraits<Scalar>::value_type, Kokkos::LayoutLeft, DeviceType>;
  template<typename Scalar>
  using DeviceViewConst = Kokkos::DynRankView<const typename ViewTraits<Scalar>::value_type, Kokkos::LayoutLeft, DeviceType>;
  #endif

  template<typename Scalar>
  using HostViewConst   = Kokkos::DynRankView<const Scalar, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  template<typename Scalar>
  using UnmanagedHostView = Kokkos::View<Scalar*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
}

#endif
