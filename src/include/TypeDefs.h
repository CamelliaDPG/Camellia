//
//  TypeDefs.h
//  Camellia
//
//  Created by Truman Ellis on 3/27/15.
//

#ifndef TypeDefs_h
#define TypeDefs_h

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

#include <Intrepid_FieldContainer.hpp>

namespace Camellia {
	// Basic typedefs
	typedef unsigned IndexType;
	typedef unsigned GlobalIndexType;
	typedef unsigned PartitionIndexType; // for partition numbering
	typedef unsigned CellIDType;
	typedef int GlobalIndexTypeToCast; // for constructing Epetra_Maps, etc.  (these like either int or long long)

	// Trilinos typedefs
	// typedef double Scalar;
	typedef Teuchos::RCP< Tpetra::CrsMatrix<double,IndexType,GlobalIndexType> > MatrixPtr;
	typedef Teuchos::RCP< Tpetra::MultiVector<double,IndexType,GlobalIndexType> > VectorPtr;
	typedef Teuchos::RCP< Tpetra::Map<IndexType,GlobalIndexType> > MapPtr;

	typedef Teuchos::RCP< Intrepid::FieldContainer<double> > FCPtr;
	typedef Teuchos::RCP< const Intrepid::FieldContainer<double> > constFCPtr;

	// Camellia forward declarations and typedefs
	class BasisCache;
	class BasisFactory;
	class BC;
	class Cell;
	class DofOrdering;
	class DofOrderingFactory;
	class Element;
	class ElementType;
	class GlobalDofAssignment;
	class LagrangeConstraints;
	class Mesh;
	class MeshPartitionPolicy;
	class MeshTopology;
  class ParameterFunction;
  class RefinementPattern;
	class RefinementStrategy;
	class RieszRep;
	class Solver;
	class SpatialFilter;
	class Var;
	class VarFactory;
  // templates
  template <typename Scalar=double>
	class TBF;
  template <typename Scalar=double>
	class TIP;
  template <typename Scalar=double>
	class TFunction;
  template <typename Scalar=double>
	class TLinearTerm;
  template <typename Scalar=double>
	class TRHS;
  template <typename Scalar=double>
	class TSolution;

	typedef Teuchos::RCP<BasisCache> BasisCachePtr;
	typedef Teuchos::RCP<BasisFactory> BasisFactoryPtr;
	typedef Teuchos::RCP<BC> BCPtr;
	typedef Teuchos::RCP<Cell> CellPtr;
	typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;
	typedef Teuchos::RCP<DofOrderingFactory> DofOrderingFactoryPtr;
	typedef Teuchos::RCP<Element> ElementPtr;
	typedef Teuchos::RCP<ElementType> ElementTypePtr;
	typedef Teuchos::RCP<GlobalDofAssignment> GlobalDofAssignmentPtr;
	typedef Teuchos::RCP<Mesh> MeshPtr;
	typedef Teuchos::RCP<MeshPartitionPolicy> MeshPartitionPolicyPtr;
	typedef Teuchos::RCP<MeshTopology> MeshTopologyPtr;
	typedef Teuchos::RCP<ParameterFunction> ParameterFunctionPtr;
  typedef Teuchos::RCP<RefinementPattern> RefinementPatternPtr;
	typedef Teuchos::RCP<RefinementStrategy> RefinementStrategyPtr;
	typedef Teuchos::RCP<RieszRep> RieszRepPtr;
	typedef Teuchos::RCP<Solver> SolverPtr;
	typedef Teuchos::RCP<SpatialFilter> SpatialFilterPtr;
	typedef Teuchos::RCP<Var> VarPtr;
  // templates
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
    using TRHSPtr = Teuchos::RCP<TRHS<Scalar> >;
  typedef TRHS<double> RHS;
  typedef TRHSPtr<double> RHSPtr;
  template <typename Scalar>
    using TSolutionPtr = Teuchos::RCP<TSolution<Scalar> >;
  typedef TSolution<double> Solution;
  typedef TSolutionPtr<double> SolutionPtr;

  // minor typedefs
  template <typename Scalar>
    using TLinearSummand = std::pair<TFunctionPtr<Scalar>, VarPtr>;
  typedef TLinearSummand<double> LinearSummand;
  template <typename Scalar>
    using TBilinearTerm = std::pair<TLinearTermPtr<Scalar>,TLinearTermPtr<Scalar>>;
  typedef TBilinearTerm<double> BilinearTerm;
}


#endif
