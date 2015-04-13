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
	class BF;
	class Cell;
	class DofOrdering;
	class DofOrderingFactory;
	class Element;
	class ElementType;
  template <typename Scalar=double>
	class Function;
	class GlobalDofAssignment;
	class IP;
	class LagrangeConstraints;
	class LinearTerm;
	class Mesh;
	class MeshPartitionPolicy;
	class MeshTopology;
  class RefinementPattern;
	class RefinementStrategy;
	class RieszRep;
	class RHS;
  template <typename Scalar=double>
	class Solution;
	class Solver;
	class SpatialFilter;
	class Var;
	class VarFactory;

	typedef Teuchos::RCP<BasisCache> BasisCachePtr;
	typedef Teuchos::RCP<BasisFactory> BasisFactoryPtr;
	typedef Teuchos::RCP<BC> BCPtr;
	typedef Teuchos::RCP<BF> BFPtr;
	typedef Teuchos::RCP<Cell> CellPtr;
	typedef Teuchos::RCP<DofOrdering> DofOrderingPtr;
	typedef Teuchos::RCP<DofOrderingFactory> DofOrderingFactoryPtr;
	typedef Teuchos::RCP<Element> ElementPtr;
	typedef Teuchos::RCP<ElementType> ElementTypePtr;
	typedef Teuchos::RCP<Function<double> > FunctionPtr;
	typedef Teuchos::RCP<GlobalDofAssignment> GlobalDofAssignmentPtr;
	typedef Teuchos::RCP<IP> IPPtr;
	typedef Teuchos::RCP<LinearTerm> LinearTermPtr;
	typedef Teuchos::RCP<Mesh> MeshPtr;
	typedef Teuchos::RCP<MeshPartitionPolicy> MeshPartitionPolicyPtr;
	typedef Teuchos::RCP<MeshTopology> MeshTopologyPtr;
  typedef Teuchos::RCP<RefinementPattern> RefinementPatternPtr;
	typedef Teuchos::RCP<RefinementStrategy> RefinementStrategyPtr;
	typedef Teuchos::RCP<RieszRep> RieszRepPtr;
	typedef Teuchos::RCP<RHS> RHSPtr;
	typedef Teuchos::RCP<Solution<double> > SolutionPtr;
	typedef Teuchos::RCP<Solver> SolverPtr;
	typedef Teuchos::RCP<SpatialFilter> SpatialFilterPtr;
	typedef Teuchos::RCP<Var> VarPtr;
}


#endif
