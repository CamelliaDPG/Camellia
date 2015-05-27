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

namespace Camellia
{
// Basic typedefs
typedef unsigned IndexType;
typedef unsigned GlobalIndexType;
typedef unsigned PartitionIndexType; // for partition numbering
typedef unsigned CellIDType;
typedef int GlobalIndexTypeToCast; // for constructing Epetra_Maps, etc.  (these like either int or long long)

// Trilinos typedefs
typedef Teuchos::RCP< Intrepid::FieldContainer<double> > FCPtr;
typedef Teuchos::RCP< const Intrepid::FieldContainer<double> > constFCPtr;

// Camellia forward declarations and typedefs
class BasisCache;
class BasisFactory;
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
class SpatialFilter;
class Var;
class VarFactory;
// templates
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

typedef Teuchos::RCP<BasisCache> BasisCachePtr;
typedef Teuchos::RCP<BasisFactory> BasisFactoryPtr;
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
typedef Teuchos::RCP<SpatialFilter> SpatialFilterPtr;
typedef Teuchos::RCP<Var> VarPtr;
typedef Teuchos::RCP<VarFactory> VarFactoryPtr;
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

typedef Teuchos::RCP< Tpetra::Map<IndexType,GlobalIndexType> > MapPtr;
typedef Teuchos::RCP< const Tpetra::Map<IndexType,GlobalIndexType> > ConstMapPtr;
template <typename Scalar>
using TMatrixPtr = Teuchos::RCP< Tpetra::CrsMatrix<Scalar,IndexType,GlobalIndexType> >;
// typedef TMatrixPtr<double> MatrixPtr;
template <typename Scalar>
using TVectorPtr = Teuchos::RCP< Tpetra::MultiVector<Scalar,IndexType,GlobalIndexType> >;
// typedef TVectorPtr<double> VectorPtr;
}


#endif
