//
//  TypeDefs.h
//  Camellia
//
//  Created by Truman Ellis on 3/27/15.
//

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

// Basic typedefs
typedef unsigned IndexType;
typedef unsigned GlobalIndexType;
typedef unsigned PartitionIndexType; // for partition numbering
typedef unsigned CellIDType;
typedef int GlobalIndexTypeToCast; // for constructing Epetra_Maps, etc.  (these like either int or long long)

// Trilinos typedefs
typedef double Scalar;
typedef Teuchos::RCP< Tpetra::CrsMatrix<Scalar,IndexType,GlobalIndexType> > MatrixPtr;
typedef Teuchos::RCP< Tpetra::MultiVector<Scalar,IndexType,GlobalIndexType> > VectorPtr;
typedef Teuchos::RCP< Tpetra::Map<IndexType,GlobalIndexType> > MapPtr;

typedef Teuchos::RCP< Intrepid::FieldContainer<double> > FCPtr;
typedef Teuchos::RCP< const Intrepid::FieldContainer<double> > constFCPtr;

// Camellia forward declarations and typedefs
class BasisCache;
class BC;
class Element;
class Function;
class IP;
class LagrangeConstraints;
class Mesh;
class RHS;
class Solution;

typedef Teuchos::RCP<BC> BCPtr;
typedef Teuchos::RCP<Element> ElementPtr;
typedef Teuchos::RCP<Function> FunctionPtr;
typedef Teuchos::RCP<IP> IPPtr;
typedef Teuchos::RCP<Mesh> MeshPtr;
typedef Teuchos::RCP<RHS> RHSPtr;
typedef Teuchos::RCP<Solution> SolutionPtr;
