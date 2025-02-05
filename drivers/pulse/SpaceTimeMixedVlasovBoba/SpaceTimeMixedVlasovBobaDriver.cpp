//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"
#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "LocalDofMapper.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"

#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

// TODO<bugfix> we shouldn't need to specify BOBA_ENABLE_EIGEN
#define BOBA_ENABLE_EIGEN On
//#define BOBA_CHECKPOINTS

#include "BOBA/boba.hpp"

#include "wrappers.hpp"

using namespace Camellia;
using namespace std;

using host_matrix = ::boba::matrix<::boba::execution_space::CPU, double, size_t>;
using host_vector = ::boba::vector<::boba::execution_space::CPU, double, size_t>;

using host_vector_tensor = ::boba::tensor<6, ::boba::execution_space::CPU, double, size_t>;

using host_ttm = ::boba::tensor_train_matrix<6, ::boba::execution_space::CPU, double, size_t>;
using host_tt = ::boba::tensor_train<6, ::boba::execution_space::CPU, double, size_t>;

struct subvectors
{
  static constexpr size_t epsw = 0;
  static constexpr size_t tnx = 1;
  static constexpr size_t tnt = 2;
  static constexpr size_t tnv = 3;
  static constexpr size_t f = 4;
  static constexpr size_t size = 5;
};

template <typename vector_t>
vector_t
extractBobaSubVector(const Teuchos::RCP<Epetra_FEVector>& originalVector,
                     const std::set<GlobalIndexType>& allIdsToKeep) {
  // TODO add static_assert to ensure construction on CPU?

  // use index type exposed by the vector type
  using index_t = typename vector_t::index_type;

  TEUCHOS_TEST_FOR_EXCEPTION(originalVector->Comm().NumProc() != 1,
                             std::runtime_error,
                             "Boba vector extraction requires single-rank MPI execution.");

  // construct contiguous indices
  std::map<GlobalIndexType, index_t> idToIndex;
  index_t numElem = 0;
  for (const auto &id : allIdsToKeep) {
    idToIndex[id] = numElem;
    ++numElem;
  }

  // allocate boba object for storing subvector
  vector_t subVector({numElem});
  subVector.fill_with_zeros();

  // fill in subvector
  auto subVectorView = subVector.view();
  for (const auto &id : allIdsToKeep) {
    // first [0] index on RHS selects the first (only?) vector out of the MultiVector
    subVectorView(idToIndex[id]) += (*originalVector)[0][id];
  }

  return subVector;
}

template <typename matrix_t>
matrix_t
extractBobaSubMatrix(const Teuchos::RCP<Epetra_CrsMatrix>& originalMatrix,
                     const std::set<GlobalIndexType>& allRowsToKeep,
                     const std::set<GlobalIndexType>& allColsToKeep) {
  // TODO add static_assert to ensure construction on CPU?

  // use index type exposed by the matrix type
  using index_t = typename matrix_t::index_type;

  TEUCHOS_TEST_FOR_EXCEPTION(originalMatrix->Comm().NumProc() != 1,
                             std::runtime_error,
                             "Boba vector extraction requires single-rank MPI execution.");

  // construct contiguous indices for rows and columns
  std::map<GlobalIndexType, index_t> rowIdToIndex;
  index_t numRow = 0;
  for (const auto &rowId : allRowsToKeep) {
    rowIdToIndex[rowId] = numRow;
    ++numRow;
  }

  std::map<GlobalIndexType, index_t> colIdToIndex;
  index_t numCol = 0;
  for (const auto &colId : allColsToKeep) {
    colIdToIndex[colId] = numCol;
    ++numCol;
  }

  // allocate boba objects for storing submatrix
  matrix_t subMatrix({numRow, numCol});
  subMatrix.fill_with_zeros();

  // fill in submatrix
  auto subMatrixView = subMatrix.view();
  for (const auto &rowId : allRowsToKeep) {
    const index_t rowIndex = rowIdToIndex[rowId];
    // following types are hard coded in Epetra
    int numEntries = 0;
    double *values = nullptr;
    int *indices = nullptr;
    originalMatrix->ExtractMyRowView(static_cast<int>(rowId), numEntries, values, indices);
    for (int i = 0; i < numEntries; ++i) {
      const GlobalIndexType colId = static_cast<GlobalIndexType>(indices[i]);
      const index_t colIndex = colIdToIndex[colId];
      if (allColsToKeep.count(colId) != 0) {
        subMatrixView({rowIndex, colIndex}) += values[i];
      }
    }
  }

  return subMatrix;
}

template <typename matrix_t, typename vector_t, size_t rows, size_t cols>
std::tuple<operator_wrapper<matrix_t, vector_t, rows, cols>,
           vector_wrapper<vector_t, rows>,
           vector_wrapper<vector_t, cols>>
setupBlockSystem(const Teuchos::RCP<Epetra_CrsMatrix>& originalMatrix,
                 const Teuchos::RCP<Epetra_FEVector>& originalRhsVector,
                 const Teuchos::RCP<Epetra_FEVector>& originalLhsVector,
                 const boba::Array<std::set<GlobalIndexType>, rows>& rowPartition,
                 const boba::Array<std::set<GlobalIndexType>, cols>& colPartition) {
  operator_wrapper<matrix_t, vector_t, rows, cols> blockMatrix;
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      blockMatrix({i, j}) = extractBobaSubMatrix<matrix_t>(originalMatrix, rowPartition[i], colPartition[j]);
    }
  }

  vector_wrapper<vector_t, rows> blockRhsVector;
  for (size_t i = 0; i < rows; ++i) {
    blockRhsVector(i) = extractBobaSubVector<vector_t>(originalRhsVector, rowPartition[i]);
  }

  vector_wrapper<vector_t, cols> blockLhsVector;
  for (size_t j = 0; j < cols; ++j) {
    blockLhsVector(j) = extractBobaSubVector<vector_t>(originalLhsVector, colPartition[j]);
  }

  return {blockMatrix, blockRhsVector, blockLhsVector};
}

//! This implementation is thanks to ChatGPT.  We'll see if it works!
void reorderMatrix(const Epetra_CrsMatrix& originalMatrix,
                   const std::map<int, int>& reorderMap,
                   Epetra_CrsMatrix& reorderedMatrix) {
    int numRows = originalMatrix.NumMyRows();
    int numCols = originalMatrix.NumMyCols();

    // Iterate over the original matrix
    for (int i = 0; i < numRows; ++i) {
        double* values;
        int* indices;
        int numEntries;

        // Get the row entries
        originalMatrix.ExtractMyRowView(i, numEntries, values, indices);

        // Find the new row index using the reorder map
        auto it = reorderMap.find(i);
        if (it != reorderMap.end()) {
            int newRow = it->second;

            // Insert the values into the reordered matrix
            for (int j = 0; j < numEntries; ++j) {
                // Find the new column index using the reorder map
                auto colIt = reorderMap.find(indices[j]);
                if (colIt != reorderMap.end()) {
                    int newCol = colIt->second;
                    reorderedMatrix.InsertGlobalValues(newRow, 1, &values[j], &newCol);
                }
            }
        }
    }

    // Finish the construction of the reordered matrix
    reorderedMatrix.FillComplete();
}

std::pair<Epetra_CrsMatrix, Epetra_Vector>
subMatrixAndSubVector(const Epetra_CrsMatrix& originalMatrix, const Epetra_FEVector& originalVector,
                      const std::set<int>& allRowsToKeep) {
  Epetra_Map originalRowMap = originalMatrix.RowMatrixRowMap();
  Epetra_Map originalColMap = originalMatrix.RowMatrixColMap();
  
  std::vector<int> myRowsToKeep;
  for (auto row : allRowsToKeep)
  {
    if (originalRowMap.LID(row) != -1)
    {
      // owned locally
      myRowsToKeep.push_back(row);
    }
  }
  
  Epetra_Map subRowMap(-1, int(myRowsToKeep.size()), &myRowsToKeep[0], 0, originalRowMap.Comm());

  int maxRowSize = 0; // will cause more mallocs during insertion into the CrsMatrix, but will minimize the amount of memory allocated now.
  Epetra_FECrsMatrix subMatrix(::Copy, subRowMap, maxRowSize);
  
  // Epetra_Vector(const Epetra_BlockMap& Map, bool zeroOut = true);
//  int ReplaceMyValues(int NumEntries, int BlockOffset, const double * Values, const int * Indices);
  
  Epetra_Vector subVector(subRowMap);
  
  // Iterate over the original matrix
  for (auto row : myRowsToKeep) {
    int i = originalRowMap.LID(row);
    int i_new = subRowMap.LID(row);
    subVector[i_new] = originalVector[0][i];
    
    double* values;
    int* indices;
    int numEntries;

    // Get the row entries
    originalMatrix.ExtractMyRowView(i, numEntries, values, indices);

    for (int j = 0; j < numEntries; ++j) {
      auto col_lid = indices[j];
      auto col = originalColMap.GID(col_lid);
      if (allRowsToKeep.find(col) != allRowsToKeep.end())
      {
        subMatrix.InsertGlobalValues(row, 1, &values[j], &col);
      }
    }
  }

  // Finish the construction of the submatrix
  subMatrix.FillComplete();
  
  return {subMatrix, subVector};
}

std::tuple<Teuchos::RCP<Epetra_CrsMatrix>, Teuchos::RCP<Epetra_Vector>, Teuchos::RCP<Epetra_Vector> >
subMatrixAndSubVectors(Teuchos::RCP<Epetra_CrsMatrix> originalMatrix, const Teuchos::RCP<Epetra_FEVector> originalLHSVector, const Teuchos::RCP<Epetra_FEVector> originalRHSVector,
                       const std::set<int>& allRowsToKeep, const std::set<int> &allColsToKeep) {
  Epetra_Map originalRowMap = originalMatrix->RowMatrixRowMap();
  Epetra_Map originalColMap = originalMatrix->RowMatrixColMap();
  
  std::vector<int> myRowsToKeep;
  for (auto row : allRowsToKeep)
  {
    if (originalRowMap.LID(row) != -1)
    {
      // owned locally
      myRowsToKeep.push_back(row);
    }
  }
  
  std::vector<int> myColsToKeep;
  for (auto col : allColsToKeep)
  {
    if (originalColMap.LID(col) != -1)
    {
      // owned locally
      myColsToKeep.push_back(col);
    }
  }
  
  // rhsVector's map will match the submatrix row map, which extracts the rows to keep from the original row map
  // lhsVector's map will be a separate map that extracts the *columns* to keep from the original row map
  
  Epetra_Map subMatrixRowMap(-1, int(myRowsToKeep.size()), &myRowsToKeep[0], 0, originalRowMap.Comm());
  Epetra_Map    lhsVectorMap(-1, int(myColsToKeep.size()), &myColsToKeep[0], 0, originalRowMap.Comm());

  int maxRowSize = 0; // will cause more mallocs during insertion into the CrsMatrix, but will minimize the amount of memory allocated now.
  Teuchos::RCP<Epetra_FECrsMatrix> subMatrix = Teuchos::rcp(new Epetra_FECrsMatrix(::Copy, subMatrixRowMap, maxRowSize));
  
  // Epetra_Vector(const Epetra_BlockMap& Map, bool zeroOut = true);
//  int ReplaceMyValues(int NumEntries, int BlockOffset, const double * Values, const int * Indices);
  
  Teuchos::RCP<Epetra_Vector> lhsVector = Teuchos::rcp(new Epetra_Vector(lhsVectorMap));
  
  // we support having a null RHS vector, in which case we just skip that extraction and return a null value
  Teuchos::RCP<Epetra_Vector> rhsVector;
  if (originalRHSVector != Teuchos::null)
  {
    rhsVector = Teuchos::rcp(new Epetra_Vector(subMatrixRowMap));
  }
  
  // Iterate over the original matrix
  for (auto row : myRowsToKeep) {
    int i     =  originalRowMap.LID(row);
    int i_new = subMatrixRowMap.LID(row);
    if (originalRHSVector != Teuchos::null)
    {
      (*rhsVector)[i_new] = (*originalRHSVector)[0][i];
    }
    
    double* values;
    int* indices;
    int numEntries;

    // Get the row entries
    originalMatrix->ExtractMyRowView(i, numEntries, values, indices);

    for (int j = 0; j < numEntries; ++j) {
      auto col_lid = indices[j];
      auto col = originalColMap.GID(col_lid);
      if (allColsToKeep.find(col) != allColsToKeep.end())
      {
        subMatrix->InsertGlobalValues(row, 1, &values[j], &col);
      }
    }
  }
  
  for (auto col : myColsToKeep) {
    int i = originalRowMap.LID(col);
    int i_new = lhsVectorMap.LID(col);
    (*lhsVector)[i_new] = (*originalLHSVector)[0][i];
  }

  // Finish the construction of the submatrix
  subMatrix->FillComplete(lhsVectorMap, subMatrixRowMap);
  
  return {subMatrix, lhsVector, rhsVector};
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  const int rank = mpiSession.getRank();
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  {
    int spaceDim = 1;
    int velocityDim = 1;
    int numElements_x = 2;
    int numElements_t = 2;
    int numElements_v = 2;
    int elements_factor = 1;
    size_t krylov_method = 0;
    int meshDim = spaceDim + 1 + velocityDim;
    double beta = 1.0;
    
    vector<vector<double>> domainDim(meshDim,vector<double>{0.0,1.0}); // first index: spaceDim; second: 0/1 for x0, x1, etc.
    int p = 2;
    int dp = spaceDim+velocityDim;
    
    std::string modeString = "single"; // single run; other options are "h" and "p" for h and p convergence tests
    std::string sortedMatrixPath = "";
    
    cmdp.setOption("krylov_method", &krylov_method );
    cmdp.setOption("elements_factor", &elements_factor );
    cmdp.setOption("xElements", &numElements_x );
    cmdp.setOption("tElements", &numElements_t );
    cmdp.setOption("vElements", &numElements_v );
    cmdp.setOption("p",  &p );
    cmdp.setOption("dp", &dp);
    cmdp.setOption("x0", &domainDim[0][0] );
    cmdp.setOption("x1", &domainDim[0][1] );
    cmdp.setOption("t0", &domainDim[1][0] );
    cmdp.setOption("t1", &domainDim[1][1] );
    cmdp.setOption("v0", &domainDim[2][0] );
    cmdp.setOption("v1", &domainDim[2][1] );
    cmdp.setOption("mode", &modeString, "single, h, or p");
    cmdp.setOption("beta", &beta, "beta weight for L^2 term in graph norm (beta -> 0 should be L^2 projection, according to usual DPG theory)");
    cmdp.setOption("varSortedMatrixOutputPath", &sortedMatrixPath);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      return -1;
    }

    numElements_x *= elements_factor;
    numElements_t *= elements_factor;
    numElements_v *= elements_factor;
    
    always_checkpoint();
    std::vector<int> numElements_x_series, numElements_t_series, numElements_v_series;
    std::vector<int> p_series, dp_series;
    if (modeString == "single")
    {
      numElements_x_series.push_back(numElements_x);
      numElements_t_series.push_back(numElements_t);
      numElements_v_series.push_back(numElements_v);
      p_series.push_back(p);
      dp_series.push_back(dp);
    }
    else if (modeString == "h")
    {
      const int numRefinements = 5;
      int elementMultiplier = 1;
      for (int i=0; i<numRefinements; i++)
      {
        numElements_x_series.push_back(numElements_x * elementMultiplier);
        numElements_t_series.push_back(numElements_t * elementMultiplier);
        numElements_v_series.push_back(numElements_v * elementMultiplier);
        p_series.push_back(p);
        dp_series.push_back(dp);
        elementMultiplier *= 2;
      }
    }
    else if (modeString == "p")
    {
      const int numRefinements = 5;
      for (int i=0; i<numRefinements; i++)
      {
        numElements_x_series.push_back(numElements_x);
        numElements_t_series.push_back(numElements_t);
        numElements_v_series.push_back(numElements_v);
        p_series.push_back(p + i);
        dp_series.push_back(dp);
      }
    }
    else if (modeString == "dp")
    {
      const int numRefinements = 5;
      for (int i=0; i<numRefinements; i++)
      {
        int maxTraceDegree = p;
        int maxTestDegree  = p + dp + i;
        
        numElements_x_series.push_back(numElements_x);
        numElements_t_series.push_back(numElements_t);
        numElements_v_series.push_back(numElements_v);
        p_series.push_back(p);
        dp_series.push_back(dp+i);
      }
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported mode");
    }
    
    always_checkpoint();
    VarFactoryPtr vf = VarFactory::varFactory();
    bool matchMatlab = false;
    Space L2_space   = matchMatlab ? HGRAD_DISC : L2; // use HGRAD_DISC (and modify polyOrderEnhancements below) if seeking to match values with the MATLAB code
    VarPtr tn_hat    = vf->fluxVar("tn_hat", L2_space);
    VarPtr f         = vf->fieldVar("f",     L2_space);
    VarPtr eps_w     = vf->fieldVar("eps_w", HGRAD_DISC);
    
    // test variables are discretized the same way as trial, so can be represented by the same VarPtrs; we relabel them for clarity:
    auto w            = eps_w;
    auto delta_tn_hat = tn_hat;
    auto delta_f      = f;
    
    always_checkpoint();
    auto bubnovVF = vf->getBubnovFactory(Camellia::VarFactory::BUBNOV_TRIAL);
    
    auto vlasovBF = BF::bf(bubnovVF);
    
    static const int DX_DIM = 1;
    static const int DT_DIM = 2;
    static const int DV_DIM = 3;
    
    FunctionPtr E = Function::constant(1.0);
    FunctionPtr v = Function::zn(1); // v^1
    
    always_checkpoint();

    // B:
    vlasovBF->addTerm(   tn_hat, w);
    vlasovBF->addTerm(     -f, w->di(DT_DIM)); // dw/dt
    vlasovBF->addTerm(  - v*f, w->di(DX_DIM)); // dw/dx
    vlasovBF->addTerm(  - E*f, w->di(DV_DIM)); // dw/dv
    
    // B^T
    vlasovBF->addTerm(            eps_w, delta_tn_hat);
    vlasovBF->addTerm(eps_w->di(DT_DIM),     -delta_f); // dw/dt
    vlasovBF->addTerm(eps_w->di(DX_DIM), -v * delta_f); // dw/dx
    vlasovBF->addTerm(eps_w->di(DV_DIM), -E * delta_f); // dw/dv
    
    // G:
    vlasovBF->addTerm( v * eps_w->di(DX_DIM) + eps_w->di(DT_DIM) + E * eps_w->di(DV_DIM),
                       v *     w->di(DX_DIM) +     w->di(DT_DIM) + E *     w->di(DV_DIM));
    vlasovBF->addTerm( beta * eps_w, w );
    
    always_checkpoint();
    MeshPartitionPolicyPtr nullPartitionPolicy = Teuchos::null;
    Epetra_CommPtr nullComm = Teuchos::null;
    const bool allowPruning = false;  // to keep inter-mesh geometry determinations from getting too complex, don't allow the mesh topology to be pruned to keep only MPI-local geometry.
    
    BCPtr  bc  = BC::bc();
    RHSPtr rhs = RHS::rhs();
    
    FunctionPtr x = Function::xn(1); // x^1
    FunctionPtr t = Function::yn(1); // t^1
    
    FunctionPtr sin_pi_x_over_2 = TrigFunctions<double>::sin(0.5 * M_PI * x);
    FunctionPtr cos_2_pi_t      = TrigFunctions<double>::cos(2.0 * M_PI * t);
    FunctionPtr cos_pi_v        = TrigFunctions<double>::cos(      M_PI * v);
    
    FunctionPtr f_exact = sin_pi_x_over_2 * cos_2_pi_t + cos_pi_v;
    
    FunctionPtr rhs_exact = v * f_exact->dx() + f_exact->dy() + E * f_exact->dz();
    rhs->addTerm(rhs_exact * w);
    
    // impose boundary conditions on tn at the inflow.  This is defined by:
    //   (v,1,E) \cdot n < 0.
    // n for us is always parallel to one of the axes.  So depending on which side we are on,
    // there will be a requirement on v * n_x, n_t, or E n_v.  Since E is 1, and our domain is [0,1]^3,
    // the inflow boundary is as follows:
    // - the plane x = 0,
    // - the plane t = 0,
    // - the plane v = 0
    SpatialFilterPtr inflow = SpatialFilter::lessThanOrEqualToX(0.0) | SpatialFilter::lessThanOrEqualToY(0.0) | SpatialFilter::lessThanOrEqualToZ(0.0);
    FunctionPtr n = Function::normal();
    FunctionPtr tn_exact = v * f_exact * n->x() + f_exact * n->y() + f_exact * E * n->z();
    bc->addDirichlet(tn_hat, inflow, tn_exact);
    
    always_checkpoint();
    int numRuns = int(numElements_x_series.size());
    for (int i=0; i<numRuns; i++)
    {
      const int numElements_x_i = numElements_x_series[i];
      const int numElements_t_i = numElements_t_series[i];
      const int numElements_v_i = numElements_v_series[i];
      const int  p_i            =  p_series[i];
      const int dp_i            = dp_series[i];
      if (rank == 0)
      {
        std::cout << "Run " << i << " with p=" << p_i << " (dp=" << dp_i << ") on " << numElements_x_i << " x " << numElements_t_i << " x " << numElements_v_i << " mesh.\n";
      }
      
      auto spatialMeshTopo    = MeshFactory::intervalMeshTopology(domainDim[0][0], domainDim[0][1], numElements_x_i);
      auto spaceTimeMeshTopo  = MeshFactory::extrudedMeshTopology(spatialMeshTopo, {domainDim[1][0]}, {domainDim[1][1]}, {numElements_t_i});
      auto phaseSpaceMeshTopo = MeshFactory::extrudedMeshTopology(spaceTimeMeshTopo, {domainDim[2][0]}, {domainDim[2][1]}, {numElements_v_i});
      
      std::map<int,int> polyOrderEnhancements;
      // Nominally, f is in L^2, and the conforming trace f_hat would be a trace of H^1.
      // The natural thing would be for f_hat to be discretized with order 1 higher than f.
      // Although we use a non-conforming trace tn_hat, we still use 1 order higher, to match the conforming situation in basis degree.
      // The test functions are discretized with order 1 + dp higher than f.
      if (matchMatlab)
      {
        // HGRAD_DISC for f, tn_hat
        polyOrderEnhancements[f->ID()]      = -1; // subtract 1 to go from H^1 order to L^2 order
        polyOrderEnhancements[eps_w->ID()]  = dp_i;
      }
      else
      {
        polyOrderEnhancements[tn_hat->ID()] = 1; // add 1 to the natural L^2 degree
        polyOrderEnhancements[eps_w->ID()]  = dp_i;
      }
      always_checkpoint();

      MeshPtr mesh = Teuchos::rcp( new Mesh (phaseSpaceMeshTopo, vlasovBF, p_i, 0, polyOrderEnhancements, polyOrderEnhancements, nullPartitionPolicy, nullComm, allowPruning) );

      always_checkpoint();

      // we separate tn into three components: the space-normal, time-normal, and velocity-normal sides
      // map from the side ordinal in a cell to which sub-trace ID.
      // we start the numbering for the new sub-trace IDs after the last variable we created, eps_w
      int tn_x = eps_w->ID() + 1;
      int tn_t = tn_x + 1;
      int tn_v = tn_x + 2;
      map<unsigned,int> traceVariableForSideOrdinal;
      
      /*
       *  For tensor-product topologies T x L (L being the line topology), there are two "copies" of T, T0 and T1,
       *  and the enumeration of subcells of dimension d goes as follows:
          - d-dimensional subcells from T0
          - d-dimensional subcells from T1
          - ((d-1)-dimensional subcells of T) x L.
       
       We can view our cell topology as a line in x, extruded in t.  The result is then extruded in v.
       
       For the xt quad, the sides are numbered like this:
       
        ___1___
       |       |
       |2     3|
       |       |
        ---0---
           
       The xtv hexahedron has sides numbered like this:
       0: near xt quad
       1:  far xt quad
       2: 0 side from xt quad extruded in v: bottom xv quad
       3: 1 side from xt quad extruded in v: top xv quad
       4: 2 side from xt quad extruded in v: left tv quad
       5: 3 side from xt quad extruded in v: right tv quad
       
       x-normal is the tv quads: 4,5
       t-normal is the xv quads: 2,3
       v-normal is the xt quads: 0,1
       */
      traceVariableForSideOrdinal[0] = tn_v;
      traceVariableForSideOrdinal[1] = tn_v;
      traceVariableForSideOrdinal[2] = tn_t;
      traceVariableForSideOrdinal[3] = tn_t;
      traceVariableForSideOrdinal[4] = tn_x;
      traceVariableForSideOrdinal[5] = tn_x;
      
      always_checkpoint();

      SolutionPtr soln = Solution::solution(vlasovBF, mesh, bc, rhs);

      always_checkpoint();

      soln->solve();

      always_checkpoint();
      // compute L^2 error in f:
      auto K = soln->getStiffnessMatrix();
      auto lhs = soln->getLHSVector();
      auto rhs = soln->getRHSVector();

      always_checkpoint();

      int quadratureEnrichment = 8;
      FunctionPtr f_soln = Function::solution(f, soln, quadratureEnrichment);
      double l2_err = (f_soln - f_exact)->l2norm(mesh);

      double exact_l2_norm = f_exact->l2norm(mesh, quadratureEnrichment);
      double f_l2_norm     =  f_soln->l2norm(mesh, quadratureEnrichment);

      always_checkpoint();
      double energyError;
      {
        // compute error estimate
        GDAMinimumRule* gda = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
        const auto & myCellIDs = mesh->cellIDsInPartition();

        std::set<int> gramRowsThisRank; // use set to deduplicate

        for (auto cellID : myCellIDs)
        {
          // GDAMinimumRule::globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned int dim, unsigned int subcellOrdinal, bool upgradeToContinuousBasis, bool tensorTopologyMeansSpaceTime)
          auto dofIndices = gda->globalDofIndicesForVarOnSubcell(eps_w->ID(), cellID, meshDim, 0, false, false);
          gramRowsThisRank.insert(dofIndices.begin(), dofIndices.end());
        }

        // convert to vector for allGatherVariable()
        std::vector<int> myGramRows(gramRowsThisRank.begin(),gramRowsThisRank.end());
        std::vector<int> gramRowsVector;
        std::vector<int> offsets;
        auto epetraComm = MPIWrapper::EpetraComm(mesh->Comm());
        MPIWrapper::allGatherVariable(*epetraComm, gramRowsVector, myGramRows, offsets);

        std::set<int> gramRows(gramRowsVector.begin(),gramRowsVector.end()); // deduplicate with set
        
        auto gramMatrixAndEpsW = subMatrixAndSubVector(*K, *soln->getLHSVector(), gramRows);
        
        auto & G = gramMatrixAndEpsW.first;
        auto & epsw_h = gramMatrixAndEpsW.second;
        
        Epetra_Vector G_epsw(epsw_h); // copy constructor
        
        G.Apply(epsw_h, G_epsw);
        
        epsw_h.Dot(G_epsw, &energyError);
        energyError = sqrt(energyError);
      }

      always_checkpoint();
      if (rank == 0)
      {
        std::cout << std::endl;

        std::cout << "absolute L^2 error: " << l2_err << std::endl;
        std::cout << "relative L^2 error: " << l2_err / exact_l2_norm << "\n\n";

        std::cout << "beta in graph norm:      " << beta << std::endl;
        std::cout << "energy error:            " << energyError << std::endl;
        std::cout << "energy norm effectivity: " << energyError / l2_err << std::endl << std::endl;

        std::cout << "exact L^2 norm: " << exact_l2_norm << std::endl;
        std::cout << "L^2 norm of soln: " << f_l2_norm << std::endl;

        std::cout << std::endl;
      }

      // ---------------------------------------------
      // Block structure IDs
      // ---------------------------------------------

      std::set<GlobalIndexType> epsw_ids;
      std::set<GlobalIndexType> tn_x_ids;
      std::set<GlobalIndexType> tn_t_ids;
      std::set<GlobalIndexType> tn_v_ids;
      std::set<GlobalIndexType> f_ids;

      always_checkpoint();
      {
        // below, a quick and dirty way to extract Epetra submatrices and subvectors for the block system.
        
        // determine which global IDs belong to which variables
        auto trialIDs = vf->trialIDs();
        auto cellIDs = mesh->cellIDsInPartition();
        
        std::map<int, std::set<GlobalIndexType> > idsForTrialVar;
        std::map<GlobalIndexType,GlobalIndexType> idToSortedID;
        
        GDAMinimumRule* gda = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
        
        for (const auto &cellID: cellIDs)
        {
          auto dofMapper = gda->getDofMapper(cellID);
          
          auto cellTopo = mesh->getElementType(cellID)->cellTopoPtr;
          
          auto dim = cellTopo->getDimension();
          for (int d=0; d<=dim; d++)
          {
            int scCount = cellTopo->getSubcellCount(d);
            for (int sc=0; sc<scCount; sc++)
            {
              for (const int &trialID : trialIDs)
              {
                if (trialID == tn_hat->ID())
                {
                  // subdivide tn variables according to side ID:
                  if (d == dim-1)
                  {
                    const bool includeSubSubcells = false; // should not matter: L^2/HGRAD_DISC for trace/flux variable should place all the dofs on the side
                    auto globalIDs = dofMapper->globalIndicesForSubcell(trialID, d, sc, false, false, includeSubSubcells);
                    int tn_id = traceVariableForSideOrdinal[sc];
                    idsForTrialVar[tn_id].insert(globalIDs.begin(), globalIDs.end());
                  }
                }
                else
                {
                  const bool includeSubSubcells = false; // should not matter: L^2/HGRAD_DISC for field variable should place all the dofs on the cell itself
                  auto globalIDs = dofMapper->globalIndicesForSubcell(trialID, d, sc, false, false, includeSubSubcells);
                  idsForTrialVar[trialID].insert(globalIDs.begin(), globalIDs.end());
                }
              }
            }
          }
        }
        
        std::map<int,std::string> tn_ids;
        tn_ids[tn_x] = "tn_x";
        tn_ids[tn_t] = "tn_t";
        tn_ids[tn_v] = "tn_v";
        
        for (const auto & entry : tn_ids)
        {
          const auto & tn_id = entry.first;
          const auto & tn_str = entry.second;
          auto oldTrialGIDs = idsForTrialVar[tn_id];
          cout << tn_str << " has " << oldTrialGIDs.size() << " global dofs.\n";
        }
        
        for (const int &trialID : trialIDs)
        {
          if (trialID == tn_hat->ID()) continue; // skip tn; we covered it above
          auto oldTrialGIDs = idsForTrialVar[trialID];
          cout << vf->trial(trialID)->name() << " has " << oldTrialGIDs.size() << " global dofs.\n";
        }
        
        epsw_ids = idsForTrialVar[eps_w->ID()];
        tn_x_ids = idsForTrialVar[tn_x];
        tn_t_ids = idsForTrialVar[tn_t];
        tn_v_ids = idsForTrialVar[tn_v];
        f_ids    = idsForTrialVar[f->ID()];
      }
      always_checkpoint();

      // ---------------------------------------------
      // Block - structured matrices and vectors
      // ---------------------------------------------
      always_checkpoint();

      // subblocks is probably a better name than subvectors? also, the struct is specific to vlasov
      boba::Array<std::set<GlobalIndexType>, subvectors::size> partition;
      partition[subvectors::epsw] = epsw_ids;
      partition[subvectors::tnx] = tn_x_ids;
      partition[subvectors::tnt] = tn_t_ids;
      partition[subvectors::tnv] = tn_v_ids;
      partition[subvectors::f] = f_ids;

      auto block_system = setupBlockSystem<host_matrix, host_vector, subvectors::size, subvectors::size>(K, rhs, lhs, partition, partition);
      auto block_matrix = std::get<0>(block_system);
      auto block_rhs = std::get<1>(block_system);
      auto block_lhs = std::get<2>(block_system);

      for (size_t row = 0; row < subvectors::size; row++)
      {
        for (size_t col = 0; col < subvectors::size; col++)
        {
          block_matrix({row, col}).write_to_file("target_matrix_" + std::to_string(row) + "_" + std::to_string(col));
        }
      }

      if(true)
      {
        const auto prefix = std::string("run_") + std::to_string(i) + std::string("_");

        EpetraExt::RowMatrixToMatrixMarketFile((prefix + std::string("matrix.dat")).c_str(), *K, NULL, NULL, false);
        EpetraExt::MultiVectorToMatrixMarketFile((prefix + std::string("lhs.dat")).c_str(), *lhs, NULL, NULL, false);
        EpetraExt::MultiVectorToMatrixMarketFile((prefix + std::string("rhs.dat")).c_str(), *rhs, NULL, NULL, false);

        block_matrix.savetxt(prefix + "block_matrix");
        block_rhs.savetxt(prefix + "block_rhs");
        block_lhs.savetxt(prefix + "block_lhs");
      }

      always_checkpoint();
      auto residual_full = block_rhs - block_matrix * block_lhs;
      boba_print(residual_full.norm_frobenius());

      // ---------------------------------------------
      // Tensorized vectors
      // ---------------------------------------------
      // Determine tensorized dof ordering
      size_t numElements_x_st = numElements_x;
      size_t numElements_t_st = numElements_t;
      size_t numElements_v_st = numElements_v;
      size_t pField = p;
      size_t pTrace = p + 1;
      size_t pTest = p + dp + 1;

      vector_wrapper<host_vector_tensor, subvectors::size> block_tensor_lhs;
      vector_wrapper<host_vector_tensor, subvectors::size> block_tensor_rhs;
      block_tensor_lhs(subvectors::epsw).resize({pTest, pTest, pTest, numElements_x_st, numElements_t_st, numElements_v_st});
      block_tensor_lhs(subvectors::tnx).resize({1, pTrace, pTrace, numElements_x_st + 1, numElements_t_st, numElements_v_st});
      block_tensor_lhs(subvectors::tnt).resize({pTrace, 1, pTrace, numElements_x_st, numElements_t_st + 1, numElements_v_st});
      block_tensor_lhs(subvectors::tnv).resize({pTrace, pTrace, 1, numElements_x_st, numElements_t_st, numElements_v_st + 1});
      block_tensor_lhs(subvectors::f).resize({pField, pField, pField, numElements_x_st, numElements_t_st, numElements_v_st});

      // Use same ordering for rhs
      for(size_t row = 0; row < subvectors::size; row++)
      {
        block_tensor_rhs(row).resize(block_tensor_lhs(row).sizes());
      }

      // Copy lhs and rhs from block_lhs and block_rhs
      for(size_t row = 0; row < subvectors::size; row++)
      {
        boba_print(block_lhs(row).sizes());
        boba_print(block_tensor_lhs(row).sizes());
        block_tensor_lhs(row).reshape(block_lhs(row));

        boba_print(block_rhs(row).sizes());
        boba_print(block_tensor_rhs(row).sizes());
        block_tensor_rhs(row).reshape(block_rhs(row));
      }
      boba::Array<boba::Array<size_t, 6>, subvectors::size> unpermuted_sizes;
      for(size_t row = 0; row < subvectors::size; row++)
      {
        unpermuted_sizes[row] = block_tensor_lhs(row).sizes();
      }

#define PERMUTE 0
      // EXPERIMENTAL: permute the dof ordering
      // px, pv, pt, nx, nv, nt -> px, nx, pv, nv, pt, nt
#if PERMUTE
      for(size_t row = 0; row < subvectors::size; row++)
      {
        ::boba::permute(block_tensor_lhs(row), {0, 3, 1, 4, 2, 5});
        ::boba::permute(block_tensor_rhs(row), {0, 3, 1, 4, 2, 5});
      }
#endif
      // ---------------------------------------------
      // Compressed block vectors 
      // ---------------------------------------------
      vector_wrapper<host_tt, subvectors::size> block_tt_lhs;
      vector_wrapper<host_tt, subvectors::size> block_tt_rhs;

      // Compress lhs
      for(size_t row = 0; row < subvectors::size; row++)
      {
        block_tt_lhs(row).setup(block_tensor_lhs(row).sizes());
        block_tt_lhs(row).svd_tolerance_relative = 1.0e-03;
        block_tt_lhs(row).svd_tolerance_absolute = 1.0e-12;
        // Optional: see the 'spectra
        // boba::compute_tt_spectra(block_tensor_lhs(row)).print();
        block_tt_lhs(row).compress(block_tensor_lhs(row));
        std::cout << "block_tt_lhs[" << row << "] CR " << block_tt_lhs(row).compression_rate() << " rank " << block_tt_lhs(row).ranks_string() << std::endl;
      }

      // Compress rhs
      for(size_t row = 0; row < subvectors::size; row++)
      {
        block_tt_rhs(row).setup(block_tensor_rhs(row).sizes());
        block_tt_rhs(row).svd_tolerance_relative = 1.0e-03;
        block_tt_rhs(row).svd_tolerance_absolute = 1.0e-12;
        // Optional: see the 'spectra
        // boba::compute_tt_spectra(block_tensor_rhs(row)).print();
        block_tt_rhs(row).compress(block_tensor_rhs(row));
        std::cout << "block_tt_rhs[" << row << "] CR " << block_tt_rhs(row).compression_rate() << " rank " << block_tt_rhs(row).ranks_string() << std::endl;
      }

      // ---------------------------------------------
      // Compressed block matrices
      // ---------------------------------------------
      operator_wrapper<host_ttm, host_tt, subvectors::size, subvectors::size> block_tt_matrix;

      // Compress matrix
      // note we compress the smaller operators FIRST
      // this way we free more memory before we need to compress the largest operators
      for(size_t rowp = subvectors::size; rowp > 0; rowp--)
      {
        for(size_t colp = subvectors::size; colp > 0; colp--)
        {
          auto row = rowp - 1;
          auto col = colp - 1;
          if((row == 0) and (col == 0))
          {
            continue;
          }
          auto rows = block_tt_lhs(row).sizes();
          auto cols = block_tt_lhs(col).sizes();
          block_tt_matrix({row, col}).rows = rows;
          block_tt_matrix({row, col}).cols = cols;
          // We attempt a manual compression of the operator to precisely control memory
          boba::Array<size_t, 12> unpermuted_matrix_sizes = ::boba::concatenate(unpermuted_sizes[row], unpermuted_sizes[col]);
          boba::tensor<12, ::boba::execution_space::CPU, double, size_t> umpermuted_temp(unpermuted_matrix_sizes);
          umpermuted_temp.reshape(block_matrix({row, col}));
          // Free memory (uncomment this to study higher resolutions.. however the solver will not work)
          // block_matrix({row, col}).resize({0});
#if PERMUTE
          // Permute the operator (see above)
          ::boba::permute(umpermuted_temp, {0, 3, 1, 4, 2, 5, 6+0, 6+3, 6+1, 6+4, 6+2, 6+5});
#endif
          // Permute (row_0, row_1,... , col_0, col_1, ...) -> (row_0, col_0, row_1, col_1, ...)
          permute(umpermuted_temp, {0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11});
          // Reshape into a tensor (row_0 + ROWS_0*col_0, row_1 + ROWS_1*col_1, ...)
          auto sizes = rows*cols;
          ::boba::tensor<6, boba::execution_space::CPU, double, size_t> matrix_tensor(sizes);
          matrix_tensor.reshape(umpermuted_temp);
          // Free memory
          umpermuted_temp.resize({0});
          // Compress
          auto tt_of_matrices = ::boba::compress_to_tensor_train(matrix_tensor, 1.0e-10, 1.0e-14);
          // Free
          matrix_tensor.resize({0});
          // Shape into ttm
          for(size_t d = 0; d < 6; d++)
          {
            auto left_ranks = tt_of_matrices.get_ranks_left(d);
            auto right_ranks = tt_of_matrices.get_ranks_right(d);
            block_tt_matrix({row, col}).cores[d].resize({rows[d], cols[d], left_ranks, right_ranks});
            block_tt_matrix({row, col}).cores[d].reshape(tt_of_matrices.cores[d]);
          }
          // Complete!
          std::cout << "block_tt_matrix[" << row << "," << col << "] CR " << block_tt_matrix({row, col}).compression_rate() << " rank " << block_tt_matrix({row, col}).ranks_string() << std::endl;
        }
      }

      // Manual Hack for row == col == 0
      // We assume that all blocks are identical
      // so then M = \sum_r\sum_l A_0r x B_rl x C_l0 x I x I x I
      // So we then only need to compress the matrix corresponding to
      // N = \sum_r\sum_l A_0r x B_rl x C_l0
#if PERMUTE
      boba_warn("WArning! Permute has not be verified yet.");
#endif
      {
        size_t row = 0;
        size_t col = 0;
        auto rows = block_tt_lhs(row).sizes();
        auto cols = block_tt_lhs(col).sizes();
        block_tt_matrix({row, col}).rows = rows;
        block_tt_matrix({row, col}).cols = cols;
        // Extract a single block
        boba::Multiindexer<3, size_t> block_mider({pTest, pTest, pTest});
        auto block_size = block_mider.size();
        host_matrix single_block({block_size, block_size});
        auto single_block_view = single_block.view();
        auto full_view = block_matrix({row, col}).const_view();
        for(size_t r = 0; r < block_size; r++)
        {
          for(size_t c = 0; c < block_size; c++)
          {
            single_block_view({r, c}) = full_view({r, c});
          }
        }
        // Free memory
        // block_matrix({row, col}).resize({0});
        // Compress the block
        ::boba::tensor_train_matrix<3, boba::execution_space::CPU, double, size_t> single_ttm(block_mider.sizes(), block_mider.sizes());
        single_ttm.compress(single_block);
        // Shape into ttm
        for(size_t d = 0; d < 3; d++)
        {
#if PERMUTE
          auto whole_ttm_dim = 2*d;
#else
          auto whole_ttm_dim = d;
#endif
          block_tt_matrix({row, col}).cores[whole_ttm_dim] = single_ttm.cores[d];
        }
#if PERMUTE
        for(size_t d = 0; d < 3; d++)
        {
          auto whole_ttm_d = 2*d + 1;
          auto left_ranks = single_ttm.get_ranks_right(whole_ttm_d - 1);
          auto right_ranks = (d < 2) ? single_ttm.get_ranks_left(whole_ttm_d + 1) : 1;
          block_tt_matrix({row, col}).cores[whole_ttm_d].resize({unpermuted_sizes[0][3+d], unpermuted_sizes[0][3+d], left_ranks, right_ranks});
          auto core_view = block_tt_matrix({row, col}).cores[whole_ttm_d].view();
          for(size_t r = 0; r < block_size; r++)
          {
            for(size_t c = 0; c < block_size; c++)
            {
              for(size_t l = 0; l < left_ranks; l++)
              {
                for(size_t p = 0; p < right_ranks; p++)
                {
                  core_view({r, c, l, p}) = (r == c) ? 1.0 : 0.0;
                }
              }
            }
          }
        }
#else
        for(size_t d = 0; d < 3; d++)
        {
          block_tt_matrix({row, col}).cores[3+d].resize({unpermuted_sizes[0][3+d], unpermuted_sizes[0][3+d], 1, 1});
          ::boba::set_to_identity_core(block_tt_matrix({row, col}).cores[d]);
        }
#endif
        // Complete!
        std::cout << "block_tt_matrix[" << row << "," << col << "] CR " << block_tt_matrix({row, col}).compression_rate() << " rank " << block_tt_matrix({row, col}).ranks_string() << std::endl;
      }

      // ---------------------------------------------
      // Compression vs sparsity
      // ---------------------------------------------

      always_checkpoint();

      const auto compression_from_sparsity = static_cast<double>(K->NumGlobalRows()) * static_cast<double>(K->NumGlobalCols()) / static_cast<double>(K->NumGlobalNonzeros());
      std::cout << "sparse_matrix   CR " << compression_from_sparsity << std::endl;
      std::cout << "block_tt_matrix CR " << block_tt_matrix.compression_rate() << std::endl;

      boba_print("finished compression");
      //exit(0);

      // ---------------------------------------------
      // Verify
      // ---------------------------------------------
      // Sanity check
      always_checkpoint();
      auto residual_tt = block_tt_rhs - block_tt_matrix*block_tt_lhs;

      for(size_t row = 0; row < subvectors::size; row++)
      {
        host_vector temp({residual_full(row).size()});
        temp.reshape(residual_tt(row).unroll());
        boba_print(::boba::norm_difference_frobenius(residual_full(row), temp));
      }

      // ---------------------------------------------
      // Block system solve
      // ---------------------------------------------
      if(true)
      {
        size_t outer_iterations = 10;
        size_t inner_iterations = static_cast<size_t>(K->NumGlobalRows());
        double tolerance = 1.0e-08;

        ::boba::krylov<
          operator_wrapper<host_matrix, host_vector, subvectors::size, subvectors::size>,
          vector_wrapper<host_vector, subvectors::size>> block_solver(outer_iterations, inner_iterations, tolerance);
        block_solver.verbose = true;
        block_solver.method = krylov_method;

        block_solver.linear_operator = block_matrix;

        auto guess = block_lhs;
        guess.fill_with_zeros();

        auto solution = guess;

        always_checkpoint();
        block_solver.solve(block_rhs, guess, solution);

        always_checkpoint();
        auto residual_block = block_rhs - block_matrix * solution;
        boba_print(residual_block.norm_frobenius());
      }

      // ---------------------------------------------
      // Block system solve, tensorized matrices only
      // ---------------------------------------------

      {
        size_t outer_iterations = 20;
        size_t inner_iterations = 4;//static_cast<size_t>(K->NumGlobalRows());
        double tolerance = 1.0e-04;
        ::boba::krylov<
          operator_wrapper<host_ttm, host_vector, subvectors::size, subvectors::size>,
          vector_wrapper<host_vector, subvectors::size>> block_mixed_solver(outer_iterations, inner_iterations, tolerance);
        block_mixed_solver.verbose = true;
        block_mixed_solver.method = krylov_method;

        for(size_t r = 0; r < subvectors::size; r++)
        {
          for(size_t c = 0; c < subvectors::size; c++)
          {
            block_mixed_solver.linear_operator({r, c}) = block_tt_matrix({r, c});
          }
        }
        
        auto guess = block_lhs;
        guess.fill_with_zeros();

        auto solution = guess;

        always_checkpoint();
        auto initial_residual = block_rhs - block_matrix * solution;
        boba_print(initial_residual.norm_frobenius());

        always_checkpoint();
        boba_print("phase 1");
        block_mixed_solver.solve(block_rhs, guess, solution);

        // Restart solver
        boba_print("phase 2");
        outer_iterations = 20;
        inner_iterations = 6;
        tolerance = 1.0e-06;
        block_mixed_solver.setup(outer_iterations, inner_iterations, tolerance);
        guess = solution;
        block_mixed_solver.solve(block_rhs, guess, solution);

        boba_print("phase 3");
        outer_iterations = 20;
        inner_iterations = 8;
        tolerance = 1.0e-06;
        block_mixed_solver.setup(outer_iterations, inner_iterations, tolerance);
        guess = solution;
        block_mixed_solver.solve(block_rhs, guess, solution);

        boba_print("phase 4");
        outer_iterations = 20;
        inner_iterations = static_cast<size_t>(K->NumGlobalRows());
        tolerance = 1.0e-08;
        block_mixed_solver.setup(outer_iterations, inner_iterations, tolerance);
        guess = solution;
        block_mixed_solver.solve(block_rhs, guess, solution);

        always_checkpoint();
        auto residual_block = block_rhs - block_matrix * solution;
        boba_print(residual_block.norm_frobenius());
      }

      boba_error("Finished conventional solve");

      // ---------------------------------------------
      // TT Solver
      // ---------------------------------------------

      {
        size_t outer_iterations = 20;
        size_t inner_iterations = 4;// static_cast<size_t>(K->NumGlobalRows());
        double tolerance = 1.0e-04;

        boba_print("phase 1");
        ::boba::krylov<
          operator_wrapper<host_ttm, host_tt, subvectors::size, subvectors::size>,
          vector_wrapper<host_tt, subvectors::size>> block_tt_solver(outer_iterations, inner_iterations, tolerance);
        block_tt_solver.verbose = true;
        block_tt_solver.method = krylov_method;

        block_tt_solver.linear_operator = block_tt_matrix;

        auto guess = block_tt_lhs;
        guess.fill_with_zeros();

        auto solution = guess;

        always_checkpoint();
        block_tt_solver.solve(block_tt_rhs, guess, solution);

        always_checkpoint();
        auto residual_block_tt = block_tt_rhs - block_tt_matrix * solution;
        boba_print(residual_block_tt.norm_frobenius());

        // Restart solver
        boba_print("phase 2");
        outer_iterations = 20;
        inner_iterations = 6;
        tolerance = 1.0e-06;
        block_tt_solver.setup(outer_iterations, inner_iterations, tolerance);
        guess = solution;
        block_tt_solver.solve(block_tt_rhs, guess, solution);

        boba_print("phase 3");
        outer_iterations = 20;
        inner_iterations = 8;
        tolerance = 1.0e-06;
        block_tt_solver.setup(outer_iterations, inner_iterations, tolerance);
        guess = solution;
        block_tt_solver.solve(block_tt_rhs, guess, solution);

        // solution to this ? 
      }

      boba_print("finished tt example");
      exit(0);
    }
  }
  Camellia::finalize(false);
  
  return 0;
}
