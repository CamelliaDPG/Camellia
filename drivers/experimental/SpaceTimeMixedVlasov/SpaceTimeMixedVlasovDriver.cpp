//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia.h"
#include "E_to_T.hpp"
#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "LocalDofMapper.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"

#include <MatrixMarket_Tpetra.hpp>

using namespace Camellia;
using namespace std;

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
    int meshDim = spaceDim + 1 + velocityDim;
    double beta = 1.0;
    bool pMultigrid = false;
    
    vector<vector<double>> domainDim(meshDim,vector<double>{0.0,1.0}); // first index: spaceDim; second: 0/1 for x0, x1, etc.
    int p = 2;
    int dp = spaceDim+velocityDim;
    
    std::string modeString = "single"; // single run; other options are "h" and "p" for h and p convergence tests
    std::string sortedMatrixPath = "";
    
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
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
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
    
    auto bubnovVF = vf->getBubnovFactory(Camellia::VarFactory::BUBNOV_TRIAL);
    
    auto vlasovBF = BF::bf(bubnovVF);
    
    static const int DX_DIM = 1;
    static const int DT_DIM = 2;
    static const int DV_DIM = 3;
    
    FunctionPtr E = Function::constant(1.0);
    FunctionPtr v = Function::zn(1); // v^1
    
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
    
    MeshPartitionPolicyPtr nullPartitionPolicy = Teuchos::null;
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
        std::cout << "Running with p=" << p_i << " (dp=" << dp_i << ") on " << numElements_x_i << " x " << numElements_t_i << " x " << numElements_v_i << " mesh.\n";
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

      MeshPtr mesh = Teuchos::rcp( new Mesh (phaseSpaceMeshTopo, vlasovBF, p_i, 0, polyOrderEnhancements, polyOrderEnhancements, nullPartitionPolicy, Teuchos::null, allowPruning) );

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
      
      SolutionPtr soln = Solution::solution(vlasovBF, mesh, bc, rhs);
      
      if (pMultigrid)
      {
        const double gmresTol = 1e-6;
        const int maxIters = 10000;
        auto solver = GMGSolver::gmgSolver(soln, false, gmresTol, maxIters);
//        solver->setSmootherType(GMGOperator::POINT_JACOBI);
        solver->setSmootherType(GMGOperator::CHEBYSHEV);
        solver->setAztecOutput(100);
        soln->solve(solver);
      }
      else
      {
        soln->solve();
      }
      
      auto K = soln->getStiffnessMatrix();
      
      if (sortedMatrixPath != "")
      {
        // please don't emulate the code below; it's just a quick and dirty way to get a reordered Epetra matrix
        
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
        
        int newGIDOffset = 0;
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
          for (auto &oldID : oldTrialGIDs)
          {
            idToSortedID[oldID] = newGIDOffset++;
          }
        }
        
        for (const int &trialID : trialIDs)
        {
          if (trialID == tn_hat->ID()) continue; // skip tn; we covered it above
          auto oldTrialGIDs = idsForTrialVar[trialID];
          cout << vf->trial(trialID)->name() << " has " << oldTrialGIDs.size() << " global dofs.\n";
          for (auto &oldID : oldTrialGIDs)
          {
            idToSortedID[oldID] = newGIDOffset++;
          }
        }
        
        Epetra_Map partitionMap = soln->getPartitionMap();
        
        int maxRowSize = 0; // will cause more mallocs during insertion into the CrsMatrix, but will minimize the amount of memory allocated now.
        Epetra_FECrsMatrix K_sorted(::Copy, partitionMap, maxRowSize);
                
        reorderMatrix(*K, idToSortedID, K_sorted);
        
        auto K_sorted_ptr = Teuchos::rcp(&K_sorted, false);
        Tpetra::MatrixMarket::Writer<TMatrix<double> >::writeSparseFile(sortedMatrixPath, E_to_T::matrix(K_sorted_ptr));
      }
      
      // compute L^2 error in f:
      int quadratureEnrichment = 8;
      FunctionPtr f_soln = Function::solution(f, soln, quadratureEnrichment);
      double l2_err = (f_soln - f_exact)->l2norm(mesh);
      
      double exact_l2_norm = f_exact->l2norm(mesh, quadratureEnrichment);
      double f_l2_norm     =  f_soln->l2norm(mesh, quadratureEnrichment);
      
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
        MPIWrapper::allGatherVariable(mesh->Comm(), gramRowsVector, myGramRows, offsets);
        
        std::set<int> gramRows(gramRowsVector.begin(),gramRowsVector.end()); // deduplicate with set
        
        auto gramMatrixAndEpsW = subMatrixAndSubVector(*K, *soln->getLHSVector(), gramRows);
        
        auto & G = gramMatrixAndEpsW.first;
        auto & epsw_h = gramMatrixAndEpsW.second;
        
        Epetra_Vector G_epsw(epsw_h); // copy constructor
        
        G.Apply(epsw_h, G_epsw);
        
        epsw_h.Dot(G_epsw, &energyError);
        energyError = sqrt(energyError);
      }
      
      if (rank == 0)
      {
        std::cout << "absolute L^2 error: " << l2_err << std::endl;
        std::cout << "relative L^2 error: " << l2_err / exact_l2_norm << "\n\n";
        
        std::cout << "beta in graph norm:      " << beta << std::endl;
        std::cout << "energy error:            " << energyError << std::endl;
        std::cout << "energy norm effectivity: " << energyError / l2_err << std::endl << std::endl;
        
        std::cout << "exact L^2 norm: " << exact_l2_norm << std::endl;
        std::cout << "L^2 norm of soln: " << f_l2_norm << std::endl;
      }
    }
  }
  Camellia::finalize(false);
  
  return 0;
}
