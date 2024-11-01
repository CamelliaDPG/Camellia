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

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  Kokkos::initialize(argc, argv); // initialize Kokkos
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  {
    int spaceDim = 1;
    int velocityDim = 1;
    int numElements_x = 2;
    int numElements_t = 2;
    int numElements_v = 2;
    
    vector<vector<double>> domainDim(spaceDim+velocityDim+1,vector<double>{0.0,1.0}); // first index: spaceDim; second: 0/1 for x0, x1, etc.
    int polyOrder = 2;
    int pToAdd = spaceDim+velocityDim;
    
    std::string modeString = "single"; // single run; other options are "h" and "p" for h and p convergence tests
    std::string sortedMatrixPath = "";
    
    cmdp.setOption("xElements", &numElements_x );
    cmdp.setOption("tElements", &numElements_t );
    cmdp.setOption("vElements", &numElements_v );
    cmdp.setOption("polyOrder", &polyOrder );
    cmdp.setOption("deltaP", &pToAdd);
    cmdp.setOption("x0", &domainDim[0][0] );
    cmdp.setOption("x1", &domainDim[0][1] );
    cmdp.setOption("t0", &domainDim[1][0] );
    cmdp.setOption("t1", &domainDim[1][1] );
    cmdp.setOption("v0", &domainDim[2][0] );
    cmdp.setOption("v1", &domainDim[2][1] );
    cmdp.setOption("mode", &modeString, "single, h, or p");
    cmdp.setOption("varSortedMatrixOutputPath", &sortedMatrixPath);
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    
    std::vector<int> numElements_x_series, numElements_t_series, numElements_v_series;
    std::vector<int> polyOrder_series;
    if (modeString == "single")
    {
      numElements_x_series.push_back(numElements_x);
      numElements_t_series.push_back(numElements_t);
      numElements_v_series.push_back(numElements_v);
      polyOrder_series.push_back(polyOrder);
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
        polyOrder_series.push_back(polyOrder);
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
        numElements_t_series.push_back(numElements_v);
        polyOrder_series.push_back(polyOrder + i);
      }
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported mode");
    }
    
    VarFactoryPtr vf = VarFactory::varFactory();
    VarPtr tn_hat    = vf->fluxVar("tn_hat", HGRAD_DISC);
    VarPtr f         = vf->fieldVar("f", HGRAD_DISC);
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
    vlasovBF->addTerm( eps_w, w );
    
    std::map<int,int> polyOrderEnhancements;
    // nominally, E and phi are in L^2, and E_hat and phi_hat are traces of H^1.
    // so the natural thing is for the traces to be discretized with order 1 higher
    // and the test functions to be discretized with order 1 + pToAdd higher
    polyOrderEnhancements[tn_hat->ID()] = 1;
    polyOrderEnhancements[eps_w->ID()]  = 1 + pToAdd;
    MeshPartitionPolicyPtr nullPartitionPolicy = Teuchos::null;
    Epetra_CommPtr nullComm = Teuchos::null;
    const bool allowPruning = false;  // to keep inter-mesh geometry determinations from getting too complex, don't allow the mesh topology to be pruned to keep only MPI-local geometry.
    
    BCPtr  bc  = BC::bc();
    RHSPtr rhs = RHS::rhs();
    
    FunctionPtr x = Function::xn(1); // x^1
    FunctionPtr t = Function::yn(1); // t^1
    
    FunctionPtr f_exact = x + t + v;
    FunctionPtr sin_pi_x_over_2 = TrigFunctions<double>::sin(0.5 * M_PI * x);
    FunctionPtr cos_2_pi_t      = TrigFunctions<double>::cos(2.0 * M_PI * t);
    
    FunctionPtr rhs_exact = v + 1 + E;
    rhs->addTerm(rhs_exact * w);
    
    // impose boundary conditions on the potential along the whole (spatial) mesh boundary ( phi_hat is only defined on the spatial part of the boundary):
    FunctionPtr n = Function::normal();
    FunctionPtr tn_exact = v * f_exact * n->x() + f_exact * n->y() + f_exact * E * n->z();
    bc->addDirichlet(tn_hat, SpatialFilter::allSpace(), tn_exact);
    
    int numRuns = int(numElements_x_series.size());
    for (int i=0; i<numRuns; i++)
    {
      const int polyOrder_i     = polyOrder_series[i];
      const int numElements_x_i = numElements_x_series[i];
      const int numElements_t_i = numElements_t_series[i];
      const int numElements_v_i = numElements_v_series[i];
      std::cout << "Running with polyOrder " << polyOrder_i << " on " << numElements_x_i << " x " << numElements_t_i << " x " << numElements_v_i << " mesh.\n";

      //  static MeshPtr rectilinearMesh(TBFPtr<double> bf, vector<double> dimensions, vector<int> elementCounts,
//      int H1Order, int pToAddTest=-1, vector<double> x0 = vector<double>(),
//      map<int,int> trialOrderEnhancements = map<int,int>(),
//      map<int,int> testOrderEnhancements = map<int,int>(),
//      Epetra_CommPtr Comm = Teuchos::null,
//      bool allowMeshTopologyPruning = true);
      
//      MeshPtr mesh = MeshFactory::rectilinearMesh(vlasovBF,
//                                                  {domainDim[0][1]-domainDim[0][0],domainDim[1][1]-domainDim[1][0],domainDim[2][1]-domainDim[2][0]},
//                                                  {numElements_x_i,numElements_t_i,numElements_v_i}, polyOrder);
      
//      MeshPtr mesh = Teuchos::rcp( new Mesh (phaseSpaceMeshTopo, vlasovBF, polyOrder_i, 0, polyOrderEnhancements, polyOrderEnhancements, nullPartitionPolicy, nullComm, allowPruning) );

      
      auto spatialMeshTopo  = MeshFactory::intervalMeshTopology(domainDim[0][0], domainDim[0][1], numElements_x_i);
      auto spaceTimeMeshTopo = MeshFactory::extrudedMeshTopology(spatialMeshTopo, {domainDim[1][0]}, {domainDim[1][1]}, {numElements_t_i});
      auto phaseSpaceMeshTopo = MeshFactory::extrudedMeshTopology(spaceTimeMeshTopo, {domainDim[2][0]}, {domainDim[2][1]}, {numElements_v_i});
      
      MeshPtr mesh = Teuchos::rcp( new Mesh (phaseSpaceMeshTopo, vlasovBF, polyOrder_i, 0, polyOrderEnhancements, polyOrderEnhancements, nullPartitionPolicy, nullComm, allowPruning) );

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
      
//      static bool haveOutputted = false;
//      
//      if (!haveOutputted)
//      {
//        auto elemTypeSample = mesh->getElementType(0);
//        elemTypeSample->trialOrderPtr->print(std::cout);
//        haveOutputted = true;
//      }
      
      SolutionPtr soln = Solution::solution(vlasovBF, mesh, bc, rhs);
//      soln->setWriteMatrixToMatrixMarketFile(true, "/tmp/K.dat");
      
      soln->solve();
      
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
                    auto globalIDs = dofMapper->globalIndicesForSubcell(trialID, d, sc, false, false);
                    int tn_id = traceVariableForSideOrdinal[sc];
                    idsForTrialVar[tn_id].insert(globalIDs.begin(), globalIDs.end());
                  }
                }
                else
                {
                  auto globalIDs = dofMapper->globalIndicesForSubcell(trialID, d, sc, false, false);
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
//          Camellia::print("global IDs for trial", oldTrialGIDs);
          for (auto &oldID : oldTrialGIDs)
          {
            idToSortedID[oldID] = newGIDOffset++;
          }
        }
        
        Epetra_Map partitionMap = soln->getPartitionMap();
        
        int maxRowSize = 0; // will cause more mallocs during insertion into the CrsMatrix, but will minimize the amount of memory allocated now.
        Epetra_FECrsMatrix K_sorted(::Copy, partitionMap, maxRowSize);
        
//        for (int idOrdinal=0; idOrdinal<newGIDOffset; idOrdinal++)
//        {
//          cout << idOrdinal << " -> " << idToSortedID[idOrdinal] << endl;
//        }
        
        reorderMatrix(*K, idToSortedID, K_sorted);
        
        EpetraExt::RowMatrixToMatrixMarketFile(sortedMatrixPath.c_str(),K_sorted,NULL,NULL,false);
      }
      
      // compute L^2 error in f:
      FunctionPtr f_soln = Function::solution(f, soln);
      double l2_err = (f_soln - f_exact)->l2norm(mesh);
      
      double exact_l2_norm = f_exact->l2norm(mesh);
      
      std::cout << "absolute L^2 error: " << l2_err << std::endl;
      std::cout << "relative L^2 error: " << l2_err / exact_l2_norm << "\n\n";
    }
  }
  Camellia::finalize(false);
  
  return 0;
}
