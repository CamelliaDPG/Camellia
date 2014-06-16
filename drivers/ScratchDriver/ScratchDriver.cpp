//#include "MeshTopology.h"

#include <iostream>

#include "Epetra_SerialComm.h"

#include "Epetra_Time.h"

#include "MeshFactory.h"

#include "BF.h"

#include "Solution.h"

#include "../DPGTests/TestSuite.h"

#include "GDAMinimumRule.h"

#include "SolutionExporter.h"

vector<double> makeVertex(double v0) {
  vector<double> v;
  v.push_back(v0);
  return v;
}

vector<double> makeVertex(double v0, double v1) {
  vector<double> v;
  v.push_back(v0);
  v.push_back(v1);
  return v;
}

vector<double> makeVertex(double v0, double v1, double v2) {
  vector<double> v;
  v.push_back(v0);
  v.push_back(v1);
  v.push_back(v2);
  return v;
}

vector< vector<double> > hexPoints(double x0, double y0, double z0, double width, double height, double depth) {
  vector< vector<double> > v(8);
  v[0] = makeVertex(x0,y0,z0);
  v[1] = makeVertex(x0 + width,y0,z0);
  v[2] = makeVertex(x0 + width,y0 + height,z0);
  v[3] = makeVertex(x0,y0 + height,z0);
  v[4] = makeVertex(x0,y0,z0+depth);
  v[5] = makeVertex(x0 + width,y0,z0 + depth);
  v[6] = makeVertex(x0 + width,y0 + height,z0 + depth);
  v[7] = makeVertex(x0,y0 + height,z0 + depth);
  return v;
}

MeshTopologyPtr makeHexMesh(double x0, double y0, double z0, double width, double height, double depth,
                                       unsigned horizontalCells, unsigned verticalCells, unsigned depthCells) {
  unsigned spaceDim = 3;
  Teuchos::RCP<MeshTopology> mesh = Teuchos::rcp( new MeshTopology(spaceDim) );
  double dx = width / horizontalCells;
  double dy = height / verticalCells;
  double dz = depth / depthCells;
  CellTopoPtr hexTopo = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ) );
  for (unsigned i=0; i<horizontalCells; i++) {
    double x = x0 + dx * i;
    for (unsigned j=0; j<verticalCells; j++) {
      double y = y0 + dy * j;
      for (unsigned k=0; k<depthCells; k++) {
        double z = z0 + dz * k;
        vector< vector<double> > vertices = hexPoints(x, y, z, dx, dy, dz);
        mesh->addCell(hexTopo, vertices);
      }
    }
  }
  return mesh;
}

void refineUniformly(MeshTopologyPtr mesh) {
  set<unsigned> cellIndices = mesh->getActiveCellIndices();
  for (set<unsigned>::iterator cellIt = cellIndices.begin(); cellIt != cellIndices.end(); cellIt++) {
    mesh->refineCell(*cellIt, RefinementPattern::regularRefinementPatternHexahedron());
  }
}

// boundary value for u
class U0 : public Function {
public:
  U0() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        // solution with a boundary layer (section 5.2 in DPG Part II)
        // for x = 1, y = 1: u = 0
        if ( ( abs(x-1.0) < tol ) || (abs(y-1.0) < tol ) ) {
          values(cellIndex,ptIndex) = 0;
        } else if ( abs(x) < tol ) { // for x=0: u = 1 - y
          values(cellIndex,ptIndex) = 1.0 - y;
        } else { // for y=0: u=1-x
          values(cellIndex,ptIndex) = 1.0 - x;
        }
        
      }
    }
  }
};

int Sort_ints_( int *vals_sort,     //  values to be sorted
                int *vals_other,    // other array to be reordered with sort
                int  nvals)         // length of these two arrays
{
  // It is primarily used to sort messages to improve communication flow.
  // This routine will also insure that the ordering produced by the invert_map
  // routines is deterministic.  This should make bugs more reproducible.  This
  // is accomplished by sorting the message lists by processor ID.
  // This is a distribution count sort algorithm (see Knuth)
  //  This version assumes non negative integers.
  
  if (nvals <= 1) return 0;
  
  int i;                        // loop counter
  
  // find largest int, n, to size sorting array, then allocate and clear it
  int n = 0;
  for (i = 0; i < nvals; i++)
    if (n < vals_sort[i]) n = vals_sort[i];
  int *pos = new int [n+2];
  for (i = 0; i < n+2; i++) pos[i] = 0;
  
  // copy input arrays into temporary copies to allow sorting original arrays
  int *copy_sort  = new int [nvals];
  int *copy_other = new int [nvals];
  for (i = 0; i < nvals; i++)
  {
    copy_sort[i]  = vals_sort[i];
    copy_other[i] = vals_other[i];
  }
  
  // count the occurances of integers ("distribution count")
  int *p = pos+1;
  for (i = 0; i < nvals; i++) p[copy_sort[i]]++;
  
  // create the partial sum of distribution counts
  for (i = 1; i < n; i++) p[i] += p[i-1];
  
  // the shifted partitial sum is the index to store the data  in sort order
  p = pos;
  for (i = 0; i < nvals; i++)
  {
    vals_sort  [p[copy_sort [i]]]   = copy_sort[i];
    vals_other [p[copy_sort [i]]++] = copy_other[i];
  }
  
  delete [] copy_sort;
  delete [] copy_other;
  delete [] pos; 
  
  return 0;
}

int main(int argc, char *argv[]) {
  bool testMeshTopoMemory = false;
  
  bool tryMinRule = false;
  
  bool trySortInt = true;
  
  if (trySortInt) {
    int nvals = 3;
    int *procs_from_ = new int[nvals];
    int *lengths_from_ = new int[nvals];
    procs_from_[0] = 17;
    procs_from_[1] = 16;
    procs_from_[2] = 18;
    lengths_from_[0] = 88;
    lengths_from_[1] = 42;
    lengths_from_[2] = 15;
    Sort_ints_(procs_from_, lengths_from_, nvals);
    cout << "After sorting";
    for (int i=0; i<nvals; i++) {
      cout << ", procs_from["<< i << "] = " << procs_from_[i];
    }
    for (int i=0; i<nvals; i++) {
      cout << ", lengths_from["<< i << "] = " << lengths_from_[i];
    }
    cout << endl;
    delete [] procs_from_;
    delete [] lengths_from_;
  }
  
  if (tryMinRule) {
    double eps = 1e-4;
    int horizontalCells = 1, verticalCells = 2;
    int numRefs = 1;
    
    vector<double> beta_const;
    beta_const.push_back(2.0);
    beta_const.push_back(1.0);
    
    ////////////////////   DECLARE VARIABLES   ///////////////////////
    // define test variables
    VarFactory varFactory;
    VarPtr tau = varFactory.testVar("\\tau", HDIV);
    VarPtr v = varFactory.testVar("v", HGRAD);
    
    // define trial variables
    VarPtr uhat = varFactory.traceVar("\\widehat{u}");
    VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
    VarPtr u = varFactory.fieldVar("u");

    VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
    VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");
    
    ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
    BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
    // tau terms:
    confusionBF->addTerm(sigma1 / eps, tau->x());
    confusionBF->addTerm(sigma2 / eps, tau->y());
    confusionBF->addTerm(u, tau->div());
    confusionBF->addTerm(-uhat, tau->dot_normal());
    
    // v terms:
    confusionBF->addTerm( sigma1, v->dx() );
    confusionBF->addTerm( sigma2, v->dy() );
    confusionBF->addTerm( beta_const * u, - v->grad() );
    confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
    
    int polyOrder = 2;
    int pToAddTest = 2;
    double width = 1.0, height = 1.0;
    MeshPtr meshMinRule = MeshFactory::quadMeshMinRule(confusionBF, polyOrder, pToAddTest, width, height,
                                                       horizontalCells, verticalCells);
    
//    cout << "Before refinements, edge constraints:\n";
//    meshMinRule->getTopology()->printConstraintReport(1); // edges
    
    for (int ref=0; ref<numRefs; ref++) {
      set<GlobalIndexType> cellIDsMinRule = meshMinRule->getActiveCellIDs();
      meshMinRule->hRefine(cellIDsMinRule, RefinementPattern::regularRefinementPatternQuad());
    }
//    cout << "After refinements, edge constraints:\n";
//    meshMinRule->getTopology()->printConstraintReport(1); // edges
    
    ////////////////////   SPECIFY RHS   ///////////////////////
    RHSPtr rhs = RHS::rhs();
    FunctionPtr f = Function::zero();
    rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!
    
    ////////////////////   CREATE BCs   ///////////////////////
    BCPtr bc = BC::bc();
    FunctionPtr u0 = Teuchos::rcp( new U0 );
    bc->addDirichlet(uhat, SpatialFilter::allSpace(), u0);
    
    IPPtr ip = confusionBF->graphNorm();

//    FieldContainer<double> maxRuleSolnCoefficients;
    
//    int cellID = 0;
    
    
    { // max rule for comparison
      MeshPtr meshMaxRule = MeshFactory::quadMesh(confusionBF, polyOrder, pToAddTest, width, height,
                                                  horizontalCells, verticalCells);

      for (int ref=0; ref<numRefs; ref++) {
        set<GlobalIndexType> cellIDsMaxRule = meshMaxRule->getActiveCellIDs();
        meshMaxRule->hRefine(cellIDsMaxRule, RefinementPattern::regularRefinementPatternQuad());
      }
      
      SolutionPtr solnMaxRule = Teuchos::rcp( new Solution(meshMaxRule, bc, rhs, ip) );
      string maxStiffnessFileName = "maxRuleStiffness.dat";
      solnMaxRule->setWriteMatrixToMatrixMarketFile(true, maxStiffnessFileName);
      string maxLoadFileName = "maxRuleLoad.dat";
      solnMaxRule->setWriteRHSToMatrixMarketFile(true, maxLoadFileName);
      cout << "Will write max. rule stiffness to file " << maxStiffnessFileName << endl;
      cout << "Will write max. rule load to file " << maxLoadFileName << endl;

      solnMaxRule->solve();
//      maxRuleSolnCoefficients = solnMaxRule->allCoefficientsForCellID(cellID);
      
#ifdef USE_VTK
      VTKExporter maxExporter(solnMaxRule,meshMaxRule, varFactory);
      maxExporter.exportSolution("confusionMaxRuleSoln");
#endif
    }
    
//    set<IndexType> activeCells = meshMinRule->getActiveCellIDs();
//    GDAMinimumRule* minRule = (GDAMinimumRule*) meshMinRule->globalDofAssignment().get();
//    for (set<IndexType>::iterator cellIDIt = activeCells.begin(); cellIDIt != activeCells.end(); cellIDIt++) {
//      IndexType cellID = *cellIDIt;
//      minRule->printConstraintInfo(cellID);
//    }
    
    SolutionPtr solnMinRule = Teuchos::rcp( new Solution(meshMinRule, bc, rhs, ip) );
    
    cout << "soln constructed; about to solve.\n";

    string minStiffnessFileName = "minRuleStiffness.dat";
    solnMinRule->setWriteMatrixToMatrixMarketFile(true, minStiffnessFileName);
    string minLoadFileName = "minRuleLoad.dat";
    solnMinRule->setWriteRHSToMatrixMarketFile(true, minLoadFileName);
    cout << "Will write min. rule stiffness to file " << minStiffnessFileName << endl;
    cout << "Will write max. rule load to file " << minLoadFileName << endl;
    
    solnMinRule->solve();
    
    cout << "...solved.\n";
    
#ifdef USE_VTK
    VTKExporter minExporter(solnMinRule,meshMinRule, varFactory);
    minExporter.exportSolution("confusionMinRuleSoln");
#endif
    
//    FieldContainer<double> minRuleSolnCoefficients = solnMinRule->allCoefficientsForCellID(cellID);
//    
//    double tol=1e-14;
//    double maxDiff;
//    if (TestSuite::fcsAgree(minRuleSolnCoefficients, maxRuleSolnCoefficients, tol, maxDiff)) {
//      cout << "solution coefficients for max and min rule AGREE; max difference is " << maxDiff << endl;
//    } else {
//      cout << "solution coefficients for max and min rule DISAGREE; max difference is " << maxDiff << endl;
//    }
  }
  
  if (testMeshTopoMemory) {
    Epetra_SerialComm Comm;
    
    int nx = 100, ny = 100, nz = 10;
    
    cout << "creating " << nx * ny * nz << "-element mesh...\n";
    
    Epetra_Time timer(Comm);
    
    MeshTopologyPtr meshTopo = makeHexMesh(0, 0, 0, 1, 1, 1, nx, ny, nz);
    
    double timeMeshCreation = timer.ElapsedTime();

    cout << "...created.  Elapsed time " << timeMeshCreation << " seconds; pausing now to allow memory usage examination.  Enter a number to continue.\n";
    int n;
    cin >> n;
    
    int numRefs = 6;
    cout << "Creating mesh for " << numRefs << " uniform refinements.\n";
    timer.ResetStartTime();
    meshTopo = makeHexMesh(0, 0, 0, 1, 1, 1, 1, 1, 1);
    
    for (int ref=0; ref<numRefs; ref++) {
      refineUniformly(meshTopo);
    }
    
    double timeMeshRefinements  = timer.ElapsedTime();
    
    cout << "Completed refinements in " << timeMeshRefinements << " seconds.  Final mesh has " << meshTopo->activeCellCount() << " active cells, and " << meshTopo->cellCount() << " cells total.\n";
    
    cout << "Paused to allow memory usage examination.  Enter a number to exit.\n";
    
    cin >> n;
  }
  return 0;
}
