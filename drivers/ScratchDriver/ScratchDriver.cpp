#include <iostream>

#include "Epetra_SerialComm.h"

#include "Epetra_Time.h"

#include "MeshFactory.h"

#include "BF.h"

#include "Solution.h"

#include "../DPGTests/TestSuite.h"

#include "GDAMinimumRule.h"

#include "SolutionExporter.h"

#include "doubleBasisConstruction.h"

#include "CamelliaCellTools.h"

// #include "omp.h"

void testSerialDenseMatrix() {
  int n = 5, m = 3;
  FieldContainer<double> A(n,m);
  
  double *firstEntry = (double *) &A[0]; // a bit dangerous: cast away the const.  Not dangerous if we're doing Copy, of course.
  Epetra_SerialDenseMatrix Amatrix(::View,firstEntry,m,m,n);
  
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      A(i,j) = i - j + 1;
    }
  }
  
  Amatrix.SetUseTranspose(true); // only affects multiply, not the () operator
  
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      double diff = abs(Amatrix(j,i) - A(i,j));
      if (diff > 1e-14) {
        cout << "SDM and FC differ in entry (" << i << "," << j << ")\n";
      }
    }
  }
  
}

/***
 
 NVR - Just a playground for me to try things without having to add a new driver to the cmake lists, etc.
 
 ***/

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

void trumanCrashingCode() {
  {
    // 2D tests
    CellTopoPtr quad_4 = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
    CellTopoPtr tri_3 = Teuchos::rcp( new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() ) );
    
    // let's draw a little house
    vector<double> v0 = makeVertex(-1,0);
    vector<double> v1 = makeVertex(1,0);
    vector<double> v2 = makeVertex(1,2);
    vector<double> v3 = makeVertex(-1,2);
    vector<double> v4 = makeVertex(0.0,3);
    
    vector< vector<double> > vertices;
    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
    vertices.push_back(v3);
    vertices.push_back(v4);
    
    vector<unsigned> quadVertexList;
    quadVertexList.push_back(0);
    quadVertexList.push_back(1);
    quadVertexList.push_back(2);
    quadVertexList.push_back(3);
    
    vector<unsigned> triVertexList;
    triVertexList.push_back(3);
    triVertexList.push_back(2);
    triVertexList.push_back(4);
    
    vector< vector<unsigned> > elementVertices;
    elementVertices.push_back(quadVertexList);
    elementVertices.push_back(triVertexList);
    
    vector< CellTopoPtr > cellTopos;
    cellTopos.push_back(quad_4);
    cellTopos.push_back(tri_3);
    MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );
    
    MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry) );
    
    FunctionPtr x2 = Function::xn(2);
    FunctionPtr y2 = Function::yn(2);
    FunctionPtr function = x2 + y2;
    FunctionPtr vect = Function::vectorize(x2, y2);
    FunctionPtr fbdr = Function::restrictToCellBoundary(function);
    vector<FunctionPtr> functions;
    functions.push_back(function);
    functions.push_back(vect);
    vector<string> functionNames;
    functionNames.push_back("function");
    functionNames.push_back("vect");
    vector<FunctionPtr> bdrfunctions;
    bdrfunctions.push_back(fbdr);
    bdrfunctions.push_back(fbdr);
    vector<string> bdrfunctionNames;
    bdrfunctionNames.push_back("bdr1");
    bdrfunctionNames.push_back("bdr2");
    
    map<int, int> cellIDToNum1DPts;
    cellIDToNum1DPts[1] = 4;
    
    ////////////////////   DECLARE VARIABLES   ///////////////////////
    // define test variables
    VarFactory varFactory;
    VarPtr tau = varFactory.testVar("tau", HDIV);
    VarPtr v = varFactory.testVar("v", HGRAD);
    
    // define trial variables
    VarPtr uhat = varFactory.traceVar("uhat");
    VarPtr fhat = varFactory.fluxVar("fhat");
    VarPtr u = varFactory.fieldVar("u");
    VarPtr sigma = varFactory.fieldVar("sigma", VECTOR_L2);
    
    ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
    BFPtr bf = Teuchos::rcp( new BF(varFactory) );
    // tau terms:
    bf->addTerm(sigma, tau);
    bf->addTerm(u, tau->div());
    bf->addTerm(-uhat, tau->dot_normal());
    
    // v terms:
    bf->addTerm( sigma, v->grad() );
    bf->addTerm( fhat, v);
    
    ////////////////////   BUILD MESH   ///////////////////////
    int H1Order = 1, pToAdd = 2;
    Teuchos::RCP<Mesh> mesh = Teuchos::rcp( new Mesh (meshTopology, bf, H1Order, pToAdd) );
  }
}

void testFieldContainerSum() {
  // mimic the creation of an integrated matrix (like a Gram matrix or a stiffness matrix)
  // idea is to get some timings so that I can examine speedup from threading, as well maybe as
  // intelligent iteration through the containers to improve cache locality.
  int fields = 1000;
  int points = 1000;
  FieldContainer<double> basisVals(fields,points);
  FieldContainer<double> cubWeights(points);
  FieldContainer<double> otherBasisVals(fields,points);
  FieldContainer<double> integrals(fields,fields);
  for (int i=0; i<fields; i++) {
    for (int j=0; j<fields; j++) {
      for (int ptOrdinal=0; ptOrdinal<points; ptOrdinal++) {
        integrals(i,j) += basisVals(i,ptOrdinal) * cubWeights(ptOrdinal) * otherBasisVals(j,ptOrdinal);
      }
    }
  }
}

int main(int argc, char *argv[]) {
  bool testMeshTopoMemory = false;
  
  bool tryMinRule = false;
  
  bool trySortInt = true;
  
  bool tryCellToolsMapToRefSubcell = false;
  
  bool printLinearBasisNodes = true;
  bool printQuadBasisNodes = false;
  
  testSerialDenseMatrix();
  
//  trumanCrashingCode();

  if (tryCellToolsMapToRefSubcell) {
    // want to confirm the order that things map from reference subcell: specifically, does
    // the map honor the permutation of the subcell?
    
    shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
    
    int numPoints = 3;
    int edgeDim = 1, faceDim = 2;
    FieldContainer<double> refEdgePoints(numPoints,edgeDim);
    
    refEdgePoints(0,0) = -1;
    refEdgePoints(1,0) =  0;
    refEdgePoints(2,0) =  1;
    
    FieldContainer<double> refQuadPoints(numPoints, faceDim);
    
    int edgeOrdinal = 3;
    
    CamelliaCellTools::mapToReferenceSubcell(refQuadPoints, refEdgePoints, edgeDim, edgeOrdinal, quad_4);
    
    cout << "ref. edge points for edge " << edgeOrdinal << ":\n" << refEdgePoints;
    cout << "mapped points in quad:\n" << refQuadPoints;
    // (it's fine, honoring the permutation as expected)
  }
  
  if (printLinearBasisNodes) {
    BasisPtr linearBasis = Camellia::intrepidQuadHGRAD(1);
    
    FieldContainer<double> refPoints(4,2);
    refPoints(0,0) = -1.0;
    refPoints(0,1) = -1.0;
    
    refPoints(1,0) =  1.0;
    refPoints(1,1) = -1.0;
    
    refPoints(2,0) =  1.0;
    refPoints(2,1) =  1.0;
    
    refPoints(3,0) = -1.0;
    refPoints(3,1) =  1.0;
    
    FieldContainer<double> values(linearBasis->getCardinality(), 4);
    linearBasis->getValues(values, refPoints, OPERATOR_VALUE);
    double tol = 1e-14;
    for (int dofOrdinal=0; dofOrdinal<linearBasis->getCardinality(); dofOrdinal++) {
      for (int ptIndex=0; ptIndex < values.dimension(1); ptIndex++) {
        if (abs(values(dofOrdinal,ptIndex)-1.0) < tol) {
          double x = refPoints(ptIndex,0);
          double y = refPoints(ptIndex,1);
          cout << "dofOrdinal " << dofOrdinal << " has node at point (" << x << "," << y << ")\n";
        }
      }
    }
  }
  
  if (printQuadBasisNodes) {
    BasisPtr quadBasis = Camellia::intrepidQuadHGRAD(2);
    
    FieldContainer<double> refPoints(9,2);
    int ptIndex = 0;
    for (int i=0; i<3; i++) {
      for (int j=0; j<3; j++) {
        refPoints(ptIndex,0) = i-1.0;
        refPoints(ptIndex,1) = j-1.0;
        ptIndex++;
      }
    }
    
    FieldContainer<double> values(quadBasis->getCardinality(), 9);
    quadBasis->getValues(values, refPoints, OPERATOR_VALUE);
    double tol = 1e-14;
    for (int dofOrdinal=0; dofOrdinal<quadBasis->getCardinality(); dofOrdinal++) {
      for (int ptIndex=0; ptIndex < values.dimension(1); ptIndex++) {
        if (abs(values(dofOrdinal,ptIndex)-1.0) < tol) {
          double x = refPoints(ptIndex,0);
          double y = refPoints(ptIndex,1);
          cout << "dofOrdinal " << dofOrdinal << " has node at point (" << x << "," << y << ")\n";
        }
      }
    }
  }

  if (trySortInt) {
    int nvals = 48;
    int *procs_from_ = new int[nvals];
    int *lengths_from_ = new int[nvals];
    procs_from_[0] = 21; procs_from_[1] = 23; procs_from_[2] = 29; procs_from_[3] = 17; procs_from_[4] = 20; procs_from_[5] = 22; procs_from_[6] = 19; procs_from_[7] = 28; procs_from_[8] = 16; procs_from_[9] = 25; procs_from_[10] = 18; procs_from_[11] = 30; procs_from_[12] = 27; procs_from_[13] = 24; procs_from_[14] = 26; procs_from_[15] = 5; procs_from_[16] = 7; procs_from_[17] = 4; procs_from_[18] = 13; procs_from_[19] = 1; procs_from_[20] = 15; procs_from_[21] = 3; procs_from_[22] = 6; procs_from_[23] = 12; procs_from_[24] = 2; procs_from_[25] = 0; procs_from_[26] = 11; procs_from_[27] = 9; procs_from_[28] = 14; procs_from_[29] = 8; procs_from_[30] = 10; procs_from_[31] = 43; procs_from_[32] = 40; procs_from_[33] = 42; procs_from_[34] = 34; procs_from_[35] = 41; procs_from_[36] = 46; procs_from_[37] = 35; procs_from_[38] = 38; procs_from_[39] = 44; procs_from_[40] = 39; procs_from_[41] = 32; procs_from_[42] = 45; procs_from_[43] = 33; procs_from_[44] = 36; procs_from_[45] = 37; procs_from_[46] = 47; procs_from_[47] = 31;
    lengths_from_[0] = 683; lengths_from_[1] = 683; lengths_from_[2] = 683; lengths_from_[3] = 683; lengths_from_[4] = 683; lengths_from_[5] = 683; lengths_from_[6] = 683; lengths_from_[7] = 683; lengths_from_[8] = 683; lengths_from_[9] = 683; lengths_from_[10] = 683; lengths_from_[11] = 518; lengths_from_[12] = 683; lengths_from_[13] = 683; lengths_from_[14] = 683; lengths_from_[15] = 683; lengths_from_[16] = 683; lengths_from_[17] = 683; lengths_from_[18] = 683; lengths_from_[19] = 683; lengths_from_[20] = 683; lengths_from_[21] = 683; lengths_from_[22] = 683; lengths_from_[23] = 683; lengths_from_[24] = 683; lengths_from_[25] = 683; lengths_from_[26] = 683; lengths_from_[27] = 683; lengths_from_[28] = 683; lengths_from_[29] = 683; lengths_from_[30] = 683; lengths_from_[31] = 683; lengths_from_[32] = 683; lengths_from_[33] = 683; lengths_from_[34] = 683; lengths_from_[35] = 683; lengths_from_[36] = 683; lengths_from_[37] = 683; lengths_from_[38] = 683; lengths_from_[39] = 683; lengths_from_[40] = 683; lengths_from_[41] = 683; lengths_from_[42] = 683; lengths_from_[43] = 683; lengths_from_[44] = 683; lengths_from_[45] = 683; lengths_from_[46] = 683; lengths_from_[47] = 165;

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
