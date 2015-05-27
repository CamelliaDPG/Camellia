//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "PreviousSolutionFunction.h"
#include "RefinementStrategy.h"
#include "RefinementHistory.h"
#include "SolutionExporter.h"
#include "MeshFactory.h"
#include "CheckConservation.h"
#include "LagrangeConstraints.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

class ConstantXBoundary : public SpatialFilter
{
private:
  double xval;
public:
  ConstantXBoundary(double xval): xval(xval) {};
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(x-xval) < tol);
  }
};

class ConstantYBoundary : public SpatialFilter
{
private:
  double yval;
public:
  ConstantYBoundary(double yval): yval(yval) {};
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(y-yval) < tol);
  }
};

class BottomStepBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(y-0.5) < tol && x <= 2);
  }
};

class RightStepBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(x-2) < tol && y <= 0.5);
  }
};

class InflowU1 : public Function
{
public:
  InflowU1() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex, ptIndex) = 8*(y-0.5)*(1-y);
      }
    }
  }
};

class OutflowU1 : public Function
{
public:
  OutflowU1() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex, ptIndex) = y*(1-y);
      }
    }
  }
};

int main(int argc, char *argv[])
{
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args( argc, argv );
#endif
  int commRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

  // Required arguments
  int numRefs = args.Input<int>("--numRefs", "number of refinement steps");
  int norm = args.Input<int>("--norm", "0 = graph\n    1 = robust\n    2 = coupled robust");

  // Optional arguments (have defaults)
  bool enforceLocalConservation = args.Input<bool>("--conserve", "enforce local conservation", false);
  int Re = args.Input("--Re", "Reynolds number", 1);
  int maxNewtonIterations = args.Input("--maxIterations", "maximum number of Newton iterations", 1);
  int polyOrder = args.Input("--polyOrder", "polynomial order for field variables", 2);
  int deltaP = args.Input("--deltaP", "how much to enrich test space", 2);
  string saveFile = args.Input<string>("--meshSaveFile", "file to which to save refinement history", "");
  string replayFile = args.Input<string>("--meshLoadFile", "file with refinement history to replay", "");
  args.Process();

  if (commRank==0)
  {
    cout << "saveFile is " << saveFile << endl;
    cout << "loadFile is " << replayFile << endl;
  }

  ////////////////////   PROBLEM DEFINITIONS   ///////////////////////
  int H1Order = polyOrder+1;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau1 = varFactory.testVar("tau1", HDIV);
  VarPtr tau2 = varFactory.testVar("tau2", HDIV);
  VarPtr v1 = varFactory.testVar("v1", HGRAD);
  VarPtr v2 = varFactory.testVar("v2", HGRAD);
  VarPtr vc = varFactory.testVar("vc", HGRAD);

  // define trial variables
  VarPtr u1 = varFactory.fieldVar("u1");
  VarPtr u2 = varFactory.fieldVar("u2");
  VarPtr p = varFactory.fieldVar("p");
  VarPtr u1hat = varFactory.traceVar("u1hat");
  VarPtr u2hat = varFactory.traceVar("u2hat");
  VarPtr t1hat = varFactory.fluxVar("t1hat");
  VarPtr t2hat = varFactory.fluxVar("t2hat");
  VarPtr sigma1 = varFactory.fieldVar("sigma1", VECTOR_L2);
  VarPtr sigma2 = varFactory.fieldVar("sigma2", VECTOR_L2);

  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );

  // define nodes for mesh
  vector< FieldContainer<double> > vertices;
  FieldContainer<double> pt(2);
  vector< vector<int> > elementIndices;
  vector<int> e(4);

  for (int c=0; c < 4; c++)
  {
    double xpos = c*0.5;
    pt(0) = xpos;
    pt(1) = 0.5;
    vertices.push_back(pt);
    pt(0) = xpos;
    pt(1) = 1.0;
    vertices.push_back(pt);
  }
  for (int c=4; c <= 20; c++)
  {
    double xpos = c*0.5;
    pt(0) = xpos;
    pt(1) = 0.0;
    vertices.push_back(pt);
    pt(0) = xpos;
    pt(1) = 0.5;
    vertices.push_back(pt);
    pt(0) = xpos;
    pt(1) = 1.0;
    vertices.push_back(pt);
  }

  // Columns 0-2
  for (int c=0; c < 3; c++)
  {
    e[0] = 2*c;
    e[1] = 2*(c+1);
    e[2] = 2*(c+1)+1;
    e[3] = 2*c+1;
    elementIndices.push_back(e);
  }
  // Column 3
  e[0] = 6;
  e[1] = 9;
  e[2] = 10;
  e[3] = 7;
  elementIndices.push_back(e);
  // Columns 4-19
  for (int c=0; c < 16; c++)
  {
    e[0] = 3*c+8;
    e[1] = 3*(c+1)+8;
    e[2] = 3*(c+1)+9;
    e[3] = 3*c+9;
    elementIndices.push_back(e);
    // Columns 4-19
    e[0] = 3*c+9;
    e[1] = 3*(c+1)+9;
    e[2] = 3*(c+1)+10;
    e[3] = 3*c+10;
    elementIndices.push_back(e);
  }

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, bf, H1Order, deltaP) );

  ////////////////////////////////////////////////////////////////////
  // INITIALIZE BACKGROUND FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );

  vector<double> e1(2); // (1,0)
  e1[0] = 1;
  vector<double> e2(2); // (0,1)
  e2[1] = 1;

  FunctionPtr u1_prev = Function::solution(u1, backgroundFlow);
  FunctionPtr u2_prev = Function::solution(u2, backgroundFlow);
  FunctionPtr sigma1_prev = Function::solution(sigma1, backgroundFlow);
  FunctionPtr sigma2_prev = Function::solution(sigma2, backgroundFlow);

  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  FunctionPtr beta = e1 * u1_prev + e2 * u2_prev;

  // ==================== SET INITIAL GUESS ==========================
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[u1->ID()] = one;
  functionMap[u2->ID()] = zero;
  functionMap[sigma1->ID()] = Function::vectorize(zero,zero);
  functionMap[sigma2->ID()] = Function::vectorize(zero,zero);
  functionMap[p->ID()] = zero;

  backgroundFlow->projectOntoMesh(functionMap);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  // // stress equation
  bf->addTerm( sigma1, tau1 );
  bf->addTerm( sigma2, tau2 );
  bf->addTerm( u1, tau1->div() );
  bf->addTerm( u2, tau2->div() );
  bf->addTerm( -u1hat, tau1->dot_normal() );
  bf->addTerm( -u2hat, tau2->dot_normal() );

  // momentum equation
  // bf->addTerm( Function::xPart(sigma1_prev)*u1, v1 );
  // bf->addTerm( Function::yPart(sigma1_prev)*u2, v1 );
  // bf->addTerm( Function::xPart(sigma2_prev)*u1, v2 );
  // bf->addTerm( Function::yPart(sigma2_prev)*u2, v2 );
  // bf->addTerm( beta*sigma1, v1);
  // bf->addTerm( beta*sigma2, v2);
  bf->addTerm( 1./Re*sigma1, v1->grad() );
  bf->addTerm( 1./Re*sigma2, v2->grad() );
  bf->addTerm( t1hat, v1);
  bf->addTerm( t2hat, v2);
  bf->addTerm( -p, v1->dx() );
  bf->addTerm( -p, v2->dy() );

  // continuity equation
  bf->addTerm( -u1, vc->dx() );
  bf->addTerm( -u2, vc->dy() );
  bf->addTerm( u1hat, vc->times_normal_x() );
  bf->addTerm( u2hat, vc->times_normal_y() );

  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );

  // stress equation
  rhs->addTerm( -sigma1_prev * tau1 );
  rhs->addTerm( -sigma2_prev * tau2 );
  rhs->addTerm( -u1_prev * tau1->div() );
  rhs->addTerm( -u2_prev * tau2->div() );

  // momentum equation
  // rhs->addTerm( -beta*sigma1_prev * v1 );
  // rhs->addTerm( -beta*sigma2_prev * v2 );
  rhs->addTerm( -1./Re*sigma1_prev * v1->grad() );
  rhs->addTerm( -1./Re*sigma2_prev * v2->grad() );

  // continuity equation
  rhs->addTerm( u1_prev * vc->dx() );
  rhs->addTerm( u2_prev * vc->dy() );

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = Teuchos::rcp(new IP);
  if (norm == 0)
  {
    ip = bf->graphNorm();
  }
  else if (norm == 1)
  {
    // ip = bf->l2Norm();
  }

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(0) );
  SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(10) );
  SpatialFilterPtr top = Teuchos::rcp( new ConstantYBoundary(1) );
  SpatialFilterPtr bottomLeft = Teuchos::rcp( new BottomStepBoundary );
  SpatialFilterPtr step = Teuchos::rcp( new RightStepBoundary );
  SpatialFilterPtr bottomRight = Teuchos::rcp( new ConstantYBoundary(0) );
  FunctionPtr inflowU1 = Teuchos::rcp( new InflowU1 );
  FunctionPtr outflowU1 = Teuchos::rcp( new OutflowU1 );
  bc->addDirichlet(u1hat, left, inflowU1);
  bc->addDirichlet(u2hat, left, zero);
  bc->addDirichlet(u1hat, right, outflowU1);
  bc->addDirichlet(u2hat, right, zero);
  bc->addDirichlet(u1hat, top, zero);
  bc->addDirichlet(u2hat, top, zero);
  bc->addDirichlet(u1hat, bottomLeft, zero);
  bc->addDirichlet(u2hat, bottomLeft, zero);
  bc->addDirichlet(u1hat, step, zero);
  bc->addDirichlet(u2hat, step, zero);
  bc->addDirichlet(u1hat, bottomRight, zero);
  bc->addDirichlet(u2hat, bottomRight, zero);

  // zero mean constraint on pressure
  bc->addZeroMeanConstraint(p);

  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

  if (enforceLocalConservation)
  {
    solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y() == zero);
  }

  // ==================== Register Solutions ==========================
  mesh->registerSolution(solution);
  mesh->registerSolution(backgroundFlow);

  Teuchos::RCP< RefinementHistory > refHistory = Teuchos::rcp( new RefinementHistory );
  mesh->registerObserver(refHistory);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  VTKExporter exporter(backgroundFlow, mesh, varFactory);
  ofstream errOut;
  ofstream fluxOut;
  if (commRank == 0)
  {
    errOut.open("stokesstep_err.txt");
    fluxOut.open("stokesstep_flux.txt");
  }
  errOut.precision(15);
  fluxOut.precision(15);

  // Cell IDs for flux calculations
  vector< vector< pair<ElementPtr, int> > > cellFaceSets;
  vector< pair<ElementPtr, int> > cellFaces;
  cellFaces.push_back(make_pair(mesh->getElement(0), 3));
  cellFaceSets.push_back(cellFaces);
  for (int c = 0; c < 4; c++)
  {
    vector< pair<ElementPtr, int> > cellFaces;
    cellFaces.push_back(make_pair(mesh->getElement(c), 1));
    cellFaceSets.push_back(cellFaces);
  }
  for (int c = 0; c < 16; c++)
  {
    vector< pair<ElementPtr, int> > cellFaces;
    cellFaces.push_back(make_pair(mesh->getElement(2*c+4), 1));
    cellFaces.push_back(make_pair(mesh->getElement(2*c+5), 1));
    cellFaceSets.push_back(cellFaces);
  }

  // for loading refinement history
  if (replayFile.length() > 0)
  {
    RefinementHistory refHistory;
    replayFile = replayFile;
    refHistory.loadFromFile(replayFile);
    refHistory.playback(mesh);
    int numElems = mesh->numActiveElements();
    if (commRank==0)
    {
      cout << "after replay, num elems = " << numElems << endl;
    }
  }

  double nonlinearRelativeEnergyTolerance = 1e-5; // used to determine convergence of the nonlinear solution
  for (int refIndex=0; refIndex<=numRefs; refIndex++)
  {
    double L2Update = 1e10;
    int iterCount = 0;
    while (L2Update > nonlinearRelativeEnergyTolerance && iterCount < maxNewtonIterations)
    {
      solution->solve(false);
      double u1L2Update = solution->L2NormOfSolutionGlobal(u1->ID());
      double u2L2Update = solution->L2NormOfSolutionGlobal(u2->ID());
      L2Update = sqrt(u1L2Update*u1L2Update + u2L2Update*u2L2Update);
      double energy_error = solution->energyErrorTotal();

      // Check local conservation
      if (commRank == 0)
      {
        FunctionPtr n = Function::normal();
        FunctionPtr u1_prev = Function::solution(u1hat, solution);
        FunctionPtr u2_prev = Function::solution(u2hat, solution);
        FunctionPtr flux = u1_prev*n->x() + u2_prev*n->y();
        Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, zero, mesh);
        cout << "Mass flux: Largest Local = " << fluxImbalances[0]
             << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;

        errOut << mesh->numGlobalDofs() << " " << energy_error << " "
               << fluxImbalances[0] << " " << fluxImbalances[1] << " " << fluxImbalances[2] << endl;

        cout << "Total mass flux = ";
        for (int i = 0; i < cellFaceSets.size(); i++)
        {
          double massFlux = computeFluxOverElementSides(u1_prev, mesh, cellFaceSets[i]);
          cout << massFlux << " ";
          fluxOut << massFlux << " ";
        }
        cout << endl;
        fluxOut << endl;

        if (saveFile.length() > 0)
        {
          std::ostringstream oss;
          oss << string(saveFile) << refIndex ;
          cout << "on refinement " << refIndex << " saving mesh file to " << oss.str() << endl;
          refHistory->saveToFile(oss.str());
        }
      }

      // line search algorithm
      double alpha = 1.0;
      // bool useLineSearch = false;
      // int posEnrich = 5; // amount of enriching of grid points on which to ensure positivity
      // if (useLineSearch){ // to enforce positivity of density rho
      //   double lineSearchFactor = .5; double eps = .001; // arbitrary
      //   FunctionPtr rhoTemp = Function::solution(rho,backgroundFlow) + alpha*Function::solution(rho,solution) - Function::constant(eps);
      //   FunctionPtr eTemp = Function::solution(e,backgroundFlow) + alpha*Function::solution(e,solution) - Function::constant(eps);
      //   bool rhoIsPositive = rhoTemp->isPositive(mesh,posEnrich);
      //   bool eIsPositive = eTemp->isPositive(mesh,posEnrich);
      //   int iter = 0; int maxIter = 20;
      //   while (!(rhoIsPositive && eIsPositive) && iter < maxIter){
      //     alpha = alpha*lineSearchFactor;
      //     rhoTemp = Function::solution(rho,backgroundFlow) + alpha*Function::solution(rho,solution);
      //     eTemp = Function::solution(e,backgroundFlow) + alpha*Function::solution(e,solution);
      //     rhoIsPositive = rhoTemp->isPositive(mesh,posEnrich);
      //     eIsPositive = eTemp->isPositive(mesh,posEnrich);
      //     iter++;
      //   }
      //   if (commRank==0 && alpha < 1.0){
      //     cout << "line search factor alpha = " << alpha << endl;
      //   }
      // }

      backgroundFlow->addSolution(solution, alpha, false, true);
      iterCount++;
      // if (commRank == 0)
      //   cout << "L2 Norm of Update = " << L2Update << endl;
    }
    if (commRank == 0)
      cout << endl;

    if (commRank == 0)
    {
      stringstream outfile;
      outfile << "stokesstep" << "_" << refIndex;
      exporter.exportSolution(outfile.str());
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
  }
  if (commRank == 0)
  {
    errOut.close();
    fluxOut.close();
  }

  return 0;
}

