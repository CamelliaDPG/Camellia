//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "SolutionExporter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

double pi = 2.0*acos(0.0);

int H1Order = 3, pToAdd = 2;

class ConstantXBoundary : public SpatialFilter {
   private:
      double xval;
   public:
      ConstantXBoundary(double xval): xval(xval) {};
      bool matchesPoint(double x, double y) {
         double tol = 1e-14;
         return (abs(x-xval) < tol);
      }
};

class ConstantYBoundary : public SpatialFilter {
   private:
      double yval;
   public:
      ConstantYBoundary(double yval): yval(yval) {};
      bool matchesPoint(double x, double y) {
         double tol = 1e-14;
         return (abs(y-yval) < tol);
      }
};

class DiscontinuousInitialCondition : public Function {
   private:
      double xloc;
      double valL;
      double valR;
   public:
      DiscontinuousInitialCondition(double xloc, double valL, double valR) : Function(0), xloc(xloc), valL(valL), valR(valR) {}
      void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
         int numCells = values.dimension(0);
         int numPoints = values.dimension(1);

         const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
         for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
               double x = (*points)(cellIndex,ptIndex,0);
               if (x <= xloc)
                  values(cellIndex, ptIndex) = valL;
               else
                  values(cellIndex, ptIndex) = valR;
            }
         }
      }
};

class RampedInitialCondition : public Function {
   private:
      double xloc;
      double valL;
      double valR;
      double h;
   public:
      RampedInitialCondition(double xloc, double valL, double valR, double h) : Function(0), xloc(xloc), valL(valL), valR(valR), h(h) {}
      void setH(double h_) { h = h_; }
      void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
         int numCells = values.dimension(0);
         int numPoints = values.dimension(1);

         const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
         for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++) 
            {
               double x = (*points)(cellIndex,ptIndex,0);
               if (x <= xloc-h/2.)
                  values(cellIndex, ptIndex) = valL;
               else if (x >= xloc+h/2.)
                  values(cellIndex, ptIndex) = valR;
               else
                  values(cellIndex, ptIndex) = valL+(valR-valL)/h*(x-xloc+h/2);
            }
         }
      }
};

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args( argc, argv );
#endif
  int commRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

   // Required arguments

   // Optional arguments (have defaults)
  int numRefs = args.Input("--numRefs", "number of refinement steps", 0);
  int norm = args.Input("--norm", "norm", 0);
  double mu = args.Input("--mu", "viscosity", 1e-2);
  int polyOrder = args.Input("--polyOrder", "polynomial order for field variables", 2);
  int deltaP = args.Input("--deltaP", "how much to enrich test space", 2);
  int xCells = args.Input("--xCells", "number of cells in the x direction", 8);
  int tCells = args.Input("--tCells", "number of cells in the t direction", 4);
  int maxNewtonIterations = args.Input("--maxIterations", "maximum number of Newton iterations", 20);
  double nlTol = args.Input("--nlTol", "nonlinear tolerance", 1e-6);
  int numPreRefs = args.Input<int>("--numPreRefs","pre-refinements on singularity",0);

  args.Process();

   ////////////////////   PROBLEM DEFINITIONS   ///////////////////////
  int H1Order = polyOrder+1;

  double xmin, xmax, xint, tmax;
  double uL, uR;
  string problemName;

  // Strong shock tube
  problemName = "ConfusionShock";
  xmin = 0;
  xmax = 5;
  xint = 2.5;
  tmax = 4e-1;

  uL = 10;
  uR = 1;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("tau", HGRAD);
  VarPtr v = varFactory.testVar("v", HGRAD);

  // define trial variables
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma = varFactory.fieldVar("sigma", L2);
  VarPtr uhat = varFactory.spatialTraceVar("uhat");
  VarPtr fhat = varFactory.fluxVar("fhat");

  ////////////////////   INITIALIZE USEFUL VARIABLES   ///////////////////////
  // Define useful functions
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr zero = Function::zero();

  // Initialize useful variables
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );

  ////////////////////   BUILD MESH   ///////////////////////
  FieldContainer<double> meshBoundary(4,2);

  meshBoundary(0,0) =  xmin; // x1
  meshBoundary(0,1) =  0; // y1
  meshBoundary(1,0) =  xmax;
  meshBoundary(1,1) =  0;
  meshBoundary(2,0) =  xmax;
  meshBoundary(2,1) =  tmax;
  meshBoundary(3,0) =  xmin;
  meshBoundary(3,1) =  tmax;

  MeshPtr mesh = Mesh::buildQuadMesh(meshBoundary, xCells, tCells,
      bf, H1Order, H1Order+pToAdd, false);

  ////////////////////   SET INITIAL CONDITIONS   ///////////////////////
  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );

  map<int, Teuchos::RCP<Function> > initialGuess;
  // initialGuess[u->ID() ]   = Teuchos::rcp( new DiscontinuousInitialCondition(xint, uL, uR) );
  initialGuess[u->ID()]   = Teuchos::rcp( new RampedInitialCondition(xint, uL, uR,     (xmax-xmin)/xCells/pow(2.,numPreRefs)) );

  backgroundFlow->projectOntoMesh(initialGuess);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  // Set up problem
  FunctionPtr u_prev   = Function::solution(u, backgroundFlow);
  FunctionPtr sigma_prev   = Function::solution(sigma, backgroundFlow);

  // tau terms:
  bf->addTerm( sigma/mu, tau );
  bf->addTerm( u, tau->dx() );
  bf->addTerm( -uhat, tau->times_normal_x() );

  // v terms:
  bf->addTerm( -u, v->dx() );
  bf->addTerm( sigma, v->dx() );
  bf->addTerm( -u, v->dy() );
  bf->addTerm( fhat, v);

  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );

  // tau terms:
  rhs->addTerm( -sigma_prev/mu * tau );
  rhs->addTerm( -u_prev * tau->dx() );

  // v terms:
  rhs->addTerm( u_prev * v->dx() );
  rhs->addTerm( -sigma_prev * v->dx() );
  rhs->addTerm( u_prev * v->dy() );


  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = Teuchos::rcp(new IP);
  switch (norm)
  {
    // Automatic graph norm
    case 0:
    ip = bf->graphNorm();
    break;

    // Manual Graph norm
    case 1:
    // ip->addTerm(vc);
    // ip->addTerm(vm);
    // ip->addTerm(ve);
    break;

    // Decoupled Eulerian and viscous norm
    // Might need to also elimnate D_prev term...
    case 2:
    break;

    default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid problem number");
  }

  ////////////////////   CREATE BCs   ///////////////////////
  SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(xmin) );
  SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(xmax) );
  SpatialFilterPtr init = Teuchos::rcp( new ConstantYBoundary(0) );
  FunctionPtr u0  = Teuchos::rcp( new DiscontinuousInitialCondition(xint, uL, uR) );
  // FunctionPtr rho0  = Teuchos::rcp( new RampedInitialCondition(xint, rhoL, rhoR, (xmax-xmin)/xCells/pow(2.,numPreRefs)) );
  bc->addDirichlet(fhat, init, -u0);
  bc->addDirichlet(fhat, left, -uL*one);
  bc->addDirichlet(fhat, right, uR*one);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  mesh->registerSolution(backgroundFlow);
  mesh->registerSolution(solution);
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  VTKExporter exporter(backgroundFlow, mesh, varFactory);
  set<int> nonlinearVars;
  nonlinearVars.insert(u->ID());
  nonlinearVars.insert(sigma->ID());

  if (commRank==0){
    cout << "Number of pre-refinements = " << numPreRefs << endl;
  }
  for (int i =0;i<=numPreRefs;i++){
    vector<ElementPtr> elems = mesh->activeElements();
    vector<ElementPtr>::iterator elemIt;
    vector<int> pointCells;
    for (elemIt=elems.begin();elemIt != elems.end();elemIt++){
      int cellID = (*elemIt)->cellID();
      int numSides = mesh->getElement(cellID)->numSides();
      FieldContainer<double> vertices(numSides,2); //for quads

      mesh->verticesForCell(vertices, cellID);
      bool cellIDset = false;
      for (int j = 0;j<numSides;j++){ // num sides = 4
        if ((abs(vertices(j,0)-xint)<1e-7) && (abs(vertices(j,1))<1e-7) && !cellIDset)
        {
          pointCells.push_back(cellID);
          cellIDset = true;
        }
      }
    }
    if (i<numPreRefs){
      refinementStrategy.refineCells(pointCells);
    }
  }

  for (int refIndex=0; refIndex<=numRefs; refIndex++)
  {
    double L2Update = 1e7;
    int iterCount = 0;
    while (L2Update > nlTol && iterCount < maxNewtonIterations)
    {
      solution->condensedSolve();
      double uL2Update = solution->L2NormOfSolutionGlobal(u->ID());
      double sigmaL2Update = solution->L2NormOfSolutionGlobal(sigma->ID());
      L2Update = sqrt(uL2Update*uL2Update + sigmaL2Update*sigmaL2Update);

      // line search algorithm
      double alpha = 1.0;
      bool useLineSearch = true;
      // amount of enriching of grid points on which to ensure positivity
      int posEnrich = 5; 
      if (useLineSearch)
      {
        double lineSearchFactor = .5; 
        double eps = .001;
        FunctionPtr uTemp = Function::solution(u,backgroundFlow) + alpha*Function::solution(u,solution) - Function::constant(eps);
        bool uIsPositive = uTemp->isPositive(mesh,posEnrich);
        int iter = 0; int maxIter = 20;
        while (!uIsPositive && iter < maxIter)
        {
          alpha = alpha*lineSearchFactor;
          uTemp = Function::solution(u,backgroundFlow) + alpha*Function::solution(u,solution);
          uIsPositive = uTemp->isPositive(mesh,posEnrich);
          iter++;
        }
        if (commRank==0 && alpha < 1.0){
          cout << "line search factor alpha = " << alpha << endl;
        }
      }

      backgroundFlow->addSolution(solution, alpha, nonlinearVars);
      iterCount++;
      if (commRank == 0)
        cout << "L2 Norm of Update = " << L2Update << endl;
      if (alpha < 1e-6)
        break;
    }
    if (commRank == 0)
      cout << endl;

    if (commRank == 0)
    {
      stringstream outfile;
      outfile << problemName << norm << "_" << refIndex;
      exporter.exportSolution(outfile.str());
    }

    if (refIndex < numRefs)
    {
      refinementStrategy.refine(commRank==0);
      double newRamp = (xmax-xmin)/(xCells*pow(2., numPreRefs+refIndex+1));
      // if (commRank == 0)
      //   cout << "New ramp width = " << newRamp << endl;
      // dynamic_cast< RampedInitialCondition* >(initialGuess[rho->ID()].get())->setH(newRamp);
      // dynamic_cast< RampedInitialCondition* >(initialGuess[u->ID()].get())->setH(newRamp);
      // dynamic_cast< RampedInitialCondition* >(initialGuess[T->ID()].get())->setH(newRamp);
      // dynamic_cast< RampedInitialCondition* >(rho0.get())->setH(newRamp);
      // dynamic_cast< RampedInitialCondition* >(mom0.get())->setH(newRamp);
      // dynamic_cast< RampedInitialCondition* >(E0.get())->setH(newRamp);
      // backgroundFlow->projectOntoMesh(initialGuess);
    }
  }

  return 0;
}

