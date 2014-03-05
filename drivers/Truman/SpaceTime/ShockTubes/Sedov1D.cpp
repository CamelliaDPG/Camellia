//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "SolutionExporter.h"
#include "PreviousSolutionFunction.h"

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

class PulsedInitialCondition : public Function {
   private:
      double halfWidth;
      double intI;
      double valO;
   public:
      PulsedInitialCondition(double halfWidth, double intI, double valO) : Function(0), halfWidth(halfWidth), intI(intI), valO(valO) {}
      void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
         int numCells = values.dimension(0);
         int numPoints = values.dimension(1);

         const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
         for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
               double x = (*points)(cellIndex,ptIndex,0);
               // halfWidth = 0.25;
               if (abs(x) <= 4*halfWidth)
                  values(cellIndex, ptIndex) = 0.5*intI/halfWidth*(1-abs(x/(4*halfWidth)));
               else
                  values(cellIndex, ptIndex) = valO;
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
  int problem = args.Input<int>("--dimension", "1D, Cylindrical, or Spherical", 1);
  int numRefs = args.Input("--numRefs", "number of refinement steps", 0);
  int norm = args.Input("--norm", "norm", 0);
  double mu = args.Input("--mu", "viscosity", 1e-2);
  int numSlabs = args.Input("--numSlabs", "number of time slabs", 1);
  bool useLineSearch = args.Input("--lineSearch", "use line search", true);
  int polyOrder = args.Input("--polyOrder", "polynomial order for field variables", 2);
  int deltaP = args.Input("--deltaP", "how much to enrich test space", 2);
  int numX = args.Input("--numX", "number of cells in the x direction", 4);
  int numT = args.Input("--numT", "number of cells in the t direction", 4);
  int numPreRefs = args.Input<int>("--numPreRefs","pre-refinements on singularity",0);
  int maxNewtonIterations = args.Input("--maxIterations", "maximum number of Newton iterations", 20);
  double nlTol = args.Input("--nlTol", "nonlinear tolerance", 1e-6);

  args.Process();

   ////////////////////   PROBLEM DEFINITIONS   ///////////////////////
  int H1Order = polyOrder+1;

  double xmin, xmax, xint, tmin, tmax;
  double rhoL, rhoR, uL, uR, pL, pR, eL, eR, TL, TR;
  double M_inf = 1;
  double gamma = 1.4;
  double Cv = 1/(gamma*(gamma-1)*M_inf*M_inf);
  double Cp = gamma*Cv;
  double R = Cp-Cv;
  double Pr = 0.713;

  string problemName = "Sedov";
  xmin = -.5;
  xmax = .5;
  xint = 0;
  tmax = .2;

  rhoL = 1;
  rhoR = 1;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr S = varFactory.testVar("S", HGRAD);
  VarPtr tau = varFactory.testVar("tau", HGRAD);
  VarPtr vc = varFactory.testVar("vc", HGRAD);
  VarPtr vm = varFactory.testVar("vm", HGRAD);
  VarPtr ve = varFactory.testVar("ve", HGRAD);

  // define trial variables
  VarPtr D = varFactory.fieldVar("D");
  VarPtr q = varFactory.fieldVar("q");
  VarPtr rho = varFactory.fieldVar("rho");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr T = varFactory.fieldVar("T");
  VarPtr uhat = varFactory.spatialTraceVar("uhat");
  VarPtr That = varFactory.spatialTraceVar("That");
  VarPtr Fc = varFactory.fluxVar("Fc");
  VarPtr Fm = varFactory.fluxVar("Fm");
  VarPtr Fe = varFactory.fluxVar("Fe");

  ////////////////////   INITIALIZE USEFUL VARIABLES   ///////////////////////
  // Define useful functions
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr zero = Function::zero();

  // Initialize useful variables
  vector< BFPtr > bfs;
  vector< Teuchos::RCP<RHSEasy> > rhss;
  vector< IPPtr > ips;
  for (int slab=0; slab < numSlabs; slab++)
  {
    BFPtr bf = Teuchos::rcp( new BF(varFactory) );
    Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
    IPPtr ip = Teuchos::rcp(new IP);
    bfs.push_back(bf);
    rhss.push_back(rhs);
    ips.push_back(ip);
  }

  ////////////////////   BUILD MESH   ///////////////////////
  vector< Teuchos::RCP<Mesh> > meshes;
  vector<double> tmins;
  vector<double> tmaxs;
  for (int slab=0; slab < numSlabs; slab++)
  {
    // define nodes for mesh
    FieldContainer<double> meshBoundary(4,2);
    double t1 = tmax*double(slab)/numSlabs;
    double t2 = tmax*double(slab+1)/numSlabs;

    meshBoundary(0,0) =  xmin; // x1
    meshBoundary(0,1) =  t1; // y1
    meshBoundary(1,0) =  xmax;
    meshBoundary(1,1) =  t1;
    meshBoundary(2,0) =  xmax;
    meshBoundary(2,1) =  t2;
    meshBoundary(3,0) =  xmin;
    meshBoundary(3,1) =  t2;

    // create a pointer to a new mesh:
    Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, numX, numT,
      bfs[slab], H1Order, H1Order+pToAdd);
    tmins.push_back(t1);
    tmaxs.push_back(t2);
    if (slab == 0)
    {
      LinearTermPtr nullLT = Teuchos::rcp((LinearTerm*)NULL);
      IPPtr nullIP = Teuchos::rcp((IP*)NULL);
      RefinementStrategy refinementStrategy(mesh, nullLT, nullIP, 0.2);
      if (commRank==0)
      {
        cout << "Number of pre-refinements = " << numPreRefs << endl;
      }
      for (int i =0;i<=numPreRefs;i++)
      {
        vector<ElementPtr> elems = mesh->activeElements();
        vector<ElementPtr>::iterator elemIt;
        vector<int> pointCells;
        for (elemIt=elems.begin();elemIt != elems.end();elemIt++)
        {
          int cellID = (*elemIt)->cellID();
          int numSides = mesh->getElement(cellID)->numSides();
          FieldContainer<double> vertices(numSides,2); //for quads

          mesh->verticesForCell(vertices, cellID);
          bool cellIDset = false;
          for (int j = 0;j<numSides;j++)
          {
            if ((abs(vertices(j,0))<1e-7) && (abs(vertices(j,1))<1e-7) && !cellIDset)
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
    }
    meshes.push_back(mesh);
  }

  ////////////////////   SET INITIAL CONDITIONS   ///////////////////////
  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  map<int, Teuchos::RCP<Function> > initialGuess;
  // initialGuess[rho->ID()] = Teuchos::rcp( new DiscontinuousInitialCondition(xint, rhoL, rhoR) ) ;
  // initialGuess[u->ID()]   = Teuchos::rcp( new DiscontinuousInitialCondition(xint, uL, uR) );
  // initialGuess[T->ID()]   = Teuchos::rcp( new DiscontinuousInitialCondition(xint, TL, TR) );
  initialGuess[rho->ID()] = rhoR*one;
  initialGuess[u->ID()]   = zero;
  initialGuess[T->ID()]   = zero;

  vector< SolutionPtr > backgroundFlows;
  for (int slab=0; slab < numSlabs; slab++)
  {
    SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(meshes[slab], nullBC, nullRHS, nullIP) );
    backgroundFlow->projectOntoMesh(initialGuess);
    backgroundFlows.push_back(backgroundFlow);
  }

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  for (int slab=0; slab < numSlabs; slab++)
  {
    BFPtr bf = bfs[slab];
    Teuchos::RCP<RHSEasy> rhs = rhss[slab];
    IPPtr ip = ips[slab];
    FunctionPtr rho_prev = Function::solution(rho, backgroundFlows[slab]);
    FunctionPtr u_prev   = Function::solution(u, backgroundFlows[slab]);
    FunctionPtr T_prev   = Function::solution(T, backgroundFlows[slab]);
    FunctionPtr D_prev   = Function::solution(D, backgroundFlows[slab]);

    // S terms:
    bf->addTerm( D/mu, S);
    bf->addTerm( 4./3*u, S->dx());
    bf->addTerm( -4./3*uhat, S->times_normal_x());

    // tau terms:
    bf->addTerm( Pr/(mu*Cp)*q, tau);
    bf->addTerm( -T, tau->dx());
    bf->addTerm( That, tau->times_normal_x());

    // vc terms:
    bf->addTerm( -rho_prev*u, vc->dx());
    bf->addTerm( -u_prev*rho, vc->dx());
    bf->addTerm( -rho, vc->dy());
    bf->addTerm( Fc, vc);

    // vm terms:
    bf->addTerm( -rho_prev*u_prev*u, vm->dx());
    bf->addTerm( -rho_prev*u_prev*u, vm->dx());
    bf->addTerm( -u_prev*u_prev*rho, vm->dx());
    bf->addTerm( -R*rho_prev*T, vm->dx());
    bf->addTerm( -R*T_prev*rho, vm->dx());
    bf->addTerm( D, vm->dx());
    bf->addTerm( -rho_prev*u, vm->dy());
    bf->addTerm( -u_prev*rho, vm->dy());
    bf->addTerm( Fm, vm);

    // ve terms:
    bf->addTerm( -Cv*rho_prev*T_prev*u, ve->dx());
    bf->addTerm( -Cv*u_prev*rho_prev*T, ve->dx());
    bf->addTerm( -Cv*T_prev*u_prev*rho, ve->dx());
    bf->addTerm( -0.5*rho_prev*u_prev*u_prev*u, ve->dx());
    bf->addTerm( -0.5*rho_prev*u_prev*u_prev*u, ve->dx());
    bf->addTerm( -0.5*rho_prev*u_prev*u_prev*u, ve->dx());
    bf->addTerm( -0.5*u_prev*u_prev*u_prev*rho, ve->dx());
    bf->addTerm( -R*rho_prev*T_prev*u, ve->dx());
    bf->addTerm( -R*rho_prev*u_prev*T, ve->dx());
    bf->addTerm( -R*u_prev*T_prev*rho, ve->dx());
    bf->addTerm( -q, ve->dx());
    bf->addTerm( u_prev*D, ve->dx());
    bf->addTerm( D_prev*u, ve->dx());
    bf->addTerm( -Cv*rho_prev*T, ve->dy());
    bf->addTerm( -Cv*T_prev*rho, ve->dy());
    bf->addTerm( -0.5*rho_prev*u_prev*u, ve->dy());
    bf->addTerm( -0.5*rho_prev*u_prev*u, ve->dy());
    bf->addTerm( -0.5*u_prev*u_prev*rho, ve->dy());
    bf->addTerm( Fe, ve);

    ////////////////////   SPECIFY RHS   ///////////////////////

    // S terms:
    rhs->addTerm( -1./mu*D_prev * S );
    rhs->addTerm( -4./3*u_prev * S->dx() );

    // tau terms:
    rhs->addTerm( T_prev * tau->dx() );

    // vc terms:
    rhs->addTerm( rho_prev*u_prev * vc->dx() );
    rhs->addTerm( rho_prev * vc->dy() );

    // vc terms:
    rhs->addTerm( rho_prev*u_prev*u_prev * vm->dx() );
    rhs->addTerm( R*rho_prev*T_prev * vm->dx() );
    rhs->addTerm( -D_prev * vm->dx() );
    rhs->addTerm( rho_prev*u_prev * vm->dy() );

    // ve terms:
    rhs->addTerm( Cv*rho_prev*u_prev*T_prev * ve->dx() );
    rhs->addTerm( 0.5*rho_prev*u_prev*u_prev*u_prev * ve->dx() );
    rhs->addTerm( R*rho_prev*u_prev*T_prev * ve->dx() );
    rhs->addTerm( -u_prev*D_prev * ve->dx() );
    rhs->addTerm( Cv*rho_prev*T_prev * ve->dy() );
    rhs->addTerm( 0.5*rho_prev*u_prev*u_prev * ve->dy() );

    ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
    switch (norm)
    {
      // Automatic graph norm
      case 0:
      ip = bf->graphNorm();
      break;

      // Manual Graph norm
      case 1:
      ip->addTerm(1./mu*S + vm->dx() + u_prev*ve->dx());
      ip->addTerm(Pr/(Cp*mu)*tau - ve->dx());
      ip->addTerm(u_prev*vc->dx()+vc->dy()+u_prev*u_prev*vm->dx()+R*T_prev*vm->dx()+u_prev*vm->dy()
        +Cv*T_prev*u_prev*ve->dx()+0.5*u_prev*u_prev*u_prev*ve->dx()+R*T_prev*u_prev*ve->dx()+Cv*T_prev*ve->dy()+0.5*u_prev*u_prev*ve->dy());
      ip->addTerm(-4./3*S->dx()+rho_prev*vc->dx()+rho_prev*u_prev*vm->dx()+rho_prev*u_prev*vm->dx()+rho_prev*vm->dy()+Cv*rho_prev*T_prev*ve->dx()
        +0.5*rho_prev*u_prev*u_prev*ve->dx()+0.5*rho_prev*u_prev*u_prev*ve->dx()+0.5*rho_prev*u_prev*u_prev*ve->dx()
        +R*rho_prev*T_prev*ve->dx()-D_prev*ve->dx()+0.5*rho_prev*u_prev*ve->dy()+0.5*rho_prev*u_prev*ve->dy());
      ip->addTerm(tau->dx()+R*rho_prev*vm->dx()+Cv*rho_prev*u_prev*ve->dx()+R*rho_prev*u_prev*ve->dx()+Cv*rho_prev*ve->dy());
      ip->addTerm(vc);
      ip->addTerm(vm);
      ip->addTerm(ve);
      break;

      // Decoupled Eulerian and viscous norm
      // Might need to also elimnate D_prev term...
      case 2:
      ip->addTerm(vm->dx() + u_prev*ve->dx());
      ip->addTerm(ve->dx());
      ip->addTerm(u_prev*vc->dx()+vc->dy()+u_prev*u_prev*vm->dx()+R*T_prev*vm->dx()+u_prev*vm->dy()
        +Cv*T_prev*u_prev*ve->dx()+0.5*u_prev*u_prev*u_prev*ve->dx()+R*T_prev*u_prev*ve->dx()+Cv*T_prev*ve->dy()+0.5*u_prev*u_prev*ve->dy());
      ip->addTerm(rho_prev*vc->dx()+rho_prev*u_prev*vm->dx()+rho_prev*u_prev*vm->dx()+rho_prev*vm->dy()+Cv*rho_prev*T_prev*ve->dx()
        +0.5*rho_prev*u_prev*u_prev*ve->dx()+0.5*rho_prev*u_prev*u_prev*ve->dx()+0.5*rho_prev*u_prev*u_prev*ve->dx()
        +R*rho_prev*T_prev*ve->dx()-D_prev*ve->dx()+0.5*rho_prev*u_prev*ve->dy()+0.5*rho_prev*u_prev*ve->dy());
      ip->addTerm(R*rho_prev*vm->dx()+Cv*rho_prev*u_prev*ve->dx()+R*rho_prev*u_prev*ve->dx()+Cv*rho_prev*ve->dy());
      ip->addTerm(1./mu*S);
      ip->addTerm(Pr/(Cp*mu)*tau);
      ip->addTerm(vc);
      ip->addTerm(vm);
      ip->addTerm(ve);
      break;

      // Alternative Decoupled Eulerian and viscous norm
      case 3:
      ip->addTerm(vm->dx() + u_prev*ve->dx());
      ip->addTerm(ve->dx());
      ip->addTerm(u_prev*vc->dx()+vc->dy()+u_prev*u_prev*vm->dx()+R*T_prev*vm->dx()+u_prev*vm->dy()
        +Cv*T_prev*u_prev*ve->dx()+0.5*u_prev*u_prev*u_prev*ve->dx()+R*T_prev*u_prev*ve->dx()+Cv*T_prev*ve->dy()+0.5*u_prev*u_prev*ve->dy());
      ip->addTerm(rho_prev*vc->dx()+rho_prev*u_prev*vm->dx()+rho_prev*u_prev*vm->dx()+rho_prev*vm->dy()+Cv*rho_prev*T_prev*ve->dx()
        +0.5*rho_prev*u_prev*u_prev*ve->dx()+0.5*rho_prev*u_prev*u_prev*ve->dx()+0.5*rho_prev*u_prev*u_prev*ve->dx()
        +R*rho_prev*T_prev*ve->dx()+0.5*rho_prev*u_prev*ve->dy()+0.5*rho_prev*u_prev*ve->dy());
      ip->addTerm(R*rho_prev*vm->dx()+Cv*rho_prev*u_prev*ve->dx()+R*rho_prev*u_prev*ve->dx()+Cv*rho_prev*ve->dy());
      ip->addTerm(1./mu*S);
      ip->addTerm(Pr/(Cp*mu)*tau);
      ip->addTerm(vc);
      ip->addTerm(vm);
      ip->addTerm(ve);
      break;

      // Decoupled Eulerian and viscous norm
      // Might need to also elimnate D_prev term...
      case 4:
      ip->addTerm(vm->dx() + u_prev*ve->dx());
      ip->addTerm(ve->dx());
      ip->addTerm(u_prev*vc->dx()+vc->dy()+u_prev*u_prev*vm->dx()+R*T_prev*vm->dx()+u_prev*vm->dy()
        +Cv*T_prev*u_prev*ve->dx()+0.5*u_prev*u_prev*u_prev*ve->dx()+R*T_prev*u_prev*ve->dx()+Cv*T_prev*ve->dy()+0.5*u_prev*u_prev*ve->dy());
      ip->addTerm(rho_prev*vc->dx()+rho_prev*u_prev*vm->dx()+rho_prev*u_prev*vm->dx()+rho_prev*vm->dy()+Cv*rho_prev*T_prev*ve->dx()
        +0.5*rho_prev*u_prev*u_prev*ve->dx()+0.5*rho_prev*u_prev*u_prev*ve->dx()+0.5*rho_prev*u_prev*u_prev*ve->dx()
        +R*rho_prev*T_prev*ve->dx()-D_prev*ve->dx()+0.5*rho_prev*u_prev*ve->dy()+0.5*rho_prev*u_prev*ve->dy());
      ip->addTerm(R*rho_prev*vm->dx()+Cv*rho_prev*u_prev*ve->dx()+R*rho_prev*u_prev*ve->dx()+Cv*rho_prev*ve->dy());
      ip->addTerm(1./mu*S);
      ip->addTerm(4./3*S->dx());
      ip->addTerm(Pr/(Cp*mu)*tau);
      ip->addTerm(tau->dx());
      ip->addTerm(vc);
      ip->addTerm(vm);
      ip->addTerm(ve);
      break;

      // Alternative Decoupled Eulerian and viscous norm
      case 5:
      ip->addTerm(vm->dx() + u_prev*ve->dx());
      ip->addTerm(ve->dx());
      ip->addTerm(u_prev*vc->dx()+vc->dy()+u_prev*u_prev*vm->dx()+R*T_prev*vm->dx()+u_prev*vm->dy()
        +Cv*T_prev*u_prev*ve->dx()+0.5*u_prev*u_prev*u_prev*ve->dx()+R*T_prev*u_prev*ve->dx()+Cv*T_prev*ve->dy()+0.5*u_prev*u_prev*ve->dy());
      ip->addTerm(rho_prev*vc->dx()+rho_prev*u_prev*vm->dx()+rho_prev*u_prev*vm->dx()+rho_prev*vm->dy()+Cv*rho_prev*T_prev*ve->dx()
        +0.5*rho_prev*u_prev*u_prev*ve->dx()+0.5*rho_prev*u_prev*u_prev*ve->dx()+0.5*rho_prev*u_prev*u_prev*ve->dx()
        +R*rho_prev*T_prev*ve->dx()+0.5*rho_prev*u_prev*ve->dy()+0.5*rho_prev*u_prev*ve->dy());
      ip->addTerm(R*rho_prev*vm->dx()+Cv*rho_prev*u_prev*ve->dx()+R*rho_prev*u_prev*ve->dx()+Cv*rho_prev*ve->dy());
      ip->addTerm(1./mu*S);
      ip->addTerm(4./3*S->dx());
      ip->addTerm(Pr/(Cp*mu)*tau);
      ip->addTerm(tau->dx());
      ip->addTerm(vc);
      ip->addTerm(vm);
      ip->addTerm(ve);
      break;

      default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid problem number");
    }
  }

  ////////////////////   CREATE BCs   ///////////////////////
  vector< Teuchos::RCP<BCEasy> > bcs;
  // FunctionPtr Fc_prev;
  // FunctionPtr Fm_prev;
  // FunctionPtr Fe_prev;
  for (int slab=0; slab < numSlabs; slab++)
  {
    Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
    SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(xmin) );
    SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(xmax) );
    SpatialFilterPtr init = Teuchos::rcp( new ConstantYBoundary(tmins[slab]) );
    FunctionPtr rho0  = rhoR*one;
    FunctionPtr mom0 = zero;
    FunctionPtr E0    = Teuchos::rcp( new PulsedInitialCondition(0.5/pow(2.,numPreRefs), 1, 0) );
    bc->addDirichlet(Fc, left, zero);
    bc->addDirichlet(Fc, right, zero);
    bc->addDirichlet(Fm, left, zero);
    bc->addDirichlet(Fm, right, zero);
    bc->addDirichlet(Fe, left, zero);
    bc->addDirichlet(Fe, right, zero);
    if (slab == 0)
    {
      bc->addDirichlet(Fc, init, -rho0);
      bc->addDirichlet(Fm, init, -mom0);
      bc->addDirichlet(Fe, init, -E0);
    }
    bcs.push_back(bc);
  }

  ////////////////////   SOLVE & REFINE   ///////////////////////
  vector< Teuchos::RCP<Solution> > solutions;
  for (int slab=0; slab < numSlabs; slab++)
  {
    Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(meshes[slab], bcs[slab], rhss[slab], ips[slab]) );
    // ips[slab]->printInteractions();
    solutions.push_back(solution);
    if (slab > 0)
    {
      // fhat_prev = Function::solution(fhat, solutions[slab-1]);
      FunctionPtr rho_prev = Teuchos::RCP<Function>( new PreviousSolutionFunction(backgroundFlows[slab-1], rho) );
      FunctionPtr u_prev = Teuchos::RCP<Function>( new PreviousSolutionFunction(backgroundFlows[slab-1], u) );
      FunctionPtr T_prev = Teuchos::RCP<Function>( new PreviousSolutionFunction(backgroundFlows[slab-1], T) );
      SpatialFilterPtr init = Teuchos::rcp( new ConstantYBoundary(tmins[slab]) );
      bcs[slab]->addDirichlet(Fc, init, -rho_prev);
      bcs[slab]->addDirichlet(Fm, init, -rho_prev*u_prev);
      bcs[slab]->addDirichlet(Fe, init, -Cv*rho_prev*T_prev-0.5*rho_prev*u_prev*u_prev);
    }
    meshes[slab]->registerSolution(backgroundFlows[slab]);
    meshes[slab]->registerSolution(solutions[slab]);
    double energyThreshold = 0.2; // for mesh refinements
    RefinementStrategy refinementStrategy( solution, energyThreshold );
    VTKExporter exporter(backgroundFlows[slab], meshes[slab], varFactory);
    set<int> nonlinearVars;
    nonlinearVars.insert(D->ID());
    nonlinearVars.insert(rho->ID());
    nonlinearVars.insert(u->ID());
    nonlinearVars.insert(T->ID());

    for (int refIndex=0; refIndex<=numRefs; refIndex++)
    {
      double L2Update = 1e7;
      int iterCount = 0;
      while (L2Update > nlTol && iterCount < maxNewtonIterations)
      {
        solution->condensedSolve();
        double rhoL2Update = solution->L2NormOfSolutionGlobal(rho->ID());
        double uL2Update = solution->L2NormOfSolutionGlobal(u->ID());
        double TL2Update = solution->L2NormOfSolutionGlobal(T->ID());
        L2Update = sqrt(rhoL2Update*rhoL2Update + uL2Update*uL2Update + TL2Update*TL2Update);

        // line search algorithm
        double alpha = 1.0;
        // bool useLineSearch = true;
        // amount of enriching of grid points on which to ensure positivity
        int posEnrich = 5;
        if (useLineSearch)
        {
          double lineSearchFactor = .5;
          double eps = .001;
          FunctionPtr rhoTemp = Function::solution(rho,backgroundFlows[slab]) + alpha*Function::solution(rho,solution) - Function::constant(eps);
          FunctionPtr TTemp = Function::solution(T,backgroundFlows[slab]) + alpha*Function::solution(T,solution) - Function::constant(eps);
          bool rhoIsPositive = rhoTemp->isPositive(meshes[slab],posEnrich);
          // bool TIsPositive = TTemp->isPositive(mesh,posEnrich);
          bool TIsPositive = true;
          int iter = 0; int maxIter = 20;
          while (!(rhoIsPositive && TIsPositive) && iter < maxIter)
          {
            alpha = alpha*lineSearchFactor;
            rhoTemp = Function::solution(rho,backgroundFlows[slab]) + alpha*Function::solution(rho,solution);
            TTemp = Function::solution(T,backgroundFlows[slab]) + alpha*Function::solution(T,solution);
            rhoIsPositive = rhoTemp->isPositive(meshes[slab],posEnrich);
            // TIsPositive = TTemp->isPositive(mesh,posEnrich);
            TIsPositive = true;
            iter++;
          }
          if (commRank==0 && alpha < 1.0){
            cout << "line search factor alpha = " << alpha << endl;
          }
        }

        backgroundFlows[slab]->addSolution(solution, alpha, nonlinearVars);
        iterCount++;
        if (commRank == 0)
          cout << "L2 Norm of Update = " << L2Update << endl;
        if (alpha < 1e-2)
          break;
      }
      if (commRank == 0)
        cout << endl;

      if (commRank == 0)
      {
        stringstream outfile;
        outfile << problemName << norm << "_" << slab << "_" << refIndex;
        exporter.exportSolution(outfile.str());
      }

      if (refIndex < numRefs)
        refinementStrategy.refine(commRank==0);
    }
  }

  return 0;
}

