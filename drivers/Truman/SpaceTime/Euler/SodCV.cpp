//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "PreviousSolutionFunction.h"
#include "RefinementStrategy.h"
#include "SolutionExporter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

typedef map< int, FunctionPtr > sparseFxnVector;    // dim = {trialID}
typedef map< int, sparseFxnVector > sparseFxnMatrix; // dim = {testID, trialID}

double GAMMA = 1.4;
double halfwidth = 0.5;
double xmin = 0.5-halfwidth;
double xmax = 0.5+halfwidth;
double tmax = 0.2;

int H1Order = 3, pToAdd = 2;

class TimeZero : public SpatialFilter {
   public:
      bool matchesPoint(double x, double y) {
         double tol = 1e-14;
         return (abs(y) < tol);
      }
};

class LeftBoundary : public SpatialFilter {
   public:
      bool matchesPoint(double x, double y) {
         double tol = 1e-14;
         return (abs(x-xmin) < tol);
      }
};

class RightBoundary : public SpatialFilter {
   public:
      bool matchesPoint(double x, double y) {
         double tol = 1e-14;
         return (abs(x-xmax) < tol);
      }
};

class InitialDensity : public Function {
   public:
      InitialDensity() : Function(0) {}
      void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
         int numCells = values.dimension(0);
         int numPoints = values.dimension(1);

         const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
         for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
               double x = (*points)(cellIndex,ptIndex,0);
               double y = (*points)(cellIndex,ptIndex,1);
               if (x <= 0.5)
                  values(cellIndex, ptIndex) = 1.0;
               else
                  values(cellIndex, ptIndex) = 0.125;
            }
         }
      }
};

class InitialMassFlux : public Function {
   public:
      InitialMassFlux() : Function(0) {}
      void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
         int numCells = values.dimension(0);
         int numPoints = values.dimension(1);

         const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
         for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
               double x = (*points)(cellIndex,ptIndex,0);
               double y = (*points)(cellIndex,ptIndex,1);
               values(cellIndex, ptIndex) = 0.0;
            }
         }
      }
};

class InitialTotalEnergy : public Function {
   public:
      InitialTotalEnergy() : Function(0) {}
      void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
         int numCells = values.dimension(0);
         int numPoints = values.dimension(1);

         const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
         for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
               double x = (*points)(cellIndex,ptIndex,0);
               double y = (*points)(cellIndex,ptIndex,1);
               if (x <= 0.5)
                  values(cellIndex, ptIndex) = 1.0/(GAMMA-1);
               else
                  values(cellIndex, ptIndex) = 0.1/(GAMMA-1);
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
   int numRefs = args.Input<int>("--numRefs", "number of refinement steps");
   int maxNewtonIterations = args.Input<int>("--maxIterations", "maximum number of Newton iterations");

   // Optional arguments (have defaults)
   args.Process();

   ////////////////////   DECLARE VARIABLES   ///////////////////////
   // define test variables
   VarFactory varFactory;
   VarPtr vm = varFactory.testVar("vm", HGRAD);
   VarPtr vx = varFactory.testVar("vx", HGRAD);
   VarPtr ve = varFactory.testVar("ve", HGRAD);

   // define trial variables
   VarPtr rho = varFactory.fieldVar("rho");
   VarPtr m = varFactory.fieldVar("m");
   VarPtr E = varFactory.fieldVar("E");
   VarPtr Fm = varFactory.fluxVar("Fm");
   VarPtr Fx = varFactory.fluxVar("Fx");
   VarPtr Fe = varFactory.fluxVar("Fe");

   ////////////////////   BUILD MESH   ///////////////////////
   BFPtr bf = Teuchos::rcp( new BF(varFactory) );
   // define nodes for mesh
   FieldContainer<double> meshBoundary(4,2);

   meshBoundary(0,0) =  xmin; // x1
   meshBoundary(0,1) =  0.0; // y1
   meshBoundary(1,0) =  xmax;
   meshBoundary(1,1) =  0.0;
   meshBoundary(2,0) =  xmax;
   meshBoundary(2,1) =  tmax;
   meshBoundary(3,0) =  xmin;
   meshBoundary(3,1) =  tmax;

   int horizontalCells = 32, verticalCells = 8;

   // create a pointer to a new mesh:
   Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
         bf, H1Order, H1Order+pToAdd);

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

   FunctionPtr rho_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, rho) );
   FunctionPtr m_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, m) );
   FunctionPtr E_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, E) );

   // ==================== SET INITIAL GUESS ==========================
   map<int, Teuchos::RCP<Function> > functionMap;
   FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
   FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
   functionMap[rho->ID()] = Teuchos::rcp( new InitialDensity );
   functionMap[m->ID()] = zero;
   functionMap[E->ID()] = Teuchos::rcp( new InitialTotalEnergy );

   backgroundFlow->projectOntoMesh(functionMap);

   ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

   // conservation law fluxes
   bf->addTerm(Fm, vm);
   bf->addTerm(Fx, vx);
   bf->addTerm(Fe, ve);

   // Jacobians
   sparseFxnMatrix Jm;
   sparseFxnMatrix Jx;
   sparseFxnMatrix Je;

   map<int, VarPtr> U;
   U[rho->ID()] = rho;
   U[m->ID()] = m;
   U[E->ID()] = E;

   int x_comp = 0;
   int t_comp = 1;

   Jm[x_comp][rho->ID()] = zero;
   Jm[x_comp][m->ID()]   = one;
   Jm[x_comp][E->ID()]   = zero;
   Jm[t_comp][rho->ID()] = one;
   Jm[t_comp][m->ID()]   = zero;
   Jm[t_comp][E->ID()]   = zero;

   Jx[x_comp][rho->ID()] = 0.5*(GAMMA-3)*m_prev*m_prev/(rho_prev*rho_prev);
   Jx[x_comp][m->ID()]   = (3-GAMMA)*m_prev/rho_prev;
   Jx[x_comp][E->ID()]   = (GAMMA-1)*one;
   Jx[t_comp][rho->ID()] = zero;
   Jx[t_comp][m->ID()]   = one;
   Jx[t_comp][E->ID()]   = zero;

   Je[x_comp][rho->ID()] = (GAMMA-1)*m_prev*m_prev*m_prev/(rho_prev*rho_prev*rho_prev)
      -GAMMA*m_prev/(rho_prev*rho_prev)*E_prev;
   Je[x_comp][m->ID()]   = 1.5*(1-GAMMA)*m_prev*m_prev/(rho_prev*rho_prev)
      +GAMMA*E_prev/rho_prev;
   Je[x_comp][E->ID()]   = GAMMA*m_prev/rho_prev;
   Je[t_comp][rho->ID()] = zero;
   Je[t_comp][m->ID()]   = zero;
   Je[t_comp][E->ID()]   = one;

   for (int j=0; j < 3; j++)
   {
      bf->addTerm( Jm[x_comp][j]*U[j], -vm->dx() );
      bf->addTerm( Jm[t_comp][j]*U[j], -vm->dy() );
      bf->addTerm( Jx[x_comp][j]*U[j], -vx->dx() );
      bf->addTerm( Jx[t_comp][j]*U[j], -vx->dy() );
      bf->addTerm( Je[x_comp][j]*U[j], -ve->dx() );
      bf->addTerm( Je[t_comp][j]*U[j], -ve->dy() );
   }

   ////////////////////   SPECIFY RHS   ///////////////////////
   Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
   rhs->addTerm( (m_prev) * vm->dx() );
   rhs->addTerm( (rho_prev) * vm->dy() );

   rhs->addTerm( (0.5*(3-GAMMA)*m_prev*m_prev/rho_prev+(GAMMA-1)*E_prev) * vx->dx() );
   rhs->addTerm( (m_prev) * vx->dy() );

   rhs->addTerm( (0.5*(1-GAMMA)*m_prev*m_prev*m_prev/(rho_prev*rho_prev)+GAMMA*m_prev/rho_prev*E_prev) * ve->dx() );
   rhs->addTerm( (E_prev) * ve->dy() );

   ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
   IPPtr ip = bf->graphNorm();
   ip->addTerm( 5e-2*vm->dx() );
   ip->addTerm( 5e-2*vx->dx() );
   ip->addTerm( 5e-2*ve->dx() );

   ////////////////////   CREATE BCs   ///////////////////////
   Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
   SpatialFilterPtr timezero = Teuchos::rcp( new TimeZero );
   SpatialFilterPtr left = Teuchos::rcp( new LeftBoundary );
   SpatialFilterPtr right = Teuchos::rcp( new RightBoundary );
   FunctionPtr rho0 = Teuchos::rcp( new InitialDensity );
   FunctionPtr mass0 = Teuchos::rcp( new InitialMassFlux );
   FunctionPtr E0 = Teuchos::rcp( new InitialTotalEnergy );
   bc->addDirichlet(Fm, timezero, -rho0);
   bc->addDirichlet(Fx, timezero, -mass0);
   bc->addDirichlet(Fe, timezero, -E0);
   bc->addDirichlet(Fm, left, zero);
   bc->addDirichlet(Fm, right, zero);
   bc->addDirichlet(Fx, left, -one);
   bc->addDirichlet(Fx, right, 0.1*one);
   bc->addDirichlet(Fe, left, zero);
   bc->addDirichlet(Fe, right, zero);

   Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

   // ==================== Register Solutions ==========================
   mesh->registerSolution(solution);
   mesh->registerSolution(backgroundFlow);

   ////////////////////   SOLVE & REFINE   ///////////////////////
   double energyThreshold = 0.2; // for mesh refinements
   RefinementStrategy refinementStrategy( solution, energyThreshold );
   VTKExporter exporter(backgroundFlow, mesh, varFactory);

   double nonlinearRelativeEnergyTolerance = 1e-5; // used to determine convergence of the nonlinear solution
   for (int refIndex=0; refIndex<=numRefs; refIndex++)
   {
      double L2Update = 1e7;
      int iterCount = 0;
      while (L2Update > nonlinearRelativeEnergyTolerance && iterCount < maxNewtonIterations)
      {
         solution->solve(false);
         double rhoL2Update = solution->L2NormOfSolutionGlobal(rho->ID());
         double mL2Update = solution->L2NormOfSolutionGlobal(m->ID());
         double EL2Update = solution->L2NormOfSolutionGlobal(E->ID());
         L2Update = sqrt(rhoL2Update*rhoL2Update + mL2Update*mL2Update + EL2Update*EL2Update);

         // line search algorithm
         double alpha = 1.0;
         bool useLineSearch = true;
         int posEnrich = 5; // amount of enriching of grid points on which to ensure positivity
         if (useLineSearch){ // to enforce positivity of density rho
            double lineSearchFactor = .5; double eps = .001; // arbitrary
            FunctionPtr rhoTemp = Function::solution(rho,backgroundFlow) + alpha*Function::solution(rho,solution) - Function::constant(eps);
            FunctionPtr ETemp = Function::solution(E,backgroundFlow) + alpha*Function::solution(E,solution) - Function::constant(eps);
            bool rhoIsPositive = rhoTemp->isPositive(mesh,posEnrich);
            bool EIsPositive = ETemp->isPositive(mesh,posEnrich);
            int iter = 0; int maxIter = 20;
            while (!(rhoIsPositive && EIsPositive) && iter < maxIter){
               alpha = alpha*lineSearchFactor;
               rhoTemp = Function::solution(rho,backgroundFlow) + alpha*Function::solution(rho,solution);
               ETemp = Function::solution(E,backgroundFlow) + alpha*Function::solution(E,solution);
               rhoIsPositive = rhoTemp->isPositive(mesh,posEnrich);
               EIsPositive = ETemp->isPositive(mesh,posEnrich);
               iter++;
            }
            alpha = max(alpha, 0.05);
            if (commRank==0 && alpha < 1.0){
               cout << "line search factor alpha = " << alpha << endl;
            }
         }

         backgroundFlow->addSolution(solution, alpha);
         iterCount++;
         if (commRank == 0)
            cout << "L2 Norm of Update = " << L2Update << endl;
      }
      if (commRank == 0)
         cout << endl;

      if (commRank == 0)
      {
         stringstream outfile;
         outfile << "SodCV_" << refIndex;
         exporter.exportSolution(outfile.str());
      }

      if (refIndex < numRefs)
         refinementStrategy.refine(commRank==0); // print to console on commRank 0
   }

   return 0;
}

