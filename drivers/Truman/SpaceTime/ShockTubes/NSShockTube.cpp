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

class PowerFunction : public Function {
  private:
    FunctionPtr _function;
    double _power;
  public:
    PowerFunction(FunctionPtr function, double power) : Function(0), _function(function), _power(power) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      _function->values(values, basisCache);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          values(cellIndex, ptIndex) = pow(values(cellIndex, ptIndex), _power);
        }
      }
    }
};

class ExpFunction : public Function {
  private:
    FunctionPtr _function;
  public:
    ExpFunction(FunctionPtr function) : Function(0), _function(function) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      _function->values(values, basisCache);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          values(cellIndex, ptIndex) = exp(values(cellIndex, ptIndex));
        }
      }
    }
};

class SqrtFunction : public Function {
  private:
    FunctionPtr _function;
  public:
    SqrtFunction(FunctionPtr function) : Function(0), _function(function) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      _function->values(values, basisCache);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          values(cellIndex, ptIndex) = sqrt(values(cellIndex, ptIndex));
        }
      }
    }
};

class BoundedBelowFunction : public Function {
  private:
    FunctionPtr _function;
    double _bound;
  public:
    BoundedBelowFunction(FunctionPtr function, double bound) : Function(0), _function(function), _bound(bound) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      _function->values(values, basisCache);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          values(cellIndex, ptIndex) = max(values(cellIndex, ptIndex),_bound);
        }
      }
    }
};

class BoundedSqrtFunction : public Function {
  private:
    FunctionPtr _function;
    double _bound;
  public:
    BoundedSqrtFunction(FunctionPtr function, double bound) : Function(0), _function(function), _bound(bound) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      _function->values(values, basisCache);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          values(cellIndex, ptIndex) = sqrt(max(values(cellIndex, ptIndex),_bound));
        }
      }
    }
};

class ArtificialViscosity : public Function {
  private:
    FunctionPtr _h;
    FunctionPtr _rho;
    FunctionPtr _dudx;
    FunctionPtr _sqrtT;
    FunctionPtr _D;
    FunctionPtr _dDdx;
    double _gamma;
    double _R;
    double _qlin;
    double _qquad;
  public:
    ArtificialViscosity(FunctionPtr h, FunctionPtr rho, FunctionPtr dudx, FunctionPtr sqrtT, FunctionPtr D, FunctionPtr dDdx, double gamma, double R, double qlin=0.25, double qquad=2) : Function(0),
      _h(h), _rho(rho), _dudx(dudx), _sqrtT(sqrtT), _D(D), _dDdx(dDdx), _gamma(gamma), _R(R), _qlin(qlin), _qquad(qquad) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      FieldContainer<double> h(values);
      FieldContainer<double> rho(values);
      FieldContainer<double> dudx(values);
      FieldContainer<double> sqrtT(values);
      FieldContainer<double> D(values);
      FieldContainer<double> dDdx(values);
      _h->values(h, basisCache);
      _rho->values(rho, basisCache);
      _dudx->values(dudx, basisCache);
      _sqrtT->values(sqrtT, basisCache);
      _D->values(D, basisCache);
      _dDdx->values(dDdx, basisCache);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double a0 = sqrt(_gamma*_R)*sqrtT(cellIndex, ptIndex);
          double Vz = 0;
          double Cz = dudx(cellIndex, ptIndex);
          double psi0 = 1;
          double psi1 = 1;
          double psi2 = 1;
          if (Cz >= 0)
            psi1 = 0;
          if (abs(dDdx(cellIndex, ptIndex)) < 1e-2)
            psi1 = 0;
          // if (Cz < 1e-6)
          // {
          //   psi1 = max(-Cz/abs(Cz),0.);
          //   psi2 = Cz/(Cz+Vz);
          // }
          // else
          // {
          //   psi1 = 0;
          //   psi2 = 0;
          // }
          double artViscScaling = 1e-0;
          values(cellIndex, ptIndex) = 1e-4 + artViscScaling*psi0*psi1*rho(cellIndex, ptIndex)*h(cellIndex, ptIndex)*(_qquad*h(cellIndex, ptIndex)*Cz + psi2*_qlin*a0);
          // values(cellIndex, ptIndex) = 1e-4 + rho(cellIndex, ptIndex)*h(cellIndex, ptIndex)*(_qquad*h(cellIndex, ptIndex)*Cz + psi2*_qlin*a0);
        }
      }
    }
};

class ErrorFunction : public Function {
  private:
    SolutionPtr _solution;
  public:
    ErrorFunction(SolutionPtr solution) : Function(0), _solution(solution) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      // _function->values(values, basisCache);
      map<int, double> errorMap = _solution->energyError();

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        map<int, double>::iterator it = errorMap.find(cellIndex);
        if (it != errorMap.end())
          cout << "cell " << cellIndex << " error " << errorMap[cellIndex] << endl;
        else
          cout << " Problem " << endl;
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          // values(cellIndex, ptIndex) = errorMap[cellIndex];
          values(cellIndex, ptIndex) = 0;
        }
      }
    }
};

class PulseInitialCondition : public Function {
   private:
      double _width;
      double _valI;
      double _valO;
      FunctionPtr _h;
   public:
      PulseInitialCondition(double width, double valI, double valO, FunctionPtr h) : Function(0), _width(width), _valI(valI), _valO(valO), _h(h) {}
      void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
         int numCells = values.dimension(0);
         int numPoints = values.dimension(1);

         FieldContainer<double> h(values);
         _h->values(h, basisCache);

         const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
         for (int cellIndex=0; cellIndex<numCells; cellIndex++)
         {
          for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
          {
            double x = (*points)(cellIndex,ptIndex,0);
            double y = (*points)(cellIndex,ptIndex,1);
            double w = 0.2*h(cellIndex, ptIndex);
            if (abs(x) <= w)
              values(cellIndex, ptIndex) = _valI/w;
            else
              values(cellIndex, ptIndex) = _valO;
          }
               // double w = min(_width, 0.5*h(cellIndex, ptIndex));
               // double w;
               // for (int i = 0; i <= 8; i++)
               //  if (h(cellIndex, ptIndex) < pow(2,-i))
               //    w = 0.5*pow(2,-i);
               // // double w = h(cellIndex, ptIndex);
               // // if (abs(x) <= 1./pow(2,_numPreRefs))
               // if (abs(x) < 0.5*w)
               //    // values(cellIndex, ptIndex) = _valI*pow(2,_numPreRefs);
               //    // values(cellIndex, ptIndex) = _valO+2*(_valI-_valO)/w*(1-abs(2*x/w));
               // {
               //    values(cellIndex, ptIndex) = 1;
               //    cout << "(" << h(cellIndex, ptIndex) << " " << w << ") ";
               // }
               // else
               //    // values(cellIndex, ptIndex) = _valO;
               //    values(cellIndex, ptIndex) = 0;
         }
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
  int problem = args.Input<int>("--problem", "which problem to run");
  int formulation = args.Input<int>("--formulation", "which formulation to use: 0 = primitive, 1 = conservation, 2 = entropy");

   // Optional arguments (have defaults)
  int numRefs = args.Input("--numRefs", "number of refinement steps", 0);
  int numPreRefs = args.Input<int>("--numPreRefs","pre-refinements on singularity",0);
  int norm = args.Input("--norm", "norm", 0);
  double physicalViscosity = args.Input("--mu", "viscosity", 1e-2);
  double mu = physicalViscosity;
  double mu_sqrt = sqrt(mu);
  int numSlabs = args.Input("--numSlabs", "number of time slabs", 1);
  bool useLineSearch = args.Input("--lineSearch", "use line search", true);
  int polyOrder = args.Input("--polyOrder", "polynomial order for field variables", 2);
  int deltaP = args.Input("--deltaP", "how much to enrich test space", 2);
  int cubatureEnrichment = args.Input("--cubatureEnrichment", "how much to enrich cubature", 0);
  int numX = args.Input("--numX", "number of cells in the x direction", 4);
  int numT = args.Input("--numT", "number of cells in the t direction", 1);
  int maxNewtonIterations = args.Input("--maxIterations", "maximum number of Newton iterations", 10);
  double nlTol = args.Input("--nlTol", "nonlinear tolerance", 1e-6);

  args.Process();

   ////////////////////   PROBLEM DEFINITIONS   ///////////////////////
  int H1Order = polyOrder+1;

  double xmin, xmax, xint, tmin, tmax;
  double rhoL, rhoR, uL, uR, pL, pR, eL, eR, TL, TR, UcL, UcR, UmL, UmR, UeL, UeR, mL, mR, EL, ER;
  double Pr = 0.713;
  double gamma = 1.4;
  double p0 = 1;
  double rho0 = 1;
  double u0 = 1;
  double a0 = sqrt(gamma*p0/rho0);
  double M_inf = u0/a0;
  double Cv = 1./(gamma*(gamma-1)*M_inf*M_inf);
  double Cp = gamma*Cv;
  double R = Cp-Cv;
  string problemName;

  switch (problem)
  {
    case 0:
    // Trivial problem for testing
    problemName = "Trivial";
    xmin = 0;
    xmax = 1;
    xint = 0.5;
    tmax = 0.1;

    rhoL = 1;
    rhoR = 1;
    uL = 1;
    uR = 1;
    TL = 1;
    TR = 1;
    break;
    case 1:
    // Simple shock for testing
    problemName = "SimpleShock";
    xmin = 0;
    xmax = 1;
    xint = 0.5;
    tmax = 0.1;

    rhoL = 1;
    rhoR = 1;
    uL = 1;
    uR = 0;
    TL = 1;
    TR = 1;
    break;
    case 2:
    // Simple rarefaction for testing
    problemName = "SimpleRarefaction";
    xmin = 0;
    xmax = 1;
    xint = 0.5;
    tmax = 0.1;

    rhoL = 1;
    rhoR = 1;
    uL = -1;
    uR = 1;
    TL = 1;
    TR = 1;
    break;
    case 3:
    // Sod shock tube
    problemName = "Sod";
    xmin = 0;
    xmax = 1;
    xint = 0.5;
    tmax = 0.2;

    rhoL = 1;
    rhoR = .125;
    uL = 0;
    uR = 0;
    TL = 1/(rhoL*R);
    TR = .1/(rhoR*R);
    break;
    case 4:
    // Strong shock tube
    problemName = "StrongShock";
    xmin = 0;
    xmax = 5;
    xint = 2.5;
    tmax = 4e-1;

    rhoL = 10;
    rhoR = 1;
    uL = 0;
    uR = 0;
    TL = 100/(rhoL*R);
    TR = 1/(rhoR*R);
    break;
    case 5:
    // Strong shock tube
    problemName = "Noh";
    gamma = 5./3;
    a0 = sqrt(gamma*p0/rho0);
    M_inf = u0/a0;
    Cv = 1./(gamma*(gamma-1)*M_inf*M_inf);
    Cp = gamma*Cv;
    R = Cp-Cv;
    xmin = 0;
    xmax = 1;
    xint = .5;
    tmax = .25;

    rhoL = 1;
    rhoR = 1;
    uL = 1;
    uR = -1;
    // TL = p0/(rho0*R*T0);
    // TR = p0/(rho0*R*T0);
    TL = 0;
    TR = 0;
    break;
    case 6:
    // Strong shock tube
    problemName = "Sedov";
    xmin = -3.;
    xmax = 3;
    // xint = -.5+1./64;
    xint = 0;
    tmax = 1;

    rhoL = 1;
    rhoR = 1;
    uL = 0;
    uR = 0;
    TL = 0;
    TR = 0;
    break;
    default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid problem number");
  }
  if (commRank == 0)
    cout << "Running the " << problemName << " problem" << endl;

  switch (formulation)
  {
    case 0:
    UcL = rhoL;
    UcR = rhoR;
    UmL = uL;
    UmR = uR;
    UeL = TL;
    UeR = TR;
    break;
    case 1:
    UcL = rhoL;
    UcR = rhoR;
    UmL = rhoL*uL;
    UmR = rhoL*uR;
    UeL = rhoL*(Cv*TL+0.5*uL*uL);
    UeR = rhoR*(Cv*TR+0.5*uR*uR);
    break;
    case 2:
    mL = rhoL*uL;
    mR = rhoL*uR;
    EL = rhoL*(Cv*TL+0.5*uL*uL);
    ER = rhoR*(Cv*TR+0.5*uR*uR);
    UcL = (-EL+(EL-0.5*mL*mL/rhoL)*(gamma+1-log(((gamma-1)*(EL-0.5*mL*mL/rhoL)/pow(rhoL,gamma)))))/(EL-0.5*mL*mL/rhoL);
    UcR = (-ER+(ER-0.5*mR*mR/rhoR)*(gamma+1-log(((gamma-1)*(ER-0.5*mR*mR/rhoR)/pow(rhoR,gamma)))))/(ER-0.5*mR*mR/rhoR);
    UmL = mL/(EL-0.5*mL*mL/rhoL);
    UmR = mR/(ER-0.5*mR*mR/rhoR);
    UeL = -rhoL/(EL-0.5*mL*mL/rhoL);
    UeR = -rhoR/(ER-0.5*mR*mR/rhoR);
    break;
    default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid formulation");
  }

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr S = varFactory.testVar("S", HGRAD);
  VarPtr tau = varFactory.testVar("tau", HGRAD);
  VarPtr vc = varFactory.testVar("vc", HGRAD);
  VarPtr vm = varFactory.testVar("vm", HGRAD);
  VarPtr ve = varFactory.testVar("ve", HGRAD);

  // define trial variables
  // common variables amongst formulations
  VarPtr D = varFactory.fieldVar("D");
  VarPtr q = varFactory.fieldVar("q");
  VarPtr uhat = varFactory.spatialTraceVar("uhat");
  VarPtr That = varFactory.spatialTraceVar("That");
  VarPtr tc = varFactory.fluxVar("tc");
  VarPtr tm = varFactory.fluxVar("tm");
  VarPtr te = varFactory.fluxVar("te");
  // variables that vary among formulations
  VarPtr Uc = varFactory.fieldVar("Uc");
  VarPtr Um = varFactory.fieldVar("Um");
  VarPtr Ue = varFactory.fieldVar("Ue");
  // create aliases for these for each formulation
  VarPtr rho = Uc;
  VarPtr u = Um;
  VarPtr T = Ue;
  VarPtr m = Um;
  VarPtr E = Ue;
  VarPtr zc = Uc;
  VarPtr zm = Um;
  VarPtr ze = Ue;

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
    // xmin = 0.0;
    // xmax = 1.0;
    double tminslab = tmax*double(slab)/numSlabs;
    double tmaxslab = tmax*double(slab+1)/numSlabs;

    meshBoundary(0,0) =  xmin; // x1
    meshBoundary(0,1) =  tminslab; // y1
    meshBoundary(1,0) =  xmax;
    meshBoundary(1,1) =  tminslab;
    meshBoundary(2,0) =  xmax;
    meshBoundary(2,1) =  tmaxslab;
    meshBoundary(3,0) =  xmin;
    meshBoundary(3,1) =  tmaxslab;

    // create a pointer to a new mesh:
    Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, numX, numT,
      bfs[slab], H1Order, H1Order+pToAdd);

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
            if ((abs(vertices(j,0)-xint)<1e-14) && (abs(vertices(j,1))<1e-14) && !cellIDset)
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
    tmins.push_back(tminslab);
    tmaxs.push_back(tmaxslab);
  }

  ////////////////////   SET INITIAL CONDITIONS   ///////////////////////
  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  map<int, Teuchos::RCP<Function> > initialGuess;
  // initialGuess[rho->ID()] = Teuchos::rcp( new DiscontinuousInitialCondition(xint, rhoL, rhoR) ) ;
  // initialGuess[u->ID()]   = Teuchos::rcp( new DiscontinuousInitialCondition(xint, uL, uR) );
  // initialGuess[T->ID()]   = Teuchos::rcp( new DiscontinuousInitialCondition(xint, TL, TR) );
  initialGuess[Uc->ID()] = Teuchos::rcp( new RampedInitialCondition(xint, UcL, UcR, (xmax-xmin)/numX) ) ;
  initialGuess[Um->ID()]   = Teuchos::rcp( new RampedInitialCondition(xint, UmL, UmR,     (xmax-xmin)/numX) );
  initialGuess[Ue->ID()]   = Teuchos::rcp( new RampedInitialCondition(xint, UeL, UeR,     (xmax-xmin)/numX) );

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

    FunctionPtr Uc_prev = Function::solution(Uc, backgroundFlows[slab]);
    FunctionPtr Um_prev   = Function::solution(Um, backgroundFlows[slab]);
    FunctionPtr Ue_prev   = Function::solution(Ue, backgroundFlows[slab]);
    FunctionPtr D_prev   = Function::solution(D, backgroundFlows[slab]);
    FunctionPtr q_prev   = Function::solution(q, backgroundFlows[slab]);
    FunctionPtr rho_prev = Uc_prev;
    FunctionPtr u_prev = Um_prev;
    FunctionPtr T_prev = Ue_prev;
    FunctionPtr m_prev = Um_prev;
    FunctionPtr E_prev = Ue_prev;
    FunctionPtr zc_prev = Uc_prev;
    FunctionPtr zm_prev = Um_prev;
    FunctionPtr ze_prev = Ue_prev;
    // FunctionPtr dudx_prev   = Function::solution(Um->dx(), backgroundFlows[slab]);
    // FunctionPtr dDdx_prev   = Function::solution(D->dx(), backgroundFlows[slab]);
    // Nonlinear Residual Terms
    FunctionPtr Fc;
    FunctionPtr Fm;
    FunctionPtr Fe;
    FunctionPtr Cc;
    FunctionPtr Cm;
    FunctionPtr Ce;
    FunctionPtr Km;
    FunctionPtr Ke;
    FunctionPtr MD;
    FunctionPtr Mq;
    FunctionPtr GD;
    FunctionPtr Gq;
    // Linearized Terms
    LinearTermPtr Fc_dU = Teuchos::rcp( new LinearTerm );
    LinearTermPtr Fm_dU = Teuchos::rcp( new LinearTerm );
    LinearTermPtr Fe_dU = Teuchos::rcp( new LinearTerm );
    LinearTermPtr Cc_dU = Teuchos::rcp( new LinearTerm );
    LinearTermPtr Cm_dU = Teuchos::rcp( new LinearTerm );
    LinearTermPtr Ce_dU = Teuchos::rcp( new LinearTerm );
    LinearTermPtr Km_dU = Teuchos::rcp( new LinearTerm );
    LinearTermPtr Ke_dU = Teuchos::rcp( new LinearTerm );
    LinearTermPtr MD_dU = Teuchos::rcp( new LinearTerm );
    LinearTermPtr Mq_dU = Teuchos::rcp( new LinearTerm );
    LinearTermPtr GD_dU = Teuchos::rcp( new LinearTerm );
    LinearTermPtr Gq_dU = Teuchos::rcp( new LinearTerm );
    // Adjoint Terms
    LinearTermPtr adj_Fc = Teuchos::rcp( new LinearTerm );
    LinearTermPtr adj_Fm = Teuchos::rcp( new LinearTerm );
    LinearTermPtr adj_Fe = Teuchos::rcp( new LinearTerm );
    LinearTermPtr adj_Cc = Teuchos::rcp( new LinearTerm );
    LinearTermPtr adj_Cm = Teuchos::rcp( new LinearTerm );
    LinearTermPtr adj_Ce = Teuchos::rcp( new LinearTerm );
    LinearTermPtr adj_KD = Teuchos::rcp( new LinearTerm );
    LinearTermPtr adj_Kq = Teuchos::rcp( new LinearTerm );
    LinearTermPtr adj_MD = Teuchos::rcp( new LinearTerm );
    LinearTermPtr adj_Mq = Teuchos::rcp( new LinearTerm );
    LinearTermPtr adj_Gc = Teuchos::rcp( new LinearTerm );
    LinearTermPtr adj_Gm = Teuchos::rcp( new LinearTerm );
    LinearTermPtr adj_Ge = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr M_DxS = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr M_qxtau = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr Msqrt_DxS = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr Msqrt_qxtau = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr K_DxGradV = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr K_qxGradV = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr MinvsqrtxK_DxGradV = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr MinvsqrtxK_qxGradV = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr F_cxGradV = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr C_cxdVdt = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr F_mxGradV = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr C_mxdVdt = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr F_exGradV = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr C_exdVdt = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr G_cxGradPsi = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr G_mxGradPsi = Teuchos::rcp( new LinearTerm );
    // LinearTermPtr G_exGradPsi = Teuchos::rcp( new LinearTerm );

    // Entropy Scaling
    // FunctionPtr T_sqrt = Teuchos::rcp( new BoundedSqrtFunction(T_prev, 1e-2) );
    // FunctionPtr rho_sqrt = Teuchos::rcp( new BoundedSqrtFunction(rho_prev, 1e-2) );
    // FunctionPtr A0p_c = sqrt(gamma-1)/rho_sqrt;
    // FunctionPtr A0p_m = rho_sqrt/(sqrt(Cv)*T_sqrt);
    // FunctionPtr A0p_e = rho_sqrt/(T_sqrt*T_sqrt);
    // // FunctionPtr A0p_e = rho_sqrt/(T_sqrt*T_sqrt);
    // // FunctionPtr invA0p_c = rho_sqrt/sqrt(gamma-1);
    // // FunctionPtr invA0p_m = (sqrt(Cv)*T_sqrt)/rho_sqrt;
    // // FunctionPtr invA0p_e = T_prev/rho_sqrt;
    // FunctionPtr invA0p_c = 1./A0p_c;
    // FunctionPtr invA0p_m = 1./A0p_m;
    // FunctionPtr invA0p_e = 1./A0p_e;

    // Artificial Viscosity
    // FunctionPtr artificialViscosity = Teuchos::rcp( new ArtificialViscosity(Function::h(), rho_prev, dudx_prev, T_sqrt, D_prev, dDdx_prev, gamma, R) );
    // FunctionPtr mu = artificialViscosity;
    // FunctionPtr mu_sqrt = Teuchos::rcp( new SqrtFunction(mu) );
    switch (formulation)
    {
      case 0:
      //   /$$$$$$$            /$$               /$$   /$$     /$$
      //  | $$__  $$          |__/              |__/  | $$    |__/
      //  | $$  \ $$  /$$$$$$  /$$ /$$$$$$/$$$$  /$$ /$$$$$$   /$$ /$$    /$$  /$$$$$$
      //  | $$$$$$$/ /$$__  $$| $$| $$_  $$_  $$| $$|_  $$_/  | $$|  $$  /$$/ /$$__  $$
      //  | $$____/ | $$  \__/| $$| $$ \ $$ \ $$| $$  | $$    | $$ \  $$/$$/ | $$$$$$$$
      //  | $$      | $$      | $$| $$ | $$ | $$| $$  | $$ /$$| $$  \  $$$/  | $$_____/
      //  | $$      | $$      | $$| $$ | $$ | $$| $$  |  $$$$/| $$   \  $/   |  $$$$$$$
      //  |__/      |__/      |__/|__/ |__/ |__/|__/   \___/  |__/    \_/     \_______/
      //
      //
      //
      // Nonlinear Residual Terms
      Cc = rho_prev;
      Cm = rho_prev*u_prev;
      Ce = Cv*rho_prev*T_prev + 0.5*rho_prev*u_prev*u_prev;
      Fc = rho_prev*u_prev;
      Fm = rho_prev*u_prev*u_prev + R*rho_prev*T_prev;
      Fe = Cv*rho_prev*u_prev*T_prev + 0.5*rho_prev*u_prev*u_prev*u_prev + R*rho_prev*u_prev*T_prev;
      Km = D_prev;
      Ke = -q_prev + u_prev*D_prev;
      MD = 1./mu*D_prev;
      Mq = Pr/(Cp*mu)*q_prev;
      GD = 2*u_prev;
      Gq = -T_prev;

      // Linearized Terms
      Cc_dU->addTerm( 1*rho );
      Cm_dU->addTerm( rho_prev*u + u_prev*rho );
      Ce_dU->addTerm( Cv*T_prev*rho + Cv*rho_prev*T + 0.5*u_prev*u_prev*rho + rho_prev*u_prev*u );
      Fc_dU->addTerm( u_prev*rho + rho_prev*u );
      Fm_dU->addTerm( u_prev*u_prev*rho + 2*rho_prev*u_prev*u + R*T_prev*rho + R*rho_prev*T );
      Fe_dU->addTerm( Cv*u_prev*T_prev*rho + Cv*rho_prev*T_prev*u + Cv*rho_prev*u_prev*T
            + 0.5*u_prev*u_prev*u_prev*rho + 1.5*rho_prev*u_prev*u_prev*u
            + R*rho_prev*T_prev*u + R*u_prev*T_prev*rho + R*rho_prev*u_prev*T );
      Km_dU->addTerm( 1*D );
      Ke_dU->addTerm( -q + D_prev*u + u_prev*D );
      MD_dU->addTerm( 1./mu*D );
      Mq_dU->addTerm( Pr/(Cp*mu)*q );
      GD_dU->addTerm( 2*u );
      Gq_dU->addTerm( -T );

      // Adjoint Terms
      adj_Cc->addTerm( vc->dy() + u_prev*vm->dy() + Cv*T_prev*ve->dy() + 0.5*u_prev*u_prev*ve->dy() );
      adj_Cm->addTerm( rho_prev*vm->dy() + rho_prev*u_prev*ve->dy() );
      adj_Ce->addTerm( Cv*rho_prev*ve->dy() );
      adj_Fc->addTerm( u_prev*vc->dx() + u_prev*u_prev*vm->dx() + R*T_prev*vm->dx() + Cv*T_prev*u_prev*ve->dx()
        + 0.5*u_prev*u_prev*u_prev*ve->dx() + R*T_prev*u_prev*ve->dx() );
      adj_Fm->addTerm( rho_prev*vc->dx() + 2*rho_prev*u_prev*vm->dx() + Cv*T_prev*rho_prev*ve->dx()
        + 0.5*rho_prev*u_prev*u_prev*ve->dx() + rho_prev*u_prev*u_prev*ve->dx() + R*T_prev*rho_prev*ve->dx() - D_prev*ve->dx() );
      adj_Fe->addTerm( R*rho_prev*vm->dx() + Cv*rho_prev*u_prev*ve->dx() + R*rho_prev*u_prev*ve->dx() );
      adj_KD->addTerm( vm->dx() + u_prev*ve->dx() );
      adj_Kq->addTerm( -ve->dx() );
      adj_MD->addTerm( 1./mu*S );
      adj_Mq->addTerm( Pr/(Cp*mu)*tau );
      adj_Gm->addTerm( 2*S->dx() );
      adj_Ge->addTerm( -tau->dx() );
      break;

      case 1:
      //    /$$$$$$                                                                           /$$     /$$
      //   /$$__  $$                                                                         | $$    |__/
      //  | $$  \__/  /$$$$$$  /$$$$$$$   /$$$$$$$  /$$$$$$   /$$$$$$  /$$    /$$  /$$$$$$  /$$$$$$   /$$  /$$$$$$  /$$$$$$$
      //  | $$       /$$__  $$| $$__  $$ /$$_____/ /$$__  $$ /$$__  $$|  $$  /$$/ |____  $$|_  $$_/  | $$ /$$__  $$| $$__  $$
      //  | $$      | $$  \ $$| $$  \ $$|  $$$$$$ | $$$$$$$$| $$  \__/ \  $$/$$/   /$$$$$$$  | $$    | $$| $$  \ $$| $$  \ $$
      //  | $$    $$| $$  | $$| $$  | $$ \____  $$| $$_____/| $$        \  $$$/   /$$__  $$  | $$ /$$| $$| $$  | $$| $$  | $$
      //  |  $$$$$$/|  $$$$$$/| $$  | $$ /$$$$$$$/|  $$$$$$$| $$         \  $/   |  $$$$$$$  |  $$$$/| $$|  $$$$$$/| $$  | $$
      //   \______/  \______/ |__/  |__/|_______/  \_______/|__/          \_/     \_______/   \___/  |__/ \______/ |__/  |__/
      //
      //
      //
      // Nonlinear Residual Terms
      Cc = rho_prev;
      Cm = m_prev;
      Ce = E_prev;
      Fc = m_prev;
      Fm = m_prev*m_prev/rho_prev + (gamma-1)*(E_prev-0.5*m_prev*m_prev/rho_prev);
      Fe = gamma*m_prev*E_prev/rho_prev-0.5*(gamma-1)*m_prev*m_prev*m_prev/(rho_prev*rho_prev);
      Km = D_prev;
      Ke = -q_prev + m_prev/rho_prev*D_prev;
      MD = 1./mu*D_prev;
      Mq = Pr/(Cp*mu)*q_prev;
      GD = 2*m_prev/rho_prev;
      Gq = -(E_prev-0.5*m_prev*m_prev/rho_prev)/(Cv*rho_prev);

      // Linearized Terms
      Cc_dU->addTerm( 1*rho );
      Cm_dU->addTerm( 1*m );
      Ce_dU->addTerm( 1*E );
      Fc_dU->addTerm( 1*m );
      Fm_dU->addTerm( 2*m_prev/rho_prev*m - m_prev*m_prev/(rho_prev*rho_prev)*rho
          + (gamma-1)*E - (gamma-1)*m_prev/rho_prev*m + 0.5*(gamma-1)*m_prev*m_prev/(rho_prev*rho_prev)*rho );
      Fe_dU->addTerm( gamma*m_prev/rho_prev*E + gamma*E_prev/rho_prev*m - gamma*m_prev*E_prev/(rho_prev*rho_prev)*rho
          - 1.5*(gamma-1)*m_prev*m_prev/(rho_prev*rho_prev)*m + (gamma-1)*m_prev*m_prev*m_prev/(rho_prev*rho_prev*rho_prev)*rho );
      Km_dU->addTerm( 1*D );
      Ke_dU->addTerm( -q + D_prev/rho_prev*m + m_prev/rho_prev*D - m_prev*D_prev/(rho_prev*rho_prev)*rho );
      MD_dU->addTerm( 1./mu*D );
      Mq_dU->addTerm( Pr/(Cp*mu)*q );
      GD_dU->addTerm( 2./rho_prev*m - 2*m_prev/(rho_prev*rho_prev)*rho );
      Gq_dU->addTerm( -1./(Cv*rho_prev)*E + E_prev/(Cv*rho_prev*rho_prev)*rho
          + m_prev/(Cv*rho_prev*rho_prev)*m - m_prev*m_prev/(Cv*rho_prev*rho_prev*rho_prev)*rho );

      // Adjoint Terms
      adj_Cc->addTerm( 1*vc->dy() );
      adj_Cm->addTerm( 1*vm->dy() );
      adj_Ce->addTerm( 1*ve->dy() );
      adj_Fc->addTerm( (-m_prev*m_prev/(rho_prev*rho_prev) + 0.5*(gamma-1)*m_prev*m_prev/(rho_prev*rho_prev))*vm->dx()
            + (-gamma*E_prev*m_prev/(rho_prev*rho_prev) + (gamma-1)*m_prev*m_prev*m_prev/(rho_prev*rho_prev*rho_prev))*ve->dx()
            + m_prev*D_prev/(rho_prev*rho_prev)*ve->dx() );
      adj_Fm->addTerm( vc->dx() + 2*m_prev/rho_prev*vm->dx() - (gamma-1)*m_prev/rho_prev*vm->dx()
            + gamma*E_prev/rho_prev*ve->dx() - 1.5*(gamma-1)*m_prev*m_prev/(rho_prev*rho_prev)*ve->dx() - D_prev/rho_prev*ve->dx() );
      adj_Fe->addTerm( (gamma-1)*vm->dx() + gamma*m_prev/rho_prev*ve->dx() );
      adj_KD->addTerm( vm->dx() + m_prev/rho_prev*ve->dx() );
      adj_Kq->addTerm( -ve->dx() );
      adj_MD->addTerm( 1./mu*S );
      adj_Mq->addTerm( Pr/(Cp*mu)*tau );
      adj_Gc->addTerm( -2*m_prev/(rho_prev*rho_prev)*S->dx() + E_prev/(Cv*rho_prev*rho_prev)*tau->dx()
            - m_prev*m_prev/(Cv*rho_prev*rho_prev*rho_prev)*tau->dx() );
      adj_Gm->addTerm( 2./rho_prev*S->dx() + m_prev/(Cv*rho_prev*rho_prev)*tau->dx() );
      adj_Ge->addTerm( -1./(Cv*rho_prev)*tau->dx() );
      break;

      case 2:
      //   /$$$$$$$$             /$$
      //  | $$_____/            | $$
      //  | $$       /$$$$$$$  /$$$$$$    /$$$$$$   /$$$$$$   /$$$$$$  /$$   /$$
      //  | $$$$$   | $$__  $$|_  $$_/   /$$__  $$ /$$__  $$ /$$__  $$| $$  | $$
      //  | $$__/   | $$  \ $$  | $$    | $$  \__/| $$  \ $$| $$  \ $$| $$  | $$
      //  | $$      | $$  | $$  | $$ /$$| $$      | $$  | $$| $$  | $$| $$  | $$
      //  | $$$$$$$$| $$  | $$  |  $$$$/| $$      |  $$$$$$/| $$$$$$$/|  $$$$$$$
      //  |________/|__/  |__/   \___/  |__/       \______/ | $$____/  \____  $$
      //                                                    | $$       /$$  | $$
      //                                                    | $$      |  $$$$$$/
      //                                                    |__/       \______/
      // define alpha from notes
      FunctionPtr zePow1 = Teuchos::rcp( new PowerFunction(-ze_prev, gamma));
      FunctionPtr alphaPow1 = Teuchos::rcp( new PowerFunction((gamma-1)/zePow1, 1./(gamma-1)));
      FunctionPtr alphaExp = Teuchos::rcp( new ExpFunction((-gamma+zc_prev-0.5*zm_prev*zm_prev/ze_prev)/(gamma-1)) );
      FunctionPtr alpha = alphaPow1*alphaExp;
      LinearTermPtr alpha_dU = Teuchos::rcp( new LinearTerm );
      alpha_dU->addTerm( alpha/(gamma-1)*(zc - zm_prev/ze_prev*zm + (0.5*zm_prev*zm_prev/(ze_prev*ze_prev)-gamma/ze_prev)*ze) );

      // Define Euler fluxes and flux jacobians
      Cc = -alpha*ze_prev;
      Cm = alpha*zm_prev;
      Ce = alpha*(1-0.5*zm_prev*zm_prev/ze_prev);
      Fc = alpha*zm_prev;
      Fm = alpha*(-zm_prev*zm_prev/ze_prev+(gamma-1));
      Fe = alpha*zm_prev/ze_prev*(0.5*zm_prev*zm_prev/ze_prev-gamma);
      Km = D_prev;
      Ke = -q_prev - zm_prev/ze_prev*D_prev;
      MD = 1./mu*D_prev;
      Mq = Pr/(Cp*mu)*q_prev;
      GD = -2*zm_prev/ze_prev;
      Gq = 1/(Cv*ze_prev);

      // Linearized Terms
      Cc_dU->addTerm( -ze_prev*alpha_dU - alpha*ze );
      Cm_dU->addTerm( zm_prev*alpha_dU + alpha*zm );
      Ce_dU->addTerm( (1-0.5*zm_prev*zm_prev/ze_prev)*alpha_dU
          + alpha*(-zm_prev/ze_prev*zm + 0.5*zm_prev*zm_prev/(ze_prev*ze_prev)*ze) );
      Fc_dU->addTerm( zm_prev*alpha_dU + alpha*zm );
      Fm_dU->addTerm( (-zm_prev*zm_prev/ze_prev+(gamma-1))*alpha_dU
          + alpha*(-2*zm_prev/ze_prev*zm + zm_prev*zm_prev/(ze_prev*ze_prev)*ze) );
      Fe_dU->addTerm( zm_prev/ze_prev*(0.5*zm_prev*zm_prev/ze_prev-gamma)*alpha_dU
          + alpha*(1.5*zm_prev*zm_prev/(ze_prev*ze_prev)*zm - zm_prev*zm_prev*zm_prev/(ze_prev*ze_prev*ze_prev)*ze
            - gamma/ze_prev*zm + gamma*zm_prev/(ze_prev*ze_prev)*ze) );
      Km_dU->addTerm( 1*D );
      Ke_dU->addTerm( -q - D_prev/ze_prev*zm - zm_prev/ze_prev*D + zm_prev*D_prev/(ze_prev*ze_prev)*ze );
      MD_dU->addTerm( 1./mu*D );
      Mq_dU->addTerm( Pr/(Cp*mu)*q );
      GD_dU->addTerm( -2./ze_prev*zm + 2*zm_prev/(ze_prev*ze_prev)*ze );
      Gq_dU->addTerm( -1/(Cv*ze_prev*ze_prev)*ze );

      // Adjoint Terms
      adj_Cc->addTerm( alpha/(gamma-1)*(-ze_prev*vc->dy() + zm_prev*vm->dy() + (1-0.5*zm_prev*zm_prev/ze_prev)*ve->dy()) );
      adj_Cm->addTerm( -alpha/(gamma-1)*zm_prev/ze_prev*(-ze_prev*vc->dy() + zm_prev*vm->dy()
            + (1-0.5*zm_prev*zm_prev/ze_prev)*ve->dy())
          + alpha*vm->dy() - alpha*zm_prev/ze_prev*ve->dy() );
      adj_Ce->addTerm(
          alpha/(gamma-1)*(0.5*zm_prev*zm_prev/(ze_prev*ze_prev)-gamma/ze_prev)*(-ze_prev*vc->dy()
            + zm_prev*vm->dy() + (1-0.5*zm_prev*zm_prev/ze_prev)*ve->dy())
          -alpha*vc->dy() + 0.5*alpha*zm_prev*zm_prev/(ze_prev*ze_prev)*ve->dy() );
      adj_Fc->addTerm( alpha/(gamma-1)*(zm_prev*vc->dx() + (-zm_prev*zm_prev/ze_prev+(gamma-1))*vm->dx()
            + (0.5*zm_prev*zm_prev/ze_prev-gamma)*zm_prev/ze_prev*ve->dx()));
      adj_Fm->addTerm( -alpha/(gamma-1)*zm_prev/ze_prev*(zm_prev*vc->dx() + (-zm_prev*zm_prev/ze_prev+(gamma-1))*vm->dx()
          + (0.5*zm_prev*zm_prev/ze_prev-gamma)*zm_prev/ze_prev*ve->dx())
        + alpha*vc->dx() - 2*alpha*zm_prev/ze_prev*vm->dx()
        + alpha*((0.5*zm_prev*zm_prev/ze_prev-gamma)/ze_prev*ve->dx() + zm_prev*zm_prev/(ze_prev*ze_prev)*ve->dx())
        + D_prev/ze_prev*ve->dx() );
      adj_Fe->addTerm(
          alpha/(gamma-1)*(0.5*zm_prev*zm_prev/(ze_prev*ze_prev)-gamma/ze_prev)*(zm_prev*vc->dx()
            + (-zm_prev*zm_prev/ze_prev + (gamma-1))*vm->dx() + (0.5*zm_prev*zm_prev/ze_prev-gamma)*zm_prev/ze_prev*ve->dx())
          + alpha*zm_prev*zm_prev/(ze_prev*ze_prev)*vm->dx()
          - alpha*(zm_prev*zm_prev/ze_prev-gamma)*zm_prev/(ze_prev*ze_prev)*ve->dx()
          - zm_prev/(ze_prev*ze_prev)*D_prev*ve->dx() );
      adj_KD->addTerm( vm->dx() - zm_prev/ze_prev*ve->dx() );
      adj_Kq->addTerm( -ve->dx() );
      adj_MD->addTerm( 1./mu*S );
      adj_Mq->addTerm( Pr/(Cp*mu)*tau );
      adj_Gm->addTerm( -2./ze_prev*S->dx() );
      adj_Ge->addTerm( 2*zm_prev/(ze_prev*ze_prev)*S->dx() - 1./(Cv*ze_prev*ze_prev)*tau->dx() );
      break;

      // default:
      // TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid formulation");
    }

    // Bilinear Form
    // S terms:
    bf->addTerm( MD_dU, S );
    bf->addTerm( GD_dU, S->dx() );
    bf->addTerm( -2*uhat, S->times_normal_x() );

    // tau terms:
    bf->addTerm( Mq_dU, tau );
    bf->addTerm( Gq_dU, tau->dx() );
    bf->addTerm( That, tau->times_normal_x());

    // vc terms:
    bf->addTerm( -Fc_dU, vc->dx());
    bf->addTerm( -Cc_dU, vc->dy());
    bf->addTerm( tc, vc);

    // vm terms:
    bf->addTerm( -Fm_dU, vm->dx());
    bf->addTerm( Km_dU, vm->dx());
    bf->addTerm( -Cm_dU, vm->dy());
    bf->addTerm( tm, vm);

    // ve terms:
    bf->addTerm( -Fe_dU, ve->dx());
    bf->addTerm( Ke_dU, ve->dx());
    bf->addTerm( -Ce_dU, ve->dy());
    bf->addTerm( te, ve);

    ////////////////////   SPECIFY RHS   ///////////////////////

    // S terms:
    rhs->addTerm( -MD * S );
    rhs->addTerm( -GD * S->dx() );

    // tau terms:
    rhs->addTerm( -Mq * tau );
    rhs->addTerm( -Gq * tau->dx() );

    // vc terms:
    rhs->addTerm( Fc * vc->dx() );
    rhs->addTerm( Cc * vc->dy() );

    // vm terms:
    rhs->addTerm( Fm * vm->dx() );
    rhs->addTerm( -Km * vm->dx() );
    rhs->addTerm( Cm * vm->dy() );

    // ve terms:
    rhs->addTerm( Fe * ve->dx() );
    rhs->addTerm( -Ke * ve->dx() );
    rhs->addTerm( Ce * ve->dy() );

    ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
    switch (norm)
    {
      // Automatic graph norm
      case 0:
      ips[slab] = bf->graphNorm();
      break;

      // Manual Graph Norm
      case 1:
      ip->addTerm( adj_MD + adj_KD );
      ip->addTerm( adj_Mq + adj_Kq );
      ip->addTerm( adj_Gc - adj_Fc - adj_Cc );
      ip->addTerm( adj_Gm - adj_Fm - adj_Cm );
      ip->addTerm( adj_Ge - adj_Fe - adj_Ce );
      ip->addTerm( vc );
      ip->addTerm( vm );
      ip->addTerm( ve );
      ip->addTerm( S );
      ip->addTerm( tau );
      break;

      // Decoupled Norm
      case 2:
      ip->addTerm( adj_MD );
      ip->addTerm( adj_Mq );
      ip->addTerm( adj_KD );
      ip->addTerm( adj_Kq );
      ip->addTerm( adj_Fc + adj_Cc );
      ip->addTerm( adj_Fm + adj_Cm );
      ip->addTerm( adj_Fe + adj_Ce );
      ip->addTerm( adj_Gc );
      ip->addTerm( adj_Gm );
      ip->addTerm( adj_Ge );
      ip->addTerm( vc );
      ip->addTerm( vm );
      ip->addTerm( ve );
      // ip->addTerm( S );
      // ip->addTerm( tau );
      break;

      default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid inner product");
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
    FunctionPtr rho0  = Teuchos::rcp( new DiscontinuousInitialCondition(xint, rhoL, rhoR) );
    FunctionPtr mom0 = Teuchos::rcp( new DiscontinuousInitialCondition(xint, uL*rhoL, uR*rhoR) );
    FunctionPtr E0    = Teuchos::rcp( new DiscontinuousInitialCondition(xint, (rhoL*Cv*TL+0.5*rhoL*uL*uL), (rhoR*Cv*TR+0.5*rhoR*uR*uR)) );
    if (problem == 6)
      E0 = Teuchos::rcp( new PulseInitialCondition(1./8, 1./Cv, 0, Function::h()) );
    // FunctionPtr rho0  = Teuchos::rcp( new RampedInitialCondition(xint, rhoL, rhoR, (xmax-xmin)/numX) );
    // FunctionPtr mom0 = Teuchos::rcp( new RampedInitialCondition(xint, uL*rhoL, uR*rhoR, (xmax-xmin)/numX) );
    // FunctionPtr E0    = Teuchos::rcp( new RampedInitialCondition(xint, (rhoL*Cv*TL+0.5*rhoL*uL*uL), (rhoR*Cv*TR+0.5*rhoR*uR*uR), (xmax-xmin)/numX) );
    bc->addDirichlet(tc, left, -rhoL*uL*one);
    bc->addDirichlet(tc, right, rhoR*uR*one);
    bc->addDirichlet(tm, left, -(rhoL*uL*uL+R*rhoL*TL)*one);
    bc->addDirichlet(tm, right, (rhoR*uR*uR+R*rhoR*TR)*one);
    // bc->addDirichlet(uhat, right, zero);
    // bc->addDirichlet(uhat, left, zero);
    bc->addDirichlet(te, left, -(rhoL*Cv*TL+0.5*rhoL*uL*uL+R*rhoL*TL)*uL*one);
    bc->addDirichlet(te, right, (rhoR*Cv*TR+0.5*rhoR*uR*uR+R*rhoR*TR)*uR*one);
    // cout << "R = " << R << " Cv = " << Cv << " Cp = " << Cp << " gamma = " << gamma << endl;
    // cout << "left " << rhoL*uL << " " << (rhoL*uL*uL+R*rhoL*TL) << " " << (rhoL*Cv*TL+0.5*rhoL*uL*uL+R*Cv*TL)*uL << endl;
    if (slab == 0)
    {
      bc->addDirichlet(tc, init, -rho0);
      bc->addDirichlet(tm, init, -mom0);
      bc->addDirichlet(te, init, -E0);
    }
    bcs.push_back(bc);
  }

  // rhs->addTerm( -1./mu*D_prev * S );
  // rhs->addTerm( -4./3*m_prev/rho_prev * S->dx() );
  // cout <<

  ////////////////////   SOLVE & REFINE   ///////////////////////
  vector< Teuchos::RCP<Solution> > solutions;
  for (int slab=0; slab < numSlabs; slab++)
  {
    Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(meshes[slab], bcs[slab], rhss[slab], ips[slab]) );
    solution->setCubatureEnrichmentDegree(cubatureEnrichment);
    solutions.push_back(solution);
    if (slab > 0)
    {
      // fhat_prev = Function::solution(fhat, solutions[slab-1]);
      FunctionPtr rho_prev = Teuchos::RCP<Function>( new PreviousSolutionFunction(backgroundFlows[slab-1], rho) );
      FunctionPtr u_prev = Teuchos::RCP<Function>( new PreviousSolutionFunction(backgroundFlows[slab-1], u) );
      FunctionPtr T_prev = Teuchos::RCP<Function>( new PreviousSolutionFunction(backgroundFlows[slab-1], T) );
      SpatialFilterPtr init = Teuchos::rcp( new ConstantYBoundary(tmins[slab]) );
      bcs[slab]->addDirichlet(tc, init, -rho_prev);
      bcs[slab]->addDirichlet(tm, init, -rho_prev*u_prev);
      bcs[slab]->addDirichlet(te, init, -Cv*rho_prev*T_prev-0.5*rho_prev*u_prev*u_prev);
    }
    meshes[slab]->registerSolution(backgroundFlows[slab]);
    meshes[slab]->registerSolution(solutions[slab]);
    double energyThreshold = 0.2; // for mesh refinements
    RefinementStrategy refinementStrategy( solution, energyThreshold );
    VTKExporter exporter(backgroundFlows[slab], meshes[slab], varFactory);
    set<int> nonlinearVars;
    nonlinearVars.insert(D->ID());
    nonlinearVars.insert(q->ID());
    nonlinearVars.insert(Uc->ID());
    nonlinearVars.insert(Um->ID());
    nonlinearVars.insert(Ue->ID());

    vector<FunctionPtr> positiveFunctions;
    vector<FunctionPtr> positiveUpdates;
    switch (formulation)
    {
      case 0:
      positiveFunctions.push_back(Function::solution(rho,backgroundFlows[slab]));
      positiveUpdates.push_back(Function::solution(rho,solution));
      positiveFunctions.push_back(Function::solution(T,backgroundFlows[slab]));
      positiveUpdates.push_back(Function::solution(T,solution));
      break;
      case 1:
      positiveFunctions.push_back(Function::solution(rho,backgroundFlows[slab]));
      positiveUpdates.push_back(Function::solution(rho,solution));
      positiveFunctions.push_back(Function::solution(E,backgroundFlows[slab]));
      positiveUpdates.push_back(Function::solution(E,solution));
      break;
      case 2:
      positiveFunctions.push_back(-Function::solution(ze,backgroundFlows[slab]));
      positiveUpdates.push_back(-Function::solution(ze,solution));
      break;

      default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid formulation");
    }

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
        // amount of enriching of grid points on which to ensure positivity
        int posEnrich = 5;
        if (useLineSearch)
        {
          double lineSearchFactor = .5;
          double eps = .001;
          bool isPositive=true;
          for (int i=0; i < positiveFunctions.size(); i++)
          {
            FunctionPtr temp = positiveFunctions[i] + alpha*positiveUpdates[i] - Function::constant(eps);
            isPositive = isPositive and temp->isPositive(meshes[slab],posEnrich);
          }
          int iter = 0; int maxIter = 20;
          while (!isPositive && iter < maxIter)
          {
            alpha = alpha*lineSearchFactor;
            isPositive = true;
            for (int i=0; i < positiveFunctions.size(); i++)
            {
              FunctionPtr temp = positiveFunctions[i] + alpha*positiveUpdates[i] - Function::constant(eps);
              isPositive = isPositive and temp->isPositive(meshes[slab],posEnrich);
            }
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
        if (alpha <= 1e-3)
          break;
      }
      if (commRank == 0)
        cout << endl;

      if (commRank == 0)
      {
        stringstream outfile;
        outfile << problemName << formulation << "_" << norm << "_" << slab << "_" << refIndex;
        exporter.exportSolution(outfile.str());
        FunctionPtr density;
        FunctionPtr velocity;
        FunctionPtr temperature;
        stringstream denFile, velFile, tempFile;
        denFile << problemName << "-density" << formulation << "_" << norm << "_" << slab << "_" << refIndex;
        velFile << problemName << "-velocity" << formulation << "_" << norm << "_" << slab << "_" << refIndex;
        tempFile << problemName << "-temperature" << formulation << "_" << norm << "_" << slab << "_" << refIndex;
        FunctionPtr rho_prev = Function::solution(rho,backgroundFlows[slab]);
        FunctionPtr m_prev = Function::solution(m,backgroundFlows[slab]);
        FunctionPtr E_prev = Function::solution(E,backgroundFlows[slab]);
        FunctionPtr zc_prev = Function::solution(zc,backgroundFlows[slab]);
        FunctionPtr zm_prev = Function::solution(zm,backgroundFlows[slab]);
        FunctionPtr ze_prev = Function::solution(ze,backgroundFlows[slab]);
        switch (formulation)
        {
          case 0:
          density = Function::solution(rho,backgroundFlows[slab]);
          velocity = Function::solution(u,backgroundFlows[slab]);
          temperature = Function::solution(T,backgroundFlows[slab]);
          break;
          case 1:
          density = rho_prev;
          velocity = m_prev/rho_prev;
          temperature = (E_prev-0.5*m_prev*m_prev/rho_prev)/(Cv*rho_prev);
          break;
          case 2:
          FunctionPtr zePow1 = Teuchos::rcp( new PowerFunction(-ze_prev, gamma));
          FunctionPtr alphaPow1 = Teuchos::rcp( new PowerFunction((gamma-1)/zePow1, 1./(gamma-1)));
          FunctionPtr alphaExp = Teuchos::rcp( new ExpFunction((-gamma+zc_prev-0.5*zm_prev*zm_prev/ze_prev)/(gamma-1)) );
          FunctionPtr alpha = alphaPow1*alphaExp;
          density = -alpha*ze_prev;
          velocity = -zm_prev/ze_prev;
          temperature = -1/(Cv*ze_prev);
          break;
        }
        exporter.exportFunction(density,denFile.str());
        exporter.exportFunction(velocity,velFile.str());
        exporter.exportFunction(temperature,tempFile.str());
        // FunctionPtr D_prev = Function::solution(D,backgroundFlows[slab]);
        // FunctionPtr dDdx = Function::solution(D->dx(),backgroundFlows[slab]);
        // FunctionPtr dudx = Function::solution(u->dx(),backgroundFlows[slab]);
        // FunctionPtr div_indicator = Teuchos::rcp( new DivergenceIndicator(rho_prev, rho_prev, D_prev) );
        // exporter.exportFunction(dDdx,"dDdx"+Teuchos::toString(refIndex));
        // exporter.exportFunction(dudx,"dudx"+Teuchos::toString(refIndex));
        // FunctionPtr errorFunction = Teuchos::rcp( new ErrorFunction(solution) );
        // exporter.exportFunction(errorFunction,"Error"+Teuchos::toString(refIndex));
      }

      if (refIndex < numRefs)
      {
        refinementStrategy.refine(commRank==0);
        double newRamp = (xmax-xmin)/(numX*pow(2., refIndex+1));
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
  }

  return 0;
}

