//  ConfusionDriver.cpp
//  Driver for Conservative Convection-Diffusion
//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "CamelliaConfig.h"
#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"
#include "CheckConservation.h"
#include "SolutionExporter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

double halfwidth;

class EpsilonScaling : public hFunction {
  double _epsilon;
public:
  EpsilonScaling(double epsilon) {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h) {
    double scaling = min(_epsilon/(h*h), 1.0);
    // since this is used in inner product term a like (a,a), take square root
    return sqrt(scaling);
  }
};

class ZeroMeanScaling : public hFunction {
  public:
  double value(double x, double y, double h) {
    return 1.0/(h*h);
  }
};

class Inflow: public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool xMatch = (abs(x+halfwidth) < tol) ;
      return xMatch;
    }
};

class Outflow: public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool xMatch = (abs(x-halfwidth) < tol) ;
      return xMatch;
    }
};

class LeadingWedge: public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool match = abs(1./halfwidth*x-y) < tol;
      return match;
    }
};

class TrailingWedge: public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool match = abs(1./halfwidth*x+y) < tol;
      return match;
    }
};

class Top: public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool yMatch = (abs(y-halfwidth) < tol) ;
      return yMatch;
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
  double epsilon = args.Input<double>("--epsilon", "diffusion parameter");
  int numRefs = args.Input<int>("--numRefs", "number of refinement steps");
  bool enforceLocalConservation = args.Input<bool>("--conserve", "enforce local conservation");
  bool graphNorm = args.Input<bool>("--graphNorm", "use the graph norm rather than robust test norm");

  // Optional arguments (have defaults)
  halfwidth = args.Input("--halfwidth", "half the width of the wedge", 0.5);
  bool allQuads = args.Input("--allQuads", "use only quads in mesh", false);
  bool zeroL2 = args.Input("--zeroL2", "take L2 term on v in robust norm to zero", false);
  args.Process();

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("uhat");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("fhat");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma = varFactory.fieldVar("sigma", VECTOR_L2);

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  bf->addTerm(sigma / epsilon, tau);
  bf->addTerm(u, tau->div());
  bf->addTerm(-uhat, tau->dot_normal());
  
  // v terms:
  bf->addTerm( sigma, v->grad() );
  bf->addTerm( beta * u, - v->grad() );
  bf->addTerm( beta_n_u_minus_sigma_n, v);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = Teuchos::rcp(new IP);
  if (graphNorm)
  {
    ip = bf->graphNorm();
  }
  else
  {
    // robust test norm
    FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
    FunctionPtr h2_scaling = Teuchos::rcp( new ZeroMeanScaling ); 
    if (!zeroL2)
      ip->addTerm( ip_scaling * v );
    ip->addTerm( sqrt(epsilon) * v->grad() );
    // Weight these two terms for inflow
    ip->addTerm( beta * v->grad() );
    ip->addTerm( tau->div() );
    ip->addTerm( ip_scaling/sqrt(epsilon) * tau );
    if (zeroL2)
      ip->addZeroMeanTerm( h2_scaling*v );
  }
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp( new PenaltyConstraints );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );

  SpatialFilterPtr inflow = Teuchos::rcp( new Inflow );
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  bc->addDirichlet(beta_n_u_minus_sigma_n, inflow, zero);

  SpatialFilterPtr leadingWedge = Teuchos::rcp( new LeadingWedge );
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  bc->addDirichlet(uhat, leadingWedge, one);

  SpatialFilterPtr trailingWedge = Teuchos::rcp( new TrailingWedge );
  bc->addDirichlet(beta_n_u_minus_sigma_n, trailingWedge, beta*n*one);
  // bc->addDirichlet(uhat, trailingWedge, one);

  SpatialFilterPtr top = Teuchos::rcp( new Top );
  bc->addDirichlet(uhat, top, zero);
  // bc->addDirichlet(beta_n_u_minus_sigma_n, top, zero);

  SpatialFilterPtr outflow = Teuchos::rcp( new Outflow );
  pc->addConstraint(beta*uhat->times_normal() - beta_n_u_minus_sigma_n == zero, outflow);
  
  ////////////////////   BUILD MESH   ///////////////////////
  int H1Order = 3, pToAdd = 2;
  // define nodes for mesh
  vector< FieldContainer<double> > vertices;
  FieldContainer<double> pt(2);
  vector< vector<int> > elementIndices;
  vector<int> q(4);
  vector<int> t(3);

  if (allQuads)
  {
    pt(0) = -halfwidth; pt(1) = -1;
    vertices.push_back(pt);
    pt(0) =  0;         pt(1) =  0;
    vertices.push_back(pt);
    pt(0) =  halfwidth; pt(1) = -1;
    vertices.push_back(pt);
    pt(0) =  halfwidth; pt(1) =  halfwidth;
    vertices.push_back(pt);
    pt(0) =  0;         pt(1) =  halfwidth;
    vertices.push_back(pt);
    pt(0) = -halfwidth; pt(1) =  halfwidth;
    vertices.push_back(pt);

    q[0] = 0; q[1] = 1; q[2] = 4; q[3] = 5;
    elementIndices.push_back(q);
    q[0] = 1; q[1] = 2; q[2] = 3; q[3] = 4;
    elementIndices.push_back(q);
  }
  else
  {
    pt(0) = -halfwidth; pt(1) = -1;
    vertices.push_back(pt);
    pt(0) =  0;         pt(1) =  0;
    vertices.push_back(pt);
    pt(0) =  halfwidth; pt(1) = -1;
    vertices.push_back(pt);
    pt(0) =  halfwidth; pt(1) =  0;
    vertices.push_back(pt);
    pt(0) =  halfwidth; pt(1) =  halfwidth;
    vertices.push_back(pt);
    pt(0) =  0;         pt(1) =  halfwidth;
    vertices.push_back(pt);
    pt(0) = -halfwidth; pt(1) =  halfwidth;
    vertices.push_back(pt);
    pt(0) = -halfwidth; pt(1) =  0;
    vertices.push_back(pt);

    t[0] = 0; t[1] = 1; t[2] = 7;
    elementIndices.push_back(t);
    t[0] = 1; t[1] = 2; t[2] = 3;
    elementIndices.push_back(t);
    q[0] = 1; q[1] = 3; q[2] = 4; q[3] = 5;
    elementIndices.push_back(q);
    q[0] = 7; q[1] = 1; q[2] = 5; q[3] = 6;
    elementIndices.push_back(q);
  }

  Teuchos::RCP<Mesh> mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, bf, H1Order, pToAdd) );  
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  solution->setFilter(pc);

  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == zero);
  }
  
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  VTKExporter exporter(solution, mesh, varFactory);
  ofstream errOut;
  if (commRank == 0)
    errOut.open("singularwedge_err.txt");
  
  for (int refIndex=0; refIndex<=numRefs; refIndex++){    
    solution->solve(false);

    double energy_error = solution->energyErrorTotal();
    if (commRank==0){
      stringstream outfile;
      outfile << "singularwedge_" << refIndex;
      exporter.exportSolution(outfile.str());

      // Check local conservation
      FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_minus_sigma_n) );
      FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
      Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, zero, varFactory, mesh);
      cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
        << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;

      errOut << mesh->numGlobalDofs() << " " << energy_error << " "
        << fluxImbalances[0] << " " << fluxImbalances[1] << " " << fluxImbalances[2] << endl;
    }

    if (refIndex < numRefs)
    {
      // refinementStrategy.refine(commRank==0); // print to console on commRank 0
      vector<int> cellsToRefine;
      vector<int> cells_h;
      vector<int> cells_p;
      refinementStrategy.getCellsAboveErrorThreshhold(cellsToRefine);
      for (int i=0; i < cellsToRefine.size(); i++)
        if (sqrt(mesh->getCellMeasure(cellsToRefine[i])) < 5e-4)
        {
          int pOrder = mesh->cellPolyOrder(cellsToRefine[i]);
          if (allQuads)
            cells_p.push_back(cellsToRefine[i]);
          else
            if (pOrder < 8)
              cells_p.push_back(cellsToRefine[i]);
            else
              cout << "Reached cell size and polynomial order limits" << endl;
          //   cells_h.push_back(cellsToRefine[i]);
        }
        else
          cells_h.push_back(cellsToRefine[i]);
      refinementStrategy.pRefineCells(mesh, cells_p);
      refinementStrategy.hRefineCells(mesh, cells_h);
    }
  }
  if (commRank == 0)
    errOut.close();
  
  return 0;
}
