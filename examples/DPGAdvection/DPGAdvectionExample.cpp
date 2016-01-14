#include "BC.h"
#include "BF.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "GlobalDofAssignment.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"
#include "SimpleFunction.h"
#include "Solution.h"
#include "Solver.h"
#include "SpatialFilter.h"
#include "VarFactory.h"

#include "Epetra_FECrsMatrix.h"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"

#include "Intrepid_FunctionSpaceTools.hpp"

/*
 
 Applying Camellia with *DPG* to the same DG example problem in DGAdvectionExample.
 
 See
 
 https://dealii.org/developer/doxygen/deal.II/step_12.html
 
 for a deal.II tutorial discussion of this example in the DG context.
 
 */

using namespace Camellia;
using namespace std;

// beta: 1/|x| (-x2, x1)

class BetaX : public SimpleFunction<double>
{
public:
  double value(double x, double y)
  {
    double mag = sqrt(x*x + y*y);
    if (mag > 0)
      return (1.0 / mag) * -y;
    else
      return 0;
  }
};

class BetaY : public SimpleFunction<double>
{
public:
  double value(double x, double y)
  {
    double mag = sqrt(x*x + y*y);
    if (mag > 0)
      return (1.0 / mag) * x;
    else
      return 0;
  }
};

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  int polyOrder = 1;
  int horizontalElements = 2, verticalElements = 2;
  int pToAddTest = 2; // using at least spaceDim seems to be important for pure convection

  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("delta_k", &pToAddTest);
  cmdp.setOption("horizontalElements", &horizontalElements);
  cmdp.setOption("verticalElements", &verticalElements);
  
  string convectiveDirectionChoice = "CCW"; // counter-clockwise, the default.  Other options: left, right, up, down
  cmdp.setOption("convectiveDirection", &convectiveDirectionChoice);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  FunctionPtr beta_x, beta_y;
  
  SpatialFilterPtr unitInflow, zeroInflow;
  
  if (convectiveDirectionChoice == "CCW")
  {
    beta_x = Teuchos::rcp( new BetaX );
    beta_y = Teuchos::rcp( new BetaY );
    
    // set g = 1 on [0,0.5] x {0}
    unitInflow = SpatialFilter::matchingY(0) & (! SpatialFilter::greaterThanX(0.5));
    
    // set g = 0 on {1} x [0,1], (0.5,1.0] x {0}
    zeroInflow = SpatialFilter::matchingX(1) | (SpatialFilter::matchingY(0) & SpatialFilter::greaterThanX(0.5));
  }
  else if (convectiveDirectionChoice == "left")
  {
    beta_x = Function::constant(-1);
    beta_y = Function::zero();
    unitInflow = SpatialFilter::matchingX(1.0) & SpatialFilter::lessThanY(0.5);
    zeroInflow = SpatialFilter::matchingX(1.0) & !SpatialFilter::lessThanY(0.5);
  }
  else if (convectiveDirectionChoice == "right")
  {
    beta_x = Function::constant(1.0);
    beta_y = Function::constant(0.0);
    
    unitInflow = SpatialFilter::matchingX(0.0) & SpatialFilter::lessThanY(0.5); // | SpatialFilter::matchingY(0.0);
    zeroInflow = SpatialFilter::matchingX(0.0) & !SpatialFilter::lessThanY(0.5);
  }
  else if (convectiveDirectionChoice == "up")
  {
    beta_x = Function::zero();
    beta_y = Function::constant(1);
    
    unitInflow = SpatialFilter::matchingY(0.0);
    zeroInflow = !SpatialFilter::allSpace();
  }
  else if (convectiveDirectionChoice == "down")
  {
    beta_x = Function::zero();
    beta_y = Function::constant(-1);
    
    unitInflow = SpatialFilter::matchingY(1.0) & SpatialFilter::lessThanX(0.5);
    zeroInflow = SpatialFilter::matchingY(1.0) & !SpatialFilter::lessThanX(0.5);
  }
  else
  {
    cout << "convective direction " << convectiveDirectionChoice << " is not a supported option.\n";
  }
  
  FunctionPtr beta = Function::vectorize(beta_x, beta_y);
    
  VarFactoryPtr vf = VarFactory::varFactory();
  
  VarPtr u = vf->fieldVar("u");
  FunctionPtr n = Function::normal();
  FunctionPtr parity = Function::sideParity();
  VarPtr u_n = vf->fluxVar("u_n", u * beta * n * parity, L2);
  VarPtr v = vf->testVar("v", HGRAD);
  
  BFPtr bf = BF::bf(vf);
  bf->addTerm(-u, beta * v->grad());
  bf->addTerm(u_n, v);
  
  BCPtr bc = BC::bc();
  
  bc->addDirichlet(u_n, unitInflow, Function::constant(1.0) * beta * n * parity);
  bc->addDirichlet(u_n, zeroInflow, Function::zero() * beta * n * parity);
  
  /******* Define the mesh ********/
  // solve on [0,1]^2 with 8x8 initial elements
  double width = 1.0, height = 1.0;
  bool divideIntoTriangles = false;
  double x0 = 0.0, y0 = 0.0;
  int H1Order = polyOrder + 1; // polyOrder refers to the order of the fields, which here are L^2
  MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, H1Order, pToAddTest,
                                              width, height,
                                              horizontalElements, verticalElements,
                                              divideIntoTriangles, x0, y0);
  
  RHSPtr rhs = RHS::rhs(); // zero forcing
  IPPtr ip = IP::ip(); // manually construct "graph" norm to minimize L^2 error of (beta u).  (bf->graphNorm() minimizes L^2 error of u.)
  ip->addTerm(v->grad());
  ip->addTerm(v);
  SolutionPtr soln = Solution::solution(mesh, bc, rhs, ip);
  
  SolverPtr solver = Solver::getDirectSolver();
  int solveSuccess = soln->solve(solver);
  if (solveSuccess != 0)
    cout << "solve returned with error code " << solveSuccess << endl;
  
  ostringstream name;
  name << "DPGAdvectionExample_" << convectiveDirectionChoice;
  
  HDF5Exporter exporter(mesh, name.str(), ".");
  
  exporter.exportSolution(soln,0); // 0 for the refinement number (or time step)
  
  return 0;
}