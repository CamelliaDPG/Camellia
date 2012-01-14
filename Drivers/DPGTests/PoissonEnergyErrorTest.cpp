#include "PoissonExactSolution.h"
#include "PoissonBilinearForm.h"

#include "HConvergenceStudy.h"
#include "MathInnerProduct.h"
#include "OptimalInnerProduct.h"
#include "Solution.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

using namespace std;

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  // TODO: figure out the right thing to do here...
  // may want to modify argc and argv before we make the following call:
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
  cout << "rank: " << rank << endl;
  cout << "numProcs: " << numProcs << endl;
#else
#endif
  
  int polyOrder = 1;
  bool useTriangles = false;

  int pToAdd = 1; // for tests
  bool useConformingTraces = true;
  Teuchos::RCP<PoissonExactSolution> mySolution = 
  Teuchos::rcp( new PoissonExactSolution(PoissonExactSolution::POLYNOMIAL, 
                                        polyOrder, useConformingTraces) );
  mySolution->setUseSinglePointBCForPHI(false); // impose zero-mean constraint
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new MathInnerProduct( mySolution->bilinearForm() ) );
  //  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new OptimalInnerProduct( mySolution->bilinearForm() ) );
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;

  int H1Order = polyOrder + 1;
  int horizontalCells = 1, verticalCells = 1;
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, 
                                                mySolution->ExactSolution::bilinearForm(), H1Order, H1Order+pToAdd, useTriangles);
  
  //  Solution solution(mesh, mySolution->bc(), mySolution->ExactSolution::rhs(), ip);
  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp(new Solution(mesh, mySolution->bc(), mySolution->ExactSolution::rhs(), ip));
  solution->solve();

  int cubDegree = polyOrder; // for error computations
  double phiError =  mySolution->L2NormOfError(*solution, PoissonBilinearForm::PHI,   cubDegree);
  double psi1Error = mySolution->L2NormOfError(*solution, PoissonBilinearForm::PSI_1, cubDegree);
  double psi2Error = mySolution->L2NormOfError(*solution, PoissonBilinearForm::PSI_2, cubDegree);
  
  double phiHatError =  mySolution->L2NormOfError(*solution, PoissonBilinearForm::PHI_HAT, cubDegree+1);
  //double psiHatError =  mySolution->L2NormOfError(*solution, PoissonBilinearForm::PSI_HAT_N, cubDegree);
  
  string meshType = (useTriangles) ? "triangular" : "quad";
  
  cout << horizontalCells << "x" << verticalCells << " " << meshType << " mesh, phi error: " << phiError << endl;
  cout << horizontalCells << "x" << verticalCells << " " << meshType << " mesh, psi1 error: " << psi1Error << endl;
  cout << horizontalCells << "x" << verticalCells << " " << meshType << " mesh, psi2 error: " << psi2Error << endl;
  cout << horizontalCells << "x" << verticalCells << " " << meshType << " mesh, phiHat error: " << phiHatError << endl;
  //cout << horizontalCells << "x" << verticalCells << " " << meshType << " mesh, psiHat_n error: " << psiHatError << endl;

  double totalEnergyErrorSq;  
  map<int,double> energyErr;  
  solution->energyError(energyErr);
  int numActiveCell = energyErr.size();
  totalEnergyErrorSq = 0.0;
  map<int,double>::iterator energyErrIt;
  for (energyErrIt = energyErr.begin(); energyErrIt != energyErr.end(); energyErrIt++) {
    double err = energyErrIt->second;
    totalEnergyErrorSq += err * err;
  }
  cout << "Energy error: " << sqrt(totalEnergyErrorSq) << endl;  

  //solution->writeToFile(PoissonBilinearForm::PHI, "PoissonEnergyErrorTest_phi.dat");

  return 0;
 
}
