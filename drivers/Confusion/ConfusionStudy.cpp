#include "ConfusionManufacturedSolution.h"
#include "ConfusionBilinearForm.h"

#include "HConvergenceStudy.h"
#include "OptimalInnerProduct.h"

using namespace std;

int main(int argc, char *argv[])
{
  int polyOrder = 2, minLogElements = 2, maxLogElements = 5;
  int pToAdd = 2; // for tests
  bool useTriangles = true;

  if (argc == 2)   // one command-line argument: interpret as polyOrder
  {
    polyOrder = atoi(argv[1]);
  }
  else if (argc == 4)     // polyOrder minLogElements maxLogElements
  {
    polyOrder = atoi(argv[1]);
    minLogElements = atoi(argv[2]);
    maxLogElements = atoi(argv[3]);
  }
  else if (argc == 5)     // polyOrder minLogElements maxLogElements NOTRIANGLES
  {
    polyOrder = atoi(argv[1]);
    minLogElements = atoi(argv[2]);
    maxLogElements = atoi(argv[3]);
    useTriangles = false;
  }
  // compute the order for traces (the order for H^1 space):
  int H1Order = polyOrder + 1;

  // print out the polyOrder and the test order
  int testOrder = H1Order + pToAdd;
  cout << "ConfusionStudy: running for polyOrder " << polyOrder << ", testOrder " << testOrder << endl;

  double epsilon = 1e-2;
  double beta_x  = 2.0;
  double beta_y  = 1.0;

  Teuchos::RCP<ConfusionManufacturedSolution> solution =
    Teuchos::rcp( new ConfusionManufacturedSolution(epsilon,beta_x,beta_y) );

  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new OptimalInnerProduct( solution->bilinearForm() ) );

  FieldContainer<double> quadPoints(4,2);

  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;

  bool useRandomRefinements = false;

  HConvergenceStudy study(solution,
                          solution->bilinearForm(),
                          solution->ExactSolution::rhs(),
                          solution->bc(), ip,
                          minLogElements, maxLogElements,
                          H1Order, pToAdd, useRandomRefinements,
                          useTriangles);

  study.solve(quadPoints);

  // note: for writing to file to work, will need to make a confusion directory...
  ostringstream filePathPrefix;
  filePathPrefix << "confusion/u_p" << polyOrder;
  study.writeToFiles(filePathPrefix.str(),ConfusionBilinearForm::U);

  // clear filePathPrefix
  filePathPrefix.str("");

  filePathPrefix << "confusion/sigma1_p" << polyOrder;
  study.writeToFiles(filePathPrefix.str(),ConfusionBilinearForm::SIGMA_1);

  // clear filePathPrefix
  filePathPrefix.str("");

  filePathPrefix << "confusion/sigma2_p" << polyOrder;
  study.writeToFiles(filePathPrefix.str(),ConfusionBilinearForm::SIGMA_2);
}
