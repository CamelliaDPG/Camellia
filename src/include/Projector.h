#ifndef PROJECTOR
#define PROJECTOR

//#include "DPGInnerProduct.h"
#include "L2InnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "AbstractFunction.h"

using namespace Intrepid;
using namespace std;

class Projector{
 protected:
  //  Teuchos::RCP<DPGInnerProduct> _ip;
  Teuchos::RCP<L2InnerProduct> _ip;
 public:
  //  Projector(Teuchos::RCP<DPGInnerProduct>ip); //constructor
  Projector(Teuchos::RCP<L2InnerProduct>ip); //constructor

  void projectFunction(Teuchos::RCP<Solution> solution, Teuchos::RCP<Mesh> mesh, int trialID, Teuchos::RCP<AbstractFunction>fxn); // should project function "fxn" onto global solution. may call projectFunctionOntoSingleCell in the future

  //  void projectFunctionOntoSingleCell(int cellID, Teuchos::RCP<Solution> solution, Teuchos::RCP<Mesh> mesh, int trialID, Teuchos::RCP<AbstractFunction>fxn); // should project function "fxn" onto a given local trial solution on a single cell
  
};
#endif
