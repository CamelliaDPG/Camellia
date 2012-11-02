#ifndef FUNCTIONTESTS_H
#define FUNCTIONTESTS_H

#include "gtest/gtest.h"
#include "TestUtil.h"

#include "Mesh.h"
#include "InnerProductScratchPad.h"
#include "BasisCache.h"

/*
   For now, this is sort of a grab bag for tests against all the "new-style"
   (a.k.a. "ScratchPad") items.  There are some tests against these elsewhere
   (as of this writing, there's one against RHSEasy in RHSTests), and other
   places are probably the best spot for tests that compare the results of the
   old code to that of the new--as with RHS, the sensible place to add such tests
   is where we already test the old code.

   All that to say, the tests here are glommed together for convenience and
   quick test development.  Once they grow to a certain size, it may be better
   to split them apart...
*/

class FunctionTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    void checkFunctionsAgree(FunctionPtr f1, FunctionPtr f2, BasisCachePtr basisCache);

    Teuchos::RCP<Mesh> _spectralConfusionMesh; // 1x1 mesh, H1 order = 1, pToAdd = 0
    Teuchos::RCP<BF> _confusionBF; // standard confusion bilinear form
    FieldContainer<double> _testPoints;
    ElementTypePtr _elemType;
    BasisCachePtr _basisCache;
};

// "previous solution" value for u -- what Burgers would see, according to InitialGuess.h, in first linear step
class UPrev : public Function {
public:
  UPrev() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex,ptIndex) = 1 - 2*x;
      }
    }
  }
};

#endif /* end of include guard: FUNCTIONTESTS_H */
