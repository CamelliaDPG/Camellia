#ifndef LINEARTERMTESTS_H
#define LINEARTERMTESTS_H

#include "gtest/gtest.h"
#include "TestUtil.h"

#include "LinearTerm.h"
#include "VarFactory.h"
#include "Mesh.h"
#include "BasisCache.h"
#include "BF.h"
#include "IP.h"
#include "BCEasy.h"
#include "RHSEasy.h"
#include "RieszRep.h"
#include "PreviousSolutionFunction.h"

typedef pair< FunctionPtr, VarPtr > LinearSummand;

class LinearTermTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    VarFactory varFactory;

    VarPtr v1, v2, v3; // HGRAD members (test variables)
    VarPtr q1, q2, q3; // HDIV members (test variables)
    VarPtr u1, u2, u3; // L2 members (trial variables)
    VarPtr u1_hat, u2_hat; // trace variables
    VarPtr u3_hat_n; // flux variable
    FunctionPtr sine_x, cos_y;
    Teuchos::RCP<Mesh> mesh;

    DofOrderingPtr trialOrder, testOrder;

    BasisCachePtr basisCache;

    BFPtr bf;
};

void transposeFieldContainer(FieldContainer<double> &fc){
  // this is NOT meant for production code.  Could do the transpose in place if we were concerned with efficiency.
  FieldContainer<double> fcCopy = fc;
  int numCells = fc.dimension(0);
  int dim1 = fc.dimension(1);
  int dim2 = fc.dimension(2);
  fc.resize(numCells,dim2,dim1);
  for (int i=0; i<numCells; i++) {
    for (int j=0; j<dim1; j++) {
      for (int k=0; k<dim2; k++) {
        fc(i,k,j) = fcCopy(i,j,k);
      }
    }
  }
}

class Sine_x : public Function {
  public:
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex,ptIndex) = sin(x);
        }
      }
    }
};


class Cosine_y : public Function {
  public:
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex,ptIndex) = cos(y);
        }
      }
    }
};

bool checkLTSumConsistency(LinearTermPtr a, LinearTermPtr b, DofOrderingPtr dofOrdering, BasisCachePtr basisCache) {
  double tol = 1e-14;

  int numCells = basisCache->cellIDs().size();
  int numDofs = dofOrdering->totalDofs();
  bool forceBoundaryTerm = false;
  FieldContainer<double> aValues(numCells,numDofs), bValues(numCells,numDofs), sumValues(numCells,numDofs);
  a->integrate(aValues,dofOrdering,basisCache,forceBoundaryTerm);
  b->integrate(bValues,dofOrdering,basisCache,forceBoundaryTerm);
  (a+b)->integrate(sumValues, dofOrdering, basisCache, forceBoundaryTerm);

  int size = aValues.size();

  for (int i=0; i<size; i++) {
    double expectedValue = aValues[i] + bValues[i];
    double diff = abs( expectedValue - sumValues[i] );
    if (diff > tol) {
      return false;
    }
  }
  return true;
}

#endif /* end of include guard: LINEARTERMTESTS_H */
