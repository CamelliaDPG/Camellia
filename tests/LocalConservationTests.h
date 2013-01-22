#ifndef LOCALCONSERVATIONTESTS_H
#define LOCALCONSERVATIONTESTS_H

#include "gtest/gtest.h"

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"

class LocalConservationTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    VarFactory varFactory; 
    VarPtr beta_n_u_hat;
    VarPtr u;
    BFPtr bf;
    FunctionPtr f;
    MeshPtr mesh;
    SolutionPtr prevTimeFlow;  
    SolutionPtr flowResidual;  
    SolutionPtr solution;  
};


#endif /* end of include guard: LOCALCONSERVATIONTESTS_H */
