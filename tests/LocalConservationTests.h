#ifndef LOCALCONSERVATIONTESTS_H
#define LOCALCONSERVATIONTESTS_H

#include "gtest/gtest.h"

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"

class OneTermConservationTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    VarFactory varFactory; 
    VarPtr beta_n_u_hat;
    VarPtr u;
    BFPtr bf;
    FunctionPtr f;
    MeshPtr mesh;
    SolutionPtr solution;  
};

class MixedTermConservationTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    VarFactory varFactory; 
    VarPtr beta_n_u_hat;
    VarPtr u;
    BFPtr bf;
    FunctionPtr f;
    MeshPtr mesh;
    SolutionPtr solution;  
    FunctionPtr invDt;    
};


#endif /* end of include guard: LOCALCONSERVATIONTESTS_H */
