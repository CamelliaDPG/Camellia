#ifndef TRANSIENTTESTS_H
#define TRANSIENTTESTS_H

#include "gtest/gtest.h"

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"

class TransientTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    void stepToSteadyState();

    VarFactory varFactory; 
    VarPtr beta_n_u_hat;
    VarPtr u;
    BFPtr bf;
    MeshPtr mesh;
    SolutionPtr prevTimeFlow;  
    SolutionPtr flowResidual;  
    SolutionPtr solution;  
};

#endif /* end of include guard: TRANSIENTTESTS_H */
