#ifndef LOCALCONSERVATIONTESTS_H
#define LOCALCONSERVATIONTESTS_H

#include "gtest/gtest.h"

#include "BasicConfusionProblem.h"

class SteadyConservationTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    BasicConfusionProblem cp;
};

class TransientConservationTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    BasicConfusionProblem cp;
};

#endif /* end of include guard: LOCALCONSERVATIONTESTS_H */
