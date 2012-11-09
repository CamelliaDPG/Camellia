#ifndef LOCALCONSERVATIONTESTS_H
#define LOCALCONSERVATIONTESTS_H

#include "gtest/gtest.h"

#include "BasicConfusionProblem.h"

class LocalConservationTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    BasicConfusionProblem confusionProb;
};

#endif /* end of include guard: LOCALCONSERVATIONTESTS_H */
