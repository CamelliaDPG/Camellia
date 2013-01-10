#ifndef MPIWRAPPERTESTS_H
#define MPIWRAPPERTESTS_H

#include "gtest/gtest.h"

#include "MPIWrapper.h"

class MPIWrapperTests : public ::testing::Test {
  protected:
    virtual void SetUp();
};

#endif /* end of include guard: MPIWRAPPERTESTS_H */
