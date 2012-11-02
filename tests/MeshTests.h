#ifndef TESTMESH_H
#define TESTMESH_H

#include "gtest/gtest.h"

#include "InnerProductScratchPad.h"
#include "BasisFactory.h"
#include "MeshTestUtility.h"

class MeshTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    Teuchos::RCP<BF> bilinearForm;
};

#endif /* end of include guard: TESTMESH_H */
