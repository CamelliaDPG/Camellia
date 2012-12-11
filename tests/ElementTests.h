#ifndef ELEMENTTESTS_H
#define ELEMENTTESTS_H

#include "gtest/gtest.h"
#include "TestUtil.h"

#include "Mesh.h"
#include "InnerProductScratchPad.h"

class ElementTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    FieldContainer<double> _testPoints1D;

    Teuchos::RCP<Mesh> _mesh; // a 2x2 mesh refined in SW, and then in the SE of the SW
    Teuchos::RCP<BF> _confusionBF; // standard confusion bilinear form
    ElementPtr _sw, _se, _nw, _ne, _sw_se, _sw_ne, _sw_se_se, _sw_se_ne;
};

#endif /* end of include guard: ELEMENTTESTS_H */
