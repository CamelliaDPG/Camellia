#ifndef BASISCACHETESTS_H
#define BASISCACHETESTS_H

#include "gtest/gtest.h"
#include "TestUtil.h"

#include "Mesh.h"
#include "InnerProductScratchPad.h"
#include "BasisCache.h"

class BasisCacheTests : public ::testing::Test {
  protected:
    virtual void SetUp();

    Teuchos::RCP<Mesh> _spectralConfusionMesh; // 1x1 mesh, H1 order = 1, pToAdd = 0
    Teuchos::RCP<BF> _confusionBF; // standard confusion bilinear form
    VarPtr _uhat_confusion; // confusion variable u_hat
    FieldContainer<double> _testPoints;
    ElementTypePtr _elemType;
    BasisCachePtr _basisCache;
};

#endif /* end of include guard: BASISCACHETESTS_H */
