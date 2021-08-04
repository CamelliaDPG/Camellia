#ifndef DPG_MATH_INNER_PRODUCT
#define DPG_MATH_INNER_PRODUCT

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#include "BF.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "IP.h"

/*
  Implements "standard" inner product for H1, H(div) test functions
 */

namespace Camellia
{
class MathInnerProduct : public IP
{
public:
  MathInnerProduct(Teuchos::RCP< BF > bfs) : IP(bfs) {}

  void operators(int testID1, int testID2,
                 vector<Camellia::EOperator> &testOp1,
                 vector<Camellia::EOperator> &testOp2);

  using IP::applyInnerProductData;
  void applyInnerProductData(Camellia::FieldContainer<double> &testValues1,
                             Camellia::FieldContainer<double> &testValues2,
                             int testID1, int testID2, int operatorIndex,
                             const Camellia::FieldContainer<double>& physicalPoints);
};
}

#endif
