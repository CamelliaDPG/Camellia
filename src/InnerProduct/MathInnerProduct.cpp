// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of
// conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

#include "Intrepid_CellTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "MathInnerProduct.h"

using namespace Intrepid;
using namespace Camellia;

void MathInnerProduct::operators(int testID1, int testID2,
               vector<Camellia::EOperator> &testOp1,
               vector<Camellia::EOperator> &testOp2) {
  testOp1.clear();
  testOp2.clear();
  if (testID1 == testID2) {
    Camellia::EOperator dOperator;
    if (_bilinearForm->functionSpaceForTest(testID1) == Camellia::FUNCTION_SPACE_REAL_SCALAR) {
      testOp1.push_back( Camellia::OP_VALUE);
      testOp2.push_back( Camellia::OP_VALUE);
    } else {
      if (_bilinearForm->functionSpaceForTest(testID1) == Camellia::FUNCTION_SPACE_HGRAD) {
        dOperator = Camellia::OP_GRAD;
      } else if (_bilinearForm->functionSpaceForTest(testID1) == Camellia::FUNCTION_SPACE_HDIV) {
        dOperator = Camellia::OP_DIV;
      } else if (_bilinearForm->functionSpaceForTest(testID1) == Camellia::FUNCTION_SPACE_HCURL) {
        dOperator = Camellia::OP_CURL;
      } else if ( _bilinearForm->functionSpaceForTest(testID1) == Camellia::FUNCTION_SPACE_VECTOR_HGRAD ) {
        dOperator = Camellia::OP_GRAD; // will the integration routine do the right thing here??
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unknown test space.");
      }
      testOp1.push_back( Camellia::OP_VALUE);
      testOp1.push_back(dOperator);
      testOp2.push_back( Camellia::OP_VALUE);
      testOp2.push_back(dOperator);
    }
  }
}

void MathInnerProduct::applyInnerProductData(FieldContainer<double> &testValues1,
                                             FieldContainer<double> &testValues2,
                                             int testID1, int testID2, int operatorIndex,
                                             const FieldContainer<double>& physicalPoints) {
  // empty implementation -- no weights needed...
}
