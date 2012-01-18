#ifndef DPG_CONFUSION_INNER_PRODUCT
#define DPG_CONFUSION_INNER_PRODUCT

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

#include "ConfusionBilinearForm.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "DPGInnerProduct.h"

/*
  Implements Confusion inner product for L2 stability in u
*/

/* TO ASK NATE
   - how do I apply three operators to v (weighted beta dot grad v, epsilon grad v, and v)?

   - BUG - resizing of testValues1 and 2 seems to carry over and cause us to throw exceptions!  
*/

class ConfusionInnerProduct : public DPGInnerProduct {
 private:
  Teuchos::RCP<ConfusionBilinearForm> _ConfusionBilinearForm;
 public:
 ConfusionInnerProduct(Teuchos::RCP< ConfusionBilinearForm > bfs) : DPGInnerProduct(bfs) {
    _ConfusionBilinearForm=bfs;
  } 
  
  void operators(int testID1, int testID2, 
                 vector<IntrepidExtendedTypes::EOperatorExtended> &testOp1,
                 vector<IntrepidExtendedTypes::EOperatorExtended> &testOp2){
    testOp1.clear();
    testOp2.clear();
    
    if (testID1 == testID2) {

      if (ConfusionBilinearForm::TAU==testID1) {

	// L2 portion of tau
	testOp1.push_back(IntrepidExtendedTypes::OPERATOR_VALUE);
	testOp2.push_back(IntrepidExtendedTypes::OPERATOR_VALUE);

	// div portion of tau
	testOp1.push_back(IntrepidExtendedTypes::OPERATOR_DIV);
	testOp2.push_back(IntrepidExtendedTypes::OPERATOR_DIV);

      } else if (ConfusionBilinearForm::V==testID1) {
	
	// L2 portion of v (should be scaled by epsilon later)
	testOp1.push_back(IntrepidExtendedTypes::OPERATOR_VALUE);
	testOp2.push_back(IntrepidExtendedTypes::OPERATOR_VALUE);

	// grad portion of v (should be scaled by epsilon later);
	testOp1.push_back(IntrepidExtendedTypes::OPERATOR_GRAD);
	testOp2.push_back(IntrepidExtendedTypes::OPERATOR_GRAD);

	// grad dotted with beta for v (applied in next routine)
	testOp1.push_back(IntrepidExtendedTypes::OPERATOR_GRAD);
	testOp2.push_back(IntrepidExtendedTypes::OPERATOR_GRAD);

      }
    }

  }
  
  void applyInnerProductData(FieldContainer<double> &testValues1,
                             FieldContainer<double> &testValues2,
			     int testID1, int testID2, int operatorIndex,
			     FieldContainer<double>& physicalPoints){
    //assumption is testID1==testID2 already
    //    TEST_FOR_EXCEPTION(testID1==testID2, std::invalid_argument, 
    //		       "testID 1 and 2 not equal for ConfusionInnerProduct!");

    if (testID1==testID2){

      double epsilon = _ConfusionBilinearForm->getEpsilon();
      double _beta_x = _ConfusionBilinearForm->getBeta()[0];
      double _beta_y = _ConfusionBilinearForm->getBeta()[1];
    
      if (testID1==ConfusionBilinearForm::V ) {
	if ((operatorIndex==1)||(operatorIndex==2)) { // if it	
	  _bilinearForm->multiplyFCByWeight(testValues1,epsilon);
	  _bilinearForm->multiplyFCByWeight(testValues2,1.0);
	} else if (operatorIndex==3) { // if it's the beta dot grad term

	  int numCells = testValues1.dimension(0);
	  int basisCardinality = testValues1.dimension(1);
	  int numPoints = testValues1.dimension(2);
	  int spaceDim = testValues1.dimension(3);
	cout << "dimensions are " << numCells <<","<<basisCardinality<<","<<numPoints<<","<<spaceDim<< endl;

	  TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument,
			     "ConfusionBilinearForm only supports 2 dimensions right now.");

	  // because we change dimensions of the values, by dotting with beta, 
	  // we'll need to copy the values and resize the original container
	  FieldContainer<double> testValuesCopy1 = testValues1;
	  FieldContainer<double> testValuesCopy2 = testValues2;
	  testValues1.resize(numCells,basisCardinality,numPoints);
	  testValues2.resize(numCells,basisCardinality,numPoints);
	  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
	    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
	      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
		double x = physicalPoints(cellIndex,basisOrdinal,ptIndex,0);
		double y = physicalPoints(cellIndex,basisOrdinal,ptIndex,1);
		double weight = getWeight(x,y);
		testValues1(cellIndex,basisOrdinal,ptIndex)  = -_beta_x * testValuesCopy1(cellIndex,basisOrdinal,ptIndex,0) * weight 
		  + -_beta_y * testValuesCopy1(cellIndex,basisOrdinal,ptIndex,1) * weight;
		testValues2(cellIndex,basisOrdinal,ptIndex)  = -_beta_x * testValuesCopy2(cellIndex,basisOrdinal,ptIndex,0) * weight 
		  + -_beta_y * testValuesCopy2(cellIndex,basisOrdinal,ptIndex,1) * weight;
	      }
	    }
	  }
	}
      } else if (testID1==ConfusionBilinearForm::TAU){

	int numCells = testValues1.dimension(0);
	int basisCardinality = testValues1.dimension(1);
	int numPoints = testValues1.dimension(2);
	int spaceDim = testValues1.dimension(3);
	cout << "dimensions are " << numCells <<","<<basisCardinality<<","<<numPoints<<","<<spaceDim<< endl;
	TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument,
			   "ConfusionBilinearForm 2nd time only supports 2 dimensions right now.");
	
	// because we change dimensions of the values, by dotting with beta, 
	// we'll need to copy the values and resize the original container
	FieldContainer<double> testValuesCopy1 = testValues1;
	FieldContainer<double> testValuesCopy2 = testValues2;
	testValues1.resize(numCells,basisCardinality,numPoints);
	testValues2.resize(numCells,basisCardinality,numPoints);
	for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
	  for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
	    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
	      double x = physicalPoints(cellIndex,basisOrdinal,ptIndex,0);
	      double y = physicalPoints(cellIndex,basisOrdinal,ptIndex,1);
	      double weight = getWeight(x,y);
	      testValues1(cellIndex,basisOrdinal,ptIndex) = testValues1(cellIndex,basisOrdinal,ptIndex)*weight;
	      testValues2(cellIndex,basisOrdinal,ptIndex) = testValues2(cellIndex,basisOrdinal,ptIndex)*weight; 
	    }
	  }
	}
      }       
    }
  }

  // get weight that biases the outflow over the inflow (for math stability purposes)
  double getWeight(double x,double y){

    return _ConfusionBilinearForm->getEpsilon()+x*y;
  }
};

#endif
