#include "ConfusionBilinearForm.h"
#include "Mesh.h"
#include "BasisCache.h" // for Jacobian/cell measure computation

#include "ExperimentalInnerProduct.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "DPGInnerProduct.h"
/*
  Implements experimental Confusion inner product 
*/

ExperimentalInnerProduct::ExperimentalInnerProduct(Teuchos::RCP< ConfusionBilinearForm > bfs, Teuchos::RCP<Mesh> mesh) : DPGInnerProduct((Teuchos::RCP< ConfusionBilinearForm>) bfs) {
  _confusionBilinearForm=bfs; // redundant, but no way around it.
  _mesh=mesh; // for h-scaling
} 
  
void ExperimentalInnerProduct::operators(int testID1, int testID2, 
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
        
      // L2 portion of v (should be scaled by epsilon/h later)
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
  
  if (testID1==ConfusionBilinearForm::TAU){
    if (testID2==ConfusionBilinearForm::V){
      //      testOp1.push_back(IntrepidExtendedTypes::OPERATOR_DIV); // conservation cross term 1
      //      testOp2.push_back(IntrepidExtendedTypes::OPERATOR_GRAD);

      //      testOp1.push_back(IntrepidExtendedTypes::OPERATOR_VALUE); // constitutive cross term 1
      //      testOp2.push_back(IntrepidExtendedTypes::OPERATOR_GRAD);
	
    }
  }
  
  if (testID1==ConfusionBilinearForm::V){
    if (testID2==ConfusionBilinearForm::TAU){
      //      testOp1.push_back(IntrepidExtendedTypes::OPERATOR_GRAD); // conservation cross term 2
      //      testOp2.push_back(IntrepidExtendedTypes::OPERATOR_DIV);

      //      testOp1.push_back(IntrepidExtendedTypes::OPERATOR_GRAD); // constitutive cross term 2
      //      testOp2.push_back(IntrepidExtendedTypes::OPERATOR_VALUE);

    }
  }
  
    
}
  
void ExperimentalInnerProduct::applyInnerProductData(FieldContainer<double> &testValues1,
						     FieldContainer<double> &testValues2,
						     int testID1, int testID2, int operatorIndex,
						     const FieldContainer<double>& physicalPoints){
    
  int numCells = physicalPoints.dimension(0);
  int basisCardinality = testValues1.dimension(1);
  int numPoints = testValues1.dimension(2);

  FieldContainer<double> testValuesCopy1 = testValues1;
  FieldContainer<double> testValuesCopy2 = testValues2;

  double epsilon = _confusionBilinearForm->getEpsilon();

  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {                        

    double C = 1.0; // peclet?

    for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
	double x = physicalPoints(cellIndex,ptIndex,0);
	double y = physicalPoints(cellIndex,ptIndex,1);
	double beta_x = _confusionBilinearForm->getBeta(x,y)[0];
	double beta_y = _confusionBilinearForm->getBeta(x,y)[1];	      
	double weight = getWeight(x,y);		  
	    
	// tau terms together
	if ((testID1==ConfusionBilinearForm::TAU) && (testID2==ConfusionBilinearForm::TAU)){
	  if (operatorIndex==0){ // L^2 of tau
	    testValues1(cellIndex,basisOrdinal,ptIndex,0) *= (1.0+(C/epsilon)*(C/epsilon)); // scale all terms by constant
	    testValues1(cellIndex,basisOrdinal,ptIndex,1) *= (1.0+(C/epsilon)*(C/epsilon)); // scale all terms by constant
	  } else if (operatorIndex==1) { // div tau
	    // do nothing, both are already correct
	  } else {
	    TEST_FOR_EXCEPTION(false, std::invalid_argument,"Op index too big");
	  }
	}

	// v terms together
	if ((testID1==ConfusionBilinearForm::V) && (testID2==ConfusionBilinearForm::V)){	    
	  if (operatorIndex==0){ // L^2 of v
	    // do nothing
	  } else if (operatorIndex==1) { // grad v
	    testValues1(cellIndex,basisOrdinal,ptIndex,0) *= ((C/epsilon)*(C/epsilon)); // scale all terms by constant
	    testValues1(cellIndex,basisOrdinal,ptIndex,1) *= ((C/epsilon)*(C/epsilon)); // scale all terms by constant
	  } else if (operatorIndex==2) { // beta dot grad v
	    testValues1.resize(numCells,basisCardinality,numPoints);
	    testValues2.resize(numCells,basisCardinality,numPoints);	      
	    testValues1(cellIndex,basisOrdinal,ptIndex)  = beta_x * testValuesCopy1(cellIndex,basisOrdinal,ptIndex,0) * weight 
	      + beta_y * testValuesCopy1(cellIndex,basisOrdinal,ptIndex,1) * weight;
	    testValues2(cellIndex,basisOrdinal,ptIndex)  = beta_x * testValuesCopy2(cellIndex,basisOrdinal,ptIndex,0)
	      + beta_y * testValuesCopy2(cellIndex,basisOrdinal,ptIndex,1);	      
	  } else {
	    TEST_FOR_EXCEPTION(false, std::invalid_argument,"Op index too big");
	  }
	}
	  
	// tau cross v (negative conservation eq terms)
	if ((testID1==ConfusionBilinearForm::TAU) && (testID2==ConfusionBilinearForm::V)){
	  if (operatorIndex==0){ // beta dot grad v dot div tau
	    if (testValues2.rank()==4){
	      testValues2.resize(numCells,basisCardinality,numPoints);	      
	    }
	    //testValues2(cellIndex,basisOrdinal,ptIndex)  = -(beta_x * testValuesCopy2(cellIndex,basisOrdinal,ptIndex,0) + beta_y * testValuesCopy2(cellIndex,basisOrdinal,ptIndex,1));
	  } else if (operatorIndex==1) { 
	    testValues1(cellIndex,basisOrdinal,ptIndex,0) *= (C/epsilon); // scale by c/eps only
	    testValues1(cellIndex,basisOrdinal,ptIndex,1) *= (C/epsilon); // scale by c/eps only
	  } else {
	    TEST_FOR_EXCEPTION(false, std::invalid_argument,"Op index too big");
	  }	    
	}

	// v cross tau (negative conservation eq terms)
	if ((testID1==ConfusionBilinearForm::V) && (testID2==ConfusionBilinearForm::TAU)){
	  if (operatorIndex==0){
	    if (testValues1.rank()==4){
	      testValues1.resize(numCells,basisCardinality,numPoints);	      
	    }
	    //testValues1(cellIndex,basisOrdinal,ptIndex)  = -(beta_x * testValuesCopy1(cellIndex,basisOrdinal,ptIndex,0) + beta_y * testValuesCopy1(cellIndex,basisOrdinal,ptIndex,1));
	  } else if (operatorIndex==1) { 
	    testValues1(cellIndex,basisOrdinal,ptIndex,0) *= (C/epsilon); // scale by c/eps only
	    testValues1(cellIndex,basisOrdinal,ptIndex,1) *= (C/epsilon); // scale by c/eps only
	  } else {
	    TEST_FOR_EXCEPTION(false, std::invalid_argument,"Op index too big");
	  }
	}
	  
      }
    }
  }
    
}
  
// get weight that biases the outflow over the inflow (for math stability purposes)
double ExperimentalInnerProduct::getWeight(double x,double y){

  //    cout << "epsilon = " << _confusionBilinearForm->getEpsilon()<<endl;    
  //    return _confusionBilinearForm->getEpsilon() + x;

  return 1.0; // for the new inflow condition
}

