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
#include "Mesh.h"
#include "BasisValueCache.h" // for Jacobian/cell measure computation
// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "DPGInnerProduct.h"

/*
 Implements Confusion inner product for L2 stability in u
 */

class ConfusionInnerProduct : public DPGInnerProduct {
private:
  Teuchos::RCP<ConfusionBilinearForm> _confusionBilinearForm;
  Teuchos::RCP<Mesh> _mesh;
public:
  typedef Teuchos::RCP< ElementType > ElementTypePtr;
  typedef Teuchos::RCP< Element > ElementPtr;
  
  ConfusionInnerProduct(Teuchos::RCP< ConfusionBilinearForm > bfs, Teuchos::RCP<Mesh> mesh) : DPGInnerProduct((Teuchos::RCP< ConfusionBilinearForm>) bfs) {
    _confusionBilinearForm=bfs; // redundant, but no way around it.
    _mesh=mesh; // for h-scaling
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
      
      double epsilon = _confusionBilinearForm->getEpsilon();
      double beta_x = _confusionBilinearForm->getBeta()[0];
      double beta_y = _confusionBilinearForm->getBeta()[1];
      //      cout << "Beta = " << beta_x << ", " << beta_y << ", and epsilon = " << epsilon << endl;
      
      if (testID1==ConfusionBilinearForm::V ) {
        if ((operatorIndex==0)||(operatorIndex==1)) { // if it	
          
          int numCells = testValues1.dimension(0);
          int basisCardinality = testValues1.dimension(1);
          int numPoints = testValues1.dimension(2);
          int spaceDim = _confusionBilinearForm->getBeta().size();
                    
          for (int cellIndex=0; cellIndex<numCells; cellIndex++) {              

            //////////////////// 
            /*
            cout << "got here" << endl;
            // compute measure of current cell, scale epsilon term by this 
            FieldContainer<double> physicalPointForCell(1,spaceDim);
            physicalPointForCell(0,0) = physicalPoints(cellIndex,0,0); // assuming that all all pts corresponding to 1 cell index are the same...
            physicalPointForCell(0,1) = physicalPoints(cellIndex,0,1);
            vector<ElementPtr> elemVector = _mesh->elementsForPoints(physicalPointForCell);
            TEST_FOR_EXCEPTION(elemVector.size()>1, std::invalid_argument,
                               "More than one element returned for a single pt!");
            ElementPtr elem = elemVector[0];
            cout << "got here" << endl;
            
            // it turns out i don't have access to physicalCellNodes, used in constructor for basisCache. 
            // code below just get's one cell's physicalCellNodes
            FieldContainer<double> allPhysicalNodesForType = _mesh->physicalCellNodes(elem->elementType());
            int numSides = allPhysicalNodesForType.dimension(1);
            FieldContainer<double> physicalCellNodes(1,numSides,spaceDim);
            for (int sideIndex=0;sideIndex<numSides;sideIndex++){
              for (int dimIndex=0;dimIndex<spaceDim;dimIndex++){              
                physicalCellNodes(0,sideIndex,spaceDim)=allPhysicalNodesForType(cellIndex,sideIndex,spaceDim);                                                                                
              }
            }            
            cout << "got here" << endl;
            
            // create basisCache
            int cubDegree = elem->elementType()->testOrderPtr->maxBasisDegree();
            BasisValueCache basisCache = BasisValueCache(physicalCellNodes, *(elem->elementType()->cellTopoPtr), cubDegree);
            FieldContainer<double> cellMeasures = basisCache.getCellMeasures();

            double geometryScaling = cellMeasures(cellIndex);
            cout << "cell measure for cell " << elem->cellID() << " is " << geometryScaling <<endl;
*/
            double geometryScaling=1.0;
            
            //////////////////// 
            
            for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
              for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
                testValues1(cellIndex,basisOrdinal,ptIndex) = testValues1(cellIndex,basisOrdinal,ptIndex)*epsilon*geometryScaling;
              }
            }
          }
          
          _bilinearForm->multiplyFCByWeight(testValues1,epsilon);
          _bilinearForm->multiplyFCByWeight(testValues2,1.0);
          
        } else if (operatorIndex==2) { // if it's the beta dot grad term
          
          int numCells = testValues1.dimension(0);
          int basisCardinality = testValues1.dimension(1);
          int numPoints = testValues1.dimension(2);
          int spaceDim = testValues1.dimension(3);
          //	  cout << "dimensions are " << numCells <<","<<basisCardinality<<","<<numPoints<<","<<spaceDim<< endl;
          
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
                double x = physicalPoints(cellIndex,ptIndex,0);
                double y = physicalPoints(cellIndex,ptIndex,1);
                double weight = getWeight(x,y);
                testValues1(cellIndex,basisOrdinal,ptIndex)  = beta_x * testValuesCopy1(cellIndex,basisOrdinal,ptIndex,0) * weight 
                + beta_y * testValuesCopy1(cellIndex,basisOrdinal,ptIndex,1) * sqrt(weight);
                testValues2(cellIndex,basisOrdinal,ptIndex)  = beta_x * testValuesCopy2(cellIndex,basisOrdinal,ptIndex,0) * weight 
                + beta_y * testValuesCopy2(cellIndex,basisOrdinal,ptIndex,1) * sqrt(weight);
              }
            }
          }
        }
      } else if (testID1==ConfusionBilinearForm::TAU){
        
        if (operatorIndex==0) {
          int numCells = testValues1.dimension(0);
          int basisCardinality = testValues1.dimension(1);
          int numPoints = testValues1.dimension(2);
          int spaceDim = testValues1.dimension(3);
          
          // because we change dimensions of the values, by dotting with beta, 
          // we'll need to copy the values and resize the original container
          FieldContainer<double> testValuesCopy1 = testValues1;
          FieldContainer<double> testValuesCopy2 = testValues2;
          for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
              for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
                double x = physicalPoints(cellIndex,ptIndex,0);
                double y = physicalPoints(cellIndex,ptIndex,1);
                double weight = getWeight(x,y);
                testValues1(cellIndex,basisOrdinal,ptIndex) = testValues1(cellIndex,basisOrdinal,ptIndex)*sqrt(weight);
                testValues2(cellIndex,basisOrdinal,ptIndex) = testValues2(cellIndex,basisOrdinal,ptIndex)*sqrt(weight);
              }
            }
          }
        } else if (operatorIndex==1) {
          
          int numCells = testValues1.dimension(0);
          int basisCardinality = testValues1.dimension(1);
          int numPoints = testValues1.dimension(2);
          
          FieldContainer<double> testValuesCopy1 = testValues1;
          FieldContainer<double> testValuesCopy2 = testValues2;
          for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
              for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
                double x = physicalPoints(cellIndex,ptIndex,0);
                double y = physicalPoints(cellIndex,ptIndex,1);
                double weight = getWeight(x,y);
                testValues1(cellIndex,basisOrdinal,ptIndex) = testValues1(cellIndex,basisOrdinal,ptIndex)*sqrt(weight);
                testValues2(cellIndex,basisOrdinal,ptIndex) = testValues2(cellIndex,basisOrdinal,ptIndex)*sqrt(weight);
              }
            }
          }
        }
      }       
    }
  }
  
  // get weight that biases the outflow over the inflow (for math stability purposes)
  double getWeight(double x,double y){
    
    return _confusionBilinearForm->getEpsilon() + x*y;
  }
};

#endif
