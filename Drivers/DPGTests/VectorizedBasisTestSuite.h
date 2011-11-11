#ifndef DPGTrilinos_VectorizedBasisTestSuite_h
#define DPGTrilinos_VectorizedBasisTestSuite_h

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

//
//  VectorizedBasisTestSuite.h
//  DPGTrilinos
//


#include "Vectorized_Basis.hpp"
#include "BasisFactory.h"

class VectorizedBasisTestSuite {
  
public:
  static void runTests(int &numTestsRun, int &numTestsPassed) {
    numTestsRun++;
    if ( testVectorizedBasis() ) {
      numTestsPassed++;
    }
  }
  static bool testVectorizedBasis() {
    bool success = true;
    
    string myName = "testVectorizedBasis";
    
    shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
    
    int polyOrder = 3, numPoints = 5, spaceDim = 2;
    
    int basisRank;
    Teuchos::RCP< Basis<double,FieldContainer<double> > > hgradBasis
    = 
    BasisFactory::getBasis(basisRank,polyOrder,
                           quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
    
    // first test: make a single-component vector basis.  This should agree in every entry with the basis itself, but its field container will have one higher rank...
    Vectorized_Basis<double, FieldContainer<double> > oneComp(hgradBasis, 1);
    
    FieldContainer<double> linePoints(numPoints, spaceDim);
    for (int i=0; i<numPoints; i++) {
      for (int j=0; j<spaceDim; j++) {
        linePoints(i,j) = ((double)(i + j)) / (numPoints + spaceDim);
      }
    }
    
    FieldContainer<double> compValues(hgradBasis->getCardinality(),linePoints.dimension(0));
    hgradBasis->getValues(compValues, linePoints, Intrepid::OPERATOR_VALUE);
    
    FieldContainer<double> values(hgradBasis->getCardinality(),linePoints.dimension(0),1); // one component
    oneComp.getValues(values, linePoints, Intrepid::OPERATOR_VALUE);
    
    for (int i=0; i<compValues.size(); i++) {
      double diff = abs(values[i]-compValues[i]);
      if (diff != 0.0) {
        success = false;
        cout << myName << ": one-component vector basis doesn't produce same values as component basis." << endl;
        cout << "difference: " << diff << " in enumerated value " << i << endl;
        cout << "values:\n" << values;
        cout << "compValues:\n" << compValues;
        return success;
      }
    }
    
    vector< Teuchos::RCP< Basis<double, FieldContainer<double> > > > twoComps;
    twoComps.push_back( Teuchos::rcp( new Vectorized_Basis<double, FieldContainer<double> >(hgradBasis, 2) ) );
    twoComps.push_back( BasisFactory::getBasis( polyOrder,
                                               quad_4.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) );
    
    
    vector< Teuchos::RCP< Basis<double, FieldContainer<double> > > >::iterator twoCompIt;
    for (twoCompIt = twoComps.begin(); twoCompIt != twoComps.end(); twoCompIt++) {
      Teuchos::RCP< Basis<double, FieldContainer<double> > > twoComp = *twoCompIt;
    
      int componentCardinality = hgradBasis->getCardinality();
      
      if (twoComp->getCardinality() != 2 * hgradBasis->getCardinality() ) {
        success = false;
        cout << myName << ": two-component vector basis cardinality != one-component cardinality * 2." << endl;
        cout << "twoComp->getCardinality(): " << twoComp->getCardinality() << endl;
        cout << "oneComp->getCardinality(): " << oneComp.getCardinality() << endl;
      }
      
      values.resize(twoComp->getCardinality(),linePoints.dimension(0),2); // two components
      twoComp->getValues(values, linePoints, Intrepid::OPERATOR_VALUE);
      for (int basisIndex=0; basisIndex<twoComp->getCardinality(); basisIndex++) {
        for (int k=0; k<numPoints; k++) {
          double xValueExpected = (basisIndex < componentCardinality) ? compValues(basisIndex,k) : 0;
          double xValueActual = values(basisIndex,k,0);
          double yValueExpected = (basisIndex >= componentCardinality) ? compValues(basisIndex - componentCardinality,k) : 0;
          double yValueActual = values(basisIndex,k,1);
          if ( ( abs(xValueActual - xValueExpected) != 0) || ( abs(yValueActual - yValueExpected) != 0) ) {
            success = false;
            cout << myName << ": expected differs from actual\n";
            cout << "component\n" << compValues;
            cout << "vector values:\n" << values;
            return success;
          }
        }
      }
    }
    return success;
  }
  static bool testHGRAD_2D() {
    bool success = true;
    // on a single quad element, evaluate the various operations
    // TODO: cross normal
    // TODO: div
    // TODO: curl
    return success;
  }
};

#endif
