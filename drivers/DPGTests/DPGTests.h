#ifndef DPG_TESTS
#define DPG_TESTS

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

using namespace Intrepid;
using namespace std;

class Epetra_SerialDenseMatrix;

class DPGTests {
public:
  static void runExceptionThrowingTest();
  
  static void runTests();
  static void createBases();
  static bool testDofOrdering();
  static bool testComputeStiffnessDx();
  static bool testAnalyticBoundaryIntegral(bool);
  static bool testComputeStiffnessConformingVertices();
  static bool testComputeStiffnessTrace();
  static bool testComputeStiffnessFlux();
  static bool testMathInnerProductDx();
  static bool testLowOrderTrialCubicTest();
//  static bool testOptimalStiffnessByIntegrating();
  static bool testOptimalStiffnessByMultiplying();
  static bool testComputeOptimalTest();
//  static bool testComputeOptimalTestPoisson();
  static bool testTestBilinearFormAnalyticBoundaryIntegralExpectedConformingMatrices();
  static bool testWeightBasis();
  static bool checkOptTestWeights(FieldContainer<double> &optWeights,
                                  FieldContainer<double> &ipMatrix,
                                  FieldContainer<double> &preStiffness,
                                  double tol);
  static bool fcsAgree(string &testName, FieldContainer<double> &expected, 
                       FieldContainer<double> &actual, double tol);
  static bool fcEqualsSDM(FieldContainer<double> &fc, int cellIndex,
                          Epetra_SerialDenseMatrix &sdm, double tol, bool transpose);
  static bool fcIsSymmetric(FieldContainer<double> &fc, double tol, 
                            int &cellOfAsymmetry,
                            int &rowOfAsymmetry, int &colOfAsymmetry);
};

#endif
