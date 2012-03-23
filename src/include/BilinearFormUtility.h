#ifndef BILINEAR_FORM_UTILITY
#define BILINEAR_FORM_UTILITY

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

#include "DofOrdering.h"
#include "BilinearForm.h"
#include "DPGInnerProduct.h"
#include "RHS.h"
#include "BasisCache.h"

// Shards includes
#include "Shards_CellTopology.hpp"

#include "Mesh.h"

using namespace std;
using namespace Intrepid;

class DPGTests;

typedef Teuchos::RCP<BasisCache> BasisCachePtr;
typedef Teuchos::RCP<BilinearForm> BilinearFormPtr;

class BilinearFormUtility {
private:
  static bool _warnAboutZeroRowsAndColumns;
  static void setWarnAboutZeroRowsAndColumns( bool value );
  static bool warnAboutZeroRowsAndColumns();
public:
  friend class DPGTests;

  static int computeOptimalTest(FieldContainer<double> &optimalTestWeights,
                                FieldContainer<double> &innerProductMatrix,
                                BilinearFormPtr bilinearForm,
                                Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                FieldContainer<double> &cellSideParities,
                                Teuchos::RCP<BasisCache> stiffnessBasisCache); // stiffness as opposed to the test-test cache
  
  // deprecated computeOptimalTests:
  static int computeOptimalTest(FieldContainer<double> &optimalTestWeights,
                                 DPGInnerProduct &innerProduct,
                                 BilinearFormPtr bilinearForm,
                                 Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                 shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes,
                                 FieldContainer<double> &cellSideParities);
  
  static int computeOptimalTest(FieldContainer<double> &optimalTestWeights,
                                FieldContainer<double> &innerProductMatrix,
                                BilinearFormPtr bilinearForm,
                                Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes,
                                FieldContainer<double> &cellSideParities);
  
  // the "pre-stiffness" (rectangular) matrix methods:
  static void computeStiffnessMatrix(FieldContainer<double> &stiffness, BilinearFormPtr bilinearForm,
                                     Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                     shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes,
                                     FieldContainer<double> &cellSideParities);

  // the real one:
//  static void computeStiffnessMatrix(FieldContainer<double> &stiffness, BilinearFormPtr bilinearForm,
//                                     Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
//                                     FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache);
  
  static void computeStiffnessMatrixForCell(FieldContainer<double> &stiffness, Teuchos::RCP<Mesh> mesh, int cellID);
  
  // final (square) stiffness methods, with optimal test functions applied:
  // the following is meant for testing; the three-argument computeStiffnessMatrix below will be more efficient...
  static void computeOptimalStiffnessMatrix(FieldContainer<double> &stiffness, 
                                            FieldContainer<double> &optimalTestWeights,
                                            BilinearFormPtr bilinearForm,
                                            Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                            shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes,
                                            FieldContainer<double> &cellSideParities);
  
  static void computeStiffnessMatrix(FieldContainer<double> &stiffness, FieldContainer<double> &innerProductMatrix,
                                     FieldContainer<double> &optimalTestWeights);
  
  // this method is deprecated; use the next one
  static void computeRHS(FieldContainer<double> &rhsVector, BilinearFormPtr bilinearForm, RHS &rhs, 
                         FieldContainer<double> &optimalTestWeights, Teuchos::RCP<DofOrdering> testOrdering,
                         shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes);
  
  static void computeRHS(FieldContainer<double> &rhsVector, BilinearFormPtr bilinearForm, RHS &rhs, 
                  FieldContainer<double> &optimalTestWeights, Teuchos::RCP<DofOrdering> testOrdering,
                  BasisCachePtr basisCache);

  static void transposeFCMatrices(FieldContainer<double> &fcTranspose,
                                  const FieldContainer<double> &fc);
  
  static bool checkForZeroRowsAndColumns(string name, FieldContainer<double> &array);
private:
  static void weightCellBasisValues(FieldContainer<double> &basisValues, 
                                    const FieldContainer<double> &weights, int offset);
  
};
#endif
