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

// Shards includes
#include "Shards_CellTopology.hpp"

using namespace std;
using namespace Intrepid;

class DPGTests;

class BilinearFormUtility {
private:
  static bool checkForZeroRowsAndColumns(string name, FieldContainer<double> &array);
public:
  friend class DPGTests;

  static int computeOptimalTest(FieldContainer<double> &optimalTestWeights,
                                 DPGInnerProduct &innerProduct,
                                 BilinearForm &bilinearForm,
                                 Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                 shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes,
                                 FieldContainer<double> &cellSideParities);
  
  static int computeOptimalTest(FieldContainer<double> &optimalTestWeights,
                                FieldContainer<double> &innerProductMatrix,
                                BilinearForm &bilinearForm,
                                Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes,
                                FieldContainer<double> &cellSideParities);
  
  static void computeStiffnessMatrix(FieldContainer<double> &stiffness, BilinearForm &bilinearForm,
                                     Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                     shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes,
                                     FieldContainer<double> &cellSideParities);

  // the following is meant for testing; the three-argument computeStiffnessMatrix below will be more efficient...
  static void computeOptimalStiffnessMatrix(FieldContainer<double> &stiffness, 
                                            FieldContainer<double> &optimalTestWeights,
                                            BilinearForm &bilinearForm,
                                            Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                            shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes,
                                            FieldContainer<double> &cellSideParities);
  
  static void computeStiffnessMatrix(FieldContainer<double> &stiffness, FieldContainer<double> &innerProductMatrix,
                                     FieldContainer<double> &optimalTestWeights);
  
  static void computeRHS(FieldContainer<double> &rhsVector, BilinearForm &bilinearForm, RHS &rhs, 
                         FieldContainer<double> &optimalTestWeights, Teuchos::RCP<DofOrdering> testOrdering,
                         shards::CellTopology &cellTopo, FieldContainer<double> &physicalCellNodes);

  static void transposeFCMatrices(FieldContainer<double> &fcTranspose,
                                  const FieldContainer<double> &fc);
private:
  static void weightCellBasisValues(FieldContainer<double> &basisValues, 
                                    const FieldContainer<double> &weights, int offset);
  
};
#endif
