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
#include "BF.h"
#include "IP.h"
#include "RHS.h"
#include "BasisCache.h"

// Shards includes
#include "Shards_CellTopology.hpp"

#include "Mesh.h"

#include "CamelliaIntrepidExtendedTypes.h"

using namespace std;
// using namespace Intrepid;
using namespace Camellia;

class DPGTests;

class BilinearFormUtility {
private:
  static bool _warnAboutZeroRowsAndColumns;
  static void setWarnAboutZeroRowsAndColumns( bool value );
  static bool warnAboutZeroRowsAndColumns();
public:
  friend class DPGTests;
  
  // the "pre-stiffness" (rectangular) matrix methods:
  static void computeStiffnessMatrix(Intrepid::FieldContainer<double> &stiffness, BFPtr bilinearForm,
                                     Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                     CellTopoPtr cellTopo, Intrepid::FieldContainer<double> &physicalCellNodes,
                                     Intrepid::FieldContainer<double> &cellSideParities);

  // the real one:
//  static void computeStiffnessMatrix(Intrepid::FieldContainer<double> &stiffness, BFPtr bilinearForm,
//                                     Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
//                                     Intrepid::FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache);
  
  static void computeStiffnessMatrixForCell(Intrepid::FieldContainer<double> &stiffness, Teuchos::RCP<Mesh> mesh, int cellID);
  
  // final (square) stiffness methods, with optimal test functions applied:
  // the following is meant for testing; the three-argument computeStiffnessMatrix below will be more efficient...
  static void computeOptimalStiffnessMatrix(Intrepid::FieldContainer<double> &stiffness, 
                                            Intrepid::FieldContainer<double> &optimalTestWeights,
                                            BFPtr bilinearForm,
                                            Teuchos::RCP<DofOrdering> trialOrdering, Teuchos::RCP<DofOrdering> testOrdering,
                                            CellTopoPtr cellTopo, Intrepid::FieldContainer<double> &physicalCellNodes,
                                            Intrepid::FieldContainer<double> &cellSideParities);
  
  static void computeStiffnessMatrix(Intrepid::FieldContainer<double> &stiffness, Intrepid::FieldContainer<double> &innerProductMatrix,
                                     Intrepid::FieldContainer<double> &optimalTestWeights);
  
  // this method is deprecated; use the next one
  static void computeRHS(Intrepid::FieldContainer<double> &rhsVector, BFPtr bilinearForm, RHS &rhs, 
                         Intrepid::FieldContainer<double> &optimalTestWeights, Teuchos::RCP<DofOrdering> testOrdering,
                         shards::CellTopology &cellTopo, Intrepid::FieldContainer<double> &physicalCellNodes);
  
//  static void computeRHS(Intrepid::FieldContainer<double> &rhsVector, BFPtr bilinearForm, RHS &rhs, 
//                  Intrepid::FieldContainer<double> &optimalTestWeights, Teuchos::RCP<DofOrdering> testOrdering,
//                  BasisCachePtr basisCache);

  static void transposeFCMatrices(Intrepid::FieldContainer<double> &fcTranspose,
                                  const Intrepid::FieldContainer<double> &fc);
  
  static bool checkForZeroRowsAndColumns(string name, Intrepid::FieldContainer<double> &array, bool checkRows = true, bool checkCols = true);
  
  static void weightCellBasisValues(Intrepid::FieldContainer<double> &basisValues, 
                                    const Intrepid::FieldContainer<double> &weights, int offset); 
};
#endif
