#ifndef DOF_ORDERING
#define DOF_ORDERING

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

// Intrepid includes
#include "Intrepid_Basis.hpp"
#include "Intrepid_FieldContainer.hpp"

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Basis.h"

using namespace Intrepid;
using namespace std;

class DofOrdering {
  int _indexNeedsToBeRebuilt;
  int _nextIndex;
//  vector<int> varIDs;
  set<int> varIDs;
  vector<int> varIDsVector;
// outer vector: indexed by parent's sides; inner vector: (child index in children, index of child's side shared with parent)
  map< pair<int, pair<int, int> >, pair<int, int> > dofIdentifications; // keys: <varID, <sideIndex, dofOrdinal> >
                                                                        // values: <sideIndex, dofOrdinal>
  map<int,int> numSidesForVarID;
  map< pair<int,int>, vector<int> > indices; // keys for indices are <varID, sideIndex >, where sideIndex = 0 for field (volume) variables
//  map< pair<int,pair<int,int> >, int> indices; // keys for indices are <varID, <sideIndex, dofOrdinal> >, where sideIndex = 0 for field (volume) variables
  map< pair<int,int>, BasisPtr > bases; // keys are <varID, sideIndex>
  map< int, int > basisRanks; // keys are varIDs; values are 0,1,2,... (scalar, vector, tensor)
  
  map< int, Teuchos::RCP< shards::CellTopology > > _cellTopologyForSide; // -1 is field variable
public:
  DofOrdering(); // constructor
  
  void addEntry(int varID, BasisPtr basis, int basisRank, int sideIndex = 0);
  
  bool hasBasisEntry(int varID, int sideIndex);
  bool hasSideVarIDs();
  
  void copyLikeCoefficients( FieldContainer<double> &newValues, Teuchos::RCP<DofOrdering> oldDofOrdering,
                            const FieldContainer<double> &oldValues );
  
  // get the varIndex variable's dof with basis ordinal dofId in the Dof ordering:
  int getDofIndex(int varID, int basisDofOrdinal, int sideIndex=0, int subSideIndex = -1);
  
  const vector<int> & getDofIndices(int varID, int sideIndex=0);
  
  const set<int> & getVarIDs();
  
  int getNumSidesForVarID(int varID);
  
  int getBasisCardinality(int varID, int sideIndex);
  
  BasisPtr getBasis(int varID, int sideIndex = 0);
  
  int getBasisRank(int varID) {
    return basisRanks[varID];
  }
  
  void addIdentification(int varID, int side1, int basisDofOrdinal1,
                         int side2, int basisDofOrdinal2);
  
  Teuchos::RCP< shards::CellTopology > cellTopology(int sideIndex = -1);
  
  int maxBasisDegree();
  int maxBasisDegreeForVolume();
  
  int totalDofs() {
    return _nextIndex;
  }
  
  void rebuildIndex();
};

typedef Teuchos::RCP< DofOrdering> DofOrderingPtr;

std::ostream& operator << (std::ostream& os, DofOrdering& dofOrdering);

#endif
