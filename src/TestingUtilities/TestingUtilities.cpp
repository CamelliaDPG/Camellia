#include "TestingUtilities.h"

// @HEADER
//
// Original Version Copyright © 2011 Sandia Corporation. All Rights Reserved.
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

/*
 *  TestingUtilities.cpp
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */

bool TestingUtilities::isFluxOrTraceDof(MeshPtr mesh, int globalDofIndex){
  map<int,set<int> > fluxInds, fieldInds;
  mesh->getGlobalFieldFluxDofInds(fluxInds,fieldInds);
  bool value = false;
  if (fluxInds.find(globalDofIndex)!=fluxInds.end()){
    value = true;
  }
  return value;
}

void TestingUtilities::getDofIndices(MeshPtr mesh, set<int> &allFluxInds, map<int,vector<int> > &globalFluxInds, map<int, vector<int> > &globalFieldInds, map<int,vector<int> > &localFluxInds, map<int,vector<int> > &localFieldInds){
  
 
  // determine trialIDs
  vector< int > trialIDs = mesh->bilinearForm()->trialIDs();
  vector< int > fieldIDs;
  vector< int > fluxIDs;
  vector< int >::iterator idIt;

  for (idIt = trialIDs.begin();idIt!=trialIDs.end();idIt++){
    int trialID = *(idIt);
    if (!mesh->bilinearForm()->isFluxOrTrace(trialID)){ // if field
      fieldIDs.push_back(trialID);
    } else {
      fluxIDs.push_back(trialID);
    }
  } 

  // get all elems in mesh (more than just local info)
  vector< ElementPtr > activeElems = mesh->activeElements();
  vector< ElementPtr >::iterator elemIt;

  // gets dof indices
  for (elemIt=activeElems.begin();elemIt!=activeElems.end();elemIt++){
    int cellID = (*elemIt)->cellID();
    int globalCellIndex = (*elemIt)->globalCellIndex();
    int cellIndex = (*elemIt)->cellIndex();
    int numSides = (*elemIt)->numSides();
    ElementTypePtr elemType = (*elemIt)->elementType();
    
    // get local indices (for cell)
    vector<int> inds;
    for (idIt = fieldIDs.begin(); idIt != fieldIDs.end(); idIt++){
      int trialID = (*idIt);
      inds = elemType->trialOrderPtr->getDofIndices(trialID, 0);
      localFieldInds[cellID].insert(localFieldInds[cellID].end(), inds.begin(), inds.end()); 
    }
    inds.clear();
    for (idIt = fluxIDs.begin(); idIt != fluxIDs.end(); idIt++){
      int trialID = (*idIt);
      for (int sideIndex = 0;sideIndex<numSides;sideIndex++){	
	inds = elemType->trialOrderPtr->getDofIndices(trialID, sideIndex);
	localFluxInds[cellID].insert(localFluxInds[cellID].end(), inds.begin(), inds.end()); 
      }
    }

    // gets global indices (across all cells/all procs)
    for (int i = 0;i<localFieldInds[cellID].size();i++){
      int dofIndex = mesh->globalDofIndex(cellID,localFieldInds[cellID][i]);
      globalFieldInds[cellID].push_back(dofIndex);
    }
    for (int i = 0;i<localFluxInds[cellID].size();i++){
      int dofIndex = mesh->globalDofIndex(cellID,localFluxInds[cellID][i]);
      globalFluxInds[cellID].push_back(dofIndex);
      allFluxInds.insert(dofIndex); // all flux indices      
    }    
  }  
}

void TestingUtilities::testRoutine(){
  cout << "baaaaaaaaah" << endl;
}
