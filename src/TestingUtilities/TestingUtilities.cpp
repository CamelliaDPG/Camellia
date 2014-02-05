#include "TestingUtilities.h"
#include "ElementType.h"
#include "Element.h"

void TestingUtilities::initializeSolnCoeffs(SolutionPtr solution){
  map< pair<IndexType,IndexType>, IndexType> localToGlobalMap = solution->mesh()->getLocalToGlobalMap();
  map< pair<IndexType,IndexType>, IndexType>::iterator it;
  for (it = localToGlobalMap.begin();it!=localToGlobalMap.end();it++){
    pair<int,int> cellID_dofIndex = it->first;
    int cellID = cellID_dofIndex.first;
    int numLocalTrialDofs = solution->mesh()->getElement(cellID)->elementType()->trialOrderPtr->totalDofs();
    FieldContainer<double> dofs(numLocalTrialDofs);
    dofs.initialize(0.0); 
    solution->setSolnCoeffsForCellID(dofs,cellID);
  }
}

// checks if dof has a BC applied to it
bool TestingUtilities::isBCDof(GlobalIndexType globalDofIndex, SolutionPtr solution){
  FieldContainer<GlobalIndexType> globalIndices;
  FieldContainer<double> globalValues;
  solution->mesh()->boundary().bcsToImpose(globalIndices, globalValues, *(solution->bc()));
  for (int i = 0;i < globalIndices.size(); i++){
    if (globalIndices[i]==globalDofIndex){
      return true;
    }
  }
  return false;  
}

bool TestingUtilities::isFluxOrTraceDof(MeshPtr mesh, GlobalIndexType globalDofIndex){
  map<GlobalIndexType,set<GlobalIndexType> > fluxInds, fieldInds;
  getGlobalFieldFluxDofInds(mesh, fluxInds,fieldInds);
  bool value = false;
  set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIt = activeCellIDs.begin(); cellIt != activeCellIDs.end(); cellIt++) {
    GlobalIndexType cellID = *cellIt;
    ElementPtr elem = mesh->getElement(cellID);
    if (fluxInds[cellID].find(globalDofIndex)!=fluxInds[cellID].end()){
      value = true;
    }
  }
  return value;
}
void TestingUtilities::setSolnCoeffForGlobalDofIndex(SolutionPtr solution, double solnCoeff, GlobalIndexType dofIndex) {
  map< pair<GlobalIndexType,IndexType>, GlobalIndexType> localToGlobalMap = solution->mesh()->getLocalToGlobalMap();
  map< pair<GlobalIndexType,IndexType>, GlobalIndexType>::iterator it;
  for (it = localToGlobalMap.begin();it!=localToGlobalMap.end();it++){
    pair<IndexType,GlobalIndexType> cellID_dofIndex = it->first;
    GlobalIndexType currentGlobalDofIndex = it->second;
    if (currentGlobalDofIndex==dofIndex) {
      GlobalIndexType cellID = cellID_dofIndex.first;
      int localDofIndex = cellID_dofIndex.second;
      int numLocalTrialDofs = solution->mesh()->getElement(cellID)->elementType()->trialOrderPtr->totalDofs();
      FieldContainer<double> dofs(numLocalTrialDofs);
      dofs.initialize(0.0); // inefficient; can do better.
      dofs(localDofIndex) = solnCoeff;
      solution->setSolnCoeffsForCellID(dofs,cellID);
    }
  }
}

void TestingUtilities::getGlobalFieldFluxDofInds(MeshPtr mesh, map<GlobalIndexType,set<GlobalIndexType> > &fluxIndices, map<GlobalIndexType,set<GlobalIndexType> > &fieldIndices) {
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
    GlobalIndexType cellID = (*elemIt)->cellID();
    int numSides = (*elemIt)->numSides();
    ElementTypePtr elemType = (*elemIt)->elementType();
    
    // get indices (for cell)
    vector<int> indices;
    for (idIt = fieldIDs.begin(); idIt != fieldIDs.end(); idIt++){
      int trialID = (*idIt);
      indices = elemType->trialOrderPtr->getDofIndices(trialID, 0);
      for (int i = 0;i<indices.size();++i){
        fieldIndices[cellID].insert(mesh->globalDofIndex(cellID,indices[i]));
      }
    }
    vector<int> fInds;
    for (idIt = fluxIDs.begin(); idIt != fluxIDs.end(); idIt++){
      int trialID = (*idIt);
      for (int sideIndex = 0;sideIndex < numSides;sideIndex++){	
        fInds = elemType->trialOrderPtr->getDofIndices(trialID, sideIndex);
        for (int i = 0;i<fInds.size();++i){
          fluxIndices[cellID].insert(mesh->globalDofIndex(cellID,fInds[i]));
        }
      }
    }
  }  
}
/*
// added by Jesse - accumulates flux/field local dof indices into user-provided maps
void TestingUtilities::getFieldFluxDofInds(MeshPtr mesh, map<int,set<int> > &localFluxInds, map<int,set<int> > &localFieldInds){
  
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
    int numSides = (*elemIt)->numSides();
    ElementTypePtr elemType = (*elemIt)->elementType();
    
    // get local indices (for cell)
    vector<int> inds;
    for (idIt = fieldIDs.begin(); idIt != fieldIDs.end(); idIt++){
      int trialID = (*idIt);
      inds = elemType->trialOrderPtr->getDofIndices(trialID, 0);
      for (int i = 0;i<inds.size();++i){
	localFieldInds[cellID].insert(inds[i]);
      }
    }
    inds.clear();
    for (idIt = fluxIDs.begin(); idIt != fluxIDs.end(); idIt++){
      int trialID = (*idIt);
      for (int sideIndex = 0;sideIndex<numSides;sideIndex++){	
	inds = elemType->trialOrderPtr->getDofIndices(trialID, sideIndex);
	for (int i = 0;i<inds.size();++i){
	  localFluxInds[cellID].insert(inds[i]);
	}	
      }
    }
  }  
}
*/

/*
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
*/
