//
//  RefinementHistory.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/2/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "RefinementHistory.h"

using namespace std;

RefinementType refinementTypeForString(string refTypeStr) {
  if (refTypeStr == "h") {
    return H_REFINEMENT;
  } else if (refTypeStr == "hx") {
    return H_X_REFINEMENT;
  } else if (refTypeStr == "hy") {
    return H_Y_REFINEMENT;
  } else if (refTypeStr == "hu") {
    return H_UNREFINEMENT;
  } else if (refTypeStr == "p") {
    return P_REFINEMENT;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unknown refinement type string.");
    return UNKNOWN_REFINEMENT;
  }
}

string stringForRefinementType(RefinementType refType) {
  switch (refType) {
    case H_REFINEMENT:
      return "h";
    case H_X_REFINEMENT:
      return "hx";
    case H_Y_REFINEMENT:
      return "hy";
    case P_REFINEMENT:
      return "p";
    case H_UNREFINEMENT:
      return "hu";
    case UNKNOWN_REFINEMENT:
    default:
      return "UNKNOWN";
  }
}

void RefinementHistory::hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
  if (cellIDs.size() == 0) return;
  // figure out what type of refinement we have:
  int numChildren = refPattern->numChildren();
  RefinementType refType;
  if (numChildren == 4) {
    refType = H_REFINEMENT;
  } else if (numChildren == 2) {
    if (refPattern->refinedNodes()(0,3,1) == 0.0) {
      // yAnisotropic: horizontal cut
      refType = H_Y_REFINEMENT;
    } else if (refPattern->refinedNodes()(1,0,0)==0.0) {
      refType = H_X_REFINEMENT;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported refinement pattern");
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported refinement pattern");
  }
  Refinement ref = make_pair(refType, cellIDs);
  _refinements.push_back(ref);
}

void RefinementHistory::pRefine(const set<GlobalIndexType> &cellIDs) {
  if (cellIDs.size() == 0) return;
  Refinement ref = make_pair(P_REFINEMENT, cellIDs);
  _refinements.push_back(ref);
}

void RefinementHistory::hUnrefine(const set<GlobalIndexType> &cellIDs) {
  if (cellIDs.size() == 0) return;
  Refinement ref = make_pair(H_UNREFINEMENT, cellIDs);
  _refinements.push_back(ref);
}

void RefinementHistory::playback(MeshPtr mesh) {
  for (vector< Refinement >::iterator refIt = _refinements.begin(); refIt != _refinements.end(); refIt++) {
    Refinement ref = *refIt;
    RefinementType refType = ref.first;
    set<GlobalIndexType> cellIDs = ref.second;
    
    // check that the cellIDs are all active nodes
    if (refType != H_UNREFINEMENT) {
//      cout << stringForRefinementType(refType) << " ";
      set<GlobalIndexType> activeIDs = mesh->getActiveCellIDs();
      for (set<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
        int cellID = *cellIt;
//        cout << cellID << " ";
        if (activeIDs.find(cellID) == activeIDs.end()) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID for refinement is not an active cell of the mesh");
        }
      }
//      cout << endl;
    }
    GlobalIndexType sampleCellID = *(cellIDs.begin());
    bool quadCells = mesh->getElement(sampleCellID)->numSides() == 4;
    
    switch (refType) {
      case H_REFINEMENT:
        if (quadCells)
          mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternQuad());
        else
          mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternTriangle());
        break;
      case H_X_REFINEMENT:
        mesh->hRefine(cellIDs, RefinementPattern::xAnisotropicRefinementPatternQuad());
        break;
      case H_Y_REFINEMENT:
        mesh->hRefine(cellIDs, RefinementPattern::yAnisotropicRefinementPatternQuad());
        break;
      case P_REFINEMENT:
        mesh->pRefine(cellIDs);
        break;
      case H_UNREFINEMENT:
        mesh->hUnrefine(cellIDs);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled refinement type");
    }
  }
}

void RefinementHistory::saveToFile(string fileName) {
  ofstream fout(fileName.c_str());
  for (vector< Refinement >::iterator refIt = _refinements.begin(); refIt != _refinements.end(); refIt++) {
    Refinement ref = *refIt;
    RefinementType refType = ref.first;
    set<GlobalIndexType> cellIDs = ref.second;
    fout << stringForRefinementType(refType) << " ";
    for (set<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
      GlobalIndexType cellID = *cellIt;
      fout << cellID << " ";
    }
    fout << endl;
  }
}

void RefinementHistory::loadFromFile(string fileName) {
  ifstream fin(fileName.c_str());
  
  while (fin.good()) {
    string refTypeStr;
    GlobalIndexType cellID;
    
    string line;
    std::getline(fin, line, '\n');
    std::istringstream linestream(line);
    linestream >> refTypeStr;
    set<GlobalIndexType> cellIDs;
    while (linestream.good()) {
      linestream >> cellID;
      cellIDs.insert(cellID);
    }
    if (refTypeStr.length() > 0) {
      RefinementType refType = refinementTypeForString(refTypeStr);
      Refinement ref = make_pair(refType, cellIDs);
      _refinements.push_back(ref);
    }
  }
}
