//
//  RefinementHistory.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/2/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include "RefinementHistory.h"

#include "Mesh.h"

using namespace std;

using namespace Camellia;

RefinementType refinementTypeForString(string refTypeStr) {
  if (refTypeStr == "h") {
    return H_REFINEMENT;
  } else if (refTypeStr == "hx") {
    return H_X_REFINEMENT;
  } else if (refTypeStr == "hy") {
    return H_Y_REFINEMENT;
  } else if (refTypeStr == "hu") {
    return H_UNREFINEMENT;
  } else if (refTypeStr == "hn") {
    return NULL_REFINEMENT;
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
    case NULL_REFINEMENT:
      return "hn";
    case UNKNOWN_REFINEMENT:
    default:
      return "UNKNOWN";
  }
}

RefinementPatternPtr refPatternForRefTypeLegacy(RefinementType refType, CellTopoPtrLegacy cellTopo) {
  if (refType==H_REFINEMENT) return RefinementPattern::regularRefinementPattern(cellTopo->getKey());
  else if (refType==NULL_REFINEMENT) return RefinementPattern::noRefinementPattern(cellTopo);
  else if ((refType==H_X_REFINEMENT) && (cellTopo->getKey() == shards::Quadrilateral<4>::key)) {
    return RefinementPattern::xAnisotropicRefinementPatternQuad();
  } else if ((refType==H_Y_REFINEMENT) && (cellTopo->getKey() == shards::Quadrilateral<4>::key)) {
    return RefinementPattern::yAnisotropicRefinementPatternQuad();
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled refinement type");
  }
}

RefinementPatternPtr RefinementHistory::refPatternForRefType(RefinementType refType, CellTopoPtr cellTopo) {
  if (refType==H_REFINEMENT) return RefinementPattern::regularRefinementPattern(cellTopo);
  else if (refType==NULL_REFINEMENT) return RefinementPattern::noRefinementPattern(cellTopo);
  else {
    if (cellTopo->getTensorialDegree() > 0) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensorial degree > 0 not handled for anisotropic refinements");
    }
    if ((refType==H_X_REFINEMENT) && (cellTopo->getKey().first == shards::Quadrilateral<4>::key)) {
      return RefinementPattern::xAnisotropicRefinementPatternQuad();
    } else if ((refType==H_Y_REFINEMENT) && (cellTopo->getKey().second == shards::Quadrilateral<4>::key)) {
      return RefinementPattern::yAnisotropicRefinementPatternQuad();
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled refinement type");
    }
  }
}

void RefinementHistory::hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern) {
  if (cellIDs.size() == 0) return;
  // figure out what type of refinement we have:
  int numChildren = refPattern->numChildren();
  int spaceDim = refPattern->verticesOnReferenceCell().dimension(1);
  RefinementType refType;
  if (numChildren == 1) {
    refType = NULL_REFINEMENT;
  } else if (spaceDim==1) {
    refType = H_REFINEMENT;
  } else if (spaceDim==2) {
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
  } else if (spaceDim==3) {
    if (numChildren==8) {
      refType = H_REFINEMENT;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "RefinementHistory does not yet support anisotropic refinements in 3D.");
    }
  } else {
    if ((spaceDim==4) && (numChildren==16)) {
      refType = H_REFINEMENT;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "RefinementHistory does not yet support this h-refinement and spaceDim combination.");
    }
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
      set<GlobalIndexType> activeIDs = mesh->getActiveCellIDs();
      for (set<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
        int cellID = *cellIt;
        if (activeIDs.find(cellID) == activeIDs.end()) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID for refinement is not an active cell of the mesh");
        }
      }
    }
    GlobalIndexType sampleCellID = *(cellIDs.begin());
    CellTopoPtr cellTopo = mesh->getElementType(sampleCellID)->cellTopoPtr;
    
    switch (refType) {
      case P_REFINEMENT:
        mesh->pRefine(cellIDs);
        break;
      case H_UNREFINEMENT:
        mesh->hUnrefine(cellIDs);
        break;
      default: // if we get here, it should be an h-refinement with a ref pattern
        mesh->hRefine(cellIDs, refPatternForRefType(refType, cellTopo));
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

#ifdef HAVE_EPETRAEXT_HDF5
void RefinementHistory::saveToHDF5(EpetraExt::HDF5 &hdf5) {
  vector<int> histArray;
  for (vector< Refinement >::iterator refIt = _refinements.begin(); refIt != _refinements.end(); refIt++) {
    Refinement ref = *refIt;
    RefinementType refType = ref.first;
    set<GlobalIndexType> cellIDs = ref.second;
    histArray.push_back(refType);
    histArray.push_back(cellIDs.size());
    for (set<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
      GlobalIndexType cellID = *cellIt;
      histArray.push_back(cellID);
    }
  }
  int histArraySize = histArray.size();
  hdf5.Write("Mesh", "histArraySize", histArraySize);
  if (histArraySize > 0) {
    hdf5.Write("Mesh", "refinementHistory", H5T_NATIVE_INT, histArraySize, &histArray[0]);
  } else {
  }
}
#endif
