//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include <iostream>

#include "RefinementHistory.h"

#include "Mesh.h"

using namespace std;

using namespace Camellia;

RefinementType refinementTypeForString(string refTypeStr)
{
  if (refTypeStr == "h")
  {
    return H_REFINEMENT;
  }
  else if (refTypeStr == "hx")
  {
    return H_X_REFINEMENT;
  }
  else if (refTypeStr == "hy")
  {
    return H_Y_REFINEMENT;
  }
  else if (refTypeStr == "hu")
  {
    return H_UNREFINEMENT;
  }
  else if (refTypeStr == "hn")
  {
    return NULL_REFINEMENT;
  }
  else if (refTypeStr == "p")
  {
    return P_REFINEMENT;
  }
  else if (refTypeStr == "pu")
  {
    return P_UNREFINEMENT;
  }
  else if (refTypeStr == "re")
  {
    return REPARTITION_AND_REBUILD;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unknown refinement type string.");
//    return UNKNOWN_REFINEMENT;
  }
}

string stringForRefinementType(RefinementType refType)
{
  switch (refType)
  {
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
  case P_UNREFINEMENT:
    return "pu";
  case NULL_REFINEMENT:
    return "hn";
  case REPARTITION_AND_REBUILD:
    return "re";
  case UNKNOWN_REFINEMENT:
  default:
    return "UNKNOWN";
  }
}

RefinementPatternPtr refPatternForRefTypeLegacy(RefinementType refType, CellTopoPtrLegacy cellTopo)
{
  if (refType==H_REFINEMENT) return RefinementPattern::regularRefinementPattern(cellTopo->getKey());
  else if (refType==NULL_REFINEMENT) return RefinementPattern::noRefinementPattern(cellTopo);
  else if ((refType==H_X_REFINEMENT) && (cellTopo->getKey() == shards::Quadrilateral<4>::key))
  {
    return RefinementPattern::xAnisotropicRefinementPatternQuad();
  }
  else if ((refType==H_Y_REFINEMENT) && (cellTopo->getKey() == shards::Quadrilateral<4>::key))
  {
    return RefinementPattern::yAnisotropicRefinementPatternQuad();
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled refinement type");
  }
}

void RefinementHistory::didRepartition(MeshTopologyPtr meshTopo)
{
  _refinements.push_back(make_pair(REPARTITION_AND_REBUILD, std::set<GlobalIndexType>()));
  saveToFile(_saveFile);
}

RefinementPatternPtr RefinementHistory::refPatternForRefType(RefinementType refType, CellTopoPtr cellTopo)
{
  if (refType==H_REFINEMENT) return RefinementPattern::regularRefinementPattern(cellTopo);
  else if (refType==NULL_REFINEMENT) return RefinementPattern::noRefinementPattern(cellTopo);
  else
  {
    if (cellTopo->getTensorialDegree() > 1)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensorial degree > 1 not handled for anisotropic refinements");
    }
    if (refType==H_X_REFINEMENT)
    {
      if (cellTopo->getKey().first == shards::Quadrilateral<4>::key)
      {
        if (cellTopo->getKey().second == 0)
          return RefinementPattern::xAnisotropicRefinementPatternQuad();
        else if (cellTopo->getKey().second == 1)
          return RefinementPattern::xAnisotropicRefinementPatternQuadTimeExtruded();
      }
      else if (cellTopo->getKey().first == shards::Line<2>::key)
      {
        if (cellTopo->getKey().second == 1) // once-extruded line
          return RefinementPattern::xAnisotropicRefinementPatternExtrudedLine();
      }
    }
    else if (refType == H_Y_REFINEMENT)
    {
      if (cellTopo->getKey().second == shards::Quadrilateral<4>::key)
      {
        if (cellTopo->getKey().second == 0)
          return RefinementPattern::yAnisotropicRefinementPatternQuad();
        else if (cellTopo->getKey().second == 1)
          return RefinementPattern::yAnisotropicRefinementPatternQuadTimeExtruded();
      }
      else if (cellTopo->getKey().first == shards::Line<2>::key)
      {
        if (cellTopo->getKey().second == 1) // once-extruded line
          return RefinementPattern::yAnisotropicRefinementPatternExtrudedLine();
      }
    }
    
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled refinement type");
}

void RefinementHistory::hRefine(const set<GlobalIndexType> &cellIDs, Teuchos::RCP<RefinementPattern> refPattern,
                                bool repartitionAndRebuild)
{
  if (cellIDs.size() == 0) return;
  // figure out what type of refinement we have:
  int numChildren = refPattern->numChildren();
  int spaceDim = refPattern->verticesOnReferenceCell().dimension(1);
  RefinementType refType;
  if (numChildren == 1)
  {
    refType = NULL_REFINEMENT;
  }
  else if (spaceDim==1)
  {
    refType = H_REFINEMENT;
  }
  else if (spaceDim==2)
  {
    if (numChildren == 4)
    {
      refType = H_REFINEMENT;
    }
    else if (numChildren == 2)
    {
      if (refPattern->refinedNodes()(0,3,1) == 0.0)
      {
        // yAnisotropic: horizontal cut
        refType = H_Y_REFINEMENT;
      }
      else if (refPattern->refinedNodes()(1,0,0)==0.0)
      {
        refType = H_X_REFINEMENT;
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported refinement pattern");
      }
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported refinement pattern");
    }
  }
  else if (spaceDim==3)
  {
    if (numChildren==8)
    {
      refType = H_REFINEMENT;
    }
    else if (refPattern->parentTopology()->getTensorialDegree() == 0)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "RefinementHistory does not yet support anisotropic refinements in 3D.");
    }
    else
    {
      // get the spatial refinement pattern (on a temporal side)
      int sideCount = refPattern->parentTopology()->getSideCount();
      RefinementPatternPtr spatialRefPattern;
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
      {
        if (!refPattern->parentTopology()->sideIsSpatial(sideOrdinal))
        {
          spatialRefPattern = refPattern->sideRefinementPatterns()[sideOrdinal];
          break;
        }
      }
      if (spatialRefPattern->numChildren() != 2)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "RefinementHistory does not recognize this refinement pattern.");
      }
      else
      {
        if (spatialRefPattern->refinedNodes()(0,3,1) == 0.0)
        {
          // yAnisotropic: horizontal cut
          refType = H_Y_REFINEMENT;
        }
        else if (refPattern->refinedNodes()(1,0,0)==0.0)
        {
          refType = H_X_REFINEMENT;
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "RefinementHistory does not recognize this refinement pattern.");
        }
      }
    }
  }
  else
  {
    if ((spaceDim==4) && (numChildren==16))
    {
      refType = H_REFINEMENT;
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "RefinementHistory does not yet support this h-refinement and spaceDim combination.");
    }
  }
  Refinement ref = make_pair(refType, cellIDs);
  _refinements.push_back(ref);
  saveToFile(_saveFile);
}

void RefinementHistory::pRefine(const set<GlobalIndexType> &cellIDs, int pToAdd, bool repartitionAndRebuild)
{
  if (cellIDs.size() == 0) return;
  // right now, Refinement structure does not have support for different pToAdd values; we just do multiple p-refinements with pToAdd = 1.
  Refinement ref = (pToAdd >= 1) ? make_pair(P_REFINEMENT, cellIDs) : make_pair(P_UNREFINEMENT, cellIDs);
  const int numRefs = std::abs(pToAdd);
  for (int i=0; i<numRefs; i++)
  {
    _refinements.push_back(ref);
  }
  saveToFile(_saveFile);
}

void RefinementHistory::hUnrefine(const set<GlobalIndexType> &cellIDs, bool repartitionAndRebuild)
{
  if (cellIDs.size() == 0) return;
  Refinement ref = make_pair(H_UNREFINEMENT, cellIDs);
  _refinements.push_back(ref);
  saveToFile(_saveFile);
}

void RefinementHistory::playback(MeshPtr mesh)
{
  int refNumber = 0;
  int numRefs = int(_refinements.size());
  for (int refNumber=0; refNumber<numRefs; refNumber++)
  {
    const auto &ref = _refinements[refNumber];
    RefinementType refType = ref.first;
    set<GlobalIndexType> cellIDs = ref.second;

    // check that the cellIDs are all active nodes
    if (refType != H_UNREFINEMENT)
    {
      set<GlobalIndexType> activeIDs = mesh->getActiveCellIDsGlobal();
      for (set<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++)
      {
        int cellID = *cellIt;
        if (activeIDs.find(cellID) == activeIDs.end())
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID for refinement is not an active cell of the mesh");
        }
      }
    }

    switch (refType)
    {
    case P_REFINEMENT:
      mesh->pRefine(cellIDs,1,false);
      break;
    case P_UNREFINEMENT:
      mesh->pRefine(cellIDs,-1,false);
      break;
    case H_UNREFINEMENT:
      mesh->hUnrefine(cellIDs,false);
      break;
    case REPARTITION_AND_REBUILD:
      mesh->repartitionAndRebuild();
      break;
    default: // if we get here, it should be an h-refinement with a ref pattern
      {
        GlobalIndexType sampleCellID = *(cellIDs.begin());
        CellTopoPtr cellTopo = mesh->getElementType(sampleCellID)->cellTopoPtr;
        mesh->hRefine(cellIDs, refPatternForRefType(refType, cellTopo), false);
      }
    }
  }
}

void RefinementHistory::saveToFile(string fileName)
{
  if (fileName != "")
  {
    ofstream fout(fileName.c_str());
    for (vector< Refinement >::iterator refIt = _refinements.begin(); refIt != _refinements.end(); refIt++)
    {
      Refinement ref = *refIt;
      RefinementType refType = ref.first;
      set<GlobalIndexType> cellIDs = ref.second;
      fout << stringForRefinementType(refType) << " ";
      for (set<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++)
      {
        GlobalIndexType cellID = *cellIt;
        fout << cellID << " ";
      }
      fout << endl;
    }
  }
}

void RefinementHistory::loadFromFile(string fileName)
{
  ifstream fin(fileName.c_str());

  while (fin.good())
  {
    string refTypeStr;
    GlobalIndexType cellID;

    string line;
    std::getline(fin, line, '\n');
    std::istringstream linestream(line);
    linestream >> refTypeStr;
    set<GlobalIndexType> cellIDs;
    while (linestream.good())
    {
      cellID = -1;
      linestream >> cellID;
      if (cellID != -1)
      {
        cellIDs.insert(cellID);
      }
    }
    if (refTypeStr.length() > 0)
    {
      RefinementType refType = refinementTypeForString(refTypeStr);
      Refinement ref = make_pair(refType, cellIDs);
      _refinements.push_back(ref);
    }
  }
}

void RefinementHistory::setSaveFile(std::string &saveFile)
{
  _saveFile = saveFile;
}
