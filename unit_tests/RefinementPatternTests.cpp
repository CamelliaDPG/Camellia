//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  RefinementPatternTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 12/10/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Camellia_FieldContainer.hpp"

#include "CamelliaCellTools.h"
#include "CamelliaTestingHelpers.h"
#include "RefinementPattern.h"

using namespace Camellia;

namespace
{
  vector< CellTopoPtr > getShardsTopologies()
  {
    vector< CellTopoPtr > shardsTopologies;

    shardsTopologies.push_back(CellTopology::point());
    shardsTopologies.push_back(CellTopology::line());
    shardsTopologies.push_back(CellTopology::quad());
    shardsTopologies.push_back(CellTopology::triangle());
    shardsTopologies.push_back(CellTopology::hexahedron());
    //  shardsTopologies.push_back(CellTopology::tetrahedron()); // tetrahedron not yet supported by permutation
    return shardsTopologies;
  }

  void testGeneralizedRefinementPatternTiersAreCompatible(CellTopoPtr volumeTopo, int refLevels, bool &success, Teuchos::FancyOStream &out)
  {

    /*
     The GeneralizedRefinementBranch structure is a bit redundant when there are multiple tiers, in that:
       Tier n's leafSubcellDimension == Tier (n+1)'s rootDimension
     In this test, we construct an n-level refinement branch (where n = refLevels), and then consider the subcells of the
     leaf cell.  For each of these we request a GeneralizedRefinementBranch.  When the returned branch has two tiers,
     we check that the above constraint is satisfied.
     */
    RefinementPatternPtr volumeRefinement = RefinementPattern::regularRefinementPattern(volumeTopo);
    // create a n-level refinement branch

    RefinementBranch refBranch;
    for (int i=0; i<refLevels; i++)
    {
      int childOrdinal = (i==0) ? 0 : 1;
      refBranch.push_back({volumeRefinement.get(),childOrdinal});
    }

    for (int d=0; d<volumeTopo->getDimension(); d++)
    {
      int subcellCount = volumeTopo->getSubcellCount(d);
      for (int subcellOrdinal=0; subcellOrdinal<subcellCount; subcellOrdinal++)
      {
        GeneralizedRefinementBranch genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(refBranch, d,
                                                                                                                subcellOrdinal);
        { // DEBUGGING
          if (genRefBranch.size() >= 2)
          {
            out << "genRefBranch for d=" << d << ", subcellOrdinal=" << subcellOrdinal << " has " << genRefBranch.size() << " tiers.\n";
            GeneralizedRefinementBranch genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(refBranch, d,
                                                                                                                    subcellOrdinal);
          }
        }
        for (int tierNumber=0; tierNumber<genRefBranch.size()-1; tierNumber++)
        { // note that this for loop only does something for ref genRefBranches with >= 2 tiers
          RefinementBranchTier* refTier = &genRefBranch[tierNumber];
          RefinementBranchTier* nextRefTier = &genRefBranch[tierNumber+1];
          TEST_EQUALITY(refTier->leafSubcellDimension, nextRefTier->rootDimension);
        }
      }
    }
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, ChildIsInteriorTriangle )
  {
    RefinementPatternPtr triangleRef = RefinementPattern::regularRefinementPatternTriangle();
    TEST_EQUALITY(triangleRef->childIsInterior(0), false);
    TEST_EQUALITY(triangleRef->childIsInterior(1), true);
    TEST_EQUALITY(triangleRef->childIsInterior(2), false);
    TEST_EQUALITY(triangleRef->childIsInterior(3), false);
  }
  
  TEUCHOS_UNIT_TEST( RefinementPattern, ChildIsInteriorQuad )
  {
    RefinementPatternPtr quadRef = RefinementPattern::regularRefinementPatternQuad();
    for (int childOrdinal = 0; childOrdinal < quadRef->numChildren(); childOrdinal++)
    {
      TEST_EQUALITY(quadRef->childIsInterior(0), false);
    }
  }
  
  TEUCHOS_UNIT_TEST( RefinementPattern, ChildIsInteriorHexahedron )
  {
    RefinementPatternPtr hexRef = RefinementPattern::regularRefinementPatternHexahedron();
    for (int childOrdinal = 0; childOrdinal < hexRef->numChildren(); childOrdinal++)
    {
      TEST_EQUALITY(hexRef->childIsInterior(0), false);
    }
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, ExtrudedLineAnisotropicRefinement )
  {
    MeshTopologyPtr baseMeshTopo = MeshFactory::intervalMeshTopology(0.0, 1.0, 2);
    MeshTopologyPtr meshTopo = MeshFactory::extrudedMeshTopology(baseMeshTopo, {{0.,1.}});
    
//    cout << "All mesh entities before refinement:\n";
//    meshTopo->printAllEntities();
    
    auto extrudedLine_xRef = RefinementPattern::xAnisotropicRefinementPatternExtrudedLine();
    
    meshTopo->refineCell(0,extrudedLine_xRef,meshTopo->nextCellID());
    
//    cout << "All mesh entities after refinement:\n";
//    meshTopo->printAllEntities();
//    TODO: add some actual correctness test here!
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, GeneralizedRefinementBranchForLeafSubcell_HangingVertexOnOutsideEdge )
  {
    RefinementPatternPtr quadRefinement = RefinementPattern::regularRefinementPatternQuad();
    // create a two-refinement branch, selecting lower-left quad each time:
    RefinementBranch refBranch = { {quadRefinement.get(),0}, {quadRefinement.get(),0} };

    /* taking the 1 vertex as our subcell on the leaf cell, we expect GeneralizedRefinementBranch like so.
     From leaf to root:
     1.       previousTierTopo = Quad;
                 rootDimension = 1;
    previousTierSubcellOrdinal = 0
                     refBranch = {{lineRefinement,0},{lineRefinement,0}};
                leafSubcellDim = 0;
            leafSubcellOrdinal = 1 // 1 in the leaf line
     */
    unsigned vertexDim = 0;
    unsigned vertexOrdinalInCell = 1;
    GeneralizedRefinementBranch genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(refBranch,
                                                                                                            vertexDim,
                                                                                                            vertexOrdinalInCell);
    TEST_EQUALITY(genRefBranch.size(), 1);

    TEST_EQUALITY(genRefBranch[0].previousTierTopo->getKey(), CellTopology::quad()->getKey());
    TEST_EQUALITY(genRefBranch[0].rootDimension, 1);
    TEST_EQUALITY(genRefBranch[0].previousTierSubcellOrdinal, 0);

    TEST_EQUALITY(genRefBranch[0].refBranch.size(), 2);
    TEST_EQUALITY(genRefBranch[0].refBranch[0].first->parentTopology()->getKey(), CellTopology::line()->getKey());
    TEST_EQUALITY(genRefBranch[0].refBranch[0].second, 0);
    TEST_EQUALITY(genRefBranch[0].refBranch.size(), 2);
    TEST_EQUALITY(genRefBranch[0].refBranch[1].first->parentTopology()->getKey(), CellTopology::line()->getKey());
    TEST_EQUALITY(genRefBranch[0].refBranch[1].second, 0);
    TEST_EQUALITY(genRefBranch[0].leafSubcellDimension, 0);
    TEST_EQUALITY(genRefBranch[0].leafSubcellOrdinal, 1);

    // second test: much the same thing, but this time select the 1 child in the second refinement:
    // create a two-refinement branch, selecting lower-left quad, then the lower right:
    refBranch = { {quadRefinement.get(),0}, {quadRefinement.get(),1} };
    /* taking the 1 vertex as our subcell on the leaf cell, we expect GeneralizedRefinementBranch like so.
     From leaf to root:
     1. previousTierTopo = Line;
        rootDimension = 0;
        previousTierSubcellOrdinal = 1; // node 1 in the line
        refBranch = {{nodeRefinement,0}};
        leafSubcellDim = 0;
        leafSubcellOrdinal = 0 // 0 in the leaf node
     0. previousTierTopo = Quad;
        rootDimension = 1;
        previousTierSubcellOrdinal = 0; // edge 0 in the quad
        refBranch = {{lineRefinement,0}};
        leafSubcellDim = 0;
        leafSubcellOrdinal = 1 // 1 in the leaf line
     */
    vertexOrdinalInCell = 1;
    genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(refBranch, vertexDim, vertexOrdinalInCell);
    TEST_EQUALITY(genRefBranch.size(), 2);

    TEST_EQUALITY(genRefBranch[0].previousTierTopo->getKey(), CellTopology::quad()->getKey());
    TEST_EQUALITY(genRefBranch[0].rootDimension, 1);
    TEST_EQUALITY(genRefBranch[0].previousTierSubcellOrdinal, 0);

    TEST_EQUALITY(genRefBranch[0].refBranch.size(), 1);
    TEST_EQUALITY(genRefBranch[0].refBranch[0].first->parentTopology()->getKey(), CellTopology::line()->getKey());
    TEST_EQUALITY(genRefBranch[0].refBranch[0].second, 0);
    TEST_EQUALITY(genRefBranch[0].leafSubcellDimension, 0);
    TEST_EQUALITY(genRefBranch[0].leafSubcellOrdinal, 1);

    TEST_EQUALITY(genRefBranch[1].previousTierTopo->getKey(), CellTopology::line()->getKey());
    TEST_EQUALITY(genRefBranch[1].rootDimension, 0);
    TEST_EQUALITY(genRefBranch[1].previousTierSubcellOrdinal, 1);

    TEST_EQUALITY(genRefBranch[1].refBranch.size(), 1);
    TEST_EQUALITY(genRefBranch[1].refBranch[0].first->parentTopology()->getKey(), CellTopology::point()->getKey());
    TEST_EQUALITY(genRefBranch[1].refBranch[0].second, 0);
    TEST_EQUALITY(genRefBranch[1].leafSubcellDimension, 0);
    TEST_EQUALITY(genRefBranch[1].leafSubcellOrdinal, 0);
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, GeneralizedRefinementBranchForLeafSubcell_HangingVertexOnOutsideEdgeForInteriorTriangle )
  {
    RefinementPatternPtr triangleRefinement = RefinementPattern::regularRefinementPatternTriangle();
    // create a one-refinement branch, selecting middle triangle (child 1):
    RefinementBranch refBranch = { {triangleRefinement.get(),1} };

    /* taking the 2 vertex as our subcell on the leaf cell, we expect GeneralizedRefinementBranch like so.
     From leaf to root:
     1.       previousTierTopo = Triangle;
     rootDimension = 1;
     previousTierSubcellOrdinal = 2
     refBranch = {{lineRefinement,0}}; OR {{lineRefinement,1}};
     leafSubcellDim = 0;
     leafSubcellOrdinal = 1 OR 0
     */
    unsigned vertexDim = 0;
    unsigned vertexOrdinalInCell = 2;
    GeneralizedRefinementBranch genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(refBranch,
                                                                                                            vertexDim,
                                                                                                            vertexOrdinalInCell);
    TEST_EQUALITY(genRefBranch.size(), 1);

    TEST_EQUALITY(genRefBranch[0].previousTierTopo->getKey(), CellTopology::triangle()->getKey());
    TEST_EQUALITY(genRefBranch[0].rootDimension, 1);
    TEST_EQUALITY(genRefBranch[0].previousTierSubcellOrdinal, 2);

    TEST_EQUALITY(genRefBranch[0].refBranch.size(), 1);
    TEST_EQUALITY(genRefBranch[0].refBranch[0].first->parentTopology()->getKey(), CellTopology::line()->getKey());

    TEST_EQUALITY(genRefBranch[0].leafSubcellDimension, 0);
    // which vertex ordinal depends on which of the two edges was selected:
    if (genRefBranch[0].refBranch[0].second == 0)
    {
      TEST_EQUALITY(genRefBranch[0].leafSubcellOrdinal, 1);
    }
    else
    {
      TEST_EQUALITY(genRefBranch[0].leafSubcellOrdinal, 0);
    }
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, GeneralizedRefinementBranchForLeafSubcell_InteriorTriangleEdge )
  {
    RefinementPatternPtr triangleRefinement = RefinementPattern::regularRefinementPatternTriangle();
    // create a one-refinement branch, selecting middle triangle (child 1):
    RefinementBranch refBranch = { {triangleRefinement.get(),1} };

    /* taking the 0 edge as our subcell on the leaf cell, we expect GeneralizedRefinementBranch like so.
     From leaf to root:
     1.  previousTierTopo = Triangle;
         rootDimension = 2;
         previousTierSubcellOrdinal = 0
         refBranch = {{triangleRefinement,1}};
         leafSubcellDim = 1;
         leafSubcellOrdinal = 0
     */
    unsigned edgeDim = 1;
    unsigned edgeOrdinalInCell = 0;
    GeneralizedRefinementBranch genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(refBranch,
                                                                                                            edgeDim,
                                                                                                            edgeOrdinalInCell);
    TEST_EQUALITY(genRefBranch.size(), 1);

    TEST_EQUALITY(genRefBranch[0].previousTierTopo->getKey(), CellTopology::triangle()->getKey());
    TEST_EQUALITY(genRefBranch[0].rootDimension, 2);
    TEST_EQUALITY(genRefBranch[0].previousTierSubcellOrdinal, 0);

    TEST_EQUALITY(genRefBranch[0].refBranch.size(), 1);
    TEST_EQUALITY(genRefBranch[0].refBranch[0].first->parentTopology()->getKey(), CellTopology::triangle()->getKey());
    TEST_EQUALITY(genRefBranch[0].refBranch[0].second, 1);

    TEST_EQUALITY(genRefBranch[0].leafSubcellDimension, 1);
    TEST_EQUALITY(genRefBranch[0].leafSubcellOrdinal, 0);
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, GeneralizedRefinementBranchForLeafSubcell_ConformingVertex )
  {
    RefinementPatternPtr quadRefinement = RefinementPattern::regularRefinementPatternQuad();
    // create a two-refinement branch, selecting lower-left quad each time:
    RefinementBranch refBranch = { {quadRefinement.get(),0}, {quadRefinement.get(),0} };

    /* taking the 0 vertex as our subcell on the leaf cell, we expect GeneralizedRefinementBranch like so.
     From leaf to root:
     1.       previousTierTopo = Quad;
                 rootDimension = 0;
    previousTierSubcellOrdinal = 0
                     refBranch = {{nodeNoRefinement,0},{nodeNoRefinement,0}};
                leafSubcellDim = 0;
            leafSubcellOrdinal = 0
     */
    unsigned vertexDim = 0;
    unsigned vertexOrdinalInCell = 0;
    GeneralizedRefinementBranch genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(refBranch,
                                                                                                            vertexDim,
                                                                                                            vertexOrdinalInCell);
    TEST_EQUALITY(genRefBranch[0].previousTierTopo->getKey(), CellTopology::quad()->getKey());
    TEST_EQUALITY(genRefBranch[0].rootDimension, 0);
    TEST_EQUALITY(genRefBranch[0].previousTierSubcellOrdinal, 0);
    TEST_EQUALITY(genRefBranch[0].refBranch[0].first->parentTopology()->getKey(), CellTopology::point()->getKey());
    TEST_EQUALITY(genRefBranch[0].refBranch[0].second, 0);
    TEST_EQUALITY(genRefBranch[0].refBranch[1].first->parentTopology()->getKey(), CellTopology::point()->getKey());
    TEST_EQUALITY(genRefBranch[0].refBranch[1].second, 0);
    TEST_EQUALITY(genRefBranch[0].leafSubcellDimension, 0);
    TEST_EQUALITY(genRefBranch[0].leafSubcellOrdinal, 0);
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, GeneralizedRefinementBranchForLeafSubcell_HangingVertexInsideVolume )
  {
    RefinementPatternPtr quadRefinement = RefinementPattern::regularRefinementPatternQuad();
    // create a two-refinement branch, selecting lower-left quad each time:
    RefinementBranch refBranch = { {quadRefinement.get(),0}, {quadRefinement.get(),0} };

    /* taking the 2 vertex as our subcell on the leaf cell, we expect GeneralizedRefinementBranch like so.
     From leaf to root:
     1.       previousTierTopo = Quad;
     rootDimension = 2;
     previousTierSubcellOrdinal = 0
     refBranch = {{Quad,0},{Quad,0}};
     leafSubcellDim = 0;
     leafSubcellOrdinal = 2
     */
    unsigned vertexDim = 0;
    unsigned vertexOrdinalInCell = 2;
    GeneralizedRefinementBranch genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(refBranch,
                                                                                                            vertexDim,
                                                                                                            vertexOrdinalInCell);
    TEST_EQUALITY(genRefBranch[0].previousTierTopo->getKey(), CellTopology::quad()->getKey());
    TEST_EQUALITY(genRefBranch[0].rootDimension, 2);
    TEST_EQUALITY(genRefBranch[0].previousTierSubcellOrdinal, 0);
    TEST_EQUALITY(genRefBranch[0].refBranch[0].first->parentTopology()->getKey(), CellTopology::quad()->getKey());
    TEST_EQUALITY(genRefBranch[0].refBranch[0].second, 0);
    TEST_EQUALITY(genRefBranch[0].refBranch[1].first->parentTopology()->getKey(), CellTopology::quad()->getKey());
    TEST_EQUALITY(genRefBranch[0].refBranch[1].second, 0);
    TEST_EQUALITY(genRefBranch[0].leafSubcellDimension, 0);
    TEST_EQUALITY(genRefBranch[0].leafSubcellOrdinal, 2);
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, GeneralizedRefinementBranchForLeafSubcell_QuadrilateralTwoLevelTiersCompatible )
  {
    CellTopoPtr quadTopo = CellTopology::quad();
    int refLevels = 2;
    testGeneralizedRefinementPatternTiersAreCompatible(quadTopo, refLevels, success, out);
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, GeneralizedRefinementBranchForLeafSubcell_HexahedronTwoLevelTiersCompatible )
  {
    /*
     The GeneralizedRefinementBranch structure is a bit redundant when there are multiple tiers, in that:
       Tier n's leafSubcellDimension == Tier (n+1)'s previousTierTopo->getDimension()
     and
       Tier n's leafSubcellOrdinal == Tier (n+1)'s previousTierSubcellOrdinal.
     In this test, we construct a two-level hexahedral refinement branch, and then consider the subcells of the
     leaf cell.  For each of these we request a GeneralizedRefinementBranch.  When the returned branch has two tiers,
     we check that the above constraint is satisfied.
     */
    CellTopoPtr volumeTopo = CellTopology::hexahedron();
    int refLevels = 2;
    testGeneralizedRefinementPatternTiersAreCompatible(volumeTopo, refLevels, success, out);
  }
  
  TEUCHOS_UNIT_TEST( RefinementPattern, KeysAreDistinct )
  {
    vector<CellTopoPtr> shardsTopologies = getShardsTopologies();
    vector<CellTopoPtr> cellTopologies = shardsTopologies;
    for (CellTopoPtr shardsTopo : shardsTopologies)
    {
      // let's also include regular refinements of cell topologies of tensorial degree 1
      int tensorialDegree = 1;
      CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(shardsTopo, tensorialDegree);
      cellTopologies.push_back(spaceTimeTopo);
    }
    
    vector<RefinementPatternPtr> regularRefinements;
    map<RefinementPatternKey,RefinementPatternPtr> regularRefinementsMap;
    for (CellTopoPtr cellTopo : cellTopologies)
    {
      if (cellTopo->getKey() == CellTopology::point()->getKey()) continue; // skip this one because it's the same as the null refinement we count below.
      RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo);
      regularRefinementsMap[refPattern->getKey()] = refPattern;
      regularRefinements.push_back(refPattern);
    }
    // if there is no key collision, then the map size and the vector size will be the same
    TEST_EQUALITY(regularRefinements.size(), regularRefinementsMap.size());
    
    vector<RefinementPatternPtr> noRefinements;
    map<RefinementPatternKey,RefinementPatternPtr> noRefinementsMap;
    for (CellTopoPtr cellTopo : cellTopologies)
    {
      RefinementPatternPtr refPattern = RefinementPattern::noRefinementPattern(cellTopo);
      
      if (noRefinementsMap.find(refPattern->getKey()) != noRefinementsMap.end())
      {
        out << "Key collision for null refinement pattern on cellTopo " << cellTopo->getName();
        out << " and cellTopo " << noRefinementsMap[refPattern->getKey()]->parentTopology()->getName() << endl;
        success = false;
      }
      if (regularRefinementsMap.find(refPattern->getKey()) != regularRefinementsMap.end())
      {
        out << "Key collision for null refinement pattern and regular refinement pattern for cellTopo " << cellTopo->getName() << endl;
        success = false;
      }
      
      noRefinementsMap[refPattern->getKey()] = refPattern;
      noRefinements.push_back(refPattern);
    }
    // if there is no key collision, then the map size and the vector size will be the same
    TEST_EQUALITY(noRefinements.size(), noRefinementsMap.size());
    
    // anisotropic refinements -- right now, these are supported for quads in 2D and for some space-time extrusions of 2D quads
    vector<RefinementPatternPtr> anisotropicRefinements;
    map<RefinementPatternKey,RefinementPatternPtr> anisotropicRefinementsMap;
    anisotropicRefinements.push_back(RefinementPattern::xAnisotropicRefinementPatternQuad());
    anisotropicRefinements.push_back(RefinementPattern::yAnisotropicRefinementPatternQuad());
    anisotropicRefinements.push_back(RefinementPattern::xAnisotropicRefinementPatternQuadTimeExtruded());
    anisotropicRefinements.push_back(RefinementPattern::yAnisotropicRefinementPatternQuadTimeExtruded());
    for (RefinementPatternPtr refPattern : anisotropicRefinements)
    {
      if (anisotropicRefinementsMap.find(refPattern->getKey()) != anisotropicRefinementsMap.end())
      {
        out << "Key collision for anisotropic refinement pattern on cellTopo " << refPattern->parentTopology()->getName();
        out << " and cellTopo " << anisotropicRefinementsMap[refPattern->getKey()]->parentTopology()->getName() << endl;
        success = false;
      }
      if (noRefinementsMap.find(refPattern->getKey()) != noRefinementsMap.end())
      {
        out << "Key collision for anisotropic refinement pattern and null refinement pattern on cellTopo " << refPattern->parentTopology()->getName();
        out << " and cellTopo " << noRefinementsMap[refPattern->getKey()]->parentTopology()->getName() << endl;
        success = false;
      }
      if (regularRefinementsMap.find(refPattern->getKey()) != regularRefinementsMap.end())
      {
        out << "Key collision for anisotropic refinement pattern and regular refinement pattern for cellTopo " << refPattern->parentTopology()->getName();
        success = false;
      }
      anisotropicRefinementsMap[refPattern->getKey()] = refPattern;
    }
    TEST_EQUALITY(anisotropicRefinements.size(), anisotropicRefinementsMap.size());
    
    // finally, put all refinements into one big container:
    vector<RefinementPatternPtr> allRefinements;
    map<RefinementPatternKey,RefinementPatternPtr> allRefinementsMap;
    allRefinements.insert(allRefinements.end(), regularRefinements.begin(), regularRefinements.end());
    allRefinements.insert(allRefinements.end(), noRefinements.begin(), noRefinements.end());
    allRefinements.insert(allRefinements.end(), anisotropicRefinements.begin(), anisotropicRefinements.end());
    for (RefinementPatternPtr refPattern : allRefinements)
    {
      allRefinementsMap[refPattern->getKey()] = refPattern;
    }
    TEST_EQUALITY(allRefinements.size(), allRefinementsMap.size());
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, MapSubcellFromChildToParent_CommonVertices )
  {
    // test checks that regular refinement patterns do the right thing mapping from each child vertex to the parent,
    // in the case of vertices that are shared by parent.

    vector< CellTopoPtr > shardsTopologies = getShardsTopologies();

    for (int topoOrdinal = 0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
    {
      CellTopoPtr shardsTopo = shardsTopologies[topoOrdinal];
      if (shardsTopo->getDimension() == 0) continue; // skip the point topology
      RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(shardsTopo);

      Camellia::FieldContainer<double> childVertices = refPattern->refinedNodes();
      Camellia::FieldContainer<double> parentVertices(shardsTopo->getVertexCount(),shardsTopo->getDimension());
      CamelliaCellTools::refCellNodesForTopology(parentVertices, shardsTopo);

      double tol = 1e-15; // for vertex coordinate equality

      unsigned vertexDim = 0;
      for (int childOrdinal=0; childOrdinal<childVertices.dimension(0); childOrdinal++)
      {
        CellTopoPtr childTopo = refPattern->childTopology(childOrdinal);
        for (int childVertexOrdinal = 0; childVertexOrdinal < childTopo->getVertexCount(); childVertexOrdinal++)
        {
          int parentVertexOrdinal = -1;
          // see if this vertex is present in parent:
          for (int vertexOrdinal=0; vertexOrdinal<shardsTopo->getVertexCount(); vertexOrdinal++)
          {
            bool matches = true;
            for (int d=0; d < shardsTopo->getDimension(); d++)
            {
              if (abs(childVertices(childOrdinal,childVertexOrdinal,d)-parentVertices(vertexOrdinal,d)) > tol)
              {
                matches = false;
              }
            }
            if (matches)
            {
              parentVertexOrdinal = vertexOrdinal;
              break; // we've found our match
            }
          }
          if (parentVertexOrdinal == -1) continue; // no match in parent: we don't test this vertex
          pair<unsigned,unsigned> vertexAncestor = refPattern->mapSubcellFromChildToParent(childOrdinal, vertexDim, childVertexOrdinal);

          TEST_EQUALITY(vertexAncestor.first, vertexDim);
          TEST_EQUALITY(vertexAncestor.second, parentVertexOrdinal);
        }
      }
    }
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, MapSubcellFromChildToParent_LineMiddleVertex)
  {
    // for the case of the line refinement pattern, tests the vertex we don't test in MapSubcellFromChildToParent_CommonVertices

    CellTopoPtr shardsTopo = CellTopology::line();
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(shardsTopo);

    Camellia::FieldContainer<double> childVertices = refPattern->refinedNodes();
    Camellia::FieldContainer<double> parentVertices(shardsTopo->getVertexCount(),shardsTopo->getDimension());
    CamelliaCellTools::refCellNodesForTopology(parentVertices, shardsTopo);

    double tol = 1e-15; // for vertex coordinate equality

    unsigned vertexDim = 0, lineDim = 1;
    unsigned parentLineOrdinal = 0;
    for (int childOrdinal=0; childOrdinal<childVertices.dimension(0); childOrdinal++)
    {
      CellTopoPtr childTopo = refPattern->childTopology(childOrdinal);
      for (int childVertexOrdinal = 0; childVertexOrdinal < childTopo->getVertexCount(); childVertexOrdinal++)
      {
        int parentVertexOrdinal = -1;
        // see if this vertex is present in parent:
        for (int vertexOrdinal=0; vertexOrdinal<shardsTopo->getVertexCount(); vertexOrdinal++)
        {
          bool matches = true;
          for (int d=0; d < shardsTopo->getDimension(); d++)
          {
            if (abs(childVertices(childOrdinal,childVertexOrdinal,d)-parentVertices(vertexOrdinal,d)) > tol)
            {
              matches = false;
            }
          }
          if (matches)
          {
            parentVertexOrdinal = vertexOrdinal;
            break; // we've found our match
          }
        }
        if (parentVertexOrdinal != -1) continue; // found match in parent: this is a vertex we test above
        pair<unsigned,unsigned> vertexAncestor = refPattern->mapSubcellFromChildToParent(childOrdinal, vertexDim, childVertexOrdinal);

        TEST_EQUALITY(vertexAncestor.first, lineDim);
        TEST_EQUALITY(vertexAncestor.second, parentLineOrdinal);
      }
    }
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, MapRefCellPointsToAncestor_ChildNodeIdentity )
  {
    // test takes regular refinement patterns and checks that RefinementPattern::mapRefCellPointsToAncestor
    // does the right thing when we use the reference cell points on child cell in a one-level refinement branch

    vector< CellTopoPtr > cellTopos = { CellTopology::line(), CellTopology::quad(), CellTopology::triangle(), CellTopology::hexahedron() };

    for (int topoOrdinal=0; topoOrdinal<cellTopos.size(); topoOrdinal++)
    {
      CellTopoPtr cellTopo = cellTopos[topoOrdinal];

      int nodeCount = cellTopo->getNodeCount();
      int spaceDim = cellTopo->getDimension();

      Camellia::FieldContainer<double> refCellNodes(nodeCount,spaceDim);

      CamelliaCellTools::refCellNodesForTopology(refCellNodes, cellTopo);

      Camellia::FieldContainer<double> childRefCellNodes = refCellNodes;

      RefinementPatternPtr regularRefinement = RefinementPattern::regularRefinementPattern(cellTopo->getKey());

      int childCount = regularRefinement->numChildren();

      for (int childOrdinal=0; childOrdinal<childCount; childOrdinal++)
      {
        RefinementBranch refBranch;
        refBranch.push_back(make_pair(regularRefinement.get(),childOrdinal));

        Camellia::FieldContainer<double> expectedPoints(nodeCount,spaceDim);
        for (int nodeOrdinal=0; nodeOrdinal<nodeCount; nodeOrdinal++)
        {
          for (int d=0; d<spaceDim; d++)
          {
            expectedPoints(nodeOrdinal,d) = regularRefinement->refinedNodes()(childOrdinal,nodeOrdinal,d);
          }
        }
        Camellia::FieldContainer<double> actualPoints(nodeCount,spaceDim);
        RefinementPattern::mapRefCellPointsToAncestor(refBranch, childRefCellNodes, actualPoints);

        TEST_COMPARE_FLOATING_ARRAYS(expectedPoints, actualPoints, 1e-15);
      }
    }
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, MapRefCellPointsToAncestor_GeneralizedRefBranch_ChildNodeIdentity )
  {
    // test takes regular refinement patterns and checks that RefinementPattern::mapRefCellPointsToAncestor
    // does the right thing when we use the reference subcell nodes on child cell in a one-level generalized refinement branch

    vector< CellTopoPtr > cellTopos = { CellTopology::line(), CellTopology::quad(), CellTopology::triangle(), CellTopology::hexahedron() };

    for (int topoOrdinal=0; topoOrdinal<cellTopos.size(); topoOrdinal++)
    {
      CellTopoPtr cellTopo = cellTopos[topoOrdinal];

      int spaceDim = cellTopo->getDimension();

      RefinementPatternPtr regularRefinement = RefinementPattern::regularRefinementPattern(cellTopo->getKey());

      int childCount = regularRefinement->numChildren();

      for (int childOrdinal=0; childOrdinal<childCount; childOrdinal++)
      {
        RefinementBranch refBranch = {{regularRefinement.get(),childOrdinal}};
        CellTopoPtr childTopo = regularRefinement->childTopology(childOrdinal);
        for (int subcdim=0; subcdim<spaceDim; subcdim++)
        {
          int subcellCount = childTopo->getSubcellCount(subcdim);
          for (int subcord=0; subcord<subcellCount; subcord++)
          {
            GeneralizedRefinementBranch genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(refBranch, subcdim, subcord);
            CellTopoPtr subcellTopo = childTopo->getSubcell(subcdim, subcord);
            int nodeCount = subcellTopo->getNodeCount();
            Camellia::FieldContainer<double> refSubcellNodes(nodeCount,subcdim);
            CamelliaCellTools::refCellNodesForTopology(refSubcellNodes, subcellTopo);
            Camellia::FieldContainer<double> expectedPoints(nodeCount,spaceDim);
            for (int subcellNode=0; subcellNode<nodeCount; subcellNode++)
            {
              int childNode = childTopo->getNodeMap(subcdim, subcord, subcellNode);
              for (int d=0; d<spaceDim; d++)
              {
                expectedPoints(subcellNode,d) = regularRefinement->refinedNodes()(childOrdinal,childNode,d);
              }
            }
            Camellia::FieldContainer<double> actualPoints(nodeCount,spaceDim);
            RefinementPattern::mapRefCellPointsToAncestor(genRefBranch, refSubcellNodes, actualPoints);
            bool oldSuccess = success;
            success = true;
            TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(expectedPoints, actualPoints, 1e-15);
            if (success==false)
            {
              // local failure
              out << "failure for cellTopo " << cellTopo->getName() << ", child " << childOrdinal;
              out << ", " << CamelliaCellTools::entityTypeString(subcdim) << " ordinal " << subcord << endl;
              out << "expectedPoints:\n" << expectedPoints;
              out << "actualPoints:\n" << actualPoints;

              // rerun for debugging:
              RefinementPattern::mapRefCellPointsToAncestor(genRefBranch, refSubcellNodes, actualPoints);
            }
            success = success && oldSuccess;
          }
        }
      }
    }
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, MapRefCellPointsToAncestor_LineChild )
  {
    RefinementPatternPtr lineRefinement = RefinementPattern::regularRefinementPatternLine();
    // create a one-refinement branch, selecting the right child (child 1):
    RefinementBranch refBranch = { {lineRefinement.get(),1} }; // child goes from 0 to 1 in parent

    int numPoints = 2;
    int spaceDim = 1;
    Camellia::FieldContainer<double> fineLinePoints(numPoints,spaceDim);
    fineLinePoints(0,0) = -1.0;
    fineLinePoints(1,0) =  1.0;
    
    Camellia::FieldContainer<double> expectedCoarsePoints(numPoints,spaceDim);
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      expectedCoarsePoints(pointOrdinal,0) = fineLinePoints(pointOrdinal,0) * 0.5 + 0.5;
    }
    
    Camellia::FieldContainer<double> actualPoints(numPoints,spaceDim);
    RefinementPattern::mapRefCellPointsToAncestor(refBranch, fineLinePoints, actualPoints);
    
    double tol = 1e-15;
    TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(expectedCoarsePoints,actualPoints,tol);
  }
  
  TEUCHOS_UNIT_TEST( RefinementPattern, MapRefCellPointsToAncestor_InteriorTriangleEdge )
  {
    // The genRefBranch here is identical to the one we use in GeneralizedRefinementBranchForLeafSubcell_InteriorTriangleEdge, above
    RefinementPatternPtr triangleRefinement = RefinementPattern::regularRefinementPatternTriangle();
    // create a one-refinement branch, selecting middle triangle (child 1):
    RefinementBranch refBranch = { {triangleRefinement.get(),1} };

    /* taking the 0 edge as our subcell on the leaf cell, we expect GeneralizedRefinementBranch like so.
     From leaf to root:
     1.  previousTierTopo = Triangle;
     rootDimension = 2;
     previousTierSubcellOrdinal = 0
     refBranch = {{triangleRefinement,1}};
     leafSubcellDim = 1;
     leafSubcellOrdinal = 0
     */
    unsigned edgeDim = 1;
    unsigned edgeOrdinalInCell = 0;
    GeneralizedRefinementBranch genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(refBranch,
                                                                                                            edgeDim,
                                                                                                            edgeOrdinalInCell);

    int numPoints = 2;
    int fineDim  = 1;
    Camellia::FieldContainer<double> leafSubcellPoints(numPoints, fineDim);
    leafSubcellPoints(0,0) = -0.25;
    leafSubcellPoints(1,0) = 0.5;

    /* The 0 edge in the triangle's child 1 (the interior triangle) extends from (0.5,0) to (0.5,0.5) in parent */

    int volumeDim = 2;
    Camellia::FieldContainer<double> expectedPointsInParent(numPoints,volumeDim);
    expectedPointsInParent(0,0) = 0.5;
    expectedPointsInParent(0,1) = 0.25 * leafSubcellPoints(0,0) + 0.25;
    expectedPointsInParent(1,0) = 0.5;
    expectedPointsInParent(1,1) = 0.25 * leafSubcellPoints(1,0) + 0.25;

    Camellia::FieldContainer<double> actualPoints(numPoints,volumeDim);
    RefinementPattern::mapRefCellPointsToAncestor(genRefBranch, leafSubcellPoints, actualPoints);

    double tol = 1e-15;
    TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(expectedPointsInParent,actualPoints,tol);
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, MapRefCellPointsToAncestor_HexahedronHangingEdge )
  {
    // A test roughly corresponding to a case that's failing in GDAMinimumRule.  That issue likely has to do with
    // the permutation of the edge as seen by the side and the edge as seen by the cell; here, we only have
    // the edge seen by the cell.  I.e., I do expect this to pass.

    RefinementPatternPtr hexahedronRefinement = RefinementPattern::regularRefinementPatternHexahedron();
    // create a one-refinement branch, selecting child 1:
    RefinementBranch refBranch = { {hexahedronRefinement.get(),1} };

    /* taking the 10 edge as our subcell on the leaf cell, we expect GeneralizedRefinementBranch like so.
     From leaf to root:
     1.  previousTierTopo = Hexahedron;
     rootDimension = 2;
     previousTierSubcellOrdinal = 1;
     refBranch = {{quadRefinement,1}};
     leafSubcellDim = 1;
     leafSubcellOrdinal = 3
     */
    unsigned edgeDim = 1;
    unsigned edgeOrdinalInCell = 10;
    GeneralizedRefinementBranch genRefBranch = RefinementPattern::generalizedRefinementBranchForLeafSubcell(refBranch,
                                                                                                            edgeDim,
                                                                                                            edgeOrdinalInCell);

    int numPoints = 3;
    int fineDim  = 1;
    Camellia::FieldContainer<double> leafSubcellPoints(numPoints, fineDim);
    leafSubcellPoints(0,0) = -1.0;
    leafSubcellPoints(1,0) = 0.0;
    leafSubcellPoints(2,0) = 1.0;

    /* The 10 edge in the hexahedron child 1 extends from vertex 2 to 6 in the child, which is (1,0,-1) to (1,0,0) in parent */

    int volumeDim = 3;
    Camellia::FieldContainer<double> expectedPointsInParent(numPoints,volumeDim);
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      expectedPointsInParent(pointOrdinal,0) = 1.0;
      expectedPointsInParent(pointOrdinal,1) = 0.0;
      expectedPointsInParent(pointOrdinal,2) = 0.5 * (leafSubcellPoints(pointOrdinal,0) - 1.0);
    }

    Camellia::FieldContainer<double> actualPoints(numPoints,volumeDim);
    RefinementPattern::mapRefCellPointsToAncestor(genRefBranch, leafSubcellPoints, actualPoints);

    double tol = 1e-15;
    TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(expectedPointsInParent,actualPoints,tol);
  }
  
  TEUCHOS_UNIT_TEST( RefinementPattern, NullRefinementKey )
  {
    // the refinement ordinal of a null refinement for a d-dimensional cell topology
    // should be 2^d - 1.
    
    vector<CellTopoPtr> shardsTopologies = getShardsTopologies();
    vector<CellTopoPtr> cellTopologies = shardsTopologies;
    for (CellTopoPtr shardsTopo : shardsTopologies)
    {
      // let's also include regular refinements of cell topologies of tensorial degree 1
      int tensorialDegree = 1;
      CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(shardsTopo, tensorialDegree);
      cellTopologies.push_back(spaceTimeTopo);
    }
    
    vector<RefinementPatternPtr> noRefinements;
    map<RefinementPatternKey,RefinementPatternPtr> noRefinementsMap;
    for (CellTopoPtr cellTopo : cellTopologies)
    {
      RefinementPatternPtr refPattern = RefinementPattern::noRefinementPattern(cellTopo);
      RefinementPatternKey key = refPattern->getKey();
      unsigned refinementOrdinal = key.second;
      unsigned expectedOrdinal = (1 << cellTopo->getDimension()) - 1;
      TEST_EQUALITY(refinementOrdinal, expectedOrdinal);
      
      CellTopologyKey cellTopoKey = refPattern->getKey().first;
      CellTopologyKey expectedKey = cellTopo->getKey();
      TEST_EQUALITY(cellTopoKey, expectedKey);
    }
  }

  TEUCHOS_UNIT_TEST( RefinementPattern, SpaceTimeTopology )
  {
    // tests refinement patterns for space-time topologies.
    int tensorialDegree = 1;
    vector<CellTopoPtr> shardsTopologies = getShardsTopologies();
    for (int topoOrdinal=0; topoOrdinal < shardsTopologies.size(); topoOrdinal++)
    {
      shards::CellTopology shardsTopo = shardsTopologies[topoOrdinal]->getShardsTopology();
      CellTopoPtr cellTopo = CellTopology::cellTopology(shardsTopo, tensorialDegree);

      Camellia::FieldContainer<double> spaceRefinedNodes;
      RefinementPatternPtr refPatternSpace = RefinementPattern::regularRefinementPattern(shardsTopo.getKey());
      spaceRefinedNodes = refPatternSpace->refinedNodes();

      RefinementPatternPtr refPatternTime = RefinementPattern::regularRefinementPattern(CellTopology::line());
      Camellia::FieldContainer<double> timeRefinedNodes = refPatternTime->refinedNodes();

      RefinementPatternPtr refPatternSpaceTime = RefinementPattern::regularRefinementPattern(cellTopo);
      Camellia::FieldContainer<double> spaceTimeRefinedNodes = refPatternSpaceTime->refinedNodes();

      int numChildrenSpace = spaceRefinedNodes.dimension(0);
      int numChildrenTime = timeRefinedNodes.dimension(0);
      int numChildrenSpaceTime = spaceTimeRefinedNodes.dimension(0);

      TEST_EQUALITY(numChildrenSpace * numChildrenTime, numChildrenSpaceTime);

      int numNodesSpace = spaceRefinedNodes.dimension(1);
      int numNodesTime = timeRefinedNodes.dimension(1);
      int numNodesSpaceTime = spaceTimeRefinedNodes.dimension(1);

      TEST_EQUALITY(numNodesSpace * numNodesTime, numNodesSpaceTime);

      int dSpace = spaceRefinedNodes.dimension(2);
      int dTime = timeRefinedNodes.dimension(2);
      int dSpaceTime = spaceTimeRefinedNodes.dimension(2);

      TEST_EQUALITY(dSpace + dTime, dSpaceTime);

      Camellia::FieldContainer<double> expectedRefinedNodes(numChildrenSpaceTime,numNodesSpaceTime,dSpaceTime);

      int spaceTimeChildOrdinal = 0;
      for (int timeChildOrdinal=0; timeChildOrdinal<numChildrenTime; timeChildOrdinal++)
      {
        for (int spaceChildOrdinal=0; spaceChildOrdinal<numChildrenSpace; spaceChildOrdinal++, spaceTimeChildOrdinal++)
        {
          int spaceTimeNodeOrdinal=0;
          for (int timeNodeOrdinal=0; timeNodeOrdinal<numNodesTime; timeNodeOrdinal++)
          {
            for (int spaceNodeOrdinal=0; spaceNodeOrdinal<numNodesSpace; spaceNodeOrdinal++, spaceTimeNodeOrdinal++)
            {
              for (int d_space=0; d_space<dSpace; d_space++)
              {
                expectedRefinedNodes(spaceTimeChildOrdinal,spaceTimeNodeOrdinal,d_space) = spaceRefinedNodes(spaceChildOrdinal,spaceNodeOrdinal,d_space);
              }
              for (int d_time=0; d_time<dTime; d_time++)
              {
                expectedRefinedNodes(spaceTimeChildOrdinal,spaceTimeNodeOrdinal,d_time + dSpace) = timeRefinedNodes(timeChildOrdinal,timeNodeOrdinal,d_time);
              }
            }
          }
        }
      }
      TEST_COMPARE_FLOATING_ARRAYS(spaceTimeRefinedNodes, expectedRefinedNodes, 1e-15);
    }
  }
} // namespace
