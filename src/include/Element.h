#ifndef DPG_ELEMENT
#define DPG_ELEMENT

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

/*
 *  Element.h
 *
 */

#include "TypeDefs.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"
#include "ElementType.h"
#include "RefinementPattern.h"

#include "Cell.h"

using namespace std;

namespace Camellia
{
class Element
{
private:
  CellPtr _cell;
  Mesh* _mesh;

  Teuchos::RCP< ElementType > _elemTypePtr;
  IndexType _cellIndex; // index into the Mesh's Elements vector for ElementType for a given partition
  GlobalIndexType _globalCellIndex; // index into a vector of *all* elements of a given type across partitions...

  bool _deleted;
public:
  //constructor:
  Element(Mesh* mesh, GlobalIndexType cellID, Teuchos::RCP< ElementType > elemType, IndexType cellIndex, GlobalIndexType globalCellIndex=-1);

  Teuchos::RCP< ElementType > elementType()
  {
    return _elemTypePtr;
  }
  void setElementType( Teuchos::RCP< ElementType > newElementType)
  {
    _elemTypePtr = newElementType;
  }

  int cellID()
  {
    return _cell->cellIndex();
  }
  int cellIndex()
  {
    return _cellIndex;
  }
  void setCellIndex(int newCellIndex)
  {
    _cellIndex = newCellIndex;
  }
  int globalCellIndex()
  {
    return _globalCellIndex;
  }
  void setGlobalCellIndex(int globalCellIndex)
  {
    _globalCellIndex = globalCellIndex;
  }
  int numSides();
  void getSidePointsInNeighborRefCoords(Camellia::FieldContainer<double> &neighborRefPoints, int sideIndex,
                                        const Camellia::FieldContainer<double> &refPoints);
  void getSidePointsInParentRefCoords(Camellia::FieldContainer<double> &parentRefPoints, int sideIndex,
                                      const Camellia::FieldContainer<double> &childRefPoints);
  ElementPtr getNeighbor(int & mySideIndexInNeighbor, int neighborsSideIndexInMe);
  int getNeighborCellID(int sideIndex);
  int getSideIndexInNeighbor(int sideIndex);

  int indexInParentSide(int parentSide);

  void setNeighbor(int neighborsSideIndexInMe, Teuchos::RCP< Element > elemPtr, int mySideIndexInNeighbor);
  //int subSideIndexInNeighbor(int neighborsSideIndexInMe);

  void setRefinementPattern(Teuchos::RCP<RefinementPattern> &refPattern);
  vector< pair<unsigned, unsigned> > & childIndicesForSide(unsigned sideIndex); // pair( child index, sideIndex in child of the side shared with parent)

  ElementPtr getParent();

  int parentSideForSideIndex(int mySideIndex);
  int numChildren();
  ElementPtr getChild(int childIndex);
  bool isActive();
  bool isParent();
  bool isChild();
  vector< pair<int,int> > getDescendantsForSide(int sideIndex, bool leafNodesOnly = true);

  set<int> getDescendants(bool leafNodesOnly = true);

  void deleteChildrenFromMesh(set< pair<int,int> > &affectedNeighborSides, set<int> &deletedElements);
  void deleteFromMesh(set< pair<int,int> > &affectedNeighborSides, set<int> &deletedElements);
  //destructor:
  ~Element();
};
}

#endif
