//
//  MeshTopologyView.h
//  Camellia
//
//  Created by Nate Roberts on 6/23/15.
//
//

//! MeshTopologyView: a class that defines a minimal interface for MeshTopology objects used by Mesh, GlobalDofAssignment, and
//! subclasses of GlobalDofAssignment.

/*!
 \author Nathan V. Roberts, ALCF.
 
 \date Last modified on 23-June-2015.
 */


#ifndef Camellia_MeshTopologyView_h
#define Camellia_MeshTopologyView_h

#include "TypeDefs.h"

namespace Camellia {

  class MeshTopologyView
  {
    MeshTopologyPtr _meshTopo; // null when subclass constructor is used
    std::set<IndexType> _activeCellIDs; // empty when subclass constructor is used
  protected:
    std::set<IndexType> _rootCellIndices; // lazily filled when meshTopoPtr is not null; otherwise responsibility belongs to subclass.
    GlobalDofAssignment* _gda; // for cubature degree lookups
    
    std::vector<IndexType> getActiveCellsForSide(IndexType sideEntityIndex);
  public:
    // ! Constructor for use by MeshTopology and any other subclasses
    MeshTopologyView();
    
    // ! Constructor that defines a view in terms of an existing MeshTopology and a set of cells selected to be active.
    MeshTopologyView(MeshTopologyPtr meshTopoPtr, const std::set<IndexType> &activeCellIDs);
    
    // ! This method only gets within a factor of 2 or so, but can give a rough estimate
    virtual long long approximateMemoryFootprint();
    
    virtual std::vector<IndexType> cellIDsForPoints(const Intrepid::FieldContainer<double> &physicalPoints);
    
    // ! creates a copy of this, deep-copying each Cell and all lookup tables (but does not deep copy any other objects, e.g. PeriodicBCPtrs).  Not supported for MeshTopologyViews with _meshTopo defined (i.e. those that are themselves defined in terms of another MeshTopology object).
    virtual Teuchos::RCP<MeshTopology> deepCopy();

    virtual bool entityIsAncestor(unsigned d, IndexType ancestor, IndexType descendent);
    
    virtual const std::set<IndexType> &getActiveCellIndices();
    virtual std::vector< std::pair<IndexType,unsigned> > getActiveCellIndices(unsigned d, IndexType entityIndex); // first entry in pair is the cellIndex, the second is the index of the entity in that cell (the subcord).
    
    virtual CellPtr getCell(IndexType cellIndex);
    virtual std::set< std::pair<IndexType, unsigned> > getCellsContainingEntity(unsigned d, unsigned entityIndex);
    virtual std::vector<IndexType> getCellsForSide(IndexType sideEntityIndex);
    
    virtual std::pair<IndexType, unsigned> getConstrainingEntity(unsigned d, IndexType entityIndex);
    virtual IndexType getConstrainingEntityIndexOfLikeDimension(unsigned d, IndexType entityIndex);
    virtual std::vector< std::pair<IndexType,unsigned> > getConstrainingSideAncestry(unsigned int sideEntityIndex);
    
    virtual unsigned getDimension();
    
    virtual std::vector<IndexType> getEntityVertexIndices(unsigned d, IndexType entityIndex);
    
    virtual IndexType getMaximumCellIndex();
    
    virtual const std::set<IndexType> &getRootCellIndices();
    
    virtual std::vector< IndexType > getSidesContainingEntity(unsigned d, IndexType entityIndex);
    
    virtual const std::vector<double>& getVertex(IndexType vertexIndex);
    
    virtual bool getVertexIndex(const std::vector<double> &vertex, IndexType &vertexIndex, double tol=1e-14);
    
    virtual std::pair<IndexType,IndexType> owningCellIndexForConstrainingEntity(unsigned d, unsigned constrainingEntityIndex);
    
    virtual void setGlobalDofAssignment(GlobalDofAssignment* gda); // for cubature degree lookups
  };

}
#endif
