//
//  MeshGeometry.h
//  Camellia
//
//  Created by Nate Roberts on 1/3/14.
//
//

#ifndef Camellia_MeshGeometry_h
#define Camellia_MeshGeometry_h

#include "ParametricCurve.h"

#include "IndexType.h"

typedef pair<IndexType, IndexType> Edge;
typedef Teuchos::RCP< shards::CellTopology > CellTopoPtrLegacy;

class MeshGeometry {
  vector< vector<double> > _vertices;
  vector< vector<IndexType> > _elementVertices;
  map< Edge, ParametricCurvePtr > _edgeToCurveMap;
  vector< CellTopoPtrLegacy > _cellTopos;
  
  void initializeVerticesFromFieldContainer(const vector<FieldContainer<double> > &vertices) {
    IndexType numVertices = vertices.size();
    if (numVertices == 0) return;
    int spaceDim = vertices[0].size();
    vector<double> vertex(spaceDim);
    _vertices.clear();
    for (IndexType i=0; i<numVertices; i++) {
      for (int d=0; d<spaceDim; d++) {
        vertex[d] = vertices[i](d);
      }
      _vertices.push_back(vertex);
    }
  }
  
  void populateCellToposFromElementVertices() {
    // guesses the cell topology based on the number of vertices.
    CellTopoPtrLegacy line = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Line<2> >() ) );
    CellTopoPtrLegacy quad = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
    CellTopoPtrLegacy triangle = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Triangle<3> >() ) );
    CellTopoPtrLegacy hex = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ) );
    IndexType numCells = _elementVertices.size();
    _cellTopos.clear();
    for (IndexType cellIndex=0; cellIndex<numCells; cellIndex++) {
      IndexType numVertices = _elementVertices[cellIndex].size();
      CellTopoPtrLegacy cellTopo;
      switch (numVertices) {
        case 2:
          cellTopo = line;
          break;
        case 3:
          cellTopo = triangle;
          break;
        case 4:
          cellTopo = quad;
          break;
        case 8:
          cellTopo = hex;
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled number of vertices.  Must use the constructor that includes CellTopology assignment for each cell.");
      }
      _cellTopos.push_back(cellTopo);
    }
  }
public:
  MeshGeometry(const vector< vector<double> > &vertices,
                  const vector< vector<IndexType> > &elementVertices,
                  const vector< CellTopoPtrLegacy > &cellTopos,
                  const map< Edge, ParametricCurvePtr > &edgeToCurveMap) {
    _vertices = vertices;
    _elementVertices = elementVertices;
    _cellTopos = cellTopos;
    _edgeToCurveMap = edgeToCurveMap;
  }
  
  MeshGeometry(const vector< vector<double> > &vertices,
                  const vector< vector<IndexType> > &elementVertices,
                  const vector< CellTopoPtrLegacy > &cellTopos) {
    _vertices = vertices;
    _elementVertices = elementVertices;
    _cellTopos = cellTopos;
  }
  
  // deprecated method; included for compatibility with earlier version of MeshGeometry
  MeshGeometry(const vector< vector<double> > &vertices,
               const vector< vector<IndexType> > &elementVertices,
               const map< Edge, ParametricCurvePtr > &edgeToCurveMap) {
//    initializeVerticesFromFieldContainer(vertices);
    _vertices = vertices;
    _elementVertices = elementVertices;
    _edgeToCurveMap = edgeToCurveMap;
    populateCellToposFromElementVertices();
  }

  // deprecated method; included for compatibility with earlier version of MeshGeometry
  MeshGeometry(const vector< vector<double> > &vertices,
               const vector< vector<IndexType> > &elementVertices) {
//    initializeVerticesFromFieldContainer(vertices);
    _vertices = vertices;
    _elementVertices = elementVertices;
    populateCellToposFromElementVertices();
  }
  
  map< Edge, ParametricCurvePtr > &edgeToCurveMap() {
    return _edgeToCurveMap;
  }
  
  vector< vector<IndexType> > &elementVertices() {
    return _elementVertices;
  }
  
  vector< vector<double> > &vertices() {
    return _vertices;
  }
  
  const vector< CellTopoPtrLegacy > &cellTopos() {
    return _cellTopos;
  }
};

typedef Teuchos::RCP<MeshGeometry> MeshGeometryPtr;

#endif