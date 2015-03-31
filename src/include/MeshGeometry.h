//
//  MeshGeometry.h
//  Camellia
//
//  Created by Nate Roberts on 1/3/14.
//
//

#ifndef Camellia_MeshGeometry_h
#define Camellia_MeshGeometry_h

#include "TypeDefs.h"

#include "ParametricCurve.h"

#include "CellTopology.h"

typedef pair<IndexType, IndexType> Edge;
typedef Teuchos::RCP< shards::CellTopology > CellTopoPtrLegacy;

class MeshGeometry {
  vector< vector<double> > _vertices;
  vector< vector<IndexType> > _elementVertices;
  map< Edge, ParametricCurvePtr > _edgeToCurveMap;
  vector< CellTopoPtr > _cellTopos;
  
  void initializeVerticesFromFieldContainer(const vector<Intrepid::FieldContainer<double> > &vertices) {
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
    IndexType numCells = _elementVertices.size();
    _cellTopos.clear();
    for (IndexType cellIndex=0; cellIndex<numCells; cellIndex++) {
      IndexType numVertices = _elementVertices[cellIndex].size();
      CellTopoPtr cellTopo;
      switch (numVertices) {
        case 2:
          cellTopo = Camellia::CellTopology::line();
          break;
        case 3:
          cellTopo = Camellia::CellTopology::triangle();
          break;
        case 4:
          cellTopo = Camellia::CellTopology::quad();
          break;
        case 8:
          cellTopo = Camellia::CellTopology::hexahedron();
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
                  const vector< CellTopoPtr > &cellTopos,
                  const map< Edge, ParametricCurvePtr > &edgeToCurveMap) {
    _vertices = vertices;
    _elementVertices = elementVertices;
    _cellTopos = cellTopos;
    _edgeToCurveMap = edgeToCurveMap;
  }
  
  MeshGeometry(const vector< vector<double> > &vertices,
                  const vector< vector<IndexType> > &elementVertices,
                  const vector< CellTopoPtr > &cellTopos) {
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
  
  const vector< CellTopoPtr > &cellTopos() {
    return _cellTopos;
  }
};

typedef Teuchos::RCP<MeshGeometry> MeshGeometryPtr;

#endif