//
//  MeshFactory.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/21/13.
//
//

#include "MeshFactory.h"

#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "GlobalDofAssignment.h"
#include "GnuPlotUtil.h"
#include "MOABReader.h"
#include "ParametricCurve.h"
#include "RefinementHistory.h"

#ifdef HAVE_EPETRAEXT_HDF5
#include <EpetraExt_HDF5.h>
#include <Epetra_SerialComm.h>
#endif

using namespace Intrepid;
using namespace Camellia;

static ParametricCurvePtr parametricRect(double width, double height, double x0, double y0)
{
  // starts at the positive x axis and proceeds counter-clockwise, just like our parametric circle
  vector< pair<double, double> > vertices;
  vertices.push_back(make_pair(x0 + width/2.0, y0 + 0));
  vertices.push_back(make_pair(x0 + width/2.0, y0 + height/2.0));
  vertices.push_back(make_pair(x0 - width/2.0, y0 + height/2.0));
  vertices.push_back(make_pair(x0 - width/2.0, y0 - height/2.0));
  vertices.push_back(make_pair(x0 + width/2.0, y0 - height/2.0));
  return ParametricCurve::polygon(vertices);
}

map<int,int> MeshFactory::_emptyIntIntMap;

#ifdef HAVE_EPETRAEXT_HDF5
MeshPtr MeshFactory::loadFromHDF5(TBFPtr<double> bf, string filename)
{
  // 8-13-15 modified to use MPI communicator instead of serial.  This should give
  // HDF5 a chance to do smart things with I/O and the network (e.g. read in on rank 0,
  // and broadcast to all).  (Though the fact that it has a *chance* to do smart things
  // doesn't mean it will.)
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  EpetraExt::HDF5 hdf5(Comm);
  hdf5.Open(filename);
  int vertexIndicesSize, topoKeysSize, verticesSize, trialOrderEnhancementsSize, testOrderEnhancementsSize, histArraySize, H1OrderSize;
  hdf5.Read("Mesh", "vertexIndicesSize", vertexIndicesSize);
  hdf5.Read("Mesh", "topoKeysSize", topoKeysSize);
  hdf5.Read("Mesh", "verticesSize", verticesSize);
  hdf5.Read("Mesh", "trialOrderEnhancementsSize", trialOrderEnhancementsSize);
  hdf5.Read("Mesh", "testOrderEnhancementsSize", testOrderEnhancementsSize);
  hdf5.Read("Mesh", "histArraySize", histArraySize);
  hdf5.Read("Mesh", "H1OrderSize", H1OrderSize);

  int topoKeysIntSize = topoKeysSize * sizeof(Camellia::CellTopologyKey) / sizeof(int);

  int numPartitions, maxPartitionSize;

  hdf5.Read("Mesh", "numPartitions", numPartitions);
  hdf5.Read("Mesh", "maxPartitionSize", maxPartitionSize);
  FieldContainer<int> partitionsCastToInt(numPartitions,maxPartitionSize);
  FieldContainer<GlobalIndexType> partitions;
  if (numPartitions > 0)
  {
    hdf5.Read("Mesh", "partitions", H5T_NATIVE_INT, partitionsCastToInt.size(), &partitionsCastToInt[0]);
    partitions.resize(numPartitions,maxPartitionSize);
    for (int i=0; i<numPartitions; i++)
    {
      for (int j=0; j<maxPartitionSize; j++)
      {
        partitions(i,j) = (GlobalIndexType) partitionsCastToInt(i,j);
      }
    }
  }
  else
  {
  }

  int dimension, deltaP;
  vector<int> vertexIndices(vertexIndicesSize);
  vector<Camellia::CellTopologyKey> topoKeys(topoKeysSize);
  vector<int> trialOrderEnhancementsVec(trialOrderEnhancementsSize);
  vector<int> testOrderEnhancementsVec(testOrderEnhancementsSize);
  vector<double> vertices(verticesSize);
  vector<int> histArray(histArraySize);
  vector<int> H1Order(H1OrderSize);
  string GDARule;
  hdf5.Read("Mesh", "dimension", dimension);
  hdf5.Read("Mesh", "vertexIndices", H5T_NATIVE_INT, vertexIndicesSize, &vertexIndices[0]);
  hdf5.Read("Mesh", "topoKeys", H5T_NATIVE_INT, topoKeysIntSize, &topoKeys[0]);
  hdf5.Read("Mesh", "vertices", H5T_NATIVE_DOUBLE, verticesSize, &vertices[0]);
  hdf5.Read("Mesh", "deltaP", deltaP);
  hdf5.Read("Mesh", "GDARule", GDARule);
  if (GDARule == "min")
  {
  }
  else if(GDARule == "max")
  {
  }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid GDA");
  hdf5.Read("Mesh", "trialOrderEnhancements", H5T_NATIVE_INT, trialOrderEnhancementsSize, &trialOrderEnhancementsVec[0]);
  hdf5.Read("Mesh", "testOrderEnhancements", H5T_NATIVE_INT, testOrderEnhancementsSize, &testOrderEnhancementsVec[0]);
  hdf5.Read("Mesh", "H1Order", H5T_NATIVE_INT, H1OrderSize, &H1Order[0]);

  if (histArraySize > 0) hdf5.Read("Mesh", "refinementHistory", H5T_NATIVE_INT, histArraySize, &histArray[0]);
  hdf5.Close();

  CellTopoPtr line_2 = Camellia::CellTopology::line();
  CellTopoPtr quad_4 = Camellia::CellTopology::quad();
  CellTopoPtr tri_3 = Camellia::CellTopology::triangle();
  CellTopoPtr hex_8 = Camellia::CellTopology::hexahedron();

  vector< CellTopoPtr > cellTopos;
  vector< vector<unsigned> > elementVertices;
  int vindx = 0;
  for (unsigned cellNumber = 0; cellNumber < topoKeysSize; cellNumber++)
  {
    CellTopoPtr cellTopo = CamelliaCellTools::cellTopoForKey(topoKeys[cellNumber]);
    cellTopos.push_back(cellTopo);
    vector<unsigned> elemVertices;
    for (int i=0; i < cellTopo->getVertexCount(); i++)
    {
      elemVertices.push_back(vertexIndices[vindx]);
      vindx++;
    }
    elementVertices.push_back(elemVertices);
  }

//    cout << "Elements:\n";
//    for (int i=0; i<elementVertices.size(); i++) {
//      cout << "Element " << i << ":\n";
//      Camellia::print("vertex indices", elementVertices[i]);
//    }

  vector< vector<double> > verticesList;
  for (int i=0; i < vertices.size()/dimension; i++)
  {
    vector<double> vertex;
    for (int d=0; d<dimension; d++)
    {
      vertex.push_back(vertices[dimension*i+d]);
    }
    verticesList.push_back(vertex);
  }

//    cout << "Vertices:\n";
//    for (int i=0; i<verticesList.size(); i++) {
//      cout << "Vertex " << i << ":\n";
//      Camellia::print("vertex coordinates", verticesList[i]);
//    }

  map<int, int> trialOrderEnhancements;
  map<int, int> testOrderEnhancements;
  for (int i=0; i < trialOrderEnhancementsVec.size()/2; i++) // divide by two because we have 2 entries per var; map goes varID --> enhancement
  {
    trialOrderEnhancements[trialOrderEnhancementsVec[2*i]] = trialOrderEnhancementsVec[2*i+1];
  }
  for (int i=0; i < testOrderEnhancementsVec.size()/2; i++)
  {
    testOrderEnhancements[testOrderEnhancementsVec[2*i]] = testOrderEnhancementsVec[2*i+1];
  }

  MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(verticesList, elementVertices, cellTopos) );
  MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry) );
  MeshPtr mesh = Teuchos::rcp( new Mesh (meshTopology, bf, H1Order, deltaP, trialOrderEnhancements, testOrderEnhancements) );

  for (int i=0; i < histArraySize;)
  {
    RefinementType refType = RefinementType(histArray[i]);
    i++;
    int numCells = histArray[i];
    i++;
    CellTopoPtr cellTopo; // we assume all cells for the refinement have the same type
    if (numCells > 0)
    {
      GlobalIndexType firstCellID = histArray[i];
      cellTopo = mesh->getElementType(firstCellID)->cellTopoPtr;
    }
    set<GlobalIndexType> cellIDs;
    for (int c=0; c < numCells; c++)
    {
      GlobalIndexType cellID = histArray[i];
      i++;
      cellIDs.insert(cellID);
      // check that the cellIDs are all active nodes
      if (refType != H_UNREFINEMENT)
      {
        set<GlobalIndexType> activeIDs = mesh->getActiveCellIDs();
        if (activeIDs.find(cellID) == activeIDs.end())
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID for refinement is not an active cell of the mesh");
        }
      }
    }
    bool repartitionAndRebuild = false; // we'll do this at the end: if partitions were set, we'll use those.  Otherwise, we'll just do a standard repartition

    switch (refType)
    {
    case P_REFINEMENT:
      mesh->pRefine(cellIDs);
      break;
    case H_UNREFINEMENT:
      mesh->hUnrefine(cellIDs);
      break;
    default:
      // if we get here, it should be an h-refinement with a ref pattern
      mesh->hRefine(cellIDs, RefinementHistory::refPatternForRefType(refType, cellTopo), repartitionAndRebuild);
    }
  }
  if (numPartitions > 0)
  {
    mesh->globalDofAssignment()->setPartitions(partitions);
  }
  else
  {
    mesh->globalDofAssignment()->repartitionAndMigrate(); // since no Solution registered, won't actually migrate anything
  }
  return mesh;
}
#endif

MeshPtr MeshFactory::quadMesh(Teuchos::ParameterList &parameters)
{
  bool useMinRule = parameters.get<bool>("useMinRule",true);
  TBFPtr<double> bf = parameters.get< TBFPtr<double> >("bf");
  int H1Order = parameters.get<int>("H1Order");
  int spaceDim = 2;
  int delta_k = parameters.get<int>("delta_k",spaceDim);
  double width = parameters.get<double>("width",1.0);
  double height = parameters.get<double>("height",1.0);
  int horizontalElements = parameters.get<int>("horizontalElements", 1);
  int verticalElements = parameters.get<int>("verticalElements", 1);
  bool divideIntoTriangles = parameters.get<bool>("divideIntoTriangles",false);
  double x0 = parameters.get<double>("x0",0.0);
  double y0 = parameters.get<double>("y0",0.0);
  map<int,int> emptyMap;
  map<int,int>* trialOrderEnhancements = parameters.get< map<int,int>* >("trialOrderEnhancements",&emptyMap);
  map<int,int>* testOrderEnhancements = parameters.get< map<int,int>* >("testOrderEnhancements",&emptyMap);
  vector< PeriodicBCPtr > emptyPeriodicBCs;
  vector< PeriodicBCPtr >* periodicBCs = parameters.get< vector< PeriodicBCPtr >* >("periodicBCs",&emptyPeriodicBCs);

  if (useMinRule)
  {
    MeshTopologyPtr meshTopology = quadMeshTopology(width,height,horizontalElements,verticalElements,divideIntoTriangles,x0,y0,*periodicBCs);
    return Teuchos::rcp( new Mesh(meshTopology, bf, H1Order, delta_k, *trialOrderEnhancements, *testOrderEnhancements) );
  }
  else
  {
    bool useConformingTraces = parameters.get<bool>("useConformingTraces", true);
//  cout << "periodicBCs size is " << periodicBCs->size() << endl;
    vector<vector<double> > vertices;
    vector< vector<unsigned> > allElementVertices;

    int numElements = divideIntoTriangles ? horizontalElements * verticalElements * 2 : horizontalElements * verticalElements;

    CellTopoPtr topo;
    if (divideIntoTriangles)
    {
      topo = Camellia::CellTopology::triangle();
    }
    else
    {
      topo = Camellia::CellTopology::quad();
    }
    vector< CellTopoPtr > cellTopos(numElements, topo);

    FieldContainer<double> quadBoundaryPoints(4,2);
    quadBoundaryPoints(0,0) = x0;
    quadBoundaryPoints(0,1) = y0;
    quadBoundaryPoints(1,0) = x0 + width;
    quadBoundaryPoints(1,1) = y0;
    quadBoundaryPoints(2,0) = x0 + width;
    quadBoundaryPoints(2,1) = y0 + height;
    quadBoundaryPoints(3,0) = x0;
    quadBoundaryPoints(3,1) = y0 + height;
    //  cout << "creating mesh with boundary points:\n" << quadBoundaryPoints;

    double southWest_x = quadBoundaryPoints(0,0),
           southWest_y = quadBoundaryPoints(0,1);

    double elemWidth = width / horizontalElements;
    double elemHeight = height / verticalElements;

    // set up vertices:
    // vertexIndices is for easy vertex lookup by (x,y) index for our Cartesian grid:
    vector< vector<int> > vertexIndices(horizontalElements+1, vector<int>(verticalElements+1));
    for (int i=0; i<=horizontalElements; i++)
    {
      for (int j=0; j<=verticalElements; j++)
      {
        vertexIndices[i][j] = vertices.size();
        vector<double> vertex(spaceDim);
        vertex[0] = southWest_x + elemWidth*i;
        vertex[1] = southWest_y + elemHeight*j;
        vertices.push_back(vertex);
      }
    }

    for (int i=0; i<horizontalElements; i++)
    {
      for (int j=0; j<verticalElements; j++)
      {
        if (!divideIntoTriangles)
        {
          vector<unsigned> elemVertices;
          elemVertices.push_back(vertexIndices[i][j]);
          elemVertices.push_back(vertexIndices[i+1][j]);
          elemVertices.push_back(vertexIndices[i+1][j+1]);
          elemVertices.push_back(vertexIndices[i][j+1]);
          allElementVertices.push_back(elemVertices);
        }
        else
        {
          vector<unsigned> elemVertices1, elemVertices2; // elem1 is SE of quad, elem2 is NW
          elemVertices1.push_back(vertexIndices[i][j]);     // SIDE1 is SOUTH side of quad
          elemVertices1.push_back(vertexIndices[i+1][j]);   // SIDE2 is EAST
          elemVertices1.push_back(vertexIndices[i+1][j+1]); // SIDE3 is diagonal
          elemVertices2.push_back(vertexIndices[i][j+1]);   // SIDE1 is WEST
          elemVertices2.push_back(vertexIndices[i][j]);     // SIDE2 is diagonal
          elemVertices2.push_back(vertexIndices[i+1][j+1]); // SIDE3 is NORTH

          allElementVertices.push_back(elemVertices1);
          allElementVertices.push_back(elemVertices2);
        }
      }
    }

    return Teuchos::rcp( new Mesh(vertices, allElementVertices, bf, H1Order, delta_k, useConformingTraces, *trialOrderEnhancements, *testOrderEnhancements, *periodicBCs) );
  }
}

/*class ParametricRect : public ParametricCurve {
  double _width, _height, _x0, _y0;
  vector< ParametricCurvePtr > _edgeLines;
  vector< double > _switchValues;
public:
  ParametricRect(double width, double height, double x0, double y0) {
    // starts at the positive x axis and proceeds counter-clockwise, just like our parametric circle

    _width = width; _height = height; _x0 = x0; _y0 = y0;
    _edgeLines.push_back(ParametricCurve::line(x0 + width/2.0, y0 + 0, x0 + width/2.0, y0 + height/2.0));
    _edgeLines.push_back(ParametricCurve::line(x0 + width/2.0, y0 + height/2.0, x0 - width/2.0, y0 + height/2.0));
    _edgeLines.push_back(ParametricCurve::line(x0 - width/2.0, y0 + height/2.0, x0 - width/2.0, y0 - height/2.0));
    _edgeLines.push_back(ParametricCurve::line(x0 - width/2.0, y0 - height/2.0, x0 + width/2.0, y0 - height/2.0));
    _edgeLines.push_back(ParametricCurve::line(x0 + width/2.0, y0 - height/2.0, x0 + width/2.0, y0 + 0));

    // switchValues are the points in (0,1) where we switch from one edge line to the next
    _switchValues.push_back(0.0);
    _switchValues.push_back(0.125);
    _switchValues.push_back(0.375);
    _switchValues.push_back(0.625);
    _switchValues.push_back(0.875);
    _switchValues.push_back(1.0);
  }
  void value(double t, double &x, double &y) {
    for (int i=0; i<_edgeLines.size(); i++) {
      if ( (t >= _switchValues[i]) && (t <= _switchValues[i+1]) ) {
        double edge_t = (t - _switchValues[i]) / (_switchValues[i+1] - _switchValues[i]);
        _edgeLines[i]->value(edge_t, x, y);
        return;
      }
    }
  }
};*/

MeshPtr MeshFactory::quadMesh(TBFPtr<double> bf, int H1Order, FieldContainer<double> &quadNodes, int pToAddTest)
{
  if (quadNodes.size() != 8)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "quadNodes must be 4 x 2");
  }
  int spaceDim = 2;
  vector< vector<double> > vertices;
  for (int i=0; i<4; i++)
  {
    vector<double> vertex(spaceDim);
    vertex[0] = quadNodes[2*i];
    vertex[1] = quadNodes[2*i+1];
    vertices.push_back(vertex);
  }
  vector< vector<unsigned> > elementVertices;
  vector<unsigned> cell0;
  cell0.push_back(0);
  cell0.push_back(1);
  cell0.push_back(2);
  cell0.push_back(3);
  elementVertices.push_back(cell0);

  MeshPtr mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, bf, H1Order, pToAddTest) );
  return mesh;
}

MeshPtr MeshFactory::quadMesh(TBFPtr<double> bf, int H1Order, int pToAddTest,
                              double width, double height, int horizontalElements, int verticalElements, bool divideIntoTriangles,
                              double x0, double y0, vector<PeriodicBCPtr> periodicBCs)
{

  Teuchos::ParameterList pl;

  pl.set("useMinRule", false);
  pl.set("bf",bf);
  pl.set("H1Order", H1Order);
  pl.set("delta_k", pToAddTest);
  pl.set("horizontalElements", horizontalElements);
  pl.set("verticalElements", verticalElements);
  pl.set("width", width);
  pl.set("height", height);
  pl.set("divideIntoTriangles", divideIntoTriangles);
  pl.set("x0",x0);
  pl.set("y0",y0);
  pl.set("periodicBCs", &periodicBCs);

  return quadMesh(pl);
}

MeshPtr MeshFactory::quadMeshMinRule(TBFPtr<double> bf, int H1Order, int pToAddTest,
                                     double width, double height, int horizontalElements, int verticalElements,
                                     bool divideIntoTriangles, double x0, double y0, vector<PeriodicBCPtr> periodicBCs)
{
  Teuchos::ParameterList pl;

  pl.set("useMinRule", true);
  pl.set("bf",bf);
  pl.set("H1Order", H1Order);
  pl.set("delta_k", pToAddTest);
  pl.set("horizontalElements", horizontalElements);
  pl.set("verticalElements", verticalElements);
  pl.set("width", width);
  pl.set("height", height);
  pl.set("divideIntoTriangles", divideIntoTriangles);
  pl.set("x0",x0);
  pl.set("y0",y0);
  pl.set("periodicBCs", &periodicBCs);

  return quadMesh(pl);
}

MeshTopologyPtr MeshFactory::quadMeshTopology(double width, double height, int horizontalElements, int verticalElements, bool divideIntoTriangles,
    double x0, double y0, vector<PeriodicBCPtr> periodicBCs)
{
  vector<vector<double> > vertices;
  vector< vector<unsigned> > allElementVertices;

  int numElements = divideIntoTriangles ? horizontalElements * verticalElements * 2 : horizontalElements * verticalElements;

  CellTopoPtr topo;
  if (divideIntoTriangles)
  {
    topo = Camellia::CellTopology::triangle();
  }
  else
  {
    topo = Camellia::CellTopology::quad();
  }
  vector< CellTopoPtr > cellTopos(numElements, topo);

  int spaceDim = 2;

  FieldContainer<double> quadBoundaryPoints(4,spaceDim);
  quadBoundaryPoints(0,0) = x0;
  quadBoundaryPoints(0,1) = y0;
  quadBoundaryPoints(1,0) = x0 + width;
  quadBoundaryPoints(1,1) = y0;
  quadBoundaryPoints(2,0) = x0 + width;
  quadBoundaryPoints(2,1) = y0 + height;
  quadBoundaryPoints(3,0) = x0;
  quadBoundaryPoints(3,1) = y0 + height;
  //  cout << "creating mesh with boundary points:\n" << quadBoundaryPoints;

  double southWest_x = quadBoundaryPoints(0,0),
         southWest_y = quadBoundaryPoints(0,1);

  double elemWidth = width / horizontalElements;
  double elemHeight = height / verticalElements;

  // set up vertices:
  // vertexIndices is for easy vertex lookup by (x,y) index for our Cartesian grid:
  vector< vector<int> > vertexIndices(horizontalElements+1, vector<int>(verticalElements+1));
  for (int i=0; i<=horizontalElements; i++)
  {
    for (int j=0; j<=verticalElements; j++)
    {
      vertexIndices[i][j] = vertices.size();
      vector<double> vertex(spaceDim);
      vertex[0] = southWest_x + elemWidth*i;
      vertex[1] = southWest_y + elemHeight*j;
      vertices.push_back(vertex);
    }
  }

  for (int i=0; i<horizontalElements; i++)
  {
    for (int j=0; j<verticalElements; j++)
    {
      if (!divideIntoTriangles)
      {
        vector<unsigned> elemVertices;
        elemVertices.push_back(vertexIndices[i][j]);
        elemVertices.push_back(vertexIndices[i+1][j]);
        elemVertices.push_back(vertexIndices[i+1][j+1]);
        elemVertices.push_back(vertexIndices[i][j+1]);
        allElementVertices.push_back(elemVertices);
      }
      else
      {
        vector<unsigned> elemVertices1, elemVertices2; // elem1 is SE of quad, elem2 is NW
        elemVertices1.push_back(vertexIndices[i][j]);
        elemVertices1.push_back(vertexIndices[i+1][j]);
        elemVertices1.push_back(vertexIndices[i+1][j+1]);
        elemVertices2.push_back(vertexIndices[i][j+1]);
        elemVertices2.push_back(vertexIndices[i][j]);
        elemVertices2.push_back(vertexIndices[i+1][j+1]);

        allElementVertices.push_back(elemVertices1);
        allElementVertices.push_back(elemVertices2);
      }
    }
  }

  MeshGeometryPtr geometry = Teuchos::rcp( new MeshGeometry(vertices, allElementVertices, cellTopos));
  return Teuchos::rcp( new MeshTopology(geometry, periodicBCs) );
}

MeshPtr MeshFactory::hemkerMesh(double meshWidth, double meshHeight, double cylinderRadius, // cylinder is centered in quad mesh.
                                TBFPtr<double> bilinearForm, int H1Order, int pToAddTest)
{
  return shiftedHemkerMesh(-meshWidth/2, meshWidth/2, meshHeight, cylinderRadius, bilinearForm, H1Order, pToAddTest);
}

MeshTopologyPtr MeshFactory::importMOABMesh(string filePath)
{
  return MOABReader::readMOABMesh(filePath);
}

MeshPtr MeshFactory::intervalMesh(TBFPtr<double> bf, double xLeft, double xRight, int numElements, int H1Order, int delta_k)
{
  MeshTopologyPtr meshTopology = intervalMeshTopology(xLeft, xRight, numElements);
  return Teuchos::rcp( new Mesh(meshTopology, bf, H1Order, delta_k) );
}

MeshTopologyPtr MeshFactory::intervalMeshTopology(double xLeft, double xRight, int numElements)
{
  int n = numElements;
  vector< vector<double> > vertices(n+1);
  vector<double> vertex(1);
  double length = xRight - xLeft;
  vector< vector<IndexType> > elementVertices(n);
  vector<IndexType> oneElement(2);
  for (int i=0; i<n+1; i++)
  {
    vertex[0] = xLeft + (i * length) / n;
    //    cout << "vertex " << i << ": " << vertex[0] << endl;
    vertices[i] = vertex;
    if (i != n)
    {
      oneElement[0] = i;
      oneElement[1] = i+1;
      elementVertices[i] = oneElement;
    }
  }
  CellTopoPtr topo = Camellia::CellTopology::line();
  vector< CellTopoPtr > cellTopos(numElements, topo);
  MeshGeometryPtr geometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos));

  MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(geometry) );
  return meshTopology;
}

MeshPtr MeshFactory::rectilinearMesh(TBFPtr<double> bf, vector<double> dimensions, vector<int> elementCounts, int H1Order, int pToAddTest, vector<double> x0)
{
  int spaceDim = dimensions.size();
  if (pToAddTest==-1)
  {
    pToAddTest = spaceDim;
  }

  MeshTopologyPtr meshTopology = rectilinearMeshTopology(dimensions, elementCounts, x0);

  return Teuchos::rcp( new Mesh(meshTopology, bf, H1Order, pToAddTest) );
}

MeshTopologyPtr MeshFactory::rectilinearMeshTopology(vector<double> dimensions, vector<int> elementCounts, vector<double> x0)
{
  int spaceDim = dimensions.size();

  if (x0.size()==0)
  {
    for (int d=0; d<spaceDim; d++)
    {
      x0.push_back(0.0);
    }
  }

  if (elementCounts.size() != dimensions.size())
  {
    cout << "Element count container must match dimensions container in length.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Element count container must match dimensions container in length.\n");
  }

  if (spaceDim == 1)
  {
    double xLeft = x0[0];
    double xRight = dimensions[0] + xLeft;
    return MeshFactory::intervalMeshTopology(xLeft, xRight, elementCounts[0]);
  }

  if (spaceDim == 2)
  {
    return MeshFactory::quadMeshTopology(dimensions[0], dimensions[1], elementCounts[0], elementCounts[1], false, x0[0], x0[1]);
  }

  if (spaceDim != 3)
  {
    cout << "For now, only spaceDim 1,2,3 are supported by this MeshFactory method.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "For now, only spaceDim 1,2,3 are is supported by this MeshFactory method.");
  }

  CellTopoPtr topo;
  if (spaceDim==1)
  {
    topo = Camellia::CellTopology::line();
  }
  else if (spaceDim==2)
  {
    topo = Camellia::CellTopology::quad();
  }
  else if (spaceDim==3)
  {
    topo = Camellia::CellTopology::hexahedron();
  }
  else
  {
    cout << "Unsupported spatial dimension.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported spatial dimension");
  }

  int numElements = 1;
  vector<double> elemLinearMeasures(spaceDim);
  vector<double> origin = x0;
  for (int d=0; d<spaceDim; d++)
  {
    numElements *= elementCounts[d];
    elemLinearMeasures[d] = dimensions[d] / elementCounts[d];
  }
  vector< CellTopoPtr > cellTopos(numElements, topo);

  map< vector<int>, unsigned> vertexLookup;
  vector< vector<double> > vertices;

  for (int i=0; i<elementCounts[0]+1; i++)
  {
    double x = origin[0] + elemLinearMeasures[0] * i;

    for (int j=0; j<elementCounts[1]+1; j++)
    {
      double y = origin[1] + elemLinearMeasures[1] * j;

      for (int k=0; k<elementCounts[2]+1; k++)
      {
        double z = origin[2] + elemLinearMeasures[2] * k;

        vector<int> vertexIndex;
        vertexIndex.push_back(i);
        vertexIndex.push_back(j);
        vertexIndex.push_back(k);

        vector<double> vertex;
        vertex.push_back(x);
        vertex.push_back(y);
        vertex.push_back(z);

        vertexLookup[vertexIndex] = vertices.size();
        vertices.push_back(vertex);
      }
    }
  }

  vector< vector<unsigned> > elementVertices;
  for (int i=0; i<elementCounts[0]; i++)
  {
    for (int j=0; j<elementCounts[1]; j++)
    {
      for (int k=0; k<elementCounts[2]; k++)
      {
        vector< vector<int> > vertexIntCoords(8, vector<int>(3));
        vertexIntCoords[0][0] = i;
        vertexIntCoords[0][1] = j;
        vertexIntCoords[0][2] = k;
        vertexIntCoords[1][0] = i+1;
        vertexIntCoords[1][1] = j;
        vertexIntCoords[1][2] = k;
        vertexIntCoords[2][0] = i+1;
        vertexIntCoords[2][1] = j+1;
        vertexIntCoords[2][2] = k;
        vertexIntCoords[3][0] = i;
        vertexIntCoords[3][1] = j+1;
        vertexIntCoords[3][2] = k;
        vertexIntCoords[4][0] = i;
        vertexIntCoords[4][1] = j;
        vertexIntCoords[4][2] = k+1;
        vertexIntCoords[5][0] = i+1;
        vertexIntCoords[5][1] = j;
        vertexIntCoords[5][2] = k+1;
        vertexIntCoords[6][0] = i+1;
        vertexIntCoords[6][1] = j+1;
        vertexIntCoords[6][2] = k+1;
        vertexIntCoords[7][0] = i;
        vertexIntCoords[7][1] = j+1;
        vertexIntCoords[7][2] = k+1;

        vector<unsigned> elementVertexOrdinals;
        for (int n=0; n<8; n++)
        {
          elementVertexOrdinals.push_back(vertexLookup[vertexIntCoords[n]]);
        }

        elementVertices.push_back(elementVertexOrdinals);
      }
    }
  }

  MeshGeometryPtr geometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos));

  MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(geometry) );
  return meshTopology;
}

MeshGeometryPtr MeshFactory::shiftedHemkerGeometry(double xLeft, double xRight, double meshHeight, double cylinderRadius)
{
  return shiftedHemkerGeometry(xLeft, xRight, -meshHeight/2.0, meshHeight/2.0, cylinderRadius);
}

MeshGeometryPtr MeshFactory::shiftedSquareCylinderGeometry(double xLeft, double xRight, double meshHeight, double squareDiameter)
{
  vector< vector<double> > vertices;
  vector<double> xs = {xLeft, -squareDiameter/2, squareDiameter/2, xRight};
  vector<double> ys = {-meshHeight/2, -squareDiameter/2, squareDiameter/2, meshHeight/2};
  for (int j=0; j < 4; j++)
  {
    for (int i=0; i < 4; i++)
    {
      vector<double> vertex(2);
      vertex[0] = xs[i];
      vertex[1] = ys[j];
      vertices.push_back(vertex);
    }
  }

  vector< vector<unsigned> > elementVertices;
  vector< CellTopoPtr > cellTopos;
  CellTopoPtr quad_4 = Camellia::CellTopology::quad();
  for (unsigned j=0; j < 3; j++)
  {
    for (unsigned i=0; i < 3; i++)
    {
      vector<unsigned> elVertex;
      elVertex.push_back(4*j+i);
      elVertex.push_back(4*j+i+1);
      elVertex.push_back(4*(j+1)+i+1);
      elVertex.push_back(4*(j+1)+i);
      if (!(i == 1 && j == 1))
      {
        elementVertices.push_back(elVertex);
        cellTopos.push_back(quad_4);
      }
    }
  }


  return Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );
}

MeshPtr MeshFactory::readMesh(string filePath, TBFPtr<double> bilinearForm, int H1Order, int pToAdd)
{
  ifstream mshFile;
  mshFile.open(filePath.c_str());
  TEUCHOS_TEST_FOR_EXCEPTION(mshFile.fail(), std::invalid_argument, "Could not open msh file");
  string line;
  getline(mshFile, line);
  while (line != "$Nodes")
  {
    getline(mshFile, line);
  }
  int numNodes;
  mshFile >> numNodes;
  vector<vector<double> > vertices;
  int dummy;
  for (int i=0; i < numNodes; i++)
  {
    vector<double> vertex(2);
    mshFile >> dummy;
    mshFile >> vertex[0] >> vertex[1] >> dummy;
    vertices.push_back(vertex);
  }
  while (line != "$Elements")
  {
    getline(mshFile, line);
  }
  int numElems;
  mshFile >> numElems;
  int elemType;
  int numTags;
  vector< vector<unsigned> > elementIndices;
  for (int i=0; i < numElems; i++)
  {
    mshFile >> dummy >> elemType >> numTags;
    for (int j=0; j < numTags; j++)
      mshFile >> dummy;
    if (elemType == 2)
    {
      vector<unsigned> elemIndices(3);
      mshFile >> elemIndices[0] >> elemIndices[1] >> elemIndices[2];
      elemIndices[0]--;
      elemIndices[1]--;
      elemIndices[2]--;
      elementIndices.push_back(elemIndices);
    }
    if (elemType == 4)
    {
      vector<unsigned> elemIndices(3);
      mshFile >> elemIndices[0] >> elemIndices[1] >> elemIndices[2];
      elemIndices[0]--;
      elemIndices[1]--;
      elemIndices[2]--;
      elementIndices.push_back(elemIndices);
    }
    else
    {
      getline(mshFile, line);
    }
  }
  mshFile.close();

  MeshPtr mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, bilinearForm, H1Order, pToAdd) );
  return mesh;
}

MeshPtr MeshFactory::readTriangle(string filePath, TBFPtr<double> bilinearForm, int H1Order, int pToAdd)
{
  ifstream nodeFile;
  ifstream eleFile;
  string nodeFileName = filePath+".node";
  string eleFileName = filePath+".ele";
  nodeFile.open(nodeFileName.c_str());
  eleFile.open(eleFileName.c_str());
  TEUCHOS_TEST_FOR_EXCEPTION(nodeFile.fail(), std::invalid_argument, "Could not open node file: "+nodeFileName);
  TEUCHOS_TEST_FOR_EXCEPTION(eleFile.fail(), std::invalid_argument, "Could not open ele file: "+eleFileName);
  // Read node file
  string line;
  int numNodes;
  nodeFile >> numNodes;
  getline(nodeFile, line);
  vector<vector<double> > vertices;
  int dummy;
  int spaceDim = 2;
  vector<double> pt(spaceDim);
  for (int i=0; i < numNodes; i++)
  {
    nodeFile >> dummy >> pt[0] >> pt[1];
    getline(nodeFile, line);
    vertices.push_back(pt);
  }
  nodeFile.close();
  // Read ele file
  int numElems;
  eleFile >> numElems;
  getline(eleFile, line);
  vector< vector<unsigned> > elementIndices;
  vector<unsigned> el(3);
  for (int i=0; i < numElems; i++)
  {
    eleFile >> dummy >> el[0] >> el[1] >> el[2];
    el[0]--;
    el[1]--;
    el[2]--;
    elementIndices.push_back(el);
  }
  eleFile.close();

  MeshPtr mesh = Teuchos::rcp( new Mesh(vertices, elementIndices, bilinearForm, H1Order, pToAdd) );
  return mesh;
}

MeshPtr MeshFactory::buildQuadMesh(const FieldContainer<double> &quadBoundaryPoints,
                                   int horizontalElements, int verticalElements,
                                   TBFPtr<double> bilinearForm,
                                   int H1Order, int pTest, bool triangulate, bool useConformingTraces,
                                   map<int,int> trialOrderEnhancements,
                                   map<int,int> testOrderEnhancements)
{
  //  if (triangulate) cout << "Mesh: Triangulating\n" << endl;
  int pToAddTest = pTest - H1Order;
  // rectBoundaryPoints dimensions: (4,2) -- and should be in counterclockwise order

  // check that inputs match the assumptions (of a rectilinear mesh)
  TEUCHOS_TEST_FOR_EXCEPTION( ( quadBoundaryPoints.dimension(0) != 4 ) || ( quadBoundaryPoints.dimension(1) != 2 ),
                              std::invalid_argument,
                              "quadBoundaryPoints should be dimensions (4,2), points in ccw order.");
  double southWest_x = quadBoundaryPoints(0,0),
         southWest_y = quadBoundaryPoints(0,1),
         southEast_x = quadBoundaryPoints(1,0),
//  southEast_y = quadBoundaryPoints(1,1),
//  northEast_x = quadBoundaryPoints(2,0),
//  northEast_y = quadBoundaryPoints(2,1),
//  northWest_x = quadBoundaryPoints(3,0),
         northWest_y = quadBoundaryPoints(3,1);

  double width = southEast_x - southWest_x;
  double height = northWest_y - southWest_y;

  Teuchos::ParameterList pl;

  pl.set("useMinRule", false);
  pl.set("bf",bilinearForm);
  pl.set("H1Order", H1Order);
  pl.set("delta_k", pToAddTest);
  pl.set("horizontalElements", horizontalElements);
  pl.set("verticalElements", verticalElements);
  pl.set("divideIntoTriangles", triangulate);
  pl.set("useConformingTraces", useConformingTraces);
  pl.set("trialOrderEnhancements", &trialOrderEnhancements);
  pl.set("testOrderEnhancements", &testOrderEnhancements);
  pl.set("x0",southWest_x);
  pl.set("y0",southWest_y);
  pl.set("width", width);
  pl.set("height",height);

  return quadMesh(pl);
}

MeshPtr MeshFactory::buildQuadMeshHybrid(const FieldContainer<double> &quadBoundaryPoints,
    int horizontalElements, int verticalElements,
    TBFPtr<double> bilinearForm,
    int H1Order, int pTest, bool useConformingTraces)
{
  int pToAddToTest = pTest - H1Order;
  int spaceDim = 2;
  // rectBoundaryPoints dimensions: (4,2) -- and should be in counterclockwise order

  vector<vector<double> > vertices;
  vector< vector<unsigned> > allElementVertices;

  TEUCHOS_TEST_FOR_EXCEPTION( ( quadBoundaryPoints.dimension(0) != 4 ) || ( quadBoundaryPoints.dimension(1) != 2 ),
                              std::invalid_argument,
                              "quadBoundaryPoints should be dimensions (4,2), points in ccw order.");

  int numDimensions = 2;

  double southWest_x = quadBoundaryPoints(0,0),
         southWest_y = quadBoundaryPoints(0,1),
         southEast_x = quadBoundaryPoints(1,0),
         southEast_y = quadBoundaryPoints(1,1),
         northEast_x = quadBoundaryPoints(2,0),
         northEast_y = quadBoundaryPoints(2,1),
         northWest_x = quadBoundaryPoints(3,0),
         northWest_y = quadBoundaryPoints(3,1);

  double elemWidth = (southEast_x - southWest_x) / horizontalElements;
  double elemHeight = (northWest_y - southWest_y) / verticalElements;

  int cellID = 0;

  // set up vertices:
  // vertexIndices is for easy vertex lookup by (x,y) index for our Cartesian grid:
  vector< vector<int> > vertexIndices(horizontalElements+1, vector<int>(verticalElements+1));
  for (int i=0; i<=horizontalElements; i++)
  {
    for (int j=0; j<=verticalElements; j++)
    {
      vertexIndices[i][j] = vertices.size();
      vector<double> vertex(spaceDim);
      vertex[0] = southWest_x + elemWidth*i;
      vertex[1] = southWest_y + elemHeight*j;
      vertices.push_back(vertex);
    }
  }

  int SOUTH = 0, EAST = 1, NORTH = 2, WEST = 3;
  int SIDE1 = 0, SIDE2 = 1, SIDE3 = 2;
  for (int i=0; i<horizontalElements; i++)
  {
    for (int j=0; j<verticalElements; j++)
    {
      bool triangulate = (i >= horizontalElements / 2); // triangles on right half of mesh
      if ( ! triangulate )
      {
        vector<unsigned> elemVertices;
        elemVertices.push_back(vertexIndices[i][j]);
        elemVertices.push_back(vertexIndices[i+1][j]);
        elemVertices.push_back(vertexIndices[i+1][j+1]);
        elemVertices.push_back(vertexIndices[i][j+1]);
        allElementVertices.push_back(elemVertices);
      }
      else
      {
        vector<unsigned> elemVertices1, elemVertices2; // elem1 is SE of quad, elem2 is NW
        elemVertices1.push_back(vertexIndices[i][j]);     // SIDE1 is SOUTH side of quad
        elemVertices1.push_back(vertexIndices[i+1][j]);   // SIDE2 is EAST
        elemVertices1.push_back(vertexIndices[i+1][j+1]); // SIDE3 is diagonal
        elemVertices2.push_back(vertexIndices[i][j+1]);   // SIDE1 is WEST
        elemVertices2.push_back(vertexIndices[i][j]);     // SIDE2 is diagonal
        elemVertices2.push_back(vertexIndices[i+1][j+1]); // SIDE3 is NORTH

        allElementVertices.push_back(elemVertices1);
        allElementVertices.push_back(elemVertices2);
      }
    }
  }
  return Teuchos::rcp( new Mesh(vertices,allElementVertices,bilinearForm,H1Order,pToAddToTest,useConformingTraces));
}

void MeshFactory::quadMeshCellIDs(FieldContainer<int> &cellIDs, int horizontalElements, int verticalElements, bool useTriangles)
{
  // populates cellIDs with either (h,v) or (h,v,2)
  // where h: horizontalElements (indexed by i, below)
  //       v: verticalElements   (indexed by j)
  //       2: triangles per quad (indexed by k)

  TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.dimension(0)!=horizontalElements,
                             std::invalid_argument,
                             "cellIDs should have dimensions: (horizontalElements, verticalElements) or (horizontalElements, verticalElements,2)");
  TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.dimension(1)!=verticalElements,
                             std::invalid_argument,
                             "cellIDs should have dimensions: (horizontalElements, verticalElements) or (horizontalElements, verticalElements,2)");
  if (useTriangles)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.dimension(2)!=2,
                               std::invalid_argument,
                               "cellIDs should have dimensions: (horizontalElements, verticalElements,2)");
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.rank() != 3,
                               std::invalid_argument,
                               "cellIDs should have dimensions: (horizontalElements, verticalElements,2)");
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.rank() != 2,
                               std::invalid_argument,
                               "cellIDs should have dimensions: (horizontalElements, verticalElements)");
  }

  int cellID = 0;
  for (int i=0; i<horizontalElements; i++)
  {
    for (int j=0; j<verticalElements; j++)
    {
      if (useTriangles)
      {
        cellIDs(i,j,0) = cellID++;
        cellIDs(i,j,1) = cellID++;
      }
      else
      {
        cellIDs(i,j) = cellID++;
      }
    }
  }
}

MeshGeometryPtr MeshFactory::shiftedHemkerGeometry(double xLeft, double xRight, double yBottom, double yTop, double cylinderRadius)
{
  double meshHeight = yTop - yBottom;
  double embeddedSquareSideLength = cylinderRadius+meshHeight/2;
  return shiftedHemkerGeometry(xLeft, xRight, yBottom, yTop, cylinderRadius, embeddedSquareSideLength);
}

MeshGeometryPtr MeshFactory::shiftedHemkerGeometry(double xLeft, double xRight, double yBottom, double yTop, double cylinderRadius, double embeddedSquareSideLength)
{
  // first, set up an 8-element mesh, centered at the origin
  ParametricCurvePtr circle = ParametricCurve::circle(cylinderRadius, 0, 0);
  double meshHeight = yTop - yBottom;
  ParametricCurvePtr rect = parametricRect(embeddedSquareSideLength, embeddedSquareSideLength, 0, 0);

  int numPoints = 8; // 8 points on rect, 8 on circle
  int spaceDim = 2;
  vector< vector<double> > vertices;
  vector<double> innerVertex(spaceDim), outerVertex(spaceDim);
  FieldContainer<double> innerVertices(numPoints,spaceDim), outerVertices(numPoints,spaceDim); // these are just for easy debugging output

  vector<IndexType> innerVertexIndices;
  vector<IndexType> outerVertexIndices;

  double t = 0;
  for (int i=0; i<numPoints; i++)
  {
    circle->value(t, innerVertices(i,0), innerVertices(i,1));
    rect  ->value(t, outerVertices(i,0), outerVertices(i,1));
    circle->value(t, innerVertex[0], innerVertex[1]);
    rect  ->value(t, outerVertex[0], outerVertex[1]);
    innerVertexIndices.push_back(vertices.size());
    vertices.push_back(innerVertex);
    outerVertexIndices.push_back(vertices.size());
    vertices.push_back(outerVertex);
    t += 1.0 / numPoints;
  }

  //  cout << "innerVertices:\n" << innerVertices;
  //  cout << "outerVertices:\n" << outerVertices;

//  GnuPlotUtil::writeXYPoints("/tmp/innerVertices.dat", innerVertices);
//  GnuPlotUtil::writeXYPoints("/tmp/outerVertices.dat", outerVertices);

  vector< vector<IndexType> > elementVertices;

  int totalVertices = vertices.size();

  t = 0;
  map< pair<IndexType, IndexType>, ParametricCurvePtr > edgeToCurveMap;
  for (int i=0; i<numPoints; i++)   // numPoints = numElements
  {
    vector<IndexType> vertexIndices;
    int innerIndex0 = (i * 2) % totalVertices;
    int innerIndex1 = ((i+1) * 2) % totalVertices;
    int outerIndex0 = (i * 2 + 1) % totalVertices;
    int outerIndex1 = ((i+1) * 2 + 1) % totalVertices;
    vertexIndices.push_back(innerIndex0);
    vertexIndices.push_back(outerIndex0);
    vertexIndices.push_back(outerIndex1);
    vertexIndices.push_back(innerIndex1);
    elementVertices.push_back(vertexIndices);

    //    cout << "innerIndex0: " << innerIndex0 << endl;
    //    cout << "innerIndex1: " << innerIndex1 << endl;
    //    cout << "outerIndex0: " << outerIndex0 << endl;
    //    cout << "outerIndex1: " << outerIndex1 << endl;

    pair<int, int> innerEdge = make_pair(innerIndex1, innerIndex0); // order matters
    edgeToCurveMap[innerEdge] = ParametricCurve::subCurve(circle, t+1.0/numPoints, t);
    t += 1.0/numPoints;
  }

  int boundaryVertexOffset = vertices.size();
  // make some new vertices, going counter-clockwise:
  ParametricCurvePtr meshRect = parametricRect(xRight-xLeft, meshHeight, 0.5*(xLeft+xRight), 0.5*(yBottom + yTop));
  vector<double> boundaryVertex(spaceDim);
  boundaryVertex[0] = xRight;
  boundaryVertex[1] = 0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = meshHeight / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = 0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = -embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = xLeft;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = 0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = -embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = -meshHeight / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = -embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = 0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  boundaryVertex[0] = xRight;
  vertices.push_back(boundaryVertex);

  boundaryVertex[1] = -embeddedSquareSideLength / 2.0;
  vertices.push_back(boundaryVertex);

  vector<IndexType> vertexIndices(4);
  vertexIndices[0] = outerVertexIndices[0];
  vertexIndices[1] = boundaryVertexOffset;
  vertexIndices[2] = boundaryVertexOffset + 1;
  vertexIndices[3] = outerVertexIndices[1];
  elementVertices.push_back(vertexIndices);

  // mesh NE corner
  vertexIndices[0] = outerVertexIndices[1];
  vertexIndices[1] = boundaryVertexOffset + 1;
  vertexIndices[2] = boundaryVertexOffset + 2;
  vertexIndices[3] = boundaryVertexOffset + 3;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = outerVertexIndices[2];
  vertexIndices[1] = outerVertexIndices[1];
  vertexIndices[2] = boundaryVertexOffset + 3;
  vertexIndices[3] = boundaryVertexOffset + 4;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = outerVertexIndices[3];
  vertexIndices[1] = outerVertexIndices[2];
  vertexIndices[2] = boundaryVertexOffset + 4;
  vertexIndices[3] = boundaryVertexOffset + 5;
  elementVertices.push_back(vertexIndices);

  // NW corner
  vertexIndices[0] = boundaryVertexOffset + 7;
  vertexIndices[1] = outerVertexIndices[3];
  vertexIndices[2] = boundaryVertexOffset + 5;
  vertexIndices[3] = boundaryVertexOffset + 6;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 8;
  vertexIndices[1] = outerVertexIndices[4];
  vertexIndices[2] = outerVertexIndices[3];
  vertexIndices[3] = boundaryVertexOffset + 7;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 9;
  vertexIndices[1] = outerVertexIndices[5];
  vertexIndices[2] = outerVertexIndices[4];
  vertexIndices[3] = boundaryVertexOffset + 8;
  elementVertices.push_back(vertexIndices);

  // SW corner
  vertexIndices[0] = boundaryVertexOffset + 10;
  vertexIndices[1] = boundaryVertexOffset + 11;
  vertexIndices[2] = outerVertexIndices[5];
  vertexIndices[3] = boundaryVertexOffset + 9;
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 11;
  vertexIndices[1] = boundaryVertexOffset + 12;
  vertexIndices[2] = outerVertexIndices[6];
  vertexIndices[3] = outerVertexIndices[5];
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = boundaryVertexOffset + 12;
  vertexIndices[1] = boundaryVertexOffset + 13;
  vertexIndices[2] = outerVertexIndices[7];
  vertexIndices[3] = outerVertexIndices[6];
  elementVertices.push_back(vertexIndices);

  // SE corner
  vertexIndices[0] = boundaryVertexOffset + 13;
  vertexIndices[1] = boundaryVertexOffset + 14;
  vertexIndices[2] = boundaryVertexOffset + 15;
  vertexIndices[3] = outerVertexIndices[7];
  elementVertices.push_back(vertexIndices);

  vertexIndices[0] = outerVertexIndices[7];
  vertexIndices[1] = boundaryVertexOffset + 15;
  vertexIndices[2] = boundaryVertexOffset;
  vertexIndices[3] = outerVertexIndices[0];
  elementVertices.push_back(vertexIndices);

  return Teuchos::rcp( new MeshGeometry(vertices, elementVertices, edgeToCurveMap) );
}

MeshPtr MeshFactory::shiftedHemkerMesh(double xLeft, double xRight, double meshHeight, double cylinderRadius, // cylinder is centered in quad mesh.
                                       TBFPtr<double> bilinearForm, int H1Order, int pToAddTest)
{
  MeshGeometryPtr geometry = MeshFactory::shiftedHemkerGeometry(xLeft, xRight, meshHeight, cylinderRadius);
  MeshPtr mesh = Teuchos::rcp( new Mesh(geometry->vertices(), geometry->elementVertices(),
                                        bilinearForm, H1Order, pToAddTest) );

  map< pair<IndexType, IndexType>, ParametricCurvePtr > localEdgeToCurveMap = geometry->edgeToCurveMap();
  map< pair<GlobalIndexType, GlobalIndexType>, ParametricCurvePtr > globalEdgeToCurveMap(localEdgeToCurveMap.begin(),localEdgeToCurveMap.end());
  mesh->setEdgeToCurveMap(globalEdgeToCurveMap);
  return mesh;
}

// TODO: test this!
MeshPtr MeshFactory::spaceTimeMesh(MeshTopologyPtr spatialMeshTopology, double t0, double t1,
                                   TBFPtr<double> bf, int spatialH1Order, int temporalH1Order, int pToAdd)
{
  MeshTopologyPtr meshTopology = spaceTimeMeshTopology(spatialMeshTopology, t0, t1);

  vector<int> H1Order(2);
  H1Order[0] = spatialH1Order;
  H1Order[1] = temporalH1Order;

  MeshPtr mesh = Teuchos::rcp( new Mesh (meshTopology, bf, H1Order, pToAdd) );

  return mesh;
}

// TODO: test this!
MeshTopologyPtr MeshFactory::spaceTimeMeshTopology(MeshTopologyPtr spatialMeshTopology, double t0, double t1, int temporalDivisions)
{
  // we allow spatialMeshTopology to have been refined; we start with a coarse space-time topology matching the root spatial topology,
  // and then refine accordingly...

  // (For now, though, we do make the assumption that all refinements are regular (isotropic).)

  int spaceDim = spatialMeshTopology->getDimension();
  int spaceTimeDim = spaceDim + 1;

  MeshTopologyPtr rootSpatialTopology = spatialMeshTopology->getRootMeshTopology();
  MeshTopologyPtr spaceTimeTopology = Teuchos::rcp( new MeshTopology( spaceTimeDim ));

  /*
   This is something of a conceit, but it's nice if the vertex indices in the space-time mesh topology are
   in the following relationship to the spatialMeshTopology:

   If v is a vertexIndex in spatialMeshTopology and spatialMeshTopology has N vertices, then
   - (v,t0) has vertexIndex v in spaceTimeMeshTopology, and
   - (v,t1) has vertexIndex v+N in spaceTimeMeshTopology.
  */

  IndexType N = spatialMeshTopology->getEntityCount(0);
  for (int timeSubdivision=0; timeSubdivision<temporalDivisions; timeSubdivision++)
  {
    vector<double> spaceTimeVertex(spaceTimeDim);
    FieldContainer<double> timeValues(2,1);
    timeValues[0] = t0 + timeSubdivision * (t1-t0) / temporalDivisions;
    timeValues[1] = t0 + (timeSubdivision+1) * (t1-t0) / temporalDivisions;
    for (int i=0; i<timeValues.size(); i++)
    {
      for (IndexType vertexIndex=0; vertexIndex<N; vertexIndex++)
      {
        const vector<double> *spaceVertex = &spatialMeshTopology->getVertex(vertexIndex);
        for (int d=0; d<spaceDim; d++)
        {
          spaceTimeVertex[d] = (*spaceVertex)[d];
        }
        spaceTimeVertex[spaceDim] = timeValues(i,0);
        spaceTimeTopology->addVertex(spaceTimeVertex);
      }
    }
  }

  // for now, we only do refinements on the first temporal subdivision
  // later, we might want to enforce 1-irregularity, at least
  set<IndexType> cellIndices = rootSpatialTopology->getRootCellIndices();
  int tensorialDegree = 1;
  vector< FieldContainer<double> > componentNodes(2);
  FieldContainer<double> spatialCellNodes;
  FieldContainer<double> spaceTimeCellNodes;

  map<IndexType,IndexType> cellIDMap; // from space-time ID (in first temporal subdivision) to corresponding spatial ID

  for (int timeSubdivision=0; timeSubdivision<temporalDivisions; timeSubdivision++)
  {
    FieldContainer<double> timeValues(2,1);
    timeValues[0] = t0 + timeSubdivision * (t1-t0) / temporalDivisions;
    timeValues[1] = t0 + (timeSubdivision+1) * (t1-t0) / temporalDivisions;
    componentNodes[1] = timeValues;

    for (set<IndexType>::iterator cellIt = cellIndices.begin(); cellIt != cellIndices.end(); cellIt++)
    {
      IndexType cellIndex = *cellIt;
      CellPtr spatialCell = rootSpatialTopology->getCell(cellIndex);
      CellTopoPtr spaceTimeCellTopology = CellTopology::cellTopology(spatialCell->topology(), tensorialDegree);
      int vertexCount = spatialCell->topology()->getVertexCount();

      spatialCellNodes.resize(vertexCount,spaceDim);
      const vector<IndexType>* vertexIndices = &spatialCell->vertices();
      for (int vertex=0; vertex<vertexCount; vertex++)
      {
        IndexType vertexIndex = (*vertexIndices)[vertex];
        for (int i=0; i<spaceDim; i++)
        {
          spatialCellNodes(vertex,i) = spatialMeshTopology->getVertex(vertexIndex)[i];
        }
      }
      componentNodes[0] = spatialCellNodes;

      spaceTimeCellNodes.resize(spaceTimeCellTopology->getVertexCount(),spaceTimeDim);
      spaceTimeCellTopology->initializeNodes(componentNodes, spaceTimeCellNodes);

      CellPtr spaceTimeCell = spaceTimeTopology->addCell(spaceTimeCellTopology, spaceTimeCellNodes);
      if (timeSubdivision==0) cellIDMap[spaceTimeCell->cellIndex()] = cellIndex;
    }
  }

  bool noCellsToRefine = false;

  while (!noCellsToRefine)
  {
    noCellsToRefine = true;

    set<IndexType> activeSpaceTimeCellIndices = spaceTimeTopology->getActiveCellIndices();
    for (set<IndexType>::iterator cellIt = activeSpaceTimeCellIndices.begin(); cellIt != activeSpaceTimeCellIndices.end(); cellIt++)
    {
      IndexType spaceTimeCellIndex = *cellIt;
      if (cellIDMap.find(spaceTimeCellIndex) != cellIDMap.end())
      {
        IndexType spatialCellIndex = cellIDMap[spaceTimeCellIndex];
        CellPtr spatialCell = spatialMeshTopology->getCell(spatialCellIndex);
        if (spatialCell->isParent(spatialMeshTopology))
        {
          noCellsToRefine = false; // indicate we refined some on this pass...

          CellPtr spaceTimeCell = spaceTimeTopology->getCell(*cellIt);
          RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(spaceTimeCell->topology());
          spaceTimeTopology->refineCell(spaceTimeCellIndex, refPattern);

          vector<CellPtr> spatialChildren = spatialCell->children();
          for (int childOrdinal=0; childOrdinal<spatialChildren.size(); childOrdinal++)
          {
            CellPtr spatialChild = spatialChildren[childOrdinal];
            int vertexCount = spatialChild->topology()->getVertexCount();

            vector< vector<double> > childNodes(vertexCount);

            spatialCellNodes.resize(vertexCount,spaceDim);
            const vector<IndexType>* vertexIndices = &spatialChild->vertices();
            for (int vertex=0; vertex<vertexCount; vertex++)
            {
              IndexType vertexIndex = (*vertexIndices)[vertex];
              childNodes[vertex] = spatialMeshTopology->getVertex(vertexIndex);
              childNodes[vertex].push_back(t0);
            }

            CellPtr spaceTimeChild = spaceTimeTopology->findCellWithVertices(childNodes);
            cellIDMap[spaceTimeChild->cellIndex()] = spatialChild->cellIndex();
          }
        }
      }
    }
  }

  return spaceTimeTopology;
}

