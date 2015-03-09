//TODO: solution output
#include "HDF5Exporter.h"

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5

#include "CamelliaCellTools.h"
#include "MeshTools.h"
#include "GlobalDofAssignment.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#endif

#include <Epetra_SerialComm.h>
#include <EpetraExt_HDF5.h>

#include <sys/types.h>
#include <sys/stat.h>

#include "CellTopology.h"

using namespace Camellia;

HDF5Exporter::HDF5Exporter(MeshPtr mesh, string outputDirName, string outputDirSuperPath) : _mesh(mesh), _dirName(outputDirName),
  _dirSuperPath(outputDirSuperPath), _fieldXdmf("Xdmf"), _traceXdmf("Xdmf"),
  _fieldDomain("Domain"), _traceDomain("Domain"), _fieldGrids("Grid"), _traceGrids("Grid")
{
  int commRank = Teuchos::GlobalMPISession::getRank();

#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  if (commRank==0) {
    ostringstream dirPath;

    dirPath << _dirSuperPath << "/" << _dirName;
    int success = mkdir(dirPath.str().c_str(), S_IRWXU | S_IRWXG);

    dirPath.str("");
    dirPath << _dirSuperPath << "/" << _dirName << "/HDF5";
    success = mkdir(dirPath.str().c_str(), S_IRWXU | S_IRWXG);

    dirPath.str("");
    dirPath << _dirSuperPath << "/" << _dirName << "/XMF";
    success = mkdir(dirPath.str().c_str(), S_IRWXU | S_IRWXG);
  }

  Comm.Barrier(); // everyone should wait until rank 0 has created the directories

  if (commRank == 0)
  {
    _fieldXdmf.addAttribute("xmlns:xi", "http://www.w3.org/2003/XInclude");
    _traceXdmf.addAttribute("xmlns:xi", "http://www.w3.org/2003/XInclude");
    _fieldXdmf.addAttribute("Version", "2");
    _traceXdmf.addAttribute("Version", "2");
    _fieldXdmf.addChild(_fieldDomain);
    _traceXdmf.addChild(_traceDomain);
    _fieldDomain.addChild(_fieldGrids);
    _traceDomain.addChild(_traceGrids);
    _fieldGrids.addAttribute("Name", "Field Grids");
    _traceGrids.addAttribute("Name", "Trace Grids");
    _fieldGrids.addAttribute("GridType", "Collection");
    _traceGrids.addAttribute("GridType", "Collection");
    _fieldGrids.addAttribute("CollectionType", "Temporal");
    _traceGrids.addAttribute("CollectionType", "Temporal");
  }
}

HDF5Exporter::~HDF5Exporter()
{
}

void HDF5Exporter::exportSolution(SolutionPtr solution, double timeVal, unsigned int defaultNum1DPts, map<int, int> cellIDToNum1DPts, set<GlobalIndexType> cellIndices)
{
  VarFactory varFactory = _mesh->bilinearForm()->varFactory();
  
  vector<int> fieldTrialIDs = _mesh->bilinearForm()->trialVolumeIDs();
  vector<int> traceTrialIDs = _mesh->bilinearForm()->trialBoundaryIDs();
  vector<VarPtr> fieldVars;
  vector<VarPtr> traceVars;
  
  vector<FunctionPtr> fieldFunctions;
  vector<string> fieldFunctionNames;
  for (int i=0; i < fieldTrialIDs.size(); i++)
  {
    fieldVars.push_back(varFactory.trial(fieldTrialIDs[i]));
    FunctionPtr fieldFunction = Function::solution(fieldVars[i], solution);
    string fieldFunctionName = fieldVars[i]->name();
    fieldFunctions.push_back(fieldFunction);
    fieldFunctionNames.push_back(fieldFunctionName);
  }
  vector<FunctionPtr> traceFunctions;
  vector<string> traceFunctionNames;
  for (int i=0; i < traceTrialIDs.size(); i++)
  {
    traceVars.push_back(varFactory.trial(traceTrialIDs[i]));
    FunctionPtr traceFunction = Function::solution(traceVars[i], solution);
    string traceFunctionName = traceVars[i]->name();
    traceFunctions.push_back(traceFunction);
    traceFunctionNames.push_back(traceFunctionName);
  }
  exportFunction(fieldFunctions, fieldFunctionNames, timeVal, defaultNum1DPts, cellIDToNum1DPts, cellIndices);
  exportFunction(traceFunctions, traceFunctionNames, timeVal, defaultNum1DPts, cellIDToNum1DPts, cellIndices);
}

void HDF5Exporter::exportSolution(SolutionPtr solution, VarFactory varFactory, double timeVal, unsigned int defaultNum1DPts, map<int, int> cellIDToNum1DPts, set<GlobalIndexType> cellIndices)
{
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) cout << "NOTE: this version of HDF5Exporter::exportSolution() is deprecated.  Remove the VarFactory argument to get rid of this message.\n";
  this->exportSolution(solution,timeVal,defaultNum1DPts,cellIDToNum1DPts,cellIndices);
}

void HDF5Exporter::exportFunction(FunctionPtr function, string functionName, double timeVal, unsigned int defaultNum1DPts, map<int, int> cellIDToNum1DPts, set<GlobalIndexType> cellIndices)
{
  vector<FunctionPtr> functions;
  functions.push_back(function);
  vector<string> functionNames;
  functionNames.push_back(functionName);
  exportFunction(functions, functionNames, timeVal, defaultNum1DPts, cellIDToNum1DPts, cellIndices);
}

void HDF5Exporter::exportFunction(vector<FunctionPtr> functions, vector<string> functionNames, double timeVal, unsigned int defaultNum1DPts, map<int, int> cellIDToNum1DPts, set<GlobalIndexType> cellIndices)
{
  int commRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

  bool exportingBoundaryValues = functions[0]->boundaryValueOnly();

  XMLObject partitionCollection("Grid");
  if (commRank == 0)
  {
    if (!exportingBoundaryValues)
    {
      if (!_fieldTimeVals.count(timeVal))
      {
        _fieldGrids.addChild(partitionCollection);
        stringstream timeName;
        timeName << "Time" << timeVal;
        partitionCollection.addAttribute("Name", timeName.str());
        partitionCollection.addAttribute("GridType", "Collection");
        partitionCollection.addAttribute("CollectionType", "Spatial");
        XMLObject time("Time");
        partitionCollection.addChild(time);
        time.addAttribute("TimeType", "Single");
        time.addDouble("Value", timeVal);
        for (int p=0; p < numProcs; p++)
        {
          XMLObject xiinclude("xi:include");
          partitionCollection.addChild(xiinclude);
          stringstream partitionFileName;
          partitionFileName << "XMF/field" << "-part" << p << "-time" << timeVal << ".xmf";
          xiinclude.addAttribute("href", partitionFileName.str());
        }
      }
      else
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Field collection at timeVal already inserted");
    }
    else
    {
      if (!_traceTimeVals.count(timeVal))
      {
        _traceGrids.addChild(partitionCollection);
        stringstream timeName;
        timeName << "Time" << timeVal;
        partitionCollection.addAttribute("Name", timeName.str());
        partitionCollection.addAttribute("GridType", "Collection");
        partitionCollection.addAttribute("CollectionType", "Spatial");
        XMLObject time("Time");
        partitionCollection.addChild(time);
        time.addAttribute("TimeType", "Single");
        time.addDouble("Value", timeVal);
        for (int p=0; p < numProcs; p++)
        {
          XMLObject xiinclude("xi:include");
          partitionCollection.addChild(xiinclude);
          stringstream partitionFileName;
          partitionFileName << "XMF/trace" << "-part" << p << "-time" << timeVal << ".xmf";
          xiinclude.addAttribute("href", partitionFileName.str());
        }
      }
      else
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Trace collection at timeVal already inserted");
    }
  }
  ofstream gridFile;
  stringstream partitionFileName;
  if (!exportingBoundaryValues)
    partitionFileName << _dirSuperPath << "/" << _dirName << "/XMF/field" << "-part" << commRank << "-time" << timeVal << ".xmf";
  else
    partitionFileName << _dirSuperPath << "/" << _dirName << "/XMF/trace" << "-part" << commRank << "-time" << timeVal << ".xmf";
  gridFile.open(partitionFileName.str().c_str());
  XMLObject grid("Grid");
  stringstream gridName;
  gridName << "Time" << timeVal << "Partition" << commRank;
  grid.addAttribute("Name", gridName.str());
  grid.addAttribute("GridType", "Uniform");

  int nFcns = functions.size();

  if (defaultNum1DPts < 2)
    defaultNum1DPts = 2;

  int spaceDim = _mesh->getTopology()->getSpaceDim();

  for (int i=0; i < nFcns; i++)
    if (exportingBoundaryValues != functions[i]->boundaryValueOnly())
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can not export trace and field variables together");

  stringstream h5OutRel, h5OutFull, connOutRel2, connOutFull2;//, ptOutRel, ptOutFull;
  if (!exportingBoundaryValues)
  {
    h5OutRel << "HDF5/" << "field-part" << commRank << "-time" << timeVal << ".h5";
    h5OutFull << _dirSuperPath << "/" << _dirName << "/" << h5OutRel.str();
  }
  else
  {
    h5OutRel << "HDF5/" << "trace-part" << commRank << "-time" << timeVal << ".h5";
    h5OutFull << _dirSuperPath << "/" << _dirName << "/" << h5OutRel.str();
  }
  Epetra_SerialComm Comm;
  EpetraExt::HDF5 hdf5(Comm);
  hdf5.Create(h5OutFull.str());

  unsigned int total_vertices = 0;

  if (cellIndices.size()==0) cellIndices = _mesh->globalDofAssignment()->cellsInPartition(commRank);
  // Number of line elements in 1D mesh
  int numLines=0;
  // Number of triangle elements in 2D mesh
  int numTriangles=0;
  // Number of quad elements in 2D mesh
  int numQuads=0;
  // Number of tet elements in 3D mesh
  int numTets=0;
  // Number of wedge elements in 3D mesh
  int numWedges=0;
  // Number of hex elements in 3D mesh
  int numHexas=0;
  // Number of line subdivisions in 1D mesh
  int totalSubLines=0;
  // Number of triangle subdivisions in 2D mesh
  int totalSubTriangles=0;
  // Number of quad subdivisions in 2D mesh
  int totalSubQuads=0;
  // Number of tet subdivisions in 3D mesh
  int totalSubTets=0;
  // Number of wedge subdivisions in 3D mesh
  int totalSubWedges=0;
  // Number of hex subdivisions in 3D mesh
  int totalSubHexas=0;
  // Total number of subdivisions
  int totalSubcells=0;
  // Total number of points needed to construct subdivisions
  int totalPts=0;
  // Total number of boundary points in 1D mesh
  int totalBoundaryPts=0;
  // Total number of boundary lines in 2D mesh
  int totalBoundaryLines=0;
  // Total number of boundary quads in 3D mesh
  int totalBoundaryQuads=0;
  // Total number of boundary tris in 3D mesh
  int totalBoundaryTriangles=0;
  for (set<GlobalIndexType>::iterator cellIt = cellIndices.begin(); cellIt != cellIndices.end(); cellIt++) {
    CellPtr cell = _mesh->getTopology()->getCell(*cellIt);
    if (!cellIDToNum1DPts[cell->cellIndex()] || cellIDToNum1DPts[cell->cellIndex()] < 2)
      cellIDToNum1DPts[cell->cellIndex()] = defaultNum1DPts;
    int num1DPts = cellIDToNum1DPts[cell->cellIndex()];
    
    unsigned baseCellTopoKey = cell->topology()->getKey().first;
    unsigned cellTopoKey;
    // Designate effective topologies for tensor meshes
    switch (baseCellTopoKey)
    {
      case shards::Node::key:
        switch (cell->topology()->getTensorialDegree())
        {
          case 0:
          cellTopoKey = shards::Node::key;
          break;
          case 1:
          cellTopoKey = shards::Line<2>::key;
          break;
          case 2:
          cellTopoKey = shards::Quadrilateral<4>::key;
          break;
          case 3:
          cellTopoKey = shards::Hexahedron<8>::key;
          break;
          default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
        }
        break;
      case shards::Line<2>::key:
        switch (cell->topology()->getTensorialDegree())
        {
          case 0:
          cellTopoKey = shards::Line<2>::key;
          break;
          case 1:
          cellTopoKey = shards::Quadrilateral<4>::key;
          break;
          case 2:
          cellTopoKey = shards::Hexahedron<8>::key;
          break;
          default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
        }
        break;
      case shards::Quadrilateral<4>::key:
        switch (cell->topology()->getTensorialDegree())
        {
          case 0:
          cellTopoKey = shards::Quadrilateral<4>::key;
          break;
          case 1:
          cellTopoKey = shards::Hexahedron<8>::key;
          break;
          default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
        }
        break;
      case shards::Triangle<3>::key:
        switch (cell->topology()->getTensorialDegree())
        {
          case 0:
          cellTopoKey = shards::Triangle<3>::key;
          break;
          case 1:
          cellTopoKey = shards::Wedge<6>::key;
          break;
          default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
        }
        break;
      case shards::Tetrahedron<4>::key:
        switch (cell->topology()->getTensorialDegree())
        {
          case 0:
          cellTopoKey = shards::Tetrahedron<4>::key;
          break;
          default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
        }
        break;
      case shards::Wedge<6>::key:
        switch (cell->topology()->getTensorialDegree())
        {
          case 0:
          cellTopoKey = shards::Wedge<6>::key;
          break;
          default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
        }
        break;
      case shards::Hexahedron<8>::key:
        switch (cell->topology()->getTensorialDegree())
        {
          case 0:
          cellTopoKey = shards::Hexahedron<8>::key;
          break;
          default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
        }
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
    }   

    switch (cellTopoKey)
    {
      case shards::Line<2>::key:
      numLines++;
      if (!exportingBoundaryValues)
      {
        totalSubLines += num1DPts-1;
        totalPts += num1DPts;
      }
      else
      {
        totalBoundaryPts += 2;
        totalPts += 2;
      }
      break;
      case shards::Triangle<3>::key:
      numTriangles++;
      if (!exportingBoundaryValues)
      {
        totalSubTriangles += (num1DPts-1)*(num1DPts-1);
        totalPts += num1DPts*(num1DPts+1)/2;
      }
      else
      {
        totalBoundaryLines += 3;
        totalSubLines += 3*(num1DPts-1);
        totalPts += 3*num1DPts;
      }
      break;
      case shards::Quadrilateral<4>::key:
      numQuads++;
      if (!exportingBoundaryValues)
      {
        totalSubQuads += (num1DPts-1)*(num1DPts-1);
        totalPts += num1DPts*num1DPts;
      }
      else
      {
        totalBoundaryLines += 4;
        totalSubLines += 4*(num1DPts-1);
        totalPts += 4*num1DPts;
      }
      break;
      case shards::Tetrahedron<4>::key:
      numTets++;
      if (!exportingBoundaryValues)
      {
        totalSubTets += 1;
        totalPts += 4;
      }
      else
      {
        totalBoundaryTriangles += 4;
        totalSubTriangles += 4*(num1DPts-1)*(num1DPts-1);
        totalPts += 4*num1DPts*(num1DPts+1)/2;
      }
      break;
      case shards::Wedge<6>::key:
      numWedges++;
      if (!exportingBoundaryValues)
      {
        totalSubWedges += (num1DPts-1)*(num1DPts-1)*(num1DPts-1);
        totalPts += num1DPts*num1DPts*(num1DPts+1)/2;
      }
      else
      {
        totalBoundaryQuads += 3;
        totalBoundaryTriangles += 2;
        totalSubQuads += 3*(num1DPts-1)*(num1DPts-1);
        totalSubTriangles += 2*(num1DPts-1)*(num1DPts-1);
        totalPts += 2*num1DPts*(num1DPts+1)/2 + 3*num1DPts*num1DPts;
      }
      break;
      case shards::Hexahedron<8>::key:
      numHexas++;
      if (!exportingBoundaryValues)
      {
        totalSubHexas += (num1DPts-1)*(num1DPts-1)*(num1DPts-1);
        totalPts += num1DPts*num1DPts*num1DPts;
      }
      else
      {
        totalBoundaryQuads += 6;
        totalSubQuads += 6*(num1DPts-1)*(num1DPts-1);
        totalPts += 6*num1DPts*num1DPts;
      }
      break;
    }

  }
  totalSubcells = totalBoundaryPts + totalSubLines + totalSubTriangles + totalSubQuads + totalSubTets + totalSubWedges + totalSubHexas;

  // Topology
  XMLObject topology("Topology");
  grid.addChild(topology);
  topology.addAttribute("TopologyType", "Mixed");
  if (!exportingBoundaryValues)
  {
    if (spaceDim == 1)
      topology.addInt("Dimensions", numLines);
    else
      topology.addInt("Dimensions", totalSubcells);
  }
  else
  {
    if (spaceDim == 1)
      topology.addInt("Dimensions", totalBoundaryPts);
    else if (spaceDim == 2)
      topology.addInt("Dimensions", totalBoundaryLines);
    else if (spaceDim == 3)
      topology.addInt("Dimensions", totalSubQuads + totalSubTriangles);
  }
  hsize_t connDimsf;
  if (!exportingBoundaryValues)
  {
    if (spaceDim == 1)
      connDimsf = 2*numLines+totalPts;
    else
      connDimsf = totalSubcells + 3*totalSubTriangles + 4*totalSubQuads + 4*totalSubTets + 6*totalSubWedges + 8*totalSubHexas;
  }
  else
  {
    if (spaceDim == 1)
      connDimsf = 3*totalBoundaryPts;
    if (spaceDim == 2)
      connDimsf = 2*totalBoundaryLines + totalPts;
    if (spaceDim == 3)
      connDimsf = 5*totalSubQuads + 4*totalSubTriangles;
  }
  int connArray[connDimsf];
  XMLObject topoDataItem("DataItem");
  topology.addChild(topoDataItem);
  topoDataItem.addAttribute("ItemType", "Uniform");
  topoDataItem.addAttribute("Format", "HDF");
  topoDataItem.addAttribute("NumberType", "Int");
  topoDataItem.addAttribute("Precision", "4");
  topoDataItem.addInt("Dimensions", connDimsf);
  stringstream connOutRel;
  connOutRel << h5OutRel.str() << ":/Data/Conns";
  topoDataItem.addContent(connOutRel.str());

  // Geometry
  XMLObject geometry("Geometry");
  grid.addChild(geometry);
  if (spaceDim < 3)
    geometry.addAttribute("GeometryType", "XY");
  else if (spaceDim == 3)
    geometry.addAttribute("GeometryType", "XYZ");
  hsize_t ptDimsf;
  if (spaceDim == 1)
    ptDimsf = 2 * totalPts;
  else
    ptDimsf = spaceDim * totalPts;
  double ptArray[ptDimsf];

  XMLObject geoDataItem("DataItem");
  geometry.addChild(geoDataItem);
  geoDataItem.addAttribute("ItemType", "Uniform");
  geoDataItem.addAttribute("Format", "HDF");
  geoDataItem.addAttribute("NumberType", "Float");
  geoDataItem.addAttribute("Precision", "8");
  geoDataItem.addInt("Dimensions", ptDimsf);
  stringstream ptOutRel;
  ptOutRel << h5OutRel.str() << ":/Data/Points";
  geoDataItem.addContent(ptOutRel.str());

  // Node Data
  vector<XMLObject> vals;
  for (int i=0; i<nFcns; i++)
  {
    vals.push_back( XMLObject("Attribute") );
    grid.addChild(vals[i]);
    vals[i].addAttribute("Name", functionNames[i].c_str());
    vals[i].addAttribute("Center", "Node");
  }

  vector< vector<double> > valArrays;
  valArrays.resize(nFcns);
  hsize_t valDimsf[nFcns];
  int numFcnComponents[nFcns];
  for (int i = 0; i < nFcns; i++)
  {
    if (functions[i]->rank() == 0)
      numFcnComponents[i] = 1;
    else if (functions[i]->rank() == 1)
      numFcnComponents[i] = spaceDim;
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled function rank");
    }
    if (numFcnComponents[i] == 1)
    {
      vals[i].addAttribute("AttributeType", "Scalar");
    }
    else
    {
      vals[i].addAttribute("AttributeType", "Vector");
    }
    if (numFcnComponents[i] == 1)
    {
      valDimsf[i] = totalPts;
    }
    else
    {
      valDimsf[i] = 3*totalPts;
    }
    valArrays[i].resize(valDimsf[i], 0);
    XMLObject valDataItem("DataItem");
    vals[i].addChild(valDataItem);
    valDataItem.addAttribute("ItemType", "Uniform");
    valDataItem.addAttribute("Format", "HDF");
    valDataItem.addAttribute("NumberType", "Float");
    valDataItem.addAttribute("Precision", "8");
    valDataItem.addInt("Dimensions", valDimsf[i]);
    stringstream valOutRel;
    valOutRel << h5OutRel.str() << ":/Data/" << functionNames[i];
    valDataItem.addContent(valOutRel.str());
  }

  int connIndex = 0;
  int ptIndex = 0;
  int valIndex[nFcns];
  for (int i = 0; i < nFcns; i++)
    valIndex[i] = 0;

  for (set<GlobalIndexType>::iterator cellIt = cellIndices.begin(); cellIt != cellIndices.end(); cellIt++)
  {
    GlobalIndexType cellIndex = *cellIt;
    CellPtr cell = _mesh->getTopology()->getCell(cellIndex);

    FieldContainer<double> physicalCellNodes = _mesh->getTopology()->physicalCellNodesForCell(cellIndex);

    CellTopoPtr cellTopoPtr = cell->topology();
    int num1DPts = cellIDToNum1DPts[cell->cellIndex()];
    int numPoints = 0;

    if (physicalCellNodes.rank() == 2)
      physicalCellNodes.resize(1,physicalCellNodes.dimension(0), physicalCellNodes.dimension(1));
    bool createSideCache = functions[0]->boundaryValueOnly();

    BasisCachePtr volumeBasisCache = Teuchos::rcp( new BasisCache(cellTopoPtr, 1, createSideCache) );
    volumeBasisCache->setPhysicalCellNodes(physicalCellNodes, vector<GlobalIndexType>(1,cellIndex), createSideCache);

    int numSides = createSideCache ? cellTopoPtr->getSideCount() : 1;

    int sideDim = spaceDim - 1;

    for (int sideOrdinal = 0; sideOrdinal < numSides; sideOrdinal++)
    {
      CellTopoPtr topo = createSideCache ? cellTopoPtr->getSubcell(sideDim, sideOrdinal) : cellTopoPtr;
      
      unsigned baseCellTopoKey = topo->getKey().first;
      unsigned cellTopoKey;
      // Designate effective topologies for tensor meshes
      switch (baseCellTopoKey)
      {
        case shards::Node::key:
          switch (cell->topology()->getTensorialDegree())
          {
            case 0:
            cellTopoKey = shards::Node::key;
            break;
            case 1:
            cellTopoKey = shards::Line<2>::key;
            break;
            case 2:
            cellTopoKey = shards::Quadrilateral<4>::key;
            break;
            case 3:
            cellTopoKey = shards::Hexahedron<8>::key;
            break;
            default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
          }
          break;
        case shards::Line<2>::key:
          switch (cell->topology()->getTensorialDegree())
          {
            case 0:
            cellTopoKey = shards::Line<2>::key;
            break;
            case 1:
            if (!exportingBoundaryValues || (exportingBoundaryValues && spaceDim == 3))
              cellTopoKey = shards::Quadrilateral<4>::key;
            else if (spaceDim == 2)
              cellTopoKey = shards::Line<2>::key;
            // cellTopoKey = shards::Quadrilateral<4>::key;
            break;
            case 2:
            cellTopoKey = shards::Hexahedron<8>::key;
            break;
            default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
          }
          break;
        case shards::Triangle<3>::key:
          switch (cell->topology()->getTensorialDegree())
          {
            case 0:
            cellTopoKey = shards::Triangle<3>::key;
            break;
            case 1:
            if (!exportingBoundaryValues)
              cellTopoKey = shards::Wedge<6>::key;
            else
              cellTopoKey = shards::Triangle<3>::key;
            break;
            default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
          }
          break;
        case shards::Quadrilateral<4>::key:
          switch (cell->topology()->getTensorialDegree())
          {
            case 0:
            cellTopoKey = shards::Quadrilateral<4>::key;
            break;
            case 1:
            if (!exportingBoundaryValues)
              cellTopoKey = shards::Hexahedron<8>::key;
            else
              cellTopoKey = shards::Quadrilateral<4>::key;
            break;
            default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
          }
          break;
        case shards::Tetrahedron<4>::key:
          switch (cell->topology()->getTensorialDegree())
          {
            case 0:
            cellTopoKey = shards::Tetrahedron<4>::key;
            break;
            default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
          }
          break;
        case shards::Wedge<6>::key:
          switch (cell->topology()->getTensorialDegree())
          {
            case 0:
            cellTopoKey = shards::Wedge<6>::key;
            break;
            default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
          }
          break;
        case shards::Hexahedron<8>::key:
          switch (cell->topology()->getTensorialDegree())
          {
            case 0:
            cellTopoKey = shards::Hexahedron<8>::key;
            break;
            default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensor degree not supported for this spatial topology");
          }
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
      }

      BasisCachePtr basisCache = createSideCache ? volumeBasisCache->getSideBasisCache(sideOrdinal) : volumeBasisCache;
      basisCache->setMesh(_mesh);
      FunctionPtr transformFxn = _mesh->getTransformationFunction();
      if (transformFxn.get())
        basisCache->setTransformationFunction(transformFxn);

      unsigned domainDim = createSideCache ? sideDim : spaceDim;

      switch (cellTopoKey)
      {
        case shards::Node::key:
          numPoints = 1;
          break;
        case shards::Line<2>::key:
          numPoints = num1DPts;
          break;
        case shards::Triangle<3>::key:
          numPoints = num1DPts*(num1DPts+1)/2;
          break;
        case shards::Quadrilateral<4>::key:
          numPoints = num1DPts*num1DPts;
          break;
        case shards::Tetrahedron<4>::key:
          numPoints = 4;
          break;
        case shards::Wedge<6>::key:
          numPoints = num1DPts*num1DPts*(num1DPts+1)/2;
          break;
        case shards::Hexahedron<8>::key:
          numPoints = num1DPts*num1DPts*num1DPts;
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
      }
      
      FieldContainer<double> refPoints(numPoints,domainDim);
      if (domainDim == 0)
        refPoints.resize(numPoints);
      switch (cellTopoKey)
      {
        case shards::Node::key:
          {
            refPoints(0,0) = 0;
          }
          break;
        case shards::Line<2>::key:
          {
            for (int i=0; i < num1DPts; i++)
            {
              int pointIndex = i;
              double x = -1.0 + 2.0*(double(i)/double(num1DPts-1));
              refPoints(pointIndex,0) = x;
            }
          }
          break;
        case shards::Triangle<3>::key:
          {
            int pointIndex = 0;
            for (int j = 0; j < num1DPts; j++)
              for (int i=0; i < num1DPts-j; i++)
              {
                double x = (double(i)/double(num1DPts-1));
                double y = (double(j)/double(num1DPts-1));
                refPoints(pointIndex,0) = x;
                refPoints(pointIndex,1) = y;
                pointIndex++;
              }
          }
          break;
        case shards::Quadrilateral<4>::key:
          {
            for (int j = 0; j < num1DPts; j++)
              for (int i=0; i < num1DPts; i++)
              {
                int pointIndex = j*num1DPts + i;
                double x = -1.0 + 2.0*(double(i)/double(num1DPts-1));
                double y = -1.0 + 2.0*(double(j)/double(num1DPts-1));
                refPoints(pointIndex,0) = x;
                refPoints(pointIndex,1) = y;
              }
          }
          break;
        case shards::Tetrahedron<4>::key:
          {
            refPoints(0,0) = 0;
            refPoints(0,1) = 0;
            refPoints(0,2) = 0;
            refPoints(1,0) = 1;
            refPoints(1,1) = 0;
            refPoints(1,2) = 0;
            refPoints(2,0) = 0;
            refPoints(2,1) = 1;
            refPoints(2,2) = 0;
            refPoints(3,0) = 0;
            refPoints(3,1) = 0;
            refPoints(3,2) = 1;
          }
          break;
        case shards::Wedge<6>::key:
          {
            int pointIndex = 0;
            for (int k = 0; k < num1DPts; k++)
              for (int j = 0; j < num1DPts; j++)
                for (int i=0; i < num1DPts-j; i++)
                {
                  double x = (double(i)/double(num1DPts-1));
                  double y = (double(j)/double(num1DPts-1));
                  double z = -1.0 + 2.0*(double(k)/double(num1DPts-1));
                  refPoints(pointIndex,0) = x;
                  refPoints(pointIndex,1) = y;
                  refPoints(pointIndex,2) = z;
                  pointIndex++;
                }
          }
          break;
        case shards::Hexahedron<8>::key:
          {
            for (int k = 0; k < num1DPts; k++)
              for (int j=0; j < num1DPts; j++)
                for (int i=0; i < num1DPts; i++)
                {
                  int pointIndex = k*num1DPts*num1DPts + j*num1DPts + i;
                  double x = -1.0 + 2.0*(double(i)/double(num1DPts-1));
                  double y = -1.0 + 2.0*(double(j)/double(num1DPts-1));
                  double z = -1.0 + 2.0*(double(k)/double(num1DPts-1));
                  refPoints(pointIndex,0) = x;
                  refPoints(pointIndex,1) = y;
                  refPoints(pointIndex,2) = z;
                }
              }
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
      }

      basisCache->setRefCellPoints(refPoints);
      const FieldContainer<double> *physicalPoints = &basisCache->getPhysicalCubaturePoints();
      // Function Values
      std::vector< FieldContainer<double> > computedValues(nFcns);
      for (int i = 0; i < nFcns; i++)
      {
        if (functions[i]->rank() == 0)
          computedValues[i].resize(1, numPoints);
        else if (functions[i]->rank() == 1)
          computedValues[i].resize(1, numPoints, spaceDim);
        else
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled function rank");
        functions[i]->values(computedValues[i], basisCache);
      }

      int subcellStartIndex = total_vertices;
      switch (cellTopoKey)
      {
        case shards::Node::key:
        {
          connArray[connIndex] = 1;
          connIndex++;
          connArray[connIndex] = 1;
          connIndex++;
          int ind1 = total_vertices;
          connArray[connIndex] = ind1;
          connIndex++;
        }
        break;
        case shards::Line<2>::key:
        {
          connArray[connIndex] = 2;
          connIndex++;
          connArray[connIndex] = num1DPts;
          connIndex++;
          for (int i=0; i < num1DPts; i++)
          {
            int ind1 = total_vertices + i;
            connArray[connIndex] = ind1;
            connIndex++;
          }
        }
        break;
        case shards::Triangle<3>::key:
        {
          for (int j=0; j < num1DPts-1; j++)
          {
            for (int i=0; i < num1DPts-1-j; i++)
            {
              int ind1 = subcellStartIndex;
              int ind2 = ind1 + 1;
              int ind3 = ind1 + num1DPts-j;
              connArray[connIndex] = 4;
              connIndex++;
              connArray[connIndex] = ind1;
              connIndex++;
              connArray[connIndex] = ind2;
              connIndex++;
              connArray[connIndex] = ind3;
              connIndex++;

              if (i < num1DPts-2-j)
              {
                int ind1 = subcellStartIndex+1;
                int ind2 = ind1 + num1DPts - j;
                int ind3 = ind1 + num1DPts -j - 1;
                connArray[connIndex] = 4;
                connIndex++;
                connArray[connIndex] = ind1;
                connIndex++;
                connArray[connIndex] = ind2;
                connIndex++;
                connArray[connIndex] = ind3;
                connIndex++;
              }

              subcellStartIndex++;
            }
            subcellStartIndex++;
          }
        }
        break;
        case shards::Quadrilateral<4>::key:
        {
          for (int j=0; j < num1DPts-1; j++)
          {
            for (int i=0; i < num1DPts-1; i++)
            {
              int ind1 = total_vertices + i + j*num1DPts;
              int ind2 = ind1 + 1;
              int ind3 = ind2 + num1DPts;
              int ind4 = ind1 + num1DPts;
              connArray[connIndex] = 5;
              connIndex++;
              connArray[connIndex] = ind1;
              connIndex++;
              connArray[connIndex] = ind2;
              connIndex++;
              connArray[connIndex] = ind3;
              connIndex++;
              connArray[connIndex] = ind4;
              connIndex++;
            }
          }
        }
        break;
        case shards::Tetrahedron<4>::key:
        {
          int ind1 = total_vertices;
          int ind2 = ind1 + 1;
          int ind3 = ind1 + 2;
          int ind4 = ind1 + 3;
          connArray[connIndex] = 6;
          connIndex++;
          connArray[connIndex] = ind1;
          connIndex++;
          connArray[connIndex] = ind2;
          connIndex++;
          connArray[connIndex] = ind3;
          connIndex++;
          connArray[connIndex] = ind4;
          connIndex++;
        }
        break;
        case shards::Wedge<6>::key:
        {
          for (int k=0; k < num1DPts-1; k++)
          {
            for (int j=0; j < num1DPts-1; j++)
            {
              for (int i=0; i < num1DPts-1-j; i++)
              {
                int ind1 = subcellStartIndex;
                int ind2 = ind1 + 1;
                int ind3 = ind1 + num1DPts-j;
                int ind4 = ind1 + num1DPts*(num1DPts+1)/2;
                int ind5 = ind4 + 1;
                int ind6 = ind4 + num1DPts-j;
                connArray[connIndex] = 8;
                connIndex++;
                connArray[connIndex] = ind1;
                connIndex++;
                connArray[connIndex] = ind2;
                connIndex++;
                connArray[connIndex] = ind3;
                connIndex++;
                connArray[connIndex] = ind4;
                connIndex++;
                connArray[connIndex] = ind5;
                connIndex++;
                connArray[connIndex] = ind6;
                connIndex++;

                if (i < num1DPts-2-j)
                {
                  int ind1 = subcellStartIndex+1;
                  int ind2 = ind1 + num1DPts - j;
                  int ind3 = ind1 + num1DPts -j - 1;
                  int ind4 = ind1 + num1DPts*(num1DPts+1)/2;
                  int ind5 = ind4 + num1DPts - j;
                  int ind6 = ind4 + num1DPts -j - 1;
                  connArray[connIndex] = 8;
                  connIndex++;
                  connArray[connIndex] = ind1;
                  connIndex++;
                  connArray[connIndex] = ind2;
                  connIndex++;
                  connArray[connIndex] = ind3;
                  connIndex++;
                  connArray[connIndex] = ind4;
                  connIndex++;
                  connArray[connIndex] = ind5;
                  connIndex++;
                  connArray[connIndex] = ind6;
                  connIndex++;
                }

                subcellStartIndex++;
              }
              subcellStartIndex++;
            }
            subcellStartIndex++;
          }
        }
        break;
        case shards::Hexahedron<8>::key:
        {
          for (int k=0; k < num1DPts-1; k++)
          {
            for (int j=0; j < num1DPts-1; j++)
            {
              for (int i=0; i < num1DPts-1; i++)
              {
                int ind1 = total_vertices + i + j*num1DPts + k*num1DPts*num1DPts;
                int ind2 = ind1 + 1;
                int ind3 = ind2 + num1DPts;
                int ind4 = ind1 + num1DPts;
                int ind5 = ind1 + num1DPts*num1DPts;
                int ind6 = ind5 + 1;
                int ind7 = ind6 + num1DPts;
                int ind8 = ind5 + num1DPts;
                connArray[connIndex] = 9;
                connIndex++;
                connArray[connIndex] = ind1;
                connIndex++;
                connArray[connIndex] = ind2;
                connIndex++;
                connArray[connIndex] = ind3;
                connIndex++;
                connArray[connIndex] = ind4;
                connIndex++;
                connArray[connIndex] = ind5;
                connIndex++;
                connArray[connIndex] = ind6;
                connIndex++;
                connArray[connIndex] = ind7;
                connIndex++;
                connArray[connIndex] = ind8;
                connIndex++;
              }
            }
          }
        }
        break;
        default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
      }
      for (int pointIndex = 0; pointIndex < numPoints; pointIndex++)
      {
        if (spaceDim == 1)
        {
          ptArray[ptIndex] = (*physicalPoints)(0, pointIndex, 0);
          ptIndex++;
          ptArray[ptIndex] = 0;
          ptIndex++;
        }
        else if (spaceDim == 2)
        {
          ptArray[ptIndex] = (*physicalPoints)(0, pointIndex, 0);
          ptIndex++;
          ptArray[ptIndex] = (*physicalPoints)(0, pointIndex, 1);
          ptIndex++;
        }
        else
        {
          ptArray[ptIndex] = (*physicalPoints)(0, pointIndex, 0);
          ptIndex++;
          ptArray[ptIndex] = (*physicalPoints)(0, pointIndex, 1);
          ptIndex++;
          ptArray[ptIndex] = (*physicalPoints)(0, pointIndex, 2);
          ptIndex++;
        }
        for (int i = 0; i < nFcns; i++)
        {

          switch(numFcnComponents[i])
          {
            case 1:
            valArrays[i][valIndex[i]] = computedValues[i](0, pointIndex);
            valIndex[i]++;
            break;
            case 2:
            valArrays[i][valIndex[i]] = computedValues[i](0, pointIndex, 0);
            valIndex[i]++;
            valArrays[i][valIndex[i]] = computedValues[i](0, pointIndex, 1);
            valIndex[i]++;
            valArrays[i][valIndex[i]] = 0;
            valIndex[i]++;
            break;
            case 3:
            valArrays[i][valIndex[i]] = computedValues[i](0, pointIndex, 0);
            valIndex[i]++;
            valArrays[i][valIndex[i]] = computedValues[i](0, pointIndex, 1);
            valIndex[i]++;
            valArrays[i][valIndex[i]] = computedValues[i](0, pointIndex, 2);
            valIndex[i]++;
            break;
            default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported number of components");
          }
        }
        total_vertices++;
      }
    }
  }
  if (connDimsf > 0)
  {
    hdf5.Write("Data", "Conns", H5T_NATIVE_INT, connDimsf, &connArray[0]);
    hdf5.Write("Data", "Points", H5T_NATIVE_DOUBLE, ptDimsf, &ptArray[0]);
    for (int i = 0; i < nFcns; i++)
      hdf5.Write("Data", functionNames[i], H5T_NATIVE_DOUBLE, valDimsf[i], &valArrays[i][0]);
  }
  hdf5.Close();

  gridFile << grid.toString();
  gridFile.close();

  if (commRank == 0)
  {
    if (_fieldGrids.numChildren() > 0)
    {
      ofstream xmfFieldFile;
      xmfFieldFile.open((_dirSuperPath + "/" + _dirName+"/"+_dirName+"-field.xmf").c_str());
      xmfFieldFile << "<?xml version=\"1.0\" ?>" << endl
      << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>" << endl;
      xmfFieldFile << _fieldXdmf.toString();
      xmfFieldFile.close();
    }
    if (_traceGrids.numChildren() > 0)
    {
      ofstream xmfTraceFile;
      xmfTraceFile.open((_dirSuperPath + "/" + _dirName+"/"+_dirName+"-trace.xmf").c_str());
      xmfTraceFile << "<?xml version=\"1.0\" ?>" << endl
      << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>" << endl;
      xmfTraceFile << _traceXdmf.toString();
      xmfTraceFile.close();
    }
  }
}

void HDF5Exporter::exportTimeSlab(FunctionPtr function, string functionName, double tInit, double tFinal, unsigned int numSlices, unsigned int sliceH1Order, unsigned int defaultNum1DPts)
{
  vector<FunctionPtr> functions;
  functions.push_back(function);
  vector<string> functionNames;
  functionNames.push_back(functionName);
  exportTimeSlab(functions, functionNames, tInit, tFinal, numSlices, sliceH1Order, defaultNum1DPts);
}

void HDF5Exporter::exportTimeSlab(vector<FunctionPtr> functions, vector<string> functionNames, double tInit, double tFinal, unsigned int numSlices, unsigned int sliceH1Order, unsigned int defaultNum1DPts)
{
  HDF5Exporter exporter(_mesh, _dirName, _dirSuperPath);
  for (int slice=0; slice < numSlices; slice++)
  {
    double sliceTime = tInit + slice*(tFinal-tInit)/(numSlices-1);
    map<GlobalIndexType,GlobalIndexType> cellMap;
    MeshPtr meshSlice = MeshTools::timeSliceMesh(_mesh, sliceTime, cellMap, sliceH1Order);
    vector<FunctionPtr> functionSlices;
    for (vector<FunctionPtr>::iterator fcnIt = functions.begin(); fcnIt != functions.end(); ++fcnIt)
    {
      FunctionPtr functionSlice = MeshTools::timeSliceFunction(_mesh, cellMap, *fcnIt, sliceTime);
      functionSlices.push_back(functionSlice);
    }
    exporter.setMesh(meshSlice);
    exporter.exportFunction(functionSlices, functionNames, sliceTime, defaultNum1DPts);
  }
}

void HDF5Exporter::exportFunction(string superDirectory, string functionName, FunctionPtr function, MeshPtr mesh) {
  HDF5Exporter exporter(mesh, functionName, superDirectory);
  exporter.exportFunction(function);
}

void HDF5Exporter::exportSolution(std::string superDirectory, std::string solnName, SolutionPtr solution) {
  MeshPtr mesh = solution->mesh();
  HDF5Exporter exporter(mesh, solnName, superDirectory);
  exporter.exportSolution(solution, mesh->bilinearForm()->varFactory());
}

map<int,int> cellIDToSubdivision(MeshPtr mesh, unsigned int subdivisionFactor, set<GlobalIndexType> cellIndices)
{
  if (cellIndices.size()==0) cellIndices = mesh->getTopology()->getActiveCellIndices();
  map<int,int> cellIDToPolyOrder;
  for (set<GlobalIndexType>::iterator cellIt = cellIndices.begin(); cellIt != cellIndices.end(); cellIt++)
  {
    cellIDToPolyOrder[*cellIt] =  (subdivisionFactor*(mesh->cellPolyOrder(*cellIt)-2)+1);
  }
  return cellIDToPolyOrder;
}

// end HAVE_EPETRAEXT_HDF5 include guard
#endif
