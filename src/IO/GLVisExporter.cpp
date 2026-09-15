//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "GLVisExporter.h"

#include "BasisCache.h"
#include "Cell.h"
#include "CellTopology.h"
#include "Function.h"
#include "GlobalDofAssignment.h"
#include "MeshTopology.h"
#include "MPIWrapper.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>

using namespace Camellia;
using namespace std;

static double legendre(int k, double x)
{
  double p0 = 1.0, p1 = x;
  for (int i = 2; i <= k; i++)
  {
    double p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i;
    p0 = p1;
    p1 = p2;
  }
  return (k > 0) ? p1 : p0;
}

static double dLegendre(int k, double x)
{
  if (abs(abs(x) - 1.0) < 1e-15)
  {
    return k * (k + 1) / 2.0 * ((x > 0) ? 1.0 : pow(-1.0, k + 1));
  }
  return k * (legendre(k - 1, x) - x * legendre(k, x)) / (1 - x * x);
}

static double ddLegendre(int k, double x)
{
  return (2 * x * dLegendre(k, x) - k * (k + 1) * legendre(k, x)) / (1 - x * x);
}

static vector<double> gaussLobatto(int n)
{
  if (n == 1) return {0.5};
  if (n == 2) return {0.0, 1.0};
  int k = n - 1;
  vector<double> xs = {-1.0};
  for (int i = 1; i < k; i++)
  {
    double x = -cos(M_PI * i / k);
    for (int iter = 0; iter < 100; iter++)
    {
      double dx = -dLegendre(k, x) / ddLegendre(k, x);
      x += dx;
      if (abs(dx) < 1e-16) break;
    }
    xs.push_back(x);
  }
  xs.push_back(1.0);
  for (double &x : xs) x = (x + 1.0) / 2.0;
  return xs;
}

static bool pathExists(const string &path)
{
  struct stat st;
  return stat(path.c_str(), &st) == 0;
}

static bool directoryExists(const string &path)
{
  struct stat st;
  return (stat(path.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
}

static string fullPath(const string &superPath, const string &dirName)
{
  return superPath + "/" + dirName;
}

static string firstAvailableDirName(const string &superPath, const string &requestedDirName)
{
  string candidate = requestedDirName;
  int suffix = 2;
  while (pathExists(fullPath(superPath, candidate)))
  {
    candidate = requestedDirName + "-" + to_string(suffix++);
  }
  return candidate;
}

static vector<int> mfemNodePerm(CellTopoPtr topo)
{
  int d = topo->getDimension();
  unsigned baseKey = topo->getKey().first;
  if (d == 1) return {0, 1};
  if (d == 2)
  {
    if (baseKey == shards::Quadrilateral<4>::key) return {0, 1, 2, 3};
    if ((baseKey == shards::Line<2>::key) || (baseKey == shards::Node::key)) return {0, 1, 3, 2};
  }
  else if (d == 3)
  {
    if ((baseKey == shards::Hexahedron<8>::key) || (baseKey == shards::Quadrilateral<4>::key)) return {0, 1, 2, 3, 4, 5, 6, 7};
    if ((baseKey == shards::Line<2>::key) || (baseKey == shards::Node::key)) return {0, 1, 3, 2, 4, 5, 7, 6};
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "GLVisExporter supports segment, quad, and hex topologies only");
}

static bool cellHasCurvedEdges(MeshTopologyViewPtr meshTopo, GlobalIndexType cellID)
{
  const MeshTopology *concreteTopo = dynamic_cast<const MeshTopology *>(meshTopo.get());
  if ((concreteTopo != NULL) && concreteTopo->isValidCellIndex(cellID))
  {
    return concreteTopo->cellHasCurvedEdges(cellID);
  }
  const MeshTopology *baseTopo = meshTopo->baseMeshTopology();
  if ((baseTopo != NULL) && baseTopo->isValidCellIndex(cellID))
  {
    return baseTopo->cellHasCurvedEdges(cellID);
  }
  return false;
}

static string fileSafe(const string &name)
{
  string safe;
  for (char c : name)
  {
    if (isalnum((unsigned char) c) || (c == '_') || (c == '-') || (c == '.')) safe += c;
  }
  return safe.empty() ? "var" : safe;
}

struct GLVisSideKey
{
  int cellID;
  int sideOrdinal;
};

static bool operator<(const GLVisSideKey &a, const GLVisSideKey &b)
{
  return (a.cellID < b.cellID) || ((a.cellID == b.cellID) && (a.sideOrdinal < b.sideOrdinal));
}

GLVisExporter::GLVisExporter(MeshPtr mesh, string outputDirName, string outputDirSuperPath, bool allowOverwrite) :
  _dirName(outputDirName), _dirSuperPath(outputDirSuperPath), _mesh(mesh)
{
  auto comm = MPIWrapper::TeuchosCommWorld();
  string requestedDirName = _dirName;
  if (!allowOverwrite)
  {
    _dirName = firstAvailableDirName(_dirSuperPath, requestedDirName);
  }
  comm->barrier();

  if (comm->getRank() == 0)
  {
    string dirPath = outputDirPath();
    if (_dirName != requestedDirName)
    {
      cout << "GLVisExporter: output directory '" << fullPath(_dirSuperPath, requestedDirName)
           << "' already exists; writing to '" << dirPath << "' instead."
           << " Pass allowOverwrite=true to reuse the requested directory.\n";
    }
    errno = 0;
    int mkdirResult = mkdir(dirPath.c_str(), S_IRWXU | S_IRWXG | S_IXOTH | S_IROTH);
    if (mkdirResult != 0)
    {
      TEUCHOS_TEST_FOR_EXCEPTION((errno != EEXIST) || !allowOverwrite || !directoryExists(dirPath),
                                 std::runtime_error, "GLVisExporter: cannot create output directory " << dirPath);
    }
    remove((dirPath + "/index.txt").c_str());
  }
  comm->barrier();
}

GLVisExporter::~GLVisExporter()
{
}

string GLVisExporter::outputDirPath() const
{
  return fullPath(_dirSuperPath, _dirName);
}

void GLVisExporter::setMinimumDegree(int degree)
{
  _minDegree = max(_minDegree, degree);
}

void GLVisExporter::setGatherToRankZero(bool value)
{
  _gatherToRankZero = value;
}

void GLVisExporter::exportFunction(TFunctionPtr<double> function, string functionName, double timeVal, set<GlobalIndexType> cellIndices)
{
  exportFunction(vector<TFunctionPtr<double>>{function}, vector<string>{functionName}, timeVal, cellIndices);
}

void GLVisExporter::exportFunction(vector<TFunctionPtr<double>> functions, vector<string> functionNames, double timeVal, set<GlobalIndexType> cellIndices)
{
  TEUCHOS_TEST_FOR_EXCEPTION(functions.size() != functionNames.size(), std::invalid_argument, "functions must match functionNames in length!");
  TEUCHOS_TEST_FOR_EXCEPTION(functions.size() == 0, std::invalid_argument, "empty function container");

  int commRank = _mesh->Comm()->getRank();
  if (cellIndices.size() == 0) cellIndices = _mesh->globalDofAssignment()->cellsInPartition(commRank);

  int localDegree = 0;
  for (GlobalIndexType cellID : cellIndices)
  {
    for (int order : _mesh->cellTensorPolyOrder(cellID))
    {
      localDegree = max(localDegree, order);
    }
  }

  vector<TFunctionPtr<double>> fieldFunctions, boundaryFunctions;
  vector<string> fieldNames, boundaryNames;
  for (int i = 0; i < functions.size(); i++)
  {
    if (functions[i]->boundaryValueOnly())
    {
      boundaryFunctions.push_back(functions[i]);
      boundaryNames.push_back(functionNames[i]);
    }
    else
    {
      fieldFunctions.push_back(functions[i]);
      fieldNames.push_back(functionNames[i]);
    }
  }

  int step = _exportStep;
  if (fieldFunctions.size() > 0) writeExport(fieldFunctions, fieldNames, localDegree, timeVal, cellIndices);
  if ((fieldFunctions.size() > 0) && (boundaryFunctions.size() > 0)) _exportStep = step;
  if (boundaryFunctions.size() > 0) writeBoundaryExport(boundaryFunctions, boundaryNames, localDegree, timeVal, cellIndices);
}

void GLVisExporter::exportSolution(TSolutionPtr<double> solution, double timeVal, set<GlobalIndexType> cellIndices)
{
  int commRank = _mesh->Comm()->getRank();
  if (cellIndices.size() == 0) cellIndices = _mesh->globalDofAssignment()->cellsInPartition(commRank);

  VarFactoryPtr varFactory = _mesh->varFactory();
  vector<int> fieldTrialIDs = solution->bf()->trialVolumeIDs();
  vector<int> traceTrialIDs = solution->bf()->trialBoundaryIDs();
  TEUCHOS_TEST_FOR_EXCEPTION((fieldTrialIDs.size() == 0) && (traceTrialIDs.size() == 0), std::invalid_argument, "solution has no field or trace/flux variables");

  int h1Max = 0;
  for (GlobalIndexType cellID : cellIndices)
  {
    for (int order : _mesh->cellTensorPolyOrder(cellID))
    {
      h1Max = max(h1Max, order);
    }
  }

  bool dontWeightByParity = false;
  vector<TFunctionPtr<double>> fieldFunctions, boundaryFunctions;
  vector<string> fieldNames, boundaryNames;
  int localDegree = 0;
  for (int trialID : fieldTrialIDs)
  {
    VarPtr var = varFactory->trial(trialID);
    fieldFunctions.push_back(TFunction<double>::solution(var, solution, dontWeightByParity));
    fieldNames.push_back(var->name());
    bool hvol = (var->space() == L2) || (var->space() == VECTOR_L2);
    localDegree = max(localDegree, hvol ? h1Max - 1 : h1Max);
  }
  for (int trialID : traceTrialIDs)
  {
    VarPtr var = varFactory->trial(trialID);
    boundaryFunctions.push_back(TFunction<double>::solution(var, solution, dontWeightByParity));
    boundaryNames.push_back(var->name());
    localDegree = max(localDegree, h1Max);
  }

  int step = _exportStep;
  if (fieldFunctions.size() > 0) writeExport(fieldFunctions, fieldNames, localDegree, timeVal, cellIndices);
  if ((fieldFunctions.size() > 0) && (boundaryFunctions.size() > 0)) _exportStep = step;
  if (boundaryFunctions.size() > 0) writeBoundaryExport(boundaryFunctions, boundaryNames, localDegree, timeVal, cellIndices);
}

void GLVisExporter::exportTimeSlab(TFunctionPtr<double> function, string functionName, double tInit, double tFinal,
                                   unsigned int numSlices, unsigned int sliceH1Order)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                             "GLVisExporter: 4D/time-slab export is unsupported; export 3D slices instead");
}

void GLVisExporter::getGaussLobattoPointsFC(FieldContainer<double> &points, int spaceDim, int degree)
{
  vector<double> gl = gaussLobatto(degree + 1);
  int n1D = degree + 1;
  int numPoints = points.dimension(0);
  for (int pt = 0; pt < numPoints; pt++)
  {
    int rest = pt;
    for (int d = 0; d < spaceDim; d++)
    {
      points(pt, d) = 2.0 * gl[rest % n1D] - 1.0;
      rest /= n1D;
    }
  }
}

void GLVisExporter::writeExport(const vector<TFunctionPtr<double>> &functions, const vector<string> &functionNames,
                                int localDegree, double timeVal, const set<GlobalIndexType> &cellIndices)
{
  int dim = _mesh->getTopology()->getDimension();
  TEUCHOS_TEST_FOR_EXCEPTION(dim > 3, std::invalid_argument,
                             "GLVisExporter: total dimension 4 is unsupported; export 3D slices instead");

  Teuchos_CommPtr comm = _mesh->Comm();
  int commRank = comm->getRank();
  int numProcs = comm->getSize();

  MeshTopologyViewPtr meshTopo = _mesh->getTopology();
  bool localUnsupportedTopology = false;
  for (GlobalIndexType cellID : cellIndices)
  {
    if (cellHasCurvedEdges(meshTopo, cellID))
    {
      localUnsupportedTopology = true;
      continue;
    }
    try
    {
      mfemNodePerm(meshTopo->getCell(cellID)->topology());
    }
    catch (const std::invalid_argument &)
    {
      localUnsupportedTopology = true;
    }
  }
  bool unsupportedTopology = MPIWrapper::globalOr(comm, localUnsupportedTopology);
  TEUCHOS_TEST_FOR_EXCEPTION(unsupportedTopology, std::invalid_argument,
                             "GLVisExporter supports straight-sided segment, quadrilateral, and hexahedral topologies only; curvilinear cells and simplices are unsupported");

  int degree = MPIWrapper::max(comm, max(localDegree, _minDegree));
  int n1D = degree + 1;
  int numPoints = 1;
  for (int d = 0; d < dim; d++) numPoints *= n1D;
  int numVertices = 1 << dim;
  int geomCode = (dim == 1) ? 1 : (dim == 2) ? 3 : 5;

  FieldContainer<double> refPoints(numPoints, dim);
  getGaussLobattoPointsFC(refPoints, dim, degree);

  int nFcns = functions.size();
  vector<string> names(nFcns);
  for (int i = 0; i < nFcns; i++) names[i] = fileSafe(functionNames[i]);
  vector<int> vdims(nFcns);
  for (int i = 0; i < nFcns; i++)
  {
    if (functions[i]->rank() == 0) vdims[i] = 1;
    else if (functions[i]->rank() == 1) vdims[i] = dim;
    else TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled function rank");
  }

  vector<int> cellIDs;
  vector<double> coords;
  vector<int> faceCounts;
  vector<int> faceNodes;
  vector<vector<double>> fcnValues(nFcns);

  for (GlobalIndexType cellID : cellIndices)
  {
    CellPtr cell = meshTopo->getCell(cellID);
    CellTopoPtr cellTopo = cell->topology();
    vector<int> perm = mfemNodePerm(cellTopo);
    vector<int> invPerm(numVertices);
    for (int i = 0; i < numVertices; i++) invPerm[perm[i]] = i;

    cellIDs.push_back(cellID);
    const vector<IndexType> &vertices = cell->vertices();
    for (int i = 0; i < numVertices; i++)
    {
      const vector<double> &vertex = meshTopo->getVertex(vertices[perm[i]]);
      for (int d = 0; d < dim; d++) coords.push_back(vertex[d]);
    }

    int faceCount = 0;
    if (dim == 3)
    {
      for (unsigned side = 0; side < cell->getSideCount(); side++)
      {
        if (!cell->isBoundary(side)) continue;
        vector<int> sidePerm = mfemNodePerm(cellTopo->getSide(side));
        for (int f = 0; f < 4; f++)
        {
          unsigned cellNode = cellTopo->getNodeMap(2, side, sidePerm[f]);
          faceNodes.push_back(invPerm[cellNode]);
        }
        faceCount++;
      }
    }
    faceCounts.push_back(faceCount);

    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh, cellID);
    basisCache->setRefCellPointsFC(refPoints);
    for (int i = 0; i < nFcns; i++)
    {
      if (vdims[i] == 1)
      {
        FieldContainer<double> values(1, numPoints);
        functions[i]->values(values, basisCache);
        for (int pt = 0; pt < numPoints; pt++) fcnValues[i].push_back(values(0, pt));
      }
      else
      {
        FieldContainer<double> values(1, numPoints, dim);
        functions[i]->values(values, basisCache);
        for (int c = 0; c < vdims[i]; c++)
        {
          for (int pt = 0; pt < numPoints; pt++) fcnValues[i].push_back(values(0, pt, c));
        }
      }
    }
  }

  bool gather = _gatherToRankZero && (numProcs > 1);
  if (gather)
  {
    vector<int> offsets;
    vector<int> allCellIDs, allFaceCounts, allFaceNodes;
    vector<double> allCoords;
    MPIWrapper::allGatherVariable(comm, allCellIDs, cellIDs, offsets);
    MPIWrapper::allGatherVariable(comm, allCoords, coords, offsets);
    MPIWrapper::allGatherVariable(comm, allFaceCounts, faceCounts, offsets);
    MPIWrapper::allGatherVariable(comm, allFaceNodes, faceNodes, offsets);
    for (int i = 0; i < nFcns; i++)
    {
      vector<double> allValues;
      MPIWrapper::allGatherVariable(comm, allValues, fcnValues[i], offsets);
      fcnValues[i] = allValues;
    }
    cellIDs = allCellIDs;
    coords = allCoords;
    faceCounts = allFaceCounts;
    faceNodes = allFaceNodes;
  }

  int numCells = cellIDs.size();
  vector<int> faceOffsets(numCells);
  int offset = 0;
  for (int k = 0; k < numCells; k++)
  {
    faceOffsets[k] = offset;
    offset += faceCounts[k];
  }
  vector<int> order(numCells);
  iota(order.begin(), order.end(), 0);
  sort(order.begin(), order.end(), [&](int a, int b) { return cellIDs[a] < cellIDs[b]; });

  string suffix = "";
  if (!gather && (numProcs > 1))
  {
    char rankBuf[16];
    snprintf(rankBuf, sizeof(rankBuf), ".%06d", commRank);
    suffix = rankBuf;
  }
  ostringstream basePath;
  basePath << outputDirPath() << "/step" << _exportStep;

  if (!gather || (commRank == 0))
  {
    char buf[32];
    ofstream meshFile(basePath.str() + ".mesh" + suffix);
    TEUCHOS_TEST_FOR_EXCEPTION(!meshFile, std::runtime_error, "GLVisExporter: cannot open " << basePath.str() << ".mesh" << suffix);
    meshFile << "MFEM mesh v1.0\n\ndimension\n" << dim << "\n\n";
    meshFile << "elements\n" << numCells << "\n";
    for (int k = 0; k < numCells; k++)
    {
      meshFile << "1 " << geomCode;
      for (int v = 0; v < numVertices; v++) meshFile << " " << k * numVertices + v;
      meshFile << "\n";
    }
    int numFaces = 0;
    for (int k = 0; k < numCells; k++) numFaces += faceCounts[k];
    meshFile << "\nboundary\n" << numFaces << "\n";
    for (int k = 0; k < numCells; k++)
    {
      int cellOrdinal = order[k];
      for (int f = 0; f < faceCounts[cellOrdinal]; f++)
      {
        meshFile << "1 3";
        for (int v = 0; v < 4; v++) meshFile << " " << k * numVertices + faceNodes[(faceOffsets[cellOrdinal] + f) * 4 + v];
        meshFile << "\n";
      }
    }
    meshFile << "\nvertices\n" << numCells * numVertices << "\n" << dim << "\n";
    for (int k = 0; k < numCells; k++)
    {
      int cellOrdinal = order[k];
      for (int v = 0; v < numVertices; v++)
      {
        for (int d = 0; d < dim; d++)
        {
          snprintf(buf, sizeof(buf), "%.17g", coords[(cellOrdinal * numVertices + v) * dim + d]);
          meshFile << ((d > 0) ? " " : "") << buf;
        }
        meshFile << "\n";
      }
    }
    meshFile.close();

    for (int i = 0; i < nFcns; i++)
    {
      ofstream gfFile(basePath.str() + "_" + names[i] + ".gf" + suffix);
      TEUCHOS_TEST_FOR_EXCEPTION(!gfFile, std::runtime_error, "GLVisExporter: cannot open " << basePath.str() << "_" << names[i] << ".gf" << suffix);
      gfFile << "FiniteElementSpace\n";
      gfFile << "FiniteElementCollection: L2_T1_" << dim << "D_P" << degree << "\n";
      gfFile << "VDim: " << vdims[i] << "\nOrdering: 0\n\n";
      int stride = vdims[i] * numPoints;
      for (int c = 0; c < vdims[i]; c++)
      {
        for (int k = 0; k < numCells; k++)
        {
          int cellOrdinal = order[k];
          for (int pt = 0; pt < numPoints; pt++)
          {
            snprintf(buf, sizeof(buf), "%.17g", fcnValues[i][cellOrdinal * stride + c * numPoints + pt]);
            gfFile << buf << "\n";
          }
        }
      }
      gfFile.close();
    }
  }

  if (commRank == 0)
  {
    ofstream indexFile(outputDirPath() + "/index.txt", ios::app);
    indexFile << "step " << _exportStep << " time " << timeVal << " degree " << degree
              << " mode " << (gather ? "gathered" : "pieces") << " ranks " << (gather ? 1 : numProcs) << " vars";
    for (auto &name : names) indexFile << " " << name;
    indexFile << "\n";
  }
  _exportStep++;
}

void GLVisExporter::writeBoundaryExport(const vector<TFunctionPtr<double>> &functions, const vector<string> &functionNames,
                                          int localDegree, double timeVal, const set<GlobalIndexType> &cellIndices)
{
  int dim = _mesh->getTopology()->getDimension();
  TEUCHOS_TEST_FOR_EXCEPTION(dim > 3, std::invalid_argument,
                             "GLVisExporter: total dimension 4 is unsupported; export 3D slices instead");
  TEUCHOS_TEST_FOR_EXCEPTION(dim <= 1, std::invalid_argument,
                             "GLVisExporter: trace/flux export is supported for 2D and 3D volume meshes only");
  int sideDim = dim - 1;

  Teuchos_CommPtr comm = _mesh->Comm();
  int commRank = comm->getRank();
  int numProcs = comm->getSize();
  MeshTopologyViewPtr meshTopo = _mesh->getTopology();

  bool localUnsupportedTopology = false;
  for (GlobalIndexType cellID : cellIndices)
  {
    if (cellHasCurvedEdges(meshTopo, cellID))
    {
      localUnsupportedTopology = true;
      continue;
    }
    try
    {
      CellPtr cell = meshTopo->getCell(cellID);
      mfemNodePerm(cell->topology());
      for (unsigned side = 0; side < cell->getSideCount(); side++) mfemNodePerm(cell->topology()->getSubcell(sideDim, side));
    }
    catch (const std::invalid_argument &)
    {
      localUnsupportedTopology = true;
    }
  }
  bool unsupportedTopology = MPIWrapper::globalOr(comm, localUnsupportedTopology);
  TEUCHOS_TEST_FOR_EXCEPTION(unsupportedTopology, std::invalid_argument,
                             "GLVisExporter supports straight-sided segment, quadrilateral, and hexahedral topologies only; curvilinear cells and simplices are unsupported");

  int degree = MPIWrapper::max(comm, max(localDegree, _minDegree));
  int n1D = degree + 1;
  int numPoints = 1;
  for (int d = 0; d < sideDim; d++) numPoints *= n1D;
  int numVertices = 1 << sideDim;
  int geomCode = (sideDim == 1) ? 1 : 3;

  FieldContainer<double> refPoints(numPoints, sideDim);
  getGaussLobattoPointsFC(refPoints, sideDim, degree);

  int nFcns = functions.size();
  vector<string> names(nFcns);
  for (int i = 0; i < nFcns; i++) names[i] = fileSafe(functionNames[i]);
  vector<int> vdims(nFcns);
  for (int i = 0; i < nFcns; i++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(!functions[i]->boundaryValueOnly(), std::invalid_argument, "writeBoundaryExport expects trace/flux functions only");
    if (functions[i]->rank() == 0) vdims[i] = 1;
    else if (functions[i]->rank() == 1) vdims[i] = dim;
    else TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled function rank");
  }

  vector<GLVisSideKey> sideKeys;
  vector<double> coords;
  vector<vector<double>> fcnValues(nFcns);

  for (GlobalIndexType cellID : cellIndices)
  {
    CellPtr cell = meshTopo->getCell(cellID);
    CellTopoPtr cellTopo = cell->topology();
    const vector<IndexType> &vertices = cell->vertices();
    BasisCachePtr volumeBasisCache = BasisCache::basisCacheForCell(_mesh, cellID);
    for (unsigned side = 0; side < cell->getSideCount(); side++)
    {
      CellTopoPtr sideTopo = cellTopo->getSubcell(sideDim, side);
      vector<int> sidePerm = mfemNodePerm(sideTopo);
      sideKeys.push_back({cellID, (int) side});
      for (int i = 0; i < numVertices; i++)
      {
        unsigned cellNode = cellTopo->getNodeMap(sideDim, side, sidePerm[i]);
        const vector<double> &vertex = meshTopo->getVertex(vertices[cellNode]);
        for (int d = 0; d < dim; d++) coords.push_back(vertex[d]);
      }

      BasisCachePtr sideCache = volumeBasisCache->getSideBasisCache(side);
      sideCache->setRefCellPointsFC(refPoints);
      for (int i = 0; i < nFcns; i++)
      {
        if (vdims[i] == 1)
        {
          FieldContainer<double> values(1, numPoints);
          if (functions[i]->isDefined(sideCache)) functions[i]->values(values, sideCache);
          for (int pt = 0; pt < numPoints; pt++) fcnValues[i].push_back(values(0, pt));
        }
        else
        {
          FieldContainer<double> values(1, numPoints, dim);
          if (functions[i]->isDefined(sideCache)) functions[i]->values(values, sideCache);
          for (int c = 0; c < vdims[i]; c++)
          {
            for (int pt = 0; pt < numPoints; pt++) fcnValues[i].push_back(values(0, pt, c));
          }
        }
      }
    }
  }

  bool gather = _gatherToRankZero && (numProcs > 1);
  if (gather)
  {
    vector<int> offsets;
    vector<GLVisSideKey> allSideKeys;
    vector<double> allCoords;
    MPIWrapper::allGatherVariable(comm, allSideKeys, sideKeys, offsets);
    MPIWrapper::allGatherVariable(comm, allCoords, coords, offsets);
    for (int i = 0; i < nFcns; i++)
    {
      vector<double> allValues;
      MPIWrapper::allGatherVariable(comm, allValues, fcnValues[i], offsets);
      fcnValues[i] = allValues;
    }
    sideKeys = allSideKeys;
    coords = allCoords;
  }

  int numSides = sideKeys.size();
  vector<int> order(numSides);
  iota(order.begin(), order.end(), 0);
  sort(order.begin(), order.end(), [&](int a, int b) { return sideKeys[a] < sideKeys[b]; });

  string suffix = "";
  if (!gather && (numProcs > 1))
  {
    char rankBuf[16];
    snprintf(rankBuf, sizeof(rankBuf), ".%06d", commRank);
    suffix = rankBuf;
  }
  ostringstream basePath;
  basePath << outputDirPath() << "/step" << _exportStep << "_trace";

  if (!gather || (commRank == 0))
  {
    char buf[32];
    ofstream meshFile(basePath.str() + ".mesh" + suffix);
    TEUCHOS_TEST_FOR_EXCEPTION(!meshFile, std::runtime_error, "GLVisExporter: cannot open " << basePath.str() << ".mesh" << suffix);
    meshFile << "MFEM mesh v1.0\n\ndimension\n" << sideDim << "\n\n";
    meshFile << "elements\n" << numSides << "\n";
    for (int k = 0; k < numSides; k++)
    {
      meshFile << "1 " << geomCode;
      for (int v = 0; v < numVertices; v++) meshFile << " " << k * numVertices + v;
      meshFile << "\n";
    }
    meshFile << "\nboundary\n0\n";
    meshFile << "\nvertices\n" << numSides * numVertices << "\n" << dim << "\n";
    for (int k = 0; k < numSides; k++)
    {
      int sideOrdinal = order[k];
      for (int v = 0; v < numVertices; v++)
      {
        for (int d = 0; d < dim; d++)
        {
          snprintf(buf, sizeof(buf), "%.17g", coords[(sideOrdinal * numVertices + v) * dim + d]);
          meshFile << ((d > 0) ? " " : "") << buf;
        }
        meshFile << "\n";
      }
    }
    meshFile.close();

    for (int i = 0; i < nFcns; i++)
    {
      ofstream gfFile(basePath.str() + "_" + names[i] + ".gf" + suffix);
      TEUCHOS_TEST_FOR_EXCEPTION(!gfFile, std::runtime_error, "GLVisExporter: cannot open " << basePath.str() << "_" << names[i] << ".gf" << suffix);
      gfFile << "FiniteElementSpace\n";
      gfFile << "FiniteElementCollection: L2_T1_" << sideDim << "D_P" << degree << "\n";
      gfFile << "VDim: " << vdims[i] << "\nOrdering: 0\n\n";
      int stride = vdims[i] * numPoints;
      for (int c = 0; c < vdims[i]; c++)
      {
        for (int k = 0; k < numSides; k++)
        {
          int sideOrdinal = order[k];
          for (int pt = 0; pt < numPoints; pt++)
          {
            snprintf(buf, sizeof(buf), "%.17g", fcnValues[i][sideOrdinal * stride + c * numPoints + pt]);
            gfFile << buf << "\n";
          }
        }
      }
      gfFile.close();
    }
  }

  if (commRank == 0)
  {
    ofstream indexFile(outputDirPath() + "/index.txt", ios::app);
    indexFile << "step " << _exportStep << " trace time " << timeVal << " degree " << degree
              << " mode " << (gather ? "gathered" : "pieces") << " ranks " << (gather ? 1 : numProcs) << " vars";
    for (auto &name : names) indexFile << " " << name;
    indexFile << "\n";
  }
  _exportStep++;
}

void GLVisExporter::exportFunction(string directoryPath, string functionName, TFunctionPtr<double> function, MeshPtr mesh)
{
  GLVisExporter exporter(mesh, functionName, directoryPath);
  exporter.exportFunction(function, functionName);
}

void GLVisExporter::exportSolution(string directoryPath, string solutionName, TSolutionPtr<double> solution)
{
  GLVisExporter exporter(solution->mesh(), solutionName, directoryPath);
  exporter.exportSolution(solution);
}
