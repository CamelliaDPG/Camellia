//TODO: solution output
#include "XDMFExporter.h"
#include "CamelliaConfig.h"

#include "CamelliaCellTools.h"

#ifdef USE_XDMF
#include <Xdmf.h>

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#endif

void XDMFExporter::exportSolution(SolutionPtr solution, MeshPtr mesh, VarFactory varFactory, double timeVal, unsigned int defaultNum1DPts, map<int, int> cellIDToNum1DPts, set<GlobalIndexType> cellIndices)
{
  vector<int> fieldTrialIDs = mesh->bilinearForm()->trialVolumeIDs();
  vector<int> traceTrialIDs = mesh->bilinearForm()->trialBoundaryIDs();
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
  exportFunction(fieldFunctions, fieldFunctionNames, timeVal, defaultNum1DPts, cellIDToNum1DPts, mesh, cellIndices);
  exportFunction(traceFunctions, traceFunctionNames, timeVal, defaultNum1DPts, cellIDToNum1DPts, mesh, cellIndices);
}

void XDMFExporter::exportFunction(FunctionPtr function, string functionName, double timeVal, unsigned int defaultNum1DPts, map<int, int> cellIDToNum1DPts, MeshPtr mesh, set<GlobalIndexType> cellIndices)
{
  vector<FunctionPtr> functions;
  functions.push_back(function);
  vector<string> functionNames;
  functionNames.push_back(functionName);
  exportFunction(functions, functionNames, timeVal, defaultNum1DPts, cellIDToNum1DPts, mesh, cellIndices);
}

void XDMFExporter::exportFunction(vector<FunctionPtr> functions, vector<string> functionNames, double timeVal, unsigned int defaultNum1DPts, map<int, int> cellIDToNum1DPts, MeshPtr mesh, set<GlobalIndexType> cellIndices)
{
  int nFcns = functions.size();

  // XdmfDOM         dom;
  // XdmfRoot        root;
  // XdmfDomain      domain;
  XdmfGrid        grid;
  XdmfTime        time;
  XdmfTopology    *topology;
  XdmfGeometry    *geometry;
  XdmfAttribute   nodedata[nFcns];
  XdmfAttribute   celldata;
  XdmfArray       *ptArray;
  XdmfArray       *connArray;
  XdmfArray       *valArray[nFcns];

  // root.SetDOM(&dom);
  // root.SetVersion(2.0);
  // root.Build();
  //     // Domain
  // root.Insert(&domain);
  //     // Grid
  // grid.SetName("Grid");
  // domain.Insert(&grid);

  // int num1DPts = defaultNum1DPts;
  // bool defaultPts = (num1DPts == 0);
  // int pOrder = 6;
  // if (defaultPts)
  //   if (pOrder < 2)
  //     num1DPts = 2;
  //   else
  //     // num1DPts = 2*pOrder+2;
  //     num1DPts = pOrder+1;
  if (defaultNum1DPts < 2)
    defaultNum1DPts = 2;

  int spaceDim = _meshTopology->getSpaceDim();

  bool exportingBoundaryValues = functions[0]->boundaryValueOnly();
  if (!exportingBoundaryValues)
  {
    grid.SetName("Field Grid");
    fieldTemporalCollection.Insert(&grid);
  }
  else
  {
    grid.SetName("Trace Grid");
    traceTemporalCollection.Insert(&grid);
  }

  time.SetTimeType(XDMF_TIME_SINGLE);
  time.SetValue(timeVal);
  grid.Insert(&time);

  for (int i=0; i < nFcns; i++)
    if (exportingBoundaryValues != functions[i]->boundaryValueOnly())
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Can not export trace and field variables together");

  // Topology
  topology = grid.GetTopology();
  topology->SetTopologyType(XDMF_MIXED);

  unsigned int total_vertices = 0;
  
  if (cellIndices.size()==0) cellIndices = _meshTopology->getActiveCellIndices();
  // Number of line elements in 1D mesh
  int numLines=0;
  // Number of triangle elements in 2D mesh
  int numTriangles=0;
  // Number of quad elements in 2D mesh
  int numQuads=0;
  // Number of hex elements in 3D mesh
  int numHexas=0;
  // Number of line subdivisions in 1D mesh
  int totalSubLines=0;
  // Number of triangle subdivisions in 2D mesh
  int totalSubTriangles=0;
  // Number of quad subdivisions in 2D mesh
  int totalSubQuads=0;
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
  // Total number of boundary faces in 3D mesh
  int totalBoundaryFaces=0;
  for (set<GlobalIndexType>::iterator cellIt = cellIndices.begin(); cellIt != cellIndices.end(); cellIt++) {
    CellPtr cell = _meshTopology->getCell(*cellIt);
    if (!cellIDToNum1DPts[cell->cellIndex()] || cellIDToNum1DPts[cell->cellIndex()] < 2)
      cellIDToNum1DPts[cell->cellIndex()] = defaultNum1DPts;
    int num1DPts = cellIDToNum1DPts[cell->cellIndex()];
    if (cell->topology()->getKey() == shards::Line<2>::key) 
    {
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
    }
    if (cell->topology()->getKey() == shards::Triangle<3>::key) 
    {
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
    }
    if (cell->topology()->getKey() == shards::Quadrilateral<4>::key) 
    {
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
    }
    if (cell->topology()->getKey() == shards::Hexahedron<8>::key) 
    {
      numHexas++;
      if (!exportingBoundaryValues)
      {
        totalSubHexas += (num1DPts-1)*(num1DPts-1)*(num1DPts-1);
        totalPts += num1DPts*num1DPts*num1DPts;
      }
      else
      {
        totalBoundaryFaces += 6;
        totalSubQuads += 6*(num1DPts-1)*(num1DPts-1);
        totalPts += 6*num1DPts*num1DPts;
      }
    }
  }
  totalSubcells = totalBoundaryPts + totalSubLines + totalSubTriangles + totalSubQuads + totalSubHexas;
  // if (!exportingBoundaryValues)
  // {
  //   totalSubLines = (num1DPts-1)*numLines;
  //   totalSubTriangles = (num1DPts-1)*(num1DPts-1)*numTriangles;
  //   totalSubQuads = (num1DPts-1)*(num1DPts-1)*numQuads;
  //   totalSubHexas = (num1DPts-1)*(num1DPts-1)*(num1DPts-1)*numHexas;
    // totalPts = num1DPts*numLines + num1DPts*(num1DPts+1)/2*numTriangles + num1DPts*num1DPts*numQuads + num1DPts*num1DPts*num1DPts*numHexas;
    // totalSubcells = totalSubLines + totalSubTriangles + totalSubQuads + totalSubHexas;
  // }
  // else
  // {
  //   totalBoundaryPts = 2*numLines;
  //   totalBoundaryLines = 3*numTriangles + 4*numQuads;
  //   totalBoundaryFaces = 6*numHexas;
    // totalSubLines = 3*(num1DPts-1)*numTriangles + 4*(num1DPts-1)*numQuads;
    // totalSubTriangles = 0;
    // totalSubQuads = 6*(num1DPts-1)*(num1DPts-1)*numHexas;
    // totalSubHexas = 0;
    // totalPts = 2*numLines + 3*num1DPts*numTriangles + 4*num1DPts*numQuads + 6*num1DPts*num1DPts*numHexas;
    // totalSubcells = totalBoundaryPts + totalSubLines + totalSubTriangles + totalSubQuads + totalSubHexas;
  // }

  // Topology
  topology = grid.GetTopology();
  topology->SetTopologyType(XDMF_MIXED);
  if (!exportingBoundaryValues)
  {
    if (spaceDim == 1)
      topology->SetNumberOfElements(numLines);
    else
      topology->SetNumberOfElements(totalSubcells);
  }
  else
  {
    if (spaceDim == 1)
    {
      // TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "XDMFExporter does not work on 1D boundary value only functions");
      topology->SetNumberOfElements(totalBoundaryPts);
    }
    else if (spaceDim == 2)
      topology->SetNumberOfElements(totalBoundaryLines);
    else if (spaceDim == 3)
      topology->SetNumberOfElements(totalSubQuads);
  }
  connArray = topology->GetConnectivity();
  if (!exportingBoundaryValues)
  {
    if (spaceDim == 1)
      connArray->SetNumberOfElements(2*numLines+totalPts);
    else
      connArray->SetNumberOfElements(totalSubcells + 3*totalSubTriangles + 4*totalSubQuads + 8*totalSubHexas);
  }
  else
  {
    if (spaceDim == 1)
      connArray->SetNumberOfElements(3*totalBoundaryPts);
    if (spaceDim == 2)
      connArray->SetNumberOfElements(2*totalBoundaryLines + totalPts);
    if (spaceDim == 3)
      connArray->SetNumberOfElements(5*totalSubQuads);
  }
  // Geometry
  geometry = grid.GetGeometry();
  if (spaceDim < 3)
    geometry->SetGeometryType(XDMF_GEOMETRY_XY);
  else if (spaceDim == 3)
    geometry->SetGeometryType(XDMF_GEOMETRY_XYZ);
  geometry->SetNumberOfPoints(totalPts);
  ptArray = geometry->GetPoints();
  ptArray->SetNumberType(XDMF_FLOAT64_TYPE);
  if (spaceDim == 1)
    ptArray->SetNumberOfElements(2 * totalPts);
  else
    ptArray->SetNumberOfElements(spaceDim * totalPts);
  // Node Data
  for (int i=0; i<nFcns; i++)
  {
    nodedata[i].SetName(functionNames[i].c_str());
    nodedata[i].SetAttributeCenter(XDMF_ATTRIBUTE_CENTER_NODE);
  }
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
      nodedata[i].SetAttributeType(XDMF_ATTRIBUTE_TYPE_SCALAR);
    }
    else
    {
      nodedata[i].SetAttributeType(XDMF_ATTRIBUTE_TYPE_VECTOR);
    }
    valArray[i] = nodedata[i].GetValues();
    valArray[i]->SetNumberType(XDMF_FLOAT64_TYPE);
    if (numFcnComponents[i] == 1)
    {
      valArray[i]->SetNumberOfElements(totalPts);
    }
    else
    {
      valArray[i]->SetNumberOfElements(3*totalPts);
    }
  }


  int connIndex = 0;
  int ptIndex = 0;
  int valIndex[nFcns];
  for (int i = 0; i < nFcns; i++)
    valIndex[i] = 0;
  
  for (set<GlobalIndexType>::iterator cellIt = cellIndices.begin(); cellIt != cellIndices.end(); cellIt++) 
  {
    GlobalIndexType cellIndex = *cellIt;
    CellPtr cell = _meshTopology->getCell(cellIndex);

    FieldContainer<double> physicalCellNodes = _meshTopology->physicalCellNodesForCell(cellIndex);

    CellTopoPtr cellTopoPtr = cell->topology();
    int num1DPts = cellIDToNum1DPts[cell->cellIndex()];
    int numPoints = 0;
    
    if (physicalCellNodes.rank() == 2)
      physicalCellNodes.resize(1,physicalCellNodes.dimension(0), physicalCellNodes.dimension(1));
    bool createSideCache = functions[0]->boundaryValueOnly();
    
    BasisCachePtr volumeBasisCache = Teuchos::rcp( new BasisCache(*cellTopoPtr, 1, createSideCache) );
    volumeBasisCache->setPhysicalCellNodes(physicalCellNodes, vector<GlobalIndexType>(1,cellIndex), createSideCache);

    int numSides = createSideCache ? CamelliaCellTools::getSideCount(*cellTopoPtr) : 1;
    
    int sideDim = spaceDim - 1;
    
    for (int sideOrdinal = 0; sideOrdinal < numSides; sideOrdinal++) 
    {
      shards::CellTopology topo = createSideCache ? cellTopoPtr->getBaseCellTopologyData(sideDim, sideOrdinal) : *cellTopoPtr;
      unsigned cellTopoKey = topo.getKey();
      
      BasisCachePtr basisCache = createSideCache ? volumeBasisCache->getSideBasisCache(sideOrdinal) : volumeBasisCache;
      if (mesh.get())
        basisCache->setMesh(mesh);
      
      unsigned domainDim = createSideCache ? sideDim : spaceDim;
      
      switch (cellTopoKey)
      {
        case shards::Node::key:
          numPoints = 1;
          break;
        case shards::Line<2>::key:
          numPoints = num1DPts;
          break;
        case shards::Quadrilateral<4>::key:
          numPoints = num1DPts*num1DPts;
          break;
        case shards::Triangle<3>::key:
          numPoints = num1DPts*(num1DPts+1)/2;
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


      int subcellStartIndex = total_vertices;
      switch (cellTopoKey)
      {
        case shards::Node::key:
        {
          connArray->SetValue(connIndex, 1);
          connIndex++;
          connArray->SetValue(connIndex, 1);
          connIndex++;
          int ind1 = total_vertices;
          connArray->SetValue(connIndex, ind1);
          connIndex++;
        }
        break;
        case shards::Line<2>::key:
        {
          connArray->SetValue(connIndex, 2);
          connIndex++;
          connArray->SetValue(connIndex, num1DPts);
          connIndex++;
          for (int i=0; i < num1DPts; i++)
          {
            int ind1 = total_vertices + i;
            connArray->SetValue(connIndex, ind1);
            connIndex++;
          }
          // for (int i=0; i < num1DPts-1; i++)
          // {
          //   int ind1 = total_vertices + i;
          //   int ind2 = ind1 + 1;
          //   vtkIdType subCell[2] = {ind1, ind2};
          //   ug->InsertNextCell((int)VTK_LINE, 2, subCell);
          // }
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
              connArray->SetValue(connIndex, 5);
              connIndex++;
              connArray->SetValue(connIndex, ind1);
              connIndex++;
              connArray->SetValue(connIndex, ind2);
              connIndex++;
              connArray->SetValue(connIndex, ind3);
              connIndex++;
              connArray->SetValue(connIndex, ind4);
              connIndex++;
            }
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
              connArray->SetValue(connIndex, 4);
              connIndex++;
              connArray->SetValue(connIndex, ind1);
              connIndex++;
              connArray->SetValue(connIndex, ind2);
              connIndex++;
              connArray->SetValue(connIndex, ind3);
              connIndex++;

              if (i < num1DPts-2-j)
              {
                int ind1 = subcellStartIndex+1;
                int ind2 = ind1 + num1DPts - j;
                int ind3 = ind1 + num1DPts -j - 1;
                connArray->SetValue(connIndex, 4);
                connIndex++;
                connArray->SetValue(connIndex, ind1);
                connIndex++;
                connArray->SetValue(connIndex, ind2);
                connIndex++;
                connArray->SetValue(connIndex, ind3);
                connIndex++;
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
                connArray->SetValue(connIndex, 9);
                connIndex++;
                connArray->SetValue(connIndex, ind1);
                connIndex++;
                connArray->SetValue(connIndex, ind2);
                connIndex++;
                connArray->SetValue(connIndex, ind3);
                connIndex++;
                connArray->SetValue(connIndex, ind4);
                connIndex++;
                connArray->SetValue(connIndex, ind5);
                connIndex++;
                connArray->SetValue(connIndex, ind6);
                connIndex++;
                connArray->SetValue(connIndex, ind7);
                connIndex++;
                connArray->SetValue(connIndex, ind8);
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
          ptArray->SetValue(ptIndex, (*physicalPoints)(0, pointIndex, 0));
          ptIndex++;
          ptArray->SetValue(ptIndex, 0);
          ptIndex++;
        }
        else if (spaceDim == 2)
        {
          ptArray->SetValue(ptIndex, (*physicalPoints)(0, pointIndex, 0));
          ptIndex++;
          ptArray->SetValue(ptIndex, (*physicalPoints)(0, pointIndex, 1));
          ptIndex++;
        }
        else
        {
          ptArray->SetValue(ptIndex, (*physicalPoints)(0, pointIndex, 0));
          ptIndex++;
          ptArray->SetValue(ptIndex, (*physicalPoints)(0, pointIndex, 1));
          ptIndex++;
          ptArray->SetValue(ptIndex, (*physicalPoints)(0, pointIndex, 2));
          ptIndex++;
        }
        for (int i = 0; i < nFcns; i++)
        {
        // Function Values
          FieldContainer<double> computedValues;
          if (functions[i]->rank() == 0)
            computedValues.resize(1, numPoints);
          else if (functions[i]->rank() == 1)
            computedValues.resize(1, numPoints, spaceDim);
          else
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled function rank");

          functions[i]->values(computedValues, basisCache);

          switch(numFcnComponents[i])
          {
            case 1:
            valArray[i]->SetValue(valIndex[i], computedValues(0, pointIndex));
            valIndex[i]++;
            break;
            case 2:
            valArray[i]->SetValue(valIndex[i], computedValues(0, pointIndex, 0));
            valIndex[i]++;
            valArray[i]->SetValue(valIndex[i], computedValues(0, pointIndex, 1));
            valIndex[i]++;
            valArray[i]->SetValue(valIndex[i], 0);
            valIndex[i]++;
            break;
            case 3:
            valArray[i]->SetValue(valIndex[i], computedValues(0, pointIndex, 0));
            valIndex[i]++;
            valArray[i]->SetValue(valIndex[i], computedValues(0, pointIndex, 1));
            valIndex[i]++;
            valArray[i]->SetValue(valIndex[i], computedValues(0, pointIndex, 2));
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
  if (!exportingBoundaryValues)
  {
    stringstream connOut, ptOut;
    connOut << "HDF5/" << filename << timeVal << "-field.h5:/Conns";
    ptOut << "HDF5/" << filename << timeVal << "-field.h5:/Points";
    connArray->SetHeavyDataSetName(connOut.str().c_str());
    ptArray->SetHeavyDataSetName(ptOut.str().c_str());
  }
  else
  {
    stringstream connOut, ptOut;
    connOut << "HDF5/" << filename << timeVal << "-trace.h5:/Conns";
    ptOut << "HDF5/" << filename << timeVal << "-trace.h5:/Points";
    connArray->SetHeavyDataSetName(connOut.str().c_str());
    ptArray->SetHeavyDataSetName(ptOut.str().c_str());
  }
  for (int i = 0; i < nFcns; i++)
  {
    stringstream nodeOut;
    nodeOut << "HDF5/" << filename << timeVal << "-" << functionNames[i] << ".h5:/NodeData";
    // valArray[i]->SetHeavyDataSetName(("HDF5/"+filename+"-"+functionNames[i]+".h5:/NodeData").c_str());
    valArray[i]->SetHeavyDataSetName(nodeOut.str().c_str());
    // Attach and Write
    grid.Insert(&nodedata[i]);
  }
  // grid.Insert(&celldata);
  // Build is recursive ... it will be called on all of the child nodes.
  // This updates the DOM and writes the HDF5
  root.Build();
  // Write the XML
  dom.Write((filename+".xmf").c_str());

  cout << "Wrote to " <<  filename << ".xmf iteration " << timeVal << endl;
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

#endif