#include "Solution.h"

#define USE_VTK
#ifdef USE_VTK
#include "vtkPointData.h"
#include "vtkFloatArray.h"
#include "vtkUnstructuredGrid.h"
#include "vtkXMLUnstructuredGridWriter.h"
#include "vtkCellType.h"
#include "vtkIdList.h"

void Solution::writeTracesToVTK(const string& filePath)
{
  vtkUnstructuredGrid* trace_ug = vtkUnstructuredGrid::New();
  vector<vtkFloatArray*> traceData;
  vtkPoints* trace_points = vtkPoints::New();

  // Get trialIDs
  vector<int> trialIDs = _mesh->bilinearForm()->trialIDs();
  vector<int> traceTrialIDs;

  // Allocate memory for VTK unstructured grid
  int totalCells = _mesh->activeElements().size();
  trace_ug->Allocate(4*totalCells, 4*totalCells);

  // Count trace variables
  int numTraceVars = 0;
  for (unsigned int i=0; i < trialIDs.size(); i++)
  {
    if (_mesh->bilinearForm()->isFluxOrTrace(trialIDs[i]))
    {
      numTraceVars++;
      traceTrialIDs.push_back(trialIDs[i]);
      traceData.push_back(vtkFloatArray::New());
    }
  }
  for (int varIdx = 0; varIdx < numTraceVars; varIdx++)
  {
    traceData[varIdx]->SetNumberOfComponents(1);
    traceData[varIdx]->SetName(_mesh->bilinearForm()->trialName(traceTrialIDs[varIdx]).c_str());
  }
  unsigned int trace_vertex_count = 0;

  BasisCachePtr basisCache;
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) 
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    FieldContainer<double> physicalCellNodes = _mesh()->physicalCellNodesGlobal(elemTypePtr);
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    basisCache = Teuchos::rcp( new BasisCache(physicalCellNodes, cellTopo, 1) );
    if (basisCache.get() == NULL)
      cout << "NULL Basis" << endl;
    int numSides = cellTopo.getSideCount();

    int numCells = physicalCellNodes.dimension(0);
    // determine cellIDs
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) 
    {
      int cellID = _mesh->cellID(elemTypePtr, cellIndex, -1); // -1: "global" lookup (independent of MPI node)
      cellIDs.push_back(cellID);
    }
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, true); // true: create side caches

    // int numPoints = 2;
    // FieldContainer<double> refTracePoints(numPoints);
    // refTracePoints(0) = -1.0;
    // refTracePoints(1) =  1.0;
    // refTracePoints(2) =  0.0;
    FieldContainer<double> refTracePoints(2, 1);
    refTracePoints(0, 0) =  0.0;
    refTracePoints(1, 0) =  1.0;
    // refTracePoints(2, 0) =  0.0;
    for (int sideIndex=0; sideIndex < numSides; sideIndex++)
    {
      BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideIndex);
      //TODO: Set correct reference cell points.
      // The line below causes the code to crash
      sideBasisCache->setRefCellPoints(refTracePoints);
      int numPoints = sideBasisCache->getPhysicalCubaturePoints().dimension(1);
      cout << "numPoints = " << numPoints << endl;
      if (sideBasisCache.get() == NULL)
        cout << "NULL Side Basis" << endl;

      vector< FieldContainer<double> > computedValues;
      computedValues.resize(numTraceVars);
      for (int i=0; i < numTraceVars; i++)
      {
        computedValues[i].resize(numCells, numPoints);
        solutionValues(computedValues[i], traceTrialIDs[i], sideBasisCache);
      }
      FieldContainer<double> physCubPoints = sideBasisCache->getPhysicalCubaturePoints();

      for (int cellIndex=0;cellIndex < numCells;cellIndex++)
      {
        vtkIdList* edge = vtkIdList::New();
        edge->Initialize();
        for (int i=0; i < numPoints; i++)
        {
          edge->InsertNextId(trace_vertex_count+i);
        }
        trace_ug->InsertNextCell((int)VTK_POLY_LINE, edge);
        edge->Delete();

        for (int pointIndex = 0; pointIndex < numPoints; pointIndex++)
        {
          trace_points->InsertNextPoint(physCubPoints(cellIndex, pointIndex, 0),
              physCubPoints(cellIndex, pointIndex, 1), 0.0);
          for (int varIdx=0; varIdx < numTraceVars; varIdx++)
          {
            traceData[varIdx]->InsertNextValue(computedValues[varIdx](cellIndex, pointIndex));
          }
          trace_vertex_count++;
        }
      }
    }
  }
  trace_ug->SetPoints(trace_points);
  trace_points->Delete();
  for (int varIdx=0; varIdx < numTraceVars; varIdx++)
  {
    trace_ug->GetPointData()->AddArray(traceData[varIdx]);
    traceData[varIdx]->Delete();
  }
  vtkXMLUnstructuredGridWriter* trace_wr = vtkXMLUnstructuredGridWriter::New();
  trace_wr->SetInput(trace_ug);
  trace_ug->Delete();
  trace_wr->SetFileName((filePath+"_trace.vtu").c_str());
  trace_wr->SetDataModeToAscii();
  trace_wr->Update();
  trace_wr->Write();
  trace_wr->Delete();
}

// Write solution to unstructured VTK format
void Solution::writeToVTK(const string& filePath, unsigned int refinementLevel)
{
  vtkUnstructuredGrid* ug = vtkUnstructuredGrid::New();
  // vtkUnstructuredGrid* trace_ug = vtkUnstructuredGrid::New();
  vector<vtkFloatArray*> fieldData;
  // vector<vtkFloatArray*> traceData;
  vtkPoints* points = vtkPoints::New();
  // vtkPoints* trace_points = vtkPoints::New();

  // Get trialIDs
  vector<int> trialIDs = _mesh->bilinearForm()->trialIDs();
  vector<int> fieldTrialIDs;
  // vector<int> traceTrialIDs;

  int spaceDim = 2; // TODO: generalize to 3D...
  int totalCells = _mesh->activeElements().size();
  int totalSubCells = refinementLevel  * refinementLevel * totalCells;
  ug->Allocate(totalSubCells, totalSubCells);
  // trace_ug->Allocate(totalCells, totalCells);
  int numFieldVars = 0;
  // int numTraceVars = 0;
  for (unsigned int i=0; i < trialIDs.size(); i++)
  {
    if (!(_mesh->bilinearForm()->isFluxOrTrace(trialIDs[i])))
    {
      numFieldVars++;
      fieldTrialIDs.push_back(trialIDs[i]);
      fieldData.push_back(vtkFloatArray::New());
    }
    // else
    // {
    //   numTraceVars++;
    //   traceTrialIDs.push_back(trialIDs[i]);
    //   traceData.push_back(vtkFloatArray::New());
    // }
  }

  for (int varIdx = 0; varIdx < numFieldVars; varIdx++)
  {
    fieldData[varIdx]->SetNumberOfComponents(1);
    fieldData[varIdx]->SetName(_mesh->bilinearForm()->trialName(fieldTrialIDs[varIdx]).c_str());
  }
  // for (int varIdx = 0; varIdx < numTraceVars; varIdx++)
  // {
  //   traceData[varIdx]->SetNumberOfComponents(1);
  //   traceData[varIdx]->SetName(_mesh->bilinearForm()->trialName(traceTrialIDs[varIdx]).c_str());
  // }

  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;

  unsigned int total_vertices = 0;
  // unsigned int total_trace_vertices = 0;

  // Loop through Quads, Triangles, etc
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) 
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    Teuchos::RCP<shards::CellTopology> cellTopoPtr = elemTypePtr->cellTopoPtr;
    
    FieldContainer<double> vertexPoints;    
    _mesh->verticesForElementType(vertexPoints,elemTypePtr); //stores vertex points for this element
    FieldContainer<double> physicalCellNodes = _mesh()->physicalCellNodesGlobal(elemTypePtr);
    
    int numCells = physicalCellNodes.dimension(0);
    bool createSideCacheToo = false;
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh, createSideCacheToo));
    
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) 
    {
      int cellID = _mesh->cellID(elemTypePtr, cellIndex, -1); // -1: global cellID
      cellIDs.push_back(cellID);
    }

    int num1DPts = 3; //refinementLevel+1;
    int numPoints = num1DPts * num1DPts;
    FieldContainer<double> refPoints(numPoints,spaceDim);
    for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++){
      for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++){
        int pointIndex = xPointIndex*num1DPts + yPointIndex;
        double x = -1.0 + 2.0*(double)xPointIndex/((double)num1DPts-1.0);
        double y = -1.0 + 2.0*(double)yPointIndex/((double)num1DPts-1.0);
        refPoints(pointIndex,0) = x;
        refPoints(pointIndex,1) = y;
      }
    }
    // int numPoints = 9;
    // FieldContainer<double> refPoints(numPoints,spaceDim);
    // refPoints(0, 0) = -1.0; refPoints(0, 1) = -1.0;
    // refPoints(1, 0) =  1.0; refPoints(1, 1) = -1.0;
    // refPoints(2, 0) =  1.0; refPoints(2, 1) =  1.0;
    // refPoints(3, 0) = -1.0; refPoints(3, 1) =  1.0;
    // refPoints(4, 0) =  0.0; refPoints(4, 1) = -1.0;
    // refPoints(5, 0) =  1.0; refPoints(5, 1) =  0.0;
    // refPoints(6, 0) =  0.0; refPoints(6, 1) =  1.0;
    // refPoints(7, 0) = -1.0; refPoints(7, 1) =  0.0;
    // refPoints(8, 0) =  0.0; refPoints(8, 1) =  0.0;

    int numTracePoints = 3;
    // Assumes num1DPts = 3
    FieldContainer<double> refTracePoints(numTracePoints);
    refTracePoints(0) = -1.0;
    refTracePoints(1) =  0.0;
    refTracePoints(2) =  1.0;
    
    basisCache->setRefCellPoints(refPoints);
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, createSideCacheToo);
    const FieldContainer<double> *physicalPoints = &basisCache->getPhysicalCubaturePoints();
    // basisCache->setRefCellPoints(refTracePoints);
    // // Assumes quad
    // for (int s=0; s < 4; s++)
    // {
    //   const FieldContainer<double> *tracePoints = &basisCache->getPhysicalCubaturePointsForSide(s);
    //   vector< FieldContainer<double> > computedTraceValues;
    //   computedTraceValues.resize(numTraceVars);
    //   for (int i=0; i < numTraceVars; i++)
    //   {
    //     computedTraceValues[i].resize(numCells, numTracePoints);
    //     //TODO: Loop through sides, pass in 1-D set of points and vertexPoints
    //     solutionValues(computedTraceValues[i], elemTypePtr, 
    //         traceTrialIDs[i], *physicalPoints, refPoints, s);
    //   }
    //   for (int cellIndex=0; cellIndex<numCells; cellIndex++ )
    //   {
    //     vtkIdType side[3] = {
    //       total_trace_vertices,
    //       total_trace_vertices+2,
    //       total_trace_vertices+1
    //     };
    //     trace_ug->InsertNextCell((int)VTK_QUADRATIC_EDGE, 3, side);

    //     for (int pointIndex=0; pointIndex < numTracePoints; pointIndex++)
    //     {
    //       trace_points->InsertNextPoint((*tracePoints)(cellIndex, pointIndex, 0),
    //           (*tracePoints)(cellIndex, pointIndex, 1), 0.0);
    //       for (int varIdx=0; varIdx < numTraceVars; varIdx++)
    //       {
    //         traceData[varIdx]->InsertNextValue(computedTraceValues[varIdx](cellIndex, pointIndex));
    //       }
    //       total_trace_vertices++;
    //     }
    //   }
    // }

    vector< FieldContainer<double> > computedValues;
    computedValues.resize(numFieldVars);
    for (int i=0; i < numFieldVars; i++)
    {
      computedValues[i].resize(numCells, numPoints);
      solutionValues(computedValues[i], fieldTrialIDs[i], basisCache);
    }

    for (int cellIndex=0; cellIndex<numCells; cellIndex++ )
    {
      // vtkIdType cell[4] = {
      //   total_trace_vertices,
      //   total_trace_vertices+1,
      //   total_trace_vertices+2,
      //   total_trace_vertices+3
      // };
      // trace_ug->InsertNextCell((int)VTK_QUAD, 4, cell);

      int subcellStartIndex = total_vertices;
      for (int I=0; I < num1DPts-1; I++)
      {
        for (int J=0; J < num1DPts-1; J++)
        {
          int ind1 = subcellStartIndex;
          int ind2 = ind1 + num1DPts;
          int ind3 = ind2 + 1;
          int ind4 = ind1 + 1;
          vtkIdType subCell[4] = {
            ind1, ind2, ind3, ind4};
          ug->InsertNextCell((int)VTK_QUAD, 4, subCell);

          subcellStartIndex++;
        }
        subcellStartIndex++;
      }
      // int subcellStartIndex = total_vertices;
      // vtkIdType base = total_vertices;
      // vtkIdType quadCell1[4] = {
      //   base,
      //   base+4,
      //   base+8,
      //   base+7
      // };
      // vtkIdType quadCell2[4] = {
      //   base+4,
      //   base+1,
      //   base+5,
      //   base+8
      // };
      // vtkIdType quadCell3[4] = {
      //   base+7,
      //   base+8,
      //   base+6,
      //   base+3
      // };
      // vtkIdType quadCell4[4] = {
      //   base+8,
      //   base+5,
      //   base+2,
      //   base+6
      // };
      // vtkIdType quadCell[8] = {
      //   base,
      //   base+1,
      //   base+2,
      //   base+3,
      //   base+4,
      //   base+5,
      //   base+6,
      //   base+7
      // };
      // vtkIdType triCell1[6] = {
      //   base,
      //   base+1,
      //   base+2,
      //   base+4,
      //   base+5,
      //   base+8
      // };
      // vtkIdType triCell2[6] = {
      //   base,
      //   base+2,
      //   base+3,
      //   base+8,
      //   base+6,
      //   base+7
      // };
      // ug->InsertNextCell((int)VTK_QUADRATIC_TRIANGLE, 6, triCell1);
      // ug->InsertNextCell((int)VTK_QUADRATIC_TRIANGLE, 6, triCell2);
      // ug->InsertNextCell((int)VTK_QUADRATIC_QUAD, 8, quadCell);
      // ug->InsertNextCell((int)VTK_QUAD, 4, quadCell1);
      // ug->InsertNextCell((int)VTK_QUAD, 4, quadCell2);
      // ug->InsertNextCell((int)VTK_QUAD, 4, quadCell3);
      // ug->InsertNextCell((int)VTK_QUAD, 4, quadCell4);

      for (int pointIndex = 0; pointIndex < numPoints; pointIndex++)
      {
        points->InsertNextPoint((*physicalPoints)(cellIndex, pointIndex, 0),
                                (*physicalPoints)(cellIndex, pointIndex, 1), 0.0);
        for (int varIdx=0; varIdx < numFieldVars; varIdx++)
        {
          fieldData[varIdx]->InsertNextValue(computedValues[varIdx](cellIndex, pointIndex));
        }
        total_vertices++;
      }
      // for (int pointIndex=0; pointIndex < numTracePoints; pointIndex++)
      // {
      //   trace_points->InsertNextPoint(vertexPoints(cellIndex, pointIndex, 0),
      //                                 vertexPoints(cellIndex, pointIndex, 1), 0.0);
      //   for (int varIdx=0; varIdx < numTraceVars; varIdx++)
      //   {
      //     traceData[varIdx]->InsertNextValue(computedTraceValues[varIdx](cellIndex, pointIndex));
      //   }
      //   total_trace_vertices++;
      // }
    }
  }
  ug->SetPoints(points);
  // trace_ug->SetPoints(trace_points);
  points->Delete();
  // trace_points->Delete();

  for (int varIdx=0; varIdx < numFieldVars; varIdx++)
  {
    ug->GetPointData()->AddArray(fieldData[varIdx]);
    fieldData[varIdx]->Delete();
  }
  // for (int varIdx=0; varIdx < numTraceVars; varIdx++)
  // {
  //   trace_ug->GetPointData()->AddArray(traceData[varIdx]);
  //   traceData[varIdx]->Delete();
  // }

  vtkXMLUnstructuredGridWriter* wr = vtkXMLUnstructuredGridWriter::New();
  // vtkXMLUnstructuredGridWriter* trace_wr = vtkXMLUnstructuredGridWriter::New();
  wr->SetInput(ug);
  // trace_wr->SetInput(trace_ug);
  ug->Delete();
  // trace_ug->Delete();
  wr->SetFileName((filePath+".vtu").c_str());
  // trace_wr->SetFileName((filePath+"_trace.vtu").c_str());
  // wr->SetDataModeToAscii();
  wr->SetDataModeToBinary();
  // trace_wr->SetDataModeToBinary();
  // trace_wr->SetDataModeToAscii();
  wr->Update();
  // trace_wr->Update();
  wr->Write();
  // trace_wr->Write();
  wr->Delete();
  // trace_wr->Delete();
  writeTracesToVTK(filePath);
}
#else

// Write solution to new VTK format
// Sorry this code gets a little ugly, I had to make compromises by not using VTK
// Prefer the VTK version, it is cleaner and probably more efficient
void Solution::writeToVTK(const string& filePath, unsigned int refinementLevel)
{
  // Get trialIDs
  vector<int> trialIDs = _mesh->bilinearForm()->trialIDs();
  vector<int> fieldTrialIDs;

  int spaceDim = 2; // TODO: generalize to 3D...
  int num1DPts = refinementLevel+1;
  int numFieldVars = 0;
  for (unsigned int i=0; i < trialIDs.size(); i++)
  {
    if (!(_mesh->bilinearForm()->isFluxOrTrace(trialIDs[i])))
    {
      numFieldVars++;
      fieldTrialIDs.push_back(trialIDs[i]);
    }
  }

  ofstream fout(filePath.c_str());
  fout << setprecision(15);

  // To store points as we loop through elements
  Teuchos::RCP< ostringstream > pout(new ostringstream);
  *pout << setprecision(15);
  // To store connectivity as we loop through elements
  Teuchos::RCP< ostringstream > conn(new ostringstream);
  // To store point data as we loop through elements
  vector< Teuchos::RCP< ostringstream> > pdata(0);
  pdata.resize(numFieldVars);
  for (unsigned int k=0; k < numFieldVars; k++)
  {
    pdata[k] = Teuchos::RCP< ostringstream > (new ostringstream);
    *pdata[k] << setprecision(15);
  }
  // To store offset data as we loop through elements
  ostringstream offsets;
  offsets << "          ";
  // To store type data as we loop through elements
  ostringstream types;
  types << "          ";

  fout << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << endl;
  fout << "  <UnstructuredGrid>" << endl;

  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;

  unsigned int total_vertices = 0;
  unsigned int total_cells = 0;
  int offsetCount = 0;

  // Loop through Quads, Triangles, etc
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) 
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    Teuchos::RCP<shards::CellTopology> cellTopoPtr = elemTypePtr->cellTopoPtr;
    
    FieldContainer<double> vertexPoints;    
    _mesh->verticesForElementType(vertexPoints,elemTypePtr); //stores vertex points for this element
    FieldContainer<double> physicalCellNodes = _mesh()->physicalCellNodesGlobal(elemTypePtr);
    
    int numCells = physicalCellNodes.dimension(0);
    bool createSideCacheToo = true;
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh, createSideCacheToo));
    
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) 
    {
      int cellID = _mesh->cellID(elemTypePtr, cellIndex, -1); // -1: global cellID
      cellIDs.push_back(cellID);
    }

    int numPoints = num1DPts * num1DPts;
    FieldContainer<double> refPoints(numPoints,spaceDim);
    for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++){
      for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++){
        int pointIndex = xPointIndex*num1DPts + yPointIndex;
        double x = -1.0 + 2.0*(double)xPointIndex/((double)num1DPts-1.0);
        double y = -1.0 + 2.0*(double)yPointIndex/((double)num1DPts-1.0);
        refPoints(pointIndex,0) = x;
        refPoints(pointIndex,1) = y;
      }
    }
    
    basisCache->setRefCellPoints(refPoints);
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, createSideCacheToo);
    const FieldContainer<double> *physicalPoints = &basisCache->getPhysicalCubaturePoints();

    vector< FieldContainer<double> > computedValues;
    computedValues.resize(numFieldVars);
    for (int i=0; i < numFieldVars; i++)
    {
      computedValues[i].resize(numCells, numPoints);
      solutionValues(computedValues[i], fieldTrialIDs[i], basisCache);
    }

    for (int cellIndex=0; cellIndex<numCells; cellIndex++ )
    {
      int subcellStartIndex = total_vertices;
      for (int I=0; I < num1DPts-1; I++)
      {
        for (int J=0; J < num1DPts-1; J++)
        {
          int ind1 = subcellStartIndex;
          int ind2 = ind1 + num1DPts;
          int ind3 = ind2 + 1;
          int ind4 = ind1 + 1;
          *conn << "          " 
            << ind1 <<" "<< ind2 <<" "<< ind3 <<" "<< ind4 << endl;
          subcellStartIndex++;
          total_cells++;
          offsetCount += 4;
          offsets << offsetCount << " ";
          types << 9 << " ";
        }
        subcellStartIndex++;
      }
      for (int pointIndex = 0; pointIndex < num1DPts*num1DPts; pointIndex++)
      {
        *pout << "          " 
          << (*physicalPoints)(cellIndex, pointIndex, 0) << " "
          << (*physicalPoints)(cellIndex, pointIndex, 1) << " "
          << 0 << endl;
        total_vertices++;
      }
      for (int varIdx=0; varIdx < numFieldVars; varIdx++)
      {
        for (int pointIndex = 0; pointIndex < num1DPts*num1DPts; pointIndex++)
        {
          *pdata[varIdx] << "          " 
            << computedValues[varIdx](cellIndex, pointIndex) << endl;
        }
      }
    }
  }
  fout << "    <Piece NumberOfPoints=\"" << total_vertices << "\" "
    << "NumberOfCells=\"" << total_cells << "\">" << endl;
  fout << "      <Points>" << endl;
  fout << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" "
    << "Format=\"ascii\">" << endl;
  fout << pout->str();
  fout << "        </DataArray>" << endl;
  fout << "      </Points>" << endl;
  fout << "      <Cells>" << endl;
  fout << "        <DataArray type=\"Int32\" Name=\"connectivity\" " << "Format=\"ascii\">" << endl;
  fout << conn->str();
  fout << "        </DataArray>" << endl;
  fout << "        <DataArray type=\"Int32\" Name=\"offsets\" " << "Format=\"ascii\">" << endl;
  fout << offsets.str() << endl;
  fout << "        </DataArray>" << endl;
  fout << "        <DataArray type=\"Int32\" Name=\"types\" " << "Format=\"ascii\">" << endl;
  fout << types.str() << endl;
  fout << "        </DataArray>" << endl;
  fout << "      </Cells>" << endl;
  fout << "      <PointData Scalars=\"scalars\">" << endl;
  for (int varIdx = 0; varIdx < numFieldVars; varIdx++)
  {
  fout << "        <DataArray type=\"Float32\" Name=\""
    << _mesh->bilinearForm()->trialName(fieldTrialIDs[varIdx]) << "\" " 
    << "NumberOfComponents=\"1\" Format=\"ascii\">" << endl;
  fout << pdata[varIdx]->str();
  fout << "        </DataArray>" << endl;
  }
  fout << "      </PointData>" << endl;
  fout << "    </Piece>" << endl;
  fout << "  </UnstructuredGrid>" << endl;
  fout << "</VTKFile>" << endl;
  fout.close();
}

#endif
