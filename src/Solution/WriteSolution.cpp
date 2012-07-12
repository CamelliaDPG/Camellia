#include "Solution.h"

#include "vtkVersion.h"
// #define USE_VTK
#ifdef USE_VTK
#include "vtkFloatArray.h"

// Write solution to unstructured VTK format
void Solution::writeToVTK(const string& filePath, unsigned int refinementLevel)
{
  int spaceDim = 2; // TODO: generalize to 3D...
  int num1DPts = 2;
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  // Loop through quads, triangles, etc
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) 
  { 
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    Teuchos::RCP<shards::CellTopology> cellTopoPtr = elemTypePtr->cellTopoPtr;
    
    FieldContainer<double> vertexPoints, physPoints;    
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

    int numPoints = num1DPts * num1DPts;
    FieldContainer<double> refPoints(numPoints,spaceDim);
    for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++)
    {
      for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++)
      {
        int pointIndex = yPointIndex*num1DPts + 2*xPointIndex;
        double x = -1.0 + 2.0*(double)xPointIndex/((double)num1DPts-1.0);
        double y = -1.0 + 2.0*(double)yPointIndex/((double)num1DPts-1.0);
        refPoints(pointIndex,0) = x;
        refPoints(pointIndex,1) = y;
      }
    }
    
    basisCache->setRefCellPoints(refPoints);
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, createSideCacheToo);
    FieldContainer<double> computedValues(numCells,numPoints);

    // for (int cellIndex=0; cellIndex<numCells; cellIndex++ ) {
    //   for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++){
    
    // this->solutionValues(computedValues, trialID, basisCache);
    // const FieldContainer<double> *physicalPoints = &basisCache->getPhysicalCubaturePoints();
  }
}
#else

// Write solution to new VTK format
// Sorry this code gets a little ugly, I had to make compromises by not using VTK
// Prefer the VTK version, it is cleaner and probably more efficient
void Solution::writeToVTK(const string& filePath, unsigned int refinementLevel)
{
  vtkVersion* version = vtkVersion::New();
  cout << "VTK Version: " << version->GetVTKVersion() << endl;
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
    
    FieldContainer<double> vertexPoints, physPoints;    
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

#if 0
// Write solution to new VTK format
void Solution::writeToVTK(const string& filePath){

  // Get number total number of vertices
  vector< ElementPtr > elems = _mesh->activeElements();
  vector< ElementPtr >::iterator elem_it;
  unsigned int total_vertices = 0;
  for (elem_it = elems.begin(); elem_it != elems.end(); ++elem_it){
    total_vertices += _mesh->vertexIndicesForCell((*elem_it)->cellID()).size();
  }
  // Get number of cells
  unsigned int total_cells = _mesh->activeElements().size();

  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  fout << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << endl;
  fout << "  <UnstructuredGrid>" << endl;
  fout << "    <Piece NumberOfPoints=\"" << total_vertices << "\" "
    << "NumberOfCells=\"" << total_cells << "\">" << endl;
  fout << "      <Points>" << endl;
  fout << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" "
    << "Format=\"ascii\">" << endl;
  // Loop over cells and print vertices
  for (elem_it = elems.begin(); elem_it != elems.end(); ++elem_it)
  {
    FieldContainer< double > vertices;
    _mesh->verticesForCell(vertices, (*elem_it)->cellID());
    for (unsigned int v = 0; v < vertices.dimension(0); ++v)
    {
      fout << "          ";
      for (unsigned int d = 0; d < vertices.dimension(1); ++d)
        fout << vertices(v, d) << " ";
      fout << "0" << endl;
    }
  }
  fout << "        </DataArray>" << endl;
  fout << "      </Points>" << endl;
  fout << "      <Cells>" << endl;
  fout << "        <DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">" << endl;
  unsigned int connectivity_count = 0;
  for (elem_it = elems.begin(); elem_it != elems.end(); ++elem_it)
  {
    FieldContainer< double > vertices;
    _mesh->verticesForCell(vertices, (*elem_it)->cellID());
    fout << "          ";
    for (unsigned int v = 0; v < vertices.dimension(0); ++v)
    {
      fout << connectivity_count++ << " ";
    }
    fout << endl;
  }
  fout << "        </DataArray>" << endl;
  fout << "        <DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">" << endl;
  fout << "          ";
  unsigned int offset_count = 0;
  for (elem_it = elems.begin(); elem_it != elems.end(); ++elem_it){
    offset_count += _mesh->vertexIndicesForCell((*elem_it)->cellID()).size();
    fout << offset_count << " ";
  }
  fout << endl;
  fout << "        </DataArray>" << endl;
  fout << "        <DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">" << endl;
  fout << "          ";
  for (elem_it = elems.begin(); elem_it != elems.end(); ++elem_it){
    unsigned int num_verts = _mesh->vertexIndicesForCell((*elem_it)->cellID()).size();
    unsigned int type = 0;
    if (num_verts == 4)
      type = 9;
    else if (num_verts == 3)
      type = 5;
    fout << type << " ";
  }
  fout << endl;
  fout << "        </DataArray>" << endl;
  fout << "      </Cells>" << endl;
  fout << "      <PointData Scalars=\"scalars\">" << endl;
  vector<int> tIDs = _mesh->bilinearForm()->trialIDs();
  for (int i=0; i != tIDs.size(); i++)
  {
    if (!(_mesh->bilinearForm()->isFluxOrTrace(tIDs[i])))
    {
      fout << "        <DataArray type=\"Float32\" Name=\""
        << _mesh->bilinearForm()->trialName(tIDs[i]) << "\" " 
        << "NumberOfComponents=\"1\" Format=\"ascii\">" << endl;
      // Loop over cells and print values
      for (elem_it = elems.begin(); elem_it != elems.end(); ++elem_it)
      {
        // FieldContainer< double > vertices;
        // _mesh->verticesForCell(vertices, (*elem_it)->cellID());
        // FieldContainer< double > values;
        // solutionValues(values, tIDs[i], vertices);
        // for (unsigned int v = 0; v < values; ++v)
        // {
        //   fout << "          ";
        //   for (unsigned int d = 0; d < vertices.dimension(1); ++d)
        //     fout << vertices(v, d) << " ";
        //   fout << "0" << endl;
        // }
      }
      fout << "        </DataArray>" << endl;
    }
  }
  fout << "      </PointData>" << endl;
  fout << "    </Piece>" << endl;
  fout << "  </UnstructuredGrid>" << endl;
  fout << "</VTKFile>" << endl;
  fout.close();
}
#endif

#endif
