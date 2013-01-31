/*
 *  VTKExporter.cpp
 *
 *  Created by Truman Ellis on 12/12/2012.
 *
 */
 
//#include "VTKExporterCamellia.h"
#include "SolutionExporter.h"
#include "CamelliaConfig.h"

#ifdef USE_VTK
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkFloatArray.h"
#include "vtkIntArray.h"
#include "vtkUnstructuredGrid.h"
#include "vtkXMLUnstructuredGridWriter.h"
#include "vtkCellType.h"
#include "vtkIdList.h"

void VTKExporter::exportSolution(const string& filePath, unsigned int num1DPts)
{
  exportFields(filePath, num1DPts);
  exportTraces(filePath, num1DPts);
}

void VTKExporter::exportFields(const string& filePath, unsigned int num1DPts)
{
  bool defaultPts = (num1DPts == 0);

  vtkUnstructuredGrid* ug = vtkUnstructuredGrid::New();
  vector<vtkFloatArray*> fieldData;
  vtkPoints* points = vtkPoints::New();
  vtkIntArray* polyOrderData = vtkIntArray::New();

  // Get trialIDs
  vector<int> fieldTrialIDs = _mesh->bilinearForm()->trialVolumeIDs();
  vector<VarPtr> vars;
  int numVars = fieldTrialIDs.size();

  int spaceDim = 2; // TODO: generalize to 3D...

  for (int varIdx = 0; varIdx < numVars; varIdx++)
  {
    fieldData.push_back(vtkFloatArray::New());
    vars.push_back(_varFactory.trial(fieldTrialIDs[varIdx]));

    bool vectorValued = vars.back()->rank() == 1;
    if (vectorValued)
      fieldData[varIdx]->SetNumberOfComponents(3);
    else
      fieldData[varIdx]->SetNumberOfComponents(1);
    fieldData[varIdx]->SetName(vars.back()->name().c_str());
  }
  polyOrderData->SetNumberOfComponents(1);
  polyOrderData->SetName("Polynomial Order");

  unsigned int total_vertices = 0;

  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;

  // Loop through Quads, Triangles, etc
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) 
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    Teuchos::RCP<shards::CellTopology> cellTopoPtr = elemTypePtr->cellTopoPtr;
    
    FieldContainer<double> vertexPoints;    
    _mesh->verticesForElementType(vertexPoints,elemTypePtr); //stores vertex points for this element
    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesGlobal(elemTypePtr);
    
    int numCells = physicalCellNodes.dimension(0);
    bool createSideCacheToo = false;
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh, createSideCacheToo));
    
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) 
    {
      int cellID = _mesh->cellID(elemTypePtr, cellIndex, -1); // -1: global cellID
      cellIDs.push_back(cellID);
    }

    int pOrder = _mesh->cellPolyOrder(cellIDs[0]);

    int numVertices = vertexPoints.dimension(1);
    unsigned cellTopoKey = cellTopoPtr->getKey();
    int numPoints = 0;
    if (defaultPts)
      num1DPts = pow(2.0, pOrder-1);

    switch (cellTopoKey)
    {
      case shards::Quadrilateral<4>::key:
        numPoints = num1DPts*num1DPts;
        break;
      case shards::Triangle<3>::key:
        for (int i=1; i <= num1DPts; i++)
          numPoints += i;
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
    }

    FieldContainer<double> refPoints(numPoints,spaceDim);
    switch (cellTopoKey)
    {
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
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
    }
    
    basisCache->setRefCellPoints(refPoints);
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, createSideCacheToo);
    const FieldContainer<double> *physicalPoints = &basisCache->getPhysicalCubaturePoints();

    vector< FieldContainer<double> > computedValues;
    // computedValues.resize(numVars);
    for (int i=0; i < numVars; i++)
    {
      int numberComponents = fieldData[i]->GetNumberOfComponents();
      FieldContainer<double> values;
      if (numberComponents == 1)
        values.resize(numCells, numPoints);
      else
        values.resize(numCells, numPoints, spaceDim);

      _solution->solutionValues(values, fieldTrialIDs[i], basisCache);
      //_solution->solutionValues(values, elemTypePtr, fieldTrialIDs[i], *physicalPoints);
      computedValues.push_back(values);
    }
    // cout << "After All" << endl;

    for (int cellIndex=0; cellIndex<numCells; cellIndex++ )
    {
      int subcellStartIndex = total_vertices;
      switch (cellTopoKey)
      {
        case shards::Quadrilateral<4>::key:
          for (int j=0; j < num1DPts-1; j++)
          {
            for (int i=0; i < num1DPts-1; i++)
            {
              int ind1 = subcellStartIndex;
              int ind2 = ind1 + 1;
              int ind3 = ind2 + num1DPts;
              int ind4 = ind1 + num1DPts;
              vtkIdType subCell[4] = {
                ind1, ind2, ind3, ind4};
              ug->InsertNextCell((int)VTK_QUAD, 4, subCell);
              polyOrderData->InsertNextValue(pOrder-1);

              subcellStartIndex++;
            }
            subcellStartIndex++;
          }
          break;
        case shards::Triangle<3>::key:
          for (int j=0; j < num1DPts-1; j++)
          {
            for (int i=0; i < num1DPts-1-j; i++)
            {
              int ind1 = subcellStartIndex;
              int ind2 = ind1 + 1;
              int ind3 = ind1 + num1DPts-j;
              vtkIdType subCell[3] = {
                ind1, ind2, ind3};
              ug->InsertNextCell((int)VTK_TRIANGLE, 3, subCell);
              polyOrderData->InsertNextValue(pOrder-1);

              if (i < num1DPts-2-j)
              {
                int ind1 = subcellStartIndex+1;
                int ind2 = ind1 + num1DPts - j;
                int ind3 = ind1 + num1DPts -j - 1;
                vtkIdType subCell[3] = {
                  ind1, ind2, ind3};
                ug->InsertNextCell((int)VTK_TRIANGLE, 3, subCell);
                polyOrderData->InsertNextValue(pOrder-1);
              }

              subcellStartIndex++;
            }
            subcellStartIndex++;
          }
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
      }

      for (int pointIndex = 0; pointIndex < numPoints; pointIndex++)
      {
        points->InsertNextPoint((*physicalPoints)(cellIndex, pointIndex, 0),
                                (*physicalPoints)(cellIndex, pointIndex, 1), 0.0);
        for (int varIdx=0; varIdx < numVars; varIdx++)
        {
          // fieldData[varIdx]->InsertNextValue(computedValues[varIdx](cellIndex, pointIndex));
          switch(fieldData[varIdx]->GetNumberOfComponents())
          {
            case 1:
              fieldData[varIdx]->InsertNextTuple1(computedValues[varIdx](cellIndex, pointIndex));
              break;
            case 2:
              fieldData[varIdx]->InsertNextTuple2(computedValues[varIdx](cellIndex, pointIndex, 0), computedValues[varIdx](cellIndex, pointIndex, 1));
              break;
            case 3:
              fieldData[varIdx]->InsertNextTuple3(computedValues[varIdx](cellIndex, pointIndex, 0), computedValues[varIdx](cellIndex, pointIndex, 1), 0);
              break;
            default:
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported number of components");
          }
        }
        total_vertices++;
      }
    }
  }

  ug->SetPoints(points);
  points->Delete();

  for (int varIdx=0; varIdx < numVars; varIdx++)
  {
    ug->GetPointData()->AddArray(fieldData[varIdx]);
    fieldData[varIdx]->Delete();
  }
  ug->GetCellData()->AddArray(polyOrderData);

  vtkXMLUnstructuredGridWriter* wr = vtkXMLUnstructuredGridWriter::New();
  wr->SetInput(ug);
  ug->Delete();
  wr->SetFileName((filePath+".vtu").c_str());
  wr->SetDataModeToBinary();
  wr->Update();
  wr->Write();
  wr->Delete();

  cout << "Wrote Field Variables to " << filePath << ".vtu" << endl;
}

void VTKExporter::exportTraces(const string& filePath, unsigned int num1DPts)
{
  bool defaultPts = (num1DPts == 0);

  vtkUnstructuredGrid* trace_ug = vtkUnstructuredGrid::New();
  vector<vtkFloatArray*> traceData;
  vtkPoints* points = vtkPoints::New();
  vtkIntArray* polyOrderData = vtkIntArray::New();

  // Get trialIDs
  vector<int> traceTrialIDs = _mesh->bilinearForm()->trialBoundaryIDs();
  vector<VarPtr> vars;
  int numVars = traceTrialIDs.size();

  for (int varIdx = 0; varIdx < numVars; varIdx++)
  {
    traceData.push_back(vtkFloatArray::New());
    vars.push_back(_varFactory.trial(traceTrialIDs[varIdx]));

    traceData[varIdx]->SetNumberOfComponents(1);
    traceData[varIdx]->SetName(vars.back()->name().c_str());
  }
  unsigned int total_vertices = 0;

  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;

  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) 
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    Teuchos::RCP<shards::CellTopology> cellTopoPtr = elemTypePtr->cellTopoPtr;
    int numSides = cellTopoPtr->getSideCount();
    
    FieldContainer<double> vertexPoints;    
    _mesh->verticesForElementType(vertexPoints,elemTypePtr); //stores vertex points for this element
    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesGlobal(elemTypePtr);
    int numCells = physicalCellNodes.dimension(0);
    
    bool createSideCacheToo = true;
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh, createSideCacheToo));
    
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) 
    {
      int cellID = _mesh->cellID(elemTypePtr, cellIndex, -1); // -1: global cellID
      cellIDs.push_back(cellID);
    }

    int pOrder = _mesh->cellPolyOrder(cellIDs[0]);
    int numVertices = vertexPoints.dimension(1);
    unsigned cellTopoKey = cellTopoPtr->getKey();
    if (defaultPts)
      num1DPts = pow(2.0, pOrder-1);

    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, true); // true: create side caches

    FieldContainer<double> refPoints(num1DPts,1);
    for (int i=0; i < num1DPts; i++)
    {
      double x = -1.0 + 2.0*(double(i)/double(num1DPts-1));
      refPoints(i,0) = x;
    }

    for (int sideIndex=0; sideIndex < numSides; sideIndex++)
    {
      BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideIndex);
      sideBasisCache->setRefCellPoints(refPoints);
      int numPoints = sideBasisCache->getPhysicalCubaturePoints().dimension(1);
      if (sideBasisCache.get() == NULL)
        cout << "NULL Side Basis" << endl;

      vector< FieldContainer<double> > computedValues;
      computedValues.resize(numVars);
      for (int i=0; i < numVars; i++)
      {
        computedValues[i].resize(numCells, numPoints);
        _solution->solutionValues(computedValues[i], traceTrialIDs[i], sideBasisCache);
      }
      const FieldContainer<double> *physicalPoints = &sideBasisCache->getPhysicalCubaturePoints();
      // FieldContainer<double> physCubPoints = sideBasisCache->getPhysicalCubaturePoints();
      // cout << " physPoints dim = " << physicalPoints->dimension(0) << " " << physicalPoints->dimension(1)<< endl;

      for (int cellIndex=0;cellIndex < numCells;cellIndex++)
      {
        vtkIdList* edge = vtkIdList::New();
        edge->Initialize();
        for (int i=0; i < numPoints; i++)
        {
          edge->InsertNextId(total_vertices+i);
        }
        trace_ug->InsertNextCell((int)VTK_POLY_LINE, edge);
        edge->Delete();

        // cout << "Physical Points: " << endl;
        for (int pointIndex = 0; pointIndex < numPoints; pointIndex++)
        {
          points->InsertNextPoint((*physicalPoints)(cellIndex, pointIndex, 0),
              (*physicalPoints)(cellIndex, pointIndex, 1), 0.0);
          // cout << (*physicalPoints)(cellIndex, pointIndex, 0)<<" "<<(*physicalPoints)(cellIndex, pointIndex, 1) << endl;
          // points->InsertNextPoint(physCubPoints(cellIndex, pointIndex, 0),
          //     physCubPoints(cellIndex, pointIndex, 1), 0.0);
          for (int varIdx=0; varIdx < numVars; varIdx++)
          {
            traceData[varIdx]->InsertNextValue(computedValues[varIdx](cellIndex, pointIndex));
          }
          total_vertices++;
        }
      }
    }
  }
  trace_ug->SetPoints(points);
  points->Delete();
  for (int varIdx=0; varIdx < numVars; varIdx++)
  {
    trace_ug->GetPointData()->AddArray(traceData[varIdx]);
    traceData[varIdx]->Delete();
  }
  vtkXMLUnstructuredGridWriter* trace_wr = vtkXMLUnstructuredGridWriter::New();
  trace_wr->SetInput(trace_ug);
  trace_ug->Delete();
  trace_wr->SetFileName(("trace_"+filePath+".vtu").c_str());
  trace_wr->SetDataModeToBinary();
  trace_wr->Update();
  trace_wr->Write();
  trace_wr->Delete();

  cout << "Wrote Trace Variables to " << "trace_"+filePath << ".vtu" << endl;
}

void VTKExporter::exportFunction(FunctionPtr function, const string& functionName, unsigned int num1DPts)
{
  bool defaultPts = (num1DPts == 0);
  vtkUnstructuredGrid* ug = vtkUnstructuredGrid::New();
  vtkFloatArray* vals = vtkFloatArray::New();
  vtkPoints* points = vtkPoints::New();

  int spaceDim = 2; // TODO: generalize to 3D...
  if (function->rank() == 0)
    vals->SetNumberOfComponents(1);
  else if (function->rank() == 1)
    vals->SetNumberOfComponents(3);
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled function rank");
  vals->SetName(functionName.c_str());

  unsigned int total_vertices = 0;

  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;

  // Loop through Quads, Triangles, etc
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) 
  {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    Teuchos::RCP<shards::CellTopology> cellTopoPtr = elemTypePtr->cellTopoPtr;
    
    FieldContainer<double> vertexPoints;    
    _mesh->verticesForElementType(vertexPoints,elemTypePtr); //stores vertex points for this element
    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesGlobal(elemTypePtr);
    
    int numCells = physicalCellNodes.dimension(0);
    bool createSideCacheToo = false;
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh, createSideCacheToo));
    
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) 
    {
      int cellID = _mesh->cellID(elemTypePtr, cellIndex, -1); // -1: global cellID
      cellIDs.push_back(cellID);
    }

    int pOrder = _mesh->cellPolyOrder(cellIDs[0]);

    int numVertices = vertexPoints.dimension(1);
    unsigned cellTopoKey = cellTopoPtr->getKey();
    int numPoints = 0;
    if (defaultPts)
      num1DPts = pow(2.0, pOrder-1);

    switch (cellTopoKey)
    {
      case shards::Quadrilateral<4>::key:
        numPoints = num1DPts*num1DPts;
        break;
      case shards::Triangle<3>::key:
        for (int i=1; i <= num1DPts; i++)
          numPoints += i;
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
    }

    FieldContainer<double> refPoints(numPoints,spaceDim);
    switch (cellTopoKey)
    {
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
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
    }
    
    basisCache->setRefCellPoints(refPoints);
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, createSideCacheToo);
    const FieldContainer<double> *physicalPoints = &basisCache->getPhysicalCubaturePoints();

    FieldContainer<double> computedValues;
    if (function->rank() == 0)
      computedValues.resize(numCells, numPoints);
    else if (function->rank() == 1)
      computedValues.resize(numCells, numPoints, spaceDim);
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled function rank");

    function->values(computedValues, basisCache);

    for (int cellIndex=0; cellIndex<numCells; cellIndex++ )
    {
      int subcellStartIndex = total_vertices;
      switch (cellTopoKey)
      {
        case shards::Quadrilateral<4>::key:
          for (int j=0; j < num1DPts-1; j++)
          {
            for (int i=0; i < num1DPts-1; i++)
            {
              int ind1 = subcellStartIndex;
              int ind2 = ind1 + 1;
              int ind3 = ind2 + num1DPts;
              int ind4 = ind1 + num1DPts;
              vtkIdType subCell[4] = {
                ind1, ind2, ind3, ind4};
              ug->InsertNextCell((int)VTK_QUAD, 4, subCell);

              subcellStartIndex++;
            }
            subcellStartIndex++;
          }
          break;
        case shards::Triangle<3>::key:
          for (int j=0; j < num1DPts-1; j++)
          {
            for (int i=0; i < num1DPts-1-j; i++)
            {
              int ind1 = subcellStartIndex;
              int ind2 = ind1 + 1;
              int ind3 = ind1 + num1DPts-j;
              vtkIdType subCell[3] = {
                ind1, ind2, ind3};
              ug->InsertNextCell((int)VTK_TRIANGLE, 3, subCell);

              if (i < num1DPts-2-j)
              {
                int ind1 = subcellStartIndex+1;
                int ind2 = ind1 + num1DPts - j;
                int ind3 = ind1 + num1DPts -j - 1;
                vtkIdType subCell[3] = {
                  ind1, ind2, ind3};
                ug->InsertNextCell((int)VTK_TRIANGLE, 3, subCell);
              }

              subcellStartIndex++;
            }
            subcellStartIndex++;
          }
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
      }

      for (int pointIndex = 0; pointIndex < numPoints; pointIndex++)
      {
        points->InsertNextPoint((*physicalPoints)(cellIndex, pointIndex, 0),
                                (*physicalPoints)(cellIndex, pointIndex, 1), 0.0);
        switch(vals->GetNumberOfComponents())
        {
          case 1:
            vals->InsertNextTuple1(computedValues(cellIndex, pointIndex));
            break;
          case 2:
            vals->InsertNextTuple2(computedValues(cellIndex, pointIndex, 0), computedValues(cellIndex, pointIndex, 1));
            break;
          case 3:
            vals->InsertNextTuple3(computedValues(cellIndex, pointIndex, 0), computedValues(cellIndex, pointIndex, 1), 0);
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported number of components");
        }
        total_vertices++;
      }
    }
  }

  ug->SetPoints(points);
  points->Delete();

  ug->GetPointData()->AddArray(vals);
  vals->Delete();

  vtkXMLUnstructuredGridWriter* wr = vtkXMLUnstructuredGridWriter::New();
  wr->SetInput(ug);
  ug->Delete();
  wr->SetFileName((functionName+".vtu").c_str());
  // wr->SetDataModeToBinary();
  wr->SetDataModeToBinary();
  wr->Update();
  wr->Write();
  wr->Delete();

  cout << "Wrote " <<  functionName << ".vtu" << endl;
}

#endif
