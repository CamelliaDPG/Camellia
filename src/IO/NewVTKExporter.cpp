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
#include "vtkVersion.h"

void NewVTKExporter::exportFunction(FunctionPtr function, const string& functionName, unsigned int num1DPts)
{
  bool defaultPts = (num1DPts == 0);
  vtkUnstructuredGrid* ug = vtkUnstructuredGrid::New();
  vtkFloatArray* vals = vtkFloatArray::New();
  vtkPoints* points = vtkPoints::New();

  int spaceDim = _mesh->getSpaceDim();
  if (function->rank() == 0)
    vals->SetNumberOfComponents(1);
  else if (function->rank() == 1)
    vals->SetNumberOfComponents(3);
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled function rank");
  vals->SetName(functionName.c_str());

  unsigned int total_vertices = 0;

  for (unsigned cellIndex=0; cellIndex<_mesh->cellCount(); cellIndex++) {
    NewMeshCellPtr cell = _mesh->getCell(cellIndex);
    // Skip the rest of the block if cell is a parent cell
    if (cell->isParent())
      continue;

    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellIndex);

    // unsigned sideCount = cell->topology()->getSideCount();
    // const vector< unsigned > vertices = cell->vertices();
    CellTopoPtr cellTopoPtr = cell->topology();
    unsigned cellTopoKey = cellTopoPtr->getKey();
    int numPoints = 0;
    int pOrder = 4;
    if (defaultPts)
      num1DPts = pow(2.0, pOrder);
    
    if (physicalCellNodes.rank() == 2)
      physicalCellNodes.resize(1,physicalCellNodes.dimension(0), physicalCellNodes.dimension(1));
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(physicalCellNodes, *cellTopoPtr, 1, false) );

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
    const FieldContainer<double> *physicalPoints = &basisCache->getPhysicalCubaturePoints();

    FieldContainer<double> computedValues;
    if (function->rank() == 0)
      computedValues.resize(1, numPoints);
    else if (function->rank() == 1)
      computedValues.resize(1, numPoints, spaceDim);
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled function rank");

    function->values(computedValues, basisCache);

    int subcellStartIndex = total_vertices;
    switch (cellTopoKey)
    {
      case shards::Quadrilateral<4>::key:
        {
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
        }
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellTopoKey unrecognized");
      }
      for (int pointIndex = 0; pointIndex < numPoints; pointIndex++)
      {
        points->InsertNextPoint((*physicalPoints)(0, pointIndex, 0),
          (*physicalPoints)(0, pointIndex, 1), 0.0);
        switch(vals->GetNumberOfComponents())
        {
          case 1:
          vals->InsertNextTuple1(computedValues(0, pointIndex));
          break;
          case 2:
          vals->InsertNextTuple2(computedValues(0, pointIndex, 0), computedValues(0, pointIndex, 1));
          break;
          case 3:
          vals->InsertNextTuple3(computedValues(0, pointIndex, 0), computedValues(0, pointIndex, 1), 0);
          break;
          default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported number of components");
        }
        total_vertices++;
      }
    }

    ug->SetPoints(points);
    points->Delete();

    ug->GetPointData()->AddArray(vals);
    vals->Delete();

    vtkXMLUnstructuredGridWriter* wr = vtkXMLUnstructuredGridWriter::New();
#if VTK_MAJOR_VERSION <= 5
    wr->SetInput(ug);
#else
    wr->SetInputData(ug);
#endif
    ug->Delete();
    wr->SetFileName((functionName+".vtu").c_str());
  // wr->SetDataModeToBinary();
    wr->SetDataModeToAscii();
    wr->Update();
    wr->Write();
    wr->Delete();

    cout << "Wrote " <<  functionName << ".vtu" << endl;
  }

#endif
