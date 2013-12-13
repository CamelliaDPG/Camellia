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
    cout << physicalCellNodes << endl;

    // unsigned sideCount = cell->topology()->getSideCount();
    // const vector< unsigned > vertices = cell->vertices();
    CellTopoPtr cellTopoPtr = cell->topology();
    unsigned cellTopoKey = cellTopoPtr->getKey();
    int numPoints = 0;
    int pOrder = 2;
    if (defaultPts)
      num1DPts = pow(2.0, pOrder-1);

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

    // basisCache->setRefCellPoints(refPoints);
  }

  cout << "Wrote " <<  functionName << ".vtu" << endl;
}

#endif
