//TODO: Vector support, boundary functions, adaptive subdivision
// pass in vector of functions, override creates vector of one, another override packages solutions into vector
#include "SolutionExporter.h"
#include "CamelliaConfig.h"

#ifdef USE_XDMF
#include <Xdmf.h>
void XDMFExporter::exportFunction(FunctionPtr function, const string& functionName, set<GlobalIndexType> cellIndices, unsigned int num1DPts)
{
  XdmfDOM         dom;
  XdmfRoot        root;
  XdmfDomain      domain;
  XdmfGrid        grid;
  XdmfTime        time;
  XdmfTopology    *topology;
  XdmfGeometry    *geometry;
  XdmfAttribute   nodedata;
  XdmfAttribute   celldata;
  XdmfArray       *ptArray;
  XdmfArray       *connArray;
  XdmfArray       *valArray;

  root.SetDOM(&dom);
  root.SetVersion(2.0);
  root.Build();
  // Domain
  root.Insert(&domain);
  // Grid
  grid.SetName("Grid");
  domain.Insert(&grid);
  time.SetTimeType(XDMF_TIME_SINGLE);
  time.SetValue(0);
  grid.Insert(&time);
  // Topology
  topology = grid.GetTopology();
  topology->SetTopologyType(XDMF_MIXED);
  
  bool defaultPts = (num1DPts == 0);
  int pOrder = 4;
  if (defaultPts)
    if (pOrder < 2)
      num1DPts = 2;
    else
      num1DPts = 2*pOrder+2;
      // num1DPts = pOrder+1;

  int spaceDim = _mesh->getSpaceDim();
  // if (function->rank() == 0)
//     vals->SetNumberOfComponents(1);
//   else if (function->rank() == 1)
//     vals->SetNumberOfComponents(3);
//   else
//     TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled function rank");
//   vals->SetName(functionName.c_str());

  unsigned int total_vertices = 0;
  
  if (cellIndices.size()==0) cellIndices = _mesh->getActiveCellIndices();
  int numLines=0;
  int numTriangles=0;
  int numQuads=0;
  int numHexas=0;
  // set<GlobalIndexType> cellIDs = mesh->getTopology()->getActiveCellIndices();
  for (set<GlobalIndexType>::iterator cellIt = cellIndices.begin(); cellIt != cellIndices.end(); cellIt++) {
    CellPtr cell = _mesh->getCell(*cellIt);
    if (cell->topology()->getKey() == shards::Line<2>::key) numLines++;
    if (cell->topology()->getKey() == shards::Triangle<3>::key) numTriangles++;
    if (cell->topology()->getKey() == shards::Quadrilateral<4>::key) numQuads++;
    if (cell->topology()->getKey() == shards::Hexahedron<8>::key) numHexas++;
    // cout << _mesh->cellPolyOrder(*cellIt) << endl;
  }
  int totalSubLines = (num1DPts-1)*numLines;
  int totalSubTriangles = (num1DPts-1)*(num1DPts-1)*numTriangles;
  int totalSubQuads = (num1DPts-1)*(num1DPts-1)*numQuads;
  int totalSubHexas = (num1DPts-1)*(num1DPts-1)*(num1DPts-1)*numHexas;
  int totalSubcells = totalSubLines + totalSubTriangles + totalSubQuads + totalSubHexas;
  int totalPts = num1DPts*numLines + num1DPts*(num1DPts+1)/2*numTriangles + num1DPts*num1DPts*numQuads + num1DPts*num1DPts*num1DPts*numHexas;

  // Topology
  topology = grid.GetTopology();
  topology->SetTopologyType(XDMF_MIXED);
  if (spaceDim == 1)
    topology->SetNumberOfElements(numLines);
  else
    topology->SetNumberOfElements(totalSubcells);
  connArray = topology->GetConnectivity();
  if (spaceDim == 1)
    connArray->SetNumberOfElements(2*numLines+totalPts);
  else
    connArray->SetNumberOfElements(totalSubcells + 3*totalSubTriangles + 4*totalSubQuads + 8*totalSubHexas);
  // Geometry
  geometry = grid.GetGeometry();
  if (spaceDim == 1)
    geometry->SetGeometryType(XDMF_GEOMETRY_XY);
  else if (spaceDim == 2)
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
  nodedata.SetName(functionName.c_str());
  nodedata.SetAttributeCenter(XDMF_ATTRIBUTE_CENTER_NODE);
  nodedata.SetAttributeType(XDMF_ATTRIBUTE_TYPE_SCALAR);
  valArray = nodedata.GetValues();
  valArray->SetNumberType(XDMF_FLOAT64_TYPE);
  valArray->SetNumberOfElements(totalPts);

  int connIndex = 0;
  int ptIndex = 0;
  int valIndex = 0;

  
  for (set<GlobalIndexType>::iterator cellIt = cellIndices.begin(); cellIt != cellIndices.end(); cellIt++) {
    GlobalIndexType cellIndex = *cellIt;
    CellPtr cell = _mesh->getCell(cellIndex);

    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellIndex);

    CellTopoPtr cellTopoPtr = cell->topology();
    int numPoints = 0;
    // int pOrder = 4;
    // if (defaultPts)
    //   num1DPts = pow(pOrder, 2)+1;
    
    if (physicalCellNodes.rank() == 2)
      physicalCellNodes.resize(1,physicalCellNodes.dimension(0), physicalCellNodes.dimension(1));
    bool createSideCache = function->boundaryValueOnly();
    
    BasisCachePtr volumeBasisCache = Teuchos::rcp( new BasisCache(*cellTopoPtr, 1, createSideCache) );
    volumeBasisCache->setPhysicalCellNodes(physicalCellNodes, vector<GlobalIndexType>(1,cellIndex), createSideCache);

    int numSides = createSideCache ? cellTopoPtr->getSideCount() : 1;
    
    int sideDim = spaceDim - 1;
    
    for (int sideOrdinal = 0; sideOrdinal < numSides; sideOrdinal++) {
      shards::CellTopology topo = createSideCache ? cellTopoPtr->getBaseCellTopologyData(sideDim, sideOrdinal) : *cellTopoPtr;
      unsigned cellTopoKey = topo.getKey();
      
      BasisCachePtr basisCache = createSideCache ? volumeBasisCache->getSideBasisCache(sideOrdinal) : volumeBasisCache;
      
      unsigned domainDim = createSideCache ? sideDim : spaceDim;
      
      switch (cellTopoKey)
      {
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
      switch (cellTopoKey)
      {
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
              // vtkIdType subCell[4] = {ind1, ind2, ind3, ind4};
              // ug->InsertNextCell((int)VTK_QUAD, 4, subCell);
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
              // vtkIdType subCell[3] = {ind1, ind2, ind3};
              // ug->InsertNextCell((int)VTK_TRIANGLE, 3, subCell);

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
                // vtkIdType subCell[3] = {ind1, ind2, ind3};
                // ug->InsertNextCell((int)VTK_TRIANGLE, 3, subCell);
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
                // vtkIdType subCell[8] = {ind1, ind2, ind3, ind4, ind5, ind6, ind7, ind8};
                // ug->InsertNextCell((int)VTK_HEXAHEDRON, 8, subCell);
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
            // points->InsertNextPoint((*physicalPoints)(0, pointIndex, 0), 0.0, 0.0);
          }
          else if (spaceDim == 2)
          {
            ptArray->SetValue(ptIndex, (*physicalPoints)(0, pointIndex, 0));
            ptIndex++;
            ptArray->SetValue(ptIndex, (*physicalPoints)(0, pointIndex, 1));
            ptIndex++;
          }
            // points->InsertNextPoint((*physicalPoints)(0, pointIndex, 0),
            //   (*physicalPoints)(0, pointIndex, 1), 0.0);
          else
          {
            ptArray->SetValue(ptIndex, (*physicalPoints)(0, pointIndex, 0));
            ptIndex++;
            ptArray->SetValue(ptIndex, (*physicalPoints)(0, pointIndex, 1));
            ptIndex++;
            ptArray->SetValue(ptIndex, (*physicalPoints)(0, pointIndex, 2));
            ptIndex++;
          }
          valArray->SetValue(valIndex, computedValues(0, pointIndex));
          valIndex++;
            // points->InsertNextPoint((*physicalPoints)(0, pointIndex, 0),
            //   (*physicalPoints)(0, pointIndex, 1), (*physicalPoints)(0, pointIndex, 2));
          // switch(vals->GetNumberOfComponents())
          // {
          //   case 1:
          //   // vals->InsertNextTuple1(computedValues(0, pointIndex));
          //   break;
          //   case 2:
          //   // vals->InsertNextTuple2(computedValues(0, pointIndex, 0), computedValues(0, pointIndex, 1));
          //   break;
          //   case 3:
          //   // vals->InsertNextTuple3(computedValues(0, pointIndex, 0), computedValues(0, pointIndex, 1), computedValues(0, pointIndex, 2));
          //   break;
          //   default:
          //   TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported number of components");
          // }
          total_vertices++;
        }
      }
    }
    connArray->SetHeavyDataSetName((functionName+".h5:/Conns").c_str());
    ptArray->SetHeavyDataSetName((functionName+".h5:/Points").c_str());
    valArray->SetHeavyDataSetName((functionName+".h5:/NodeData").c_str());
    // Attach and Write
    grid.Insert(&nodedata);
    // grid.Insert(&celldata);
    // Build is recursive ... it will be called on all of the child nodes.
    // This updates the DOM and writes the HDF5
    root.Build();
    // Write the XML
    dom.Write((functionName+".xmf").c_str());

    cout << "Wrote " <<  functionName << ".xmf" << endl;

//     ug->SetPoints(points);
//     points->Delete();

//     ug->GetPointData()->AddArray(vals);
//     vals->Delete();

//     vtkXMLUnstructuredGridWriter* wr = vtkXMLUnstructuredGridWriter::New();
// #if VTK_MAJOR_VERSION <= 5
//     wr->SetInput(ug);
// #else
//     wr->SetInputData(ug);
// #endif
//     ug->Delete();
//     wr->SetFileName((functionName+".vtu").c_str());
//     wr->SetDataModeToBinary();
//     // wr->SetDataModeToAscii();
//     wr->Update();
//     wr->Write();
//     wr->Delete();

//     cout << "Wrote " <<  functionName << ".vtu" << endl;
//   }
}
#endif