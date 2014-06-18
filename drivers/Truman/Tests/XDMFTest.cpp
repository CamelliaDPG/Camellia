#include <iostream>
#include <Xdmf.h>

int main()
{
  int nel = 4;
  int npts = 9;

  XdmfDOM         dom;
  XdmfRoot        root;
  XdmfDomain      domain;
  XdmfGrid        grid;
  XdmfTime        time;
  XdmfTopology    *topology;
  XdmfGeometry    *geometry;
  XdmfAttribute   nodedata;
  XdmfAttribute   celldata;
  XdmfArray       *array;
  XdmfInt32       *Conns;
  // XdmfInt32 Conns[1][1] =
  // {
  //   {0},
  // };

  root.SetDOM(&dom);
  root.SetVersion(2.0);
  root.Build();
  // Domain
  root.Insert(&domain);
  // Grid
  grid.SetName("Demonstration Grid");
  domain.Insert(&grid);
  time.SetTimeType(XDMF_TIME_SINGLE);
  time.SetValue(0);
  grid.Insert(&time);
  // Topology
  topology = grid.GetTopology();
  topology->SetTopologyType(XDMF_MIXED);
  topology->SetNumberOfElements(nel+1);
  array = topology->GetConnectivity();
  array->SetNumberOfElements(nel * 4 + 4 + 1 + 3);
  // array->SetValues(0, Conns, nel * 4);
  array->SetValue(0, 5);
  array->SetValue(1, 0);
  array->SetValue(2, 1);
  array->SetValue(3, 4);
  array->SetValue(4, 3);
  array->SetValue(5, 5);
  array->SetValue(6, 1);
  array->SetValue(7, 2);
  array->SetValue(8, 5);
  array->SetValue(9, 4);
  array->SetValue(10, 5);
  array->SetValue(11, 3);
  array->SetValue(12, 4);
  array->SetValue(13, 7);
  array->SetValue(14, 6);
  array->SetValue(15, 5);
  array->SetValue(16, 4);
  array->SetValue(17, 5);
  array->SetValue(18, 8);
  array->SetValue(19, 7);
  array->SetValue(20, 4);
  array->SetValue(21, 7);
  array->SetValue(22, 8);
  array->SetValue(23, 9);
  array->SetHeavyDataSetName("Test.h5:/Conns");
  // Geometry
  geometry = grid.GetGeometry();
  geometry->SetGeometryType(XDMF_GEOMETRY_XY);
  geometry->SetNumberOfPoints(npts);
  array = geometry->GetPoints();
  array->SetNumberType(XDMF_FLOAT64_TYPE);
  array->SetNumberOfElements(npts * 2 + 2);
  array->SetValue(0, 0);
  array->SetValue(1, 0);
  array->SetValue(2, 1);
  array->SetValue(3, 0);
  array->SetValue(4, 2);
  array->SetValue(5, 0);
  array->SetValue(6, 0);
  array->SetValue(7, 1);
  array->SetValue(8, 1);
  array->SetValue(9, 1);
  array->SetValue(10, 2);
  array->SetValue(11, 1);
  array->SetValue(12, 0);
  array->SetValue(13, 2);
  array->SetValue(14, 1);
  array->SetValue(15, 2);
  array->SetValue(16, 2);
  array->SetValue(17, 2);
  array->SetValue(18, 2);
  array->SetValue(19, 3);
  array->SetHeavyDataSetName("Test.h5:/Points");
  // Node Data
  nodedata.SetName("Node Scalar");
  nodedata.SetAttributeCenter(XDMF_ATTRIBUTE_CENTER_NODE);
  nodedata.SetAttributeType(XDMF_ATTRIBUTE_TYPE_SCALAR);
  array = nodedata.GetValues();
  array->SetNumberType(XDMF_FLOAT64_TYPE);
  array->SetNumberOfElements(npts+1);
  array->SetValue(0, 0);
  array->SetValue(1, 1);
  array->SetValue(2, 2);
  array->SetValue(3, 3);
  array->SetValue(4, 4);
  array->SetValue(5, 5);
  array->SetValue(6, 6);
  array->SetValue(7, 7);
  array->SetValue(8, 8);
  array->SetValue(9, 9);
  array->SetHeavyDataSetName("Test.h5:/NodeData");
  // Cell Data
  celldata.SetName("Cell Scalar");
  celldata.SetAttributeCenter(XDMF_ATTRIBUTE_CENTER_CELL);
  celldata.SetAttributeType(XDMF_ATTRIBUTE_TYPE_SCALAR);
  array = celldata.GetValues();
  array->SetNumberType(XDMF_FLOAT64_TYPE);
  array->SetNumberOfElements(nel+1);
  array->SetValue(0, 0);
  array->SetValue(1, 1);
  array->SetValue(2, 2);
  array->SetValue(3, 3);
  array->SetValue(4, 4);
  array->SetHeavyDataSetName("Test.h5:/CellData");
  // Attach and Write
  grid.Insert(&nodedata);
  grid.Insert(&celldata);
  // Build is recursive ... it will be called on all of the child nodes.
  // This updates the DOM and writes the HDF5
  root.Build();
  // Write the XML
  dom.Write("Test.xmf");
}