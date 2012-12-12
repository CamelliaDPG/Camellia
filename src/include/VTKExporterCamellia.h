#ifndef VTKEXPORTER_H
#define VTKEXPORTER_H

#warning "processing VTKEXPORTER_H"
/*
 *  VTKExporter.h
 *
 *  Created by Truman Ellis on 12/12/2012.
 *
 */

#include "SolutionExporter.h"
 
class VTKExporter : public SolutionExporter {
private:
public:
  VTKExporter(SolutionPtr solution, MeshPtr mesh, VarFactory& varFactory) :
    SolutionExporter(solution, mesh, varFactory) {};
  void exportSolution(const string& filePath, unsigned int num1DPts=0);
  void exportFields(const string& filePath, unsigned int num1DPts=0);
  void exportTraces(const string& filePath, unsigned int num1DPts=0);
  void setRefPoints(FieldContainer<double> refPoints, int num1DPts, int spaceDim);
};

#endif /* end of include guard: VTKEXPORTER_H */
