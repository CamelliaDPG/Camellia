#ifndef SOLUTIONEXPORTER_H
#define SOLUTIONEXPORTER_H

// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of
// conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

/*
 *  SolutionExporter.h
 *
 *  Created by Truman Ellis on 12/11/2012.
 *
 */

#include "Solution.h"
#include "Mesh.h"
#include "MeshTopology.h"
#include "VarFactory.h"

class SolutionExporter {
protected:
  SolutionPtr _solution;
  MeshPtr _mesh;
  VarFactory& _varFactory;

public:
  SolutionExporter(SolutionPtr solution, MeshPtr mesh, VarFactory& varFactory) :
    _solution(solution), _mesh(mesh), _varFactory(varFactory) {}
  virtual void exportSolution(const string& filePath, unsigned int num1DPts) = 0;
};

class VTKExporter : public SolutionExporter {
private:
public:
  VTKExporter(SolutionPtr solution, MeshPtr mesh, VarFactory& varFactory) :
    SolutionExporter(solution, mesh, varFactory) {};
  void exportSolution(const string& filePath, unsigned int num1DPts=0);
  void exportFields(const string& filePath, unsigned int num1DPts=0);
  void exportTraces(const string& filePath, unsigned int num1DPts=0);
  void exportFunction(FunctionPtr function, const string& functionName="function", unsigned int num1DPts=0);
  void exportBoundaryValuedFunctions(vector< FunctionPtr > &functions, const string& filePath, unsigned int num1DPts=0);
  void setRefPoints(FieldContainer<double> refPoints, int num1DPts, int spaceDim);
};

class NewExporter {
protected:
  // SolutionPtr _solution;
  MeshTopologyPtr _mesh;
  // VarFactory& _varFactory;

public:
  NewExporter(MeshTopologyPtr mesh) :
    _mesh(mesh) {}
  virtual void exportFunction(FunctionPtr function, const string& functionName="function", set<GlobalIndexType> cellIndices=set<GlobalIndexType>(), unsigned int num1DPts=0) = 0;
};

class NewVTKExporter : public NewExporter {
public:
  NewVTKExporter(MeshTopologyPtr mesh) :
    NewExporter(mesh) {}
  void exportFunction(FunctionPtr function, const string& functionName="function", set<GlobalIndexType> cellIndices=set<GlobalIndexType>(), unsigned int num1DPts=0);
};

class XDMFExporter : public NewExporter {
public:
  XDMFExporter(MeshTopologyPtr mesh) :
    NewExporter(mesh) {}
  void exportFunction(FunctionPtr function, const string& functionName="function", set<GlobalIndexType> cellIndices=set<GlobalIndexType>(), unsigned int num1DPts=0);
};

#endif /* end of include guard: SOLUTIONEXPORTER_H */
