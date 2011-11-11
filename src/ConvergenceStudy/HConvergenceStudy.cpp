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
 *  HConvergenceStudy.cpp
 *
 *  Created by Nathan Roberts on 7/21/11.
 *
 */

#include "HConvergenceStudy.h"

HConvergenceStudy::HConvergenceStudy(Teuchos::RCP<ExactSolution> exactSolution,
                                     Teuchos::RCP<BilinearForm> bilinearForm,
                                     Teuchos::RCP<RHS> rhs,
                                     Teuchos::RCP<BC> bc,
                                     Teuchos::RCP<DPGInnerProduct> ip, 
                                     int minLogElements, int maxLogElements, 
                                     int H1Order, int pToAdd, bool randomRefinements,
                                     bool useTriangles, bool useHybrid) {
  _exactSolution = exactSolution;
  _bilinearForm = bilinearForm;
  _rhs = rhs;  
  _bc = bc;
  _ip = ip;
  _minLogElements = minLogElements;
  _maxLogElements = maxLogElements;
  _H1Order = H1Order;
  _pToAdd = pToAdd;
  _bilinearForm->printTrialTestInteractions();
  _randomRefinements = randomRefinements;
  _useTriangles = useTriangles;
  _useHybrid = useHybrid;
  _reportRelativeErrors = true;
  if (_useTriangles)
    cout << "HConvergenceStudy: Using triangles\n" << endl;
}

void HConvergenceStudy::randomlyRefine(Teuchos::RCP<Mesh> mesh) {
  int numElements = mesh->activeElements().size();
  vector<int> elementsToRefineP,elementsToRefineH;
/*  // every third element is: untouched, refined once, refined twice
  for (int i=0; i<numElements; i++) {
    if ((i%3)==1) {
      elementsToRefineP.push_back(i);
    } else if ((i%3)==2) {
      elementsToRefineP.push_back(i);
      elementsToRefineP.push_back(i);
    }
  }*/
  // new rule: if all vertices' x >= 0, refine once.
  //           if all vertices' y >= 0, refine once (more).
  // for up to two refinements.
  for (int i=0; i<numElements; i++) {
    int cellID = mesh->activeElements()[i]->cellID();
    FieldContainer<double> vertices;
    mesh->verticesForCell(vertices,cellID);
    int numVertices = vertices.dimension(0);
    bool positiveX = true, positiveY = true;
    for (int vertexIndex=0; vertexIndex<numVertices; vertexIndex++) {
      double x = vertices(vertexIndex,0);
      double y = vertices(vertexIndex,1);
      if ( x < 0.0 ) {
        positiveX = false;
      }
      if (y < 0.0 ) {
        positiveY = false;
      }
    }
    if ( positiveX ) {
      elementsToRefineP.push_back(cellID);
    } 
    if ( positiveY ) {
      elementsToRefineP.push_back(cellID);
    }
  }
  mesh->refine(elementsToRefineP,elementsToRefineH);
}

void HConvergenceStudy::solve(const FieldContainer<double> &quadPoints) {
  _solutions.clear();
  int minNumElements = 1;
  for (int i=0; i<_minLogElements; i++) {
    minNumElements *= 2;
  }
  
  int numElements = minNumElements;
  for (int i=_minLogElements; i<=_maxLogElements; i++) {
    Teuchos::RCP<Mesh> mesh;
    if (! _useHybrid ) {
      mesh = Mesh::buildQuadMesh(quadPoints, numElements, numElements, 
                                 _bilinearForm, _H1Order, _H1Order + _pToAdd, _useTriangles);
    } else {
      mesh = Mesh::buildQuadMeshHybrid(quadPoints, numElements, numElements, 
                                       _bilinearForm, _H1Order, _H1Order + _pToAdd);
    }
    if (_randomRefinements) {
      randomlyRefine(mesh);
    }
    Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, _bc, _rhs, _ip) );
    _solutions.push_back(solution);
    numElements *= 2;
    if (i == _maxLogElements) {
      // We'll use solution to compute L2 norm of true Solution
      _fineZeroSolution = Teuchos::rcp( new Solution(mesh, _bc, _rhs, _ip) );
    }
  }
  vector< Teuchos::RCP<Solution> >::iterator solutionIt;
  // now actually compute all the solutions:
  for (solutionIt = _solutions.begin(); solutionIt != _solutions.end(); solutionIt++) {
    (*solutionIt)->solve();
  }
}

void HConvergenceStudy::setReportRelativeErrors(bool reportRelativeErrors) {
  _reportRelativeErrors = reportRelativeErrors;
}

void HConvergenceStudy::writeToFiles(const string & filePathPrefix, int trialID) {
  vector< Teuchos::RCP<Solution> >::iterator solutionIt;
  int minNumElements = 1;
  for (int i=0; i<_minLogElements; i++) {
    minNumElements *= 2;
  }
  
  int numElements = minNumElements;
  
  ostringstream fileName;
  fileName << filePathPrefix << "_h1_convergence.txt";
  ofstream fout(fileName.str().c_str());
  fout << setprecision(15);
  
  cout << "********************************************" << endl;
  if (_reportRelativeErrors) {
    cout << "Relative ";
  }
  cout << "L2 error for variable " << _bilinearForm->trialName(trialID) << ":" << endl;
  
  double l2norm = _exactSolution->L2NormOfError(*_fineZeroSolution, trialID, 15); // 15 == cubature degree to use...
  
  double previousL2Error = -1.0;
  for (solutionIt = _solutions.begin(); solutionIt != _solutions.end(); solutionIt++) {
    Solution solution = *(solutionIt->get());
    double l2error = _exactSolution->L2NormOfError(solution, trialID, 15); // 15 == cubature degree to use...
    if (_reportRelativeErrors) {
      l2error /= l2norm;
    }
    
    string spaces = (numElements > 10) ? " " : "   ";
    
    cout << numElements << "x" << numElements << " mesh:" << spaces << l2error;
    
    fout << numElements << "x" << numElements << "\t" << l2error;
    if (previousL2Error >= 0.0) {
      double rate = - log(l2error/previousL2Error) / log(2.0);
      cout << "\t(rate: " << rate << ")";
      fout << "\t(rate: " << rate << ")"; 
    }
    cout << endl;
    fout << endl;
    
    // now write out the solution for MATLAB plotting...
    fileName.str(""); // clear out the filename
    fileName << filePathPrefix << "_solution_" << numElements << "x" << numElements << ".dat";
    solution.writeToFile(trialID, fileName.str());
    numElements *= 2;
    previousL2Error = l2error;
  }
  
  cout << "L2 norm of solution: " << l2norm  << endl;
  
  fout.close();
}
