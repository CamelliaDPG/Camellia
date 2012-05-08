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
#include "Function.h"

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
  _randomRefinements = randomRefinements;
  _useTriangles = useTriangles;
  _useHybrid = useHybrid;
  _reportRelativeErrors = true;
  vector<int> trialIDs = bilinearForm->trialIDs();
  for (vector<int>::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    FunctionPtr exactSoln = Teuchos::rcp( new ExactSolutionFunction(_exactSolution,trialID) );
    _exactSolutionFunctions[trialID] = exactSoln;
  }
  
  if (_useTriangles)
    cout << "HConvergenceStudy: Using triangles\n" << endl;
}

Teuchos::RCP<Solution> HConvergenceStudy::getSolution(int logElements) {
  int index = logElements - _minLogElements;
  return _solutions[index];
}

void HConvergenceStudy::randomlyRefine(Teuchos::RCP<Mesh> mesh) {
  int numElements = mesh->activeElements().size();
  vector<int> elementsToRefineP;
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
    int numVertices = mesh->activeElements()[i]->numSides();
    int spaceDim = 2;
    FieldContainer<double> vertices(numVertices,spaceDim);
    mesh->verticesForCell(vertices,cellID);
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
  mesh->pRefine(elementsToRefineP);
}

int HConvergenceStudy::minNumElements() {
  int minNumElements = 1;
  for (int i=0; i<_minLogElements; i++) {
    minNumElements *= 2;
  }
  return minNumElements;
}

Teuchos::RCP<BilinearForm> HConvergenceStudy::bilinearForm() {
  return _bilinearForm;
}

vector<int> HConvergenceStudy::meshSizes() {
  vector<int> sizes;
  int size = minNumElements();
  for (int logSize = _minLogElements; logSize <= _maxLogElements; logSize++) {
    sizes.push_back(size);
    size *= 2;
  }
  return sizes;
}

map< int, vector<double> > HConvergenceStudy::bestApproximationErrors() {
  return _bestApproximationErrors;
}

map< int, vector<double> > HConvergenceStudy::solutionErrors() {
  return _solutionErrors;
}

map< int, vector<double> > HConvergenceStudy::bestApproximationRates() {
  return _bestApproximationRates;
}

map< int, vector<double> > HConvergenceStudy::solutionRates() {
  return _solutionRates;
}

map< int, double > HConvergenceStudy::exactSolutionNorm() {
  return _exactSolutionNorm;
}

void HConvergenceStudy::computeErrors() {
  SolutionPtr solution = _solutions[0];
  vector<int> trialIDs = _bilinearForm->trialIDs();
  for (vector<int>::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    vector< Teuchos::RCP<Solution> >::iterator solutionIt;
    
    int numElements = minNumElements();
    
    double l2norm = _exactSolution->L2NormOfError(*_fineZeroSolution, trialID, 15); // 15 == cubature degree to use...
    _exactSolutionNorm[trialID] = l2norm;
    
    double previousL2Error = -1.0;
    for (solutionIt = _solutions.begin(); solutionIt != _solutions.end(); solutionIt++) {
      SolutionPtr solution = *solutionIt;
      double l2error = _exactSolution->L2NormOfError(*solution, trialID, 15); // 15 == cubature degree to use...
      _solutionErrors[trialID].push_back(l2error);
      
      if (previousL2Error >= 0.0) {
        double rate = - log(l2error/previousL2Error) / log(2.0);
        _solutionRates[trialID].push_back(rate);
      }
      numElements *= 2;
      previousL2Error = l2error;
    }
    
    numElements = minNumElements();
    previousL2Error = -1;
    for (solutionIt = _bestApproximations.begin(); solutionIt != _bestApproximations.end(); solutionIt++) {
      SolutionPtr bestApproximation = *solutionIt;
      double l2error = _exactSolution->L2NormOfError(*bestApproximation, trialID, 15); // 15 == cubature degree to use...
      _bestApproximationErrors[trialID].push_back(l2error);
      
      if (previousL2Error >= 0.0) {
        double rate = - log(l2error/previousL2Error) / log(2.0);
        _bestApproximationRates[trialID].push_back(rate);
      }
      
      numElements *= 2;
      previousL2Error = l2error;
    }
  }
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
    
    Teuchos::RCP<Solution> bestApproximation = Teuchos::rcp( new Solution(mesh, _bc, _rhs, _ip) );
    _bestApproximations.push_back(bestApproximation);
    bestApproximation->projectOntoMesh(_exactSolutionFunctions);

    numElements *= 2;
    if (i == _maxLogElements) {
      // We'll use solution to compute L2 norm of true Solution
      _fineZeroSolution = Teuchos::rcp( new Solution(mesh, _bc, _rhs, _ip) );
    }
  }
  vector< Teuchos::RCP<Solution> >::iterator solutionIt;
  // now actually compute all the solutions:
  for (solutionIt = _solutions.begin(); solutionIt != _solutions.end(); solutionIt++) {
    (*solutionIt)->solve(false); // False: don't use mumps (use KLU)
  }
  computeErrors();
}

void HConvergenceStudy::setReportRelativeErrors(bool reportRelativeErrors) {
  _reportRelativeErrors = reportRelativeErrors;
}

string HConvergenceStudy::TeXErrorRateTable() {
  vector<int> trialIDs = _bilinearForm->trialVolumeIDs();
  return TeXErrorRateTable(trialIDs);
}

string HConvergenceStudy::TeXErrorRateTable( const vector<int> &trialIDs ) {
  ostringstream texString;
  texString << scientific;
  
  int numColumns = trialIDs.size() * 2 + 1;
  string newLine = "\\\\\n";
  texString << "\\multicolumn{" << numColumns << "}{| c |}{k=" << _H1Order-1 << "} " << newLine;

  texString << "\\hline \n";
  texString << "Mesh Size ";
  for (vector<int>::const_iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    texString << "& $" << _bilinearForm->trialName(trialID) << "$ & rate ";
  }
  texString << newLine;
  texString << "\\hline \n";
  vector<int> meshSizes = this->meshSizes();
  for (int i=0; i<meshSizes.size(); i++) {
    int size = meshSizes[i];
    texString << size << " $\\times$ " << size;
    for (vector<int>::const_iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *trialIt;
      double l2error = _solutionErrors[trialID][i];
      if (_reportRelativeErrors) {
        l2error /= _exactSolutionNorm[trialID];
      }
      texString << setprecision(1) << scientific << "\t&" << l2error;
      if (i > 0) {
        double rate = _solutionRates[trialID][i-1];
        texString << "\t&" << setprecision(2) << fixed << rate;
      } else {
        texString << "\t&-";
      }
    }
    texString << newLine;
  }
  texString << "\\hline \n";
  return texString.str();
}

string HConvergenceStudy::convergenceDataMATLAB(int trialID) {
  ostringstream ss;
  int k = _H1Order - 1; // order of the solution
  ss << scientific;
  ss << _bilinearForm->trialName(trialID) << "Error{" << k << "} = [";

  vector<int> meshSizes = this->meshSizes();
  for (int i=0; i<meshSizes.size(); i++) {
    int size = meshSizes[i];
    double h = 1.0 / size;
    ss << k << " " << h << " ";
    double l2error = _solutionErrors[trialID][i];
    if (_reportRelativeErrors) {
      l2error /= _exactSolutionNorm[trialID];
    }
    ss << l2error << "\n";
  }
  
  ss << "];\n";
  return ss.str();
}

string HConvergenceStudy::TeXBestApproximationComparisonTable() {
  vector<int> trialIDs = _bilinearForm->trialVolumeIDs();
  return this->TeXBestApproximationComparisonTable(trialIDs);
}

string HConvergenceStudy::TeXBestApproximationComparisonTable( const vector<int> &trialIDs ) {
  /*
   \multicolumn{3}{| c |}{k=3} \\
   \hline
   \multirow{2}{*}{Mesh Size} & \multicolumn{2}{ c |}{$p$}  \\ \cline{2-3}
   & actual & best \\
   \hline
   1 $\times$ 1    &4.4e-02		&4.1e-03		\\
   2 $\times$ 2    &3.1e-03		&6.4e-04	     	\\
   4 $\times$ 4    &3.3e-04		&4.2e-05	     	\\
   8 $\times$ 8    &3.2e-05		&2.6e-06	     	\\
   16 $\times$ 16  &3.3e-06		&1.6e-07	     	\\
   32 $\times$ 32  &4.1e-07		&1.0e-08	         	\\
   \hline
   */
  ostringstream texString;
  texString << scientific;
    
  int numColumns = trialIDs.size() * 2 + 1;
  string newLine = "\\\\\n";
  texString << "\\multicolumn{" << numColumns << "}{| c |}{k=" << _H1Order-1 << "} " << newLine;
  
  texString << "\\hline \n";
  texString << "\\multirow{2}{*}{Mesh Size}";
  for (vector<int>::const_iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    texString << " & \\multicolumn{2}{ c |}{$" << _bilinearForm->trialName(trialID) << "$}";
  }
  texString << newLine << "\\cline{2-" << numColumns << "}" << "\n";
  for (vector<int>::const_iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    texString << " & actual & best ";
  }
  texString << newLine;
  texString << "\\hline \n";
  vector<int> meshSizes = this->meshSizes();
  for (int i=0; i<meshSizes.size(); i++) {
    int size = meshSizes[i];
    texString << size << " $\\times$ " << size;
    for (vector<int>::const_iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *trialIt;
      double l2error = _solutionErrors[trialID][i];
      if (_reportRelativeErrors) {
        l2error /= _exactSolutionNorm[trialID];
      }
      texString << setprecision(1) << scientific << "\t&" << l2error;
      double bestError = _bestApproximationErrors[trialID][i];
      if (_reportRelativeErrors) {
        bestError /= _exactSolutionNorm[trialID];
      }
      texString << "\t&" << setprecision(1) << scientific << bestError;
    }
    texString << newLine;
  }
  texString << "\\hline \n";
  return texString.str();
}

void HConvergenceStudy::writeToFiles(const string & filePathPrefix, int trialID, int traceID) {
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
  
  cout << "**************************************************************\n";
  if (_reportRelativeErrors) {
    cout << "Relative ";
  }
  cout << "L2 error for variable " << _bilinearForm->trialName(trialID) << ":" << endl;
  
  double l2norm = _exactSolution->L2NormOfError(*_fineZeroSolution, trialID, 15); // 15 == cubature degree to use...
  
  double previousL2Error = -1.0;
  for (solutionIt = _solutions.begin(); solutionIt != _solutions.end(); solutionIt++) {
    SolutionPtr solution = *solutionIt;
    double l2error = _exactSolution->L2NormOfError(*solution, trialID, 15); // 15 == cubature degree to use...
    if (_reportRelativeErrors) {
      l2error /= l2norm;
    }
    
    string spaces = (numElements > 10) ? " " : "   ";
    
    cout << scientific;
    fout << scientific;
    cout << numElements << "x" << numElements << " mesh:" << spaces << setprecision(1) <<  l2error;
    fout << numElements << "x" << numElements << "\t" << setprecision(1) << l2error;
    if (previousL2Error >= 0.0) {
      double rate = - log(l2error/previousL2Error) / log(2.0);
//      cout.unsetf(ios::floatfield); // don't show rates with exponents (they'll be 0 anyway)
//      fout.unsetf(ios::floatfield);
      cout << "\t(rate: " << setprecision(2) << fixed << rate << ")";
      fout << "\t(rate: " << setprecision(2) << fixed << rate << ")"; 
    }
    cout << endl;
    fout << endl;
    
//    for (int cellID=0; cellID<numElements; cellID++) {
//      FieldContainer<double> solnCoeffs;
//      solution.solnCoeffsForCellID(solnCoeffs, cellID, trialID);
//      cout << "solnCoeffs for cell " << cellID << ":" << endl << solnCoeffs;
//    }
    
    // now write out the solution for MATLAB plotting...
    fileName.str(""); // clear out the filename
    fileName << filePathPrefix << "_solution_" << numElements << "x" << numElements << ".m";
//    solution.writeToFile(trialID, fileName.str());
    solution->writeFieldsToFile(trialID, fileName.str());
    fileName.str(""); // clear out the filename
    if (traceID != -1) {
      fileName << filePathPrefix << "_trace_solution_" << numElements << "x" << numElements << ".dat";
      solution->writeFluxesToFile(traceID, fileName.str());
    }
    numElements *= 2;
    previousL2Error = l2error;
  }
  
  cout << "***********  BEST APPROXIMATION ERROR for : " << _bilinearForm->trialName(trialID) << " **************\n";
  numElements = minNumElements;
  previousL2Error = -1;
  for (solutionIt = _bestApproximations.begin(); solutionIt != _bestApproximations.end(); solutionIt++) {
    SolutionPtr bestApproximation = *solutionIt;
    double l2error = _exactSolution->L2NormOfError(*bestApproximation, trialID, 15); // 15 == cubature degree to use...
    if (_reportRelativeErrors) {
      l2error /= l2norm;
    }
    
    string spaces = (numElements > 10) ? " " : "   ";
    cout << scientific;
    cout << numElements << "x" << numElements << " mesh:" << spaces << setprecision(1) <<  l2error;
    if (previousL2Error >= 0.0) {
      double rate = - log(l2error/previousL2Error) / log(2.0);
      cout << "\t(rate: " << setprecision(2) << fixed << rate << ")";
    }
    cout << endl;

    numElements *= 2;
    previousL2Error = l2error;
  }
  cout << "**************************************************************\n";

  
  cout << "L2 norm of solution: " << l2norm  << endl;
  
  fout.close();
}
