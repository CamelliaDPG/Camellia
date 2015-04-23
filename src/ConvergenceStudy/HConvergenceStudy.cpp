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

#include "ExactSolutionFunction.h"
#include "Function.h"
#include "RefinementStrategy.h"

#include "DataIO.h"
#include "SerialDenseMatrixUtility.h"
#include "MeshFactory.h"

using namespace Intrepid;
using namespace Camellia;

HConvergenceStudy::HConvergenceStudy(Teuchos::RCP<ExactSolution<double>> exactSolution,
                                     BFPtr bilinearForm,
                                     Teuchos::RCP<RHS> rhs,
                                     Teuchos::RCP<BC> bc,
                                     IPPtr ip,
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
  _reportConditionNumber = false;
  _useTriangles = useTriangles;
  _useHybrid = useHybrid;
  _reportRelativeErrors = true;
  _cubatureDegreeForExact = 10; // an enrichment degree
  _cubatureEnrichmentForSolutions = 0;
  _solver = Teuchos::rcp( (Solver*) NULL ); // redundant code, but I like to be explicit
  _useCondensedSolve = false;
//  vector<int> trialIDs = bilinearForm->trialIDs();
  vector<int> trialIDs = bilinearForm->trialVolumeIDs(); // so far, we don't have a good analytic way to measure flux and trace errors.
  for (vector<int>::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    TFunctionPtr<double> exactSoln = Teuchos::rcp( new ExactSolutionFunction<double>(_exactSolution,trialID) );
    _exactSolutionFunctions[trialID] = exactSoln;
  }
  // in the past, we haven't done any projection of boundary Functions.
  // here, we do so if it's a "modern" ExactSolution (defined in terms of TFunctionPtr<double>s, etc.)
  // and that ExactSolution defines a given boundary variable's Function.
  trialIDs = bilinearForm->trialBoundaryIDs();
  for (vector<int>::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    if (_exactSolution->functionDefined(trialID)) {
      TFunctionPtr<double> exactSoln = Teuchos::rcp( new ExactSolutionFunction<double>(_exactSolution,trialID) );
      _exactSolutionFunctions[trialID] = exactSoln;
    }
  }
  if (_useTriangles)
    cout << "HConvergenceStudy: Using triangles\n" << endl;
}

void HConvergenceStudy::addDerivedVariable(LinearTermPtr derivedVar, const string &name) {
  DerivedVariable dv;
  dv.term = derivedVar;
  dv.name = name;
  _derivedVariables.push_back(dv);
}

TSolutionPtr<double> HConvergenceStudy::getSolution(int logElements) {
  int index = logElements - _minLogElements;
  return _solutions[index];
}

void HConvergenceStudy::randomlyRefine(Teuchos::RCP<Mesh> mesh) {
  int numElements = mesh->activeElements().size();
  vector<GlobalIndexType> elementsToRefineP;
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
    GlobalIndexType cellID = mesh->activeElements()[i]->cellID();
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

BFPtr HConvergenceStudy::bilinearForm() {
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

vector< TSolutionPtr<double> > & HConvergenceStudy::bestApproximations() {
  return _bestApproximations;
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

double HConvergenceStudy::computeJacobiPreconditionedConditionNumber(int logElements) {
  // in so many ways, this is not the best way to do this: it's slow both because
  // we do the disk I/O and because we form a sparse matrix using a dense construct
  // (the FieldContainer), but this allows us to use a condition number computation that
  // I actually trust...
  if ((logElements < _minLogElements) || (logElements > _maxLogElements)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "logElements argument out of range");
  }
  if (!_writeGlobalStiffnessToDisk) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "computeConditionNumber only supported when writeGlobalStiffnessToDisk == true");
  }
  if (_solutions.size()==0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "computeConditionNumber only supported after the solve is complete");
  }
  ostringstream fileName;
  fileName << _globalStiffnessFilePrefix << "_" << logElements << ".dat";
  FieldContainer<double> globalStiffnessMatrix;
  DataIO::readMatrixFromSparseDataFile(globalStiffnessMatrix, fileName.str());
  SerialDenseMatrixUtility::jacobiScaleMatrix(globalStiffnessMatrix);
  return SerialDenseMatrixUtility::estimate2NormConditionNumber(globalStiffnessMatrix);
}

void HConvergenceStudy::computeErrors() {
  TSolutionPtr<double> solution = _solutions[0];
  vector<int> trialIDs = _bilinearForm->trialVolumeIDs();

  // clear all the data structures:
  _solutionErrors.clear();
  _solutionRates.clear();
  _bestApproximationErrors.clear();
  _bestApproximationRates.clear();


  for (vector<int>::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    vector< TSolutionPtr<double> >::iterator solutionIt;

    int numElements = minNumElements();

    double l2norm = _exactSolution->L2NormOfError(_fineZeroSolution, trialID, _cubatureDegreeForExact);
    _exactSolutionNorm[trialID] = l2norm;

    double previousL2Error = -1.0;
    for (solutionIt = _solutions.begin(); solutionIt != _solutions.end(); solutionIt++) {
      TSolutionPtr<double> solution = *solutionIt;
      double l2error = _exactSolution->L2NormOfError(solution, trialID, _cubatureDegreeForExact);
      _solutionErrors[trialID].push_back(l2error);

      if (previousL2Error != -1.0) {
        double rate = - log(l2error/previousL2Error) / log(2.0);
        _solutionRates[trialID].push_back(rate);
      }
      numElements *= 2;
      previousL2Error = l2error;
    }

    numElements = minNumElements();
    previousL2Error = -1;
    for (solutionIt = _bestApproximations.begin(); solutionIt != _bestApproximations.end(); solutionIt++) {
      TSolutionPtr<double> bestApproximation = *solutionIt;

      // OLD CODE: testing this by reimplementing in terms of Functions
      double l2error = _exactSolution->L2NormOfError(bestApproximation, trialID, _cubatureDegreeForExact);

      // test code below--doesn't seem to be much difference
//      {
//        VarPtr trialVar = Teuchos::rcp( new Var(trialID, 0, "dummyVar"));
//
//        TFunctionPtr<double> bestFxnError = Function::solution(trialVar, bestApproximation) - _exactSolutionFunctions[trialID];
//        double l2error = bestFxnError->l2norm(bestApproximation->mesh(),_cubatureDegreeForExact); // here the cubature is actually an enrichment....
//
//        cout << "HConvergenceStudy: best l2Error for trial ID " << trialID << ": " << l2error << endl;
//      }

      _bestApproximationErrors[trialID].push_back(l2error);

      if (previousL2Error != -1.0) {
        double rate = - log(l2error/previousL2Error) / log(2.0);
//        cout << "HConvergenceStudy: best rate for trial ID " << trialID << ": " << rate << endl;
        _bestApproximationRates[trialID].push_back(rate);
      }

      numElements *= 2;
      previousL2Error = l2error;
    }
  }
}

void HConvergenceStudy::setCubatureDegreeForExact(int value) {
  _cubatureDegreeForExact = value;
}

void HConvergenceStudy::setCubatureEnrichmentForSolutions(int value) {
  _cubatureEnrichmentForSolutions = value;
}

void HConvergenceStudy::setLagrangeConstraints(Teuchos::RCP<LagrangeConstraints> lagrangeConstraints) {
  _lagrangeConstraints = lagrangeConstraints;
}

Teuchos::RCP<Mesh> HConvergenceStudy::buildMesh( Teuchos::RCP<MeshGeometry> geometry, int numRefinements, bool useConformingTraces ) {
  Teuchos::RCP<Mesh> mesh;
  mesh = Teuchos::rcp( new Mesh(geometry->vertices(), geometry->elementVertices(), _bilinearForm,
                                _H1Order, _pToAdd, useConformingTraces) );

  map< pair<IndexType,IndexType>, ParametricCurvePtr > localMap = geometry->edgeToCurveMap();

  map< pair<GlobalIndexType,GlobalIndexType>, ParametricCurvePtr > globalMap(localMap.begin(),localMap.end());
  mesh->setEdgeToCurveMap(globalMap);

  for (int i=0; i<numRefinements; i++) {
    RefinementStrategy::hRefineUniformly(mesh);
  }
  return mesh;
}

TSolutionPtr<double> HConvergenceStudy::bestApproximation(Teuchos::RCP<Mesh> mesh) {
  // this probably should be a feature of ExactSolution
  TSolutionPtr<double> bestApproximation = Teuchos::rcp( new TSolution<double>(mesh, _bc, _rhs, _ip) );
  bestApproximation->setCubatureEnrichmentDegree(_cubatureDegreeForExact);
  bestApproximation->projectOntoMesh(_exactSolutionFunctions);
  return bestApproximation;
}

void HConvergenceStudy::setSolutions( vector< TSolutionPtr<double> > &solutions) {
  // alternative to solve() if you want to do your own solving (e.g. for nonlinear problems)
  //
  // must be in the right order, from minLogElements to maxLogElements
  // check the count is right:
  if (solutions.size() != _maxLogElements - _minLogElements + 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "solutions doesn't match the expected length");
  }
  _solutions = solutions;
  for (int i=_minLogElements; i<=_maxLogElements; i++) {
    TSolutionPtr<double> soln = solutions[i-_minLogElements];
    Teuchos::RCP<Mesh> mesh = soln->mesh();
    _bestApproximations.push_back(bestApproximation(mesh));

    if (i == _maxLogElements) {
      // We'll use solution to compute L2 norm of true Solution
      _fineZeroSolution = Teuchos::rcp( new TSolution<double>(mesh, _bc, _rhs, _ip) );
    }
  }
  computeErrors();
}

void HConvergenceStudy::solve(Teuchos::RCP<MeshGeometry> geometry, bool useConformingTraces) {
  // TODO: refactor to make this and the straight quad mesh version share code...
  _solutions.clear();
  int minNumElements = 1;
  for (int i=0; i<_minLogElements; i++) {
    minNumElements *= 2;
  }

  int numElements = minNumElements;
  for (int i=_minLogElements; i<=_maxLogElements; i++) {
    Teuchos::RCP<Mesh> mesh = buildMesh(geometry, i, useConformingTraces);
    if (_randomRefinements) {
      randomlyRefine(mesh);
    }
    TSolutionPtr<double> solution = Teuchos::rcp( new TSolution<double>(mesh, _bc, _rhs, _ip) );
    if (_writeGlobalStiffnessToDisk) {
      ostringstream fileName;
      fileName << _globalStiffnessFilePrefix << "_" << i << ".dat";
      solution->setWriteMatrixToMatrixMarketFile(true, fileName.str());
    }
    if (_lagrangeConstraints.get())
      solution->setLagrangeConstraints(_lagrangeConstraints);
    solution->setReportConditionNumber(_reportConditionNumber);
    solution->setCubatureEnrichmentDegree(_cubatureEnrichmentForSolutions);
    _solutions.push_back(solution);

    _bestApproximations.push_back(bestApproximation(mesh));

    numElements *= 2;
    if (i == _maxLogElements) {
      // We'll use solution to compute L2 norm of true Solution
      _fineZeroSolution = Teuchos::rcp( new TSolution<double>(mesh, _bc, _rhs, _ip) );
    }
  }
  vector< TSolutionPtr<double> >::iterator solutionIt;
  // now actually compute all the solutions:
  for (solutionIt = _solutions.begin(); solutionIt != _solutions.end(); solutionIt++) {
    if ( _solver.get() == NULL )
      (*solutionIt)->solve(false);   // False: don't use mumps (use KLU)
    else
      (*solutionIt)->solve(_solver); // Use whatever Solver the user specified
  }
  computeErrors();
}

void HConvergenceStudy::solve(const FieldContainer<double> &quadPoints, bool useConformingTraces) {
  // empty enhancements by default
  map<int,int> trialOrderEnhancements;
  map<int,int> testOrderEnhancements;
  this->solve(quadPoints, useConformingTraces, trialOrderEnhancements, testOrderEnhancements);
}

void HConvergenceStudy::solve(const FieldContainer<double> &quadPoints, bool useConformingTraces,
                              map<int,int> trialOrderEnhancements,
                              map<int,int> testOrderEnhancements) {
  _solutions.clear();
  int minNumElements = 1;
  for (int i=0; i<_minLogElements; i++) {
    minNumElements *= 2;
  }

  int numElements = minNumElements;
  for (int i=_minLogElements; i<=_maxLogElements; i++) {
    Teuchos::RCP<Mesh> mesh;
    if (! _useHybrid ) {
      mesh = MeshFactory::buildQuadMesh(quadPoints, numElements, numElements,
                                 _bilinearForm, _H1Order, _H1Order + _pToAdd, _useTriangles, useConformingTraces,
                                 trialOrderEnhancements,testOrderEnhancements);
    } else {
      mesh = MeshFactory::buildQuadMeshHybrid(quadPoints, numElements, numElements,
                                       _bilinearForm, _H1Order, _H1Order + _pToAdd, useConformingTraces);
    }
    if (_randomRefinements) {
      randomlyRefine(mesh);
    }
    TSolutionPtr<double> solution = Teuchos::rcp( new TSolution<double>(mesh, _bc, _rhs, _ip) );
    if (_writeGlobalStiffnessToDisk) {
      ostringstream fileName;
      fileName << _globalStiffnessFilePrefix << "_" << i << ".dat";
      solution->setWriteMatrixToMatrixMarketFile(true, fileName.str());
    }
    if (_lagrangeConstraints.get())
      solution->setLagrangeConstraints(_lagrangeConstraints);
    solution->setCubatureEnrichmentDegree(_cubatureEnrichmentForSolutions);
    solution->setReportConditionNumber(_reportConditionNumber);
    _solutions.push_back(solution);

    _bestApproximations.push_back(bestApproximation(mesh));


    numElements *= 2;
    if (i == _maxLogElements) {
      // We'll use solution to compute L2 norm of true Solution
      _fineZeroSolution = Teuchos::rcp( new TSolution<double>(mesh, _bc, _rhs, _ip) );
    }
  }
  vector< TSolutionPtr<double> >::iterator solutionIt;
  // now actually compute all the solutions:
  for (solutionIt = _solutions.begin(); solutionIt != _solutions.end(); solutionIt++) {
    if ( _solver.get() == NULL )
      if (_useCondensedSolve) {
        (*solutionIt)->condensedSolve(); // defaults to KLU
      } else {
        (*solutionIt)->solve(false);   // False: don't use mumps (use KLU)
      }
      else {
        if (_useCondensedSolve) {
          (*solutionIt)->condensedSolve(_solver); // Use whatever Solver the user specified
        } else {
          (*solutionIt)->solve(_solver); // Use whatever Solver the user specified
        }
      }
  }
  computeErrors();
}

void HConvergenceStudy::setReportConditionNumber(bool value) {
  _reportConditionNumber = value;
}

void HConvergenceStudy::setReportRelativeErrors(bool reportRelativeErrors) {
  _reportRelativeErrors = reportRelativeErrors;
}

string HConvergenceStudy::TeXErrorRateTable( const string & filePathPrefix ) {
  vector<int> trialIDs = _bilinearForm->trialVolumeIDs();
  return TeXErrorRateTable(trialIDs, filePathPrefix);
}

string HConvergenceStudy::TeXErrorRateTable( const vector<int> &trialIDs, const string &filePathPrefix ) {
  ostringstream texString;
  texString << scientific;

  bool reportCombinedRelativeError = true;

  int numColumns = reportCombinedRelativeError ?  trialIDs.size() * 2 + 3 : trialIDs.size() * 2 + 1;
  string newLine = "\\\\\n";
  texString << "\\multicolumn{" << numColumns << "}{| c |}{k=" << _H1Order-1 << "} " << newLine;

  texString << "\\hline \n";
  texString << "Mesh Size ";
  for (vector<int>::const_iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    texString << "& $" << _bilinearForm->trialName(trialID) << "$ & rate ";
  }
  if (reportCombinedRelativeError) {
    texString << "& Combined relative error & rate ";
  }
  texString << newLine;
  texString << "\\hline \n";
  vector<int> meshSizes = this->meshSizes();
  double previousRelativeError = -1;
  for (int i=0; i<meshSizes.size(); i++) {
    int size = meshSizes[i];
    texString << size << " $\\times$ " << size;
    double totalErrorSquared = 0;
    double totalSolutionNormSquared = 0;
    for (vector<int>::const_iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      int trialID = *trialIt;
      double l2error = _solutionErrors[trialID][i];
      totalErrorSquared += l2error * l2error;
      totalSolutionNormSquared += _exactSolutionNorm[trialID] * _exactSolutionNorm[trialID];
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
    if (reportCombinedRelativeError) {
      double combinedRelativeError = sqrt( totalErrorSquared / totalSolutionNormSquared );
      texString << setprecision(1) << scientific << "\t&" << combinedRelativeError;
      if (i > 0) {
        if (previousRelativeError >= 0.0) {
          double rate = - log(combinedRelativeError/previousRelativeError) / log(2.0);
          texString << "\t&" << setprecision(2) << fixed << rate;
        } else {
          // bad practice: defensive coding...
          cout << "Unexpected previousRelativeError.\n";
        }
      } else {
        texString << "\t&-";
      }
      previousRelativeError = combinedRelativeError;
    }
    texString << newLine;
  }
  texString << "\\hline \n";

  if (filePathPrefix.length() > 0) { // then write to file
    ostringstream fileName;
    fileName << filePathPrefix << "_error_rate_table.tex";
    ofstream fout(fileName.str().c_str());
    fout << texString.str();
    fout.close();
  }
  return texString.str();
}

string HConvergenceStudy::TeXNumGlobalDofsTable(const string &filePathPrefix) {
  ostringstream texString;
  texString << scientific;

  int numColumns = 2;
  string newLine = "\\\\\n";
  texString << "\\multicolumn{" << numColumns << "}{| c |}{k=" << _H1Order-1 << "} " << newLine;

  texString << "\\hline \n";
  texString << "Mesh Size & Global Dofs";
  texString << newLine << "\\cline{2-" << numColumns << "}" << "\n";
  texString << newLine;
  texString << "\\hline \n";
  vector<int> meshSizes = this->meshSizes();
  for (int i=0; i<meshSizes.size(); i++) {
    int size = meshSizes[i];
    texString << size << " $\\times$ " << size;
    texString << "\t&" << _solutions[i]->mesh()->numGlobalDofs();
    texString << newLine;
  }
  texString << "\\hline \n";
  if (filePathPrefix.length() > 0) { // then write to file
    ostringstream fileName;
    fileName << filePathPrefix << "_mesh_sizes.tex";
    ofstream fout(fileName.str().c_str());
    fout << texString.str();
    fout.close();
  }
  return texString.str();
}

string HConvergenceStudy::convergenceDataMATLAB(int trialID, int minPolyOrder) {
  ostringstream ss;
  int k = _H1Order - 1; // order of the solution
  int matlabIndex = k - minPolyOrder + 1;
//  cout << "matlabIndex:" << matlabIndex << endl;
  ss << scientific;
  ss << _bilinearForm->trialName(trialID) << "Error{" << matlabIndex << "} = [";

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

  ss << _bilinearForm->trialName(trialID) << "BestError{" << matlabIndex << "} = [";

  for (int i=0; i<meshSizes.size(); i++) {
    int size = meshSizes[i];
    double h = 1.0 / size;
    ss << k << " " << h << " ";
    double bestError = _bestApproximationErrors[trialID][i];
    if (_reportRelativeErrors) {
      bestError /= _exactSolutionNorm[trialID];
    }
    ss << bestError << "\n";
  }

  ss << "];\n";
  return ss.str();
}

string HConvergenceStudy::TeXBestApproximationComparisonTable(const string &filePathPrefix) {
  vector<int> trialIDs = _bilinearForm->trialVolumeIDs();
  return this->TeXBestApproximationComparisonTable(trialIDs, filePathPrefix);
}

string HConvergenceStudy::TeXBestApproximationComparisonTable( const vector<int> &trialIDs, const string &filePathPrefix ) {
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

  if (filePathPrefix.length() > 0) { // then write to file
    ostringstream fileName;
    fileName << filePathPrefix << "_best_compare_table.tex";
    ofstream fout(fileName.str().c_str());
    fout << texString.str();
    fout.close();
  }
  return texString.str();
}

void HConvergenceStudy::writeToFiles(const string & filePathPrefix, int trialID, int traceID, bool writeMATLABPlotData) {
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

  double l2norm = _exactSolution->L2NormOfError(_fineZeroSolution, trialID, _cubatureDegreeForExact);

  for (int i=0; i<_solutions.size(); i++) {
    TSolutionPtr<double> solution = _solutions[i];
    TSolutionPtr<double> bestApproximation = _bestApproximations[i];
    double l2error = _solutionErrors[trialID][i];

    if (_reportRelativeErrors) {
      l2error /= l2norm;
    }

    string spaces = (numElements > 10) ? " " : "   ";

    cout << scientific;
    fout << scientific;
    cout << numElements << "x" << numElements << " mesh:" << spaces << setprecision(1) <<  l2error;
    fout << numElements << "x" << numElements << "\t" << setprecision(1) << l2error;
    if (i > 0) {
      double rate = _solutionRates[trialID][i-1];
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

    if (writeMATLABPlotData) {
      // now write out the solution for MATLAB plotting...
      fileName.str(""); // clear out the filename
      fileName << filePathPrefix << "_solution_" << numElements << "x" << numElements << ".m";
      //    solution.writeToFile(trialID, fileName.str());
      solution->writeFieldsToFile(trialID, fileName.str());
      fileName.str("");

      fileName.str(""); // clear out the filename
      fileName << filePathPrefix << "_best_approximation_" << numElements << "x" << numElements << ".m";
      //    solution.writeToFile(trialID, fileName.str());
      bestApproximation->writeFieldsToFile(trialID, fileName.str());
      fileName.str("");

      fileName << filePathPrefix << "_exact_" << numElements << "x" << numElements << ".m";
      _exactSolutionFunctions[trialID]->writeValuesToMATLABFile(solution->mesh(), fileName.str());
      fileName.str(""); // clear out the filename
      if (traceID != -1) {
        fileName << filePathPrefix << "_trace_solution_" << numElements << "x" << numElements << ".dat";
        solution->writeFluxesToFile(traceID, fileName.str());
      }
    }
    numElements *= 2;
  }
  fout.close();

  cout << "***********  BEST APPROXIMATION ERROR for : " << _bilinearForm->trialName(trialID) << " **************\n";
  numElements = minNumElements;
  for (int i=0; i<_bestApproximations.size(); i++) {
    TSolutionPtr<double> bestApproximation = _bestApproximations[i];
    double l2error = _bestApproximationErrors[trialID][i];
    double newl2error = _exactSolution->L2NormOfError(bestApproximation, trialID, _cubatureDegreeForExact); // 15 == cubature degree to use...
    if (l2error != newl2error) {
      cout << "best L2 error has changed since computeErrors() was called.\n";
      cout << "old error: " << l2error << "; new: " << newl2error << endl;
    }
    if (_reportRelativeErrors) {
      l2error /= l2norm;
    }

    string spaces = (numElements > 10) ? " " : "   ";
    cout << scientific;
    cout << numElements << "x" << numElements << " mesh:" << spaces << setprecision(1) <<  l2error;
    if (i > 0) {
      double rate = _bestApproximationRates[trialID][i-1];
      cout << "\t(rate: " << setprecision(2) << fixed << rate << ")";
    }
    cout << endl;

    numElements *= 2;
  }
  cout << "**************************************************************\n";

  cout << scientific << setprecision(1);
  cout << "L2 norm of solution: " << l2norm  << endl;
}

void HConvergenceStudy::setSolver(Teuchos::RCP<Solver> solver) {
  _solver = solver;
}

void HConvergenceStudy::setUseCondensedSolve(bool value) {
  _useCondensedSolve = value;
}

void HConvergenceStudy::setWriteGlobalStiffnessToDisk(bool value, string globalStiffnessFilePrefix) {
  _writeGlobalStiffnessToDisk = value;
  _globalStiffnessFilePrefix = globalStiffnessFilePrefix;
}

vector<double> HConvergenceStudy::weightedL2Error(map<int, double> &weights, bool bestApproximation, bool relativeErrors) {
  map< int, vector<double> > errors = bestApproximation ? bestApproximationErrors() : solutionErrors();

  vector<double> weightedError(_solutions.size());
  double weightedSolutionNorm = 0;

  for (map<int, double>::iterator weightIt = weights.begin(); weightIt != weights.end(); weightIt++) {
    int trialID = weightIt->first;
    double weight = weightIt->second;
    for (int i=0; i < errors[trialID].size(); i++) {
      double err = errors[trialID][i];
      weightedError[i] += err * err * weight * weight;
    }
    weightedSolutionNorm += _exactSolutionNorm[trialID] * _exactSolutionNorm[trialID] * weight * weight;
  }
  // take square roots:
  weightedSolutionNorm = sqrt(weightedSolutionNorm);
  for (int i=0; i<weightedError.size(); i++) {
    weightedError[i] = sqrt(weightedError[i]);
    if (relativeErrors) {
      weightedError[i] /= weightedSolutionNorm;
    }
  }
  return weightedError;
}
