#ifndef DPG_H_CONVERGENCE_STUDY
#define DPG_H_CONVERGENCE_STUDY

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
 *  HConvergenceStudy.h
 *
 *  Created by Nathan Roberts on 7/21/11.
 *
 */

#include "TypeDefs.h"

#include "Mesh.h"
#include "ExactSolution.h"
#include "Solver.h"
#include "LinearTerm.h"
#include "Constraint.h"

using namespace std;

namespace Camellia {
  struct DerivedVariable {
    string name;
    LinearTermPtr term;
  };

  class HConvergenceStudy {
    Teuchos::RCP<ExactSolution> _exactSolution;
    BFPtr _bilinearForm;
    Teuchos::RCP<RHS> _rhs;
    Teuchos::RCP<BC> _bc;
    IPPtr _ip;
    Teuchos::RCP<LagrangeConstraints> _lagrangeConstraints;

    bool _reportConditionNumber;

    int _H1Order, _minLogElements, _maxLogElements, _pToAdd;
    int _cubatureDegreeForExact;
    int _cubatureEnrichmentForSolutions;
    vector< SolutionPtr<double> > _solutions;
    vector< SolutionPtr<double> > _bestApproximations;

    map< int, TFunctionPtr<double> > _exactSolutionFunctions;

    SolutionPtr<double> _fineZeroSolution;
    bool _randomRefinements;
    bool _useTriangles;
    bool _useHybrid;
    void randomlyRefine(Teuchos::RCP<Mesh> mesh);
    bool _reportRelativeErrors;

    bool _writeGlobalStiffnessToDisk;
    string _globalStiffnessFilePrefix;

    map< int, vector<double> > _bestApproximationErrors; // trialID --> vector of errors for various meshes
    map< int, vector<double> > _solutionErrors;
    map< int, vector<double> > _bestApproximationErrorsDerivedVariables; // derived var index --> vector of errors for various meshes
    map< int, vector<double> > _solutionErrorsDerivedVariables;

    map< int, vector<double> > _bestApproximationRates;
    map< int, vector<double> > _solutionRates;
    map< int, vector<double> > _bestApproximationRatesDerivedVariables;
    map< int, vector<double> > _solutionRatesDerivedVariables;

    map< int, double > _exactSolutionNorm;

    vector< DerivedVariable > _derivedVariables;

    Teuchos::RCP<Solver> _solver;
    bool _useCondensedSolve;

    int minNumElements();

    SolutionPtr<double> bestApproximation(Teuchos::RCP<Mesh> mesh);

    Teuchos::RCP<Mesh> buildMesh(Teuchos::RCP<MeshGeometry> geometry, int numRefinements,
                                 bool useConformingTraces );
  public:
    HConvergenceStudy(Teuchos::RCP<ExactSolution> exactSolution,
                      BFPtr bilinearForm,
                      Teuchos::RCP<RHS> rhs,
                      Teuchos::RCP<BC> bc,
                      IPPtr ip,
                      int minLogElements, int maxLogElements, int H1Order, int pToAdd,
                      bool randomRefinements=false, bool useTriangles=false, bool useHybrid=false);
    void setLagrangeConstraints(Teuchos::RCP<LagrangeConstraints> lagrangeConstraints);
    void setReportConditionNumber(bool value);
    void setReportRelativeErrors(bool reportRelativeErrors);
    void solve(const Intrepid::FieldContainer<double> &quadPoints, bool useConformingTraces,
               map<int,int> trialOrderEnhancements,
               map<int,int> testOrderEnhancements);
    void solve(const Intrepid::FieldContainer<double> &quadPoints, bool useConformingTraces = true);
    void solve(Teuchos::RCP< MeshGeometry > geometry, bool useConformingTraces=true);
    SolutionPtr<double> getSolution(int logElements); // logElements: a number between minLogElements and maxLogElements
    void writeToFiles(const string & filePathPrefix, int trialID, int traceID = -1, bool writeMATLABPlotData = false);

    void addDerivedVariable( LinearTermPtr derivedVar, const string & name );

    BFPtr bilinearForm();

    vector<int> meshSizes();
    vector< SolutionPtr<double> >& bestApproximations();

    map< int, vector<double> > bestApproximationErrors();
    map< int, vector<double> > solutionErrors();

    map< int, vector<double> > bestApproximationRates();
    map< int, vector<double> > solutionRates();

    map< int, double > exactSolutionNorm();

    vector<double> weightedL2Error(map<int, double> &weights, bool bestApproximation=false, bool relativeErrors=true);

    void computeErrors();
    double computeJacobiPreconditionedConditionNumber(int logElements);
    string convergenceDataMATLAB(int trialID, int minPolyOrder = 1);
    string TeXErrorRateTable(const string &filePathPrefix="");
    string TeXErrorRateTable(const vector<int> &trialIDs, const string &filePathPrefix="");
    string TeXBestApproximationComparisonTable(const string &filePathPrefix="");
    string TeXBestApproximationComparisonTable(const vector<int> &trialIDs, const string &filePathPrefix="");
    string TeXNumGlobalDofsTable(const string &filePathPrefix="");

    void setCubatureDegreeForExact(int value);

    void setCubatureEnrichmentForSolutions(int value);

    void setSolutions( vector< SolutionPtr<double> > &solutions); // must be in the right order, from minLogElements to maxLogElements

    void setSolver( Teuchos::RCP<Solver> solver);

    void setUseCondensedSolve(bool value);

    void setWriteGlobalStiffnessToDisk(bool value, string globalStiffnessFilePrefix);
  };
}

#endif
