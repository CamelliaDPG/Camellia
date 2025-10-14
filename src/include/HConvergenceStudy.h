// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#ifndef DPG_H_CONVERGENCE_STUDY
#define DPG_H_CONVERGENCE_STUDY

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

namespace Camellia
{
struct DerivedVariable
{
  string name;
  LinearTermPtr term;
};

class HConvergenceStudy
{
  Teuchos::RCP<ExactSolution<double>> _exactSolution;
  BFPtr _bilinearForm;
  Teuchos::RCP<RHS> _rhs;
  Teuchos::RCP<BC> _bc;
  IPPtr _ip;
  Teuchos::RCP<LagrangeConstraints> _lagrangeConstraints;

  bool _reportConditionNumber;

  int _H1Order, _minLogElements, _maxLogElements, _pToAdd;
  int _cubatureDegreeForExact;
  int _cubatureEnrichmentForSolutions;
  vector< TSolutionPtr<double> > _solutions;
  vector< TSolutionPtr<double> > _bestApproximations;

  map< int, TFunctionPtr<double> > _exactSolutionFunctions;

  TSolutionPtr<double> _fineZeroSolution;
  bool _useTriangles;
  bool _useHybrid;
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

  TSolutionPtr<double> bestApproximation(Teuchos::RCP<Mesh> mesh);

  Teuchos::RCP<Mesh> buildMesh(Teuchos::RCP<MeshGeometry> geometry, int numRefinements);
public:
  HConvergenceStudy(Teuchos::RCP<ExactSolution<double>> exactSolution,
                    BFPtr bilinearForm,
                    Teuchos::RCP<RHS> rhs,
                    Teuchos::RCP<BC> bc,
                    IPPtr ip,
                    int minLogElements, int maxLogElements, int H1Order, int pToAdd,
                    bool randomRefinements=false, bool useTriangles=false, bool useHybrid=false);
  void setLagrangeConstraints(Teuchos::RCP<LagrangeConstraints> lagrangeConstraints);
  void setReportConditionNumber(bool value);
  void setReportRelativeErrors(bool reportRelativeErrors);
  void solve(const Camellia::FieldContainer<double> &quadPoints,
             map<int,int> trialOrderEnhancements,
             map<int,int> testOrderEnhancements);
  void solve(const Camellia::FieldContainer<double> &quadPoints);
  void solve(Teuchos::RCP< MeshGeometry > geometry);
  TSolutionPtr<double> getSolution(int logElements); // logElements: a number between minLogElements and maxLogElements
  void writeToFiles(const string & filePathPrefix, int trialID, int traceID = -1, bool writeMATLABPlotData = false);

  void addDerivedVariable( LinearTermPtr derivedVar, const string & name );

  BFPtr bilinearForm();

  vector<int> meshSizes();
  vector< TSolutionPtr<double> >& bestApproximations();

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

  void setSolutions( vector< TSolutionPtr<double> > &solutions); // must be in the right order, from minLogElements to maxLogElements

  void setSolver( Teuchos::RCP<Solver> solver);

  void setUseCondensedSolve(bool value);

  void setWriteGlobalStiffnessToDisk(bool value, string globalStiffnessFilePrefix);
};
}

#endif
