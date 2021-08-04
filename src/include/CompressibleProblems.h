// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

#include "Function.h"
#include "PenaltyConstraints.h"
#include "SimpleFunction.h"
#include "SpaceTimeCompressibleFormulation.h"

namespace Camellia
{
class CompressibleProblem
{
  protected:
    Teuchos::RCP<PenaltyConstraints> _pc = Teuchos::null;
    FunctionPtr _rho_exact;
    FunctionPtr _u1_exact;
    FunctionPtr _u2_exact;
    FunctionPtr _u3_exact;
    FunctionPtr _T_exact;
    double _gamma = 1.4;
    double _Pr = 0.713;
    double _Cv;
    double _Cp;
    double _R;
    double _tInit;
    double _tFinal;
    int _numSlabs = 1;
    int _currentStep = 0;
    bool _steady;
  public:
    double gamma() { return _gamma; }
    double Pr() { return _Pr; }
    double Cv() { return _Cv; }
    double Cp() { return _Cp; }
    double R() { return _R; }
    LinearTermPtr forcingTerm = Teuchos::null;
    FunctionPtr rho_exact() { return _rho_exact; }
    FunctionPtr u1_exact() { return _u1_exact; }
    FunctionPtr u2_exact() { return _u2_exact; }
    FunctionPtr u3_exact() { return _u3_exact; }
    FunctionPtr T_exact() { return _T_exact; }
    virtual MeshTopologyPtr meshTopology(int temporalDivisions=1) = 0;
    virtual MeshGeometryPtr meshGeometry() { return Teuchos::null; }
    virtual void preprocessMesh(MeshPtr proxyMesh) {};
    virtual void setBCs(SpaceTimeCompressibleFormulationPtr form) = 0;
    Teuchos::RCP<PenaltyConstraints> pc() { return _pc; }
    virtual double computeL2Error(SpaceTimeCompressibleFormulationPtr form, SolutionPtr solutionBackground) { return 0; }
    int numSlabs() { return _numSlabs; }
    int currentStep() { return _currentStep; }
    void advanceStep() { _currentStep++; }
    double stepSize() { return (_tFinal-_tInit)/_numSlabs; }
    double currentT0() { return stepSize()*_currentStep; }
    double currentT1() { return stepSize()*(_currentStep+1); }
};

class AnalyticalCompressibleProblem : public CompressibleProblem
{
protected:
    vector<double> _x0;
    vector<double> _dimensions;
    vector<int> _elementCounts;
    map<int, FunctionPtr> _exactMap;
public:
  virtual MeshTopologyPtr meshTopology(int temporalDivisions=1);

  void initializeExactMap(SpaceTimeCompressibleFormulationPtr form);

  void projectExactSolution(SolutionPtr solution);

  virtual void setBCs(SpaceTimeCompressibleFormulationPtr form);

  double computeL2Error(SpaceTimeCompressibleFormulationPtr form, SolutionPtr solutionBackground);
};

class TrivialCompressible : public AnalyticalCompressibleProblem
{
public:
  TrivialCompressible(bool steady, double Re, int spaceDim);
};

class SimpleRarefaction : public AnalyticalCompressibleProblem
{
  public:
  SimpleRarefaction(bool steady, double Re, int spaceDim);
};

class SimpleShock : public AnalyticalCompressibleProblem
{
  public:
  SimpleShock(bool steady, double Re, int spaceDim);
  void setBCs(SpaceTimeCompressibleFormulationPtr form);
};

class Noh_u1 : public SimpleFunction<double>
{
public:
  Noh_u1();
  
  using SimpleFunction<double>::value;
  double value(double x, double y);
};

class Noh_u2 : public SimpleFunction<double>
{
public:
  Noh_u2();
  
  using SimpleFunction<double>::value;
  double value(double x, double y);
};

class Noh : public AnalyticalCompressibleProblem
{
  public:
  Noh(bool steady, double Re, int spaceDim);

  void setBCs(SpaceTimeCompressibleFormulationPtr form);
};

class Sedov : public AnalyticalCompressibleProblem
{
public:
  Sedov(bool steady, double Re, int spaceDim);
};

}
