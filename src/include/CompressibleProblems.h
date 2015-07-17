
#include "Function.h"
#include "SpatialFilter.h"
#include "ExpFunction.h"
#include "TrigFunctions.h"
#include "PenaltyConstraints.h"
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
    virtual MeshTopologyPtr meshTopology(int temporalDivisions=1)
    {
      MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(_dimensions, _elementCounts, _x0);
      MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, currentT0(), currentT1(), temporalDivisions);
      if (_steady)
        return spatialMeshTopo;
      else
        return spaceTimeMeshTopo;
    }

    void initializeExactMap(SpaceTimeCompressibleFormulationPtr form)
    {
      _exactMap[form->rho()->ID()] = _rho_exact;
      _exactMap[form->T()->ID()] = _T_exact;
      switch (form->spaceDim())
      {
        case 1:
          _exactMap[form->u(1)->ID()] = _u1_exact;
          _exactMap[form->uhat(1)->ID()] = form->uhat(1)->termTraced()->evaluate(_exactMap);
          break;
        case 2:
          _exactMap[form->u(1)->ID()] = _u1_exact;
          _exactMap[form->u(2)->ID()] = _u2_exact;
          _exactMap[form->uhat(1)->ID()] = form->uhat(1)->termTraced()->evaluate(_exactMap);
          _exactMap[form->uhat(2)->ID()] = form->uhat(2)->termTraced()->evaluate(_exactMap);
          break;
        case 3:
          _exactMap[form->u(1)->ID()] = _u1_exact;
          _exactMap[form->u(2)->ID()] = _u2_exact;
          _exactMap[form->u(3)->ID()] = _u3_exact;
          _exactMap[form->uhat(1)->ID()] = form->uhat(1)->termTraced()->evaluate(_exactMap);
          _exactMap[form->uhat(2)->ID()] = form->uhat(2)->termTraced()->evaluate(_exactMap);
          _exactMap[form->uhat(3)->ID()] = form->uhat(3)->termTraced()->evaluate(_exactMap);
          break;
      }
    }

    void projectExactSolution(SolutionPtr solution)
    {
      solution->projectOntoMesh(_exactMap);
    }

    virtual void setBCs(SpaceTimeCompressibleFormulationPtr form)
    {
      initializeExactMap(form);

      BCPtr bc = form->solutionUpdate()->bc();
      SpatialFilterPtr initTime = SpatialFilter::matchingT(_tInit);
      SpatialFilterPtr leftX  = SpatialFilter::matchingX(_x0[0]);
      SpatialFilterPtr rightX = SpatialFilter::matchingX(_x0[0]+_dimensions[0]);
      SpatialFilterPtr leftY  = SpatialFilter::matchingY(_x0[1]);
      SpatialFilterPtr rightY = SpatialFilter::matchingY(_x0[1]+_dimensions[1]);
      FunctionPtr one = Function::constant(1);
      switch (form->spaceDim())
      {
        case 1:
          bc->addDirichlet(form->tc(),   leftX,  -_rho_exact*_u1_exact );
          bc->addDirichlet(form->tc(),   rightX,  _rho_exact*_u1_exact );
          bc->addDirichlet(form->tm(1),  leftX, -(_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact));
          bc->addDirichlet(form->tm(1),  rightX, (_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact));
          bc->addDirichlet(form->te(),   leftX, -(_rho_exact*_Cv*_T_exact+0.5*_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact)*_u1_exact);
          bc->addDirichlet(form->te(),   rightX, (_rho_exact*_Cv*_T_exact+0.5*_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact)*_u1_exact);

          // bc->addDirichlet(form->uhat(1), leftX,    _exactMap[form->uhat(1)->ID()]);
          // bc->addDirichlet(form->uhat(2), leftX,    _exactMap[form->uhat(2)->ID()]);
          // bc->addDirichlet(form->uhat(1), rightX,   _exactMap[form->uhat(1)->ID()]);
          // bc->addDirichlet(form->uhat(2), rightX,   _exactMap[form->uhat(2)->ID()]);
          // bc->addDirichlet(form->T(),     leftX,    _exactMap[form->T()->ID()]);
          // bc->addDirichlet(form->T(),     rightX,   _exactMap[form->T()->ID()]);
          if (!_steady)
          {
            FunctionPtr rho_init = _exactMap[form->rho()->ID()];
            FunctionPtr u1_init  = _exactMap[form->u(1)->ID()];
            FunctionPtr T_init   = _exactMap[form->T()->ID()];
            FunctionPtr m1_init = rho_init*u1_init;
            FunctionPtr E_init = rho_init*(_Cv*T_init + 0.5*u1_init*u1_init);
            bc->addDirichlet(form->tc(), initTime,-rho_init);
            bc->addDirichlet(form->tm(1),initTime,-m1_init);
            bc->addDirichlet(form->te(), initTime,-E_init);
          }
          break;
        case 2:
          bc->addDirichlet(form->tc(),   leftX,  -_rho_exact*_u1_exact );
          bc->addDirichlet(form->tc(),   rightX,  _rho_exact*_u1_exact );
          bc->addDirichlet(form->tc(),   leftY,  -_rho_exact*_u2_exact );
          bc->addDirichlet(form->tc(),   rightY,  _rho_exact*_u2_exact );
          bc->addDirichlet(form->tm(1),  leftX, -(_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact));
          bc->addDirichlet(form->tm(1),  rightX, (_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact));
          bc->addDirichlet(form->tm(2),  leftX, -(_rho_exact*_u1_exact*_u2_exact));
          bc->addDirichlet(form->tm(2),  rightX, (_rho_exact*_u1_exact*_u2_exact));
          bc->addDirichlet(form->tm(1),  leftY, -(_rho_exact*_u1_exact*_u2_exact));
          bc->addDirichlet(form->tm(1),  rightY, (_rho_exact*_u1_exact*_u2_exact));
          bc->addDirichlet(form->tm(2),  leftY, -(_rho_exact*_u2_exact*_u2_exact+_R*_rho_exact*_T_exact));
          bc->addDirichlet(form->tm(2),  rightY, (_rho_exact*_u2_exact*_u2_exact+_R*_rho_exact*_T_exact));
          bc->addDirichlet(form->te(),   leftX, -(_rho_exact*_Cv*_T_exact+0.5*_rho_exact*(_u1_exact*_u1_exact+_u2_exact*_u2_exact)+_R*_rho_exact*_T_exact)*_u1_exact);
          bc->addDirichlet(form->te(),   rightX, (_rho_exact*_Cv*_T_exact+0.5*_rho_exact*(_u1_exact*_u1_exact+_u2_exact*_u2_exact)+_R*_rho_exact*_T_exact)*_u1_exact);
          bc->addDirichlet(form->te(),   leftY, -(_rho_exact*_Cv*_T_exact+0.5*_rho_exact*(_u1_exact*_u1_exact+_u2_exact*_u2_exact)+_R*_rho_exact*_T_exact)*_u2_exact);
          bc->addDirichlet(form->te(),   rightY, (_rho_exact*_Cv*_T_exact+0.5*_rho_exact*(_u1_exact*_u1_exact+_u2_exact*_u2_exact)+_R*_rho_exact*_T_exact)*_u2_exact);

          if (!_steady)
          {
            FunctionPtr rho_init = _exactMap[form->rho()->ID()];
            FunctionPtr u1_init  = _exactMap[form->u(1)->ID()];
            FunctionPtr u2_init  = _exactMap[form->u(2)->ID()];
            FunctionPtr T_init   = _exactMap[form->T()->ID()];
            FunctionPtr m1_init = rho_init*u1_init;
            FunctionPtr m2_init = rho_init*u2_init;
            FunctionPtr E_init = rho_init*(_Cv*T_init + 0.5*(u1_init*u1_init+u2_init*u2_init));
            bc->addDirichlet(form->tc(), initTime,-rho_init);
            bc->addDirichlet(form->tm(1),initTime,-m1_init);
            bc->addDirichlet(form->tm(2),initTime,-m2_init);
            bc->addDirichlet(form->te(), initTime,-E_init);
          }
          // bc->addDirichlet(form->rho(),   leftX,    _exactMap[form->rho()->ID()]);
          // bc->addDirichlet(form->rho(),   rightX,   _exactMap[form->rho()->ID()]);
          // bc->addDirichlet(form->rho(),   leftY,    _exactMap[form->rho()->ID()]);
          // bc->addDirichlet(form->rho(),   rightY,   _exactMap[form->rho()->ID()]);
          // bc->addDirichlet(form->uhat(1), leftX,    _exactMap[form->uhat(1)->ID()]);
          // bc->addDirichlet(form->uhat(2), leftX,    _exactMap[form->uhat(2)->ID()]);
          // bc->addDirichlet(form->uhat(1), rightX,   _exactMap[form->uhat(1)->ID()]);
          // bc->addDirichlet(form->uhat(2), rightX,   _exactMap[form->uhat(2)->ID()]);
          // bc->addDirichlet(form->uhat(1), leftY,    _exactMap[form->uhat(1)->ID()]);
          // bc->addDirichlet(form->uhat(2), leftY,    _exactMap[form->uhat(2)->ID()]);
          // bc->addDirichlet(form->uhat(1), rightY,   _exactMap[form->uhat(1)->ID()]);
          // bc->addDirichlet(form->uhat(2), rightY,   _exactMap[form->uhat(2)->ID()]);
          // bc->addDirichlet(form->T(),     leftX,    _exactMap[form->T()->ID()]);
          // bc->addDirichlet(form->T(),     rightX,   _exactMap[form->T()->ID()]);
          // bc->addDirichlet(form->T(),     leftY,    _exactMap[form->T()->ID()]);
          // bc->addDirichlet(form->T(),     rightY,   _exactMap[form->T()->ID()]);
          break;
        case 3:
          break;
      }
    }

    double computeL2Error(SpaceTimeCompressibleFormulationPtr form, SolutionPtr solutionBackground)
    {
      FunctionPtr rho_soln, u1_soln, u2_soln, u3_soln, T_soln,
                  rho_diff, u1_diff, u2_diff, u3_diff, T_diff,
                  rho_sqr, u1_sqr, u2_sqr, u3_sqr, T_sqr;
      rho_soln = Function::solution(form->rho(), solutionBackground);
      u1_soln = Function::solution(form->u(1), solutionBackground);
      if (form->spaceDim() > 1)
        u2_soln = Function::solution(form->u(2), solutionBackground);
      if (form->spaceDim() > 2)
        u3_soln = Function::solution(form->u(3), solutionBackground);
      T_soln = Function::solution(form->T(), solutionBackground);
      rho_diff = rho_soln - _rho_exact;
      u1_diff = u1_soln - _u1_exact;
      if (form->spaceDim() > 1)
        u2_diff = u2_soln - _u2_exact;
      if (form->spaceDim() > 2)
        u3_diff = u3_soln - _u3_exact;
      T_diff = T_soln - _T_exact;
      rho_sqr = rho_diff*rho_diff;
      u1_sqr = u1_diff*u1_diff;
      if (form->spaceDim() > 1)
        u2_sqr = u2_diff*u2_diff;
      if (form->spaceDim() > 2)
        u3_sqr = u3_diff*u3_diff;
      T_sqr = T_diff*T_diff;
      double rho_l2, u1_l2, u2_l2 = 0, u3_l2 = 0, T_l2;
      rho_l2 = rho_sqr->integrate(solutionBackground->mesh(), 5);
      u1_l2 = u1_sqr->integrate(solutionBackground->mesh(), 5);
      if (form->spaceDim() > 1)
        u2_l2 = u2_sqr->integrate(solutionBackground->mesh(), 5);
      if (form->spaceDim() > 2)
        u3_l2 = u3_sqr->integrate(solutionBackground->mesh(), 5);
      T_l2 = T_sqr->integrate(solutionBackground->mesh(), 5);
      double l2Error = sqrt(rho_l2+u1_l2+u2_l2+u3_l2+T_l2);
      return l2Error;
    }
};

class TrivialCompressible : public AnalyticalCompressibleProblem
{
  private:
  public:
    TrivialCompressible(bool steady, double Re, int spaceDim)
    {
      _steady = steady;
      _gamma = 1.4;
      // double p0 = 1;
      double rho0 = 1;
      double u0 = 0;
      // double a0 = sqrt(_gamma*p0/rho0);
      // double M_inf = u0/a0;
      // _Cv = 1./(_gamma*(_gamma-1)*M_inf*M_inf);
      _Cv = 1;
      _Cp = _gamma*_Cv;
      _R = _Cp-_Cv;
      _rho_exact = Function::constant(rho0);
      _u1_exact = Function::constant(u0);
      _u2_exact = Function::constant(2*u0);
      _u3_exact = Function::constant(3*u0);
      // _u1_exact = Function::zero();
      // _u2_exact = Function::zero();
      // _u3_exact = Function::zero();
      // _T_exact = Function::constant(p0/(rho0*_R));
      _T_exact = Function::constant(1);
      // _T_exact = Function::zero();

      for (int d=0; d < spaceDim; d++)
      {
        _x0.push_back(0);
        _dimensions.push_back(1);
        _elementCounts.push_back(2);
      }
      _tInit = 0.0;
      _tFinal = 0.5;
    }
};

class SimpleRarefaction : public AnalyticalCompressibleProblem
{
  private:
  public:
    SimpleRarefaction(bool steady, double Re, int spaceDim)
    {
      _steady = steady;
      _gamma = 1.4;
      double p0 = 1;
      double rho0 = 1;
      double u0 = 1;
      double a0 = sqrt(_gamma*p0/rho0);
      double M_inf = u0/a0;
      _Cv = 1./(_gamma*(_gamma-1)*M_inf*M_inf);
      _Cp = _gamma*_Cv;
      _R = _Cp-_Cv;
      _rho_exact = Function::constant(rho0);
      _T_exact = Function::constant(p0/(rho0*_R));
      _u1_exact = Function::heaviside(0.5);
      _u2_exact = Function::zero();
      _u3_exact = Function::zero();
      // _T_exact = Function::constant(1);

      for (int d=0; d < spaceDim; d++)
      {
        _x0.push_back(0);
        _dimensions.push_back(1);
        _elementCounts.push_back(8);
      }
      _tInit = 0.0;
      _tFinal = 0.1;
    }
};

class SimpleShock : public AnalyticalCompressibleProblem
{
  private:
  public:
    SimpleShock(bool steady, double Re, int spaceDim)
    {
      _steady = steady;
      _gamma = 1.4;
      double p0 = 1;
      double rho0 = 1;
      double u0 = 1;
      double a0 = sqrt(_gamma*p0/rho0);
      double M_inf = u0/a0;
      _Cv = 1./(_gamma*(_gamma-1)*M_inf*M_inf);
      _Cp = _gamma*_Cv;
      _R = _Cp-_Cv;
      _rho_exact = Function::constant(rho0);
      _T_exact = Function::constant(p0/(rho0*_R));
      _u1_exact = Function::constant(1) - Function::heaviside(0.5);
      _u2_exact = Function::zero();
      _u3_exact = Function::zero();
      // _T_exact = Function::constant(1);

      for (int d=0; d < spaceDim; d++)
      {
        _x0.push_back(0);
        _dimensions.push_back(1);
        if (d == 0)
          _elementCounts.push_back(4);
        else
          _elementCounts.push_back(1);
      }
      _tInit = 0.0;
      _tFinal = 0.1;
    }

    void setBCs(SpaceTimeCompressibleFormulationPtr form)
    {
      initializeExactMap(form);

      BCPtr bc = form->solutionUpdate()->bc();
      SpatialFilterPtr initTime = SpatialFilter::matchingT(_tInit);
      SpatialFilterPtr leftX  = SpatialFilter::matchingX(_x0[0]);
      SpatialFilterPtr rightX = SpatialFilter::matchingX(_x0[0]+_dimensions[0]);
      SpatialFilterPtr leftY  = SpatialFilter::matchingY(_x0[1]);
      SpatialFilterPtr rightY = SpatialFilter::matchingY(_x0[1]+_dimensions[1]);
      FunctionPtr one = Function::constant(1);
      switch (form->spaceDim())
      {
        case 1:
          bc->addDirichlet(form->tc(),   leftX,  -_rho_exact*_u1_exact );
          bc->addDirichlet(form->tc(),   rightX,  _rho_exact*_u1_exact );
          bc->addDirichlet(form->tm(1),  leftX, -(_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact));
          bc->addDirichlet(form->tm(1),  rightX, (_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact));
          bc->addDirichlet(form->te(),   leftX, -(_rho_exact*_Cv*_T_exact+0.5*_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact)*_u1_exact);
          bc->addDirichlet(form->te(),   rightX, (_rho_exact*_Cv*_T_exact+0.5*_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact)*_u1_exact);

          if (!_steady)
          {
            FunctionPtr rho_init = _exactMap[form->rho()->ID()];
            FunctionPtr u1_init  = _exactMap[form->u(1)->ID()];
            FunctionPtr T_init   = _exactMap[form->T()->ID()];
            FunctionPtr m1_init = rho_init*u1_init;
            FunctionPtr E_init = rho_init*(_Cv*T_init + 0.5*u1_init*u1_init);
            bc->addDirichlet(form->tc(), initTime,-rho_init);
            bc->addDirichlet(form->tm(1),initTime,-m1_init);
            bc->addDirichlet(form->te(), initTime,-E_init);
          }
          break;
        case 2:
          bc->addDirichlet(form->tc(),   leftX,  -_rho_exact*_u1_exact );
          bc->addDirichlet(form->tc(),   rightX,  _rho_exact*_u1_exact );
          bc->addDirichlet(form->tc(),   leftY,  -_rho_exact*_u2_exact );
          bc->addDirichlet(form->tc(),   rightY,  _rho_exact*_u2_exact );
          bc->addDirichlet(form->tm(1),  leftX, -(_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact));
          bc->addDirichlet(form->tm(1),  rightX, (_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact));
          bc->addDirichlet(form->tm(2),  leftX, -(_rho_exact*_u1_exact*_u2_exact));
          bc->addDirichlet(form->tm(2),  rightX, (_rho_exact*_u1_exact*_u2_exact));
          // bc->addDirichlet(form->tm(1),  leftY, -(_rho_exact*_u1_exact*_u2_exact));
          // bc->addDirichlet(form->tm(1),  rightY, (_rho_exact*_u1_exact*_u2_exact));
          // bc->addDirichlet(form->tm(2),  leftY, -(_rho_exact*_u2_exact*_u2_exact+_R*_rho_exact*_T_exact));
          // bc->addDirichlet(form->tm(2),  rightY, (_rho_exact*_u2_exact*_u2_exact+_R*_rho_exact*_T_exact));
          bc->addDirichlet(form->tm(1),  leftY, -(_rho_exact*_u1_exact*_u2_exact));
          bc->addDirichlet(form->tm(1),  rightY, (_rho_exact*_u1_exact*_u2_exact));
          bc->addDirichlet(form->uhat(2),  leftY,  Function::zero());
          bc->addDirichlet(form->uhat(2),  rightY, Function::zero());
          bc->addDirichlet(form->te(),   leftX, -(_rho_exact*_Cv*_T_exact+0.5*_rho_exact*(_u1_exact*_u1_exact+_u2_exact*_u2_exact)+_R*_rho_exact*_T_exact)*_u1_exact);
          bc->addDirichlet(form->te(),   rightX, (_rho_exact*_Cv*_T_exact+0.5*_rho_exact*(_u1_exact*_u1_exact+_u2_exact*_u2_exact)+_R*_rho_exact*_T_exact)*_u1_exact);
          bc->addDirichlet(form->te(),   leftY, -(_rho_exact*_Cv*_T_exact+0.5*_rho_exact*(_u1_exact*_u1_exact+_u2_exact*_u2_exact)+_R*_rho_exact*_T_exact)*_u2_exact);
          bc->addDirichlet(form->te(),   rightY, (_rho_exact*_Cv*_T_exact+0.5*_rho_exact*(_u1_exact*_u1_exact+_u2_exact*_u2_exact)+_R*_rho_exact*_T_exact)*_u2_exact);

          if (!_steady)
          {
            FunctionPtr rho_init = _exactMap[form->rho()->ID()];
            FunctionPtr u1_init  = _exactMap[form->u(1)->ID()];
            FunctionPtr u2_init  = _exactMap[form->u(2)->ID()];
            FunctionPtr T_init   = _exactMap[form->T()->ID()];
            FunctionPtr m1_init = rho_init*u1_init;
            FunctionPtr m2_init = rho_init*u2_init;
            FunctionPtr E_init = rho_init*(_Cv*T_init + 0.5*(u1_init*u1_init+u2_init*u2_init));
            bc->addDirichlet(form->tc(), initTime,-rho_init);
            bc->addDirichlet(form->tm(1),initTime,-m1_init);
            bc->addDirichlet(form->tm(2),initTime,-m2_init);
            bc->addDirichlet(form->te(), initTime,-E_init);
          }
          break;
        case 3:
          break;
      }
    }
};

class Noh : public AnalyticalCompressibleProblem
{
  private:
  public:
    Noh(bool steady, double Re, int spaceDim)
    {
      _steady = steady;
      _gamma = 5./3;
      double p0 = 1;
      double rho0 = 1;
      double u0 = 1;
      double a0 = sqrt(_gamma*p0/rho0);
      double M_inf = u0/a0;
      _Cv = 1./(_gamma*(_gamma-1)*M_inf*M_inf);
      _Cp = _gamma*_Cv;
      _R = _Cp-_Cv;
      _rho_exact = Function::constant(rho0);
      // _T_exact = Function::constant(p0/(rho0*_R));
      _T_exact = Function::constant(0);
      if (spaceDim == 1)
        _u1_exact = Function::constant(1) - 2*Function::heaviside(0.0);
      else if (spaceDim == 2)
      {
        _u1_exact = Function::constant(1) - 2*Function::heaviside(0.0);
      }
      // _T_exact = Function::constant(1);

      for (int d=0; d < spaceDim; d++)
      {
        _x0.push_back(-.5);
        _dimensions.push_back(1);
        _elementCounts.push_back(2);
      }
      _tInit = 0.0;
      _tFinal = 0.5;
    }
};

class Sedov : public AnalyticalCompressibleProblem
{
  private:
  public:
    Sedov(bool steady, double Re, int spaceDim)
    {
      _steady = steady;
      _gamma = 5./3.;
      // double p0 = 1e-3;
      double rho0 = 1;
      double u0 = 1;
      // double a0 = sqrt(_gamma*p0/rho0);
      // double M_inf = u0/a0;
      // _Cv = 1./(_gamma*(_gamma-1)*M_inf*M_inf);
      _Cv = 1;
      _Cp = _gamma*_Cv;
      _R = _Cp-_Cv;
      // double T0 = p0/(rho0*_R);
      _rho_exact = Function::constant(rho0);
      double pulseStrength = 32;
      FunctionPtr pulseX = (Function::constant(1) - Function::heaviside(1./pulseStrength))*Function::heaviside(-1./pulseStrength);
      FunctionPtr pulseY = (Function::constant(1) - Function::heavisideY(1./pulseStrength))*Function::heavisideY(-1./pulseStrength);
      FunctionPtr smoothX = Function::constant(1)-pulseStrength*pulseStrength*Function::xn(2);
      FunctionPtr smoothXY = Function::constant(1)-pulseStrength*pulseStrength*(Function::xn(2)+Function::yn(2));
      if (spaceDim == 1)
        _T_exact = pulseStrength*smoothX*pulseX + Function::constant(1e-3);
      else
        _T_exact = pulseStrength*smoothXY*pulseX*pulseY + Function::constant(1e-3);
      _u1_exact = Function::zero();
      _u2_exact = Function::zero();
      _u3_exact = Function::zero();

      for (int d=0; d < spaceDim; d++)
      {
        _x0.push_back(-.5);
        _dimensions.push_back(1);
        _elementCounts.push_back(4);
      }
      _tInit = 0.0;
      _tFinal = 0.1;
    }

    // void setBCs(SpaceTimeCompressibleFormulationPtr form)
    // {
    //   initializeExactMap(form);

    //   BCPtr bc = form->solutionUpdate()->bc();
    //   SpatialFilterPtr initTime = SpatialFilter::matchingT(_tInit);
    //   SpatialFilterPtr leftX  = SpatialFilter::matchingX(_x0[0]);
    //   SpatialFilterPtr rightX = SpatialFilter::matchingX(_x0[0]+_dimensions[0]);
    //   SpatialFilterPtr leftY  = SpatialFilter::matchingY(_x0[1]);
    //   SpatialFilterPtr rightY = SpatialFilter::matchingY(_x0[1]+_dimensions[1]);
    //   FunctionPtr one = Function::constant(1);
    //   switch (form->spaceDim())
    //   {
    //     case 1:
    //       bc->addDirichlet(form->tc(),   leftX,  -_rho_exact*_u1_exact );
    //       bc->addDirichlet(form->tc(),   rightX,  _rho_exact*_u1_exact );
    //       bc->addDirichlet(form->te(),   leftX, -(_rho_exact*_Cv*_T_exact+0.5*_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact)*_u1_exact);
    //       bc->addDirichlet(form->te(),   rightX, (_rho_exact*_Cv*_T_exact+0.5*_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact)*_u1_exact);

    //       if (!_steady)
    //       {
    //         FunctionPtr rho_init = _exactMap[form->rho()->ID()];
    //         FunctionPtr u1_init  = _exactMap[form->u(1)->ID()];
    //         FunctionPtr T_init   = _exactMap[form->T()->ID()];
    //         FunctionPtr m1_init = rho_init*u1_init;
    //         FunctionPtr E_init = rho_init*(_Cv*T_init + 0.5*u1_init*u1_init);
    //         bc->addDirichlet(form->tc(), initTime,-rho_init);
    //         bc->addDirichlet(form->tm(1),initTime,-m1_init);
    //         bc->addDirichlet(form->te(), initTime,-E_init);
    //       }
    //       break;
    //     case 2:
    //       bc->addDirichlet(form->tc(),   leftX,  -_rho_exact*_u1_exact );
    //       bc->addDirichlet(form->tc(),   rightX,  _rho_exact*_u1_exact );
    //       bc->addDirichlet(form->tc(),   leftY,  -_rho_exact*_u2_exact );
    //       bc->addDirichlet(form->tc(),   rightY,  _rho_exact*_u2_exact );
    //       // bc->addDirichlet(form->tm(1),  leftX, -(_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact));
    //       // bc->addDirichlet(form->tm(1),  rightX, (_rho_exact*_u1_exact*_u1_exact+_R*_rho_exact*_T_exact));
    //       // bc->addDirichlet(form->tm(2),  leftX, -(_rho_exact*_u1_exact*_u2_exact));
    //       // bc->addDirichlet(form->tm(2),  rightX, (_rho_exact*_u1_exact*_u2_exact));
    //       // bc->addDirichlet(form->tm(1),  leftY, -(_rho_exact*_u1_exact*_u2_exact));
    //       // bc->addDirichlet(form->tm(1),  rightY, (_rho_exact*_u1_exact*_u2_exact));
    //       // bc->addDirichlet(form->tm(2),  leftY, -(_rho_exact*_u2_exact*_u2_exact+_R*_rho_exact*_T_exact));
    //       // bc->addDirichlet(form->tm(2),  rightY, (_rho_exact*_u2_exact*_u2_exact+_R*_rho_exact*_T_exact));
    //       bc->addDirichlet(form->tm(2),  leftX,  Function::zero());
    //       bc->addDirichlet(form->tm(2),  rightX, Function::zero());
    //       bc->addDirichlet(form->uhat(1), leftX,  Function::zero());
    //       bc->addDirichlet(form->uhat(1), rightX, Function::zero());
    //       bc->addDirichlet(form->tm(1),  leftY, -(_rho_exact*_u1_exact*_u2_exact));
    //       bc->addDirichlet(form->tm(1),  rightY, (_rho_exact*_u1_exact*_u2_exact));
    //       bc->addDirichlet(form->uhat(2),  leftY,  Function::zero());
    //       bc->addDirichlet(form->uhat(2),  rightY, Function::zero());
    //       bc->addDirichlet(form->te(),   leftX, -(_rho_exact*_Cv*_T_exact+0.5*_rho_exact*(_u1_exact*_u1_exact+_u2_exact*_u2_exact)+_R*_rho_exact*_T_exact)*_u1_exact);
    //       bc->addDirichlet(form->te(),   rightX, (_rho_exact*_Cv*_T_exact+0.5*_rho_exact*(_u1_exact*_u1_exact+_u2_exact*_u2_exact)+_R*_rho_exact*_T_exact)*_u1_exact);
    //       bc->addDirichlet(form->te(),   leftY, -(_rho_exact*_Cv*_T_exact+0.5*_rho_exact*(_u1_exact*_u1_exact+_u2_exact*_u2_exact)+_R*_rho_exact*_T_exact)*_u2_exact);
    //       bc->addDirichlet(form->te(),   rightY, (_rho_exact*_Cv*_T_exact+0.5*_rho_exact*(_u1_exact*_u1_exact+_u2_exact*_u2_exact)+_R*_rho_exact*_T_exact)*_u2_exact);

    //       if (!_steady)
    //       {
    //         FunctionPtr rho_init = _exactMap[form->rho()->ID()];
    //         FunctionPtr u1_init  = _exactMap[form->u(1)->ID()];
    //         FunctionPtr u2_init  = _exactMap[form->u(2)->ID()];
    //         FunctionPtr T_init   = _exactMap[form->T()->ID()];
    //         FunctionPtr m1_init = rho_init*u1_init;
    //         FunctionPtr m2_init = rho_init*u2_init;
    //         FunctionPtr E_init = rho_init*(_Cv*T_init + 0.5*(u1_init*u1_init+u2_init*u2_init));
    //         bc->addDirichlet(form->tc(), initTime,-rho_init);
    //         bc->addDirichlet(form->tm(1),initTime,-m1_init);
    //         bc->addDirichlet(form->tm(2),initTime,-m2_init);
    //         bc->addDirichlet(form->te(), initTime,-E_init);
    //       }
    //       break;
    //     case 3:
    //       break;
    //   }
    // }
};

}
