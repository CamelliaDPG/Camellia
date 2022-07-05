//
//  ConfigurationSpaceFormulation.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 1/13/22.
//

#include "ConfigurationSpaceFormulation.h"
#include "ConfigurationSpaceFormulationPoisson.h"

using namespace Camellia;

ConfigurationSpaceFormulation::ConfigurationSpaceFormulation(bool useSerendipityBasis)
:
_useSerendipityBasis(useSerendipityBasis)
{}

void ConfigurationSpaceFormulation::copySolutionToPrevious()
{
  _configurationSolnPrev->setSolution(_configurationSoln);
}

void ConfigurationSpaceFormulation::copySolutionToPriorTime()
{
  _configurationSolnPriorTime->setSolution(_configurationSoln);
}

VarPtr ConfigurationSpaceFormulation::B_fieldVar(int zeroBasedComp)
{
  return Teuchos::null;
}

VarPtr ConfigurationSpaceFormulation::B_hat(int zeroBasedComp)
{
  return Teuchos::null;
}

MeshPtr ConfigurationSpaceFormulation::configurationSpaceMesh()
{
  return _configurationSoln->mesh();
}

MeshTopologyPtr ConfigurationSpaceFormulation::configurationSpaceMeshTopology()
{
  return _configurationMeshTopo;
}

VarPtr ConfigurationSpaceFormulation::E_fieldVar(int zeroBasedComp)
{
  return Teuchos::null;
}

VarPtr ConfigurationSpaceFormulation::E_hat(int zeroBasedComp)
{
  return Teuchos::null;
}

VarPtr ConfigurationSpaceFormulation::En_hat()
{
  return Teuchos::null;
}

double ConfigurationSpaceFormulation::l2NormOfSolutionUpdate()
{
  return 0; // default implementation
}

Teuchos::RCP<ConfigurationSpaceFormulation> ConfigurationSpaceFormulation::poissonFormulation(const double &q, const double &eps0, const TimeDiscretizationType &timeDiscretizationType)
{
  return Teuchos::rcp(new ConfigurationSpaceFormulationPoisson(q,eps0,timeDiscretizationType));
}

void ConfigurationSpaceFormulation::setConfigurationSolution(SolutionPtr configurationSolution)
{
  if (_configurationSoln != Teuchos::null)
  {
    _configurationSoln->mesh()->unregisterSolution(_configurationSoln);
  }
  _configurationSoln = configurationSolution;
  if (_configurationSoln != Teuchos::null)
  {
    _configurationSoln->mesh()->registerSolution(_configurationSoln);
  }
}

void ConfigurationSpaceFormulation::setConfigurationSolutionPrevious(SolutionPtr configurationSolutionPrev)
{
  if (_configurationSolnPrev != Teuchos::null)
  {
    _configurationSolnPrev->mesh()->unregisterSolution(_configurationSolnPrev);
  }
  _configurationSolnPrev = configurationSolutionPrev;
  if (_configurationSolnPrev != Teuchos::null)
  {
    _configurationSolnPrev->mesh()->registerSolution(_configurationSolnPrev);
  }
}

void ConfigurationSpaceFormulation::setConfigurationSolutionPriorTime(SolutionPtr configurationSolutionPriorTime)
{
  if (_configurationSolnPriorTime != Teuchos::null)
  {
    _configurationSolnPriorTime->mesh()->unregisterSolution(_configurationSolnPriorTime);
  }
  _configurationSolnPriorTime = configurationSolutionPriorTime;
  if (_configurationSolnPriorTime != Teuchos::null)
  {
    _configurationSolnPriorTime->mesh()->registerSolution(_configurationSolnPriorTime);
  }
}

//! should be called prior to first solve: set the solution moments object, and optionally a background density n0 (used when quasi-neutrality is enforced).
void ConfigurationSpaceFormulation::setMoments(Teuchos::RCP<SolutionMoments> fMoments, FunctionPtr n0)
{
  _fMoments = fMoments;
}

//! When updateMoments is true, will call fMoments->update() prior to solve.
void ConfigurationSpaceFormulation::solve(const bool updateMoments)
{
  if (_configurationSoln != Teuchos::null)
  {
    if (updateMoments)
    {
      _fMoments->update();
    }
    _configurationSoln->solve(); // TODO: add support for GMG, etc. (this is just a direct solve: OK for 1D + time, but expensive beyond that)
  }
}

SolutionPtr ConfigurationSpaceFormulation::solution()
{
  return _configurationSoln;
}

bool ConfigurationSpaceFormulation::useSerendipityBasis() const
{
  return _useSerendipityBasis;
}

VarPtr ConfigurationSpaceFormulation::VE_fieldVar()
{
  return Teuchos::null;
}

VarPtr ConfigurationSpaceFormulation::VE_hat()
{
  return Teuchos::null;
}
