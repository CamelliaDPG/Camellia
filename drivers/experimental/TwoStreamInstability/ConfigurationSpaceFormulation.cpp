//
//  ConfigurationSpaceFormulation.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 1/13/22.
//

#include "ConfigurationSpaceFormulation.h"

using namespace Camellia;

ConfigurationSpaceFormulation::ConfigurationSpaceFormulation()
{}

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

VarPtr ConfigurationSpaceFormulation::E_fieldVar(int zeroBasedComp)
{
  return Teuchos::null;
}

VarPtr ConfigurationSpaceFormulation::E_hat(int zeroBasedComp)
{
  return Teuchos::null;
}

void ConfigurationSpaceFormulation::setConfigurationSolution(SolutionPtr configurationSolution)
{
  _configurationSoln = configurationSolution;
}

void ConfigurationSpaceFormulation::setConfigurationSolutionPrevious(SolutionPtr configurationSolutionPrev)
{
  _configurationSolnPrev = configurationSolutionPrev;
}

//! should be called prior to first solve: set the solution moments object.
void ConfigurationSpaceFormulation::setMoments(Teuchos::RCP<SolutionMoments> fMoments)
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

VarPtr ConfigurationSpaceFormulation::VE_fieldVar()
{
  return Teuchos::null;
}

VarPtr ConfigurationSpaceFormulation::VE_hat()
{
  return Teuchos::null;
}
