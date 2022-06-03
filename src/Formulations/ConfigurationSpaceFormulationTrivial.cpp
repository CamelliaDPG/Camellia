//
//  ConfigurationSpaceFormulationTrivial.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 1/13/22.
//

#include "ConfigurationSpaceFormulationTrivial.h"

using namespace Camellia;

ConfigurationSpaceFormulationTrivial::ConfigurationSpaceFormulationTrivial()
:
ConfigurationSpaceFormulation()
{}

void ConfigurationSpaceFormulationTrivial::initializeMesh(MeshTopologyPtr meshTopo, const int H1Order, const int delta_k)
{
  this->_configurationMeshTopo = meshTopo;
}

std::string ConfigurationSpaceFormulationTrivial::name() const
{
  return "Trivial";
}

FunctionPtr ConfigurationSpaceFormulationTrivial::B_i(int zeroBasedComp)
{
  return Function::zero();
}

FunctionPtr ConfigurationSpaceFormulationTrivial::E_i(int zeroBasedComp)
{
  return Function::zero();
}

//! maximum moment number (starting from 0 for the unweighted integral over velocity space) required by the ConfigurationSpaceFormulation.
int ConfigurationSpaceFormulationTrivial::maxMomentNumber()
{
  return 0;
}

void ConfigurationSpaceFormulationTrivial::updateConfigurationSpaceFunctions()
{
  
}

