//
//  ConfigurationSpaceFormulationTrivial.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 1/13/22.
//

#include "ConfigurationSpaceFormulationTrivial.h"

using namespace Camellia;

ConfigurationSpaceFormulationTrivial::ConfigurationSpaceFormulationTrivial(TimeDiscretizationType timeDiscretizationType)
:
_timeDiscretizationType(timeDiscretizationType),
ConfigurationSpaceFormulation()
{}

void ConfigurationSpaceFormulationTrivial::initializeMesh(MeshTopologyPtr meshTopo, const int H1Order, const int delta_k)
{
  this->_configurationMeshTopo = meshTopo;
  
  const bool spaceTime = (_timeDiscretizationType == SPACE_TIME) || (_timeDiscretizationType == EXPONENTIAL_INTEGRATOR);
  _spaceDim = spaceTime ? meshTopo->getDimension() - 1 : meshTopo->getDimension();
  
  TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim < 1, std::invalid_argument, "Invalid spaceDim");
  TEUCHOS_TEST_FOR_EXCEPTION(_spaceDim > 3, std::invalid_argument, "Invalid spaceDim");
  
  VarFactoryPtr trivialVF = VarFactory::varFactory();
  BasisFamilyPtr basisFamily = BasisFamily::basisFamily(_basisFamily);
  trivialVF->setBasisFamily(basisFamily);
  
  auto bf = BF::bf(trivialVF);
  
  Teuchos_CommPtr comm = Teuchos::null;
  const bool allowPruning = false;
  map<int,int> trialOrderEnhancements;
  map<int,int> testOrderEnhancements;
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::null; // use default partition policy
  MeshPtr trivialMesh = Teuchos::rcp( new Mesh(meshTopo, bf, H1Order, delta_k,
                                               trialOrderEnhancements, testOrderEnhancements, partitionPolicy,
                                               comm, allowPruning) );
  
  auto trivialSoln     = Solution::solution(bf, trivialMesh);
  auto trivialSolnPrev = Solution::solution(bf, trivialMesh); // will copy out after fixed point solve
  
  trivialSoln->setIP(IP::ip());
  
  // nullify _configurationSoln, _configurationSolnPrev: if these are initialized, they refer to a previous mesh, not the poissonMesh we just created
  _configurationSoln     = Teuchos::null;
  _configurationSolnPrev = Teuchos::null;
  setConfigurationSolution(trivialSoln);
  setConfigurationSolutionPrevious(trivialSolnPrev);
  
  if (spaceTime)
  {
    setConfigurationSolutionPriorTime(Teuchos::null);
  }
  else
  {
    auto trivialSolnPriorTime = Solution::solution(bf, trivialMesh);
    setConfigurationSolutionPriorTime(trivialSolnPriorTime);
  }
}

std::string ConfigurationSpaceFormulationTrivial::name() const
{
  return "Trivial";
}

Teuchos::RCP<ConfigurationSpaceFormulation> ConfigurationSpaceFormulationTrivial::newInstance() const
{
  auto instance = Teuchos::rcp(new ConfigurationSpaceFormulationTrivial(_timeDiscretizationType));
  instance->setE(_E);
  instance->setB(_B);
  instance->setE_priorTime(_E_priorTime);
  instance->setB_priorTime(_B_priorTime);
  instance->setExtraAcceleration(_extraAcceleration);
  instance->setExportViz(_exportViz, _vizOutputDir);
  return instance;
}

FunctionPtr ConfigurationSpaceFormulationTrivial::B_i_priorTime(int zeroBasedComp)
{
  if (zeroBasedComp >= _B_priorTime.size())
  {
    return Function::zero();
  }
  else
  {
    return _B_priorTime[zeroBasedComp];
  }
}

FunctionPtr ConfigurationSpaceFormulationTrivial::B_i(int zeroBasedComp)
{
  if (zeroBasedComp >= _B.size())
  {
    return Function::zero();
  }
  else
  {
    return _B[zeroBasedComp];
  }
}

FunctionPtr ConfigurationSpaceFormulationTrivial::E_i(int zeroBasedComp)
{
  if (zeroBasedComp >= _E.size())
  {
    return Function::zero();
  }
  else
  {
    return _E[zeroBasedComp];
  }
}

FunctionPtr ConfigurationSpaceFormulationTrivial::E_i_priorTime(int zeroBasedComp)
{
  if (zeroBasedComp >= _E_priorTime.size())
  {
    return Function::zero();
  }
  else
  {
    return _E_priorTime[zeroBasedComp];
  }
}

//! maximum moment number (starting from 0 for the unweighted integral over velocity space) required by the ConfigurationSpaceFormulation.
int ConfigurationSpaceFormulationTrivial::maxMomentNumber()
{
  return 0;
}

void ConfigurationSpaceFormulationTrivial::setB(std::vector<FunctionPtr> B)
{
  _B = B;
}

void ConfigurationSpaceFormulationTrivial::setB_priorTime(std::vector<FunctionPtr> B)
{
  _B_priorTime = B;
}

void ConfigurationSpaceFormulationTrivial::setE(std::vector<FunctionPtr> E)
{
  _E = E;
}

void ConfigurationSpaceFormulationTrivial::setE_priorTime(std::vector<FunctionPtr> E)
{
  _E_priorTime = E;
}

void ConfigurationSpaceFormulationTrivial::updateConfigurationSpaceFunctions()
{
  
}

