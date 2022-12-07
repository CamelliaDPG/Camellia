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

void ConfigurationSpaceFormulation::loadFromCheckpoint(const std::string &checkpointDir)
{
  // for now, we require that the mesh agrees (i.e., refinements are replayed externally -- they aren't loaded from disk here; we haven't added ExtrudedMeshTopology to the Solution::load() infrastructure)
  // TODO: add some sort of sanity check on the mesh vs. loaded solution (cell numbers are present in mesh, say)
  
  std::string configName = this->name();
  std::string solnPath = checkpointDir + "/" + configName + ".soln";
  _configurationSoln->loadFromHDF5(solnPath);
  
  if (_configurationSolnPrev != Teuchos::null)
  {
    std::string solnPrevPath = checkpointDir + "/" + configName + "Prev.soln";
    _configurationSolnPrev->loadFromHDF5(solnPrevPath);
  }
  
  if (_configurationSolnPriorTime != Teuchos::null)
  {
    std::string solnPriorTimePath = checkpointDir + "/" + configName + "PriorTime.soln";
    _configurationSolnPriorTime->loadFromHDF5(solnPriorTimePath);
  }
}

Teuchos::RCP<ConfigurationSpaceFormulation> ConfigurationSpaceFormulation::poissonFormulation(const double &q, const double &eps0, const TimeDiscretizationType &timeDiscretizationType)
{
  return Teuchos::rcp(new ConfigurationSpaceFormulationPoisson(q,eps0,timeDiscretizationType));
}

void ConfigurationSpaceFormulation::saveCheckpoint(const std::string &checkpointDir)
{
  std::string configName = this->name();
  std::string solnPrefix = checkpointDir + "/" + configName;
  _configurationSoln->save(solnPrefix);
  
  if (_configurationSolnPrev != Teuchos::null)
  {
    std::string solnPrevPrefix = checkpointDir + "/" + configName + "Prev";
    _configurationSolnPrev->save(solnPrevPrefix);
  }
  
  if (_configurationSolnPriorTime != Teuchos::null)
  {
    std::string solnPriorTimePrefix = checkpointDir + "/" + configName + "PriorTime";
    _configurationSolnPriorTime->save(solnPriorTimePrefix);
  }
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

void ConfigurationSpaceFormulation::setExportViz(bool exportViz, const std::string outputDir)
{
  _exportViz = exportViz;
  _vizOutputDir = outputDir;
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
    
    if (_exportViz)
    {
      static int solveNumber = 0;
      std::string prefix = "ConfigSpace_";
      
      std::ostringstream configurationSS;
      configurationSS << prefix << this->name() << solveNumber;
              
      Teuchos::RCP<HDF5Exporter> configurationViz = Teuchos::rcp(new HDF5Exporter (_configurationSoln->mesh(), configurationSS.str(), _vizOutputDir) );
      configurationViz->exportSolution(_configurationSoln,0);
      
      solveNumber++;
    }
  }
}

SolutionPtr ConfigurationSpaceFormulation::solution()
{
  return _configurationSoln;
}

SolutionPtr ConfigurationSpaceFormulation::solutionPriorTime()
{
  return _configurationSolnPriorTime;
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
