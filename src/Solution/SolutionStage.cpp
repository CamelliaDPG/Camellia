//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  SolutionStage.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 3/25/20.
//

#include "SolutionStage.h"

#include "ParameterFunction.h"

using namespace Camellia;

SolutionStage::SolutionStage(ERKTableauPtr tableau, double initialTimeStep)
:
_solutionStage(1),
_tableau(tableau)
{
  _t = Teuchos::rcp(new ParameterFunction(0));
  _dt = Teuchos::rcp(new ParameterFunction(initialTimeStep));
}

void SolutionStage::addObserver(SolutionStageObserverPtr solnStageObserver)
{
  _solnStageObservers.push_back(solnStageObserver);
}

ParameterFunctionPtr SolutionStage::dt() const
{
  return _dt;
}

SolutionStagePtr SolutionStage::solutionStage(ERKTableauPtr tableau, double initialTime)
{
  return Teuchos::rcp(new SolutionStage(tableau,initialTime));
}

int SolutionStage::getStage() const
{
  return _solutionStage;
}

void SolutionStage::setStage(int solutionStage)
{
  if (_solutionStage != solutionStage) // only notify observers if solution stage *changes*
  {
    _solutionStage = solutionStage;
    for (auto observer : _solnStageObservers)
    {
      observer->setStage(solutionStage);
    }
  }
}

ERKTableauPtr SolutionStage::tableau() const
{
  return _tableau;
}

FunctionPtr SolutionStage::t() const
{
  return _t;
}

double SolutionStage::getTime() const
{
  return _t->getConstantValue();
}

void SolutionStage::setTime(double t)
{
  _t->setValue(t);
}

double SolutionStage::getTimeStep() const
{
  return _dt->getConstantValue();
}

void SolutionStage::setTimeStep(double dt)
{
  _dt->setValue(dt);
}
