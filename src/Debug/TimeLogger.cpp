//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  TimeLogger.cpp
//  Camellia
//
//  Created by Nate Roberts on 6/9/16.
//
//

#include "TimeLogger.h"

#include "Teuchos_RCP.hpp"

using namespace Camellia;

Teuchos::RCP<TimeLogger> TimeLogger::_sharedInstance;

Teuchos::RCP<TimeLogger> TimeLogger::sharedInstance()
{
  if (_sharedInstance == Teuchos::null)
  {
    _sharedInstance = Teuchos::rcp( new TimeLogger() );
  }
  return _sharedInstance;
}

//! Sets all time totals to zero
void TimeLogger::clearTimes()
{
  for (auto entry : _totalTimes)
  {
    _totalTimes[entry.first] = 0.0;
  }
}

void TimeLogger::createTimeEntry(const std::string &timerName)
{
  if (_totalTimes.find(timerName) == _totalTimes.end())
  {
    _totalTimes[timerName] = 0;
  }
}

int TimeLogger::startTimer(const std::string &timerName)
{
  int handle;
  const auto startTime = std::chrono::steady_clock::now();
  if (_inactiveTimerHandles.size() > 0)
  {
    handle = _inactiveTimerHandles[_inactiveTimerHandles.size()-1];
    _inactiveTimerHandles.pop_back();
    
    _timers[handle] = startTime;
    _timerNames[handle] = timerName;
  }
  else
  {
    handle = _timers.size();
    _timers.push_back(startTime);
    _timerNames.push_back(timerName);
  }
  return handle;
}

void TimeLogger::stopTimer(int timerHandle)
{
  TEUCHOS_TEST_FOR_EXCEPTION((timerHandle < 0) || (timerHandle >= _timers.size()), std::invalid_argument, "timerHandle out of bounds");
  
  // check that this is not an inactive timer:
  auto foundEntry = std::find(_inactiveTimerHandles.begin(), _inactiveTimerHandles.end(), timerHandle);
  TEUCHOS_TEST_FOR_EXCEPTION(foundEntry != _inactiveTimerHandles.end(), std::invalid_argument, "stopTimer() called with timerHandle for inactive timer.");
  
  std::string timerName = _timerNames[timerHandle];
  auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - _timers[timerHandle]).count();
  _totalTimes[timerName] += elapsed;
  _inactiveTimerHandles.push_back(timerHandle);
  std::sort(_inactiveTimerHandles.begin(), _inactiveTimerHandles.end());
}

double TimeLogger::totalTime(const std::string &timerName) const
{
  auto foundEntry = _totalTimes.find(timerName);
  if (foundEntry == _totalTimes.end()) return 0;
  else return foundEntry->second;
}

const std::map<std::string,double> TimeLogger::totalTimes() const
{
  return _totalTimes;
}
