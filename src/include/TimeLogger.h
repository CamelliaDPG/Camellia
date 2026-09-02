// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  TimeLogger.h
//  Camellia
//
//  Created by Nate Roberts on 6/9/16.
//
//

#ifndef Camellia_TimeLogger_h
#define Camellia_TimeLogger_h

#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "Teuchos_RCP.hpp"

namespace Camellia {
  class TimeLogger
  {
    static Teuchos::RCP<TimeLogger> _sharedInstance;
  
    std::vector<std::string> _timerNames;
    std::vector<std::chrono::steady_clock::time_point> _timers;
    std::vector<int> _inactiveTimerHandles;
    std::map<std::string,double> _totalTimes;
  public:
    int startTimer(const std::string &timerName);
    void stopTimer(int timerHandle);
    
    //! Allows creation of time entries even if a timer doesn't actually get started/stopped on all ranks
    void createTimeEntry(const std::string &timerName);
    
    double totalTime(const std::string &timerName) const;
    const std::map<std::string,double> totalTimes() const;
    
    //! Sets all time totals to zero
    void clearTimes();
    
    static Teuchos::RCP<TimeLogger> sharedInstance();
  };
}

#endif
