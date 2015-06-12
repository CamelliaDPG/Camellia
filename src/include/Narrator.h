//
//  Narrator.h
//  Camellia
//
//  Created by Nate Roberts on 6/12/15.
//
//

#ifndef Camellia_Narrator_h
#define Camellia_Narrator_h

#include "MPIWrapper.h"

namespace Camellia {
  class Narrator
  {
  private:
    bool _narrateOnThisRank; // if true, print various events to console
    std::string _nameForNarration; // will be printed on each line of narration
    std::string _defaultNameForNarration; // will be printed on each line of narration
    
  protected:
    void narrate(std::string event)
    {
      if (_narrateOnThisRank)
      {
        int rank = MPIWrapper::rank();
        cout << _nameForNarration << " (" << rank << "): " << event << endl;
      }
    }
    
  public:
    Narrator(std::string defaultNameForNarration)
    {
      _defaultNameForNarration = defaultNameForNarration;
      _narrateOnThisRank = false;
    }
    
    void setNarrateOnRankZero(bool value)
    {
      if (MPIWrapper::rank() == 0)
      {
        _narrateOnThisRank = value;
        _nameForNarration = _defaultNameForNarration;
      }
    }
    
    void setNarrateOnThisRank(bool value)
    {
      _narrateOnThisRank = value;
      _nameForNarration = _defaultNameForNarration;
    }
    
    void setNarrateOnRankZero(bool value, std::string nameForNarration)
    {
      if (MPIWrapper::rank() == 0)
      {
        _narrateOnThisRank = value;
        _nameForNarration = nameForNarration;
      }
    }
    
    void setNarrateOnThisRank(bool value, std::string nameForNarration)
    {
      _narrateOnThisRank = value;
      _nameForNarration = nameForNarration;
    }
  };
}


#endif
