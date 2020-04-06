//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//
//  ERKTableau.h
//  Camellia
//
//  Created by Roberts, Nathan V on 3/25/20.
//

#ifndef ERKTableau_h
#define ERKTableau_h

#include "TypeDefs.h"

#include "Camellia_FieldContainer.hpp"

namespace Camellia
{
  class ERKTableau
  {
    int _s; // num stages
    std::vector<double> _a; // lower-triangular, numStages*(numStages-1)/2 entries
    std::vector<double> _b; // numStages entries
    std::vector<double> _c; // numStages entries; first entry is 0
    
    double _expectedRate;
  public:
    ERKTableau(const int numStages, const FieldContainer<double> &a, const FieldContainer<double> &b, const FieldContainer<double> &c, const double expectedRate);
    double a(int i, int j) const;
    double b(int i) const;
    double c(int i) const;
    
    //! returns 1 for linear, 2 for quadratic, etc.
    double expectedRate() const;
    
    int numStages() const;
    
    static ERKTableauPtr forwardEuler();
    static ERKTableauPtr heun();
    static ERKTableauPtr rk4();
    static ERKTableauPtr ssprk3();
  };
}

#endif /* ERKTableau_h */
