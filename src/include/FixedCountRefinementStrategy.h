// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  FixedCountRefinementStrategy.h
//  Camellia
//
//  Created by Nathan Roberts on 2/7/22.
//

#ifndef Camellia_FixedCountRefinementStrategy_h
#define Camellia_FixedCountRefinementStrategy_h

#include "Teuchos_RCP.hpp"

#include "ErrorIndicator.h"
#include "IP.h"
#include "LinearTerm.h"
#include "RefinementStrategy.h"
#include "RieszRep.h"
#include "TypeDefs.h"

using namespace std;

namespace Camellia
{


class FixedCountRefinementStrategy : public TRefinementStrategy<double>
{
protected:
  int _targetRefinementCount;
public:
  FixedCountRefinementStrategy( ErrorIndicatorPtr errorIndicator, int numElementsToRefine, double min_h = 0, int max_p = 10, bool preferPRefinements = false);

  void setTargetRefinementCount(const int targetRefinementCount);
  
  virtual void refineAndUnrefine(bool unrefine, bool printToConsole) override;
};

}


#endif
