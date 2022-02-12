// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  DoerflerRefinementStrategy.h
//  Camellia
//
//  Created by Nathan Roberts on 2/7/22.
//

#ifndef Camellia_DoerflerRefinementStrategy_h
#define Camellia_DoerflerRefinementStrategy_h

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


class DoerflerRefinementStrategy : public TRefinementStrategy<double>
{
protected:
  double _doerflerTarget;
public:
  DoerflerRefinementStrategy( ErrorIndicatorPtr errorIndicator, double doerflerTarget, double min_h = 0, int max_p = 10, bool preferPRefinements = false);

  virtual void refineAndUnrefine(bool unrefine, bool printToConsole) override;
};

}


#endif
