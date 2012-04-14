//
//  ConvergenceReport.h
//  Camellia
//
//  Created by Nathan Roberts on 4/13/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_ConvergenceReport_h
#define Camellia_ConvergenceReport_h

#include "HConvergenceStudy.h"

class ConvergenceReport {
public:
  void writeReportToTeXFile(HConvergenceStudy &study, string fileName);
};

#endif
