//
//  IntrepidBasisConstruction.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/10/13.
//
//

#ifndef Camellia_debug_doubleBasisConstruction_h
#define Camellia_debug_doubleBasisConstruction_h

#include "Basis.h"

namespace Camellia {
  BasisPtr lobattoQuadHGRAD(int polyOrder, bool conforming=false);
  BasisPtr intrepidQuadHGRAD(int polyOrder);
  BasisPtr intrepidLineHGRAD(int polyOrder);
}

#endif
