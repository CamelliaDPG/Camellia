//
//  Formulation.h
//  Camellia
//
//  Created by Nathan Roberts on 4/2/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Formulation_h
#define Camellia_Formulation_h

#include "TypeDefs.h"

#include "Teuchos_RCP.hpp"
#include "BF.h"
#include "BC.h"
#include "RHS.h"
#include "IP.h"

class Formulation {
protected:
  BFPtr _bilinearForm;
  BCPtr _bc;
  RHSPtr _rhs;
  InnerProductPtr _ip;
  Intrepid::FieldContainer<double> _quadDomain; // leave empty/unspecified for non-quad domains (not ideal)
public:
  BFPtr bilinearForm();
  BCPtr bc();
  RHSPtr rhs();
  InnerProductPtr innerProduct();
  Intrepid::FieldContainer<double> quadDomain();
};

#endif
