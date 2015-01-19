//
//  Formulation.h
//  Camellia
//
//  Created by Nathan Roberts on 4/2/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Formulation_h
#define Camellia_Formulation_h

#include "Teuchos_RCP.hpp"
#include "BilinearForm.h"
#include "BC.h"
#include "RHS.h"
#include "IP.h"

class Formulation {
protected:
  BilinearFormPtr _bilinearForm;
  BCPtr _bc;
  RHSPtr _rhs;
  InnerProductPtr _ip;
  FieldContainer<double> _quadDomain; // leave empty/unspecified for non-quad domains (not ideal)
public:
  BilinearFormPtr bilinearForm();
  BCPtr bc();
  RHSPtr rhs();
  InnerProductPtr innerProduct();
  FieldContainer<double> quadDomain();
};

#endif
