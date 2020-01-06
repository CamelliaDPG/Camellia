// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  Formulation.h
//  Camellia
//
//  Created by Nathan Roberts on 4/2/12.
//

#ifndef Camellia_Formulation_h
#define Camellia_Formulation_h

#include "TypeDefs.h"

#include "Teuchos_RCP.hpp"
#include "BF.h"
#include "BC.h"
#include "RHS.h"
#include "IP.h"

namespace Camellia
{
class Formulation
{
protected:
  BFPtr _bilinearForm;
  BCPtr _bc;
  RHSPtr _rhs;
  InnerProductPtr _ip;
  Camellia::FieldContainer<double> _quadDomain; // leave empty/unspecified for non-quad domains (not ideal)
public:
  BFPtr bilinearForm();
  BCPtr bc();
  RHSPtr rhs();
  InnerProductPtr innerProduct();
  Camellia::FieldContainer<double> quadDomain();
};
}

#endif
