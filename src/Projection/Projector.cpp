#include "Projector.h"
#include <stdlib.h> //for vectors

#include "Shards_CellTopology.hpp"
#include "Intrepid_FieldContainer.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

typedef Teuchos::RCP< ElementType > ElementTypePtr;
typedef Teuchos::RCP< Element > ElementPtr;
typedef Teuchos::RCP< shards::CellTopology > CellTopoPtr;


void Projector::projectFunctionOntoBasis(FieldContainer<double> &basisCoefficients, Teuchos::RCP<AbstractFunction> fxn, Teuchos::RCP< Basis<double,FieldContainer<double> > > basis){

}

