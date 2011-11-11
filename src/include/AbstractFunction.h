#ifndef ABSTRACT_FUNCTION
#define ABSTRACT_FUNCTION

// Teuchos includes
#include "Teuchos_RCP.hpp"
#include "Intrepid_FieldContainer.hpp"

using namespace Intrepid;
using namespace std;

// should compute (possibly vector valued) function values 
class AbstractFunction {
 private:
  //  int _numComponents;
 public:
  AbstractFunction(); 
  virtual void getValues(FieldContainer<double> &functionValues,
			     FieldContainer<double> &physicalPoints);
  virtual void getDerivatives(FieldContainer<double> &functionDerivatives,
			     FieldContainer<double> &physicalPoints);
};

#endif
