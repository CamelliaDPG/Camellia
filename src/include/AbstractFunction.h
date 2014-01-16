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
  virtual void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints) = 0;
  virtual ~AbstractFunction() {}
};

#endif
