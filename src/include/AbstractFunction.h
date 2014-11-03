#ifndef ABSTRACT_FUNCTION
#define ABSTRACT_FUNCTION

// Teuchos includes
#include "Teuchos_RCP.hpp"
#include "Intrepid_FieldContainer.hpp"

// should compute (possibly vector valued) function values 
class AbstractFunction {
 private:
  //  int _numComponents;
 public:
  virtual void getValues(Intrepid::FieldContainer<double> &functionValues, const Intrepid::FieldContainer<double> &physicalPoints) = 0;
  virtual ~AbstractFunction() {}
};

#endif
