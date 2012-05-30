#ifndef DPG_EXPERIMENTAL_INNER_PRODUCT
#define DPG_EXPERIMENTAL_INNER_PRODUCT

#include "ConfusionBilinearForm.h"
#include "Mesh.h"
#include "BasisCache.h" // for Jacobian/cell measure computation

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "DPGInnerProduct.h"

/*
  Implements experimental Confusion inner product 
*/

class ExperimentalInnerProduct : public DPGInnerProduct {
 private:
  Teuchos::RCP<ConfusionBilinearForm> _confusionBilinearForm;
  Teuchos::RCP<Mesh> _mesh;
 public:
  typedef Teuchos::RCP< ElementType > ElementTypePtr;
  typedef Teuchos::RCP< Element > ElementPtr;  

  ExperimentalInnerProduct(Teuchos::RCP< ConfusionBilinearForm > bfs, Teuchos::RCP<Mesh> mesh);
  
  void operators(int testID1, int testID2, 
                 vector<IntrepidExtendedTypes::EOperatorExtended> &testOp1,
                 vector<IntrepidExtendedTypes::EOperatorExtended> &testOp2);
  
  void applyInnerProductData(FieldContainer<double> &testValues1,
                             FieldContainer<double> &testValues2,
                             int testID1, int testID2, int operatorIndex,
                             const FieldContainer<double>& physicalPoints);
			     
  
  // get weight that biases the outflow over the inflow (for math stability purposes)
  double getWeight(double x,double y);

};

#endif
