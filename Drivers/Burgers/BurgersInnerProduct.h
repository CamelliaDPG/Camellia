#ifndef DPG_BURGERS_INNER_PRODUCT
#define DPG_BURGERS_INNER_PRODUCT

#include "BurgersBilinearForm.h"
#include "Mesh.h"
#include "BasisCache.h" // for Jacobian/cell measure computation

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "DPGInnerProduct.h"

/*
 Implements Burgers inner product for L2 stability in u
 */

class BurgersInnerProduct : public DPGInnerProduct {
private:
  Teuchos::RCP<BurgersBilinearForm> _burgersBilinearForm;
  Teuchos::RCP<Mesh> _mesh;
public:
  typedef Teuchos::RCP< ElementType > ElementTypePtr;
  typedef Teuchos::RCP< Element > ElementPtr;
  
  BurgersInnerProduct(Teuchos::RCP< BurgersBilinearForm > bfs, Teuchos::RCP<Mesh> mesh);
  
  void operators(int testID1, int testID2, 
                 vector<IntrepidExtendedTypes::EOperatorExtended> &testOp1,
                 vector<IntrepidExtendedTypes::EOperatorExtended> &testOp2);
  
  void applyInnerProductData(FieldContainer<double> &testValues1,
                             FieldContainer<double> &testValues2,
                             int testID1, int testID2, int operatorIndex,
                             Teuchos::RCP<BasisCache> basisCache);
  
  // get weight that biases the outflow over the inflow (for math stability purposes)
  double getWeight(double x,double y);
};

#endif
