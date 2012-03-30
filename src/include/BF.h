//
//  BF.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_BF_h
#define Camellia_BF_h

#include "BilinearForm.h"
#include "LinearTerm.h"

#include "VarFactory.h"

class BF : public BilinearForm {
  typedef pair< LinearTermPtr, LinearTermPtr > BilinearTerm;
  vector< BilinearTerm > _terms;
  VarFactory _varFactory;
public:
  BF( VarFactory varFactory ); // copies (note that external changes in VarFactory won't be registered by BF)

  void addTerm( LinearTermPtr trialTerm, LinearTermPtr testTerm );
  void addTerm( VarPtr trialVar, LinearTermPtr testTerm );
  void addTerm( VarPtr trialVar, VarPtr testVar );
  void addTerm( LinearTermPtr trialTerm, VarPtr testVar);
  
  // BilinearForm implementation:
  const string & testName(int testID);
  const string & trialName(int trialID);
  
  EFunctionSpaceExtended functionSpaceForTest(int testID);
  EFunctionSpaceExtended functionSpaceForTrial(int trialID);
  
  bool isFluxOrTrace(int trialID);
  
  void printTrialTestInteractions();
  
  void stiffnessMatrix(FieldContainer<double> &stiffness, Teuchos::RCP<ElementType> elemType,
                       FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache);
};

#endif
