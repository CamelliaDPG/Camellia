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

#include "IP.h"

class BF : public BilinearForm {
  typedef pair< LinearTermPtr, LinearTermPtr > BilinearTerm;
  vector< BilinearTerm > _terms;
  VarFactory _varFactory;
public:  
  BF( VarFactory varFactory ); // copies (note that external changes in VarFactory won't be registered by BF)
  BF( VarFactory varFactory, VarFactory::BubnovChoice choice);

  void addTerm( LinearTermPtr trialTerm, LinearTermPtr testTerm );
  void addTerm( VarPtr trialVar, LinearTermPtr testTerm );
  void addTerm( VarPtr trialVar, VarPtr testVar );
  void addTerm( LinearTermPtr trialTerm, VarPtr testVar);
  
  // BilinearForm implementation:
  const string & testName(int testID);
  const string & trialName(int trialID);
  
  IntrepidExtendedTypes::EFunctionSpaceExtended functionSpaceForTest(int testID);
  IntrepidExtendedTypes::EFunctionSpaceExtended functionSpaceForTrial(int trialID);
  
  IPPtr graphNorm();
  
  bool isFluxOrTrace(int trialID);
  
  void printTrialTestInteractions();
  
  void stiffnessMatrix(FieldContainer<double> &stiffness, Teuchos::RCP<ElementType> elemType,
                       FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache);
  
  LinearTermPtr testFunctional(SolutionPtr trialSolution);
};

typedef Teuchos::RCP<BF> BFPtr;

#endif
