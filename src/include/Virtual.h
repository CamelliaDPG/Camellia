//
//  Virtual.h
//  Camellia-debug
//
//  Created by Nate Roberts on 7/28/14.
//
//

#ifndef __Camellia_debug__Virtual__
#define __Camellia_debug__Virtual__

#include "LinearTerm.h"
#include "VarFactory.h"

#include "TypeDefs.h"

namespace Camellia {
  class Virtual {
    vector<LinearTermPtr> _fieldOperators;
    vector<VarPtr> _traceVars;
    vector<VarPtr> _fieldTests; // test variables corresponding to field and trace operators
    vector<LinearTermPtr> _testNormOperators, _testBoundaryOperators;
    
    map< int, pair<VarPtr, VarPtr> > _testAssociations; // test ID --> (fieldVar, traceVar)
    
    int _testEnrichment; // difference between the order of the "interior" test functions, and that of the field trial functions
  public:
    Virtual(int testEnrichment = 0);
    void addAssociation(VarPtr testVar, VarPtr fieldVar, VarPtr traceVar);
    void addEquation(LinearTermPtr strongFieldOperator, VarPtr traceVar, VarPtr testVar);
    void addTestNormTerm(LinearTermPtr testNormOperator, LinearTermPtr testBoundaryOperator);
    
    VarPtr getAssociatedField(VarPtr testVar);
    VarPtr getAssociatedTrace(VarPtr testVar);
    
    const vector<LinearTermPtr> &getFieldOperators();
    const vector<VarPtr> &getTraceVars();
    const vector<VarPtr> &getFieldTestVars();
    
    const vector<LinearTermPtr> &getTestNormOperators();
    const vector<LinearTermPtr> &getTestNormBoundaryOperators();
    
    int getTestEnrichment();
    
    static BFPtr virtualBF(Virtual &virtualTerms, VarFactoryPtr vf);
  };
}
#endif /* defined(__Camellia_debug__Virtual__) */
