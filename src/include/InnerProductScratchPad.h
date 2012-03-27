#ifndef CAMELLIA_INNER_PRODUCT_SCRATCH_PAD
#define CAMELLIA_INNER_PRODUCT_SCRATCH_PAD

// several of the following classes probably should be defined elsewhere
// since they'll be used by several ScratchPads

#include "BilinearForm.h"
#include "DPGInnerProduct.h"
#include "Intrepid_Basis.hpp"

class Var;
class LinearTerm;
class Function;

typedef Teuchos::RCP<Var> VarPtr;
typedef Teuchos::RCP<LinearTerm> LinearTermPtr;
typedef Teuchos::RCP<Function> FunctionPtr;
typedef Teuchos::RCP<BasisCache> BasisCachePtr;
typedef Teuchos::RCP<DPGInnerProduct> InnerProductPtr;
typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;

class BasisCache;
class DPGInnerProduct;

using namespace std;

class Function {
protected:
  int _rank;
public:
  Function(int rank) { _rank = rank; }
  
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) = 0;
  int rank() { return _rank; }
};

class ConstantScalarFunction : public Function {
  double _weight;
public:
  ConstantScalarFunction(double weight) : Function(0) { _weight = weight; }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    for (int i=0; i < values.size(); i++) {
      values[i] = _weight;
    }
  }
};

class ConstantVectorFunction : public Function {
  vector<double> _weight;
public:
  ConstantVectorFunction(vector<double> weight) : Function(1) { _weight = weight; }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    // values are stored in (C,P,D) order, the important thing here being that we can do this:
    for (int i=0; i < values.size(); ) {
      for (int d=0; d < _weight.size(); d++) {
        values[i++] = _weight[d];
      }
    }
  }
};

namespace FunctionSpaces {
  enum Space { HGRAD, HCURL, HDIV, L2, UNKNOWN_FS };
}

using namespace FunctionSpaces;

class Var { // really Var x Operator
  int _rank;
  int _id;
  string _name;
  Space _fs;
  EOperatorExtended _op; // default is OPERATOR_VALUE
//  map< EOperatorExtended, VarPtr > _relatedVars; // grad, div, etc. could be cached here
public:
  Var(int ID, int rank, string name, EOperatorExtended op = IntrepidExtendedTypes::OPERATOR_VALUE,
      Space fs = UNKNOWN_FS) {
    _id = ID;
    _rank = rank;
    _name = name;
    _op = op;
    _fs = fs;
  }
  
  int rank() { return _rank; }  // 0 for scalar, 1 for vector, etc.
    
  int ID() { return _id; }
  EOperatorExtended op() { return _op; }

  Space space() { return _fs; }
  
  VarPtr grad() {
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "grad() only supported for vars of rank 0.");
    return Teuchos::rcp( new Var(_id, _rank + 1, _name, IntrepidExtendedTypes::OPERATOR_GRAD ) );
  }
  VarPtr div() {
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "div() only supported for vars of rank 1.");
    return Teuchos::rcp( new Var(_id, _rank - 1, _name, IntrepidExtendedTypes::OPERATOR_DIV ) );
  }
  VarPtr curl() {
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    TEST_FOR_EXCEPTION( (_rank != 0) && (_rank != 1), std::invalid_argument, "curl() can only be applied to vars of ranks 0 or 1.");
    int newRank = (_rank == 0) ? 1 : 0;
    return Teuchos::rcp( new Var(_id, newRank, _name, IntrepidExtendedTypes::OPERATOR_CURL ) );
  }
  VarPtr dx() {
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dx() only supported for vars of rank 0.");
    return Teuchos::rcp( new Var(_id, _rank, _name, IntrepidExtendedTypes::OPERATOR_DX ) );
  }
  VarPtr dy() {
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dy() only supported for vars of rank 0.");
    return Teuchos::rcp( new Var(_id, _rank, _name, IntrepidExtendedTypes::OPERATOR_DY ) );
  }
  VarPtr dz() {
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    TEST_FOR_EXCEPTION( _rank != 0, std::invalid_argument, "dz() only supported for vars of rank 0.");
    return Teuchos::rcp( new Var(_id, _rank, _name, IntrepidExtendedTypes::OPERATOR_DZ ) );
  }
  VarPtr x() { 
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "x() only supported for vars of rank 1.");
    return Teuchos::rcp( new Var(_id, _rank-1, _name, OPERATOR_X ) );
  }
  VarPtr y() { 
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "y() only supported for vars of rank 1.");
    return Teuchos::rcp( new Var(_id, _rank-1, _name, OPERATOR_Y ) );
  }
  VarPtr z() { 
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "z() only supported for vars of rank 1.");
    return Teuchos::rcp( new Var(_id, _rank-1, _name, OPERATOR_Z ) );
  }
  
  VarPtr cross_normal() {
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "cross_normal() only supported for vars of rank 1.");
    return Teuchos::rcp( new Var(_id, _rank-1, _name, OPERATOR_CROSS_NORMAL ) );
  }
  VarPtr dot_normal() {
    TEST_FOR_EXCEPTION( _rank != 1, std::invalid_argument, "dot_normal() only supported for vars of rank 1.");
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    return Teuchos::rcp( new Var(_id, _rank-1, _name, OPERATOR_DOT_NORMAL ) );
  }
  VarPtr times_normal() {
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    return Teuchos::rcp( new Var(_id, _rank + 1, _name, OPERATOR_TIMES_NORMAL ) );
  }
  VarPtr times_normal_x() {
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    return Teuchos::rcp( new Var(_id, _rank, _name, OPERATOR_TIMES_NORMAL_X ) );
  }
  VarPtr times_normal_y() {
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    return Teuchos::rcp( new Var(_id, _rank, _name, OPERATOR_TIMES_NORMAL_Y ) );
  }
  VarPtr times_normal_z() {
    TEST_FOR_EXCEPTION( _op != IntrepidExtendedTypes::OPERATOR_VALUE, std::invalid_argument, "operators can only be applied to raw vars, not vars that have been operated on.");
    return Teuchos::rcp( new Var(_id, _rank, _name, OPERATOR_TIMES_NORMAL_Z ) );
  }
  
//  // the following will make tests with names v_i, trials with names u_i
//  static VarPtr testVar(int ID, int rank);
//  static VarPtr trialVar(int ID, int rank);
};

class LinearTerm {
  int _rank; // gets set after first var is added
  typedef pair< FunctionPtr, VarPtr > LinearSummand;
  vector< LinearSummand > _summands;
  set<int> _varIDs;
protected:
  const vector< LinearSummand > & summands() const { return _summands; }
public:
  LinearTerm(FunctionPtr weight, VarPtr var) {
    _rank = -1;
    addVar(weight,var);
  }
  LinearTerm(double weight, VarPtr var) {
    _rank = -1;
    addVar(weight,var);
  }
  LinearTerm(vector<double> weight, VarPtr var) {
    _rank = -1;
    addVar(weight,var);
  }
  LinearTerm( VarPtr v ) {
    _rank = -1;
    addVar( 1.0, v);
  }
  
  void addVar(FunctionPtr weight, VarPtr var) {
    int rank; // rank of weight * var
    if (weight->rank() == var->rank() ) { // then we dot like terms together, getting a scalar
      rank = 0;
    } else if ( weight->rank() == 0 || var->rank() == 0) { // then we multiply each term by scalar
      rank = (weight->rank() == 0) ? var->rank() : weight->rank(); // rank is the non-zero one
    } else {
      TEST_FOR_EXCEPTION( true, std::invalid_argument, "Unhandled rank combination.");
    }
    if (_rank == -1) { // LinearTerm's rank is unassigned
      _rank = rank;
    }
    if (_rank != rank) {
      TEST_FOR_EXCEPTION( true, std::invalid_argument, "Attempting to add terms of unlike rank." );
    }
    _summands.push_back( make_pair( weight, var ) );
    _varIDs.insert(var->ID());
  }
  void addVar(double weight, VarPtr var) {
    FunctionPtr weightFn = Teuchos::rcp( new ConstantScalarFunction(weight) );
    addVar( weightFn, var );
  }
  void addVar(vector<double> vector_weight, VarPtr var) { // dots weight vector with vector var, makes a vector out of a scalar var
    FunctionPtr weightFn = Teuchos::rcp( new ConstantVectorFunction(vector_weight) );
    addVar( weightFn, var );
  }
  
  const set<int> & varIDs() const {
    return _varIDs;
  }
//  vector< EOperatorExtended > varOps(int varID);
  
  // compute the value of linearTerm for non-zero varID at the cubature points, for each basis function in basis
  // values shape: (C,F,P), (C,F,P,D), or (C,F,P,D,D)
  void values(FieldContainer<double> &values, int varID, BasisPtr basis, BasisCachePtr basisCache, bool applyCubatureWeights = false) {
    // can speed things up a lot by handling specially constant weights and 1.0 weights
    // (would need to move this logic into the Function class, and then ConstantFunction can
    //  override to provide the speedup)
    
    int valuesRankExpected = _rank + 3; // 3 for scalar, 4 for vector, etc.
    TEST_FOR_EXCEPTION( valuesRankExpected != values.rank(), std::invalid_argument,
                       "values FC does not have the expected rank" );
    int numCells = values.dimension(0);
    int numFields = values.dimension(1);
    int numPoints = values.dimension(2);
    int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
    
    values.initialize(0.0);
    Teuchos::Array<int> scalarFunctionValueDim;
    scalarFunctionValueDim.append(numCells);
    scalarFunctionValueDim.append(numPoints);
    
    // (could tune things by pre-allocating this storage)
    FieldContainer<double> fValues;
    
    Teuchos::Array<int> vectorFunctionValueDim = scalarFunctionValueDim;
    vectorFunctionValueDim.append(spaceDim);
    Teuchos::Array<int> tensorFunctionValueDim = vectorFunctionValueDim;
    tensorFunctionValueDim.append(spaceDim);
    
    TEST_FOR_EXCEPTION( numCells != basisCache->getPhysicalCubaturePoints().dimension(0),
                       std::invalid_argument, "values FC numCells disagrees with cubature points container");
    TEST_FOR_EXCEPTION( numFields != basis->getCardinality(),
                       std::invalid_argument, "values FC numFields disagrees with basis cardinality");
    TEST_FOR_EXCEPTION( numPoints != basisCache->getPhysicalCubaturePoints().dimension(1),
                       std::invalid_argument, "values FC numPoints disagrees with cubature points container");
    for (vector< LinearSummand >::iterator lsIt = _summands.begin(); lsIt != _summands.end(); lsIt++) {
      LinearSummand ls = *lsIt;
      if (ls.second->ID() == varID) {        
        if (ls.first->rank() == 0) {
          fValues.resize(scalarFunctionValueDim);
        } else if (ls.first->rank() == 1) {
          fValues.resize(vectorFunctionValueDim);
        } else if (ls.first->rank() == 2) {
          fValues.resize(tensorFunctionValueDim);
        } else {
          Teuchos::Array<int> fDim = tensorFunctionValueDim;
          for (int d=3; d < ls.first->rank(); d++) {
            fDim.append(spaceDim);
          }
          fValues.resize(fDim);
        }
        
        ls.first->values(fValues,basisCache);
        constFCPtr basisValues;
        if (applyCubatureWeights) {
          basisValues = basisCache->getTransformedWeightedValues(basis, ls.second->op());
        } else {
          basisValues = basisCache->getTransformedValues(basis, ls.second->op());
        }
        int numFields = basis->getCardinality();
        
        Teuchos::Array<int> fDim(fValues.rank());
        Teuchos::Array<int> bDim(basisValues->rank());
        
        // compute f * basisValues
        if ( ls.first->rank() == ls.second->rank() ) { // scalar result
          int entriesPerPoint = 1;
          int fRank = ls.first->rank();
          for (int d=0; d<fRank; d++) {
            entriesPerPoint *= spaceDim;
          }
          for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            fDim[0] = cellIndex; bDim[0] = cellIndex;
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
              fDim[1] = ptIndex; bDim[2] = ptIndex;
              for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
                const double *fValue = &fValues[fValues.getEnumeration(fDim)];
                bDim[1] = fieldIndex;
                const double *bValue = &((*basisValues)[basisValues->getEnumeration(bDim)]);
                for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
                  values(cellIndex,fieldIndex,ptIndex) += *fValue * *bValue;
                  
//                  cout << "fValue: " << *fValue << endl;
//                  cout << "bValue: " << *bValue << endl;
                  
                  fValue++;
                  bValue++;
                }
              }
            }
          }
        } else { // vector/tensor result
          // could pretty easily fold the scalar case above into the code below
          // (just change the logic in the pointer increments)
          int entriesPerPoint = 1;
          bool scalarF = ls.first->rank() == 0;
          int resultRank = scalarF ? ls.second->rank() : ls.first->rank();
          for (int d=0; d<resultRank; d++) {
            entriesPerPoint *= spaceDim;
          }
          Teuchos::Array<int> vDim( values.rank() );
          for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
            fDim[0] = cellIndex; bDim[0] = cellIndex; vDim[0] = cellIndex;
            for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
              fDim[1] = ptIndex; bDim[2] = ptIndex; vDim[2] = ptIndex;
              for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
                const double *fValue = &fValues[fValues.getEnumeration(fDim)];
                bDim[1] = fieldIndex; vDim[1] = fieldIndex;
                const double *bValue = &(*basisValues)[basisValues->getEnumeration(bDim)];
                
                double *value = &values[values.getEnumeration(vDim)];
                
                for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
                  *value += *fValue * *bValue;
//                  cout << "fValue: " << *fValue << endl;
//                  cout << "bValue: " << *bValue << endl;
                  if (scalarF) {
                    value++; bValue++;
                  } else {
                    value++; fValue++;
                  }
                }
              }
            }
          }
        }
      }
    }
//    cout << "values:\n" << values;
  }
  
  int rank() const { return _rank; }  // 0 for scalar, 1 for vector, etc.
  
  // operator overloading niceties:
  
  LinearTerm& operator=(const LinearTerm &rhs) {
    if ( this == &rhs ) {
      return *this;
    }
    _rank = rhs.rank();
    _summands = rhs.summands();
  }    
  LinearTerm& operator+=(const LinearTerm &rhs) {
    vector< LinearSummand > rhsSummands = rhs.summands();
    for ( vector< LinearSummand >::iterator lsIt = rhsSummands.begin(); 
         lsIt != rhsSummands.end(); lsIt++) {
      _summands.push_back( *lsIt );
    }
    return *this;
  }
  
  LinearTerm& operator+=(VarPtr v) {
    this->addVar(1.0, v);
    return *this;
  }
};

class VarFactory {
  map< string, VarPtr > _testVars;
  map< string, VarPtr > _trialVars;
  int _nextTrialID;
  int _nextTestID;
public:
  VarFactory() {
    _nextTestID = 0;
    _nextTrialID = 0;
  }
  // when there are other scratchpads (e.g. BilinearFormScratchPad), we'll want to share
  // the variables.  The basic function of the factory is to assign unique test/trial IDs.
  
  VarPtr testVar(string name, Space fs) {
    int rank = ((fs == HGRAD) || (fs == L2)) ? 0 : 1;
    _testVars[name] = Teuchos::rcp( new Var( _nextTestID++, rank, name, 
                                            IntrepidExtendedTypes::OPERATOR_VALUE, fs) );
    return _testVars[name];
  }
  VarPtr fieldVar(string name, Space fs = L2) {
    int rank = ((fs == HGRAD) || (fs == L2)) ? 0 : 1;
    _trialVars[name] = Teuchos::rcp( new Var( _nextTrialID++, rank, name,
                                             IntrepidExtendedTypes::OPERATOR_VALUE, fs) );
    return _trialVars[name];
  }
  VarPtr fluxVar(string name, Space fs = L2) { // trace of HDIV  (implemented as L2 on boundary)
    int rank = 0;
    _trialVars[name] = Teuchos::rcp( new Var( _nextTrialID++, rank, name, 
                                             IntrepidExtendedTypes::OPERATOR_VALUE, fs) );
    return _trialVars[name];
  }
  VarPtr traceVar(string name, Space fs = HGRAD) { // trace of HGRAD (implemented as HGRAD on boundary)
    int rank = 0;
    _trialVars[name] = Teuchos::rcp( new Var( _nextTrialID++, rank, name, 
                                             IntrepidExtendedTypes::OPERATOR_VALUE, fs) );    
    return _trialVars[name];
  }
};

class IP : public DPGInnerProduct {
  vector< LinearTermPtr > _linearTerms;
public:
  // to satisfy the compiler, call the DPGInnerProduct constructor with a null argument:
  IP() : DPGInnerProduct( Teuchos::rcp( (BilinearForm*) NULL ) ) {}
  // if the terms are a1, a2, ..., then the inner product is (a1,a1) + (a2,a2) + ... 
  
  void addTerm( LinearTermPtr a) {
    _linearTerms.push_back(a);
  }
  
  void addTerm( VarPtr v ) {
    _linearTerms.push_back( Teuchos::rcp( new LinearTerm(v) ) );
  }
  
  void computeInnerProductMatrix(FieldContainer<double> &innerProduct, 
                                 Teuchos::RCP<DofOrdering> dofOrdering,
                                 Teuchos::RCP<BasisCache> basisCache) {
    // innerProduct FC is sized as (C,F,F)
    FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePoints();
    
    unsigned numCells = physicalCubaturePoints.dimension(0);
    unsigned numPoints = physicalCubaturePoints.dimension(1);
    unsigned spaceDim = physicalCubaturePoints.dimension(2);
    
    shards::CellTopology cellTopo = basisCache->cellTopology();
    
    Teuchos::Array<int> ltValueDim;
    ltValueDim.push_back(numCells);
    ltValueDim.push_back(0); // # fields -- empty until we have a particular basis
    ltValueDim.push_back(numPoints);
    
    innerProduct.initialize(0.0);
    
    for ( vector< LinearTermPtr >:: iterator ltIt = _linearTerms.begin();
         ltIt != _linearTerms.end(); ltIt++) {
      LinearTermPtr lt = *ltIt;
      set<int> testIDs = lt->varIDs();
      int rank = lt->rank();
      
      set<int>::iterator testIt1;
      set<int>::iterator testIt2;
      
      Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > test1Basis, test2Basis;
      
      for (testIt1= testIDs.begin(); testIt1 != testIDs.end(); testIt1++) {
        int testID1 = *testIt1;
        test1Basis = dofOrdering->getBasis(testID1);
        int numDofs1 = test1Basis->getCardinality();
        
        // set up values container for test1
        Teuchos::Array<int> ltValueDim1 = ltValueDim;
        ltValueDim1[1] = numDofs1;
        for (int d=0; d<rank; d++) {
          ltValueDim1.push_back(spaceDim);
        }
        FieldContainer<double> test1Values(ltValueDim1);
        lt->values(test1Values,testID1,test1Basis,basisCache);
        
        for (testIt2= testIDs.begin(); testIt2 != testIDs.end(); testIt2++) {
          int testID2 = *testIt2;
          test2Basis = dofOrdering->getBasis(testID2);
          int numDofs2 = test2Basis->getCardinality();
          
          // set up values container for test2:
          Teuchos::Array<int> ltValueDim2 = ltValueDim1;
          ltValueDim2[1] = numDofs2;
          
          FieldContainer<double> test2ValuesWeighted(ltValueDim2);
          
          lt->values(test2ValuesWeighted,testID2,test2Basis,basisCache,true);
          
          FieldContainer<double> miniMatrix( numCells, numDofs1, numDofs2 );
          
          FunctionSpaceTools::integrate<double>(miniMatrix,test1Values,test2ValuesWeighted,COMP_CPP);
          
//          cout << "miniMatrix:\n" << miniMatrix;
          
          int test1DofOffset = dofOrdering->getDofIndex(testID1,0);
          int test2DofOffset = dofOrdering->getDofIndex(testID2,0);
          
          // there may be a more efficient way to do this copying:
          for (unsigned k=0; k < numCells; k++) {
            for (int i=0; i < numDofs1; i++) {
              for (int j=0; j < numDofs2; j++) {
                innerProduct(k,i+test1DofOffset,j+test2DofOffset) += miniMatrix(k,i,j);
              }
            }
          }
//          cout << "innerProduct:\n" << innerProduct;
        }
      }
    }
//    cout << "final innerProduct:\n" << innerProduct;
  }
  
  void operators(int testID1, int testID2, 
                         vector<IntrepidExtendedTypes::EOperatorExtended> &testOp1,
                         vector<IntrepidExtendedTypes::EOperatorExtended> &testOp2) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "IP::operators() not implemented.");
  }
  
  void printInteractions() {
    cout << "IP::printInteractions() not yet implemented.\n";
  }
};

// operator overloading for syntax sugar:
LinearTermPtr operator+(LinearTermPtr a1, LinearTermPtr a2) {
  LinearTermPtr sum = Teuchos::rcp( new LinearTerm(*a1) );
  *sum += *a2;
  return sum;
}

LinearTermPtr operator+(VarPtr v, LinearTermPtr a) {
  LinearTermPtr sum = Teuchos::rcp( new LinearTerm(*a) );
  *sum += v;
  return sum;
}

LinearTermPtr operator+(LinearTermPtr a, VarPtr v) {
  return v + a;
}

LinearTermPtr operator*(FunctionPtr f, VarPtr v) {
  return Teuchos::rcp( new LinearTerm(f, v) );
}

LinearTermPtr operator*(double weight, VarPtr v) {
  return Teuchos::rcp( new LinearTerm(weight, v) );
}

LinearTermPtr operator*(VarPtr v, double weight) {
  return weight * v;
}

LinearTermPtr operator*(vector<double> weight, VarPtr v) {
  return Teuchos::rcp( new LinearTerm(weight, v) );
}

LinearTermPtr operator*(VarPtr v, vector<double> weight) {
  return weight * v;
}

LinearTermPtr operator/(VarPtr v, double weight) {
  return (1.0 / weight) * v;
}

LinearTermPtr operator-(VarPtr v1, VarPtr v2) {
  return v1 + (-1.0) * v2;
}

LinearTermPtr operator-(LinearTermPtr a, VarPtr v) {
  return a + (-1.0) * v;
}

#endif