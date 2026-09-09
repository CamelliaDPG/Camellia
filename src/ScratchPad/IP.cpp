//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Camellia_Integrate.hpp"

#include "IP.h"
#include "SerialDenseWrapper.h"
#include "VarFactory.h"
#include "BasisCache.h"
#include "CellTopology.h"

using namespace Camellia;

template <typename Scalar>
TIP<Scalar>::TIP()
{
  _isLegacySubclass = false;
}
// if the terms are a1, a2, ..., then the inner product is (a1,a1) + (a2,a2) + ...

template <typename Scalar>
TIP<Scalar>::TIP(TBFPtr<Scalar> bfs)
{
  _bilinearForm = bfs;
  _isLegacySubclass = true;
}

// added by Nate
template <typename Scalar>
TLinearTermPtr<Scalar> TIP<Scalar>::evaluate(const map< int, TFunctionPtr<Scalar>> &varFunctions)
{
  // include both the boundary and non-boundary parts
  return evaluate(varFunctions,true) + evaluate(varFunctions,false);
}

// added by Jesse - evaluate inner product at given varFunctions
template <typename Scalar>
TLinearTermPtr<Scalar> TIP<Scalar>::evaluate(const map< int, TFunctionPtr<Scalar>> &varFunctions, bool boundaryPart)
{
  TLinearTermPtr<Scalar> ltEval = Teuchos::rcp(new LinearTerm);
  for (typename vector< TLinearTermPtr<Scalar> >:: const_iterator ltIt = _linearTerms.begin(); ltIt != _linearTerms.end(); ltIt++)
  {
    TLinearTermPtr<Scalar> lt = *ltIt;
    TFunctionPtr<Scalar> weight = lt->evaluate(varFunctions,boundaryPart);
    ltEval->addTerm(weight*lt);
  }
  return ltEval;
}

template <typename Scalar>
void TIP<Scalar>::addTerm( TLinearTermPtr<Scalar> a )
{
  _linearTerms.push_back(a);
}

template <typename Scalar>
void TIP<Scalar>::addTerm( VarPtr v )
{
  _linearTerms.push_back( Teuchos::rcp( new LinearTerm(v) ) );
}

template <typename Scalar>
void TIP<Scalar>::addZeroMeanTerm( TLinearTermPtr<Scalar> a)
{
  _zeroMeanTerms.push_back(a);
}

template <typename Scalar>
void TIP<Scalar>::addZeroMeanTerm( VarPtr v )
{
  _zeroMeanTerms.push_back( Teuchos::rcp( new LinearTerm(v) ) );
}

template <typename Scalar>
void TIP<Scalar>::addBoundaryTerm( TLinearTermPtr<Scalar> a )
{
  _boundaryTerms.push_back(a);
}

template <typename Scalar>
void TIP<Scalar>::addBoundaryTerm( VarPtr v )
{
  _boundaryTerms.push_back( Teuchos::rcp( new LinearTerm(v) ) );
}

template <typename Scalar>
void TIP<Scalar>::applyInnerProductData(Camellia::FieldContainer<Scalar> &testValues1,
                                        Camellia::FieldContainer<Scalar> &testValues2,
                                        int testID1, int testID2, int operatorIndex,
                                        Teuchos::RCP<BasisCache> basisCache)
{
  applyInnerProductData(testValues1, testValues2, testID1, testID2, operatorIndex, basisCache->getPhysicalCubaturePointsFC());
}

template <typename Scalar>
void TIP<Scalar>::applyInnerProductData(Camellia::FieldContainer<Scalar> &testValues1,
                                        Camellia::FieldContainer<Scalar> &testValues2,
                                        int testID1, int testID2, int operatorIndex,
                                        const Camellia::FieldContainer<double>& physicalPoints)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "You must override some version of applyInnerProductData!");
}

template <typename Scalar>
void TIP<Scalar>::computeInnerProductMatrix(Camellia::FieldContainer<Scalar> &innerProduct,
    Teuchos::RCP<DofOrdering> dofOrdering, shards::CellTopology &cellTopo,
    Camellia::FieldContainer<double>& physicalCellNodes)
{
  if (_isLegacySubclass)
  {
    CellTopoPtr cellTopoPtr = CellTopology::cellTopology(cellTopo);
    Teuchos::RCP<ElementType> elemTypePtr = Teuchos::rcp( new ElementType(dofOrdering,dofOrdering, cellTopoPtr) );
    Teuchos::RCP<Mesh> nullMeshPtr = Teuchos::rcp( (Mesh*) NULL );
    BasisCachePtr ipBasisCache = Teuchos::rcp(new BasisCache(elemTypePtr, nullMeshPtr,true));
    ipBasisCache->setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(), false);
    computeInnerProductMatrix(innerProduct,dofOrdering,ipBasisCache);
  }
}

template <typename Scalar>
void TIP<Scalar>::computeInnerProductMatrix(Camellia::FieldContainer<Scalar> &innerProduct,
    Teuchos::RCP<DofOrdering> dofOrdering,
    Teuchos::RCP<BasisCache> basisCache)
{
  if (_isLegacySubclass)
  {
    // much of this code is the same as what's in the volume integration in computeStiffness...
    Camellia::FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePointsFC();

    unsigned numCells = physicalCubaturePoints.dimension(0);

    vector<int> testIDs = _bilinearForm->testIDs();
    vector<int>::iterator testIterator1;
    vector<int>::iterator testIterator2;

    BasisPtr test1Basis, test2Basis;

    innerProduct.initialize(0.0);

    for (testIterator1= testIDs.begin(); testIterator1 != testIDs.end(); testIterator1++)
    {
      int testID1 = *testIterator1;
      for (testIterator2= testIDs.begin(); testIterator2 != testIDs.end(); testIterator2++)
      {
        int testID2 = *testIterator2;

        vector<Camellia::EOperator> test1Operators;
        vector<Camellia::EOperator> test2Operators;

        operators(testID1,testID2,test1Operators,test2Operators);

        // check dimensions
        TEUCHOS_TEST_FOR_EXCEPTION( ( test1Operators.size() != test2Operators.size() ),
                                    std::invalid_argument,
                                    "test1Operators.size() and test2Operators.size() do not match.");

        vector<Camellia::EOperator>::iterator op1It;
        vector<Camellia::EOperator>::iterator op2It = test2Operators.begin();
        int operatorIndex = 0;
        for (op1It=test1Operators.begin(); op1It != test1Operators.end(); op1It++)
        {
          Camellia::EOperator op1 = *(op1It);
          Camellia::EOperator op2 = *(op2It);
          Camellia::FieldContainer<Scalar> test1Values; // these will be resized inside applyOperator..
          Camellia::FieldContainer<Scalar> test2Values; // derivative values

          test1Basis = dofOrdering->getBasis(testID1);
          test2Basis = dofOrdering->getBasis(testID2);

          int numDofs1 = test1Basis->getCardinality();
          int numDofs2 = test2Basis->getCardinality();

          Camellia::FieldContainer<Scalar> miniMatrix( numCells, numDofs1, numDofs2 );

          Teuchos::RCP< const Camellia::FieldContainer<Scalar> > test1ValuesTransformedWeighted, test2ValuesTransformed;

          test1ValuesTransformedWeighted = basisCache->getTransformedWeightedValues(test1Basis,op1);
          test2ValuesTransformed = basisCache->getTransformedValues(test2Basis,op2);

          Camellia::FieldContainer<Scalar> innerProductDataAppliedToTest2 = *test2ValuesTransformed; // copy first
          Camellia::FieldContainer<Scalar> innerProductDataAppliedToTest1 = *test1ValuesTransformedWeighted; // copy first

          //cout << "rank of test2ValuesTransformed: " << test2ValuesTransformed->rank() << endl;
          applyInnerProductData(innerProductDataAppliedToTest1, innerProductDataAppliedToTest2,
                                testID1, testID2, operatorIndex, basisCache);

          // Use Camellia::FieldContainer directly.  The generic FunctionSpaceTools overload wraps
          // array-like containers without requiring Intrepid::FieldContainer aliases.
          Camellia::FunctionSpaceTools::integrate<Scalar>(miniMatrix,
                                                          innerProductDataAppliedToTest1,
                                                          innerProductDataAppliedToTest2);
          
          int test1DofOffset = dofOrdering->getDofIndex(testID1,0);
          int test2DofOffset = dofOrdering->getDofIndex(testID2,0);

          // there may be a more efficient way to do this copying:
          for (int i=0; i < numDofs1; i++)
          {
            for (int j=0; j < numDofs2; j++)
            {
              for (unsigned k=0; k < numCells; k++)
              {
                innerProduct(k,i+test1DofOffset,j+test2DofOffset) += miniMatrix(k,i,j);
              }
            }
          }

          op2It++;
          operatorIndex++;
        }

      }
    }
  }
  else     // _isLegacySubclass is false
  {
    // innerProduct FC is sized as (C,F,F)
    const Camellia::FieldContainer<double>* physicalCubaturePoints = &basisCache->getPhysicalCubaturePointsFC();

    unsigned numCells = physicalCubaturePoints->dimension(0);
    unsigned numDofs = dofOrdering->totalDofs();

    innerProduct.initialize(0.0);

    for (typename vector< TLinearTermPtr<Scalar> >:: iterator ltIt = _linearTerms.begin();
         ltIt != _linearTerms.end(); ltIt++)
    {
      TLinearTermPtr<Scalar> lt = *ltIt;
      // integrate lt against itself
      lt->integrate(innerProduct,dofOrdering,lt,dofOrdering,basisCache,basisCache->isSideCache());
    }

    bool enforceNumericalSymmetry = false;
    if (enforceNumericalSymmetry)
    {
      for (unsigned int c=0; c < numCells; c++)
        for (unsigned int i=0; i < numDofs; i++)
          for (unsigned int j=i+1; j < numDofs; j++)
          {
            innerProduct(c,i,j) = (innerProduct(c,i,j) + innerProduct(c,j,i)) / 2.0;
            innerProduct(c,j,i) = innerProduct(c,i,j);
          }
    }

    // boundary terms:
    for (typename vector< TLinearTermPtr<Scalar> >:: iterator btIt = _boundaryTerms.begin();
         btIt != _boundaryTerms.end(); btIt++)
    {
      TLinearTermPtr<Scalar> bt = *btIt;
      bool forceBoundary = true; // force interpretation of this as a term on the element boundary
      bt->integrate(innerProduct,dofOrdering,bt,dofOrdering,basisCache,forceBoundary);
    }

    // zero mean terms:
    for (typename vector< TLinearTermPtr<Scalar> >:: iterator ztIt = _zeroMeanTerms.begin();
         ztIt != _zeroMeanTerms.end(); ztIt++)
    {
      TLinearTermPtr<Scalar> zt = *ztIt;
      Camellia::FieldContainer<Scalar> avgVector(numCells, numDofs);
      // Integrate against 1
      zt->integrate(avgVector, dofOrdering, basisCache);


      // cout << numDofs << avgVector << endl;

      // Sum into innerProduct
      for (unsigned int c=0; c < numCells; c++)
        for (unsigned int i=0; i < numDofs; i++)
          for (unsigned int j=0; j < numDofs; j++)
          {
            Scalar valAdd = avgVector(c, i) * avgVector(c, j);
            // cout << "(" << innerProduct(c, i, j) << ", " << valAdd << ") ";
            innerProduct(c, i, j) += valAdd;
          }
    }
  }
  if (_homogeneousBCs.size() > 0)
  {
    const Camellia::FieldContainer<double>* physicalCubaturePoints = &basisCache->getPhysicalCubaturePointsFC();
    
    unsigned numCells = physicalCubaturePoints->dimension(0);
    
    for (IndexType cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
    {
      auto cellID = basisCache->cellIDs()[cellOrdinal];
      
      auto dofOrdinals = this->homogeneousBCDofOrdinalsForCell(basisCache->mesh(),cellID);
      
      for (auto dofOrdinal : dofOrdinals)
      {
        unsigned numDofs = dofOrdering->totalDofs();
        // zero out (dofOrdinal,i), (i,dofOrdinal)
        for (int i=0; i<numDofs; i++)
        {
          innerProduct(cellOrdinal,dofOrdinal,i) = 0.0;
          innerProduct(cellOrdinal,i,dofOrdinal) = 0.0;
        }
        innerProduct(cellOrdinal,dofOrdinal,dofOrdinal) = 1.0; // 1 in the diagonal
      }
    }
  }
}

template <typename Scalar>
double TIP<Scalar>::computeMaxConditionNumber(DofOrderingPtr testSpace, BasisCachePtr basisCache)
{
  int testDofs = testSpace->totalDofs();
  int numCells = basisCache->cellIDs().size();
  Camellia::FieldContainer<Scalar> innerProduct(numCells,testDofs,testDofs);
  this->computeInnerProductMatrix(innerProduct, testSpace, basisCache);
  double maxConditionNumber = -1;
  Teuchos::Array<int> cellIP_dim;
  cellIP_dim.push_back(testDofs);
  cellIP_dim.push_back(testDofs);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    Camellia::FieldContainer<Scalar> cellIP = Camellia::FieldContainer<Scalar>(cellIP_dim,&innerProduct(cellIndex,0,0) );
    double conditionNumber = SerialDenseWrapper::estimate2NormConditionNumber(cellIP);
    maxConditionNumber = std::max(maxConditionNumber,conditionNumber);
  }
  return maxConditionNumber;
}

// compute TIP vector when var==fxn
template <typename Scalar>
void TIP<Scalar>::computeInnerProductVector(Camellia::FieldContainer<Scalar> &ipVector,
    VarPtr var, TFunctionPtr<Scalar> fxn,
    Teuchos::RCP<DofOrdering> dofOrdering,
    Teuchos::RCP<BasisCache> basisCache, bool sumInto)
{
  // ipVector FC is sized as (C,F)
  Camellia::FieldContainer<double> physicalCubaturePoints = basisCache->getPhysicalCubaturePointsFC();

  if (!fxn.get())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fxn cannot be null!");
  }

  if (!sumInto)
  {
    ipVector.initialize(0.0);
  }

  for (TLinearTermPtr<Scalar> lt : _linearTerms)
  {
    // integrate lt against itself, evaluated at var = fxn
    lt->integrate(ipVector,dofOrdering,lt,var,fxn,basisCache);
  }

  // boundary terms:
  for (TLinearTermPtr<Scalar> bt : _boundaryTerms)
  {
    bool forceBoundary = true; // force interpretation of this as a term on the element boundary
    bt->integrate(ipVector,dofOrdering,bt,var,fxn,basisCache,forceBoundary);
  }

  // zero mean terms:
  for (typename vector< TLinearTermPtr<Scalar> >:: iterator ztIt = _zeroMeanTerms.begin();
       ztIt != _zeroMeanTerms.end(); ztIt++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "zero mean terms not yet supported in TIP vector computation");
  }
}

template <typename Scalar>
std::string TIP<Scalar>::displayString() const
{
  ostringstream displaySS;
  displaySS << "\\Vert v \\Vert_V^2 = ";
  
  const int numTerms = _linearTerms.size();
  for (int termOrdinal = 0; termOrdinal < numTerms; termOrdinal++)
  {
    const auto & term = _linearTerms[termOrdinal];
    displaySS << "\\Vert " << term->displayString() << "\\Vert^2";
    
    if (termOrdinal < numTerms - 1)
    {
      displaySS << " + ";
    }
  }
  return displaySS.str();
}

template <typename Scalar>
std::vector< TLinearTermPtr<Scalar> > TIP<Scalar>::getBoundaryTerms() const
{
  return _boundaryTerms;
}

template <typename Scalar>
std::vector< TLinearTermPtr<Scalar> > TIP<Scalar>::getLinearTerms() const
{
  return _linearTerms;
}

template <typename Scalar>
std::vector< TLinearTermPtr<Scalar> > TIP<Scalar>::getZeroMeanTerms() const
{
  return _zeroMeanTerms;
}

template <typename Scalar>
bool TIP<Scalar>::hasBCsImposed()
{
  return _homogeneousBCs.size() > 0;
}

template <typename Scalar>
bool TIP<Scalar>::hasBoundaryTerms()
{
  if (_isLegacySubclass) return false;
  else return _boundaryTerms.size() > 0;
}

template <typename Scalar>
std::vector<int> TIP<Scalar>::homogeneousBCDofOrdinalsForCell(MeshPtr mesh, GlobalIndexType cellID)
{
  vector<int> dofIndices;
  if (_homogeneousBCs.size() == 0) return dofIndices;
  
  auto cell = mesh->getTopology()->getCell(cellID);
  auto boundarySubcells = cell->boundarySubcells();
  if (boundarySubcells.size() == 0) return dofIndices;
  
  auto dofOrdering = mesh->getElementType(cellID)->testOrderPtr;
  
  for (auto bcEntry : _homogeneousBCs)
  {
    VarPtr var = bcEntry.first;
    
    bool varIsDefinedOnVolume = dofOrdering->isVolumeVar(var->ID());
    if (!varIsDefinedOnVolume)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Imposition of test-space BCs is not (yet) supported for trace test variables");
      // would not be hard to implement; just need to map the subcell ordinals from the cell to the sides (using the reverse map provided in CamelliaCellTools)
    }
    BasisPtr basis = dofOrdering->getBasis(var->ID());
    for (auto boundarySubcell : boundarySubcells)
    {
      unsigned scdim = boundarySubcell.first;
      unsigned scord = boundarySubcell.second;
      auto basisDofOrdinals = basis->dofOrdinalsForSubcell(scdim, scord);
      for (auto basisDofOrdinal : basisDofOrdinals)
      {
        auto cellDofOrdinal = dofOrdering->getDofIndex(var->ID(), basisDofOrdinal);
        dofIndices.push_back(cellDofOrdinal);
      }
    }
  }
  return dofIndices;
}

template <typename Scalar>
void TIP<Scalar>::imposeHomogeneousBCs(VarPtr var, SpatialFilterPtr meshBoundaryFilter)
{
  _homogeneousBCs.push_back({var,meshBoundaryFilter});
}

// ! returns the number of potential nonzeros for the given trial ordering and test ordering
template <typename Scalar>
int TIP<Scalar>::nonZeroEntryCount(DofOrderingPtr testOrdering)
{
  int nonZeros = 0;
  
  set<pair<int,int>> testInteractions;
  
  for (TLinearTermPtr<Scalar> lt : _linearTerms)
  {
    set<int> varIDs = lt->varIDs();
    for (int test1 : varIDs)
    {
      for (int test2 : varIDs)
      {
        testInteractions.insert({test1,test2});
      }
    }
  }
  for (TLinearTermPtr<Scalar> lt : _boundaryTerms)
  {
    set<int> varIDs = lt->varIDs();
    for (int test1 : varIDs)
    {
      for (int test2 : varIDs)
      {
        testInteractions.insert({test1,test2});
      }
    }
  }
  for (TLinearTermPtr<Scalar> lt : _zeroMeanTerms)
  {
    set<int> varIDs = lt->varIDs();
    for (int test1 : varIDs)
    {
      for (int test2 : varIDs)
      {
        testInteractions.insert({test1,test2});
      }
    }
  }

  for (pair<int,int> testPair : testInteractions)
  {
    int test1Cardinality = testOrdering->getBasis(testPair.first)->getCardinality();
    int test2Cardinality = testOrdering->getBasis(testPair.second)->getCardinality();
    nonZeros += test1Cardinality * test2Cardinality;
  }
  
  return nonZeros;
}

template <typename Scalar>
void TIP<Scalar>::operators(int testID1, int testID2,
                            vector<Camellia::EOperator> &testOp1,
                            vector<Camellia::EOperator> &testOp2)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "TIP<Scalar>::operators() not implemented.");
}

template <typename Scalar>
void TIP<Scalar>::printInteractions()
{
  if (_isLegacySubclass)
  {
    cout << "Inner product: test interactions\n";
    vector<int> testIDs = _bilinearForm->testIDs();
    for (vector<int>::iterator testIt = testIDs.begin(); testIt != testIDs.end(); testIt++)
    {
      int testID = *testIt;
      cout << endl << "****** Interactions with test variable " << _bilinearForm->testName(testID) << " ******* " << endl;
      bool first = true;
      for (vector<int>::iterator testIt2 = testIDs.begin(); testIt2 != testIDs.end(); testIt2++)
      {
        int testID2 = *testIt2;
        vector<Camellia::EOperator> ops1, ops2;
        operators(testID, testID2, ops1, ops2);
        int numOps = ops1.size();
        for (int i=0; i<numOps; i++)
        {
          if ( ! first) cout << " + ";
          cout << Camellia::operatorName(ops1[i]) << " " << _bilinearForm->testName(testID) << " ";
          cout << Camellia::operatorName(ops2[i]) << " " << _bilinearForm->testName(testID2);
          first = false;
        }
      }
      cout << endl;
    }
    return;
  }
  else
  {
    cout << "_linearTerms:\n";
    for (typename vector< TLinearTermPtr<Scalar> >::iterator ltIt = _linearTerms.begin();
         ltIt != _linearTerms.end(); ltIt++)
    {
      cout << (*ltIt)->displayString() << endl;
    }
    cout << "_boundaryTerms:\n";
    for (typename vector< TLinearTermPtr<Scalar> >::iterator ltIt = _boundaryTerms.begin();
         ltIt != _boundaryTerms.end(); ltIt++)
    {
      cout << (*ltIt)->displayString() << endl;
    }
    cout << "_zeroMeanTerms:\n";
    for (typename vector< TLinearTermPtr<Scalar> >::iterator ltIt = _zeroMeanTerms.begin();
         ltIt != _zeroMeanTerms.end(); ltIt++)
    {
      cout << (*ltIt)->displayString() << endl;
    }
  }
}

template <typename Scalar>
pair<TIPPtr<Scalar>, VarPtr> TIP<Scalar>::standardInnerProductForFunctionSpace(Camellia::EFunctionSpace fs, bool useTraceVar, int spaceDim)
{
  TIPPtr<Scalar> ip = Teuchos::rcp( new TIP<Scalar> );
  VarFactoryPtr vf = VarFactory::varFactory();
  Camellia::Space space = Camellia::spaceForEFS(fs);
  VarPtr var = useTraceVar ? vf->traceVar("v",space) : vf->testVar("v", space);

  ip->addTerm(var);

  switch (fs)
  {
  case Camellia::FUNCTION_SPACE_HVOL:
  case Camellia::FUNCTION_SPACE_HVOL_DISC:
  case Camellia::FUNCTION_SPACE_VECTOR_HVOL:
    break;
  case Camellia::FUNCTION_SPACE_HGRAD:
  case Camellia::FUNCTION_SPACE_HGRAD_DISC:
  case Camellia::FUNCTION_SPACE_VECTOR_HGRAD:
    ip->addTerm(var->grad());
    break;
  case Camellia::FUNCTION_SPACE_HCURL:
  case Camellia::FUNCTION_SPACE_HCURL_DISC:
    ip->addTerm(var->curl(spaceDim));
    break;
  case Camellia::FUNCTION_SPACE_HDIV:
  case Camellia::FUNCTION_SPACE_HDIV_DISC:
    ip->addTerm(var->div());
    break;
      // TODO: check the logic of this for space-time test variables, now that we've revised Var to have an explicit, separate function space defined in temporal dimension
//      // space-time:
//    case Camellia::FUNCTION_SPACE_HGRAD_SPACE_HVOL_TIME:
//      ip->addTerm(var->grad()); // by default, grad means spatial gradient
//      // space-time:
//    case Camellia::FUNCTION_SPACE_HVOL_SPACE_HGRAD_TIME:
//      ip->addTerm(var->dt());
  case Camellia::FUNCTION_SPACE_REAL_SCALAR:
    break;

  default:
    cout << "Error: unhandled function space.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled function space");
  }
  return make_pair(ip,var);
}

template <typename Scalar>
TIPPtr<Scalar> TIP<Scalar>::ip()
{
  return Teuchos::rcp( new TIP<Scalar> );
}

namespace Camellia
{
template class TIP<double>;
}
