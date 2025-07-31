//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
#include "TypeDefs.h"

#include "Projector.h"

#include "BasisCache.h"
#include "BasisCacheFactory.hpp"
#include "BasisFactory.h"
#include "BasisSumFunction.h"
#include "CamelliaCellTools.h"
#include "Function.h"
#include "SerialDenseWrapper.h"
#include "VarFactory.h"

#include <stdlib.h>

#include "Shards_CellTopology.hpp"

#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "Camellia_FieldContainer.hpp"
#include "Camellia_FieldContainerConversion.hpp"
// Teuchos includes
#include "Teuchos_RCP.hpp"

using namespace Camellia;

template <typename Scalar>
void Projector<Scalar>::projectFunctionOntoBasis(Camellia::FieldContainer<Scalar> &basisCoefficients, TFunctionPtr<Scalar> fxn,
    BasisPtr basis, BasisCachePtr basisCache, TIPPtr<Scalar> ip, VarPtr v,
    set<int> fieldIndicesToSkip)
{
  TEUCHOS_FUNC_TIME_MONITOR("Projector<Scalar>::projectFunctionOntoBasis");
  CellTopoPtr cellTopo = basis->domainTopology();
  DofOrderingPtr dofOrderPtr = Teuchos::rcp(new DofOrdering(cellTopo));

  if (! fxn.get())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fxn cannot be null!");
  }

  if (fxn->rank() != basis->rangeRank())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function rank must agree with basis rank");
  }

  if (fxn->rank() != v->rank())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function rank must agree with variable rank");
  }

  int cardinality = basis->getCardinality();
  int numCells = basisCache->getPhysicalCubaturePointsFC().dimension(0);
  int numDofs = cardinality - fieldIndicesToSkip.size();
  if (numDofs==0)
  {
    // we're skipping all the fields, so just initialize basisCoefficients to 0 and return
    basisCoefficients.resize(numCells,cardinality);
    basisCoefficients.initialize(0);
    return;
  }

  Camellia::FieldContainer<Scalar> gramMatrix(numCells,cardinality,cardinality);
  Camellia::FieldContainer<Scalar> ipVector(numCells,cardinality);

  // fake a DofOrdering
  DofOrderingPtr dofOrdering = Teuchos::rcp( new DofOrdering(basisCache->cellTopology()) );
  if (basisCache->isSideCache() && ((v->varType()==FLUX) || (v->varType()==TRACE)))
  {
    dofOrdering->addEntry(v->ID(), basis, v->rank(), basisCache->getSideIndex());
  }
  else
  {
    dofOrdering->addEntry(v->ID(), basis, v->rank());
  }

  ip->computeInnerProductMatrix(gramMatrix, dofOrdering, basisCache);
  ip->computeInnerProductVector(ipVector, v, fxn, dofOrdering, basisCache);

  map<int,int> oldToNewIndices;
  if (fieldIndicesToSkip.size() > 0)
  {
    // the code to do with fieldIndicesToSkip might not be terribly efficient...
    // (but it's not likely to be called too frequently)
    int i_indices_skipped = 0;
    for (int i=0; i<cardinality; i++)
    {
      int new_index;
      if (fieldIndicesToSkip.find(i) != fieldIndicesToSkip.end())
      {
        i_indices_skipped++;
        new_index = -1;
      }
      else
      {
        new_index = i - i_indices_skipped;
      }
      oldToNewIndices[i] = new_index;
    }

    Camellia::FieldContainer<Scalar> gramMatrixFiltered(numCells,numDofs,numDofs);
    Camellia::FieldContainer<Scalar> ipVectorFiltered(numCells,numDofs);
    // now filter out the values that we're to skip

    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int i=0; i<cardinality; i++)
      {
        int i_filtered = oldToNewIndices[i];
        if (i_filtered == -1)
        {
          continue;
        }
        ipVectorFiltered(cellIndex,i_filtered) = ipVector(cellIndex,i);

        for (int j=0; j<cardinality; j++)
        {
          int j_filtered = oldToNewIndices[j];
          if (j_filtered == -1)
          {
            continue;
          }
          gramMatrixFiltered(cellIndex,i_filtered,j_filtered) = gramMatrix(cellIndex,i,j);
        }
      }
    }
//    cout << "gramMatrixFiltered:\n" << gramMatrixFiltered;
//    cout << "ipVectorFiltered:\n" << ipVectorFiltered;
    gramMatrix = gramMatrixFiltered;
    ipVector = ipVectorFiltered;
  }

//  cout << "physical points for projection:\n" << basisCache->getPhysicalCubaturePointsFC();
//  cout << "gramMatrix:\n" << gramMatrix;
//  cout << "ipVector:\n" << ipVector;

  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    Teuchos::Array<int> localIPDim(2);
    localIPDim[0] = gramMatrix.dimension(1);
    localIPDim[1] = gramMatrix.dimension(2);
    Teuchos::Array<int> rhsDim(2);
    rhsDim[0] = ipVector.dimension(1);
    rhsDim[1] = 1;
    
    Camellia::FieldContainer<Scalar> cellIPMatrix(localIPDim, &gramMatrix(cellIndex,0,0));
    Camellia::FieldContainer<Scalar> cellRHS(rhsDim, &ipVector(cellIndex,0));
    Camellia::FieldContainer<Scalar> x(gramMatrix.dimension(1),1); // 1: number of RHSes in our current solve
    
    // impose any BCs from the test space:
    if (ip->hasBCsImposed())
    {
      auto testDofOrdinalsToZero = ip->homogeneousBCDofOrdinalsForCell(basisCache->mesh(),basisCache->cellIDs()[cellIndex]);
      for (auto testDofOrdinal : testDofOrdinalsToZero)
      {
        cellRHS(testDofOrdinal,0) = 0.0;
      }
    }
    
    // sanity check: scalar basis should be evaluated at at least as many quadrature points as there are basis functions
    //               (otherwise, the projection will not be uniquely defined)
    // For a vector basis, the rule is that each scalar component needs to have that many quadrature points.
    
    int numQuadraturePoints = basisCache->getPhysicalCubaturePointsFC().dimension(1); // C,P,D
    int numScalarComponents = 1;
    int rangeRank = basis->rangeRank();
    int rangeDim  = basis->rangeDimension();
    for (int rank=0; rank<rangeRank; rank++)
    {
      numScalarComponents *= rangeDim;
    }
    int minQuadraturePoints = numDofs / numScalarComponents;
    
    if (numQuadraturePoints < minQuadraturePoints)
    {
      cout << "numQuadraturePoints = " << numQuadraturePoints << endl;
      cout << "minQuadraturePoints = " << minQuadraturePoints << endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "num quadrature points should be at least as large as the number of dofs onto which we're projecting.");
    }
    
    int result = SerialDenseWrapper::solveSPDSystemMultipleRHS(x, cellIPMatrix, cellRHS);
    if (result != 0)
    {
      cout << "WARNING: in Projector, SerialDenseWrapper::solveSPDSystemMultipleRHS returned result code " << result << endl;
      cout << "physical points for projection:\n" << basisCache->getPhysicalCubaturePointsFC();
      
      ostringstream matrixFileName;
      auto cellID = cellIndex; // basisCache->cellIDs()[cellIndex];
      matrixFileName << "/tmp/A" << cellID << ".dat";
      SerialDenseWrapper::writeMatrixToMatlabFile(matrixFileName.str(), cellIPMatrix);
      cout << "Wrote matrix to " << matrixFileName.str() << endl;
      ostringstream rhsFileName;
      rhsFileName << "/tmp/b" << cellID << ".dat";
      SerialDenseWrapper::writeMatrixToMatlabFile(rhsFileName.str(), cellRHS);
      cout << "Wrote rhs to " << rhsFileName.str() << endl;
      
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "SerialDenseWrapper::solveSPDSystemMultipleRHS failure in Projector");
    }
    
    basisCoefficients.resize(numCells,cardinality);
    for (int i=0; i<cardinality; i++)
    {
      if (fieldIndicesToSkip.size()==0)
      {
        basisCoefficients(cellIndex,i) = x(i,0);
      }
      else
      {
        int i_filtered = oldToNewIndices[i];
        if (i_filtered==-1)
        {
          basisCoefficients(cellIndex,i) = 0.0;
        }
        else
        {
          basisCoefficients(cellIndex,i) = x(i_filtered,0);
        }
      }
    }
  }
}

template <typename Scalar>
void Projector<Scalar>::projectFunctionOntoBasis(Camellia::FieldContainer<Scalar> &basisCoefficients, TFunctionPtr<Scalar> fxn,
    BasisPtr basis, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("Projector<Scalar>::projectFunctionOntoBasis - default IP");
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr var;
  if (! basisCache->isSideCache())
  {
    if (fxn->rank()==0)
    {
      var = varFactory->fieldVar("dummyField");
    }
    else if (fxn->rank()==1)
    {
      var = varFactory->fieldVar("dummyField",VECTOR_L2);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "projectFunctionOntoBasis does not yet support functions of rank > 1.");
    }
  }
  else
  {
    // for present purposes, distinction between trace and flux doesn't really matter,
    // except that parities come into the IP computation for fluxes (even though they'll cancel),
    // and since basisCache doesn't necessarily have parities defined (especially in tests),
    // it's simpler all around to use traces.
    var = varFactory->traceVar("dummyTrace");
  }
  TIPPtr<Scalar> ip = IP::ip();
  ip->addTerm(var); // simple L^2 IP
  projectFunctionOntoBasis(basisCoefficients, fxn, basis, basisCache, ip, var);
}

template <typename Scalar>
void Projector<Scalar>::projectFunctionOntoBasisInterpolating(Camellia::FieldContainer<Scalar> &basisCoefficients, TFunctionPtr<Scalar> fxn,
                                                              BasisPtr basis, BasisCachePtr domainBasisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("Projector<Scalar>::projectFunctionOntoBasisInterpolating");
  basisCoefficients.initialize(0);
  CellTopoPtr basisDomainTopo = basis->domainTopology();
  unsigned basisDomainDim = basisDomainTopo->getDimension();

  TIPPtr<Scalar> ip;

  bool traceVar;
  int sideOrdinal = -1;
  CellTopoPtr domainTopo;
  if (!domainBasisCache->isSideCache())
  {
    domainTopo = domainBasisCache->cellTopology();
    traceVar = false;
  }
  else
  {
    sideOrdinal = domainBasisCache->getSideIndex();
    domainTopo = domainBasisCache->cellTopology()->getSide(sideOrdinal);
    if (domainTopo->getDimension() == basisDomainDim)
    {
      traceVar = true; // side cache with a side basis
    }
    else
    {
      traceVar = false; // side cache with a volume basis
    }
  }
  unsigned domainDim = domainTopo->getDimension();
  
  pair<TIPPtr<Scalar>, VarPtr> ipVarPair = IP::standardInnerProductForFunctionSpace(basis->functionSpace(), traceVar, domainDim);
  ip = ipVarPair.first;
  VarPtr v = ipVarPair.second;

  TIPPtr<Scalar> ip_l2 = Teuchos::rcp( new TIP<Scalar> );
  ip_l2->addTerm(v);

  // for now, make all projections use L^2... (having some issues with gradients and cell Jacobians--I think we need the restriction of the cell Jacobian to the subcell, e.g., and it's not clear how to do that...)
  ip = ip_l2;

  Camellia::FieldContainer<double> referenceDomainNodes(domainTopo->getVertexCount(),domainDim);
  CamelliaCellTools::refCellNodesForTopology(referenceDomainNodes, domainTopo);

  int basisCardinality = basis->getCardinality();

  set<int> allDofs;
  for (int i=0; i<basisCardinality; i++)
  {
    allDofs.insert(i);
  }
  
  BasisPtr conformingBasis;
  int startingDimensionForProjection; // 0 to start with vertices, e.g.
  bool wrapsHGRADBasis = false;
  {
    if ((basis->intrepid2Basis() != Teuchos::null) && (basis->intrepid2Basis()->getFunctionSpace() == Intrepid2::FUNCTION_SPACE_HGRAD))
    {
      wrapsHGRADBasis = true;
    }
  }
  if (Camellia::functionSpaceIsBroken(basis->functionSpace()))
  {
    const bool spaceTime = domainBasisCache->cellTopologyIsSpaceTime();
    conformingBasis = basis->continuousBasis();
    TEUCHOS_TEST_FOR_EXCEPTION(conformingBasis->getCardinality() != basis->getCardinality(), std::logic_error, "continuous basis should have same cardinality as discontinuous");
    startingDimensionForProjection = domainDim;
  }
  else if (wrapsHGRADBasis && (basis->getDegree() > 0) && (basis->functionSpace() == FUNCTION_SPACE_HVOL))
  {
    // sometimes discretizations wrap an Intrepid2 HGRAD basis with a Camellia basis that claims to be HVOL (discontinuous);
    // by using HGRAD basis for projections, we allow "cheating" imposition of BCs on e.g. vertices, since the underlying basis has well-defined vertex functions.
    auto intrepid2Basis = basis->intrepid2Basis();
    conformingBasis = BasisFactory::basisFactory()->getBasis(intrepid2Basis->getDegree(), basis->domainTopology(), FUNCTION_SPACE_HGRAD);
    startingDimensionForProjection = 0;
  }
  else if (Camellia::functionSpaceIsDiscontinuous(basis->functionSpace()))
  {
    conformingBasis = basis;
    startingDimensionForProjection = domainDim;
  }
  else
  {
    conformingBasis = basis;
    startingDimensionForProjection = 0;
  }
  
  basisCoefficients.initialize(0.0);
  set<int> dofsForDomain;
  if (basisDomainDim == domainDim)
  {
    dofsForDomain = allDofs;
  }
  else
  {
    dofsForDomain = conformingBasis->dofOrdinalsForSide(sideOrdinal);
  }
  
  int numCells = basisCoefficients.dimension(0);
  Camellia::FieldContainer<Scalar> wholeBasisCoefficients(numCells, basisCardinality); // will include zeros for any coefficients belonging to other sides for a volume basis, e.g.
  
  FunctionPtr traceFxn = Function::meshSkeletonCharacteristic();
  for (int d=startingDimensionForProjection; d<=domainDim; d++)
  {
    TFunctionPtr<Scalar> projectionThusFar = BasisSumFunction::basisSumFunction(basis, wholeBasisCoefficients);
    if (fxn->boundaryValueOnly())
    {
      projectionThusFar = projectionThusFar * traceFxn;
    }
    TFunctionPtr<Scalar> fxnToApproximate = fxn - projectionThusFar;
    int subcellCount = domainTopo->getSubcellCount(d);
    for (int subcord=0; subcord<subcellCount; subcord++)
    {
      vector<int> subcellDofOrdinals;
      if (basisDomainDim == domainDim)
      {
        if (d > startingDimensionForProjection)
          subcellDofOrdinals = conformingBasis->dofOrdinalsForSubcell(d, subcord);
        else
          subcellDofOrdinals = conformingBasis->dofOrdinalsForSubcell(d, subcord, 0);
      }
      else
      {
        // need to map from the subcord on the side to the one the basis sees
        int subcordBasis = CamelliaCellTools::subcellOrdinalMap(basisDomainTopo, domainDim, sideOrdinal, d, subcord);
        if (d > startingDimensionForProjection)
          subcellDofOrdinals = conformingBasis->dofOrdinalsForSubcell(d, subcordBasis);
        else
          subcellDofOrdinals = conformingBasis->dofOrdinalsForSubcell(d, subcordBasis, 0);
      }
        
      if (subcellDofOrdinals.size() > 0)
      {
        Intrepid2::TensorPoints<double,DeviceType> pointsRefCell;
        Intrepid2::TensorData<double,DeviceType> weightsRefCell;
          CellTopoPtr subcellTopo = domainTopo->getSubcell(d, subcord);
          BasisCachePtr subcellCache = Teuchos::rcp( new BasisCache(subcellTopo, domainBasisCache->cubatureDegree(), false) );
          int numPoints = subcellCache->getRefCellPoints().extent_int(0);
          weightsRefCell = subcellCache->getCubatureWeights();

          if (d == domainDim)
          {
            pointsRefCell = subcellCache->getRefCellPoints();
          }
          else
          {
            CamelliaCellTools::mapToReferenceSubcell(pointsRefCell, subcellCache->getRefCellPoints(), d, subcord, domainTopo);
            
            if (pointsRefCell.numTensorComponents() > weightsRefCell.numTensorComponents())
            {
              // then one (or more) of the point components should be a singleton, corresponding to a node
              // we want to pad weightsRefCell accordingly.
              std::vector< Intrepid2::Data<double,DeviceType> > weightComponents;
              
              int weightComponentOrdinal = 0; // in original weightsRefCell
              
              Intrepid2::Data<double,DeviceType> one(1.0, Kokkos::Array<int,1>{1});
              for (int r = 0; r<pointsRefCell.numTensorComponents(); r++)
              {
                const auto & pointComponent = pointsRefCell.getTensorComponent(r);
                if (pointComponent.extent_int(0) == 1)
                {
                  weightComponents.push_back(one);
                }
                else
                {
                  weightComponents.push_back(weightsRefCell.getTensorComponent(weightComponentOrdinal++));
                }
              }
              weightsRefCell = Intrepid2::TensorData<double, DeviceType>(weightComponents);
            }
            else if (pointsRefCell.numTensorComponents() < weightsRefCell.numTensorComponents())
            {
              TEUCHOS_TEST_FOR_EXCEPTION(pointsRefCell.numTensorComponents() != 1, std::invalid_argument, "Error: weightsRefCell does not have the same number of tensorial components as pointsRefCell, and pointsRefCell has non-trivial tensor structure…");
              
              // if pointsRefCell has trivial tensor structure, we can resolve this by converting weightsRefCell to also have trivial tensor structure.
              auto weightsFC = FieldContainerConversion<double>::fromTensorData(weightsRefCell);
              weightsRefCell = FieldContainerConversion<double>::toTensorData(weightsFC);
            }
          }
        domainBasisCache->setRefCellPoints(pointsRefCell, weightsRefCell, domainBasisCache->cubatureDegree());
        TIPPtr<Scalar> ipForProjection = (d==0) ? ip_l2 : ip; // just use values at vertices (ignore derivatives)
        set<int> dofsToSkip = allDofs;
        for (auto dofOrdinal : subcellDofOrdinals)
        {
          dofsToSkip.erase(dofOrdinal);
        }
         
        Camellia::FieldContainer<Scalar> newBasisCoefficients;
        projectFunctionOntoBasis(newBasisCoefficients, fxnToApproximate, basis, domainBasisCache, ipForProjection, v, dofsToSkip);
        for (int cellOrdinal=0; cellOrdinal<newBasisCoefficients.dimension(0); cellOrdinal++)
        {
          for (auto dofOrdinal : subcellDofOrdinals)
          {
            wholeBasisCoefficients(cellOrdinal,dofOrdinal) = newBasisCoefficients(cellOrdinal,dofOrdinal);
          }
        }
      }
      else
      {
//        cout << "no dof ordinals found for subcell " << subcord << " of dimension " << d << endl;
      }
    }
  }
  
  // now, copy out values corresponding to dofsForDomain into the provided basisCoefficients container
  for (int cellOrdinal = 0; cellOrdinal < numCells; cellOrdinal++)
  {
    int ordinalInBasisCoefficients = 0;
    for (auto dofOrdinal : dofsForDomain)
    {
      basisCoefficients(cellOrdinal,ordinalInBasisCoefficients) = wholeBasisCoefficients(cellOrdinal,dofOrdinal);
      ordinalInBasisCoefficients++;
    }
  }
}

template <typename Scalar>
void Projector<Scalar>::projectChildFunctionOntoParentBasis(Camellia::FieldContainer<Scalar> &basisCoefficients, TFunctionPtr<Scalar> childFxn, BasisPtr parentBasis,
                                                            RefinementPatternPtr refPattern, const std::vector<BasisCachePtr> &subdomainCaches,
                                                            TIPPtr<Scalar> ip, VarPtr v, set<int> fieldIndicesToSkip)
{
  TEUCHOS_FUNC_TIME_MONITOR("Projector<Scalar>::projectChildFunctionOntoParentBasis");
  auto unionCache = BasisCacheFactory<Scalar>::unionBasisCacheForParentCell(refPattern, subdomainCaches);
  
  CellTopoPtr cellTopo = parentBasis->domainTopology();
  DofOrderingPtr dofOrderPtr = Teuchos::rcp(new DofOrdering(cellTopo));
  
  // keep track of whether we're doing this on the skeleton
  bool isSideComputation = subdomainCaches[0]->isSideCache();
  
  VarFactoryPtr varFactory = VarFactory::varFactory();
  
  if (! childFxn.get())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "fxn cannot be null!");
  }
  
  if (childFxn->rank() != parentBasis->rangeRank())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function rank must agree with basis rank");
  }
  
  if (childFxn->rank() != v->rank())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function rank must agree with variable rank");
  }
  
  int cardinality = parentBasis->getCardinality();
  int numCells = unionCache->getPhysicalCubaturePointsFC().dimension(0);
  int numDofs = cardinality - fieldIndicesToSkip.size();
  if (numDofs==0)
  {
    // we're skipping all the fields, so just initialize basisCoefficients to 0 and return
    basisCoefficients.resize(numCells,cardinality);
    basisCoefficients.initialize(0);
    return;
  }
  
  Camellia::FieldContainer<Scalar> gramMatrix(numCells,cardinality,cardinality);
  Camellia::FieldContainer<Scalar> ipVector(numCells,cardinality);
  
  // fake a DofOrdering
  DofOrderingPtr dofOrdering = Teuchos::rcp( new DofOrdering(unionCache->cellTopology()) );
  if (isSideComputation && ((v->varType()==FLUX) || (v->varType()==TRACE)))
  {
    int sideOrdinal = subdomainCaches[0]->getSideIndex();
    dofOrdering->addEntry(v->ID(), parentBasis, v->rank(), sideOrdinal);
  }
  else
  {
    dofOrdering->addEntry(v->ID(), parentBasis, v->rank());
  }
  
  ip->computeInnerProductMatrix(gramMatrix, dofOrdering, unionCache);
  
  auto wrappedChildFxn = BasisCacheFactory<Scalar>::unionBasisCacheSubdomainFunctionWrapper(childFxn, subdomainCaches);
  ip->computeInnerProductVector(ipVector, v, wrappedChildFxn, dofOrdering, unionCache);
  
  map<int,int> oldToNewIndices;
  if (fieldIndicesToSkip.size() > 0)
  {
    // the code to do with fieldIndicesToSkip might not be terribly efficient...
    // (but it's not likely to be called too frequently)
    int i_indices_skipped = 0;
    for (int i=0; i<cardinality; i++)
    {
      int new_index;
      if (fieldIndicesToSkip.find(i) != fieldIndicesToSkip.end())
      {
        i_indices_skipped++;
        new_index = -1;
      }
      else
      {
        new_index = i - i_indices_skipped;
      }
      oldToNewIndices[i] = new_index;
    }
    
    Camellia::FieldContainer<Scalar> gramMatrixFiltered(numCells,numDofs,numDofs);
    Camellia::FieldContainer<Scalar> ipVectorFiltered(numCells,numDofs);
    // now filter out the values that we're to skip
    
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int i=0; i<cardinality; i++)
      {
        int i_filtered = oldToNewIndices[i];
        if (i_filtered == -1)
        {
          continue;
        }
        ipVectorFiltered(cellIndex,i_filtered) = ipVector(cellIndex,i);
        
        for (int j=0; j<cardinality; j++)
        {
          int j_filtered = oldToNewIndices[j];
          if (j_filtered == -1)
          {
            continue;
          }
          gramMatrixFiltered(cellIndex,i_filtered,j_filtered) = gramMatrix(cellIndex,i,j);
        }
      }
    }
    //    cout << "gramMatrixFiltered:\n" << gramMatrixFiltered;
    //    cout << "ipVectorFiltered:\n" << ipVectorFiltered;
    gramMatrix = gramMatrixFiltered;
    ipVector = ipVectorFiltered;
  }
  
  //  cout << "physical points for projection:\n" << basisCache->getPhysicalCubaturePointsFC();
  //  cout << "gramMatrix:\n" << gramMatrix;
  //  cout << "ipVector:\n" << ipVector;
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    Teuchos::Array<int> localIPDim(2);
    localIPDim[0] = gramMatrix.dimension(1);
    localIPDim[1] = gramMatrix.dimension(2);
    Teuchos::Array<int> rhsDim(2);
    rhsDim[0] = ipVector.dimension(1);
    rhsDim[1] = 1;
    
    Camellia::FieldContainer<Scalar> cellIPMatrix(localIPDim, &gramMatrix(cellIndex,0,0));
    Camellia::FieldContainer<Scalar> cellRHS(rhsDim, &ipVector(cellIndex,0));
    Camellia::FieldContainer<Scalar> x(gramMatrix.dimension(1),1); // 1: number of RHSes in our current solve
    
    // impose any BCs from the test space:
    TEUCHOS_TEST_FOR_EXCEPTION(ip->hasBCsImposed(), std::invalid_argument, "ip BC imposition not supported for subdomain projection");
    
    // sanity check: scalar basis should be evaluated at at least as many quadrature points as there are basis functions
    //               (otherwise, the projection will not be uniquely defined)
    // For a vector basis, the rule is that each scalar component needs to have that many quadrature points.
    
    int numQuadraturePoints = unionCache->getPhysicalCubaturePointsFC().dimension(1); // C,P,D
    int numScalarComponents = 1;
    int rangeRank = parentBasis->rangeRank();
    int rangeDim  = parentBasis->rangeDimension();
    for (int rank=0; rank<rangeRank; rank++)
    {
      numScalarComponents *= rangeDim;
    }
    int minQuadraturePoints = numDofs / numScalarComponents;
    
    if (numQuadraturePoints < minQuadraturePoints)
    {
      cout << "numQuadraturePoints = " << numQuadraturePoints << endl;
      cout << "minQuadraturePoints = " << minQuadraturePoints << endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "num quadrature points should be at least as large as the number of dofs onto which we're projecting.");
    }
    
//    {
//      // DEBUGGING
//      cout << "Solving system: Ax=b with A:\n";
//      cout << cellIPMatrix;
//      cout << "b:\n" << cellRHS;
//      Camellia::print("fieldIndicesToSkip", fieldIndicesToSkip);
//    }
    
    int result = SerialDenseWrapper::solveSPDSystemMultipleRHS(x, cellIPMatrix, cellRHS);
    if (result != 0)
    {
      cout << "WARNING: in Projector, SerialDenseWrapper::solveSPDSystemMultipleRHS returned result code " << result << endl;
      cout << "physical points for projection:\n" << unionCache->getPhysicalCubaturePointsFC();
      
      ostringstream matrixFileName;
      auto cellID = cellIndex; // basisCache->cellIDs()[cellIndex];
      matrixFileName << "/tmp/A" << cellID << ".dat";
      SerialDenseWrapper::writeMatrixToMatlabFile(matrixFileName.str(), cellIPMatrix);
      cout << "Wrote matrix to " << matrixFileName.str() << endl;
      ostringstream rhsFileName;
      rhsFileName << "/tmp/b" << cellID << ".dat";
      SerialDenseWrapper::writeMatrixToMatlabFile(rhsFileName.str(), cellRHS);
      cout << "Wrote rhs to " << rhsFileName.str() << endl;
      
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "SerialDenseWrapper::solveSPDSystemMultipleRHS failure in Projector");
    }
    
    basisCoefficients.resize(numCells,cardinality);
    for (int i=0; i<cardinality; i++)
    {
      if (fieldIndicesToSkip.size()==0)
      {
        basisCoefficients(cellIndex,i) = x(i,0);
      }
      else
      {
        int i_filtered = oldToNewIndices[i];
        if (i_filtered==-1)
        {
          basisCoefficients(cellIndex,i) = 0.0;
        }
        else
        {
          basisCoefficients(cellIndex,i) = x(i_filtered,0);
        }
      }
    }
  }
}

template <typename Scalar>
void Projector<Scalar>::projectChildFunctionOntoParentBasisInterpolating(Camellia::FieldContainer<Scalar> &basisCoefficients, TFunctionPtr<Scalar> childFxn,
                                                                         BasisPtr parentBasis, RefinementPatternPtr refPattern, const std::vector<BasisCachePtr> &subdomainCaches)
{
  TEUCHOS_FUNC_TIME_MONITOR("Projector<Scalar>::projectChildFunctionOntoParentBasisInterpolating");
  basisCoefficients.initialize(0);
  CellTopoPtr basisDomainTopo = parentBasis->domainTopology();
  unsigned basisDomainDim = basisDomainTopo->getDimension();
  
  TIPPtr<Scalar> ip;
  
  bool traceVar;
  unsigned sideOrdinal = INVALID_CELLTOPOLOGY_SIDE_ORDINAL;
  CellTopoPtr domainTopo;
  if (!subdomainCaches[0]->isSideCache())
  {
    domainTopo = refPattern->parentTopology();
    traceVar = false;
  }
  else
  {
    // feeling is, probably we want to modify the interface a bit before supporting side evaluations
    // likely want to pass in parent side to which the subdomain caches contribute
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "side caches not (yet) supported by projectChildFunctionOntoParentBasisInterpolating()");
//    domainTopo = domainBasisCache->cellTopology()->getSide(sideOrdinal);
//    if (domainTopo->getDimension() == basisDomainDim)
//    {
//      traceVar = true; // side cache with a side basis
//    }
//    else
//    {
//      traceVar = false; // side cache with a volume basis
//    }
  }
  unsigned domainDim = domainTopo->getDimension();
  
  pair<TIPPtr<Scalar>, VarPtr> ipVarPair = IP::standardInnerProductForFunctionSpace(parentBasis->functionSpace(), traceVar, domainDim);
  ip = ipVarPair.first;
  VarPtr v = ipVarPair.second;
  
  TIPPtr<Scalar> ip_l2 = Teuchos::rcp( new TIP<Scalar> );
  ip_l2->addTerm(v);
  
  // for now, make all projections use L^2... (having some issues with gradients and cell Jacobians--I think we need the restriction of the cell Jacobian to the subcell, e.g., and it's not clear how to do that...)
  ip = ip_l2;
  
  Camellia::FieldContainer<double> referenceDomainNodes(domainTopo->getVertexCount(),domainDim);
  CamelliaCellTools::refCellNodesForTopology(referenceDomainNodes, domainTopo);
  
  int basisCardinality = parentBasis->getCardinality();
  const int minCubatureDegree = basisCardinality * 2;
  
  set<int> allDofs;
  for (int i=0; i<basisCardinality; i++)
  {
    allDofs.insert(i);
  }
  
  BasisPtr conformingBasis;
  int startingDimensionForProjection = -1;
  if (Camellia::functionSpaceIsBroken(parentBasis->functionSpace()))
  {
    conformingBasis = parentBasis->continuousBasis();
    TEUCHOS_TEST_FOR_EXCEPTION(conformingBasis->getCardinality() != parentBasis->getCardinality(), std::logic_error, "continuous basis should have same cardinality as discontinuous");
    startingDimensionForProjection = domainDim;
  }
  else if (Camellia::functionSpaceIsDiscontinuous(parentBasis->functionSpace()))
  {
    conformingBasis = parentBasis;
    startingDimensionForProjection = domainDim;
  }
  else
  {
    conformingBasis = parentBasis;
    startingDimensionForProjection = 0;
  }
  
  basisCoefficients.initialize(0.0);
  set<int> dofsForDomain;
  if (basisDomainDim == domainDim)
  {
    dofsForDomain = allDofs;
  }
  else
  {
    dofsForDomain = conformingBasis->dofOrdinalsForSide(sideOrdinal);
  }
  
  int numCells = basisCoefficients.dimension(0);
  Camellia::FieldContainer<Scalar> wholeBasisCoefficients(numCells, basisCardinality); // will include zeros for any coefficients belonging to other sides for a volume basis, e.g.
  
  int maxCubatureDegreeSubdomain = minCubatureDegree;
  for (auto subdomainCache : subdomainCaches)
  {
    maxCubatureDegreeSubdomain = std::max(maxCubatureDegreeSubdomain, subdomainCache->cubatureDegree());
  }
  
  for (int d=startingDimensionForProjection; d<=domainDim; d++)
  {
    TFunctionPtr<Scalar> projectionThusFar = BasisSumFunction::basisSumFunction(parentBasis, wholeBasisCoefficients);
    TFunctionPtr<Scalar> fxnToApproximate = childFxn - projectionThusFar;
    int subcellCount = domainTopo->getSubcellCount(d);
    for (int subcord=0; subcord<subcellCount; subcord++)
    {
      vector<int> subcellDofOrdinals;
      if (basisDomainDim == domainDim)
      {
        if (d > startingDimensionForProjection)
          subcellDofOrdinals = conformingBasis->dofOrdinalsForSubcell(d, subcord);
        else
          subcellDofOrdinals = conformingBasis->dofOrdinalsForSubcell(d, subcord, 0);
      }
      else
      {
        // need to map from the subcord on the side to the one the basis sees
        int subcordBasis = CamelliaCellTools::subcellOrdinalMap(basisDomainTopo, domainDim, sideOrdinal, d, subcord);
        if (d > startingDimensionForProjection)
          subcellDofOrdinals = conformingBasis->dofOrdinalsForSubcell(d, subcordBasis);
        else
          subcellDofOrdinals = conformingBasis->dofOrdinalsForSubcell(d, subcordBasis, 0);
      }
      
      if (subcellDofOrdinals.size() > 0)
      {
        Camellia::FieldContainer<double> refCellPoints;
        Camellia::FieldContainer<double> cubatureWeightsSubcell; // allows us to integrate over the fine subcell even when domain is higher-dimensioned
        if (d == 0)
        {
          refCellPoints.resize(1,domainDim);
          for (int d1=0; d1<domainDim; d1++)
          {
            refCellPoints(0,d1) = referenceDomainNodes(subcord,d1);
          }
          cubatureWeightsSubcell.resize(1);
          cubatureWeightsSubcell(0) = 1.0;
        }
        else
        {
          CellTopoPtr subcellTopo = domainTopo->getSubcell(d, subcord);
          BasisCachePtr subcellCache = Teuchos::rcp( new BasisCache(subcellTopo, maxCubatureDegreeSubdomain, false) );
          int numPoints = subcellCache->getRefCellPointsFC().dimension(0);
          refCellPoints.resize(numPoints,domainDim);
          cubatureWeightsSubcell = subcellCache->getCubatureWeightsFC();
          
          if (d == domainDim)
          {
            refCellPoints = subcellCache->getRefCellPointsFC();
          }
          else
          {
            CamelliaCellTools::mapToReferenceSubcell(refCellPoints, subcellCache->getRefCellPointsFC(), d,
                                                     subcord, domainTopo);
          }
        }
        for (auto subdomainCache : subdomainCaches)
        {
          subdomainCache->setRefCellPointsFC(refCellPoints, cubatureWeightsSubcell, subdomainCache->cubatureDegree());
        }
        TIPPtr<Scalar> ipForProjection = (d==0) ? ip_l2 : ip; // just use values at vertices (ignore derivatives)
        set<int> dofsToSkip = allDofs;
//        Camellia::print("allDofs", allDofs);
//        Camellia::print("subcellDofOrdinals", subcellDofOrdinals);
        for (auto dofOrdinal : subcellDofOrdinals)
        {
          dofsToSkip.erase(dofOrdinal);
        }
//        Camellia::print("dofsToSkip", dofsToSkip);
        Camellia::FieldContainer<Scalar> newBasisCoefficients;
        
        projectChildFunctionOntoParentBasis(newBasisCoefficients, fxnToApproximate, parentBasis, refPattern, subdomainCaches, ipForProjection, v, dofsToSkip);
        
        for (int cellOrdinal=0; cellOrdinal<newBasisCoefficients.dimension(0); cellOrdinal++)
        {
          for (auto dofOrdinal : subcellDofOrdinals)
          {
            wholeBasisCoefficients(cellOrdinal,dofOrdinal) = newBasisCoefficients(cellOrdinal,dofOrdinal);
            //            cout << "Assigned dofOrdinal " << dofOrdinal << " " << " coefficient " << newBasisCoefficients(cellOrdinal,dofOrdinal);
            //            cout << " (subcord,d) = (" << subcord << "," << d << ")" << endl;
          }
        }
      }
      else
      {
        //        cout << "no dof ordinals found for subcell " << subcord << " of dimension " << d << endl;
      }
    }
  }
  
  // now, copy out values corresponding to dofsForDomain into the provided basisCoefficients container
  for (int cellOrdinal = 0; cellOrdinal < numCells; cellOrdinal++)
  {
    int ordinalInBasisCoefficients = 0;
    for (auto dofOrdinal : dofsForDomain)
    {
      basisCoefficients(cellOrdinal,ordinalInBasisCoefficients) = wholeBasisCoefficients(cellOrdinal,dofOrdinal);
      ordinalInBasisCoefficients++;
    }
  }
}

namespace Camellia
{
template class Projector<double>;
}

