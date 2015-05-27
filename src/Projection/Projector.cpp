#include "TypeDefs.h"

#include "Projector.h"

#include "BasisCache.h"
#include "BasisFactory.h"
#include "BasisSumFunction.h"
#include "CamelliaCellTools.h"
#include "Function.h"
#include "VarFactory.h"

#include <stdlib.h>

#include "Shards_CellTopology.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "Intrepid_FieldContainer.hpp"
// Teuchos includes
#include "Teuchos_RCP.hpp"

#include <Epetra_SerialDenseVector.h>
#include <Epetra_SerialDenseMatrix.h>
#include <Epetra_LAPACK.h>
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_DataAccess.h"

using namespace Intrepid;
using namespace Camellia;

template <typename Scalar>
void Projector<Scalar>::projectFunctionOntoBasis(FieldContainer<Scalar> &basisCoefficients, TFunctionPtr<Scalar> fxn,
    BasisPtr basis, BasisCachePtr basisCache, TIPPtr<Scalar> ip, VarPtr v,
    set<int> fieldIndicesToSkip)
{
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
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numDofs = cardinality - fieldIndicesToSkip.size();
  if (numDofs==0)
  {
    // we're skipping all the fields, so just initialize basisCoefficients to 0 and return
    basisCoefficients.resize(numCells,cardinality);
    basisCoefficients.initialize(0);
    return;
  }

  FieldContainer<Scalar> gramMatrix(numCells,cardinality,cardinality);
  FieldContainer<Scalar> ipVector(numCells,cardinality);

  // fake a DofOrdering
  DofOrderingPtr dofOrdering = Teuchos::rcp( new DofOrdering(cellTopo) );
  if (! basisCache->isSideCache())
  {
    dofOrdering->addEntry(v->ID(), basis, v->rank());
  }
  else
  {
    dofOrdering->addEntry(v->ID(), basis, v->rank(), basisCache->getSideIndex());
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

    FieldContainer<Scalar> gramMatrixFiltered(numCells,numDofs,numDofs);
    FieldContainer<Scalar> ipVectorFiltered(numCells,numDofs);
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

//  cout << "physical points for projection:\n" << basisCache->getPhysicalCubaturePoints();
//  cout << "gramMatrix:\n" << gramMatrix;
//  cout << "ipVector:\n" << ipVector;

  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {

    // TODO: rewrite to take advantage of SerialDenseWrapper...
    Epetra_SerialDenseSolver solver;

    Epetra_SerialDenseMatrix A(Copy,
                               &gramMatrix(cellIndex,0,0),
                               gramMatrix.dimension(2),
                               gramMatrix.dimension(2),
                               gramMatrix.dimension(1)); // stride -- fc stores in row-major order (a.o.t. SDM)

    Epetra_SerialDenseVector b(Copy,
                               &ipVector(cellIndex,0),
                               ipVector.dimension(1));

    Epetra_SerialDenseVector x(gramMatrix.dimension(1));

    solver.SetMatrix(A);
    int info = solver.SetVectors(x,b);
    if (info!=0)
    {
      cout << "projectFunctionOntoBasis: failed to SetVectors with error " << info << endl;
    }

    bool equilibrated = false;
    if (solver.ShouldEquilibrate())
    {
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();
      equilibrated = true;
    }

    info = solver.Solve();
    if (info!=0)
    {
      cout << "projectFunctionOntoBasis: failed to solve with error " << info << endl;
    }

    if (equilibrated)
    {
      int successLocal = solver.UnequilibrateLHS();
      if (successLocal != 0)
      {
        cout << "projection: unequilibration FAILED with error: " << successLocal << endl;
      }
    }

    basisCoefficients.resize(numCells,cardinality);
    for (int i=0; i<cardinality; i++)
    {
      if (fieldIndicesToSkip.size()==0)
      {
        basisCoefficients(cellIndex,i) = x(i);
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
          basisCoefficients(cellIndex,i) = x(i_filtered);
        }
      }
    }

  }
}

template <typename Scalar>
void Projector<Scalar>::projectFunctionOntoBasis(FieldContainer<Scalar> &basisCoefficients, TFunctionPtr<Scalar> fxn,
    BasisPtr basis, BasisCachePtr basisCache)
{
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
void Projector<Scalar>::projectFunctionOntoBasisInterpolating(FieldContainer<Scalar> &basisCoefficients, TFunctionPtr<Scalar> fxn,
    BasisPtr basis, BasisCachePtr domainBasisCache)
{
  basisCoefficients.initialize(0);
  CellTopoPtr domainTopo = basis->domainTopology();
  unsigned domainDim = domainTopo->getDimension();

  TIPPtr<Scalar> ip;

  bool traceVar = domainBasisCache->isSideCache();

  pair<TIPPtr<Scalar>, VarPtr> ipVarPair = IP::standardInnerProductForFunctionSpace(basis->functionSpace(), traceVar, domainDim);
  ip = ipVarPair.first;
  VarPtr v = ipVarPair.second;

  TIPPtr<Scalar> ip_l2 = Teuchos::rcp( new TIP<Scalar> );
  ip_l2->addTerm(v);

  // for now, make all projections use L^2... (having some issues with gradients and cell Jacobians--I think we need the restriction of the cell Jacobian to the subcell, e.g., and it's not clear how to do that...)
  ip = ip_l2;

  FieldContainer<double> referenceDomainNodes(domainTopo->getVertexCount(),domainDim);
  CamelliaCellTools::refCellNodesForTopology(referenceDomainNodes, domainTopo);

  int basisCardinality = basis->getCardinality();

  set<int> allDofs;
  for (int i=0; i<basisCardinality; i++)
  {
    allDofs.insert(i);
  }

  for (int d=0; d<=domainDim; d++)
  {
    TFunctionPtr<Scalar> projectionThusFar = BasisSumFunction::basisSumFunction(basis, basisCoefficients);
    TFunctionPtr<Scalar> fxnToApproximate = fxn - projectionThusFar;
    int subcellCount = domainTopo->getSubcellCount(d);
    for (int subcord=0; subcord<subcellCount; subcord++)
    {
      set<int> subcellDofOrdinals = basis->dofOrdinalsForSubcell(d, subcord);
      if (subcellDofOrdinals.size() > 0)
      {
        FieldContainer<double> refCellPoints;
        FieldContainer<double> cubatureWeightsSubcell; // allows us to integrate over the fine subcell even when domain is higher-dimensioned
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
//          Teuchos::RCP<Cubature<double> > subcellCubature = cubFactory.create(subcellTopo, domainBasisCache->cubatureDegree());
          BasisCachePtr subcellCache = Teuchos::rcp( new BasisCache(subcellTopo, domainBasisCache->cubatureDegree(), false) );
          int numPoints = subcellCache->getRefCellPoints().dimension(0);
          refCellPoints.resize(numPoints,domainDim);
          cubatureWeightsSubcell = subcellCache->getCubatureWeights();

          if (d == domainDim)
          {
            refCellPoints = subcellCache->getRefCellPoints();
          }
          else
          {
            CamelliaCellTools::mapToReferenceSubcell(refCellPoints, subcellCache->getRefCellPoints(), d,
                subcord, domainTopo);
          }
        }
        domainBasisCache->setRefCellPoints(refCellPoints, cubatureWeightsSubcell);
        TIPPtr<Scalar> ipForProjection = (d==0) ? ip_l2 : ip; // just use values at vertices (ignore derivatives)
        set<int> dofsToSkip = allDofs;
        for (auto dofOrdinal : subcellDofOrdinals)
        {
          dofsToSkip.erase(dofOrdinal);
        }
        FieldContainer<Scalar> newBasisCoefficients;
        projectFunctionOntoBasis(newBasisCoefficients, fxnToApproximate, basis, domainBasisCache, ipForProjection, v, dofsToSkip);
        for (int cellOrdinal=0; cellOrdinal<newBasisCoefficients.dimension(0); cellOrdinal++)
        {
          for (auto dofOrdinal : subcellDofOrdinals)
          {
            basisCoefficients(cellOrdinal,dofOrdinal) = newBasisCoefficients(cellOrdinal,dofOrdinal);
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
}

namespace Camellia
{
template class Projector<double>;
}

