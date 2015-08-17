//
//  ConditioningExperimentDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 3/12/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include <iostream>

#include <Teuchos_GlobalMPISession.hpp>

#include "BasisFactory.h"
#include "BF.h"
#include "DataIO.h"
#include "Function.h"
#include "hFunction.h"
#include "IP.h"
#include "MeshFactory.h"
#include "MeshUtilities.h"
#include "SerialDenseMatrixUtility.h"
#include "VarFactory.h"

#include "Legendre.hpp"
#include "Lobatto.hpp"

#include "Teuchos_CommandLineProcessor.hpp"

enum TestType
{
  Mass,
  Stiffness,
  FullNorm
};

using namespace Camellia;
using namespace Intrepid;

void setupHCurlTest(TestType testType, VarFactory &varFactory, VarPtr &var, IPPtr &ip)
{
  int spaceDim = 2;
  var = varFactory.testVar("\\omega", HCURL);
  ip = Teuchos::rcp( new IP );
  if (testType==Mass)
  {
    ip->addTerm(var);
  }
  else if (testType==FullNorm)
  {
    FunctionPtr h = Teuchos::rcp( new hFunction );
    ip->addTerm(var);
    ip->addTerm(h * var->curl(spaceDim));
  }
  else if (testType==Stiffness)
  {
    ip->addTerm(var->curl(spaceDim));
  }
}

void setupHDivTest(TestType testType, VarFactory &varFactory, VarPtr &var, IPPtr &ip)
{
  var = varFactory.testVar("\\tau", HDIV);
  ip = Teuchos::rcp( new IP );
  if (testType==Mass)
  {
    ip->addTerm(var);
  }
  else if (testType==FullNorm)
  {
    FunctionPtr h = Teuchos::rcp( new hFunction );
    ip->addTerm(var);
    ip->addTerm(h * var->div());
  }
  else if (testType==Stiffness)
  {
    ip->addTerm(var->div());
  }
}

void setupHGradTest(TestType testType, VarFactory &varFactory, VarPtr &var, IPPtr &ip)
{
  var = varFactory.testVar("q", HGRAD);
  ip = Teuchos::rcp( new IP );
  if (testType==Mass)
  {
    ip->addTerm(var);
  }
  else if (testType==FullNorm)
  {
    FunctionPtr h = Teuchos::rcp( new hFunction );
    ip->addTerm(var);
    ip->addTerm(h * var->grad());
  }
  else if (testType==Stiffness)
  {
    ip->addTerm(var->grad());
  }
}

void setupHGradStiffness(VarFactoryPtr &varFactory, VarPtr &var, IPPtr &ip)
{
  varFactory = Teuchos::rcp( new VarFactory() );
  var = varFactory->testVar("q", HGRAD);
  ip = Teuchos::rcp( new IP );
  ip->addTerm(var->grad());
}

void setupHDivStiffness(VarFactoryPtr &varFactory, VarPtr &var, IPPtr &ip)
{
  varFactory = Teuchos::rcp( new VarFactory() );
  var = varFactory->testVar("q", HDIV);
  ip = Teuchos::rcp( new IP );
  ip->addTerm(var->div());
}

void setupHDivMass(VarFactoryPtr &varFactory, VarPtr &var, IPPtr &ip)
{
  varFactory = Teuchos::rcp( new VarFactory() );
  var = varFactory->testVar("q", HDIV);
  ip = Teuchos::rcp( new IP );
  ip->addTerm(var);
}

void printLobattoL2norm()
{
  VarFactoryPtr varFactory = Teuchos::rcp(new VarFactory);
  VarPtr u = varFactory->fieldVar("u"); // we don't really care about the trial space
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );

  int maxOrder = 20;
  int pToAdd = 0;
  MeshPtr mesh = MeshFactory::quadMesh(bf, maxOrder, pToAdd, 2.0, 2.0); // width = 2, height = 2: shifted reference element

  cout << setprecision(15);

  cout << "L^2 norm squared of (non-conforming) Lobatto polynomials:\n";
  for (int i=0; i<maxOrder; i++)
  {
    FunctionPtr lobatto = Teuchos::rcp(new LobattoFunction<>(i, false) );
    cout << i << ": " << (lobatto * lobatto)->integrate(mesh) << endl;
  }
}

string testTypeName(TestType testType)
{
  switch (testType)
  {
  case Mass:
    return "Mass";
    break;
  case FullNorm:
    return "Full Norm";
    break;
  case Stiffness:
    return "Stiffness";
    break;

  default:
    break;
  }
}

int main(int argc, char *argv[])
{
  int maxTestOrder = 15;
  bool useLobatto = false;
  double h = 1.0;

  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  // read args:
  cmdp.setOption("maxOrder", &maxTestOrder, "test space polynomial order enrichment");
  cmdp.setOption("useLobatto", "useIntrepid", &useLobatto, "Use Lobatto basis (otherwise will use Intrepid's nodal basis)");
  cmdp.setOption("h", &h, "mesh width");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  cout << "maxOrder = " << maxTestOrder << endl;
  if (useLobatto)
  {
    cout << "Using Lobatto HGRAD and HDIV.\n";
  }
  else
  {
    cout << "Using Intrepid's HGRAD, HDIV, HCURL bases.\n";
  }

  BasisFactory::basisFactory()->setUseLobattoForQuadHDiv(useLobatto);
  BasisFactory::basisFactory()->setUseLobattoForQuadHGrad(useLobatto);

//  printLobattoL2norm();

//  FieldContainer<double> conditionTest(2,2);
//  conditionTest(0,0) = 1;
//  conditionTest(1,1) = 1e-17;
//  SerialDenseMatrixUtility::jacobiScaleMatrix(conditionTest);

//  double condest = SerialDenseMatrixUtility::estimate1NormConditionNumber(conditionTest);
//  cout << "condest for diagonal matrix: " << condest << endl;

  vector< Space > spaces;
  spaces.push_back(HDIV);
  spaces.push_back(HGRAD);
  if (! useLobatto)
  {
    spaces.push_back(HCURL);
  }
  else
  {
    cout << "Skipping HCURL because this isn't yet supported for Lobatto.\n";
  }
  vector< TestType > testTypes;
  testTypes.push_back(Stiffness);
  testTypes.push_back(Mass);
  testTypes.push_back(FullNorm);
  for (vector<TestType>::iterator typeIt = testTypes.begin(); typeIt != testTypes.end(); typeIt++)
  {
    TestType testType = *typeIt;
    string typeName = testTypeName(testType);
    cout << "*************** " << typeName << " tests ***************\n";
    for (vector< Space >::iterator spaceIt = spaces.begin(); spaceIt != spaces.end(); spaceIt++)
    {
      Space space = *spaceIt;
      VarFactoryPtr varFactory = Teuchos::rcp(new VarFactory);
      VarPtr var;
      IPPtr ip;
      string spaceName;
      if (space==HGRAD)
      {
        spaceName = "grad";
        setupHGradTest(testType, *varFactory, var, ip);
      }
      else if (space==HDIV)
      {
        spaceName = "div";
        setupHDivTest(testType, *varFactory, var, ip);
      }
      else if (space==HCURL)
      {
        spaceName = "curl";
        setupHCurlTest(testType, *varFactory, var, ip);
      }
      cout << spaceName << ":\n";
      VarPtr u = varFactory->fieldVar("u"); // we don't really care about the trial space
      BFPtr bf = Teuchos::rcp( new BF(varFactory) );
      int pToAdd = 0;
      for (int testOrder=1; testOrder<=maxTestOrder; testOrder++)
      {
        MeshPtr mesh = MeshFactory::quadMesh(bf, testOrder, pToAdd, h, h); // width = h, height = h
        ostringstream fileNameStream;
        fileNameStream << spaceName << "_" << typeName << "_p" << testOrder << ".dat";
        string fileName = fileNameStream.str();
        bool jacobiScaling = true; // (testType != Stiffness);
        double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ip, mesh, jacobiScaling, fileName);
        cout << scientific << setprecision(3) << maxConditionNumber << endl;
      }
    }


    // finally, write out the HGrad stiffness matrix to disk:

    map<string, VarFactoryPtr > varFactories;
    map<string, IPPtr > ips;
    map<string, VarPtr > vars;

    VarFactoryPtr varFactory;
    VarPtr var;
    IPPtr ip;
    setupHGradStiffness(varFactory, var, ip);
    string HGRAD_STIFFNESS = "HGRAD_stiffness.dat";
    string HDIV_STIFFNESS = "HDIV_stiffness.dat";
    string HDIV_MASS = "HDIV_MASS.dat";
    varFactories[HGRAD_STIFFNESS] = varFactory;
    ips[HGRAD_STIFFNESS] = ip;
    vars[HGRAD_STIFFNESS] = var;
    setupHDivMass(varFactory, var, ip);
    varFactories[HDIV_MASS] = varFactory;
    ips[HDIV_MASS] = ip;
    vars[HDIV_MASS] = var;
    setupHDivStiffness(varFactory, var, ip);
    varFactories[HDIV_STIFFNESS] = varFactory;
    ips[HDIV_STIFFNESS] = ip;
    vars[HDIV_STIFFNESS] = var;

    for (auto namedVarFactory : varFactories)
    {
      string name = namedVarFactory.first;
      varFactory = namedVarFactory.second;
      var = vars[name];
      ip = ips[name];
      VarPtr u = varFactory->fieldVar("u"); // we don't really care about the trial space
      BFPtr bf = Teuchos::rcp( new BF(varFactory) );

      int testOrder = 5;
      int pToAdd = 0;
      MeshPtr mesh = MeshFactory::quadMesh(bf, testOrder, pToAdd, 1.0, 1.0); // width = 1, height = 1: unit quad

      bool testVsTest = true;
      int cellID = 0;
      BasisCachePtr cellBasisCache = BasisCache::basisCacheForCell(mesh, cellID, testVsTest);

      DofOrderingPtr testSpace = mesh->getElement(cellID)->elementType()->testOrderPtr;
      int testDofs = testSpace->totalDofs();
      int numCells = 1;
      FieldContainer<double> innerProductMatrix(numCells,testDofs,testDofs);
      ip->computeInnerProductMatrix(innerProductMatrix, testSpace, cellBasisCache);
      // reshape:
      innerProductMatrix.resize(testDofs,testDofs);
      DataIO::writeMatrixToSparseDataFile(innerProductMatrix, name);
      cout << "Wrote " << name << endl;
    }
  }

//  int polyOrder = 20;
//  FieldContainer<double> values(polyOrder+1), dvalues(polyOrder+1);
//  double x = 0.5;
//  Legendre<>::values(values,dvalues,x,polyOrder);
//  cout << "Legendre values at x=0.5:\n";
//  for (int i=0; i<values.size(); i++) {
//    cout << i << ": " << values[i] << endl;
//  }
//  cout << "Legendre derivatives at x=0.5:\n";
//  for (int i=0; i<dvalues.size(); i++) {
//    cout << i << ": " << dvalues[i] << endl;
//  }
//
//  Lobatto<>::values(values,dvalues,x,polyOrder);
//  cout << "Lobatto values at x=0.5:\n";
//  for (int i=0; i<values.size(); i++) {
//    cout << i << ": " << values[i] << endl;
//  }
//  cout << "Lobatto derivatives at x=0.5:\n";
//  for (int i=0; i<dvalues.size(); i++) {
//    cout << i << ": " << dvalues[i] << endl;
//  }
  return 0;
}