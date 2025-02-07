//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  BasisCacheTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 12/11/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include "BasisFactory.h"
#include "BasisSumFunction.h"
#include "CamelliaCellTools.h"
#include "CamelliaTestingHelpers.h"
#include "CellTopology.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "SerialDenseWrapper.h"
#include "Solution.h"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include <Intrepid2_TestUtils.hpp>

#include "Basis1DIntegralCache.hpp"

using namespace Camellia;

namespace
{

void testIntegrate(BasisFamilyPtr basisFamily,
                   Space space1, Space space2,
                   int p1, int p2, bool differentiate1, bool differentiate2,
                   int monomialDegree, bool layoutLeft,
                   Teuchos::FancyOStream &out, bool &success)
{
  int p = p1, dp = p2 - p1;
  double x0 = -1, x1 = 1; // match reference space with physical space: so far, anyway, Basis1DIntegralCache does everything in reference space
  
  VarFactoryPtr vf = VarFactory::varFactory();
  auto u = vf->fieldVar("u", space1);
  auto v = vf->testVar("v", space2);
  
  vf->setBasisFamily(basisFamily);
  
  FunctionPtr xn = Function::xn(monomialDegree);
  
  BFPtr bf = BF::bf(vf);
  auto uForIntegral = differentiate1 ? u->dx() : u;
  auto vForIntegral = differentiate2 ? v->dx() : v;
  bf->addTerm(uForIntegral, xn * vForIntegral);
  
  auto mesh = MeshFactory::intervalMesh(bf, x0, x1, 1, p, dp);
  
  auto elemType = mesh->getElementType(0);
  auto uBasis = elemType->trialOrderPtr->getBasis(u->ID());
  auto vBasis = elemType->testOrderPtr->getBasis(v->ID());
  
  const int numCells = 1;
  
  FieldContainer<double> cellSideParities(numCells,2); // unused
  FieldContainer<double> expectedIntegral;
  
  if (layoutLeft)
  {
    expectedIntegral = FieldContainer<double>(numCells, vBasis->getCardinality(), uBasis->getCardinality());
  }
  else
  {
    expectedIntegral = FieldContainer<double>(numCells, uBasis->getCardinality(), vBasis->getCardinality());
  }
  
  auto basisCache = BasisCache::basisCache1D(x0, x1, p * 2 + dp + monomialDegree);
  
  bf->stiffnessMatrix(expectedIntegral, elemType, cellSideParities, basisCache, !layoutLeft, false);
  
  expectedIntegral.resize(vBasis->getCardinality() * uBasis->getCardinality());
  
  auto expectedIntegralView = expectedIntegral.checkoutDeviceView();
  
  Basis1DIntegralCache<double> integralCache;
  
  int quadratureDegree = -1; // lets integralCache select degree
  auto refIntegral = integralCache.getReferenceSpaceIntegral(uBasis, differentiate1, vBasis, differentiate2,
                                                             monomialDegree, quadratureDegree, layoutLeft);
  
  double tol = 1e-14;
  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(refIntegral, expectedIntegralView, tol);
  
  expectedIntegral.checkinDeviceView(false);
  
  // now try getting the refIntegral a second time; should load from cache:
  auto refIntegral2 = integralCache.getReferenceSpaceIntegral(uBasis, differentiate1, vBasis, differentiate2,
                                                              monomialDegree, quadratureDegree, layoutLeft);
  
  // check that the two views are at the same location in memory:
  TEST_EQUALITY(refIntegral.data(), refIntegral2.data());
}

void testIntegrate3(BasisFamilyPtr basisFamily,
                    Space space1, Space space2, Space space3,
                    int p1, int p2, int p3,
                    bool differentiate1, bool differentiate2, bool differentiate3,
                    int monomialDegree, bool layoutLeft,
                    Teuchos::FancyOStream &out, bool &success)
{
  // first test: can we run to completion for the trilinear integral?
  
  CellTopoPtr line = CellTopology::line();
  auto basis1 = basisFamily->getBasis(line, efsForSpace(space1), p1);
  auto basis2 = basisFamily->getBasis(line, efsForSpace(space2), p2);
  auto basis3 = basisFamily->getBasis(line, efsForSpace(space3), p3);
  
  Basis1DIntegralCache<double> integralCache;
  
  int quadratureDegree = -1; // lets integralCache select degree
  auto refIntegral = integralCache.getReferenceSpaceIntegral(basis1, differentiate1, basis2, differentiate2, basis3, differentiate3,
                                                             monomialDegree, quadratureDegree, layoutLeft);
  
  // now try getting the refIntegral a second time; should load from cache:
  auto refIntegral2 = integralCache.getReferenceSpaceIntegral(basis1, differentiate1, basis2, differentiate2, basis3, differentiate3,
                                                              monomialDegree, quadratureDegree, layoutLeft);
  
  // check that the two views are at the same location in memory:
  TEST_EQUALITY(refIntegral.data(), refIntegral2.data());
}

void testIntegrate3LinearBasis(BasisFamilyPtr basisFamily,
                               bool layoutLeft,
                               Teuchos::FancyOStream &out, bool &success)
{
  // test the values for a particular case: linear basis functions, which (independent of BasisFamily) are (1 - x)/2 and (1 + x)/2, in that order.
  // symmetries are such that the result of the integration is a function only of the number of like multiplicands, which is 3 or 2.
  // 3 like: Wolfram Alpha: "integrate ((1-x)/2)^3 from -1 to 1"; result: 1/2
  // 2 like: Wolfram Alpha: "Integrate (1-x)(1-x)(1+x)/8 from -1 to 1"; result: 1/6
  
  // Note that the symmetries involved in this test also mean that right layout and left layout coincide in the expected layout
  // of values in memory; this does not verify that the layouts are correct in more general cases.
  
  CellTopoPtr line = CellTopology::line();
  
  const int p = 1;
  const int monomialDegree = 0;
  const Space space = HGRAD;
  const bool differentiate = false;
  
  auto basis = basisFamily->getBasis(line, efsForSpace(space), p);
  
  Basis1DIntegralCache<double> integralCache;
  
  int quadratureDegree = -1; // lets integralCache select degree
  auto refIntegral = integralCache.getReferenceSpaceIntegral(basis, differentiate, basis, differentiate, basis, differentiate,
                                                             monomialDegree, quadratureDegree, layoutLeft);
  
  // now try getting the refIntegral a second time; should load from cache:
  auto refIntegral2 = integralCache.getReferenceSpaceIntegral(basis, differentiate, basis, differentiate, basis, differentiate,
                                                              monomialDegree, quadratureDegree, layoutLeft);
  
  // check that the two views are at the same location in memory:
  TEST_EQUALITY(refIntegral.data(), refIntegral2.data());

  const int basisCardinality = basis->getCardinality();
  FieldContainer<double> expectedIntegral(basisCardinality,basisCardinality,basisCardinality);
  for (int i0=0; i0<basisCardinality; i0++)
  {
    for (int i1=0; i1<basisCardinality; i1++)
    {
      for (int i2=0; i2<basisCardinality; i2++)
      {
        // since i0, i1, i2 are all 0 or 1:
        // 2 like basis functions: sum is 1 or 2
        // 3 like basis functions: sum is 0 or 3
        const int iSum = i0 + i1 + i2;
        const double integral = ((iSum == 1) || (iSum == 2)) ? 1./6. : 1./2.;
        expectedIntegral(i0,i1,i2) = integral;
      }
    }
  }
  
  // flatten to 1D view
  expectedIntegral.resize(basisCardinality * basisCardinality * basisCardinality);
  auto expectedIntegralView = expectedIntegral.checkoutDeviceView();
  
  double tol = 1e-14;
  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(refIntegral, expectedIntegralView, tol);
  
  expectedIntegral.checkinDeviceView(false);
}

void testIntegrate4LinearBasis(BasisFamilyPtr basisFamily,
                               bool layoutLeft,
                               Teuchos::FancyOStream &out, bool &success)
{
  // test the values for a particular case: linear basis functions, which (independent of BasisFamily) are (1 - x)/2 and (1 + x)/2, in that order.
  // symmetries are such that the result of the integration is a function only of the number of like multiplicands, which is 4, 3, or 2.
  // 4 like: Wolfram Alpha: "integrate ((1-x)/2)^4 from -1 to 1"; result: 0.4
  // 3 like: Wolfram Alpha: "integrate (1-x)(1-x)(1-x)(1+x)/16 from -1 to 1"; result: 0.1
  // 2 like: Wolfram Alpha: "Integrate (1-x)(1-x)(1+x)(1+x)/16 from -1 to 1"; result: 1/15
  
  // Note that the symmetries involved in this test also mean that right layout and left layout coincide in the expected layout
  // of values in memory; this does not verify that the layouts are correct in more general cases.
  
  CellTopoPtr line = CellTopology::line();
  
  const int p = 1;
  const int monomialDegree = 0;
  const Space space = HGRAD;
  const bool differentiate = false;
  
  auto basis = basisFamily->getBasis(line, efsForSpace(space), p);
  
  Basis1DIntegralCache<double> integralCache;
  
  int quadratureDegree = -1; // lets integralCache select degree
  auto refIntegral = integralCache.getReferenceSpaceIntegral(basis, differentiate, basis, differentiate, basis, differentiate, basis, differentiate,
                                                             monomialDegree, quadratureDegree, layoutLeft);
  
  // now try getting the refIntegral a second time; should load from cache:
  auto refIntegral2 = integralCache.getReferenceSpaceIntegral(basis, differentiate, basis, differentiate, basis, differentiate, basis, differentiate,
                                                              monomialDegree, quadratureDegree, layoutLeft);
  
  // check that the two views are at the same location in memory:
  TEST_EQUALITY(refIntegral.data(), refIntegral2.data());

  const int basisCardinality = basis->getCardinality();
  FieldContainer<double> expectedIntegral(basisCardinality,basisCardinality,basisCardinality,basisCardinality);
  for (int i0=0; i0<basisCardinality; i0++)
  {
    for (int i1=0; i1<basisCardinality; i1++)
    {
      for (int i2=0; i2<basisCardinality; i2++)
      {
        for (int i3=0; i3<basisCardinality; i3++)
        {
          // since i0, i1, i2, i3 are all 0 or 1:
          // 4 like basis functions: sum is 0 or 4
          // 3 like basis functions: sum is 1 or 3
          // 2 like basis functions: sum is 2
          const int iSum = i0 + i1 + i2 + i3;
          double integral;
          if ((iSum == 4) || (iSum == 0))
          {
            integral = 2. / 5.;
          }
          else if ((iSum == 1) || (iSum == 3))
          {
            integral = 1. / 10.;
          }
          else // (iSum == 2)
          {
            integral = 1. / 15.;
          }
          expectedIntegral(i0,i1,i2,i3) = integral;
        }
      }
    }
  }
  
  // flatten to 1D view
  expectedIntegral.resize(basisCardinality * basisCardinality * basisCardinality * basisCardinality);
  auto expectedIntegralView = expectedIntegral.checkoutDeviceView();
  
  double tol = 1e-14;
  TEST_COMPARE_FLOATING_ARRAYS_CAMELLIA(refIntegral, expectedIntegralView, tol);
  
  expectedIntegral.checkinDeviceView(false);
}

void testIntegrate4(BasisFamilyPtr basisFamily,
                    Space space1, Space space2, Space space3, Space space4,
                    int p1, int p2, int p3, int p4,
                    bool differentiate1, bool differentiate2, bool differentiate3, bool differentiate4,
                    int monomialDegree, bool layoutLeft,
                    Teuchos::FancyOStream &out, bool &success)
{
  // basic test: can we run to completion for the quadrilinear integral?
  
  CellTopoPtr line = CellTopology::line();
  auto basis1 = basisFamily->getBasis(line, efsForSpace(space1), p1);
  auto basis2 = basisFamily->getBasis(line, efsForSpace(space2), p2);
  auto basis3 = basisFamily->getBasis(line, efsForSpace(space3), p3);
  auto basis4 = basisFamily->getBasis(line, efsForSpace(space4), p4);
  
  Basis1DIntegralCache<double> integralCache;
  
  int quadratureDegree = -1; // lets integralCache select degree
  auto refIntegral = integralCache.getReferenceSpaceIntegral(basis1, differentiate1, basis2, differentiate2, basis3, differentiate3, basis4, differentiate4,
                                                             monomialDegree, quadratureDegree, layoutLeft);
  
  // now try getting the refIntegral a second time; should load from cache:
  auto refIntegral2 = integralCache.getReferenceSpaceIntegral(basis1, differentiate1, basis2, differentiate2, basis3, differentiate3, basis4, differentiate4,
                                                              monomialDegree, quadratureDegree, layoutLeft);
  
  // check that the two views are at the same location in memory:
  TEST_EQUALITY(refIntegral.data(), refIntegral2.data());
}

TEUCHOS_UNIT_TEST( Basis1DIntegralCache, HGRADValueHGRADValue )
{
  bool differentiate1 = false;
  bool differentiate2 = false;
  
  int monomialDegree = 0;
  
  std::vector<bool> layoutLeftChoices {false, true};
  std::vector<BasisFamilyPtr> basisFamilyChoices {BasisFamily::hierarchicalBasisFamily(), BasisFamily::nodalBasisFamily()};
  
  for (const auto &basisFamily : basisFamilyChoices)
  {
    out << "BasisFamily: " << basisFamily->name() << std::endl;
    for (const auto &layoutLeft : layoutLeftChoices)
    {
      if (layoutLeft) out << "LayoutLeft\n";
      else            out << "LayoutRight\n";
      
      testIntegrate(basisFamily, HGRAD, HGRAD, 5, 7, differentiate1, differentiate2, monomialDegree, layoutLeft, out, success);
    }
  }
}

TEUCHOS_UNIT_TEST( Basis1DIntegralCache, HGRADValueHGRADValueTimesXn )
{
  bool differentiate1 = false;
  bool differentiate2 = false;
  
  std::vector<bool> layoutLeftChoices {false, true};
  std::vector<BasisFamilyPtr> basisFamilyChoices {BasisFamily::hierarchicalBasisFamily(), BasisFamily::nodalBasisFamily()};
  
  for (int monomialDegree=1; monomialDegree<10; monomialDegree++)
  {
    out << "integrating against x^" << monomialDegree << std::endl;
    for (const auto &basisFamily : basisFamilyChoices)
    {
      out << "BasisFamily: " << basisFamily->name() << std::endl;
      for (const auto &layoutLeft : layoutLeftChoices)
      {
        if (layoutLeft) out << "LayoutLeft\n";
        else            out << "LayoutRight\n";
        
        testIntegrate(basisFamily, HGRAD, HGRAD, 5, 7, differentiate1, differentiate2, monomialDegree, layoutLeft, out, success);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( Basis1DIntegralCache, HGRADValue3 )
{
  bool differentiate1 = false;
  bool differentiate2 = false;
  bool differentiate3 = false;
  
  int monomialDegree = 0;
  
  std::vector<bool> layoutLeftChoices {false, true};
  std::vector<BasisFamilyPtr> basisFamilyChoices {BasisFamily::hierarchicalBasisFamily(), BasisFamily::nodalBasisFamily()};
  
  for (const auto &basisFamily : basisFamilyChoices)
  {
    out << "BasisFamily: " << basisFamily->name() << std::endl;
    for (const auto &layoutLeft : layoutLeftChoices)
    {
      if (layoutLeft) out << "LayoutLeft\n";
      else            out << "LayoutRight\n";
      
      testIntegrate3(basisFamily, HGRAD, HGRAD, HGRAD, 3, 5, 7, differentiate1, differentiate2, differentiate3, monomialDegree, layoutLeft, out, success);
    }
  }
}

TEUCHOS_UNIT_TEST( Basis1DIntegralCache, HGRADValue3Linear )
{
  std::vector<bool> layoutLeftChoices {false, true};
  std::vector<BasisFamilyPtr> basisFamilyChoices {BasisFamily::hierarchicalBasisFamily(), BasisFamily::nodalBasisFamily()};
  
  for (const auto &basisFamily : basisFamilyChoices)
  {
    out << "BasisFamily: " << basisFamily->name() << std::endl;
    for (const auto &layoutLeft : layoutLeftChoices)
    {
      if (layoutLeft) out << "LayoutLeft\n";
      else            out << "LayoutRight\n";
      
      testIntegrate3LinearBasis(basisFamily, layoutLeft, out, success);
    }
  }
}

TEUCHOS_UNIT_TEST( Basis1DIntegralCache, HGRADValue4 )
{
  bool differentiate1 = false;
  bool differentiate2 = false;
  bool differentiate3 = false;
  bool differentiate4 = false;
  
  int monomialDegree = 0;
  
  std::vector<bool> layoutLeftChoices {false, true};
  std::vector<BasisFamilyPtr> basisFamilyChoices {BasisFamily::hierarchicalBasisFamily(), BasisFamily::nodalBasisFamily()};
  
  for (const auto &basisFamily : basisFamilyChoices)
  {
    out << "BasisFamily: " << basisFamily->name() << std::endl;
    for (const auto &layoutLeft : layoutLeftChoices)
    {
      if (layoutLeft) out << "LayoutLeft\n";
      else            out << "LayoutRight\n";
      
      testIntegrate4(basisFamily, HGRAD, HGRAD, HGRAD, HGRAD, 1, 3, 5, 7, differentiate1, differentiate2, differentiate3, differentiate4, monomialDegree, layoutLeft, out, success);
    }
  }
}

TEUCHOS_UNIT_TEST( Basis1DIntegralCache, HGRADValue4Linear )
{
  std::vector<bool> layoutLeftChoices {false, true};
  std::vector<BasisFamilyPtr> basisFamilyChoices {BasisFamily::hierarchicalBasisFamily(), BasisFamily::nodalBasisFamily()};
  
  for (const auto &basisFamily : basisFamilyChoices)
  {
    out << "BasisFamily: " << basisFamily->name() << std::endl;
    for (const auto &layoutLeft : layoutLeftChoices)
    {
      if (layoutLeft) out << "LayoutLeft\n";
      else            out << "LayoutRight\n";
      
      testIntegrate4LinearBasis(basisFamily, layoutLeft, out, success);
    }
  }
}

TEUCHOS_UNIT_TEST( Basis1DIntegralCache, HGRADDxHGRADValue )
{
  bool differentiate1 = true;
  bool differentiate2 = false;
  
  int monomialDegree = 0;
  
  std::vector<bool> layoutLeftChoices {false, true};
  std::vector<BasisFamilyPtr> basisFamilyChoices {BasisFamily::hierarchicalBasisFamily(), BasisFamily::nodalBasisFamily()};
  
  for (const auto &basisFamily : basisFamilyChoices)
  {
    out << "BasisFamily: " << basisFamily->name() << std::endl;
    for (const auto &layoutLeft : layoutLeftChoices)
    {
      if (layoutLeft) out << "LayoutLeft\n";
      else            out << "LayoutRight\n";
      
      testIntegrate(basisFamily, HGRAD, HGRAD, 5, 7, differentiate1, differentiate2, monomialDegree, layoutLeft, out, success);
    }
  }
}

TEUCHOS_UNIT_TEST( Basis1DIntegralCache, HGRADDxHGRADDx )
{
  bool differentiate1 = true;
  bool differentiate2 = true;
  
  int monomialDegree = 0;
  
  std::vector<bool> layoutLeftChoices {false, true};
  std::vector<BasisFamilyPtr> basisFamilyChoices {BasisFamily::hierarchicalBasisFamily(), BasisFamily::nodalBasisFamily()};
  
  for (const auto &basisFamily : basisFamilyChoices)
  {
    out << "BasisFamily: " << basisFamily->name() << std::endl;
    for (const auto &layoutLeft : layoutLeftChoices)
    {
      if (layoutLeft) out << "LayoutLeft\n";
      else            out << "LayoutRight\n";
      
      testIntegrate(basisFamily, HGRAD, HGRAD, 5, 7, differentiate1, differentiate2, monomialDegree, layoutLeft, out, success);
    }
  }
}

TEUCHOS_UNIT_TEST( Basis1DIntegralCache, HGRADValueL2Value )
{
  bool differentiate1 = false;
  bool differentiate2 = false;
  
  int monomialDegree = 0;
  
  std::vector<bool> layoutLeftChoices {false, true};
  std::vector<BasisFamilyPtr> basisFamilyChoices {BasisFamily::hierarchicalBasisFamily(), BasisFamily::nodalBasisFamily()};
  
  for (const auto &basisFamily : basisFamilyChoices)
  {
    out << "BasisFamily: " << basisFamily->name() << std::endl;
    for (const auto &layoutLeft : layoutLeftChoices)
    {
      if (layoutLeft) out << "LayoutLeft\n";
      else            out << "LayoutRight\n";
      
      testIntegrate(basisFamily, HGRAD, L2, 5, 7, differentiate1, differentiate2, monomialDegree, layoutLeft, out, success);
    }
  }
}

} // namespace
