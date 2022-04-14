#include <stdio.h>

//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  BGKFunctionTests.cpp
//  runTests
//
//  Created by Roberts, Nathan V on 3/24/21.
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia_TensorIndices.h"
#include "CamelliaDataHelpers.h"
#include "CamelliaTestingHelpers.h"
#include "ExtrudedMeshTopology.h"
#include "SolutionMoments.h"
#include "MLCollisionFunction.h"
#include "MaxwellianFunction.h"
#include "SubExtrudedMeshTopology.h"
#include "MeshFactory.h"
#include "TensorBasis.h"

using namespace Camellia;
using namespace std;

namespace
{
  //! moments at a single (cell, point)
  //! maxMomentNumber: use zero for 0th moment, etc.
  DeviceView<double> momentsForMaxwellian(const double driftVelocity, const double thermalVelocitySquared, const double n, const int maxMomentNumber)
  {
  const int numCells = 1;
  const int numPoints = 1;
  const int numVelocityDims = 3;
  const int numMoments = MomentIndices::getNumMomentEntries(maxMomentNumber+1, numVelocityDims);
  DeviceView<double> moments("MLCollisionFunctionTests - momentsForMaxwellian", numCells, numPoints, numMoments);
  
  auto momentsHost = Kokkos::create_mirror(moments);
  
  const int cellOrdinal = 0;
  const int pointOrdinal = 0;
  for(int a=0,abc=0; a<=maxMomentNumber; ++a)
  {
    for(int b=0; b<=maxMomentNumber-a; ++b)
    {
      for(int c=0; c<=maxMomentNumber-a-b; ++c, ++abc)
      {
        momentsHost(cellOrdinal,pointOrdinal,abc) = MaxwellianFunction::moment(n, driftVelocity, 0, 0, thermalVelocitySquared, a, b, c);
      }
    }
  }
    
  Kokkos::deep_copy(moments, momentsHost);
  
  return moments;
}

#ifdef HAVE_MLC

  void testMaxwellianDistributionPrimitiveResponse(const double u0, const double sigmaSquared, const double n,
                                                   std::string modelPath,
                                                   Teuchos::FancyOStream &out, bool &success)
  {
    // compute the moments of a Maxwellian according to formulas computed by Mathematica
    // Then compute the MLC response.  It should be that our idea of the input density, drift velocity, and thermal velocity match those of the output from MLC, modulo certain scalings.
    
    const double n0_scaling = 1e+21;
    const double L0_scaling = 0.1;
    const double u0_scaling = 6949.83;
    const double tol = 1e-13;
    
    Teuchos::RCP<MLCollisionFunction> mlcFxn = Teuchos::rcp(new MLCollisionFunction(n0_scaling, L0_scaling, u0_scaling, Teuchos::null, Teuchos::null, modelPath));
    
    const int maxMomentNumber = MLCollisionFunction::maxMomentNumber(modelPath);
    auto camelliaMoments = momentsForMaxwellian(u0, sigmaSquared, n, maxMomentNumber);
    
    const int numCellPoints = 1;
    auto numMoments = camelliaMoments.extent_int(2);
    mlc::Array mlc_input_conservedMoments({numCellPoints,unsigned(numMoments)});
    mlcFxn->convertMomentsForMLCInput(camelliaMoments, mlc_input_conservedMoments);
    
    auto camelliaMomentsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), camelliaMoments);
    const double expectedZeroMoment = camelliaMomentsHost(0,0,0) * n0_scaling * mlcFxn->getMass() * std::pow(u0_scaling,3.0);
    const double actualZeroMoment   = mlc_input_conservedMoments[0];
    TEST_FLOATING_EQUALITY(expectedZeroMoment, actualZeroMoment, tol);
    
    mlc::Array ml_output_tau({numCellPoints});
    mlc::Array ml_output_g0({numCellPoints});
    mlc::Array ml_output_ux({numCellPoints});
    mlc::Array ml_output_uy({numCellPoints});
    mlc::Array ml_output_uz({numCellPoints});
    mlc::Array ml_output_sigma({numCellPoints});
    
    const int numVelocityDims = 3;
    const unsigned numTotalMoments = MomentIndices::getNumMomentEntries(maxMomentNumber+1, numVelocityDims);
    mlc::Array ml_output_G({numCellPoints,numTotalMoments});
    
    mlcFxn->generateCollisionalResponse(mlc_input_conservedMoments, ml_output_tau, ml_output_g0, ml_output_ux, ml_output_uy, ml_output_uz, ml_output_sigma, ml_output_G);
    
    const double ux_expected    = u0 * u0_scaling;
    const double uy_expected    = 0. * u0_scaling;
    const double uz_expected    = 0. * u0_scaling;
    const double sigma_expected = sqrt(sigmaSquared) * u0_scaling;
    
    TEST_FLOATING_EQUALITY(ux_expected, ml_output_ux[0], tol);
    TEST_FLOATING_EQUALITY(uy_expected, ml_output_uy[0], tol);
    TEST_FLOATING_EQUALITY(uz_expected, ml_output_uz[0], tol);
    TEST_FLOATING_EQUALITY(sigma_expected, ml_output_sigma[0], tol);
    
    // now, try a Maxwellian defined in terms of physical quantities
    const double mass = mlcFxn->getMass();
    auto physicalMoments = momentsForMaxwellian(u0 * u0_scaling, sigmaSquared * u0_scaling * u0_scaling, n * n0_scaling * mass * std::pow(u0_scaling,3.0), maxMomentNumber);
    mlc::Array mlc_input_conservedMomentsPhysicalMaxwellian({numCellPoints,unsigned(numMoments)});
    auto physicalMomentsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), physicalMoments);
    for (int cellPointOrdinal=0; cellPointOrdinal<numCellPoints; cellPointOrdinal++)
    {
      for (int i=0; i<numMoments; i++)
      {
        const int index_1D = cellPointOrdinal * numMoments + i;
        mlc_input_conservedMomentsPhysicalMaxwellian[index_1D] = physicalMomentsHost(cellPointOrdinal,i);
      }
    }
    
    for (int cellPointOrdinal=0; cellPointOrdinal<numCellPoints; cellPointOrdinal++)
    {
      for(int a=0,abc=0; a<=maxMomentNumber; ++a)
      {
        for(int b=0; b<=maxMomentNumber-a; ++b)
        {
          for(int c=0; c<=maxMomentNumber-a-b; ++c, ++abc)
          {
            const int index_1D = cellPointOrdinal * numMoments + abc;
            out << "\n(" << a << "," << b << "," << c  << "):";
            const double inputMomentScaled   = mlc_input_conservedMoments[index_1D];
            const double inputMomentPhysical = mlc_input_conservedMomentsPhysicalMaxwellian[index_1D];
            
            TEST_FLOATING_EQUALITY(inputMomentScaled, inputMomentPhysical, tol);
          }
        }
      }
    }
    
    // want to compare the two G values to verify they agree; can reuse ux, uy, uz, sigma, since we know what those should be (ux_expected, etc.)
    // TODO: create new ml_output_{tau,g0} and compare those with the earlier
    mlc::Array ml_output_G_forPhysicalMaxwellian({numCellPoints,numTotalMoments});
    mlcFxn->generateCollisionalResponse(mlc_input_conservedMomentsPhysicalMaxwellian, ml_output_tau, ml_output_g0, ml_output_ux, ml_output_uy, ml_output_uz, ml_output_sigma, ml_output_G_forPhysicalMaxwellian);
    
    for (int cellPointOrdinal=0; cellPointOrdinal<numCellPoints; cellPointOrdinal++)
    {
      for (int i=0; i<numMoments; i++)
      {
        const int index_1D = cellPointOrdinal * numMoments + i;
        const double & scaledG   = ml_output_G[index_1D];
        const double & physicalG = ml_output_G_forPhysicalMaxwellian[index_1D];
        TEST_FLOATING_EQUALITY(scaledG, physicalG, tol);
      }
      
      TEST_FLOATING_EQUALITY(   ux_expected,    ml_output_ux[cellPointOrdinal], tol);
      TEST_FLOATING_EQUALITY(   uy_expected,    ml_output_uy[cellPointOrdinal], tol);
      TEST_FLOATING_EQUALITY(   uz_expected,    ml_output_uz[cellPointOrdinal], tol);
      TEST_FLOATING_EQUALITY(sigma_expected, ml_output_sigma[cellPointOrdinal], tol);
    }
  }

  TEUCHOS_UNIT_TEST( MLCollisionFunction, MaxwellianDistribution_ZeroDriftSyntheticModel )
  {
    const double u0           = 0.0; // zero drift velocity
    const double sigmaSquared = 0.5;
    const double n            = 1.0;
    
    // TODO: make this not hard-coded!
    std::string modelPath = "/Users/nvrober/camellia/drivers/experimental/mlc/example_model";
    
    testMaxwellianDistributionPrimitiveResponse(u0, sigmaSquared, n, modelPath, out, success);
  }


  TEUCHOS_UNIT_TEST( MLCollisionFunction, MaxwellianDistribution_ZeroDriftArgonModel )
  {
    const double u0           = 0.0; // zero drift velocity
    const double sigmaSquared = 0.5;
    const double n            = 1.0;
    
    // TODO: make this not hard-coded!
    std::string modelPath = "/Users/nvrober/fluid-kinetic/codes/fv_boltzmann/tests/ml_dsmc_shock_tube/model_2l_2w_full";
    
    testMaxwellianDistributionPrimitiveResponse(u0, sigmaSquared, n, modelPath, out, success);
  }

  TEUCHOS_UNIT_TEST( MLCollisionFunction, MaxwellianDistribution_PositiveDrift )
  {
    const double u0           = 1.0; // unit drift velocity
    const double sigmaSquared = 0.5;
    const double n            = 1.0;
    
    // TODO: make this not hard-coded!
    std::string modelPath = "/Users/nvrober/camellia/drivers/experimental/mlc/example_model";
    
    testMaxwellianDistributionPrimitiveResponse(u0, sigmaSquared, n, modelPath, out, success);
  }

#endif // HAVE_MLC guard
} // end anonymous namespace
