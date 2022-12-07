//
//  MLCollisionFunction.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 1/14/22.
//

#include "MLCollisionFunction.h"

#include "BGKFunction.h"
#include "SharedScratchStorage.h"

#ifdef HAVE_MLC
#include "MLC_CollisionModel.hpp"
#include "MLC_MomentConversion.hpp"
#endif

using namespace Camellia;

MLCollisionFunction::MLCollisionFunction(const double n0_scaling, const double L0_scaling, const double u0_scaling,
                                         FunctionPtr f_soln, Teuchos::RCP<SolutionMoments> f_solnMoments, std::string &modelDirPath)
:
_n0(n0_scaling),
_L0(L0_scaling),
_u0(u0_scaling),
TFunction<double>("MLCollisionFunction", 0)
{
#ifndef HAVE_MLC
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Camellia was not configured with mlc_ROOT set; MLCollisionFunction requires mlc.");
#endif
  
  _f_soln         = f_soln;
  _f_solnMoments  = f_solnMoments;
  _modelDirectory = modelDirPath;
  
//  std::cout << "_n0 = " << _n0 << std::endl;
//  std::cout << "_L0 = " << _L0 << std::endl;
//  std::cout << "_u0 = " << _u0 << std::endl;
}

std::vector<double> MLCollisionFunction::computeMomentScaling(const int maxMomentNumber) const
{
#ifdef HAVE_MLC
  const auto mass = _loadedModel->getMass();
#else
  const double mass = 0.0;
#endif
  
  const int numDims = 3; // vx, vy, vz
  const unsigned numTotalMoments = MomentIndices::getNumMomentEntries(maxMomentNumber+1, numDims);
  std::vector<int> mlcMomentIndexToCamelliaMomentIndex(numTotalMoments);
  std::vector<double> momentScaling(numTotalMoments); // scaling from Camellia's moment to the moment with physical units expected by mlc
  for (int a=0; a<=maxMomentNumber; a++)
  {
    for (int b=0; b<=maxMomentNumber-a; b++)
    {
      for (int c=0; c<=maxMomentNumber-a-b; c++)
      {
        int i = MomentIndices::getMomentIndex(a, b, c, maxMomentNumber+1);
        momentScaling[i] = pow(_u0, a+b+c+3) * mass * _n0;
      }
    }
  }
//  for (int i=0; i<numTotalMoments; i++)
//  {
//    std::cout << "momentScaling[" << i << "] = " << momentScaling[i] << std::endl;
//  }
  return momentScaling;
}

void MLCollisionFunction::contributeBGKResponse(DeviceView<double> responseView, DeviceView<double> fValues, DeviceView<double> valuesAllMoments, DeviceView<double> tauView,
                                                DeviceView<double> points_vx, DeviceView<double> points_vy, DeviceView<double> points_vz) const
{
#ifdef HAVE_MLC
  const int numCells       = responseView.extent_int(0);
  const int numPhasePoints = responseView.extent_int(1);
  
  const int numConfigurationPoints = tauView.extent_int(1); // (C,P) shape
  
  const int numVelocityDims = _f_solnMoments->numVelocityDims();
  
  TEUCHOS_TEST_FOR_EXCEPTION(responseView.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  auto rangePolicy = FieldContainer<double>::getMDRangePolicy2(tauView);
  Kokkos::fence();
  // TODO: either switch this to use Camellia::MomentIndices, or guard this method with #ifdef HAVE_MLC.  I'd rather do the MomentIndices thing.
  mlc::MomentIndexer standardMomentIndexer; // as opposed to the ones used by the model, which eliminates some moments
  standardMomentIndexer.setupForMoments(_f_solnMoments->maxMomentNumber() + 1);
  const int zerothMomentIndex = standardMomentIndexer(0,0,0);
  Kokkos::Array<int,3> firstMomentIndices {standardMomentIndexer(1,0,0),standardMomentIndexer(0,1,0),standardMomentIndexer(0,0,1)};
  Kokkos::Array<int,3> secondMomentIndices {standardMomentIndexer(2,0,0),standardMomentIndexer(0,2,0),standardMomentIndexer(0,0,2)};
  const int numPoints_vx = points_vx.extent_int(1); // Each PhysicalCubaturePoint tensorial component has dimensions (C,P,D)
  const int numPoints_vy = points_vy.extent_int(1); // Each PhysicalCubaturePoint tensorial component has dimensions (C,P,D)
  const int numPoints_vz = points_vz.extent_int(1); // Each PhysicalCubaturePoint tensorial component has dimensions (C,P,D)
  
//  {
//    // DEBUGGING
//    const double n = valuesAllMoments(0,1,standardMomentIndexer(0,0,0));
//    std::cout << "Camellia moment (0,0,0) at (0,1): " << n << std::endl;
//    std::cout << "Camellia moment (1,0,0) at (0,1): " << valuesAllMoments(0,1,firstMomentIndices[0]) << std::endl;
//    std::cout << "Camellia moment (0,1,0) at (0,1): " << valuesAllMoments(0,1,firstMomentIndices[1]) << std::endl;
//    std::cout << "Camellia moment (0,0,1) at (0,1): " << valuesAllMoments(0,1,firstMomentIndices[2]) << std::endl;
//
//    double sigmaSquared = 0.0;
//    for (int d=0; d<numVelocityDims; d++)
//    {
//      const double firstMoment  = valuesAllMoments(0,1,firstMomentIndices[d]);
//      const double secondMoment = valuesAllMoments(0,1,secondMomentIndices[d]);
//
//      const double u_d = firstMoment / n;
//      sigmaSquared += secondMoment / n  - u_d * u_d;
//    }
//    sigmaSquared *= 1.0 / numVelocityDims;
//    std::cout << "Camellia sigma squared at (0,1): " << sigmaSquared << std::endl;
//  }
  const double tauScaling = _u0 / _L0; // multiply the mlc output by this to get our non-dimensional tau
  Kokkos::parallel_for("MLCollisionFunction: compute BGK contribution", rangePolicy,
  KOKKOS_LAMBDA (const int cellOrdinal, const int configurationPointOrdinal) {
    const double n = valuesAllMoments(cellOrdinal,configurationPointOrdinal,standardMomentIndexer(0,0,0));
    
    Kokkos::Array<double,3> u;
    
    double sigmaSquared = 0.0;
    for (int d=0; d<numVelocityDims; d++)
    {
      const double firstMoment  = valuesAllMoments(cellOrdinal,configurationPointOrdinal,firstMomentIndices[d]);
      const double secondMoment = valuesAllMoments(cellOrdinal,configurationPointOrdinal,secondMomentIndices[d]);
      
      const double u_d = firstMoment / n;
      sigmaSquared += secondMoment / n  - u_d * u_d;
      
      u[d] = u_d;
    }
    sigmaSquared *= (1./numVelocityDims);
    const double weight = n / pow(2 * M_PI * sigmaSquared, numVelocityDims / 2.);
    
    const double tau = tauScaling * tauView(cellOrdinal,configurationPointOrdinal);
    
    const int numPointComponents = 4; // 1 for config space ordinal, 3 for velocity ordinals
    Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> pointComponentOrdinals;
    Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> pointComponentCardinalities;
    pointComponentOrdinals[0] = configurationPointOrdinal;
    pointComponentCardinalities[0] = numConfigurationPoints;
    pointComponentCardinalities[1] = numPoints_vx;
    pointComponentCardinalities[2] = numPoints_vy;
    pointComponentCardinalities[3] = numPoints_vz;
    
    int phasePointOrdinal;
    Kokkos::Array<double,3> v;
    for (int ivx=0; ivx<numPoints_vx; ivx++)
    {
      v[0] = points_vx(cellOrdinal,ivx,0);
      pointComponentOrdinals[1] = ivx;
      for (int ivy=0; ivy<numPoints_vy; ivy++)
      {
        v[1] = points_vy(cellOrdinal,ivy,0);
        pointComponentOrdinals[2] = ivy;
        for (int ivz=0; ivz<numPoints_vz; ivz++)
        {
          v[2] = points_vz(cellOrdinal,ivz,0);
          pointComponentOrdinals[3] = ivz;
          
          // set phasePointOrdinal:
          TensorIndices::setTensorOrdinal(phasePointOrdinal, pointComponentOrdinals, pointComponentCardinalities, numPointComponents);
          
          double v_minus_u_squared = (v[0]-u[0])*(v[0]-u[0]) + (v[1]-u[1])*(v[1]-u[1]) + (v[2]-u[2])*(v[2]-u[2]);
          const double exponent = - 1. / (2. * sigmaSquared) * v_minus_u_squared;
          
          const double f_M = weight * std::exp(exponent);
          const double f = fValues(cellOrdinal,phasePointOrdinal);
          responseView(cellOrdinal,phasePointOrdinal) += (-1.0 / tau) * (f - f_M);
          
//          {
//            // DEBUGGING
//            auto value = responseView(cellOrdinal,phasePointOrdinal);
//            if (abs(value) > 0.1)
//            {
//              std::cout << "BGK collisional response value(" << cellOrdinal << ", " << ivx << "," << ivy << "," << ivz << ") = " << responseView(cellOrdinal,phasePointOrdinal) << std::endl;
//              std::cout << "f_M(" << v[0] << "," << v[1] << "," << v[2] << ") = " << f_M << std::endl;
//              std::cout << "  f(" << v[0] << "," << v[1] << "," << v[2] << ") = " << f   << std::endl;
//            }
//          }
        }
      }
    }
  });
  
  {
    // DEBUGGING
//    int numPhasePoints = valuesToAssign.extent_int(1);
//    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
//    {
//      for (int pointOrdinal=0; pointOrdinal<numPhasePoints; pointOrdinal++)
//      {
//        std::cout << "BGK collisional response value(" << cellOrdinal << ", " << pointOrdinal << ") = " << valuesToAssign(cellOrdinal,pointOrdinal) << std::endl;
//      }
//    }
  }
#endif
}

void MLCollisionFunction::contributeDifferentialResponseFromBGK(DeviceView<double> responseView, DeviceView<double> g0View,
                                                                DeviceView<double> uxView, DeviceView<double> uyView, DeviceView<double> uzView,
                                                                DeviceView<double> sigmaView, DeviceView<double> GView,
                                                                DeviceView<double> points_vx, DeviceView<double> points_vy, DeviceView<double> points_vz) const
{
  const int maxMomentNumber = _f_solnMoments->maxMomentNumber();
  
  const int numConfigurationPoints = g0View.extent_int(1); // (C,P) shape
  
  const int numPoints_vx = points_vx.extent_int(1); // Each PhysicalCubaturePoint tensorial component has dimensions (C,P,D)
  const int numPoints_vy = points_vy.extent_int(1); // Each PhysicalCubaturePoint tensorial component has dimensions (C,P,D)
  const int numPoints_vz = points_vz.extent_int(1); // Each PhysicalCubaturePoint tensorial component has dimensions (C,P,D)
  
  const double g0_scaling = 1.0 / _n0;
  const double u_scaling  = 1.0 / _u0;
//  {
//    // DEBUGGING
//
//    const double g0    = g0View(0,1)    * g0_scaling;
//    const double ux    = uxView(0,1)    * u_scaling;
//    const double uy    = uyView(0,1)    * u_scaling;
//    const double uz    = uzView(0,1)    * u_scaling;
//    const double sigma = sigmaView(0,1) * u_scaling;
//
//    std::cout << "Camellia g0 (converted from MLC) at (0,1): " << g0 << std::endl;
//    std::cout << "Camellia ux (converted from MLC) at (0,1): " << ux << std::endl;
//    std::cout << "Camellia uy (converted from MLC) at (0,1): " << uy << std::endl;
//    std::cout << "Camellia uz (converted from MLC) at (0,1): " << uz << std::endl;
//    std::cout << "Camellia sigma squared (converted from MLC) at (0,1): " << sigma * sigma << std::endl;
//  }
  
  
  auto rangePolicy = FieldContainer<double>::getMDRangePolicy2(sigmaView);
  Kokkos::parallel_for("MLCollisionFunction: compute ML collisional response contribution", rangePolicy,
  KOKKOS_LAMBDA (const int cellOrdinal, const int configurationPointOrdinal) {
    const double g0    = g0View(cellOrdinal,configurationPointOrdinal)    * g0_scaling;
    const double ux    = uxView(cellOrdinal,configurationPointOrdinal)    * u_scaling;
    const double uy    = uyView(cellOrdinal,configurationPointOrdinal)    * u_scaling;
    const double uz    = uzView(cellOrdinal,configurationPointOrdinal)    * u_scaling;
    const double sigma = sigmaView(cellOrdinal,configurationPointOrdinal) * u_scaling;

    const double w_scale = 1. / M_SQRT2 / sigma;

    const int numPointComponents = 4; // 1 for config space ordinal, 3 for velocity ordinals
    Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> pointComponentOrdinals;
    Kokkos::Array<int,Intrepid2::Parameters::MaxTensorComponents> pointComponentCardinalities;
    pointComponentOrdinals[0] = configurationPointOrdinal;
    pointComponentCardinalities[0] = numConfigurationPoints;
    pointComponentCardinalities[1] = numPoints_vx;
    pointComponentCardinalities[2] = numPoints_vy;
    pointComponentCardinalities[3] = numPoints_vz;
    
    int phasePointOrdinal;
    for (int ivx=0; ivx<numPoints_vx; ivx++)
    {
      const double &vx = points_vx(cellOrdinal,ivx,0);
      const double wx = w_scale * (vx - ux);
      pointComponentOrdinals[1] = ivx;
      for (int ivy=0; ivy<numPoints_vy; ivy++)
      {
        const double &vy = points_vy(cellOrdinal,ivy,0);
        const double wy = w_scale * (vy - uy);
        pointComponentOrdinals[2] = ivy;
        for (int ivz=0; ivz<numPoints_vz; ivz++)
        {
          const double &vz = points_vz(cellOrdinal,ivz,0);
          const double wz = w_scale * (vz - uz);
          pointComponentOrdinals[3] = ivz;
          
          const double mult = g0 * std::exp(-(wx*wx+wy*wy+wz*wz));
          
          double rhsijkl = 0;
          double mult_wxa = mult;
          for(unsigned int a=0,abc=0; a<=maxMomentNumber; ++a){
            double mult_wxa_wyb = mult_wxa;
            for(unsigned int b=0; b<=maxMomentNumber-a; ++b){
              double mult_wxa_wyb_wzc = mult_wxa_wyb;
              for(unsigned int c=0; c<=maxMomentNumber-a-b; ++c, ++abc){
                rhsijkl += GView(cellOrdinal,configurationPointOrdinal,abc) * mult_wxa_wyb_wzc;
                mult_wxa_wyb_wzc *= wz;
              }
              mult_wxa_wyb *= wy;
            }
            mult_wxa *= wx;
          }

          // set phasePointOrdinal:
          TensorIndices::setTensorOrdinal(phasePointOrdinal, pointComponentOrdinals, pointComponentCardinalities, numPointComponents);
          responseView(cellOrdinal,phasePointOrdinal) += rhsijkl;
        }
      }
    }
  });
}

#ifdef HAVE_MLC
void MLCollisionFunction::convertMomentsForMLCInput(const DeviceView<double> &valuesAllMoments, mlc::Array &mlc_input_conservedMoments)
{
  auto valuesAllMomentsHost = Kokkos::create_mirror_view_and_copy(HostMemorySpace(), valuesAllMoments);
  
  const int numCells               = valuesAllMoments.extent_int(0); // shape (C,P,M)
  const int numConfigurationPoints = valuesAllMoments.extent_int(1);
  const int numTotalMoments        = valuesAllMoments.extent_int(2);
  
  if (_loadedModel == Teuchos::null) ensureModelLoaded(1, 1);
  const int maxMomentNumber = _loadedModel->getInputMomentIndexer().getNumMoments() - 1;
  
  std::vector<double> momentScaling = computeMomentScaling(maxMomentNumber);
  
  // Clean up valuesAllMomentsHost, by rounding near-zero values to zero
  // assumption is that moments are approximately unit-scale
  const double zeroTol = 1e-8;
  for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
  {
    for (int pointOrdinal=0; pointOrdinal<numConfigurationPoints; pointOrdinal++)
    {
      for (int i=0; i<numTotalMoments; i++)
      {
        if (abs(valuesAllMomentsHost(cellOrdinal,pointOrdinal,i)) < zeroTol )
        {
          valuesAllMomentsHost(cellOrdinal,pointOrdinal,i) = 0.0;
        }
      }
    }
  }
      
  int cellPointOrdinal = 0;
  for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
  {
    for (int pointOrdinal=0; pointOrdinal<numConfigurationPoints; pointOrdinal++, cellPointOrdinal++)
    {
      for (int i=0; i<numTotalMoments; i++)
      {
        int index1D = cellPointOrdinal*numTotalMoments + i;
        mlc_input_conservedMoments[index1D] = momentScaling[i] * valuesAllMomentsHost(cellOrdinal,pointOrdinal,i); // this is where we rely on moment indexing being identical between mlc and SolutionMoments
        
//          std::cout << "ml_input_conservedMoments(" << cellOrdinal << ", " << pointOrdinal << ", " << i << ") = " << ml_input_conservedMoments[index1D] << std::endl;
      }
    }
  }
  
//  {
//    // DEBUGGING
//
//
//    for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
//    {
//      for (int pointOrdinal=0; pointOrdinal<numConfigurationPoints; pointOrdinal++)
//      {
//        for (int i=0; i<numTotalMoments; i++)
//        {
//          int index1D = cellPointOrdinal*numTotalMoments + i;
//          for (int k = 1; k < maxMomentNumber; k++)
//          {
//            int ix = MomentIndices::getMomentIndex(k, 0, 0, maxMomentNumber+1);
//            int iy = MomentIndices::getMomentIndex(0, k, 0, maxMomentNumber+1);
//            int iz = MomentIndices::getMomentIndex(0, 0, k, maxMomentNumber+1);
//
//            double maxDiff = 0;
//            maxDiff = std::max(maxDiff, abs(valuesAllMomentsHost(cellOrdinal,pointOrdinal,ix) - valuesAllMomentsHost(cellOrdinal,pointOrdinal,iy)));
//            maxDiff = std::max(maxDiff, abs(valuesAllMomentsHost(cellOrdinal,pointOrdinal,iy) - valuesAllMomentsHost(cellOrdinal,pointOrdinal,iz)));
//            maxDiff = std::max(maxDiff, abs(valuesAllMomentsHost(cellOrdinal,pointOrdinal,ix) - valuesAllMomentsHost(cellOrdinal,pointOrdinal,iz)));
//            if (maxDiff > 1e-8)
//            {
//              std::cout << "Camellia moment (" << k << ",0,0) for cell " << cellOrdinal << ", point " << pointOrdinal << ": " << valuesAllMomentsHost(cellOrdinal,pointOrdinal,ix) << std::endl;
//              std::cout << "Camellia moment (0," << k << ",0) for cell " << cellOrdinal << ", point " << pointOrdinal << ": " << valuesAllMomentsHost(cellOrdinal,pointOrdinal,iy) << std::endl;
//              std::cout << "Camellia moment (0,0," << k << ") for cell " << cellOrdinal << ", point " << pointOrdinal << ": " << valuesAllMomentsHost(cellOrdinal,pointOrdinal,iz) << std::endl;
//            }
//          }
//        }
//      }
//    }
//
//    int cellPointOrdinal = 0;
//    for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
//    {
//      for (int pointOrdinal=0; pointOrdinal<numConfigurationPoints; pointOrdinal++, cellPointOrdinal++)
//      {
//        for (int i=0; i<numTotalMoments; i++)
//        {
//          int index1D = cellPointOrdinal*numTotalMoments + i;
//          for (int k = 1; k < maxMomentNumber; k++)
//          {
//            int ix = MomentIndices::getMomentIndex(k, 0, 0, maxMomentNumber+1);
//            int iy = MomentIndices::getMomentIndex(0, k, 0, maxMomentNumber+1);
//            int iz = MomentIndices::getMomentIndex(0, 0, k, maxMomentNumber+1);
//
//            double maxDiff = 0;
//            int index1D_x = cellPointOrdinal*numTotalMoments + ix;
//            int index1D_y = cellPointOrdinal*numTotalMoments + iy;
//            int index1D_z = cellPointOrdinal*numTotalMoments + iz;
//            maxDiff = std::max(maxDiff, abs(mlc_input_conservedMoments[ix] - mlc_input_conservedMoments[iy]));
//            maxDiff = std::max(maxDiff, abs(mlc_input_conservedMoments[iy] - mlc_input_conservedMoments[iz]));
//            maxDiff = std::max(maxDiff, abs(mlc_input_conservedMoments[ix] - mlc_input_conservedMoments[iz]));
//            if (maxDiff > 1e-8 * abs(mlc_input_conservedMoments[ix]))
//            {
//              std::cout << "MLC input moment (" << k << ",0,0) for cell " << cellOrdinal << ", point " << pointOrdinal << ": " << mlc_input_conservedMoments[ix] << std::endl;
//              std::cout << "MLC input moment (0," << k << ",0) for cell " << cellOrdinal << ", point " << pointOrdinal << ": " << mlc_input_conservedMoments[iy] << std::endl;
//              std::cout << "MLC input moment (0,0," << k << ") for cell " << cellOrdinal << ", point " << pointOrdinal << ": " << mlc_input_conservedMoments[iz] << std::endl;
//            }
//          }
//        }
//      }
//    }
//  }
}
#endif

void MLCollisionFunction::ensureModelLoaded(const int &numCells, const int &numConfigurationPoints)
{
#ifdef HAVE_MLC
  using namespace mlc;
  
  // Create the mlc model:
  const int numCellPoints = numConfigurationPoints * numCells; // mlc expects a single points dimension (no cell dimension), so we need to flatten…
  
  if ((numCellPoints != _numPointsForLoadedModel) || _loadedModel == Teuchos::null)
  {
    _loadedModel = Teuchos::rcp(new CollisionModel(numCellPoints, _modelDirectory));
    _numPointsForLoadedModel = numCellPoints;
    
    std::cout << "model loaded with storage for " << numCellPoints << " points.\n";
    std::cout << "boltzmann constant = " << _loadedModel->getBoltzmannConstant() << std::endl;
    std::cout << "mass = " << _loadedModel->getMass() << std::endl;
  }
#endif
}

#ifdef HAVE_MLC
void MLCollisionFunction::generateCollisionalResponse(const mlc::Array &ml_input_conservedMoments,
                                                      mlc::Array &ml_output_tau, mlc::Array &ml_output_g0,
                                                      mlc::Array &ml_output_ux, mlc::Array &ml_output_uy, mlc::Array &ml_output_uz,
                                                      mlc::Array &ml_output_sigma, mlc::Array &ml_output_G) const
{
  const bool evaluatePrimitive = true; // Note: Sean Miller says we should always use primitive.  The alternative is expensive (and from our POV, also harder to use).
  _loadedModel->generateCollisionalResponse(ml_input_conservedMoments,
                                            ml_output_tau, ml_output_g0, ml_output_ux, ml_output_uy, ml_output_uz, ml_output_sigma, ml_output_G,
                                            evaluatePrimitive);
}

double MLCollisionFunction::getMass()
{
  if (_loadedModel == Teuchos::null) ensureModelLoaded(1, 1);
  return _loadedModel->getMass();
}

int MLCollisionFunction::maxMomentNumber(std::string &modelDirPath)
{
  // because mlc CollisionModel requires a numPoints argument at construction, we can't reuse model objects in a straightforward way.  We might want to revise mlc to make it more flexible in this regard.
  using namespace mlc;
  const unsigned numPoints = 1;
  CollisionModel model(numPoints, modelDirPath);
  
  return model.getInputMomentIndexer().getNumMoments() - 1;
}

void MLCollisionFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("MLCollisionFunction values()");
  
  auto physicalPoints = basisCache->getPhysicalCubaturePoints();
  
  // we require that points have 3 velocity space dimensions separately stored
  // points must therefore have at least 4 tensor components
  TEUCHOS_TEST_FOR_EXCEPTION(physicalPoints.numTensorComponents() < 4, std::invalid_argument, "basisCache's points container must contain at least 4 tensorial components");
  
  using PointsView = Intrepid2::ScalarView<double,Camellia::DeviceType>;
  
  using TensorPoints = Intrepid2::TensorPoints<double,DeviceType>;
  
  int numConfigurationTensorComponents = physicalPoints.numTensorComponents() - 3;
  std::vector<PointsView> configurationSpaceRefPointsVector(numConfigurationTensorComponents);
  
  TensorPoints refPoints = basisCache->getRefCellPoints();
  
  std::vector<int> configTensorialComponentOrdinals(numConfigurationTensorComponents);
  for (int d=0; d<numConfigurationTensorComponents; d++)
  {
    configurationSpaceRefPointsVector[d] = refPoints.getTensorComponent(d);
  }
  TensorPoints configurationSpaceRefPoints(configurationSpaceRefPointsVector);
  
  auto points_vx = physicalPoints.getTensorComponent(numConfigurationTensorComponents);   // shape (C,Pvx,D)
  auto points_vy = physicalPoints.getTensorComponent(numConfigurationTensorComponents+1); // shape (C,Pvy,D)
  auto points_vz = physicalPoints.getTensorComponent(numConfigurationTensorComponents+2); // shape (C,Pvz,D)
  
  // note that we might easily hit the same configuration-space cells and points several times here; we *could*
  // track this and eliminate the redundancy.  If this method turns out to be a bottleneck, that could be worth doing.
  const auto & phaseSpaceCellIDs = basisCache->cellIDs();
  std::vector<GlobalIndexType> configurationSpaceCellIDs(phaseSpaceCellIDs.size());
  
  if (phaseSpaceCellIDs.size() == 0) return; // nothing to do…
  
  auto phaseSpaceMeshTopology = dynamic_cast<const ExtrudedMeshTopology*>(basisCache->mesh()->getTopology()->baseMeshTopology());
  
  for (int cellOrdinal=0; cellOrdinal<phaseSpaceCellIDs.size(); cellOrdinal++)
  {
    const GlobalIndexType &phaseSpaceCellID = phaseSpaceCellIDs[cellOrdinal];
    const GlobalIndexType configurationCellID = phaseSpaceMeshTopology->physicalCellIDForPhaseSpaceCellID(phaseSpaceCellID);
    configurationSpaceCellIDs[cellOrdinal] = configurationCellID;
  }
  auto configurationMesh = _f_solnMoments->mesh();
  auto configurationBasisCache = BasisCache::basisCacheForCellType(configurationMesh, configurationMesh->getElementType(configurationSpaceCellIDs[0]));
  
  auto physicalCellNodes = configurationMesh->physicalCellNodes(configurationSpaceCellIDs);
  configurationBasisCache->setRefCellPoints(configurationSpaceRefPoints);
  configurationBasisCache->setPhysicalCellNodes(physicalCellNodes, configurationSpaceCellIDs, false); // false: don't create side caches
  
  const int numConfigurationPoints = configurationSpaceRefPoints.extent_int(0); // (P,D)
  const int numCells = configurationSpaceCellIDs.size();
  
  // load the model if it isn't already loaded with the right number of points
  ensureModelLoaded(numCells, numConfigurationPoints);
  
  typename FieldContainer<double>::DeviceView valuesToAssign;
  
  if (modType == ASSIGN)
  {
    valuesToAssign = values;
  }
  else
  {
    valuesToAssign = basisCache->checkoutScratchStorage<double>(values);
  }
  
  const auto & inputMomentIndexer  = _loadedModel->getInputMomentIndexer();
  const unsigned maxMomentNumber = inputMomentIndexer.getNumMoments()-1;

  int numMomentsEntries = _f_solnMoments->numMomentEntries();
  
  const int numVelocityDims = _f_solnMoments->numVelocityDims();
  const unsigned numTotalMoments = MomentIndices::getNumMomentEntries(maxMomentNumber+1, numVelocityDims);
  
  // prepare model inputs
  auto valuesAllMoments = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints,numMomentsEntries},"MLCollisionFunction::values(): valuesAllMoments");
  _f_solnMoments->valuesForAllMoments(valuesAllMoments, configurationBasisCache);
  
  const unsigned numCellPoints = numConfigurationPoints * numCells; // mlc expects a single points dimension (no cell dimension), so we need to flatten…
  mlc::Array ml_input_conservedMoments({numCellPoints,numTotalMoments});
  
  convertMomentsForMLCInput(valuesAllMoments, ml_input_conservedMoments);
  
  
//  {
//    // DEBUGGING
//
//    auto cellIDs = basisCache->cellIDs();
//
//    if (cellIDs[0] == 0)
//    {
//      const double camelliaZerothMoment = valuesAllMoments(0,0,0);
//      const double scaledZerothMoment   = ml_input_conservedMoments[0];
//      const double scalingFactor        = ml_input_conservedMoments[0] / valuesAllMoments(0,0,0);
//
//      const double mass = _loadedModel->getMass();
//      const double expectedScalingFactor = mass * _n0 * _u0 * _u0 * _u0;
//      std::cout << "Camellia zeroth moment at (0,1): " << valuesAllMoments(0,1,0) << std::endl;
//      int index1D = 1*numTotalMoments + 0; // int index1D = cellPointOrdinal*numTotalMoments + i;
//      std::cout << "Scaled zeroth moment at (0,1):   " << ml_input_conservedMoments[index1D] << std::endl;
//      std::cout << "expected scaling factor:         " << expectedScalingFactor << std::endl;
//      std::cout << "actual scaling factor:           " << scalingFactor << std::endl;
//    }
//  }
  
  // declare/allocate the Kokkos views which we'll populate with mlc outputs
  auto    tauView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints},                  "MLCollisionFunction::values(): tauView");
  auto     g0View = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints},                  "MLCollisionFunction::values(): g0View");
  auto     uxView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints},                  "MLCollisionFunction::values(): uxView");
  auto     uyView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints},                  "MLCollisionFunction::values(): uyView");
  auto     uzView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints},                  "MLCollisionFunction::values(): uzView");
  auto  sigmaView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints},                  "MLCollisionFunction::values(): sigmaView");
  auto      GView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints,numMomentsEntries},"MLCollisionFunction::values():   GView");
    
  // Evaluate the mlc model
  {
    using namespace mlc;
        
    // We require the max moment numbers to match.  We could relax this, requiring only that SolutionMoments have at least as many moments as the model, with some modest revision to the below
    // logic, and perhaps the introduction of a MomentIndexer-like facility for SolutionMoments.
    TEUCHOS_TEST_FOR_EXCEPTION(maxMomentNumber != _f_solnMoments->maxMomentNumber(), std::invalid_argument, "max moment number for model must match max moment number in SolutionMoments");
    
    // The model defines these coefficients (important since the model internally converts things to number density and temperature)
    const double kB = _loadedModel->getBoltzmannConstant();
    const double mass = _loadedModel->getMass();
            
    Array ml_output_tau({numCellPoints});
    Array ml_output_g0({numCellPoints});
    Array ml_output_ux({numCellPoints});
    Array ml_output_uy({numCellPoints});
    Array ml_output_uz({numCellPoints});
    Array ml_output_sigma({numCellPoints});
    Array ml_output_G({numCellPoints,numTotalMoments});
    
    generateCollisionalResponse(ml_input_conservedMoments, ml_output_tau, ml_output_g0, ml_output_ux, ml_output_uy, ml_output_uz, ml_output_sigma, ml_output_G);
  
    // get the mlc outputs into Kokkos Views
    auto    tauHost = Kokkos::create_mirror_view(HostMemorySpace(),   tauView);
    auto     g0Host = Kokkos::create_mirror_view(HostMemorySpace(),    g0View);
    auto     uxHost = Kokkos::create_mirror_view(HostMemorySpace(),    uxView);
    auto     uyHost = Kokkos::create_mirror_view(HostMemorySpace(),    uyView);
    auto     uzHost = Kokkos::create_mirror_view(HostMemorySpace(),    uzView);
    auto  sigmaHost = Kokkos::create_mirror_view(HostMemorySpace(), sigmaView);
    auto      GHost = Kokkos::create_mirror_view(HostMemorySpace(),     GView);
    
    int cellPointOrdinal = 0;
    for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<numConfigurationPoints; pointOrdinal++, cellPointOrdinal++)
      {
          tauHost(cellOrdinal,pointOrdinal) = ml_output_tau  [cellPointOrdinal];
           g0Host(cellOrdinal,pointOrdinal) = ml_output_g0   [cellPointOrdinal];
           uxHost(cellOrdinal,pointOrdinal) = ml_output_ux   [cellPointOrdinal];
           uyHost(cellOrdinal,pointOrdinal) = ml_output_uy   [cellPointOrdinal];
           uzHost(cellOrdinal,pointOrdinal) = ml_output_uz   [cellPointOrdinal];
        sigmaHost(cellOrdinal,pointOrdinal) = ml_output_sigma[cellPointOrdinal];
        
        for (int i=0; i<numTotalMoments; i++)
        {
          int index1D = cellPointOrdinal*numTotalMoments + i;
          GHost(cellOrdinal,pointOrdinal,i) = ml_output_G[index1D];
        }
//        {
//          // DEBUGGING
//          for(unsigned int a=0,abc=0; a<=maxMomentNumber; ++a){
//            for(unsigned int b=0; b<=maxMomentNumber-a; ++b){
//              for(unsigned int c=0; c<=maxMomentNumber-a-b; ++c, ++abc){
//                // check for a,b,c odd that the value of G is small
////                if ((a % 2 == 1) || (b % 2 == 1) || (c % 2 == 1))
//                {
//                  double g_coefficient = GHost(cellOrdinal,pointOrdinal,abc);
//                  if ( std::abs(g_coefficient) > 1e-8)
//                  {
//                    std::cout << "g(" << cellOrdinal << "," << pointOrdinal << ")(" << a << "," << b << "," << c << ") = " << g_coefficient << std::endl;
//                  }
//                }
//              }
//            }
//          }
//        }
//        {
//          // DEBUGGING
//          std::cout << "  tau(" << cellOrdinal << "," << pointOrdinal << ") = " <<   tauHost(cellOrdinal,pointOrdinal) << std::endl;
//          std::cout << "   g0(" << cellOrdinal << "," << pointOrdinal << ") = " <<    g0Host(cellOrdinal,pointOrdinal) << std::endl;
//          std::cout << "   ux(" << cellOrdinal << "," << pointOrdinal << ") = " <<    uxHost(cellOrdinal,pointOrdinal) << std::endl;
//          std::cout << "   uy(" << cellOrdinal << "," << pointOrdinal << ") = " <<    uyHost(cellOrdinal,pointOrdinal) << std::endl;
//          std::cout << "   uz(" << cellOrdinal << "," << pointOrdinal << ") = " <<    uzHost(cellOrdinal,pointOrdinal) << std::endl;
//          std::cout << "sigma(" << cellOrdinal << "," << pointOrdinal << ") = " << sigmaHost(cellOrdinal,pointOrdinal) << std::endl;
//          for (int i=0; i<numTotalMoments; i++)
//          {
//            std::cout << "GView(" << cellOrdinal << "," << pointOrdinal << "," << i << ") = " << GHost(cellOrdinal,pointOrdinal,i) << std::endl;
//          }
//        }
      }
    }
    Kokkos::deep_copy(  tauView,   tauHost);
    Kokkos::deep_copy(   g0View,    g0Host);
    Kokkos::deep_copy(   uxView,    uxHost);
    Kokkos::deep_copy(   uyView,    uyHost);
    Kokkos::deep_copy(   uzView,    uzHost);
    Kokkos::deep_copy(sigmaView, sigmaHost);
    Kokkos::deep_copy(    GView,     GHost);
  }
  
  auto fValues = basisCache->checkoutScratchStorage<double>(values);
  _f_soln->values(fValues, ASSIGN, basisCache);
  
  {
    auto cellIDs = basisCache->cellIDs();
    
    if (cellIDs[0] == 0)
    {
      // DEBUGGING
      
      // TODO: either switch this to use Camellia::MomentIndices, or guard this method with #ifdef HAVE_MLC.  I'd rather do the MomentIndices thing.
      mlc::MomentIndexer standardMomentIndexer; // as opposed to the ones used by the model, which eliminates some moments
      standardMomentIndexer.setupForMoments(_f_solnMoments->maxMomentNumber() + 1);
      const int zerothMomentIndex = standardMomentIndexer(0,0,0);
      Kokkos::Array<int,3> firstMomentIndices {standardMomentIndexer(1,0,0),standardMomentIndexer(0,1,0),standardMomentIndexer(0,0,1)};
      Kokkos::Array<int,3> secondMomentIndices {standardMomentIndexer(2,0,0),standardMomentIndexer(0,2,0),standardMomentIndexer(0,0,2)};
      
      const double n = valuesAllMoments(0,1,standardMomentIndexer(0,0,0));
      std::cout << "Camellia moment (0,0,0) at (0,1): " << n << std::endl;
      std::cout << "Camellia moment (1,0,0) at (0,1): " << valuesAllMoments(0,1,firstMomentIndices[0]) << std::endl;
      std::cout << "Camellia moment (0,1,0) at (0,1): " << valuesAllMoments(0,1,firstMomentIndices[1]) << std::endl;
      std::cout << "Camellia moment (0,0,1) at (0,1): " << valuesAllMoments(0,1,firstMomentIndices[2]) << std::endl;
      
      double sigmaSquared = 0.0;
      for (int d=0; d<numVelocityDims; d++)
      {
        const double firstMoment  = valuesAllMoments(0,1,firstMomentIndices[d]);
        const double secondMoment = valuesAllMoments(0,1,secondMomentIndices[d]);
        
        const double u_d = firstMoment / n;
        sigmaSquared += secondMoment / n  - u_d * u_d;
      }
      sigmaSquared *= 1.0 / numVelocityDims;
      std::cout << "Camellia sigma squared at (0,1): " << sigmaSquared << std::endl;
    }
  }
  // initialize valuesToAssign with 0, since contribute…() methods sum into.
  Kokkos::deep_copy(valuesToAssign, 0.0);
  contributeBGKResponse(valuesToAssign, fValues, valuesAllMoments, tauView, points_vx, points_vy, points_vz);
    
  // the below is adapted from evaluateMLDSMC_Primitive() in fvBoltzmann's SolverModule_MLDSMCCollisions.cpp.
  
  const bool includeMLPart = true; // can turn this off when debugging
  if (includeMLPart)
  {
//    {
//      // DEBUGGING
//
//      auto cellIDs = basisCache->cellIDs();
//
//      if (cellIDs[0] == 0)
//      {
//        const int maxMomentNumber = _f_solnMoments->maxMomentNumber();
//
//        const int numConfigurationPoints = g0View.extent_int(1); // (C,P) shape
//
//        const int numPoints_vx = points_vx.extent_int(1); // Each PhysicalCubaturePoint tensorial component has dimensions (C,P,D)
//        const int numPoints_vy = points_vy.extent_int(1); // Each PhysicalCubaturePoint tensorial component has dimensions (C,P,D)
//        const int numPoints_vz = points_vz.extent_int(1); // Each PhysicalCubaturePoint tensorial component has dimensions (C,P,D)
//
//        const double g0_scaling = 1.0 / _n0;
//        const double u_scaling  = 1.0 / _u0;
//        {
//          const double g0    = g0View(0,1)    * g0_scaling;
//          const double ux    = uxView(0,1)    * u_scaling;
//          const double uy    = uyView(0,1)    * u_scaling;
//          const double uz    = uzView(0,1)    * u_scaling;
//          const double sigma = sigmaView(0,1) * u_scaling;
//
//          std::cout << "Camellia g0 (converted from MLC) at (0,1): " << g0 << std::endl;
//          std::cout << "Camellia ux (converted from MLC) at (0,1): " << ux << std::endl;
//          std::cout << "Camellia uy (converted from MLC) at (0,1): " << uy << std::endl;
//          std::cout << "Camellia uz (converted from MLC) at (0,1): " << uz << std::endl;
//          std::cout << "Camellia sigma squared (converted from MLC) at (0,1): " << sigma * sigma << std::endl;
//        }
//      }
//    }
    
    contributeDifferentialResponseFromBGK(valuesToAssign, g0View, uxView, uyView, uzView, sigmaView, GView, points_vx, points_vy, points_vz);
  }
  
//  {
//    // DEBUGGING
//    int numPhasePoints = valuesToAssign.extent_int(1);
//    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
//    {
//      for (int pointOrdinal=0; pointOrdinal<numPhasePoints; pointOrdinal++)
//      {
//        std::cout << "total collisional response value(" << cellOrdinal << ", " << pointOrdinal << ") = " << valuesToAssign(cellOrdinal,pointOrdinal) << std::endl;
//      }
//    }
//  }
  
  if (modType != ASSIGN)
  {
    assignValues<double>(values, modType, valuesToAssign);
    // we checked out scratch storage for valuesToAssign; check it back in now.
    basisCache->checkinScratchStorage<double>(valuesToAssign);
  }
  
  basisCache->checkinScratchStorage<double>(fValues);
  
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(valuesAllMoments);
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(         tauView);
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(          g0View);
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(          uxView);
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(          uyView);
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(          uzView);
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(       sigmaView);
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(           GView);
}
#else
void MLCollisionFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  // this empty implementation is just to allow compilation when Camellia is built without mlc; it will never get invoked,
  // because MLCollisionFunction constructor throws an exception if construction is attempted without mlc present…
}
int MLCollisionFunction::maxMomentNumber(std::string &modelDirPath)
{
  // dummy implementation
  return -1;
}
#endif
