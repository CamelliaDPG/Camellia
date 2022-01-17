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

MLCollisionFunction::MLCollisionFunction(FunctionPtr f_soln, Teuchos::RCP<SolutionMoments> f_solnMoments, std::string &modelDirPath)
:
TFunction<double>("MLCollisionFunction", 0)
{
#ifndef HAVE_MLC
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Camellia was not configured with mlc_ROOT set; MLCollisionFunction requires mlc.");
#endif
  
  _f_soln         = f_soln;
  _f_solnMoments  = f_solnMoments;
  _modelDirectory = modelDirPath;
}

#ifdef HAVE_MLC
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
  
  auto points = basisCache->getPhysicalCubaturePoints();
  
  // we require that points have 3 velocity space dimensions separately stored
  // points must therefore have at least 4 tensor components
  TEUCHOS_TEST_FOR_EXCEPTION(points.numTensorComponents() < 4, std::invalid_argument, "basisCache's points container must contain at least 4 tensorial components");
  
  using PointsView = Intrepid2::ScalarView<double,Camellia::DeviceType>;
  
  using TensorPoints = Intrepid2::TensorPoints<double,DeviceType>;
  
  int numConfigurationTensorComponents = points.numTensorComponents() - 3;
  std::vector<PointsView> configurationSpaceRefPointsVector(numConfigurationTensorComponents);
  
  TensorPoints refPoints = basisCache->getRefCellPoints();
  
  for (int d=0; d<numConfigurationTensorComponents; d++)
  {
    configurationSpaceRefPointsVector[d] = refPoints.getTensorComponent(d);
  }
  TensorPoints configurationSpaceRefPoints(configurationSpaceRefPointsVector);
  
  auto points_vx = points.getTensorComponent(numConfigurationTensorComponents);
  auto points_vy = points.getTensorComponent(numConfigurationTensorComponents+1);
  auto points_vz = points.getTensorComponent(numConfigurationTensorComponents+2);
  
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
  
  typename FieldContainer<double>::DeviceView valuesToAssign;
  
  if (modType == ASSIGN)
  {
    valuesToAssign = values;
  }
  else
  {
    valuesToAssign = basisCache->checkoutScratchStorage<double>(values);
  }
  
  int numMomentsEntries = _f_solnMoments->numMomentEntries();
  
  auto valuesAllMoments = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints,numMomentsEntries},"MLCollisionFunction::values(): valuesAllMoments");
  
  _f_solnMoments->valuesForAllMoments(valuesAllMoments, configurationBasisCache);
  auto valuesAllMomentsHost = Kokkos::create_mirror_view_and_copy(HostMemorySpace(), valuesAllMoments);
  
  // declare/allocate the Kokkos views which we'll populate with mlc outputs
  auto    tauView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints},                  "MLCollisionFunction::values(): tauView");
  auto     g0View = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints},                  "MLCollisionFunction::values(): g0View");
  auto     uxView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints},                  "MLCollisionFunction::values(): uxView");
  auto     uyView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints},                  "MLCollisionFunction::values(): uyView");
  auto     uzView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints},                  "MLCollisionFunction::values(): uzView");
  auto  sigmaView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints},                  "MLCollisionFunction::values(): sigmaView");
  auto      GView = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorage({numCells,numConfigurationPoints,numMomentsEntries},"MLCollisionFunction::values():   GView");
  
  // TODO: compute tau pointwise here, using model, and revise the BGK computation below to use those tau values…
  
  // create and evaluate mlc model
  {
    using namespace mlc;
    
    // Create the mlc model:
    const unsigned numCellPoints = numConfigurationPoints * numCells; // mlc expects a single points dimension (no cell dimension), so we need to flatten…
    
    if ((numCellPoints != _numPointsForLoadedModel) || _loadedModel == Teuchos::null)
    {
      _loadedModel = Teuchos::rcp(new CollisionModel(numCellPoints, _modelDirectory));
      _numPointsForLoadedModel = numCellPoints;
      
      std::cout << "boltzmann constant = " << _loadedModel->getBoltzmannConstant() << std::endl;
      std::cout << "mass = " << _loadedModel->getMass() << std::endl;
    }
    
    const auto & inputMomentIndexer  = _loadedModel->getInputMomentIndexer();
    
    const unsigned maxMomentNumber = inputMomentIndexer.getNumMoments()-1;
    
    // We require the max moment numbers to match.  We could relax this, requiring only that SolutionMoments have at least as many moments as the model, with some modest revision to the below
    // logic, and perhaps the introduction of a MomentIndexer-like facility for SolutionMoments.
    TEUCHOS_TEST_FOR_EXCEPTION(maxMomentNumber != _f_solnMoments->maxMomentNumber(), std::invalid_argument, "max moment number for model must match max moment number in SolutionMoments");
    
    MomentIndexer standardMomentIndexer; // as opposed to the ones used by the model, which eliminates some moments
    standardMomentIndexer.setupForMoments(maxMomentNumber+1);
    const unsigned numTotalMoments = standardMomentIndexer.getNumCoefficients();
    
    const unsigned int numInputMomentsForModel  = inputMomentIndexer.getNumCoefficients();
    
    Array ml_input_conservedMoments({numCellPoints,numTotalMoments});
    
    // prepare model inputs
    // The model defines these coefficients (important since the model internally converts things to number density and temperature)
    const double kB = _loadedModel->getBoltzmannConstant();
    const double mass = _loadedModel->getMass();
    
    std::vector<int> mlcMomentIndexToCamelliaMomentIndex(numTotalMoments);
    for (int a=0; a<=maxMomentNumber; a++)
    {
      for (int b=0; b<=maxMomentNumber-a; b++)
      {
        for (int c=0; c<=maxMomentNumber-a-b; c++)
        {
          int i = standardMomentIndexer(a,b,c);
          int camelliaIndex = MomentIndices::getMomentIndex(a, b, c, maxMomentNumber+1);
          mlcMomentIndexToCamelliaMomentIndex[i] = camelliaIndex;
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
          ml_input_conservedMoments[index1D] = valuesAllMomentsHost(cellOrdinal,pointOrdinal,i); // this is where we rely on moment indexing being identical between mlc and SolutionMoments
        }
      }
    }

    Array ml_output_tau({numCellPoints});
    Array ml_output_g0({numCellPoints});
    Array ml_output_ux({numCellPoints});
    Array ml_output_uy({numCellPoints});
    Array ml_output_uz({numCellPoints});
    Array ml_output_sigma({numCellPoints});
    Array ml_output_G({numCellPoints,numTotalMoments});
    
    const bool evaluatePrimitive = true; // TODO: find out from Sean whether we should prefer primitive versus Hermite.  Using primitive for now because it's a bit easier to implement (since I don't have to compute the Hermite values).
    _loadedModel->generateCollisionalResponse(ml_input_conservedMoments,
                                              ml_output_tau, ml_output_g0, ml_output_ux, ml_output_uy, ml_output_uz, ml_output_sigma, ml_output_G,
                                              evaluatePrimitive);
    
    // the block below probably recreates code internal to generateCollisionalResponse(); leaving it here in case I'm wrong about that.
    // (i.e., based on MLCDriver, I thought I needed a separate call to evaluateModel(), but now I think generateCollisionalResponse() is all I need.)
//    {
//      // convert between indices for (restricted) ML moments and standard moment numbering
//      std::vector<unsigned int> ml_to_moment_index(numInputMomentsForModel);
//      for(unsigned int i=0; i<maxMomentNumber; ++i)
//        for(unsigned int j=0; j<maxMomentNumber-i; ++j)
//          for(unsigned int k=0; k<maxMomentNumber-i-j; ++k){
//            int idx = inputMomentIndexer(i,j,k);
//            if(idx >= 0)
//              ml_to_moment_index[idx] = standardMomentIndexer(i,j,k);
//          }
//      MLArray ml_input_F({numCellPoints,numInputMomentsForModel});
//      MLArray ml_input_n({numCellPoints});
//      MLArray ml_input_T({numCellPoints});
//      Array conservedMoments({numTotalMoments});
//      Array normalizedPrimitiveMoments({numTotalMoments});
//      Array hermiteCoefficients({numTotalMoments});
//      int cellPointOrdinal = 0;
//      for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
//      {
//        for (int pointOrdinal=0; pointOrdinal<numConfigurationPoints; pointOrdinal++, cellPointOrdinal++)
//        {
//          const double n  = valuesAllMomentsHost(cellOrdinal,pointOrdinal,standardMomentIndexer(0,0,0));
//          const double ux = valuesAllMomentsHost(cellOrdinal,pointOrdinal,standardMomentIndexer(1,0,0)) / n;
//          const double uy = valuesAllMomentsHost(cellOrdinal,pointOrdinal,standardMomentIndexer(0,1,0)) / n;
//          const double uz = valuesAllMomentsHost(cellOrdinal,pointOrdinal,standardMomentIndexer(0,0,1)) / n;
//
//          const double secondMomentSum = valuesAllMomentsHost(cellOrdinal,pointOrdinal,standardMomentIndexer(2,0,0))
//                                       + valuesAllMomentsHost(cellOrdinal,pointOrdinal,standardMomentIndexer(0,2,0))
//                                       + valuesAllMomentsHost(cellOrdinal,pointOrdinal,standardMomentIndexer(0,0,2));
//
//          const double sigmaSquared = (1./3.) * (secondMomentSum / n - ux * ux - uy * uy - uz * uz);
//
//          // P = sigma^2 * n * mass (this formula derived from MLCDriver code, along with the formula for sigmaSquared that we use in BGK)
//          const double P = sigmaSquared * n * mass;
//
//          // P = kB * n * T => T = P / (kB * n)  (formula taken from MLCDriver code)
//          const double T = P / (kB * n);
//
//          for (int i=0; i<numTotalMoments; i++)
//          {
//            conservedMoments[i] = valuesAllMomentsHost(cellOrdinal,pointOrdinal,i);
//          }
//
//          convertConservedMomentsToNormalizedPrimitiveMoments(mass, n, ux, uy, uz, P,
//                                                              standardMomentIndexer,
//                                                              conservedMoments.data(),
//                                                              normalizedPrimitiveMoments.data());
//
//          convertNormalizedPrimitiveMomentsToHermiteCoefficients(mass, n, P, standardMomentIndexer, normalizedPrimitiveMoments.data(), hermiteCoefficients.data());
//
//          for (int i=0; i<numInputMomentsForModel; i++)
//          {
//            const int hermiteIndex = cellPointOrdinal*numInputMomentsForModel + i;
//
//            ml_input_F[hermiteIndex]     = hermiteCoefficients[ml_to_moment_index[i]];
//            ml_input_T[cellPointOrdinal] = T;
//            ml_input_n[cellPointOrdinal] = n;
//          }
//        }
//      }
//
//      const auto & outputMomentIndexer = _loadedModel->getOutputMomentIndexer();
//      const unsigned int numOutputMomentsForModel = outputMomentIndexer.getNumCoefficients();
//      MLArray ml_output_G({numCellPoints,numOutputMomentsForModel});
//      MLArray ml_output_tau({numCellPoints});
//
//
//      _loadedModel->evaluateModel(ml_input_n, ml_input_T, ml_input_F, ml_output_tau, ml_output_G);
//    }
//  }
  
    // get the mlc outputs into Kokkos Views
    auto    tauHost = Kokkos::create_mirror_view(HostMemorySpace(),   tauView);
    auto     g0Host = Kokkos::create_mirror_view(HostMemorySpace(),    g0View);
    auto     uxHost = Kokkos::create_mirror_view(HostMemorySpace(),    uxView);
    auto     uyHost = Kokkos::create_mirror_view(HostMemorySpace(),    uyView);
    auto     uzHost = Kokkos::create_mirror_view(HostMemorySpace(),    uzView);
    auto  sigmaHost = Kokkos::create_mirror_view(HostMemorySpace(), sigmaView);
    auto      GHost = Kokkos::create_mirror_view(HostMemorySpace(),     GView);
    
    cellPointOrdinal = 0;
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
        
        {
          // DEBUGGING
          std::cout << "  tau(" << cellOrdinal << "," << pointOrdinal << ") = " <<   tauHost(cellOrdinal,pointOrdinal) << std::endl;
          std::cout << "   g0(" << cellOrdinal << "," << pointOrdinal << ") = " <<    g0Host(cellOrdinal,pointOrdinal) << std::endl;
          std::cout << "   ux(" << cellOrdinal << "," << pointOrdinal << ") = " <<    uxHost(cellOrdinal,pointOrdinal) << std::endl;
          std::cout << "   uy(" << cellOrdinal << "," << pointOrdinal << ") = " <<    uyHost(cellOrdinal,pointOrdinal) << std::endl;
          std::cout << "   uz(" << cellOrdinal << "," << pointOrdinal << ") = " <<    uzHost(cellOrdinal,pointOrdinal) << std::endl;
          std::cout << "sigma(" << cellOrdinal << "," << pointOrdinal << ") = " << sigmaHost(cellOrdinal,pointOrdinal) << std::endl;
        }
        for (int i=0; i<numTotalMoments; i++)
        {
          int index1D = cellPointOrdinal*numTotalMoments + i;
          GHost(cellOrdinal,pointOrdinal,i) = ml_output_G[index1D];
        }
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
  
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  auto rangePolicy = FieldContainer<double>::getMDRangePolicy2(tauView);
  Kokkos::fence();
  const int numVelocityDims = 3;
  const int numSpatialDims  = configurationSpaceRefPoints.extent_int(1); // (P,D)
  mlc::MomentIndexer standardMomentIndexer; // as opposed to the ones used by the model, which eliminates some moments
  standardMomentIndexer.setupForMoments(_f_solnMoments->maxMomentNumber());
  const int zerothMomentIndex = standardMomentIndexer(0,0,0);
  Kokkos::Array<int,3> firstMomentIndices {standardMomentIndexer(1,0,0),standardMomentIndexer(0,1,0),standardMomentIndexer(0,0,1)};
  Kokkos::Array<int,3> secondMomentIndices {standardMomentIndexer(2,0,0),standardMomentIndexer(0,2,0),standardMomentIndexer(0,0,2)};
  const int numPoints_vx = points_vx.extent_int(0);
  const int numPoints_vy = points_vy.extent_int(0);
  const int numPoints_vz = points_vz.extent_int(0);
  const int maxMomentNumber = _f_solnMoments->maxMomentNumber();
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
    const double weight   = - n / pow(2 * M_PI * sigmaSquared, 1.5);
    
    const double &tau = tauView(cellOrdinal,configurationPointOrdinal);
    
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
      v[0] = points_vx(ivx,0);
      pointComponentOrdinals[1] = ivx;
      for (int ivy=0; ivy<numPoints_vy; ivy++)
      {
        v[1] = points_vy(ivy,0);
        pointComponentOrdinals[2] = ivy;
        for (int ivz=0; ivz<numPoints_vz; ivz++)
        {
          v[2] = points_vz(ivz,0);
          pointComponentOrdinals[3] = ivz;
          
          // set phasePointOrdinal:
          TensorIndices::setTensorOrdinal(phasePointOrdinal, pointComponentOrdinals, pointComponentCardinalities, numPointComponents);
          
          double v_minus_u_squared = (v[0]-u[0])*(v[0]-u[0]) + (v[1]-u[1])*(v[1]-u[1]) + (v[2]-u[2])*(v[2]-u[2]);
          const double exponent = - 1. / (2. * sigmaSquared) * v_minus_u_squared;
          
          const double f_M = weight * std::exp(exponent);
          const double f = fValues(cellOrdinal,phasePointOrdinal);
          valuesToAssign(cellOrdinal,phasePointOrdinal) = (-1.0 / tau) * (f - f_M);
        }
      }
    }
  });
  
  // the below is adapted from evaluateMLDSMC_Primitive() in fvBoltzmann's SolverModule_MLDSMCCollisions.cpp.
  rangePolicy = FieldContainer<double>::getMDRangePolicy2(sigmaView);
  Kokkos::parallel_for("MLCollisionFunction: compute ML collisional response contribution", rangePolicy,
  KOKKOS_LAMBDA (const int cellOrdinal, const int configurationPointOrdinal) {
    const double g0    = g0View(cellOrdinal,configurationPointOrdinal);
    const double ux    = uxView(cellOrdinal,configurationPointOrdinal);
    const double uy    = uyView(cellOrdinal,configurationPointOrdinal);
    const double uz    = uzView(cellOrdinal,configurationPointOrdinal);
    const double sigma = sigmaView(cellOrdinal,configurationPointOrdinal);

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
      const double &vx = points_vx(ivx,0);
      const double wx = w_scale * (vx - ux);
      pointComponentOrdinals[1] = ivx;
      for (int ivy=0; ivy<numPoints_vy; ivy++)
      {
        const double &vy = points_vy(ivy,0);
        const double wy = w_scale * (vy - uy);
        pointComponentOrdinals[2] = ivy;
        for (int ivz=0; ivz<numPoints_vz; ivz++)
        {
          const double &vz = points_vz(ivz,0);
          const double wz = w_scale * (vz - uz);
          pointComponentOrdinals[3] = ivz;
          
          // set phasePointOrdinal:
          TensorIndices::setTensorOrdinal(phasePointOrdinal, pointComponentOrdinals, pointComponentCardinalities, numPointComponents);
          
          const double mult = g0 * std::exp(-(wx*wx+wy*wy+wz*wz));
          
          double rhsijkl = 0;
          double mult_wxa = mult;
          for(unsigned int a=0,abc=0; a<maxMomentNumber; ++a){
            double mult_wxa_wyb = mult_wxa;
            for(unsigned int b=0; b<maxMomentNumber-a; ++b){
              double mult_wxa_wyb_wzc = mult_wxa_wyb;
              for(unsigned int c=0; c<maxMomentNumber-a-b; ++c, ++abc){
                rhsijkl += GView(cellOrdinal,configurationPointOrdinal,abc) * mult_wxa_wyb_wzc;
                mult_wxa_wyb_wzc *= wz;
              }
              mult_wxa_wyb *= wy;
            }
            mult_wxa *= wx;
          }

          valuesToAssign(cellOrdinal,phasePointOrdinal) += rhsijkl;
        }
      }
    }
  });
  
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
