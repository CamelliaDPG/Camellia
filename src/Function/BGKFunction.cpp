//
//  BGKFunction.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 12/22/21.
//

#include "BGKFunction.h"

using namespace Camellia;

BGKFunction::BGKFunction(const double tau, FunctionPtr f_soln, Teuchos::RCP<SolutionMoments> f_solnMoments)
:
TFunction<double>("BGKFunction", 0)
{
  _tau           = tau;
  _f_soln        = f_soln;
  _f_solnMoments = f_solnMoments;
  
  const int numVelocityDims = f_solnMoments->numVelocityDims();
  if (numVelocityDims == 1)
  {
    _zeroMoment = f_solnMoments->solutionMomentFunction(0);
    _firstMoments.push_back(f_solnMoments->solutionMomentFunction(1));
    _secondMoment = f_solnMoments->solutionMomentFunction(2);
  }
  else if (numVelocityDims == 2)
  {
    _zeroMoment = f_solnMoments->solutionMomentFunction(0,0);
    _firstMoments.push_back(f_solnMoments->solutionMomentFunction(1,0));
    _firstMoments.push_back(f_solnMoments->solutionMomentFunction(0,1));
    _secondMoment = f_solnMoments->solutionMomentFunction(2,0) + f_solnMoments->solutionMomentFunction(0,2);
  }
  else if (numVelocityDims == 3)
  {
    _zeroMoment = f_solnMoments->solutionMomentFunction(0,0,0);
    _firstMoments.push_back(f_solnMoments->solutionMomentFunction(1,0,0));
    _firstMoments.push_back(f_solnMoments->solutionMomentFunction(0,1,0));
    _firstMoments.push_back(f_solnMoments->solutionMomentFunction(0,0,1));
    _secondMoment = f_solnMoments->solutionMomentFunction(2,0,0) + f_solnMoments->solutionMomentFunction(0,2,0) + f_solnMoments->solutionMomentFunction(0,0,2);
  }
}

void BGKFunction::importCellData(std::vector<GlobalIndexType> phaseSpaceCellIDs)
{
  if (_f_soln.get() != nullptr)
  {
    _f_soln->importCellData(phaseSpaceCellIDs);
  }
  
  // Do not return early when phaseSpaceCellIDs is empty.  The delegated
  // SolutionMomentFunction/SolutionMoments imports are collective MPI operations,
  // so ranks with no local phase-space cells must still participate.
  auto phaseSpaceMesh = _f_solnMoments->phaseSpaceMesh();
  auto phaseSpaceMeshTopology = dynamic_cast<const ExtrudedMeshTopology*>(phaseSpaceMesh->getTopology()->baseMeshTopology());
  TEUCHOS_TEST_FOR_EXCEPTION(phaseSpaceMeshTopology == nullptr, std::invalid_argument, "BGKFunction::importCellData() requires an ExtrudedMeshTopology phase-space mesh");
  
  std::set<GlobalIndexType> configurationSpaceCellIDSet;
  for (const auto &phaseSpaceCellID : phaseSpaceCellIDs)
  {
    configurationSpaceCellIDSet.insert(phaseSpaceMeshTopology->physicalCellIDForPhaseSpaceCellID(phaseSpaceCellID));
  }
  std::vector<GlobalIndexType> configurationSpaceCellIDs(configurationSpaceCellIDSet.begin(), configurationSpaceCellIDSet.end());
  
  // All BGK moment functions share the same SolutionMoments object, and
  // SolutionMoments::importCellData() imports coefficient data for all moment
  // entries.  Call it once here rather than redundantly through each
  // SolutionMomentFunction.
  _f_solnMoments->importCellData(configurationSpaceCellIDs);
}

void BGKFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("BGKFunction values()");
  
  using Data = Intrepid2::Data<double, Camellia::DeviceType>;
  
  auto points = basisCache->getPhysicalCubaturePoints();
  
  auto phaseSpaceMeshTopology = dynamic_cast<const ExtrudedMeshTopology*>(basisCache->mesh()->getTopology()->baseMeshTopology());
  
  const auto & phaseSpaceCellIDs = basisCache->cellIDs();
  std::vector<GlobalIndexType> configurationSpaceCellIDs(phaseSpaceCellIDs.size());
  
  for (int cellOrdinal=0; cellOrdinal<phaseSpaceCellIDs.size(); cellOrdinal++)
  {
    const GlobalIndexType &phaseSpaceCellID = phaseSpaceCellIDs[cellOrdinal];
    const GlobalIndexType configurationCellID = phaseSpaceMeshTopology->physicalCellIDForPhaseSpaceCellID(phaseSpaceCellID);
    configurationSpaceCellIDs[cellOrdinal] = configurationCellID;
  }
  
  auto configurationMesh = _f_solnMoments->mesh();
  auto configurationBasisCache = BasisCache::basisCacheForCellType(configurationMesh, configurationMesh->getElementType(configurationSpaceCellIDs[0]));
  
  using PointsView = Intrepid2::ScalarView<double,Camellia::DeviceType>;
  
  using TensorPoints = Intrepid2::TensorPoints<double,DeviceType>;
  
  const int numVelocityDims = _f_solnMoments->numVelocityDims();
  
  int numConfigurationTensorComponents = basisCache->getPhysicalCubaturePoints().numTensorComponents() - numVelocityDims;
  std::vector<PointsView> configurationSpaceRefPointsVector(numConfigurationTensorComponents);
  
  TensorPoints refPoints = basisCache->getRefCellPoints();
  
  std::vector<int> configTensorialComponentOrdinals(numConfigurationTensorComponents);
  for (int d=0; d<numConfigurationTensorComponents; d++)
  {
    configurationSpaceRefPointsVector[d] = refPoints.getTensorComponent(d);
  }
  TensorPoints configurationSpaceRefPoints(configurationSpaceRefPointsVector);
  
  auto physicalCellNodes = configurationMesh->physicalCellNodes(configurationSpaceCellIDs);
  configurationBasisCache->setRefCellPoints(configurationSpaceRefPoints);
  configurationBasisCache->setPhysicalCellNodes(physicalCellNodes, configurationSpaceCellIDs, false); // false: don't create side caches
  
  // checkout storage and store values:
  Data zeroMomentData = _zeroMoment->checkoutValuesData(configurationBasisCache);
  _zeroMoment->values(zeroMomentData, configurationBasisCache);
  Kokkos::Array<Data,3> firstMomentData;
  
  const int numPhaseDims = points.extent_int(2); // (C,P,D)
  const int numConfigurationDims = numPhaseDims - numVelocityDims;
  for (int i=0; i<numVelocityDims; i++)
  {
    firstMomentData[i] = _firstMoments[i]->checkoutValuesData(configurationBasisCache);
    _firstMoments[i]->values(firstMomentData[i], configurationBasisCache);
  }
  Data secondMomentData = _secondMoment->checkoutValuesData(configurationBasisCache);
  _secondMoment->values(secondMomentData, configurationBasisCache);
  
  auto fValues = basisCache->checkoutScratchStorage<double>(values);

  _f_soln->values(fValues, ASSIGN, basisCache);
  
  typename FieldContainer<double>::DeviceView valuesToAssign;
  
  if (modType == ASSIGN)
  {
    valuesToAssign = values;
  }
  else
  {
    valuesToAssign = basisCache->checkoutScratchStorage<double>(values);
  }
    
  const auto tau = _tau; // avoid implicit reference to this in the KOKKOS_LAMBDA blocks below.
  
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  auto rangePolicy = FieldContainer<double>::getMDRangePolicy2(values);
  Kokkos::fence();
  const int momentDataRank = zeroMomentData.getUnderlyingViewRank();
  TEUCHOS_TEST_FOR_EXCEPTION((momentDataRank < 2) || (momentDataRank > 4), std::invalid_argument, "Unsupported BGK moment Data rank");
  TEUCHOS_TEST_FOR_EXCEPTION(secondMomentData.getUnderlyingViewRank() != momentDataRank, std::invalid_argument, "BGK moment Data ranks do not match");
  for (int i=0; i<numVelocityDims; i++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(firstMomentData[i].getUnderlyingViewRank() != momentDataRank, std::invalid_argument, "BGK moment Data ranks do not match");
  }
  
  if (momentDataRank == 2)
  {
    auto zeroMomentView   = zeroMomentData.getUnderlyingView<2>();
    auto secondMomentView = secondMomentData.getUnderlyingView<2>();
    Kokkos::Array<decltype(zeroMomentView),3> firstMomentViews;
    for (int i=0; i<numVelocityDims; i++)
    {
      firstMomentViews[i] = firstMomentData[i].getUnderlyingView<2>();
    }
    
    const int numPoint0 = zeroMomentView.extent_int(1);
    
    Kokkos::parallel_for("BGK_values", rangePolicy,
    KOKKOS_LAMBDA (const int i0, const int i1) {
      const int p0 = i1 % numPoint0;
      const double n = zeroMomentView(i0,p0);
      
      double sigmaSquared = (1.0 / n) * secondMomentView(i0,p0);
      double v_minus_u_squared = 0.0;
      for (int d=0; d<numVelocityDims; d++)
      {
        const double u_d = (1.0 / n) * firstMomentViews[d](i0,p0);
        sigmaSquared -= u_d * u_d;
        const double v_d = points(i0,i1,d+numConfigurationDims);
        v_minus_u_squared += (v_d - u_d) * (v_d - u_d);
      }
      
      sigmaSquared *= (1.0 / numVelocityDims);
      
      const double normalization = n / std::pow(2.0 * M_PI * sigmaSquared, 0.5 * numVelocityDims);
      const double exponent = - 1.0 / (2.0 * sigmaSquared) * v_minus_u_squared;
      const double f_M = normalization * std::exp(exponent);
      const double f = fValues(i0,i1);
      valuesToAssign(i0,i1) = (-1.0 / tau) * (f - f_M);
    });
  }
  else if (momentDataRank == 3)
  {
    auto zeroMomentView   = zeroMomentData.getUnderlyingView<3>();
    auto secondMomentView = secondMomentData.getUnderlyingView<3>();
    Kokkos::Array<decltype(zeroMomentView),3> firstMomentViews;
    for (int i=0; i<numVelocityDims; i++)
    {
      firstMomentViews[i] = firstMomentData[i].getUnderlyingView<3>();
    }
    
    const int numPoint0 = zeroMomentView.extent_int(1);
    const int numPoint1 = zeroMomentView.extent_int(2);
    
    Kokkos::parallel_for("BGK_values", rangePolicy,
    KOKKOS_LAMBDA (const int i0, const int i1) {
      const int p0 = i1 % numPoint0;
      const int p1 = (i1 / numPoint0) % numPoint1;
      const double n = zeroMomentView(i0,p0,p1);
      
      double sigmaSquared = (1.0 / n) * secondMomentView(i0,p0,p1);
      double v_minus_u_squared = 0.0;
      for (int d=0; d<numVelocityDims; d++)
      {
        const double u_d = (1.0 / n) * firstMomentViews[d](i0,p0,p1);
        sigmaSquared -= u_d * u_d;
        const double v_d = points(i0,i1,d+numConfigurationDims);
        v_minus_u_squared += (v_d - u_d) * (v_d - u_d);
      }
      
      sigmaSquared *= (1.0 / numVelocityDims);
      
      const double normalization = n / std::pow(2.0 * M_PI * sigmaSquared, 0.5 * numVelocityDims);
      const double exponent = - 1.0 / (2.0 * sigmaSquared) * v_minus_u_squared;
      const double f_M = normalization * std::exp(exponent);
      const double f = fValues(i0,i1);
      valuesToAssign(i0,i1) = (-1.0 / tau) * (f - f_M);
    });
  }
  else if (momentDataRank == 4)
  {
    auto zeroMomentView   = zeroMomentData.getUnderlyingView<4>();
    auto secondMomentView = secondMomentData.getUnderlyingView<4>();
    Kokkos::Array<decltype(zeroMomentView),3> firstMomentViews;
    for (int i=0; i<numVelocityDims; i++)
    {
      firstMomentViews[i] = firstMomentData[i].getUnderlyingView<4>();
    }
    
    const int numPoint0 = zeroMomentView.extent_int(1);
    const int numPoint1 = zeroMomentView.extent_int(2);
    const int numPoint2 = zeroMomentView.extent_int(3);
    
    Kokkos::parallel_for("BGK_values", rangePolicy,
    KOKKOS_LAMBDA (const int i0, const int i1) {
      const int p0 = i1 % numPoint0;
      const int p1 = (i1 / numPoint0) % numPoint1;
      const int p2 = (i1 / (numPoint0 * numPoint1)) % numPoint2;
      const double n = zeroMomentView(i0,p0,p1,p2);
      
      double sigmaSquared = (1.0 / n) * secondMomentView(i0,p0,p1,p2);
      double v_minus_u_squared = 0.0;
      for (int d=0; d<numVelocityDims; d++)
      {
        const double u_d = (1.0 / n) * firstMomentViews[d](i0,p0,p1,p2);
        sigmaSquared -= u_d * u_d;
        const double v_d = points(i0,i1,d+numConfigurationDims);
        v_minus_u_squared += (v_d - u_d) * (v_d - u_d);
      }
      
      sigmaSquared *= (1.0 / numVelocityDims);
      
      const double normalization = n / std::pow(2.0 * M_PI * sigmaSquared, 0.5 * numVelocityDims);
      const double exponent = - 1.0 / (2.0 * sigmaSquared) * v_minus_u_squared;
      const double f_M = normalization * std::exp(exponent);
      const double f = fValues(i0,i1);
      valuesToAssign(i0,i1) = (-1.0 / tau) * (f - f_M);
    });
  }
  
  if (modType != ASSIGN)
  {
    assignValues<double>(values, modType, valuesToAssign);
  }
    
  if (modType != ASSIGN)
  {
    // then we checked out scratch storage for valuesToAssign
    basisCache->checkinScratchStorage<double>(valuesToAssign);
  }
  
  basisCache->checkinScratchStorage<double>(fValues);
  
  _zeroMoment->checkinValuesData(zeroMomentData);
  for (int i=0; i<numVelocityDims; i++)
  {
    _firstMoments[i]->checkinValuesData(firstMomentData[i]);
  }
  _secondMoment->checkinValuesData(secondMomentData);
}
