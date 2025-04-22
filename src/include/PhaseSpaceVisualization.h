//
//  PhaseSpaceVisualization.h
//  Camellia
//
//  Created by Roberts, Nathan V on 4/20/21.
//

#ifndef PhaseSpaceVisualization_h
#define PhaseSpaceVisualization_h

#include "HDF5Exporter.h"
#include "SolutionMoments.h"
#include "SolutionMomentFunction.h"

namespace Camellia {
  class PhaseSpaceVisualization
  {
    std::vector<Teuchos::RCP<SolutionMoments> > _solutionMoments; //
    std::vector<FunctionPtr > _solutionMomentFunctions; // used for extruded meshes beyond 3D
    std::vector<Teuchos::RCP<HDF5Exporter> > _exporters;
    std::vector<std::string> _names;
    
    int _numExtrusionDims;
  public:
    PhaseSpaceVisualization(SolutionPtr phaseSpaceSoln, VarPtr fieldVar, std::string outputDirName = "phaseSpaceOutput", std::string outputDirSuperPath = ".")
    {
      // we support this only for ExtrudedMeshTopology
      auto phaseSpaceMesh = phaseSpaceSoln->mesh();
      ExtrudedMeshTopology* extrudedMeshTopo = dynamic_cast<ExtrudedMeshTopology*>(phaseSpaceMesh->getTopology().get());
      if (extrudedMeshTopo == NULL)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Only ExtrudedMeshTopology meshes are supported by PhaseSpaceVisualization.");
      }
      const int physicalDim = extrudedMeshTopo->physicalMeshTopology()->getDimension();
      _numExtrusionDims = extrudedMeshTopo->numExtrusionDimensions();
      
      TEUCHOS_TEST_FOR_EXCEPTION(physicalDim != 1, std::invalid_argument, "physicalDim > 1 not yet supported");
      
      Teuchos::RCP<ExtrudedMeshTopology> extrudedMeshTopoRCP = Teuchos::rcp(extrudedMeshTopo,false); // weak pointer
      
      std::vector<std::string> v_comp_str {"vx","vy","vz"};
      
      int numVelocityDims = phaseSpaceMesh->getDimension() - physicalDim;
      
      if (numVelocityDims > 1)
      {
        for (int d=0; d<_numExtrusionDims; d++)
        {
          MeshTopologyViewPtr extrudedPhysicalMeshTopology;
          extrudedPhysicalMeshTopology = Teuchos::rcp( new SubExtrudedMeshTopology(extrudedMeshTopoRCP, d) );
          
          const bool allowPruning = false;
          const int delta_k = 0;
          const int physicalH1Order = phaseSpaceMesh->globalDofAssignment()->getInitialH1Order()[0];
          MeshPtr extrudedPhysicalMesh = MeshFactory::minRuleMesh(extrudedPhysicalMeshTopology, phaseSpaceMesh->bilinearForm(), physicalH1Order, delta_k, MPIWrapper::EpetraComm(phaseSpaceMesh->Comm()), allowPruning);
          const int momentNumber = 0; // only zeroth moment used/supported thus far
          
          auto solutionMoments = Teuchos::rcp( new SolutionMoments(momentNumber, phaseSpaceSoln, fieldVar, extrudedPhysicalMesh, d) );
          
          _solutionMoments.push_back(solutionMoments);
          
          int numVelocityDimsToIntegrate = numVelocityDims - 1; // the "d" dimension remains unintegrated…
          
          FunctionPtr solnMomentFxn;
          if (numVelocityDimsToIntegrate == 1)
          {
            solnMomentFxn = solutionMoments->solutionMomentFunction(0);
          }
          else if (numVelocityDimsToIntegrate == 2)
          {
            solnMomentFxn = solutionMoments->solutionMomentFunction(0,0);
          }
          _solutionMomentFunctions.push_back(solnMomentFxn);
          
          std::string suffix = std::string("_x") + v_comp_str[d];
          
          auto hdf5Exporter = Teuchos::rcp( new HDF5Exporter(extrudedPhysicalMesh, outputDirName + suffix, outputDirSuperPath));
          _exporters.push_back(hdf5Exporter);
          
          _names.push_back( fieldVar->name() + suffix);
        }
      }
    }
    
    void exportTimeStep(double timeVal)
    {
      for (int d=0; d<_numExtrusionDims; d++)
      {
        _solutionMoments[d]->update();
        _exporters[d]->exportFunction(_solutionMomentFunctions[d], _names[d], timeVal);
      }
      // DEBUGGING
//      _exporters[0]->exportFunction(Function::constant(1.0), "ONE", timeVal);
    }

  };
}

#endif /* PhaseSpaceVisualization_h */
