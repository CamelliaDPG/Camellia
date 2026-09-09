//
//  ConfigurationSpaceFormulationTrivial.h
//  Camellia
//
//  Created by Roberts, Nathan V on 1/11/22.
//

#ifndef ConfigurationSpaceFormulationTrivial_h
#define ConfigurationSpaceFormulationTrivial_h

#include "Camellia.h"
#include "ConfigurationSpaceFormulation.h"
#include "TypeDefs.h"

namespace Camellia
{
  /**
    \brief Class for specifying a trivial (i.e., zero) configuration-space solution.
   */
  class ConfigurationSpaceFormulationTrivial
:
  public ConfigurationSpaceFormulation
  {
    std::vector<FunctionPtr> _E, _B;
    std::vector<FunctionPtr> _E_priorTime, _B_priorTime;
    
    TimeDiscretizationType _timeDiscretizationType;
  public:
    ConfigurationSpaceFormulationTrivial(TimeDiscretizationType timeDiscretizationType);
    
    void initializeMesh(MeshTopologyPtr meshTopo, const int H1Order, const int delta_k) override;
        
    void setE(std::vector<FunctionPtr> E);
    
    void setB(std::vector<FunctionPtr> B);
    
    void setE_priorTime(std::vector<FunctionPtr> E);
    
    void setB_priorTime(std::vector<FunctionPtr> B);
    
    //! Returns a name distinguishing this configuration-space formulation from others.  "Maxwell", "Poisson", "Gauss" are all possibilities.
    virtual std::string name() const override;
    virtual Teuchos::RCP<ConfigurationSpaceFormulation> newInstance() const override;
        
    //! returns the specified component of the magnetic field, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for B_x, 1 for B_y, 2 for B_z.
    virtual FunctionPtr B_i(int zeroBasedComp) override;
    
    //! returns the specified component of the magnetic field, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for E_x, 1 for E_y, 2 for E_z.
    virtual FunctionPtr E_i(int zeroBasedComp) override;
    
    //! returns the specified component of the magnetic field at prior time, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for B_x, 1 for B_y, 2 for B_z.
    virtual FunctionPtr B_i_priorTime(int zeroBasedComp) override;
    
    //! returns the specified component of the electric field at prior timeas a Function, extruded if necessary to the phase mesh.  Argument is zero-based: 0 for E_x, 1 for E_y, 2 for E_z.
    virtual FunctionPtr E_i_priorTime(int zeroBasedComp) override;
    
    //! maximum moment number (starting from 0 for the unweighted integral over velocity space) required by the ConfigurationSpaceFormulation.
    virtual int maxMomentNumber() override;
    
    //! Perform MPI communication, etc. to ensure that the B_i, E_i function values for a given configuration-space cell are available on each MPI rank that has a corresponding phase-space cell.
    virtual void updateConfigurationSpaceFunctions() override;
  };
}

#endif /* ConfigurationSpaceFormulationTrivial_h */
