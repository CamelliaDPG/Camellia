project(StokesDrivers)

set( LID_REF_STRATEGY "LidDrivenFlowRefinementStrategy.cpp" )
set( STOKES_CAVITY_FLOW_DRIVER "StokesCavityFlowDriver.cpp" )

add_executable(StokesCavityFlowDriver ${STOKES_CAVITY_FLOW_DRIVER} ${LID_REF_STRATEGY})
target_link_libraries(StokesCavityFlowDriver 
  Camellia
)

set( STOKES_CAVITY_FLOW_DRIVER_3D "StokesCavityFlowDriver3D.cpp" )
add_executable(StokesCavityFlowDriver3D ${STOKES_CAVITY_FLOW_DRIVER_3D})
target_link_libraries(StokesCavityFlowDriver3D 
  Camellia
)