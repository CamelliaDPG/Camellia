project(StokesDrivers)

set( LID_REF_STRATEGY "LidDrivenFlowRefinementStrategy.cpp" )
set( STOKES_CAVITY_FLOW_DRIVER "StokesCavityFlowDriver.cpp" )

add_executable(StokesCavityFlowDriver ${STOKES_CAVITY_FLOW_DRIVER} ${LID_REF_STRATEGY})
target_link_libraries(StokesCavityFlowDriver 
  Camellia
)
