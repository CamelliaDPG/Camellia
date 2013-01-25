project(IncompressibleNSDrivers)

set( LID_REF_STRATEGY "../Stokes/LidDrivenFlowRefinementStrategy.cpp" )
set( NAVIER_STOKES_CAVITY_FLOW_DRIVER "NavierStokesCavityFlowDriver.cpp" ) 
set( NAVIER_STOKES_HEMKER_DRIVER "NavierStokesHemkerDriver.cpp" ) 

add_executable(NavierStokesCavityFlowDriver ${NAVIER_STOKES_CAVITY_FLOW_DRIVER} ${LID_REF_STRATEGY}) 
add_executable(NavierStokesHemkerDriver ${NAVIER_STOKES_HEMKER_DRIVER} ${LID_REF_STRATEGY}) 

target_link_libraries(NavierStokesHemkerDriver 
  Camellia
)
target_link_libraries(NavierStokesCavityFlowDriver 
  Camellia
)
