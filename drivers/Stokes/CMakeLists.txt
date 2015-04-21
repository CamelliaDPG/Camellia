project(StokesDrivers)

set( BACK_STEP_REF_STRATEGY "BackwardFacingStepRefinementStrategy.cpp" )
set( LID_REF_STRATEGY "LidDrivenFlowRefinementStrategy.cpp" )
set( STOKES_CAVITY_FLOW_DRIVER "StokesCavityFlowDriver.cpp" )
set( STOKES_CAVITY_FLOW_DRIVER_GMG "StokesCavityFlowDriverGMGExperiment.cpp" )
set( STOKES_BACK_STEP_DRIVER "StokesBackwardFacingStepDriver.cpp" )

add_executable(StokesBackwardFacingStepDriver ${STOKES_BACK_STEP_DRIVER} ${BACK_STEP_REF_STRATEGY})
target_link_libraries(StokesBackwardFacingStepDriver 
  Camellia
)

add_executable(StokesCavityFlowDriver ${STOKES_CAVITY_FLOW_DRIVER} ${LID_REF_STRATEGY})
target_link_libraries(StokesCavityFlowDriver 
  Camellia
)

add_executable(StokesCavityFlowDriverForPaper "StokesCavityFlowDriverForPaper.cpp")
target_link_libraries(StokesCavityFlowDriver 
  Camellia
)

add_executable(StokesCavityFlowGMGDriver ${STOKES_CAVITY_FLOW_DRIVER_GMG})
target_link_libraries(StokesCavityFlowGMGDriver 
  Camellia
)

set( STOKES_CAVITY_FLOW_DRIVER_3D "StokesCavityFlowDriver3D.cpp" )
add_executable(StokesCavityFlowDriver3D ${STOKES_CAVITY_FLOW_DRIVER_3D})
target_link_libraries(StokesCavityFlowDriver3D 
  Camellia
)