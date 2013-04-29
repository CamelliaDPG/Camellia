project(IncompressibleNSDrivers)

set( LID_REF_STRATEGY "../Stokes/LidDrivenFlowRefinementStrategy.cpp" )
set( NAVIER_STOKES_CAVITY_FLOW_DRIVER "NavierStokesCavityFlowDriver.cpp" ) 
set( NAVIER_STOKES_HEMKER_DRIVER "NavierStokesHemkerDriver.cpp" ) 
set( NAVIER_STOKES_CAVITY_FLOW_CONTINUATION_ADAPTIVE "NavierStokesCavityFlowContinuationAdaptive.cpp" ) 
set( NAVIER_STOKES_CAVITY_FLOW_CONTINUATION_FIXED_MESH "NavierStokesCavityFlowContinuationFixedMesh.cpp" ) 
set( NAVIER_STOKES_STUDY "NavierStokesStudy.cpp" ) 
set( OSEEN_STUDY "OseenStudy.cpp" ) 

add_executable(NavierStokesCavityFlowContinuationAdaptive ${NAVIER_STOKES_CAVITY_FLOW_CONTINUATION_ADAPTIVE} ${LID_REF_STRATEGY}) 
add_executable(NavierStokesCavityFlowContinuationFixedMesh ${NAVIER_STOKES_CAVITY_FLOW_CONTINUATION_FIXED_MESH} ${LID_REF_STRATEGY}) 
add_executable(NavierStokesCavityFlowDriver ${NAVIER_STOKES_CAVITY_FLOW_DRIVER} ${LID_REF_STRATEGY}) 
add_executable(NavierStokesHemkerDriver ${NAVIER_STOKES_HEMKER_DRIVER} ${LID_REF_STRATEGY}) 
add_executable(NavierStokesStudy ${NAVIER_STOKES_STUDY}) 
add_executable(OseenStudyDriver ${OSEEN_STUDY}) 

target_link_libraries(NavierStokesHemkerDriver 
  Camellia
)
target_link_libraries(NavierStokesCavityFlowDriver 
  Camellia
)
target_link_libraries(NavierStokesCavityFlowContinuationAdaptive 
  Camellia
)
target_link_libraries(NavierStokesCavityFlowContinuationFixedMesh 
  Camellia
)
target_link_libraries(NavierStokesStudy
  Camellia
)
target_link_libraries(OseenStudy
  Camellia
)