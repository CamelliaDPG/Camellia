project(NavierStokesCavityFlowExamples)

set( NAVIER_STOKES_CAVITY_FLOW_EXAMPLE "NavierStokesCavityFlowExample.cpp" )

add_executable(NavierStokesCavityFlowExample ${NAVIER_STOKES_CAVITY_FLOW_EXAMPLE})
target_link_libraries(NavierStokesCavityFlowExample 
  Camellia
)

