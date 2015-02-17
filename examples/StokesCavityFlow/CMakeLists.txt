project(StokesCavityFlowExamples)

set( STOKES_CAVITY_FLOW_EXAMPLE "StokesCavityFlowExample.cpp" )

add_executable(StokesCavityFlowExample ${STOKES_CAVITY_FLOW_EXAMPLE})
target_link_libraries(StokesCavityFlowExample 
  Camellia
)

