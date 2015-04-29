project(HeatSpaceTimeExamples)

set( HEAT_SPACE_TIME_EXAMPLE "HeatSpaceTimeExample.cpp" )

add_executable(HeatSpaceTimeExample ${HEAT_SPACE_TIME_EXAMPLE})
target_link_libraries(
  HeatSpaceTimeExample 
  Camellia
)

