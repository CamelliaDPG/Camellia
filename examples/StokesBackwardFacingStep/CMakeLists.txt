project(StokesBackwardFacingStepExamples)

set( STOKES_BACKWARD_FACING_STEP_EXAMPLE "StokesBackwardFacingStepExample.cpp" )

add_executable(StokesBackwardFacingStepExample ${STOKES_BACKWARD_FACING_STEP_EXAMPLE})
target_link_libraries(StokesBackwardFacingStepExample 
  Camellia
)

