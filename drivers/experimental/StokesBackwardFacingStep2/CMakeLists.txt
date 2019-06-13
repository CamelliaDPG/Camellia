project(StokesBackwardFacingStep2)

set( STOKES_BACKWARD_FACING_STEP_2 "StokesBackwardFacingStepExample2.cpp" )

add_executable(StokesBackwardFacingStepExample2 ${STOKES_BACKWARD_FACING_STEP_2})
target_link_libraries(StokesBackwardFacingStepExample2
  Camellia
)
