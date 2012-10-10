project(Euler)

add_executable(EulerRamp EulerRamp.cpp)
target_link_libraries(EulerRamp 
  Camellia
)

add_executable(SteadyEulerRamp SteadyEulerRamp.cpp)
target_link_libraries(SteadyEulerRamp 
  Camellia
)

add_executable(DiamondAirfoil DiamondAirfoil.cpp)
target_link_libraries(DiamondAirfoil 
  Camellia
)

add_executable(ForwardFacingStep ForwardFacingStep.cpp)
target_link_libraries(ForwardFacingStep 
  Camellia
)
