project(SpaceTimeMixedPoissonDriver)

MESSAGE("Adding executable for SpaceTimeMixedPoissonDriver.")
add_executable(SpaceTimeMixedPoissonDriver "SpaceTimeMixedPoissonDriver.cpp")
target_link_libraries(SpaceTimeMixedPoissonDriver 
  Camellia
)

#install (TARGETS SpaceTimeMixedPoissonDriver DESTINATION bin/examples/)

