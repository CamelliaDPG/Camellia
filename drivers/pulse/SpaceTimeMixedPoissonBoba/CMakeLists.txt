project(SpaceTimeMixedPoissonDriverBoba)

MESSAGE("Adding executable for SpaceTimeMixedPoissonBobaDriver.")
add_executable(SpaceTimeMixedPoissonBobaDriver "SpaceTimeMixedPoissonBobaDriver.cpp")
set_target_properties(SpaceTimeMixedPoissonBobaDriver PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)
target_link_libraries(SpaceTimeMixedPoissonBobaDriver 
  Camellia
)

#install (TARGETS SpaceTimeMixedPoissonDriver DESTINATION bin/examples/)
