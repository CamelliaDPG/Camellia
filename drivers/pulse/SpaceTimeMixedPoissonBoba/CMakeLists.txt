project(SpaceTimeMixedPoissonDriverBoba)

MESSAGE("Adding executable for SpaceTimeMixedPoissonBobaDriver.")
add_executable(SpaceTimeMixedPoissonBobaDriver "SpaceTimeMixedPoissonBobaDriver.cpp")
target_link_libraries(SpaceTimeMixedPoissonBobaDriver 
  Camellia
)

#install (TARGETS SpaceTimeMixedPoissonDriver DESTINATION bin/examples/)

