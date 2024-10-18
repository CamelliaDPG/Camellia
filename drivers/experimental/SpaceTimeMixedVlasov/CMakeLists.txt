project(SpaceTimeMixedVlasovDriver)

MESSAGE("Adding executable for SpaceTimeMixedVlasovDriver.")
add_executable(SpaceTimeMixedVlasovDriver "SpaceTimeMixedVlasovDriver.cpp")
target_link_libraries(SpaceTimeMixedVlasovDriver 
  Camellia
)

#install (TARGETS SpaceTimeMixedVlasovDriver DESTINATION bin/examples/)

