project(SpaceTimeMixedVlasovDriver)

MESSAGE("Adding executable for SpaceTimeMixedVlasovDriver.")
add_executable(SpaceTimeMixedVlasovDriver "SpaceTimeMixedVlasovDriver.cpp")
target_link_libraries(SpaceTimeMixedVlasovDriver 
  Camellia
)

MESSAGE("Adding executable for SpaceTimeMixedConformingVlasovDriver.")
add_executable(SpaceTimeMixedConformingVlasovDriver "SpaceTimeMixedConformingVlasovDriver.cpp")
target_link_libraries(SpaceTimeMixedConformingVlasovDriver 
  Camellia
)

#install (TARGETS SpaceTimeMixedVlasovDriver DESTINATION bin/examples/)

