project(SpaceTimeMixedVlasovBobaDriver)

MESSAGE("Adding executable for SpaceTimeMixedVlasovBobaDriver.")
add_executable(SpaceTimeMixedVlasovBobaDriver "SpaceTimeMixedVlasovBobaDriver.cpp")
target_link_libraries(SpaceTimeMixedVlasovBobaDriver 
  Camellia
)
