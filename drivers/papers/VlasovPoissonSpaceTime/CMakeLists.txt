project(VlasovPoissonSpaceTime)

MESSAGE("Adding executable for LandauSpaceTime.")
add_executable( LandauSpaceTime "LandauSpaceTime.cpp" )

target_link_libraries( LandauSpaceTime Camellia )

MESSAGE("Adding executable for SpaceTimeVlasovPoissonDriver.")
add_executable( SpaceTimeVlasovPoissonDriver "SpaceTimeVlasovPoissonDriver.cpp")

target_link_libraries(SpaceTimeVlasovPoissonDriver 
  Camellia
)