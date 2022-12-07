project(VlasovPoissonSpaceTime)

MESSAGE("Adding executable for LandauSpaceTime.")
add_executable( LandauSpaceTime "LandauSpaceTime.cpp" )

target_link_libraries( LandauSpaceTime Camellia )
