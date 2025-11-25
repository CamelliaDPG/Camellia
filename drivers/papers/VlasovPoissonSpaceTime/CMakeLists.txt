project(VlasovPoissonSpaceTime)

MESSAGE("Adding executable for LandauSpaceTime.")
add_executable( LandauSpaceTime "LandauSpaceTime.cpp" )
target_link_libraries( LandauSpaceTime Camellia )

MESSAGE("Adding executable for ColdDiodeSpaceTime.")
add_executable( ColdDiodeSpaceTime "ColdDiodeSpaceTime.cpp")
target_link_libraries(ColdDiodeSpaceTime Camellia)
